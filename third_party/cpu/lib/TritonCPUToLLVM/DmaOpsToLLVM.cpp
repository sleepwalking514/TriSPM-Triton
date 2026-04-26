#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DMAOPSTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

// ============================================================================
// DMA MMIO register layout (must match gem5 DmaEngine model)
//
//   Base address: 0xF000'0000  (configurable via pass option)
//
//   Offset  Register    Description
//   0x00    SRC         Source address (64-bit)
//   0x08    DST         Destination address (64-bit)
//   0x10    LEN         Transfer width in bytes — writing triggers enqueue
//   0x18    STATUS      Read → 0 = idle, non-zero = busy (used by dma_wait)
//   0x20    SRC_STRIDE  Source row stride in bytes (64-bit)
//   0x28    DST_STRIDE  Destination row stride in bytes (64-bit)
//   0x30    HEIGHT      Number of rows (64-bit)
// ============================================================================

static constexpr uint64_t DMA_MMIO_BASE = 0xF0000000ULL;
static constexpr uint64_t DMA_REG_SRC = 0x00;
static constexpr uint64_t DMA_REG_DST = 0x08;
static constexpr uint64_t DMA_REG_LEN = 0x10;
static constexpr uint64_t DMA_REG_STATUS = 0x18;
static constexpr uint64_t DMA_REG_SRC_STRIDE = 0x20;
static constexpr uint64_t DMA_REG_DST_STRIDE = 0x28;
static constexpr uint64_t DMA_REG_HEIGHT = 0x30;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addIllegalOp<triton::cpu::DmaEnqueue2DOp>();
    addIllegalOp<triton::cpu::DmaWaitOp>();
  }
};

// ---------------------------------------------------------------------------
// Helper: create an i64 constant
// ---------------------------------------------------------------------------
static Value createI64Constant(ConversionPatternRewriter &rewriter,
                               Location loc, uint64_t val) {
  auto i64Ty = rewriter.getI64Type();
  return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                  rewriter.getI64IntegerAttr(val));
}

// ---------------------------------------------------------------------------
// Helper: emit a volatile store of `val` to MMIO address `base + offset`.
//
// We use inttoptr to form the MMIO address and mark the store volatile so
// that LLVM will not reorder or eliminate it.
// ---------------------------------------------------------------------------
static void emitVolatileStore(ConversionPatternRewriter &rewriter, Location loc,
                              Value val, uint64_t base, uint64_t offset) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value addr = createI64Constant(rewriter, loc, base + offset);
  Value ptr = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, addr);
  auto store = LLVM::StoreOp::create(rewriter, loc, val, ptr);
  store.setVolatile_(true);
}

// ---------------------------------------------------------------------------
// Helper: emit a volatile load of an i64 from MMIO address `base + offset`.
// ---------------------------------------------------------------------------
static Value emitVolatileLoad(ConversionPatternRewriter &rewriter, Location loc,
                              uint64_t base, uint64_t offset) {
  auto i64Ty = rewriter.getI64Type();
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value addr = createI64Constant(rewriter, loc, base + offset);
  Value ptr = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, addr);
  auto load = LLVM::LoadOp::create(rewriter, loc, i64Ty, ptr);
  load.setVolatile_(true);
  return load;
}

// ---------------------------------------------------------------------------
// Helper: emit a RISC-V `fence iorw, iorw` via inline assembly.
//
// We deliberately do NOT use `LLVM::FenceOp(seq_cst)` here.  The mapping
// `seq_cst -> fence iorw, iorw` was true on older LLVM versions, but modern
// LLVM (≥17) lowers `seq_cst` on RISC-V to the lighter `fence rw, rw`,
// which only orders ordinary memory accesses.  For MMIO peripherals such
// as the DMA engine we need to order I/O accesses too — otherwise an
// implementation that distinguishes I/O from memory (or a future LLVM
// change) could re-order register writes around the LEN trigger.
//
// Emitting the fence as raw inline assembly with `has_side_effects = true`
// pins the exact instruction we want and prevents the optimizer from
// touching it.
// ---------------------------------------------------------------------------
static void emitFence(ConversionPatternRewriter &rewriter, Location loc) {
  auto *ctx = rewriter.getContext();
  LLVM::InlineAsmOp::create(
      rewriter, loc,
      /*resultTypes=*/TypeRange(),
      /*operands=*/ValueRange(),
      /*asm_string=*/"fence iorw, iorw",
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false,
      /*tail_call_kind=*/LLVM::TailCallKind::None,
      /*asm_dialect=*/
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
      /*operand_attrs=*/ArrayAttr());
}

// ===----------------------------------------------------------------------===
// DmaEnqueue2DOp → volatile MMIO stores
// ===----------------------------------------------------------------------===
struct DmaEnqueue2DOpConversion
    : public OpConversionPattern<triton::cpu::DmaEnqueue2DOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::cpu::DmaEnqueue2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Write all DMA configuration registers BEFORE the trigger register.
    emitVolatileStore(rewriter, loc, adaptor.getSrc(), DMA_MMIO_BASE,
                      DMA_REG_SRC);
    emitVolatileStore(rewriter, loc, adaptor.getDst(), DMA_MMIO_BASE,
                      DMA_REG_DST);
    emitVolatileStore(rewriter, loc, adaptor.getSrcStride(), DMA_MMIO_BASE,
                      DMA_REG_SRC_STRIDE);
    emitVolatileStore(rewriter, loc, adaptor.getDstStride(), DMA_MMIO_BASE,
                      DMA_REG_DST_STRIDE);
    emitVolatileStore(rewriter, loc, adaptor.getHeight(), DMA_MMIO_BASE,
                      DMA_REG_HEIGHT);

    // Fence: ensure all config registers are visible before trigger.
    emitFence(rewriter, loc);

    // Write LEN register — this triggers the DMA enqueue.
    emitVolatileStore(rewriter, loc, adaptor.getWidth(), DMA_MMIO_BASE,
                      DMA_REG_LEN);

    // Fence: ensure the trigger write is ordered before any subsequent
    // memory operations (especially SPM accesses).
    emitFence(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// DmaWaitOp → polling loop on STATUS register until idle (== 0)
// ===----------------------------------------------------------------------===
struct DmaWaitOpConversion
    : public OpConversionPattern<triton::cpu::DmaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::cpu::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto i1Ty = rewriter.getI1Type();

    // Split the block at the DmaWaitOp.  Everything after the op goes
    // into continuationBB; currentBlock keeps everything before the op.
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *continuationBB = rewriter.splitBlock(currentBlock,
                                                 Block::iterator(op));

    // Create the poll loop block between current and continuation.
    Block *pollBB = rewriter.createBlock(continuationBB);

    // currentBlock: fence + branch to pollBB
    rewriter.setInsertionPointToEnd(currentBlock);
    emitFence(rewriter, loc);
    LLVM::BrOp::create(rewriter, loc, pollBB);

    // pollBB: volatile load STATUS, branch back if busy, else to continuation
    rewriter.setInsertionPointToStart(pollBB);
    Value status = emitVolatileLoad(rewriter, loc, DMA_MMIO_BASE,
                                    DMA_REG_STATUS);
    Value zero = createI64Constant(rewriter, loc, 0);
    Value busy = LLVM::ICmpOp::create(rewriter, loc, i1Ty,
                                      LLVM::ICmpPredicate::ne,
                                      status, zero);
    LLVM::CondBrOp::create(rewriter, loc, busy, pollBB, continuationBB);

    // continuationBB: fence at the start, then original ops follow
    rewriter.setInsertionPointToStart(continuationBB);
    emitFence(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pass definition
// ===----------------------------------------------------------------------===
struct DmaOpsToLLVM
    : public triton::impl::DmaOpsToLLVMBase<DmaOpsToLLVM> {
  using DmaOpsToLLVMBase::DmaOpsToLLVMBase;

  DmaOpsToLLVM() : DmaOpsToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<DmaEnqueue2DOpConversion>(typeConverter, context);
    patterns.add<DmaWaitOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createDmaOpsToLLVMPass() {
  return std::make_unique<DmaOpsToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
