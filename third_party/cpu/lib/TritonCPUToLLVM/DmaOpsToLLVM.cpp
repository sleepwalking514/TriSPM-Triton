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
// Helper: emit a RISC-V fence (fence iorw, iorw).
//
// In LLVM IR this is `fence seq_cst` which the RISC-V backend lowers to
// `fence iorw, iorw`.  We use seq_cst because we need full ordering of
// MMIO writes — the DMA engine samples registers on the LEN write, so all
// preceding register writes must be globally visible.
// ---------------------------------------------------------------------------
static void emitFence(ConversionPatternRewriter &rewriter, Location loc) {
  LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
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
// DmaWaitOp → volatile load from STATUS register (blocks until idle)
// ===----------------------------------------------------------------------===
struct DmaWaitOpConversion
    : public OpConversionPattern<triton::cpu::DmaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::cpu::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Fence before the status check to ensure all preceding stores
    // (including DMA trigger) are globally visible.
    emitFence(rewriter, loc);

    // Volatile load from STATUS register.  The gem5 DMA engine delays
    // the response until idle, so this single load acts as a blocking wait.
    emitVolatileLoad(rewriter, loc, DMA_MMIO_BASE, DMA_REG_STATUS);

    // Fence after to ensure subsequent SPM reads see the DMA'd data.
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
