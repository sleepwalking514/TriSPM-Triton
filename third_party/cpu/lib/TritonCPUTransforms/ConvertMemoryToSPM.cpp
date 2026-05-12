//===- ConvertMemoryToSPM.cpp - Convert tiled loads to DMA+SPM transfers --===//
//
// This pass transforms tiled DRAM loads inside scf.for loops into
// DMA-based scratchpad memory (SPM) transfers.
//
// Supported patterns:
//   1) GEMM double-buffering: K-loop with two tiled loads feeding a dot
//      product.  Each load gets two SPM buffers; while the current tiles
//      are computed, the next tiles are prefetched asynchronously.
//   2) Attention v2 residency: attention-style QK/PV loops stage the
//      loop-invariant Q tile once in SPM.  An experimental K/V streaming
//      schedule can also double-buffer loop-local K/V tiles without unrolling
//      the original loop.
//   3) Reduction/streaming double-buffering: single loop with one or more
//      tiled loads sharing the loop IV.  Each load gets two SPM buffers; the
//      next chunk is prefetched while the current chunk is consumed.
//
// Loads that don't match these patterns are left unchanged (cache path).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "cpu/include/TritonCPUTransforms/SPMSpaceManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/SPMAttrs.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTMEMORYTOSPM
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Single source of truth lives in `triton/Dialect/TritonCPU/IR/SPMAttrs.h`.
// Re-export under the file-local name we already use everywhere below.
static constexpr unsigned SPM_ADDR_SPACE = triton::cpu::kSPMAddressSpace;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Tracks IR inserted immediately before an anchor operation so a failed
/// speculative rewrite can leave the original cache path untouched.
class InsertedBeforeGuard {
public:
  explicit InsertedBeforeGuard(Operation *anchor)
      : anchor(anchor), prev(anchor->getPrevNode()) {}

  ~InsertedBeforeGuard() { cleanup(); }

  void commit() { active = false; }

  void cleanup() {
    if (!active)
      return;
    Operation *op = anchor->getPrevNode();
    while (op && op != prev) {
      Operation *toErase = op;
      op = op->getPrevNode();
      toErase->erase();
    }
    active = false;
  }

private:
  Operation *anchor;
  Operation *prev;
  bool active = true;
};

static Value i64Cst(OpBuilder &b, Location loc, int64_t val) {
  return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(val));
}

static Value i32Cst(OpBuilder &b, Location loc, int32_t val) {
  return arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(val));
}

static Value idxCst(OpBuilder &b, Location loc, int64_t val) {
  return arith::ConstantIndexOp::create(b, loc, val);
}

static int64_t getEnvInt64(StringRef name, int64_t defaultValue) {
  if (const char *value = std::getenv(name.str().c_str())) {
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 0);
    if (end && *end == '\0')
      return parsed;
  }
  return defaultValue;
}

static bool getEnvBool(StringRef name, bool defaultValue = false) {
  if (const char *value = std::getenv(name.str().c_str())) {
    StringRef text(value);
    return !(text.empty() || text == "0" || text.equals_insensitive("false") ||
             text.equals_insensitive("no") || text.equals_insensitive("off"));
  }
  return defaultValue;
}

static std::optional<bool> getEnvBoolOverride(StringRef name) {
  if (std::getenv(name.str().c_str()))
    return getEnvBool(name);
  return std::nullopt;
}

struct SimpleLinearExpr {
  Value symbol;
  int64_t coeff = 0;
  int64_t constant = 0;
};

static bool combineSimpleLinearExpr(const SimpleLinearExpr &lhs,
                                    const SimpleLinearExpr &rhs, int64_t sign,
                                    SimpleLinearExpr &result) {
  if (lhs.symbol && rhs.symbol && lhs.symbol != rhs.symbol)
    return false;

  result.symbol = lhs.symbol ? lhs.symbol : rhs.symbol;
  result.coeff = lhs.coeff + sign * rhs.coeff;
  result.constant = lhs.constant + sign * rhs.constant;
  if (result.coeff == 0)
    result.symbol = Value();
  return true;
}

static bool getSimpleLinearExpr(Value value, SimpleLinearExpr &expr) {
  if (auto constant = getConstantIntValue(value)) {
    expr = SimpleLinearExpr{/*symbol=*/Value(), /*coeff=*/0, *constant};
    return true;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    expr = SimpleLinearExpr{/*symbol=*/value, /*coeff=*/1, /*constant=*/0};
    return true;
  }

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return getSimpleLinearExpr(castOp.getIn(), expr);
  if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp))
    return getSimpleLinearExpr(extOp.getIn(), expr);
  if (auto extOp = dyn_cast<arith::ExtUIOp>(defOp))
    return getSimpleLinearExpr(extOp.getIn(), expr);
  if (auto truncOp = dyn_cast<arith::TruncIOp>(defOp))
    return getSimpleLinearExpr(truncOp.getIn(), expr);

  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    SimpleLinearExpr lhs, rhs;
    if (!getSimpleLinearExpr(addOp.getLhs(), lhs) ||
        !getSimpleLinearExpr(addOp.getRhs(), rhs))
      return false;
    return combineSimpleLinearExpr(lhs, rhs, /*sign=*/1, expr);
  }

  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    SimpleLinearExpr lhs, rhs;
    if (!getSimpleLinearExpr(subOp.getLhs(), lhs) ||
        !getSimpleLinearExpr(subOp.getRhs(), rhs))
      return false;
    return combineSimpleLinearExpr(lhs, rhs, /*sign=*/-1, expr);
  }

  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    if (auto rhsCst = getConstantIntValue(mulOp.getRhs())) {
      SimpleLinearExpr lhs;
      if (!getSimpleLinearExpr(mulOp.getLhs(), lhs))
        return false;
      expr = SimpleLinearExpr{lhs.symbol, lhs.coeff * *rhsCst,
                              lhs.constant * *rhsCst};
      if (expr.coeff == 0)
        expr.symbol = Value();
      return true;
    }
    if (auto lhsCst = getConstantIntValue(mulOp.getLhs())) {
      SimpleLinearExpr rhs;
      if (!getSimpleLinearExpr(mulOp.getRhs(), rhs))
        return false;
      expr = SimpleLinearExpr{rhs.symbol, rhs.coeff * *lhsCst,
                              rhs.constant * *lhsCst};
      if (expr.coeff == 0)
        expr.symbol = Value();
      return true;
    }
    return false;
  }

  expr = SimpleLinearExpr{/*symbol=*/value, /*coeff=*/1, /*constant=*/0};
  return true;
}

static std::optional<int64_t> getExactStaticTripCount(scf::ForOp forOp) {
  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!stepCst || *stepCst <= 0)
    return std::nullopt;

  SimpleLinearExpr lb, ub;
  if (!getSimpleLinearExpr(forOp.getLowerBound(), lb) ||
      !getSimpleLinearExpr(forOp.getUpperBound(), ub))
    return std::nullopt;

  if (lb.symbol != ub.symbol || lb.coeff != ub.coeff)
    return std::nullopt;

  int64_t distance = ub.constant - lb.constant;
  if (distance < 0 || distance % *stepCst != 0)
    return std::nullopt;

  return distance / *stepCst;
}

/// Cast a value to i64.  Handles index, i64 (no-op), and narrower integers.
static Value toI64(OpBuilder &b, Location loc, Value val) {
  Type ty = val.getType();
  if (ty.isInteger(64))
    return val;
  if (ty.isIndex())
    return arith::IndexCastOp::create(b, loc, b.getI64Type(), val);
  return arith::ExtSIOp::create(b, loc, b.getI64Type(), val);
}

/// Byte size of a vector tile.
static int64_t getTileBytes(VectorType vecTy) {
  int64_t elems = vecTy.getNumElements();
  unsigned bitWidth = vecTy.getElementType().getIntOrFloatBitWidth();
  return elems * (bitWidth / 8);
}

/// Get static strides from a memref type.  Returns false if dynamic.
static bool getStaticStrides(MemRefType ty, SmallVectorImpl<int64_t> &strides) {
  int64_t offset;
  if (failed(ty.getStridesAndOffset(strides, offset)))
    return false;
  for (auto s : strides)
    if (ShapedType::isDynamic(s))
      return false;
  return true;
}

/// Compute the DRAM byte address for a vector.transfer_read's source.
/// Returns nullptr if strides are dynamic.
///
/// If `ivOverride` is non-null, it replaces the loop induction variable
/// in the index computation (used to compute addresses for specific
/// iterations, e.g. the first iteration in the prologue).
static Value computeDramAddr(OpBuilder &b, Location loc,
                             vector::TransferReadOp readOp,
                             Value ivOverride = nullptr,
                             Value origIv = nullptr) {
  Value base = readOp.getBase();
  auto memRefTy = cast<MemRefType>(base.getType());
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return nullptr;

  // Base pointer as i64.
  Value ptrIdx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, base);
  Value ptrI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), ptrIdx);

  // Linear byte offset = sum(index_i * stride_i) * elemBytes.
  Value byteOff = i64Cst(b, loc, 0);
  for (unsigned i = 0; i < readOp.getIndices().size(); ++i) {
    Value idx = readOp.getIndices()[i];
    // If this index IS the loop induction variable, substitute override.
    if (ivOverride && origIv && idx == origIv)
      idx = ivOverride;
    Value idxI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), idx);
    Value contrib = arith::MulIOp::create(
        b, loc, idxI64, i64Cst(b, loc, strides[i] * elemBytes));
    byteOff = arith::AddIOp::create(b, loc, byteOff, contrib);
  }
  return arith::AddIOp::create(b, loc, ptrI64, byteOff);
}

/// Compute the DRAM byte address for the *first iteration* of a loop.
///
/// Handles two IR forms:
///   1. Regular: readOp's base/indices are defined outside the loop (or
///      include the loop IV which we substitute with the lower bound).
///   2. Block-pointer: readOp's base comes from extract_memref on a loop
///      iter_arg.  We trace back to the initial block pointer value and
///      create extract_memref / extract_indices on it outside the loop.
static Value computePrologueDramAddr(OpBuilder &b, Location loc,
                                     vector::TransferReadOp readOp,
                                     scf::ForOp forOp) {
  Value base = readOp.getBase();
  auto memRefTy = cast<MemRefType>(base.getType());
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return nullptr;

  auto *baseDefOp = base.getDefiningOp();
  bool isBlockPtr =
      baseDefOp && baseDefOp->getParentRegion() == &forOp.getRegion();

  Value ptrI64;
  SmallVector<Value> indices;

  if (isBlockPtr) {
    auto extractMR = dyn_cast<triton::cpu::ExtractMemRefOp>(baseDefOp);
    if (!extractMR)
      return nullptr;

    Value blockPtr = extractMR.getSrc();
    auto blockArg = dyn_cast<BlockArgument>(blockPtr);
    if (!blockArg || blockArg.getOwner() != forOp.getBody())
      return nullptr;

    unsigned iterArgIdx = blockArg.getArgNumber() - 1;
    if (iterArgIdx >= forOp.getInitArgs().size())
      return nullptr;

    Value initBlockPtr = forOp.getInitArgs()[iterArgIdx];
    Value initMemRef =
        triton::cpu::ExtractMemRefOp::create(b, loc, memRefTy, initBlockPtr);
    auto initIndicesOp =
        triton::cpu::ExtractIndicesOp::create(b, loc, initBlockPtr);

    Value ptrIdx =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, initMemRef);
    ptrI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), ptrIdx);

    for (auto r : initIndicesOp.getResults())
      indices.push_back(r);
  } else {
    Value ptrIdx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, base);
    ptrI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), ptrIdx);

    Value lb = forOp.getLowerBound();
    Value origIv = forOp.getInductionVar();
    for (auto idx : readOp.getIndices()) {
      if (idx == origIv)
        indices.push_back(lb);
      else
        indices.push_back(idx);
    }
  }

  Value byteOff = i64Cst(b, loc, 0);
  for (unsigned i = 0; i < indices.size(); ++i) {
    Value idxI64 =
        arith::IndexCastOp::create(b, loc, b.getI64Type(), indices[i]);
    Value contrib = arith::MulIOp::create(
        b, loc, idxI64, i64Cst(b, loc, strides[i] * elemBytes));
    byteOff = arith::AddIOp::create(b, loc, byteOff, contrib);
  }
  return arith::AddIOp::create(b, loc, ptrI64, byteOff);
}

/// Compute the DRAM byte address represented by a tensor pointer value.
static Value computeTensorPtrDramAddr(OpBuilder &b, Location loc,
                                      Value tensorPtr, MemRefType memRefTy) {
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return nullptr;

  Value memRef =
      triton::cpu::ExtractMemRefOp::create(b, loc, memRefTy, tensorPtr);
  auto indicesOp = triton::cpu::ExtractIndicesOp::create(b, loc, tensorPtr);
  if (indicesOp.getNumResults() != strides.size())
    return nullptr;

  Value ptrIdx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, memRef);
  Value ptrI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), ptrIdx);

  Value byteOff = i64Cst(b, loc, 0);
  for (auto [idx, stride] : llvm::zip_equal(indicesOp.getResults(), strides)) {
    Value idxI64 = toI64(b, loc, idx);
    Value contrib = arith::MulIOp::create(b, loc, idxI64,
                                          i64Cst(b, loc, stride * elemBytes));
    byteOff = arith::AddIOp::create(b, loc, byteOff, contrib);
  }
  return arith::AddIOp::create(b, loc, ptrI64, byteOff);
}

static bool isLoopBlockPtrTransfer(vector::TransferReadOp readOp,
                                   scf::ForOp forOp) {
  auto *baseDefOp = readOp.getBase().getDefiningOp();
  if (!baseDefOp || baseDefOp->getParentRegion() != &forOp.getRegion())
    return false;

  auto extractMR = dyn_cast<triton::cpu::ExtractMemRefOp>(baseDefOp);
  if (!extractMR)
    return false;

  auto blockArg = dyn_cast<BlockArgument>(extractMR.getSrc());
  if (!blockArg || blockArg.getOwner() != forOp.getBody())
    return false;

  unsigned iterArgIdx = blockArg.getArgNumber() - 1;
  return iterArgIdx < forOp.getInitArgs().size();
}

static bool canComputePrologueDramAddr(vector::TransferReadOp readOp,
                                       scf::ForOp forOp) {
  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!memRefTy)
    return false;
  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return false;

  auto *baseDefOp = readOp.getBase().getDefiningOp();
  if (baseDefOp && baseDefOp->getParentRegion() == &forOp.getRegion())
    return isLoopBlockPtrTransfer(readOp, forOp);

  return true;
}

static std::optional<int64_t> getLoopStepBytes(vector::TransferReadOp readOp,
                                               scf::ForOp forOp,
                                               bool requireLoopIv) {
  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!memRefTy)
    return std::nullopt;
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return std::nullopt;

  Value origIv = forOp.getInductionVar();
  for (unsigned i = 0; i < readOp.getIndices().size(); ++i) {
    if (readOp.getIndices()[i] == origIv)
      return strides[i] * elemBytes;
  }

  if (isLoopBlockPtrTransfer(readOp, forOp))
    return (strides.empty() ? 1 : strides[0]) * elemBytes;

  if (requireLoopIv)
    return std::nullopt;

  return (strides.empty() ? 1 : strides[0]) * elemBytes;
}

/// Emit triton_cpu.dma_enqueue_2d for a 2D tile.
static void emitDmaEnqueue(OpBuilder &b, Location loc, Value spmAddr,
                           Value dramAddr, VectorType vecTy,
                           MemRefType memRefTy) {
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  auto shape = vecTy.getShape();
  int64_t rows = (shape.size() >= 2) ? shape[0] : 1;
  int64_t cols = (shape.size() >= 2) ? shape[1] : shape[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  (void)memRefTy.getStridesAndOffset(strides, offset);
  int64_t srcStrideBytes = (shape.size() >= 2 ? strides[0] : cols) * elemBytes;

  triton::cpu::DmaEnqueue2DOp::create(
      b, loc, spmAddr, dramAddr, i64Cst(b, loc, cols * elemBytes), // width
      i64Cst(b, loc, rows),                                        // height
      i64Cst(b, loc, srcStrideBytes),                              // src_stride
      i64Cst(b, loc, cols * elemBytes)); // dst_stride (packed)
}

static bool hasColMajorRowBlockDmaLayout(VectorType vecTy,
                                         MemRefType memRefTy) {
  if (vecTy.getRank() != 2)
    return false;
  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides) || strides.size() < 2)
    return false;
  return strides[0] == 1 && strides[1] >= vecTy.getShape()[0];
}

static void emitRowBlockDmaEnqueue(OpBuilder &b, Location loc, Value spmAddr,
                                   Value dramAddr, VectorType vecTy,
                                   MemRefType memRefTy) {
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  auto shape = vecTy.getShape();
  if (!hasColMajorRowBlockDmaLayout(vecTy, memRefTy)) {
    emitDmaEnqueue(b, loc, spmAddr, dramAddr, vecTy, memRefTy);
    return;
  }

  SmallVector<int64_t> strides;
  int64_t offset;
  (void)memRefTy.getStridesAndOffset(strides, offset);
  int64_t columns = shape[0];
  int64_t rows = shape[1];
  triton::cpu::DmaEnqueue2DOp::create(
      b, loc, spmAddr, dramAddr,
      i64Cst(b, loc, columns * elemBytes),    // width: one row chunk
      i64Cst(b, loc, rows),                   // height: row block
      i64Cst(b, loc, strides[1] * elemBytes), // source row stride
      i64Cst(b, loc, columns * elemBytes));   // packed SPM row stride
}

static SmallVector<int64_t, 2> getDefaultSpmMemStrides(VectorType vecTy) {
  auto shape = vecTy.getShape();
  int64_t cols = (shape.size() >= 2) ? shape[1] : shape[0];
  if (vecTy.getRank() == 1)
    return SmallVector<int64_t, 2>{1};
  return SmallVector<int64_t, 2>{cols, 1};
}

static SmallVector<int64_t, 2>
getRowBlockDmaSpmMemStrides(VectorType vecTy, MemRefType memRefTy) {
  if (!hasColMajorRowBlockDmaLayout(vecTy, memRefTy))
    return getDefaultSpmMemStrides(vecTy);
  return SmallVector<int64_t, 2>{1, vecTy.getShape()[0]};
}

static SmallVector<int64_t, 2> getDmaFilledSpmMemStrides(VectorType vecTy,
                                                         MemRefType memRefTy) {
  if (hasColMajorRowBlockDmaLayout(vecTy, memRefTy))
    return getRowBlockDmaSpmMemStrides(vecTy, memRefTy);
  return getDefaultSpmMemStrides(vecTy);
}

static void emitDmaFilledEnqueue(OpBuilder &b, Location loc, Value spmAddr,
                                 Value dramAddr, VectorType vecTy,
                                 MemRefType memRefTy) {
  if (hasColMajorRowBlockDmaLayout(vecTy, memRefTy))
    emitRowBlockDmaEnqueue(b, loc, spmAddr, dramAddr, vecTy, memRefTy);
  else
    emitDmaEnqueue(b, loc, spmAddr, dramAddr, vecTy, memRefTy);
}

/// Create a memref in SPM address space (3) via reinterpret_cast, and
/// emit a vector.transfer_read from it.
///
/// We create a flat 1-element memref<1xi8, 3> from the SPM base pointer
/// using unrealized_conversion_cast (i64 → memref), then reinterpret_cast
/// it to the desired shape.  The LLVM lowering will turn this into an
/// inttoptr + load from address space 3.
static Value emitSpmReadWithStrides(OpBuilder &b, Location loc, Value spmAddr,
                                    VectorType vecTy,
                                    ArrayRef<int64_t> memStrides) {
  auto shape = vecTy.getShape();
  auto elemTy = vecTy.getElementType();

  // Build the target SPM memref type: packed layout, address space 3.
  SmallVector<int64_t> memShape(shape.begin(), shape.end());
  auto layout = StridedLayoutAttr::get(b.getContext(), 0, memStrides);
  auto spmMemRefTy = MemRefType::get(memShape, elemTy, layout,
                                     b.getI64IntegerAttr(SPM_ADDR_SPACE));

  // Create a 0-d base memref in address space 3 from the i64 address.
  // unrealized_conversion_cast will be resolved during LLVM lowering.
  auto baseMemRefTy = MemRefType::get({}, elemTy, /*layout=*/nullptr,
                                      b.getI64IntegerAttr(SPM_ADDR_SPACE));
  Value baseMemRef =
      UnrealizedConversionCastOp::create(b, loc, baseMemRefTy, spmAddr)
          ->getResult(0);

  // Reinterpret to the tiled shape with static offset 0.
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> stridesFold;
  for (auto s : memShape)
    sizes.push_back(b.getIndexAttr(s));
  for (auto s : memStrides)
    stridesFold.push_back(b.getIndexAttr(s));

  Value spmView = memref::ReinterpretCastOp::create(
      b, loc, spmMemRefTy, baseMemRef,
      /*offset=*/b.getIndexAttr(0), sizes, stridesFold);

  // Read from SPM memref.
  SmallVector<Value> zeroIndices(vecTy.getRank(), idxCst(b, loc, 0));
  auto padVal =
      arith::ConstantOp::create(b, loc, elemTy, b.getZeroAttr(elemTy));
  return vector::TransferReadOp::create(
      b, loc, vecTy, spmView, zeroIndices, padVal,
      SmallVector<bool>(vecTy.getRank(), true));
}

static Value emitSpmRead(OpBuilder &b, Location loc, Value spmAddr,
                         VectorType vecTy) {
  return emitSpmReadWithStrides(b, loc, spmAddr, vecTy,
                                getDefaultSpmMemStrides(vecTy));
}

/// Emit a vector.transfer_write to an SPM address.
static void emitSpmWriteWithStrides(OpBuilder &b, Location loc, Value spmAddr,
                                    Value value, ArrayRef<int64_t> memStrides) {
  auto vecTy = cast<VectorType>(value.getType());
  auto shape = vecTy.getShape();
  auto elemTy = vecTy.getElementType();

  SmallVector<int64_t> memShape(shape.begin(), shape.end());
  auto layout = StridedLayoutAttr::get(b.getContext(), 0, memStrides);
  auto spmMemRefTy = MemRefType::get(memShape, elemTy, layout,
                                     b.getI64IntegerAttr(SPM_ADDR_SPACE));

  auto baseMemRefTy = MemRefType::get({}, elemTy, /*layout=*/nullptr,
                                      b.getI64IntegerAttr(SPM_ADDR_SPACE));
  Value baseMemRef =
      UnrealizedConversionCastOp::create(b, loc, baseMemRefTy, spmAddr)
          ->getResult(0);

  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> stridesFold;
  for (auto s : memShape)
    sizes.push_back(b.getIndexAttr(s));
  for (auto s : memStrides)
    stridesFold.push_back(b.getIndexAttr(s));

  Value spmView = memref::ReinterpretCastOp::create(
      b, loc, spmMemRefTy, baseMemRef,
      /*offset=*/b.getIndexAttr(0), sizes, stridesFold);

  SmallVector<Value> zeroIndices(vecTy.getRank(), idxCst(b, loc, 0));
  vector::TransferWriteOp::create(b, loc, value, spmView, zeroIndices,
                                  SmallVector<bool>(vecTy.getRank(), true));
}

static void emitSpmWrite(OpBuilder &b, Location loc, Value spmAddr,
                         Value value) {
  auto vecTy = cast<VectorType>(value.getType());
  emitSpmWriteWithStrides(b, loc, spmAddr, value,
                          getDefaultSpmMemStrides(vecTy));
}

//===----------------------------------------------------------------------===//
// Tiled load descriptor.
//===----------------------------------------------------------------------===//
struct TiledLoadInfo {
  vector::TransferReadOp readOp;
  VectorType vecTy;
  int64_t tileBytes;
  bool feedsDot; // true if a user is vector.contract or triton_cpu.dot
};

static bool isShapePreservingCastLikeOp(Operation *op) {
  if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  return isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp,
             arith::TruncIOp, arith::IndexCastOp, arith::BitcastOp,
             vector::ShapeCastOp, UnrealizedConversionCastOp>(op);
}

static bool valueFeedsDot(Value value, unsigned depth = 0) {
  if (!value || depth > 4)
    return false;
  for (auto *user : value.getUsers()) {
    if (isa<vector::ContractionOp>(user) || isa<triton::cpu::DotOp>(user))
      return true;
    if (isShapePreservingCastLikeOp(user) &&
        valueFeedsDot(user->getResult(0), depth + 1))
      return true;
  }
  return false;
}

static vector::TransferReadOp
getTransferReadThroughShapePreservingCasts(Value value, unsigned depth = 0) {
  if (!value || depth > 4)
    return {};
  if (auto readOp = value.getDefiningOp<vector::TransferReadOp>())
    return readOp;
  Operation *defOp = value.getDefiningOp();
  if (!isShapePreservingCastLikeOp(defOp))
    return {};
  return getTransferReadThroughShapePreservingCasts(defOp->getOperand(0),
                                                    depth + 1);
}

/// Check if a transfer_read feeds a dot-like operation, allowing the common
/// tl.load(...).to(f32) cast between the read and the contract.
static bool readFeedsDot(vector::TransferReadOp readOp) {
  return valueFeedsDot(readOp.getResult());
}

/// Find eligible tiled loads inside an scf.for body.
/// Eligible = vector.transfer_read from a DRAM memref (not SPM), with
/// static strides, and rank >= 1.
static SmallVector<TiledLoadInfo> findTiledLoads(scf::ForOp forOp) {
  SmallVector<TiledLoadInfo> results;
  forOp.getBody()->walk([&](vector::TransferReadOp readOp) {
    auto vecTy = dyn_cast<VectorType>(readOp.getType());
    if (!vecTy || vecTy.getRank() < 1)
      return;
    auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
    if (!memRefTy)
      return;
    if (memRefTy.getMemorySpaceAsInt() == SPM_ADDR_SPACE)
      return;
    SmallVector<int64_t> strides;
    if (!getStaticStrides(memRefTy, strides))
      return;

    results.push_back(
        {readOp, vecTy, getTileBytes(vecTy), readFeedsDot(readOp)});
  });
  return results;
}

enum class ReductionProducerPass {
  FillOnFirstPass,
  ProducerStore,
  DmaPrefetch,
};

enum class ReductionBufferRole {
  ResidentRow,
  ResidentRowBlock,
  PingPongChunk,
  TempVector,
  OutputTile,
};

enum class ReductionRotationPolicy {
  None,
  DoubleBuffer,
  Ring,
};

enum class ReductionCopyInMode {
  CpuDirect,
  Dma,
};

static StringRef stringifyProducerPass(ReductionProducerPass pass) {
  switch (pass) {
  case ReductionProducerPass::FillOnFirstPass:
    return "fill_on_first_pass";
  case ReductionProducerPass::ProducerStore:
    return "producer_store";
  case ReductionProducerPass::DmaPrefetch:
    return "dma_prefetch";
  }
  llvm_unreachable("unknown reduction producer pass");
}

static StringRef stringifyBufferRole(ReductionBufferRole role) {
  switch (role) {
  case ReductionBufferRole::ResidentRow:
    return "resident_row";
  case ReductionBufferRole::ResidentRowBlock:
    return "resident_row_block";
  case ReductionBufferRole::PingPongChunk:
    return "ping_pong_chunk";
  case ReductionBufferRole::TempVector:
    return "temp_vector";
  case ReductionBufferRole::OutputTile:
    return "output_tile";
  }
  llvm_unreachable("unknown reduction buffer role");
}

static StringRef stringifyRotationPolicy(ReductionRotationPolicy policy) {
  switch (policy) {
  case ReductionRotationPolicy::None:
    return "none";
  case ReductionRotationPolicy::DoubleBuffer:
    return "double_buffer";
  case ReductionRotationPolicy::Ring:
    return "ring_N";
  }
  llvm_unreachable("unknown reduction rotation policy");
}

static StringRef stringifyCopyInMode(ReductionCopyInMode mode) {
  switch (mode) {
  case ReductionCopyInMode::CpuDirect:
    return "cpu_direct";
  case ReductionCopyInMode::Dma:
    return "dma";
  }
  llvm_unreachable("unknown reduction copy-in mode");
}

struct ReductionLoopResidencyUse {
  std::string passName;
  scf::ForOp forOp;
  TiledLoadInfo xLoad;
};

struct ReductionResidencyPlan {
  std::string pattern = "row_resident_reduction";
  std::string source;
  std::string scope = "program-row";
  BlockArgument sourceArg;
  scf::ForOp rowBlockGroupLoop;
  ReductionLoopResidencyUse producer;
  SmallVector<ReductionLoopResidencyUse, 2> consumers;
  SmallVector<scf::ForOp, 3> loops;
  SmallVector<int64_t, 2> shape;
  int64_t trips = 0;
  int64_t rowBlockGroupTrips = 0;
  int64_t bytes = 0;
  unsigned elemBytes = 0;
  int64_t uses = 0;
  int64_t requiredSpmSlots = 1;
  ReductionProducerPass producerPass = ReductionProducerPass::FillOnFirstPass;
  ReductionBufferRole bufferRole = ReductionBufferRole::ResidentRow;
  ReductionRotationPolicy rotationPolicy = ReductionRotationPolicy::None;
  ReductionCopyInMode copyInMode = ReductionCopyInMode::CpuDirect;
  std::string copyIn = "CPU/vector store";
  std::string copyOut = "none";
  std::string overhead;
  std::string benefit;
  SmallVector<std::string, 4> expectedMarkers;
  bool hasLowering = false;
};

static bool isRowBlockDmaProducerPassMode(StringRef mode) {
  return mode == "row_block_dma" || mode == "dma_row_block" ||
         mode == "row_block_dma_prefetch";
}

static ReductionProducerPass parseReductionProducerPass(StringRef mode) {
  if (mode.empty())
    return ReductionProducerPass::FillOnFirstPass;
  if (mode == "fill_on_first_pass" || mode == "first" || mode == "first_pass")
    return ReductionProducerPass::FillOnFirstPass;
  if (mode == "producer_store" || mode == "second" || mode == "second_pass" ||
      mode == "consumer_store")
    return ReductionProducerPass::ProducerStore;
  if (mode == "dma_prefetch" || mode == "dma" || mode == "prefetch")
    return ReductionProducerPass::DmaPrefetch;
  if (isRowBlockDmaProducerPassMode(mode))
    return ReductionProducerPass::DmaPrefetch;
  return ReductionProducerPass::FillOnFirstPass;
}

static void applyConfiguredProducerPass(ReductionResidencyPlan &plan,
                                        StringRef producerPassMode) {
  ReductionProducerPass producerPass =
      parseReductionProducerPass(producerPassMode);
  if (producerPass == ReductionProducerPass::FillOnFirstPass)
    return;

  if (producerPass == ReductionProducerPass::ProducerStore) {
    if (plan.consumers.empty())
      return;
    plan.producerPass = ReductionProducerPass::ProducerStore;
    plan.copyIn = "CPU/vector store from " + plan.consumers.front().passName;
    plan.overhead =
        "first reduction pass stays on the original path; the next pass writes "
        "each loaded x chunk into SPM for later reuse";
    plan.benefit =
        "avoids one SPM read from the fill-on-first-pass schedule while still "
        "using SPM for the final x reuse pass";
    return;
  }

  bool preferRowBlock = isRowBlockDmaProducerPassMode(producerPassMode);
  plan.producerPass = ReductionProducerPass::DmaPrefetch;
  plan.bufferRole = preferRowBlock ? ReductionBufferRole::ResidentRowBlock
                                   : ReductionBufferRole::ResidentRow;
  plan.rotationPolicy = ReductionRotationPolicy::DoubleBuffer;
  plan.copyInMode = ReductionCopyInMode::Dma;
  if (preferRowBlock) {
    plan.copyIn = "row-block DMA prefetch";
    plan.overhead =
        "producer pass waits for each DMA-filled row-block chunk and "
        "prefetches the next chunk while computing the current reduction chunk";
    plan.benefit =
        "x[row_block, :] is materialized through DMA into SPM and reused by "
        "every reduction pass";
  } else {
    plan.copyIn = "DMA prefetch";
    plan.overhead =
        "producer pass waits for each DMA-filled x chunk and prefetches the "
        "next chunk while computing the current reduction chunk";
    plan.benefit =
        "x[row, :] is materialized through DMA into SPM and reused by every "
        "reduction pass";
  }
  plan.expectedMarkers.clear();
  plan.expectedMarkers.push_back("addrspace(3)");
  plan.expectedMarkers.push_back("dma_descriptors");
  plan.expectedMarkers.push_back("fence_iorw");
}

static bool isGemmContract(vector::ContractionOp op) {
  auto iterTypes = op.getIteratorTypes().getValue();
  if (iterTypes.size() != 3)
    return false;
  using IT = vector::IteratorType;
  auto get = [](Attribute a) {
    return cast<vector::IteratorTypeAttr>(a).getValue();
  };
  if (get(iterTypes[0]) != IT::parallel || get(iterTypes[1]) != IT::parallel ||
      get(iterTypes[2]) != IT::reduction)
    return false;
  auto maps = op.getIndexingMaps();
  MLIRContext *ctx = op.getContext();
  return cast<AffineMapAttr>(maps[0]).getValue() ==
             AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx) &&
         cast<AffineMapAttr>(maps[1]).getValue() ==
             AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx) &&
         cast<AffineMapAttr>(maps[2]).getValue() ==
             AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
}

struct GemmContractInfo {
  vector::ContractionOp contractOp;
  unsigned accIdx;
  VectorType accTy;
};

static std::optional<GemmContractInfo>
analyzeGemmContract(scf::ForOp forOp, vector::TransferReadOp readA,
                    vector::TransferReadOp readB) {
  vector::ContractionOp contractOp;
  for (auto *user : readA->getUsers()) {
    auto candidate = dyn_cast<vector::ContractionOp>(user);
    if (!candidate || !isGemmContract(candidate))
      continue;
    if (candidate.getLhs() == readA.getResult() &&
        candidate.getRhs() == readB.getResult()) {
      contractOp = candidate;
      break;
    }
  }
  if (!contractOp)
    return std::nullopt;

  auto accTy = dyn_cast<VectorType>(contractOp.getAcc().getType());
  if (!accTy || accTy.getRank() != 2)
    return std::nullopt;

  auto blockArg = dyn_cast<BlockArgument>(contractOp.getAcc());
  if (!blockArg || blockArg.getOwner() != forOp.getBody())
    return std::nullopt;

  unsigned idx = blockArg.getArgNumber() - 1;
  if (idx >= forOp.getRegionIterArgs().size())
    return std::nullopt;

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (yieldOp.getOperand(idx) != contractOp.getResult())
    return std::nullopt;

  return GemmContractInfo{contractOp, idx, accTy};
}

static int64_t chooseWindowK(int64_t trips, int64_t requestedWindowK) {
  int64_t limit = std::max<int64_t>(1, std::min(trips, requestedWindowK));
  for (int64_t w = limit; w >= 1; --w)
    if (trips % w == 0)
      return w;
  return 1;
}

//===----------------------------------------------------------------------===//
// Promotion evidence.
//===----------------------------------------------------------------------===//

struct SPMProfitabilityEvidence {
  bool present = false;
  std::string model = "phase35_p3_static_best_baseline_v1";
  std::string baseline = "best_legal_cache_schedule";
  std::string decision;
  std::string reasonCode;
  std::string reason;
  int64_t dmaDescriptors = 0;
  int64_t mmioStores = 0;
  int64_t waits = 0;
  int64_t fences = 0;
  int64_t copyBytes = 0;
  int64_t spmWriteBytes = 0;
  int64_t spmReadBytes = 0;
  int64_t avoidedRepeatedReadBytes = 0;
  int64_t liveSpmBytes = 0;
  int64_t estimatedExtraOps = 0;
  int64_t measuredBankConflicts = 0;
  int64_t uses = 0;
};

struct SPMResidencyPlanEvidence {
  bool present = false;
  std::string producerPass;
  SmallVector<std::string, 4> consumerPasses;
  std::string bufferRole;
  std::string rotationPolicy;
  std::string copyInMode;
  int64_t requiredSpmSlots = 0;
  SmallVector<std::string, 4> expectedMarkers;
};

struct SPMPromotionRecord {
  std::string status = "accepted";
  std::string source;
  std::string scope;
  SmallVector<int64_t> shape;
  int64_t uses = 0;
  std::string copyIn;
  std::string copyOut;
  int64_t bytes = 0;
  int64_t spmAddress = 0;
  std::string overhead;
  std::string benefit;
  std::string reasonCode = "accepted_existing_schedule";
  std::string reason =
      "accepted by the existing fused SPM schedule; D1 report only";
  SPMResidencyPlanEvidence residencyPlan;
  SPMProfitabilityEvidence profitability;
};

struct SPMPromotionRejection {
  std::string status = "rejected";
  std::string pattern;
  std::string source;
  std::string scope;
  SmallVector<int64_t> shape;
  int64_t uses = 0;
  std::string copyIn = "none";
  std::string copyOut = "none";
  int64_t bytes = 0;
  std::string reasonCode;
  std::string reason;
  SPMResidencyPlanEvidence residencyPlan;
  SPMProfitabilityEvidence profitability;
};

static constexpr int64_t kDmaMmioStoresPerDescriptor = 4;

static SPMProfitabilityEvidence makeD3ProfitabilityEvidence(
    StringRef decision, StringRef reasonCode, StringRef reason,
    int64_t dmaDescriptors, int64_t waits, int64_t copyBytes,
    int64_t spmWriteBytes, int64_t spmReadBytes,
    int64_t avoidedRepeatedReadBytes, int64_t liveSpmBytes,
    int64_t estimatedExtraOps, int64_t measuredBankConflicts, int64_t uses) {
  SPMProfitabilityEvidence evidence;
  evidence.present = true;
  evidence.decision = decision.str();
  evidence.reasonCode = reasonCode.str();
  evidence.reason = reason.str();
  evidence.dmaDescriptors = dmaDescriptors;
  evidence.mmioStores = dmaDescriptors * kDmaMmioStoresPerDescriptor;
  evidence.waits = waits;
  evidence.fences = waits;
  evidence.copyBytes = copyBytes;
  evidence.spmWriteBytes = spmWriteBytes;
  evidence.spmReadBytes = spmReadBytes;
  evidence.avoidedRepeatedReadBytes = avoidedRepeatedReadBytes;
  evidence.liveSpmBytes = liveSpmBytes;
  evidence.estimatedExtraOps = estimatedExtraOps;
  evidence.measuredBankConflicts = measuredBankConflicts;
  evidence.uses = uses;
  return evidence;
}

static void attachD3Profitability(SPMPromotionRecord &record,
                                  SPMProfitabilityEvidence evidence) {
  record.profitability = std::move(evidence);
}

static void attachD3Profitability(SPMPromotionRejection &rejection,
                                  SPMProfitabilityEvidence evidence) {
  rejection.profitability = std::move(evidence);
}

static SPMResidencyPlanEvidence
makeResidencyPlanEvidence(const ReductionResidencyPlan &plan) {
  SPMResidencyPlanEvidence evidence;
  evidence.present = true;
  evidence.producerPass = stringifyProducerPass(plan.producerPass).str();
  evidence.bufferRole = stringifyBufferRole(plan.bufferRole).str();
  evidence.rotationPolicy = stringifyRotationPolicy(plan.rotationPolicy).str();
  evidence.copyInMode = stringifyCopyInMode(plan.copyInMode).str();
  evidence.requiredSpmSlots = plan.requiredSpmSlots;
  for (const ReductionLoopResidencyUse &use : plan.consumers)
    evidence.consumerPasses.push_back(use.passName);
  evidence.expectedMarkers.append(plan.expectedMarkers.begin(),
                                  plan.expectedMarkers.end());
  return evidence;
}

struct SPMPromotionReport {
  SmallVector<SPMPromotionRecord, 4> records;
  SmallVector<SPMPromotionRejection, 4> rejections;
};

static void appendShape(SmallVectorImpl<int64_t> &shape, VectorType vecTy) {
  for (int64_t dim : vecTy.getShape())
    shape.push_back(dim);
}

static void writeJsonString(llvm::raw_ostream &os, StringRef value) {
  os << "\"";
  for (char c : value) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << "\"";
}

static void writeJsonShape(llvm::raw_ostream &os, ArrayRef<int64_t> shape) {
  os << "[";
  for (auto [index, dim] : llvm::enumerate(shape)) {
    if (index)
      os << ", ";
    os << dim;
  }
  os << "]";
}

static void writeJsonStringArray(llvm::raw_ostream &os,
                                 ArrayRef<std::string> values) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index)
      os << ", ";
    writeJsonString(os, value);
  }
  os << "]";
}

static void writePromotionFieldKinds(llvm::raw_ostream &os, StringRef indent) {
  os << indent << "\"field_kinds\": {\n";
  os << indent << "  \"source\": \"exact\",\n";
  os << indent << "  \"scope\": \"exact\",\n";
  os << indent << "  \"shape\": \"exact\",\n";
  os << indent << "  \"uses\": \"exact-static\",\n";
  os << indent << "  \"bytes\": \"exact-static\",\n";
  os << indent << "  \"copy_in\": \"exact\",\n";
  os << indent << "  \"copy_out\": \"exact\",\n";
  os << indent << "  \"spm_address\": \"exact-static\",\n";
  os << indent << "  \"overhead\": \"estimated-structural\",\n";
  os << indent << "  \"benefit\": \"estimated-structural\"\n";
  os << indent << "}";
}

static void writeProfitabilityEvidence(llvm::raw_ostream &os,
                                       const SPMProfitabilityEvidence &evidence,
                                       StringRef indent) {
  os << indent << "\"profitability\": {\n";
  os << indent << "  \"model\": ";
  writeJsonString(os, evidence.model);
  os << ",\n";
  os << indent << "  \"baseline\": ";
  writeJsonString(os, evidence.baseline);
  os << ",\n";
  os << indent << "  \"decision\": ";
  writeJsonString(os, evidence.decision);
  os << ",\n";
  os << indent << "  \"reason_code\": ";
  writeJsonString(os, evidence.reasonCode);
  os << ",\n";
  os << indent << "  \"reason\": ";
  writeJsonString(os, evidence.reason);
  os << ",\n";
  os << indent << "  \"dma_descriptors\": " << evidence.dmaDescriptors << ",\n";
  os << indent << "  \"mmio_stores\": " << evidence.mmioStores << ",\n";
  os << indent << "  \"waits\": " << evidence.waits << ",\n";
  os << indent << "  \"fences\": " << evidence.fences << ",\n";
  os << indent << "  \"copy_bytes\": " << evidence.copyBytes << ",\n";
  os << indent << "  \"spm_write_bytes\": " << evidence.spmWriteBytes << ",\n";
  os << indent << "  \"spm_read_bytes\": " << evidence.spmReadBytes << ",\n";
  os << indent << "  \"avoided_repeated_read_bytes\": "
     << evidence.avoidedRepeatedReadBytes << ",\n";
  os << indent << "  \"live_spm_bytes\": " << evidence.liveSpmBytes << ",\n";
  os << indent << "  \"estimated_extra_ops\": " << evidence.estimatedExtraOps
     << ",\n";
  os << indent
     << "  \"measured_bank_conflicts\": " << evidence.measuredBankConflicts
     << ",\n";
  os << indent << "  \"uses\": " << evidence.uses << "\n";
  os << indent << "}";
}

static void writeResidencyPlanEvidence(llvm::raw_ostream &os,
                                       const SPMResidencyPlanEvidence &evidence,
                                       StringRef indent) {
  os << indent << "\"residency_plan\": {\n";
  os << indent << "  \"producer_pass\": ";
  writeJsonString(os, evidence.producerPass);
  os << ",\n";
  os << indent << "  \"consumer_passes\": ";
  writeJsonStringArray(os, evidence.consumerPasses);
  os << ",\n";
  os << indent << "  \"buffer_role\": ";
  writeJsonString(os, evidence.bufferRole);
  os << ",\n";
  os << indent << "  \"rotation_policy\": ";
  writeJsonString(os, evidence.rotationPolicy);
  os << ",\n";
  os << indent << "  \"copy_in_mode\": ";
  writeJsonString(os, evidence.copyInMode);
  os << ",\n";
  os << indent << "  \"required_spm_slots\": " << evidence.requiredSpmSlots
     << ",\n";
  os << indent << "  \"expected_markers\": ";
  writeJsonStringArray(os, evidence.expectedMarkers);
  os << "\n";
  os << indent << "}";
}

static SPMPromotionRejection makePromotionRejection(StringRef pattern,
                                                    StringRef reasonCode,
                                                    StringRef reason) {
  SPMPromotionRejection rejection;
  rejection.pattern = pattern.str();
  rejection.source = pattern.str();
  rejection.scope = "candidate";
  rejection.reasonCode = reasonCode.str();
  rejection.reason = reason.str();
  return rejection;
}

static SPMPromotionRecord
makeReductionResidencyRecord(const ReductionResidencyPlan &plan,
                             int64_t spmAddress) {
  SPMPromotionRecord record;
  record.source = plan.source;
  record.scope = plan.scope;
  record.shape.append(plan.shape.begin(), plan.shape.end());
  record.uses = plan.uses;
  record.copyIn = plan.copyIn;
  record.copyOut = plan.copyOut;
  record.bytes = plan.bytes;
  record.spmAddress = spmAddress;
  record.overhead = plan.overhead;
  record.benefit = plan.benefit;
  if (plan.producerPass == ReductionProducerPass::ProducerStore) {
    record.reasonCode = "accepted_producer_store_row_resident";
    record.reason =
        "accepted by the opt-in producer-store row-resident reduction "
        "prototype";
  } else if (plan.producerPass == ReductionProducerPass::DmaPrefetch) {
    if (plan.bufferRole == ReductionBufferRole::ResidentRowBlock) {
      record.reasonCode = "accepted_block_resident_fill_first";
      record.reason =
          "accepted by the opt-in row-block resident reduction prototype";
    } else {
      record.reasonCode = "accepted_dma_prefetch_row_resident";
      record.reason = "accepted by the opt-in DMA-prefetch row-resident "
                      "reduction prototype";
    }
  } else {
    record.reasonCode = "accepted_row_resident_fill_first";
    record.reason =
        "accepted by the opt-in fill-on-first-pass row-resident reduction "
        "prototype";
  }
  record.residencyPlan = makeResidencyPlanEvidence(plan);
  return record;
}

static SPMPromotionRejection
makeRowResidentRejection(StringRef reasonCode, StringRef reason,
                         ArrayRef<int64_t> shape = ArrayRef<int64_t>(),
                         int64_t uses = 0, int64_t bytes = 0) {
  SPMPromotionRejection rejection =
      makePromotionRejection("row_resident_reduction", reasonCode, reason);
  rejection.source = "LayerNorm x row";
  rejection.scope = "program-row candidate";
  rejection.uses = uses;
  rejection.copyIn = "CPU/vector store";
  rejection.copyOut = "none";
  rejection.bytes = bytes;
  rejection.shape.append(shape.begin(), shape.end());
  return rejection;
}

static SPMPromotionRejection
makeReductionResidencyRejection(const ReductionResidencyPlan &plan,
                                StringRef reasonCode, StringRef reason) {
  SPMPromotionRejection rejection =
      makePromotionRejection(plan.pattern, reasonCode, reason);
  rejection.source = plan.source;
  rejection.scope = plan.scope + " candidate";
  rejection.uses = plan.uses;
  rejection.copyIn = plan.copyIn;
  rejection.copyOut = plan.copyOut;
  rejection.bytes = plan.bytes;
  rejection.shape.append(plan.shape.begin(), plan.shape.end());
  rejection.residencyPlan = makeResidencyPlanEvidence(plan);
  return rejection;
}

static LogicalResult writePromotionReport(FunctionOpInterface funcOp,
                                          const SPMPromotionReport &report) {
  const char *auxDir = std::getenv("KERNEL_AUX_FILE_DIR");
  if (!auxDir || StringRef(auxDir).empty())
    return success();

  SmallString<256> path(auxDir);
  std::string filename = funcOp.getName().str() + "_promotions.json";
  llvm::sys::path::append(path, filename);

  std::error_code error;
  llvm::raw_fd_ostream os(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return funcOp.emitError("failed to write SPM promotion sidecar '")
           << path << "': " << error.message();

  os << "{\n";
  os << "  \"schema_version\": 1,\n";
  os << "  \"schema\": \"triton_cpu_spm_promotion_d1\",\n";
  os << "  \"contract\": \"debug/evidence sidecar; not a graph manifest or "
        "durable IR contract\",\n";
  os << "  \"kernel\": ";
  writeJsonString(os, funcOp.getName());
  os << ",\n";

  os << "  \"promotions\": [\n";
  for (auto [index, record] : llvm::enumerate(report.records)) {
    os << "    {\n";
    os << "      \"status\": ";
    writeJsonString(os, record.status);
    os << ",\n";
    os << "      \"source\": ";
    writeJsonString(os, record.source);
    os << ",\n";
    os << "      \"scope\": ";
    writeJsonString(os, record.scope);
    os << ",\n";
    os << "      \"shape\": ";
    writeJsonShape(os, record.shape);
    os << ",\n";
    os << "      \"uses\": " << record.uses << ",\n";
    os << "      \"copy_in\": ";
    writeJsonString(os, record.copyIn);
    os << ",\n";
    os << "      \"copy_out\": ";
    writeJsonString(os, record.copyOut);
    os << ",\n";
    os << "      \"bytes\": " << record.bytes << ",\n";
    os << "      \"spm_address\": " << record.spmAddress << ",\n";
    os << "      \"overhead\": ";
    writeJsonString(os, record.overhead);
    os << ",\n";
    os << "      \"benefit\": ";
    writeJsonString(os, record.benefit);
    os << ",\n";
    os << "      \"reason_code\": ";
    writeJsonString(os, record.reasonCode);
    os << ",\n";
    os << "      \"reason\": ";
    writeJsonString(os, record.reason);
    os << ",\n";
    writePromotionFieldKinds(os, "      ");
    if (record.residencyPlan.present) {
      os << ",\n";
      writeResidencyPlanEvidence(os, record.residencyPlan, "      ");
    }
    if (record.profitability.present) {
      os << ",\n";
      writeProfitabilityEvidence(os, record.profitability, "      ");
    }
    os << "\n";
    os << "    }";
    if (index + 1 != report.records.size())
      os << ",";
    os << "\n";
  }
  os << "  ],\n";

  os << "  \"rejections\": [\n";
  for (auto [index, rejection] : llvm::enumerate(report.rejections)) {
    os << "    {\n";
    os << "      \"status\": ";
    writeJsonString(os, rejection.status);
    os << ",\n";
    os << "      \"pattern\": ";
    writeJsonString(os, rejection.pattern);
    os << ",\n";
    os << "      \"source\": ";
    writeJsonString(os, rejection.source);
    os << ",\n";
    os << "      \"scope\": ";
    writeJsonString(os, rejection.scope);
    os << ",\n";
    os << "      \"shape\": ";
    writeJsonShape(os, rejection.shape);
    os << ",\n";
    os << "      \"uses\": " << rejection.uses << ",\n";
    os << "      \"copy_in\": ";
    writeJsonString(os, rejection.copyIn);
    os << ",\n";
    os << "      \"copy_out\": ";
    writeJsonString(os, rejection.copyOut);
    os << ",\n";
    os << "      \"bytes\": " << rejection.bytes << ",\n";
    os << "      \"reason_code\": ";
    writeJsonString(os, rejection.reasonCode);
    os << ",\n";
    os << "      \"reason\": ";
    writeJsonString(os, rejection.reason);
    os << ",\n";
    os << "      \"field_kinds\": {\n";
    os << "        \"reason_code\": \"exact\",\n";
    os << "        \"reason\": \"exact\",\n";
    os << "        \"source\": \"exact\",\n";
    os << "        \"scope\": \"exact\",\n";
    os << "        \"shape\": \"exact-if-known\",\n";
    os << "        \"uses\": \"exact-if-known\",\n";
    os << "        \"bytes\": \"exact-if-known\"\n";
    os << "      }";
    if (rejection.residencyPlan.present) {
      os << ",\n";
      writeResidencyPlanEvidence(os, rejection.residencyPlan, "      ");
    }
    if (rejection.profitability.present) {
      os << ",\n";
      writeProfitabilityEvidence(os, rejection.profitability, "      ");
      os << "\n";
    } else {
      os << "\n";
    }
    os << "    }";
    if (index + 1 != report.rejections.size())
      os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";

  return success();
}

//===----------------------------------------------------------------------===//
// D2 row-resident reduction prototype.
//
// This is intentionally separate from the older reduction/streaming lowering:
// it only runs behind enable-row-resident-reductions and only stages the
// LayerNorm x row.  D1 sidecar records explain accepted/rejected evidence, but
// no report field is consulted to choose a schedule.
//===----------------------------------------------------------------------===//

static BlockArgument traceFunctionArgument(Value value,
                                           FunctionOpInterface funcOp,
                                           unsigned depth = 0) {
  if (depth > 8)
    return nullptr;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (parentOp == funcOp)
      return blockArg;

    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      for (auto [iterArg, initArg] :
           llvm::zip_equal(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
        if (blockArg == iterArg)
          return traceFunctionArgument(initArg, funcOp, depth + 1);
      }
    }

    return nullptr;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;

  if (auto extractMemRef = dyn_cast<triton::cpu::ExtractMemRefOp>(defOp))
    return traceFunctionArgument(extractMemRef.getSrc(), funcOp, depth + 1);

  if (auto makeTensorPtr = dyn_cast<triton::MakeTensorPtrOp>(defOp))
    return traceFunctionArgument(makeTensorPtr.getBase(), funcOp, depth + 1);

  return nullptr;
}

static bool sameStaticLoopShape(scf::ForOp lhs, scf::ForOp rhs) {
  auto lhsLb = getConstantIntValue(lhs.getLowerBound());
  auto lhsUb = getConstantIntValue(lhs.getUpperBound());
  auto lhsStep = getConstantIntValue(lhs.getStep());
  auto rhsLb = getConstantIntValue(rhs.getLowerBound());
  auto rhsUb = getConstantIntValue(rhs.getUpperBound());
  auto rhsStep = getConstantIntValue(rhs.getStep());
  return lhsLb && lhsUb && lhsStep && rhsLb && rhsUb && rhsStep &&
         *lhsLb == *rhsLb && *lhsUb == *rhsUb && *lhsStep == *rhsStep;
}

static bool isSupportedRowResidentXLoad(TiledLoadInfo load, scf::ForOp forOp,
                                        unsigned &elemBytes) {
  if (load.feedsDot || load.vecTy.getRank() != 1)
    return false;
  Type elemTy = load.vecTy.getElementType();
  if (!elemTy.isF32())
    return false;

  auto memRefTy = dyn_cast<MemRefType>(load.readOp.getBase().getType());
  if (!memRefTy || !memRefTy.getElementType().isF32())
    return false;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides) || strides.size() != 1 ||
      strides[0] != 1)
    return false;

  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!stepCst || *stepCst != load.vecTy.getNumElements())
    return false;

  auto stepBytes = getLoopStepBytes(load.readOp, forOp, /*requireLoopIv=*/true);
  elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  return stepBytes && *stepBytes == static_cast<int64_t>(elemBytes) &&
         canComputePrologueDramAddr(load.readOp, forOp);
}

static bool isSupportedSoftmaxRowBlockXLoad(TiledLoadInfo load,
                                            scf::ForOp forOp,
                                            unsigned &elemBytes) {
  vector::TransferReadOp readOp = load.readOp;
  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!memRefTy || load.vecTy.getRank() != 2 || memRefTy.getRank() != 2 ||
      !memRefTy.getElementType().isF32())
    return false;

  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!stepCst || *stepCst != load.vecTy.getShape()[0])
    return false;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides) || strides.size() < 2 ||
      strides[0] != 1 || strides[1] <= load.vecTy.getShape()[0])
    return false;

  auto stepBytes = getLoopStepBytes(readOp, forOp, /*requireLoopIv=*/true);
  elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  return stepBytes && *stepBytes == static_cast<int64_t>(elemBytes) &&
         canComputePrologueDramAddr(readOp, forOp);
}

static std::optional<ReductionResidencyPlan> matchLayerNormResidencyPlan(
    ArrayRef<scf::ForOp> loops, FunctionOpInterface funcOp,
    std::string &rejectReasonCode, std::string &rejectReason,
    StringRef producerPassMode) {
  if (loops.size() < 3) {
    rejectReasonCode = "unsupported_pattern";
    rejectReason = "expected three top-level LayerNorm row loops";
    return std::nullopt;
  }

  for (size_t i = 0; i + 2 < loops.size(); ++i) {
    scf::ForOp meanLoop = loops[i];
    scf::ForOp varLoop = loops[i + 1];
    scf::ForOp normLoop = loops[i + 2];
    if (!sameStaticLoopShape(meanLoop, varLoop) ||
        !sameStaticLoopShape(meanLoop, normLoop))
      continue;

    SmallVector<TiledLoadInfo> meanLoads = findTiledLoads(meanLoop);
    SmallVector<TiledLoadInfo> varLoads = findTiledLoads(varLoop);
    SmallVector<TiledLoadInfo> normLoads = findTiledLoads(normLoop);
    if (meanLoads.size() != 1 || varLoads.size() != 1 || normLoads.size() < 3)
      continue;

    BlockArgument meanArg =
        traceFunctionArgument(meanLoads[0].readOp.getBase(), funcOp);
    BlockArgument varArg =
        traceFunctionArgument(varLoads[0].readOp.getBase(), funcOp);
    if (!meanArg || !varArg || meanArg != varArg)
      continue;

    std::optional<TiledLoadInfo> normXLoad;
    for (TiledLoadInfo load : normLoads) {
      if (traceFunctionArgument(load.readOp.getBase(), funcOp) == meanArg) {
        normXLoad = load;
        break;
      }
    }
    if (!normXLoad)
      continue;

    unsigned elemBytes = 0;
    unsigned elemBytesVar = 0;
    unsigned elemBytesNorm = 0;
    if (!isSupportedRowResidentXLoad(meanLoads[0], meanLoop, elemBytes) ||
        !isSupportedRowResidentXLoad(varLoads[0], varLoop, elemBytesVar) ||
        !isSupportedRowResidentXLoad(*normXLoad, normLoop, elemBytesNorm) ||
        elemBytes != elemBytesVar || elemBytes != elemBytesNorm) {
      rejectReasonCode = "unsupported_pattern";
      rejectReason =
          "candidate requires rank-1 contiguous fp32 x loads with static "
          "BLOCK_N-sized loop steps";
      return std::nullopt;
    }

    auto lbCst = getConstantIntValue(meanLoop.getLowerBound());
    auto ubCst = getConstantIntValue(meanLoop.getUpperBound());
    auto stepCst = getConstantIntValue(meanLoop.getStep());
    if (!lbCst || !ubCst || !stepCst || *stepCst <= 0 ||
        (*ubCst - *lbCst) % *stepCst != 0) {
      rejectReasonCode = "dynamic_shape_or_stride";
      rejectReason = "loop bounds/step are not static with exact trip count";
      return std::nullopt;
    }

    int64_t trips = (*ubCst - *lbCst) / *stepCst;
    if (trips <= 0) {
      rejectReasonCode = "unsupported_pattern";
      rejectReason = "loop trip count must be positive";
      return std::nullopt;
    }

    int64_t rowBytes = trips * meanLoads[0].tileBytes;
    int64_t rowElements = trips * meanLoads[0].vecTy.getNumElements();
    ReductionResidencyPlan plan;
    plan.source = "LayerNorm x row";
    plan.sourceArg = meanArg;
    plan.producer = ReductionLoopResidencyUse{"mean", meanLoop, meanLoads[0]};
    plan.consumers.push_back(
        ReductionLoopResidencyUse{"variance", varLoop, varLoads[0]});
    plan.consumers.push_back(
        ReductionLoopResidencyUse{"normalize", normLoop, *normXLoad});
    plan.loops.push_back(meanLoop);
    plan.loops.push_back(varLoop);
    plan.loops.push_back(normLoop);
    plan.shape.push_back(rowElements);
    plan.trips = trips;
    plan.bytes = rowBytes;
    plan.elemBytes = elemBytes;
    plan.uses = 3;
    plan.requiredSpmSlots = 1;
    plan.producerPass = ReductionProducerPass::FillOnFirstPass;
    plan.bufferRole = ReductionBufferRole::ResidentRow;
    plan.rotationPolicy = ReductionRotationPolicy::None;
    plan.copyInMode = ReductionCopyInMode::CpuDirect;
    plan.copyIn = "CPU/vector store";
    plan.copyOut = "none";
    plan.overhead =
        "first reduction pass writes each loaded x chunk into SPM; no DMA wait";
    plan.benefit =
        "x[row, :] is materialized once and reused by variance and normalize";
    plan.expectedMarkers.push_back("addrspace(3)");
    plan.expectedMarkers.push_back("no_dma_descriptors");
    plan.expectedMarkers.push_back("no_fence_iorw");
    plan.hasLowering = true;
    applyConfiguredProducerPass(plan, producerPassMode);
    return plan;
  }

  rejectReasonCode = "unsupported_pattern";
  rejectReason = "no LayerNorm-style row-resident candidate matched";
  return std::nullopt;
}

static bool loopContainsTransferWrite(scf::ForOp forOp) {
  bool found = false;
  forOp.getBody()->walk([&](vector::TransferWriteOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static std::optional<ReductionResidencyPlan> matchSoftmaxResidencyPlan(
    ArrayRef<scf::ForOp> loops, FunctionOpInterface funcOp,
    std::string &rejectReasonCode, std::string &rejectReason,
    StringRef producerPassMode) {
  if (loops.size() < 3) {
    rejectReasonCode = "unsupported_pattern";
    rejectReason = "expected three top-level Softmax row loops";
    return std::nullopt;
  }

  for (size_t i = 0; i + 2 < loops.size(); ++i) {
    scf::ForOp maxLoop = loops[i];
    scf::ForOp sumLoop = loops[i + 1];
    scf::ForOp normLoop = loops[i + 2];
    if (!sameStaticLoopShape(maxLoop, sumLoop) ||
        !sameStaticLoopShape(maxLoop, normLoop))
      continue;

    SmallVector<TiledLoadInfo> maxLoads = findTiledLoads(maxLoop);
    SmallVector<TiledLoadInfo> sumLoads = findTiledLoads(sumLoop);
    SmallVector<TiledLoadInfo> normLoads = findTiledLoads(normLoop);
    if (maxLoads.size() != 1 || sumLoads.size() != 1 || normLoads.empty() ||
        !loopContainsTransferWrite(normLoop))
      continue;

    BlockArgument maxArg =
        traceFunctionArgument(maxLoads[0].readOp.getBase(), funcOp);
    BlockArgument sumArg =
        traceFunctionArgument(sumLoads[0].readOp.getBase(), funcOp);
    if (!maxArg || !sumArg || maxArg != sumArg)
      continue;

    std::optional<TiledLoadInfo> normXLoad;
    for (TiledLoadInfo load : normLoads) {
      if (traceFunctionArgument(load.readOp.getBase(), funcOp) == maxArg) {
        normXLoad = load;
        break;
      }
    }
    if (!normXLoad)
      continue;

    bool rowBlockPlan = isRowBlockDmaProducerPassMode(producerPassMode);
    unsigned elemBytes = 0;
    unsigned elemBytesSum = 0;
    unsigned elemBytesNorm = 0;
    bool supportedLoads =
        rowBlockPlan
            ? (isSupportedSoftmaxRowBlockXLoad(maxLoads[0], maxLoop,
                                               elemBytes) &&
               isSupportedSoftmaxRowBlockXLoad(sumLoads[0], sumLoop,
                                               elemBytesSum) &&
               isSupportedSoftmaxRowBlockXLoad(*normXLoad, normLoop,
                                               elemBytesNorm))
            : (isSupportedRowResidentXLoad(maxLoads[0], maxLoop, elemBytes) &&
               isSupportedRowResidentXLoad(sumLoads[0], sumLoop,
                                           elemBytesSum) &&
               isSupportedRowResidentXLoad(*normXLoad, normLoop,
                                           elemBytesNorm));
    if (!supportedLoads || elemBytes != elemBytesSum ||
        elemBytes != elemBytesNorm) {
      rejectReasonCode = "unsupported_pattern";
      rejectReason = rowBlockPlan
                         ? "row-block DMA candidate requires rank-2 "
                           "col-major fp32 x tiles with static "
                           "BLOCK_N-sized column steps"
                         : "candidate requires rank-1 contiguous fp32 x "
                           "loads with static BLOCK_N-sized loop steps";
      return std::nullopt;
    }

    auto lbCst = getConstantIntValue(maxLoop.getLowerBound());
    auto ubCst = getConstantIntValue(maxLoop.getUpperBound());
    auto stepCst = getConstantIntValue(maxLoop.getStep());
    if (!lbCst || !ubCst || !stepCst || *stepCst <= 0 ||
        (*ubCst - *lbCst) % *stepCst != 0) {
      rejectReasonCode = "dynamic_shape_or_stride";
      rejectReason = "loop bounds/step are not static with exact trip count";
      return std::nullopt;
    }

    int64_t trips = (*ubCst - *lbCst) / *stepCst;
    if (trips <= 0) {
      rejectReasonCode = "unsupported_pattern";
      rejectReason = "loop trip count must be positive";
      return std::nullopt;
    }

    int64_t rowBytes = trips * maxLoads[0].tileBytes;
    int64_t rowElements = trips * maxLoads[0].vecTy.getNumElements();
    ReductionResidencyPlan plan;
    plan.source = rowBlockPlan ? "Softmax x row block" : "Softmax x row";
    plan.sourceArg = maxArg;
    plan.producer = ReductionLoopResidencyUse{"max", maxLoop, maxLoads[0]};
    plan.consumers.push_back(
        ReductionLoopResidencyUse{"exp_sum", sumLoop, sumLoads[0]});
    plan.consumers.push_back(
        ReductionLoopResidencyUse{"normalize_store", normLoop, *normXLoad});
    plan.loops.push_back(maxLoop);
    plan.loops.push_back(sumLoop);
    plan.loops.push_back(normLoop);
    if (rowBlockPlan) {
      plan.scope = "program-row-block";
      plan.shape.push_back(trips * maxLoads[0].vecTy.getShape()[0]);
      plan.shape.push_back(maxLoads[0].vecTy.getShape()[1]);
    } else {
      plan.shape.push_back(rowElements);
    }
    plan.trips = trips;
    plan.bytes = rowBytes;
    plan.elemBytes = elemBytes;
    plan.uses = 3;
    plan.requiredSpmSlots = 1;
    plan.producerPass = ReductionProducerPass::FillOnFirstPass;
    plan.bufferRole = ReductionBufferRole::ResidentRow;
    plan.rotationPolicy = ReductionRotationPolicy::None;
    plan.copyInMode = ReductionCopyInMode::CpuDirect;
    plan.copyIn = "CPU/vector store";
    plan.copyOut = "none";
    plan.overhead =
        "first reduction pass writes each loaded x chunk into SPM; no DMA wait";
    plan.benefit = "x[row, :] is materialized once and reused by exp/sum and "
                   "normalize/store";
    plan.expectedMarkers.push_back("addrspace(3)");
    plan.expectedMarkers.push_back("no_dma_descriptors");
    plan.expectedMarkers.push_back("no_fence_iorw");
    plan.hasLowering = true;
    applyConfiguredProducerPass(plan, producerPassMode);
    return plan;
  }

  rejectReasonCode = "unsupported_pattern";
  rejectReason = "no Softmax-style row-resident candidate matched";
  return std::nullopt;
}

static SmallVector<scf::ForOp, 3> collectDirectChildLoops(scf::ForOp outer) {
  SmallVector<scf::ForOp, 3> loops;
  for (Operation &op : outer.getBody()->without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op))
      loops.push_back(forOp);
  }
  return loops;
}

static Value constantVector(OpBuilder &b, Location loc, VectorType ty,
                            double value) {
  auto elemTy = ty.getElementType();
  Attribute elemAttr;
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    elemAttr = b.getFloatAttr(floatTy, value);
  else
    elemAttr = b.getZeroAttr(elemTy);
  return arith::ConstantOp::create(b, loc, ty,
                                   DenseElementsAttr::get(ty, elemAttr));
}

static triton::MakeTensorPtrOp findLoopInitMakeTensorPtr(scf::ForOp forOp) {
  if (forOp.getInitArgs().empty())
    return {};
  return forOp.getInitArgs().front().getDefiningOp<triton::MakeTensorPtrOp>();
}

static std::optional<int64_t> firstStaticDim(ValueRange values) {
  if (values.empty())
    return std::nullopt;
  return getConstantIntValue(values.front());
}

static Value emitTensorPtrTransferRead(OpBuilder &b, Location loc,
                                       Value tensorPtr, MemRefType memRefTy,
                                       VectorType vecTy) {
  Value memRef =
      triton::cpu::ExtractMemRefOp::create(b, loc, memRefTy, tensorPtr);
  auto indices = triton::cpu::ExtractIndicesOp::create(b, loc, tensorPtr);
  if (indices.getNumResults() != static_cast<unsigned>(vecTy.getRank()))
    return {};
  SmallVector<Value> indexVals(indices.getResults().begin(),
                               indices.getResults().end());
  auto padVal = arith::ConstantOp::create(
      b, loc, vecTy.getElementType(), b.getZeroAttr(vecTy.getElementType()));
  return vector::TransferReadOp::create(
      b, loc, vecTy, memRef, indexVals, padVal,
      SmallVector<bool>(vecTy.getRank(), true));
}

static bool emitTensorPtrTransferWrite(OpBuilder &b, Location loc,
                                       Value tensorPtr, MemRefType memRefTy,
                                       Value value) {
  auto vecTy = dyn_cast<VectorType>(value.getType());
  if (!vecTy)
    return false;
  Value memRef =
      triton::cpu::ExtractMemRefOp::create(b, loc, memRefTy, tensorPtr);
  auto indices = triton::cpu::ExtractIndicesOp::create(b, loc, tensorPtr);
  if (indices.getNumResults() != static_cast<unsigned>(vecTy.getRank()))
    return false;
  SmallVector<Value> indexVals(indices.getResults().begin(),
                               indices.getResults().end());
  vector::TransferWriteOp::create(b, loc, value, memRef, indexVals,
                                  SmallVector<bool>(vecTy.getRank(), true));
  return true;
}

static Value emitLeadingDimReduction(OpBuilder &b, Location loc, Value input,
                                     vector::CombiningKind kind) {
  auto vecTy = dyn_cast<VectorType>(input.getType());
  if (!vecTy || vecTy.getRank() != 2)
    return {};

  Value result;
  for (int64_t i = 0; i < vecTy.getShape()[0]; ++i) {
    Value row = vector::ExtractOp::create(b, loc, input, i);
    if (!result) {
      result = row;
      continue;
    }
    switch (kind) {
    case vector::CombiningKind::ADD:
      result = arith::AddFOp::create(b, loc, result, row);
      break;
    case vector::CombiningKind::MAXNUMF:
      result = arith::MaxNumFOp::create(b, loc, result, row);
      break;
    default:
      return {};
    }
  }
  return result;
}

static Value emitTensorPtrAdvance(OpBuilder &b, Location loc, Value ptr,
                                  Value colStep, Value rowStep) {
  SmallVector<Value, 2> offsets{colStep, rowStep};
  return triton::AdvanceOp::create(b, loc, ptr.getType(), ptr, offsets)
      .getResult();
}

static bool eraseContiguousOps(Operation *start, Operation *end) {
  SmallVector<Operation *> eraseOps;
  for (Operation *op = start; op;) {
    Operation *next = op->getNextNode();
    eraseOps.push_back(op);
    if (op == end)
      break;
    op = next;
  }
  if (eraseOps.empty() || eraseOps.back() != end)
    return false;
  for (Operation *op : llvm::reverse(eraseOps))
    op->erase();
  return true;
}

static Operation *setInsertionPointBeforeTerminator(OpBuilder &b, Block *body) {
  Operation *terminator =
      body->mightHaveTerminator() ? body->getTerminator() : nullptr;
  if (terminator)
    b.setInsertionPoint(terminator);
  else
    b.setInsertionPointToEnd(body);
  return terminator;
}

static bool
lowerCanonicalSoftmaxToRowBlockGroup(ArrayRef<scf::ForOp> topLevelLoops,
                                     FunctionOpInterface funcOp,
                                     int64_t rowBlock, int64_t rowGroupBlocks) {
  if (rowBlock <= 1 || rowGroupBlocks <= 0 || topLevelLoops.size() < 3)
    return false;

  std::string rejectCode;
  std::string rejectReason;
  auto plan =
      matchSoftmaxResidencyPlan(topLevelLoops, funcOp, rejectCode, rejectReason,
                                /*producerPassMode=*/"");
  if (!plan || plan->producer.xLoad.vecTy.getRank() != 1)
    return false;

  scf::ForOp maxLoop = plan->producer.forOp;
  scf::ForOp sumLoop = plan->consumers[0].forOp;
  scf::ForOp normLoop = plan->consumers[1].forOp;
  if (maxLoop->getBlock() != sumLoop->getBlock() ||
      maxLoop->getBlock() != normLoop->getBlock())
    return false;

  auto xMakeTensorPtr = findLoopInitMakeTensorPtr(maxLoop);
  if (!xMakeTensorPtr)
    return false;
  auto outMakeTensorPtr =
      normLoop.getInitArgs().size() >= 2
          ? normLoop.getInitArgs()[1].getDefiningOp<triton::MakeTensorPtrOp>()
          : triton::MakeTensorPtrOp();
  if (!outMakeTensorPtr)
    return false;

  auto flatXShape = firstStaticDim(xMakeTensorPtr.getShape());
  auto flatXStride = firstStaticDim(xMakeTensorPtr.getStrides());
  auto flatOutShape = firstStaticDim(outMakeTensorPtr.getShape());
  auto flatOutStride = firstStaticDim(outMakeTensorPtr.getStrides());
  if (!flatXShape || !flatOutShape || flatXShape != flatOutShape ||
      !flatXStride || !flatOutStride || *flatXStride != 1 ||
      *flatOutStride != 1)
    return false;
  if (*flatXShape % plan->shape.front() != 0)
    return false;

  int64_t cols = plan->shape.front();
  int64_t rows = *flatXShape / cols;
  if (rows % (rowBlock * rowGroupBlocks) != 0)
    return false;

  Location loc = maxLoop.getLoc();
  InsertedBeforeGuard guard(maxLoop.getOperation());
  OpBuilder b(maxLoop);

  Value c0I32 = i32Cst(b, loc, 0);
  Value c1I32 = i32Cst(b, loc, 1);
  Value c0RowStepI32 = i32Cst(b, loc, 0);
  Value cRowBlockI32 = i32Cst(b, loc, rowBlock);
  Value cRowGroupBlocksI32 = i32Cst(b, loc, rowGroupBlocks);
  Value cRowsI64 = i64Cst(b, loc, rows);
  Value cColsI64 = i64Cst(b, loc, cols);
  Value c1I64 = i64Cst(b, loc, 1);
  Value pid = triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X)
                  .getResult();
  Value rowsPerProgram = i32Cst(b, loc, rowBlock * rowGroupBlocks);
  Value groupRowBase = arith::MulIOp::create(b, loc, pid, rowsPerProgram);

  auto outer = scf::ForOp::create(b, loc, c0I32, cRowGroupBlocksI32, c1I32);
  Block *outerBody = outer.getBody();
  Operation *outerTerminator = setInsertionPointBeforeTerminator(b, outerBody);

  Value rbOffset =
      arith::MulIOp::create(b, loc, outer.getInductionVar(), cRowBlockI32);
  Value rowBase = arith::AddIOp::create(b, loc, groupRowBase, rbOffset);
  auto oldVecTy = plan->producer.xLoad.vecTy;
  auto elemTy = oldVecTy.getElementType();
  auto rowBlockVecTy =
      VectorType::get({oldVecTy.getShape()[0], rowBlock}, elemTy);
  auto rowVecTy = VectorType::get({rowBlock}, elemTy);
  auto rowBlockMemRefTy =
      MemRefType::get({cols, rows}, elemTy,
                      StridedLayoutAttr::get(b.getContext(), 0, {1, cols}));

  SmallVector<Value> shapeVals{cColsI64, cRowsI64};
  SmallVector<Value> strideVals{c1I64, cColsI64};
  SmallVector<Value> offsets{c0I32, rowBase};
  SmallVector<int32_t> tensorShape{static_cast<int32_t>(oldVecTy.getShape()[0]),
                                   static_cast<int32_t>(rowBlock)};
  SmallVector<int32_t> order{0, 1};
  Value xPtr = triton::MakeTensorPtrOp::create(b, loc, xMakeTensorPtr.getBase(),
                                               shapeVals, strideVals, offsets,
                                               tensorShape, order);
  Value outPtr = triton::MakeTensorPtrOp::create(
      b, loc, outMakeTensorPtr.getBase(), shapeVals, strideVals, offsets,
      tensorShape, order);
  Value maxInit = constantVector(b, loc, rowVecTy, -3.4028234663852886e38);
  Value sumInit = constantVector(b, loc, rowVecTy, 0.0);

  auto maxNew = scf::ForOp::create(b, loc, maxLoop.getLowerBound(),
                                   maxLoop.getUpperBound(), maxLoop.getStep(),
                                   ValueRange{xPtr, maxInit});
  Operation *maxTerminator =
      setInsertionPointBeforeTerminator(b, maxNew.getBody());
  Value maxTile = emitTensorPtrTransferRead(
      b, loc, maxNew.getRegionIterArgs()[0], rowBlockMemRefTy, rowBlockVecTy);
  if (!maxTile) {
    guard.cleanup();
    return false;
  }
  Value tileMax =
      emitLeadingDimReduction(b, loc, maxTile, vector::CombiningKind::MAXNUMF);
  if (!tileMax) {
    guard.cleanup();
    return false;
  }
  Value maxAcc =
      arith::MaxNumFOp::create(b, loc, maxNew.getRegionIterArgs()[1], tileMax);
  Value maxPtr = emitTensorPtrAdvance(b, loc, maxNew.getRegionIterArgs()[0],
                                      maxLoop.getStep(), c0RowStepI32);
  scf::YieldOp::create(b, loc, ValueRange{maxPtr, maxAcc});
  if (maxTerminator)
    maxTerminator->erase();
  b.setInsertionPointAfter(maxNew);

  Value maxBroadcast =
      vector::BroadcastOp::create(b, loc, rowBlockVecTy, maxNew.getResult(1));

  auto sumNew = scf::ForOp::create(b, loc, sumLoop.getLowerBound(),
                                   sumLoop.getUpperBound(), sumLoop.getStep(),
                                   ValueRange{xPtr, sumInit});
  Operation *sumTerminator =
      setInsertionPointBeforeTerminator(b, sumNew.getBody());
  Value sumTile = emitTensorPtrTransferRead(
      b, loc, sumNew.getRegionIterArgs()[0], rowBlockMemRefTy, rowBlockVecTy);
  if (!sumTile) {
    guard.cleanup();
    return false;
  }
  Value centered = arith::SubFOp::create(b, loc, sumTile, maxBroadcast);
  Value expVals = math::ExpOp::create(b, loc, centered);
  Value tileSum =
      emitLeadingDimReduction(b, loc, expVals, vector::CombiningKind::ADD);
  if (!tileSum) {
    guard.cleanup();
    return false;
  }
  Value sumAcc =
      arith::AddFOp::create(b, loc, sumNew.getRegionIterArgs()[1], tileSum);
  Value sumPtr = emitTensorPtrAdvance(b, loc, sumNew.getRegionIterArgs()[0],
                                      sumLoop.getStep(), c0RowStepI32);
  scf::YieldOp::create(b, loc, ValueRange{sumPtr, sumAcc});
  if (sumTerminator)
    sumTerminator->erase();
  b.setInsertionPointAfter(sumNew);

  Value denomBroadcast =
      vector::BroadcastOp::create(b, loc, rowBlockVecTy, sumNew.getResult(1));
  auto normNew = scf::ForOp::create(
      b, loc, normLoop.getLowerBound(), normLoop.getUpperBound(),
      normLoop.getStep(), ValueRange{xPtr, outPtr});
  Operation *normTerminator =
      setInsertionPointBeforeTerminator(b, normNew.getBody());
  Value normTile = emitTensorPtrTransferRead(
      b, loc, normNew.getRegionIterArgs()[0], rowBlockMemRefTy, rowBlockVecTy);
  if (!normTile) {
    guard.cleanup();
    return false;
  }
  Value normCentered = arith::SubFOp::create(b, loc, normTile, maxBroadcast);
  Value normExp = math::ExpOp::create(b, loc, normCentered);
  Value normalized = arith::DivFOp::create(b, loc, normExp, denomBroadcast);
  if (!emitTensorPtrTransferWrite(b, loc, normNew.getRegionIterArgs()[1],
                                  rowBlockMemRefTy, normalized)) {
    guard.cleanup();
    return false;
  }
  Value normXPtr = emitTensorPtrAdvance(b, loc, normNew.getRegionIterArgs()[0],
                                        normLoop.getStep(), c0RowStepI32);
  Value normOutPtr = emitTensorPtrAdvance(
      b, loc, normNew.getRegionIterArgs()[1], normLoop.getStep(), c0RowStepI32);
  scf::YieldOp::create(b, loc, ValueRange{normXPtr, normOutPtr});
  if (normTerminator)
    normTerminator->erase();
  b.setInsertionPointAfter(normNew);
  if (!outerTerminator)
    scf::YieldOp::create(b, loc);

  if (!eraseContiguousOps(maxLoop.getOperation(), normLoop.getOperation())) {
    guard.cleanup();
    return false;
  }

  guard.commit();
  return true;
}

static void rejectUnsupportedRowBlockGroup(std::string &rejectReasonCode,
                                           std::string &rejectReason) {
  rejectReasonCode = "unsupported_pattern";
  rejectReason =
      "row-block DMA group candidate requires one outer row-block loop "
      "containing max/sum/normalize Softmax loops";
}

static std::optional<ReductionResidencyPlan>
matchSoftmaxRowBlockGroupResidencyPlan(ArrayRef<scf::ForOp> topLevelLoops,
                                       FunctionOpInterface funcOp,
                                       std::string &rejectReasonCode,
                                       std::string &rejectReason,
                                       StringRef producerPassMode) {
  if (!isRowBlockDmaProducerPassMode(producerPassMode))
    return std::nullopt;

  for (scf::ForOp outerLoop : topLevelLoops) {
    SmallVector<scf::ForOp, 3> childLoops = collectDirectChildLoops(outerLoop);
    if (childLoops.size() < 3)
      continue;

    std::string childRejectCode;
    std::string childRejectReason;
    auto plan = matchSoftmaxResidencyPlan(childLoops, funcOp, childRejectCode,
                                          childRejectReason, producerPassMode);
    if (!plan)
      continue;

    auto lbCst = getConstantIntValue(outerLoop.getLowerBound());
    auto ubCst = getConstantIntValue(outerLoop.getUpperBound());
    auto stepCst = getConstantIntValue(outerLoop.getStep());
    if (!lbCst || !ubCst || !stepCst || *stepCst <= 0 ||
        (*ubCst - *lbCst) % *stepCst != 0) {
      rejectReasonCode = "dynamic_shape_or_stride";
      rejectReason = "outer row-block loop bounds/step are not static with "
                     "exact trip count";
      return std::nullopt;
    }

    int64_t outerTrips = (*ubCst - *lbCst) / *stepCst;
    if (outerTrips <= 1) {
      rejectReasonCode = "unsupported_pattern";
      rejectReason =
          "outer row-block loop needs at least two trips for A/B DMA "
          "double-buffering";
      return std::nullopt;
    }

    plan->rowBlockGroupLoop = outerLoop;
    plan->rowBlockGroupTrips = outerTrips;
    plan->scope = "program-row-block-group";
    plan->requiredSpmSlots = 2;
    plan->overhead =
        "row-block group schedule waits for the current DMA-filled row block "
        "and prefetches the next row block into the alternate SPM slot while "
        "max/sum/normalize compute consumes the current slot";
    plan->benefit =
        "x[row_block, :] is DMA-filled once per row block and reused by all "
        "Softmax passes while the next row block is in flight";
    return plan;
  }

  rejectUnsupportedRowBlockGroup(rejectReasonCode, rejectReason);
  return std::nullopt;
}

static SPMProfitabilityEvidence
evaluateD3RowResidentProfitability(const ReductionResidencyPlan &plan,
                                   int64_t spmSize) {
  int64_t rowResidentDescriptors = 0;
  int64_t rowResidentWaits = 0;
  int64_t spmConsumerPasses = std::max<int64_t>(0, plan.uses - 1);
  if (plan.producerPass == ReductionProducerPass::ProducerStore)
    spmConsumerPasses =
        std::max<int64_t>(0, static_cast<int64_t>(plan.consumers.size()) - 1);
  int64_t groupTrips = std::max<int64_t>(1, plan.rowBlockGroupTrips);
  int64_t copyBytes = plan.copyInMode == ReductionCopyInMode::Dma
                          ? plan.bytes * groupTrips
                          : plan.bytes;
  int64_t spmWriteBytes = plan.bytes;
  int64_t spmReadBytes = plan.bytes * spmConsumerPasses;
  int64_t liveSpmBytes = plan.bytes * plan.requiredSpmSlots;
  int64_t estimatedExtraOps = plan.trips * (1 + spmConsumerPasses);
  int64_t measuredBankConflicts = 0;
  if (plan.bufferRole == ReductionBufferRole::ResidentRowBlock) {
    rowResidentDescriptors = groupTrips;
    rowResidentWaits = groupTrips;
    spmWriteBytes = copyBytes;
    spmReadBytes = plan.bytes * plan.uses * groupTrips;
    estimatedExtraOps = groupTrips * (plan.trips * plan.uses + 1);
  }
  int64_t avoidedBytes = plan.bytes * spmConsumerPasses * groupTrips;
  int64_t rowElements = plan.shape.empty() ? 0 : plan.shape.front();

  if (liveSpmBytes > spmSize) {
    return makeD3ProfitabilityEvidence(
        "reject", "spm_capacity_overflow",
        "P3 static model rejects reduction residency because the required "
        "resident SPM slots exceed SPM capacity",
        rowResidentDescriptors, rowResidentWaits, copyBytes, spmWriteBytes,
        spmReadBytes, avoidedBytes, liveSpmBytes, estimatedExtraOps,
        measuredBankConflicts, plan.uses);
  }

  if (rowElements < 512) {
    return makeD3ProfitabilityEvidence(
        "reject", "small_row_spm_overhead",
        "P3 static model rejects small rows because measured Phase 3.5 "
        "evidence shows SPM store/read overhead is not reliably amortized "
        "against the best legal cache schedule",
        rowResidentDescriptors, rowResidentWaits, copyBytes, spmWriteBytes,
        spmReadBytes, avoidedBytes, liveSpmBytes, estimatedExtraOps,
        measuredBankConflicts, plan.uses);
  }

  if (plan.producerPass == ReductionProducerPass::DmaPrefetch &&
      plan.bufferRole != ReductionBufferRole::ResidentRowBlock) {
    return makeD3ProfitabilityEvidence(
        "reject", "chunk_dma_spm_overhead",
        "P3 static model rejects chunk-DMA row residency because Phase 3.5 "
        "measurements show per-chunk descriptors, waits, and fences are worse "
        "than the available CPU-direct or row-block schedules",
        rowResidentDescriptors, rowResidentWaits, copyBytes, spmWriteBytes,
        spmReadBytes, avoidedBytes, liveSpmBytes, estimatedExtraOps,
        measuredBankConflicts, plan.uses);
  }

  if (plan.producerPass == ReductionProducerPass::ProducerStore) {
    return makeD3ProfitabilityEvidence(
        "reject", "producer_store_spm_overhead",
        "P3 static model rejects producer-store row residency because "
        "Phase 3.5 measurements show the extra cache-path pass is not "
        "competitive with fill-on-first-pass against the best legal cache "
        "schedule",
        rowResidentDescriptors, rowResidentWaits, copyBytes, spmWriteBytes,
        spmReadBytes, avoidedBytes, liveSpmBytes, estimatedExtraOps,
        measuredBankConflicts, plan.uses);
  }

  if (plan.bufferRole == ReductionBufferRole::ResidentRowBlock) {
    return makeD3ProfitabilityEvidence(
        "accept", "accepted_block_resident_fill_first",
        "P3 static model accepts Softmax block residency only for the "
        "measured row-block DMA schedule: bounded double-buffered row-block "
        "lifetime, coarse DMA copies, zero measured bank conflicts, and "
        "comparison against the best legal cache baseline",
        rowResidentDescriptors, rowResidentWaits, copyBytes, spmWriteBytes,
        spmReadBytes, avoidedBytes, liveSpmBytes, estimatedExtraOps,
        measuredBankConflicts, plan.uses);
  }

  return makeD3ProfitabilityEvidence(
      "accept", "accepted_row_resident_fill_first",
      "P3 static model accepts fill-on-first-pass row residency as opt-in "
      "evidence for large rows: no DMA descriptors or waits, bounded row "
      "lifetime, and measured Phase 3.5 data is at least near parity or better "
      "against the best legal cache baseline",
      rowResidentDescriptors, rowResidentWaits, copyBytes, spmWriteBytes,
      spmReadBytes, avoidedBytes, liveSpmBytes, estimatedExtraOps,
      measuredBankConflicts, plan.uses);
}

static SPMProfitabilityEvidence evaluateD3StreamingReductionProfitability(
    scf::ForOp forOp, ArrayRef<TiledLoadInfo> loads, int64_t spmSize) {
  auto lbCst = getConstantIntValue(forOp.getLowerBound());
  auto ubCst = getConstantIntValue(forOp.getUpperBound());
  auto stepCst = getConstantIntValue(forOp.getStep());

  int64_t trips = 0;
  if (lbCst && ubCst && stepCst && *stepCst > 0 &&
      (*ubCst - *lbCst) % *stepCst == 0)
    trips = (*ubCst - *lbCst) / *stepCst;
  int64_t descriptors = std::max<int64_t>(0, trips) * loads.size();
  int64_t copyBytes = 0;
  for (const TiledLoadInfo &load : loads)
    copyBytes += std::max<int64_t>(0, trips) * load.tileBytes;

  return makeD3ProfitabilityEvidence(
      "reject", "streaming_reduction_no_residency",
      "P3 static model rejects streaming reductions by default because each "
      "chunk has one compute use and no bounded SPM residency; keep the cache "
      "path unless an explicit row/block-resident schedule is selected",
      descriptors, descriptors, copyBytes,
      /*spmWriteBytes=*/copyBytes, /*spmReadBytes=*/copyBytes,
      /*avoidedRepeatedReadBytes=*/0, std::min(copyBytes, spmSize),
      /*estimatedExtraOps=*/descriptors,
      /*measuredBankConflicts=*/0, /*uses=*/1);
}

static SPMPromotionRejection
makeD3RowResidentProfitabilityRejection(const ReductionResidencyPlan &plan,
                                        SPMProfitabilityEvidence evidence) {
  SPMPromotionRejection rejection = makeReductionResidencyRejection(
      plan, evidence.reasonCode, evidence.reason);
  attachD3Profitability(rejection, std::move(evidence));
  return rejection;
}

static SPMPromotionRejection
makeD3StreamingReductionRejection(scf::ForOp forOp,
                                  ArrayRef<TiledLoadInfo> loads,
                                  SPMProfitabilityEvidence evidence) {
  SPMPromotionRejection rejection = makePromotionRejection(
      "reduction_streaming", evidence.reasonCode, evidence.reason);
  rejection.uses = 1;
  rejection.copyIn = "DMA";
  rejection.copyOut = "none";
  rejection.bytes = evidence.copyBytes;
  if (!loads.empty())
    appendShape(rejection.shape, loads.front().vecTy);
  attachD3Profitability(rejection, std::move(evidence));
  (void)forOp;
  return rejection;
}

static scf::ForOp cloneLoopWithRowResidentX(ReductionLoopResidencyUse loopInfo,
                                            int64_t rowSpmAddress,
                                            unsigned elemBytes,
                                            bool fillSpmFromOriginalRead,
                                            bool useRowBlockDmaLayout = false) {
  scf::ForOp forOp = loopInfo.forOp;
  vector::TransferReadOp xRead = loopInfo.xLoad.readOp;
  Location loc = forOp.getLoc();
  OpBuilder b(forOp);
  auto memRefTy = dyn_cast<MemRefType>(xRead.getBase().getType());
  bool rowBlockLayout =
      useRowBlockDmaLayout && memRefTy &&
      hasColMajorRowBlockDmaLayout(loopInfo.xLoad.vecTy, memRefTy);
  SmallVector<int64_t, 2> spmMemStrides =
      rowBlockLayout
          ? getRowBlockDmaSpmMemStrides(loopInfo.xLoad.vecTy, memRefTy)
          : getDefaultSpmMemStrides(loopInfo.xLoad.vecTy);
  int64_t spmStepBytes = elemBytes;
  if (rowBlockLayout)
    spmStepBytes *= loopInfo.xLoad.vecTy.getShape()[1];

  auto newForOp =
      scf::ForOp::create(b, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                         forOp.getStep(), forOp.getInitArgs());

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (auto [oldArg, newArg] :
       llvm::zip_equal(forOp.getRegionIterArgs(), newForOp.getRegionIterArgs()))
    mapping.map(oldArg, newArg);

  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();
  b.setInsertionPointToStart(newBody);

  Value elemOffset =
      arith::SubIOp::create(b, loc, toI64(b, loc, newForOp.getInductionVar()),
                            toI64(b, loc, newForOp.getLowerBound()));
  Value byteOffset =
      arith::MulIOp::create(b, loc, elemOffset, i64Cst(b, loc, spmStepBytes));
  Value xSpmAddr =
      arith::AddIOp::create(b, loc, i64Cst(b, loc, rowSpmAddress), byteOffset);

  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;

    if (&op == xRead.getOperation()) {
      if (fillSpmFromOriginalRead) {
        Operation *cloned = b.clone(op, mapping);
        auto clonedRead = cast<vector::TransferReadOp>(cloned);
        emitSpmWriteWithStrides(b, loc, xSpmAddr, clonedRead.getResult(),
                                spmMemStrides);
        mapping.map(xRead.getResult(), clonedRead.getResult());
        continue;
      }

      Value spmVal = emitSpmReadWithStrides(
          b, loc, xSpmAddr, loopInfo.xLoad.vecTy, spmMemStrides);
      mapping.map(xRead.getResult(), spmVal);
      continue;
    }

    b.clone(op, mapping);
  }

  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (Value val : oldYield.getOperands())
    yieldVals.push_back(mapping.lookupOrDefault(val));
  scf::YieldOp::create(b, loc, yieldVals);

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

static bool lowerDmaPrefetchRowResidentX(ReductionLoopResidencyUse loopInfo,
                                         int64_t rowSpmAddress,
                                         unsigned elemBytes,
                                         bool useRowBlockDmaLayout = false) {
  scf::ForOp forOp = loopInfo.forOp;
  vector::TransferReadOp xRead = loopInfo.xLoad.readOp;
  auto memRefTy = dyn_cast<MemRefType>(xRead.getBase().getType());
  if (!memRefTy)
    return false;
  bool rowBlockLayout =
      useRowBlockDmaLayout &&
      hasColMajorRowBlockDmaLayout(loopInfo.xLoad.vecTy, memRefTy);
  SmallVector<int64_t, 2> spmMemStrides =
      rowBlockLayout
          ? getRowBlockDmaSpmMemStrides(loopInfo.xLoad.vecTy, memRefTy)
          : getDefaultSpmMemStrides(loopInfo.xLoad.vecTy);
  int64_t spmStepBytes = elemBytes;
  if (rowBlockLayout)
    spmStepBytes *= loopInfo.xLoad.vecTy.getShape()[1];

  auto stepBytes = getLoopStepBytes(xRead, forOp, /*requireLoopIv=*/true);
  if (!stepBytes)
    return false;

  Location loc = forOp.getLoc();
  InsertedBeforeGuard guard(forOp.getOperation());
  OpBuilder b(forOp);

  Value dramAddr = computePrologueDramAddr(b, loc, xRead, forOp);
  if (!dramAddr) {
    guard.cleanup();
    return false;
  }

  if (rowBlockLayout)
    emitRowBlockDmaEnqueue(b, loc, i64Cst(b, loc, rowSpmAddress), dramAddr,
                           loopInfo.xLoad.vecTy, memRefTy);
  else
    emitDmaEnqueue(b, loc, i64Cst(b, loc, rowSpmAddress), dramAddr,
                   loopInfo.xLoad.vecTy, memRefTy);

  auto newForOp =
      scf::ForOp::create(b, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                         forOp.getStep(), forOp.getInitArgs());

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  unsigned numOldArgs = forOp.getRegionIterArgs().size();
  for (unsigned i = 0; i < numOldArgs; ++i)
    mapping.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);

  b.setInsertionPointToStart(newBody);
  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  triton::cpu::DmaWaitOp::create(b, loc);

  Value iv = newForOp.getInductionVar();
  Value step = newForOp.getStep();
  Value ub = newForOp.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  Value lbInLoop = newForOp.getLowerBound();
  Value currentOff = arith::SubIOp::create(b, loc, iv, lbInLoop);
  Value currentOffI64 = toI64(b, loc, currentOff);
  Value currentByteOff = arith::MulIOp::create(b, loc, currentOffI64,
                                               i64Cst(b, loc, spmStepBytes));
  Value residentSpmAddr = arith::AddIOp::create(
      b, loc, i64Cst(b, loc, rowSpmAddress), currentByteOff);

  Value nextOff = arith::SubIOp::create(b, loc, nextIv, lbInLoop);
  Value nextOffI64 = toI64(b, loc, nextOff);
  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value nextByteOff =
      arith::MulIOp::create(b, loc, nextOffI64, i64Cst(b, loc, *stepBytes));
  Value nextDram = arith::AddIOp::create(b, loc, dramAddr, nextByteOff);
  Value nextSpmAddr = arith::AddIOp::create(
      b, loc, i64Cst(b, loc, rowSpmAddress),
      arith::MulIOp::create(b, loc, nextOffI64, i64Cst(b, loc, spmStepBytes)));
  if (rowBlockLayout)
    emitRowBlockDmaEnqueue(b, loc, nextSpmAddr, nextDram, loopInfo.xLoad.vecTy,
                           memRefTy);
  else
    emitDmaEnqueue(b, loc, nextSpmAddr, nextDram, loopInfo.xLoad.vecTy,
                   memRefTy);
  b.setInsertionPointAfter(ifOp);

  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;

    if (&op == xRead.getOperation()) {
      Value spmVal = emitSpmReadWithStrides(
          b, loc, residentSpmAddr, loopInfo.xLoad.vecTy, spmMemStrides);
      mapping.map(xRead.getResult(), spmVal);
      continue;
    }

    b.clone(op, mapping);
  }

  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (Value val : oldYield.getOperands())
    yieldVals.push_back(mapping.lookupOrDefault(val));
  scf::YieldOp::create(b, loc, yieldVals);

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  guard.commit();
  forOp.erase();
  return true;
}

static Value mapOrDefault(IRMapping &mapping, Value value) {
  if (mapping.contains(value))
    return mapping.lookup(value);
  return value;
}

static bool cloneLoopBodyWithRowBlockSpm(
    OpBuilder &b, Location loc, scf::ForOp oldLoop, scf::ForOp newLoop,
    TiledLoadInfo xLoad, Value rowSpmBase, unsigned elemBytes,
    int64_t spmStepBytes, ArrayRef<int64_t> spmMemStrides, IRMapping &mapping,
    Value expSpmBase = nullptr, bool writeExpToSpm = false,
    bool readExpFromSpm = false) {
  Block *oldBody = oldLoop.getBody();
  Block *newBody = newLoop.getBody();
  mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
  for (auto [oldArg, newArg] : llvm::zip_equal(oldLoop.getRegionIterArgs(),
                                               newLoop.getRegionIterArgs()))
    mapping.map(oldArg, newArg);

  b.setInsertionPointToStart(newBody);
  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  Value elemOffset =
      arith::SubIOp::create(b, loc, toI64(b, loc, newLoop.getInductionVar()),
                            toI64(b, loc, newLoop.getLowerBound()));
  Value byteOffset =
      arith::MulIOp::create(b, loc, elemOffset, i64Cst(b, loc, spmStepBytes));
  Value xSpmAddr = arith::AddIOp::create(b, loc, rowSpmBase, byteOffset);

  Value expSpmAddr;
  if (expSpmBase && (writeExpToSpm || readExpFromSpm))
    expSpmAddr = arith::AddIOp::create(b, loc, expSpmBase, byteOffset);

  // When readExpFromSpm is set, identify the sub+exp chain after the x read
  // so we can skip them and use the exp buffer value directly.
  Operation *subOp = nullptr;
  Operation *expOp = nullptr;
  if (readExpFromSpm) {
    for (auto &op : oldBody->getOperations()) {
      if (&op == xLoad.readOp.getOperation())
        continue;
      if (isa<arith::SubFOp>(op) &&
          op.getOperand(0) == xLoad.readOp.getResult()) {
        subOp = &op;
        continue;
      }
      if (subOp && isa<math::ExpOp>(op) &&
          op.getOperand(0) == subOp->getResult(0)) {
        expOp = &op;
        break;
      }
    }
  }

  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;

    if (&op == xLoad.readOp.getOperation()) {
      if (readExpFromSpm && expSpmAddr) {
        // Skip the x read entirely; the exp value comes from SPM.
        mapping.map(xLoad.readOp.getResult(), xSpmAddr); // placeholder
        continue;
      }
      Value spmVal =
          emitSpmReadWithStrides(b, loc, xSpmAddr, xLoad.vecTy, spmMemStrides);
      mapping.map(xLoad.readOp.getResult(), spmVal);
      continue;
    }

    if (readExpFromSpm && subOp && &op == subOp) {
      // Skip sub; it's part of the replaced exp(x - max) chain.
      continue;
    }

    if (readExpFromSpm && expOp && &op == expOp) {
      // Replace exp(x - max) with a read from the exp SPM buffer.
      Value expVal = emitSpmReadWithStrides(b, loc, expSpmAddr, xLoad.vecTy,
                                            spmMemStrides);
      mapping.map(expOp->getResult(0), expVal);
      continue;
    }

    Operation *cloned = b.clone(op, mapping);

    if (writeExpToSpm && expSpmAddr && isa<math::ExpOp>(op)) {
      // Write the exp result to the exp SPM buffer.
      Value expResult = cloned->getResult(0);
      emitSpmWriteWithStrides(b, loc, expSpmAddr, expResult, spmMemStrides);
    }
  }

  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (Value val : oldYield.getOperands())
    yieldVals.push_back(mapOrDefault(mapping, val));
  scf::YieldOp::create(b, loc, yieldVals);

  return true;
}

static bool emitFullRowBlockDma(OpBuilder &b, Location loc, Value rowSpmAddr,
                                Value firstDramAddr, VectorType vecTy,
                                MemRefType memRefTy, int64_t trips) {
  if (!hasColMajorRowBlockDmaLayout(vecTy, memRefTy) || trips <= 0)
    return false;

  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t columns = trips * vecTy.getShape()[0];
  int64_t rows = vecTy.getShape()[1];
  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides) || strides.size() < 2)
    return false;

  triton::cpu::DmaEnqueue2DOp::create(
      b, loc, rowSpmAddr, firstDramAddr, i64Cst(b, loc, columns * elemBytes),
      i64Cst(b, loc, rows), i64Cst(b, loc, strides[1] * elemBytes),
      i64Cst(b, loc, columns * elemBytes));
  return true;
}

static Operation *cloneBlockPrefixThrough(OpBuilder &b, Operation *target,
                                          IRMapping &mapping) {
  Block *block = target->getBlock();
  Operation *clonedTarget = nullptr;
  for (Operation &op : block->without_terminator()) {
    Operation *cloned = b.clone(op, mapping);
    if (&op == target) {
      clonedTarget = cloned;
      break;
    }
  }
  return clonedTarget;
}

static bool lowerSoftmaxRowBlockGroupDma(
    ReductionResidencyPlan &plan, int64_t rowSpmAddress0,
    int64_t rowSpmAddress1,
    llvm::DenseSet<Operation *> &rowResidentHandledLoops,
    int64_t expSpmAddress = -1) {
  scf::ForOp outerLoop = plan.rowBlockGroupLoop;
  if (!outerLoop)
    return false;

  vector::TransferReadOp xRead = plan.producer.xLoad.readOp;
  auto memRefTy = dyn_cast<MemRefType>(xRead.getBase().getType());
  if (!memRefTy ||
      !hasColMajorRowBlockDmaLayout(plan.producer.xLoad.vecTy, memRefTy))
    return false;

  auto makeTensorPtr =
      plan.producer.forOp
          .getTiedLoopInit(plan.producer.forOp.getRegionIterArgs().front())
          ->get()
          .template getDefiningOp<triton::MakeTensorPtrOp>();
  if (!makeTensorPtr)
    return false;

  Location loc = outerLoop.getLoc();
  InsertedBeforeGuard guard(outerLoop.getOperation());
  OpBuilder b(outerLoop);

  IRMapping firstPtrMapping;
  firstPtrMapping.map(outerLoop.getInductionVar(), outerLoop.getLowerBound());
  for (auto [iterArg, initArg] :
       llvm::zip_equal(outerLoop.getRegionIterArgs(), outerLoop.getInitArgs()))
    firstPtrMapping.map(iterArg, initArg);
  Operation *firstTensorPtrOp =
      cloneBlockPrefixThrough(b, makeTensorPtr.getOperation(), firstPtrMapping);
  if (!firstTensorPtrOp) {
    guard.cleanup();
    return false;
  }
  Value firstDramAddr = computeTensorPtrDramAddr(
      b, loc, firstTensorPtrOp->getResult(0), memRefTy);
  if (!firstDramAddr) {
    guard.cleanup();
    return false;
  }
  if (!emitFullRowBlockDma(b, loc, i64Cst(b, loc, rowSpmAddress0),
                           firstDramAddr, plan.producer.xLoad.vecTy, memRefTy,
                           plan.trips)) {
    guard.cleanup();
    return false;
  }

  SmallVector<Value> outerInitArgs(outerLoop.getInitArgs());
  outerInitArgs.push_back(i64Cst(b, loc, 0));
  auto newOuter = scf::ForOp::create(b, loc, outerLoop.getLowerBound(),
                                     outerLoop.getUpperBound(),
                                     outerLoop.getStep(), outerInitArgs);

  Block *oldOuterBody = outerLoop.getBody();
  Block *newOuterBody = newOuter.getBody();
  IRMapping outerMapping;
  outerMapping.map(outerLoop.getInductionVar(), newOuter.getInductionVar());
  unsigned oldOuterArgs = outerLoop.getRegionIterArgs().size();
  for (unsigned i = 0; i < oldOuterArgs; ++i)
    outerMapping.map(outerLoop.getRegionIterArgs()[i],
                     newOuter.getRegionIterArgs()[i]);
  Value bufIdx = newOuter.getRegionIterArgs()[oldOuterArgs];

  b.setInsertionPointToStart(newOuterBody);
  if (!newOuterBody->empty() && newOuterBody->mightHaveTerminator())
    newOuterBody->getTerminator()->erase();

  triton::cpu::DmaWaitOp::create(b, loc);

  Value zero = i64Cst(b, loc, 0);
  Value isZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bufIdx, zero);
  Value spmBuf0 = i64Cst(b, loc, rowSpmAddress0);
  Value spmBuf1 = i64Cst(b, loc, rowSpmAddress1);
  Value currentSpm = arith::SelectOp::create(b, loc, isZero, spmBuf0, spmBuf1);
  Value nextSpm = arith::SelectOp::create(b, loc, isZero, spmBuf1, spmBuf0);

  Value iv = newOuter.getInductionVar();
  Value step = newOuter.getStep();
  Value ub = newOuter.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  IRMapping nextPtrMapping;
  nextPtrMapping.map(outerLoop.getInductionVar(), nextIv);
  for (auto [iterArg, newIterArg] :
       llvm::zip_equal(outerLoop.getRegionIterArgs(),
                       newOuter.getRegionIterArgs().take_front(oldOuterArgs)))
    nextPtrMapping.map(iterArg, newIterArg);
  Operation *nextTensorPtrOp =
      cloneBlockPrefixThrough(b, makeTensorPtr.getOperation(), nextPtrMapping);
  if (!nextTensorPtrOp) {
    guard.cleanup();
    return false;
  }
  Value nextDramAddr =
      computeTensorPtrDramAddr(b, loc, nextTensorPtrOp->getResult(0), memRefTy);
  if (!nextDramAddr) {
    guard.cleanup();
    return false;
  }
  if (!emitFullRowBlockDma(b, loc, nextSpm, nextDramAddr,
                           plan.producer.xLoad.vecTy, memRefTy, plan.trips)) {
    guard.cleanup();
    return false;
  }
  b.setInsertionPointAfter(ifOp);

  SmallVector<int64_t, 2> spmMemStrides{
      1, plan.trips * plan.producer.xLoad.vecTy.getShape()[0]};
  int64_t spmStepBytes = plan.elemBytes;

  bool cacheExp = expSpmAddress >= 0;
  Value expSpmBase;
  if (cacheExp)
    expSpmBase = i64Cst(b, loc, expSpmAddress);

  for (Operation &op : oldOuterBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      std::optional<ReductionLoopResidencyUse> loopUse;
      bool isExpSumLoop = false;
      bool isNormLoop = false;
      if (forOp == plan.producer.forOp)
        loopUse = plan.producer;
      else {
        for (unsigned i = 0; i < plan.consumers.size(); ++i) {
          if (forOp == plan.consumers[i].forOp) {
            loopUse = plan.consumers[i];
            if (i == 0)
              isExpSumLoop = true;
            else if (i == 1)
              isNormLoop = true;
            break;
          }
        }
      }

      if (loopUse) {
        SmallVector<Value> initArgs;
        initArgs.reserve(forOp.getInitArgs().size());
        for (Value initArg : forOp.getInitArgs())
          initArgs.push_back(mapOrDefault(outerMapping, initArg));
        auto newLoop = scf::ForOp::create(
            b, loc, mapOrDefault(outerMapping, forOp.getLowerBound()),
            mapOrDefault(outerMapping, forOp.getUpperBound()),
            mapOrDefault(outerMapping, forOp.getStep()), initArgs);
        cloneLoopBodyWithRowBlockSpm(b, loc, forOp, newLoop, loopUse->xLoad,
                                     currentSpm, plan.elemBytes, spmStepBytes,
                                     spmMemStrides, outerMapping,
                                     cacheExp ? expSpmBase : nullptr,
                                     /*writeExpToSpm=*/cacheExp && isExpSumLoop,
                                     /*readExpFromSpm=*/cacheExp && isNormLoop);
        b.setInsertionPointAfter(newLoop);
        for (auto [oldResult, newResult] :
             llvm::zip_equal(forOp.getResults(), newLoop.getResults()))
          outerMapping.map(oldResult, newResult);
        continue;
      }
    }

    b.clone(op, outerMapping);
  }

  auto oldYield = cast<scf::YieldOp>(oldOuterBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (Value val : oldYield.getOperands())
    yieldVals.push_back(mapOrDefault(outerMapping, val));
  Value one = i64Cst(b, loc, 1);
  Value flipped = arith::SubIOp::create(b, loc, one, bufIdx);
  yieldVals.push_back(flipped);
  scf::YieldOp::create(b, loc, yieldVals);

  for (unsigned i = 0; i < outerLoop.getNumResults(); ++i)
    outerLoop.getResult(i).replaceAllUsesWith(newOuter.getResult(i));
  rowResidentHandledLoops.insert(newOuter.getOperation());
  rowResidentHandledLoops.insert(outerLoop.getOperation());
  for (scf::ForOp loop : plan.loops)
    rowResidentHandledLoops.insert(loop.getOperation());

  guard.commit();
  outerLoop.erase();
  return true;
}

static void markResidencyPlanLoopsHandled(
    const ReductionResidencyPlan &plan,
    llvm::DenseSet<Operation *> &rowResidentHandledLoops) {
  if (scf::ForOp groupLoop = plan.rowBlockGroupLoop)
    rowResidentHandledLoops.insert(groupLoop.getOperation());
  for (scf::ForOp forOp : plan.loops)
    rowResidentHandledLoops.insert(forOp.getOperation());
}

static bool transformReductionResidencyPlan(
    ReductionResidencyPlan &plan, int64_t spmBase, int64_t spmSize,
    int64_t rowResidentMaxBytes, bool enablePromotionProfitability,
    SPMPromotionReport *report,
    llvm::DenseSet<Operation *> &rowResidentHandledLoops) {
  bool useSoftmaxExpCache =
      getEnvBool("TRITON_SPM_SOFTMAX_CACHE_EXP", false) &&
      plan.bufferRole == ReductionBufferRole::ResidentRowBlock &&
      plan.source == "Softmax x row block";
  if (useSoftmaxExpCache) {
    plan.requiredSpmSlots = std::max<int64_t>(plan.requiredSpmSlots, 3);
    plan.overhead +=
        "; exp-cache adds one resident SPM buffer for exp(x - max) values";
    plan.benefit +=
        "; normalize/store reads cached exp values instead of recomputing exp";
  }

  auto reject = [&](StringRef reasonCode, StringRef reason) {
    if (report)
      report->rejections.push_back(
          makeReductionResidencyRejection(plan, reasonCode, reason));
    markResidencyPlanLoopsHandled(plan, rowResidentHandledLoops);
    return false;
  };

  if (!plan.hasLowering)
    return reject("unsupported_reduction_residency_plan",
                  "row/block-resident reduction plan was detected, but no "
                  "lowering template exists yet; leave cache-path IR");
  if (rowResidentMaxBytes <= 0)
    return reject("unsupported_config",
                  "row-resident max bytes must be positive");
  if (plan.bytes > rowResidentMaxBytes)
    return reject("spm_capacity_overflow",
                  "row bytes exceed the D2 row-resident prototype budget");
  if (enablePromotionProfitability) {
    SPMProfitabilityEvidence evidence =
        evaluateD3RowResidentProfitability(plan, spmSize);
    if (evidence.decision != "accept") {
      if (report)
        report->rejections.push_back(
            makeD3RowResidentProfitabilityRejection(plan, std::move(evidence)));
      markResidencyPlanLoopsHandled(plan, rowResidentHandledLoops);
      return false;
    }
  }

  SPMSpaceManager spmLayout(spmBase, spmSize);
  auto allocRow = spmLayout.alloc(plan.bytes, /*alignment=*/1,
                                  SPMSpaceManager::Lifetime::Loop);
  if (!allocRow)
    return reject("spm_capacity_overflow",
                  "SPM capacity cannot fit the resident x row schedule");

  bool useRowBlockDmaLayout =
      plan.producerPass == ReductionProducerPass::DmaPrefetch &&
      plan.bufferRole == ReductionBufferRole::ResidentRowBlock;

  if (plan.rowBlockGroupLoop) {
    if (!useRowBlockDmaLayout)
      return reject("unsupported_reduction_residency_plan",
                    "row-block group residency currently requires "
                    "row-block DMA producer mode");
    auto allocRow1 = spmLayout.alloc(plan.bytes, /*alignment=*/1,
                                     SPMSpaceManager::Lifetime::Loop);
    if (!allocRow1)
      return reject("spm_capacity_overflow",
                    "SPM capacity cannot fit both row-block DMA buffers");
    int64_t expSpmAddress = -1;
    if (useSoftmaxExpCache) {
      auto allocExp = spmLayout.alloc(plan.bytes, /*alignment=*/1,
                                      SPMSpaceManager::Lifetime::Loop);
      if (!allocExp)
        return reject("spm_capacity_overflow",
                      "SPM capacity cannot fit the row-block exp-cache buffer");
      expSpmAddress = allocExp->address;
    }
    if (!lowerSoftmaxRowBlockGroupDma(plan, allocRow->address,
                                      allocRow1->address,
                                      rowResidentHandledLoops, expSpmAddress))
      return reject("unsupported_reduction_residency_plan",
                    "row-block group DMA double-buffer lowering failed");
  } else if (plan.producerPass == ReductionProducerPass::FillOnFirstPass) {
    scf::ForOp newProducer = cloneLoopWithRowResidentX(
        plan.producer, allocRow->address, plan.elemBytes,
        /*fillSpmFromOriginalRead=*/true);
    rowResidentHandledLoops.insert(newProducer.getOperation());
    for (const ReductionLoopResidencyUse &consumer : plan.consumers) {
      scf::ForOp newConsumer =
          cloneLoopWithRowResidentX(consumer, allocRow->address, plan.elemBytes,
                                    /*fillSpmFromOriginalRead=*/false);
      rowResidentHandledLoops.insert(newConsumer.getOperation());
    }
  } else if (plan.producerPass == ReductionProducerPass::ProducerStore) {
    if (plan.consumers.empty())
      return reject("unsupported_pattern",
                    "producer-store row residency needs a later x consumer");
    rowResidentHandledLoops.insert(plan.producer.forOp.getOperation());
    scf::ForOp newMaterializer = cloneLoopWithRowResidentX(
        plan.consumers.front(), allocRow->address, plan.elemBytes,
        /*fillSpmFromOriginalRead=*/true);
    rowResidentHandledLoops.insert(newMaterializer.getOperation());
    for (const ReductionLoopResidencyUse &consumer :
         ArrayRef<ReductionLoopResidencyUse>(plan.consumers).drop_front()) {
      scf::ForOp newConsumer =
          cloneLoopWithRowResidentX(consumer, allocRow->address, plan.elemBytes,
                                    /*fillSpmFromOriginalRead=*/false);
      rowResidentHandledLoops.insert(newConsumer.getOperation());
    }
  } else if (plan.producerPass == ReductionProducerPass::DmaPrefetch) {
    if (!lowerDmaPrefetchRowResidentX(plan.producer, allocRow->address,
                                      plan.elemBytes, useRowBlockDmaLayout))
      return reject("unsupported_reduction_residency_plan",
                    "DMA-prefetch row residency could not lower the producer "
                    "loop");
    rowResidentHandledLoops.insert(plan.producer.forOp.getOperation());
    for (const ReductionLoopResidencyUse &consumer : plan.consumers) {
      scf::ForOp newConsumer = cloneLoopWithRowResidentX(
          consumer, allocRow->address, plan.elemBytes,
          /*fillSpmFromOriginalRead=*/false, useRowBlockDmaLayout);
      rowResidentHandledLoops.insert(newConsumer.getOperation());
    }
  } else {
    return reject("unsupported_reduction_residency_plan",
                  "DMA row/block-resident lowering is not implemented yet");
  }

  if (report) {
    SPMPromotionRecord record =
        makeReductionResidencyRecord(plan, allocRow->address);
    if (enablePromotionProfitability) {
      SPMProfitabilityEvidence evidence =
          evaluateD3RowResidentProfitability(plan, spmSize);
      attachD3Profitability(record, std::move(evidence));
    }
    report->records.push_back(std::move(record));
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Attention v2 Q-resident transformation.
//
// Attention has an unusually useful loop-invariant Q tile and loop-local K/V
// streams.  The old K/V SPM window schedule reduced some cache misses but made
// LLVM lower many more scalar SPM loads and inflated the loop body.  This path
// deliberately stages only Q, which has real reuse across both stage loops, and
// leaves K/V on the cache path.
//===----------------------------------------------------------------------===//

struct AttentionOuterQPlan {
  vector::TransferReadOp readOp;
  VectorType vecTy;
  MemRefType memRefTy;
  SmallVector<int64_t, 2> spmMemStrides;
  int64_t tileBytes = 0;
  int64_t addr = 0;
};

struct AttentionStreamPlan {
  TiledLoadInfo load;
  MemRefType memRefTy;
  SmallVector<int64_t, 2> spmMemStrides;
  int64_t stepBytes = 0;
  int64_t addrBuf0 = 0;
  int64_t addrBuf1 = 0;
  Value dramAddr;
  Value spmCur;
  Value spmNxt;
};

struct AttentionResidentQState {
  int64_t addr = 0;
  int64_t tileBytes = 0;
};

struct AttentionFunctionState {
  int64_t spmBase = 0;
  int64_t spmSize = 0;
  int64_t persistentBytes = 0;
  DenseMap<Operation *, AttentionResidentQState> residentQ;

  void initialize(int64_t base, int64_t size) {
    if (spmSize != 0)
      return;
    spmBase = base;
    spmSize = size;
  }

  static std::optional<int64_t> alignUp(int64_t value, int64_t alignment) {
    if (alignment <= 1)
      return value;
    int64_t remainder = value % alignment;
    if (remainder == 0)
      return value;
    if (value > std::numeric_limits<int64_t>::max() - (alignment - remainder))
      return std::nullopt;
    return value + (alignment - remainder);
  }

  std::optional<int64_t> nextPersistentTop(int64_t bytes,
                                           int64_t alignment = 1) const {
    auto aligned = alignUp(persistentBytes, alignment);
    if (!aligned)
      return std::nullopt;
    if (bytes < 0 || *aligned > spmSize || bytes > spmSize - *aligned)
      return std::nullopt;
    return *aligned + bytes;
  }
};

static bool contractConsumesRead(vector::ContractionOp contract,
                                 vector::TransferReadOp readOp) {
  for (Value operand : {contract.getLhs(), contract.getRhs()}) {
    auto operandRead = getTransferReadThroughShapePreservingCasts(operand);
    if (operandRead == readOp)
      return true;
  }
  return false;
}

static bool isOneOfReads(vector::TransferReadOp readOp,
                         ArrayRef<TiledLoadInfo> reads) {
  for (const TiledLoadInfo &read : reads)
    if (read.readOp == readOp)
      return true;
  return false;
}

static bool shouldTryAttentionV2QResident(scf::ForOp forOp,
                                          ArrayRef<TiledLoadInfo> dotLoads) {
  if (dotLoads.size() != 2)
    return false;

  // GEMM still owns the case where the two loop-local loads feed the same
  // contraction.  Attention has two contractions: QK and P/V.
  if (analyzeGemmContract(forOp, dotLoads[0].readOp, dotLoads[1].readOp) ||
      analyzeGemmContract(forOp, dotLoads[1].readOp, dotLoads[0].readOp))
    return false;

  return true;
}

static std::optional<AttentionOuterQPlan>
findAttentionOuterQPlan(scf::ForOp forOp, ArrayRef<TiledLoadInfo> streams,
                        AttentionFunctionState &state) {
  std::optional<AttentionOuterQPlan> result;
  forOp.getBody()->walk([&](vector::ContractionOp contract) {
    if (result || !isGemmContract(contract))
      return WalkResult::advance();

    bool consumesStream = false;
    for (const TiledLoadInfo &stream : streams) {
      if (contractConsumesRead(contract, stream.readOp)) {
        consumesStream = true;
        break;
      }
    }
    if (!consumesStream)
      return WalkResult::advance();

    for (Value operand : {contract.getLhs(), contract.getRhs()}) {
      auto readOp = getTransferReadThroughShapePreservingCasts(operand);
      if (!readOp || isOneOfReads(readOp, streams))
        continue;
      if (forOp.getBodyRegion().isAncestor(readOp->getParentRegion()))
        continue;
      if (readOp->getBlock() != forOp->getBlock() ||
          !readOp->isBeforeInBlock(forOp))
        continue;

      auto vecTy = dyn_cast<VectorType>(readOp.getType());
      auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
      if (!vecTy || vecTy.getRank() != 2 || !memRefTy ||
          memRefTy.getRank() != 2)
        continue;

      if (memRefTy.getMemorySpaceAsInt() == SPM_ADDR_SPACE) {
        auto existing = state.residentQ.find(readOp.getOperation());
        if (existing == state.residentQ.end())
          continue;
        AttentionOuterQPlan plan;
        plan.readOp = readOp;
        plan.vecTy = vecTy;
        plan.memRefTy = memRefTy;
        plan.spmMemStrides = getDefaultSpmMemStrides(vecTy);
        plan.tileBytes = getTileBytes(vecTy);
        plan.addr = existing->second.addr;
        result = std::move(plan);
        return WalkResult::interrupt();
      }

      SmallVector<int64_t> strides;
      if (!getStaticStrides(memRefTy, strides))
        continue;

      AttentionOuterQPlan plan;
      plan.readOp = readOp;
      plan.vecTy = vecTy;
      plan.memRefTy = memRefTy;
      plan.spmMemStrides = getDmaFilledSpmMemStrides(vecTy, memRefTy);
      plan.tileBytes = getTileBytes(vecTy);
      result = std::move(plan);
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return result;
}

static void eraseDeadReadIfUnused(vector::TransferReadOp readOp) {
  if (readOp && readOp.getResult().use_empty())
    readOp.erase();
}

static std::optional<int64_t>
getAttentionStreamStepBytes(vector::TransferReadOp readOp, scf::ForOp forOp) {
  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  auto vecTy = dyn_cast<VectorType>(readOp.getType());
  if (!memRefTy || !vecTy || vecTy.getRank() != 2)
    return std::nullopt;

  if (auto stepBytes = getLoopStepBytes(readOp, forOp,
                                        /*requireLoopIv=*/true)) {
    auto stepCst = getConstantIntValue(forOp.getStep());
    if (!stepCst || *stepCst <= 0)
      return std::nullopt;
    return *stepBytes * *stepCst;
  }

  BlockArgument blockPtrArg;
  auto *baseDefOp = readOp.getBase().getDefiningOp();
  if (!baseDefOp || baseDefOp->getParentRegion() != &forOp.getRegion())
    return std::nullopt;

  auto extractMR = dyn_cast<triton::cpu::ExtractMemRefOp>(baseDefOp);
  if (!extractMR)
    return std::nullopt;

  auto arg = dyn_cast<BlockArgument>(extractMR.getSrc());
  if (!arg || arg.getOwner() != forOp.getBody() || arg.getArgNumber() == 0)
    return std::nullopt;
  unsigned iterArgIdx = arg.getArgNumber() - 1;
  if (iterArgIdx >= forOp.getRegionIterArgs().size())
    return std::nullopt;
  blockPtrArg = arg;

  OpOperand *yielded = forOp.getTiedLoopYieldedValue(blockPtrArg);
  if (!yielded)
    return std::nullopt;

  auto advance = yielded->get().getDefiningOp<triton::AdvanceOp>();
  if (!advance || advance.getPtr() != blockPtrArg)
    return std::nullopt;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return std::nullopt;
  auto offsets = advance.getOffsets();
  if (offsets.size() != strides.size())
    return std::nullopt;

  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t stepElems = 0;
  bool sawNonZero = false;
  for (auto [offset, stride] : llvm::zip_equal(offsets, strides)) {
    auto offsetValue = getConstantIntValue(offset);
    if (!offsetValue)
      return std::nullopt;
    if (*offsetValue != 0)
      sawNonZero = true;
    stepElems += *offsetValue * stride;
  }
  if (!sawNonZero || stepElems <= 0)
    return std::nullopt;
  return stepElems * elemBytes;
}

static bool materializeAttentionQ(AttentionOuterQPlan &qPlan,
                                  AttentionFunctionState &state) {
  if (qPlan.memRefTy.getMemorySpaceAsInt() == SPM_ADDR_SPACE)
    return true;

  auto nextTop = state.nextPersistentTop(qPlan.tileBytes);
  if (!nextTop)
    return false;

  auto alignedTop = AttentionFunctionState::alignUp(state.persistentBytes, 1);
  if (!alignedTop)
    return false;
  int64_t qAddr = state.spmBase + *alignedTop;
  Location loc = qPlan.readOp.getLoc();
  InsertedBeforeGuard qGuard(qPlan.readOp.getOperation());
  OpBuilder qBuilder(qPlan.readOp);
  Value qDram = computeDramAddr(qBuilder, loc, qPlan.readOp);
  if (!qDram) {
    qGuard.cleanup();
    return false;
  }

  emitDmaFilledEnqueue(qBuilder, loc, i64Cst(qBuilder, loc, qAddr), qDram,
                       qPlan.vecTy, qPlan.memRefTy);
  triton::cpu::DmaWaitOp::create(qBuilder, loc);
  Value qSpm =
      emitSpmReadWithStrides(qBuilder, loc, i64Cst(qBuilder, loc, qAddr),
                             qPlan.vecTy, qPlan.spmMemStrides);
  qPlan.readOp.getResult().replaceAllUsesWith(qSpm);
  eraseDeadReadIfUnused(qPlan.readOp);
  qGuard.commit();

  if (auto spmRead = qSpm.getDefiningOp<vector::TransferReadOp>()) {
    state.residentQ[spmRead.getOperation()] =
        AttentionResidentQState{qAddr, qPlan.tileBytes};
    qPlan.readOp = spmRead;
    qPlan.memRefTy = cast<MemRefType>(spmRead.getBase().getType());
    qPlan.spmMemStrides = getDefaultSpmMemStrides(qPlan.vecTy);
  }
  qPlan.addr = qAddr;
  state.persistentBytes = *nextTop;
  return true;
}

static int64_t getAttentionQReuseCount(vector::TransferReadOp qRead) {
  int64_t uses = 0;
  for (Operation *user : qRead.getResult().getUsers()) {
    if (isa<vector::ContractionOp>(user)) {
      ++uses;
      continue;
    }
    if (!isShapePreservingCastLikeOp(user))
      continue;
    for (Operation *castUser : user->getResult(0).getUsers())
      if (isa<vector::ContractionOp>(castUser))
        ++uses;
  }
  return std::max<int64_t>(1, uses);
}

static void appendAttentionV2QRecord(SPMPromotionReport *report,
                                     StringRef source, StringRef scope,
                                     VectorType shapeTy, int64_t uses,
                                     int64_t bytes, int64_t spmAddress,
                                     int64_t liveBytes, int64_t dmaDescriptors,
                                     int64_t waits, StringRef bufferRole,
                                     StringRef rotationPolicy,
                                     StringRef reasonCode, StringRef reason) {
  if (!report)
    return;

  SPMPromotionRecord record;
  record.source = source.str();
  record.scope = scope.str();
  appendShape(record.shape, shapeTy);
  record.uses = uses;
  record.copyIn = "DMA";
  record.copyOut = "none";
  record.bytes = bytes;
  record.spmAddress = spmAddress;
  record.overhead = "one Q DMA and one wait before the attention loop";
  record.benefit = "keeps the loop-invariant Q tile resident across QK loops";
  record.reasonCode = reasonCode.str();
  record.reason = reason.str();
  record.residencyPlan.present = true;
  record.residencyPlan.producerPass = "dma_prefetch";
  record.residencyPlan.consumerPasses.push_back("attention_v2_q_resident");
  record.residencyPlan.bufferRole = bufferRole.str();
  record.residencyPlan.rotationPolicy = rotationPolicy.str();
  record.residencyPlan.copyInMode = "dma";
  record.residencyPlan.requiredSpmSlots =
      std::max<int64_t>(1, bytes / getTileBytes(shapeTy));
  record.residencyPlan.expectedMarkers.push_back("addrspace(3)");
  record.residencyPlan.expectedMarkers.push_back("dma_descriptors");
  record.residencyPlan.expectedMarkers.push_back("fence_iorw");
  record.profitability = makeD3ProfitabilityEvidence(
      "accept", reasonCode, reason, dmaDescriptors, waits,
      /*copyBytes=*/bytes, /*spmWriteBytes=*/bytes,
      /*spmReadBytes=*/getTileBytes(shapeTy) * uses,
      /*avoidedRepeatedReadBytes=*/getTileBytes(shapeTy) *
          std::max<int64_t>(0, uses - 1),
      liveBytes, /*estimatedExtraOps=*/uses,
      /*measuredBankConflicts=*/0, uses);
  report->records.push_back(std::move(record));
}

static bool transformAttentionV2QResidentLoop(scf::ForOp forOp,
                                              ArrayRef<TiledLoadInfo> dotLoads,
                                              AttentionFunctionState &state,
                                              SPMPromotionReport *report) {
  auto reject = [&](StringRef reasonCode, StringRef reason) {
    if (report)
      report->rejections.push_back(
          makePromotionRejection("attention_v2_q_resident", reasonCode,
                                 reason));
    return false;
  };

  if (!shouldTryAttentionV2QResident(forOp, dotLoads))
    return false;

  auto trips = getExactStaticTripCount(forOp);
  if (trips && *trips <= 0)
    return reject("dynamic_shape_or_stride",
                  "attention-v2 requires a positive trip count");

  std::optional<AttentionOuterQPlan> qPlan =
      findAttentionOuterQPlan(forOp, dotLoads, state);
  if (!qPlan)
    return reject("unsupported_pattern",
                  "attention-v2 requires an outer loop-invariant Q tile");

  bool qAlreadyResident =
      qPlan->memRefTy.getMemorySpaceAsInt() == SPM_ADDR_SPACE;
  if (qAlreadyResident)
    return true;

  int64_t qReuseCount = getAttentionQReuseCount(qPlan->readOp);
  std::optional<bool> qResidentPolicy =
      getEnvBoolOverride("TRITON_SPM_ATTENTION_Q_RESIDENT");
  if (qResidentPolicy && !*qResidentPolicy)
    return reject("policy_disabled",
                  "attention Q residency is disabled by policy");

  bool forceQResident = qResidentPolicy.value_or(false);
  if (!forceQResident) {
    if (!trips)
      return reject("dynamic_trip_count_not_profitable",
                    "auto attention Q residency requires a known static trip "
                    "count; use TRITON_SPM_ATTENTION_Q_RESIDENT=1 to force");
    int64_t minTrips =
        getEnvInt64("TRITON_SPM_ATTENTION_Q_RESIDENT_MIN_TRIPS", 4);
    if (minTrips > 1 && *trips < minTrips)
      return reject("short_attention_loop_not_profitable",
                    "short attention loops do not amortize Q DMA/wait cost");
    int64_t minUses =
        getEnvInt64("TRITON_SPM_ATTENTION_Q_RESIDENT_MIN_USES", 2);
    if (minUses > 1 && qReuseCount < minUses)
      return reject("low_q_reuse_not_profitable",
                    "Q is already loaded outside the attention loop; single "
                    "QK consumers do not amortize DMA/wait cost");
  }

  int64_t maxQTileBytes =
      getEnvInt64("TRITON_SPM_ATTENTION_Q_RESIDENT_MAX_BYTES", 2048);
  if (maxQTileBytes >= 0 && qPlan->tileBytes > maxQTileBytes)
    return reject("large_q_tile_not_safe",
                  "large attention Q tiles are kept cacheable because wide "
                  "addrspace(3) vector loads are not correctness-stable");
  auto nextTop = state.nextPersistentTop(qPlan->tileBytes);
  if (!nextTop)
    return reject("spm_capacity_overflow",
                  "SPM capacity cannot fit attention-v2 resident Q tile");

  int64_t liveBytes = qPlan->tileBytes;

  if (!materializeAttentionQ(*qPlan, state))
    return reject("dynamic_shape_or_stride",
                  "failed to compute resident Q tile DRAM address");

  appendAttentionV2QRecord(
      report, "attention Q resident tile", "function-scope attention tile",
      qPlan->vecTy, qReuseCount, qPlan->tileBytes, qPlan->addr, liveBytes,
      /*dmaDescriptors=*/1, /*waits=*/1, "resident_q_tile", "none",
      "accepted_attention_v2_q_resident",
      "Q is staged once and reused by attention QK loops while K/V remain "
      "cacheable");

  return true;
}

static bool transformAttentionV2KVStreamingLoop(
    scf::ForOp forOp, ArrayRef<TiledLoadInfo> dotLoads,
    AttentionFunctionState &state, SPMPromotionReport *report) {
  auto reject = [&](StringRef reasonCode, StringRef reason) {
    if (report)
      report->rejections.push_back(
          makePromotionRejection("attention_v2_kv_stream", reasonCode,
                                 reason));
    return false;
  };

  if (!getEnvBool("TRITON_SPM_ATTENTION_KV_STREAM", false))
    return reject("policy_disabled",
                  "attention K/V streaming is disabled by default");

  if (!shouldTryAttentionV2QResident(forOp, dotLoads))
    return false;

  auto trips = getExactStaticTripCount(forOp);
  if (trips && *trips <= 0)
    return reject("dynamic_shape_or_stride",
                  "attention-v2 requires a positive trip count");
  if (trips && *trips == 1)
    return reject("single_iteration_not_profitable",
                  "single-iteration attention K/V streaming is not useful");

  std::optional<AttentionOuterQPlan> qPlan =
      findAttentionOuterQPlan(forOp, dotLoads, state);
  if (!qPlan)
    return reject("unsupported_pattern",
                  "attention-v2 requires an outer loop-invariant Q tile");

  bool qAlreadyResident =
      qPlan->memRefTy.getMemorySpaceAsInt() == SPM_ADDR_SPACE;
  bool stageQ = qAlreadyResident ||
                getEnvBool("TRITON_SPM_ATTENTION_KV_STREAM_STAGE_Q", false);
  if (stageQ && !qAlreadyResident) {
    int64_t maxQTileBytes =
        getEnvInt64("TRITON_SPM_ATTENTION_Q_RESIDENT_MAX_BYTES", 2048);
    if (maxQTileBytes >= 0 && qPlan->tileBytes > maxQTileBytes)
      return reject("large_q_tile_not_safe",
                    "large attention Q tiles are kept cacheable because wide "
                    "addrspace(3) vector loads are not correctness-stable");
    if (!state.nextPersistentTop(qPlan->tileBytes))
      return reject("spm_capacity_overflow",
                    "SPM capacity cannot fit attention-v2 resident Q tile");
  }

  int64_t persistentTop =
      stageQ ? (qAlreadyResident ? state.persistentBytes
                                 : *state.nextPersistentTop(qPlan->tileBytes))
             : state.persistentBytes;

  SmallVector<AttentionStreamPlan, 2> streams;
  streams.reserve(dotLoads.size());
  for (const TiledLoadInfo &load : dotLoads) {
    vector::TransferReadOp readOp = load.readOp;
    auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
    if (!memRefTy || memRefTy.getRank() != 2 || load.vecTy.getRank() != 2)
      return reject("unsupported_pattern",
                    "attention K/V streams must be rank-2 tiled loads");
    if (!canComputePrologueDramAddr(readOp, forOp))
      return reject("dynamic_shape_or_stride",
                    "failed to compute attention stream prologue address");
    auto stepBytes = getAttentionStreamStepBytes(readOp, forOp);
    if (!stepBytes)
      return reject("dynamic_shape_or_stride",
                    "failed to compute attention stream step bytes");

    AttentionStreamPlan plan;
    plan.load = load;
    plan.memRefTy = memRefTy;
    plan.spmMemStrides = getDmaFilledSpmMemStrides(load.vecTy, memRefTy);
    plan.stepBytes = *stepBytes;
    streams.push_back(std::move(plan));
  }

  SPMSpaceManager spmLayout(state.spmBase + persistentTop,
                            state.spmSize - persistentTop);
  int64_t liveBytes = stageQ ? qPlan->tileBytes : 0;
  for (AttentionStreamPlan &stream : streams) {
    auto alloc0 = spmLayout.alloc(stream.load.tileBytes, /*alignment=*/1,
                                  SPMSpaceManager::Lifetime::Loop);
    auto alloc1 = spmLayout.alloc(stream.load.tileBytes, /*alignment=*/1,
                                  SPMSpaceManager::Lifetime::Loop);
    if (!alloc0 || !alloc1)
      return reject("spm_capacity_overflow",
                    "SPM capacity cannot fit double-buffered K/V tiles");
    stream.addrBuf0 = alloc0->address;
    stream.addrBuf1 = alloc1->address;
    liveBytes += stream.load.tileBytes * 2;
  }

  int64_t qReuseCount = getAttentionQReuseCount(qPlan->readOp);
  if (stageQ && !materializeAttentionQ(*qPlan, state))
    return reject("dynamic_shape_or_stride",
                  "failed to compute resident Q tile DRAM address");

  Location loc = forOp.getLoc();
  InsertedBeforeGuard guard(forOp.getOperation());
  OpBuilder b(forOp);

  for (AttentionStreamPlan &stream : streams) {
    stream.dramAddr = computePrologueDramAddr(b, loc, stream.load.readOp, forOp);
    if (!stream.dramAddr) {
      guard.cleanup();
      return reject("dynamic_shape_or_stride",
                    "failed to compute attention stream DRAM address");
    }
    emitDmaFilledEnqueue(b, loc, i64Cst(b, loc, stream.addrBuf0),
                         stream.dramAddr, stream.load.vecTy, stream.memRefTy);
  }

  SmallVector<Value> initArgs(forOp.getInitArgs());
  initArgs.push_back(i64Cst(b, loc, 0));
  auto newForOp =
      scf::ForOp::create(b, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                         forOp.getStep(), initArgs);

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();
  b.setInsertionPointToStart(newBody);
  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  unsigned numOldArgs = forOp.getRegionIterArgs().size();
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < numOldArgs; ++i)
    mapping.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);

  triton::cpu::DmaWaitOp::create(b, loc);

  Value bufIdx = newForOp.getRegionIterArgs()[numOldArgs];
  Value zero = i64Cst(b, loc, 0);
  Value one = i64Cst(b, loc, 1);
  Value isZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bufIdx, zero);
  for (AttentionStreamPlan &stream : streams) {
    Value buf0 = i64Cst(b, loc, stream.addrBuf0);
    Value buf1 = i64Cst(b, loc, stream.addrBuf1);
    stream.spmCur = arith::SelectOp::create(b, loc, isZero, buf0, buf1);
    stream.spmNxt = arith::SelectOp::create(b, loc, isZero, buf1, buf0);
  }

  Value iv = newForOp.getInductionVar();
  Value nextIv = arith::AddIOp::create(b, loc, iv, newForOp.getStep());
  Value hasNext = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::slt, nextIv, newForOp.getUpperBound());
  Value nextOff = arith::SubIOp::create(b, loc, nextIv,
                                        newForOp.getLowerBound());
  Value nextIterNum =
      arith::DivSIOp::create(b, loc, toI64(b, loc, nextOff),
                             toI64(b, loc, newForOp.getStep()));

  auto prefetchIf =
      scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&prefetchIf.getThenRegion().front());
  for (AttentionStreamPlan &stream : streams) {
    Value nextByteOff = arith::MulIOp::create(
        b, loc, nextIterNum, i64Cst(b, loc, stream.stepBytes));
    Value nextDram = arith::AddIOp::create(b, loc, stream.dramAddr,
                                           nextByteOff);
    emitDmaFilledEnqueue(b, loc, stream.spmNxt, nextDram,
                         stream.load.vecTy, stream.memRefTy);
  }
  b.setInsertionPointAfter(prefetchIf);

  for (Operation &op : oldBody->without_terminator()) {
    bool replacedRead = false;
    for (AttentionStreamPlan &stream : streams) {
      if (&op != stream.load.readOp.getOperation())
        continue;
      Value spmVal = emitSpmReadWithStrides(
          b, loc, stream.spmCur, stream.load.vecTy, stream.spmMemStrides);
      mapping.map(stream.load.readOp.getResult(), spmVal);
      replacedRead = true;
      break;
    }
    if (replacedRead)
      continue;
    b.clone(op, mapping);
  }

  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (Value operand : oldYield.getOperands())
    yieldVals.push_back(mapping.lookupOrDefault(operand));
  Value flipped = arith::SubIOp::create(b, loc, one, bufIdx);
  yieldVals.push_back(flipped);
  scf::YieldOp::create(b, loc, yieldVals);

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  guard.commit();
  forOp.erase();

  if (stageQ && !qAlreadyResident)
    appendAttentionV2QRecord(
        report, "attention Q resident tile", "function-scope attention tile",
        qPlan->vecTy, qReuseCount, qPlan->tileBytes, qPlan->addr, liveBytes,
        /*dmaDescriptors=*/1, /*waits=*/1, "resident_q_tile", "none",
        "accepted_attention_v2_q_resident",
        "Q is staged once and reused by attention QK loops");

  int64_t recordUses = trips ? *trips : 2;
  for (const AttentionStreamPlan &stream : streams) {
    std::string source =
        hasColMajorRowBlockDmaLayout(stream.load.vecTy, stream.memRefTy)
            ? "attention K stream tile"
            : "attention V stream tile";
    appendAttentionV2QRecord(
        report, source, "double-buffered attention K/V stream",
        stream.load.vecTy, recordUses, stream.load.tileBytes * 2,
        stream.addrBuf0, liveBytes,
        /*dmaDescriptors=*/recordUses, /*waits=*/recordUses,
        "stream_kv_tile", "double_buffered",
        "accepted_attention_v2_kv_stream",
        "K/V tiles are streamed through SPM without unrolling the attention "
        "loop");
  }

  return true;
}

//===----------------------------------------------------------------------===//
// GEMM double-buffering transformation.
//
// Input pattern (K-loop):
//   scf.for %k = 0 to K step BK iter_args(%acc, ...) {
//     %a = vector.transfer_read %memA[...] : memref<MxK>, vector<BM×BK>
//     %b = vector.transfer_read %memB[...] : memref<KxN>, vector<BK×BN>
//     %c = vector.contract(%a, %b, %acc)
//     yield %c, ...
//   }
//
// Output pattern:
//   // Prologue: DMA first tiles into buffer 0
//   dma_enqueue_2d(spm_a0, dram_a_start, ...)
//   dma_enqueue_2d(spm_b0, dram_b_start, ...)
//   dma_wait
//
//   scf.for %k iter_args(%buf_idx=0, %acc, ...) {
//     // Select current SPM buffer
//     %spm_a_cur = select(%buf_idx==0, spm_a0, spm_a1)
//     %spm_b_cur = select(%buf_idx==0, spm_b0, spm_b1)
//
//     // Prefetch NEXT tiles into alternate buffer (async)
//     scf.if (%k + BK < K) {
//       %spm_a_nxt = select(%buf_idx==0, spm_a1, spm_a0)
//       dma_enqueue_2d(%spm_a_nxt, dram_a_next, ...)
//       dma_enqueue_2d(%spm_b_nxt, dram_b_next, ...)
//     }
//
//     // Compute from SPM
//     %a = transfer_read spm_a_cur : memref<BM×BK, addr_space=3>
//     %b = transfer_read spm_b_cur : memref<BK×BN, addr_space=3>
//     %c = vector.contract(%a, %b, %acc)
//
//     dma_wait  // wait before swapping
//     yield (1 - %buf_idx), %c, ...
//   }
//===----------------------------------------------------------------------===//

/// Transform a GEMM K-loop into a fused microM-aware SPM schedule.
///
/// This path keeps B resident for a small K window and only DMA-loads the
/// microM rows of A that the current contract consumes.  The full accumulator
/// tile is spilled to SPM between K windows so each inner loop carries only a
/// microM x N vector accumulator.
static bool transformFusedMicroGemmLoop(
    scf::ForOp forOp, ArrayRef<TiledLoadInfo> dotLoads, int64_t spmBase,
    int64_t spmSize, int64_t microM, int64_t requestedWindowK,
    bool enablePromotionProfitability, SPMPromotionReport *report) {
  auto reject = [&](StringRef reasonCode, StringRef reason) {
    if (report)
      report->rejections.push_back(
          makePromotionRejection("fused_micro_gemm", reasonCode, reason));
    return false;
  };

  if (dotLoads.size() != 2)
    return reject("unsupported_pattern",
                  "expected exactly two dot-feeding tiled loads");
  if (microM <= 0)
    return reject("unsupported_config", "microM must be positive");
  if (requestedWindowK <= 0)
    return reject("unsupported_config", "requested windowK must be positive");

  auto makeRecord = [](StringRef source, StringRef scope, VectorType shapeTy,
                       int64_t uses, StringRef copyIn, StringRef copyOut,
                       int64_t bytes, int64_t spmAddress, StringRef overhead,
                       StringRef benefit) {
    SPMPromotionRecord record;
    record.source = source.str();
    record.scope = scope.str();
    appendShape(record.shape, shapeTy);
    record.uses = uses;
    record.copyIn = copyIn.str();
    record.copyOut = copyOut.str();
    record.bytes = bytes;
    record.spmAddress = spmAddress;
    record.overhead = overhead.str();
    record.benefit = benefit.str();
    return record;
  };

  auto makeBWindowRecord = [&](VectorType shapeTy, int64_t window, int64_t uses,
                               int64_t bytes, int64_t spmAddress) {
    SPMPromotionRecord record;
    record.source = "B tile window";
    record.scope = "loop-window";
    appendShape(record.shape, shapeTy);
    record.shape.push_back(window);
    record.uses = uses;
    record.copyIn = "DMA";
    record.copyOut = "none";
    record.bytes = bytes;
    record.spmAddress = spmAddress;
    record.overhead = "windowK DMA descriptors plus one wait per K window";
    record.benefit =
        "B window is loaded once and reused across all microM slices";
    if (enablePromotionProfitability)
      attachD3Profitability(
          record,
          makeD3ProfitabilityEvidence(
              "accept", "accepted_reused_loop_window",
              "D3 static model accepts the existing fused matmul B window: "
              "bounded loop-window lifetime and reuse across microM slices",
              window, /*waits=*/1, bytes,
              /*spmWriteBytes=*/bytes, /*spmReadBytes=*/bytes,
              (bytes / window) * std::max<int64_t>(0, uses - window), bytes,
              /*estimatedExtraOps=*/window,
              /*measuredBankConflicts=*/0, uses));
    return record;
  };

  auto makeAccumulatorRecord = [&](VectorType shapeTy, int64_t uses,
                                   int64_t bytes, int64_t spmAddress) {
    SPMPromotionRecord record;
    record.source = "accumulator tile";
    record.scope = "loop-window temporary";
    appendShape(record.shape, shapeTy);
    record.uses = uses;
    record.copyIn = "CPU/vector store";
    record.copyOut = "CPU/vector transfer read";
    record.bytes = bytes;
    record.spmAddress = spmAddress;
    record.overhead = "SPM read/write around each microM slice";
    record.benefit =
        "keeps the full accumulator tile resident while inner loops carry "
        "only microM rows";
    if (enablePromotionProfitability)
      attachD3Profitability(
          record,
          makeD3ProfitabilityEvidence(
              "accept", "accepted_bounded_temporary",
              "D3 static model accepts the accumulator tile as a bounded "
              "loop-window temporary in the existing fused matmul schedule",
              /*dmaDescriptors=*/0, /*waits=*/0, /*copyBytes=*/0,
              /*spmWriteBytes=*/bytes * std::max<int64_t>(0, uses - 1),
              /*spmReadBytes=*/bytes * std::max<int64_t>(0, uses - 1),
              bytes * std::max<int64_t>(0, uses - 1), bytes,
              /*estimatedExtraOps=*/uses,
              /*measuredBankConflicts=*/0, uses));
    return record;
  };

  auto makeAMicroRecord = [&](VectorType shapeTy, int64_t uses, int64_t bytes,
                              int64_t spmAddress) {
    return makeRecord("A micro tile", "single-iteration", shapeTy, uses, "DMA",
                      "none", bytes, spmAddress,
                      "two SPM buffers, one DMA descriptor per microM/K "
                      "step, and one wait-at-top per step",
                      "pipelines A micro-tile staging while limiting A to "
                      "the rows consumed by the current microM contract");
  };

  auto lbCst = getConstantIntValue(forOp.getLowerBound());
  auto ubCst = getConstantIntValue(forOp.getUpperBound());
  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!lbCst || !ubCst || !stepCst || *stepCst <= 0 ||
      (*ubCst - *lbCst) % *stepCst != 0)
    return reject("dynamic_shape_or_stride",
                  "loop bounds/step are not static with exact trip count");

  int64_t trips = (*ubCst - *lbCst) / *stepCst;
  if (trips <= 0)
    return reject("unsupported_pattern", "loop trip count must be positive");
  int64_t windowK = chooseWindowK(trips, requestedWindowK);

  TiledLoadInfo loadA = dotLoads[0];
  TiledLoadInfo loadB = dotLoads[1];
  auto contractInfo = analyzeGemmContract(forOp, loadA.readOp, loadB.readOp);
  if (!contractInfo) {
    contractInfo = analyzeGemmContract(forOp, loadB.readOp, loadA.readOp);
    if (!contractInfo)
      return reject("unsupported_pattern",
                    "could not identify vector.contract consuming A/B loads");
    std::swap(loadA, loadB);
  }

  vector::ContractionOp contractOp = contractInfo->contractOp;
  unsigned accIdx = contractInfo->accIdx;
  VectorType accTy = contractInfo->accTy;
  auto aTy = dyn_cast<VectorType>(loadA.readOp.getType());
  auto bTy = dyn_cast<VectorType>(loadB.readOp.getType());
  if (!aTy || !bTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      accTy.getRank() != 2)
    return reject("unsupported_pattern",
                  "A, B, and accumulator must be rank-2 vectors");

  int64_t BM = accTy.getDimSize(0);
  int64_t BN = accTy.getDimSize(1);
  int64_t BK = aTy.getDimSize(1);
  if (BM < microM || BM % microM != 0 || aTy.getDimSize(0) != BM ||
      bTy.getDimSize(0) != BK || bTy.getDimSize(1) != BN)
    return reject("unsupported_pattern",
                  "matrix tile shape is incompatible with microM schedule");

  // This fused schedule only materializes the accumulator result.  The block
  // pointer loop results in the matmul kernel are dead; if a future pattern
  // uses them, fall back to the conservative double-buffer path.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    if (i != accIdx && !forOp.getResult(i).use_empty())
      return reject("no_bounded_lifetime",
                    "non-accumulator loop result is used");

  auto memRefTyA = cast<MemRefType>(loadA.readOp.getBase().getType());
  auto memRefTyB = cast<MemRefType>(loadB.readOp.getBase().getType());
  SmallVector<int64_t> stridesA, stridesB;
  if (!getStaticStrides(memRefTyA, stridesA) ||
      !getStaticStrides(memRefTyB, stridesB) || stridesA.size() < 2 ||
      stridesB.size() < 2)
    return reject("dynamic_shape_or_stride",
                  "A/B memrefs require static rank-2 strides");

  unsigned elemBytesA = memRefTyA.getElementType().getIntOrFloatBitWidth() / 8;
  unsigned elemBytesB = memRefTyB.getElementType().getIntOrFloatBitWidth() / 8;
  unsigned elemBytesAcc = accTy.getElementType().getIntOrFloatBitWidth() / 8;

  int64_t stepBytesA = BK * stridesA[1] * elemBytesA;
  int64_t stepBytesB = BK * stridesB[0] * elemBytesB;
  int64_t rowBytesA = stridesA[0] * elemBytesA;
  int64_t accRowBytes = BN * elemBytesAcc;

  auto microATy = VectorType::get({microM, BK}, aTy.getElementType());
  auto microAccTy = VectorType::get({microM, BN}, accTy.getElementType());
  int64_t microABytes = getTileBytes(microATy);
  int64_t bWindowBytes = loadB.tileBytes * windowK;
  int64_t accBytes = getTileBytes(accTy);

  SPMSpaceManager spmLayout(spmBase, spmSize);
  auto allocBWindow = spmLayout.alloc(bWindowBytes, /*alignment=*/1,
                                      SPMSpaceManager::Lifetime::Loop);
  auto allocAMicro0 = spmLayout.alloc(microABytes, /*alignment=*/1,
                                      SPMSpaceManager::Lifetime::Loop);
  auto allocAMicro1 = spmLayout.alloc(microABytes, /*alignment=*/1,
                                      SPMSpaceManager::Lifetime::Loop);
  auto allocAcc = spmLayout.alloc(accBytes, /*alignment=*/1,
                                  SPMSpaceManager::Lifetime::Loop);
  if (!allocBWindow || !allocAMicro0 || !allocAMicro1 || !allocAcc)
    return reject("spm_capacity_overflow",
                  "SPM capacity cannot fit B window, A micro double buffer, "
                  "and accumulator");

  int64_t addrBWindow = allocBWindow->address;
  int64_t addrAMicro0 = allocAMicro0->address;
  int64_t addrAMicro1 = allocAMicro1->address;
  int64_t addrAcc = allocAcc->address;

  Location loc = forOp.getLoc();
  InsertedBeforeGuard guard(forOp.getOperation());
  OpBuilder b(forOp);

  Value dramAddrA = computePrologueDramAddr(b, loc, loadA.readOp, forOp);
  Value dramAddrB = computePrologueDramAddr(b, loc, loadB.readOp, forOp);
  if (!dramAddrA || !dramAddrB) {
    guard.cleanup();
    return reject("dynamic_shape_or_stride",
                  "failed to compute prologue DRAM address");
  }

  if (report) {
    int64_t microSlices = BM / microM;
    report->records.push_back(makeBWindowRecord(
        bTy, windowK, microSlices * windowK, bWindowBytes, addrBWindow));
    report->records.push_back(
        makeAMicroRecord(microATy, /*uses=*/1, microABytes * 2, addrAMicro0));
    report->records.push_back(
        makeAccumulatorRecord(accTy, microSlices * 2 + 1, accBytes, addrAcc));
  }

  MLIRContext *ctx = forOp.getContext();
  auto aMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
  auto bMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
  auto cMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
  auto mapsAttr = b.getAffineMapArrayAttr({aMap, bMap, cMap});
  auto iterAttr = b.getArrayAttr(
      {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
       vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
       vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});

  Value fullInit = forOp.getInitArgs()[accIdx];
  for (int64_t mOff = 0; mOff < BM; mOff += microM) {
    Value microInit = vector::ExtractStridedSliceOp::create(
        b, loc, fullInit,
        /*offsets=*/{mOff, 0}, /*sizes=*/{microM, BN},
        /*strides=*/{1, 1});
    emitSpmWrite(b, loc, i64Cst(b, loc, addrAcc + mOff * accRowBytes),
                 microInit);
  }

  auto winFor =
      scf::ForOp::create(b, loc, i64Cst(b, loc, 0), i64Cst(b, loc, trips),
                         i64Cst(b, loc, windowK));
  Block *winBody = winFor.getBody();
  if (!winBody->empty() && winBody->mightHaveTerminator())
    winBody->getTerminator()->erase();

  b.setInsertionPointToStart(winBody);
  Value winIter = winFor.getInductionVar();

  // Stage the resident B window once.  windowK is capped to the DMA queue
  // depth default (4) by the caller/env setting.
  auto bStageFor = scf::ForOp::create(
      b, loc, i64Cst(b, loc, 0), i64Cst(b, loc, windowK), i64Cst(b, loc, 1));
  Block *bStageBody = bStageFor.getBody();
  if (!bStageBody->empty() && bStageBody->mightHaveTerminator())
    bStageBody->getTerminator()->erase();

  b.setInsertionPointToStart(bStageBody);
  Value bLocal = bStageFor.getInductionVar();
  Value bAbsIter = arith::AddIOp::create(b, loc, winIter, bLocal);
  Value bDram = arith::AddIOp::create(
      b, loc, dramAddrB,
      arith::MulIOp::create(b, loc, bAbsIter, i64Cst(b, loc, stepBytesB)));
  Value bSpm = arith::AddIOp::create(
      b, loc, i64Cst(b, loc, addrBWindow),
      arith::MulIOp::create(b, loc, bLocal, i64Cst(b, loc, loadB.tileBytes)));
  emitDmaEnqueue(b, loc, bSpm, bDram, bTy, memRefTyB);
  scf::YieldOp::create(b, loc);

  b.setInsertionPointAfter(bStageFor);
  triton::cpu::DmaWaitOp::create(b, loc);

  for (int64_t mOff = 0; mOff < BM; mOff += microM) {
    Value accAddr = i64Cst(b, loc, addrAcc + mOff * accRowBytes);
    Value aDram0 = arith::AddIOp::create(
        b, loc, dramAddrA,
        arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(b, loc, winIter, i64Cst(b, loc, stepBytesA)),
            i64Cst(b, loc, mOff * rowBytesA)));
    emitDmaEnqueue(b, loc, i64Cst(b, loc, addrAMicro0), aDram0, microATy,
                   memRefTyA);
    Value microInit = emitSpmRead(b, loc, accAddr, microAccTy);

    auto kFor = scf::ForOp::create(b, loc, i64Cst(b, loc, 0),
                                   i64Cst(b, loc, windowK), i64Cst(b, loc, 1),
                                   ValueRange{microInit, i64Cst(b, loc, 0)});
    Block *kBody = kFor.getBody();
    if (!kBody->empty() && kBody->mightHaveTerminator())
      kBody->getTerminator()->erase();

    b.setInsertionPointToStart(kBody);
    triton::cpu::DmaWaitOp::create(b, loc);

    Value kLocal = kFor.getInductionVar();
    Value kAbsIter = arith::AddIOp::create(b, loc, winIter, kLocal);
    Value aBufIdx = kFor.getRegionIterArgs()[1];
    Value zero = i64Cst(b, loc, 0);
    Value one = i64Cst(b, loc, 1);
    Value isZero =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, aBufIdx, zero);
    Value aCur =
        arith::SelectOp::create(b, loc, isZero, i64Cst(b, loc, addrAMicro0),
                                i64Cst(b, loc, addrAMicro1));
    Value aNxt =
        arith::SelectOp::create(b, loc, isZero, i64Cst(b, loc, addrAMicro1),
                                i64Cst(b, loc, addrAMicro0));

    Value nextKLocal = arith::AddIOp::create(b, loc, kLocal, one);
    Value hasNextA = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                           nextKLocal, i64Cst(b, loc, windowK));
    Value nextKAbsIter = arith::AddIOp::create(b, loc, kAbsIter, one);
    Value nextADram = arith::AddIOp::create(
        b, loc, dramAddrA,
        arith::AddIOp::create(b, loc,
                              arith::MulIOp::create(b, loc, nextKAbsIter,
                                                    i64Cst(b, loc, stepBytesA)),
                              i64Cst(b, loc, mOff * rowBytesA)));
    auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNextA, false);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    emitDmaEnqueue(b, loc, aNxt, nextADram, microATy, memRefTyA);
    b.setInsertionPointAfter(ifOp);

    Value aVal = emitSpmRead(b, loc, aCur, microATy);
    Value bReadAddr = arith::AddIOp::create(
        b, loc, i64Cst(b, loc, addrBWindow),
        arith::MulIOp::create(b, loc, kLocal, i64Cst(b, loc, loadB.tileBytes)));
    Value bVal = emitSpmRead(b, loc, bReadAddr, bTy);
    Value microAcc = kFor.getRegionIterArgs()[0];
    Value contracted = vector::ContractionOp::create(
        b, loc, microAccTy, aVal, bVal, microAcc, mapsAttr, iterAttr);
    Value flipped = arith::SubIOp::create(b, loc, one, aBufIdx);
    scf::YieldOp::create(b, loc, ValueRange{contracted, flipped});

    b.setInsertionPointAfter(kFor);
    emitSpmWrite(b, loc, accAddr, kFor.getResult(0));
  }

  scf::YieldOp::create(b, loc);
  b.setInsertionPointAfter(winFor);

  Value finalAcc = emitSpmRead(b, loc, i64Cst(b, loc, addrAcc), accTy);
  forOp.getResult(accIdx).replaceAllUsesWith(finalAcc);
  guard.commit();
  forOp.erase();

  (void)contractOp;
  return true;
}

/// Transform a GEMM K-loop with double-buffered SPM.
/// `dotLoads` must have exactly 2 entries (A and B tile loads).
/// Returns true on success.
static bool transformGemmLoop(scf::ForOp forOp,
                              ArrayRef<TiledLoadInfo> dotLoads, int64_t spmBase,
                              int64_t spmSize) {
  if (dotLoads.size() != 2)
    return false;

  TiledLoadInfo loadA = dotLoads[0];
  TiledLoadInfo loadB = dotLoads[1];
  auto contractInfo = analyzeGemmContract(forOp, loadA.readOp, loadB.readOp);
  if (!contractInfo) {
    contractInfo = analyzeGemmContract(forOp, loadB.readOp, loadA.readOp);
    if (!contractInfo)
      return false;
    std::swap(loadA, loadB);
  }

  int64_t tileA = loadA.tileBytes;
  int64_t tileB = loadB.tileBytes;

  // Loop boundary guard: trip count must be a known multiple of the step.
  // If not provable at compile time, bail out to the cache path.
  auto lbCst = getConstantIntValue(forOp.getLowerBound());
  auto ubCst = getConstantIntValue(forOp.getUpperBound());
  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!lbCst || !ubCst || !stepCst || (*ubCst - *lbCst) % *stepCst != 0)
    return false;

  SPMSpaceManager spmLayout(spmBase, spmSize);
  auto allocA0 =
      spmLayout.alloc(tileA, /*alignment=*/1, SPMSpaceManager::Lifetime::Loop);
  auto allocA1 =
      spmLayout.alloc(tileA, /*alignment=*/1, SPMSpaceManager::Lifetime::Loop);
  auto allocB0 =
      spmLayout.alloc(tileB, /*alignment=*/1, SPMSpaceManager::Lifetime::Loop);
  auto allocB1 =
      spmLayout.alloc(tileB, /*alignment=*/1, SPMSpaceManager::Lifetime::Loop);
  if (!allocA0 || !allocA1 || !allocB0 || !allocB1)
    return false;

  int64_t addrA0 = allocA0->address;
  int64_t addrA1 = allocA1->address;
  int64_t addrB0 = allocB0->address;
  int64_t addrB1 = allocB1->address;

  Location loc = forOp.getLoc();
  InsertedBeforeGuard guard(forOp.getOperation());
  OpBuilder b(forOp);

  auto readA = loadA.readOp;
  auto readB = loadB.readOp;
  auto memRefTyA = cast<MemRefType>(readA.getBase().getType());
  auto memRefTyB = cast<MemRefType>(readB.getBase().getType());

  // --- Prologue: DMA first tiles into buffer 0 ---
  Value dramAddrA = computePrologueDramAddr(b, loc, readA, forOp);
  Value dramAddrB = computePrologueDramAddr(b, loc, readB, forOp);
  if (!dramAddrA || !dramAddrB) {
    guard.cleanup();
    return false;
  }

  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrA0), dramAddrA, loadA.vecTy,
                 memRefTyA);
  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrB0), dramAddrB, loadB.vecTy,
                 memRefTyB);
  // No explicit prologue wait: the DmaWait emitted at the top of the
  // body block (before any SPM read of buffer 0) already polls until
  // the prologue DMAs complete.  Issuing another wait here just adds a
  // redundant volatile-load BB and bookkeeping for ~zero stall (the
  // body-top wait then sees status=0 immediately on iter 0).

  // --- Add buf_idx iter_arg to the loop ---
  // We need to add a new i64 iter_arg for the buffer index (0 or 1).
  // Also need to thread the DRAM addresses for next-iteration prefetch.
  //
  // For simplicity in the MVP, we recompute DRAM addresses inside the loop
  // rather than threading them as iter_args.  The K-loop's induction variable
  // already encodes the iteration, and the original transfer_read indices
  // are recomputed each iteration.

  // Insert buf_idx as a new iter_arg.
  Value initBufIdx = i64Cst(b, loc, 0);

  // We'll rebuild the loop with the extra iter_arg.
  // Collect existing init args.
  SmallVector<Value> newInitArgs(forOp.getInitArgs());
  newInitArgs.push_back(initBufIdx);

  // Create new for loop.
  auto newForOp =
      scf::ForOp::create(b, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                         forOp.getStep(), newInitArgs);

  // The new loop's body block has: iv, then one block arg per init arg.
  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  // Map old block args to new block args.
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  unsigned numOldArgs = forOp.getRegionIterArgs().size();
  for (unsigned i = 0; i < numOldArgs; ++i)
    mapping.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);

  Value bufIdx = newForOp.getRegionIterArgs()[numOldArgs]; // new buf_idx arg

  // Clone old body into new body (before the terminator).
  b.setInsertionPointToStart(newBody);
  // Remove the auto-generated yield in the new loop (if present).
  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  // DMA wait at TOP of body — wait for the prefetch that filled CURRENT buffer
  // (issued in the previous iteration, or in the prologue for iter 0).
  //
  // Why at top, not at end: the wait lowers to a volatile-poll spin loop,
  // i.e. its own basic block.  Placed at the end, LLVM schedules it between
  // the SPM loads and the contract-derived fmuladds — which splits load+
  // shufflevector(splat) across BBs and defeats the RISC-V backend's
  // load+splat -> vfmacc.vf folding (the cache path's main matmul codegen).
  // At the top, the prefetch scf.if joins back into a single block that
  // contains loads -> FMAs -> yield, which the backend folds correctly.
  triton::cpu::DmaWaitOp::create(b, loc);

  // Build buffer selection BEFORE cloning the body so these ops dominate
  // the SPM reads that will replace the cloned transfer_reads.
  Value zero = i64Cst(b, loc, 0);
  Value isZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bufIdx, zero);

  // SPM addresses for A
  Value spmA0 = i64Cst(b, loc, addrA0);
  Value spmA1 = i64Cst(b, loc, addrA1);
  Value spmACur = arith::SelectOp::create(b, loc, isZero, spmA0, spmA1);

  // SPM addresses for B
  Value spmB0 = i64Cst(b, loc, addrB0);
  Value spmB1 = i64Cst(b, loc, addrB1);
  Value spmBCur = arith::SelectOp::create(b, loc, isZero, spmB0, spmB1);

  // --- Prefetch next tiles FIRST (async, overlaps with compute below) ---
  Value iv = newForOp.getInductionVar();
  Value step = newForOp.getStep();
  Value ub = newForOp.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  Value spmANxt = arith::SelectOp::create(b, loc, isZero, spmA1, spmA0);
  Value spmBNxt = arith::SelectOp::create(b, loc, isZero, spmB1, spmB0);

  auto shapeA = loadA.vecTy.getShape();
  auto shapeB = loadB.vecTy.getShape();
  unsigned elemBytesA = memRefTyA.getElementType().getIntOrFloatBitWidth() / 8;
  unsigned elemBytesB = memRefTyB.getElementType().getIntOrFloatBitWidth() / 8;

  SmallVector<int64_t> stridesA, stridesB;
  int64_t offA, offB;
  (void)memRefTyA.getStridesAndOffset(stridesA, offA);
  (void)memRefTyB.getStridesAndOffset(stridesB, offB);

  int64_t kDimA = (shapeA.size() >= 2) ? 1 : 0;
  int64_t stepBytesA = shapeA[kDimA] * stridesA[kDimA] * elemBytesA;
  int64_t kDimB = 0;
  int64_t stepBytesB = shapeB[kDimB] * stridesB[kDimB] * elemBytesB;

  Value newLb = newForOp.getLowerBound();
  Value kOffset = arith::SubIOp::create(b, loc, iv, newLb);
  Value kOffI64 = toI64(b, loc, kOffset);
  Value stepI64 = toI64(b, loc, step);

  Value iterNum = arith::DivSIOp::create(b, loc, kOffI64, stepI64);
  Value one = i64Cst(b, loc, 1);
  Value nextIterNum = arith::AddIOp::create(b, loc, iterNum, one);

  Value nextDramA = arith::AddIOp::create(
      b, loc, dramAddrA,
      arith::MulIOp::create(b, loc, nextIterNum, i64Cst(b, loc, stepBytesA)));
  Value nextDramB = arith::AddIOp::create(
      b, loc, dramAddrB,
      arith::MulIOp::create(b, loc, nextIterNum, i64Cst(b, loc, stepBytesB)));

  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  emitDmaEnqueue(b, loc, spmANxt, nextDramA, loadA.vecTy, memRefTyA);
  emitDmaEnqueue(b, loc, spmBNxt, nextDramB, loadB.vecTy, memRefTyB);
  b.setInsertionPointAfter(ifOp);

  // --- SPM read + compute (from CURRENT buffer, while prefetch runs) ---
  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (&op == readA.getOperation()) {
      Value spmValA = emitSpmRead(b, loc, spmACur, loadA.vecTy);
      mapping.map(readA.getResult(), spmValA);
      continue;
    }
    if (&op == readB.getOperation()) {
      Value spmValB = emitSpmRead(b, loc, spmBCur, loadB.vecTy);
      mapping.map(readB.getResult(), spmValB);
      continue;
    }
    b.clone(op, mapping);
  }

  // (DMA wait is at TOP of body, not here — see comment above.)
  b.setInsertionPointToEnd(newBody);

  // --- Yield with flipped buf_idx ---
  Value flipped = arith::SubIOp::create(b, loc, one, bufIdx);
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (auto val : oldYield.getOperands())
    yieldVals.push_back(mapping.lookupOrDefault(val));
  yieldVals.push_back(flipped);
  scf::YieldOp::create(b, loc, yieldVals);

  // Replace uses of old loop results with new loop results.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  guard.commit();
  forOp.erase();

  return true;
}

//===----------------------------------------------------------------------===//
// Reduction double-buffering transformation.
//
// For loops with a single tiled load (not feeding a dot), insert a
// double-buffered DMA prefetch: DMA the next chunk into the alternate
// SPM buffer while reducing the current chunk.
//
// Input:
//   scf.for %i = 0 to N step BS {
//     %x = transfer_read %mem[%i, ...] : vector<BS×f32>
//     ... reduce ...
//   }
//
// Output:
//   // Prologue: DMA first chunk
//   dma_enqueue_2d(spm_buf0, dram_start, ...)
//
//   scf.for %i = 0 to N step BS iter_args(%buf_idx = 0) {
//     dma_wait  // wait for current chunk: prologue on iter 0, prior prefetch
//     %spm_cur = select(%buf_idx == 0, spm_buf0, spm_buf1)
//     %spm_nxt = select(%buf_idx == 0, spm_buf1, spm_buf0)
//     // Prefetch next chunk (async, into alternate buffer)
//     scf.if (%i + BS < N) {
//       dma_enqueue_2d(spm_nxt, dram_next, ...)
//     }
//     // Read current from SPM
//     %x = transfer_read spm_cur : memref<..., 3>
//     ... reduce ...
//     yield (1 - %buf_idx), ...
//   }
//===----------------------------------------------------------------------===//

struct ReductionLoadPlan {
  TiledLoadInfo load;
  MemRefType memRefTy;
  int64_t addrBuf0;
  int64_t addrBuf1;
  int64_t stepBytes;
  Value dramAddr;
  Value spmCur;
  Value spmNxt;
};

static bool transformReductionLoop(scf::ForOp forOp,
                                   ArrayRef<TiledLoadInfo> loads,
                                   int64_t spmBase, int64_t spmSize) {
  if (loads.empty())
    return false;

  auto lbCst = getConstantIntValue(forOp.getLowerBound());
  auto ubCst = getConstantIntValue(forOp.getUpperBound());
  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!lbCst || !ubCst || !stepCst || (*ubCst - *lbCst) % *stepCst != 0)
    return false;

  SPMSpaceManager spmLayout(spmBase, spmSize);
  SmallVector<ReductionLoadPlan> plans;
  plans.reserve(loads.size());

  for (const TiledLoadInfo &load : loads) {
    auto readOp = load.readOp;
    auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
    if (!memRefTy || !canComputePrologueDramAddr(readOp, forOp))
      return false;

    auto stepBytes = getLoopStepBytes(readOp, forOp,
                                      /*requireLoopIv=*/true);
    if (!stepBytes)
      return false;

    auto allocBuf0 = spmLayout.alloc(load.tileBytes, /*alignment=*/1,
                                     SPMSpaceManager::Lifetime::Loop);
    auto allocBuf1 = spmLayout.alloc(load.tileBytes, /*alignment=*/1,
                                     SPMSpaceManager::Lifetime::Loop);
    if (!allocBuf0 || !allocBuf1)
      return false;

    plans.push_back(ReductionLoadPlan{load, memRefTy, allocBuf0->address,
                                      allocBuf1->address, *stepBytes, Value(),
                                      Value(), Value()});
  }

  Location loc = forOp.getLoc();
  InsertedBeforeGuard guard(forOp.getOperation());
  OpBuilder b(forOp);

  // Prologue: DMA first chunk for every stream.
  for (ReductionLoadPlan &plan : plans) {
    plan.dramAddr = computePrologueDramAddr(b, loc, plan.load.readOp, forOp);
    if (!plan.dramAddr) {
      guard.cleanup();
      return false;
    }
    emitDmaEnqueue(b, loc, i64Cst(b, loc, plan.addrBuf0), plan.dramAddr,
                   plan.load.vecTy, plan.memRefTy);
  }

  // Rebuild loop with one extra iter_arg: the current buffer index.
  SmallVector<Value> initArgs(forOp.getInitArgs());
  initArgs.push_back(i64Cst(b, loc, 0));
  auto newForOp =
      scf::ForOp::create(b, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                         forOp.getStep(), initArgs);

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  unsigned numOldArgs = forOp.getRegionIterArgs().size();
  for (unsigned i = 0; i < numOldArgs; ++i)
    mapping.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);
  Value bufIdx = newForOp.getRegionIterArgs()[numOldArgs];

  b.setInsertionPointToStart(newBody);
  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  // Wait for the DMA that filled the current buffer.  Iteration 0 waits for
  // the prologue; later iterations wait for the prior iteration's prefetch.
  triton::cpu::DmaWaitOp::create(b, loc);

  Value zero = i64Cst(b, loc, 0);
  Value isZero =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bufIdx, zero);
  for (ReductionLoadPlan &plan : plans) {
    Value spmBuf0 = i64Cst(b, loc, plan.addrBuf0);
    Value spmBuf1 = i64Cst(b, loc, plan.addrBuf1);
    plan.spmCur = arith::SelectOp::create(b, loc, isZero, spmBuf0, spmBuf1);
    plan.spmNxt = arith::SelectOp::create(b, loc, isZero, spmBuf1, spmBuf0);
  }

  // --- Prefetch next chunk into the alternate buffer. ---
  Value iv = newForOp.getInductionVar();
  Value step = newForOp.getStep();
  Value ub = newForOp.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  Value lbInLoop = newForOp.getLowerBound();
  Value nextOff = arith::SubIOp::create(b, loc, nextIv, lbInLoop);
  Value nextOffI64 = toI64(b, loc, nextOff);

  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  for (ReductionLoadPlan &plan : plans) {
    Value nextByteOff = arith::MulIOp::create(b, loc, nextOffI64,
                                              i64Cst(b, loc, plan.stepBytes));
    Value nextDram = arith::AddIOp::create(b, loc, plan.dramAddr, nextByteOff);
    emitDmaEnqueue(b, loc, plan.spmNxt, nextDram, plan.load.vecTy,
                   plan.memRefTy);
  }
  b.setInsertionPointAfter(ifOp);

  // Clone the original body, replacing the transfer_read with the current
  // SPM buffer reads.  The prefetch above targets the alternate buffers, so
  // DMA can overlap with reduction/streaming compute without racing reads.
  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;

    bool replacedRead = false;
    for (const ReductionLoadPlan &plan : plans) {
      auto readOp = plan.load.readOp;
      if (&op == readOp.getOperation()) {
        Value spmVal = emitSpmRead(b, loc, plan.spmCur, plan.load.vecTy);
        mapping.map(readOp.getResult(), spmVal);
        replacedRead = true;
        break;
      }
    }
    if (replacedRead)
      continue;

    b.clone(op, mapping);
  }

  // Yield.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (auto val : oldYield.getOperands())
    yieldVals.push_back(mapping.lookupOrDefault(val));
  Value one = i64Cst(b, loc, 1);
  Value flipped = arith::SubIOp::create(b, loc, one, bufIdx);
  yieldVals.push_back(flipped);
  scf::YieldOp::create(b, loc, yieldVals);

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  guard.commit();
  forOp.erase();

  return true;
}

//===----------------------------------------------------------------------===//
// Pass definition.
//===----------------------------------------------------------------------===//

struct ConvertMemoryToSPM
    : public triton::cpu::impl::ConvertMemoryToSPMBase<ConvertMemoryToSPM> {
  ConvertMemoryToSPM() = default;
  ConvertMemoryToSPM(int64_t spmBase_, int64_t spmSize_) {
    this->spmBase = spmBase_;
    this->spmSize = spmSize_;
  }
  ConvertMemoryToSPM(int64_t spmBase_, int64_t spmSize_, int64_t microM_,
                     int64_t windowK_) {
    this->spmBase = spmBase_;
    this->spmSize = spmSize_;
    this->microM = microM_;
    this->windowK = windowK_;
  }
  ConvertMemoryToSPM(int64_t spmBase_, int64_t spmSize_, int64_t microM_,
                     int64_t windowK_, bool enableReductions_) {
    this->spmBase = spmBase_;
    this->spmSize = spmSize_;
    this->microM = microM_;
    this->windowK = windowK_;
    this->enableReductions = enableReductions_;
  }
  ConvertMemoryToSPM(int64_t spmBase_, int64_t spmSize_, int64_t microM_,
                     int64_t windowK_, bool enableReductions_,
                     bool promotionReport_) {
    this->spmBase = spmBase_;
    this->spmSize = spmSize_;
    this->microM = microM_;
    this->windowK = windowK_;
    this->enableReductions = enableReductions_;
    this->promotionReport = promotionReport_;
  }
  ConvertMemoryToSPM(int64_t spmBase_, int64_t spmSize_, int64_t microM_,
                     int64_t windowK_, bool enableReductions_,
                     bool enableRowResidentReductions_,
                     int64_t rowResidentMaxBytes_,
                     StringRef rowResidentProducerPass_,
                     bool enablePromotionProfitability_,
                     bool promotionReport_) {
    this->spmBase = spmBase_;
    this->spmSize = spmSize_;
    this->microM = microM_;
    this->windowK = windowK_;
    this->enableReductions = enableReductions_;
    this->enableRowResidentReductions = enableRowResidentReductions_;
    this->rowResidentMaxBytes = rowResidentMaxBytes_;
    this->rowResidentProducerPass = rowResidentProducerPass_.str();
    this->enablePromotionProfitability = enablePromotionProfitability_;
    this->promotionReport = promotionReport_;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    DenseMap<Operation *, SPMPromotionReport> reports;
    DenseMap<Operation *, AttentionFunctionState> attentionStates;
    llvm::DenseSet<Operation *> rowResidentHandledLoops;

    if (enableRowResidentReductions) {
      mod.walk([&](FunctionOpInterface funcOp) {
        if (funcOp.getFunctionBody().empty())
          return;

        SmallVector<scf::ForOp> topLevelLoops;
        Block &entryBlock = funcOp.getFunctionBody().front();
        for (Operation &op : entryBlock) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op))
            topLevelLoops.push_back(forOp);
        }

        SPMPromotionReport *report = nullptr;
        if (promotionReport)
          report = &reports[funcOp.getOperation()];

        if (isRowBlockDmaProducerPassMode(rowResidentProducerPass) &&
            getEnvBool("TRITON_SPM_SOFTMAX_INTERNAL_ROW_BLOCK", false)) {
          int64_t rowBlock =
              getEnvInt64("TRITON_SPM_SOFTMAX_ROW_BLOCK",
                          getEnvInt64("SOFTMAX_SPM_ROW_BLOCK", 2));
          int64_t rowGroupBlocks =
              getEnvInt64("TRITON_SPM_SOFTMAX_ROW_GROUP_BLOCKS",
                          getEnvInt64("SOFTMAX_SPM_ROW_GROUP_BLOCKS", 8));
          if (lowerCanonicalSoftmaxToRowBlockGroup(topLevelLoops, funcOp,
                                                   rowBlock, rowGroupBlocks)) {
            topLevelLoops.clear();
            for (Operation &op : entryBlock) {
              if (auto forOp = dyn_cast<scf::ForOp>(&op))
                topLevelLoops.push_back(forOp);
            }
          }
        }

        std::string rejectReasonCode;
        std::string rejectReason;
        auto plan = matchSoftmaxRowBlockGroupResidencyPlan(
            topLevelLoops, funcOp, rejectReasonCode, rejectReason,
            rowResidentProducerPass);
        if (!plan) {
          plan = matchLayerNormResidencyPlan(topLevelLoops, funcOp,
                                             rejectReasonCode, rejectReason,
                                             rowResidentProducerPass);
        }
        if (!plan) {
          plan =
              matchSoftmaxResidencyPlan(topLevelLoops, funcOp, rejectReasonCode,
                                        rejectReason, rowResidentProducerPass);
        }
        if (!plan) {
          if (report && !rejectReasonCode.empty()) {
            report->rejections.push_back(
                makeRowResidentRejection(rejectReasonCode, rejectReason));
          }
          return;
        }

        transformReductionResidencyPlan(
            *plan, spmBase, spmSize, rowResidentMaxBytes,
            enablePromotionProfitability, report, rowResidentHandledLoops);
      });
    }

    // Collect scf.for ops (avoid modifying while iterating).
    SmallVector<scf::ForOp> forOps;
    mod.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : forOps) {
      if (rowResidentHandledLoops.contains(forOp.getOperation()))
        continue;
      auto loads = findTiledLoads(forOp);
      if (loads.empty())
        continue;

      // Classify: GEMM if exactly 2 loads feed a dot product.
      SmallVector<TiledLoadInfo> dotLoads, nonDotLoads;
      for (auto &l : loads) {
        if (l.feedsDot)
          dotLoads.push_back(l);
        else
          nonDotLoads.push_back(l);
      }

      if (dotLoads.size() == 2) {
        SPMPromotionReport *report = nullptr;
        auto parentFunc = forOp->getParentOfType<FunctionOpInterface>();
        if (promotionReport && parentFunc)
          report = &reports[parentFunc.getOperation()];
        AttentionFunctionState *attentionState = nullptr;
        if (parentFunc) {
          attentionState = &attentionStates[parentFunc.getOperation()];
          attentionState->initialize(spmBase, spmSize);
        }
        if (!(attentionState &&
              (transformAttentionV2KVStreamingLoop(forOp, dotLoads,
                                                   *attentionState, report) ||
               transformAttentionV2QResidentLoop(forOp, dotLoads,
                                                 *attentionState, report))) &&
            !transformFusedMicroGemmLoop(forOp, dotLoads, spmBase, spmSize,
                                         microM, windowK,
                                         enablePromotionProfitability, report))
          transformGemmLoop(forOp, dotLoads, spmBase, spmSize);
      } else if (dotLoads.empty() && enableReductions &&
                 !enableRowResidentReductions) {
        if (enablePromotionProfitability) {
          if (promotionReport && !nonDotLoads.empty()) {
            auto funcOp = forOp->getParentOfType<FunctionOpInterface>();
            if (funcOp) {
              SPMProfitabilityEvidence evidence =
                  evaluateD3StreamingReductionProfitability(forOp, nonDotLoads,
                                                            spmSize);
              reports[funcOp.getOperation()].rejections.push_back(
                  makeD3StreamingReductionRejection(forOp, nonDotLoads,
                                                    std::move(evidence)));
            }
          }
          continue;
        }
        bool transformed =
            transformReductionLoop(forOp, nonDotLoads, spmBase, spmSize);
        if (!transformed && promotionReport && !nonDotLoads.empty()) {
          auto funcOp = forOp->getParentOfType<FunctionOpInterface>();
          if (funcOp)
            reports[funcOp.getOperation()].rejections.push_back(
                makePromotionRejection(
                    "reduction_streaming", "unsupported_pattern",
                    "reduction/streaming loop did not match the current SPM "
                    "lowering guards"));
        }
      } else if (promotionReport && dotLoads.empty() && !nonDotLoads.empty()) {
        auto funcOp = forOp->getParentOfType<FunctionOpInterface>();
        if (funcOp)
          reports[funcOp.getOperation()].rejections.push_back(
              makePromotionRejection(
                  "reduction_streaming", "policy_disabled",
                  "reduction/streaming SPM promotion is disabled by default; "
                  "leave the candidate on the cache path"));
      }
      // Otherwise: leave unchanged (cache path).
    }

    if (promotionReport) {
      bool failedSidecarWrite = false;
      mod.walk([&](FunctionOpInterface funcOp) {
        if (failedSidecarWrite)
          return;
        if (funcOp.getVisibility() != SymbolTable::Visibility::Public ||
            funcOp.getFunctionBody().empty())
          return;
        auto it = reports.find(funcOp.getOperation());
        const SPMPromotionReport emptyReport;
        const SPMPromotionReport &report =
            it == reports.end() ? emptyReport : it->second;
        if (failed(writePromotionReport(funcOp, report)))
          failedSidecarWrite = true;
      });
      if (failedSidecarWrite)
        signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryToSPM() {
  return std::make_unique<ConvertMemoryToSPM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMemoryToSPM(int64_t spmBase, int64_t spmSize) {
  return std::make_unique<ConvertMemoryToSPM>(spmBase, spmSize);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMemoryToSPM(int64_t spmBase, int64_t spmSize, int64_t microM,
                         int64_t windowK) {
  return std::make_unique<ConvertMemoryToSPM>(spmBase, spmSize, microM,
                                              windowK);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMemoryToSPM(int64_t spmBase, int64_t spmSize, int64_t microM,
                         int64_t windowK, bool enableReductions) {
  return std::make_unique<ConvertMemoryToSPM>(spmBase, spmSize, microM, windowK,
                                              enableReductions);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMemoryToSPM(int64_t spmBase, int64_t spmSize, int64_t microM,
                         int64_t windowK, bool enableReductions,
                         bool promotionReport) {
  return std::make_unique<ConvertMemoryToSPM>(
      spmBase, spmSize, microM, windowK, enableReductions, promotionReport);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryToSPM(
    int64_t spmBase, int64_t spmSize, int64_t microM, int64_t windowK,
    bool enableReductions, bool enableRowResidentReductions,
    int64_t rowResidentMaxBytes, StringRef rowResidentProducerPass,
    bool enablePromotionProfitability, bool promotionReport) {
  return std::make_unique<ConvertMemoryToSPM>(
      spmBase, spmSize, microM, windowK, enableReductions,
      enableRowResidentReductions, rowResidentMaxBytes, rowResidentProducerPass,
      enablePromotionProfitability, promotionReport);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
