//===- ConvertMemoryToSPM.cpp - Convert tiled loads to DMA+SPM transfers --===//
//
// This pass transforms tiled DRAM loads inside scf.for loops into
// DMA-based scratchpad memory (SPM) transfers.
//
// Supported patterns:
//   1) GEMM double-buffering: K-loop with two tiled loads feeding a dot
//      product.  Each load gets two SPM buffers; while the current tiles
//      are computed, the next tiles are prefetched asynchronously.
//   2) Reduction prefetch: single loop with one tiled load.  Single-buffer
//      DMA prefetch for the next chunk.
//
// Loads that don't match these patterns are left unchanged (cache path).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <optional>
#include <utility>

#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "cpu/include/TritonCPUTransforms/SPMSpaceManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/SPMAttrs.h"

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

static Value i64Cst(OpBuilder &b, Location loc, int64_t val) {
  return arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(val));
}

static Value idxCst(OpBuilder &b, Location loc, int64_t val) {
  return arith::ConstantIndexOp::create(b, loc, val);
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
  bool isBlockPtr = baseDefOp &&
                    baseDefOp->getParentRegion() == &forOp.getRegion();

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
    Value initMemRef = triton::cpu::ExtractMemRefOp::create(
        b, loc, memRefTy, initBlockPtr);
    auto initIndicesOp = triton::cpu::ExtractIndicesOp::create(
        b, loc, initBlockPtr);

    Value ptrIdx = memref::ExtractAlignedPointerAsIndexOp::create(
        b, loc, initMemRef);
    ptrI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), ptrIdx);

    for (auto r : initIndicesOp.getResults())
      indices.push_back(r);
  } else {
    Value ptrIdx = memref::ExtractAlignedPointerAsIndexOp::create(
        b, loc, base);
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
    Value idxI64 = arith::IndexCastOp::create(
        b, loc, b.getI64Type(), indices[i]);
    Value contrib = arith::MulIOp::create(
        b, loc, idxI64, i64Cst(b, loc, strides[i] * elemBytes));
    byteOff = arith::AddIOp::create(b, loc, byteOff, contrib);
  }
  return arith::AddIOp::create(b, loc, ptrI64, byteOff);
}

/// Emit triton_cpu.dma_enqueue_2d for a 2D tile.
static void emitDmaEnqueue(OpBuilder &b, Location loc,
                           Value spmAddr, Value dramAddr,
                           VectorType vecTy, MemRefType memRefTy) {
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  auto shape = vecTy.getShape();
  int64_t rows = (shape.size() >= 2) ? shape[0] : 1;
  int64_t cols = (shape.size() >= 2) ? shape[1] : shape[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  (void)memRefTy.getStridesAndOffset(strides, offset);
  int64_t srcStrideBytes =
      (shape.size() >= 2 ? strides[0] : cols) * elemBytes;

  triton::cpu::DmaEnqueue2DOp::create(
      b, loc,
      spmAddr, dramAddr,
      i64Cst(b, loc, cols * elemBytes),   // width
      i64Cst(b, loc, rows),               // height
      i64Cst(b, loc, srcStrideBytes),      // src_stride
      i64Cst(b, loc, cols * elemBytes));   // dst_stride (packed)
}

/// Create a memref in SPM address space (3) via reinterpret_cast, and
/// emit a vector.transfer_read from it.
///
/// We create a flat 1-element memref<1xi8, 3> from the SPM base pointer
/// using unrealized_conversion_cast (i64 → memref), then reinterpret_cast
/// it to the desired shape.  The LLVM lowering will turn this into an
/// inttoptr + load from address space 3.
static Value emitSpmRead(OpBuilder &b, Location loc,
                         Value spmAddr, VectorType vecTy) {
  auto shape = vecTy.getShape();
  int64_t cols = (shape.size() >= 2) ? shape[1] : shape[0];
  auto elemTy = vecTy.getElementType();

  // Build the target SPM memref type: packed layout, address space 3.
  SmallVector<int64_t> memShape(shape.begin(), shape.end());
  SmallVector<int64_t> memStrides;
  if (vecTy.getRank() == 1) {
    memStrides = {1};
  } else {
    memStrides = {cols, 1};
  }
  auto layout = StridedLayoutAttr::get(b.getContext(), 0, memStrides);
  auto spmMemRefTy = MemRefType::get(
      memShape, elemTy, layout,
      b.getI64IntegerAttr(SPM_ADDR_SPACE));

  // Create a 0-d base memref in address space 3 from the i64 address.
  // unrealized_conversion_cast will be resolved during LLVM lowering.
  auto baseMemRefTy = MemRefType::get(
      {}, elemTy, /*layout=*/nullptr,
      b.getI64IntegerAttr(SPM_ADDR_SPACE));
  Value baseMemRef = UnrealizedConversionCastOp::create(
      b, loc, baseMemRefTy, spmAddr)->getResult(0);

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
  auto padVal = arith::ConstantOp::create(
      b, loc, elemTy, b.getZeroAttr(elemTy));
  return vector::TransferReadOp::create(
      b, loc, vecTy, spmView, zeroIndices, padVal,
      SmallVector<bool>(vecTy.getRank(), true));
}

/// Emit a vector.transfer_write to an SPM address.
static void emitSpmWrite(OpBuilder &b, Location loc,
                         Value spmAddr, Value value) {
  auto vecTy = cast<VectorType>(value.getType());
  auto shape = vecTy.getShape();
  int64_t cols = (shape.size() >= 2) ? shape[1] : shape[0];
  auto elemTy = vecTy.getElementType();

  SmallVector<int64_t> memShape(shape.begin(), shape.end());
  SmallVector<int64_t> memStrides;
  if (vecTy.getRank() == 1) {
    memStrides = {1};
  } else {
    memStrides = {cols, 1};
  }
  auto layout = StridedLayoutAttr::get(b.getContext(), 0, memStrides);
  auto spmMemRefTy = MemRefType::get(
      memShape, elemTy, layout,
      b.getI64IntegerAttr(SPM_ADDR_SPACE));

  auto baseMemRefTy = MemRefType::get(
      {}, elemTy, /*layout=*/nullptr,
      b.getI64IntegerAttr(SPM_ADDR_SPACE));
  Value baseMemRef = UnrealizedConversionCastOp::create(
      b, loc, baseMemRefTy, spmAddr)->getResult(0);

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
  vector::TransferWriteOp::create(
      b, loc, value, spmView, zeroIndices,
      SmallVector<bool>(vecTy.getRank(), true));
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

/// Check if a transfer_read feeds a dot-like operation.
static bool readFeedsDot(vector::TransferReadOp readOp) {
  for (auto *user : readOp->getUsers()) {
    if (isa<vector::ContractionOp>(user) ||
        isa<triton::cpu::DotOp>(user))
      return true;
  }
  return false;
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

static bool isGemmContract(vector::ContractionOp op) {
  auto iterTypes = op.getIteratorTypes().getValue();
  if (iterTypes.size() != 3)
    return false;
  using IT = vector::IteratorType;
  auto get = [](Attribute a) {
    return cast<vector::IteratorTypeAttr>(a).getValue();
  };
  if (get(iterTypes[0]) != IT::parallel ||
      get(iterTypes[1]) != IT::parallel ||
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
analyzeGemmContract(scf::ForOp forOp,
                    vector::TransferReadOp readA,
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
static bool transformFusedMicroGemmLoop(scf::ForOp forOp,
                                         ArrayRef<TiledLoadInfo> dotLoads,
                                         int64_t spmBase, int64_t spmSize,
                                         int64_t microM,
                                         int64_t requestedWindowK) {
  if (dotLoads.size() != 2 || microM <= 0 || requestedWindowK <= 0)
    return false;

  auto lbCst = getConstantIntValue(forOp.getLowerBound());
  auto ubCst = getConstantIntValue(forOp.getUpperBound());
  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!lbCst || !ubCst || !stepCst || *stepCst <= 0 ||
      (*ubCst - *lbCst) % *stepCst != 0)
    return false;

  int64_t trips = (*ubCst - *lbCst) / *stepCst;
  if (trips <= 0)
    return false;
  int64_t windowK = chooseWindowK(trips, requestedWindowK);

  TiledLoadInfo loadA = dotLoads[0];
  TiledLoadInfo loadB = dotLoads[1];
  auto contractInfo =
      analyzeGemmContract(forOp, loadA.readOp, loadB.readOp);
  if (!contractInfo) {
    contractInfo =
        analyzeGemmContract(forOp, loadB.readOp, loadA.readOp);
    if (!contractInfo)
      return false;
    std::swap(loadA, loadB);
  }

  vector::ContractionOp contractOp = contractInfo->contractOp;
  unsigned accIdx = contractInfo->accIdx;
  VectorType accTy = contractInfo->accTy;
  auto aTy = dyn_cast<VectorType>(loadA.readOp.getType());
  auto bTy = dyn_cast<VectorType>(loadB.readOp.getType());
  if (!aTy || !bTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      accTy.getRank() != 2)
    return false;

  int64_t BM = accTy.getDimSize(0);
  int64_t BN = accTy.getDimSize(1);
  int64_t BK = aTy.getDimSize(1);
  if (BM < microM || BM % microM != 0 || aTy.getDimSize(0) != BM ||
      bTy.getDimSize(0) != BK || bTy.getDimSize(1) != BN)
    return false;

  // This fused schedule only materializes the accumulator result.  The block
  // pointer loop results in the matmul kernel are dead; if a future pattern
  // uses them, fall back to the conservative double-buffer path.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    if (i != accIdx && !forOp.getResult(i).use_empty())
      return false;

  auto memRefTyA = cast<MemRefType>(loadA.readOp.getBase().getType());
  auto memRefTyB = cast<MemRefType>(loadB.readOp.getBase().getType());
  SmallVector<int64_t> stridesA, stridesB;
  if (!getStaticStrides(memRefTyA, stridesA) ||
      !getStaticStrides(memRefTyB, stridesB) ||
      stridesA.size() < 2 || stridesB.size() < 2)
    return false;

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
  auto allocAMicro = spmLayout.alloc(microABytes, /*alignment=*/1,
                                     SPMSpaceManager::Lifetime::Loop);
  auto allocAcc = spmLayout.alloc(accBytes, /*alignment=*/1,
                                  SPMSpaceManager::Lifetime::Loop);
  if (!allocBWindow || !allocAMicro || !allocAcc)
    return false;

  int64_t addrBWindow = allocBWindow->address;
  int64_t addrAMicro = allocAMicro->address;
  int64_t addrAcc = allocAcc->address;

  Location loc = forOp.getLoc();
  OpBuilder b(forOp);

  Value dramAddrA = computePrologueDramAddr(b, loc, loadA.readOp, forOp);
  Value dramAddrB = computePrologueDramAddr(b, loc, loadB.readOp, forOp);
  if (!dramAddrA || !dramAddrB)
    return false;

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

  auto winFor = scf::ForOp::create(
      b, loc, i64Cst(b, loc, 0), i64Cst(b, loc, trips),
      i64Cst(b, loc, windowK));
  Block *winBody = winFor.getBody();
  if (!winBody->empty() && winBody->mightHaveTerminator())
    winBody->getTerminator()->erase();

  b.setInsertionPointToStart(winBody);
  Value winIter = winFor.getInductionVar();

  // Stage the resident B window once.  windowK is capped to the DMA queue
  // depth default (4) by the caller/env setting.
  auto bStageFor = scf::ForOp::create(
      b, loc, i64Cst(b, loc, 0), i64Cst(b, loc, windowK),
      i64Cst(b, loc, 1));
  Block *bStageBody = bStageFor.getBody();
  if (!bStageBody->empty() && bStageBody->mightHaveTerminator())
    bStageBody->getTerminator()->erase();

  b.setInsertionPointToStart(bStageBody);
  Value bLocal = bStageFor.getInductionVar();
  Value bAbsIter = arith::AddIOp::create(b, loc, winIter, bLocal);
  Value bDram = arith::AddIOp::create(
      b, loc, dramAddrB,
      arith::MulIOp::create(b, loc, bAbsIter,
                            i64Cst(b, loc, stepBytesB)));
  Value bSpm = arith::AddIOp::create(
      b, loc, i64Cst(b, loc, addrBWindow),
      arith::MulIOp::create(b, loc, bLocal,
                            i64Cst(b, loc, loadB.tileBytes)));
  emitDmaEnqueue(b, loc, bSpm, bDram, bTy, memRefTyB);
  scf::YieldOp::create(b, loc);

  b.setInsertionPointAfter(bStageFor);
  triton::cpu::DmaWaitOp::create(b, loc);

  for (int64_t mOff = 0; mOff < BM; mOff += microM) {
    Value accAddr = i64Cst(b, loc, addrAcc + mOff * accRowBytes);
    Value microInit = emitSpmRead(b, loc, accAddr, microAccTy);

    auto kFor = scf::ForOp::create(
        b, loc, i64Cst(b, loc, 0), i64Cst(b, loc, windowK),
        i64Cst(b, loc, 1), ValueRange{microInit});
    Block *kBody = kFor.getBody();
    if (!kBody->empty() && kBody->mightHaveTerminator())
      kBody->getTerminator()->erase();

    b.setInsertionPointToStart(kBody);
    Value kLocal = kFor.getInductionVar();
    Value kAbsIter = arith::AddIOp::create(b, loc, winIter, kLocal);
    Value aDram = arith::AddIOp::create(
        b, loc, dramAddrA,
        arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(b, loc, kAbsIter,
                                  i64Cst(b, loc, stepBytesA)),
            i64Cst(b, loc, mOff * rowBytesA)));
    emitDmaEnqueue(b, loc, i64Cst(b, loc, addrAMicro), aDram,
                   microATy, memRefTyA);
    triton::cpu::DmaWaitOp::create(b, loc);

    Value aVal = emitSpmRead(b, loc, i64Cst(b, loc, addrAMicro), microATy);
    Value bReadAddr = arith::AddIOp::create(
        b, loc, i64Cst(b, loc, addrBWindow),
        arith::MulIOp::create(b, loc, kLocal,
                              i64Cst(b, loc, loadB.tileBytes)));
    Value bVal = emitSpmRead(b, loc, bReadAddr, bTy);
    Value microAcc = kFor.getRegionIterArgs()[0];
    Value contracted = vector::ContractionOp::create(
        b, loc, microAccTy, aVal, bVal, microAcc, mapsAttr, iterAttr);
    scf::YieldOp::create(b, loc, contracted);

    b.setInsertionPointAfter(kFor);
    emitSpmWrite(b, loc, accAddr, kFor.getResult(0));
  }

  scf::YieldOp::create(b, loc);
  b.setInsertionPointAfter(winFor);

  Value finalAcc = emitSpmRead(b, loc, i64Cst(b, loc, addrAcc), accTy);
  forOp.getResult(accIdx).replaceAllUsesWith(finalAcc);
  forOp.erase();

  (void)contractOp;
  return true;
}

/// Transform a GEMM K-loop with double-buffered SPM.
/// `dotLoads` must have exactly 2 entries (A and B tile loads).
/// Returns true on success.
static bool transformGemmLoop(scf::ForOp forOp,
                              ArrayRef<TiledLoadInfo> dotLoads,
                              int64_t spmBase, int64_t spmSize) {
  if (dotLoads.size() != 2)
    return false;

  auto &loadA = dotLoads[0];
  auto &loadB = dotLoads[1];
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
  auto allocA0 = spmLayout.alloc(tileA, /*alignment=*/1,
                                 SPMSpaceManager::Lifetime::Loop);
  auto allocA1 = spmLayout.alloc(tileA, /*alignment=*/1,
                                 SPMSpaceManager::Lifetime::Loop);
  auto allocB0 = spmLayout.alloc(tileB, /*alignment=*/1,
                                 SPMSpaceManager::Lifetime::Loop);
  auto allocB1 = spmLayout.alloc(tileB, /*alignment=*/1,
                                 SPMSpaceManager::Lifetime::Loop);
  if (!allocA0 || !allocA1 || !allocB0 || !allocB1)
    return false;

  int64_t addrA0 = allocA0->address;
  int64_t addrA1 = allocA1->address;
  int64_t addrB0 = allocB0->address;
  int64_t addrB1 = allocB1->address;

  Location loc = forOp.getLoc();
  OpBuilder b(forOp);

  auto readA = loadA.readOp;
  auto readB = loadB.readOp;
  auto memRefTyA = cast<MemRefType>(readA.getBase().getType());
  auto memRefTyB = cast<MemRefType>(readB.getBase().getType());

  // --- Prologue: DMA first tiles into buffer 0 ---
  Value dramAddrA = computePrologueDramAddr(b, loc, readA, forOp);
  Value dramAddrB = computePrologueDramAddr(b, loc, readB, forOp);
  if (!dramAddrA || !dramAddrB)
    return false;

  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrA0), dramAddrA,
                 loadA.vecTy, memRefTyA);
  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrB0), dramAddrB,
                 loadB.vecTy, memRefTyB);
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
  auto newForOp = scf::ForOp::create(
      b, loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newInitArgs);

  // The new loop's body block has: iv, then one block arg per init arg.
  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  // Map old block args to new block args.
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  unsigned numOldArgs = forOp.getRegionIterArgs().size();
  for (unsigned i = 0; i < numOldArgs; ++i)
    mapping.map(forOp.getRegionIterArgs()[i],
                newForOp.getRegionIterArgs()[i]);

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
  Value isZero = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::eq, bufIdx, zero);

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
  Value hasNext = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::slt, nextIv, ub);

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
    b.clone(op, mapping);
  }

  vector::TransferReadOp clonedReadA = nullptr, clonedReadB = nullptr;
  newBody->walk([&](vector::TransferReadOp clonedRead) {
    if (clonedRead.getBase() == mapping.lookupOrDefault(readA.getBase()) &&
        clonedRead.getLoc() == readA.getLoc() && !clonedReadA)
      clonedReadA = clonedRead;
    else if (clonedRead.getBase() == mapping.lookupOrDefault(readB.getBase()) &&
             clonedRead.getLoc() == readB.getLoc() && !clonedReadB)
      clonedReadB = clonedRead;
  });

  if (!clonedReadA || !clonedReadB)
    return false;

  b.setInsertionPoint(clonedReadA);
  Value spmValA = emitSpmRead(b, loc, spmACur, loadA.vecTy);
  clonedReadA.replaceAllUsesWith(spmValA);
  clonedReadA->erase();

  b.setInsertionPoint(clonedReadB);
  Value spmValB = emitSpmRead(b, loc, spmBCur, loadB.vecTy);
  clonedReadB.replaceAllUsesWith(spmValB);
  clonedReadB->erase();

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
  forOp.erase();

  return true;
}

//===----------------------------------------------------------------------===//
// Reduction single-buffer prefetch transformation.
//
// For loops with a single tiled load (not feeding a dot), insert a
// single-buffer DMA prefetch: DMA the next chunk while computing the
// current one.
//
// Input:
//   scf.for %i = 0 to N step BS {
//     %x = transfer_read %mem[%i, ...] : vector<BS×f32>
//     ... reduce ...
//   }
//
// Output:
//   // Prologue: DMA first chunk
//   dma_enqueue_2d(spm_buf, dram_start, ...)
//   dma_wait
//
//   scf.for %i = 0 to N step BS {
//     // Prefetch next chunk (async)
//     scf.if (%i + BS < N) {
//       dma_enqueue_2d(spm_buf, dram_next, ...)
//     }
//     // Read current from SPM
//     %x = transfer_read spm_buf : memref<..., 3>
//     ... reduce ...
//     dma_wait
//   }
//===----------------------------------------------------------------------===//

static bool transformReductionLoop(scf::ForOp forOp,
                                   const TiledLoadInfo &load,
                                   int64_t spmBase, int64_t spmSize) {
  SPMSpaceManager spmLayout(spmBase, spmSize);
  auto allocBuf = spmLayout.alloc(load.tileBytes, /*alignment=*/1,
                                  SPMSpaceManager::Lifetime::Loop);
  if (!allocBuf)
    return false;

  auto lbCst = getConstantIntValue(forOp.getLowerBound());
  auto ubCst = getConstantIntValue(forOp.getUpperBound());
  auto stepCst = getConstantIntValue(forOp.getStep());
  if (!lbCst || !ubCst || !stepCst || (*ubCst - *lbCst) % *stepCst != 0)
    return false;

  int64_t addrBuf = allocBuf->address;
  Location loc = forOp.getLoc();
  OpBuilder b(forOp);

  auto readOp = load.readOp;
  auto memRefTy = cast<MemRefType>(readOp.getBase().getType());

  // Prologue: DMA first chunk.
  Value dramAddr = computePrologueDramAddr(b, loc, readOp, forOp);
  if (!dramAddr)
    return false;

  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrBuf), dramAddr,
                 load.vecTy, memRefTy);
  triton::cpu::DmaWaitOp::create(b, loc);

  // Rebuild loop (no new iter_args needed for single-buffer).
  // We insert prefetch + SPM read inside the existing loop body.
  // To keep things simple, clone the loop like we did for GEMM.
  SmallVector<Value> initArgs(forOp.getInitArgs());
  auto newForOp = scf::ForOp::create(
      b, loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      initArgs);

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i)
    mapping.map(forOp.getRegionIterArgs()[i],
                newForOp.getRegionIterArgs()[i]);

  b.setInsertionPointToStart(newBody);
  if (!newBody->empty() && newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  // Clone the original body, replacing the transfer_read with SPM read.
  // SPM read happens FIRST, before prefetch, to avoid single-buffer race:
  // the current chunk must be fully read before prefetching overwrites it.
  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (&op == readOp.getOperation()) {
      Value spmVal = emitSpmRead(b, loc, i64Cst(b, loc, addrBuf), load.vecTy);
      mapping.map(readOp.getResult(), spmVal);
      continue;
    }
    b.clone(op, mapping);
  }

  // --- Prefetch next chunk (AFTER compute to avoid single-buffer race) ---
  Value iv = newForOp.getInductionVar();
  Value step = newForOp.getStep();
  Value ub = newForOp.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  SmallVector<int64_t> strides;
  int64_t offset;
  (void)memRefTy.getStridesAndOffset(strides, offset);

  // Find which dimension the loop IV indexes by matching the original
  // transfer_read indices against the loop induction variable.
  int64_t ivStride = strides.empty() ? 1 : strides[0];
  Value origIv = forOp.getInductionVar();
  for (unsigned i = 0; i < readOp.getIndices().size(); ++i) {
    if (readOp.getIndices()[i] == origIv) {
      ivStride = strides[i];
      break;
    }
  }

  Value lbInLoop = newForOp.getLowerBound();
  Value nextOff = arith::SubIOp::create(b, loc, nextIv, lbInLoop);
  Value nextOffI64 = toI64(b, loc, nextOff);
  Value nextByteOff = arith::MulIOp::create(
      b, loc, nextOffI64, i64Cst(b, loc, ivStride * elemBytes));
  Value nextDram = arith::AddIOp::create(b, loc, dramAddr, nextByteOff);

  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrBuf), nextDram,
                 load.vecTy, memRefTy);
  b.setInsertionPointAfter(ifOp);

  // DMA wait at end of body.
  triton::cpu::DmaWaitOp::create(b, loc);

  // Yield.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> yieldVals;
  for (auto val : oldYield.getOperands())
    yieldVals.push_back(mapping.lookupOrDefault(val));
  scf::YieldOp::create(b, loc, yieldVals);

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
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
  ConvertMemoryToSPM(int64_t spmBase_, int64_t spmSize_,
                     int64_t microM_, int64_t windowK_) {
    this->spmBase = spmBase_;
    this->spmSize = spmSize_;
    this->microM = microM_;
    this->windowK = windowK_;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect scf.for ops (avoid modifying while iterating).
    SmallVector<scf::ForOp> forOps;
    mod.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    for (auto forOp : forOps) {
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
        if (!transformFusedMicroGemmLoop(forOp, dotLoads, spmBase, spmSize,
                                         microM, windowK))
          transformGemmLoop(forOp, dotLoads, spmBase, spmSize);
      } else if (dotLoads.empty() && nonDotLoads.size() == 1) {
        transformReductionLoop(forOp, nonDotLoads[0], spmBase, spmSize);
      }
      // Otherwise: leave unchanged (cache path).
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
createConvertMemoryToSPM(int64_t spmBase, int64_t spmSize,
                         int64_t microM, int64_t windowK) {
  return std::make_unique<ConvertMemoryToSPM>(
      spmBase, spmSize, microM, windowK);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
