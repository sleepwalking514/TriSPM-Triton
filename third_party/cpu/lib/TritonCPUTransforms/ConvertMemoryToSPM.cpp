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

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
  int64_t rows = (shape.size() >= 2) ? shape[0] : 1;
  int64_t cols = (shape.size() >= 2) ? shape[1] : shape[0];
  auto elemTy = vecTy.getElementType();
  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

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

  // Validate: 2*(tileA + tileB) must fit in SPM.
  int64_t totalSpm = 2 * (tileA + tileB);
  if (totalSpm > spmSize)
    return false;

  // SPM buffer layout (compile-time constants):
  //   spm_a0 = spmBase + 0
  //   spm_a1 = spmBase + tileA
  //   spm_b0 = spmBase + 2*tileA
  //   spm_b1 = spmBase + 2*tileA + tileB
  int64_t addrA0 = spmBase;
  int64_t addrA1 = spmBase + tileA;
  int64_t addrB0 = spmBase + 2 * tileA;
  int64_t addrB1 = spmBase + 2 * tileA + tileB;

  Location loc = forOp.getLoc();
  OpBuilder b(forOp);

  auto readA = loadA.readOp;
  auto readB = loadB.readOp;
  auto memRefTyA = cast<MemRefType>(readA.getBase().getType());
  auto memRefTyB = cast<MemRefType>(readB.getBase().getType());

  // --- Prologue: DMA first tiles into buffer 0 ---
  // Use the loop's lower bound as the induction variable for the first
  // iteration's DRAM address computation.
  Value lb = forOp.getLowerBound();
  Value origIv = forOp.getInductionVar();
  Value dramAddrA = computeDramAddr(b, loc, readA, lb, origIv);
  Value dramAddrB = computeDramAddr(b, loc, readB, lb, origIv);
  if (!dramAddrA || !dramAddrB)
    return false;

  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrA0), dramAddrA,
                 loadA.vecTy, memRefTyA);
  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrB0), dramAddrB,
                 loadB.vecTy, memRefTyB);
  triton::cpu::DmaWaitOp::create(b, loc);

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

  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue; // handle yield separately
    b.clone(op, mapping);
  }

  // Now replace the transfer_reads with SPM reads.
  // Find the cloned readA and readB.
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

  // Insert SPM reads right before the cloned reads.
  b.setInsertionPoint(clonedReadA);
  Value spmValA = emitSpmRead(b, loc, spmACur, loadA.vecTy);
  clonedReadA.replaceAllUsesWith(spmValA);
  clonedReadA->erase();

  b.setInsertionPoint(clonedReadB);
  Value spmValB = emitSpmRead(b, loc, spmBCur, loadB.vecTy);
  clonedReadB.replaceAllUsesWith(spmValB);
  clonedReadB->erase();

  // --- Prefetch next tiles (async) ---
  // Insert after the SPM reads but before dma_wait.
  // We need to check if there's a next iteration: k + step < upper_bound.
  b.setInsertionPointToEnd(newBody);

  Value iv = newForOp.getInductionVar();
  Value step = newForOp.getStep();
  Value ub = newForOp.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  // Alternate buffer for prefetch.
  Value spmANxt = arith::SelectOp::create(b, loc, isZero, spmA1, spmA0);
  Value spmBNxt = arith::SelectOp::create(b, loc, isZero, spmB1, spmB0);

  // Compute next DRAM addresses.
  // The next iteration's DRAM address = current DRAM address recomputed
  // with the next iteration's indices.  Since the original loop already
  // advances pointers, we can compute the next DRAM addr by looking at
  // what the transfer_read indices would be at k+step.
  //
  // For the MVP, we compute next DRAM addr as: current + step_bytes.
  // This works because the K-loop advances linearly.
  auto shapeA = loadA.vecTy.getShape();
  auto shapeB = loadB.vecTy.getShape();
  unsigned elemBytesA = memRefTyA.getElementType().getIntOrFloatBitWidth() / 8;
  unsigned elemBytesB = memRefTyB.getElementType().getIntOrFloatBitWidth() / 8;

  // For A[M,K]: advancing K by BLOCK_K means offset += BLOCK_K * elemBytes
  // (stride along K dimension = 1 for row-major, or strides[1])
  SmallVector<int64_t> stridesA, stridesB;
  int64_t offA, offB;
  (void)memRefTyA.getStridesAndOffset(stridesA, offA);
  (void)memRefTyB.getStridesAndOffset(stridesB, offB);

  // The K-dimension step in bytes for A: BLOCK_K * stride_k * elemBytes
  // For row-major A[M,K]: stride_k = 1, so step = BLOCK_K * elemBytes
  int64_t kDimA = (shapeA.size() >= 2) ? 1 : 0; // K is last dim for A
  int64_t stepBytesA = shapeA[kDimA] * stridesA[kDimA] * elemBytesA;

  // For B[K,N]: K is first dim, stride_k = N, step = BLOCK_K * N * elemBytes
  int64_t kDimB = 0; // K is first dim for B
  int64_t stepBytesB = shapeB[kDimB] * stridesB[kDimB] * elemBytesB;

  // Recompute current DRAM addresses inside the loop body.
  b.setInsertionPointToEnd(newBody);
  // We need the DRAM addresses that were computed for the current iteration.
  // Since we cloned the body, the original computeDramAddr logic is embedded
  // in the cloned ops.  For the next iteration, we add the step.
  //
  // Actually, we need to recompute dramAddr inside the loop.  Let's find
  // the memref.extract_aligned_pointer_as_index ops we need.
  // Simpler approach: recompute from the mapped readOp's base and indices.
  //
  // Even simpler for MVP: compute dramAddr inside the loop from the original
  // base + current indices (which are already in the cloned body).
  // But the cloned reads are erased.  We need to save the DRAM addr before
  // erasing.
  //
  // Let me restructure: compute DRAM addr BEFORE replacing reads.
  // This requires restructuring the code flow.  For now, use the linear
  // advancement approach: dramAddr_next = dramAddr_first + (k/step) * stepBytes.

  // Compute iteration index.
  Value newLb = newForOp.getLowerBound();
  Value kOffset = arith::SubIOp::create(b, loc, iv, newLb);
  // Convert to i64 for address arithmetic.
  Value kOffI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), kOffset);
  Value stepI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), step);

  // Current iteration number (0-based).
  Value iterNum = arith::DivSIOp::create(b, loc, kOffI64, stepI64);
  // Next iteration number.
  Value one = i64Cst(b, loc, 1);
  Value nextIterNum = arith::AddIOp::create(b, loc, iterNum, one);

  Value nextDramA = arith::AddIOp::create(
      b, loc, dramAddrA,
      arith::MulIOp::create(b, loc, nextIterNum, i64Cst(b, loc, stepBytesA)));
  Value nextDramB = arith::AddIOp::create(
      b, loc, dramAddrB,
      arith::MulIOp::create(b, loc, nextIterNum, i64Cst(b, loc, stepBytesB)));

  // Conditional prefetch.
  auto ifOp = scf::IfOp::create(b, loc, /*resultTypes=*/TypeRange{}, hasNext,
                                  /*withElseRegion=*/false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  emitDmaEnqueue(b, loc, spmANxt, nextDramA, loadA.vecTy, memRefTyA);
  emitDmaEnqueue(b, loc, spmBNxt, nextDramB, loadB.vecTy, memRefTyB);

  // DMA wait at end of loop body (after prefetch, before yield).
  b.setInsertionPointAfter(ifOp);
  triton::cpu::DmaWaitOp::create(b, loc);

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
  if (load.tileBytes > spmSize)
    return false;

  int64_t addrBuf = spmBase;
  Location loc = forOp.getLoc();
  OpBuilder b(forOp);

  auto readOp = load.readOp;
  auto memRefTy = cast<MemRefType>(readOp.getBase().getType());

  // Prologue: DMA first chunk.
  Value origIv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value dramAddr = computeDramAddr(b, loc, readOp, lb, origIv);
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

  // --- Prefetch next chunk (before compute) ---
  Value iv = newForOp.getInductionVar();
  Value step = newForOp.getStep();
  Value ub = newForOp.getUpperBound();
  Value nextIv = arith::AddIOp::create(b, loc, iv, step);
  Value hasNext = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::slt, nextIv, ub);

  // Compute next DRAM address.
  // Use the base DRAM address (at lb) computed in the prologue, and
  // advance by (nextIv - lb) * leading_stride * elemBytes.
  unsigned elemBytes = memRefTy.getElementType().getIntOrFloatBitWidth() / 8;
  SmallVector<int64_t> strides;
  int64_t offset;
  (void)memRefTy.getStridesAndOffset(strides, offset);
  int64_t leadingStride = strides.empty() ? 1 : strides[0];

  // dramAddr is the DRAM address at iteration lb.
  // nextDram = dramAddr + (nextIv - lb) * leadingStride * elemBytes
  Value lbInLoop = newForOp.getLowerBound();
  Value nextOff = arith::SubIOp::create(b, loc, nextIv, lbInLoop);
  Value nextOffI64 = arith::IndexCastOp::create(b, loc, b.getI64Type(), nextOff);
  Value nextByteOff = arith::MulIOp::create(
      b, loc, nextOffI64, i64Cst(b, loc, leadingStride * elemBytes));
  Value nextDram = arith::AddIOp::create(b, loc, dramAddr, nextByteOff);

  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, hasNext, false);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  emitDmaEnqueue(b, loc, i64Cst(b, loc, addrBuf), nextDram,
                 load.vecTy, memRefTy);
  b.setInsertionPointAfter(ifOp);

  // Clone the original body, replacing the transfer_read with SPM read.
  for (auto &op : oldBody->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (&op == readOp.getOperation()) {
      // Replace with SPM read.
      Value spmVal = emitSpmRead(b, loc, i64Cst(b, loc, addrBuf), load.vecTy);
      mapping.map(readOp.getResult(), spmVal);
      continue;
    }
    b.clone(op, mapping);
  }

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

} // namespace cpu
} // namespace triton
} // namespace mlir
