//===- SplitLargeContract.cpp - Split large vector.contract into micro-tiles ==//
//
// This pass splits K-loops whose vector.contract accumulator M dimension
// exceeds a threshold.  The single K-loop is replaced by ceil(M / MICRO_M)
// sequential K-loops, each accumulating only MICRO_M rows.  Each loop gets
// its own small loop-carried accumulator that fits in registers.
//
// Runs AFTER ConvertMemoryToSPM (DMA granularity unaffected) and BEFORE
// LLVM lowering.
//
//===----------------------------------------------------------------------===//

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_SPLITLARGECONTRACT
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

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

struct LoopContractInfo {
  scf::ForOp forOp;
  vector::ContractionOp contractOp;
  unsigned accIdx; // index into getRegionIterArgs()
  VectorType accTy;
};

static std::optional<LoopContractInfo>
analyzeLoopContract(vector::ContractionOp contractOp) {
  auto forOp = dyn_cast<scf::ForOp>(contractOp->getParentOp());
  if (!forOp)
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
  return LoopContractInfo{forOp, contractOp, idx, accTy};
}

static bool splitLoopContract(const LoopContractInfo &info, int64_t microM) {
  scf::ForOp origFor = info.forOp;
  vector::ContractionOp origContract = info.contractOp;
  unsigned accIdx = info.accIdx;
  VectorType accTy = info.accTy;

  int64_t M = accTy.getDimSize(0);
  int64_t N = accTy.getDimSize(1);
  Type elemTy = accTy.getElementType();
  Location loc = origFor.getLoc();
  MLIRContext *ctx = origFor.getContext();

  auto aMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
  auto bMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
  auto cMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);

  OpBuilder b(origFor);

  // Current init args — will be updated as micro-loops chain.
  SmallVector<Value> curInitArgs(origFor.getInitArgs());
  Value fullAcc = curInitArgs[accIdx];
  scf::ForOp lastFor;

  for (int64_t mOff = 0; mOff < M; mOff += microM) {
    int64_t curM = std::min(microM, M - mOff);
    auto microAccTy = VectorType::get({curM, N}, elemTy);

    // Extract micro-tile init accumulator.
    Value microInit = vector::ExtractStridedSliceOp::create(
        b, loc, fullAcc,
        /*offsets=*/{mOff, 0}, /*sizes=*/{curM, N}, /*strides=*/{1, 1});

    // Replace acc init with micro-tile.
    SmallVector<Value> loopInits(curInitArgs);
    loopInits[accIdx] = microInit;

    // Adjust the init arg types: the acc slot changes type.
    auto newFor = scf::ForOp::create(
        b, loc, origFor.getLowerBound(), origFor.getUpperBound(),
        origFor.getStep(), loopInits);

    Block *newBody = newFor.getBody();
    Block *oldBody = origFor.getBody();

    if (!newBody->empty() && newBody->mightHaveTerminator())
      newBody->getTerminator()->erase();

    IRMapping mapping;
    mapping.map(origFor.getInductionVar(), newFor.getInductionVar());
    for (unsigned i = 0; i < origFor.getRegionIterArgs().size(); ++i)
      mapping.map(origFor.getRegionIterArgs()[i],
                  newFor.getRegionIterArgs()[i]);

    b.setInsertionPointToStart(newBody);

    auto mapsAttr = b.getAffineMapArrayAttr({aMap, bMap, cMap});
    auto iterAttr = b.getArrayAttr(
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});

    for (auto &op : oldBody->getOperations()) {
      if (isa<scf::YieldOp>(op))
        continue;

      if (&op == origContract.getOperation()) {
        Value mappedA = mapping.lookupOrDefault(origContract.getLhs());
        Value mappedB = mapping.lookupOrDefault(origContract.getRhs());
        Value mappedAcc = mapping.lookupOrDefault(origContract.getAcc());

        auto origATy = cast<VectorType>(mappedA.getType());
        int64_t K = origATy.getDimSize(1);
        Value aSlice = vector::ExtractStridedSliceOp::create(
            b, loc, mappedA,
            /*offsets=*/{mOff, 0}, /*sizes=*/{curM, K}, /*strides=*/{1, 1});

        Value microResult = vector::ContractionOp::create(
            b, loc, microAccTy, aSlice, mappedB, mappedAcc,
            mapsAttr, iterAttr);

        mapping.map(origContract.getResult(), microResult);
        continue;
      }

      b.clone(op, mapping);
    }

    // Yield.
    auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
    SmallVector<Value> yieldVals;
    for (auto val : oldYield.getOperands())
      yieldVals.push_back(mapping.lookupOrDefault(val));
    scf::YieldOp::create(b, loc, yieldVals);

    b.setInsertionPointAfter(newFor);

    // Insert micro-tile result back into full accumulator.
    Value microResult = newFor.getResult(accIdx);
    fullAcc = vector::InsertStridedSliceOp::create(
        b, loc, microResult, fullAcc,
        /*offsets=*/{mOff, 0}, /*strides=*/{1, 1});

    // Each micro-loop is a row slice of the original K-loop, so loop-carried
    // block pointers must restart from the original init args.  Only the full
    // accumulator is updated between slices.
    curInitArgs[accIdx] = fullAcc;

    lastFor = newFor;
  }

  // Replace original loop results.
  for (unsigned i = 0; i < origFor.getNumResults(); ++i) {
    if (i == accIdx)
      origFor.getResult(i).replaceAllUsesWith(fullAcc);
    else
      origFor.getResult(i).replaceAllUsesWith(lastFor.getResult(i));
  }
  origFor.erase();
  return true;
}

struct SplitLargeContract
    : public triton::cpu::impl::SplitLargeContractBase<SplitLargeContract> {
  SplitLargeContract() = default;
  SplitLargeContract(int64_t microM_) { this->microM = microM_; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    SmallVector<LoopContractInfo> targets;
    mod.walk([&](vector::ContractionOp op) {
      if (!isGemmContract(op))
        return;
      auto accTy = dyn_cast<VectorType>(op.getAcc().getType());
      if (!accTy || accTy.getRank() != 2 || accTy.getDimSize(0) <= microM)
        return;
      if (auto info = analyzeLoopContract(op))
        targets.push_back(*info);
    });

    for (auto &info : targets)
      splitLoopContract(info, microM);
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createSplitLargeContract() {
  return std::make_unique<SplitLargeContract>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createSplitLargeContract(int64_t microM) {
  return std::make_unique<SplitLargeContract>(microM);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
