//===- SPMTensorPlacement.cpp - Three-tier SPM placement analysis ---------===//
//
// This pass is the pipeline hook for selecting tensor placement tiers before
// ConvertMemoryToSPM lowers tiled loads to DMA+SPM staging.
//
// M4 annotates eligible function arguments with tt_cpu.spm_tier.  Later
// milestones add sidecar export and richer scalar-reuse analysis.
//
//===----------------------------------------------------------------------===//

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_SPMTENSORPLACEMENT
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

static constexpr StringLiteral kSPMTierAttrName = "tt_cpu.spm_tier";

enum class SPMTier : int32_t {
  SPMResident = 1,
  CacheableDram = 2,
  UncacheableDmaBuffer = 3,
};

static bool getEnvBool(StringRef name, bool defaultValue = false) {
  if (const char *value = std::getenv(name.str().c_str())) {
    StringRef text(value);
    return !(text.empty() || text == "0" || text.equals_insensitive("false") ||
             text.equals_insensitive("no") || text.equals_insensitive("off"));
  }
  return defaultValue;
}

/// Get static strides from a memref type. Returns false if dynamic.
static bool getStaticStrides(MemRefType ty, SmallVectorImpl<int64_t> &strides) {
  int64_t offset;
  if (failed(ty.getStridesAndOffset(strides, offset)))
    return false;
  for (auto stride : strides)
    if (ShapedType::isDynamic(stride))
      return false;
  return true;
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
  for (Operation *user : value.getUsers()) {
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

static bool readFeedsLoopLocalGemm(vector::TransferReadOp readOp) {
  scf::ForOp parentFor = readOp->getParentOfType<scf::ForOp>();
  if (!parentFor)
    return false;

  for (Operation *user : readOp->getUsers()) {
    auto contract = dyn_cast<vector::ContractionOp>(user);
    if (!contract || !isGemmContract(contract))
      continue;

    auto lhsRead =
        getTransferReadThroughShapePreservingCasts(contract.getLhs());
    auto rhsRead =
        getTransferReadThroughShapePreservingCasts(contract.getRhs());
    if (!lhsRead || !rhsRead)
      continue;
    if (lhsRead->getParentOfType<scf::ForOp>() == parentFor &&
        rhsRead->getParentOfType<scf::ForOp>() == parentFor)
      return true;
  }

  return false;
}

static bool contractOperandTracesToRead(vector::ContractionOp contract,
                                        vector::TransferReadOp readOp) {
  for (Value operand : {contract.getLhs(), contract.getRhs()}) {
    auto operandRead = getTransferReadThroughShapePreservingCasts(operand);
    if (operandRead == readOp)
      return true;
  }
  return false;
}

static bool readFeedsAttentionV2OuterQ(vector::TransferReadOp readOp) {
  if (readOp->getParentOfType<scf::ForOp>())
    return false;

  auto vecTy = dyn_cast<VectorType>(readOp.getType());
  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!vecTy || vecTy.getRank() != 2 || !memRefTy || memRefTy.getRank() != 2)
    return false;

  for (Operation *user : readOp.getResult().getUsers()) {
    SmallVector<Operation *, 4> stack{user};
    while (!stack.empty()) {
      Operation *cur = stack.pop_back_val();
      if (auto contract = dyn_cast<vector::ContractionOp>(cur)) {
        if (!isGemmContract(contract) ||
            !contractOperandTracesToRead(contract, readOp))
          continue;
        if (contract->getParentOfType<scf::ForOp>())
          return true;
        continue;
      }

      if (!isShapePreservingCastLikeOp(cur))
        continue;
      for (Operation *next : cur->getResult(0).getUsers())
        stack.push_back(next);
    }
  }
  return false;
}

static bool hasLocalConsumerOutput(vector::ContractionOp contract) {
  if (contract->getParentOfType<scf::ForOp>())
    return false;
  if (contract.getResult().use_empty())
    return false;
  for (Operation *user : contract.getResult().getUsers())
    if (isa<vector::TransferWriteOp, scf::YieldOp>(user))
      return false;
  return true;
}

static bool readFeedsAttentionQKTile(vector::TransferReadOp readOp) {
  if (!getEnvBool("TRITON_SPM_ATTENTION_QK_TILE", false))
    return false;
  if (readOp->getParentOfType<scf::ForOp>())
    return false;

  auto vecTy = dyn_cast<VectorType>(readOp.getType());
  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!vecTy || vecTy.getRank() != 2 || !memRefTy || memRefTy.getRank() != 2)
    return false;

  for (Operation *user : readOp.getResult().getUsers()) {
    SmallVector<Operation *, 4> stack{user};
    while (!stack.empty()) {
      Operation *cur = stack.pop_back_val();
      if (auto contract = dyn_cast<vector::ContractionOp>(cur)) {
        if (isGemmContract(contract) &&
            contractOperandTracesToRead(contract, readOp) &&
            hasLocalConsumerOutput(contract))
          return true;
        continue;
      }

      if (!isShapePreservingCastLikeOp(cur))
        continue;
      for (Operation *next : cur->getResult(0).getUsers())
        stack.push_back(next);
    }
  }
  return false;
}

static BlockArgument traceFunctionArgument(Value value,
                                           FunctionOpInterface funcOp,
                                           unsigned depth = 0) {
  if (depth > 8)
    return nullptr;

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg) {
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

static bool isEligibleTiledRead(vector::TransferReadOp readOp,
                                FunctionOpInterface funcOp, BlockArgument &arg,
                                bool allowOutsideLoop = false) {
  auto vecTy = dyn_cast<VectorType>(readOp.getType());
  if (!vecTy || vecTy.getRank() < 1)
    return false;

  auto memRefTy = dyn_cast<MemRefType>(readOp.getBase().getType());
  if (!memRefTy)
    return false;

  SmallVector<int64_t> strides;
  if (!getStaticStrides(memRefTy, strides))
    return false;

  if (!allowOutsideLoop && !readOp->getParentOfType<scf::ForOp>())
    return false;

  arg = traceFunctionArgument(readOp.getBase(), funcOp);
  return static_cast<bool>(arg);
}

static bool isVector1Read(vector::TransferReadOp readOp) {
  auto vecTy = dyn_cast<VectorType>(readOp.getType());
  return vecTy && vecTy.getNumElements() == 1;
}

static bool readFeedsDot(vector::TransferReadOp readOp) {
  return valueFeedsDot(readOp.getResult());
}

static bool hasScalarReuse(FunctionOpInterface funcOp, BlockArgument arg) {
  bool hasReuse = false;
  funcOp->walk([&](Operation *op) {
    if (hasReuse)
      return WalkResult::interrupt();

    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      if (traceFunctionArgument(loadOp.getMemRef(), funcOp) == arg) {
        hasReuse = true;
        return WalkResult::interrupt();
      }
    }

    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      if (traceFunctionArgument(readOp.getBase(), funcOp) == arg &&
          isVector1Read(readOp)) {
        hasReuse = true;
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return hasReuse;
}

static SPMTier chooseTier(bool scalarReuse, bool cacheableDmaSource) {
  if (cacheableDmaSource)
    return SPMTier::CacheableDram;
  if (!scalarReuse)
    return SPMTier::UncacheableDmaBuffer;

  // Tier 1 is not implemented in the MVP.  Even when the whole tensor fits in
  // SPM, use Tier 2 as the effective tier until resident SPM support lands.
  return SPMTier::CacheableDram;
}

static LogicalResult
writeTierSidecar(FunctionOpInterface funcOp,
                 ArrayRef<std::pair<unsigned, int32_t>> tiers) {
  const char *auxDir = std::getenv("KERNEL_AUX_FILE_DIR");
  if (!auxDir || StringRef(auxDir).empty())
    return success();

  SmallString<256> path(auxDir);
  std::string filename = funcOp.getName().str() + "_tiers.json";
  llvm::sys::path::append(path, filename);

  std::error_code error;
  llvm::raw_fd_ostream os(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return funcOp.emitError("failed to write SPM tier sidecar '")
           << path << "': " << error.message();

  os << "{\n";
  for (auto [index, tier] : llvm::enumerate(tiers)) {
    os << "  \"" << tier.first << "\": " << tier.second;
    if (index + 1 != tiers.size())
      os << ",";
    os << "\n";
  }
  os << "}\n";

  return success();
}

struct SPMTensorPlacement
    : public triton::cpu::impl::SPMTensorPlacementBase<SPMTensorPlacement> {
  SPMTensorPlacement() = default;
  explicit SPMTensorPlacement(bool enableReductions_) {
    this->enableReductions = enableReductions_;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *context = mod.getContext();
    auto i32Ty = IntegerType::get(context, 32);
    bool failedSidecarWrite = false;

    mod.walk([&](FunctionOpInterface funcOp) {
      if (failedSidecarWrite)
        return;

      llvm::DenseSet<unsigned> candidateArgs;
      llvm::DenseSet<unsigned> cacheableDmaSourceArgs;
      funcOp->walk([&](vector::TransferReadOp readOp) {
        bool feedsDot = readFeedsDot(readOp);
        bool attentionOuterQ = readFeedsAttentionV2OuterQ(readOp);
        bool attentionQKTile = readFeedsAttentionQKTile(readOp);
        if (feedsDot) {
          if (!readFeedsLoopLocalGemm(readOp) && !attentionOuterQ &&
              !attentionQKTile)
            return;
        } else if (!enableReductions) {
          return;
        }

        BlockArgument arg;
        if (isEligibleTiledRead(
                readOp, funcOp, arg,
                /*allowOutsideLoop=*/attentionOuterQ || attentionQKTile)) {
          candidateArgs.insert(arg.getArgNumber());
          if (attentionOuterQ)
            cacheableDmaSourceArgs.insert(arg.getArgNumber());
        }
      });

      SmallVector<std::pair<unsigned, int32_t>> tiers;
      for (unsigned argIndex : candidateArgs) {
        BlockArgument arg = funcOp.getArgument(argIndex);
        bool scalarReuse = hasScalarReuse(funcOp, arg);
        SPMTier tier =
            chooseTier(scalarReuse, cacheableDmaSourceArgs.contains(argIndex));
        int32_t tierValue = static_cast<int32_t>(tier);
        funcOp.setArgAttr(argIndex, kSPMTierAttrName,
                          IntegerAttr::get(i32Ty, tierValue));
        tiers.push_back({argIndex, tierValue});
      }

      std::sort(tiers.begin(), tiers.end());

      // AOT kernels are public functions with bodies.  Internal helpers do not
      // need a sidecar, and tests without KERNEL_AUX_FILE_DIR remain pure.
      if (funcOp.getVisibility() == SymbolTable::Visibility::Public &&
          !funcOp.getFunctionBody().empty() &&
          failed(writeTierSidecar(funcOp, tiers)))
        failedSidecarWrite = true;
    });

    if (failedSidecarWrite)
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createSPMTensorPlacement() {
  return std::make_unique<SPMTensorPlacement>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createSPMTensorPlacement(bool enableReductions) {
  return std::make_unique<SPMTensorPlacement>(enableReductions);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
