#include "TypeConverter.h"

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "triton/Dialect/TritonCPU/IR/SPMAttrs.h"

using namespace mlir;
using namespace mlir::triton;

TritonCPUToLLVMTypeConverter::TritonCPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([this](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  addConversion([&](amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(type.getContext());
  });

  // SPM address-space contract.
  //
  // Phase 3 (`ConvertMemoryToSPM`) emits memrefs whose memory-space attribute
  // is the i64 integer `triton::cpu::kSPMAddressSpace` (== 3).  The base
  // `LLVMTypeConverter` already registers an identity callback that turns any
  // `IntegerAttr` memory space into the matching LLVM address space, but we
  // re-register an explicit, named callback here so:
  //   1. The "memory space 3 == SPM" contract is visible in code, not buried
  //      in MLIR defaults.
  //   2. Drift between Phase 2 and Phase 3 (e.g., someone changing the
  //      constant in only one place) becomes a compile-time signal — both
  //      sides include the same header.
  //   3. Future address spaces (e.g., a separate I/O space) can be added
  //      here without surprising anyone.
  //
  // The callback is registered LAST, so it shadows the base-class default and
  // wins.  We keep DRAM (memory space 0) on the default path.
  addTypeAttributeConversion(
      [](BaseMemRefType /*memref*/, IntegerAttr addrspace)
          -> TypeConverter::AttributeConversionResult {
        // Identity for both DRAM (0) and SPM (kSPMAddressSpace).  Asserting
        // here that the value is one of {0, kSPMAddressSpace} would catch
        // accidental introduction of new address spaces, but we keep the
        // converter permissive (return the attribute unchanged) so unrelated
        // memrefs that pre-exist in MLIR core continue to lower.
        (void)triton::cpu::kSPMAddressSpace;
        return addrspace;
      });
}

Type TritonCPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (isa<RankedTensorType>(pointeeType)) {
    // struct {
    //   ptr base_ptr;
    //   array<rank x i64> offsets;
    //   array<rank x i64> shape;
    //   array<rank x i64> strides;
    // }
    auto tensorTy = cast<RankedTensorType>(pointeeType);
    auto rank = tensorTy.getShape().size();
    auto i64Ty = IntegerType::get(ctx, 64);
    SmallVector<Type, 4> types;
    types.push_back(LLVM::LLVMPointerType::get(ctx));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx);
}

Type TritonCPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  if (isa<PointerType>(type.getElementType()))
    return VectorType::get(type.getShape(),
                           IntegerType::get(type.getContext(), 64));
  llvm_unreachable("No tensor types are expected in TTCIR");
}
