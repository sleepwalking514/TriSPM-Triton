//===- SPMAttrs.h - Shared SPM constants for the TritonCPU dialect --------===//
//
// Single source of truth for the LLVM/MLIR address space used by all SPM
// (scratchpad memory) memrefs and pointers.  Both the Phase 2 lowering pass
// (TypeConverter / DmaOpsToLLVM) and the Phase 3 transformation pass
// (ConvertMemoryToSPM) MUST use this constant — drifting between them would
// silently break the alias-no-alias guarantee that justifies treating SPM
// loads/stores as independent of DRAM loads/stores.
//
// Address space layout (LLVM):
//   0  -> ordinary DRAM
//   3  -> SPM scratchpad (routed via spm_port at L1 latency)
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONCPU_IR_SPMATTRS_H
#define TRITON_DIALECT_TRITONCPU_IR_SPMATTRS_H

namespace mlir {
namespace triton {
namespace cpu {

/// LLVM/MLIR address space for scratchpad memory (SPM).
///
/// Wherever Phase 3 emits a `memref<MxNxT, kSPMAddressSpace>` or this code
/// emits an `llvm.ptr<3>`, that value points at the SPM device.  The lowering
/// of `memref<...,kSPMAddressSpace>` to LLVM is wired up in
/// `TritonCPUToLLVMTypeConverter` (see TypeConverter.cpp).
static constexpr unsigned kSPMAddressSpace = 3;

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONCPU_IR_SPMATTRS_H
