// RUN: mlir-opt %s --expand-strided-metadata --finalize-memref-to-llvm \
// RUN:              --convert-func-to-llvm --reconcile-unrealized-casts \
// RUN:   | FileCheck %s
//
// Uses upstream `mlir-opt` (not `triton-opt`) on purpose: this test verifies
// the contract that the **standard** MLIR memref-to-LLVM pass preserves the
// integer memory-space attribute.  The AOT pipeline (`compiler.py`) uses the
// same upstream pass (`createFinalizeMemRefToLLVMConversionPass`) for its
// memref lowering, so this test pins exactly the behavior the production
// pipeline relies on.

// Pins the SPM address-space contract that ties Phase 2 (TypeConverter,
// DmaOpsToLLVM) and Phase 3 (ConvertMemoryToSPM) together:
//
//     memref<...,3>  --memref-to-llvm-->  !llvm.ptr<3>
//
// The shared constant `triton::cpu::kSPMAddressSpace == 3` lives in
// `include/triton/Dialect/TritonCPU/IR/SPMAttrs.h`.  If this test ever fails
// it means either the constant moved, or some upstream MLIR change altered
// how an integer memory-space attribute lowers — both should be handled
// before lowering Phase 3 SPM tiles.

// CHECK-LABEL: llvm.func @spm_roundtrip
// All pointer values that originate from a memref<..., 3> MUST be tagged
// addrspace 3 in the lowered LLVM dialect.
// CHECK:       !llvm.ptr<3>
// CHECK:       llvm.store{{.*}}!llvm.ptr<3>
// CHECK:       llvm.load{{.*}}!llvm.ptr<3>
// And nothing the test produces should accidentally end up in addrspace 0
// for the SPM memref's pointer column.
// CHECK-NOT:   !llvm.ptr<0>

module {
  func.func @spm_roundtrip(%spm_i64: i64, %v: f32) -> f32 {
    %c0 = arith.constant 0 : index
    // Phase 3 emits exactly this pattern: an i64 SPM base address turned
    // into a 0-d memref in addrspace 3 via unrealized_conversion_cast,
    // then a reinterpret_cast to the tile shape.
    %base = builtin.unrealized_conversion_cast %spm_i64
        : i64 to memref<f32, 3>
    %tile = memref.reinterpret_cast %base to
        offset: [0], sizes: [16], strides: [1]
        : memref<f32, 3> to memref<16xf32, strided<[1]>, 3>
    memref.store %v, %tile[%c0] : memref<16xf32, strided<[1]>, 3>
    %loaded = memref.load %tile[%c0] : memref<16xf32, strided<[1]>, 3>
    return %loaded : f32
  }
}
