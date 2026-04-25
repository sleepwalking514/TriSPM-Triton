// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536" | FileCheck %s

// ============================================================================
// Test: GEMM K-loop with two tiled loads feeding vector.contract
//       → double-buffered DMA + SPM reads
// ============================================================================

// CHECK-LABEL: @gemm_double_buffer
//
// Prologue: DMA first tiles into buffer 0, then wait.
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_wait
//
// Loop body should contain:
//   - arith.cmpi eq (buffer index check)
//   - arith.select (buffer selection)
//   - memref.reinterpret_cast {{.*}} memref<{{.*}}, 3>  (SPM read for A)
//   - memref.reinterpret_cast {{.*}} memref<{{.*}}, 3>  (SPM read for B)
//   - vector.contract (the dot product, unchanged)
//   - scf.if (conditional prefetch)
//   - triton_cpu.dma_enqueue_2d (prefetch A next)
//   - triton_cpu.dma_enqueue_2d (prefetch B next)
//   - triton_cpu.dma_wait (end of body)
//
// CHECK:       scf.for
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         vector.contract
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         triton_cpu.dma_wait
// CHECK:         scf.yield

module {
  tt.func public @gemm_double_buffer(
      %A: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %B: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %C: memref<64x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>

    // K-loop: two tiled loads feeding vector.contract
    %result = scf.for %k = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
      %a_tile = vector.transfer_read %A[%c0, %k], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>
      %b_tile = vector.transfer_read %B[%k, %c0], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>

      %dot = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %a_tile, %b_tile, %acc : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>

      scf.yield %dot : vector<16x16xf32>
    }

    // Store result
    vector.transfer_write %result, %C[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<64x64xf32, strided<[64, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: Reduction loop with single tiled load (not feeding dot)
//       → single-buffer DMA prefetch
// ============================================================================

// CHECK-LABEL: @reduction_prefetch
//
// Prologue: DMA first chunk, then wait.
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_wait
//
// Loop body:
//   - scf.if (conditional prefetch for next chunk)
//   - triton_cpu.dma_enqueue_2d (prefetch next)
//   - vector.transfer_read from SPM memref (address space 3)
//   - arith.addf (reduction)
//   - triton_cpu.dma_wait
//
// CHECK:       scf.for
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// CHECK:         arith.addf
// CHECK:         triton_cpu.dma_wait
// CHECK:         scf.yield

module {
  tt.func public @reduction_prefetch(
      %X: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %chunk = vector.transfer_read %X[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %chunk : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }

    tt.return
  }
}

// -----

// ============================================================================
// Test: Loop with no eligible loads → unchanged (no DMA ops inserted)
// ============================================================================

// CHECK-LABEL: @no_transform_scalar
// CHECK-NOT:   triton_cpu.dma_enqueue_2d
// CHECK-NOT:   triton_cpu.dma_wait
// CHECK:       scf.for
// CHECK:         arith.addi
// CHECK:         scf.yield

module {
  tt.func public @no_transform_scalar() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %init = arith.constant 0 : i32

    %result = scf.for %i = %c0 to %c64 step %c1
        iter_args(%acc = %init) -> (i32) {
      %one = arith.constant 1 : i32
      %sum = arith.addi %acc, %one : i32
      scf.yield %sum : i32
    }

    tt.return
  }
}