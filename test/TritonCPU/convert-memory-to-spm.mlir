// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=32" | FileCheck %s
// RUN: env TRITON_SPM_ATTENTION_Q_RESIDENT=1 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=32" | FileCheck %s --check-prefix=MULTI
// RUN: env TRITON_SPM_ATTENTION_Q_RESIDENT=1 TRITON_SPM_ATTENTION_KV_STREAM=1 TRITON_SPM_ATTENTION_KV_STREAM_STAGE_Q=1 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=32" | FileCheck %s --check-prefix=KV-STREAM
// RUN: env TRITON_SPM_ATTENTION_QK_TILE=1 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=32" | FileCheck %s --check-prefix=QK-TILE
// RUN: env TRITON_SPM_ATTENTION_PV_GENERATED_TILE=1 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=32" | FileCheck %s --check-prefix=PV-GEN

// ============================================================================
// Test: GEMM K-loop with two tiled loads feeding vector.contract
//       → double-buffered DMA + SPM reads
// ============================================================================

// CHECK-LABEL: @gemm_double_buffer
//
// Prologue: DMA first tiles into buffer 0 (no wait in prologue — wait is
// at the top of the loop body to overlap with the previous iteration's
// prefetch).
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_enqueue_2d
//
// Loop body: wait → select buffers → prefetch next → SPM read → compute.
// CHECK:       scf.for
// CHECK:         triton_cpu.dma_wait
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         vector.contract
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
// Test: GEMM with transfer reads in B/A IR order still identifies A and B via
//       vector.contract lhs/rhs operands, not walk order.
// ============================================================================

// CHECK-LABEL: @gemm_reversed_load_order
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       scf.for
// CHECK:         triton_cpu.dma_wait
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         vector.contract

module {
  tt.func public @gemm_reversed_load_order(
      %A: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %B: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %C: memref<64x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>

    %result = scf.for %k = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
      %b_tile = vector.transfer_read %B[%k, %c0], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>
      %a_tile = vector.transfer_read %A[%c0, %k], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>

      %dot = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %a_tile, %b_tile, %acc : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>

      scf.yield %dot : vector<16x16xf32>
    }

    vector.transfer_write %result, %C[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<64x64xf32, strided<[64, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: GEMM loop with an extra non-dot tiled load still transforms, because
//       exactly two transfer reads feed vector.contract.
// ============================================================================

// CHECK-LABEL: @gemm_extra_non_dot_load
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       scf.for
// CHECK:         triton_cpu.dma_wait
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// CHECK:         vector.transfer_read {{.*}} vector<16xf32>
// CHECK:         vector.contract

module {
  tt.func public @gemm_extra_non_dot_load(
      %A: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %B: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %Bias: memref<64xf32, strided<[1], offset: 0>>,
      %C: memref<64x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>
    %bias_init = arith.constant dense<0.0> : vector<16xf32>

    %result:2 = scf.for %k = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init, %bias_acc = %bias_init)
        -> (vector<16x16xf32>, vector<16xf32>) {
      %a_tile = vector.transfer_read %A[%c0, %k], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>
      %b_tile = vector.transfer_read %B[%k, %c0], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>
      %bias = vector.transfer_read %Bias[%k], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>

      %dot = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %a_tile, %b_tile, %acc : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>

      %bias_next = arith.addf %bias_acc, %bias : vector<16xf32>
      scf.yield %dot, %bias_next : vector<16x16xf32>, vector<16xf32>
    }

    vector.transfer_write %result#0, %C[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<64x64xf32, strided<[64, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: Reduction loop with single tiled load (not feeding dot)
//       → double-buffered DMA prefetch
// ============================================================================

// CHECK-LABEL: @reduction_prefetch
//
// Prologue: DMA first chunk.  The loop-body top wait covers it on iter 0.
// CHECK:       triton_cpu.dma_enqueue_2d
//
// Loop body: wait → select current/next buffers → prefetch next → SPM read
// → compute → yield flipped buffer index.
//
// CHECK:       scf.for
// CHECK:         triton_cpu.dma_wait
// CHECK:         arith.cmpi eq
// CHECK:         arith.select
// CHECK:         arith.select
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// CHECK:         arith.addf
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
// Test: Reduction-like loop with multiple tiled loads sharing the same IV
//       → all streams are double-buffered through SPM.
// ============================================================================

// CHECK-LABEL: @reduction_multi_load_prefetch
//
// Prologue: one DMA for each current stream.
// CHECK:       triton_cpu.dma_enqueue_2d
// CHECK:       triton_cpu.dma_enqueue_2d
//
// Loop body: one wait, two alternate-buffer prefetches, two SPM reads.
// CHECK:       scf.for
// CHECK:         triton_cpu.dma_wait
// CHECK:         scf.if
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// CHECK:         vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         scf.yield

module {
  tt.func public @reduction_multi_load_prefetch(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %Scale: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %x = vector.transfer_read %X[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %scale = vector.transfer_read %Scale[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %scaled = arith.mulf %x, %scale : vector<16xf32>
      %sum = arith.addf %acc, %scaled : vector<16xf32>
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

// -----

// ============================================================================
// Test: 2D reduction with IV on non-leading dimension (stride=1).
//       Verifies that the prefetch address uses the correct stride for the
//       IV dimension, not always the leading stride.
//
//       memref<8x64xf32, strided<[64, 1]>> with transfer_read %X[%c0, %i]
//       → IV indexes dim 1 (stride=1), so byte step = 1 * 4 = 4 per element.
//       Bug (before fix): would use leading stride 64 → 256 bytes per step.
// ============================================================================

// CHECK-LABEL: @reduction_2d_non_leading_iv
//
// Prologue DMA; loop-body top wait covers it on iter 0.
// CHECK:       triton_cpu.dma_enqueue_2d
//
// Loop body: wait → prefetch next (with stride=1*4=4) → SPM read → compute.
// The key assertion: arith.muli uses constant 4 (not 256) for the byte offset.
// CHECK:       scf.for
// CHECK:         triton_cpu.dma_wait
// CHECK:         arith.subi
// CHECK:         arith.index_cast
// CHECK:         scf.if
// CHECK-NOT:       arith.constant 256 : i64
// CHECK:           arith.constant 4 : i64
// CHECK:           arith.muli {{.*}} : i64
// CHECK:           triton_cpu.dma_enqueue_2d
// CHECK:         memref.reinterpret_cast
// CHECK:         vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// CHECK:         arith.addf
// CHECK:         scf.yield

module {
  tt.func public @reduction_2d_non_leading_iv(
      %X: memref<8x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    // Reduce along dim 1 (columns): IV indexes the non-leading dimension.
    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %chunk = vector.transfer_read %X[%c0, %i], %cst
          {in_bounds = [true]} : memref<8x64xf32, strided<[64, 1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %chunk : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }

    tt.return
  }
}

// -----

// ============================================================================
// Test: GEMM bail-out after matching dot loads leaves the loop unchanged.
//       The dynamic step makes the boundary guard fail; no speculative prologue
//       DMA or replacement loop should remain.
// ============================================================================

// CHECK-LABEL: @gemm_bailout_dynamic_step_no_dma
// CHECK-NOT:   triton_cpu.dma_enqueue_2d
// CHECK-NOT:   triton_cpu.dma_wait
// CHECK:       scf.for
// CHECK:         vector.transfer_read {{.*}} memref<64x64xf32
// CHECK:         vector.transfer_read {{.*}} memref<64x64xf32
// CHECK:         vector.contract

module {
  tt.func public @gemm_bailout_dynamic_step_no_dma(
      %A: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %B: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %C: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %step: index) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>

    %result = scf.for %k = %c0 to %c64 step %step
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

    vector.transfer_write %result, %C[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<64x64xf32, strided<[64, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: GEMM bail-out after partially computing the prologue address cleans up
//       speculative IR.  A's prologue address is computable, but B's transfer
//       reads from a loop-local subview, so prologue address recovery fails.
// ============================================================================

// CHECK-LABEL: @gemm_bailout_partial_prologue_cleanup
// CHECK-NOT:   memref.extract_aligned_pointer_as_index
// CHECK-NOT:   triton_cpu.dma_enqueue_2d
// CHECK-NOT:   triton_cpu.dma_wait
// CHECK:       scf.for
// CHECK:         vector.transfer_read {{.*}} memref<64x64xf32
// CHECK:         memref.subview
// CHECK:         vector.transfer_read
// CHECK:         vector.contract

module {
  tt.func public @gemm_bailout_partial_prologue_cleanup(
      %A: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %B: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %C: memref<64x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>

    %result = scf.for %k = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
      %a_tile = vector.transfer_read %A[%c0, %k], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>
      %b_view = memref.subview %B[%k, %c0] [16, 16] [1, 1]
          : memref<64x64xf32, strided<[64, 1], offset: 0>> to memref<16x16xf32, strided<[64, 1], offset: ?>>
      %b_tile = vector.transfer_read %b_view[%c0, %c0], %cst
          {in_bounds = [true, true]} : memref<16x16xf32, strided<[64, 1], offset: ?>>, vector<16x16xf32>

      %dot = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %a_tile, %b_tile, %acc : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>

      scf.yield %dot : vector<16x16xf32>
    }

    vector.transfer_write %result, %C[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<64x64xf32, strided<[64, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: Reduction bail-out after matching tiled loads leaves the loop unchanged.
//       The dynamic step makes the boundary guard fail; no speculative prologue
//       DMA or replacement loop should remain.
// ============================================================================

// CHECK-LABEL: @reduction_bailout_dynamic_step_no_dma
// CHECK-NOT:   triton_cpu.dma_enqueue_2d
// CHECK-NOT:   triton_cpu.dma_wait
// CHECK:       scf.for
// CHECK:         vector.transfer_read {{.*}} memref<64xf32
// CHECK:         arith.addf

module {
  tt.func public @reduction_bailout_dynamic_step_no_dma(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %step: index) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %step
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
// Test: attention-v2 Q-resident lowering.
//       Q is materialized once into SPM before the loop. K/V remain normal
//       ordinary DRAM transfer_read operations to avoid SPM scalarization blow-up.
// ============================================================================

// MULTI-LABEL: @attention_v2_window_qkv
// MULTI:       triton_cpu.dma_enqueue_2d
// MULTI:       triton_cpu.dma_wait
// MULTI:       vector.transfer_read {{.*}} memref<16x32xf16, strided<[32, 1]>, 3>
// MULTI:       arith.extf
// MULTI:       scf.for
// MULTI:         vector.transfer_read {{.*}} memref<32x64xf32, strided<[1, 32]>>
// MULTI:         vector.transfer_read {{.*}} memref<64x32xf32, strided<[32, 1]>>
// MULTI:         vector.contract
// MULTI:         vector.contract
// MULTI-NOT:   double-buffered attention K/V window
// MULTI-NOT:   vector.transfer_read {{.*}} memref<64x32xf16, strided<[32, 1], offset: 0>>

module {
  tt.func public @attention_v2_window_qkv(
      %Q: memref<64x32xf16, strided<[32, 1], offset: 0>>,
      %K: memref<32x64xf32, strided<[1, 32], offset: 0>>,
      %V: memref<64x32xf32, strided<[32, 1], offset: 0>>,
      %O: memref<64x32xf32, strided<[32, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %cst_h = arith.constant 0.0 : f16
    %score_init = arith.constant dense<0.0> : vector<16x16xf32>
    %acc_init = arith.constant dense<0.0> : vector<16x32xf32>

    %q_tile = vector.transfer_read %Q[%c0, %c0], %cst_h
        {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: 0>>, vector<16x32xf16>
    %q_cast = arith.extf %q_tile : vector<16x32xf16> to vector<16x32xf32>

    %result = scf.for %n = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16x32xf32>) {
      %k_tile = vector.transfer_read %K[%c0, %n], %cst
          {in_bounds = [true, true]} : memref<32x64xf32, strided<[1, 32], offset: 0>>, vector<32x16xf32>
      %v_tile = vector.transfer_read %V[%n, %c0], %cst
          {in_bounds = [true, true]} : memref<64x32xf32, strided<[32, 1], offset: 0>>, vector<16x32xf32>

      %scores = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %q_cast, %k_tile, %score_init : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>

      %next = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %scores, %v_tile, %acc : vector<16x16xf32>, vector<16x32xf32> into vector<16x32xf32>

      scf.yield %next : vector<16x32xf32>
    }

    vector.transfer_write %result, %O[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x32xf32>, memref<64x32xf32, strided<[32, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: attention-v2 K/V streaming with loop-carried tensor pointers.
//       The next-tile DRAM step must come from tt.advance offsets rather than
//       the transfer_read indices extracted from the current block pointer.
// ============================================================================

// KV-STREAM-LABEL: @attention_v2_blockptr_kv_stream_step
// KV-STREAM:       triton_cpu.dma_enqueue_2d
// KV-STREAM:       triton_cpu.dma_enqueue_2d
// KV-STREAM:       scf.for
// KV-STREAM:         triton_cpu.dma_wait
// KV-STREAM:         scf.if
// KV-STREAM:           arith.constant 1024 : i64
// KV-STREAM:           arith.muli
// KV-STREAM:           triton_cpu.dma_enqueue_2d
// KV-STREAM:           arith.constant 1024 : i64
// KV-STREAM:           arith.muli
// KV-STREAM:           triton_cpu.dma_enqueue_2d
// KV-STREAM:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[1, 16]>, 3>
// KV-STREAM:         vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>

module {
  tt.func public @attention_v2_blockptr_kv_stream_step(
      %Q: memref<16x16xf32, strided<[16, 1], offset: 0>>,
      %K: !tt.ptr<f32>,
      %V: !tt.ptr<f32>,
      %O: memref<16x16xf32, strided<[16, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %score_init = arith.constant dense<0.0> : vector<16x16xf32>
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>

    %q_tile = vector.transfer_read %Q[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<16x16xf32, strided<[16, 1], offset: 0>>, vector<16x16xf32>
    %k_ptr = tt.make_tensor_ptr %K, [%c16_i64, %c32_i64], [%c1_i64, %c16_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %v_ptr = tt.make_tensor_ptr %V, [%c32_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32]
        {order = array<i32: 1, 0>} : <tensor<16x16xf32>>

    %result:3 = scf.for %n = %c0_i32 to %c32_i32 step %c16_i32
        iter_args(%acc = %acc_init, %k_iter = %k_ptr, %v_iter = %v_ptr)
        -> (vector<16x16xf32>, !tt.ptr<tensor<16x16xf32>>, !tt.ptr<tensor<16x16xf32>>) : i32 {
      %k_mem = triton_cpu.extract_memref %k_iter
          : <tensor<16x16xf32>> -> memref<16x32xf32, strided<[1, 16]>>
      %k_idx:2 = triton_cpu.extract_indices %k_iter
          : <tensor<16x16xf32>> -> index, index
      %k_tile = vector.transfer_read %k_mem[%k_idx#0, %k_idx#1], %cst
          {in_bounds = [true, true]} : memref<16x32xf32, strided<[1, 16]>>, vector<16x16xf32>
      %v_mem = triton_cpu.extract_memref %v_iter
          : <tensor<16x16xf32>> -> memref<32x16xf32, strided<[16, 1]>>
      %v_idx:2 = triton_cpu.extract_indices %v_iter
          : <tensor<16x16xf32>> -> index, index
      %v_tile = vector.transfer_read %v_mem[%v_idx#0, %v_idx#1], %cst
          {in_bounds = [true, true]} : memref<32x16xf32, strided<[16, 1]>>, vector<16x16xf32>

      %scores = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %q_tile, %k_tile, %score_init : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>

      %next = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %scores, %v_tile, %acc : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>

      %k_next = tt.advance %k_iter, [%c0_i32, %c16_i32] : <tensor<16x16xf32>>
      %v_next = tt.advance %v_iter, [%c16_i32, %c0_i32] : <tensor<16x16xf32>>
      scf.yield %next, %k_next, %v_next
          : vector<16x16xf32>, !tt.ptr<tensor<16x16xf32>>, !tt.ptr<tensor<16x16xf32>>
    }

    vector.transfer_write %result#0, %O[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[16, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: two attention-v2 loops share one resident Q tile.  The first loop
//       materializes Q into SPM; the second loop recognizes that SPM read as
//       the same resident Q instead of issuing another Q DMA/prologue.
// ============================================================================

// MULTI-LABEL: @attention_v2_two_stage_shared_q
// MULTI:       triton_cpu.dma_enqueue_2d
// MULTI:       triton_cpu.dma_wait
// MULTI:       vector.transfer_read {{.*}} memref<16x32xf16, strided<[32, 1]>, 3>
// MULTI:       scf.for
// MULTI:         vector.transfer_read {{.*}} memref<32x64xf32, strided<[1, 32]>>
// MULTI:         vector.transfer_read {{.*}} memref<64x32xf32, strided<[32, 1]>>
// MULTI:       scf.for
// MULTI:         vector.transfer_read {{.*}} memref<32x64xf32, strided<[1, 32]>>
// MULTI:         vector.transfer_read {{.*}} memref<64x32xf32, strided<[32, 1]>>
// MULTI-NOT:   vector.transfer_read {{.*}} memref<64x32xf16, strided<[32, 1], offset: 0>>

module {
  tt.func public @attention_v2_two_stage_shared_q(
      %Q: memref<64x32xf16, strided<[32, 1], offset: 0>>,
      %K0: memref<32x64xf32, strided<[1, 32], offset: 0>>,
      %V0: memref<64x32xf32, strided<[32, 1], offset: 0>>,
      %K1: memref<32x64xf32, strided<[1, 32], offset: 0>>,
      %V1: memref<64x32xf32, strided<[32, 1], offset: 0>>,
      %O: memref<64x32xf32, strided<[32, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %cst_h = arith.constant 0.0 : f16
    %score_init = arith.constant dense<0.0> : vector<16x16xf32>
    %acc_init = arith.constant dense<0.0> : vector<16x32xf32>

    %q_tile = vector.transfer_read %Q[%c0, %c0], %cst_h
        {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: 0>>, vector<16x32xf16>
    %q_cast = arith.extf %q_tile : vector<16x32xf16> to vector<16x32xf32>

    %result0 = scf.for %n = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16x32xf32>) {
      %k_tile = vector.transfer_read %K0[%c0, %n], %cst
          {in_bounds = [true, true]} : memref<32x64xf32, strided<[1, 32], offset: 0>>, vector<32x16xf32>
      %v_tile = vector.transfer_read %V0[%n, %c0], %cst
          {in_bounds = [true, true]} : memref<64x32xf32, strided<[32, 1], offset: 0>>, vector<16x32xf32>

      %scores = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %q_cast, %k_tile, %score_init : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>

      %next = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %scores, %v_tile, %acc : vector<16x16xf32>, vector<16x32xf32> into vector<16x32xf32>

      scf.yield %next : vector<16x32xf32>
    }

    %result1 = scf.for %n = %c0 to %c64 step %c16
        iter_args(%acc = %result0) -> (vector<16x32xf32>) {
      %k_tile = vector.transfer_read %K1[%c0, %n], %cst
          {in_bounds = [true, true]} : memref<32x64xf32, strided<[1, 32], offset: 0>>, vector<32x16xf32>
      %v_tile = vector.transfer_read %V1[%n, %c0], %cst
          {in_bounds = [true, true]} : memref<64x32xf32, strided<[32, 1], offset: 0>>, vector<16x32xf32>

      %scores = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %q_cast, %k_tile, %score_init : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>

      %next = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %scores, %v_tile, %acc : vector<16x16xf32>, vector<16x32xf32> into vector<16x32xf32>

      scf.yield %next : vector<16x32xf32>
    }

    vector.transfer_write %result1, %O[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x32xf32>, memref<64x32xf32, strided<[32, 1], offset: 0>>
    tt.return
  }
}

// -----

// ============================================================================
// Test: opt-in function-scope fused-attention QK tile staging.
//       Q/K are DMA-staged into SPM before the local-consumer QK contraction;
//       softmax-like local consumers remain on the original vector path.
// ============================================================================

// QK-TILE-LABEL: @fused_attention_qk_tile
// QK-TILE:       triton_cpu.dma_enqueue_2d
// QK-TILE:       triton_cpu.dma_enqueue_2d
// QK-TILE:       triton_cpu.dma_wait
// QK-TILE:       vector.transfer_read {{.*}} memref<16x16xf32, strided<[16, 1]>, 3>
// QK-TILE:       vector.transfer_read {{.*}} memref<16x32xf32, strided<[1, 16]>, 3>
// QK-TILE:       vector.contract
// QK-TILE:       arith.mulf

module {
  tt.func public @fused_attention_qk_tile(
      %Q: memref<64x16xf32, strided<[16, 1], offset: 0>>,
      %K: memref<16x64xf32, strided<[1, 16], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %score_init = arith.constant dense<0.0> : vector<16x32xf32>

    %q_tile = vector.transfer_read %Q[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<64x16xf32, strided<[16, 1], offset: 0>>, vector<16x16xf32>
    %k_tile = vector.transfer_read %K[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<16x64xf32, strided<[1, 16], offset: 0>>, vector<16x32xf32>

    %scores = vector.contract {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %q_tile, %k_tile, %score_init : vector<16x16xf32>, vector<16x32xf32> into vector<16x32xf32>

    %scaled = arith.mulf %scores, %scores : vector<16x32xf32>
    tt.return
  }
}

// -----

// ============================================================================
// Test: opt-in function-scope PV generated-operand residency.
//       The generated probability tile is written to SPM and read back for PV;
//       the memory-backed V tile remains on the cache path at this stage.
// ============================================================================

// PV-GEN-LABEL: @fused_attention_pv_generated_tile
// PV-GEN:       vector.contract
// PV-GEN:       arith.mulf
// PV-GEN:       vector.transfer_read {{.*}} memref<64x16xf32, strided<[16, 1]>>, vector<32x16xf32>
// PV-GEN:       vector.transfer_write {{.*}} memref<16x32xf32, strided<[32, 1]>, 3>
// PV-GEN:       vector.transfer_read {{.*}} memref<16x32xf32, strided<[32, 1]>, 3>
// PV-GEN:       vector.contract
// PV-GEN:       vector.transfer_write

module {
  tt.func public @fused_attention_pv_generated_tile(
      %Q: memref<64x16xf32, strided<[16, 1], offset: 0>>,
      %K: memref<16x64xf32, strided<[1, 16], offset: 0>>,
      %V: memref<64x16xf32, strided<[16, 1], offset: 0>>,
      %O: memref<64x16xf32, strided<[16, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %score_init = arith.constant dense<0.0> : vector<16x32xf32>
    %out_init = arith.constant dense<0.0> : vector<16x16xf32>

    %q_tile = vector.transfer_read %Q[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<64x16xf32, strided<[16, 1], offset: 0>>, vector<16x16xf32>
    %k_tile = vector.transfer_read %K[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<16x64xf32, strided<[1, 16], offset: 0>>, vector<16x32xf32>
    %scores = vector.contract {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %q_tile, %k_tile, %score_init : vector<16x16xf32>, vector<16x32xf32> into vector<16x32xf32>

    %prob = arith.mulf %scores, %scores : vector<16x32xf32>
    %v_tile = vector.transfer_read %V[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<64x16xf32, strided<[16, 1], offset: 0>>, vector<32x16xf32>
    %out = vector.contract {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %prob, %v_tile, %out_init : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>

    vector.transfer_write %out, %O[%c0, %c0]
        {in_bounds = [true, true]} : vector<16x16xf32>, memref<64x16xf32, strided<[16, 1], offset: 0>>
    tt.return
  }
}
