// RUN: env TRITON_SPM_PAGED_KV_DECODE=0 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=BASELINE
// RUN: env TRITON_SPM_PAGED_KV_DECODE_DOUBLE_BUFFER=0 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=PAGED-KV-ACCEPT
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=PAGED-KV-REJECT
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=PAGED-KV-DOUBLE

// ============================================================================
// Lit fixture for the paged_kv_decode SPM matcher.  The matcher is default-on
// inside ConvertMemoryToSPM.cpp; this file pins the canonical pre-transform IR
// shape that the matcher recognises, plus two rejection cases that must
// continue to compile through the cache lowering.
//
// Check prefixes:
//   BASELINE          — matcher disabled, no transform; assertions describe the
//                       canonical IR shape produced by the W1 frontend
//                       (see workloads/kernels/paged_kv_decode/kernel.py).
//   PAGED-KV-REJECT   — default matcher on, but the function does not match the
//                       paged-gather pattern; the pass must leave it alone
//                       and emit no DMA ops.
//   PAGED-KV-ACCEPT   — matcher on with double-buffer disabled; the canonical
//                       pattern rewrites to C1 setup-loop staging.
//   PAGED-KV-DOUBLE   — default matcher and double-buffer path; the canonical
//                       pattern rewrites to page-level ping-pong: prologue
//                       prefetch, in-loop wait, conditional next-page prefetch,
//                       and SPM-backed consume.
// ============================================================================

// ============================================================================
// Accept case: canonical W1 page-gather pattern.
//   for p in [0, NUM_PAGES):
//     phys = tt.load (page_ids_ptr + p)
//     row_off = phys * PAGE_SIZE
//     K/V tile = make_tensor_ptr <[NUM_PHYS_PAGES*PAGE_SIZE, HEAD_DIM]>,
//                strides <[HEAD_DIM, 1]>, offsets <[row_off, 0]>,
//                block_shape <PAGE_SIZE x HEAD_DIM>
//     vector.transfer_read of the page tile from DRAM
//
// W1 (cache lowering) emits no DMA in the loop body.  With
// With the default paged-kv matcher, this shape rewrites to DMA descriptors +
// SPM-backed reads (addrspace 3).
// ============================================================================
// BASELINE-LABEL: @paged_kv_decode_w1
// BASELINE:       scf.for
// BASELINE:         tt.load %{{.*}} : !tt.ptr<i32>
// BASELINE:         arith.muli %{{.*}}, %{{.*}} : i32
// BASELINE:         tt.make_tensor_ptr %{{.*}} : <tensor<16x64xf32>>
// BASELINE:         tt.make_tensor_ptr %{{.*}} : <tensor<16x64xf32>>
// BASELINE:         vector.transfer_read {{.*}} memref<128x64xf32, strided<[64, 1]>>, vector<16x64xf32>
// BASELINE:         vector.transfer_read {{.*}} memref<128x64xf32, strided<[64, 1]>>, vector<16x64xf32>
// BASELINE-NOT:   triton_cpu.dma_enqueue_2d

// PAGED-KV-ACCEPT-LABEL: @paged_kv_decode_w1
// PAGED-KV-ACCEPT:       scf.for
// PAGED-KV-ACCEPT:         triton_cpu.dma_enqueue_2d
// PAGED-KV-ACCEPT:         triton_cpu.dma_enqueue_2d
// PAGED-KV-ACCEPT:       triton_cpu.dma_wait
// PAGED-KV-ACCEPT:       scf.for
// PAGED-KV-ACCEPT:         memref.reinterpret_cast
// PAGED-KV-ACCEPT:         vector.transfer_read {{.*}} memref<16x64xf32, strided<[64, 1]>, 3>, vector<16x64xf32>
// PAGED-KV-ACCEPT:         memref.reinterpret_cast
// PAGED-KV-ACCEPT:         vector.transfer_read {{.*}} memref<16x64xf32, strided<[64, 1]>, 3>, vector<16x64xf32>

// PAGED-KV-DOUBLE-LABEL: @paged_kv_decode_w1
// Prologue prefetch of page 0, K + V:
// PAGED-KV-DOUBLE:       triton_cpu.dma_enqueue_2d
// PAGED-KV-DOUBLE:       triton_cpu.dma_enqueue_2d
// Replacement page loop derives the ping-pong slot from page-IV parity:
// PAGED-KV-DOUBLE:       scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (vector<64xf32>)
// PAGED-KV-DOUBLE:         triton_cpu.dma_wait
// PAGED-KV-DOUBLE:         arith.andi
// PAGED-KV-DOUBLE:         scf.if
// PAGED-KV-DOUBLE:           arith.subi
// PAGED-KV-DOUBLE:           triton_cpu.dma_enqueue_2d
// PAGED-KV-DOUBLE:           triton_cpu.dma_enqueue_2d
// PAGED-KV-DOUBLE:         vector.transfer_read {{.*}} memref<16x64xf32, strided<[64, 1]>, 3>, vector<16x64xf32>
// PAGED-KV-DOUBLE:         vector.transfer_read {{.*}} memref<16x64xf32, strided<[64, 1]>, 3>, vector<16x64xf32>

module {
  tt.func public @paged_kv_decode_w1(
      %q_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %k_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %v_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %page_ids_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.0 : f32
    %acc0 = arith.constant dense<0.0> : vector<64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64

    %acc = scf.for %p = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%a = %acc0) -> (vector<64xf32>) : i32 {
      %phys_p = tt.addptr %page_ids_ptr, %p : !tt.ptr<i32>, i32
      %phys = tt.load %phys_p : !tt.ptr<i32>
      %row_off = arith.muli %phys, %c16_i32 : i32
      %kp = tt.make_tensor_ptr %k_cache_ptr, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%row_off, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf32>>
      %vp = tt.make_tensor_ptr %v_cache_ptr, [%c128_i64, %c64_i64], [%c64_i64, %c1_i64], [%row_off, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf32>>
      %kmr = triton_cpu.extract_memref %kp : <tensor<16x64xf32>> -> memref<128x64xf32, strided<[64, 1]>>
      %ki:2 = triton_cpu.extract_indices %kp : <tensor<16x64xf32>> -> index, index
      %k = vector.transfer_read %kmr[%ki#0, %ki#1], %cst {in_bounds = [true, true]} : memref<128x64xf32, strided<[64, 1]>>, vector<16x64xf32>
      %vmr = triton_cpu.extract_memref %vp : <tensor<16x64xf32>> -> memref<128x64xf32, strided<[64, 1]>>
      %vi:2 = triton_cpu.extract_indices %vp : <tensor<16x64xf32>> -> index, index
      %v = vector.transfer_read %vmr[%vi#0, %vi#1], %cst {in_bounds = [true, true]} : memref<128x64xf32, strided<[64, 1]>>, vector<16x64xf32>
      %k_row0 = vector.extract %k[0] : vector<64xf32> from vector<16x64xf32>
      %v_row0 = vector.extract %v[0] : vector<64xf32> from vector<16x64xf32>
      %kv = arith.addf %k_row0, %v_row0 : vector<64xf32>
      %new_a = arith.addf %a, %kv : vector<64xf32>
      scf.yield %new_a : vector<64xf32>
    }
    %otp = tt.make_tensor_ptr %out_ptr, [%c64_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<64xf32>>
    %omr = triton_cpu.extract_memref %otp : <tensor<64xf32>> -> memref<64xf32, strided<[1]>>
    %oi = triton_cpu.extract_indices %otp : <tensor<64xf32>> -> index
    vector.transfer_write %acc, %omr[%oi] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1]>>
    tt.return
  }
}

// -----

// ============================================================================
// Rejection case 1: row-granular gather without `phys * PAGE_SIZE`.
//
// The indirect index addresses individual rows of the source tensor; the
// block_shape is 1D, not a 2D page tile.  This is the embedding-bag shape
// that TRITON_SPM_INDIRECT_TILE handles, NOT TRITON_SPM_PAGED_KV_DECODE.
// The paged-kv matcher must distinguish itself by requiring a 2D tile with
// the structured `phys * PAGE_SIZE` row-offset multiplier.
// ============================================================================
// BASELINE-LABEL: @paged_kv_reject_no_page_size_mul
// BASELINE:       scf.for
// BASELINE:         tt.load %{{.*}} : !tt.ptr<i32>
// BASELINE:         tt.make_tensor_ptr %{{.*}} : <tensor<64xf32>>
// BASELINE-NOT:     tt.make_tensor_ptr %{{.*}} : <tensor<{{[0-9]+}}x{{[0-9]+}}xf32>>
// PAGED-KV-REJECT-LABEL: @paged_kv_reject_no_page_size_mul
// PAGED-KV-REJECT-NOT:   triton_cpu.dma_enqueue_2d
// PAGED-KV-REJECT-NOT:   triton_cpu.dma_wait

module {
  tt.func public @paged_kv_reject_no_page_size_mul(
      %k_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %row_ids_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.0 : f32
    %acc0 = arith.constant dense<0.0> : vector<64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %acc = scf.for %p = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%a = %acc0) -> (vector<64xf32>) : i32 {
      %row_p = tt.addptr %row_ids_ptr, %p : !tt.ptr<i32>, i32
      %row = tt.load %row_p : !tt.ptr<i32>
      // No multiplication by PAGE_SIZE; the loaded index is the row offset.
      %tp = tt.make_tensor_ptr %k_cache_ptr, [%c8192_i64], [%c1_i64], [%row] {order = array<i32: 0>} : <tensor<64xf32>>
      %mr = triton_cpu.extract_memref %tp : <tensor<64xf32>> -> memref<8192xf32, strided<[1]>>
      %i = triton_cpu.extract_indices %tp : <tensor<64xf32>> -> index
      %r = vector.transfer_read %mr[%i], %cst {in_bounds = [true]} : memref<8192xf32, strided<[1]>>, vector<64xf32>
      %new_a = arith.addf %a, %r : vector<64xf32>
      scf.yield %new_a : vector<64xf32>
    }
    %otp = tt.make_tensor_ptr %out_ptr, [%c64_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<64xf32>>
    %omr = triton_cpu.extract_memref %otp : <tensor<64xf32>> -> memref<64xf32, strided<[1]>>
    %oi = triton_cpu.extract_indices %otp : <tensor<64xf32>> -> index
    vector.transfer_write %acc, %omr[%oi] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1]>>
    tt.return
  }
}

// -----

// ============================================================================
// Rejection case 2: dynamic page-extent in the source tensor's shape.
//
// Both K- and V-cache memrefs have a row count of `?` (runtime
// `%dyn_extent`).  Without a compile-time NUM_PHYS_PAGES the matcher
// cannot prove the staging buffer fits, nor can it construct a fixed 2D
// DMA descriptor, so it must reject *on the dynamic-extent check* (not on
// the "exactly two reads" check — both K and V are present here) and let
// the cache lowering handle the gather.
// ============================================================================
// BASELINE-LABEL: @paged_kv_reject_dynamic_extent
// BASELINE:       tt.make_tensor_ptr %{{.*}} : <tensor<16x64xf32>>
// BASELINE:       tt.make_tensor_ptr %{{.*}} : <tensor<16x64xf32>>
// BASELINE:       vector.transfer_read {{.*}} memref<?x64xf32, strided<[64, 1]>>, vector<16x64xf32>
// BASELINE:       vector.transfer_read {{.*}} memref<?x64xf32, strided<[64, 1]>>, vector<16x64xf32>
// PAGED-KV-REJECT-LABEL: @paged_kv_reject_dynamic_extent
// PAGED-KV-REJECT-NOT:   triton_cpu.dma_enqueue_2d
// PAGED-KV-REJECT-NOT:   triton_cpu.dma_wait

module {
  tt.func public @paged_kv_reject_dynamic_extent(
      %k_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %v_cache_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %page_ids_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dyn_extent: i64) {
    %cst = arith.constant 0.0 : f32
    %acc0 = arith.constant dense<0.0> : vector<64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64

    %acc = scf.for %p = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%a = %acc0) -> (vector<64xf32>) : i32 {
      %phys_p = tt.addptr %page_ids_ptr, %p : !tt.ptr<i32>, i32
      %phys = tt.load %phys_p : !tt.ptr<i32>
      %row_off = arith.muli %phys, %c16_i32 : i32
      // Dynamic shape dim 0: number of physical pages unknown at compile
      // time for both K and V.  The matcher must reach the dynamic-extent
      // check before rejecting (i.e. the "exactly two reads" gate passes).
      %kp = tt.make_tensor_ptr %k_cache_ptr, [%dyn_extent, %c64_i64], [%c64_i64, %c1_i64], [%row_off, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf32>>
      %vp = tt.make_tensor_ptr %v_cache_ptr, [%dyn_extent, %c64_i64], [%c64_i64, %c1_i64], [%row_off, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf32>>
      %kmr = triton_cpu.extract_memref %kp : <tensor<16x64xf32>> -> memref<?x64xf32, strided<[64, 1]>>
      %ki:2 = triton_cpu.extract_indices %kp : <tensor<16x64xf32>> -> index, index
      %k = vector.transfer_read %kmr[%ki#0, %ki#1], %cst {in_bounds = [true, true]} : memref<?x64xf32, strided<[64, 1]>>, vector<16x64xf32>
      %vmr = triton_cpu.extract_memref %vp : <tensor<16x64xf32>> -> memref<?x64xf32, strided<[64, 1]>>
      %vi:2 = triton_cpu.extract_indices %vp : <tensor<16x64xf32>> -> index, index
      %v = vector.transfer_read %vmr[%vi#0, %vi#1], %cst {in_bounds = [true, true]} : memref<?x64xf32, strided<[64, 1]>>, vector<16x64xf32>
      %k_row0 = vector.extract %k[0] : vector<64xf32> from vector<16x64xf32>
      %v_row0 = vector.extract %v[0] : vector<64xf32> from vector<16x64xf32>
      %kv = arith.addf %k_row0, %v_row0 : vector<64xf32>
      %new_a = arith.addf %a, %kv : vector<64xf32>
      scf.yield %new_a : vector<64xf32>
    }
    %otp = tt.make_tensor_ptr %out_ptr, [%c64_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<64xf32>>
    %omr = triton_cpu.extract_memref %otp : <tensor<64xf32>> -> memref<64xf32, strided<[1]>>
    %oi = triton_cpu.extract_indices %otp : <tensor<64xf32>> -> index
    vector.transfer_write %acc, %omr[%oi] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1]>>
    tt.return
  }
}
