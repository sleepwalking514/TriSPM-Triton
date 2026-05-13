// RUN: rm -rf %t.multi && mkdir -p %t.multi
// RUN: env KERNEL_AUX_FILE_DIR=%t.multi triton-opt %s -split-input-file -triton-cpu-spm-tensor-placement="enable-reductions=0" >/dev/null
// RUN: cat %t.multi/attention_v2_window_placement_tiers.json | FileCheck %s --check-prefix=MULTI
// RUN: rm -rf %t.qk && mkdir -p %t.qk
// RUN: env KERNEL_AUX_FILE_DIR=%t.qk TRITON_SPM_ATTENTION_QK_TILE=1 triton-opt %s -split-input-file -triton-cpu-spm-tensor-placement="enable-reductions=0" >/dev/null
// RUN: cat %t.qk/fused_attention_qk_placement_tiers.json | FileCheck %s --check-prefix=QK-TIER

// MULTI:      {
// MULTI-NEXT:   "0": 2
// MULTI-NEXT: }

// QK-TIER:      {
// QK-TIER-NEXT:   "0": 3,
// QK-TIER-NEXT:   "1": 3
// QK-TIER-NEXT: }

module {
  tt.func public @attention_v2_window_placement(
      %Q: memref<64x32xf32, strided<[32, 1], offset: 0>>,
      %K: memref<32x64xf32, strided<[1, 32], offset: 0>>,
      %V: memref<64x32xf32, strided<[32, 1], offset: 0>>,
      %O: memref<64x32xf32, strided<[32, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %score_init = arith.constant dense<0.0> : vector<16x16xf32>
    %acc_init = arith.constant dense<0.0> : vector<16x32xf32>

    %q_tile = vector.transfer_read %Q[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<64x32xf32, strided<[32, 1], offset: 0>>, vector<16x32xf32>

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
      } %q_tile, %k_tile, %score_init : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>

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

module {
  tt.func public @fused_attention_qk_placement(
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
