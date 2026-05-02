// RUN: rm -rf %t && mkdir -p %t
// RUN: env KERNEL_AUX_FILE_DIR=%t triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 promotion-report=1" >/dev/null
// RUN: cat %t/gemm_fused_report_promotions.json | FileCheck %s --check-prefix=REPORT

// REPORT:      "kernel": "gemm_fused_report"
// REPORT:      "source": "B tile window"
// REPORT:      "scope": "loop-window"
// REPORT:      "shape": [16, 16, 4]
// REPORT:      "uses": 16
// REPORT:      "source": "A micro tile"
// REPORT:      "scope": "single-iteration"
// REPORT:      "shape": [8, 16]
// REPORT:      "uses": 1
// REPORT:      "source": "accumulator tile"
// REPORT:      "scope": "loop-window temporary"
// REPORT:      "shape": [32, 16]
// REPORT:      "uses": 9
// REPORT:      "rejections": [

module {
  tt.func public @gemm_fused_report(
      %A: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %B: memref<64x64xf32, strided<[64, 1], offset: 0>>,
      %C: memref<64x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<32x16xf32>

    %result = scf.for %k = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<32x16xf32>) {
      %a_tile = vector.transfer_read %A[%c0, %k], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<32x16xf32>
      %b_tile = vector.transfer_read %B[%k, %c0], %cst
          {in_bounds = [true, true]} : memref<64x64xf32, strided<[64, 1], offset: 0>>, vector<16x16xf32>

      %dot = vector.contract {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
      } %a_tile, %b_tile, %acc : vector<32x16xf32>, vector<16x16xf32> into vector<32x16xf32>

      scf.yield %dot : vector<32x16xf32>
    }

    vector.transfer_write %result, %C[%c0, %c0]
        {in_bounds = [true, true]} : vector<32x16xf32>, memref<64x64xf32, strided<[64, 1], offset: 0>>
    tt.return
  }
}
