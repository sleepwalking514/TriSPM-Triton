// RUN: rm -rf %t && mkdir -p %t
// RUN: env KERNEL_AUX_FILE_DIR=%t triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-reductions=0 promotion-report=1" >/dev/null
// RUN: cat %t/gemm_fused_report_promotions.json | FileCheck %s --check-prefix=REPORT
// RUN: cat %t/reduction_report_promotions.json | FileCheck %s --check-prefix=REJECT
// RUN: rm -rf %t.off && mkdir -p %t.off
// RUN: env KERNEL_AUX_FILE_DIR=%t.off triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-reductions=0 promotion-report=0" >/dev/null
// RUN: not test -e %t.off/gemm_fused_report_promotions.json
// RUN: not test -e %t.off/reduction_report_promotions.json

// REPORT:      "schema_version": 1
// REPORT:      "schema": "triton_cpu_spm_promotion_d1"
// REPORT:      "contract": "debug/evidence sidecar; not a graph manifest or durable IR contract"
// REPORT:      "kernel": "gemm_fused_report"
// REPORT:      "status": "accepted"
// REPORT:      "source": "B tile window"
// REPORT:      "scope": "loop-window"
// REPORT:      "shape": [16, 16, 4]
// REPORT:      "uses": 16
// REPORT:      "bytes": 4096
// REPORT:      "reason_code": "accepted_existing_schedule"
// REPORT:      "field_kinds": {
// REPORT:      "bytes": "exact-static"
// REPORT:      "overhead": "estimated-structural"
// REPORT:      "status": "accepted"
// REPORT:      "source": "A micro tile"
// REPORT:      "scope": "single-iteration"
// REPORT:      "shape": [8, 16]
// REPORT:      "uses": 1
// REPORT:      "bytes": 512
// REPORT:      "status": "accepted"
// REPORT:      "source": "accumulator tile"
// REPORT:      "scope": "loop-window temporary"
// REPORT:      "shape": [32, 16]
// REPORT:      "uses": 9
// REPORT:      "bytes": 2048
// REPORT:      "rejections": [

// REJECT:      "schema_version": 1
// REJECT:      "kernel": "reduction_report"
// REJECT:      "promotions": [
// REJECT-NEXT:   ],
// REJECT:      "rejections": [
// REJECT:      "status": "rejected"
// REJECT:      "pattern": "reduction_streaming"
// REJECT:      "scope": "candidate"
// REJECT:      "uses": 0
// REJECT:      "copy_in": "none"
// REJECT:      "copy_out": "none"
// REJECT:      "bytes": 0
// REJECT:      "reason_code": "policy_disabled"
// REJECT:      "reason": "reduction/streaming SPM promotion is disabled by default; leave the candidate on the cache path"
// REJECT:      "shape": "exact-if-known"

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

// -----

module {
  tt.func public @reduction_report(
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
