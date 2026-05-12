// RUN: rm -rf %t && mkdir -p %t
// RUN: env KERNEL_AUX_FILE_DIR=%t triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-reductions=0 promotion-report=1" >/dev/null
// RUN: cat %t/gemm_fused_report_promotions.json | FileCheck %s --check-prefix=REPORT
// RUN: cat %t/reduction_report_promotions.json | FileCheck %s --check-prefix=REJECT
// RUN: rm -rf %t.off && mkdir -p %t.off
// RUN: env KERNEL_AUX_FILE_DIR=%t.off triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-reductions=0 promotion-report=0" >/dev/null
// RUN: not test -e %t.off/gemm_fused_report_promotions.json
// RUN: not test -e %t.off/reduction_report_promotions.json
// RUN: rm -rf %t.d3 && mkdir -p %t.d3
// RUN: env KERNEL_AUX_FILE_DIR=%t.d3 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-reductions=1 enable-promotion-profitability=1 promotion-report=1" >/dev/null
// RUN: cat %t.d3/gemm_fused_report_promotions.json | FileCheck %s --check-prefix=D3GEMM
// RUN: cat %t.d3/reduction_report_promotions.json | FileCheck %s --check-prefix=D3REDUCE
// RUN: rm -rf %t.multi && mkdir -p %t.multi
// RUN: env KERNEL_AUX_FILE_DIR=%t.multi TRITON_SPM_ATTENTION_Q_RESIDENT=1 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-reductions=0 promotion-report=1" >/dev/null
// RUN: cat %t.multi/attention_v2_window_report_promotions.json | FileCheck %s --check-prefix=MULTI

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
// REPORT:      "bytes": 1024
// REPORT:      "overhead": "two SPM buffers, one DMA descriptor per microM/K step, and one wait-at-top per step"
// REPORT:      "benefit": "pipelines A micro-tile staging while limiting A to the rows consumed by the current microM contract"
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

// D3GEMM:      "kernel": "gemm_fused_report"
// D3GEMM:      "source": "B tile window"
// D3GEMM:      "reason_code": "accepted_existing_schedule"
// D3GEMM:      "profitability": {
// D3GEMM:      "model": "phase35_p3_static_best_baseline_v1"
// D3GEMM:      "baseline": "best_legal_cache_schedule"
// D3GEMM:      "decision": "accept"
// D3GEMM:      "reason_code": "accepted_reused_loop_window"
// D3GEMM:      "dma_descriptors": 4
// D3GEMM:      "mmio_stores": 16
// D3GEMM:      "waits": 1
// D3GEMM:      "copy_bytes": 4096
// D3GEMM:      "spm_write_bytes": 4096
// D3GEMM:      "spm_read_bytes": 4096
// D3GEMM:      "avoided_repeated_read_bytes": 12288
// D3GEMM:      "live_spm_bytes": 4096
// D3GEMM:      "estimated_extra_ops": 4
// D3GEMM:      "measured_bank_conflicts": 0
// D3GEMM:      "source": "accumulator tile"
// D3GEMM:      "reason_code": "accepted_bounded_temporary"

// D3REDUCE:      "kernel": "reduction_report"
// D3REDUCE:      "promotions": [
// D3REDUCE-NEXT:   ],
// D3REDUCE:      "status": "rejected"
// D3REDUCE:      "pattern": "reduction_streaming"
// D3REDUCE:      "copy_in": "DMA"
// D3REDUCE:      "bytes": 256
// D3REDUCE:      "reason_code": "streaming_reduction_no_residency"
// D3REDUCE:      "profitability": {
// D3REDUCE:      "model": "phase35_p3_static_best_baseline_v1"
// D3REDUCE:      "baseline": "best_legal_cache_schedule"
// D3REDUCE:      "decision": "reject"
// D3REDUCE:      "dma_descriptors": 4
// D3REDUCE:      "mmio_stores": 16
// D3REDUCE:      "waits": 4
// D3REDUCE:      "copy_bytes": 256
// D3REDUCE:      "spm_write_bytes": 256
// D3REDUCE:      "spm_read_bytes": 256
// D3REDUCE:      "avoided_repeated_read_bytes": 0
// D3REDUCE:      "estimated_extra_ops": 4
// D3REDUCE:      "measured_bank_conflicts": 0
// D3REDUCE:      "uses": 1

// MULTI:      "kernel": "attention_v2_window_report"
// MULTI:      "source": "attention Q resident tile"
// MULTI:      "scope": "function-scope attention tile"
// MULTI:      "reason_code": "accepted_attention_v2_q_resident"
// MULTI:      "buffer_role": "resident_q_tile"
// MULTI-NOT:  "source": "attention K window tile"
// MULTI-NOT:  "source": "attention V window tile"
// MULTI-NOT:  "reason_code": "accepted_attention_v2_kv_window"
// MULTI:      "live_spm_bytes": 2048

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
  tt.func public @attention_v2_window_report(
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
