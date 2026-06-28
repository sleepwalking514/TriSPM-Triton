// RUN: rm -rf %t && mkdir -p %t
// RUN: env KERNEL_AUX_FILE_DIR=%t triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 promotion-report=1" >/dev/null
// RUN: cat %t/gemm_fused_report_promotions.json | FileCheck %s --check-prefix=REPORT
// RUN: cat %t/reduction_report_promotions.json | FileCheck %s --check-prefix=GENERIC
// RUN: rm -rf %t.off && mkdir -p %t.off
// RUN: env KERNEL_AUX_FILE_DIR=%t.off triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 promotion-report=0" >/dev/null
// RUN: not test -e %t.off/gemm_fused_report_promotions.json
// RUN: not test -e %t.off/reduction_report_promotions.json
// RUN: rm -rf %t.d3 && mkdir -p %t.d3
// RUN: env KERNEL_AUX_FILE_DIR=%t.d3 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 enable-promotion-profitability=1 promotion-report=1" >/dev/null
// RUN: cat %t.d3/gemm_fused_report_promotions.json | FileCheck %s --check-prefix=D3GEMM
// RUN: cat %t.d3/reduction_report_promotions.json | FileCheck %s --check-prefix=D3GENERIC
// RUN: rm -rf %t.multi && mkdir -p %t.multi
// RUN: env KERNEL_AUX_FILE_DIR=%t.multi TRITON_SPM_ATTENTION_Q_RESIDENT=1 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 promotion-report=1" >/dev/null
// RUN: cat %t.multi/attention_v2_window_report_promotions.json | FileCheck %s --check-prefix=MULTI
// RUN: rm -rf %t.qk && mkdir -p %t.qk
// RUN: env KERNEL_AUX_FILE_DIR=%t.qk TRITON_SPM_ATTENTION_QK_TILE=1 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 promotion-report=1" >/dev/null
// RUN: cat %t.qk/fused_attention_qk_report_promotions.json | FileCheck %s --check-prefix=QKREPORT
// RUN: rm -rf %t.pv && mkdir -p %t.pv
// RUN: env KERNEL_AUX_FILE_DIR=%t.pv TRITON_SPM_ATTENTION_PV_GENERATED_TILE=1 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 micro-m=8 window-k=4 promotion-report=1" >/dev/null
// RUN: cat %t.pv/fused_attention_pv_generated_report_promotions.json | FileCheck %s --check-prefix=PVREPORT

// REPORT:      "schema_version": 1
// REPORT:      "schema": "triton_cpu_spm_promotion_d1"
// REPORT:      "contract": "debug/evidence sidecar; not a graph manifest or durable IR contract"
// REPORT:      "kernel": "gemm_fused_report"
// REPORT:      "status": "accepted"
// REPORT:      "source": "B tile window"
// REPORT:      "scope": "loop-window"
// REPORT:      "footprint_class": "contraction_reuse_window"
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
// REPORT:      "contractions": [
// REPORT:      "pattern": "gemm_like_contraction"
// REPORT:      "operation": "vector.contract"
// REPORT:      "mnk_shape": [32, 16, 16]
// REPORT:      "source": "memory_backed"
// REPORT:      "output_relation": "loop_carried_accumulator_to_memory_store"
// REPORT:      "schedule_status": "existing_gemm_schedule_candidate"
// REPORT:      "affine_tile_candidates": [
// REPORT:      "candidate_id": "affine_tile_0"
// REPORT:      "loop_ordinal": 0
// REPORT:      "op_ordinal": 0
// REPORT:      "op": "vector.transfer_read"
// REPORT:      "schedule_class": "streaming_ping_pong"
// REPORT:      "reason_code": "candidate_streaming_ping_pong_contraction_fallback"

// GENERIC:      "schema_version": 1
// GENERIC:      "kernel": "reduction_report"
// GENERIC:      "promotions": [
// GENERIC:      "status": "accepted"
// GENERIC:      "source": "generic affine tile"
// GENERIC:      "scope": "loop-local streaming tile"
// GENERIC:      "footprint_class": "streaming_ping_pong"
// GENERIC:      "shape": [16]
// GENERIC:      "uses": 1
// GENERIC:      "copy_in": "DMA"
// GENERIC:      "copy_out": "none"
// GENERIC:      "bytes": 128
// GENERIC:      "reason_code": "accepted_generic_affine_tile_streaming"

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

// D3GENERIC:      "kernel": "reduction_report"
// D3GENERIC:      "status": "accepted"
// D3GENERIC:      "source": "generic affine tile"
// D3GENERIC:      "footprint_class": "streaming_ping_pong"
// D3GENERIC:      "copy_in": "DMA"
// D3GENERIC:      "bytes": 128
// D3GENERIC:      "reason_code": "accepted_generic_affine_tile_streaming"

// MULTI:      "kernel": "attention_v2_window_report"
// MULTI:      "source": "attention Q resident tile"
// MULTI:      "scope": "function-scope attention tile"
// MULTI:      "reason_code": "accepted_attention_v2_q_resident"
// MULTI:      "buffer_role": "resident_q_tile"
// MULTI-NOT:  "source": "attention K window tile"
// MULTI-NOT:  "source": "attention V window tile"
// MULTI-NOT:  "reason_code": "accepted_attention_v2_kv_window"
// MULTI:      "live_spm_bytes": 2048
// MULTI:      "contractions": [
// MULTI:      "mnk_shape": [16, 16, 32]
// MULTI:      "lhs": {
// MULTI:      "source": "memory_backed"
// MULTI:      "rhs": {
// MULTI:      "source": "memory_backed"
// MULTI:      "output_relation": "local_consumer"
// MULTI:      "consumer": "vector.contract"
// MULTI:      "reason_code": "local_consumer_output"
// MULTI:      "mnk_shape": [16, 32, 16]
// MULTI:      "lhs": {
// MULTI:      "source": "generated_contraction"
// MULTI:      "rhs": {
// MULTI:      "source": "memory_backed"
// MULTI:      "output_relation": "loop_carried_accumulator_to_memory_store"
// MULTI:      "reason_code": "generated_operand_not_resident"

// QKREPORT:      "kernel": "fused_attention_qk_report"
// QKREPORT:      "source": "attention QK lhs tile"
// QKREPORT:      "scope": "function-scope QK tile"
// QKREPORT:      "shape": [16, 16]
// QKREPORT:      "reason_code": "accepted_attention_qk_tile_staging"
// QKREPORT:      "source": "attention QK rhs tile"
// QKREPORT:      "shape": [16, 32]
// QKREPORT:      "reason_code": "accepted_attention_qk_tile_staging"
// QKREPORT:      "contractions": [
// QKREPORT:      "status": "accepted"
// QKREPORT:      "mnk_shape": [16, 32, 16]
// QKREPORT:      "schedule_status": "accepted_attention_qk_tile_staging"

// PVREPORT:      "kernel": "fused_attention_pv_generated_report"
// PVREPORT:      "source": "attention PV generated tile"
// PVREPORT:      "scope": "function-scope PV generated operand"
// PVREPORT:      "shape": [16, 32]
// PVREPORT:      "copy_in": "CPU/vector store"
// PVREPORT:      "copy_out": "CPU/vector transfer read"
// PVREPORT:      "reason_code": "accepted_attention_pv_generated_operand_residency"
// PVREPORT:      "buffer_role": "generated_probability_tile"
// PVREPORT:      "contractions": [
// PVREPORT:      "status": "rejected"
// PVREPORT:      "mnk_shape": [16, 32, 16]
// PVREPORT:      "reason_code": "local_consumer_output"
// PVREPORT:      "status": "accepted"
// PVREPORT:      "mnk_shape": [16, 16, 32]
// PVREPORT:      "schedule_status": "accepted_attention_pv_generated_operand_residency"

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
  tt.func public @fused_attention_qk_report(
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

module {
  tt.func public @fused_attention_pv_generated_report(
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
