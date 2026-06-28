// RUN: rm -rf %t.report && mkdir -p %t.report
// RUN: env KERNEL_AUX_FILE_DIR=%t.report TRITON_SPM_PAGED_KV_DECODE=0 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-row-resident-reductions=0 generic-affine-tile-min-bytes=128 promotion-report=1" >/dev/null
// RUN: cat %t.report/generic_affine_report_promotions.json | FileCheck %s --check-prefix=REPORT
// RUN: cat %t.report/generic_affine_alias_promotions.json | FileCheck %s --check-prefix=ALIAS
// RUN: cat %t.report/generic_affine_unknown_alias_promotions.json | FileCheck %s --check-prefix=UNKNOWNALIAS
// RUN: cat %t.report/generic_affine_write_before_read_promotions.json | FileCheck %s --check-prefix=WRITEBEFORE
// RUN: cat %t.report/generic_affine_multi_use_promotions.json | FileCheck %s --check-prefix=MULTIUSE
// RUN: cat %t.report/generic_affine_multistream_promotions.json | FileCheck %s --check-prefix=MULTISTREAM
// RUN: cat %t.report/generic_affine_map_promotions.json | FileCheck %s --check-prefix=MAP
// RUN: cat %t.report/generic_affine_colmajor_promotions.json | FileCheck %s --check-prefix=COLMAJOR
// RUN: rm -rf %t.off && mkdir -p %t.off
// RUN: env KERNEL_AUX_FILE_DIR=%t.off TRITON_SPM_PAGED_KV_DECODE=0 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-row-resident-reductions=0 promotion-report=0" >/dev/null
// RUN: not test -e %t.off/generic_affine_report_promotions.json
// RUN: env TRITON_SPM_PAGED_KV_DECODE=0 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-row-resident-reductions=0 generic-affine-tile-min-bytes=0 promotion-report=0" | FileCheck %s --check-prefix=LOWER

// REPORT:      "kernel": "generic_affine_report"
// REPORT:      "promotions": [
// REPORT-NEXT:   ],
// REPORT:      "rejections": [
// REPORT:      "pattern": "generic_affine_tile"
// REPORT:      "footprint_class": "streaming_ping_pong"
// REPORT:      "copy_in": "DMA"
// REPORT:      "bytes": 64
// REPORT:      "reason_code": "small_tile_spm_overhead"
// REPORT:      "contractions": [
// REPORT-NEXT:   ],
// REPORT:      "affine_tile_candidates": [
// REPORT:      "candidate_id": "affine_tile_0"
// REPORT:      "loop_ordinal": 0
// REPORT:      "op_ordinal": 0
// REPORT:      "op": "vector.transfer_read"
// REPORT:      "status": "candidate"
// REPORT:      "schedule_class": "streaming_ping_pong"
// REPORT:      "reason_code": "candidate_streaming_ping_pong"
// REPORT:      "full_tile": true
// REPORT:      "static_trip_count": true
// REPORT:      "static_stride": true
// REPORT:      "row_major_inner_contiguous": true
// REPORT:      "dma_representable": true

// ALIAS:      "kernel": "generic_affine_alias"
// ALIAS:      "candidate_id": "affine_tile_0"
// ALIAS:      "status": "candidate"
// ALIAS:      "schedule_class": "streaming_ping_pong"
// ALIAS:      "reason_code": "candidate_streaming_ping_pong_post_write_prefetch"
// ALIAS:      "post_write_prefetch_safe": true
// ALIAS:      "candidate_id": "affine_tile_1"
// ALIAS:      "op": "vector.transfer_write"
// ALIAS:      "status": "report_only"
// ALIAS:      "schedule_class": "generated_value_residency"
// ALIAS:      "footprint_kind": "generated_value"

// UNKNOWNALIAS:      "kernel": "generic_affine_unknown_alias"
// UNKNOWNALIAS:      "candidate_id": "affine_tile_0"
// UNKNOWNALIAS:      "status": "candidate"
// UNKNOWNALIAS:      "reason_code": "candidate_streaming_ping_pong_post_write_prefetch"
// UNKNOWNALIAS:      "may_alias_store": true
// UNKNOWNALIAS:      "post_write_prefetch_safe": true

// WRITEBEFORE:      "kernel": "generic_affine_write_before_read"
// WRITEBEFORE:      "candidate_id": "affine_tile_0"
// WRITEBEFORE:      "status": "rejected"
// WRITEBEFORE:      "reason_code": "unknown_transfer_write_may_alias"
// WRITEBEFORE:      "may_alias_store": true
// WRITEBEFORE:      "post_write_prefetch_safe": false

// MULTIUSE:      "kernel": "generic_affine_multi_use"
// MULTIUSE:      "candidate_id": "affine_tile_0"
// MULTIUSE:      "status": "candidate"
// MULTIUSE:      "schedule_class": "streaming_ping_pong"
// MULTIUSE:      "reason_code": "candidate_streaming_ping_pong"

// MULTISTREAM:      "kernel": "generic_affine_multistream"
// MULTISTREAM:      "candidate_id": "affine_tile_0"
// MULTISTREAM:      "status": "candidate"
// MULTISTREAM:      "candidate_id": "affine_tile_1"
// MULTISTREAM:      "status": "candidate"

// MAP:      "kernel": "generic_affine_map"
// MAP:      "candidate_id": "affine_tile_0"
// MAP:      "status": "candidate"
// MAP:      "reason_code": "candidate_streaming_ping_pong"
// MAP:      "transfer_map": "contiguous_projected_slice"
// MAP:      "dma_filled_layout_supported": true

// COLMAJOR:      "kernel": "generic_affine_colmajor"
// COLMAJOR:      "candidate_id": "affine_tile_0"
// COLMAJOR:      "status": "candidate"
// COLMAJOR:      "reason_code": "candidate_streaming_ping_pong"
// COLMAJOR:      "row_major_inner_contiguous": false
// COLMAJOR:      "dma_filled_layout_supported": true

// LOWER-LABEL: @generic_affine_report
// LOWER:       triton_cpu.dma_enqueue_2d
// LOWER:       scf.for
// LOWER:         triton_cpu.dma_wait
// LOWER:         scf.if
// LOWER:           triton_cpu.dma_enqueue_2d
// LOWER:         vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER-LABEL: @generic_affine_alias
// LOWER:       vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER:       vector.transfer_write
// LOWER:       scf.if
// LOWER:       triton_cpu.dma_enqueue_2d
// LOWER-LABEL: @generic_affine_unknown_alias
// LOWER:       vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER:       vector.transfer_write
// LOWER:       scf.if
// LOWER:       triton_cpu.dma_enqueue_2d
// LOWER-LABEL: @generic_affine_write_before_read
// LOWER-NOT:   memref<16xf32, strided<[1]>, 3>
// LOWER-LABEL: @generic_affine_multi_use
// LOWER:       vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER-LABEL: @generic_affine_multistream
// LOWER:       triton_cpu.dma_enqueue_2d
// LOWER:       triton_cpu.dma_enqueue_2d
// LOWER:       vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER:       vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER-LABEL: @generic_affine_colmajor
// LOWER:       vector.transfer_read {{.*}} memref<16x16xf32, strided<[1, 16]>, 3>
// LOWER-LABEL: @generic_affine_dynamic_bound
// LOWER:       scf.if
// LOWER:         triton_cpu.dma_enqueue_2d
// LOWER:         scf.for
// LOWER:           triton_cpu.dma_wait
// LOWER:           vector.transfer_read {{.*}} memref<16xf32, strided<[1]>, 3>
// LOWER:       else
// LOWER:         scf.yield

module {
  tt.func public @generic_affine_report(
      %X: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %tile = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %tile : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_alias(
      %X: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %tile = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %tile : vector<16xf32>
      vector.transfer_write %sum, %X[%i]
          {in_bounds = [true]} : vector<16xf32>, memref<64xf32, strided<[1], offset: 0>>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_unknown_alias(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %Y: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %tile = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %tile : vector<16xf32>
      vector.transfer_write %sum, %Y[%i]
          {in_bounds = [true]} : vector<16xf32>, memref<64xf32, strided<[1], offset: 0>>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_map(
      %X: memref<4x64xf32, strided<[64, 1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %tile = vector.transfer_read %X[%c0, %i], %pad
          {permutation_map = affine_map<(d0, d1) -> (d1)>, in_bounds = [true]}
          : memref<4x64xf32, strided<[64, 1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %tile : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_write_before_read(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %Y: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      vector.transfer_write %acc, %Y[%i]
          {in_bounds = [true]} : vector<16xf32>, memref<64xf32, strided<[1], offset: 0>>
      %tile = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %tile : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_multi_use(
      %X: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %tile = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %twice = arith.addf %tile, %tile : vector<16xf32>
      %sum = arith.addf %acc, %twice : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_multistream(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %Y: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %x = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %y = vector.transfer_read %Y[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %xy = arith.addf %x, %y : vector<16xf32>
      %sum = arith.addf %acc, %xy : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_colmajor(
      %X: memref<64x64xf32, strided<[1, 64], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16x16xf32>

    %result = scf.for %i = %c0 to %c64 step %c16
        iter_args(%acc = %acc_init) -> (vector<16x16xf32>) {
      %tile = vector.transfer_read %X[%i, %c0], %pad
          {in_bounds = [true, true]}
          : memref<64x64xf32, strided<[1, 64], offset: 0>>, vector<16x16xf32>
      %sum = arith.addf %acc, %tile : vector<16x16xf32>
      scf.yield %sum : vector<16x16xf32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @generic_affine_dynamic_bound(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %ub: index) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %pad = arith.constant 0.0 : f32
    %acc_init = arith.constant dense<0.0> : vector<16xf32>

    %result = scf.for %i = %c0 to %ub step %c16
        iter_args(%acc = %acc_init) -> (vector<16xf32>) {
      %tile = vector.transfer_read %X[%i], %pad
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<16xf32>
      %sum = arith.addf %acc, %tile : vector<16xf32>
      scf.yield %sum : vector<16xf32>
    }
    tt.return
  }
}
