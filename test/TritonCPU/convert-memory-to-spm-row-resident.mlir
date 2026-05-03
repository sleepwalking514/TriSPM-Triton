// RUN: triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096" | FileCheck %s --check-prefix=ROW
// RUN: triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=DEFAULT
// RUN: rm -rf %t.row && mkdir -p %t.row
// RUN: env KERNEL_AUX_FILE_DIR=%t.row triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096 promotion-report=1" >/dev/null
// RUN: cat %t.row/layer_norm_row_resident_promotions.json | FileCheck %s --check-prefix=REPORT
// RUN: rm -rf %t.reject && mkdir -p %t.reject
// RUN: env KERNEL_AUX_FILE_DIR=%t.reject triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=128 promotion-report=1" >/dev/null
// RUN: cat %t.reject/layer_norm_row_resident_promotions.json | FileCheck %s --check-prefix=REJECT
// RUN: triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096 enable-promotion-profitability=1" | FileCheck %s --check-prefix=D3IR
// RUN: rm -rf %t.d3 && mkdir -p %t.d3
// RUN: env KERNEL_AUX_FILE_DIR=%t.d3 triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096 enable-promotion-profitability=1 promotion-report=1" >/dev/null
// RUN: cat %t.d3/layer_norm_row_resident_promotions.json | FileCheck %s --check-prefix=D3REPORT
// RUN: rm -rf %t.softmax && mkdir -p %t.softmax
// RUN: env KERNEL_AUX_FILE_DIR=%t.softmax triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=8192 promotion-report=1" | FileCheck %s --check-prefix=SOFTMAXIR
// RUN: cat %t.softmax/softmax_row_resident_plan_promotions.json | FileCheck %s --check-prefix=SOFTMAXREPORT
// RUN: triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096 row-resident-producer-pass=producer_store" | FileCheck %s --check-prefix=PRODUCER

// ROW-LABEL: @layer_norm_row_resident
// ROW:      scf.for
// ROW:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// ROW:        vector.transfer_write {{.*}} memref<8xf32, strided<[1]>, 3>
// ROW:      scf.for
// ROW:        vector.transfer_read {{.*}} memref<8xf32, strided<[1]>, 3>
// ROW:      scf.for
// ROW:        vector.transfer_read {{.*}} memref<8xf32, strided<[1]>, 3>
// ROW:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// ROW:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// ROW-NOT:  triton_cpu.dma_enqueue_2d
// ROW:      tt.return

// DEFAULT-LABEL: @layer_norm_row_resident
// DEFAULT-NOT:   triton_cpu.dma_enqueue_2d
// DEFAULT-NOT:   memref<8xf32, strided<[1]>, 3>
// DEFAULT:       vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// DEFAULT:       tt.return

// REPORT:      "schema": "triton_cpu_spm_promotion_d1"
// REPORT:      "contract": "debug/evidence sidecar; not a graph manifest or durable IR contract"
// REPORT:      "kernel": "layer_norm_row_resident"
// REPORT:      "status": "accepted"
// REPORT:      "source": "LayerNorm x row"
// REPORT:      "scope": "program-row"
// REPORT:      "shape": [64]
// REPORT:      "uses": 3
// REPORT:      "copy_in": "CPU/vector store"
// REPORT:      "copy_out": "none"
// REPORT:      "bytes": 256
// REPORT:      "reason_code": "accepted_fill_on_first_pass_row_resident"
// REPORT:      "residency_plan": {
// REPORT:      "producer_pass": "fill_on_first_pass"
// REPORT:      "consumer_passes": ["variance", "normalize"]
// REPORT:      "buffer_role": "resident_row"
// REPORT:      "rotation_policy": "none"
// REPORT:      "copy_in_mode": "cpu_direct"
// REPORT:      "required_spm_slots": 1
// REPORT:      "expected_markers": ["addrspace(3)", "no_dma_descriptors", "no_fence_iorw"]

// REJECT:      "kernel": "layer_norm_row_resident"
// REJECT:      "promotions": [
// REJECT-NEXT:   ],
// REJECT:      "status": "rejected"
// REJECT:      "pattern": "row_resident_reduction"
// REJECT:      "source": "LayerNorm x row"
// REJECT:      "scope": "program-row candidate"
// REJECT:      "shape": [64]
// REJECT:      "uses": 3
// REJECT:      "copy_in": "CPU/vector store"
// REJECT:      "copy_out": "none"
// REJECT:      "bytes": 256
// REJECT:      "reason_code": "spm_capacity_overflow"
// REJECT:      "residency_plan": {
// REJECT:      "producer_pass": "fill_on_first_pass"
// REJECT:      "consumer_passes": ["variance", "normalize"]
// REJECT:      "buffer_role": "resident_row"
// REJECT:      "rotation_policy": "none"

// D3IR-LABEL: @layer_norm_row_resident
// D3IR-NOT:   triton_cpu.dma_enqueue_2d
// D3IR-NOT:   memref<8xf32, strided<[1]>, 3>
// D3IR:       vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// D3IR:       tt.return

// D3REPORT:      "kernel": "layer_norm_row_resident"
// D3REPORT:      "promotions": [
// D3REPORT-NEXT:   ],
// D3REPORT:      "status": "rejected"
// D3REPORT:      "pattern": "row_resident_reduction"
// D3REPORT:      "reason_code": "insufficient_row_work"
// D3REPORT:      "residency_plan": {
// D3REPORT:      "producer_pass": "fill_on_first_pass"
// D3REPORT:      "consumer_passes": ["variance", "normalize"]
// D3REPORT:      "profitability": {
// D3REPORT:      "model": "d3_static_conservative_v1"
// D3REPORT:      "decision": "reject"
// D3REPORT:      "dma_descriptors": 0
// D3REPORT:      "mmio_stores": 0
// D3REPORT:      "waits": 0
// D3REPORT:      "fences": 0
// D3REPORT:      "copy_bytes": 256
// D3REPORT:      "avoided_repeated_read_bytes": 512
// D3REPORT:      "live_spm_bytes": 256
// D3REPORT:      "uses": 3

// SOFTMAXIR-LABEL: @softmax_row_resident_plan
// SOFTMAXIR-NOT:   triton_cpu.dma_enqueue_2d
// SOFTMAXIR:       scf.for
// SOFTMAXIR:         vector.transfer_read {{.*}} memref<1024xf32, strided<[1]>>
// SOFTMAXIR:         vector.transfer_write {{.*}} memref<64xf32, strided<[1]>, 3>
// SOFTMAXIR:       scf.for
// SOFTMAXIR:         vector.transfer_read {{.*}} memref<64xf32, strided<[1]>, 3>
// SOFTMAXIR:       scf.for
// SOFTMAXIR:         vector.transfer_read {{.*}} memref<64xf32, strided<[1]>, 3>
// SOFTMAXIR:       vector.transfer_write {{.*}} memref<1024xf32, strided<[1]>>
// SOFTMAXIR:       tt.return

// SOFTMAXREPORT:      "kernel": "softmax_row_resident_plan"
// SOFTMAXREPORT:      "status": "accepted"
// SOFTMAXREPORT:      "source": "Softmax x row"
// SOFTMAXREPORT:      "scope": "program-row"
// SOFTMAXREPORT:      "shape": [1024]
// SOFTMAXREPORT:      "uses": 3
// SOFTMAXREPORT:      "copy_in": "CPU/vector store"
// SOFTMAXREPORT:      "copy_out": "none"
// SOFTMAXREPORT:      "bytes": 4096
// SOFTMAXREPORT:      "reason_code": "accepted_fill_on_first_pass_row_resident"
// SOFTMAXREPORT:      "residency_plan": {
// SOFTMAXREPORT:      "producer_pass": "fill_on_first_pass"
// SOFTMAXREPORT:      "consumer_passes": ["exp_sum", "normalize_store"]
// SOFTMAXREPORT:      "buffer_role": "resident_row"
// SOFTMAXREPORT:      "rotation_policy": "none"
// SOFTMAXREPORT:      "copy_in_mode": "cpu_direct"
// SOFTMAXREPORT:      "required_spm_slots": 1

// PRODUCER-LABEL: @layer_norm_row_resident
// PRODUCER:      scf.for
// PRODUCER:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// PRODUCER-NOT:    memref<8xf32, strided<[1]>, 3>
// PRODUCER:      scf.for
// PRODUCER:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>>
// PRODUCER:        vector.transfer_write {{.*}} memref<8xf32, strided<[1]>, 3>
// PRODUCER:      scf.for
// PRODUCER:        vector.transfer_read {{.*}} memref<8xf32, strided<[1]>, 3>
// PRODUCER:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>
// PRODUCER:        vector.transfer_read {{.*}} memref<64xf32, strided<[1]>
// PRODUCER-NOT:  triton_cpu.dma_enqueue_2d
// PRODUCER:      tt.return

module {
  tt.func public @layer_norm_row_resident(
      %X: memref<64xf32, strided<[1], offset: 0>>,
      %Gamma: memref<64xf32, strided<[1], offset: 0>>,
      %Beta: memref<64xf32, strided<[1], offset: 0>>,
      %Out: memref<64xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %scale = arith.constant dense<6.400000e+01> : vector<1xf32>
    %acc_init = arith.constant dense<0.0> : vector<1xf32>

    %sum = scf.for %i = %c0 to %c64 step %c8
        iter_args(%acc = %acc_init) -> (vector<1xf32>) {
      %x = vector.transfer_read %X[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<8xf32>
      %sum_scalar = vector.reduction <add>, %x, %cst : vector<8xf32> into f32
      %sum_vec = vector.broadcast %sum_scalar : f32 to vector<1xf32>
      %next = arith.addf %acc, %sum_vec : vector<1xf32>
      scf.yield %next : vector<1xf32>
    }

    %mean = arith.divf %sum, %scale : vector<1xf32>
    %mean8 = vector.broadcast %mean : vector<1xf32> to vector<8xf32>

    %var_sum = scf.for %i = %c0 to %c64 step %c8
        iter_args(%acc = %acc_init) -> (vector<1xf32>) {
      %x = vector.transfer_read %X[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<8xf32>
      %centered = arith.subf %x, %mean8 : vector<8xf32>
      %sq = arith.mulf %centered, %centered : vector<8xf32>
      %var_scalar = vector.reduction <add>, %sq, %cst : vector<8xf32> into f32
      %var_vec = vector.broadcast %var_scalar : f32 to vector<1xf32>
      %next = arith.addf %acc, %var_vec : vector<1xf32>
      scf.yield %next : vector<1xf32>
    }

    %var = arith.divf %var_sum, %scale : vector<1xf32>
    %var8 = vector.broadcast %var : vector<1xf32> to vector<8xf32>

    scf.for %i = %c0 to %c64 step %c8 {
      %x = vector.transfer_read %X[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<8xf32>
      %g = vector.transfer_read %Gamma[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<8xf32>
      %b = vector.transfer_read %Beta[%i], %cst
          {in_bounds = [true]} : memref<64xf32, strided<[1], offset: 0>>, vector<8xf32>
      %centered = arith.subf %x, %mean8 : vector<8xf32>
      %norm = arith.mulf %centered, %var8 : vector<8xf32>
      %scaled = arith.mulf %norm, %g : vector<8xf32>
      %out = arith.addf %scaled, %b : vector<8xf32>
      vector.transfer_write %out, %Out[%i]
          {in_bounds = [true]} : vector<8xf32>, memref<64xf32, strided<[1], offset: 0>>
    }

    tt.return
  }
}

// -----

module {
  tt.func public @softmax_row_resident_plan(
      %X: memref<1024xf32, strided<[1], offset: 0>>,
      %Out: memref<1024xf32, strided<[1], offset: 0>>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %max_init = arith.constant dense<-3.40282347E+38> : vector<1xf32>
    %sum_init = arith.constant dense<0.0> : vector<1xf32>

    %row_max = scf.for %i = %c0 to %c1024 step %c64
        iter_args(%acc = %max_init) -> (vector<1xf32>) {
      %x = vector.transfer_read %X[%i], %neg_inf
          {in_bounds = [true]} : memref<1024xf32, strided<[1], offset: 0>>, vector<64xf32>
      %chunk_max = vector.reduction <maximumf>, %x, %neg_inf : vector<64xf32> into f32
      %chunk_vec = vector.broadcast %chunk_max : f32 to vector<1xf32>
      %next = arith.maximumf %acc, %chunk_vec : vector<1xf32>
      scf.yield %next : vector<1xf32>
    }

    %row_max64 = vector.broadcast %row_max : vector<1xf32> to vector<64xf32>
    %denom = scf.for %i = %c0 to %c1024 step %c64
        iter_args(%acc = %sum_init) -> (vector<1xf32>) {
      %x = vector.transfer_read %X[%i], %zero
          {in_bounds = [true]} : memref<1024xf32, strided<[1], offset: 0>>, vector<64xf32>
      %centered = arith.subf %x, %row_max64 : vector<64xf32>
      %exp = math.exp %centered : vector<64xf32>
      %chunk_sum = vector.reduction <add>, %exp, %zero : vector<64xf32> into f32
      %chunk_vec = vector.broadcast %chunk_sum : f32 to vector<1xf32>
      %next = arith.addf %acc, %chunk_vec : vector<1xf32>
      scf.yield %next : vector<1xf32>
    }

    %denom64 = vector.broadcast %denom : vector<1xf32> to vector<64xf32>
    scf.for %i = %c0 to %c1024 step %c64 {
      %x = vector.transfer_read %X[%i], %zero
          {in_bounds = [true]} : memref<1024xf32, strided<[1], offset: 0>>, vector<64xf32>
      %centered = arith.subf %x, %row_max64 : vector<64xf32>
      %exp = math.exp %centered : vector<64xf32>
      %y = arith.divf %exp, %denom64 : vector<64xf32>
      vector.transfer_write %y, %Out[%i]
          {in_bounds = [true]} : vector<64xf32>, memref<1024xf32, strided<[1], offset: 0>>
    }

    %unused = arith.addf %one, %zero : f32
    tt.return
  }
}
