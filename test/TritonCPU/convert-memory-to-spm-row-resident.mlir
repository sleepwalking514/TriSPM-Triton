// RUN: triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096" | FileCheck %s --check-prefix=ROW
// RUN: triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=DEFAULT
// RUN: rm -rf %t.row && mkdir -p %t.row
// RUN: env KERNEL_AUX_FILE_DIR=%t.row triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=4096 promotion-report=1" >/dev/null
// RUN: cat %t.row/layer_norm_row_resident_promotions.json | FileCheck %s --check-prefix=REPORT
// RUN: rm -rf %t.reject && mkdir -p %t.reject
// RUN: env KERNEL_AUX_FILE_DIR=%t.reject triton-opt %s -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-reductions=0 enable-row-resident-reductions=1 row-resident-max-bytes=128 promotion-report=1" >/dev/null
// RUN: cat %t.reject/layer_norm_row_resident_promotions.json | FileCheck %s --check-prefix=REJECT

// ROW-LABEL: @layer_norm_row_resident
// ROW:       triton_cpu.dma_enqueue_2d
// ROW-NEXT: triton_cpu.dma_wait
// ROW-NOT:  triton_cpu.dma_enqueue_2d
// ROW:      scf.for
// ROW:        vector.transfer_read {{.*}} memref<8xf32, strided<[1]>, 3>
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
// REPORT:      "copy_in": "DMA"
// REPORT:      "copy_out": "none"
// REPORT:      "bytes": 256
// REPORT:      "reason_code": "accepted_d2_opt_in_row_resident"

// REJECT:      "kernel": "layer_norm_row_resident"
// REJECT:      "promotions": [
// REJECT-NEXT:   ],
// REJECT:      "status": "rejected"
// REJECT:      "pattern": "row_resident_reduction"
// REJECT:      "source": "LayerNorm x row"
// REJECT:      "scope": "program-row candidate"
// REJECT:      "shape": [64]
// REJECT:      "uses": 3
// REJECT:      "copy_in": "DMA"
// REJECT:      "copy_out": "none"
// REJECT:      "bytes": 256
// REJECT:      "reason_code": "spm_capacity_overflow"

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
