// RUN: env TRITON_SPM_INDIRECT_TILE=1 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=ACCEPTED
// RUN: env TRITON_SPM_INDIRECT_TILE=1 TRITON_ENABLE_SPM_PROMOTION_PROFITABILITY=1 TRITON_SPM_INDIRECT_TILE_MIN_BYTES=4096 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-row-resident-reductions=0" | FileCheck %s --check-prefix=REJECTED

// ============================================================================
// Accepted case: a fixed-trip gather loop with table[idx, :] body.
// Staged-bytes = 8 * 64 * 4 = 2048; below default threshold 8192 but with
// profitability OFF this is accepted and the loop body reads from SPM.
// ============================================================================
// ACCEPTED-LABEL: @embedding_bag_smoke
// ACCEPTED:   scf.for
// ACCEPTED:     triton_cpu.dma_enqueue_2d
// ACCEPTED:   triton_cpu.dma_wait
// ACCEPTED:   scf.for
// ACCEPTED:     memref.reinterpret_cast
// ACCEPTED:     vector.transfer_read {{.*}} memref<{{.*}}3>

// REJECTED-LABEL: @embedding_bag_smoke
// REJECTED-NOT: triton_cpu.dma_enqueue_2d
// REJECTED-NOT: memref.reinterpret_cast {{.*}}3
// REJECTED:   scf.for
// REJECTED:     tt.load {{.*}} !tt.ptr<i32>
// REJECTED:     tt.make_tensor_ptr
// REJECTED:     vector.transfer_read

module {
  tt.func public @embedding_bag_smoke(
      %table_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %indices_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %offsets_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.0 : f32
    %acc0 = arith.constant dense<0.0> : vector<64xf32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %c64 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c256_i64 = arith.constant 256 : i64
    %bag = tt.get_program_id x : i32
    %off_ptr = tt.addptr %offsets_ptr, %bag : !tt.ptr<i32>, i32
    %off_start = tt.load %off_ptr : !tt.ptr<i32>
    %idx_base = tt.addptr %indices_ptr, %off_start : !tt.ptr<i32>, i32
    %acc = scf.for %k = %c0 to %c8 step %c1 iter_args(%a = %acc0) -> (vector<64xf32>) : i32 {
      %idx_k = tt.addptr %idx_base, %k : !tt.ptr<i32>, i32
      %idx = tt.load %idx_k : !tt.ptr<i32>
      %row_off = arith.muli %idx, %c64 : i32
      %tp = tt.make_tensor_ptr %table_ptr, [%c262144_i64], [%c1_i64], [%row_off] {order = array<i32: 0>} : <tensor<64xf32>>
      %mr = triton_cpu.extract_memref %tp : <tensor<64xf32>> -> memref<262144xf32, strided<[1]>>
      %off_idx = triton_cpu.extract_indices %tp : <tensor<64xf32>> -> index
      %v = vector.transfer_read %mr[%off_idx], %cst {in_bounds = [true]} : memref<262144xf32, strided<[1]>>, vector<64xf32>
      %new = arith.addf %a, %v : vector<64xf32>
      scf.yield %new : vector<64xf32>
    }
    %out_off = arith.muli %bag, %c64 : i32
    %otp = tt.make_tensor_ptr %out_ptr, [%c256_i64], [%c1_i64], [%out_off] {order = array<i32: 0>} : <tensor<64xf32>>
    %omr = triton_cpu.extract_memref %otp : <tensor<64xf32>> -> memref<256xf32, strided<[1]>>
    %oi = triton_cpu.extract_indices %otp : <tensor<64xf32>> -> index
    vector.transfer_write %acc, %omr[%oi] {in_bounds = [true]} : vector<64xf32>, memref<256xf32, strided<[1]>>
    tt.return
  }
}

// -----

// ============================================================================
// C2: grouped-bag (BAG_GROUP=2) pattern, double-buffered lowering.
//
// Run with both TRITON_SPM_INDIRECT_TILE=1 and
// TRITON_SPM_INDIRECT_TILE_DOUBLE_BUFFER=1 the pass should:
//   - emit one pre-loop dma_enqueue_2d setup loop into tile 0
//   - rewrite the outer scf.for to have a single i32 iter-arg (cur_buf)
//   - inside, emit dma_wait, conditional next-bag prefetch via scf.if, and
//     read the consume side from SPM (addrspace 3) at curBuf*tileBytes+...
// ============================================================================
// RUN: env TRITON_SPM_INDIRECT_TILE=1 TRITON_SPM_INDIRECT_TILE_DOUBLE_BUFFER=1 triton-opt %s -split-input-file -triton-cpu-convert-memory-to-spm="spm-base=0x40000000 spm-size=65536 enable-row-resident-reductions=0" 2>&1 | FileCheck %s --check-prefix=DOUBLE_BUFFER

// DOUBLE_BUFFER-LABEL: @embedding_bag_grouped
// Pre-loop prefetch: one dma_enqueue_2d setup loop before the outer scf.for.
// DOUBLE_BUFFER:   scf.for
// DOUBLE_BUFFER:     triton_cpu.dma_enqueue_2d
// Outer loop with one i32 iter-arg:
// DOUBLE_BUFFER:   scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (i32)
// In-iter wait + conditional next-bag prefetch:
// DOUBLE_BUFFER:     triton_cpu.dma_wait
// DOUBLE_BUFFER:     scf.if
// DOUBLE_BUFFER:       triton_cpu.dma_enqueue_2d
// Consume reads from SPM (addrspace 3):
// DOUBLE_BUFFER:     scf.for
// DOUBLE_BUFFER:       vector.transfer_read {{.*}} memref<{{.*}}3>

module {
  tt.func public @embedding_bag_grouped(
      %table_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %indices_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %offsets_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.0 : f32
    %acc0 = arith.constant dense<0.0> : vector<64xf32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c8 = arith.constant 8 : i32
    %c64 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c256_i64 = arith.constant 256 : i64
    %pid = tt.get_program_id x : i32
    %pbg = arith.muli %pid, %c2 : i32
    scf.for %local_b = %c0 to %c2 step %c1 : i32 {
      %bag = arith.addi %pbg, %local_b : i32
      %off_ptr = tt.addptr %offsets_ptr, %bag : !tt.ptr<i32>, i32
      %off_start = tt.load %off_ptr : !tt.ptr<i32>
      %idx_base = tt.addptr %indices_ptr, %off_start : !tt.ptr<i32>, i32
      %acc = scf.for %k = %c0 to %c8 step %c1 iter_args(%a = %acc0) -> (vector<64xf32>) : i32 {
        %idx_k = tt.addptr %idx_base, %k : !tt.ptr<i32>, i32
        %idx = tt.load %idx_k : !tt.ptr<i32>
        %row_off = arith.muli %idx, %c64 : i32
        %tp = tt.make_tensor_ptr %table_ptr, [%c262144_i64], [%c1_i64], [%row_off] {order = array<i32: 0>} : <tensor<64xf32>>
        %mr = triton_cpu.extract_memref %tp : <tensor<64xf32>> -> memref<262144xf32, strided<[1]>>
        %off_idx = triton_cpu.extract_indices %tp : <tensor<64xf32>> -> index
        %v = vector.transfer_read %mr[%off_idx], %cst {in_bounds = [true]} : memref<262144xf32, strided<[1]>>, vector<64xf32>
        %new = arith.addf %a, %v : vector<64xf32>
        scf.yield %new : vector<64xf32>
      }
      %out_off = arith.muli %bag, %c64 : i32
      %otp = tt.make_tensor_ptr %out_ptr, [%c256_i64], [%c1_i64], [%out_off] {order = array<i32: 0>} : <tensor<64xf32>>
      %omr = triton_cpu.extract_memref %otp : <tensor<64xf32>> -> memref<256xf32, strided<[1]>>
      %oi = triton_cpu.extract_indices %otp : <tensor<64xf32>> -> index
      vector.transfer_write %acc, %omr[%oi] {in_bounds = [true]} : vector<64xf32>, memref<256xf32, strided<[1]>>
    }
    tt.return
  }
}
