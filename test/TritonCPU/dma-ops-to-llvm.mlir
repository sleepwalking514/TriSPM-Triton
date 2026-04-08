// RUN: triton-opt %s -split-input-file -triton-cpu-dma-ops-to-llvm | FileCheck %s

// ============================================================================
// Test: DmaEnqueue2DOp lowers to volatile MMIO stores + fences
// ============================================================================

// CHECK-LABEL: @dma_enqueue_2d_basic
// CHECK:       %[[SRC_ADDR:.+]] = llvm.mlir.constant(4026531840 : i64) : i64
// CHECK-NEXT:  %[[SRC_PTR:.+]] = llvm.inttoptr %[[SRC_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %arg1, %[[SRC_PTR]] : i64, !llvm.ptr
// CHECK:       %[[DST_ADDR:.+]] = llvm.mlir.constant(4026531848 : i64) : i64
// CHECK-NEXT:  %[[DST_PTR:.+]] = llvm.inttoptr %[[DST_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %arg0, %[[DST_PTR]] : i64, !llvm.ptr
// CHECK:       %[[SRCSTRIDE_ADDR:.+]] = llvm.mlir.constant(4026531872 : i64) : i64
// CHECK-NEXT:  %[[SRCSTRIDE_PTR:.+]] = llvm.inttoptr %[[SRCSTRIDE_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %arg4, %[[SRCSTRIDE_PTR]] : i64, !llvm.ptr
// CHECK:       %[[DSTSTRIDE_ADDR:.+]] = llvm.mlir.constant(4026531880 : i64) : i64
// CHECK-NEXT:  %[[DSTSTRIDE_PTR:.+]] = llvm.inttoptr %[[DSTSTRIDE_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %arg5, %[[DSTSTRIDE_PTR]] : i64, !llvm.ptr
// CHECK:       %[[HEIGHT_ADDR:.+]] = llvm.mlir.constant(4026531888 : i64) : i64
// CHECK-NEXT:  %[[HEIGHT_PTR:.+]] = llvm.inttoptr %[[HEIGHT_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %arg3, %[[HEIGHT_PTR]] : i64, !llvm.ptr
// CHECK:       llvm.fence seq_cst
// CHECK:       %[[LEN_ADDR:.+]] = llvm.mlir.constant(4026531856 : i64) : i64
// CHECK-NEXT:  %[[LEN_PTR:.+]] = llvm.inttoptr %[[LEN_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %arg2, %[[LEN_PTR]] : i64, !llvm.ptr
// CHECK:       llvm.fence seq_cst

module {
  tt.func public @dma_enqueue_2d_basic(
      %dst: i64, %src: i64, %width: i64,
      %height: i64, %src_stride: i64, %dst_stride: i64) {
    triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
    tt.return
  }
}

// -----

// ============================================================================
// Test: DmaWaitOp lowers to fence + volatile load from STATUS + fence
// ============================================================================

// CHECK-LABEL: @dma_wait_basic
// CHECK:       llvm.fence seq_cst
// CHECK:       %[[STATUS_ADDR:.+]] = llvm.mlir.constant(4026531864 : i64) : i64
// CHECK-NEXT:  %[[STATUS_PTR:.+]] = llvm.inttoptr %[[STATUS_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  %{{.+}} = llvm.load volatile %[[STATUS_PTR]] : !llvm.ptr -> i64
// CHECK:       llvm.fence seq_cst

module {
  tt.func public @dma_wait_basic() {
    triton_cpu.dma_wait
    tt.return
  }
}

// -----

// ============================================================================
// Test: DMA enqueue followed by wait — full sequence
// ============================================================================

// CHECK-LABEL: @dma_enqueue_then_wait
// Enqueue: 5 config stores + fence + LEN store + fence
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.fence seq_cst
// CHECK:       llvm.store volatile
// CHECK:       llvm.fence seq_cst
// Wait: fence + load + fence
// CHECK:       llvm.fence seq_cst
// CHECK:       llvm.load volatile
// CHECK:       llvm.fence seq_cst

module {
  tt.func public @dma_enqueue_then_wait(
      %dst: i64, %src: i64, %width: i64,
      %height: i64, %src_stride: i64, %dst_stride: i64) {
    triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
    triton_cpu.dma_wait
    tt.return
  }
}

// -----

// ============================================================================
// Test: Double-buffered pattern — two enqueues then wait
// ============================================================================

// CHECK-LABEL: @dma_double_buffer
// First enqueue
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.fence seq_cst
// CHECK:       llvm.store volatile
// CHECK:       llvm.fence seq_cst
// Second enqueue
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.fence seq_cst
// CHECK:       llvm.store volatile
// CHECK:       llvm.fence seq_cst
// Wait
// CHECK:       llvm.fence seq_cst
// CHECK:       llvm.load volatile
// CHECK:       llvm.fence seq_cst

module {
  tt.func public @dma_double_buffer(
      %dst0: i64, %src0: i64, %w0: i64, %h0: i64, %ss0: i64, %ds0: i64,
      %dst1: i64, %src1: i64, %w1: i64, %h1: i64, %ss1: i64, %ds1: i64) {
    triton_cpu.dma_enqueue_2d(%dst0, %src0, %w0, %h0, %ss0, %ds0)
    triton_cpu.dma_enqueue_2d(%dst1, %src1, %w1, %h1, %ss1, %ds1)
    triton_cpu.dma_wait
    tt.return
  }
}
