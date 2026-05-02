// RUN: triton-opt %s -split-input-file -triton-cpu-dma-ops-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-cpu-dma-ops-to-llvm="dma-mmio-base=0xE0000000" | FileCheck %s --check-prefix=BASE
// RUN: not triton-opt %s -split-input-file -triton-cpu-dma-ops-to-llvm="use-xspm-insn=1" 2>&1 | FileCheck %s --check-prefix=XSPMERR

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
// CHECK:       %[[STRIDES_ADDR:.+]] = llvm.mlir.constant(4026531896 : i64) : i64
// CHECK-NEXT:  %[[STRIDES_PTR:.+]] = llvm.inttoptr %[[STRIDES_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %{{.+}}, %[[STRIDES_PTR]] : i64, !llvm.ptr
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       %[[LEN_ADDR:.+]] = llvm.mlir.constant(4026531856 : i64) : i64
// CHECK-NEXT:  %[[LEN_PTR:.+]] = llvm.inttoptr %[[LEN_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  llvm.store volatile %{{.+}}, %[[LEN_PTR]] : i64, !llvm.ptr
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
//
// BASE-LABEL: @dma_enqueue_2d_basic
// BASE:       llvm.mlir.constant(3758096384 : i64) : i64
// BASE:       llvm.mlir.constant(3758096392 : i64) : i64
// BASE:       llvm.mlir.constant(3758096440 : i64) : i64
// BASE:       llvm.mlir.constant(3758096400 : i64) : i64

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
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       %[[STATUS_ADDR:.+]] = llvm.mlir.constant(4026531864 : i64) : i64
// CHECK-NEXT:  %[[STATUS_PTR:.+]] = llvm.inttoptr %[[STATUS_ADDR]] : i64 to !llvm.ptr
// CHECK-NEXT:  %{{.+}} = llvm.load volatile %[[STATUS_PTR]] : !llvm.ptr -> i64
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
//
// BASE-LABEL: @dma_wait_basic
// BASE:       llvm.mlir.constant(3758096408 : i64) : i64
//
// XSPMERR: use-xspm-insn path is not implemented yet

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
// Enqueue: SRC/DST/packed-strides stores + fence + packed LEN/HEIGHT store + fence
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       llvm.store volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// Wait: fence + load + fence
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       llvm.load volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"

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
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       llvm.store volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// Second enqueue
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.store volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       llvm.store volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// Wait
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"
// CHECK:       llvm.load volatile
// CHECK:       llvm.inline_asm has_side_effects {{.*}}"fence iorw, iorw", "~{memory}"

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
