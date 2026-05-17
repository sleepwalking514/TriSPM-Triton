"""
Tests for TritonCPU DMA dialect ops and their LLVM lowering.

These tests verify that:
1. DMA ops (dma_enqueue_2d, dma_wait, dma_wait_count) are correctly defined in
   the dialect
2. The DmaOpsToLLVM pass lowers them to the expected volatile MMIO stores/loads
3. The ops integrate with the AOT compilation pipeline

Since DMA ops will be inserted by a future SPM transformation pass (Phase 3),
these tests manually construct MLIR modules with DMA ops and verify their
lowering through the pass pipeline.
"""
import os
import subprocess
import tempfile
import textwrap
import pytest


def get_triton_opt():
    """Find the triton-opt binary in the build directory."""
    # Try common build paths relative to the compiler source tree
    compiler_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    candidates = [
        os.path.join(compiler_root, "build", "bin", "triton-opt"),
        # cmake.linux-x86_64-cpython-3.12 style
    ]
    import glob
    candidates += glob.glob(
        os.path.join(compiler_root, "build", "cmake.*", "bin", "triton-opt"))
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    # Fall back to PATH
    return "triton-opt"


TRITON_OPT = get_triton_opt()

# DMA MMIO register addresses (must match DmaOpsToLLVM.cpp constants)
DMA_MMIO_BASE   = 0xF0000000
DMA_REG_SRC     = 0x00
DMA_REG_DST     = 0x08
DMA_REG_LEN     = 0x10
DMA_REG_STATUS  = 0x18
DMA_REG_SRC_STRIDE = 0x20
DMA_REG_DST_STRIDE = 0x28
DMA_REG_HEIGHT  = 0x30
DMA_REG_STRIDES_PACKED = 0x38


def run_triton_opt(mlir_source: str, passes: list[str]) -> str:
    """Run triton-opt on the given MLIR source with specified passes.
    Returns the output MLIR text.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_source)
        f.flush()
        try:
            cmd = [TRITON_OPT, f.name] + passes
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                pytest.fail(
                    f"triton-opt failed (rc={result.returncode}):\n"
                    f"STDERR: {result.stderr}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"INPUT:\n{mlir_source}")
            return result.stdout
        finally:
            os.unlink(f.name)


# ===----------------------------------------------------------------------=== #
# Test: DMA ops are valid in the TritonCPU dialect (parse + verify)
# ===----------------------------------------------------------------------=== #

class TestDmaDialectOps:
    """Test that DMA ops can be parsed and verified."""

    def test_dma_enqueue_2d_parses(self):
        """dma_enqueue_2d should parse with 6 i64 arguments."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_enqueue(
                  %dst: i64, %src: i64, %width: i64,
                  %height: i64, %src_stride: i64, %dst_stride: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
                tt.return
              }
            }
        """)
        output = run_triton_opt(mlir, [])
        assert "triton_cpu.dma_enqueue_2d" in output

    def test_dma_wait_parses(self):
        """dma_wait should parse with no arguments."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_wait() {
                triton_cpu.dma_wait
                tt.return
              }
            }
        """)
        output = run_triton_opt(mlir, [])
        assert "triton_cpu.dma_wait" in output

    def test_dma_wait_count_parses(self):
        """dma_wait_count should parse with one i64 watermark argument."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_wait_count() {
                %max = arith.constant 2 : i64
                triton_cpu.dma_wait_count(%max)
                tt.return
              }
            }
        """)
        output = run_triton_opt(mlir, [])
        assert "triton_cpu.dma_wait_count" in output

    def test_dma_enqueue_then_wait_parses(self):
        """Full enqueue + wait sequence should parse."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_seq(
                  %dst: i64, %src: i64, %w: i64,
                  %h: i64, %ss: i64, %ds: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %w, %h, %ss, %ds)
                triton_cpu.dma_wait
                tt.return
              }
            }
        """)
        output = run_triton_opt(mlir, [])
        assert "triton_cpu.dma_enqueue_2d" in output
        assert "triton_cpu.dma_wait" in output


# ===----------------------------------------------------------------------=== #
# Test: DmaOpsToLLVM lowering pass
# ===----------------------------------------------------------------------=== #

class TestDmaOpsToLLVM:
    """Test that DMA ops are correctly lowered to LLVM MMIO operations."""

    LOWERING_PASS = "-triton-cpu-dma-ops-to-llvm"

    def _lower(self, mlir: str) -> str:
        return run_triton_opt(mlir, [self.LOWERING_PASS])

    def test_enqueue_produces_volatile_stores(self):
        """dma_enqueue_2d should lower to packed MMIO descriptor stores."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_enqueue(
                  %dst: i64, %src: i64, %width: i64,
                  %height: i64, %src_stride: i64, %dst_stride: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        # The op should be completely consumed
        assert "triton_cpu.dma_enqueue_2d" not in output
        # SRC + DST + packed strides + packed LEN/HEIGHT trigger.
        assert output.count("llvm.store volatile") == 4
        # Should have exactly 2 `fence iorw, iorw` (pre-trigger + post-trigger),
        # emitted as inline asm with has_side_effects so the optimizer cannot
        # weaken or eliminate them.
        assert output.count('"fence iorw, iorw"') == 2
        assert output.count("llvm.inline_asm has_side_effects") == 2

    def test_enqueue_mmio_addresses(self):
        """Check that the correct MMIO addresses are used."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_addrs(
                  %dst: i64, %src: i64, %width: i64,
                  %height: i64, %src_stride: i64, %dst_stride: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        # Verify key MMIO addresses appear (as decimal i64 constants)
        assert str(DMA_MMIO_BASE + DMA_REG_SRC) in output         # SRC = 0xF0000000
        assert str(DMA_MMIO_BASE + DMA_REG_DST) in output         # DST = 0xF0000008
        assert str(DMA_MMIO_BASE + DMA_REG_LEN) in output         # LEN = 0xF0000010
        assert str(DMA_MMIO_BASE + DMA_REG_STRIDES_PACKED) in output
        assert str(DMA_MMIO_BASE + DMA_REG_SRC_STRIDE) not in output
        assert str(DMA_MMIO_BASE + DMA_REG_DST_STRIDE) not in output
        assert str(DMA_MMIO_BASE + DMA_REG_HEIGHT) not in output

    def test_custom_mmio_base_option(self):
        """The lowering pass should honor a non-default DMA MMIO base."""
        custom_base = 0xE0000000
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_custom_base(
                  %dst: i64, %src: i64, %width: i64,
                  %height: i64, %src_stride: i64, %dst_stride: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
                triton_cpu.dma_wait
                tt.return
              }
            }
        """)
        output = run_triton_opt(
            mlir, ["-triton-cpu-dma-ops-to-llvm=dma-mmio-base=0xE0000000"])
        assert str(custom_base + DMA_REG_SRC) in output
        assert str(custom_base + DMA_REG_DST) in output
        assert str(custom_base + DMA_REG_LEN) in output
        assert str(custom_base + DMA_REG_STATUS) in output

    def test_wait_produces_volatile_load(self):
        """dma_wait should lower to a volatile load from STATUS register."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_wait() {
                triton_cpu.dma_wait
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        # The op should be completely consumed
        assert "triton_cpu.dma_wait" not in output
        # Should have a volatile load (STATUS register)
        assert "llvm.load volatile" in output
        # STATUS address = 0xF0000018
        assert str(DMA_MMIO_BASE + DMA_REG_STATUS) in output
        # Should have 2 `fence iorw, iorw` (pre-load + post-load), emitted as
        # inline asm with has_side_effects to prevent weakening/elimination.
        assert output.count('"fence iorw, iorw"') == 2
        assert output.count("llvm.inline_asm has_side_effects") == 2

    def test_wait_count_produces_watermark_poll(self):
        """dma_wait_count should poll while pending descriptors exceed a watermark."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_wait_count() {
                %max = arith.constant 2 : i64
                triton_cpu.dma_wait_count(%max)
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        assert "triton_cpu.dma_wait_count" not in output
        assert "llvm.load volatile" in output
        assert str(DMA_MMIO_BASE + DMA_REG_STATUS) in output
        assert 'llvm.icmp "ugt"' in output
        assert output.count('"fence iorw, iorw"') == 2

    def test_enqueue_fence_ordering(self):
        """The trigger (LEN write) must be after the fence, not before config stores."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_ordering(
                  %dst: i64, %src: i64, %width: i64,
                  %height: i64, %src_stride: i64, %dst_stride: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %width, %height, %src_stride, %dst_stride)
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        lines = output.split('\n')

        # Find positions of key operations.  Fences are emitted as inline asm
        # `fence iorw, iorw` with has_side_effects.
        fence_positions = [i for i, l in enumerate(lines) if '"fence iorw, iorw"' in l]
        store_positions = [i for i, l in enumerate(lines) if "llvm.store volatile" in l]

        # Must have at least 2 fences
        assert len(fence_positions) >= 2, f"Expected >=2 fences, got {len(fence_positions)}"
        # Must have four stores: SRC, DST, packed strides, packed LEN/HEIGHT.
        assert len(store_positions) == 4, f"Expected 4 stores, got {len(store_positions)}"

        # First 3 stores (config) should be before the first fence.
        first_fence = fence_positions[0]
        for pos in store_positions[:3]:
            assert pos < first_fence, \
                f"Config store at line {pos} should be before first fence at line {first_fence}"

        # 4th store (packed LEN/HEIGHT trigger) should be after the first fence.
        assert store_positions[3] > first_fence, \
            f"LEN trigger store at line {store_positions[3]} should be after first fence at line {first_fence}"

        # 4th store should be before the second fence.
        second_fence = fence_positions[1]
        assert store_positions[3] < second_fence, \
            f"LEN trigger at line {store_positions[3]} should be before second fence at line {second_fence}"

    def test_full_sequence_enqueue_wait(self):
        """Enqueue + wait should produce the complete MMIO sequence."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_full(
                  %dst: i64, %src: i64, %w: i64,
                  %h: i64, %ss: i64, %ds: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %w, %h, %ss, %ds)
                triton_cpu.dma_wait
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        # Both ops should be consumed
        assert "triton_cpu.dma_enqueue_2d" not in output
        assert "triton_cpu.dma_wait" not in output
        # 4 stores (enqueue) + 1 load (wait)
        assert output.count("llvm.store volatile") == 4
        assert output.count("llvm.load volatile") == 1
        # 4 `fence iorw, iorw` inline-asm fences: 2 from enqueue + 2 from wait
        assert output.count('"fence iorw, iorw"') == 4
        assert output.count("llvm.inline_asm has_side_effects") == 4

    def test_double_buffer_two_enqueues(self):
        """Two enqueues + one wait (double-buffer pattern)."""
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_double_buf(
                  %d0: i64, %s0: i64, %w0: i64, %h0: i64, %ss0: i64, %ds0: i64,
                  %d1: i64, %s1: i64, %w1: i64, %h1: i64, %ss1: i64, %ds1: i64) {
                triton_cpu.dma_enqueue_2d(%d0, %s0, %w0, %h0, %ss0, %ds0)
                triton_cpu.dma_enqueue_2d(%d1, %s1, %w1, %h1, %ss1, %ds1)
                triton_cpu.dma_wait
                tt.return
              }
            }
        """)
        output = self._lower(mlir)
        # 8 stores (2 × 4) + 1 load (wait)
        assert output.count("llvm.store volatile") == 8
        assert output.count("llvm.load volatile") == 1
        # 6 `fence iorw, iorw` inline-asm fences: 2×2 from enqueues + 2 from wait
        assert output.count('"fence iorw, iorw"') == 6
        assert output.count("llvm.inline_asm has_side_effects") == 6


# ===----------------------------------------------------------------------=== #
# Test: DMA ops use inttoptr for MMIO addresses
# ===----------------------------------------------------------------------=== #

class TestDmaIntToPtr:
    """Verify that MMIO addresses are formed via inttoptr (not GEP)."""

    def test_uses_inttoptr(self):
        mlir = textwrap.dedent("""\
            module {
              tt.func public @test_inttoptr(
                  %dst: i64, %src: i64, %w: i64,
                  %h: i64, %ss: i64, %ds: i64) {
                triton_cpu.dma_enqueue_2d(%dst, %src, %w, %h, %ss, %ds)
                tt.return
              }
            }
        """)
        output = run_triton_opt(mlir, ["-triton-cpu-dma-ops-to-llvm"])
        # All packed MMIO descriptor addresses should use inttoptr.
        assert output.count("llvm.inttoptr") >= 4
        # Should NOT use getelementptr for MMIO
        assert "llvm.getelementptr" not in output
