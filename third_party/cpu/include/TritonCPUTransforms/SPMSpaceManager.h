//===- SPMSpaceManager.h - Compile-time SPM layout helper -------*- C++ -*-===//
//
// This helper assigns non-overlapping SPM address ranges for compiler-generated
// scratchpad users.  It does not allocate runtime memory; it only computes the
// offsets that lowering will materialize as SPM addresses.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONCPU_TRANSFORMS_SPMSPACEMANAGER_H
#define TRITONCPU_TRANSFORMS_SPMSPACEMANAGER_H

#include <cstdint>
#include <optional>

namespace mlir {
namespace triton {
namespace cpu {

class SPMSpaceManager {
public:
  enum class Lifetime {
    Function,
    Loop,
    Temporary,
  };

  struct Allocation {
    int64_t address;
    int64_t offset;
    int64_t size;
    int64_t alignment;
    Lifetime lifetime;
  };

  SPMSpaceManager(int64_t base, int64_t size);

  std::optional<Allocation> alloc(
      int64_t size, int64_t alignment = 1,
      Lifetime lifetime = Lifetime::Temporary);

  // Reserved for future lifetime-aware reuse.  M1/M2 use high-water allocation.
  void free(const Allocation &allocation);

  void reset();

  int64_t getBase() const { return base; }
  int64_t getSize() const { return size; }
  int64_t getTopOffset() const { return top; }
  int64_t getRemainingBytes() const { return size - top; }

private:
  static std::optional<int64_t> alignUp(int64_t value, int64_t alignment);

  int64_t base;
  int64_t size;
  int64_t top = 0;
};

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITONCPU_TRANSFORMS_SPMSPACEMANAGER_H
