//===- SPMSpaceManager.cpp - Compile-time SPM layout helper ---------------===//

#include "cpu/include/TritonCPUTransforms/SPMSpaceManager.h"

#include <limits>

namespace mlir {
namespace triton {
namespace cpu {

SPMSpaceManager::SPMSpaceManager(int64_t base, int64_t size)
    : base(base), size(size) {}

std::optional<int64_t> SPMSpaceManager::alignUp(int64_t value,
                                                int64_t alignment) {
  if (value < 0 || alignment <= 0)
    return std::nullopt;

  int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;

  int64_t delta = alignment - remainder;
  if (value > std::numeric_limits<int64_t>::max() - delta)
    return std::nullopt;
  return value + delta;
}

std::optional<SPMSpaceManager::Allocation> SPMSpaceManager::alloc(
    int64_t allocSize, int64_t alignment, Lifetime lifetime) {
  if (allocSize < 0 || size < 0)
    return std::nullopt;

  std::optional<int64_t> alignedTop = alignUp(top, alignment);
  if (!alignedTop || *alignedTop > size || allocSize > size - *alignedTop)
    return std::nullopt;

  if (base > std::numeric_limits<int64_t>::max() - *alignedTop)
    return std::nullopt;

  Allocation allocation;
  allocation.address = base + *alignedTop;
  allocation.offset = *alignedTop;
  allocation.size = allocSize;
  allocation.alignment = alignment;
  allocation.lifetime = lifetime;
  top = *alignedTop + allocSize;
  return allocation;
}

void SPMSpaceManager::free(const Allocation &allocation) {
  (void)allocation;
}

void SPMSpaceManager::reset() { top = 0; }

} // namespace cpu
} // namespace triton
} // namespace mlir
