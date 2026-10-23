#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace rpmd_ja_device
{
template <typename T>
class DeviceBuffer
{
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), capacity_(other.capacity_)
  {
    other.pointer_ = nullptr;
    other.capacity_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
  {
    if (this != &other) {
      release();
      pointer_ = other.pointer_;
      capacity_ = other.capacity_;
      other.pointer_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  void reserve(const std::size_t count, const char* label)
  {
    if (count <= capacity_) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error(std::string("rpmd_ja workspace size overflow at ") + label);
    }
    T* replacement = nullptr;
    const cudaError_t allocated = cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T));
    if (allocated != cudaSuccess) {
      throw std::runtime_error(std::string("rpmd_ja workspace allocation failed at ") + label + ": " +
        cudaGetErrorString(allocated));
    }
    if (pointer_ != nullptr) {
      const cudaError_t freed = cudaFree(pointer_);
      if (freed != cudaSuccess) {
        cudaFree(replacement);
        throw std::runtime_error(std::string("rpmd_ja workspace release failed at ") + label + ": " +
          cudaGetErrorString(freed));
      }
    }
    pointer_ = replacement;
    capacity_ = count;
  }

  void release() noexcept
  {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
      pointer_ = nullptr;
      capacity_ = 0;
    }
  }

  T* data() { return pointer_; }
  const T* data() const { return pointer_; }
  std::size_t capacity() const { return capacity_; }

private:
  T* pointer_ = nullptr;
  std::size_t capacity_ = 0;
};

struct CovarianceFitWorkspace
{
  DeviceBuffer<double> curvature_ambient;
  DeviceBuffer<double> curvature_action;
  DeviceBuffer<double> curvature_compact_rhs;
  DeviceBuffer<double> curvature_rhs_reference;
  DeviceBuffer<double> curvature_z;
  DeviceBuffer<double> curvature_dots;
  DeviceBuffer<double> curvature_gram;
  DeviceBuffer<double> solver_matrix;
  DeviceBuffer<double> solver_workspace;
  DeviceBuffer<double> solver_values;
  DeviceBuffer<int> solver_info;
  std::uint64_t curvature_allocation_count = 0;
  std::uint64_t curvature_peak_bytes = 0;

  std::uint64_t curvature_capacity_bytes() const
  {
    const std::size_t capacities[] = {curvature_ambient.capacity(), curvature_action.capacity(),
      curvature_compact_rhs.capacity(), curvature_rhs_reference.capacity(), curvature_z.capacity(),
      curvature_dots.capacity(), curvature_gram.capacity()};
    std::uint64_t bytes = 0;
    for (const std::size_t count : capacities) {
      if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(double) ||
          bytes > std::numeric_limits<std::uint64_t>::max() - count * sizeof(double))
        throw std::overflow_error("rpmd_ja covariance curvature capacity overflows");
      bytes += count * sizeof(double);
    }
    return bytes;
  }

  std::uint64_t curvature_growth_transient_bytes(const std::size_t ambient, const std::size_t compact,
    const std::size_t z, const std::size_t dots, const std::size_t gram) const
  {
    const std::size_t old_counts[] = {curvature_ambient.capacity(), curvature_action.capacity(),
      curvature_compact_rhs.capacity(), curvature_rhs_reference.capacity(), curvature_z.capacity(),
      curvature_dots.capacity(), curvature_gram.capacity()};
    const std::size_t new_counts[] = {ambient, ambient, compact, compact, z, dots, gram};
    std::uint64_t growth = 0, largest_replaced = 0;
    for (std::size_t index = 0; index < 7; ++index) {
      if (new_counts[index] <= old_counts[index]) continue;
      const std::uint64_t old_bytes = static_cast<std::uint64_t>(old_counts[index]) * sizeof(double);
      const std::uint64_t new_bytes = static_cast<std::uint64_t>(new_counts[index]) * sizeof(double);
      if (new_bytes > std::numeric_limits<std::uint64_t>::max() - growth)
        throw std::overflow_error("rpmd_ja covariance curvature growth estimate overflows");
      growth += new_bytes - old_bytes;
      largest_replaced = std::max(largest_replaced, old_bytes);
    }
    if (largest_replaced > std::numeric_limits<std::uint64_t>::max() - growth)
      throw std::overflow_error("rpmd_ja covariance curvature growth estimate overflows");
    return growth + largest_replaced;
  }

  void reserve_curvature(const std::size_t ambient, const std::size_t compact, const std::size_t z,
    const std::size_t dots, const std::size_t gram)
  {
    auto reserve = [this](auto& buffer, const std::size_t count, const char* label) {
      const std::size_t old_capacity = buffer.capacity();
      buffer.reserve(count, label);
      if (buffer.capacity() > old_capacity) ++curvature_allocation_count;
    };
    reserve(curvature_ambient, ambient, "curvature ambient");
    reserve(curvature_action, ambient, "curvature action");
    reserve(curvature_compact_rhs, compact, "curvature compact rhs");
    reserve(curvature_rhs_reference, compact, "curvature rhs reference");
    reserve(curvature_z, z, "curvature z");
    reserve(curvature_dots, dots, "curvature dots");
    reserve(curvature_gram, gram, "curvature gram");
    curvature_peak_bytes = std::max(curvature_peak_bytes, curvature_capacity_bytes());
  }

  void release_curvature() noexcept
  {
    curvature_ambient.release();
    curvature_action.release();
    curvature_compact_rhs.release();
    curvature_rhs_reference.release();
    curvature_z.release();
    curvature_dots.release();
    curvature_gram.release();
  }
};
} // namespace rpmd_ja_device
