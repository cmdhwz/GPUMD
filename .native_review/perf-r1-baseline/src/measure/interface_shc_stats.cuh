#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

inline std::uint64_t interface_shc_complete_block_count(
  const std::uint64_t sample_count, const std::uint64_t block_samples)
{
  return block_samples == 0 ? 0 : sample_count / block_samples;
}

inline bool interface_shc_cache_sample_index(
  const std::uint64_t block_index,
  const std::uint64_t sample_in_block,
  const std::uint64_t block_samples,
  const std::uint64_t total_samples,
  std::uint64_t& sample_index)
{
  if (block_samples == 0 || sample_in_block >= block_samples ||
      block_index > std::numeric_limits<std::uint64_t>::max() / block_samples) return false;
  const std::uint64_t block_offset = block_index * block_samples;
  if (sample_in_block > std::numeric_limits<std::uint64_t>::max() - block_offset) return false;
  sample_index = block_offset + sample_in_block;
  return sample_index < total_samples;
}

inline bool interface_shc_accumulate_range(
  const double block_start_ps,
  const double block_end_ps,
  const std::uint64_t block_samples,
  bool& have_range,
  double& accumulation_start_ps,
  double& accumulation_end_ps,
  std::uint64_t& accumulation_samples)
{
  if (!std::isfinite(block_start_ps) || !std::isfinite(block_end_ps) ||
      block_end_ps <= block_start_ps || block_samples == 0 ||
      block_samples > std::numeric_limits<std::uint64_t>::max() - accumulation_samples ||
      (have_range && block_start_ps <= accumulation_end_ps)) {
    return false;
  }
  if (!have_range) accumulation_start_ps = block_start_ps;
  accumulation_end_ps = block_end_ps;
  accumulation_samples += block_samples;
  have_range = true;
  return true;
}

inline void interface_shc_update_mean_m2(
  const double value, const double count, double& mean, double& m2)
{
  const double delta = value - mean;
  mean += delta / count;
  m2 += delta * (value - mean);
}

inline double interface_shc_standard_error(const std::uint64_t count, const double m2)
{
  return count > 1
    ? std::sqrt(m2 / (static_cast<double>(count) * (count - 1.0)))
    : std::numeric_limits<double>::quiet_NaN();
}
