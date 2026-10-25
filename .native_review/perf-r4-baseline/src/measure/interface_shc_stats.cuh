#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

inline bool interface_shc_gpu_fill_size_valid(const std::size_t elements)
{
  return elements <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

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

inline bool interface_shc_cache_flux_span(
  const std::uint64_t block_index,
  const std::uint64_t samples_in_block,
  const std::uint64_t block_samples,
  const std::uint64_t total_samples,
  const std::uint64_t plane_count,
  std::uint64_t& flux_offset,
  std::uint64_t& flux_count)
{
  if (samples_in_block == 0 || block_samples == 0 || samples_in_block > block_samples || plane_count == 0 ||
      block_index > std::numeric_limits<std::uint64_t>::max() / block_samples) return false;
  const std::uint64_t complete_blocks = total_samples / block_samples;
  const std::uint64_t tail_samples = total_samples % block_samples;
  if (block_index > complete_blocks ||
      (block_index < complete_blocks && samples_in_block != block_samples) ||
      (block_index == complete_blocks && samples_in_block > tail_samples)) return false;
  const std::uint64_t sample_offset = block_index * block_samples;
  if (sample_offset > total_samples || samples_in_block > total_samples - sample_offset ||
      sample_offset > std::numeric_limits<std::uint64_t>::max() / plane_count ||
      samples_in_block > std::numeric_limits<std::uint64_t>::max() / plane_count) return false;
  flux_offset = sample_offset * plane_count;
  flux_count = samples_in_block * plane_count;
  return true;
}

inline bool interface_shc_cache_correlation_offset(
  const std::uint64_t block_index,
  const std::uint64_t complete_block_count,
  const std::uint64_t correlation_values_per_sign,
  const std::uint64_t total_correlation_values,
  std::uint64_t& correlation_offset)
{
  if (block_index >= complete_block_count || correlation_values_per_sign == 0 ||
      correlation_values_per_sign > std::numeric_limits<std::uint64_t>::max() / 2) return false;
  const std::uint64_t block_values = 2 * correlation_values_per_sign;
  if (block_index > std::numeric_limits<std::uint64_t>::max() / block_values) return false;
  correlation_offset = block_index * block_values;
  return correlation_offset <= total_correlation_values &&
         block_values <= total_correlation_values - correlation_offset;
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

inline double interface_shc_block_mean_flux(
  const double* sample_major_flux,
  const std::size_t sample_count,
  const std::size_t plane_count,
  const std::size_t plane)
{
  if (sample_count == 0 || plane_count == 0 || plane >= plane_count) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    sum += sample_major_flux[sample * plane_count + plane];
  }
  return sum / static_cast<double>(sample_count);
}

inline double interface_shc_standard_error(const std::uint64_t count, const double m2)
{
  return count > 1
    ? std::sqrt(m2 / (static_cast<double>(count) * (count - 1.0)))
    : std::numeric_limits<double>::quiet_NaN();
}
