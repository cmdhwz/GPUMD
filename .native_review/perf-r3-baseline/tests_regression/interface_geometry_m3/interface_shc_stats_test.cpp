#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/force/nep_plane_flux_view.cuh"
#include "../../src/measure/interface_shc_stats.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

int main()
{
  const size_t gpu_fill_limit = static_cast<size_t>(std::numeric_limits<int>::max());
  assert(interface_shc_gpu_fill_size_valid(gpu_fill_limit));
  if (gpu_fill_limit < std::numeric_limits<size_t>::max()) {
    assert(!interface_shc_gpu_fill_size_valid(gpu_fill_limit + 1));
  }

  constexpr int series_count = 4; // C0, analytic integral, exported-grid integral, and spectrum.
  double means[series_count] = {};
  double m2[series_count] = {};
  for (int i = 0; i < series_count; ++i) {
    interface_shc_update_mean_m2(100.0, 1.0, means[i], m2[i]);
    assert(std::isnan(interface_shc_standard_error(1, m2[i])));
    interface_shc_update_mean_m2(10.0, 2.0, means[i], m2[i]);
    assert(means[i] == 55.0);
    assert(interface_shc_standard_error(2, m2[i]) == 45.0);
  }

  bool have_range = false;
  double start_ps = 0.0;
  double end_ps = 0.0;
  std::uint64_t samples = 0;
  assert(interface_shc_accumulate_range(0.5, 3.5, 4, have_range, start_ps, end_ps, samples));
  assert(start_ps == 0.5 && end_ps == 3.5 && samples == 4);
  assert(interface_shc_accumulate_range(4.5, 7.5, 4, have_range, start_ps, end_ps, samples));
  assert(start_ps == 0.5 && end_ps == 7.5 && samples == 8);
  assert(!interface_shc_accumulate_range(7.0, 8.0, 4, have_range, start_ps, end_ps, samples));
  assert(!interface_shc_accumulate_range(
    8.5, 9.5, std::numeric_limits<std::uint64_t>::max(),
    have_range, start_ps, end_ps, samples));

  constexpr std::uint64_t buffered_samples = 7;
  constexpr std::uint64_t block_samples = 3;
  const std::uint64_t complete_blocks =
    interface_shc_complete_block_count(buffered_samples, block_samples);
  assert(complete_blocks == 2);
  std::uint64_t sample_index = 0;
  assert(interface_shc_cache_sample_index(1, 2, block_samples, buffered_samples, sample_index));
  assert(sample_index == 5);
  assert(!interface_shc_cache_sample_index(2, 1, block_samples, buffered_samples - 1, sample_index));

  constexpr std::uint64_t planes = 2;
  std::vector<double> flux_cache(buffered_samples * planes, -1.0);
  std::vector<double> flux_block(block_samples * planes);
  std::vector<double> correlation_cache(complete_blocks * 2, -1.0);
  for (std::uint64_t block = 0; block <= complete_blocks; ++block) {
    const std::uint64_t sample_offset = block * block_samples;
    const std::uint64_t samples_in_block = std::min(block_samples, buffered_samples - sample_offset);
    for (std::size_t value = 0; value < flux_block.size(); ++value) {
      flux_block[value] = 100.0 * block + value;
    }
    std::uint64_t flux_offset = 0;
    std::uint64_t flux_count = 0;
    assert(interface_shc_cache_flux_span(
      block, samples_in_block, block_samples, buffered_samples, planes, flux_offset, flux_count));
    std::copy(flux_block.begin(), flux_block.begin() + flux_count, flux_cache.begin() + flux_offset);
    if (samples_in_block == block_samples) {
      std::uint64_t correlation_offset = 0;
      assert(interface_shc_cache_correlation_offset(
        block, complete_blocks, 1, correlation_cache.size(), correlation_offset));
      correlation_cache[correlation_offset] = static_cast<double>(block + 1);
      correlation_cache[correlation_offset + 1] = -static_cast<double>(block + 1);
    }
  }
  assert((flux_cache == std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                             100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                                             200.0, 201.0}));
  assert((correlation_cache == std::vector<double>{1.0, -1.0, 2.0, -2.0}));
  std::uint64_t bad_offset = 0;
  std::uint64_t bad_count = 0;
  assert(!interface_shc_cache_flux_span(0, 2, block_samples, buffered_samples, planes, bad_offset, bad_count));
  assert(!interface_shc_cache_flux_span(3, 1, block_samples, buffered_samples, planes, bad_offset, bad_count));
  assert(!interface_shc_cache_correlation_offset(
    complete_blocks, complete_blocks, 1, correlation_cache.size(), bad_offset));

  int error_info[2] = {0, -1};
  nep_plane_flux_record_error(error_info, error_info + 1, NEP_PLANE_FLUX_ERROR_MISSING_SLOT, 12);
  nep_plane_flux_record_error(error_info, error_info + 1, NEP_PLANE_FLUX_ERROR_NONFINITE, 18);
  assert(error_info[0] == (NEP_PLANE_FLUX_ERROR_MISSING_SLOT | NEP_PLANE_FLUX_ERROR_NONFINITE));
  assert(error_info[1] == 12);
  nep_plane_flux_reset_error(error_info, error_info + 1);
  assert(error_info[0] == 0 && error_info[1] == -1);
  nep_plane_flux_record_error(error_info, error_info + 1, NEP_PLANE_FLUX_ERROR_DISPLACEMENT, 21);
  assert(error_info[0] == NEP_PLANE_FLUX_ERROR_DISPLACEMENT && error_info[1] == 21);
  return 0;
}
