#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/force/nep_plane_flux_view.cuh"
#include "../../src/measure/interface_shc_stats.cuh"
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

int main()
{
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
  std::vector<int> flux_steps(buffered_samples, -1);
  std::vector<int> spectrum_blocks;
  for (std::uint64_t step = 0; step < buffered_samples; ++step) {
    std::uint64_t cache_index = 0;
    assert(interface_shc_cache_sample_index(
      step / block_samples, step % block_samples, block_samples,
      buffered_samples, cache_index));
    assert(cache_index == step);
    flux_steps[cache_index] = static_cast<int>(step + 1);
    if ((step + 1) % block_samples == 0) spectrum_blocks.push_back(static_cast<int>(step / block_samples));
  }
  assert((flux_steps == std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
  assert((spectrum_blocks == std::vector<int>{0, 1}));
  std::uint64_t invalid_index = 0;
  assert(!interface_shc_cache_sample_index(2, 1, block_samples, buffered_samples - 1, invalid_index));

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
