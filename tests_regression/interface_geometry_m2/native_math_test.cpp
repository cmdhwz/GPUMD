#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/measure/interface_geometry.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

int main()
{
  double wrapped = -1.0;
  assert(interface_wrap_coordinate(wrapped, 10.0));
  assert(std::abs(wrapped - 9.0) < 1.0e-12);
  wrapped = 32.0;
  assert(interface_wrap_coordinate(wrapped, 10.0));
  assert(std::abs(wrapped - 2.0) < 1.0e-12);
  assert(interface_bin_index(0.0, 5.0, 3, 12.0) == 0);
  assert(interface_bin_index(5.0, 5.0, 3, 12.0) == 1);
  assert(interface_bin_index(11.999, 5.0, 3, 12.0) == 2);
  assert(interface_bin_index(12.0, 5.0, 3, 12.0) == -1);
  assert(interface_bin_index(-0.1, 5.0, 3, 12.0) == -1);

  InterfaceMaterialAccumulator material;
  assert(material.result() == -1);
  material.add(2);
  material.add(2);
  assert(material.result() == 2);
  material.add(3);
  assert(material.result() == -1);
  assert(interface_profile_atom_bin(1, true, false) == -1);
  assert(interface_profile_atom_bin(1, false, true) == -1);
  assert(interface_profile_atom_bin(1, false, false) == 1);
  assert(interface_bin_crosses_plane(0.0, 5.0, 3.0));
  assert(interface_bin_crosses_plane(5.0, 10.0, 7.0));
  assert(!interface_bin_crosses_plane(0.0, 5.0, 5.0));
  assert(interface_vacf_origin_count(8, 0) == 8);
  assert(interface_vacf_origin_count(8, 3) == 5);
  assert(interface_vacf_origin_count(8, 8) == 0);
  assert(interface_ring_slot(0, 4) == 0);
  assert(interface_valid_lag_count(0, 4) == 1);
  assert(interface_ring_previous_slot(0, 0, 4) == 0);
  assert(interface_ring_slot(4, 4) == 0);
  assert(interface_valid_lag_count(0, 4) == 1);

  const int block_samples = 1024;
  const int correlation_samples = 512;
  const int num_omega = 4097;
  const double dt_ps = 0.01;
  const double target_omega = 2.0 * 3.14159265358979323846 * 5.0;
  std::vector<double> velocity(block_samples);
  for (int n = 0; n < block_samples; ++n) {
    velocity[n] = std::sin(target_omega * n * dt_ps);
  }
  std::vector<double> correlation(correlation_samples, 0.0);
  for (int lag = 0; lag < correlation_samples; ++lag) {
    const int origins = interface_vacf_origin_count(block_samples, lag);
    for (int origin = 0; origin < origins; ++origin) {
      correlation[lag] += velocity[origin] * velocity[origin + lag];
    }
    correlation[lag] /= origins;
  }
  const double raw_c0 = correlation[0];
  assert(raw_c0 > 0.0);
  for (double& value : correlation) value /= raw_c0;
  assert(std::abs(correlation[0] - 1.0) < 1.0e-12);

  const double max_omega = 3.14159265358979323846 / dt_ps;
  double peak_omega = 0.0;
  double peak_value = -1.0e100;
  for (int w = 0; w < num_omega; ++w) {
    const double omega = max_omega * w / (num_omega - 1);
    double spectrum = 0.0;
    for (int lag = 0; lag < correlation_samples; ++lag) {
      spectrum += interface_one_sided_factor(lag) * interface_hann_window(lag, correlation_samples) *
        correlation[lag] * std::cos(omega * lag * dt_ps);
    }
    spectrum *= dt_ps;
    if (spectrum > peak_value) {
      peak_value = spectrum;
      peak_omega = omega;
    }
  }
  assert(std::abs(peak_omega - target_omega) < 1.0);

  double frequency_integral = raw_c0 * max_omega;
  for (int lag = 1; lag < correlation_samples; ++lag) {
    const double tau = lag * dt_ps;
    frequency_integral += 2.0 * interface_hann_window(lag, correlation_samples) *
      raw_c0 * correlation[lag] * std::sin(max_omega * tau) / tau;
  }
  frequency_integral *= dt_ps / 3.14159265358979323846;
  assert(std::abs(frequency_integral - raw_c0) < 1.0e-10 * std::max(1.0, raw_c0));

  // Each block has its own origin count; a reset starts again at block_samples.
  assert(interface_vacf_origin_count(block_samples, 0) == block_samples);
  assert(interface_vacf_origin_count(block_samples, correlation_samples - 1) ==
         block_samples - correlation_samples + 1);
  return 0;
}
