#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/measure/interface_profile_math.cuh"
#include <cassert>
#include <cmath>
#include <limits>

int main()
{
  double corrected = -1.0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  assert(!interface_profile_corrected_kinetic(2, true, nan, 2.0, 0.0, 0.0, 0.0, corrected));
  assert(!interface_profile_corrected_kinetic(2, true, 1.0, 2.0, inf, 0.0, 0.0, corrected));
  assert(!interface_profile_corrected_kinetic(
    2, true, 1.0, 2.0, std::numeric_limits<double>::max() / 2.0, 0.0, 0.0, corrected));
  assert(!interface_profile_corrected_kinetic(2, true, 1.0, 2.0, 3.0, 0.0, 0.0, corrected));

  const double epsilon = std::numeric_limits<double>::epsilon();
  const double rounded_com_px = std::sqrt(4.0 * (1.0 + 16.0 * epsilon));
  assert(interface_profile_corrected_kinetic(
    2, true, 1.0, 2.0, rounded_com_px, 0.0, 0.0, corrected));
  assert(corrected == 0.0);

  assert(interface_profile_corrected_kinetic(0, true, 0.0, 0.0, 0.0, 0.0, 0.0, corrected));
  assert(corrected == 0.0 && interface_profile_dof(0, true) == 0.0);

  assert(interface_profile_corrected_kinetic(1, true, 0.5, 1.0, 1.0, 0.0, 0.0, corrected));
  assert(corrected == 0.0 && interface_profile_dof(1, true) == 0.0);
  assert(std::isnan(interface_profile_temperature(corrected, 0.0, 8.617343e-5)));

  assert(interface_profile_corrected_kinetic(2, true, 0.0, 2.0, 0.0, 0.0, 0.0, corrected));
  assert(corrected == 0.0 && interface_profile_dof(2, true) == 3.0);
  assert(interface_profile_temperature(corrected, 3.0, 8.617343e-5) == 0.0);

  assert(interface_profile_corrected_kinetic(2, false, 1.0, 0.0, 0.0, 0.0, 0.0, corrected));
  const double temperature = interface_profile_temperature(corrected, 6.0, 8.617343e-5);
  assert(std::isfinite(temperature) && temperature > 0.0);

  constexpr double k_b = 8.617343e-5;
  InterfaceProfileBlockAccum paired{};
  assert(interface_profile_accumulate_sample(
    paired, 2, true, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(interface_profile_accumulate_sample(
    paired, 2, true, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(paired.atom_count_sum == 4 && paired.temperature_samples == 2);
  assert(paired.kinetic_sum == 2.0 && paired.potential_sum == 5.0 && paired.dof_sum == 6.0);
  assert(std::abs(paired.temperature_sum - 4.0 / (3.0 * k_b)) < 1.0e-10 / k_b);
  InterfaceProfileBlockAccum cpu_reference{};
  const double raw_samples[2] = {1.0, 2.0};
  const double momentum_samples[2] = {2.0, 0.0};
  const double potential_samples[2] = {2.0, 3.0};
  for (int sample = 0; sample < 2; ++sample) {
    double sample_ke = 0.0;
    assert(interface_profile_corrected_kinetic(
      2, true, raw_samples[sample], 2.0, momentum_samples[sample], 0.0, 0.0, sample_ke));
    const double sample_dof = interface_profile_dof(2, true);
    const double sample_temperature = interface_profile_temperature(sample_ke, sample_dof, k_b);
    cpu_reference.atom_count_sum += 2;
    cpu_reference.kinetic_sum += sample_ke;
    cpu_reference.potential_sum += potential_samples[sample];
    cpu_reference.dof_sum += sample_dof;
    cpu_reference.temperature_sum += sample_temperature;
    ++cpu_reference.temperature_samples;
  }
  assert(cpu_reference.atom_count_sum == paired.atom_count_sum);
  assert(cpu_reference.temperature_samples == paired.temperature_samples);
  assert(cpu_reference.kinetic_sum == paired.kinetic_sum);
  assert(cpu_reference.potential_sum == paired.potential_sum);
  assert(cpu_reference.dof_sum == paired.dof_sum);
  assert(cpu_reference.temperature_sum == paired.temperature_sum);
  const double average_raw_kinetic = (1.0 + 2.0) / 2.0;
  const double average_momentum = (2.0 + 0.0) / 2.0;
  double incorrectly_corrected = 0.0;
  assert(interface_profile_corrected_kinetic(
    2, true, average_raw_kinetic, 2.0, average_momentum, 0.0, 0.0,
    incorrectly_corrected));
  assert(incorrectly_corrected == 1.25);
  assert(paired.kinetic_sum / 2.0 != incorrectly_corrected);

  InterfaceProfileBlockAccum variable_dof{};
  assert(interface_profile_accumulate_sample(
    variable_dof, 1, true, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(interface_profile_accumulate_sample(
    variable_dof, 3, true, 3.0, 2.0, 3.0, 3.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(variable_dof.atom_count_sum == 4 && variable_dof.dof_sum == 6.0);
  assert(variable_dof.temperature_samples == 1);
  assert(std::abs(variable_dof.temperature_sum - 0.5 / k_b) < 1.0e-10 / k_b);

  InterfaceProfileBlockAccum empty_bin{};
  assert(interface_profile_accumulate_sample(
    empty_bin, 0, true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(empty_bin.atom_count_sum == 0 && empty_bin.dof_sum == 0.0);
  assert(empty_bin.temperature_samples == 0 && empty_bin.temperature_sum == 0.0);
  InterfaceProfileBlockAccum zero_temperature{};
  assert(interface_profile_accumulate_sample(
    zero_temperature, 2, true, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(zero_temperature.temperature_samples == 1 && zero_temperature.temperature_sum == 0.0);

  InterfaceProfileBlockAccum unchanged = empty_bin;
  assert(interface_profile_accumulate_sample(
    empty_bin, 2, false, 0.0, nan, 0.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_INVALID);
  assert(empty_bin.atom_count_sum == unchanged.atom_count_sum &&
         empty_bin.potential_sum == unchanged.potential_sum);

  InterfaceProfileBlockAccum long_count{};
  constexpr int large_bin_count = 1500000000;
  assert(interface_profile_accumulate_sample(
    long_count, large_bin_count, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(interface_profile_accumulate_sample(
    long_count, large_bin_count, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(long_count.atom_count_sum == 3000000000ULL);
  InterfaceProfileBlockAccum overflowing_count{};
  overflowing_count.atom_count_sum = std::numeric_limits<unsigned long long>::max();
  assert(interface_profile_accumulate_sample(
    overflowing_count, 1, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k_b) ==
    INTERFACE_PROFILE_SAMPLE_COUNTER_OVERFLOW);
  assert(overflowing_count.atom_count_sum == std::numeric_limits<unsigned long long>::max());

  int error_info[3] = {};
  double error_values[INTERFACE_PROFILE_ERROR_VALUE_COUNT] = {};
  interface_profile_record_first_error(
    error_info, error_values, INTERFACE_PROFILE_SAMPLE_INVALID, 42, 4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  interface_profile_record_first_error(
    error_info, error_values, INTERFACE_PROFILE_SAMPLE_SUM_OVERFLOW, 43, 2, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
  assert(error_info[0] == INTERFACE_PROFILE_SAMPLE_INVALID && error_info[1] == 42 && error_info[2] == 4);
  assert(error_values[INTERFACE_PROFILE_ERROR_RAW_KINETIC] == 1.0);
  interface_profile_reset_error(error_info, error_values);
  interface_profile_reset_block_accum(paired);
  assert(error_info[0] == 0 && error_info[1] == 0 && error_info[2] == 0);
  assert(error_values[INTERFACE_PROFILE_ERROR_POTENTIAL] == 0.0);
  assert(paired.atom_count_sum == 0 && paired.temperature_samples == 0 && paired.kinetic_sum == 0.0);
  assert(interface_profile_accumulate_sample(
    paired, 2, false, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, k_b) == INTERFACE_PROFILE_SAMPLE_OK);
  assert(paired.potential_sum == 1.0 && paired.temperature_samples == 1);

  assert(interface_profile_block_cache_capacity(11, 4) == 3);
  assert(interface_profile_expected_block_samples(11, 4, 0) == 4);
  assert(interface_profile_expected_block_samples(11, 4, 1) == 4);
  assert(interface_profile_expected_block_samples(11, 4, 2) == 3);
  assert(interface_profile_expected_block_samples(11, 4, 3) == 0);
  assert(interface_profile_block_cache_capacity(0, 4) == 0);
  assert(interface_profile_block_cache_capacity(3, 1) == 3);
  std::size_t cache_offset = 99;
  assert(interface_profile_block_cache_offset(2, 7, 3, cache_offset) && cache_offset == 14);
  assert(!interface_profile_block_cache_offset(3, 7, 3, cache_offset));
  return 0;
}
