#pragma once

#include <cmath>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define INTERFACE_PROFILE_HOST_DEVICE __host__ __device__
#else
#define INTERFACE_PROFILE_HOST_DEVICE
#endif

struct InterfaceProfileBlockAccum
{
  unsigned long long atom_count_sum = 0;
  unsigned long long temperature_samples = 0;
  double kinetic_sum = 0.0;
  double potential_sum = 0.0;
  double temperature_sum = 0.0;
  double dof_sum = 0.0;
};

struct InterfaceProfileBlockMetadata
{
  int step = 0;
  double time_ps = 0.0;
  double start_time_ps = 0.0;
  int samples = 0;
  double bath_source_eV = 0.0;
  double bath_sink_eV = 0.0;
};

enum InterfaceProfileSampleError
{
  INTERFACE_PROFILE_SAMPLE_OK = 0,
  INTERFACE_PROFILE_SAMPLE_INVALID = 1,
  INTERFACE_PROFILE_SAMPLE_COUNTER_OVERFLOW = 2,
  INTERFACE_PROFILE_SAMPLE_SUM_OVERFLOW = 3
};

enum InterfaceProfileErrorValue
{
  INTERFACE_PROFILE_ERROR_RAW_KINETIC = 0,
  INTERFACE_PROFILE_ERROR_MASS_SUM = 1,
  INTERFACE_PROFILE_ERROR_MOMENTUM_X = 2,
  INTERFACE_PROFILE_ERROR_MOMENTUM_Y = 3,
  INTERFACE_PROFILE_ERROR_MOMENTUM_Z = 4,
  INTERFACE_PROFILE_ERROR_POTENTIAL = 5,
  INTERFACE_PROFILE_ERROR_VALUE_COUNT = 6
};

INTERFACE_PROFILE_HOST_DEVICE inline bool interface_profile_isfinite(const double value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

INTERFACE_PROFILE_HOST_DEVICE inline double interface_profile_abs(const double value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return fabs(value);
#else
  return std::fabs(value);
#endif
}

INTERFACE_PROFILE_HOST_DEVICE inline double interface_profile_dof(
  const int atom_count, const bool remove_drift)
{
  return 3.0 * atom_count - (remove_drift && atom_count > 0 ? 3.0 : 0.0);
}

INTERFACE_PROFILE_HOST_DEVICE inline double interface_profile_kinetic_roundoff_tolerance(
  const double raw_kinetic, const double com_kinetic, const int atom_count)
{
  if (!interface_profile_isfinite(raw_kinetic) ||
      !interface_profile_isfinite(com_kinetic) || atom_count < 0) {
    return NAN;
  }
  // Bound atomic-reduction and COM arithmetic roundoff by a count-scaled energy error envelope.
  const int count = atom_count > 1 ? atom_count : 1;
  const double reduction_error_factor = 64.0 * count * 2.2204460492503131e-16;
  return reduction_error_factor *
         (interface_profile_abs(raw_kinetic) + interface_profile_abs(com_kinetic));
}

INTERFACE_PROFILE_HOST_DEVICE inline bool interface_profile_corrected_kinetic(
  const int atom_count,
  const bool remove_drift,
  const double raw_kinetic,
  const double mass_sum,
  const double px,
  const double py,
  const double pz,
  double& corrected_kinetic)
{
  if (atom_count < 0 || !interface_profile_isfinite(raw_kinetic) || raw_kinetic < 0.0) return false;
  if (!remove_drift) {
    if (atom_count == 0 && raw_kinetic != 0.0) return false;
    corrected_kinetic = raw_kinetic;
    return true;
  }
  if (!interface_profile_isfinite(mass_sum) || !interface_profile_isfinite(px) ||
      !interface_profile_isfinite(py) || !interface_profile_isfinite(pz)) {
    return false;
  }
  if (atom_count == 0) {
    if (raw_kinetic != 0.0 || mass_sum != 0.0 || px != 0.0 || py != 0.0 || pz != 0.0) return false;
    corrected_kinetic = 0.0;
    return true;
  }
  if (mass_sum <= 0.0) return false;

  const double momentum_squared = px * px + py * py + pz * pz;
  if (!interface_profile_isfinite(momentum_squared) || momentum_squared < 0.0) return false;
  const double com_kinetic = 0.5 * momentum_squared / mass_sum;
  if (!interface_profile_isfinite(com_kinetic) || com_kinetic < 0.0) return false;
  double adjusted = raw_kinetic - com_kinetic;
  if (!interface_profile_isfinite(adjusted)) return false;
  if (adjusted < 0.0) {
    const double tolerance = interface_profile_kinetic_roundoff_tolerance(
      raw_kinetic, com_kinetic, atom_count);
    if (!interface_profile_isfinite(tolerance) || adjusted < -tolerance) return false;
    adjusted = 0.0;
  }
  corrected_kinetic = adjusted;
  return true;
}

INTERFACE_PROFILE_HOST_DEVICE inline double interface_profile_temperature(
  const double kinetic, const double dof, const double boltzmann_constant)
{
  if (!interface_profile_isfinite(kinetic) || kinetic < 0.0 ||
      !interface_profile_isfinite(dof) || dof <= 0.0 ||
      !interface_profile_isfinite(boltzmann_constant) || boltzmann_constant <= 0.0) {
    return NAN;
  }
  return 2.0 * kinetic / (boltzmann_constant * dof);
}

INTERFACE_PROFILE_HOST_DEVICE inline void interface_profile_reset_block_accum(
  InterfaceProfileBlockAccum& block)
{
  block.atom_count_sum = 0;
  block.temperature_samples = 0;
  block.kinetic_sum = 0.0;
  block.potential_sum = 0.0;
  block.temperature_sum = 0.0;
  block.dof_sum = 0.0;
}

INTERFACE_PROFILE_HOST_DEVICE inline int interface_profile_accumulate_sample(
  InterfaceProfileBlockAccum& block,
  const int atom_count,
  const bool remove_drift,
  const double raw_kinetic,
  const double potential,
  const double mass_sum,
  const double px,
  const double py,
  const double pz,
  const double boltzmann_constant)
{
  if (atom_count < 0 || !interface_profile_isfinite(potential)) {
    return INTERFACE_PROFILE_SAMPLE_INVALID;
  }
  double kinetic = 0.0;
  if (!interface_profile_corrected_kinetic(
        atom_count, remove_drift, raw_kinetic, mass_sum, px, py, pz, kinetic)) {
    return INTERFACE_PROFILE_SAMPLE_INVALID;
  }
  const double dof = interface_profile_dof(atom_count, remove_drift);
  double temperature = 0.0;
  if (dof > 0.0) {
    temperature = interface_profile_temperature(kinetic, dof, boltzmann_constant);
    if (!interface_profile_isfinite(temperature)) return INTERFACE_PROFILE_SAMPLE_INVALID;
  }
  const unsigned long long count = static_cast<unsigned long long>(atom_count);
  const unsigned long long max_count = ~static_cast<unsigned long long>(0);
  if (block.atom_count_sum > max_count - count ||
      (dof > 0.0 && block.temperature_samples == max_count)) {
    return INTERFACE_PROFILE_SAMPLE_COUNTER_OVERFLOW;
  }
  const double new_kinetic_sum = block.kinetic_sum + kinetic;
  const double new_potential_sum = block.potential_sum + potential;
  const double new_dof_sum = block.dof_sum + dof;
  const double new_temperature_sum = block.temperature_sum + (dof > 0.0 ? temperature : 0.0);
  if (!interface_profile_isfinite(new_kinetic_sum) ||
      !interface_profile_isfinite(new_potential_sum) ||
      !interface_profile_isfinite(new_dof_sum) ||
      !interface_profile_isfinite(new_temperature_sum)) {
    return INTERFACE_PROFILE_SAMPLE_SUM_OVERFLOW;
  }
  block.atom_count_sum += count;
  block.kinetic_sum = new_kinetic_sum;
  block.potential_sum = new_potential_sum;
  block.dof_sum = new_dof_sum;
  block.temperature_sum = new_temperature_sum;
  if (dof > 0.0) ++block.temperature_samples;
  return INTERFACE_PROFILE_SAMPLE_OK;
}

inline std::size_t interface_profile_block_cache_capacity(
  const std::size_t sample_count, const std::size_t samples_per_block)
{
  if (samples_per_block == 0) return 0;
  return sample_count / samples_per_block + (sample_count % samples_per_block != 0);
}

inline std::size_t interface_profile_expected_block_samples(
  const std::size_t sample_count,
  const std::size_t samples_per_block,
  const std::size_t block_index)
{
  if (samples_per_block == 0 || block_index > sample_count / samples_per_block) return 0;
  const std::size_t full_blocks = sample_count / samples_per_block;
  if (block_index < full_blocks) return samples_per_block;
  return block_index == full_blocks ? sample_count % samples_per_block : 0;
}

inline bool interface_profile_block_cache_offset(
  const std::size_t block_index,
  const std::size_t bins,
  const std::size_t capacity,
  std::size_t& offset)
{
  if (block_index >= capacity || (bins && block_index > static_cast<std::size_t>(-1) / bins)) {
    return false;
  }
  offset = block_index * bins;
  return true;
}

INTERFACE_PROFILE_HOST_DEVICE inline void interface_profile_reset_error(
  int* error_info, double* error_values)
{
  for (int i = 0; i < 3; ++i) error_info[i] = 0;
  for (int i = 0; i < INTERFACE_PROFILE_ERROR_VALUE_COUNT; ++i) error_values[i] = 0.0;
}

INTERFACE_PROFILE_HOST_DEVICE inline void interface_profile_record_first_error(
  int* error_info,
  double* error_values,
  const int error_code,
  const int step,
  const int bin,
  const double raw_kinetic,
  const double mass_sum,
  const double px,
  const double py,
  const double pz,
  const double potential)
{
  int won = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  won = atomicCAS(error_info, 0, error_code) == 0;
#else
  if (error_info[0] == 0) {
    error_info[0] = error_code;
    won = 1;
  }
#endif
  if (!won) return;
  error_info[1] = step;
  error_info[2] = bin;
  error_values[INTERFACE_PROFILE_ERROR_RAW_KINETIC] = raw_kinetic;
  error_values[INTERFACE_PROFILE_ERROR_MASS_SUM] = mass_sum;
  error_values[INTERFACE_PROFILE_ERROR_MOMENTUM_X] = px;
  error_values[INTERFACE_PROFILE_ERROR_MOMENTUM_Y] = py;
  error_values[INTERFACE_PROFILE_ERROR_MOMENTUM_Z] = pz;
  error_values[INTERFACE_PROFILE_ERROR_POTENTIAL] = potential;
}

#undef INTERFACE_PROFILE_HOST_DEVICE
