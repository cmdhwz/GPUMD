#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

struct NEPPlaneFluxView
{
  int enabled = 0;
  int atom_count = 0;
  int plane_count = 0;
  int axis = 0;
  int pbc[3] = {0, 0, 0};
  double box_lengths[3] = {0.0, 0.0, 0.0};
  double max_reference_displacement = 0.0;
  const double* reference_positions = nullptr;
  const double* plane_positions = nullptr;
  const int* plane_offsets = nullptr;
  const int* candidate_offsets = nullptr;
  const int* candidate_planes = nullptr;
  const int* selected_atoms = nullptr;
  double* coefficients = nullptr;
  int* error_flag = nullptr;
  int* first_error_step = nullptr;
  int sample_step = -1;
};

constexpr int NEP_PLANE_FLUX_ERROR_DISPLACEMENT = 1;
constexpr int NEP_PLANE_FLUX_ERROR_MISSING_SLOT = 2;
constexpr int NEP_PLANE_FLUX_ERROR_NONFINITE = 4;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NEP_PLANE_FLUX_ERROR_HD __host__ __device__ __forceinline__
#else
#define NEP_PLANE_FLUX_ERROR_HD inline
#endif

NEP_PLANE_FLUX_ERROR_HD void nep_plane_flux_record_error(
  int* error_flag, int* first_error_step, const int error, const int step)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  atomicOr(error_flag, error);
  if (first_error_step) atomicCAS(first_error_step, -1, step);
#else
  if (error_flag) *error_flag |= error;
  if (first_error_step && *first_error_step < 0) *first_error_step = step;
#endif
}

NEP_PLANE_FLUX_ERROR_HD void nep_plane_flux_reset_error(int* error_flag, int* first_error_step)
{
  if (error_flag) *error_flag = 0;
  if (first_error_step) *first_error_step = -1;
}

#undef NEP_PLANE_FLUX_ERROR_HD

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NEP_PLANE_FLUX_HD __host__ __device__
#else
#define NEP_PLANE_FLUX_HD
#endif

NEP_PLANE_FLUX_HD inline int nep_plane_crossing_direction(
  const double x1,
  const double wrapped_x2,
  const double plane,
  const double length,
  const bool periodic)
{
  double displacement = wrapped_x2 - x1;
  if (periodic) displacement -= ::nearbyint(displacement / length) * length;
  const double x2 = x1 + displacement;
  if (periodic) {
    const double midpoint = 0.5 * (x1 + x2);
    const int image = static_cast<int>(::floor((midpoint - plane) / length + 0.5));
    for (int k = image - 1; k <= image + 1; ++k) {
      const double plane_image = plane + k * length;
      if (x1 < plane_image && plane_image <= x2) return 1;
      if (x2 < plane_image && plane_image <= x1) return -1;
    }
    return 0;
  }
  if (x1 < plane && plane <= x2) return 1;
  if (x2 < plane && plane <= x1) return -1;
  return 0;
}

inline bool nep_plane_flux_bounds_valid(
  const double cutoff,
  const double max_displacement,
  const double length,
  const bool periodic)
{
  return !periodic || (std::isfinite(cutoff) && std::isfinite(max_displacement) &&
                       std::isfinite(length) && cutoff > 0.0 && max_displacement >= 0.0 &&
                       length > 0.0 && cutoff + 2.0 * max_displacement < 0.5 * length);
}

inline bool nep_plane_atom_selected(
  const double x,
  const double plane,
  const double length,
  const bool periodic,
  const double cutoff)
{
  double distance = std::abs(x - plane);
  if (periodic) distance = std::min(distance, length - distance);
  return distance <= cutoff;
}

NEP_PLANE_FLUX_HD inline int nep_plane_find_selected_slot(
  const int* selected_atoms,
  const int* plane_offsets,
  const int plane_count,
  const int plane,
  const int atom)
{
  if (!selected_atoms || !plane_offsets || plane < 0 || plane >= plane_count || atom < 0) return -1;
  int lower = plane_offsets[plane];
  int upper = plane_offsets[plane + 1];
  while (lower < upper) {
    const int middle = lower + (upper - lower) / 2;
    if (selected_atoms[middle] < atom) lower = middle + 1;
    else upper = middle;
  }
  return lower < plane_offsets[plane + 1] && selected_atoms[lower] == atom ? lower : -1;
}

NEP_PLANE_FLUX_HD inline double nep_plane_flux_coefficient_sign(const int crossing_direction)
{
  return -static_cast<double>(crossing_direction);
}

inline bool nep_plane_build_candidate_index(
  const int atom_count,
  const int plane_count,
  const int* selected_atoms,
  const int selected_count,
  const int* plane_offsets,
  std::vector<int>& candidate_offsets,
  std::vector<int>& candidate_planes)
{
  if (atom_count < 0 || plane_count < 0 || selected_count < 0 || !plane_offsets ||
      (selected_count > 0 && !selected_atoms)) return false;
  const std::size_t offset_count = static_cast<std::size_t>(atom_count) + 1;
  if (offset_count > candidate_offsets.max_size() ||
      static_cast<std::size_t>(atom_count) > std::vector<int>().max_size() ||
      static_cast<std::size_t>(selected_count) > candidate_planes.max_size() ||
      plane_offsets[0] != 0 || plane_offsets[plane_count] != selected_count) return false;

  std::vector<int> counts(static_cast<std::size_t>(atom_count), 0);
  candidate_offsets.assign(offset_count, 0);
  candidate_planes.resize(static_cast<std::size_t>(selected_count));
  for (int p = 0; p < plane_count; ++p) {
    if (plane_offsets[p] < 0 || plane_offsets[p] > plane_offsets[p + 1] ||
        plane_offsets[p + 1] > selected_count) return false;
    int previous_atom = -1;
    for (int slot = plane_offsets[p]; slot < plane_offsets[p + 1]; ++slot) {
      const int atom = selected_atoms[slot];
      if (atom <= previous_atom || atom >= atom_count || counts[atom] == std::numeric_limits<int>::max()) return false;
      previous_atom = atom;
      ++counts[atom];
    }
  }
  for (int atom = 0; atom < atom_count; ++atom) {
    const long long next = static_cast<long long>(candidate_offsets[atom]) + counts[atom];
    if (next > std::numeric_limits<int>::max()) return false;
    candidate_offsets[static_cast<std::size_t>(atom) + 1] = static_cast<int>(next);
    counts[atom] = candidate_offsets[atom];
  }
  for (int p = 0; p < plane_count; ++p) {
    for (int slot = plane_offsets[p]; slot < plane_offsets[p + 1]; ++slot) {
      const int atom = selected_atoms[slot];
      candidate_planes[counts[atom]++] = p;
    }
  }
  return candidate_offsets.back() == selected_count;
}

inline double interface_shc_spectrum_value(
  const double dt_ps,
  const double omega,
  const double* correlation_plus,
  const double* correlation_minus,
  const int correlation_samples)
{
  double value = correlation_plus[0];
  for (int lag = 1; lag < correlation_samples; ++lag) {
    value += (correlation_plus[lag] + correlation_minus[lag]) *
             std::cos(omega * (lag * dt_ps));
  }
  return dt_ps * value / 3.14159265358979323846;
}

inline double interface_shc_frequency_integral(
  const double dt_ps,
  const double max_omega,
  const double* correlation_plus,
  const double* correlation_minus,
  const int correlation_samples)
{
  double integral = correlation_plus[0] * max_omega;
  for (int lag = 1; lag < correlation_samples; ++lag) {
    const double tau = lag * dt_ps;
    integral += (correlation_plus[lag] + correlation_minus[lag]) * std::sin(max_omega * tau) / tau;
  }
  return dt_ps * integral / 3.14159265358979323846;
}

inline bool interface_shc_frequency_grid_resolved(
  const double sample_dt_ps,
  const double max_omega,
  const int num_omega,
  const int correlation_samples)
{
  if (!std::isfinite(sample_dt_ps) || sample_dt_ps <= 0.0 ||
      !std::isfinite(max_omega) || max_omega <= 0.0 || num_omega < 2 || correlation_samples < 1)
    return false;
  const double max_phase_step =
    max_omega / (num_omega - 1) * (correlation_samples - 1) * sample_dt_ps;
  return std::isfinite(max_phase_step) && max_phase_step <= 3.14159265358979323846;
}

inline double interface_shc_trapezoid_integral(
  const double* values,
  const int num_omega,
  const double max_omega)
{
  if (num_omega < 2) return std::numeric_limits<double>::quiet_NaN();
  double sum = 0.5 * (values[0] + values[num_omega - 1]);
  for (int i = 1; i < num_omega - 1; ++i) sum += values[i];
  return sum * max_omega / (num_omega - 1);
}

#undef NEP_PLANE_FLUX_HD
