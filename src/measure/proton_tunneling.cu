/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Track centroid O-H-O geometry, local proton-state bias, persistent state changes, and optional
bead-resolved tunneling-like event diagnostics.

This is deliberately an observer. It does not alter the NEP/qNEP energy, force, charge, or
heat-current paths. The sparse transfer and defect streams can be reconstructed into defect
propagation chains offline.
--------------------------------------------------------------------------------------------------*/

#include "proton_tunneling.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#ifdef USE_NETCDF
#include "netcdf.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <utility>

namespace
{
#ifdef USE_NETCDF
void netcdf_check(const int status, const char* operation)
{
  if (status != NC_NOERR) {
    fprintf(stderr, "Proton observer NetCDF error in %s: %s\n", operation, nc_strerror(status));
    std::exit(2);
  }
}

void netcdf_text_attribute(const int group, const int variable, const char* name, const char* value)
{
  netcdf_check(
    nc_put_att_text(group, variable, name, std::strlen(value), value), "nc_put_att_text");
}

int netcdf_dimension(const int group, const char* name, const size_t length)
{
  int dimension = -1;
  netcdf_check(nc_def_dim(group, name, length, &dimension), "nc_def_dim");
  return dimension;
}

int netcdf_variable(
  const int group,
  const char* name,
  const nc_type type,
  const std::vector<int>& dimensions,
  const std::vector<size_t>& lengths,
  const int compression_level)
{
  int variable = -1;
  netcdf_check(
    nc_def_var(group, name, type, static_cast<int>(dimensions.size()), dimensions.data(), &variable),
    "nc_def_var");
  if (!dimensions.empty()) {
    std::vector<size_t> chunks(dimensions.size(), 1);
    for (size_t i = 0; i < dimensions.size(); ++i)
      chunks[i] = std::max<size_t>(1, std::min<size_t>(lengths[i], 16384));
    netcdf_check(nc_def_var_chunking(group, variable, NC_CHUNKED, chunks.data()), "nc_def_var_chunking");
    netcdf_check(
      nc_def_var_deflate(group, variable, 1, 1, compression_level), "nc_def_var_deflate");
  }
  return variable;
}

void netcdf_write_double(const int group, const int variable, const std::vector<double>& values)
{
  if (!values.empty())
    netcdf_check(nc_put_var_double(group, variable, values.data()), "nc_put_var_double");
}

void netcdf_write_int(const int group, const int variable, const std::vector<int>& values)
{
  if (!values.empty())
    netcdf_check(nc_put_var_int(group, variable, values.data()), "nc_put_var_int");
}

void netcdf_write_ubyte(
  const int group,
  const int variable,
  const std::vector<unsigned char>& values)
{
  if (!values.empty())
    netcdf_check(nc_put_var_ubyte(group, variable, values.data()), "nc_put_var_ubyte");
}

void netcdf_write_longlong(
  const int group,
  const int variable,
  const std::vector<long long>& values)
{
  if (!values.empty())
    netcdf_check(nc_put_var_longlong(group, variable, values.data()), "nc_put_var_longlong");
}

#endif

unsigned long long make_bond_key(const int oxygen_low, const int oxygen_high)
{
  return (static_cast<unsigned long long>(static_cast<unsigned int>(oxygen_low)) << 32) |
    static_cast<unsigned int>(oxygen_high);
}

void decode_bond_key(
  const unsigned long long key,
  int& oxygen_low,
  int& oxygen_high)
{
  oxygen_low = static_cast<int>(key >> 32);
  oxygen_high = static_cast<int>(key & 0xffffffffULL);
}

__global__ void pack_bead_positions(
  const int position_size,
  const int number_of_beads,
  double* const* position_beads,
  double* packed_positions)
{
  const int component = blockDim.x * blockIdx.x + threadIdx.x;
  const int bead = blockIdx.y;
  if (bead < number_of_beads && component < position_size)
    packed_positions[static_cast<size_t>(bead) * position_size + component] =
      position_beads[bead][component];
}

__device__ bool geometry_candidate_is_better(
  const GeometryResultGPU& first,
  const GeometryResultGPU& second)
{
  if (first.assignment_score != second.assignment_score)
    return first.assignment_score < second.assignment_score;
  return first.rperp < second.rperp;
}

__device__ void compute_ion_field_gpu(
  const double* position,
  const int number_of_atoms,
  const Box& box,
  const GeometryResultGPU& base_geometry,
  const int* ion1_indices,
  const int ion1_count,
  const double ion1_charge,
  const int* ion2_indices,
  const int ion2_count,
  const double ion2_charge,
  const double ion_field_cutoff,
  GeometryResultGPU& geometry)
{
  geometry.E_ion_nominal_parallel = 0.0;
  geometry.delta_phi_ion = 0.0;
  geometry.nearest_ion_id = -1;
  geometry.nearest_ion_distance = 1.0e300;
  geometry.nearest_ion1_distance = 1.0e300;
  geometry.nearest_ion2_distance = 1.0e300;
  geometry.nearest_ion1_to_low = 1.0e300;
  geometry.nearest_ion1_to_high = 1.0e300;
  geometry.nearest_ion2_to_low = 1.0e300;
  geometry.nearest_ion2_to_high = 1.0e300;
  const double inverse_dOO = 1.0 / base_geometry.dOO;
  const double ex = base_geometry.low_to_high_dx * inverse_dOO;
  const double ey = base_geometry.low_to_high_dy * inverse_dOO;
  const double ez = base_geometry.low_to_high_dz * inverse_dOO;

  for (int species = 0; species < 2; ++species) {
    const int* ions = (species == 0) ? ion1_indices : ion2_indices;
    const int ion_count = (species == 0) ? ion1_count : ion2_count;
    const double charge = (species == 0) ? ion1_charge : ion2_charge;
    double& nearest_species_distance = (species == 0)
      ? geometry.nearest_ion1_distance
      : geometry.nearest_ion2_distance;
    double& nearest_species_to_low = (species == 0)
      ? geometry.nearest_ion1_to_low
      : geometry.nearest_ion2_to_low;
    double& nearest_species_to_high = (species == 0)
      ? geometry.nearest_ion1_to_high
      : geometry.nearest_ion2_to_high;
    for (int i = 0; i < ion_count; ++i) {
      const int ion = ions[i];
      double ion_dx = position[ion] - position[base_geometry.oxygen_low];
      double ion_dy = position[ion + number_of_atoms] -
        position[base_geometry.oxygen_low + number_of_atoms];
      double ion_dz = position[ion + 2 * number_of_atoms] -
        position[base_geometry.oxygen_low + 2 * number_of_atoms];
      apply_mic(box, ion_dx, ion_dy, ion_dz);
      double midpoint_to_ion_x = 0.5 * base_geometry.low_to_high_dx - ion_dx;
      double midpoint_to_ion_y = 0.5 * base_geometry.low_to_high_dy - ion_dy;
      double midpoint_to_ion_z = 0.5 * base_geometry.low_to_high_dz - ion_dz;
      apply_mic(box, midpoint_to_ion_x, midpoint_to_ion_y, midpoint_to_ion_z);
      double ion_to_high_x = position[ion] - position[base_geometry.oxygen_high];
      double ion_to_high_y = position[ion + number_of_atoms] -
        position[base_geometry.oxygen_high + number_of_atoms];
      double ion_to_high_z = position[ion + 2 * number_of_atoms] -
        position[base_geometry.oxygen_high + 2 * number_of_atoms];
      apply_mic(box, ion_to_high_x, ion_to_high_y, ion_to_high_z);
      const double low_distance_square = ion_dx * ion_dx + ion_dy * ion_dy + ion_dz * ion_dz;
      const double high_distance_square = ion_to_high_x * ion_to_high_x +
        ion_to_high_y * ion_to_high_y + ion_to_high_z * ion_to_high_z;
      const double low_distance = sqrt(low_distance_square);
      const double high_distance = sqrt(high_distance_square);
      const double distance_square = midpoint_to_ion_x * midpoint_to_ion_x +
        midpoint_to_ion_y * midpoint_to_ion_y + midpoint_to_ion_z * midpoint_to_ion_z;
      const double distance = sqrt(distance_square);
      if (distance < nearest_species_distance)
        nearest_species_distance = distance;
      if (low_distance < nearest_species_to_low)
        nearest_species_to_low = low_distance;
      if (high_distance < nearest_species_to_high)
        nearest_species_to_high = high_distance;
      if (distance < geometry.nearest_ion_distance) {
        geometry.nearest_ion_distance = distance;
        geometry.nearest_ion_id = ion;
      }
      if (distance <= ion_field_cutoff && distance > 1.0e-12) {
        geometry.E_ion_nominal_parallel += K_C * charge *
          (midpoint_to_ion_x * ex + midpoint_to_ion_y * ey + midpoint_to_ion_z * ez) /
          (distance_square * distance);
        if (low_distance > 1.0e-12 && high_distance > 1.0e-12)
          geometry.delta_phi_ion += K_C * charge *
            (1.0 / high_distance - 1.0 / low_distance);
      }
    }
  }
}

__global__ void gpu_find_proton_geometry(
  const double* position,
  const int number_of_atoms,
  const int* oxygen_indices,
  const int number_of_oxygen,
  const int* hydrogen_indices,
  const int* oxygen_local_index,
  const int* shell_offsets,
  const int* shell_neighbors,
  const int* ion1_indices,
  const int ion1_count,
  const double ion1_charge,
  const int* ion2_indices,
  const int ion2_count,
  const double ion2_charge,
  const Box box,
  const double dOO_min,
  const double dOO_max,
  const double rperp_max,
  const double angle_cosine_limit,
  const double assignment_path_excess_weight,
  const double assignment_score_gap_min,
  const int ion_field_enabled,
  const double ion_field_cutoff,
  GeometryResultGPU* output)
{
  const int hydrogen_index = blockIdx.x;
  const int hydrogen = hydrogen_indices[hydrogen_index];
  const int thread = threadIdx.x;
  __shared__ double nearest_distance_square[128];
  __shared__ int nearest_oxygen[128];
  __shared__ GeometryResultGPU candidate_results[8];
  __shared__ int candidate_valid[8];

  double local_distance_square = 1.0e300;
  int local_oxygen = -1;
  const double hx = position[hydrogen];
  const double hy = position[hydrogen + number_of_atoms];
  const double hz = position[hydrogen + 2 * number_of_atoms];
  for (int i = thread; i < number_of_oxygen; i += blockDim.x) {
    const int oxygen = oxygen_indices[i];
    double dx = position[oxygen] - hx;
    double dy = position[oxygen + number_of_atoms] - hy;
    double dz = position[oxygen + 2 * number_of_atoms] - hz;
    apply_mic(box, dx, dy, dz);
    const double distance_square = dx * dx + dy * dy + dz * dz;
    if (distance_square < local_distance_square ||
        (distance_square == local_distance_square && oxygen < local_oxygen)) {
      local_distance_square = distance_square;
      local_oxygen = oxygen;
    }
  }
  nearest_distance_square[thread] = local_distance_square;
  nearest_oxygen[thread] = local_oxygen;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      const double other_distance_square = nearest_distance_square[thread + stride];
      const int other_oxygen = nearest_oxygen[thread + stride];
      if (other_distance_square < nearest_distance_square[thread] ||
          (other_distance_square == nearest_distance_square[thread] &&
           other_oxygen >= 0 &&
           (nearest_oxygen[thread] < 0 || other_oxygen < nearest_oxygen[thread]))) {
        nearest_distance_square[thread] = other_distance_square;
        nearest_oxygen[thread] = other_oxygen;
      }
    }
    __syncthreads();
  }

  if (thread == 0) {
    GeometryResultGPU empty_geometry = {};
    empty_geometry.nearest_oxygen = nearest_oxygen[0];
    empty_geometry.nearest_ion_id = -1;
    output[hydrogen_index] = empty_geometry;
  }
  __syncthreads();
  const int first_oxygen = nearest_oxygen[0];
  if (first_oxygen < 0)
    return;

  const int anchor_local = (first_oxygen < number_of_atoms)
    ? oxygen_local_index[first_oxygen]
    : -1;
  const int shell_begin = (anchor_local >= 0) ? shell_offsets[anchor_local] : 0;
  const int shell_end = (anchor_local >= 0) ? shell_offsets[anchor_local + 1] : 0;
  const int shell_size = (shell_end - shell_begin < 8)
    ? (shell_end - shell_begin)
    : 8;
  if (thread < 8) {
    candidate_results[thread] = {};
    candidate_valid[thread] = 0;
  }
  __syncthreads();

  if (thread < shell_size) {
    const int second_oxygen = shell_neighbors[shell_begin + thread];
    const int second_local = (second_oxygen < number_of_atoms)
      ? oxygen_local_index[second_oxygen]
      : -1;
    bool mutual_neighbor = false;
    if (second_local >= 0) {
      for (int i = shell_offsets[second_local]; i < shell_offsets[second_local + 1]; ++i) {
        if (shell_neighbors[i] == first_oxygen) {
          mutual_neighbor = true;
          break;
        }
      }
    }
    if (mutual_neighbor) {
      GeometryResultGPU candidate = {};
      candidate.nearest_oxygen = first_oxygen;
      candidate.nearest_ion_id = -1;
      candidate.oxygen_low = (first_oxygen < second_oxygen) ? first_oxygen : second_oxygen;
      candidate.oxygen_high = (first_oxygen < second_oxygen) ? second_oxygen : first_oxygen;
      double ox = position[candidate.oxygen_high] - position[candidate.oxygen_low];
      double oy = position[candidate.oxygen_high + number_of_atoms] -
        position[candidate.oxygen_low + number_of_atoms];
      double oz = position[candidate.oxygen_high + 2 * number_of_atoms] -
        position[candidate.oxygen_low + 2 * number_of_atoms];
      apply_mic(box, ox, oy, oz);
      const double dOO_square = ox * ox + oy * oy + oz * oz;
      if (dOO_square >= dOO_min * dOO_min && dOO_square <= dOO_max * dOO_max) {
        candidate.dOO = sqrt(dOO_square);
        candidate.low_to_high_dx = ox;
        candidate.low_to_high_dy = oy;
        candidate.low_to_high_dz = oz;

        double hx_from_low = hx - position[candidate.oxygen_low];
        double hy_from_low = hy - position[candidate.oxygen_low + number_of_atoms];
        double hz_from_low = hz - position[candidate.oxygen_low + 2 * number_of_atoms];
        apply_mic(box, hx_from_low, hy_from_low, hz_from_low);
        const double projection = (hx_from_low * ox + hy_from_low * oy + hz_from_low * oz) /
          dOO_square;
        if (projection >= -1.0e-10 && projection <= 1.0 + 1.0e-10) {
          const double px = hx_from_low - projection * ox;
          const double py = hy_from_low - projection * oy;
          const double pz = hz_from_low - projection * oz;
          candidate.rperp = sqrt(fmax(0.0, px * px + py * py + pz * pz));
          if (candidate.rperp <= rperp_max) {
            double low_x = position[candidate.oxygen_low] - hx;
            double low_y = position[candidate.oxygen_low + number_of_atoms] - hy;
            double low_z = position[candidate.oxygen_low + 2 * number_of_atoms] - hz;
            double high_x = position[candidate.oxygen_high] - hx;
            double high_y = position[candidate.oxygen_high + number_of_atoms] - hy;
            double high_z = position[candidate.oxygen_high + 2 * number_of_atoms] - hz;
            apply_mic(box, low_x, low_y, low_z);
            apply_mic(box, high_x, high_y, high_z);
            const double low_distance = sqrt(low_x * low_x + low_y * low_y + low_z * low_z);
            const double high_distance = sqrt(high_x * high_x + high_y * high_y + high_z * high_z);
            if (low_distance > 0.0 && high_distance > 0.0 &&
                low_distance <= 1.60 && high_distance <= 1.60) {
              const double angle_cosine = (low_x * high_x + low_y * high_y + low_z * high_z) /
                (low_distance * high_distance);
              if (angle_cosine <= angle_cosine_limit) {
                candidate.delta = low_distance - high_distance;
                candidate.path_excess = fmax(0.0,
                  low_distance + high_distance - candidate.dOO);
                candidate.assignment_score = candidate.rperp +
                  assignment_path_excess_weight * candidate.path_excess;
                if (ion_field_enabled != 0) {
                  compute_ion_field_gpu(
                    position,
                    number_of_atoms,
                    box,
                    candidate,
                    ion1_indices,
                    ion1_count,
                    ion1_charge,
                    ion2_indices,
                    ion2_count,
                    ion2_charge,
                    ion_field_cutoff,
                    candidate);
                }
                candidate.valid = 1;
                candidate_valid[thread] = 1;
                candidate_results[thread] = candidate;
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();

  if (thread == 0) {
    int best_candidate = -1;
    int second_candidate = -1;
    int candidate_count = 0;
    for (int i = 0; i < shell_size; ++i) {
      if (candidate_valid[i] == 0)
        continue;
      ++candidate_count;
      if (best_candidate < 0 ||
          geometry_candidate_is_better(candidate_results[i], candidate_results[best_candidate])) {
        second_candidate = best_candidate;
        best_candidate = i;
      } else if (second_candidate < 0 ||
                 geometry_candidate_is_better(candidate_results[i], candidate_results[second_candidate])) {
        second_candidate = i;
      }
    }
    if (best_candidate >= 0) {
      GeometryResultGPU selected = candidate_results[best_candidate];
      selected.candidate_count = candidate_count;
      selected.second_assignment_score = (second_candidate >= 0)
        ? candidate_results[second_candidate].assignment_score
        : 1.0e300;
      selected.assignment_score_gap = selected.second_assignment_score -
        selected.assignment_score;
      selected.assignment_ambiguous = (candidate_count > 1 &&
        selected.assignment_score_gap < assignment_score_gap_min) ? 1 : 0;
      selected.pair_conflict = 0;
      selected.valid = (selected.assignment_ambiguous == 0) ? 1 : 0;
      output[hydrogen_index] = selected;
    }
  }
}

__device__ void update_nearest_three(
  const double distance,
  double& first,
  double& second,
  double& third)
{
  if (distance < first) {
    third = second;
    second = first;
    first = distance;
  } else if (distance < second) {
    third = second;
    second = distance;
  } else if (distance < third) {
    third = distance;
  }
}

__device__ double angle_from_vectors(
  const double ax,
  const double ay,
  const double az,
  const double bx,
  const double by,
  const double bz)
{
  const double a2 = ax * ax + ay * ay + az * az;
  const double b2 = bx * bx + by * by + bz * bz;
  if (a2 <= 1.0e-24 || b2 <= 1.0e-24)
    return -1.0;
  double cosine = (ax * bx + ay * by + az * bz) / sqrt(a2 * b2);
  cosine = fmax(-1.0, fmin(1.0, cosine));
  return acos(cosine) * 57.29577951308232;
}

__global__ void gpu_compute_proton_local_environment(
  const double* position,
  const int number_of_atoms,
  const int number_of_hydrogens,
  const int* hydrogen_indices,
  const GeometryResultGPU* geometries,
  const int* ion1_indices,
  const int ion1_count,
  const int* ion2_indices,
  const int ion2_count,
  const Box box,
  const double ion1_cutoff,
  const double ion2_cutoff,
  const double hcl_cutoff,
  const double hcl_angle_min_deg,
  LocalEnvironmentGPU* output)
{
  const int hydrogen_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (hydrogen_index >= number_of_hydrogens)
    return;
  const GeometryResultGPU geometry = geometries[hydrogen_index];
  LocalEnvironmentGPU environment = {};
  environment.valid = 0;
  environment.ion1_low_d1 = 1.0e300;
  environment.ion1_low_d2 = 1.0e300;
  environment.ion1_low_d3 = 1.0e300;
  environment.ion1_high_d1 = 1.0e300;
  environment.ion1_high_d2 = 1.0e300;
  environment.ion1_high_d3 = 1.0e300;
  environment.ion2_low_d1 = 1.0e300;
  environment.ion2_low_d2 = 1.0e300;
  environment.ion2_low_d3 = 1.0e300;
  environment.ion2_high_d1 = 1.0e300;
  environment.ion2_high_d2 = 1.0e300;
  environment.ion2_high_d3 = 1.0e300;
  environment.nearest_ion2_to_H = 1.0e300;
  environment.angle_Olow_H_ion2 = -1.0;
  environment.angle_Ohigh_H_ion2 = -1.0;
  if (geometry.valid == 0) {
    output[hydrogen_index] = environment;
    return;
  }

  const int hydrogen = hydrogen_indices[hydrogen_index];
  const int oxygen_low = geometry.oxygen_low;
  const int oxygen_high = geometry.oxygen_high;
  double low_x = position[oxygen_low] - position[hydrogen];
  double low_y = position[oxygen_low + number_of_atoms] - position[hydrogen + number_of_atoms];
  double low_z = position[oxygen_low + 2 * number_of_atoms] -
    position[hydrogen + 2 * number_of_atoms];
  double high_x = position[oxygen_high] - position[hydrogen];
  double high_y = position[oxygen_high + number_of_atoms] -
    position[hydrogen + number_of_atoms];
  double high_z = position[oxygen_high + 2 * number_of_atoms] -
    position[hydrogen + 2 * number_of_atoms];
  apply_mic(box, low_x, low_y, low_z);
  apply_mic(box, high_x, high_y, high_z);
  const double low_distance = sqrt(low_x * low_x + low_y * low_y + low_z * low_z);
  const double high_distance = sqrt(high_x * high_x + high_y * high_y + high_z * high_z);
  environment.valid = (low_distance > 1.0e-12 && high_distance > 1.0e-12) ? 1 : 0;
  if (environment.valid == 0) {
    output[hydrogen_index] = environment;
    return;
  }
  environment.rOH_low = low_distance;
  environment.rOH_high = high_distance;
  environment.oho_angle = angle_from_vectors(low_x, low_y, low_z, high_x, high_y, high_z);
  environment.dOO = geometry.dOO;
  environment.rperp = geometry.rperp;
  environment.path_excess = geometry.path_excess;
  environment.delta_d_ion1 = 0.0;
  environment.delta_d_ion2 = 0.0;
  environment.E_parallel = geometry.E_ion_nominal_parallel;
  environment.delta_phi_ion = geometry.delta_phi_ion;

  for (int species = 0; species < 2; ++species) {
    const int* ions = (species == 0) ? ion1_indices : ion2_indices;
    const int ion_count = (species == 0) ? ion1_count : ion2_count;
    for (int i = 0; i < ion_count; ++i) {
      const int ion = ions[i];
      double ion_low_x = position[ion] - position[oxygen_low];
      double ion_low_y = position[ion + number_of_atoms] -
        position[oxygen_low + number_of_atoms];
      double ion_low_z = position[ion + 2 * number_of_atoms] -
        position[oxygen_low + 2 * number_of_atoms];
      double ion_high_x = position[ion] - position[oxygen_high];
      double ion_high_y = position[ion + number_of_atoms] -
        position[oxygen_high + number_of_atoms];
      double ion_high_z = position[ion + 2 * number_of_atoms] -
        position[oxygen_high + 2 * number_of_atoms];
      apply_mic(box, ion_low_x, ion_low_y, ion_low_z);
      apply_mic(box, ion_high_x, ion_high_y, ion_high_z);
      const double low_ion_distance = sqrt(
        ion_low_x * ion_low_x + ion_low_y * ion_low_y + ion_low_z * ion_low_z);
      const double high_ion_distance = sqrt(
        ion_high_x * ion_high_x + ion_high_y * ion_high_y + ion_high_z * ion_high_z);
      if (species == 0) {
        if (low_ion_distance <= ion1_cutoff)
          ++environment.coord_ion1_low;
        if (high_ion_distance <= ion1_cutoff)
          ++environment.coord_ion1_high;
        update_nearest_three(
          low_ion_distance, environment.ion1_low_d1, environment.ion1_low_d2,
          environment.ion1_low_d3);
        update_nearest_three(
          high_ion_distance, environment.ion1_high_d1, environment.ion1_high_d2,
          environment.ion1_high_d3);
      } else {
        if (low_ion_distance <= ion2_cutoff)
          ++environment.coord_ion2_low;
        if (high_ion_distance <= ion2_cutoff)
          ++environment.coord_ion2_high;
        update_nearest_three(
          low_ion_distance, environment.ion2_low_d1, environment.ion2_low_d2,
          environment.ion2_low_d3);
        update_nearest_three(
          high_ion_distance, environment.ion2_high_d1, environment.ion2_high_d2,
          environment.ion2_high_d3);
        double ion_h_x = position[ion] - position[hydrogen];
        double ion_h_y = position[ion + number_of_atoms] - position[hydrogen + number_of_atoms];
        double ion_h_z = position[ion + 2 * number_of_atoms] -
          position[hydrogen + 2 * number_of_atoms];
        apply_mic(box, ion_h_x, ion_h_y, ion_h_z);
        const double h_distance = sqrt(ion_h_x * ion_h_x + ion_h_y * ion_h_y + ion_h_z * ion_h_z);
        if (h_distance < environment.nearest_ion2_to_H) {
          environment.nearest_ion2_to_H = h_distance;
          environment.angle_Olow_H_ion2 = angle_from_vectors(
            low_x, low_y, low_z, ion_h_x, ion_h_y, ion_h_z);
          environment.angle_Ohigh_H_ion2 = angle_from_vectors(
            high_x, high_y, high_z, ion_h_x, ion_h_y, ion_h_z);
          environment.hcl_like_low =
            (h_distance <= hcl_cutoff &&
             environment.angle_Olow_H_ion2 >= hcl_angle_min_deg) ? 1 : 0;
          environment.hcl_like_high =
            (h_distance <= hcl_cutoff &&
             environment.angle_Ohigh_H_ion2 >= hcl_angle_min_deg) ? 1 : 0;
        }
      }
    }
  }
  environment.delta_d_ion1 =
    (environment.ion1_low_d1 < 1.0e299 && environment.ion1_high_d1 < 1.0e299)
      ? environment.ion1_high_d1 - environment.ion1_low_d1 : 1.0e300;
  environment.delta_d_ion2 =
    (environment.ion2_low_d1 < 1.0e299 && environment.ion2_high_d1 < 1.0e299)
      ? environment.ion2_high_d1 - environment.ion2_low_d1 : 1.0e300;
  output[hydrogen_index] = environment;
}
}

Proton_Tunneling::Proton_Tunneling(
  const char** param,
  const int num_param,
  const Atom& atom)
{
  parse(param, num_param, atom);
  property_name = "compute_proton_tunneling";
}

void Proton_Tunneling::parse(
  const char** param,
  const int num_param,
  const Atom& atom)
{
  printf("Compute centroid proton tunneling observer.\n");

  if (num_param < 8) {
    PRINT_INPUT_ERROR("compute_proton_tunneling should have at least 7 parameters.");
  }
  if (!is_valid_int(param[1], &sample_interval_) || sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("proton tunneling sample interval should be a positive integer.");
  }
  if (!is_valid_int(param[2], &window_samples_) || window_samples_ <= 0) {
    PRINT_INPUT_ERROR("proton tunneling window size should be a positive integer.");
  }
  if (!is_valid_real(param[3], &delta_cutoff_) || delta_cutoff_ <= 0.0) {
    PRINT_INPUT_ERROR("proton tunneling delta cutoff should be a positive number.");
  }
  if (!is_valid_int(param[4], &hold_samples_) || hold_samples_ <= 0) {
    PRINT_INPUT_ERROR("proton tunneling hold samples should be a positive integer.");
  }
  if (!is_valid_real(param[5], &dOO_min_) || !is_valid_real(param[6], &dOO_max_) ||
      dOO_min_ <= 0.0 || dOO_max_ <= dOO_min_) {
    PRINT_INPUT_ERROR("proton tunneling O-O distance range is invalid.");
  }
  if (!is_valid_real(param[7], &rperp_max_) || rperp_max_ <= 0.0) {
    PRINT_INPUT_ERROR("proton tunneling perpendicular-distance cutoff should be positive.");
  }
  int next_param = 8;
  const auto is_option = [](const char* value) {
    return std::strcmp(value, "ion_field") == 0 || std::strcmp(value, "oho_angle") == 0 ||
      std::strcmp(value, "bead_diagnostic") == 0 ||
      std::strcmp(value, "local_environment") == 0 ||
      std::strcmp(value, "output") == 0 || std::strcmp(value, "output_level") == 0 ||
      std::strcmp(value, "snapshots") == 0;
  };
  if (next_param < num_param && !is_option(param[next_param])) {
    if (next_param + 1 >= num_param || is_option(param[next_param + 1])) {
      PRINT_INPUT_ERROR("proton tunneling O and H symbols must be provided as a pair.");
    }
    oxygen_symbol_ = param[next_param++];
    hydrogen_symbol_ = param[next_param++];
  }
  bool angle_seen = false;
  bool bead_diagnostic_seen = false;
  bool local_environment_seen = false;
  bool output_seen = false;
  bool output_level_seen = false;
  bool snapshots_seen = false;
  while (next_param < num_param) {
    if (std::strcmp(param[next_param], "oho_angle") == 0) {
      if (angle_seen || next_param + 1 >= num_param ||
          !is_valid_real(param[next_param + 1], &oho_angle_min_deg_)) {
        PRINT_INPUT_ERROR("oho_angle must be followed by one angle in degrees.");
      }
      angle_seen = true;
      next_param += 2;
    } else if (std::strcmp(param[next_param], "ion_field") == 0) {
      if (ion_field_enabled_ || next_param + 5 >= num_param) {
        PRINT_INPUT_ERROR(
          "ion_field must be followed by ion1_symbol ion1_charge ion2_symbol ion2_charge cutoff.");
      }
      ion_field_enabled_ = true;
      ion1_symbol_ = param[next_param + 1];
      if (!is_valid_real(param[next_param + 2], &ion1_charge_)) {
        PRINT_INPUT_ERROR("ion_field ion1 charge should be a number.");
      }
      ion2_symbol_ = param[next_param + 3];
      if (!is_valid_real(param[next_param + 4], &ion2_charge_)) {
        PRINT_INPUT_ERROR("ion_field ion2 charge should be a number.");
      }
      if (!is_valid_real(param[next_param + 5], &ion_field_cutoff_) || ion_field_cutoff_ <= 0.0) {
        PRINT_INPUT_ERROR("ion_field cutoff should be a positive number.");
      }
      next_param += 6;
    } else if (std::strcmp(param[next_param], "bead_diagnostic") == 0) {
      if (bead_diagnostic_seen) {
        PRINT_INPUT_ERROR("bead_diagnostic may only be specified once.");
      }
      bead_diagnostic_seen = true;
      bead_diagnostic_enabled_ = true;
      ++next_param;
      if (next_param < num_param && !is_option(param[next_param])) {
        if (next_param + 1 >= num_param || is_option(param[next_param + 1]) ||
            !is_valid_real(param[next_param], &bead_f_min_) ||
            !is_valid_real(param[next_param + 1], &bead_span_min_)) {
          PRINT_INPUT_ERROR(
            "bead_diagnostic must be followed by f_min span_min, optionally center_max centroid_max.");
        }
        next_param += 2;
        if (next_param < num_param && !is_option(param[next_param])) {
          if (next_param + 1 >= num_param || is_option(param[next_param + 1]) ||
              !is_valid_real(param[next_param], &bead_center_max_) ||
              !is_valid_real(param[next_param + 1], &bead_centroid_max_)) {
            PRINT_INPUT_ERROR(
              "bead_diagnostic optional thresholds require center_max and centroid_max.");
          }
          next_param += 2;
        }
      } else {
        bead_span_min_ = 2.0 * delta_cutoff_;
      }
    } else if (std::strcmp(param[next_param], "local_environment") == 0) {
      if (local_environment_seen || next_param + 4 >= num_param) {
        PRINT_INPUT_ERROR(
          "local_environment must be followed by ion1 cutoff ion2 cutoff H-Cl cutoff H-Cl angle.");
      }
      local_environment_seen = true;
      local_environment_enabled_ = true;
      if (!is_valid_real(param[next_param + 1], &local_ion1_cutoff_) ||
          local_ion1_cutoff_ <= 0.0 ||
          !is_valid_real(param[next_param + 2], &local_ion2_cutoff_) ||
          local_ion2_cutoff_ <= 0.0 ||
          !is_valid_real(param[next_param + 3], &local_hcl_cutoff_) ||
          local_hcl_cutoff_ <= 0.0 ||
          !is_valid_real(param[next_param + 4], &local_hcl_angle_min_deg_) ||
          local_hcl_angle_min_deg_ < 0.0 || local_hcl_angle_min_deg_ > 180.0) {
        PRINT_INPUT_ERROR(
          "local_environment requires positive ion/H-Cl cutoffs and an H-Cl angle between 0 and 180 degrees.");
      }
      next_param += 5;
    } else if (std::strcmp(param[next_param], "output") == 0) {
      if (output_seen || next_param + 1 >= num_param) {
        PRINT_INPUT_ERROR("output must be followed by text or netcdf settings.");
      }
      output_seen = true;
      if (std::strcmp(param[next_param + 1], "text") == 0) {
        output_format_ = OutputFormat::TEXT;
        next_param += 2;
      } else if (std::strcmp(param[next_param + 1], "netcdf") == 0) {
#ifndef USE_NETCDF
        PRINT_INPUT_ERROR(
          "output netcdf requires a GPUMD build with USE_NETCDF=1 and NetCDF4 support.");
#endif
        if (next_param + 2 >= num_param || is_option(param[next_param + 2]) ||
            std::strlen(param[next_param + 2]) == 0) {
          PRINT_INPUT_ERROR("output netcdf must be followed by a filename and optional level.");
        }
        output_format_ = OutputFormat::NETCDF;
        output_filename_ = param[next_param + 2];
        next_param += 3;
        if (next_param < num_param && !is_option(param[next_param])) {
          if (!is_valid_int(param[next_param], &compression_level_) ||
              compression_level_ < 0 || compression_level_ > 9) {
            PRINT_INPUT_ERROR("NetCDF compression level should be an integer from 0 to 9.");
          }
          ++next_param;
        }
      } else {
        PRINT_INPUT_ERROR("output format should be text or netcdf.");
      }
    } else if (std::strcmp(param[next_param], "output_level") == 0) {
      if (output_level_seen || next_param + 1 >= num_param) {
        PRINT_INPUT_ERROR("output_level must be followed by summary, events, or full.");
      }
      output_level_seen = true;
      if (std::strcmp(param[next_param + 1], "summary") == 0) {
        output_level_ = OutputLevel::SUMMARY;
      } else if (std::strcmp(param[next_param + 1], "events") == 0) {
        output_level_ = OutputLevel::EVENTS;
      } else if (std::strcmp(param[next_param + 1], "full") == 0) {
        output_level_ = OutputLevel::FULL;
      } else {
        PRINT_INPUT_ERROR("output_level should be summary, events, or full.");
      }
      next_param += 2;
    } else if (std::strcmp(param[next_param], "snapshots") == 0) {
      if (snapshots_seen || next_param + 1 >= num_param) {
        PRINT_INPUT_ERROR("snapshots must be followed by endpoints, best, or all.");
      }
      snapshots_seen = true;
      if (std::strcmp(param[next_param + 1], "endpoints") == 0) {
        snapshot_mode_ = SnapshotMode::ENDPOINTS;
      } else if (std::strcmp(param[next_param + 1], "best") == 0) {
        snapshot_mode_ = SnapshotMode::BEST;
      } else if (std::strcmp(param[next_param + 1], "all") == 0) {
        snapshot_mode_ = SnapshotMode::ALL;
      } else {
        PRINT_INPUT_ERROR("snapshots should be endpoints, best, or all.");
      }
      next_param += 2;
    } else {
      PRINT_INPUT_ERROR("unknown optional compute_proton_tunneling setting.");
    }
  }
  if (oho_angle_min_deg_ < 0.0 || oho_angle_min_deg_ > 180.0) {
    PRINT_INPUT_ERROR("oho_angle should be between 0 and 180 degrees.");
  }
  if (bead_diagnostic_enabled_ &&
      (bead_f_min_ <= 0.0 || bead_f_min_ > 0.5 || bead_span_min_ <= 0.0 ||
       bead_center_max_ < 0.0 || bead_center_max_ > 1.0 || bead_centroid_max_ < 0.0 ||
       2.0 * bead_f_min_ + bead_center_max_ > 1.0)) {
    PRINT_INPUT_ERROR(
      "bead_diagnostic requires 0 < f_min <= 0.5, span_min > 0, 0 <= center_max <= 1, "
      "centroid_max >= 0, and 2*f_min + center_max <= 1.");
  }
  if (oxygen_symbol_.empty() || hydrogen_symbol_.empty() || oxygen_symbol_ == hydrogen_symbol_) {
    PRINT_INPUT_ERROR("proton tunneling O and H symbols should be different and non-empty.");
  }
  if (ion_field_enabled_ &&
      (ion1_symbol_.empty() || ion2_symbol_.empty() || ion1_symbol_ == ion2_symbol_ ||
       ion1_symbol_ == oxygen_symbol_ || ion1_symbol_ == hydrogen_symbol_ ||
       ion2_symbol_ == oxygen_symbol_ || ion2_symbol_ == hydrogen_symbol_)) {
    PRINT_INPUT_ERROR("ion_field species should be distinct from each other and from O/H.");
  }

  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_atom_symbol[i] == oxygen_symbol_)
      oxygen_indices_.push_back(i);
    if (atom.cpu_atom_symbol[i] == hydrogen_symbol_)
      hydrogen_indices_.push_back(i);
    if ((ion_field_enabled_ || local_environment_enabled_) && atom.cpu_atom_symbol[i] == ion1_symbol_)
      ion1_indices_.push_back(i);
    if ((ion_field_enabled_ || local_environment_enabled_) && atom.cpu_atom_symbol[i] == ion2_symbol_)
      ion2_indices_.push_back(i);
  }
  if (oxygen_indices_.empty() || hydrogen_indices_.empty()) {
    PRINT_INPUT_ERROR("compute_proton_tunneling could not find the requested O and H species.");
  }
  if (ion_field_enabled_ && (ion1_indices_.empty() || ion2_indices_.empty())) {
    PRINT_INPUT_ERROR("ion_field could not find both requested ion species.");
  }
  if (output_format_ != OutputFormat::NETCDF &&
      (output_level_seen || snapshots_seen)) {
    PRINT_INPUT_ERROR("output_level and snapshots require output netcdf.");
  }

  printf("    sample interval is %d steps.\n", sample_interval_);
  printf("    window size is %d sampled frames.\n", window_samples_);
  printf("    delta cutoff is %.6f Angstrom.\n", delta_cutoff_);
  printf("    state hold time is %d sampled frames.\n", hold_samples_);
  printf("    O-O range is %.6f to %.6f Angstrom.\n", dOO_min_, dOO_max_);
  printf("    perpendicular-distance cutoff is %.6f Angstrom.\n", rperp_max_);
  printf("    minimum O-H-O angle is %.6f degrees.\n", oho_angle_min_deg_);
  printf("    using oxygen symbol %s and hydrogen symbol %s.\n",
    oxygen_symbol_.c_str(), hydrogen_symbol_.c_str());
  if (output_format_ == OutputFormat::NETCDF) {
    const char* level = output_level_ == OutputLevel::SUMMARY ? "summary" :
      (output_level_ == OutputLevel::EVENTS ? "events" : "full");
    const char* snapshots = snapshot_mode_ == SnapshotMode::ENDPOINTS ? "endpoints" :
      (snapshot_mode_ == SnapshotMode::BEST ? "best" : "all");
    printf("    compressed NetCDF output is %s, deflate level %d, output level %s, snapshots %s.\n",
      output_filename_.c_str(), compression_level_, level, snapshots);
  }
  if (bead_diagnostic_enabled_) {
    printf("    strict bead diagnostic uses f_min %.6f, center_max %.6f, centroid_max %.6f, "
           "and span_min %.6f Angstrom.\n",
      bead_f_min_, bead_center_max_, bead_centroid_max_, bead_span_min_);
  }
  if (ion_field_enabled_) {
    printf("    nominal ion field uses %s charge %.6f and %s charge %.6f within %.6f Angstrom.\n",
      ion1_symbol_.c_str(), ion1_charge_, ion2_symbol_.c_str(), ion2_charge_, ion_field_cutoff_);
  }
  if (local_environment_enabled_) {
    printf("    local environment uses %s cutoff %.6f, %s cutoff %.6f, H-%s cutoff %.6f, "
           "and H-%s angle %.6f degrees.\n",
      ion1_symbol_.c_str(), local_ion1_cutoff_, ion2_symbol_.c_str(), local_ion2_cutoff_,
      ion2_symbol_.c_str(), local_hcl_cutoff_, ion2_symbol_.c_str(), local_hcl_angle_min_deg_);
    if (ion1_indices_.empty() || ion2_indices_.empty())
      printf("    local environment warning: one or both configured ion species are absent; "
             "missing descriptors will be reported as nan/0.\n");
  }
}

void Proton_Tunneling::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  (void)number_of_steps;
  (void)integrate;
  (void)group;
  (void)force;

  number_of_atoms_ = atom.number_of_atoms;
  time_step_ = time_step;
  cpu_position_.resize(number_of_atoms_ * 3);
  bead_positions_cached_ = false;
  cached_number_of_beads_ = 0;
  observer_frame_count_ = 0;
  bead_probe_frame_count_ = 0;
  bead_d2h_copy_count_ = 0;
  bead_bytes_copied_ = 0;
  bead_copy_wall_time_ = 0.0;
  bead_analysis_wall_time_ = 0.0;
  total_observer_wall_time_ = 0.0;
  geometry_kernel_wall_time_ = 0.0;
  geometry_D2H_wall_time_ = 0.0;
  state_machine_wall_time_ = 0.0;
  local_environment_copy_count_ = 0;
  local_environment_bytes_copied_ = 0;
  local_environment_kernel_wall_time_ = 0.0;
  local_environment_D2H_wall_time_ = 0.0;
  local_environment_host_analysis_wall_time_ = 0.0;
  oxygen_local_index_.assign(number_of_atoms_, -1);
  oxygen_shell_neighbors_.resize(oxygen_indices_.size());
  hydrogen_count_.resize(number_of_atoms_, 0);
  previous_hydrogen_count_.resize(number_of_atoms_, 0);
  event_hydrogen_count_.resize(number_of_atoms_, 0);
  frame_cause_event_ids_.resize(number_of_atoms_, -1);
  frame_geometries_.resize(hydrogen_indices_.size());
  hydrogen_states_.resize(hydrogen_indices_.size());
  window_assignment_ambiguous_count_ = 0;
  window_pair_conflict_count_ = 0;
  defect_state_initialized_ = false;
  attempt_records_.clear();
  defect_records_.clear();
  window_records_.clear();
  edge_window_records_.clear();
  local_environment_window_records_.clear();
  attempt_records_.reserve(1024);
  defect_records_.reserve(1024);
  window_records_.reserve(64);
  edge_window_records_.reserve(1024);
  local_environment_window_records_.reserve(64);
  atom.position_per_atom.copy_to_host(cpu_position_.data());
  build_oxygen_shell(box);
  initialize_geometry_gpu();
  CHECK(gpuEventCreate(&geometry_kernel_start_event_));
  CHECK(gpuEventCreate(&geometry_kernel_end_event_));
  if (local_environment_enabled_) {
    CHECK(gpuEventCreate(&local_environment_kernel_start_event_));
    CHECK(gpuEventCreate(&local_environment_kernel_end_event_));
  }
  initialized_ = true;
}

void Proton_Tunneling::build_oxygen_shell(const Box& box)
{
  // ponytail: build the initial O shell once with O(N_O^2) CPU distances; use a shared
  // neighbor list if this observer is later applied to large systems.
  oxygen_shell_neighbors_.assign(oxygen_indices_.size(), std::vector<int>());
  std::fill(oxygen_local_index_.begin(), oxygen_local_index_.end(), -1);
  for (size_t i = 0; i < oxygen_indices_.size(); ++i)
    oxygen_local_index_[oxygen_indices_[i]] = static_cast<int>(i);

  for (size_t i = 0; i < oxygen_indices_.size(); ++i) {
    const int oxygen_i = oxygen_indices_[i];
    std::vector<std::pair<double, int>> distances;
    distances.reserve(oxygen_indices_.size() - 1);
    for (size_t j = 0; j < oxygen_indices_.size(); ++j) {
      if (i == j)
        continue;
      const int oxygen_j = oxygen_indices_[j];
      double dx = cpu_position_[oxygen_j] - cpu_position_[oxygen_i];
      double dy = cpu_position_[oxygen_j + number_of_atoms_] -
        cpu_position_[oxygen_i + number_of_atoms_];
      double dz = cpu_position_[oxygen_j + 2 * number_of_atoms_] -
        cpu_position_[oxygen_i + 2 * number_of_atoms_];
      apply_mic(box, dx, dy, dz);
      distances.emplace_back(dx * dx + dy * dy + dz * dz, oxygen_j);
    }
    std::sort(distances.begin(), distances.end());
    const int shell_size = std::min(oxygen_shell_k_, static_cast<int>(distances.size()));
    oxygen_shell_neighbors_[i].reserve(shell_size);
    for (int j = 0; j < shell_size; ++j)
      oxygen_shell_neighbors_[i].push_back(distances[j].second);
  }

  oxygen_shell_offsets_cpu_.assign(oxygen_indices_.size() + 1, 0);
  oxygen_shell_neighbors_cpu_.clear();
  for (size_t i = 0; i < oxygen_shell_neighbors_.size(); ++i) {
    oxygen_shell_neighbors_cpu_.insert(
      oxygen_shell_neighbors_cpu_.end(),
      oxygen_shell_neighbors_[i].begin(),
      oxygen_shell_neighbors_[i].end());
    oxygen_shell_offsets_cpu_[i + 1] =
      static_cast<int>(oxygen_shell_neighbors_cpu_.size());
  }
}

void Proton_Tunneling::initialize_geometry_gpu()
{
  oxygen_indices_gpu_.resize(oxygen_indices_.size());
  oxygen_indices_gpu_.copy_from_host(oxygen_indices_.data());
  hydrogen_indices_gpu_.resize(hydrogen_indices_.size());
  hydrogen_indices_gpu_.copy_from_host(hydrogen_indices_.data());
  oxygen_local_index_gpu_.resize(oxygen_local_index_.size());
  oxygen_local_index_gpu_.copy_from_host(oxygen_local_index_.data());
  oxygen_shell_offsets_gpu_.resize(oxygen_shell_offsets_cpu_.size());
  oxygen_shell_offsets_gpu_.copy_from_host(oxygen_shell_offsets_cpu_.data());
  if (!oxygen_shell_neighbors_cpu_.empty()) {
    oxygen_shell_neighbors_gpu_.resize(oxygen_shell_neighbors_cpu_.size());
    oxygen_shell_neighbors_gpu_.copy_from_host(oxygen_shell_neighbors_cpu_.data());
  }
  if (ion_field_enabled_ || local_environment_enabled_) {
    ion1_indices_gpu_.resize(ion1_indices_.size());
    if (!ion1_indices_.empty())
      ion1_indices_gpu_.copy_from_host(ion1_indices_.data());
    ion2_indices_gpu_.resize(ion2_indices_.size());
    if (!ion2_indices_.empty())
      ion2_indices_gpu_.copy_from_host(ion2_indices_.data());
  }
  frame_geometries_gpu_.resize(frame_geometries_.size());
  cpu_geometry_gpu_.resize(frame_geometries_.size());
  if (local_environment_enabled_) {
    frame_local_environments_gpu_.resize(frame_geometries_.size());
    cpu_local_environments_gpu_.resize(frame_geometries_.size());
    frame_local_environments_.resize(frame_geometries_.size());
  }
}

void Proton_Tunneling::compute_geometry_gpu(const Box& box, Atom& atom)
{
  const int number_of_hydrogens = static_cast<int>(hydrogen_indices_.size());
  const auto total_start = std::chrono::high_resolution_clock::now();
  CHECK(gpuEventRecord(geometry_kernel_start_event_, 0));
  gpu_find_proton_geometry<<<number_of_hydrogens, 128>>>(
    atom.position_per_atom.data(),
    number_of_atoms_,
    oxygen_indices_gpu_.data(),
    static_cast<int>(oxygen_indices_.size()),
    hydrogen_indices_gpu_.data(),
    oxygen_local_index_gpu_.data(),
    oxygen_shell_offsets_gpu_.data(),
    oxygen_shell_neighbors_cpu_.empty() ? nullptr : oxygen_shell_neighbors_gpu_.data(),
    ion_field_enabled_ ? ion1_indices_gpu_.data() : nullptr,
    ion_field_enabled_ ? static_cast<int>(ion1_indices_.size()) : 0,
    ion1_charge_,
    ion_field_enabled_ ? ion2_indices_gpu_.data() : nullptr,
    ion_field_enabled_ ? static_cast<int>(ion2_indices_.size()) : 0,
    ion2_charge_,
    box,
    dOO_min_,
    dOO_max_,
    rperp_max_,
    std::cos(oho_angle_min_deg_ * 0.017453292519943295),
    assignment_path_excess_weight_,
    assignment_score_gap_min_,
    ion_field_enabled_ ? 1 : 0,
    ion_field_cutoff_,
    frame_geometries_gpu_.data());
  GPU_CHECK_KERNEL
  CHECK(gpuEventRecord(geometry_kernel_end_event_, 0));
  frame_geometries_gpu_.copy_to_host(cpu_geometry_gpu_.data());
  const auto total_end = std::chrono::high_resolution_clock::now();

  float kernel_time_ms = 0.0f;
  CHECK(gpuEventElapsedTime(
    &kernel_time_ms,
    geometry_kernel_start_event_,
    geometry_kernel_end_event_));
  const double total_time =
    std::chrono::duration<double>(total_end - total_start).count();
  const double kernel_time = static_cast<double>(kernel_time_ms) * 1.0e-3;
  geometry_kernel_wall_time_ += kernel_time;
  geometry_D2H_wall_time_ += std::max(0.0, total_time - kernel_time);

  for (size_t i = 0; i < frame_geometries_.size(); ++i) {
    const GeometryResultGPU& source = cpu_geometry_gpu_[i];
    GeometryResult& target = frame_geometries_[i];
    target = GeometryResult();
    target.valid = source.valid != 0;
    target.assignment_ambiguous = source.assignment_ambiguous != 0;
    target.pair_conflict = source.pair_conflict != 0;
    target.nearest_oxygen = source.nearest_oxygen;
    target.oxygen_low = source.oxygen_low;
    target.oxygen_high = source.oxygen_high;
    target.candidate_count = source.candidate_count;
    target.delta = source.delta;
    target.dOO = source.dOO;
    target.rperp = source.rperp;
    target.path_excess = source.path_excess;
    target.assignment_score = source.assignment_score;
    target.second_assignment_score = source.second_assignment_score;
    target.assignment_score_gap = source.assignment_score_gap;
    target.low_to_high_dx = source.low_to_high_dx;
    target.low_to_high_dy = source.low_to_high_dy;
    target.low_to_high_dz = source.low_to_high_dz;
    target.E_ion_nominal_parallel = source.E_ion_nominal_parallel;
    target.delta_phi_ion = source.delta_phi_ion;
    target.nearest_ion_id = source.nearest_ion_id;
    target.nearest_ion_distance = source.nearest_ion_distance;
    target.nearest_ion1_distance = source.nearest_ion1_distance;
    target.nearest_ion2_distance = source.nearest_ion2_distance;
    target.nearest_ion1_to_low = source.nearest_ion1_to_low;
    target.nearest_ion1_to_high = source.nearest_ion1_to_high;
    target.nearest_ion2_to_low = source.nearest_ion2_to_low;
    target.nearest_ion2_to_high = source.nearest_ion2_to_high;
  }
}

void Proton_Tunneling::compute_local_environment_gpu(const Box& box, Atom& atom)
{
  if (!local_environment_enabled_)
    return;
  const int number_of_hydrogens = static_cast<int>(hydrogen_indices_.size());
  const int block_size = 128;
  const int grid_size = (number_of_hydrogens + block_size - 1) / block_size;
  const auto total_start = std::chrono::high_resolution_clock::now();
  CHECK(gpuEventRecord(local_environment_kernel_start_event_, 0));
  gpu_compute_proton_local_environment<<<grid_size, block_size>>>(
    atom.position_per_atom.data(),
    number_of_atoms_,
    number_of_hydrogens,
    hydrogen_indices_gpu_.data(),
    frame_geometries_gpu_.data(),
    ion1_indices_.empty() ? nullptr : ion1_indices_gpu_.data(),
    static_cast<int>(ion1_indices_.size()),
    ion2_indices_.empty() ? nullptr : ion2_indices_gpu_.data(),
    static_cast<int>(ion2_indices_.size()),
    box,
    local_ion1_cutoff_,
    local_ion2_cutoff_,
    local_hcl_cutoff_,
    local_hcl_angle_min_deg_,
    frame_local_environments_gpu_.data());
  GPU_CHECK_KERNEL
  CHECK(gpuEventRecord(local_environment_kernel_end_event_, 0));
  frame_local_environments_gpu_.copy_to_host(cpu_local_environments_gpu_.data());
  const auto copy_end = std::chrono::high_resolution_clock::now();
  float kernel_time_ms = 0.0f;
  CHECK(gpuEventElapsedTime(
    &kernel_time_ms,
    local_environment_kernel_start_event_,
    local_environment_kernel_end_event_));
  const double total_time =
    std::chrono::duration<double>(copy_end - total_start).count();
  const double kernel_time = static_cast<double>(kernel_time_ms) * 1.0e-3;
  local_environment_kernel_wall_time_ += kernel_time;
  local_environment_D2H_wall_time_ += std::max(0.0, total_time - kernel_time);
  ++local_environment_copy_count_;
  local_environment_bytes_copied_ += static_cast<unsigned long long>(
    frame_local_environments_.size() * sizeof(LocalEnvironmentGPU));
  const auto host_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < frame_local_environments_.size(); ++i) {
    const LocalEnvironmentGPU& source = cpu_local_environments_gpu_[i];
    LocalEnvironment& target = frame_local_environments_[i];
    target = LocalEnvironment();
    target.valid = source.valid != 0 && frame_geometries_[i].valid;
    target.rOH_low = source.rOH_low;
    target.rOH_high = source.rOH_high;
    target.oho_angle = source.oho_angle;
    target.dOO = source.dOO;
    target.rperp = source.rperp;
    target.path_excess = source.path_excess;
    target.ion1_low_d1 = source.ion1_low_d1;
    target.ion1_low_d2 = source.ion1_low_d2;
    target.ion1_low_d3 = source.ion1_low_d3;
    target.ion1_high_d1 = source.ion1_high_d1;
    target.ion1_high_d2 = source.ion1_high_d2;
    target.ion1_high_d3 = source.ion1_high_d3;
    target.ion2_low_d1 = source.ion2_low_d1;
    target.ion2_low_d2 = source.ion2_low_d2;
    target.ion2_low_d3 = source.ion2_low_d3;
    target.ion2_high_d1 = source.ion2_high_d1;
    target.ion2_high_d2 = source.ion2_high_d2;
    target.ion2_high_d3 = source.ion2_high_d3;
    target.coord_ion1_low = source.coord_ion1_low;
    target.coord_ion1_high = source.coord_ion1_high;
    target.coord_ion2_low = source.coord_ion2_low;
    target.coord_ion2_high = source.coord_ion2_high;
    target.delta_d_ion1 = source.delta_d_ion1;
    target.delta_d_ion2 = source.delta_d_ion2;
    target.nearest_ion2_to_H = source.nearest_ion2_to_H;
    target.angle_Olow_H_ion2 = source.angle_Olow_H_ion2;
    target.angle_Ohigh_H_ion2 = source.angle_Ohigh_H_ion2;
    target.hcl_like_low = source.hcl_like_low;
    target.hcl_like_high = source.hcl_like_high;
    target.E_parallel = source.E_parallel;
    target.delta_phi_ion = source.delta_phi_ion;
  }
  const auto host_end = std::chrono::high_resolution_clock::now();
  local_environment_host_analysis_wall_time_ +=
    std::chrono::duration<double>(host_end - host_start).count();
}

void Proton_Tunneling::assign_local_environment_topology()
{
  if (!local_environment_enabled_)
    return;
  std::vector<int> donor_count(number_of_atoms_, 0);
  std::vector<int> acceptor_count(number_of_atoms_, 0);
  for (size_t h_index = 0; h_index < frame_geometries_.size(); ++h_index) {
    const GeometryResult& geometry = frame_geometries_[h_index];
    LocalEnvironment& environment = frame_local_environments_[h_index];
    if (!environment.valid)
      continue;
    environment.nH_low = hydrogen_count_[geometry.oxygen_low];
    environment.nH_high = hydrogen_count_[geometry.oxygen_high];
    const int state = classify_delta(geometry.delta);
    if (state < 0) {
      ++donor_count[geometry.oxygen_low];
      ++acceptor_count[geometry.oxygen_high];
    } else if (state > 0) {
      ++donor_count[geometry.oxygen_high];
      ++acceptor_count[geometry.oxygen_low];
    }
  }
  for (size_t h_index = 0; h_index < frame_geometries_.size(); ++h_index) {
    const GeometryResult& geometry = frame_geometries_[h_index];
    LocalEnvironment& environment = frame_local_environments_[h_index];
    if (!environment.valid)
      continue;
    environment.donor_edges_low = donor_count[geometry.oxygen_low];
    environment.donor_edges_high = donor_count[geometry.oxygen_high];
    environment.acceptor_edges_low = acceptor_count[geometry.oxygen_low];
    environment.acceptor_edges_high = acceptor_count[geometry.oxygen_high];
  }
}

void Proton_Tunneling::release_geometry_timing_events()
{
  if (geometry_kernel_start_event_ != nullptr) {
    CHECK(gpuEventDestroy(geometry_kernel_start_event_));
    geometry_kernel_start_event_ = nullptr;
  }
  if (geometry_kernel_end_event_ != nullptr) {
    CHECK(gpuEventDestroy(geometry_kernel_end_event_));
    geometry_kernel_end_event_ = nullptr;
  }
  if (local_environment_kernel_start_event_ != nullptr) {
    CHECK(gpuEventDestroy(local_environment_kernel_start_event_));
    local_environment_kernel_start_event_ = nullptr;
  }
  if (local_environment_kernel_end_event_ != nullptr) {
    CHECK(gpuEventDestroy(local_environment_kernel_end_event_));
    local_environment_kernel_end_event_ = nullptr;
  }
}

void Proton_Tunneling::compute_ion_field(const Box& box, GeometryResult& geometry) const
{
  if (!ion_field_enabled_)
    return;

  const double ox = geometry.low_to_high_dx;
  const double oy = geometry.low_to_high_dy;
  const double oz = geometry.low_to_high_dz;
  const double inverse_dOO = 1.0 / geometry.dOO;
  const double ex = ox * inverse_dOO;
  const double ey = oy * inverse_dOO;
  const double ez = oz * inverse_dOO;
  geometry.delta_phi_ion = 0.0;
  geometry.nearest_ion_distance = std::numeric_limits<double>::max();
  geometry.nearest_ion1_distance = std::numeric_limits<double>::max();
  geometry.nearest_ion2_distance = std::numeric_limits<double>::max();
  geometry.nearest_ion1_to_low = std::numeric_limits<double>::max();
  geometry.nearest_ion1_to_high = std::numeric_limits<double>::max();
  geometry.nearest_ion2_to_low = std::numeric_limits<double>::max();
  geometry.nearest_ion2_to_high = std::numeric_limits<double>::max();
  auto accumulate_ion_field = [&](const std::vector<int>& ions, const double charge,
                                  double& nearest_species_distance,
                                  double& nearest_species_to_low,
                                  double& nearest_species_to_high) {
    for (const int ion : ions) {
      double ion_dx = cpu_position_[ion] - cpu_position_[geometry.oxygen_low];
      double ion_dy = cpu_position_[ion + number_of_atoms_] -
        cpu_position_[geometry.oxygen_low + number_of_atoms_];
      double ion_dz = cpu_position_[ion + 2 * number_of_atoms_] -
        cpu_position_[geometry.oxygen_low + 2 * number_of_atoms_];
      apply_mic(box, ion_dx, ion_dy, ion_dz);
      double midpoint_to_ion_x = 0.5 * ox - ion_dx;
      double midpoint_to_ion_y = 0.5 * oy - ion_dy;
      double midpoint_to_ion_z = 0.5 * oz - ion_dz;
      apply_mic(box, midpoint_to_ion_x, midpoint_to_ion_y, midpoint_to_ion_z);
      double ion_to_high_x = cpu_position_[ion] - cpu_position_[geometry.oxygen_high];
      double ion_to_high_y = cpu_position_[ion + number_of_atoms_] -
        cpu_position_[geometry.oxygen_high + number_of_atoms_];
      double ion_to_high_z = cpu_position_[ion + 2 * number_of_atoms_] -
        cpu_position_[geometry.oxygen_high + 2 * number_of_atoms_];
      apply_mic(box, ion_to_high_x, ion_to_high_y, ion_to_high_z);
      const double low_distance_square = ion_dx * ion_dx + ion_dy * ion_dy + ion_dz * ion_dz;
      const double high_distance_square = ion_to_high_x * ion_to_high_x +
        ion_to_high_y * ion_to_high_y + ion_to_high_z * ion_to_high_z;
      const double low_distance = std::sqrt(low_distance_square);
      const double high_distance = std::sqrt(high_distance_square);
      const double distance_square = midpoint_to_ion_x * midpoint_to_ion_x +
        midpoint_to_ion_y * midpoint_to_ion_y + midpoint_to_ion_z * midpoint_to_ion_z;
      const double distance = std::sqrt(distance_square);
      nearest_species_distance = std::min(nearest_species_distance, distance);
      nearest_species_to_low = std::min(nearest_species_to_low, low_distance);
      nearest_species_to_high = std::min(nearest_species_to_high, high_distance);
      if (distance < geometry.nearest_ion_distance) {
        geometry.nearest_ion_distance = distance;
        geometry.nearest_ion_id = ion;
      }
      if (distance <= ion_field_cutoff_ && distance > 1.0e-12) {
        geometry.E_ion_nominal_parallel += K_C * charge *
          (midpoint_to_ion_x * ex + midpoint_to_ion_y * ey + midpoint_to_ion_z * ez) /
          (distance_square * distance);
        if (low_distance > 1.0e-12 && high_distance > 1.0e-12)
          geometry.delta_phi_ion += K_C * charge *
            (1.0 / high_distance - 1.0 / low_distance);
      }
    }
  };
  accumulate_ion_field(
    ion1_indices_, ion1_charge_, geometry.nearest_ion1_distance,
    geometry.nearest_ion1_to_low, geometry.nearest_ion1_to_high);
  accumulate_ion_field(
    ion2_indices_, ion2_charge_, geometry.nearest_ion2_distance,
    geometry.nearest_ion2_to_low, geometry.nearest_ion2_to_high);
}

bool Proton_Tunneling::ensure_bead_positions(Atom& atom)
{
  if (!bead_diagnostic_enabled_ || atom.number_of_beads <= 1)
    return true;
  if (atom.position_beads.size() < static_cast<size_t>(atom.number_of_beads))
    return false;
  if (bead_positions_cached_ && cached_number_of_beads_ == atom.number_of_beads)
    return true;

  const size_t position_size = static_cast<size_t>(number_of_atoms_) * 3;
  for (int bead = 0; bead < atom.number_of_beads; ++bead) {
    if (atom.position_beads[bead].size() != position_size)
      return false;
  }

  bool rebuild_pointer_array = bead_position_ptrs_cpu_.size() !=
    static_cast<size_t>(atom.number_of_beads);
  if (!rebuild_pointer_array) {
    for (int bead = 0; bead < atom.number_of_beads; ++bead) {
      if (bead_position_ptrs_cpu_[bead] != atom.position_beads[bead].data()) {
        rebuild_pointer_array = true;
        break;
      }
    }
  }
  if (rebuild_pointer_array) {
    bead_position_ptrs_cpu_.resize(atom.number_of_beads);
    for (int bead = 0; bead < atom.number_of_beads; ++bead)
      bead_position_ptrs_cpu_[bead] = atom.position_beads[bead].data();
    bead_position_ptrs_gpu_.resize(atom.number_of_beads);
    bead_position_ptrs_gpu_.copy_from_host(bead_position_ptrs_cpu_.data());
  }

  const size_t packed_size = position_size * static_cast<size_t>(atom.number_of_beads);
  cpu_position_beads_.resize(position_size * static_cast<size_t>(atom.number_of_beads));
  if (bead_position_staging_gpu_.size() != packed_size)
    bead_position_staging_gpu_.resize(packed_size);

  // ponytail: one small packing kernel plus one synchronous D2H copy keeps the
  // observer simple while removing the per-bead synchronization bubbles.
  const auto copy_start = std::chrono::high_resolution_clock::now();
  const int block_size = 128;
  const int grid_x = (static_cast<int>(position_size) + block_size - 1) / block_size;
  pack_bead_positions<<<dim3(grid_x, atom.number_of_beads), block_size>>>(
    static_cast<int>(position_size),
    atom.number_of_beads,
    bead_position_ptrs_gpu_.data(),
    bead_position_staging_gpu_.data());
  GPU_CHECK_KERNEL
  bead_position_staging_gpu_.copy_to_host(cpu_position_beads_.data());
  const auto copy_end = std::chrono::high_resolution_clock::now();
  bead_copy_wall_time_ += std::chrono::duration<double>(copy_end - copy_start).count();
  ++bead_d2h_copy_count_;
  bead_bytes_copied_ += static_cast<unsigned long long>(packed_size * sizeof(double));

  cached_number_of_beads_ = atom.number_of_beads;
  bead_positions_cached_ = true;
  return true;
}

bool Proton_Tunneling::is_better_delocalization_diagnostic(
  const BeadDiagnostic& candidate,
  const BeadDiagnostic& current) const
{
  if (!candidate.valid)
    return false;
  if (!current.valid)
    return true;
  const double candidate_well_occupancy = std::min(candidate.f_minus, candidate.f_plus);
  const double current_well_occupancy = std::min(current.f_minus, current.f_plus);
  if (candidate_well_occupancy != current_well_occupancy)
    return candidate_well_occupancy > current_well_occupancy;
  if (candidate.f_zero != current.f_zero)
    return candidate.f_zero < current.f_zero;
  if (candidate.simple_two_domain_path != current.simple_two_domain_path)
    return candidate.simple_two_domain_path > current.simple_two_domain_path;
  if (candidate.robust_span != current.robust_span)
    return candidate.robust_span > current.robust_span;
  return std::abs(candidate.delta_centroid) < std::abs(current.delta_centroid);
}

Proton_Tunneling::QuantumCharacter Proton_Tunneling::classify_quantum_character(
  const BeadDiagnostic& diagnostic) const
{
  if (diagnostic.num_beads <= 1)
    return QuantumCharacter::CLASSICAL_ONLY;
  if (!diagnostic.valid)
    return QuantumCharacter::AMBIGUOUS;

  if (diagnostic.barrier_centered != 0)
    return QuantumCharacter::BARRIER_CENTERED_TUNNELING_LIKE;
  if (diagnostic.simple_two_domain_path != 0)
    return QuantumCharacter::TWO_WELL_DELOCALIZED;
  if (diagnostic.multi_kink_or_multi_domain != 0)
    return QuantumCharacter::MULTI_KINK_OR_MULTI_DOMAIN;

  const int dominant_count = std::max(
    diagnostic.n_zero, std::max(diagnostic.n_minus, diagnostic.n_plus));
  if (diagnostic.kink_count == 0 &&
      static_cast<double>(dominant_count) / diagnostic.num_beads >= 1.0 - bead_f_min_)
    return QuantumCharacter::COMPACT_SINGLE_DOMAIN;

  return QuantumCharacter::AMBIGUOUS;
}

const char* Proton_Tunneling::quantum_character_name(const QuantumCharacter character) const
{
  switch (character) {
  case QuantumCharacter::CLASSICAL_ONLY:
    return "classical_only";
  case QuantumCharacter::TWO_WELL_DELOCALIZED:
    return "two_well_delocalized";
  case QuantumCharacter::BARRIER_CENTERED_TUNNELING_LIKE:
    return "barrier_centered_tunneling_like";
  case QuantumCharacter::COMPACT_SINGLE_DOMAIN:
    return "compact_single_domain";
  case QuantumCharacter::MULTI_KINK_OR_MULTI_DOMAIN:
    return "multi_kink_or_multi_domain";
  case QuantumCharacter::AMBIGUOUS:
    return "ambiguous";
  }
  return "ambiguous";
}

bool Proton_Tunneling::evaluate_bead_diagnostic(
  Atom& atom,
  const Box& box,
  const int hydrogen,
  const GeometryResult& geometry,
  const double probe_time_fs,
  BeadDiagnostic& diagnostic)
{
  diagnostic = BeadDiagnostic();
  diagnostic.probe_time_fs = probe_time_fs;
  diagnostic.delta_centroid = geometry.delta;
  diagnostic.num_beads = std::max(1, atom.number_of_beads);

  if (diagnostic.num_beads <= 1) {
    const auto analysis_start = std::chrono::high_resolution_clock::now();
    diagnostic.valid = true;
    diagnostic.mean_delta = geometry.delta;
    diagnostic.sigma_delta = 0.0;
    diagnostic.delta_min = geometry.delta;
    diagnostic.delta_max = geometry.delta;
    diagnostic.span = 0.0;
    const int state = classify_delta(geometry.delta);
    if (state < 0)
      diagnostic.n_minus = 1;
    else if (state > 0)
      diagnostic.n_plus = 1;
    else
      diagnostic.n_zero = 1;
    diagnostic.f_minus = static_cast<double>(diagnostic.n_minus);
    diagnostic.f_zero = static_cast<double>(diagnostic.n_zero);
    diagnostic.f_plus = static_cast<double>(diagnostic.n_plus);
    diagnostic.center_domain_count = (state == 0) ? 1 : 0;
    diagnostic.total_state_domain_count = 1;
    diagnostic.channel_valid_count = 1;
    diagnostic.f_channel_valid = 1.0;
    diagnostic.delta_q20 = geometry.delta;
    diagnostic.delta_q80 = geometry.delta;
    diagnostic.character = classify_quantum_character(diagnostic);
    const auto analysis_end = std::chrono::high_resolution_clock::now();
    bead_analysis_wall_time_ +=
      std::chrono::duration<double>(analysis_end - analysis_start).count();
    return true;
  }

  if (!ensure_bead_positions(atom)) {
    diagnostic.character = classify_quantum_character(diagnostic);
    return false;
  }

  const auto analysis_start = std::chrono::high_resolution_clock::now();
  const size_t position_size = static_cast<size_t>(number_of_atoms_) * 3;
  const double angle_cosine_limit = std::cos(oho_angle_min_deg_ * 0.017453292519943295);
  const auto bead_geometry = [&](const int bead, double& delta) {
    const double* position = cpu_position_beads_.data() + position_size * bead;
    double low_x = position[geometry.oxygen_low] - position[hydrogen];
    double low_y = position[geometry.oxygen_low + number_of_atoms_] -
      position[hydrogen + number_of_atoms_];
    double low_z = position[geometry.oxygen_low + 2 * number_of_atoms_] -
      position[hydrogen + 2 * number_of_atoms_];
    double high_x = position[geometry.oxygen_high] - position[hydrogen];
    double high_y = position[geometry.oxygen_high + number_of_atoms_] -
      position[hydrogen + number_of_atoms_];
    double high_z = position[geometry.oxygen_high + 2 * number_of_atoms_] -
      position[hydrogen + 2 * number_of_atoms_];
    apply_mic(box, low_x, low_y, low_z);
    apply_mic(box, high_x, high_y, high_z);
    const double low_distance = std::sqrt(low_x * low_x + low_y * low_y + low_z * low_z);
    const double high_distance = std::sqrt(high_x * high_x + high_y * high_y + high_z * high_z);
    if (low_distance > 0.0 && high_distance > 0.0)
      delta = low_distance - high_distance;
    double ox = position[geometry.oxygen_high] - position[geometry.oxygen_low];
    double oy = position[geometry.oxygen_high + number_of_atoms_] -
      position[geometry.oxygen_low + number_of_atoms_];
    double oz = position[geometry.oxygen_high + 2 * number_of_atoms_] -
      position[geometry.oxygen_low + 2 * number_of_atoms_];
    apply_mic(box, ox, oy, oz);
    const double dOO_square = ox * ox + oy * oy + oz * oz;
    if (dOO_square < dOO_min_ * dOO_min_ || dOO_square > dOO_max_ * dOO_max_)
      return false;
    double hx_from_low = position[hydrogen] - position[geometry.oxygen_low];
    double hy_from_low = position[hydrogen + number_of_atoms_] -
      position[geometry.oxygen_low + number_of_atoms_];
    double hz_from_low = position[hydrogen + 2 * number_of_atoms_] -
      position[geometry.oxygen_low + 2 * number_of_atoms_];
    apply_mic(box, hx_from_low, hy_from_low, hz_from_low);
    const double projection = (hx_from_low * ox + hy_from_low * oy + hz_from_low * oz) /
      dOO_square;
    if (projection < -1.0e-10 || projection > 1.0 + 1.0e-10)
      return false;
    const double px = hx_from_low - projection * ox;
    const double py = hy_from_low - projection * oy;
    const double pz = hz_from_low - projection * oz;
    const double rperp = std::sqrt(std::max(0.0, px * px + py * py + pz * pz));
    if (rperp > rperp_max_)
      return false;
    if (low_distance <= 0.0 || high_distance <= 0.0 ||
        low_distance > 1.60 || high_distance > 1.60)
      return false;
    const double angle_cosine = (low_x * high_x + low_y * high_y + low_z * high_z) /
      (low_distance * high_distance);
    if (angle_cosine > angle_cosine_limit)
      return false;
    return true;
  };

  std::vector<int> nonzero_signs;
  nonzero_signs.reserve(diagnostic.num_beads);
  std::vector<int> bead_states(diagnostic.num_beads, 0);
  std::vector<double> bead_deltas(diagnostic.num_beads, 0.0);
  double sum_delta = 0.0;
  double sum_delta_square = 0.0;
  diagnostic.delta_min = std::numeric_limits<double>::max();
  diagnostic.delta_max = -std::numeric_limits<double>::max();
  for (int bead = 0; bead < diagnostic.num_beads; ++bead) {
    double delta = 0.0;
    const bool channel_valid = bead_geometry(bead, delta);
    bead_deltas[bead] = delta;
    if (channel_valid)
      ++diagnostic.channel_valid_count;
    sum_delta += delta;
    sum_delta_square += delta * delta;
    diagnostic.delta_min = std::min(diagnostic.delta_min, delta);
    diagnostic.delta_max = std::max(diagnostic.delta_max, delta);
    const int state = classify_delta(delta);
    bead_states[bead] = state;
    if (state < 0) {
      ++diagnostic.n_minus;
      nonzero_signs.push_back(-1);
    } else if (state > 0) {
      ++diagnostic.n_plus;
      nonzero_signs.push_back(1);
    } else {
      ++diagnostic.n_zero;
    }
  }
  diagnostic.mean_delta = sum_delta / diagnostic.num_beads;
  const double inverse_num_beads = 1.0 / diagnostic.num_beads;
  diagnostic.f_minus = diagnostic.n_minus * inverse_num_beads;
  diagnostic.f_zero = diagnostic.n_zero * inverse_num_beads;
  diagnostic.f_plus = diagnostic.n_plus * inverse_num_beads;
  diagnostic.f_channel_valid = diagnostic.channel_valid_count * inverse_num_beads;
  diagnostic.sigma_delta = std::sqrt(std::max(
    0.0, sum_delta_square / diagnostic.num_beads - diagnostic.mean_delta * diagnostic.mean_delta));
  diagnostic.span = diagnostic.delta_max - diagnostic.delta_min;
  if (nonzero_signs.size() > 1) {
    for (size_t i = 0; i < nonzero_signs.size(); ++i) {
      if (nonzero_signs[i] != nonzero_signs[(i + 1) % nonzero_signs.size()])
        ++diagnostic.kink_count;
    }
  }
  int state_changes = 0;
  int zero_beads = 0;
  double sum_neighbor_delta_jump_square = 0.0;
  diagnostic.max_neighbor_delta_jump = 0.0;
  for (int bead = 0; bead < diagnostic.num_beads; ++bead) {
    const int next_bead = (bead + 1) % diagnostic.num_beads;
    const int previous_bead = (bead + diagnostic.num_beads - 1) % diagnostic.num_beads;
    if (bead_states[bead] != bead_states[next_bead])
      ++state_changes;
    if (bead_states[bead] == 0) {
      ++zero_beads;
      if (bead_states[previous_bead] != 0)
        ++diagnostic.center_domain_count;
    }
    const double jump = bead_deltas[next_bead] - bead_deltas[bead];
    sum_neighbor_delta_jump_square += jump * jump;
    diagnostic.max_neighbor_delta_jump = std::max(
      diagnostic.max_neighbor_delta_jump, std::abs(jump));
  }
  if (zero_beads == diagnostic.num_beads)
    diagnostic.center_domain_count = 1;
  diagnostic.total_state_domain_count = (state_changes == 0) ? 1 : state_changes;
  std::vector<double> sorted_deltas = bead_deltas;
  std::sort(sorted_deltas.begin(), sorted_deltas.end());
  const auto quantile = [&](const double fraction) {
    const double position = fraction * (diagnostic.num_beads - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted_deltas.size() - 1);
    const double weight = position - lower;
    return sorted_deltas[lower] + weight * (sorted_deltas[upper] - sorted_deltas[lower]);
  };
  diagnostic.delta_q20 = quantile(0.20);
  diagnostic.delta_q80 = quantile(0.80);
  diagnostic.robust_span = diagnostic.delta_q80 - diagnostic.delta_q20;
  diagnostic.centroid_minus_mean = diagnostic.delta_centroid - diagnostic.mean_delta;
  diagnostic.rms_neighbor_delta_jump = std::sqrt(
    sum_neighbor_delta_jump_square * inverse_num_beads);
  diagnostic.two_well_occupied =
    (diagnostic.f_minus >= bead_f_min_ && diagnostic.f_plus >= bead_f_min_) ? 1 : 0;
  diagnostic.two_well_span =
    (diagnostic.two_well_occupied != 0 &&
     diagnostic.span >= bead_span_min_) ? 1 : 0;
  diagnostic.multi_kink_or_multi_domain =
    (diagnostic.kink_count > 2 || diagnostic.center_domain_count > 2 ||
     diagnostic.total_state_domain_count > 4) ? 1 : 0;
  // Reject split same-sign domains such as LL00LLRR00RR even though their
  // center-skipped signed sequence has only two L/R interfaces.
  diagnostic.simple_two_domain_path =
    (diagnostic.two_well_occupied != 0 && diagnostic.f_zero <= bead_center_max_ &&
     diagnostic.kink_count == 2 && diagnostic.center_domain_count <= 2 &&
     diagnostic.total_state_domain_count <= 4) ? 1 : 0;
  diagnostic.barrier_centered = (diagnostic.simple_two_domain_path != 0 &&
    std::abs(diagnostic.delta_centroid) <= bead_centroid_max_) ? 1 : 0;
  diagnostic.strict_tunneling_like = diagnostic.barrier_centered;
  diagnostic.valid = true;
  diagnostic.character = classify_quantum_character(diagnostic);
  const auto analysis_end = std::chrono::high_resolution_clock::now();
  bead_analysis_wall_time_ +=
    std::chrono::duration<double>(analysis_end - analysis_start).count();
  return true;
}

bool Proton_Tunneling::find_geometry(
  const int hydrogen,
  const Box& box,
  GeometryResult& geometry) const
{
  geometry = GeometryResult();

  int first_oxygen = -1;
  double first_distance_square = std::numeric_limits<double>::max();
  const double hx = cpu_position_[hydrogen];
  const double hy = cpu_position_[hydrogen + number_of_atoms_];
  const double hz = cpu_position_[hydrogen + 2 * number_of_atoms_];
  for (const int oxygen : oxygen_indices_) {
    double dx = cpu_position_[oxygen] - hx;
    double dy = cpu_position_[oxygen + number_of_atoms_] - hy;
    double dz = cpu_position_[oxygen + 2 * number_of_atoms_] - hz;
    apply_mic(box, dx, dy, dz);
    const double distance_square = dx * dx + dy * dy + dz * dz;
    if (distance_square < first_distance_square) {
      first_oxygen = oxygen;
      first_distance_square = distance_square;
    }
  }
  if (first_oxygen < 0)
    return false;
  geometry.nearest_oxygen = first_oxygen;
  const int anchor_local = (first_oxygen < static_cast<int>(oxygen_local_index_.size()))
    ? oxygen_local_index_[first_oxygen]
    : -1;
  if (anchor_local < 0 || anchor_local >= static_cast<int>(oxygen_shell_neighbors_.size()))
    return false;

  std::vector<GeometryResult> candidates;
  const std::vector<int>& shell = oxygen_shell_neighbors_[anchor_local];
  candidates.reserve(shell.size());
  const double angle_cosine_limit = std::cos(oho_angle_min_deg_ * 0.017453292519943295);
  for (const int second_oxygen : shell) {
    const int second_local = (second_oxygen < static_cast<int>(oxygen_local_index_.size()))
      ? oxygen_local_index_[second_oxygen]
      : -1;
    if (second_local < 0 ||
        std::find(oxygen_shell_neighbors_[second_local].begin(),
                  oxygen_shell_neighbors_[second_local].end(), first_oxygen) ==
          oxygen_shell_neighbors_[second_local].end())
      continue;

    GeometryResult candidate;
    candidate.nearest_oxygen = first_oxygen;
    candidate.oxygen_low = std::min(first_oxygen, second_oxygen);
    candidate.oxygen_high = std::max(first_oxygen, second_oxygen);
    double ox = cpu_position_[candidate.oxygen_high] - cpu_position_[candidate.oxygen_low];
    double oy = cpu_position_[candidate.oxygen_high + number_of_atoms_] -
      cpu_position_[candidate.oxygen_low + number_of_atoms_];
    double oz = cpu_position_[candidate.oxygen_high + 2 * number_of_atoms_] -
      cpu_position_[candidate.oxygen_low + 2 * number_of_atoms_];
    apply_mic(box, ox, oy, oz);
    const double dOO_square = ox * ox + oy * oy + oz * oz;
    if (dOO_square < dOO_min_ * dOO_min_ || dOO_square > dOO_max_ * dOO_max_)
      continue;
    candidate.dOO = std::sqrt(dOO_square);
    candidate.low_to_high_dx = ox;
    candidate.low_to_high_dy = oy;
    candidate.low_to_high_dz = oz;

    double hx_from_low = hx - cpu_position_[candidate.oxygen_low];
    double hy_from_low = hy - cpu_position_[candidate.oxygen_low + number_of_atoms_];
    double hz_from_low = hz - cpu_position_[candidate.oxygen_low + 2 * number_of_atoms_];
    apply_mic(box, hx_from_low, hy_from_low, hz_from_low);
    const double projection = (hx_from_low * ox + hy_from_low * oy + hz_from_low * oz) /
      dOO_square;
    if (projection < -1.0e-10 || projection > 1.0 + 1.0e-10)
      continue;
    const double px = hx_from_low - projection * ox;
    const double py = hy_from_low - projection * oy;
    const double pz = hz_from_low - projection * oz;
    candidate.rperp = std::sqrt(std::max(0.0, px * px + py * py + pz * pz));
    if (candidate.rperp > rperp_max_)
      continue;

    double low_x = cpu_position_[candidate.oxygen_low] - hx;
    double low_y = cpu_position_[candidate.oxygen_low + number_of_atoms_] - hy;
    double low_z = cpu_position_[candidate.oxygen_low + 2 * number_of_atoms_] - hz;
    double high_x = cpu_position_[candidate.oxygen_high] - hx;
    double high_y = cpu_position_[candidate.oxygen_high + number_of_atoms_] - hy;
    double high_z = cpu_position_[candidate.oxygen_high + 2 * number_of_atoms_] - hz;
    apply_mic(box, low_x, low_y, low_z);
    apply_mic(box, high_x, high_y, high_z);
    const double low_distance = std::sqrt(low_x * low_x + low_y * low_y + low_z * low_z);
    const double high_distance = std::sqrt(high_x * high_x + high_y * high_y + high_z * high_z);
    if (low_distance <= 0.0 || high_distance <= 0.0 ||
        low_distance > 1.60 || high_distance > 1.60)
      continue;
    const double angle_cosine = (low_x * high_x + low_y * high_y + low_z * high_z) /
      (low_distance * high_distance);
    if (angle_cosine > angle_cosine_limit)
      continue;

    candidate.delta = low_distance - high_distance;
    candidate.path_excess = std::max(0.0, low_distance + high_distance - candidate.dOO);
    candidate.assignment_score = candidate.rperp +
      assignment_path_excess_weight_ * candidate.path_excess;
    compute_ion_field(box, candidate);
    candidates.push_back(candidate);
  }

  if (candidates.empty())
    return false;
  std::sort(candidates.begin(), candidates.end(), [](const GeometryResult& first,
                                                     const GeometryResult& second) {
    if (first.assignment_score != second.assignment_score)
      return first.assignment_score < second.assignment_score;
    return first.rperp < second.rperp;
  });
  geometry = candidates.front();
  geometry.candidate_count = static_cast<int>(candidates.size());
  geometry.second_assignment_score = (candidates.size() > 1)
    ? candidates[1].assignment_score
    : std::numeric_limits<double>::infinity();
  geometry.assignment_score_gap = geometry.second_assignment_score - geometry.assignment_score;
  if (candidates.size() > 1 && geometry.assignment_score_gap < assignment_score_gap_min_) {
    geometry.valid = false;
    geometry.assignment_ambiguous = true;
    return false;
  }
  geometry.valid = true;
  return true;
}

int Proton_Tunneling::classify_delta(const double delta) const
{
  if (delta > delta_cutoff_)
    return 1;
  if (delta < -delta_cutoff_)
    return -1;
  return 0;
}

void Proton_Tunneling::record_bond(
  std::unordered_map<unsigned long long, BondStats>& bond_stats,
  const GeometryResult& geometry,
  const int state)
{
  BondStats& stats = bond_stats[make_bond_key(geometry.oxygen_low, geometry.oxygen_high)];
  ++stats.geometry_samples;
  stats.sum_abs_delta += std::abs(geometry.delta);
  stats.sum_delta += geometry.delta;
  stats.sum_delta_square += geometry.delta * geometry.delta;
  stats.sum_dOO += geometry.dOO;
  stats.sum_rperp += geometry.rperp;
  if (ion_field_enabled_) {
    stats.sum_E_parallel += geometry.E_ion_nominal_parallel;
    stats.sum_E2_parallel += geometry.E_ion_nominal_parallel * geometry.E_ion_nominal_parallel;
    stats.sum_delta_E_parallel += geometry.delta * geometry.E_ion_nominal_parallel;
    stats.sum_nearest_ion1_distance += geometry.nearest_ion1_distance;
    stats.sum_nearest_ion2_distance += geometry.nearest_ion2_distance;
    stats.sum_delta_phi += geometry.delta_phi_ion;
    stats.sum_delta_phi2 += geometry.delta_phi_ion * geometry.delta_phi_ion;
    stats.sum_delta_delta_phi += geometry.delta * geometry.delta_phi_ion;
    stats.sum_ion1_to_low += geometry.nearest_ion1_to_low;
    stats.sum_ion1_to_high += geometry.nearest_ion1_to_high;
    stats.sum_ion2_to_low += geometry.nearest_ion2_to_low;
    stats.sum_ion2_to_high += geometry.nearest_ion2_to_high;
  }
  if (state > 0)
    ++stats.n_plus;
  else if (state < 0)
    ++stats.n_minus;
  else
    ++stats.n_deadband;
}

void Proton_Tunneling::record_local_environment(
  const GeometryResult& geometry,
  const LocalEnvironment& environment)
{
  if (!local_environment_enabled_ || !environment.valid)
    return;
  LocalEnvironmentStats& stats = window_local_environment_stats_[
    make_bond_key(geometry.oxygen_low, geometry.oxygen_high)];
  ++stats.samples;
  stats.sum_delta += geometry.delta;
  stats.sum_delta2 += geometry.delta * geometry.delta;
  stats.sum_rOH_low += environment.rOH_low;
  stats.sum_rOH_high += environment.rOH_high;
  stats.sum_oho_angle += environment.oho_angle;
  stats.sum_dOO += environment.dOO;
  stats.sum_rperp += environment.rperp;
  stats.sum_path_excess += environment.path_excess;
  stats.sum_nH_low += environment.nH_low;
  stats.sum_nH_high += environment.nH_high;
  stats.sum_donor_edges_low += environment.donor_edges_low;
  stats.sum_donor_edges_high += environment.donor_edges_high;
  stats.sum_acceptor_edges_low += environment.acceptor_edges_low;
  stats.sum_acceptor_edges_high += environment.acceptor_edges_high;
  const auto add_distance = [](const double value, double& sum, long long& count) {
    if (std::isfinite(value) && value < 1.0e299) {
      sum += value;
      ++count;
    }
  };
  add_distance(environment.ion1_low_d1, stats.sum_ion1_low_d1, stats.count_ion1_low[0]);
  add_distance(environment.ion1_low_d2, stats.sum_ion1_low_d2, stats.count_ion1_low[1]);
  add_distance(environment.ion1_low_d3, stats.sum_ion1_low_d3, stats.count_ion1_low[2]);
  add_distance(environment.ion1_high_d1, stats.sum_ion1_high_d1, stats.count_ion1_high[0]);
  add_distance(environment.ion1_high_d2, stats.sum_ion1_high_d2, stats.count_ion1_high[1]);
  add_distance(environment.ion1_high_d3, stats.sum_ion1_high_d3, stats.count_ion1_high[2]);
  add_distance(environment.ion2_low_d1, stats.sum_ion2_low_d1, stats.count_ion2_low[0]);
  add_distance(environment.ion2_low_d2, stats.sum_ion2_low_d2, stats.count_ion2_low[1]);
  add_distance(environment.ion2_low_d3, stats.sum_ion2_low_d3, stats.count_ion2_low[2]);
  add_distance(environment.ion2_high_d1, stats.sum_ion2_high_d1, stats.count_ion2_high[0]);
  add_distance(environment.ion2_high_d2, stats.sum_ion2_high_d2, stats.count_ion2_high[1]);
  add_distance(environment.ion2_high_d3, stats.sum_ion2_high_d3, stats.count_ion2_high[2]);
  stats.sum_coord_ion1_low += environment.coord_ion1_low;
  stats.sum_coord_ion1_high += environment.coord_ion1_high;
  stats.sum_coord_ion2_low += environment.coord_ion2_low;
  stats.sum_coord_ion2_high += environment.coord_ion2_high;
  add_distance(environment.delta_d_ion1, stats.sum_delta_d_ion1, stats.count_delta_d_ion1);
  add_distance(environment.delta_d_ion2, stats.sum_delta_d_ion2, stats.count_delta_d_ion2);
  add_distance(
    environment.nearest_ion2_to_H,
    stats.sum_nearest_ion2_to_H,
    stats.count_nearest_ion2_to_H);
  if (environment.angle_Olow_H_ion2 >= 0.0) {
    stats.sum_angle_Olow_H_ion2 += environment.angle_Olow_H_ion2;
    ++stats.count_angle_Olow_H_ion2;
  }
  if (environment.angle_Ohigh_H_ion2 >= 0.0) {
    stats.sum_angle_Ohigh_H_ion2 += environment.angle_Ohigh_H_ion2;
    ++stats.count_angle_Ohigh_H_ion2;
  }
  stats.sum_hcl_like_low += environment.hcl_like_low;
  stats.sum_hcl_like_high += environment.hcl_like_high;
  if (ion_field_enabled_) {
    stats.sum_E_parallel += environment.E_parallel;
    stats.sum_delta_phi_ion += environment.delta_phi_ion;
    stats.sum_delta_phi2 += environment.delta_phi_ion * environment.delta_phi_ion;
    stats.sum_delta_delta_phi += geometry.delta * environment.delta_phi_ion;
  }
}

void Proton_Tunneling::start_attempt(
  HydrogenState& hydrogen_state,
  const int stable_state,
  const double time_fs,
  const GeometryResult& geometry,
  const LocalEnvironment& environment)
{
  hydrogen_state.attempt_active = true;
  hydrogen_state.attempt_from_state = stable_state;
  hydrogen_state.attempt_id = next_attempt_id_++;
  hydrogen_state.attempt_start_time_fs = time_fs;
  hydrogen_state.attempt_delta_start = geometry.delta;
  hydrogen_state.attempt_E_start = geometry.E_ion_nominal_parallel;
  hydrogen_state.attempt_delta_phi_start = geometry.delta_phi_ion;
  hydrogen_state.attempt_delta_d_ion1_start = geometry.nearest_ion1_to_high -
    geometry.nearest_ion1_to_low;
  hydrogen_state.attempt_delta_d_ion2_start = geometry.nearest_ion2_to_high -
    geometry.nearest_ion2_to_low;
  hydrogen_state.attempt_min_abs_delta = std::abs(geometry.delta);
  hydrogen_state.last_environment = environment;
  hydrogen_state.attempt_environment_start = environment;
  hydrogen_state.pending_state = 0;
  hydrogen_state.pending_count = 0;
  hydrogen_state.pending_start_time_fs = 0.0;
  hydrogen_state.centroid_best = BeadDiagnostic();
  hydrogen_state.delocalization_best = BeadDiagnostic();
}

const char* Proton_Tunneling::outcome_name(const AttemptOutcome outcome) const
{
  switch (outcome) {
  case AttemptOutcome::success:
    return "success";
  case AttemptOutcome::return_to_state:
    return "return";
  case AttemptOutcome::geometry_lost:
    return "geometry_lost";
  case AttemptOutcome::run_end:
    return "run_end";
  }
  return "unknown";
}

void Proton_Tunneling::finish_attempt(
  const int hydrogen,
  HydrogenState& hydrogen_state,
  const AttemptOutcome outcome,
  const double time_fs,
  const double delta_end,
  const GeometryResult* geometry,
  const LocalEnvironment* environment)
{
  if (!hydrogen_state.attempt_active)
    return;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double E_end = (geometry != nullptr)
    ? geometry->E_ion_nominal_parallel
    : hydrogen_state.last_E_parallel;
  const int nearest_ion_id = (geometry != nullptr)
    ? geometry->nearest_ion_id
    : hydrogen_state.last_nearest_ion_id;
  const double nearest_ion_distance = (geometry != nullptr)
    ? geometry->nearest_ion_distance
    : hydrogen_state.last_nearest_ion_distance;
  const double delta_phi_end = (geometry != nullptr)
    ? geometry->delta_phi_ion
    : hydrogen_state.last_delta_phi_ion;
  const double E_start_output = ion_field_enabled_ ? hydrogen_state.attempt_E_start : nan;
  const double E_end_output = ion_field_enabled_ ? E_end : nan;
  const double delta_phi_start_output = ion_field_enabled_
    ? hydrogen_state.attempt_delta_phi_start
    : nan;
  const double delta_phi_end_output = ion_field_enabled_ ? delta_phi_end : nan;
  const double delta_d_ion1_start_output = ion_field_enabled_
    ? hydrogen_state.attempt_delta_d_ion1_start
    : nan;
  const double delta_d_ion2_start_output = ion_field_enabled_
    ? hydrogen_state.attempt_delta_d_ion2_start
    : nan;
  const int nearest_ion_id_output = ion_field_enabled_ ? nearest_ion_id : -1;
  const double nearest_ion_distance_output = ion_field_enabled_ ? nearest_ion_distance : nan;

  const int oxygen_low = hydrogen_state.oxygen_low;
  const int oxygen_high = hydrogen_state.oxygen_high;
  const int oxygen_from = (hydrogen_state.attempt_from_state < 0)
    ? oxygen_low
    : oxygen_high;
  const int oxygen_target = (hydrogen_state.attempt_from_state < 0)
    ? oxygen_high
    : oxygen_low;
  const unsigned long long key = make_bond_key(oxygen_low, oxygen_high);
  BondStats& total_stats = total_bonds_[key];
  BondStats& window_stats = window_bonds_[key];
  ++total_stats.attempts;
  ++window_stats.attempts;
  if (outcome == AttemptOutcome::success) {
    ++total_stats.successes;
    ++window_stats.successes;
    ++total_stats.transitions;
    ++window_stats.transitions;
    ++window_flip_count_;
  } else if (outcome == AttemptOutcome::return_to_state) {
    ++total_stats.returns;
    ++window_stats.returns;
  } else if (outcome == AttemptOutcome::geometry_lost) {
    ++total_stats.geometry_lost;
    ++window_stats.geometry_lost;
  }
  if (ion_field_enabled_ && outcome == AttemptOutcome::success) {
    total_stats.sum_E_success += E_end;
    window_stats.sum_E_success += E_end;
    ++total_stats.n_E_success;
    ++window_stats.n_E_success;
  } else if (ion_field_enabled_ && outcome == AttemptOutcome::return_to_state) {
    total_stats.sum_E_return += E_end;
    window_stats.sum_E_return += E_end;
    ++total_stats.n_E_return;
    ++window_stats.n_E_return;
  }

  AttemptRecord record;
  record.attempt_id = hydrogen_state.attempt_id;
  record.time_start_fs = hydrogen_state.attempt_start_time_fs;
  record.time_end_fs = time_fs;
  record.hydrogen = hydrogen;
  record.oxygen_low = oxygen_low;
  record.oxygen_high = oxygen_high;
  record.oxygen_from = oxygen_from;
  record.oxygen_target = oxygen_target;
  record.outcome = outcome;
  record.delta_start = hydrogen_state.attempt_delta_start;
  record.min_abs_delta = hydrogen_state.attempt_min_abs_delta;
  record.delta_end = delta_end;
  record.E_parallel_start = E_start_output;
  record.E_parallel_end = E_end_output;
  record.delta_phi_start = delta_phi_start_output;
  record.delta_phi_end = delta_phi_end_output;
  record.delta_d_ion1_start = delta_d_ion1_start_output;
  record.delta_d_ion2_start = delta_d_ion2_start_output;
  record.nearest_ion_id = nearest_ion_id_output;
  record.nearest_ion_distance = nearest_ion_distance_output;
  record.environment_start = hydrogen_state.attempt_environment_start;
  record.environment_last_valid = hydrogen_state.last_environment;
  record.environment_end = (environment != nullptr)
    ? *environment
    : hydrogen_state.last_environment;
  record.centroid_best = hydrogen_state.centroid_best;
  record.delocalization_best = hydrogen_state.delocalization_best;

  if (outcome == AttemptOutcome::success && geometry != nullptr) {
    const int nH_from_before = event_hydrogen_count_[oxygen_from];
    const int nH_to_before = event_hydrogen_count_[oxygen_target];
    const int nH_from_after = nH_from_before - 1;
    const int nH_to_after = nH_to_before + 1;
    event_hydrogen_count_[oxygen_from] = nH_from_after;
    event_hydrogen_count_[oxygen_target] = nH_to_after;
    if (oxygen_from >= 0 && oxygen_from < static_cast<int>(frame_cause_event_ids_.size()))
      frame_cause_event_ids_[oxygen_from] = hydrogen_state.attempt_id;
    if (oxygen_target >= 0 && oxygen_target < static_cast<int>(frame_cause_event_ids_.size()))
      frame_cause_event_ids_[oxygen_target] = hydrogen_state.attempt_id;
    double dx = geometry->low_to_high_dx;
    double dy = geometry->low_to_high_dy;
    double dz = geometry->low_to_high_dz;
    if (oxygen_from != geometry->oxygen_low) {
      dx = -dx;
      dy = -dy;
      dz = -dz;
    }
    record.has_transfer = true;
    record.nH_from_before = nH_from_before;
    record.nH_to_before = nH_to_before;
    record.nH_from_after = nH_from_after;
    record.nH_to_after = nH_to_after;
    record.dx = dx;
    record.dy = dy;
    record.dz = dz;
  }
  if (local_environment_enabled_) {
    LocalEnvironmentStats& environment_stats = window_local_environment_stats_[key];
    ++environment_stats.attempts;
    if (outcome == AttemptOutcome::success)
      ++environment_stats.successes;
    else if (outcome == AttemptOutcome::return_to_state)
      ++environment_stats.returns;
    else if (outcome == AttemptOutcome::geometry_lost)
      ++environment_stats.geometry_lost;
  }
  attempt_records_.push_back(record);

  if (outcome == AttemptOutcome::success)
    hydrogen_state.stable_state = -hydrogen_state.attempt_from_state;
  else if (outcome == AttemptOutcome::geometry_lost || outcome == AttemptOutcome::run_end)
    hydrogen_state.stable_state = 0;
  hydrogen_state.attempt_active = false;
  hydrogen_state.attempt_from_state = 0;
  hydrogen_state.attempt_id = 0;
  hydrogen_state.attempt_start_time_fs = 0.0;
  hydrogen_state.attempt_delta_start = 0.0;
  hydrogen_state.attempt_E_start = 0.0;
  hydrogen_state.attempt_delta_phi_start = 0.0;
  hydrogen_state.attempt_delta_d_ion1_start = 0.0;
  hydrogen_state.attempt_delta_d_ion2_start = 0.0;
  hydrogen_state.attempt_min_abs_delta = 0.0;
  hydrogen_state.last_environment = LocalEnvironment();
  hydrogen_state.attempt_environment_start = LocalEnvironment();
  hydrogen_state.pending_state = 0;
  hydrogen_state.pending_count = 0;
  hydrogen_state.pending_start_time_fs = 0.0;
  hydrogen_state.centroid_best = BeadDiagnostic();
  hydrogen_state.delocalization_best = BeadDiagnostic();
}

void Proton_Tunneling::observe_frame(
  const double time_fs,
  const Box& box,
  Atom& atom)
{
  const auto observer_start = std::chrono::high_resolution_clock::now();
  ++observer_frame_count_;
  bead_positions_cached_ = false;
  cached_number_of_beads_ = 0;
  if (window_sample_count_ == 0)
    window_start_time_fs_ = time_fs;

  std::fill(hydrogen_count_.begin(), hydrogen_count_.end(), 0);
  std::fill(frame_cause_event_ids_.begin(), frame_cause_event_ids_.end(), -1);
  compute_geometry_gpu(box, atom);
  const auto state_machine_start = std::chrono::high_resolution_clock::now();
  for (const GeometryResult& geometry : frame_geometries_)
    if (geometry.nearest_oxygen >= 0)
      ++hydrogen_count_[geometry.nearest_oxygen];

  std::unordered_map<unsigned long long, int> pair_assignment_counts;
  for (const GeometryResult& geometry : frame_geometries_) {
    if (geometry.valid)
      ++pair_assignment_counts[make_bond_key(geometry.oxygen_low, geometry.oxygen_high)];
  }
  for (GeometryResult& geometry : frame_geometries_) {
    if (geometry.valid && pair_assignment_counts[
          make_bond_key(geometry.oxygen_low, geometry.oxygen_high)] > 1) {
      geometry.valid = false;
      geometry.assignment_ambiguous = true;
      geometry.pair_conflict = true;
    }
  }
  compute_local_environment_gpu(box, atom);
  assign_local_environment_topology();

  if (!defect_state_initialized_) {
    for (const int oxygen : oxygen_indices_) {
      DefectRecord record;
      record.time_fs = time_fs;
      record.oxygen = oxygen;
      record.q_defect = hydrogen_count_[oxygen] - 2;
      record.hydrogen_count = hydrogen_count_[oxygen];
      record.cause_event_id = 0;
      defect_records_.push_back(record);
    }
    previous_hydrogen_count_ = hydrogen_count_;
    defect_state_initialized_ = true;
  }
  event_hydrogen_count_ = previous_hydrogen_count_;

  bool bead_probe_frame = false;
  const auto probe_attempt = [&](const int hydrogen,
                                 const GeometryResult& geometry,
                                 HydrogenState& hydrogen_state,
                                 const LocalEnvironment& environment) {
    if (!bead_diagnostic_enabled_)
      return;
    if (!bead_probe_frame) {
      ++bead_probe_frame_count_;
      bead_probe_frame = true;
    }
    BeadDiagnostic diagnostic;
    const bool evaluated = evaluate_bead_diagnostic(
      atom, box, hydrogen, geometry, time_fs, diagnostic);
    if (!evaluated)
      return;
    diagnostic.environment = environment;
    if (!hydrogen_state.centroid_best.valid ||
        std::abs(diagnostic.delta_centroid) <
          std::abs(hydrogen_state.centroid_best.delta_centroid))
      hydrogen_state.centroid_best = diagnostic;
    if (is_better_delocalization_diagnostic(
          diagnostic, hydrogen_state.delocalization_best))
      hydrogen_state.delocalization_best = diagnostic;
  };

  const LocalEnvironment empty_environment;
  for (size_t h_index = 0; h_index < hydrogen_indices_.size(); ++h_index) {
    const int hydrogen = hydrogen_indices_[h_index];
    const GeometryResult& geometry = frame_geometries_[h_index];
    const LocalEnvironment& local_environment = local_environment_enabled_
      ? frame_local_environments_[h_index]
      : empty_environment;
    const bool valid_geometry = geometry.valid;

    HydrogenState& hydrogen_state = hydrogen_states_[h_index];
    if (!valid_geometry) {
      if (geometry.assignment_ambiguous) {
        ++window_assignment_ambiguous_count_;
        if (geometry.pair_conflict)
          ++window_pair_conflict_count_;
        continue;
      }
      if (hydrogen_state.attempt_active) {
        finish_attempt(
          hydrogen,
          hydrogen_state,
           AttemptOutcome::geometry_lost,
           time_fs,
           hydrogen_state.last_delta,
           nullptr,
           nullptr);
      }
      hydrogen_state = HydrogenState();
      continue;
    }

    ++window_valid_pair_count_;
    const int state = classify_delta(geometry.delta);

    if (hydrogen_state.oxygen_low != geometry.oxygen_low ||
        hydrogen_state.oxygen_high != geometry.oxygen_high) {
      if (hydrogen_state.attempt_active) {
        finish_attempt(
          hydrogen,
          hydrogen_state,
           AttemptOutcome::geometry_lost,
           time_fs,
           hydrogen_state.last_delta,
           nullptr,
           nullptr);
      }
      hydrogen_state = HydrogenState();
      hydrogen_state.oxygen_low = geometry.oxygen_low;
      hydrogen_state.oxygen_high = geometry.oxygen_high;
    }

    if (local_environment.valid)
      record_local_environment(geometry, local_environment);
    record_bond(window_bonds_, geometry, state);
    record_bond(total_bonds_, geometry, state);

    if (hydrogen_state.stable_state == 0) {
      if (state == 0) {
        hydrogen_state.pending_state = 0;
        hydrogen_state.pending_count = 0;
      } else {
        if (hydrogen_state.pending_state == state) {
          ++hydrogen_state.pending_count;
        } else {
          hydrogen_state.pending_state = state;
          hydrogen_state.pending_count = 1;
        }
        if (hydrogen_state.pending_count >= hold_samples_) {
          hydrogen_state.stable_state = state;
          hydrogen_state.pending_state = 0;
          hydrogen_state.pending_count = 0;
        }
      }
    } else {
      if (!hydrogen_state.attempt_active && state != hydrogen_state.stable_state) {
        // If sampling jumps across the dead band, retain the event rather than silently
        // losing it. With a sufficiently small sample interval, normal attempts start in state 0.
        start_attempt(
          hydrogen_state, hydrogen_state.stable_state, time_fs, geometry, local_environment);
      }

      if (hydrogen_state.attempt_active) {
        probe_attempt(hydrogen, geometry, hydrogen_state, local_environment);
        const double abs_delta = std::abs(geometry.delta);
        if (abs_delta < hydrogen_state.attempt_min_abs_delta) {
          hydrogen_state.attempt_min_abs_delta = abs_delta;
        }
        if (state == hydrogen_state.stable_state) {
          finish_attempt(
            hydrogen,
            hydrogen_state,
            AttemptOutcome::return_to_state,
            time_fs,
            geometry.delta,
            &geometry,
            &local_environment);
        } else if (state == -hydrogen_state.attempt_from_state) {
          if (hydrogen_state.pending_state == state) {
            ++hydrogen_state.pending_count;
          } else {
            hydrogen_state.pending_state = state;
            hydrogen_state.pending_count = 1;
            hydrogen_state.pending_start_time_fs = time_fs;
          }
          if (hydrogen_state.pending_count >= hold_samples_) {
            finish_attempt(
              hydrogen,
              hydrogen_state,
              AttemptOutcome::success,
              time_fs,
              geometry.delta,
              &geometry,
              &local_environment);
          }
        } else {
          hydrogen_state.pending_state = 0;
          hydrogen_state.pending_count = 0;
          hydrogen_state.pending_start_time_fs = 0.0;
        }
      } else if (state == 0) {
        start_attempt(
          hydrogen_state, hydrogen_state.stable_state, time_fs, geometry, local_environment);
        probe_attempt(hydrogen, geometry, hydrogen_state, local_environment);
      }
    }
    hydrogen_state.last_delta = geometry.delta;
    hydrogen_state.last_E_parallel = geometry.E_ion_nominal_parallel;
    hydrogen_state.last_delta_phi_ion = geometry.delta_phi_ion;
    hydrogen_state.last_nearest_ion_id = geometry.nearest_ion_id;
    hydrogen_state.last_nearest_ion_distance = geometry.nearest_ion_distance;
    hydrogen_state.last_environment = local_environment;
  }

  for (const int oxygen : oxygen_indices_) {
    if (hydrogen_count_[oxygen] != previous_hydrogen_count_[oxygen]) {
      DefectRecord record;
      record.time_fs = time_fs;
      record.oxygen = oxygen;
      record.q_defect = hydrogen_count_[oxygen] - 2;
      record.hydrogen_count = hydrogen_count_[oxygen];
      record.cause_event_id = frame_cause_event_ids_[oxygen];
      defect_records_.push_back(record);
    }
  }
  previous_hydrogen_count_ = hydrogen_count_;

  for (const int oxygen : oxygen_indices_) {
    if (hydrogen_count_[oxygen] > 2)
      ++window_positive_defect_sum_;
    else if (hydrogen_count_[oxygen] < 2)
      ++window_negative_defect_sum_;
  }

  ++window_sample_count_;
  last_time_fs_ = time_fs;
  if (window_sample_count_ == window_samples_)
    write_window(time_fs);
  const auto observer_end = std::chrono::high_resolution_clock::now();
  state_machine_wall_time_ +=
    std::chrono::duration<double>(observer_end - state_machine_start).count();
  total_observer_wall_time_ +=
    std::chrono::duration<double>(observer_end - observer_start).count();
}

void Proton_Tunneling::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  (void)number_of_steps;
  (void)fixed_group;
  (void)move_group;
  (void)temperature;
  (void)integrate;
  (void)group;
  (void)thermo;
  (void)force;
  if (!initialized_ || (step + 1) % sample_interval_ != 0)
    return;

  observe_frame(global_time * TIME_UNIT_CONVERSION, box, atom);
}

void Proton_Tunneling::write_window(const double time_fs)
{
  if (window_sample_count_ == 0)
    return;

  double b_mean = 0.0;
  double f_02 = 0.0;
  double f_04 = 0.0;
  double delta_f_sum = 0.0;
  int active_bonds = 0;
  int delta_f_bonds = 0;
  for (const auto& item : window_bonds_) {
    const BondStats& stats = item.second;
    const long long biased_samples = stats.n_plus + stats.n_minus;
    if (biased_samples == 0)
      continue;
    ++active_bonds;
    const double asymmetry = static_cast<double>(stats.n_plus - stats.n_minus) /
      static_cast<double>(biased_samples);
    const double abs_asymmetry = std::abs(asymmetry);
    b_mean += abs_asymmetry;
    if (abs_asymmetry > 0.2)
      f_02 += 1.0;
    if (abs_asymmetry > 0.4)
      f_04 += 1.0;
    if (stats.n_plus > 0 && stats.n_minus > 0) {
      delta_f_sum += std::abs(std::log(
        static_cast<double>(stats.n_plus) / static_cast<double>(stats.n_minus)));
      ++delta_f_bonds;
    }
  }

  if (active_bonds > 0) {
    b_mean /= active_bonds;
    f_02 /= active_bonds;
    f_04 /= active_bonds;
  }
  const double mean_abs_delta_f = (delta_f_bonds > 0)
    ? delta_f_sum / delta_f_bonds
    : std::numeric_limits<double>::quiet_NaN();
  const double sample_dt_ps = time_step_ * sample_interval_ * TIME_UNIT_CONVERSION / 1000.0;
  const double window_time_ps = sample_dt_ps * window_sample_count_;
  const double flip_rate = (window_time_ps > 0.0)
    ? static_cast<double>(window_flip_count_) / window_time_ps
    : 0.0;
  const double valid_pairs_per_frame = static_cast<double>(window_valid_pair_count_) /
    window_sample_count_;
  const double positive_defects = static_cast<double>(window_positive_defect_sum_) /
    window_sample_count_;
  const double negative_defects = static_cast<double>(window_negative_defect_sum_) /
    window_sample_count_;

  WindowRecord record;
  record.window_id = window_id_;
  record.time_start_fs = window_start_time_fs_;
  record.time_end_fs = time_fs;
  record.B_mean = b_mean;
  record.f_02 = f_02;
  record.f_04 = f_04;
  record.mean_abs_delta_f = mean_abs_delta_f;
  record.flip_rate = flip_rate;
  record.active_bonds = active_bonds;
  record.positive_defects = positive_defects;
  record.negative_defects = negative_defects;
  record.valid_pairs_per_frame = valid_pairs_per_frame;
  record.assignment_ambiguous_samples = window_assignment_ambiguous_count_;
  record.pair_conflict_samples = window_pair_conflict_count_;
  window_records_.push_back(record);

  write_edge_window(window_start_time_fs_, time_fs);
  if (local_environment_enabled_)
    write_local_environment_window(window_start_time_fs_, time_fs);

  window_bonds_.clear();
  window_local_environment_stats_.clear();
  window_sample_count_ = 0;
  window_start_time_fs_ = 0.0;
  window_flip_count_ = 0;
  window_valid_pair_count_ = 0;
  window_positive_defect_sum_ = 0;
  window_negative_defect_sum_ = 0;
  window_assignment_ambiguous_count_ = 0;
  window_pair_conflict_count_ = 0;
}

void Proton_Tunneling::write_local_environment_window(
  const double time_start_fs,
  const double time_end_fs)
{
  for (const auto& item : window_local_environment_stats_) {
    LocalEnvironmentWindowRecord record;
    record.window_id = window_id_;
    record.time_start_fs = time_start_fs;
    record.time_end_fs = time_end_fs;
    decode_bond_key(item.first, record.oxygen_low, record.oxygen_high);
    record.stats = item.second;
    local_environment_window_records_.push_back(record);
  }
}

void Proton_Tunneling::write_edge_window(
  const double time_start_fs,
  const double time_end_fs)
{
  BondStats empty_stats;
  for (const auto& item : total_bonds_) {
    int oxygen_low;
    int oxygen_high;
    decode_bond_key(item.first, oxygen_low, oxygen_high);
    const auto window_item = window_bonds_.find(item.first);
    const BondStats& stats = (window_item == window_bonds_.end())
      ? empty_stats
      : window_item->second;
    const long long biased_samples = stats.n_plus + stats.n_minus;
    const long long completed_attempts = stats.successes + stats.returns;
    const double geometry_occupancy = static_cast<double>(stats.geometry_samples) /
      window_sample_count_;
    const double asymmetry = (biased_samples > 0)
      ? static_cast<double>(stats.n_plus - stats.n_minus) / static_cast<double>(biased_samples)
      : std::numeric_limits<double>::quiet_NaN();
    const double delta_f = (stats.n_plus > 0 && stats.n_minus > 0)
      ? std::abs(std::log(static_cast<double>(stats.n_plus) /
                          static_cast<double>(stats.n_minus)))
      : std::numeric_limits<double>::quiet_NaN();
    const double log_population_ratio = (stats.n_plus > 0 && stats.n_minus > 0)
      ? std::log(static_cast<double>(stats.n_plus) / static_cast<double>(stats.n_minus))
      : std::numeric_limits<double>::quiet_NaN();
    const double beta_DeltaF_high_minus_low = (stats.n_plus > 0 && stats.n_minus > 0)
      ? -log_population_ratio
      : std::numeric_limits<double>::quiet_NaN();
    const double abs_beta_DeltaF = (stats.n_plus > 0 && stats.n_minus > 0)
      ? std::abs(log_population_ratio)
      : std::numeric_limits<double>::quiet_NaN();
    const double success_probability = (completed_attempts > 0)
      ? static_cast<double>(stats.successes) / static_cast<double>(completed_attempts)
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_delta = (stats.geometry_samples > 0)
      ? stats.sum_delta / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_abs_delta = (stats.geometry_samples > 0)
      ? stats.sum_abs_delta / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_dOO = (stats.geometry_samples > 0)
      ? stats.sum_dOO / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_rperp = (stats.geometry_samples > 0)
      ? stats.sum_rperp / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_E_parallel = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_E_parallel / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double std_E_parallel = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? std::sqrt(std::max(0.0, stats.sum_E2_parallel / stats.geometry_samples -
          mean_E_parallel * mean_E_parallel))
      : std::numeric_limits<double>::quiet_NaN();
    double corr_delta_E_parallel = std::numeric_limits<double>::quiet_NaN();
    if (ion_field_enabled_ && stats.geometry_samples > 0) {
      const double variance_delta = std::max(0.0,
        stats.sum_delta_square / stats.geometry_samples - mean_delta * mean_delta);
      const double variance_E = std::max(0.0,
        stats.sum_E2_parallel / stats.geometry_samples - mean_E_parallel * mean_E_parallel);
      if (variance_delta > 0.0 && variance_E > 0.0) {
        const double covariance = stats.sum_delta_E_parallel / stats.geometry_samples -
          mean_delta * mean_E_parallel;
        corr_delta_E_parallel = covariance / std::sqrt(variance_delta * variance_E);
      }
    }
    const double mean_E_success = (ion_field_enabled_ && stats.n_E_success > 0)
      ? stats.sum_E_success / stats.n_E_success
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_E_return = (ion_field_enabled_ && stats.n_E_return > 0)
      ? stats.sum_E_return / stats.n_E_return
      : std::numeric_limits<double>::quiet_NaN();
    const double nearest_ion1_distance = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_nearest_ion1_distance / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double nearest_ion2_distance = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_nearest_ion2_distance / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_delta_phi_ion = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_delta_phi / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double std_delta_phi_ion = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? std::sqrt(std::max(0.0, stats.sum_delta_phi2 / stats.geometry_samples -
          mean_delta_phi_ion * mean_delta_phi_ion))
      : std::numeric_limits<double>::quiet_NaN();
    double corr_delta_delta_phi = std::numeric_limits<double>::quiet_NaN();
    if (ion_field_enabled_ && stats.geometry_samples > 0) {
      const double variance_delta = std::max(0.0,
        stats.sum_delta_square / stats.geometry_samples - mean_delta * mean_delta);
      const double variance_delta_phi = std::max(0.0,
        stats.sum_delta_phi2 / stats.geometry_samples -
        mean_delta_phi_ion * mean_delta_phi_ion);
      if (variance_delta > 0.0 && variance_delta_phi > 0.0) {
        const double covariance = stats.sum_delta_delta_phi / stats.geometry_samples -
          mean_delta * mean_delta_phi_ion;
        corr_delta_delta_phi = covariance /
          std::sqrt(variance_delta * variance_delta_phi);
      }
    }
    const double mean_ion1_to_O_low = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_ion1_to_low / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_ion1_to_O_high = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_ion1_to_high / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_delta_d_ion1 = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? mean_ion1_to_O_high - mean_ion1_to_O_low
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_ion2_to_O_low = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_ion2_to_low / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_ion2_to_O_high = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? stats.sum_ion2_to_high / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_delta_d_ion2 = (ion_field_enabled_ && stats.geometry_samples > 0)
      ? mean_ion2_to_O_high - mean_ion2_to_O_low
      : std::numeric_limits<double>::quiet_NaN();
    EdgeWindowRecord record;
    record.window_id = window_id_;
    record.time_start_fs = time_start_fs;
    record.time_end_fs = time_end_fs;
    record.oxygen_low = oxygen_low;
    record.oxygen_high = oxygen_high;
    record.geometry_occupancy = geometry_occupancy;
    record.n_plus = stats.n_plus;
    record.n_minus = stats.n_minus;
    record.n_deadband = stats.n_deadband;
    record.asymmetry = asymmetry;
    record.abs_asymmetry = std::abs(asymmetry);
    record.delta_f = delta_f;
    record.log_population_ratio = log_population_ratio;
    record.beta_DeltaF_high_minus_low = beta_DeltaF_high_minus_low;
    record.abs_beta_DeltaF = abs_beta_DeltaF;
    record.attempts = stats.attempts;
    record.successes = stats.successes;
    record.returns = stats.returns;
    record.geometry_lost = stats.geometry_lost;
    record.success_probability = success_probability;
    record.mean_delta = mean_delta;
    record.mean_abs_delta = mean_abs_delta;
    record.mean_dOO = mean_dOO;
    record.mean_rperp = mean_rperp;
    record.mean_E_parallel = mean_E_parallel;
    record.std_E_parallel = std_E_parallel;
    record.corr_delta_E_parallel = corr_delta_E_parallel;
    record.mean_E_success = mean_E_success;
    record.mean_E_return = mean_E_return;
    record.nearest_ion1_distance = nearest_ion1_distance;
    record.nearest_ion2_distance = nearest_ion2_distance;
    record.mean_delta_phi_ion = mean_delta_phi_ion;
    record.std_delta_phi_ion = std_delta_phi_ion;
    record.corr_delta_delta_phi = corr_delta_delta_phi;
    record.mean_ion1_to_O_low = mean_ion1_to_O_low;
    record.mean_ion1_to_O_high = mean_ion1_to_O_high;
    record.mean_delta_d_ion1 = mean_delta_d_ion1;
    record.mean_ion2_to_O_low = mean_ion2_to_O_low;
    record.mean_ion2_to_O_high = mean_ion2_to_O_high;
    record.mean_delta_d_ion2 = mean_delta_d_ion2;
    edge_window_records_.push_back(record);
  }
  ++window_id_;
}

void Proton_Tunneling::write_final_bonds()
{
  for (const auto& item : total_bonds_) {
    int oxygen_low;
    int oxygen_high;
    decode_bond_key(item.first, oxygen_low, oxygen_high);
    const BondStats& stats = item.second;
    const long long biased_samples = stats.n_plus + stats.n_minus;
    const double asymmetry = (biased_samples > 0)
      ? static_cast<double>(stats.n_plus - stats.n_minus) / static_cast<double>(biased_samples)
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_abs_delta = (stats.geometry_samples > 0)
      ? stats.sum_abs_delta / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    fprintf(
      bond_file_,
      "%d %d %lld %lld %lld %lld %.10e %.10e %.10e\n",
      oxygen_low,
      oxygen_high,
      stats.geometry_samples,
      stats.n_plus,
      stats.n_minus,
      stats.transitions,
      asymmetry,
      std::abs(asymmetry),
      mean_abs_delta);
  }
}

void Proton_Tunneling::write_local_environment_event(
  FILE* file,
  const AttemptRecord& attempt,
  const char* snapshot_kind,
  const double time_fs,
  const LocalEnvironment& environment,
  const char* quantum_class)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto value = [&](const double x) {
    return (environment.valid && std::isfinite(x) && x < 1.0e299) ? x : nan;
  };
  const auto angle_value = [&](const double x) {
    return (environment.valid && std::isfinite(x) && x >= 0.0) ? x : nan;
  };
  const auto count = [&](const int x) {
    return environment.valid ? x : -1;
  };
  fprintf(
    file,
    "%lld %s %s %.10e %d %d %d %d "
    "%.10e %.10e %.10e %.10e %.10e %.10e "
    "%.10e %.10e %.10e %.10e %.10e %.10e "
    "%.10e %.10e %.10e %.10e %.10e %.10e "
    "%d %d %d %d "
    "%.10e %.10e %.10e %.10e %.10e "
    "%d %d %.10e %.10e "
    "%d %d %d %d %d %d %s\n",
    attempt.attempt_id,
    snapshot_kind,
    outcome_name(attempt.outcome),
    environment.valid ? time_fs : nan,
    attempt.hydrogen,
    attempt.oxygen_low,
    attempt.oxygen_high,
    environment.valid ? 1 : 0,
    value(environment.rOH_low),
    value(environment.rOH_high),
    value(environment.oho_angle),
    value(environment.dOO),
    value(environment.rperp),
    value(environment.path_excess),
    value(environment.ion1_low_d1),
    value(environment.ion1_low_d2),
    value(environment.ion1_low_d3),
    value(environment.ion1_high_d1),
    value(environment.ion1_high_d2),
    value(environment.ion1_high_d3),
    value(environment.ion2_low_d1),
    value(environment.ion2_low_d2),
    value(environment.ion2_low_d3),
    value(environment.ion2_high_d1),
    value(environment.ion2_high_d2),
    value(environment.ion2_high_d3),
    count(environment.coord_ion1_low),
    count(environment.coord_ion1_high),
    count(environment.coord_ion2_low),
    count(environment.coord_ion2_high),
    value(environment.delta_d_ion1),
    value(environment.delta_d_ion2),
    value(environment.nearest_ion2_to_H),
    angle_value(environment.angle_Olow_H_ion2),
    angle_value(environment.angle_Ohigh_H_ion2),
    count(environment.hcl_like_low),
    count(environment.hcl_like_high),
    value(environment.E_parallel),
    value(environment.delta_phi_ion),
    count(environment.nH_low),
    count(environment.nH_high),
    count(environment.donor_edges_low),
    count(environment.donor_edges_high),
    count(environment.acceptor_edges_low),
    count(environment.acceptor_edges_high),
    quantum_class);
}

void Proton_Tunneling::write_text_output_files()
{
  bias_file_ = my_fopen("proton_bias.out", "a");
  transfer_file_ = my_fopen("proton_transfer.out", "a");
  attempt_file_ = my_fopen("proton_attempt.out", "a");
  if (bead_diagnostic_enabled_)
    bead_event_file_ = my_fopen("proton_bead_event.out", "a");
  edge_window_file_ = my_fopen("proton_edge_window.out", "a");
  bond_file_ = my_fopen("proton_bond.out", "a");
  defect_file_ = my_fopen("proton_defect.out", "a");
  if (local_environment_enabled_) {
    local_environment_window_file_ = my_fopen("proton_local_environment_window.out", "a");
    local_environment_event_file_ = my_fopen("proton_local_environment_event.out", "a");
  }

  fprintf(bias_file_,
    "# compute_proton_tunneling %d %d %.10e %d %.10e %.10e %.10e %s %s oho_angle %.10e",
    sample_interval_, window_samples_, delta_cutoff_, hold_samples_, dOO_min_, dOO_max_,
    rperp_max_, oxygen_symbol_.c_str(), hydrogen_symbol_.c_str(), oho_angle_min_deg_);
  if (ion_field_enabled_)
    fprintf(bias_file_, " ion_field %s %.10e %s %.10e %.10e",
      ion1_symbol_.c_str(), ion1_charge_, ion2_symbol_.c_str(), ion2_charge_, ion_field_cutoff_);
  if (bead_diagnostic_enabled_)
    fprintf(bias_file_, " bead_diagnostic %.10e %.10e %.10e %.10e",
      bead_f_min_, bead_span_min_, bead_center_max_, bead_centroid_max_);
  if (local_environment_enabled_)
    fprintf(bias_file_, " local_environment %.10e %.10e %.10e %.10e",
      local_ion1_cutoff_, local_ion2_cutoff_, local_hcl_cutoff_, local_hcl_angle_min_deg_);
  fprintf(bias_file_, "\n");
  fprintf(bias_file_,
    "# columns time_fs B_mean F_A_gt_0.2 F_A_gt_0.4 mean_abs_DeltaF_over_kBT "
    "flip_rate_per_ps active_bonds positive_defects negative_defects valid_pairs_per_frame "
    "assignment_ambiguous_samples pair_conflict_samples\n");

  fprintf(transfer_file_,
    "# columns event_id time_start_fs time_confirm_fs H_id O_from O_to O_pair_low O_pair_high "
    "nH_from_before nH_to_before nH_from_after nH_to_after "
    "q_from_before q_to_before q_from_after q_to_after dx dy dz delta_start delta_confirm\n");
  fprintf(attempt_file_,
    "# columns attempt_id time_start_fs time_end_fs H_id O_low O_high O_from O_target outcome "
    "delta_start min_abs_delta delta_end E_parallel_start E_parallel_end "
    "nearest_ion_id nearest_ion_distance delta_phi_start delta_phi_end ");
  if (ion_field_enabled_)
    fprintf(attempt_file_, "delta_d_%s_start delta_d_%s_start\n",
      ion1_symbol_.c_str(), ion2_symbol_.c_str());
  else
    fprintf(attempt_file_, "delta_d_ion1_start delta_d_ion2_start\n");
  if (bead_diagnostic_enabled_) {
    fprintf(bead_event_file_,
      "# columns attempt_id selection_kind probe_time_fs H_id O_low O_high outcome num_beads "
      "delta_centroid f_minus f_zero f_plus mean_delta centroid_minus_mean sigma_delta "
      "delta_min delta_max span delta_q20 delta_q80 robust_span kink_count "
      "center_domain_count total_state_domain_count two_well_occupied two_well_span "
      "simple_two_domain_path barrier_centered strict_tunneling_like channel_valid_count "
      "f_channel_valid multi_kink_or_multi_domain rms_neighbor_delta_jump "
      "max_neighbor_delta_jump quantum_class\n");
  }
  fprintf(defect_file_, "# columns time_fs O_id q_defect nH cause_event_id\n");
  fprintf(edge_window_file_,
    "# columns window_id time_start_fs time_end_fs O_low O_high geometry_occupancy "
    "n_plus n_minus n_deadband A abs_A DeltaF_over_kBT attempts successes returns geometry_lost "
    "success_probability mean_delta mean_abs_delta mean_dOO mean_rperp "
    "mean_E_parallel std_E_parallel corr_delta_E_parallel mean_E_success mean_E_return ");
  if (ion_field_enabled_)
    fprintf(edge_window_file_, "nearest_%s_distance nearest_%s_distance",
      ion1_symbol_.c_str(), ion2_symbol_.c_str());
  else
    fprintf(edge_window_file_, "nearest_ion1_distance nearest_ion2_distance");
  if (ion_field_enabled_)
    fprintf(edge_window_file_,
      " log_population_ratio beta_DeltaF_high_minus_low abs_beta_DeltaF "
      "mean_delta_phi_ion std_delta_phi_ion corr_delta_delta_phi "
      "mean_%s_to_O_low mean_%s_to_O_high mean_delta_d_%s "
      "mean_%s_to_O_low mean_%s_to_O_high mean_delta_d_%s\n",
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion1_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str());
  else
    fprintf(edge_window_file_,
      " log_population_ratio beta_DeltaF_high_minus_low abs_beta_DeltaF "
      "mean_delta_phi_ion std_delta_phi_ion corr_delta_delta_phi "
      "mean_ion1_to_O_low mean_ion1_to_O_high mean_delta_d_ion1 "
      "mean_ion2_to_O_low mean_ion2_to_O_high mean_delta_d_ion2\n");
  fprintf(bond_file_,
    "# columns O_pair_low O_pair_high geometry_samples n_plus n_minus transitions "
    "A abs_A mean_abs_delta\n");
  if (local_environment_enabled_) {
    fprintf(
      local_environment_window_file_,
      "# columns window_id time_start_fs time_end_fs O_low O_high samples "
      "mean_delta mean_rOH_low mean_rOH_high mean_oho_angle mean_dOO mean_rperp mean_path_excess "
      "mean_nH_low mean_nH_high mean_donor_edges_low mean_donor_edges_high "
      "mean_acceptor_edges_low mean_acceptor_edges_high "
      "mean_%s_low_d1 mean_%s_low_d2 mean_%s_low_d3 mean_%s_high_d1 mean_%s_high_d2 mean_%s_high_d3 "
      "mean_%s_low_d1 mean_%s_low_d2 mean_%s_low_d3 mean_%s_high_d1 mean_%s_high_d2 mean_%s_high_d3 "
      "mean_coord_%s_low mean_coord_%s_high mean_coord_%s_low mean_coord_%s_high "
      "mean_delta_d_%s mean_delta_d_%s mean_nearest_%s_to_H mean_angle_Olow_H_%s "
      "mean_angle_Ohigh_H_%s fraction_hcl_like_low fraction_hcl_like_high "
      "mean_E_parallel mean_delta_phi_ion std_delta_phi_ion "
      "corr_delta_delta_phi attempts successes returns geometry_lost\n",
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion1_symbol_.c_str(),
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion1_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion1_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str());
    fprintf(
      local_environment_event_file_,
      "# columns attempt_id snapshot_kind outcome time_fs H_id O_low O_high valid "
      "rOH_low rOH_high oho_angle dOO rperp path_excess "
      "%s_low_d1 %s_low_d2 %s_low_d3 %s_high_d1 %s_high_d2 %s_high_d3 "
      "%s_low_d1 %s_low_d2 %s_low_d3 %s_high_d1 %s_high_d2 %s_high_d3 "
      "coord_%s_low coord_%s_high coord_%s_low coord_%s_high delta_d_%s delta_d_%s "
      "nearest_%s_to_H angle_Olow_H_%s angle_Ohigh_H_%s hcl_like_low hcl_like_high "
      "E_parallel delta_phi_ion "
      "nH_low nH_high donor_edges_low donor_edges_high acceptor_edges_low acceptor_edges_high quantum_class\n",
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion1_symbol_.c_str(),
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion1_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion1_symbol_.c_str(), ion1_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion1_symbol_.c_str(), ion2_symbol_.c_str(), ion2_symbol_.c_str(),
      ion2_symbol_.c_str(), ion2_symbol_.c_str());
  }

  const double bead_nan = std::numeric_limits<double>::quiet_NaN();
  for (const AttemptRecord& record : attempt_records_) {
    fprintf(
      attempt_file_,
      "%lld %.10e %.10e %d %d %d %d %d %s %.10e %.10e %.10e %.10e %.10e %d %.10e "
      "%.10e %.10e %.10e %.10e\n",
      record.attempt_id,
      record.time_start_fs,
      record.time_end_fs,
      record.hydrogen,
      record.oxygen_low,
      record.oxygen_high,
      record.oxygen_from,
      record.oxygen_target,
      outcome_name(record.outcome),
      record.delta_start,
      record.min_abs_delta,
      record.delta_end,
      record.E_parallel_start,
      record.E_parallel_end,
      record.nearest_ion_id,
      record.nearest_ion_distance,
      record.delta_phi_start,
      record.delta_phi_end,
      record.delta_d_ion1_start,
      record.delta_d_ion2_start);

    if (bead_event_file_ != nullptr) {
      const auto write_bead_event = [&](const char* selection_kind,
                                        const BeadDiagnostic& diagnostic) {
        const bool valid = diagnostic.valid;
        fprintf(
          bead_event_file_,
          "%lld %s %.10e %d %d %d %s %d "
          "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e "
          "%d %d %d %d %d %d %d %d %d %d "
          "%.10e %.10e %.10e %s\n",
          record.attempt_id,
          selection_kind,
          valid ? diagnostic.probe_time_fs : bead_nan,
          record.hydrogen,
          record.oxygen_low,
          record.oxygen_high,
          outcome_name(record.outcome),
          diagnostic.num_beads,
          valid ? diagnostic.delta_centroid : bead_nan,
          valid ? diagnostic.f_minus : bead_nan,
          valid ? diagnostic.f_zero : bead_nan,
          valid ? diagnostic.f_plus : bead_nan,
          valid ? diagnostic.mean_delta : bead_nan,
          valid ? diagnostic.centroid_minus_mean : bead_nan,
          valid ? diagnostic.sigma_delta : bead_nan,
          valid ? diagnostic.delta_min : bead_nan,
          valid ? diagnostic.delta_max : bead_nan,
          valid ? diagnostic.span : bead_nan,
          valid ? diagnostic.delta_q20 : bead_nan,
          valid ? diagnostic.delta_q80 : bead_nan,
          valid ? diagnostic.robust_span : bead_nan,
          valid ? diagnostic.kink_count : -1,
          valid ? diagnostic.center_domain_count : -1,
          valid ? diagnostic.total_state_domain_count : -1,
          valid ? diagnostic.two_well_occupied : -1,
          valid ? diagnostic.two_well_span : -1,
          valid ? diagnostic.simple_two_domain_path : -1,
          valid ? diagnostic.barrier_centered : -1,
          valid ? diagnostic.strict_tunneling_like : -1,
          valid ? diagnostic.channel_valid_count : -1,
          valid ? diagnostic.multi_kink_or_multi_domain : -1,
          valid ? diagnostic.f_channel_valid : bead_nan,
          valid ? diagnostic.rms_neighbor_delta_jump : bead_nan,
          valid ? diagnostic.max_neighbor_delta_jump : bead_nan,
          quantum_character_name(diagnostic.character));
      };
      write_bead_event("centroid_best", record.centroid_best);
      write_bead_event("delocalization_best", record.delocalization_best);
    }

    if (local_environment_event_file_ != nullptr) {
      write_local_environment_event(
        local_environment_event_file_, record, "start", record.time_start_fs,
        record.environment_start, "not_applicable");
      write_local_environment_event(
        local_environment_event_file_, record, "end", record.time_end_fs,
        record.environment_end, "not_applicable");
      write_local_environment_event(
        local_environment_event_file_, record, "last_valid", record.time_end_fs,
        record.environment_last_valid, "not_applicable");
      if (record.centroid_best.valid)
        write_local_environment_event(
          local_environment_event_file_, record, "centroid_best",
          record.centroid_best.probe_time_fs, record.centroid_best.environment,
          quantum_character_name(record.centroid_best.character));
      if (record.delocalization_best.valid)
        write_local_environment_event(
          local_environment_event_file_, record, "delocalization_best",
          record.delocalization_best.probe_time_fs, record.delocalization_best.environment,
          quantum_character_name(record.delocalization_best.character));
    }

    if (record.has_transfer) {
      fprintf(
        transfer_file_,
        "%lld %.10e %.10e %d %d %d %d %d %d %d %d %d %d %d %d %d "
        "%.10e %.10e %.10e %.10e %.10e\n",
        record.attempt_id,
        record.time_start_fs,
        record.time_end_fs,
        record.hydrogen,
        record.oxygen_from,
        record.oxygen_target,
        record.oxygen_low,
        record.oxygen_high,
        record.nH_from_before,
        record.nH_to_before,
        record.nH_from_after,
        record.nH_to_after,
        record.nH_from_before - 2,
        record.nH_to_before - 2,
        record.nH_from_after - 2,
        record.nH_to_after - 2,
        record.dx,
        record.dy,
        record.dz,
        record.delta_start,
        record.delta_end);
    }
  }

  for (const DefectRecord& record : defect_records_)
    fprintf(defect_file_, "%.10e %d %d %d %lld\n", record.time_fs, record.oxygen,
      record.q_defect, record.hydrogen_count, record.cause_event_id);

  for (const WindowRecord& record : window_records_)
    fprintf(
      bias_file_,
      "%.10e %.10e %.10e %.10e %.10e %.10e %d %.10e %.10e %.10e %lld %lld\n",
      record.time_end_fs,
      record.B_mean,
      record.f_02,
      record.f_04,
      record.mean_abs_delta_f,
      record.flip_rate,
      record.active_bonds,
      record.positive_defects,
      record.negative_defects,
      record.valid_pairs_per_frame,
      record.assignment_ambiguous_samples,
      record.pair_conflict_samples);

  for (const EdgeWindowRecord& record : edge_window_records_)
    fprintf(
      edge_window_file_,
      "%lld %.10e %.10e %d %d %.10e %lld %lld %lld %.10e %.10e %.10e "
      "%lld %lld %lld %lld %.10e %.10e %.10e %.10e %.10e "
      "%.10e %.10e %.10e %.10e %.10e %.10e %.10e "
      "%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
      record.window_id,
      record.time_start_fs,
      record.time_end_fs,
      record.oxygen_low,
      record.oxygen_high,
      record.geometry_occupancy,
      record.n_plus,
      record.n_minus,
      record.n_deadband,
      record.asymmetry,
      record.abs_asymmetry,
      record.delta_f,
      record.attempts,
      record.successes,
      record.returns,
      record.geometry_lost,
      record.success_probability,
      record.mean_delta,
      record.mean_abs_delta,
      record.mean_dOO,
      record.mean_rperp,
      record.mean_E_parallel,
      record.std_E_parallel,
      record.corr_delta_E_parallel,
      record.mean_E_success,
      record.mean_E_return,
      record.nearest_ion1_distance,
      record.nearest_ion2_distance,
      record.log_population_ratio,
      record.beta_DeltaF_high_minus_low,
      record.abs_beta_DeltaF,
      record.mean_delta_phi_ion,
      record.std_delta_phi_ion,
      record.corr_delta_delta_phi,
      record.mean_ion1_to_O_low,
      record.mean_ion1_to_O_high,
      record.mean_delta_d_ion1,
      record.mean_ion2_to_O_low,
      record.mean_ion2_to_O_high,
      record.mean_delta_d_ion2);

  if (local_environment_window_file_ != nullptr) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const LocalEnvironmentWindowRecord& record : local_environment_window_records_) {
      const LocalEnvironmentStats& stats = record.stats;
      const auto mean = [&](const double sum, const long long count) {
        return count > 0 ? sum / static_cast<double>(count) : nan;
      };
      const long long samples = stats.samples;
      const double mean_delta = mean(stats.sum_delta, samples);
      const double mean_phi = (ion_field_enabled_ && samples > 0)
        ? stats.sum_delta_phi_ion / static_cast<double>(samples)
        : nan;
      const double std_phi = (ion_field_enabled_ && samples > 0)
        ? std::sqrt(std::max(
            0.0,
            stats.sum_delta_phi2 / static_cast<double>(samples) - mean_phi * mean_phi))
        : nan;
      double corr_phi = nan;
      if (ion_field_enabled_ && samples > 0) {
        const double variance_delta = std::max(
          0.0,
          stats.sum_delta2 / static_cast<double>(samples) - mean_delta * mean_delta);
        const double variance_phi = std::max(
          0.0,
          stats.sum_delta_phi2 / static_cast<double>(samples) - mean_phi * mean_phi);
        if (variance_delta > 0.0 && variance_phi > 0.0) {
          const double covariance = stats.sum_delta_delta_phi / static_cast<double>(samples) -
            mean_delta * mean_phi;
          corr_phi = covariance / std::sqrt(variance_delta * variance_phi);
        }
      }
      fprintf(
        local_environment_window_file_,
        "%lld %.10e %.10e %d %d %lld",
        record.window_id,
        record.time_start_fs,
        record.time_end_fs,
        record.oxygen_low,
        record.oxygen_high,
        samples);
      const auto write_mean_samples = [&](const double sum) {
        fprintf(local_environment_window_file_, " %.10e", mean(sum, samples));
      };
      const auto write_mean_count = [&](const double sum, const long long count) {
        fprintf(local_environment_window_file_, " %.10e", mean(sum, count));
      };
      write_mean_samples(stats.sum_delta);
      write_mean_samples(stats.sum_rOH_low);
      write_mean_samples(stats.sum_rOH_high);
      write_mean_samples(stats.sum_oho_angle);
      write_mean_samples(stats.sum_dOO);
      write_mean_samples(stats.sum_rperp);
      write_mean_samples(stats.sum_path_excess);
      write_mean_samples(static_cast<double>(stats.sum_nH_low));
      write_mean_samples(static_cast<double>(stats.sum_nH_high));
      write_mean_samples(static_cast<double>(stats.sum_donor_edges_low));
      write_mean_samples(static_cast<double>(stats.sum_donor_edges_high));
      write_mean_samples(static_cast<double>(stats.sum_acceptor_edges_low));
      write_mean_samples(static_cast<double>(stats.sum_acceptor_edges_high));
      write_mean_count(stats.sum_ion1_low_d1, stats.count_ion1_low[0]);
      write_mean_count(stats.sum_ion1_low_d2, stats.count_ion1_low[1]);
      write_mean_count(stats.sum_ion1_low_d3, stats.count_ion1_low[2]);
      write_mean_count(stats.sum_ion1_high_d1, stats.count_ion1_high[0]);
      write_mean_count(stats.sum_ion1_high_d2, stats.count_ion1_high[1]);
      write_mean_count(stats.sum_ion1_high_d3, stats.count_ion1_high[2]);
      write_mean_count(stats.sum_ion2_low_d1, stats.count_ion2_low[0]);
      write_mean_count(stats.sum_ion2_low_d2, stats.count_ion2_low[1]);
      write_mean_count(stats.sum_ion2_low_d3, stats.count_ion2_low[2]);
      write_mean_count(stats.sum_ion2_high_d1, stats.count_ion2_high[0]);
      write_mean_count(stats.sum_ion2_high_d2, stats.count_ion2_high[1]);
      write_mean_count(stats.sum_ion2_high_d3, stats.count_ion2_high[2]);
      write_mean_samples(static_cast<double>(stats.sum_coord_ion1_low));
      write_mean_samples(static_cast<double>(stats.sum_coord_ion1_high));
      write_mean_samples(static_cast<double>(stats.sum_coord_ion2_low));
      write_mean_samples(static_cast<double>(stats.sum_coord_ion2_high));
      write_mean_count(stats.sum_delta_d_ion1, stats.count_delta_d_ion1);
      write_mean_count(stats.sum_delta_d_ion2, stats.count_delta_d_ion2);
      write_mean_count(stats.sum_nearest_ion2_to_H, stats.count_nearest_ion2_to_H);
      write_mean_count(stats.sum_angle_Olow_H_ion2, stats.count_angle_Olow_H_ion2);
      write_mean_count(stats.sum_angle_Ohigh_H_ion2, stats.count_angle_Ohigh_H_ion2);
      fprintf(
        local_environment_window_file_,
        " %.10e %.10e %.10e %.10e %.10e %.10e %lld %lld %lld %lld\n",
        mean(static_cast<double>(stats.sum_hcl_like_low), samples),
        mean(static_cast<double>(stats.sum_hcl_like_high), samples),
        (ion_field_enabled_ && samples > 0)
          ? stats.sum_E_parallel / static_cast<double>(samples) : nan,
        mean_phi,
        std_phi,
        corr_phi,
        stats.attempts,
        stats.successes,
        stats.returns,
        stats.geometry_lost);
    }
  }

  write_final_bonds();
  fclose(bias_file_);
  fclose(transfer_file_);
  fclose(attempt_file_);
  if (bead_event_file_ != nullptr)
    fclose(bead_event_file_);
  fclose(defect_file_);
  fclose(edge_window_file_);
  fclose(bond_file_);
  if (local_environment_window_file_ != nullptr)
    fclose(local_environment_window_file_);
  if (local_environment_event_file_ != nullptr)
    fclose(local_environment_event_file_);
  bias_file_ = nullptr;
  transfer_file_ = nullptr;
  attempt_file_ = nullptr;
  bead_event_file_ = nullptr;
  defect_file_ = nullptr;
  edge_window_file_ = nullptr;
  bond_file_ = nullptr;
  local_environment_window_file_ = nullptr;
  local_environment_event_file_ = nullptr;
}

void Proton_Tunneling::write_netcdf_output_file()
{
#ifndef USE_NETCDF
  PRINT_INPUT_ERROR(
    "proton observer NetCDF output requires a GPUMD build with USE_NETCDF=1 and NetCDF4 support.");
#else
  const bool include_events = output_level_ != OutputLevel::SUMMARY;
  const bool include_full = output_level_ == OutputLevel::FULL;
  int ncid = -1;
  const int create_status = nc_create(
    output_filename_.c_str(), NC_NETCDF4 | NC_NOCLOBBER, &ncid);
  if (create_status != NC_NOERR) {
    if (create_status == NC_EEXIST) {
      fprintf(stderr,
        "Proton observer NetCDF output %s already exists; remove or rename it before rerunning.\n",
        output_filename_.c_str());
    } else {
      fprintf(stderr, "Proton observer cannot create NetCDF output %s: %s\n",
        output_filename_.c_str(), nc_strerror(create_status));
    }
    std::exit(2);
  }

  netcdf_text_attribute(ncid, NC_GLOBAL, "program", "GPUMD");
  netcdf_text_attribute(ncid, NC_GLOBAL, "observer", "compute_proton_tunneling");
  netcdf_text_attribute(ncid, NC_GLOBAL, "format", "GPUMD proton observer NetCDF-4");
  netcdf_text_attribute(ncid, NC_GLOBAL, "format_version", "1");
  netcdf_text_attribute(ncid, NC_GLOBAL, "oxygen_symbol", oxygen_symbol_.c_str());
  netcdf_text_attribute(ncid, NC_GLOBAL, "hydrogen_symbol", hydrogen_symbol_.c_str());
  netcdf_text_attribute(ncid, NC_GLOBAL, "ion1_symbol", ion1_symbol_.c_str());
  netcdf_text_attribute(ncid, NC_GLOBAL, "ion2_symbol", ion2_symbol_.c_str());
  const char* output_level = output_level_ == OutputLevel::SUMMARY ? "summary" :
    (output_level_ == OutputLevel::EVENTS ? "events" : "full");
  const char* snapshot_mode = snapshot_mode_ == SnapshotMode::ENDPOINTS ? "endpoints" :
    (snapshot_mode_ == SnapshotMode::BEST ? "best" : "all");
  netcdf_text_attribute(ncid, NC_GLOBAL, "output_level", output_level);
  netcdf_text_attribute(ncid, NC_GLOBAL, "snapshot_mode", snapshot_mode);
  netcdf_text_attribute(ncid, NC_GLOBAL, "outcome_enum", "0=success,1=return,2=geometry_lost,3=run_end");
  netcdf_text_attribute(ncid, NC_GLOBAL, "quantum_class_enum",
    "0=classical_only,1=two_well_delocalized,2=barrier_centered_tunneling_like,"
    "3=compact_single_domain,4=multi_kink_or_multi_domain,5=ambiguous,255=not_applicable");
  netcdf_text_attribute(ncid, NC_GLOBAL, "compression", "shuffle=1,deflate=1");
  netcdf_check(nc_put_att_int(ncid, NC_GLOBAL, "compression_level", NC_INT, 1,
    &compression_level_), "nc_put_att_int");
  netcdf_check(nc_put_att_int(ncid, NC_GLOBAL, "sample_interval", NC_INT, 1,
    &sample_interval_), "nc_put_att_int");
  netcdf_check(nc_put_att_int(ncid, NC_GLOBAL, "window_samples", NC_INT, 1,
    &window_samples_), "nc_put_att_int");

  std::vector<unsigned long long> edge_keys;
  const auto add_edge = [&](const int oxygen_low, const int oxygen_high) {
    if (oxygen_low >= 0 && oxygen_high >= 0)
      edge_keys.push_back(make_bond_key(oxygen_low, oxygen_high));
  };
  for (const EdgeWindowRecord& record : edge_window_records_)
    add_edge(record.oxygen_low, record.oxygen_high);
  for (const LocalEnvironmentWindowRecord& record : local_environment_window_records_)
    add_edge(record.oxygen_low, record.oxygen_high);
  for (const AttemptRecord& record : attempt_records_)
    add_edge(record.oxygen_low, record.oxygen_high);
  for (const auto& item : total_bonds_) {
    int oxygen_low;
    int oxygen_high;
    decode_bond_key(item.first, oxygen_low, oxygen_high);
    add_edge(oxygen_low, oxygen_high);
  }
  std::sort(edge_keys.begin(), edge_keys.end());
  edge_keys.erase(std::unique(edge_keys.begin(), edge_keys.end()), edge_keys.end());
  std::unordered_map<unsigned long long, int> edge_ids;
  edge_ids.reserve(edge_keys.size());
  for (size_t i = 0; i < edge_keys.size(); ++i)
    edge_ids[edge_keys[i]] = static_cast<int>(i);

  std::unordered_map<long long, int> window_ids;
  window_ids.reserve(window_records_.size());
  for (size_t i = 0; i < window_records_.size(); ++i)
    window_ids[window_records_[i].window_id] = static_cast<int>(i);

  int edge_group = -1;
  int edge_oxygen_var = -1;
  if (!edge_keys.empty()) {
    netcdf_check(nc_def_grp(ncid, "edge", &edge_group), "nc_def_grp");
    const int edge_dim = netcdf_dimension(edge_group, "edge", edge_keys.size());
    const int endpoint_dim = netcdf_dimension(edge_group, "endpoint", 2);
    edge_oxygen_var = netcdf_variable(edge_group, "oxygen", NC_INT,
      {edge_dim, endpoint_dim}, {edge_keys.size(), 2}, compression_level_);
    netcdf_text_attribute(edge_group, edge_oxygen_var,
      "description", "zero-based O atom indices; edge row is the edge_id");
  }

  int window_group = -1;
  int window_time_var = -1;
  int window_value_var = -1;
  int window_count_var = -1;
  std::vector<double> window_times;
  std::vector<double> window_values;
  std::vector<long long> window_counts;
  const std::vector<const char*> window_value_names = {
    "B_mean", "f_02", "f_04", "mean_abs_delta_f", "flip_rate_per_ps",
    "positive_defects", "negative_defects", "valid_pairs_per_frame"};
  const std::vector<const char*> window_count_names = {
    "active_bonds", "assignment_ambiguous_samples", "pair_conflict_samples"};
  if (!window_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "window", &window_group), "nc_def_grp");
    const int window_dim = netcdf_dimension(window_group, "window", window_records_.size());
    const int endpoint_dim = netcdf_dimension(window_group, "endpoint", 2);
    const int value_dim = netcdf_dimension(window_group, "value", window_value_names.size());
    const int count_dim = netcdf_dimension(window_group, "count", window_count_names.size());
    window_time_var = netcdf_variable(window_group, "time_fs", NC_DOUBLE,
      {window_dim, endpoint_dim}, {window_records_.size(), 2}, compression_level_);
    window_value_var = netcdf_variable(window_group, "value", NC_DOUBLE,
      {window_dim, value_dim}, {window_records_.size(), window_value_names.size()}, compression_level_);
    window_count_var = netcdf_variable(window_group, "count", NC_INT64,
      {window_dim, count_dim}, {window_records_.size(), window_count_names.size()}, compression_level_);
    netcdf_text_attribute(window_group, window_value_var, "field_names",
      "B_mean,f_02,f_04,mean_abs_delta_f,flip_rate_per_ps,positive_defects,negative_defects,valid_pairs_per_frame");
    netcdf_text_attribute(window_group, window_count_var, "field_names",
      "active_bonds,assignment_ambiguous_samples,pair_conflict_samples");
    window_times.reserve(window_records_.size() * 2);
    window_values.reserve(window_records_.size() * window_value_names.size());
    window_counts.reserve(window_records_.size() * window_count_names.size());
    for (const WindowRecord& record : window_records_) {
      window_times.push_back(record.time_start_fs);
      window_times.push_back(record.time_end_fs);
      window_values.insert(window_values.end(), {
        record.B_mean, record.f_02, record.f_04, record.mean_abs_delta_f, record.flip_rate,
        record.positive_defects, record.negative_defects, record.valid_pairs_per_frame});
      window_counts.insert(window_counts.end(), {
        record.active_bonds, record.assignment_ambiguous_samples, record.pair_conflict_samples});
    }
  }

  int edge_window_group = -1;
  int edge_window_edge_var = -1;
  int edge_window_window_var = -1;
  int edge_window_value_var = -1;
  int edge_window_count_var = -1;
  std::vector<int> edge_window_edges;
  std::vector<int> edge_window_windows;
  std::vector<double> edge_window_values;
  std::vector<long long> edge_window_counts;
  const std::vector<const char*> edge_window_value_names = {
    "geometry_occupancy", "asymmetry", "abs_asymmetry", "delta_f", "success_probability",
    "mean_delta", "mean_abs_delta", "mean_dOO", "mean_rperp", "mean_E_parallel",
    "std_E_parallel", "corr_delta_E_parallel", "mean_E_success", "mean_E_return",
    "nearest_ion1_distance", "nearest_ion2_distance", "log_population_ratio",
    "beta_DeltaF_high_minus_low", "abs_beta_DeltaF", "mean_delta_phi_ion",
    "std_delta_phi_ion", "corr_delta_delta_phi", "mean_ion1_to_O_low", "mean_ion1_to_O_high",
    "mean_delta_d_ion1", "mean_ion2_to_O_low", "mean_ion2_to_O_high", "mean_delta_d_ion2"};
  const std::vector<const char*> edge_window_count_names = {
    "n_plus", "n_minus", "n_deadband", "attempts", "successes", "returns", "geometry_lost"};
  if (!edge_window_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "edge_window", &edge_window_group), "nc_def_grp");
    const int row_dim = netcdf_dimension(edge_window_group, "row", edge_window_records_.size());
    const int value_dim = netcdf_dimension(edge_window_group, "value", edge_window_value_names.size());
    const int count_dim = netcdf_dimension(edge_window_group, "count", edge_window_count_names.size());
    edge_window_edge_var = netcdf_variable(edge_window_group, "edge_id", NC_INT,
      {row_dim}, {edge_window_records_.size()}, compression_level_);
    edge_window_window_var = netcdf_variable(edge_window_group, "window_index", NC_INT,
      {row_dim}, {edge_window_records_.size()}, compression_level_);
    edge_window_value_var = netcdf_variable(edge_window_group, "value", NC_DOUBLE,
      {row_dim, value_dim}, {edge_window_records_.size(), edge_window_value_names.size()}, compression_level_);
    edge_window_count_var = netcdf_variable(edge_window_group, "count", NC_INT64,
      {row_dim, count_dim}, {edge_window_records_.size(), edge_window_count_names.size()}, compression_level_);
    netcdf_text_attribute(edge_window_group, edge_window_value_var, "field_names",
      "geometry_occupancy,asymmetry,abs_asymmetry,delta_f,success_probability,mean_delta,"
      "mean_abs_delta,mean_dOO,mean_rperp,mean_E_parallel,std_E_parallel,corr_delta_E_parallel,"
      "mean_E_success,mean_E_return,nearest_ion1_distance,nearest_ion2_distance,log_population_ratio,"
      "beta_DeltaF_high_minus_low,abs_beta_DeltaF,mean_delta_phi_ion,std_delta_phi_ion,"
      "corr_delta_delta_phi,mean_ion1_to_O_low,mean_ion1_to_O_high,mean_delta_d_ion1,"
      "mean_ion2_to_O_low,mean_ion2_to_O_high,mean_delta_d_ion2");
    netcdf_text_attribute(edge_window_group, edge_window_count_var, "field_names",
      "n_plus,n_minus,n_deadband,attempts,successes,returns,geometry_lost");
    edge_window_edges.reserve(edge_window_records_.size());
    edge_window_windows.reserve(edge_window_records_.size());
    edge_window_values.reserve(edge_window_records_.size() * edge_window_value_names.size());
    edge_window_counts.reserve(edge_window_records_.size() * edge_window_count_names.size());
    for (const EdgeWindowRecord& record : edge_window_records_) {
      edge_window_edges.push_back(edge_ids[make_bond_key(record.oxygen_low, record.oxygen_high)]);
      const auto window_item = window_ids.find(record.window_id);
      edge_window_windows.push_back(window_item == window_ids.end() ? -1 : window_item->second);
      edge_window_values.insert(edge_window_values.end(), {
        record.geometry_occupancy, record.asymmetry, record.abs_asymmetry, record.delta_f,
        record.success_probability, record.mean_delta, record.mean_abs_delta, record.mean_dOO,
        record.mean_rperp, record.mean_E_parallel, record.std_E_parallel,
        record.corr_delta_E_parallel, record.mean_E_success, record.mean_E_return,
        record.nearest_ion1_distance, record.nearest_ion2_distance, record.log_population_ratio,
        record.beta_DeltaF_high_minus_low, record.abs_beta_DeltaF, record.mean_delta_phi_ion,
        record.std_delta_phi_ion, record.corr_delta_delta_phi, record.mean_ion1_to_O_low,
        record.mean_ion1_to_O_high, record.mean_delta_d_ion1, record.mean_ion2_to_O_low,
        record.mean_ion2_to_O_high, record.mean_delta_d_ion2});
      edge_window_counts.insert(edge_window_counts.end(), {
        record.n_plus, record.n_minus, record.n_deadband, record.attempts, record.successes,
        record.returns, record.geometry_lost});
    }
  }

  int bond_group = -1;
  int bond_edge_var = -1;
  int bond_value_var = -1;
  int bond_count_var = -1;
  std::vector<int> bond_edges;
  std::vector<double> bond_values;
  std::vector<long long> bond_counts;
  if (!total_bonds_.empty()) {
    const std::vector<const char*> bond_value_names = {"asymmetry", "abs_asymmetry", "mean_abs_delta"};
    const std::vector<const char*> bond_count_names = {
      "geometry_samples", "n_plus", "n_minus", "transitions"};
    netcdf_check(nc_def_grp(ncid, "bond", &bond_group), "nc_def_grp");
    const int row_dim = netcdf_dimension(bond_group, "row", total_bonds_.size());
    const int value_dim = netcdf_dimension(bond_group, "value", bond_value_names.size());
    const int count_dim = netcdf_dimension(bond_group, "count", bond_count_names.size());
    bond_edge_var = netcdf_variable(bond_group, "edge_id", NC_INT,
      {row_dim}, {total_bonds_.size()}, compression_level_);
    bond_value_var = netcdf_variable(bond_group, "value", NC_DOUBLE,
      {row_dim, value_dim}, {total_bonds_.size(), bond_value_names.size()}, compression_level_);
    bond_count_var = netcdf_variable(bond_group, "count", NC_INT64,
      {row_dim, count_dim}, {total_bonds_.size(), bond_count_names.size()}, compression_level_);
    netcdf_text_attribute(bond_group, bond_value_var, "field_names",
      "asymmetry,abs_asymmetry,mean_abs_delta");
    netcdf_text_attribute(bond_group, bond_count_var, "field_names",
      "geometry_samples,n_plus,n_minus,transitions");
    for (const auto& item : total_bonds_) {
      int oxygen_low;
      int oxygen_high;
      decode_bond_key(item.first, oxygen_low, oxygen_high);
      bond_edges.push_back(edge_ids[make_bond_key(oxygen_low, oxygen_high)]);
      const BondStats& stats = item.second;
      const long long biased_samples = stats.n_plus + stats.n_minus;
      const double asymmetry = biased_samples > 0
        ? static_cast<double>(stats.n_plus - stats.n_minus) / biased_samples :
        std::numeric_limits<double>::quiet_NaN();
      bond_values.insert(bond_values.end(), {
        asymmetry, std::abs(asymmetry),
        stats.geometry_samples > 0 ? stats.sum_abs_delta / stats.geometry_samples :
          std::numeric_limits<double>::quiet_NaN()});
      bond_counts.insert(bond_counts.end(), {
        stats.geometry_samples, stats.n_plus, stats.n_minus, stats.transitions});
    }
  }

  int local_window_group = -1;
  int local_window_edge_var = -1;
  int local_window_window_var = -1;
  int local_window_value_var = -1;
  int local_window_count_var = -1;
  std::vector<int> local_window_edges;
  std::vector<int> local_window_windows;
  std::vector<double> local_window_values;
  std::vector<long long> local_window_counts;
  const std::vector<const char*> local_window_value_names = {
    "mean_delta", "mean_rOH_low", "mean_rOH_high", "mean_oho_angle", "mean_dOO",
    "mean_rperp", "mean_path_excess", "mean_nH_low", "mean_nH_high",
    "mean_donor_edges_low", "mean_donor_edges_high", "mean_acceptor_edges_low",
    "mean_acceptor_edges_high", "mean_ion1_low_d1", "mean_ion1_low_d2", "mean_ion1_low_d3",
    "mean_ion1_high_d1", "mean_ion1_high_d2", "mean_ion1_high_d3", "mean_ion2_low_d1",
    "mean_ion2_low_d2", "mean_ion2_low_d3", "mean_ion2_high_d1", "mean_ion2_high_d2",
    "mean_ion2_high_d3", "mean_coord_ion1_low", "mean_coord_ion1_high",
    "mean_coord_ion2_low", "mean_coord_ion2_high", "mean_delta_d_ion1", "mean_delta_d_ion2",
    "mean_nearest_ion2_to_H", "mean_angle_Olow_H_ion2", "mean_angle_Ohigh_H_ion2",
    "fraction_hcl_like_low", "fraction_hcl_like_high", "mean_E_parallel",
    "mean_delta_phi_ion", "std_delta_phi_ion", "corr_delta_delta_phi"};
  const std::vector<const char*> local_window_count_names = {
    "samples", "attempts", "successes", "returns", "geometry_lost"};
  const auto local_mean = [](const double sum, const long long count) {
    return count > 0 ? sum / static_cast<double>(count) :
      std::numeric_limits<double>::quiet_NaN();
  };
  if (local_environment_enabled_ && !local_environment_window_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "local_environment_window", &local_window_group), "nc_def_grp");
    const int row_dim = netcdf_dimension(
      local_window_group, "row", local_environment_window_records_.size());
    const int value_dim = netcdf_dimension(
      local_window_group, "value", local_window_value_names.size());
    const int count_dim = netcdf_dimension(
      local_window_group, "count", local_window_count_names.size());
    local_window_edge_var = netcdf_variable(local_window_group, "edge_id", NC_INT,
      {row_dim}, {local_environment_window_records_.size()}, compression_level_);
    local_window_window_var = netcdf_variable(local_window_group, "window_index", NC_INT,
      {row_dim}, {local_environment_window_records_.size()}, compression_level_);
    local_window_value_var = netcdf_variable(local_window_group, "value", NC_DOUBLE,
      {row_dim, value_dim}, {local_environment_window_records_.size(), local_window_value_names.size()},
      compression_level_);
    local_window_count_var = netcdf_variable(local_window_group, "count", NC_INT64,
      {row_dim, count_dim}, {local_environment_window_records_.size(), local_window_count_names.size()},
      compression_level_);
    netcdf_text_attribute(local_window_group, local_window_value_var, "field_names",
      "mean_delta,mean_rOH_low,mean_rOH_high,mean_oho_angle,mean_dOO,mean_rperp,"
      "mean_path_excess,mean_nH_low,mean_nH_high,mean_donor_edges_low,mean_donor_edges_high,"
      "mean_acceptor_edges_low,mean_acceptor_edges_high,mean_ion1_low_d1,mean_ion1_low_d2,"
      "mean_ion1_low_d3,mean_ion1_high_d1,mean_ion1_high_d2,mean_ion1_high_d3,mean_ion2_low_d1,"
      "mean_ion2_low_d2,mean_ion2_low_d3,mean_ion2_high_d1,mean_ion2_high_d2,mean_ion2_high_d3,"
      "mean_coord_ion1_low,mean_coord_ion1_high,mean_coord_ion2_low,mean_coord_ion2_high,"
      "mean_delta_d_ion1,mean_delta_d_ion2,mean_nearest_ion2_to_H,mean_angle_Olow_H_ion2,"
      "mean_angle_Ohigh_H_ion2,fraction_hcl_like_low,fraction_hcl_like_high,mean_E_parallel,"
      "mean_delta_phi_ion,std_delta_phi_ion,corr_delta_delta_phi");
    netcdf_text_attribute(local_window_group, local_window_count_var, "field_names",
      "samples,attempts,successes,returns,geometry_lost");
    for (const LocalEnvironmentWindowRecord& record : local_environment_window_records_) {
      const LocalEnvironmentStats& stats = record.stats;
      const long long samples = stats.samples;
      const double mean_delta = local_mean(stats.sum_delta, samples);
      const double mean_phi = ion_field_enabled_ && samples > 0
        ? stats.sum_delta_phi_ion / samples : std::numeric_limits<double>::quiet_NaN();
      const double std_phi = ion_field_enabled_ && samples > 0
        ? std::sqrt(std::max(0.0, stats.sum_delta_phi2 / samples - mean_phi * mean_phi)) :
        std::numeric_limits<double>::quiet_NaN();
      double corr_phi = std::numeric_limits<double>::quiet_NaN();
      if (ion_field_enabled_ && samples > 0) {
        const double variance_delta = std::max(0.0, stats.sum_delta2 / samples - mean_delta * mean_delta);
        const double variance_phi = std::max(0.0, stats.sum_delta_phi2 / samples - mean_phi * mean_phi);
        if (variance_delta > 0.0 && variance_phi > 0.0) {
          const double covariance = stats.sum_delta_delta_phi / samples - mean_delta * mean_phi;
          corr_phi = covariance / std::sqrt(variance_delta * variance_phi);
        }
      }
      local_window_edges.push_back(edge_ids[make_bond_key(record.oxygen_low, record.oxygen_high)]);
      const auto window_item = window_ids.find(record.window_id);
      local_window_windows.push_back(window_item == window_ids.end() ? -1 : window_item->second);
      local_window_values.insert(local_window_values.end(), {
        mean_delta,
        local_mean(stats.sum_rOH_low, samples), local_mean(stats.sum_rOH_high, samples),
        local_mean(stats.sum_oho_angle, samples), local_mean(stats.sum_dOO, samples),
        local_mean(stats.sum_rperp, samples), local_mean(stats.sum_path_excess, samples),
        local_mean(static_cast<double>(stats.sum_nH_low), samples),
        local_mean(static_cast<double>(stats.sum_nH_high), samples),
        local_mean(static_cast<double>(stats.sum_donor_edges_low), samples),
        local_mean(static_cast<double>(stats.sum_donor_edges_high), samples),
        local_mean(static_cast<double>(stats.sum_acceptor_edges_low), samples),
        local_mean(static_cast<double>(stats.sum_acceptor_edges_high), samples),
        local_mean(stats.sum_ion1_low_d1, stats.count_ion1_low[0]),
        local_mean(stats.sum_ion1_low_d2, stats.count_ion1_low[1]),
        local_mean(stats.sum_ion1_low_d3, stats.count_ion1_low[2]),
        local_mean(stats.sum_ion1_high_d1, stats.count_ion1_high[0]),
        local_mean(stats.sum_ion1_high_d2, stats.count_ion1_high[1]),
        local_mean(stats.sum_ion1_high_d3, stats.count_ion1_high[2]),
        local_mean(stats.sum_ion2_low_d1, stats.count_ion2_low[0]),
        local_mean(stats.sum_ion2_low_d2, stats.count_ion2_low[1]),
        local_mean(stats.sum_ion2_low_d3, stats.count_ion2_low[2]),
        local_mean(stats.sum_ion2_high_d1, stats.count_ion2_high[0]),
        local_mean(stats.sum_ion2_high_d2, stats.count_ion2_high[1]),
        local_mean(stats.sum_ion2_high_d3, stats.count_ion2_high[2]),
        local_mean(static_cast<double>(stats.sum_coord_ion1_low), samples),
        local_mean(static_cast<double>(stats.sum_coord_ion1_high), samples),
        local_mean(static_cast<double>(stats.sum_coord_ion2_low), samples),
        local_mean(static_cast<double>(stats.sum_coord_ion2_high), samples),
        local_mean(stats.sum_delta_d_ion1, stats.count_delta_d_ion1),
        local_mean(stats.sum_delta_d_ion2, stats.count_delta_d_ion2),
        local_mean(stats.sum_nearest_ion2_to_H, stats.count_nearest_ion2_to_H),
        local_mean(stats.sum_angle_Olow_H_ion2, stats.count_angle_Olow_H_ion2),
        local_mean(stats.sum_angle_Ohigh_H_ion2, stats.count_angle_Ohigh_H_ion2),
        local_mean(static_cast<double>(stats.sum_hcl_like_low), samples),
        local_mean(static_cast<double>(stats.sum_hcl_like_high), samples),
        ion_field_enabled_ && samples > 0 ? stats.sum_E_parallel / samples :
          std::numeric_limits<double>::quiet_NaN(),
        mean_phi, std_phi, corr_phi});
      local_window_counts.insert(local_window_counts.end(), {
        stats.samples, stats.attempts, stats.successes, stats.returns, stats.geometry_lost});
    }
  }

  int attempt_group = -1;
  int attempt_time_var = -1;
  int attempt_hydrogen_var = -1;
  int attempt_edge_var = -1;
  int attempt_from_var = -1;
  int attempt_target_var = -1;
  int attempt_outcome_var = -1;
  int attempt_transfer_var = -1;
  int attempt_nearest_ion_var = -1;
  int attempt_value_var = -1;
  std::vector<double> attempt_times;
  std::vector<int> attempt_hydrogens;
  std::vector<int> attempt_edges;
  std::vector<int> attempt_from;
  std::vector<int> attempt_target;
  std::vector<unsigned char> attempt_outcomes;
  std::vector<unsigned char> attempt_has_transfer;
  std::vector<int> attempt_nearest_ions;
  std::vector<double> attempt_values;
  std::unordered_map<long long, int> attempt_rows;
  const std::vector<const char*> attempt_value_names = {
    "delta_start", "min_abs_delta", "delta_end", "E_parallel_start", "E_parallel_end",
    "delta_phi_start", "delta_phi_end", "delta_d_ion1_start", "delta_d_ion2_start",
    "nearest_ion_distance"};
  if (include_events && !attempt_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "attempt", &attempt_group), "nc_def_grp");
    const int attempt_dim = netcdf_dimension(attempt_group, "attempt", attempt_records_.size());
    const int endpoint_dim = netcdf_dimension(attempt_group, "endpoint", 2);
    const int value_dim = netcdf_dimension(attempt_group, "value", attempt_value_names.size());
    attempt_time_var = netcdf_variable(attempt_group, "time_fs", NC_DOUBLE,
      {attempt_dim, endpoint_dim}, {attempt_records_.size(), 2}, compression_level_);
    attempt_hydrogen_var = netcdf_variable(attempt_group, "hydrogen", NC_INT,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_edge_var = netcdf_variable(attempt_group, "edge_id", NC_INT,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_from_var = netcdf_variable(attempt_group, "oxygen_from", NC_INT,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_target_var = netcdf_variable(attempt_group, "oxygen_target", NC_INT,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_outcome_var = netcdf_variable(attempt_group, "outcome", NC_UBYTE,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_transfer_var = netcdf_variable(attempt_group, "has_transfer", NC_UBYTE,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_nearest_ion_var = netcdf_variable(attempt_group, "nearest_ion_id", NC_INT,
      {attempt_dim}, {attempt_records_.size()}, compression_level_);
    attempt_value_var = netcdf_variable(attempt_group, "value", NC_DOUBLE,
      {attempt_dim, value_dim}, {attempt_records_.size(), attempt_value_names.size()}, compression_level_);
    netcdf_text_attribute(attempt_group, attempt_time_var,
      "description", "start and final times; attempt row is the implicit attempt_id order");
    netcdf_text_attribute(attempt_group, attempt_outcome_var,
      "enum", "0=success,1=return,2=geometry_lost,3=run_end");
    netcdf_text_attribute(attempt_group, attempt_value_var, "field_names",
      "delta_start,min_abs_delta,delta_end,E_parallel_start,E_parallel_end,delta_phi_start,"
      "delta_phi_end,delta_d_ion1_start,delta_d_ion2_start,nearest_ion_distance");
    attempt_times.reserve(attempt_records_.size() * 2);
    attempt_hydrogens.reserve(attempt_records_.size());
    attempt_edges.reserve(attempt_records_.size());
    attempt_from.reserve(attempt_records_.size());
    attempt_target.reserve(attempt_records_.size());
    attempt_outcomes.reserve(attempt_records_.size());
    attempt_has_transfer.reserve(attempt_records_.size());
    attempt_nearest_ions.reserve(attempt_records_.size());
    attempt_values.reserve(attempt_records_.size() * attempt_value_names.size());
    for (size_t row = 0; row < attempt_records_.size(); ++row) {
      const AttemptRecord& record = attempt_records_[row];
      attempt_rows[record.attempt_id] = static_cast<int>(row);
      attempt_times.push_back(record.time_start_fs);
      attempt_times.push_back(record.time_end_fs);
      attempt_hydrogens.push_back(record.hydrogen);
      attempt_edges.push_back(edge_ids[make_bond_key(record.oxygen_low, record.oxygen_high)]);
      attempt_from.push_back(record.oxygen_from);
      attempt_target.push_back(record.oxygen_target);
      const auto outcome_code = [&](const AttemptOutcome outcome) {
        switch (outcome) {
          case AttemptOutcome::success: return static_cast<unsigned char>(0);
          case AttemptOutcome::return_to_state: return static_cast<unsigned char>(1);
          case AttemptOutcome::geometry_lost: return static_cast<unsigned char>(2);
          case AttemptOutcome::run_end: return static_cast<unsigned char>(3);
        }
        return static_cast<unsigned char>(3);
      };
      attempt_outcomes.push_back(outcome_code(record.outcome));
      attempt_has_transfer.push_back(record.has_transfer ? 1 : 0);
      attempt_nearest_ions.push_back(record.nearest_ion_id);
      attempt_values.insert(attempt_values.end(), {
        record.delta_start, record.min_abs_delta, record.delta_end,
        record.E_parallel_start, record.E_parallel_end, record.delta_phi_start,
        record.delta_phi_end, record.delta_d_ion1_start, record.delta_d_ion2_start,
        record.nearest_ion_distance});
    }
  }

  int transfer_group = -1;
  int transfer_attempt_var = -1;
  int transfer_value_var = -1;
  int transfer_count_var = -1;
  std::vector<int> transfer_attempts;
  std::vector<double> transfer_values;
  std::vector<long long> transfer_counts;
  if (include_events && !attempt_records_.empty()) {
    size_t transfer_count = 0;
    for (const AttemptRecord& record : attempt_records_)
      if (record.has_transfer)
        ++transfer_count;
    if (transfer_count > 0) {
      const std::vector<const char*> transfer_value_names = {
        "dx", "dy", "dz", "delta_start", "delta_confirm"};
      const std::vector<const char*> transfer_count_names = {
        "nH_from_before", "nH_to_before", "nH_from_after", "nH_to_after"};
      netcdf_check(nc_def_grp(ncid, "transfer", &transfer_group), "nc_def_grp");
      const int row_dim = netcdf_dimension(transfer_group, "row", transfer_count);
      const int value_dim = netcdf_dimension(transfer_group, "value", transfer_value_names.size());
      const int count_dim = netcdf_dimension(transfer_group, "count", transfer_count_names.size());
      transfer_attempt_var = netcdf_variable(transfer_group, "attempt_index", NC_INT,
        {row_dim}, {transfer_count}, compression_level_);
      transfer_value_var = netcdf_variable(transfer_group, "value", NC_DOUBLE,
        {row_dim, value_dim}, {transfer_count, transfer_value_names.size()}, compression_level_);
      transfer_count_var = netcdf_variable(transfer_group, "count", NC_INT64,
        {row_dim, count_dim}, {transfer_count, transfer_count_names.size()}, compression_level_);
      netcdf_text_attribute(transfer_group, transfer_attempt_var,
        "description", "zero-based row index into /attempt");
      netcdf_text_attribute(transfer_group, transfer_value_var,
        "field_names", "dx,dy,dz,delta_start,delta_confirm");
      netcdf_text_attribute(transfer_group, transfer_count_var, "field_names",
        "nH_from_before,nH_to_before,nH_from_after,nH_to_after");
      transfer_attempts.reserve(transfer_count);
      transfer_values.reserve(transfer_count * transfer_value_names.size());
      transfer_counts.reserve(transfer_count * transfer_count_names.size());
      for (const AttemptRecord& record : attempt_records_) {
        if (!record.has_transfer)
          continue;
        transfer_attempts.push_back(attempt_rows[record.attempt_id]);
        transfer_values.insert(transfer_values.end(), {
          record.dx, record.dy, record.dz, record.delta_start, record.delta_end});
        transfer_counts.insert(transfer_counts.end(), {
          record.nH_from_before, record.nH_to_before,
          record.nH_from_after, record.nH_to_after});
      }
    }
  }

  int defect_group = -1;
  int defect_time_var = -1;
  int defect_oxygen_var = -1;
  int defect_hydrogen_count_var = -1;
  int defect_cause_var = -1;
  std::vector<double> defect_times;
  std::vector<int> defect_oxygens;
  std::vector<int> defect_hydrogen_counts;
  std::vector<long long> defect_causes;
  if (include_events && !defect_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "defect", &defect_group), "nc_def_grp");
    const int row_dim = netcdf_dimension(defect_group, "row", defect_records_.size());
    defect_time_var = netcdf_variable(defect_group, "time_fs", NC_DOUBLE,
      {row_dim}, {defect_records_.size()}, compression_level_);
    defect_oxygen_var = netcdf_variable(defect_group, "oxygen", NC_INT,
      {row_dim}, {defect_records_.size()}, compression_level_);
    defect_hydrogen_count_var = netcdf_variable(defect_group, "hydrogen_count", NC_INT,
      {row_dim}, {defect_records_.size()}, compression_level_);
    defect_cause_var = netcdf_variable(defect_group, "cause_event_id", NC_INT64,
      {row_dim}, {defect_records_.size()}, compression_level_);
    netcdf_text_attribute(defect_group, defect_cause_var,
      "description", "one-based attempt_id; 0 is the initial state and -1 means unassigned");
    for (const DefectRecord& record : defect_records_) {
      defect_times.push_back(record.time_fs);
      defect_oxygens.push_back(record.oxygen);
      defect_hydrogen_counts.push_back(record.hydrogen_count);
      defect_causes.push_back(record.cause_event_id);
    }
  }

  int bead_group = -1;
  int bead_valid_var = -1;
  int bead_class_var = -1;
  int bead_value_var = -1;
  int bead_count_var = -1;
  int bead_flag_var = -1;
  std::vector<unsigned char> bead_valid;
  std::vector<unsigned char> bead_classes;
  std::vector<double> bead_values;
  std::vector<int> bead_counts;
  std::vector<unsigned char> bead_flags;
  const std::vector<const char*> bead_value_names = {
    "probe_time_fs", "delta_centroid", "f_minus", "f_zero", "f_plus", "f_channel_valid",
    "mean_delta", "centroid_minus_mean", "sigma_delta", "delta_min", "delta_max", "span",
    "delta_q20", "delta_q80", "robust_span", "rms_neighbor_delta_jump",
    "max_neighbor_delta_jump"};
  const std::vector<const char*> bead_count_names = {
    "num_beads", "n_minus", "n_zero", "n_plus", "kink_count", "center_domain_count",
    "total_state_domain_count", "channel_valid_count"};
  const std::vector<const char*> bead_flag_names = {
    "two_well_occupied", "two_well_span", "simple_two_domain_path", "barrier_centered",
    "strict_tunneling_like", "multi_kink_or_multi_domain"};
  if (include_events && bead_diagnostic_enabled_ && !attempt_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "bead", &bead_group), "nc_def_grp");
    const int attempt_dim = netcdf_dimension(bead_group, "attempt", attempt_records_.size());
    const int selection_dim = netcdf_dimension(bead_group, "selection", 2);
    const int value_dim = netcdf_dimension(bead_group, "value", bead_value_names.size());
    const int count_dim = netcdf_dimension(bead_group, "count", bead_count_names.size());
    const int flag_dim = netcdf_dimension(bead_group, "flag", bead_flag_names.size());
    bead_valid_var = netcdf_variable(bead_group, "valid", NC_UBYTE,
      {attempt_dim, selection_dim}, {attempt_records_.size(), 2}, compression_level_);
    bead_class_var = netcdf_variable(bead_group, "quantum_class", NC_UBYTE,
      {attempt_dim, selection_dim}, {attempt_records_.size(), 2}, compression_level_);
    bead_value_var = netcdf_variable(bead_group, "value", NC_DOUBLE,
      {attempt_dim, selection_dim, value_dim},
      {attempt_records_.size(), 2, bead_value_names.size()}, compression_level_);
    bead_count_var = netcdf_variable(bead_group, "count", NC_INT,
      {attempt_dim, selection_dim, count_dim},
      {attempt_records_.size(), 2, bead_count_names.size()}, compression_level_);
    bead_flag_var = netcdf_variable(bead_group, "flag", NC_UBYTE,
      {attempt_dim, selection_dim, flag_dim},
      {attempt_records_.size(), 2, bead_flag_names.size()}, compression_level_);
    netcdf_text_attribute(bead_group, bead_valid_var,
      "description", "selection index 0=centroid_best, 1=delocalization_best");
    netcdf_text_attribute(bead_group, bead_class_var, "enum",
      "0=classical_only,1=two_well_delocalized,2=barrier_centered_tunneling_like,"
      "3=compact_single_domain,4=multi_kink_or_multi_domain,5=ambiguous");
    netcdf_text_attribute(bead_group, bead_value_var, "field_names",
      "probe_time_fs,delta_centroid,f_minus,f_zero,f_plus,f_channel_valid,mean_delta,"
      "centroid_minus_mean,sigma_delta,delta_min,delta_max,span,delta_q20,delta_q80,"
      "robust_span,rms_neighbor_delta_jump,max_neighbor_delta_jump");
    netcdf_text_attribute(bead_group, bead_count_var, "field_names",
      "num_beads,n_minus,n_zero,n_plus,kink_count,center_domain_count,total_state_domain_count,"
      "channel_valid_count");
    netcdf_text_attribute(bead_group, bead_flag_var, "field_names",
      "two_well_occupied,two_well_span,simple_two_domain_path,barrier_centered,"
      "strict_tunneling_like,multi_kink_or_multi_domain");
    const auto class_code = [](const QuantumCharacter character) {
      switch (character) {
        case QuantumCharacter::CLASSICAL_ONLY: return static_cast<unsigned char>(0);
        case QuantumCharacter::TWO_WELL_DELOCALIZED: return static_cast<unsigned char>(1);
        case QuantumCharacter::BARRIER_CENTERED_TUNNELING_LIKE: return static_cast<unsigned char>(2);
        case QuantumCharacter::COMPACT_SINGLE_DOMAIN: return static_cast<unsigned char>(3);
        case QuantumCharacter::MULTI_KINK_OR_MULTI_DOMAIN: return static_cast<unsigned char>(4);
        case QuantumCharacter::AMBIGUOUS: return static_cast<unsigned char>(5);
      }
      return static_cast<unsigned char>(5);
    };
    const auto append_bead = [&](const BeadDiagnostic& diagnostic) {
      const bool valid = diagnostic.valid;
      const double nan = std::numeric_limits<double>::quiet_NaN();
      bead_valid.push_back(valid ? 1 : 0);
      bead_classes.push_back(valid ? class_code(diagnostic.character) : 5);
      bead_values.insert(bead_values.end(), {
        valid ? diagnostic.probe_time_fs : nan,
        valid ? diagnostic.delta_centroid : nan,
        valid ? diagnostic.f_minus : nan,
        valid ? diagnostic.f_zero : nan,
        valid ? diagnostic.f_plus : nan,
        valid ? diagnostic.f_channel_valid : nan,
        valid ? diagnostic.mean_delta : nan,
        valid ? diagnostic.centroid_minus_mean : nan,
        valid ? diagnostic.sigma_delta : nan,
        valid ? diagnostic.delta_min : nan,
        valid ? diagnostic.delta_max : nan,
        valid ? diagnostic.span : nan,
        valid ? diagnostic.delta_q20 : nan,
        valid ? diagnostic.delta_q80 : nan,
        valid ? diagnostic.robust_span : nan,
        valid ? diagnostic.rms_neighbor_delta_jump : nan,
        valid ? diagnostic.max_neighbor_delta_jump : nan});
      bead_counts.insert(bead_counts.end(), {
        valid ? diagnostic.num_beads : 0,
        valid ? diagnostic.n_minus : -1,
        valid ? diagnostic.n_zero : -1,
        valid ? diagnostic.n_plus : -1,
        valid ? diagnostic.kink_count : -1,
        valid ? diagnostic.center_domain_count : -1,
        valid ? diagnostic.total_state_domain_count : -1,
        valid ? diagnostic.channel_valid_count : -1});
      bead_flags.insert(bead_flags.end(), {
        static_cast<unsigned char>(valid ? diagnostic.two_well_occupied : 0),
        static_cast<unsigned char>(valid ? diagnostic.two_well_span : 0),
        static_cast<unsigned char>(valid ? diagnostic.simple_two_domain_path : 0),
        static_cast<unsigned char>(valid ? diagnostic.barrier_centered : 0),
        static_cast<unsigned char>(valid ? diagnostic.strict_tunneling_like : 0),
        static_cast<unsigned char>(valid ? diagnostic.multi_kink_or_multi_domain : 0)});
    };
    bead_valid.reserve(attempt_records_.size() * 2);
    bead_classes.reserve(attempt_records_.size() * 2);
    bead_values.reserve(attempt_records_.size() * 2 * bead_value_names.size());
    bead_counts.reserve(attempt_records_.size() * 2 * bead_count_names.size());
    bead_flags.reserve(attempt_records_.size() * 2 * bead_flag_names.size());
    for (const AttemptRecord& record : attempt_records_) {
      append_bead(record.centroid_best);
      append_bead(record.delocalization_best);
    }
  }

  int local_event_group = -1;
  int local_event_time_var = -1;
  int local_event_valid_var = -1;
  int local_event_class_var = -1;
  int local_event_value_var = -1;
  int local_event_count_var = -1;
  std::vector<double> local_event_times;
  std::vector<unsigned char> local_event_valid;
  std::vector<unsigned char> local_event_classes;
  std::vector<double> local_event_values;
  std::vector<int> local_event_counts;
  const std::vector<const char*> local_event_value_names = {
    "rOH_low", "rOH_high", "oho_angle", "dOO", "rperp", "path_excess",
    "ion1_low_d1", "ion1_low_d2", "ion1_low_d3", "ion1_high_d1", "ion1_high_d2",
    "ion1_high_d3", "ion2_low_d1", "ion2_low_d2", "ion2_low_d3", "ion2_high_d1",
    "ion2_high_d2", "ion2_high_d3", "delta_d_ion1", "delta_d_ion2", "nearest_ion2_to_H",
    "angle_Olow_H_ion2", "angle_Ohigh_H_ion2", "E_parallel", "delta_phi_ion"};
  const std::vector<const char*> local_event_count_names = {
    "nH_low", "nH_high", "donor_edges_low", "donor_edges_high", "acceptor_edges_low",
    "acceptor_edges_high", "coord_ion1_low", "coord_ion1_high", "coord_ion2_low",
    "coord_ion2_high", "hcl_like_low", "hcl_like_high"};
  if (include_full && local_environment_enabled_ && !attempt_records_.empty()) {
    netcdf_check(nc_def_grp(ncid, "local_environment_event", &local_event_group), "nc_def_grp");
    const int attempt_dim = netcdf_dimension(local_event_group, "attempt", attempt_records_.size());
    const int snapshot_dim = netcdf_dimension(local_event_group, "snapshot", 5);
    const int value_dim = netcdf_dimension(local_event_group, "value", local_event_value_names.size());
    const int count_dim = netcdf_dimension(local_event_group, "count", local_event_count_names.size());
    local_event_time_var = netcdf_variable(local_event_group, "time_fs", NC_DOUBLE,
      {attempt_dim, snapshot_dim}, {attempt_records_.size(), 5}, compression_level_);
    local_event_valid_var = netcdf_variable(local_event_group, "valid", NC_UBYTE,
      {attempt_dim, snapshot_dim}, {attempt_records_.size(), 5}, compression_level_);
    local_event_class_var = netcdf_variable(local_event_group, "quantum_class", NC_UBYTE,
      {attempt_dim, snapshot_dim}, {attempt_records_.size(), 5}, compression_level_);
    local_event_value_var = netcdf_variable(local_event_group, "value", NC_DOUBLE,
      {attempt_dim, snapshot_dim, value_dim},
      {attempt_records_.size(), 5, local_event_value_names.size()}, compression_level_);
    local_event_count_var = netcdf_variable(local_event_group, "count", NC_INT,
      {attempt_dim, snapshot_dim, count_dim},
      {attempt_records_.size(), 5, local_event_count_names.size()}, compression_level_);
    netcdf_text_attribute(local_event_group, local_event_time_var,
      "snapshot_names", "0=start,1=end,2=last_valid,3=centroid_best,4=delocalization_best");
    netcdf_text_attribute(local_event_group, local_event_valid_var,
      "description", "selected snapshots are controlled by the global snapshot_mode attribute");
    netcdf_text_attribute(local_event_group, local_event_class_var, "enum",
      "0=classical_only,1=two_well_delocalized,2=barrier_centered_tunneling_like,"
      "3=compact_single_domain,4=multi_kink_or_multi_domain,5=ambiguous,255=not_applicable");
    netcdf_text_attribute(local_event_group, local_event_value_var, "field_names",
      "rOH_low,rOH_high,oho_angle,dOO,rperp,path_excess,ion1_low_d1,ion1_low_d2,ion1_low_d3,"
      "ion1_high_d1,ion1_high_d2,ion1_high_d3,ion2_low_d1,ion2_low_d2,ion2_low_d3,"
      "ion2_high_d1,ion2_high_d2,ion2_high_d3,delta_d_ion1,delta_d_ion2,nearest_ion2_to_H,"
      "angle_Olow_H_ion2,angle_Ohigh_H_ion2,E_parallel,delta_phi_ion");
    netcdf_text_attribute(local_event_group, local_event_count_var, "field_names",
      "nH_low,nH_high,donor_edges_low,donor_edges_high,acceptor_edges_low,acceptor_edges_high,"
      "coord_ion1_low,coord_ion1_high,coord_ion2_low,coord_ion2_high,hcl_like_low,hcl_like_high");
    local_event_times.reserve(attempt_records_.size() * 5);
    local_event_valid.reserve(attempt_records_.size() * 5);
    local_event_classes.reserve(attempt_records_.size() * 5);
    local_event_values.reserve(attempt_records_.size() * 5 * local_event_value_names.size());
    local_event_counts.reserve(attempt_records_.size() * 5 * local_event_count_names.size());
    const auto class_code = [](const QuantumCharacter character) {
      switch (character) {
        case QuantumCharacter::CLASSICAL_ONLY: return static_cast<unsigned char>(0);
        case QuantumCharacter::TWO_WELL_DELOCALIZED: return static_cast<unsigned char>(1);
        case QuantumCharacter::BARRIER_CENTERED_TUNNELING_LIKE: return static_cast<unsigned char>(2);
        case QuantumCharacter::COMPACT_SINGLE_DOMAIN: return static_cast<unsigned char>(3);
        case QuantumCharacter::MULTI_KINK_OR_MULTI_DOMAIN: return static_cast<unsigned char>(4);
        case QuantumCharacter::AMBIGUOUS: return static_cast<unsigned char>(5);
      }
      return static_cast<unsigned char>(255);
    };
    for (const AttemptRecord& record : attempt_records_) {
      const LocalEnvironment* environments[5] = {
        &record.environment_start, &record.environment_end, &record.environment_last_valid,
        &record.centroid_best.environment, &record.delocalization_best.environment};
      const double times[5] = {
        record.time_start_fs, record.time_end_fs, record.time_end_fs,
        record.centroid_best.probe_time_fs, record.delocalization_best.probe_time_fs};
      const unsigned char classes[5] = {
        255, 255, 255,
        class_code(record.centroid_best.character), class_code(record.delocalization_best.character)};
      for (int snapshot = 0; snapshot < 5; ++snapshot) {
        const bool selected = snapshot_mode_ == SnapshotMode::ALL ||
          (snapshot_mode_ == SnapshotMode::BEST && snapshot != 2) ||
          (snapshot_mode_ == SnapshotMode::ENDPOINTS && snapshot < 2);
        const LocalEnvironment& environment = *environments[snapshot];
        const bool valid = selected && environment.valid;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        local_event_times.push_back(selected ? times[snapshot] : nan);
        local_event_valid.push_back(valid ? 1 : 0);
        local_event_classes.push_back(valid ? classes[snapshot] : 255);
        local_event_values.insert(local_event_values.end(), {
          valid ? environment.rOH_low : nan, valid ? environment.rOH_high : nan,
          valid ? environment.oho_angle : nan, valid ? environment.dOO : nan,
          valid ? environment.rperp : nan, valid ? environment.path_excess : nan,
          valid ? environment.ion1_low_d1 : nan, valid ? environment.ion1_low_d2 : nan,
          valid ? environment.ion1_low_d3 : nan, valid ? environment.ion1_high_d1 : nan,
          valid ? environment.ion1_high_d2 : nan, valid ? environment.ion1_high_d3 : nan,
          valid ? environment.ion2_low_d1 : nan, valid ? environment.ion2_low_d2 : nan,
          valid ? environment.ion2_low_d3 : nan, valid ? environment.ion2_high_d1 : nan,
          valid ? environment.ion2_high_d2 : nan, valid ? environment.ion2_high_d3 : nan,
          valid ? environment.delta_d_ion1 : nan, valid ? environment.delta_d_ion2 : nan,
          valid ? environment.nearest_ion2_to_H : nan,
          valid ? environment.angle_Olow_H_ion2 : nan,
          valid ? environment.angle_Ohigh_H_ion2 : nan,
          valid ? environment.E_parallel : nan, valid ? environment.delta_phi_ion : nan});
        local_event_counts.insert(local_event_counts.end(), {
          valid ? environment.nH_low : -1, valid ? environment.nH_high : -1,
          valid ? environment.donor_edges_low : -1, valid ? environment.donor_edges_high : -1,
          valid ? environment.acceptor_edges_low : -1,
          valid ? environment.acceptor_edges_high : -1,
          valid ? environment.coord_ion1_low : -1, valid ? environment.coord_ion1_high : -1,
          valid ? environment.coord_ion2_low : -1, valid ? environment.coord_ion2_high : -1,
          valid ? environment.hcl_like_low : -1, valid ? environment.hcl_like_high : -1});
      }
    }
  }

  netcdf_check(nc_enddef(ncid), "nc_enddef");
  if (edge_group >= 0) {
    std::vector<int> edge_oxygen;
    edge_oxygen.reserve(edge_keys.size() * 2);
    for (const unsigned long long key : edge_keys) {
      int oxygen_low;
      int oxygen_high;
      decode_bond_key(key, oxygen_low, oxygen_high);
      edge_oxygen.push_back(oxygen_low);
      edge_oxygen.push_back(oxygen_high);
    }
    netcdf_write_int(edge_group, edge_oxygen_var, edge_oxygen);
  }
  if (window_group >= 0) {
    netcdf_write_double(window_group, window_time_var, window_times);
    netcdf_write_double(window_group, window_value_var, window_values);
    netcdf_write_longlong(window_group, window_count_var, window_counts);
  }
  if (edge_window_group >= 0) {
    netcdf_write_int(edge_window_group, edge_window_edge_var, edge_window_edges);
    netcdf_write_int(edge_window_group, edge_window_window_var, edge_window_windows);
    netcdf_write_double(edge_window_group, edge_window_value_var, edge_window_values);
    netcdf_write_longlong(edge_window_group, edge_window_count_var, edge_window_counts);
  }
  if (bond_group >= 0) {
    netcdf_write_int(bond_group, bond_edge_var, bond_edges);
    netcdf_write_double(bond_group, bond_value_var, bond_values);
    netcdf_write_longlong(bond_group, bond_count_var, bond_counts);
  }
  if (local_window_group >= 0) {
    netcdf_write_int(local_window_group, local_window_edge_var, local_window_edges);
    netcdf_write_int(local_window_group, local_window_window_var, local_window_windows);
    netcdf_write_double(local_window_group, local_window_value_var, local_window_values);
    netcdf_write_longlong(local_window_group, local_window_count_var, local_window_counts);
  }
  if (attempt_group >= 0) {
    netcdf_write_double(attempt_group, attempt_time_var, attempt_times);
    netcdf_write_int(attempt_group, attempt_hydrogen_var, attempt_hydrogens);
    netcdf_write_int(attempt_group, attempt_edge_var, attempt_edges);
    netcdf_write_int(attempt_group, attempt_from_var, attempt_from);
    netcdf_write_int(attempt_group, attempt_target_var, attempt_target);
    netcdf_write_ubyte(attempt_group, attempt_outcome_var, attempt_outcomes);
    netcdf_write_ubyte(attempt_group, attempt_transfer_var, attempt_has_transfer);
    netcdf_write_int(attempt_group, attempt_nearest_ion_var, attempt_nearest_ions);
    netcdf_write_double(attempt_group, attempt_value_var, attempt_values);
  }
  if (transfer_group >= 0) {
    netcdf_write_int(transfer_group, transfer_attempt_var, transfer_attempts);
    netcdf_write_double(transfer_group, transfer_value_var, transfer_values);
    netcdf_write_longlong(transfer_group, transfer_count_var, transfer_counts);
  }
  if (defect_group >= 0) {
    netcdf_write_double(defect_group, defect_time_var, defect_times);
    netcdf_write_int(defect_group, defect_oxygen_var, defect_oxygens);
    netcdf_write_int(defect_group, defect_hydrogen_count_var, defect_hydrogen_counts);
    netcdf_write_longlong(defect_group, defect_cause_var, defect_causes);
  }
  if (bead_group >= 0) {
    netcdf_write_ubyte(bead_group, bead_valid_var, bead_valid);
    netcdf_write_ubyte(bead_group, bead_class_var, bead_classes);
    netcdf_write_double(bead_group, bead_value_var, bead_values);
    netcdf_write_int(bead_group, bead_count_var, bead_counts);
    netcdf_write_ubyte(bead_group, bead_flag_var, bead_flags);
  }
  if (local_event_group >= 0) {
    netcdf_write_double(local_event_group, local_event_time_var, local_event_times);
    netcdf_write_ubyte(local_event_group, local_event_valid_var, local_event_valid);
    netcdf_write_ubyte(local_event_group, local_event_class_var, local_event_classes);
    netcdf_write_double(local_event_group, local_event_value_var, local_event_values);
    netcdf_write_int(local_event_group, local_event_count_var, local_event_counts);
  }
  netcdf_check(nc_close(ncid), "nc_close");
  printf("Proton observer wrote compressed NetCDF output %s (level=%s, deflate=%d).\n",
    output_filename_.c_str(), output_level, compression_level_);
#endif
}

void Proton_Tunneling::write_output_files()
{
  if (output_format_ == OutputFormat::NETCDF)
    write_netcdf_output_file();
  else
    write_text_output_files();
}

void Proton_Tunneling::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  (void)atom;
  (void)box;
  (void)integrate;
  (void)number_of_steps;
  (void)time_step;
  (void)temperature;
  if (!initialized_)
    return;

  for (size_t h_index = 0; h_index < hydrogen_states_.size(); ++h_index) {
    HydrogenState& hydrogen_state = hydrogen_states_[h_index];
    if (hydrogen_state.attempt_active) {
      finish_attempt(
        hydrogen_indices_[h_index],
        hydrogen_state,
        AttemptOutcome::run_end,
        last_time_fs_,
        hydrogen_state.last_delta,
        nullptr,
        nullptr);
    }
  }
  if (window_sample_count_ > 0)
    write_window(last_time_fs_);
  write_output_files();
  release_geometry_timing_events();
  printf("Proton tunneling observer timing:\n");
  printf("    sampled observer frames: %lld\n", observer_frame_count_);
  if (bead_diagnostic_enabled_) {
    printf("    bead probe frames: %lld\n", bead_probe_frame_count_);
    printf("    packed bead D2H copies: %lld\n", bead_d2h_copy_count_);
    printf("    bead bytes copied: %llu\n", bead_bytes_copied_);
    printf("    bead pack + D2H wall time: %.6f s\n", bead_copy_wall_time_);
    printf("    bead analysis wall time: %.6f s\n", bead_analysis_wall_time_);
  }
  printf("    geometry kernel wall time: %.6f s\n", geometry_kernel_wall_time_);
  printf("    geometry D2H + host copy wall time: %.6f s\n", geometry_D2H_wall_time_);
  if (local_environment_enabled_) {
    printf("    local environment copies: %lld\n", local_environment_copy_count_);
    printf("    local environment bytes copied: %llu\n", local_environment_bytes_copied_);
    printf("    local environment kernel wall time: %.6f s\n",
      local_environment_kernel_wall_time_);
    printf("    local environment D2H wall time: %.6f s\n",
      local_environment_D2H_wall_time_);
    printf("    local environment host analysis wall time: %.6f s\n",
      local_environment_host_analysis_wall_time_);
  }
  printf("    CPU state-machine wall time: %.6f s\n", state_machine_wall_time_);
  printf("    total observer wall time: %.6f s\n", total_observer_wall_time_);
  initialized_ = false;
}
