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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace
{
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
  geometry.nearest_ion_id = -1;
  geometry.nearest_ion_distance = 1.0e300;
  geometry.nearest_ion1_distance = 1.0e300;
  geometry.nearest_ion2_distance = 1.0e300;
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
      const double distance_square = midpoint_to_ion_x * midpoint_to_ion_x +
        midpoint_to_ion_y * midpoint_to_ion_y + midpoint_to_ion_z * midpoint_to_ion_z;
      const double distance = sqrt(distance_square);
      if (distance < nearest_species_distance)
        nearest_species_distance = distance;
      if (distance < geometry.nearest_ion_distance) {
        geometry.nearest_ion_distance = distance;
        geometry.nearest_ion_id = ion;
      }
      if (distance <= ion_field_cutoff && distance > 1.0e-12) {
        geometry.E_ion_nominal_parallel += K_C * charge *
          (midpoint_to_ion_x * ex + midpoint_to_ion_y * ey + midpoint_to_ion_z * ez) /
          (distance_square * distance);
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
      std::strcmp(value, "bead_diagnostic") == 0;
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
    if (ion_field_enabled_ && atom.cpu_atom_symbol[i] == ion1_symbol_)
      ion1_indices_.push_back(i);
    if (ion_field_enabled_ && atom.cpu_atom_symbol[i] == ion2_symbol_)
      ion2_indices_.push_back(i);
  }
  if (oxygen_indices_.empty() || hydrogen_indices_.empty()) {
    PRINT_INPUT_ERROR("compute_proton_tunneling could not find the requested O and H species.");
  }
  if (ion_field_enabled_ && (ion1_indices_.empty() || ion2_indices_.empty())) {
    PRINT_INPUT_ERROR("ion_field could not find both requested ion species.");
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
  if (bead_diagnostic_enabled_) {
    printf("    strict bead diagnostic uses f_min %.6f, center_max %.6f, centroid_max %.6f, "
           "and span_min %.6f Angstrom.\n",
      bead_f_min_, bead_center_max_, bead_centroid_max_, bead_span_min_);
  }
  if (ion_field_enabled_) {
    printf("    nominal ion field uses %s charge %.6f and %s charge %.6f within %.6f Angstrom.\n",
      ion1_symbol_.c_str(), ion1_charge_, ion2_symbol_.c_str(), ion2_charge_, ion_field_cutoff_);
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
  attempt_records_.reserve(1024);
  defect_records_.reserve(1024);
  window_records_.reserve(64);
  edge_window_records_.reserve(1024);
  atom.position_per_atom.copy_to_host(cpu_position_.data());
  build_oxygen_shell(box);
  initialize_geometry_gpu();
  CHECK(gpuEventCreate(&geometry_kernel_start_event_));
  CHECK(gpuEventCreate(&geometry_kernel_end_event_));
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
  if (ion_field_enabled_) {
    ion1_indices_gpu_.resize(ion1_indices_.size());
    ion1_indices_gpu_.copy_from_host(ion1_indices_.data());
    ion2_indices_gpu_.resize(ion2_indices_.size());
    ion2_indices_gpu_.copy_from_host(ion2_indices_.data());
  }
  frame_geometries_gpu_.resize(frame_geometries_.size());
  cpu_geometry_gpu_.resize(frame_geometries_.size());
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
    target.nearest_ion_id = source.nearest_ion_id;
    target.nearest_ion_distance = source.nearest_ion_distance;
    target.nearest_ion1_distance = source.nearest_ion1_distance;
    target.nearest_ion2_distance = source.nearest_ion2_distance;
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
  geometry.nearest_ion_distance = std::numeric_limits<double>::max();
  geometry.nearest_ion1_distance = std::numeric_limits<double>::max();
  geometry.nearest_ion2_distance = std::numeric_limits<double>::max();
  auto accumulate_ion_field = [&](const std::vector<int>& ions, const double charge,
                                  double& nearest_species_distance) {
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
      const double distance_square = midpoint_to_ion_x * midpoint_to_ion_x +
        midpoint_to_ion_y * midpoint_to_ion_y + midpoint_to_ion_z * midpoint_to_ion_z;
      const double distance = std::sqrt(distance_square);
      nearest_species_distance = std::min(nearest_species_distance, distance);
      if (distance < geometry.nearest_ion_distance) {
        geometry.nearest_ion_distance = distance;
        geometry.nearest_ion_id = ion;
      }
      if (distance <= ion_field_cutoff_ && distance > 1.0e-12) {
        geometry.E_ion_nominal_parallel += K_C * charge *
          (midpoint_to_ion_x * ex + midpoint_to_ion_y * ey + midpoint_to_ion_z * ez) /
          (distance_square * distance);
      }
    }
  };
  accumulate_ion_field(ion1_indices_, ion1_charge_, geometry.nearest_ion1_distance);
  accumulate_ion_field(ion2_indices_, ion2_charge_, geometry.nearest_ion2_distance);
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

Proton_Tunneling::QuantumCharacter Proton_Tunneling::classify_quantum_character(
  const BeadDiagnostic& diagnostic) const
{
  if (diagnostic.num_beads <= 1)
    return QuantumCharacter::CLASSICAL_ONLY;
  if (!diagnostic.valid)
    return QuantumCharacter::AMBIGUOUS;

  const bool two_well_delocalized = diagnostic.two_well_span != 0 &&
    diagnostic.f_zero <= bead_center_max_ && diagnostic.simple_two_domain_path != 0;
  if (diagnostic.barrier_centered != 0)
    return QuantumCharacter::BARRIER_CENTERED_TUNNELING_LIKE;
  if (two_well_delocalized)
    return QuantumCharacter::TWO_WELL_DELOCALIZED;

  const int dominant_count = std::max(
    diagnostic.n_zero, std::max(diagnostic.n_minus, diagnostic.n_plus));
  if (diagnostic.kink_count == 0 &&
      static_cast<double>(dominant_count) / diagnostic.num_beads >= 1.0 - bead_f_min_)
    return QuantumCharacter::OVERBARRIER_LIKE;

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
  case QuantumCharacter::OVERBARRIER_LIKE:
    return "overbarrier_like";
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
  const auto bead_delta = [&](const int bead) {
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
    return std::sqrt(low_x * low_x + low_y * low_y + low_z * low_z) -
      std::sqrt(high_x * high_x + high_y * high_y + high_z * high_z);
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
    const double delta = bead_delta(bead);
    bead_deltas[bead] = delta;
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
  diagnostic.rms_neighbor_delta_jump = std::sqrt(
    sum_neighbor_delta_jump_square * inverse_num_beads);
  diagnostic.two_well_span =
    (diagnostic.f_minus >= bead_f_min_ && diagnostic.f_plus >= bead_f_min_ &&
     diagnostic.span >= bead_span_min_) ? 1 : 0;
  // Reject split same-sign domains such as LL00LLRR00RR even though their
  // center-skipped signed sequence has only two L/R interfaces.
  diagnostic.simple_two_domain_path =
    (diagnostic.kink_count == 2 && diagnostic.center_domain_count <= 2 &&
     diagnostic.total_state_domain_count <= 4) ? 1 : 0;
  const bool two_well_delocalized = diagnostic.two_well_span != 0 &&
    diagnostic.f_zero <= bead_center_max_ && diagnostic.simple_two_domain_path != 0;
  diagnostic.barrier_centered = (two_well_delocalized &&
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
  }
  if (state > 0)
    ++stats.n_plus;
  else if (state < 0)
    ++stats.n_minus;
  else
    ++stats.n_deadband;
}

void Proton_Tunneling::start_attempt(
  HydrogenState& hydrogen_state,
  const int stable_state,
  const double time_fs,
  const GeometryResult& geometry)
{
  hydrogen_state.attempt_active = true;
  hydrogen_state.attempt_from_state = stable_state;
  hydrogen_state.attempt_id = next_attempt_id_++;
  hydrogen_state.attempt_start_time_fs = time_fs;
  hydrogen_state.attempt_delta_start = geometry.delta;
  hydrogen_state.attempt_E_start = geometry.E_ion_nominal_parallel;
  hydrogen_state.attempt_min_abs_delta = std::abs(geometry.delta);
  hydrogen_state.pending_state = 0;
  hydrogen_state.pending_count = 0;
  hydrogen_state.pending_start_time_fs = 0.0;
  hydrogen_state.best_bead_diagnostic = BeadDiagnostic();
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
  const GeometryResult* geometry)
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
  const double E_start_output = ion_field_enabled_ ? hydrogen_state.attempt_E_start : nan;
  const double E_end_output = ion_field_enabled_ ? E_end : nan;
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
  record.nearest_ion_id = nearest_ion_id_output;
  record.nearest_ion_distance = nearest_ion_distance_output;
  record.bead_diagnostic = hydrogen_state.best_bead_diagnostic;

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
  hydrogen_state.attempt_min_abs_delta = 0.0;
  hydrogen_state.pending_state = 0;
  hydrogen_state.pending_count = 0;
  hydrogen_state.pending_start_time_fs = 0.0;
  hydrogen_state.best_bead_diagnostic = BeadDiagnostic();
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
                                 HydrogenState& hydrogen_state) {
    if (!bead_diagnostic_enabled_)
      return;
    if (!bead_probe_frame) {
      ++bead_probe_frame_count_;
      bead_probe_frame = true;
    }
    BeadDiagnostic diagnostic;
    const bool evaluated = evaluate_bead_diagnostic(
      atom, box, hydrogen, geometry, time_fs, diagnostic);
    if (evaluated || !hydrogen_state.best_bead_diagnostic.valid)
      hydrogen_state.best_bead_diagnostic = diagnostic;
  };

  for (size_t h_index = 0; h_index < hydrogen_indices_.size(); ++h_index) {
    const int hydrogen = hydrogen_indices_[h_index];
    const GeometryResult& geometry = frame_geometries_[h_index];
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
          nullptr);
      }
      hydrogen_state = HydrogenState();
      hydrogen_state.oxygen_low = geometry.oxygen_low;
      hydrogen_state.oxygen_high = geometry.oxygen_high;
    }

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
        start_attempt(hydrogen_state, hydrogen_state.stable_state, time_fs, geometry);
        probe_attempt(hydrogen, geometry, hydrogen_state);
      }

      if (hydrogen_state.attempt_active) {
        const double abs_delta = std::abs(geometry.delta);
        if (abs_delta < hydrogen_state.attempt_min_abs_delta) {
          hydrogen_state.attempt_min_abs_delta = abs_delta;
          probe_attempt(hydrogen, geometry, hydrogen_state);
        }
        if (state == hydrogen_state.stable_state) {
          finish_attempt(
            hydrogen,
            hydrogen_state,
            AttemptOutcome::return_to_state,
            time_fs,
            geometry.delta,
            &geometry);
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
              &geometry);
          }
        } else {
          hydrogen_state.pending_state = 0;
          hydrogen_state.pending_count = 0;
          hydrogen_state.pending_start_time_fs = 0.0;
        }
      } else if (state == 0) {
        start_attempt(hydrogen_state, hydrogen_state.stable_state, time_fs, geometry);
        probe_attempt(hydrogen, geometry, hydrogen_state);
      }
    }
    hydrogen_state.last_delta = geometry.delta;
    hydrogen_state.last_E_parallel = geometry.E_ion_nominal_parallel;
    hydrogen_state.last_nearest_ion_id = geometry.nearest_ion_id;
    hydrogen_state.last_nearest_ion_distance = geometry.nearest_ion_distance;
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

  window_bonds_.clear();
  window_sample_count_ = 0;
  window_start_time_fs_ = 0.0;
  window_flip_count_ = 0;
  window_valid_pair_count_ = 0;
  window_positive_defect_sum_ = 0;
  window_negative_defect_sum_ = 0;
  window_assignment_ambiguous_count_ = 0;
  window_pair_conflict_count_ = 0;
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

void Proton_Tunneling::write_output_files()
{
  bias_file_ = my_fopen("proton_bias.out", "a");
  transfer_file_ = my_fopen("proton_transfer.out", "a");
  attempt_file_ = my_fopen("proton_attempt.out", "a");
  if (bead_diagnostic_enabled_)
    bead_event_file_ = my_fopen("proton_bead_event.out", "a");
  edge_window_file_ = my_fopen("proton_edge_window.out", "a");
  bond_file_ = my_fopen("proton_bond.out", "a");
  defect_file_ = my_fopen("proton_defect.out", "a");

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
    "nearest_ion_id nearest_ion_distance\n");
  if (bead_diagnostic_enabled_) {
    fprintf(bead_event_file_,
      "# columns attempt_id probe_time_fs H_id O_low O_high outcome num_beads "
      "delta_centroid f_minus f_zero f_plus sigma_delta delta_min delta_max span "
      "kink_count center_domain_count total_state_domain_count two_well_span "
      "simple_two_domain_path barrier_centered strict_tunneling_like "
      "rms_neighbor_delta_jump max_neighbor_delta_jump quantum_class\n");
  }
  fprintf(defect_file_, "# columns time_fs O_id q_defect nH cause_event_id\n");
  fprintf(edge_window_file_,
    "# columns window_id time_start_fs time_end_fs O_low O_high geometry_occupancy "
    "n_plus n_minus n_deadband A abs_A DeltaF_over_kBT attempts successes returns geometry_lost "
    "success_probability mean_delta mean_abs_delta mean_dOO mean_rperp "
    "mean_E_parallel std_E_parallel corr_delta_E_parallel mean_E_success mean_E_return ");
  if (ion_field_enabled_)
    fprintf(edge_window_file_, "nearest_%s_distance nearest_%s_distance\n",
      ion1_symbol_.c_str(), ion2_symbol_.c_str());
  else
    fprintf(edge_window_file_, "nearest_ion1_distance nearest_ion2_distance\n");
  fprintf(bond_file_,
    "# columns O_pair_low O_pair_high geometry_samples n_plus n_minus transitions "
    "A abs_A mean_abs_delta\n");

  const double bead_nan = std::numeric_limits<double>::quiet_NaN();
  for (const AttemptRecord& record : attempt_records_) {
    fprintf(
      attempt_file_,
      "%lld %.10e %.10e %d %d %d %d %d %s %.10e %.10e %.10e %.10e %.10e %d %.10e\n",
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
      record.nearest_ion_distance);

    if (bead_event_file_ != nullptr) {
      const BeadDiagnostic& diagnostic = record.bead_diagnostic;
      const double f_minus = diagnostic.valid ? diagnostic.f_minus : bead_nan;
      const double f_zero = diagnostic.valid ? diagnostic.f_zero : bead_nan;
      const double f_plus = diagnostic.valid ? diagnostic.f_plus : bead_nan;
      fprintf(
        bead_event_file_,
        "%lld %.10e %d %d %d %s %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e "
        "%d %d %d %d %d %d %d %.10e %.10e %s\n",
        record.attempt_id,
        diagnostic.valid ? diagnostic.probe_time_fs : bead_nan,
        record.hydrogen,
        record.oxygen_low,
        record.oxygen_high,
        outcome_name(record.outcome),
        diagnostic.num_beads,
        diagnostic.valid ? diagnostic.delta_centroid : bead_nan,
        f_minus,
        f_zero,
        f_plus,
        diagnostic.valid ? diagnostic.sigma_delta : bead_nan,
        diagnostic.valid ? diagnostic.delta_min : bead_nan,
        diagnostic.valid ? diagnostic.delta_max : bead_nan,
        diagnostic.valid ? diagnostic.span : bead_nan,
        diagnostic.valid ? diagnostic.kink_count : -1,
        diagnostic.valid ? diagnostic.center_domain_count : -1,
        diagnostic.valid ? diagnostic.total_state_domain_count : -1,
        diagnostic.valid ? diagnostic.two_well_span : 0,
        diagnostic.valid ? diagnostic.simple_two_domain_path : 0,
        diagnostic.valid ? diagnostic.barrier_centered : 0,
        diagnostic.valid ? diagnostic.strict_tunneling_like : 0,
        diagnostic.valid ? diagnostic.rms_neighbor_delta_jump : bead_nan,
        diagnostic.valid ? diagnostic.max_neighbor_delta_jump : bead_nan,
        quantum_character_name(diagnostic.character));
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
      "%.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
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
      record.nearest_ion2_distance);

  write_final_bonds();
  fclose(bias_file_);
  fclose(transfer_file_);
  fclose(attempt_file_);
  if (bead_event_file_ != nullptr)
    fclose(bead_event_file_);
  fclose(defect_file_);
  fclose(edge_window_file_);
  fclose(bond_file_);
  bias_file_ = nullptr;
  transfer_file_ = nullptr;
  attempt_file_ = nullptr;
  bead_event_file_ = nullptr;
  defect_file_ = nullptr;
  edge_window_file_ = nullptr;
  bond_file_ = nullptr;
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
  printf("    CPU state-machine wall time: %.6f s\n", state_machine_wall_time_);
  printf("    total observer wall time: %.6f s\n", total_observer_wall_time_);
  initialized_ = false;
}
