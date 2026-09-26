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

/*----------------------------------------------------------------------------80
Calculate the heat current autocorrelation (HAC) function.
------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "force/force.cuh"
#include "force/nep_charge.cuh"
#include "integrate/integrate.cuh"
#include "hac.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

bool HAC::get_current_for_step(const int step, double current[3]) const
{
  if (!centroid_force_source_is_immediate() || step <= 0 || step % sample_interval != 0) return false;
  const int frame = step / sample_interval - 1;
  const int number_of_frames = static_cast<int>(heat_all.size() / 5);
  if (frame < 0 || frame >= number_of_frames) return false;
  double components[5] = {};
  for (int component = 0; component < 5; ++component) {
    heat_all.copy_to_host(&components[component], 1, frame + number_of_frames * component);
  }
  current[0] = components[0] + components[1];
  current[1] = components[2] + components[3];
  current[2] = components[4];
  return std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2]);
}

#define NUM_OF_HEAT_COMPONENTS 5
#define NUM_OF_TYPE_HEAT_COMPONENTS 3
#define FILE_NAME_LENGTH 200
#define DIM 3

constexpr int REDUCE_THREADS = 1024;
constexpr int NUM_CHARGE_HEAT_CHANNELS = 6;

static bool qnep_sample_interval_matches(
  const double previous_time_fs,
  const double current_time_fs,
  const double reference_dt_fs)
{
  const double actual_dt_fs = current_time_fs - previous_time_fs;
  const double time_scale = std::max(
    1.0, std::max(std::fabs(previous_time_fs), std::fabs(current_time_fs)));
  const double tolerance =
    1.0e-10 * std::max(1.0, std::fabs(reference_dt_fs)) +
    64.0 * std::numeric_limits<double>::epsilon() * time_scale;

  return
    std::isfinite(actual_dt_fs) &&
    std::isfinite(reference_dt_fs) &&
    actual_dt_fs > 0.0 &&
    reference_dt_fs > 0.0 &&
    std::fabs(actual_dt_fs - reference_dt_fs) <= tolerance;
}

// Allocate memory for recording heat current data
void HAC::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (compute) {
    force_ = &force;
    deferred_centroid_enabled_ = false;
    centroid_frame_size_ = 0;
    centroid_frame_count_ = 0;
    centroid_chunk_frame_count_ = 0;
    deferred_staging_wall_time_ = 0.0;
    deferred_upload_wall_time_ = 0.0;
    deferred_qnep_wall_time_ = 0.0;
    deferred_heat_wall_time_ = 0.0;
    deferred_hac_wall_time_ = 0.0;
    centroid_sampled_frames_ = 0;
    centroid_direct_evaluations_ = 0;
    centroid_force_step_ = -1;
    int number_of_frames = number_of_steps / sample_interval;
    if (number_of_frames <= 0 || Nc > number_of_frames) {
      PRINT_INPUT_ERROR("Nc must not exceed the number of sampled HAC frames.");
    }

    if (qnep_full_a_) {
      const bool centroid_qnep_full_a = use_centroid_heat_flux_ != 0;
      const bool ring_polymer_run = integrate.type >= 31 && integrate.type <= 33;
      if (split_qnep_heat_by_type_ != 0) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a does not support split HAC output.");
      }
      if (deferred_centroid_qnep_ != 0) {
        PRINT_INPUT_ERROR(
          "hac_current qnep_full_a + deferred centroid is not implemented in 4C-immediate; "
          "use immediate centroid or legacy deferred centroid.");
      }
      if (centroid_qnep_full_a && !ring_polymer_run) {
        PRINT_INPUT_ERROR(
          "hac_current qnep_full_a centroid mode requires a PIMD/RPMD/TRPMD ensemble.");
      }
      if (
        centroid_qnep_full_a &&
        (integrate.deform_x != 0 || integrate.deform_y != 0 || integrate.deform_z != 0 ||
         integrate.use_scr_barostat ||
         (integrate.type == 33 && integrate.num_target_pressure_components != 0))) {
        PRINT_INPUT_ERROR(
          "hac_current qnep_full_a centroid mode requires a fixed-cell ring-polymer ensemble.");
      }
      if (output_interval > Nc) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a requires output_interval not to exceed Nc.");
      }
      const bool supported_ensemble =
        centroid_qnep_full_a ? ring_polymer_run
                             : (integrate.type == 0 || (integrate.type >= 1 && integrate.type <= 10));
      if (!supported_ensemble) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a uses an unsupported ensemble.");
      }
      const double normalization_temperature =
        integrate.type == 0 ? integrate.hac_normalization_temperature : integrate.temperature2;
      if (!std::isfinite(normalization_temperature) || !(normalization_temperature > 0.0)) {
        PRINT_INPUT_ERROR(
          "hac_current qnep_full_a requires a positive finite normalization temperature; "
          "for NVE use ensemble nve <temperature>.\n");
      }
      if (integrate.type != 0) {
        const double temperature_scale =
          std::max(1.0, std::max(std::fabs(integrate.temperature1), std::fabs(integrate.temperature2)));
        if (std::fabs(integrate.temperature1 - integrate.temperature2) >
            1.0e-12 * temperature_scale) {
          PRINT_INPUT_ERROR(
            "hac_current qnep_full_a requires a constant target temperature during the HAC run.");
        }
      }
      if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1) {
        PRINT_INPUT_ERROR(
          "hac_current qnep_full_a requires three-dimensional periodic boundary conditions.");
      }
      box.set_is_orthogonal();
      if (!box.is_orthogonal) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a requires an orthogonal simulation cell.");
      }
      if (force.potentials.size() != 1) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a requires exactly one qNEP potential.");
      }
      qnep_full_a_qnep_ = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
      if (qnep_full_a_qnep_ == nullptr) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a requires an NEP-Charge potential.");
      }
      if (!qnep_full_a_qnep_->uses_pppm()) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a requires kspace_method pppm.");
      }
      if (
        qnep_full_a_qnep_->get_charge_mode() != 1 &&
        qnep_full_a_qnep_->get_charge_mode() != 2) {
        PRINT_INPUT_ERROR("hac_current qnep_full_a supports qNEP charge mode 1 or mode 2 only.");
      }
      if (!qnep_existing_file_has_schema(
            "heat_current_type_resolved_qnep_full_a.out",
            {"# component_schema_version 4",
             "# segment_metadata_version 2",
             "# J_virial_existing = J_virial_remainder + J_dyn_local",
             "# J_dyn_local_source = projected_D_charge_gradient_channel"})) {
        PRINT_INPUT_ERROR(
          "heat_current_type_resolved_qnep_full_a.out has an incompatible component schema; "
          "remove or rename it before starting a new run.\n");
      }
      if (!qnep_existing_file_has_schema(
            "heat_current_qnep_full_a.out",
            {"# segment_metadata_version 2", "# columns step time_fs Jx Jy Jz"})) {
        PRINT_INPUT_ERROR(
          "heat_current_qnep_full_a.out has incompatible segment metadata; "
          "remove or rename it before starting a new run.\n");
      }
      if (!qnep_existing_file_has_schema(
            "hac_qnep_full_a.out",
            {"# segment_metadata_version 2",
             "# normalization 1/(k_B*T^2*V), trapezoid_running_integral",
             "# columns lag_index_first lag_time_ps HAC_x HAC_y HAC_z RTC_x RTC_y RTC_z"})) {
        PRINT_INPUT_ERROR(
          "hac_qnep_full_a.out has incompatible segment metadata; "
          "remove or rename it before starting a new run.\n");
      }
      for (int i = 0; i < 9; ++i) qnep_full_a_initial_cell_[i] = box.cpu_h[i];
      qnep_full_a_qnep_->enable_charge_diagnostics();
      qnep_full_a_qnep_->reset_dynamic_charge_cache();
      qnep_full_a_local_channel_validation_done_ = false;
      qnep_full_a_local_channel_validation_passed_ = false;
      qnep_full_a_local_channel_validation_error_ = 0.0;
      atom.enable_unwrapped_position();
      qnep_full_a_workspace_.resize(atom.number_of_atoms);
      qnep_full_a_dynamic_local_channel_per_atom_.resize(
        static_cast<size_t>(atom.number_of_atoms) * NUM_CHARGE_HEAT_CHANNELS);
      qnep_full_a_dynamic_local_channel_total_.resize(NUM_CHARGE_HEAT_CHANNELS);
      const int number_of_types = static_cast<int>(atom.cpu_type_size.size());
      atom.heat_per_atom.resize(static_cast<size_t>(atom.number_of_atoms) * 5);
      if (centroid_qnep_full_a) {
        centroid_potential_per_atom_.resize(atom.number_of_atoms);
        centroid_force_per_atom_.resize(static_cast<size_t>(atom.number_of_atoms) * 3);
        centroid_virial_per_atom_.resize(static_cast<size_t>(atom.number_of_atoms) * 9);
        centroid_position_work_.resize(static_cast<size_t>(atom.number_of_atoms) * 3);
        centroid_charge_backup_.resize(atom.number_of_atoms);
        centroid_bec_backup_.resize(static_cast<size_t>(atom.number_of_atoms) * 9);
      }
      qnep_full_a_base_by_type_current_.resize(static_cast<size_t>(number_of_types) * 3);
      qnep_full_a_current_history_.assign(
        static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_base_by_type_history_.assign(
        static_cast<size_t>(number_of_types) * 3 * number_of_frames, 0.0);
      qnep_full_a_j_conv_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_j_virial_existing_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_j_dyn_local_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_j_virial_remainder_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_j_reference_static_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_j_base_existing_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_j_added_dynamic_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_delta_j_q_pppm_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_delta_j_q_real_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_projection_a_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_closure_error_history_.assign(static_cast<size_t>(3) * number_of_frames, 0.0);
      qnep_full_a_sample_times_fs_.assign(number_of_frames, 0.0);
      qnep_full_a_sample_steps_.assign(number_of_frames, -1);
      printf("    HAC current operator is qnep_full_a (A-route, full current).\n");
      return;
    }

    heat_all.resize(NUM_OF_HEAT_COMPONENTS * number_of_frames);
    heat_all_by_type_.resize(atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS * number_of_frames);
    atom.heat_per_atom.resize(atom.number_of_atoms * 5);
    if (use_centroid_heat_flux_) {
      centroid_potential_per_atom_.resize(atom.number_of_atoms);
      centroid_force_per_atom_.resize(atom.number_of_atoms * 3);
      centroid_virial_per_atom_.resize(atom.number_of_atoms * 9);

      const bool is_ring_polymer_run = integrate.type >= 31 && integrate.type <= 33;
      const bool fixed_box =
        is_ring_polymer_run && integrate.deform_x == 0 && integrate.deform_y == 0 &&
        integrate.deform_z == 0 && !integrate.use_scr_barostat &&
        (integrate.type != 33 || integrate.num_target_pressure_components == 0);
      const bool deferred_supported =
        deferred_centroid_qnep_ != 0 && !split_qnep_heat_by_type_ && fixed_box &&
        atom.number_of_beads > 1 && force.pimd_qnep_batch_available();
      if (deferred_supported) {
        deferred_centroid_enabled_ = true;
        centroid_frame_size_ = atom.number_of_atoms * 3;
        centroid_position_chunk_gpu_.resize(
          static_cast<size_t>(deferred_centroid_chunk_size_) * centroid_frame_size_);
        centroid_velocity_chunk_gpu_.resize(
          static_cast<size_t>(deferred_centroid_chunk_size_) * centroid_frame_size_);
        centroid_position_frames_cpu_.resize(
          static_cast<size_t>(number_of_frames) * centroid_frame_size_);
        centroid_velocity_frames_cpu_.resize(
          static_cast<size_t>(number_of_frames) * centroid_frame_size_);
        deferred_position_frames_gpu_.resize(deferred_centroid_chunk_size_);
        deferred_velocity_frames_gpu_.resize(deferred_centroid_chunk_size_);
        deferred_potential_frames_gpu_.resize(deferred_centroid_chunk_size_);
        deferred_force_frames_gpu_.resize(deferred_centroid_chunk_size_);
        deferred_virial_frames_gpu_.resize(deferred_centroid_chunk_size_);
        for (int frame = 0; frame < deferred_centroid_chunk_size_; ++frame) {
          deferred_position_frames_gpu_[frame].resize(centroid_frame_size_);
          deferred_velocity_frames_gpu_[frame].resize(centroid_frame_size_);
          deferred_potential_frames_gpu_[frame].resize(atom.number_of_atoms);
          deferred_force_frames_gpu_[frame].resize(atom.number_of_atoms * 3);
          deferred_virial_frames_gpu_[frame].resize(atom.number_of_atoms * 9);
        }
        printf(
          "    centroid qNEP evaluation is deferred to postprocess in fixed %d-frame batches.\n",
          deferred_centroid_chunk_size_);
      } else if (fixed_box && atom.number_of_beads > 1) {
        if (deferred_centroid_qnep_ != 0) {
          printf(
            "    deferred centroid qNEP mode unavailable; using the normal sampled centroid path.\n");
        }
        printf(
          "    centroid HAC will use a direct single-configuration qNEP evaluation at sampled frames.\n");
      }
    }
    if (split_qnep_heat_by_type_) {
      heat_all_by_type_electro_.resize(
        atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS * number_of_frames);
      non_electro_potential_per_atom_.resize(atom.number_of_atoms);
      non_electro_force_per_atom_.resize(atom.number_of_atoms * 3);
      non_electro_virial_per_atom_.resize(atom.number_of_atoms * 9);
      electro_virial_per_atom_.resize(atom.number_of_atoms * 9);
      electro_heat_per_atom_.resize(atom.number_of_atoms * NUM_OF_HEAT_COMPONENTS);
      if (use_centroid_heat_flux_) {
        electro_potential_per_atom_.resize(atom.number_of_atoms);
      }
    }
  }
}

static __global__ void
gpu_sum_heat(const int N, const int Nd, const int nd, const double* g_heat, double* g_heat_all);
static __global__ void gpu_sum_heat_by_type(
  const int N,
  const int Nd,
  const int nd,
  const int number_of_types,
  const int* g_type,
  const double* g_heat,
  double* g_heat_all_by_type);
static __global__ void gpu_sum_components(
  const int N, const int number_of_components, const double* g_values, double* g_total);
static void compute_full_heat_per_atom(
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom,
  const bool include_kinetic);

static __global__ void gpu_store_centroid_frame(
  const int frame_size,
  const double* position,
  const double* velocity,
  double* position_frames,
  double* velocity_frames,
  const int frame)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < frame_size) {
    const int offset = frame * frame_size;
    position_frames[offset + n] = position[n];
    velocity_frames[offset + n] = velocity[n];
  }
}

void HAC::flush_deferred_centroid_chunk_()
{
  if (!deferred_centroid_enabled_ || centroid_chunk_frame_count_ == 0) {
    return;
  }
  const int first_frame = centroid_frame_count_ - centroid_chunk_frame_count_;
  const size_t offset = static_cast<size_t>(first_frame) * centroid_frame_size_;
  const size_t size = static_cast<size_t>(centroid_chunk_frame_count_) * centroid_frame_size_;
  const auto begin = std::chrono::high_resolution_clock::now();
  centroid_position_chunk_gpu_.copy_to_host(centroid_position_frames_cpu_.data() + offset, size);
  centroid_velocity_chunk_gpu_.copy_to_host(centroid_velocity_frames_cpu_.data() + offset, size);
  deferred_staging_wall_time_ += std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now() - begin).count();
  centroid_chunk_frame_count_ = 0;
}

void HAC::process_deferred_centroid_frames_(
  Atom& atom, Box& box, const int number_of_frames, const int number_of_types, const int Nd)
{
  if (!deferred_centroid_enabled_ || force_ == nullptr || number_of_frames == 0) {
    return;
  }
  const int N = atom.number_of_atoms;

  for (int first_frame = 0; first_frame < number_of_frames;
       first_frame += deferred_centroid_chunk_size_) {
    const int frames_in_chunk = std::min(
      deferred_centroid_chunk_size_, number_of_frames - first_frame);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < deferred_centroid_chunk_size_; ++frame) {
      const int source_frame = first_frame + std::min(frame, frames_in_chunk - 1);
      const size_t offset = static_cast<size_t>(source_frame) * centroid_frame_size_;
      deferred_position_frames_gpu_[frame].copy_from_host(
        centroid_position_frames_cpu_.data() + offset, centroid_frame_size_);
      deferred_velocity_frames_gpu_[frame].copy_from_host(
        centroid_velocity_frames_cpu_.data() + offset, centroid_frame_size_);
    }
    deferred_upload_wall_time_ += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - upload_begin).count();

    const auto qnep_begin = std::chrono::high_resolution_clock::now();
    if (!force_->compute_qnep_centroid_frames_batch(
          box,
          atom.type,
          deferred_position_frames_gpu_,
          deferred_potential_frames_gpu_,
          deferred_force_frames_gpu_,
          deferred_virial_frames_gpu_)) {
      PRINT_INPUT_ERROR(
        "Deferred centroid qNEP batch evaluation failed; use a fixed-box qNEP PIMD batch run.");
    }
    CHECK(gpuDeviceSynchronize());
    deferred_qnep_wall_time_ += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - qnep_begin).count();

    const auto heat_begin = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < frames_in_chunk; ++frame) {
      const int nd = first_frame + frame;
      compute_full_heat_per_atom(
        atom.mass,
        deferred_potential_frames_gpu_[frame],
        deferred_virial_frames_gpu_[frame],
        deferred_velocity_frames_gpu_[frame],
        atom.heat_per_atom,
        true);
      gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>(
        N, Nd, nd, atom.heat_per_atom.data(), heat_all.data());
      gpu_sum_heat_by_type<<<number_of_types * NUM_OF_TYPE_HEAT_COMPONENTS, 1024>>>(
        N,
        Nd,
        nd,
        number_of_types,
        atom.type.data(),
        atom.heat_per_atom.data(),
        heat_all_by_type_.data());
    }
    GPU_CHECK_KERNEL
    CHECK(gpuDeviceSynchronize());
    deferred_heat_wall_time_ += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - heat_begin).count();
  }
}

// sum up the per-atom heat current to get the total heat current
static __global__ void
gpu_sum_heat(const int N, const int Nd, const int nd, const double* g_heat, double* g_heat_all)
{
  // <<<NUM_OF_HEAT_COMPONENTS, 1024>>>
  const int tid = threadIdx.x;
  const int number_of_patches = (N - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * 1024;
    if (n < N) {
      s_data[tid] += g_heat[n + N * blockIdx.x];
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_heat_all[nd + Nd * blockIdx.x] = s_data[0];
  }
}

static __global__ void gpu_sum_heat_by_type(
  const int N,
  const int Nd,
  const int nd,
  const int number_of_types,
  const int* g_type,
  const double* g_heat,
  double* g_heat_all_by_type)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int type_index = bid / 3;
  const int component = bid % 3;
  const int number_of_patches = (N - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  if (type_index < number_of_types) {
    for (int patch = 0; patch < number_of_patches; ++patch) {
      const int n = tid + patch * 1024;
      if (n < N && g_type[n] == type_index) {
        if (component == 0) {
          s_data[tid] += g_heat[n] + g_heat[n + N];
        } else if (component == 1) {
          s_data[tid] += g_heat[n + N * 2] + g_heat[n + N * 3];
        } else {
          s_data[tid] += g_heat[n + N * 4];
        }
      }
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_heat_all_by_type[nd + Nd * bid] = s_data[0];
  }
}

static __global__ void gpu_sum_components(
  const int N, const int number_of_components, const double* g_values, double* g_total)
{
  const int component = blockIdx.x;
  const int tid = threadIdx.x;
  if (component >= number_of_components) return;

  __shared__ double s_data[REDUCE_THREADS];
  double sum = 0.0;
  for (int n = tid; n < N; n += REDUCE_THREADS)
    sum += g_values[n + component * N];
  s_data[tid] = sum;
  __syncthreads();

  for (int offset = REDUCE_THREADS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0) g_total[component] = s_data[0];
}

static __global__ void gpu_subtract_array(
  const int size,
  const double* total,
  const double* part,
  double* difference)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    difference[n] = total[n] - part[n];
  }
}

static __global__ void gpu_compute_full_heat_per_atom(
  const int N,
  const double* mass,
  const double* potential,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  const int include_kinetic,
  double* jx_in,
  double* jx_out,
  double* jy_in,
  double* jy_out,
  double* jz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    const double v_x = vx[n];
    const double v_y = vy[n];
    const double v_z = vz[n];
    double energy = potential[n];
    if (include_kinetic) {
      energy += mass[n] * (v_x * v_x + v_y * v_y + v_z * v_z) * 0.5;
    }
    jx_in[n] = (energy + sxx[n]) * v_x + sxy[n] * v_y;
    jx_out[n] = sxz[n] * v_z;
    jy_in[n] = syx[n] * v_x + (energy + syy[n]) * v_y;
    jy_out[n] = syz[n] * v_z;
    jz[n] = szx[n] * v_x + szy[n] * v_y + (energy + szz[n]) * v_z;
  }
}

static void compute_full_heat_per_atom(
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom,
  const bool include_kinetic = true)
{
  const int N = velocity_per_atom.size() / 3;
  gpu_compute_full_heat_per_atom<<<(N - 1) / 128 + 1, 128>>>(
    N,
    mass.data(),
    potential_per_atom.data(),
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    include_kinetic ? 1 : 0,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2,
    heat_per_atom.data() + N * 3,
    heat_per_atom.data() + N * 4);
  GPU_CHECK_KERNEL
}

void HAC::check_qnep_full_a_fixed_cell_(const Box& box) const
{
  if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 || !box.is_orthogonal) {
    PRINT_INPUT_ERROR(
      "hac_current qnep_full_a requires a three-dimensional orthogonal periodic box.");
  }
  for (int i = 0; i < 9; ++i) {
    if (box.cpu_h[i] != qnep_full_a_initial_cell_[i]) {
      PRINT_INPUT_ERROR(
        "hac_current qnep_full_a requires a fixed simulation cell; the cell changed during the run.");
    }
  }
}

void HAC::pre_force(
  const int step,
  const double,
  Integrate&,
  std::vector<Group>&,
  Atom&,
  Box& box,
  Force&)
{
  if (!compute || !qnep_full_a_ || (step + 1) % sample_interval != 0) return;
  box.set_is_orthogonal();
  check_qnep_full_a_fixed_cell_(box);
  if (use_centroid_heat_flux_) return;
  qnep_full_a_qnep_->request_charge_diagnostics_for_next_force();
  qnep_full_a_qnep_->request_peratom_virial_for_next_force();
}

// sample heat current data for HAC calculations.
void HAC::end_of_step(
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
  if (!compute)
    return;
  if (qnep_full_a_) {
    if ((step + 1) % sample_interval != 0) return;

    box.set_is_orthogonal();
    check_qnep_full_a_fixed_cell_(box);
    const int Nd = static_cast<int>(qnep_full_a_sample_steps_.size());
    const int nd = (step + 1) / sample_interval - 1;
    if (nd < 0 || nd >= Nd || qnep_full_a_sample_steps_[nd] != -1) {
      PRINT_INPUT_ERROR("qnep_full_a samples did not arrive in sequential frame order.");
    }

    const double sample_time_fs = global_time * TIME_UNIT_CONVERSION;
    if (!std::isfinite(sample_time_fs)) {
      PRINT_INPUT_ERROR("qnep_full_a requires finite sample times.");
    }
    if (nd >= 1) {
      const double actual_dt_fs =
        sample_time_fs - qnep_full_a_sample_times_fs_[nd - 1];
      if (!std::isfinite(actual_dt_fs) || !(actual_dt_fs > 0.0)) {
        PRINT_INPUT_ERROR(
          "qnep_full_a requires positive finite sample-time intervals.");
      }
      if (
        nd >= 2 &&
        !qnep_sample_interval_matches(
          qnep_full_a_sample_times_fs_[nd - 1],
          sample_time_fs,
          qnep_full_a_sample_times_fs_[1] - qnep_full_a_sample_times_fs_[0])) {
        PRINT_INPUT_ERROR(
          "qnep_full_a requires uniformly spaced sample times "
          "for HAC correlation and integration.");
      }
    }

    const GPU_Vector<double>* current_position = &atom.position_per_atom;
    const GPU_Vector<double>* current_unwrapped_position = &atom.unwrapped_position;
    const GPU_Vector<double>* current_potential = &atom.potential_per_atom;
    const GPU_Vector<double>* current_virial = &atom.virial_per_atom;
    const GPU_Vector<double>* current_velocity = &atom.velocity_per_atom;
    if (use_centroid_heat_flux_) {
      ++centroid_sampled_frames_;
      ++centroid_direct_evaluations_;
      centroid_position_work_.copy_from_device(atom.position_per_atom.data());
      centroid_charge_backup_.copy_from_device(
        qnep_full_a_qnep_->get_charge_reference().data());
      if (qnep_full_a_qnep_->md_qnep_bec_enabled()) {
        centroid_bec_backup_.copy_from_device(qnep_full_a_qnep_->get_bec_reference().data());
      }
      qnep_full_a_qnep_->request_charge_diagnostics_for_next_force();
      qnep_full_a_qnep_->request_peratom_virial_for_next_force();
      force.compute(
        box,
        centroid_position_work_,
        atom.type,
        group,
        centroid_potential_per_atom_,
        centroid_force_per_atom_,
        centroid_virial_per_atom_,
        atom.velocity_per_atom,
        atom.mass);
      current_position = &centroid_position_work_;
      current_potential = &centroid_potential_per_atom_;
      current_virial = &centroid_virial_per_atom_;
    }
    if (!qnep_full_a_qnep_->has_charge_diagnostics_for_current_force_frame()) {
      PRINT_INPUT_ERROR(
        "hac_current qnep_full_a force did not capture diagnostics for its current frame.");
    }

    double delta_j_q_pppm[3] = {0.0, 0.0, 0.0};
    double delta_j_q_real[3] = {0.0, 0.0, 0.0};
    double delta_j_q_total[3] = {0.0, 0.0, 0.0};
    qnep_full_a_qnep_->compute_charge_rate_for_current_force_frame(
      box,
      atom.type,
      *current_position,
      *current_velocity,
      &qnep_full_a_dynamic_local_channel_per_atom_);
    if (!qnep_full_a_qnep_->compute_dynamic_charge_correction(
          atom.number_of_atoms,
          0,
          atom.number_of_atoms,
          0,
          step + 1,
          sample_time_fs,
          box,
          *current_position,
          delta_j_q_pppm,
          delta_j_q_real,
          delta_j_q_total)) {
      PRINT_INPUT_ERROR(
        "hac_current qnep_full_a could not obtain a valid dynamic-q correction for the sampled frame.");
    }

    double j_conv[3] = {0.0, 0.0, 0.0};
    double j_virial[3] = {0.0, 0.0, 0.0};
    double j_base_existing[3] = {0.0, 0.0, 0.0};
    double j_projection[3][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0}};
    double j_full_a[3] = {0.0, 0.0, 0.0};
    if (!compute_qnep_full_a_current(
          *qnep_full_a_qnep_,
          atom.number_of_atoms,
          box,
          *current_position,
          *current_unwrapped_position,
          atom.mass,
          *current_potential,
          *current_virial,
          *current_velocity,
          delta_j_q_total,
          false,
          qnep_full_a_workspace_,
          j_conv,
          j_virial,
          j_base_existing,
          j_projection,
          j_full_a)) {
      PRINT_INPUT_ERROR("hac_current qnep_full_a could not assemble a finite full current.");
    }

    gpu_sum_components<<<NUM_CHARGE_HEAT_CHANNELS, REDUCE_THREADS>>>(
      atom.number_of_atoms,
      NUM_CHARGE_HEAT_CHANNELS,
      qnep_full_a_dynamic_local_channel_per_atom_.data(),
      qnep_full_a_dynamic_local_channel_total_.data());
    GPU_CHECK_KERNEL
    double dynamic_local_channel_total[NUM_CHARGE_HEAT_CHANNELS] =
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    qnep_full_a_dynamic_local_channel_total_.copy_to_host(dynamic_local_channel_total);
    double j_dyn_local[3] = {0.0, 0.0, 0.0};
    double j_virial_remainder[3] = {0.0, 0.0, 0.0};
    double j_reference_static[3] = {0.0, 0.0, 0.0};
    double j_added_dynamic[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < 3; ++d) {
      j_dyn_local[d] = dynamic_local_channel_total[d] + dynamic_local_channel_total[3 + d];
      j_virial_remainder[d] = j_virial[d] - j_dyn_local[d];
      j_reference_static[d] = j_conv[d] + j_virial_remainder[d];
      j_added_dynamic[d] =
        j_dyn_local[d] + delta_j_q_pppm[d] + delta_j_q_real[d] + j_projection[0][d];
    }

    const int number_of_types = static_cast<int>(atom.cpu_type_size.size());
    gpu_sum_heat_by_type<<<number_of_types * NUM_OF_TYPE_HEAT_COMPONENTS, 1024>>>(
      atom.number_of_atoms,
      1,
      0,
      number_of_types,
      atom.type.data(),
      qnep_full_a_workspace_.gpu_virial_heat_per_atom.data(),
      qnep_full_a_base_by_type_current_.data());
    GPU_CHECK_KERNEL

    const size_t base_history_offset =
      static_cast<size_t>(nd) * number_of_types * NUM_OF_TYPE_HEAT_COMPONENTS;
    qnep_full_a_base_by_type_current_.copy_to_host(
      qnep_full_a_base_by_type_history_.data() + base_history_offset);

    double base_by_type_sum[3] = {0.0, 0.0, 0.0};
    for (int type_index = 0; type_index < number_of_types; ++type_index) {
      const size_t type_offset = base_history_offset + static_cast<size_t>(type_index) * 3;
      for (int d = 0; d < 3; ++d) base_by_type_sum[d] +=
      qnep_full_a_base_by_type_history_[type_offset + d];
    }

    const double closure_tolerance_floor = 1.0e-8 * TIME_UNIT_CONVERSION;
    double local_channel_validation_error = qnep_full_a_local_channel_validation_error_;
    bool local_channel_valid = qnep_full_a_local_channel_validation_passed_;
    if (!qnep_full_a_local_channel_validation_done_) {
      GPU_Vector<double> virial_nep;
      GPU_Vector<double> virial_electrostatic_fixed;
      GPU_Vector<double> virial_dynamic_charge;
      const size_t virial_size = static_cast<size_t>(atom.number_of_atoms) * 9;
      virial_nep.resize(virial_size);
      virial_electrostatic_fixed.resize(virial_size);
      virial_dynamic_charge.resize(virial_size);
      qnep_full_a_qnep_->compute_virial_components(
        box,
        atom.type,
        *current_position,
        *current_virial,
        true,
        true,
        true,
        virial_nep,
        virial_electrostatic_fixed,
        virial_dynamic_charge);
      compute_heat(
        virial_dynamic_charge,
        *current_velocity,
        qnep_full_a_workspace_.gpu_virial_heat_per_atom);
      gpu_sum_components<<<NUM_OF_HEAT_COMPONENTS, REDUCE_THREADS>>>(
        atom.number_of_atoms,
        NUM_OF_HEAT_COMPONENTS,
        qnep_full_a_workspace_.gpu_virial_heat_per_atom.data(),
        qnep_full_a_workspace_.gpu_virial_heat_total.data());
      GPU_CHECK_KERNEL
      qnep_full_a_workspace_.gpu_virial_heat_total.copy_to_host(
        qnep_full_a_workspace_.cpu_virial_heat_total.data());
      double j_dyn_local_residual[3] = {0.0, 0.0, 0.0};
      j_dyn_local_residual[0] =
        qnep_full_a_workspace_.cpu_virial_heat_total[0] +
        qnep_full_a_workspace_.cpu_virial_heat_total[1];
      j_dyn_local_residual[1] =
        qnep_full_a_workspace_.cpu_virial_heat_total[2] +
        qnep_full_a_workspace_.cpu_virial_heat_total[3];
      j_dyn_local_residual[2] = qnep_full_a_workspace_.cpu_virial_heat_total[4];

      double local_channel_validation_error_squared = 0.0;
      double channel_norm_squared = 0.0;
      double residual_norm_squared = 0.0;
      bool local_channel_finite = true;
      for (int d = 0; d < 3; ++d) {
        const double error = j_dyn_local[d] - j_dyn_local_residual[d];
        if (!std::isfinite(error) || !std::isfinite(j_dyn_local[d]) ||
            !std::isfinite(j_dyn_local_residual[d])) {
          local_channel_finite = false;
        } else {
          local_channel_validation_error_squared += error * error;
          channel_norm_squared += j_dyn_local[d] * j_dyn_local[d];
          residual_norm_squared += j_dyn_local_residual[d] * j_dyn_local_residual[d];
        }
      }
      local_channel_validation_error = local_channel_finite
        ? std::sqrt(local_channel_validation_error_squared)
        : std::numeric_limits<double>::quiet_NaN();
      const double scale = local_channel_finite
        ? std::max(std::sqrt(channel_norm_squared), std::sqrt(residual_norm_squared))
        : std::numeric_limits<double>::quiet_NaN();
      const double tolerance = closure_tolerance_floor + 1.0e-5 * scale;
      local_channel_valid =
        local_channel_finite && local_channel_validation_error <= tolerance;
      qnep_full_a_local_channel_validation_done_ = true;
      qnep_full_a_local_channel_validation_passed_ = local_channel_valid;
      qnep_full_a_local_channel_validation_error_ = local_channel_validation_error;
    }
    if (!local_channel_valid) {
      PRINT_INPUT_ERROR(
        "hac_current qnep_full_a local dynamic-charge channel failed independent residual validation.");
    }

    for (int d = 0; d < 3; ++d) {
      const double reconstructed =
        base_by_type_sum[d] + delta_j_q_pppm[d] + delta_j_q_real[d] + j_projection[0][d];
      const double closure_error = j_full_a[d] - reconstructed;
      const double scale = std::max(std::fabs(j_full_a[d]), std::fabs(reconstructed));
      const double closure_tolerance = closure_tolerance_floor + 1.0e-5 * scale;
      if (
        !std::isfinite(j_dyn_local[d]) || !std::isfinite(j_virial_remainder[d]) ||
        !std::isfinite(j_added_dynamic[d]) ||
        !std::isfinite(closure_error) || std::fabs(closure_error) > closure_tolerance) {
        PRINT_INPUT_ERROR(
          "hac_current qnep_full_a component currents do not close to the full current.");
      }
      qnep_full_a_j_conv_history_[nd + Nd * d] = j_conv[d];
      qnep_full_a_j_virial_existing_history_[nd + Nd * d] = j_virial[d];
      qnep_full_a_j_dyn_local_history_[nd + Nd * d] = j_dyn_local[d];
      qnep_full_a_j_virial_remainder_history_[nd + Nd * d] = j_virial_remainder[d];
      qnep_full_a_j_reference_static_history_[nd + Nd * d] = j_reference_static[d];
      qnep_full_a_j_base_existing_history_[nd + Nd * d] = j_base_existing[d];
      qnep_full_a_j_added_dynamic_history_[nd + Nd * d] = j_added_dynamic[d];
      qnep_full_a_closure_error_history_[nd + Nd * d] = closure_error;
      qnep_full_a_delta_j_q_pppm_history_[nd + Nd * d] = delta_j_q_pppm[d];
      qnep_full_a_delta_j_q_real_history_[nd + Nd * d] = delta_j_q_real[d];
      qnep_full_a_projection_a_history_[nd + Nd * d] = j_projection[0][d];
    }
    for (int d = 0; d < 3; ++d)
      qnep_full_a_current_history_[nd + Nd * d] = j_full_a[d];

    if (use_centroid_heat_flux_) {
      qnep_full_a_qnep_->get_charge_reference().copy_from_device(
        centroid_charge_backup_.data());
      if (qnep_full_a_qnep_->md_qnep_bec_enabled()) {
        qnep_full_a_qnep_->get_bec_reference().copy_from_device(centroid_bec_backup_.data());
      }
      qnep_full_a_qnep_->invalidate_current_force_caches();
      qnep_full_a_qnep_->mark_single_frame_neighbor_reference_pending();
      centroid_force_step_ = step + 1;
    }
    qnep_full_a_sample_steps_[nd] = step + 1;
    qnep_full_a_sample_times_fs_[nd] = sample_time_fs;
    return;
  }
  if ((step + 1) % sample_interval != 0)
    return;

  const int N = atom.number_of_atoms;
  const GPU_Vector<double>* centroid_potential_source = &centroid_potential_per_atom_;
  const GPU_Vector<double>* centroid_virial_source = &centroid_virial_per_atom_;
  if (use_centroid_heat_flux_) {
    ++centroid_sampled_frames_;
    if (deferred_centroid_enabled_) {
      const int frame = (step + 1) / sample_interval - 1;
      if (frame != centroid_frame_count_ || frame < 0 || frame >= number_of_steps / sample_interval) {
        PRINT_INPUT_ERROR(
          "Deferred centroid samples must arrive in sequential HAC frame order.");
      }
      gpu_store_centroid_frame<<<(centroid_frame_size_ - 1) / 128 + 1, 128>>>(
        centroid_frame_size_,
        atom.position_per_atom.data(),
        atom.velocity_per_atom.data(),
        centroid_position_chunk_gpu_.data(),
        centroid_velocity_chunk_gpu_.data(),
        centroid_chunk_frame_count_);
      GPU_CHECK_KERNEL
      ++centroid_chunk_frame_count_;
      ++centroid_frame_count_;
      if (centroid_chunk_frame_count_ == deferred_centroid_chunk_size_) {
        flush_deferred_centroid_chunk_();
      }
      return;
    }
    ++centroid_direct_evaluations_;
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      centroid_potential_per_atom_,
      centroid_force_per_atom_,
      centroid_virial_per_atom_,
      atom.velocity_per_atom,
      atom.mass);
    centroid_force_step_ = step + 1;
    centroid_potential_source = &centroid_potential_per_atom_;
    centroid_virial_source = &centroid_virial_per_atom_;
    compute_full_heat_per_atom(
      atom.mass,
      centroid_potential_per_atom_,
      centroid_virial_per_atom_,
      atom.velocity_per_atom,
      atom.heat_per_atom);
  } else {
    compute_heat(atom.virial_per_atom, atom.velocity_per_atom, atom.heat_per_atom);
  }

  if (split_qnep_heat_by_type_) {
    const bool has_qnep_split = force.compute_qnep_non_electro(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      non_electro_potential_per_atom_,
      non_electro_force_per_atom_,
      non_electro_virial_per_atom_);
    if (has_qnep_split) {
      if (use_centroid_heat_flux_) {
        gpu_subtract_array<<<(N - 1) / 128 + 1, 128>>>(
          N,
          centroid_potential_source->data(),
          non_electro_potential_per_atom_.data(),
          electro_potential_per_atom_.data());
        gpu_subtract_array<<<((N * 9) - 1) / 128 + 1, 128>>>(
          N * 9,
          centroid_virial_source->data(),
          non_electro_virial_per_atom_.data(),
          electro_virial_per_atom_.data());
        GPU_CHECK_KERNEL
        compute_full_heat_per_atom(
          atom.mass,
          electro_potential_per_atom_,
          electro_virial_per_atom_,
          atom.velocity_per_atom,
          electro_heat_per_atom_,
          false);
      } else {
        gpu_subtract_array<<<((N * 9) - 1) / 128 + 1, 128>>>(
          N * 9,
          atom.virial_per_atom.data(),
          non_electro_virial_per_atom_.data(),
          electro_virial_per_atom_.data());
        GPU_CHECK_KERNEL
        compute_heat(electro_virial_per_atom_, atom.velocity_per_atom, electro_heat_per_atom_);
      }
    } else {
      CHECK(gpuMemset(
        electro_heat_per_atom_.data(), 0, sizeof(double) * N * NUM_OF_HEAT_COMPONENTS));
    }
  }

  int nd = (step + 1) / sample_interval - 1;
  int Nd = number_of_steps / sample_interval;
  gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>(N, Nd, nd, atom.heat_per_atom.data(), heat_all.data());
  gpu_sum_heat_by_type<<<atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS, 1024>>>(
    N,
    Nd,
    nd,
    atom.cpu_type_size.size(),
    atom.type.data(),
    atom.heat_per_atom.data(),
    heat_all_by_type_.data());
  if (split_qnep_heat_by_type_) {
    gpu_sum_heat_by_type<<<atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS, 1024>>>(
      N,
      Nd,
      nd,
      atom.cpu_type_size.size(),
      atom.type.data(),
      electro_heat_per_atom_.data(),
      heat_all_by_type_electro_.data());
  }
  GPU_CHECK_KERNEL
}

// Calculate the Heat current Auto-Correlation function (HAC)
static __global__ void gpu_find_hac(const int Nc, const int Nd, const double* g_heat, double* g_hac)
{
  //<<<Nc, 128>>>

  __shared__ double s_hac_xi[128];
  __shared__ double s_hac_xo[128];
  __shared__ double s_hac_yi[128];
  __shared__ double s_hac_yo[128];
  __shared__ double s_hac_z[128];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (Nd - 1) / 128 + 1;
  int number_of_data = Nd - bid;

  s_hac_xi[tid] = 0.0;
  s_hac_xo[tid] = 0.0;
  s_hac_yi[tid] = 0.0;
  s_hac_yo[tid] = 0.0;
  s_hac_z[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int index = tid + patch * 128;
    if (index + bid < Nd) {
      s_hac_xi[tid] += g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 0] +
                       g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 1];
      s_hac_xo[tid] += g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 1] +
                       g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 0];
      s_hac_yi[tid] += g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 2] +
                       g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 3];
      s_hac_yo[tid] += g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 3] +
                       g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 2];
      s_hac_z[tid] += g_heat[index + Nd * 4] * g_heat[index + bid + Nd * 4];
    }
  }
  __syncthreads();


  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_hac_xi[tid] += s_hac_xi[tid + offset];
      s_hac_xo[tid] += s_hac_xo[tid + offset];
      s_hac_yi[tid] += s_hac_yi[tid + offset];
      s_hac_yo[tid] += s_hac_yo[tid + offset];
      s_hac_z[tid] += s_hac_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_hac[bid + Nc * 0] = s_hac_xi[0] / number_of_data;
    g_hac[bid + Nc * 1] = s_hac_xo[0] / number_of_data;
    g_hac[bid + Nc * 2] = s_hac_yi[0] / number_of_data;
    g_hac[bid + Nc * 3] = s_hac_yo[0] / number_of_data;
    g_hac[bid + Nc * 4] = s_hac_z[0] / number_of_data;
  }
}

static __global__ void gpu_find_hac_3(const int Nc, const int Nd, const double* g_current, double* g_hac)
{
  __shared__ double s_x[128];
  __shared__ double s_y[128];
  __shared__ double s_z[128];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int number_of_patches = (Nd - 1) / 128 + 1;
  const int number_of_data = Nd - bid;

  s_x[tid] = 0.0;
  s_y[tid] = 0.0;
  s_z[tid] = 0.0;
  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int index = tid + patch * 128;
    if (index + bid < Nd) {
      s_x[tid] += g_current[index] * g_current[index + bid];
      s_y[tid] += g_current[index + Nd] * g_current[index + bid + Nd];
      s_z[tid] += g_current[index + 2 * Nd] * g_current[index + bid + 2 * Nd];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_x[tid] += s_x[tid + offset];
      s_y[tid] += s_y[tid + offset];
      s_z[tid] += s_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_hac[bid + Nc * 0] = s_x[0] / number_of_data;
    g_hac[bid + Nc * 1] = s_y[0] / number_of_data;
    g_hac[bid + Nc * 2] = s_z[0] / number_of_data;
  }
}

// Calculate the Running Thermal Conductivity (RTC) from the HAC
static void find_rtc_components(
  const int Nc, const int number_of_components, const double factor, const double* hac, double* rtc)
{
  for (int k = 0; k < number_of_components; k++) {
    for (int nc = 1; nc < Nc; nc++) {
      const int index = Nc * k + nc;
      rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index]) * factor;
    }
  }
}

static void find_rtc(const int Nc, const double factor, const double* hac, double* rtc)
{
  find_rtc_components(Nc, NUM_OF_HEAT_COMPONENTS, factor, hac, rtc);
}

void HAC::post_run_qnep_full_a_(
  Atom& atom,
  Box& box,
  const int number_of_steps,
  const double time_step,
  const double temperature,
  const char* temperature_source)
{
  box.set_is_orthogonal();
  check_qnep_full_a_fixed_cell_(box);
  const int Nd = number_of_steps / sample_interval;
  const int number_of_types = static_cast<int>(atom.cpu_type_size.size());
  if (
    Nd <= 0 || static_cast<int>(qnep_full_a_sample_steps_.size()) != Nd ||
    static_cast<int>(qnep_full_a_sample_times_fs_.size()) != Nd ||
    qnep_full_a_current_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_base_by_type_history_.size() !=
      static_cast<size_t>(number_of_types) * 3 * Nd ||
    qnep_full_a_j_conv_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_j_virial_existing_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_j_dyn_local_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_j_virial_remainder_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_j_reference_static_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_j_base_existing_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_j_added_dynamic_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_delta_j_q_pppm_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_delta_j_q_real_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_projection_a_history_.size() != static_cast<size_t>(3 * Nd) ||
    qnep_full_a_closure_error_history_.size() != static_cast<size_t>(3 * Nd)) {
    PRINT_INPUT_ERROR("qnep_full_a sample storage does not match the requested HAC frames.");
  }
  for (int nd = 0; nd < Nd; ++nd) {
    if (qnep_full_a_sample_steps_[nd] < 0 || !std::isfinite(qnep_full_a_sample_times_fs_[nd])) {
      PRINT_INPUT_ERROR("qnep_full_a did not produce every requested HAC sample.");
    }
  }

  double dt_in_fs = time_step * sample_interval * TIME_UNIT_CONVERSION;
  if (Nd > 1) {
    dt_in_fs = qnep_full_a_sample_times_fs_[1] - qnep_full_a_sample_times_fs_[0];
  }
  if (!std::isfinite(dt_in_fs) || !(dt_in_fs > 0.0)) {
    PRINT_INPUT_ERROR("qnep_full_a requires a positive finite sampling interval.");
  }
  for (int nd = 1; nd < Nd; ++nd) {
    if (qnep_full_a_sample_steps_[nd] - qnep_full_a_sample_steps_[nd - 1] != sample_interval) {
      PRINT_INPUT_ERROR(
        "qnep_full_a requires a uniform sampling interval for HAC correlation and integration.");
    }
    if (!qnep_sample_interval_matches(
          qnep_full_a_sample_times_fs_[nd - 1],
          qnep_full_a_sample_times_fs_[nd],
          dt_in_fs)) {
      PRINT_INPUT_ERROR(
        "qnep_full_a requires uniformly spaced sample times for HAC correlation and integration.");
    }
  }
  const double dt_in_natural = dt_in_fs / TIME_UNIT_CONVERSION;
  const double volume = box.get_volume();
  if (
    !(dt_in_natural > 0.0) || !std::isfinite(temperature) || !(temperature > 0.0) ||
    !std::isfinite(volume) || !(volume > 0.0)) {
    PRINT_INPUT_ERROR("qnep_full_a requires positive sampling time, temperature, and cell volume.");
  }

  GPU_Vector<double> current_gpu(static_cast<size_t>(3) * Nd);
  current_gpu.copy_from_host(qnep_full_a_current_history_.data());
  GPU_Vector<double> hac_gpu(static_cast<size_t>(3) * Nc);
  std::vector<double> hac_cpu(static_cast<size_t>(3) * Nc);
  gpu_find_hac_3<<<Nc, 128>>>(Nc, Nd, current_gpu.data(), hac_gpu.data());
  GPU_CHECK_KERNEL
  hac_gpu.copy_to_host(hac_cpu.data());
  for (const double value : hac_cpu) {
    if (!std::isfinite(value)) {
      PRINT_INPUT_ERROR("qnep_full_a HAC contains non-finite values.");
    }
  }

  const double factor =
    dt_in_natural * 0.5 / (K_B * temperature * temperature * volume) *
    KAPPA_UNIT_CONVERSION;
  if (!std::isfinite(factor) || !(factor > 0.0)) {
    PRINT_INPUT_ERROR(
      "qnep_full_a conductivity normalization factor must be positive and finite.");
  }
  std::vector<double> rtc(static_cast<size_t>(3) * Nc, 0.0);
  find_rtc_components(Nc, 3, factor, hac_cpu.data(), rtc.data());
  for (const double value : rtc) {
    if (!std::isfinite(value)) {
      PRINT_INPUT_ERROR("qnep_full_a running conductivity contains non-finite values.");
    }
  }

  const double inv_time_conversion = 1.0 / TIME_UNIT_CONVERSION;
  const double hac_conversion = inv_time_conversion * inv_time_conversion;
  const double dt_in_ps = dt_in_fs / 1000.0;
  const int charge_mode = qnep_full_a_qnep_->get_charge_mode();
  const bool centroid_qnep_full_a = use_centroid_heat_flux_ != 0;
  const char* current_configuration = centroid_qnep_full_a ? "centroid" : "classical";
  const char* centroid_evaluation =
    centroid_qnep_full_a ? "immediate_single_frame" : "not_applicable";
  const char* sampling_stage = centroid_qnep_full_a
    ? "post_compute2_centroid_force_final_velocity"
    : "post_compute2_final_velocity";
  const auto segment_stamp =
    std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::string segment_id =
    "qnep_full_a_" + std::to_string(segment_stamp) + "_" +
    std::to_string(qnep_full_a_sample_steps_.front()) + "_" +
    std::to_string(qnep_full_a_sample_steps_.back()) + "_" + std::to_string(Nd) + "_" +
    std::to_string(sample_interval) + "_" + std::to_string(charge_mode);
  const auto write_segment_metadata = [&](FILE* file) {
    fprintf(file, "# segment_metadata_version 2\n");
    fprintf(file, "# segment_id %s\n", segment_id.c_str());
    fprintf(file, "# charge_mode %d\n", charge_mode);
    fprintf(
      file,
      "# dynamic_q_formula_version %s\n",
      qnep_full_a_qnep_->get_dynamic_q_formula_version());
    fprintf(file, "# current_configuration %s\n", current_configuration);
    fprintf(file, "# centroid_evaluation %s\n", centroid_evaluation);
    fprintf(file, "# temperature_K %.17g\n", temperature);
    fprintf(file, "# temperature_source %s\n", temperature_source);
    fprintf(file, "# volume_Angstrom3 %.17g\n", volume);
    fprintf(file, "# number_of_atoms %d\n", atom.number_of_atoms);
    fprintf(file, "# number_of_types %d\n", number_of_types);
    fprintf(file, "# cell_h0_h1_h2_h3_h4_h5_h6_h7_h8");
    for (int i = 0; i < 9; ++i) fprintf(file, " %.17g", qnep_full_a_initial_cell_[i]);
    fprintf(file, "\n");
  };

  FILE* fid_current = my_fopen("heat_current_qnep_full_a.out", "a");
  fprintf(fid_current, "# segment_begin operator qnep_full_a projection_route A full_current 1\n");
  write_segment_metadata(fid_current);
  fprintf(fid_current, "# sampling_stage %s\n", sampling_stage);
  fprintf(fid_current, "# sampled_frames %d\n", Nd);
  fprintf(fid_current, "# sample_interval_md_steps %d\n", sample_interval);
  fprintf(fid_current, "# sample_interval_fs %.17g\n", dt_in_fs);
  fprintf(fid_current, "# current_internal_units eV*Angstrom/natural_time\n");
  fprintf(fid_current, "# current_output_units eV*Angstrom/fs\n");
  fprintf(fid_current, "# columns step time_fs Jx Jy Jz\n");
  for (int nd = 0; nd < Nd; ++nd) {
    fprintf(
      fid_current,
      "%d %25.15e %25.15e %25.15e %25.15e\n",
      qnep_full_a_sample_steps_[nd],
      qnep_full_a_sample_times_fs_[nd],
      qnep_full_a_current_history_[nd + Nd * 0] * inv_time_conversion,
      qnep_full_a_current_history_[nd + Nd * 1] * inv_time_conversion,
      qnep_full_a_current_history_[nd + Nd * 2] * inv_time_conversion);
  }
  fflush(fid_current);
  fclose(fid_current);

  std::vector<std::string> type_symbols(number_of_types);
  std::vector<int> type_symbol_found(number_of_types, 0);
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    const int type_index = atom.cpu_type[n];
    if (!type_symbol_found[type_index]) {
      type_symbols[type_index] = atom.cpu_atom_symbol[n];
      type_symbol_found[type_index] = 1;
    }
  }
  for (int type_index = 0; type_index < number_of_types; ++type_index) {
    if (!type_symbol_found[type_index]) {
      type_symbols[type_index] = std::string("type") + std::to_string(type_index);
    }
  }

  FILE* fid_type_resolved = my_fopen("heat_current_type_resolved_qnep_full_a.out", "a");
  fprintf(fid_type_resolved, "# segment_begin operator qnep_full_a projection_route A full_current 1\n");
  fprintf(fid_type_resolved, "# component_schema_version 4\n");
  write_segment_metadata(fid_type_resolved);
  fprintf(
    fid_type_resolved,
    "# local_channel_validation_frame %d\n",
    qnep_full_a_sample_steps_.front());
  fprintf(
    fid_type_resolved,
    "# local_channel_validation_error %.17g\n",
    qnep_full_a_local_channel_validation_error_ * inv_time_conversion);
  fprintf(
    fid_type_resolved,
    "# local_channel_valid %d\n",
    qnep_full_a_local_channel_validation_passed_ ? 1 : 0);
  fprintf(fid_type_resolved, "# sampling_stage %s\n", sampling_stage);
  fprintf(fid_type_resolved, "# sampled_frames %d\n", Nd);
  fprintf(fid_type_resolved, "# sample_interval_md_steps %d\n", sample_interval);
  fprintf(fid_type_resolved, "# sample_interval_fs %.17g\n", dt_in_fs);
  fprintf(fid_type_resolved, "# current_internal_units eV*Angstrom/natural_time\n");
  fprintf(fid_type_resolved, "# current_output_units eV*Angstrom/fs\n");
  fprintf(fid_type_resolved, "# J_virial_existing = J_virial_remainder + J_dyn_local\n");
  fprintf(fid_type_resolved, "# J_reference_static = J_conv + J_virial_remainder\n");
  fprintf(fid_type_resolved, "# J_base_existing = J_conv + J_virial_existing\n");
  fprintf(fid_type_resolved, "# J_added_dynamic = J_dyn_local + DeltaJ_q_pppm + DeltaJ_q_real + J_A\n");
  fprintf(fid_type_resolved, "# J_dyn_local_source = projected_D_charge_gradient_channel\n");
  fprintf(fid_type_resolved, "# local_channel_validation = first_hac_sample_then_cached\n");
  fprintf(
    fid_type_resolved,
    "# local_channel_validation_error_definition = l2_norm_channel_minus_residual\n");
  fprintf(
    fid_type_resolved,
    "# local_channel_validation_tolerance = 1e-8 eV*Angstrom/fs + 1e-5*max(norm(channel),norm(residual))\n");
  fprintf(fid_type_resolved, "# type_channels = base_current_only\n");
  fprintf(fid_type_resolved, "# dynamic_corrections = global_unassigned\n");
  fprintf(
    fid_type_resolved,
    "# sum_rule = J_reference_static + J_added_dynamic\n");
  fprintf(
    fid_type_resolved,
    "# closure_tolerance = 1e-8 eV*Angstrom/fs + 1e-5*max(abs(reference),abs(reconstructed))\n");
  fprintf(
    fid_type_resolved,
    "# columns step time_fs J_full_x J_full_y J_full_z"
    " J_conv_x J_conv_y J_conv_z"
    " J_virial_remainder_x J_virial_remainder_y J_virial_remainder_z"
    " J_dyn_local_x J_dyn_local_y J_dyn_local_z"
    " J_virial_existing_x J_virial_existing_y J_virial_existing_z"
    " J_reference_static_x J_reference_static_y J_reference_static_z"
    " J_base_existing_x J_base_existing_y J_base_existing_z"
    " J_added_dynamic_x J_added_dynamic_y J_added_dynamic_z");
  for (int type_index = 0; type_index < number_of_types; ++type_index) {
    fprintf(
      fid_type_resolved,
      " type%d_%s_Jbase_existing_x type%d_%s_Jbase_existing_y type%d_%s_Jbase_existing_z",
      type_index,
      type_symbols[type_index].c_str(),
      type_index,
      type_symbols[type_index].c_str(),
      type_index,
      type_symbols[type_index].c_str());
  }
  fprintf(
    fid_type_resolved,
    " DeltaJ_q_pppm_x DeltaJ_q_pppm_y DeltaJ_q_pppm_z"
    " DeltaJ_q_real_x DeltaJ_q_real_y DeltaJ_q_real_z"
    " J_A_x J_A_y J_A_z DeltaJ_q_total_x DeltaJ_q_total_y DeltaJ_q_total_z"
    " closure_error_x closure_error_y closure_error_z\n");
  for (int nd = 0; nd < Nd; ++nd) {
    fprintf(
      fid_type_resolved,
      "%d %25.15e %25.15e %25.15e %25.15e",
      qnep_full_a_sample_steps_[nd],
      qnep_full_a_sample_times_fs_[nd],
      qnep_full_a_current_history_[nd + Nd * 0] * inv_time_conversion,
      qnep_full_a_current_history_[nd + Nd * 1] * inv_time_conversion,
      qnep_full_a_current_history_[nd + Nd * 2] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(fid_type_resolved, " %25.15e", qnep_full_a_j_conv_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(
        fid_type_resolved,
        " %25.15e",
        qnep_full_a_j_virial_remainder_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(fid_type_resolved, " %25.15e", qnep_full_a_j_dyn_local_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(
        fid_type_resolved,
        " %25.15e",
        qnep_full_a_j_virial_existing_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(
        fid_type_resolved,
        " %25.15e",
        qnep_full_a_j_reference_static_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(
        fid_type_resolved,
        " %25.15e",
        qnep_full_a_j_base_existing_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(
        fid_type_resolved,
        " %25.15e",
        qnep_full_a_j_added_dynamic_history_[nd + Nd * d] * inv_time_conversion);
    for (int type_index = 0; type_index < number_of_types; ++type_index) {
      const size_t type_offset =
        static_cast<size_t>(nd) * number_of_types * 3 + static_cast<size_t>(type_index) * 3;
      fprintf(
        fid_type_resolved,
        " %25.15e %25.15e %25.15e",
        qnep_full_a_base_by_type_history_[type_offset + 0] * inv_time_conversion,
        qnep_full_a_base_by_type_history_[type_offset + 1] * inv_time_conversion,
        qnep_full_a_base_by_type_history_[type_offset + 2] * inv_time_conversion);
    }
    for (int d = 0; d < 3; ++d)
      fprintf(fid_type_resolved, " %25.15e", qnep_full_a_delta_j_q_pppm_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(fid_type_resolved, " %25.15e", qnep_full_a_delta_j_q_real_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d)
      fprintf(fid_type_resolved, " %25.15e", qnep_full_a_projection_a_history_[nd + Nd * d] * inv_time_conversion);
    for (int d = 0; d < 3; ++d) {
      const double delta_total =
        qnep_full_a_delta_j_q_pppm_history_[nd + Nd * d] +
        qnep_full_a_delta_j_q_real_history_[nd + Nd * d];
      fprintf(fid_type_resolved, " %25.15e", delta_total * inv_time_conversion);
    }
    for (int d = 0; d < 3; ++d)
      fprintf(fid_type_resolved, " %25.15e", qnep_full_a_closure_error_history_[nd + Nd * d] * inv_time_conversion);
    fprintf(fid_type_resolved, "\n");
  }
  fflush(fid_type_resolved);
  fclose(fid_type_resolved);

  FILE* fid_hac = my_fopen("hac_qnep_full_a.out", "a");
  fprintf(fid_hac, "# segment_begin operator qnep_full_a projection_route A full_current 1\n");
  write_segment_metadata(fid_hac);
  fprintf(fid_hac, "# sampling_stage %s\n", sampling_stage);
  fprintf(fid_hac, "# sampled_frames %d\n", Nd);
  fprintf(fid_hac, "# correlation_points %d\n", Nc);
  fprintf(fid_hac, "# sample_interval_md_steps %d\n", sample_interval);
  fprintf(fid_hac, "# output_interval_lags %d\n", output_interval);
  fprintf(fid_hac, "# sample_interval_fs %.17g\n", dt_in_fs);
  fprintf(fid_hac, "# normalization 1/(k_B*T^2*V), trapezoid_running_integral\n");
  fprintf(fid_hac, "# correlation_origins all_valid, denominator sampled_frames-lag, mean_subtraction none\n");
  fprintf(fid_hac, "# current_output_units eV*Angstrom/fs\n");
  fprintf(fid_hac, "# hac_output_units (eV*Angstrom/fs)^2\n");
  fprintf(fid_hac, "# rtc_output_units W/m/K\n");
  fprintf(fid_hac, "# rtc_internal_factor %.17g\n", factor);
  fprintf(
    fid_hac,
    "# rtc_factor_from_output_hac %.17g\n",
    dt_in_fs * 0.5 / (K_B * temperature * temperature * volume) *
      KAPPA_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  fprintf(fid_hac, "# columns lag_index_first lag_time_ps HAC_x HAC_y HAC_z RTC_x RTC_y RTC_z\n");
  const int number_of_output_data = Nc / output_interval;
  for (int nd = 0; nd < number_of_output_data; ++nd) {
    const int nc = nd * output_interval;
    double hac_ave[3] = {0.0, 0.0, 0.0};
    double rtc_ave[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
      for (int m = 0; m < output_interval; ++m) {
        const int count = Nc * k + nc + m;
        hac_ave[k] += hac_cpu[count] * hac_conversion;
        rtc_ave[k] += rtc[count];
      }
      hac_ave[k] /= output_interval;
      rtc_ave[k] /= output_interval;
    }
    fprintf(
      fid_hac,
      "%d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e\n",
      nc,
      (nc + 0.5 * (output_interval - 1)) * dt_in_ps,
      hac_ave[0],
      hac_ave[1],
      hac_ave[2],
      rtc_ave[0],
      rtc_ave[1],
      rtc_ave[2]);
  }
  fflush(fid_hac);
  fclose(fid_hac);

  printf("qnep_full_a HAC and running thermal conductivity are calculated.\n");
}

// Calculate HAC (heat currant auto-correlation function)
// and RTC (running thermal conductivity)
void HAC::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute)
    return;
  if (qnep_full_a_) {
    const double normalization_temperature =
      integrate.type == 0 ? integrate.hac_normalization_temperature : temperature;
    const char* temperature_source =
      integrate.type == 0
      ? "explicit_nve_hac"
      : (use_centroid_heat_flux_ ? "ring_polymer_target" : "fixed_nvt_target");
    post_run_qnep_full_a_(
      atom,
      box,
      number_of_steps,
      time_step,
      normalization_temperature,
      temperature_source);
    compute = 0;
    return;
  }
  print_line_1();
  printf("Start to calculate HAC and related quantities.\n");

  const int Nd = number_of_steps / sample_interval;
  const double dt = time_step * sample_interval;
  const double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

  if (deferred_centroid_enabled_) {
    flush_deferred_centroid_chunk_();
    if (centroid_frame_count_ != Nd) {
      PRINT_INPUT_ERROR("The deferred centroid frame count does not match the HAC samples.");
    }
    process_deferred_centroid_frames_(
      atom, box, Nd, static_cast<int>(atom.cpu_type_size.size()), Nd);
  }

  std::vector<double> heat_current_cpu(Nd * NUM_OF_HEAT_COMPONENTS);
  heat_all.copy_to_host(heat_current_cpu.data());

  const char* heat_current_file_name =
    use_centroid_heat_flux_ ? "heat_current_centroid.out" : "heat_current.out";
  FILE* fid_heat_current = fopen(heat_current_file_name, "a");
  fprintf(fid_heat_current, "# time_ps Jx Jy Jz\n");
  for (int nd = 0; nd < Nd; ++nd) {
    const double jx = heat_current_cpu[nd + Nd * 0] + heat_current_cpu[nd + Nd * 1];
    const double jy = heat_current_cpu[nd + Nd * 2] + heat_current_cpu[nd + Nd * 3];
    const double jz = heat_current_cpu[nd + Nd * 4];
    fprintf(fid_heat_current, "%25.15e%25.15e%25.15e%25.15e\n", (nd + 1) * dt_in_ps, jx, jy, jz);
  }
  fflush(fid_heat_current);
  fclose(fid_heat_current);

  const int number_of_types = atom.cpu_type_size.size();
  std::vector<double> heat_current_by_type_cpu(Nd * number_of_types * 3);
  heat_all_by_type_.copy_to_host(heat_current_by_type_cpu.data());
  std::vector<std::string> type_symbols(number_of_types);
  std::vector<int> type_symbol_found(number_of_types, 0);
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    const int type_index = atom.cpu_type[n];
    if (!type_symbol_found[type_index]) {
      type_symbols[type_index] = atom.cpu_atom_symbol[n];
      type_symbol_found[type_index] = 1;
    }
  }
  for (int type_index = 0; type_index < number_of_types; ++type_index) {
    if (!type_symbol_found[type_index]) {
      type_symbols[type_index] = std::string("type") + std::to_string(type_index);
    }
  }

  const char* type_resolved_file_name =
    use_centroid_heat_flux_ ? "heat_current_type_resolved_centroid.out" : "heat_current_type_resolved.out";
  FILE* fid_type_resolved = fopen(type_resolved_file_name, "a");
  fprintf(fid_type_resolved, "# time_ps total_Jx total_Jy total_Jz");
  for (int type_index = 0; type_index < number_of_types; ++type_index) {
    fprintf(
      fid_type_resolved,
      " type%d_%s_Jx type%d_%s_Jy type%d_%s_Jz",
      type_index,
      type_symbols[type_index].c_str(),
      type_index,
      type_symbols[type_index].c_str(),
      type_index,
      type_symbols[type_index].c_str());
  }
  fprintf(fid_type_resolved, "\n");
  for (int nd = 0; nd < Nd; ++nd) {
    const double jx_total = heat_current_cpu[nd + Nd * 0] + heat_current_cpu[nd + Nd * 1];
    const double jy_total = heat_current_cpu[nd + Nd * 2] + heat_current_cpu[nd + Nd * 3];
    const double jz_total = heat_current_cpu[nd + Nd * 4];
    fprintf(
      fid_type_resolved,
      "%25.15e%25.15e%25.15e%25.15e",
      (nd + 1) * dt_in_ps,
      jx_total,
      jy_total,
      jz_total);
    for (int type_index = 0; type_index < number_of_types; ++type_index) {
      const int column_offset = type_index * 3;
      const double jx_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 0)];
      const double jy_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 1)];
      const double jz_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 2)];
      fprintf(fid_type_resolved, "%25.15e%25.15e%25.15e", jx_type, jy_type, jz_type);
    }
    fprintf(fid_type_resolved, "\n");
  }
  fflush(fid_type_resolved);
  fclose(fid_type_resolved);

  if (split_qnep_heat_by_type_) {
    std::vector<double> heat_current_by_type_electro_cpu(Nd * number_of_types * 3);
    heat_all_by_type_electro_.copy_to_host(heat_current_by_type_electro_cpu.data());
    const char* type_resolved_split_file_name = use_centroid_heat_flux_
      ? "heat_current_type_resolved_qnep_split_centroid.out"
      : "heat_current_type_resolved_qnep_split.out";
    FILE* fid_type_resolved_split = fopen(type_resolved_split_file_name, "a");
    fprintf(
      fid_type_resolved_split,
      "# time_ps total_Jx total_Jy total_Jz electro_Jx electro_Jy electro_Jz non_electro_Jx non_electro_Jy non_electro_Jz");
    for (int type_index = 0; type_index < number_of_types; ++type_index) {
      fprintf(
        fid_type_resolved_split,
        " type%d_%s_electro_Jx type%d_%s_electro_Jy type%d_%s_electro_Jz"
        " type%d_%s_non_electro_Jx type%d_%s_non_electro_Jy type%d_%s_non_electro_Jz",
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str());
    }
    fprintf(fid_type_resolved_split, "\n");
    for (int nd = 0; nd < Nd; ++nd) {
      const double jx_total = heat_current_cpu[nd + Nd * 0] + heat_current_cpu[nd + Nd * 1];
      const double jy_total = heat_current_cpu[nd + Nd * 2] + heat_current_cpu[nd + Nd * 3];
      const double jz_total = heat_current_cpu[nd + Nd * 4];
      double jx_electro_total = 0.0;
      double jy_electro_total = 0.0;
      double jz_electro_total = 0.0;
      for (int type_index = 0; type_index < number_of_types; ++type_index) {
        const int column_offset = type_index * 3;
        jx_electro_total += heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 0)];
        jy_electro_total += heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 1)];
        jz_electro_total += heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 2)];
      }
      fprintf(
        fid_type_resolved_split,
        "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e",
        (nd + 1) * dt_in_ps,
        jx_total,
        jy_total,
        jz_total,
        jx_electro_total,
        jy_electro_total,
        jz_electro_total,
        jx_total - jx_electro_total,
        jy_total - jy_electro_total,
        jz_total - jz_electro_total);
      for (int type_index = 0; type_index < number_of_types; ++type_index) {
        const int column_offset = type_index * 3;
        const double jx_electro = heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 0)];
        const double jy_electro = heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 1)];
        const double jz_electro = heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 2)];
        const double jx_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 0)];
        const double jy_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 1)];
        const double jz_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 2)];
        fprintf(
          fid_type_resolved_split,
          "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e",
          jx_electro,
          jy_electro,
          jz_electro,
          jx_type - jx_electro,
          jy_type - jy_electro,
          jz_type - jz_electro);
      }
      fprintf(fid_type_resolved_split, "\n");
    }
    fflush(fid_type_resolved_split);
    fclose(fid_type_resolved_split);
  }

  // major data
  std::vector<double> rtc(Nc * NUM_OF_HEAT_COMPONENTS, 0.0);
  GPU_Vector<double> hac_gpu(Nc * NUM_OF_HEAT_COMPONENTS);
  std::vector<double> hac_cpu(Nc * NUM_OF_HEAT_COMPONENTS);

  // Here, the block size is fixed to 128, which is a good choice
  const auto hac_begin = std::chrono::high_resolution_clock::now();
  gpu_find_hac<<<Nc, 128>>>(Nc, Nd, heat_all.data(), hac_gpu.data());
  GPU_CHECK_KERNEL

  hac_gpu.copy_to_host(hac_cpu.data());

  double factor = dt * 0.5 / (K_B * temperature * temperature * box.get_volume());
  factor *= KAPPA_UNIT_CONVERSION;

  find_rtc(Nc, factor, hac_cpu.data(), rtc.data());
  if (deferred_centroid_enabled_) {
    deferred_hac_wall_time_ += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - hac_begin).count();
  }
  const char* output_file_name = use_centroid_heat_flux_ ? "hac_centroid.out" : "hac.out";
  FILE* fid = fopen(output_file_name, "a");
  const int number_of_output_data = Nc / output_interval;
  for (int nd = 0; nd < number_of_output_data; nd++) {
    const int nc = nd * output_interval;
    double hac_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    double rtc_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
      for (int m = 0; m < output_interval; m++) {
        const int count = Nc * k + nc + m;
        hac_ave[k] += hac_cpu[count];
        rtc_ave[k] += rtc[count];
      }
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      hac_ave[m] /= output_interval;
      rtc_ave[m] /= output_interval;
    }
    fprintf(fid, "%25.15e", (nc + output_interval * 0.5) * dt_in_ps);
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", hac_ave[m]);
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", rtc_ave[m]);
    }
    fprintf(fid, "\n");
  }
  fflush(fid);
  fclose(fid);

  printf("HAC and related quantities are calculated.\n");
  if (use_centroid_heat_flux_) {
    printf("Centroid HAC force source:\n");
    printf("    sampled frames = %lld\n", centroid_sampled_frames_);
    printf("    direct centroid evaluations = %lld\n", centroid_direct_evaluations_);
    if (deferred_centroid_enabled_) {
      printf("    deferred centroid frames = %d\n", centroid_frame_count_);
      printf("    trajectory staging/D2H wall time = %g s\n", deferred_staging_wall_time_);
      printf("    deferred frame upload wall time = %g s\n", deferred_upload_wall_time_);
      printf("    deferred qNEP batch wall time = %g s\n", deferred_qnep_wall_time_);
      printf("    deferred heat accumulation wall time = %g s\n", deferred_heat_wall_time_);
      printf("    HAC correlation wall time = %g s\n", deferred_hac_wall_time_);
    }
  }
  print_line_2();

  compute = 0;
}

void HAC::parse(const char** param, int num_param)
{
  compute = 1;

  printf("Compute HAC.\n");

  if (!(num_param == 4 || num_param == 5 || num_param == 6 || num_param == 7)) {
    PRINT_INPUT_ERROR("compute_hac should have 3, 4, 5, or 6 parameters.\n");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for HAC should be an integer number.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("sample interval for HAC should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for HAC should be an integer number.\n");
  }
  if (Nc <= 0) {
    PRINT_INPUT_ERROR("Nc for HAC should be positive.\n");
  }
  printf("    Nc is %d\n", Nc);

  if (!is_valid_int(param[3], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HAC should be an integer number.\n");
  }
  if (output_interval <= 0) {
    PRINT_INPUT_ERROR("output_interval for HAC should be positive.\n");
  }
  printf("    output_interval is %d\n", output_interval);
  if (num_param >= 5) {
    if (!is_valid_int(param[4], &use_centroid_heat_flux_)) {
      PRINT_INPUT_ERROR("centroid heat flux flag for HAC should be an integer.\n");
    }
    if (use_centroid_heat_flux_ != 0) {
      printf("    use the full classical heat-flux operator on the current centroid structure.\n");
    }
  }
  if (num_param >= 6) {
    if (!is_valid_int(param[5], &split_qnep_heat_by_type_)) {
      PRINT_INPUT_ERROR("qNEP electrostatic split flag for HAC should be an integer.\n");
    }
    if (split_qnep_heat_by_type_ != 0) {
      printf("    output type-resolved electrostatic and non-electrostatic heat currents for qNEP.\n");
    }
  }
  if (num_param == 7) {
    if (!is_valid_int(param[6], &deferred_centroid_qnep_)) {
      PRINT_INPUT_ERROR("deferred centroid qNEP flag for HAC should be an integer.\n");
    }
    if (deferred_centroid_qnep_ != 0) {
      printf(
        "    defer centroid qNEP evaluation to postprocess using fixed 32-frame batches.\n");
    }
  }
}

HAC::HAC(const char** param, int num_param, const bool qnep_full_a)
  : qnep_full_a_(qnep_full_a)
{
  parse(param, num_param);
  action_name = "compute_hac";
}
