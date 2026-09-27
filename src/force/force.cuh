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

#pragma once

#include "model/box.cuh"
#include "model/group.cuh"
#include "potential.cuh"
#ifdef GPUMD_WPE_ENABLED
#include "wpe_stage.cuh"
#endif
#include "utilities/common.cuh"
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

enum class PIMD_DP_Source_Count_Mode
{
  Atomic,
  Check,
  NeighborCounts
};
class RunInput;

class Force
{
public:
  struct PIMD_Bead_Timing
  {
    double total = 0.0;
    double wrap_positions = 0.0;
    double stage_remote = 0.0;
    double compute_workers = 0.0;
    double gather_remote = 0.0;
    long long calls = 0;
    std::vector<double> worker_compute;
  };

  struct PIMD_Bead_GPU_Worker
  {
    int device_id = 0;
    std::unique_ptr<Potential> potential;
    GPU_Vector<int> type;
    std::vector<GPU_Vector<double>> position_beads;
    std::vector<GPU_Vector<double>> potential_beads;
    std::vector<GPU_Vector<double>> force_beads;
    std::vector<GPU_Vector<double>> virial_beads;
  };

  Force(void);

  void parse_potential(
    const std::vector<std::string>& tokens,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& mass_per_atom,
    int* position_image = nullptr);
  bool compute_qnep_non_electro(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute_pimd_beads(
    Box& box,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    std::vector<GPU_Vector<double>>& position_beads,
    std::vector<GPU_Vector<double>>& potential_beads,
    std::vector<GPU_Vector<double>>& force_beads,
    std::vector<GPU_Vector<double>>& virial_beads,
    std::vector<GPU_Vector<double>>& velocity_beads,
    GPU_Vector<double>& mass_per_atom);
  void compute_pimd_bead_range_on_device(
    int device_id,
    Box& box,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    std::vector<GPU_Vector<double>>& position_beads,
    std::vector<GPU_Vector<double>>& potential_beads,
    std::vector<GPU_Vector<double>>& force_beads,
    std::vector<GPU_Vector<double>>& virial_beads,
    std::vector<GPU_Vector<double>>& velocity_beads,
    GPU_Vector<double>& mass_per_atom,
    int bead_begin,
    int bead_end,
    double initial_temperature);

#ifdef GPUMD_WPE_ENABLED
  void wpe_process_command(
    const std::vector<std::string>& tokens,
    int current_number_of_atoms);
#endif

  void finalize();
  void notify_velocity_update();

  int get_number_of_types(FILE* fid_potential);
  void set_multiple_potentials_mode(std::string mode);
  void set_pimd_bead_gpu_parallel(const int num_devices);
  void set_pimd_bead_neighbor_rebuild(const bool always_rebuild);
  void set_pppm_mesh_spacing(const double spacing);
  void set_md_qnep_bec_mode(const int mode);
  void set_md_qnep_bec_required(const bool required);
  void set_pimd_bead_batch(const bool enabled);
  void enable_pimd_centroid_probe(
    const int number_of_atoms, const int number_of_beads, const int sample_interval);
  bool pimd_centroid_probe_enabled() const { return pimd_centroid_probe_enabled_; }
  bool pimd_centroid_probe_ready() const { return pimd_centroid_probe_ready_; }
  bool pimd_qnep_batch_available() const { return can_use_pimd_qnep_batch_(); }
  bool compute_qnep_centroid_frames_batch(
    Box& box,
    GPU_Vector<int>& type,
    std::vector<GPU_Vector<double>>& position_frames,
    std::vector<GPU_Vector<double>>& potential_frames,
    std::vector<GPU_Vector<double>>& force_frames,
    std::vector<GPU_Vector<double>>& virial_frames);
  const GPU_Vector<double>& get_pimd_centroid_potential() const
  {
    return pimd_centroid_potential_;
  }
  const GPU_Vector<double>& get_pimd_centroid_virial() const { return pimd_centroid_virial_; }
  void set_pimd_qnep_batch_bec_mode(const int mode);
  void set_pimd_qnep_batch_bec_required(const bool required);
  void set_pimd_nep_batch_profile(const bool enabled);
  void set_pimd_dp_batch_profile(const bool enabled);
  void set_pimd_dp_batch_source_count_mode(const PIMD_DP_Source_Count_Mode mode);
  void set_pimd_dp_batch_edge_fill_4_threads(const bool enabled);
  void reset_pimd_nep_batch_profile();
  void print_pimd_nep_batch_profile() const;
  bool pimd_nep_batch_profile_enabled() const { return pimd_nep_batch_profile_enabled_; }
  bool pimd_dp_batch_profile_enabled() const { return pimd_dp_batch_profile_enabled_; }
  int get_pimd_bead_gpu_parallel_devices() const { return pimd_bead_gpu_parallel_devices_; }
  int get_pimd_bead_gpu_worker_count() const { return int(pimd_bead_gpu_workers_.size()); }
  bool pimd_bead_gpu_parallel_available() const { return can_use_pimd_bead_gpu_parallel_(); }
  void reset_pimd_bead_timing();
  const PIMD_Bead_Timing& get_pimd_bead_timing() const { return pimd_bead_timing_; }
  double temperature = 0;
  double delta_T;
  std::vector<std::unique_ptr<Potential>> potentials;
  const std::string& primary_nep_model_path() const { return primary_nep_model_path_; }

  void set_temperature_range(
    const double temperature1, const double temperature2, const int number_of_steps);
  void advance_temperature();
  int get_number_of_potentials() const;
  Potential& get_potential(const int index);
  const RunInput& get_run_input() const;

private:
#ifdef GPUMD_WPE_ENABLED
  WpeGpumdStageState wpe_stage_state_;
#endif
  std::unique_ptr<Potential> create_potential(
    const std::vector<std::string>& tokens,
    FILE* fid_potential,
    char* potential_name,
    const int num_types,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input,
    bool& is_nep);

  int number_of_atoms_ = -1;
  bool is_fcp = false;
  bool has_non_nep = false;
  std::string multiple_potentials_mode_ = "observe"; // "observe" or "average"
  const RunInput* run_input_ = nullptr;
  int pimd_bead_gpu_parallel_devices_ = 1;
  bool pimd_bead_neighbor_always_rebuild_ = true;
  double pppm_mesh_spacing_ = 1.0;
  int md_qnep_bec_mode_ = 0; // 0: auto, 1: on, 2: off
  bool md_qnep_bec_required_ = false;
  bool pimd_bead_batch_enabled_ = false;
  bool pimd_centroid_probe_enabled_ = false;
  bool pimd_centroid_probe_ready_ = false;
  bool pimd_centroid_active_this_call_ = false;
  int pimd_centroid_number_of_beads_ = 0;
  int pimd_centroid_sample_interval_ = 0;
  long long pimd_force_call_count_ = 0;
  long long pimd_physical_batch_calls_ = 0;
  long long pimd_centroid_active_batch_calls_ = 0;
  long long pimd_centroid_inactive_batch_calls_ = 0;
  GPU_Vector<double> pimd_centroid_position_;
  GPU_Vector<double> pimd_centroid_potential_;
  GPU_Vector<double> pimd_centroid_force_;
  GPU_Vector<double> pimd_centroid_virial_;
  GPU_Vector<double*> pimd_centroid_position_ptrs_;
  std::vector<double*> pimd_centroid_position_ptrs_host_;
  int pimd_qnep_batch_bec_mode_ = 0; // 0: auto, 1: on, 2: off
  bool pimd_qnep_batch_bec_required_ = false;
  bool pimd_nep_batch_profile_enabled_ = false;
  bool pimd_dp_batch_profile_enabled_ = false;
  PIMD_DP_Source_Count_Mode pimd_dp_source_count_mode_ =
    PIMD_DP_Source_Count_Mode::Atomic;
  bool pimd_dp_batch_edge_fill_4_threads_ = false;
  std::string primary_nep_model_path_;
  std::string atom_types[NUM_ELEMENTS];
  std::unique_ptr<Potential> pimd_nep_single_gpu_batch_potential_;
  std::vector<std::unique_ptr<PIMD_Bead_GPU_Worker>> pimd_bead_gpu_workers_;
  PIMD_Bead_Timing pimd_bead_timing_;

  void check_types(const std::string& file_potential);
  void apply_md_qnep_bec_setting_();
  bool can_use_pimd_bead_gpu_parallel_() const;
  bool can_use_pimd_qnep_batch_() const;
  bool can_use_pimd_nep_batch_() const;
  bool can_use_pimd_dp_batch_() const;
  void compute_pimd_centroid_position_(
    Box& box, std::vector<GPU_Vector<double>>& position_beads);
  void apply_pimd_qnep_batch_bec_setting_();
  bool pimd_qnep_bead_batch_active_() const;
  bool try_compute_pimd_qnep_batch_(
    Box& box,
    GPU_Vector<int>& type,
    std::vector<GPU_Vector<double>>& position_beads,
    std::vector<GPU_Vector<double>>& potential_beads,
    std::vector<GPU_Vector<double>>& force_beads,
    std::vector<GPU_Vector<double>>& virial_beads);
  bool try_compute_pimd_nep_batch_(
    Box& box,
    GPU_Vector<int>& type,
    std::vector<GPU_Vector<double>>& position_beads,
    std::vector<GPU_Vector<double>>& potential_beads,
    std::vector<GPU_Vector<double>>& force_beads,
    std::vector<GPU_Vector<double>>& virial_beads);
  bool try_compute_pimd_dp_batch_(
    Box& box,
    GPU_Vector<int>& type,
    std::vector<GPU_Vector<double>>& position_beads,
    std::vector<GPU_Vector<double>>& potential_beads,
    std::vector<GPU_Vector<double>>& force_beads,
    std::vector<GPU_Vector<double>>& virial_beads);
  Potential* get_pimd_bead_potential_(int device_id) const;
  void refresh_pimd_bead_gpu_workers_();
  bool pimd_qnep_batch_bec_enabled_() const
  {
    return pimd_qnep_batch_bec_mode_ == 1 ||
           (pimd_qnep_batch_bec_mode_ == 0 && pimd_qnep_batch_bec_required_);
  }

  void prepare_compute(
    const int number_of_atoms,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    int* position_image);
  void compute_potentials(
    const int number_of_atoms,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
  void compute_single_potential(
    Potential& potential,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
};
