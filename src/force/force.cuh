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
#include "utilities/common.cuh"
#include <memory>
#include <stdio.h>
#include <vector>

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

  void
  parse_potential(const char** param, int num_param, const Box& box, const int number_of_atoms);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& mass_per_atom);
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

  void finalize();

  int get_number_of_types(FILE* fid_potential);
  void set_hnemd_parameters(const double, const double, const double);
  void set_hnemdec_parameters(
    const int compute_hnemdec,
    const double hnemd_fe_x,
    const double hnemd_fe_y,
    const double hnemd_fe_z,
    const std::vector<double>& mass,
    const std::vector<int>& type,
    const std::vector<int>& type_size,
    const double T);
  void set_multiple_potentials_mode(std::string mode);
  void set_pimd_bead_gpu_parallel(const int num_devices);
  void set_pimd_bead_neighbor_rebuild(const bool always_rebuild);
  void set_pimd_qnep_bead_batch(const bool enabled);
  void set_pimd_nep_bead_batch(const bool enabled);
  void set_pimd_nep_batch_profile(const bool enabled);
  void set_pimd_nep_batch_geometry_cache(const bool enabled);
  void reset_pimd_nep_batch_profile();
  void print_pimd_nep_batch_profile() const;
  bool pimd_nep_batch_profile_enabled() const { return pimd_nep_batch_profile_enabled_; }
  int get_pimd_bead_gpu_parallel_devices() const { return pimd_bead_gpu_parallel_devices_; }
  int get_pimd_bead_gpu_worker_count() const { return int(pimd_bead_gpu_workers_.size()); }
  bool pimd_bead_gpu_parallel_available() const { return can_use_pimd_bead_gpu_parallel_(); }
  void reset_pimd_bead_timing();
  const PIMD_Bead_Timing& get_pimd_bead_timing() const { return pimd_bead_timing_; }

  bool compute_hnemd_ = false;
  int compute_hnemdec_ = -1;
  double hnemd_fe_[3];
  double temperature = 0;
  double delta_T;
  GPU_Vector<double> coefficient;
  std::vector<std::unique_ptr<Potential>> potentials;

private:
  int number_of_atoms_ = -1;
  bool is_fcp = false;
  bool has_non_nep = false;
  std::string multiple_potentials_mode_ = "observe"; // "observe" or "average"
  int pimd_bead_gpu_parallel_devices_ = 1;
  bool pimd_bead_neighbor_always_rebuild_ = true;
  bool pimd_qnep_bead_batch_enabled_ = false;
  bool pimd_nep_bead_batch_enabled_ = false;
  bool pimd_nep_batch_profile_enabled_ = false;
  bool pimd_nep_batch_geometry_cache_enabled_ = false;
  std::string primary_nep_model_path_;
  std::string atom_types[NUM_ELEMENTS];
  std::unique_ptr<Potential> pimd_nep_single_gpu_batch_potential_;
  std::vector<std::unique_ptr<PIMD_Bead_GPU_Worker>> pimd_bead_gpu_workers_;
  PIMD_Bead_Timing pimd_bead_timing_;

  void check_types(const char* file_potential);
  bool can_use_pimd_bead_gpu_parallel_() const;
  bool can_use_pimd_qnep_batch_() const;
  bool can_use_pimd_nep_batch_() const;
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
  Potential* get_pimd_bead_potential_(int device_id) const;
  void refresh_pimd_bead_gpu_workers_();
};
