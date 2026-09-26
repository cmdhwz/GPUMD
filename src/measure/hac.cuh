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
#include "action.cuh"
#include "qnep_projection.cuh"
#include "utilities/gpu_vector.cuh"

class NEP_Charge;

class HAC : public Action
{
public:
  HAC(const char**, int, bool qnep_full_a = false);

  void set_qnep_full_a(const bool enabled) { qnep_full_a_ = enabled; }

  bool centroid_force_source_is_immediate() const
  {
    return compute != 0 && use_centroid_heat_flux_ != 0 && !deferred_centroid_enabled_;
  }

  bool centroid_force_ready(const int step) const
  {
    return centroid_force_source_is_immediate() && centroid_force_step_ == step;
  }

  const GPU_Vector<double>& centroid_force_per_atom() const
  {
    return centroid_force_per_atom_;
  }

  const GPU_Vector<double>& centroid_potential_per_atom() const
  {
    return centroid_potential_per_atom_;
  }

  int compute = 0;
  int sample_interval; // sample interval for heat current
  int Nc;              // number of correlation points
  int output_interval; // only output Nc/output_interval data

  bool get_current_for_step(int step, double current[3]) const;

  virtual void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void end_of_step(
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
      Force& force);

  void pre_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  virtual void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

  void parse(const char**, int);

private:
  static constexpr int deferred_centroid_chunk_size_ = 32;

  GPU_Vector<double> heat_all;
  GPU_Vector<double> heat_all_by_type_;
  GPU_Vector<double> heat_all_by_type_electro_;
  GPU_Vector<double> centroid_potential_per_atom_;
  GPU_Vector<double> centroid_force_per_atom_;
  GPU_Vector<double> centroid_virial_per_atom_;
  GPU_Vector<double> centroid_position_work_;
  GPU_Vector<float> centroid_charge_backup_;
  GPU_Vector<float> centroid_bec_backup_;
  GPU_Vector<double> non_electro_potential_per_atom_;
  GPU_Vector<double> non_electro_force_per_atom_;
  GPU_Vector<double> non_electro_virial_per_atom_;
  GPU_Vector<double> electro_potential_per_atom_;
  GPU_Vector<double> electro_virial_per_atom_;
  GPU_Vector<double> electro_heat_per_atom_;
  int use_centroid_heat_flux_ = 0;
  int split_qnep_heat_by_type_ = 0;
  int deferred_centroid_qnep_ = 0;
  bool deferred_centroid_enabled_ = false;
  int centroid_force_step_ = -1;
  Force* force_ = nullptr;
  int centroid_frame_size_ = 0;
  int centroid_frame_count_ = 0;
  int centroid_chunk_frame_count_ = 0;
  GPU_Vector<double> centroid_position_chunk_gpu_;
  GPU_Vector<double> centroid_velocity_chunk_gpu_;
  std::vector<double> centroid_position_frames_cpu_;
  std::vector<double> centroid_velocity_frames_cpu_;
  std::vector<GPU_Vector<double>> deferred_position_frames_gpu_;
  std::vector<GPU_Vector<double>> deferred_velocity_frames_gpu_;
  std::vector<GPU_Vector<double>> deferred_potential_frames_gpu_;
  std::vector<GPU_Vector<double>> deferred_force_frames_gpu_;
  std::vector<GPU_Vector<double>> deferred_virial_frames_gpu_;
  long long centroid_sampled_frames_ = 0;
  long long centroid_direct_evaluations_ = 0;
  double deferred_staging_wall_time_ = 0.0;
  double deferred_upload_wall_time_ = 0.0;
  double deferred_qnep_wall_time_ = 0.0;
  double deferred_heat_wall_time_ = 0.0;
  double deferred_hac_wall_time_ = 0.0;
  bool qnep_full_a_ = false;
  NEP_Charge* qnep_full_a_qnep_ = nullptr;
  QNEP_Full_A_Current_Workspace qnep_full_a_workspace_;
  GPU_Vector<double> qnep_full_a_dynamic_local_channel_per_atom_;
  GPU_Vector<double> qnep_full_a_dynamic_local_channel_total_;
  std::vector<double> qnep_full_a_current_history_;
  GPU_Vector<double> qnep_full_a_base_by_type_current_;
  std::vector<double> qnep_full_a_base_by_type_history_;
  std::vector<double> qnep_full_a_j_conv_history_;
  std::vector<double> qnep_full_a_j_virial_existing_history_;
  std::vector<double> qnep_full_a_j_dyn_local_history_;
  std::vector<double> qnep_full_a_j_virial_remainder_history_;
  std::vector<double> qnep_full_a_j_reference_static_history_;
  std::vector<double> qnep_full_a_j_base_existing_history_;
  std::vector<double> qnep_full_a_j_added_dynamic_history_;
  std::vector<double> qnep_full_a_delta_j_q_pppm_history_;
  std::vector<double> qnep_full_a_delta_j_q_real_history_;
  std::vector<double> qnep_full_a_projection_a_history_;
  std::vector<double> qnep_full_a_closure_error_history_;
  std::vector<double> qnep_full_a_sample_times_fs_;
  std::vector<int> qnep_full_a_sample_steps_;
  double qnep_full_a_initial_cell_[9] = {0.0};
  bool qnep_full_a_local_channel_validation_done_ = false;
  bool qnep_full_a_local_channel_validation_passed_ = false;
  double qnep_full_a_local_channel_validation_error_ = 0.0;

  void flush_deferred_centroid_chunk_();
  void process_deferred_centroid_frames_(
    Atom& atom, Box& box, const int number_of_frames, const int number_of_types, const int Nd);
  void check_qnep_full_a_fixed_cell_(const Box& box) const;
  void post_run_qnep_full_a_(
    Atom& atom,
    Box& box,
    const int number_of_steps,
    const double time_step,
    const double temperature,
    const char* temperature_source);
};
