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
#include "utilities/gpu_vector.cuh"
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>

class NEP_Charge;

class QNEP_Projection : public Action
{
public:
  QNEP_Projection(const char**, int, bool complete_current = false);

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void pre_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_force(
    const int step,
    const double time_step,
    const double global_time,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void end_of_step(
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
    Force& force) override;

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  void parse(const char**, int);
  void check_fixed_cell(const Box&) const;
  void compute_projection_currents(
    const int N,
    const Box& box,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& unwrapped_position,
    const GPU_Vector<float>& D,
    const GPU_Vector<float>& s);
  void sum_virial_current(
    const GPU_Vector<double>& virial,
    const GPU_Vector<double>& velocity,
    double current[3]);
  void write_complete_current(
    const int step,
    const double global_time,
    Box& box,
    Atom& atom,
    const GPU_Vector<double>& velocity);

  int sample_interval_ = 1;
  bool g1_channel_ = false;
  bool complete_current_ = false;
  NEP_Charge* qnep_ = nullptr;
  double initial_cell_[9] = {0.0};
  int number_of_pair_blocks_ = 0;
  GPU_Vector<double> gpu_means_;
  GPU_Vector<double> gpu_partial_;
  GPU_Vector<double> gpu_total_;
  GPU_Vector<double> gpu_current_total_;
  GPU_Vector<double> gpu_velocity_sample_;
  GPU_Vector<double> gpu_diagnostic_potential_;
  GPU_Vector<double> gpu_diagnostic_force_;
  GPU_Vector<double> gpu_diagnostic_virial_;
  std::vector<double> cpu_total_;
  GPU_Vector<double> gpu_channel_per_atom_;
  GPU_Vector<double> gpu_channel_total_;
  GPU_Vector<double> gpu_virial_nep_;
  GPU_Vector<double> gpu_virial_electrostatic_fixed_;
  GPU_Vector<double> gpu_virial_dynamic_charge_;
  GPU_Vector<double> gpu_virial_heat_per_atom_;
  GPU_Vector<double> gpu_virial_heat_total_;
  GPU_Vector<double> gpu_delta_j_q_k_;
  std::vector<double> cpu_channel_total_;
  std::vector<double> cpu_virial_heat_total_;
  std::vector<double> cpu_delta_j_q_k_;
  std::ostringstream output_buffer_;
  std::ostringstream channel_buffer_;
  std::ostringstream delta_j_q_k_buffer_;
  std::ostringstream projection_current_diag_buffer_;
  std::ostringstream complete_current_buffer_;
};
