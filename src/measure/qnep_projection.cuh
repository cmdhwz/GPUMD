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
#include <vector>

class NEP_Charge;

class QNEP_Projection : public Action
{
public:
  QNEP_Projection(const char**, int);

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

  int sample_interval_ = 1;
  bool g1_channel_ = false;
  NEP_Charge* qnep_ = nullptr;
  double initial_cell_[9] = {0.0};
  int number_of_pair_blocks_ = 0;
  GPU_Vector<double> gpu_means_;
  GPU_Vector<double> gpu_partial_;
  GPU_Vector<double> gpu_total_;
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
  FILE* fid_ = nullptr;
  FILE* fid_channel_ = nullptr;
  FILE* fid_delta_j_q_k_ = nullptr;
};
