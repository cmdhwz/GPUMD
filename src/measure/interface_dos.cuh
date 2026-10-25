/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once
#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class InterfaceGeometry;

struct InterfaceDOSBlockMetadata
{
  double start_ps = 0.0;
  double end_ps = 0.0;
  int samples = 0;
};

class ComputeInterfaceDOS : public Action
{
public:
  ComputeInterfaceDOS(
    const std::vector<std::string>& tokens,
    const std::shared_ptr<InterfaceGeometry>& interface_geometry,
    int number_of_types);

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
  std::shared_ptr<InterfaceGeometry> interface_geometry_;
  int sample_interval_ = 1;
  int correlation_samples_ = 1;
  int block_samples_ = 2;
  int num_omega_ = 2;
  double max_omega_rad_ps_ = 0.0;
  int bin_lo_ = 0;
  int bin_hi_ = 0;
  int type_id_ = -1;
  int number_of_types_ = 0;
  int samples_in_block_ = 0;
  int block_index_ = 0;
  size_t complete_block_capacity_ = 0;
  double block_start_ps_ = 0.0;
  double block_end_ps_ = 0.0;
  double last_step_time_ps_ = 0.0;
  double nominal_step_size_ = 0.0;
  double sample_dt_ps_ = 0.0;
  bool have_step_time_ = false;
  bool have_sample_dt_ = false;
  bool frequency_grid_underresolved_ = false;
  bool active_ = false;
  std::string prefix_;

  std::vector<int> selected_atoms_;
  std::vector<int> selected_local_bins_;
  std::vector<int> atoms_per_bin_;
  GPU_Vector<int> gpu_selected_atoms_;
  GPU_Vector<int> gpu_selected_bins_;
  GPU_Vector<int> gpu_atoms_per_bin_;
  GPU_Vector<int> gpu_error_info_;
  GPU_Vector<double> gpu_velocity_ring_;
  GPU_Vector<double> gpu_vacf_sum_;
  GPU_Vector<double> gpu_vacf_;
  GPU_Vector<double> gpu_vdos_;
  std::vector<double> cpu_vacf_;
  std::vector<double> cpu_vdos_;
  std::vector<InterfaceDOSBlockMetadata> cpu_block_metadata_;
  FILE* fid_vacf_ = nullptr;
  FILE* fid_vdos_ = nullptr;
  size_t allocated_bytes_ = 0;

  void cache_complete_block();
  void output_complete_block(size_t block_index);
  void reset_block();
};
