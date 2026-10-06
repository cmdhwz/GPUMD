/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

class RpmdJA_Fit final : public Action
{
public:
  explicit RpmdJA_Fit(const std::vector<std::string>& tokens);
  ~RpmdJA_Fit() override;

  void pre_run(
    int number_of_steps,
    double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_force(
    int step,
    double time_step,
    double global_time,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    int number_of_steps,
    double time_step,
    double temperature) override;

private:
  void release_lock_();

  std::string output_path_;
  std::string kernel_table_;
  std::string spool_path_;
  std::string lock_path_;
  int sample_interval_ = 0;
  int max_stability_rounds_ = 160;
  double cutoff_ = 0.0;
  double epsilon_ = 0.0;
  double response_tolerance_ = 0.0;
  double fd_step_ = 0.0;
  double temperature_ = 0.0;
  int number_of_atoms_ = 0;
  int number_of_beads_ = 0;
  std::uint64_t frame_count_ = 0;
  std::ofstream spool_;
  std::FILE* lock_file_ = nullptr;
  Force* force_ = nullptr;
  std::vector<double> masses_;
  std::vector<const double*> bead_position_ptrs_;
  std::vector<const double*> bead_force_ptrs_;
  GPU_Vector<const double*> bead_position_ptrs_gpu_;
  GPU_Vector<const double*> bead_force_ptrs_gpu_;
  GPU_Vector<double> sample_output_gpu_;
  std::vector<double> sample_output_;
  std::vector<double> previous_centroid_;
  std::vector<double> centroid_;
  std::vector<double> mean_force_;
  std::vector<double> frame_buffer_;
  double com_shift_[3] = {0.0, 0.0, 0.0};
  double cell_[9] = {};
  int pbc_[3] = {};
  bool has_previous_centroid_ = false;
};
