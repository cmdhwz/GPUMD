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
#include "rpmd_ja_utf8_file_io.h"
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
  std::string provenance_path_;
  std::string provenance_temp_path_;
  std::string lock_path_;
  std::string run_input_sha256_;
  std::string physical_force_configuration_sha256_;
  std::string integrator_configuration_sha256_;
  std::string active_configuration_scope_;
  std::string model_sha256_;
  std::string atom_signature_sha256_;
  std::string thermostat_tokens_json_;
  std::string model_path_;
  std::string source_job_id_;
  std::string source_job_id_status_;
  std::string source_run_id_;
  int sample_interval_ = 0;
  int max_stability_rounds_ = 160;
  double cutoff_ = 0.0;
  double epsilon_ = 0.0;
  double response_tolerance_ = 0.0;
  double fd_step_ = 0.0;
  double temperature_ = 0.0;
  double timestep_internal_ = 0.0;
  double timestep_fs_ = 0.0;
  int number_of_atoms_ = 0;
  int number_of_beads_ = 0;
  std::uint64_t frame_count_ = 0;
  unsigned active_run_ordinal_ = 0;
  int first_sample_step_ = -1;
  int last_sample_step_ = -1;
  rpmd_ja_file_io::ofstream spool_;
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
  bool sample_only_ = false;
  bool internal_mass_com_ = false;
  bool provenance_snapshot_initialized_ = false;
  bool configuration_stable_during_sampling_ = true;
};
