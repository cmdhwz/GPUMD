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

struct InterfaceAtomsFrame
{
  int step = 0;
  double time_ps = 0.0;
};

class DumpInterfaceAtoms : public Action
{
public:
  DumpInterfaceAtoms(
    const std::vector<std::string>& tokens,
    const std::shared_ptr<InterfaceGeometry>& interface_geometry,
    Atom& atom);

  void pre_run(
    const int number_of_steps,
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
  int sample_steps_ = 1;
  int samples_per_block_ = 1;
  int bin_lo_ = 0;
  int bin_hi_ = 0;
  std::string filename_;
  FILE* output_ = nullptr;
  bool sampling_enabled_ = false;
  size_t total_frame_capacity_ = 0;
  size_t selected_count_ = 0;
  size_t frames_cached_ = 0;
  int samples_in_block_ = 0;
  std::vector<int> selected_atoms_;
  std::vector<InterfaceAtomsFrame> cpu_frames_;
  std::vector<double> cpu_records_;
  GPU_Vector<int> gpu_selected_atoms_;
  GPU_Vector<int> gpu_error_info_;
  GPU_Vector<double> gpu_block_records_;

  void cache_sample_block();
  void write_rows();
};
