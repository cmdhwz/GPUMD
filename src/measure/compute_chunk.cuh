/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "action.cuh"
#include "interface_profile_math.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <string>
#include <vector>

class InterfaceGeometry;

class ComputeChunk : public Action
{
public:
  ComputeChunk(const std::vector<std::string>& tokens, Box& box);
  ComputeChunk(
    const std::vector<std::string>& tokens,
    const std::shared_ptr<InterfaceGeometry>& interface_geometry);
  bool uses_legacy_interface_guard() const { return interface_mode_ && !geometry_mode_; }

  virtual void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

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
    Force& force) override;

  virtual void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  // Sampling control
  int sample_interval_ = 1;
  int output_interval_ = 1;

  // Bin configuration
  int dim_;                      // 1 = bin/1d, 2 = bin/2d, 3 = bin/3d
  int axis_[3];                  // Which axes to bin: 0=x, 1=y, 2=z
  double origin_[3];             // Bin origin coordinates
  double delta_[3];              // Bin width per dimension
  double invdelta_[3];           // 1/delta for fast binning
  int nlayers_[3];               // Number of bins per dimension
  double box_length_[3];         // Box length along each binned axis
  int nchunk_;                   // Total number of chunks
  int ncoord_;                   // Number of coordinates per chunk (= dim_)

  // Chunk assignments (GPU)
  GPU_Vector<int> ichunk_;       // ichunk_[i] = chunk ID for atom i

  // Chunk geometry (CPU)
  std::vector<double> chunk_volume_cpu_;
  std::vector<double> chunk_coords_cpu_;

  // Properties to compute
  int compute_temperature_ = 0;
  int compute_potential_ = 0;
  int compute_density_number_ = 0;
  int compute_density_mass_ = 0;
  int compute_vx_ = 0;
  int compute_vy_ = 0;
  int compute_vz_ = 0;
  int compute_fx_ = 0;
  int compute_fy_ = 0;
  int compute_fz_ = 0;
  int number_of_scalars_ = 0;

  // Accumulation arrays
  GPU_Vector<int> gpu_count_;
  GPU_Vector<double> gpu_values_;
  GPU_Vector<double> gpu_auxiliary_;
  GPU_Vector<InterfaceProfileBlockAccum> gpu_interface_block_accum_;
  GPU_Vector<int> gpu_interface_error_info_;
  GPU_Vector<double> gpu_interface_error_values_;
  std::vector<int> cpu_count_sum_;
  std::vector<double> cpu_values_sum_;
  int num_samples_ = 0;

  // Optional one-interface profile output; legacy compute_chunk remains unchanged.
  bool interface_mode_ = false;
  bool geometry_mode_ = false;
  std::shared_ptr<InterfaceGeometry> interface_geometry_;
  bool remove_drift_ = false;
  double interface_coordinate_ = 0.0;
  int material_lower_id_ = 1;
  int material_upper_id_ = 2;
  std::string interface_output_file_;
  double block_start_time_ps_ = 0.0;
  double last_time_ps_ = 0.0;
  int last_step_ = 0;
  double interface_initial_box_h_[9] = {};
  int interface_initial_pbc_[3] = {};
  double last_bath_source_eV_ = 0.0;
  double last_bath_sink_eV_ = 0.0;
  std::size_t interface_expected_sample_count_ = 0;
  std::size_t interface_block_cache_capacity_ = 0;
  std::size_t interface_cached_block_count_ = 0;
  std::vector<InterfaceProfileBlockAccum> cpu_interface_block_cache_;
  std::vector<InterfaceProfileBlockMetadata> cpu_interface_block_metadata_;
  int interface_error_info_host_[3] = {};
  double interface_error_values_host_[INTERFACE_PROFILE_ERROR_VALUE_COUNT] = {};

  // Output
  FILE* fid_ = nullptr;

  void parse(const std::vector<std::string>& tokens, Box& box);
  int parse_bin_params(
    const std::vector<std::string>& tokens, int num_param, int start, Box& box,
    bool require_finite_width);
  void assign_chunks(const Atom& atom, const Box& box);
  void mask_fixed_and_moving_atoms(
    const std::vector<Group>& group, const Integrate& integrate,
    const int fixed_group, const int move_group);
  void sample(const Atom& atom, int step);
  void output_results(int step);
  void cache_interface_block();
  void output_interface_results(std::size_t block_index);
  void reset_interface_block();
  void calculate_chunk_volumes(const Box& box);
  void calculate_chunk_coords(const Box& box);
};
