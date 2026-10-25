/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "interface_atoms.cuh"
#include "interface_atoms_pack.cuh"
#include "interface_geometry.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <limits>

namespace
{
bool checked_sum(const size_t a, const size_t b, size_t& result)
{
  if (b > std::numeric_limits<size_t>::max() - a) return false;
  result = a + b;
  return true;
}

bool checked_bytes(const size_t elements, const size_t element_size, size_t& bytes)
{
  return interface_atoms_checked_product(elements, element_size, bytes);
}

bool output_exists(const std::string& filename)
{
  FILE* file = std::fopen(filename.c_str(), "r");
  if (!file) return false;
  std::fclose(file);
  return true;
}

__global__ void gpu_gather_interface_atoms(
  const int selected_count,
  const int atom_count,
  const int block_frame,
  const int step,
  const int* selected_atoms,
  const double* position,
  const double* unwrapped_position,
  const double* velocity,
  const double* force,
  const double* potential,
  const double velocity_conversion,
  double* block_records,
  int* error_info)
{
  const size_t selected = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (selected >= static_cast<size_t>(selected_count)) return;

  const int atom_id = selected_atoms[selected];
  const size_t row =
    (static_cast<size_t>(block_frame) * selected_count + selected) * INTERFACE_ATOMS_COMPONENTS;
  if (!interface_atoms_pack_record(
        static_cast<size_t>(atom_id),
        atom_count,
        position,
        unwrapped_position,
        velocity,
        force,
        potential,
        velocity_conversion,
        block_records + row)) {
    if (atomicCAS(error_info, 0, 1) == 0) {
      error_info[1] = step;
      error_info[2] = atom_id;
    }
  }
}
} // namespace

DumpInterfaceAtoms::DumpInterfaceAtoms(
  const std::vector<std::string>& tokens,
  const std::shared_ptr<InterfaceGeometry>& interface_geometry,
  Atom& atom)
  : interface_geometry_(interface_geometry)
{
  action_name = "dump_interface_atoms";
  if (tokens.size() != 6 ||
      !is_valid_int(tokens[1], &sample_steps_) ||
      !is_valid_int(tokens[2], &samples_per_block_) ||
      !is_valid_int(tokens[3], &bin_lo_) ||
      !is_valid_int(tokens[4], &bin_hi_) || tokens[5].empty()) {
    PRINT_INPUT_ERROR(
      "dump_interface_atoms syntax: dump_interface_atoms <sample_steps> <samples_per_block> "
      "<bin_lo> <bin_hi> <prefix>.\n");
  }
  if (sample_steps_ <= 0 || samples_per_block_ <= 0 || bin_lo_ < 0 ||
      bin_hi_ <= bin_lo_) {
    PRINT_INPUT_ERROR("dump_interface_atoms requires positive intervals and a nonempty bin range.\n");
  }
  filename_ = tokens[5] + "_atoms.out";
  if (output_exists(filename_)) {
    PRINT_INPUT_ERROR("dump_interface_atoms refuses to overwrite its output file.\n");
  }
  atom.enable_unwrapped_position();
}

void DumpInterfaceAtoms::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  (void)time_step;
  (void)group;
  (void)box;
  (void)force;
  if (!interface_geometry_ || !interface_geometry_->initialized()) {
    PRINT_INPUT_ERROR("dump_interface_atoms requires an earlier initialized interface_geometry.\n");
  }
  if (is_pimd(integrate.get_type())) {
    PRINT_INPUT_ERROR("dump_interface_atoms supports classical MD only.\n");
  }
  if (number_of_steps < 0 ||
      interface_geometry_->atom_bin().size() != static_cast<size_t>(atom.number_of_atoms)) {
    PRINT_INPUT_ERROR("dump_interface_atoms run length or geometry atom count is invalid.\n");
  }
  if (bin_hi_ > interface_geometry_->bin_count()) {
    PRINT_INPUT_ERROR("dump_interface_atoms bin range exceeds the shared geometry.\n");
  }
  if (output_exists(filename_)) {
    PRINT_INPUT_ERROR("dump_interface_atoms refuses to overwrite its output file.\n");
  }

  selected_atoms_.clear();
  selected_count_ = 0;
  for (size_t atom_id = 0; atom_id < interface_geometry_->atom_bin().size(); ++atom_id) {
    const int bin = interface_geometry_->atom_bin()[atom_id];
    if (bin >= bin_lo_ && bin < bin_hi_) ++selected_count_;
  }
  if (selected_count_ > selected_atoms_.max_size()) {
    PRINT_INPUT_ERROR("dump_interface_atoms selected atom count exceeds a host vector limit.\n");
  }
  if (selected_count_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("dump_interface_atoms selected atom count exceeds its kernel atom-ID range.\n");
  }
  selected_atoms_.reserve(selected_count_);
  for (size_t atom_id = 0; atom_id < interface_geometry_->atom_bin().size(); ++atom_id) {
    const int bin = interface_geometry_->atom_bin()[atom_id];
    if (bin >= bin_lo_ && bin < bin_hi_) selected_atoms_.push_back(static_cast<int>(atom_id));
  }
  total_frame_capacity_ = static_cast<size_t>(number_of_steps / sample_steps_);
  sampling_enabled_ = total_frame_capacity_ > 0 && selected_count_ > 0;
  const size_t cpu_frame_capacity = selected_count_ > 0 ? total_frame_capacity_ : 0;

  size_t cpu_record_values = 0;
  size_t gpu_record_values = 0;
  if (!interface_atoms_frame_value_count(
        total_frame_capacity_, selected_count_, cpu_record_values) ||
      !interface_atoms_frame_value_count(
        sampling_enabled_ ? static_cast<size_t>(samples_per_block_) : 0,
        selected_count_,
        gpu_record_values)) {
    PRINT_INPUT_ERROR("dump_interface_atoms record capacity overflows size_t.\n");
  }
  size_t cpu_record_bytes = 0;
  size_t cpu_frame_bytes = 0;
  size_t cpu_id_bytes = 0;
  size_t gpu_record_bytes = 0;
  size_t gpu_id_bytes = 0;
  size_t gpu_error_bytes = 0;
  const size_t gpu_id_count = sampling_enabled_ ? selected_count_ : 0;
  if (!checked_bytes(cpu_record_values, sizeof(double), cpu_record_bytes) ||
      !checked_bytes(cpu_frame_capacity, sizeof(InterfaceAtomsFrame), cpu_frame_bytes) ||
      !checked_bytes(selected_atoms_.capacity(), sizeof(int), cpu_id_bytes) ||
      !checked_bytes(gpu_record_values, sizeof(double), gpu_record_bytes) ||
      !checked_bytes(gpu_id_count, sizeof(int), gpu_id_bytes) ||
      !checked_bytes(3, sizeof(int), gpu_error_bytes)) {
    PRINT_INPUT_ERROR("dump_interface_atoms allocation byte count overflows size_t.\n");
  }
  if (cpu_record_values > cpu_records_.max_size() ||
      cpu_frame_capacity > cpu_frames_.max_size()) {
    PRINT_INPUT_ERROR("dump_interface_atoms allocation exceeds a host vector limit.\n");
  }

  size_t cpu_bytes = 0;
  size_t gpu_bytes = 0;
  if (!checked_sum(cpu_record_bytes, cpu_frame_bytes, cpu_bytes) ||
      !checked_sum(cpu_bytes, cpu_id_bytes, cpu_bytes) ||
      !checked_sum(gpu_record_bytes, gpu_id_bytes, gpu_bytes) ||
      (total_frame_capacity_ > 0 && selected_count_ > 0 &&
       !checked_sum(gpu_bytes, gpu_error_bytes, gpu_bytes))) {
    PRINT_INPUT_ERROR("dump_interface_atoms total allocation byte count overflows size_t.\n");
  }
  std::printf(
    "dump_interface_atoms selected %zu atoms; estimated CPU cache %.3f MiB, GPU buffers %.3f MiB.\n",
    selected_count_, cpu_bytes / (1024.0 * 1024.0), gpu_bytes / (1024.0 * 1024.0));

  cpu_records_.resize(cpu_record_values);
  cpu_frames_.resize(cpu_frame_capacity);
  if (sampling_enabled_) {
    gpu_selected_atoms_.resize(selected_count_);
    gpu_selected_atoms_.copy_from_host(selected_atoms_.data());
    gpu_error_info_.resize(3, 0);
    gpu_block_records_.resize(gpu_record_values);
  }

  output_ = my_fopen(filename_.c_str(), "w");
  std::fprintf(output_, "# dump_interface_atoms version 1\n");
  std::fprintf(
    output_, "# geometry_source %s_geometry.json\n",
    interface_geometry_->output_prefix().c_str());
  std::fprintf(output_, "# bin_window_zero_based_half_open %d %d\n", bin_lo_, bin_hi_);
  std::fprintf(output_, "# sample_steps %d samples_per_block %d\n", sample_steps_, samples_per_block_);
  std::fprintf(output_, "# frame_zero_based atom_id_zero_based_global step_one_based_completed\n");
  std::fprintf(output_, "# units position_A velocity_A_per_ps force_eV_per_A potential_eV\n");
  std::fprintf(output_, "# velocity is sampled after compute2 at the same MD step\n");
  std::fprintf(
    output_, "# force is the current atom.force_per_atom, including existing action contributions\n");
  std::fprintf(output_, "# all selected atoms are kept in global atom-id order, including fixed/moving/bath\n");
  std::fprintf(output_, "# complete and final partial sample blocks are both retained\n");
  std::fprintf(
    output_,
    "# columns: frame step time_ps atom_id x_A y_A z_A xu_A yu_A zu_A vx_A_ps vy_A_ps vz_A_ps "
    "fx_eV_A fy_eV_A fz_eV_A potential_eV\n");
}

void DumpInterfaceAtoms::cache_sample_block()
{
  if (samples_in_block_ == 0) return;

  int error_info[3] = {0, -1, -1};
  gpu_error_info_.copy_to_host(error_info, 3);
  if (error_info[0] != 0) {
    std::fprintf(
      stderr,
      "dump_interface_atoms detected a non-finite sampled value at step %d, atom %d.\n",
      error_info[1], error_info[2]);
    PRINT_INPUT_ERROR("dump_interface_atoms rejected the sample block before caching it.\n");
  }

  size_t values = 0;
  size_t offset = 0;
  if (!interface_atoms_frame_value_count(
        static_cast<size_t>(samples_in_block_), selected_count_, values) ||
      !interface_atoms_frame_offset(frames_cached_ - samples_in_block_, selected_count_, offset)) {
    PRINT_INPUT_ERROR("dump_interface_atoms block cache offset overflows size_t.\n");
  }
  gpu_block_records_.copy_to_host(cpu_records_.data() + offset, values);
  gpu_error_info_.fill(0);
  samples_in_block_ = 0;
}

void DumpInterfaceAtoms::end_of_step(
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
  Force& force)
{
  (void)number_of_steps;
  (void)fixed_group;
  (void)move_group;
  (void)temperature;
  (void)integrate;
  (void)box;
  (void)group;
  (void)thermo;
  (void)force;
  if (!sampling_enabled_ || (step + 1) % sample_steps_ != 0) return;

  const double time_ps = global_time * TIME_UNIT_CONVERSION / 1000.0;
  if (!std::isfinite(time_ps)) {
    PRINT_INPUT_ERROR("dump_interface_atoms received a non-finite sample time.\n");
  }
  if (frames_cached_ >= total_frame_capacity_ || samples_in_block_ >= samples_per_block_) {
    PRINT_INPUT_ERROR("dump_interface_atoms exceeded its precomputed sample capacity.\n");
  }

  cpu_frames_[frames_cached_].step = step + 1;
  cpu_frames_[frames_cached_].time_ps = time_ps;
  const int threads = 256;
  const int blocks = static_cast<int>((selected_count_ + threads - 1) / threads);
  gpu_gather_interface_atoms<<<blocks, threads>>>(
    static_cast<int>(selected_count_),
    atom.number_of_atoms,
    samples_in_block_,
    step + 1,
    gpu_selected_atoms_.data(),
    atom.position_per_atom.data(),
    atom.unwrapped_position.data(),
    atom.velocity_per_atom.data(),
    atom.force_per_atom.data(),
    atom.potential_per_atom.data(),
    1000.0 / TIME_UNIT_CONVERSION,
    gpu_block_records_.data(),
    gpu_error_info_.data());
  GPU_CHECK_KERNEL

  ++frames_cached_;
  ++samples_in_block_;
  if (samples_in_block_ == samples_per_block_) cache_sample_block();
}

void DumpInterfaceAtoms::write_rows()
{
  for (size_t frame = 0; frame < frames_cached_; ++frame) {
    size_t offset = 0;
    if (!interface_atoms_frame_offset(frame, selected_count_, offset)) {
      PRINT_INPUT_ERROR("dump_interface_atoms output row offset overflows size_t.\n");
    }
    const InterfaceAtomsFrame& metadata = cpu_frames_[frame];
    for (size_t selected = 0; selected < selected_count_; ++selected) {
      const size_t row = offset + selected * INTERFACE_ATOMS_COMPONENTS;
      std::fprintf(
        output_, "%zu %d %.17g %d", frame, metadata.step, metadata.time_ps,
        selected_atoms_[selected]);
      for (size_t component = 0; component < INTERFACE_ATOMS_COMPONENTS; ++component) {
        std::fprintf(output_, " %.17g", cpu_records_[row + component]);
      }
      std::fprintf(output_, "\n");
    }
  }
}

void DumpInterfaceAtoms::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  (void)atom;
  (void)box;
  (void)integrate;
  (void)number_of_steps;
  (void)time_step;
  (void)temperature;
  if (sampling_enabled_ && samples_in_block_ > 0) cache_sample_block();
  if (output_) {
    write_rows();
    std::fflush(output_);
    std::fclose(output_);
    output_ = nullptr;
  }
  sampling_enabled_ = false;
}
