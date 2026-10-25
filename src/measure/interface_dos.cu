/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "interface_dos.cuh"
#include "interface_geometry.cuh"
#include "interface_sampling_time.cuh"
#include "force/nep_plane_flux_view.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstdio>
#include <limits>

namespace
{
size_t checked_product(const size_t a, const size_t b)
{
  if (a && b > std::numeric_limits<size_t>::max() / a) {
    PRINT_INPUT_ERROR("compute_interface_dos allocation size overflows size_t.\n");
  }
  return a * b;
}

size_t checked_sum(const size_t a, const size_t b)
{
  if (b > std::numeric_limits<size_t>::max() - a) {
    PRINT_INPUT_ERROR("compute_interface_dos allocation byte count overflows size_t.\n");
  }
  return a + b;
}

bool output_exists(const std::string& filename)
{
  FILE* file = std::fopen(filename.c_str(), "r");
  if (!file) return false;
  std::fclose(file);
  return true;
}

__global__ void gpu_gather_interface_velocities(
  const int selected_count,
  const int total_atoms,
  const int correlation_samples,
  const int slot,
  const int* selected_atoms,
  const double* velocity,
    const double velocity_conversion,
  double* ring)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < selected_count) {
    const int atom = selected_atoms[n];
    for (int d = 0; d < 3; ++d) {
      ring[((static_cast<size_t>(slot) * 3 + d) * selected_count) + n] =
        velocity[static_cast<size_t>(d) * total_atoms + atom] * velocity_conversion;
    }
  }
}

__global__ void gpu_accumulate_interface_vacf(
  const int selected_count,
  const int correlation_samples,
  const int slot,
  const int valid_lags,
  const int* local_bins,
  const double* ring,
  double* vacf_sum)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < selected_count) {
    const int bin = local_bins[n];
    for (int lag = 0; lag < valid_lags; ++lag) {
      const int previous_slot = interface_ring_previous_slot(slot, lag, correlation_samples);
      for (int d = 0; d < 3; ++d) {
        const double current = ring[((static_cast<size_t>(slot) * 3 + d) * selected_count) + n];
        const double previous = ring[((static_cast<size_t>(previous_slot) * 3 + d) * selected_count) + n];
        const size_t output = ((static_cast<size_t>(bin) * 3 + d) * correlation_samples) + lag;
        atomicAdd(&vacf_sum[output], current * previous);
      }
    }
  }
}

__global__ void gpu_normalize_interface_vacf(
  const size_t vacf_size,
  const int correlation_samples,
  const int block_samples,
  const int* atoms_per_bin,
  const double* vacf_sum,
  double* vacf,
  int* error_info)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vacf_size) return;
  const int lag = static_cast<int>(index % correlation_samples);
  const size_t channel = index / correlation_samples;
  const size_t bin = channel / 3;
  const int atom_count = atoms_per_bin[bin];
  const double nan = NAN;
  const double c0 = atom_count > 0
    ? vacf_sum[channel * correlation_samples] /
        (static_cast<double>(block_samples) * atom_count)
    : nan;
  double raw = nan;
  double normalized = nan;
  const int status = interface_vacf_normalize_sample(
    vacf_sum[index], c0, block_samples, lag, atom_count, raw, normalized);
  if (status == INTERFACE_VACF_SAMPLE_INVALID) {
    if (atomicCAS(error_info, 0, 1) == 0) error_info[1] = static_cast<int>(index);
    vacf[index] = nan;
    return;
  }
  vacf[index] = raw;
}

__global__ void gpu_interface_vdos(
  const int bin_count,
  const int correlation_samples,
  const int num_omega,
  const double max_omega,
  const double dt_ps,
  const double* vacf,
  double* vdos)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(bin_count) * 3 * num_omega;
  if (index < total) {
    const int omega_id = index % num_omega;
    const size_t channel = index / num_omega;
    const double omega = max_omega * omega_id / (num_omega - 1);
    double spectrum = 0.0;
    for (int lag = 0; lag < correlation_samples; ++lag) {
      const double window = interface_hann_window(lag, correlation_samples);
      const double factor = interface_one_sided_factor(lag);
      spectrum += factor * window * vacf[channel * correlation_samples + lag] *
        cos(omega * lag * dt_ps);
    }
    vdos[index] = spectrum * dt_ps;
  }
}
} // namespace

ComputeInterfaceDOS::ComputeInterfaceDOS(
  const std::vector<std::string>& tokens,
  const std::shared_ptr<InterfaceGeometry>& interface_geometry,
  const int number_of_types)
  : interface_geometry_(interface_geometry), number_of_types_(number_of_types)
{
  action_name = "compute_interface_dos";
  if (tokens.size() != 10 ||
      !is_valid_int(tokens[1], &sample_interval_) ||
      !is_valid_int(tokens[2], &correlation_samples_) ||
      !is_valid_int(tokens[3], &block_samples_) ||
      !is_valid_int(tokens[4], &num_omega_) ||
      !is_valid_real(tokens[5], &max_omega_rad_ps_) ||
      !is_valid_int(tokens[6], &bin_lo_) ||
      !is_valid_int(tokens[7], &bin_hi_) ||
      !is_valid_int(tokens[8], &type_id_)) {
    PRINT_INPUT_ERROR(
      "compute_interface_dos syntax: <sample_interval> <Nc> <block_samples> <num_omega> "
      "<max_omega_rad_ps> <bin_lo> <bin_hi> <type_id|-1> <prefix>.\n");
  }
  if (sample_interval_ <= 0 || correlation_samples_ <= 0 ||
      static_cast<long long>(block_samples_) < 2LL * correlation_samples_ || num_omega_ < 2 ||
      !std::isfinite(max_omega_rad_ps_) || max_omega_rad_ps_ <= 0.0 ||
      type_id_ < -1 || type_id_ >= number_of_types_ || tokens[9].empty()) {
    PRINT_INPUT_ERROR("Invalid compute_interface_dos range or sampling parameter.\n");
  }
  prefix_ = tokens[9];
}

void ComputeInterfaceDOS::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!interface_geometry_ || !interface_geometry_->initialized()) {
    PRINT_INPUT_ERROR("compute_interface_dos requires initialized interface geometry.\n");
  }
  if (is_pimd(integrate.get_type())) {
    PRINT_INPUT_ERROR("compute_interface_dos supports classical MD only; PIMD/RPMD velocities are not accepted.\n");
  }
  if (interface_geometry_->atom_bin().size() != static_cast<size_t>(atom.number_of_atoms)) {
    PRINT_INPUT_ERROR("compute_interface_dos atom count differs from shared geometry.\n");
  }
  if (bin_lo_ < 0 || bin_hi_ <= bin_lo_ || bin_hi_ > interface_geometry_->bin_count()) {
    PRINT_INPUT_ERROR("compute_interface_dos uses a zero-based half-open bin range [lo, hi).\n");
  }
  if (number_of_steps < 0) {
    PRINT_INPUT_ERROR("compute_interface_dos requires a nonnegative run length.\n");
  }
  nominal_step_size_ = time_step;
  double nominal_dt_ps = 0.0;
  if (!interface_sample_interval_ps(
        time_step, TIME_UNIT_CONVERSION, sample_interval_, nominal_dt_ps) ||
      max_omega_rad_ps_ > PI / nominal_dt_ps) {
    PRINT_INPUT_ERROR("compute_interface_dos max omega exceeds the sampling Nyquist limit.\n");
  }
  frequency_grid_underresolved_ = !interface_shc_frequency_grid_resolved(
    nominal_dt_ps, max_omega_rad_ps_, num_omega_, correlation_samples_);
  if (block_samples_ > number_of_steps / sample_interval_) {
    PRINT_INPUT_ERROR("compute_interface_dos block_samples exceeds available samples in this run.\n");
  }

  const int bins_in_window = bin_hi_ - bin_lo_;
  atoms_per_bin_.resize(bins_in_window, 0);
  for (size_t n = 0; n < interface_geometry_->atom_bin().size(); ++n) {
    const int bin = interface_geometry_->atom_bin()[n];
    if (bin < bin_lo_ || bin >= bin_hi_ || interface_geometry_->atom_bath()[n] != 0 ||
        interface_geometry_->atom_fixed()[n] || interface_geometry_->atom_moving()[n] ||
        (type_id_ >= 0 && interface_geometry_->atom_type()[n] != type_id_)) {
      continue;
    }
    selected_atoms_.emplace_back(static_cast<int>(n));
    selected_local_bins_.emplace_back(bin - bin_lo_);
    ++atoms_per_bin_[bin - bin_lo_];
  }

  const size_t selected_count = selected_atoms_.size();
  const size_t ring_size = checked_product(checked_product(selected_count, correlation_samples_), 3);
  const size_t channels = checked_product(static_cast<size_t>(bins_in_window), 3);
  const size_t vacf_size = checked_product(channels, correlation_samples_);
  const size_t vdos_size = checked_product(channels, num_omega_);
  const size_t total_samples = static_cast<size_t>(number_of_steps) / sample_interval_;
  complete_block_capacity_ = total_samples / block_samples_;
  const size_t cached_vacf_size = checked_product(complete_block_capacity_, vacf_size);
  const size_t cached_vdos_size = checked_product(complete_block_capacity_, vdos_size);
  if (selected_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      vacf_size > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      ring_size > std::numeric_limits<size_t>::max() / sizeof(double) ||
      vacf_size > std::numeric_limits<size_t>::max() / sizeof(double) ||
      vdos_size > std::numeric_limits<size_t>::max() / sizeof(double) ||
      vdos_size > static_cast<size_t>(std::numeric_limits<int>::max()) * 128) {
    PRINT_INPUT_ERROR("compute_interface_dos allocation size is outside the supported range.\n");
  }
  size_t cpu_cache_bytes = checked_sum(
    checked_product(checked_sum(cached_vacf_size, cached_vdos_size), sizeof(double)),
    checked_product(complete_block_capacity_, sizeof(InterfaceDOSBlockMetadata)));
  cpu_cache_bytes = checked_sum(
    cpu_cache_bytes,
    checked_product(checked_sum(checked_product(selected_count, 2), bins_in_window), sizeof(int)));
  size_t allocated_elements = checked_sum(ring_size, checked_product(vacf_size, 2));
  allocated_elements = checked_sum(allocated_elements, vdos_size);
  const size_t selection_elements = checked_sum(
    checked_product(selected_count, 2), checked_sum(bins_in_window, 2));
  allocated_bytes_ = checked_sum(
    checked_product(allocated_elements, sizeof(double)),
    checked_product(selection_elements, sizeof(int)));
  if (cached_vacf_size > std::vector<double>().max_size() ||
      cached_vdos_size > std::vector<double>().max_size() ||
      complete_block_capacity_ > std::vector<InterfaceDOSBlockMetadata>().max_size()) {
    PRINT_INPUT_ERROR("compute_interface_dos host cache exceeds vector capacity.\n");
  }
  if (selected_count) {
    gpu_selected_atoms_.resize(selected_count);
    gpu_selected_bins_.resize(selected_count);
    gpu_velocity_ring_.resize(ring_size);
    gpu_selected_atoms_.copy_from_host(selected_atoms_.data(), selected_count);
    gpu_selected_bins_.copy_from_host(selected_local_bins_.data(), selected_count);
  }
  gpu_atoms_per_bin_.resize(bins_in_window);
  gpu_atoms_per_bin_.copy_from_host(atoms_per_bin_.data(), bins_in_window);
  gpu_vacf_sum_.resize(vacf_size);
  gpu_vacf_.resize(vacf_size);
  gpu_vdos_.resize(vdos_size);
  gpu_error_info_.resize(2);
  cpu_vacf_.resize(cached_vacf_size);
  cpu_vdos_.resize(cached_vdos_size);
  cpu_block_metadata_.resize(complete_block_capacity_);
  printf(
    "Interface DOS buffers allocate GPU %zu bytes, CPU %zu bytes (%zu complete blocks).\n",
    allocated_bytes_, cpu_cache_bytes, complete_block_capacity_);

  const std::string vacf_name = prefix_ + "_vacf.out";
  const std::string vdos_name = prefix_ + "_vdos.out";
  if (output_exists(vacf_name) || output_exists(vdos_name)) {
    PRINT_INPUT_ERROR("compute_interface_dos output already exists; choose a new prefix.\n");
  }
  fid_vacf_ = my_fopen(vacf_name.c_str(), "w");
  fid_vdos_ = my_fopen(vdos_name.c_str(), "w");
  std::fprintf(fid_vacf_, "# compute_interface_dos_vacf version 1\n# geometry_source %s_geometry.json\n", interface_geometry_->output_prefix().c_str());
  std::fprintf(fid_vacf_, "# classical-MD VACF per selected atom and lag origins; raw correlation and lag0-normalized correlation are both retained\n");
  std::fprintf(fid_vacf_, "# type_id -1 means all types; fixed, moving, and bath atoms excluded; bin range [bin_lo,bin_hi)\n");
  std::fprintf(fid_vacf_, "# sample_interval %d Nc %d block_samples %d selected_type %d bins [%d,%d)\n",
    sample_interval_, correlation_samples_, block_samples_, type_id_, bin_lo_, bin_hi_);
  std::fprintf(fid_vacf_, "# columns block block_start_ps block_end_ps samples origins bin_id type_id direction lag time_ps vacf_raw_A2_ps2 vacf_normalized\n");
  std::fprintf(fid_vdos_, "# compute_interface_dos_vdos version 2\n# geometry_source %s_geometry.json\n", interface_geometry_->output_prefix().c_str());
  std::fprintf(fid_vdos_, "# classical-MD one-sided spectrum S(omega)=dt*[C0+2*sum(lag>0)]; integral_0^Nyquist S dOmega/pi equals C0\n");
  std::fprintf(fid_vdos_, "# sample_interval %d Nc %d block_samples %d num_omega %d max_omega_rad_ps %.12g dt_ps %.12g nyquist_rad_ps %.12g band_limited %d selected_type %d bins [%d,%d)\n",
    sample_interval_, correlation_samples_, block_samples_, num_omega_, max_omega_rad_ps_,
    nominal_dt_ps, PI / nominal_dt_ps, max_omega_rad_ps_ < PI / nominal_dt_ps, type_id_, bin_lo_, bin_hi_);
  std::fprintf(fid_vdos_, "# frequency_grid_underresolved %d; exported_grid_trapezoid is numeric integration of the written rows and may differ from the analytic integral\n", frequency_grid_underresolved_);
  std::fprintf(fid_vdos_, "# columns block block_start_ps block_end_ps samples origins_lag0 bin_id type_id direction omega_rad_ps raw_C0_A2_ps2 analytic_integral_0_max_over_pi exported_grid_trapezoid_0_max_over_pi vdos_A2_ps\n");
  std::fflush(fid_vacf_);
  std::fflush(fid_vdos_);
  reset_block();
  samples_in_block_ = 0;
  block_index_ = 0;
  last_step_time_ps_ = 0.0;
  sample_dt_ps_ = 0.0;
  have_step_time_ = false;
  have_sample_dt_ = false;
  active_ = true;
}

void ComputeInterfaceDOS::pre_force(
  const int step,
  const double time_step,
  Integrate&,
  std::vector<Group>&,
  Atom&,
  Box&,
  Force&)
{
  if (!active_) return;
  if (!interface_time_step_matches(time_step, nominal_step_size_)) {
    PRINT_INPUT_ERROR("compute_interface_dos rejects adaptive or changing time steps.\n");
  }
  if (step == 0) {
    if (!interface_sample_interval_ps(
          time_step, TIME_UNIT_CONVERSION, sample_interval_, sample_dt_ps_) ||
        max_omega_rad_ps_ > PI / sample_dt_ps_) {
      PRINT_INPUT_ERROR("compute_interface_dos observed time step exceeds its sampling Nyquist limit.\n");
    }
    frequency_grid_underresolved_ = !interface_shc_frequency_grid_resolved(
      sample_dt_ps_, max_omega_rad_ps_, num_omega_, correlation_samples_);
    have_sample_dt_ = true;
  } else if (!have_sample_dt_) {
    PRINT_INPUT_ERROR("compute_interface_dos did not observe the first integration step.\n");
  }
}

void ComputeInterfaceDOS::end_of_step(
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
  if (!active_) return;
  const double now_ps = global_time * TIME_UNIT_CONVERSION / 1000.0;
  if (!interface_time_ps_is_increasing(now_ps, last_step_time_ps_, have_step_time_)) {
    PRINT_INPUT_ERROR("compute_interface_dos requires finite increasing time.\n");
  }
  last_step_time_ps_ = now_ps;
  have_step_time_ = true;
  if ((step + 1) % sample_interval_ != 0) return;
  if (!have_sample_dt_) PRINT_INPUT_ERROR("compute_interface_dos has no observed sample interval.\n");
  if (samples_in_block_ == 0) block_start_ps_ = now_ps;
  block_end_ps_ = now_ps;
  const int slot = interface_ring_slot(samples_in_block_, correlation_samples_);
  const int valid_lags = interface_valid_lag_count(samples_in_block_, correlation_samples_);
  const int selected_count = static_cast<int>(selected_atoms_.size());
  if (selected_count) {
    gpu_gather_interface_velocities<<<(selected_count - 1) / 128 + 1, 128>>>(
      selected_count,
      atom.number_of_atoms,
      correlation_samples_,
      slot,
      gpu_selected_atoms_.data(),
      atom.velocity_per_atom.data(),
      1000.0 / TIME_UNIT_CONVERSION,
      gpu_velocity_ring_.data());
    GPU_CHECK_KERNEL
    gpu_accumulate_interface_vacf<<<(selected_count - 1) / 128 + 1, 128>>>(
      selected_count,
      correlation_samples_,
      slot,
      valid_lags,
      gpu_selected_bins_.data(),
      gpu_velocity_ring_.data(),
      gpu_vacf_sum_.data());
    GPU_CHECK_KERNEL
  }
  ++samples_in_block_;
  if (samples_in_block_ == block_samples_) {
    cache_complete_block();
    ++block_index_;
    reset_block();
  }
}

void ComputeInterfaceDOS::cache_complete_block()
{
  const int bins_in_window = bin_hi_ - bin_lo_;
  const size_t channels = static_cast<size_t>(bins_in_window) * 3;
  const size_t vacf_size = channels * correlation_samples_;
  const int threads = 128;
  gpu_normalize_interface_vacf<<<static_cast<int>((vacf_size + threads - 1) / threads), threads>>>(
    vacf_size, correlation_samples_, block_samples_, gpu_atoms_per_bin_.data(),
    gpu_vacf_sum_.data(), gpu_vacf_.data(), gpu_error_info_.data());
  GPU_CHECK_KERNEL
  int error_info[2] = {};
  gpu_error_info_.copy_to_host(error_info, 2);
  if (error_info[0] != 0) {
    const int channel = error_info[1] / correlation_samples_;
    const int lag = error_info[1] % correlation_samples_;
    const int local_bin = channel / 3;
    std::fprintf(
      stderr,
      "compute_interface_dos: non-finite populated-bin VACF at bin %d, direction %d, lag %d.\n",
      bin_lo_ + local_bin, channel % 3, lag);
    PRINT_INPUT_ERROR("compute_interface_dos rejected a non-finite block VACF before caching.\n");
  }

  const size_t vdos_size = channels * num_omega_;
  const int blocks = static_cast<int>((vdos_size + threads - 1) / threads);
  gpu_interface_vdos<<<blocks, threads>>>(
    bins_in_window,
    correlation_samples_,
    num_omega_,
    max_omega_rad_ps_,
    sample_dt_ps_,
    gpu_vacf_.data(),
    gpu_vdos_.data());
  GPU_CHECK_KERNEL

  const size_t vacf_offset = static_cast<size_t>(block_index_) * vacf_size;
  const size_t vdos_offset = static_cast<size_t>(block_index_) * vdos_size;
  gpu_vacf_.copy_to_host(cpu_vacf_.data() + vacf_offset, vacf_size);
  gpu_vdos_.copy_to_host(cpu_vdos_.data() + vdos_offset, vdos_size);
  for (int local_bin = 0; local_bin < bins_in_window; ++local_bin) {
    if (atoms_per_bin_[local_bin] == 0) continue;
    for (int d = 0; d < 3; ++d) {
      const size_t channel = static_cast<size_t>(local_bin * 3 + d);
      const size_t raw_base = vacf_offset + channel * correlation_samples_;
      const size_t spectrum_base = vdos_offset + channel * num_omega_;
      for (int lag = 0; lag < correlation_samples_; ++lag) {
        if (!std::isfinite(cpu_vacf_[raw_base + lag])) {
          PRINT_INPUT_ERROR("compute_interface_dos rejected a non-finite cached VACF for a populated bin.\n");
        }
      }
      for (int omega = 0; omega < num_omega_; ++omega) {
        if (!std::isfinite(cpu_vdos_[spectrum_base + omega])) {
          PRINT_INPUT_ERROR("compute_interface_dos rejected a non-finite cached VDOS for a populated bin.\n");
        }
      }
    }
  }
  cpu_block_metadata_[block_index_].start_ps = block_start_ps_;
  cpu_block_metadata_[block_index_].end_ps = block_end_ps_;
  cpu_block_metadata_[block_index_].samples = samples_in_block_;
}

void ComputeInterfaceDOS::output_complete_block(const size_t block_index)
{
  const int bins_in_window = bin_hi_ - bin_lo_;
  const size_t channels = static_cast<size_t>(bins_in_window) * 3;
  const size_t vacf_size = channels * correlation_samples_;
  const size_t vdos_size = channels * num_omega_;
  const size_t vacf_offset = block_index * vacf_size;
  const size_t vdos_offset = block_index * vdos_size;
  const InterfaceDOSBlockMetadata& metadata = cpu_block_metadata_[block_index];
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const char* directions[3] = {"x", "y", "z"};
  for (int local_bin = 0; local_bin < bins_in_window; ++local_bin) {
    for (int d = 0; d < 3; ++d) {
      const size_t channel = static_cast<size_t>(local_bin * 3 + d);
      const size_t raw_base = vacf_offset + channel * correlation_samples_;
      const double c0 = cpu_vacf_[raw_base];
      const bool c0_valid = atoms_per_bin_[local_bin] > 0 && std::isfinite(c0);
      const size_t spectrum_base = vdos_offset + channel * num_omega_;
      double frequency_integral = c0 * max_omega_rad_ps_;
      for (int lag = 1; lag < correlation_samples_; ++lag) {
        const double tau = lag * sample_dt_ps_;
        frequency_integral += 2.0 * interface_hann_window(lag, correlation_samples_) *
          cpu_vacf_[raw_base + lag] * std::sin(max_omega_rad_ps_ * tau) / tau;
      }
      frequency_integral *= sample_dt_ps_ / PI;
      const double grid_integral = interface_shc_trapezoid_integral(
        cpu_vdos_.data() + spectrum_base, num_omega_, max_omega_rad_ps_) / PI;
      for (int lag = 0; lag < correlation_samples_; ++lag) {
        const int origins = interface_vacf_origin_count(block_samples_, lag);
        const double raw = cpu_vacf_[raw_base + lag];
        double normalized = nan;
        if (c0_valid && c0 > 0.0 && std::isfinite(raw)) {
          normalized = raw / c0;
          if (!std::isfinite(normalized)) normalized = nan;
        }
        std::fprintf(
          fid_vacf_, "%d %.12g %.12g %d %d %d %d %s %d %.12g %.12g %.12g\n",
          static_cast<int>(block_index), metadata.start_ps, metadata.end_ps, metadata.samples,
          origins, bin_lo_ + local_bin, type_id_, directions[d], lag,
          lag * sample_dt_ps_, raw, normalized);
      }
      for (int omega = 0; omega < num_omega_; ++omega) {
        const double frequency = max_omega_rad_ps_ * omega / (num_omega_ - 1);
        std::fprintf(
          fid_vdos_, "%d %.12g %.12g %d %d %d %d %s %.12g %.12g %.12g %.12g %.12g\n",
          static_cast<int>(block_index), metadata.start_ps, metadata.end_ps, metadata.samples,
          block_samples_, bin_lo_ + local_bin, type_id_, directions[d], frequency, c0,
          frequency_integral, grid_integral, cpu_vdos_[spectrum_base + omega]);
      }
    }
  }
}

void ComputeInterfaceDOS::reset_block()
{
  gpu_vacf_sum_.fill(0.0);
  gpu_error_info_.fill(0);
  samples_in_block_ = 0;
}

void ComputeInterfaceDOS::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (samples_in_block_ > 0) {
    printf(
      "compute_interface_dos dropped %d incomplete tail samples after %d complete blocks.\n",
      samples_in_block_, block_index_);
  }
  for (int block = 0; block < block_index_; ++block) {
    output_complete_block(static_cast<size_t>(block));
  }
  if (fid_vacf_) std::fflush(fid_vacf_);
  if (fid_vdos_) std::fflush(fid_vdos_);
  if (fid_vacf_) std::fclose(fid_vacf_);
  if (fid_vdos_) std::fclose(fid_vdos_);
  fid_vacf_ = nullptr;
  fid_vdos_ = nullptr;
  active_ = false;
}
