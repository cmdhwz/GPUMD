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

#ifdef USE_NETCDF

#include "centroid_deltaF_O.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "measure/hac.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "netcdf.h"
#include "netcdf_meta.h"
#include "parse_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#include <fstream>

namespace
{
constexpr int DELTA_FORCE_THREADS = 128;

static __global__ void gpu_collect_oxygen_delta_force(
  const int number_of_oxygen,
  const int number_of_atoms,
  const int* g_oxygen_indices,
  const double* g_fbar,
  const double* g_fc,
  float* g_delta_force)
{
  const int oxygen = blockIdx.x * blockDim.x + threadIdx.x;
  if (oxygen >= number_of_oxygen) return;

  const int atom = g_oxygen_indices[oxygen];
  for (int d = 0; d < 3; ++d) {
    g_delta_force[3 * oxygen + d] = static_cast<float>(
      g_fbar[atom + number_of_atoms * d] - g_fc[atom + number_of_atoms * d]);
  }
}

void check_netcdf(const int status)
{
  if (status != NC_NOERR) {
    fprintf(stderr, "NetCDF error: %s\n", nc_strerror(status));
    exit(1);
  }
}
} // namespace

Centroid_DeltaF_O::Centroid_DeltaF_O(const char** param, const int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("centroid_deltaF_O requires exactly one parameter.\n");
  }
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval for centroid_deltaF_O should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval for centroid_deltaF_O should be positive.\n");
  }
#if !defined(NC_HAS_HDF5) || !NC_HAS_HDF5
  PRINT_INPUT_ERROR(
    "centroid_deltaF_O requires NetCDF4/HDF5 support in the NetCDF-C library.\n");
#endif
  std::ifstream existing_file("centroid_deltaF_O.nc");
  if (existing_file.good()) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O.nc already exists; remove or rename it before starting a new run.\n");
  }
  printf("O-atom centroid force mismatch NetCDF diagnostic:\n");
  printf("    sample interval is %d.\n", sample_interval_);
  property_name = "centroid_deltaF_O";
}

void Centroid_DeltaF_O::preprocess(
  const int number_of_steps,
  const double,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (integrate.type < 31 || integrate.type > 33) {
    PRINT_INPUT_ERROR("centroid_deltaF_O requires a PIMD/RPMD/TRPMD ensemble.\n");
  }
  if (force.compute_hnemd_ || force.compute_hnemdec_ >= 0) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O requires physical forces without HNEMD/HNEMDEC driving.\n");
  }
  if (
    integrate.num_target_pressure_components != 0 || integrate.use_scr_barostat ||
    integrate.deform_x != 0 || integrate.deform_y != 0 || integrate.deform_z != 0 ||
    integrate.deform_xy != 0 || integrate.deform_xz != 0 || integrate.deform_yz != 0) {
    PRINT_INPUT_ERROR("centroid_deltaF_O requires a fixed-cell ring-polymer ensemble.\n");
  }
  if (hac_ == nullptr) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O requires compute_hac with immediate centroid HAC.\n");
  }
  if (!hac_->centroid_force_source_is_immediate()) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O requires an immediate centroid HAC force source; "
      "deferred centroid HAC cannot be reused.\n");
  }
  if (hac_->sample_interval != sample_interval_) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O sample interval must match compute_hac sample interval.\n");
  }

  number_of_atoms_ = atom.number_of_atoms;
  oxygen_atom_ids_.clear();
  for (int n = 0; n < number_of_atoms_; ++n) {
    if (atom.cpu_atom_symbol[n] == "O") {
      oxygen_atom_ids_.push_back(n);
    }
  }
  number_of_oxygen_ = static_cast<int>(oxygen_atom_ids_.size());
  if (number_of_oxygen_ <= 0) {
    PRINT_INPUT_ERROR("centroid_deltaF_O requires at least one O atom.\n");
  }

  oxygen_indices_.resize(number_of_oxygen_);
  oxygen_indices_.copy_from_host(oxygen_atom_ids_.data());
  gpu_delta_force_.resize(static_cast<size_t>(number_of_oxygen_) * 3);
  sampled_steps_.clear();
  sample_times_fs_.clear();
  delta_force_history_.clear();
  reference_positions_.clear();
  reference_position_set_ = false;

  box_matrix_[0] = box.cpu_h[0];
  box_matrix_[1] = box.cpu_h[3];
  box_matrix_[2] = box.cpu_h[6];
  box_matrix_[3] = box.cpu_h[1];
  box_matrix_[4] = box.cpu_h[4];
  box_matrix_[5] = box.cpu_h[7];
  box_matrix_[6] = box.cpu_h[2];
  box_matrix_[7] = box.cpu_h[5];
  box_matrix_[8] = box.cpu_h[8];

  const size_t number_of_samples = static_cast<size_t>(number_of_steps / sample_interval_);
  const size_t values_per_sample = static_cast<size_t>(number_of_oxygen_) * 3;
  const size_t estimated_cache_bytes = number_of_samples *
    (values_per_sample * sizeof(float) + sizeof(double) + sizeof(long long));
  printf(
    "    centroid_deltaF_O cache estimate: %zu frames, %zu bytes (%.3f MiB).\n",
    number_of_samples,
    estimated_cache_bytes,
    static_cast<double>(estimated_cache_bytes) / (1024.0 * 1024.0));
  sampled_steps_.reserve(number_of_samples);
  sample_times_fs_.reserve(number_of_samples);
  delta_force_history_.reserve(number_of_samples * values_per_sample);
}

void Centroid_DeltaF_O::process(
  const int,
  const int step,
  const int,
  const int,
  const double global_time,
  const double,
  Integrate&,
  Box&,
  std::vector<Group>&,
  GPU_Vector<double>&,
  Atom& atom,
  Force&)
{
  const int sampled_step = step + 1;
  if (sampled_step % sample_interval_ != 0) {
    return;
  }
  if (!hac_->centroid_force_ready(sampled_step)) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O did not receive the immediate centroid force for this sample.\n");
  }

  if (!reference_position_set_) {
    std::vector<double> centroid_positions(static_cast<size_t>(number_of_atoms_) * 3);
    atom.position_per_atom.copy_to_host(centroid_positions.data());
    reference_positions_.resize(static_cast<size_t>(number_of_oxygen_) * 3);
    for (int oxygen = 0; oxygen < number_of_oxygen_; ++oxygen) {
      const int atom_id = oxygen_atom_ids_[oxygen];
      for (int d = 0; d < 3; ++d) {
        reference_positions_[3 * oxygen + d] = centroid_positions[atom_id + number_of_atoms_ * d];
      }
    }
    reference_position_set_ = true;
  }

  const GPU_Vector<double>& centroid_force = hac_->centroid_force_per_atom();
  gpu_collect_oxygen_delta_force<<<
    (number_of_oxygen_ - 1) / DELTA_FORCE_THREADS + 1,
    DELTA_FORCE_THREADS>>>(
    number_of_oxygen_,
    number_of_atoms_,
    oxygen_indices_.data(),
    atom.force_per_atom.data(),
    centroid_force.data(),
    gpu_delta_force_.data());
  GPU_CHECK_KERNEL
  const size_t values_per_sample = static_cast<size_t>(number_of_oxygen_) * 3;
  const size_t frame_offset = delta_force_history_.size();
  delta_force_history_.resize(frame_offset + values_per_sample);
  gpu_delta_force_.copy_to_host(delta_force_history_.data() + frame_offset);

  sampled_steps_.push_back(static_cast<long long>(sampled_step));
  sample_times_fs_.push_back(global_time * TIME_UNIT_CONVERSION);
}

void Centroid_DeltaF_O::write_netcdf_()
{
  if (sampled_steps_.empty() || !reference_position_set_) {
    PRINT_INPUT_ERROR("centroid_deltaF_O did not collect any diagnostic frame.\n");
  }

  int ncid = -1;
  const int create_status = nc_create(
    "centroid_deltaF_O.nc", NC_NETCDF4 | NC_NOCLOBBER, &ncid);
  if (create_status == NC_EEXIST) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O.nc already exists; remove or rename it before starting a new run.\n");
  }
  if (create_status == NC_ENOTBUILT || create_status == NC_ENOTNC4) {
    PRINT_INPUT_ERROR(
      "centroid_deltaF_O requires NetCDF4/HDF5 support in the NetCDF-C library.\n");
  }
  check_netcdf(create_status);

  int frame_dim = -1;
  int oxygen_dim = -1;
  int xyz_dim = -1;
  int cell_dim = -1;
  check_netcdf(nc_def_dim(ncid, "frame", NC_UNLIMITED, &frame_dim));
  check_netcdf(nc_def_dim(ncid, "oxygen", number_of_oxygen_, &oxygen_dim));
  check_netcdf(nc_def_dim(ncid, "xyz", 3, &xyz_dim));
  check_netcdf(nc_def_dim(ncid, "cell", 3, &cell_dim));

  int frame_dims[1] = {frame_dim};
  int oxygen_dims[1] = {oxygen_dim};
  int box_dims[2] = {cell_dim, xyz_dim};
  int oxygen_xyz_dims[2] = {oxygen_dim, xyz_dim};
  int delta_dims[3] = {frame_dim, oxygen_dim, xyz_dim};
  int time_var = -1;
  int step_var = -1;
  int box_var = -1;
  int oxygen_id_var = -1;
  int reference_position_var = -1;
  int delta_force_var = -1;
  check_netcdf(nc_def_var(ncid, "time_fs", NC_DOUBLE, 1, frame_dims, &time_var));
  check_netcdf(nc_def_var(ncid, "step", NC_INT64, 1, frame_dims, &step_var));
  check_netcdf(nc_def_var(ncid, "box_matrix", NC_DOUBLE, 2, box_dims, &box_var));
  check_netcdf(nc_def_var(ncid, "O_atom_id", NC_INT, 1, oxygen_dims, &oxygen_id_var));
  check_netcdf(
    nc_def_var(ncid, "O_reference_position", NC_DOUBLE, 2, oxygen_xyz_dims, &reference_position_var));
  check_netcdf(
    nc_def_var(ncid, "deltaF_O", NC_FLOAT, 3, delta_dims, &delta_force_var));

  const char* fs = "fs";
  const char* angstrom = "Angstrom";
  const char* ev_per_angstrom = "eV/Angstrom";
  check_netcdf(nc_put_att_text(ncid, time_var, "units", std::strlen(fs), fs));
  check_netcdf(nc_put_att_text(ncid, box_var, "units", std::strlen(angstrom), angstrom));
  check_netcdf(
    nc_put_att_text(ncid, reference_position_var, "units", std::strlen(angstrom), angstrom));
  check_netcdf(
    nc_put_att_text(ncid, delta_force_var, "units", std::strlen(ev_per_angstrom), ev_per_angstrom));
  check_netcdf(nc_enddef(ncid));

  check_netcdf(nc_put_var_double(ncid, box_var, box_matrix_));
  check_netcdf(nc_put_var_int(ncid, oxygen_id_var, oxygen_atom_ids_.data()));
  check_netcdf(nc_put_var_double(ncid, reference_position_var, reference_positions_.data()));

  check_netcdf(nc_put_var_double(ncid, time_var, sample_times_fs_.data()));
  check_netcdf(nc_put_var_longlong(ncid, step_var, sampled_steps_.data()));
  check_netcdf(nc_put_var_float(ncid, delta_force_var, delta_force_history_.data()));
  check_netcdf(nc_close(ncid));
}

void Centroid_DeltaF_O::postprocess(
  Atom&,
  Box&,
  Integrate&,
  const int,
  const double,
  const double)
{
  write_netcdf_();
}

#endif
