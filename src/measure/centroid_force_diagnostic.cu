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

#include "centroid_force_diagnostic.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "measure/hac.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <limits>

namespace
{
constexpr int DIAGNOSTIC_COUNT = 0;
constexpr int DIAGNOSTIC_DELTA_SQUARED = 1;
constexpr int DIAGNOSTIC_DELTA_ABS = 2;
constexpr int DIAGNOSTIC_DELTA_MAX = 3;
constexpr int DIAGNOSTIC_FBAR_SQUARED = 4;
constexpr int DIAGNOSTIC_FC_SQUARED = 5;
constexpr int DIAGNOSTIC_POWER = 6;
constexpr int DIAGNOSTIC_KINETIC = 7;
constexpr int DIAGNOSTIC_POTENTIAL = 8;
constexpr int DIAGNOSTIC_STATISTICS = 9;
constexpr int DIAGNOSTIC_SPECIES = 4;
constexpr int DIAGNOSTIC_BLOCKS = DIAGNOSTIC_SPECIES + 1;
constexpr int DIAGNOSTIC_THREADS = 128;

static const char* const DIAGNOSTIC_SPECIES_NAMES[DIAGNOSTIC_SPECIES] = {"H", "O", "Na", "Cl"};

static __global__ void gpu_reduce_centroid_force_diagnostic(
  const int number_of_atoms,
  const int number_of_types,
  const int* g_type,
  const int* g_species_by_type,
  const double* g_fbar,
  const double* g_fc,
  const double* g_vc,
  const double* g_mass,
  const double* g_potential,
  double* g_statistics)
{
  __shared__ double s_statistics[DIAGNOSTIC_STATISTICS][DIAGNOSTIC_THREADS];
  const int tid = threadIdx.x;
  for (int statistic = 0; statistic < DIAGNOSTIC_STATISTICS; ++statistic) {
    s_statistics[statistic][tid] = 0.0;
  }

  const int species = static_cast<int>(blockIdx.x) - 1;
  for (int n = tid; n < number_of_atoms; n += blockDim.x) {
    if (
      species >= 0 &&
      (g_type[n] < 0 || g_type[n] >= number_of_types || g_species_by_type[g_type[n]] != species)) {
      continue;
    }

    const double fbar_x = g_fbar[n];
    const double fbar_y = g_fbar[number_of_atoms + n];
    const double fbar_z = g_fbar[2 * number_of_atoms + n];
    const double fc_x = g_fc[n];
    const double fc_y = g_fc[number_of_atoms + n];
    const double fc_z = g_fc[2 * number_of_atoms + n];
    const double delta_x = fbar_x - fc_x;
    const double delta_y = fbar_y - fc_y;
    const double delta_z = fbar_z - fc_z;
    const double delta_squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    const double delta_abs = sqrt(delta_squared);
    const double vc_x = g_vc[n];
    const double vc_y = g_vc[number_of_atoms + n];
    const double vc_z = g_vc[2 * number_of_atoms + n];
    const double fbar_squared = fbar_x * fbar_x + fbar_y * fbar_y + fbar_z * fbar_z;
    const double fc_squared = fc_x * fc_x + fc_y * fc_y + fc_z * fc_z;
    const double power = vc_x * delta_x + vc_y * delta_y + vc_z * delta_z;
    const double kinetic = 0.5 * g_mass[n] * (vc_x * vc_x + vc_y * vc_y + vc_z * vc_z);

    s_statistics[DIAGNOSTIC_COUNT][tid] += 1.0;
    s_statistics[DIAGNOSTIC_DELTA_SQUARED][tid] += delta_squared;
    s_statistics[DIAGNOSTIC_DELTA_ABS][tid] += delta_abs;
    if (delta_abs > s_statistics[DIAGNOSTIC_DELTA_MAX][tid]) {
      s_statistics[DIAGNOSTIC_DELTA_MAX][tid] = delta_abs;
    }
    s_statistics[DIAGNOSTIC_FBAR_SQUARED][tid] += fbar_squared;
    s_statistics[DIAGNOSTIC_FC_SQUARED][tid] += fc_squared;
    s_statistics[DIAGNOSTIC_POWER][tid] += power;
    s_statistics[DIAGNOSTIC_KINETIC][tid] += kinetic;
    s_statistics[DIAGNOSTIC_POTENTIAL][tid] += g_potential[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int statistic = 0; statistic < DIAGNOSTIC_STATISTICS; ++statistic) {
        if (statistic == DIAGNOSTIC_DELTA_MAX) {
          if (s_statistics[statistic][tid + offset] > s_statistics[statistic][tid]) {
            s_statistics[statistic][tid] = s_statistics[statistic][tid + offset];
          }
        } else {
          s_statistics[statistic][tid] += s_statistics[statistic][tid + offset];
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int statistic = 0; statistic < DIAGNOSTIC_STATISTICS; ++statistic) {
      g_statistics[static_cast<int>(blockIdx.x) * DIAGNOSTIC_STATISTICS + statistic] =
        s_statistics[statistic][0];
    }
  }
}

double rms_from_statistics(const double* statistics, const double count, const int offset)
{
  return count > 0.0
    ? sqrt(statistics[offset] / count)
    : std::numeric_limits<double>::quiet_NaN();
}

double relative_rms(const double delta_rms, const double fbar_rms)
{
  if (!std::isfinite(delta_rms) || !std::isfinite(fbar_rms) || !(fbar_rms > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double relative = delta_rms / fbar_rms;
  return std::isfinite(relative) ? relative : std::numeric_limits<double>::quiet_NaN();
}
} // namespace

Centroid_Force_Diagnostic::Centroid_Force_Diagnostic(const char** param, const int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("centroid_force_diagnostic requires exactly one parameter.\n");
  }
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval for centroid_force_diagnostic should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval for centroid_force_diagnostic should be positive.\n");
  }
  printf("Centroid force mismatch diagnostic:\n");
  printf("    sample interval is %d.\n", sample_interval_);
  property_name = "centroid_force_diagnostic";
}

void Centroid_Force_Diagnostic::preprocess(
  const int,
  const double,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box&,
  Force& force)
{
  if (integrate.type < 31 || integrate.type > 33) {
    PRINT_INPUT_ERROR(
      "centroid_force_diagnostic requires a PIMD/RPMD/TRPMD ensemble.\n");
  }
  if (force.compute_hnemd_ || force.compute_hnemdec_ >= 0) {
    PRINT_INPUT_ERROR(
      "centroid_force_diagnostic requires physical forces without HNEMD/HNEMDEC driving.\n");
  }
  if (
    integrate.num_target_pressure_components != 0 || integrate.use_scr_barostat ||
    integrate.deform_x != 0 || integrate.deform_y != 0 || integrate.deform_z != 0 ||
    integrate.deform_xy != 0 || integrate.deform_xz != 0 || integrate.deform_yz != 0) {
    PRINT_INPUT_ERROR(
      "centroid_force_diagnostic requires a fixed-cell ring-polymer ensemble.\n");
  }
  if (hac_ == nullptr) {
    PRINT_INPUT_ERROR(
      "centroid_force_diagnostic requires compute_hac with immediate centroid HAC.\n");
  }
  if (!hac_->centroid_force_source_is_immediate()) {
    PRINT_INPUT_ERROR(
      "centroid_force_diagnostic requires an immediate centroid HAC force source; "
      "deferred centroid HAC cannot be reused.\n");
  }
  if (hac_->sample_interval != sample_interval_) {
    PRINT_INPUT_ERROR(
      "centroid_force_diagnostic sample interval must match compute_hac sample interval.\n");
  }

  number_of_atoms_ = atom.number_of_atoms;
  number_of_types_ = static_cast<int>(atom.cpu_type_size.size());
  std::vector<int> species_by_type_cpu(number_of_types_, -1);
  for (int n = 0; n < number_of_atoms_; ++n) {
    const int type = atom.cpu_type[n];
    if (type < 0 || type >= number_of_types_) {
      PRINT_INPUT_ERROR("centroid_force_diagnostic found an invalid atom type mapping.\n");
    }
    for (int species = 0; species < DIAGNOSTIC_SPECIES; ++species) {
      if (atom.cpu_atom_symbol[n] == DIAGNOSTIC_SPECIES_NAMES[species]) {
        if (species_by_type_cpu[type] != -1 && species_by_type_cpu[type] != species) {
          PRINT_INPUT_ERROR(
            "centroid_force_diagnostic found a type mapped to multiple species.\n");
        }
        species_by_type_cpu[type] = species;
        break;
      }
    }
  }

  species_by_type_.resize(number_of_types_);
  species_by_type_.copy_from_host(species_by_type_cpu.data());
  gpu_statistics_.resize(DIAGNOSTIC_BLOCKS * DIAGNOSTIC_STATISTICS);
  cpu_statistics_.resize(DIAGNOSTIC_BLOCKS * DIAGNOSTIC_STATISTICS);

  fid_ = my_fopen("centroid_force_diagnostic.out", "a");
  write_header_();
}

void Centroid_Force_Diagnostic::write_header_()
{
  fprintf(
    fid_,
    "# columns step time_fs[fs] deltaF_rms[eV/Angstrom] "
    "deltaF_mean_abs[eV/Angstrom] deltaF_max[eV/Angstrom] Fbar_rms[eV/Angstrom] "
    "Fc_rms[eV/Angstrom] relative_deltaF_rms[1] P_delta[eV/fs]");
  for (int species = 0; species < DIAGNOSTIC_SPECIES; ++species) {
    fprintf(
      fid_,
      " %s_N_type %s_deltaF_rms[eV/Angstrom] %s_deltaF_mean_abs[eV/Angstrom] "
      "%s_deltaF_max[eV/Angstrom] %s_Fbar_rms[eV/Angstrom] %s_Fc_rms[eV/Angstrom] "
      "%s_relative_deltaF_rms[1] %s_P_delta[eV/fs]",
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species],
      DIAGNOSTIC_SPECIES_NAMES[species]);
  }
  fprintf(fid_, " K_centroid[eV] U_centroid[eV] E_centroid[eV]\n");
  fprintf(fid_, "# velocity_internal_unit Angstrom/natural_time\n");
  fprintf(fid_, "# P_delta_is_power_diagnostic_not_heat_current\n");
  fflush(fid_);
}

void Centroid_Force_Diagnostic::process(
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
      "centroid_force_diagnostic did not receive the immediate centroid force for this sample.\n");
  }

  const GPU_Vector<double>& centroid_force = hac_->centroid_force_per_atom();
  const GPU_Vector<double>& centroid_potential = hac_->centroid_potential_per_atom();
  gpu_reduce_centroid_force_diagnostic<<<DIAGNOSTIC_BLOCKS, DIAGNOSTIC_THREADS>>>(
    number_of_atoms_,
    number_of_types_,
    atom.type.data(),
    species_by_type_.data(),
    atom.force_per_atom.data(),
    centroid_force.data(),
    atom.velocity_per_atom.data(),
    atom.mass.data(),
    centroid_potential.data(),
    gpu_statistics_.data());
  GPU_CHECK_KERNEL
  gpu_statistics_.copy_to_host(cpu_statistics_.data());
  write_row_(sampled_step, global_time);
}

void Centroid_Force_Diagnostic::write_row_(const int step, const double global_time)
{
  const double* global = cpu_statistics_.data();
  const double global_count = global[DIAGNOSTIC_COUNT];
  const double delta_rms = rms_from_statistics(
    global, global_count, DIAGNOSTIC_DELTA_SQUARED);
  const double fbar_rms = rms_from_statistics(
    global, global_count, DIAGNOSTIC_FBAR_SQUARED);
  const double delta_max = std::isfinite(delta_rms)
    ? global[DIAGNOSTIC_DELTA_MAX]
    : std::numeric_limits<double>::quiet_NaN();
  fprintf(
    fid_,
    "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e",
    step,
    global_time * TIME_UNIT_CONVERSION,
    delta_rms,
    global_count > 0.0
      ? global[DIAGNOSTIC_DELTA_ABS] / global_count
      : std::numeric_limits<double>::quiet_NaN(),
    delta_max,
    fbar_rms,
    rms_from_statistics(global, global_count, DIAGNOSTIC_FC_SQUARED),
    relative_rms(delta_rms, fbar_rms),
    global[DIAGNOSTIC_POWER] / TIME_UNIT_CONVERSION);

  for (int species = 0; species < DIAGNOSTIC_SPECIES; ++species) {
    const double* statistics =
      cpu_statistics_.data() + (species + 1) * DIAGNOSTIC_STATISTICS;
    const double count = statistics[DIAGNOSTIC_COUNT];
    const double delta_species_rms = rms_from_statistics(
      statistics, count, DIAGNOSTIC_DELTA_SQUARED);
    const double fbar_species_rms = rms_from_statistics(
      statistics, count, DIAGNOSTIC_FBAR_SQUARED);
    const double delta_species_max = std::isfinite(delta_species_rms)
      ? statistics[DIAGNOSTIC_DELTA_MAX]
      : std::numeric_limits<double>::quiet_NaN();
    fprintf(
      fid_,
      " %.0f %.15e %.15e %.15e %.15e %.15e %.15e",
      count,
      delta_species_rms,
      count > 0.0
        ? statistics[DIAGNOSTIC_DELTA_ABS] / count
        : std::numeric_limits<double>::quiet_NaN(),
      delta_species_max,
      fbar_species_rms,
      rms_from_statistics(statistics, count, DIAGNOSTIC_FC_SQUARED),
      relative_rms(delta_species_rms, fbar_species_rms));
    fprintf(
      fid_,
      " %.15e",
      count > 0.0
        ? statistics[DIAGNOSTIC_POWER] / TIME_UNIT_CONVERSION
        : std::numeric_limits<double>::quiet_NaN());
  }

  const double centroid_kinetic = global[DIAGNOSTIC_KINETIC];
  const double centroid_potential = global[DIAGNOSTIC_POTENTIAL];
  fprintf(
    fid_,
    " %.15e %.15e %.15e\n",
    centroid_kinetic,
    centroid_potential,
    centroid_kinetic + centroid_potential);
  fflush(fid_);
}

void Centroid_Force_Diagnostic::postprocess(
  Atom&,
  Box&,
  Integrate&,
  const int,
  const double,
  const double)
{
  if (fid_ != nullptr) {
    fflush(fid_);
    fclose(fid_);
    fid_ = nullptr;
  }
}
