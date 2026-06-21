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

/*----------------------------------------------------------------------------80
Calculate the heat current autocorrelation (HAC) function.
------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "force/force.cuh"
#include "hac.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <string>
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 5
#define NUM_OF_TYPE_HEAT_COMPONENTS 3
#define FILE_NAME_LENGTH 200
#define DIM 3

// Allocate memory for recording heat current data
void HAC::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (compute) {
    int number_of_frames = number_of_steps / sample_interval;
    heat_all.resize(NUM_OF_HEAT_COMPONENTS * number_of_frames);
    heat_all_by_type_.resize(atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS * number_of_frames);
    atom.heat_per_atom.resize(atom.number_of_atoms * 5);
    if (use_centroid_heat_flux_) {
      centroid_potential_per_atom_.resize(atom.number_of_atoms);
      centroid_force_per_atom_.resize(atom.number_of_atoms * 3);
      centroid_virial_per_atom_.resize(atom.number_of_atoms * 9);
    }
    if (split_qnep_heat_by_type_) {
      heat_all_by_type_electro_.resize(
        atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS * number_of_frames);
      non_electro_potential_per_atom_.resize(atom.number_of_atoms);
      non_electro_force_per_atom_.resize(atom.number_of_atoms * 3);
      non_electro_virial_per_atom_.resize(atom.number_of_atoms * 9);
      electro_virial_per_atom_.resize(atom.number_of_atoms * 9);
      electro_heat_per_atom_.resize(atom.number_of_atoms * NUM_OF_HEAT_COMPONENTS);
      if (use_centroid_heat_flux_) {
        electro_potential_per_atom_.resize(atom.number_of_atoms);
      }
    }
  }
}

// sum up the per-atom heat current to get the total heat current
static __global__ void
gpu_sum_heat(const int N, const int Nd, const int nd, const double* g_heat, double* g_heat_all)
{
  // <<<NUM_OF_HEAT_COMPONENTS, 1024>>>
  const int tid = threadIdx.x;
  const int number_of_patches = (N - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * 1024;
    if (n < N) {
      s_data[tid] += g_heat[n + N * blockIdx.x];
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_heat_all[nd + Nd * blockIdx.x] = s_data[0];
  }
}

static __global__ void gpu_sum_heat_by_type(
  const int N,
  const int Nd,
  const int nd,
  const int number_of_types,
  const int* g_type,
  const double* g_heat,
  double* g_heat_all_by_type)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int type_index = bid / 3;
  const int component = bid % 3;
  const int number_of_patches = (N - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  if (type_index < number_of_types) {
    for (int patch = 0; patch < number_of_patches; ++patch) {
      const int n = tid + patch * 1024;
      if (n < N && g_type[n] == type_index) {
        if (component == 0) {
          s_data[tid] += g_heat[n] + g_heat[n + N];
        } else if (component == 1) {
          s_data[tid] += g_heat[n + N * 2] + g_heat[n + N * 3];
        } else {
          s_data[tid] += g_heat[n + N * 4];
        }
      }
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_heat_all_by_type[nd + Nd * bid] = s_data[0];
  }
}

static __global__ void gpu_subtract_array(
  const int size,
  const double* total,
  const double* part,
  double* difference)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    difference[n] = total[n] - part[n];
  }
}

static __global__ void gpu_compute_centroid_heat(
  const int N,
  const double* mass,
  const double* potential,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  const int include_kinetic,
  double* jx_in,
  double* jx_out,
  double* jy_in,
  double* jy_out,
  double* jz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    const double v_x = vx[n];
    const double v_y = vy[n];
    const double v_z = vz[n];
    double energy = potential[n];
    if (include_kinetic) {
      energy += mass[n] * (v_x * v_x + v_y * v_y + v_z * v_z) * 0.5;
    }
    jx_in[n] = (energy + sxx[n]) * v_x + sxy[n] * v_y;
    jx_out[n] = sxz[n] * v_z;
    jy_in[n] = syx[n] * v_x + (energy + syy[n]) * v_y;
    jy_out[n] = syz[n] * v_z;
    jz[n] = szx[n] * v_x + szy[n] * v_y + (energy + szz[n]) * v_z;
  }
}

static void compute_centroid_heat(
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom,
  const bool include_kinetic = true)
{
  const int N = velocity_per_atom.size() / 3;
  gpu_compute_centroid_heat<<<(N - 1) / 128 + 1, 128>>>(
    N,
    mass.data(),
    potential_per_atom.data(),
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    include_kinetic ? 1 : 0,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2,
    heat_per_atom.data() + N * 3,
    heat_per_atom.data() + N * 4);
  GPU_CHECK_KERNEL
}

// sample heat current data for HAC calculations.
void HAC::process(
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
  if (!compute)
    return;
  if ((step + 1) % sample_interval != 0)
    return;

  const int N = atom.number_of_atoms;
  if (use_centroid_heat_flux_) {
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      centroid_potential_per_atom_,
      centroid_force_per_atom_,
      centroid_virial_per_atom_,
      atom.velocity_per_atom,
      atom.mass);
    compute_centroid_heat(
      atom.mass,
      centroid_potential_per_atom_,
      centroid_virial_per_atom_,
      atom.velocity_per_atom,
      atom.heat_per_atom);
  } else {
    compute_heat(atom.virial_per_atom, atom.velocity_per_atom, atom.heat_per_atom);
  }

  if (split_qnep_heat_by_type_) {
    const bool has_qnep_split = force.compute_qnep_non_electro(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      non_electro_potential_per_atom_,
      non_electro_force_per_atom_,
      non_electro_virial_per_atom_);
    if (has_qnep_split) {
      if (use_centroid_heat_flux_) {
        gpu_subtract_array<<<(N - 1) / 128 + 1, 128>>>(
          N,
          centroid_potential_per_atom_.data(),
          non_electro_potential_per_atom_.data(),
          electro_potential_per_atom_.data());
        gpu_subtract_array<<<((N * 9) - 1) / 128 + 1, 128>>>(
          N * 9,
          centroid_virial_per_atom_.data(),
          non_electro_virial_per_atom_.data(),
          electro_virial_per_atom_.data());
        GPU_CHECK_KERNEL
        compute_centroid_heat(
          atom.mass,
          electro_potential_per_atom_,
          electro_virial_per_atom_,
          atom.velocity_per_atom,
          electro_heat_per_atom_,
          false);
      } else {
        gpu_subtract_array<<<((N * 9) - 1) / 128 + 1, 128>>>(
          N * 9,
          atom.virial_per_atom.data(),
          non_electro_virial_per_atom_.data(),
          electro_virial_per_atom_.data());
        GPU_CHECK_KERNEL
        compute_heat(electro_virial_per_atom_, atom.velocity_per_atom, electro_heat_per_atom_);
      }
    } else {
      CHECK(gpuMemset(
        electro_heat_per_atom_.data(), 0, sizeof(double) * N * NUM_OF_HEAT_COMPONENTS));
    }
  }

  int nd = (step + 1) / sample_interval - 1;
  int Nd = number_of_steps / sample_interval;
  gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>(N, Nd, nd, atom.heat_per_atom.data(), heat_all.data());
  gpu_sum_heat_by_type<<<atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS, 1024>>>(
    N,
    Nd,
    nd,
    atom.cpu_type_size.size(),
    atom.type.data(),
    atom.heat_per_atom.data(),
    heat_all_by_type_.data());
  if (split_qnep_heat_by_type_) {
    gpu_sum_heat_by_type<<<atom.cpu_type_size.size() * NUM_OF_TYPE_HEAT_COMPONENTS, 1024>>>(
      N,
      Nd,
      nd,
      atom.cpu_type_size.size(),
      atom.type.data(),
      electro_heat_per_atom_.data(),
      heat_all_by_type_electro_.data());
  }
  GPU_CHECK_KERNEL
}

// Calculate the Heat current Auto-Correlation function (HAC)
static __global__ void gpu_find_hac(const int Nc, const int Nd, const double* g_heat, double* g_hac)
{
  //<<<Nc, 128>>>

  __shared__ double s_hac_xi[128];
  __shared__ double s_hac_xo[128];
  __shared__ double s_hac_yi[128];
  __shared__ double s_hac_yo[128];
  __shared__ double s_hac_z[128];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (Nd - 1) / 128 + 1;
  int number_of_data = Nd - bid;

  s_hac_xi[tid] = 0.0;
  s_hac_xo[tid] = 0.0;
  s_hac_yi[tid] = 0.0;
  s_hac_yo[tid] = 0.0;
  s_hac_z[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int index = tid + patch * 128;
    if (index + bid < Nd) {
      s_hac_xi[tid] += g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 0] +
                       g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 1];
      s_hac_xo[tid] += g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 1] +
                       g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 0];
      s_hac_yi[tid] += g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 2] +
                       g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 3];
      s_hac_yo[tid] += g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 3] +
                       g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 2];
      s_hac_z[tid] += g_heat[index + Nd * 4] * g_heat[index + bid + Nd * 4];
    }
  }
  __syncthreads();


  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_hac_xi[tid] += s_hac_xi[tid + offset];
      s_hac_xo[tid] += s_hac_xo[tid + offset];
      s_hac_yi[tid] += s_hac_yi[tid + offset];
      s_hac_yo[tid] += s_hac_yo[tid + offset];
      s_hac_z[tid] += s_hac_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_hac[bid + Nc * 0] = s_hac_xi[0] / number_of_data;
    g_hac[bid + Nc * 1] = s_hac_xo[0] / number_of_data;
    g_hac[bid + Nc * 2] = s_hac_yi[0] / number_of_data;
    g_hac[bid + Nc * 3] = s_hac_yo[0] / number_of_data;
    g_hac[bid + Nc * 4] = s_hac_z[0] / number_of_data;
  }
}

// Calculate the Running Thermal Conductivity (RTC) from the HAC
static void find_rtc(const int Nc, const double factor, const double* hac, double* rtc)
{
  for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
    for (int nc = 1; nc < Nc; nc++) {
      const int index = Nc * k + nc;
      rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index]) * factor;
    }
  }
}

// Calculate HAC (heat currant auto-correlation function)
// and RTC (running thermal conductivity)
void HAC::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute)
    return;
  print_line_1();
  printf("Start to calculate HAC and related quantities.\n");

  const int Nd = number_of_steps / sample_interval;
  const double dt = time_step * sample_interval;
  const double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

  std::vector<double> heat_current_cpu(Nd * NUM_OF_HEAT_COMPONENTS);
  heat_all.copy_to_host(heat_current_cpu.data());

  const char* heat_current_file_name =
    use_centroid_heat_flux_ ? "heat_current_centroid.out" : "heat_current.out";
  FILE* fid_heat_current = fopen(heat_current_file_name, "a");
  fprintf(fid_heat_current, "# time_ps Jx Jy Jz\n");
  for (int nd = 0; nd < Nd; ++nd) {
    const double jx = heat_current_cpu[nd + Nd * 0] + heat_current_cpu[nd + Nd * 1];
    const double jy = heat_current_cpu[nd + Nd * 2] + heat_current_cpu[nd + Nd * 3];
    const double jz = heat_current_cpu[nd + Nd * 4];
    fprintf(fid_heat_current, "%25.15e%25.15e%25.15e%25.15e\n", (nd + 1) * dt_in_ps, jx, jy, jz);
  }
  fflush(fid_heat_current);
  fclose(fid_heat_current);

  const int number_of_types = atom.cpu_type_size.size();
  std::vector<double> heat_current_by_type_cpu(Nd * number_of_types * 3);
  heat_all_by_type_.copy_to_host(heat_current_by_type_cpu.data());
  std::vector<std::string> type_symbols(number_of_types);
  std::vector<int> type_symbol_found(number_of_types, 0);
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    const int type_index = atom.cpu_type[n];
    if (!type_symbol_found[type_index]) {
      type_symbols[type_index] = atom.cpu_atom_symbol[n];
      type_symbol_found[type_index] = 1;
    }
  }
  for (int type_index = 0; type_index < number_of_types; ++type_index) {
    if (!type_symbol_found[type_index]) {
      type_symbols[type_index] = std::string("type") + std::to_string(type_index);
    }
  }

  const char* type_resolved_file_name =
    use_centroid_heat_flux_ ? "heat_current_type_resolved_centroid.out" : "heat_current_type_resolved.out";
  FILE* fid_type_resolved = fopen(type_resolved_file_name, "a");
  fprintf(fid_type_resolved, "# time_ps total_Jx total_Jy total_Jz");
  for (int type_index = 0; type_index < number_of_types; ++type_index) {
    fprintf(
      fid_type_resolved,
      " type%d_%s_Jx type%d_%s_Jy type%d_%s_Jz",
      type_index,
      type_symbols[type_index].c_str(),
      type_index,
      type_symbols[type_index].c_str(),
      type_index,
      type_symbols[type_index].c_str());
  }
  fprintf(fid_type_resolved, "\n");
  for (int nd = 0; nd < Nd; ++nd) {
    const double jx_total = heat_current_cpu[nd + Nd * 0] + heat_current_cpu[nd + Nd * 1];
    const double jy_total = heat_current_cpu[nd + Nd * 2] + heat_current_cpu[nd + Nd * 3];
    const double jz_total = heat_current_cpu[nd + Nd * 4];
    fprintf(
      fid_type_resolved,
      "%25.15e%25.15e%25.15e%25.15e",
      (nd + 1) * dt_in_ps,
      jx_total,
      jy_total,
      jz_total);
    for (int type_index = 0; type_index < number_of_types; ++type_index) {
      const int column_offset = type_index * 3;
      const double jx_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 0)];
      const double jy_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 1)];
      const double jz_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 2)];
      fprintf(fid_type_resolved, "%25.15e%25.15e%25.15e", jx_type, jy_type, jz_type);
    }
    fprintf(fid_type_resolved, "\n");
  }
  fflush(fid_type_resolved);
  fclose(fid_type_resolved);

  if (split_qnep_heat_by_type_) {
    std::vector<double> heat_current_by_type_electro_cpu(Nd * number_of_types * 3);
    heat_all_by_type_electro_.copy_to_host(heat_current_by_type_electro_cpu.data());
    const char* type_resolved_split_file_name = use_centroid_heat_flux_
      ? "heat_current_type_resolved_qnep_split_centroid.out"
      : "heat_current_type_resolved_qnep_split.out";
    FILE* fid_type_resolved_split = fopen(type_resolved_split_file_name, "a");
    fprintf(
      fid_type_resolved_split,
      "# time_ps total_Jx total_Jy total_Jz electro_Jx electro_Jy electro_Jz non_electro_Jx non_electro_Jy non_electro_Jz");
    for (int type_index = 0; type_index < number_of_types; ++type_index) {
      fprintf(
        fid_type_resolved_split,
        " type%d_%s_electro_Jx type%d_%s_electro_Jy type%d_%s_electro_Jz"
        " type%d_%s_non_electro_Jx type%d_%s_non_electro_Jy type%d_%s_non_electro_Jz",
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str(),
        type_index,
        type_symbols[type_index].c_str());
    }
    fprintf(fid_type_resolved_split, "\n");
    for (int nd = 0; nd < Nd; ++nd) {
      const double jx_total = heat_current_cpu[nd + Nd * 0] + heat_current_cpu[nd + Nd * 1];
      const double jy_total = heat_current_cpu[nd + Nd * 2] + heat_current_cpu[nd + Nd * 3];
      const double jz_total = heat_current_cpu[nd + Nd * 4];
      double jx_electro_total = 0.0;
      double jy_electro_total = 0.0;
      double jz_electro_total = 0.0;
      for (int type_index = 0; type_index < number_of_types; ++type_index) {
        const int column_offset = type_index * 3;
        jx_electro_total += heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 0)];
        jy_electro_total += heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 1)];
        jz_electro_total += heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 2)];
      }
      fprintf(
        fid_type_resolved_split,
        "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e",
        (nd + 1) * dt_in_ps,
        jx_total,
        jy_total,
        jz_total,
        jx_electro_total,
        jy_electro_total,
        jz_electro_total,
        jx_total - jx_electro_total,
        jy_total - jy_electro_total,
        jz_total - jz_electro_total);
      for (int type_index = 0; type_index < number_of_types; ++type_index) {
        const int column_offset = type_index * 3;
        const double jx_electro = heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 0)];
        const double jy_electro = heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 1)];
        const double jz_electro = heat_current_by_type_electro_cpu[nd + Nd * (column_offset + 2)];
        const double jx_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 0)];
        const double jy_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 1)];
        const double jz_type = heat_current_by_type_cpu[nd + Nd * (column_offset + 2)];
        fprintf(
          fid_type_resolved_split,
          "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e",
          jx_electro,
          jy_electro,
          jz_electro,
          jx_type - jx_electro,
          jy_type - jy_electro,
          jz_type - jz_electro);
      }
      fprintf(fid_type_resolved_split, "\n");
    }
    fflush(fid_type_resolved_split);
    fclose(fid_type_resolved_split);
  }

  // major data
  std::vector<double> rtc(Nc * NUM_OF_HEAT_COMPONENTS, 0.0);
  GPU_Vector<double> hac_gpu(Nc * NUM_OF_HEAT_COMPONENTS);
  std::vector<double> hac_cpu(Nc * NUM_OF_HEAT_COMPONENTS);

  // Here, the block size is fixed to 128, which is a good choice
  gpu_find_hac<<<Nc, 128>>>(Nc, Nd, heat_all.data(), hac_gpu.data());
  GPU_CHECK_KERNEL

  hac_gpu.copy_to_host(hac_cpu.data());

  double factor = dt * 0.5 / (K_B * temperature * temperature * box.get_volume());
  factor *= KAPPA_UNIT_CONVERSION;

  find_rtc(Nc, factor, hac_cpu.data(), rtc.data());
  const char* output_file_name = use_centroid_heat_flux_ ? "hac_centroid.out" : "hac.out";
  FILE* fid = fopen(output_file_name, "a");
  const int number_of_output_data = Nc / output_interval;
  for (int nd = 0; nd < number_of_output_data; nd++) {
    const int nc = nd * output_interval;
    double hac_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    double rtc_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
      for (int m = 0; m < output_interval; m++) {
        const int count = Nc * k + nc + m;
        hac_ave[k] += hac_cpu[count];
        rtc_ave[k] += rtc[count];
      }
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      hac_ave[m] /= output_interval;
      rtc_ave[m] /= output_interval;
    }
    fprintf(fid, "%25.15e", (nc + output_interval * 0.5) * dt_in_ps);
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", hac_ave[m]);
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", rtc_ave[m]);
    }
    fprintf(fid, "\n");
  }
  fflush(fid);
  fclose(fid);

  printf("HAC and related quantities are calculated.\n");
  print_line_2();

  compute = 0;
}

void HAC::parse(const char** param, int num_param)
{
  compute = 1;

  printf("Compute HAC.\n");

  if (!(num_param == 4 || num_param == 5 || num_param == 6)) {
    PRINT_INPUT_ERROR("compute_hac should have 3, 4, or 5 parameters.\n");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for HAC should be an integer number.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for HAC should be an integer number.\n");
  }
  printf("    Nc is %d\n", Nc);

  if (!is_valid_int(param[3], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HAC should be an integer number.\n");
  }
  printf("    output_interval is %d\n", output_interval);
  if (num_param >= 5) {
    if (!is_valid_int(param[4], &use_centroid_heat_flux_)) {
      PRINT_INPUT_ERROR("centroid heat flux flag for HAC should be an integer.\n");
    }
    if (use_centroid_heat_flux_ != 0) {
      printf("    use the full classical heat-flux operator on the current centroid structure.\n");
    }
  }
  if (num_param == 6) {
    if (!is_valid_int(param[5], &split_qnep_heat_by_type_)) {
      PRINT_INPUT_ERROR("qNEP electrostatic split flag for HAC should be an integer.\n");
    }
    if (split_qnep_heat_by_type_ != 0) {
      printf("    output type-resolved electrostatic and non-electrostatic heat currents for qNEP.\n");
    }
  }
}

HAC::HAC(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "compute_hac";
}
