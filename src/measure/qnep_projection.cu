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
Classical qNEP projection diagnostic for heat-current derivation.
------------------------------------------------------------------------------*/

#include "qnep_projection.cuh"
#include "compute_heat.cuh"
#include "force/force.cuh"
#include "force/nep_charge.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

namespace {

constexpr int REDUCE_THREADS = 1024;
constexpr int PAIR_THREADS = 128;
constexpr int NUM_OUTPUTS = 10;
constexpr int NUM_CHANNEL_COMPONENTS = 6;
constexpr int NUM_HEAT_COMPONENTS = 5;

void __global__ gpu_reduce_means(
  const int N, const float* g_D, const float* g_s, double* g_means)
{
  const int tid = threadIdx.x;
  __shared__ double s_D[REDUCE_THREADS];
  __shared__ double s_s[REDUCE_THREADS];

  double sum_D = 0.0;
  double sum_s = 0.0;
  for (int n = tid; n < N; n += REDUCE_THREADS) {
    sum_D += static_cast<double>(g_D[n]);
    sum_s += static_cast<double>(g_s[n]);
  }
  s_D[tid] = sum_D;
  s_s[tid] = sum_s;
  __syncthreads();

  for (int offset = REDUCE_THREADS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_D[tid] += s_D[tid + offset];
      s_s[tid] += s_s[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_means[0] = s_D[0] / static_cast<double>(N);
    g_means[1] = s_s[0] / static_cast<double>(N);
  }
}

// ponytail: keep all pairs; optimize blocking/data reuse only if sampling cost matters.
void __global__ gpu_compute_pair_sums(
  const int N,
  const Box box,
  const double* g_position,
  const double* g_unwrapped_position,
  const float* g_D,
  const float* g_s,
  const double* g_means,
  double* g_partial)
{
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  __shared__ double s_data[NUM_OUTPUTS][PAIR_THREADS];

  double JA[3] = {0.0, 0.0, 0.0};
  double JB[3] = {0.0, 0.0, 0.0};
  double JD[3] = {0.0, 0.0, 0.0};
  double sum_c = 0.0;

  if (a < N) {
    const double inv_N = 1.0 / static_cast<double>(N);
    const double D_a = static_cast<double>(g_D[a]);
    const double s_a = static_cast<double>(g_s[a]);
    const double c_a = g_means[0] * s_a - D_a * g_means[1];
    JB[0] = g_unwrapped_position[a] * c_a;
    JB[1] = g_unwrapped_position[a + N] * c_a;
    JB[2] = g_unwrapped_position[a + 2 * N] * c_a;
    sum_c = c_a;

    for (int b = a + 1; b < N; ++b) {
      double dx = g_position[b] - g_position[a];
      double dy = g_position[b + N] - g_position[a + N];
      double dz = g_position[b + 2 * N] - g_position[a + 2 * N];
      apply_mic(box, dx, dy, dz);

      const double D_b = static_cast<double>(g_D[b]);
      const double s_b = static_cast<double>(g_s[b]);
      const double c_b = g_means[0] * s_b - D_b * g_means[1];
      const double pair_c = (c_b - c_a) * inv_N;
      const double pair_D = (D_a * s_b - D_b * s_a) * inv_N;
      JA[0] += dx * pair_c;
      JA[1] += dy * pair_c;
      JA[2] += dz * pair_c;
      JD[0] += dx * pair_D;
      JD[1] += dy * pair_D;
      JD[2] += dz * pair_D;
    }
  }

  s_data[0][tid] = JA[0];
  s_data[1][tid] = JA[1];
  s_data[2][tid] = JA[2];
  s_data[3][tid] = JB[0];
  s_data[4][tid] = JB[1];
  s_data[5][tid] = JB[2];
  s_data[6][tid] = JD[0];
  s_data[7][tid] = JD[1];
  s_data[8][tid] = JD[2];
  s_data[9][tid] = sum_c;
  __syncthreads();

  for (int offset = PAIR_THREADS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int output = 0; output < NUM_OUTPUTS; ++output) {
        s_data[output][tid] += s_data[output][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int output = 0; output < NUM_OUTPUTS; ++output) {
      g_partial[blockIdx.x * NUM_OUTPUTS + output] = s_data[output][0];
    }
  }
}

void __global__ gpu_reduce_pair_sums(
  const int number_of_blocks, const double* g_partial, double* g_total)
{
  const int output = threadIdx.x;
  if (output < NUM_OUTPUTS) {
    double sum = 0.0;
    for (int block = 0; block < number_of_blocks; ++block) {
      sum += g_partial[block * NUM_OUTPUTS + output];
    }
    g_total[output] = sum;
  }
}

void __global__ gpu_sum_components(
  const int N, const int number_of_components, const double* g_values, double* g_total)
{
  const int component = blockIdx.x;
  const int tid = threadIdx.x;
  if (component >= number_of_components) return;
  __shared__ double s_data[REDUCE_THREADS];

  double sum = 0.0;
  for (int n = tid; n < N; n += REDUCE_THREADS)
    sum += g_values[n + component * N];
  s_data[tid] = sum;
  __syncthreads();

  for (int offset = REDUCE_THREADS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0) g_total[component] = s_data[0];
}

void __global__ gpu_sum_conventional_current(
  const int N,
  const double* mass,
  const double* potential,
  const double* velocity,
  double* total)
{
  const int tid = threadIdx.x;
  __shared__ double sx[REDUCE_THREADS];
  __shared__ double sy[REDUCE_THREADS];
  __shared__ double sz[REDUCE_THREADS];

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  for (int n = tid; n < N; n += REDUCE_THREADS) {
    const double vx = velocity[n];
    const double vy = velocity[n + N];
    const double vz = velocity[n + 2 * N];
    const double energy = 0.5 * mass[n] * (vx * vx + vy * vy + vz * vz) + potential[n];
    sum_x += energy * vx;
    sum_y += energy * vy;
    sum_z += energy * vz;
  }
  sx[tid] = sum_x;
  sy[tid] = sum_y;
  sz[tid] = sum_z;
  __syncthreads();

  for (int offset = REDUCE_THREADS >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sx[tid] += sx[tid + offset];
      sy[tid] += sy[tid + offset];
      sz[tid] += sz[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    total[0] = sx[0];
    total[1] = sy[0];
    total[2] = sz[0];
  }
}

} // namespace

void QNEP_Projection::pre_run(
  const int,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (integrate.type >= 31 && integrate.type <= 33) {
    PRINT_INPUT_ERROR("compute_qnep_projection currently supports classical MD only.\n");
  }
  if (
    (integrate.type >= 11 && integrate.type <= 20) || integrate.type == -1 ||
    integrate.type == -3 || integrate.type == -4 || integrate.type == -5 ||
    integrate.type == -12) {
    PRINT_INPUT_ERROR(
      "compute_qnep_projection requires a fixed simulation cell; NPT, MSST, MTTK, "
      "piston, NPHug, and NPT-QTB integrators are unsupported.\n");
  }
  if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1) {
    PRINT_INPUT_ERROR("compute_qnep_projection requires three-dimensional periodic boundary conditions.\n");
  }
  if (force.potentials.size() != 1) {
    PRINT_INPUT_ERROR("compute_qnep_projection requires exactly one qNEP potential.\n");
  }

  qnep_ = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
  if (qnep_ == nullptr) {
    PRINT_INPUT_ERROR("compute_qnep_projection requires an NEP-Charge potential.\n");
  }

  for (int i = 0; i < 9; ++i)
    initial_cell_[i] = box.cpu_h[i];

  qnep_->enable_charge_diagnostics();
  if (!complete_current_) qnep_->enable_delta_j_q_k_diagnostics();
  if (complete_current_ || !g1_channel_) atom.enable_unwrapped_position();

  const int N = atom.number_of_atoms;
  if (complete_current_) {
    if (time_step != 0.0) {
      PRINT_INPUT_ERROR(
        "compute_qnep_current_diag currently requires time_step 0 so charge/geometry remain in one frame.\n");
    }
    if (!qnep_->uses_pppm()) {
      PRINT_INPUT_ERROR("compute_qnep_current_diag requires kspace_method pppm.\n");
    }
    box.set_is_orthogonal();
    if (!box.is_orthogonal) {
      PRINT_INPUT_ERROR(
        "compute_qnep_current_diag candidate_v1 currently requires an orthogonal cell.\n");
    }

    fid_complete_current_ = my_fopen("qnep_complete_current_diag.csv", "a");
    fseek(fid_complete_current_, 0, SEEK_END);
    if (ftell(fid_complete_current_) == 0) {
      fprintf(fid_complete_current_, "# units eV*Angstrom/fs\n");
      fprintf(
        fid_complete_current_,
        "step,time_fs,J_conv_x,J_conv_y,J_conv_z,J_nep_x,J_nep_y,J_nep_z,"
        "J_elec_fixed_x,J_elec_fixed_y,J_elec_fixed_z,J_dyn_local_x,J_dyn_local_y,J_dyn_local_z,"
        "J_base_x,J_base_y,J_base_z,DeltaJ_q_pppm_x,DeltaJ_q_pppm_y,DeltaJ_q_pppm_z,"
        "J_proj_A_x,J_proj_A_y,J_proj_A_z,J_proj_B_x,J_proj_B_y,J_proj_B_z,"
        "J_proj_D_x,J_proj_D_y,J_proj_D_z,J_cand_A_x,J_cand_A_y,J_cand_A_z,"
        "J_cand_B_x,J_cand_B_y,J_cand_B_z,J_cand_D_x,J_cand_D_y,J_cand_D_z,"
        "virial_decomposition_error\n");
    }

    number_of_pair_blocks_ = (N + PAIR_THREADS - 1) / PAIR_THREADS;
    gpu_means_.resize(2);
    gpu_partial_.resize(number_of_pair_blocks_ * NUM_OUTPUTS);
    gpu_total_.resize(NUM_OUTPUTS);
    gpu_current_total_.resize(3);
    cpu_total_.resize(NUM_OUTPUTS);
    gpu_virial_nep_.resize(static_cast<size_t>(N) * 9);
    gpu_virial_electrostatic_fixed_.resize(static_cast<size_t>(N) * 9);
    gpu_virial_dynamic_charge_.resize(static_cast<size_t>(N) * 9);
    gpu_virial_heat_per_atom_.resize(static_cast<size_t>(N) * NUM_HEAT_COMPONENTS);
    gpu_virial_heat_total_.resize(NUM_HEAT_COMPONENTS);
    cpu_virial_heat_total_.resize(NUM_HEAT_COMPONENTS);
    return;
  }

  fid_delta_j_q_k_ = my_fopen("delta_j_q_k.out", "a");
  fprintf(fid_delta_j_q_k_, "# compute_qnep_projection %d\n", sample_interval_);
  fprintf(fid_delta_j_q_k_, "# observable delta_j_q_k\n");
  fprintf(fid_delta_j_q_k_, "# format_version 1\n");
  fprintf(fid_delta_j_q_k_, "# num_atoms %d\n", N);
  fprintf(
    fid_delta_j_q_k_,
    "# cell %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
    box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
    box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
  fprintf(fid_delta_j_q_k_, "# phase exp(+ikr)\n");
  fprintf(fid_delta_j_q_k_, "# charge projected\n");
  fprintf(fid_delta_j_q_k_, "# charge_rate projected\n");
  fprintf(fid_delta_j_q_k_, "# k0 excluded\n");
  fprintf(fid_delta_j_q_k_, "# kernel continuous_ewald_reference\n");
  fprintf(fid_delta_j_q_k_, "# ewald_alpha %.17g 1/Angstrom\n", qnep_->get_ewald_alpha());
  fprintf(fid_delta_j_q_k_, "# reciprocal_k_cutoff k_squared < (2*pi*alpha)^2\n");
  fprintf(
    fid_delta_j_q_k_,
    "# reciprocal_k_convention n1>0 or n1=0,n2>0 or n1=n2=0,n3>0\n");
  fprintf(fid_delta_j_q_k_, "# units sum_q=e sum_qdot=e/fs delta_j_q_k=eV Angstrom/fs\n");
  fprintf(
    fid_delta_j_q_k_,
    "# columns step time_fs sum_q sum_qdot nk delta_j_q_k_x delta_j_q_k_y delta_j_q_k_z\n");
  gpu_delta_j_q_k_.resize(3);
  cpu_delta_j_q_k_.resize(3);

  if (g1_channel_) {
    fid_channel_ = my_fopen("charge_heat_diagnostic.out", "a");
    fprintf(fid_channel_, "# compute_qnep_projection %d g1_channel\n", sample_interval_);
    fprintf(fid_channel_, "# format_version 1\n");
    fprintf(fid_channel_, "# num_atoms %d\n", N);
    fprintf(
      fid_channel_,
      "# cell %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
      box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
      box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
      box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
    fprintf(
      fid_channel_,
      "# nominal_dt_output %.17g fs\n",
      time_step * sample_interval_ * TIME_UNIT_CONVERSION);
    fprintf(fid_channel_, "# units J_channel=J_virial=eV Angstrom/fs\n");
    fprintf(fid_channel_, "# definitions r12=r_image-r_center; ell_G=-r12; D=D_projected\n");
    fprintf(fid_channel_, "# J_channel=-sum_b,a,n D_b*r12_ban*(g_ban dot v_a)\n");
    fprintf(fid_channel_, "# J_channel_total=J_channel_radial+J_channel_angular\n");
    fprintf(
      fid_channel_,
      "# columns step time_fs J_channel_radial_x J_channel_radial_y J_channel_radial_z "
      "J_channel_angular_x J_channel_angular_y J_channel_angular_z "
      "J_channel_total_x J_channel_total_y J_channel_total_z "
      "J_charge_virial_x J_charge_virial_y J_charge_virial_z\n");

    gpu_channel_per_atom_.resize(static_cast<size_t>(N) * NUM_CHANNEL_COMPONENTS);
    gpu_channel_total_.resize(NUM_CHANNEL_COMPONENTS);
    gpu_virial_nep_.resize(static_cast<size_t>(N) * 9);
    gpu_virial_electrostatic_fixed_.resize(static_cast<size_t>(N) * 9);
    gpu_virial_dynamic_charge_.resize(static_cast<size_t>(N) * 9);
    gpu_virial_heat_per_atom_.resize(static_cast<size_t>(N) * NUM_HEAT_COMPONENTS);
    gpu_virial_heat_total_.resize(NUM_HEAT_COMPONENTS);
    cpu_channel_total_.resize(NUM_CHANNEL_COMPONENTS);
    cpu_virial_heat_total_.resize(NUM_HEAT_COMPONENTS);
  } else {
    fid_ = my_fopen("qnep_projection.out", "a");
    fprintf(fid_, "# compute_qnep_projection %d\n", sample_interval_);
    fprintf(fid_, "# format_version 1\n");
    fprintf(fid_, "# num_atoms %d\n", N);
    fprintf(
      fid_,
      "# cell %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
      box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
      box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
      box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
    fprintf(fid_, "# nominal_dt_output %.17g fs\n", time_step * sample_interval_ * TIME_UNIT_CONVERSION);
    fprintf(fid_, "# units JA JB JD=eV Angstrom/fs; sum_c=eV/fs\n");
    fprintf(fid_, "# definitions D_i=D_raw_i, s_i=Qdot_raw_i, c_i=mean(D)*s_i-D_i*mean(s)\n");
    fprintf(fid_, "# JA=sum_{a<b} d_ab*(c_b-c_a)/N; JB=sum_a R_unwrapped_a*c_a\n");
    fprintf(fid_, "# JD=sum_{a<b} d_ab*(D_a*s_b-D_b*s_a)/N\n");
    fprintf(fid_, "# columns step time_fs JA_x JA_y JA_z JB_x JB_y JB_z JD_x JD_y JD_z sum_c\n");

    fid_projection_current_diag_ = my_fopen("projection_current_diag.csv", "a");
    fseek(fid_projection_current_diag_, 0, SEEK_END);
    if (ftell(fid_projection_current_diag_) == 0) {
      fprintf(fid_projection_current_diag_, "# units eV*Angstrom/fs\n");
      fprintf(
        fid_projection_current_diag_,
        "step,time_fs,J_proj_A_x,J_proj_A_y,J_proj_A_z,J_proj_B_x,J_proj_B_y,J_proj_B_z,"
        "J_proj_D_x,J_proj_D_y,J_proj_D_z\n");
    }

    number_of_pair_blocks_ = (N + PAIR_THREADS - 1) / PAIR_THREADS;
    gpu_means_.resize(2);
    gpu_partial_.resize(number_of_pair_blocks_ * NUM_OUTPUTS);
    gpu_total_.resize(NUM_OUTPUTS);
    cpu_total_.resize(NUM_OUTPUTS);
  }
}

void QNEP_Projection::check_fixed_cell(const Box& box) const
{
  for (int i = 0; i < 9; ++i) {
    if (box.cpu_h[i] != initial_cell_[i]) {
      PRINT_INPUT_ERROR(
        "compute_qnep_projection requires a fixed simulation cell; the cell changed during the run.\n");
    }
  }
}

void QNEP_Projection::compute_projection_currents(
  const int N,
  const Box& box,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& unwrapped_position,
  const GPU_Vector<float>& D,
  const GPU_Vector<float>& s)
{
  gpu_reduce_means<<<1, REDUCE_THREADS>>>(N, D.data(), s.data(), gpu_means_.data());
  GPU_CHECK_KERNEL
  gpu_compute_pair_sums<<<number_of_pair_blocks_, PAIR_THREADS>>>(
    N,
    box,
    position.data(),
    unwrapped_position.data(),
    D.data(),
    s.data(),
    gpu_means_.data(),
    gpu_partial_.data());
  GPU_CHECK_KERNEL
  gpu_reduce_pair_sums<<<1, NUM_OUTPUTS>>>(
    number_of_pair_blocks_, gpu_partial_.data(), gpu_total_.data());
  GPU_CHECK_KERNEL
  gpu_total_.copy_to_host(cpu_total_.data());
}

void QNEP_Projection::sum_virial_current(
  const GPU_Vector<double>& virial,
  const GPU_Vector<double>& velocity,
  double current[3])
{
  const int N = velocity.size() / 3;
  compute_heat(virial, velocity, gpu_virial_heat_per_atom_);
  gpu_sum_components<<<NUM_HEAT_COMPONENTS, REDUCE_THREADS>>>(
    N,
    NUM_HEAT_COMPONENTS,
    gpu_virial_heat_per_atom_.data(),
    gpu_virial_heat_total_.data());
  GPU_CHECK_KERNEL
  gpu_virial_heat_total_.copy_to_host(cpu_virial_heat_total_.data());
  current[0] = cpu_virial_heat_total_[0] + cpu_virial_heat_total_[1];
  current[1] = cpu_virial_heat_total_[2] + cpu_virial_heat_total_[3];
  current[2] = cpu_virial_heat_total_[4];
}

void QNEP_Projection::write_complete_current(
  const int step,
  const double global_time,
  Box& box,
  Atom& atom)
{
  const int N = atom.number_of_atoms;
  double delta_j_q_pppm[3] = {0.0, 0.0, 0.0};
  qnep_->compute_charge_rate(box, atom.type, atom.position_per_atom, atom.velocity_per_atom);
  qnep_->diagnose_dynamic_charge(
    N,
    0,
    N,
    0,
    step + 1,
    global_time * TIME_UNIT_CONVERSION,
    box,
    atom.position_per_atom,
    false,
    delta_j_q_pppm);

  // Capture the main-force qNEP projection buffers before diagnostic virial
  // recomputation can refresh shared qNEP work arrays.
  const GPU_Vector<float>& D = qnep_->get_raw_D_reference();
  const GPU_Vector<float>& s = qnep_->get_raw_charge_rate_reference();
  compute_projection_currents(
    N,
    box,
    atom.position_per_atom,
    atom.unwrapped_position,
    D,
    s);

  qnep_->compute_virial_components(
    box,
    atom.type,
    atom.position_per_atom,
    atom.virial_per_atom,
    true,
    true,
    true,
    gpu_virial_nep_,
    gpu_virial_electrostatic_fixed_,
    gpu_virial_dynamic_charge_);

  double j_conv[3] = {0.0, 0.0, 0.0};
  gpu_sum_conventional_current<<<1, REDUCE_THREADS>>>(
    N,
    atom.mass.data(),
    atom.potential_per_atom.data(),
    atom.velocity_per_atom.data(),
    gpu_current_total_.data());
  GPU_CHECK_KERNEL
  gpu_current_total_.copy_to_host(j_conv);

  double j_nep[3] = {0.0, 0.0, 0.0};
  double j_elec_fixed[3] = {0.0, 0.0, 0.0};
  double j_dyn_local[3] = {0.0, 0.0, 0.0};
  double j_virial_total[3] = {0.0, 0.0, 0.0};
  sum_virial_current(gpu_virial_nep_, atom.velocity_per_atom, j_nep);
  sum_virial_current(
    gpu_virial_electrostatic_fixed_, atom.velocity_per_atom, j_elec_fixed);
  sum_virial_current(gpu_virial_dynamic_charge_, atom.velocity_per_atom, j_dyn_local);
  sum_virial_current(atom.virial_per_atom, atom.velocity_per_atom, j_virial_total);

  const double inv_time_conversion = 1.0 / TIME_UNIT_CONVERSION;
  double j_base[3] = {0.0, 0.0, 0.0};
  double j_projection[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  double j_candidate[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (int d = 0; d < 3; ++d) {
    j_base[d] = (j_conv[d] + j_nep[d] + j_elec_fixed[d] + j_dyn_local[d]) * inv_time_conversion;
    j_projection[0][d] = cpu_total_[d] * inv_time_conversion;
    j_projection[1][d] = cpu_total_[d + 3] * inv_time_conversion;
    j_projection[2][d] = cpu_total_[d + 6] * inv_time_conversion;
    for (int x = 0; x < 3; ++x)
      j_candidate[x][d] = j_base[d] + delta_j_q_pppm[d] + j_projection[x][d];
  }

  double virial_decomposition_error = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double error = std::fabs(
      (j_virial_total[d] - j_nep[d] - j_elec_fixed[d] - j_dyn_local[d]) * inv_time_conversion);
    if (error > virial_decomposition_error) virial_decomposition_error = error;
  }

  fprintf(
    fid_complete_current_,
    "%d,%.17g",
    step + 1,
    global_time * TIME_UNIT_CONVERSION);
  for (int d = 0; d < 3; ++d) fprintf(fid_complete_current_, ",%.17g", j_conv[d] * inv_time_conversion);
  for (int d = 0; d < 3; ++d) fprintf(fid_complete_current_, ",%.17g", j_nep[d] * inv_time_conversion);
  for (int d = 0; d < 3; ++d)
    fprintf(fid_complete_current_, ",%.17g", j_elec_fixed[d] * inv_time_conversion);
  for (int d = 0; d < 3; ++d)
    fprintf(fid_complete_current_, ",%.17g", j_dyn_local[d] * inv_time_conversion);
  for (int d = 0; d < 3; ++d) fprintf(fid_complete_current_, ",%.17g", j_base[d]);
  for (int d = 0; d < 3; ++d) fprintf(fid_complete_current_, ",%.17g", delta_j_q_pppm[d]);
  for (int x = 0; x < 3; ++x)
    for (int d = 0; d < 3; ++d) fprintf(fid_complete_current_, ",%.17g", j_projection[x][d]);
  for (int x = 0; x < 3; ++x)
    for (int d = 0; d < 3; ++d) fprintf(fid_complete_current_, ",%.17g", j_candidate[x][d]);
  fprintf(fid_complete_current_, ",%.17g\n", virial_decomposition_error);
}

void QNEP_Projection::pre_force(
  const int step,
  const double,
  Integrate&,
  std::vector<Group>&,
  Atom&,
  Box& box,
  Force&)
{
  check_fixed_cell(box);
  if ((step + 1) % sample_interval_ == 0) {
    qnep_->request_charge_diagnostics_for_next_force();
    if (g1_channel_) qnep_->request_peratom_virial_for_next_force();
  }
}

void QNEP_Projection::end_of_step(
  const int,
  int step,
  const int,
  const int,
  const double global_time,
  const double,
  Integrate&,
  Box& box,
  std::vector<Group>&,
  GPU_Vector<double>&,
  Atom& atom,
  Force&)
{
  if ((step + 1) % sample_interval_ != 0)
    return;

  check_fixed_cell(box);
  if (complete_current_) {
    write_complete_current(step, global_time, box, atom);
    return;
  }
  qnep_->compute_charge_rate(box, atom.type, atom.position_per_atom, atom.velocity_per_atom);
  double sum_charge = 0.0;
  double sum_charge_rate = 0.0;
  const int num_kpoints = qnep_->compute_delta_j_q_k(
    box,
    atom.position_per_atom,
    gpu_delta_j_q_k_,
    sum_charge,
    sum_charge_rate);
  gpu_delta_j_q_k_.copy_to_host(cpu_delta_j_q_k_.data());
  const double inv_time_conversion = 1.0 / TIME_UNIT_CONVERSION;
  fprintf(
    fid_delta_j_q_k_,
    "%d %.17g %.17g %.17g %d %.17g %.17g %.17g\n",
    step + 1,
    global_time * TIME_UNIT_CONVERSION,
    sum_charge,
    sum_charge_rate * inv_time_conversion,
    num_kpoints,
    cpu_delta_j_q_k_[0] * inv_time_conversion,
    cpu_delta_j_q_k_[1] * inv_time_conversion,
    cpu_delta_j_q_k_[2] * inv_time_conversion);

  if (g1_channel_) {
    qnep_->compute_charge_heat_channels(
      box, atom.type, atom.position_per_atom, atom.velocity_per_atom, gpu_channel_per_atom_);
    qnep_->compute_virial_components(
      box,
      atom.type,
      atom.position_per_atom,
      atom.virial_per_atom,
      true,
      true,
      true,
      gpu_virial_nep_,
      gpu_virial_electrostatic_fixed_,
      gpu_virial_dynamic_charge_);
    compute_heat(
      gpu_virial_dynamic_charge_, atom.velocity_per_atom, gpu_virial_heat_per_atom_);
    gpu_sum_components<<<NUM_CHANNEL_COMPONENTS, REDUCE_THREADS>>>(
      atom.number_of_atoms,
      NUM_CHANNEL_COMPONENTS,
      gpu_channel_per_atom_.data(),
      gpu_channel_total_.data());
    gpu_sum_components<<<NUM_HEAT_COMPONENTS, REDUCE_THREADS>>>(
      atom.number_of_atoms,
      NUM_HEAT_COMPONENTS,
      gpu_virial_heat_per_atom_.data(),
      gpu_virial_heat_total_.data());
    GPU_CHECK_KERNEL
    gpu_channel_total_.copy_to_host(cpu_channel_total_.data());
    gpu_virial_heat_total_.copy_to_host(cpu_virial_heat_total_.data());

    const double inv_time_conversion = 1.0 / TIME_UNIT_CONVERSION;
    const double jx_virial =
      (cpu_virial_heat_total_[0] + cpu_virial_heat_total_[1]) * inv_time_conversion;
    const double jy_virial =
      (cpu_virial_heat_total_[2] + cpu_virial_heat_total_[3]) * inv_time_conversion;
    const double jz_virial = cpu_virial_heat_total_[4] * inv_time_conversion;
    fprintf(
      fid_channel_,
      "%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
      step + 1,
      global_time * TIME_UNIT_CONVERSION,
      cpu_channel_total_[0] * inv_time_conversion,
      cpu_channel_total_[1] * inv_time_conversion,
      cpu_channel_total_[2] * inv_time_conversion,
      cpu_channel_total_[3] * inv_time_conversion,
      cpu_channel_total_[4] * inv_time_conversion,
      cpu_channel_total_[5] * inv_time_conversion,
      (cpu_channel_total_[0] + cpu_channel_total_[3]) * inv_time_conversion,
      (cpu_channel_total_[1] + cpu_channel_total_[4]) * inv_time_conversion,
      (cpu_channel_total_[2] + cpu_channel_total_[5]) * inv_time_conversion,
      jx_virial,
      jy_virial,
      jz_virial);
  }

  if (g1_channel_)
    return;

  const GPU_Vector<float>& D = qnep_->get_raw_D_reference();
  const GPU_Vector<float>& s = qnep_->get_raw_charge_rate_reference();
  const int N = atom.number_of_atoms;
  compute_projection_currents(
    N,
    box,
    atom.position_per_atom,
    atom.unwrapped_position,
    D,
    s);

  fprintf(
    fid_,
    "%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
    step + 1,
    global_time * TIME_UNIT_CONVERSION,
    cpu_total_[0] * inv_time_conversion,
    cpu_total_[1] * inv_time_conversion,
    cpu_total_[2] * inv_time_conversion,
    cpu_total_[3] * inv_time_conversion,
    cpu_total_[4] * inv_time_conversion,
    cpu_total_[5] * inv_time_conversion,
    cpu_total_[6] * inv_time_conversion,
    cpu_total_[7] * inv_time_conversion,
    cpu_total_[8] * inv_time_conversion,
    cpu_total_[9] * inv_time_conversion);

  fprintf(
    fid_projection_current_diag_,
    "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
    step + 1,
    global_time * TIME_UNIT_CONVERSION,
    cpu_total_[0] * inv_time_conversion,
    cpu_total_[1] * inv_time_conversion,
    cpu_total_[2] * inv_time_conversion,
    cpu_total_[3] * inv_time_conversion,
    cpu_total_[4] * inv_time_conversion,
    cpu_total_[5] * inv_time_conversion,
    cpu_total_[6] * inv_time_conversion,
    cpu_total_[7] * inv_time_conversion,
    cpu_total_[8] * inv_time_conversion);
}

void QNEP_Projection::post_run(
  Atom&,
  Box&,
  Integrate&,
  const int,
  const double,
  const double)
{
  if (fid_ != nullptr) {
    fclose(fid_);
    fid_ = nullptr;
  }
  if (fid_channel_ != nullptr) {
    fclose(fid_channel_);
    fid_channel_ = nullptr;
  }
  if (fid_delta_j_q_k_ != nullptr) {
    fclose(fid_delta_j_q_k_);
    fid_delta_j_q_k_ = nullptr;
  }
  if (fid_projection_current_diag_ != nullptr) {
    fclose(fid_projection_current_diag_);
    fid_projection_current_diag_ = nullptr;
  }
  if (fid_complete_current_ != nullptr) {
    fclose(fid_complete_current_);
    fid_complete_current_ = nullptr;
  }
}

void QNEP_Projection::parse(const char** param, int num_param)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("compute_qnep_projection should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval for compute_qnep_projection should be an integer number.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval for compute_qnep_projection should be positive.\n");
  }
  if (num_param == 3) {
    if (complete_current_) {
      PRINT_INPUT_ERROR("compute_qnep_current_diag has only one parameter: the sample interval.\n");
    }
    if (std::strcmp(param[2], "g1_channel") != 0) {
      PRINT_INPUT_ERROR("The optional compute_qnep_projection mode must be g1_channel.\n");
    }
    g1_channel_ = true;
  }
  printf(
    complete_current_ ? "Compute complete qNEP A/B/D candidate-current diagnostics.\n"
                       : g1_channel_ ? "Compute classical qNEP G[1] channel and reciprocal diagnostics.\n"
                                     : "Compute classical qNEP projection diagnostics.\n");
  printf("    sample interval is %d.\n", sample_interval_);
}

QNEP_Projection::QNEP_Projection(const char** param, int num_param, bool complete_current)
  : complete_current_(complete_current)
{
  parse(param, num_param);
  action_name = complete_current_ ? "compute_qnep_current_diag" : "compute_qnep_projection";
}
