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
The k-space part of the PPPM method.
------------------------------------------------------------------------------*/

#include "pppm.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <vector>

#ifdef USE_HIP
using PPPMDoubleComplex = hipfftDoubleComplex;
#else
using PPPMDoubleComplex = cufftDoubleComplex;
#endif

namespace{

bool pppm_make_double_plan(gpufftHandle& plan, const int K0, const int K1, const int K2)
{
#ifdef USE_HIP
  return hipfftPlan3d(&plan, K2, K1, K0, HIPFFT_Z2Z) == HIPFFT_SUCCESS;
#else
  return cufftPlan3d(&plan, K2, K1, K0, CUFFT_Z2Z) == CUFFT_SUCCESS;
#endif
}

bool pppm_forward_double(gpufftHandle plan, PPPMDoubleComplex* data)
{
#ifdef USE_HIP
  return hipfftExecZ2Z(plan, data, data, HIPFFT_FORWARD) == HIPFFT_SUCCESS;
#else
  return cufftExecZ2Z(plan, data, data, CUFFT_FORWARD) == CUFFT_SUCCESS;
#endif
}

struct PPPMDoublePlan
{
  gpufftHandle handle = 0;
  bool initialized = false;
  ~PPPMDoublePlan()
  {
    if (initialized) {
#ifdef USE_HIP
      hipfftDestroy(handle);
#else
      cufftDestroy(handle);
#endif
    }
  }
};

constexpr int max_mesh_points = 512 * 512 * 512;

bool is_good_K(int n)
{
  const int primes[4] = {2, 3, 5, 7};
  for (const int p : primes) {
    while (n % p == 0) n /= p;
  }
  return n == 1;
}

int get_best_K(const double required)
{
  int n = static_cast<int>(std::ceil(std::fmin(required, max_mesh_points)));
  if (n < 16) n = 16;
  if (n % 2 != 0) ++n;
  while (!is_good_K(n)) n += 2;
  return n;
}

constexpr const char* PPPM_DEBUG_SOURCE_SIGNATURE = "PPPM_ASSIGN_DEBUG_20260910_V1";
constexpr const char* PPPM_DYNAMIC_SOURCE_SIGNATURE = "PPPM_DYNAMIC_Q_DIAG";
constexpr const char* PPPM_DYNAMIC_DIAG_SCHEMA_MARKER = "# dynamic_q_diag_schema_version = 4";
constexpr const char* PPPM_DYNAMIC_CHECK_COLUMN_HEADER =
  "# columns: step time_fs bead_id pppm_call_index max_abs_J_mesh_path "
  "assignment_charge_sum_error assignment_qdot_sum_error assignment_sums_ok "
  "mesh_path_ok all_values_finite max_odd_error_dx max_odd_error_dy max_odd_error_dz "
  "max_imag_L1S_x max_imag_L1S_y max_imag_L1S_z compute_valid "
  "diagnostic_checks_pass gpu_cpu_match max_abs_DeltaJ_pppm_gpu_cpu_error";
constexpr const char* PPPM_DYNAMIC_ATOM_COLUMN_HEADER =
  "# columns atom_id x y z q qdot_internal qdot_e_per_fs";
constexpr const char* PPPM_DYNAMIC_KSPACE_COLUMN_HEADER =
  "# columns ix iy iz nx ny nz kx ky kz Gopt g ell "
  "d_raw_x d_raw_y d_raw_z d_x d_y d_z rho_real rho_imag s_real s_imag";
constexpr const char* PPPM_DYNAMIC_DIAG_COLUMN_HEADER =
  "source_signature,dynamic_formula_version,step,time_fs,pppm_call_index,bead_id,N,N1,N2,"
  "mesh_x,mesh_y,mesh_z,Ng,alpha,K_C_SP,TIME_UNIT_CONVERSION,"
  "sum_q_proj_0_N,sum_qdot_proj_0_N,sum_q_assign_N1_N2,sum_qdot_assign_N1_N2,"
  "sum_Q,sum_S,rho_zero_real,rho_zero_imag,s_zero_real,s_zero_imag,"
  "J_ass_left_x,J_ass_left_y,J_ass_left_z,J_ass_right_x,J_ass_right_y,J_ass_right_z,"
  "J_ass_x,J_ass_y,J_ass_z,J_mesh_fourier_x,J_mesh_fourier_y,J_mesh_fourier_z,"
  "J_mesh_realspace_x,J_mesh_realspace_y,J_mesh_realspace_z,J_mesh_x,J_mesh_y,J_mesh_z,"
  "DeltaJ_pppm_x,DeltaJ_pppm_y,DeltaJ_pppm_z,max_odd_error_dx,max_odd_error_dy,"
  "max_odd_error_dz,max_imag_L1S_x,max_imag_L1S_y,max_imag_L1S_z,"
  "assignment_charge_sum_error,assignment_qdot_sum_error,max_abs_J_mesh_path,"
  "assignment_sums_ok,mesh_path_ok,all_values_finite,"
  "h00,h01,h02,h10,h11,h12,h20,h21,h22,"
  "mesh_path_scale_internal,mesh_path_threshold_internal,"
  "mesh_path_scale_eV_A_fs,mesh_path_threshold_eV_A_fs,"
  "DeltaJ_pppm_gpu_x,DeltaJ_pppm_gpu_y,DeltaJ_pppm_gpu_z,"
  "DeltaJ_pppm_cpu_x,DeltaJ_pppm_cpu_y,DeltaJ_pppm_cpu_z,"
  "DeltaJ_pppm_gpu_cpu_error_x,DeltaJ_pppm_gpu_cpu_error_y,"
  "DeltaJ_pppm_gpu_cpu_error_z,compute_valid,diagnostic_checks_pass,gpu_cpu_match,"
  "DeltaJ_q_real_x,DeltaJ_q_real_y,DeltaJ_q_real_z,"
  "DeltaJ_q_total_x,DeltaJ_q_total_y,DeltaJ_q_total_z,dynamic_q_valid,charge_mode";

std::string pppm_dynamic_formula_marker()
{
  return "# dynamic_formula_version = " +
    std::string(PPPM::dynamic_q_formula_version());
}

void write_dynamic_metadata(std::ostream& file)
{
  file << "# source_signature = " << PPPM_DYNAMIC_SOURCE_SIGNATURE << "\n";
  file << PPPM_DYNAMIC_DIAG_SCHEMA_MARKER << "\n";
  file << pppm_dynamic_formula_marker() << "\n";
  file << "# q_source = nep_data.charge\n";
  file << "# qdot_source = nep_data.charge_rate\n";
  file << "# q_projection = zero_total_charge(0:N)\n";
  file << "# qdot_projection = zero_total_charge(0:N)\n";
  file << "# assignment_domain = N1:N2\n";
  file << "# fft_forward = unnormalized\n";
  file << "# fft_inverse = unnormalized\n";
  file << "# phase_convention = forward exp(-i*2pi*n.j/K), inverse exp(+i*2pi*n.j/K)\n";
  file << "# g_m = Gopt_m\n";
  file << "# ell_m = Ng*Gopt_m\n";
  file << "# L_prefactor = 1/Ng\n";
  file << "# D_prefactor = 2*K_C_SP\n";
  file << "# d_zero_mode = 0\n";
  file << "# d_nyquist_component_plane = 0\n";
  file << "# d_pair_projection = 0.5*(d_raw[m]-d_raw[mbar])\n";
  file << "# d_odd_error = max_abs(d[mbar]+d[m])\n";
  file << "# qdot_internal_unit = e/natural_time\n";
  file << "# qdot_output_unit = e/fs\n";
  file << "# sum_qdot_proj_0_N_unit = e/natural_time\n";
  file << "# sum_qdot_assign_N1_N2_unit = e/natural_time\n";
  file << "# sum_S_unit = e/natural_time\n";
  file << "# s_zero_unit = e/natural_time\n";
  file << "# J_internal_unit = eV*Angstrom/natural_time\n";
  file << "# J_output_unit = eV*Angstrom/fs\n";
  file << "# J_conversion = divide_by_TIME_UNIT_CONVERSION_at_CSV_write\n";
  file << "# dynamic_q_component_output_unit = DeltaJ_pppm,DeltaJ_q_real,DeltaJ_q_total = eV*Angstrom/fs\n";
  file << "# delta_j_q_pppm = reciprocal_only\n";
  file << "# DeltaJ_pppm = GPU production reduction; DeltaJ_pppm_cpu = independent host reference\n";
  file << "# DeltaJ_pppm_gpu_cpu_error = absolute output-unit difference; gpu_cpu_match uses mixed tolerance\n";
  file << "# delta_j_q_real = qNEP_real_space_mode1; mode2_zero\n";
  file << "# delta_j_q_real_formula = K_C_SP/2 * sum_a(qdot_a * sum_b(q_b*erfc(alpha*r_ab)/r_ab*d_ab))\n";
  file << "# delta_j_q_total = delta_j_q_pppm + mode1*delta_j_q_real; mode2=delta_j_q_pppm\n";
  file << "# charge_mode = per_row\n";
  file << "# compute_valid = finite production GPU reciprocal current was obtained\n";
  file << "# diagnostic_checks_pass = independent diagnostic checks passed; does not gate HAC compute_valid\n";
  file << "# dynamic_q_valid = combined reciprocal/real correction compute_valid\n";
  file << "# gpu_cpu_match = production GPU reciprocal current matches independent CPU reference within mixed tolerance\n";
  file << "# geometry_restriction = fixed orthogonal cell; diagnostic sampling stage is caller-defined\n";
  file << "# nyquist_rule = Cartesian component plane zero (orthogonal-cell "
       << PPPM::dynamic_q_formula_version() << " only)\n";
  file << "# diagnostic_relative_tolerance = 1e-5\n";
  file << "# mesh_path_scale_internal_unit = eV*Angstrom/natural_time\n";
  file << "# mesh_path_threshold_internal = diagnostic_relative_tolerance * mesh_path_scale_internal\n";
  file << "# mesh_path_scale_output_unit = eV*Angstrom/fs\n";
  file << "# mesh_path_threshold_output_unit = eV*Angstrom/fs\n";
  file << "# csv_frequency = every sampled diagnostic call; file_write = post_run\n";
  file << "# detailed_debug_frequency = first diagnostic call only\n";
}

bool existing_file_has_schema(
  const char* filename, const std::vector<std::string>& required_lines)
{
  int stat_status = 0;
#ifdef _WIN32
  struct _stat file_info;
  stat_status = _stat(filename, &file_info);
#else
  struct stat file_info;
  stat_status = stat(filename, &file_info);
#endif
  if (stat_status != 0) {
    const int stat_errno = errno;
    return stat_errno == ENOENT;
  }

  std::ifstream probe(filename, std::ios::binary);
  if (!probe) return false;

  bool saw_line = false;
  std::vector<bool> found(required_lines.size(), false);
  std::string line;
  while (std::getline(probe, line)) {
    saw_line = true;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    for (size_t i = 0; i < required_lines.size(); ++i) {
      if (line == required_lines[i]) found[i] = true;
    }
    if (!line.empty() && line.front() != '#') break;
  }
  if (probe.bad()) return false;
  if (!saw_line) return true;
  for (const bool line_found : found) {
    if (!line_found) return false;
  }
  return true;
}

bool append_text_file(
  const char* filename,
  const std::string& header,
  const std::string& rows,
  const std::vector<std::string>& required_lines)
{
  if (rows.empty()) return true;
  int stat_status = 0;
#ifdef _WIN32
  struct _stat file_info;
  stat_status = _stat(filename, &file_info);
#else
  struct stat file_info;
  stat_status = stat(filename, &file_info);
#endif
  bool empty = false;
  if (stat_status != 0) {
    const int stat_errno = errno;
    if (stat_errno != ENOENT) return false;
    empty = true;
  } else {
    if (!required_lines.empty() && !existing_file_has_schema(filename, required_lines))
      return false;
    std::ifstream probe(filename, std::ios::binary | std::ios::ate);
    if (!probe) return false;
    const std::streampos end = probe.tellg();
    if (end == std::streampos(-1)) return false;
    empty = end == std::streampos(0);
  }
  std::ofstream file(filename, std::ios::app);
  if (!file) return false;
  if (empty) file << header;
  file << rows;
  file.flush();
  return file.good();
}

__constant__ float sinc_coeff[6] = {1.0f, -1.6666667e-1f, 8.3333333e-3f, -1.9841270e-4f, 2.7557319e-6f, -2.5052108e-8f};
__constant__ float G_coeff[5] = {1.0000000e+00f, -1.6666667e+00f, 7.7777778e-01f, -8.9947090e-02f, 7.0546737e-04f};
__constant__ float W_coeff[5][5] = {
  {2.6041667e-03f, -2.0833333e-02f, 6.2500000e-02f, -8.3333333e-02f, 4.1666667e-02f},
  {1.9791667e-01f, -4.5833333e-01f, 2.5000000e-01f, 1.6666667e-01f, -1.6666667e-01f},
  {5.9895833e-01f, 0.0000000e+00f, -6.2500000e-01f, 0.0000000e+00f, 2.5000000e-01f},
  {1.9791667e-01f, 4.5833333e-01f, 2.5000000e-01f, -1.6666667e-01f, -1.6666667e-01f},
  {2.6041667e-03f, 2.0833333e-02f, 6.2500000e-02f, 8.3333333e-02f, 4.1666667e-02f}
};

struct PPPMReferenceStencil
{
  int index[3];
  float weight[3][5];
  float derivative[3][5];
  double ds_dR[3][3];
};

__device__ inline PPPMReferenceStencil make_reference_stencil(
  const PPPM::Para para, const Box box, const double x, const double y, const double z)
{
  const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
  const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
  const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
  const float delta[3] = {
    sx - int(sx + 0.5f), sy - int(sy + 0.5f), sz - int(sz + 0.5f)};
  PPPMReferenceStencil result;
  result.index[0] = int(sx + 0.5f);
  result.index[1] = int(sy + 0.5f);
  result.index[2] = int(sz + 0.5f);
  for (int axis = 0; axis < 3; ++axis) {
    for (int j = 0; j < 5; ++j) {
      result.weight[axis][j] = pppm_reference_weight(W_coeff, j, delta[axis]);
      result.derivative[axis][j] = pppm_reference_weight_derivative(W_coeff, j, delta[axis]);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (int mu = 0; mu < 3; ++mu)
      result.ds_dR[axis][mu] = static_cast<double>(para.K[axis]) * box.cpu_h[9 + 3 * axis + mu];
  }
  return result;
}

__device__ inline double reference_stencil_dW_dR(
  const PPPMReferenceStencil& stencil, const int n0, const int n1, const int n2, const int mu)
{
  const int i = n0 + 2, j = n1 + 2, k = n2 + 2;
  const double wx = stencil.weight[0][i], wy = stencil.weight[1][j], wz = stencil.weight[2][k];
  return static_cast<double>(stencil.derivative[0][i]) * wy * wz * stencil.ds_dR[0][mu] +
         wx * static_cast<double>(stencil.derivative[1][j]) * wz * stencil.ds_dR[1][mu] +
         wx * wy * static_cast<double>(stencil.derivative[2][k]) * stencil.ds_dR[2][mu];
}

__device__ inline float sinc(const float x)
{
  float y = 0.0f;
  if (x * x <= 1.0f) {
    float term = 1.0f;
    for (int i = 0; i < 6; ++i) {
      y += sinc_coeff[i] * term;
      term *= x * x;
    }
  } else {
    y = sin(x) / x;
  }
  return y;
}

void __global__ find_k_and_G_opt(
  const PPPM::Para para,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    int nk[3];
    nk[2] = n / para.K0K1;
    nk[1] = (n - nk[2] * para.K0K1) / para.K[0];
    nk[0] = n % para.K[0];

    // Eqs. (2.25) and (2.26) in V. Ballenegger, J. J. Cerda, and C. Holm, JCTC 8, 936 (2012)
    float denominator[3] = {0.0f};
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) {
        nk[d] -= para.K[d];
      }
      float t = sin(0.5f * para.two_pi_over_K[d] * nk[d]);
      t *= t;
      t = (((G_coeff[4] * t + G_coeff[3]) * t + G_coeff[2]) * t + G_coeff[1]) * t + G_coeff[0];
      denominator[d] = t * t;
    }
    const float kx = nk[0] * para.b[0][0] + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    const float ky = nk[0] * para.b[0][1] + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    const float kz = nk[0] * para.b[0][2] + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    g_kx[n] = kx;
    g_ky[n] = ky;
    g_kz[n] = kz;
    const float ksq = kx * kx + ky * ky + kz * kz;

    // Eqs. (2.21) and (2.25) in V. Ballenegger, J. J. Cerda, and C. Holm, JCTC 8, 936 (2012)
    float numerator = sinc(0.5f * para.two_pi_over_K[0] * nk[0]);
    numerator *= sinc(0.5f * para.two_pi_over_K[1] * nk[1]);
    numerator *= sinc(0.5f * para.two_pi_over_K[2] * nk[2]);
    numerator = numerator * numerator * numerator * numerator * numerator;
    numerator *= numerator;

    // Eqs. (2.25) in V. Ballenegger, J. J. Cerda, and C. Holm, JCTC 8, 936 (2012)
    if (ksq == 0.0f) {
      g_G[n] = 0.0f;
    } else {
      float G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
      G_opt /= denominator[0] * denominator[1] * denominator[2];
      g_G[n] = G_opt;
    }
  }
}

void __global__ set_mesh_to_zero(const PPPM::Para para, gpufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    g_mesh[n].x = 0.0f;
    g_mesh[n].y = 0.0f;
  }
}

__device__ inline int get_index_within_mesh(const int K, const int n)
{
  int y = n;
  if (n >= K) {
    y = n - K;
  } else if (n < 0) {
    y = n + K;
  }
  return y;
}

__global__ void find_mesh(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  gpufftComplex* g_mesh,
  PPPMAssignmentAtomDebug* g_debug_atoms,
  PPPMAssignmentStencilDebug* g_debug_stencil)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = g_charge[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f); // can be 0, ..., K[0]
    const int iy = int(sy + 0.5f); // can be 0, ..., K[1]
    const int iz = int(sz + 0.5f); // can be 0, ..., K[2]
    const float dx = sx - ix; // (-0.5, 0.5)
    const float dy = sy - iy; // (-0.5, 0.5)
    const float dz = sz - iz; // (-0.5, 0.5)
    // Appendix E in M. Deserno and C. Holm, JCP 109, 7678 (1998)
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    const int debug_slot = n - N1;
    if (g_debug_atoms != nullptr && debug_slot < 8) {
      PPPMAssignmentAtomDebug& d = g_debug_atoms[debug_slot];
      d.atom_id = n;
      d.q = q;
      d.x = x;
      d.y = y;
      d.z = z;
      d.sx = sx;
      d.sy = sy;
      d.sz = sz;
      d.ix = ix;
      d.iy = iy;
      d.iz = iz;
      d.dx = dx;
      d.dy = dy;
      d.dz = dz;
      for (int d0 = 0; d0 < 5; ++d0) {
        d.Wx[d0] = Wx[d0];
        d.Wy[d0] = Wy[d0];
        d.Wz[d0] = Wz[d0];
      }
    }
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          const float qW = q * W;
          if (g_debug_stencil != nullptr && n == 0) {
            const int debug_index = (n0 + 2) * 25 + (n1 + 2) * 5 + (n2 + 2);
            PPPMAssignmentStencilDebug& d = g_debug_stencil[debug_index];
            d.n0 = n0;
            d.n1 = n1;
            d.n2 = n2;
            d.neighbor0 = neighbor0;
            d.neighbor1 = neighbor1;
            d.neighbor2 = neighbor2;
            d.neighbor012 = neighbor012;
            d.W = W;
            d.qW = qW;
          }
          atomicAdd(&g_mesh[neighbor012].x, qW);
        }
      }
    }
  }
}

__global__ void clear_reference_delta_Q(double* delta_Q, const int M)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) delta_Q[i] = 0.0;
}

__global__ void assign_reference_delta_Q(
  const int N,
  const PPPM::Para para,
  const Box box,
  const float* charge,
  const double* position,
  const double* direction,
  const double* charge_direction,
  double* delta_Q)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const PPPMReferenceStencil stencil = make_reference_stencil(
    para, box, position[i], position[i + N], position[i + 2 * N]);
  const double qi = charge[i];
  const double dqi = charge_direction[i];
  for (int n0 = -2; n0 <= 2; ++n0) {
    const int j0 = get_index_within_mesh(para.K[0], stencil.index[0] + n0);
    for (int n1 = -2; n1 <= 2; ++n1) {
      const int j1 = get_index_within_mesh(para.K[1], stencil.index[1] + n1);
      for (int n2 = -2; n2 <= 2; ++n2) {
        const int j2 = get_index_within_mesh(para.K[2], stencil.index[2] + n2);
        const int mesh_index = j0 + para.K[0] * (j1 + para.K[1] * j2);
        const float W = stencil.weight[0][n0 + 2] * stencil.weight[1][n1 + 2] *
                        stencil.weight[2][n2 + 2];
        const double dW = reference_stencil_dW_dR(stencil, n0, n1, n2, 0) * direction[i] +
                          reference_stencil_dW_dR(stencil, n0, n1, n2, 1) * direction[i + N] +
                          reference_stencil_dW_dR(stencil, n0, n1, n2, 2) * direction[i + 2 * N];
        const double value = dqi * static_cast<double>(W) + qi * dW;
        atomicAdd(&delta_Q[mesh_index], value);
      }
    }
  }
}

__global__ void convert_reference_delta_Q(
  const double* delta_Q, gpufftComplex* mesh, const int M)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) mesh[i] = {static_cast<float>(delta_Q[i]), 0.0f};
}

__global__ void apply_reference_G(
  const float* G, gpufftComplex* mesh, const int M)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) {
    mesh[i].x *= G[i];
    mesh[i].y *= G[i];
  }
}

__global__ void gather_reference_energy_tangent(
  const int N,
  const PPPM::Para para,
  const Box box,
  const float* charge,
  const double* position,
  const double* direction,
  const double* charge_direction,
  const gpufftComplex* phi,
  const gpufftComplex* delta_phi,
  const gpufftComplex* field_x,
  const gpufftComplex* field_y,
  const gpufftComplex* field_z,
  double* dsite,
  double* explicit_gradient,
  double* native_ik_force)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const PPPMReferenceStencil stencil = make_reference_stencil(
    para, box, position[i], position[i + N], position[i + 2 * N]);
  double potential = 0.0, dpotential_shape = 0.0, dpotential_field = 0.0;
  double grad[3] = {0.0, 0.0, 0.0}, ik[3] = {0.0, 0.0, 0.0};
  for (int n0 = -2; n0 <= 2; ++n0) {
    const int j0 = get_index_within_mesh(para.K[0], stencil.index[0] + n0);
    for (int n1 = -2; n1 <= 2; ++n1) {
      const int j1 = get_index_within_mesh(para.K[1], stencil.index[1] + n1);
      for (int n2 = -2; n2 <= 2; ++n2) {
        const int j2 = get_index_within_mesh(para.K[2], stencil.index[2] + n2);
        const int mesh_index = j0 + para.K[0] * (j1 + para.K[1] * j2);
        const int a = n0 + 2, b = n1 + 2, c = n2 + 2;
        const float W = stencil.weight[0][a] * stencil.weight[1][b] * stencil.weight[2][c];
        const double value = phi[mesh_index].x;
        potential += static_cast<double>(W) * value;
        if (delta_phi != nullptr) dpotential_field += static_cast<double>(W) * delta_phi[mesh_index].x;
        const double shape = reference_stencil_dW_dR(stencil, n0, n1, n2, 0);
        const double shape_y = reference_stencil_dW_dR(stencil, n0, n1, n2, 1);
        const double shape_z = reference_stencil_dW_dR(stencil, n0, n1, n2, 2);
        dpotential_shape += (direction == nullptr ? 0.0 :
          (shape * direction[i] + shape_y * direction[i + N] + shape_z * direction[i + 2 * N])) * value;
        grad[0] += shape * value;
        grad[1] += shape_y * value;
        grad[2] += shape_z * value;
        ik[0] += static_cast<double>(W) * field_x[mesh_index].x;
        ik[1] += static_cast<double>(W) * field_y[mesh_index].x;
        ik[2] += static_cast<double>(W) * field_z[mesh_index].x;
      }
    }
  }
  const double qi = charge[i];
  const double prefactor = static_cast<double>(K_C_SP) * qi;
  if (dsite != nullptr) {
    const double dqi = charge_direction[i];
    dsite[i] = static_cast<double>(K_C_SP) *
      (dqi * potential + qi * dpotential_shape + qi * dpotential_field);
  }
  for (int mu = 0; mu < 3; ++mu) {
    explicit_gradient[i + mu * N] = 2.0 * prefactor * grad[mu];
    native_ik_force[i + mu * N] = 2.0 * prefactor * ik[mu];
  }
}

__device__ inline float dynamic_sinc_with_derivative(const float x, float& derivative)
{
  const float x2 = x * x;
  if (x2 <= 1.0f) {
    float value = 0.0f;
    float power = 1.0f;
    float previous_power = 1.0f;
    derivative = 0.0f;
    for (int i = 0; i < 6; ++i) {
      value += sinc_coeff[i] * power;
      if (i > 0) {
        derivative += 2.0f * i * sinc_coeff[i] * x * previous_power;
      }
      previous_power = power;
      power *= x2;
    }
    return value;
  }
  const float sin_x = sin(x);
  derivative = (x * cos(x) - sin_x) / (x * x);
  return sin_x / x;
}

__device__ inline float dynamic_G_polynomial(const float z)
{
  return (((G_coeff[4] * z + G_coeff[3]) * z + G_coeff[2]) * z + G_coeff[1]) * z + G_coeff[0];
}

__device__ inline float dynamic_G_polynomial_derivative(const float z)
{
  return ((4.0f * G_coeff[4] * z + 3.0f * G_coeff[3]) * z + 2.0f * G_coeff[2]) * z + G_coeff[1];
}

__device__ inline float dynamic_log_influence_derivative(const float u)
{
  float sinc_derivative = 0.0f;
  const float sinc_value = dynamic_sinc_with_derivative(u, sinc_derivative);
  const float sin_u = sin(u);
  const float z = sin_u * sin_u;
  const float denominator = dynamic_G_polynomial(z);
  if (sinc_value == 0.0f || denominator == 0.0f) return 0.0f;
  return 10.0f * sinc_derivative / sinc_value -
         4.0f * sin_u * cos(u) * dynamic_G_polynomial_derivative(z) / denominator;
}

__global__ void find_dynamic_d_raw(
  const PPPM::Para para,
  const Box box,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  float* g_d_raw_x,
  float* g_d_raw_y,
  float* g_d_raw_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    int nk[3];
    nk[2] = n / para.K0K1;
    nk[1] = (n - nk[2] * para.K0K1) / para.K[0];
    nk[0] = n % para.K[0];
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) nk[d] -= para.K[d];
    }

    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float ksq = kx * kx + ky * ky + kz * kz;
    if (ksq == 0.0f) {
      g_d_raw_x[n] = 0.0f;
      g_d_raw_y[n] = 0.0f;
      g_d_raw_z[n] = 0.0f;
      return;
    }

    const float gamma0 = dynamic_log_influence_derivative(
      0.5f * para.two_pi_over_K[0] * nk[0]);
    const float gamma1 = dynamic_log_influence_derivative(
      0.5f * para.two_pi_over_K[1] * nk[1]);
    const float gamma2 = dynamic_log_influence_derivative(
      0.5f * para.two_pi_over_K[2] * nk[2]);

    const float h0x = static_cast<float>(box.cpu_h[0]) / para.K[0];
    const float h0y = static_cast<float>(box.cpu_h[3]) / para.K[0];
    const float h0z = static_cast<float>(box.cpu_h[6]) / para.K[0];
    const float h1x = static_cast<float>(box.cpu_h[1]) / para.K[1];
    const float h1y = static_cast<float>(box.cpu_h[4]) / para.K[1];
    const float h1z = static_cast<float>(box.cpu_h[7]) / para.K[1];
    const float h2x = static_cast<float>(box.cpu_h[2]) / para.K[2];
    const float h2y = static_cast<float>(box.cpu_h[5]) / para.K[2];
    const float h2z = static_cast<float>(box.cpu_h[8]) / para.K[2];
    const float radial_derivative = -2.0f / ksq - 2.0f * para.alpha_factor;
    const float prefactor = static_cast<float>(para.K0K1K2) * g_G[n];
    g_d_raw_x[n] = prefactor *
      (radial_derivative * kx + 0.5f * (gamma0 * h0x + gamma1 * h1x + gamma2 * h2x));
    g_d_raw_y[n] = prefactor *
      (radial_derivative * ky + 0.5f * (gamma0 * h0y + gamma1 * h1y + gamma2 * h2y));
    g_d_raw_z[n] = prefactor *
      (radial_derivative * kz + 0.5f * (gamma0 * h0z + gamma1 * h1z + gamma2 * h2z));
  }
}

__global__ void project_dynamic_d(
  const PPPM::Para para,
  const float* g_d_raw_x,
  const float* g_d_raw_y,
  const float* g_d_raw_z,
  float* g_d_x,
  float* g_d_y,
  float* g_d_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const int iz = n / para.K0K1;
    const int iy = (n - iz * para.K0K1) / para.K[0];
    const int ix = n % para.K[0];
    const int ix_bar = (para.K[0] - ix) % para.K[0];
    const int iy_bar = (para.K[1] - iy) % para.K[1];
    const int iz_bar = (para.K[2] - iz) % para.K[2];
    const int n_bar = ix_bar + para.K[0] * (iy_bar + para.K[1] * iz_bar);
    if (n == n_bar) {
      g_d_x[n] = 0.0f;
      g_d_y[n] = 0.0f;
      g_d_z[n] = 0.0f;
      return;
    }
    // Cartesian Nyquist-plane projection is valid only for orthogonal cells.
    g_d_x[n] = (para.K[0] % 2 == 0 && ix == para.K_half[0])
      ? 0.0f
      : 0.5f * (g_d_raw_x[n] - g_d_raw_x[n_bar]);
    g_d_y[n] = (para.K[1] % 2 == 0 && iy == para.K_half[1])
      ? 0.0f
      : 0.5f * (g_d_raw_y[n] - g_d_raw_y[n_bar]);
    g_d_z[n] = (para.K[2] % 2 == 0 && iz == para.K_half[2])
      ? 0.0f
      : 0.5f * (g_d_raw_z[n] - g_d_raw_z[n_bar]);
  }
}

__global__ void find_dynamic_mesh(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const float* g_charge_rate,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  gpufftComplex* g_Q,
  gpufftComplex* g_S,
  gpufftComplex* g_Ax,
  gpufftComplex* g_Ay,
  gpufftComplex* g_Az,
  gpufftComplex* g_Bx,
  gpufftComplex* g_By,
  gpufftComplex* g_Bz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = g_charge[n];
    const float qdot = g_charge_rate[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f);
    const int iy = int(sy + 0.5f);
    const int iz = int(sz + 0.5f);
    const float dx = sx - ix;
    const float dy = sy - iy;
    const float dz = sz - iz;
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          const float qW = q * W;
          const float qdotW = qdot * W;
          const double s0 = static_cast<double>(ix + n0) / para.K[0];
          const double s1 = static_cast<double>(iy + n1) / para.K[1];
          const double s2 = static_cast<double>(iz + n2) / para.K[2];
          const double image_x = box.cpu_h[0] * s0 + box.cpu_h[1] * s1 + box.cpu_h[2] * s2;
          const double image_y = box.cpu_h[3] * s0 + box.cpu_h[4] * s1 + box.cpu_h[5] * s2;
          const double image_z = box.cpu_h[6] * s0 + box.cpu_h[7] * s1 + box.cpu_h[8] * s2;
          if (g_Q != nullptr) atomicAdd(&g_Q[neighbor012].x, qW);
          atomicAdd(&g_S[neighbor012].x, qdotW);
          atomicAdd(&g_Ax[neighbor012].x, static_cast<float>(image_x - x) * qW);
          atomicAdd(&g_Ay[neighbor012].x, static_cast<float>(image_y - y) * qW);
          atomicAdd(&g_Az[neighbor012].x, static_cast<float>(image_z - z) * qW);
          atomicAdd(&g_Bx[neighbor012].x, static_cast<float>(image_x - x) * qdotW);
          atomicAdd(&g_By[neighbor012].x, static_cast<float>(image_y - y) * qdotW);
          atomicAdd(&g_Bz[neighbor012].x, static_cast<float>(image_z - z) * qdotW);
        }
      }
    }
  }
}

__global__ void dynamic_i_d_times_s(
  const PPPM::Para para,
  const float* g_d_x,
  const float* g_d_y,
  const float* g_d_z,
  const gpufftComplex* g_S,
  gpufftComplex* g_L1S_x,
  gpufftComplex* g_L1S_y,
  gpufftComplex* g_L1S_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const gpufftComplex S = g_S[n];
    g_L1S_x[n] = {-g_d_x[n] * S.y, g_d_x[n] * S.x};
    g_L1S_y[n] = {-g_d_y[n] * S.y, g_d_y[n] * S.x};
    g_L1S_z[n] = {-g_d_z[n] * S.y, g_d_z[n] * S.x};
  }
}

void __global__ ik_times_mesh_times_G(
  const PPPM::Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const gpufftComplex* g_mesh_fft,
  gpufftComplex* g_mesh_fft_x,
  gpufftComplex* g_mesh_fft_y,
  gpufftComplex* g_mesh_fft_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float G = g_G[n];
    gpufftComplex mesh_fft = g_mesh_fft[n];
    g_mesh_fft_x[n] = {mesh_fft.y * kx * G, -mesh_fft.x * kx * G};
    g_mesh_fft_y[n] = {mesh_fft.y * ky * G, -mesh_fft.x * ky * G};
    g_mesh_fft_z[n] = {mesh_fft.y * kz * G, -mesh_fft.x * kz * G};
  }
}

void __global__ find_mesh_G(
  const PPPM::Para para,
  const float* g_G,
  const gpufftComplex* g_mesh,
  gpufftComplex* g_mesh_G)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float G = g_G[n];
    gpufftComplex mesh = g_mesh[n];
    g_mesh_G[n] = {mesh.x * G, mesh.y * G};
  }
}

__global__ void reduce_dynamic_mesh_current(
  const int M,
  const gpufftComplex* g_Q,
  const gpufftComplex* g_S,
  const float* g_d_x,
  const float* g_d_y,
  const float* g_d_z,
  double* g_current)
{
  __shared__ double s_data[1024];
  const int component = blockIdx.x;
  const float* g_d = component == 0 ? g_d_x : (component == 1 ? g_d_y : g_d_z);
  double sum = 0.0;
  for (int n = threadIdx.x; n < M; n += blockDim.x) {
    const gpufftComplex Q = g_Q[n];
    const gpufftComplex S = g_S[n];
    const double im_conjugate_QS = double(Q.x) * double(S.y) - double(Q.y) * double(S.x);
    sum -= double(K_C_SP) / double(M) * double(g_d[n]) * im_conjugate_QS;
  }
  s_data[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) g_current[component] = s_data[0];
}

__global__ void reduce_dynamic_assignment_current(
  const int M,
  const gpufftComplex* g_Q,
  const gpufftComplex* g_S,
  const gpufftComplex* g_Ax,
  const gpufftComplex* g_Ay,
  const gpufftComplex* g_Az,
  const gpufftComplex* g_Bx,
  const gpufftComplex* g_By,
  const gpufftComplex* g_Bz,
  double* g_current)
{
  __shared__ double s_left[1024];
  __shared__ double s_right[1024];
  const int component = blockIdx.x;
  const gpufftComplex* g_A = component == 0 ? g_Ax : (component == 1 ? g_Ay : g_Az);
  const gpufftComplex* g_B = component == 0 ? g_Bx : (component == 1 ? g_By : g_Bz);
  double left = 0.0;
  double right = 0.0;
  for (int n = threadIdx.x; n < M; n += blockDim.x) {
    left -= double(K_C_SP) * double(g_A[n].x) * double(g_S[n].x);
    right += double(K_C_SP) * double(g_Q[n].x) * double(g_B[n].x);
  }
  s_left[threadIdx.x] = left;
  s_right[threadIdx.x] = right;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_left[threadIdx.x] += s_left[threadIdx.x + offset];
      s_right[threadIdx.x] += s_right[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    g_current[3 + component] = s_left[0];
    g_current[6 + component] = s_right[0];
  }
}

void __global__ find_mesh_virial(
  const PPPM::Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const gpufftComplex* g_S,
  gpufftComplex* g_mesh_virial_xx,
  gpufftComplex* g_mesh_virial_yy,
  gpufftComplex* g_mesh_virial_zz,
  gpufftComplex* g_mesh_virial_xy,
  gpufftComplex* g_mesh_virial_yz,
  gpufftComplex* g_mesh_virial_zx)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float ksq = kx * kx + ky * ky + kz * kz;
    if (ksq != 0.0f) {
      const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
      const float G = g_G[n];
      const gpufftComplex S = g_S[n];
      const float GSx = G * S.x;
      const float GSy = G * S.y;
      float B = 1.0f - alpha_k_factor * kx * kx;
      g_mesh_virial_xx[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * ky * ky;
      g_mesh_virial_yy[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * kz * kz;
      g_mesh_virial_zz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kx * ky;
      g_mesh_virial_xy[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * ky * kz;
      g_mesh_virial_yz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kz * kx;
      g_mesh_virial_zx[n] = {B * GSx, B * GSy};
    } else {
      // The k = 0 mode must be reset explicitly because mesh_virial is reused in-place
      // across steps and later overwritten by inverse FFT output.
      g_mesh_virial_xx[n] = {0.0f, 0.0f};
      g_mesh_virial_yy[n] = {0.0f, 0.0f};
      g_mesh_virial_zz[n] = {0.0f, 0.0f};
      g_mesh_virial_xy[n] = {0.0f, 0.0f};
      g_mesh_virial_yz[n] = {0.0f, 0.0f};
      g_mesh_virial_zx[n] = {0.0f, 0.0f};
    }
  }
}

__global__ void find_force_from_field(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = K_C_SP * g_charge[n] * 2.0f;
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f); // can be 0, ..., K[0]
    const int iy = int(sy + 0.5f); // can be 0, ..., K[1]
    const int iz = int(sz + 0.5f); // can be 0, ..., K[2]
    const float dx = sx - ix; // (-0.5, 0.5)
    const float dy = sy - iy; // (-0.5, 0.5)
    const float dz = sz - iz; // (-0.5, 0.5)
    // Appendix E in M. Deserno and C. Holm, JCP 109, 7678 (1998)
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    float D_real = 0.0f;
    float E[3] = {0.0f, 0.0f, 0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D_real += W * g_mesh_G[neighbor012].x;
          E[0] += W * g_mesh_fft_x_ifft[neighbor012].x;
          E[1] += W * g_mesh_fft_y_ifft[neighbor012].x;
          E[2] += W * g_mesh_fft_z_ifft[neighbor012].x;
        }
      }
    }
    g_D_real[n] = 2.0f * K_C_SP * D_real;
    g_fx[n] += q * E[0];
    g_fy[n] += q * E[1];
    g_fz[n] += q * E[2];
  } 
}

__global__ void find_force_virial_potential_from_field(
  const int N,
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  const gpufftComplex* g_mesh_virial_xx,
  const gpufftComplex* g_mesh_virial_yy,
  const gpufftComplex* g_mesh_virial_zz,
  const gpufftComplex* g_mesh_virial_xy,
  const gpufftComplex* g_mesh_virial_yz,
  const gpufftComplex* g_mesh_virial_zx,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = K_C_SP * g_charge[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f); // can be 0, ..., K[0]
    const int iy = int(sy + 0.5f); // can be 0, ..., K[1]
    const int iz = int(sz + 0.5f); // can be 0, ..., K[2]
    const float dx = sx - ix; // (-0.5, 0.5)
    const float dy = sy - iy; // (-0.5, 0.5)
    const float dz = sz - iz; // (-0.5, 0.5)
    // Appendix E in M. Deserno and C. Holm, JCP 109, 7678 (1998)
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    float D_real = 0.0f;
    float E[3] = {0.0f, 0.0f, 0.0f};
    float V[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D_real += W * g_mesh_G[neighbor012].x;
          E[0] += W * g_mesh_fft_x_ifft[neighbor012].x;
          E[1] += W * g_mesh_fft_y_ifft[neighbor012].x;
          E[2] += W * g_mesh_fft_z_ifft[neighbor012].x;
          V[0] += W * g_mesh_virial_xx[neighbor012].x;
          V[1] += W * g_mesh_virial_yy[neighbor012].x;
          V[2] += W * g_mesh_virial_zz[neighbor012].x;
          V[3] += W * g_mesh_virial_xy[neighbor012].x;
          V[4] += W * g_mesh_virial_yz[neighbor012].x;
          V[5] += W * g_mesh_virial_zx[neighbor012].x;
        }
      }
    }
    g_D_real[n] = 2.0f * K_C_SP * D_real;
    g_fx[n] += 2.0f * q * E[0];
    g_fy[n] += 2.0f * q * E[1];
    g_fz[n] += 2.0f * q * E[2];
    // virial order
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n + 0 * N] += q * V[0]; // xx
    g_virial[n + 1 * N] += q * V[1]; // yy
    g_virial[n + 2 * N] += q * V[2]; // zz
    g_virial[n + 3 * N] += q * V[3]; // xy
    g_virial[n + 6 * N] += q * V[3]; // yx
    g_virial[n + 5 * N] += q * V[4]; // yz
    g_virial[n + 8 * N] += q * V[4]; // zy
    g_virial[n + 4 * N] += q * V[5]; // xz
    g_virial[n + 7 * N] += q * V[5]; // zx
    g_pe[n] += q * D_real;
  } 
}

void __global__ find_potential_and_virial(
  const int N,
  const PPPM::Para para,
  const gpufftComplex* g_S,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  double* g_virial,
  double* g_pe)
{
  const int tid = threadIdx.x;
  int number_of_batches = (para.K0K1K2 - 1) / 1024 + 1;
  __shared__ float s_data[1024];
  float data = 0.0f;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < para.K0K1K2) {
      gpufftComplex S = g_S[n];
      const float GSS = g_G[n] * (S.x * S.x + S.y * S.y);
      const float kx = g_kx[n];
      const float ky = g_ky[n];
      const float kz = g_kz[n];
      const float ksq = kx * kx + ky * ky + kz * kz;
      if (ksq != 0.0f) {
        const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
        switch (blockIdx.x) {
          case 0:
          data += GSS * (1.0f - alpha_k_factor * kx * kx); // xx
          break;
          case 1:
          data += GSS * (1.0f - alpha_k_factor * ky * ky); // yy
          break;
          case 2:
          data += GSS * (1.0f - alpha_k_factor * kz * kz); // zz
          break;
          case 3:
          data -= GSS * (alpha_k_factor * kx * ky); // xy
          break;
          case 4:
          data -= GSS * (alpha_k_factor * ky * kz); // yz
          break;
          case 5:
          data -= GSS * (alpha_k_factor * kz * kx); // zx
          break;
          case 6:
          data += GSS; // potential
          break;
        }
      }
    }
  }
  s_data[tid] = data;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  number_of_batches = (N - 1) / 1024 + 1;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      // virial order
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      switch (blockIdx.x) {
        case 0:
          g_virial[n + 0 * N] += s_data[0] * para.potential_factor; // xx
          break;
        case 1:
          g_virial[n + 1 * N] += s_data[0] * para.potential_factor; // yy
          break;
        case 2:
          g_virial[n + 2 * N] += s_data[0] * para.potential_factor; // zz
          break;
        case 3:
          g_virial[n + 3 * N] += s_data[0] * para.potential_factor; // xy
          g_virial[n + 6 * N] += s_data[0] * para.potential_factor; // yx
          break;
        case 4:
          g_virial[n + 5 * N] += s_data[0] * para.potential_factor; // yz
          g_virial[n + 8 * N] += s_data[0] * para.potential_factor; // zy
          break;
        case 5:
          g_virial[n + 4 * N] += s_data[0] * para.potential_factor; // xz
          g_virial[n + 7 * N] += s_data[0] * para.potential_factor; // zx
          break;
        case 6:
          g_pe[n] += s_data[0] * para.potential_factor;
          break;
      }
    }
  }
}

void __global__ set_mesh_to_zero_batch(
  const PPPM::Para para,
  const int number_of_beads,
  gpufftComplex* g_mesh)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (bead < number_of_beads && n < para.K0K1K2) {
    g_mesh[static_cast<size_t>(bead) * para.K0K1K2 + n] = {0.0f, 0.0f};
  }
}

__global__ void find_mesh_batch(
  const int N,
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  float* const* g_charge,
  double* const* g_position,
  gpufftComplex* g_mesh)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (bead >= gridDim.y || n >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const float* charge = g_charge[bead];
  const double x = position[n];
  const double y = position[n + N];
  const double z = position[n + N * 2];
  const float q = charge[n];
  const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
  const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
  const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
  const int ix = int(sx + 0.5f);
  const int iy = int(sy + 0.5f);
  const int iz = int(sz + 0.5f);
  const float dx = sx - ix;
  const float dy = sy - iy;
  const float dz = sz - iz;
  float Wx[5] = {0.0f};
  float Wy[5] = {0.0f};
  float Wz[5] = {0.0f};
  for (int d = 0; d < 5; ++d) {
    Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
    Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
    Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
  }
  gpufftComplex* mesh = g_mesh + static_cast<size_t>(bead) * para.K0K1K2;
  for (int n0 = -2; n0 <= 2; ++n0) {
    const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);
    for (int n1 = -2; n1 <= 2; ++n1) {
      const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);
      for (int n2 = -2; n2 <= 2; ++n2) {
        const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);
        const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
        atomicAdd(&mesh[neighbor012].x, q * Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2]);
      }
    }
  }
}

void __global__ prepare_inverse_fields_batch(
  const PPPM::Para para,
  const int number_of_beads,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const gpufftComplex* g_mesh,
  gpufftComplex* g_mesh_inverse)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (bead < number_of_beads && n < para.K0K1K2) {
    const size_t offset = static_cast<size_t>(bead) * para.K0K1K2 + n;
    const size_t field_stride = static_cast<size_t>(number_of_beads) * para.K0K1K2;
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float G = g_G[n];
    const gpufftComplex mesh = g_mesh[offset];
    g_mesh_inverse[offset] = {mesh.x * G, mesh.y * G};
    g_mesh_inverse[field_stride + offset] = {mesh.y * kx * G, -mesh.x * kx * G};
    g_mesh_inverse[2 * field_stride + offset] = {mesh.y * ky * G, -mesh.x * ky * G};
    g_mesh_inverse[3 * field_stride + offset] = {mesh.y * kz * G, -mesh.x * kz * G};
  }
}

void __global__ find_mesh_virial_batch(
  const PPPM::Para para,
  const int number_of_beads,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const gpufftComplex* g_S,
  gpufftComplex* g_mesh_virial)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (bead < number_of_beads && n < para.K0K1K2) {
    const size_t mesh_offset = static_cast<size_t>(bead) * para.K0K1K2;
    const size_t offset = static_cast<size_t>(bead) * 6 * para.K0K1K2 + n;
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float ksq = kx * kx + ky * ky + kz * kz;
    if (ksq == 0.0f) {
      for (int component = 0; component < 6; ++component) {
        g_mesh_virial[offset + static_cast<size_t>(component) * para.K0K1K2] = {0.0f, 0.0f};
      }
      return;
    }
    const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
    const float G = g_G[n];
    const gpufftComplex S = g_S[mesh_offset + n];
    const float GSx = G * S.x;
    const float GSy = G * S.y;
    float B = 1.0f - alpha_k_factor * kx * kx;
    g_mesh_virial[offset + static_cast<size_t>(0) * para.K0K1K2] = {B * GSx, B * GSy};
    B = 1.0f - alpha_k_factor * ky * ky;
    g_mesh_virial[offset + static_cast<size_t>(1) * para.K0K1K2] = {B * GSx, B * GSy};
    B = 1.0f - alpha_k_factor * kz * kz;
    g_mesh_virial[offset + static_cast<size_t>(2) * para.K0K1K2] = {B * GSx, B * GSy};
    B = -alpha_k_factor * kx * ky;
    g_mesh_virial[offset + static_cast<size_t>(3) * para.K0K1K2] = {B * GSx, B * GSy};
    B = -alpha_k_factor * ky * kz;
    g_mesh_virial[offset + static_cast<size_t>(4) * para.K0K1K2] = {B * GSx, B * GSy};
    B = -alpha_k_factor * kz * kx;
    g_mesh_virial[offset + static_cast<size_t>(5) * para.K0K1K2] = {B * GSx, B * GSy};
  }
}

__global__ void find_force_from_field_batch(
  const int N,
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  float* const* g_charge,
  double* const* g_position,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  float* const* g_D_real,
  double* const* g_force,
  const int number_of_beads)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (bead >= number_of_beads || n >= N2) {
    return;
  }
  const double* position = g_position[bead];
  double* force = g_force[bead];
  const float* charge = g_charge[bead];
  const size_t mesh_offset = static_cast<size_t>(bead) * para.K0K1K2;
  const double x = position[n];
  const double y = position[n + N];
  const double z = position[n + N * 2];
  const float q = K_C_SP * charge[n] * 2.0f;
  const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
  const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
  const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
  const int ix = int(sx + 0.5f);
  const int iy = int(sy + 0.5f);
  const int iz = int(sz + 0.5f);
  const float dx = sx - ix;
  const float dy = sy - iy;
  const float dz = sz - iz;
  float Wx[5] = {0.0f};
  float Wy[5] = {0.0f};
  float Wz[5] = {0.0f};
  for (int d = 0; d < 5; ++d) {
    Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
    Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
    Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
  }
  float D_real = 0.0f;
  float E[3] = {0.0f, 0.0f, 0.0f};
  for (int n0 = -2; n0 <= 2; ++n0) {
    const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);
    for (int n1 = -2; n1 <= 2; ++n1) {
      const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);
      for (int n2 = -2; n2 <= 2; ++n2) {
        const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);
        const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
        const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
        const size_t mesh_index = mesh_offset + neighbor012;
        D_real += W * g_mesh_G[mesh_index].x;
        E[0] += W * g_mesh_fft_x_ifft[mesh_index].x;
        E[1] += W * g_mesh_fft_y_ifft[mesh_index].x;
        E[2] += W * g_mesh_fft_z_ifft[mesh_index].x;
      }
    }
  }
  g_D_real[bead][n] = 2.0f * K_C_SP * D_real;
  force[ n] += q * E[0];
  force[n + N] += q * E[1];
  force[n + N * 2] += q * E[2];
}

__global__ void find_force_virial_potential_from_field_batch(
  const int N,
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  float* const* g_charge,
  double* const* g_position,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  const gpufftComplex* g_mesh_virial,
  float* const* g_D_real,
  double* const* g_force,
  double* const* g_virial,
  double* const* g_pe,
  const int number_of_beads)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (bead >= number_of_beads || n >= N2) {
    return;
  }
  const double* position = g_position[bead];
  double* force = g_force[bead];
  double* virial = g_virial[bead];
  double* pe = g_pe[bead];
  const float* charge = g_charge[bead];
  const size_t mesh_offset = static_cast<size_t>(bead) * para.K0K1K2;
  const size_t virial_offset = static_cast<size_t>(bead) * 6 * para.K0K1K2;
  const double x = position[n];
  const double y = position[n + N];
  const double z = position[n + N * 2];
  const float q = K_C_SP * charge[n];
  const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
  const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
  const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
  const int ix = int(sx + 0.5f);
  const int iy = int(sy + 0.5f);
  const int iz = int(sz + 0.5f);
  const float dx = sx - ix;
  const float dy = sy - iy;
  const float dz = sz - iz;
  float Wx[5] = {0.0f};
  float Wy[5] = {0.0f};
  float Wz[5] = {0.0f};
  for (int d = 0; d < 5; ++d) {
    Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
    Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
    Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
  }
  float D_real = 0.0f;
  float E[3] = {0.0f, 0.0f, 0.0f};
  float V[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int n0 = -2; n0 <= 2; ++n0) {
    const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);
    for (int n1 = -2; n1 <= 2; ++n1) {
      const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);
      for (int n2 = -2; n2 <= 2; ++n2) {
        const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);
        const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
        const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
        const size_t mesh_index = mesh_offset + neighbor012;
        D_real += W * g_mesh_G[mesh_index].x;
        E[0] += W * g_mesh_fft_x_ifft[mesh_index].x;
        E[1] += W * g_mesh_fft_y_ifft[mesh_index].x;
        E[2] += W * g_mesh_fft_z_ifft[mesh_index].x;
        V[0] += W * g_mesh_virial[virial_offset + neighbor012].x;
        V[1] += W * g_mesh_virial[virial_offset + para.K0K1K2 + neighbor012].x;
        V[2] += W * g_mesh_virial[virial_offset + 2 * para.K0K1K2 + neighbor012].x;
        V[3] += W * g_mesh_virial[virial_offset + 3 * para.K0K1K2 + neighbor012].x;
        V[4] += W * g_mesh_virial[virial_offset + 4 * para.K0K1K2 + neighbor012].x;
        V[5] += W * g_mesh_virial[virial_offset + 5 * para.K0K1K2 + neighbor012].x;
      }
    }
  }
  g_D_real[bead][n] = 2.0f * K_C_SP * D_real;
  force[n] += 2.0f * q * E[0];
  force[n + N] += 2.0f * q * E[1];
  force[n + N * 2] += 2.0f * q * E[2];
  virial[n + 0 * N] += q * V[0];
  virial[n + 1 * N] += q * V[1];
  virial[n + 2 * N] += q * V[2];
  virial[n + 3 * N] += q * V[3];
  virial[n + 6 * N] += q * V[3];
  virial[n + 5 * N] += q * V[4];
  virial[n + 8 * N] += q * V[4];
  virial[n + 4 * N] += q * V[5];
  virial[n + 7 * N] += q * V[5];
  pe[n] += q * D_real;
}

void __global__ find_potential_and_virial_batch(
  const int N,
  const PPPM::Para para,
  const int number_of_beads,
  const gpufftComplex* g_S,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  double* const* g_virial,
  double* const* g_pe)
{
  const int bead = blockIdx.y;
  const int tid = threadIdx.x;
  if (bead >= number_of_beads) {
    return;
  }
  const size_t mesh_offset = static_cast<size_t>(bead) * para.K0K1K2;
  int number_of_batches = (para.K0K1K2 - 1) / 1024 + 1;
  __shared__ float s_data[1024];
  float data = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < para.K0K1K2) {
      const gpufftComplex S = g_S[mesh_offset + n];
      const float GSS = g_G[n] * (S.x * S.x + S.y * S.y);
      const float kx = g_kx[n];
      const float ky = g_ky[n];
      const float kz = g_kz[n];
      const float ksq = kx * kx + ky * ky + kz * kz;
      if (ksq != 0.0f) {
        const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
        switch (blockIdx.x) {
          case 0: data += GSS * (1.0f - alpha_k_factor * kx * kx); break;
          case 1: data += GSS * (1.0f - alpha_k_factor * ky * ky); break;
          case 2: data += GSS * (1.0f - alpha_k_factor * kz * kz); break;
          case 3: data -= GSS * (alpha_k_factor * kx * ky); break;
          case 4: data -= GSS * (alpha_k_factor * ky * kz); break;
          case 5: data -= GSS * (alpha_k_factor * kz * kx); break;
          case 6: data += GSS; break;
        }
      }
    }
  }
  s_data[tid] = data;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  number_of_batches = (N - 1) / 1024 + 1;
  double* virial = g_virial[bead];
  double* pe = g_pe[bead];
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      switch (blockIdx.x) {
        case 0: virial[n + 0 * N] += s_data[0] * para.potential_factor; break;
        case 1: virial[n + 1 * N] += s_data[0] * para.potential_factor; break;
        case 2: virial[n + 2 * N] += s_data[0] * para.potential_factor; break;
        case 3:
          virial[n + 3 * N] += s_data[0] * para.potential_factor;
          virial[n + 6 * N] += s_data[0] * para.potential_factor;
          break;
        case 4:
          virial[n + 5 * N] += s_data[0] * para.potential_factor;
          virial[n + 8 * N] += s_data[0] * para.potential_factor;
          break;
        case 5:
          virial[n + 4 * N] += s_data[0] * para.potential_factor;
          virial[n + 7 * N] += s_data[0] * para.potential_factor;
          break;
        case 6: pe[n] += s_data[0] * para.potential_factor; break;
      }
    }
  }
}

}

PPPM::PPPM()
{
  // nothing
}

PPPM::~PPPM()
{
  flush_dynamic_charge_diagnostics();
  destroy_plans();
}

void PPPM::destroy_plans()
{
  if (plan_initialized) gpufftDestroy(plan);
  if (plan_virial_initialized) gpufftDestroy(plan_virial);
  if (plan_batch_initialized_) gpufftDestroy(plan_batch);
  if (plan_inverse_batch_initialized_) gpufftDestroy(plan_inverse_batch);
  if (plan_virial_batch_initialized_) gpufftDestroy(plan_virial_batch);
  plan = 0;
  plan_virial = 0;
  plan_batch = 0;
  plan_inverse_batch = 0;
  plan_virial_batch = 0;
  plan_initialized = false;
  plan_virial_initialized = false;
  plan_batch_initialized_ = false;
  plan_inverse_batch_initialized_ = false;
  plan_virial_batch_initialized_ = false;
  batch_capacity = 0;
}

void PPPM::flush_dynamic_charge_diagnostics()
{
  if (!dynamic_diagnostics_enabled_) return;
  if (dynamic_csv_row_pending_) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double invalid[3] = {nan, nan, nan};
    finalize_dynamic_charge_diagnostic(invalid, invalid, false, -1);
  }
  const std::string csv_rows = dynamic_csv_buffer_.str();
  std::ostringstream csv_header;
  write_dynamic_metadata(csv_header);
  csv_header << PPPM_DYNAMIC_DIAG_COLUMN_HEADER << "\n";
  if (!append_text_file(
        "pppm_dynamic_q_diag.csv",
        csv_header.str(),
        csv_rows,
        {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
         pppm_dynamic_formula_marker(),
         PPPM_DYNAMIC_DIAG_COLUMN_HEADER})) {
    std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_diag.csv; "
                 "the existing file may have an incompatible schema."
              << std::endl;
  } else if (!csv_rows.empty()) {
    dynamic_csv_buffer_.str("");
    dynamic_csv_buffer_.clear();
  }

  const std::string check_rows = dynamic_check_buffer_.str();
  std::ostringstream check_header;
  check_header << "# PPPM dynamic-q runtime checks; written after the run\n"
               << "# source_signature = " << PPPM_DYNAMIC_SOURCE_SIGNATURE << "\n"
               << PPPM_DYNAMIC_DIAG_SCHEMA_MARKER << "\n"
               << pppm_dynamic_formula_marker() << "\n"
               << "# units: mesh_path=eV*Angstrom/fs; DeltaJ_gpu_cpu_error=eV*Angstrom/fs; "
                  "charge_sum=e; qdot_sum=e/natural_time\n"
               << PPPM_DYNAMIC_CHECK_COLUMN_HEADER << "\n";
  if (!append_text_file(
        "pppm_dynamic_q_check.out",
        check_header.str(),
        check_rows,
        {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
         pppm_dynamic_formula_marker(),
         PPPM_DYNAMIC_CHECK_COLUMN_HEADER})) {
    std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_check.out." << std::endl;
  } else if (!check_rows.empty()) {
    dynamic_check_buffer_.str("");
    dynamic_check_buffer_.clear();
  }

  std::ostringstream atom_header;
  write_dynamic_metadata(atom_header);
  atom_header << PPPM_DYNAMIC_ATOM_COLUMN_HEADER << "\n";
  if (!append_text_file(
        "pppm_dynamic_q_atom_debug.out",
        atom_header.str(),
        dynamic_atom_debug_buffer_,
        {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
         pppm_dynamic_formula_marker(),
         PPPM_DYNAMIC_ATOM_COLUMN_HEADER})) {
    if (!dynamic_atom_debug_buffer_.empty()) {
      std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_atom_debug.out."
                << std::endl;
    }
  } else {
    dynamic_atom_debug_buffer_.clear();
  }

  std::ostringstream kspace_header;
  write_dynamic_metadata(kspace_header);
  kspace_header << PPPM_DYNAMIC_KSPACE_COLUMN_HEADER << "\n";
  if (!append_text_file(
        "pppm_dynamic_q_kspace_debug.out",
        kspace_header.str(),
        dynamic_kspace_debug_buffer_,
        {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
         pppm_dynamic_formula_marker(),
         PPPM_DYNAMIC_KSPACE_COLUMN_HEADER})) {
    if (!dynamic_kspace_debug_buffer_.empty()) {
      std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_kspace_debug.out."
                << std::endl;
    }
  } else {
    dynamic_kspace_debug_buffer_.clear();
  }
}

bool PPPM::dynamic_charge_diagnostic_files_are_compatible(const bool check_debug_files) const
{
  if (!existing_file_has_schema(
        "pppm_dynamic_q_diag.csv",
        {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
         pppm_dynamic_formula_marker(),
         PPPM_DYNAMIC_DIAG_COLUMN_HEADER}))
    return false;
  if (!existing_file_has_schema(
        "pppm_dynamic_q_check.out",
        {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
         pppm_dynamic_formula_marker(),
         PPPM_DYNAMIC_CHECK_COLUMN_HEADER}))
    return false;
  if (check_debug_files &&
      (!existing_file_has_schema(
         "pppm_dynamic_q_atom_debug.out",
         {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
          pppm_dynamic_formula_marker(),
          PPPM_DYNAMIC_ATOM_COLUMN_HEADER}) ||
       !existing_file_has_schema(
         "pppm_dynamic_q_kspace_debug.out",
         {PPPM_DYNAMIC_DIAG_SCHEMA_MARKER,
          pppm_dynamic_formula_marker(),
          PPPM_DYNAMIC_KSPACE_COLUMN_HEADER})))
    return false;
  return true;
}

void PPPM::finalize_dynamic_charge_diagnostic(
  const double* delta_j_q_real,
  const double* delta_j_q_total,
  const bool dynamic_q_valid,
  const int charge_mode)
{
  if (!dynamic_csv_row_pending_) return;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  bool row_valid = dynamic_q_valid && delta_j_q_real != nullptr && delta_j_q_total != nullptr;
  if (row_valid) {
    for (int d = 0; d < 3; ++d) {
      row_valid = row_valid && std::isfinite(delta_j_q_real[d]) && std::isfinite(delta_j_q_total[d]);
    }
  }

  dynamic_csv_buffer_ << std::scientific << std::setprecision(16) << dynamic_csv_row_;
  const double* real = row_valid ? delta_j_q_real : nullptr;
  const double* total = row_valid ? delta_j_q_total : nullptr;
  for (int d = 0; d < 3; ++d)
    dynamic_csv_buffer_ << "," << (real == nullptr ? nan : real[d] / TIME_UNIT_CONVERSION);
  for (int d = 0; d < 3; ++d)
    dynamic_csv_buffer_ << "," << (total == nullptr ? nan : total[d] / TIME_UNIT_CONVERSION);
  dynamic_csv_buffer_ << "," << (row_valid ? 1 : 0) << "," << charge_mode << "\n";
  dynamic_csv_row_.clear();
  dynamic_csv_row_pending_ = false;
}

void PPPM::write_debug(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position,
  const GPU_Vector<float>& D_real,
  const int pppm_call_index)
{
  const int M = para.K0K1K2;
  std::vector<gpufftComplex> h_mesh_charge(M), h_mesh(M), h_mesh_fourier(M), h_mesh_G(M);
  std::vector<float> h_kx(M), h_ky(M), h_kz(M), h_G(M);
  std::vector<float> h_charge(N), h_D_real(N);
  std::vector<double> h_position(3 * N);
  std::vector<PPPMAssignmentAtomDebug> h_assignment_atoms(8);
  std::vector<PPPMAssignmentStencilDebug> h_assignment_stencil(125);

  debug_mesh_charge_.copy_to_host(h_mesh_charge.data());
  mesh.copy_to_host(h_mesh.data());
  debug_mesh_fourier_.copy_to_host(h_mesh_fourier.data());
  mesh_G.copy_to_host(h_mesh_G.data());
  kx.copy_to_host(h_kx.data());
  ky.copy_to_host(h_ky.data());
  kz.copy_to_host(h_kz.data());
  G.copy_to_host(h_G.data());
  charge.copy_to_host(h_charge.data(), N);
  position.copy_to_host(h_position.data(), 3 * N);
  D_real.copy_to_host(h_D_real.data(), N);
  debug_assignment_atoms_.copy_to_host(h_assignment_atoms.data(), 8);
  debug_assignment_stencil_.copy_to_host(h_assignment_stencil.data(), 125);
  double sum_charge_N1_N2 = 0.0;
  for (int n = N1; n < N2; ++n) {
    sum_charge_N1_N2 += static_cast<double>(h_charge[n]);
  }

  std::ofstream mesh_file(debug_prefix_ + "_mesh.out", std::ios::app);
  std::ofstream kspace_file(debug_prefix_ + "_kspace.out", std::ios::app);
  std::ofstream atom_file(debug_prefix_ + "_atom.out", std::ios::app);
  std::ofstream assignment_atom_file(debug_prefix_ + "_assignment_atom.out", std::ios::app);
  std::ofstream assignment_stencil_file(
    debug_prefix_ + "_assignment_stencil_atom0.out", std::ios::app);
  if (!mesh_file || !kspace_file || !atom_file || !assignment_atom_file || !assignment_stencil_file) {
    std::cerr << "Cannot open PPPM debug output files with prefix " << debug_prefix_ << ".\n";
    exit(1);
  }
  mesh_file << std::setprecision(17);
  kspace_file << std::setprecision(17);
  atom_file << std::setprecision(17);
  assignment_atom_file << std::setprecision(17);
  assignment_stencil_file << std::setprecision(17);

  auto write_header = [&](std::ofstream& file) {
    file << "# pppm_frame " << debug_frame_ << "\n";
    file << "# source_signature " << PPPM_DEBUG_SOURCE_SIGNATURE << "\n";
    file << "# pppm_N " << N << "\n";
    file << "# pppm_N1 " << N1 << "\n";
    file << "# pppm_N2 " << N2 << "\n";
    file << "# pppm_call_index " << pppm_call_index << "\n";
    file << "# sum_charge_N1_N2 " << sum_charge_N1_N2 << "\n";
    file << "# mesh_before_assignment_max_real "
         << debug_mesh_before_assignment_max_real_ << "\n";
    file << "# mesh_before_assignment_max_imag "
         << debug_mesh_before_assignment_max_imag_ << "\n";
    file << "# mesh_before_assignment_rms_real "
         << debug_mesh_before_assignment_rms_real_ << "\n";
    file << "# mesh_before_assignment_rms_imag "
         << debug_mesh_before_assignment_rms_imag_ << "\n";
    file << "# mesh_before_assignment_sum_real "
         << debug_mesh_before_assignment_sum_real_ << "\n";
    file << "# mesh_after_assignment_sum_real "
         << debug_mesh_after_assignment_sum_real_ << "\n";
    file << "# pppm_number_of_atoms " << N << "\n";
    file << "# pppm_mesh_size " << para.K[0] << " " << para.K[1] << " " << para.K[2] << "\n";
    file << "# pppm_indexing gx-fastest gy-middle gz-slowest\n";
    file << "# K_C_SP " << K_C_SP << " potential_factor " << para.potential_factor << "\n";
    file << "# box_h";
    for (int d = 0; d < 9; ++d) file << " " << box.cpu_h[d];
    file << "\n";
    file << "# box_h_inverse";
    for (int d = 0; d < 9; ++d) file << " " << box.cpu_h[9 + d];
    file << "\n";
  };

  write_header(mesh_file);
  mesh_file << "# fft_forward unnormalized fft_inverse unnormalized\n";
  mesh_file << "# mesh_charge=Q_g=sum_i q_i W_gi (not a density)\n";
  mesh_file << "# mesh_potential=IFFT_unnormalized(Gopt*S)\n";
  mesh_file << "# columns gx gy gz mesh_charge_real mesh_charge_imag mesh_potential_real mesh_potential_imag\n";
  for (int iz = 0; iz < para.K[2]; ++iz) {
    for (int iy = 0; iy < para.K[1]; ++iy) {
      for (int ix = 0; ix < para.K[0]; ++ix) {
        const int n = ix + para.K[0] * (iy + para.K[1] * iz);
        mesh_file << ix << " " << iy << " " << iz << " " << h_mesh_charge[n].x << " "
                  << h_mesh_charge[n].y << " " << h_mesh_G[n].x << " " << h_mesh_G[n].y << "\n";
      }
    }
  }

  write_header(kspace_file);
  kspace_file << "# zero_mode Gopt=0\n";
  kspace_file << "# columns ix iy iz nx ny nz kx ky kz S_real S_imag Gopt Phi_real Phi_imag\n";
  for (int iz = 0; iz < para.K[2]; ++iz) {
    for (int iy = 0; iy < para.K[1]; ++iy) {
      for (int ix = 0; ix < para.K[0]; ++ix) {
        const int n = ix + para.K[0] * (iy + para.K[1] * iz);
        const int nx = ix >= para.K_half[0] ? ix - para.K[0] : ix;
        const int ny = iy >= para.K_half[1] ? iy - para.K[1] : iy;
        const int nz = iz >= para.K_half[2] ? iz - para.K[2] : iz;
        kspace_file << ix << " " << iy << " " << iz << " " << nx << " " << ny << " " << nz
                    << " " << h_kx[n] << " " << h_ky[n] << " " << h_kz[n] << " " << h_mesh[n].x
                    << " " << h_mesh[n].y << " " << h_G[n] << " " << h_mesh_fourier[n].x << " "
                    << h_mesh_fourier[n].y << "\n";
      }
    }
  }

  write_header(atom_file);
  atom_file << "# pppm_atom_range " << N1 << " " << N2 << "\n";
  atom_file << "# D_recip=2*K_C_SP*T^T*mesh_potential before real-space and zero-mean projection; "
               "u_recip=0.5*q*D_recip\n";
  atom_file << "# columns atom_id x y z charge D_recip u_recip\n";
  for (int n = 0; n < N; ++n) {
    const double u_recip = 0.5 * double(h_charge[n]) * double(h_D_real[n]);
    atom_file << n << " " << h_position[n] << " " << h_position[N + n] << " " << h_position[2 * N + n]
               << " " << h_charge[n] << " " << h_D_real[n] << " " << u_recip << "\n";
  }

  write_header(assignment_atom_file);
  assignment_atom_file << "# columns atom_id q x y z sx sy sz ix iy iz dx dy dz"
                          " Wx0 Wx1 Wx2 Wx3 Wx4 Wy0 Wy1 Wy2 Wy3 Wy4 Wz0 Wz1 Wz2 Wz3 Wz4\n";
  const int debug_atom_count = N2 - N1 < 8 ? N2 - N1 : 8;
  for (int i = 0; i < debug_atom_count; ++i) {
    const PPPMAssignmentAtomDebug& d = h_assignment_atoms[i];
    assignment_atom_file << d.atom_id << " " << d.q << " " << d.x << " " << d.y << " " << d.z
                         << " " << d.sx << " " << d.sy << " " << d.sz << " " << d.ix << " "
                         << d.iy << " " << d.iz << " " << d.dx << " " << d.dy << " " << d.dz;
    for (int j = 0; j < 5; ++j) assignment_atom_file << " " << d.Wx[j];
    for (int j = 0; j < 5; ++j) assignment_atom_file << " " << d.Wy[j];
    for (int j = 0; j < 5; ++j) assignment_atom_file << " " << d.Wz[j];
    assignment_atom_file << "\n";
  }

  write_header(assignment_stencil_file);
  assignment_stencil_file << "# pppm_stencil_atom 0\n";
  assignment_stencil_file << "# columns n0 n1 n2 neighbor0 neighbor1 neighbor2 neighbor012 W qW\n";
  if (N1 <= 0 && 0 < N2) {
    for (int i = 0; i < 125; ++i) {
      const PPPMAssignmentStencilDebug& d = h_assignment_stencil[i];
      assignment_stencil_file << d.n0 << " " << d.n1 << " " << d.n2 << " " << d.neighbor0
                              << " " << d.neighbor1 << " " << d.neighbor2 << " " << d.neighbor012
                              << " " << d.W << " " << d.qW << "\n";
    }
  }
}

void PPPM::allocate_virial_memory()
{
  if (plan_virial_initialized) return;
  mesh_virial.resize(para.K0K1K2 * 6);
  int n[3] = {para.K[2], para.K[1], para.K[0]};
  if (gpufftPlanMany(
        &plan_virial,
        3,
        n,
        NULL,
        1,
        para.K0K1K2,
        NULL,
        1,
        para.K0K1K2,
        GPUFFT_C2C,
        6) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: plan_virial creation failed" << std::endl;
    exit(1);
  }
  plan_virial_initialized = true;
}

void PPPM::allocate_memory()
{
  destroy_plans();
  kx.resize(para.K0K1K2);
  ky.resize(para.K0K1K2);
  kz.resize(para.K0K1K2);
  G.resize(para.K0K1K2);
  mesh.resize(para.K0K1K2);
  mesh_G.resize(para.K0K1K2);
  mesh_x.resize(para.K0K1K2);
  mesh_y.resize(para.K0K1K2);
  mesh_z.resize(para.K0K1K2);
  // para.K[2] is the slowest changing dimension; para.K[0] is the fastest changing dimension
  if (gpufftPlan3d(&plan, para.K[2], para.K[1], para.K[0], GPUFFT_C2C) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: Plan creation failed" << std::endl;
    exit(1);
  }
  plan_initialized = true;
}

void PPPM::allocate_batch_memory(const int number_of_beads)
{
  if (number_of_beads == batch_capacity && plan_batch_initialized_ &&
      plan_inverse_batch_initialized_ &&
      (!need_peratom_virial || plan_virial_batch_initialized_)) {
    return;
  }
  if (plan_batch_initialized_) gpufftDestroy(plan_batch);
  if (plan_virial_batch_initialized_) gpufftDestroy(plan_virial_batch);
  if (plan_inverse_batch_initialized_) gpufftDestroy(plan_inverse_batch);
  plan_batch = 0;
  plan_virial_batch = 0;
  plan_inverse_batch = 0;
  plan_batch_initialized_ = false;
  plan_virial_batch_initialized_ = false;
  plan_inverse_batch_initialized_ = false;
  batch_capacity = 0;
  batch_capacity = number_of_beads;
  const size_t batch_mesh_size = static_cast<size_t>(number_of_beads) * para.K0K1K2;
  mesh_batch.resize(batch_mesh_size);
  mesh_inverse_batch.resize(batch_mesh_size * 4);
  int n[3] = {para.K[2], para.K[1], para.K[0]};
  if (gpufftPlanMany(
        &plan_batch,
        3,
        n,
        NULL,
        1,
        para.K0K1K2,
        NULL,
        1,
        para.K0K1K2,
        GPUFFT_C2C,
        number_of_beads) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: plan_batch creation failed" << std::endl;
    exit(1);
  }
  plan_batch_initialized_ = true;
  if (gpufftPlanMany(
        &plan_inverse_batch,
        3,
        n,
        NULL,
        1,
        para.K0K1K2,
        NULL,
        1,
        para.K0K1K2,
        GPUFFT_C2C,
        number_of_beads * 4) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: plan_inverse_batch creation failed" << std::endl;
    exit(1);
  }
  plan_inverse_batch_initialized_ = true;
  if (need_peratom_virial) {
    mesh_virial_batch.resize(static_cast<size_t>(number_of_beads) * 6 * para.K0K1K2);
    if (gpufftPlanMany(
          &plan_virial_batch,
          3,
          n,
          NULL,
          1,
          para.K0K1K2,
          NULL,
          1,
          para.K0K1K2,
          GPUFFT_C2C,
          number_of_beads * 6) != GPUFFT_SUCCESS) {
      std::cout << "GPUFFT error: plan_virial_batch creation failed" << std::endl;
      exit(1);
    }
    plan_virial_batch_initialized_ = true;
  }
}

void PPPM::initialize(
  const float alpha_input,
  const bool need_peratom_virial_input,
  const bool need_peratom_virial_every_batch_input,
  const double mesh_spacing_input)
{
  destroy_plans();
  para = {};
  mesh_spacing = mesh_spacing_input;
  current_force_mesh_valid_ = false;
  current_force_mesh_peratom_ = false;
  dynamic_operator_cache_valid_ = false;
  dynamic_operator_host_cache_valid_ = false;
  need_peratom_virial = need_peratom_virial_input;
  need_peratom_virial_every_batch = need_peratom_virial_every_batch_input;
  para.alpha = alpha_input;
  para.alpha_factor = 0.25f / (para.alpha * para.alpha);
  para.K[0] = 16;
  para.K[1] = 16;
  para.K[2] = 16;
}

void PPPM::find_para(const int N, const Box& box)
{
  const float two_pi = 6.2831853f;
  const double volume = box.get_volume();
  para.two_pi_over_V = two_pi / volume;
  for (int d = 0; d < 3; ++d) {
    const double required = volume / box.get_area(d) / mesh_spacing;
    if (required > para.K[d]) para.K[d] = get_best_K(required);
  }
  const double number_of_points =
    static_cast<double>(para.K[0]) * para.K[1] * para.K[2];
  if (number_of_points > max_mesh_points) {
    PRINT_INPUT_ERROR("PPPM mesh is too large; increase spacing or reduce the box size.");
  }
  const bool first_mesh = !plan_initialized;
  for (int d = 0; d < 3; ++d) {
    para.K_half[d] = para.K[d] / 2;
    para.two_pi_over_K[d] = two_pi / para.K[d];
  }
  const double old_number_of_points = para.K0K1K2;
  if (number_of_points != old_number_of_points) {
    para.K0K1 = para.K[0] * para.K[1];
    para.K0K1K2 = static_cast<int>(number_of_points);
    current_force_mesh_valid_ = false;
    allocate_memory();
  }
  if (first_mesh) {
    printf(
      "PPPM mesh: %d x %d x %d (target spacing %.17g A; actual spacing %.17g %.17g %.17g A).\n",
      para.K[0], para.K[1], para.K[2], mesh_spacing,
      volume / box.get_area(0) / para.K[0],
      volume / box.get_area(1) / para.K[1],
      volume / box.get_area(2) / para.K[2]);
  }
  para.potential_factor = K_C_SP / N;
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * (float)box.cpu_h[9 + d];
    para.b[1][d] = two_pi * (float)box.cpu_h[12 + d];
    para.b[2][d] = two_pi * (float)box.cpu_h[15 + d];
  }
}

bool PPPM::current_force_mesh_matches(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position,
  const unsigned long long force_evaluation_id,
  const bool require_orthogonal) const
{
  if (
    !current_force_mesh_valid_ || force_evaluation_id == 0 ||
    current_force_mesh_force_evaluation_id_ != force_evaluation_id ||
    N1 != 0 || N2 != N || current_force_mesh_N_ != N ||
    current_force_mesh_N1_ != N1 || current_force_mesh_N2_ != N2 ||
    current_force_mesh_charge_ != charge.data() || current_force_mesh_position_ != position.data() ||
    (require_orthogonal && !box.is_orthogonal) || mesh.size() != static_cast<size_t>(para.K0K1K2) ||
    mesh_G.size() != static_cast<size_t>(para.K0K1K2)) {
    return false;
  }
  for (int d = 0; d < 3; ++d) {
    if (current_force_mesh_K_[d] != para.K[d]) return false;
  }
  for (int i = 0; i < 18; ++i) {
    if (current_force_mesh_box_[i] != box.cpu_h[i]) return false;
  }
  return true;
}

void PPPM::resize_dynamic_charge_workspace(const int M, const bool diagnostic)
{
  if (dynamic_Q_.size() != static_cast<size_t>(M)) {
    dynamic_Q_.resize(M);
    dynamic_S_.resize(M);
    dynamic_Ax_.resize(M);
    dynamic_Ay_.resize(M);
    dynamic_Az_.resize(M);
    dynamic_Bx_.resize(M);
    dynamic_By_.resize(M);
    dynamic_Bz_.resize(M);
    dynamic_d_raw_x_.resize(M);
    dynamic_d_raw_y_.resize(M);
    dynamic_d_raw_z_.resize(M);
    dynamic_d_x_.resize(M);
    dynamic_d_y_.resize(M);
    dynamic_d_z_.resize(M);
    dynamic_operator_cache_valid_ = false;
    dynamic_operator_host_cache_valid_ = false;
  }
  if (diagnostic && dynamic_L1S_x_.size() != static_cast<size_t>(M)) {
    dynamic_L1S_x_.resize(M);
    dynamic_L1S_y_.resize(M);
    dynamic_L1S_z_.resize(M);
    dynamic_h_d_x_.resize(M);
    dynamic_h_d_y_.resize(M);
    dynamic_h_d_z_.resize(M);
    dynamic_operator_host_cache_valid_ = false;
  }
  if (dynamic_current_total_.size() != 9) dynamic_current_total_.resize(9);
}

void PPPM::prepare_dynamic_operator(const int N, const Box& box, const int grid_size)
{
  bool operator_cache_match = dynamic_operator_cache_valid_ && dynamic_operator_N_ == N &&
    dynamic_operator_alpha_ == para.alpha;
  for (int d = 0; d < 3; ++d) {
    operator_cache_match = operator_cache_match && dynamic_operator_K_[d] == para.K[d];
  }
  for (int i = 0; i < 9; ++i) {
    operator_cache_match = operator_cache_match && dynamic_operator_box_[i] == box.cpu_h[i];
  }
  if (operator_cache_match) return;

  find_k_and_G_opt<<<grid_size, 64>>>(para, kx.data(), ky.data(), kz.data(), G.data());
  GPU_CHECK_KERNEL
  find_dynamic_d_raw<<<grid_size, 64>>>(
    para,
    box,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    dynamic_d_raw_x_.data(),
    dynamic_d_raw_y_.data(),
    dynamic_d_raw_z_.data());
  GPU_CHECK_KERNEL
  project_dynamic_d<<<grid_size, 64>>>(
    para,
    dynamic_d_raw_x_.data(),
    dynamic_d_raw_y_.data(),
    dynamic_d_raw_z_.data(),
    dynamic_d_x_.data(),
    dynamic_d_y_.data(),
    dynamic_d_z_.data());
  GPU_CHECK_KERNEL

  dynamic_operator_N_ = N;
  dynamic_operator_alpha_ = para.alpha;
  for (int d = 0; d < 3; ++d) dynamic_operator_K_[d] = para.K[d];
  for (int i = 0; i < 9; ++i) dynamic_operator_box_[i] = box.cpu_h[i];
  dynamic_operator_cache_valid_ = true;
  dynamic_operator_host_cache_valid_ = false;
}

void PPPM::cache_dynamic_operator_on_host()
{
  const int M = para.K0K1K2;
  dynamic_d_x_.copy_to_host(dynamic_h_d_x_.data(), M);
  dynamic_d_y_.copy_to_host(dynamic_h_d_y_.data(), M);
  dynamic_d_z_.copy_to_host(dynamic_h_d_z_.data(), M);
  dynamic_operator_finite_ = true;
  dynamic_operator_max_odd_error_[0] = 0.0;
  dynamic_operator_max_odd_error_[1] = 0.0;
  dynamic_operator_max_odd_error_[2] = 0.0;
  for (int n = 0; n < M; ++n) {
    if (!std::isfinite(static_cast<double>(dynamic_h_d_x_[n])) ||
        !std::isfinite(static_cast<double>(dynamic_h_d_y_[n])) ||
        !std::isfinite(static_cast<double>(dynamic_h_d_z_[n]))) {
      dynamic_operator_finite_ = false;
    }
    const int iz = n / para.K0K1;
    const int iy = (n - iz * para.K0K1) / para.K[0];
    const int ix = n % para.K[0];
    const int ix_bar = (para.K[0] - ix) % para.K[0];
    const int iy_bar = (para.K[1] - iy) % para.K[1];
    const int iz_bar = (para.K[2] - iz) % para.K[2];
    const int n_bar = ix_bar + para.K[0] * (iy_bar + para.K[1] * iz_bar);
    const double odd_x =
      std::fabs(double(dynamic_h_d_x_[n_bar]) + double(dynamic_h_d_x_[n]));
    const double odd_y =
      std::fabs(double(dynamic_h_d_y_[n_bar]) + double(dynamic_h_d_y_[n]));
    const double odd_z =
      std::fabs(double(dynamic_h_d_z_[n_bar]) + double(dynamic_h_d_z_[n]));
    if (odd_x > dynamic_operator_max_odd_error_[0])
      dynamic_operator_max_odd_error_[0] = odd_x;
    if (odd_y > dynamic_operator_max_odd_error_[1])
      dynamic_operator_max_odd_error_[1] = odd_y;
    if (odd_z > dynamic_operator_max_odd_error_[2])
      dynamic_operator_max_odd_error_[2] = odd_z;
  }
  dynamic_operator_host_cache_valid_ = true;
}

bool PPPM::compute_dynamic_charge_correction(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<float>& charge_rate,
  const GPU_Vector<double>& position,
  double* delta_j_q_pppm,
  const unsigned long long force_evaluation_id)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dynamic_q_last_compute_valid_ = false;
  dynamic_q_last_diagnostic_checks_pass_ = false;
  if (delta_j_q_pppm != nullptr) {
    delta_j_q_pppm[0] = nan;
    delta_j_q_pppm[1] = nan;
    delta_j_q_pppm[2] = nan;
  }
  if (
    N <= 0 || N1 < 0 || N2 > N || N1 >= N2 || charge.size() < static_cast<size_t>(N) ||
    charge_rate.size() < static_cast<size_t>(N) || position.size() < static_cast<size_t>(3 * N)) {
    std::cerr << "PPPM dynamic-q current: invalid atom range or buffer sizes." << std::endl;
    return false;
  }
  if (!box.is_orthogonal) {
    std::cerr << "PPPM dynamic-q current requires an orthogonal cell."
              << std::endl;
    return false;
  }

  find_para(N, box);
  const int M = para.K0K1K2;
  const int mesh_grid_size = (M - 1) / 64 + 1;
  const int atom_grid_size = (N2 - N1 - 1) / 64 + 1;
  const bool reuse_current_force_mesh = current_force_mesh_matches(
    N, N1, N2, box, charge, position, force_evaluation_id);
  resize_dynamic_charge_workspace(M, false);
  const gpufftComplex zero = {0.0f, 0.0f};
  if (!reuse_current_force_mesh) dynamic_Q_.fill(zero);
  dynamic_S_.fill(zero);
  dynamic_Ax_.fill(zero);
  dynamic_Ay_.fill(zero);
  dynamic_Az_.fill(zero);
  dynamic_Bx_.fill(zero);
  dynamic_By_.fill(zero);
  dynamic_Bz_.fill(zero);
  prepare_dynamic_operator(N, box, mesh_grid_size);

  find_dynamic_mesh<<<atom_grid_size, 64>>>(
    N1,
    N2,
    para,
    box,
    charge.data(),
    charge_rate.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    reuse_current_force_mesh ? nullptr : dynamic_Q_.data(),
    dynamic_S_.data(),
    dynamic_Ax_.data(),
    dynamic_Ay_.data(),
    dynamic_Az_.data(),
    dynamic_Bx_.data(),
    dynamic_By_.data(),
    dynamic_Bz_.data());
  GPU_CHECK_KERNEL

  if (!reuse_current_force_mesh) {
    if (gpufftExecC2C(plan, dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
      std::cerr << "GPUFFT error: dynamic-q current Q forward failed" << std::endl;
      return false;
    }
  }
  if (gpufftExecC2C(plan, dynamic_S_.data(), dynamic_S_.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q current S forward failed" << std::endl;
    return false;
  }
  reduce_dynamic_mesh_current<<<3, 1024>>>(
    M,
    reuse_current_force_mesh ? mesh.data() : dynamic_Q_.data(),
    dynamic_S_.data(),
    dynamic_d_x_.data(),
    dynamic_d_y_.data(),
    dynamic_d_z_.data(),
    dynamic_current_total_.data());
  GPU_CHECK_KERNEL

  if (!reuse_current_force_mesh) {
    find_mesh_G<<<mesh_grid_size, 64>>>(
      para, G.data(), dynamic_Q_.data(), dynamic_Q_.data());
    GPU_CHECK_KERNEL
    if (gpufftExecC2C(plan, dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
      std::cerr << "GPUFFT error: dynamic-q current LQ inverse failed" << std::endl;
      return false;
    }
  }
  find_mesh_G<<<mesh_grid_size, 64>>>(
    para, G.data(), dynamic_S_.data(), dynamic_S_.data());
  GPU_CHECK_KERNEL
  if (gpufftExecC2C(plan, dynamic_S_.data(), dynamic_S_.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q current LS inverse failed" << std::endl;
    return false;
  }
  reduce_dynamic_assignment_current<<<3, 1024>>>(
    M,
    reuse_current_force_mesh ? mesh_G.data() : dynamic_Q_.data(),
    dynamic_S_.data(),
    dynamic_Ax_.data(),
    dynamic_Ay_.data(),
    dynamic_Az_.data(),
    dynamic_Bx_.data(),
    dynamic_By_.data(),
    dynamic_Bz_.data(),
    dynamic_current_total_.data());
  GPU_CHECK_KERNEL

  double host_current[9] = {0.0};
  dynamic_current_total_.copy_to_host(host_current);
  bool valid = true;
  for (int n = 0; n < 9; ++n) valid = valid && std::isfinite(host_current[n]);
  for (int d = 0; d < 3; ++d) {
    const double value = host_current[d] + host_current[3 + d] + host_current[6 + d];
    valid = valid && std::isfinite(value);
    if (delta_j_q_pppm != nullptr) delta_j_q_pppm[d] = value;
  }
  dynamic_q_last_compute_valid_ = valid;
  return valid;
}

void PPPM::find_force(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<float>& D_real,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& potential_per_atom,
  const bool request_peratom_virial,
  const unsigned long long force_evaluation_id)
{
  current_force_mesh_valid_ = false;
  find_para(N, box);
  const int pppm_call_index = debug_requested_ ? debug_call_index_++ : -1;
  if (debug_requested_) {
    if (
      debug_mesh_charge_.size() != static_cast<size_t>(para.K0K1K2) ||
      debug_mesh_fourier_.size() != static_cast<size_t>(para.K0K1K2)) {
      debug_mesh_charge_.resize(para.K0K1K2);
      debug_mesh_fourier_.resize(para.K0K1K2);
    }
    if (debug_assignment_atoms_.size() != 8) {
      debug_assignment_atoms_.resize(8);
    }
    if (debug_assignment_stencil_.size() != 125) {
      debug_assignment_stencil_.resize(125);
    }
  }
  const bool calculate_peratom_virial = need_peratom_virial || request_peratom_virial;
  if (calculate_peratom_virial && plan_virial == 0) {
    allocate_virial_memory();
  }

  find_k_and_G_opt<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para, 
    kx.data(), 
    ky.data(), 
    kz.data(), 
    G.data());
  GPU_CHECK_KERNEL

  set_mesh_to_zero<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(para, mesh.data());
  GPU_CHECK_KERNEL

  if (debug_requested_) {
    std::vector<gpufftComplex> h_mesh_before(para.K0K1K2);
    mesh.copy_to_host(h_mesh_before.data(), para.K0K1K2);
    double sum_sq_real = 0.0;
    double sum_sq_imag = 0.0;
    debug_mesh_before_assignment_max_real_ = 0.0;
    debug_mesh_before_assignment_max_imag_ = 0.0;
    debug_mesh_before_assignment_sum_real_ = 0.0;
    for (const gpufftComplex& value : h_mesh_before) {
      const double real = static_cast<double>(value.x);
      const double imag = static_cast<double>(value.y);
      const double abs_real = std::fabs(real);
      const double abs_imag = std::fabs(imag);
      if (abs_real > debug_mesh_before_assignment_max_real_) {
        debug_mesh_before_assignment_max_real_ = abs_real;
      }
      if (abs_imag > debug_mesh_before_assignment_max_imag_) {
        debug_mesh_before_assignment_max_imag_ = abs_imag;
      }
      sum_sq_real += real * real;
      sum_sq_imag += imag * imag;
      debug_mesh_before_assignment_sum_real_ += real;
    }
    debug_mesh_before_assignment_rms_real_ =
      std::sqrt(sum_sq_real / para.K0K1K2);
    debug_mesh_before_assignment_rms_imag_ =
      std::sqrt(sum_sq_imag / para.K0K1K2);
  }

  find_mesh<<<(N - 1) / 64 + 1, 64>>>(
    N1,
    N2,
    para,
    box,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    mesh.data(),
    debug_requested_ ? debug_assignment_atoms_.data() : nullptr,
    debug_requested_ ? debug_assignment_stencil_.data() : nullptr);
  GPU_CHECK_KERNEL
  if (debug_requested_) {
    debug_mesh_charge_.copy_from_device(mesh.data());
    std::vector<gpufftComplex> h_mesh_after(para.K0K1K2);
    mesh.copy_to_host(h_mesh_after.data(), para.K0K1K2);
    debug_mesh_after_assignment_sum_real_ = 0.0;
    for (const gpufftComplex& value : h_mesh_after) {
      debug_mesh_after_assignment_sum_real_ += static_cast<double>(value.x);
    }
  }

  if (gpufftExecC2C(plan, mesh.data(), mesh.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Forward failed" << std::endl;
    exit(1);
  }

  ik_times_mesh_times_G<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    mesh.data(),
    mesh_x.data(),
    mesh_y.data(),
    mesh_z.data());
  GPU_CHECK_KERNEL

  find_mesh_G<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para,
    G.data(),
    mesh.data(),
    mesh_G.data());
  GPU_CHECK_KERNEL
  if (debug_requested_) {
    debug_mesh_fourier_.copy_from_device(mesh_G.data());
  }

  if (calculate_peratom_virial) {
    find_mesh_virial<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
      para,
      kx.data(),
      ky.data(),
      kz.data(),
      G.data(),
      mesh.data(),
      mesh_virial.data() + para.K0K1K2 * 0,
      mesh_virial.data() + para.K0K1K2 * 1,
      mesh_virial.data() + para.K0K1K2 * 2,
      mesh_virial.data() + para.K0K1K2 * 3,
      mesh_virial.data() + para.K0K1K2 * 4,
      mesh_virial.data() + para.K0K1K2 * 5);
    GPU_CHECK_KERNEL
  }

  if (gpufftExecC2C(plan, mesh_G.data(), mesh_G.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(plan, mesh_x.data(), mesh_x.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(plan, mesh_y.data(), mesh_y.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(plan, mesh_z.data(), mesh_z.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (calculate_peratom_virial) {
    if (gpufftExecC2C(plan_virial, mesh_virial.data(), mesh_virial.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
      std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
      exit(1);
    }

    // get force, virial, and potential in single kernel
    find_force_virial_potential_from_field<<<(N - 1) / 64 + 1, 64>>>(
      N,
      N1,
      N2,
      para,
      box,
      charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      mesh_G.data(),
      mesh_x.data(),
      mesh_y.data(),
      mesh_z.data(),
      mesh_virial.data() + para.K0K1K2 * 0,
      mesh_virial.data() + para.K0K1K2 * 1,
      mesh_virial.data() + para.K0K1K2 * 2,
      mesh_virial.data() + para.K0K1K2 * 3,
      mesh_virial.data() + para.K0K1K2 * 4,
      mesh_virial.data() + para.K0K1K2 * 5,
      D_real.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  } else {
    // get force only
    find_force_from_field<<<(N - 1) / 64 + 1, 64>>>(
      N1,
      N2,
      para,
      box,
      charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      mesh_G.data(),
      mesh_x.data(),
      mesh_y.data(),
      mesh_z.data(),
      D_real.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2);
    GPU_CHECK_KERNEL

    // then get average potential and virial
    find_potential_and_virial<<<7, 1024>>>(
      N,
      para,
      mesh.data(),
      kx.data(),
      ky.data(),
      kz.data(),
      G.data(),
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
  if (debug_requested_) {
    write_debug(N, N1, N2, box, charge, position_per_atom, D_real, pppm_call_index);
  }
  if (force_evaluation_id != 0) {
    current_force_mesh_valid_ = true;
    current_force_mesh_peratom_ = calculate_peratom_virial;
    current_force_mesh_force_evaluation_id_ = force_evaluation_id;
    current_force_mesh_N_ = N;
    current_force_mesh_N1_ = N1;
    current_force_mesh_N2_ = N2;
    current_force_mesh_charge_ = charge.data();
    current_force_mesh_position_ = position_per_atom.data();
    for (int d = 0; d < 3; ++d) current_force_mesh_K_[d] = para.K[d];
    for (int i = 0; i < 18; ++i) current_force_mesh_box_[i] = box.cpu_h[i];
  }
}

bool PPPM::compute_reference_energy_tangent(
  const int N,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>* direction,
  const GPU_Vector<double>* charge_direction,
  const unsigned long long force_evaluation_id,
  GPU_Vector<double>* dsite,
  GPU_Vector<double>& explicit_space_gradient,
  GPU_Vector<double>& native_ik_force)
{
  const bool tangent_requested = dsite != nullptr;
  const bool tangent_inputs_valid = tangent_requested
    ? direction != nullptr && charge_direction != nullptr
    : direction == nullptr && charge_direction == nullptr;
  if (
    N <= 0 || charge.size() < static_cast<size_t>(N) || position.size() < static_cast<size_t>(3) * N ||
    !tangent_inputs_valid ||
    (direction != nullptr &&
      (direction->size() < static_cast<size_t>(3) * N ||
       charge_direction->size() < static_cast<size_t>(N))) ||
    box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 || !plan_initialized ||
    mesh_x.size() != static_cast<size_t>(para.K0K1K2) ||
    mesh_y.size() != static_cast<size_t>(para.K0K1K2) ||
    mesh_z.size() != static_cast<size_t>(para.K0K1K2) ||
    !current_force_mesh_peratom_ ||
    !current_force_mesh_matches(N, 0, N, box, charge, position, force_evaluation_id, false)) {
    return false;
  }
  if (para.K[0] < 5 || para.K[1] < 5 || para.K[2] < 5) return false;

  if (explicit_space_gradient.size() != static_cast<size_t>(3) * N)
    explicit_space_gradient.resize(static_cast<size_t>(3) * N);
  if (native_ik_force.size() != static_cast<size_t>(3) * N)
    native_ik_force.resize(static_cast<size_t>(3) * N);
  if (tangent_requested && dsite->size() != static_cast<size_t>(N))
    dsite->resize(static_cast<size_t>(N));

  const int M = para.K0K1K2;
  if (tangent_requested) {
    if (reference_delta_Q_.size() != static_cast<size_t>(M)) reference_delta_Q_.resize(M);
    if (reference_delta_phi_.size() != static_cast<size_t>(M)) reference_delta_phi_.resize(M);
    clear_reference_delta_Q<<<(M - 1) / 256 + 1, 256>>>(reference_delta_Q_.data(), M);
    GPU_CHECK_KERNEL
    assign_reference_delta_Q<<<(N - 1) / 64 + 1, 64>>>(
      N, para, box, charge.data(), position.data(), direction->data(), charge_direction->data(),
      reference_delta_Q_.data());
    GPU_CHECK_KERNEL
    convert_reference_delta_Q<<<(M - 1) / 256 + 1, 256>>>(
      reference_delta_Q_.data(), reference_delta_phi_.data(), M);
    GPU_CHECK_KERNEL
    if (gpufftExecC2C(plan, reference_delta_phi_.data(), reference_delta_phi_.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS)
      return false;
    apply_reference_G<<<(M - 1) / 256 + 1, 256>>>(G.data(), reference_delta_phi_.data(), M);
    GPU_CHECK_KERNEL
    if (gpufftExecC2C(plan, reference_delta_phi_.data(), reference_delta_phi_.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS)
      return false;
  }

  gather_reference_energy_tangent<<<(N - 1) / 64 + 1, 64>>>(
    N, para, box, charge.data(), position.data(),
    tangent_requested ? direction->data() : nullptr,
    tangent_requested ? charge_direction->data() : nullptr,
    mesh_G.data(), tangent_requested ? reference_delta_phi_.data() : nullptr,
    mesh_x.data(), mesh_y.data(), mesh_z.data(),
    tangent_requested ? dsite->data() : nullptr,
    explicit_space_gradient.data(), native_ik_force.data());
  GPU_CHECK_KERNEL
  return true;
}

bool PPPM::diagnose_reference_translation_energy(
  const int N,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position,
  const unsigned long long force_evaluation_id,
  PPPMReferenceTranslationReport& report,
  const double precision_target)
{
  report = {};
  if (!(precision_target > 0.0) || !std::isfinite(precision_target)) {
    report.reason = PPPMReferenceTranslationReason::invalid_precision_target;
    return false;
  }
  report.precision_target = precision_target;
  if (N <= 0 || charge.size() < static_cast<size_t>(N) ||
      position.size() < static_cast<size_t>(3) * N ||
      !current_force_mesh_matches(N, 0, N, box, charge, position, force_evaluation_id, false) ||
      !plan_initialized || para.K0K1K2 <= 0) {
    report.reason = PPPMReferenceTranslationReason::invalid_frame;
    return false;
  }

  const int K0 = para.K[0], K1 = para.K[1], K2 = para.K[2], M = para.K0K1K2;
  if (!box.is_orthogonal || box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 ||
      K0 < 5 || K1 < 5 || K2 < 5) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = (!box.is_orthogonal || box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1)
      ? PPPMReferenceTranslationReason::unsupported_box
      : PPPMReferenceTranslationReason::unsupported_mesh;
    return true;
  }
  if (static_cast<size_t>(K0) * K1 * K2 != static_cast<size_t>(M) ||
      static_cast<size_t>(M) > std::numeric_limits<size_t>::max() / sizeof(PPPMDoubleComplex)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::unsupported_mesh;
    return true;
  }
  const size_t fp64_mesh_bytes = static_cast<size_t>(M) * sizeof(PPPMDoubleComplex);
  if (fp64_mesh_bytes > (std::numeric_limits<size_t>::max() - 16u * 1024u * 1024u) / 2) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::unsupported_mesh;
    return true;
  }
  size_t free_bytes = 0, total_bytes = 0;
#ifdef USE_HIP
  const bool memory_query_ok = hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess;
#else
  const bool memory_query_ok = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
#endif
  const size_t required_bytes = 2 * fp64_mesh_bytes + 16u * 1024u * 1024u;
  if (!memory_query_ok || free_bytes < required_bytes) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::insufficient_memory;
    return true;
  }

  std::vector<float> host_q(N), host_G(M);
  std::vector<double> host_r(static_cast<size_t>(3) * N);
  std::vector<gpufftComplex> native_phi(M);
  float host_W_coeff[5][5];
  charge.copy_to_host(host_q.data());
  position.copy_to_host(host_r.data());
  G.copy_to_host(host_G.data());
  mesh_G.copy_to_host(native_phi.data());
  if (gpuMemcpyFromSymbol(host_W_coeff, W_coeff, sizeof(host_W_coeff)) != gpuSuccess) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::coefficient_copy_failed;
    return true;
  }
  for (float q : host_q) if (!std::isfinite(q)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }
  for (double r : host_r) if (!std::isfinite(r)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }
  for (float g : host_G) if (!std::isfinite(g)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }
  for (const gpufftComplex phi : native_phi) if (!std::isfinite(phi.x) || !std::isfinite(phi.y)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }
  for (const auto& row : host_W_coeff) for (float coefficient : row) if (!std::isfinite(coefficient)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }

  const double inverse[9] = {box.cpu_h[9], box.cpu_h[10], box.cpu_h[11],
                             box.cpu_h[12], box.cpu_h[13], box.cpu_h[14],
                             box.cpu_h[15], box.cpu_h[16], box.cpu_h[17]};
  for (double value : inverse) if (!std::isfinite(value)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }
  for (int axis = 0; axis < 3; ++axis) {
    const double row_offdiag0 = inverse[axis * 3 + (axis + 1) % 3];
    const double row_offdiag1 = inverse[axis * 3 + (axis + 2) % 3];
    if (std::abs(row_offdiag0) > 1.0e-14 || std::abs(row_offdiag1) > 1.0e-14 ||
        std::abs(box.cpu_h[axis * 3 + (axis + 1) % 3]) > 1.0e-14 ||
        std::abs(box.cpu_h[axis * 3 + (axis + 2) % 3]) > 1.0e-14) {
      report.status = PPPMReferenceTranslationStatus::inconclusive;
      report.reason = PPPMReferenceTranslationReason::unsupported_box;
      return true;
    }
  }

  double scale_G = 0.0;
  for (float g : host_G) scale_G = std::max(scale_G, std::abs(static_cast<double>(g)));
  for (int iz = 0; iz < K2; ++iz) {
    for (int iy = 0; iy < K1; ++iy) {
      for (int ix = 0; ix < K0; ++ix) {
        const int index = ix + K0 * (iy + K1 * iz);
        const int opposite = ((K0 - ix) % K0) + K0 * (((K1 - iy) % K1) + K1 * ((K2 - iz) % K2));
        report.max_even_G_error = std::max(report.max_even_G_error,
          std::abs(static_cast<double>(host_G[index]) - host_G[opposite]));
      }
    }
  }
  report.even_G = report.max_even_G_error <= 32.0 * std::numeric_limits<float>::epsilon() * scale_G;
  report.mesh_zero_mode_value = host_G[0];
  report.mesh_zero_mode = host_G[0] == 0.0f;

  auto compensated_add = [](double value, double& sum, double& correction) {
    const double next = sum + value;
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
  };
  auto wrap_index = [](int value, const int size) {
    value %= size;
    return value < 0 ? value + size : value;
  };
  auto native_energy_and_derivative = [&](double& energy, double derivative[3]) {
    double esum = 0.0, ecorr = 0.0, dsum[3] = {}, dcorr[3] = {};
    for (int atom = 0; atom < N; ++atom) {
      float s[3];
      for (int a = 0; a < 3; ++a)
        s[a] = static_cast<float>((inverse[3 * a] * host_r[atom] +
          inverse[3 * a + 1] * host_r[atom + N] + inverse[3 * a + 2] * host_r[atom + 2 * N]) * para.K[a]);
      int center[3]; float delta[3], w[3][5], dw[3][5];
      for (int a = 0; a < 3; ++a) {
        center[a] = static_cast<int>(s[a] + 0.5f);
        delta[a] = s[a] - center[a];
        for (int j = 0; j < 5; ++j) {
          w[a][j] = pppm_reference_weight(host_W_coeff, j, delta[a]);
          dw[a][j] = pppm_reference_weight_derivative(host_W_coeff, j, delta[a]);
        }
      }
      double potential = 0.0, grad[3] = {};
      for (int a = -2; a <= 2; ++a) for (int b = -2; b <= 2; ++b) for (int c = -2; c <= 2; ++c) {
        const int index = wrap_index(center[0] + a, K0) + K0 * (wrap_index(center[1] + b, K1) + K1 * wrap_index(center[2] + c, K2));
        const float W = w[0][a + 2] * w[1][b + 2] * w[2][c + 2];
        const double phi = native_phi[index].x;
        potential += static_cast<double>(W) * phi;
        const double shape[3] = {
          static_cast<double>(dw[0][a + 2]) * w[1][b + 2] * w[2][c + 2] * para.K[0] * inverse[0],
          static_cast<double>(w[0][a + 2]) * dw[1][b + 2] * w[2][c + 2] * para.K[1] * inverse[4],
          static_cast<double>(w[0][a + 2]) * w[1][b + 2] * dw[2][c + 2] * para.K[2] * inverse[8]};
        for (int axis = 0; axis < 3; ++axis) grad[axis] += shape[axis] * phi;
      }
      compensated_add(static_cast<double>(K_C_SP) * host_q[atom] * potential, esum, ecorr);
      for (int axis = 0; axis < 3; ++axis)
        compensated_add(2.0 * static_cast<double>(K_C_SP) * host_q[atom] * grad[axis], dsum[axis], dcorr[axis]);
    }
    energy = esum + ecorr;
    for (int axis = 0; axis < 3; ++axis) derivative[axis] = dsum[axis] + dcorr[axis];
  };
  double native_derivative[3] = {};
  native_energy_and_derivative(report.native_reciprocal_energy, native_derivative);
  if (!std::isfinite(report.native_reciprocal_energy)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::nonfinite_input;
    return true;
  }
  for (int axis = 0; axis < 3; ++axis)
    report.axis[axis].native_energy_derivative = native_derivative[axis];

  PPPMDoublePlan double_plan;
  if (!pppm_make_double_plan(double_plan.handle, K0, K1, K2)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::fft_plan_failed;
    return true;
  }
  double_plan.initialized = true;
  size_t free_after_plan = 0, total_after_plan = 0;
#ifdef USE_HIP
  const bool post_plan_memory_ok =
    hipMemGetInfo(&free_after_plan, &total_after_plan) == hipSuccess && free_after_plan >= 2 * fp64_mesh_bytes;
#else
  const bool post_plan_memory_ok =
    cudaMemGetInfo(&free_after_plan, &total_after_plan) == cudaSuccess && free_after_plan >= 2 * fp64_mesh_bytes;
#endif
  if (!post_plan_memory_ok) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::insufficient_memory;
    return true;
  }
  GPU_Vector<PPPMDoubleComplex> fp64_mesh(static_cast<size_t>(M));
  GPU_Vector<PPPMDoubleComplex> fp64_dot_mesh(static_cast<size_t>(M));
  std::vector<PPPMDoubleComplex> host_mesh(static_cast<size_t>(M));
  std::vector<PPPMDoubleComplex> host_dot_mesh(static_cast<size_t>(M));
  std::vector<double> base_charge_mesh(static_cast<size_t>(M));
  bool oracle_fft_ok = true;
  constexpr double exact_W[5][5] = {
    {1.0 / 384.0, -1.0 / 48.0, 1.0 / 16.0, -1.0 / 12.0, 1.0 / 24.0},
    {19.0 / 96.0, -11.0 / 24.0, 1.0 / 4.0, 1.0 / 6.0, -1.0 / 6.0},
    {115.0 / 192.0, 0.0, -5.0 / 8.0, 0.0, 1.0 / 4.0},
    {19.0 / 96.0, 11.0 / 24.0, 1.0 / 4.0, -1.0 / 6.0, -1.0 / 6.0},
    {1.0 / 384.0, 1.0 / 48.0, 1.0 / 16.0, 1.0 / 12.0, 1.0 / 24.0}
  };
  auto exact_G_at = [&](const int index) {
    int nk[3] = {index % K0, (index / K0) % K1, index / (K0 * K1)};
    double u[3], sine[3], denominator[3];
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) nk[d] -= para.K[d];
      u[d] = 0.5 * static_cast<double>(para.two_pi_over_K[d]) * nk[d];
      sine[d] = std::sin(u[d]);
      const double z = sine[d] * sine[d];
      denominator[d] = 1.0 - (5.0 / 3.0) * z + (7.0 / 9.0) * z * z -
        (17.0 / 189.0) * z * z * z + (2.0 / 2835.0) * z * z * z * z;
    }
    const double kx = nk[0] * static_cast<double>(para.b[0][0]) + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    const double ky = nk[0] * static_cast<double>(para.b[0][1]) + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    const double kz = nk[0] * static_cast<double>(para.b[0][2]) + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    const double ksq = kx * kx + ky * ky + kz * kz;
    if (ksq == 0.0) return 0.0;
    double sinc_product = 1.0;
    for (int d = 0; d < 3; ++d) sinc_product *= u[d] == 0.0 ? 1.0 : sine[d] / u[d];
    const double sinc_fifth = sinc_product * sinc_product * sinc_product * sinc_product * sinc_product;
    return sinc_fifth * sinc_fifth * static_cast<double>(para.two_pi_over_V) / ksq *
      std::exp(-ksq * static_cast<double>(para.alpha_factor)) /
      (denominator[0] * denominator[0] * denominator[1] * denominator[1] * denominator[2] * denominator[2]);
  };
  std::vector<double> exact_G(M);
  for (int i = 0; i < M; ++i) {
    exact_G[i] = exact_G_at(i);
    if (!std::isfinite(exact_G[i])) {
      report.status = PPPMReferenceTranslationStatus::inconclusive;
      report.reason = PPPMReferenceTranslationReason::nonfinite_input;
      return true;
    }
  }
  auto fp64_energy = [&](const int translate_axis, const double shift, double* assignment_error,
                         const int source, double* c_energy, double* derivative, double* c_derivative) {
    std::fill(host_mesh.begin(), host_mesh.end(), PPPMDoubleComplex{0.0, 0.0});
    if (derivative != nullptr || c_derivative != nullptr)
      std::fill(host_dot_mesh.begin(), host_dot_mesh.end(), PPPMDoubleComplex{0.0, 0.0});
    for (int atom = 0; atom < N; ++atom) {
      double r[3] = {host_r[atom] + (translate_axis == 0 ? shift : 0.0),
                     host_r[atom + N] + (translate_axis == 1 ? shift : 0.0),
                     host_r[atom + 2 * N] + (translate_axis == 2 ? shift : 0.0)};
      double s[3]; int center[3]; double w[3][5], dw[3][5];
      for (int a = 0; a < 3; ++a) {
        s[a] = (inverse[3 * a] * r[0] + inverse[3 * a + 1] * r[1] + inverse[3 * a + 2] * r[2]) * para.K[a];
        center[a] = static_cast<int>(std::floor(s[a] + 0.5));
        const double delta = s[a] - center[a];
        for (int j = 0; j < 5; ++j) {
          const double* coeff = source == 1 ? exact_W[j] : nullptr;
          const double c0 = coeff != nullptr ? coeff[0] : static_cast<double>(host_W_coeff[j][0]);
          const double c1 = coeff != nullptr ? coeff[1] : static_cast<double>(host_W_coeff[j][1]);
          const double c2 = coeff != nullptr ? coeff[2] : static_cast<double>(host_W_coeff[j][2]);
          const double c3 = coeff != nullptr ? coeff[3] : static_cast<double>(host_W_coeff[j][3]);
          const double c4 = coeff != nullptr ? coeff[4] : static_cast<double>(host_W_coeff[j][4]);
          w[a][j] = (((c4 * delta + c3) * delta + c2) * delta + c1) * delta + c0;
          dw[a][j] = ((4.0 * c4 * delta + 3.0 * c3) * delta + 2.0 * c2) * delta + c1;
        }
      }
      for (int a = -2; a <= 2; ++a) for (int b = -2; b <= 2; ++b) for (int c = -2; c <= 2; ++c) {
        const int index = wrap_index(center[0] + a, K0) + K0 * (wrap_index(center[1] + b, K1) + K1 * wrap_index(center[2] + c, K2));
        const double q = static_cast<double>(host_q[atom]);
        const double weight = w[0][a + 2] * w[1][b + 2] * w[2][c + 2];
        host_mesh[index].x += q * weight;
        if ((derivative != nullptr || c_derivative != nullptr) && translate_axis >= 0) {
          const int selected = translate_axis == 0 ? a + 2 : translate_axis == 1 ? b + 2 : c + 2;
          const double dweight = (translate_axis == 0 ? dw[0][selected] * w[1][b + 2] * w[2][c + 2] :
            translate_axis == 1 ? w[0][a + 2] * dw[1][selected] * w[2][c + 2] :
                                  w[0][a + 2] * w[1][b + 2] * dw[2][selected]) *
            static_cast<double>(para.K[translate_axis]) * inverse[3 * translate_axis + translate_axis];
          host_dot_mesh[index].x += q * dweight;
        }
      }
    }
    if (assignment_error != nullptr) {
      double assigned = 0.0, assigned_correction = 0.0;
      double requested = 0.0, requested_correction = 0.0;
      for (int i = 0; i < M; ++i) compensated_add(host_mesh[i].x, assigned, assigned_correction);
      for (float q : host_q) compensated_add(q, requested, requested_correction);
      *assignment_error = std::abs((assigned + assigned_correction) - (requested + requested_correction));
      if (translate_axis < 0) {
        for (int i = 0; i < M; ++i) base_charge_mesh[i] = host_mesh[i].x;
      } else {
        double cycle_error = 0.0;
        for (int iz = 0; iz < K2; ++iz) for (int iy = 0; iy < K1; ++iy) for (int ix = 0; ix < K0; ++ix) {
          const int source = ix + K0 * (iy + K1 * iz);
          const int shifted = (ix + (translate_axis == 0)) % K0 + K0 *
            ((iy + (translate_axis == 1)) % K1 + K1 * ((iz + (translate_axis == 2)) % K2));
          cycle_error = std::max(cycle_error, std::abs(host_mesh[shifted].x - base_charge_mesh[source]));
        }
        *assignment_error = std::max(*assignment_error, cycle_error);
      }
    }
    fp64_mesh.copy_from_host(host_mesh.data());
    const bool fft_ok = pppm_forward_double(double_plan.handle, fp64_mesh.data());
    if (!fft_ok) {
      oracle_fft_ok = false;
      return std::numeric_limits<double>::quiet_NaN();
    }
    fp64_mesh.copy_to_host(host_mesh.data());
    if (derivative != nullptr || c_derivative != nullptr) {
      fp64_dot_mesh.copy_from_host(host_dot_mesh.data());
      if (!pppm_forward_double(double_plan.handle, fp64_dot_mesh.data())) {
        oracle_fft_ok = false;
        return std::numeric_limits<double>::quiet_NaN();
      }
      fp64_dot_mesh.copy_to_host(host_dot_mesh.data());
    }
    double sum = 0.0, correction = 0.0, csum = 0.0, ccorr = 0.0;
    double dsum = 0.0, dcorr = 0.0, dcsum = 0.0, dccorr = 0.0;
    for (int i = 0; i < M; ++i) {
      const double re = host_mesh[i].x, im = host_mesh[i].y;
      const double g = (c_energy != nullptr || c_derivative != nullptr) && source == 1
        ? exact_G[i] : static_cast<double>(host_G[i]);
      const double prefactor = static_cast<double>(K_C_SP);
      compensated_add(prefactor * static_cast<double>(host_G[i]) * (re * re + im * im), sum, correction);
      if (c_energy != nullptr && source == 1)
        compensated_add(prefactor * g * (re * re + im * im), csum, ccorr);
      if (derivative != nullptr || c_derivative != nullptr) {
        const double dre = host_dot_mesh[i].x, dim = host_dot_mesh[i].y;
        const double product = re * dre + im * dim;
        if (derivative != nullptr)
          compensated_add(2.0 * prefactor * static_cast<double>(host_G[i]) * product, dsum, dcorr);
        if (c_derivative != nullptr)
          compensated_add(2.0 * prefactor * g * product, dcsum, dccorr);
      }
    }
    if (c_energy != nullptr) *c_energy = csum + ccorr;
    if (derivative != nullptr) *derivative = dsum + dcorr;
    if (c_derivative != nullptr) *c_derivative = dcsum + dccorr;
    return sum + correction;
  };
  report.fp64_forward_energy = fp64_energy(-1, 0.0, &report.assignment_charge_error, -1, nullptr, nullptr, nullptr);
  if (!std::isfinite(report.fp64_forward_energy)) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::fft_execute_failed;
    return true;
  }
  report.native_vs_fp64_energy_error = std::abs(report.native_reciprocal_energy - report.fp64_forward_energy);

  constexpr double steps[3] = {0.01, 0.005, 0.0025};
  bool all_fd_platforms_ok = true;
  bool source_checks_pass = true;
  bool source_checks_fail = false;
  bool source_checks_inconclusive = false;
  bool native_comparison_inconclusive = false;
  bool native_precision_limited = false;
  for (int axis = 0; axis < 3; ++axis) {
    const double lattice_length = box.cpu_h[axis * 3 + axis];
    const double grid_shift = lattice_length / para.K[axis];
    const double integer_energy = fp64_energy(axis, grid_shift, &report.axis[axis].integer_shift_assignment_error, -1, nullptr, nullptr, nullptr);
    report.axis[axis].integer_shift_energy_error = std::abs(integer_energy - report.fp64_forward_energy);
    const double half_energy = fp64_energy(axis, 0.5 * grid_shift, nullptr, -1, nullptr, nullptr, nullptr);
    report.axis[axis].half_shift_energy_change = half_energy - report.fp64_forward_energy;
    for (int phase = 0; phase < 2; ++phase) {
      const double origin = phase == 0 ? 0.0 : 0.5 * grid_shift;
      double terms_abs = 0.0;
      for (int step_id = 0; step_id < 3; ++step_id) {
        const double h = steps[step_id];
        const double plus = fp64_energy(axis, origin + h, nullptr, 0, nullptr, nullptr, nullptr);
        const double minus = fp64_energy(axis, origin - h, nullptr, 0, nullptr, nullptr, nullptr);
        terms_abs += std::abs(plus) + std::abs(minus);
        report.axis[axis].fd_derivative[phase][step_id] = (plus - minus) / (2.0 * h);
      }
      const double* d = report.axis[axis].fd_derivative[phase];
      const double d4_coarse = (4.0 * d[1] - d[0]) / 3.0;
      const double d4_fine = (4.0 * d[2] - d[1]) / 3.0;
      report.axis[axis].richardson_derivative[phase][0] = d4_coarse;
      report.axis[axis].richardson_derivative[phase][1] = d4_fine;
      const double extrapolation_difference = std::abs(d4_fine - d4_coarse);
      report.axis[axis].fd_error_estimate[phase] = extrapolation_difference / 15.0;
      report.axis[axis].fd_roundoff_estimate[phase] =
        std::numeric_limits<double>::epsilon() * terms_abs / (2.0 * steps[2]);
      report.axis[axis].fd_plateau_error[phase] =
        extrapolation_difference + report.axis[axis].fd_roundoff_estimate[phase];
      report.axis[axis].fp64_forward_energy_derivative[phase] = d4_fine;
      report.axis[axis].fd_signal_resolved[phase] =
        std::abs(d4_fine) > 3.0 * report.axis[axis].fd_plateau_error[phase];
      if (!std::isfinite(d4_fine) || report.axis[axis].fd_plateau_error[phase] > precision_target)
        all_fd_platforms_ok = false;
      if (phase == 0) {
        report.axis[axis].native_vs_fp64_error[phase] =
          std::abs(report.axis[axis].native_energy_derivative - d4_fine);
        if (report.axis[axis].fd_plateau_error[phase] > precision_target ||
            !report.axis[axis].fd_signal_resolved[phase]) {
          native_comparison_inconclusive = true;
        } else if (report.axis[axis].native_vs_fp64_error[phase] > precision_target) {
          native_precision_limited = true;
        }
      }

      double analytic[3] = {};
      fp64_energy(axis, origin, nullptr, 0, nullptr, &analytic[0], nullptr);
      double analytic_b = 0.0, analytic_c = 0.0;
      fp64_energy(axis, origin, nullptr, 1, nullptr, &analytic_b, &analytic_c);
      analytic[1] = analytic_b;
      analytic[2] = analytic_c;
      double source_fd[3][3] = {};
      double source_abs[3] = {};
      for (int step_id = 0; step_id < 3; ++step_id) {
        source_fd[0][step_id] = report.axis[axis].fd_derivative[phase][step_id];
        source_abs[0] += terms_abs / 3.0;
        const double h = steps[step_id];
        double cplus = 0.0, cminus = 0.0;
        const double plus_b = fp64_energy(axis, origin + h, nullptr, 1, &cplus, nullptr, nullptr);
        const double minus_b = fp64_energy(axis, origin - h, nullptr, 1, &cminus, nullptr, nullptr);
        source_fd[1][step_id] = (plus_b - minus_b) / (2.0 * h);
        source_fd[2][step_id] = (cplus - cminus) / (2.0 * h);
        source_abs[1] += std::abs(plus_b) + std::abs(minus_b);
        source_abs[2] += std::abs(cplus) + std::abs(cminus);
      }
      const double fft_roundoff_factor = 64.0 * (1.0 + std::log2(static_cast<double>(M)));
      for (int source = 0; source < 3; ++source) {
        auto& result = report.axis[axis].source[source].phase[phase];
        const double* d = source_fd[source];
        const double d4_coarse = (4.0 * d[1] - d[0]) / 3.0;
        const double d4_fine = (4.0 * d[2] - d[1]) / 3.0;
        result.analytic_derivative = analytic[source];
        result.fd_derivative = d4_fine;
        result.fd_uncertainty = std::abs(d4_fine - d4_coarse);
        result.roundoff = std::numeric_limits<double>::epsilon() * fft_roundoff_factor *
          source_abs[source] / (2.0 * steps[2]);
        result.analytic_fd_difference = std::abs(result.analytic_derivative - result.fd_derivative);
        const double total_uncertainty = result.fd_uncertainty + result.roundoff;
        const bool finite = std::isfinite(result.analytic_derivative) && std::isfinite(result.fd_derivative) &&
          std::isfinite(result.fd_uncertainty) && std::isfinite(result.roundoff) &&
          std::isfinite(result.analytic_fd_difference);
        result.valid = finite;
        result.signal_resolved = finite && std::abs(result.fd_derivative) > 3.0 * total_uncertainty;
        result.pass = finite && total_uncertainty <= precision_target &&
          result.analytic_fd_difference <= total_uncertainty;
        source_checks_pass = source_checks_pass && result.pass;
        source_checks_fail = source_checks_fail ||
          (finite && total_uncertainty <= precision_target && result.analytic_fd_difference > total_uncertainty);
        source_checks_inconclusive = source_checks_inconclusive || !finite || total_uncertainty > precision_target;
      }
    }
  }
  if (!oracle_fft_ok) {
    report.status = PPPMReferenceTranslationStatus::inconclusive;
    report.reason = PPPMReferenceTranslationReason::fft_execute_failed;
    return true;
  }
  double charge_scale = 0.0;
  for (float q : host_q) charge_scale += std::abs(static_cast<double>(q));
  const double assignment_tol = 8.0 * std::numeric_limits<float>::epsilon() * std::max(1.0, charge_scale);
  report.assignment_closure_pass = report.assignment_charge_error <= assignment_tol;
  for (int axis = 0; axis < 3; ++axis)
    report.assignment_closure_pass = report.assignment_closure_pass &&
      report.axis[axis].integer_shift_assignment_error <= assignment_tol;
  report.mesh_invariant_pass = report.even_G && report.mesh_zero_mode && report.assignment_closure_pass;
  for (int axis = 0; axis < 3; ++axis) {
    const double energy_tol = 4096.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(report.fp64_forward_energy));
    report.mesh_invariant_pass = report.mesh_invariant_pass &&
      report.axis[axis].integer_shift_energy_error <= energy_tol;
  }
  report.fd_platform_pass = all_fd_platforms_ok;
  report.source_log_confirmation = true;
  report.native_derivative_comparison_inconclusive = native_comparison_inconclusive || !all_fd_platforms_ok;
  report.native_derivative_precision_limited = native_precision_limited;
  report.native_derivative_comparison_pass = !native_precision_limited && !report.native_derivative_comparison_inconclusive;
  if (!report.mesh_invariant_pass || source_checks_fail)
    report.status = PPPMReferenceTranslationStatus::fail;
  else if (!report.fd_platform_pass || !source_checks_pass || source_checks_inconclusive ||
           report.native_derivative_comparison_inconclusive || native_precision_limited)
    report.status = PPPMReferenceTranslationStatus::inconclusive;
  else
    report.status = PPPMReferenceTranslationStatus::pass;
  return true;
}

void PPPM::find_force_batch(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float*>& charge,
  const GPU_Vector<double*>& position_per_atom,
  const GPU_Vector<float*>& D_real,
  const GPU_Vector<double*>& force_per_atom,
  const GPU_Vector<double*>& virial_per_atom,
  const GPU_Vector<double*>& potential_per_atom,
  const int number_of_beads,
  const bool request_peratom_virial)
{
  current_force_mesh_valid_ = false;
  if (number_of_beads <= 0) {
    last_batch_used_peratom_virial_ = false;
    return;
  }
  const bool calculate_peratom_virial =
    need_peratom_virial_every_batch || request_peratom_virial;
  last_batch_used_peratom_virial_ = calculate_peratom_virial;
  find_para(N, box);
  allocate_batch_memory(number_of_beads);
  const int mesh_grid = (para.K0K1K2 - 1) / 64 + 1;
  const dim3 mesh_grid_batch(mesh_grid, number_of_beads);
  find_k_and_G_opt<<<mesh_grid, 64>>>(para, kx.data(), ky.data(), kz.data(), G.data());
  GPU_CHECK_KERNEL
  set_mesh_to_zero_batch<<<mesh_grid_batch, 64>>>(para, number_of_beads, mesh_batch.data());
  GPU_CHECK_KERNEL
  find_mesh_batch<<<dim3((N2 - N1 - 1) / 64 + 1, number_of_beads), 64>>>(
    N,
    N1,
    N2,
    para,
    box,
    charge.data(),
    position_per_atom.data(),
    mesh_batch.data());
  GPU_CHECK_KERNEL
  if (gpufftExecC2C(plan_batch, mesh_batch.data(), mesh_batch.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: batched forward failed" << std::endl;
    exit(1);
  }
  const size_t inverse_field_stride =
    static_cast<size_t>(number_of_beads) * para.K0K1K2;
  prepare_inverse_fields_batch<<<mesh_grid_batch, 64>>>(
    para,
    number_of_beads,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    mesh_batch.data(),
    mesh_inverse_batch.data());
  GPU_CHECK_KERNEL
  if (calculate_peratom_virial) {
    find_mesh_virial_batch<<<mesh_grid_batch, 64>>>(
      para,
      number_of_beads,
      kx.data(),
      ky.data(),
      kz.data(),
      G.data(),
      mesh_batch.data(),
      mesh_virial_batch.data());
    GPU_CHECK_KERNEL
  }
  if (gpufftExecC2C(
        plan_inverse_batch,
        mesh_inverse_batch.data(),
        mesh_inverse_batch.data(),
        GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: batched inverse failed" << std::endl;
    exit(1);
  }
  if (calculate_peratom_virial) {
    if (gpufftExecC2C(
          plan_virial_batch,
          mesh_virial_batch.data(),
          mesh_virial_batch.data(),
          GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
      std::cout << "GPUFFT error: batched virial inverse failed" << std::endl;
      exit(1);
    }
    find_force_virial_potential_from_field_batch<<<
      dim3((N2 - N1 - 1) / 64 + 1, number_of_beads), 64>>>(
      N,
      N1,
      N2,
      para,
      box,
      charge.data(),
      position_per_atom.data(),
      mesh_inverse_batch.data(),
      mesh_inverse_batch.data() + inverse_field_stride,
      mesh_inverse_batch.data() + 2 * inverse_field_stride,
      mesh_inverse_batch.data() + 3 * inverse_field_stride,
      mesh_virial_batch.data(),
      D_real.data(),
      force_per_atom.data(),
      virial_per_atom.data(),
      potential_per_atom.data(),
      number_of_beads);
    GPU_CHECK_KERNEL
  } else {
    find_force_from_field_batch<<<dim3((N2 - N1 - 1) / 64 + 1, number_of_beads), 64>>>(
      N,
      N1,
      N2,
      para,
      box,
      charge.data(),
      position_per_atom.data(),
      mesh_inverse_batch.data(),
      mesh_inverse_batch.data() + inverse_field_stride,
      mesh_inverse_batch.data() + 2 * inverse_field_stride,
      mesh_inverse_batch.data() + 3 * inverse_field_stride,
      D_real.data(),
      force_per_atom.data(),
      number_of_beads);
    GPU_CHECK_KERNEL
    find_potential_and_virial_batch<<<dim3(7, number_of_beads), 1024>>>(
      N,
      para,
      number_of_beads,
      mesh_batch.data(),
      kx.data(),
      ky.data(),
      kz.data(),
      G.data(),
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

bool PPPM::diagnose_dynamic_charge(
  const int N,
  const int N1,
  const int N2,
  const int bead_id,
  const int step,
  const double time_fs,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<float>& charge_rate,
  const GPU_Vector<double>& position,
  const bool write_debug,
  double* delta_j_q_pppm)
{
  dynamic_diagnostics_enabled_ = true;
  dynamic_q_last_compute_valid_ = false;
  dynamic_q_last_diagnostic_checks_pass_ = false;
  if (delta_j_q_pppm != nullptr) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    delta_j_q_pppm[0] = nan;
    delta_j_q_pppm[1] = nan;
    delta_j_q_pppm[2] = nan;
  }
  if (
    N <= 0 || N1 < 0 || N2 > N || N1 >= N2 || charge.size() < static_cast<size_t>(N) ||
    charge_rate.size() < static_cast<size_t>(N) || position.size() < static_cast<size_t>(3 * N)) {
    std::cerr << "PPPM dynamic-q diagnostic: invalid atom range or buffer sizes." << std::endl;
    return false;
  }
  if (!box.is_orthogonal) {
    std::cerr << "PPPM dynamic-q diagnostic requires an orthogonal cell."
              << std::endl;
    return false;
  }

  const long long pppm_call_index = dynamic_call_index_++;
  const bool emit_debug =
    (write_debug || dynamic_debug_requested_step_ == step) && !dynamic_debug_written_;
  find_para(N, box);
  const int M = para.K0K1K2;
  const int grid_size = (M - 1) / 64 + 1;
  const gpufftComplex zero = {0.0f, 0.0f};

  GPU_Vector<gpufftComplex>& Q = dynamic_Q_;
  GPU_Vector<gpufftComplex>& S = dynamic_S_;
  GPU_Vector<gpufftComplex>& Ax = dynamic_Ax_;
  GPU_Vector<gpufftComplex>& Ay = dynamic_Ay_;
  GPU_Vector<gpufftComplex>& Az = dynamic_Az_;
  GPU_Vector<gpufftComplex>& Bx = dynamic_Bx_;
  GPU_Vector<gpufftComplex>& By = dynamic_By_;
  GPU_Vector<gpufftComplex>& Bz = dynamic_Bz_;
  GPU_Vector<gpufftComplex>& L1S_x = dynamic_L1S_x_;
  GPU_Vector<gpufftComplex>& L1S_y = dynamic_L1S_y_;
  GPU_Vector<gpufftComplex>& L1S_z = dynamic_L1S_z_;
  GPU_Vector<float>& d_raw_x = dynamic_d_raw_x_;
  GPU_Vector<float>& d_raw_y = dynamic_d_raw_y_;
  GPU_Vector<float>& d_raw_z = dynamic_d_raw_z_;
  GPU_Vector<float>& d_x = dynamic_d_x_;
  GPU_Vector<float>& d_y = dynamic_d_y_;
  GPU_Vector<float>& d_z = dynamic_d_z_;

  resize_dynamic_charge_workspace(M, true);

  Q.fill(zero);
  S.fill(zero);
  Ax.fill(zero);
  Ay.fill(zero);
  Az.fill(zero);
  Bx.fill(zero);
  By.fill(zero);
  Bz.fill(zero);

  prepare_dynamic_operator(N, box, grid_size);
  if (!dynamic_operator_host_cache_valid_) cache_dynamic_operator_on_host();

  find_dynamic_mesh<<<(N2 - N1 - 1) / 64 + 1, 64>>>(
    N1,
    N2,
    para,
    box,
    charge.data(),
    charge_rate.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    Q.data(),
    S.data(),
    Ax.data(),
    Ay.data(),
    Az.data(),
    Bx.data(),
    By.data(),
    Bz.data());
  GPU_CHECK_KERNEL

  std::vector<float> h_q(N), h_qdot(N);
  charge.copy_to_host(h_q.data(), N);
  charge_rate.copy_to_host(h_qdot.data(), N);
  std::vector<gpufftComplex> h_Q(M), h_S(M);
  std::vector<gpufftComplex> h_Ax(M), h_Ay(M), h_Az(M);
  std::vector<gpufftComplex> h_Bx(M), h_By(M), h_Bz(M);
  Q.copy_to_host(h_Q.data(), M);
  S.copy_to_host(h_S.data(), M);
  Ax.copy_to_host(h_Ax.data(), M);
  Ay.copy_to_host(h_Ay.data(), M);
  Az.copy_to_host(h_Az.data(), M);
  Bx.copy_to_host(h_Bx.data(), M);
  By.copy_to_host(h_By.data(), M);
  Bz.copy_to_host(h_Bz.data(), M);

  auto stage_dynamic_fft_failure = [&](const char* stage) {
    std::cerr << "GPUFFT error: dynamic-q " << stage << " failed" << std::endl;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (dynamic_csv_row_pending_) {
      const double invalid[3] = {nan, nan, nan};
      finalize_dynamic_charge_diagnostic(invalid, invalid, false, -1);
    }
    std::ostringstream row;
    row << std::scientific << std::setprecision(16) << PPPM_DYNAMIC_SOURCE_SIGNATURE << ","
        << PPPM::dynamic_q_formula_version() << "," << step << "," << time_fs << ","
        << pppm_call_index << "," << bead_id << "," << N << "," << N1 << "," << N2 << ","
        << para.K[0] << "," << para.K[1] << "," << para.K[2] << "," << M << "," << para.alpha
        << "," << K_C_SP << "," << TIME_UNIT_CONVERSION;
    // Keep the deferred real/total correction fields for finalize_dynamic_charge_diagnostic.
    for (int column = 16; column < 84; ++column) row << "," << nan;
    dynamic_csv_row_ = row.str();
    dynamic_csv_row_pending_ = true;
    dynamic_check_buffer_ << std::scientific << std::setprecision(16) << step << " " << time_fs
                          << " " << bead_id << " " << pppm_call_index << " " << nan << " " << nan
                          << " " << nan << " 0 0 0 " << nan << " " << nan << " " << nan << " "
                          << nan << " " << nan << " " << nan << " 0 0 0 " << nan << "\n";
    return false;
  };

  if (gpufftExecC2C(plan, Q.data(), Q.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    return stage_dynamic_fft_failure("Q forward");
  }
  if (gpufftExecC2C(plan, S.data(), S.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    return stage_dynamic_fft_failure("S forward");
  }
  reduce_dynamic_mesh_current<<<3, 1024>>>(
    M,
    Q.data(),
    S.data(),
    d_x.data(),
    d_y.data(),
    d_z.data(),
    dynamic_current_total_.data());
  GPU_CHECK_KERNEL
  std::vector<gpufftComplex> h_rho(M), h_s(M);
  Q.copy_to_host(h_rho.data(), M);
  S.copy_to_host(h_s.data(), M);

  dynamic_i_d_times_s<<<grid_size, 64>>>(
    para,
    d_x.data(),
    d_y.data(),
    d_z.data(),
    S.data(),
    L1S_x.data(),
    L1S_y.data(),
    L1S_z.data());
  GPU_CHECK_KERNEL

  find_mesh_G<<<grid_size, 64>>>(para, G.data(), Q.data(), Q.data());
  GPU_CHECK_KERNEL
  find_mesh_G<<<grid_size, 64>>>(para, G.data(), S.data(), S.data());
  GPU_CHECK_KERNEL
  if (gpufftExecC2C(plan, Q.data(), Q.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    return stage_dynamic_fft_failure("LQ inverse");
  }
  if (gpufftExecC2C(plan, S.data(), S.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    return stage_dynamic_fft_failure("LS inverse");
  }
  reduce_dynamic_assignment_current<<<3, 1024>>>(
    M,
    Q.data(),
    S.data(),
    Ax.data(),
    Ay.data(),
    Az.data(),
    Bx.data(),
    By.data(),
    Bz.data(),
    dynamic_current_total_.data());
  GPU_CHECK_KERNEL
  double h_gpu_current[9] = {0.0};
  dynamic_current_total_.copy_to_host(h_gpu_current);
  bool compute_valid = true;
  for (int n = 0; n < 9; ++n) compute_valid = compute_valid && std::isfinite(h_gpu_current[n]);
  double production_current[3] = {0.0, 0.0, 0.0};
  for (int d = 0; d < 3; ++d) {
    production_current[d] = h_gpu_current[d] + h_gpu_current[3 + d] + h_gpu_current[6 + d];
    compute_valid = compute_valid && std::isfinite(production_current[d]);
  }
  dynamic_q_last_compute_valid_ = compute_valid;
  if (delta_j_q_pppm != nullptr && compute_valid) {
    for (int d = 0; d < 3; ++d) delta_j_q_pppm[d] = production_current[d];
  }
  auto stage_diagnostic_fft_failure = [&](const char* stage) {
    std::cerr << "GPUFFT error: dynamic-q diagnostic " << stage << " failed" << std::endl;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (dynamic_csv_row_pending_) {
      const double invalid[3] = {nan, nan, nan};
      finalize_dynamic_charge_diagnostic(invalid, invalid, false, -1);
    }
    std::ostringstream row;
    row << std::scientific << std::setprecision(16) << PPPM_DYNAMIC_SOURCE_SIGNATURE << ","
        << PPPM::dynamic_q_formula_version() << "," << step << "," << time_fs << ","
        << pppm_call_index << "," << bead_id << "," << N << "," << N1 << "," << N2 << ","
        << para.K[0] << "," << para.K[1] << "," << para.K[2] << "," << M << "," << para.alpha
        << "," << K_C_SP << "," << TIME_UNIT_CONVERSION;
    for (int column = 16; column < 84; ++column) {
      if (column >= 44 && column <= 46) {
        row << "," << production_current[column - 44] / TIME_UNIT_CONVERSION;
      } else if (column >= 72 && column <= 74) {
        row << "," << production_current[column - 72] / TIME_UNIT_CONVERSION;
      } else if (column == 81) {
        row << "," << (compute_valid ? 1 : 0);
      } else if (column == 82 || column == 83) {
        row << ",0";
      } else {
        row << "," << nan;
      }
    }
    dynamic_csv_row_ = row.str();
    dynamic_csv_row_pending_ = true;
    dynamic_check_buffer_ << std::scientific << std::setprecision(16) << step << " " << time_fs
                          << " " << bead_id << " " << pppm_call_index << " " << nan << " " << nan
                          << " " << nan << " 0 0 0 " << nan << " " << nan << " " << nan << " "
                          << nan << " " << nan << " " << nan << " " << (compute_valid ? 1 : 0)
                          << " 0 0 " << nan << "\n";
    dynamic_q_last_diagnostic_checks_pass_ = false;
    return compute_valid;
  };
  if (gpufftExecC2C(plan, L1S_x.data(), L1S_x.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    return stage_diagnostic_fft_failure("L1S-x inverse");
  }
  if (gpufftExecC2C(plan, L1S_y.data(), L1S_y.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    return stage_diagnostic_fft_failure("L1S-y inverse");
  }
  if (gpufftExecC2C(plan, L1S_z.data(), L1S_z.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    return stage_diagnostic_fft_failure("L1S-z inverse");
  }
  std::vector<gpufftComplex> h_LQ(M), h_LS(M), h_L1S_x(M), h_L1S_y(M), h_L1S_z(M);
  Q.copy_to_host(h_LQ.data(), M);
  S.copy_to_host(h_LS.data(), M);
  L1S_x.copy_to_host(h_L1S_x.data(), M);
  L1S_y.copy_to_host(h_L1S_y.data(), M);
  L1S_z.copy_to_host(h_L1S_z.data(), M);

  const std::vector<float>& h_d_x = dynamic_h_d_x_;
  const std::vector<float>& h_d_y = dynamic_h_d_y_;
  const std::vector<float>& h_d_z = dynamic_h_d_z_;
  std::vector<float> h_kx, h_ky, h_kz, h_G;
  std::vector<float> h_d_raw_x, h_d_raw_y, h_d_raw_z;
  if (emit_debug) {
    h_kx.resize(M);
    h_ky.resize(M);
    h_kz.resize(M);
    h_G.resize(M);
    h_d_raw_x.resize(M);
    h_d_raw_y.resize(M);
    h_d_raw_z.resize(M);
    kx.copy_to_host(h_kx.data(), M);
    ky.copy_to_host(h_ky.data(), M);
    kz.copy_to_host(h_kz.data(), M);
    G.copy_to_host(h_G.data(), M);
    d_raw_x.copy_to_host(h_d_raw_x.data(), M);
    d_raw_y.copy_to_host(h_d_raw_y.data(), M);
    d_raw_z.copy_to_host(h_d_raw_z.data(), M);
  }

  auto finite_complex = [](const std::vector<gpufftComplex>& values) {
    for (const gpufftComplex& value : values) {
      if (!std::isfinite(static_cast<double>(value.x)) ||
          !std::isfinite(static_cast<double>(value.y))) {
        return false;
      }
    }
    return true;
  };
  auto finite_float = [](const std::vector<float>& values) {
    for (const float value : values) {
      if (!std::isfinite(static_cast<double>(value))) return false;
    }
    return true;
  };
  bool all_values_finite =
    finite_float(h_q) && finite_float(h_qdot) && dynamic_operator_finite_ && finite_complex(h_Q) &&
    finite_complex(h_S) && finite_complex(h_Ax) &&
    finite_complex(h_Ay) && finite_complex(h_Az) && finite_complex(h_Bx) && finite_complex(h_By) &&
    finite_complex(h_Bz) && finite_complex(h_rho) && finite_complex(h_s) && finite_complex(h_LQ) &&
    finite_complex(h_LS) && finite_complex(h_L1S_x) && finite_complex(h_L1S_y) &&
    finite_complex(h_L1S_z);
  if (emit_debug) {
    all_values_finite = all_values_finite && finite_float(h_kx) && finite_float(h_ky) &&
      finite_float(h_kz) && finite_float(h_G) && finite_float(h_d_raw_x) &&
      finite_float(h_d_raw_y) && finite_float(h_d_raw_z);
  }

  double sum_q = 0.0;
  double sum_qdot = 0.0;
  double sum_q_assign = 0.0;
  double sum_qdot_assign = 0.0;
  for (int n = 0; n < N; ++n) {
    sum_q += h_q[n];
    sum_qdot += h_qdot[n];
    if (n >= N1 && n < N2) {
      sum_q_assign += h_q[n];
      sum_qdot_assign += h_qdot[n];
    }
  }

  double sum_Q = 0.0;
  double sum_S = 0.0;
  for (int n = 0; n < M; ++n) {
    sum_Q += h_Q[n].x;
    sum_S += h_S[n].x;
  }

  const double dynamic_relative_tolerance = 1.0e-5;
  auto within_relative_tolerance = [dynamic_relative_tolerance](
                                     const double error,
                                     const double reference) {
    if (!std::isfinite(error) || !std::isfinite(reference)) return false;
    const double scale = std::fabs(reference) > 1.0 ? std::fabs(reference) : 1.0;
    return std::fabs(error) <= dynamic_relative_tolerance * scale;
  };
  const double assignment_charge_sum_error = sum_Q - sum_q_assign;
  const double assignment_qdot_sum_error = sum_S - sum_qdot_assign;

  double J_ass_left[3] = {0.0, 0.0, 0.0};
  double J_ass_right[3] = {0.0, 0.0, 0.0};
  double J_mesh_fourier[3] = {0.0, 0.0, 0.0};
  double J_mesh_realspace[3] = {0.0, 0.0, 0.0};
  double max_odd_error[3] = {
    dynamic_operator_max_odd_error_[0],
    dynamic_operator_max_odd_error_[1],
    dynamic_operator_max_odd_error_[2]};
  double max_imag_L1S[3] = {0.0, 0.0, 0.0};
  for (int n = 0; n < M; ++n) {
    J_ass_left[0] -= double(K_C_SP) * double(h_Ax[n].x) * double(h_LS[n].x);
    J_ass_left[1] -= double(K_C_SP) * double(h_Ay[n].x) * double(h_LS[n].x);
    J_ass_left[2] -= double(K_C_SP) * double(h_Az[n].x) * double(h_LS[n].x);
    J_ass_right[0] += double(K_C_SP) * double(h_LQ[n].x) * double(h_Bx[n].x);
    J_ass_right[1] += double(K_C_SP) * double(h_LQ[n].x) * double(h_By[n].x);
    J_ass_right[2] += double(K_C_SP) * double(h_LQ[n].x) * double(h_Bz[n].x);

    const double im_conjugate_rho_s =
      double(h_rho[n].x) * double(h_s[n].y) - double(h_rho[n].y) * double(h_s[n].x);
    J_mesh_fourier[0] -= double(K_C_SP) / M * double(h_d_x[n]) * im_conjugate_rho_s;
    J_mesh_fourier[1] -= double(K_C_SP) / M * double(h_d_y[n]) * im_conjugate_rho_s;
    J_mesh_fourier[2] -= double(K_C_SP) / M * double(h_d_z[n]) * im_conjugate_rho_s;

    J_mesh_realspace[0] += double(K_C_SP) * double(h_Q[n].x) * double(h_L1S_x[n].x) / M;
    J_mesh_realspace[1] += double(K_C_SP) * double(h_Q[n].x) * double(h_L1S_y[n].x) / M;
    J_mesh_realspace[2] += double(K_C_SP) * double(h_Q[n].x) * double(h_L1S_z[n].x) / M;

    const double imag_x = std::fabs(double(h_L1S_x[n].y) / M);
    const double imag_y = std::fabs(double(h_L1S_y[n].y) / M);
    const double imag_z = std::fabs(double(h_L1S_z[n].y) / M);
    if (imag_x > max_imag_L1S[0]) max_imag_L1S[0] = imag_x;
    if (imag_y > max_imag_L1S[1]) max_imag_L1S[1] = imag_y;
    if (imag_z > max_imag_L1S[2]) max_imag_L1S[2] = imag_z;
  }

  double mesh_path_error = 0.0;
  double mesh_path_scale = 1.0;
  for (int d = 0; d < 3; ++d) {
    const double mesh_error = std::fabs(J_mesh_fourier[d] - J_mesh_realspace[d]);
    if (mesh_error > mesh_path_error) mesh_path_error = mesh_error;
    if (std::fabs(J_mesh_fourier[d]) > mesh_path_scale) {
      mesh_path_scale = std::fabs(J_mesh_fourier[d]);
    }
    if (std::fabs(J_mesh_realspace[d]) > mesh_path_scale) {
      mesh_path_scale = std::fabs(J_mesh_realspace[d]);
    }
  }
  const double mesh_path_threshold = dynamic_relative_tolerance * mesh_path_scale;

  const double J_ass[3] = {
    J_ass_left[0] + J_ass_right[0],
    J_ass_left[1] + J_ass_right[1],
    J_ass_left[2] + J_ass_right[2]};
  const double J_mesh[3] = {J_mesh_fourier[0], J_mesh_fourier[1], J_mesh_fourier[2]};
  const double DeltaJ_cpu[3] = {
    J_ass[0] + J_mesh[0],
    J_ass[1] + J_mesh[1],
    J_ass[2] + J_mesh[2]};
  const double inv_time = 1.0 / TIME_UNIT_CONVERSION;
  double DeltaJ_gpu[3] = {0.0, 0.0, 0.0};
  double DeltaJ_gpu_cpu_error[3] = {0.0, 0.0, 0.0};
  bool gpu_cpu_match = true;
  double max_abs_gpu_cpu_error = 0.0;
  for (int d = 0; d < 3; ++d) {
    DeltaJ_gpu[d] = production_current[d];
    DeltaJ_gpu_cpu_error[d] = std::fabs((DeltaJ_gpu[d] - DeltaJ_cpu[d]) * inv_time);
    const double scale = std::max(
      std::fabs(DeltaJ_gpu[d] * inv_time), std::fabs(DeltaJ_cpu[d] * inv_time));
    gpu_cpu_match = gpu_cpu_match &&
      DeltaJ_gpu_cpu_error[d] <= 1.0e-8 + 1.0e-5 * scale;
    if (!std::isfinite(DeltaJ_gpu_cpu_error[d])) {
      max_abs_gpu_cpu_error = std::numeric_limits<double>::quiet_NaN();
    } else if (std::isfinite(max_abs_gpu_cpu_error) &&
               DeltaJ_gpu_cpu_error[d] > max_abs_gpu_cpu_error) {
      max_abs_gpu_cpu_error = DeltaJ_gpu_cpu_error[d];
    }
  }
  const double DeltaJ[3] = {DeltaJ_gpu[0], DeltaJ_gpu[1], DeltaJ_gpu[2]};
  bool derived_values_finite = std::isfinite(mesh_path_error) && std::isfinite(mesh_path_scale) &&
    std::isfinite(mesh_path_threshold);
  for (int d = 0; d < 3; ++d) {
    derived_values_finite = derived_values_finite && std::isfinite(J_ass_left[d]) &&
      std::isfinite(J_ass_right[d]) && std::isfinite(J_mesh_fourier[d]) &&
      std::isfinite(J_mesh_realspace[d]) && std::isfinite(J_ass[d]) && std::isfinite(J_mesh[d]) &&
      std::isfinite(DeltaJ_cpu[d]) && std::isfinite(DeltaJ[d]);
  }
  all_values_finite = all_values_finite && derived_values_finite;
  const bool assignment_sums_ok =
    all_values_finite && within_relative_tolerance(assignment_charge_sum_error, sum_q_assign) &&
    within_relative_tolerance(assignment_qdot_sum_error, sum_qdot_assign);
  const bool mesh_path_ok =
    all_values_finite && mesh_path_error <= mesh_path_threshold;
  const bool diagnostic_checks_pass =
    compute_valid && all_values_finite && assignment_sums_ok && mesh_path_ok && gpu_cpu_match;
  dynamic_q_last_compute_valid_ = compute_valid;
  dynamic_q_last_diagnostic_checks_pass_ = diagnostic_checks_pass;
  if (delta_j_q_pppm != nullptr && compute_valid) {
    delta_j_q_pppm[0] = DeltaJ[0];
    delta_j_q_pppm[1] = DeltaJ[1];
    delta_j_q_pppm[2] = DeltaJ[2];
  }

  std::ostringstream dynamic_csv_row;
  dynamic_csv_row << std::scientific << std::setprecision(16)
                  << PPPM_DYNAMIC_SOURCE_SIGNATURE << "," << PPPM::dynamic_q_formula_version() << ","
                  << step << "," << time_fs << "," << pppm_call_index << "," << bead_id << ","
                  << N << "," << N1 << "," << N2 << "," << para.K[0] << "," << para.K[1] << ","
                  << para.K[2] << "," << M << "," << para.alpha << "," << K_C_SP << ","
                  << TIME_UNIT_CONVERSION << "," << sum_q << "," << sum_qdot << "," << sum_q_assign
                  << "," << sum_qdot_assign << "," << sum_Q << "," << sum_S << "," << h_rho[0].x
                  << "," << h_rho[0].y << "," << h_s[0].x << "," << h_s[0].y;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << J_ass_left[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << J_ass_right[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << J_ass[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << J_mesh_fourier[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << J_mesh_realspace[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << J_mesh[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << DeltaJ[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << max_odd_error[d];
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << max_imag_L1S[d];
  dynamic_csv_row << "," << assignment_charge_sum_error << "," << assignment_qdot_sum_error << ","
                  << mesh_path_error * inv_time << "," << (assignment_sums_ok ? 1 : 0) << ","
                  << (mesh_path_ok ? 1 : 0) << "," << (all_values_finite ? 1 : 0);
  for (int i = 0; i < 9; ++i) dynamic_csv_row << "," << box.cpu_h[i];
  dynamic_csv_row << "," << mesh_path_scale << "," << mesh_path_threshold << ","
                  << mesh_path_scale * inv_time << "," << mesh_path_threshold * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << DeltaJ_gpu[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << DeltaJ_cpu[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_row << "," << DeltaJ_gpu_cpu_error[d];
  dynamic_csv_row << "," << (compute_valid ? 1 : 0) << ","
                  << (diagnostic_checks_pass ? 1 : 0) << "," << (gpu_cpu_match ? 1 : 0);
  if (dynamic_csv_row_pending_) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double invalid[3] = {nan, nan, nan};
    finalize_dynamic_charge_diagnostic(invalid, invalid, false, -1);
  }
  dynamic_csv_row_ = dynamic_csv_row.str();
  dynamic_csv_row_pending_ = true;

  dynamic_check_buffer_ << std::scientific << std::setprecision(16) << step << " " << time_fs << " "
                        << bead_id << " " << pppm_call_index << " " << mesh_path_error * inv_time << " "
                        << assignment_charge_sum_error << " " << assignment_qdot_sum_error << " "
                        << (assignment_sums_ok ? 1 : 0) << " " << (mesh_path_ok ? 1 : 0) << " "
                        << (all_values_finite ? 1 : 0) << " " << max_odd_error[0] << " "
                        << max_odd_error[1] << " " << max_odd_error[2] << " " << max_imag_L1S[0] << " "
                        << max_imag_L1S[1] << " " << max_imag_L1S[2] << " "
                        << (compute_valid ? 1 : 0) << " " << (diagnostic_checks_pass ? 1 : 0) << " "
                        << (gpu_cpu_match ? 1 : 0) << " " << max_abs_gpu_cpu_error << "\n";

  if (emit_debug) {
    std::vector<double> h_position(3 * N);
    position.copy_to_host(h_position.data(), 3 * N);

    std::ostringstream atom_rows;
    atom_rows << std::scientific << std::setprecision(16)
              << "# step " << step << " time_fs " << time_fs << " pppm_call_index "
              << pppm_call_index << " bead_id " << bead_id << " N " << N << " N1 " << N1
              << " N2 " << N2 << "\n";
    for (int n = 0; n < N; ++n) {
      atom_rows << n << " " << h_position[n] << " " << h_position[N + n] << " "
                << h_position[2 * N + n] << " " << h_q[n] << " " << h_qdot[n] << " "
                << h_qdot[n] * inv_time << "\n";
    }
    dynamic_atom_debug_buffer_ = atom_rows.str();

    std::ostringstream kspace_rows;
    kspace_rows << std::scientific << std::setprecision(16)
                << "# step " << step << " time_fs " << time_fs << " pppm_call_index "
                << pppm_call_index << " bead_id " << bead_id << " N " << N << " N1 " << N1
                << " N2 " << N2 << "\n";
    for (int iz = 0; iz < para.K[2]; ++iz) {
      for (int iy = 0; iy < para.K[1]; ++iy) {
        for (int ix = 0; ix < para.K[0]; ++ix) {
          const int n = ix + para.K[0] * (iy + para.K[1] * iz);
          const int nx = ix >= para.K_half[0] ? ix - para.K[0] : ix;
          const int ny = iy >= para.K_half[1] ? iy - para.K[1] : iy;
          const int nz = iz >= para.K_half[2] ? iz - para.K[2] : iz;
          kspace_rows << ix << " " << iy << " " << iz << " " << nx << " " << ny << " " << nz
                      << " " << h_kx[n] << " " << h_ky[n] << " " << h_kz[n] << " " << h_G[n]
                      << " " << h_G[n] << " " << double(M) * h_G[n] << " " << h_d_raw_x[n]
                      << " " << h_d_raw_y[n] << " " << h_d_raw_z[n] << " " << h_d_x[n] << " "
                      << h_d_y[n] << " " << h_d_z[n] << " " << h_rho[n].x << " " << h_rho[n].y
                      << " " << h_s[n].x << " " << h_s[n].y << "\n";
        }
      }
    }
    dynamic_kspace_debug_buffer_ = kspace_rows.str();
    dynamic_debug_written_ = true;
  }

  return compute_valid;
}
