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
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

namespace{

constexpr const char* PPPM_DEBUG_SOURCE_SIGNATURE = "PPPM_ASSIGN_DEBUG_20260910_V1";
constexpr const char* PPPM_DYNAMIC_SOURCE_SIGNATURE = "PPPM_DYNAMIC_Q_DIAG";
constexpr const char* PPPM_DYNAMIC_FORMULA_VERSION = "candidate_v1";

void write_dynamic_metadata(std::ostream& file)
{
  file << "# source_signature = " << PPPM_DYNAMIC_SOURCE_SIGNATURE << "\n";
  file << "# dynamic_formula_version = " << PPPM_DYNAMIC_FORMULA_VERSION << "\n";
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
  file << "# J_unit = eV*Angstrom/fs\n";
  file << "# J_conversion = divide_by_TIME_UNIT_CONVERSION\n";
  file << "# geometry_restriction = fixed orthogonal cell, post_force before compute2\n";
  file << "# nyquist_rule = Cartesian component plane zero (orthogonal-cell candidate_v1 only)\n";
  file << "# diagnostic_relative_tolerance = 1e-5\n";
  file << "# csv_frequency = every sampled diagnostic call; file_write = post_run\n";
  file << "# detailed_debug_frequency = first diagnostic call only\n";
}

bool append_text_file(const char* filename, const std::string& header, const std::string& rows)
{
  if (rows.empty()) return true;
  std::ifstream probe(filename, std::ios::binary | std::ios::ate);
  const bool empty = !probe || probe.tellg() == std::streampos(0);
  probe.close();
  std::ofstream file(filename, std::ios::app);
  if (!file) return false;
  if (empty) file << header;
  file << rows;
  file.flush();
  return file.good();
}

int get_best_K(const int m)
{
  int n = 16;
  while (n < m) {
    n *= 2;
  }
  return n;
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
    // candidate_v1: Cartesian Nyquist-plane projection is valid only for orthogonal cells.
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
          atomicAdd(&g_Q[neighbor012].x, qW);
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
  if (plan != 0) {
    gpufftDestroy(plan);
  }
  if (plan_batch != 0) {
    gpufftDestroy(plan_batch);
  }
  if (plan_inverse_batch != 0) {
    gpufftDestroy(plan_inverse_batch);
  }
  if (plan_virial != 0) {
    gpufftDestroy(plan_virial);
  }
  if (plan_virial_batch != 0) {
    gpufftDestroy(plan_virial_batch);
  }
}

void PPPM::flush_dynamic_charge_diagnostics()
{
  const std::string csv_rows = dynamic_csv_buffer_.str();
  std::ostringstream csv_header;
  write_dynamic_metadata(csv_header);
  csv_header
    << "source_signature,dynamic_formula_version,step,time_fs,pppm_call_index,bead_id,N,N1,N2,"
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
       "h00,h01,h02,h10,h11,h12,h20,h21,h22\n";
  if (!append_text_file("pppm_dynamic_q_diag.csv", csv_header.str(), csv_rows)) {
    std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_diag.csv." << std::endl;
  } else if (!csv_rows.empty()) {
    dynamic_csv_buffer_.str("");
    dynamic_csv_buffer_.clear();
  }

  const std::string check_rows = dynamic_check_buffer_.str();
  const std::string check_header =
    "# PPPM dynamic-q runtime checks; written after the run\n"
    "# source_signature = PPPM_DYNAMIC_Q_DIAG\n"
    "# dynamic_formula_version = candidate_v1\n"
    "# units: mesh_path=eV*Angstrom/fs; charge_sum=e; qdot_sum=e/natural_time\n"
    "# columns: step time_fs bead_id pppm_call_index max_abs_J_mesh_path "
    "assignment_charge_sum_error assignment_qdot_sum_error assignment_sums_ok "
    "mesh_path_ok all_values_finite max_odd_error_dx max_odd_error_dy max_odd_error_dz "
    "max_imag_L1S_x max_imag_L1S_y max_imag_L1S_z\n";
  if (!append_text_file("pppm_dynamic_q_check.out", check_header, check_rows)) {
    std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_check.out." << std::endl;
  } else if (!check_rows.empty()) {
    dynamic_check_buffer_.str("");
    dynamic_check_buffer_.clear();
  }

  std::ostringstream atom_header;
  write_dynamic_metadata(atom_header);
  atom_header << "# columns atom_id x y z q qdot_internal qdot_e_per_fs\n";
  if (!append_text_file(
        "pppm_dynamic_q_atom_debug.out", atom_header.str(), dynamic_atom_debug_buffer_)) {
    if (!dynamic_atom_debug_buffer_.empty()) {
      std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_atom_debug.out."
                << std::endl;
    }
  } else {
    dynamic_atom_debug_buffer_.clear();
  }

  std::ostringstream kspace_header;
  write_dynamic_metadata(kspace_header);
  kspace_header << "# columns ix iy iz nx ny nz kx ky kz Gopt g ell "
                   "d_raw_x d_raw_y d_raw_z d_x d_y d_z rho_real rho_imag s_real s_imag\n";
  if (!append_text_file(
        "pppm_dynamic_q_kspace_debug.out", kspace_header.str(), dynamic_kspace_debug_buffer_)) {
    if (!dynamic_kspace_debug_buffer_.empty()) {
      std::cerr << "PPPM dynamic-q diagnostic: cannot write pppm_dynamic_q_kspace_debug.out."
                << std::endl;
    }
  } else {
    dynamic_kspace_debug_buffer_.clear();
  }
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
  if (plan_virial != 0) {
    return;
  }
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
}

void PPPM::allocate_memory()
{
  if (plan != 0) {
    gpufftDestroy(plan);
    plan = 0;
  }
  if (plan_virial != 0) {
    gpufftDestroy(plan_virial);
    plan_virial = 0;
  }
  if (plan_batch != 0) {
    gpufftDestroy(plan_batch);
    plan_batch = 0;
  }
  if (plan_inverse_batch != 0) {
    gpufftDestroy(plan_inverse_batch);
    plan_inverse_batch = 0;
  }
  if (plan_virial_batch != 0) {
    gpufftDestroy(plan_virial_batch);
    plan_virial_batch = 0;
  }
  batch_capacity = 0;
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

}

void PPPM::allocate_batch_memory(const int number_of_beads)
{
  if (number_of_beads == batch_capacity) {
    return;
  }
  if (plan_batch != 0) {
    gpufftDestroy(plan_batch);
    plan_batch = 0;
  }
  if (plan_virial_batch != 0) {
    gpufftDestroy(plan_virial_batch);
    plan_virial_batch = 0;
  }
  if (plan_inverse_batch != 0) {
    gpufftDestroy(plan_inverse_batch);
    plan_inverse_batch = 0;
  }
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
  }
}

void PPPM::initialize(const float alpha_input)
{
  need_peratom_virial = check_need_peratom_virial();
  need_peratom_virial_every_batch = check_need_peratom_virial_every_batch();
  para.alpha = alpha_input;
  para.alpha_factor = 0.25f / (para.alpha * para.alpha);
  para.K[0] = 16;
  para.K[1] = 16;
  para.K[2] = 16;
  para.K0K1K2 = para.K[0] * para.K[1] * para.K[2];
  allocate_memory();
}

void PPPM::find_para(const int N, const Box& box)
{
  const float two_pi = 6.2831853f;
  const double volume = box.get_volume();
  para.two_pi_over_V = two_pi / volume;
  int K[3] = {0};
  for (int d = 0; d < 3; ++d) {
    const double box_thickness = volume / box.get_area(d);
    K[d] = box_thickness / mesh_spacing;
    K[d] = get_best_K(K[d]);
    para.K_half[d] = K[d] / 2;
    para.two_pi_over_K[d] = two_pi / K[d];
  }
  para.K0K1 = K[0] * K[1];
  para.K0K1K2 = para.K0K1 * K[2];
  if (K[0] != para.K[0] || K[1] != para.K[1] || K[2] != para.K[2]) {
    para.K[0] = K[0];
    para.K[1] = K[1];
    para.K[2] = K[2];
    allocate_memory();
  }
  para.potential_factor = K_C_SP / N;
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * (float)box.cpu_h[9 + d];
    para.b[1][d] = two_pi * (float)box.cpu_h[12 + d];
    para.b[2][d] = two_pi * (float)box.cpu_h[15 + d];
  }
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
  const bool request_peratom_virial)
{
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
  if (delta_j_q_pppm != nullptr) {
    delta_j_q_pppm[0] = 0.0;
    delta_j_q_pppm[1] = 0.0;
    delta_j_q_pppm[2] = 0.0;
  }
  if (
    dynamic_cache_set_ && step == dynamic_cache_step_ && bead_id == dynamic_cache_bead_ &&
    N1 == dynamic_cache_N1_ && N2 == dynamic_cache_N2_ && time_fs == dynamic_cache_time_fs_) {
    if (delta_j_q_pppm != nullptr) {
      delta_j_q_pppm[0] = dynamic_cache_delta_j_[0];
      delta_j_q_pppm[1] = dynamic_cache_delta_j_[1];
      delta_j_q_pppm[2] = dynamic_cache_delta_j_[2];
    }
    return dynamic_cache_result_valid_;
  }
  if (N <= 0 || N1 < 0 || N2 > N || N1 >= N2) {
    std::cerr << "PPPM dynamic-q diagnostic: invalid atom range." << std::endl;
    return false;
  }
  if (!box.is_orthogonal) {
    std::cerr << "PPPM dynamic-q diagnostic: candidate_v1 requires an orthogonal cell."
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

  GPU_Vector<gpufftComplex> Q(M, zero);
  GPU_Vector<gpufftComplex> S(M, zero);
  GPU_Vector<gpufftComplex> Ax(M, zero);
  GPU_Vector<gpufftComplex> Ay(M, zero);
  GPU_Vector<gpufftComplex> Az(M, zero);
  GPU_Vector<gpufftComplex> Bx(M, zero);
  GPU_Vector<gpufftComplex> By(M, zero);
  GPU_Vector<gpufftComplex> Bz(M, zero);
  GPU_Vector<gpufftComplex> L1S_x(M, zero);
  GPU_Vector<gpufftComplex> L1S_y(M, zero);
  GPU_Vector<gpufftComplex> L1S_z(M, zero);
  GPU_Vector<float> d_raw_x(M);
  GPU_Vector<float> d_raw_y(M);
  GPU_Vector<float> d_raw_z(M);
  GPU_Vector<float> d_x(M);
  GPU_Vector<float> d_y(M);
  GPU_Vector<float> d_z(M);

  find_k_and_G_opt<<<grid_size, 64>>>(para, kx.data(), ky.data(), kz.data(), G.data());
  GPU_CHECK_KERNEL
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

  find_dynamic_d_raw<<<grid_size, 64>>>(
    para,
    box,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    d_raw_x.data(),
    d_raw_y.data(),
    d_raw_z.data());
  GPU_CHECK_KERNEL
  project_dynamic_d<<<grid_size, 64>>>(
    para,
    d_raw_x.data(),
    d_raw_y.data(),
    d_raw_z.data(),
    d_x.data(),
    d_y.data(),
    d_z.data());
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

  if (gpufftExecC2C(plan, Q.data(), Q.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q Q forward failed" << std::endl;
    return false;
  }
  if (gpufftExecC2C(plan, S.data(), S.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q S forward failed" << std::endl;
    return false;
  }
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
    std::cerr << "GPUFFT error: dynamic-q LQ inverse failed" << std::endl;
    return false;
  }
  if (gpufftExecC2C(plan, S.data(), S.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q LS inverse failed" << std::endl;
    return false;
  }
  if (gpufftExecC2C(plan, L1S_x.data(), L1S_x.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q L1S-x inverse failed" << std::endl;
    return false;
  }
  if (gpufftExecC2C(plan, L1S_y.data(), L1S_y.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q L1S-y inverse failed" << std::endl;
    return false;
  }
  if (gpufftExecC2C(plan, L1S_z.data(), L1S_z.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cerr << "GPUFFT error: dynamic-q L1S-z inverse failed" << std::endl;
    return false;
  }
  std::vector<gpufftComplex> h_LQ(M), h_LS(M), h_L1S_x(M), h_L1S_y(M), h_L1S_z(M);
  Q.copy_to_host(h_LQ.data(), M);
  S.copy_to_host(h_LS.data(), M);
  L1S_x.copy_to_host(h_L1S_x.data(), M);
  L1S_y.copy_to_host(h_L1S_y.data(), M);
  L1S_z.copy_to_host(h_L1S_z.data(), M);

  std::vector<float> h_d_x(M), h_d_y(M), h_d_z(M);
  d_x.copy_to_host(h_d_x.data(), M);
  d_y.copy_to_host(h_d_y.data(), M);
  d_z.copy_to_host(h_d_z.data(), M);
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
    finite_float(h_q) && finite_float(h_qdot) && finite_float(h_d_x) && finite_float(h_d_y) &&
    finite_float(h_d_z) && finite_complex(h_Q) && finite_complex(h_S) && finite_complex(h_Ax) &&
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
  double max_odd_error[3] = {0.0, 0.0, 0.0};
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

    const int iz = n / para.K0K1;
    const int iy = (n - iz * para.K0K1) / para.K[0];
    const int ix = n % para.K[0];
    const int ix_bar = (para.K[0] - ix) % para.K[0];
    const int iy_bar = (para.K[1] - iy) % para.K[1];
    const int iz_bar = (para.K[2] - iz) % para.K[2];
    const int n_bar = ix_bar + para.K[0] * (iy_bar + para.K[1] * iz_bar);
    const double odd_x = std::fabs(double(h_d_x[n_bar]) + double(h_d_x[n]));
    const double odd_y = std::fabs(double(h_d_y[n_bar]) + double(h_d_y[n]));
    const double odd_z = std::fabs(double(h_d_z[n_bar]) + double(h_d_z[n]));
    if (odd_x > max_odd_error[0]) max_odd_error[0] = odd_x;
    if (odd_y > max_odd_error[1]) max_odd_error[1] = odd_y;
    if (odd_z > max_odd_error[2]) max_odd_error[2] = odd_z;
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
  const bool assignment_sums_ok =
    all_values_finite && within_relative_tolerance(assignment_charge_sum_error, sum_q_assign) &&
    within_relative_tolerance(assignment_qdot_sum_error, sum_qdot_assign);
  const bool mesh_path_ok =
    all_values_finite && mesh_path_error <= dynamic_relative_tolerance * mesh_path_scale;
  const bool result_valid = all_values_finite && assignment_sums_ok && mesh_path_ok;

  const double J_ass[3] = {
    J_ass_left[0] + J_ass_right[0],
    J_ass_left[1] + J_ass_right[1],
    J_ass_left[2] + J_ass_right[2]};
  const double J_mesh[3] = {J_mesh_fourier[0], J_mesh_fourier[1], J_mesh_fourier[2]};
  const double DeltaJ[3] = {
    J_ass[0] + J_mesh[0],
    J_ass[1] + J_mesh[1],
    J_ass[2] + J_mesh[2]};
  const double inv_time = 1.0 / TIME_UNIT_CONVERSION;
  if (delta_j_q_pppm != nullptr) {
    delta_j_q_pppm[0] = DeltaJ[0] * inv_time;
    delta_j_q_pppm[1] = DeltaJ[1] * inv_time;
    delta_j_q_pppm[2] = DeltaJ[2] * inv_time;
  }

  dynamic_csv_buffer_ << std::scientific << std::setprecision(16)
                      << PPPM_DYNAMIC_SOURCE_SIGNATURE << "," << PPPM_DYNAMIC_FORMULA_VERSION << ","
                      << step << "," << time_fs << "," << pppm_call_index << "," << bead_id << ","
                      << N << "," << N1 << "," << N2 << "," << para.K[0] << "," << para.K[1] << ","
                      << para.K[2] << "," << M << "," << para.alpha << "," << K_C_SP << ","
                      << TIME_UNIT_CONVERSION << "," << sum_q << "," << sum_qdot << "," << sum_q_assign
                      << "," << sum_qdot_assign << "," << sum_Q << "," << sum_S << "," << h_rho[0].x
                      << "," << h_rho[0].y << "," << h_s[0].x << "," << h_s[0].y;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << J_ass_left[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << J_ass_right[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << J_ass[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << J_mesh_fourier[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << J_mesh_realspace[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << J_mesh[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << DeltaJ[d] * inv_time;
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << max_odd_error[d];
  for (int d = 0; d < 3; ++d) dynamic_csv_buffer_ << "," << max_imag_L1S[d];
  dynamic_csv_buffer_ << "," << assignment_charge_sum_error << "," << assignment_qdot_sum_error << ","
                      << mesh_path_error * inv_time << "," << (assignment_sums_ok ? 1 : 0) << ","
                      << (mesh_path_ok ? 1 : 0) << "," << (all_values_finite ? 1 : 0);
  for (int i = 0; i < 9; ++i) dynamic_csv_buffer_ << "," << box.cpu_h[i];
  dynamic_csv_buffer_ << "\n";

  dynamic_check_buffer_ << std::scientific << std::setprecision(16) << step << " " << time_fs << " "
                        << bead_id << " " << pppm_call_index << " " << mesh_path_error * inv_time << " "
                        << assignment_charge_sum_error << " " << assignment_qdot_sum_error << " "
                        << (assignment_sums_ok ? 1 : 0) << " " << (mesh_path_ok ? 1 : 0) << " "
                        << (all_values_finite ? 1 : 0) << " " << max_odd_error[0] << " "
                        << max_odd_error[1] << " " << max_odd_error[2] << " " << max_imag_L1S[0] << " "
                        << max_imag_L1S[1] << " " << max_imag_L1S[2] << "\n";

  dynamic_cache_set_ = true;
  dynamic_cache_step_ = step;
  dynamic_cache_bead_ = bead_id;
  dynamic_cache_N1_ = N1;
  dynamic_cache_N2_ = N2;
  dynamic_cache_time_fs_ = time_fs;
  dynamic_cache_result_valid_ = result_valid;
  for (int d = 0; d < 3; ++d)
    dynamic_cache_delta_j_[d] = DeltaJ[d] * inv_time;

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

  return result_valid;
}
