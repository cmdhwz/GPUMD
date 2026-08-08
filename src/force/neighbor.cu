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
neighbor list.
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <chrono>
#include <cstring>

static __device__ void find_cell_id(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(box, x, y, z, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);
}

static __global__ void find_cell_counts(
  const Box box,
  const int N,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents(
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

static __global__ void gpu_find_neighbor_ON1(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const float cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          else if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          else if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          else if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              float x12 = x[n2] - x1;
              float y12 = y[n2] - y1;
              float z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const float d2 = x12 * x12 + y12 * y12 + z12 * z12;

              if (d2 < cutoff_square) {
                NL[static_cast<size_t>(N) * count++ + n1] = n2;
              }
            }
          }
        }
      }
    }
    NN[n1] = count;
  }
}

void find_cell_list(
  const double rc,
  const int* num_bins,
  Box& box,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int N = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  CHECK(gpuMemset(cell_count.data(), 0, sizeof(int) * N_cells));
  CHECK(gpuMemset(cell_count_sum.data(), 0, sizeof(int) * N_cells));
  CHECK(gpuMemset(cell_contents.data(), 0, sizeof(int) * N));

  find_cell_counts<<<grid_size, block_size>>>(
    box, N, cell_count.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
    thrust::device, cell_count.data(), cell_count.data() + N_cells, cell_count_sum.data());

  CHECK(gpuMemset(cell_count.data(), 0, sizeof(int) * N_cells));

  find_cell_contents<<<grid_size, block_size>>>(
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL
}

static void __global__ set_to_zero(int size, int* data)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    data[n] = 0;
  }
}

void find_cell_list(
  gpuStream_t& stream,
  const double rc,
  const int* num_bins,
  Box& box,
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int offset = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + offset;
  const double* z = position_per_atom.data() + offset * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_count_sum.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count_sum.size(), cell_count_sum.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_contents.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_contents.size(), cell_contents.data());
  GPU_CHECK_KERNEL

  find_cell_counts<<<grid_size, block_size, 0, stream>>>(
    box, N, cell_count.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
#ifdef USE_HIP
    thrust::hip::par.on(stream),
#else
    thrust::cuda::par.on(stream),
#endif
    cell_count.data(),
    cell_count.data() + N_cells,
    cell_count_sum.data());

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  find_cell_contents<<<grid_size, block_size, 0, stream>>>(
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL
}

void find_neighbor(
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}

void find_neighbor(
  gpuStream_t& stream,
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    stream,
    rc_cell_list,
    num_bins,
    box,
    N,
    position_per_atom,
    cell_count,
    cell_count_sum,
    cell_contents);

  gpu_find_neighbor_ON1<<<grid_size, block_size, 0, stream>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int), stream>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}

// For ILP, the neighbor could not contain atoms in the same layer
static __global__ void gpu_find_neighbor_ON1_ilp(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  int* big_ilp_NN,
  int* big_ilp_NL,
  const int* group_label,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square,
  const double big_ilp_cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  int ilp_count = 0;
  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            // neighbors in different layers
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              bool different_layer = group_label[n1] != group_label[n2];
              if (different_layer && d2 < cutoff_square) {
                NL[count++ * N + n1] = n2;
              } else if (!different_layer && d2 < big_ilp_cutoff_square) {
                big_ilp_NL[ilp_count++ * N + n1] = n2;
              }

            }
          }
        }
      }
    }
    NN[n1] = count;
    big_ilp_NN[n1] = ilp_count;
  }
}

void find_neighbor_ilp(
  const int N1,
  const int N2,
  double rc,
  double big_ilp_cutoff_square,
  Box& box,
  const int* group_label,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL,
  GPU_Vector<int>& big_ilp_NN,
  GPU_Vector<int>& big_ilp_NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_ilp<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    big_ilp_NN.data(),
    big_ilp_NL.data(),
    group_label,
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc,
    big_ilp_cutoff_square);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list_ilp<<<N, min(1024, MN), MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL

  const int big_ilp_MN = big_ilp_NL.size() / big_ilp_NN.size();
  gpu_sort_neighbor_list<<<N, big_ilp_MN, big_ilp_MN * sizeof(int)>>>(N, big_ilp_NN.data(), big_ilp_NL.data());
  GPU_CHECK_KERNEL
}

static __global__ void gpu_find_neighbor_ON1_SW(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const int* group_label,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              if (d2 < cutoff_square && group_label[n1] ==  group_label[n2]) {
                NL[static_cast<size_t>(N) * count++ + n1] = n2;
              }
            }
          }
        }
      }
    }
    NN[n1] = count;
  }
}

void find_neighbor_SW(
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const int* group_label,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_SW<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    group_label,
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}

namespace {

__global__ void gpu_check_atom_distance(
  const Box box,
  int N,
  double d2,
  const double* x_old,
  const double* y_old,
  const double* z_old,
  const double* x_new,
  const double* y_new,
  const double* z_new,
  int* g_sum)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int n = bid * blockDim.x + tid;
  __shared__ int s_sum[128];
  s_sum[tid] = 0;
  if (n < N) {
    float dx = x_new[n] - x_old[n];
    float dy = y_new[n] - y_old[n];
    float dz = z_new[n] - z_old[n];
    apply_mic(box, dx, dy, dz);
    if ((dx * dx + dy * dy + dz * dz) > d2) {
      s_sum[tid] = 1;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(g_sum, s_sum[0]);
  }
}

__global__ void gpu_check_atom_distance_batch(
  const Box box,
  const int N,
  const double d2,
  double* const* x_old_batch,
  double* const* y_old_batch,
  double* const* z_old_batch,
  double* const* position_batch,
  int* rebuild_flags,
  int* any_rebuild)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int rebuild_block;
  if (threadIdx.x == 0) {
    rebuild_block = 0;
  }
  __syncthreads();

  if (n < N) {
    const double* position = position_batch[bead];
    float dx = position[n] - x_old_batch[bead][n];
    float dy = position[n + N] - y_old_batch[bead][n];
    float dz = position[n + N * 2] - z_old_batch[bead][n];
    apply_mic(box, dx, dy, dz);
    if ((dx * dx + dy * dy + dz * dz) > d2) {
      atomicExch(&rebuild_block, 1);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && rebuild_block != 0) {
    atomicExch(&rebuild_flags[bead], 1);
    if (any_rebuild != nullptr) {
      atomicExch(any_rebuild, 1);
    }
  }
}

__device__ int static_s2[1];

__global__ void
gpu_update_xyz0(int N, const double* x, const double* y, const double* z, double* x0, double* y0, double* z0)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    x0[n] = x[n];
    y0[n] = y[n];
    z0[n] = z[n];
  }
}

__global__ void gpu_update_xyz0_batch_if_rebuild(
  const int N,
  double* const* position_batch,
  double* const* x0_batch,
  double* const* y0_batch,
  double* const* z0_batch,
  const int* rebuild_flags)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N || rebuild_flags[bead] == 0) {
    return;
  }
  const double* position = position_batch[bead];
  x0_batch[bead][n] = position[n];
  y0_batch[bead][n] = position[n + N];
  z0_batch[bead][n] = position[n + N * 2];
}

__global__ void gpu_find_local_neighbor_from_global(
  const int N,
  const Box box,
  const float rc_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  int* g_NN_local,
  int* g_NL_local)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];

  int count_local = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[static_cast<size_t>(N) * i1 + n1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;

    if (d12_square >= rc_square) {
      continue;
    }
    g_NL_local[static_cast<size_t>(N) * count_local++ + n1] = n2;
  }

  g_NN_local[n1] = count_local;
}

}

int Neighbor::check_atom_distance(Box& box, const double* x, const double* y, const double* z)
{
  const int N = NN.size();
  double d2 = skin * skin * 0.25;
  int* gpu_s2;
  CHECK(gpuGetSymbolAddress((void**)&gpu_s2, static_s2));
  int cpu_s2[1] = {0};
  CHECK(gpuMemcpy(gpu_s2, cpu_s2, sizeof(int), gpuMemcpyHostToDevice));
  gpu_check_atom_distance<<<(N - 1) / 128 + 1, 128>>>(
    box, N, d2, x0.data(), y0.data(), z0.data(), x, y, z, gpu_s2);
  GPU_CHECK_KERNEL
  CHECK(gpuMemcpy(cpu_s2, gpu_s2, sizeof(int), gpuMemcpyDeviceToHost));
  return cpu_s2[0];
}

void Neighbor::find_neighbor_global(
  const double rc,
  Box& box, 
  const GPU_Vector<int>& type, 
  const GPU_Vector<double>& position_per_atom)
{
  const int N = type.size();
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;

  bool is_first_time = false;

  if (x0.size() == 0) {
    is_first_time = true;
    x0.resize(N);
    y0.resize(N);
    z0.resize(N);
  }

  if (always_rebuild || is_first_time || check_atom_distance(box, x, y, z)) {
    find_neighbor(
      0,
      N,
      rc + skin,
      box, 
      type, 
      position_per_atom,
      cell_count,
      cell_count_sum,
      cell_contents,
      NN,
      NL);

    gpu_update_xyz0<<<(N - 1) / 128 + 1, 128>>>(
      N, 
      x, 
      y, 
      z, 
      x0.data(), 
      y0.data(), 
      z0.data());
    GPU_CHECK_KERNEL
  }
}

void Neighbor::find_neighbor_global_batch(
  const double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const std::vector<GPU_Vector<double>*>& position_beads,
  const std::vector<Neighbor*>& neighbors,
  const GPU_Vector<double*>& position_ptrs,
  GPU_Vector<double*>& x0_batch,
  GPU_Vector<double*>& y0_batch,
  GPU_Vector<double*>& z0_batch,
  GPU_Vector<int>& rebuild_flags,
  GPU_Vector<int>& any_rebuild,
  std::vector<double*>& x0_ptrs_host,
  std::vector<double*>& y0_ptrs_host,
  std::vector<double*>& z0_ptrs_host,
  bool& pointer_arrays_initialized,
  std::vector<gpuStream_t>& rebuild_streams,
  Neighbor_Batch_Timing* timing)
{
  const int number_of_beads = static_cast<int>(neighbors.size());
  const int N = type.size();
  if (
    number_of_beads == 0 || position_beads.size() != neighbors.size() ||
    position_ptrs.size() != static_cast<size_t>(number_of_beads) ||
    x0_batch.size() != static_cast<size_t>(number_of_beads) ||
    y0_batch.size() != static_cast<size_t>(number_of_beads) ||
    z0_batch.size() != static_cast<size_t>(number_of_beads) ||
    rebuild_flags.size() != static_cast<size_t>(number_of_beads) ||
    any_rebuild.size() != 1 ||
    x0_ptrs_host.size() != static_cast<size_t>(number_of_beads) ||
    y0_ptrs_host.size() != static_cast<size_t>(number_of_beads) ||
    z0_ptrs_host.size() != static_cast<size_t>(number_of_beads)) {
    return;
  }

  using Clock = std::chrono::high_resolution_clock;
  const auto pointer_begin = Clock::now();
  std::vector<int> initial_flags(number_of_beads, 0);
  bool need_distance_check = false;
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    Neighbor* neighbor = neighbors[bead_id];
    const bool first = neighbor->prepare_reference_positions(N);
    x0_ptrs_host[bead_id] = neighbor->reference_x_data();
    y0_ptrs_host[bead_id] = neighbor->reference_y_data();
    z0_ptrs_host[bead_id] = neighbor->reference_z_data();
    if (first || neighbor->always_rebuild) {
      initial_flags[bead_id] = 1;
    } else {
      need_distance_check = true;
    }
  }
  if (!pointer_arrays_initialized) {
    x0_batch.copy_from_host(x0_ptrs_host.data());
    y0_batch.copy_from_host(y0_ptrs_host.data());
    z0_batch.copy_from_host(z0_ptrs_host.data());
    pointer_arrays_initialized = true;
  }
  if (timing) {
    CHECK(gpuDeviceSynchronize());
    timing->pointer_setup += std::chrono::duration<double>(Clock::now() - pointer_begin).count();
  }

  const auto flags_begin = Clock::now();
  bool any_rebuild_host = false;
  for (const int flag : initial_flags) {
    if (flag != 0) {
      any_rebuild_host = true;
      break;
    }
  }
  if (any_rebuild_host) {
    rebuild_flags.copy_from_host(initial_flags.data());
    int any_rebuild_value = 1;
    any_rebuild.copy_from_host(&any_rebuild_value);
  } else {
    rebuild_flags.fill(0);
    any_rebuild.fill(0);
  }
  if (timing) {
    CHECK(gpuDeviceSynchronize());
    timing->flag_transfer += std::chrono::duration<double>(Clock::now() - flags_begin).count();
  }

  const auto distance_begin = Clock::now();
  if (need_distance_check && !any_rebuild_host) {
    gpu_check_atom_distance_batch<<<
      dim3((N - 1) / 128 + 1, number_of_beads), 128>>>(
      box,
      N,
      neighbors[0]->skin * neighbors[0]->skin * 0.25,
      x0_batch.data(),
      y0_batch.data(),
      z0_batch.data(),
      position_ptrs.data(),
      rebuild_flags.data(),
      any_rebuild.data());
    GPU_CHECK_KERNEL
  }
  if (timing && need_distance_check && !any_rebuild_host) {
    CHECK(gpuDeviceSynchronize());
    timing->distance_check += std::chrono::duration<double>(Clock::now() - distance_begin).count();
  }

  int any_rebuild_value = any_rebuild_host ? 1 : 0;
  if (!any_rebuild_host && need_distance_check) {
    const auto any_flag_begin = Clock::now();
    any_rebuild.copy_to_host(&any_rebuild_value);
    if (timing) {
      timing->flag_transfer += std::chrono::duration<double>(Clock::now() - any_flag_begin).count();
    }
  }
  if (any_rebuild_value == 0) {
    return;
  }

  std::vector<int> host_flags(number_of_beads, 0);
  const auto flag_copy_begin = Clock::now();
  rebuild_flags.copy_to_host(host_flags.data());
  if (timing) {
    timing->flag_transfer += std::chrono::duration<double>(Clock::now() - flag_copy_begin).count();
  }
  const auto rebuild_begin = Clock::now();
  int rebuild_beads = 0;
  if (rebuild_streams.empty()) {
    for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
      if (host_flags[bead_id] == 0) {
        continue;
      }
      Neighbor* neighbor = neighbors[bead_id];
      find_neighbor(
        0,
        N,
        rc + neighbor->skin,
        box,
        type,
        *position_beads[bead_id],
        neighbor->cell_count,
        neighbor->cell_count_sum,
        neighbor->cell_contents,
        neighbor->NN,
        neighbor->NL);
      const double* position = position_beads[bead_id]->data();
      gpu_update_xyz0<<<(N - 1) / 128 + 1, 128>>>(
        N,
        position,
        position + N,
        position + N * 2,
        neighbor->x0.data(),
        neighbor->y0.data(),
        neighbor->z0.data());
      GPU_CHECK_KERNEL
      ++rebuild_beads;
    }
    if (rebuild_beads > 0) {
      CHECK(gpuDeviceSynchronize());
    }
    if (timing) {
      timing->rebuild_beads += rebuild_beads;
      timing->rebuild += std::chrono::duration<double>(Clock::now() - rebuild_begin).count();
    }
    return;
  }
  // Ensure all per-bead cell-list buffers are sized before any stream starts.
  // Resizing device buffers from concurrent streams would invalidate pointers.
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    if (host_flags[bead_id] == 0) {
      continue;
    }
    Neighbor* neighbor = neighbors[bead_id];
    int num_bins[3];
    box.get_num_bins(0.5 * (rc + neighbor->skin), num_bins);
    const int number_of_cells = num_bins[0] * num_bins[1] * num_bins[2];
    if (number_of_cells > static_cast<int>(neighbor->cell_count.size())) {
      neighbor->cell_count.resize(number_of_cells);
      neighbor->cell_count_sum.resize(number_of_cells);
    }
  }
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    if (host_flags[bead_id] == 0) {
      continue;
    }
    Neighbor* neighbor = neighbors[bead_id];
    gpuStream_t& stream = rebuild_streams[rebuild_beads % rebuild_streams.size()];
    find_neighbor(
      stream,
      0,
      N,
      rc + neighbor->skin,
      box,
      type,
      *position_beads[bead_id],
      neighbor->cell_count,
      neighbor->cell_count_sum,
      neighbor->cell_contents,
      neighbor->NN,
      neighbor->NL);
    const double* position = position_beads[bead_id]->data();
    gpu_update_xyz0<<<(N - 1) / 128 + 1, 128, 0, stream>>>(
      N,
      position,
      position + N,
      position + N * 2,
      neighbor->x0.data(),
      neighbor->y0.data(),
      neighbor->z0.data());
    GPU_CHECK_KERNEL
    ++rebuild_beads;
  }
  if (rebuild_beads > 0) {
    CHECK(gpuDeviceSynchronize());
  }
  if (timing) {
    timing->rebuild_beads += rebuild_beads;
    timing->rebuild += std::chrono::duration<double>(Clock::now() - rebuild_begin).count();
  }
}

void Neighbor::check_atom_distance_batch(
  const Box& box,
  const int number_of_atoms,
  const double skin,
  const GPU_Vector<double*>& x0_batch,
  const GPU_Vector<double*>& y0_batch,
  const GPU_Vector<double*>& z0_batch,
  const GPU_Vector<double*>& position_batch,
  GPU_Vector<int>& rebuild_flags)
{
  const int number_of_beads = static_cast<int>(position_batch.size());
  if (
    number_of_beads == 0 || x0_batch.size() != static_cast<size_t>(number_of_beads) ||
    y0_batch.size() != static_cast<size_t>(number_of_beads) ||
    z0_batch.size() != static_cast<size_t>(number_of_beads) ||
    rebuild_flags.size() != static_cast<size_t>(number_of_beads)) {
    return;
  }
  gpu_check_atom_distance_batch<<<
    dim3((number_of_atoms - 1) / 128 + 1, number_of_beads), 128>>>(
    box,
    number_of_atoms,
    skin * skin * 0.25,
    x0_batch.data(),
    y0_batch.data(),
    z0_batch.data(),
    position_batch.data(),
    rebuild_flags.data(),
    nullptr);
  GPU_CHECK_KERNEL
}

void Neighbor::update_reference_positions_batch(
  const int number_of_atoms,
  const GPU_Vector<double*>& position_batch,
  const GPU_Vector<double*>& x0_batch,
  const GPU_Vector<double*>& y0_batch,
  const GPU_Vector<double*>& z0_batch,
  const GPU_Vector<int>& rebuild_flags)
{
  const int number_of_beads = static_cast<int>(position_batch.size());
  if (
    number_of_beads == 0 || x0_batch.size() != static_cast<size_t>(number_of_beads) ||
    y0_batch.size() != static_cast<size_t>(number_of_beads) ||
    z0_batch.size() != static_cast<size_t>(number_of_beads) ||
    rebuild_flags.size() != static_cast<size_t>(number_of_beads)) {
    return;
  }
  gpu_update_xyz0_batch_if_rebuild<<<
    dim3((number_of_atoms - 1) / 128 + 1, number_of_beads), 128>>>(
    number_of_atoms,
    position_batch.data(),
    x0_batch.data(),
    y0_batch.data(),
    z0_batch.data(),
    rebuild_flags.data());
  GPU_CHECK_KERNEL
}

void Neighbor::find_local_neighbor_from_global(
  const double rc,
  Box& box, 
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& NN_local,
  GPU_Vector<int>& NL_local)
{
  const int N = position_per_atom.size() / 3;
  gpu_find_local_neighbor_from_global<<<(N - 1) / 128 + 1, 128>>>(
    N,
    box,
    rc * rc,
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    NN.data(),
    NL.data(),
    NN_local.data(),
    NL_local.data());
  GPU_CHECK_KERNEL
}

void Neighbor::initialize(const double rc, const int num_atoms, const int num_neighbors)
{
  const double rc_plus_skin = rc + skin;
  const int MN = num_neighbors * rc_plus_skin * rc_plus_skin * rc_plus_skin / (rc * rc * rc);
  NN.resize(num_atoms);
  NL.resize(static_cast<size_t>(num_atoms) * MN);
  cell_count.resize(num_atoms);
  cell_count_sum.resize(num_atoms);
  cell_contents.resize(num_atoms);
}
