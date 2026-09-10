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

#pragma once
#include "utilities/gpu_vector.cuh"
#include "model/box.cuh"
#include <string>
#ifdef USE_HIP
  #include <hipfft/hipfft.h>
#else
  #include <cufft.h>
#endif

struct PPPMAssignmentAtomDebug
{
  int atom_id;
  float q;
  double x, y, z;
  float sx, sy, sz;
  int ix, iy, iz;
  float dx, dy, dz;
  float Wx[5], Wy[5], Wz[5];
};

struct PPPMAssignmentStencilDebug
{
  int n0, n1, n2;
  int neighbor0, neighbor1, neighbor2;
  int neighbor012;
  float W, qW;
};

class PPPM
{
public:
  PPPM();
  ~PPPM();
  void initialize(const float alpha_input);
  void set_mesh_spacing(const double value) { mesh_spacing = value; }
  double get_mesh_spacing() const { return mesh_spacing; }
  const int* get_mesh() const { return para.K; }
  void request_debug_for_next_force(const char* prefix, const int frame)
  {
    debug_requested_ = true;
    debug_call_index_ = 0;
    debug_prefix_ = prefix;
    debug_frame_ = frame;
  }
  void finish_debug_force_evaluation() { debug_requested_ = false; }
  void find_force(
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
    const bool request_peratom_virial = false);
  void find_force_batch(
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
    const bool request_peratom_virial = false);
  bool last_batch_used_peratom_virial() const { return last_batch_used_peratom_virial_; }
  struct Para {
    int K0K1K2;             // total number of mesh points
    int K0K1;               // K[0] * K[1]
    int K[3];               // number of mesh points in the box vector directions
    int K_half[3];          // K/2
    float alpha;            // The Ewald parameter
    float alpha_factor;     // 1 / (4 * alpha * alpha)
    float two_pi_over_V;    // 4pi/(2V)
    float potential_factor; // K_C_SP / N
    float b[3][3];          // b-vectors in reciprocal space
    float two_pi_over_K[3]; // 2 * pi ./ K
  };
private:
  double mesh_spacing = 1.0;
  Para para;
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<gpufftComplex> mesh;
  GPU_Vector<gpufftComplex> mesh_G;
  GPU_Vector<gpufftComplex> mesh_x;
  GPU_Vector<gpufftComplex> mesh_y;
  GPU_Vector<gpufftComplex> mesh_z;
  bool debug_requested_ = false;
  int debug_call_index_ = 0;
  std::string debug_prefix_;
  int debug_frame_ = 0;
  GPU_Vector<gpufftComplex> debug_mesh_charge_;
  GPU_Vector<gpufftComplex> debug_mesh_fourier_;
  GPU_Vector<PPPMAssignmentAtomDebug> debug_assignment_atoms_;
  GPU_Vector<PPPMAssignmentStencilDebug> debug_assignment_stencil_;
  double debug_mesh_before_assignment_max_real_ = 0.0;
  double debug_mesh_before_assignment_max_imag_ = 0.0;
  double debug_mesh_before_assignment_rms_real_ = 0.0;
  double debug_mesh_before_assignment_rms_imag_ = 0.0;
  double debug_mesh_before_assignment_sum_real_ = 0.0;
  double debug_mesh_after_assignment_sum_real_ = 0.0;
  gpufftHandle plan = 0;
  GPU_Vector<gpufftComplex> mesh_batch;
  GPU_Vector<gpufftComplex> mesh_inverse_batch;
  gpufftHandle plan_batch = 0;
  gpufftHandle plan_inverse_batch = 0;
  int batch_capacity = 0;
  void allocate_memory();
  void allocate_virial_memory();
  void allocate_batch_memory(const int number_of_beads);
  void find_para(const int N, const Box& box);
  void find_k_and_G(const double* box);
  void write_debug(
    const int N,
    const int N1,
    const int N2,
    const Box& box,
    const GPU_Vector<float>& charge,
    const GPU_Vector<double>& position,
    const GPU_Vector<float>& D_real,
    const int pppm_call_index);

  bool need_peratom_virial = false;
  bool need_peratom_virial_every_batch = false;
  bool last_batch_used_peratom_virial_ = false;
  GPU_Vector<gpufftComplex> mesh_virial;
  gpufftHandle plan_virial = 0;
  GPU_Vector<gpufftComplex> mesh_virial_batch;
  gpufftHandle plan_virial_batch = 0;
};
