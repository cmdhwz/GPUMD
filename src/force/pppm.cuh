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
#include <sstream>
#include <string>
#include <vector>
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
  static constexpr const char* dynamic_q_formula_version()
  {
    return "candidate_v2_real_space";
  }
  void initialize(const float alpha_input);
  void set_mesh_spacing(const double value)
  {
    mesh_spacing = value;
    current_force_mesh_valid_ = false;
  }
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
  void reset_dynamic_charge_cache()
  {
    dynamic_debug_requested_step_ = -1;
    dynamic_q_last_compute_valid_ = false;
    dynamic_q_last_diagnostic_checks_pass_ = false;
    current_force_mesh_valid_ = false;
  }
  void invalidate_current_force_mesh() { current_force_mesh_valid_ = false; }
  void request_dynamic_charge_debug(const int step) { dynamic_debug_requested_step_ = step; }
  void enable_dynamic_charge_diagnostics() { dynamic_diagnostics_enabled_ = true; }
  bool get_last_dynamic_q_diagnostic_checks_pass() const
  {
    return dynamic_q_last_diagnostic_checks_pass_;
  }
  void flush_dynamic_charge_diagnostics();
  bool dynamic_charge_diagnostic_files_are_compatible(const bool check_debug_files) const;
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
    const bool request_peratom_virial = false,
    const unsigned long long force_evaluation_id = 0);
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
  bool diagnose_dynamic_charge(
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
    // The reciprocal correction is returned in eV*Angstrom/natural_time.
    double* delta_j_q_pppm = nullptr);
  bool compute_dynamic_charge_correction(
    const int N,
    const int N1,
    const int N2,
    const Box& box,
    const GPU_Vector<float>& charge,
    const GPU_Vector<float>& charge_rate,
    const GPU_Vector<double>& position,
    // The reciprocal correction is returned in eV*Angstrom/natural_time.
    double* delta_j_q_pppm = nullptr,
    const unsigned long long force_evaluation_id = 0);
  void finalize_dynamic_charge_diagnostic(
    const double* delta_j_q_real,
    const double* delta_j_q_total,
    const bool dynamic_q_valid,
    const int charge_mode);
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
  long long dynamic_call_index_ = 0;
  bool dynamic_diagnostics_enabled_ = false;
  bool dynamic_q_last_compute_valid_ = false;
  bool dynamic_q_last_diagnostic_checks_pass_ = false;
  bool dynamic_debug_written_ = false;
  bool dynamic_csv_row_pending_ = false;
  std::string dynamic_csv_row_;
  int dynamic_debug_requested_step_ = -1;
  std::ostringstream dynamic_csv_buffer_;
  std::ostringstream dynamic_check_buffer_;
  std::string dynamic_atom_debug_buffer_;
  std::string dynamic_kspace_debug_buffer_;
  GPU_Vector<gpufftComplex> dynamic_Q_;
  GPU_Vector<gpufftComplex> dynamic_S_;
  GPU_Vector<gpufftComplex> dynamic_Ax_;
  GPU_Vector<gpufftComplex> dynamic_Ay_;
  GPU_Vector<gpufftComplex> dynamic_Az_;
  GPU_Vector<gpufftComplex> dynamic_Bx_;
  GPU_Vector<gpufftComplex> dynamic_By_;
  GPU_Vector<gpufftComplex> dynamic_Bz_;
  GPU_Vector<gpufftComplex> dynamic_L1S_x_;
  GPU_Vector<gpufftComplex> dynamic_L1S_y_;
  GPU_Vector<gpufftComplex> dynamic_L1S_z_;
  GPU_Vector<double> dynamic_current_total_;
  GPU_Vector<float> dynamic_d_raw_x_;
  GPU_Vector<float> dynamic_d_raw_y_;
  GPU_Vector<float> dynamic_d_raw_z_;
  GPU_Vector<float> dynamic_d_x_;
  GPU_Vector<float> dynamic_d_y_;
  GPU_Vector<float> dynamic_d_z_;
  bool dynamic_operator_cache_valid_ = false;
  int dynamic_operator_N_ = -1;
  int dynamic_operator_K_[3] = {-1, -1, -1};
  float dynamic_operator_alpha_ = 0.0f;
  double dynamic_operator_box_[9] = {0.0};
  bool dynamic_operator_finite_ = false;
  bool dynamic_operator_host_cache_valid_ = false;
  double dynamic_operator_max_odd_error_[3] = {0.0, 0.0, 0.0};
  std::vector<float> dynamic_h_d_x_;
  std::vector<float> dynamic_h_d_y_;
  std::vector<float> dynamic_h_d_z_;
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
  bool current_force_mesh_valid_ = false;
  unsigned long long current_force_mesh_force_evaluation_id_ = 0;
  int current_force_mesh_N_ = -1;
  int current_force_mesh_N1_ = -1;
  int current_force_mesh_N2_ = -1;
  const void* current_force_mesh_charge_ = nullptr;
  const void* current_force_mesh_position_ = nullptr;
  int current_force_mesh_K_[3] = {-1, -1, -1};
  double current_force_mesh_box_[18] = {0.0};
  GPU_Vector<gpufftComplex> mesh_batch;
  GPU_Vector<gpufftComplex> mesh_inverse_batch;
  gpufftHandle plan_batch = 0;
  gpufftHandle plan_inverse_batch = 0;
  int batch_capacity = 0;
  void allocate_memory();
  void allocate_virial_memory();
  void allocate_batch_memory(const int number_of_beads);
  void find_para(const int N, const Box& box);
  bool current_force_mesh_matches(
    const int N,
    const int N1,
    const int N2,
    const Box& box,
    const GPU_Vector<float>& charge,
    const GPU_Vector<double>& position,
    const unsigned long long force_evaluation_id) const;
  void resize_dynamic_charge_workspace(const int M, const bool diagnostic);
  void prepare_dynamic_operator(const int N, const Box& box, const int grid_size);
  void cache_dynamic_operator_on_host();
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
