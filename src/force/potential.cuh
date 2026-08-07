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
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <map>
#include <string>

class Potential
{
public:
  // size of the B vector (for each atom) in extrapolation grade calculation
  int B_projection_size = 0;
  // this points to GPU
  double* B_projection = nullptr;
  bool need_B_projection = false;

  int N1;
  int N2;
  double rc; // maximum cutoff distance
  int nep_model_type =
    -1; // -1 for non_nep, 0 for potential, 1 for dipole, 2 for polarizability, 3 for temperature
  int ilp_flag = 0; // 0 for non_ilp, 1 for ilp
  Potential(void);
  virtual ~Potential(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial) = 0;

  virtual void compute(
    const float /* temperature */,
    Box& /* box */,
    const GPU_Vector<int>& /* type */,
    const GPU_Vector<double>& /* position */,
    GPU_Vector<double>& /* potential */,
    GPU_Vector<double>& /* force */,
    GPU_Vector<double>& /* virial */){}

  // add group message for ILPs
  virtual void compute_ilp(
    Box& /* box */,
    const GPU_Vector<int>& /* type */,
    const GPU_Vector<double>& /* position */,
    GPU_Vector<double>& /* potential */,
    GPU_Vector<double>& /* force */,
    GPU_Vector<double>& /* virial */,
    std::vector<Group>& /* group */){}

  // Enable the opt-in fine-grained NEP path. Non-NEP potentials ignore it.
  virtual void set_md_nep_fine_parallel(bool /* enabled */) {}

  // Enable optional NEP/qNEP kernel timing. Non-NEP potentials ignore it.
  virtual void set_md_nep_timing(bool /* enabled */) {}

  virtual const GPU_Vector<int>& get_NN_radial_ptr()
  {
    static GPU_Vector<int> dummy_NN;
    return dummy_NN; // Return the const reference to NN_radial
  }

  virtual const GPU_Vector<int>& get_NL_radial_ptr()
  {
    static GPU_Vector<int> dummy_NL;
    return dummy_NL; // Return the const reference to NL_radial
  }

  virtual GPU_Vector<float>& get_charge_reference()
  {
    static GPU_Vector<float> dummy_charge;
    return dummy_charge;
  }

  virtual GPU_Vector<float>& get_bec_reference()
  {
    static GPU_Vector<float> dummy_bec;
    return dummy_bec;
  }

protected:
  void find_properties_many_body(
    Box& box,
    const int* NN,
    const int* NL,
    const double* f12x,
    const double* f12y,
    const double* f12z,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    const int* reverse_edge = nullptr);

  void find_properties_many_body_segmented(
    Box& box,
    const int* NN,
    const int* NL,
    const float* f12x,
    const float* f12y,
    const float* f12z,
    const bool is_dipole,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    const int* reverse_edge = nullptr);

  void find_properties_many_body_segmented(
    Box& box,
    const int* NN,
    const int* NL,
    const double* f12x,
    const double* f12y,
    const double* f12z,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    const int* reverse_edge = nullptr);

  void find_properties_many_body(
    Box& box,
    const int* NN,
    const int* NL,
    const float* f12x,
    const float* f12y,
    const float* f12z,
    const bool is_dipole,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    const int* reverse_edge = nullptr);

  void build_reverse_edge(
    const int N,
    const int N1,
    const int N2,
    const GPU_Vector<int>& neighbor_number,
    const GPU_Vector<int>& neighbor_list,
    GPU_Vector<int>& reverse_edge);

  void configure_md_nep_timing(bool enabled);
  void md_nep_timing_begin(const char* label);
  void md_nep_timing_end(const char* label);
  void md_nep_timing_begin_step();
  void md_nep_timing_end_step();
  bool md_nep_timing_enabled() const { return md_nep_timing_enabled_; }

private:
  void flush_md_nep_timing();

  bool md_nep_timing_enabled_ = false;
  bool md_nep_timing_events_ready_ = false;
  gpuEvent_t md_nep_timing_start_ = nullptr;
  gpuEvent_t md_nep_timing_stop_ = nullptr;
  size_t md_nep_timing_steps_ = 0;
  std::map<std::string, double> md_nep_timing_total_ms_;
  std::map<std::string, size_t> md_nep_timing_counts_;
};
