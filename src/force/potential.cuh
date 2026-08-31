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
#include "utilities/gpu_vector.cuh"

struct PIMD_Batch_Timing
{
  double setup = 0.0;
  double neighbor = 0.0;
  double neighbor_global = 0.0;
  double neighbor_filter = 0.0;
  double neighbor_pointer = 0.0;
  double neighbor_check = 0.0;
  double neighbor_flags = 0.0;
  double neighbor_rebuild = 0.0;
  double initialize = 0.0;
  double descriptor = 0.0;
  double bec = 0.0;
  double electrostatics = 0.0;
  double radial = 0.0;
  double angular = 0.0;
  double many_body = 0.0;
  double corrections = 0.0;
  double total = 0.0;
  long long calls = 0;
  long long neighbor_rebuild_beads = 0;
  long long pppm_full_peratom_virial_batch_calls = 0;
  long long pppm_global_virial_batch_calls = 0;
};

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

  virtual void set_neighbor_rebuild(const bool /* value */) {}
  virtual void set_pppm_mesh_spacing(const double /* value */) {}
  virtual void set_md_qnep_bec(const bool /* enabled */) {}

  virtual void set_pimd_batch_profile(const bool /* enabled */) {}
  virtual void set_pimd_batch_bec(const bool /* enabled */) {}
  virtual bool pimd_batch_profile_enabled() const { return false; }
  virtual const PIMD_Batch_Timing& get_pimd_batch_timing() const
  {
    static const PIMD_Batch_Timing dummy;
    return dummy;
  }
  virtual void reset_pimd_batch_timing() {}

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
    GPU_Vector<double>& virial_per_atom);

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
    GPU_Vector<double>& virial_per_atom);
};
