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
#include "dftd3.cuh"
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>

struct NEP_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> descriptor_parameters_type_pair;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
};

class NEP : public Potential
{
public:
  NEP_Data nep_data;
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    int version = 4; // NEP version, 3 for NEP3 and 4 for NEP4
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    float rc_radial_max = 0.0f;
    float rc_radial_max_inv = 0.0f; 
    float rc_radial[NUM_ELEMENTS];     // radial cutoff
    float rc_angular[NUM_ELEMENTS];    // angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int has_q_222 = 0;
    int has_q_1111 = 0;
    int has_q_112 = 0;
    int has_q_123 = 0;
    int has_q_233 = 0;
    int has_q_134 = 0;
    int num_L;
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
    int num_types = 0;
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c;
    const float* c_type_pair;
    // for the scalar part of polarizability
    const float* w0_pol[10];
    const float* b0_pol[10];
    const float* w1_pol[10];
    const float* b1_pol;
    const float* q_scaler;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct Small_Box_Data {
        GPU_Vector<int> NN_radial;
        GPU_Vector<int> NL_radial;
        GPU_Vector<int> NN_angular;
        GPU_Vector<int> NL_angular;
        GPU_Vector<float> r12;
    } small_box_data;

  NEP(const char* file_potential, const int num_atoms);
  virtual ~NEP(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  bool compute_pimd_batch(
    Box& box,
    const GPU_Vector<int>& type,
    const std::vector<GPU_Vector<double>*>& position_beads,
    const std::vector<GPU_Vector<double>*>& potential_beads,
    const std::vector<GPU_Vector<double>*>& force_beads,
    const std::vector<GPU_Vector<double>*>& virial_beads);

  virtual void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

  virtual void set_neighbor_rebuild(const bool value);
  void set_pimd_batch_profile(const bool enabled) override { pimd_batch_profile_enabled_ = enabled; }
  bool pimd_batch_profile_enabled() const override { return pimd_batch_profile_enabled_; }
  const PIMD_Batch_Timing& get_pimd_batch_timing() const override { return pimd_batch_timing_; }
  void reset_pimd_batch_timing() override { pimd_batch_timing_ = PIMD_Batch_Timing(); }

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  DFTD3 dftd3;
  Neighbor neighbor;

  struct PIMD_Bead_Data
  {
    std::unique_ptr<Neighbor> neighbor;
    GPU_Vector<double> small_box_x0;
    GPU_Vector<double> small_box_y0;
    GPU_Vector<double> small_box_z0;
  };

  struct PIMD_Batch_Data
  {
    int number_of_atoms = 0;
    int number_of_beads = 0;
    std::vector<std::unique_ptr<PIMD_Bead_Data>> beads;
    std::vector<double*> position_ptrs_host;
    std::vector<double*> potential_ptrs_host;
    std::vector<double*> force_ptrs_host;
    std::vector<double*> virial_ptrs_host;
    GPU_Vector<double*> position_ptrs;
    GPU_Vector<double*> potential_ptrs;
    GPU_Vector<double*> force_ptrs;
    GPU_Vector<double*> virial_ptrs;
    GPU_Vector<int*> NN_global_ptrs;
    GPU_Vector<int*> NL_global_ptrs;
    GPU_Vector<double*> x0_ptrs;
    GPU_Vector<double*> y0_ptrs;
    GPU_Vector<double*> z0_ptrs;
    GPU_Vector<int> rebuild_flags;
    GPU_Vector<int> any_rebuild;
    GPU_Vector<int> active_bead_ids;
    GPU_Vector<int> cell_count_batch;
    GPU_Vector<int> cell_count_sum_batch;
    GPU_Vector<int> cell_contents_batch;
    GPU_Vector<int> cell_keys_batch;
    int cell_stride = 0;
    std::vector<double*> x0_ptrs_host;
    std::vector<double*> y0_ptrs_host;
    std::vector<double*> z0_ptrs_host;
    bool pointer_arrays_initialized = false;
    GPU_Vector<double*> small_box_x0_ptrs;
    GPU_Vector<double*> small_box_y0_ptrs;
    GPU_Vector<double*> small_box_z0_ptrs;
    GPU_Vector<int> small_box_rebuild_flags;
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<int> small_NN_radial;
    GPU_Vector<int> small_NL_radial;
    GPU_Vector<int> small_NN_angular;
    GPU_Vector<int> small_NL_angular;
    GPU_Vector<float> small_x12_radial;
    GPU_Vector<float> small_y12_radial;
    GPU_Vector<float> small_z12_radial;
    GPU_Vector<float> small_x12_angular;
    GPU_Vector<float> small_y12_angular;
    GPU_Vector<float> small_z12_angular;
    GPU_Vector<int> small_image_x_radial;
    GPU_Vector<int> small_image_y_radial;
    GPU_Vector<int> small_image_z_radial;
    GPU_Vector<int> small_image_x_angular;
    GPU_Vector<int> small_image_y_angular;
    GPU_Vector<int> small_image_z_angular;
    GPU_Vector<float> Fp;
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> f12x;
    GPU_Vector<float> f12y;
    GPU_Vector<float> f12z;
    std::vector<Neighbor*> neighbor_ptrs;
    bool small_box_data_allocated = false;
    bool small_box_initialized = false;
    double small_box_h[9] = {0.0};
  };

  std::unique_ptr<PIMD_Batch_Data> pimd_batch_data_;
  bool neighbor_always_rebuild_ = false;
  bool pimd_batch_profile_enabled_ = false;
  PIMD_Batch_Timing pimd_batch_timing_;

  void initialize_pimd_batch_(
    int number_of_atoms,
    const std::vector<GPU_Vector<double>*>& position_beads,
    const std::vector<GPU_Vector<double>*>& potential_beads,
    const std::vector<GPU_Vector<double>*>& force_beads,
    const std::vector<GPU_Vector<double>*>& virial_beads,
    bool is_small_box);

  void update_potential(float* parameters, ANN& ann);

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_small_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  bool has_dftd3 = false;
  void initialize_dftd3();
};
