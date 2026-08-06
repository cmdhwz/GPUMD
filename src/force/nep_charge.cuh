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
#include "ewald.cuh"
#include "pppm.cuh"
#include <memory>

struct NEP_Charge_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<float> S_real;
  GPU_Vector<float> S_imag;
  GPU_Vector<float> D_real;
  GPU_Vector<float> charge;
  GPU_Vector<float> charge_derivative;
  GPU_Vector<float> bec;               // BEC
};

class NEP_Charge : public Potential
{
public:
  using Potential::compute;

  NEP_Charge_Data nep_data;

  struct ParaMB {
    int charge_mode = 0;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
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
    const float* sqrt_epsilon_inf; // sqrt(epsilon_inf) related to BEC
    const float* b1;               // bias for the output layer
    const float* c;
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

  struct Charge_Para {
    int num_kpoints_max = 1;
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
    float A;
    float B;
  };

  NEP_Charge(const char* file_potential, const int num_atoms);
  virtual ~NEP_Charge(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void compute_non_electro(
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

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

  GPU_Vector<float>& get_charge_reference();

  GPU_Vector<float>& get_bec_reference();

  virtual void set_neighbor_rebuild(const bool value);
  void set_neighbor_diagnostics(const bool enabled) { neighbor_diagnostics_enabled_ = enabled; }

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  DFTD3 dftd3;
  Charge_Para charge_para;
  Ewald ewald;
  PPPM pppm;
  Neighbor neighbor;

  struct PIMD_Bead_Data
  {
    std::unique_ptr<Neighbor> neighbor;
    GPU_Vector<float> charge;
    GPU_Vector<float> D_real;
    GPU_Vector<float> bec;
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
    GPU_Vector<float*> charge_ptrs;
    GPU_Vector<float*> D_real_ptrs;
    GPU_Vector<float*> bec_ptrs;
    GPU_Vector<double*> x0_ptrs;
    GPU_Vector<double*> y0_ptrs;
    GPU_Vector<double*> z0_ptrs;
    GPU_Vector<int> rebuild_flags;
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
    GPU_Vector<float> charge_derivative;
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> f12x;
    GPU_Vector<float> f12y;
    GPU_Vector<float> f12z;
    bool small_box_initialized = false;
    double small_box_h[9] = {0.0};
  };

  std::unique_ptr<PIMD_Batch_Data> pimd_batch_data_;
  bool neighbor_always_rebuild_ = false;
  bool neighbor_diagnostics_enabled_ = true;
  long long neighbor_large_box_calls_ = 0;
  long long neighbor_small_box_calls_ = 0;

  void update_potential(float* parameters, ANN& ann);

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    const bool include_electro = true);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    const bool include_electro = true);

  void find_k_and_G(const double* box);

  bool use_pppm = true; // use PPPM by default
  void check_ewald_pppm();
  bool has_dftd3 = false;
  void initialize_pimd_batch_(
    int number_of_atoms,
    const std::vector<GPU_Vector<double>*>& position_beads,
    const std::vector<GPU_Vector<double>*>& potential_beads,
    const std::vector<GPU_Vector<double>*>& force_beads,
    const std::vector<GPU_Vector<double>*>& virial_beads);
  void initialize_dftd3();
};
