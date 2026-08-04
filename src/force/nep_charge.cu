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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "nep_charge.cuh"
#include "nep_charge_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP_Charge::check_ewald_pppm()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  use_pppm = true;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "kspace") {
        if (tokens.size() != 2) {
          std::cout << "kspace must have 1 parameter\n";
          exit(1);
        }
        std::string kspace_method = tokens[1];
        if (kspace_method == "ewald") {
          use_pppm = false;
        } else if (kspace_method == "pppm") {
          use_pppm = true;
        } else {
          std::cout << "kspace method can only be ewald or pppm\n";
          exit(1);
        }
      }
    }
  }

  input_run.close();
}

void NEP_Charge::initialize_dftd3()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  has_dftd3 = false;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "dftd3") {
        has_dftd3 = true;
        if (tokens.size() != 4) {
          std::cout << "dftd3 must have 3 parameters\n";
          exit(1);
        }
        std::string xc_functional = tokens[1];
        float rc_potential = get_double_from_token(tokens[2], __FILE__, __LINE__);
        float rc_coordination_number = get_double_from_token(tokens[3], __FILE__, __LINE__);
        dftd3.initialize(xc_functional, rc_potential, rc_coordination_number);
        break;
      }
    }
  }

  input_run.close();
}

NEP_Charge::NEP_Charge(const char* file_potential, const int num_atoms)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4_charge1") {
    zbl.enabled = false;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_zbl_charge1") {
    zbl.enabled = true;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_charge2") {
    zbl.enabled = false;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    zbl.enabled = true;
    paramb.charge_mode = 2;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 charge models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("Use the NEP4-Charge%d potential with %d atom type.\n", 
      paramb.charge_mode, paramb.num_types);
  } else {
    printf("Use the NEP4-Charge%d potential with %d atom types.\n", 
      paramb.charge_mode, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  // cutoff
  tokens = get_tokens(input);
  if (tokens.size() != 5) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular.\n";
    exit(1);
  }
  paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
  printf("    radial cutoff = %g A.\n", paramb.rc_radial);
  printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  printf("    MN_radial = %d.\n", MN_radial);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    std::cout << "This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] [has_q_233] [has_q_134]." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  paramb.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) {
    paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  if (tokens.size() >= 6) {
    paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  }
  if (tokens.size() >= 7) {
    paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  }
  if (tokens.size() >= 8) {
    paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  }
  printf("    has_q_222 = %d.\n", paramb.has_q_222);
  printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  printf("    has_q_112 = %d.\n", paramb.has_q_112);
  printf("    has_q_123 = %d.\n", paramb.has_q_123);
  printf("    has_q_233 = %d.\n", paramb.has_q_233);
  printf("    has_q_134 = %d.\n", paramb.has_q_134);
  if (paramb.has_q_222) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_1111) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_112) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_123) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_233) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_134) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_para_ann = (annmb.dim + 3) * annmb.num_neurons1 * paramb.num_types + 2;

  printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;
  printf("    total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para + annmb.dim);
  for (int n = 0; n < annmb.num_para + annmb.dim; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  // charge related parameters and data
  charge_para.alpha = float(PI) / paramb.rc_radial; // a good value
  check_ewald_pppm();
  if (use_pppm) {
    pppm.initialize(charge_para.alpha);
  } else {
    ewald.initialize(charge_para.alpha);
  }
  charge_para.two_alpha_over_sqrt_pi = 2.0f * charge_para.alpha / sqrt(float(PI));
  charge_para.A = erfc(float(PI)) / (paramb.rc_radial * paramb.rc_radial);
  charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-float(PI * PI)) / paramb.rc_radial;
  charge_para.B = - erfc(float(PI)) / paramb.rc_radial - charge_para.A * paramb.rc_radial;
  nep_data.D_real.resize(num_atoms);
  nep_data.charge.resize(num_atoms);
  nep_data.charge_derivative.resize(num_atoms * annmb.dim);
  nep_data.bec.resize(num_atoms * 9);

  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(num_atoms * annmb.dim);
  nep_data.sum_fxyz.resize(
    num_atoms * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);
  neighbor.initialize(rc, num_atoms, paramb.MN_radial);

  initialize_dftd3();
}

NEP_Charge::~NEP_Charge(void)
{
  // nothing
}

void NEP_Charge::set_neighbor_rebuild(const bool value)
{
  neighbor_always_rebuild_ = value;
  neighbor.set_always_rebuild(value);
  if (pimd_batch_data_) {
    for (auto& bead : pimd_batch_data_->beads) {
      bead->neighbor->set_always_rebuild(value);
    }
  }
}

void NEP_Charge::initialize_pimd_batch_(
  const int number_of_atoms,
  const std::vector<GPU_Vector<double>*>& position_beads,
  const std::vector<GPU_Vector<double>*>& potential_beads,
  const std::vector<GPU_Vector<double>*>& force_beads,
  const std::vector<GPU_Vector<double>*>& virial_beads)
{
  const int number_of_beads = int(position_beads.size());
  const bool needs_allocation =
    !pimd_batch_data_ || pimd_batch_data_->number_of_atoms != number_of_atoms ||
    pimd_batch_data_->number_of_beads != number_of_beads;
  if (needs_allocation) {
    pimd_batch_data_.reset(new PIMD_Batch_Data());
    auto& batch = *pimd_batch_data_;
    batch.number_of_atoms = number_of_atoms;
    batch.number_of_beads = number_of_beads;
    batch.position_ptrs.resize(number_of_beads);
    batch.potential_ptrs.resize(number_of_beads);
    batch.force_ptrs.resize(number_of_beads);
    batch.virial_ptrs.resize(number_of_beads);
    batch.NN_global_ptrs.resize(number_of_beads);
    batch.NL_global_ptrs.resize(number_of_beads);
    batch.charge_ptrs.resize(number_of_beads);
    batch.D_real_ptrs.resize(number_of_beads);
    batch.x0_ptrs.resize(number_of_beads);
    batch.y0_ptrs.resize(number_of_beads);
    batch.z0_ptrs.resize(number_of_beads);
    batch.rebuild_flags.resize(number_of_beads);
    batch.NN_radial.resize(static_cast<size_t>(number_of_beads) * number_of_atoms);
    batch.NL_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * paramb.MN_radial);
    batch.NN_angular.resize(static_cast<size_t>(number_of_beads) * number_of_atoms);
    batch.NL_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * paramb.MN_angular);
    batch.Fp.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * annmb.dim);
    batch.charge_derivative.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * annmb.dim);
    const int sum_components =
      (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1);
    batch.sum_fxyz.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * sum_components);
    const size_t partial_force_size =
      static_cast<size_t>(number_of_beads) * number_of_atoms * paramb.MN_angular;
    batch.f12x.resize(partial_force_size);
    batch.f12y.resize(partial_force_size);
    batch.f12z.resize(partial_force_size);

    std::vector<int*> NN_global_ptrs(number_of_beads);
    std::vector<int*> NL_global_ptrs(number_of_beads);
    std::vector<float*> charge_ptrs(number_of_beads);
    std::vector<float*> D_real_ptrs(number_of_beads);
    std::vector<float*> bec_ptrs(number_of_beads);
    batch.beads.reserve(number_of_beads);
    for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
      std::unique_ptr<PIMD_Bead_Data> bead(new PIMD_Bead_Data());
      bead->neighbor.reset(new Neighbor());
      bead->neighbor->initialize(rc, number_of_atoms, paramb.MN_radial);
      bead->neighbor->set_always_rebuild(neighbor_always_rebuild_);
      bead->charge.resize(number_of_atoms);
      bead->D_real.resize(number_of_atoms);
      bead->bec.resize(static_cast<size_t>(number_of_atoms) * 9);
      NN_global_ptrs[bead_id] = bead->neighbor->NN.data();
      NL_global_ptrs[bead_id] = bead->neighbor->NL.data();
      charge_ptrs[bead_id] = bead->charge.data();
      D_real_ptrs[bead_id] = bead->D_real.data();
      bec_ptrs[bead_id] = bead->bec.data();
      batch.beads.push_back(std::move(bead));
    }
    batch.NN_global_ptrs.copy_from_host(NN_global_ptrs.data());
    batch.NL_global_ptrs.copy_from_host(NL_global_ptrs.data());
    batch.charge_ptrs.copy_from_host(charge_ptrs.data());
    batch.D_real_ptrs.copy_from_host(D_real_ptrs.data());
    batch.bec_ptrs.resize(number_of_beads);
    batch.bec_ptrs.copy_from_host(bec_ptrs.data());
    printf(
      "Using qNEP ring-polymer bead-batched local kernels for %d beads on one GPU.\n",
      number_of_beads);
    fflush(stdout);
  }

  auto& batch = *pimd_batch_data_;
  std::vector<double*> position_ptrs(number_of_beads);
  std::vector<double*> potential_ptrs(number_of_beads);
  std::vector<double*> force_ptrs(number_of_beads);
  std::vector<double*> virial_ptrs(number_of_beads);
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    position_ptrs[bead_id] = position_beads[bead_id]->data();
    potential_ptrs[bead_id] = potential_beads[bead_id]->data();
    force_ptrs[bead_id] = force_beads[bead_id]->data();
    virial_ptrs[bead_id] = virial_beads[bead_id]->data();
  }
  if (position_ptrs != batch.position_ptrs_host) {
    batch.position_ptrs.copy_from_host(position_ptrs.data());
    batch.position_ptrs_host = position_ptrs;
  }
  if (potential_ptrs != batch.potential_ptrs_host) {
    batch.potential_ptrs.copy_from_host(potential_ptrs.data());
    batch.potential_ptrs_host = potential_ptrs;
  }
  if (force_ptrs != batch.force_ptrs_host) {
    batch.force_ptrs.copy_from_host(force_ptrs.data());
    batch.force_ptrs_host = force_ptrs;
  }
  if (virial_ptrs != batch.virial_ptrs_host) {
    batch.virial_ptrs.copy_from_host(virial_ptrs.data());
    batch.virial_ptrs_host = virial_ptrs;
  }
}

void NEP_Charge::update_potential(float* parameters, ANN& ann)
{
  const int num_outputs = 2;
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * num_outputs;
  }
  ann.sqrt_epsilon_inf = pointer;
  pointer += 1;
  ann.b1 = pointer;
  pointer += 1;

  ann.c = pointer;
}

static __global__ void find_neighbor_list_large_box(
  NEP_Charge::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int count_radial = 0;
  int count_angular = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    float rc_radial = paramb.rc_radial;
    float rc_angular = paramb.rc_angular;
    if (d12_square >= rc_radial * rc_radial) {
      continue;
    }
    g_NL_radial[count_radial++ * N + n1] = n2;
    if (d12_square < rc_angular * rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  float* g_charge,
  float* g_charge_derivative,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

      float F = 0.0f, Fp[MAX_DIM] = {0.0f};
      float charge = 0.0f;
      float charge_derivative[MAX_DIM] = {0.0f};

      apply_ann_one_layer_charge(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        charge,
        charge_derivative);

      g_pe[n1] += F;
      g_charge[n1] = charge;

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
        g_charge_derivative[d * N + n1] = charge_derivative[d] * annmb.q_scaler[d];
      }
  }
}

static __global__ void initialize_pimd_batch_properties(
  const int N,
  double* const* g_potential,
  double* const* g_force,
  double* const* g_virial)
{
  const int bead = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }
  double* potential = g_potential[bead];
  double* force = g_force[bead];
  double* virial = g_virial[bead];
  potential[n] = 0.0;
  force[n] = 0.0;
  force[n + N] = 0.0;
  force[n + N * 2] = 0.0;
  for (int component = 0; component < 9; ++component) {
    virial[n + component * N] = 0.0;
  }
}

static __global__ void find_neighbor_list_large_box_pimd_batch(
  NEP_Charge::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  double* const* g_position,
  int* const* g_NN_global_batch,
  int* const* g_NL_global_batch,
  int* g_NN_radial_batch,
  int* g_NL_radial_batch,
  int* g_NN_angular_batch,
  int* g_NL_angular_batch)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  const int* g_NN_global = g_NN_global_batch[bead];
  const int* g_NL_global = g_NL_global_batch[bead];
  int* g_NN_radial = g_NN_radial_batch + static_cast<size_t>(bead) * N;
  int* g_NL_radial =
    g_NL_radial_batch + static_cast<size_t>(bead) * N * paramb.MN_radial;
  int* g_NN_angular = g_NN_angular_batch + static_cast<size_t>(bead) * N;
  int* g_NL_angular =
    g_NL_angular_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  int count_radial = 0;
  int count_angular = 0;
  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    const int n2 = g_NL_global[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    if (d12_square >= paramb.rc_radial * paramb.rc_radial) {
      continue;
    }
    g_NL_radial[count_radial++ * N + n1] = n2;
    if (d12_square < paramb.rc_angular * paramb.rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }
  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor_pimd_batch(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_radial_batch,
  const int* g_NL_radial_batch,
  const int* g_NN_angular_batch,
  const int* g_NL_angular_batch,
  const int* __restrict__ g_type,
  double* const* g_position,
  double* const* g_potential,
  float* g_Fp_batch,
  float* const* g_charge,
  float* g_charge_derivative_batch,
  float* g_sum_fxyz_batch)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  double* g_pe = g_potential[bead];
  const int* g_NN = g_NN_radial_batch + static_cast<size_t>(bead) * N;
  const int* g_NL =
    g_NL_radial_batch + static_cast<size_t>(bead) * N * paramb.MN_radial;
  const int* g_NN_angular = g_NN_angular_batch + static_cast<size_t>(bead) * N;
  const int* g_NL_angular =
    g_NL_angular_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  float* g_Fp = g_Fp_batch + static_cast<size_t>(bead) * N * annmb.dim;
  float* bead_charge = g_charge[bead];
  float* g_charge_derivative =
    g_charge_derivative_batch + static_cast<size_t>(bead) * N * annmb.dim;
  const int sum_components =
    (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1);
  float* g_sum_fxyz =
    g_sum_fxyz_batch + static_cast<size_t>(bead) * N * sum_components;

  const int t1 = g_type[n1];
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  float q[MAX_DIM] = {0.0f};
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    float fc12;
    const int t2 = g_type[n2];
    find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index =
          (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      q[n] += gn12;
    }
  }
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      const int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      const int t2 = g_type[n2];
      find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index =
          (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }
    find_q(
      paramb.L_max,
      paramb.has_q_222,
      paramb.has_q_1111,
      paramb.has_q_112,
      paramb.has_q_123,
      paramb.has_q_233,
      paramb.has_q_134,
      paramb.n_max_angular + 1,
      n,
      s,
      q + (paramb.n_max_radial + 1));
    const int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    for (int abc = 0; abc < num_abc; ++abc) {
      g_sum_fxyz[(n * num_abc + abc) * N + n1] = s[abc];
    }
  }
  for (int d = 0; d < annmb.dim; ++d) {
    q[d] *= annmb.q_scaler[d];
  }
  float energy = 0.0f;
  float Fp[MAX_DIM] = {0.0f};
  float charge = 0.0f;
  float charge_derivative[MAX_DIM] = {0.0f};
  apply_ann_one_layer_charge(
    annmb.dim,
    annmb.num_neurons1,
    annmb.w0[t1],
    annmb.b0[t1],
    annmb.w1[t1],
    annmb.b1,
    q,
    energy,
    Fp,
    charge,
    charge_derivative);
  g_pe[n1] += energy;
  bead_charge[n1] = charge;
  for (int d = 0; d < annmb.dim; ++d) {
    g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
    g_charge_derivative[d * N + n1] = charge_derivative[d] * annmb.q_scaler[d];
  }
}

static __global__ void zero_total_charge(const int N, float* g_charge)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      charge += g_charge[n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_charge[n] -= s_charge[0] / N;
    }
  }
}


// Chain rule correction: zero_total_charge shifted q by -mean(q),
// so D_real must be shifted by -mean(D_real) for consistent forces.
// Uses double accumulator for numerical precision.
static __global__ void zero_mean_D_real(const int N, float* g_D_real)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      sum += (double)g_D_real[n];
    }
  }
  s_sum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  float mean_D = (float)(s_sum[0] / N);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_D_real[n] -= mean_D;
    }
  }
}

static __global__ void find_bec_diagonal(const int N, const float* g_q, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_bec[n1 + N * 0] = g_q[n1];
    g_bec[n1 + N * 1] = 0.0f;
    g_bec[n1 + N * 2] = 0.0f;
    g_bec[n1 + N * 3] = 0.0f;
    g_bec[n1 + N * 4] = g_q[n1];
    g_bec[n1 + N * 5] = 0.0f;
    g_bec[n1 + N * 6] = 0.0f;
    g_bec[n1 + N * 7] = 0.0f;
    g_bec[n1 + N * 8] = g_q[n1];
  }
}

static __global__ void find_bec_radial(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      float bec_xx = 0.5f* (r12[0] * f12[0]);
      float bec_xy = 0.5f* (r12[0] * f12[1]);
      float bec_xz = 0.5f* (r12[0] * f12[2]);
      float bec_yx = 0.5f* (r12[1] * f12[0]);
      float bec_yy = 0.5f* (r12[1] * f12[1]);
      float bec_yz = 0.5f* (r12[1] * f12[2]);
      float bec_zx = 0.5f* (r12[2] * f12[0]);
      float bec_zy = 0.5f* (r12[2] * f12[1]);
      float bec_zz = 0.5f* (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_bec_angular(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }

      float bec_xx = 0.5f* (r12[0] * f12[0]);
      float bec_xy = 0.5f* (r12[0] * f12[1]);
      float bec_xz = 0.5f* (r12[0] * f12[2]);
      float bec_yx = 0.5f* (r12[1] * f12[0]);
      float bec_yy = 0.5f* (r12[1] * f12[1]);
      float bec_yz = 0.5f* (r12[1] * f12[2]);
      float bec_zx = 0.5f* (r12[2] * f12[0]);
      float bec_zy = 0.5f* (r12[2] * f12[1]);
      float bec_zz = 0.5f* (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void scale_bec(const int N, const float* sqrt_epsilon_inf, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

static __global__ void find_force_radial(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* g_charge_derivative,
  const float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
          gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
        }
        float tmp12 = g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1];
        float tmp21 = g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * g_D_real[n2];
        tmp12 *= gnp12 * d12inv;
        tmp21 *= gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx += r12[0] * f21[0];
      s_syy += r12[1] * f21[1];
      s_szz += r12[2] * f21[2];
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
  }
}

static __global__ void find_force_radial_pimd_batch(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_radial_batch,
  const int* g_NL_radial_batch,
  const int* __restrict__ g_type,
  double* const* g_position,
  const float* g_Fp_batch,
  const float* g_charge_derivative_batch,
  float* const* g_D_real,
  double* const* g_force,
  double* const* g_virial)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  double* force = g_force[bead];
  double* g_fx = force;
  double* g_fy = force + N;
  double* g_fz = force + N * 2;
  double* bead_virial = g_virial[bead];
  const int* g_NN = g_NN_radial_batch + static_cast<size_t>(bead) * N;
  const int* g_NL =
    g_NL_radial_batch + static_cast<size_t>(bead) * N * paramb.MN_radial;
  const float* g_Fp = g_Fp_batch + static_cast<size_t>(bead) * N * annmb.dim;
  const float* g_charge_derivative =
    g_charge_derivative_batch + static_cast<size_t>(bead) * N * annmb.dim;
  const float* bead_D_real = g_D_real[bead];

  const int t1 = g_type[n1];
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    const int t2 = g_type[n2];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float r12[3] = {x12, y12, z12};
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    const float d12inv = 1.0f / d12;
    float f12[3] = {0.0f};
    float f21[3] = {0.0f};
    float fc12;
    float fcp12;
    find_fc_and_fcp(
      paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      paramb.basis_size_radial,
      paramb.rcinv_radial,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gnp12 = 0.0f;
      float gnp21 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        const int c_index =
          (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
        gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
      }
      float tmp12 =
        g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * bead_D_real[n1];
      float tmp21 =
        g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * bead_D_real[n2];
      tmp12 *= gnp12 * d12inv;
      tmp21 *= gnp21 * d12inv;
      for (int d = 0; d < 3; ++d) {
        f12[d] += tmp12 * r12[d];
        f21[d] -= tmp21 * r12[d];
      }
    }
    s_fx += f12[0] - f21[0];
    s_fy += f12[1] - f21[1];
    s_fz += f12[2] - f21[2];
    s_sxx += r12[0] * f21[0];
    s_syy += r12[1] * f21[1];
    s_szz += r12[2] * f21[2];
    s_sxy += r12[0] * f21[1];
    s_sxz += r12[0] * f21[2];
    s_syx += r12[1] * f21[0];
    s_syz += r12[1] * f21[2];
    s_szx += r12[2] * f21[0];
    s_szy += r12[2] * f21[1];
  }
  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;
  bead_virial[n1 + 0 * N] += s_sxx;
  bead_virial[n1 + 1 * N] += s_syy;
  bead_virial[n1 + 2 * N] += s_szz;
  bead_virial[n1 + 3 * N] += s_sxy;
  bead_virial[n1 + 4 * N] += s_sxz;
  bead_virial[n1 + 5 * N] += s_syz;
  bead_virial[n1 + 6 * N] += s_syx;
  bead_virial[n1 + 7 * N] += s_szx;
  bead_virial[n1 + 8 * N] += s_szy;
}

static __global__ void find_partial_force_angular(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* g_charge_derivative,
  const float* g_D_real,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      float tmp = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1] 
        + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
      Fp[d] = tmp;
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_partial_force_angular_pimd_batch(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular_batch,
  const int* g_NL_angular_batch,
  const int* __restrict__ g_type,
  double* const* g_position,
  const float* g_Fp_batch,
  const float* g_charge_derivative_batch,
  float* const* g_D_real,
  const float* g_sum_fxyz_batch,
  float* g_f12x_batch,
  float* g_f12y_batch,
  float* g_f12z_batch)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  const int* g_NN_angular = g_NN_angular_batch + static_cast<size_t>(bead) * N;
  const int* g_NL_angular =
    g_NL_angular_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const float* g_Fp = g_Fp_batch + static_cast<size_t>(bead) * N * annmb.dim;
  const float* g_charge_derivative =
    g_charge_derivative_batch + static_cast<size_t>(bead) * N * annmb.dim;
  const float* bead_D_real = g_D_real[bead];
  const int sum_components =
    (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1);
  const float* g_sum_fxyz =
    g_sum_fxyz_batch + static_cast<size_t>(bead) * N * sum_components;
  float* g_f12x =
    g_f12x_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  float* g_f12y =
    g_f12y_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  float* g_f12z =
    g_f12z_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;

  float Fp[MAX_DIM_ANGULAR] = {0.0f};
  float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1] +
            g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] *
              bead_D_real[n1];
  }
  const int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
    for (int abc = 0; abc < num_abc; ++abc) {
      sum_fxyz[n * NUM_OF_ABC + abc] =
        g_sum_fxyz[(n * num_abc + abc) * N + n1];
    }
  }
  const int t1 = g_type[n1];
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL_angular[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float r12[3] = {x12, y12, z12};
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    float f12[3] = {0.0f};
    float fc12;
    float fcp12;
    const int t2 = g_type[n2];
    find_fc_and_fcp(
      paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      paramb.basis_size_angular,
      paramb.rcinv_angular,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index =
          (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
        gnp12 += fnp12[k] * annmb.c[c_index];
      }
      accumulate_f12(
        paramb.L_max,
        paramb.has_q_222,
        paramb.has_q_1111,
        paramb.has_q_112,
        paramb.has_q_123,
        paramb.has_q_233,
        paramb.has_q_134,
        paramb.num_L,
        n,
        paramb.n_max_angular + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp,
        sum_fxyz,
        f12);
    }
    g_f12x[index] = f12[0];
    g_f12y[index] = f12[1];
    g_f12z[index] = f12[2];
  }
}

static __global__ void find_force_many_body_pimd_batch(
  NEP_Charge::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular_batch,
  const int* g_NL_angular_batch,
  const float* g_f12x_batch,
  const float* g_f12y_batch,
  const float* g_f12z_batch,
  double* const* g_position,
  double* const* g_force,
  double* const* g_virial)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }
  const int* g_NN = g_NN_angular_batch + static_cast<size_t>(bead) * N;
  const int* g_NL =
    g_NL_angular_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const float* g_f12x =
    g_f12x_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const float* g_f12y =
    g_f12y_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const float* g_f12z =
    g_f12z_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  double* force = g_force[bead];
  double* g_fx = force;
  double* g_fy = force + N;
  double* g_fz = force + N * 2;
  double* bead_virial = g_virial[bead];

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    int index = i1 * N + n1;
    const int n2 = g_NL[index];
    double x12_double = g_x[n2] - x1;
    double y12_double = g_y[n2] - y1;
    double z12_double = g_z[n2] - z1;
    apply_mic(box, x12_double, y12_double, z12_double);
    const float x12 = float(x12_double);
    const float y12 = float(y12_double);
    const float z12 = float(z12_double);
    const float f12x = g_f12x[index];
    const float f12y = g_f12y[index];
    const float f12z = g_f12z[index];
    int left = 0;
    int right = g_NN[n2];
    while (left < right) {
      const int middle = (left + right) >> 1;
      const int value = g_NL[n2 + N * middle];
      if (value < n1) {
        left = middle + 1;
      } else if (value > n1) {
        right = middle - 1;
      } else {
        left = middle;
        right = middle;
      }
    }
    index = ((left + right) >> 1) * N + n2;
    const float f21x = g_f12x[index];
    const float f21y = g_f12y[index];
    const float f21z = g_f12z[index];
    s_fx += f12x - f21x;
    s_fy += f12y - f21y;
    s_fz += f12z - f21z;
    s_sxx += x12 * f21x;
    s_sxy += x12 * f21y;
    s_sxz += x12 * f21z;
    s_syx += y12 * f21x;
    s_syy += y12 * f21y;
    s_syz += y12 * f21z;
    s_szx += z12 * f21x;
    s_szy += z12 * f21y;
    s_szz += z12 * f21z;
  }
  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;
  bead_virial[n1 + 0 * N] += s_sxx;
  bead_virial[n1 + 1 * N] += s_syy;
  bead_virial[n1 + 2 * N] += s_szz;
  bead_virial[n1 + 3 * N] += s_sxy;
  bead_virial[n1 + 4 * N] += s_sxz;
  bead_virial[n1 + 5 * N] += s_syz;
  bead_virial[n1 + 6 * N] += s_syx;
  bead_virial[n1 + 7 * N] += s_szx;
  bead_virial[n1 + 8 * N] += s_szy;
}

static __global__ void find_force_ZBL(
  NEP_Charge::ParaMB paramb,
  const int N,
  const NEP_Charge::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}

static __global__ void find_force_charge_real_space(
  const int N,
  const NEP_Charge::Charge_Para charge_para,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  float* g_D_real)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q1 = g_charge[n1];
    float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1; // self energy part
    float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi; // self energy part

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float q2 = g_charge[n2];
      float qq = q1 * q2;
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      s_pe += 0.5f * qq * erfc_r;
      float f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

static __global__ void find_force_charge_real_space_pimd_batch(
  const int N,
  const int MN_radial,
  const NEP_Charge::Charge_Para charge_para,
  const int N1,
  const int N2,
  const int number_of_beads,
  const Box box,
  const int* g_NN_batch,
  const int* g_NL_batch,
  float* const* g_charge,
  double* const* g_position,
  double* const* g_force,
  double* const* g_virial,
  double* const* g_pe,
  float* const* g_D_real)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (bead >= number_of_beads || n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  double* force = g_force[bead];
  double* virial = g_virial[bead];
  double* pe = g_pe[bead];
  float* D_real_out = g_D_real[bead];
  const float* charge = g_charge[bead];
  const int* g_NN = g_NN_batch + static_cast<size_t>(bead) * N;
  const int* g_NL = g_NL_batch + static_cast<size_t>(bead) * N * MN_radial;
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;
  const double x1 = position[n1];
  const double y1 = position[n1 + N];
  const double z1 = position[n1 + N * 2];
  const float q1 = charge[n1];
  float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1;
  float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi;
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    const float q2 = charge[n2];
    const float qq = q1 * q2;
    float x12 = position[n2] - x1;
    float y12 = position[n2 + N] - y1;
    float z12 = position[n2 + N * 2] - z1;
    apply_mic(box, x12, y12, z12);
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    const float d12inv = 1.0f / d12;
    const float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
    D_real += q2 * erfc_r;
    s_pe += 0.5f * qq * erfc_r;
    float f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(
      -charge_para.alpha * charge_para.alpha * d12 * d12);
    f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
    const float fx = x12 * f2;
    const float fy = y12 * f2;
    const float fz = z12 * f2;
    s_fx += 2.0f * fx;
    s_fy += 2.0f * fy;
    s_fz += 2.0f * fz;
    s_sxx -= x12 * fx;
    s_sxy -= x12 * fy;
    s_sxz -= x12 * fz;
    s_syx -= y12 * fx;
    s_syy -= y12 * fy;
    s_syz -= y12 * fz;
    s_szx -= z12 * fx;
    s_szy -= z12 * fy;
    s_szz -= z12 * fz;
  }
  force[n1] += s_fx;
  force[n1 + N] += s_fy;
  force[n1 + N * 2] += s_fz;
  virial[n1 + 0 * N] += s_sxx;
  virial[n1 + 1 * N] += s_syy;
  virial[n1 + 2 * N] += s_szz;
  virial[n1 + 3 * N] += s_sxy;
  virial[n1 + 4 * N] += s_sxz;
  virial[n1 + 5 * N] += s_syz;
  virial[n1 + 6 * N] += s_syx;
  virial[n1 + 7 * N] += s_szx;
  virial[n1 + 8 * N] += s_szy;
  D_real_out[n1] += K_C_SP * D_real;
  pe[n1] += K_C_SP * s_pe;
}

// large box fo MD applications
void NEP_Charge::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  const bool include_electro)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor.find_neighbor_global(
    rc,
    box, 
    type, 
    position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    neighbor.NN.data(),
    neighbor.NL.data(),
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  long long& num_calls = neighbor_large_box_calls_;
  if (neighbor_diagnostics_enabled_ && num_calls++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    nep_data.charge.data(),
    nep_data.charge_derivative.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  if (include_electro) {
    zero_total_charge<<<1, 1024>>>(N, nep_data.charge.data());
    GPU_CHECK_KERNEL

    // get BEC (the diagonal part)
    find_bec_diagonal<<<grid_size, BLOCK_SIZE>>>(
      N,
      nep_data.charge.data(),
      nep_data.bec.data());
    GPU_CHECK_KERNEL

    // get BEC (radial descriptor part)
    find_bec_radial<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.charge_derivative.data(),
      nep_data.bec.data());
    GPU_CHECK_KERNEL

    // get BEC (angular descriptor part)
    find_bec_angular<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data(),
      nep_data.bec.data());
    GPU_CHECK_KERNEL

    // scale q to q * sqrt(epsilon_inf)
    scale_bec<<<grid_size, BLOCK_SIZE>>>(
      N,
      annmb.sqrt_epsilon_inf,
      nep_data.bec.data());
    GPU_CHECK_KERNEL
    if (use_pppm) {
      pppm.find_force(
        N,
        N1,
        N2,
        box,
        nep_data.charge,
        position_per_atom,
        nep_data.D_real,
        force_per_atom,
        virial_per_atom,
        potential_per_atom);
    } else {
      ewald.find_force(
        N,
        N1,
        N2,
        box.cpu_h,
        nep_data.charge,
        position_per_atom,
        nep_data.D_real,
        force_per_atom,
        virial_per_atom,
        potential_per_atom);
    }

    if (paramb.charge_mode == 1) {
      find_force_charge_real_space<<<grid_size, BLOCK_SIZE>>>(
        N,
        charge_para,
        N1,
        N2,
        box,
        nep_data.NN_radial.data(),
        nep_data.NL_radial.data(),
        nep_data.charge.data(),
        position_per_atom.data(),
        position_per_atom.data() + N,
        position_per_atom.data() + N * 2,
        force_per_atom.data(),
        force_per_atom.data() + N,
        force_per_atom.data() + N * 2,
        virial_per_atom.data(),
        potential_per_atom.data(),
        nep_data.D_real.data());
      GPU_CHECK_KERNEL
    }

    zero_mean_D_real<<<1, 1024>>>(N, nep_data.D_real.data());
    GPU_CHECK_KERNEL
  } else {
    CHECK(gpuMemset(nep_data.D_real.data(), 0, sizeof(float) * N));
  }

  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body(
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    false,
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP_Charge::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  const bool include_electro)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  long long& num_calls = neighbor_small_box_calls_;
  if (neighbor_diagnostics_enabled_ && num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < cpu_NN_radial[n]) {
        radial_actual = cpu_NN_radial[n];
      }
      if (angular_actual < cpu_NN_angular[n]) {
        angular_actual = cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    nep_data.charge.data(),
    nep_data.charge_derivative.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  if (include_electro) {
    zero_total_charge<<<N, 1024>>>(N, nep_data.charge.data());
    GPU_CHECK_KERNEL

    // get BEC (the diagonal part)
    find_bec_diagonal<<<grid_size, BLOCK_SIZE>>>(
      N,
      nep_data.charge.data(),
      nep_data.bec.data());
    GPU_CHECK_KERNEL

    // get BEC (radial descriptor part)
    find_bec_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      small_box_data.NN_radial.data(),
      small_box_data.NL_radial.data(),
      type.data(),
      small_box_data.r12.data(),
      small_box_data.r12.data() + size_x12,
      small_box_data.r12.data() + size_x12 * 2,
      nep_data.charge_derivative.data(),
      nep_data.bec.data());
    GPU_CHECK_KERNEL

    // get BEC (angular descriptor part)
    find_bec_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data(),
      nep_data.bec.data());
    GPU_CHECK_KERNEL

    // scale q to q * sqrt(epsilon_inf)
    scale_bec<<<grid_size, BLOCK_SIZE>>>(
      N,
      annmb.sqrt_epsilon_inf,
      nep_data.bec.data());
    GPU_CHECK_KERNEL
    if (use_pppm) {
      pppm.find_force(
        N,
        N1,
        N2,
        box,
        nep_data.charge,
        position_per_atom,
        nep_data.D_real,
        force_per_atom,
        virial_per_atom,
        potential_per_atom);
    } else {
      ewald.find_force(
        N,
        N1,
        N2,
        box.cpu_h,
        nep_data.charge,
        position_per_atom,
        nep_data.D_real,
        force_per_atom,
        virial_per_atom,
        potential_per_atom);
    }

    if (paramb.charge_mode == 1) {
      find_force_charge_real_space_small_box<<<grid_size, BLOCK_SIZE>>>(
        N,
        charge_para,
        N1,
        N2,
        box,
        small_box_data.NN_radial.data(),
        small_box_data.NL_radial.data(),
        nep_data.charge.data(),
        small_box_data.r12.data(),
        small_box_data.r12.data() + size_x12,
        small_box_data.r12.data() + size_x12 * 2,
        force_per_atom.data(),
        force_per_atom.data() + N,
        force_per_atom.data() + N * 2,
        virial_per_atom.data(),
        potential_per_atom.data(),
        nep_data.D_real.data());
      GPU_CHECK_KERNEL
    }

    zero_mean_D_real<<<1, 1024>>>(N, nep_data.D_real.data());
    GPU_CHECK_KERNEL
  } else {
    CHECK(gpuMemset(nep_data.D_real.data(), 0, sizeof(float) * N));
  }

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.sum_fxyz.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP_Charge::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }

  if (is_small_box) {
    if (thickness_x > 10 * rc || thickness_y > 10 * rc || thickness_z > 10 * rc) {
      std::cout << "Error:\n"
                << "    The box has\n"
                << "        a thickness < 2.5 radial cutoffs in a periodic direction.\n"
                << "        and a thickness > 10 radial cutoffs in another direction.\n"
                << "    Please increase the periodic direction(s).\n";
      exit(1);
    }

    ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
    ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
    ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
    ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
    ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
    ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
    ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
    ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
    ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

    ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
    ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
    ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
    ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
    ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
    ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
    ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
    ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
    ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
    double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                 ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                 ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
    for (int n = 9; n < 18; n++) {
      ebox.h[n] /= det;
    }
  }

  return is_small_box;
}

void NEP_Charge::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
    PRINT_INPUT_ERROR("Cannot use non-periodic boundaries for qNEP models.");
  }

  const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);
  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
      const int big_neighbor_size = 2000;
      const int size_x12 = current_num_atoms * big_neighbor_size;

      small_box_data.NN_radial.resize(current_num_atoms);
      small_box_data.NL_radial.resize(size_x12);
      small_box_data.NN_angular.resize(current_num_atoms);
      small_box_data.NL_angular.resize(size_x12);
      small_box_data.r12.resize(size_x12 * 6);
    }
    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  } else {
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

static __global__ void zero_total_charge_pimd_batch(
  const int N,
  const int number_of_beads,
  float* const* g_charge);
static __global__ void zero_mean_D_real_pimd_batch(
  const int N,
  const int number_of_beads,
  float* const* g_D_real);
static __global__ void find_bec_diagonal_pimd_batch(
  const int N,
  const int number_of_beads,
  float* const* g_charge,
  float* const* g_bec);
static __global__ void find_bec_radial_pimd_batch(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int number_of_beads,
  const Box box,
  const int* g_NN_batch,
  const int* g_NL_batch,
  const int* g_type,
  double* const* g_position,
  const float* g_charge_derivative_batch,
  float* const* g_bec);
static __global__ void find_bec_angular_pimd_batch(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int number_of_beads,
  const Box box,
  const int* g_NN_batch,
  const int* g_NL_batch,
  const int* g_type,
  double* const* g_position,
  const float* g_charge_derivative_batch,
  const float* g_sum_fxyz_batch,
  float* const* g_bec);
static __global__ void scale_bec_pimd_batch(
  const int N,
  const int number_of_beads,
  const float* sqrt_epsilon_inf,
  float* const* g_bec);

bool NEP_Charge::compute_pimd_batch(
  Box& box,
  const GPU_Vector<int>& type,
  const std::vector<GPU_Vector<double>*>& position_beads,
  const std::vector<GPU_Vector<double>*>& potential_beads,
  const std::vector<GPU_Vector<double>*>& force_beads,
  const std::vector<GPU_Vector<double>*>& virial_beads)
{
  const int number_of_beads = int(position_beads.size());
  if (
    number_of_beads < 2 || potential_beads.size() != position_beads.size() ||
    force_beads.size() != position_beads.size() ||
    virial_beads.size() != position_beads.size()) {
    return false;
  }
  if (get_expanded_box(paramb.rc_radial, box, ebox)) {
    return false;
  }

  const int N = type.size();
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    if (
      position_beads[bead_id]->size() != static_cast<size_t>(N) * 3 ||
      potential_beads[bead_id]->size() != static_cast<size_t>(N) ||
      force_beads[bead_id]->size() != static_cast<size_t>(N) * 3 ||
      virial_beads[bead_id]->size() != static_cast<size_t>(N) * 9) {
      return false;
    }
  }

  initialize_pimd_batch_(
    N, position_beads, potential_beads, force_beads, virial_beads);
  auto& batch = *pimd_batch_data_;
  std::vector<Neighbor*> neighbor_ptrs;
  neighbor_ptrs.reserve(number_of_beads);
  for (auto& bead : batch.beads) {
    neighbor_ptrs.push_back(bead->neighbor.get());
  }
  Neighbor::find_neighbor_global_batch(
    rc,
    box,
    type,
    position_beads,
    neighbor_ptrs,
    batch.position_ptrs,
    batch.x0_ptrs,
    batch.y0_ptrs,
    batch.z0_ptrs,
    batch.rebuild_flags);

  const int block_size = 64;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const dim3 grid(grid_size, number_of_beads);
  initialize_pimd_batch_properties<<<dim3((N - 1) / 128 + 1, number_of_beads), 128>>>(
    N,
    batch.potential_ptrs.data(),
    batch.force_ptrs.data(),
    batch.virial_ptrs.data());
  GPU_CHECK_KERNEL

  find_neighbor_list_large_box_pimd_batch<<<grid, block_size>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    batch.position_ptrs.data(),
    batch.NN_global_ptrs.data(),
    batch.NL_global_ptrs.data(),
    batch.NN_radial.data(),
    batch.NL_radial.data(),
    batch.NN_angular.data(),
    batch.NL_angular.data());
  GPU_CHECK_KERNEL

  find_descriptor_pimd_batch<<<grid, block_size>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    batch.NN_radial.data(),
    batch.NL_radial.data(),
    batch.NN_angular.data(),
    batch.NL_angular.data(),
    type.data(),
    batch.position_ptrs.data(),
    batch.potential_ptrs.data(),
    batch.Fp.data(),
    batch.charge_ptrs.data(),
    batch.charge_derivative.data(),
    batch.sum_fxyz.data());
  GPU_CHECK_KERNEL

  const dim3 bead_grid(grid_size, number_of_beads);
  zero_total_charge_pimd_batch<<<number_of_beads, 1024>>>(
    N, number_of_beads, batch.charge_ptrs.data());
  GPU_CHECK_KERNEL
  find_bec_diagonal_pimd_batch<<<bead_grid, block_size>>>(
    N, number_of_beads, batch.charge_ptrs.data(), batch.bec_ptrs.data());
  GPU_CHECK_KERNEL
  find_bec_radial_pimd_batch<<<bead_grid, block_size>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    number_of_beads,
    box,
    batch.NN_radial.data(),
    batch.NL_radial.data(),
    type.data(),
    batch.position_ptrs.data(),
    batch.charge_derivative.data(),
    batch.bec_ptrs.data());
  GPU_CHECK_KERNEL
  find_bec_angular_pimd_batch<<<bead_grid, block_size>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    number_of_beads,
    box,
    batch.NN_angular.data(),
    batch.NL_angular.data(),
    type.data(),
    batch.position_ptrs.data(),
    batch.charge_derivative.data(),
    batch.sum_fxyz.data(),
    batch.bec_ptrs.data());
  GPU_CHECK_KERNEL
  scale_bec_pimd_batch<<<bead_grid, block_size>>>(
    N, number_of_beads, annmb.sqrt_epsilon_inf, batch.bec_ptrs.data());
  GPU_CHECK_KERNEL

  if (use_pppm) {
    pppm.find_force_batch(
      N,
      N1,
      N2,
      box,
      batch.charge_ptrs,
      batch.position_ptrs,
      batch.D_real_ptrs,
      batch.force_ptrs,
      batch.virial_ptrs,
      batch.potential_ptrs,
      number_of_beads);
  } else {
    for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
      auto& bead = *batch.beads[bead_id];
      ewald.find_force(
        N,
        N1,
        N2,
        box.cpu_h,
        bead.charge,
        *position_beads[bead_id],
        bead.D_real,
        *force_beads[bead_id],
        *virial_beads[bead_id],
        *potential_beads[bead_id]);
    }
  }
  if (paramb.charge_mode == 1) {
    find_force_charge_real_space_pimd_batch<<<bead_grid, block_size>>>(
      N,
      paramb.MN_radial,
      charge_para,
      N1,
      N2,
      number_of_beads,
      box,
      batch.NN_radial.data(),
      batch.NL_radial.data(),
      batch.charge_ptrs.data(),
      batch.position_ptrs.data(),
      batch.force_ptrs.data(),
      batch.virial_ptrs.data(),
      batch.potential_ptrs.data(),
      batch.D_real_ptrs.data());
    GPU_CHECK_KERNEL
  }
  zero_mean_D_real_pimd_batch<<<number_of_beads, 1024>>>(
    N, number_of_beads, batch.D_real_ptrs.data());
  GPU_CHECK_KERNEL

  find_force_radial_pimd_batch<<<grid, block_size>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    batch.NN_radial.data(),
    batch.NL_radial.data(),
    type.data(),
    batch.position_ptrs.data(),
    batch.Fp.data(),
    batch.charge_derivative.data(),
    batch.D_real_ptrs.data(),
    batch.force_ptrs.data(),
    batch.virial_ptrs.data());
  GPU_CHECK_KERNEL
  find_partial_force_angular_pimd_batch<<<grid, block_size>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    batch.NN_angular.data(),
    batch.NL_angular.data(),
    type.data(),
    batch.position_ptrs.data(),
    batch.Fp.data(),
    batch.charge_derivative.data(),
    batch.D_real_ptrs.data(),
    batch.sum_fxyz.data(),
    batch.f12x.data(),
    batch.f12y.data(),
    batch.f12z.data());
  GPU_CHECK_KERNEL
  find_force_many_body_pimd_batch<<<grid, block_size>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    batch.NN_angular.data(),
    batch.NL_angular.data(),
    batch.f12x.data(),
    batch.f12y.data(),
    batch.f12z.data(),
    batch.position_ptrs.data(),
    batch.force_ptrs.data(),
    batch.virial_ptrs.data());
  GPU_CHECK_KERNEL

  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    const size_t atom_offset = static_cast<size_t>(bead_id) * N;
    const size_t angular_offset = atom_offset * paramb.MN_angular;
    if (zbl.enabled) {
      find_force_ZBL<<<grid_size, block_size>>>(
        paramb,
        N,
        zbl,
        N1,
        N2,
        box,
        batch.NN_angular.data() + atom_offset,
        batch.NL_angular.data() + angular_offset,
        type.data(),
        position_beads[bead_id]->data(),
        position_beads[bead_id]->data() + N,
        position_beads[bead_id]->data() + N * 2,
        force_beads[bead_id]->data(),
        force_beads[bead_id]->data() + N,
        force_beads[bead_id]->data() + N * 2,
        virial_beads[bead_id]->data(),
        potential_beads[bead_id]->data());
      GPU_CHECK_KERNEL
    }
    if (has_dftd3) {
      dftd3.compute(
        box,
        type,
        *position_beads[bead_id],
        *potential_beads[bead_id],
        *force_beads[bead_id],
        *virial_beads[bead_id]);
    }
  }
  return true;
}

static __global__ void zero_total_charge_pimd_batch(
  const int N,
  const int number_of_beads,
  float* const* g_charge)
{
  const int bead = blockIdx.x;
  const int tid = threadIdx.x;
  if (bead >= number_of_beads) {
    return;
  }
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  const int number_of_batches = (N - 1) / 1024 + 1;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      charge += g_charge[bead][n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      g_charge[bead][n] -= s_charge[0] / N;
    }
  }
}

static __global__ void zero_mean_D_real_pimd_batch(
  const int N,
  const int number_of_beads,
  float* const* g_D_real)
{
  const int bead = blockIdx.x;
  const int tid = threadIdx.x;
  if (bead >= number_of_beads) {
    return;
  }
  __shared__ double s_sum[1024];
  double sum = 0.0;
  const int number_of_batches = (N - 1) / 1024 + 1;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      sum += static_cast<double>(g_D_real[bead][n]);
    }
  }
  s_sum[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }
  const float mean_D = static_cast<float>(s_sum[0] / N);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      g_D_real[bead][n] -= mean_D;
    }
  }
}

static __global__ void find_bec_diagonal_pimd_batch(
  const int N,
  const int number_of_beads,
  float* const* g_charge,
  float* const* g_bec)
{
  const int bead = blockIdx.y;
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (bead < number_of_beads && n1 < N) {
    const float q = g_charge[bead][n1];
    float* bec = g_bec[bead];
    bec[n1 + N * 0] = q;
    bec[n1 + N * 1] = 0.0f;
    bec[n1 + N * 2] = 0.0f;
    bec[n1 + N * 3] = 0.0f;
    bec[n1 + N * 4] = q;
    bec[n1 + N * 5] = 0.0f;
    bec[n1 + N * 6] = 0.0f;
    bec[n1 + N * 7] = 0.0f;
    bec[n1 + N * 8] = q;
  }
}

static __global__ void find_bec_radial_pimd_batch(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int number_of_beads,
  const Box box,
  const int* g_NN_batch,
  const int* g_NL_batch,
  const int* g_type,
  double* const* g_position,
  const float* g_charge_derivative_batch,
  float* const* g_bec)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (bead >= number_of_beads || n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  const int* g_NN = g_NN_batch + static_cast<size_t>(bead) * N;
  const int* g_NL = g_NL_batch + static_cast<size_t>(bead) * N * paramb.MN_radial;
  const float* g_charge_derivative =
    g_charge_derivative_batch + static_cast<size_t>(bead) * N * annmb.dim;
  float* bec = g_bec[bead];
  const int t1 = g_type[n1];
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    const int t2 = g_type[n2];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float r12[3] = {x12, y12, z12};
    const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    float fc12, fcp12;
    find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f};
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gnp12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        const int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
      }
      const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }
    const float bec_values[9] = {
      0.5f * r12[0] * f12[0], 0.5f * r12[0] * f12[1], 0.5f * r12[0] * f12[2],
      0.5f * r12[1] * f12[0], 0.5f * r12[1] * f12[1], 0.5f * r12[1] * f12[2],
      0.5f * r12[2] * f12[0], 0.5f * r12[2] * f12[1], 0.5f * r12[2] * f12[2]};
    for (int component = 0; component < 9; ++component) {
      atomicAdd(&bec[n1 + component * N], bec_values[component]);
      atomicAdd(&bec[n2 + component * N], -bec_values[component]);
    }
  }
}

static __global__ void find_bec_angular_pimd_batch(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int number_of_beads,
  const Box box,
  const int* g_NN_batch,
  const int* g_NL_batch,
  const int* g_type,
  double* const* g_position,
  const float* g_charge_derivative_batch,
  const float* g_sum_fxyz_batch,
  float* const* g_bec)
{
  const int bead = blockIdx.y;
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (bead >= number_of_beads || n1 >= N2) {
    return;
  }
  const double* position = g_position[bead];
  const double* g_x = position;
  const double* g_y = position + N;
  const double* g_z = position + N * 2;
  const int* g_NN = g_NN_batch + static_cast<size_t>(bead) * N;
  const int* g_NL = g_NL_batch + static_cast<size_t>(bead) * N * paramb.MN_angular;
  const float* g_charge_derivative =
    g_charge_derivative_batch + static_cast<size_t>(bead) * N * annmb.dim;
  const int sum_components =
    (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1);
  const float* g_sum_fxyz = g_sum_fxyz_batch + static_cast<size_t>(bead) * N * sum_components;
  float* bec = g_bec[bead];
  float Fp[MAX_DIM_ANGULAR] = {0.0f};
  float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
  }
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
      sum_fxyz[n * NUM_OF_ABC + abc] =
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
    }
  }
  const int t1 = g_type[n1];
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float r12[3] = {x12, y12, z12};
    const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float fc12, fcp12;
    find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f};
    const int t2 = g_type[n2];
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        const int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        gn12 += fn12[k] * annmb.c[c_index + t1 * paramb.num_types + t2 + paramb.num_c_radial];
        gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2 + paramb.num_c_radial];
      }
      accumulate_f12(
        paramb.L_max,
        paramb.has_q_222,
        paramb.has_q_1111,
        paramb.has_q_112,
        paramb.has_q_123,
        paramb.has_q_233,
        paramb.has_q_134,
        paramb.num_L,
        n,
        paramb.n_max_angular + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp,
        sum_fxyz,
        f12);
    }
    const float bec_values[9] = {
      0.5f * r12[0] * f12[0], 0.5f * r12[0] * f12[1], 0.5f * r12[0] * f12[2],
      0.5f * r12[1] * f12[0], 0.5f * r12[1] * f12[1], 0.5f * r12[1] * f12[2],
      0.5f * r12[2] * f12[0], 0.5f * r12[2] * f12[1], 0.5f * r12[2] * f12[2]};
    for (int component = 0; component < 9; ++component) {
      atomicAdd(&bec[n1 + component * N], bec_values[component]);
      atomicAdd(&bec[n2 + component * N], -bec_values[component]);
    }
  }
}

static __global__ void scale_bec_pimd_batch(
  const int N,
  const int number_of_beads,
  const float* sqrt_epsilon_inf,
  float* const* g_bec)
{
  const int bead = blockIdx.y;
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (bead < number_of_beads && n1 < N) {
    float* bec = g_bec[bead];
    for (int d = 0; d < 9; ++d) {
      bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

void NEP_Charge::compute_non_electro(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
    PRINT_INPUT_ERROR("Cannot use non-periodic boundaries for qNEP models.");
  }

  const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);
  if (is_small_box) {
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
      const int big_neighbor_size = 2000;
      const int size_x12 = current_num_atoms * big_neighbor_size;

      small_box_data.NN_radial.resize(current_num_atoms);
      small_box_data.NL_radial.resize(size_x12);
      small_box_data.NN_angular.resize(current_num_atoms);
      small_box_data.NL_angular.resize(size_x12);
      small_box_data.r12.resize(size_x12 * 6);
    }
    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, false);
  } else {
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, false);
  }
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

const GPU_Vector<int>& NEP_Charge::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP_Charge::get_NL_radial_ptr() { return nep_data.NL_radial; }

GPU_Vector<float>& NEP_Charge::get_charge_reference() { return nep_data.charge; }

GPU_Vector<float>& NEP_Charge::get_bec_reference() { return nep_data.bec; }
