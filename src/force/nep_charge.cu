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
#include "utilities/compact_nep.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_parameters.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP_Charge::check_ewald_pppm(const RunInput& run_input)
{
  use_pppm = true;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty() && tokens[0] == "kspace") {
      if (tokens.size() < 2 || tokens.size() > 3) {
        PRINT_INPUT_ERROR("kspace requires ewald or pppm [spacing].");
      }
      if (tokens[1] == "ewald") {
        if (tokens.size() != 2) {
          PRINT_INPUT_ERROR("kspace ewald does not accept spacing.");
        }
        use_pppm = false;
      } else if (tokens[1] == "pppm") {
        use_pppm = true;
        if (tokens.size() == 3 &&
            (!is_valid_real(tokens[2], &pppm_spacing) ||
             !(pppm_spacing >= 0.2 && pppm_spacing <= 2.0))) {
          PRINT_INPUT_ERROR("PPPM spacing must be a finite number between 0.2 and 2.0 A.");
        }
      } else {
        std::cout << "kspace method can only be ewald or pppm\n";
        exit(1);
      }
    }
  }
}

void NEP_Charge::check_need_bec(const RunInput& run_input)
{
  need_bec = false;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty()) {
      if (tokens[0] == "compute_dpdt") {
        need_bec = true;
        break;
      }

      if (tokens[0] == "dump_xyz" || tokens[0] == "dump_netcdf") {
        for (int n = 3; n < tokens.size(); ++n) {
          if (tokens[n] == "bec") {
            need_bec = true;
            break;
          }
        }
        if (need_bec) {
          break;
        }
      }

      if (tokens[0] == "add_efield") {
        if (
          tokens.size() == 4 || tokens.size() == 6 ||
          ((tokens.size() == 5 || tokens.size() == 7) && tokens.back() == "bec")) {
          need_bec = true;
          break;
        }
      }
    }
  }
}

struct PerAtomVirialRequirements
{
  bool anywhere = false;
  bool every_batch = false;
};

static PerAtomVirialRequirements get_peratom_virial_requirements(const RunInput& run_input)
{
  PerAtomVirialRequirements requirements;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0] == "compute_hac") {
      requirements.anywhere = true;
      int centroid_flag = 0;
      if (
        tokens.size() < 5 || !is_valid_int(tokens[4].c_str(), &centroid_flag) ||
        centroid_flag == 0) {
        requirements.every_batch = true;
      }
    } else if (
      tokens[0] == "compute_hnemd" || tokens[0] == "compute_hnemdec" ||
      tokens[0] == "compute_shc" || tokens[0] == "compute_gkma" ||
      tokens[0] == "compute_hnema") {
      requirements.anywhere = true;
      requirements.every_batch = true;
    }
    if (tokens[0] == "compute") {
      for (const auto& token : tokens) {
        if (token == "virial" || token == "jp") {
          requirements.anywhere = true;
          requirements.every_batch = true;
          break;
        }
      }
    }
    if (tokens[0] == "dump_xyz" || tokens[0] == "dump_netcdf") {
      for (const auto& token : tokens) {
        if (token == "virial") {
          requirements.anywhere = true;
          requirements.every_batch = true;
          break;
        }
      }
    }
    if (requirements.every_batch) {
      break;
    }
  }
  return requirements;
}

void NEP_Charge::initialize_dftd3(const RunInput& run_input)
{
  has_dftd3 = false;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty() && tokens[0] == "dftd3") {
      has_dftd3 = true;
      if (tokens.size() != 4) {
        std::cout << "dftd3 must have 3 parameters\n";
        exit(1);
      }
      std::string xc_functional = tokens[1];
      float rc_potential = get_double_from_token(tokens[2], __FILE__, __LINE__);
      float rc_coordination_number = get_double_from_token(tokens[3], __FILE__, __LINE__);
      dftd3.initialize(
        xc_functional,
        rc_potential,
        rc_coordination_number,
        get_first_potential_filename(run_input));
      break;
    }
  }
}

NEP_Charge::NEP_Charge(
  const char* file_potential, const int num_atoms, const RunInput& run_input)
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
      zbl.flexible = true;
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
  std::vector<float> descriptor_parameters = get_descriptor_parameters_type_pair(
    parameters,
    annmb.num_para_ann,
    paramb.num_types,
    paramb.n_max_radial,
    paramb.n_max_angular,
    paramb.basis_size_radial,
    paramb.basis_size_angular);
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(parameters.data());
  nep_data.descriptor_parameters_type_pair.resize(num_para_descriptor);
  nep_data.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  annmb.c_type_pair = nep_data.descriptor_parameters_type_pair.data();
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;

  // flexible zbl potential parameters
  if (zbl.flexible) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  // charge related parameters and data
  charge_para.alpha = float(PI) / paramb.rc_radial; // a good value
  check_ewald_pppm(run_input);
  check_need_bec(run_input);
  if (use_pppm) {
    const PerAtomVirialRequirements virial_requirements = get_peratom_virial_requirements(run_input);
    pppm.initialize(
      charge_para.alpha,
      virial_requirements.anywhere,
      virial_requirements.every_batch,
      pppm_spacing);
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
  neighbor_manager.initialize(rc, num_atoms, paramb.MN_radial);

  initialize_dftd3(run_input);
}

NEP_Charge::~NEP_Charge(void)
{
  // GPU_Vector members release the batch buffers.
}

void NEP_Charge::set_neighbor_rebuild(const bool value)
{
  neighbor_always_rebuild_ = value;
  neighbor_manager.set_always_rebuild(value);
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
  const std::vector<GPU_Vector<double>*>& virial_beads,
  const bool is_small_box)
{
  const int number_of_beads = int(position_beads.size());
  const int small_box_neighbor_size = 2000;
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
    batch.rebuild_reason_flags.resize(number_of_beads);
    batch.any_rebuild.resize(1);
    batch.active_bead_ids.resize(number_of_beads);
    batch.x0_ptrs_host.resize(number_of_beads);
    batch.y0_ptrs_host.resize(number_of_beads);
    batch.z0_ptrs_host.resize(number_of_beads);
    batch.small_box_x0_ptrs.resize(number_of_beads);
    batch.small_box_y0_ptrs.resize(number_of_beads);
    batch.small_box_z0_ptrs.resize(number_of_beads);
    batch.small_box_rebuild_flags.resize(number_of_beads);
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
    batch.neighbor_ptrs.reserve(number_of_beads);
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
      batch.neighbor_ptrs.push_back(bead->neighbor.get());
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
  if (is_small_box && !batch.small_box_data_allocated) {
    batch.small_NN_radial.resize(static_cast<size_t>(number_of_beads) * number_of_atoms);
    batch.small_NL_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_NN_angular.resize(static_cast<size_t>(number_of_beads) * number_of_atoms);
    batch.small_NL_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_x12_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_y12_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_z12_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_x12_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_y12_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_z12_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_image_x_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_image_y_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_image_z_radial.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_image_x_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_image_y_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);
    batch.small_image_z_angular.resize(
      static_cast<size_t>(number_of_beads) * number_of_atoms * small_box_neighbor_size);

    std::vector<double*> small_box_x0_ptrs(number_of_beads);
    std::vector<double*> small_box_y0_ptrs(number_of_beads);
    std::vector<double*> small_box_z0_ptrs(number_of_beads);
    for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
      batch.beads[bead_id]->small_box_x0.resize(number_of_atoms);
      batch.beads[bead_id]->small_box_y0.resize(number_of_atoms);
      batch.beads[bead_id]->small_box_z0.resize(number_of_atoms);
      small_box_x0_ptrs[bead_id] = batch.beads[bead_id]->small_box_x0.data();
      small_box_y0_ptrs[bead_id] = batch.beads[bead_id]->small_box_y0.data();
      small_box_z0_ptrs[bead_id] = batch.beads[bead_id]->small_box_z0.data();
    }
    batch.small_box_x0_ptrs.copy_from_host(small_box_x0_ptrs.data());
    batch.small_box_y0_ptrs.copy_from_host(small_box_y0_ptrs.data());
    batch.small_box_z0_ptrs.copy_from_host(small_box_z0_ptrs.data());
    batch.small_box_data_allocated = true;
  }
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
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
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
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
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
  const int num_angular_channels = paramb.n_max_angular + 1;
  const int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  for (int n = 0; n < num_angular_channels; n += 2) {
    float s0[NUM_OF_ABC] = {0.0f};
    float s1[NUM_OF_ABC] = {0.0f};
    const bool has_second_channel = n + 1 < num_angular_channels;
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
      float gn0 = 0.0f;
      float gn1 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index =
          (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn0 += fn12[k] * annmb.c[c_index];
        if (has_second_channel) {
          c_index =
            ((n + 1) * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn1 += fn12[k] * annmb.c[c_index];
        }
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn0, s0);
      if (has_second_channel) {
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn1, s1);
      }
    }
    find_q(
      paramb.L_max,
      paramb.has_q_222,
      paramb.has_q_1111,
      paramb.has_q_112,
      paramb.has_q_123,
      paramb.has_q_233,
      paramb.has_q_134,
      num_angular_channels,
      n,
      s0,
      q + (paramb.n_max_radial + 1));
    for (int abc = 0; abc < num_abc; ++abc) {
      g_sum_fxyz[(n * num_abc + abc) * N + n1] = s0[abc];
    }
    if (has_second_channel) {
      find_q(
        paramb.L_max,
        paramb.has_q_222,
        paramb.has_q_1111,
        paramb.has_q_112,
        paramb.has_q_123,
        paramb.has_q_233,
        paramb.has_q_134,
        num_angular_channels,
        n + 1,
        s1,
        q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < num_abc; ++abc) {
        g_sum_fxyz[((n + 1) * num_abc + abc) * N + n1] = s1[abc];
      }
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
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
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
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
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
          int c_index_12 = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          int c_index_21 = get_c_index(
            t2 * paramb.num_types + t1, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index_12];
          gnp21 += fnp12[k] * annmb.c_type_pair[c_index_21];
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
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
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

template <int ABC_STRIDE>
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
  float sum_fxyz[ABC_STRIDE * MAX_NUM_N];
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1] +
            g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] *
              bead_D_real[n1];
  }
  const int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
    for (int abc = 0; abc < num_abc; ++abc) {
      sum_fxyz[n * ABC_STRIDE + abc] =
        g_sum_fxyz[(n * num_abc + abc) * N + n1];
    }
  }
  const int t1 = g_type[n1];
  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];
  const int shard = blockIdx.z;
  const int shards = gridDim.z;
  for (int i1 = shard; i1 < g_NN_angular[n1]; i1 += shards) {
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
      accumulate_f12<ABC_STRIDE>(
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
      if (zbl.flexible) {
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

static __global__ void find_delta_j_q_real_space(
  const int N,
  const int N1,
  const int N2,
  const NEP_Charge::Charge_Para charge_para,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const float* g_charge_rate,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_partial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    double moment_x = 0.0;
    double moment_y = 0.0;
    double moment_z = 0.0;
    const double x1 = g_x[n1];
    const double y1 = g_y[n1];
    const double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int n2 = g_NL[n1 + N * i1];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      const double r_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (r_square <= 0.0) continue;
      const double r = sqrt(r_square);
      const double weight = erfc(static_cast<double>(charge_para.alpha) * r) / r;
      const double q_weight = static_cast<double>(g_charge[n2]) * weight;
      moment_x += q_weight * x12;
      moment_y += q_weight * y12;
      moment_z += q_weight * z12;
    }
    const double prefactor = 0.5 * static_cast<double>(K_C_SP) * g_charge_rate[n1];
    g_partial[n1] = prefactor * moment_x;
    g_partial[n1 + N] = prefactor * moment_y;
    g_partial[n1 + 2 * N] = prefactor * moment_z;
  }
}

static __global__ void find_delta_j_q_real_space_small_box(
  const int N,
  const int N1,
  const int N2,
  const NEP_Charge::Charge_Para charge_para,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const float* g_charge_rate,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  double* g_partial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    double moment_x = 0.0;
    double moment_y = 0.0;
    double moment_z = 0.0;
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = i1 * N + n1;
      const double x12 = static_cast<double>(g_x12[index]);
      const double y12 = static_cast<double>(g_y12[index]);
      const double z12 = static_cast<double>(g_z12[index]);
      const double r_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (r_square <= 0.0) continue;
      const double r = sqrt(r_square);
      const double weight = erfc(static_cast<double>(charge_para.alpha) * r) / r;
      const double q_weight = static_cast<double>(g_charge[g_NL[index]]) * weight;
      moment_x += q_weight * x12;
      moment_y += q_weight * y12;
      moment_z += q_weight * z12;
    }
    const double prefactor = 0.5 * static_cast<double>(K_C_SP) * g_charge_rate[n1];
    g_partial[n1] = prefactor * moment_x;
    g_partial[n1 + N] = prefactor * moment_y;
    g_partial[n1 + 2 * N] = prefactor * moment_z;
  }
}

static __global__ void reduce_delta_j_q_real_space(
  const int N,
  const int N1,
  const int N2,
  const double* g_partial,
  double* g_total)
{
  const int component = blockIdx.x;
  const int tid = threadIdx.x;
  if (component >= 3) return;
  __shared__ double s_data[1024];
  double sum = 0.0;
  for (int n = N1 + tid; n < N2; n += 1024)
    sum += g_partial[n + component * N];
  s_data[tid] = sum;
  __syncthreads();

  for (int offset = 512; offset > 0; offset >>= 1) {
    if (tid < offset) s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0) g_total[component] = s_data[0];
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
  const bool include_electro,
  const bool include_bec)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;
  const bool capture_charge_diagnostics = charge_diagnostics_requested_;

  neighbor_manager.update(
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
    neighbor_manager.get_candidate_NN().data(),
    neighbor_manager.get_candidate_NL().data(),
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

  if (capture_charge_diagnostics) {
    nep_data.charge_raw.copy_from_device(nep_data.charge.data());
  }

  if (include_electro) {
    zero_total_charge<<<1, 1024>>>(N, nep_data.charge.data());
    GPU_CHECK_KERNEL

    if (include_bec) {
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
    }
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
        potential_per_atom,
        peratom_virial_requested_,
        force_evaluation_id_);
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

    if (capture_charge_diagnostics) {
      nep_data.D_raw.copy_from_device(nep_data.D_real.data());
    }
    zero_mean_D_real<<<1, 1024>>>(N, nep_data.D_real.data());
    GPU_CHECK_KERNEL
    if (capture_charge_diagnostics) {
      nep_data.D_projected.copy_from_device(nep_data.D_real.data());
      charge_diagnostics_requested_ = false;
      charge_diagnostics_available_ = true;
      charge_diagnostics_force_evaluation_id_ = force_evaluation_id_;
    }
    peratom_virial_requested_ = false;
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
  const bool include_electro,
  const bool include_bec)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;
  const bool capture_charge_diagnostics = charge_diagnostics_requested_;

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

  if (capture_charge_diagnostics) {
    nep_data.charge_raw.copy_from_device(nep_data.charge.data());
  }

  if (include_electro) {
    zero_total_charge<<<1, 1024>>>(N, nep_data.charge.data());
    GPU_CHECK_KERNEL

    if (include_bec) {
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
    }
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
        potential_per_atom,
        peratom_virial_requested_,
        force_evaluation_id_);
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
        paramb.rc_radial,
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

    if (capture_charge_diagnostics) {
      nep_data.D_raw.copy_from_device(nep_data.D_real.data());
    }
    zero_mean_D_real<<<1, 1024>>>(N, nep_data.D_real.data());
    GPU_CHECK_KERNEL
    if (capture_charge_diagnostics) {
      nep_data.D_projected.copy_from_device(nep_data.D_real.data());
      charge_diagnostics_requested_ = false;
      charge_diagnostics_available_ = true;
      charge_diagnostics_force_evaluation_id_ = force_evaluation_id_;
    }
    peratom_virial_requested_ = false;
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

void NEP_Charge::invalidate_current_force_caches_()
{
  reference_force_frame_valid_ = false;
  pppm.invalidate_current_force_mesh();
  charge_rate_cache_set_ = false;
  charge_heat_channel_cache_set_ = false;
  full_a_current_cache_set_ = false;
  dynamic_q_cache_set_ = false;
  dynamic_q_last_pppm_valid_ = false;
  dynamic_q_last_diagnostic_checks_pass_ = false;
  charge_diagnostics_available_ = false;
}

void NEP_Charge::begin_force_evaluation_()
{
  ++force_evaluation_id_;
  invalidate_current_force_caches_();
}

void NEP_Charge::invalidate_current_force_caches()
{
  invalidate_current_force_caches_();
}

void NEP_Charge::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  begin_force_evaluation_();
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
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      true,
      md_qnep_bec_enabled_);
  } else {
    compute_large_box(
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      true,
      md_qnep_bec_enabled_);
  }
  pppm.finish_debug_force_evaluation();
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
  reference_force_frame_valid_ = true;
  reference_force_frame_N_ = type.size();
  reference_force_frame_id_ = force_evaluation_id_;
  reference_force_frame_type_ = type.data();
  reference_force_frame_position_ = position_per_atom.data();
  reference_force_frame_force_ = force_per_atom.data();
  reference_force_frame_small_box_ = is_small_box;
  reference_force_frame_N1_ = N1;
  reference_force_frame_N2_ = N2;
  for (int i = 0; i < 18; ++i) reference_force_frame_box_[i] = box.cpu_h[i];
}

static __device__ __forceinline__ void find_charge_gradient_radial_pair(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int n1,
  const int n2,
  const int* g_type,
  const float r12[3],
  const float* g_charge_derivative,
  float f12[3])
{
  const int t1 = g_type[n1];
  const int t2 = g_type[n2];
  const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
  float fc12, fcp12, fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
  find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
  find_fn_and_fnp(
    paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    float gnp12 = 0.0f;
    for (int k = 0; k <= paramb.basis_size_radial; ++k) {
      const int c_index = get_c_index(
        t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
      gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
    }
    const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 / d12;
    for (int d = 0; d < 3; ++d) f12[d] += tmp12 * r12[d];
  }
}

static __device__ __forceinline__ void find_charge_gradient_angular_pair(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int n1,
  const int n2,
  const int* g_type,
  const float r12[3],
  const float* Fp,
  const float* sum_fxyz,
  float f12[3])
{
  const int t1 = g_type[n1];
  const int t2 = g_type[n2];
  const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
  float fc12, fcp12, fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
  find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
  find_fn_and_fnp(
    paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float gn12 = 0.0f, gnp12 = 0.0f;
    for (int k = 0; k <= paramb.basis_size_angular; ++k) {
      const int c_index = get_c_index(
        t1 * paramb.num_types + t2,
        n,
        k,
        paramb.n_max_angular,
        paramb.basis_size_angular,
        paramb.num_c_radial);
      gn12 += fn12[k] * annmb.c_type_pair[c_index];
      gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
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
}

static __global__ void compute_reference_local_tangent(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const NEP_Charge::ZBL zbl,
  const int N,
  const Box box,
  const bool small_box,
  const int* NN_radial,
  const int* NL_radial,
  const int* NN_angular,
  const int* NL_angular,
  const int* type,
  const double* position,
  const double* direction,
  const float* x12_radial,
  const float* y12_radial,
  const float* z12_radial,
  const float* x12_angular,
  const float* y12_angular,
  const float* z12_angular,
  const float* Fp,
  const float* charge_derivative,
  const float* sum_fxyz,
  double* charge_direction_raw,
  double* short_site_derivative)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float energy_fp[MAX_DIM_ANGULAR] = {0.0f};
  float charge_fp[MAX_DIM_ANGULAR] = {0.0f};
  float angular_sum[NUM_OF_ABC * MAX_NUM_N];
  for (int d = 0; d < paramb.dim_angular; ++d)
    energy_fp[d] = Fp[(paramb.n_max_radial + 1 + d) * N + i];
  for (int d = 0; d < paramb.dim_angular; ++d)
    charge_fp[d] = charge_derivative[(paramb.n_max_radial + 1 + d) * N + i];
  for (int n = 0; n <= paramb.n_max_angular; ++n)
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc)
      angular_sum[n * NUM_OF_ABC + abc] =
        sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + i];

  double local = 0.0;
  double charge_tangent = 0.0;
  const double ux = direction[i], uy = direction[i + N], uz = direction[i + 2 * N];
  for (int pass = 0; pass < 2; ++pass) {
    const bool angular = pass != 0;
    const int count = angular ? NN_angular[i] : NN_radial[i];
    const int* NL = angular ? NL_angular : NL_radial;
    const float* xx = angular ? x12_angular : x12_radial;
    const float* yy = angular ? y12_angular : y12_radial;
    const float* zz = angular ? z12_angular : z12_radial;
    for (int edge = 0; edge < count; ++edge) {
      const int idx = i + N * edge;
      const int j = NL[idx];
      float r[3];
      if (small_box) {
        r[0] = xx[idx]; r[1] = yy[idx]; r[2] = zz[idx];
      } else {
        r[0] = static_cast<float>(position[j] - position[i]);
        r[1] = static_cast<float>(position[j + N] - position[i + N]);
        r[2] = static_cast<float>(position[j + 2 * N] - position[i + 2 * N]);
        apply_mic(box, r[0], r[1], r[2]);
      }
      const double dux = direction[j] - ux, duy = direction[j + N] - uy, duz = direction[j + 2 * N] - uz;
      if (dux == 0.0 && duy == 0.0 && duz == 0.0) continue;
      float energy_gradient[3] = {0.0f, 0.0f, 0.0f};
      float charge_gradient[3] = {0.0f, 0.0f, 0.0f};
      if (angular) {
        find_charge_gradient_angular_pair(paramb, annmb, i, j, type, r, energy_fp, angular_sum, energy_gradient);
        find_charge_gradient_angular_pair(paramb, annmb, i, j, type, r, charge_fp, angular_sum, charge_gradient);
      } else {
        find_charge_gradient_radial_pair(paramb, annmb, N, i, j, type, r, Fp, energy_gradient);
        find_charge_gradient_radial_pair(paramb, annmb, N, i, j, type, r, charge_derivative, charge_gradient);
      }
      local += energy_gradient[0] * dux + energy_gradient[1] * duy + energy_gradient[2] * duz;
      charge_tangent += charge_gradient[0] * dux + charge_gradient[1] * duy + charge_gradient[2] * duz;
      if (zbl.enabled && angular) {
        const float distance = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (distance > 0.0f) {
          const int t1 = type[i], t2 = type[j];
          const int zi = zbl.atomic_numbers[t1], zj = zbl.atomic_numbers[t2];
          const float a_inv = (powf(static_cast<float>(zi), 0.23f) + powf(static_cast<float>(zj), 0.23f)) * 2.134563f;
          const float zizj = K_C_SP * zi * zj;
          float value, derivative;
          if (zbl.flexible) {
            const int lo = min(t1, t2), hi = max(t1, t2);
            const int zbl_index = lo * zbl.num_types - (lo * (lo - 1)) / 2 + (hi - lo);
            float parameters[10];
            for (int k = 0; k < 10; ++k) parameters[k] = zbl.para[10 * zbl_index + k];
            find_f_and_fp_zbl(parameters, zizj, a_inv, distance, 1.0f / distance, value, derivative);
          } else {
            float inner = zbl.rc_inner, outer = zbl.rc_outer;
            if (paramb.use_typewise_cutoff_zbl) {
              outer = min((COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor, outer);
              inner = 0.0f;
            }
            find_f_and_fp_zbl(zizj, a_inv, inner, outer, distance, 1.0f / distance, value, derivative);
          }
          local += 0.5 * derivative * (r[0] * dux + r[1] * duy + r[2] * duz) / distance;
        }
      }
    }
  }
  charge_direction_raw[i] = charge_tangent;
  short_site_derivative[i] = local;
}

static __global__ void project_reference_charge_direction(
  const int N, const double* raw, double* projected)
{
  __shared__ double sums[1024];
  const int tid = threadIdx.x;
  double sum = 0.0;
  for (int i = tid; i < N; i += 1024) sum += raw[i];
  sums[tid] = sum;
  __syncthreads();
  for (int offset = 512; offset > 0; offset >>= 1) {
    if (tid < offset) sums[tid] += sums[tid + offset];
    __syncthreads();
  }
  const double mean = sums[0] / N;
  for (int i = tid; i < N; i += 1024) projected[i] = raw[i] - mean;
}

static __global__ void add_reference_real_site_tangent(
  const int N,
  const NEP_Charge::Charge_Para charge_para,
  const Box box,
  const bool small_box,
  const int* NN,
  const int* NL,
  const float* x12,
  const float* y12,
  const float* z12,
  const double* position,
  const double* direction,
  const float* charge,
  const double* charge_direction,
  double* site_derivative,
  double* real_site_tangent)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const double qi = charge[i], dqi = charge_direction[i];
  double value = -static_cast<double>(charge_para.two_alpha_over_sqrt_pi) * qi * dqi;
  for (int e = 0; e < NN[i]; ++e) {
    const int index = i + N * e, j = NL[index];
    double r[3];
    if (small_box) {
      r[0] = x12[index]; r[1] = y12[index]; r[2] = z12[index];
    } else {
      float r_float[3] = {
        static_cast<float>(position[j] - position[i]),
        static_cast<float>(position[j + N] - position[i + N]),
        static_cast<float>(position[j + 2 * N] - position[i + 2 * N])};
      apply_mic(box, r_float[0], r_float[1], r_float[2]);
      r[0] = r_float[0]; r[1] = r_float[1]; r[2] = r_float[2];
    }
    const double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (r2 <= 0.0) continue;
    const double distance = sqrt(r2), invr = 1.0 / distance;
    const double alpha = charge_para.alpha;
    const double phi = erfc(alpha * distance) * invr;
    const double dphi = -charge_para.two_alpha_over_sqrt_pi * exp(-alpha * alpha * r2) * invr - phi * invr;
    const double dqj = charge_direction[j];
    const double dr = (r[0] * (direction[j] - direction[i]) +
                       r[1] * (direction[j + N] - direction[i + N]) +
                       r[2] * (direction[j + 2 * N] - direction[i + 2 * N])) * invr;
    value += 0.5 * ((dqi * charge[j] + qi * dqj) * phi + qi * charge[j] * dphi * dr);
  }
  if (real_site_tangent != nullptr) real_site_tangent[i] = static_cast<double>(K_C_SP) * value;
  site_derivative[i] += static_cast<double>(K_C_SP) * value;
}

static bool reduce_reference_tangent_values(
  const std::vector<double>& values, QNEPReferenceTangentReduction& result)
{
  result = {};
  long double squares = 0.0L;
  double compensation = 0.0;
  for (double value : values) {
    if (!std::isfinite(value)) return false;
    result.ordered_sum += value;
    const double next = result.compensated_sum + value;
    compensation += std::abs(result.compensated_sum) >= std::abs(value)
      ? (result.compensated_sum - next) + value
      : (value - next) + result.compensated_sum;
    result.compensated_sum = next;
    result.extended_sum += static_cast<long double>(value);
    result.sum_abs += std::abs(value);
    result.max_abs = std::max(result.max_abs, std::abs(value));
    squares += static_cast<long double>(value) * value;
  }
  result.compensated_sum += compensation;
  result.rms = std::sqrt(static_cast<double>(squares / values.size()));
  return std::isfinite(result.ordered_sum) && std::isfinite(result.compensated_sum) &&
    std::isfinite(result.extended_sum) && std::isfinite(result.sum_abs) &&
    std::isfinite(result.max_abs) && std::isfinite(result.rms);
}

static bool reduce_reference_tangent_xyz(
  const std::vector<double>& values, const int N, std::array<double, 3>& result)
{
  result = {};
  for (int d = 0; d < 3; ++d) {
    for (int i = 0; i < N; ++i) {
      const double value = values[d * N + i];
      if (!std::isfinite(value)) return false;
      result[d] += value;
    }
    if (!std::isfinite(result[d])) return false;
  }
  return true;
}

static __global__ void combine_reference_tangent(
  const int N,
  const double* short_site,
  const double* pppm_site,
  double* site_derivative,
  const double* native_force,
  const double* pppm_ik_force,
  const double* pppm_explicit_gradient,
  double* total_gradient)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && site_derivative != nullptr) site_derivative[i] = short_site[i] + pppm_site[i];
  if (i < 3 * N)
    total_gradient[i] = -native_force[i] + pppm_ik_force[i] + pppm_explicit_gradient[i];
}

bool NEP_Charge::compute_reference_site_energy_derivative(
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& native_force,
  const GPU_Vector<double>* direction,
  GPU_Vector<double>* site_derivative,
  GPU_Vector<double>& total_energy_gradient,
  QNEPReferenceTangentDiagnostics* diagnostics)
{
  const int N = type.size();
  if (diagnostics != nullptr) *diagnostics = {};
  if (diagnostics != nullptr && direction == nullptr) return false;
  if (!reference_force_frame_valid_ || reference_force_frame_id_ != force_evaluation_id_ ||
      N <= 0 || reference_force_frame_N_ != N || reference_force_frame_type_ != type.data() ||
      reference_force_frame_position_ != position.data() || reference_force_frame_force_ != native_force.data() ||
      reference_force_frame_N1_ != 0 || reference_force_frame_N2_ != N || has_dftd3 || !use_pppm ||
      (paramb.charge_mode != 1 && paramb.charge_mode != 2) ||
      box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 ||
      position.size() < static_cast<size_t>(3) * N || native_force.size() < static_cast<size_t>(3) * N ||
      ((direction == nullptr) != (site_derivative == nullptr)) ||
      (direction != nullptr && direction->size() < static_cast<size_t>(3) * N) ||
      nep_data.charge.size() != static_cast<size_t>(N))
    return false;
  for (int i = 0; i < 18; ++i)
    if (reference_force_frame_box_[i] != box.cpu_h[i]) return false;

  const bool small_box = reference_force_frame_small_box_;
  const int* NN_radial = small_box ? small_box_data.NN_radial.data() : nep_data.NN_radial.data();
  const int* NL_radial = small_box ? small_box_data.NL_radial.data() : nep_data.NL_radial.data();
  const int* NN_angular = small_box ? small_box_data.NN_angular.data() : nep_data.NN_angular.data();
  const int* NL_angular = small_box ? small_box_data.NL_angular.data() : nep_data.NL_angular.data();
  const float* x12_radial = small_box ? small_box_data.r12.data() : nullptr;
  const size_t small_box_stride = small_box ? small_box_data.r12.size() / 6 : 0;
  const float* y12_radial = small_box ? small_box_data.r12.data() + small_box_stride : nullptr;
  const float* z12_radial = small_box ? small_box_data.r12.data() + 2 * small_box_stride : nullptr;
  const float* x12_angular = small_box ? small_box_data.r12.data() + 3 * small_box_stride : nullptr;
  const float* y12_angular = small_box ? small_box_data.r12.data() + 4 * small_box_stride : nullptr;
  const float* z12_angular = small_box ? small_box_data.r12.data() + 5 * small_box_stride : nullptr;
  const size_t size_n = static_cast<size_t>(N);
  const size_t size_3n = 3 * size_n;
  if (reference_charge_direction_.size() != size_n) reference_charge_direction_.resize(size_n);
  if (reference_charge_direction_raw_.size() != size_n) reference_charge_direction_raw_.resize(size_n);
  if (reference_short_site_derivative_.size() != size_n) reference_short_site_derivative_.resize(size_n);
  if (reference_pppm_site_derivative_.size() != size_n) reference_pppm_site_derivative_.resize(size_n);
  if (reference_pppm_explicit_gradient_.size() != size_3n) reference_pppm_explicit_gradient_.resize(size_3n);
  if (reference_pppm_ik_force_.size() != size_3n) reference_pppm_ik_force_.resize(size_3n);
  if (site_derivative != nullptr && site_derivative->size() != size_n) site_derivative->resize(size_n);
  if (total_energy_gradient.size() != size_3n) total_energy_gradient.resize(size_3n);

  if (direction != nullptr) {
    if (diagnostics != nullptr) {
      diagnostics->local_site_tangent.resize(size_n);
      diagnostics->real_site_tangent.resize(size_n);
      diagnostics->pppm_site_tangent.resize(size_n);
    }
    compute_reference_local_tangent<<<(N - 1) / 64 + 1, 64>>>(
      paramb, annmb, zbl, N, box, small_box, NN_radial, NL_radial, NN_angular, NL_angular,
      type.data(), position.data(), direction->data(), x12_radial, y12_radial, z12_radial,
      x12_angular, y12_angular, z12_angular, nep_data.Fp.data(), nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data(), reference_charge_direction_raw_.data(), reference_short_site_derivative_.data());
    GPU_CHECK_KERNEL
    project_reference_charge_direction<<<1, 1024>>>(
      N, reference_charge_direction_raw_.data(), reference_charge_direction_.data());
    GPU_CHECK_KERNEL
    if (diagnostics != nullptr) {
      std::vector<double> raw(size_n), projected(size_n);
      reference_charge_direction_raw_.copy_to_host(raw.data());
      reference_charge_direction_.copy_to_host(projected.data());
      if (!reduce_reference_tangent_values(raw, diagnostics->raw_charge_direction) ||
          !reduce_reference_tangent_values(projected, diagnostics->projected_charge_direction))
        return false;
      reference_short_site_derivative_.copy_to_host(diagnostics->local_site_tangent.data());
    }
    if (paramb.charge_mode == 1) {
      add_reference_real_site_tangent<<<(N - 1) / 64 + 1, 64>>>(
        N, charge_para, box, small_box, NN_radial, NL_radial, x12_radial, y12_radial, z12_radial,
        position.data(), direction->data(), nep_data.charge.data(), reference_charge_direction_.data(),
        reference_short_site_derivative_.data(),
        diagnostics == nullptr ? nullptr : reference_pppm_site_derivative_.data());
      GPU_CHECK_KERNEL
      if (diagnostics != nullptr)
        reference_pppm_site_derivative_.copy_to_host(diagnostics->real_site_tangent.data());
    } else if (diagnostics != nullptr) {
      std::fill(diagnostics->real_site_tangent.begin(), diagnostics->real_site_tangent.end(), 0.0);
    }
  }
  if (!pppm.compute_reference_energy_tangent(
        N, box, nep_data.charge, position, direction,
        direction == nullptr ? nullptr : &reference_charge_direction_, force_evaluation_id_,
        site_derivative == nullptr ? nullptr : &reference_pppm_site_derivative_,
        reference_pppm_explicit_gradient_, reference_pppm_ik_force_))
    return false;
  if (diagnostics != nullptr) {
    reference_pppm_site_derivative_.copy_to_host(diagnostics->pppm_site_tangent.data());
    std::vector<double> ik(size_3n), explicit_gradient(size_3n), native(size_3n);
    reference_pppm_ik_force_.copy_to_host(ik.data());
    reference_pppm_explicit_gradient_.copy_to_host(explicit_gradient.data());
    native_force.copy_to_host(native.data());
    if (!reduce_reference_tangent_xyz(ik, N, diagnostics->pppm_ik_force_sum) ||
        !reduce_reference_tangent_xyz(explicit_gradient, N, diagnostics->pppm_explicit_gradient_sum) ||
        !reduce_reference_tangent_xyz(native, N, diagnostics->native_force_sum)) return false;
    for (size_t i = 0; i < size_n; ++i)
      if (!std::isfinite(diagnostics->local_site_tangent[i]) ||
          !std::isfinite(diagnostics->real_site_tangent[i]) ||
          !std::isfinite(diagnostics->pppm_site_tangent[i])) return false;
  }
  combine_reference_tangent<<<(3 * N - 1) / 64 + 1, 64>>>(
    N, direction == nullptr ? nullptr : reference_short_site_derivative_.data(),
    direction == nullptr ? nullptr : reference_pppm_site_derivative_.data(),
    site_derivative == nullptr ? nullptr : site_derivative->data(),
    native_force.data(), reference_pppm_ik_force_.data(), reference_pppm_explicit_gradient_.data(),
    total_energy_gradient.data());
  GPU_CHECK_KERNEL
  if (diagnostics != nullptr) {
    std::vector<double> gradient(size_3n), site(size_n);
    total_energy_gradient.copy_to_host(gradient.data());
    site_derivative->copy_to_host(site.data());
    for (double value : gradient) if (!std::isfinite(value)) return false;
    for (double value : site) if (!std::isfinite(value)) return false;
  }
  if (diagnostics != nullptr) diagnostics->valid = true;
  return true;
}

bool NEP_Charge::diagnose_reference_translation_energy(
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& native_force,
  PPPMReferenceTranslationReport& report,
  const double precision_target)
{
  report = {};
  const int N = type.size();
  if (!reference_force_frame_valid_ || reference_force_frame_id_ != force_evaluation_id_ ||
      N <= 0 || reference_force_frame_N_ != N || reference_force_frame_type_ != type.data() ||
      reference_force_frame_position_ != position.data() || reference_force_frame_force_ != native_force.data() ||
      reference_force_frame_N1_ != 0 || reference_force_frame_N2_ != N || has_dftd3 || !use_pppm ||
      (paramb.charge_mode != 1 && paramb.charge_mode != 2) ||
      box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 ||
      position.size() < static_cast<size_t>(3) * N || native_force.size() < static_cast<size_t>(3) * N ||
      nep_data.charge.size() != static_cast<size_t>(N))
    return false;
  for (int i = 0; i < 18; ++i)
    if (reference_force_frame_box_[i] != box.cpu_h[i]) return false;
  return pppm.diagnose_reference_translation_energy(
    N, box, nep_data.charge, position, force_evaluation_id_, report, precision_target);
}

static __device__ __forceinline__ void accumulate_charge_heat_channel(
  const int N,
  const int n1,
  const int n2,
  const float r12[3],
  const float f12[3],
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const float* g_D_projected,
  const int component_offset,
  double* g_channel)
{
  if (g_channel == nullptr) return;
  const double gv = static_cast<double>(f12[0]) * g_vx[n2] +
                    static_cast<double>(f12[1]) * g_vy[n2] +
                    static_cast<double>(f12[2]) * g_vz[n2];
  const double factor = -static_cast<double>(g_D_projected[n1]) * gv;
  for (int d = 0; d < 3; ++d)
    g_channel[n1 + (component_offset + d) * N] += factor * static_cast<double>(r12[d]);
}

static __global__ void find_charge_rate_radial(
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
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const float* g_charge_derivative,
  float* g_charge_rate,
  const float* g_D_projected,
  double* g_channel)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    float r12[3] = {
      float(g_x[n2] - g_x[n1]), float(g_y[n2] - g_y[n1]), float(g_z[n2] - g_z[n1])};
    apply_mic(box, r12[0], r12[1], r12[2]);
    float f12[3] = {0.0f, 0.0f, 0.0f};
    find_charge_gradient_radial_pair(paramb, annmb, N, n1, n2, g_type, r12, g_charge_derivative, f12);
    if (g_charge_rate != nullptr) {
      g_charge_rate[n1] += f12[0] * (g_vx[n2] - g_vx[n1]) +
                            f12[1] * (g_vy[n2] - g_vy[n1]) +
                            f12[2] * (g_vz[n2] - g_vz[n1]);
    }
    accumulate_charge_heat_channel(
      N, n1, n2, r12, f12, g_vx, g_vy, g_vz, g_D_projected, 0, g_channel);
  }
}

static __global__ void find_charge_rate_angular(
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
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* g_charge_rate,
  const float* g_D_projected,
  double* g_channel)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  float Fp[MAX_DIM_ANGULAR] = {0.0f};
  float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
  for (int d = 0; d < paramb.dim_angular; ++d)
    Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
  for (int n = 0; n <= paramb.n_max_angular; ++n)
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc)
      sum_fxyz[n * NUM_OF_ABC + abc] =
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int n2 = g_NL[n1 + N * i1];
    float r12[3] = {
      float(g_x[n2] - g_x[n1]), float(g_y[n2] - g_y[n1]), float(g_z[n2] - g_z[n1])};
    apply_mic(box, r12[0], r12[1], r12[2]);
    float f12[3] = {0.0f, 0.0f, 0.0f};
    find_charge_gradient_angular_pair(paramb, annmb, n1, n2, g_type, r12, Fp, sum_fxyz, f12);
    if (g_charge_rate != nullptr) {
      g_charge_rate[n1] += f12[0] * (g_vx[n2] - g_vx[n1]) +
                            f12[1] * (g_vy[n2] - g_vy[n1]) +
                            f12[2] * (g_vz[n2] - g_vz[n1]);
    }
    accumulate_charge_heat_channel(
      N, n1, n2, r12, f12, g_vx, g_vy, g_vz, g_D_projected, 3, g_channel);
  }
}

static __global__ void find_charge_rate_radial_small_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const float* g_charge_derivative,
  float* g_charge_rate,
  const float* g_D_projected,
  double* g_channel)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int index = n1 + N * i1;
    const int n2 = g_NL[index];
    const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float f12[3] = {0.0f, 0.0f, 0.0f};
    find_charge_gradient_radial_pair(paramb, annmb, N, n1, n2, g_type, r12, g_charge_derivative, f12);
    if (g_charge_rate != nullptr) {
      g_charge_rate[n1] += f12[0] * (g_vx[n2] - g_vx[n1]) +
                            f12[1] * (g_vy[n2] - g_vy[n1]) +
                            f12[2] * (g_vz[n2] - g_vz[n1]);
    }
    accumulate_charge_heat_channel(
      N, n1, n2, r12, f12, g_vx, g_vy, g_vz, g_D_projected, 0, g_channel);
  }
}

static __global__ void find_charge_rate_angular_small_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* g_charge_rate,
  const float* g_D_projected,
  double* g_channel)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  float Fp[MAX_DIM_ANGULAR] = {0.0f};
  float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
  for (int d = 0; d < paramb.dim_angular; ++d)
    Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
  for (int n = 0; n <= paramb.n_max_angular; ++n)
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc)
      sum_fxyz[n * NUM_OF_ABC + abc] =
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    const int index = n1 + N * i1;
    const int n2 = g_NL[index];
    const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float f12[3] = {0.0f, 0.0f, 0.0f};
    find_charge_gradient_angular_pair(paramb, annmb, n1, n2, g_type, r12, Fp, sum_fxyz, f12);
    if (g_charge_rate != nullptr) {
      g_charge_rate[n1] += f12[0] * (g_vx[n2] - g_vx[n1]) +
                            f12[1] * (g_vy[n2] - g_vy[n1]) +
                            f12[2] * (g_vz[n2] - g_vz[n1]);
    }
    accumulate_charge_heat_channel(
      N, n1, n2, r12, f12, g_vx, g_vy, g_vz, g_D_projected, 3, g_channel);
  }
}

static __global__ void subtract_virial_components(
  const int size,
  const double* total,
  const double* nep,
  const double* fixed,
  double* dynamic)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) dynamic[i] = total[i] - nep[i] - fixed[i];
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
  const std::vector<GPU_Vector<double>*>& virial_beads,
  const int active_number_of_beads,
  const bool request_peratom_virial)
{
  const int capacity_number_of_beads = int(position_beads.size());
  const int number_of_beads =
    active_number_of_beads < 0 ? capacity_number_of_beads : active_number_of_beads;
  if (
    number_of_beads < 2 || number_of_beads > capacity_number_of_beads ||
    capacity_number_of_beads < 2 || potential_beads.size() != position_beads.size() ||
    force_beads.size() != position_beads.size() ||
    virial_beads.size() != position_beads.size()) {
    return false;
  }

  const int N = type.size();
  for (int bead_id = 0; bead_id < capacity_number_of_beads; ++bead_id) {
    if (
      position_beads[bead_id]->size() != static_cast<size_t>(N) * 3 ||
      potential_beads[bead_id]->size() != static_cast<size_t>(N) ||
      force_beads[bead_id]->size() != static_cast<size_t>(N) * 3 ||
      virial_beads[bead_id]->size() != static_cast<size_t>(N) * 9) {
      return false;
    }
  }

  begin_force_evaluation_();
  const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);
  const bool profile = pimd_batch_profile_enabled_;
  const auto total_begin = std::chrono::high_resolution_clock::now();
  const auto setup_begin = std::chrono::high_resolution_clock::now();
  initialize_pimd_batch_(
    N, position_beads, potential_beads, force_beads, virial_beads, is_small_box);
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.setup += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - setup_begin).count();
  }
  auto& batch = *pimd_batch_data_;
  const int previous_active_number_of_beads = batch.active_number_of_beads;
  const bool active_lanes_added = number_of_beads > previous_active_number_of_beads;
  if (active_lanes_added) {
    batch.pointer_arrays_initialized = false;
  }
  batch.active_number_of_beads = number_of_beads;
  const bool box_mode_switched =
    batch.box_mode_initialized && (batch.last_box_was_small != is_small_box);
  if (profile && box_mode_switched) {
    ++pimd_batch_timing_.neighbor_box_mode_switches;
  }

  if (is_small_box) {
    // The cached list includes the skin, so explicit images must cover it too.
    const auto neighbor_begin = std::chrono::high_resolution_clock::now();
    get_expanded_box(paramb.rc_radial + 1.0, box, ebox);
    const int small_box_neighbor_size = 2000;
    std::vector<int> initial_flags(capacity_number_of_beads, 0);
    std::vector<int> host_reason_flags;
    const bool first_small_box_build = !batch.small_box_initialized;
    bool box_or_pbc_changed = false;
    if (batch.small_box_initialized) {
      for (int component = 0; component < 9 && !box_or_pbc_changed; ++component) {
        if (batch.small_box_h[component] != box.cpu_h[component]) {
          box_or_pbc_changed = true;
        }
      }
      box_or_pbc_changed =
        box_or_pbc_changed || batch.small_box_pbc[0] != box.pbc_x ||
        batch.small_box_pbc[1] != box.pbc_y || batch.small_box_pbc[2] != box.pbc_z;
    }
    const bool box_changed = first_small_box_build || box_or_pbc_changed;
    if (profile) {
      host_reason_flags.resize(number_of_beads, 0);
      batch.rebuild_reason_flags.fill(0);
      const int host_reason =
        box_or_pbc_changed
          ? NEIGHBOR_BATCH_REBUILD_BOX_OR_PBC
          : (first_small_box_build
               ? (batch.box_mode_initialized
                    ? NEIGHBOR_BATCH_REBUILD_FORCED
                    : NEIGHBOR_BATCH_REBUILD_FIRST_BUILD)
               : (neighbor_always_rebuild_ ? NEIGHBOR_BATCH_REBUILD_FORCED : 0));
      for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
        host_reason_flags[bead_id] = host_reason;
      }
      if (!box_changed && !neighbor_always_rebuild_ && active_lanes_added) {
        std::fill(
          host_reason_flags.begin() + previous_active_number_of_beads,
          host_reason_flags.end(),
          NEIGHBOR_BATCH_REBUILD_FIRST_BUILD);
      }
    }
    if (box_changed || neighbor_always_rebuild_) {
      std::fill(initial_flags.begin(), initial_flags.end(), 1);
    } else if (active_lanes_added) {
      std::fill(
        initial_flags.begin() + previous_active_number_of_beads,
        initial_flags.begin() + number_of_beads,
        1);
    }
    batch.small_box_rebuild_flags.copy_from_host(initial_flags.data());
    if (!box_changed && !neighbor_always_rebuild_) {
      const int lanes_to_check = active_lanes_added
                                   ? previous_active_number_of_beads
                                   : number_of_beads;
      if (lanes_to_check > 0) {
        Neighbor::check_atom_distance_batch(
          box,
          N,
          1.0,
          batch.small_box_x0_ptrs,
          batch.small_box_y0_ptrs,
          batch.small_box_z0_ptrs,
          batch.position_ptrs,
          batch.small_box_rebuild_flags,
          lanes_to_check,
          profile ? &batch.rebuild_reason_flags : nullptr);
      }
    }

    const int block_size = 64;
    const int grid_size = (N2 - N1 - 1) / block_size + 1;
    const dim3 bead_grid(grid_size, number_of_beads);
    initialize_pimd_batch_properties<<<dim3((N - 1) / 128 + 1, number_of_beads), 128>>>(
      N,
      batch.potential_ptrs.data(),
      batch.force_ptrs.data(),
      batch.virial_ptrs.data());
    GPU_CHECK_KERNEL

    const auto neighbor_filter_begin = std::chrono::high_resolution_clock::now();
    find_neighbor_list_small_box_pimd_batch<<<bead_grid, block_size>>>(
      paramb,
      N,
      N1,
      N2,
      number_of_beads,
      small_box_neighbor_size,
      1.0f,
      box,
      ebox,
      type.data(),
      batch.position_ptrs.data(),
      batch.small_box_rebuild_flags.data(),
      batch.small_NN_radial.data(),
      batch.small_NL_radial.data(),
      batch.small_NN_angular.data(),
      batch.small_NL_angular.data(),
      batch.small_x12_radial.data(),
      batch.small_y12_radial.data(),
      batch.small_z12_radial.data(),
      batch.small_x12_angular.data(),
      batch.small_y12_angular.data(),
      batch.small_z12_angular.data(),
      batch.small_image_x_radial.data(),
      batch.small_image_y_radial.data(),
      batch.small_image_z_radial.data(),
      batch.small_image_x_angular.data(),
      batch.small_image_y_angular.data(),
      batch.small_image_z_angular.data());
    GPU_CHECK_KERNEL
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.neighbor_filter += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - neighbor_filter_begin).count();
    }

    long long& num_calls = neighbor_small_box_calls_;
    if (neighbor_diagnostics_enabled_ && num_calls++ % 1000 == 0) {
      std::vector<int> cpu_NN_radial(static_cast<size_t>(number_of_beads) * N);
      std::vector<int> cpu_NN_angular(static_cast<size_t>(number_of_beads) * N);
      batch.small_NN_radial.copy_to_host(cpu_NN_radial.data());
      batch.small_NN_angular.copy_to_host(cpu_NN_angular.data());
      int radial_actual = 0;
      int angular_actual = 0;
      for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
        const size_t atom_offset = static_cast<size_t>(bead_id) * N;
        for (int n = N1; n < N2; ++n) {
          const size_t i = atom_offset + n;
          if (radial_actual < cpu_NN_radial[i]) {
            radial_actual = cpu_NN_radial[i];
          }
          if (angular_actual < cpu_NN_angular[i]) {
            angular_actual = cpu_NN_angular[i];
          }
        }
      }
      std::ofstream output_file("neighbor.out", std::ios_base::app);
      output_file << "PIMD small-box neighbor info at force call " << num_calls - 1 << ": "
                  << "beads=" << number_of_beads << ", radial(max=" << paramb.MN_radial
                  << ",actual=" << radial_actual << "), angular(max=" << paramb.MN_angular
                  << ",actual=" << angular_actual << ")." << std::endl;
    }
    Neighbor::update_reference_positions_batch(
      N,
      batch.position_ptrs,
      batch.small_box_x0_ptrs,
      batch.small_box_y0_ptrs,
      batch.small_box_z0_ptrs,
      batch.small_box_rebuild_flags,
      number_of_beads);
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.neighbor += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - neighbor_begin).count();
    }

    const auto descriptor_begin = std::chrono::high_resolution_clock::now();
    find_descriptor_small_box_pimd_batch<<<bead_grid, block_size>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      number_of_beads,
      small_box_neighbor_size,
      batch.small_NN_radial.data(),
      batch.small_NL_radial.data(),
      batch.small_NN_angular.data(),
      batch.small_NL_angular.data(),
      type.data(),
      batch.small_x12_radial.data(),
      batch.small_y12_radial.data(),
      batch.small_z12_radial.data(),
      batch.small_x12_angular.data(),
      batch.small_y12_angular.data(),
      batch.small_z12_angular.data(),
      batch.potential_ptrs.data(),
      batch.Fp.data(),
      batch.charge_ptrs.data(),
      batch.charge_derivative.data(),
      batch.virial_ptrs.data(),
      batch.sum_fxyz.data());
    GPU_CHECK_KERNEL
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.descriptor += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - descriptor_begin).count();
    }

    // Charge neutrality is required by PPPM/Ewald and is independent of BEC output.
    zero_total_charge_pimd_batch<<<number_of_beads, 1024>>>(
      N, number_of_beads, batch.charge_ptrs.data());
    GPU_CHECK_KERNEL
    if (pimd_batch_bec_enabled_) {
      const auto bec_begin = std::chrono::high_resolution_clock::now();
      find_bec_diagonal_pimd_batch<<<bead_grid, block_size>>>(
        N, number_of_beads, batch.charge_ptrs.data(), batch.bec_ptrs.data());
      GPU_CHECK_KERNEL
      find_bec_radial_small_box_pimd_batch<<<bead_grid, block_size>>>(
        paramb,
        annmb,
        N,
        N1,
        N2,
        number_of_beads,
        small_box_neighbor_size,
        batch.small_NN_radial.data(),
        batch.small_NL_radial.data(),
        type.data(),
        batch.small_x12_radial.data(),
        batch.small_y12_radial.data(),
        batch.small_z12_radial.data(),
        batch.charge_derivative.data(),
        batch.bec_ptrs.data());
      GPU_CHECK_KERNEL
      find_bec_angular_small_box_pimd_batch<<<bead_grid, block_size>>>(
        paramb,
        annmb,
        N,
        N1,
        N2,
        number_of_beads,
        small_box_neighbor_size,
        batch.small_NN_angular.data(),
        batch.small_NL_angular.data(),
        type.data(),
        batch.small_x12_angular.data(),
        batch.small_y12_angular.data(),
        batch.small_z12_angular.data(),
        batch.charge_derivative.data(),
        batch.sum_fxyz.data(),
        batch.bec_ptrs.data());
      GPU_CHECK_KERNEL
      scale_bec_pimd_batch<<<bead_grid, block_size>>>(
        N, number_of_beads, annmb.sqrt_epsilon_inf, batch.bec_ptrs.data());
      GPU_CHECK_KERNEL
      if (profile) {
        CHECK(gpuDeviceSynchronize());
        pimd_batch_timing_.bec += std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - bec_begin).count();
      }
    }

    const auto electrostatics_begin = std::chrono::high_resolution_clock::now();
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
        number_of_beads,
        request_peratom_virial);
      if (pppm.last_batch_used_peratom_virial()) {
        ++pimd_batch_timing_.pppm_full_peratom_virial_batch_calls;
      } else {
        ++pimd_batch_timing_.pppm_global_virial_batch_calls;
      }
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
      find_force_charge_real_space_small_box_pimd_batch<<<bead_grid, block_size>>>(
        N,
        charge_para,
        N1,
        N2,
        number_of_beads,
        small_box_neighbor_size,
        box,
        paramb.rc_radial,
        batch.small_NN_radial.data(),
        batch.small_NL_radial.data(),
        batch.charge_ptrs.data(),
        batch.small_x12_radial.data(),
        batch.small_y12_radial.data(),
        batch.small_z12_radial.data(),
        batch.force_ptrs.data(),
        batch.virial_ptrs.data(),
        batch.potential_ptrs.data(),
        batch.D_real_ptrs.data());
      GPU_CHECK_KERNEL
    }
    zero_mean_D_real_pimd_batch<<<number_of_beads, 1024>>>(
      N, number_of_beads, batch.D_real_ptrs.data());
    GPU_CHECK_KERNEL
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.electrostatics += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - electrostatics_begin).count();
    }

    const auto radial_begin = std::chrono::high_resolution_clock::now();
    find_force_radial_small_box_pimd_batch<<<bead_grid, block_size>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      number_of_beads,
      small_box_neighbor_size,
      batch.small_NN_radial.data(),
      batch.small_NL_radial.data(),
      type.data(),
      batch.small_x12_radial.data(),
      batch.small_y12_radial.data(),
      batch.small_z12_radial.data(),
      batch.Fp.data(),
      batch.charge_derivative.data(),
      batch.D_real_ptrs.data(),
      batch.force_ptrs.data(),
      batch.virial_ptrs.data());
    GPU_CHECK_KERNEL
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.radial += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - radial_begin).count();
    }
    const auto angular_begin = std::chrono::high_resolution_clock::now();
    find_force_angular_small_box_pimd_batch<<<bead_grid, block_size>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      number_of_beads,
      small_box_neighbor_size,
      batch.small_NN_angular.data(),
      batch.small_NL_angular.data(),
      type.data(),
      batch.small_x12_angular.data(),
      batch.small_y12_angular.data(),
      batch.small_z12_angular.data(),
      batch.Fp.data(),
      batch.charge_derivative.data(),
      batch.D_real_ptrs.data(),
      batch.sum_fxyz.data(),
      batch.force_ptrs.data(),
      batch.virial_ptrs.data());
    GPU_CHECK_KERNEL
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.angular += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - angular_begin).count();
    }

    const auto corrections_begin = std::chrono::high_resolution_clock::now();
    for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
      const size_t atom_offset = static_cast<size_t>(bead_id) * N;
      const size_t neighbor_offset = atom_offset * small_box_neighbor_size;
      if (zbl.enabled) {
        find_force_ZBL_small_box<<<grid_size, block_size>>>(
          paramb,
          N,
          zbl,
          N1,
          N2,
          batch.small_NN_angular.data() + atom_offset,
          batch.small_NL_angular.data() + neighbor_offset,
          type.data(),
          batch.small_x12_angular.data() + neighbor_offset,
          batch.small_y12_angular.data() + neighbor_offset,
          batch.small_z12_angular.data() + neighbor_offset,
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
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.corrections += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - corrections_begin).count();
    }
    batch.small_box_initialized = true;
    for (int component = 0; component < 9; ++component) {
      batch.small_box_h[component] = box.cpu_h[component];
    }
    batch.small_box_pbc[0] = box.pbc_x;
    batch.small_box_pbc[1] = box.pbc_y;
    batch.small_box_pbc[2] = box.pbc_z;
    batch.box_mode_initialized = true;
    batch.last_box_was_small = true;
    if (profile) {
      std::vector<int> device_reason_flags(capacity_number_of_beads, 0);
      batch.rebuild_reason_flags.copy_to_host(device_reason_flags.data());
      for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
        host_reason_flags[bead_id] |= device_reason_flags[bead_id];
      }
      Neighbor_Batch_Timing reason_timing;
      Neighbor::accumulate_batch_rebuild_reasons(
        reason_timing, host_reason_flags, number_of_beads);
      Neighbor::accumulate_batch_rebuild_diagnostics(pimd_batch_timing_, reason_timing);
      std::vector<int> device_rebuild_flags(capacity_number_of_beads, 0);
      batch.small_box_rebuild_flags.copy_to_host(device_rebuild_flags.data());
      for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
        const int rebuild = device_rebuild_flags[bead_id] != 0;
        pimd_batch_timing_.neighbor_rebuild_beads += rebuild;
        pimd_batch_timing_.neighbor_small_box_rebuild_beads += rebuild;
      }
    }
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.total += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - total_begin).count();
      ++pimd_batch_timing_.calls;
    }
    return true;
  }

  const auto neighbor_begin = std::chrono::high_resolution_clock::now();
  batch.small_box_initialized = false;
  bool box_or_pbc_changed = false;
  if (batch.large_box_initialized) {
    for (int component = 0; component < 9 && !box_or_pbc_changed; ++component) {
      if (batch.large_box_h[component] != box.cpu_h[component]) {
        box_or_pbc_changed = true;
      }
    }
    box_or_pbc_changed =
      box_or_pbc_changed || batch.large_box_pbc[0] != box.pbc_x ||
      batch.large_box_pbc[1] != box.pbc_y || batch.large_box_pbc[2] != box.pbc_z;
  }
  const bool force_rebuild_all =
    !batch.large_box_initialized || box_or_pbc_changed || box_mode_switched;
  Neighbor_Batch_Timing neighbor_timing;
  Neighbor::find_neighbor_global_batch(
    rc,
    box,
    type,
    N1,
    N2,
    position_beads,
    batch.neighbor_ptrs,
    batch.position_ptrs,
    batch.NN_global_ptrs,
    batch.NL_global_ptrs,
    batch.x0_ptrs,
    batch.y0_ptrs,
    batch.z0_ptrs,
    batch.rebuild_flags,
    batch.any_rebuild,
    batch.x0_ptrs_host,
    batch.y0_ptrs_host,
    batch.z0_ptrs_host,
    batch.pointer_arrays_initialized,
    batch.active_bead_ids,
    batch.cell_count_batch,
    batch.cell_count_sum_batch,
    batch.cell_contents_batch,
    batch.cell_keys_batch,
    batch.cell_stride,
    profile ? &neighbor_timing : nullptr,
    number_of_beads,
    force_rebuild_all,
    true,
    profile ? &batch.rebuild_reason_flags : nullptr,
    box_or_pbc_changed);
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    const double neighbor_time = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - neighbor_begin).count();
    pimd_batch_timing_.neighbor += neighbor_time;
    pimd_batch_timing_.neighbor_global += neighbor_time;
    pimd_batch_timing_.neighbor_pointer += neighbor_timing.pointer_setup;
    pimd_batch_timing_.neighbor_check += neighbor_timing.distance_check;
    pimd_batch_timing_.neighbor_flags += neighbor_timing.flag_transfer;
    pimd_batch_timing_.neighbor_rebuild += neighbor_timing.rebuild;
    Neighbor::accumulate_batch_rebuild_diagnostics(pimd_batch_timing_, neighbor_timing);
  }
  batch.large_box_initialized = true;
  for (int component = 0; component < 9; ++component) {
    batch.large_box_h[component] = box.cpu_h[component];
  }
  batch.large_box_pbc[0] = box.pbc_x;
  batch.large_box_pbc[1] = box.pbc_y;
  batch.large_box_pbc[2] = box.pbc_z;
  batch.box_mode_initialized = true;
  batch.last_box_was_small = false;

  const int block_size = 64;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const dim3 grid(grid_size, number_of_beads);
  constexpr int angular_force_shards = 4;
  const dim3 angular_grid(grid_size, number_of_beads, angular_force_shards);
  const auto initialize_begin = std::chrono::high_resolution_clock::now();
  initialize_pimd_batch_properties<<<dim3((N - 1) / 128 + 1, number_of_beads), 128>>>(
    N,
    batch.potential_ptrs.data(),
    batch.force_ptrs.data(),
    batch.virial_ptrs.data());
  GPU_CHECK_KERNEL
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.initialize += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - initialize_begin).count();
  }

  const auto neighbor_filter_begin = std::chrono::high_resolution_clock::now();
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
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    const double neighbor_filter_time = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - neighbor_filter_begin).count();
    pimd_batch_timing_.neighbor += neighbor_filter_time;
    pimd_batch_timing_.neighbor_filter += neighbor_filter_time;
  }

  const auto descriptor_begin = std::chrono::high_resolution_clock::now();
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
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.descriptor += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - descriptor_begin).count();
  }

  const dim3 bead_grid(grid_size, number_of_beads);
  // Charge neutrality is required by PPPM/Ewald and is independent of BEC output.
  zero_total_charge_pimd_batch<<<number_of_beads, 1024>>>(
    N, number_of_beads, batch.charge_ptrs.data());
  GPU_CHECK_KERNEL
  if (pimd_batch_bec_enabled_) {
    const auto bec_begin = std::chrono::high_resolution_clock::now();
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
    if (profile) {
      CHECK(gpuDeviceSynchronize());
      pimd_batch_timing_.bec += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - bec_begin).count();
    }
  }

  const auto electrostatics_begin = std::chrono::high_resolution_clock::now();
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
      number_of_beads,
      request_peratom_virial);
    if (pppm.last_batch_used_peratom_virial()) {
      ++pimd_batch_timing_.pppm_full_peratom_virial_batch_calls;
    } else {
      ++pimd_batch_timing_.pppm_global_virial_batch_calls;
    }
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
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.electrostatics += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - electrostatics_begin).count();
  }

  const auto radial_begin = std::chrono::high_resolution_clock::now();
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
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.radial += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - radial_begin).count();
  }
  const auto angular_begin = std::chrono::high_resolution_clock::now();
  if (paramb.L_max <= 4) {
    find_partial_force_angular_pimd_batch<24><<<angular_grid, block_size>>>(
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
  } else {
    find_partial_force_angular_pimd_batch<NUM_OF_ABC><<<angular_grid, block_size>>>(
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
  }
  GPU_CHECK_KERNEL
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.angular += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - angular_begin).count();
  }
  const auto many_body_begin = std::chrono::high_resolution_clock::now();
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
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.many_body += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - many_body_begin).count();
  }

  const auto corrections_begin = std::chrono::high_resolution_clock::now();
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
  if (profile) {
    CHECK(gpuDeviceSynchronize());
    pimd_batch_timing_.corrections += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - corrections_begin).count();
    pimd_batch_timing_.total += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - total_begin).count();
    ++pimd_batch_timing_.calls;
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
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      false,
      false);
  } else {
    compute_large_box(
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      false,
      false);
  }
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
  // Keep the shared charge output neutral after a descriptor-only pass.
  zero_total_charge<<<1, 1024>>>(type.size(), nep_data.charge.data());
  GPU_CHECK_KERNEL
}

const GPU_Vector<int>& NEP_Charge::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP_Charge::get_NL_radial_ptr() { return nep_data.NL_radial; }

GPU_Vector<float>& NEP_Charge::get_charge_reference() { return nep_data.charge; }

GPU_Vector<float>& NEP_Charge::get_bec_reference() { return nep_data.bec; }

void NEP_Charge::configure_mechanical_observer()
{
  need_bec = false;
  md_qnep_bec_enabled_ = false;
  pimd_batch_bec_enabled_ = false;
  neighbor_diagnostics_enabled_ = false;
  charge_diagnostics_requested_ = false;
  dynamic_charge_diagnostics_enabled_ = false;
}

void NEP_Charge::enable_charge_diagnostics()
{
  if (charge_diagnostics_enabled_) return;
  charge_diagnostics_enabled_ = true;
  charge_diagnostics_requested_ = false;
  const int N = nep_data.charge.size();
  nep_data.charge_raw.resize(N);
  nep_data.D_raw.resize(N);
  nep_data.D_projected.resize(N);
  nep_data.charge_rate_raw.resize(N);
  nep_data.charge_rate.resize(N);
}

void NEP_Charge::enable_dynamic_charge_diagnostics()
{
  dynamic_charge_diagnostics_enabled_ = true;
  pppm.enable_dynamic_charge_diagnostics();
}

void NEP_Charge::enable_delta_j_q_k_diagnostics()
{
  ewald.initialize(charge_para.alpha);
}

void NEP_Charge::reset_dynamic_charge_cache()
{
  neighbor_manager.invalidate_reference_positions();
  single_frame_neighbor_invalidation_pending_ = false;
  pppm.reset_dynamic_charge_cache();
  dynamic_q_cache_set_ = false;
  dynamic_q_cache_result_valid_ = false;
  dynamic_q_cache_pppm_valid_ = false;
  dynamic_q_cache_diagnostic_recorded_ = false;
  dynamic_q_cache_diagnostic_checks_pass_ = false;
  dynamic_q_last_diagnostic_checks_pass_ = false;
  dynamic_q_cache_N_ = -1;
  dynamic_q_cache_step_ = -1;
  dynamic_q_cache_bead_ = -1;
  dynamic_q_cache_N1_ = -1;
  dynamic_q_cache_N2_ = -1;
  dynamic_q_cache_time_fs_ = 0.0;
  dynamic_q_cache_force_evaluation_id_ = 0;
  dynamic_q_cache_charge_rate_generation_ = 0;
  dynamic_q_cache_position_ = nullptr;
  dynamic_q_last_pppm_valid_ = false;
  charge_rate_cache_set_ = false;
  charge_heat_channel_cache_set_ = false;
  charge_heat_channel_cache_position_ = nullptr;
  charge_heat_channel_cache_velocity_ = nullptr;
  full_a_current_cache_set_ = false;
  charge_diagnostics_available_ = false;
  charge_diagnostics_force_evaluation_id_ = 0;
}

void NEP_Charge::request_charge_diagnostics_for_next_force()
{
  charge_diagnostics_requested_ = true;
}

void NEP_Charge::request_peratom_virial_for_next_force()
{
  peratom_virial_requested_ = true;
}

void NEP_Charge::compute_charge_rate(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& velocity,
  GPU_Vector<double>* channel_per_atom,
  const bool update_charge_rate)
{
  const int N = nep_data.charge.size();
  const int block_size = 64;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  float* charge_rate = nullptr;
  const float* D_projected = nullptr;
  double* channel = nullptr;
  if (
    !update_charge_rate && channel_per_atom != nullptr && charge_heat_channel_cache_set_ &&
    charge_heat_channel_cache_force_evaluation_id_ == force_evaluation_id_ &&
    charge_heat_channel_cache_position_ == position.data() &&
    charge_heat_channel_cache_velocity_ == velocity.data() &&
    charge_heat_channel_cache_.size() == static_cast<size_t>(N) * 6) {
    if (channel_per_atom->size() != static_cast<size_t>(N) * 6)
      channel_per_atom->resize(static_cast<size_t>(N) * 6);
    channel_per_atom->copy_from_device(charge_heat_channel_cache_.data());
    return;
  }
  if (update_charge_rate) {
    charge_rate_cache_set_ = false;
    dynamic_q_cache_set_ = false;
    full_a_current_cache_set_ = false;
    charge_heat_channel_cache_set_ = false;
    nep_data.charge_rate_raw.fill(0.0f);
    charge_rate = nep_data.charge_rate_raw.data();
  }
  if (channel_per_atom != nullptr) {
    if (channel_per_atom->size() != static_cast<size_t>(N) * 6)
      channel_per_atom->resize(static_cast<size_t>(N) * 6);
    channel_per_atom->fill(0.0);
    D_projected = nep_data.D_projected.data();
    channel = channel_per_atom->data();
    if (charge_heat_channel_cache_.size() != static_cast<size_t>(N) * 6)
      charge_heat_channel_cache_.resize(static_cast<size_t>(N) * 6);
  }
  if (get_expanded_box(paramb.rc_radial, box, ebox)) {
    const int size_x12 = small_box_data.r12.size() / 6;
    find_charge_rate_radial_small_box<<<grid_size, block_size>>>(
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
      velocity.data(),
      velocity.data() + N,
      velocity.data() + N * 2,
      nep_data.charge_derivative.data(),
      charge_rate,
      D_projected,
      channel);
    find_charge_rate_angular_small_box<<<grid_size, block_size>>>(
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
      velocity.data(),
      velocity.data() + N,
      velocity.data() + N * 2,
      nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data(),
      charge_rate,
      D_projected,
      channel);
  } else {
    find_charge_rate_radial<<<grid_size, block_size>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + N * 2,
      velocity.data(),
      velocity.data() + N,
      velocity.data() + N * 2,
      nep_data.charge_derivative.data(),
      charge_rate,
      D_projected,
      channel);
    find_charge_rate_angular<<<grid_size, block_size>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + N * 2,
      velocity.data(),
      velocity.data() + N,
      velocity.data() + N * 2,
      nep_data.charge_derivative.data(),
      nep_data.sum_fxyz.data(),
      charge_rate,
      D_projected,
      channel);
  }
  GPU_CHECK_KERNEL
  if (channel_per_atom != nullptr) {
    charge_heat_channel_cache_.copy_from_device(channel_per_atom->data());
    charge_heat_channel_cache_set_ = true;
    charge_heat_channel_cache_force_evaluation_id_ = force_evaluation_id_;
    charge_heat_channel_cache_position_ = position.data();
    charge_heat_channel_cache_velocity_ = velocity.data();
  }
  if (update_charge_rate) {
    nep_data.charge_rate.copy_from_device(nep_data.charge_rate_raw.data());
    zero_total_charge<<<1, 1024>>>(N, nep_data.charge_rate.data());
    GPU_CHECK_KERNEL
    charge_rate_cache_set_ = true;
    charge_rate_cache_force_evaluation_id_ = force_evaluation_id_;
    charge_rate_cache_position_ = position.data();
    charge_rate_cache_velocity_ = velocity.data();
    ++charge_rate_generation_;
    dynamic_q_cache_set_ = false;
    full_a_current_cache_set_ = false;
  }
}

void NEP_Charge::compute_charge_rate_for_current_force_frame(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& velocity,
  GPU_Vector<double>* channel_per_atom)
{
  const int N = nep_data.charge.size();
  const bool charge_rate_cache_match =
    charge_rate_cache_set_ && charge_rate_cache_force_evaluation_id_ == force_evaluation_id_ &&
    charge_rate_cache_position_ == position.data() && charge_rate_cache_velocity_ == velocity.data() &&
    nep_data.charge_rate_raw.size() == static_cast<size_t>(N);
  const bool channel_cache_match =
    charge_heat_channel_cache_set_ &&
    charge_heat_channel_cache_force_evaluation_id_ == force_evaluation_id_ &&
    charge_heat_channel_cache_position_ == position.data() &&
    charge_heat_channel_cache_velocity_ == velocity.data() &&
    charge_heat_channel_cache_.size() == static_cast<size_t>(N) * 6;
  if (channel_per_atom == nullptr && charge_rate_cache_match) {
    return;
  }
  if (channel_per_atom != nullptr && charge_rate_cache_match && channel_cache_match) {
    if (channel_per_atom->size() != static_cast<size_t>(N) * 6)
      channel_per_atom->resize(static_cast<size_t>(N) * 6);
    channel_per_atom->copy_from_device(charge_heat_channel_cache_.data());
    return;
  }
  if (channel_per_atom != nullptr && charge_rate_cache_match) {
    compute_charge_rate(box, type, position, velocity, channel_per_atom, false);
    return;
  }
  compute_charge_rate(box, type, position, velocity, channel_per_atom, true);
}

void NEP_Charge::notify_velocity_update()
{
  charge_rate_cache_set_ = false;
  charge_heat_channel_cache_set_ = false;
  charge_heat_channel_cache_position_ = nullptr;
  charge_heat_channel_cache_velocity_ = nullptr;
  dynamic_q_cache_set_ = false;
  full_a_current_cache_set_ = false;
  dynamic_q_last_pppm_valid_ = false;
  dynamic_q_last_diagnostic_checks_pass_ = false;
}

bool NEP_Charge::get_cached_full_a_current(
  const int N,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& unwrapped_position,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential,
  const GPU_Vector<double>& virial,
  const GPU_Vector<double>& velocity,
  const double delta_j_q_total[3],
  double j_conv[3],
  double j_virial[3],
  double j_base[3],
  double j_projection[3][3],
  double j_candidate_a[3]) const
{
  const bool cache_match =
    full_a_current_cache_set_ && N == full_a_current_cache_N_ &&
    full_a_current_cache_force_evaluation_id_ == force_evaluation_id_ &&
    full_a_current_cache_charge_rate_generation_ == charge_rate_generation_ &&
    position.data() == full_a_current_cache_position_ &&
    unwrapped_position.data() == full_a_current_cache_unwrapped_position_ &&
    mass.data() == full_a_current_cache_mass_ &&
    potential.data() == full_a_current_cache_potential_ &&
    virial.data() == full_a_current_cache_virial_ &&
    velocity.data() == full_a_current_cache_velocity_ &&
    delta_j_q_total[0] == full_a_current_cache_delta_j_q_total_[0] &&
    delta_j_q_total[1] == full_a_current_cache_delta_j_q_total_[1] &&
    delta_j_q_total[2] == full_a_current_cache_delta_j_q_total_[2];
  if (!cache_match) return false;

  for (int d = 0; d < 3; ++d) {
    j_conv[d] = full_a_current_cache_j_conv_[d];
    j_virial[d] = full_a_current_cache_j_virial_[d];
    j_base[d] = full_a_current_cache_j_base_[d];
    j_candidate_a[d] = full_a_current_cache_j_candidate_a_[d];
    for (int x = 0; x < 3; ++x)
      j_projection[x][d] = full_a_current_cache_j_projection_[x * 3 + d];
  }
  return true;
}

void NEP_Charge::cache_full_a_current(
  const int N,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& unwrapped_position,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential,
  const GPU_Vector<double>& virial,
  const GPU_Vector<double>& velocity,
  const double delta_j_q_total[3],
  const double j_conv[3],
  const double j_virial[3],
  const double j_base[3],
  const double j_projection[3][3],
  const double j_candidate_a[3])
{
  full_a_current_cache_set_ = true;
  full_a_current_cache_N_ = N;
  full_a_current_cache_force_evaluation_id_ = force_evaluation_id_;
  full_a_current_cache_charge_rate_generation_ = charge_rate_generation_;
  full_a_current_cache_position_ = position.data();
  full_a_current_cache_unwrapped_position_ = unwrapped_position.data();
  full_a_current_cache_mass_ = mass.data();
  full_a_current_cache_potential_ = potential.data();
  full_a_current_cache_virial_ = virial.data();
  full_a_current_cache_velocity_ = velocity.data();
  for (int d = 0; d < 3; ++d) {
    full_a_current_cache_delta_j_q_total_[d] = delta_j_q_total[d];
    full_a_current_cache_j_conv_[d] = j_conv[d];
    full_a_current_cache_j_virial_[d] = j_virial[d];
    full_a_current_cache_j_base_[d] = j_base[d];
    full_a_current_cache_j_candidate_a_[d] = j_candidate_a[d];
    for (int x = 0; x < 3; ++x)
      full_a_current_cache_j_projection_[x * 3 + d] = j_projection[x][d];
  }
}

void NEP_Charge::compute_charge_heat_channels(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& velocity,
  GPU_Vector<double>& channel_per_atom)
{
  compute_charge_rate(box, type, position, velocity, &channel_per_atom, false);
}

bool NEP_Charge::compute_dynamic_charge_correction_impl(
  const int N,
  const int N1,
  const int N2,
  const int bead_id,
  const int step,
  const double time_fs,
  const Box& box,
  const GPU_Vector<double>& position,
  const bool record_diagnostic,
  const bool write_debug,
  double* delta_j_q_pppm,
  double* delta_j_q_real,
  double* delta_j_q_total)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dynamic_q_last_pppm_valid_ = false;
  dynamic_q_last_diagnostic_checks_pass_ = false;
  auto set_output_nan = [nan](double* output) {
    if (output != nullptr) {
      output[0] = nan;
      output[1] = nan;
      output[2] = nan;
    }
  };
  set_output_nan(delta_j_q_pppm);
  set_output_nan(delta_j_q_real);
  set_output_nan(delta_j_q_total);

  if (
    N <= 0 || N != static_cast<int>(nep_data.charge.size()) || N1 != 0 || N2 != N || this->N1 != 0 ||
    this->N2 != N) {
    std::cerr << "NEP dynamic-q diagnostic: 4A requires the full atom range N1=0,N2=N."
              << std::endl;
    return false;
  }
  if (!use_pppm) {
    std::cerr << "NEP dynamic-q diagnostic: 4A requires kspace_method pppm." << std::endl;
    return false;
  }
  if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1 || !box.is_orthogonal) {
    std::cerr << "NEP dynamic-q diagnostic: 4A requires a three-dimensional orthogonal periodic box."
              << std::endl;
    return false;
  }
  if (position.size() != static_cast<size_t>(3 * N) || nep_data.charge_rate.size() != static_cast<size_t>(N)) {
    std::cerr << "NEP dynamic-q diagnostic: charge, charge-rate, and position sizes are inconsistent."
              << std::endl;
    return false;
  }
  if (paramb.charge_mode != 1 && paramb.charge_mode != 2) {
    std::cerr << "NEP dynamic-q diagnostic: unsupported charge mode." << std::endl;
    return false;
  }

  const bool cache_match =
    dynamic_q_cache_set_ && N == dynamic_q_cache_N_ && step == dynamic_q_cache_step_ &&
    bead_id == dynamic_q_cache_bead_ && N1 == dynamic_q_cache_N1_ && N2 == dynamic_q_cache_N2_ &&
    time_fs == dynamic_q_cache_time_fs_ &&
    dynamic_q_cache_force_evaluation_id_ == force_evaluation_id_ &&
    dynamic_q_cache_charge_rate_generation_ == charge_rate_generation_ &&
    position.data() == dynamic_q_cache_position_ &&
    (!record_diagnostic || dynamic_q_cache_diagnostic_recorded_);
  if (cache_match) {
    dynamic_q_last_pppm_valid_ = dynamic_q_cache_pppm_valid_;
    dynamic_q_last_diagnostic_checks_pass_ = dynamic_q_cache_diagnostic_checks_pass_;
    if (dynamic_q_cache_pppm_valid_ && delta_j_q_pppm != nullptr) {
      for (int d = 0; d < 3; ++d) delta_j_q_pppm[d] = dynamic_q_cache_delta_j_pppm_[d];
    }
    if (dynamic_q_cache_result_valid_) {
      if (delta_j_q_real != nullptr) {
        for (int d = 0; d < 3; ++d) delta_j_q_real[d] = dynamic_q_cache_delta_j_real_[d];
      }
      if (delta_j_q_total != nullptr) {
        for (int d = 0; d < 3; ++d) delta_j_q_total[d] = dynamic_q_cache_delta_j_total_[d];
      }
    }
    return dynamic_q_cache_result_valid_;
  }

  auto cache_result = [&](const bool result_valid,
                          const bool pppm_valid,
                          const double pppm_result[3],
                          const double real_result[3],
                          const double total_result[3]) {
    dynamic_q_cache_set_ = true;
    dynamic_q_cache_result_valid_ = result_valid;
    dynamic_q_cache_pppm_valid_ = pppm_valid;
    dynamic_q_cache_diagnostic_recorded_ = record_diagnostic;
    dynamic_q_cache_diagnostic_checks_pass_ =
      record_diagnostic && pppm.get_last_dynamic_q_diagnostic_checks_pass();
    dynamic_q_last_diagnostic_checks_pass_ = dynamic_q_cache_diagnostic_checks_pass_;
    dynamic_q_cache_N_ = N;
    dynamic_q_cache_step_ = step;
    dynamic_q_cache_bead_ = bead_id;
    dynamic_q_cache_N1_ = N1;
    dynamic_q_cache_N2_ = N2;
    dynamic_q_cache_time_fs_ = time_fs;
    dynamic_q_cache_force_evaluation_id_ = force_evaluation_id_;
    dynamic_q_cache_charge_rate_generation_ = charge_rate_generation_;
    dynamic_q_cache_position_ = position.data();
    for (int d = 0; d < 3; ++d) {
      dynamic_q_cache_delta_j_pppm_[d] = pppm_valid ? pppm_result[d] : nan;
      dynamic_q_cache_delta_j_real_[d] = result_valid ? real_result[d] : nan;
      dynamic_q_cache_delta_j_total_[d] = result_valid ? total_result[d] : nan;
    }
  };
  auto finalize_diagnostic = [&](const double* real_result,
                                 const double* total_result,
                                 const bool result_valid) {
    if (record_diagnostic) {
      pppm.finalize_dynamic_charge_diagnostic(
        real_result, total_result, result_valid, paramb.charge_mode);
    }
  };

  double pppm_result[3] = {nan, nan, nan};
  const bool pppm_valid = record_diagnostic
    ? pppm.diagnose_dynamic_charge(
        N,
        N1,
        N2,
        bead_id,
        step,
        time_fs,
        box,
        nep_data.charge,
        nep_data.charge_rate,
        position,
        write_debug,
        pppm_result)
    : pppm.compute_dynamic_charge_correction(
        N,
        N1,
        N2,
        box,
        nep_data.charge,
        nep_data.charge_rate,
        position,
        pppm_result,
        force_evaluation_id_);
  dynamic_q_last_pppm_valid_ = pppm_valid;
  if (pppm_valid && delta_j_q_pppm != nullptr) {
    for (int d = 0; d < 3; ++d) delta_j_q_pppm[d] = pppm_result[d];
  }

  double real_result[3] = {nan, nan, nan};
  double total_result[3] = {nan, nan, nan};
  if (!pppm_valid) {
    for (int d = 0; d < 3; ++d) pppm_result[d] = nan;
    cache_result(false, false, pppm_result, real_result, total_result);
    finalize_diagnostic(real_result, total_result, false);
    return false;
  }

  if (paramb.charge_mode == 2) {
    real_result[0] = 0.0;
    real_result[1] = 0.0;
    real_result[2] = 0.0;
  } else {
    const int block_size = 64;
    const int grid_size = (N2 - N1 - 1) / block_size + 1;
    if (dynamic_q_real_per_atom_.size() != static_cast<size_t>(3 * N))
      dynamic_q_real_per_atom_.resize(static_cast<size_t>(3 * N));
    if (dynamic_q_real_total_.size() != 3) dynamic_q_real_total_.resize(3);

    const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);
    if (is_small_box) {
      const size_t size_x12 = small_box_data.r12.size() / 6;
      if (small_box_data.NN_radial.size() != static_cast<size_t>(N) ||
          small_box_data.NL_radial.size() < size_x12 || size_x12 < static_cast<size_t>(N) ||
          small_box_data.r12.size() < static_cast<size_t>(6 * N)) {
        std::cerr << "NEP dynamic-q diagnostic: small-box radial neighbor data is unavailable."
                  << std::endl;
        cache_result(false, true, pppm_result, real_result, total_result);
        finalize_diagnostic(real_result, total_result, false);
        return false;
      }
      find_delta_j_q_real_space_small_box<<<grid_size, block_size>>>(
        N,
        N1,
        N2,
        charge_para,
        small_box_data.NN_radial.data(),
        small_box_data.NL_radial.data(),
        nep_data.charge.data(),
        nep_data.charge_rate.data(),
        small_box_data.r12.data(),
        small_box_data.r12.data() + size_x12,
        small_box_data.r12.data() + size_x12 * 2,
        dynamic_q_real_per_atom_.data());
    } else {
      if (nep_data.NN_radial.size() != static_cast<size_t>(N) ||
          nep_data.NL_radial.size() < static_cast<size_t>(N)) {
        std::cerr << "NEP dynamic-q diagnostic: radial neighbor data is unavailable." << std::endl;
        cache_result(false, true, pppm_result, real_result, total_result);
        finalize_diagnostic(real_result, total_result, false);
        return false;
      }
      find_delta_j_q_real_space<<<grid_size, block_size>>>(
        N,
        N1,
        N2,
        charge_para,
        box,
        nep_data.NN_radial.data(),
        nep_data.NL_radial.data(),
        nep_data.charge.data(),
        nep_data.charge_rate.data(),
        position.data(),
        position.data() + N,
        position.data() + 2 * N,
        dynamic_q_real_per_atom_.data());
    }
    GPU_CHECK_KERNEL
    reduce_delta_j_q_real_space<<<3, 1024>>>(
      N, N1, N2, dynamic_q_real_per_atom_.data(), dynamic_q_real_total_.data());
    GPU_CHECK_KERNEL
    dynamic_q_real_total_.copy_to_host(real_result);
  }

  bool real_valid = true;
  for (int d = 0; d < 3; ++d) real_valid = real_valid && std::isfinite(real_result[d]);
  if (!real_valid) {
    cache_result(false, true, pppm_result, real_result, total_result);
    finalize_diagnostic(real_result, total_result, false);
    return false;
  }
  for (int d = 0; d < 3; ++d) total_result[d] = pppm_result[d] + real_result[d];
  bool total_valid = true;
  for (int d = 0; d < 3; ++d) total_valid = total_valid && std::isfinite(total_result[d]);
  if (!total_valid) {
    cache_result(false, true, pppm_result, real_result, total_result);
    finalize_diagnostic(real_result, total_result, false);
    return false;
  }

  cache_result(true, true, pppm_result, real_result, total_result);
  finalize_diagnostic(real_result, total_result, true);
  if (delta_j_q_real != nullptr) {
    for (int d = 0; d < 3; ++d) delta_j_q_real[d] = real_result[d];
  }
  if (delta_j_q_total != nullptr) {
    for (int d = 0; d < 3; ++d) delta_j_q_total[d] = total_result[d];
  }
  return true;
}

bool NEP_Charge::compute_dynamic_charge_correction(
  const int N,
  const int N1,
  const int N2,
  const int bead_id,
  const int step,
  const double time_fs,
  const Box& box,
  const GPU_Vector<double>& position,
  double* delta_j_q_pppm,
  double* delta_j_q_real,
  double* delta_j_q_total)
{
  return compute_dynamic_charge_correction_impl(
    N,
    N1,
    N2,
    bead_id,
    step,
    time_fs,
    box,
    position,
    false,
    false,
    delta_j_q_pppm,
    delta_j_q_real,
    delta_j_q_total);
}

bool NEP_Charge::diagnose_dynamic_charge(
  const int N,
  const int N1,
  const int N2,
  const int bead_id,
  const int step,
  const double time_fs,
  const Box& box,
  const GPU_Vector<double>& position,
  const bool write_debug,
  double* delta_j_q_pppm,
  double* delta_j_q_real,
  double* delta_j_q_total)
{
  enable_dynamic_charge_diagnostics();
  return compute_dynamic_charge_correction_impl(
    N,
    N1,
    N2,
    bead_id,
    step,
    time_fs,
    box,
    position,
    dynamic_charge_diagnostics_enabled_,
    write_debug,
    delta_j_q_pppm,
    delta_j_q_real,
    delta_j_q_total);
}

int NEP_Charge::compute_delta_j_q_k(
  const Box& box,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& delta_j_q_k,
  double& sum_charge,
  double& sum_charge_rate)
{
  const int N = nep_data.charge.size();
  return ewald.compute_delta_j_q_k(
    N,
    N1,
    N2,
    box.cpu_h,
    nep_data.charge,
    nep_data.charge_rate,
    position,
    delta_j_q_k,
    sum_charge,
    sum_charge_rate);
}

void NEP_Charge::compute_virial_components(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& total_virial,
  const bool need_nep,
  const bool need_electrostatic_fixed,
  const bool need_dynamic_charge,
  GPU_Vector<double>& virial_nep,
  GPU_Vector<double>& virial_electrostatic_fixed,
  GPU_Vector<double>& virial_dynamic_charge)
{
  const int N = type.size();
  const int virial_size = N * 9;
  GPU_Vector<double>& potential = virial_diag_potential_;
  GPU_Vector<double>& force = virial_diag_force_;
  GPU_Vector<float>& charge_saved = virial_diag_charge_saved_;
  GPU_Vector<float>& D_real_saved = virial_diag_D_real_saved_;
  if (need_nep || need_electrostatic_fixed) {
    if (virial_diag_N_ != N) {
      potential.resize(N, 0.0);
      force.resize(N * 3, 0.0);
      charge_saved.resize(N);
      D_real_saved.resize(N);
      virial_diag_N_ = N;
    }
    charge_saved.copy_from_device(nep_data.charge.data());
    D_real_saved.copy_from_device(nep_data.D_real.data());
  }
  if (need_electrostatic_fixed) {
    potential.fill(0.0);
    force.fill(0.0);
    virial_electrostatic_fixed.fill(0.0);
    if (use_pppm) {
      pppm.find_force(
        N,
        N1,
        N2,
        box,
        nep_data.charge,
        position,
        nep_data.D_real,
        force,
        virial_electrostatic_fixed,
        potential,
        true);
    } else {
      ewald.find_force(
        N,
        N1,
        N2,
        box.cpu_h,
        nep_data.charge,
        position,
        nep_data.D_real,
        force,
        virial_electrostatic_fixed,
        potential);
    }

    const int block_size = 64;
    const int grid_size = (N2 - N1 - 1) / block_size + 1;
    if (paramb.charge_mode == 1) {
      if (get_expanded_box(paramb.rc_radial, box, ebox)) {
        const int size_x12 = small_box_data.r12.size() / 6;
        find_force_charge_real_space_small_box<<<grid_size, block_size>>>(
          N,
          charge_para,
          N1,
          N2,
          box,
          paramb.rc_radial,
          small_box_data.NN_radial.data(),
          small_box_data.NL_radial.data(),
          nep_data.charge.data(),
          small_box_data.r12.data(),
          small_box_data.r12.data() + size_x12,
          small_box_data.r12.data() + size_x12 * 2,
          force.data(),
          force.data() + N,
          force.data() + N * 2,
          virial_electrostatic_fixed.data(),
          potential.data(),
          nep_data.D_real.data());
      } else {
        find_force_charge_real_space<<<grid_size, block_size>>>(
          N,
          charge_para,
          N1,
          N2,
          box,
          nep_data.NN_radial.data(),
          nep_data.NL_radial.data(),
          nep_data.charge.data(),
          position.data(),
          position.data() + N,
          position.data() + N * 2,
          force.data(),
          force.data() + N,
          force.data() + N * 2,
          virial_electrostatic_fixed.data(),
          potential.data(),
          nep_data.D_real.data());
      }
      GPU_CHECK_KERNEL
    }
    zero_mean_D_real<<<1, 1024>>>(N, nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  if (need_nep) {
    potential.fill(0.0);
    force.fill(0.0);
    virial_nep.fill(0.0);
    // compute_non_electro() intentionally neutralizes its shared charge
    // output.  Run it after the fixed-charge component so that the latter
    // uses the projected charge from the same qNEP force evaluation.
    compute_non_electro(box, type, position, potential, force, virial_nep);
  }

  if (need_dynamic_charge) {
    virial_dynamic_charge.fill(0.0);
    subtract_virial_components<<<(virial_size - 1) / 128 + 1, 128>>>(
      virial_size,
      total_virial.data(),
      virial_nep.data(),
      virial_electrostatic_fixed.data(),
      virial_dynamic_charge.data());
    GPU_CHECK_KERNEL
  }

  // This helper is diagnostic-only; do not leave shared qNEP outputs changed
  // by its descriptor/electrostatic recomputation.
  if (need_nep || need_electrostatic_fixed) {
    nep_data.charge.copy_from_device(charge_saved.data());
    nep_data.D_real.copy_from_device(D_real_saved.data());
  }
}
