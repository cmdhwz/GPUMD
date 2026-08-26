/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version. GPUMD is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have
   received a copy of the GNU General Public License along with GPUMD.  If not, see
   <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The driver class calculating force and related quantities.
------------------------------------------------------------------------------*/

#ifdef USE_DEEPMD
#include "dp.cuh"
#endif
#ifdef USE_NNAP
#include "nnap.cuh"
#endif
#include "adp.cuh"
#include "eam.cuh"
#include "eam_alloy.cuh"
#include "fcp.cuh"
#include "force.cuh"
#include "ilp_nep.cuh"
#include "ilp_tmd_sw.cuh"
#include "ilp_tersoff.cuh"
#include "lj.cuh"
#include "nep.cuh"
#include "nep_multigpu.cuh"
#include "nep_charge.cuh"
#include "potential.cuh"
#include "tersoff1988.cuh"
#include "tersoff1989.cuh"
#include "tersoff_mini.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#define BLOCK_SIZE 128

static __global__ void initialize_properties(
  int number_of_atoms,
  double* force_x,
  double* force_y,
  double* force_z,
  double* potential,
  double* virial);

static __global__ void gpu_apply_pbc(int N, Box box, double* g_x, double* g_y, double* g_z);

template <typename T>
static void copy_gpu_buffer_between_devices(
  const int dst_device, T* dst, const int src_device, const T* src, const size_t count)
{
  if (count == 0) {
    return;
  }
  const size_t bytes = sizeof(T) * count;
  if (dst_device == src_device) {
    CHECK(gpuSetDevice(dst_device));
    CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToDevice));
  } else {
    CHECK(gpuMemcpyPeer(dst, dst_device, src, src_device, bytes));
  }
}

static void ensure_pimd_worker_bead_buffers(
  Force::PIMD_Bead_GPU_Worker& worker, const int number_of_beads, const int number_of_atoms)
{
  if (!worker.position_beads.empty()) {
    if (int(worker.position_beads.size()) != number_of_beads) {
      PRINT_INPUT_ERROR("Cannot change the number of PIMD beads between runs.\n");
    }
    return;
  }

  worker.position_beads.resize(number_of_beads);
  worker.potential_beads.resize(number_of_beads);
  worker.force_beads.resize(number_of_beads);
  worker.virial_beads.resize(number_of_beads);
  for (int bead = 0; bead < number_of_beads; ++bead) {
    worker.position_beads[bead].resize(number_of_atoms * 3);
    worker.potential_beads[bead].resize(number_of_atoms);
    worker.force_beads[bead].resize(number_of_atoms * 3);
    worker.virial_beads[bead].resize(number_of_atoms * 9);
  }
}

Force::Force(void)
{
  is_fcp = false;
  has_non_nep = false;
}

void Force::check_types(const char* file_potential)
{
  std::ifstream input(file_potential);
  std::vector<std::string> tokens = get_tokens(input);
  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  for (int n = 0; n < num_types; ++n) {
    std::string token = tokens[2 + n];
    if (potentials.size() == 0) {
      atom_types[n] = token;
    } else {
      if (token != atom_types[n]) {
        PRINT_INPUT_ERROR(
          "The atomic species and/or the order of the species are not consistent "
          "between the multiple potentials.\n");
      }
    }
  }
}

void Force::parse_potential(
  const char** param, int num_param, const Box& box, const int number_of_atoms)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("potential should have 1 or 2 parameters.\n");
  }

  std::unique_ptr<Potential> potential;
  FILE* fid_potential = my_fopen(param[1], "r");
  char potential_name[100];
  int count = fscanf(fid_potential, "%s", potential_name);
  if (count != 1) {
    PRINT_INPUT_ERROR("reading error for potential file.");
  }
  int num_types = get_number_of_types(fid_potential);
  number_of_atoms_ = number_of_atoms;
  bool is_nep = false;
  // determine the potential
  if (strcmp(potential_name, "tersoff_1989") == 0) {
    potential.reset(new Tersoff1989(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "tersoff_1988") == 0) {
    potential.reset(new Tersoff1988(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "tersoff_mini") == 0) {
    potential.reset(new Tersoff_mini(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "eam_zhou_2004") == 0) {
    potential.reset(new EAM(fid_potential, potential_name, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "eam_dai_2006") == 0) {
    potential.reset(new EAM(fid_potential, potential_name, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "eam/alloy") == 0) {
    int max_neigh = 400;
    if (num_param == 3) {
      if (!is_valid_int(param[2], &max_neigh) || max_neigh <= 0 || max_neigh > 1024) {
        PRINT_INPUT_ERROR(
          "max_neighbor for eam/alloy must be a positive integer in (0, 1024].");
      }
    }
    potential.reset(new EAMAlloy(param[1], number_of_atoms, max_neigh));
  } else if (strcmp(potential_name, "adp") == 0) {
    potential.reset(new ADP(param[1], number_of_atoms));
  } else if (strcmp(potential_name, "fcp") == 0) {
    potential.reset(new FCP(fid_potential, num_types, number_of_atoms, box));
    is_fcp = true;
  } else if (
    strcmp(potential_name, "nep4_charge1") == 0 ||
    strcmp(potential_name, "nep4_charge2") == 0 ||
    strcmp(potential_name, "nep4_charge3") == 0 ||
    strcmp(potential_name, "nep4_zbl_charge1") == 0 ||
    strcmp(potential_name, "nep4_zbl_charge2") == 0 ||
    strcmp(potential_name, "nep4_zbl_charge3") == 0) {
    potential.reset(new NEP_Charge(param[1], number_of_atoms));
    is_nep = true;
    primary_nep_model_path_ = param[1];
    check_types(param[1]);
  } else if (
    strcmp(potential_name, "nep5") == 0 || strcmp(potential_name, "nep5_zbl") == 0 ||
    strcmp(potential_name, "nep3") == 0 || strcmp(potential_name, "nep3_zbl") == 0 ||
    strcmp(potential_name, "nep4") == 0 || strcmp(potential_name, "nep4_zbl") == 0 ||
    strcmp(potential_name, "nep3_dipole") == 0 ||
    strcmp(potential_name, "nep3_polarizability") == 0 ||
    strcmp(potential_name, "nep4_dipole") == 0 ||
    strcmp(potential_name, "nep4_polarizability") == 0 ||
    strcmp(potential_name, "nep3_temperature") == 0 ||
    strcmp(potential_name, "nep3_zbl_temperature") == 0 ||
    strcmp(potential_name, "nep4_temperature") == 0 ||
    strcmp(potential_name, "nep4_zbl_temperature") == 0) {
    int num_gpus;
    CHECK(gpuGetDeviceCount(&num_gpus));
#ifdef ZHEYONG
    num_gpus = 3;
#endif
    if (num_gpus == 1) {
      potential.reset(new NEP(param[1], number_of_atoms));
    } else {
      int partition_direction = -1;
      if (num_param == 3) {
        if (strcmp(param[2], "x") == 0) {
          partition_direction = 0;
        } else if (strcmp(param[2], "y") == 0) {
          partition_direction = 1;
        } else if (strcmp(param[2], "z") == 0) {
          partition_direction = 2;
        } else {
          PRINT_INPUT_ERROR("partition direction for multi-GPU NEP can only be x or y or z.\n");
        }
      }
      potential.reset(new NEP_MULTIGPU(num_gpus, param[1], number_of_atoms, partition_direction));
    }
    is_nep = true;
    primary_nep_model_path_ = param[1];
    // Check if the types for this potential are compatible with the possibly other potentials
    check_types(param[1]);
#ifdef USE_DEEPMD
  } else if (strcmp(potential_name, "dp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR(
        "The potential command should contain two parameters, the setting file and the DP potential file.\n");
    }
    potential.reset(new DP(param[2], number_of_atoms));
#endif
#ifdef USE_NNAP
  } else if (strcmp(potential_name, "nnap") == 0 || strcmp(potential_name, "nnap_zbl") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR(
        "The potential command should contain two parameters, the setting file and the NNAP potential file.\n");
    }
    potential.reset(new NNAP(param[1], param[2], number_of_atoms));
#endif
  } else if (strcmp(potential_name, "lj") == 0) {
    potential.reset(new LJ(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "nep_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain an ILP potential file and a NEP map file.\n");
    }
    FILE* fid_nep_map = my_fopen(param[2], "r");
    potential.reset(new ILP_NEP(fid_potential, fid_nep_map, num_types, number_of_atoms));
    fclose(fid_nep_map);
  } else if (strcmp(potential_name, "tersoff_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain ILP potential file and Tersoff potential file.\n");
    }
    FILE* fid_tersoff = my_fopen(param[2], "r");
    potential.reset(new ILP_TERSOFF(fid_potential, fid_tersoff, num_types, number_of_atoms));
    fclose(fid_tersoff);
  } else if (strcmp(potential_name, "sw_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain ILP potential file and SW potential file.\n");
    }
    FILE* fid_sw = my_fopen(param[2], "r");
    potential.reset(new ILP_TMD_SW(fid_potential, fid_sw, num_types, number_of_atoms));
    fclose(fid_sw);
  } else {
    PRINT_INPUT_ERROR("illegal potential model.\n");
  }
  fclose(fid_potential);

  potential->N1 = 0;
  potential->N2 = number_of_atoms;
  potential->set_pppm_mesh_spacing(pppm_mesh_spacing_);
  potential->set_md_qnep_bec(md_qnep_bec_mode_ == 1);

  // Move the pointer into the list of potentials
  potentials.push_back(std::move(potential));
  // Check if a non-NEP potential has previously been defined
  has_non_nep = has_non_nep || !is_nep;
  if (potentials.size() > 1 && has_non_nep) {
    PRINT_INPUT_ERROR("Multiple potentials may only be used with NEP potentials.\n");
  }
  refresh_pimd_bead_gpu_workers_();
}

void Force::reset_pimd_bead_timing()
{
  pimd_bead_timing_ = PIMD_Bead_Timing();
  pimd_bead_timing_.worker_compute.resize(pimd_bead_gpu_workers_.size(), 0.0);
}

int Force::get_number_of_types(FILE* fid_potential)
{
  int num_of_types;
  int count = fscanf(fid_potential, "%d", &num_of_types);
  PRINT_SCANF_ERROR(count, 1, "Reading error for number of types.");
  return num_of_types;
}

void Force::set_pimd_bead_gpu_parallel(const int num_devices)
{
  pimd_bead_gpu_parallel_devices_ = num_devices;
  refresh_pimd_bead_gpu_workers_();
}

void Force::set_pimd_bead_neighbor_rebuild(const bool always_rebuild)
{
  pimd_bead_neighbor_always_rebuild_ = always_rebuild;
  if (
    potentials.size() == 1 && potentials[0] &&
    (pimd_bead_gpu_parallel_devices_ > 1 || pimd_bead_batch_enabled_)) {
    potentials[0]->set_neighbor_rebuild(always_rebuild);
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->set_neighbor_rebuild(always_rebuild);
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->set_neighbor_rebuild(always_rebuild);
  }
}

void Force::set_pppm_mesh_spacing(const double spacing)
{
  pppm_mesh_spacing_ = spacing;
  for (auto& potential : potentials) {
    potential->set_pppm_mesh_spacing(spacing);
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->set_pppm_mesh_spacing(spacing);
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->set_pppm_mesh_spacing(spacing);
  }
}

void Force::apply_md_qnep_bec_setting_()
{
  const bool enabled =
    md_qnep_bec_mode_ == 1 || (md_qnep_bec_mode_ == 0 && md_qnep_bec_required_);
  for (auto& potential : potentials) {
    potential->set_md_qnep_bec(enabled);
  }
}

void Force::set_md_qnep_bec_mode(const int mode)
{
  if (mode < 0 || mode > 2) {
    PRINT_INPUT_ERROR("Invalid classical MD qNEP BEC mode.\n");
  }
  md_qnep_bec_mode_ = mode;
  apply_md_qnep_bec_setting_();
}

void Force::set_md_qnep_bec_required(const bool required)
{
  bool has_qnep = false;
  for (const auto& potential : potentials) {
    if (dynamic_cast<NEP_Charge*>(potential.get())) {
      has_qnep = true;
      break;
    }
  }
  if (required && md_qnep_bec_mode_ == 2 && has_qnep) {
    PRINT_INPUT_ERROR(
      "md_qnep_bec off cannot be used with a BEC-dependent command.\n");
  }
  md_qnep_bec_required_ = required;
  apply_md_qnep_bec_setting_();
  if (has_qnep) {
    const bool enabled =
      md_qnep_bec_mode_ == 1 || (md_qnep_bec_mode_ == 0 && required);
    printf("qNEP classical MD BEC evaluation = %s.\n", enabled ? "on" : "off");
  }
}

void Force::set_pimd_bead_batch(const bool enabled)
{
  pimd_bead_batch_enabled_ = enabled;
  if (potentials.size() == 1 && potentials[0]) {
    const bool is_batch_potential =
      dynamic_cast<NEP_Charge*>(potentials[0].get()) ||
      dynamic_cast<NEP*>(potentials[0].get());
    if (is_batch_potential) {
      potentials[0]->set_neighbor_rebuild(
        enabled ? pimd_bead_neighbor_always_rebuild_ : false);
    }
  }
  apply_pimd_qnep_batch_bec_setting_();
}

bool Force::pimd_qnep_bead_batch_active_() const
{
  return pimd_bead_batch_enabled_ && potentials.size() == 1 && potentials[0] &&
         dynamic_cast<NEP_Charge*>(potentials[0].get());
}

void Force::apply_pimd_qnep_batch_bec_setting_()
{
  const bool enabled = pimd_qnep_batch_bec_enabled_();
  for (auto& potential : potentials) {
    potential->set_pimd_batch_bec(enabled);
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->set_pimd_batch_bec(enabled);
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->set_pimd_batch_bec(enabled);
  }
}

void Force::set_pimd_qnep_batch_bec_mode(const int mode)
{
  if (mode < 0 || mode > 2) {
    PRINT_INPUT_ERROR("Invalid qNEP PIMD batch BEC mode.\n");
  }
  pimd_qnep_batch_bec_mode_ = mode;
  if (mode == 2 && pimd_qnep_batch_bec_required_ && pimd_qnep_bead_batch_active_()) {
    PRINT_INPUT_ERROR(
      "pimd_qnep_batch_bec off cannot be used with a BEC-dependent command.\n");
  }
  apply_pimd_qnep_batch_bec_setting_();
}

void Force::set_pimd_qnep_batch_bec_required(const bool required)
{
  if (required && pimd_qnep_batch_bec_mode_ == 2 && pimd_qnep_bead_batch_active_()) {
    PRINT_INPUT_ERROR(
      "pimd_qnep_batch_bec off cannot be used with a BEC-dependent command.\n");
  }
  pimd_qnep_batch_bec_required_ = required;
  apply_pimd_qnep_batch_bec_setting_();
  if (pimd_qnep_bead_batch_active_()) {
    printf(
      "PIMD qNEP batch BEC evaluation = %s.\n",
      pimd_qnep_batch_bec_enabled_() ? "on" : "off");
  }
}

void Force::set_pimd_nep_batch_profile(const bool enabled)
{
  pimd_nep_batch_profile_enabled_ = enabled;
  for (auto& potential : potentials) {
    potential->set_pimd_batch_profile(enabled);
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->set_pimd_batch_profile(enabled);
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->set_pimd_batch_profile(enabled);
  }
}

void Force::reset_pimd_nep_batch_profile()
{
  for (auto& potential : potentials) {
    potential->reset_pimd_batch_timing();
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->reset_pimd_batch_timing();
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->reset_pimd_batch_timing();
  }
}

void Force::print_pimd_nep_batch_profile() const
{
  if (!pimd_nep_batch_profile_enabled_) {
    return;
  }

  auto print_timing = [](const char* label, const PIMD_Batch_Timing& timing) {
    if (timing.calls == 0) {
      return;
    }
    printf("    %s (%lld force calls):\n", label, timing.calls);
    printf("        setup = %g s.\n", timing.setup);
    printf("        neighbor = %g s.\n", timing.neighbor);
    printf("            global check/rebuild = %g s.\n", timing.neighbor_global);
    printf("                pointer setup = %g s.\n", timing.neighbor_pointer);
    printf("                distance check = %g s.\n", timing.neighbor_check);
    printf("                flag transfer = %g s.\n", timing.neighbor_flags);
    printf(
      "                rebuild/update = %g s (%lld bead rebuilds).\n",
      timing.neighbor_rebuild,
      timing.neighbor_rebuild_beads);
    printf("            compact filter/geometry = %g s.\n", timing.neighbor_filter);
    printf("        initialize = %g s.\n", timing.initialize);
    printf("        descriptor/ANN = %g s.\n", timing.descriptor);
    printf("        BEC = %g s.\n", timing.bec);
    printf("        electrostatics = %g s.\n", timing.electrostatics);
    printf("        radial force = %g s.\n", timing.radial);
    printf("        angular force = %g s.\n", timing.angular);
    printf("        many-body force = %g s.\n", timing.many_body);
    printf("        ZBL/DFTD3 = %g s.\n", timing.corrections);
    printf("        total batch potential = %g s.\n", timing.total);
  };

  printf("PIMD NEP/qNEP batch stage timing:\n");
  for (size_t potential_id = 0; potential_id < potentials.size(); ++potential_id) {
    char label[64];
    snprintf(label, sizeof(label), "GPU 0 potential %zu", potential_id);
    print_timing(label, potentials[potential_id]->get_pimd_batch_timing());
  }
  for (const auto& worker : pimd_bead_gpu_workers_) {
    char label[64];
    snprintf(label, sizeof(label), "GPU %d worker", worker->device_id);
    print_timing(label, worker->potential->get_pimd_batch_timing());
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    print_timing(
      "single-GPU fallback potential",
      pimd_nep_single_gpu_batch_potential_->get_pimd_batch_timing());
  }
}

bool Force::can_use_pimd_bead_gpu_parallel_() const
{
  if (pimd_bead_gpu_parallel_devices_ <= 1 || pimd_bead_gpu_workers_.size() <= 1) {
    return false;
  }
  if (potentials.size() != 1 || multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }
  if (is_fcp || compute_hnemd_ || compute_hnemdec_ != -1 || primary_nep_model_path_.empty()) {
    return false;
  }
  Potential* primary = potentials[0].get();
  return !primary->need_B_projection &&
         (dynamic_cast<NEP*>(primary) || dynamic_cast<NEP_MULTIGPU*>(primary) ||
          dynamic_cast<NEP_Charge*>(primary));
}

bool Force::can_use_pimd_qnep_batch_() const
{
  if (!pimd_bead_batch_enabled_ || potentials.size() != 1 ||
      multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }
  if (is_fcp || compute_hnemd_ || compute_hnemdec_ != -1) {
    return false;
  }
  Potential* primary = potentials[0].get();
  return primary && !primary->need_B_projection && dynamic_cast<NEP_Charge*>(primary);
}

bool Force::can_use_pimd_nep_batch_() const
{
  if (!pimd_bead_batch_enabled_ || potentials.size() != 1 ||
      multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }
  if (is_fcp || compute_hnemd_ || compute_hnemdec_ != -1) {
    return false;
  }
  Potential* primary = potentials[0].get();
  return primary && !primary->need_B_projection &&
         (dynamic_cast<NEP*>(primary) ||
          (dynamic_cast<NEP_MULTIGPU*>(primary) && !primary_nep_model_path_.empty()));
}

bool Force::try_compute_pimd_qnep_batch_(
  Box& box,
  GPU_Vector<int>& type,
  std::vector<GPU_Vector<double>>& position_beads,
  std::vector<GPU_Vector<double>>& potential_beads,
  std::vector<GPU_Vector<double>>& force_beads,
  std::vector<GPU_Vector<double>>& virial_beads)
{
  if (!can_use_pimd_qnep_batch_()) {
    return false;
  }

  CHECK(gpuSetDevice(0));
  box.set_is_orthogonal();
  const int number_of_atoms = type.size();
  const int number_of_beads = int(position_beads.size());
  if (
    number_of_beads < 2 || potential_beads.size() != position_beads.size() ||
    force_beads.size() != position_beads.size() ||
    virial_beads.size() != position_beads.size()) {
    return false;
  }
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_beads[bead_id].data(),
      position_beads[bead_id].data() + number_of_atoms,
      position_beads[bead_id].data() + number_of_atoms * 2);
  }
  GPU_CHECK_KERNEL

  std::vector<GPU_Vector<double>*> positions;
  std::vector<GPU_Vector<double>*> potentials_per_bead;
  std::vector<GPU_Vector<double>*> forces;
  std::vector<GPU_Vector<double>*> virials;
  positions.reserve(number_of_beads);
  potentials_per_bead.reserve(number_of_beads);
  forces.reserve(number_of_beads);
  virials.reserve(number_of_beads);
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    positions.push_back(&position_beads[bead_id]);
    potentials_per_bead.push_back(&potential_beads[bead_id]);
    forces.push_back(&force_beads[bead_id]);
    virials.push_back(&virial_beads[bead_id]);
  }

  NEP_Charge* qnep = dynamic_cast<NEP_Charge*>(potentials[0].get());
  const bool used_batch = qnep->compute_pimd_batch(
    box, type, positions, potentials_per_bead, forces, virials);
  if (used_batch) {
    temperature += number_of_beads * delta_T;
  }
  return used_batch;
}

bool Force::try_compute_pimd_nep_batch_(
  Box& box,
  GPU_Vector<int>& type,
  std::vector<GPU_Vector<double>>& position_beads,
  std::vector<GPU_Vector<double>>& potential_beads,
  std::vector<GPU_Vector<double>>& force_beads,
  std::vector<GPU_Vector<double>>& virial_beads)
{
  if (!can_use_pimd_nep_batch_()) {
    return false;
  }

  CHECK(gpuSetDevice(0));
  box.set_is_orthogonal();
  const int number_of_atoms = type.size();
  const int number_of_beads = int(position_beads.size());
  if (
    number_of_beads < 2 || potential_beads.size() != position_beads.size() ||
    force_beads.size() != position_beads.size() ||
    virial_beads.size() != position_beads.size()) {
    return false;
  }
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_beads[bead_id].data(),
      position_beads[bead_id].data() + number_of_atoms,
      position_beads[bead_id].data() + number_of_atoms * 2);
  }
  GPU_CHECK_KERNEL

  std::vector<GPU_Vector<double>*> positions;
  std::vector<GPU_Vector<double>*> potentials_per_bead;
  std::vector<GPU_Vector<double>*> forces;
  std::vector<GPU_Vector<double>*> virials;
  positions.reserve(number_of_beads);
  potentials_per_bead.reserve(number_of_beads);
  forces.reserve(number_of_beads);
  virials.reserve(number_of_beads);
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    positions.push_back(&position_beads[bead_id]);
    potentials_per_bead.push_back(&potential_beads[bead_id]);
    forces.push_back(&force_beads[bead_id]);
    virials.push_back(&virial_beads[bead_id]);
  }

  NEP* nep = dynamic_cast<NEP*>(potentials[0].get());
  if (!nep) {
    if (!pimd_nep_single_gpu_batch_potential_) {
      pimd_nep_single_gpu_batch_potential_.reset(
        new NEP(primary_nep_model_path_.c_str(), number_of_atoms_));
      pimd_nep_single_gpu_batch_potential_->N1 = 0;
      pimd_nep_single_gpu_batch_potential_->N2 = number_of_atoms_;
      pimd_nep_single_gpu_batch_potential_->set_neighbor_rebuild(
        pimd_bead_neighbor_always_rebuild_);
      pimd_nep_single_gpu_batch_potential_->set_pimd_batch_profile(
        pimd_nep_batch_profile_enabled_);
    }
    nep = dynamic_cast<NEP*>(pimd_nep_single_gpu_batch_potential_.get());
  }
  const bool used_batch = nep->compute_pimd_batch(
    box, type, positions, potentials_per_bead, forces, virials);
  if (used_batch) {
    temperature += number_of_beads * delta_T;
  }
  return used_batch;
}

Potential* Force::get_pimd_bead_potential_(const int device_id) const
{
  if (device_id == 0) {
    return potentials[0].get();
  }
  for (const auto& worker_ptr : pimd_bead_gpu_workers_) {
    if (worker_ptr->device_id == device_id) {
      return worker_ptr->potential.get();
    }
  }
  PRINT_INPUT_ERROR("Cannot find the requested GPU worker for PIMD bead parallel.\n");
  return nullptr;
}

void Force::refresh_pimd_bead_gpu_workers_()
{
  pimd_bead_gpu_workers_.clear();
  if (potentials.size() == 1 && potentials[0]) {
    const bool use_primary_batch_neighbor_setting =
      pimd_bead_batch_enabled_ &&
      (dynamic_cast<NEP_Charge*>(potentials[0].get()) ||
       dynamic_cast<NEP*>(potentials[0].get()));
    potentials[0]->set_neighbor_rebuild(
      use_primary_batch_neighbor_setting ? pimd_bead_neighbor_always_rebuild_ : false);
    potentials[0]->set_pppm_mesh_spacing(pppm_mesh_spacing_);
    potentials[0]->set_pimd_batch_bec(pimd_qnep_batch_bec_enabled_());
    potentials[0]->set_pimd_batch_profile(pimd_nep_batch_profile_enabled_);
  }
  if (pimd_bead_gpu_parallel_devices_ <= 1 || primary_nep_model_path_.empty() ||
      number_of_atoms_ <= 0) {
    return;
  }
  if (potentials.size() != 1 || multiple_potentials_mode_.compare("observe") != 0) {
    return;
  }
  if (is_fcp || compute_hnemd_ || compute_hnemdec_ != -1) {
    return;
  }
  Potential* primary = potentials[0].get();
  const bool is_nep_worker = dynamic_cast<NEP*>(primary) || dynamic_cast<NEP_MULTIGPU*>(primary);
  const bool is_qnep_worker = dynamic_cast<NEP_Charge*>(primary);
  if (!(is_nep_worker || is_qnep_worker)) {
    return;
  }
  if (primary->need_B_projection) {
    return;
  }

  int available_devices = 0;
  CHECK(gpuGetDeviceCount(&available_devices));
  const int num_workers = std::min(pimd_bead_gpu_parallel_devices_, available_devices);
  if (num_workers <= 1) {
    return;
  }

  primary->set_neighbor_rebuild(pimd_bead_neighbor_always_rebuild_);

  for (int device_id = 0; device_id < num_workers; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    std::unique_ptr<Force::PIMD_Bead_GPU_Worker> worker(new Force::PIMD_Bead_GPU_Worker());
    worker->device_id = device_id;
    if (is_qnep_worker) {
      std::unique_ptr<NEP_Charge> qnep_worker(
        new NEP_Charge(primary_nep_model_path_.c_str(), number_of_atoms_));
      qnep_worker->set_neighbor_diagnostics(false);
      worker->potential = std::move(qnep_worker);
    } else {
      worker->potential.reset(new NEP(primary_nep_model_path_.c_str(), number_of_atoms_));
    }
    worker->potential->set_pimd_batch_profile(pimd_nep_batch_profile_enabled_);
    worker->potential->set_pimd_batch_bec(pimd_qnep_batch_bec_enabled_());
    worker->potential->set_neighbor_rebuild(pimd_bead_neighbor_always_rebuild_);
    worker->potential->set_pppm_mesh_spacing(pppm_mesh_spacing_);
    worker->potential->N1 = 0;
    worker->potential->N2 = number_of_atoms_;
    worker->type.resize(number_of_atoms_);
    pimd_bead_gpu_workers_.push_back(std::move(worker));
  }
  apply_pimd_qnep_batch_bec_setting_();
  CHECK(gpuSetDevice(0));
}

static __global__ void gpu_add_driving_force(
  int N,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] += fe_x * g_sxx[i] + fe_y * g_syx[i] + fe_z * g_szx[i];
    g_fy[i] += fe_x * g_sxy[i] + fe_y * g_syy[i] + fe_z * g_szy[i];
    g_fz[i] += fe_x * g_sxz[i] + fe_y * g_syz[i] + fe_z * g_szz[i];
  }
}

void Force::compute_pimd_beads(
  Box& box,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  std::vector<GPU_Vector<double>>& position_beads,
  std::vector<GPU_Vector<double>>& potential_beads,
  std::vector<GPU_Vector<double>>& force_beads,
  std::vector<GPU_Vector<double>>& virial_beads,
  std::vector<GPU_Vector<double>>& velocity_beads,
  GPU_Vector<double>& mass_per_atom)
{
  if (!can_use_pimd_bead_gpu_parallel_()) {
    if (
      try_compute_pimd_qnep_batch_(
        box, type, position_beads, potential_beads, force_beads, virial_beads) ||
      try_compute_pimd_nep_batch_(
        box, type, position_beads, potential_beads, force_beads, virial_beads)) {
      return;
    }
    static bool warned_once = false;
    if (pimd_bead_gpu_parallel_devices_ > 1 && !warned_once) {
      printf("Warning: falling back to serial ring-polymer bead force evaluation.\n");
      printf(
        "    bead-to-GPU mode currently requires a single-potential NEP/qNEP run without HNEMD/FCP.\n");
      warned_once = true;
    }
    static bool warned_batch_once = false;
    if (pimd_bead_batch_enabled_ && !warned_batch_once) {
      const bool is_qnep = potentials.size() == 1 && potentials[0] &&
                           dynamic_cast<NEP_Charge*>(potentials[0].get());
      const bool is_nep = potentials.size() == 1 && potentials[0] &&
                          (dynamic_cast<NEP*>(potentials[0].get()) ||
                           dynamic_cast<NEP_MULTIGPU*>(potentials[0].get()));
      if (is_qnep) {
        printf("Warning: qNEP ring-polymer bead batching is unavailable; using serial bead forces.\n");
        printf("    batching requires at least two beads, qNEP, and the large-box path.\n");
      } else if (is_nep) {
        printf("Warning: NEP ring-polymer bead batching is unavailable; using serial bead forces.\n");
        printf("    batching requires at least two beads, a standard NEP energy model, and the large-box path.\n");
      } else {
        printf("Warning: ring-polymer bead batching is unavailable; using serial bead forces.\n");
      }
      warned_batch_once = true;
    }
    for (int k = 0; k < position_beads.size(); ++k) {
      compute(
        box,
        position_beads[k],
        type,
        group,
        potential_beads[k],
        force_beads[k],
        virial_beads[k],
        velocity_beads[k],
        mass_per_atom);
    }
    return;
  }

  (void)group;
  (void)velocity_beads;
  (void)mass_per_atom;

  box.set_is_orthogonal();
  const int number_of_atoms = type.size();
  const int number_of_beads = int(position_beads.size());
  const int number_of_workers = int(pimd_bead_gpu_workers_.size());
  const double initial_temperature = temperature;
  using Clock = std::chrono::high_resolution_clock;
  const auto total_begin = Clock::now();

  // Keep the authoritative coordinates on GPU 0 wrapped before staging remote beads.
  const auto wrap_begin = Clock::now();
  CHECK(gpuSetDevice(0));
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_beads[bead_id].data(),
      position_beads[bead_id].data() + number_of_atoms,
      position_beads[bead_id].data() + number_of_atoms * 2);
  }
  GPU_CHECK_KERNEL
  CHECK(gpuDeviceSynchronize());
  pimd_bead_timing_.wrap_positions +=
    std::chrono::duration<double>(Clock::now() - wrap_begin).count();

  // Stage all remote coordinates first. Each remote bead then has dedicated buffers,
  // so force kernels can be queued without synchronizing and reusing one scratch output.
  const auto stage_begin = Clock::now();
  for (int worker_id = 1; worker_id < number_of_workers; ++worker_id) {
    const int bead_begin = worker_id * number_of_beads / number_of_workers;
    const int bead_end = (worker_id + 1) * number_of_beads / number_of_workers;
    const int local_beads = bead_end - bead_begin;
    auto& worker = *pimd_bead_gpu_workers_[worker_id];
    CHECK(gpuSetDevice(worker.device_id));
    ensure_pimd_worker_bead_buffers(worker, local_beads, number_of_atoms);
    copy_gpu_buffer_between_devices(
      worker.device_id, worker.type.data(), 0, type.data(), number_of_atoms);
    for (int local_bead = 0; local_bead < local_beads; ++local_bead) {
      const int bead_id = bead_begin + local_bead;
      copy_gpu_buffer_between_devices(
        worker.device_id,
        worker.position_beads[local_bead].data(),
        0,
        position_beads[bead_id].data(),
        size_t(number_of_atoms) * 3);
    }
  }
  pimd_bead_timing_.stage_remote +=
    std::chrono::duration<double>(Clock::now() - stage_begin).count();

  const auto compute_begin = Clock::now();
  std::vector<double> worker_compute(number_of_workers, 0.0);
  std::vector<std::thread> workers;
  workers.reserve(number_of_workers);
  for (int worker_id = 0; worker_id < number_of_workers; ++worker_id) {
    workers.emplace_back([&, worker_id]() {
      const auto worker_begin = Clock::now();
      const int bead_begin = worker_id * number_of_beads / number_of_workers;
      const int bead_end = (worker_id + 1) * number_of_beads / number_of_workers;
      auto& worker = *pimd_bead_gpu_workers_[worker_id];
      const int device_id = worker.device_id;
      CHECK(gpuSetDevice(device_id));
      Box worker_box = box;

      bool used_pimd_batch = false;
      if (pimd_bead_batch_enabled_) {
        NEP_Charge* qnep = dynamic_cast<NEP_Charge*>(worker.potential.get());
        NEP* nep = dynamic_cast<NEP*>(worker.potential.get());
        if (
          qnep || nep) {
          std::vector<GPU_Vector<double>*> worker_positions;
          std::vector<GPU_Vector<double>*> worker_potentials;
          std::vector<GPU_Vector<double>*> worker_forces;
          std::vector<GPU_Vector<double>*> worker_virials;
          worker_positions.reserve(bead_end - bead_begin);
          worker_potentials.reserve(bead_end - bead_begin);
          worker_forces.reserve(bead_end - bead_begin);
          worker_virials.reserve(bead_end - bead_begin);
          for (int bead_id = bead_begin; bead_id < bead_end; ++bead_id) {
            const int local_bead = bead_id - bead_begin;
            worker_positions.push_back(
              device_id == 0 ? &position_beads[bead_id] : &worker.position_beads[local_bead]);
            worker_potentials.push_back(
              device_id == 0 ? &potential_beads[bead_id] : &worker.potential_beads[local_bead]);
            worker_forces.push_back(
              device_id == 0 ? &force_beads[bead_id] : &worker.force_beads[local_bead]);
            worker_virials.push_back(
              device_id == 0 ? &virial_beads[bead_id] : &worker.virial_beads[local_bead]);
          }
          if (qnep) {
            used_pimd_batch = qnep->compute_pimd_batch(
              worker_box,
              device_id == 0 ? type : worker.type,
              worker_positions,
              worker_potentials,
              worker_forces,
              worker_virials);
          } else {
            used_pimd_batch = nep->compute_pimd_batch(
              worker_box,
              device_id == 0 ? type : worker.type,
              worker_positions,
              worker_potentials,
              worker_forces,
              worker_virials);
          }
        }
      }

      for (int bead_id = bead_begin; !used_pimd_batch && bead_id < bead_end; ++bead_id) {
        const int local_bead = bead_id - bead_begin;
        GPU_Vector<int>& worker_type = device_id == 0 ? type : worker.type;
        GPU_Vector<double>& worker_position =
          device_id == 0 ? position_beads[bead_id] : worker.position_beads[local_bead];
        GPU_Vector<double>& worker_potential =
          device_id == 0 ? potential_beads[bead_id] : worker.potential_beads[local_bead];
        GPU_Vector<double>& worker_force =
          device_id == 0 ? force_beads[bead_id] : worker.force_beads[local_bead];
        GPU_Vector<double>& worker_virial =
          device_id == 0 ? virial_beads[bead_id] : worker.virial_beads[local_bead];

        initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
          number_of_atoms,
          worker_force.data(),
          worker_force.data() + number_of_atoms,
          worker_force.data() + number_of_atoms * 2,
          worker_potential.data(),
          worker_virial.data());
        GPU_CHECK_KERNEL

        const double bead_temperature = initial_temperature + (bead_id + 1) * delta_T;
        if (3 == worker.potential->nep_model_type) {
          worker.potential->compute(
            bead_temperature,
            worker_box,
            worker_type,
            worker_position,
            worker_potential,
            worker_force,
            worker_virial);
        } else {
          worker.potential->compute(
            worker_box,
            worker_type,
            worker_position,
            worker_potential,
            worker_force,
            worker_virial);
        }
      }
      CHECK(gpuDeviceSynchronize());
      worker_compute[worker_id] =
        std::chrono::duration<double>(Clock::now() - worker_begin).count();
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  pimd_bead_timing_.compute_workers +=
    std::chrono::duration<double>(Clock::now() - compute_begin).count();
  if (pimd_bead_timing_.worker_compute.size() != size_t(number_of_workers)) {
    pimd_bead_timing_.worker_compute.assign(number_of_workers, 0.0);
  }
  for (int worker_id = 0; worker_id < number_of_workers; ++worker_id) {
    pimd_bead_timing_.worker_compute[worker_id] += worker_compute[worker_id];
  }

  // PIMD integration and restart state stay authoritative on GPU 0. Coordinates
  // were wrapped before staging, so only force-related outputs need to return.
  const auto gather_begin = Clock::now();
  for (int worker_id = 1; worker_id < number_of_workers; ++worker_id) {
    const int bead_begin = worker_id * number_of_beads / number_of_workers;
    const int bead_end = (worker_id + 1) * number_of_beads / number_of_workers;
    auto& worker = *pimd_bead_gpu_workers_[worker_id];
    CHECK(gpuSetDevice(worker.device_id));
    for (int bead_id = bead_begin; bead_id < bead_end; ++bead_id) {
      const int local_bead = bead_id - bead_begin;
      copy_gpu_buffer_between_devices(
        0,
        potential_beads[bead_id].data(),
        worker.device_id,
        worker.potential_beads[local_bead].data(),
        number_of_atoms);
      copy_gpu_buffer_between_devices(
        0,
        force_beads[bead_id].data(),
        worker.device_id,
        worker.force_beads[local_bead].data(),
        size_t(number_of_atoms) * 3);
      copy_gpu_buffer_between_devices(
        0,
        virial_beads[bead_id].data(),
        worker.device_id,
        worker.virial_beads[local_bead].data(),
        size_t(number_of_atoms) * 9);
    }
  }
  pimd_bead_timing_.gather_remote +=
    std::chrono::duration<double>(Clock::now() - gather_begin).count();

  temperature = initial_temperature + number_of_beads * delta_T;
  CHECK(gpuSetDevice(0));
  pimd_bead_timing_.total +=
    std::chrono::duration<double>(Clock::now() - total_begin).count();
  ++pimd_bead_timing_.calls;
}

void Force::compute_pimd_bead_range_on_device(
  const int device_id,
  Box& box,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  std::vector<GPU_Vector<double>>& position_beads,
  std::vector<GPU_Vector<double>>& potential_beads,
  std::vector<GPU_Vector<double>>& force_beads,
  std::vector<GPU_Vector<double>>& virial_beads,
  std::vector<GPU_Vector<double>>& velocity_beads,
  GPU_Vector<double>& mass_per_atom,
  const int bead_begin,
  const int bead_end,
  const double initial_temperature)
{
  (void)group;
  (void)velocity_beads;
  (void)mass_per_atom;

  if (bead_begin >= bead_end) {
    return;
  }

  CHECK(gpuSetDevice(device_id));
  box.set_is_orthogonal();

  Potential* potential = get_pimd_bead_potential_(device_id);
  const int number_of_atoms = type.size();
  for (int bead_id = bead_begin; bead_id < bead_end; ++bead_id) {
    if (!is_fcp) {
      gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms,
        box,
        position_beads[bead_id].data(),
        position_beads[bead_id].data() + number_of_atoms,
        position_beads[bead_id].data() + number_of_atoms * 2);
    }

    initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      force_beads[bead_id].data(),
      force_beads[bead_id].data() + number_of_atoms,
      force_beads[bead_id].data() + number_of_atoms * 2,
      potential_beads[bead_id].data(),
      virial_beads[bead_id].data());
    GPU_CHECK_KERNEL

    const double bead_temperature = initial_temperature + (bead_id + 1) * delta_T;
    if (3 == potential->nep_model_type) {
      potential->compute(
        bead_temperature,
        box,
        type,
        position_beads[bead_id],
        potential_beads[bead_id],
        force_beads[bead_id],
        virial_beads[bead_id]);
    } else {
      potential->compute(
        box,
        type,
        position_beads[bead_id],
        potential_beads[bead_id],
        force_beads[bead_id],
        virial_beads[bead_id]);
    }
  }
}

// get the total force
static __global__ void gpu_sum_force(int N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_f[1024];
  double f = 0.0;

  switch (bid) {
    case 0:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fx[n];
      }
      break;
    case 1:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fy[n];
      }
      break;
    case 2:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fz[n];
      }
      break;
  }
  s_f[tid] = f;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_f[bid] = s_f[0];
  }
}

// correct the total force
static __global__ void
gpu_correct_force(int N, double one_over_N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] -= g_f[0] * one_over_N;
    g_fy[i] -= g_f[1] * one_over_N;
    g_fz[i] -= g_f[2] * one_over_N;
  }
}

static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    g_virial[n1 + 0 * N] = 0.0;
    g_virial[n1 + 1 * N] = 0.0;
    g_virial[n1 + 2 * N] = 0.0;
    g_virial[n1 + 3 * N] = 0.0;
    g_virial[n1 + 4 * N] = 0.0;
    g_virial[n1 + 5 * N] = 0.0;
    g_virial[n1 + 6 * N] = 0.0;
    g_virial[n1 + 7 * N] = 0.0;
    g_virial[n1 + 8 * N] = 0.0;
  }
}

void Force::finalize()
{
  compute_hnemd_ = false;
  compute_hnemdec_ = -1;
  refresh_pimd_bead_gpu_workers_();
}

void Force::set_hnemd_parameters(
  const double hnemd_fe_x, const double hnemd_fe_y, const double hnemd_fe_z)
{
  if (compute_hnemd_ || compute_hnemdec_ >= 0) {
    PRINT_INPUT_ERROR("Cannot have more than one HNEMD method within one run.");
  }
  compute_hnemd_ = true;
  hnemd_fe_[0] = hnemd_fe_x;
  hnemd_fe_[1] = hnemd_fe_y;
  hnemd_fe_[2] = hnemd_fe_z;
  refresh_pimd_bead_gpu_workers_();
}

void Force::set_hnemdec_parameters(
  const int compute_hnemdec,
  const double hnemd_fe_x,
  const double hnemd_fe_y,
  const double hnemd_fe_z,
  const std::vector<double>& mass,
  const std::vector<int>& type,
  const std::vector<int>& type_size,
  const double T)
{
  if (compute_hnemd_ || compute_hnemdec_ >= 0) {
    PRINT_INPUT_ERROR("Cannot have more than one HNEMD method within one run.");
  }

  int N = mass.size();
  int number_of_types = type_size.size();
  compute_hnemdec_ = compute_hnemdec;
  temperature = T;

  double total_mass = 0;
  std::vector<double> cpu_coefficient;
  std::vector<double> mass_type;
  mass_type.resize(number_of_types);
  int find_mass_type = 0;
  for (int i = 0; i < N; i++) {
    if (mass_type[type[i]] != mass[i]) {
      mass_type[type[i]] = mass[i];
      find_mass_type += 1;
    }
    total_mass += mass[i];
  }
  if (find_mass_type != number_of_types) {
    PRINT_INPUT_ERROR("mass type and element type do not match.\n");
  }

  // find atom types' fraction
  if (compute_hnemdec_ == 0) {
    cpu_coefficient.resize(number_of_types * 2);
    coefficient.resize(number_of_types * 2);

    for (int i = 0; i < number_of_types; i++) {
      double c_hv = (total_mass - N * mass_type[i]) / total_mass;
      cpu_coefficient[i * 2] = (c_hv - 1) / N;
      cpu_coefficient[i * 2 + 1] = K_B * temperature * c_hv;
    }

    coefficient.copy_from_host(cpu_coefficient.data());
  } else if ((compute_hnemdec_ > 0) && (compute_hnemdec_ <= number_of_types)) {
    int element_index = compute_hnemdec_ - 1;
    cpu_coefficient.resize(number_of_types);
    cpu_coefficient[element_index] = double(N) / type_size[element_index];
    double patial_mass = 0;
    for (int i = 0; i < number_of_types; i++) {
      if (i != element_index) {
        patial_mass += mass_type[i] * type_size[i];
      }
    }
    for (int i = 0; i < number_of_types; i++) {
      if (i != element_index) {
        cpu_coefficient[i] = -1 * N * mass_type[i] / patial_mass;
      }
    }
    coefficient.resize(number_of_types);
    coefficient.copy_from_host(cpu_coefficient.data());
  }

  hnemd_fe_[0] = hnemd_fe_x;
  hnemd_fe_[1] = hnemd_fe_y;
  hnemd_fe_[2] = hnemd_fe_z;
  refresh_pimd_bead_gpu_workers_();
}

static __global__ void gpu_apply_pbc(
  int N, Box box, double* g_x, double* g_y, double* g_z, int* g_position_image)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    if (box.pbc_x == 1) {
      if (sx < 0.0) {
        sx += 1.0;
        if (g_position_image != nullptr)
          g_position_image[n]--;
      } else if (sx > 1.0) {
        sx -= 1.0;
        if (g_position_image != nullptr)
          g_position_image[n]++;
      }
    }
    if (box.pbc_y == 1) {
      if (sy < 0.0) {
        sy += 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N]--;
      } else if (sy > 1.0) {
        sy -= 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N]++;
      }
    }
    if (box.pbc_z == 1) {
      if (sz < 0.0) {
        sz += 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N * 2]--;
      } else if (sz > 1.0) {
        sz -= 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N * 2]++;
      }
    }
    g_x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
    g_y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
    g_z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
  }
}

static __global__ void gpu_average_properties(
  int N, double* g_potential, double* g_force, double* g_virial, double denominator)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_potential[n1] /= denominator;
    g_force[n1 + 0 * N] /= denominator;
    g_force[n1 + 1 * N] /= denominator;
    g_force[n1 + 2 * N] /= denominator;
    g_virial[n1 + 0 * N] /= denominator;
    g_virial[n1 + 1 * N] /= denominator;
    g_virial[n1 + 2 * N] /= denominator;
    g_virial[n1 + 3 * N] /= denominator;
    g_virial[n1 + 4 * N] /= denominator;
    g_virial[n1 + 5 * N] /= denominator;
    g_virial[n1 + 6 * N] /= denominator;
    g_virial[n1 + 7 * N] /= denominator;
    g_virial[n1 + 8 * N] /= denominator;
  }
}

void Force::set_multiple_potentials_mode(std::string mode)
{
  multiple_potentials_mode_ = mode;
  refresh_pimd_bead_gpu_workers_();
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  box.set_is_orthogonal();
  
  const int number_of_atoms = type.size();
  if (!is_fcp) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_per_atom.data(),
      position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2,
      nullptr);
  }

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (multiple_potentials_mode_.compare("observe") == 0) {
    // If observing, calculate using main potential only
    if (3 == potentials[0]->nep_model_type) {
      potentials[0]->compute(
        temperature,
        box,
        type,
        position_per_atom,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);
    } else if (1 == potentials[0]->ilp_flag) {
      // compute the potential with ILP
      potentials[0]->compute_ilp(
        box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, group);
    } else {
      potentials[0]->compute(
        box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
    }
  } else if (multiple_potentials_mode_.compare("average") == 0) {
    // Calculate average potential, force and virial per atom.
    for (int i = 0; i < potentials.size(); i++) {
      // potential->compute automatically adds the properties
      if (3 == potentials[i]->nep_model_type) {
        potentials[i]->compute(
          temperature,
          box,
          type,
          position_per_atom,
          potential_per_atom,
          force_per_atom,
          virial_per_atom);
      } else if (1 == potentials[i]->ilp_flag) {
        // compute the potential with ILP
        potentials[i]->compute_ilp(
          box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, group);
      } else {
        potentials[i]->compute(
          box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
      }
    }
    // Compute average and copy properties back into original vectors.
    gpu_average_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      potential_per_atom.data(),
      force_per_atom.data(),
      virial_per_atom.data(),
      (double)potentials.size());
    GPU_CHECK_KERNEL
  } else {
    PRINT_INPUT_ERROR("Invalid mode for multiple potentials.\n");
  }

  if (compute_hnemd_) {
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      hnemd_fe_[0],
      hnemd_fe_[1],
      hnemd_fe_[2],
      virial_per_atom.data() + 0 * number_of_atoms,
      virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms,
      virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms,
      virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms,
      virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);

    GPU_Vector<double> ftot(3); // total force vector of the system

    gpu_sum_force<<<3, 1024>>>(
      number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    GPU_CHECK_KERNEL

    gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      1.0 / number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    GPU_CHECK_KERNEL
  }

  // always correct the force when using the FCP potential
  if (is_fcp) {
    if (!compute_hnemd_) {
      GPU_Vector<double> ftot(3); // total force vector of the system
      gpu_sum_force<<<3, 1024>>>(
        number_of_atoms,
        force_per_atom.data(),
        force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms,
        ftot.data());
      GPU_CHECK_KERNEL

      gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms,
        1.0 / number_of_atoms,
        force_per_atom.data(),
        force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms,
        ftot.data());
      GPU_CHECK_KERNEL
    }
  }
}

bool Force::compute_qnep_non_electro(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  (void)group;
  if (potentials.size() != 1 || multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }

  auto* qnep = dynamic_cast<NEP_Charge*>(potentials[0].get());
  if (qnep == nullptr) {
    return false;
  }

  box.set_is_orthogonal();
  const int number_of_atoms = type.size();
  if (!is_fcp) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_per_atom.data(),
      position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2);
  }

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  qnep->compute_non_electro(
    box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  return true;
}

static __global__ void gpu_find_per_atom_tensor(
  int N,
  double* g_mass,
  double* g_potential,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_tensor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double mass = g_mass[i];
    double potential = g_potential[i];
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    double energy = mass * (vx * vx + vy * vy + vz * vz) * 0.5 + potential;
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_tensor[i] = energy + g_sxx[i];
    g_tensor[i + 3 * N] = g_sxy[i];
    g_tensor[i + 4 * N] = g_sxz[i];
    g_tensor[i + 6 * N] = g_syx[i];
    g_tensor[i + N] = energy + g_syy[i];
    g_tensor[i + 5 * N] = g_syz[i];
    g_tensor[i + 7 * N] = g_szx[i];
    g_tensor[i + 8 * N] = g_szy[i];
    g_tensor[i + 2 * N] = energy + g_szz[i];
  }
}

static __global__ void gpu_sum_tensor(int N, double* g_tensor, double* g_sum_tensor)
{
  //<<<9,1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_t[1024];
  double t = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 1024;
    if (n < N)
      t += g_tensor[bid * N + n];
  }
  s_t[tid] = t;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_t[tid] += s_t[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_sum_tensor[bid] = s_t[0];
  }
}

static __global__ void gpu_add_driving_force(
  int N,
  const double* g_coefficient,
  const int* g_type,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_tensor_tot,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  // heat flow algorithm
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int type2 = g_type[i] * 2;
    double coefficient1 = g_coefficient[type2];
    double coefficient2 = g_coefficient[type2 + 1];
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_fx[i] += fe_x * (g_sxx[i] + coefficient1 * g_tensor_tot[0] + coefficient2) +
               fe_y * (g_syx[i] + coefficient1 * g_tensor_tot[6]) +
               fe_z * (g_szx[i] + coefficient1 * g_tensor_tot[7]);

    g_fy[i] += fe_x * (g_sxy[i] + coefficient1 * g_tensor_tot[3]) +
               fe_y * (g_syy[i] + coefficient1 * g_tensor_tot[1] + coefficient2) +
               fe_z * (g_szy[i] + coefficient1 * g_tensor_tot[8]);

    g_fz[i] += fe_x * (g_sxz[i] + coefficient1 * g_tensor_tot[4]) +
               fe_y * (g_syz[i] + coefficient1 * g_tensor_tot[5]) +
               fe_z * (g_szz[i] + coefficient1 * g_tensor_tot[2] + coefficient2);
  }
}

static __global__ void gpu_add_driving_force(
  int N,
  const double* g_coefficient,
  const int* g_type,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  // color conductivity algorithm
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double coefficient = g_coefficient[g_type[i]];
    g_fx[i] += fe_x * coefficient;
    g_fy[i] += fe_y * coefficient;
    g_fz[i] += fe_z * coefficient;
  }
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& mass_per_atom,
  int* position_image)
{
  box.set_is_orthogonal();

  const int number_of_atoms = type.size();
  if (!is_fcp) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_per_atom.data(),
      position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2,
      position_image);
  }

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (multiple_potentials_mode_.compare("observe") == 0) {
    // If observing, calculate using main potential only
    if (3 == potentials[0]->nep_model_type) {
      potentials[0]->compute(
        temperature,
        box,
        type,
        position_per_atom,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);
    } else if (1 == potentials[0]->ilp_flag) {
      // compute the potential with ILP
      potentials[0]->compute_ilp(
        box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, group);
    } else {
      potentials[0]->compute(
        box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
    }
  } else if (multiple_potentials_mode_.compare("average") == 0) {
    // Calculate average potential, force and virial per atom.
    for (int i = 0; i < potentials.size(); i++) {
      // potential->compute automatically adds the properties
      if (3 == potentials[i]->nep_model_type) {
        potentials[i]->compute(
          temperature,
          box,
          type,
          position_per_atom,
          potential_per_atom,
          force_per_atom,
          virial_per_atom);
      } else if (1 == potentials[i]->ilp_flag) {
        // compute the potential with ILP
        potentials[i]->compute_ilp(
          box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, group);
      } else {
        potentials[i]->compute(
          box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
      }
    }
    // Compute average and copy properties back into original vectors.
    gpu_average_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      potential_per_atom.data(),
      force_per_atom.data(),
      virial_per_atom.data(),
      (double)potentials.size());
    GPU_CHECK_KERNEL
  } else {
    PRINT_INPUT_ERROR("Invalid mode for multiple potentials.\n");
  }

  if (compute_hnemd_) {
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      hnemd_fe_[0],
      hnemd_fe_[1],
      hnemd_fe_[2],
      virial_per_atom.data() + 0 * number_of_atoms,
      virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms,
      virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms,
      virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms,
      virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);

    GPU_Vector<double> ftot(3); // total force vector of the system

    gpu_sum_force<<<3, 1024>>>(
      number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    GPU_CHECK_KERNEL

    gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      1.0 / number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    GPU_CHECK_KERNEL
  } else if (compute_hnemdec_ == 0) {
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    GPU_Vector<double> tensor_per_atom(number_of_atoms * 9);
    GPU_Vector<double> tensor_tot(9);

    gpu_find_per_atom_tensor<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      mass_per_atom.data(),
      potential_per_atom.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms,
      velocity_per_atom.data() + 2 * number_of_atoms,
      virial_per_atom.data() + 0 * number_of_atoms,
      virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms,
      virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms,
      virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms,
      virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms,
      tensor_per_atom.data());
    GPU_CHECK_KERNEL

    gpu_sum_tensor<<<9, 1024>>>(number_of_atoms, tensor_per_atom.data(), tensor_tot.data());
    GPU_CHECK_KERNEL

    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      coefficient.data(),
      type.data(),
      hnemd_fe_[0],
      hnemd_fe_[1],
      hnemd_fe_[2],
      tensor_per_atom.data() + 0 * number_of_atoms,
      tensor_per_atom.data() + 3 * number_of_atoms,
      tensor_per_atom.data() + 4 * number_of_atoms,
      tensor_per_atom.data() + 6 * number_of_atoms,
      tensor_per_atom.data() + 1 * number_of_atoms,
      tensor_per_atom.data() + 5 * number_of_atoms,
      tensor_per_atom.data() + 7 * number_of_atoms,
      tensor_per_atom.data() + 8 * number_of_atoms,
      tensor_per_atom.data() + 2 * number_of_atoms,
      tensor_tot.data(),
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);
    GPU_CHECK_KERNEL

  } else if (compute_hnemdec_ != -1) {
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      coefficient.data(),
      type.data(),
      hnemd_fe_[0],
      hnemd_fe_[1],
      hnemd_fe_[2],
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);
  }

  // always correct the force when using the FCP potential
  if (is_fcp) {
    if (!compute_hnemd_) {
      GPU_Vector<double> ftot(3); // total force vector of the system
      gpu_sum_force<<<3, 1024>>>(
        number_of_atoms,
        force_per_atom.data(),
        force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms,
        ftot.data());
      GPU_CHECK_KERNEL

      gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms,
        1.0 / number_of_atoms,
        force_per_atom.data(),
        force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms,
        ftot.data());
      GPU_CHECK_KERNEL
    }
  }
}
