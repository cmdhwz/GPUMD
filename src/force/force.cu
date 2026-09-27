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
#ifdef GPUMD_WPE_ENABLED
#include "wpe_adapter.cuh"
#endif
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
#include "utilities/run_input.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

static __global__ void initialize_properties(
  int number_of_atoms,
  double* force_x,
  double* force_y,
  double* force_z,
  double* potential,
  double* virial);

static __global__ void gpu_apply_pbc(
  int N, Box box, double* g_x, double* g_y, double* g_z, int* g_position_image);

static __global__ void gpu_compute_pimd_centroid_position(
  const int N,
  const int number_of_beads,
  const Box box,
  double* const* g_position_beads,
  double* g_centroid_position);

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

void Force::check_types(const std::string& file_potential)
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

std::unique_ptr<Potential> Force::create_potential(
  const std::vector<std::string>& tokens,
  FILE* fid_potential,
  char* potential_name,
  const int num_types,
  const Box& box,
  const int number_of_atoms,
  const RunInput& run_input,
  bool& is_nep)
{
  const int num_param = tokens.size();
  std::unique_ptr<Potential> potential;

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
      if (!is_valid_int(tokens[2], &max_neigh) || max_neigh <= 0 || max_neigh > 1024) {
        PRINT_INPUT_ERROR(
          "max_neighbor for eam/alloy must be a positive integer in (0, 1024].");
      }
    }
    potential.reset(new EAMAlloy(tokens[1].c_str(), number_of_atoms, max_neigh));
  } else if (strcmp(potential_name, "adp") == 0) {
    potential.reset(new ADP(tokens[1].c_str(), number_of_atoms));
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
    potential.reset(new NEP_Charge(tokens[1].c_str(), number_of_atoms, run_input));
    is_nep = true;
    primary_nep_model_path_ = tokens[1];
  } else if (
    strcmp(potential_name, "nep4") == 0 || strcmp(potential_name, "nep4_zbl") == 0 ||
    strcmp(potential_name, "nep4_temperature") == 0 ||
    strcmp(potential_name, "nep4_zbl_temperature") == 0) {
    int num_gpus;
    CHECK(gpuGetDeviceCount(&num_gpus));
#ifdef ZHEYONG
    num_gpus = 3;
#endif
    if (num_gpus == 1) {
      potential.reset(new NEP(tokens[1].c_str(), number_of_atoms, run_input));
    } else {
      int partition_direction = -1;
      if (num_param == 3) {
        if (tokens[2] == "x") {
          partition_direction = 0;
        } else if (tokens[2] == "y") {
          partition_direction = 1;
        } else if (tokens[2] == "z") {
          partition_direction = 2;
        } else {
          PRINT_INPUT_ERROR("partition direction for multi-GPU NEP can only be x or y or z.\n");
        }
      }
      potential.reset(
        new NEP_MULTIGPU(
          num_gpus, tokens[1].c_str(), number_of_atoms, partition_direction, run_input));
    }
    is_nep = true;
    primary_nep_model_path_ = tokens[1];
#ifdef USE_DEEPMD
  } else if (strcmp(potential_name, "dp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR(
        "The potential command should contain two parameters, the setting file and the DP potential file.\n");
    }
    potential.reset(new DP(tokens[2].c_str(), number_of_atoms));
#endif
#ifdef USE_NNAP
  } else if (strcmp(potential_name, "nnap") == 0 || strcmp(potential_name, "nnap_zbl") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR(
        "The potential command should contain two parameters, the setting file and the NNAP potential file.\n");
    }
    potential.reset(new NNAP(tokens[1].c_str(), tokens[2].c_str(), number_of_atoms));
#endif
  } else if (strcmp(potential_name, "lj") == 0) {
    potential.reset(new LJ(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "nep_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain an ILP potential file and a NEP map file.\n");
    }
    FILE* fid_nep_map = my_fopen(tokens[2].c_str(), "r");
    potential.reset(new ILP_NEP(fid_potential, fid_nep_map, num_types, number_of_atoms));
    fclose(fid_nep_map);
  } else if (strcmp(potential_name, "tersoff_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain ILP potential file and Tersoff potential file.\n");
    }
    FILE* fid_tersoff = my_fopen(tokens[2].c_str(), "r");
    potential.reset(new ILP_TERSOFF(fid_potential, fid_tersoff, num_types, number_of_atoms));
    fclose(fid_tersoff);
  } else if (strcmp(potential_name, "sw_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain ILP potential file and SW potential file.\n");
    }
    FILE* fid_sw = my_fopen(tokens[2].c_str(), "r");
    potential.reset(new ILP_TMD_SW(fid_potential, fid_sw, num_types, number_of_atoms));
    fclose(fid_sw);
  } else {
    PRINT_INPUT_ERROR("illegal potential model.\n");
  }

  return potential;
}

void Force::parse_potential(
  const std::vector<std::string>& tokens,
  const Box& box,
  const int number_of_atoms,
  const RunInput& run_input)
{
#ifdef GPUMD_WPE_ENABLED
  if (tokens.size() != 2u) {
    PRINT_INPUT_ERROR(
      "Wisevolve Potential Engine integration supports exactly one potential file and no additional potential parameters.\n");
  }
  if (!potentials.empty()) {
    PRINT_INPUT_ERROR(
      "Wisevolve Potential Engine integration supports exactly one potential. Use standard GPUMD for multiple potentials.\n");
  }
  (void)box;
  (void)run_input;
  std::unique_ptr<Potential> wpe_potential(new WpePotential(
    tokens[1].c_str(), number_of_atoms));
  wpe_potential->N1 = 0;
  wpe_potential->N2 = number_of_atoms;
  potentials.push_back(std::move(wpe_potential));
  return;
#else
  number_of_atoms_ = number_of_atoms;
  run_input_ = &run_input;
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("potential should have 1 or 2 parameters.\n");
  }

  FILE* fid_potential = my_fopen(tokens[1].c_str(), "r");
  char potential_name[100];
  int count = fscanf(fid_potential, "%s", potential_name);
  if (count != 1) {
    PRINT_INPUT_ERROR("reading error for potential file.");
  }
  int num_types = get_number_of_types(fid_potential);
  bool is_nep = false;
  std::unique_ptr<Potential> potential = create_potential(
    tokens,
    fid_potential,
    potential_name,
    num_types,
    box,
    number_of_atoms,
    run_input,
    is_nep);

  if (is_nep) {
    // Check if the types for this potential are compatible with the possibly other potentials
    check_types(tokens[1]);
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
#endif
}

void Force::reset_pimd_bead_timing()
{
  pimd_bead_timing_ = PIMD_Bead_Timing();
  pimd_bead_timing_.worker_compute.resize(pimd_bead_gpu_workers_.size(), 0.0);
}

#ifdef GPUMD_WPE_ENABLED
void Force::wpe_process_command(
  const std::vector<std::string>& tokens,
  const int current_number_of_atoms)
{
  WpeStageInfo stage{};
  if (wpe_stage_state_.process_command(tokens, current_number_of_atoms, stage))
    gpumd_wpe_activate_stage(potentials, stage);
}
#endif

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
    pimd_qnep_batch_bec_required_ || md_qnep_bec_mode_ == 1 ||
    (md_qnep_bec_mode_ == 0 && md_qnep_bec_required_);
  for (auto& potential : potentials) {
    potential->set_md_qnep_bec(enabled);
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->set_md_qnep_bec(enabled);
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->set_md_qnep_bec(enabled);
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
  pimd_centroid_probe_ready_ = false;
  pimd_centroid_active_this_call_ = false;
  if (!enabled) {
    pimd_centroid_probe_enabled_ = false;
  }
  if (potentials.size() == 1 && potentials[0]) {
    const bool is_batch_potential =
      dynamic_cast<NEP_Charge*>(potentials[0].get()) ||
      dynamic_cast<NEP*>(potentials[0].get())
#ifdef USE_DEEPMD
      || dynamic_cast<DP*>(potentials[0].get())
#endif
      ;
    if (is_batch_potential) {
      potentials[0]->set_neighbor_rebuild(
        enabled ? pimd_bead_neighbor_always_rebuild_ : false);
    }
  }
  apply_pimd_qnep_batch_bec_setting_();
}

void Force::enable_pimd_centroid_probe(
  const int number_of_atoms, const int number_of_beads, const int sample_interval)
{
  pimd_centroid_probe_enabled_ = false;
  pimd_centroid_probe_ready_ = false;
  pimd_centroid_active_this_call_ = false;
  pimd_centroid_number_of_beads_ = 0;
  pimd_centroid_sample_interval_ = 0;
  pimd_force_call_count_ = -1;
  pimd_physical_batch_calls_ = 0;
  pimd_centroid_active_batch_calls_ = 0;
  pimd_centroid_inactive_batch_calls_ = 0;

  if (
    number_of_atoms <= 0 || number_of_beads < 2 || number_of_atoms != number_of_atoms_ ||
    sample_interval <= 0 || !can_use_pimd_qnep_batch_()) {
    return;
  }

  CHECK(gpuSetDevice(0));
  pimd_centroid_position_.resize(number_of_atoms * 3);
  pimd_centroid_potential_.resize(number_of_atoms);
  pimd_centroid_force_.resize(number_of_atoms * 3);
  pimd_centroid_virial_.resize(number_of_atoms * 9);
  pimd_centroid_position_ptrs_.resize(number_of_beads);
  pimd_centroid_position_ptrs_host_.clear();
  pimd_centroid_position_ptrs_host_.reserve(number_of_beads);
  pimd_centroid_number_of_beads_ = number_of_beads;
  pimd_centroid_sample_interval_ = sample_interval;
  pimd_centroid_probe_enabled_ = true;

  printf(
    "Enabled qNEP PIMD centroid auxiliary batch lane (%d physical beads + 1 centroid lane).\n",
    number_of_beads);
  printf(
    "    centroid lane is active only every %d PIMD force call(s) matching HAC sampling.\n",
    sample_interval);
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
  apply_md_qnep_bec_setting_();
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
#ifdef USE_DEEPMD
    if (dynamic_cast<DP*>(potential.get())) {
      continue;
    }
#endif
    potential->set_pimd_batch_profile(enabled);
  }
  for (auto& worker : pimd_bead_gpu_workers_) {
    worker->potential->set_pimd_batch_profile(enabled);
  }
  if (pimd_nep_single_gpu_batch_potential_) {
    pimd_nep_single_gpu_batch_potential_->set_pimd_batch_profile(enabled);
  }
}

void Force::set_pimd_dp_batch_profile(const bool enabled)
{
  pimd_dp_batch_profile_enabled_ = enabled;
#ifdef USE_DEEPMD
  for (auto& potential : potentials) {
    if (auto* dp = dynamic_cast<DP*>(potential.get())) {
      dp->set_pimd_batch_profile(enabled);
    }
  }
#endif
}

void Force::set_pimd_dp_batch_source_count_mode(
  const PIMD_DP_Source_Count_Mode mode)
{
  pimd_dp_source_count_mode_ = mode;
#ifdef USE_DEEPMD
  for (auto& potential : potentials) {
    if (auto* dp = dynamic_cast<DP*>(potential.get())) {
      dp->set_pimd_batch_source_count_options(
        mode == PIMD_DP_Source_Count_Mode::NeighborCounts,
        mode == PIMD_DP_Source_Count_Mode::Check);
    }
  }
#endif
}

void Force::set_pimd_dp_batch_edge_fill_4_threads(const bool enabled)
{
  pimd_dp_batch_edge_fill_4_threads_ = enabled;
#ifdef USE_DEEPMD
  for (auto& potential : potentials) {
    if (auto* dp = dynamic_cast<DP*>(potential.get())) {
      dp->set_pimd_batch_edge_fill_4_threads(enabled);
    }
  }
#endif
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
  if (!pimd_nep_batch_profile_enabled_ && !pimd_dp_batch_profile_enabled_) {
    return;
  }

  if (pimd_dp_batch_profile_enabled_) {
#ifdef USE_DEEPMD
    if (potentials.size() == 1) {
      if (auto* dp = dynamic_cast<DP*>(potentials[0].get())) {
        dp->print_pimd_batch_timing();
      } else {
        printf("DP PIMD batch stage timing unavailable: no single DP potential is selected.\n");
      }
    } else {
      printf("DP PIMD batch stage timing unavailable: no single DP potential is selected.\n");
    }
#else
    printf("DP PIMD batch stage timing unavailable: DeePMD support is not enabled.\n");
#endif
  }

  if (!pimd_nep_batch_profile_enabled_) {
    return;
  }

  bool printed_nep_batch_timing = false;
  auto print_timing = [&](const char* label, const PIMD_Batch_Timing& timing) {
    if (timing.calls == 0) {
      return;
    }
    if (!printed_nep_batch_timing) {
      printf("PIMD NEP/qNEP batch stage timing:\n");
      printed_nep_batch_timing = true;
    }
    printf("    %s (%lld force calls):\n", label, timing.calls);
    printf("        setup = %g s.\n", timing.setup);
    printf("        neighbor = %g s.\n", timing.neighbor);
    printf("            global check/rebuild = %g s.\n", timing.neighbor_global);
    printf("                pointer setup = %g s.\n", timing.neighbor_pointer);
    printf("                distance check = %g s.\n", timing.neighbor_check);
    printf("                flag transfer = %g s.\n", timing.neighbor_flags);
    printf(
      "                global rebuild/update = %g s (%lld large-box bead rebuilds).\n",
      timing.neighbor_rebuild,
      timing.neighbor_rebuild_beads - timing.neighbor_small_box_rebuild_beads);
    printf(
      "                small-box bead rebuilds = %lld; total bead rebuilds = %lld.\n",
      timing.neighbor_small_box_rebuild_beads,
      timing.neighbor_rebuild_beads);
    printf(
      "                rebuild causes (bead calls): displacement-only=%lld, "
      "image-shift-only=%lld (%lld skipped in large-box), both=%lld, "
      "box/PBC-change=%lld, forced=%lld, no-trigger=%lld.\n",
      timing.neighbor_displacement_only_beads,
      timing.neighbor_image_shift_only_beads,
      timing.neighbor_image_shift_only_skipped_beads,
      timing.neighbor_displacement_and_image_shift_beads,
      timing.neighbor_box_or_pbc_change_beads,
      timing.neighbor_forced_rebuild_beads,
      timing.neighbor_no_rebuild_beads);
    printf(
      "                first builds = %lld bead calls; small/large switches = %lld.\n",
      timing.neighbor_first_build_beads,
      timing.neighbor_box_mode_switches);
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
    printf(
      "        PPPM full per-atom virial batch calls = %lld.\n",
      timing.pppm_full_peratom_virial_batch_calls);
    printf(
      "        PPPM global-virial batch calls = %lld.\n",
      timing.pppm_global_virial_batch_calls);
  };

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
  if (is_fcp || primary_nep_model_path_.empty()) {
    return false;
  }
  Potential* primary = potentials[0].get();
  return dynamic_cast<NEP*>(primary) || dynamic_cast<NEP_MULTIGPU*>(primary) ||
         dynamic_cast<NEP_Charge*>(primary);
}

bool Force::can_use_pimd_qnep_batch_() const
{
  if (!pimd_bead_batch_enabled_ || potentials.size() != 1 ||
      multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }
  if (is_fcp) {
    return false;
  }
  Potential* primary = potentials[0].get();
  return primary && dynamic_cast<NEP_Charge*>(primary);
}

bool Force::can_use_pimd_nep_batch_() const
{
  if (!pimd_bead_batch_enabled_ || potentials.size() != 1 ||
      multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }
  if (is_fcp) {
    return false;
  }
  Potential* primary = potentials[0].get();
  return primary && (dynamic_cast<NEP*>(primary) ||
           (dynamic_cast<NEP_MULTIGPU*>(primary) && !primary_nep_model_path_.empty()));
}

bool Force::can_use_pimd_dp_batch_() const
{
#ifndef USE_DEEPMD
  return false;
#else
  if (!pimd_bead_batch_enabled_ || potentials.size() != 1 ||
      multiple_potentials_mode_.compare("observe") != 0) {
    return false;
  }
  if (is_fcp) {
    return false;
  }
  return potentials[0] && dynamic_cast<DP*>(potentials[0].get());
#endif
}

void Force::compute_pimd_centroid_position_(
  Box& box, std::vector<GPU_Vector<double>>& position_beads)
{
  const int number_of_beads = int(position_beads.size());
  const int number_of_atoms = int(pimd_centroid_position_.size() / 3);
  if (
    number_of_beads < 2 || number_of_beads != pimd_centroid_number_of_beads_ ||
    number_of_atoms <= 0) {
    return;
  }

  std::vector<double*> position_ptrs(number_of_beads);
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    position_ptrs[bead_id] = position_beads[bead_id].data();
  }
  if (position_ptrs != pimd_centroid_position_ptrs_host_) {
    pimd_centroid_position_ptrs_.copy_from_host(position_ptrs.data());
    pimd_centroid_position_ptrs_host_ = std::move(position_ptrs);
  }

  gpu_compute_pimd_centroid_position<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    number_of_beads,
    box,
    pimd_centroid_position_ptrs_.data(),
    pimd_centroid_position_.data());
  GPU_CHECK_KERNEL
  gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    box,
    pimd_centroid_position_.data(),
    pimd_centroid_position_.data() + number_of_atoms,
    pimd_centroid_position_.data() + number_of_atoms * 2,
    nullptr);
  GPU_CHECK_KERNEL
}

bool Force::try_compute_pimd_qnep_batch_(
  Box& box,
  GPU_Vector<int>& type,
  std::vector<GPU_Vector<double>>& position_beads,
  std::vector<GPU_Vector<double>>& potential_beads,
  std::vector<GPU_Vector<double>>& force_beads,
  std::vector<GPU_Vector<double>>& virial_beads)
{
  pimd_centroid_probe_ready_ = false;
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
      position_beads[bead_id].data() + number_of_atoms * 2,
      nullptr);
  }
  GPU_CHECK_KERNEL

  std::vector<GPU_Vector<double>*> positions;
  std::vector<GPU_Vector<double>*> potentials_per_bead;
  std::vector<GPU_Vector<double>*> forces;
  std::vector<GPU_Vector<double>*> virials;
  const bool add_centroid_lane =
    pimd_centroid_probe_enabled_ && pimd_centroid_number_of_beads_ == number_of_beads;
  const int capacity_number_of_beads = number_of_beads + (add_centroid_lane ? 1 : 0);
  positions.reserve(capacity_number_of_beads);
  potentials_per_bead.reserve(capacity_number_of_beads);
  forces.reserve(capacity_number_of_beads);
  virials.reserve(capacity_number_of_beads);
  for (int bead_id = 0; bead_id < number_of_beads; ++bead_id) {
    positions.push_back(&position_beads[bead_id]);
    potentials_per_bead.push_back(&potential_beads[bead_id]);
    forces.push_back(&force_beads[bead_id]);
    virials.push_back(&virial_beads[bead_id]);
  }

  if (add_centroid_lane) {
    if (pimd_centroid_active_this_call_) {
      compute_pimd_centroid_position_(box, position_beads);
    }
    positions.push_back(&pimd_centroid_position_);
    potentials_per_bead.push_back(&pimd_centroid_potential_);
    forces.push_back(&pimd_centroid_force_);
    virials.push_back(&pimd_centroid_virial_);
  }

  NEP_Charge* qnep = dynamic_cast<NEP_Charge*>(potentials[0].get());
  const int active_number_of_beads =
    number_of_beads + (add_centroid_lane && pimd_centroid_active_this_call_ ? 1 : 0);
  const bool used_batch = qnep->compute_pimd_batch(
    box,
    type,
    positions,
    potentials_per_bead,
    forces,
    virials,
    active_number_of_beads,
    add_centroid_lane && pimd_centroid_active_this_call_);
  if (used_batch) {
    temperature += number_of_beads * delta_T;
    if (pimd_centroid_probe_enabled_) {
      ++pimd_physical_batch_calls_;
      if (pimd_centroid_active_this_call_) {
        ++pimd_centroid_active_batch_calls_;
      } else {
        ++pimd_centroid_inactive_batch_calls_;
      }
    }
    pimd_centroid_probe_ready_ =
      add_centroid_lane && pimd_centroid_active_this_call_;
  }
  return used_batch;
}

bool Force::compute_qnep_centroid_frames_batch(
  Box& box,
  GPU_Vector<int>& type,
  std::vector<GPU_Vector<double>>& position_frames,
  std::vector<GPU_Vector<double>>& potential_frames,
  std::vector<GPU_Vector<double>>& force_frames,
  std::vector<GPU_Vector<double>>& virial_frames)
{
  if (!can_use_pimd_qnep_batch_() || position_frames.size() < 2 ||
      potential_frames.size() != position_frames.size() ||
      force_frames.size() != position_frames.size() ||
      virial_frames.size() != position_frames.size()) {
    return false;
  }

  std::vector<GPU_Vector<double>*> position_ptrs;
  std::vector<GPU_Vector<double>*> potential_ptrs;
  std::vector<GPU_Vector<double>*> force_ptrs;
  std::vector<GPU_Vector<double>*> virial_ptrs;
  const int number_of_frames = static_cast<int>(position_frames.size());
  position_ptrs.reserve(number_of_frames);
  potential_ptrs.reserve(number_of_frames);
  force_ptrs.reserve(number_of_frames);
  virial_ptrs.reserve(number_of_frames);
  for (int frame = 0; frame < number_of_frames; ++frame) {
    position_ptrs.push_back(&position_frames[frame]);
    potential_ptrs.push_back(&potential_frames[frame]);
    force_ptrs.push_back(&force_frames[frame]);
    virial_ptrs.push_back(&virial_frames[frame]);
  }

  CHECK(gpuSetDevice(0));
  box.set_is_orthogonal();
  NEP_Charge* qnep = dynamic_cast<NEP_Charge*>(potentials[0].get());
  if (!qnep) {
    return false;
  }
  const int number_of_atoms = type.size();
  for (GPU_Vector<double>& position_frame : position_frames) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_frame.data(),
      position_frame.data() + number_of_atoms,
      position_frame.data() + number_of_atoms * 2,
      nullptr);
  }
  GPU_CHECK_KERNEL
  return qnep->compute_pimd_batch(
    box,
    type,
    position_ptrs,
    potential_ptrs,
    force_ptrs,
    virial_ptrs,
    number_of_frames,
    true);
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
      position_beads[bead_id].data() + number_of_atoms * 2,
      nullptr);
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
        new NEP(primary_nep_model_path_.c_str(), number_of_atoms_, *run_input_));
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

bool Force::try_compute_pimd_dp_batch_(
  Box& box,
  GPU_Vector<int>& type,
  std::vector<GPU_Vector<double>>& position_beads,
  std::vector<GPU_Vector<double>>& potential_beads,
  std::vector<GPU_Vector<double>>& force_beads,
  std::vector<GPU_Vector<double>>& virial_beads)
{
#ifndef USE_DEEPMD
  (void)box;
  (void)type;
  (void)position_beads;
  (void)potential_beads;
  (void)force_beads;
  (void)virial_beads;
  return false;
#else
  if (!can_use_pimd_dp_batch_()) {
    return false;
  }
  const int number_of_beads = static_cast<int>(position_beads.size());
  if (
    number_of_beads < 2) {
    return false;
  }
  if (
    potential_beads.size() != position_beads.size() ||
    force_beads.size() != position_beads.size() ||
    virial_beads.size() != position_beads.size()) {
    PRINT_INPUT_ERROR("PIMD DP bead output arrays do not match the position bead count.");
  }
  const int number_of_atoms = type.size();
  DP* dp = dynamic_cast<DP*>(potentials[0].get());
  if (!dp) {
    return false;
  }
  for (int bead = 0; bead < number_of_beads; ++bead) {
    if (
      position_beads[bead].size() != static_cast<size_t>(number_of_atoms) * 3 ||
      potential_beads[bead].size() != static_cast<size_t>(number_of_atoms) ||
      force_beads[bead].size() != static_cast<size_t>(number_of_atoms) * 3 ||
      virial_beads[bead].size() != static_cast<size_t>(number_of_atoms) * 9) {
      PRINT_INPUT_ERROR("PIMD DP bead buffer has an unexpected size.");
    }
  }
  box.set_is_orthogonal();
  if (!dp->can_compute_pimd_batch(box, number_of_atoms, number_of_beads)) {
    return false;
  }
  CHECK(gpuSetDevice(0));
  for (int bead = 0; bead < number_of_beads; ++bead) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_beads[bead].data(),
      position_beads[bead].data() + number_of_atoms,
      position_beads[bead].data() + number_of_atoms * 2,
      nullptr);
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
  for (int bead = 0; bead < number_of_beads; ++bead) {
    positions.push_back(&position_beads[bead]);
    potentials_per_bead.push_back(&potential_beads[bead]);
    forces.push_back(&force_beads[bead]);
    virials.push_back(&virial_beads[bead]);
  }
  const bool used_batch = dp->compute_pimd_batch(
    box, type, positions, potentials_per_bead, forces, virials);
  if (used_batch) {
    temperature += number_of_beads * delta_T;
  }
  return used_batch;
#endif
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
#ifdef USE_DEEPMD
    const bool use_primary_dp_neighbor_setting =
      pimd_bead_batch_enabled_ && dynamic_cast<DP*>(potentials[0].get());
#else
    const bool use_primary_dp_neighbor_setting = false;
#endif
    bool primary_batch_profile = pimd_nep_batch_profile_enabled_;
#ifdef USE_DEEPMD
    if (auto* dp = dynamic_cast<DP*>(potentials[0].get())) {
      primary_batch_profile = pimd_dp_batch_profile_enabled_;
      dp->set_pimd_batch_source_count_options(
        pimd_dp_source_count_mode_ == PIMD_DP_Source_Count_Mode::NeighborCounts,
        pimd_dp_source_count_mode_ == PIMD_DP_Source_Count_Mode::Check);
      dp->set_pimd_batch_edge_fill_4_threads(
        pimd_dp_batch_edge_fill_4_threads_);
    }
#endif
    potentials[0]->set_neighbor_rebuild(
      (use_primary_batch_neighbor_setting || use_primary_dp_neighbor_setting)
        ? pimd_bead_neighbor_always_rebuild_
        : false);
    potentials[0]->set_pppm_mesh_spacing(pppm_mesh_spacing_);
    potentials[0]->set_pimd_batch_bec(pimd_qnep_batch_bec_enabled_());
    potentials[0]->set_pimd_batch_profile(primary_batch_profile);
  }
  if (pimd_bead_gpu_parallel_devices_ <= 1 || primary_nep_model_path_.empty() ||
      number_of_atoms_ <= 0) {
    return;
  }
  if (potentials.size() != 1 || multiple_potentials_mode_.compare("observe") != 0) {
    return;
  }
  if (is_fcp) {
    return;
  }
  Potential* primary = potentials[0].get();
  const bool is_nep_worker = dynamic_cast<NEP*>(primary) || dynamic_cast<NEP_MULTIGPU*>(primary);
  const bool is_qnep_worker = dynamic_cast<NEP_Charge*>(primary);
  if (!(is_nep_worker || is_qnep_worker)) {
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
        new NEP_Charge(primary_nep_model_path_.c_str(), number_of_atoms_, *run_input_));
      qnep_worker->set_neighbor_diagnostics(false);
      worker->potential = std::move(qnep_worker);
    } else {
      worker->potential.reset(
        new NEP(primary_nep_model_path_.c_str(), number_of_atoms_, *run_input_));
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
  ++pimd_force_call_count_;
  pimd_centroid_probe_ready_ = false;
  pimd_centroid_active_this_call_ =
    pimd_centroid_probe_enabled_ && pimd_centroid_sample_interval_ > 0 &&
    pimd_force_call_count_ > 0 &&
    pimd_force_call_count_ % pimd_centroid_sample_interval_ == 0;
  if (!can_use_pimd_bead_gpu_parallel_()) {
    if (
      try_compute_pimd_qnep_batch_(
        box, type, position_beads, potential_beads, force_beads, virial_beads) ||
      try_compute_pimd_dp_batch_(
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
#ifdef USE_DEEPMD
      const bool is_dp = potentials.size() == 1 && potentials[0] &&
                         dynamic_cast<DP*>(potentials[0].get());
#else
      const bool is_dp = false;
#endif
      if (is_qnep) {
        printf("Warning: qNEP ring-polymer bead batching is unavailable; using serial bead forces.\n");
        printf("    batching requires at least two beads, qNEP, and the large-box path.\n");
      } else if (is_nep) {
        printf("Warning: NEP ring-polymer bead batching is unavailable; using serial bead forces.\n");
        printf("    batching requires at least two beads, a standard NEP energy model, and the large-box path.\n");
      } else if (is_dp) {
        printf("Warning: DP ring-polymer bead batching is unavailable; using serial bead forces.\n");
        printf("    batching requires at least two beads, a canonical .pt2 model, and the large-box path.\n");
      } else {
        printf("Warning: ring-polymer bead batching is unavailable; using serial bead forces.\n");
      }
      warned_batch_once = true;
    }
    if (potentials.size() == 1 && potentials[0]) {
      if (auto* qnep = dynamic_cast<NEP_Charge*>(potentials[0].get())) {
        qnep->consume_single_frame_neighbor_reference_invalidation();
      }
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
      position_beads[bead_id].data() + number_of_atoms * 2,
      nullptr);
  }
  GPU_CHECK_KERNEL
  CHECK(gpuDeviceSynchronize());
  pimd_bead_timing_.wrap_positions +=
    std::chrono::duration<double>(Clock::now() - wrap_begin).count();

  if (
    pimd_centroid_active_this_call_ &&
    pimd_centroid_number_of_beads_ == number_of_beads) {
    compute_pimd_centroid_position_(box, position_beads);
    // Worker 0 may use a different host thread/default stream in the multi-GPU
    // path, so make the auxiliary coordinates visible before launching it.
    CHECK(gpuDeviceSynchronize());
  }

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
  std::vector<int> worker_pimd_batch_used(number_of_workers, 0);
  std::vector<int> worker_centroid_batch_used(number_of_workers, 0);
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
          const bool add_centroid_lane =
            worker_id == 0 && qnep && pimd_centroid_probe_enabled_ &&
            pimd_centroid_number_of_beads_ == number_of_beads;
          const int reserve_size = bead_end - bead_begin + (add_centroid_lane ? 1 : 0);
          worker_positions.reserve(reserve_size);
          worker_potentials.reserve(reserve_size);
          worker_forces.reserve(reserve_size);
          worker_virials.reserve(reserve_size);
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
          if (add_centroid_lane) {
            worker_positions.push_back(&pimd_centroid_position_);
            worker_potentials.push_back(&pimd_centroid_potential_);
            worker_forces.push_back(&pimd_centroid_force_);
            worker_virials.push_back(&pimd_centroid_virial_);
          }
          if (qnep) {
            const int active_number_of_beads =
              bead_end - bead_begin +
              (add_centroid_lane && pimd_centroid_active_this_call_ ? 1 : 0);
            used_pimd_batch = qnep->compute_pimd_batch(
              worker_box,
              device_id == 0 ? type : worker.type,
              worker_positions,
              worker_potentials,
              worker_forces,
              worker_virials,
              active_number_of_beads,
              add_centroid_lane && pimd_centroid_active_this_call_);
          } else {
            used_pimd_batch = nep->compute_pimd_batch(
              worker_box,
              device_id == 0 ? type : worker.type,
              worker_positions,
              worker_potentials,
              worker_forces,
              worker_virials);
          }
          worker_pimd_batch_used[worker_id] = used_pimd_batch ? 1 : 0;
          if (used_pimd_batch && add_centroid_lane && pimd_centroid_active_this_call_) {
            worker_centroid_batch_used[worker_id] = 1;
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
  pimd_centroid_probe_ready_ = worker_centroid_batch_used[0] != 0;
  bool all_workers_used_pimd_batch = true;
  for (const int used : worker_pimd_batch_used) {
    if (used == 0) {
      all_workers_used_pimd_batch = false;
      break;
    }
  }
  if (pimd_centroid_probe_enabled_ && all_workers_used_pimd_batch) {
    ++pimd_physical_batch_calls_;
    if (pimd_centroid_active_this_call_) {
      ++pimd_centroid_active_batch_calls_;
    } else {
      ++pimd_centroid_inactive_batch_calls_;
    }
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

  if (device_id == 0 && bead_begin == 0) {
    pimd_centroid_probe_ready_ = false;
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
        position_beads[bead_id].data() + number_of_atoms * 2,
        nullptr);
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
  if (pimd_centroid_probe_enabled_) {
    printf("PIMD centroid auxiliary lane scheduling:\n");
    printf(
      "    physical PIMD batch calls (including initial force) = %lld\n",
      pimd_physical_batch_calls_);
    printf("    centroid-active batch calls = %lld\n", pimd_centroid_active_batch_calls_);
    printf("    centroid-inactive batch calls = %lld\n", pimd_centroid_inactive_batch_calls_);
  }
  pimd_centroid_probe_enabled_ = false;
  pimd_centroid_probe_ready_ = false;
  pimd_centroid_active_this_call_ = false;
  pimd_centroid_sample_interval_ = 0;
  pimd_force_call_count_ = 0;
  pimd_physical_batch_calls_ = 0;
  pimd_centroid_active_batch_calls_ = 0;
  pimd_centroid_inactive_batch_calls_ = 0;
  multiple_potentials_mode_ = "observe";
  refresh_pimd_bead_gpu_workers_();
}

void Force::notify_velocity_update()
{
  for (auto& potential : potentials) {
    if (auto* qnep = dynamic_cast<NEP_Charge*>(potential.get())) {
      qnep->notify_velocity_update();
    }
  }
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

static __global__ void gpu_compute_pimd_centroid_position(
  const int N,
  const int number_of_beads,
  const Box box,
  double* const* g_position_beads,
  double* g_centroid_position)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double reference_x = g_position_beads[0][n];
    const double reference_y = g_position_beads[0][n + N];
    const double reference_z = g_position_beads[0][n + N * 2];
    double centroid_x = reference_x;
    double centroid_y = reference_y;
    double centroid_z = reference_z;

    for (int bead_id = 1; bead_id < number_of_beads; ++bead_id) {
      double dx = g_position_beads[bead_id][n] - reference_x;
      double dy = g_position_beads[bead_id][n + N] - reference_y;
      double dz = g_position_beads[bead_id][n + N * 2] - reference_z;
      apply_mic(box, dx, dy, dz);
      centroid_x += reference_x + dx;
      centroid_y += reference_y + dy;
      centroid_z += reference_z + dz;
    }

    const double inverse_number_of_beads = 1.0 / number_of_beads;
    g_centroid_position[n] = centroid_x * inverse_number_of_beads;
    g_centroid_position[n + N] = centroid_y * inverse_number_of_beads;
    g_centroid_position[n + N * 2] = centroid_z * inverse_number_of_beads;
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

void Force::set_temperature_range(
  const double temperature1, const double temperature2, const int number_of_steps)
{
  temperature = temperature1;
  delta_T = (temperature2 - temperature1) / number_of_steps;
}

void Force::advance_temperature() { temperature += delta_T; }

int Force::get_number_of_potentials() const { return potentials.size(); }

Potential& Force::get_potential(const int index) { return *potentials[index]; }

const RunInput& Force::get_run_input() const { return *run_input_; }

void Force::prepare_compute(
  const int number_of_atoms,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  int* position_image)
{
  box.set_is_orthogonal();

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

  qnep->compute_non_electro(
    box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  return true;
}
void Force::compute_single_potential(
  Potential& potential,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (3 == potential.nep_model_type) {
    potential.compute(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else if (1 == potential.ilp_flag) {
    potential.compute_ilp(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, group);
  } else {
    potential.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

void Force::compute_potentials(
  const int number_of_atoms,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (multiple_potentials_mode_.compare("observe") == 0) {
    // If observing, calculate using main potential only
    compute_single_potential(
      *potentials[0],
      box,
      position_per_atom,
      type,
      group,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else if (multiple_potentials_mode_.compare("average") == 0) {
    // Calculate average potential, force and virial per atom.
    for (int i = 0; i < potentials.size(); i++) {
      // potential->compute automatically adds the properties
      compute_single_potential(
        *potentials[i],
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);
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
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  prepare_compute(
    number_of_atoms,
    box,
    position_per_atom,
    potential_per_atom,
    force_per_atom,
    virial_per_atom,
    nullptr);
  compute_potentials(
    number_of_atoms,
    box,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& mass_per_atom,
  int* position_image)
{
  const int number_of_atoms = type.size();
  prepare_compute(
    number_of_atoms,
    box,
    position_per_atom,
    potential_per_atom,
    force_per_atom,
    virial_per_atom,
    position_image);
  compute_potentials(
    number_of_atoms,
    box,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

}
