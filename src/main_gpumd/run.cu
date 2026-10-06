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
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "force/force.cuh"
#include "force/nep.cuh"
#include "force/nep_charge.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"

#include "measure/active.cuh"
#include "measure/add_efield.cuh"
#include "measure/add_force.cuh"
#include "measure/add_spring.cuh"
#include "measure/adf.cuh"
#include "measure/angular_rdf.cuh"
#include "measure/compute.cuh"
#include "measure/compute_chunk.cuh"
#include "measure/compute_dpdt.cuh"
#include "measure/compute_es.cuh"
#include "measure/centroid_force_diagnostic.cuh"
#include "measure/centroid_deltaF_O.cuh"
#include "measure/qnep_projection.cuh"
#include "measure/dos.cuh"
#include "measure/deform.cuh"
#include "measure/dump_beads.cuh"
#include "measure/dump_dipole.cuh"
#include "measure/dump_exyz.cuh"
#include "measure/dump_force.cuh"
#include "measure/dump_netcdf.cuh"
#include "measure/dump_observer.cuh"
#include "measure/dump_polarizability.cuh"
#include "measure/dump_position.cuh"
#include "measure/dump_pimd_restart.cuh"
#include "measure/dump_restart.cuh"
#include "measure/dump_shock_nemd.cuh"
#include "measure/dump_thermo.cuh"
#include "measure/dump_velocity.cuh"
#include "measure/dump_xyz.cuh"
#include "measure/dump_cg.cuh"
#include "measure/extrapolation.cuh"
#include "measure/hac.cuh"
#include "measure/quantum_heat_moments.cuh"
#include "measure/hnemd_kappa.cuh"
#include "measure/hnemdec_kappa.cuh"
#include "measure/lsqt.cuh"
#include "measure/measure.cuh"
#include "measure/modal_analysis.cuh"
#include "measure/iron_conductivity.cuh"
#include "measure/msd.cuh"
#include "measure/orientorder.cuh"
#include "measure/plumed.cuh"
#include "measure/property.cuh"
#include "measure/proton_tunneling.cuh"
#include "measure/rpmd_ja_reference.cuh"
#include "measure/rpmd_ja_fit.cuh"
#include "measure/rpmd_ja_native_fit.cuh"
#include "measure/rpmd_ja_qnep_prepare.cuh"
#include "measure/rdf.cuh"
#include "measure/sdc.cuh"
#include "measure/shc.cuh"
#include "measure/viscosity.cuh"
#include "mc/mc.cuh"

#include "measure/measure.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "model/read_pimd_restart.cuh"
#include "phonon/hessian.cuh"
#include "replicate.cuh"
#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include "velocity.cuh"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <string>

#include <cmath>
#include <cstring>

static __global__ void gpu_find_largest_v2(
  int N, int number_of_rounds, double* g_vx, double* g_vy, double* g_vz, double* g_v2_max)
{
  int tid = threadIdx.x;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = round * 1024 + tid;
    if (n < N) {
      double vx = g_vx[n];
      double vy = g_vy[n];
      double vz = g_vz[n];
      double v2 = vx * vx + vy * vy + vz * vz;
      if (s_data[tid] < v2) {
        s_data[tid] = v2;
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_data[tid] < s_data[tid + offset]) {
        s_data[tid] = s_data[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_v2_max[0] = s_data[0];
  }
}

__device__ double device_v2_max[1];

static void calculate_time_step(
  double max_distance_per_step,
  GPU_Vector<double>& velocity_per_atom,
  double initial_time_step,
  double& time_step)
{
  if (max_distance_per_step <= 0.0) {
    return;
  }
  const int N = velocity_per_atom.size() / 3;
  double* gpu_v2_max;
  CHECK(gpuGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N,
    (N - 1) / 1024 + 1,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    gpu_v2_max);
  GPU_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(gpuMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), gpuMemcpyDeviceToHost));
  double cpu_v_max = sqrt(cpu_v2_max[0]);
  double time_step_min = max_distance_per_step / cpu_v_max;

  if (time_step_min < initial_time_step) {
    time_step = time_step_min;
  } else {
    time_step = initial_time_step;
  }
}

static bool parse_initial_replicate(
  const RunInput& run_input, int replicate_size[3])
{
  for (const auto& line : run_input.lines()) {
    if (line.tokens.empty()) {
      continue;
    }
    if (line.tokens[0] != "replicate") {
      return false;
    }
    parse_replicate(line.tokens, replicate_size);
    return true;
  }
  return false;
}

Run::Run(const RunInput& run_input)
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  has_replicate_ = parse_initial_replicate(run_input, replicate_size_);

  initialize_position(
    run_input,
    has_velocity_in_xyz,
    number_of_types,
    box,
    group,
    atom);
  first_potential_filename_ = get_first_potential_filename(run_input);

  if (has_replicate_) {
    Replicate(replicate_size_, box, atom, group);
  }

  if (atom.number_of_atoms < 2) {
    PRINT_INPUT_ERROR("Number of atoms should >= 2.");
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz, 300, atom, false, 123);

  if (has_velocity_in_xyz) {
    printf("Initialized velocities with data in model.xyz.\n");
  } else {
    printf("Initialized velocities with default T = 300 K.\n");
  }

  allocate_memory_gpu(group, atom, thermo);

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  execute_run_in(run_input);
}

void Run::execute_run_in(const RunInput& run_input)
{
  print_line_1();
  printf("Started executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  bool first_effective_command = true;
  for (const auto& line : run_input.lines()) {
    if (!line.tokens.empty()) {
      std::vector<std::string> tokens = line.tokens;
      if (tokens.size() >= 2 && tokens[0] == "potential") {
        tokens[1] = get_compact_nep_filename(tokens[1]);
      }
      if (has_replicate_ && first_effective_command &&
          tokens[0] == "replicate") {
        first_effective_command = false;
        continue;
      }
      first_effective_command = false;
      parse_one_keyword(tokens, run_input);
    }
  }

  if (integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("The last ensemble is not followed by a run.");
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

}

void Run::compute_force()
{
  if (is_pimd(integrate.get_type())) {
    force.compute_pimd_beads(
      box,
      atom.type,
      group,
      atom.position_beads,
      atom.potential_beads,
      atom.force_beads,
      atom.virial_beads,
      atom.velocity_beads,
      atom.mass);
  } else {
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.velocity_per_atom,
      atom.mass,
      atom.position_image.size() > 0 ? atom.position_image.data() : nullptr);
  }
}

void Run::perform_a_run(const int number_of_steps)
{

  HAC* centroid_force_hac = nullptr;
  QuantumHeatMoments* quantum_heat_moments = nullptr;
  Centroid_Force_Diagnostic* centroid_force_diagnostic = nullptr;
#ifdef USE_NETCDF
  Centroid_DeltaF_O* centroid_deltaF_O = nullptr;
#endif
  auto& actions = measure.get_actions();
  RpmdJA_Fit* rpmd_ja_fit = nullptr;
  for (const auto& action : actions) {
    if (action->action_name == "rpmd_ja_fit") {
      if (rpmd_ja_fit != nullptr) PRINT_INPUT_ERROR("rpmd_ja fit may be specified only once per run.");
      rpmd_ja_fit = dynamic_cast<RpmdJA_Fit*>(action.get());
    }
  }
  if (rpmd_ja_fit != nullptr) {
    if (rpmd_ja_enabled_) PRINT_INPUT_ERROR("rpmd_ja fit cannot run in the same segment as rpmd_ja on.");
    if (max_distance_per_step > 0.0) {
      PRINT_INPUT_ERROR("rpmd_ja fit requires a fixed integration time step; adaptive max_distance_per_step is unsupported.");
    }
    if (force.get_number_of_potentials() != 1) {
      PRINT_INPUT_ERROR("rpmd_ja fit requires exactly one supported qNEP potential.");
    }
    auto* active_qnep = dynamic_cast<NEP_Charge*>(&force.get_potential(0));
    if (active_qnep == nullptr ||
        (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) ||
        !active_qnep->uses_pppm()) {
      PRINT_INPUT_ERROR("rpmd_ja fit currently supports exactly one qNEP potential in charge mode 1 or 2 with PPPM.");
    }
    try {
      rpmd_ja_qnep_config_fingerprint(force);
    } catch (const std::exception& error) {
      const std::string message = std::string("rpmd_ja fit cannot use the current qNEP configuration: ") + error.what();
      PRINT_INPUT_ERROR(message.c_str());
    }
    for (const auto& action : actions) {
      if (action.get() == rpmd_ja_fit) continue;
      const std::string& name = action->action_name;
      if (action->modifies_force() || name == "compute_hnemd" || name == "compute_hnemdec" ||
          name == "compute_hnema" || name == "plumed" || name == "deform" ||
          name == "compute_es" || name == "active" || name == "dump_observer") {
        const std::string error = "rpmd_ja fit does not support external drive/action " + name + ".";
        PRINT_INPUT_ERROR(error.c_str());
      }
    }
  }
  if (is_pimd(integrate.get_type())) {
    for (const auto& action : actions) {
      if (action->modifies_force() && !action->supports_ring_polymer_force()) {
        const std::string error =
          action->action_name +
          " is not currently supported with ring-polymer dynamics because its force modification is not applied to force_beads.";
        PRINT_INPUT_ERROR(error.c_str());
      }
      if (action->has_undefined_ring_polymer_charge_bec_output()) {
        PRINT_INPUT_ERROR(
          "qNEP charge/BEC outputs (dump_xyz, dump_netcdf, compute_dpdt) currently do not "
          "support ring-polymer dynamics.\n");
      }
    }
  }
  for (const auto& action : actions) {
    if (action->action_name == "compute_hac" && centroid_force_hac == nullptr) {
      centroid_force_hac = dynamic_cast<HAC*>(action.get());
    } else if (action->action_name == "compute_quantum_heat_moments") {
      quantum_heat_moments = dynamic_cast<QuantumHeatMoments*>(action.get());
    }
  }
  if (rpmd_ja_enabled_ && centroid_force_hac == nullptr) {
    PRINT_INPUT_ERROR("rpmd_ja on requires compute_hac in the same run.");
  }
  if (rpmd_ja_enabled_) {
    if (max_distance_per_step > 0.0) {
      PRINT_INPUT_ERROR("rpmd_ja requires a fixed integration time step; adaptive max_distance_per_step is unsupported.");
    }
    if (integrate.get_fixed_group() != -1 || integrate.get_move_group() != -1) {
      PRINT_INPUT_ERROR("rpmd_ja does not support fixed or moving atom groups.");
    }
    for (const auto& action : actions) {
      const std::string& name = action->action_name;
      if (
        action->modifies_force() || name == "compute_hnemd" || name == "compute_hnemdec" ||
        name == "compute_hnema" || name == "plumed" || name == "deform") {
        const std::string error = "rpmd_ja does not support the external drive/action " + name + ".";
        PRINT_INPUT_ERROR(error.c_str());
      }
    }
  }
  const bool has_hnemd_driving = std::any_of(
    actions.begin(), actions.end(), [](const std::unique_ptr<Action>& action) {
      return action->action_name == "compute_hnemd" || action->action_name == "compute_hnemdec";
    });
  if (quantum_heat_moments != nullptr && has_hnemd_driving) {
    PRINT_INPUT_ERROR("compute_quantum_heat_moments does not support HNEMD or HNEMDEC driven sampling.");
  }
  if (quantum_heat_moments != nullptr) {
    quantum_heat_moments->set_hac(centroid_force_hac);
    auto quantum_it = std::find_if(actions.begin(), actions.end(),
      [quantum_heat_moments](const std::unique_ptr<Action>& action) {
        return action.get() == quantum_heat_moments;
      });
    auto hac_it = std::find_if(actions.begin(), actions.end(),
      [centroid_force_hac](const std::unique_ptr<Action>& action) {
        return action.get() == centroid_force_hac;
      });
    if (hac_it != actions.end() && quantum_it < hac_it) {
      std::rotate(quantum_it, quantum_it + 1, hac_it + 1);
    }
  }
  for (const auto& property : measure.properties) {
    if (property->property_name == "centroid_force_diagnostic") {
      centroid_force_diagnostic =
        dynamic_cast<Centroid_Force_Diagnostic*>(property.get());
#ifdef USE_NETCDF
    } else if (property->property_name == "centroid_deltaF_O") {
      centroid_deltaF_O = dynamic_cast<Centroid_DeltaF_O*>(property.get());
#endif
    }
  }
#ifdef USE_NETCDF
  if (centroid_deltaF_O != nullptr) {
    if (centroid_force_diagnostic == nullptr) {
      PRINT_INPUT_ERROR(
        "centroid_deltaF_O requires centroid_force_diagnostic in the same run.\n");
    }
    if (centroid_deltaF_O->sample_interval() != centroid_force_diagnostic->sample_interval()) {
      PRINT_INPUT_ERROR(
        "centroid_deltaF_O sample interval must match centroid_force_diagnostic.\n");
    }
    if (number_of_steps < centroid_deltaF_O->sample_interval()) {
      PRINT_INPUT_ERROR(
        "centroid_deltaF_O requires at least one diagnostic sampling frame.\n");
    }
  }
#endif
  if (centroid_force_diagnostic != nullptr) {
    if (has_hnemd_driving) {
      PRINT_INPUT_ERROR(
        "centroid_force_diagnostic requires physical forces without HNEMD/HNEMDEC driving.\n");
    }
    centroid_force_diagnostic->set_hac(centroid_force_hac);
#ifdef USE_NETCDF
    if (centroid_deltaF_O != nullptr) {
      centroid_deltaF_O->set_hac(centroid_force_hac);
    }
#endif
    for (const auto& action : actions) {
      if (
        action->action_name == "compute_es" || action->action_name == "active" ||
        action->action_name == "dump_observer" || action->action_name == "plumed") {
        PRINT_INPUT_ERROR(
          "centroid_force_diagnostic cannot be combined with actions that rewrite "
          "atom.force_per_atom at end_of_step.\n");
      }
    }
  }

  integrate.initialize(time_step, atom, box, group);
  measure.pre_run(number_of_steps, time_step, integrate, group, atom, box, force);

  const bool requires_bec = measure.requires_bec();
  const bool classical_md = !is_pimd(integrate.get_type());
  force.set_md_qnep_bec_required(classical_md && requires_bec);
  force.set_pimd_qnep_batch_bec_required(!classical_md && requires_bec);

  // setup force for the first integrate step
  compute_force();
  atom.update_unwrapped_position(box);
  measure.setup_force(time_step, integrate, group, atom, box, force);
  measure.process_dynamics(0, box, atom);

  double initial_time_step = time_step;
  const bool profile_pimd_bead_parallel =
    is_pimd(integrate.get_type()) &&
    force.pimd_bead_gpu_parallel_available();
  const bool profile_pimd_batch = force.pimd_nep_batch_profile_enabled() ||
    force.pimd_dp_batch_profile_enabled();
  double pimd_compute1_time = 0.0;
  double pimd_compute2_time = 0.0;
  if (profile_pimd_bead_parallel) {
    force.reset_pimd_bead_timing();
  }
  if (profile_pimd_batch) {
    force.reset_pimd_nep_batch_profile();
  }

  const auto time_begin = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < number_of_steps; ++step) {

    velocity.correct_velocity(step, group, atom);

    calculate_time_step(
      max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;

    std::chrono::high_resolution_clock::time_point compute1_begin;
    if (profile_pimd_bead_parallel) {
      compute1_begin = std::chrono::high_resolution_clock::now();
    }
    integrate.compute1(time_step, step, number_of_steps, group, box, atom, thermo);
    if (profile_pimd_bead_parallel) {
      CHECK(gpuSetDevice(0));
      CHECK(gpuDeviceSynchronize());
      pimd_compute1_time += std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now() - compute1_begin)
                              .count();
    }

    measure.post_integrate1(step, time_step, integrate, group, atom, box, force);

    force.advance_temperature();

    measure.pre_force(step, time_step, integrate, group, atom, box, force);
    compute_force();

    atom.update_unwrapped_position(box);
    measure.post_force(step, time_step, global_time, integrate, group, atom, box, force);
    measure.process_dynamics(step + 1, box, atom);

    std::chrono::high_resolution_clock::time_point compute2_begin;
    if (profile_pimd_bead_parallel) {
      compute2_begin = std::chrono::high_resolution_clock::now();
    }
    integrate.compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
    force.notify_velocity_update();
    atom.update_unwrapped_position(box);
    if (profile_pimd_bead_parallel) {
      CHECK(gpuSetDevice(0));
      CHECK(gpuDeviceSynchronize());
      pimd_compute2_time += std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now() - compute2_begin)
                              .count();
    }

    measure.end_of_step(
      number_of_steps,
      step,
      integrate.get_fixed_group(),
      integrate.get_move_group(),
      global_time,
      integrate.get_temperature2(),
      integrate,
      box,
      group,
      thermo,
      atom,
      force);

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
      fflush(stdout);
    }
  }

  print_line_1();
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  printf("Time used for this run = %g second.\n", time_used.count());
  double run_speed = atom.number_of_atoms * (number_of_steps * 1.0 / time_used.count());
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  if (profile_pimd_bead_parallel) {
    const auto& timing = force.get_pimd_bead_timing();
    const double measured = pimd_compute1_time + timing.total + pimd_compute2_time;
    const double other = std::max(0.0, time_used.count() - measured);
    const auto percentage = [&](const double seconds) {
      return time_used.count() > 0.0 ? seconds * 100.0 / time_used.count() : 0.0;
    };
    printf("Ring-polymer bead-parallel timing (%lld force calls):\n", timing.calls);
    printf("    compute1/integrator = %g s (%g%%).\n", pimd_compute1_time, percentage(pimd_compute1_time));
    printf("    force/wrap positions = %g s (%g%%).\n", timing.wrap_positions, percentage(timing.wrap_positions));
    printf("    force/stage remote = %g s (%g%%).\n", timing.stage_remote, percentage(timing.stage_remote));
    printf("    force/worker wall = %g s (%g%%).\n", timing.compute_workers, percentage(timing.compute_workers));
    for (size_t worker_id = 0; worker_id < timing.worker_compute.size(); ++worker_id) {
      printf("        GPU %zu worker compute = %g s.\n", worker_id, timing.worker_compute[worker_id]);
    }
    printf("    force/gather remote = %g s (%g%%).\n", timing.gather_remote, percentage(timing.gather_remote));
    printf("    compute2/integrator = %g s (%g%%).\n", pimd_compute2_time, percentage(pimd_compute2_time));
    printf("    other run work = %g s (%g%%).\n", other, percentage(other));
  }
  if (profile_pimd_batch) {
    force.print_pimd_nep_batch_profile();
  }
  print_line_2();

  measure.post_run(
    atom, box, integrate, number_of_steps, time_step, integrate.get_temperature2());

  integrate.finalize(atom, box);
  velocity.finalize();
  force.finalize();
  const auto total_finish = std::chrono::high_resolution_clock::now();
  printf(
    "Total wall time including dynamics and postprocess = %g second.\n",
    std::chrono::duration<double>(total_finish - time_begin).count());
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(
  const std::vector<std::string>& tokens, const RunInput& run_input)
{
  if (tokens.empty()) return;
  if (tokens[0] == "replicate") {
    PRINT_INPUT_ERROR("replicate must be the first effective command.");
  }
  const int num_param = static_cast<int>(tokens.size());
  const int max_num_param = 64;
  if (num_param > max_num_param) {
    PRINT_INPUT_ERROR("The number of parameters should be less than 64.\n");
  }
  const char* param[max_num_param];
  for (int n = 0; n < num_param; ++n) param[n] = tokens[n].c_str();

#ifdef GPUMD_WPE_ENABLED
  force.wpe_process_command(tokens, atom.number_of_atoms);
#endif

  if (tokens[0] == "potential") {
    force.parse_potential(tokens, box, atom.type.size(), run_input);
  } else if (tokens[0] == "minimize") {
    Minimize minimize;
    minimize.parse_minimize(
      tokens, integrate.get_fixed_group(), integrate.get_fixed_grouping_method(),
      force, box, atom, group);
  } else if (tokens[0] == "compute_phonon") {
    Hessian hessian;
    hessian.parse(tokens);
    if (!has_replicate_) PRINT_INPUT_ERROR("replicate keyword not found in run.in file.");
    hessian.compute(force, box, atom, group, replicate_size_);
  } else if (tokens[0] == "compute_cohesive") {
    Cohesive cohesive;
    cohesive.parse(tokens, 0);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "compute_elastic") {
    Cohesive cohesive;
    cohesive.parse(tokens, 1);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "change_box") {
    parse_change_box(tokens);
  } else if (tokens[0] == "velocity") {
    parse_velocity(tokens);
  } else if (tokens[0] == "ensemble") {
    integrate.parse_ensemble(tokens, atom, box, group);
  } else if (tokens[0] == "pimd_propagator") {
    parse_pimd_propagator(param, num_param);
  } else if (tokens[0] == "pimd_pile_scale") {
    parse_pimd_pile_scale(param, num_param);
  } else if (tokens[0] == "pimd_fix_com") {
    parse_pimd_fix_com(param, num_param);
  } else if (tokens[0] == "pimd_reseed_from_centroid") {
    parse_pimd_reseed_from_centroid(param, num_param);
  } else if (tokens[0] == "pimd_bead_gpu_parallel") {
    parse_pimd_bead_gpu_parallel(param, num_param);
  } else if (tokens[0] == "pimd_bead_neighbor_rebuild") {
    parse_pimd_bead_neighbor_rebuild(param, num_param);
  } else if (tokens[0] == "md_qnep_bec") {
    parse_md_qnep_bec(param, num_param);
  } else if (tokens[0] == "pimd_bead_batch") {
    parse_pimd_bead_batch(param, num_param);
  } else if (tokens[0] == "pimd_qnep_bead_batch") {
    parse_pimd_qnep_bead_batch(param, num_param);
  } else if (tokens[0] == "pimd_qnep_batch_bec") {
    parse_pimd_qnep_batch_bec(param, num_param);
  } else if (tokens[0] == "pimd_nep_bead_batch") {
    parse_pimd_nep_bead_batch(param, num_param);
  } else if (tokens[0] == "pimd_nep_batch_profile" || tokens[0] == "pimd_dp_batch_profile") {
    parse_pimd_nep_batch_profile(param, num_param);
  } else if (tokens[0] == "pimd_dp_batch_source_count") {
    parse_pimd_dp_batch_source_count(param, num_param);
  } else if (tokens[0] == "pimd_dp_batch_edge_fill_4_threads") {
    parse_pimd_dp_batch_edge_fill_4_threads(param, num_param);
  } else if (tokens[0] == "pimd_nep_batch_geometry_cache") {
    parse_pimd_nep_batch_geometry_cache(param, num_param);
  } else if (tokens[0] == "pppm_mesh_spacing") {
    parse_pppm_mesh_spacing(param, num_param);
  } else if (tokens[0] == "read_pimd_restart") {
    parse_read_pimd_restart(param, num_param);
  } else if (tokens[0] == "time_step") {
    parse_time_step(tokens);
  } else if (tokens[0] == "correct_velocity") {
    parse_correct_velocity(tokens, group);
  } else if (tokens[0] == "fix") {
    integrate.parse_fix(tokens, group);
  } else if (tokens[0] == "move") {
    integrate.parse_move(tokens, group);
  } else if (tokens[0] == "kspace") {
    if (has_seen_kspace_command) PRINT_INPUT_ERROR("kspace can only appear once.");
    has_seen_kspace_command = true;
  } else if (tokens[0] == "dftd3") {
    if (has_seen_dftd3_command) PRINT_INPUT_ERROR("dftd3 can only appear once.");
    has_seen_dftd3_command = true;
  } else if (tokens[0] == "hac_current") {
    if (num_param != 2) {
      PRINT_INPUT_ERROR("hac_current should have exactly one parameter: legacy or qnep_full_a.\n");
    }
    if (hac_current_option_seen_) PRINT_INPUT_ERROR("hac_current may appear only once in one run.\n");
    if (tokens[1] == "legacy") {
      hac_current_qnep_full_a_ = false;
    } else if (tokens[1] == "qnep_full_a") {
      hac_current_qnep_full_a_ = true;
    } else {
      PRINT_INPUT_ERROR("hac_current must be legacy or qnep_full_a.\n");
    }
    hac_current_option_seen_ = true;
    measure.set_hac_current(hac_current_qnep_full_a_);
  } else if (tokens[0] == "rpmd_ja") {
    parse_rpmd_ja(tokens);
  } else if (tokens[0] == "run") {
    parse_run(tokens);
    hac_current_option_seen_ = false;
    hac_current_qnep_full_a_ = false;
    measure.set_hac_current(false);
    rpmd_ja_option_seen_ = false;
    rpmd_ja_enabled_ = false;
    rpmd_ja_reference_path_.clear();
    measure.set_rpmd_ja(false);
  } else if (!measure.parse_action(
               tokens, number_of_types, integrate, group, atom, box, force,
               first_potential_filename_)) {
    PRINT_KEYWORD_ERROR(tokens[0].c_str());
  }
}

void Run::parse_rpmd_ja(const std::vector<std::string>& tokens)
{
  if (tokens.size() < 2) {
    PRINT_INPUT_ERROR("rpmd_ja expects off, on <referencefile>, fit_samples <samples_file> <outfile> <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> [<qraw>], diagnose <fd_step> [full], diagnose_samples <samples_file> <fd_step> [full], generate <file> <T> <fd_step>, generate_sparse <file> <T> <fd_step> <kernel_table>, generate_raw <rawfile> <T> <fd_step> <kernel_table>, or prepare <rawfile> <outfile> <kernel_table> [<additive-pack>].");
  }
  if (tokens[1] == "fit") {
    if (!measure.parse_action(
          tokens, number_of_types, integrate, group, atom, box, force, first_potential_filename_)) {
      PRINT_INPUT_ERROR("Could not register rpmd_ja fit sampler.");
    }
    return;
  }
  if (tokens[1] == "fit_samples") {
    if (tokens.size() != 9 && tokens.size() != 10)
      PRINT_INPUT_ERROR("rpmd_ja fit_samples requires <samples_file> <outfile> <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> [<qraw>].");
    if (global_time != 0.0)
      PRINT_INPUT_ERROR("rpmd_ja fit_samples must appear before any run.");
    if (force.potentials.size() != 1 || force.primary_nep_model_path().empty())
      PRINT_INPUT_ERROR("rpmd_ja fit_samples requires exactly one qNEP potential.");
    auto* active_qnep = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
    if (active_qnep == nullptr || (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) ||
        !active_qnep->uses_pppm())
      PRINT_INPUT_ERROR("rpmd_ja fit_samples supports qNEP charge mode 1 or 2 with PPPM only.");
    if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1)
      PRINT_INPUT_ERROR("rpmd_ja fit_samples requires fully periodic boundaries.");
    RpmdJANativeFitOptions options;
    options.output_path = tokens[3];
    options.kernel_table = tokens[8];
    if (tokens.size() == 10) options.raw_input_path = tokens[9];
    options.internal_mass_com = integrate.get_pimd_fix_com();
    if (tokens[2].empty() || options.output_path.empty() || options.kernel_table.empty() ||
        (tokens.size() == 10 && options.raw_input_path.empty()) ||
        tokens[2] == options.output_path || (!options.raw_input_path.empty() &&
        (options.raw_input_path == options.output_path || options.raw_input_path == tokens[2])))
      PRINT_INPUT_ERROR("rpmd_ja fit_samples paths must be nonempty and the output must differ from its inputs.");
    double* values[] = {&options.cutoff, &options.epsilon, &options.response_tolerance, &options.fd_step};
    for (int i = 0; i < 4; ++i) {
      char* end = nullptr;
      *values[i] = std::strtod(tokens[4 + i].c_str(), &end);
      if (end == tokens[4 + i].c_str() || *end != '\0' || !std::isfinite(*values[i]) || *values[i] <= 0.0)
        PRINT_INPUT_ERROR("rpmd_ja fit_samples numeric arguments must be positive finite numbers.");
    }
#ifdef USE_HIP
    PRINT_INPUT_ERROR("rpmd_ja fit_samples native reference preparation is currently unavailable in HIP builds.");
#endif
    fit_rpmd_ja_native_reference_from_samples(options, tokens[2], atom, box, force);
    return;
  }
  if (tokens[1] == "prepare") {
    if (tokens.size() != 5 && tokens.size() != 6) PRINT_INPUT_ERROR("rpmd_ja prepare requires <rawfile> <outfile> <kernel_table> [<additive-pack>].");
    if (integrate.has_ensemble() || global_time != 0.0)
      PRINT_INPUT_ERROR("rpmd_ja prepare must appear before any ensemble or run.");
    prepare_rpmd_ja_qnep_reference(tokens[2], tokens[3], tokens[4], make_rpmd_ja_qnep_mode_validator(atom, box, force), tokens.size() == 6 ? tokens[5] : std::string());
    return;
  }
  if (tokens[1] == "generate_raw") {
    if (tokens.size() != 6) PRINT_INPUT_ERROR("rpmd_ja generate_raw requires <rawfile> <T> <fd_step> <kernel_table>.");
    if (integrate.has_ensemble() || global_time != 0.0)
      PRINT_INPUT_ERROR("rpmd_ja generate_raw must appear before any ensemble or run.");
    if (force.potentials.size() != 1 || force.primary_nep_model_path().empty() ||
        dynamic_cast<NEP_Charge*>(force.potentials[0].get()) == nullptr)
      PRINT_INPUT_ERROR("rpmd_ja generate_raw requires exactly one qNEP potential.");
    auto* active_qnep = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
    if ((active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) || !active_qnep->uses_pppm())
      PRINT_INPUT_ERROR("rpmd_ja generate_raw supports qNEP charge mode 1 or 2 with PPPM only.");
    char* end = nullptr;
    const double temperature = std::strtod(tokens[3].c_str(), &end);
    if (end == tokens[3].c_str() || *end != '\0' || !std::isfinite(temperature) || temperature <= 0.0)
      PRINT_INPUT_ERROR("rpmd_ja raw temperature must be a positive finite number.");
    end = nullptr;
    const double fd_step = std::strtod(tokens[4].c_str(), &end);
    if (end == tokens[4].c_str() || *end != '\0' || !std::isfinite(fd_step) || fd_step <= 0.0)
      PRINT_INPUT_ERROR("rpmd_ja raw finite-difference step must be a positive finite number.");
    generate_rpmd_ja_qnep_raw(tokens[2], temperature, fd_step, tokens[5], atom, box, force);
    return;
  }
  if (tokens[1] == "diagnose") {
    const bool full = tokens.size() == 4 && tokens[3] == "full";
    if (tokens.size() != 3 && !full) PRINT_INPUT_ERROR("rpmd_ja diagnose requires <fd_step> [full].");
    if (integrate.has_ensemble() || global_time != 0.0)
      PRINT_INPUT_ERROR("rpmd_ja diagnose must appear after potential and before any ensemble or run.");
    if (force.potentials.size() != 1 || force.primary_nep_model_path().empty())
      PRINT_INPUT_ERROR("rpmd_ja diagnose requires exactly one qNEP potential.");
    auto* active_qnep = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
    if (active_qnep == nullptr || (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) ||
        !active_qnep->uses_pppm())
      PRINT_INPUT_ERROR("rpmd_ja diagnose supports qNEP charge mode 1 or 2 with PPPM only.");
    char* end = nullptr;
    const double fd_step = std::strtod(tokens[2].c_str(), &end);
    if (end == tokens[2].c_str() || *end != '\0' || !std::isfinite(fd_step) || fd_step <= 0.0)
      PRINT_INPUT_ERROR("rpmd_ja diagnose fd_step must be a positive finite number.");
    diagnose_rpmd_ja_qnep_reference(fd_step, atom, box, force, full);
    return;
  }
  if (tokens[1] == "diagnose_samples") {
    const bool full = tokens.size() == 5 && tokens[4] == "full";
    if (tokens.size() != 4 && !full) PRINT_INPUT_ERROR("rpmd_ja diagnose_samples requires <samples_file> <fd_step> [full].");
    if (integrate.has_ensemble() || global_time != 0.0)
      PRINT_INPUT_ERROR("rpmd_ja diagnose_samples must appear after potential and before any ensemble or run.");
    if (force.potentials.size() != 1 || force.primary_nep_model_path().empty())
      PRINT_INPUT_ERROR("rpmd_ja diagnose_samples requires exactly one qNEP potential.");
    auto* active_qnep = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
    if (active_qnep == nullptr || (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) ||
        !active_qnep->uses_pppm())
      PRINT_INPUT_ERROR("rpmd_ja diagnose_samples supports qNEP charge mode 1 or 2 with PPPM only.");
    if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1)
      PRINT_INPUT_ERROR("rpmd_ja diagnose_samples requires fully periodic boundaries.");
    char* end = nullptr;
    const double fd_step = std::strtod(tokens[3].c_str(), &end);
    if (end == tokens[3].c_str() || *end != '\0' || !std::isfinite(fd_step) || fd_step <= 0.0)
      PRINT_INPUT_ERROR("rpmd_ja diagnose_samples fd_step must be a positive finite number.");
    diagnose_rpmd_ja_native_fit_samples(tokens[2], fd_step, atom, box, force, full);
    return;
  }
  if (tokens[1] == "generate" || tokens[1] == "generate_sparse") {
    const bool sparse = tokens[1] == "generate_sparse";
    if (tokens.size() != (sparse ? 6U : 5U)) {
      PRINT_INPUT_ERROR("rpmd_ja generate requires <file> <T> <fd_step>.");
    }
    if (integrate.has_ensemble() || global_time != 0.0) {
      PRINT_INPUT_ERROR("rpmd_ja generate must appear before any ensemble or run.");
    }
    if (force.potentials.size() != 1 || force.primary_nep_model_path().empty())
      PRINT_INPUT_ERROR("rpmd_ja generate requires exactly one supported NEP or qNEP potential.");
    auto* active_qnep = dynamic_cast<NEP_Charge*>(force.potentials[0].get());
    const bool qnep = active_qnep != nullptr;
    if (qnep && ((active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) ||
                 !active_qnep->uses_pppm())) {
      PRINT_INPUT_ERROR("rpmd_ja qNEP references support only charge mode 1 or 2 with PPPM.");
    }
    if (qnep && !sparse) {
      PRINT_INPUT_ERROR(
        "qNEP references require rpmd_ja generate_sparse with a kernel table; it writes the final v3 reference and stability sidecar.");
    }
    if (!qnep && dynamic_cast<NEP*>(force.potentials[0].get()) == nullptr)
      PRINT_INPUT_ERROR("rpmd_ja generate requires exactly one NEP or qNEP potential.");
    char* end = nullptr;
    const double temperature = std::strtod(tokens[3].c_str(), &end);
    if (end == tokens[3].c_str() || *end != '\0' || !std::isfinite(temperature) || temperature <= 0.0) {
      PRINT_INPUT_ERROR("rpmd_ja reference temperature must be a positive finite number.");
    }
    end = nullptr;
    const double fd_step = std::strtod(tokens[4].c_str(), &end);
    if (end == tokens[4].c_str() || *end != '\0' || !std::isfinite(fd_step) || fd_step <= 0.0) {
      PRINT_INPUT_ERROR("rpmd_ja finite-difference step must be a positive finite number.");
    }
    if (sparse) {
      if (qnep) {
        generate_rpmd_ja_qnep_reference(tokens[2], temperature, fd_step, tokens[5], atom, box, force);
      } else {
        generate_rpmd_ja_sparse_reference(tokens[2], temperature, fd_step, tokens[5], atom, box, force);
      }
    } else {
      generate_rpmd_ja_reference(tokens[2], temperature, fd_step, atom, box, force);
    }
    return;
  }
  if (rpmd_ja_option_seen_) {
    PRINT_INPUT_ERROR("rpmd_ja may be specified only once per run.");
  }
  rpmd_ja_option_seen_ = true;
  if (tokens[1] == "off" && tokens.size() == 2) {
    rpmd_ja_enabled_ = false;
    rpmd_ja_reference_path_.clear();
  } else if (tokens[1] == "on" && tokens.size() == 3) {
    rpmd_ja_enabled_ = true;
    rpmd_ja_reference_path_ = tokens[2];
  } else {
    PRINT_INPUT_ERROR("rpmd_ja expects off or on <referencefile>.");
  }
  measure.set_rpmd_ja(rpmd_ja_enabled_, rpmd_ja_reference_path_);
}
void Run::parse_velocity(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  double initial_temperature;
  int seed = 0;
  bool use_seed = false;
  if (!(num_param == 2 || num_param == 4)) {
    PRINT_INPUT_ERROR("velocity should have 1 or 3 parameters.\n");
  }

  if (!is_valid_real(tokens[1], &initial_temperature)) {
    PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
  }
  if (initial_temperature <= 0.0) {
    PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
  }

  if (num_param == 4) {
    if (tokens[2] != "seed") {
      PRINT_INPUT_ERROR("The second parameter for velocity should be 'seed'.\n");
    }
    use_seed = true;
    if (!is_valid_int(tokens[3], &seed) || seed <= 0) {
      PRINT_INPUT_ERROR("seed should be a positive integer.\n");
    }
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz,
    initial_temperature,
    atom,
    use_seed,
    seed);
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  if (!has_velocity_in_xyz) {
    printf("Initialized velocities with input T = %g K.\n", initial_temperature);
  }
}


void Run::parse_read_pimd_restart(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("read_pimd_restart should have 1 parameter.\n");
  }
  if (!is_pimd(integrate.get_type()) || integrate.get_number_of_beads() < 2) {
    PRINT_INPUT_ERROR("read_pimd_restart should be used after a PIMD-related ensemble keyword.\n");
  }
  if (integrate.pimd_reseed_from_centroid()) {
    PRINT_INPUT_ERROR(
      "read_pimd_restart cannot be combined with pimd_reseed_from_centroid in the same run.");
  }

  PIMD_Restart_Metadata restart_metadata;
  read_pimd_restart(param[1], integrate.get_number_of_beads(), box, atom, &restart_metadata);
  integrate.mark_pimd_restart_read();
  if (restart_metadata.has_temperature) {
    if (integrate.ring_polymer_temperature_is_explicit()) {
      const double temperature_tolerance =
        1.0e-8 * std::max(1.0, std::abs(integrate.get_temperature2()));
      if (std::abs(integrate.get_temperature2() - restart_metadata.temperature) > temperature_tolerance) {
        PRINT_INPUT_ERROR(
          "The explicit RPMD temperature does not match restart_beads.xyz temperature.");
      }
    }
    integrate.restore_pimd_restart_temperature(restart_metadata.temperature);
  } else if (!integrate.ring_polymer_temperature_is_set()) {
    PRINT_INPUT_ERROR(
      "restart_beads.xyz has no temperature. Use ensemble rpmd/trpmd <beads> <temperature>, "
      "or precede read_pimd_restart with an ensemble pimd declaration.");
  }
  has_velocity_in_xyz = 1;

  printf("Read PIMD restart data from %s.\n", param[1]);
  printf("    number of beads = %d.\n", atom.number_of_beads);
}

void Run::parse_pimd_reseed_from_centroid(const char** param, int num_param)
{
  if (num_param != 1) {
    PRINT_INPUT_ERROR("pimd_reseed_from_centroid should have no parameters.\n");
  }
  if (integrate.get_type() != EnsembleType::PIMD || integrate.get_number_of_beads() < 2) {
    PRINT_INPUT_ERROR(
      "pimd_reseed_from_centroid should be used after an ensemble pimd keyword.\n");
  }
  if (integrate.pimd_restart_read_this_run()) {
    PRINT_INPUT_ERROR(
      "pimd_reseed_from_centroid cannot be combined with read_pimd_restart in the same run.\n");
  }
  if (atom.number_of_beads < 2) {
    PRINT_INPUT_ERROR(
      "pimd_reseed_from_centroid requires an already initialized PIMD ring polymer.\n");
  }
  if (!integrate.pimd_previous_run_was_pimd()) {
    PRINT_INPUT_ERROR(
      "pimd_reseed_from_centroid requires the immediately preceding run to be PIMD.");
  }
  if (atom.number_of_beads == integrate.get_number_of_beads()) {
    PRINT_INPUT_ERROR(
      "pimd_reseed_from_centroid requires a different target bead count.\n");
  }
  integrate.arm_pimd_reseed_from_centroid();
  printf("The next PIMD run will reseed all beads from the current centroid.\n");
}

void Run::parse_pimd_bead_gpu_parallel(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_bead_gpu_parallel should have 1 parameter.\n");
  }
  int num_devices = 0;
  if (!is_valid_int(param[1], &num_devices)) {
    PRINT_INPUT_ERROR("number of GPUs for PIMD bead parallel should be an integer.\n");
  }
  if (num_devices < 1) {
    PRINT_INPUT_ERROR("number of GPUs for PIMD bead parallel should >= 1.\n");
  }
  force.set_pimd_bead_gpu_parallel(num_devices);
  if (num_devices == 1) {
    printf("Disabled PIMD bead-to-GPU parallel force evaluation.\n");
  } else {
    printf("Requested PIMD bead-to-GPU parallel force evaluation on %d GPUs.\n", num_devices);
    printf("    PIMD, RPMD, and TRPMD will use this scheduling rule.\n");
  }
}

void Run::parse_pimd_propagator(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_propagator should have 1 parameter.");
  }
  if (strcmp(param[1], "exact") == 0) {
    integrate.set_pimd_use_exact_propagator(true);
    printf("PIMD free ring-polymer propagator is exact.\n");
  } else if (strcmp(param[1], "cayley") == 0) {
    integrate.set_pimd_use_exact_propagator(false);
    printf("PIMD free ring-polymer propagator is Cayley.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_propagator should be exact or cayley.");
  }
}

void Run::parse_pimd_pile_scale(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_pile_scale should have 1 parameter.");
  }
  double pile_scale = 0.0;
  if (!is_valid_real(param[1], &pile_scale)) {
    PRINT_INPUT_ERROR("pimd_pile_scale should be a number.");
  }
  if (pile_scale <= 0.0) {
    PRINT_INPUT_ERROR("pimd_pile_scale should be > 0.");
  }
  integrate.set_pimd_pile_scale(pile_scale);
  printf("PIMD internal-mode Langevin scale is %g.\n", pile_scale);
}

void Run::parse_pimd_fix_com(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_fix_com should have 1 parameter.");
  }
  if (strcmp(param[1], "on") == 0) {
    integrate.set_pimd_fix_com(true);
    printf("PIMD global ring-polymer center-of-mass momentum correction is on.\n");
  } else if (strcmp(param[1], "off") == 0) {
    integrate.set_pimd_fix_com(false);
    printf("PIMD global ring-polymer center-of-mass momentum correction is off.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_fix_com should be on or off.");
  }
}

void Run::parse_pimd_bead_neighbor_rebuild(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_bead_neighbor_rebuild should have 1 parameter.\n");
  }
  if (strcmp(param[1], "auto") == 0) {
    force.set_pimd_bead_neighbor_rebuild(false);
    printf("PIMD bead neighbor lists will rebuild based on the skin distance.\n");
  } else if (strcmp(param[1], "always") == 0) {
    force.set_pimd_bead_neighbor_rebuild(true);
    printf("PIMD bead neighbor lists will rebuild on every force call.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_bead_neighbor_rebuild should be auto or always.\n");
  }
}

void Run::parse_md_qnep_bec(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("md_qnep_bec should have 1 parameter.\n");
  }
  if (strcmp(param[1], "auto") == 0) {
    force.set_md_qnep_bec_mode(0);
    printf("qNEP classical MD BEC mode is auto.\n");
  } else if (strcmp(param[1], "on") == 0) {
    force.set_md_qnep_bec_mode(1);
    printf("qNEP classical MD BEC evaluation is on.\n");
  } else if (strcmp(param[1], "off") == 0) {
    force.set_md_qnep_bec_mode(2);
    printf("qNEP classical MD BEC evaluation is off.\n");
  } else {
    PRINT_INPUT_ERROR("md_qnep_bec should be auto, on or off.\n");
  }
}

void Run::parse_pimd_bead_batch(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_bead_batch should have 1 parameter.\n");
  }
  if (strcmp(param[1], "on") == 0) {
    force.set_pimd_bead_batch(true);
    printf("Requested automatic DP/NEP/qNEP ring-polymer bead-batched kernels.\n");
  } else if (strcmp(param[1], "off") == 0) {
    force.set_pimd_bead_batch(false);
    printf("Disabled automatic DP/NEP/qNEP ring-polymer bead-batched kernels.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_bead_batch should be on or off.\n");
  }
}

void Run::parse_pimd_qnep_bead_batch(const char** param, int num_param)
{
  parse_pimd_bead_batch(param, num_param);
  printf("Warning: pimd_qnep_bead_batch is deprecated; use pimd_bead_batch instead.\n");
}

void Run::parse_pimd_nep_bead_batch(const char** param, int num_param)
{
  parse_pimd_bead_batch(param, num_param);
  printf("Warning: pimd_nep_bead_batch is deprecated; use pimd_bead_batch instead.\n");
}

void Run::parse_pimd_qnep_batch_bec(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_qnep_batch_bec should have 1 parameter.\n");
  }
  if (strcmp(param[1], "auto") == 0) {
    force.set_pimd_qnep_batch_bec_mode(0);
    printf("qNEP PIMD batch BEC mode is auto.\n");
  } else if (strcmp(param[1], "on") == 0) {
    force.set_pimd_qnep_batch_bec_mode(1);
    printf("qNEP PIMD batch BEC evaluation is on.\n");
  } else if (strcmp(param[1], "off") == 0) {
    force.set_pimd_qnep_batch_bec_mode(2);
    printf("qNEP PIMD batch BEC evaluation is off.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_qnep_batch_bec should be auto, on or off.\n");
  }
}

void Run::parse_pimd_nep_batch_profile(const char** param, int num_param)
{
  const bool dp_profile = strcmp(param[0], "pimd_dp_batch_profile") == 0;
  if (num_param != 2) {
    if (dp_profile) {
      PRINT_INPUT_ERROR("pimd_dp_batch_profile should have 1 parameter.\n");
    } else {
      PRINT_INPUT_ERROR("pimd_nep_batch_profile should have 1 parameter.\n");
    }
  }
  if (strcmp(param[1], "on") == 0) {
    if (dp_profile) {
      force.set_pimd_dp_batch_profile(true);
      printf("Enabled DP PIMD batch stage profiling.\n");
    } else {
      force.set_pimd_nep_batch_profile(true);
      printf("Enabled PIMD NEP/qNEP batch stage profiling.\n");
    }
  } else if (strcmp(param[1], "off") == 0) {
    if (dp_profile) {
      force.set_pimd_dp_batch_profile(false);
      printf("Disabled DP PIMD batch stage profiling.\n");
    } else {
      force.set_pimd_nep_batch_profile(false);
      printf("Disabled PIMD NEP/qNEP batch stage profiling.\n");
    }
  } else {
    if (dp_profile) {
      PRINT_INPUT_ERROR("pimd_dp_batch_profile should be on or off.\n");
    } else {
      PRINT_INPUT_ERROR("pimd_nep_batch_profile should be on or off.\n");
    }
  }
}

void Run::parse_pimd_dp_batch_source_count(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_dp_batch_source_count should have 1 parameter.");
  }
  if (strcmp(param[1], "off") == 0) {
    force.set_pimd_dp_batch_source_count_mode(PIMD_DP_Source_Count_Mode::Atomic);
    printf("DP PIMD batch source counting uses the existing atomic path.\n");
  } else if (strcmp(param[1], "check") == 0) {
    force.set_pimd_dp_batch_source_count_mode(PIMD_DP_Source_Count_Mode::Check);
    printf(
      "DP PIMD batch source-count validation is on; it compares every batch call and synchronizes to the host, so do not use this mode for timing.\n");
  } else if (strcmp(param[1], "on") == 0) {
    force.set_pimd_dp_batch_source_count_mode(
      PIMD_DP_Source_Count_Mode::NeighborCounts);
    printf(
      "DP PIMD batch source counts will use NN_local; enable only after source-count check passes for the tested trajectory.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_dp_batch_source_count should be off, check, or on.");
  }
}

void Run::parse_pimd_dp_batch_edge_fill_4_threads(
  const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_dp_batch_edge_fill_4_threads should have 1 parameter.");
  }
  if (strcmp(param[1], "on") == 0) {
    force.set_pimd_dp_batch_edge_fill_4_threads(true);
    printf("DP PIMD batch edge-fill uses 4 threads per atom.\n");
  } else if (strcmp(param[1], "off") == 0) {
    force.set_pimd_dp_batch_edge_fill_4_threads(false);
    printf("DP PIMD batch edge-fill uses 1 thread per atom.\n");
  } else {
    PRINT_INPUT_ERROR("pimd_dp_batch_edge_fill_4_threads should be on or off.");
  }
}



void Run::parse_correct_velocity(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  printf("Correct linear and angular momenta.\n");

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("correct_velocity should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(tokens[1], &velocity.velocity_correction_interval)) {
    PRINT_INPUT_ERROR("velocity correction interval should be an integer.\n");
  }
  if (velocity.velocity_correction_interval < 10) {
    PRINT_INPUT_ERROR("velocity correction interval should >= 10.\n");
  }

  printf("    every %d steps.\n", velocity.velocity_correction_interval);

  if (num_param == 3) {
    if (!is_valid_int(tokens[2], &velocity.velocity_correction_group_method)) {
      PRINT_INPUT_ERROR("velocity correction group method should be an integer.\n");
    }
    if (velocity.velocity_correction_group_method < 0) {
      PRINT_INPUT_ERROR("grouping method should >= 0.\n");
    }
    if (velocity.velocity_correction_group_method >= group.size()) {
      PRINT_INPUT_ERROR("grouping method should < maximum number of grouping methods.\n");
    }
  }

  if (velocity.velocity_correction_group_method < 0) {
    printf("    for the whole system.\n");
  } else {
    printf(
      "    for individual groups in group method %d.\n", velocity.velocity_correction_group_method);
  }

  velocity.do_velocity_correction = true;
}

void Run::parse_time_step(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("time_step should have 1 or 2 parameters.\n");
  }
  if (!is_valid_real(tokens[1], &time_step)) {
    PRINT_INPUT_ERROR("time_step should be a real number.\n");
  }
  printf("Time step for this run is %g fs.\n", time_step);
  time_step /= TIME_UNIT_CONVERSION;
  if (num_param == 3) {
    if (!is_valid_real(tokens[2], &max_distance_per_step)) {
      PRINT_INPUT_ERROR("max distance per step should be a real number.\n");
    }
    if (max_distance_per_step <= 0.0) {
      PRINT_INPUT_ERROR("max distance per step should > 0.\n");
    }
    printf("    max distance per step = %g A.\n", max_distance_per_step);
  }
}

void Run::parse_run(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  int number_of_steps;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("run should have 1 parameter.\n");
  }
  if (!is_valid_int(tokens[1], &number_of_steps)) {
    PRINT_INPUT_ERROR("number of steps should be an integer.\n");
  }
  if (number_of_steps <= 0) {
    PRINT_INPUT_ERROR("number of steps should be positive.\n");
  }
  if (!integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("An ensemble must be specified before each run.");
  }
  printf("Run %d steps.\n", number_of_steps);

  // set target temperature for temperature-dependent NEP
  force.set_temperature_range(
    integrate.get_temperature1(), integrate.get_temperature2(), number_of_steps);


  perform_a_run(number_of_steps);
}

static __global__ void gpu_deform_atom(
  int N,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    g_x[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_y[i] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_z[i] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
  }
}

void Run::parse_change_box(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 4 && num_param != 7) {
    PRINT_INPUT_ERROR("change_box can only have 1 or 3 or 6 parameters\n.");
  }

  double deformation_matrix[3][3] = {0.0};

  if (!is_valid_real(tokens[1], &deformation_matrix[0][0])) {
    PRINT_INPUT_ERROR("box change parameter in xx should be a number.");
  }
  deformation_matrix[1][1] = deformation_matrix[2][2] = deformation_matrix[0][0];

  if (num_param >= 4) {
    if (!is_valid_real(tokens[2], &deformation_matrix[1][1])) {
      PRINT_INPUT_ERROR("box change parameter in yy should be a number.");
    }
    if (!is_valid_real(tokens[3], &deformation_matrix[2][2])) {
      PRINT_INPUT_ERROR("box change parameter in zz should be a number.");
    }
  }

  if (num_param == 7) {
    if (!is_valid_real(tokens[4], &deformation_matrix[1][2])) {
      PRINT_INPUT_ERROR("box change parameter in yz should be a number.");
    }
    if (!is_valid_real(tokens[5], &deformation_matrix[0][2])) {
      PRINT_INPUT_ERROR("box change parameter in xz should be a number.");
    }
    if (!is_valid_real(tokens[6], &deformation_matrix[0][1])) {
      PRINT_INPUT_ERROR("box change parameter in xy should be a number.");
    }
    deformation_matrix[1][0] = deformation_matrix[0][1];
    deformation_matrix[2][0] = deformation_matrix[0][2];
    deformation_matrix[2][1] = deformation_matrix[1][2];
  }

  printf("Change box:\n");
  printf("    in xx by %g A.\n", deformation_matrix[0][0]);
  printf("    in yy by %g A.\n", deformation_matrix[1][1]);
  printf("    in zz by %g A.\n", deformation_matrix[2][2]);
  printf("    in yz and zy by strain %g.\n", deformation_matrix[1][2]);
  printf("    in xz and zx by strain %g.\n", deformation_matrix[0][2]);
  printf("    in xy and yz by strain %g.\n", deformation_matrix[0][1]);

  for (int d = 0; d < 3; ++d) {
    deformation_matrix[d][d] =
      (box.cpu_h[d * 3 + d] + deformation_matrix[d][d]) / box.cpu_h[d * 3 + d];
  }

  printf("    Deformation matrix =\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", deformation_matrix[d1][d2]);
    }
    printf("\n");
  }

  printf("    Original box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }

  double h_old[9];
  for (int i = 0; i < 9; ++i) {
    h_old[i] = box.cpu_h[i];
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < 3; ++k) {
        tmp += deformation_matrix[r][k] * h_old[k * 3 + c];
      }
      box.cpu_h[r * 3 + c] = tmp;
    }
  }
  box.get_inverse();

  const int number_of_atoms = atom.position_per_atom.size() / 3;
  gpu_deform_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    deformation_matrix[0][0],
    deformation_matrix[0][1],
    deformation_matrix[0][2],
    deformation_matrix[1][0],
    deformation_matrix[1][1],
    deformation_matrix[1][2],
    deformation_matrix[2][0],
    deformation_matrix[2][1],
    deformation_matrix[2][2],
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL

  printf("    Changed box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }
}

void Run::parse_pimd_nep_batch_geometry_cache(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pimd_nep_batch_geometry_cache should have 1 parameter.\n");
  }
  if (strcmp(param[1], "on") != 0 && strcmp(param[1], "off") != 0) {
    PRINT_INPUT_ERROR("pimd_nep_batch_geometry_cache should be on or off.\n");
  }
  printf(
    "Ignored pimd_nep_batch_geometry_cache; using the baseline compact-neighbor path.\n");
}

void Run::parse_pppm_mesh_spacing(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("pppm_mesh_spacing should have 1 parameter.\n");
  }
  const double spacing = get_double_from_token(param[1], __FILE__, __LINE__);
  if (!(spacing > 0.0)) {
    PRINT_INPUT_ERROR("pppm_mesh_spacing should be greater than zero.\n");
  }
  force.set_pppm_mesh_spacing(spacing);
  printf("PPPM mesh spacing = %g Angstrom.\n", spacing);
}
