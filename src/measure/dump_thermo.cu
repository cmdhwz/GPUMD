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

/*-----------------------------------------------------------------------------------------------100
Dump thermo data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_thermo.cuh"
#include "integrate/ensemble_pimd.cuh"
#include "integrate/integrate.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

Dump_Thermo::Dump_Thermo(const char** param, int num_param) 
{
  parse(param, num_param);
  action_name = "dump_thermo";
}

void Dump_Thermo::parse(const char** param, int num_param)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("dump_thermo should have 1 or 2 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("thermo dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("thermo dump interval should > 0.");
  }
  if (num_param == 3) {
    if (strcmp(param[2], "rp_energy") != 0) {
      PRINT_INPUT_ERROR("Unknown dump_thermo option.");
    }
    rp_energy_ = true;
  }
  printf("Dump thermo every %d steps.\n", dump_interval_);
}

void Dump_Thermo::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  conserved_reference_set_ = false;
  rp_ensemble_ = nullptr;
  if (rp_energy_) {
    if (integrate.type != 31 && integrate.type != 33) {
      PRINT_INPUT_ERROR(
        "dump_thermo rp_energy supports fixed-cell RPMD and NVT-PIMD only.");
    }
    rp_ensemble_ = dynamic_cast<Ensemble_PIMD*>(integrate.ensemble.get());
    if (rp_ensemble_ == nullptr) {
      PRINT_INPUT_ERROR("dump_thermo rp_energy requires an initialized ring-polymer ensemble.");
    }
    if (
      (integrate.type == 33 &&
       (integrate.temperature1 != integrate.temperature2 ||
        integrate.num_target_pressure_components != 0)) ||
      integrate.deform_x != 0 || integrate.deform_y != 0 || integrate.deform_z != 0 ||
      integrate.deform_xy != 0 || integrate.deform_xz != 0 || integrate.deform_yz != 0) {
      PRINT_INPUT_ERROR(
        "dump_thermo rp_energy supports fixed-cell RPMD and fixed-temperature NVT-PIMD only.");
    }
    if (integrate.type == 33) {
      rp_ensemble_->reset_nonham_work();
    }
  }

  fid_ = my_fopen("thermo.out", "a");
  fprintf(fid_, "# dump_thermo %d\n", dump_interval_);
  fprintf(fid_, "# format_version %d\n", rp_energy_ ? 3 : 1);
  fprintf(fid_, "# num_atoms %d\n", atom.number_of_atoms);
  fprintf(fid_, "# dt_output %.10e fs\n", time_step * dump_interval_ * TIME_UNIT_CONVERSION);
  if (rp_energy_) {
    fprintf(fid_, "# rp_energy_normalization per_bead\n");
    fprintf(fid_, "# nonham_work_sign positive_into_ring_polymer\n");
    fprintf(fid_, "# conserved_energy_reference first_dumped_frame\n");
    if (integrate.type == 31) {
      fprintf(fid_, "# conserved_quantity H_rp_avg # RPMD\n");
    } else {
      fprintf(fid_, "# conserved_quantity H_conserved_avg # NVT-PIMD\n");
    }
  }
  fprintf(
    fid_,
    "# columns %s PE sxx syy szz syz sxz sxy ax ay az bx by bz cx cy cz%s\n",
    integrate.type >= 31 ? "T_target KE_quantum" : "T KE",
    rp_energy_
      ? " KE_rp_avg PE_rp_avg E_spring_avg H_rp_avg W_nonham_avg H_conserved_avg dH_conserved_meV_per_atom"
      : "");
}

void Dump_Thermo::end_of_step(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature_target,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& gpu_thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  int number_of_atoms_fixed =
    (fixed_group < 0) ? 0 : group[integrate.fixed_grouping_method].cpu_size[fixed_group];

  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);
  double energy_kin, temperature;
  if (integrate.type >= 31) {
    energy_kin = thermo[0];
    temperature = temperature_target;
  } else {
    const int number_of_atoms_moving = atom.number_of_atoms - number_of_atoms_fixed;
    energy_kin = 1.5 * number_of_atoms_moving * K_B * thermo[0];
    temperature = thermo[0];
  }

  double ke_rp = 0.0;
  double pe_rp = 0.0;
  double e_spring = 0.0;
  double h_rp = 0.0;
  double w_nonham = 0.0;
  double h_conserved = 0.0;
  double d_h_conserved = 0.0;
  if (rp_energy_) {
    rp_ensemble_->get_ring_polymer_energy(ke_rp, e_spring, w_nonham);
    pe_rp = thermo[1];
    h_rp = ke_rp + pe_rp + e_spring;
    h_conserved = h_rp - w_nonham;
    if (!conserved_reference_set_) {
      h_conserved_reference_ = h_conserved;
      conserved_reference_set_ = true;
    }
    d_h_conserved =
      1000.0 * (h_conserved - h_conserved_reference_) / atom.number_of_atoms;
  }

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(
    fid_,
    "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e",
    temperature,
    energy_kin,
    thermo[1],
    thermo[2] * PRESSURE_UNIT_CONVERSION,
    thermo[3] * PRESSURE_UNIT_CONVERSION,
    thermo[4] * PRESSURE_UNIT_CONVERSION,
    thermo[7] * PRESSURE_UNIT_CONVERSION,
    thermo[6] * PRESSURE_UNIT_CONVERSION,
    thermo[5] * PRESSURE_UNIT_CONVERSION);

  fprintf(
    fid_,
    "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);
  if (rp_energy_) {
    fprintf(
      fid_,
      "%25.17e%25.17e%25.17e%25.17e%25.17e%25.17e%25.17e",
      ke_rp,
      pe_rp,
      e_spring,
      h_rp,
      w_nonham,
      h_conserved,
      d_h_conserved);
  }
  fprintf(fid_, "\n");
  fflush(fid_);
}

void Dump_Thermo::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(fid_);
}
