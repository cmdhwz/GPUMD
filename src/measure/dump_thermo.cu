/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
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

Dump_Thermo::Dump_Thermo(const std::vector<std::string>& tokens)
{
  parse(tokens);
  action_name = "dump_thermo";
}

void Dump_Thermo::parse(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("dump_thermo should have 1 or 2 parameters.");
  }
  if (!is_valid_int(tokens[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("thermo dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("thermo dump interval should > 0.");
  }
  if (num_param == 3) {
    if (tokens[2] != "rp_energy") {
      PRINT_INPUT_ERROR("Unknown dump_thermo option.");
    }
    rp_energy_ = true;
  }
  printf("Dump thermo every %d steps.\n", dump_interval_);
}

void write_thermo_row(
  FILE* fid,
  const double* thermo,
  const bool thermo_holds_kinetic_energy,
  const double temperature_target,
  const int number_of_atoms_for_temperature,
  const Box& box,
  const double* extra_values)
{
  double energy_kin, temperature;
  if (thermo_holds_kinetic_energy) {
    energy_kin = thermo[0];
    temperature = temperature_target;
  } else {
    energy_kin = 1.5 * number_of_atoms_for_temperature * K_B * thermo[0];
    temperature = thermo[0];
  }

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(
    fid,
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
    fid,
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
  if (extra_values != nullptr) {
    fprintf(
      fid,
      "%25.17e%25.17e%25.17e%25.17e%25.17e%25.17e%25.17e",
      extra_values[0],
      extra_values[1],
      extra_values[2],
      extra_values[3],
      extra_values[4],
      extra_values[5],
      extra_values[6]);
  }
  fprintf(fid, "\n");
  fflush(fid);
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
    const EnsembleType type = integrate.get_type();
    if (type != EnsembleType::RPMD && type != EnsembleType::PIMD) {
      PRINT_INPUT_ERROR(
        "dump_thermo rp_energy supports fixed-cell RPMD and NVT-PIMD only.");
    }
    rp_ensemble_ = dynamic_cast<Ensemble_PIMD*>(integrate.get_ensemble());
    if (rp_ensemble_ == nullptr) {
      PRINT_INPUT_ERROR("dump_thermo rp_energy requires an initialized ring-polymer ensemble.");
    }
    if (
      (type == EnsembleType::PIMD &&
       (integrate.get_temperature1() != integrate.get_temperature2() ||
        integrate.get_num_target_pressure_components() != 0)) ||
      integrate.get_deform_x() != 0 || integrate.get_deform_y() != 0 ||
      integrate.get_deform_z() != 0 || integrate.get_deform_xy() != 0 ||
      integrate.get_deform_xz() != 0 || integrate.get_deform_yz() != 0) {
      PRINT_INPUT_ERROR(
        "dump_thermo rp_energy supports fixed-cell RPMD and fixed-temperature NVT-PIMD only.");
    }
    if (type == EnsembleType::PIMD) {
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
    if (integrate.get_type() == EnsembleType::RPMD) {
      fprintf(fid_, "# conserved_quantity H_rp_avg # RPMD\n");
    } else {
      fprintf(fid_, "# conserved_quantity H_conserved_avg # NVT-PIMD\n");
    }
  }
  fprintf(
    fid_,
    "# columns %s PE sxx syy szz syz sxz sxy ax ay az bx by bz cx cy cz%s\n",
    is_pimd(integrate.get_type()) ? "T_target KE_quantum" : "T KE",
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

  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);

  double ring_polymer_values[7];
  const double* extra_values = nullptr;
  if (rp_energy_) {
    double ke_rp, e_spring, w_nonham;
    rp_ensemble_->get_ring_polymer_energy(atom.mass, ke_rp, e_spring, w_nonham);
    const double pe_rp = thermo[1];
    const double h_rp = ke_rp + pe_rp + e_spring;
    const double h_conserved = h_rp - w_nonham;
    if (!conserved_reference_set_) {
      h_conserved_reference_ = h_conserved;
      conserved_reference_set_ = true;
    }
    const double d_h_conserved =
      1000.0 * (h_conserved - h_conserved_reference_) / atom.number_of_atoms;
    ring_polymer_values[0] = ke_rp;
    ring_polymer_values[1] = pe_rp;
    ring_polymer_values[2] = e_spring;
    ring_polymer_values[3] = h_rp;
    ring_polymer_values[4] = w_nonham;
    ring_polymer_values[5] = h_conserved;
    ring_polymer_values[6] = d_h_conserved;
    extra_values = ring_polymer_values;
  }

  write_thermo_row(
    fid_,
    thermo,
    is_pimd(integrate.get_type()),
    integrate.get_target_temperature_of_step(),
    integrate.get_number_of_atoms_for_temperature(atom.number_of_atoms, group),
    box,
    extra_values);
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
