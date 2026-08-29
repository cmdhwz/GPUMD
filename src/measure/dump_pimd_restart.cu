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
Dump a single-file restart container for centroid and all beads in PIMD-related runs
--------------------------------------------------------------------------------------------------*/

#include "dump_pimd_restart.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

namespace
{
bool is_valid_backup_directory_name(const char* directory)
{
  if (directory[0] == '\0' || strcmp(directory, ".") == 0 || strcmp(directory, "..") == 0) {
    return false;
  }
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(directory); *p != '\0'; ++p) {
    if (*p < 32 || *p == '/' || *p == '\\' || *p == ':') {
      return false;
    }
  }
  return true;
}

bool make_restart_backup_directory(const std::string& directory)
{
#ifdef _WIN32
  return _mkdir(directory.c_str()) == 0 || errno == EEXIST;
#else
  return mkdir(directory.c_str(), 0777) == 0 || errno == EEXIST;
#endif
}

void copy_restart_file(const std::string& source, const std::string& destination)
{
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(destination, std::ios::binary | std::ios::trunc);
  if (!input || !output) {
    PRINT_INPUT_ERROR("Failed to copy the PIMD restart backup to restart_beads.xyz.");
  }
  output << input.rdbuf();
  if (!output) {
    PRINT_INPUT_ERROR("Failed while copying the PIMD restart backup.");
  }
}
} // namespace

Dump_PIMD_Restart::Dump_PIMD_Restart(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "dump_pimd_restart";
}

void Dump_PIMD_Restart::parse(const char** param, int num_param)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("dump_pimd_restart should have 1 or 2 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("PIMD restart dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("PIMD restart dump interval should > 0.");
  }
  if (num_param == 3) {
    if (!is_valid_backup_directory_name(param[2])) {
      PRINT_INPUT_ERROR(
        "The optional backup directory name should be a single directory name without '/', '\\\\', or ':'.");
    }
    backup_ = true;
    backup_directory_ = param[2];
  }
  dump_ = true;
  printf("Dump PIMD restart every %d steps.\n", dump_interval_);
  if (backup_) {
    printf("    Save backup files in %s.\n", backup_directory_.c_str());
  }
}

void Dump_PIMD_Restart::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!dump_) {
    return;
  }
  if (atom.number_of_beads == 0) {
    PRINT_INPUT_ERROR("Cannot use dump_pimd_restart for non-PIMD-related runs.");
  }
  if (backup_ && !make_restart_backup_directory(backup_directory_)) {
    PRINT_INPUT_ERROR("Failed to create the PIMD restart backup directory.");
  }
  cpu_position_.resize(atom.number_of_atoms * 3);
  cpu_velocity_.resize(atom.number_of_atoms * 3);
}

void Dump_PIMD_Restart::output_line_2(
  FILE* fid, const Box& box, int number_of_beads, int bead_index, double temperature)
{
  fprintf(
    fid,
    "pimd_restart=1 num_beads=%d bead=%d role=%s temperature=%.17g ",
    number_of_beads,
    bead_index,
    bead_index < 0 ? "centroid" : "bead",
    temperature);

  fprintf(
    fid, "pbc=\"%c %c %c\" ", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  fprintf(
    fid,
    "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" ",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);

  fprintf(fid, "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n");
}

void Dump_PIMD_Restart::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if (!dump_) {
    return;
  }
  if ((step + 1) % dump_interval_ != 0) {
    return;
  }

  std::string filename = "restart_beads.xyz";
  if (backup_) {
    std::ostringstream backup_filename;
    backup_filename << backup_directory_ << "/restart_beads_step_" << std::setw(10)
                    << std::setfill('0') << (step + 1) << ".xyz";
    filename = backup_filename.str();
  }

  FILE* fid = my_fopen(filename.c_str(), "w");
  const int number_of_atoms = atom.number_of_atoms;
  const int number_of_beads = atom.number_of_beads;
  const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
  const double restart_temperature =
    integrate.ensemble != nullptr ? integrate.ensemble->temperature : temperature;

  // The centroid frame is written first so restart readers can reuse the averaged state directly.
  atom.position_per_atom.copy_to_host(cpu_position_.data());
  atom.velocity_per_atom.copy_to_host(cpu_velocity_.data());
  fprintf(fid, "%d\n", number_of_atoms);
  output_line_2(fid, box, number_of_beads, -1, restart_temperature);
  for (int n = 0; n < number_of_atoms; ++n) {
    fprintf(
      fid,
      "%s %.17g %.17g %.17g %.8f %.17g %.17g %.17g\n",
      atom.cpu_atom_symbol[n].c_str(),
      cpu_position_[n],
      cpu_position_[n + number_of_atoms],
      cpu_position_[n + 2 * number_of_atoms],
      atom.cpu_mass[n],
      cpu_velocity_[n] * natural_to_A_per_fs,
      cpu_velocity_[n + number_of_atoms] * natural_to_A_per_fs,
      cpu_velocity_[n + 2 * number_of_atoms] * natural_to_A_per_fs);
  }

  for (int k = 0; k < number_of_beads; ++k) {
    atom.position_beads[k].copy_to_host(cpu_position_.data());
    atom.velocity_beads[k].copy_to_host(cpu_velocity_.data());
    fprintf(fid, "%d\n", number_of_atoms);
    output_line_2(fid, box, number_of_beads, k, restart_temperature);
    for (int n = 0; n < number_of_atoms; ++n) {
      fprintf(
        fid,
        "%s %.17g %.17g %.17g %.8f %.17g %.17g %.17g\n",
        atom.cpu_atom_symbol[n].c_str(),
        cpu_position_[n],
        cpu_position_[n + number_of_atoms],
        cpu_position_[n + 2 * number_of_atoms],
        atom.cpu_mass[n],
        cpu_velocity_[n] * natural_to_A_per_fs,
        cpu_velocity_[n + number_of_atoms] * natural_to_A_per_fs,
        cpu_velocity_[n + 2 * number_of_atoms] * natural_to_A_per_fs);
    }
  }

  fflush(fid);
  fclose(fid);

  if (backup_) {
    copy_restart_file(filename, "restart_beads.xyz");
  }
}

void Dump_PIMD_Restart::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (dump_) {
    dump_ = false;
  }
}
