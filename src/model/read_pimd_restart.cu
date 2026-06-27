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
Read a single-file PIMD restart container with centroid and all beads
--------------------------------------------------------------------------------------------------*/

#include "read_pimd_restart.cuh"
#include "atom.cuh"
#include "box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace
{
void read_frame_atom_data(
  std::ifstream& input,
  const int number_of_atoms,
  const int num_columns,
  const int* property_offset,
  const std::vector<std::string>& atom_symbols,
  const std::vector<double>& reference_mass,
  std::vector<double>& position,
  std::vector<double>& velocity)
{
  const double A_per_fs_to_natural = TIME_UNIT_CONVERSION;
  // Allow modest formatting/rounding differences between model.xyz and restart_beads.xyz
  // while still rejecting physically different mass assignments.
  const double mass_tolerance_absolute = 5.0e-4;
  const double mass_tolerance_relative = 1.0e-6;
  for (int n = 0; n < number_of_atoms; ++n) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != num_columns) {
      PRINT_INPUT_ERROR("number of columns in restart_beads.xyz does not match properties.\n");
    }
    if (tokens[property_offset[0]] != atom_symbols[n]) {
      PRINT_INPUT_ERROR("species ordering in restart_beads.xyz does not match model.xyz.\n");
    }
    double mass = get_double_from_token(tokens[property_offset[2]], __FILE__, __LINE__);
    const double mass_difference = std::abs(mass - reference_mass[n]);
    const double mass_tolerance =
      std::max(mass_tolerance_absolute, mass_tolerance_relative * std::max(1.0, reference_mass[n]));
    if (mass_difference > mass_tolerance) {
      PRINT_INPUT_ERROR("mass values in restart_beads.xyz do not match model.xyz.\n");
    }
    for (int d = 0; d < 3; ++d) {
      position[n + number_of_atoms * d] =
        get_double_from_token(tokens[property_offset[1] + d], __FILE__, __LINE__);
      velocity[n + number_of_atoms * d] =
        get_double_from_token(tokens[property_offset[3] + d], __FILE__, __LINE__) *
        A_per_fs_to_natural;
    }
  }
}

void read_pimd_restart_line_2(
  std::ifstream& input,
  const int expected_number_of_beads,
  bool& read_box,
  Box& box,
  int& bead_index,
  std::string& role,
  int& num_columns,
  int* property_offset)
{
  std::vector<std::string> tokens = get_tokens_without_unwanted_spaces(input);
  for (auto& token : tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }

  bool has_restart_flag = false;
  bool has_num_beads = false;
  bool has_bead_index = false;
  bool has_role = false;
  bool has_lattice = false;
  bool has_properties = false;
  num_columns = 0;
  property_offset[0] = 0;
  property_offset[1] = 0;
  property_offset[2] = 0;
  property_offset[3] = 0;
  int property_position[4] = {-1, -1, -1, -1};
  std::string property_name[4] = {"species", "pos", "mass", "vel"};

  for (int n = 0; n < int(tokens.size()); ++n) {
    if (tokens[n].substr(0, 13) == "pimd_restart=") {
      int restart_flag =
        get_int_from_token(tokens[n].substr(13, tokens[n].length() - 13), __FILE__, __LINE__);
      if (restart_flag != 1) {
        PRINT_INPUT_ERROR("pimd_restart flag in restart_beads.xyz should be 1.\n");
      }
      has_restart_flag = true;
    } else if (tokens[n].substr(0, 10) == "num_beads=") {
      int num_beads =
        get_int_from_token(tokens[n].substr(10, tokens[n].length() - 10), __FILE__, __LINE__);
      if (num_beads != expected_number_of_beads) {
        PRINT_INPUT_ERROR("num_beads in restart_beads.xyz does not match the ensemble setting.\n");
      }
      has_num_beads = true;
    } else if (tokens[n].substr(0, 5) == "bead=") {
      bead_index =
        get_int_from_token(tokens[n].substr(5, tokens[n].length() - 5), __FILE__, __LINE__);
      has_bead_index = true;
    } else if (tokens[n].substr(0, 5) == "role=") {
      role = tokens[n].substr(5, tokens[n].length() - 5);
      has_role = true;
    } else if (tokens[n].substr(0, 4) == "pbc=") {
      bool pbc[3] = {false, false, false};
      if (tokens[n].back() == 't') {
        pbc[0] = true;
      } else if (tokens[n].back() == 'f') {
        pbc[0] = false;
      } else {
        PRINT_INPUT_ERROR("periodic boundary in x direction should be T or F.");
      }
      if (tokens[n + 1] == "t") {
        pbc[1] = true;
      } else if (tokens[n + 1] == "f") {
        pbc[1] = false;
      } else {
        PRINT_INPUT_ERROR("periodic boundary in y direction should be T or F.");
      }
      if (tokens[n + 2].front() == 't') {
        pbc[2] = true;
      } else if (tokens[n + 2].front() == 'f') {
        pbc[2] = false;
      } else {
        PRINT_INPUT_ERROR("periodic boundary in z direction should be T or F.");
      }
      if (!read_box) {
        box.pbc_x = pbc[0];
        box.pbc_y = pbc[1];
        box.pbc_z = pbc[2];
      }
    } else if (tokens[n].substr(0, 8) == "lattice=") {
      const int transpose_index[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
      if (!read_box) {
        for (int m = 0; m < 9; ++m) {
          box.cpu_h[transpose_index[m]] = get_double_from_token(
            tokens[n + m].substr(
              (m == 0) ? 9 : 0,
              (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
            __FILE__,
            __LINE__);
        }
      }
      has_lattice = true;
    } else if (tokens[n].substr(0, 11) == "properties=") {
      std::string line = tokens[n].substr(11, tokens[n].length() - 11);
      for (auto& letter : line) {
        if (letter == ':') {
          letter = ' ';
        }
      }
      std::vector<std::string> sub_tokens = get_tokens(line);
      for (int k = 0; k < int(sub_tokens.size()) / 3; ++k) {
        for (int prop = 0; prop < 4; ++prop) {
          if (sub_tokens[k * 3] == property_name[prop]) {
            property_position[prop] = k;
          }
        }
      }
      for (int k = 0; k < int(sub_tokens.size()) / 3; ++k) {
        const int tmp_length = get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
        for (int prop = 0; prop < 4; ++prop) {
          if (k < property_position[prop]) {
            property_offset[prop] += tmp_length;
          }
        }
        num_columns += tmp_length;
      }
      if (property_position[0] < 0 || property_position[1] < 0 || property_position[2] < 0 ||
          property_position[3] < 0) {
        PRINT_INPUT_ERROR("restart_beads.xyz should contain species, pos, mass, and vel.\n");
      }
      has_properties = true;
    }
  }

  if (!has_restart_flag || !has_num_beads || !has_bead_index || !has_role || !has_lattice ||
      !has_properties) {
    PRINT_INPUT_ERROR("The header line in restart_beads.xyz is incomplete.\n");
  }

  if (!read_box) {
    box.get_inverse();
    read_box = true;
  }
}
} // namespace

void read_pimd_restart(const char* filename, int expected_number_of_beads, Box& box, Atom& atom)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open the PIMD restart file.");
  }
  if (expected_number_of_beads < 2) {
    PRINT_INPUT_ERROR("The number of beads for PIMD restart should >= 2.");
  }

  const int number_of_atoms = atom.number_of_atoms;
  std::vector<double> position(number_of_atoms * 3);
  std::vector<double> velocity(number_of_atoms * 3);
  std::vector<int> bead_frame_seen(expected_number_of_beads, 0);
  bool read_box = false;
  bool centroid_frame_seen = false;

  atom.position_beads.resize(expected_number_of_beads);
  atom.velocity_beads.resize(expected_number_of_beads);
  atom.force_beads.resize(expected_number_of_beads);
  atom.potential_beads.resize(expected_number_of_beads);
  atom.virial_beads.resize(expected_number_of_beads);
  for (int k = 0; k < expected_number_of_beads; ++k) {
    atom.position_beads[k].resize(number_of_atoms * 3);
    atom.velocity_beads[k].resize(number_of_atoms * 3);
    atom.force_beads[k].resize(number_of_atoms * 3);
    atom.potential_beads[k].resize(number_of_atoms);
    atom.virial_beads[k].resize(number_of_atoms * 9);
  }

  int frame_count = 0;
  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() == 0) {
      continue;
    }
    if (tokens.size() != 1) {
      PRINT_INPUT_ERROR("Each frame in restart_beads.xyz should start with the number of atoms.\n");
    }
    int num_atoms_in_frame = get_int_from_token(tokens[0], __FILE__, __LINE__);
    if (num_atoms_in_frame != number_of_atoms) {
      PRINT_INPUT_ERROR("The number of atoms in restart_beads.xyz does not match model.xyz.\n");
    }

    int bead_index = -2;
    std::string role;
    int property_offset[4] = {0, 0, 0, 0};
    int num_columns = 0;
    read_pimd_restart_line_2(
      input,
      expected_number_of_beads,
      read_box,
      box,
      bead_index,
      role,
      num_columns,
      property_offset);
    read_frame_atom_data(
      input,
      number_of_atoms,
      num_columns,
      property_offset,
      atom.cpu_atom_symbol,
      atom.cpu_mass,
      position,
      velocity);

    if (role == "centroid") {
      if (bead_index != -1) {
        PRINT_INPUT_ERROR("The centroid frame in restart_beads.xyz should have bead=-1.\n");
      }
      if (centroid_frame_seen) {
        PRINT_INPUT_ERROR("restart_beads.xyz should contain exactly one centroid frame.\n");
      }
      atom.cpu_position_per_atom.assign(position.begin(), position.end());
      atom.cpu_velocity_per_atom.assign(velocity.begin(), velocity.end());
      centroid_frame_seen = true;
    } else if (role == "bead") {
      if (bead_index < 0 || bead_index >= expected_number_of_beads) {
        PRINT_INPUT_ERROR("The bead index in restart_beads.xyz is out of range.\n");
      }
      if (bead_frame_seen[bead_index] == 1) {
        PRINT_INPUT_ERROR("restart_beads.xyz contains duplicated bead frames.\n");
      }
      atom.position_beads[bead_index].copy_from_host(position.data());
      atom.velocity_beads[bead_index].copy_from_host(velocity.data());
      bead_frame_seen[bead_index] = 1;
    } else {
      PRINT_INPUT_ERROR("The role field in restart_beads.xyz should be centroid or bead.\n");
    }
    ++frame_count;
  }

  if (frame_count != expected_number_of_beads + 1) {
    PRINT_INPUT_ERROR("The number of frames in restart_beads.xyz should equal num_beads + 1.\n");
  }
  if (!centroid_frame_seen) {
    PRINT_INPUT_ERROR("restart_beads.xyz does not contain a centroid frame.\n");
  }
  for (int k = 0; k < expected_number_of_beads; ++k) {
    if (bead_frame_seen[k] == 0) {
      PRINT_INPUT_ERROR("restart_beads.xyz is missing one or more bead frames.\n");
    }
  }

  atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  atom.number_of_beads = expected_number_of_beads;
}
