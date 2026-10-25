/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "interface_geometry.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
bool parse_int_token(const std::string& token, int& value)
{
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(token.c_str(), &end, 10);
  if (errno || end == token.c_str() || *end != '\0' ||
      parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())
    return false;
  value = static_cast<int>(parsed);
  return true;
}

bool parse_real_token(const std::string& token, double& value)
{
  char* end = nullptr;
  errno = 0;
  value = std::strtod(token.c_str(), &end);
  return !errno && end != token.c_str() && *end == '\0' && std::isfinite(value);
}

bool output_exists(const std::string& path)
{
  FILE* file = std::fopen(path.c_str(), "r");
  if (!file) return false;
  std::fclose(file);
  return true;
}

int pbc_flag(const Box& box, const int d)
{
  if (d == 0) return box.pbc_x;
  if (d == 1) return box.pbc_y;
  return box.pbc_z;
}

const int* group_labels(const std::vector<Group>& groups, const int method, const int atom_count)
{
  if (method < 0 || method >= static_cast<int>(groups.size()) ||
      groups[method].cpu_label.size() != static_cast<size_t>(atom_count)) {
    PRINT_INPUT_ERROR("Interface group method is invalid for the current atom set.\n");
  }
  return groups[method].cpu_label.data();
}
} // namespace

InterfaceGeometry::InterfaceGeometry(
  const std::string& config_file, const std::string& output_prefix)
  : config_file_(config_file), output_prefix_(output_prefix)
{
  if (config_file_.empty() || output_prefix_.empty()) {
    PRINT_INPUT_ERROR("interface_geometry requires a config file and output prefix.\n");
  }
  parse_config();
}

void InterfaceGeometry::parse_config()
{
  std::ifstream input(config_file_);
  if (!input) PRINT_INPUT_ERROR("Cannot open interface geometry config file.\n");

  bool axis_seen = false;
  bool width_seen = false;
  bool interfaces_seen = false;
  bool material_seen = false;
  bool bath_seen = false;
  std::string line;
  while (std::getline(input, line)) {
    const size_t comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    std::istringstream parser(line);
    std::vector<std::string> fields;
    std::string field;
    while (parser >> field) fields.emplace_back(field);
    if (fields.empty()) continue;

    if (fields[0] == "axis") {
      if (axis_seen || fields.size() != 2) PRINT_INPUT_ERROR("Invalid or duplicate axis config key.\n");
      if (fields[1] == "x") axis_ = 0;
      else if (fields[1] == "y") axis_ = 1;
      else if (fields[1] == "z") axis_ = 2;
      else PRINT_INPUT_ERROR("interface_geometry axis must be x, y, or z.\n");
      axis_seen = true;
    } else if (fields[0] == "bin_width_A") {
      if (width_seen || fields.size() != 2 || !parse_real_token(fields[1], bin_width_A_) || bin_width_A_ <= 0.0)
        PRINT_INPUT_ERROR("Invalid or duplicate bin_width_A config key.\n");
      width_seen = true;
    } else if (fields[0] == "interfaces") {
      int count = 0;
      if (interfaces_seen || fields.size() < 2 || !parse_int_token(fields[1], count) || count < 0 ||
          fields.size() != static_cast<size_t>(count) + 2) {
        PRINT_INPUT_ERROR("Invalid or duplicate interfaces config key.\n");
      }
      for (int i = 0; i < count; ++i) {
        double coordinate = 0.0;
        if (!parse_real_token(fields[i + 2], coordinate)) {
          PRINT_INPUT_ERROR("Interface coordinates must be finite numbers.\n");
        }
        interfaces_A_.emplace_back(coordinate);
      }
      interfaces_seen = true;
    } else if (fields[0] == "material_group") {
      if (material_seen || fields.size() != 2 || !parse_int_token(fields[1], material_method_))
        PRINT_INPUT_ERROR("Invalid or duplicate material_group config key.\n");
      material_seen = true;
    } else if (fields[0] == "bath_groups") {
      if (bath_seen || fields.size() != 4 || !parse_int_token(fields[1], bath_method_) ||
          !parse_int_token(fields[2], source_group_) || !parse_int_token(fields[3], sink_group_))
        PRINT_INPUT_ERROR("Invalid or duplicate bath_groups config key.\n");
      bath_seen = true;
    } else if (fields[0] == "bin_group") {
      if (has_bin_method_ || fields.size() != 2 || !parse_int_token(fields[1], bin_method_))
        PRINT_INPUT_ERROR("Invalid or duplicate bin_group config key.\n");
      has_bin_method_ = true;
    } else {
      PRINT_INPUT_ERROR("Unknown interface geometry config key.\n");
    }
  }
  if (!input.eof() || !axis_seen || !width_seen || !interfaces_seen || !material_seen || !bath_seen) {
    PRINT_INPUT_ERROR("Incomplete or unreadable interface geometry config file.\n");
  }
}

void InterfaceGeometry::initialize(
  Atom& atom,
  const Box& box,
  const std::vector<Group>& groups,
  const Integrate& integrate)
{
  if (initialized_) PRINT_INPUT_ERROR("Interface geometry was initialized more than once in one run.\n");
  if (!box.is_orthogonal) PRINT_INPUT_ERROR("Interface geometry requires an orthogonal box.\n");

  const int N = atom.number_of_atoms;
  const double lengths[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(box.cpu_h[i])) PRINT_INPUT_ERROR("Interface geometry box must be finite.\n");
  }
  for (int d = 0; d < 3; ++d) {
    if (!std::isfinite(lengths[d]) || lengths[d] <= 0.0 ||
        pbc_flag(box, d) < 0 || pbc_flag(box, d) > 1) {
      PRINT_INPUT_ERROR("Interface geometry box lengths and PBC flags are invalid.\n");
    }
  }
  for (int d = 0; d < 9; ++d) box_matrix_.emplace_back(box.cpu_h[d]);
  pbc_ = {box.pbc_x, box.pbc_y, box.pbc_z};
  for (int i : {1, 2, 3, 5, 6, 7}) {
    if (std::abs(box.cpu_h[i]) > 1.0e-12) {
      PRINT_INPUT_ERROR("Interface geometry requires a diagonal orthogonal box.\n");
    }
  }

  box_length_A_ = lengths[axis_];
  cross_section_A2_ = box.get_area(axis_);
  const double raw_bin_count = std::ceil(box_length_A_ / bin_width_A_);
  if (!std::isfinite(cross_section_A2_) || cross_section_A2_ <= 0.0 ||
      !std::isfinite(raw_bin_count) || raw_bin_count < 1.0 ||
      raw_bin_count > std::numeric_limits<int>::max()) {
    PRINT_INPUT_ERROR("Interface geometry bin count or area is invalid.\n");
  }
  const int num_bins = static_cast<int>(raw_bin_count);
  for (double& coordinate : interfaces_A_) {
    if (pbc_[axis_]) {
      if (!interface_wrap_coordinate(coordinate, box_length_A_)) {
        PRINT_INPUT_ERROR("Interface coordinates cannot be wrapped into the periodic box.\n");
      }
    } else if (coordinate < 0.0 || coordinate >= box_length_A_) {
      PRINT_INPUT_ERROR("Nonperiodic interface coordinates must lie in [0,L).\n");
    }
  }
  std::sort(interfaces_A_.begin(), interfaces_A_.end());
  for (size_t i = 1; i < interfaces_A_.size(); ++i) {
    if (interfaces_A_[i] <= interfaces_A_[i - 1]) {
      PRINT_INPUT_ERROR("Normalized interface coordinates must be unique.\n");
    }
  }
  const int* material_labels = group_labels(groups, material_method_, N);
  const int* bath_labels = group_labels(groups, bath_method_, N);
  const int* bin_labels = has_bin_method_ ? group_labels(groups, bin_method_, N) : nullptr;
  if (source_group_ < 0 || sink_group_ < 0 || source_group_ == sink_group_ ||
      source_group_ >= groups[bath_method_].number || sink_group_ >= groups[bath_method_].number) {
    PRINT_INPUT_ERROR("Interface source and sink groups must be distinct valid IDs.\n");
  }
  int heat_source = -1;
  int heat_sink = -1;
  bool heat_uses_region = false;
  if (integrate.get_heat_bath_groups(heat_source, heat_sink, heat_uses_region)) {
    if (heat_uses_region) {
      PRINT_INPUT_ERROR("Interface geometry does not support fractional-region heat baths.\n");
    }
    if (bath_method_ != 0 || source_group_ != heat_source || sink_group_ != heat_sink) {
      PRINT_INPUT_ERROR(
        "Interface bath groups must match the active heat ensemble source/sink in grouping method 0.\n");
    }
  }

  const int fixed_group = integrate.get_fixed_group();
  const int move_group = integrate.get_move_group();
  fixed_method_ = fixed_group >= 0 ? integrate.get_fixed_grouping_method() : -1;
  fixed_group_id_ = fixed_group;
  moving_method_ = move_group >= 0 ? integrate.get_move_grouping_method() : -1;
  moving_group_id_ = move_group;
  const int* fixed_labels = fixed_group >= 0
    ? group_labels(groups, integrate.get_fixed_grouping_method(), N) : nullptr;
  const int* moving_labels = move_group >= 0
    ? group_labels(groups, integrate.get_move_grouping_method(), N) : nullptr;
  if ((fixed_group >= 0 && fixed_group >= groups[integrate.get_fixed_grouping_method()].number) ||
      (move_group >= 0 && move_group >= groups[integrate.get_move_grouping_method()].number)) {
    PRINT_INPUT_ERROR("Active fix/move group is invalid for interface mapping.\n");
  }

  std::vector<double> positions(static_cast<size_t>(N) * 3);
  atom.position_per_atom.copy_to_host(positions.data(), positions.size());
  atom_bin_.resize(N, -1);
  atom_material_.resize(N, -1);
  atom_bath_.resize(N, 0);
  atom_type_ = atom.cpu_type;
  atom_fixed_.resize(N, 0);
  atom_moving_.resize(N, 0);
  reference_positions_.resize(static_cast<size_t>(N) * 3);
  masses_ = atom.cpu_mass;
  if (atom_type_.size() != static_cast<size_t>(N) || masses_.size() != static_cast<size_t>(N)) {
    PRINT_INPUT_ERROR("Interface geometry requires atom type and mass data for every atom.\n");
  }

  bin_edges_A_.resize(static_cast<size_t>(num_bins) + 1);
  for (int b = 0; b < num_bins; ++b) {
    bin_edges_A_[b] = b * bin_width_A_;
  }
  bin_edges_A_[num_bins] = box_length_A_;
  bins_.resize(num_bins);
  std::vector<InterfaceMaterialAccumulator> material_by_bin(num_bins);
  for (int b = 0; b < num_bins; ++b) {
    bins_[b].lower_A = bin_edges_A_[b];
    bins_[b].upper_A = bin_edges_A_[b + 1];
    bins_[b].volume_A3 = (bins_[b].upper_A - bins_[b].lower_A) * cross_section_A2_;
    if (!std::isfinite(bins_[b].volume_A3) || bins_[b].volume_A3 <= 0.0) {
      PRINT_INPUT_ERROR("Interface bin volume must be finite and positive.\n");
    }
    for (double plane : interfaces_A_) {
      if (interface_bin_crosses_plane(bins_[b].lower_A, bins_[b].upper_A, plane)) {
        bins_[b].interface_crossing = 1;
      }
    }
  }

  for (int n = 0; n < N; ++n) {
    double xyz[3];
    for (int d = 0; d < 3; ++d) {
      double coordinate = positions[d * N + n];
      if (!std::isfinite(coordinate)) PRINT_INPUT_ERROR("Reference atom position is non-finite.\n");
      if (pbc_[d]) {
        if (!interface_wrap_coordinate(coordinate, lengths[d])) {
          PRINT_INPUT_ERROR("Cannot wrap reference atom position into the periodic box.\n");
        }
      } else if (coordinate < 0.0 || coordinate >= lengths[d]) {
        PRINT_INPUT_ERROR("A nonperiodic reference atom lies outside [0,L); no clamping is applied.\n");
      }
      xyz[d] = coordinate;
      reference_positions_[3 * n + d] = coordinate;
    }
    const int b = interface_bin_index(xyz[axis_], bin_width_A_, num_bins, box_length_A_);
    if (b < 0 || b >= num_bins || xyz[axis_] >= box_length_A_) {
      PRINT_INPUT_ERROR("Reference atom cannot be assigned to an interface bin.\n");
    }
    if (!std::isfinite(masses_[n]) || masses_[n] <= 0.0 || atom_type_[n] < 0 || material_labels[n] < 0 ||
        material_labels[n] >= groups[material_method_].number || bath_labels[n] < 0 ||
        bath_labels[n] >= groups[bath_method_].number) {
      PRINT_INPUT_ERROR("Interface geometry atom type, mass, and material labels must be valid.\n");
    }
    if (has_bin_method_ && bin_labels[n] != b) {
      PRINT_INPUT_ERROR("bin_group label does not match the geometric reference bin.\n");
    }
    const bool source = bath_labels[n] == source_group_;
    const bool sink = bath_labels[n] == sink_group_;
    if (source && sink) PRINT_INPUT_ERROR("An atom cannot belong to both interface bath groups.\n");
    const int bath = source ? 1 : (sink ? 2 : 0);
    atom_bin_[n] = b;
    atom_bath_[n] = bath;
    atom_material_[n] = material_labels[n];
    atom_fixed_[n] = fixed_group >= 0 && fixed_labels[n] == fixed_group;
    atom_moving_[n] = move_group >= 0 && moving_labels[n] == move_group;
    ++bins_[b].reference_atoms;
    if (bath) bins_[b].bath_mask |= bath;
    material_by_bin[b].add(material_labels[n]);
  }
  for (int b = 0; b < num_bins; ++b) {
    bins_[b].material_id = material_by_bin[b].result();
  }

  write_outputs();
  initialized_ = true;
}

void InterfaceGeometry::check_box(const Box& box) const
{
  if (!initialized_) PRINT_INPUT_ERROR("Interface geometry has not been initialized.\n");
  if (!box.is_orthogonal || box.pbc_x != pbc_[0] || box.pbc_y != pbc_[1] || box.pbc_z != pbc_[2]) {
    PRINT_INPUT_ERROR("Interface geometry does not support changing box shape or PBC.\n");
  }
  for (int i = 0; i < 9; ++i) {
    if (box.cpu_h[i] != box_matrix_[i]) {
      PRINT_INPUT_ERROR("Interface geometry does not support box deformation.\n");
    }
  }
}

void InterfaceGeometry::write_outputs() const
{
  const std::string json_name = output_prefix_ + "_geometry.json";
  const std::string groups_name = output_prefix_ + "_groups.dat";
  const std::string bins_name = output_prefix_ + "_bins.dat";
  if (output_exists(json_name) || output_exists(groups_name) || output_exists(bins_name)) {
    PRINT_INPUT_ERROR("Interface geometry output already exists; choose a new prefix.\n");
  }
  FILE* json = my_fopen(json_name.c_str(), "w");
  std::fprintf(json, "{\n  \"algorithm_version\": 1,\n  \"axis\": \"%c\",\n", "xyz"[axis_]);
  std::fprintf(json, "  \"box_matrix_A\": [");
  for (size_t i = 0; i < box_matrix_.size(); ++i) std::fprintf(json, "%s%.17g", i ? ", " : "", box_matrix_[i]);
  std::fprintf(json, "],\n  \"pbc\": [%d, %d, %d],\n", pbc_[0], pbc_[1], pbc_[2]);
  std::fprintf(
    json,
    "  \"geometry_validity\": \"orthogonal_static_reference\",\n  \"box_thickness_A\": %.17g,\n  \"cross_section_A2\": %.17g,\n  \"bin_count\": %d,\n",
    box_length_A_, cross_section_A2_, bin_count());
  std::fprintf(json, "  \"bin_width_A\": %.17g,\n  \"bin_edges_A\": [", bin_width_A_);
  for (size_t i = 0; i < bin_edges_A_.size(); ++i) std::fprintf(json, "%s%.17g", i ? ", " : "", bin_edges_A_[i]);
  std::fprintf(json, "],\n  \"interfaces_A\": [");
  for (size_t i = 0; i < interfaces_A_.size(); ++i) std::fprintf(json, "%s%.17g", i ? ", " : "", interfaces_A_[i]);
  std::fprintf(
    json,
    "],\n  \"group_methods\": {\"material\": %d, \"bath\": %d, \"bin\": %d, \"fixed\": %d, \"moving\": %d},\n",
    material_method_, bath_method_, has_bin_method_ ? bin_method_ : -1, fixed_method_, moving_method_);
  std::fprintf(
    json,
    "  \"bath_group_ids\": {\"source\": %d, \"sink\": %d},\n  \"excluded_group_ids\": {\"fixed\": %d, \"moving\": %d},\n  \"atom_mapping\": \"pre_run_wrapped_reference_positions\",\n  \"options\": {\"nonperiodic_out_of_range\": \"error\", \"bin_group_labels\": \"must_equal_geometric_bin_id\"},\n  \"units\": {\"length\": \"A\", \"area\": \"A^2\", \"volume\": \"A^3\", \"mass\": \"amu\"}\n}\n",
    source_group_, sink_group_, fixed_group_id_, moving_group_id_);
  std::fclose(json);

  FILE* groups = my_fopen(groups_name.c_str(), "w");
  std::fprintf(groups, "# atom_id bin_id material_id bath_id fixed moving ref_x_A ref_y_A ref_z_A mass_amu type_id\n");
  for (size_t n = 0; n < atom_bin_.size(); ++n) {
    std::fprintf(
      groups,
      "%zu %d %d %d %d %d %.17g %.17g %.17g %.17g %d\n",
      n, atom_bin_[n], atom_material_[n], atom_bath_[n], atom_fixed_[n], atom_moving_[n],
      reference_positions_[3 * n], reference_positions_[3 * n + 1], reference_positions_[3 * n + 2],
      masses_[n], atom_type_[n]);
  }
  std::fclose(groups);

  FILE* bins = my_fopen(bins_name.c_str(), "w");
  std::fprintf(bins, "# bin_id lower_A upper_A volume_A3 reference_atoms material_id bath_mask interface_crossing\n");
  for (size_t b = 0; b < bins_.size(); ++b) {
    const InterfaceBinGeometry& bin = bins_[b];
    std::fprintf(
      bins,
      "%zu %.17g %.17g %.17g %d %d %d %d\n",
      b, bin.lower_A, bin.upper_A, bin.volume_A3, bin.reference_atoms, bin.material_id,
      bin.bath_mask, bin.interface_crossing);
  }
  std::fclose(bins);
}
