/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

class Atom;
class Box;
class Group;
class Integrate;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define INTERFACE_GEOMETRY_HD __host__ __device__
#else
#define INTERFACE_GEOMETRY_HD
#endif

INTERFACE_GEOMETRY_HD inline bool interface_geometry_isfinite(const double value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

inline bool interface_wrap_coordinate(double& x, const double length)
{
  if (!std::isfinite(x) || !std::isfinite(length) || length <= 0.0) return false;
  x -= std::floor(x / length) * length;
  if (x >= length) x = 0.0;
  return x >= 0.0;
}

inline bool interface_normalize_plane_coordinates(
  std::vector<double>& coordinates, const double length, const bool periodic)
{
  if (!std::isfinite(length) || length <= 0.0) return false;
  for (double& coordinate : coordinates) {
    if (!std::isfinite(coordinate)) return false;
    if (periodic) {
      if (!interface_wrap_coordinate(coordinate, length)) return false;
    } else if (coordinate < 0.0 || coordinate >= length) {
      return false;
    }
  }
  std::sort(coordinates.begin(), coordinates.end());
  for (size_t i = 1; i < coordinates.size(); ++i) {
    if (coordinates[i] <= coordinates[i - 1]) return false;
  }
  return true;
}

inline int interface_bin_index(
  const double x, const double width, const int bin_count, const double box_length)
{
  if (!std::isfinite(x) || !std::isfinite(width) || !std::isfinite(box_length) ||
      width <= 0.0 || box_length <= 0.0 || bin_count <= 0 || x < 0.0 || x >= box_length)
    return -1;
  const double raw = std::floor(x / width);
  if (!std::isfinite(raw) || raw < 0.0 || raw >= bin_count) return -1;
  return static_cast<int>(raw);
}

INTERFACE_GEOMETRY_HD inline int interface_vacf_origin_count(
  const int block_samples, const int lag)
{
  return (lag >= 0 && lag < block_samples) ? block_samples - lag : 0;
}

enum InterfaceVACFSampleStatus
{
  INTERFACE_VACF_SAMPLE_EMPTY = 0,
  INTERFACE_VACF_SAMPLE_VALID = 1,
  INTERFACE_VACF_SAMPLE_INVALID = 2
};

INTERFACE_GEOMETRY_HD inline int interface_vacf_normalize_sample(
  const double lag_sum,
  const double c0,
  const int block_samples,
  const int lag,
  const int atom_count,
  double& raw,
  double& normalized)
{
  if (atom_count < 0 || block_samples <= 0 || lag < 0 || lag >= block_samples) {
    return INTERFACE_VACF_SAMPLE_INVALID;
  }
  if (atom_count == 0) {
    raw = NAN;
    normalized = NAN;
    return INTERFACE_VACF_SAMPLE_EMPTY;
  }
  const int origins = interface_vacf_origin_count(block_samples, lag);
  if (origins <= 0 || !interface_geometry_isfinite(lag_sum) ||
      !interface_geometry_isfinite(c0)) {
    return INTERFACE_VACF_SAMPLE_INVALID;
  }
  raw = lag_sum / (static_cast<double>(origins) * atom_count);
  if (!interface_geometry_isfinite(raw)) return INTERFACE_VACF_SAMPLE_INVALID;
  normalized = NAN;
  if (c0 > 0.0) {
    normalized = raw / c0;
    if (!interface_geometry_isfinite(normalized)) return INTERFACE_VACF_SAMPLE_INVALID;
  }
  return INTERFACE_VACF_SAMPLE_VALID;
}

inline int interface_ring_slot(const int block_sample_index, const int ring_size)
{
  return ring_size > 0 ? block_sample_index % ring_size : -1;
}

inline int interface_valid_lag_count(const int block_sample_index, const int ring_size)
{
  return block_sample_index < 0 || ring_size <= 0
    ? 0
    : (block_sample_index + 1 < ring_size ? block_sample_index + 1 : ring_size);
}

INTERFACE_GEOMETRY_HD inline int interface_ring_previous_slot(
  const int slot, const int lag, const int ring_size)
{
  if (ring_size <= 0 || slot < 0 || lag < 0 || lag >= ring_size) return -1;
  const int previous = slot - lag;
  return previous < 0 ? previous + ring_size : previous;
}

inline int interface_profile_atom_bin(const int bin, const bool fixed, const bool moving)
{
  return (fixed || moving) ? -1 : bin;
}

inline bool interface_bin_crosses_plane(
  const double lower, const double upper, const double plane)
{
  return plane > lower && plane < upper;
}

inline bool interface_bin_crosses_any_plane(
  const double lower, const double upper, const std::vector<double>& planes)
{
  for (double plane : planes) {
    if (interface_bin_crosses_plane(lower, upper, plane)) return true;
  }
  return false;
}

struct InterfaceMaterialAccumulator
{
  bool has_material = false;
  bool mixed = false;
  int material_id = -1;

  void add(const int material)
  {
    if (!has_material) {
      material_id = material;
      has_material = true;
    } else if (material_id != material) {
      mixed = true;
    }
  }

  int result() const { return mixed || !has_material ? -1 : material_id; }
};

INTERFACE_GEOMETRY_HD inline double interface_hann_window(const int lag, const int count)
{
  return count == 1 ? 1.0 : 0.5 * (1.0 + ::cos(3.14159265358979323846 * lag / (count - 1)));
}

INTERFACE_GEOMETRY_HD inline double interface_one_sided_factor(const int lag)
{
  return lag == 0 ? 1.0 : 2.0;
}

#undef INTERFACE_GEOMETRY_HD

struct InterfaceBinGeometry
{
  double lower_A = 0.0;
  double upper_A = 0.0;
  double volume_A3 = 0.0;
  int reference_atoms = 0;
  int material_id = -1;
  int bath_mask = 0;
  int interface_crossing = 0;
};

class InterfaceGeometry
{
public:
  InterfaceGeometry(const std::string& config_file, const std::string& output_prefix);

  void initialize(
    Atom& atom,
    const Box& box,
    const std::vector<Group>& groups,
    const Integrate& integrate);
  void check_box(const Box& box) const;

  bool initialized() const { return initialized_; }
  int axis() const { return axis_; }
  int bin_count() const { return static_cast<int>(bins_.size()); }
  double bin_width_A() const { return bin_width_A_; }
  double box_length_A() const { return box_length_A_; }
  double cross_section_A2() const { return cross_section_A2_; }
  const std::vector<double>& interfaces_A() const { return interfaces_A_; }
  const std::vector<double>& sampling_planes_A() const { return sampling_planes_A_; }
  const std::vector<double>& box_matrix() const { return box_matrix_; }
  const std::vector<int>& pbc() const { return pbc_; }
  const std::vector<int>& atom_bin() const { return atom_bin_; }
  const std::vector<int>& atom_material() const { return atom_material_; }
  const std::vector<int>& atom_bath() const { return atom_bath_; }
  const std::vector<int>& atom_type() const { return atom_type_; }
  const std::vector<int>& atom_fixed() const { return atom_fixed_; }
  const std::vector<int>& atom_moving() const { return atom_moving_; }
  const std::vector<double>& reference_positions() const { return reference_positions_; }
  const std::vector<double>& masses() const { return masses_; }
  const std::vector<InterfaceBinGeometry>& bins() const { return bins_; }
  const std::string& output_prefix() const { return output_prefix_; }

private:
  std::string config_file_;
  std::string output_prefix_;
  int axis_ = -1;
  double bin_width_A_ = 0.0;
  std::vector<double> interfaces_A_;
  std::vector<double> sampling_planes_A_;
  bool sampling_planes_explicit_ = false;
  int material_method_ = -1;
  int bath_method_ = -1;
  int source_group_ = -1;
  int sink_group_ = -1;
  int bin_method_ = -1;
  int fixed_method_ = -1;
  int fixed_group_id_ = -1;
  int moving_method_ = -1;
  int moving_group_id_ = -1;
  bool has_bin_method_ = false;
  bool initialized_ = false;
  double box_length_A_ = 0.0;
  double cross_section_A2_ = 0.0;
  std::vector<double> box_matrix_;
  std::vector<int> pbc_;
  std::vector<double> bin_edges_A_;
  std::vector<int> atom_bin_;
  std::vector<int> atom_material_;
  std::vector<int> atom_bath_;
  std::vector<int> atom_type_;
  std::vector<int> atom_fixed_;
  std::vector<int> atom_moving_;
  std::vector<double> reference_positions_;
  std::vector<double> masses_;
  std::vector<InterfaceBinGeometry> bins_;

  void parse_config();
  void write_outputs() const;
};
