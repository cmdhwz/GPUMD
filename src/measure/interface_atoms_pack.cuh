/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "interface_geometry.cuh"
#include <cstddef>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define INTERFACE_ATOMS_HD __host__ __device__
#else
#define INTERFACE_ATOMS_HD
#endif

constexpr size_t INTERFACE_ATOMS_COMPONENTS = 13;

inline bool interface_atoms_checked_product(
  const size_t a, const size_t b, size_t& result)
{
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  result = a * b;
  return true;
}

inline bool interface_atoms_frame_value_count(
  const size_t frames, const size_t selected_atoms, size_t& values)
{
  size_t rows = 0;
  return interface_atoms_checked_product(frames, selected_atoms, rows) &&
    interface_atoms_checked_product(rows, INTERFACE_ATOMS_COMPONENTS, values);
}

inline bool interface_atoms_frame_offset(
  const size_t frame, const size_t selected_atoms, size_t& offset)
{
  return interface_atoms_frame_value_count(frame, selected_atoms, offset);
}

INTERFACE_ATOMS_HD inline bool interface_atoms_pack_record(
  const size_t atom_id,
  const int atom_count,
  const double* position,
  const int* position_image,
  const double* velocity,
  const double* force,
  const double* potential,
  const double box_length_x,
  const double box_length_y,
  const double box_length_z,
  const double velocity_conversion,
  double* record)
{
  if (atom_count <= 0 || atom_id >= static_cast<size_t>(atom_count) ||
      !interface_geometry_isfinite(velocity_conversion)) {
    return false;
  }
  for (size_t d = 0; d < 3; ++d) {
    const size_t index = atom_id + static_cast<size_t>(atom_count) * d;
    const double box_length = d == 0 ? box_length_x : (d == 1 ? box_length_y : box_length_z);
    record[d] = position[index];
    record[d + 3] = position[index] + box_length * position_image[index];
    record[d + 6] = velocity[index] * velocity_conversion;
    record[d + 9] = force[index];
    if (!interface_geometry_isfinite(record[d]) ||
        !interface_geometry_isfinite(record[d + 3]) ||
        !interface_geometry_isfinite(record[d + 6]) ||
        !interface_geometry_isfinite(record[d + 9])) {
      return false;
    }
  }
  record[12] = potential[atom_id];
  return interface_geometry_isfinite(record[12]);
}

#undef INTERFACE_ATOMS_HD
