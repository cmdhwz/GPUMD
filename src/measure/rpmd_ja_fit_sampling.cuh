#pragma once

#include <cmath>
#include <cstddef>

struct RpmdJAFitCell
{
  double cell[9];
  double inverse[9];
};

__host__ __device__ inline bool rpmd_ja_fit_mic(
  const double* inverse, const double* cell, double& x, double& y, double& z)
{
  double sx = inverse[0] * x + inverse[1] * y + inverse[2] * z;
  double sy = inverse[3] * x + inverse[4] * y + inverse[5] * z;
  double sz = inverse[6] * x + inverse[7] * y + inverse[8] * z;
  if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz)) return false;
  sx -= nearbyint(sx);
  sy -= nearbyint(sy);
  sz -= nearbyint(sz);
  x = cell[0] * sx + cell[1] * sy + cell[2] * sz;
  y = cell[3] * sx + cell[4] * sy + cell[5] * sz;
  z = cell[6] * sx + cell[7] * sy + cell[8] * sz;
  return isfinite(x) && isfinite(y) && isfinite(z);
}

static __global__ void rpmd_ja_fit_sampling_kernel(
  const int atoms,
  const int beads,
  const double* const* positions,
  const double* const* forces,
  const RpmdJAFitCell box,
  double* output)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;

  const std::size_t x_index = static_cast<std::size_t>(atom);
  const std::size_t y = x_index + atoms;
  const std::size_t z = y + atoms;
  double ring_x = 0.0, ring_y = 0.0, ring_z = 0.0;
  double prev_x = 0.0, prev_y = 0.0, prev_z = 0.0;
  double centroid_x = 0.0, centroid_y = 0.0, centroid_z = 0.0;
  double mic_x = 0.0, mic_y = 0.0, mic_z = 0.0;
  double force_x = 0.0, force_y = 0.0, force_z = 0.0;
  double scale = 1.0;
  unsigned int error = 0;

  for (int bead = 0; bead < beads; ++bead) {
    const double* position = positions[bead];
    const double* force = forces[bead];
    const double px = position[atom], py = position[y], pz = position[z];
    const double fx = force[atom], fy = force[y], fz = force[z];
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) error |= 1;
    if (!isfinite(fx) || !isfinite(fy) || !isfinite(fz)) error |= 2;
    scale = fmax(scale, fmax(fabs(px), fmax(fabs(py), fabs(pz))));
    force_x += fx / beads;
    force_y += fy / beads;
    force_z += fz / beads;

    if (bead == 0) {
      ring_x = prev_x = px;
      ring_y = prev_y = py;
      ring_z = prev_z = pz;
      centroid_x = px / beads;
      centroid_y = py / beads;
      centroid_z = pz / beads;
    } else {
      double dx = px - prev_x, dy = py - prev_y, dz = pz - prev_z;
      if (!rpmd_ja_fit_mic(box.inverse, box.cell, dx, dy, dz)) error |= 4;
      prev_x += dx; prev_y += dy; prev_z += dz;
      centroid_x += prev_x / beads;
      centroid_y += prev_y / beads;
      centroid_z += prev_z / beads;
    }
    double dx = px - ring_x, dy = py - ring_y, dz = pz - ring_z;
    if (!rpmd_ja_fit_mic(box.inverse, box.cell, dx, dy, dz)) error |= 8;
    mic_x += dx / beads;
    mic_y += dy / beads;
    mic_z += dz / beads;
  }

  double close_x = ring_x - prev_x, close_y = ring_y - prev_y, close_z = ring_z - prev_z;
  if (!rpmd_ja_fit_mic(box.inverse, box.cell, close_x, close_y, close_z)) error |= 16;
  const double winding_x = prev_x - ring_x + close_x;
  const double winding_y = prev_y - ring_y + close_y;
  const double winding_z = prev_z - ring_z + close_z;
  const double branch_x = centroid_x - (ring_x + mic_x);
  const double branch_y = centroid_y - (ring_y + mic_y);
  const double branch_z = centroid_z - (ring_z + mic_z);
  const double winding_norm = sqrt(winding_x * winding_x + winding_y * winding_y + winding_z * winding_z);
  const double branch_norm = sqrt(branch_x * branch_x + branch_y * branch_y + branch_z * branch_z);
  if (!isfinite(winding_norm)) error |= 32;
  if (!isfinite(branch_norm)) error |= 64;

  output[atom] = centroid_x;
  output[y] = centroid_y;
  output[z] = centroid_z;
  const std::size_t force_offset = static_cast<std::size_t>(3) * atoms;
  output[force_offset + atom] = force_x;
  output[force_offset + y] = force_y;
  output[force_offset + z] = force_z;
  output[static_cast<std::size_t>(6) * atoms + atom] = scale;
  output[static_cast<std::size_t>(7) * atoms + atom] = winding_norm;
  output[static_cast<std::size_t>(8) * atoms + atom] = branch_norm;
  output[static_cast<std::size_t>(9) * atoms + atom] = static_cast<double>(error);
}
