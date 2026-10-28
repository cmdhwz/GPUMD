#pragma once

#include "rpmd_ja.cuh"

static_assert(3 * RPMD_JA_SAMPLE_BATCH_SIZE <= 128, "JA batch exceeds store kernel threads");

static __global__ void store_current(
  const int Nd, const int start, const int count, const double* centroid_heat,
  const double* delta, double* jcent_history, double* delta_history,
  double* ja_history, int* error)
{
  const int index = threadIdx.x;
  if (index >= 3 * count) return;
  const int slot = index / 3;
  const int d = index % 3;
  const int frame = start + slot;
  const double jcent = d == 0 ? centroid_heat[frame + Nd * 0] + centroid_heat[frame + Nd * 1]
                    : d == 1 ? centroid_heat[frame + Nd * 2] + centroid_heat[frame + Nd * 3]
                             : centroid_heat[frame + Nd * 4];
  const double correction = delta[index];
  if (!isfinite(jcent) || !isfinite(correction) || !isfinite(jcent + correction))
    atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  const size_t output = static_cast<size_t>(frame) + static_cast<size_t>(Nd) * d;
  jcent_history[output] = jcent;
  delta_history[output] = correction;
  ja_history[output] = jcent + correction;
}
