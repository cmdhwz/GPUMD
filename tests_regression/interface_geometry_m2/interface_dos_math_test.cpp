#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/measure/interface_geometry.cuh"
#include <cassert>
#include <cmath>
#include <limits>

int main()
{
  double raw = 0.0;
  double normalized = 0.0;
  const int block_samples = 5;
  const double c0 = 12.0 / (block_samples * 3.0);

  assert(interface_vacf_normalize_sample(
    12.0, c0, block_samples, 0, 3, raw, normalized) == INTERFACE_VACF_SAMPLE_VALID);
  assert(raw == c0 && normalized == 1.0);
  assert(interface_vacf_normalize_sample(
    24.0, c0, block_samples, 1, 3, raw, normalized) == INTERFACE_VACF_SAMPLE_VALID);
  assert(raw == 2.0 && normalized == 2.0 / c0);
  assert(interface_vacf_origin_count(block_samples, 1) == 4);

  const double next_block_c0 = 30.0 / (block_samples * 3.0);
  assert(interface_vacf_normalize_sample(
    15.0, next_block_c0, block_samples, 1, 3, raw, normalized) == INTERFACE_VACF_SAMPLE_VALID);
  assert(raw == 1.25 && normalized == 1.25 / next_block_c0);

  assert(interface_vacf_normalize_sample(
    0.0, 0.0, block_samples, 2, 0, raw, normalized) == INTERFACE_VACF_SAMPLE_EMPTY);
  assert(std::isnan(raw) && std::isnan(normalized));
  assert(interface_vacf_normalize_sample(
    0.0, 0.0, block_samples, 0, 2, raw, normalized) == INTERFACE_VACF_SAMPLE_VALID);
  assert(raw == 0.0 && std::isnan(normalized));

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  assert(interface_vacf_normalize_sample(
    nan, 1.0, block_samples, 1, 2, raw, normalized) == INTERFACE_VACF_SAMPLE_INVALID);
  assert(interface_vacf_normalize_sample(
    1.0, inf, block_samples, 1, 2, raw, normalized) == INTERFACE_VACF_SAMPLE_INVALID);
  assert(interface_vacf_normalize_sample(
    inf, 1.0, block_samples, 1, 2, raw, normalized) == INTERFACE_VACF_SAMPLE_INVALID);
  return 0;
}
