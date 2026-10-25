#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/measure/interface_sampling_time.cuh"
#include <cassert>
#include <cmath>
#include <limits>

static int simulate_constant_run(
  const double start_ps, const int steps, const int sample_interval, const int block_samples)
{
  constexpr double fs_per_internal_time = 10.18051;
  const double dt = 1.0 / fs_per_internal_time;
  double sample_dt = 0.0;
  assert(interface_sample_interval_ps(dt, fs_per_internal_time, sample_interval, sample_dt));
  assert(std::abs(sample_dt - 0.001 * sample_interval) < 1.0e-12);

  double global_time = start_ps * 1000.0 / fs_per_internal_time;
  double previous_ps = 0.0;
  double previous_sample_ps = 0.0;
  bool have_time = false;
  bool have_sample = false;
  int samples = 0;
  int complete_blocks = 0;
  for (int step = 0; step < steps; ++step) {
    assert(interface_time_step_matches(dt, dt));
    global_time += dt;
    const double now_ps = global_time * fs_per_internal_time / 1000.0;
    assert(interface_time_ps_is_increasing(now_ps, previous_ps, have_time));
    previous_ps = now_ps;
    have_time = true;
    if ((step + 1) % sample_interval == 0) {
      if (have_sample) assert(interface_time_ps_is_increasing(now_ps, previous_sample_ps, true));
      previous_sample_ps = now_ps;
      have_sample = true;
      if (++samples % block_samples == 0) ++complete_blocks;
    }
  }
  assert(complete_blocks >= 2);
  return samples;
}

int main()
{
  constexpr double fs_per_internal_time = 10.18051;
  const double one_fs = 1.0 / fs_per_internal_time;
  const double half_fs = 0.5 / fs_per_internal_time;
  assert(interface_time_step_matches(one_fs, one_fs));
  assert(!interface_time_step_matches(half_fs, one_fs));
  assert(!interface_time_step_matches(std::numeric_limits<double>::quiet_NaN(), one_fs));

  assert(simulate_constant_run(0.0, 1000000, 1, 1000) == 1000000);
  assert(simulate_constant_run(2000.0, 12, 3, 2) == 4);
  assert(simulate_constant_run(10000.0, 8, 1, 4) == 8);
  assert(!interface_time_ps_is_increasing(1.0, 1.0, true));
  assert(!interface_time_ps_is_increasing(std::numeric_limits<double>::infinity(), 1.0, true));
  return 0;
}
