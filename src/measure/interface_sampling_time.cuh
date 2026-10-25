#pragma once

#include <cmath>

inline bool interface_time_step_matches(const double actual, const double nominal)
{
  if (!std::isfinite(actual) || !std::isfinite(nominal) || actual <= 0.0 || nominal <= 0.0) {
    return false;
  }
  return actual == nominal;
}

inline bool interface_time_step_to_ps(
  const double time_step, const double fs_per_internal_time, double& time_step_ps)
{
  if (!std::isfinite(time_step) || time_step <= 0.0 ||
      !std::isfinite(fs_per_internal_time) || fs_per_internal_time <= 0.0) {
    return false;
  }
  time_step_ps = time_step * fs_per_internal_time / 1000.0;
  return std::isfinite(time_step_ps) && time_step_ps > 0.0;
}

inline bool interface_sample_interval_ps(
  const double time_step,
  const double fs_per_internal_time,
  const int sample_interval,
  double& sample_dt_ps)
{
  double time_step_ps = 0.0;
  if (sample_interval <= 0 || !interface_time_step_to_ps(time_step, fs_per_internal_time, time_step_ps)) {
    return false;
  }
  sample_dt_ps = time_step_ps * sample_interval;
  return std::isfinite(sample_dt_ps) && sample_dt_ps > 0.0;
}

inline bool interface_time_ps_is_increasing(
  const double current_ps, const double previous_ps, const bool have_previous)
{
  return std::isfinite(current_ps) &&
         (!have_previous || (std::isfinite(previous_ps) && current_ps > previous_ps));
}
