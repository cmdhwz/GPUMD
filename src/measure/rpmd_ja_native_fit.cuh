#pragma once

#include <cstdint>
#include <string>

class Atom;
class Box;
class Force;

struct RpmdJANativeFitOptions
{
  std::string output_path;
  std::string kernel_table;
  double temperature = 0.0;
  double cutoff = 0.0;
  double epsilon = 0.0;
  double response_tolerance = 0.0;
  double fd_step = 0.0;
  int sample_interval = 0;
};

void fit_rpmd_ja_native_reference(
  const RpmdJANativeFitOptions& options,
  const std::string& spool_path,
  std::uint64_t frame_count,
  Atom& atom,
  Box& box,
  Force& force);

void diagnose_rpmd_ja_native_fit_samples(
  const std::string& spool_path,
  double fd_step,
  Atom& atom,
  Box& box,
  Force& force,
  bool full = false);
