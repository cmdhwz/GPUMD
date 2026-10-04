#pragma once

#include <string>
#include <functional>
#include <vector>

class Atom;
class Box;
class Force;
struct RpmdJAReference;

struct RpmdJADiagnosticMode
{
  double eigenvalue = 0.0;
  double residual = 0.0;
  std::vector<double> mass_weighted_direction;
};

using RpmdJAModeValidator = std::function<std::string(
  const RpmdJAReference&, const std::vector<RpmdJADiagnosticMode>&)>;

RpmdJAModeValidator make_rpmd_ja_qnep_mode_validator(Atom& atom, Box& box, Force& force);

void prepare_rpmd_ja_qnep_reference(
  const std::string& raw_path,
  const std::string& output_path,
  const std::string& kernel_table_path,
  const RpmdJAModeValidator& mode_validator = {});
