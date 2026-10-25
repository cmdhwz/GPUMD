#pragma once

#include <string>
#include <functional>
#include <cstdint>
#include <vector>

#ifdef GPUMD_RPMD_JA_QNEP_PREPARE_FAULT_INJECTION_TEST
namespace rpmd_ja_qnep_prepare_test
{
extern std::function<void(const std::string&)> after_publish_before_readback;
}
#endif

class Atom;
class Box;
class Force;
struct RpmdJAReference;
struct RpmdJAAdditiveData;

struct RpmdJADiagnosticMode
{
  double eigenvalue = 0.0;
  double residual = 0.0;
  std::vector<double> mass_weighted_direction;
};

using RpmdJAModeValidator = std::function<std::string(
  const RpmdJAReference&, const std::vector<RpmdJADiagnosticMode>&)>;

struct RpmdJAPrepareResourcePlan
{
  std::uint64_t free_gpu_bytes = 0;
  std::uint64_t planned_peak_gpu_bytes = 0;
  std::uint64_t cartesian_matrix_bytes = 0;
  std::uint64_t internal_matrix_bytes = 0;
  std::uint64_t potrf_workspace_bytes = 0;
  std::uint64_t svd_workspace_bytes = 0;
  std::uint64_t svd_peak_bytes = 0;
  std::uint64_t allocator_safety_margin_bytes = 0;
};

struct RpmdJAPrepareWorkspaceSizes
{
  std::uint64_t potrf_workspace_bytes = 0;
  std::uint64_t svd_workspace_bytes = 0;
};

using RpmdJAPrepareResourceObserver = std::function<void(const RpmdJAPrepareResourcePlan&)>;
using RpmdJAPrepareAllocationObserver = std::function<void(const char*, std::uint64_t)>;

RpmdJAModeValidator make_rpmd_ja_qnep_mode_validator(Atom& atom, Box& box, Force& force);
RpmdJAPrepareWorkspaceSizes query_rpmd_ja_qnep_prepare_workspace_sizes(int atom_count);

void prepare_rpmd_ja_qnep_reference(
  const std::string& raw_path,
  const std::string& output_path,
  const std::string& kernel_table_path,
  const RpmdJAModeValidator& mode_validator = {},
  const std::string& additive_path = {},
  const std::function<void(const RpmdJAAdditiveData&)>& additive_validator = {},
  const RpmdJAPrepareResourceObserver& resource_observer = {},
  const RpmdJAPrepareAllocationObserver& allocation_observer = {},
  const std::function<void(const RpmdJAReference&)>& readback_validator = {});
