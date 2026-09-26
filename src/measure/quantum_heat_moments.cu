#include "quantum_heat_moments.cuh"
#include "hac.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>

#ifndef GPUMD_GIT_COMMIT
#define GPUMD_GIT_COMMIT "unknown"
#endif
#ifndef GPUMD_GIT_COMMIT_FULL
#define GPUMD_GIT_COMMIT_FULL "unknown"
#endif
#ifndef GPUMD_GIT_DESCRIBE
#define GPUMD_GIT_DESCRIBE "unknown"
#endif
#ifndef GPUMD_GIT_DIRTY
#define GPUMD_GIT_DIRTY "unknown"
#endif

namespace
{
using quantum_heat_moments::Complex;
using quantum_heat_moments::ComplexStats;
using quantum_heat_moments::MoyalProbe;
using quantum_heat_moments::parse_integer;
using quantum_heat_moments::parse_seed;
using quantum_heat_moments::RecursionSettings;

unsigned long long model_fingerprint(const char* filename)
{
  FILE* file = my_fopen(filename, "rb");
  unsigned long long hash = 14695981039346656037ULL;
  unsigned char buffer[4096];
  size_t count;
  while ((count = std::fread(buffer, 1, sizeof(buffer), file)) != 0)
    for (size_t i = 0; i < count; ++i) hash = (hash ^ buffer[i]) * 1099511628211ULL;
  const bool read_error = std::ferror(file) != 0;
  std::fclose(file);
  if (read_error) PRINT_INPUT_ERROR("Could not fingerprint the QHM NEP model file.");
  return hash;
}

bool existing_segmented_output(const char* filename, const std::string& expected_contract)
{
  errno = 0;
  FILE* file = std::fopen(filename, "rb");
  if (file == nullptr) {
    if (errno == ENOENT) return false;
    std::fprintf(stderr, "Could not inspect QHM output before appending: %s\n", filename);
    PRINT_INPUT_ERROR("Existing QHM output must be readable before appending.");
  }
  char first_line[64], version_line[64], contract_line[512];
  const bool nonempty = std::fgets(first_line, sizeof(first_line), file) != nullptr;
  const bool has_contract = nonempty &&
    std::fgets(version_line, sizeof(version_line), file) != nullptr &&
    std::fgets(contract_line, sizeof(contract_line), file) != nullptr;
  const bool read_error = std::ferror(file) != 0;
  std::fclose(file);
  if (read_error) PRINT_INPUT_ERROR("Could not inspect an existing QHM output file.");
  if (nonempty) {
    if (!has_contract || std::strncmp(first_line, "# segment_begin id ", 19) != 0 ||
        std::strncmp(version_line, "# qhm_segment_metadata_version 2", 32) != 0) {
      std::fprintf(stderr, "Existing QHM output has no compatible segment contract: %s\n", filename);
      PRINT_INPUT_ERROR("Move or rename legacy QHM outputs before appending a new segment.");
    }
    contract_line[std::strcspn(contract_line, "\r\n")] = '\0';
    const std::string expected_line = "# qhm_append_contract " + expected_contract;
    if (expected_line != contract_line) {
      std::fprintf(stderr, "Existing QHM output has a different model or column contract: %s\n", filename);
      PRINT_INPUT_ERROR("Move or rename incompatible QHM outputs before appending a new segment.");
    }
  }
  return nonempty;
}

bool parse_real(const char* text, double& value)
{
  char* end = nullptr;
  value = std::strtod(text, &end);
  return end != text && *end == '\0' && std::isfinite(value);
}

bool parse_yes_no(const char* text, bool& value)
{
  if (std::strcmp(text, "yes") == 0) value = true;
  else if (std::strcmp(text, "no") == 0) value = false;
  else return false;
  return true;
}

char alpha_name(const int alpha) { return "xyz"[alpha]; }

bool finite(const Complex value)
{
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void print_real(FILE* file, const double value)
{
  std::fprintf(file, " %.16e", value);
}

void print_complex(FILE* file, const Complex value)
{
  std::fprintf(file, " %.16e %.16e", value.real(), value.imag());
}

double norm(const std::vector<double>& values)
{
  double squared = 0.0;
  for (const double value : values) squared += value * value;
  return std::sqrt(squared);
}

std::array<double, 9> lattice_matrix(const Box& box)
{
  std::array<double, 9> h{};
  std::copy(box.cpu_h, box.cpu_h + 9, h.begin());
  return h;
}

std::vector<double> wrap_position(const Box& box, const std::vector<double>& position)
{
  const int number_of_atoms = static_cast<int>(position.size() / 3);
  std::vector<double> wrapped(position.size());
  for (int i = 0; i < number_of_atoms; ++i) {
    double fractional[3] = {
      box.cpu_h[9] * position[i] + box.cpu_h[10] * position[i + number_of_atoms] +
        box.cpu_h[11] * position[i + 2 * number_of_atoms],
      box.cpu_h[12] * position[i] + box.cpu_h[13] * position[i + number_of_atoms] +
        box.cpu_h[14] * position[i + 2 * number_of_atoms],
      box.cpu_h[15] * position[i] + box.cpu_h[16] * position[i + number_of_atoms] +
        box.cpu_h[17] * position[i + 2 * number_of_atoms]};
    for (double& value : fractional) value -= std::floor(value);
    wrapped[i] = box.cpu_h[0] * fractional[0] + box.cpu_h[1] * fractional[1] + box.cpu_h[2] * fractional[2];
    wrapped[i + number_of_atoms] = box.cpu_h[3] * fractional[0] + box.cpu_h[4] * fractional[1] + box.cpu_h[5] * fractional[2];
    wrapped[i + 2 * number_of_atoms] = box.cpu_h[6] * fractional[0] + box.cpu_h[7] * fractional[1] + box.cpu_h[8] * fractional[2];
  }
  return wrapped;
}

double dot(const std::vector<double>& left, const std::vector<double>& right)
{
  double result = 0.0;
  for (size_t i = 0; i < left.size(); ++i) result += left[i] * right[i];
  return result;
}

double kinetic_cubic_third_contraction(
  const std::vector<double>& xi,
  const std::vector<double>& eta,
  const std::vector<double>& zeta,
  const std::vector<double>& mass,
  const int number_of_atoms,
  const int alpha)
{
  double result = 0.0;
  for (int i = 0; i < number_of_atoms; ++i) {
    double xi_eta = 0.0, xi_zeta = 0.0, eta_zeta = 0.0;
    for (int mu = 0; mu < 3; ++mu) {
      const int d = i + mu * number_of_atoms;
      xi_eta += xi[d] * eta[d];
      xi_zeta += xi[d] * zeta[d];
      eta_zeta += eta[d] * zeta[d];
    }
    const int a = i + alpha * number_of_atoms;
    result += (xi_eta * zeta[a] + xi_zeta * eta[a] + eta_zeta * xi[a]) /
      (mass[i] * mass[i]);
  }
  return result;
}
} // namespace

QuantumHeatMoments::QuantumHeatMoments(const char** param, const int num_param)
{
  action_name = "compute_quantum_heat_moments";
  if (num_param < 3 || (num_param - 1) % 2 != 0) {
    PRINT_INPUT_ERROR(
      "compute_quantum_heat_moments expects keyword/value pairs; see the Stage 6.5 run.in syntax.");
  }
  std::set<std::string> seen;
  for (int i = 1; i < num_param; i += 2) {
    const std::string key(param[i]);
    const char* value = param[i + 1];
    if (!seen.insert(key).second) PRINT_INPUT_ERROR("Duplicate compute_quantum_heat_moments option.");
    if (key == "sample_interval") {
      if (!parse_integer(value, sample_interval_) || sample_interval_ <= 0)
        PRINT_INPUT_ERROR("sample_interval must be a positive integer.");
    } else if (key == "static_sample_interval") {
      if (!parse_integer(value, static_sample_interval_) || static_sample_interval_ <= 0)
        PRINT_INPUT_ERROR("static_sample_interval must be a positive integer.");
    } else if (key == "dynamic_sample_interval") {
      if (!parse_integer(value, dynamic_sample_interval_) || dynamic_sample_interval_ <= 0)
        PRINT_INPUT_ERROR("dynamic_sample_interval must be a positive integer.");
    } else if (key == "candidate_sample_interval") {
      if (!parse_integer(value, candidate_sample_interval_) || candidate_sample_interval_ <= 0)
        PRINT_INPUT_ERROR("candidate_sample_interval must be a positive integer.");
    } else if (key == "weyl_max_order") {
      if (!parse_integer(value, weyl_max_order_) ||
          (weyl_max_order_ != 0 && weyl_max_order_ != 2 && weyl_max_order_ != 4 && weyl_max_order_ != 6))
        PRINT_INPUT_ERROR("weyl_max_order must be 0, 2, 4, or 6.");
    } else if (key == "exact_static_max_order") {
      if (!parse_integer(value, exact_static_max_order_) ||
          (exact_static_max_order_ != 0 && exact_static_max_order_ != 2 && exact_static_max_order_ != 4))
        PRINT_INPUT_ERROR("exact_static_max_order must be 0, 2, or 4 (mu6 is not implemented).");
    } else if (key == "compute_exact_mu0") {
      if (!parse_yes_no(value, compute_exact_mu0_)) PRINT_INPUT_ERROR("compute_exact_mu0 must be yes or no.");
    } else if (key == "static_backend") {
      if (std::strcmp(value, "directional") != 0)
        PRINT_INPUT_ERROR("Only static_backend directional is implemented; cartesian_debug is not available.");
    } else if (key == "fd_step_r") {
      if (!parse_real(value, fd_step_r_) || fd_step_r_ <= 0.0)
        PRINT_INPUT_ERROR("fd_step_r must be positive and finite.");
    } else if (key == "fd_step_p") {
      if (!parse_real(value, fd_step_p_) || fd_step_p_ <= 0.0)
        PRINT_INPUT_ERROR("fd_step_p must be positive and finite.");
    } else if (key == "n_moyal_probe") {
      if (!parse_integer(value, n_moyal_probe_) || n_moyal_probe_ < 0)
        PRINT_INPUT_ERROR("n_moyal_probe must be a nonnegative integer.");
    } else if (key == "n_static_trace_probe") {
      if (!parse_integer(value, n_static_trace_probe_) || n_static_trace_probe_ < 0)
        PRINT_INPUT_ERROR("n_static_trace_probe must be a nonnegative integer.");
    } else if (key == "enable_complex_pi") {
      if (!parse_yes_no(value, enable_complex_pi_)) PRINT_INPUT_ERROR("enable_complex_pi must be yes or no.");
    } else if (key == "n_aux_probe") {
      if (!parse_integer(value, n_aux_probe_) || n_aux_probe_ < 0)
        PRINT_INPUT_ERROR("n_aux_probe must be a nonnegative integer.");
    } else if (key == "imag_lag_max") {
      if (!parse_integer(value, imag_lag_max_) || imag_lag_max_ < 0)
        PRINT_INPUT_ERROR("imag_lag_max must be a nonnegative integer.");
    } else if (key == "winding_mode") {
      if (std::strcmp(value, "diagnostic_only") != 0)
        PRINT_INPUT_ERROR("Only winding_mode diagnostic_only is supported; winding diagnostics do not filter frames.");
    } else if (key == "seed") {
      if (!parse_seed(value, seed_)) PRINT_INPUT_ERROR("seed must be a nonnegative integer.");
    } else if (key == "fd_warning_threshold") {
      if (!parse_real(value, fd_warning_threshold_) || fd_warning_threshold_ <= 0.0)
        PRINT_INPUT_ERROR("fd_warning_threshold must be positive and finite.");
    } else if (key == "stochastic_warning_threshold") {
      if (!parse_real(value, stochastic_warning_threshold_) || stochastic_warning_threshold_ <= 0.0)
        PRINT_INPUT_ERROR("stochastic_warning_threshold must be positive and finite.");
    } else if (key == "output_level") {
      if (std::strcmp(value, "summary") == 0) output_level_ = 0;
      else if (std::strcmp(value, "bead") == 0) output_level_ = 1;
      else if (std::strcmp(value, "debug") == 0) output_level_ = 2;
      else PRINT_INPUT_ERROR("output_level must be summary, bead, or debug.");
    } else if (key == "short_range_mechanical_only") {
      if (!parse_yes_no(value, short_range_mechanical_only_))
        PRINT_INPUT_ERROR("short_range_mechanical_only must be yes or no.");
    } else if (key == "validate_edge_derivatives") {
      if (!parse_yes_no(value, validate_edge_derivatives_))
        PRINT_INPUT_ERROR("validate_edge_derivatives must be yes or no.");
    } else if (key == "validate_fd_smoothness") {
      if (!parse_yes_no(value, validate_fd_smoothness_))
        PRINT_INPUT_ERROR("validate_fd_smoothness must be yes or no.");
    } else if (key == "validate_link_images") {
      if (!parse_yes_no(value, validate_link_images_))
        PRINT_INPUT_ERROR("validate_link_images must be yes or no.");
    } else if (key == "edgecheck_step_r") {
      if (!parse_real(value, edgecheck_step_) || edgecheck_step_ <= 0.0)
        PRINT_INPUT_ERROR("edgecheck_step_r must be positive and finite.");
    } else if (key == "expensive_static_warning_threshold") {
      if (!parse_real(value, expensive_static_warning_threshold_) || expensive_static_warning_threshold_ <= 0.0)
        PRINT_INPUT_ERROR("expensive_static_warning_threshold must be positive and finite.");
    } else if (key == "force_debug_large") {
      if (!parse_yes_no(value, force_debug_large_)) PRINT_INPUT_ERROR("force_debug_large must be yes or no.");
    } else {
      PRINT_INPUT_ERROR("Unknown compute_quantum_heat_moments option.");
    }
  }
  if (!seen.count("static_sample_interval")) static_sample_interval_ = sample_interval_;
  if (!seen.count("dynamic_sample_interval")) dynamic_sample_interval_ = sample_interval_;
  if (!seen.count("candidate_sample_interval")) candidate_sample_interval_ = sample_interval_;
  if (!short_range_mechanical_only_)
    PRINT_INPUT_ERROR("Stage 6.5 QHM currently supports short-range mechanical NEP only.");
  exact_mu0_enabled_ = compute_exact_mu0_ || exact_static_max_order_ >= 2;
  exact_mu2_enabled_ = exact_static_max_order_ >= 2;
  exact_mu4_enabled_ = exact_static_max_order_ >= 4;
  exact_mu6_enabled_ = false;
  required_operator_order_ = std::max(
    exact_mu4_enabled_ ? 2 : exact_mu2_enabled_ ? 1 : exact_mu0_enabled_ ? 0 : 0,
    weyl_max_order_ / 2);
  if (weyl_max_order_ > 0 && n_moyal_probe_ == 0)
    PRINT_INPUT_ERROR("n_moyal_probe must be positive when weyl_max_order is nonzero.");
  if (exact_mu2_enabled_ && n_moyal_probe_ == 0)
    PRINT_INPUT_ERROR("mu2 requires positive n_moyal_probe.");
  if ((exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_) && n_static_trace_probe_ == 0)
    PRINT_INPUT_ERROR("exact static moments require positive n_static_trace_probe.");
  if (enable_complex_pi_ && n_aux_probe_ == 0)
    PRINT_INPUT_ERROR("enable_complex_pi yes requires n_aux_probe > 0.");
}

QuantumHeatMoments::~QuantumHeatMoments() { close_files(); }

FILE* QuantumHeatMoments::open_segment_file(const char* filename)
{
  FILE* file = my_fopen(filename, "a");
  std::fprintf(file,
    "# segment_begin id %s\n# qhm_segment_metadata_version 2\n# qhm_append_contract %s\n"
    "# qhm_append_contract_fields model_fnv1a64 N P mu0 mu2 mu4 complex_pi weyl_order dynamic a0_only T_start T_end dt fd_r fd_p integrate_type sample_interval static_interval dynamic_interval candidate_interval n_moyal n_trace n_aux\n"
    "# analysis_rule group_numeric_rows_by_segment_id\n"
    "# force_model_filename %s\n# force_model_fnv1a64 %016llx\n"
    "# N %d\n# P %d\n# exact_mu0 %d\n# exact_mu2 %d\n# exact_mu4 %d\n# weyl_max_order %d\n",
    segment_id_.c_str(), append_contract_.c_str(), model_path_storage_.c_str(), model_fingerprint_,
    number_of_atoms_, number_of_beads_, exact_mu0_enabled_ ? 1 : 0,
    exact_mu2_enabled_ ? 1 : 0, exact_mu4_enabled_ ? 1 : 0, weyl_max_order_);
  return file;
}

void QuantumHeatMoments::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (integrate.type != 31 && integrate.type != 33)
    PRINT_INPUT_ERROR("compute_quantum_heat_moments supports RPMD NVE or PIMD only.");
  dynamic_enabled_ = integrate.type == 31;
  pimd_a0_only_enabled_ = integrate.type == 33 &&
    !(exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_);
  int active_interval = std::numeric_limits<int>::max();
  if (exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_)
    active_interval = std::min(active_interval, static_sample_interval_);
  if (dynamic_enabled_) active_interval = std::min(active_interval, dynamic_sample_interval_);
  if (enable_complex_pi_) active_interval = std::min(active_interval, candidate_sample_interval_);
  if (pimd_a0_only_enabled_) active_interval = std::min(active_interval, sample_interval_);
  if (validate_edge_derivatives_ || validate_fd_smoothness_ || validate_link_images_ || output_level_ == 2)
    active_interval = std::min(active_interval, sample_interval_);
  if (active_interval == std::numeric_limits<int>::max() || active_interval > number_of_steps)
    PRINT_INPUT_ERROR("compute_quantum_heat_moments has no samples in this run.");
  if ((exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_) && static_sample_interval_ > number_of_steps)
    PRINT_INPUT_ERROR("static_sample_interval produces no exact-static samples in this run.");
  if (dynamic_enabled_ && dynamic_sample_interval_ > number_of_steps)
    PRINT_INPUT_ERROR("dynamic_sample_interval produces no RPMD-current samples in this run.");
  if (enable_complex_pi_ && candidate_sample_interval_ > number_of_steps)
    PRINT_INPUT_ERROR("candidate_sample_interval produces no candidate samples in this run.");
  if ((validate_edge_derivatives_ || validate_fd_smoothness_ || validate_link_images_ || output_level_ == 2) &&
      sample_interval_ > number_of_steps)
    PRINT_INPUT_ERROR("sample_interval produces no requested QHM diagnostic/debug samples in this run.");
  if (atom.number_of_beads < 2 || atom.position_beads.size() != static_cast<size_t>(atom.number_of_beads) ||
      atom.velocity_beads.size() != static_cast<size_t>(atom.number_of_beads))
    PRINT_INPUT_ERROR("compute_quantum_heat_moments requires complete ring-polymer bead arrays.");
  if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1)
    PRINT_INPUT_ERROR("compute_quantum_heat_moments requires three-dimensional PBC.");
  if (force.potentials.size() != 1 || dynamic_cast<NEP*>(force.potentials[0].get()) == nullptr) {
    PRINT_INPUT_ERROR("compute_quantum_heat_moments requires one pure NEP potential; qNEP is rejected by default.");
  }
  if (force.compute_hnemd_ || force.compute_hnemdec_ != -1)
    PRINT_INPUT_ERROR("compute_quantum_heat_moments does not support HNEMD or HNEMDEC driven sampling.");
  number_of_atoms_ = atom.number_of_atoms;
  number_of_beads_ = atom.number_of_beads;
  number_of_steps_ = number_of_steps;
  time_step_ = time_step;
  profile_sample_count_ = 0;
  profile_energy_only_nep_calls_ = 0;
  profile_derivative_nep_calls_ = 0;
  profile_energy_only_nep_seconds_ = 0.0;
  profile_derivative_nep_seconds_ = 0.0;
  profile_energy_only_geometry_calls_ = 0;
  profile_derivative_geometry_calls_ = 0;
  profile_energy_only_geometry_seconds_ = 0.0;
  profile_derivative_geometry_seconds_ = 0.0;
  profile_derivative_edge_extract_seconds_ = 0.0;
  profile_candidate_seconds_ = 0.0;
  profile_hac_lookup_seconds_ = 0.0;
  profile_sample_wall_seconds_ = 0.0;
  static_profile_sample_count_ = 0;
  static_profile_nep_eval_requested_ = 0;
  static_profile_nep_eval_executed_ = 0;
  static_profile_nep_eval_cache_hits_ = 0;
  static_profile_geometry_eval_requested_ = 0;
  static_profile_geometry_eval_executed_ = 0;
  winding_sample_count_ = 0;
  closest_winding_violation_total_ = 0;
  temperature_start_ = integrate.type == 33 ? integrate.temperature1 : integrate.temperature2;
  temperature_end_ = integrate.temperature2;
  if (!(temperature_start_ > 0.0) || !std::isfinite(temperature_start_) ||
      !(temperature_end_ > 0.0) || !std::isfinite(temperature_end_))
    PRINT_INPUT_ERROR("compute_quantum_heat_moments requires a positive ring-polymer temperature.");
  const bool constant_temperature = integrate.temperature1 == integrate.temperature2;
  const bool has_pressure_control = integrate.num_target_pressure_components > 0;
  const bool canonical_primitive_pimd = quantum_heat_moments::is_canonical_primitive_pimd(
    integrate.type == 33, constant_temperature, has_pressure_control,
    integrate.use_scr_barostat, integrate.use_eco_pimd);
  diagnostic_noncanonical_ = !canonical_primitive_pimd;
  if (dynamic_enabled_) {
    pimd_action_scheme_ = "not_applicable_RPMD";
  } else if (integrate.use_scr_barostat) {
    pimd_action_scheme_ = integrate.use_eco_pimd ? "pimd_scr_eco" : "pimd_scr";
  } else if (has_pressure_control) {
    pimd_action_scheme_ = integrate.use_eco_pimd ? "pimd_npt_eco" : "pimd_npt";
  } else if (integrate.use_eco_pimd) {
    pimd_action_scheme_ = "pimd_eco";
  } else {
    pimd_action_scheme_ = constant_temperature ? "primitive_symmetric" :
      "primitive_symmetric_variable_temperature";
  }
  if ((exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_ || enable_complex_pi_) &&
      !canonical_primitive_pimd)
    PRINT_INPUT_ERROR(
      "exact static estimators require constant-temperature NVT primitive-frequency PIMD without SCR or Eco-PIMD.");
  if ((exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_) && integrate.type != 33)
    PRINT_INPUT_ERROR("exact static moments are available only for canonical primitive PIMD.");
  if (output_level_ == 2 && number_of_atoms_ > 128 && !force_debug_large_)
    PRINT_INPUT_ERROR("debug output for N>128 requires force_debug_large yes.");

  auto* active_nep = dynamic_cast<NEP*>(force.potentials[0].get());
  if (!active_nep->supports_local_edge_derivatives())
    PRINT_INPUT_ERROR("NEP with DFTD3 or ZBL corrections has no supported edge-resolved current path.");
  model_path_storage_ = force.primary_nep_model_path();
  model_fingerprint_ = model_fingerprint(model_path_storage_.c_str());
  segment_id_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
  append_contract_ = std::to_string(model_fingerprint_) + " " +
    std::to_string(number_of_atoms_) + " " + std::to_string(number_of_beads_) + " " +
    std::to_string(exact_mu0_enabled_) + " " + std::to_string(exact_mu2_enabled_) + " " +
    std::to_string(exact_mu4_enabled_) + " " + std::to_string(enable_complex_pi_) + " " +
    std::to_string(weyl_max_order_) + " " + std::to_string(dynamic_enabled_) + " " +
    std::to_string(pimd_a0_only_enabled_);
  char numeric_contract[160];
  std::snprintf(numeric_contract, sizeof(numeric_contract), " %.17g %.17g %.17g %.17g %.17g",
    temperature_start_, temperature_end_, time_step_, fd_step_r_, fd_step_p_);
  append_contract_ += numeric_contract;
  append_contract_ += " " + std::to_string(integrate.type) + " " + std::to_string(sample_interval_) +
    " " + std::to_string(static_sample_interval_) + " " + std::to_string(dynamic_sample_interval_) +
    " " + std::to_string(candidate_sample_interval_) + " " + std::to_string(n_moyal_probe_) +
    " " + std::to_string(n_static_trace_probe_) + " " + std::to_string(n_aux_probe_);
  cutoff_ = active_nep->rc;
  nep_sr_.reset(new NEP(model_path_storage_.c_str(), number_of_atoms_));
  if (!nep_sr_->supports_local_edge_derivatives())
    PRINT_INPUT_ERROR("diagnostic NEP model includes a non-edge-resolved correction.");
  nep_sr_->enable_local_edge_derivatives();
  nep_sr_->set_neighbor_rebuild(true);
  nep_sr_->set_neighbor_log_enabled(false);

  box_ = &box;
  box.get_inverse();
  box.set_is_orthogonal();
  shortest_lattice_vector_ = quantum_heat_moments::shortest_lattice_vector(lattice_matrix(box));
  cutoff_image_ratio_ = 2.0 * cutoff_ / shortest_lattice_vector_;
  unique_image_safe_ = cutoff_image_ratio_ < 1.0;
  if (!unique_image_safe_) {
    std::fprintf(stderr,
      "Warning: NEP cutoff is not image-unique: 2rc/shortest_lattice_vector = %.8g.\n",
      cutoff_image_ratio_);
  }

  mass_by_atom_.resize(number_of_atoms_);
  atom.mass.copy_to_host(mass_by_atom_.data());
  mass_by_dof_.resize(static_cast<size_t>(number_of_atoms_) * 3);
  for (int mu = 0; mu < 3; ++mu) {
    for (int i = 0; i < number_of_atoms_; ++i) {
      if (!(mass_by_atom_[i] > 0.0) || !std::isfinite(mass_by_atom_[i]))
        PRINT_INPUT_ERROR("compute_quantum_heat_moments requires positive finite masses.");
      mass_by_dof_[i + mu * number_of_atoms_] = mass_by_atom_[i];
    }
  }
  type_gpu_.resize(number_of_atoms_);
  type_gpu_.copy_from_device(atom.type.data());
  atom_type_host_.resize(number_of_atoms_);
  atom.type.copy_to_host(atom_type_host_.data());
  position_gpu_.resize(static_cast<size_t>(number_of_atoms_) * 3);
  potential_gpu_.resize(number_of_atoms_);
  force_gpu_.resize(static_cast<size_t>(number_of_atoms_) * 3);
  virial_gpu_.resize(static_cast<size_t>(number_of_atoms_) * 9);
  bead_position_host_.assign(number_of_beads_, std::vector<double>(static_cast<size_t>(number_of_atoms_) * 3));
  bead_velocity_host_.assign(number_of_beads_, std::vector<double>(static_cast<size_t>(number_of_atoms_) * 3));
  for (auto& by_order : imaginary_stats_)
    for (auto& by_alpha : by_order) by_alpha.resize(std::min(imag_lag_max_, number_of_beads_ - 1) + 1);

  const bool any_exact_static = exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_;
  bool appending_existing_output = false;
  const auto check_output = [&](const char* filename, const bool enabled) {
    if (enabled) appending_existing_output =
      existing_segmented_output(filename, append_contract_) || appending_existing_output;
  };
  check_output("quantum_heat_meta.out", true);
  check_output("quantum_heat_estimator_stats.out", any_exact_static || enable_complex_pi_);
  check_output("quantum_heat_imaginary.out", any_exact_static);
  check_output("quantum_heat_static.out", any_exact_static || pimd_a0_only_enabled_);
  check_output("quantum_heat_dynamic.out", dynamic_enabled_);
  check_output("quantum_heat_atom_debug.out", output_level_ == 2);
  check_output("quantum_heat_winding.out", validate_link_images_ || output_level_ == 2);
  check_output("quantum_heat_profile.out", any_exact_static);
  check_output("quantum_heat_edgecheck.out", validate_edge_derivatives_ || validate_fd_smoothness_);
  check_output("quantum_heat_fdcheck.out", any_exact_static || (enable_complex_pi_ && weyl_max_order_ > 0));
  check_output("quantum_heat_link.out", any_exact_static && output_level_ >= 1);
  if (appending_existing_output)
    std::fprintf(stderr,
      "Warning: appending QHM output segment %s; analyze files by segment id and per-segment columns.\n",
      segment_id_.c_str());
  meta_file_ = my_fopen("quantum_heat_meta.out", "a");
  if (any_exact_static || enable_complex_pi_)
    estimator_stats_file_ = open_segment_file("quantum_heat_estimator_stats.out");
  if (any_exact_static) imaginary_file_ = open_segment_file("quantum_heat_imaginary.out");
  if (any_exact_static || pimd_a0_only_enabled_)
    static_file_ = open_segment_file("quantum_heat_static.out");
  if (dynamic_enabled_) dynamic_file_ = open_segment_file("quantum_heat_dynamic.out");
  if (output_level_ == 2) atom_debug_file_ = open_segment_file("quantum_heat_atom_debug.out");
  if (validate_link_images_ || output_level_ == 2)
    winding_file_ = open_segment_file("quantum_heat_winding.out");
  if (any_exact_static) profile_file_ = open_segment_file("quantum_heat_profile.out");

  const double P = number_of_beads_;
  const double N = number_of_atoms_;
  const double mu0_evals = exact_mu0_enabled_ ?
    P * 3.0 * 2.0 * (1.0 + 2.0 * n_static_trace_probe_ + 48.0 * N) : 0.0;
  const double mu2_evals = exact_mu2_enabled_ ?
    P * 3.0 * 2.0 * (6.0 + 8.0 * n_moyal_probe_ + 44.0 * n_static_trace_probe_) : 0.0;
  const double mu4_evals = exact_mu4_enabled_ ?
    P * 3.0 * 2.0 * ((39.0 + 48.0 * N) + n_static_trace_probe_ * (568.0 + 96.0 * N)) : 0.0;
  estimated_nep_evaluations_per_static_frame_ = mu0_evals + mu2_evals + mu4_evals;
  estimated_geometry_evaluations_per_static_frame_ = estimated_nep_evaluations_per_static_frame_;
  if (any_exact_static && estimated_nep_evaluations_per_static_frame_ > expensive_static_warning_threshold_)
    std::fprintf(stderr,
      "Warning: estimated QHM static NEP geometry requests/frame %.0f exceed configured warning threshold %.0f; the no-cache count includes full NEP computations and is not an executed-call prediction.\n",
      estimated_nep_evaluations_per_static_frame_, expensive_static_warning_threshold_);
  write_headers();
  write_meta(box, temperature_start_);
}

std::shared_ptr<QuantumHeatMoments::Geometry> QuantumHeatMoments::evaluate_geometry(
  const std::vector<double>& position,
  bool with_derivatives,
  const bool retain_edges)
{
  if (retain_edges) with_derivatives = true;
  ++geometry_eval_requested_;
  ++nep_eval_requested_;
  const auto geometry_begin = std::chrono::steady_clock::now();
  const auto record_geometry_evaluation = [this, with_derivatives, geometry_begin]() {
    const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - geometry_begin).count();
    if (with_derivatives) {
      ++derivative_geometry_calls_;
      derivative_geometry_seconds_ += seconds;
    } else {
      ++energy_only_geometry_calls_;
      energy_only_geometry_seconds_ += seconds;
    }
  };
  std::shared_ptr<Geometry> geometry;
  for (auto it = geometry_cache_.rbegin(); it != geometry_cache_.rend(); ++it) {
    const auto& cached_geometry = *it;
    if (cached_geometry->position == position) {
      geometry = cached_geometry;
      break;
    }
  }
  if (geometry && (!with_derivatives || geometry->has_derivatives) &&
      (!retain_edges || geometry->has_edges)) {
    ++geometry_eval_cache_hits_;
    ++nep_eval_cache_hits_;
    record_geometry_evaluation();
    return geometry;
  }
  ++geometry_eval_executed_;
  ++nep_eval_executed_;
  const bool cache_new_geometry = !geometry;
  if (cache_new_geometry) {
    geometry.reset(new Geometry());
    geometry->position = position;
  }

  const auto nep_setup_compute_potential_copy_begin = std::chrono::steady_clock::now();
  position_gpu_.copy_from_host(position.data());
  potential_gpu_.fill(0.0);
  force_gpu_.fill(0.0);
  virial_gpu_.fill(0.0);
  nep_sr_->set_local_edge_derivatives_enabled(with_derivatives);
  nep_sr_->compute(*box_, type_gpu_, position_gpu_, potential_gpu_, force_gpu_, virial_gpu_);
  geometry->potential.resize(number_of_atoms_);
  potential_gpu_.copy_to_host(geometry->potential.data());
  const double nep_setup_compute_potential_copy_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - nep_setup_compute_potential_copy_begin).count();
  if (with_derivatives) {
    ++derivative_nep_calls_;
    derivative_nep_seconds_ += nep_setup_compute_potential_copy_seconds;
  } else {
    ++energy_only_nep_calls_;
    energy_only_nep_seconds_ += nep_setup_compute_potential_copy_seconds;
  }
  geometry->total_potential = std::accumulate(geometry->potential.begin(), geometry->potential.end(), 0.0);
  if (with_derivatives) {
    const auto derivative_extract_begin = std::chrono::steady_clock::now();
    geometry->force.resize(static_cast<size_t>(number_of_atoms_) * 3);
    force_gpu_.copy_to_host(geometry->force.data());
    nep_sr_->copy_local_energy_edges(*box_, position, geometry->edges);
    geometry->transport.assign(static_cast<size_t>(number_of_atoms_) * 9, 0.0);
    for (const NEP_Local_Edge& edge : geometry->edges) {
      for (int mu = 0; mu < 3; ++mu) {
        for (int alpha = 0; alpha < 3; ++alpha) {
          geometry->transport[(edge.neighbor * 3 + mu) * 3 + alpha] -=
            edge.displacement[alpha] * edge.derivative[mu];
        }
      }
    }
    geometry->has_edges = retain_edges;
    if (!retain_edges)
      std::vector<NEP_Local_Edge>().swap(geometry->edges);
    geometry->has_derivatives = true;
    derivative_edge_extract_seconds_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - derivative_extract_begin).count();
  }
  if (cache_new_geometry) {
    geometry_cache_.push_back(geometry);
    if (geometry_cache_.size() > 64) geometry_cache_.pop_front();
  }
  record_geometry_evaluation();
  return geometry;
}

std::vector<double> QuantumHeatMoments::linear_coefficients(const Geometry& geometry, const int alpha) const
{
  std::vector<double> a(static_cast<size_t>(number_of_atoms_) * 3);
  for (int j = 0; j < number_of_atoms_; ++j) {
    for (int mu = 0; mu < 3; ++mu) {
      const size_t d = static_cast<size_t>(j + mu * number_of_atoms_);
      a[d] = ((mu == alpha ? geometry.potential[j] : 0.0) +
        geometry.transport[(j * 3 + mu) * 3 + alpha]) / mass_by_atom_[j];
    }
  }
  return a;
}

QuantumHeatMoments::A0Parts QuantumHeatMoments::evaluate_A0_parts(
  const std::vector<double>& position,
  const std::vector<Complex>& momentum,
  const int alpha,
  const double)
{
  if (momentum.size() != mass_by_dof_.size())
    PRINT_INPUT_ERROR("A0 phase-space dimension mismatch.");
  const std::shared_ptr<Geometry> geometry = evaluate_geometry(position);
  const std::vector<double> a = linear_coefficients(*geometry, alpha);
  A0Parts result;
  for (int d = 0; d < static_cast<int>(momentum.size()); ++d) result.linear += a[d] * momentum[d];
  for (int i = 0; i < number_of_atoms_; ++i) {
    Complex p_squared = 0.0;
    for (int mu = 0; mu < 3; ++mu) {
      const Complex p = momentum[i + mu * number_of_atoms_];
      p_squared += p * p;
    }
    result.cubic += p_squared * momentum[i + alpha * number_of_atoms_] /
      (2.0 * mass_by_atom_[i] * mass_by_atom_[i]);
  }
  return result;
}

Complex QuantumHeatMoments::evaluate_A0(
  const std::vector<double>& position,
  const std::vector<Complex>& momentum,
  const int alpha,
  const double fd_step_r)
{
  const A0Parts parts = evaluate_A0_parts(position, momentum, alpha, fd_step_r);
  return parts.linear + parts.cubic;
}

double QuantumHeatMoments::energy(const std::vector<double>& position)
{
  return evaluate_geometry(position, false)->total_potential;
}

std::vector<double> QuantumHeatMoments::force(const std::vector<double>& position)
{
  return evaluate_geometry(position)->force;
}

double QuantumHeatMoments::evaluate_B(
  const std::vector<double>& position,
  const std::vector<double>& left,
  const std::vector<double>& right,
  const int alpha,
  const double step)
{
  const size_t D = mass_by_dof_.size();
  std::vector<double> direction_left(D), direction_right(D);
  double norm_left = 0.0, norm_right = 0.0;
  for (size_t d = 0; d < D; ++d) {
    direction_left[d] = left[d] / mass_by_dof_[d];
    direction_right[d] = right[d] / mass_by_dof_[d];
    norm_left += direction_left[d] * direction_left[d];
    norm_right += direction_right[d] * direction_right[d];
  }
  norm_left = std::sqrt(norm_left);
  norm_right = std::sqrt(norm_right);
  double directional_left = 0.0, directional_right = 0.0;
  if (norm_left > 0.0) {
    std::vector<double> plus = position, minus = position;
    for (size_t d = 0; d < D; ++d) {
      plus[d] += step * direction_left[d] / norm_left;
      minus[d] -= step * direction_left[d] / norm_left;
    }
    directional_left = norm_left *
      (dot(linear_coefficients(*evaluate_geometry(plus), alpha), right) -
       dot(linear_coefficients(*evaluate_geometry(minus), alpha), right)) / (2.0 * step);
  }
  if (norm_right > 0.0) {
    std::vector<double> plus = position, minus = position;
    for (size_t d = 0; d < D; ++d) {
      plus[d] += step * direction_right[d] / norm_right;
      minus[d] -= step * direction_right[d] / norm_right;
    }
    directional_right = norm_right *
      (dot(linear_coefficients(*evaluate_geometry(plus), alpha), left) -
       dot(linear_coefficients(*evaluate_geometry(minus), alpha), left)) / (2.0 * step);
  }

  const std::shared_ptr<Geometry> geometry = evaluate_geometry(position);
  double force_contraction = 0.0;
  for (int i = 0; i < number_of_atoms_; ++i) {
    double left_dot_right = 0.0;
    for (int mu = 0; mu < 3; ++mu) {
      const int d = i + mu * number_of_atoms_;
      left_dot_right += left[d] * right[d];
    }
    for (int k = 0; k < 3; ++k) {
      const int d = i + k * number_of_atoms_;
      const double c = (k == alpha ? left_dot_right : 0.0) +
        left[d] * right[i + alpha * number_of_atoms_] +
        right[d] * left[i + alpha * number_of_atoms_];
      const double u_gradient = -geometry->force[d];
      force_contraction += u_gradient * c /
        (mass_by_atom_[i] * mass_by_atom_[i]);
    }
  }
  return 0.5 * (directional_left + directional_right) - 0.5 * force_contraction;
}

double QuantumHeatMoments::evaluate_A1_constant(
  const std::vector<double>& position,
  const int alpha,
  const double step)
{
  const size_t D = mass_by_dof_.size();
  const auto geometry = evaluate_geometry(position);
  const std::vector<double> a = linear_coefficients(*geometry, alpha);
  double value = 0.0;
  for (size_t d = 0; d < D; ++d) value += geometry->force[d] * a[d];
  double kinetic_third = 0.0;
  for (int i = 0; i < number_of_atoms_; ++i) {
    for (int mu = 0; mu < 3; ++mu) {
      const int a_index = i + alpha * number_of_atoms_;
      const int b_index = i + mu * number_of_atoms_;
      std::vector<double> a_direction(D, 0.0), b_direction(D, 0.0);
      a_direction[a_index] = 1.0;
      b_direction[b_index] = 1.0;
      const double u3 = quantum_heat_moments::mixed_directional_derivative<double>(
        position, {a_direction, b_direction, b_direction}, {step, step, step},
        [&](const std::vector<double>& displaced) { return energy(displaced); });
      kinetic_third += 3.0 * u3 / (mass_by_atom_[i] * mass_by_atom_[i]);
    }
  }
  return value + HBAR * HBAR * kinetic_third / 24.0;
}

double QuantumHeatMoments::evaluate_D_contraction(
  const std::vector<double>& position,
  const std::vector<double>& first,
  const std::vector<double>& second,
  const std::vector<double>& third,
  const int alpha,
  const double step_r,
  const double step_p)
{
  (void)step_p;
  auto derivative = [&](const std::vector<double>& direction,
                        const std::vector<double>& left,
                        const std::vector<double>& right) {
    std::vector<double> plus = position, minus = position;
    for (size_t d = 0; d < direction.size(); ++d) {
      const double shift = step_r * direction[d] / mass_by_dof_[d];
      plus[d] += shift;
      minus[d] -= shift;
    }
    return (evaluate_B(plus, left, right, alpha, step_r) -
            evaluate_B(minus, left, right, alpha, step_r)) / (2.0 * step_r);
  };
  return (derivative(first, second, third) + derivative(second, first, third) +
          derivative(third, first, second)) / 3.0;
}

double QuantumHeatMoments::evaluate_E_contraction(
  const std::vector<double>& position,
  const std::vector<double>& direction,
  const int alpha,
  const double step_r,
  const double step_p)
{
  (void)step_p;
  std::vector<double> plus = position, minus = position;
  for (size_t d = 0; d < direction.size(); ++d) {
    const double shift = step_r * direction[d] / mass_by_dof_[d];
    plus[d] += shift;
    minus[d] -= shift;
  }
  const double constant_derivative = (evaluate_A1_constant(plus, alpha, step_r) -
    evaluate_A1_constant(minus, alpha, step_r)) / (2.0 * step_r);
  const std::vector<double> force_at_position = this->force(position);
  return constant_derivative + 2.0 * evaluate_B(position, force_at_position, direction, alpha, step_r);
}

Complex QuantumHeatMoments::evaluate_gamma0(
  const std::vector<double>& position,
  const std::vector<double>& link_displacement,
  const int alpha,
  const double beta,
  const double step,
  const std::vector<MoyalProbe>& trace_probes,
  double& imaginary_residual)
{
  const size_t D = mass_by_dof_.size();
  const double epsilon = beta / number_of_beads_;
  const auto geometry = evaluate_geometry(position);
  const std::vector<double> a = linear_coefficients(*geometry, alpha);
  std::vector<double> u_gradient(D);
  std::vector<double> g(D);
  double a_dot_g = 0.0;
  for (size_t d = 0; d < D; ++d) {
    u_gradient[d] = -geometry->force[d];
    g[d] = mass_by_dof_[d] * link_displacement[d] / (HBAR * HBAR * epsilon) -
      0.5 * epsilon * u_gradient[d];
    a_dot_g += a[d] * g[d];
  }

  double divergence_a = 0.0;
  for (const MoyalProbe& probe : trace_probes) {
    const std::vector<double>& xi = probe.direction[0];
    std::vector<double> plus = position, minus = position;
    for (size_t d = 0; d < D; ++d) {
      plus[d] += step * xi[d];
      minus[d] -= step * xi[d];
    }
    divergence_a += dot(
      linear_coefficients(*evaluate_geometry(plus), alpha), xi) / (2.0 * step);
    divergence_a -= dot(
      linear_coefficients(*evaluate_geometry(minus), alpha), xi) / (2.0 * step);
  }
  if (!trace_probes.empty()) divergence_a /= trace_probes.size();

  double C_R3 = 0.0;
  for (int i = 0; i < number_of_atoms_; ++i) {
    for (int mu = 0; mu < 3; ++mu) {
      const int ia = i + alpha * number_of_atoms_;
      const int ib = i + mu * number_of_atoms_;
      std::vector<double> ea(D, 0.0), eb(D, 0.0);
      ea[ia] = 1.0;
      eb[ib] = 1.0;
      const double U_ab = quantum_heat_moments::mixed_directional_derivative<double>(
        position, {ea, eb}, {step, step},
        [&](const std::vector<double>& displaced) { return energy(displaced); });
      const double U_bb = quantum_heat_moments::mixed_directional_derivative<double>(
        position, {eb, eb}, {step, step},
        [&](const std::vector<double>& displaced) { return energy(displaced); });
      const double U_abb = quantum_heat_moments::mixed_directional_derivative<double>(
        position, {ea, eb, eb}, {step, step, step},
        [&](const std::vector<double>& displaced) { return energy(displaced); });
      const double H_ab = (ia == ib ? -mass_by_dof_[ib] / (HBAR * HBAR * epsilon) : 0.0) -
        0.5 * epsilon * U_ab;
      const double H_bb = -mass_by_dof_[ib] / (HBAR * HBAR * epsilon) - 0.5 * epsilon * U_bb;
      const double r3 = g[ia] * g[ib] * g[ib] + 2.0 * H_ab * g[ib] + H_bb * g[ia] -
        0.5 * epsilon * U_abb;
      C_R3 += 3.0 * r3 / (mass_by_atom_[i] * mass_by_atom_[i]);
    }
  }
  const double gamma_imag = -HBAR * (a_dot_g + 0.5 * divergence_a) +
    HBAR * HBAR * HBAR * C_R3 / 6.0;
  const Complex gamma(0.0, gamma_imag);
  imaginary_residual = std::fabs(gamma.real()) / std::max(std::fabs(gamma.imag()), 1.0e-30);
  return gamma;
}

double QuantumHeatMoments::evaluate_gamma1(
  const std::vector<double>& position,
  const std::vector<double>& link_displacement,
  const int alpha,
  const double beta,
  const double step,
  const std::vector<MoyalProbe>& moyal_probes,
  const std::vector<MoyalProbe>& trace_probes,
  ComplexStats& moyal_stats,
  ComplexStats& trace_stats,
  std::vector<Complex>& moyal_samples,
  std::vector<Complex>& trace_samples)
{
  const size_t D = mass_by_dof_.size();
  const double epsilon = beta / number_of_beads_;
  const std::shared_ptr<Geometry> geometry = evaluate_geometry(position);
  const std::vector<double> a = linear_coefficients(*geometry, alpha);
  std::vector<double> U_gradient(D), g(D);
  double C0 = 0.0;
  for (size_t d = 0; d < D; ++d) {
    U_gradient[d] = -geometry->force[d];
    C0 -= U_gradient[d] * a[d];
    g[d] = mass_by_dof_[d] * link_displacement[d] /
      (HBAR * HBAR * epsilon) - 0.5 * epsilon * U_gradient[d];
  }

  moyal_samples.resize(moyal_probes.size());
  for (size_t probe_index = 0; probe_index < moyal_probes.size(); ++probe_index) {
    const MoyalProbe& probe = moyal_probes[probe_index];
    const std::vector<std::vector<double>> directions = {
      probe.direction[0], probe.direction[1], probe.direction[2]};
    const double U3 = quantum_heat_moments::mixed_directional_derivative<double>(
      position, directions, {step, step, step},
      [&](const std::vector<double>& displaced) { return energy(displaced); });
    const double C3 = kinetic_cubic_third_contraction(
      directions[0], directions[1], directions[2], mass_by_atom_, number_of_atoms_, alpha);
    moyal_samples[probe_index] = Complex(U3 * C3, 0.0);
    moyal_stats.add(moyal_samples[probe_index]);
  }
  if (moyal_stats.count > 0) C0 += HBAR * HBAR * moyal_stats.mean.real() / 24.0;

  const double Bgg = evaluate_B(position, g, g, alpha, step);
  double trace_sum = 0.0;
  trace_samples.resize(trace_probes.size());
  for (size_t probe_index = 0; probe_index < trace_probes.size(); ++probe_index) {
    const MoyalProbe& probe = trace_probes[probe_index];
    const std::vector<double>& xi = probe.direction[0];
    const std::vector<double>& eta = probe.direction[1];
    std::vector<double> mass_xi(D);
    for (size_t d = 0; d < D; ++d) mass_xi[d] = std::sqrt(mass_by_dof_[d]) * xi[d];
    const double trace_mass = evaluate_B(position, mass_xi, mass_xi, alpha, step);
    const double B_xi_eta = evaluate_B(position, xi, eta, alpha, step);
    const double U_xi_eta = quantum_heat_moments::mixed_directional_derivative<double>(
      position, {xi, eta}, {step, step},
      [&](const std::vector<double>& displaced) { return energy(displaced); });

    std::vector<double> plus = position, minus = position;
    for (size_t d = 0; d < D; ++d) {
      plus[d] += step * xi[d];
      minus[d] -= step * xi[d];
    }
    const double div_B_g =
      (evaluate_B(plus, xi, g, alpha, step) - evaluate_B(minus, xi, g, alpha, step)) /
      (2.0 * step);

    std::vector<double> pp = position, pm = position, mp = position, mm = position;
    for (size_t d = 0; d < D; ++d) {
      pp[d] += step * (xi[d] + eta[d]);
      pm[d] += step * (xi[d] - eta[d]);
      mp[d] += step * (-xi[d] + eta[d]);
      mm[d] -= step * (xi[d] + eta[d]);
    }
    const double div_div_B =
      (evaluate_B(pp, xi, eta, alpha, step) - evaluate_B(pm, xi, eta, alpha, step) -
       evaluate_B(mp, xi, eta, alpha, step) + evaluate_B(mm, xi, eta, alpha, step)) /
      (4.0 * step * step);
    const double sample = -trace_mass / (HBAR * HBAR * epsilon) -
      0.5 * epsilon * B_xi_eta * U_xi_eta + div_B_g + 0.25 * div_div_B;
    trace_samples[probe_index] = Complex(sample, 0.0);
    trace_stats.add(trace_samples[probe_index]);
    trace_sum += sample;
  }
  const double trace_mean = trace_probes.empty() ? 0.0 : trace_sum / trace_probes.size();
  return C0 - HBAR * HBAR * (Bgg + trace_mean);
}

Complex QuantumHeatMoments::evaluate_gamma2(
  const std::vector<double>& position,
  const std::vector<double>& link_displacement,
  const int alpha,
  const double beta,
  const double step_r,
  const double step_p,
  const std::vector<MoyalProbe>& probes,
  double& imaginary_residual,
  ComplexStats& contraction_stats)
{
  const size_t Dof = mass_by_dof_.size();
  const double epsilon = beta / number_of_beads_;
  const auto geometry = evaluate_geometry(position);
  const std::vector<double> force_at_position = geometry->force;
  std::vector<double> g(Dof);
  for (size_t d = 0; d < Dof; ++d) {
    const double U_gradient = -force_at_position[d];
    g[d] = mass_by_dof_[d] * link_displacement[d] / (HBAR * HBAR * epsilon) -
      0.5 * epsilon * U_gradient;
  }
  const auto h_times = [&](const std::vector<double>& direction) {
    std::vector<double> plus = position, minus = position;
    for (size_t d = 0; d < Dof; ++d) {
      plus[d] += step_r * direction[d];
      minus[d] -= step_r * direction[d];
    }
    const std::vector<double> force_plus = force(plus);
    const std::vector<double> force_minus = force(minus);
    std::vector<double> result(Dof);
    for (size_t d = 0; d < Dof; ++d) {
      result[d] = -mass_by_dof_[d] * direction[d] / (HBAR * HBAR * epsilon) +
        0.5 * epsilon * (force_plus[d] - force_minus[d]) / (2.0 * step_r);
    }
    return result;
  };
  const auto q_derivative = [&](const std::vector<std::vector<double>>& directions,
                                const auto& evaluate) {
    return quantum_heat_moments::mixed_directional_derivative<double>(
      position, directions, std::vector<double>(directions.size(), step_r), evaluate);
  };
  const double D_ggg = evaluate_D_contraction(position, g, g, g, alpha, step_r, step_p);
  const double E_g = evaluate_E_contraction(position, g, alpha, step_r, step_p);
  ComplexStats samples;
  for (const MoyalProbe& probe : probes) {
    const std::vector<double>& xi = probe.direction[0];
    const std::vector<double>& eta = probe.direction[1];
    const std::vector<double>& zeta = probe.direction[2];
    const std::vector<double> H_eta = h_times(eta);
    const std::vector<double> H_xi = h_times(xi);
    const double D_H_xi_xi_g = evaluate_D_contraction(
      position, H_xi, xi, g, alpha, step_r, step_p);
    const double K_xi_eta_zeta = -0.5 * epsilon *
      quantum_heat_moments::mixed_directional_derivative<double>(
        position, {xi, eta, zeta}, {step_r, step_r, step_r},
        [&](const std::vector<double>& displaced) { return energy(displaced); });
    const double D_xi_eta_zeta = evaluate_D_contraction(
      position, xi, eta, zeta, alpha, step_r, step_p);

    const double D_derivative_gg = q_derivative({xi}, [&](const std::vector<double>& displaced) {
      return evaluate_D_contraction(displaced, xi, g, g, alpha, step_r, step_p);
    });
    const double D_derivative_H = q_derivative({xi}, [&](const std::vector<double>& displaced) {
      return evaluate_D_contraction(displaced, xi, H_eta, eta, alpha, step_r, step_p);
    });
    const double D_second_derivative = q_derivative({xi, eta}, [&](const std::vector<double>& displaced) {
      return evaluate_D_contraction(displaced, xi, eta, g, alpha, step_r, step_p);
    });
    const double D_third_derivative = q_derivative({xi, eta, zeta}, [&](const std::vector<double>& displaced) {
      return evaluate_D_contraction(displaced, xi, eta, zeta, alpha, step_r, step_p);
    });
    const double E_divergence = q_derivative({xi}, [&](const std::vector<double>& displaced) {
      return evaluate_E_contraction(displaced, xi, alpha, step_r, step_p);
    });
    const double d_bracket = 3.0 * D_H_xi_xi_g + D_xi_eta_zeta * K_xi_eta_zeta +
      1.5 * (D_derivative_gg + D_derivative_H) +
      0.75 * D_second_derivative + 0.125 * D_third_derivative;
    const double sample = HBAR * HBAR * HBAR * (D_ggg + d_bracket) -
      HBAR * (E_g + 0.5 * E_divergence);
    if (!samples.add_if_finite(Complex(sample, 0.0))) {
      contraction_stats = ComplexStats();
      return Complex(std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    }
  }
  if (samples.count == 0) {
    contraction_stats = ComplexStats();
    return Complex(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
  }
  contraction_stats = samples;
  const Complex gamma(0.0, samples.mean.real());
  imaginary_residual = std::fabs(gamma.real()) / std::max(std::fabs(gamma.imag()), 1.0e-30);
  return gamma;
}

void QuantumHeatMoments::reconstruct_bead_chain(
  const Box& box,
  const std::vector<std::vector<double>>& positions,
  RingPath& path,
  const bool compute_link_diagnostics) const
{
  const int P = number_of_beads_;
  const int N = number_of_atoms_;
  const int D = 3 * N;
  path.wrapped.assign(P, std::vector<double>(D));
  path.unwrapped.assign(P, std::vector<double>(D));
  path.link_displacement.assign(P, std::vector<double>(D));
  if (compute_link_diagnostics) {
    path.action_link_lift.assign(static_cast<size_t>(P) * D, 0);
    path.closest_link_lift.assign(static_cast<size_t>(P) * D, 0);
    path.action_winding.assign(D, 0);
    path.closest_winding.assign(D, 0);
  }
  path.centroid.assign(D, 0.0);
  path.nonzero_action_winding_atoms = 0;
  path.max_abs_action_winding_component = 0;
  path.nonzero_closest_winding_atoms = 0;
  path.max_abs_closest_winding_component = 0;
  path.link_image_total = 0;
  path.link_image_mismatch_count = 0;
  path.max_link_length_action = 0.0;
  path.max_link_length_cartesian_closest = 0.0;
  path.unwrapped = positions;
  const std::array<double, 9> h = lattice_matrix(box);
  std::vector<std::array<int, 3>> wrapped_images;
  if (compute_link_diagnostics) wrapped_images.resize(static_cast<size_t>(P) * N);
  for (int s = 0; s < P; ++s) {
    for (int i = 0; i < N; ++i) {
      const double x = positions[s][i];
      const double y = positions[s][i + N];
      const double z = positions[s][i + 2 * N];
      double frac[3] = {
        box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z,
        box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z,
        box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z};
      for (int d = 0; d < 3; ++d) {
        const double image = std::floor(frac[d]);
        if (compute_link_diagnostics)
          wrapped_images[static_cast<size_t>(s) * N + i][d] = static_cast<int>(image);
        frac[d] -= image;
      }
      path.wrapped[s][i] = box.cpu_h[0] * frac[0] + box.cpu_h[1] * frac[1] + box.cpu_h[2] * frac[2];
      path.wrapped[s][i + N] = box.cpu_h[3] * frac[0] + box.cpu_h[4] * frac[1] + box.cpu_h[5] * frac[2];
      path.wrapped[s][i + 2 * N] = box.cpu_h[6] * frac[0] + box.cpu_h[7] * frac[1] + box.cpu_h[8] * frac[2];
    }
  }

  for (int s = 0; s < P; ++s) {
    const int next = (s + 1) % P;
    for (int i = 0; i < N; ++i) {
      std::array<double, 3> native_difference{};
      for (int mu = 0; mu < 3; ++mu) {
        const int d = i + mu * N;
        native_difference[mu] = positions[next][d] - positions[s][d];
        path.link_displacement[s][d] = native_difference[mu];
      }
      if (compute_link_diagnostics) {
        std::array<double, 3> wrapped_difference{};
        for (int mu = 0; mu < 3; ++mu) {
          const int d = i + mu * N;
          wrapped_difference[mu] = path.wrapped[next][d] - path.wrapped[s][d];
        }
        const quantum_heat_moments::ClosestImage closest =
          quantum_heat_moments::closest_cartesian_lattice_image(wrapped_difference, h);
        bool image_mismatch = false;
        for (int mu = 0; mu < 3; ++mu) {
          const int d = i + mu * N;
          const int native_lift = wrapped_images[static_cast<size_t>(next) * N + i][mu] -
            wrapped_images[static_cast<size_t>(s) * N + i][mu];
          const int closest_lift = closest.image[mu];
          path.action_link_lift[static_cast<size_t>(s) * D + d] = native_lift;
          path.closest_link_lift[static_cast<size_t>(s) * D + d] = closest_lift;
          path.action_winding[d] += native_lift;
          path.closest_winding[d] += closest_lift;
          image_mismatch = image_mismatch || native_lift != closest_lift;
        }
        ++path.link_image_total;
        if (image_mismatch) ++path.link_image_mismatch_count;
        path.max_link_length_action = std::max(path.max_link_length_action,
          std::sqrt(native_difference[0] * native_difference[0] +
            native_difference[1] * native_difference[1] + native_difference[2] * native_difference[2]));
        path.max_link_length_cartesian_closest = std::max(
          path.max_link_length_cartesian_closest, closest.length);
      }
    }
  }
  if (compute_link_diagnostics) for (int i = 0; i < N; ++i) {
    bool action_winds = false;
    bool closest_winds = false;
    for (int mu = 0; mu < 3; ++mu) {
      const int d = i + mu * N;
      path.max_abs_action_winding_component = std::max(
        path.max_abs_action_winding_component, std::abs(path.action_winding[d]));
      path.max_abs_closest_winding_component = std::max(
        path.max_abs_closest_winding_component, std::abs(path.closest_winding[d]));
      action_winds = action_winds || path.action_winding[d] != 0;
      closest_winds = closest_winds || path.closest_winding[d] != 0;
    }
    if (action_winds) ++path.nonzero_action_winding_atoms;
    if (closest_winds) ++path.nonzero_closest_winding_atoms;
  }
  for (int s = 0; s < P; ++s)
    for (int d = 0; d < D; ++d) path.centroid[d] += path.unwrapped[s][d] / P;
}

void QuantumHeatMoments::write_headers()
{
  if (static_file_ != nullptr) {
    std::fprintf(static_file_, "# format_version 8\n# columns step time_fs P temperature_K volume_A3 ");
    for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "J_centroid_sr_%c ", alpha_name(a));
    for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "J_beadavg_sr_%c ", alpha_name(a));
    if (exact_mu0_enabled_) {
      for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "est_mu0_open_%c ", alpha_name(a));
      std::fprintf(static_file_, "est_mu0_open_iso ");
    }
    if (exact_mu2_enabled_) {
      for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "est_mu2_open_%c ", alpha_name(a));
      std::fprintf(static_file_, "est_mu2_open_iso ");
    }
    if (exact_mu4_enabled_) {
      for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "est_mu4_open_%c ", alpha_name(a));
      std::fprintf(static_file_, "est_mu4_open_iso ");
    }
    if (exact_mu0_enabled_) {
      for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "est_mu0_open_imag_residual_%c ", alpha_name(a));
      std::fprintf(static_file_, "est_mu0_open_imag_residual_iso ");
    }
    if (exact_mu2_enabled_) {
      for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "est_mu2_open_imag_residual_%c ", alpha_name(a));
      std::fprintf(static_file_, "est_mu2_open_imag_residual_iso ");
    }
    if (exact_mu4_enabled_) {
      for (int a = 0; a < 3; ++a) std::fprintf(static_file_, "est_mu4_open_imag_residual_%c ", alpha_name(a));
      std::fprintf(static_file_, "est_mu4_open_imag_residual_iso ");
    }
    std::fprintf(static_file_, "static_frame_valid static_skip_reason static_numeric_finite static_fd_within_threshold ");
    std::fprintf(static_file_, "\n");
  }
  if (link_file_ != nullptr) {
    std::fprintf(link_file_, "# format_version 3\n# columns step bead_id alpha deltaR_norm ");
    if (exact_mu0_enabled_) std::fprintf(link_file_, "Gamma0_real Gamma0_imag Gamma0_real_residual ");
    if (exact_mu2_enabled_) std::fprintf(link_file_, "Gamma1_real Gamma1_imag Gamma1_imag_residual ");
    if (exact_mu4_enabled_) std::fprintf(link_file_, "Gamma2_real Gamma2_imag Gamma2_real_residual ");
    std::fprintf(link_file_, "\n");
  }
  if (imaginary_file_ != nullptr) {
    std::fprintf(imaginary_file_,
      "# lag_zero_policy unavailable_nan_same_bead_self_product_not_a_validated_same_point_estimator\n"
      "# format_version 5\n# columns order alpha lag lambda_over_beta mean_corr_real mean_corr_imag stderr_naive_real stderr_naive_imag n_frames\n");
  }
  if (estimator_stats_file_ != nullptr) {
    std::fprintf(estimator_stats_file_,
      "# format_version 8\n# exact_static columns order alpha P n_frames mean_real mean_imag variance_complex variance_real variance_imag covariance_real_imag stderr_naive_real stderr_naive_imag mean_abs max_abs n_static_frames_total n_static_frames_used n_static_frames_skipped_nan n_static_frames_skipped_other\n"
      "# candidate_frame columns step A_order alpha n_finite_probes mean_real mean_imag probe_stderr_naive_real probe_stderr_naive_imag probe_frame_complete frame_accepted\n"
      "# candidate_complex_pi columns A_order alpha n_valid_frames mean_real mean_imag variance_real variance_imag covariance_real_imag variance_complex stderr_naive_real stderr_naive_imag mean_abs max_abs n_invalid_frames\n");
  }
  if (fdcheck_file_ != nullptr) {
    std::fprintf(fdcheck_file_,
      "# format_version 8\n# probe_statistics_available mu2_only_other_rows_nan\n"
       "# counter_scope per_alpha_pipeline_candidate_order_rows_repeat_shared_alpha_counts\n"
       "# alpha_wall_scope static_pipeline_candidate_probe_loop_including_warnings_excluding_frame_output_io\n"
      "# columns step time_fs estimator order alpha value_h_real value_h_imag value_h2_real value_h2_imag "
      "value_Richardson_real value_Richardson_imag fd_abs_diff fd_rel_diff moyal_probe_count moyal_mean_real "
      "moyal_mean_imag moyal_stderr_naive trace_probe_count trace_mean_real trace_mean_imag trace_stderr_naive "
      "nep_eval_requested_alpha_pipeline nep_eval_executed_alpha_pipeline nep_eval_cache_hits_alpha_pipeline "
      "geometry_eval_requested_alpha_pipeline geometry_eval_executed_alpha_pipeline alpha_pipeline_wall_seconds\n");
  }
  if (edgecheck_file_ != nullptr) {
    std::fprintf(edgecheck_file_, "# format_version 3\n# columns step center_atom neighbor_atom image_x image_y image_z component check_type has_radial has_angular pair_distance pair_radial_cutoff analytic fd_h fd_h2 fd_h4 fd_h8 fd_Richardson abs_error rel_error truncation_delta_h_h2 small_step_spread_h4_h8 n_matching_images not_coordinate_fd_testable_count\n");
  }
  if (winding_file_ != nullptr) {
    std::fprintf(winding_file_, "# format_version 2\n# columns step record_type bead_id atom_id component action_link_lift cartesian_closest_link_lift action_winding cartesian_closest_winding n_nonzero_action_winding_atoms max_abs_action_winding_component n_nonzero_cartesian_closest_winding_atoms max_abs_cartesian_closest_winding_component link_image_total link_image_mismatch_count max_link_length_action max_link_length_cartesian_closest spring_qhm_link_rule_match\n");
  }
  if (profile_file_ != nullptr) {
    std::fprintf(profile_file_, "# format_version 3\n# columns step P static_snapshot_wall_seconds private_nep_geometry_requests_through_static_snapshot private_nep_computes_through_static_snapshot private_nep_cache_hits_through_static_snapshot cache_hit_rate geometry_requests_through_static_snapshot geometry_evaluations_through_static_snapshot moyal_probe_count static_trace_probe_count estimated_exact_static_no_cache_requests actual_private_nep_computes_through_static_snapshot\n");
  }
  if (dynamic_file_ != nullptr) {
    std::fprintf(dynamic_file_,
      "# format_version 7\n# columns step time_fs J_A0_centroid_sr_x J_A0_centroid_sr_y J_A0_centroid_sr_z "
      "J_A0_beadavg_sr_x J_A0_beadavg_sr_y J_A0_beadavg_sr_z "
      "J_HAC_full_centroid_x J_HAC_full_centroid_y J_HAC_full_centroid_z "
      "delta_J_HAC_full_centroid_minus_A0_centroid_sr_x "
      "delta_J_HAC_full_centroid_minus_A0_centroid_sr_y "
      "delta_J_HAC_full_centroid_minus_A0_centroid_sr_z "
      "private_full_nep_compute_energy_readback_calls private_full_nep_compute_energy_readback_seconds "
      "private_derivative_nep_calls private_derivative_nep_setup_compute_potential_copy_seconds "
      "private_derivative_force_edge_extract_merge_seconds "
      "energy_only_evaluate_geometry_calls energy_only_evaluate_geometry_seconds "
      "derivative_evaluate_geometry_calls derivative_evaluate_geometry_seconds "
      "hac_current_lookup_seconds sample_wall_seconds_before_dynamic_output_flush\n");
  }
  if (atom_debug_file_ != nullptr) {
    std::fprintf(atom_debug_file_,
      "# format_version 1\n# columns step bead_id atom_id neighbor_atom image_x image_y image_z "
      "Ui Fx Fy Fz u_x u_y u_z dUi_du_x dUi_du_y dUi_du_z "
      "W_xx W_xy W_xz W_yx W_yy W_yz W_zx W_zy W_zz "
      "A_xx A_xy A_xz A_yx A_yy A_yz A_zx A_zy A_zz\n");
  }
}

void QuantumHeatMoments::write_meta(const Box& box, const double temperature)
{
  const double beta = 1.0 / (K_B * temperature);
  const bool hac_available = hac_ != nullptr && hac_->centroid_force_source_is_immediate();
  std::fprintf(meta_file_,
    "# segment_begin id %s\n# qhm_segment_metadata_version 2\n# qhm_append_contract %s\n"
    "# qhm_append_contract_fields model_fnv1a64 N P mu0 mu2 mu4 complex_pi weyl_order dynamic a0_only T_start T_end dt fd_r fd_p integrate_type sample_interval static_interval dynamic_interval candidate_interval n_moyal n_trace n_aux\n"
    "# analysis_rule group_numeric_rows_by_segment_id\n"
    "# force_model_fnv1a64 %016llx\n# GPUMD_version 5.8\n# git_commit %s\n# git_commit_full %s\n"
    "# git_describe %s\n# git_dirty %s\n# git_tree_dirty_at_configure %s\n"
    "# source_tree_not_fully_reproducible %s\n",
    segment_id_.c_str(), append_contract_.c_str(), model_fingerprint_,
    GPUMD_GIT_COMMIT, GPUMD_GIT_COMMIT_FULL,
    GPUMD_GIT_DESCRIBE, GPUMD_GIT_DIRTY,
    GPUMD_GIT_DIRTY, std::strcmp(GPUMD_GIT_DIRTY, "no") == 0 ? "no" : "yes");
  std::fprintf(meta_file_, "# force_model_filename %s\n# model_type NEP\n", model_path_storage_.c_str());
  std::fprintf(meta_file_,
    "# N %d\n# P %d\n# T_K %.16e\n# T_start_K %.16e\n# T_end_K %.16e\n"
    "# beta_eV_inverse %.16e\n# epsilon_beta_over_P_eV_inverse %.16e\n# hbar_internal %.16e\n"
    "# static_temperature_rule PIMD_target_at_step_over_number_of_steps_otherwise_end_of_step_temperature\n",
    number_of_atoms_, number_of_beads_, temperature, temperature_start_, temperature_end_, beta,
    beta / number_of_beads_, HBAR);
  std::fprintf(meta_file_, "# box_matrix");
  for (int d = 0; d < 9; ++d) std::fprintf(meta_file_, " %.16e", box.cpu_h[d]);
  std::fprintf(meta_file_, "\n# NEP_cutoff %.16e\n", cutoff_);
  std::fprintf(meta_file_,
    "# shortest_lattice_vector %.16e\n# 2rc_over_shortest_lattice_vector %.16e\n# unique_image_safe %s\n"
    "# box_matrix_and_image_safety_scope initial_box_at_pre_run\n"
    "# edgecheck_image_safety_scope current_box_at_diagnostic_step\n",
    shortest_lattice_vector_, cutoff_image_ratio_, unique_image_safe_ ? "yes" : "no");
  std::fprintf(meta_file_,
    "# winding_mode diagnostic_only\n# static_and_candidate_winding_filter none\n"
    "# action_link_lift native_aligned_endpoint_image_difference\n"
    "# action_winding_definition sum_of_action_link_lifts_over_closed_ring\n"
    "# action_link_lift_sum_telescopes_to_zero yes\n"
    "# cartesian_closest_lift diagnostic_only\n# full_torus_trace no\n# winding_sector_filter none\n"
    "# pimd_spring_link_rule bead0_mic_alignment_then_native_adjacent_stored_difference\n"
    "# qhm_link_rule native_aligned_adjacent_stored_coordinate_difference\n"
    "# spring_qhm_link_pair_match yes\n# spring_link_energy_uses_squared_difference_orientation_insensitive yes\n"
    "# qhm_kernel_link_orientation q_next_minus_q_current\n# spring_qhm_link_rule_match yes\n"
    "# cartesian_closest_lattice_image diagnostic_only_does_not_replace_spring_link yes\n"
    "# local_energy_gauge NEP_local_energy_edge_resolved\n# sample_interval %d\n"
    "# static_sample_interval %d\n# dynamic_sample_interval %d\n# candidate_sample_interval %d\n"
    "# true_energy_only_kernel no\n# energy_evaluation_path full_nep_compute_energy_readback\n"
    "# pimd_a0_only_current_enabled %s\n"
    "# exact_mu0_enabled %s\n# exact_mu2_enabled %s\n# exact_mu4_enabled %s\n# exact_mu6_enabled %s\n"
    "# imaginary_time_lag_zero unavailable_nan_same_bead_self_product_not_validated\n"
    "# gamma_component_residuals analytic_pure_component_by_construction_not_independent_leakage_checks\n"
    "# open_endpoint_imag_residuals analytic_real_by_construction_not_independent_leakage_checks\n"
    "# required_operator_order %d\n"
    "# candidate_complex_pi_frame_weighting equal_weight_per_valid_frame\n"
    "# candidate_complex_pi_static_se within_frame_aux_probe_mc\n"
    "# candidate_complex_pi_frame_requires_all_aux_probes yes\n"
    "# candidate_complex_pi_invalid_frame_policy drop_order_if_any_probe_nonfinite_or_incomplete_or_statistics_overflow\n"
    "# candidate_complex_pi_estimator_stderr_naive across_valid_frame_means_uncorrected_for_time_correlation\n"
    "# dynamic_current_A0_operator centroid_A0_potential_advection_plus_kinetic_cubic\n"
    "# dynamic_current_A0_gauge NEP_local_energy_edge_resolved\n"
    "# dynamic_current_HAC_operator full_classical_heat_current_at_current_centroid\n"
    "# dynamic_current_HAC_gauge NEP_per_atom_potential_and_virial_from_Force_compute\n"
    "# dynamic_current_HAC_availability %s\n"
    "# dynamic_current_difference HAC_minus_A0_descriptive_only_equivalence_not_asserted\n"
    "# hac_baseline_available %s\n# hac_baseline_mode %s\n# hac_sampling_interval %d\n"
    "# qhm_dynamic_sampling_interval %d\n# hac_qhm_step_alignment same_MD_step_only\n"
    "# weyl_max_order %d\n# exact_static_max_order %d\n# static_backend directional\n"
    "# mu4_fdcheck_trace_statistics unavailable_nan_probe_level_values_not_retained\n"
    "# static_fd_within_threshold_rule abs(h-h2)/max(abs(h2),1e-30)_for_each_enabled_order_and_direction\n"
    "# static_fd_within_threshold_values 1_all_within_threshold_0_any_exceeds_or_nonfinite_minus1_not_evaluated\n"
    "# static_fd_within_threshold_scope diagnostic_only_does_not_filter_static_frame_statistics_or_prove_convergence\n"
    "# static_numeric_finite_scope current_and_enabled_static_moments_before_cross_frame_statistics\n"
    "# fdcheck_geometry_counters per_alpha_pipeline_from_row_start_baseline\n"
    "# fdcheck_candidate_counter_scope shared_alpha_pipeline_all_orders_repeated_on_each_order_row\n"
    "# fdcheck_alpha_wall_scope static_pipeline_candidate_probe_loop_including_warnings_excluding_frame_output_io\n"
    "# fd_step_r %.16e\n# fd_step_p %.16e\n# n_moyal_probe %d\n# n_static_trace_probe %d\n"
    "# edgecheck_local_derivatives_enabled %s\n# fd_smoothness_validation %s\n"
    "# edgecheck_step_r %.16e\n# link_image_validation %s\n"
    "# edgecheck_fixed_coordinate_steps h_h2_h4_h8\n"
    "# edgecheck_high_order_probe Ui_third_directional_energy_stencil_on_fixed_geometry\n"
    "# edgecheck_truncation_indicator abs(fd_h_minus_fd_h2)\n"
    "# edgecheck_small_step_spread_indicator abs(fd_h4_minus_fd_h8)_not_a_pure_noise_estimate\n"
    "# cutoff_side_probe nearest_unique_nonself_radial_pair_at_pair_rc_plus_minus_max(4*edgecheck_step,1e-4*pair_rc)\n"
    "# cutoff_side_probe_pair_rc average_of_typewise_radial_cutoffs\n"
    "# cutoff_side_probe_periodic_safety target_radius_plus_3_edgecheck_steps_less_than_shortest_lattice_vector_over_2_else_explicit_skip_row\n"
    "# cutoff_side_probe_no_unique_pair_policy explicit_inside_outside_skip_rows\n"
    "# static_estimate_mu0_no_cache_formula P*3*2*(1+2*n_trace+48*N)\n"
    "# static_estimate_mu2_no_cache_formula P*3*2*(6+8*n_moyal+44*n_trace)\n"
    "# static_estimate_mu4_no_cache_formula P*3*2*((39+48*N)+n_trace*(568+96*N))\n"
    "# estimated_nep_geometry_requests_per_static_frame_no_cache_reuse %.0f\n"
    "# estimated_nep_compute_requests_per_static_frame_no_cache_reuse %.0f\n"
    "# estimated_nep_compute_request_scope exact_mu0_mu2_mu4_only_excludes_A0_current_and_enabled_edgecheck\n"
    "# expensive_static_warning_threshold %.0f\n"
    "# enable_complex_pi %s\n# n_aux_probe %d\n# imag_lag_max %d\n# random_seed %llu\n"
    "# pimd_action_scheme %s\n# PPPM_disabled yes\n# qNEP_long_range_disabled yes\n"
    "# short_range_mechanical_only yes\n# diagnostic_noncanonical %s\n"
    "# frame_autocorrelation_corrected no\n# coordinate_unit angstrom\n# potential_energy_unit eV\n# temperature_unit K\n# time_output_unit fs\n"
    "# internal_energy_unit eV\n# internal_length_unit angstrom\n# internal_mass_unit amu\n"
    "# internal_time_unit sqrt(amu*angstrom^2/eV)\n# internal_time_unit_fs %.16e\n# internal_temperature_unit K\n"
    "# current_unit eV*angstrom/internal_time\n# mu0_unit current^2\n"
    "# mu2_unit current^2/internal_time^2\n# mu4_unit current^2/internal_time^4\n"
    "# smoothness_not_formally_verified\n# production_fd_choice Richardson\n"
    "# production_fd_choice_scope exact_static_only\n# candidate_complex_pi_fd_choice h\n",
    sample_interval_, static_sample_interval_, dynamic_sample_interval_, candidate_sample_interval_,
    pimd_a0_only_enabled_ ? "yes" : "no",
    exact_mu0_enabled_ ? "yes" : "no", exact_mu2_enabled_ ? "yes" : "no",
    exact_mu4_enabled_ ? "yes" : "no", exact_mu6_enabled_ ? "yes" : "no", required_operator_order_,
    hac_available ? "immediate_full_centroid_only" : "unavailable_or_not_full_centroid",
    hac_available ? "yes" : "no", hac_available ? "full_centroid_classical" : "none",
    hac_available ? hac_->sample_interval : 0, dynamic_sample_interval_,
    weyl_max_order_, exact_static_max_order_, fd_step_r_, fd_step_p_, n_moyal_probe_, n_static_trace_probe_,
    validate_edge_derivatives_ ? "yes" : "no", validate_fd_smoothness_ ? "yes" : "no",
    edgecheck_step_, validate_link_images_ ? "yes" : "no",
    estimated_nep_evaluations_per_static_frame_, estimated_geometry_evaluations_per_static_frame_,
    expensive_static_warning_threshold_, enable_complex_pi_ ? "yes" : "no",
    n_aux_probe_, imag_lag_max_, static_cast<unsigned long long>(seed_),
    pimd_action_scheme_.c_str(),
    diagnostic_noncanonical_ ? "yes" : "no", TIME_UNIT_CONVERSION);
  std::fflush(meta_file_);
}

void QuantumHeatMoments::validate_local_edge_derivatives(
  const int step,
  const std::vector<double>& position,
  const Geometry& geometry)
{
  if (edgecheck_file_ == nullptr) {
    edgecheck_file_ = open_segment_file("quantum_heat_edgecheck.out");
    std::fprintf(edgecheck_file_, "# format_version 3\n# columns step center_atom neighbor_atom image_x image_y image_z component check_type has_radial has_angular pair_distance pair_radial_cutoff analytic fd_h fd_h2 fd_h4 fd_h8 fd_Richardson abs_error rel_error truncation_delta_h_h2 small_step_spread_h4_h8 n_matching_images not_coordinate_fd_testable_count\n");
  }
  std::map<std::pair<int, int>, int> image_counts;
  std::set<size_t> selected_set;
  int self_image_count = 0;
  size_t cutoff_reference = geometry.edges.size();
  double cutoff_reference_distance = std::numeric_limits<double>::infinity();
  for (const NEP_Local_Edge& edge : geometry.edges) {
    ++image_counts[{edge.center, edge.neighbor}];
    if (edge.center == edge.neighbor) ++self_image_count;
  }
  for (size_t edge_index = 0; edge_index < geometry.edges.size(); ++edge_index) {
    const NEP_Local_Edge& edge = geometry.edges[edge_index];
    if (edge.center != edge.neighbor && edge.has_radial &&
        image_counts[{edge.center, edge.neighbor}] == 1) {
      const double distance = std::sqrt(edge.displacement[0] * edge.displacement[0] +
        edge.displacement[1] * edge.displacement[1] + edge.displacement[2] * edge.displacement[2]);
      const double pair_cutoff = nep_sr_->get_pair_radial_cutoff(
        atom_type_host_[edge.center], atom_type_host_[edge.neighbor]);
      const double cutoff_distance = std::fabs(distance - pair_cutoff);
      if (cutoff_distance < cutoff_reference_distance) {
        cutoff_reference = edge_index;
        cutoff_reference_distance = cutoff_distance;
      }
    }
  }
  std::vector<size_t> selected;
  bool selected_radial = false, selected_angular = false, selected_multi = false;
  for (size_t e = 0; e < geometry.edges.size(); ++e) {
    const NEP_Local_Edge& edge = geometry.edges[e];
    if (edge.center == edge.neighbor) continue;
    const int matching = image_counts[{edge.center, edge.neighbor}];
    if (!selected_radial && edge.has_radial) {
      if (selected_set.insert(e).second) selected.push_back(e);
      selected_radial = true;
    }
    if (!selected_angular && edge.has_angular) {
      const bool same_pair_already_selected = std::any_of(selected.begin(), selected.end(), [&](const size_t s) {
        return geometry.edges[s].center == edge.center && geometry.edges[s].neighbor == edge.neighbor;
      });
      if (!same_pair_already_selected && selected_set.insert(e).second) selected.push_back(e);
      selected_angular = true;
    }
    if (!selected_multi && matching > 1) {
      if (selected_set.insert(e).second) selected.push_back(e);
      selected_multi = true;
    }
    if (selected_radial && selected_angular && (unique_image_safe_ || selected_multi)) break;
  }
  if (cutoff_reference < geometry.edges.size() && selected_set.insert(cutoff_reference).second)
    selected.push_back(cutoff_reference);
  if (validate_edge_derivatives_ && (!selected_radial || !selected_angular))
    std::fprintf(stderr,
      "Warning: QHM edgecheck at step %d could not select both radial and angular NEP contributions.\n", step);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (validate_fd_smoothness_ && cutoff_reference < geometry.edges.size()) {
    const NEP_Local_Edge& reference = geometry.edges[cutoff_reference];
    const double length = std::sqrt(reference.displacement[0] * reference.displacement[0] +
      reference.displacement[1] * reference.displacement[1] +
      reference.displacement[2] * reference.displacement[2]);
    if (length > 0.0) {
      std::vector<double> direction(static_cast<size_t>(number_of_atoms_) * 3, 0.0);
      const double unit[3] = {reference.displacement[0] / length,
        reference.displacement[1] / length, reference.displacement[2] / length};
      for (int mu = 0; mu < 3; ++mu)
        direction[reference.neighbor + mu * number_of_atoms_] = unit[mu];
      const auto local_energy = [&](const std::vector<double>& displaced) {
        return evaluate_geometry(displaced, false)->potential[reference.center];
      };
      std::array<double, 4> third_derivative{};
      for (int level = 0; level < 4; ++level) {
        const double h = edgecheck_step_ / static_cast<double>(1 << level);
        third_derivative[level] = quantum_heat_moments::mixed_directional_derivative<double>(
          position, {direction, direction, direction}, {h, h, h}, local_energy);
      }
      const int matching = image_counts[{reference.center, reference.neighbor}];
      const double pair_cutoff = nep_sr_->get_pair_radial_cutoff(
        atom_type_host_[reference.center], atom_type_host_[reference.neighbor]);
      std::fprintf(edgecheck_file_, "%d %d %d %d %d %d r Ui_third_directional 1 %d",
        step, reference.center, reference.neighbor, reference.image[0], reference.image[1],
        reference.image[2], reference.has_angular ? 1 : 0);
      print_real(edgecheck_file_, length);
      print_real(edgecheck_file_, pair_cutoff);
      print_real(edgecheck_file_, nan);
      for (const double value : third_derivative) print_real(edgecheck_file_, value);
      const double richardson = (4.0 * third_derivative[1] - third_derivative[0]) / 3.0;
      print_real(edgecheck_file_, richardson);
      print_real(edgecheck_file_, nan);
      print_real(edgecheck_file_, nan);
      print_real(edgecheck_file_, std::fabs(third_derivative[0] - third_derivative[1]));
      print_real(edgecheck_file_, std::fabs(third_derivative[2] - third_derivative[3]));
      std::fprintf(edgecheck_file_, " %d %d\n", matching, self_image_count);
    }
  }
  if (validate_edge_derivatives_) for (const size_t index : selected) {
    const NEP_Local_Edge& edge = geometry.edges[index];
    const int n_matching = image_counts[{edge.center, edge.neighbor}];
    const bool summed_images = n_matching > 1;
    double analytic[3] = {0.0, 0.0, 0.0};
    bool has_radial = false, has_angular = false;
    for (const NEP_Local_Edge& match : geometry.edges) {
      if (match.center != edge.center || match.neighbor != edge.neighbor) continue;
      for (int mu = 0; mu < 3; ++mu) analytic[mu] += match.derivative[mu];
      has_radial = has_radial || match.has_radial;
      has_angular = has_angular || match.has_angular;
    }
    for (int mu = 0; mu < 3; ++mu) {
      const auto local_energy = [&](const std::vector<double>& displaced) {
        return evaluate_geometry(displaced, false)->potential[edge.center];
      };
      const auto finite_difference = [&](const double h) {
        std::vector<double> plus = position, minus = position;
        plus[edge.neighbor + mu * number_of_atoms_] += h;
        minus[edge.neighbor + mu * number_of_atoms_] -= h;
        return (local_energy(plus) - local_energy(minus)) / (2.0 * h);
      };
      const double fd_h = finite_difference(edgecheck_step_);
      const double fd_h2 = finite_difference(0.5 * edgecheck_step_);
      const double fd_h4 = finite_difference(0.25 * edgecheck_step_);
      const double fd_h8 = finite_difference(0.125 * edgecheck_step_);
      const double richardson = (4.0 * fd_h2 - fd_h) / 3.0;
      const double abs_error = std::fabs(richardson - analytic[mu]);
      const double rel_error = abs_error / std::max(std::fabs(analytic[mu]), 1.0e-30);
      const double pair_cutoff = nep_sr_->get_pair_radial_cutoff(
        atom_type_host_[edge.center], atom_type_host_[edge.neighbor]);
      const double distance = std::sqrt(edge.displacement[0] * edge.displacement[0] +
        edge.displacement[1] * edge.displacement[1] + edge.displacement[2] * edge.displacement[2]);
      const char* check_type = index == cutoff_reference ?
        (summed_images ? "cutoff_reference_summed_images" : "cutoff_reference_unique_edge") :
        (summed_images ? "summed_images" : "unique_edge");
      std::fprintf(edgecheck_file_, "%d %d %d %d %d %d %c %s %d %d",
        step, edge.center, edge.neighbor,
        summed_images ? 0 : edge.image[0], summed_images ? 0 : edge.image[1], summed_images ? 0 : edge.image[2],
        "xyz"[mu], check_type,
        has_radial ? 1 : 0, has_angular ? 1 : 0);
      print_real(edgecheck_file_, distance);
      print_real(edgecheck_file_, pair_cutoff);
      print_real(edgecheck_file_, analytic[mu]);
      print_real(edgecheck_file_, fd_h);
      print_real(edgecheck_file_, fd_h2);
      print_real(edgecheck_file_, fd_h4);
      print_real(edgecheck_file_, fd_h8);
      print_real(edgecheck_file_, richardson);
      print_real(edgecheck_file_, abs_error);
      print_real(edgecheck_file_, rel_error);
      print_real(edgecheck_file_, std::fabs(fd_h - fd_h2));
      print_real(edgecheck_file_, std::fabs(fd_h4 - fd_h8));
      std::fprintf(edgecheck_file_, " %d %d\n", n_matching, self_image_count);
    }
  }

  if (validate_fd_smoothness_) {
    const auto write_cutoff_skip = [&](const char* side_name, const char* reason,
                                       const NEP_Local_Edge* reference, const double target_distance,
                                       const double pair_cutoff, const int matching) {
      const int center = reference == nullptr ? -1 : reference->center;
      const int neighbor = reference == nullptr ? -1 : reference->neighbor;
      const int image_x = reference == nullptr ? 0 : reference->image[0];
      const int image_y = reference == nullptr ? 0 : reference->image[1];
      const int image_z = reference == nullptr ? 0 : reference->image[2];
      const int has_radial = reference == nullptr ? 0 : 1;
      const int has_angular = reference == nullptr ? 0 : (reference->has_angular ? 1 : 0);
      std::fprintf(edgecheck_file_, "%d %d %d %d %d %d r cutoff_side_%s_skipped_%s %d %d",
        step, center, neighbor, image_x, image_y, image_z, side_name, reason, has_radial, has_angular);
      print_real(edgecheck_file_, target_distance);
      print_real(edgecheck_file_, pair_cutoff);
      for (int field = 0; field < 10; ++field) print_real(edgecheck_file_, nan);
      std::fprintf(edgecheck_file_, " %d %d\n", matching, self_image_count);
    };
    if (cutoff_reference >= geometry.edges.size()) {
      write_cutoff_skip("inside", "no_unique_pair", nullptr, nan, nan, 0);
      write_cutoff_skip("outside", "no_unique_pair", nullptr, nan, nan, 0);
    } else {
      const NEP_Local_Edge& reference = geometry.edges[cutoff_reference];
      const double length = std::sqrt(reference.displacement[0] * reference.displacement[0] +
        reference.displacement[1] * reference.displacement[1] +
        reference.displacement[2] * reference.displacement[2]);
      const double pair_cutoff = nep_sr_->get_pair_radial_cutoff(
        atom_type_host_[reference.center], atom_type_host_[reference.neighbor]);
      if (length > 0.0) {
        const double unit[3] = {reference.displacement[0] / length,
          reference.displacement[1] / length, reference.displacement[2] / length};
        const double offset = std::max(4.0 * edgecheck_step_, 1.0e-4 * pair_cutoff);
        for (int side = -1; side <= 1; side += 2) {
          const char* side_name = side < 0 ? "inside" : "outside";
          const double target_distance = pair_cutoff + side * offset;
          if (target_distance <= 0.0) {
            write_cutoff_skip(side_name, "nonpositive_radius", &reference, target_distance, pair_cutoff, 1);
            continue;
          }
          if (target_distance + 3.0 * edgecheck_step_ >= 0.5 * shortest_lattice_vector_) {
            write_cutoff_skip(side_name, "periodic_image_boundary", &reference, target_distance, pair_cutoff, 1);
            continue;
          }
          std::vector<double> side_position = position;
          for (int mu = 0; mu < 3; ++mu)
            side_position[reference.neighbor + mu * number_of_atoms_] =
              position[reference.center + mu * number_of_atoms_] + unit[mu] * target_distance;
          const std::shared_ptr<Geometry> side_geometry = evaluate_geometry(side_position, true, true);
          double analytic = 0.0;
          int matching = 0;
          bool has_radial = false, has_angular = false;
          int image[3] = {0, 0, 0};
          for (const NEP_Local_Edge& edge : side_geometry->edges) {
            if (edge.center != reference.center || edge.neighbor != reference.neighbor) continue;
            ++matching;
            for (int mu = 0; mu < 3; ++mu) analytic += edge.derivative[mu] * unit[mu];
            has_radial = has_radial || edge.has_radial;
            has_angular = has_angular || edge.has_angular;
            for (int mu = 0; mu < 3; ++mu) image[mu] = edge.image[mu];
          }
          const auto side_local_energy = [&](const std::vector<double>& displaced) {
            return evaluate_geometry(displaced, false)->potential[reference.center];
          };
          std::array<double, 4> fd{};
          for (int level = 0; level < 4; ++level) {
            const double h = edgecheck_step_ / static_cast<double>(1 << level);
            std::vector<double> plus = side_position, minus = side_position;
            for (int mu = 0; mu < 3; ++mu) {
              plus[reference.neighbor + mu * number_of_atoms_] += unit[mu] * h;
              minus[reference.neighbor + mu * number_of_atoms_] -= unit[mu] * h;
            }
            fd[level] = (side_local_energy(plus) - side_local_energy(minus)) / (2.0 * h);
          }
          const double richardson = (4.0 * fd[1] - fd[0]) / 3.0;
          std::fprintf(edgecheck_file_, "%d %d %d %d %d %d r cutoff_side_%s %d %d",
            step, reference.center, reference.neighbor, image[0], image[1], image[2],
            side_name, has_radial ? 1 : 0, has_angular ? 1 : 0);
          print_real(edgecheck_file_, target_distance);
          print_real(edgecheck_file_, pair_cutoff);
          print_real(edgecheck_file_, analytic);
          for (const double value : fd) print_real(edgecheck_file_, value);
          print_real(edgecheck_file_, richardson);
          print_real(edgecheck_file_, std::fabs(richardson - analytic));
          print_real(edgecheck_file_, std::fabs(richardson - analytic) /
            std::max(std::fabs(analytic), 1.0e-30));
          print_real(edgecheck_file_, std::fabs(fd[0] - fd[1]));
          print_real(edgecheck_file_, std::fabs(fd[2] - fd[3]));
          std::fprintf(edgecheck_file_, " %d %d\n", matching, self_image_count);

          std::vector<double> direction(static_cast<size_t>(number_of_atoms_) * 3, 0.0);
          for (int mu = 0; mu < 3; ++mu)
            direction[reference.neighbor + mu * number_of_atoms_] = unit[mu];
          const auto local_energy = [&](const std::vector<double>& displaced) {
            return evaluate_geometry(displaced, false)->potential[reference.center];
          };
          std::array<double, 4> third_derivative{};
          for (int level = 0; level < 4; ++level) {
            const double h = edgecheck_step_ / static_cast<double>(1 << level);
            third_derivative[level] = quantum_heat_moments::mixed_directional_derivative<double>(
              side_position, {direction, direction, direction}, {h, h, h}, local_energy);
          }
          std::fprintf(edgecheck_file_, "%d %d %d %d %d %d r cutoff_side_%s_Ui_third_directional %d %d",
            step, reference.center, reference.neighbor, image[0], image[1], image[2],
            side_name, has_radial ? 1 : 0, has_angular ? 1 : 0);
          print_real(edgecheck_file_, target_distance);
          print_real(edgecheck_file_, pair_cutoff);
          print_real(edgecheck_file_, nan);
          for (const double value : third_derivative) print_real(edgecheck_file_, value);
          const double third_richardson = (4.0 * third_derivative[1] - third_derivative[0]) / 3.0;
          print_real(edgecheck_file_, third_richardson);
          print_real(edgecheck_file_, nan);
          print_real(edgecheck_file_, nan);
          print_real(edgecheck_file_, std::fabs(third_derivative[0] - third_derivative[1]));
          print_real(edgecheck_file_, std::fabs(third_derivative[2] - third_derivative[3]));
          std::fprintf(edgecheck_file_, " %d %d\n", matching, self_image_count);
        }
      }
    }
  }

  std::fprintf(edgecheck_file_, "%d -1 -1 0 0 0 - not_coordinate_fd_testable 0 0", step);
  for (int field = 0; field < 12; ++field) print_real(edgecheck_file_, nan);
  std::fprintf(edgecheck_file_, " 0 %d\n", self_image_count);
  std::fflush(edgecheck_file_);
}

void QuantumHeatMoments::write_winding_summary(const int step, const RingPath& path)
{
  if (winding_file_ == nullptr) return;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (output_level_ == 2) {
    for (int bead = 0; bead < number_of_beads_; ++bead) {
      for (int atom = 0; atom < number_of_atoms_; ++atom) {
        for (int component = 0; component < 3; ++component) {
          const int dof = atom + component * number_of_atoms_;
          const size_t index = static_cast<size_t>(bead) * 3 * number_of_atoms_ + dof;
          std::fprintf(winding_file_, "%d link_lift %d %d %c %d %d %d %d",
            step, bead, atom, "xyz"[component], path.action_link_lift[index],
            path.closest_link_lift[index], path.action_winding[dof], path.closest_winding[dof]);
          for (int field = 0; field < 8; ++field) print_real(winding_file_, nan);
          std::fprintf(winding_file_, " yes\n");
        }
      }
    }
  }
  std::fprintf(winding_file_, "%d frame_summary -1 -1 -", step);
  for (int field = 0; field < 4; ++field) print_real(winding_file_, nan);
  std::fprintf(winding_file_, " %d %d %d %d",
    path.nonzero_action_winding_atoms, path.max_abs_action_winding_component,
    path.nonzero_closest_winding_atoms, path.max_abs_closest_winding_component);
  print_real(winding_file_, path.link_image_total);
  print_real(winding_file_, path.link_image_mismatch_count);
  print_real(winding_file_, path.max_link_length_action);
  print_real(winding_file_, path.max_link_length_cartesian_closest);
  std::fprintf(winding_file_, " yes\n");
  std::fflush(winding_file_);
}

void QuantumHeatMoments::write_profile_frame(
  const int step,
  const int beads,
  const double static_wall_seconds,
  const int moyal_probes,
  const int trace_probes)
{
  if (profile_file_ == nullptr) return;
  const double hit_rate = nep_eval_requested_ > 0 ?
    static_cast<double>(nep_eval_cache_hits_) / nep_eval_requested_ :
    std::numeric_limits<double>::quiet_NaN();
  std::fprintf(profile_file_, "%d %d %.8e %lld %lld %lld %.8e %lld %lld %d %d %.0f %.0f\n",
    step, beads, static_wall_seconds, nep_eval_requested_, nep_eval_executed_, nep_eval_cache_hits_, hit_rate,
    geometry_eval_requested_, geometry_eval_executed_, moyal_probes, trace_probes,
    estimated_nep_evaluations_per_static_frame_, static_cast<double>(nep_eval_executed_));
  std::fflush(profile_file_);
}

void QuantumHeatMoments::write_atom_debug(const int step, const RingPath& path)
{
  if (atom_debug_file_ == nullptr) return;
  for (int bead = 0; bead < number_of_beads_; ++bead) {
    const auto geometry = evaluate_geometry(path.wrapped[bead], true, true);
    for (int i = 0; i < number_of_atoms_; ++i) {
      const double zero[3] = {0.0, 0.0, 0.0};
      std::fprintf(atom_debug_file_, "%d %d %d -1 0 0 0", step, bead, i);
      print_real(atom_debug_file_, geometry->potential[i]);
      print_real(atom_debug_file_, geometry->force[i]);
      print_real(atom_debug_file_, geometry->force[i + number_of_atoms_]);
      print_real(atom_debug_file_, geometry->force[i + 2 * number_of_atoms_]);
      for (double value : zero) print_real(atom_debug_file_, value);
      for (double value : zero) print_real(atom_debug_file_, value);
      for (int mu = 0; mu < 3; ++mu)
        for (int alpha = 0; alpha < 3; ++alpha)
          print_real(atom_debug_file_, geometry->transport[(i * 3 + mu) * 3 + alpha]);
      for (int alpha = 0; alpha < 3; ++alpha)
        for (int mu = 0; mu < 3; ++mu) {
          const double value = ((alpha == mu ? geometry->potential[i] : 0.0) +
            geometry->transport[(i * 3 + mu) * 3 + alpha]) / mass_by_atom_[i];
          print_real(atom_debug_file_, value);
        }
      std::fprintf(atom_debug_file_, "\n");
    }
    for (const NEP_Local_Edge& edge : geometry->edges) {
      const int j = edge.neighbor;
      std::fprintf(atom_debug_file_, "%d %d %d %d %d %d %d", step, bead,
        edge.center, edge.neighbor, edge.image[0], edge.image[1], edge.image[2]);
      print_real(atom_debug_file_, geometry->potential[edge.center]);
      for (int mu = 0; mu < 3; ++mu) print_real(atom_debug_file_, geometry->force[edge.center + mu * number_of_atoms_]);
      for (double value : edge.displacement) print_real(atom_debug_file_, value);
      for (double value : edge.derivative) print_real(atom_debug_file_, value);
      for (int mu = 0; mu < 3; ++mu)
        for (int alpha = 0; alpha < 3; ++alpha)
          print_real(atom_debug_file_, geometry->transport[(j * 3 + mu) * 3 + alpha]);
      for (int alpha = 0; alpha < 3; ++alpha)
        for (int mu = 0; mu < 3; ++mu) {
          const double value = ((alpha == mu ? geometry->potential[j] : 0.0) +
            geometry->transport[(j * 3 + mu) * 3 + alpha]) / mass_by_atom_[j];
          print_real(atom_debug_file_, value);
        }
      std::fprintf(atom_debug_file_, "\n");
    }
  }
  std::fflush(atom_debug_file_);
}

void QuantumHeatMoments::end_of_step(
  const int,
  const int step,
  const int,
  const int,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>&,
  GPU_Vector<double>&,
  Atom& atom,
  Force&)
{
  const int md_step = step + 1;
  const bool static_due = (exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_) &&
    md_step % static_sample_interval_ == 0;
  const bool pimd_a0_due = pimd_a0_only_enabled_ && md_step % sample_interval_ == 0;
  const bool dynamic_due = dynamic_enabled_ && md_step % dynamic_sample_interval_ == 0;
  const bool candidate_due = enable_complex_pi_ && md_step % candidate_sample_interval_ == 0;
  const bool debug_due = atom_debug_file_ != nullptr && md_step % sample_interval_ == 0;
  const bool diagnostic_due = (validate_edge_derivatives_ || validate_fd_smoothness_ || validate_link_images_) &&
    md_step % sample_interval_ == 0;
  if (!static_due && !pimd_a0_due && !dynamic_due && !candidate_due && !debug_due && !diagnostic_due)
    return;
  const auto sample_begin = std::chrono::steady_clock::now();
  const auto static_begin = std::chrono::steady_clock::now();
  const double sample_temperature = integrate.type == 33 ?
    quantum_heat_moments::temperature_at_step(
      integrate.temperature1, integrate.temperature2, step, number_of_steps_) : temperature;
  if (!(sample_temperature > 0.0) || !std::isfinite(sample_temperature))
    PRINT_INPUT_ERROR("compute_quantum_heat_moments encountered invalid sampled temperature.");
  box.get_inverse();
  box.set_is_orthogonal();
  box_ = &box;
  if (diagnostic_due && (validate_edge_derivatives_ || validate_fd_smoothness_)) {
    shortest_lattice_vector_ = quantum_heat_moments::shortest_lattice_vector(lattice_matrix(box));
    cutoff_image_ratio_ = 2.0 * cutoff_ / shortest_lattice_vector_;
    unique_image_safe_ = cutoff_image_ratio_ < 1.0;
  }
  geometry_cache_.clear();
  energy_only_nep_calls_ = 0;
  derivative_nep_calls_ = 0;
  energy_only_nep_seconds_ = 0.0;
  derivative_nep_seconds_ = 0.0;
  energy_only_geometry_calls_ = 0;
  derivative_geometry_calls_ = 0;
  energy_only_geometry_seconds_ = 0.0;
  derivative_geometry_seconds_ = 0.0;
  derivative_edge_extract_seconds_ = 0.0;
  geometry_eval_requested_ = geometry_eval_executed_ = geometry_eval_cache_hits_ = 0;
  nep_eval_requested_ = nep_eval_executed_ = nep_eval_cache_hits_ = 0;
  const bool current_due = static_due || pimd_a0_due || dynamic_due;
  for (int bead = 0; bead < number_of_beads_; ++bead) {
    atom.position_beads[bead].copy_to_host(bead_position_host_[bead].data());
    if (current_due) atom.velocity_beads[bead].copy_to_host(bead_velocity_host_[bead].data());
  }
  RingPath path;
  const bool winding_due = winding_file_ != nullptr && md_step % sample_interval_ == 0;
  reconstruct_bead_chain(box, bead_position_host_, path, winding_due);
  if (winding_due) {
    ++winding_sample_count_;
    nonzero_action_winding_atoms_max_ = std::max(
      nonzero_action_winding_atoms_max_, path.nonzero_action_winding_atoms);
    max_abs_action_winding_component_ = std::max(
      max_abs_action_winding_component_, path.max_abs_action_winding_component);
    nonzero_closest_winding_atoms_max_ = std::max(
      nonzero_closest_winding_atoms_max_, path.nonzero_closest_winding_atoms);
    max_abs_closest_winding_component_ = std::max(
      max_abs_closest_winding_component_, path.max_abs_closest_winding_component);
    closest_winding_violation_total_ += path.nonzero_closest_winding_atoms;
    write_winding_summary(md_step, path);
    if (path.nonzero_action_winding_atoms != 0)
      std::fprintf(stderr, "Warning: action-link lift sum failed to close for %d atoms at step %d.\n",
        path.nonzero_action_winding_atoms, md_step);
  }

  const int P = number_of_beads_;
  const int N = number_of_atoms_;
  const double beta = 1.0 / (K_B * sample_temperature);
  if (static_due) ++static_frames_total_;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto fdcheck_counter_snapshot = [this]() {
    return std::array<long long, 5>{{nep_eval_requested_, nep_eval_executed_, nep_eval_cache_hits_,
      geometry_eval_requested_, geometry_eval_executed_}};
  };
  const auto write_fdcheck_row = [&](const char* estimator, const int order, const int alpha,
                                     const Complex h, const Complex h2,
                                     const Complex richardson, const ComplexStats& moyal_stats,
                                     const ComplexStats& trace_stats,
                                     const std::array<long long, 5>& count_base,
                                     const bool probe_stats_available, const double wall_seconds) {
    if (fdcheck_file_ == nullptr) {
      fdcheck_file_ = open_segment_file("quantum_heat_fdcheck.out");
      std::fprintf(fdcheck_file_,
        "# format_version 8\n# probe_statistics_available mu2_only_other_rows_nan\n"
         "# counter_scope per_alpha_pipeline_candidate_order_rows_repeat_shared_alpha_counts\n"
         "# alpha_wall_scope static_pipeline_candidate_probe_loop_including_warnings_excluding_frame_output_io\n"
        "# columns step time_fs estimator order alpha value_h_real value_h_imag value_h2_real value_h2_imag "
        "value_Richardson_real value_Richardson_imag fd_abs_diff fd_rel_diff moyal_probe_count moyal_mean_real "
        "moyal_mean_imag moyal_stderr_naive trace_probe_count trace_mean_real trace_mean_imag trace_stderr_naive "
        "nep_eval_requested_alpha_pipeline nep_eval_executed_alpha_pipeline nep_eval_cache_hits_alpha_pipeline "
        "geometry_eval_requested_alpha_pipeline geometry_eval_executed_alpha_pipeline alpha_pipeline_wall_seconds\n");
    }
    const double abs_diff = std::abs(h - h2);
    const double rel_diff = abs_diff / std::max(std::abs(h2), 1.0e-30);
    std::fprintf(fdcheck_file_, "%d %.12e %s %d %c", md_step, global_time * TIME_UNIT_CONVERSION,
      estimator, order, alpha_name(alpha));
    print_complex(fdcheck_file_, h);
    print_complex(fdcheck_file_, h2);
    print_complex(fdcheck_file_, richardson);
    print_real(fdcheck_file_, abs_diff);
    print_real(fdcheck_file_, rel_diff);
    if (probe_stats_available) {
      std::fprintf(fdcheck_file_, " %d", moyal_stats.count);
      print_real(fdcheck_file_, moyal_stats.mean.real());
      print_real(fdcheck_file_, moyal_stats.mean.imag());
      print_real(fdcheck_file_, moyal_stats.standard_error());
      std::fprintf(fdcheck_file_, " %d", trace_stats.count);
      print_real(fdcheck_file_, trace_stats.mean.real());
      print_real(fdcheck_file_, trace_stats.mean.imag());
      print_real(fdcheck_file_, trace_stats.standard_error());
    } else {
      for (int field = 0; field < 8; ++field) print_real(fdcheck_file_, nan);
    }
    std::fprintf(fdcheck_file_, " %lld %lld %lld %lld %lld %.8e\n",
      nep_eval_requested_ - count_base[0], nep_eval_executed_ - count_base[1],
      nep_eval_cache_hits_ - count_base[2], geometry_eval_requested_ - count_base[3],
      geometry_eval_executed_ - count_base[4], wall_seconds);
  };
  if ((validate_edge_derivatives_ || validate_fd_smoothness_) && diagnostic_due) {
    const auto geometry = evaluate_geometry(path.wrapped[0], true, true);
    validate_local_edge_derivatives(md_step, path.wrapped[0], *geometry);
  }
  std::vector<std::vector<Complex>> bead_momentum(P, std::vector<Complex>(static_cast<size_t>(3 * N)));
  std::vector<Complex> centroid_momentum(static_cast<size_t>(3 * N), 0.0);
  if (current_due) {
    for (int bead = 0; bead < P; ++bead) {
      for (int d = 0; d < 3 * N; ++d) {
        const Complex p(mass_by_dof_[d] * bead_velocity_host_[bead][d], 0.0);
        bead_momentum[bead][d] = p;
        centroid_momentum[d] += p / P;
      }
    }
  }
  std::array<double, 3> centroid_current{nan, nan, nan}, bead_average_current{nan, nan, nan};
  const std::vector<double> centroid_position = wrap_position(box, path.centroid);
  if (current_due) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      const A0Parts c = evaluate_A0_parts(centroid_position, centroid_momentum, alpha, fd_step_r_);
      centroid_current[alpha] = (c.linear + c.cubic).real();
      bead_average_current[alpha] = 0.0;
      for (int bead = 0; bead < P; ++bead) {
        const A0Parts j = evaluate_A0_parts(path.wrapped[bead], bead_momentum[bead], alpha, fd_step_r_);
        bead_average_current[alpha] += (j.linear + j.cubic).real() / P;
      }
    }
  }
  bool current_finite = current_due;
  for (const double value : centroid_current) current_finite = current_finite && std::isfinite(value);
  for (const double value : bead_average_current) current_finite = current_finite && std::isfinite(value);

  std::array<double, 4> mu0, mu2, mu4;
  std::array<Complex, 4> mu0_complex, mu2_complex, mu4_complex;
  mu0.fill(nan);
  mu2.fill(nan);
  mu4.fill(nan);
  mu0_complex.fill(Complex(nan, nan));
  mu2_complex.fill(Complex(nan, nan));
  mu4_complex.fill(Complex(nan, nan));
  std::array<std::vector<Complex>, 3> gamma0_richardson, gamma2_richardson;
  for (auto& values : gamma0_richardson) values.assign(P, Complex(nan, nan));
  for (auto& values : gamma2_richardson) values.assign(P, Complex(nan, nan));
  std::array<double, 3> gamma0_residual{}, gamma2_residual{};
  bool static_numeric_valid = static_due && current_finite;
  bool static_nonfinite_failure = static_due && !current_finite;
  bool static_other_failure = false;
  int static_fd_within_threshold = -1;
  const auto check_static_fd = [&](const int order, const int alpha, const Complex h, const Complex h2) {
    const double relative_difference = std::abs(h - h2) / std::max(std::abs(h2), 1.0e-30);
    if (static_fd_within_threshold < 0) static_fd_within_threshold = 1;
    if (!std::isfinite(relative_difference) || relative_difference > fd_warning_threshold_) {
      static_fd_within_threshold = 0;
      std::fprintf(stderr, "Warning: mu%d fd h/h2 relative difference %.6g at step %d alpha=%c.\n",
        order, relative_difference, md_step, alpha_name(alpha));
    }
  };
  std::array<double, 3> mu2_h{}, mu2_h2{};
  std::array<Complex, 3> mu0_h{}, mu0_h2{}, mu4_h{}, mu4_h2{};
  std::array<std::vector<double>, 3> gamma_richardson;
  for (auto& values : gamma_richardson) values.assign(P, std::numeric_limits<double>::quiet_NaN());
  if (exact_mu2_enabled_ && static_due) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      const auto fdcheck_count_base = fdcheck_counter_snapshot();
      const auto alpha_begin = std::chrono::steady_clock::now();
      std::vector<double> gamma_h(P), gamma_h2(P);
      ComplexStats moyal_frame_h, trace_frame_h;
      std::vector<Complex> moyal_frame_samples_h(n_moyal_probe_);
      std::vector<Complex> trace_frame_samples_h(n_static_trace_probe_);
      for (int bead = 0; bead < P; ++bead) {
        RecursionSettings settings;
        settings.fd_step_r = fd_step_r_;
        settings.fd_step_p = fd_step_p_;
        settings.hbar = HBAR;
        settings.alpha = alpha;
        settings.frame = md_step;
        settings.seed = quantum_heat_moments::static_probe_seed(
          seed_, md_step, alpha, bead, 0x6d75325f6d6f7961ULL);
        const std::vector<MoyalProbe> moyal =
          quantum_heat_moments::make_moyal_probes(n_moyal_probe_, 3 * N, settings);
        RecursionSettings trace_settings = settings;
        trace_settings.seed = quantum_heat_moments::static_probe_seed(
          seed_, md_step, alpha, bead, 0x6d75325f74726163ULL);
        const std::vector<MoyalProbe> trace =
          quantum_heat_moments::make_moyal_probes(n_static_trace_probe_, 3 * N, trace_settings);
        ComplexStats moyal_h, trace_h, moyal_h2, trace_h2;
        std::vector<Complex> moyal_samples_h, trace_samples_h, moyal_samples_h2, trace_samples_h2;
        gamma_h[bead] = evaluate_gamma1(
          path.wrapped[bead], path.link_displacement[bead], alpha, beta, fd_step_r_,
          moyal, trace, moyal_h, trace_h, moyal_samples_h, trace_samples_h);
        gamma_h2[bead] = evaluate_gamma1(
          path.wrapped[bead], path.link_displacement[bead], alpha, beta, fd_step_r_ * 0.5,
          moyal, trace, moyal_h2, trace_h2, moyal_samples_h2, trace_samples_h2);
        for (size_t k = 0; k < moyal_samples_h.size(); ++k) {
          moyal_frame_samples_h[k] += moyal_samples_h[k] / static_cast<double>(P);
        }
        for (size_t k = 0; k < trace_samples_h.size(); ++k) {
          trace_frame_samples_h[k] += trace_samples_h[k] / static_cast<double>(P);
        }
        gamma_richardson[alpha][bead] = (4.0 * gamma_h2[bead] - gamma_h[bead]) / 3.0;
      }
      for (const Complex value : moyal_frame_samples_h) moyal_frame_h.add(value);
      for (const Complex value : trace_frame_samples_h) trace_frame_h.add(value);
      mu2_h[alpha] = quantum_heat_moments::open_endpoint_pair_average(gamma_h);
      mu2_h2[alpha] = quantum_heat_moments::open_endpoint_pair_average(gamma_h2);
      mu2[alpha] = (4.0 * mu2_h2[alpha] - mu2_h[alpha]) / 3.0;
      mu2_complex[alpha] = Complex(mu2[alpha], 0.0);

      check_static_fd(2, alpha, Complex(mu2_h[alpha], 0.0), Complex(mu2_h2[alpha], 0.0));
      const double moyal_rel = moyal_frame_h.standard_error() /
        std::max(std::abs(moyal_frame_h.mean), 1.0e-30);
      const double trace_rel = trace_frame_h.standard_error() /
        std::max(std::abs(trace_frame_h.mean), 1.0e-30);
      if (!std::isfinite(moyal_rel) || !std::isfinite(trace_rel) ||
          moyal_rel > stochastic_warning_threshold_ || trace_rel > stochastic_warning_threshold_) {
        std::fprintf(stderr, "Warning: mu2 stochastic relative error at step %d alpha=%c: Moyal %.6g, trace %.6g.\n",
          md_step, alpha_name(alpha), moyal_rel, trace_rel);
      }
      const double alpha_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - alpha_begin).count();
      write_fdcheck_row("mu2", 2, alpha, Complex(mu2_h[alpha], 0.0), Complex(mu2_h2[alpha], 0.0),
        Complex(mu2[alpha], 0.0), moyal_frame_h, trace_frame_h, fdcheck_count_base, true, alpha_wall_seconds);
    }
    mu2[3] = (mu2[0] + mu2[1] + mu2[2]) / 3.0;
    mu2_complex[3] = (mu2_complex[0] + mu2_complex[1] + mu2_complex[2]) / 3.0;
    if (!std::isfinite(mu2[3])) {
      static_numeric_valid = false;
      static_nonfinite_failure = true;
    }
  }

  if (static_due && exact_mu0_enabled_) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      const auto fdcheck_count_base = fdcheck_counter_snapshot();
      const auto alpha_begin = std::chrono::steady_clock::now();
      std::vector<Complex> gamma_h(P), gamma_h2(P);
      double max_residual = 0.0;
      for (int bead = 0; bead < P; ++bead) {
        RecursionSettings settings;
        settings.alpha = alpha;
        settings.frame = md_step;
        settings.seed = quantum_heat_moments::static_probe_seed(
          seed_, md_step, alpha, bead, 0x6d75305f74726163ULL);
        const auto trace_probes = quantum_heat_moments::make_moyal_probes(
          n_static_trace_probe_, 3 * N, settings);
        double residual_h = nan, residual_h2 = nan;
        gamma_h[bead] = evaluate_gamma0(path.wrapped[bead], path.link_displacement[bead],
          alpha, beta, fd_step_r_, trace_probes, residual_h);
        gamma_h2[bead] = evaluate_gamma0(path.wrapped[bead], path.link_displacement[bead],
          alpha, beta, 0.5 * fd_step_r_, trace_probes, residual_h2);
        max_residual = std::max(max_residual, std::max(residual_h, residual_h2));
        gamma0_richardson[alpha][bead] = (4.0 * gamma_h2[bead] - gamma_h[bead]) / 3.0;
        if (!finite(gamma_h[bead]) || !finite(gamma_h2[bead]) ||
            !std::isfinite(residual_h) || !std::isfinite(residual_h2)) {
          static_numeric_valid = false;
          static_nonfinite_failure = true;
        }
      }
      mu0_h[alpha] = quantum_heat_moments::open_endpoint_pair_average(gamma_h);
      mu0_h2[alpha] = quantum_heat_moments::open_endpoint_pair_average(gamma_h2);
      check_static_fd(0, alpha, mu0_h[alpha], mu0_h2[alpha]);
      const Complex mu0_richardson = (4.0 * mu0_h2[alpha] - mu0_h[alpha]) / 3.0;
      mu0_complex[alpha] = mu0_richardson;
      mu0[alpha] = mu0_richardson.real();
      const double open_residual = std::fabs(mu0_richardson.imag()) /
        std::max(std::fabs(mu0_richardson.real()), 1.0e-30);
      gamma0_residual[alpha] = max_residual;
      if (!finite(mu0_h[alpha]) || !finite(mu0_h2[alpha]) || !finite(mu0_richardson) ||
          !std::isfinite(open_residual)) {
        static_numeric_valid = false;
        static_nonfinite_failure = true;
      }
      if (open_residual > stochastic_warning_threshold_ || max_residual > stochastic_warning_threshold_) {
        static_numeric_valid = false;
        static_other_failure = true;
      }
      write_fdcheck_row("mu0", 0, alpha, mu0_h[alpha], mu0_h2[alpha], mu0_richardson,
        ComplexStats(), ComplexStats(), fdcheck_count_base, false, std::chrono::duration<double>(
          std::chrono::steady_clock::now() - alpha_begin).count());
    }
    mu0[3] = (mu0[0] + mu0[1] + mu0[2]) / 3.0;
    mu0_complex[3] = (mu0_complex[0] + mu0_complex[1] + mu0_complex[2]) / 3.0;
    if (!finite(mu0_complex[3])) {
      static_numeric_valid = false;
      static_nonfinite_failure = true;
    }
  }

  if (static_due && exact_mu2_enabled_) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      if (!std::isfinite(mu2[alpha])) {
        static_numeric_valid = false;
        static_nonfinite_failure = true;
      }
      for (const double gamma : gamma_richardson[alpha])
        if (!std::isfinite(gamma)) {
          static_numeric_valid = false;
          static_nonfinite_failure = true;
        }
    }
  }

  if (static_due && exact_mu4_enabled_) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      const auto fdcheck_count_base = fdcheck_counter_snapshot();
      const auto alpha_begin = std::chrono::steady_clock::now();
      std::vector<Complex> gamma_h(P), gamma_h2(P);
      double max_residual = 0.0;
      for (int bead = 0; bead < P; ++bead) {
        RecursionSettings settings;
        settings.alpha = alpha;
        settings.frame = md_step;
        settings.seed = quantum_heat_moments::static_probe_seed(
          seed_, md_step, alpha, bead, 0x6d75345f74726163ULL);
        const auto trace_probes = quantum_heat_moments::make_moyal_probes(
          n_static_trace_probe_, 3 * N, settings);
        ComplexStats contraction_h, contraction_h2;
        double residual_h = nan, residual_h2 = nan;
        gamma_h[bead] = evaluate_gamma2(path.wrapped[bead], path.link_displacement[bead],
          alpha, beta, fd_step_r_, fd_step_p_, trace_probes, residual_h, contraction_h);
        gamma_h2[bead] = evaluate_gamma2(path.wrapped[bead], path.link_displacement[bead],
          alpha, beta, 0.5 * fd_step_r_, 0.5 * fd_step_p_, trace_probes, residual_h2, contraction_h2);
        max_residual = std::max(max_residual, std::max(residual_h, residual_h2));
        gamma2_richardson[alpha][bead] = (4.0 * gamma_h2[bead] - gamma_h[bead]) / 3.0;
        if (!finite(gamma_h[bead]) || !finite(gamma_h2[bead]) ||
            !std::isfinite(residual_h) || !std::isfinite(residual_h2)) {
          static_numeric_valid = false;
          static_nonfinite_failure = true;
        }
      }
      mu4_h[alpha] = quantum_heat_moments::open_endpoint_pair_average(gamma_h);
      mu4_h2[alpha] = quantum_heat_moments::open_endpoint_pair_average(gamma_h2);
      check_static_fd(4, alpha, mu4_h[alpha], mu4_h2[alpha]);
      const Complex mu4_richardson = (4.0 * mu4_h2[alpha] - mu4_h[alpha]) / 3.0;
      mu4_complex[alpha] = mu4_richardson;
      mu4[alpha] = mu4_richardson.real();
      const double open_residual = std::fabs(mu4_richardson.imag()) /
        std::max(std::fabs(mu4_richardson.real()), 1.0e-30);
      gamma2_residual[alpha] = max_residual;
      if (!finite(mu4_h[alpha]) || !finite(mu4_h2[alpha]) || !finite(mu4_richardson) ||
          !std::isfinite(open_residual)) {
        static_numeric_valid = false;
        static_nonfinite_failure = true;
      }
      if (open_residual > stochastic_warning_threshold_ || max_residual > stochastic_warning_threshold_) {
        static_numeric_valid = false;
        static_other_failure = true;
      }
      write_fdcheck_row("mu4", 4, alpha, mu4_h[alpha], mu4_h2[alpha], mu4_richardson,
        ComplexStats(), ComplexStats(), fdcheck_count_base, false, std::chrono::duration<double>(
          std::chrono::steady_clock::now() - alpha_begin).count());
    }
    mu4[3] = (mu4[0] + mu4[1] + mu4[2]) / 3.0;
    mu4_complex[3] = (mu4_complex[0] + mu4_complex[1] + mu4_complex[2]) / 3.0;
    if (!finite(mu4_complex[3])) {
      static_numeric_valid = false;
      static_nonfinite_failure = true;
    }
  }

  if (static_due) {
    const double static_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - static_begin).count();
    ++static_profile_sample_count_;
    profile_static_wall_seconds_ += static_wall_seconds;
    static_profile_nep_eval_requested_ += nep_eval_requested_;
    static_profile_nep_eval_executed_ += nep_eval_executed_;
    static_profile_nep_eval_cache_hits_ += nep_eval_cache_hits_;
    static_profile_geometry_eval_requested_ += geometry_eval_requested_;
    static_profile_geometry_eval_executed_ += geometry_eval_executed_;
    write_profile_frame(md_step, P, static_wall_seconds,
      exact_mu2_enabled_ || exact_mu4_enabled_ ? n_moyal_probe_ : 0, n_static_trace_probe_);
  }

  const auto candidate_begin = std::chrono::steady_clock::now();
  std::array<std::array<ComplexStats, 4>, 3> candidate_frame_stats;
  std::array<std::array<ComplexStats, 4>, 3> candidate_frame_stats_h2;
  if (candidate_due) {
    const double omega_P = P / (beta * HBAR);
    for (int alpha = 0; alpha < 3; ++alpha) {
      RecursionSettings settings;
      settings.fd_step_r = fd_step_r_;
      settings.fd_step_p = fd_step_p_;
      settings.hbar = HBAR;
      settings.alpha = alpha;
      settings.frame = md_step;
      settings.seed = seed_;
      const auto moyal = quantum_heat_moments::make_moyal_probes(
        weyl_max_order_ == 0 ? 0 : n_moyal_probe_, 3 * N, settings);
      RecursionSettings settings_h2 = settings;
      settings_h2.fd_step_r *= 0.5;
      settings_h2.fd_step_p *= 0.5;
      const auto fdcheck_count_base = fdcheck_counter_snapshot();
      const auto alpha_candidate_begin = std::chrono::steady_clock::now();
      const bool a0_only = weyl_max_order_ == 0;
      std::vector<Complex> a0_by_probe(a0_only ? n_aux_probe_ : 0, Complex());
      const auto add_probe = [&](const int probe, const int order, const Complex value, const Complex h2) {
        if (finite(value)) {
          if (!candidate_frame_stats[alpha][order].add_if_finite(value))
            std::fprintf(stderr,
              "Warning: complex-Pi candidate A%d statistics overflow at step %d alpha=%c probe=%d.\n",
              order, md_step, alpha_name(alpha), probe);
        } else {
          std::fprintf(stderr,
            "Warning: non-finite complex-Pi candidate A%d at step %d alpha=%c probe=%d.\n",
            order, md_step, alpha_name(alpha), probe);
        }
        if (!a0_only && finite(h2)) candidate_frame_stats_h2[alpha][order].add_if_finite(h2);
      };
      for (int outer = 0; outer < (a0_only ? P : n_aux_probe_); ++outer) {
        Complex candidate_by_order[4] = {};
        Complex candidate_by_order_h2[4] = {};
        for (int inner = 0; inner < (a0_only ? n_aux_probe_ : P); ++inner) {
          const int probe = a0_only ? inner : outer;
          const int bead = a0_only ? outer : inner;
          const std::vector<double> auxiliary = quantum_heat_moments::make_auxiliary_momenta(
            probe, bead, mass_by_dof_, beta, P, settings);
          std::vector<Complex> momentum(3 * N);
          for (int d = 0; d < 3 * N; ++d) {
            momentum[d] = Complex(auxiliary[d], mass_by_dof_[d] * omega_P * path.link_displacement[bead][d]);
          }
          for (int order = 0; order <= weyl_max_order_ / 2; ++order) {
            const Complex value = quantum_heat_moments::evaluate_Ar(
              order, path.wrapped[bead], momentum, mass_by_dof_, settings, moyal, *this).value / P;
            if (a0_only) {
              a0_by_probe[probe] += value;
            } else {
              candidate_by_order[order] += value;
              candidate_by_order_h2[order] += order == 0 ? value :
                quantum_heat_moments::evaluate_Ar(
                  order, path.wrapped[bead], momentum, mass_by_dof_, settings_h2, moyal, *this).value / P;
            }
          }
        }
        if (!a0_only)
          for (int order = 0; order <= weyl_max_order_ / 2; ++order)
            add_probe(outer, order, candidate_by_order[order], candidate_by_order_h2[order]);
      }
      if (a0_only)
        for (int probe = 0; probe < n_aux_probe_; ++probe)
          add_probe(probe, 0, a0_by_probe[probe], Complex());
      const double alpha_probe_loop_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - alpha_candidate_begin).count();
      for (int order = 0; order <= weyl_max_order_ / 2; ++order) {
        const ComplexStats& frame_stats = candidate_frame_stats[alpha][order];
        const bool probe_frame_complete =
          quantum_heat_moments::has_complete_candidate_probe_set(frame_stats, n_aux_probe_);
        const bool accepted = probe_frame_complete &&
          candidate_pi_stats_[alpha][order].add_if_finite(frame_stats.mean);
        if (!accepted) {
          ++candidate_invalid_frame_count_[alpha][order];
          if (probe_frame_complete)
            std::fprintf(stderr,
              "Warning: complex-Pi candidate A%d cross-frame statistics overflow at step %d alpha=%c.\n",
              order, md_step, alpha_name(alpha));
        }
        if (estimator_stats_file_ != nullptr) {
          const double candidate_nan = std::numeric_limits<double>::quiet_NaN();
          std::fprintf(estimator_stats_file_, "candidate_frame %d A%d %c %d", md_step, order,
            alpha_name(alpha), frame_stats.count);
          print_real(estimator_stats_file_, probe_frame_complete ? frame_stats.mean.real() : candidate_nan);
          print_real(estimator_stats_file_, probe_frame_complete ? frame_stats.mean.imag() : candidate_nan);
          print_real(estimator_stats_file_, probe_frame_complete ? frame_stats.standard_error_real() : candidate_nan);
          print_real(estimator_stats_file_, probe_frame_complete ? frame_stats.standard_error_imag() : candidate_nan);
          std::fprintf(estimator_stats_file_, " %d %d\n",
            probe_frame_complete ? 1 : 0, accepted ? 1 : 0);
        }
        if (weyl_max_order_ > 0) {
          const ComplexStats& frame_h2 = candidate_frame_stats_h2[alpha][order];
          const bool valid_h = quantum_heat_moments::has_complete_candidate_probe_set(frame_stats, n_aux_probe_);
          const bool valid_h2 = quantum_heat_moments::has_complete_candidate_probe_set(frame_h2, n_aux_probe_);
          const Complex nan_complex(nan, nan);
          write_fdcheck_row("candidate_A", order, alpha,
            valid_h ? frame_stats.mean : nan_complex,
            valid_h2 ? frame_h2.mean : nan_complex,
            valid_h && valid_h2 ? (4.0 * frame_h2.mean - frame_stats.mean) / 3.0 : nan_complex,
            ComplexStats(), ComplexStats(), fdcheck_count_base, false, alpha_probe_loop_seconds);
        }
      }
    }
  }
  const double candidate_elapsed_seconds = candidate_due ? std::chrono::duration<double>(
    std::chrono::steady_clock::now() - candidate_begin).count() : 0.0;

  if (static_file_ != nullptr && (static_due || pimd_a0_due)) {
    if (static_due && static_numeric_valid) {
      auto next_mu0 = mu0_stats_;
      auto next_mu2 = mu2_stats_;
      auto next_mu4 = mu4_stats_;
      bool stats_valid = true;
      for (int alpha = 0; alpha < 3; ++alpha) {
        if (exact_mu0_enabled_ && !next_mu0[alpha].add_if_finite(mu0_complex[alpha])) stats_valid = false;
        if (exact_mu2_enabled_ && !next_mu2[alpha].add_if_finite(mu2_complex[alpha])) stats_valid = false;
        if (exact_mu4_enabled_ && !next_mu4[alpha].add_if_finite(mu4_complex[alpha])) stats_valid = false;
      }
      if (stats_valid) {
        mu0_stats_ = next_mu0;
        mu2_stats_ = next_mu2;
        mu4_stats_ = next_mu4;
        ++static_frames_used_;
        for (int order = 0; order < 3; ++order) {
          const bool order_enabled = order == 0 ? exact_mu0_enabled_ :
            order == 1 ? exact_mu2_enabled_ : exact_mu4_enabled_;
          if (!order_enabled) continue;
          for (int alpha = 0; alpha < 3; ++alpha) {
            for (int lag = 0; lag < static_cast<int>(imaginary_stats_[order][alpha].size()); ++lag) {
              if (lag == 0) continue;
              Complex correlation = 0.0;
              for (int bead = 0; bead < P; ++bead) {
                const int other = (bead + lag) % P;
                const Complex left = order == 0 ? gamma0_richardson[alpha][bead] :
                  order == 1 ? Complex(gamma_richardson[alpha][bead], 0.0) :
                    gamma2_richardson[alpha][bead];
                const Complex right = order == 0 ? gamma0_richardson[alpha][other] :
                  order == 1 ? Complex(gamma_richardson[alpha][other], 0.0) :
                    gamma2_richardson[alpha][other];
                correlation += left * right / P;
              }
              if (!imaginary_stats_[order][alpha][lag].add_if_finite(correlation))
                std::fprintf(stderr,
                  "Warning: imaginary-time correlation statistics overflow at step %d order=%d alpha=%c lag=%d.\n",
                  md_step, order, alpha_name(alpha), lag);
            }
          }
        }
      } else {
        static_numeric_valid = false;
        static_other_failure = true;
      }
    }
    if (static_due) {
      if (!static_numeric_valid) {
        if (static_nonfinite_failure) ++static_frames_skipped_nan_;
        else if (static_other_failure) ++static_frames_skipped_other_;
        else ++static_frames_skipped_nan_;
      }
    }
    std::fprintf(static_file_, "%d %.12e %d %.12e %.12e", md_step,
      global_time * TIME_UNIT_CONVERSION, P, sample_temperature, box.get_volume());
    for (double value : centroid_current) print_real(static_file_, value);
    for (double value : bead_average_current) print_real(static_file_, value);
    if (exact_mu0_enabled_)
      for (double value : mu0) print_real(static_file_, static_due && static_numeric_valid ? value : nan);
    if (exact_mu2_enabled_)
      for (double value : mu2) print_real(static_file_, static_due && static_numeric_valid ? value : nan);
    if (exact_mu4_enabled_)
      for (double value : mu4) print_real(static_file_, static_due && static_numeric_valid ? value : nan);
    const auto print_imaginary_residuals = [&](const std::array<Complex, 4>& values) {
      for (const Complex value : values) {
        const double residual = std::fabs(value.imag()) / std::max(std::fabs(value.real()), 1.0e-30);
        print_real(static_file_, static_due ? residual : nan);
      }
    };
    if (exact_mu0_enabled_) print_imaginary_residuals(mu0_complex);
    if (exact_mu2_enabled_) print_imaginary_residuals(mu2_complex);
    if (exact_mu4_enabled_) print_imaginary_residuals(mu4_complex);
    const bool static_row_valid = static_due ? static_numeric_valid : current_finite;
    const char* skip_reason = static_due ?
      (static_numeric_valid ? "none" :
        static_other_failure && !static_nonfinite_failure ? "other" : "nan") :
      (current_finite ? "none" : "nan");
    std::fprintf(static_file_, " %d %s %d %d", static_row_valid ? 1 : 0, skip_reason,
      current_finite && !static_nonfinite_failure ? 1 : 0, static_fd_within_threshold);
    std::fprintf(static_file_, "\n");
    std::fflush(static_file_);
  }

  if (static_due && output_level_ >= 1 &&
      (exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_)) {
    if (link_file_ == nullptr) {
      link_file_ = open_segment_file("quantum_heat_link.out");
      std::fprintf(link_file_, "# format_version 3\n# columns step bead_id alpha deltaR_norm ");
      if (exact_mu0_enabled_) std::fprintf(link_file_, "Gamma0_real Gamma0_imag Gamma0_real_residual ");
      if (exact_mu2_enabled_) std::fprintf(link_file_, "Gamma1_real Gamma1_imag Gamma1_imag_residual ");
      if (exact_mu4_enabled_) std::fprintf(link_file_, "Gamma2_real Gamma2_imag Gamma2_real_residual ");
      std::fprintf(link_file_, "\n");
    }
    for (int bead = 0; bead < P; ++bead) {
      const double dr = norm(path.link_displacement[bead]);
      for (int alpha = 0; alpha < 3; ++alpha) {
        std::fprintf(link_file_, "%d %d %c %.16e", md_step, bead, alpha_name(alpha), dr);
        if (exact_mu0_enabled_) {
          print_complex(link_file_, gamma0_richardson[alpha][bead]);
          print_real(link_file_, gamma0_residual[alpha]);
        }
        if (exact_mu2_enabled_) {
          print_complex(link_file_, Complex(gamma_richardson[alpha][bead], 0.0));
          print_real(link_file_, 0.0);
        }
        if (exact_mu4_enabled_) {
          print_complex(link_file_, gamma2_richardson[alpha][bead]);
          print_real(link_file_, gamma2_residual[alpha]);
        }
        std::fprintf(link_file_, "\n");
      }
    }
    std::fflush(link_file_);
  }

  if (debug_due) write_atom_debug(md_step, path);

  double hac_current_lookup_seconds = 0.0;
  if (dynamic_file_ != nullptr && dynamic_due) {
    double existing[3] = {
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
    bool has_existing = false;
    if (hac_ != nullptr) {
      const auto hac_lookup_begin = std::chrono::steady_clock::now();
      has_existing = hac_->get_current_for_step(md_step, existing);
      hac_current_lookup_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - hac_lookup_begin).count();
    }
    std::fprintf(dynamic_file_, "%d %.12e", md_step, global_time * TIME_UNIT_CONVERSION);
    for (double value : centroid_current) print_real(dynamic_file_, value);
    for (double value : bead_average_current) print_real(dynamic_file_, value);
    for (int a = 0; a < 3; ++a) print_real(dynamic_file_, has_existing ? existing[a] : std::numeric_limits<double>::quiet_NaN());
    for (int a = 0; a < 3; ++a) print_real(dynamic_file_, has_existing ? existing[a] - centroid_current[a] : std::numeric_limits<double>::quiet_NaN());
    print_real(dynamic_file_, static_cast<double>(energy_only_nep_calls_));
    print_real(dynamic_file_, energy_only_nep_seconds_);
    print_real(dynamic_file_, static_cast<double>(derivative_nep_calls_));
    print_real(dynamic_file_, derivative_nep_seconds_);
    print_real(dynamic_file_, derivative_edge_extract_seconds_);
    print_real(dynamic_file_, static_cast<double>(energy_only_geometry_calls_));
    print_real(dynamic_file_, energy_only_geometry_seconds_);
    print_real(dynamic_file_, static_cast<double>(derivative_geometry_calls_));
    print_real(dynamic_file_, derivative_geometry_seconds_);
    print_real(dynamic_file_, hac_current_lookup_seconds);
    print_real(dynamic_file_, std::chrono::duration<double>(
      std::chrono::steady_clock::now() - sample_begin).count());
    std::fprintf(dynamic_file_, "\n");
    std::fflush(dynamic_file_);
  }
  ++profile_sample_count_;
  profile_geometry_eval_requested_ += geometry_eval_requested_;
  profile_geometry_eval_executed_ += geometry_eval_executed_;
  profile_geometry_eval_cache_hits_ += geometry_eval_cache_hits_;
  profile_nep_eval_requested_ += nep_eval_requested_;
  profile_nep_eval_executed_ += nep_eval_executed_;
  profile_nep_eval_cache_hits_ += nep_eval_cache_hits_;
  profile_energy_only_nep_calls_ += energy_only_nep_calls_;
  profile_derivative_nep_calls_ += derivative_nep_calls_;
  profile_energy_only_nep_seconds_ += energy_only_nep_seconds_;
  profile_derivative_nep_seconds_ += derivative_nep_seconds_;
  profile_energy_only_geometry_calls_ += energy_only_geometry_calls_;
  profile_derivative_geometry_calls_ += derivative_geometry_calls_;
  profile_energy_only_geometry_seconds_ += energy_only_geometry_seconds_;
  profile_derivative_geometry_seconds_ += derivative_geometry_seconds_;
  profile_derivative_edge_extract_seconds_ += derivative_edge_extract_seconds_;
  profile_candidate_seconds_ += candidate_elapsed_seconds;
  profile_hac_lookup_seconds_ += hac_current_lookup_seconds;
  profile_sample_wall_seconds_ += std::chrono::duration<double>(
    std::chrono::steady_clock::now() - sample_begin).count();
}

void QuantumHeatMoments::write_finalize_outputs()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (estimator_stats_file_ != nullptr && (exact_mu0_enabled_ || exact_mu2_enabled_ || exact_mu4_enabled_)) {
    const std::array<const std::array<ComplexStats, 3>*, 3> stats_by_order{&mu0_stats_, &mu2_stats_, &mu4_stats_};
    const std::array<bool, 3> enabled{exact_mu0_enabled_, exact_mu2_enabled_, exact_mu4_enabled_};
    for (int order = 0; order < 3; ++order) {
      if (!enabled[order]) continue;
      for (int alpha = 0; alpha < 3; ++alpha) {
        const ComplexStats& stats = (*stats_by_order[order])[alpha];
        std::fprintf(estimator_stats_file_, "%d %c %d %d", order * 2, alpha_name(alpha),
          number_of_beads_, stats.count);
        print_real(estimator_stats_file_, stats.count > 0 ? stats.mean.real() : nan);
        print_real(estimator_stats_file_, stats.count > 0 ? stats.mean.imag() : nan);
        print_real(estimator_stats_file_, stats.variance());
        print_real(estimator_stats_file_, stats.variance_real());
        print_real(estimator_stats_file_, stats.variance_imag());
        print_real(estimator_stats_file_, stats.covariance());
        print_real(estimator_stats_file_, stats.standard_error_real());
        print_real(estimator_stats_file_, stats.standard_error_imag());
        print_real(estimator_stats_file_, stats.mean_abs());
        print_real(estimator_stats_file_, stats.count > 0 ? stats.max_abs : nan);
        std::fprintf(estimator_stats_file_, " %lld %lld %lld %lld\n",
          static_frames_total_, static_frames_used_, static_frames_skipped_nan_,
          static_frames_skipped_other_);
      }
    }
  }
  if (imaginary_file_ != nullptr) {
    const std::array<bool, 3> enabled{exact_mu0_enabled_, exact_mu2_enabled_, exact_mu4_enabled_};
    for (int order = 0; order < 3; ++order) {
      if (!enabled[order]) continue;
      for (int alpha = 0; alpha < 3; ++alpha) {
        for (int lag = 0; lag < static_cast<int>(imaginary_stats_[order][alpha].size()); ++lag) {
          const ComplexStats& stats = imaginary_stats_[order][alpha][lag];
          std::fprintf(imaginary_file_, "%d %c %d %.16e", 2 * order, alpha_name(alpha), lag,
            static_cast<double>(lag) / number_of_beads_);
          print_real(imaginary_file_, stats.count > 0 ? stats.mean.real() : nan);
          print_real(imaginary_file_, stats.count > 0 ? stats.mean.imag() : nan);
          print_real(imaginary_file_, stats.standard_error_real());
          print_real(imaginary_file_, stats.standard_error_imag());
          std::fprintf(imaginary_file_, " %d\n", stats.count);
        }
      }
    }
  }
  if (estimator_stats_file_ != nullptr && enable_complex_pi_) {
    for (int alpha = 0; alpha < 3; ++alpha) {
      for (int order = 0; order <= weyl_max_order_ / 2; ++order) {
        const ComplexStats& stats = candidate_pi_stats_[alpha][order];
        std::fprintf(estimator_stats_file_, "A%d %c %d", order, alpha_name(alpha), stats.count);
        print_real(estimator_stats_file_, stats.count > 0 ? stats.mean.real() : nan);
        print_real(estimator_stats_file_, stats.count > 0 ? stats.mean.imag() : nan);
        print_real(estimator_stats_file_, stats.variance_real());
        print_real(estimator_stats_file_, stats.variance_imag());
        print_real(estimator_stats_file_, stats.covariance());
        print_real(estimator_stats_file_, stats.variance());
        print_real(estimator_stats_file_, stats.standard_error_real());
        print_real(estimator_stats_file_, stats.standard_error_imag());
        print_real(estimator_stats_file_, stats.mean_abs());
        print_real(estimator_stats_file_, stats.count > 0 ? stats.max_abs : nan);
        std::fprintf(estimator_stats_file_, " %d", candidate_invalid_frame_count_[alpha][order]);
        std::fprintf(estimator_stats_file_, "\n");
      }
    }
  }
  if (meta_file_ != nullptr) {
    std::fprintf(meta_file_,
      "# profile_summary_version 4\n# profile_sample_count %d\n"
      "# profile_static_snapshot_scope action_entry_through_exact_static_pipeline_before_complex_pi_candidate\n"
      "# profile_static_snapshot_includes A0_current_and_enabled_edgecheck yes\n"
      "# profile_static_snapshot_excludes_complex_pi_candidate yes\n"
      "# profile_no_cache_estimate_scope exact_mu0_mu2_mu4_only_excludes_A0_current_and_enabled_edgecheck yes\n"
      "# profile_all_action_totals_include_complex_pi_candidate_and_debug_work yes\n"
      "# profile_nep_setup_compute_potential_copy_scope position_H2D_three_output_clears_full_NEP_compute_per_atom_potential_D2H\n"
      "# profile_energy_path full_nep_compute_energy_readback\n"
      "# profile_full_nep_energy_readback_calls %lld\n"
      "# profile_full_nep_energy_readback_seconds %.16e\n"
      "# profile_derivative_nep_calls %lld\n# profile_derivative_nep_seconds %.16e\n"
      "# profile_derivative_force_edge_extract_merge_seconds %.16e\n"
      "# profile_energy_geometry_calls %lld\n# profile_energy_geometry_seconds %.16e\n"
      "# profile_derivative_geometry_calls %lld\n# profile_derivative_geometry_seconds %.16e\n"
      "# profile_nep_eval_requested %lld\n# profile_nep_eval_executed %lld\n# profile_nep_eval_cache_hits %lld\n"
      "# profile_geometry_eval_requested %lld\n# profile_geometry_eval_executed %lld\n# profile_geometry_eval_cache_hits %lld\n"
      "# profile_static_snapshot_count %d\n# profile_static_snapshot_wall_seconds %.16e\n"
      "# profile_static_snapshot_nep_requests %lld\n# profile_static_snapshot_nep_computes %lld\n# profile_static_snapshot_nep_cache_hits %lld\n"
      "# profile_static_snapshot_geometry_requests %lld\n# profile_static_snapshot_geometry_evaluations %lld\n"
      "# profile_complex_pi_candidate_seconds %.16e\n# profile_candidate_time_overlaps_nep_and_sample_wall yes\n"
      "# profile_hac_current_lookup_seconds %.16e\n# profile_sample_wall_seconds_including_end_of_step_output_io %.16e\n"
      "# n_static_frames_total %lld\n# n_static_frames_used %lld\n"
      "# n_static_frames_skipped_nan %lld\n"
      "# n_static_frames_skipped_other %lld\n",
      profile_sample_count_, profile_energy_only_nep_calls_, profile_energy_only_nep_seconds_,
      profile_derivative_nep_calls_, profile_derivative_nep_seconds_, profile_derivative_edge_extract_seconds_,
      profile_energy_only_geometry_calls_, profile_energy_only_geometry_seconds_,
      profile_derivative_geometry_calls_, profile_derivative_geometry_seconds_,
      profile_nep_eval_requested_, profile_nep_eval_executed_, profile_nep_eval_cache_hits_,
      profile_geometry_eval_requested_, profile_geometry_eval_executed_, profile_geometry_eval_cache_hits_,
      static_profile_sample_count_, profile_static_wall_seconds_,
      static_profile_nep_eval_requested_, static_profile_nep_eval_executed_, static_profile_nep_eval_cache_hits_,
      static_profile_geometry_eval_requested_, static_profile_geometry_eval_executed_,
      profile_candidate_seconds_, profile_hac_lookup_seconds_, profile_sample_wall_seconds_,
      static_frames_total_, static_frames_used_,
      static_frames_skipped_nan_, static_frames_skipped_other_);
    std::fprintf(meta_file_, "# winding_diagnostic_frames %lld\n", winding_sample_count_);
    if (winding_sample_count_ > 0) {
      std::fprintf(meta_file_,
        "# final_nonzero_action_winding_atoms_max %d\n# final_max_abs_action_winding_component %d\n"
        "# final_nonzero_cartesian_closest_winding_atoms_max %d\n"
        "# final_max_abs_cartesian_closest_winding_component %d\n"
        "# final_cartesian_closest_winding_violation_count %lld\n",
        nonzero_action_winding_atoms_max_, max_abs_action_winding_component_,
        nonzero_closest_winding_atoms_max_, max_abs_closest_winding_component_,
        closest_winding_violation_total_);
    } else {
      std::fprintf(meta_file_, "# final_winding_diagnostics not_measured\n");
    }
    std::fprintf(meta_file_, "# end_segment\n");
    std::fflush(meta_file_);
  }
  if (profile_file_ != nullptr)
    std::fprintf(profile_file_, "# summary static_snapshot_count %d static_snapshot_wall_seconds %.16e private_nep_requests %lld private_nep_computes %lld private_nep_cache_hits %lld geometry_requests %lld geometry_evaluations %lld\n",
      static_profile_sample_count_, profile_static_wall_seconds_, static_profile_nep_eval_requested_,
      static_profile_nep_eval_executed_, static_profile_nep_eval_cache_hits_,
      static_profile_geometry_eval_requested_, static_profile_geometry_eval_executed_);
}

void QuantumHeatMoments::post_run(
  Atom&,
  Box&,
  Integrate&,
  const int,
  const double,
  const double)
{
  write_finalize_outputs();
  FILE* data_files[] = {static_file_, link_file_, imaginary_file_, estimator_stats_file_,
    fdcheck_file_, dynamic_file_, atom_debug_file_, edgecheck_file_, winding_file_, profile_file_};
  for (FILE* file : data_files)
    if (file != nullptr) std::fprintf(file, "# end_segment\n");
  close_files();
  geometry_cache_.clear();
}

void QuantumHeatMoments::close_files()
{
  FILE** files[] = {&static_file_, &link_file_, &imaginary_file_, &estimator_stats_file_,
    &fdcheck_file_, &dynamic_file_, &meta_file_, &atom_debug_file_, &edgecheck_file_,
    &winding_file_, &profile_file_};
  for (FILE** file : files) {
    if (*file != nullptr) {
      std::fflush(*file);
      std::fclose(*file);
      *file = nullptr;
    }
  }
}
