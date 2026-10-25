#pragma once

#include <cmath>
#include <cstdint>
#include <string>

class Atom;
class Box;
class Force;

enum class RpmdJANativeFitMethod
{
  ForceLeastSquares,
  GaussianCovarianceShrinkage
};

inline bool rpmd_ja_covariance_prior_ridge_valid(const double rho,const double ridge_lambda)
{
  return std::isfinite(rho)&&rho>=0.0&&std::isfinite(ridge_lambda)&&ridge_lambda>=0.0&&
    (rho>0.0||ridge_lambda>0.0);
}

struct RpmdJANativeFitOptions
{
  std::string output_path;
  std::string kernel_table;
  std::string raw_input_path;
  std::string baseline_state_path;
  double temperature = 0.0;
  double cutoff = 0.0;
  double epsilon = 0.0;
  double response_tolerance = 0.0;
  double ibp_tolerance = 0.15;
  double fd_step = 0.0;
  int sample_interval = 0;
  int max_stability_rounds = 160;
  int max_covariance_iterations = 0;
  double covariance_shrinkage = 0.0;
  double covariance_ridge_lambda = 0.0;
  int covariance_analytic_curvature_probes = 0;
  bool covariance_feasibility = false;
  bool graph_reference = false;
  bool covariance_structured_basis = false;
  std::string covariance_blind_samples;
  std::string covariance_blind_manifest_path;
  bool covariance_block_newton = false;
  bool covariance_joint_block = false;
  bool covariance_trust_region = false;
  bool covariance_staged_barrier = false;
  RpmdJANativeFitMethod method = RpmdJANativeFitMethod::ForceLeastSquares;
  bool internal_mass_com = false;
};

bool fit_rpmd_ja_native_reference(
  const RpmdJANativeFitOptions& options,
  const std::string& spool_path,
  std::uint64_t frame_count,
  Atom& atom,
  Box& box,
  Force& force);

void fit_rpmd_ja_native_reference_from_samples(
  const RpmdJANativeFitOptions& options,
  const std::string& spool_path,
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

void check_rpmd_ja_native_fit_samples(
  const std::string& spool_path,
  const std::string& report_path,
  Atom& atom,
  Box& box);

void replay_rpmd_ja_native_cg(
  const std::string& witness_path,
  const std::string& report_path,
  const bool internal_mass_com,
  Atom& atom,
  Box& box,
  Force& force);

void diagnose_rpmd_ja_native_covariance(
  const std::string& state_path,
  const std::string& report_path,
  double cutoff,
  double epsilon,
  double tau,
  double fd_step,
  bool internal_mass_com,
  Atom& atom,
  Box& box,
  Force& force);
