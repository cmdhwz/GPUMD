#pragma once

#include "action.cuh"
#include "force/nep.cuh"
#include "quantum_heat_moments_math.cuh"
#include "utilities/gpu_vector.cuh"
#include <array>
#include <cstdio>
#include <deque>
#include <memory>
#include <string>
#include <vector>

class HAC;

class QuantumHeatMoments : public Action
{
public:
  QuantumHeatMoments(const char** param, int num_param);
  ~QuantumHeatMoments() override;

  void set_hac(HAC* hac) { hac_ = hac; }

  void pre_run(
    int number_of_steps,
    double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void end_of_step(
    int number_of_steps,
    int step,
    int fixed_group,
    int move_group,
    double global_time,
    double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    int number_of_steps,
    double time_step,
    double temperature) override;

  double energy(const std::vector<double>& position);
  std::vector<double> force(const std::vector<double>& position);
  quantum_heat_moments::Complex evaluate_A0(
    const std::vector<double>& position,
    const std::vector<quantum_heat_moments::Complex>& momentum,
    int alpha,
    double fd_step_r);

private:
  struct Geometry
  {
    std::vector<double> position;
    std::vector<double> potential;
    std::vector<double> force;
    std::vector<double> transport;
    std::vector<NEP_Local_Edge> edges;
    double total_potential = 0.0;
    bool has_derivatives = false;
    bool has_edges = false;
  };

  struct A0Parts
  {
    quantum_heat_moments::Complex linear = 0.0;
    quantum_heat_moments::Complex cubic = 0.0;
  };

  struct RingPath
  {
    std::vector<std::vector<double>> wrapped;
    std::vector<std::vector<double>> unwrapped;
    std::vector<std::vector<double>> link_displacement;
    std::vector<int> action_link_lift;
    std::vector<int> closest_link_lift;
    std::vector<int> action_winding;
    std::vector<int> closest_winding;
    std::vector<double> centroid;
    int nonzero_action_winding_atoms = 0;
    int max_abs_action_winding_component = 0;
    int nonzero_closest_winding_atoms = 0;
    int max_abs_closest_winding_component = 0;
    int link_image_total = 0;
    int link_image_mismatch_count = 0;
    double max_link_length_action = 0.0;
    double max_link_length_cartesian_closest = 0.0;
  };

  int sample_interval_ = 1;
  int static_sample_interval_ = 0;
  int dynamic_sample_interval_ = 0;
  int candidate_sample_interval_ = 0;
  int weyl_max_order_ = 0;
  int required_operator_order_ = 0;
  int exact_static_max_order_ = 0;
  int n_moyal_probe_ = 64;
  int n_static_trace_probe_ = 64;
  int n_aux_probe_ = 0;
  int imag_lag_max_ = 4;
  int output_level_ = 0;
  std::uint64_t seed_ = 12345;
  double fd_step_r_ = 1.0e-4;
  double fd_step_p_ = 1.0e-4;
  double edgecheck_step_ = 1.0e-4;
  double fd_warning_threshold_ = 0.05;
  double stochastic_warning_threshold_ = 0.10;
  bool compute_exact_mu0_ = false;
  bool exact_mu0_enabled_ = false;
  bool exact_mu2_enabled_ = false;
  bool exact_mu4_enabled_ = false;
  bool exact_mu6_enabled_ = false;
  bool enable_complex_pi_ = false;
  bool short_range_mechanical_only_ = true;
  bool force_debug_large_ = false;
  bool validate_edge_derivatives_ = false;
  bool validate_fd_smoothness_ = false;
  bool validate_link_images_ = false;
  bool dynamic_enabled_ = false;
  bool pimd_a0_only_enabled_ = false;
  bool diagnostic_noncanonical_ = false;
  bool unique_image_safe_ = false;
  int number_of_atoms_ = 0;
  int number_of_beads_ = 0;
  int number_of_steps_ = 0;
  double time_step_ = 0.0;
  double temperature_start_ = 0.0;
  double temperature_end_ = 0.0;
  double cutoff_ = 0.0;
  double shortest_lattice_vector_ = 0.0;
  double cutoff_image_ratio_ = 0.0;
  double expensive_static_warning_threshold_ = 50000.0;
  double estimated_nep_evaluations_per_static_frame_ = 0.0;
  double estimated_geometry_evaluations_per_static_frame_ = 0.0;
  long long energy_only_nep_calls_ = 0;
  long long derivative_nep_calls_ = 0;
  long long geometry_eval_requested_ = 0;
  long long geometry_eval_executed_ = 0;
  long long geometry_eval_cache_hits_ = 0;
  long long nep_eval_requested_ = 0;
  long long nep_eval_executed_ = 0;
  long long nep_eval_cache_hits_ = 0;
  double energy_only_nep_seconds_ = 0.0;
  double derivative_nep_seconds_ = 0.0;
  long long energy_only_geometry_calls_ = 0;
  long long derivative_geometry_calls_ = 0;
  double energy_only_geometry_seconds_ = 0.0;
  double derivative_geometry_seconds_ = 0.0;
  double derivative_edge_extract_seconds_ = 0.0;
  int profile_sample_count_ = 0;
  long long profile_energy_only_nep_calls_ = 0;
  long long profile_derivative_nep_calls_ = 0;
  double profile_energy_only_nep_seconds_ = 0.0;
  double profile_derivative_nep_seconds_ = 0.0;
  long long profile_energy_only_geometry_calls_ = 0;
  long long profile_derivative_geometry_calls_ = 0;
  double profile_energy_only_geometry_seconds_ = 0.0;
  double profile_derivative_geometry_seconds_ = 0.0;
  double profile_derivative_edge_extract_seconds_ = 0.0;
  double profile_candidate_seconds_ = 0.0;
  double profile_hac_lookup_seconds_ = 0.0;
  double profile_sample_wall_seconds_ = 0.0;
  long long profile_geometry_eval_requested_ = 0;
  long long profile_geometry_eval_executed_ = 0;
  long long profile_geometry_eval_cache_hits_ = 0;
  long long profile_nep_eval_requested_ = 0;
  long long profile_nep_eval_executed_ = 0;
  long long profile_nep_eval_cache_hits_ = 0;
  double profile_static_wall_seconds_ = 0.0;
  int static_profile_sample_count_ = 0;
  long long static_profile_nep_eval_requested_ = 0;
  long long static_profile_nep_eval_executed_ = 0;
  long long static_profile_nep_eval_cache_hits_ = 0;
  long long static_profile_geometry_eval_requested_ = 0;
  long long static_profile_geometry_eval_executed_ = 0;
  long long static_frames_total_ = 0;
  long long static_frames_used_ = 0;
  long long static_frames_skipped_nan_ = 0;
  long long static_frames_skipped_other_ = 0;
  int nonzero_action_winding_atoms_max_ = 0;
  int max_abs_action_winding_component_ = 0;
  int nonzero_closest_winding_atoms_max_ = 0;
  int max_abs_closest_winding_component_ = 0;
  long long winding_sample_count_ = 0;
  long long closest_winding_violation_total_ = 0;
  std::string model_path_storage_;
  std::string pimd_action_scheme_ = "primitive_symmetric";
  std::vector<double> mass_by_atom_;
  std::vector<int> atom_type_host_;
  std::vector<double> mass_by_dof_;
  std::vector<std::vector<double>> bead_position_host_;
  std::vector<std::vector<double>> bead_velocity_host_;
  std::deque<std::shared_ptr<Geometry>> geometry_cache_;
  std::unique_ptr<NEP> nep_sr_;
  GPU_Vector<int> type_gpu_;
  GPU_Vector<double> position_gpu_;
  GPU_Vector<double> potential_gpu_;
  GPU_Vector<double> force_gpu_;
  GPU_Vector<double> virial_gpu_;
  std::array<quantum_heat_moments::ComplexStats, 3> mu2_stats_;
  std::array<quantum_heat_moments::ComplexStats, 3> mu0_stats_;
  std::array<quantum_heat_moments::ComplexStats, 3> mu4_stats_;
  std::array<std::array<std::vector<quantum_heat_moments::ComplexStats>, 3>, 3> imaginary_stats_;
  std::array<std::array<quantum_heat_moments::ComplexStats, 4>, 3> candidate_pi_stats_;
  std::array<std::array<int, 4>, 3> candidate_invalid_frame_count_{};
  Box* box_ = nullptr;
  HAC* hac_ = nullptr;
  FILE* static_file_ = nullptr;
  FILE* link_file_ = nullptr;
  FILE* imaginary_file_ = nullptr;
  FILE* estimator_stats_file_ = nullptr;
  FILE* fdcheck_file_ = nullptr;
  FILE* dynamic_file_ = nullptr;
  FILE* meta_file_ = nullptr;
  FILE* atom_debug_file_ = nullptr;
  FILE* edgecheck_file_ = nullptr;
  FILE* winding_file_ = nullptr;
  FILE* profile_file_ = nullptr;
  std::string segment_id_;
  std::string append_contract_;
  unsigned long long model_fingerprint_ = 0;

  std::shared_ptr<Geometry> evaluate_geometry(
    const std::vector<double>& position,
    bool with_derivatives = true,
    bool retain_edges = false);
  A0Parts evaluate_A0_parts(
    const std::vector<double>& position,
    const std::vector<quantum_heat_moments::Complex>& momentum,
    int alpha,
    double fd_step_r);
  std::vector<double> linear_coefficients(const Geometry& geometry, int alpha) const;
  double evaluate_A1_constant(const std::vector<double>& position, int alpha, double step);
  double evaluate_D_contraction(
    const std::vector<double>& position,
    const std::vector<double>& first,
    const std::vector<double>& second,
    const std::vector<double>& third,
    int alpha,
    double step_r,
    double step_p);
  double evaluate_E_contraction(
    const std::vector<double>& position,
    const std::vector<double>& direction,
    int alpha,
    double step_r,
    double step_p);
  quantum_heat_moments::Complex evaluate_gamma0(
    const std::vector<double>& position,
    const std::vector<double>& link_displacement,
    int alpha,
    double beta,
    double step,
    const std::vector<quantum_heat_moments::MoyalProbe>& trace_probes,
    double& imaginary_residual);
  double evaluate_B(
    const std::vector<double>& position,
    const std::vector<double>& left,
    const std::vector<double>& right,
    int alpha,
    double step);
  double evaluate_gamma1(
    const std::vector<double>& position,
    const std::vector<double>& link_displacement,
    int alpha,
    double beta,
    double step,
    const std::vector<quantum_heat_moments::MoyalProbe>& moyal_probes,
    const std::vector<quantum_heat_moments::MoyalProbe>& trace_probes,
    quantum_heat_moments::ComplexStats& moyal_stats,
    quantum_heat_moments::ComplexStats& trace_stats,
    std::vector<quantum_heat_moments::Complex>& moyal_samples,
    std::vector<quantum_heat_moments::Complex>& trace_samples);
  quantum_heat_moments::Complex evaluate_gamma2(
    const std::vector<double>& position,
    const std::vector<double>& link_displacement,
    int alpha,
    double beta,
    double step_r,
    double step_p,
    const std::vector<quantum_heat_moments::MoyalProbe>& probes,
    double& imaginary_residual,
    quantum_heat_moments::ComplexStats& contraction_stats);
  void reconstruct_bead_chain(
    const Box& box,
    const std::vector<std::vector<double>>& positions,
    RingPath& path,
    bool compute_link_diagnostics) const;
  void validate_local_edge_derivatives(int step, const std::vector<double>& position, const Geometry& geometry);
  void write_winding_summary(int step, const RingPath& path);
  void write_profile_frame(int step, int beads, double static_wall_seconds, int moyal_probes, int trace_probes);
  void write_meta(const Box& box, double temperature);
  void write_headers();
  void write_atom_debug(int step, const RingPath& path);
  void write_finalize_outputs();
  void close_files();
  FILE* open_segment_file(const char* filename);
};
