/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "property.cuh"
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

struct GeometryResultGPU
{
  int valid;
  int assignment_ambiguous;
  int pair_conflict;
  int nearest_oxygen;
  int oxygen_low;
  int oxygen_high;
  int candidate_count;
  double delta;
  double dOO;
  double rperp;
  double path_excess;
  double assignment_score;
  double second_assignment_score;
  double assignment_score_gap;
  double low_to_high_dx;
  double low_to_high_dy;
  double low_to_high_dz;
  double E_ion_nominal_parallel;
  double delta_phi_ion;
  int nearest_ion_id;
  double nearest_ion_distance;
  double nearest_ion1_distance;
  double nearest_ion2_distance;
  double nearest_ion1_to_low;
  double nearest_ion1_to_high;
  double nearest_ion2_to_low;
  double nearest_ion2_to_high;
};

struct LocalEnvironmentGPU
{
  int valid;
  double rOH_low;
  double rOH_high;
  double oho_angle;
  double dOO;
  double rperp;
  double path_excess;
  double ion1_low_d1;
  double ion1_low_d2;
  double ion1_low_d3;
  double ion1_high_d1;
  double ion1_high_d2;
  double ion1_high_d3;
  double ion2_low_d1;
  double ion2_low_d2;
  double ion2_low_d3;
  double ion2_high_d1;
  double ion2_high_d2;
  double ion2_high_d3;
  int coord_ion1_low;
  int coord_ion1_high;
  int coord_ion2_low;
  int coord_ion2_high;
  double delta_d_ion1;
  double delta_d_ion2;
  double nearest_ion2_to_H;
  double angle_Olow_H_ion2;
  double angle_Ohigh_H_ion2;
  int hcl_like_low;
  int hcl_like_high;
  double E_parallel;
  double delta_phi_ion;
};

struct LocalEnvironment
{
  bool valid = false;
  double rOH_low = 0.0;
  double rOH_high = 0.0;
  double oho_angle = 0.0;
  double dOO = 0.0;
  double rperp = 0.0;
  double path_excess = 0.0;
  int nH_low = 0;
  int nH_high = 0;
  int donor_edges_low = 0;
  int donor_edges_high = 0;
  int acceptor_edges_low = 0;
  int acceptor_edges_high = 0;
  double ion1_low_d1 = 0.0;
  double ion1_low_d2 = 0.0;
  double ion1_low_d3 = 0.0;
  double ion1_high_d1 = 0.0;
  double ion1_high_d2 = 0.0;
  double ion1_high_d3 = 0.0;
  double ion2_low_d1 = 0.0;
  double ion2_low_d2 = 0.0;
  double ion2_low_d3 = 0.0;
  double ion2_high_d1 = 0.0;
  double ion2_high_d2 = 0.0;
  double ion2_high_d3 = 0.0;
  int coord_ion1_low = 0;
  int coord_ion1_high = 0;
  int coord_ion2_low = 0;
  int coord_ion2_high = 0;
  double delta_d_ion1 = 0.0;
  double delta_d_ion2 = 0.0;
  double nearest_ion2_to_H = 0.0;
  double angle_Olow_H_ion2 = 0.0;
  double angle_Ohigh_H_ion2 = 0.0;
  int hcl_like_low = 0;
  int hcl_like_high = 0;
  double E_parallel = 0.0;
  double delta_phi_ion = 0.0;
};

class Proton_Tunneling : public Property
{
public:
  Proton_Tunneling(const char** param, const int num_param, const Atom& atom);

  void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void process(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

  void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  enum class QuantumCharacter
  {
    CLASSICAL_ONLY,
    TWO_WELL_DELOCALIZED,
    BARRIER_CENTERED_TUNNELING_LIKE,
    COMPACT_SINGLE_DOMAIN,
    MULTI_KINK_OR_MULTI_DOMAIN,
    AMBIGUOUS
  };

  struct BeadDiagnostic
  {
    bool valid = false;
    int num_beads = 0;
    int n_minus = 0;
    int n_zero = 0;
    int n_plus = 0;
    int kink_count = 0;
    int center_domain_count = 0;
    int total_state_domain_count = 0;
    int two_well_occupied = 0;
    int two_well_span = 0;
    int simple_two_domain_path = 0;
    int barrier_centered = 0;
    int strict_tunneling_like = 0;
    int channel_valid_count = 0;
    int multi_kink_or_multi_domain = 0;
    double probe_time_fs = 0.0;
    double delta_centroid = 0.0;
    double f_minus = 0.0;
    double f_zero = 0.0;
    double f_plus = 0.0;
    double f_channel_valid = 0.0;
    double mean_delta = 0.0;
    double centroid_minus_mean = 0.0;
    double sigma_delta = 0.0;
    double delta_min = 0.0;
    double delta_max = 0.0;
    double span = 0.0;
    double delta_q20 = 0.0;
    double delta_q80 = 0.0;
    double robust_span = 0.0;
    double rms_neighbor_delta_jump = 0.0;
    double max_neighbor_delta_jump = 0.0;
    LocalEnvironment environment;
    QuantumCharacter character = QuantumCharacter::AMBIGUOUS;
  };

  enum class AttemptOutcome
  {
    success,
    return_to_state,
    geometry_lost,
    run_end
  };

  struct AttemptRecord
  {
    long long attempt_id = 0;
    double time_start_fs = 0.0;
    double time_end_fs = 0.0;
    int hydrogen = -1;
    int oxygen_low = -1;
    int oxygen_high = -1;
    int oxygen_from = -1;
    int oxygen_target = -1;
    AttemptOutcome outcome = AttemptOutcome::run_end;
    double delta_start = 0.0;
    double min_abs_delta = 0.0;
    double delta_end = 0.0;
    double E_parallel_start = 0.0;
    double E_parallel_end = 0.0;
    double delta_phi_start = 0.0;
    double delta_phi_end = 0.0;
    double delta_d_ion1_start = 0.0;
    double delta_d_ion2_start = 0.0;
    int nearest_ion_id = -1;
    double nearest_ion_distance = 0.0;
    BeadDiagnostic centroid_best;
    BeadDiagnostic delocalization_best;
    bool has_transfer = false;
    int nH_from_before = 0;
    int nH_to_before = 0;
    int nH_from_after = 0;
    int nH_to_after = 0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    LocalEnvironment environment_start;
    LocalEnvironment environment_end;
    LocalEnvironment environment_last_valid;
  };

  struct DefectRecord
  {
    double time_fs = 0.0;
    int oxygen = -1;
    int q_defect = 0;
    int hydrogen_count = 0;
    long long cause_event_id = -1;
  };

  struct WindowRecord
  {
    long long window_id = 0;
    double time_start_fs = 0.0;
    double time_end_fs = 0.0;
    double B_mean = 0.0;
    double f_02 = 0.0;
    double f_04 = 0.0;
    double mean_abs_delta_f = 0.0;
    double flip_rate = 0.0;
    int active_bonds = 0;
    double positive_defects = 0.0;
    double negative_defects = 0.0;
    double valid_pairs_per_frame = 0.0;
    long long assignment_ambiguous_samples = 0;
    long long pair_conflict_samples = 0;
  };

  struct EdgeWindowRecord
  {
    long long window_id = 0;
    double time_start_fs = 0.0;
    double time_end_fs = 0.0;
    int oxygen_low = -1;
    int oxygen_high = -1;
    double geometry_occupancy = 0.0;
    long long n_plus = 0;
    long long n_minus = 0;
    long long n_deadband = 0;
    double asymmetry = 0.0;
    double abs_asymmetry = 0.0;
    double delta_f = 0.0;
    long long attempts = 0;
    long long successes = 0;
    long long returns = 0;
    long long geometry_lost = 0;
    double success_probability = 0.0;
    double mean_delta = 0.0;
    double mean_abs_delta = 0.0;
    double mean_dOO = 0.0;
    double mean_rperp = 0.0;
    double mean_E_parallel = 0.0;
    double std_E_parallel = 0.0;
    double corr_delta_E_parallel = 0.0;
    double mean_E_success = 0.0;
    double mean_E_return = 0.0;
    double nearest_ion1_distance = 0.0;
    double nearest_ion2_distance = 0.0;
    double log_population_ratio = 0.0;
    double beta_DeltaF_high_minus_low = 0.0;
    double abs_beta_DeltaF = 0.0;
    double mean_delta_phi_ion = 0.0;
    double std_delta_phi_ion = 0.0;
    double corr_delta_delta_phi = 0.0;
    double mean_ion1_to_O_low = 0.0;
    double mean_ion1_to_O_high = 0.0;
    double mean_delta_d_ion1 = 0.0;
    double mean_ion2_to_O_low = 0.0;
    double mean_ion2_to_O_high = 0.0;
    double mean_delta_d_ion2 = 0.0;
  };

  struct GeometryResult
  {
    bool valid = false;
    bool assignment_ambiguous = false;
    bool pair_conflict = false;
    int nearest_oxygen = -1;
    int oxygen_low = -1;
    int oxygen_high = -1;
    int candidate_count = 0;
    double delta = 0.0;
    double dOO = 0.0;
    double rperp = 0.0;
    double path_excess = 0.0;
    double assignment_score = 0.0;
    double second_assignment_score = 0.0;
    double assignment_score_gap = 0.0;
    double low_to_high_dx = 0.0;
    double low_to_high_dy = 0.0;
    double low_to_high_dz = 0.0;
    double E_ion_nominal_parallel = 0.0;
    double delta_phi_ion = 0.0;
    int nearest_ion_id = -1;
    double nearest_ion_distance = 0.0;
    double nearest_ion1_distance = 0.0;
    double nearest_ion2_distance = 0.0;
    double nearest_ion1_to_low = 0.0;
    double nearest_ion1_to_high = 0.0;
    double nearest_ion2_to_low = 0.0;
    double nearest_ion2_to_high = 0.0;
  };

  struct BondStats
  {
    long long geometry_samples = 0;
    long long n_plus = 0;
    long long n_minus = 0;
    long long n_deadband = 0;
    long long transitions = 0;
    long long attempts = 0;
    long long successes = 0;
    long long returns = 0;
    long long geometry_lost = 0;
    double sum_abs_delta = 0.0;
    double sum_delta = 0.0;
    double sum_delta_square = 0.0;
    double sum_dOO = 0.0;
    double sum_rperp = 0.0;
    double sum_E_parallel = 0.0;
    double sum_E2_parallel = 0.0;
    double sum_delta_E_parallel = 0.0;
    double sum_E_success = 0.0;
    double sum_E_return = 0.0;
    double sum_nearest_ion1_distance = 0.0;
    double sum_nearest_ion2_distance = 0.0;
    double sum_delta_phi = 0.0;
    double sum_delta_phi2 = 0.0;
    double sum_delta_delta_phi = 0.0;
    double sum_ion1_to_low = 0.0;
    double sum_ion1_to_high = 0.0;
    double sum_ion2_to_low = 0.0;
    double sum_ion2_to_high = 0.0;
    long long n_E_success = 0;
    long long n_E_return = 0;
  };

  struct LocalEnvironmentStats
  {
    long long samples = 0;
    long long attempts = 0;
    long long successes = 0;
    long long returns = 0;
    long long geometry_lost = 0;
    double sum_delta = 0.0;
    double sum_delta2 = 0.0;
    double sum_rOH_low = 0.0;
    double sum_rOH_high = 0.0;
    double sum_oho_angle = 0.0;
    double sum_dOO = 0.0;
    double sum_rperp = 0.0;
    double sum_path_excess = 0.0;
    long long sum_nH_low = 0;
    long long sum_nH_high = 0;
    long long sum_donor_edges_low = 0;
    long long sum_donor_edges_high = 0;
    long long sum_acceptor_edges_low = 0;
    long long sum_acceptor_edges_high = 0;
    double sum_ion1_low_d1 = 0.0;
    double sum_ion1_low_d2 = 0.0;
    double sum_ion1_low_d3 = 0.0;
    double sum_ion1_high_d1 = 0.0;
    double sum_ion1_high_d2 = 0.0;
    double sum_ion1_high_d3 = 0.0;
    double sum_ion2_low_d1 = 0.0;
    double sum_ion2_low_d2 = 0.0;
    double sum_ion2_low_d3 = 0.0;
    double sum_ion2_high_d1 = 0.0;
    double sum_ion2_high_d2 = 0.0;
    double sum_ion2_high_d3 = 0.0;
    long long count_ion1_low[3] = {};
    long long count_ion1_high[3] = {};
    long long count_ion2_low[3] = {};
    long long count_ion2_high[3] = {};
    long long sum_coord_ion1_low = 0;
    long long sum_coord_ion1_high = 0;
    long long sum_coord_ion2_low = 0;
    long long sum_coord_ion2_high = 0;
    double sum_delta_d_ion1 = 0.0;
    double sum_delta_d_ion2 = 0.0;
    long long count_delta_d_ion1 = 0;
    long long count_delta_d_ion2 = 0;
    double sum_nearest_ion2_to_H = 0.0;
    double sum_angle_Olow_H_ion2 = 0.0;
    double sum_angle_Ohigh_H_ion2 = 0.0;
    long long sum_hcl_like_low = 0;
    long long sum_hcl_like_high = 0;
    long long count_nearest_ion2_to_H = 0;
    long long count_angle_Olow_H_ion2 = 0;
    long long count_angle_Ohigh_H_ion2 = 0;
    double sum_E_parallel = 0.0;
    double sum_delta_phi_ion = 0.0;
    double sum_delta_phi2 = 0.0;
    double sum_delta_delta_phi = 0.0;
  };

  struct LocalEnvironmentWindowRecord
  {
    long long window_id = 0;
    double time_start_fs = 0.0;
    double time_end_fs = 0.0;
    int oxygen_low = -1;
    int oxygen_high = -1;
    LocalEnvironmentStats stats;
  };

  struct HydrogenState
  {
    int oxygen_low = -1;
    int oxygen_high = -1;
    int stable_state = 0;
    int pending_state = 0;
    int pending_count = 0;
    double last_delta = 0.0;
    double last_E_parallel = 0.0;
    double last_delta_phi_ion = 0.0;
    int last_nearest_ion_id = -1;
    double last_nearest_ion_distance = 0.0;
    bool attempt_active = false;
    int attempt_from_state = 0;
    long long attempt_id = 0;
    double attempt_start_time_fs = 0.0;
    double attempt_delta_start = 0.0;
    double attempt_E_start = 0.0;
    double attempt_delta_phi_start = 0.0;
    double attempt_delta_d_ion1_start = 0.0;
    double attempt_delta_d_ion2_start = 0.0;
    double attempt_min_abs_delta = 0.0;
    double pending_start_time_fs = 0.0;
    LocalEnvironment last_environment;
    LocalEnvironment attempt_environment_start;
    BeadDiagnostic centroid_best;
    BeadDiagnostic delocalization_best;
  };

  void parse(const char** param, const int num_param, const Atom& atom);
  void build_oxygen_shell(const Box& box);
  void initialize_geometry_gpu();
  void compute_geometry_gpu(const Box& box, Atom& atom);
  void compute_local_environment_gpu(const Box& box, Atom& atom);
  void assign_local_environment_topology();
  void release_geometry_timing_events();
  void compute_ion_field(const Box& box, GeometryResult& geometry) const;
  bool ensure_bead_positions(Atom& atom);
  bool evaluate_bead_diagnostic(
    Atom& atom,
    const Box& box,
    const int hydrogen,
    const GeometryResult& geometry,
    const double probe_time_fs,
    BeadDiagnostic& diagnostic);
  QuantumCharacter classify_quantum_character(const BeadDiagnostic& diagnostic) const;
  bool is_better_delocalization_diagnostic(
    const BeadDiagnostic& candidate,
    const BeadDiagnostic& current) const;
  const char* quantum_character_name(const QuantumCharacter character) const;
  bool find_geometry(
    const int hydrogen,
    const Box& box,
    GeometryResult& geometry) const;
  int classify_delta(const double delta) const;
  void record_bond(
    std::unordered_map<unsigned long long, BondStats>& bond_stats,
    const GeometryResult& geometry,
    const int state);
  void start_attempt(
    HydrogenState& hydrogen_state,
    const int stable_state,
    const double time_fs,
    const GeometryResult& geometry,
    const LocalEnvironment& environment);
  void finish_attempt(
    const int hydrogen,
    HydrogenState& hydrogen_state,
    const AttemptOutcome outcome,
    const double time_fs,
    const double delta_end,
    const GeometryResult* geometry,
    const LocalEnvironment* environment);
  const char* outcome_name(const AttemptOutcome outcome) const;
  void observe_frame(const double time_fs, const Box& box, Atom& atom);
  void write_window(const double time_fs);
  void write_edge_window(const double time_start_fs, const double time_end_fs);
  void record_local_environment(
    const GeometryResult& geometry,
    const LocalEnvironment& environment);
  void write_local_environment_window(const double time_start_fs, const double time_end_fs);
  void write_local_environment_event(
    FILE* file,
    const AttemptRecord& attempt,
    const char* snapshot_kind,
    const double time_fs,
    const LocalEnvironment& environment,
    const char* quantum_class);
  void write_final_bonds();
  void write_output_files();

  int sample_interval_ = 1;
  int window_samples_ = 1000;
  int hold_samples_ = 2;
  int window_sample_count_ = 0;
  int number_of_atoms_ = 0;
  long long window_id_ = 0;
  long long next_attempt_id_ = 1;
  long long window_flip_count_ = 0;
  long long window_valid_pair_count_ = 0;
  long long window_positive_defect_sum_ = 0;
  long long window_negative_defect_sum_ = 0;
  long long window_assignment_ambiguous_count_ = 0;
  long long window_pair_conflict_count_ = 0;
  double delta_cutoff_ = 0.10;
  double dOO_min_ = 2.20;
  double dOO_max_ = 2.60;
  double rperp_max_ = 0.80;
  double oho_angle_min_deg_ = 120.0;
  bool bead_diagnostic_enabled_ = false;
  double bead_f_min_ = 0.25;
  double bead_center_max_ = 0.30;
  double bead_centroid_max_ = 0.10;
  double bead_span_min_ = 0.20;
  bool local_environment_enabled_ = false;
  double local_ion1_cutoff_ = 0.0;
  double local_ion2_cutoff_ = 0.0;
  double local_hcl_cutoff_ = 3.0;
  double local_hcl_angle_min_deg_ = 150.0;
  int oxygen_shell_k_ = 8;
  double assignment_path_excess_weight_ = 0.50;
  double assignment_score_gap_min_ = 0.05;
  bool ion_field_enabled_ = false;
  std::string ion1_symbol_ = "Na";
  std::string ion2_symbol_ = "Cl";
  double ion1_charge_ = 1.0;
  double ion2_charge_ = -1.0;
  double ion_field_cutoff_ = 8.0;
  double time_step_ = 0.0;
  double window_start_time_fs_ = 0.0;
  double last_time_fs_ = 0.0;
  bool initialized_ = false;

  std::string oxygen_symbol_ = "O";
  std::string hydrogen_symbol_ = "H";
  std::vector<int> oxygen_indices_;
  std::vector<int> hydrogen_indices_;
  std::vector<int> ion1_indices_;
  std::vector<int> ion2_indices_;
  std::vector<int> oxygen_local_index_;
  std::vector<std::vector<int>> oxygen_shell_neighbors_;
  std::vector<int> oxygen_shell_offsets_cpu_;
  std::vector<int> oxygen_shell_neighbors_cpu_;
  GPU_Vector<int> oxygen_indices_gpu_;
  GPU_Vector<int> hydrogen_indices_gpu_;
  GPU_Vector<int> oxygen_local_index_gpu_;
  GPU_Vector<int> oxygen_shell_offsets_gpu_;
  GPU_Vector<int> oxygen_shell_neighbors_gpu_;
  GPU_Vector<int> ion1_indices_gpu_;
  GPU_Vector<int> ion2_indices_gpu_;
  GPU_Vector<GeometryResultGPU> frame_geometries_gpu_;
  GPU_Vector<LocalEnvironmentGPU> frame_local_environments_gpu_;
  std::vector<double> cpu_position_;
  std::vector<double> cpu_position_beads_;
  std::vector<GeometryResultGPU> cpu_geometry_gpu_;
  std::vector<LocalEnvironmentGPU> cpu_local_environments_gpu_;
  std::vector<double*> bead_position_ptrs_cpu_;
  GPU_Vector<double*> bead_position_ptrs_gpu_;
  GPU_Vector<double> bead_position_staging_gpu_;
  bool bead_positions_cached_ = false;
  int cached_number_of_beads_ = 0;
  long long observer_frame_count_ = 0;
  long long bead_probe_frame_count_ = 0;
  long long bead_d2h_copy_count_ = 0;
  unsigned long long bead_bytes_copied_ = 0;
  double bead_copy_wall_time_ = 0.0;
  double bead_analysis_wall_time_ = 0.0;
  double total_observer_wall_time_ = 0.0;
  double geometry_kernel_wall_time_ = 0.0;
  double geometry_D2H_wall_time_ = 0.0;
  double state_machine_wall_time_ = 0.0;
  long long local_environment_copy_count_ = 0;
  unsigned long long local_environment_bytes_copied_ = 0;
  double local_environment_kernel_wall_time_ = 0.0;
  double local_environment_D2H_wall_time_ = 0.0;
  double local_environment_host_analysis_wall_time_ = 0.0;
  gpuEvent_t geometry_kernel_start_event_ = nullptr;
  gpuEvent_t geometry_kernel_end_event_ = nullptr;
  gpuEvent_t local_environment_kernel_start_event_ = nullptr;
  gpuEvent_t local_environment_kernel_end_event_ = nullptr;
  std::vector<int> hydrogen_count_;
  std::vector<int> previous_hydrogen_count_;
  std::vector<int> event_hydrogen_count_;
  std::vector<long long> frame_cause_event_ids_;
  std::vector<GeometryResult> frame_geometries_;
  std::vector<LocalEnvironment> frame_local_environments_;
  std::vector<HydrogenState> hydrogen_states_;
  bool defect_state_initialized_ = false;

  std::vector<AttemptRecord> attempt_records_;
  std::vector<DefectRecord> defect_records_;
  std::vector<WindowRecord> window_records_;
  std::vector<EdgeWindowRecord> edge_window_records_;
  std::vector<LocalEnvironmentWindowRecord> local_environment_window_records_;

  std::unordered_map<unsigned long long, BondStats> window_bonds_;
  std::unordered_map<unsigned long long, BondStats> total_bonds_;
  std::unordered_map<unsigned long long, LocalEnvironmentStats> window_local_environment_stats_;
  FILE* bias_file_ = nullptr;
  FILE* transfer_file_ = nullptr;
  FILE* attempt_file_ = nullptr;
  FILE* bead_event_file_ = nullptr;
  FILE* defect_file_ = nullptr;
  FILE* edge_window_file_ = nullptr;
  FILE* bond_file_ = nullptr;
  FILE* local_environment_window_file_ = nullptr;
  FILE* local_environment_event_file_ = nullptr;
};
