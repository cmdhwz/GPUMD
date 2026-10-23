#include "../src/measure/rpmd_ja_feasibility_policy.cuh"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace rpmd_ja_feasibility;

Candidate candidate(Family family, unsigned index, bool converged, double score, double uncertainty,
  const std::vector<double>& block_scores = std::vector<double>())
{
  Candidate result;
  result.family = family;
  result.index = index;
  result.converged = converged;
  result.development_objective = score;
  result.development_objective_uncertainty = uncertainty;
  result.development_block_scores_b = block_scores.empty() ? std::vector<double>(4, score) : block_scores;
  result.development_block_scores_2b = result.development_block_scores_b;
  return result;
}

int main()
{
  const std::uint64_t resident_current[]={64,32},resident_same[]={64,16};
  std::uint64_t workspace_peak=99;
  assert(estimate_workspace_growth_peak_bytes(resident_current,resident_same,2,workspace_peak)&&workspace_peak==0);
  const std::uint64_t resident_growth[]={80,48};
  assert(estimate_workspace_growth_peak_bytes(resident_current,resident_growth,2,workspace_peak)&&workspace_peak==80);
  assert(workspace_peak<=80&&!(workspace_peak<=79));
  const std::uint64_t solver_current[]={64},solver_growth[]={96};
  assert(estimate_workspace_growth_peak_bytes(solver_current,solver_growth,1,workspace_peak)&&workspace_peak==96);
  assert(!estimate_workspace_growth_peak_bytes(nullptr,solver_growth,1,workspace_peak));

  CurvaturePerformanceWindow performance_window;
  assert(reset_curvature_performance_window(performance_window,0.5,10.0));
  assert(record_curvature_performance_gradient(performance_window,0.5,0.25));
  for(int build=0;build<3;++build)
    assert(record_curvature_performance_build(performance_window,0.5,10.0,0.4));
  assert(finish_curvature_performance_window(performance_window,0.5,10.0)==
    CurvaturePerformanceDecision::NUMERICAL_INCONCLUSIVE);
  assert(performance_window.curvature_builds==3); // Preserve the stop witness for a checkpoint/restart.

  CurvaturePerformanceWindow improving_window;
  assert(reset_curvature_performance_window(improving_window,0.5,10.0));
  assert(record_curvature_performance_gradient(improving_window,0.5,0.25));
  for(int build=0;build<3;++build)
    assert(record_curvature_performance_build(improving_window,0.5,9.98,0.4));
  assert(finish_curvature_performance_window(improving_window,0.5,9.98)==CurvaturePerformanceDecision::CONTINUE);
  assert(improving_window.curvature_builds==0);

  CurvaturePerformanceWindow cost_equal_window;
  assert(reset_curvature_performance_window(cost_equal_window,0.5,10.0));
  assert(record_curvature_performance_gradient(cost_equal_window,0.5,0.3));
  for(int build=0;build<3;++build)
    assert(record_curvature_performance_build(cost_equal_window,0.5,10.0,0.1));
  assert(finish_curvature_performance_window(cost_equal_window,0.5,10.0)==CurvaturePerformanceDecision::CONTINUE);

  CurvaturePerformanceWindow tau_window;
  assert(reset_curvature_performance_window(tau_window,0.5,10.0));
  assert(record_curvature_performance_build(tau_window,0.5,10.0,0.4));
  assert(record_curvature_performance_build(tau_window,0.5,10.0,0.4));
  assert(record_curvature_performance_build(tau_window,0.25,8.0,0.4));
  assert(tau_window.curvature_builds==1&&tau_window.tau==0.25&&tau_window.start_objective==8.0);
  assert(finish_curvature_performance_window(tau_window,0.25,8.0)==CurvaturePerformanceDecision::COLLECTING);

  assert(should_run_structured_family(true, false));
  assert(!should_run_structured_family(true, true));
  assert(!should_run_structured_family(false, false));
  assert(common_soft_probe_capacity(3, 4, 28) == 3);
  assert(common_soft_probe_capacity(8, 4, 28) == 3);
  assert(common_soft_probe_capacity(3, 1, 28) == 0);
  const std::string b_gate_attempt = feasibility_cg_witness_path("witness", "COMMON_PROBE_RESCORING",
    "M0_B0", "B_GATE", 2, 1);
  assert(b_gate_attempt != feasibility_cg_witness_path("witness", "COMMON_PROBE_RESCORING",
    "M0_B0", "B_GATE", 2, 2));
  assert(b_gate_attempt != feasibility_cg_witness_path("witness", "COMMON_PROBE_RESCORING",
    "M0_B0", "FINAL_M0_M1", 2, 1));
  assert(b_gate_attempt != feasibility_cg_witness_path("witness", "COMMON_PROBE_RESCORING",
    "M0_B0", "B_GATE", 3, 1));
  const Limits defaults;
  assert(defaults.wall_seconds == 28800.0);
  assert(defaults.max_exact_gradients == 240);
  assert(defaults.validation_reserve_fraction == 0.25);
  assert(defaults.max_a_branch_gradients == 8);
  assert(defaults.max_fit_family_gradients == 112);
  assert(defaults.a_wall_fraction == 0.10);
  assert(defaults.max_fit_family_wall_seconds == -1.0);

  Limits invalid = defaults;
  invalid.wall_seconds = -1.0;
  assert(!invalid.valid());
  invalid = defaults;
  invalid.max_exact_gradients = -1;
  assert(!invalid.valid());
  invalid = defaults;
  invalid.validation_reserve_fraction = std::numeric_limits<double>::infinity();
  assert(!invalid.valid());

  Budget budget;
  assert(begin(budget, defaults, 0.0) == BudgetResult::OK);
  assert(begin(budget, defaults, 10.0) == BudgetResult::ALREADY_STARTED);
  for (int i = 0; i < 240; ++i) {
    assert(reserve_exact_gradient(budget, defaults, 0.0) == BudgetResult::OK);
  }
  assert(reserve_exact_gradient(budget, defaults, 0.0) == BudgetResult::GRADIENT_LIMIT);
  assert(budget.exact_gradients == 240);

  Budget branches;
  Limits two_gradients = defaults;
  two_gradients.max_exact_gradients = 2;
  assert(begin(branches, two_gradients, 0.0) == BudgetResult::OK);
  assert(reserve_exact_gradient(branches, two_gradients, 1.0) == BudgetResult::OK);
  assert(enter_phase(branches, two_gradients, Phase::B, 2.0) == BudgetResult::OK);
  assert(reserve_exact_gradient(branches, two_gradients, 3.0) == BudgetResult::OK);
  assert(enter_phase(branches, two_gradients, Phase::C, 4.0) == BudgetResult::OK);
  assert(reserve_exact_gradient(branches, two_gradients, 5.0) == BudgetResult::GRADIENT_LIMIT);

  Budget reservation;
  assert(begin(reservation, defaults, 0.0) == BudgetResult::OK);
  assert(enter_phase(reservation, defaults, Phase::B, 1.0) == BudgetResult::OK);
  assert(observe_time(reservation, defaults, 21600.0) == BudgetResult::VALIDATION_RESERVED);
  assert(reserve_exact_gradient(reservation, defaults, 21600.0) == BudgetResult::VALIDATION_RESERVED);
  assert(enter_phase(reservation, defaults, Phase::FREEZE, 21600.0) == BudgetResult::OK);
  assert(enter_phase(reservation, defaults, Phase::BLIND, 21600.0) == BudgetResult::OK);
  assert(reserve_exact_gradient(reservation, defaults, 21600.0) == BudgetResult::OK);

  Budget clock;
  assert(begin(clock, defaults, 10.0) == BudgetResult::OK);
  assert(observe_time(clock, defaults, std::numeric_limits<double>::quiet_NaN()) == BudgetResult::INVALID_CLOCK);
  assert(observe_time(clock, defaults, 9.0) == BudgetResult::CLOCK_BACKWARDS);
  assert(clock.last_seconds == 10.0 && clock.elapsed_seconds == 0.0);

  Budget a_branches;
  assert(begin(a_branches, defaults, 0.0) == BudgetResult::OK);
  for (int i = 0; i < 8; ++i) {
    assert(enter_phase(a_branches, defaults, Phase::A0, 0.0) == BudgetResult::OK);
    assert(reserve_exact_gradient(a_branches, defaults, 0.0) == BudgetResult::OK);
  }
  assert(reserve_exact_gradient(a_branches, defaults, 0.0) == BudgetResult::PHASE_GRADIENT_LIMIT);
  assert(enter_phase(a_branches, defaults, Phase::A1, 0.0) == BudgetResult::OK);
  for (int i = 0; i < 8; ++i) {
    assert(reserve_exact_gradient(a_branches, defaults, 0.0) == BudgetResult::OK);
  }
  assert(reserve_exact_gradient(a_branches, defaults, 0.0) == BudgetResult::PHASE_GRADIENT_LIMIT);
  assert(enter_phase(a_branches, defaults, Phase::B, 2880.0) == BudgetResult::OK);
  assert(enter_phase(a_branches, defaults, Phase::A1, 2881.0) == BudgetResult::A_WALL_LIMIT);

  Budget family_caps;
  Limits small_family = defaults;
  small_family.max_exact_gradients = 20;
  small_family.max_fit_family_gradients = 3;
  assert(begin(family_caps, small_family, 0.0) == BudgetResult::OK);
  assert(equal_remaining_family_gradient_cap(family_caps, small_family) == 10);
  assert(enter_phase(family_caps, small_family, Phase::B, 1.0) == BudgetResult::OK);
  for (int i = 0; i < 3; ++i)
    assert(reserve_exact_gradient(family_caps, small_family, 1.0) == BudgetResult::OK);
  assert(reserve_exact_gradient(family_caps, small_family, 1.0) == BudgetResult::FAMILY_GRADIENT_LIMIT);
  assert(enter_phase(family_caps, small_family, Phase::C, 2.0) == BudgetResult::OK);
  for (int i = 0; i < 3; ++i)
    assert(reserve_exact_gradient(family_caps, small_family, 2.0) == BudgetResult::OK);
  assert(family_caps.exact_gradients == 6);
  assert(equal_remaining_family_gradient_cap(family_caps, small_family) == 7);

  Budget timed_families;
  Limits timed_limits = defaults;
  timed_limits.wall_seconds = 100.0;
  timed_limits.max_exact_gradients = 20;
  assert(begin(timed_families, timed_limits, 0.0) == BudgetResult::OK);
  assert(enter_phase(timed_families, timed_limits, Phase::A1, 10.0) == BudgetResult::OK);
  timed_limits.max_fit_family_wall_seconds = equal_remaining_family_wall_cap(timed_families, timed_limits);
  assert(std::abs(timed_limits.max_fit_family_wall_seconds - 32.5) < 1.0e-12);
  assert(enter_phase(timed_families, timed_limits, Phase::B, 10.0) == BudgetResult::OK);
  assert(observe_time(timed_families, timed_limits, 36.0) == BudgetResult::OK); // B candidate fit.
  assert(observe_time(timed_families, timed_limits, 42.5) == BudgetResult::FAMILY_WALL_LIMIT); // B common-probe gate.
  assert(enter_phase(timed_families, timed_limits, Phase::C, 42.5) == BudgetResult::OK);
  assert(reserve_exact_gradient(timed_families, timed_limits, 42.5) == BudgetResult::OK);
  assert(observe_time(timed_families, timed_limits, 75.0) == BudgetResult::FAMILY_WALL_LIMIT);
  assert(enter_phase(timed_families, timed_limits, Phase::FREEZE, 75.0) == BudgetResult::OK);

  Budget single_gate_step;
  Limits single_gate_limits = timed_limits;
  single_gate_limits.max_fit_family_wall_seconds = 32.5;
  assert(begin(single_gate_step, single_gate_limits, 0.0) == BudgetResult::OK);
  assert(enter_phase(single_gate_step, single_gate_limits, Phase::A1, 10.0) == BudgetResult::OK);
  assert(enter_phase(single_gate_step, single_gate_limits, Phase::B, 10.0) == BudgetResult::OK);
  assert(observe_time(single_gate_step, single_gate_limits, 42.0) == BudgetResult::OK);
  assert(observe_time(single_gate_step, single_gate_limits, 43.25) == BudgetResult::FAMILY_WALL_LIMIT);
  const double single_step_overrun = single_gate_step.phase_seconds[phase_index(Phase::B)] -
    single_gate_limits.max_fit_family_wall_seconds;
  assert(std::abs(single_step_overrun - 0.75) < 1.0e-12);
  assert(enter_phase(single_gate_step, single_gate_limits, Phase::C, 43.25) == BudgetResult::OK);
  const double c_wall_available = single_gate_limits.wall_seconds - single_gate_step.elapsed_seconds -
    single_gate_limits.wall_seconds * single_gate_limits.validation_reserve_fraction;
  assert(std::abs(c_wall_available - (single_gate_limits.max_fit_family_wall_seconds - single_step_overrun)) < 1.0e-12);

  Budget measured_cost;
  Limits measured_cost_limits = defaults;
  measured_cost_limits.wall_seconds = 100.0;
  measured_cost_limits.validation_reserve_fraction = 0.25;
  measured_cost_limits.max_fit_family_wall_seconds = 20.0;
  assert(begin(measured_cost, measured_cost_limits, 0.0) == BudgetResult::OK);
  assert(enter_phase(measured_cost, measured_cost_limits, Phase::B, 10.0) == BudgetResult::OK);
  CandidateCostProfile candidate_costs;
  assert(record_candidate_cost(candidate_costs, 12.0));
  assert(record_candidate_cost(candidate_costs, 8.0));
  auto cost_gate = measured_candidate_cost_gate(measured_cost, measured_cost_limits, candidate_costs, 1.5);
  assert(cost_gate.result == CandidateCostGateResult::ALLOW);
  assert(std::abs(cost_gate.projected_seconds - 18.0) < 1.0e-12);
  assert(observe_time(measured_cost, measured_cost_limits, 30.0) == BudgetResult::FAMILY_WALL_LIMIT);
  cost_gate = measured_candidate_cost_gate(measured_cost, measured_cost_limits, candidate_costs, 1.5);
  assert(cost_gate.result == CandidateCostGateResult::INSUFFICIENT_REMAINING_FIT_TIME);
  assert(cost_gate.available_seconds == 0.0);
  assert(enter_phase(measured_cost, measured_cost_limits, Phase::C, 30.0) == BudgetResult::OK);
  cost_gate = measured_candidate_cost_gate(measured_cost, measured_cost_limits, candidate_costs, 1.5);
  assert(cost_gate.result == CandidateCostGateResult::ALLOW);
  assert(cost_gate.available_seconds == 20.0);
  Budget incomplete_candidate_budget;
  Limits incomplete_candidate_limits = defaults;
  incomplete_candidate_limits.wall_seconds = 500.0;
  incomplete_candidate_limits.validation_reserve_fraction = 0.25;
  incomplete_candidate_limits.max_fit_family_wall_seconds = 200.0;
  assert(begin(incomplete_candidate_budget, incomplete_candidate_limits, 0.0) == BudgetResult::OK);
  assert(enter_phase(incomplete_candidate_budget, incomplete_candidate_limits, Phase::B, 5.0) == BudgetResult::OK);
  assert(observe_time(incomplete_candidate_budget, incomplete_candidate_limits, 125.0) == BudgetResult::OK);
  CandidateCostProfile incomplete_candidate_costs;
  assert(record_candidate_cost(incomplete_candidate_costs, 5.0));
  assert(record_partial_candidate_elapsed(incomplete_candidate_costs, 120.0));
  cost_gate = measured_candidate_cost_gate(incomplete_candidate_budget, incomplete_candidate_limits,
    incomplete_candidate_costs, 1.5);
  assert(cost_gate.result == CandidateCostGateResult::INSUFFICIENT_REMAINING_FIT_TIME);
  assert(cost_gate.available_seconds == 80.0 && cost_gate.projected_seconds == 180.0);
  assert(incomplete_candidate_costs.samples == 1 && incomplete_candidate_costs.maximum_seconds == 5.0);
  assert(incomplete_candidate_costs.partial_elapsed_lower_bound_seconds == 120.0);
  assert(enter_phase(incomplete_candidate_budget, incomplete_candidate_limits, Phase::C, 125.0) == BudgetResult::OK);
  assert(measured_candidate_cost_gate(incomplete_candidate_budget, incomplete_candidate_limits,
    incomplete_candidate_costs, 1.5).result == CandidateCostGateResult::ALLOW);
  CandidateCostProfile no_costs;
  assert(measured_candidate_cost_gate(measured_cost, measured_cost_limits, no_costs, 1.5).result ==
    CandidateCostGateResult::NO_MEASURED_BASELINE);
  assert(!record_candidate_cost(candidate_costs, std::numeric_limits<double>::quiet_NaN()));
  assert(!record_candidate_cost(candidate_costs, 0.0));
  assert(measured_candidate_cost_gate(measured_cost, measured_cost_limits, candidate_costs, 11.0).result ==
    CandidateCostGateResult::INVALID_ESTIMATE);

  Candidate candidates[] = {
    candidate(Family::M0, 0, false, 0.001, 0.0),
    candidate(Family::M1, 0, true, 0.010, 0.0),
  };
  FrozenSelection selected;
  assert(choose_development_candidate(candidates, 2, selected) == SelectionResult::SELECTED);
  assert(selected.family == Family::M1 && selected.candidate_index == 0);

  Candidate tied[] = {
    candidate(Family::M1, 0, true, 0.01, 0.02, std::vector<double>(4, 0.0)),
    candidate(Family::M0, 0, true, 0.02, 0.02, std::vector<double>(4, 0.0)),
  };
  FrozenSelection m0_selection;
  assert(choose_development_candidate(tied, 2, m0_selection) == SelectionResult::SELECTED);
  assert(m0_selection.family == Family::M0);

  Candidate paired[] = {
    candidate(Family::M0, 0, true, 50.0, 57.8, std::vector<double>{0.0, 100.0, 0.0, 100.0}),
    candidate(Family::M1, 0, true, 10.0, 57.8, std::vector<double>{-40.0, 60.0, -40.0, 60.0}),
  };
  FrozenSelection paired_selection;
  assert(choose_development_candidate(paired, 2, paired_selection) == SelectionResult::SELECTED);
  assert(paired_selection.family == Family::M1);

  Candidate two_candidates[] = {
    candidate(Family::M0, 0, true, 10.0, 0.0, std::vector<double>(4, 10.0)),
    candidate(Family::M1, 0, true, 1.0, 0.0, std::vector<double>(4, 1.0)),
  };
  FrozenSelection two_selection;
  assert(choose_development_candidate(two_candidates, 2, two_selection) == SelectionResult::SELECTED);
  assert(two_selection.family == Family::M1 && two_selection.candidate_index == 0);
  Candidate reversed_two[] = {two_candidates[1], two_candidates[0]};
  FrozenSelection reversed_two_selection;
  assert(choose_development_candidate(reversed_two, 2, reversed_two_selection) == SelectionResult::SELECTED);
  assert(reversed_two_selection.family == Family::M1 && reversed_two_selection.candidate_index == 0);

  Candidate three_candidates[] = {
    candidate(Family::M0, 0, true, 10.0, 0.0, std::vector<double>(4, 10.0)),
    candidate(Family::M0, 1, true, 5.0, 0.0, std::vector<double>(4, 5.0)),
    candidate(Family::M1, 0, true, 1.0, 0.0, std::vector<double>(4, 1.0)),
  };
  FrozenSelection three_selection;
  assert(choose_development_candidate(three_candidates, 3, three_selection) == SelectionResult::SELECTED);
  assert(three_selection.family == Family::M1 && three_selection.candidate_index == 0);
  Candidate permuted_three[] = {three_candidates[2], three_candidates[0], three_candidates[1]};
  FrozenSelection permuted_three_selection;
  assert(choose_development_candidate(permuted_three, 3, permuted_three_selection) == SelectionResult::SELECTED);
  assert(permuted_three_selection.family == Family::M1 && permuted_three_selection.candidate_index == 0);

  Candidate mismatched_blocks[] = {
    candidate(Family::M0, 0, true, 1.0, 0.0, std::vector<double>(4, 1.0)),
    candidate(Family::M1, 0, true, 0.0, 0.0, std::vector<double>(5, 0.0)),
  };
  FrozenSelection invalid_paired_selection;
  assert(choose_development_candidate(mismatched_blocks, 2, invalid_paired_selection) ==
    SelectionResult::INVALID_PAIRED_BLOCKS);
  Candidate scale_disagreement[] = {
    candidate(Family::M0, 0, true, 0.0, 0.0, std::vector<double>(4, 0.0)),
    candidate(Family::M1, 0, true, 0.0, 0.0, std::vector<double>(4, 1.0)),
  };
  scale_disagreement[1].development_block_scores_2b.assign(4, -1.0);
  FrozenSelection inconclusive_selection;
  assert(choose_development_candidate(scale_disagreement, 2, inconclusive_selection) ==
    SelectionResult::DATA_INCONCLUSIVE);
  assert(open_blind_once(m0_selection));
  assert(!open_blind_once(m0_selection));
  assert(classify_blind_result(m0_selection, false, true) == BlindStatus::DATA_INCONCLUSIVE);
  assert(choose_development_candidate(candidates, 2, m0_selection) == SelectionResult::ALREADY_FROZEN);
  assert(classify_blind_result(m0_selection, true, false) == BlindStatus::BLIND_FAILED);
  assert(classify_blind_result(m0_selection, true, true) == BlindStatus::BLIND_PASSED);
  assert(!regularization_sensitivity_allows_blind(true, false));
  assert(!regularization_sensitivity_allows_blind(false, true));
  assert(regularization_sensitivity_allows_blind(false, false));

  Candidate unconverged[] = {candidate(Family::M0, 0, false, 0.0, 0.0)};
  FrozenSelection none;
  assert(choose_development_candidate(unconverged, 1, none) == SelectionResult::NO_ELIGIBLE);
  Candidate missing_blocks = candidate(Family::M0, 0, true, 0.0, 0.0);
  missing_blocks.development_block_scores_b.clear();
  assert(choose_development_candidate(&missing_blocks, 1, none) == SelectionResult::NO_ELIGIBLE);
  Candidate over_cap[] = {
    candidate(Family::M0, 0, true, 0.0, 0.0), candidate(Family::M0, 1, true, 0.0, 0.0),
    candidate(Family::M0, 2, true, 0.0, 0.0), candidate(Family::M0, 3, true, 0.0, 0.0),
  };
  assert(choose_development_candidate(over_cap, 4, none) == SelectionResult::TOO_MANY_CANDIDATES);

  assert(can_compare_a_tau1(1.0, 1.0));
  assert(!can_compare_a_tau1(1.0, 0.2));
  assert(!can_compare_a_tau1(std::numeric_limits<double>::quiet_NaN(), 1.0));

  ResourcePlan resource_plan;
  for (unsigned stage = 0; stage < resource_stage_count; ++stage) {
    resource_plan.stages[stage].host_known = true;
    resource_plan.stages[stage].device_known = true;
    resource_plan.stages[stage].disk_known = true;
  }
  auto& fit_resources = resource_plan.stages[resource_stage_index(ResourceStage::FIT)];
  fit_resources.persistent_host_bytes = 100;
  fit_resources.transient_host_bytes = 10;
  fit_resources.host_library_workspace_bytes = 5;
  fit_resources.persistent_device_bytes = 200;
  fit_resources.transient_device_bytes = 30;
  fit_resources.device_library_workspace_bytes = 70;
  fit_resources.persistent_disk_bytes = 1000;
  fit_resources.transient_disk_bytes = 20;
  auto& curvature_resources = resource_plan.stages[resource_stage_index(ResourceStage::CURVATURE)];
  curvature_resources.persistent_host_bytes = 100;
  curvature_resources.transient_host_bytes = 20;
  curvature_resources.host_library_workspace_bytes = 3;
  curvature_resources.persistent_device_bytes = 200;
  curvature_resources.transient_device_bytes = 50;
  curvature_resources.device_library_workspace_bytes = 80;
  auto& package_resources = resource_plan.stages[resource_stage_index(ResourceStage::PACKAGE_WRITE)];
  package_resources.persistent_disk_bytes = 3000;
  package_resources.transient_disk_bytes = 2000;
  const ResourcePeakEstimate peaks = estimate_resource_peaks(resource_plan);
  assert(peaks.status == ResourcePlanStatus::COMPLETE_ESTIMATE);
  assert(peaks.host_peak_bytes == 123 && peaks.host_peak_stage == ResourceStage::CURVATURE);
  assert(peaks.device_peak_bytes == 330 && peaks.device_peak_stage == ResourceStage::CURVATURE);
  assert(peaks.disk_peak_bytes == 5000 && peaks.disk_peak_stage == ResourceStage::PACKAGE_WRITE);
  assert(resource_capacity_gate(peaks, 123, 330, 5000) == ResourceGateResult::PEAK_WITHIN_CAPACITY_ESTIMATE);
  assert(resource_capacity_gate(peaks, 123, 329, 5000) == ResourceGateResult::PEAK_EXCEEDS_CAPACITY);
  resource_plan.stages[resource_stage_index(ResourceStage::READBACK)].host_known = false;
  const ResourcePeakEstimate unknown_peaks = estimate_resource_peaks(resource_plan);
  assert(unknown_peaks.status == ResourcePlanStatus::INCONCLUSIVE);
  assert(resource_capacity_gate(unknown_peaks, 1000, 1000, 10000) == ResourceGateResult::INCONCLUSIVE);
  resource_plan.stages[resource_stage_index(ResourceStage::READBACK)].host_known = true;
  resource_plan.stages[resource_stage_index(ResourceStage::READBACK)].transient_host_bytes =
    std::numeric_limits<std::uint64_t>::max();
  assert(estimate_resource_peaks(resource_plan).status == ResourcePlanStatus::INVALID);

  Limits stage_cost_limits = defaults;
  stage_cost_limits.wall_seconds = 1000.0;
  stage_cost_limits.validation_reserve_fraction = 0.25;
  stage_cost_limits.max_fit_family_wall_seconds = 400.0;
  Budget stage_cost_budget;
  assert(begin(stage_cost_budget, stage_cost_limits, 0.0) == BudgetResult::OK);
  assert(enter_phase(stage_cost_budget, stage_cost_limits, Phase::B, 100.0) == BudgetResult::OK);
  assert(observe_time(stage_cost_budget, stage_cost_limits, 200.0) == BudgetResult::OK);
  StageCostProfile stage_costs[resource_stage_count];
  assert(record_stage_cost(stage_costs[resource_stage_index(ResourceStage::FIT)], 10.0));
  assert(record_stage_cost(stage_costs[resource_stage_index(ResourceStage::CURVATURE)], 20.0));
  assert(record_stage_cost(stage_costs[resource_stage_index(ResourceStage::BLIND)], 30.0));
  assert(record_stage_cost(stage_costs[resource_stage_index(ResourceStage::PACKAGE_WRITE)], 30.0));
  assert(record_stage_cost(stage_costs[resource_stage_index(ResourceStage::PREPARE)], 40.0));
  assert(record_stage_cost(stage_costs[resource_stage_index(ResourceStage::READBACK)], 10.0));
  RemainingWork remaining_work;
  remaining_work.calls[resource_stage_index(ResourceStage::FIT)] = 2;
  remaining_work.calls[resource_stage_index(ResourceStage::CURVATURE)] = 1;
  remaining_work.calls[resource_stage_index(ResourceStage::BLIND)] = 1;
  remaining_work.calls[resource_stage_index(ResourceStage::PACKAGE_WRITE)] = 1;
  remaining_work.calls[resource_stage_index(ResourceStage::PREPARE)] = 1;
  remaining_work.calls[resource_stage_index(ResourceStage::READBACK)] = 1;
  const CompletionCostPrediction cost_plan = predict_remaining_cost(
    stage_cost_budget, stage_cost_limits, stage_costs, remaining_work, 2.0);
  assert(cost_plan.status == CostPlanStatus::ESTIMATED_WITHIN_BUDGET);
  assert(cost_plan.fit_seconds == 80.0 && cost_plan.validation_seconds == 220.0);
  assert(cost_plan.available_fit_seconds == 300.0 && cost_plan.available_total_seconds == 800.0);
  RemainingWork one_candidate_work;
  one_candidate_work.calls[resource_stage_index(ResourceStage::FIT)] = 1;
  one_candidate_work.calls[resource_stage_index(ResourceStage::CURVATURE)] = 1;
  one_candidate_work.calls[resource_stage_index(ResourceStage::BLIND)] = 1;
  one_candidate_work.calls[resource_stage_index(ResourceStage::PACKAGE_WRITE)] = 1;
  one_candidate_work.calls[resource_stage_index(ResourceStage::PREPARE)] = 1;
  one_candidate_work.calls[resource_stage_index(ResourceStage::READBACK)] = 1;
  assert(predict_remaining_cost(stage_cost_budget, stage_cost_limits, stage_costs, one_candidate_work, 2.0).status ==
    CostPlanStatus::ESTIMATED_WITHIN_BUDGET);
  RemainingWork candidate_only_work;
  candidate_only_work.calls[resource_stage_index(ResourceStage::FIT)] = 1;
  candidate_only_work.calls[resource_stage_index(ResourceStage::CURVATURE)] = 1;
  Limits tight_completion_limits = stage_cost_limits;
  tight_completion_limits.wall_seconds = 450.0;
  assert(predict_remaining_cost(stage_cost_budget, tight_completion_limits, stage_costs, candidate_only_work, 2.0).status ==
    CostPlanStatus::ESTIMATED_WITHIN_BUDGET);
  assert(predict_remaining_cost(stage_cost_budget, tight_completion_limits, stage_costs, one_candidate_work, 2.0).status ==
    CostPlanStatus::INSUFFICIENT_REMAINING_BUDGET);
  RemainingWork all_family_work = one_candidate_work;
  all_family_work.calls[resource_stage_index(ResourceStage::FIT)] = 6;
  all_family_work.calls[resource_stage_index(ResourceStage::CURVATURE)] = 6;
  assert(predict_remaining_cost(stage_cost_budget, stage_cost_limits, stage_costs, all_family_work, 2.0).status ==
    CostPlanStatus::INSUFFICIENT_REMAINING_BUDGET);
  stage_costs[resource_stage_index(ResourceStage::READBACK)] = StageCostProfile{};
  assert(predict_remaining_cost(stage_cost_budget, stage_cost_limits, stage_costs, remaining_work, 2.0).status ==
    CostPlanStatus::INCONCLUSIVE_NO_STAGE_BASELINE);
  assert(record_partial_stage_cost(stage_costs[resource_stage_index(ResourceStage::READBACK)], 500.0));
  assert(predict_remaining_cost(stage_cost_budget, stage_cost_limits, stage_costs, remaining_work, 2.0).status ==
    CostPlanStatus::INSUFFICIENT_REMAINING_BUDGET);

  assert(!blind_statistics_declares_drift("DATA_INCONCLUSIVE", "DEVELOPMENT_BLOCK_SENSITIVE"));
  assert(blind_statistics_declares_drift("DATA_INCONCLUSIVE", "FIXED_R0_MEAN_OR_DRIFT"));
  assert(!blind_statistics_declares_drift("PASS", "FIXED_R0_MEAN_OR_DRIFT"));

  ResourcePlan lower_bound_plan;
  auto& known_prepare = lower_bound_plan.stages[resource_stage_index(ResourceStage::PREPARE)];
  known_prepare.device_known = false;
  known_prepare.persistent_device_bytes = 100;
  known_prepare.transient_device_bytes = 200;
  const ResourcePeakEstimate lower_bound_peak = estimate_resource_peaks(lower_bound_plan);
  assert(lower_bound_peak.status == ResourcePlanStatus::INCONCLUSIVE);
  assert(lower_bound_peak.device_peak_bytes == 300);
  assert(resource_capacity_gate(lower_bound_peak, 1000, 1000, 1000) == ResourceGateResult::INCONCLUSIVE);
}
