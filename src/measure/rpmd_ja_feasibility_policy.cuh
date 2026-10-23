#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace rpmd_ja_feasibility
{
inline bool blind_statistics_declares_drift(const std::string& status,const std::string& reason)
{
  return status=="DATA_INCONCLUSIVE"&&reason=="FIXED_R0_MEAN_OR_DRIFT";
}

inline std::string feasibility_cg_witness_path(const std::string& base,const std::string& stage,
  const std::string& candidate,const std::string& generation,const int probe,const std::uint64_t attempt)
{
  return base+".feasibility."+stage+"."+candidate+"."+generation+".probe"+std::to_string(probe)+
    ".attempt"+std::to_string(attempt)+".txt";
}

enum class Phase
{
  PREFLIGHT,
  A0,
  A1,
  B,
  C,
  FREEZE,
  BLIND,
  PREPARE
};

constexpr unsigned candidate_performance_curvature_window=3;
constexpr double candidate_performance_min_relative_objective_improvement=0.001;
constexpr double candidate_performance_curvature_to_gradient_cost_ratio=1.0;
constexpr const char* candidate_performance_stop_rule=
  "THREE_CURVATURE_BUILDS_GT_ONE_MAX_FULL_GRADIENT_LT_0.001_RELATIVE_OBJECTIVE_IMPROVEMENT_SAME_TAU_V1";
constexpr const char* candidate_performance_objective_normalizer="MAX_1_ABS_WINDOW_START_OBJECTIVE";
constexpr const char* candidate_performance_gradient_cost_rule=
  "MAX_MEASURED_FULL_EXACT_GRADIENT_SECONDS_WITHIN_SAME_TAU_WINDOW";

struct CurvaturePerformanceWindow
{
  bool initialized=false;
  unsigned curvature_builds=0;
  double tau=0.0,start_objective=0.0,curvature_seconds=0.0,reference_gradient_seconds=0.0;
};

enum class CurvaturePerformanceDecision
{
  COLLECTING,
  CONTINUE,
  NUMERICAL_INCONCLUSIVE
};

inline bool reset_curvature_performance_window(CurvaturePerformanceWindow& window,const double tau,
  const double objective,const double reference_gradient_seconds=0.0)
{
  if(!std::isfinite(tau)||!(tau>0.0)||!std::isfinite(objective)||
     !std::isfinite(reference_gradient_seconds)||reference_gradient_seconds<0.0)return false;
  window.initialized=true;window.curvature_builds=0;window.tau=tau;window.start_objective=objective;
  window.curvature_seconds=0.0;window.reference_gradient_seconds=reference_gradient_seconds;return true;
}

inline bool record_curvature_performance_gradient(CurvaturePerformanceWindow& window,const double tau,
  const double exact_gradient_seconds)
{
  if(!window.initialized||!std::isfinite(tau)||!(tau>0.0)||window.tau!=tau||
     !std::isfinite(exact_gradient_seconds)||exact_gradient_seconds<0.0)return false;
  window.reference_gradient_seconds=std::max(window.reference_gradient_seconds,exact_gradient_seconds);
  return true;
}

inline bool record_curvature_performance_build(CurvaturePerformanceWindow& window,const double tau,
  const double objective,const double curvature_seconds)
{
  if(!std::isfinite(curvature_seconds)||curvature_seconds<0.0||!std::isfinite(objective)||!std::isfinite(tau)||!(tau>0.0))return false;
  if(!window.initialized||window.tau!=tau){if(!reset_curvature_performance_window(window,tau,objective))return false;}
  if(window.curvature_builds==std::numeric_limits<unsigned>::max()||
     !std::isfinite(window.curvature_seconds+curvature_seconds))return false;
  ++window.curvature_builds;window.curvature_seconds+=curvature_seconds;
  return true;
}

inline CurvaturePerformanceDecision finish_curvature_performance_window(CurvaturePerformanceWindow& window,
  const double tau,const double current_objective,const double next_window_gradient_seconds=0.0)
{
  if(!window.initialized||window.curvature_builds<candidate_performance_curvature_window)
    return CurvaturePerformanceDecision::COLLECTING;
  if(!std::isfinite(tau)||!(tau>0.0)||!std::isfinite(current_objective)||
     !std::isfinite(next_window_gradient_seconds)||next_window_gradient_seconds<0.0)return CurvaturePerformanceDecision::CONTINUE;
  if(window.tau!=tau){reset_curvature_performance_window(window,tau,current_objective);return CurvaturePerformanceDecision::CONTINUE;}
  const double relative_improvement=(window.start_objective-current_objective)/
    std::max(1.0,std::abs(window.start_objective));
  const bool stop=window.reference_gradient_seconds>0.0&&
    window.curvature_seconds>candidate_performance_curvature_to_gradient_cost_ratio*window.reference_gradient_seconds&&
    relative_improvement<candidate_performance_min_relative_objective_improvement;
  if(!stop)reset_curvature_performance_window(window,tau,current_objective,next_window_gradient_seconds);
  return stop?CurvaturePerformanceDecision::NUMERICAL_INCONCLUSIVE:CurvaturePerformanceDecision::CONTINUE;
}

enum class BudgetResult
{
  OK,
  INVALID_LIMITS,
  INVALID_CLOCK,
  CLOCK_BACKWARDS,
  NOT_STARTED,
  ALREADY_STARTED,
  INVALID_PHASE,
  WALL_LIMIT,
  GRADIENT_LIMIT,
  PHASE_GRADIENT_LIMIT,
  FAMILY_GRADIENT_LIMIT,
  FAMILY_WALL_LIMIT,
  A_WALL_LIMIT,
  VALIDATION_RESERVED
};

struct Limits
{
  double wall_seconds = 8.0 * 60.0 * 60.0;
  int max_exact_gradients = 240;
  double validation_reserve_fraction = 0.25;
  int max_a_branch_gradients = 8;
  int max_fit_family_gradients = 112;
  double a_wall_fraction = 0.10;
  double max_fit_family_wall_seconds = -1.0;

  bool valid() const
  {
    return std::isfinite(wall_seconds) && wall_seconds > 0.0 && max_exact_gradients > 0 &&
      std::isfinite(validation_reserve_fraction) && validation_reserve_fraction >= 0.0 &&
      validation_reserve_fraction < 1.0 && max_a_branch_gradients > 0 &&
      max_fit_family_gradients >= 0 &&
      std::isfinite(a_wall_fraction) && a_wall_fraction > 0.0 && a_wall_fraction <= 1.0 &&
      std::isfinite(max_fit_family_wall_seconds) && max_fit_family_wall_seconds >= -1.0;
  }
};

enum class ResourceStage : unsigned
{
  FIT,
  CURVATURE,
  BLIND,
  PACKAGE_WRITE,
  PREPARE,
  READBACK
};

constexpr unsigned resource_stage_count = 6;

inline unsigned resource_stage_index(const ResourceStage stage)
{
  return static_cast<unsigned>(stage);
}

struct StageResourceUsage
{
  std::uint64_t persistent_host_bytes = 0;
  std::uint64_t transient_host_bytes = 0;
  std::uint64_t host_library_workspace_bytes = 0;
  std::uint64_t persistent_device_bytes = 0;
  std::uint64_t transient_device_bytes = 0;
  std::uint64_t device_library_workspace_bytes = 0;
  std::uint64_t persistent_disk_bytes = 0;
  std::uint64_t transient_disk_bytes = 0;
  bool host_known = false;
  bool device_known = false;
  bool disk_known = false;
};

struct ResourcePlan
{
  StageResourceUsage stages[resource_stage_count] = {};
};

enum class ResourcePlanStatus
{
  COMPLETE_ESTIMATE,
  INCONCLUSIVE,
  INVALID
};

enum class ResourceGateResult
{
  PEAK_WITHIN_CAPACITY_ESTIMATE,
  PEAK_EXCEEDS_CAPACITY,
  INCONCLUSIVE,
  INVALID
};

struct ResourcePeakEstimate
{
  ResourcePlanStatus status = ResourcePlanStatus::INVALID;
  std::uint64_t host_peak_bytes = 0;
  std::uint64_t device_peak_bytes = 0;
  std::uint64_t disk_peak_bytes = 0;
  ResourceStage host_peak_stage = ResourceStage::FIT;
  ResourceStage device_peak_stage = ResourceStage::FIT;
  ResourceStage disk_peak_stage = ResourceStage::FIT;
};

inline bool checked_add_bytes(const std::uint64_t left, const std::uint64_t right, std::uint64_t& result)
{
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

inline bool estimate_workspace_growth_peak_bytes(const std::uint64_t* current_bytes,
  const std::uint64_t* requested_bytes,const std::size_t count,std::uint64_t& peak_bytes)
{
  peak_bytes=0;
  if(count>0&&(current_bytes==nullptr||requested_bytes==nullptr))return false;
  std::uint64_t committed_growth=0;
  for(std::size_t index=0;index<count;++index){
    const std::uint64_t current=current_bytes[index],requested=requested_bytes[index];
    if(requested<=current)continue;
    std::uint64_t allocation_peak=0;
    if(!checked_add_bytes(committed_growth,requested,allocation_peak))return false;
    peak_bytes=std::max(peak_bytes,allocation_peak);
    if(!checked_add_bytes(committed_growth,requested-current,committed_growth))return false;
  }
  return true;
}

inline ResourcePeakEstimate estimate_resource_peaks(const ResourcePlan& plan)
{
  ResourcePeakEstimate result;
  bool complete = true;
  for (unsigned i = 0; i < resource_stage_count; ++i) {
    const StageResourceUsage& stage = plan.stages[i];
    std::uint64_t host = 0, device = 0, disk = 0, partial = 0;
    if (!checked_add_bytes(stage.persistent_host_bytes, stage.transient_host_bytes, partial) ||
        !checked_add_bytes(partial, stage.host_library_workspace_bytes, host) ||
        !checked_add_bytes(stage.persistent_device_bytes, stage.transient_device_bytes, partial) ||
        !checked_add_bytes(partial, stage.device_library_workspace_bytes, device) ||
        !checked_add_bytes(stage.persistent_disk_bytes, stage.transient_disk_bytes, disk)) {
      result.status = ResourcePlanStatus::INVALID;
      return result;
    }
    const ResourceStage resource_stage = static_cast<ResourceStage>(i);
    if (host > result.host_peak_bytes) {
      result.host_peak_bytes = host;
      result.host_peak_stage = resource_stage;
    }
    if (device > result.device_peak_bytes) {
      result.device_peak_bytes = device;
      result.device_peak_stage = resource_stage;
    }
    if (disk > result.disk_peak_bytes) {
      result.disk_peak_bytes = disk;
      result.disk_peak_stage = resource_stage;
    }
    complete = complete && stage.host_known && stage.device_known && stage.disk_known;
  }
  result.status = complete ? ResourcePlanStatus::COMPLETE_ESTIMATE : ResourcePlanStatus::INCONCLUSIVE;
  return result;
}

inline ResourceGateResult resource_capacity_gate(const ResourcePeakEstimate& estimate,
  const std::uint64_t host_available_bytes, const std::uint64_t device_available_bytes,
  const std::uint64_t disk_available_bytes)
{
  if (estimate.status == ResourcePlanStatus::INVALID) {
    return ResourceGateResult::INVALID;
  }
  if (estimate.status != ResourcePlanStatus::COMPLETE_ESTIMATE) {
    return ResourceGateResult::INCONCLUSIVE;
  }
  return estimate.host_peak_bytes <= host_available_bytes &&
      estimate.device_peak_bytes <= device_available_bytes && estimate.disk_peak_bytes <= disk_available_bytes ?
    ResourceGateResult::PEAK_WITHIN_CAPACITY_ESTIMATE : ResourceGateResult::PEAK_EXCEEDS_CAPACITY;
}

struct Budget
{
  bool started = false;
  double start_seconds = 0.0;
  double last_seconds = 0.0;
  double elapsed_seconds = 0.0;
  unsigned exact_gradients = 0;
  unsigned phase_gradients[8] = {};
  double phase_seconds[8] = {};
  Phase phase = Phase::PREFLIGHT;
};

inline unsigned phase_index(Phase phase);

struct CandidateCostProfile
{
  unsigned samples = 0;
  double maximum_seconds = 0.0;
  double partial_elapsed_lower_bound_seconds = 0.0;
};

struct StageCostProfile
{
  unsigned completed_calls = 0;
  double max_seconds = 0.0;
  double partial_elapsed_lower_bound_seconds = 0.0;
};

struct RemainingWork
{
  unsigned calls[resource_stage_count] = {};
};

enum class CostPlanStatus
{
  ESTIMATED_WITHIN_BUDGET,
  INSUFFICIENT_REMAINING_BUDGET,
  INCONCLUSIVE_NO_STAGE_BASELINE,
  INVALID
};

struct CompletionCostPrediction
{
  CostPlanStatus status = CostPlanStatus::INVALID;
  double stage_seconds[resource_stage_count] = {};
  double stage_lower_bound_seconds[resource_stage_count] = {};
  double fit_seconds = 0.0;
  double validation_seconds = 0.0;
  double available_total_seconds = 0.0;
  double available_fit_seconds = 0.0;
  double safety_factor = 0.0;
  bool estimate_only = true;
};

inline bool record_stage_cost(StageCostProfile& profile, const double seconds)
{
  if (!std::isfinite(seconds) || seconds <= 0.0 || profile.completed_calls == std::numeric_limits<unsigned>::max()) {
    return false;
  }
  profile.max_seconds = std::max(profile.max_seconds, seconds);
  ++profile.completed_calls;
  return true;
}

inline bool record_partial_stage_cost(StageCostProfile& profile, const double seconds)
{
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return false;
  }
  profile.partial_elapsed_lower_bound_seconds =
    std::max(profile.partial_elapsed_lower_bound_seconds, seconds);
  return true;
}

enum class CandidateCostGateResult
{
  ALLOW,
  NO_MEASURED_BASELINE,
  INSUFFICIENT_REMAINING_FIT_TIME,
  INVALID_ESTIMATE
};

struct CandidateCostGate
{
  CandidateCostGateResult result = CandidateCostGateResult::INVALID_ESTIMATE;
  double available_seconds = 0.0;
  double projected_seconds = 0.0;
};

inline bool record_candidate_cost(CandidateCostProfile& profile, const double seconds)
{
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return false;
  }
  profile.maximum_seconds = std::max(profile.maximum_seconds, seconds);
  ++profile.samples;
  return true;
}

inline bool record_partial_candidate_elapsed(CandidateCostProfile& profile, const double seconds)
{
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return false;
  }
  profile.partial_elapsed_lower_bound_seconds = std::max(profile.partial_elapsed_lower_bound_seconds, seconds);
  return true;
}

inline CandidateCostGate measured_candidate_cost_gate(const Budget& budget, const Limits& limits,
  const CandidateCostProfile& profile, const double safety_factor)
{
  CandidateCostGate result;
  if (!limits.valid() || !std::isfinite(safety_factor) || safety_factor < 1.0 || safety_factor > 10.0 ||
      !std::isfinite(budget.elapsed_seconds) || budget.elapsed_seconds < 0.0 ||
      !std::isfinite(budget.phase_seconds[phase_index(budget.phase)]) ||
      budget.phase_seconds[phase_index(budget.phase)] < 0.0) {
    result.result = CandidateCostGateResult::INVALID_ESTIMATE;
    return result;
  }
  if (profile.samples == 0 && profile.partial_elapsed_lower_bound_seconds <= 0.0) {
    result.result = CandidateCostGateResult::NO_MEASURED_BASELINE;
    return result;
  }
  result.projected_seconds = std::max(profile.maximum_seconds, profile.partial_elapsed_lower_bound_seconds) * safety_factor;
  if (!std::isfinite(result.projected_seconds)) {
    result.result = CandidateCostGateResult::INVALID_ESTIMATE;
    return result;
  }
  const double reserved = limits.wall_seconds * limits.validation_reserve_fraction;
  result.available_seconds = std::max(0.0, limits.wall_seconds - budget.elapsed_seconds - reserved);
  if ((budget.phase == Phase::B || budget.phase == Phase::C) && limits.max_fit_family_wall_seconds >= 0.0) {
    const double family_remaining = std::max(0.0, limits.max_fit_family_wall_seconds -
      budget.phase_seconds[phase_index(budget.phase)]);
    result.available_seconds = std::min(result.available_seconds, family_remaining);
  }
  result.result = result.projected_seconds <= result.available_seconds ?
    CandidateCostGateResult::ALLOW : CandidateCostGateResult::INSUFFICIENT_REMAINING_FIT_TIME;
  return result;
}

inline unsigned phase_index(Phase phase)
{
  return static_cast<unsigned>(phase);
}

inline CompletionCostPrediction predict_remaining_cost(const Budget& budget, const Limits& limits,
  const StageCostProfile profiles[resource_stage_count], const RemainingWork& work, const double safety_factor)
{
  CompletionCostPrediction result;
  result.safety_factor = safety_factor;
  if (!limits.valid() || !budget.started || profiles == nullptr || !std::isfinite(safety_factor) ||
      safety_factor < 1.0 || safety_factor > 10.0 || !std::isfinite(budget.elapsed_seconds) ||
      budget.elapsed_seconds < 0.0 || phase_index(budget.phase) >= 8 ||
      !std::isfinite(budget.phase_seconds[phase_index(budget.phase)]) ||
      budget.phase_seconds[phase_index(budget.phase)] < 0.0) {
    result.status = CostPlanStatus::INVALID;
    return result;
  }

  bool incomplete_baseline = false;
  double fit_estimate = 0.0, validation_estimate = 0.0;
  for (unsigned i = 0; i < resource_stage_count; ++i) {
    const unsigned calls = work.calls[i];
    if (calls == 0) {
      continue;
    }
    const StageCostProfile& profile = profiles[i];
    if (!std::isfinite(profile.max_seconds) || profile.max_seconds < 0.0 ||
        !std::isfinite(profile.partial_elapsed_lower_bound_seconds) ||
        profile.partial_elapsed_lower_bound_seconds < 0.0 ||
        (profile.completed_calls == 0 && profile.max_seconds != 0.0) ||
        (profile.completed_calls > 0 && profile.max_seconds <= 0.0)) {
      result.status = CostPlanStatus::INVALID;
      return result;
    }
    const double partial_lower = profile.partial_elapsed_lower_bound_seconds * calls;
    if (!std::isfinite(partial_lower)) {
      result.status = CostPlanStatus::INVALID;
      return result;
    }
    result.stage_lower_bound_seconds[i] = partial_lower;
    if (profile.completed_calls == 0) {
      incomplete_baseline = true;
      continue;
    }
    const double projection = profile.max_seconds * calls * safety_factor;
    if (!std::isfinite(projection)) {
      result.status = CostPlanStatus::INVALID;
      return result;
    }
    result.stage_seconds[i] = projection;
    if (profile.partial_elapsed_lower_bound_seconds > profile.max_seconds) {
      incomplete_baseline = true;
    }
    if (i <= resource_stage_index(ResourceStage::CURVATURE)) {
      fit_estimate += projection;
    } else {
      validation_estimate += projection;
    }
  }
  if (!std::isfinite(fit_estimate) || !std::isfinite(validation_estimate)) {
    result.status = CostPlanStatus::INVALID;
    return result;
  }
  result.fit_seconds = fit_estimate;
  result.validation_seconds = validation_estimate;
  result.available_total_seconds = std::max(0.0, limits.wall_seconds - budget.elapsed_seconds);
  result.available_fit_seconds = std::max(0.0, result.available_total_seconds -
    limits.wall_seconds * limits.validation_reserve_fraction);
  if ((budget.phase == Phase::B || budget.phase == Phase::C) && limits.max_fit_family_wall_seconds >= 0.0) {
    const double family_available = std::max(0.0, limits.max_fit_family_wall_seconds -
      budget.phase_seconds[phase_index(budget.phase)]);
    result.available_fit_seconds = std::min(result.available_fit_seconds, family_available);
  }
  double fit_need = 0.0, validation_need = 0.0;
  for (unsigned i = 0; i < resource_stage_count; ++i) {
    const double need = std::max(result.stage_seconds[i], result.stage_lower_bound_seconds[i]);
    if (i <= resource_stage_index(ResourceStage::CURVATURE)) {
      fit_need += need;
    } else {
      validation_need += need;
    }
  }
  if (!std::isfinite(fit_need) || !std::isfinite(validation_need)) {
    result.status = CostPlanStatus::INVALID;
  } else if (fit_need > result.available_fit_seconds ||
      fit_need + validation_need > result.available_total_seconds) {
    result.status = CostPlanStatus::INSUFFICIENT_REMAINING_BUDGET;
  } else if (incomplete_baseline) {
    result.status = CostPlanStatus::INCONCLUSIVE_NO_STAGE_BASELINE;
  } else {
    result.status = CostPlanStatus::ESTIMATED_WITHIN_BUDGET;
  }
  return result;
}

inline bool validation_phase(Phase phase)
{
  return phase == Phase::FREEZE || phase == Phase::BLIND || phase == Phase::PREPARE;
}

inline bool should_run_structured_family(bool has_independent_additions, bool m0_common_probe_admissible)
{
  return has_independent_additions && !m0_common_probe_admissible;
}

inline int common_soft_probe_capacity(const int internal_dimension, const int validation_frames,
  const int maximum_soft_modes)
{
  return std::min({std::max(0, internal_dimension), std::max(0, validation_frames - 1),
    std::max(0, maximum_soft_modes)});
}

inline BudgetResult advance_clock(Budget& budget, double now_seconds)
{
  if (!std::isfinite(now_seconds)) {
    return BudgetResult::INVALID_CLOCK;
  }
  if (now_seconds < budget.last_seconds) {
    return BudgetResult::CLOCK_BACKWARDS;
  }
  const double delta = now_seconds - budget.last_seconds;
  const unsigned index = phase_index(budget.phase);
  if (index >= 8) {
    return BudgetResult::INVALID_PHASE;
  }
  if (!std::isfinite(delta) || !std::isfinite(budget.elapsed_seconds + delta) ||
    !std::isfinite(budget.phase_seconds[index] + delta)) {
    return BudgetResult::INVALID_CLOCK;
  }
  budget.elapsed_seconds += delta;
  budget.phase_seconds[index] += delta;
  budget.last_seconds = now_seconds;
  return BudgetResult::OK;
}

inline BudgetResult phase_gate(const Budget& budget, const Limits& limits)
{
  if (budget.elapsed_seconds >= limits.wall_seconds) {
    return BudgetResult::WALL_LIMIT;
  }
  if (budget.phase == Phase::A0 || budget.phase == Phase::A1) {
    const double a_seconds = budget.phase_seconds[phase_index(Phase::A0)] +
      budget.phase_seconds[phase_index(Phase::A1)];
    if (a_seconds >= limits.wall_seconds * limits.a_wall_fraction) {
      return BudgetResult::A_WALL_LIMIT;
    }
  }
  if ((budget.phase == Phase::B || budget.phase == Phase::C) &&
      limits.max_fit_family_wall_seconds >= 0.0 &&
      budget.phase_seconds[phase_index(budget.phase)] >= limits.max_fit_family_wall_seconds) {
    return BudgetResult::FAMILY_WALL_LIMIT;
  }
  const double reserved_seconds = limits.wall_seconds * limits.validation_reserve_fraction;
  if (!validation_phase(budget.phase) && limits.wall_seconds - budget.elapsed_seconds <= reserved_seconds) {
    return BudgetResult::VALIDATION_RESERVED;
  }
  return BudgetResult::OK;
}

inline BudgetResult begin(Budget& budget, const Limits& limits, double now_seconds)
{
  if (!limits.valid()) {
    return BudgetResult::INVALID_LIMITS;
  }
  if (!std::isfinite(now_seconds)) {
    return BudgetResult::INVALID_CLOCK;
  }
  if (budget.started) {
    return BudgetResult::ALREADY_STARTED;
  }
  budget.started = true;
  budget.phase = Phase::PREFLIGHT;
  budget.start_seconds = now_seconds;
  budget.last_seconds = now_seconds;
  return BudgetResult::OK;
}

inline BudgetResult observe_time(Budget& budget, const Limits& limits, double now_seconds)
{
  if (!limits.valid()) {
    return BudgetResult::INVALID_LIMITS;
  }
  if (!budget.started) {
    return BudgetResult::NOT_STARTED;
  }
  const BudgetResult clock = advance_clock(budget, now_seconds);
  return clock == BudgetResult::OK ? phase_gate(budget, limits) : clock;
}

inline BudgetResult enter_phase(Budget& budget, const Limits& limits, Phase phase, double now_seconds)
{
  if (!limits.valid()) {
    return BudgetResult::INVALID_LIMITS;
  }
  if (!budget.started) {
    return BudgetResult::NOT_STARTED;
  }
  if (phase_index(phase) >= 8) {
    return BudgetResult::INVALID_PHASE;
  }
  const BudgetResult clock = advance_clock(budget, now_seconds);
  if (clock != BudgetResult::OK) {
    return clock;
  }
  budget.phase = phase;
  return phase_gate(budget, limits);
}

inline BudgetResult reserve_exact_gradient(Budget& budget, const Limits& limits, double now_seconds)
{
  if (!limits.valid()) {
    return BudgetResult::INVALID_LIMITS;
  }
  if (!budget.started) {
    return BudgetResult::NOT_STARTED;
  }
  const BudgetResult gate = observe_time(budget, limits, now_seconds);
  if (gate != BudgetResult::OK) {
    return gate;
  }
  if (budget.exact_gradients >= static_cast<unsigned>(limits.max_exact_gradients)) {
    return BudgetResult::GRADIENT_LIMIT;
  }
  const unsigned index = phase_index(budget.phase);
  if (index >= 8) {
    return BudgetResult::INVALID_PHASE;
  }
  if (budget.phase == Phase::A0 || budget.phase == Phase::A1) {
    if (budget.phase_gradients[index] >= static_cast<unsigned>(limits.max_a_branch_gradients)) {
      return BudgetResult::PHASE_GRADIENT_LIMIT;
    }
  } else if ((budget.phase == Phase::B || budget.phase == Phase::C) &&
      budget.phase_gradients[index] >= static_cast<unsigned>(limits.max_fit_family_gradients)) {
    return BudgetResult::FAMILY_GRADIENT_LIMIT;
  }
  ++budget.exact_gradients;
  ++budget.phase_gradients[index];
  return BudgetResult::OK;
}

inline int equal_remaining_family_gradient_cap(const Budget& budget, const Limits& limits)
{
  const unsigned total = static_cast<unsigned>(limits.max_exact_gradients);
  const unsigned remaining = budget.exact_gradients < total ? total - budget.exact_gradients : 0;
  return static_cast<int>(remaining / 2);
}

inline double equal_remaining_family_wall_cap(const Budget& budget, const Limits& limits)
{
  const double reserved = limits.wall_seconds * limits.validation_reserve_fraction;
  const double remaining = std::max(0.0, limits.wall_seconds - budget.elapsed_seconds - reserved);
  return 0.5 * remaining;
}

enum class Family
{
  M0,
  M1
};

struct Candidate
{
  Family family = Family::M0;
  unsigned index = 0;
  bool converged = false;
  double development_objective = 0.0;
  double development_objective_uncertainty = 0.0;
  std::vector<double> development_block_scores_b;
  std::vector<double> development_block_scores_2b;
};

struct FrozenSelection
{
  bool frozen = false;
  bool blind_opened = false;
  Family family = Family::M0;
  unsigned candidate_index = 0;
};

enum class SelectionResult
{
  SELECTED,
  NO_ELIGIBLE,
  TOO_MANY_CANDIDATES,
  INVALID_INPUT,
  INVALID_PAIRED_BLOCKS,
  DATA_INCONCLUSIVE,
  NO_PAIRED_WINNER,
  ALREADY_FROZEN
};

inline void observe_selection_budget(const std::function<void()>& observer,const std::size_t index)
{
  if(observer&&(index&4095u)==0)observer();
}

inline bool eligible(const Candidate& candidate,const std::function<void()>& observer={})
{
  if(!(candidate.converged && std::isfinite(candidate.development_objective) &&
    std::isfinite(candidate.development_objective_uncertainty) && candidate.development_objective_uncertainty >= 0.0 &&
    candidate.development_block_scores_b.size() >= 4 && candidate.development_block_scores_2b.size() >= 4))return false;
  for(std::size_t index=0;index<candidate.development_block_scores_b.size();++index){
    observe_selection_budget(observer,index);
    if(!std::isfinite(candidate.development_block_scores_b[index]))return false;}
  for(std::size_t index=0;index<candidate.development_block_scores_2b.size();++index){
    observe_selection_budget(observer,index);
    if(!std::isfinite(candidate.development_block_scores_2b[index]))return false;}
  return true;
}

inline bool paired_score_difference_at_scale(const Candidate& left, const Candidate& right,
  const bool adjacent_scale,double& mean,double& uncertainty,const std::function<void()>& observer={})
{
  const std::vector<double>& left_scores=adjacent_scale?left.development_block_scores_2b:left.development_block_scores_b;
  const std::vector<double>& right_scores=adjacent_scale?right.development_block_scores_2b:right.development_block_scores_b;
  if (left_scores.size() < 4 || left_scores.size()!=right_scores.size()) {
    return false;
  }
  std::vector<double> differences(left_scores.size());
  for (std::size_t i = 0; i < differences.size(); ++i) {
    observe_selection_budget(observer,i);
    if (!std::isfinite(left_scores[i]) || !std::isfinite(right_scores[i])) {
      return false;
    }
    differences[i] = left_scores[i] - right_scores[i];
  }
  mean = 0.0;
  for (std::size_t i=0;i<differences.size();++i) {
    observe_selection_budget(observer,i);mean += differences[i];
  }
  mean /= static_cast<double>(differences.size());
  double sum2 = 0.0;
  for (std::size_t i=0;i<differences.size();++i) {
    observe_selection_budget(observer,i);const double difference=differences[i];
    const double centered = difference - mean;
    sum2 += centered * centered;
  }
  uncertainty = 2.0 * std::sqrt(sum2 /
    (static_cast<double>(differences.size() - 1) * static_cast<double>(differences.size())));
  return std::isfinite(mean) && std::isfinite(uncertainty);
}

enum class PairedCandidateResult { LEFT_WINS, RIGHT_WINS, UNRESOLVED, DATA_INCONCLUSIVE, INVALID };

inline PairedCandidateResult compare_candidates_at_both_scales(
  const Candidate& left,const Candidate& right,double& mean_b,double& uncertainty_b,
  double& mean_2b,double& uncertainty_2b,const std::function<void()>& observer={})
{
  if(!paired_score_difference_at_scale(left,right,false,mean_b,uncertainty_b,observer)||
     !paired_score_difference_at_scale(left,right,true,mean_2b,uncertainty_2b,observer))
    return PairedCandidateResult::INVALID;
  const int winner_b=mean_b< -uncertainty_b?1:mean_b>uncertainty_b?-1:0;
  const int winner_2b=mean_2b< -uncertainty_2b?1:mean_2b>uncertainty_2b?-1:0;
  if(winner_b==0&&winner_2b==0)return PairedCandidateResult::UNRESOLVED;
  if(winner_b==winner_2b)return winner_b>0?PairedCandidateResult::LEFT_WINS:PairedCandidateResult::RIGHT_WINS;
  return PairedCandidateResult::DATA_INCONCLUSIVE;
}

inline SelectionResult choose_development_candidate(
  const Candidate* candidates, unsigned count, FrozenSelection& selection,
  const std::function<void()>& observer={})
{
  if (selection.frozen) {
    return SelectionResult::ALREADY_FROZEN;
  }
  if (count > 0 && candidates == nullptr) {
    return SelectionResult::INVALID_INPUT;
  }
  unsigned m0_count = 0;
  unsigned m1_count = 0;
  std::vector<unsigned> eligible_indices;
  for (unsigned i = 0; i < count; ++i) {
    const Candidate& candidate = candidates[i];
    if (candidate.family == Family::M0) {
      ++m0_count;
    } else if (candidate.family == Family::M1) {
      ++m1_count;
    } else {
      return SelectionResult::INVALID_INPUT;
    }
    if (m0_count > 3 || m1_count > 3) {
      return SelectionResult::TOO_MANY_CANDIDATES;
    }
    if (!eligible(candidate,observer)) {
      continue;
    }
    eligible_indices.push_back(i);
  }
  if (eligible_indices.empty()) {
    return SelectionResult::NO_ELIGIBLE;
  }
  const Candidate& first_candidate=candidates[eligible_indices.front()];
  for (const unsigned i : eligible_indices) {
    if (candidates[i].development_block_scores_b.size()!=first_candidate.development_block_scores_b.size()||
        candidates[i].development_block_scores_2b.size()!=first_candidate.development_block_scores_2b.size()) {
      return SelectionResult::INVALID_PAIRED_BLOCKS;
    }
  }
  std::vector<unsigned> unbeaten;
  for (const unsigned candidate_index : eligible_indices) {
    bool beaten = false;
    for (const unsigned other_index : eligible_indices) {
      if (candidate_index == other_index) {
        continue;
      }
      double mean_b=0.0,uncertainty_b=0.0,mean_2b=0.0,uncertainty_2b=0.0;
      const PairedCandidateResult pair=compare_candidates_at_both_scales(candidates[candidate_index],
        candidates[other_index],mean_b,uncertainty_b,mean_2b,uncertainty_2b,observer);
      if(pair==PairedCandidateResult::INVALID){
        return SelectionResult::INVALID_PAIRED_BLOCKS;
      }
      if(pair==PairedCandidateResult::DATA_INCONCLUSIVE)return SelectionResult::DATA_INCONCLUSIVE;
      if (pair==PairedCandidateResult::RIGHT_WINS) {
        beaten = true;
        break;
      }
    }
    if (!beaten) {
      unbeaten.push_back(candidate_index);
    }
  }
  if (unbeaten.empty()) {
    return SelectionResult::NO_PAIRED_WINNER;
  }
  unsigned best = unbeaten.front();
  for (const unsigned i : unbeaten) {
    const Candidate& candidate = candidates[i];
    const Candidate& best_candidate = candidates[best];
    const bool prefer_m0 = candidate.family == Family::M0 && best_candidate.family == Family::M1;
    const bool same_family_better_score = candidate.family == best_candidate.family &&
      (candidate.development_objective < best_candidate.development_objective ||
       (candidate.development_objective == best_candidate.development_objective && candidate.index < best_candidate.index));
    if (prefer_m0 || same_family_better_score) {
      best = i;
    }
  }
  selection.frozen = true;
  selection.family = candidates[best].family;
  selection.candidate_index = candidates[best].index;
  return SelectionResult::SELECTED;
}

inline bool open_blind_once(FrozenSelection& selection)
{
  if (!selection.frozen || selection.blind_opened) {
    return false;
  }
  selection.blind_opened = true;
  return true;
}

inline bool regularization_sensitivity_allows_blind(const bool sensitive, const bool inconclusive)
{
  return !sensitive && !inconclusive;
}

enum class BlindStatus
{
  DEVELOPMENT_CANDIDATE,
  DATA_INCONCLUSIVE,
  BLIND_FAILED,
  BLIND_PASSED
};

inline BlindStatus classify_blind_result(
  const FrozenSelection& selection, bool independent_data, bool passed)
{
  if (!selection.frozen || !selection.blind_opened) {
    return BlindStatus::DEVELOPMENT_CANDIDATE;
  }
  if (!independent_data) {
    return BlindStatus::DATA_INCONCLUSIVE;
  }
  return passed ? BlindStatus::BLIND_PASSED : BlindStatus::BLIND_FAILED;
}

inline bool can_compare_a_tau1(double tau_a0, double tau_a1)
{
  return std::isfinite(tau_a0) && std::isfinite(tau_a1) && tau_a0 == 1.0 && tau_a1 == 1.0;
}
} // namespace rpmd_ja_feasibility
