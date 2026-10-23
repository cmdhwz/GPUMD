#pragma once

#include "rpmd_ja_sha256.h"
#include "rpmd_ja_utf8_file_io.h"
#include "rpmd_ja_feasibility_policy.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rpmd_ja_covariance_checkpoint
{
constexpr std::uint64_t maximum_checkpoint_bytes=16ULL*1024*1024;
constexpr std::uint64_t maximum_controller_journal_bytes=512ULL*1024*1024;

struct AcceptedPoint
{
  std::string job_id,identity_sha256,branch,family,selection_state="NOT_FROZEN";
  unsigned candidate_index=0;
  unsigned active_phase=0;
  double lambda=0.0,epsilon=0.0,rho=0.0,ridge_lambda=0.0,tau=0.0,trust_radius=0.0,start_graph_delta=0.0;
  double elapsed_seconds=0.0,operation_start_epoch_seconds=0.0;
  bool operation_in_progress=false;
  std::uint64_t max_objective_evaluations=0,objective_evaluations=0,gradient_evaluations=0,total_budget_gradients=0;
  std::uint64_t initialization_evaluations=0,iterations=0;
  std::array<unsigned,8> phase_gradients{};
  std::array<double,8> phase_seconds{};
  std::array<unsigned,8> phase_cursor{};
  unsigned curvature_probe_count=0;
  bool curvature_probe_count_verified=false;
  bool performance_window_initialized=false;
  unsigned performance_window_curvature_builds=0;
  double performance_window_tau=0.0,performance_window_start_objective=0.0;
  double performance_window_curvature_seconds=0.0,performance_window_reference_gradient_seconds=0.0;
  double objective_evaluation_seconds=0.0,full_gradient_seconds=0.0,last_full_gradient_seconds=0.0;
  double factorization_seconds=0.0,analytic_curvature_seconds=0.0;
  unsigned bfgs_recent_secant_count=0;
  std::vector<double> theta,theta_start,coordinate_scale,bfgs_inverse,stagnation_best_theta;
  std::uint64_t stagnation_gradient_attempts=0,stagnation_low_progress_windows=0,
    stagnation_progress_attempts_in_window=0,same_point_curvature_rejections=0;
  double stagnation_best_kkt=std::numeric_limits<double>::quiet_NaN();
  double stagnation_window_start_kkt=std::numeric_limits<double>::quiet_NaN();
  double stagnation_last_window_relative_improvement=std::numeric_limits<double>::quiet_NaN();
  std::vector<double> bfgs_recent_secants;
  double bfgs_seed_gamma=1.0,bfgs_last_valid_gamma=std::numeric_limits<double>::quiet_NaN();
  bool bfgs_stage_seeded=false;
  std::string bfgs_seed_source="UNIT_IDENTITY";
};

struct ControllerCandidate
{
  std::string branch,fit_state_sha256,score_generation="NONE";
  unsigned index=0;
  double lambda=0.0;
  bool fit_complete=false,score_complete=false,response_replay_required=false,converged=false,development_pass=false,b_gate_pass=false;
  double objective=0.0,uncertainty=0.0;
  std::vector<double> theta,block_scores_b,block_scores_2b;
};

inline bool controller_candidate_pass_for_generation(const ControllerCandidate& candidate,const std::string& generation)
{
  return candidate.score_complete&&candidate.response_replay_required&&candidate.score_generation==generation&&
    candidate.development_pass;
}

struct ControllerJournal
{
  std::string job_id,identity_sha256,branch="A0",family="M0",stage="A0";
  std::string selection_state="NOT_FROZEN";
  std::string frozen_candidate_branch,frozen_candidate_fit_state_sha256,frozen_candidate_identity_sha256;
  std::string development_response_sha256,blind_response_sha256,publication_stage_output_sha256,publication_stage_sidecar_sha256;
  std::string published_output_sha256,published_sidecar_sha256,published_fit_sha256,published_trace_sha256;
  unsigned frozen_candidate_index=0;
  double frozen_candidate_lambda=0.0;
  std::string common_probe_stage="NONE";
  unsigned next_candidate_index=0,active_phase=0,max_fit_family_gradients=0;
  bool active_candidate_started=false;
  bool operation_in_progress=false;
  double max_fit_family_wall_seconds=-1.0,elapsed_seconds=0.0;
  double operation_start_epoch_seconds=0.0;
  double active_candidate_elapsed_seconds=0.0;
  std::uint64_t exact_gradients=0;
  std::uint64_t cg_witness_attempts=0;
  std::array<unsigned,8> phase_gradients{};
  std::array<double,8> phase_seconds{};
  bool training_statistics_frozen=false;
  std::string training_statistics_policy_version="NONE";
  std::uint64_t training_frame_count=0,development_frame_count=0,training_covered_frame_count=0,
    training_discarded_tail_frames=0;
  unsigned training_block_length=0,training_adjacent_block_length=0;
  std::vector<std::string> training_channel_names;
  std::array<unsigned,rpmd_ja_feasibility::resource_stage_count> stage_completed_calls{};
  std::array<double,rpmd_ja_feasibility::resource_stage_count> stage_max_seconds{},stage_partial_seconds{};
  unsigned candidate_cost_samples=0;
  double candidate_cost_max_seconds=0.0,candidate_cost_partial_seconds=0.0;
  bool common_probe_set_frozen=false;
  std::vector<std::string> common_probe_sources,common_probe_origins;
  std::vector<std::vector<double>> common_probe_vectors;
  std::vector<ControllerCandidate> candidates;
};

template<typename FinalizeM1Generation>
inline void commit_final_development_generation(ControllerJournal& journal,const bool has_m1_candidates,
  FinalizeM1Generation finalize_m1_generation)
{
  if(journal.active_candidate_started||journal.active_candidate_elapsed_seconds!=0.0||!journal.common_probe_set_frozen||
     !(journal.common_probe_stage=="B_GATE"||journal.common_probe_stage=="FINAL_M0_M1"))
    throw std::invalid_argument("final development generation requires completed candidates and the committed common probe set");
  journal.stage="FREEZE";journal.branch=has_m1_candidates?"C2":"B2";
  journal.family=has_m1_candidates?"M1":"M0";journal.next_candidate_index=2;
  if(has_m1_candidates){
    finalize_m1_generation();
    if(journal.common_probe_stage!="FINAL_M0_M1")
      throw std::invalid_argument("M1 final development generation did not commit its common probe set");
  }else{
    if(journal.common_probe_stage=="B_GATE"){
      for(ControllerCandidate& candidate:journal.candidates)
        if(candidate.branch.size()==2&&candidate.branch[0]=='B'&&candidate.score_complete&&candidate.score_generation=="B_GATE")
          candidate.score_generation="FINAL_M0_M1";
      journal.common_probe_stage="FINAL_M0_M1";
    }else if(std::any_of(journal.candidates.begin(),journal.candidates.end(),[](const ControllerCandidate& candidate){
        return candidate.branch.size()==2&&candidate.branch[0]=='B'&&candidate.score_complete&&
          candidate.score_generation!="FINAL_M0_M1";}))
      throw std::invalid_argument("final M0 probe generation differs from its committed candidate scores");
  }
}

inline bool controller_has_committed_b_gate_for_resume(const ControllerJournal& journal,const bool controller_resume)
{
  return controller_resume&&(journal.stage=="C"||journal.stage=="FREEZE");
}

inline void require_finite(const std::vector<double>& values,const char* name)
{
  for(const double value:values)if(!std::isfinite(value))
    throw std::runtime_error(std::string("covariance checkpoint has a non-finite ")+name);
}

inline std::string sha256(const std::string& bytes)
{
  rpmd_ja_sha256::Digest digest;
  digest.update(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size());
  return digest.finish_hex();
}

inline bool is_sha256(const std::string& value)
{
  if(value.size()!=64)return false;
  for(const char c:value)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
  return true;
}

inline bool controller_has_frozen_selection(const ControllerJournal& journal)
{
  return journal.selection_state=="FROZEN"||journal.selection_state=="BLIND_OPENED"||
    journal.selection_state=="BLIND_PASSED"||journal.selection_state=="PUBLISHING"||
    journal.selection_state=="PUBLISHED";
}

inline bool valid_sha256_or_empty(const std::string& value)
{return value.empty()||is_sha256(value);}

inline std::string frozen_candidate_identity_sha256(const ControllerJournal& state)
{
  const auto candidate=std::find_if(state.candidates.begin(),state.candidates.end(),[&](const ControllerCandidate& value){
    return value.branch==state.frozen_candidate_branch;});
  if(candidate==state.candidates.end()||!candidate->fit_complete||candidate->index!=state.frozen_candidate_index||
     candidate->lambda!=state.frozen_candidate_lambda||!is_sha256(candidate->fit_state_sha256))return {};
  std::ostringstream payload;payload<<std::setprecision(17)<<state.identity_sha256<<'\n'<<candidate->branch<<'\n'
    <<candidate->index<<'\n'<<candidate->lambda<<'\n'<<candidate->fit_state_sha256<<'\n';
  for(const double value:candidate->theta)payload<<value<<' ';
  payload<<'\n'<<state.common_probe_stage<<'\n';
  for(std::size_t index=0;index<state.common_probe_sources.size();++index){
    payload<<std::quoted(state.common_probe_sources[index])<<' '<<std::quoted(state.common_probe_origins[index])<<' ';
    for(const double value:state.common_probe_vectors[index])payload<<value<<' ';
    payload<<'\n';}
  return sha256(payload.str());
}

inline void validate_state(const AcceptedPoint& state)
{
  std::uint64_t phase_gradient_sum=0;double phase_second_sum=0.0;
  for(std::size_t phase=0;phase<state.phase_gradients.size();++phase){
    phase_gradient_sum+=state.phase_gradients[phase];
    if(!std::isfinite(state.phase_seconds[phase])||state.phase_seconds[phase]<0.0)
      throw std::invalid_argument("invalid covariance checkpoint phase elapsed time");
    phase_second_sum+=state.phase_seconds[phase];}
  const bool valid_selection=state.selection_state=="NOT_FROZEN"||state.selection_state=="FROZEN"||
    state.selection_state=="BLIND_OPENED"||state.selection_state=="BLIND_PASSED"||state.selection_state=="PUBLISHING"||
    state.selection_state=="PUBLISHED"||state.selection_state=="FAILED_CLOSED";
  if(state.job_id.empty()||!is_sha256(state.identity_sha256)||state.branch.empty()||state.family.empty()||!valid_selection||
     state.active_phase>=state.phase_gradients.size()||!std::isfinite(state.lambda)||!std::isfinite(state.epsilon)||!(state.epsilon>0.0)||
     !std::isfinite(state.rho)||state.rho<0.0||!std::isfinite(state.ridge_lambda)||state.ridge_lambda<0.0||
     !std::isfinite(state.tau)||!(state.tau>0.0)||!std::isfinite(state.trust_radius)||!(state.trust_radius>0.0)||
     !std::isfinite(state.start_graph_delta)||state.start_graph_delta<0.0||
     !std::isfinite(state.elapsed_seconds)||state.elapsed_seconds<0.0||!std::isfinite(phase_second_sum)||
     phase_second_sum>state.elapsed_seconds+1.0e-9*std::max(1.0,state.elapsed_seconds)||
     !std::isfinite(state.operation_start_epoch_seconds)||state.operation_start_epoch_seconds<0.0||state.theta.empty()||
     state.performance_window_curvature_builds>rpmd_ja_feasibility::candidate_performance_curvature_window||
     !std::isfinite(state.performance_window_tau)||state.performance_window_tau<0.0||
     !std::isfinite(state.performance_window_start_objective)||
     !std::isfinite(state.performance_window_curvature_seconds)||state.performance_window_curvature_seconds<0.0||
     !std::isfinite(state.performance_window_reference_gradient_seconds)||state.performance_window_reference_gradient_seconds<0.0||
     !std::isfinite(state.objective_evaluation_seconds)||state.objective_evaluation_seconds<0.0||
     !std::isfinite(state.full_gradient_seconds)||state.full_gradient_seconds<0.0||
     !std::isfinite(state.last_full_gradient_seconds)||state.last_full_gradient_seconds<0.0||
     !std::isfinite(state.factorization_seconds)||state.factorization_seconds<0.0||
     !std::isfinite(state.analytic_curvature_seconds)||state.analytic_curvature_seconds<0.0||
     (state.performance_window_initialized&&!(state.performance_window_tau>0.0))||
     (!state.performance_window_initialized&&(state.performance_window_curvature_builds!=0||
       state.performance_window_tau!=0.0||state.performance_window_curvature_seconds!=0.0||
       state.performance_window_reference_gradient_seconds!=0.0))||
     state.theta.size()>10000||state.theta.size()!=state.coordinate_scale.size()||
     state.theta_start.size()!=state.theta.size()||
     (!state.stagnation_best_theta.empty()&&state.stagnation_best_theta.size()!=state.theta.size())||
     state.bfgs_inverse.size()!=state.theta.size()*state.theta.size()||state.bfgs_recent_secant_count>8||
     state.bfgs_recent_secants.size()!=2ULL*state.bfgs_recent_secant_count*state.theta.size()||
     state.max_objective_evaluations==0||state.objective_evaluations>state.max_objective_evaluations||
     phase_gradient_sum!=state.total_budget_gradients||!std::isfinite(state.bfgs_seed_gamma)||!(state.bfgs_seed_gamma>0.0)||
     (!std::isnan(state.bfgs_last_valid_gamma)&&(!std::isfinite(state.bfgs_last_valid_gamma)||state.bfgs_last_valid_gamma<=0.0)))
    throw std::invalid_argument("invalid accepted-point covariance checkpoint state");
  require_finite(state.theta,"theta");require_finite(state.coordinate_scale,"coordinate scale");
  require_finite(state.theta_start,"theta start");require_finite(state.stagnation_best_theta,"stagnation best theta");
  require_finite(state.bfgs_inverse,"BFGS matrix");require_finite(state.bfgs_recent_secants,"BFGS secants");
  for(const double value:{state.stagnation_best_kkt,state.stagnation_window_start_kkt,
      state.stagnation_last_window_relative_improvement})
    if(!std::isnan(value)&&!std::isfinite(value))throw std::invalid_argument("invalid covariance checkpoint stagnation state");
  for(const double scale:state.coordinate_scale)if(!(scale>0.0))
    throw std::invalid_argument("invalid covariance checkpoint coordinate scale");
  if(state.bfgs_seed_source.empty())throw std::invalid_argument("invalid covariance checkpoint BFGS seed source");
}

inline void validate_controller_journal(const ControllerJournal& state)
{
  const rpmd_ja_feasibility::Limits frozen_limits{};
  const double max_family_wall_seconds=0.5*frozen_limits.wall_seconds*
    (1.0-frozen_limits.validation_reserve_fraction);
  const bool valid_stage=state.stage=="A0"||state.stage=="A1"||state.stage=="B"||state.stage=="C"||state.stage=="FREEZE";
  const bool valid_branch=state.branch=="A0"||state.branch=="A1"||state.branch=="B0"||state.branch=="B1"||
    state.branch=="B2"||state.branch=="C0"||state.branch=="C1"||state.branch=="C2";
  const bool valid_family=state.family=="M0"||state.family=="M1";
  const bool valid_selection=state.selection_state=="NOT_FROZEN"||state.selection_state=="FROZEN"||
    state.selection_state=="BLIND_OPENED"||state.selection_state=="BLIND_PASSED"||state.selection_state=="PUBLISHING"||state.selection_state=="PUBLISHED"||
    state.selection_state=="FAILED_CLOSED";
  const unsigned branch_index=state.branch=="A0"||state.branch=="A1"?0u:
    state.branch.size()==2?static_cast<unsigned>(state.branch[1]-'0'):std::numeric_limits<unsigned>::max();
  const bool cursor_matches=state.next_candidate_index==branch_index&&
    ((state.stage=="A0"&&state.branch=="A0"&&state.family=="M0")||
     (state.stage=="A1"&&state.branch=="A1"&&state.family=="M0")||
     (state.stage=="B"&&state.branch.size()==2&&state.branch[0]=='B'&&state.family=="M0")||
     (state.stage=="C"&&state.branch.size()==2&&state.branch[0]=='C'&&state.family=="M1")||
     (state.stage=="FREEZE"&&((state.branch=="B2"&&state.family=="M0")||
       (state.branch=="C2"&&state.family=="M1"))));
  if(state.job_id.empty()||!is_sha256(state.identity_sha256)||!valid_branch||!valid_family||!valid_stage||
     !valid_selection||!cursor_matches||state.active_phase>=8||
     !std::isfinite(state.max_fit_family_wall_seconds)||state.max_fit_family_wall_seconds< -1.0||
     !std::isfinite(state.elapsed_seconds)||state.elapsed_seconds<0.0||
     !std::isfinite(state.active_candidate_elapsed_seconds)||state.active_candidate_elapsed_seconds<0.0||
     !std::isfinite(state.operation_start_epoch_seconds)||state.operation_start_epoch_seconds<0.0||
     (state.operation_in_progress&&!(state.operation_start_epoch_seconds>0.0))||
     !std::isfinite(state.candidate_cost_max_seconds)||state.candidate_cost_max_seconds<0.0||
     !std::isfinite(state.candidate_cost_partial_seconds)||state.candidate_cost_partial_seconds<0.0||
     !valid_sha256_or_empty(state.frozen_candidate_fit_state_sha256)||
     !valid_sha256_or_empty(state.frozen_candidate_identity_sha256)||
     !valid_sha256_or_empty(state.development_response_sha256)||
     !valid_sha256_or_empty(state.blind_response_sha256)||
     !valid_sha256_or_empty(state.publication_stage_output_sha256)||
     !valid_sha256_or_empty(state.publication_stage_sidecar_sha256)||
     !valid_sha256_or_empty(state.published_output_sha256)||!valid_sha256_or_empty(state.published_sidecar_sha256)||
     !valid_sha256_or_empty(state.published_fit_sha256)||!valid_sha256_or_empty(state.published_trace_sha256)||
     !std::isfinite(state.frozen_candidate_lambda)||state.frozen_candidate_lambda<0.0||
     state.elapsed_seconds>frozen_limits.wall_seconds||state.exact_gradients>
       static_cast<std::uint64_t>(frozen_limits.max_exact_gradients)||
     state.max_fit_family_gradients>static_cast<unsigned>(frozen_limits.max_fit_family_gradients)||
     state.max_fit_family_wall_seconds>max_family_wall_seconds||
     state.phase_gradients[1]>static_cast<unsigned>(frozen_limits.max_a_branch_gradients)||
     state.phase_gradients[2]>static_cast<unsigned>(frozen_limits.max_a_branch_gradients)||
     state.phase_gradients[3]>state.max_fit_family_gradients||
     state.phase_gradients[4]>state.max_fit_family_gradients||
     !(state.common_probe_stage=="NONE"||state.common_probe_stage=="B_GATE"||state.common_probe_stage=="FINAL_M0_M1")||
     state.common_probe_sources.size()!=state.common_probe_vectors.size()||
     state.common_probe_sources.size()!=state.common_probe_origins.size()||
     (!state.common_probe_set_frozen&&(!state.common_probe_sources.empty()||!state.common_probe_vectors.empty()))||
     (!state.training_statistics_frozen&&(!state.training_channel_names.empty()||state.training_block_length!=0||
       state.training_adjacent_block_length!=0||state.training_frame_count!=0||state.development_frame_count!=0||
       state.training_covered_frame_count!=0||state.training_discarded_tail_frames!=0||
       state.training_statistics_policy_version!="NONE"))||
     (state.training_statistics_frozen&&(state.training_statistics_policy_version!="RPJASTAT2"||
       state.training_channel_names.empty()||state.training_block_length==0||
       state.training_block_length>std::numeric_limits<unsigned>::max()/2||
       state.training_adjacent_block_length!=2*state.training_block_length||state.training_frame_count==0||
       state.development_frame_count==0||state.training_covered_frame_count==0||
       state.training_covered_frame_count>state.training_frame_count||
       state.training_discarded_tail_frames!=state.training_frame_count-state.training_covered_frame_count)))
    throw std::invalid_argument("invalid feasibility controller journal state");
  if(state.selection_state=="NOT_FROZEN"){
    if(!state.frozen_candidate_branch.empty()||state.frozen_candidate_index!=0||state.frozen_candidate_lambda!=0.0||
       !state.frozen_candidate_fit_state_sha256.empty()||!state.frozen_candidate_identity_sha256.empty()||
       !state.development_response_sha256.empty()||!state.blind_response_sha256.empty()||!state.publication_stage_output_sha256.empty()||
       !state.publication_stage_sidecar_sha256.empty()||!state.published_output_sha256.empty()||
       !state.published_sidecar_sha256.empty()||!state.published_fit_sha256.empty()||!state.published_trace_sha256.empty())
      throw std::invalid_argument("unfrozen feasibility controller contains committed selection or publication data");
  }else if(state.selection_state!="FAILED_CLOSED"){
    const bool selected_branch=state.frozen_candidate_branch=="B0"||state.frozen_candidate_branch=="B1"||
      state.frozen_candidate_branch=="B2"||state.frozen_candidate_branch=="C0"||
      state.frozen_candidate_branch=="C1"||state.frozen_candidate_branch=="C2";
    const unsigned selected_index=state.frozen_candidate_branch.size()==2?
      static_cast<unsigned>(state.frozen_candidate_branch[1]-'0'):std::numeric_limits<unsigned>::max();
    const bool valid_final_cursor=(state.branch=="B2"&&state.family=="M0")||
      (state.branch=="C2"&&state.family=="M1");
    if(state.stage!="FREEZE"||!selected_branch||selected_index!=state.frozen_candidate_index||
       !valid_final_cursor||state.next_candidate_index!=2||state.common_probe_stage!="FINAL_M0_M1"||!state.common_probe_set_frozen||
       !is_sha256(state.frozen_candidate_fit_state_sha256)||!is_sha256(state.frozen_candidate_identity_sha256)||
       !is_sha256(state.development_response_sha256)||
       std::none_of(state.candidates.begin(),state.candidates.end(),[&](const ControllerCandidate& candidate){
         return candidate.branch==state.frozen_candidate_branch&&candidate.fit_complete&&candidate.converged&&
           candidate.score_complete&&candidate.response_replay_required&&candidate.development_pass&&
           candidate.score_generation=="FINAL_M0_M1"&&candidate.index==state.frozen_candidate_index&&
           candidate.lambda==state.frozen_candidate_lambda&&candidate.fit_state_sha256==state.frozen_candidate_fit_state_sha256;
       })||
       state.frozen_candidate_identity_sha256!=frozen_candidate_identity_sha256(state))
      throw std::invalid_argument("frozen feasibility selection is incomplete or does not match its candidate and probes");
    if((state.selection_state=="BLIND_OPENED"||state.selection_state=="PUBLISHING")&&!state.operation_in_progress)
      throw std::invalid_argument("interrupted blind or publication stage is missing its in-progress operation marker");
    if(state.selection_state=="PUBLISHED"&&state.operation_in_progress)
      throw std::invalid_argument("published feasibility result still has an operation in progress");
    if(state.selection_state=="BLIND_PASSED"||state.selection_state=="PUBLISHING"||state.selection_state=="PUBLISHED"){
      if(!is_sha256(state.blind_response_sha256))
        throw std::invalid_argument("committed feasibility blind result is missing its response snapshot digest");}
    if((state.selection_state=="FROZEN"||state.selection_state=="BLIND_OPENED")&&
       (!state.blind_response_sha256.empty()||!state.publication_stage_output_sha256.empty()||
        !state.publication_stage_sidecar_sha256.empty()||!state.published_output_sha256.empty()||
        !state.published_sidecar_sha256.empty()||!state.published_fit_sha256.empty()||!state.published_trace_sha256.empty()))
      throw std::invalid_argument("pre-blind feasibility selection contains later-stage publication data");
    if(state.selection_state=="PUBLISHING"&&
       (state.publication_stage_output_sha256.empty()!=state.publication_stage_sidecar_sha256.empty()||
        (!state.published_fit_sha256.empty()&&state.publication_stage_output_sha256.empty())))
      throw std::invalid_argument("feasibility publication stage has inconsistent artifact digests");
    if((!state.published_output_sha256.empty()&&state.published_output_sha256!=state.publication_stage_output_sha256)||
       (!state.published_sidecar_sha256.empty()&&state.published_sidecar_sha256!=state.publication_stage_sidecar_sha256))
      throw std::invalid_argument("feasibility final artifact digest differs from its prepared stage digest");
    if(state.selection_state=="PUBLISHED"&&(!is_sha256(state.published_output_sha256)||
       !is_sha256(state.published_sidecar_sha256)||!is_sha256(state.published_fit_sha256)||
       !is_sha256(state.published_trace_sha256)))
      throw std::invalid_argument("published feasibility result is missing an artifact identity");
  }
  std::uint64_t phase_gradients=0;double phase_seconds=0.0;
  for(std::size_t i=0;i<state.phase_gradients.size();++i){phase_gradients+=state.phase_gradients[i];
    if(!std::isfinite(state.phase_seconds[i])||state.phase_seconds[i]<0.0)
      throw std::invalid_argument("invalid feasibility controller phase accounting");
    phase_seconds+=state.phase_seconds[i];}
  if(phase_gradients!=state.exact_gradients||phase_seconds>state.elapsed_seconds+1.0e-9*std::max(1.0,state.elapsed_seconds)||
     state.active_candidate_elapsed_seconds>state.elapsed_seconds+1.0e-9*std::max(1.0,state.elapsed_seconds))
    throw std::invalid_argument("inconsistent feasibility controller budget accounting");
  if(!state.active_candidate_started&&state.active_candidate_elapsed_seconds!=0.0)
    throw std::invalid_argument("inactive feasibility candidate has a nonzero elapsed time");
  for(std::size_t stage=0;stage<state.stage_max_seconds.size();++stage)
    if(!std::isfinite(state.stage_max_seconds[stage])||state.stage_max_seconds[stage]<0.0||
       !std::isfinite(state.stage_partial_seconds[stage])||state.stage_partial_seconds[stage]<0.0)
      throw std::invalid_argument("invalid feasibility controller stage-cost accounting");
  for(const auto& probe:state.common_probe_vectors){if(probe.empty()||probe.size()>1000000)
      throw std::invalid_argument("invalid feasibility controller probe vector");require_finite(probe,"common probe");}
  for(std::size_t i=0;i<state.training_channel_names.size();++i){
    if(state.training_channel_names[i].empty())throw std::invalid_argument("empty frozen training statistics channel name");
    for(std::size_t prior=0;prior<i;++prior)if(state.training_channel_names[prior]==state.training_channel_names[i])
      throw std::invalid_argument("duplicate frozen training statistics channel name");}
  for(std::size_t index=0;index<state.candidates.size();++index){
    const ControllerCandidate& candidate=state.candidates[index];
    const bool known_branch=candidate.branch=="A0"||candidate.branch=="A1"||candidate.branch=="B0"||candidate.branch=="B1"||
      candidate.branch=="B2"||candidate.branch=="C0"||candidate.branch=="C1"||candidate.branch=="C2";
    const unsigned expected_index=candidate.branch=="A0"||candidate.branch=="A1"?0u:
      candidate.branch.size()==2?static_cast<unsigned>(candidate.branch[1]-'0'):std::numeric_limits<unsigned>::max();
    for(std::size_t prior=0;prior<index;++prior)if(state.candidates[prior].branch==candidate.branch)
      throw std::invalid_argument("duplicate candidate branch in feasibility controller journal");
    if(!known_branch||!std::isfinite(candidate.lambda)||candidate.lambda<0.0||
       candidate.index!=expected_index||
       (candidate.fit_complete&&((candidate.converged&&candidate.theta.empty())||!is_sha256(candidate.fit_state_sha256)))||
       (candidate.score_complete&&!candidate.fit_complete)||
       (candidate.score_complete&&!(candidate.score_generation=="CANDIDATE"||candidate.score_generation=="A0"||
         candidate.score_generation=="A1"||candidate.score_generation=="B_GATE"||candidate.score_generation=="FINAL_M0_M1"))||
       (candidate.score_complete&&candidate.score_generation=="FINAL_M0_M1"&&state.common_probe_stage!="FINAL_M0_M1")||
       (candidate.score_complete&&candidate.score_generation=="B_GATE"&&
         !(state.common_probe_stage=="B_GATE"||state.common_probe_stage=="FINAL_M0_M1"))||
       (candidate.score_complete&&candidate.score_generation=="A0"&&candidate.branch!="A0")||
       (candidate.score_complete&&candidate.score_generation=="A1"&&candidate.branch!="A1")||
       (candidate.score_complete&&!candidate.response_replay_required)||
       (candidate.score_complete&&(!std::isfinite(candidate.objective)||!std::isfinite(candidate.uncertainty))))
      throw std::invalid_argument("invalid feasibility controller candidate record");
    require_finite(candidate.theta,"candidate theta");require_finite(candidate.block_scores_b,"B block score");
    require_finite(candidate.block_scores_2b,"2B block score");
  }
}

inline std::string begin_controller_selection_freeze(ControllerJournal& state,const std::string& branch,
  const unsigned index)
{
  const auto candidate=std::find_if(state.candidates.begin(),state.candidates.end(),[&](const ControllerCandidate& value){
    return value.branch==branch&&value.index==index;});
  if(state.selection_state!="NOT_FROZEN"||state.stage!="FREEZE"||state.common_probe_stage!="FINAL_M0_M1"||
     !state.common_probe_set_frozen||candidate==state.candidates.end()||!candidate->fit_complete||!candidate->converged||
     !candidate->score_complete||!candidate->response_replay_required||candidate->score_generation!="FINAL_M0_M1"||
     !candidate->development_pass||!is_sha256(candidate->fit_state_sha256))
    throw std::invalid_argument("cannot freeze an incomplete final development candidate");
  state.frozen_candidate_branch=branch;state.frozen_candidate_index=index;state.frozen_candidate_lambda=candidate->lambda;
  state.frozen_candidate_fit_state_sha256=candidate->fit_state_sha256;
  state.frozen_candidate_identity_sha256=frozen_candidate_identity_sha256(state);
  if(!is_sha256(state.frozen_candidate_identity_sha256))
    throw std::invalid_argument("cannot compute frozen candidate identity");
  return state.frozen_candidate_identity_sha256;
}

inline void commit_controller_selection_freeze(ControllerJournal& state,const std::string& development_response_sha256)
{
  if(state.selection_state!="NOT_FROZEN"||!is_sha256(development_response_sha256)||
     state.frozen_candidate_identity_sha256!=frozen_candidate_identity_sha256(state))
    throw std::invalid_argument("cannot commit an incomplete frozen development response");
  state.development_response_sha256=development_response_sha256;state.selection_state="FROZEN";
  validate_controller_journal(state);
}

inline double elapsed_budget_on_resume(const AcceptedPoint& state,const double wall_budget_seconds,
  const double resume_epoch_seconds)
{
  if(!std::isfinite(wall_budget_seconds)||!(wall_budget_seconds>0.0)||
     !std::isfinite(state.elapsed_seconds)||state.elapsed_seconds<0.0||state.elapsed_seconds>wall_budget_seconds||
     !std::isfinite(resume_epoch_seconds)||resume_epoch_seconds<0.0)
    throw std::invalid_argument("invalid covariance checkpoint wall-budget accounting");
  if(!state.operation_in_progress)return state.elapsed_seconds;
  if(!(state.operation_start_epoch_seconds>0.0)||resume_epoch_seconds<state.operation_start_epoch_seconds)
    throw std::invalid_argument("interrupted covariance operation has an invalid resume clock");
  const double elapsed=state.elapsed_seconds+(resume_epoch_seconds-state.operation_start_epoch_seconds);
  if(!std::isfinite(elapsed))throw std::invalid_argument("interrupted covariance operation elapsed time overflowed");
  return elapsed;
}

inline double resume_startup_elapsed(const double process_started_at,const double resume_boundary,
  const double now_seconds,const bool operation_interrupted)
{
  if(!std::isfinite(process_started_at)||!std::isfinite(resume_boundary)||!std::isfinite(now_seconds)||
     process_started_at<0.0||resume_boundary<process_started_at||now_seconds<resume_boundary)
    throw std::invalid_argument("invalid resume startup clock interval");
  return now_seconds-(operation_interrupted?resume_boundary:process_started_at);
}

inline void reconcile_controller_operation_budget(ControllerJournal& journal,const ControllerJournal& saved,
  const rpmd_ja_feasibility::Limits& limits,const double wall_budget_seconds,const double resume_epoch_seconds)
{
  if(!saved.operation_in_progress)return;
  if(!(saved.operation_start_epoch_seconds>0.0)||!std::isfinite(resume_epoch_seconds)||
     resume_epoch_seconds<saved.operation_start_epoch_seconds)
    throw std::invalid_argument("interrupted controller operation has an invalid resume clock");
  const double delta=resume_epoch_seconds-saved.operation_start_epoch_seconds;
  const double elapsed=std::max(journal.elapsed_seconds,saved.elapsed_seconds+delta);
  const double phase_seconds=std::max(journal.phase_seconds[saved.active_phase],saved.phase_seconds[saved.active_phase]+delta);
  const double candidate_seconds=saved.active_candidate_started?
    std::max(journal.active_candidate_elapsed_seconds,saved.active_candidate_elapsed_seconds+delta):
    journal.active_candidate_elapsed_seconds;
  if(elapsed>wall_budget_seconds||elapsed>limits.wall_seconds||
     (limits.max_fit_family_wall_seconds>=0.0&&
       ((saved.active_phase==3&&phase_seconds>limits.max_fit_family_wall_seconds)||
        (saved.active_phase==4&&phase_seconds>limits.max_fit_family_wall_seconds))))
    throw std::runtime_error("interrupted controller operation exceeds the frozen hard budget");
  journal.elapsed_seconds=elapsed;journal.phase_seconds[saved.active_phase]=phase_seconds;
  journal.active_candidate_elapsed_seconds=candidate_seconds;
  const bool selection_operation_resumes=journal.selection_state=="BLIND_OPENED"||journal.selection_state=="PUBLISHING";
  journal.operation_in_progress=selection_operation_resumes;
  journal.operation_start_epoch_seconds=selection_operation_resumes?resume_epoch_seconds:0.0;
  validate_controller_journal(journal);
}

inline bool preserve_later_family_cursor(const ControllerJournal& journal,const bool resuming,
  const std::string& stage,const std::string& family,const unsigned candidate_index)
{
  return resuming&&journal.active_candidate_started&&journal.stage==stage&&journal.family==family&&
    journal.next_candidate_index>candidate_index;
}

inline void reconcile_controller_resume_budget(ControllerJournal& journal,AcceptedPoint& solver_state,
  const rpmd_ja_feasibility::Limits& limits,const double wall_budget_seconds,const double resume_epoch_seconds)
{
  if(!limits.valid())throw std::invalid_argument("invalid frozen feasibility limits during resume");
  if(solver_state.job_id!=journal.job_id||solver_state.identity_sha256!=journal.identity_sha256||
     solver_state.branch!=journal.branch||solver_state.family!=journal.family||
     solver_state.candidate_index!=journal.next_candidate_index||solver_state.active_phase!=journal.active_phase||
     solver_state.selection_state!="NOT_FROZEN")
    throw std::runtime_error("active covariance checkpoint does not match its outer controller cursor");
  ControllerJournal reconciled=journal;AcceptedPoint reconciled_solver=solver_state;
  const double saved_journal_elapsed=journal.elapsed_seconds;
  const double elapsed=elapsed_budget_on_resume(solver_state,wall_budget_seconds,resume_epoch_seconds);
  const double interrupted_delta=solver_state.operation_in_progress?elapsed-solver_state.elapsed_seconds:0.0;
  reconciled.elapsed_seconds=std::max(journal.elapsed_seconds,elapsed);
  reconciled.exact_gradients=std::max(journal.exact_gradients,solver_state.total_budget_gradients+
    static_cast<std::uint64_t>(solver_state.operation_in_progress));
  for(std::size_t phase=0;phase<journal.phase_gradients.size();++phase){
    unsigned solver_phase_gradients=solver_state.phase_gradients[phase];
    double solver_phase_seconds=solver_state.phase_seconds[phase];
    if(solver_state.operation_in_progress&&phase==solver_state.active_phase){
      ++solver_phase_gradients;solver_phase_seconds+=interrupted_delta;}
    reconciled.phase_gradients[phase]=std::max(journal.phase_gradients[phase],solver_phase_gradients);
    reconciled.phase_seconds[phase]=std::max(journal.phase_seconds[phase],solver_phase_seconds);}
  if(journal.active_candidate_started){
    const double before_operation=std::max(0.0,solver_state.elapsed_seconds-saved_journal_elapsed);
    const double journaled_operation=solver_state.operation_in_progress?std::min(interrupted_delta,
      std::max(0.0,saved_journal_elapsed-solver_state.elapsed_seconds)):0.0;
    reconciled.active_candidate_elapsed_seconds+=before_operation+interrupted_delta-journaled_operation;}
  if(reconciled.elapsed_seconds>wall_budget_seconds||reconciled.elapsed_seconds>limits.wall_seconds||
     reconciled.exact_gradients>static_cast<std::uint64_t>(limits.max_exact_gradients)||
     reconciled.phase_gradients[3]>static_cast<unsigned>(limits.max_fit_family_gradients)||
     reconciled.phase_gradients[4]>static_cast<unsigned>(limits.max_fit_family_gradients)||
     (limits.max_fit_family_wall_seconds>=0.0&&
       ((journal.active_phase==3&&reconciled.phase_seconds[3]>limits.max_fit_family_wall_seconds)||
        (journal.active_phase==4&&reconciled.phase_seconds[4]>limits.max_fit_family_wall_seconds))))
    throw std::runtime_error("interrupted covariance operation exceeds the frozen hard budget");
  validate_controller_journal(reconciled);
  reconciled_solver.elapsed_seconds=reconciled.elapsed_seconds;reconciled_solver.total_budget_gradients=reconciled.exact_gradients;
  reconciled_solver.active_phase=reconciled.active_phase;reconciled_solver.phase_gradients=reconciled.phase_gradients;
  reconciled_solver.phase_seconds=reconciled.phase_seconds;reconciled_solver.operation_in_progress=false;
  reconciled_solver.operation_start_epoch_seconds=resume_epoch_seconds;
  journal=std::move(reconciled);solver_state=std::move(reconciled_solver);
}

inline std::uint64_t next_cg_witness_attempt(ControllerJournal& journal)
{
  if(journal.cg_witness_attempts==std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("feasibility CG witness attempt counter overflow");
  return ++journal.cg_witness_attempts;
}

struct TemporaryFileCleanup
{
  std::string path;
  bool active=true;
  ~TemporaryFileCleanup(){if(active)rpmd_ja_file_io::remove_file(path);}
};

inline void write_vector(std::ostream& out,const char* key,const std::vector<double>& values)
{
  out<<key<<' '<<values.size();
  for(const double value:values)out<<' '<<value;
  out<<'\n';
}

inline void write_checkpoint(const std::string& path,const AcceptedPoint& state)
{
  validate_state(state);
  std::ostringstream payload;payload<<std::setprecision(17)<<"RPJACKPT4\n"
    <<"job_id "<<std::quoted(state.job_id)<<'\n'<<"identity_sha256 "<<std::quoted(state.identity_sha256)<<'\n'
    <<"branch "<<std::quoted(state.branch)<<'\n'<<"family "<<std::quoted(state.family)<<'\n'
    <<"candidate_index "<<state.candidate_index<<'\n'<<"selection_state "<<std::quoted(state.selection_state)<<'\n'
    <<"active_phase "<<state.active_phase<<'\n'
    <<"lambda "<<state.lambda<<'\n'<<"epsilon "<<state.epsilon<<'\n'<<"rho "<<state.rho<<'\n'
    <<"ridge_lambda "<<state.ridge_lambda<<'\n'<<"tau "<<state.tau<<'\n'<<"trust_radius "<<state.trust_radius<<'\n'
    <<"start_graph_delta "<<state.start_graph_delta<<'\n'
    <<"elapsed_seconds "<<state.elapsed_seconds<<'\n'<<"operation_start_epoch_seconds "<<state.operation_start_epoch_seconds<<'\n'
    <<"operation_in_progress "<<(state.operation_in_progress?1:0)<<'\n'<<"max_objective_evaluations "<<state.max_objective_evaluations<<'\n'
    <<"objective_evaluations "<<state.objective_evaluations<<'\n'<<"gradient_evaluations "<<state.gradient_evaluations<<'\n'
    <<"total_budget_gradients "<<state.total_budget_gradients<<'\n'
    <<"initialization_evaluations "<<state.initialization_evaluations<<'\n'<<"iterations "<<state.iterations<<'\n'
    <<"curvature_probe_count "<<state.curvature_probe_count<<'\n'
    <<"curvature_probe_count_verified "<<(state.curvature_probe_count_verified?1:0)<<'\n'
    <<"performance_window_initialized "<<(state.performance_window_initialized?1:0)<<'\n'
    <<"performance_window_curvature_builds "<<state.performance_window_curvature_builds<<'\n'
    <<"performance_window_tau "<<state.performance_window_tau<<'\n'
    <<"performance_window_start_objective "<<state.performance_window_start_objective<<'\n'
    <<"performance_window_curvature_seconds "<<state.performance_window_curvature_seconds<<'\n'
    <<"performance_window_reference_gradient_seconds "<<state.performance_window_reference_gradient_seconds<<'\n'
    <<"objective_evaluation_seconds "<<state.objective_evaluation_seconds<<'\n'
    <<"full_gradient_seconds "<<state.full_gradient_seconds<<'\n'
    <<"last_full_gradient_seconds "<<state.last_full_gradient_seconds<<'\n'
    <<"factorization_seconds "<<state.factorization_seconds<<'\n'
    <<"analytic_curvature_seconds "<<state.analytic_curvature_seconds<<'\n'<<"phase_gradients";
  for(const unsigned value:state.phase_gradients)payload<<' '<<value;
  payload<<'\n'<<"phase_seconds";for(const double value:state.phase_seconds){if(!std::isfinite(value)||value<0.0)
      throw std::invalid_argument("invalid covariance checkpoint phase elapsed time");payload<<' '<<value;}
  payload<<'\n'<<"phase_cursor";for(const unsigned value:state.phase_cursor)payload<<' '<<value;
  payload<<'\n'<<"bfgs_seed_gamma "<<state.bfgs_seed_gamma<<'\n'
    <<"bfgs_last_valid_gamma_valid "<<(!std::isnan(state.bfgs_last_valid_gamma)?1:0)<<'\n'
    <<"bfgs_last_valid_gamma "<<(std::isfinite(state.bfgs_last_valid_gamma)?state.bfgs_last_valid_gamma:0.0)<<'\n'
    <<"bfgs_stage_seeded "<<(state.bfgs_stage_seeded?1:0)<<'\n'<<"bfgs_seed_source "<<std::quoted(state.bfgs_seed_source)<<'\n';
  payload<<"stagnation_gradient_attempts "<<state.stagnation_gradient_attempts<<'\n'
    <<"stagnation_low_progress_windows "<<state.stagnation_low_progress_windows<<'\n'
    <<"stagnation_progress_attempts_in_window "<<state.stagnation_progress_attempts_in_window<<'\n'
    <<"same_point_curvature_rejections "<<state.same_point_curvature_rejections<<'\n'
    <<"stagnation_best_kkt "<<(std::isfinite(state.stagnation_best_kkt)?state.stagnation_best_kkt:0.0)<<'\n'
    <<"stagnation_window_start_kkt "<<(std::isfinite(state.stagnation_window_start_kkt)?state.stagnation_window_start_kkt:0.0)<<'\n'
    <<"stagnation_last_window_relative_improvement "
      <<(std::isfinite(state.stagnation_last_window_relative_improvement)?state.stagnation_last_window_relative_improvement:0.0)<<'\n';
  write_vector(payload,"theta",state.theta);write_vector(payload,"theta_start",state.theta_start);
  write_vector(payload,"stagnation_best_theta",state.stagnation_best_theta);write_vector(payload,"coordinate_scale",state.coordinate_scale);
  write_vector(payload,"bfgs_inverse",state.bfgs_inverse);payload<<"bfgs_recent_secant_count "<<state.bfgs_recent_secant_count<<'\n';
  write_vector(payload,"bfgs_recent_secants",state.bfgs_recent_secants);
  const std::string bytes=payload.str();
  if(bytes.size()+72>maximum_checkpoint_bytes)throw std::length_error("covariance checkpoint exceeds the size limit");
  const std::string temporary=path+".tmp."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  TemporaryFileCleanup cleanup{temporary};
  rpmd_ja_file_io::ofstream out(temporary,std::ios::out|std::ios::binary);
  if(!out)throw std::runtime_error("cannot create covariance checkpoint temporary file: "+temporary);
  out<<bytes<<"sha256 "<<sha256(bytes)<<'\n';out.flush();
  if(!out)throw std::runtime_error("failed writing covariance checkpoint temporary file: "+temporary);
  out.close();if(!out)throw std::runtime_error("failed closing covariance checkpoint temporary file: "+temporary);
  if(rpmd_ja_file_io::atomic_replace_file(temporary,path)!=0)
    throw std::runtime_error("cannot atomically replace covariance checkpoint: "+path);
  cleanup.active=false;
}

inline AcceptedPoint read_checkpoint(const std::string& path,const std::string& expected_identity_sha256)
{
  rpmd_ja_file_io::ifstream in(path,std::ios::in|std::ios::binary);
  if(!in)throw std::runtime_error("cannot open accepted-point covariance checkpoint: "+path);
  in.seekg(0,std::ios::end);const std::streamoff file_size=in.tellg();
  if(file_size<=0||static_cast<std::uint64_t>(file_size)>maximum_checkpoint_bytes)
    throw std::runtime_error("covariance checkpoint file size is invalid or exceeds the limit");
  in.clear();in.seekg(0,std::ios::beg);
  std::ostringstream contents;contents<<in.rdbuf();if(in.bad())
    throw std::runtime_error("failed reading accepted-point covariance checkpoint: "+path);
  const std::string bytes=contents.str();
  const std::size_t checksum_row=bytes.rfind("sha256 ");
  if(checksum_row==std::string::npos||checksum_row==0||bytes.back()!='\n')
    throw std::runtime_error("covariance checkpoint is truncated or missing its checksum");
  const std::string payload=bytes.substr(0,checksum_row);
  std::istringstream checksum_stream(bytes.substr(checksum_row));std::string key,checksum,extra;
  if(!(checksum_stream>>key>>checksum)||key!="sha256"||(checksum_stream>>extra)||sha256(payload)!=checksum)
    throw std::runtime_error("covariance checkpoint checksum mismatch");
  std::istringstream input(payload);std::string line;
  if(!std::getline(input,line)||line!="RPJACKPT4")throw std::runtime_error("unsupported covariance checkpoint version");
  AcceptedPoint state;
  auto row=[&](const char* expected){if(!std::getline(input,line))throw std::runtime_error("truncated covariance checkpoint");
    std::istringstream values(line);std::string key;if(!(values>>key)||key!=expected)throw std::runtime_error(std::string("invalid covariance checkpoint field: ")+expected);};
  auto quoted=[&](const char* key,std::string& value){row(key);std::istringstream values(line);std::string name,extra;
    if(!(values>>name>>std::quoted(value))||(values>>extra))throw std::runtime_error(std::string("invalid covariance checkpoint field: ")+key);};
  auto scalar=[&](const char* key,auto& value){row(key);std::istringstream values(line);std::string name,extra;
    if(!(values>>name>>value)||(values>>extra))throw std::runtime_error(std::string("invalid covariance checkpoint field: ")+key);};
  auto vector=[&](const char* key,std::vector<double>& values){row(key);std::istringstream row_stream(line);std::string name;std::size_t count=0;
    if(!(row_stream>>name>>count)||count>10000)throw std::runtime_error(std::string("invalid covariance checkpoint vector: ")+key);
    values.resize(count);for(double& value:values)if(!(row_stream>>value)||!std::isfinite(value))
      throw std::runtime_error(std::string("invalid covariance checkpoint vector: ")+key);
    row_stream>>std::ws;if(!row_stream.eof())throw std::runtime_error(std::string("trailing covariance checkpoint vector data: ")+key);};
  quoted("job_id",state.job_id);quoted("identity_sha256",state.identity_sha256);quoted("branch",state.branch);
  quoted("family",state.family);scalar("candidate_index",state.candidate_index);quoted("selection_state",state.selection_state);
  scalar("active_phase",state.active_phase);
  scalar("lambda",state.lambda);scalar("epsilon",state.epsilon);scalar("rho",state.rho);scalar("ridge_lambda",state.ridge_lambda);
  scalar("tau",state.tau);scalar("trust_radius",state.trust_radius);scalar("start_graph_delta",state.start_graph_delta);
  scalar("elapsed_seconds",state.elapsed_seconds);
  scalar("operation_start_epoch_seconds",state.operation_start_epoch_seconds);
  unsigned operation=0;scalar("operation_in_progress",operation);if(operation>1)throw std::runtime_error("invalid checkpoint operation flag");
  state.operation_in_progress=operation!=0;
  scalar("max_objective_evaluations",state.max_objective_evaluations);scalar("objective_evaluations",state.objective_evaluations);
  scalar("gradient_evaluations",state.gradient_evaluations);scalar("total_budget_gradients",state.total_budget_gradients);
  scalar("initialization_evaluations",state.initialization_evaluations);
  scalar("iterations",state.iterations);scalar("curvature_probe_count",state.curvature_probe_count);
  unsigned curvature_verified=0;scalar("curvature_probe_count_verified",curvature_verified);
  if(curvature_verified>1)throw std::runtime_error("invalid checkpoint curvature verification flag");
  state.curvature_probe_count_verified=curvature_verified!=0;
  unsigned performance_initialized=0;scalar("performance_window_initialized",performance_initialized);
  if(performance_initialized>1)throw std::runtime_error("invalid checkpoint performance-window flag");
  state.performance_window_initialized=performance_initialized!=0;
  scalar("performance_window_curvature_builds",state.performance_window_curvature_builds);
  scalar("performance_window_tau",state.performance_window_tau);
  scalar("performance_window_start_objective",state.performance_window_start_objective);
  scalar("performance_window_curvature_seconds",state.performance_window_curvature_seconds);
  scalar("performance_window_reference_gradient_seconds",state.performance_window_reference_gradient_seconds);
  scalar("objective_evaluation_seconds",state.objective_evaluation_seconds);
  scalar("full_gradient_seconds",state.full_gradient_seconds);
  scalar("last_full_gradient_seconds",state.last_full_gradient_seconds);
  scalar("factorization_seconds",state.factorization_seconds);
  scalar("analytic_curvature_seconds",state.analytic_curvature_seconds);
  row("phase_gradients");{std::istringstream values(line);std::string name,extra;if(!(values>>name))throw std::runtime_error("invalid phase gradients");
    for(unsigned& value:state.phase_gradients)if(!(values>>value))throw std::runtime_error("invalid phase gradients");values>>std::ws;
    if(!values.eof())throw std::runtime_error("trailing phase gradients");}
  row("phase_seconds");{std::istringstream values(line);std::string name,extra;if(!(values>>name))throw std::runtime_error("invalid phase seconds");
    for(double& value:state.phase_seconds)if(!(values>>value)||!std::isfinite(value)||value<0.0)throw std::runtime_error("invalid phase seconds");values>>std::ws;
    if(!values.eof())throw std::runtime_error("trailing phase seconds");}
  row("phase_cursor");{std::istringstream values(line);std::string name;if(!(values>>name))throw std::runtime_error("invalid phase cursor");
    for(unsigned& value:state.phase_cursor)if(!(values>>value))throw std::runtime_error("invalid phase cursor");values>>std::ws;
    if(!values.eof())throw std::runtime_error("trailing phase cursor");}
  scalar("bfgs_seed_gamma",state.bfgs_seed_gamma);unsigned last_gamma_valid=0;
  scalar("bfgs_last_valid_gamma_valid",last_gamma_valid);if(last_gamma_valid>1)throw std::runtime_error("invalid BFGS gamma flag");
  double last_gamma=0.0;scalar("bfgs_last_valid_gamma",last_gamma);
  state.bfgs_last_valid_gamma=last_gamma_valid?last_gamma:std::numeric_limits<double>::quiet_NaN();
  unsigned seeded=0;scalar("bfgs_stage_seeded",seeded);if(seeded>1)throw std::runtime_error("invalid BFGS stage flag");state.bfgs_stage_seeded=seeded!=0;
  quoted("bfgs_seed_source",state.bfgs_seed_source);
  scalar("stagnation_gradient_attempts",state.stagnation_gradient_attempts);
  scalar("stagnation_low_progress_windows",state.stagnation_low_progress_windows);
  scalar("stagnation_progress_attempts_in_window",state.stagnation_progress_attempts_in_window);
  scalar("same_point_curvature_rejections",state.same_point_curvature_rejections);
  scalar("stagnation_best_kkt",state.stagnation_best_kkt);scalar("stagnation_window_start_kkt",state.stagnation_window_start_kkt);
  scalar("stagnation_last_window_relative_improvement",state.stagnation_last_window_relative_improvement);
  vector("theta",state.theta);vector("theta_start",state.theta_start);vector("stagnation_best_theta",state.stagnation_best_theta);
  vector("coordinate_scale",state.coordinate_scale);
  vector("bfgs_inverse",state.bfgs_inverse);scalar("bfgs_recent_secant_count",state.bfgs_recent_secant_count);
  vector("bfgs_recent_secants",state.bfgs_recent_secants);input>>std::ws;
  if(!input.eof())throw std::runtime_error("trailing covariance checkpoint payload");
  if(!is_sha256(state.identity_sha256)||state.identity_sha256!=expected_identity_sha256)
    throw std::runtime_error("covariance checkpoint identity mismatch");
  try{validate_state(state);}catch(const std::exception& error){throw std::runtime_error(error.what());}
  return state;
}

inline void write_controller_vector(std::ostream& out,const std::vector<double>& values)
{
  out<<values.size();for(const double value:values)out<<' '<<value;
}

inline void write_controller_journal(const std::string& path,const ControllerJournal& state)
{
  validate_controller_journal(state);
  std::ostringstream payload;payload<<std::setprecision(17)<<"RPJAOUTER5\n"
    <<"job_id "<<std::quoted(state.job_id)<<'\n'<<"identity_sha256 "<<std::quoted(state.identity_sha256)<<'\n'
    <<"branch "<<std::quoted(state.branch)<<'\n'<<"family "<<std::quoted(state.family)<<'\n'
    <<"stage "<<std::quoted(state.stage)<<'\n'<<"selection_state "<<std::quoted(state.selection_state)<<'\n'
    <<"frozen_candidate_branch "<<std::quoted(state.frozen_candidate_branch)<<'\n'
    <<"frozen_candidate_index "<<state.frozen_candidate_index<<'\n'
    <<"frozen_candidate_lambda "<<state.frozen_candidate_lambda<<'\n'
    <<"frozen_candidate_fit_state_sha256 "<<std::quoted(state.frozen_candidate_fit_state_sha256)<<'\n'
    <<"frozen_candidate_identity_sha256 "<<std::quoted(state.frozen_candidate_identity_sha256)<<'\n'
    <<"development_response_sha256 "<<std::quoted(state.development_response_sha256)<<'\n'
    <<"blind_response_sha256 "<<std::quoted(state.blind_response_sha256)<<'\n'
    <<"publication_stage_output_sha256 "<<std::quoted(state.publication_stage_output_sha256)<<'\n'
    <<"publication_stage_sidecar_sha256 "<<std::quoted(state.publication_stage_sidecar_sha256)<<'\n'
    <<"published_output_sha256 "<<std::quoted(state.published_output_sha256)<<'\n'
    <<"published_sidecar_sha256 "<<std::quoted(state.published_sidecar_sha256)<<'\n'
    <<"published_fit_sha256 "<<std::quoted(state.published_fit_sha256)<<'\n'
    <<"published_trace_sha256 "<<std::quoted(state.published_trace_sha256)<<'\n'
    <<"common_probe_stage "<<std::quoted(state.common_probe_stage)<<'\n'
    <<"next_candidate_index "<<state.next_candidate_index<<'\n'<<"active_phase "<<state.active_phase<<'\n'
    <<"active_candidate_started "<<(state.active_candidate_started?1:0)<<'\n'
    <<"max_fit_family_gradients "<<state.max_fit_family_gradients<<'\n'
    <<"max_fit_family_wall_seconds "<<state.max_fit_family_wall_seconds<<'\n'
    <<"elapsed_seconds "<<state.elapsed_seconds<<'\n'
    <<"operation_in_progress "<<(state.operation_in_progress?1:0)<<'\n'
    <<"operation_start_epoch_seconds "<<state.operation_start_epoch_seconds<<'\n'
    <<"exact_gradients "<<state.exact_gradients<<'\n'
    <<"cg_witness_attempts "<<state.cg_witness_attempts<<'\n'
    <<"active_candidate_elapsed_seconds "<<state.active_candidate_elapsed_seconds<<'\n'
    <<"phase_gradients";for(const unsigned value:state.phase_gradients)payload<<' '<<value;payload<<'\n'
    <<"phase_seconds";for(const double value:state.phase_seconds)payload<<' '<<value;payload<<'\n'
    <<"stage_completed_calls";for(const unsigned value:state.stage_completed_calls)payload<<' '<<value;payload<<'\n'
    <<"stage_max_seconds";for(const double value:state.stage_max_seconds)payload<<' '<<value;payload<<'\n'
    <<"stage_partial_seconds";for(const double value:state.stage_partial_seconds)payload<<' '<<value;payload<<'\n'
    <<"candidate_cost_samples "<<state.candidate_cost_samples<<'\n'
    <<"candidate_cost_max_seconds "<<state.candidate_cost_max_seconds<<'\n'
    <<"candidate_cost_partial_seconds "<<state.candidate_cost_partial_seconds<<'\n'
    <<"training_statistics_frozen "<<(state.training_statistics_frozen?1:0)<<'\n'
    <<"training_statistics_policy_version "<<std::quoted(state.training_statistics_policy_version)<<'\n'
    <<"training_frame_count "<<state.training_frame_count<<'\n'
    <<"development_frame_count "<<state.development_frame_count<<'\n'
    <<"training_block_length "<<state.training_block_length<<'\n'
    <<"training_adjacent_block_length "<<state.training_adjacent_block_length<<'\n'
    <<"training_covered_frame_count "<<state.training_covered_frame_count<<'\n'
    <<"training_discarded_tail_frames "<<state.training_discarded_tail_frames<<'\n'
    <<"training_channel_count "<<state.training_channel_names.size()<<'\n';
  for(const std::string& name:state.training_channel_names)payload<<"training_channel "<<std::quoted(name)<<'\n';
  payload<<"common_probe_set_frozen "<<(state.common_probe_set_frozen?1:0)<<'\n'
    <<"common_probe_count "<<state.common_probe_sources.size()<<'\n';
  for(std::size_t i=0;i<state.common_probe_sources.size();++i){payload<<"probe "<<std::quoted(state.common_probe_sources[i])<<' '
      <<std::quoted(state.common_probe_origins[i])<<' ';write_controller_vector(payload,state.common_probe_vectors[i]);payload<<'\n';}
  payload<<"candidate_count "<<state.candidates.size()<<'\n';
  for(const ControllerCandidate& candidate:state.candidates){payload<<"candidate "<<std::quoted(candidate.branch)<<' '
      <<std::quoted(candidate.fit_state_sha256)<<' '<<std::quoted(candidate.score_generation)<<' '<<candidate.index<<' '
      <<candidate.lambda<<' '<<(candidate.fit_complete?1:0)<<' '<<(candidate.score_complete?1:0)<<' '
      <<(candidate.response_replay_required?1:0)<<' '
      <<(candidate.converged?1:0)<<' '<<(candidate.development_pass?1:0)<<' '<<(candidate.b_gate_pass?1:0)<<' '
      <<candidate.objective<<' '<<candidate.uncertainty<<' ';
    write_controller_vector(payload,candidate.theta);payload<<' ';write_controller_vector(payload,candidate.block_scores_b);
    payload<<' ';write_controller_vector(payload,candidate.block_scores_2b);payload<<'\n';}
  const std::string bytes=payload.str();
  if(bytes.size()+72>maximum_controller_journal_bytes)throw std::length_error("feasibility controller journal exceeds the size limit");
  const std::string temporary=path+".tmp."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  TemporaryFileCleanup cleanup{temporary};rpmd_ja_file_io::ofstream out(temporary,std::ios::out|std::ios::binary);
  if(!out)throw std::runtime_error("cannot create feasibility controller journal temporary file: "+temporary);
  out<<bytes<<"sha256 "<<sha256(bytes)<<'\n';out.flush();if(!out)throw std::runtime_error("failed writing feasibility controller journal");
  out.close();if(!out||rpmd_ja_file_io::atomic_replace_file(temporary,path)!=0)
    throw std::runtime_error("cannot atomically replace feasibility controller journal: "+path);
  cleanup.active=false;
}

inline ControllerJournal read_controller_journal(const std::string& path,const std::string& expected_identity_sha256)
{
  rpmd_ja_file_io::ifstream in(path,std::ios::in|std::ios::binary);if(!in)
    throw std::runtime_error("cannot open feasibility controller journal: "+path);
  in.seekg(0,std::ios::end);const std::streamoff size=in.tellg();
  if(size<=0||static_cast<std::uint64_t>(size)>maximum_controller_journal_bytes)
    throw std::runtime_error("feasibility controller journal size is invalid");
  in.clear();in.seekg(0,std::ios::beg);std::ostringstream contents;contents<<in.rdbuf();
  const std::string bytes=contents.str();const std::size_t checksum_row=bytes.rfind("sha256 ");
  if(checksum_row==std::string::npos||checksum_row==0||bytes.back()!='\n')
    throw std::runtime_error("feasibility controller journal is truncated or missing its checksum");
  const std::string payload=bytes.substr(0,checksum_row);std::istringstream checksum(bytes.substr(checksum_row));
  std::string key,digest,extra;if(!(checksum>>key>>digest)||(checksum>>extra)||key!="sha256"||sha256(payload)!=digest)
    throw std::runtime_error("feasibility controller journal checksum mismatch");
  std::istringstream input(payload);std::string line;
  if(!std::getline(input,line)||line!="RPJAOUTER5")throw std::runtime_error("unsupported feasibility controller journal version");
  ControllerJournal state;
  auto next=[&](const char* expected){if(!std::getline(input,line))throw std::runtime_error("truncated feasibility controller journal");
    std::istringstream row(line);std::string name;if(!(row>>name)||name!=expected)throw std::runtime_error(std::string("invalid controller journal field: ")+expected);return row;};
  auto quoted=[&](const char* name,std::string& value){auto row=next(name);std::string extra;if(!(row>>std::quoted(value))||(row>>extra))
      throw std::runtime_error(std::string("invalid controller journal field: ")+name);};
  auto scalar=[&](const char* name,auto& value){auto row=next(name);std::string extra;if(!(row>>value)||(row>>extra))
      throw std::runtime_error(std::string("invalid controller journal field: ")+name);};
  auto unsigned_scalar=[&](const char* name,std::uint64_t& value){auto row=next(name);std::string token,extra;
    if(!(row>>token)||(row>>extra)||token.empty()||
       !std::all_of(token.begin(),token.end(),[](const unsigned char c){return c>='0'&&c<='9';}))
      throw std::runtime_error(std::string("invalid controller journal field: ")+name);
    std::size_t consumed=0;try{value=std::stoull(token,&consumed);}catch(...){
      throw std::runtime_error(std::string("invalid controller journal field: ")+name);}
    if(consumed!=token.size())throw std::runtime_error(std::string("invalid controller journal field: ")+name);};
  auto array=[&](const char* name,auto& values){auto row=next(name);std::string token;if(!(row>>token))
      throw std::runtime_error(std::string("invalid controller journal field: ")+name);
    for(auto& value:values)if(!(row>>value))throw std::runtime_error(std::string("invalid controller journal field: ")+name);
    row>>std::ws;if(!row.eof())throw std::runtime_error(std::string("trailing controller journal field: ")+name);};
  auto vector=[&](std::istream& row,std::vector<double>& values){std::size_t count=0;if(!(row>>count)||count>1000000)
      throw std::runtime_error("invalid controller journal vector length");values.resize(count);
    for(double& value:values)if(!(row>>value)||!std::isfinite(value))throw std::runtime_error("invalid controller journal vector value");};
  quoted("job_id",state.job_id);quoted("identity_sha256",state.identity_sha256);quoted("branch",state.branch);
  quoted("family",state.family);quoted("stage",state.stage);quoted("selection_state",state.selection_state);
  quoted("frozen_candidate_branch",state.frozen_candidate_branch);scalar("frozen_candidate_index",state.frozen_candidate_index);
  scalar("frozen_candidate_lambda",state.frozen_candidate_lambda);quoted("frozen_candidate_fit_state_sha256",state.frozen_candidate_fit_state_sha256);
  quoted("frozen_candidate_identity_sha256",state.frozen_candidate_identity_sha256);
  quoted("development_response_sha256",state.development_response_sha256);quoted("blind_response_sha256",state.blind_response_sha256);
  quoted("publication_stage_output_sha256",state.publication_stage_output_sha256);
  quoted("publication_stage_sidecar_sha256",state.publication_stage_sidecar_sha256);
  quoted("published_output_sha256",state.published_output_sha256);quoted("published_sidecar_sha256",state.published_sidecar_sha256);
  quoted("published_fit_sha256",state.published_fit_sha256);quoted("published_trace_sha256",state.published_trace_sha256);
  quoted("common_probe_stage",state.common_probe_stage);
  scalar("next_candidate_index",state.next_candidate_index);scalar("active_phase",state.active_phase);
  unsigned active_candidate=0;scalar("active_candidate_started",active_candidate);
  if(active_candidate>1)throw std::runtime_error("invalid active-candidate flag");state.active_candidate_started=active_candidate!=0;
  scalar("max_fit_family_gradients",state.max_fit_family_gradients);scalar("max_fit_family_wall_seconds",state.max_fit_family_wall_seconds);
  scalar("elapsed_seconds",state.elapsed_seconds);unsigned operation_in_progress=0;
  scalar("operation_in_progress",operation_in_progress);if(operation_in_progress>1)throw std::runtime_error("invalid controller operation flag");
  state.operation_in_progress=operation_in_progress!=0;scalar("operation_start_epoch_seconds",state.operation_start_epoch_seconds);
  scalar("exact_gradients",state.exact_gradients);
  unsigned_scalar("cg_witness_attempts",state.cg_witness_attempts);
  scalar("active_candidate_elapsed_seconds",state.active_candidate_elapsed_seconds);
  array("phase_gradients",state.phase_gradients);array("phase_seconds",state.phase_seconds);
  array("stage_completed_calls",state.stage_completed_calls);array("stage_max_seconds",state.stage_max_seconds);
  array("stage_partial_seconds",state.stage_partial_seconds);scalar("candidate_cost_samples",state.candidate_cost_samples);
  scalar("candidate_cost_max_seconds",state.candidate_cost_max_seconds);scalar("candidate_cost_partial_seconds",state.candidate_cost_partial_seconds);
  unsigned training_frozen=0;scalar("training_statistics_frozen",training_frozen);
  if(training_frozen>1)throw std::runtime_error("invalid frozen training-statistics flag");state.training_statistics_frozen=training_frozen!=0;
  quoted("training_statistics_policy_version",state.training_statistics_policy_version);
  scalar("training_frame_count",state.training_frame_count);scalar("development_frame_count",state.development_frame_count);
  scalar("training_block_length",state.training_block_length);scalar("training_adjacent_block_length",state.training_adjacent_block_length);
  scalar("training_covered_frame_count",state.training_covered_frame_count);
  scalar("training_discarded_tail_frames",state.training_discarded_tail_frames);
  std::size_t training_channel_count=0;scalar("training_channel_count",training_channel_count);
  if(training_channel_count>1000)throw std::runtime_error("frozen training-statistics channel count exceeds its limit");
  for(std::size_t i=0;i<training_channel_count;++i){auto row=next("training_channel");std::string name,extra;
    if(!(row>>std::quoted(name))||(row>>extra))throw std::runtime_error("invalid frozen training-statistics channel");
    state.training_channel_names.push_back(std::move(name));}
  unsigned frozen=0;scalar("common_probe_set_frozen",frozen);if(frozen>1)throw std::runtime_error("invalid common probe freeze flag");
  state.common_probe_set_frozen=frozen!=0;std::size_t probe_count=0;scalar("common_probe_count",probe_count);
  if(probe_count>10000)throw std::runtime_error("feasibility controller probe count exceeds its limit");
  for(std::size_t i=0;i<probe_count;++i){auto row=next("probe");std::string source,origin;if(!(row>>std::quoted(source)>>std::quoted(origin)))
      throw std::runtime_error("invalid controller journal probe identity");std::vector<double> values;vector(row,values);
    row>>std::ws;if(!row.eof())throw std::runtime_error("trailing controller journal probe data");
    state.common_probe_sources.push_back(std::move(source));state.common_probe_origins.push_back(std::move(origin));
    state.common_probe_vectors.push_back(std::move(values));}
  std::size_t candidate_count=0;scalar("candidate_count",candidate_count);
  if(candidate_count>8)throw std::runtime_error("feasibility controller candidate count exceeds its limit");
  for(std::size_t i=0;i<candidate_count;++i){auto row=next("candidate");ControllerCandidate candidate;
    unsigned fit=0,score=0,response_replay=0,converged=0,development_pass=0,b_gate_pass=0;if(!(row>>std::quoted(candidate.branch)>>std::quoted(candidate.fit_state_sha256)
       >>std::quoted(candidate.score_generation)
       >>candidate.index>>candidate.lambda>>fit>>score>>response_replay>>converged>>development_pass>>b_gate_pass
       >>candidate.objective>>candidate.uncertainty)||fit>1||score>1||response_replay>1||converged>1||development_pass>1||b_gate_pass>1)
      throw std::runtime_error("invalid controller journal candidate");
    candidate.fit_complete=fit!=0;candidate.score_complete=score!=0;candidate.response_replay_required=response_replay!=0;
    candidate.converged=converged!=0;
    candidate.development_pass=development_pass!=0;
    candidate.b_gate_pass=b_gate_pass!=0;
    vector(row,candidate.theta);vector(row,candidate.block_scores_b);vector(row,candidate.block_scores_2b);
    row>>std::ws;if(!row.eof())throw std::runtime_error("trailing controller journal candidate data");
    state.candidates.push_back(std::move(candidate));}
  input>>std::ws;if(!input.eof())throw std::runtime_error("trailing feasibility controller journal payload");
  if(state.identity_sha256!=expected_identity_sha256)throw std::runtime_error("feasibility controller journal identity mismatch");
  try{validate_controller_journal(state);}catch(const std::exception& error){throw std::runtime_error(error.what());}
  return state;
}
}
