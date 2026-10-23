#include "../src/measure/rpmd_ja_covariance_checkpoint.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
using rpmd_ja_covariance_checkpoint::AcceptedPoint;
using rpmd_ja_covariance_checkpoint::ControllerCandidate;
using rpmd_ja_covariance_checkpoint::ControllerJournal;

void require(const bool condition,const char* message)
{
  if(!condition)throw std::runtime_error(message);
}

AcceptedPoint valid_state()
{
  AcceptedPoint state;
  state.job_id = "cpu-checkpoint-test";
  state.identity_sha256 = std::string(64, 'a');
  state.branch = "A0";
  state.family = "M0";
  state.selection_state = "NOT_FROZEN";
  state.active_phase = 0;
  state.lambda = 0.1;
  state.epsilon = 1.0e-5;
  state.tau = 1.0;
  state.trust_radius = 1.0;
  state.start_graph_delta = 0.375;
  state.elapsed_seconds = 2.0;
  state.max_objective_evaluations = 100;
  state.total_budget_gradients = 2;
  state.phase_gradients[0] = 2;
  state.phase_seconds[0] = 2.0;
  state.theta = {0.25, -0.5};
  state.theta_start = {0.5, -0.25};
  state.stagnation_best_theta = {0.2, -0.4};
  state.stagnation_gradient_attempts = 23;
  state.stagnation_low_progress_windows = 1;
  state.stagnation_progress_attempts_in_window = 11;
  state.same_point_curvature_rejections = 3;
  state.stagnation_best_kkt = 0.2;
  state.stagnation_window_start_kkt = 0.4;
  state.stagnation_last_window_relative_improvement = 0.25;
  state.coordinate_scale = {1.0, 2.0};
  state.bfgs_inverse = {1.0, 0.0, 0.0, 1.0};
  return state;
}

ControllerJournal valid_journal()
{
  ControllerJournal state;state.job_id="cpu-controller-test";state.identity_sha256=std::string(64,'b');
  state.branch="C1";state.family="M1";state.stage="C";state.next_candidate_index=1;state.active_phase=4;
  state.max_fit_family_gradients=52;state.max_fit_family_wall_seconds=120.0;state.elapsed_seconds=12.0;
  state.active_candidate_started=true;state.active_candidate_elapsed_seconds=6.0;
  state.exact_gradients=4;state.phase_gradients[2]=2;state.phase_gradients[3]=2;
  state.phase_seconds[2]=4.0;state.phase_seconds[3]=8.0;
  state.training_statistics_frozen=true;state.training_statistics_policy_version="RPJASTAT2";
  state.training_frame_count=66;state.development_frame_count=34;state.training_block_length=8;
  state.training_adjacent_block_length=16;state.training_covered_frame_count=64;state.training_discarded_tail_frames=2;
  state.training_channel_names={"FIXED_PHYSICAL_X","PARAMETER_0"};
  state.candidate_cost_samples=1;state.candidate_cost_max_seconds=6.0;state.candidate_cost_partial_seconds=2.0;
  state.common_probe_set_frozen=true;state.common_probe_stage="B_GATE";
  state.common_probe_sources={"FIXED_PHYSICAL_X","LOW_RITZ"};
  state.common_probe_origins={"PHYSICAL","CANDIDATE_SOFT_UNION"};
  state.common_probe_vectors={{1.0,0.0},{0.0,1.0}};
  ControllerCandidate completed;completed.branch="B0";completed.index=0;completed.lambda=0.01;
  completed.score_generation="B_GATE";
  completed.response_replay_required=true;
  completed.fit_state_sha256=std::string(64,'d');
  completed.fit_complete=true;completed.score_complete=true;completed.converged=true;completed.objective=2.5;
  completed.b_gate_pass=true;
  completed.uncertainty=0.25;completed.theta={0.2,0.3};completed.block_scores_b={2.4,2.6};
  completed.block_scores_2b={2.5};state.candidates.push_back(completed);
  ControllerCandidate active;active.branch="C1";active.index=1;active.lambda=0.0001;
  state.candidates.push_back(active);return state;
}

void test_score_generation_controls_selection_eligibility()
{
  ControllerCandidate m0;m0.score_complete=true;m0.response_replay_required=true;
  m0.score_generation="FINAL_M0_M1";m0.development_pass=false;m0.b_gate_pass=true;
  ControllerCandidate m1;m1.score_complete=true;m1.response_replay_required=true;
  m1.score_generation="FINAL_M0_M1";m1.development_pass=true;m1.b_gate_pass=false;
  require(!rpmd_ja_covariance_checkpoint::controller_candidate_pass_for_generation(m0,"FINAL_M0_M1")&&
    rpmd_ja_covariance_checkpoint::controller_candidate_pass_for_generation(m1,"FINAL_M0_M1"),
    "final M0/M1 eligibility must use each candidate's final generation development pass, not its B_GATE pass");
}

void test_c_resume_keeps_b_gate_committed_and_accounts_repeated_interruptions_to_c()
{
  using rpmd_ja_feasibility::Phase;
  const rpmd_ja_feasibility::Limits limits{};
  ControllerJournal saved=valid_journal();saved.stage="C";saved.active_phase=rpmd_ja_feasibility::phase_index(Phase::C);
  saved.elapsed_seconds=130.0;saved.phase_seconds[rpmd_ja_feasibility::phase_index(Phase::B)]=120.0;
  saved.phase_seconds[rpmd_ja_feasibility::phase_index(Phase::C)]=10.0;saved.active_candidate_elapsed_seconds=4.0;
  require(rpmd_ja_covariance_checkpoint::controller_has_committed_b_gate_for_resume(saved,true),
    "C cursor must preserve the already committed B_GATE result");
  saved.operation_in_progress=true;saved.operation_start_epoch_seconds=1000.0;
  ControllerJournal resumed=saved;
  rpmd_ja_covariance_checkpoint::reconcile_controller_operation_budget(resumed,saved,limits,1000.0,1005.0);
  require(!resumed.operation_in_progress&&resumed.phase_seconds[rpmd_ja_feasibility::phase_index(Phase::B)]==120.0&&
    resumed.phase_seconds[rpmd_ja_feasibility::phase_index(Phase::C)]==15.0&&
    resumed.candidates[0].b_gate_pass&&
    rpmd_ja_covariance_checkpoint::controller_has_committed_b_gate_for_resume(resumed,true),
    "first C interruption must charge only C and retain B_GATE eligibility");
  ControllerJournal interrupted_again=resumed;interrupted_again.operation_in_progress=true;
  interrupted_again.operation_start_epoch_seconds=1005.0;
  ControllerJournal resumed_again=interrupted_again;
  rpmd_ja_covariance_checkpoint::reconcile_controller_operation_budget(
    resumed_again,interrupted_again,limits,1005.0,1010.0);
  require(resumed_again.phase_seconds[rpmd_ja_feasibility::phase_index(Phase::B)]==120.0&&
    resumed_again.phase_seconds[rpmd_ja_feasibility::phase_index(Phase::C)]==20.0&&
    resumed_again.elapsed_seconds==140.0&&resumed_again.candidates[0].b_gate_pass,
    "repeated C interruption must not reopen or recharge the exhausted B phase");
}

void test_frozen_selection_journal_round_trips_and_rejects_probe_mutation()
{
  const std::string path="rpmd_ja_frozen_selection_journal_test.checkpoint";
  ControllerJournal frozen=valid_journal();frozen.stage="FREEZE";frozen.branch="B2";frozen.family="M0";
  frozen.next_candidate_index=2;frozen.active_phase=5;frozen.active_candidate_started=false;
  frozen.active_candidate_elapsed_seconds=0.0;frozen.common_probe_stage="FINAL_M0_M1";
  frozen.selection_state="FROZEN";frozen.development_response_sha256=std::string(64,'e');
  ControllerCandidate& selected=frozen.candidates[0];selected.score_generation="FINAL_M0_M1";
  selected.score_complete=true;selected.response_replay_required=true;selected.development_pass=true;
  frozen.frozen_candidate_branch=selected.branch;frozen.frozen_candidate_index=selected.index;
  frozen.frozen_candidate_lambda=selected.lambda;frozen.frozen_candidate_fit_state_sha256=selected.fit_state_sha256;
  frozen.frozen_candidate_identity_sha256=rpmd_ja_covariance_checkpoint::frozen_candidate_identity_sha256(frozen);
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,frozen);
  const ControllerJournal restored=rpmd_ja_covariance_checkpoint::read_controller_journal(path,frozen.identity_sha256);
  require(restored.selection_state=="FROZEN"&&restored.frozen_candidate_branch=="B0"&&
    restored.development_response_sha256==frozen.development_response_sha256&&
    restored.frozen_candidate_identity_sha256==frozen.frozen_candidate_identity_sha256&&
    restored.common_probe_vectors==frozen.common_probe_vectors,
    "RPJAOUTER5 must round-trip frozen selection and its integrity-bound development response");
  ControllerJournal altered=frozen;altered.common_probe_vectors[0][0]+=0.25;
  bool mutation_rejected=false;
  try{rpmd_ja_covariance_checkpoint::validate_controller_journal(altered);}
  catch(const std::invalid_argument&){mutation_rejected=true;}
  require(mutation_rejected,"frozen candidate identity must reject a mutated common probe");

  ControllerJournal opened=frozen;opened.selection_state="BLIND_OPENED";opened.operation_in_progress=true;
  opened.operation_start_epoch_seconds=100.0;
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,opened);
  ControllerJournal blind_passed=opened;blind_passed.selection_state="BLIND_PASSED";
  blind_passed.operation_in_progress=false;blind_passed.operation_start_epoch_seconds=0.0;
  blind_passed.blind_response_sha256=std::string(64,'f');
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,blind_passed);
  ControllerJournal publishing=blind_passed;publishing.selection_state="PUBLISHING";
  publishing.operation_in_progress=true;publishing.operation_start_epoch_seconds=101.0;
  publishing.publication_stage_output_sha256=std::string(64,'1');
  publishing.publication_stage_sidecar_sha256=std::string(64,'2');
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,publishing);
  ControllerJournal published=publishing;published.selection_state="PUBLISHED";
  published.operation_in_progress=false;published.operation_start_epoch_seconds=0.0;
  published.published_output_sha256=published.publication_stage_output_sha256;
  published.published_sidecar_sha256=published.publication_stage_sidecar_sha256;
  published.published_fit_sha256=std::string(64,'3');published.published_trace_sha256=std::string(64,'4');
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,published);
  const ControllerJournal published_round_trip=rpmd_ja_covariance_checkpoint::read_controller_journal(path,published.identity_sha256);
  require(published_round_trip.selection_state=="PUBLISHED"&&
    published_round_trip.published_output_sha256==published.published_output_sha256&&
    published_round_trip.published_trace_sha256==published.published_trace_sha256,
    "RPJAOUTER5 must persist the completed blind and publication transaction identities");
  std::remove(path.c_str());
}

void test_blind_and_publication_resume_keep_in_progress_across_repeated_interruptions()
{
  using rpmd_ja_feasibility::Phase;
  const rpmd_ja_feasibility::Limits limits{};
  auto frozen_state=[](){
    ControllerJournal state=valid_journal();state.stage="FREEZE";state.branch="B2";state.family="M0";
    state.next_candidate_index=2;state.active_phase=rpmd_ja_feasibility::phase_index(Phase::FREEZE);
    state.active_candidate_started=false;state.active_candidate_elapsed_seconds=0.0;
    state.common_probe_stage="FINAL_M0_M1";state.selection_state="FROZEN";
    state.development_response_sha256=std::string(64,'e');
    ControllerCandidate& candidate=state.candidates[0];candidate.score_generation="FINAL_M0_M1";
    candidate.score_complete=true;candidate.response_replay_required=true;candidate.development_pass=true;
    state.frozen_candidate_branch=candidate.branch;state.frozen_candidate_index=candidate.index;
    state.frozen_candidate_lambda=candidate.lambda;state.frozen_candidate_fit_state_sha256=candidate.fit_state_sha256;
    state.frozen_candidate_identity_sha256=rpmd_ja_covariance_checkpoint::frozen_candidate_identity_sha256(state);
    return state;
  };
  auto repeated_resume=[&](ControllerJournal state,const Phase phase,const std::string& selection_state,const char* path){
    state.selection_state=selection_state;state.active_phase=rpmd_ja_feasibility::phase_index(phase);
    state.operation_in_progress=true;state.operation_start_epoch_seconds=1000.0;
    for(unsigned interruption=0;interruption<2;++interruption){
      const ControllerJournal saved=state;
      const double resume_epoch=1005.0+5.0*interruption;
      rpmd_ja_covariance_checkpoint::reconcile_controller_operation_budget(state,saved,limits,100.0,resume_epoch);
      require(state.operation_in_progress&&state.operation_start_epoch_seconds==resume_epoch&&
        state.elapsed_seconds==17.0+5.0*interruption&&
        state.phase_seconds[rpmd_ja_feasibility::phase_index(phase)]==5.0+5.0*interruption,
        "blind/publication recovery must charge the old interval once and keep the resumed operation in progress");
      rpmd_ja_covariance_checkpoint::write_controller_journal(path,state);
      state=rpmd_ja_covariance_checkpoint::read_controller_journal(path,state.identity_sha256);
      require(state.selection_state==selection_state&&state.operation_in_progress,
        "blind/publication journal must remain resumable after each durable checkpoint");
    }
    std::remove(path);
  };
  repeated_resume(frozen_state(),Phase::BLIND,"BLIND_OPENED","rpmd_ja_blind_resume_budget_test.checkpoint");
  ControllerJournal publishing=frozen_state();publishing.selection_state="PUBLISHING";
  publishing.blind_response_sha256=std::string(64,'f');
  publishing.publication_stage_output_sha256=std::string(64,'1');
  publishing.publication_stage_sidecar_sha256=std::string(64,'2');
  repeated_resume(publishing,Phase::PREPARE,"PUBLISHING","rpmd_ja_publication_resume_budget_test.checkpoint");
}

void test_scheduler_freezes_m0_after_skip_or_m1_evaluation()
{
  const std::string path="rpmd_ja_scheduler_freeze_test.checkpoint";
  auto finish=[](ControllerJournal& state,const std::string& selected_branch){
    const unsigned selected_index=static_cast<unsigned>(selected_branch[1]-'0');
    const std::string identity=rpmd_ja_covariance_checkpoint::begin_controller_selection_freeze(state,selected_branch,selected_index);
    require(rpmd_ja_covariance_checkpoint::is_sha256(identity),"scheduler must freeze a candidate identity before the development snapshot");
    rpmd_ja_covariance_checkpoint::commit_controller_selection_freeze(state,std::string(64,'e'));
  };

  ControllerJournal m0_only=valid_journal();m0_only.active_candidate_started=false;
  m0_only.active_candidate_elapsed_seconds=0.0;m0_only.candidates.resize(1);
  m0_only.candidates[0].development_pass=true;
  const auto original_probes=m0_only.common_probe_vectors;
  rpmd_ja_covariance_checkpoint::commit_final_development_generation(m0_only,false,[](){
    throw std::runtime_error("M0-only final generation must not rebuild or rescore the B_GATE probes");});
  finish(m0_only,"B0");
  require(m0_only.stage=="FREEZE"&&m0_only.branch=="B2"&&m0_only.family=="M0"&&
    m0_only.common_probe_stage=="FINAL_M0_M1"&&m0_only.common_probe_vectors==original_probes&&
    m0_only.candidates[0].score_generation=="FINAL_M0_M1",
    "M0-only scheduler must promote the exact B_GATE score and probes to the final generation");
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,m0_only);
  const ControllerJournal restored_m0=rpmd_ja_covariance_checkpoint::read_controller_journal(path,m0_only.identity_sha256);
  require(restored_m0.selection_state=="FROZEN"&&restored_m0.frozen_candidate_branch=="B0"&&
    restored_m0.branch=="B2"&&restored_m0.family=="M0"&&restored_m0.common_probe_vectors==original_probes,
    "M0-only FROZEN journal must restore through its real B_GATE-to-final scheduler transition");

  ControllerJournal m1_tested=valid_journal();m1_tested.active_candidate_started=false;
  m1_tested.active_candidate_elapsed_seconds=0.0;
  ControllerCandidate& m1=m1_tested.candidates[1];m1.branch="C0";m1.index=0;m1.lambda=0.001;
  m1.fit_complete=true;m1.converged=true;m1.fit_state_sha256=std::string(64,'c');m1.theta={0.1,0.2};
  bool final_rescore_called=false;
  rpmd_ja_covariance_checkpoint::commit_final_development_generation(m1_tested,true,[&](){
    final_rescore_called=true;m1_tested.common_probe_stage="FINAL_M0_M1";
    m1_tested.common_probe_sources.push_back("M1_SOFT");m1_tested.common_probe_origins.push_back("CANDIDATE_SOFT_UNION");
    m1_tested.common_probe_vectors.push_back({0.6,0.8});
    for(ControllerCandidate& candidate:m1_tested.candidates)if(candidate.fit_complete){
      candidate.score_complete=true;candidate.response_replay_required=true;candidate.score_generation="FINAL_M0_M1";
      candidate.objective=1.0;candidate.uncertainty=0.1;candidate.development_pass=candidate.branch=="B0";}
  });
  m1_tested.candidates[0].development_pass=true;
  finish(m1_tested,"B0");
  require(final_rescore_called&&m1_tested.stage=="FREEZE"&&m1_tested.branch=="C2"&&m1_tested.family=="M1"&&
    m1_tested.frozen_candidate_branch=="B0"&&m1_tested.common_probe_stage=="FINAL_M0_M1"&&
    m1_tested.common_probe_vectors.size()==3,
    "M1-tested scheduler must retain its C2 cursor while freezing the selected M0 candidate");
  rpmd_ja_covariance_checkpoint::write_controller_journal(path,m1_tested);
  const ControllerJournal restored_m1=rpmd_ja_covariance_checkpoint::read_controller_journal(path,m1_tested.identity_sha256);
  require(restored_m1.selection_state=="FROZEN"&&restored_m1.frozen_candidate_branch=="B0"&&
    restored_m1.branch=="C2"&&restored_m1.family=="M1"&&restored_m1.common_probe_vectors.size()==3,
    "M1-tested/M0-selected FROZEN journal must restore its distinct scheduler cursor and selected family");
  std::remove(path.c_str());
}

std::string read_bytes(const std::string& path)
{
  std::ifstream input(path, std::ios::binary);
  assert(input.good());
  return std::string(std::istreambuf_iterator<char>(input), {});
}

void write_bytes(const std::string& path, const std::string& bytes)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  assert(output.good());
}

std::string rechecksum(const std::string& bytes)
{
  const std::size_t row = bytes.rfind("sha256 ");
  assert(row != std::string::npos);
  const std::string payload = bytes.substr(0, row);
  return payload + "sha256 " + rpmd_ja_covariance_checkpoint::sha256(payload) + "\n";
}

void expect_semantic_rejection(const std::string& path, const std::string& field,
                               const std::string& old_value, const std::string& new_value)
{
  std::string bytes = read_bytes(path);
  const std::string old_row = std::string(field) + " " + old_value;
  const std::size_t row = bytes.find(old_row);
  assert(row != std::string::npos);
  bytes.replace(row, old_row.size(), std::string(field) + " " + new_value);
  write_bytes(path, rechecksum(bytes));
  bool rejected = false;
  try {
    (void)rpmd_ja_covariance_checkpoint::read_checkpoint(path, std::string(64, 'a'));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

void expect_journal_rejection(const std::string& path,const std::string& field,
  const std::string& old_value,const std::string& new_value)
{
  std::string bytes=read_bytes(path);const std::string old_row=field+" "+old_value;
  const std::size_t row=bytes.find(old_row);require(row!=std::string::npos,"journal mutation row missing");
  bytes.replace(row,old_row.size(),field+" "+new_value);write_bytes(path,rechecksum(bytes));bool rejected=false;
  try{(void)rpmd_ja_covariance_checkpoint::read_controller_journal(path,std::string(64,'b'));}
  catch(const std::runtime_error&){rejected=true;}
  require(rejected,"semantically invalid feasibility journal was accepted");
}
}

int main()
{
  test_frozen_selection_journal_round_trips_and_rejects_probe_mutation();
  test_blind_and_publication_resume_keep_in_progress_across_repeated_interruptions();
  test_scheduler_freezes_m0_after_skip_or_m1_evaluation();
  test_score_generation_controls_selection_eligibility();
  test_c_resume_keeps_b_gate_committed_and_accounts_repeated_interruptions_to_c();
  AcceptedPoint elapsed_state=valid_state();
  elapsed_state.operation_start_epoch_seconds=1000.0;
  require(rpmd_ja_covariance_checkpoint::elapsed_budget_on_resume(elapsed_state,100.0,2000.0)==2.0,
    "clean checkpoint resume must exclude process downtime");
  const double clean_resume_startup=rpmd_ja_covariance_checkpoint::resume_startup_elapsed(100.0,100.0,107.0,false);
  require(clean_resume_startup==7.0&&100.0+clean_resume_startup==107.0,
    "clean resume startup validation and checkpoint reconstruction must consume the saved wall budget");
  const double interrupted_resume_tail=
    rpmd_ja_covariance_checkpoint::resume_startup_elapsed(100.0,106.0,107.0,true);
  require(interrupted_resume_tail==1.0&&100.0+6.0+interrupted_resume_tail==107.0,
    "startup after the reconciled in-flight interval must be charged once without recharging the first six seconds");
  elapsed_state.operation_in_progress=true;
  require(rpmd_ja_covariance_checkpoint::elapsed_budget_on_resume(elapsed_state,100.0,1005.0)==7.0,
    "interrupted operation must charge elapsed time without discarding the accepted-point cursor");
  elapsed_state=valid_state();elapsed_state.elapsed_seconds=95.0;elapsed_state.phase_seconds[0]=95.0;
  elapsed_state.operation_start_epoch_seconds=1000.0;
  elapsed_state.operation_in_progress=true;
  require(rpmd_ja_covariance_checkpoint::elapsed_budget_on_resume(elapsed_state,100.0,1010.0)==105.0,
    "in-progress checkpoint time must be charged on top of its accepted-point budget");
  bool invalid_elapsed_rejected=false;
  try{(void)rpmd_ja_covariance_checkpoint::elapsed_budget_on_resume(elapsed_state,1.0,1010.0);}
  catch(const std::invalid_argument&){invalid_elapsed_rejected=true;}
  require(invalid_elapsed_rejected,"checkpoint elapsed time above the wall budget must be rejected");

  ControllerJournal interrupted_controller=valid_journal();
  interrupted_controller.phase_seconds[2]=2.0;interrupted_controller.phase_seconds[3]=2.0;
  interrupted_controller.phase_seconds[4]=8.0;
  AcceptedPoint interrupted_solver=valid_state();interrupted_solver.job_id=interrupted_controller.job_id;
  interrupted_solver.identity_sha256=interrupted_controller.identity_sha256;interrupted_solver.branch="C1";
  interrupted_solver.family="M1";interrupted_solver.candidate_index=1;interrupted_solver.lambda=0.0001;
  interrupted_solver.active_phase=4;interrupted_solver.elapsed_seconds=10.0;
  interrupted_solver.phase_gradients.fill(0);interrupted_solver.phase_gradients[2]=2;interrupted_solver.phase_gradients[3]=2;
  interrupted_solver.total_budget_gradients=4;interrupted_solver.phase_seconds.fill(0.0);
  interrupted_solver.phase_seconds[2]=2.0;interrupted_solver.phase_seconds[3]=2.0;interrupted_solver.phase_seconds[4]=6.0;
  interrupted_solver.operation_in_progress=true;interrupted_solver.operation_start_epoch_seconds=1000.0;
  ControllerJournal restricted_controller=interrupted_controller;AcceptedPoint restricted_solver=interrupted_solver;
  rpmd_ja_feasibility::Limits restricted_limits;restricted_limits.max_exact_gradients=4;
  bool actual_gradient_limit_rejected=false;
  try{rpmd_ja_covariance_checkpoint::reconcile_controller_resume_budget(
      restricted_controller,restricted_solver,restricted_limits,100.0,1010.0);}
  catch(const std::runtime_error&){actual_gradient_limit_rejected=true;}
  require(actual_gradient_limit_rejected,"resume reconciliation must enforce the caller's frozen non-default gradient limit");
  const rpmd_ja_feasibility::Limits default_limits;
  rpmd_ja_covariance_checkpoint::reconcile_controller_resume_budget(
    interrupted_controller,interrupted_solver,default_limits,100.0,1010.0);
  require(interrupted_controller.elapsed_seconds==20.0&&interrupted_controller.phase_seconds[4]==16.0&&
    interrupted_controller.exact_gradients==5&&interrupted_controller.active_candidate_elapsed_seconds==14.0&&
    !interrupted_solver.operation_in_progress&&interrupted_solver.elapsed_seconds==20.0,
    "B/C active-fit recovery must reconcile monotonic wall, phase, gradient, and candidate budgets");
  ControllerJournal interrupted_b=valid_journal();interrupted_b.stage="B";interrupted_b.branch="B1";
  interrupted_b.family="M0";interrupted_b.next_candidate_index=1;interrupted_b.active_phase=3;
  interrupted_b.phase_seconds[2]=2.0;interrupted_b.phase_seconds[3]=10.0;interrupted_b.phase_seconds[4]=0.0;
  interrupted_b.candidates.erase(std::remove_if(interrupted_b.candidates.begin(),interrupted_b.candidates.end(),
    [](const ControllerCandidate& candidate){return candidate.branch=="C1";}),interrupted_b.candidates.end());
  AcceptedPoint interrupted_b_solver=valid_state();interrupted_b_solver.job_id=interrupted_b.job_id;
  interrupted_b_solver.identity_sha256=interrupted_b.identity_sha256;interrupted_b_solver.branch="B1";
  interrupted_b_solver.family="M0";interrupted_b_solver.candidate_index=1;interrupted_b_solver.lambda=0.001;
  interrupted_b_solver.active_phase=3;interrupted_b_solver.elapsed_seconds=10.0;
  interrupted_b_solver.phase_gradients.fill(0);interrupted_b_solver.phase_gradients[2]=2;
  interrupted_b_solver.phase_gradients[3]=2;interrupted_b_solver.total_budget_gradients=4;
  interrupted_b_solver.phase_seconds.fill(0.0);interrupted_b_solver.phase_seconds[2]=2.0;
  interrupted_b_solver.phase_seconds[3]=8.0;interrupted_b_solver.operation_in_progress=true;
  interrupted_b_solver.operation_start_epoch_seconds=2000.0;
  rpmd_ja_covariance_checkpoint::reconcile_controller_resume_budget(
    interrupted_b,interrupted_b_solver,default_limits,100.0,2005.0);
  require(interrupted_b.stage=="B"&&interrupted_b.branch=="B1"&&interrupted_b.elapsed_seconds==15.0&&
    interrupted_b.active_candidate_elapsed_seconds==9.0&&interrupted_b.exact_gradients==5,
    "interrupted B fit must resume its solver cursor and reconcile the B-family budget");
  ControllerJournal interrupted_outer=valid_journal();interrupted_outer.operation_in_progress=true;
  interrupted_outer.operation_start_epoch_seconds=3000.0;
  rpmd_ja_covariance_checkpoint::reconcile_controller_operation_budget(
    interrupted_outer,interrupted_outer,default_limits,100.0,3010.0);
  require(interrupted_outer.elapsed_seconds==22.0&&interrupted_outer.phase_seconds[4]==10.0&&
    interrupted_outer.active_candidate_elapsed_seconds==16.0&&!interrupted_outer.operation_in_progress,
    "in-progress outer operation must charge downtime and preserve cumulative phase totals");
  ControllerJournal overrun_outer=valid_journal();overrun_outer.elapsed_seconds=95.0;
  overrun_outer.phase_seconds.fill(0.0);overrun_outer.phase_seconds[2]=2.0;
  overrun_outer.phase_seconds[3]=2.0;overrun_outer.phase_seconds[4]=91.0;
  overrun_outer.active_candidate_elapsed_seconds=5.0;overrun_outer.operation_in_progress=true;
  overrun_outer.operation_start_epoch_seconds=4000.0;const ControllerJournal saved_overrun=overrun_outer;
  bool outer_wall_overrun_rejected=false;
  try{rpmd_ja_covariance_checkpoint::reconcile_controller_operation_budget(
      overrun_outer,saved_overrun,default_limits,100.0,4010.0);}
  catch(const std::runtime_error&){outer_wall_overrun_rejected=true;}
  require(outer_wall_overrun_rejected&&overrun_outer.elapsed_seconds==95.0&&overrun_outer.phase_seconds[4]==91.0,
    "outer reconstruction beyond wall budget must stop before creating an invalid checkpoint state");
  AcceptedPoint wrong_cursor=interrupted_solver;wrong_cursor.branch="B2";bool cursor_rejected=false;
  try{rpmd_ja_covariance_checkpoint::reconcile_controller_resume_budget(
      interrupted_controller,wrong_cursor,default_limits,100.0,1011.0);}
  catch(const std::runtime_error&){cursor_rejected=true;}
  require(cursor_rejected,"active solver state with a different candidate cursor must be rejected");

  const std::string path = "rpmd_ja_covariance_checkpoint_cpu_test.tmp";
  rpmd_ja_covariance_checkpoint::write_checkpoint(path, valid_state());
  AcceptedPoint in_progress=valid_state();in_progress.operation_in_progress=true;
  in_progress.performance_window_initialized=true;in_progress.performance_window_curvature_builds=3;
  in_progress.performance_window_tau=1.0;in_progress.performance_window_start_objective=10.0;
  in_progress.performance_window_curvature_seconds=1.2;in_progress.performance_window_reference_gradient_seconds=0.25;
  in_progress.objective_evaluation_seconds=2.0;in_progress.full_gradient_seconds=0.5;
  in_progress.last_full_gradient_seconds=0.2;in_progress.factorization_seconds=0.3;
  in_progress.analytic_curvature_seconds=1.2;
  rpmd_ja_covariance_checkpoint::write_checkpoint(path,in_progress);
  require(rpmd_ja_covariance_checkpoint::read_checkpoint(path,std::string(64,'a')).operation_in_progress,
    "operation-in-progress checkpoint state must round trip");
  const AcceptedPoint restored_point=rpmd_ja_covariance_checkpoint::read_checkpoint(path,std::string(64,'a'));
  require(restored_point.theta_start==std::vector<double>({0.5,-0.25})&&
    restored_point.stagnation_best_theta==std::vector<double>({0.2,-0.4})&&
    restored_point.stagnation_progress_attempts_in_window==11&&restored_point.same_point_curvature_rejections==3,
    "original fit start and partial stagnation window must survive a midpoint checkpoint");
  require(restored_point.start_graph_delta==0.375,"checkpoint must retain selected initialization displacement");
  require(restored_point.performance_window_initialized&&restored_point.performance_window_curvature_builds==3&&
    restored_point.performance_window_tau==1.0&&restored_point.performance_window_start_objective==10.0&&
    restored_point.performance_window_curvature_seconds==1.2&&restored_point.performance_window_reference_gradient_seconds==0.25&&
    restored_point.objective_evaluation_seconds==2.0&&restored_point.full_gradient_seconds==0.5&&
    restored_point.last_full_gradient_seconds==0.2&&restored_point.factorization_seconds==0.3&&
    restored_point.analytic_curvature_seconds==1.2,
    "performance-stop window and cumulative fit component timings must survive a midpoint checkpoint");
  rpmd_ja_feasibility::CurvaturePerformanceWindow restored_performance_window;
  restored_performance_window.initialized=restored_point.performance_window_initialized;
  restored_performance_window.curvature_builds=restored_point.performance_window_curvature_builds;
  restored_performance_window.tau=restored_point.performance_window_tau;
  restored_performance_window.start_objective=restored_point.performance_window_start_objective;
  restored_performance_window.curvature_seconds=restored_point.performance_window_curvature_seconds;
  restored_performance_window.reference_gradient_seconds=restored_point.performance_window_reference_gradient_seconds;
  require(rpmd_ja_feasibility::finish_curvature_performance_window(restored_performance_window,1.0,10.0)==
    rpmd_ja_feasibility::CurvaturePerformanceDecision::NUMERICAL_INCONCLUSIVE,
    "resuming a completed same-tau window must preserve its stop decision");
  AcceptedPoint replacement = valid_state();
  replacement.theta[0] = 0.75;
  rpmd_ja_covariance_checkpoint::write_checkpoint(path, replacement);
  assert(rpmd_ja_covariance_checkpoint::read_checkpoint(path, std::string(64, 'a')).theta[0] == 0.75);

  expect_semantic_rejection(path, "coordinate_scale", "2 1 2", "2 0 2");
  rpmd_ja_covariance_checkpoint::write_checkpoint(path, replacement);
  expect_semantic_rejection(path, "active_phase", "0", "8");
  rpmd_ja_covariance_checkpoint::write_checkpoint(path, replacement);
  expect_semantic_rejection(path, "total_budget_gradients", "2", "1");
  rpmd_ja_covariance_checkpoint::write_checkpoint(path, replacement);
  expect_semantic_rejection(path, "selection_state", "\"NOT_FROZEN\"", "\"FROZEN\"");

  rpmd_ja_covariance_checkpoint::write_checkpoint(path, replacement);
  std::string truncated = read_bytes(path);
  truncated.resize(truncated.size() - 12);
  write_bytes(path, truncated);
  bool truncation_rejected = false;
  try {
    (void)rpmd_ja_covariance_checkpoint::read_checkpoint(path, std::string(64, 'a'));
  } catch (const std::runtime_error&) {
    truncation_rejected = true;
  }
  assert(truncation_rejected);
  std::remove(path.c_str());
  const std::string journal_path="rpmd_ja_feasibility_controller_journal_cpu_test.tmp";
  const ControllerJournal journal=valid_journal();
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,journal);
  const ControllerJournal restored=rpmd_ja_covariance_checkpoint::read_controller_journal(journal_path,std::string(64,'b'));
  require(restored.stage=="C"&&restored.branch=="C1"&&restored.next_candidate_index==1,
    "C candidate cursor must round trip for resume");
  require(restored.candidates.size()==2&&restored.candidates[0].score_complete&&
    restored.candidates[0].block_scores_2b==std::vector<double>{2.5}&&
    restored.candidates[0].score_generation=="B_GATE"&&restored.candidates[0].response_replay_required,
    "score summaries must be restored with an explicit response-matrix replay requirement");
  require(restored.common_probe_set_frozen&&restored.common_probe_vectors.size()==2&&
    restored.common_probe_vectors[1][1]==1.0,"frozen common probes must be restored without reselection");
  require(restored.training_statistics_frozen&&restored.training_block_length==8&&
    restored.training_adjacent_block_length==16&&restored.training_channel_names.size()==2,
    "training admission block and channel plan must be restored without reselection");
  require(rpmd_ja_covariance_checkpoint::preserve_later_family_cursor(restored,true,"C","M1",0)&&
    !rpmd_ja_covariance_checkpoint::preserve_later_family_cursor(restored,true,"C","M1",1),
    "replaying C0 must preserve an active C1 cursor but not consume the active index");
  ControllerJournal b_interrupted=journal;b_interrupted.stage="B";b_interrupted.branch="B1";
  b_interrupted.family="M0";b_interrupted.next_candidate_index=1;b_interrupted.active_phase=3;
  b_interrupted.cg_witness_attempts=7;
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,b_interrupted);
  const ControllerJournal restored_b=rpmd_ja_covariance_checkpoint::read_controller_journal(journal_path,std::string(64,'b'));
  require(restored_b.stage=="B"&&restored_b.branch=="B1"&&restored_b.candidates[0].score_generation=="B_GATE",
    "interrupted B candidate must retain the completed B_GATE score and cursor");
  require(restored_b.cg_witness_attempts==7,"witness attempt sequence must survive controller journal restoration");
  ControllerJournal next_witness=restored_b;
  const std::uint64_t witness_attempt=rpmd_ja_covariance_checkpoint::next_cg_witness_attempt(next_witness);
  require(witness_attempt==8&&rpmd_ja_feasibility::feasibility_cg_witness_path(
    "witness","COMMON_PROBE_RESCORING","M0_B0","B_GATE",2,witness_attempt)!=
    rpmd_ja_feasibility::feasibility_cg_witness_path(
    "witness","COMMON_PROBE_RESCORING","M0_B0","B_GATE",2,7),
    "resumed repeated B_GATE failures must use a new witness filename");
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,next_witness);
  require(rpmd_ja_covariance_checkpoint::read_controller_journal(journal_path,std::string(64,'b')).cg_witness_attempts==8,
    "next witness attempt must be durable before evidence is written");
  ControllerJournal b2_cursor=restored_b;b2_cursor.stage="FREEZE";b2_cursor.branch="B2";
  b2_cursor.family="M0";b2_cursor.next_candidate_index=2;b2_cursor.active_candidate_started=false;
  b2_cursor.active_candidate_elapsed_seconds=0.0;
  b2_cursor.candidates.erase(std::remove_if(b2_cursor.candidates.begin(),b2_cursor.candidates.end(),
    [](const ControllerCandidate& candidate){return candidate.branch=="C1";}),b2_cursor.candidates.end());
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,b2_cursor);
  const ControllerJournal restored_b2=rpmd_ja_covariance_checkpoint::read_controller_journal(journal_path,std::string(64,'b'));
  require(restored_b2.stage=="FREEZE"&&restored_b2.branch=="B2"&&restored_b2.next_candidate_index==2&&
    restored_b2.candidates[0].score_complete&&restored_b2.common_probe_set_frozen&&restored_b2.exact_gradients==journal.exact_gradients,
    "B2 interruption must restore completed scores, frozen B_GATE probes, and cumulative budget before C dispatch");
  require(rpmd_ja_covariance_checkpoint::preserve_later_family_cursor(restored_b,true,"B","M0",0)&&
    !rpmd_ja_covariance_checkpoint::preserve_later_family_cursor(restored_b,true,"B","M0",1),
    "replaying B0 must preserve an active B1 cursor but not consume the active index");
  ControllerJournal final_scored=journal;final_scored.stage="FREEZE";final_scored.branch="C2";
  final_scored.next_candidate_index=2;final_scored.active_candidate_started=false;
  final_scored.active_candidate_elapsed_seconds=0.0;final_scored.common_probe_stage="FINAL_M0_M1";
  final_scored.candidates[0].score_generation="FINAL_M0_M1";
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,final_scored);
  const ControllerJournal restored_final=rpmd_ja_covariance_checkpoint::read_controller_journal(journal_path,std::string(64,'b'));
  require(restored_final.stage=="FREEZE"&&restored_final.common_probe_stage=="FINAL_M0_M1"&&
    restored_final.candidates[0].score_generation=="FINAL_M0_M1",
    "interrupted final rescoring must retain its completed frozen-probe score generation");
  auto rejects_journal=[&](ControllerJournal altered,const char* message){bool rejected=false;
    try{rpmd_ja_covariance_checkpoint::validate_controller_journal(altered);}
    catch(const std::invalid_argument&){rejected=true;}require(rejected,message);};
  ControllerJournal over_budget=journal;over_budget.elapsed_seconds=28801.0;
  rejects_journal(over_budget,"rechecksummed outer journal must reject elapsed time above the frozen wall budget");
  over_budget=journal;over_budget.exact_gradients=241;over_budget.phase_gradients[3]=239;
  rejects_journal(over_budget,"rechecksummed outer journal must reject gradients above the frozen limit");
  over_budget=journal;over_budget.max_fit_family_gradients=113;
  rejects_journal(over_budget,"outer journal must reject an increased B/C gradient cap");
  over_budget=journal;over_budget.max_fit_family_wall_seconds=10801.0;
  rejects_journal(over_budget,"outer journal must reject an increased B/C wall cap");
  bool identity_rejected=false;
  try{(void)rpmd_ja_covariance_checkpoint::read_controller_journal(journal_path,std::string(64,'c'));}
  catch(const std::runtime_error&){identity_rejected=true;}
  require(identity_rejected,"journal from another manifest identity must be rejected");
  expect_journal_rejection(journal_path,"exact_gradients","4","3");
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,journal);
  expect_journal_rejection(journal_path,"training_block_length","8","9");
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,journal);
  expect_journal_rejection(journal_path,"next_candidate_index","1","2");
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,journal);
  expect_journal_rejection(journal_path,"selection_state","\"NOT_FROZEN\"","\"BLIND_OPENED\"");
  rpmd_ja_covariance_checkpoint::write_controller_journal(journal_path,journal);
  expect_journal_rejection(journal_path,"cg_witness_attempts","0","-1");
  ControllerJournal exhausted_witness_counter=journal;
  exhausted_witness_counter.cg_witness_attempts=std::numeric_limits<std::uint64_t>::max();
  bool witness_counter_overflow_rejected=false;
  try{(void)rpmd_ja_covariance_checkpoint::next_cg_witness_attempt(exhausted_witness_counter);}
  catch(const std::overflow_error&){witness_counter_overflow_rejected=true;}
  require(witness_counter_overflow_rejected,"witness attempt counter overflow must fail closed");
  std::remove(journal_path.c_str());
  return 0;
}
