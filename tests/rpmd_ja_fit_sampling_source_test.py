from pathlib import Path
from math import isclose


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


run = source("src/main_gpumd/run.cu")
measure = source("src/measure/measure.cu")
sampler = source("src/measure/rpmd_ja_fit.cu")
sampler_header = source("src/measure/rpmd_ja_fit.cuh")
sampling = source("src/measure/rpmd_ja_fit_sampling.cuh")
native_header = source("src/measure/rpmd_ja_native_fit.cuh")
native_fit = source("src/measure/rpmd_ja_native_fit.cu")
sampler_fit = source("src/measure/rpmd_ja_fit.cu")

assert 'if (tokens[1] == "fit")' in run
assert "tokens, number_of_types, integrate, group, atom, box, force, first_potential_filename_" in run
assert "rpmd_ja fit requires a fixed integration time step" in run
assert "rpmd_ja fit currently supports exactly one qNEP potential in charge mode 1 or 2 with PPPM" in run
assert 'tokens[1] == "fit"' in measure and "new RpmdJA_Fit(tokens)" in measure
assert 'if (tokens[1] == "sample")' in run and 'tokens[1] == "sample"' in measure
assert "sample_only_" in sampler_fit and "sample spool retained" in sampler_fit
assert "if (sample_only_ && frame_count < 3)" in sampler_fit
assert "if (sample_only_ && frame_count_ < 3)" in sampler_fit
assert 'tokens[1] == "sample"' in sampler_fit and 'tokens.size() != 4' in sampler_fit
assert "candidate_identity" in native_fit and "sample_IBP_status" in native_fit
assert "candidate_parameters cutoff_A=" in native_fit and 'write_bootstrap_diagnostic(trace,"sample_IBP_DIAGNOSTIC"' in native_fit
assert "ibp_statistics_numerically_invalid" in native_fit
assert 'sample_statistics.ibp_band.status!="IBP_PASS"' not in native_fit
assert "response.ibp<=options.ibp_tolerance" not in native_fit
assert 'if(ibp_statistics_numerically_invalid(sample_statistics.ibp_band))' in native_fit
assert 'if(response.response>options.response_tolerance||response.response_band.status!="RESPONSE_PASS")' in native_fit
assert 'FIT_STATUS ACCEPTED_REFERENCE response=PASS full_spd=PASS sample_IBP_DIAGNOSTIC=' in native_fit
assert 'response_IBP_DIAGNOSTIC=' in native_fit and 'IBP_acceptance_role DIAGNOSTIC_ONLY' in native_fit
fit_impl_start = native_fit.index("static void fit_rpmd_ja_native_reference_impl(")
fit_impl = native_fit[fit_impl_start:]
assert fit_impl.index("collect_fixed_probe_statistics(in,header") < fit_impl.index("std::ofstream trace(trace_path")
assert fit_impl.index("std::ofstream trace(trace_path") < fit_impl.index("generate_rpmd_ja_qnep_raw(")
assert "PREFLIGHT_NUMERICAL_FAILURE sample_IBP_status" in fit_impl
assert "type-local seed vectors are translation-projected and orthogonalized" in native_fit
assert "SOURCE_ATOM_MOTION" in native_fit and "maximum_adjacent_sampled_frame_displacement" in native_fit
assert "write_probe_origin_table(out,response.probe_sources,response.probe_origins)" in native_fit
assert "write_probe_origin_table(fitout,response.probe_sources,response.probe_origins)" in native_fit
assert "test_type_local_source_and_motion_diagnostics" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert 'if (tokens[1] == "check_samples")' in run
assert 'check_rpmd_ja_native_fit_samples(tokens[2], tokens[3], atom, box)' in run
assert "void check_rpmd_ja_native_fit_samples(" in native_header
check_samples_start = native_fit.index("void check_rpmd_ja_native_fit_samples(")
check_samples_end = native_fit.index("\n}\n\nvoid replay_rpmd_ja_native_cg", check_samples_start)
check_samples_body = native_fit[check_samples_start:check_samples_end]
assert "read_header(in,0,atom,box,0.0,true)" in check_samples_body
assert "read_training_r0" in check_samples_body and "collect_fixed_probe_statistics" in check_samples_body
assert "train,r0,true,0.15,&box" in check_samples_body
assert "collect_fixed_probe_statistics" in check_samples_body and "format_fixed_probe_report" in check_samples_body
assert "qraw" not in check_samples_body and "make_graph" not in check_samples_body and "lanczos" not in check_samples_body
replay_body = native_fit[native_fit.index("void replay_rpmd_ja_native_cg("):native_fit.index("#else", native_fit.index("void replay_rpmd_ja_native_cg("))]
assert replay_body.index("std::ifstream existing_report") < replay_body.index("read_cg_witness_paths")
assert "validate_qraw_identity" in replay_body and "compute_cg_true_residual_compensated" in native_fit
assert "fast_true_relative_residual" in replay_body and "compensated_true_relative_residual" in replay_body
assert "NOT_ACCEPTED_REFERENCE" in replay_body and "write_text_exclusive" in replay_body
assert "solve_projected_cg" in replay_body and "make_graph(r0,header.types,box,snapshot.cutoff)" in replay_body
assert "replay_cg" in run and "replay_rpmd_ja_native_cg" in run
parser_body = native_fit[native_fit.index("CGWitnessSnapshot read_cg_witness("):native_fit.index("bool preserve_candidate_package_on_full_spd_failure(")]
vector_body = parser_body[parser_body.index('if(key=="theta"||'):parser_body.index('}else if(key=="status")')]
assert "vector has trailing data" in vector_body and "continue;" in vector_body
assert "test_cg_witness_vector_line_endings();" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "cannot replace final response validation" in native_fit
assert "ProbeMoments" in native_fit and "moments.ibp_matrix" in native_fit
assert "sum_q,sum_f,sum_f2,sum_qq,sum_fq" in native_fit
assert "select_ibp_noise_channels" in native_fit and "IBP_GAUSSIAN_VARIANCE_PROXY" in native_fit
assert "IBP_FIXED_CHANNEL_VALIDATION" in native_fit and "ADJACENT_SEGMENT_INTERNAL_CHECK" in native_fit
assert "sum_f2[i]+=f[i]*f[i]" in native_fit and "sum_f2[i]+=other.sum_f2[i]" in native_fit
assert "if(collect_lags&&frame>=train)" in native_fit and "validation_pending_frames==result.validation_base_block_length" in native_fit
assert "diagnose_fixed_ibp_channels(result.segments[1],result.validation_blocks" in native_fit
assert "signed_channel_interval(channel.validation_entry,fixed.band.radius)" in native_fit
assert "tail_probe_index_space LOCAL_SELECTED_PROBES" in native_fit
assert "tail_probe_original_indices" in native_fit
assert "tail_probe_scope ALL_FQ_AND_QQ_PRODUCTS_IN_SELECTED_PROBE_SUBSPACE" in native_fit
assert "moments.covariance_fq()" in native_fit and "lag_autocorrelation lags=1,2,5,10" in native_fit
assert "std::vector<std::vector<double>> history" in native_fit and "history.size()>10" in native_fit
assert "if(collect_lags){result.lags[0].add(qp)" in native_fit
assert "collect_fixed_probe_statistics(in,header,frame_count,static_cast<std::uint64_t>(train),r0,false,options.ibp_tolerance,&box)" in native_fit
assert 'if(ibp_statistics_numerically_invalid(sample_statistics.ibp_band))' in native_fit
assert "bootstrap_metric_band" in native_fit and "nonsymmetric_matrix_distance" in native_fit
assert "std::vector<int> bootstrap_block_factors(const std::size_t block_count)" in native_fit
assert "const std::vector<int> factors=bootstrap_block_factors(blocks.size());" in native_fit
assert "for(const int factor:bootstrap_block_factors(blocks.size()))" in native_fit
assert "mean_f[i]*moment.sum_q[j]" in native_fit and "mean_q[j]*moment.sum_f[i]" in native_fit
assert "blocks[(start+k)%count]" in native_fit and "result.frames=covered.frames" in native_fit
assert "if(!block_product_tail_covered(blocks,covered,&result,check_fq,check_qq))return result;" in native_fit
assert 'if(std::strcmp(reason,"NONFINITE_PRODUCT_STATISTICS")==0)diagnostic->status="NUMERICAL_FAILURE";' in native_fit
assert 'const bool check_fq=true,const bool check_qq=true' in native_fit
assert '"RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE",false,true' in native_fit
assert '"IBP_PASS","IBP_FAIL","IBP_INCONCLUSIVE",true,true' in native_fit
assert 'result.diagnostic_stage="TAIL_CHECK"' in native_fit
assert 'result.diagnostic_stage="BOOTSTRAP";result.bootstrap_started=true;' in native_fit
assert 'write_bootstrap_diagnostic(out,"IBP_DIAGNOSTIC",stats.ibp_band)' in native_fit
assert 'write_bootstrap_diagnostic(out,"RESPONSE_DIAGNOSTIC",response.response_band)' in native_fit
assert 'write_bootstrap_diagnostic(out,"response_diagnostic",r.response_band,false)' in native_fit
assert 'std::printf("sampling evidence: %s; report=%s' in native_fit
assert 'response.response<=options.response_tolerance&&response.response_band.status=="RESPONSE_PASS"' in native_fit
assert 'response.ibp_band.status=="IBP_PASS"' not in native_fit
assert "collect_fixed_probe_statistics(in,header,header.frame_count,train,r0,true,0.15,&box)" in native_fit
assert "probe_sample_status" in native_fit and "INSUFFICIENT_SAMPLES" in native_fit
assert "collect_fixed_probe_statistics(in,header,frame_count" in native_fit
assert 'response_state_path=options.output_path+".response_state.txt"' in native_fit
assert "write_response_state(response_state_path,response,theta" in native_fit
assert "response_probe_basis_coefficients(invroot,re.vectors,m,worst_response)" in native_fit
assert '"IBP_residual_nonsymmetric"' in native_fit and '"whitened_response_difference"' in native_fit
assert "worst_IBP_right_singular_probe_basis_coefficients (from eigenvector of E^T E)" in native_fit
assert 'std::fopen(path.c_str(),"wx")' in native_fit

force_call = run.index("compute_force();", run.index("for (int step = 0; step < number_of_steps"))
sample_call = run.index("measure.post_force(", force_call)
integrate2_call = run.index("integrate.compute2(", sample_call)
assert force_call < sample_call < integrate2_call
assert "if ((step + 1) % sample_interval_ != 0) return;" in sampler
post_force_start = sampler.index("void RpmdJA_Fit::post_force(")
post_run_start = sampler.index("void RpmdJA_Fit::post_run(", post_force_start)
post_force = sampler[post_force_start:post_run_start]
assert post_force.index("if ((step + 1) % sample_interval_ != 0) return;") < post_force.index("rpmd_ja_fit_sampling_kernel<<<")
assert "sample_output_gpu_.copy_to_host(sample_output_.data());" in post_force
assert post_force.count("copy_to_host(") == 1
assert "position != bead_position_ptrs_[bead] || force != bead_force_ptrs_[bead]" in post_force
assert "if (refresh_pointers)" in post_force
assert "sample_output_.resize(static_cast<std::size_t>(10) * number_of_atoms_);" in sampler
assert "coordinate_scale = std::max(coordinate_scale, sample_output_[6 * atom_count + atom_id]);" in post_force
assert "const double ring_tolerance = 128.0 * std::numeric_limits<double>::epsilon()" in post_force
assert "coordinate_scale > 1.0e12 * box_scale" in post_force
assert "previous_centroid_ = centroid_;" in post_force
assert "write_or_throw(spool_, frame_buffer_.data(), frame_buffer_.size() * sizeof(double));" in post_force
assert "bead_position_." not in sampler and "bead_force_." not in sampler

assert "'G', 'P', 'J', 'A', 'S', 'M', 'P', '1'" in sampler
assert "ring path with nonzero periodic winding" in sampler
assert "ring-unwrapped centroid differs from the bead-0 MIC centroid" in sampler
assert "previous_centroid_ = centroid_;" in sampler
assert "atom.number_of_atoms > INT_MAX / 3" in sampler
assert "pbc_[0] != 1 || pbc_[1] != 1 || pbc_[2] != 1" in sampler
assert "size_stream.close();" in sampler
assert "stream limits" in sampler
assert "fit_rpmd_ja_native_reference(options, spool_path_, frame_count_, atom, box, *force_)" in sampler
assert "struct RpmdJANativeFitOptions" in native_header
assert "std::uint64_t frame_count" in native_header
assert "int max_stability_rounds = 160;" in native_header
assert "int max_stability_rounds_ = 160;" in sampler_header
assert "tokens.size() != 9 && tokens.size() != 11" in sampler
assert 'tokens[9] != "max_rounds"' in sampler
assert "errno = 0;" in sampler and "errno == ERANGE" in sampler and "rounds > INT_MAX" in sampler
assert "end == tokens[10].c_str() || *end != '\\0' || rounds <= 0" in sampler
assert "max_stability_rounds_ = static_cast<int>(rounds);" in sampler
assert "options.max_stability_rounds = max_stability_rounds_;" in sampler_fit
assert 'if (tokens[1] == "fit_samples")' in run
assert "fit_rpmd_ja_native_reference_from_samples(options, tokens[2], atom, box, force)" in run
fit_samples_start = run.index('if (tokens[1] == "fit_samples")')
fit_samples_end = run.index('if (tokens[1] == "prepare")', fit_samples_start)
fit_samples_parser = run[fit_samples_start:fit_samples_end]
assert 'tokens[argument_count - 2] == "max_rounds"' in fit_samples_parser
assert "argument_count != 9 && argument_count != 10" in fit_samples_parser
assert "options.max_stability_rounds = max_stability_rounds;" in fit_samples_parser
assert "if (argument_count == 10) options.raw_input_path = tokens[9];" in fit_samples_parser
assert "errno = 0;" in fit_samples_parser and "errno == ERANGE" in fit_samples_parser and "rounds > INT_MAX" in fit_samples_parser
assert "end == tokens.back().c_str() || *end != '\\0' || rounds <= 0" in fit_samples_parser
assert "max_rounds <N>" in fit_samples_parser and "[<qraw>] [max_rounds <N>]" in run
assert "derived.temperature=header.temperature" in native_fit and "derived.sample_interval=static_cast<int>(std::llround(sample_interval))" in native_fit
assert "fit_rpmd_ja_native_reference_checked(derived,spool_path,header.frame_count,atom,box,force,true)" in native_fit
assert "validate_qraw_identity(raw,options,header,r0,box,force)" in native_fit
assert "model!=expected_model||config!=expected_config" in native_fit
assert "raw_types[i]==header.types[i]&&raw_masses[i]==header.masses[i]" in native_fit
assert "raw_positions[i]-r0[i]" in native_fit
assert "generate_raw&&std::remove(generated_raw_path.c_str())" in native_fit
assert "options.raw_input_path.empty()?generated_raw_path:options.raw_input_path" in native_fit
assert '#include "force/nep_charge.cuh"' in native_fit
assert 'record_search("FIXED_INDEPENDENT"' in native_fit
assert 'lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,steps,4,options.epsilon,initial_vector,&lanczos_workspace)' in native_fit
assert 'lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,steps,4,options.epsilon,{},&lanczos_workspace)' in native_fit
assert "write_response_stats(trace,response_progress)" in native_fit
assert "cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,d,columns,d" in native_fit
assert "baseline.apply_batch(q_batch,columns,base_batch)" in native_fit
assert "baseline_apply_seconds=" in native_fit
baseline_init_start = native_fit.index("void initialize(std::istream& raw")
baseline_apply_start = native_fit.index("void apply(const std::vector<double>& input", baseline_init_start)
baseline_init_body = native_fit[baseline_init_start:baseline_apply_start]
assert "const std::size_t dim_size=static_cast<std::size_t>(d)" in baseline_init_body
assert "const std::size_t dimension=static_cast<std::size_t>(d)" not in baseline_init_body
assert "test_nonfinite_tail_product_is_numerical_failure" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "columns=32" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "validation=33,frames=train+validation" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
validation_start = native_fit.index("ResponseCheck validate_probes(")
validation_end = native_fit.index("void write_response_state(", validation_start)
validation_body = native_fit[validation_start:validation_end]
assert "std::vector<double> qp(m),fp(m)" not in validation_body
assert "baseline.apply_batch(q_batch,columns,base_batch)" in validation_body
assert "validate_fit_frame_branch(x,n,*branch_check_box,r0)" in native_fit
assert "options.ibp_tolerance,&box" in native_fit
assert "observed_cov_condition" in native_fit and "predicted_cov_condition" in native_fit
assert "candidate_direction_original_mass_weighted" in source("src/measure/rpmd_ja_qnep_prepare.cu")
assert "pack_owned=false;return true;" in native_fit and "no automatic stability cut was added" in native_fit
assert 'preserve_candidate_package_on_full_spd_failure(e.what(),own_scratch.pack_owned)' in native_fit
assert 'qNEP translation-complement Hessian is not positive definite' in source("src/measure/rpmd_ja_qnep_prepare.cu")
assert 'qNEP translation-complement Hessian is not positive definite' in native_fit
assert 'std::sort(modes.begin(),modes.end()' in native_fit
assert 'low_mode_probes<4' in native_fit and 'if(add(soft[i].vector,"LOW_RITZ",-1))++low_mode_probes' in native_fit
assert 'observed_cov_status=' in native_fit and 'predicted_cov_status=' in native_fit
assert 'response_ibp_status=' in native_fit and 'NOT_COMPUTED' in native_fit
spectral_start = native_fit.index("std::vector<double> spectral_row(")
spectral_end = native_fit.index("void apply_total(", spectral_start)
assert "make_design(" not in native_fit[spectral_start:spectral_end]
assert "block_variance_relative_delta" in native_fit
assert "stability_cut_guard_fraction=1.0e-3" in native_fit
assert "cut_target=options.epsilon+cut_guard" in native_fit
assert "const double rhs=cut_target-base_value" in native_fit
assert "used==32||used==64||used==128||used==256" in native_fit
assert "const bool early_cut=steps<=96&&below_threshold&&!search_finished" in native_fit
assert 'steps<=96?"FAST_CUT":"FULL_SEARCH"' in native_fit
assert "actual_rayleigh" in native_fit and "curvature_scale" in native_fit and "dense_matrix_actions" in native_fit and "early_exit" in native_fit
extract_start = native_fit.index("auto extract_modes=")
extract_end = native_fit.index("for(int k=0;k<steps;++k)", extract_start)
assert "r.dense_matrix_actions=dense_matrix_actions" not in native_fit[extract_start:extract_end]
assert "for(auto& mode:out)mode.dense_matrix_actions=dense_matrix_actions" in native_fit[extract_start:extract_end]
inspect_start = native_fit.index("auto inspect_modes=")
inspect_end = native_fit.index("bool observed_violation=false", inspect_start)
assert "spectral_row(" not in native_fit[inspect_start:inspect_end]
assert "mode.curvature_tolerance" in native_fit[inspect_start:inspect_end]
assert "stability_cut_is_violated" in native_fit
assert "test_stability_cut_guard_reduces_rotating_mode_rounds" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_lanczos_staged_early_exit" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_lanczos_continues_across_checkpoints" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_lanczos_curvature_rounding_boundary" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_lanczos_deep_search_does_not_early_cut" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "double compensated_dot(const std::vector<double>& left,const std::vector<double>& right)" in native_fit
cg_start = native_fit.index("CGResult solve_projected_cg(")
cg_end = native_fit.index("struct ProbeOrigin", cg_start)
cg_body = native_fit[cg_start:cg_end]
assert "rr=compensated_dot(r,r)" in cg_body and "pap=compensated_dot(p,ap)" in cg_body
assert "std::fma(alpha,p[i],out.x[i])" in cg_body and "std::fma(-alpha,ap[i],r[i])" in cg_body and "std::fma(beta,p[i],r[i])" in cg_body
assert "TRUE_RESIDUAL_FAILURE" in native_fit and "set_true_residual_failure" in cg_body
assert "write_cg_witness(witness_path,response.witness,theta,raw_path,spool_path,cg_replay)" in native_fit
assert "rhs_input_xyz_soa" in native_fit and "rhs_projected_xyz_soa" in native_fit and "solution_x_xyz_soa" in native_fit and "active_config_fingerprint" in native_fit
assert 'true_residual_status "<<(w.true_residual_computed?"COMPUTED":"NOT_COMPUTED")' in native_fit
assert 'cg_true_residual_status="<<(true_residual_computed?"COMPUTED":"NOT_COMPUTED")' in native_fit
assert "CGReplayRefinement refine_cg_replay_solution(" in native_fit
refinement_start = native_fit.index("CGReplayRefinement refine_cg_replay_solution(")
refinement_body = native_fit[refinement_start:native_fit.index("struct ProbeOrigin", refinement_start)]
refinement_signature = native_fit[refinement_start:native_fit.index("{", refinement_start)]
assert "result.residuals.compensated" in refinement_body and "compare_cg_true_residuals" in refinement_body
assert "max_rounds=3" in refinement_signature and "max_dense_matrix_actions=4096" in refinement_signature
assert "record.iterations=max_iterations" not in refinement_body
assert "solver_counts_available=false" in native_fit
assert "write_cg_replay_refinement_round" in native_fit
assert "test_bounded_cg_replay_refinement" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert '"NO_RESIDUAL_DECREASE"' in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert '"MATRIX_ACTION_LIMIT"' in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert '"iterations NOT_AVAILABLE"' in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "status NOT_ACCEPTED_REFERENCE" in native_fit[native_fit.index("void replay_rpmd_ja_native_cg("):native_fit.index("#else", native_fit.index("void replay_rpmd_ja_native_cg("))]
assert "status CG_FAILURE classification=" in native_fit and "test_cg_failure_snapshot_round_trip_replay" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
cg_replay_test = source("tests/rpmd_ja_native_fit_cuda_test.cu").split("void test_cg_failure_snapshot_round_trip_replay()", 1)[1].split("\nvoid ", 1)[0]
assert "solve_projected_cg(ill_baseline" in cg_replay_test and "failed.witness.residual_checks.size()==3" in cg_replay_test
assert "check.true_relative>1e-8" in cg_replay_test and "check.restarts==i" in cg_replay_test
assert 'read_snapshot_scalar(snapshot,"epsilon")' in cg_replay_test and 'read_snapshot_scalar(early_snapshot,"epsilon")' in cg_replay_test
feedback_start = native_fit.index("const bool feedback_eligible=response.witness.finite")
feedback_end = native_fit.index("const double base_value=", feedback_start)
feedback_body = native_fit[feedback_start:feedback_end]
assert 'classification=="NONPOSITIVE_OPERATOR_DIRECTION"' in feedback_body
assert 'classification=="UNRESOLVED_SOFT_DIRECTION"' in feedback_body
assert "if(!feedback_eligible)" in feedback_body and "throw std::runtime_error" in feedback_body
assert native_fit.index("add_cut(", feedback_end) > feedback_end
assert "test_fit_samples_entry_preserves_inputs" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_full_spd_failure_keeps_candidate_pack" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_probe_selection_skips_duplicate_low_modes" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_response_uncomputed_values_are_explicit" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_probe_moments_centering_and_small_segments" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_check_samples_is_read_only" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_nonfinite_tail_product_is_numerical_failure" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_response_bootstrap_ignores_force_position_tail" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_sample_spool_minimum_frame_boundary" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_fixed_probe_collection_checks_reference_branches" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_baseline_batch_matches_gemv" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "source_atom_slot" not in native_fit
assert "test_response_snapshot_preserves_recomputable_matrices" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_active_set_qp_snapshot_and_rejection" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "tests/data/ja_reference_resample.bin.qp_state.txt" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_outer128_stationarity_joint_polish" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "ja_reference_outer128_766cuts_stationarity_v1.qp_fixture.txt" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert (ROOT / "tests/data/ja_reference_outer128_766cuts_stationarity_v1.qp_fixture.txt").stat().st_size > 1_000_000
assert "solve_active_equalities" in native_fit and "cusolverDnDgesvd(solver,'S','S',p,n" in native_fit
assert "ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT" in native_fit and "ACTIVE_EQUALITY_UNRELIABLE" in native_fit
active_svd_start = native_fit.index("bool solve_active_equalities(")
active_svd_end = native_fit.index("ActiveSetPolishResult polish_cut_qp(", active_svd_start)
active_svd = native_fit[active_svd_start:active_svd_end]
assert "NeumaierAccumulator" in native_fit and "compensated_stationarity_residual" in native_fit
assert "std::fma(left,right,-product)" in native_fit and "compensated_constraint_residual" in native_fit
assert "const double score=std::max({max_complementarity/qp_complementarity_tolerance,max_stationarity/qp_stationarity_tolerance,max_scaled_equality});" in active_svd
assert "h[k]=projected_c.value()/singular[k]+projected_s.value();" in active_svd
assert "trial_lambda[j]<0.0" in active_svd and "y=std::move(best_y);lambda_star=std::move(best_lambda);" in active_svd
assert "if(q>static_cast<std::size_t>(p)){failure_reason=\"ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT\";return false;}" in active_svd
assert "if(da){cudaFree(da);da=nullptr;}" in active_svd and "if(info){cudaFree(info);info=nullptr;}" in active_svd
assert "rounded_alpha" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "rounded_exit.accepted" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "dimension_rank_failure.accepted" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "first_polish_iteration=64,retry_polish_iteration=4096" in native_fit and "max_iterations=200000" in native_fit
assert "result.polish_attempts<3" in native_fit and "4*(m+static_cast<std::size_t>(svd.p))" in native_fit
assert 'method="active_set_polish"' in native_fit and '<<" polish_attempts="' in native_fit
assert "polish_failure_reason" in native_fit and "ACTIVE_FULL_KKT_NOT_ACCEPTED" in native_fit
assert "QP_COORDINATE_DIAGNOSTIC" in native_fit and "QP_POLISH_DIAGNOSTIC" in native_fit
assert 'diagnostic.certificate_computed?"COMPUTED":"NOT_COMPUTED"' in native_fit and "worst_complementarity_primal_slack" in native_fit
assert "diagnostic_finite=out.finite&&diagnostic_finite" in native_fit
assert "worst_stationarity_component" in native_fit and "max_lambda_index" in native_fit
assert "diagnostic_scale_overflow.accepted&&diagnostic_scale_overflow.finite&&!diagnostic_scale_overflow.diagnostic_finite" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert '<<"finite="<<(certificate.finite?1:0)' in native_fit
assert '<<" diagnostic_finite="<<(certificate.diagnostic_finite?1:0)' in native_fit
assert "worst_complementarity_residual" not in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "compensated_stationarity.worst_stationarity_residual==-1.0" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "two_to_54=18014398509481984.0" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "product_roundoff_residual==std::ldexp(1.0,-54)" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "saw_not_computed" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert '<<" coordinate_seconds="<<qp.coordinate_seconds<<" polish_seconds="<<qp.polish_seconds' in native_fit
assert ' SEARCH search_policy=' in native_fit and 'warm_seconds=' in native_fit and 'independent_seconds=' in native_fit
assert 'const int max_search_depth=std::min(384,d-3);' in native_fit
assert 'DeviceLanczosWorkspace lanczos_workspace;lanczos_workspace.initialize(graph,sqrt_mass,sqrt_mass_atom,max_search_depth,psize);' in native_fit
assert 'lanczos_translation_coefficients<<<3,256>>>' in native_fit and 'lanczos_additive_action<<<' in native_fit
assert 'for(int pass=0;pass<2;++pass)' in native_fit and 'CUBLAS_OP_T,d,k+1' in native_fit and 'CUBLAS_OP_N,d,k+1' in native_fit
assert 'r.residual=norm(baseline.out)' in native_fit and 'cudaMemcpy(r.vector.data(),workspace->ritz' in native_fit
assert 'eigen_small(solver,t,m,wanted)' in native_fit and 'out.vectors.resize(static_cast<std::size_t>(n)*returned)' in native_fit
cuda_test = source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert 'check_close(modes[k].residual,std::sqrt(residual2),2e-7)' in cuda_test
assert 'check_close(std::inner_product(v.begin(),v.end(),modes[prior].vector.begin(),0.0),0.0,2e-7)' in cuda_test
assert 'check_close(modes[k].value,exact.values[k],2e-7)' in cuda_test
assert 'workspace.initialize(graph,sqrt_mass,sqrt_atom,8,0)' in cuda_test
assert 'qp_state.find("polish_attempts 3")' in cuda_test
assert 'replay.iterations==64&&replay.polish_attempts==1' in cuda_test
assert 'replay.iterations==4096&&replay.polish_attempts==2' in cuda_test
assert 'replay.iterations==200000&&replay.polish_attempts==3' in cuda_test
assert (ROOT / "tests/data/ja_reference_resample.bin.qp_state.txt").stat().st_size > 1_000_000
assert "options.max_stability_rounds<=0" in native_fit
assert "const int max_rounds=options.max_stability_rounds;" in native_fit
assert 'printf("rpmd_ja fit stability round limit=%d\\n",max_rounds)' in native_fit
assert 'required; IBP is diagnostic only\\nrpmd_ja fit stability round limit=' in native_fit
round_loop_start = native_fit.index("for(int outer=0;outer<max_rounds;++outer)")
round_loop_end = native_fit.index("double compressed_residual2=", round_loop_start)
round_loop = native_fit[round_loop_start:round_loop_end]
assert round_loop.count("outer==max_rounds-1") == 2
assert round_loop.count("to_string(max_rounds)+\" rounds\"") == 3
assert "80 rounds" not in round_loop and "outer==79" not in round_loop
assert "POTRF_diagonal_shift_eV_per_A2_per_amu" in source("src/measure/rpmd_ja_qnep_prepare.cu")

pre_run_start = sampler_fit.index("void RpmdJA_Fit::pre_run(")
pre_run_end = sampler_fit.index("void RpmdJA_Fit::post_force(", pre_run_start)
pre_run_body = sampler_fit[pre_run_start:pre_run_end]
assert 'lock_path_ = output_path_ + ".fit.lock";' in sampler_fit
assert 'output_path_ + ".fit.txt.tmp"' in pre_run_body
assert pre_run_body.index("if (file_exists(path))") < pre_run_body.index('std::fopen(lock_path_.c_str(), "wx")')
assert pre_run_body.index('std::fopen(lock_path_.c_str(), "wx")') < pre_run_body.index("spool_.open(")

release_start = sampler_fit.index("void RpmdJA_Fit::release_lock_()")
release_end = sampler_fit.index("void RpmdJA_Fit::pre_run(", release_start)
release_body = sampler_fit[release_start:release_end]
assert "std::fclose(lock_file_);" in release_body
assert "lock_file_ = nullptr;" in release_body
assert "std::remove(lock_path_.c_str());" in release_body
assert "RpmdJA_Fit::~RpmdJA_Fit() { release_lock_(); }" in sampler_fit

post_run_start = sampler_fit.index("void RpmdJA_Fit::post_run(")
post_run_body = sampler_fit[post_run_start:]
sample_only_return = post_run_body.index("if (sample_only_) {")
native_fit_call = post_run_body.index("fit_rpmd_ja_native_reference(")
sample_only_branch = post_run_body[sample_only_return:native_fit_call]
assert "release_lock_();" in sample_only_branch and "return;" in sample_only_branch
assert "std::remove(spool_path_.c_str())" not in sample_only_branch
assert post_run_body.index("spool_.close();") < sample_only_return < native_fit_call
assert "std::printf(\"rpmd_ja sample saved" in sample_only_branch and "fitting was not run" in sample_only_branch
fit_post_body = post_run_body[native_fit_call:]
assert fit_post_body.index("catch (...) {") < fit_post_body.index("release_lock_();") < fit_post_body.index("throw;")
assert fit_post_body.rindex("release_lock_();") > 0

assert "__host__ __device__ inline bool rpmd_ja_fit_mic(" in sampling
assert "nearbyint(sx)" in sampling and "nearbyint(sy)" in sampling and "nearbyint(sz)" in sampling
assert "return isfinite(x) && isfinite(y) && isfinite(z);" in sampling
assert "output[static_cast<std::size_t>(9) * atoms + atom] = static_cast<double>(error);" in sampling
assert "rpmd_ja_fit_mic(box.cpu_h + 9, box.cpu_h, x, y, z)" in sampler
assert sampler.count("apply_fit_mic(box,") == 1


# Algebra-only multi-cell check; the C++ helper remains covered by source-contract checks above.
def matvec(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(3)) for row in matrix]


def determinant(h):
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    return a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)


def inverse(h):
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    det = determinant(h)
    return [[(e * j - f * i) / det, (c * i - b * j) / det, (b * f - c * e) / det],
            [(f * g - d * j) / det, (a * j - c * g) / det, (c * d - a * f) / det],
            [(d * i - e * g) / det, (b * g - a * i) / det, (a * e - b * d) / det]]


def check_multicell_mic(h, delta, image):
    lattice_shift = matvec(h, image)
    wrapped_input = [delta[i] + lattice_shift[i] for i in range(3)]
    fractional = matvec(inverse(h), wrapped_input)
    wrapped_fractional = [fractional[i] - round(fractional[i]) for i in range(3)]
    actual = matvec(h, wrapped_fractional)
    assert all(isclose(actual[i], delta[i], rel_tol=1e-12, abs_tol=1e-12) for i in range(3))


check_multicell_mic([[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]],
                    [0.2, -0.4, 1.1], [3.0, -2.0, 4.0])
check_multicell_mic([[4.0, 0.7, -0.3], [0.0, 3.5, 0.4], [0.2, 0.0, 5.0]],
                    [0.15, 0.2, -0.3], [-3.0, 4.0, 2.0])
