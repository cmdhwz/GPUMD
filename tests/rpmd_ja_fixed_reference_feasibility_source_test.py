from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


run = source("src/main_gpumd/run.cu")
header = source("src/measure/rpmd_ja_native_fit.cuh")
fit = source("src/measure/rpmd_ja_native_fit.cu")
policy = source("src/measure/rpmd_ja_feasibility_policy.cuh")
cuda_test = source("tests/rpmd_ja_native_fit_cuda_test.cu")
integration_test = source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu")

assert "covariance_ridge_lambda" in header
assert "covariance_analytic_curvature_probes" in header
assert "covariance_feasibility" in header and "covariance_structured_basis" in header

parser_start = run.index('if (tokens[1] == "fit_samples_covariance")')
parser_end = run.index('if (tokens[1] == "prepare")', parser_start)
parser = run[parser_start:parser_end]
assert 'tokens.size() < 14 || tokens[10] != "shrinkage" || tokens[12] != "max_iter"' in parser
assert "for (std::size_t token = 14; token < tokens.size(); ++token)" in parser
for option in ('"ridge"', '"analytic_curvature"', '"feasibility"', '"structured_basis"',
               '"blind_samples"', '"blind_manifest"'):
    assert option in parser
assert "covariance_feasibility&&!options.covariance_structured_basis" in parser
assert "covariance_feasibility&&(ridge_seen||analytic_curvature_seen)" in parser
assert "covariance_feasibility&&(options.covariance_block_newton||options.covariance_joint_block)" in parser
assert "covariance_feasibility&&options.covariance_staged_barrier" in parser
assert "blind_samples_seen!=blind_manifest_seen" in parser
assert "feasibility requires trust_region" not in parser

objective_start = fit.index("CovarianceObjectiveEvaluation evaluate_covariance_fit_objective(")
objective_end = fit.index("struct CovarianceObjectiveChange", objective_start)
objective = fit[objective_start:objective_end]
assert "ridge_lambda" in objective and "coordinate_scale" in objective
assert "value.ridge=0.5*ridge_lambda*covariance_dot(z,z)" in objective
assert "value.objective=value.data+value.prior+value.ridge+value.barrier" in objective

decomposition_start = fit.index("void covariance_decompose_gradient(")
decomposition_end = fit.index("void capture_covariance_snapshot(", decomposition_start)
decomposition = fit[decomposition_start:decomposition_end]
assert "value.gradient_ridge[i]=ridge_lambda*((*theta)[i]-(*theta_baseline)[i])/(scale*scale)" in decomposition
assert "value.gradient[i]=value.gradient_data[i]+value.gradient_prior[i]+value.gradient_ridge[i]+value.gradient_barrier[i]" in decomposition

cache_start = fit.index("class CovarianceAnalyticCurvatureWorkspace")
cache_end = fit.index("std::vector<double> covariance_curvature_probes(", cache_start)
workspace = fit[cache_start:cache_end]
assert "system.apply_parameter_operator(parameter" in workspace
assert "cublasDgemm" in workspace
assert "require_column_residuals" in workspace
assert "covariance_budget_observe_or_throw(budget,limits)" in workspace

hessian_start = fit.index("std::vector<double> covariance_analytic_curvature_hessian(")
hessian_end = fit.index("CovarianceAnalyticCurvatureCache covariance_build_analytic_curvature_cache(", hessian_start)
hessian = fit[hessian_start:hessian_end]
assert "0.5*(1.0+rho)*cache.gram_d[index]+tau*cache.gram_q[index]+(i==j?ridge_lambda:0.0)" in hessian

adapter_start = fit.index("CovarianceTrustRegionModel covariance_direct_hessian_trust_region_model(")
adapter_end = fit.index("CovarianceTrustRegionModel covariance_identity_trust_region_model(", adapter_start)
adapter = fit[adapter_start:adapter_end]
assert "coordinate_transform" in adapter and "spectral_floor" in adapter
assert "INDEFINITE_EQUILIBRATED_HESSIAN" in adapter
assert "MODEL_REGULARIZED_SOFT_EIGENVALUES" in adapter
assert "SPD_VERIFIED_EQUILIBRATED_HESSIAN" in adapter
adaptive_start = fit.index("CovarianceAnalyticCurvatureCache covariance_build_stable_analytic_curvature_cache(")
adaptive_end = fit.index("double covariance_trust_region_reduced_radius(", adaptive_start)
adaptive = fit[adaptive_start:adaptive_end]
assert "probes=16" in adaptive and "probes*=2" in adaptive
assert "probes=previous.probes*2" in adaptive and "seeded analytic covariance stability check" in adaptive
assert "STABLE_10_PERCENT_ENGINEERING_ONLY" in adaptive and "UNSTABLE_AT_128" in adaptive
assert "covariance_normalized_curvature_change" in adaptive
assert "working_probe_count_verified" in fit and "PREVIOUSLY_VERIFIED_PROBE_COUNT_REUSED" in fit
assert "same_point_curvature_rejections>=5" in fit and "PERSISTENT_REJECTIONS_AT_128" in fit
assert "test_covariance_original_residual_gates_convergence" in cuda_test
assert "test_covariance_analytic_curvature_whitened_gram" in cuda_test and "upgraded.levels_built==1" in cuda_test

assert "analytic_curvature_exact_hessian_status" in fit
assert "analytic_curvature_model_floor" in fit
assert "objective_prior=std::numeric_limits<double>::quiet_NaN(),objective_ridge=" in fit
assert "std::vector<double> theta,gradient_data,gradient_prior,gradient_ridge,gradient_barrier,gradient_total" in fit
assert "double ridge_lambda=0.0,ridge_error_estimate=" in fit
assert "ridge_error_bound_status=\"NOT_AVAILABLE\"" in fit
assert "start_gradient_ridge_scaled_inf_norm" in fit
assert "result.ridge_lambda=ridge_lambda" in fit
assert "COVARIANCE_RIDGE lambda=" in fit
assert "std::abs(options.covariance_shrinkage-0.1)>8.0*std::numeric_limits<double>::epsilon()" in fit
assert "ridge_error_norm*ridge_error_norm/(2.0*ridge_lambda)" not in fit
assert "gradient_norm*gradient_norm/(2.0*ridge_lambda)" in fit
assert 'result.ridge_error_bound_status="ESTIMATED"' in fit
assert "result.stagnation_last_window_relative_improvement<0.5" in fit
assert "result.stagnation_low_progress_windows>=2" in fit and 'result.status="NUMERICAL_STALLED"' in fit
assert "CovarianceFeasibilityBudgetError" in fit
reservation_start = fit.index("class CovarianceGradientBudgetReservation")
reservation_end = fit.index("double covariance_dot(", reservation_start)
reservation_class = fit[reservation_start:reservation_end]
assert "return BudgetResult::FAMILY_WALL_LIMIT;" not in reservation_class
assert "inline BudgetResult phase_gate(" in source("src/measure/rpmd_ja_feasibility_policy.cuh")

assert "test_covariance_direct_hessian_preconditioner" in cuda_test
assert "test_covariance_analytic_curvature_whitened_gram" in cuda_test
assert "commutator>1e-4" in cuda_test
assert "covariance_build_stable_analytic_curvature_cache(" in cuda_test
assert "cache.levels_built==2" in cuda_test and "cache.probes==32" in cuda_test
assert "near_rank_one" in cuda_test and "clipped_eigenvalues==1" in cuda_test
assert "test_covariance_centered_ridge_objective_and_tau_cache" in cuda_test
assert "ridge*(trial[i]-baseline.theta[i])/(scale[i]*scale[i])" in cuda_test

gram_start = fit.index("CovarianceAnalyticCurvatureCache covariance_build_analytic_curvature_cache(")
gram_end = fit.index("double covariance_budget_now_seconds()\n{")
gradient_start = fit.index("void complete_covariance_fit_gradient_from_factors(")
gradient_end = fit.index("std::vector<double> covariance_scaled_gradient(", gradient_start)
prepare_try = fit.index("prepare_and_take_staged_ownership(own_scratch,prepare,after_prepare)")
ownership_helper = fit.index("void prepare_and_take_staged_ownership(")
ownership_prepare = fit.index("prepare();", ownership_helper)
ownership_transfer = fit.index("take_staged_prepare_ownership(ownership);", ownership_helper)
ownership_callback = fit.index("after_prepare();", ownership_helper)
prepare_budget = fit.index('observe_prepare_budget("after_prepare")', prepare_try)
prepare_timer_complete = fit.index("prepare_stage_timer.complete();", prepare_try)
final_fit_publish = fit.index("rpmd_ja_file_io::atomic_replace_file(fit_tmp,fit_path)")
scratch_release = fit.index("own_scratch.pack_owned=false;own_scratch.raw_owned=false;", final_fit_publish)
success_endpoint = fit.index('trace<<"FEASIBILITY_ENDPOINT status REFERENCE_WRITTEN', scratch_release)
preflight_guard = fit.index("verify_feasibility_preflight(options.output_path)")
sample_statistics = fit.index("collect_fixed_probe_statistics(in,header")

missing = []
for key, present in (
    ("factor identity fields", all(token in fit for token in
        ("factors_valid", "factor_theta", "factor_epsilon", "factor_assembly_generation"))),
    ("factor identity recovery regression", "test_covariance_factor_identity_recovery" in cuda_test),
    ("common physical probe regression", "test_fixed_physical_response_probes_are_frozen_and_ranked" in cuda_test),
    ("fixed-center development regression", "test_covariance_development_statistics_keep_fixed_training_center" in cuda_test),
    ("independent curvature regression", "test_covariance_directional_curvature_requires_stable_resolved_match" in cuda_test),
    ("shared factor guard in Gram builder", "system.ensure_factors(theta,epsilon)" in fit[gram_start:gram_end]),
    ("shared factor guard in gradient completion", "system.ensure_factors(theta,epsilon)" in fit[gradient_start:gradient_end]),
    ("no obsolete reservation completion", "trace_reservation.complete()" not in fit),
    ("ridge replay enabled", "RIDGE_DIAGNOSE_NOT_IMPLEMENTED" not in fit),
    ("M1 replay enabled", "M1_DIAGNOSE_NOT_IMPLEMENTED" not in fit),
    ("new replayable state format", "state_format_version 4" in fit),
    ("ridge gradient persisted", 'write_vector(out,"last_valid_gradient_ridge"' in fit),
    ("development sufficient statistics", "development_statistics" in fit),
    ("unregularized development objective", "development_objective_unregularized" in fit),
    ("prepare outputs owned before timer and post-prepare checks",
        ownership_helper < ownership_prepare < ownership_transfer < ownership_callback and
        prepare_try < prepare_timer_complete < prepare_budget),
    ("prepare temporary outputs tracked", "output_tmp_owned" in fit and "sidecar_tmp_owned" in fit),
    ("success endpoint follows report publication and scratch cleanup", final_fit_publish < scratch_release < success_endpoint),
    ("feasibility requires execution-backed preflight before sample traversal", preflight_guard < sample_statistics and
        "GPUMD_RPMD_JA_PREFLIGHT_RECEIPT" in fit and "cuda_regression_exit_code" in fit),
    ("preflight binds manifest and executable/config identities", "manifest_fnv64" in fit and
        "native_executable_sha256" in fit and "run_config_sha256" in fit),
    ("portable startup launcher exists", (ROOT / "scripts/rpmd_ja_feasibility_preflight.py").is_file()),
    ("startup receipt captures relevant environment", all(token in source("scripts/rpmd_ja_feasibility_preflight.py")
        for token in ("CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CUDA_HOME", "CUDACXX", "NETCDF_ROOT", "NETCDF_HOME", "NETCDF_DIR"))),
    ("reference success does not claim J_A or conductivity", "J_A_relation NOT_TESTED thermal_conductivity NOT_TESTED" in fit),
    ("frozen a0 bridge polarity", "double a0;" in source("src/measure/rpmd_ja_local_basis.cuh") and
        "double chi;" not in source("src/measure/rpmd_ja_local_basis.cuh")),
    ("numerical stops classified as incomplete", "covariance_incomplete_candidate_status(status)" in fit and
        "NUMERICAL_STAGNATION" in fit and "NUMERICAL_STALLED" in fit),
    ("common physical probe channels", "make_fixed_physical_probes" in fit and
        "FIXED_PHYSICAL_OHO_NEAR_SALT" in fit and "FIXED_PHYSICAL_OXYGEN_SKELETON_" in fit and
        "FIXED_PHYSICAL_SALT_POLARITY_" in fit),
    ("development moments are emitted only when response statistics are complete",
        'write_response_consumer_statistics(trace,std::string(label)+"_DEVELOPMENT",scored)' in fit and
        '<<label<<"_MEAN_Q_FIXED_R0 "' in fit and 'label+"_CENTERED_COVARIANCE_ABOUT_SEGMENT_MEAN"' in fit and
        'label+"_SECOND_MOMENT_ABOUT_FIXED_R0"' in fit),
    ("independent curvature is wired and metered", "covariance_verify_directional_curvature" in fit and
        "ACTUAL_SUGGESTED_STEP" in fit and "FIXED_RANDOM" in fit and
        "analytic_curvature_direction_objectives" in fit and "model_floor_excluded 1" in fit),
    ("directional curvature jointly bounds feasibility stability resolution and objectives", "maximum_step_halvings=24" in fit and
        "STEP_HALVING_UNSTABLE_RETRY" in fit and "DIRECTIONAL_ARITHMETIC_UNRESOLVED" in fit and
        "objective_evaluations>=max_objective_evaluations" in fit and
        "curvature_evaluations_before+8" in cuda_test and "boundary_unstable.status==\"STEP_HALVING_UNSTABLE\"" in cuda_test and
        "boundary_stable.resolved" in cuda_test and "system.ensure_factors(current.cached_theta,epsilon)" in fit),
    ("unexpected independent-curvature budget states remain hard failures", "covariance_independent_budget_stop_or_throw(error)" in fit and
        "covariance_independent_failure_status_or_throw(check.status)" in fit and
        "invalid_budget_rethrown" in cuda_test and "unknown_result_rejected" in cuda_test),
    ("same-tau parameter updates do not repeat full directional differences", "trigger,include_reference_directions);" in fit and
        "&step,\"SUGGESTED_STEP\",g,false" in fit and fit.count("curvature_direction_checked_tau=std::numeric_limits<double>::quiet_NaN();")==1 and
        "tau_stage_summaries==count_text(\"BFGS_STAGE_RESET tau=\")" in cuda_test),
    ("cgroup preflight follows current membership and all mounted ancestors", "cgroup_memory_ancestor_paths" in source("src/measure/rpmd_ja_utf8_file_io.h") and
        "proc/self/cgroup" in source("src/measure/rpmd_ja_utf8_file_io.h") and
        "proc/self/mountinfo" in source("src/measure/rpmd_ja_utf8_file_io.h") and
        "test_cgroup_v2_nested_limit_and_tighter_ancestor" in source("tests/rpmd_ja_file_io_cpu_test.cpp") and
        "test_cgroup_memory_unlimited_missing_and_unreadable" in source("tests/rpmd_ja_file_io_cpu_test.cpp")),
    ("published reference readback failure releases final and scratch outputs", "NativeFitScratchOwnership" in fit and
        "test_published_outputs_cleaned_after_readback_failure" in cuda_test and
        "TEST_PREPARE_POST_PUBLICATION_FAILURE" in cuda_test and "take_published_prepare_ownership" in cuda_test),
    ("real prepare integration releases files after a post-publication journal failure",
        "prepare_and_take_published_ownership(ownership" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "prepare_rpmd_ja_qnep_reference(raw,owned_failure_path,kernel,{},pack)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "path_exists(owned_failure_path)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "path_exists(pack)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu")),
    ("zero-ridge v3 state has solve-write-diagnose replay regression", "test_covariance_solve_write_diagnose_replay_without_ridge" in cuda_test and
        'fields.at("ridge_coordinates")=="0"' in cuda_test and "target_objective disagrees with rho/ridge_lambda" in fit),
    ("edge attribution keeps parameter ridge outside edge totals", "parameter_ridge_gradient=value.gradient_ridge" in fit and
        "rho<0.0||!std::isfinite(rho)" in fit and "edge_total+value.gradient_ridge[i]" in fit),
    ("local and joint Hessians include zero-rho ridge curvature", "std::array<double,36> data{},prior{},barrier{},ridge{},total{}" in fit and
        "std::vector<double> data,prior,barrier,ridge,total" in fit and "hessian.ridge[static_cast<std::size_t>(a)*6+b]" in cuda_test and
        "result.ridge[at]=a==b?ridge_lambda:0.0" in fit and "result.ridge[at]=parameter_a==parameter_b?ridge_lambda:0.0" in fit and
        "const std::vector<double> scale={0.5,1.25,1.5,0.75,2.0,0.6}" in cuda_test and "rho_fixed=\"<<rho" in fit),
    ("training and blind cannot reuse the same file or content", "validate_distinct_training_blind" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "os.path.samefile(training, blind)" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "training_sha256.lower() == blind_sha256.lower()" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "feasibility_same_file_path(training_path,blind_path)" in fit and
        "test_training_and_blind_reject_same_file_aliases" in source("tests/rpmd_ja_feasibility_preflight_test.py") and
        "test_training_and_blind_reject_identical_content_copies" in source("tests/rpmd_ja_feasibility_preflight_test.py")),
    ("Windows file identity and fit I/O use UTF-8 paths", "MultiByteToWideChar(CP_UTF8" in source("src/measure/rpmd_ja_utf8_file_io.h") and
        "CreateFileW(first_wide.c_str()" in fit and "rpmd_ja_file_io::ifstream blind" in fit and
        "rpmd_ja_file_io::ifstream raw_in" in source("src/measure/rpmd_ja_qnep_prepare.cu") and
        "rpmd_ja_file_io::ifstream in(path, std::ios::binary)" in source("src/measure/rpmd_ja_additive.cuh") and
        "utf8_path" in cuda_test),
    ("M1 writer regression has an independent full K assembly", "IndependentAdditiveOperators independently_assemble_additive_operators" in cuda_test and
        "result.k[static_cast<std::size_t>(a*n+term.site[row].atom)*d+b*n+term.site[column].atom]+=" in cuda_test and
        "expected_k[static_cast<std::size_t>(a*n+term.site[row].atom)*d+b*n+term.site[column].atom]+\n              =" not in cuda_test),
    ("preflight validates all identities and normalized file identity", all(token in source("scripts/rpmd_ja_feasibility_preflight.py")
        for token in ("training_fnv64", "blind_fnv64", "declared_model_fingerprint", "declared_config_fingerprint", "same_resolved_path")) and
        "CreateFileW(first_wide.c_str()" in fit and "GetFileInformationByHandle(first_handle" in fit and
        "nFileIndexHigh" in fit and "::stat(first.c_str(),&first_info)" in fit and
        "test_feasibility_same_file_path_aliases" in cuda_test),
    ("native qraw identity is checked before fitting", fit.index("(void)verify_feasibility_manifest(feasibility_manifest_fields", fit.index("validate_qraw_identity(raw,options,header,r0,box,force)")) <
        fit.index("solve_branch(graph,m0_statistics")),
    ("native input SHA is bound to the initial FNV in one scan", "struct FitInputContentIdentity" in fit and
        "sha.update(reinterpret_cast<const unsigned char*>(buffer)" in fit and
        "verify_feasibility_source_content_hashes(feasibility_manifest_fields,kernel_content_identity,raw_content_identity)" in fit and
        fit.index("verify_feasibility_source_content_hashes(feasibility_manifest_fields,kernel_content_identity,raw_content_identity)") <
        fit.index("read_training_r0(in,header")),
    ("native startup rejects qraw and kernel replaced after launcher hashing",
        "test_manifest_sha256_rejects_qraw_and_kernel_replacement_at_native_start" in integration_test and
        "qraw content differs from the frozen manifest SHA-256" in fit and
        "qNEP kernel content differs from the frozen manifest SHA-256" in fit),
    ("offline run accepts only required setup commands before fit", all(command in source("scripts/rpmd_ja_feasibility_preflight.py")
        for command in ("potential", "pppm_mesh_spacing", "pimd_fix_com", "kspace")) and
        "unsupported = sorted({command[0]" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "test_offline_fit_rejects_computation_before_fit" in source("tests/rpmd_ja_feasibility_preflight_test.py")),
    ("A0 baseline soft direction is numerically distinct and reaches the common probe", "test_a0_soft_modes_use_covariance_baseline_point" in cuda_test and
        "lanczos_low_modes(solver,baseline,graph,a0.theta" in cuda_test and
        "lanczos_low_modes(solver,baseline,graph,mode_theta" in cuda_test and
        "covariance_dot(solved_modes.front().vector,baseline_modes.front().vector))<1.0e-8" in cuda_test and
        "sources.front()==\"LOW_RITZ\"" in cuda_test),
    ("preflight runs a separate real prepare/readback smoke", "cuda_integration_executable" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "--integration-smoke" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "run_additive(base,1);run_additive(base+\"_internal_com\",2);return 0;" in source("tests/rpmd_ja_qnep_prepare_cuda_test.cu")),
    ("integration smoke passes actual M1 writer output through real prepare", (ROOT / "tests/rpmd_ja_fit_prepare_integration_cuda_test.cu").is_file() and
        "GPUMD_RPMD_JA_NATIVE_FIT_TEST_REAL_PREPARE" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "write_native_additive_site_blocks(out,graph,theta,edge_linear,n)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "prepare_rpmd_ja_qnep_reference(raw,output_path,kernel,{},pack,[&]" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "verify_native_additive_package_entries(parsed,graph,theta,positions,box,n)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "injected independent package-entry callback failure" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "failed package-entry validation left a publishable reference artifact" in
        source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu")),
    ("production prepare checks the parsed selected candidate K/H actions before writing", "if(additive_validator)additive_validator(additive)" in
        source("src/measure/rpmd_ja_qnep_prepare.cu") and
        "verify_native_additive_package_entries(package,graph,theta,r0,box,n)" in fit and
        "native_additive_entry_relative_tolerance" in fit and "expected_abs_sum" in fit and
        "native_additive_entry_share_sum_tolerance" not in fit and
        "max_entry_error_over_limit_K" in fit and "ADDITIVE_PACKAGE_SPARSE_ENTRYWISE_PASS" in fit),
    ("preflight requires the actual M1 COM prepare/readback marker", "ACTUAL_M1_V2_PREPARE_READBACK_PASS" in
        source("scripts/rpmd_ja_feasibility_preflight.py") and "ACTUAL_M1_V2_PREPARE_READBACK_PASS" in
        source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu")),
    ("M1 integration checks independent nonsymmetric H and negative mutations", "independently_assemble_additive_operators(graph,theta,positions,box,n)" in
        source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "test_native_additive_entry_verifier_rejects_small_scale_K_H" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "small-entry positive control has invalid error-over-limit metrics" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "cancelled positive control has invalid error-over-limit metrics" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "cancelled verifier accepted missing " in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "small-entry verifier accepted missing 1e-9 " in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "actual M1 prepare/readback H matrix disagrees with the independent local-energy transport formula" in
        source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "require_entry_check_rejects(bad_k,\"K entry\")" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "fixed-probe-orthogonal local K perturbation" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "transposed nonsymmetric H" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "matches(omitted_additive)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "matches(transposed_additive)" in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu") and
        "energy_share[site]=1.0/3.0" not in source("tests/rpmd_ja_fit_prepare_integration_cuda_test.cu")),
    ("large training and blind FNV scans are native", "fnv64(sample)" not in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "fnv64(blind)" not in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "rpmd_ja_model_fingerprint(training_path)" in fit and "rpmd_ja_model_fingerprint(blind_path)" in fit),
    ("final selected lambda recorded", "selected_covariance_lambda" in fit),
    ("M1 support rule is manifest-frozen and applied before operator residualization",
        "m1_minimum_support_count" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "feasibility_manifest_minimum_support" in fit and
        "DROP_BEFORE_OPERATOR_RESIDUALIZATION" in fit and
        fit.index("raw_terms.erase(std::remove_if") < fit.index("normalize_added_operators(raw_operators,base)") and
        "low_metadata.added_parameter_count==0" in cuda_test),
    ("B and C receive equal remaining wall caps in addition to gradient caps",
        "equal_remaining_family_wall_cap" in fit and "FAMILY_WALL_LIMIT" in fit and
        "max_fit_family_wall_seconds" in source("src/measure/rpmd_ja_feasibility_policy.cuh")),
    ("candidate soft-direction union is frozen and every candidate is rescored",
        "COMMON_SOFT_PROBE_UNION" in fit and "COMMON_PROBE_RESCORING" in fit and
        "candidate_soft_union NOT_IMPLEMENTED" not in fit and "frozen_common_probe_sources" in fit),
    ("development uncertainty and lambda sensitivity use paired fixed-R0 blocks",
        "development_block_statistics" in fit and "development_objective_uncertainty_2se" in fit and
        "paired_development_objective_uncertainty_2se" in fit and
        "REGULARIZATION_SENSITIVITY status" in fit and "development_objective_uncertainty NOT_ESTIMATED" not in fit),
    ("cross-family candidate choice uses common paired block scores",
        "development_candidates[index].development_block_scores_b=candidate.development_block_scores_b" in fit and
        "development_candidates[index].development_block_scores_2b=candidate.development_block_scores_2b" in fit and
        "compare_candidates_at_both_scales(" in policy and "DATA_INCONCLUSIVE" in policy and
        "test_cross_family_selection_uses_resolved_paired_block_difference" in source("tests/rpmd_ja_feasibility_policy_test.py")),
    ("response sensitivity uses a manifest-frozen independent threshold before blind access",
        "response_delta_tolerance" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "feasibility_manifest_response_delta_tolerance(manifest)" in fit and
        "feasibility_manifest_response_delta_tolerance(receipt)" in fit and
        "regularization_sensitivity_allows_blind(regularization_sensitive" in fit and
        "regularization_sensitivity_allows_blind" in policy and
        "selection_eligible_for_blind 0" in fit and
        fit.index("selection_eligible_for_blind 0") < fit.index("open_blind_once(frozen)")),
    ("independent curvature seeds, probe cap, and tolerance are checked against receipt and native constants",
        "curvature_primary_seed" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "curvature_independent_seed" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "curvature_probe_cap" in source("scripts/rpmd_ja_feasibility_preflight.py") and
        "same_feasibility_curvature_contract" in fit and
        "curvature_directional_tolerance" in source("doc/gpumd/input_parameters/rpmd_ja.rst")),
    ("class body has no misplaced wall-budget branch",
        "return BudgetResult::FAMILY_WALL_LIMIT;" not in fit[fit.index("class CovarianceGradientBudgetReservation"):fit.index("double covariance_dot(")]),
):
    if not present:
        missing.append(key)
assert not missing, "incomplete fixed-reference integration: " + ", ".join(missing)

assert "const double correct=expected.h[alpha][static_cast<std::size_t>(row)*d+column]/mass;" in integration_test
assert "const double wrong_transpose=expected.h[alpha][static_cast<std::size_t>(column)*d+row]/mass;" in integration_test
assert 'run_additive(base,1);run_additive(base+"_internal_com",2);' in integration_test

fit_after_b = fit.index("run_family(rpmd_ja_feasibility::Family::M0,graph")
b_probe_rescore = fit.index('freeze_common_probes("B_GATE")', fit_after_b)
b_common_rescore = fit.index('rescore_candidates(rpmd_ja_feasibility::Phase::B,"B_GATE",false);', b_probe_rescore)
b_admission = fit.index("bool b_admissible=false;", fit_after_b)
c_dispatch = fit.index("should_run_structured_family(", b_admission)
assert b_probe_rescore < b_common_rescore < b_admission < c_dispatch
assert "COMMON_PROBE_B_GATE" in fit[fit_after_b:c_dispatch]
assert "enter_budget_phase(rpmd_ja_feasibility::Phase::FREEZE)" not in fit[fit_after_b:c_dispatch]
assert 'rescore_candidates(rpmd_ja_feasibility::Phase::FREEZE,"FINAL_M0_M1",false);' in fit
assert "candidate.fit,\n            phase,candidate.development" in fit
assert "response_consumer_statistics_ready(candidate.development)" in fit
assert "candidate.fit.converged&&candidate.development_scored&&!development_budget_stop" in fit
assert "std::vector<double> a0_development_scores_b,a0_development_scores_2b" in fit
assert "std::vector<double> a1_development_scores_b,a1_development_scores_2b" in fit
assert "a0_response_complete=response_consumer_statistics_ready(a0_response)" in fit
assert "a1_response_complete=response_consumer_statistics_ready(a1_response)" in fit
assert "should_run_structured_family(true, false)" in source("tests/rpmd_ja_feasibility_policy_test.cpp")
assert fit.index('for(std::size_t i=0;i<soft.size()', fit.index("std::vector<std::vector<double>> make_probes(")) < \
       fit.index("if(fixed_physical)", fit.index("std::vector<std::vector<double>> make_probes("))
assert "make_common_probe_set(common_soft_union" in fit
assert 'parse_quoted_fields(manifest_path, "RPJAFEAS9")' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "validate_performance_stop_contract(manifest)" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert '"RPJAPREFLIGHT5"' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "STATISTICS_STORAGE_BUDGET_BYTES = 512 * 1024 * 1024" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert '"statistics_storage_budget_bytes"' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "candidate_cost_stop_rule" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert 'CANDIDATE_COST_STOP_RULE = "MAX_COMPLETE_OR_PARTIAL_LOWER_BOUND_SECONDS_V1"' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "same_feasibility_candidate_cost_contract" in fit
assert "feasibility_manifest_experiment_contract" in fit and "same_feasibility_experiment_contract" in fit
assert "test_lambda_budget_and_development_block_contract_is_frozen" in source("tests/rpmd_ja_feasibility_preflight_test.py")
run_family_start = fit.index("auto run_family=")
cost_gate = fit.index("measured_candidate_cost_gate", run_family_start)
candidate_solve = fit.index("candidate.fit=solve_branch", run_family_start)
assert cost_gate < candidate_solve
assert "MEASURED_COST_STOP" in fit and "MEASURED_FINALIZATION_COST" in fit
assert "record_partial_candidate_elapsed(measured_candidate_costs,partial_seconds)" in fit
assert "record_partial_candidate_elapsed(measured_candidate_costs,candidate_total_seconds)" in fit
assert "partial_elapsed_lower_bound_seconds" in fit and "partial_elapsed_lower_bound_seconds" in source("src/measure/rpmd_ja_feasibility_policy.cuh")
assert "test_measured_candidate_cost_stops_before_spending_c_family_allowance" in source("tests/rpmd_ja_feasibility_policy_test.py")
assert "test_expensive_incomplete_candidate_blocks_next_same_family_candidate" in source("tests/rpmd_ja_feasibility_policy_test.py")
assert "maximum_frame_count=2000" not in fit
assert "make_internal_statistics_probes" in fit and "fixed_internal_projection:" in fit
assert "STATISTICAL_ADMISSION status" in fit
assert 'if(admission_statistics.status!="PASS")' in fit
assert "estimate_admission_statistics_storage_bytes" in fit and "statistics_storage_budget_bytes" in fit
statistics_collection_start = fit.index("std::vector<double> collect_covariance_fit_statistics(")
statistics_collection = fit[statistics_collection_start:fit.index("return sums;", statistics_collection_start)]
assert statistics_collection.index("estimate_admission_statistics_storage_bytes(") < statistics_collection.index("training_admission.values.resize")
assert "checked_product(2,frame_bytes)" in fit and "candidate_score_values=checked_product(22,development_frames)" in fit
assert "test_fixed_internal_probe_statistics_detect_zero_com_internal_shift" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_admission_statistics_storage_peak_boundary" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "injected statistics budget stop" in source("tests/rpmd_ja_statistics_policy_test.cpp")
assert "const BudgetObserver stop_after_twenty_two" in source("tests/rpmd_ja_statistics_policy_test.cpp")
assert "test_fixed_r0_gate_includes_anomalous_tail_outside_common_block_prefix" in source("tests/rpmd_ja_statistics_policy_test.py")
assert 'status!="PRODUCER_OBSERVED"' in fit
assert 'resolved.get("status") != "PRODUCER_OBSERVED"' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "PHYSICAL_PLUS_RING_SPRINGS" in source("tests/rpmd_ja_feasibility_preflight_test.py")
assert "feasibility_cg_witness_path" in fit and '"B_GATE"' in fit and '"FINAL_M0_M1"' in fit
assert 'const std::function<void()>& budget_observer={}' in fit
assert "stop_during_cg" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
cmake_source = source("CMakeLists.txt")
policy_source = source("src/measure/rpmd_ja_feasibility_policy.cuh")
assert 'target_include_directories(rpmd_ja_statistics_policy_cpu_test PRIVATE src src/measure)' in cmake_source
for cpu_target in ("rpmd_ja_covariance_checkpoint_cpu_test", "rpmd_ja_a0_reuse_export_cpu_test",
                   "rpmd_ja_preflight_receipt_cpu_test", "rpmd_ja_local_basis_cpu_test"):
    assert f"add_executable({cpu_target}" in cmake_source
    assert f"add_test(NAME {cpu_target}" in cmake_source
for harness in ("tests/rpmd_ja_covariance_checkpoint_test.py", "tests/rpmd_ja_feasibility_preflight_test.py",
                "tests/rpmd_ja_feasibility_policy_test.py", "tests/rpmd_ja_local_basis_test.py"):
    assert "rpmd_ja_cpp_test_support" in source(harness)
assert '"cl"' in source("tests/rpmd_ja_cpp_test_support.py")
assert "blind_statistics_declares_drift" in policy_source
assert "const bool rejected=rpmd_ja_feasibility::blind_statistics_declares_drift" in fit
assert "resume_scope PREBLIND_A0_A1_B_C" not in fit
assert "blind_or_publication_interruption_refused=1" not in fit
assert "#undef NDEBUG" in source("tests/rpmd_ja_statistics_policy_test.cpp")
assert "validate_cuda_regression_fixtures(fixture_directory)" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "FROZEN_CUDA_QP_FIXTURE_SHA256" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert 'feasibility_manifest_value(receipt,"cuda_regression_fixture_status")!="PASS"' in fit
assert 'cuda_regression_fixture_set_sha256' in fit and 'verify_frozen_qp_fixture_receipt(receipt)' in fit
assert 'frozen_qp_fixtures[i].sha256' in fit
manifest_value_definition = fit.index('const std::string& feasibility_manifest_value(')
assert manifest_value_definition < fit.index('feasibility_manifest_value(fields,"m1_minimum_support_count")')
assert 'const std::map<std::string,std::string>& fields,const std::string& key' in fit
assert 'verify_qp_fixture_bytes(path,bytes)' in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert fit.index('reject_unexpected_stop("A0",a0)') < fit.index('reject_unexpected_stop("A1",a1,false)') < fit.index('auto run_family=')
assert 'covariance_same_curvature_point(analytic_curvature_cache,independent_curvature_cache)' in fit
assert 'independent_curvature_checked_tau!=tau' in fit
assert '"M1_basis_fingerprint_fnv64","M1_parameter_names"' in fit
assert 'test_native_m1_writer_reader_cross_neighbor_images_and_transport' in source("tests/rpmd_ja_native_fit_cuda_test.cu")
prepare_source = source("src/measure/rpmd_ja_qnep_prepare.cu")
reference_source = source("src/measure/rpmd_ja_reference.cu")
reference_header = source("src/measure/rpmd_ja_reference.cuh")
assert 'expected_actions[alpha+1][probe]' in prepare_source
assert 'read_rpmd_ja_reference(output_path)' in prepare_source
assert 'block_matrix_action(*stored_matrices[matrix],stored_probes[probe])' in prepare_source
assert 'rpmd_ja_file_io::remove_file(sidecar_path)' in prepare_source
assert 'estimate_rpmd_ja_qnep_v3_max_serialized_bytes' in reference_header
assert 'estimate_rpmd_ja_qnep_v3_max_serialized_bytes' in reference_source
assert 'Bound the native prepare stream prefix, four block matrices, and stability footer' in reference_source
assert 'add(21*sizeof(double)+2*sizeof(double))' in reference_source
qnep_native_prefix = reference_source[reference_source.index('std::streampos write_rpmd_ja_qnep_v3_stream_prefix'):
                                      reference_source.index('std::uint64_t rpmd_ja_model_fingerprint')]
assert 'out.write(reinterpret_cast<const char*>(diagnostics), sizeof(diagnostics))' in qnep_native_prefix
assert 'write_vector(out, std::vector<double>(diagnostics' not in qnep_native_prefix
assert 'fixed diagnostics array without a vector-length field' in qnep_native_prefix
qnep_prepare_test = source("tests/rpmd_ja_qnep_prepare_cuda_test.cu")
assert 'actual_bytes>estimate_rpmd_ja_qnep_v3_max_serialized_bytes(r)' in qnep_prepare_test
assert 'run_dense_h_upper_bound' in qnep_prepare_test and 'tile.rank!=0' in qnep_prepare_test
assert 'full-rank D fixture did not exercise dense tile storage' in qnep_prepare_test
assert 'qnep_size_metadata.types=header.types;qnep_size_metadata.masses=header.masses;qnep_size_metadata.positions=r0' in fit
assert 'load_rpmd_ja_kernel_table(options.kernel_table,qnep_size_metadata)' in fit
assert 'estimate_rpmd_ja_qnep_v3_max_serialized_bytes(qnep_size_metadata)' in fit
assert "prioritize_soft?make_common_probe_set" in fit
assert "rank_limit=maximum_probe_rank<0?internal:std::min(internal,maximum_probe_rank)" in fit
assert "for(int k=0;k<16&&static_cast<int>(probes.size())<rank_limit;++k)" in fit
assert "validation-1,\n          expected_probes,expected_sources,expected_origins" in fit
assert "&frozen_common_probe_vectors,&frozen_common_probe_sources,&frozen_common_probe_origins" in fit
assert fit.count("&common_physical_probes,true") >= 2
assert "common_soft_probe_capacity(" in fit and "common_soft_probe_capacity(3, 4, 28) == 3" in source("tests/rpmd_ja_feasibility_policy_test.cpp")
assert '"LOW_RITZ"' in fit and "low_ritz_probe_rank" in fit and "fixed_channels_displaced" in fit
assert "NO_LOW_RITZ_PROBES_FIT" in fit
assert "validate_blind_probe_capacity" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "blind_probe_capacity = validate_blind_probe_capacity" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "abs(training_temperature - blind_temperature) > 1.0e-10 * training_temperature" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "training_masses != blind_masses or training_types != blind_types" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "blind_provenance = validate_blind_provenance(" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "validate_provenance_source_relation(training_provenance, blind_provenance)" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert 'verify_sample_provenance("blind","blind_sha256")' in fit
assert 'FEASIBILITY_SOURCE_PROVENANCE sample ' in fit and 'FEASIBILITY_SOURCE_RELATION status ' in fit
assert 'training_model_sha256' in fit and 'blind_model_sha256' in fit
assert 'training_physical_force_configuration_sha256' in fit and 'blind_integrator_configuration_sha256' in fit
assert 'save_feasibility_cg_witness("BLIND"' in fit
assert 'write_cg_witness(path,witness,candidate_theta,raw_path,spool_path,cg_replay,&branch_graph)' in fit
assert 'reconcile_controller_resume_budget(controller_journal' in fit
assert 'OUTER_JOURNAL_RECONCILED' in fit
provenance_header = source("src/measure/rpmd_ja_sample_provenance.h")
assert 'active_run_configuration_lines(lines,number_of_steps,run_ordinal,active_lines)' in provenance_header
assert 'canonical_token_lines_for_configuration(active_lines,true)' in provenance_header
assert 'canonical_token_lines_for_configuration(active_lines,false)' in provenance_header
assert 'result.active_configuration_scope=result.active_configuration_identified?"PREFIX_BEFORE_ACTIVE_RUN"' in provenance_header
assert 'snapshot(\n    force,atom,box,number_of_steps,current_run_ordinal' in source("src/measure/rpmd_ja_fit.cu")
assert "blind_header.temperature-header.temperature)>1.0e-10*header.temperature" in fit
assert "make_common_probe_set(modes" in cuda_test and "frozen_sources==development_sources&&frozen_sources==blind_sources" in cuda_test
assert "uncapped.size()==21" in cuda_test and "maximum_probe_rank=20" in cuda_test
assert "count_source(fixed_only_sources,\"FIXED_RANDOM\")==16" in cuda_test
assert "count_source(frozen_sources,\"TYPE_LOCAL\")==2" in cuda_test
assert "count_type_origins(frozen_origins)==2" in cuda_test
assert "write_blind_response_checkpoint(snapshot_path,blind" in cuda_test
assert "missing_type_local_rejected" in cuda_test and "massless_type_local_rejected" in cuda_test
assert "test_common_probe_rank_is_capped_for_nearly_full_fixed_basis" in cuda_test
assert "iteration_budget_check(k,0.0)" in fit and "iteration_budget_check(k,std::chrono::duration<double>" in fit
assert "COMMON_PROBE_B_GATE_ACCOUNTING" in fit and "max_uninterruptible_lanczos_iteration_seconds" in fit
assert "covariance_require_expected_budget_observation(clock_status,\"run_family candidate entry\")" in fit
assert "covariance_require_expected_budget_observation(b_gate_accounting_status,\"COMMON_PROBE_B_GATE_ACCOUNTING\")" in fit
assert "test_budget_observation_classifies_hard_clock_errors" in cuda_test
assert "test_covariance_prior_calibration_ignores_m1_tail" in cuda_test
assert "test_m1_covariance_calibration_uses_only_base_prefix_in_replay" in cuda_test
assert "selection=USER_SUPPLIED_RHO" in fit
assert "C_budget_preserved 1" not in fit
assert "test_lanczos_iteration_budget_interrupts_one_overrun" in cuda_test
assert "test_single_uninterruptible_b_gate_step_overrun_is_accounted" in source("tests/rpmd_ja_feasibility_policy_test.py")
assert "test_blind_temperature_uses_native_relative_tolerance" in source("tests/rpmd_ja_feasibility_preflight_test.py")
assert "test_blind_header_rejects_cell_mass_and_type_mismatches" in source("tests/rpmd_ja_feasibility_preflight_test.py")
assert "native_additive_package_header_bytes=sizeof(additive_magic)" in fit and \
       "native additive package size estimate does not match bytes written" in fit
assert "static_cast<std::uint64_t>(sized.tellg())==estimated_bytes" in cuda_test
additive_reader = source("src/measure/rpmd_ja_additive.cuh")
assert "version < 1 || version > 6" in additive_reader
assert "version == 2 || version == 4 || version == 6" in additive_reader
assert "rpmd_ja_additive_max_expanded_elements" in additive_reader
assert "local matrix exceeds the per-site element limit" in additive_reader
offline_fitter = source("tools/rpmd_ja_fit_reference.py")
offline_prepare = source("tools/rpmd_ja_qnep_prepare.py")
assert "version = 6 if data.get(\"internal_mass_com\", False) else 5" in offline_fitter
assert "version not in (1,2,3,4,5,6)" in offline_fitter
assert "internal_mass_com=version in (2,4,6)" in offline_fitter
assert "version not in (1, 2, 3, 4, 5, 6)" in offline_prepare
assert 'count>q*q' in offline_prepare and 'entry_bytes>size-f.tell()' in offline_prepare
assert "MAX_ADDITIVE_DENSE_ELEMENTS = _FITTER.MAX_ADDITIVE_DENSE_ELEMENTS" in offline_prepare
assert "read_sparse_scalars" in additive_reader and "std::signbit(value)" in additive_reader
assert "if(value!=0.0||std::signbit(value))" in fit
assert "write(out,std::uint32_t(options.internal_mass_com?6:5))" in fit
assert "publication_payload_restored?0:estimated_package_bytes" in fit
assert "publication_payload_restored?0:\n    estimated_qnep_reference_bytes+qnep_sidecar_and_failure_reserve_bytes" in fit
assert '<<" publication_payload_restored "<<(publication_payload_restored?1:0)' in fit
assert "support_entries_upper_bound" in fit and "quadratic_bytes" not in fit[fit.index("NativeAdditivePackageEstimate estimate_native_additive_package_size("):fit.index("#ifdef GPUMD_RPMD_JA_NATIVE_FIT_TEST", fit.index("NativeAdditivePackageEstimate estimate_native_additive_package_size("))]
assert "actual additive package bytes exceed the conservative support bound" in fit
assert "prefit_estimate.bytes>=estimated_bytes" in cuda_test
assert "write_sparse_package(3)" in source("tests/rpmd_ja_additive_sparse_test.cpp")
assert "write_sparse_package(4)" in source("tests/rpmd_ja_additive_sparse_test.cpp")
assert "#undef NDEBUG" in source("tests/rpmd_ja_additive_sparse_test.cpp")
assert "test_hand_authored_sparse_package_readers_without_hardlink_writer" in source("tests/rpmd_ja_additive_prepare_test.py")
assert "bad_index_rejected" in source("tests/rpmd_ja_additive_sparse_test.cpp")
assert "truncated_rejected" in source("tests/rpmd_ja_additive_sparse_test.cpp")
assert "actual M1 version-6 package byte count differs from writer estimate" in integration_test
assert "source_fnv64=0" in fit and "verify_frozen_fit_input_fingerprints(raw_path,options.kernel_table,raw_source_fingerprint,kernel_source_fingerprint)" in fit
assert "write(out,raw_source_fingerprint)" in fit and "qraw_source_fnv64 \"<<identity.source_fnv64" in fit
assert "covariance_a0_soft_mode_theta(a0)" in fit and "test_a0_soft_modes_use_covariance_baseline_point" in cuda_test
assert "line.back()=='\\r'" in fit and "test_feasibility_manifest_parser_accepts_lf_and_crlf" in cuda_test
assert "offline feasibility run.in must end with its rpmd_ja fit_samples_covariance command" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "offline feasibility run.in must not contain production run or ensemble commands" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "test_fit_command_rejects_any_command_after_offline_fit" in source("tests/rpmd_ja_feasibility_preflight_test.py")
assert "test_frozen_fit_input_fingerprints_reject_mutation" in integration_test
assert "BUILD_RPMD_JA_CUDA_REGRESSIONS" in source("CMakeLists.txt") and \
       "rpmd_ja_fit_prepare_integration_cuda_test" in source("CMakeLists.txt")
assert 'COMMAND rpmd_ja_native_fit_cuda_test --fixture-dir "${CMAKE_CURRENT_SOURCE_DIR}/tests/data"' in source("CMakeLists.txt")
assert 'native_fit_regression_command(regression, fixture_directory)' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert 'covariance_fit_test_fixture_path("ja_reference_outer128_766cuts_stationarity_v1.qp_fixture.txt")' in cuda_test
assert 'options.kernel_table=kernel' in cuda_test and '"unused-before-validation"' not in cuda_test
assert "normalized_response_matrix_delta(left.development.predicted_matrix,right.development.predicted_matrix" in fit
assert "test_regularization_response_delta_uses_full_predicted_matrix" in cuda_test
assert "dev_response_error_delta_aux" in fit and "predicted_response_matrix_relative_delta" in fit
assert "cmake --build build --config Debug --target gpumd rpmd_ja_native_fit_cuda_test rpmd_ja_fit_prepare_integration_cuda_test" in \
       source("doc/gpumd/input_parameters/rpmd_ja.rst")
assert "serialize_m1_cg_basis" in fit and "M1_basis_sidecar" in fit
assert "M1_basis_fingerprint_fnv64" in fit and "M1_minimum_support_count" in fit
assert "parameter_family" in fit and "witness_format_version" in fit
cg_reader = fit[fit.index("CGWitnessSnapshot read_cg_witness"):fit.index("bool preserve_candidate_package_on_full_spd_failure")]
assert "count%6!=0" not in cg_reader
assert "test_m1_covariance_calibration_uses_only_base_prefix_in_replay" in cuda_test
assert "M1_basis_replay_status PASS" in cuda_test and "tamper_rejected" in cuda_test
checkpoint = source("src/measure/rpmd_ja_covariance_checkpoint.h")
assert '"RPJACKPT4\\n"' in checkpoint
for field in ("performance_window_curvature_builds", "performance_window_tau",
              "performance_window_curvature_seconds", "performance_window_reference_gradient_seconds",
              "objective_evaluation_seconds", "full_gradient_seconds", "factorization_seconds",
              "analytic_curvature_seconds"):
    assert field in checkpoint
    assert field in fit
performance_stop_check = fit[fit.index("auto check_curvature_performance_stop"):fit.index(
    "persist_accepted_point();\n    while(result.iterations<max_iterations")]
optimizer = fit[fit.index("while(result.iterations<max_iterations"):fit.index("if(!result.converged&&result.status==\"OPTIMIZING\")")]
assert "finish_curvature_performance_window" in performance_stop_check
assert optimizer.index("covariance_barrier_stage_pass(current") < optimizer.index("if(check_curvature_performance_stop())")
assert "NUMERICAL_INCONCLUSIVE_PERFORMANCE_STOP" in fit
assert "NUMERICAL_INCONCLUSIVE_PERFORMANCE_STOP" in fit[fit.index("bool covariance_incomplete_candidate_status"):]
assert "candidate_performance_stop_rule" in fit and "heuristic_only=1" in fit
candidate_cost_trace_start = fit.rindex('trace<<"MEASURED_CANDIDATE_COST family "')
candidate_cost_trace = fit[candidate_cost_trace_start:fit.index("trace.flush();", candidate_cost_trace_start)]
for field in ("candidate.fit.status", "objective_evaluation_seconds", "full_gradient_seconds",
              "factorization_seconds", "curvature_seconds", "response_seconds"):
    assert field in candidate_cost_trace, field
assert "const bool numerical_inconclusive=m0_attempted>0||m1_phase_started;" in fit
assert "performance_stop_rule" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "feasibility_manifest_performance_stop_contract" in fit
assert "resume_state?resume_state->theta:result.theta_start" in fit
assert "start_metrics_point " in fit and "RESTORED_ACCEPTED_POINT" in fit
assert "atomic_replace_file(temporary,path)" in checkpoint
assert "covariance checkpoint checksum mismatch" in checkpoint
assert "covariance checkpoint identity mismatch" in checkpoint
assert "trust_radius" in checkpoint and "bfgs_inverse" in checkpoint and "phase_gradients" in checkpoint
assert "test_covariance_checkpoint_round_trip_and_corruption_rejection" in cuda_test
assert '"RPJAOUTER5\\n"' in checkpoint
assert '"cg_witness_attempts "<<state.cg_witness_attempts' in checkpoint
assert 'scalar("cg_witness_attempts",state.cg_witness_attempts)' in checkpoint
assert "next_cg_witness_attempt" in checkpoint
assert "response_replay_required" in checkpoint and "response_replay_required=score_complete" in fit
assert "controller_candidate_pass_for_generation" in checkpoint
assert "controller_has_committed_b_gate_for_resume" in checkpoint and "controller_has_committed_b_gate_for_resume(controller_journal,controller_resume)" in fit
assert "commit_final_development_generation(controller_journal,has_m1_candidates" in fit
assert "candidate.score_generation=\"FINAL_M0_M1\"" in checkpoint
assert "begin_controller_selection_freeze(" in fit and "commit_controller_selection_freeze(" in fit
assert "test_c_resume_keeps_b_gate_committed_and_accounts_repeated_interruptions_to_c" in source("tests/rpmd_ja_covariance_checkpoint_test.cpp")
assert "test_score_generation_controls_selection_eligibility" in source("tests/rpmd_ja_covariance_checkpoint_test.cpp")
assert "test_scheduler_freezes_m0_after_skip_or_m1_evaluation" in source("tests/rpmd_ja_covariance_checkpoint_test.cpp")
assert "record->b_gate_pass" in fit and '"FINAL_M0_M1"' in fit
assert "candidate.development_scored=false;" in fit
assert "if(saved_score!=controller_journal.candidates.end()&&saved_score->score_complete" not in fit
assert 'if(label=="A0")' in fit and 'if(label=="A0"&&fit.converged)' not in fit
assert "rebuild_restored_a0_baseline(fit,baseline,branch_graph" in fit
assert "test_nonconverged_a0_baseline_restore_rebuilds_factors" in cuda_test
assert 'development_response_sha256' in checkpoint and 'published_trace_sha256' in checkpoint
assert 'verify_published_feasibility_artifacts(options,controller_journal)' in fit
assert 'read_blind_response_checkpoint(options.output_path+".development_checkpoint.txt"' in fit
assert 'read_blind_response_checkpoint(options.output_path+".blind_checkpoint.txt"' in fit
assert 'development response theta differs from the frozen candidate' in fit
assert 'validate_blind_response_snapshot_dimensions(response,theta)' in fit
assert 'test_frozen_response_snapshot_round_trip_and_identity_rejection' in cuda_test
assert 'publish_verified_prepared_artifacts(options.output_path,publication_stage_output_path' in fit
assert 'prepared stage artifact disappeared before commit' in fit
assert 'prepared stage bytes differ from the journal digest' in fit
assert 'FEASIBILITY_PUBLISH_RETRYABLE: staged artifact atomic replace failed' in fit
assert 'FEASIBILITY_PUBLISH_RETRYABLE: fit diagnostics replace failed' in fit
assert 'retryable_publication_io_failure(detail)' in fit
assert 'if(!retryable_publication_io)' in fit
assert 'test_publication_io_classification_and_two_rename_recovery' in source('tests/rpmd_ja_native_fit_cuda_test.cu')
assert 'committed fit diagnostics staging file is missing' in fit
assert 'committed fit diagnostics staging digest changed' in fit
assert 'verify_publishing_artifact_pair(options.output_path,staged' in fit
assert 'frozen candidate identity must reject a mutated common probe' in source('tests/rpmd_ja_covariance_checkpoint_test.cpp')
assert 'TEST_PUBLICATION_INTERRUPTED_BETWEEN_RENAMES' in integration_test
assert 'test_resume_publishing_accepts_only_digest_bound_two_rename_state' in source('tests/rpmd_ja_feasibility_preflight_test.py')
assert 'selection_state NOT_FROZEN' not in source('doc/gpumd/input_parameters/rpmd_ja.rst')
assert 'RPJAOUTER5' in source('doc/gpumd/input_parameters/rpmd_ja.rst')
assert "test_native_covariance_mid_a0_resume_preserves_original_start_and_stagnation_window" in cuda_test
assert "resume_state?resume_state->theta_start" in fit
assert "saved.stagnation_progress_attempts_in_window=progress_attempts_in_window" in fit
assert "saved.same_point_curvature_rejections=" in fit
assert "reconcile_controller_operation_budget" in checkpoint and "operation_start_epoch_seconds" in checkpoint
assert "run_controller_operation" in fit and "reconcile_controller_operation_budget(controller_journal" in fit
assert "test_preflight_os_lock_rejects_live_owner_and_recovers_after_kill" in \
       source("tests/rpmd_ja_feasibility_preflight_test.py")
assert 'rpmd_ja_process_lock::try_lock(lock_file_)' in source("src/measure/rpmd_ja_fit.cu")
assert 'rpmd_ja_process_lock::try_lock(lock)' in fit
assert 'rpmd_ja_file_io::remove_file(lock_path)' not in fit and 'rpmd_ja_file_io::remove_file(lock_path_)' not in \
       source("src/measure/rpmd_ja_fit.cu")
assert "reconcile_controller_resume_budget(ControllerJournal& journal,AcceptedPoint& solver_state,\n  const rpmd_ja_feasibility::Limits& limits" in checkpoint
prepare_header = source("src/measure/rpmd_ja_qnep_prepare.cuh")
prepare_impl = source("src/measure/rpmd_ja_qnep_prepare.cu")
native_test = source("tests/rpmd_ja_native_fit_cuda_test.cu")
for prepare_source in (prepare_header, prepare_impl, native_test):
    assert "RpmdJAPrepareResourceObserver" in prepare_source
    assert "RpmdJAPrepareAllocationObserver" in prepare_source
assert "const RpmdJAPrepareResourceObserver& resource_observer = {},\n  const RpmdJAPrepareAllocationObserver& allocation_observer = {});" in prepare_header
assert "const RpmdJAPrepareResourceObserver& resource_observer,\n                                    const RpmdJAPrepareAllocationObserver& allocation_observer)" in prepare_impl
assert "const RpmdJAPrepareResourceObserver&,\n                                    const RpmdJAPrepareAllocationObserver&) {}" in native_test
assert "next_cg_witness_attempt(controller_journal)" in fit
assert fit.index("next_cg_witness_attempt(controller_journal)") < fit.index("persist_controller_journal();", fit.index("next_cg_witness_attempt(controller_journal)"))
assert "feasibility_cg_witness_attempt=0" not in fit
assert "const std::uint64_t package_staging_peak_bytes=publication_payload_restored?0:estimated_package_bytes" in fit
assert "const std::uint64_t reference_staging_peak_bytes=publication_payload_restored?0:" in fit
assert "persistent_disk_bytes+=persistent_reference_bytes" in fit
assert "checked_multiplicity_bytes(raw_bytes,2)" not in fit
assert "host_graph_m0_bytes+host_graph_m1_bytes" in fit
assert "CONSERVATIVE_KNOWN_HOST_OBJECTS_NOT_COMPLETE" in fit
assert "CONSERVATIVE_KNOWN_DISK_OBJECTS_NOT_COMPLETE" in fit
assert "write_controller_journal(controller_checkpoint_path,controller_journal)" in fit
assert "read_controller_journal(controller_checkpoint_path,manifest_identity)" in fit
assert "save_controller_candidate" in fit and "restore_branch" in fit
assert "resume_point,&checkpoint_metadata,checkpoint_sink" in fit
assert "reconcile_controller_resume_budget(controller_journal" in fit
assert "OUTER_JOURNAL_RECONCILED" in fit
assert "resume_state->elapsed_seconds>budget->elapsed_seconds" in fit
assert "cumulative_elapsed_budget_on_resume(" not in fit
assert "preserve_later_family_cursor" in checkpoint
run_family_start = fit.index("auto run_family=")
run_family_end = fit.index("run_family(rpmd_ja_feasibility::Family::M0", run_family_start)
run_family = fit[run_family_start:run_family_end]
assert "candidate_restored=controller_resume&&candidate_state_file.good()" in run_family
assert "candidate_resume_cursor=controller_resume&&controller_journal.active_candidate_started" in run_family
assert "preserve_later_resume_cursor=rpmd_ja_covariance_checkpoint::preserve_later_family_cursor(" in run_family
assert "!preserve_later_resume_cursor&&(active_candidate_timer||controller_journal.branch==candidate_label)" in run_family
assert "!replay_prior_family&&!preserve_later_resume_cursor&&index+1" in run_family
assert "if(candidate_restored)run_controller_operation([&](){candidate.fit=restore_branch" in run_family
assert "candidate_restored&&!candidate_score_was_complete&&!candidate_resume_cursor" in run_family
assert "else candidate.fit=solve_branch" in run_family
branch_checkpoint_start = fit.index("auto checkpoint_branch=")
branch_restore_start = fit.index("auto restore_branch=", branch_checkpoint_start)
branch_checkpoint = fit[branch_checkpoint_start:branch_restore_start]
assert branch_checkpoint.index("write_covariance_state(") < branch_checkpoint.index("save_controller_candidate(")
assert branch_checkpoint.index("save_controller_candidate(") < branch_checkpoint.index("remove_file(solver_checkpoint_path)")
branch_restore_end = fit.index("auto reject_unexpected_stop=", branch_restore_start)
branch_restore = fit[branch_restore_start:branch_restore_end]
assert "const bool active_cursor=controller_resume&&controller_journal.active_candidate_started" in branch_restore
assert "restore_covariance_fit_state(path" in branch_restore
assert "if(record==controller_journal.candidates.end()||!record->fit_complete)" in branch_restore
assert "matching completed solver checkpoint" in branch_restore
assert "score_generation" in checkpoint
rescore_start = fit.index("auto rescore_candidates=")
rescore_end = fit.index("try{collect_candidate_soft_modes", rescore_start)
assert '"COMMON_PROBE_RESCORING",generation' in fit[rescore_start:rescore_end]
assert "common_probe_stage=stage" in fit and 'freeze_common_probes("B_GATE")' in fit and "FINAL_M0_M1" in fit
assert "controller_journal.training_statistics_frozen" in fit
assert "b_gate_probe_set_restored=controller_resume&&controller_journal.common_probe_set_frozen" in fit
assert "if(!b_gate_probe_set_restored){" in fit
assert "controller_has_committed_b_gate_for_resume(controller_journal,controller_resume)" in fit
assert "if(!resuming_c_or_freeze_cursor){" in fit
assert "if(!final_probe_state_restored){" in fit
assert 'controller_resume&&controller_journal.common_probe_stage=="B_GATE")collect_candidate_soft_modes(0,true)' in fit
m1_dispatch_start = fit.index("if(frozen_m1_selection||(!frozen_selection_resume&&", fit.index("bool m1_phase_started"))
m1_dispatch_end = fit.index("run_family(rpmd_ja_feasibility::Family::M1", m1_dispatch_start)
assert 'controller_journal.branch="C0"' not in fit[m1_dispatch_start:m1_dispatch_end]
assert "training_channel_names" in fit and "training_block_length" in fit
budget_observer_start = fit.index("auto observe_controller_budget=")
budget_observer_end = fit.index("auto save_controller_candidate=", budget_observer_start)
budget_observer = fit[budget_observer_start:budget_observer_end]
assert "active_candidate_timer&&" not in budget_observer
assert ">=30.0)" in budget_observer and "persist_controller_journal();" in budget_observer
bgate_accounting = fit.index('trace<<"COMMON_PROBE_B_GATE_ACCOUNTING status "')
family_dispatch = fit.index("should_run_structured_family", bgate_accounting)
assert fit.index('controller_journal.stage="C"', bgate_accounting) < family_dispatch
assert "cost baseline bytes changed after launcher preflight" in fit
assert "cost_baseline_resolved_path" in fit and "cost_baseline_resolved_path" in source("scripts/rpmd_ja_feasibility_preflight.py")
assert '"PUBLISHING", "PUBLISHED"' in source("scripts/rpmd_ja_feasibility_preflight.py")
assert "resume checkpoint stage, family, branch, and candidate cursor disagree" in \
    source("scripts/rpmd_ja_feasibility_preflight.py")
assert "no_existing_outputs(output, resume=resume_mode, checkpoint_path=checkpoint_path," in \
    source("scripts/rpmd_ja_feasibility_preflight.py")
assert "response_consumer_statistics_ready(response)" in fit
assert 'write_response_consumer_statistics(trace,"BLIND",response)' in fit
assert "fail_first_probe" in cuda_test and "write_response_consumer_statistics(diagnostic,consumer,incomplete)" in cuda_test
sample_provenance_verifier = fit[fit.index("auto verify_sample_provenance="):fit.index('verify_sample_provenance("training"')]
assert 'expected_model!="NOT_VERIFIABLE"' in sample_provenance_verifier
assert 'reported_model!="NOT_VERIFIABLE"' in sample_provenance_verifier
assert 'qraw.model_sha256' in fit and 'sha256_file_digest(force.primary_nep_model_path())' in fit
assert 'test_concrete_sidecar_model_cannot_be_downgraded_to_unknown_manifest' in source("tests/rpmd_ja_feasibility_preflight_test.py")
assert 'test_partial_sidecar_digest_cannot_be_downgraded_to_unknown_manifest' in source("tests/rpmd_ja_feasibility_preflight_test.py")

blind_stat_start = fit.index('const auto manifest=verify_feasibility_manifest(feasibility_manifest_fields,spool_path,')
blind_stat_end = fit.index('auto blind_cg=', blind_stat_start)
blind_statistics = fit[blind_stat_start:blind_stat_end]
assert 'collect_fixed_r0_admission_channels(in,header,static_cast<std::uint64_t>(train)' in blind_statistics
assert 'collect_fixed_r0_admission_channels(blind,blind_header,blind_header.frame_count,structured_graph' in blind_statistics
assert 'assess_fixed_r0_means(blind_training_channels' in blind_statistics
assert 'admission_statistics.training_block_plan' in blind_statistics
assert 'no_blind_block_reselection 1 no_reselection 1' in blind_statistics
assert 'plan_training_blocks(' not in blind_statistics
assert 'estimate_fixed_r0_drift_storage_bytes' in blind_statistics
assert blind_statistics.index('estimate_fixed_r0_drift_storage_bytes') < blind_statistics.index('collect_fixed_r0_admission_channels(in,header')
assert 'test_blind_fixed_r0_admission_reuses_training_channels_and_block_plan' in cuda_test
assert 'test_admission_statistics_storage_peak_boundary' in cuda_test

family_start = fit.index('auto run_family=[&]')
family_end = fit.index('run_family(rpmd_ja_feasibility::Family::M0,graph', family_start)
family_controller = fit[family_start:family_end]
assert 'predict_remaining_cost(feasibility_budget,feasibility_limits' in family_controller
assert family_controller.index('for(unsigned index=0;index<feasibility_lambda_factors.size();++index)') < \
    family_controller.index('predict_remaining_cost(feasibility_budget,feasibility_limits')
assert 'remaining_candidates=phase==rpmd_ja_feasibility::Phase::B?6u:3u' not in family_controller
assert 'remaining_work.calls[rpmd_ja_feasibility::resource_stage_index(rpmd_ja_feasibility::ResourceStage::FIT)]=1' in family_controller
assert 'INCONCLUSIVE_NO_STAGE_BASELINE' in family_controller and 'unknown_is_not_zero_cost 1' in family_controller
assert 'INSUFFICIENT_REMAINING_BUDGET' in family_controller and 'hard_budget_still_authoritative 1' in family_controller
assert 'estimate_resource_peaks(allocation_plan)' in fit
assert 'resource_capacity_gate(allocation_peak' in fit
assert 'estimate_workspace_growth_peak_bytes(current_workspace_bytes.data()' in fit
assert 'workspace_growth_peak_bytes' in fit and 'solver_growth_peak_bytes' in fit
assert 'system_workspace->solver_work.capacity()' in fit
assert 'ResourcePlan acceptance_resource_plan' in fit and 'simultaneous_prepare_readback_matrices_included 1' in fit
assert 'estimate_resource_peaks(acceptance_resource_plan)' in fit and 'resource_capacity_gate(acceptance_resource_peak' in fit
assert 'device_known=false' in fit[fit.index('ResourcePlan acceptance_resource_plan'):fit.index('const auto acceptance_resource_peak')]
assert 'prepare_gpu_lower_bound_bytes' in fit and 'COMPLETE_ESTIMATE' in fit
assert 'prepare_library_workspace_query exact_before_A0 1' in fit
assert 'query_rpmd_ja_qnep_prepare_workspace_sizes(n)' in fit
assert 'fit_system_auxiliary_bytes' in fit and 'fit_gpu_lower_bound_bytes' in fit
assert 'ResourceStage::FIT' in fit and 'std::max(prepare_gpu_lower_bound,fit_gpu_lower_bound)' in fit
assert 'BASELINE_PREPARE_FIT_PACKAGE_AND_CURVATURE_WORKSPACES' in fit
assert 'observe_prepare_budget("before_prepare")' in fit
assert 'acceptance_resource_gate==rpmd_ja_feasibility::ResourceGateResult::INCONCLUSIVE' in fit
assert 'estimate_resource_peaks(resource_plan)' in fit and 'workspace.reserve_curvature(' in fit
assert 'test_covariance_fit_workspace_grows_reuses_and_releases' in cuda_test
assert 'test_covariance_fit_workspace_is_reused_by_sequential_system_curvature_builds' in cuda_test

system_workspace_start = fit.index('struct DeviceCovarianceSystemWorkspace')
system_workspace_end = fit.index('struct DeviceCovarianceSystem\n', system_workspace_start)
system_workspace = fit[system_workspace_start:system_workspace_end]
assert 'DeviceBuffer<double> full,compact,shifted' in system_workspace
assert 'cublasHandle_t blas=nullptr;cusolverDnHandle_t solver=nullptr;' in system_workspace
assert 'std::string static_graph_identity' in system_workspace
assert 'const DeviceBaseline* active_baseline=nullptr;std::uint64_t active_baseline_generation=0' in system_workspace
assert 'std::string factor_graph_identity;const DeviceBaseline* factor_baseline=nullptr' in system_workspace
assert 'allocation_count=0,static_upload_count=0,static_upload_bytes=0,factorization_count=0' in system_workspace
assert 'factor_cache_valid=false' in system_workspace and 'factor_epsilon' in system_workspace
assert 'system_workspace->static_graph_identity!=graph_identity' in fit
assert 'covariance_graph_content_sha256(const Graph& graph)' in fit
assert 'system_workspace->active_baseline_generation!=baseline.identity_generation' in fit
assert 'system_workspace->factor_cache_valid=false' in fit
assert 'system_workspace->factor_baseline_generation==baseline.identity_generation' in fit
assert 'system_workspace->factor_epsilon==epsilon&&system_workspace->factor_theta==parameters' in fit
assert fit.count('&feasibility_device_workspace') >= 4
assert 'test_covariance_device_workspace_reuses_buffers_static_graph_and_same_point_factors' in cuda_test
device_workspace_test = cuda_test[cuda_test.index('void test_covariance_device_workspace_reuses_buffers_static_graph_and_same_point_factors'):cuda_test.index('void test_covariance_factor_identity_recovery')]
assert 'const Graph*' in device_workspace_test and '&m0,&theta0,0.01},{&m0,&theta0,0.01}' in device_workspace_test
assert '&m1,&theta1,0.02},{&m0,&theta0,0.01}' in device_workspace_test
assert 'shared.static_upload_count==3' in device_workspace_test
assert 'shared.factorization_count==factorizations_before' in device_workspace_test
assert 'shared.allocation_count<independent_allocations' in device_workspace_test
assert 'check_close(shared_values[i][0],values[0],1e-12)' in device_workspace_test
assert 'same_address.local_transform[0]=0.75' in device_workspace_test
assert 'changed_baseline_value=evaluate_changed_baseline(changed_baseline,shared)' in device_workspace_test

a0_reuse = source('src/measure/rpmd_ja_a0_reuse.h')
preflight = source('scripts/rpmd_ja_feasibility_preflight.py')
assert 'RPJAA0PACKAGE1' in a0_reuse and 'RPJAA0REUSE2' in a0_reuse
assert 'write_package_atomic' in a0_reuse and 'atomic_commit_new_file' in a0_reuse
assert '_parse_a0_reuse_package(package_bytes)' in preflight
assert 'source_manifest_bytes, state_bytes, record_bytes =' in preflight
assert 'state_fields.get("state_format_version") != "4"' in preflight
assert 'a0_reuse_verified_state_bytes' in fit and 'verify_imported_a0_fit(' in fit
assert 'options.epsilon,budget,&feasibility_limits,&feasibility_device_workspace,true)' in fit
assert 'current_import_validation_seconds' in a0_reuse and 'current_response_seconds' in a0_reuse
assert 'verify_imported_a0_parameter_map' in fit and 'start_tau' in fit
assert 'AfterManifest' in a0_reuse and 'AfterState' in a0_reuse and 'AfterRecord' in a0_reuse

prepare_source = source('src/measure/rpmd_ja_qnep_prepare.cu')
prepare_header = source('src/measure/rpmd_ja_qnep_prepare.cuh')
assert 'cusolverDnDpotrf_bufferSize' in prepare_source
assert 'cusolverDnDgesvd_bufferSize' in prepare_source
assert 'const std::size_t svd_live_bytes=6ULL' in prepare_source
assert 'certificate_peak,svd_peak' in prepare_source
assert 'resource_observer(plan)' in prepare_source
assert 'allocation_observer(where,static_cast<std::uint64_t>(bytes))' in prepare_source
assert 'query_solver_workspace_elements(solver,r,query_matrix)' in prepare_source
assert 'allocate_gpu(reinterpret_cast<void**>(&query_matrix),sizeof(double),"allocate scalar POTRF query matrix")' in prepare_source
assert 'RpmdJAPrepareWorkspaceSizes query_rpmd_ja_qnep_prepare_workspace_sizes(int atom_count)' in prepare_header
assert 'RpmdJAPrepareResourcePlan' in prepare_header
assert 'RESOURCE_PREPARE_EXACT_GPU_PLAN' in fit
assert 'RESOURCE_PREPARE_ALLOCATION_REQUEST' in fit
prepare_cuda_test = source('tests/rpmd_ja_fit_prepare_integration_cuda_test.cu')
assert 'prepare_resource_plan_seen' in prepare_cuda_test
assert 'prepare_allocation_requests' in prepare_cuda_test
assert 'allocate SVD workspace' in prepare_cuda_test
assert 'MEASURED_STAGE_COST stage ' in fit
for timer_name, stage in (('blind_stage_timer', 'BLIND'), ('package_write_timer', 'PACKAGE_WRITE'),
                          ('prepare_stage_timer', 'PREPARE'), ('readback_stage_timer', 'READBACK')):
    assert f'FeasibilityStageTimer {timer_name}(trace,"{stage}",' in fit
assert 'source CURRENT_RUN_MEASUREMENT hard_wall_budget_authoritative 1' in fit
for stage in ('BLIND', 'PACKAGE_WRITE', 'PREPARE', 'READBACK'):
    assert f'stage_cost_recorder(rpmd_ja_feasibility::ResourceStage::{stage})' in fit
assert 'persist_measured_costs();' in fit
assert 'RESOURCE_LIFETIME stage ' in fit and 'observed_live_disk_peak_bytes' in fit and \
       'observed_min_host_available_bytes' in fit and 'host_allocator_peak_complete 0' in fit

material_eval = source('tests/rpmd_ja_material_evaluate_test.py')
assert 'test_native_current_export_then_rpjam2_evaluate_fixture' in material_eval
assert 'export.export_trajectory(spec_path)' in material_eval
assert 'evaluate.evaluate_manifest(manifest_path)' in material_eval
assert 'feasibility_curvature_workspace' in fit and 'system.fit_workspace->curvature_peak_bytes' in fit

workspace = source('src/measure/rpmd_ja_cuda_workspace.cuh')
assert '#include <algorithm>' in workspace
measurement = source('src/measure/hac.cu')
assert '# segment_metadata_version 2' in measurement
assert 'reference_package_sha256' in measurement
reference_header = source('src/measure/rpmd_ja_reference.cuh')
reference_reader = source('src/measure/rpmd_ja_reference.cu')
assert 'source_file_sha256' in reference_header
assert 'rpmd_ja_sha256::Digest digest' in reference_reader
assert 'result.source_file_sha256 = digest.finish_hex()' in reference_reader
assert 'test_covariance_fit_workspace_is_reused_by_sequential_system_curvature_builds' in cuda_test

a0_write = a0_reuse[a0_reuse.index('inline void write_package_atomic'):]
assert a0_write.count('const std::string bytes{std::istreambuf_iterator<char>(') == 2
assert 'const std::string bytes(std::istreambuf_iterator<char>(' not in a0_write

qraw_identity_start = fit.index('QrawIdentity validate_qraw_identity(')
qraw_identity_end = fit.index('\nvoid read_frame(', qraw_identity_start)
qraw_identity = fit[qraw_identity_start:qraw_identity_end]
assert 'QrawIdentity identity;' in qraw_identity
for field in ('data_offset', 'k_offset', 'version', 'model', 'config', 'charge', 'kspace',
              'mesh', 'active_mesh', 'model_sha256'):
    assert f'identity.{field}=' in qraw_identity
assert 'return {data,k_offset,version,model,config,charge,kspace,mesh,active_mesh,active_model_sha256};' not in qraw_identity

prepare_svd_allocations = prepare_source[prepare_source.index('allocate_gpu(reinterpret_cast<void**>(&sm)'):
    prepare_source.index('double tile_error[3]={};', prepare_source.index('allocate_gpu(reinterpret_cast<void**>(&sm)'))]
assert 'static_cast<std::size_t>(tile_size)*tile_size*sizeof(double)' in prepare_svd_allocations
assert 'maxdim' not in prepare_svd_allocations

assert 'value.verification_matrix_actions' in fit and 'value.refinement_matrix_actions' in fit
assert 'value.verification_actions' not in fit and 'value.refinement_actions' not in fit
host_required_declaration = fit.index('std::uint64_t host_required=0;')
storage_observer_declaration = fit.index('auto observe_storage_lifetime=[&]')
host_required_initialization = fit.index('host_required=host_safety_margin_bytes;')
first_storage_observation = fit.index('observe_storage_lifetime("BEFORE_PACKAGE_WRITE"')
assert host_required_declaration < storage_observer_declaration < host_required_initialization < first_storage_observation
assert '" total_common_rank "<<sample_statistics.probes.size()' in fit

resume_budget_start = fit.index('if(resume_point&&budget){')
resume_budget_end = fit.index('operation_start_epoch_seconds=', resume_budget_start)
resume_budget_copy = fit[resume_budget_start:resume_budget_end]
assert 'for(std::size_t phase=0;phase<restored.phase_gradients.size();++phase)' in resume_budget_copy
assert 'restored.phase_gradients=budget->phase_gradients' not in resume_budget_copy
assert 'restored.phase_seconds=budget->phase_seconds' not in resume_budget_copy
assert 'development_system.evaluate_factors(fit.theta,options.epsilon,development_logdet,development_shifted_logdet,nullptr)' in fit
for expression in (
    'system.evaluate_factors(accepted,epsilon,logdet,shifted_logdet,nullptr)',
    'system.evaluate_factors(candidate,epsilon,logdet,shifted_logdet,nullptr)',
    'system.evaluate_factors(candidate,1.0,logdet,shifted_logdet,nullptr)',
):
    assert expression in cuda_test
