import subprocess
import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import rpmd_ja_cpp_test_support as cpp_test_support
else:
    import rpmd_ja_cpp_test_support as cpp_test_support


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/rpmd_ja_covariance_checkpoint_test.cpp"


class CovarianceCheckpointRuntimeTest(unittest.TestCase):
    def test_scheduler_finalizes_common_probe_generation_before_freezing(self):
        checkpoint = (ROOT / "src/measure/rpmd_ja_covariance_checkpoint.h").read_text(encoding="utf-8")
        native_fit = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        runtime_source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("commit_final_development_generation(controller_journal,has_m1_candidates", native_fit)
        self.assertIn("candidate.score_generation=\"FINAL_M0_M1\"", checkpoint)
        self.assertIn("begin_controller_selection_freeze(", native_fit)
        self.assertIn("controller_journal,selected_branch,frozen.candidate_index", native_fit)
        self.assertIn("commit_controller_selection_freeze(controller_journal,development_response_sha256)", native_fit)
        self.assertIn("test_scheduler_freezes_m0_after_skip_or_m1_evaluation", runtime_source)
        self.assertIn("m1_tested.branch==\"C2\"&&m1_tested.family==\"M1\"", runtime_source)
        self.assertIn("restored_m1.branch==\"C2\"&&restored_m1.family==\"M1\"", runtime_source)

    def test_solver_reconciles_interrupted_checkpoint_without_double_charging(self):
        checkpoint = (ROOT / "src/measure/rpmd_ja_covariance_checkpoint.h").read_text(encoding="utf-8")
        native_fit = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        self.assertIn("elapsed_budget_on_resume", checkpoint)
        self.assertIn("resume_epoch_seconds-state.operation_start_epoch_seconds", checkpoint)
        self.assertIn("reconcile_controller_resume_budget", checkpoint)
        self.assertNotIn("Limits{}.max_exact_gradients", checkpoint)
        self.assertIn("const rpmd_ja_feasibility::Limits& limits", checkpoint)
        self.assertIn("restricted_limits.max_exact_gradients=4", SOURCE.read_text(encoding="utf-8"))
        self.assertIn("reconcile_controller_resume_budget(controller_journal", native_fit)
        self.assertIn("active_solver_checkpoint,feasibility_limits", native_fit)
        self.assertNotIn("cumulative_elapsed_budget_on_resume(", native_fit)
        self.assertIn("resume_state->elapsed_seconds>budget->elapsed_seconds", native_fit)
        self.assertIn("OUTER_JOURNAL_RECONCILED", native_fit)
        self.assertIn("feasibility_budget.start_seconds=budget_now-feasibility_budget.elapsed_seconds", native_fit)
        self.assertIn("saved.operation_in_progress=operation_in_progress", native_fit)
        self.assertIn("B/C active-fit recovery must reconcile monotonic wall, phase, gradient, and candidate budgets",
                      SOURCE.read_text(encoding="utf-8"))

    def test_blind_and_publication_recovery_keep_the_resumed_operation_open(self):
        checkpoint = (ROOT / "src/measure/rpmd_ja_covariance_checkpoint.h").read_text(encoding="utf-8")
        native_fit = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        runtime_source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('journal.selection_state=="BLIND_OPENED"||journal.selection_state=="PUBLISHING"', checkpoint)
        self.assertIn("journal.operation_start_epoch_seconds=selection_operation_resumes?resume_epoch_seconds:0.0", checkpoint)
        self.assertIn("preserve_interrupted_selection_operation", native_fit)
        self.assertIn("test_blind_and_publication_resume_keep_in_progress_across_repeated_interruptions", runtime_source)
        self.assertIn("read_controller_journal(path,state.identity_sha256)", runtime_source)

    def test_frozen_resume_restores_only_the_selected_fit_and_m0_only_replays_missing_response(self):
        native_fit = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        self.assertIn("if(frozen_selection_resume&&!a0_restored)", native_fit)
        self.assertIn("auto restore_frozen_candidate=", native_fit)
        self.assertIn("candidate.development=restored_development_response.response", native_fit)
        self.assertIn("if(controller_journal.frozen_candidate_branch[0]=='B')\n        restore_frozen_candidate", native_fit)
        self.assertIn("if(frozen_selection_resume)restore_frozen_candidate(structured_graph", native_fit)
        self.assertIn("development_response_needs_replay(only_missing_response", native_fit)
        self.assertIn("rescore_candidates(rpmd_ja_feasibility::Phase::FREEZE,\"FINAL_M0_M1\",true)", native_fit)

    def test_performance_window_and_component_costs_survive_solver_checkpoint(self):
        checkpoint = (ROOT / "src/measure/rpmd_ja_covariance_checkpoint.h").read_text(encoding="utf-8")
        native_fit = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        policy = (ROOT / "src/measure/rpmd_ja_feasibility_policy.cuh").read_text(encoding="utf-8")
        for field in ("performance_window_curvature_builds", "performance_window_tau",
                      "performance_window_curvature_seconds", "performance_window_reference_gradient_seconds",
                      "objective_evaluation_seconds", "full_gradient_seconds", "factorization_seconds",
                      "analytic_curvature_seconds"):
            self.assertIn(field, checkpoint)
            self.assertIn(field, native_fit)
        stop_check = native_fit[native_fit.index("auto check_curvature_performance_stop"):
                                native_fit.index("persist_accepted_point();\n    while(result.iterations<max_iterations")]
        optimizer = native_fit[native_fit.index("while(result.iterations<max_iterations"):
                               native_fit.index("if(!result.converged&&result.status==\"OPTIMIZING\")")]
        self.assertIn("finish_curvature_performance_window", stop_check)
        self.assertLess(optimizer.index("covariance_barrier_stage_pass(current"),
                        optimizer.index("if(check_curvature_performance_stop())"))
        self.assertIn("NUMERICAL_INCONCLUSIVE_PERFORMANCE_STOP", native_fit)
        self.assertIn("candidate_performance_stop_rule", policy)

    def test_cpu_checkpoint_round_trip_corruption_and_semantic_rejection(self):
        compiler = cpp_test_support.find_cpp_compiler()
        if compiler is None:
            self.skipTest("C++ runtime test skipped: c++, g++, clang++, and cl are unavailable")
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            executable = Path(temporary) / "rpmd_ja_covariance_checkpoint_test.exe"
            result = subprocess.run(cpp_test_support.compile_command(compiler, 14, SOURCE, executable),
                                    cwd=temporary, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(executable)], cwd=temporary, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
