import importlib.util
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import rpmd_ja_cpp_test_support as cpp_test_support
else:
    import rpmd_ja_cpp_test_support as cpp_test_support

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rpmd_ja_feasibility_preflight.py"
CPP_RECEIPT_TEST = ROOT / "tests/rpmd_ja_preflight_receipt_test.cpp"
CPP_A0_REUSE_EXPORT_TEST = ROOT / "tests/rpmd_ja_a0_reuse_export_test.cpp"
SPEC = importlib.util.spec_from_file_location("rpmd_ja_feasibility_preflight", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class FeasibilityPreflightTest(unittest.TestCase):
    OUTER5_REQUIRED_FIELDS = (
        'frozen_candidate_branch ""\nfrozen_candidate_index 0\nfrozen_candidate_lambda 0\n'
        'frozen_candidate_fit_state_sha256 ""\nfrozen_candidate_identity_sha256 ""\n'
        'development_response_sha256 ""\nblind_response_sha256 ""\n'
        'publication_stage_output_sha256 ""\npublication_stage_sidecar_sha256 ""\n'
        'published_output_sha256 ""\npublished_sidecar_sha256 ""\npublished_fit_sha256 ""\n'
        'published_trace_sha256 ""\ncommon_probe_set_frozen 0\ncommon_probe_count 0\ncandidate_count 0\n'
    )

    @staticmethod
    def frozen_outer5_payload(selection_state: str) -> bytes:
        run_identity, fit_identity = "a" * 64, "b" * 64
        blind_digest, dev_digest = "d" * 64, "c" * 64
        output_digest, sidecar_digest, fit_digest, trace_digest = "e" * 64, "f" * 64, "1" * 64, "2" * 64
        candidate_identity_text = (f"{run_identity}\nB1\n1\n{format(0.1, '.17g')}\n{fit_identity}\n"
                                   "0.5 \nFINAL_M0_M1\n\"P0\" \"0,0,1\" 1 \n")
        candidate_identity = __import__("hashlib").sha256(candidate_identity_text.encode("ascii")).hexdigest()
        operation = selection_state in {"BLIND_OPENED", "PUBLISHING"}
        publishing = selection_state in {"PUBLISHING", "PUBLISHED"}
        published = selection_state == "PUBLISHED"
        payload = (
            f'RPJAOUTER5\njob_id "job-17"\nidentity_sha256 "{run_identity}"\n'
            'branch "B2"\nfamily "M0"\nstage "FREEZE"\n'
            f'selection_state "{selection_state}"\nfrozen_candidate_branch "B1"\n'
            f'frozen_candidate_index 1\nfrozen_candidate_lambda 0.1\n'
            f'frozen_candidate_fit_state_sha256 "{fit_identity}"\n'
            f'frozen_candidate_identity_sha256 "{candidate_identity}"\n'
            f'development_response_sha256 "{dev_digest}"\n'
            f'blind_response_sha256 "{blind_digest if selection_state in {"BLIND_PASSED", "PUBLISHING", "PUBLISHED"} else ""}"\n'
            f'publication_stage_output_sha256 "{output_digest if publishing else ""}"\n'
            f'publication_stage_sidecar_sha256 "{sidecar_digest if publishing else ""}"\n'
            f'published_output_sha256 "{output_digest if published else ""}"\n'
            f'published_sidecar_sha256 "{sidecar_digest if published else ""}"\n'
            f'published_fit_sha256 "{fit_digest if publishing else ""}"\n'
            f'published_trace_sha256 "{trace_digest if published else ""}"\n'
            'common_probe_stage "FINAL_M0_M1"\nnext_candidate_index 2\nactive_phase 5\n'
            'active_candidate_started 0\nmax_fit_family_gradients 52\nmax_fit_family_wall_seconds 120\n'
            'elapsed_seconds 12\n'
            f'operation_in_progress {1 if operation else 0}\noperation_start_epoch_seconds {100 if operation else 0}\n'
            'exact_gradients 4\ncg_witness_attempts 0\nactive_candidate_elapsed_seconds 0\n'
            'training_statistics_frozen 1\ntraining_statistics_policy_version RPJASTAT2\n'
            'training_frame_count 66\ndevelopment_frame_count 34\ntraining_block_length 8\n'
            'training_adjacent_block_length 16\ntraining_covered_frame_count 64\n'
            'training_discarded_tail_frames 2\ntraining_channel_count 1\ntraining_channel PARAMETER_0\n'
            'common_probe_set_frozen 1\ncommon_probe_count 1\nprobe "P0" "0,0,1" 1 1\n'
            'candidate_count 1\n'
            f'candidate "B1" "{fit_identity}" "FINAL_M0_M1" 1 0.1 1 1 1 1 1 1 2 0.1 '
            '1 0.5 1 1.1 1 1.2\n'
        )
        return payload.encode("ascii")

    @staticmethod
    def write_sample(path: Path, atoms: int, frames: int, temperature: float = 300.0,
                     cell=None, masses=None, types=None) -> None:
        prefix = "="
        cell = [10.0] * 9 if cell is None else cell
        masses = [16.0] + [1.0] * (atoms - 1) if masses is None else masses
        types = [0] + [1] * (atoms - 1) if types is None else types
        header = struct.pack(prefix + "8sIIiid9d", b"GPJASMP1", 1, 0x01020304,
                              atoms, 32, temperature, *cell)
        identity = struct.pack(f"={atoms}d", *masses) + struct.pack(f"={atoms}i", *types)
        path.write_bytes(header + identity + bytes(frames * 8 * (1 + 6 * atoms)))

    def test_a0_package_header_accepts_crlf_and_rejects_trailing_payload(self):
        package = b"RPJAA0PACKAGE1\r\nmanifest_bytes 0\r\nstate_bytes 0\r\nrecord_bytes 0\r\n"
        self.assertEqual(PREFLIGHT._parse_a0_reuse_package(package), (b"", b"", b""))
        with self.assertRaisesRegex(ValueError, "payload lengths"):
            PREFLIGHT._parse_a0_reuse_package(package + b"x")
        with self.assertRaisesRegex(ValueError, "unsupported version"):
            PREFLIGHT._parse_a0_reuse_package(package.replace(b"PACKAGE1", b"PACKAGE0", 1))

    def test_receipt_fields_round_trip_with_windows_path_and_unicode(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "receipt.txt"
            PREFLIGHT.write_receipt(path, {"status": "PASS", "path": r"D:\测试数据\run.in", "output": "a \"quoted\" name"})
            fields = PREFLIGHT.parse_quoted_fields(path, "RPJAPREFLIGHT5")
            self.assertEqual(fields["path"], r"D:\测试数据\run.in")
            self.assertEqual(fields["output"], "a \"quoted\" name")
            compiler = cpp_test_support.find_cpp_compiler()
            if compiler is None:
                self.skipTest("cross-language receipt test skipped: c++, g++, clang++, and cl are unavailable")
            executable = Path(temporary) / "rpmd_ja_preflight_receipt_test.exe"
            compiled = subprocess.run(cpp_test_support.compile_command(compiler, 14, CPP_RECEIPT_TEST, executable),
                                      cwd=temporary, capture_output=True, text=True, check=False)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            parsed = subprocess.run([str(executable), str(path)], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(parsed.returncode, 0, parsed.stdout + parsed.stderr)

    def test_a0_reuse_import_uses_real_cpp_export_and_fails_closed_on_contract_changes(self):
        compiler = cpp_test_support.find_cpp_compiler()
        if compiler is None:
            self.skipTest("A0 reuse cross-language test skipped: c++, g++, clang++, and cl are unavailable")
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            source_manifest = directory / "source.manifest"
            state = directory / "source.state"
            record = directory / "export.a0_reuse.record"
            executable = directory / "a0_reuse_export.exe"
            fields = {"training_path": "old/train.bin", "training_sha256": "a" * 64,
                      "blind_path": "old/blind.bin", "blind_sha256": "b" * 64,
                      "raw_path": "old/model.qraw", "raw_sha256": "c" * 64,
                      "native_executable_path": "old/gpumd", "native_executable_sha256": "d" * 64,
                      "rho": "0.25", "epsilon": "1e-8", "lambda_factors": "0.01,0.0001,1e-06",
                      "wall_seconds": "28800", "max_exact_gradients": "240", "theta_start_contract": "fixed-R0-v1",
                      "sample_interval": "32", "statistics_policy_version": "RPJASTAT2",
                      "machine_policy": "same-host-device-runtime-v1"}
            manifest_bytes = "RPJAFEAS9\n" + "".join(
                f"{key} {PREFLIGHT.shlex.quote(value)}\n" for key, value in fields.items())
            source_manifest.write_text(manifest_bytes, encoding="utf-8")
            state.write_text("state_format_version 4\nstatus COVARIANCE_OPTIMIZER_CONVERGED\n", encoding="utf-8")
            compiled = subprocess.run(cpp_test_support.compile_command(compiler, 14, CPP_A0_REUSE_EXPORT_TEST, executable),
                                      cwd=temporary, capture_output=True, text=True, check=False)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            exported = subprocess.run([str(executable), str(source_manifest), str(state), str(record), "e" * 64,
                                       "source-job"], cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)
            current = dict(fields, training_path="new/train.bin", blind_path="new/blind.bin", raw_path="new/model.qraw",
                           native_executable_path="new/gpumd", native_output_path="new/result",
                           a0_reuse_package_path=str(record), a0_reuse_package_sha256=PREFLIGHT.sha256(record))
            current_bytes = ("RPJAFEAS9\n" + "".join(
                f"{key} {PREFLIGHT.shlex.quote(value)}\n" for key, value in current.items())).encode("utf-8")
            verified = PREFLIGHT.load_a0_reuse_package(current, directory, current_bytes, "e" * 64, "new-job")
            self.assertEqual(verified["a0_reuse_status"], "VERIFIED")
            self.assertEqual(verified["a0_reuse_source_job_id"], "source-job")
            self.assertEqual(verified["a0_reuse_development_response_pass"], "0")
            for key, value in (("training_sha256", "f" * 64), ("epsilon", "2e-8"),
                               ("rho", "0.5"), ("wall_seconds", "3600")):
                changed = dict(current, **{key: value})
                changed_bytes = ("RPJAFEAS9\n" + "".join(
                    f"{name} {PREFLIGHT.shlex.quote(item)}\n" for name, item in changed.items())).encode("utf-8")
                with self.assertRaisesRegex(ValueError, "contract"):
                    PREFLIGHT.load_a0_reuse_package(changed, directory, changed_bytes, "e" * 64, "new-job")
            with self.assertRaisesRegex(ValueError, "machine"):
                PREFLIGHT.load_a0_reuse_package(current, directory, current_bytes, "f" * 64, "new-job")
            with self.assertRaisesRegex(ValueError, "source job"):
                PREFLIGHT.load_a0_reuse_package(current, directory, current_bytes, "e" * 64, "source-job")
            state.write_text("status NUMERICAL_STAGNATION\n", encoding="utf-8")
            record.write_bytes(record.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "package SHA-256"):
                PREFLIGHT.load_a0_reuse_package(current, directory, current_bytes, "e" * 64, "new-job")

    def test_training_provenance_receipt_binds_sidecar_or_marks_legacy_unknown(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            sample = Path(temporary) / "training.bin"
            sidecar = Path(str(sample) + ".provenance.json")
            sample.write_bytes(b"sample bytes")
            legacy = PREFLIGHT.validate_training_provenance(sample)
            self.assertEqual(legacy["training_provenance_status"], "LEGACY_DECLARED_UNVERIFIED")
            self.assertEqual(legacy["training_provenance_checksum_status"], "NOT_PRESENT")
            concrete = {"model_sha256": "c" * 64,
                        "physical_force_configuration_sha256": "a" * 64,
                        "integrator_configuration_sha256": "b" * 64}
            with self.assertRaisesRegex(SystemExit, "manifest declares concrete configuration"):
                PREFLIGHT.validate_training_provenance(sample, concrete)
            unknown_manifest = {"training_model_sha256": "NOT_VERIFIABLE",
                                "training_physical_force_configuration_sha256": "NOT_VERIFIABLE",
                                "training_integrator_configuration_sha256": "NOT_VERIFIABLE"}
            unknown = PREFLIGHT._expected_sampling_configuration(unknown_manifest, "training")
            self.assertEqual(PREFLIGHT.validate_training_provenance(sample, unknown)["training_provenance_status"],
                             "LEGACY_DECLARED_UNVERIFIED")

            document = PREFLIGHT._PROVENANCE.make_sidecar(b"sample bytes", 3, "job:1",
                "RUN_LOCAL_UNVERIFIED", {})
            sidecar.write_text(PREFLIGHT._PROVENANCE.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not complete and stable"):
                PREFLIGHT.validate_training_provenance(sample)
            with self.assertRaisesRegex(SystemExit, "not complete and stable"):
                PREFLIGHT.validate_training_provenance(
                    sample, PREFLIGHT._expected_sampling_configuration(unknown_manifest, "training"))

            metadata = {
                "integrator_type": "PIMD", "timestep_internal": 0.01,
                "timestep_internal_unit": "GPUMD_INTERNAL_TIME", "timestep_fs": 1.0,
                "timestep_fs_unit": "fs", "thermostat_configuration_tokens": [["ensemble", "nvt"]],
                "model_sha256": concrete["model_sha256"], "run_input_sha256": "d" * 64,
                "physical_force_configuration_sha256": concrete["physical_force_configuration_sha256"],
                "integrator_configuration_sha256": concrete["integrator_configuration_sha256"],
                "atom_signature_sha256": "e" * 64, "physical_force_source": "PHYSICAL_PLUS_RING_SPRINGS",
                "active_configuration_scope": "PREFIX_BEFORE_ACTIVE_RUN", "active_run_ordinal": 0,
                "internal_mass_com": False,
                "sampling_com_method": "MASS_WEIGHTED_CUMULATIVE_CENTROID_DRIFT_REMOVAL",
                "sampling_com_shift_final": [0.0, 0.0, 0.0], "configuration_stable_during_run": True,
            }
            illegal = PREFLIGHT._PROVENANCE.make_sidecar(b"sample bytes", 3, "job:1",
                "SCHEDULER_DECLARED", metadata)
            sidecar.write_text(PREFLIGHT._PROVENANCE.dumps(illegal), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not complete and stable"):
                PREFLIGHT.validate_training_provenance(sample, concrete)

            sample.write_bytes(b"tampered")
            with self.assertRaisesRegex(SystemExit, "provenance sidecar is invalid"):
                PREFLIGHT.validate_training_provenance(sample)

    def test_training_and_blind_provenance_match_declared_configs_and_report_source_relation(self):
        digest = "a" * 64
        metadata = {
            "integrator_type": "PIMD",
            "timestep_internal": 0.01,
            "timestep_internal_unit": "GPUMD_INTERNAL_TIME",
            "timestep_fs": 1.0,
            "timestep_fs_unit": "fs",
            "thermostat_configuration_tokens": [["ensemble", "nvt"]],
            "model_sha256": digest,
            "run_input_sha256": digest,
            "physical_force_configuration_sha256": digest,
            "integrator_configuration_sha256": "b" * 64,
            "atom_signature_sha256": digest,
            "physical_force_source": "MEAN_OF_PHYSICAL_PIMD_BEAD_FORCES",
            "active_configuration_scope": "PREFIX_BEFORE_ACTIVE_RUN",
            "active_run_ordinal": 0,
            "internal_mass_com": False,
            "sampling_com_method": "MASS_WEIGHTED_CUMULATIVE_CENTROID_DRIFT_REMOVAL",
            "sampling_com_shift_final": [0.0, 0.0, 0.0],
            "configuration_stable_during_run": True,
        }
        expected = {
            "model_sha256": digest,
            "physical_force_configuration_sha256": digest,
            "integrator_configuration_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            training = Path(temporary) / "training.bin"
            blind = Path(temporary) / "blind.bin"
            training.write_bytes(b"training sample")
            blind.write_bytes(b"blind sample")
            for path, job, run in ((training, "job-train", "run-train"),
                                   (blind, "job-blind", "run-blind")):
                document = PREFLIGHT._PROVENANCE.make_sidecar(path.read_bytes(), 3, job,
                    "SCHEDULER_DECLARED", metadata, source_run_id=run)
                Path(str(path) + ".provenance.json").write_text(
                    PREFLIGHT._PROVENANCE.dumps(document), encoding="utf-8")
            checked_training = PREFLIGHT.validate_training_provenance(training, expected)
            checked_blind = PREFLIGHT.validate_blind_provenance(blind, expected)
            self.assertEqual(checked_training["training_integrity_status"], "VERIFIED")
            self.assertEqual(checked_blind["blind_integrity_status"], "VERIFIED")
            self.assertEqual(checked_training["training_model_sha256"], digest)
            self.assertEqual(checked_training["training_configuration_match_status"], "MATCHED_DECLARED_VALUES")
            self.assertEqual(checked_blind["blind_configuration_match_status"], "MATCHED_DECLARED_VALUES")
            self.assertEqual(PREFLIGHT.validate_provenance_source_relation(checked_training, checked_blind),
                             "SIDECAR_SOURCE_IDS_DISTINCT_DECLARED")
            wrong = dict(expected)
            wrong["physical_force_configuration_sha256"] = "c" * 64
            with self.assertRaisesRegex(SystemExit, "does not match"):
                PREFLIGHT.validate_training_provenance(training, wrong)
            wrong_model = dict(expected)
            wrong_model["model_sha256"] = "c" * 64
            with self.assertRaisesRegex(SystemExit, "does not match"):
                PREFLIGHT.validate_training_provenance(training, wrong_model)
            same_run = dict(checked_blind)
            same_run["blind_source_run_id"] = checked_training["training_source_run_id"]
            with self.assertRaisesRegex(SystemExit, "same source run"):
                PREFLIGHT.validate_provenance_source_relation(checked_training, same_run)
            unstable_metadata = dict(metadata)
            unstable_metadata["configuration_stable_during_run"] = False
            unstable = PREFLIGHT._PROVENANCE.make_sidecar(training.read_bytes(), 3, "job-train",
                "SCHEDULER_DECLARED", unstable_metadata, source_run_id="run-train")
            Path(str(training) + ".provenance.json").write_text(
                PREFLIGHT._PROVENANCE.dumps(unstable), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not complete and stable"):
                PREFLIGHT.validate_training_provenance(training, expected)

    def test_unknown_historical_configuration_stays_explicitly_unverifiable(self):
        manifest = {
            "training_model_sha256": "NOT_VERIFIABLE",
            "training_physical_force_configuration_sha256": "NOT_VERIFIABLE",
            "training_integrator_configuration_sha256": "NOT_VERIFIABLE",
        }
        expected = PREFLIGHT._expected_sampling_configuration(manifest, "training")
        self.assertEqual(set(expected.values()), {"NOT_VERIFIABLE"})
        manifest["training_integrator_configuration_sha256"] = "a" * 64
        with self.assertRaisesRegex(SystemExit, "all training model and configuration digests"):
            PREFLIGHT._expected_sampling_configuration(manifest, "training")

    def test_concrete_sidecar_model_cannot_be_downgraded_to_unknown_manifest(self):
        digest = "a" * 64
        manifest = {
            "training_model_sha256": "NOT_VERIFIABLE",
            "training_physical_force_configuration_sha256": "NOT_VERIFIABLE",
            "training_integrator_configuration_sha256": "NOT_VERIFIABLE",
        }
        expected = PREFLIGHT._expected_sampling_configuration(manifest, "training")
        metadata = {
            "integrator_type": "PIMD",
            "timestep_internal": 0.01,
            "timestep_internal_unit": "GPUMD_INTERNAL_TIME",
            "timestep_fs": 1.0,
            "timestep_fs_unit": "fs",
            "thermostat_configuration_tokens": [["ensemble", "nvt"]],
            "model_sha256": digest,
            "run_input_sha256": digest,
            "physical_force_configuration_sha256": digest,
            "integrator_configuration_sha256": digest,
            "atom_signature_sha256": digest,
            "physical_force_source": "MEAN_OF_PHYSICAL_PIMD_BEAD_FORCES",
            "active_configuration_scope": "PREFIX_BEFORE_ACTIVE_RUN",
            "active_run_ordinal": 0,
            "internal_mass_com": False,
            "sampling_com_method": "MASS_WEIGHTED_CUMULATIVE_CENTROID_DRIFT_REMOVAL",
            "sampling_com_shift_final": [0.0, 0.0, 0.0],
            "configuration_stable_during_run": True,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            sample = Path(temporary) / "training.bin"
            sample.write_bytes(b"sample bytes")
            sidecar = PREFLIGHT._PROVENANCE.make_sidecar(sample.read_bytes(), 3, "job:1",
                "SCHEDULER_DECLARED", metadata)
            Path(str(sample) + ".provenance.json").write_text(
                PREFLIGHT._PROVENANCE.dumps(sidecar), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "model or configuration digest does not match"):
                PREFLIGHT.validate_training_provenance(sample, expected)

    def test_partial_sidecar_digest_cannot_be_downgraded_to_unknown_manifest(self):
        digest = "a" * 64
        manifest = {
            "training_model_sha256": "NOT_VERIFIABLE",
            "training_physical_force_configuration_sha256": "NOT_VERIFIABLE",
            "training_integrator_configuration_sha256": "NOT_VERIFIABLE",
        }
        metadata = {"model_sha256": digest}
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            sample = Path(temporary) / "training.bin"
            sample.write_bytes(b"sample bytes")
            sidecar = PREFLIGHT._PROVENANCE.make_sidecar(sample.read_bytes(), 3, "job:1",
                "SCHEDULER_DECLARED", metadata)
            Path(str(sample) + ".provenance.json").write_text(
                PREFLIGHT._PROVENANCE.dumps(sidecar), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not complete and stable"):
                PREFLIGHT.validate_training_provenance(
                    sample, PREFLIGHT._expected_sampling_configuration(manifest, "training"))

    def test_resume_checkpoint_receipt_binds_intact_unfrozen_job(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "out.checkpoint"
            payload = ("RPJAOUTER5\njob_id \"job-17\"\nidentity_sha256 \"" + "a" * 64 +
                       "\"\nbranch \"C1\"\nfamily \"M1\"\nstage \"C\"\n"
                       "selection_state \"NOT_FROZEN\"\ncommon_probe_stage \"B_GATE\"\n"
                       "next_candidate_index 1\nactive_phase 4\nactive_candidate_started 1\n"
                       "max_fit_family_gradients 52\nmax_fit_family_wall_seconds 120\n"
                       "training_statistics_frozen 1\ntraining_statistics_policy_version RPJASTAT2\n"
                       "training_frame_count 66\ndevelopment_frame_count 34\ntraining_block_length 8\n"
                       "training_adjacent_block_length 16\ntraining_covered_frame_count 64\n"
                       "training_discarded_tail_frames 2\ntraining_channel_count 2\n"
                       "training_channel FIXED_PHYSICAL_X\ntraining_channel PARAMETER_0\n"
                       "elapsed_seconds 12\noperation_in_progress 0\noperation_start_epoch_seconds 0\n"
                       "exact_gradients 4\ncg_witness_attempts 0\nactive_candidate_elapsed_seconds 6\n" +
                       self.OUTER5_REQUIRED_FIELDS).encode()
            path.write_bytes(payload + b"sha256 " + __import__("hashlib").sha256(payload).hexdigest().encode() + b"\n")
            checkpoint = PREFLIGHT.inspect_resume_checkpoint(path)
            self.assertEqual(checkpoint["job_id"], "job-17")
            self.assertEqual(checkpoint["selection_state"], "NOT_FROZEN")
            self.assertEqual(checkpoint["stage"], "C")
            self.assertEqual(checkpoint["cg_witness_attempts"], "0")
            self.assertEqual(checkpoint["checkpoint_sha256"], PREFLIGHT.sha256(path))
            tampered = payload.replace(b"next_candidate_index 1", b"next_candidate_index 2")
            path.write_bytes(tampered + b"sha256 " + __import__("hashlib").sha256(tampered).hexdigest().encode() + b"\n")
            with self.assertRaisesRegex(SystemExit, "stage, family, branch, and candidate cursor disagree"):
                PREFLIGHT.inspect_resume_checkpoint(path)
            tampered = payload.replace(b"cg_witness_attempts 0", b"cg_witness_attempts -1")
            path.write_bytes(tampered + b"sha256 " + __import__("hashlib").sha256(tampered).hexdigest().encode() + b"\n")
            with self.assertRaisesRegex(SystemExit, "invalid cursor or cumulative budget fields"):
                PREFLIGHT.inspect_resume_checkpoint(path)

    def test_outer5_frozen_blind_and_publication_states_are_bound_to_candidate_and_probes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "out.checkpoint"

            def write(payload: bytes) -> None:
                path.write_bytes(payload + b"sha256 " + __import__("hashlib").sha256(payload).hexdigest().encode() + b"\n")

            for state in ("FROZEN", "BLIND_OPENED", "BLIND_PASSED", "PUBLISHING", "PUBLISHED"):
                payload = self.frozen_outer5_payload(state)
                with self.subTest(state=state):
                    write(payload)
                    restored = PREFLIGHT.inspect_resume_checkpoint(path)
                    self.assertEqual(restored["selection_state"], state)
                    self.assertEqual(restored["frozen_candidate_branch"], "B1")
                    self.assertEqual(len(restored["candidate_records"]), 1)

            payload = self.frozen_outer5_payload("FROZEN").replace(
                b'branch "B2"\nfamily "M0"', b'branch "C2"\nfamily "M1"')
            write(payload)
            restored = PREFLIGHT.inspect_resume_checkpoint(path)
            self.assertEqual((restored["branch"], restored["family"], restored["frozen_candidate_branch"]),
                             ("C2", "M1", "B1"))

            payload = self.frozen_outer5_payload("FROZEN").replace(b'probe "P0" "0,0,1" 1 1',
                                                                       b'probe "P0" "0,0,1" 1 2')
            write(payload)
            with self.assertRaisesRegex(SystemExit, "frozen candidate identity"):
                PREFLIGHT.inspect_resume_checkpoint(path)

            payload = self.frozen_outer5_payload("PUBLISHING").replace(
                b"operation_in_progress 1", b"operation_in_progress 0")
            write(payload)
            with self.assertRaisesRegex(SystemExit, "publication checkpoint does not have"):
                PREFLIGHT.inspect_resume_checkpoint(path)
            payload = self.frozen_outer5_payload("PUBLISHING").replace(
                b'publication_stage_output_sha256 "' + b"e" * 64 + b'"',
                b'publication_stage_output_sha256 ""').replace(
                b'publication_stage_sidecar_sha256 "' + b"f" * 64 + b'"',
                b'publication_stage_sidecar_sha256 ""')
            write(payload)
            with self.assertRaisesRegex(SystemExit, "commits fit diagnostics before"):
                PREFLIGHT.inspect_resume_checkpoint(path)

    def test_resume_checkpoint_rejects_corruption_and_frozen_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "out.checkpoint"
            def write_checkpoint(state: str) -> None:
                payload = ("RPJAOUTER5\njob_id \"job-17\"\nidentity_sha256 \"" + "a" * 64 +
                           "\"\nbranch \"B0\"\nfamily \"M0\"\nstage \"B\"\nselection_state \"" + state +
                           "\"\ncommon_probe_stage \"NONE\"\nnext_candidate_index 0\nactive_phase 3\n"
                           "active_candidate_started 1\nmax_fit_family_gradients 52\nmax_fit_family_wall_seconds 120\n"
                           "training_statistics_frozen 1\ntraining_statistics_policy_version RPJASTAT2\n"
                           "training_frame_count 66\ndevelopment_frame_count 34\ntraining_block_length 8\n"
                           "training_adjacent_block_length 16\ntraining_covered_frame_count 64\n"
                           "training_discarded_tail_frames 2\ntraining_channel_count 2\n"
                           "training_channel FIXED_PHYSICAL_X\ntraining_channel PARAMETER_0\n"
                           "elapsed_seconds 5\noperation_in_progress 0\noperation_start_epoch_seconds 0\nexact_gradients 2\n"
                           "cg_witness_attempts 0\nactive_candidate_elapsed_seconds 5\n" +
                           self.OUTER5_REQUIRED_FIELDS).encode()
                path.write_bytes(payload + b"sha256 " + __import__("hashlib").sha256(payload).hexdigest().encode() + b"\n")
            write_checkpoint("NOT_FROZEN")
            path.write_bytes(path.read_bytes().replace(b"branch", b"Branch", 1))
            with self.assertRaisesRegex(SystemExit, "checksum"):
                PREFLIGHT.inspect_resume_checkpoint(path)
            write_checkpoint("FROZEN_BLIND_NOT_OPENED")
            with self.assertRaisesRegex(SystemExit, "unsupported frozen-selection state"):
                PREFLIGHT.inspect_resume_checkpoint(path)

    def test_resume_checkpoint_rejects_manifest_identity_change_and_solver_only_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "out.checkpoint"
            payload = ("RPJAOUTER5\njob_id \"job-17\"\nidentity_sha256 \"" + "a" * 64 +
                       "\"\nbranch \"B0\"\nfamily \"M0\"\nstage \"B\"\n"
                       "selection_state \"NOT_FROZEN\"\ncommon_probe_stage \"NONE\"\n"
                       "next_candidate_index 0\nactive_phase 3\nactive_candidate_started 1\n"
                       "max_fit_family_gradients 52\nmax_fit_family_wall_seconds 120\n"
                       "training_statistics_frozen 1\ntraining_statistics_policy_version RPJASTAT2\n"
                       "training_frame_count 66\ndevelopment_frame_count 34\ntraining_block_length 8\n"
                       "training_adjacent_block_length 16\ntraining_covered_frame_count 64\n"
                       "training_discarded_tail_frames 2\ntraining_channel_count 2\n"
                       "training_channel FIXED_PHYSICAL_X\ntraining_channel PARAMETER_0\n"
                       "elapsed_seconds 5\noperation_in_progress 0\noperation_start_epoch_seconds 0\n"
                       "exact_gradients 2\ncg_witness_attempts 0\nactive_candidate_elapsed_seconds 5\n" +
                       self.OUTER5_REQUIRED_FIELDS).encode()
            path.write_bytes(payload + b"sha256 " + __import__("hashlib").sha256(payload).hexdigest().encode() + b"\n")
            with self.assertRaisesRegex(SystemExit, "manifest identity"):
                PREFLIGHT.inspect_resume_checkpoint(path, "b" * 64)
            path.write_bytes(b"RPJACKPT3\n")
            with self.assertRaisesRegex(SystemExit, "truncated"):
                PREFLIGHT.inspect_resume_checkpoint(path)

    def test_resume_checkpoint_rejects_rechecksummed_budget_overrun(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "out.checkpoint"

            def write_checkpoint(elapsed: str, gradients: str,
                                 family_gradients: str = "52", family_wall: str = "120") -> None:
                payload = ("RPJAOUTER5\njob_id \"job-17\"\nidentity_sha256 \"" + "a" * 64 +
                           "\"\nbranch \"B1\"\nfamily \"M0\"\nstage \"B\"\n"
                           "selection_state \"NOT_FROZEN\"\ncommon_probe_stage \"B_GATE\"\n"
                           "next_candidate_index 1\nactive_phase 3\nactive_candidate_started 1\n"
                           f"max_fit_family_gradients {family_gradients}\nmax_fit_family_wall_seconds {family_wall}\n"
                           "training_statistics_frozen 1\ntraining_statistics_policy_version RPJASTAT2\n"
                           "training_frame_count 66\ndevelopment_frame_count 34\ntraining_block_length 8\n"
                           "training_adjacent_block_length 16\ntraining_covered_frame_count 64\n"
                           "training_discarded_tail_frames 2\ntraining_channel_count 2\n"
                           "training_channel FIXED_PHYSICAL_X\ntraining_channel PARAMETER_0\n"
                           f"elapsed_seconds {elapsed}\noperation_in_progress 0\noperation_start_epoch_seconds 0\n"
                           f"exact_gradients {gradients}\n"
                           "cg_witness_attempts 0\nactive_candidate_elapsed_seconds 1\n" +
                           self.OUTER5_REQUIRED_FIELDS).encode()
                path.write_bytes(payload + b"sha256 " + __import__("hashlib").sha256(payload).hexdigest().encode() + b"\n")

            for elapsed, gradients, family_gradients, family_wall in (
                    ("28801", "4", "52", "120"), ("12", "241", "52", "120"),
                    ("12", "4", "113", "120"), ("12", "4", "52", "10801")):
                write_checkpoint(elapsed, gradients, family_gradients, family_wall)
                with self.subTest(elapsed=elapsed, gradients=gradients, family_gradients=family_gradients,
                                  family_wall=family_wall), \
                     self.assertRaisesRegex(SystemExit, "frozen budget"):
                    PREFLIGHT.inspect_resume_checkpoint(path)
            write_checkpoint("12", "4")
            original = path.read_bytes()
            payload = original[:original.rfind(b"sha256 ")].replace(
                b"training_adjacent_block_length 16", b"training_adjacent_block_length 8")
            path.write_bytes(payload + b"sha256 " + __import__("hashlib").sha256(payload).hexdigest().encode() + b"\n")
            with self.assertRaisesRegex(SystemExit, "frozen training statistics plan"):
                PREFLIGHT.inspect_resume_checkpoint(path)

    def test_resume_cli_keeps_manifest_and_run_directory_arguments(self):
        manifest, run_directory, resume = PREFLIGHT.parse_cli_arguments(
            ["--resume", "manifest.txt", "run-directory"])
        self.assertEqual((manifest, run_directory, resume), ("manifest.txt", "run-directory", True))
        manifest, run_directory, resume = PREFLIGHT.parse_cli_arguments(["manifest.txt", "run-directory"])
        self.assertEqual((manifest, run_directory, resume), ("manifest.txt", "run-directory", False))
        with self.assertRaisesRegex(SystemExit, "usage"):
            PREFLIGHT.parse_cli_arguments(["--resume", "--resume", "manifest.txt", "run-directory"])

    def test_resume_allows_only_checkpoint_scratch_and_rejects_publication(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "reference"
            checkpoint = Path(str(output) + ".checkpoint")
            checkpoint.write_text("checkpoint", encoding="ascii")
            Path(str(output) + ".fit.lock").write_text("stale lock marker", encoding="ascii")
            (Path(str(output) + ".fit_trace.txt")).write_text("trace", encoding="ascii")
            (Path(str(output) + ".covariance_state.txt.C1")).write_text("active branch", encoding="ascii")
            resume = {"selection_state": "NOT_FROZEN", "job_id": "job-17", "branch": "C1",
                      "active_candidate_started": "1", "candidate_records": [],
                      "published_trace_sha256": "", "development_response_sha256": "",
                      "blind_response_sha256": ""}
            PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                          resume_checkpoint=resume)
            Path(str(output) + ".stability").write_text("published", encoding="ascii")
            with self.assertRaisesRegex(SystemExit, "overwrite"):
                PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                              resume_checkpoint=resume)

    def test_resume_publishing_accepts_only_digest_bound_two_rename_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "reference"
            checkpoint = Path(str(output) + ".checkpoint")
            checkpoint.write_text("controller", encoding="ascii")
            job_id = "job-17"
            staged = Path(str(output) + ".stage." + __import__("hashlib").sha256(
                job_id.encode("utf-8")).hexdigest()[:16])
            output.write_bytes(b"prepared-reference")  # Interrupted after rename 1.
            Path(str(staged) + ".stability").write_bytes(b"prepared-sidecar")
            Path(str(output) + ".fit.txt").write_bytes(b"fit-diagnostics")
            Path(str(output) + ".fit_trace.txt").write_bytes(b"appendable-trace")
            Path(str(output) + ".development_checkpoint.txt").write_bytes(b"development")
            Path(str(output) + ".blind_checkpoint.txt").write_bytes(b"blind")
            digest = PREFLIGHT.sha256
            candidate_state = Path(str(output) + ".covariance_state.txt.B1")
            candidate_state.write_bytes(b"selected-state")
            candidate_state_digest = digest(candidate_state)
            resume = {"selection_state": "PUBLISHING", "job_id": job_id, "branch": "C2",
                      "frozen_candidate_branch": "B1", "frozen_candidate_fit_state_sha256": candidate_state_digest,
                      "active_candidate_started": "0", "candidate_records": [],
                      "published_trace_sha256": "", "development_response_sha256": digest(
                          Path(str(output) + ".development_checkpoint.txt")),
                      "blind_response_sha256": digest(Path(str(output) + ".blind_checkpoint.txt")),
                      "publication_stage_output_sha256": digest(output),
                      "publication_stage_sidecar_sha256": digest(Path(str(staged) + ".stability")),
                      "published_fit_sha256": digest(Path(str(output) + ".fit.txt"))}
            PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                          resume_checkpoint=resume)
            failure = Path(str(output) + ".failure.txt")
            failure.write_text("hard publication integrity failure", encoding="ascii")
            with self.assertRaisesRegex(SystemExit, "overwrite"):
                PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                              resume_checkpoint=resume)
            failure.unlink()
            Path(str(output) + ".stability").write_bytes(b"foreign-final")
            with self.assertRaisesRegex(SystemExit, "identity mismatch"):
                PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                              resume_checkpoint=resume)

    def test_published_resume_verifies_all_sealed_final_artifacts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "reference"
            checkpoint = Path(str(output) + ".checkpoint")
            checkpoint.write_text("controller", encoding="ascii")
            files = {suffix: Path(str(output) + suffix) for suffix in (
                "", ".stability", ".fit.txt", ".fit_trace.txt", ".development_checkpoint.txt",
                ".blind_checkpoint.txt")}
            for suffix, path in files.items():
                path.write_bytes(("sealed" + suffix).encode("ascii"))
            sha = PREFLIGHT.sha256
            candidate_state = Path(str(output) + ".covariance_state.txt.B1")
            candidate_state.write_bytes(b"selected-state")
            candidate_state_digest = sha(candidate_state)
            resume = {"selection_state": "PUBLISHED", "job_id": "job-17", "branch": "C2",
                      "frozen_candidate_branch": "B1", "frozen_candidate_fit_state_sha256": candidate_state_digest,
                      "active_candidate_started": "0", "candidate_records": [],
                      "development_response_sha256": sha(files[".development_checkpoint.txt"]),
                      "blind_response_sha256": sha(files[".blind_checkpoint.txt"]),
                      "publication_stage_output_sha256": sha(files[""]),
                      "publication_stage_sidecar_sha256": sha(files[".stability"]),
                      "published_fit_sha256": sha(files[".fit.txt"]),
                      "published_trace_sha256": sha(files[".fit_trace.txt"])}
            PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                          resume_checkpoint=resume)
            files[".fit_trace.txt"].write_bytes(b"modified after publication")
            with self.assertRaisesRegex(SystemExit, "identity mismatch"):
                PREFLIGHT.no_existing_outputs(output, resume=True, checkpoint_path=checkpoint,
                                              resume_checkpoint=resume)

    def test_preflight_os_lock_rejects_live_owner_and_recovers_after_kill(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "reference"
            child_code = (
                "import importlib.util,sys,time; "
                "spec=importlib.util.spec_from_file_location('preflight_child',sys.argv[1]); "
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "module.acquire_preflight_lock(__import__('pathlib').Path(sys.argv[2]),'child-job'); "
                "print('LOCKED',flush=True); time.sleep(60)"
            )
            process = subprocess.Popen([PREFLIGHT.sys.executable, "-c", child_code, str(SCRIPT), str(output)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "LOCKED")
                with self.assertRaisesRegex(SystemExit, "single-job lock"):
                    PREFLIGHT.acquire_preflight_lock(output, "parent-job")
            finally:
                process.kill()
                process.wait(timeout=10)
                process.stdout.close()
                process.stderr.close()
            path, _token = PREFLIGHT.acquire_preflight_lock(output, "recovered-job")
            self.assertTrue(path.exists())
            stream = PREFLIGHT._PREFLIGHT_LOCKS.pop()
            if PREFLIGHT.os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def test_fit_command_uses_gpumd_comment_and_token_rules(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_config = Path(temporary) / "run.in"
            run_config.write_text(
                "# one command only\n"
                "rpmd_ja fit_samples_covariance train.bin out 5 1e-5 0.1 1e-4 kernel.txt raw.qraw "
                "shrinkage 0.1 max_iter 50 trust_region feasibility structured_basis "
                "blind_samples blind.bin blind_manifest manifest.txt # comment\n",
                encoding="utf-8",
            )
            tokens, blind_samples, blind_manifest = PREFLIGHT.find_fit_command(run_config)
            self.assertEqual(tokens[2:4], ["train.bin", "out"])
            self.assertEqual(blind_samples, "blind.bin")
            self.assertEqual(blind_manifest, "manifest.txt")

    def test_fit_command_rejects_multiple_native_fits(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_config = Path(temporary) / "run.in"
            line = ("rpmd_ja fit_samples_covariance train out 5 1e-5 0.1 1e-4 kernel raw "
                    "shrinkage 0.1 max_iter 50 trust_region feasibility structured_basis "
                    "blind_samples blind blind_manifest manifest\n")
            run_config.write_text(line + line, encoding="utf-8")
            with self.assertRaises(SystemExit):
                PREFLIGHT.find_fit_command(run_config)

    def test_fit_command_rejects_any_command_after_offline_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_config = Path(temporary) / "run.in"
            fit = ("rpmd_ja fit_samples_covariance train out 5 1e-5 0.1 1e-4 kernel raw "
                   "shrinkage 0.1 max_iter 50 trust_region feasibility structured_basis "
                   "blind_samples blind blind_manifest manifest\n")
            for production in ("run 1000\n", "ensemble nvt_berendsen 300 100 10\n"):
                for content in (fit + production, production + fit):
                    with self.subTest(content=content):
                        run_config.write_text(content, encoding="utf-8")
                        with self.assertRaises(SystemExit):
                            PREFLIGHT.find_fit_command(run_config)

    def test_offline_fit_allows_setup_commands_before_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_config = Path(temporary) / "run.in"
            fit = ("rpmd_ja fit_samples_covariance train out 5 1e-5 0.1 1e-4 kernel raw "
                   "shrinkage 0.1 max_iter 50 trust_region feasibility structured_basis "
                   "blind_samples blind blind_manifest manifest\n")
            run_config.write_text("potential qnep.txt\npppm_mesh_spacing 1.0\npimd_fix_com on\nkspace pppm\n" + fit,
                                  encoding="utf-8")
            tokens, _, _ = PREFLIGHT.find_fit_command(run_config)
            self.assertEqual(tokens[:2], ["rpmd_ja", "fit_samples_covariance"])

    def test_offline_fit_rejects_computation_before_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_config = Path(temporary) / "run.in"
            fit = ("rpmd_ja fit_samples_covariance train out 5 1e-5 0.1 1e-4 kernel raw "
                   "shrinkage 0.1 max_iter 50 trust_region feasibility structured_basis "
                   "blind_samples blind blind_manifest manifest\n")
            for command in ("minimize 1e-6 100 100 100\n", "compute_phonon 1 1 1\n",
                            "compute_cohesive\n", "compute_elastic 0.001\n"):
                with self.subTest(command=command):
                    run_config.write_text("potential qnep.txt\n" + command + fit, encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, "unsupported commands"):
                        PREFLIGHT.find_fit_command(run_config)

    def test_fingerprint_uses_native_fnv1a64(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "bytes.bin"
            path.write_bytes(b"a")
            self.assertEqual(PREFLIGHT.fnv64(path), "af63dc4c8601ec8c")

    def test_fnv64_manifest_fields_require_full_hex_identity(self):
        self.assertTrue(PREFLIGHT.is_fnv64("0123456789abcdef"))
        self.assertTrue(PREFLIGHT.is_fnv64("0123456789ABCDEF"))
        self.assertFalse(PREFLIGHT.is_fnv64("1234"))
        self.assertFalse(PREFLIGHT.is_fnv64("0123456789abcdeg"))
        fields = {key: "0123456789abcdef" for key in (
            "training_fnv64", "blind_fnv64", "declared_model_fingerprint", "declared_config_fingerprint")}
        PREFLIGHT.validate_native_identity_fields(fields)
        del fields["declared_config_fingerprint"]
        with self.assertRaises(SystemExit):
            PREFLIGHT.validate_native_identity_fields(fields)

    def test_m1_minimum_support_rule_is_explicit_and_bounded(self):
        self.assertEqual(PREFLIGHT.validate_m1_minimum_support("3"), 3)
        for value in ("0", "1001", "3.0", ""):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                PREFLIGHT.validate_m1_minimum_support(value)

    def test_response_delta_tolerance_is_independent_and_finite(self):
        self.assertEqual(PREFLIGHT.validate_response_delta_tolerance("0.025"), 0.025)
        for value in ("0", "-0.1", "nan", "inf", "not-a-number", ""):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                PREFLIGHT.validate_response_delta_tolerance(value)

    def test_curvature_seed_cap_and_tolerance_are_manifest_frozen(self):
        contract = {
            "curvature_primary_seed": PREFLIGHT.CURVATURE_PRIMARY_SEED,
            "curvature_independent_seed": PREFLIGHT.CURVATURE_INDEPENDENT_SEED,
            "curvature_probe_cap": str(PREFLIGHT.CURVATURE_PROBE_CAP),
            "curvature_directional_tolerance": "0.1",
        }
        self.assertEqual(PREFLIGHT.validate_curvature_contract(contract)["curvature_probe_cap"], "128")
        for key, value in (("curvature_primary_seed", "1" * 16), ("curvature_probe_cap", "64"),
                           ("curvature_directional_tolerance", "0.2")):
            changed = dict(contract)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(SystemExit):
                PREFLIGHT.validate_curvature_contract(changed)

    def test_candidate_cost_stop_rule_and_safety_factor_are_manifest_frozen(self):
        self.assertEqual(PREFLIGHT.CANDIDATE_COST_STOP_RULE,
                         "MAX_COMPLETE_OR_PARTIAL_LOWER_BOUND_SECONDS_V1")
        contract = {
            "candidate_cost_stop_rule": PREFLIGHT.CANDIDATE_COST_STOP_RULE,
            "candidate_cost_safety_factor": "1.5",
        }
        self.assertEqual(PREFLIGHT.validate_candidate_cost_contract(contract)["candidate_cost_safety_factor"], "1.5")
        for key, value in (("candidate_cost_stop_rule", "OTHER"),
                           ("candidate_cost_stop_rule", "MAX_OBSERVED_CANDIDATE_SECONDS_V1"),
                           ("candidate_cost_safety_factor", "0.9"),
                           ("candidate_cost_safety_factor", "nan"), ("candidate_cost_safety_factor", "11")):
            changed = dict(contract)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(SystemExit):
                PREFLIGHT.validate_candidate_cost_contract(changed)

    def test_performance_stop_rule_is_frozen_into_manifest_and_receipt(self):
        contract = {
            "performance_stop_rule": PREFLIGHT.PERFORMANCE_STOP_RULE,
            "performance_stop_objective_normalizer": PREFLIGHT.PERFORMANCE_STOP_OBJECTIVE_NORMALIZER,
            "performance_stop_gradient_cost_rule": PREFLIGHT.PERFORMANCE_STOP_GRADIENT_COST_RULE,
            "performance_stop_curvature_window": str(PREFLIGHT.PERFORMANCE_STOP_CURVATURE_WINDOW),
            "performance_stop_min_relative_objective_improvement": "0.001",
            "performance_stop_curvature_to_gradient_cost_ratio": "1",
        }
        self.assertEqual(PREFLIGHT.validate_performance_stop_contract(contract), {
            "performance_stop_rule": PREFLIGHT.PERFORMANCE_STOP_RULE,
            "performance_stop_objective_normalizer": PREFLIGHT.PERFORMANCE_STOP_OBJECTIVE_NORMALIZER,
            "performance_stop_gradient_cost_rule": PREFLIGHT.PERFORMANCE_STOP_GRADIENT_COST_RULE,
            "performance_stop_curvature_window": "3",
            "performance_stop_min_relative_objective_improvement": "0.001",
            "performance_stop_curvature_to_gradient_cost_ratio": "1",
        })
        for key, value in (("performance_stop_curvature_window", "2"),
                           ("performance_stop_min_relative_objective_improvement", "nan"),
                           ("performance_stop_curvature_to_gradient_cost_ratio", "2"),
                           ("performance_stop_objective_normalizer", "ABS_START_OBJECTIVE")):
            changed = dict(contract)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(SystemExit):
                PREFLIGHT.validate_performance_stop_contract(changed)

    def test_cost_baseline_is_bound_to_machine_and_problem_dimensions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "cost-baseline.txt"
            inventory = "GPU 0: test device"
            fields = {
                "machine_fingerprint": PREFLIGHT.machine_fingerprint(inventory),
                "atom_count": "2", "sample_frames": "100", "training_frames": "66",
                "development_frames": "34", "m0_parameter_count": "6", "m1_parameter_count": "8",
                "candidate_seconds": "10", "fit_seconds": "7", "curvature_seconds": "2",
                "blind_seconds": "6", "package_write_seconds": "3", "prepare_seconds": "4", "readback_seconds": "5",
            }
            PREFLIGHT.write_receipt(path, fields, magic="RPJACOST1")
            verified = PREFLIGHT.validate_cost_baseline(path, inventory, 2, 100, 66, 34, 18)
            self.assertEqual(verified["cost_baseline_status"], "VERIFIED")
            self.assertEqual(verified["cost_baseline_m0_parameter_count"], "6")
            self.assertEqual(verified["cost_baseline_m0_parameter_upper_bound"], "18")
            self.assertEqual(verified["cost_baseline_prepare_seconds"], "4")
            with self.assertRaisesRegex(SystemExit, "machine or problem size"):
                PREFLIGHT.validate_cost_baseline(path, "different GPU inventory", 2, 100, 66, 34, 18)
            wrong_size = dict(fields, atom_count="3")
            PREFLIGHT.write_receipt(path, wrong_size, magic="RPJACOST1")
            with self.assertRaisesRegex(SystemExit, "machine or problem size"):
                PREFLIGHT.validate_cost_baseline(path, inventory, 2, 100, 66, 34, 18)
            missing_blind = dict(fields)
            del missing_blind["blind_seconds"]
            PREFLIGHT.write_receipt(path, missing_blind, magic="RPJACOST1")
            with self.assertRaisesRegex(SystemExit, "blind_seconds"):
                PREFLIGHT.validate_cost_baseline(path, inventory, 2, 100, 66, 34, 18)
            too_many_m0 = dict(fields, m0_parameter_count="24", m1_parameter_count="25")
            PREFLIGHT.write_receipt(path, too_many_m0, magic="RPJACOST1")
            with self.assertRaisesRegex(SystemExit, "theoretical upper bound"):
                PREFLIGHT.validate_cost_baseline(path, inventory, 2, 100, 66, 34, 18)
            wrong_m1 = dict(fields, m1_parameter_count="5")
            PREFLIGHT.write_receipt(path, wrong_m1, magic="RPJACOST1")
            with self.assertRaisesRegex(SystemExit, "parameter range"):
                PREFLIGHT.validate_cost_baseline(path, inventory, 2, 100, 66, 34, 18)

    def test_lambda_budget_and_development_block_contract_is_frozen(self):
        contract = {
            "lambda_factors": "0.01,0.0001,0.000001",
            "wall_seconds": "28800",
            "max_exact_gradients": "240",
            "validation_reserve_fraction": "0.25",
            "max_a_branch_gradients": "8",
            "max_fit_family_gradients": "112",
            "a_wall_fraction": "0.1",
            "family_wall_split_rule": "EQUAL_REMAINING_FIT_TIME_V1",
            "family_gradient_split_rule": "EQUAL_REMAINING_EXACT_GRADIENTS_V1",
            "development_max_blocks": "8",
            "development_min_blocks": "4",
            "development_block_rule": "CONTIGUOUS_EQUAL_FRAME_COUNTS_V1",
            "statistics_storage_budget_bytes": str(PREFLIGHT.STATISTICS_STORAGE_BUDGET_BYTES),
        }
        self.assertEqual(PREFLIGHT.validate_experiment_contract(contract)["development_max_blocks"], "8")
        for key, value in (("lambda_factors", "0.01,0.001,0.0001"), ("wall_seconds", "3600"),
                           ("development_min_blocks", "3"), ("development_block_rule", "AFTER_LOOKING"),
                           ("statistics_storage_budget_bytes", str(PREFLIGHT.STATISTICS_STORAGE_BUDGET_BYTES + 1))):
            changed = dict(contract)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(SystemExit):
                PREFLIGHT.validate_experiment_contract(changed)

    def test_native_cuda_regression_receives_explicit_fixture_directory(self):
        executable, fixture_directory = Path("build/native_fit_test"), Path("repo/tests/data")
        command = PREFLIGHT.native_fit_regression_command(executable, fixture_directory)
        self.assertEqual(command, [str(executable), "--fixture-dir", str(fixture_directory)])
        launcher = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('Path(__file__).resolve().parents[1] / "tests" / "data"', launcher)

    def test_native_cuda_regression_fixtures_are_pinned_and_receipted(self):
        fixture_directory = ROOT / "tests/data"
        hashes = PREFLIGHT.validate_cuda_regression_fixtures(fixture_directory)
        self.assertEqual(hashes, PREFLIGHT.FROZEN_CUDA_QP_FIXTURE_SHA256)
        native_source = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        for name, digest in hashes.items():
            self.assertIn(f'{{"{name}","{digest}"}}', native_source)
        receipt = PREFLIGHT.cuda_regression_fixture_receipt_fields(hashes)
        self.assertEqual(receipt["cuda_regression_fixture_status"], "PASS")
        self.assertEqual(receipt["cuda_regression_fixture_count"], "5")
        self.assertEqual(len(receipt["cuda_regression_fixture_set_sha256"]), 64)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            receipt_path = Path(temporary) / "receipt.txt"
            PREFLIGHT.write_receipt(receipt_path, receipt)
            written = PREFLIGHT.parse_quoted_fields(receipt_path, "RPJAPREFLIGHT5")
            self.assertEqual(written["cuda_regression_fixture_1_name"], next(iter(hashes)))
            self.assertEqual(written["cuda_regression_fixture_1_sha256"], next(iter(hashes.values())))
        launcher = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(launcher.index("fixture_hashes_before = validate_cuda_regression_fixtures"),
                        launcher.index("subprocess.run(native_fit_regression_command"))
        self.assertLess(launcher.index("fixture_hashes_after = validate_cuda_regression_fixtures"),
                        launcher.index("no_existing_outputs(output, resume=resume_mode, checkpoint_path=checkpoint_path,",
                                       launcher.index("fixture_hashes_after")))
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            copied_fixtures = Path(temporary)
            for name in PREFLIGHT.FROZEN_CUDA_QP_FIXTURE_SHA256:
                (copied_fixtures / name).write_bytes((fixture_directory / name).read_bytes())
            self.assertEqual(PREFLIGHT.validate_cuda_regression_fixtures(copied_fixtures), hashes)
            altered = copied_fixtures / next(iter(hashes))
            altered.write_bytes(altered.read_bytes() + b"changed")
            with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                PREFLIGHT.validate_cuda_regression_fixtures(copied_fixtures)

    def test_large_sample_fnv_verification_is_deferred_to_native_fit(self):
        launcher = SCRIPT.read_text(encoding="utf-8")
        native_fit = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
        self.assertNotIn("fnv64(sample)", launcher)
        self.assertNotIn("fnv64(blind)", launcher)
        self.assertIn("rpmd_ja_model_fingerprint(training_path)", native_fit)
        self.assertIn("rpmd_ja_model_fingerprint(blind_path)", native_fit)

    def test_paths_use_canonical_file_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            (base / "blind.bin").write_bytes(b"blind")
            (base / "other.bin").write_bytes(b"other")
            self.assertTrue(PREFLIGHT.same_resolved_path("blind.bin", "./blind.bin", base))
            self.assertFalse(PREFLIGHT.same_resolved_path("blind.bin", "other.bin", base))

    def test_training_and_blind_reject_same_file_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training = base / "train.bin"
            training.write_bytes(b"training")
            with self.assertRaises(SystemExit):
                PREFLIGHT.validate_distinct_training_blind(training, base / "." / "train.bin",
                                                            "0" * 64, "1" * 64)

    def test_training_and_blind_reject_identical_content_copies(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training, blind = base / "train.bin", base / "blind.bin"
            training.write_bytes(b"same trajectory")
            blind.write_bytes(b"same trajectory")
            digest = PREFLIGHT.sha256(training)
            with self.assertRaises(SystemExit):
                PREFLIGHT.validate_distinct_training_blind(training, blind, digest, digest)

    def test_training_and_blind_accept_distinct_content(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training, blind = base / "train.bin", base / "blind.bin"
            training.write_bytes(b"training trajectory")
            blind.write_bytes(b"independent blind trajectory")
            PREFLIGHT.validate_distinct_training_blind(training, blind, PREFLIGHT.sha256(training),
                                                        PREFLIGHT.sha256(blind))

    def test_blind_capacity_rejects_short_file_before_native_launch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training, blind = base / "train.bin", base / "blind.bin"
            self.write_sample(training, atoms=2, frames=300)
            self.write_sample(blind, atoms=2, frames=3)
            with self.assertRaises(SystemExit):
                PREFLIGHT.validate_blind_probe_capacity(training, blind)
            self.write_sample(blind, atoms=2, frames=4)
            self.assertEqual(PREFLIGHT.validate_blind_probe_capacity(training, blind), 3)

    def test_blind_capacity_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training, blind = base / "train.bin", base / "blind.bin"
            self.write_sample(training, atoms=2, frames=300)
            self.write_sample(blind, atoms=2, frames=4)
            with blind.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaises(SystemExit):
                PREFLIGHT.validate_blind_probe_capacity(training, blind)

    def test_blind_temperature_uses_native_relative_tolerance(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training, blind = base / "train.bin", base / "blind.bin"
            self.write_sample(training, atoms=2, frames=300)
            self.write_sample(blind, atoms=2, frames=4, temperature=300.000000001)
            self.assertEqual(PREFLIGHT.validate_blind_probe_capacity(training, blind), 3)
            self.write_sample(blind, atoms=2, frames=4, temperature=300.0001)
            with self.assertRaises(SystemExit):
                PREFLIGHT.validate_blind_probe_capacity(training, blind)

    def test_blind_header_rejects_cell_mass_and_type_mismatches(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            training, blind = base / "train.bin", base / "blind.bin"
            self.write_sample(training, atoms=2, frames=300)
            mismatches = (
                {"cell": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0]},
                {"masses": [16.0, 2.0]},
                {"types": [0, 2]},
            )
            for identity in mismatches:
                with self.subTest(identity=identity):
                    self.write_sample(blind, atoms=2, frames=4, **identity)
                    with self.assertRaises(SystemExit):
                        PREFLIGHT.validate_blind_probe_capacity(training, blind)

    def test_blind_capacity_preflight_precedes_cuda_regression_and_native_launch(self):
        launcher = SCRIPT.read_text(encoding="utf-8")
        capacity_check = launcher.index("validate_blind_probe_capacity(resolved_files")
        cuda_regression = launcher.index("Running the declared numerical CUDA regression")
        native_launch = launcher.index("Launching GPUMD with the checked run.in")
        self.assertLess(capacity_check, cuda_regression)
        self.assertLess(capacity_check, native_launch)


if __name__ == "__main__":
    unittest.main()
