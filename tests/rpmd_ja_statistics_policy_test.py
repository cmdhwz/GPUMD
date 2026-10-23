#!/usr/bin/env python3
"""CPU regressions for the versioned RPMD-JA statistics and provenance helpers."""

from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rpmd_ja_sampling_provenance as provenance
import rpmd_ja_statistics_policy as statistics_policy


def ar1(phi, count, seed):
    rng = random.Random(seed)
    value = 0.0
    series = []
    for _ in range(count):
        value = phi * value + rng.gauss(0.0, 1.0)
        series.append(value)
    return series


def configuration_metadata():
    digest = "a" * 64
    return {
        "integrator_type": "PIMD",
        "timestep_internal": 0.01,
        "timestep_internal_unit": "GPUMD_INTERNAL_TIME",
        "timestep_fs": 0.1018051,
        "timestep_fs_unit": "fs",
        "thermostat_configuration_tokens": [["ensemble", "pimd", "300", "pile"]],
        "model_sha256": digest,
        "run_input_sha256": digest,
        "physical_force_configuration_sha256": digest,
        "integrator_configuration_sha256": digest,
        "atom_signature_sha256": digest,
        "physical_force_source": "MEAN_OF_PHYSICAL_PIMD_BEAD_FORCES",
        "active_configuration_scope": "PREFIX_BEFORE_ACTIVE_RUN",
        "active_run_ordinal": 0,
        "internal_mass_com": True,
        "sampling_com_method": "MASS_WEIGHTED_CUMULATIVE_CENTROID_DRIFT_REMOVAL",
        "sampling_com_shift_final": [0.01, -0.02, 0.03],
        "configuration_stable_during_run": True,
    }


class StatisticsPolicyTests(unittest.TestCase):
    def test_policy_parameters_are_versioned_and_immutable(self):
        contract = statistics_policy.frozen_contract()
        self.assertEqual(contract["version"], "RPJASTAT2")
        self.assertEqual(contract["autocorrelation_rule"],
                         "FIXED_TRAINING_PHYSICAL_AND_PARAMETER_CHANNEL_SOKAL_WINDOW_V2")
        self.assertEqual(contract["adjacent_se_relative_tolerance"], 0.25)
        with self.assertRaises(TypeError):
            statistics_policy.FROZEN["adjacent_se_relative_tolerance"] = 0.5

    def test_training_series_selects_correlation_aware_adjacent_blocks(self):
        plan = statistics_policy.plan_training_blocks({
            "physical_mean_x": ar1(0.4, 8192, 10),
            "physical_second_x": ar1(0.2, 8192, 11),
        })
        self.assertEqual(plan["status"], "PASS")
        self.assertGreater(plan["tau_frames"], 1.0)
        self.assertEqual(plan["adjacent_block_length"], 2 * plan["block_length"])
        self.assertGreaterEqual(plan["block_count"], 4)
        self.assertGreaterEqual(plan["adjacent_block_count"], 4)
        self.assertLessEqual(plan["max_adjacent_se_relative_change"], 0.25)

    def test_unresolved_long_correlation_and_constant_channels_are_inconclusive(self):
        long = statistics_policy.plan_training_blocks({"slow": ar1(0.995, 512, 12)})
        self.assertEqual(long["status"], "DATA_INCONCLUSIVE")
        constant = statistics_policy.plan_training_blocks({"zero": [0.0] * 1024})
        self.assertEqual(constant["status"], "DATA_INCONCLUSIVE")
        self.assertEqual(constant["channels"][0]["state"], "CONSTANT_ZERO")

    def test_constant_zero_channel_change_is_not_hidden_by_a_variance_floor(self):
        training = {"constant_zero": [0.0] * 8192, "varying": ar1(0.1, 8192, 13)}
        development = {"constant_zero": [0.0] * 8192, "varying": ar1(0.1, 8192, 14)}
        plan = statistics_policy.plan_training_blocks(training)
        self.assertEqual(plan["status"], "PASS")
        stable = statistics_policy.assess_fixed_r0_means(training, development, plan)
        self.assertEqual(stable["status"], "PASS")
        self.assertEqual(stable["channels"][0]["state"], "CONSTANT_ZERO")

        changed = dict(development, constant_zero=[0.25] * 8192)
        drift = statistics_policy.assess_fixed_r0_means(training, changed, plan)
        self.assertEqual(drift["status"], "DATA_INCONCLUSIVE")
        self.assertEqual(drift["channels"][0]["state"], "CONSTANT_CHANGED")
        self.assertEqual(drift["channels"][0]["delta_z"], float("inf"))

    def test_multichannel_global_drift_threshold_avoids_unadjusted_two_se_gate(self):
        training = {f"mean_{index}": ar1(0.0, 4096, index) for index in range(16)}
        development = {
            name: ar1(0.0, 4096, 100 + index)
            for index, name in enumerate(training)
        }
        plan = statistics_policy.plan_training_blocks(training)
        self.assertEqual(plan["status"], "PASS")
        stable = statistics_policy.assess_fixed_r0_means(
            training, development, plan, expected_zero_mean_channels=training.keys()
        )
        self.assertEqual(stable["status"], "PASS")
        self.assertGreater(stable["z_tolerance"], 2.0)
        self.assertEqual(stable["comparison_count"], 96)
        self.assertEqual(stable["threshold_method"], "MAX_Z_SUBGAUSSIAN_UNION_ENGINEERING_V1")

        shifted = dict(development, mean_0=[value + 0.5 for value in development["mean_0"]])
        drift = statistics_policy.assess_fixed_r0_means(
            training, shifted, plan, expected_zero_mean_channels=training.keys()
        )
        self.assertEqual(drift["status"], "DATA_INCONCLUSIVE")
        self.assertGreater(drift["maximum_z"], drift["z_tolerance"])

    def test_fixed_r0_gate_uses_both_means_and_propagated_covariance(self):
        training = {"q_x": ar1(0.1, 4096, 20)}
        development = {"q_x": ar1(0.1, 4096, 21)}
        plan = statistics_policy.plan_training_blocks(training)
        self.assertEqual(plan["status"], "PASS")
        stable = statistics_policy.assess_fixed_r0_means(training, development, plan)
        self.assertEqual(stable["status"], "PASS")
        self.assertEqual(stable["dimension_count"], 1)
        self.assertEqual(len(stable["covariance_of_mean_difference"]), 1)
        self.assertGreater(stable["channels"][0]["se_train"], 0.0)
        self.assertGreater(stable["channels"][0]["se_development"], 0.0)

        shifted = statistics_policy.assess_fixed_r0_means(
            training, {"q_x": [value + 0.4 for value in development["q_x"]]}, plan
        )
        self.assertEqual(shifted["status"], "DATA_INCONCLUSIVE")
        self.assertGreater(shifted["channels"][0]["delta_z"], 2.0)

    def test_fixed_r0_gate_includes_anomalous_tail_outside_common_block_prefix(self):
        training = {"q_x": ar1(0.1, 4099, 22)}
        plan = statistics_policy.plan_training_blocks(training)
        self.assertEqual(plan["status"], "PASS")
        common_prefix = (len(training["q_x"]) // plan["adjacent_block_length"])
        common_prefix *= plan["adjacent_block_length"]
        self.assertLess(common_prefix, len(training["q_x"]))
        development_values = list(training["q_x"])
        development_values[common_prefix:] = [1000.0] * (len(development_values) - common_prefix)

        drift = statistics_policy.assess_fixed_r0_means(
            training, {"q_x": development_values}, plan
        )
        self.assertEqual(drift["common_prefix_frames_development"], common_prefix)
        self.assertEqual(drift["channels"][0]["mean_development"], sum(development_values) / len(development_values))
        self.assertEqual(drift["status"], "DATA_INCONCLUSIVE")
        self.assertGreater(drift["channels"][0]["delta_z"], drift["z_tolerance"])

    def test_development_longer_correlation_cannot_reselect_training_blocks(self):
        training = {"q_x": ar1(0.1, 16384, 30)}
        development = {"q_x": ar1(0.95, 16384, 31)}
        plan = statistics_policy.plan_training_blocks(training)
        self.assertEqual(plan["status"], "PASS")
        result = statistics_policy.assess_fixed_r0_means(training, development, plan)
        self.assertEqual(result["status"], "DATA_INCONCLUSIVE")
        self.assertEqual(result["reason"], "DEVELOPMENT_BLOCK_SENSITIVE")
        self.assertGreater(result["development_adjacent_se_relative_change"], 0.25)

    def test_paired_choice_requires_resolved_same_winner_at_both_block_lengths(self):
        stable = statistics_policy.assess_paired_choice(
            [-1.0] * 8, [-0.8] * 4
        )
        self.assertEqual(stable["status"], "A_WINS")
        switched = statistics_policy.assess_paired_choice(
            [-1.0] * 8, [1.0] * 4
        )
        self.assertEqual(switched["status"], "DATA_INCONCLUSIVE")
        unresolved = statistics_policy.assess_paired_choice(
            [-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]
        )
        self.assertEqual(unresolved["status"], "DATA_INCONCLUSIVE")


class SamplingProvenanceTests(unittest.TestCase):
    def test_sample_hash_is_bound_and_legacy_is_not_attested(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            sample = Path(directory) / "samples.bin"
            sidecar = Path(directory) / "samples.bin.provenance.json"
            sample.write_bytes(b"GPJASMP1-fixture")
            document = provenance.make_sidecar(
                sample_bytes=sample.read_bytes(),
                sample_frames=12,
                source_job_id="scheduler:42",
                source_job_id_status="SCHEDULER_DECLARED",
                metadata=configuration_metadata(),
                source_run_id="run:42:1",
                step_start=100,
                step_end=111,
            )
            sidecar.write_text(provenance.dumps(document), encoding="utf-8")
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("sample must be streamed")):
                resolved = provenance.resolve(sample, sidecar, configuration_metadata())
            self.assertEqual(resolved["status"], "PRODUCER_OBSERVED")
            self.assertTrue(resolved["checksum_valid"])
            self.assertTrue(resolved["required_configuration_fields_present"])
            self.assertEqual(resolved["historical_match_status"], "MATCHED_DECLARED_VALUES")
            self.assertEqual(resolved["independence_attestation"], "NOT_ASSERTED")
            config_attestation = provenance.validate_configuration_attestation(resolved, {
                "schema": provenance.CONFIG_ATTESTATION_SCHEMA,
                "sample_sha256": resolved["sample_sha256"],
                "configuration_sha256": resolved["configuration_sha256"],
                "attestor": "external-record",
                "evidence_sha256": "d" * 64,
            })
            self.assertEqual(config_attestation["status"], "EXTERNAL_ATTESTATION_DECLARED")
            self.assertEqual(
                config_attestation["configuration_attestation"],
                "EXTERNAL_ATTESTATION_DECLARED_UNAUTHENTICATED",
            )

            sample.write_bytes(b"changed")
            mismatch = provenance.resolve(sample, sidecar)
            self.assertEqual(mismatch["status"], "HASH_MISMATCH")

            legacy = provenance.resolve(sample, sidecar.with_suffix(".missing"))
            self.assertEqual(legacy["status"], "LEGACY_DECLARED_UNVERIFIED")
            self.assertIsNone(legacy["checksum_valid"])
            self.assertFalse(legacy["required_configuration_fields_present"])
            self.assertEqual(legacy["independence_attestation"], "NOT_ASSERTED")

    def test_checksum_valid_but_incomplete_configuration_is_not_verifiable(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            sample = Path(directory) / "samples.bin"
            sidecar = Path(directory) / "samples.bin.provenance.json"
            sample.write_bytes(b"sample")
            document = provenance.make_sidecar(
                sample.read_bytes(), 3, "job:1", "SCHEDULER_DECLARED", {"thermostat": "PILE"}
            )
            sidecar.write_text(provenance.dumps(document), encoding="utf-8")
            resolved = provenance.resolve(sample, sidecar)
            self.assertTrue(resolved["checksum_valid"])
            self.assertEqual(resolved["producer_metadata_status"], "PRODUCER_OBSERVED")
            self.assertFalse(resolved["required_configuration_fields_present"])
            self.assertEqual(resolved["historical_match_status"], "NOT_VERIFIABLE")

    def test_independence_attestation_rejects_same_job_and_overlapping_steps(self):
        common = {
            "schema": provenance.ATTESTATION_SCHEMA,
            "training_sha256": "a" * 64,
            "blind_sha256": "b" * 64,
            "training_source_id": "run:1",
            "blind_source_id": "run:2",
            "training_job_id": "job:1",
            "blind_job_id": "job:2",
            "training_run_id": "run:1",
            "blind_run_id": "run:2",
            "training_step_start": 0,
            "training_step_end": 100,
            "blind_step_start": 200,
            "blind_step_end": 300,
            "attestor": "external-record",
            "evidence_sha256": "c" * 64,
        }
        training = {
            "sample_sha256": "a" * 64, "source_job_id": "job:1", "source_run_id": "run:1",
            "step_start": 0, "step_end": 100,
        }
        blind = {
            "sample_sha256": "b" * 64, "source_job_id": "job:2", "source_run_id": "run:2",
            "step_start": 200, "step_end": 300,
        }
        valid = provenance.validate_independence_attestation(training, blind, common)
        self.assertEqual(valid["status"], "EXTERNAL_ATTESTATION_DECLARED")
        self.assertEqual(valid["independence_attestation"], "EXTERNAL_ATTESTATION_DECLARED_UNAUTHENTICATED")

        same_job = dict(common, blind_job_id="job:1")
        same_job_blind = dict(blind, source_job_id="job:1")
        self.assertEqual(
            provenance.validate_independence_attestation(training, same_job_blind, same_job)["status"],
            "ATTESTATION_SOURCE_COLLISION",
        )
        overlap = dict(common, blind_step_start=100, blind_step_end=200)
        overlap_blind = dict(blind, step_start=100, step_end=200)
        self.assertEqual(
            provenance.validate_independence_attestation(training, overlap_blind, overlap)["status"],
            "ATTESTATION_STEP_OVERLAP",
        )

    def test_distinct_source_strings_without_job_step_evidence_are_inconclusive(self):
        attestation = {
            "schema": provenance.ATTESTATION_SCHEMA,
            "training_sha256": "a" * 64,
            "blind_sha256": "b" * 64,
            "training_source_id": "run-a",
            "blind_source_id": "run-b",
            "attestor": "declared",
            "evidence_sha256": "c" * 64,
        }
        result = provenance.validate_independence_attestation(
            {"sample_sha256": "a" * 64}, {"sample_sha256": "b" * 64}, attestation
        )
        self.assertEqual(result["status"], "INVALID_ATTESTATION")
        self.assertEqual(result["independence_attestation"], "NOT_ASSERTED")


if __name__ == "__main__":
    unittest.main()
