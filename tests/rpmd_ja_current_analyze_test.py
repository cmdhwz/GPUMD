import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rpmd_ja_current_analyze as analyze


class CurrentAnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.directory = Path(self.temp.name)
        self.path = self.directory / "heat_current_rpmd_ja.out"
        self.frames = 400
        self.dt_ps = analyze.NATIVE_TIME_PS * 0.5
        index = np.arange(self.frames, dtype=np.float64)
        self.jc = np.column_stack((np.sin(0.37 * index), np.cos(0.19 * index), np.sin(0.11 * index + 0.3)))

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, delta, use="FORMAL", response="NOT_APPLICABLE"):
        delta = np.asarray(delta, dtype=np.float64)
        ja = self.jc + delta
        sha = hashlib.sha256(b"reference").hexdigest()
        with self.path.open("w", encoding="ascii") as stream:
            stream.write("# segment_metadata_version 2\n")
            stream.write(f"# reference_package_sha256 {sha}\n")
            stream.write("# backend qnep-block-v3\n")
            if use == "DIAGNOSTIC":
                stream.write("# mechanical_policy native_reference_transport;finite_temperature_additive_v1;beads=1;derivative=1;"
                             f"use=DIAGNOSTIC;response={response};numerical=PASS\n")
            else:
                stream.write("# mechanical_policy native_reference_transport\n")
            stream.write(f"# reference_use_class {use}\n")
            stream.write(f"# reference_numerical_status {'PASS' if use == 'DIAGNOSTIC' else 'NOT_APPLICABLE'}\n")
            stream.write(f"# reference_response_status {response}\n")
            stream.write(f"# sample_frames {self.frames}\n")
            stream.write("# T_reference_K 300\n# cell_volume_A3 1000\n")
            stream.write("# current_units eV*Angstrom/natural_time\n")
            stream.write("# columns time_ps Jcent_x Jcent_y Jcent_z DeltaJ_x DeltaJ_y DeltaJ_z JA_x JA_y JA_z\n")
            for frame in range(self.frames):
                row = [((frame + 1) * self.dt_ps), *self.jc[frame], *delta[frame], *ja[frame]]
                stream.write(" ".join(f"{value:.17g}" for value in row) + "\n")

    def _write_header_only(self, declared_frames):
        sha = hashlib.sha256(b"reference").hexdigest()
        self.path.write_text(
            "# segment_metadata_version 2\n"
            f"# reference_package_sha256 {sha}\n"
            "# backend qnep-block-v3\n"
            "# mechanical_policy native_reference_transport\n"
            "# reference_use_class FORMAL\n"
            "# reference_numerical_status NOT_APPLICABLE\n"
            "# reference_response_status NOT_APPLICABLE\n"
            f"# sample_frames {declared_frames}\n"
            "# T_reference_K 300\n# cell_volume_A3 1000\n"
            "# current_units eV*Angstrom/natural_time\n"
            "# columns time_ps Jcent_x Jcent_y Jcent_z DeltaJ_x DeltaJ_y DeltaJ_z JA_x JA_y JA_z\n",
            encoding="ascii")

    def test_zero_correction_correlations_and_paired_ci(self):
        self._write(np.zeros_like(self.jc))
        report, arrays = analyze.analyze_current(self.path, 1e-9, 12, 2)
        self.assertEqual(report["equivalence_status"], "EQUIVALENT")
        np.testing.assert_allclose(arrays["CcDelta"], 0.0, atol=0.0)
        np.testing.assert_allclose(arrays["CDeltac"], 0.0, atol=0.0)
        np.testing.assert_allclose(arrays["CDeltaDelta"], 0.0, atol=0.0)
        np.testing.assert_allclose(arrays["Caa"], arrays["Ccc"], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(arrays["delta_kappa_W_m_K"], 0.0, atol=1e-14)
        self.assertGreaterEqual(report["complete_2b_blocks"], 8)
        self.assertEqual(report["hac_integral_units"], "W/m/K")
        self.assertEqual(report["format"], "RPJACURRENT2_REPORT")
        self.assertEqual(report["arrays_format"], "RPJACURRENT2_NPZ")
        self.assertNotIn("Caa_decomposed", arrays)
        self.assertNotIn("bootstrap_delta_kappa_W_m_K", arrays)
        self.assertLess(report["ja_correlation_decomposition_max_absolute_error"], 1e-15)
        self.assertIn("does not establish", report["delta_j_rms_role"])
        self.assertEqual(report["correlation_definition"].split("; ")[1], "denominator N-lag")
        self.assertEqual(set(report["delta_kappa_equivalence_by_direction"].values()), {"EQUIVALENT"})

    def test_delta_minus_jc_cross_terms_cancel_total_current(self):
        self._write(-self.jc)
        report, arrays = analyze.analyze_current(self.path, 1e-9, 12, 2)
        np.testing.assert_allclose(arrays["Caa"], 0.0, atol=1e-15)
        np.testing.assert_allclose(arrays["Ccc"] + arrays["CcDelta"] + arrays["CDeltac"] + arrays["CDeltaDelta"],
                                   arrays["Caa"], atol=1e-15)
        np.testing.assert_allclose(arrays["kappa_Caa_W_m_K"], 0.0, atol=1e-14)
        np.testing.assert_allclose(arrays["delta_kappa_W_m_K"], -arrays["kappa_Ccc_W_m_K"], atol=1e-13)
        self.assertLess(report["ja_correlation_decomposition_max_absolute_error"], 1e-14)

    def test_native_hac_factor_matches_running_integral(self):
        constant = np.tile([2.0, 3.0, 4.0], (self.frames, 1))
        self._write(np.zeros_like(constant))
        with self.path.open("r", encoding="ascii") as stream:
            lines = stream.readlines()
        for row in range(self.frames):
            values = [float(value) for value in lines[-self.frames + row].split()]
            values[1:4] = constant[row]
            values[4:7] = [0.0, 0.0, 0.0]
            values[7:10] = constant[row]
            lines[-self.frames + row] = " ".join(f"{value:.17g}" for value in values) + "\n"
        self.path.write_text("".join(lines), encoding="ascii")
        _report, arrays = analyze.analyze_current(self.path, 1.0, 12, 2)
        expected_one_step = constant[0] ** 2 * (self.dt_ps / analyze.NATIVE_TIME_PS) / (
            analyze.KB_EV_K * 300.0**2 * 1000.0) * analyze.KAPPA_UNIT_CONVERSION
        np.testing.assert_allclose(arrays["kappa_Ccc_W_m_K"][1], expected_one_step, rtol=2e-14)
        np.testing.assert_allclose(arrays["kappa_Ccc_W_m_K"][2], 2.0 * expected_one_step, rtol=2e-14)

    def test_too_few_blocks_is_inconclusive_and_diagnostic_needs_selection(self):
        short = np.zeros((20, 3))
        self.jc = self.jc[:20]
        self.frames = 20
        self._write(short)
        report, _arrays = analyze.analyze_current(self.path, 1.0, 12, 2)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        self._write(short, use="DIAGNOSTIC", response="FAIL")
        with self.assertRaisesRegex(ValueError, "explicit --diagnostic"):
            analyze.analyze_current(self.path, 1.0, 4, 2)
        report, _arrays = analyze.analyze_current(self.path, 1.0, 12, 2, allow_diagnostic=True)
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(report["reference_response_status"], "FAIL")

    def test_inadequate_block_or_lag_window_is_inconclusive(self):
        self._write(np.zeros_like(self.jc))
        report, _arrays = analyze.analyze_current(self.path, 1e-9, 2, 2)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        self.assertEqual(report["equivalence_reason"], "BLOCK_SHORTER_THAN_FOUR_CORRELATION_WINDOWS")
        report, _arrays = analyze.analyze_current(self.path, 1e-9, 12, 0)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        self.assertEqual(report["equivalence_reason"], "CORRELATION_WINDOW_HAS_NO_LAG")
        self.assertEqual(report["bootstrap_samples"], 0)

    def test_uncovered_hot_tail_cannot_claim_equivalence(self):
        self.frames = 401
        self.jc = np.zeros((self.frames, 3), dtype=np.float64)
        delta = np.zeros_like(self.jc)
        delta[-1] = [100.0, 100.0, 100.0]
        self._write(delta)
        report, _arrays = analyze.analyze_current(self.path, 1e-9, 12, 2)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        self.assertEqual(report["equivalence_reason"], "DECISION_INTERVAL_CROSSES_OR_SPANS_TOLERANCE")
        self.assertEqual(report["bootstrap_tail_frames_omitted"], 17)
        self.assertTrue(any(abs(value) > 1e-9 for value in report["full_series_minus_common_prefix_delta_kappa_W_m_K"]))

    def test_original_short_block_hot_tail_repro_is_inconclusive(self):
        self.frames = 41
        self.dt_ps = analyze.NATIVE_TIME_PS
        self.jc = np.zeros((self.frames, 3), dtype=np.float64)
        delta = np.zeros_like(self.jc)
        delta[-1] = [100.0, 100.0, 100.0]
        self._write(delta)
        report, _arrays = analyze.analyze_current(self.path, 1e-9, 5, 2)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        self.assertEqual(report["equivalence_reason"], "BLOCK_SHORTER_THAN_FOUR_CORRELATION_WINDOWS")
        for value in report["kappa_full_series_at_max_lag_W_m_K"]["DeltaKappa"]:
            self.assertAlmostEqual(value, 2474.6355341584963, delta=1e-9)

    def test_full_series_offset_expands_decision_interval(self):
        self.frames = 401
        self.dt_ps = analyze.NATIVE_TIME_PS
        self.jc = np.zeros((self.frames, 3), dtype=np.float64)
        k = analyze.KAPPA_UNIT_CONVERSION / (analyze.KB_EV_K * 300.0**2 * 1000.0)
        amplitude = np.sqrt(0.6 / (2.0 * k))
        delta = np.full((self.frames, 3), amplitude)
        def full_delta_kappa(last_value):
            candidate = delta.copy()
            candidate[-1] = last_value
            correlation = analyze._correlate(candidate, candidate, 2)
            return analyze._running_kappa(correlation, 1.0, 300.0, 1000.0)[-1, 0]

        low, high = amplitude, amplitude * 100.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            if full_delta_kappa(middle) < 1.1:
                low = middle
            else:
                high = middle
        delta[-1] = 0.5 * (low + high)
        self._write(delta)
        report, _arrays = analyze.analyze_current(self.path, 1.0, 20, 2)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        for axis in "xyz":
            self.assertAlmostEqual(report["delta_kappa_ci95_W_m_K"][axis][0], 0.6, delta=1e-9)
            self.assertAlmostEqual(report["delta_kappa_ci95_W_m_K"][axis][1], 0.6, delta=1e-9)
            self.assertAlmostEqual(report["full_series_minus_2b_block_mean_delta_kappa_W_m_K"]["xyz".index(axis)],
                                   0.5, delta=1e-9)
            interval = report["delta_kappa_decision_interval_W_m_K"][axis]
            self.assertLess(interval[0], 1.0)
            self.assertGreater(interval[1], 1.0)
            self.assertAlmostEqual(report["kappa_full_series_at_max_lag_W_m_K"]["DeltaKappa"]["xyz".index(axis)],
                                   1.1, delta=1e-9)

    def test_invalid_windows_write_strict_json_through_cli(self):
        cases = ((20, 5, 2), (20, 12, 0), (20, 30, 2), (401, 100, 400))
        for case_index, (frames, block_frames, max_lag) in enumerate(cases):
            with self.subTest(frames=frames, block_frames=block_frames, max_lag=max_lag):
                self.frames = frames
                self.jc = np.zeros((frames, 3), dtype=np.float64)
                self._write(np.zeros_like(self.jc))
                prefix = self.directory / f"invalid_window_{case_index}"
                run = subprocess.run([
                    sys.executable, str(ROOT / "tools" / "rpmd_ja_current_analyze.py"), str(self.path),
                    "--output-prefix", str(prefix), "--epsilon-kappa", "1.0",
                    "--block-frames", str(block_frames), "--max-lag-frames", str(max_lag)],
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
                self.assertEqual(report["bootstrap_samples"], 0)
                self.assertTrue(all(values == [None, None] for values in
                                    report["delta_kappa_decision_interval_W_m_K"].values()))
                self.assertTrue(all(value is None for value in
                                    report["full_series_minus_2b_block_mean_delta_kappa_W_m_K"]))
                if frames == 401:
                    self.assertTrue(all(value is None for value in
                                        report["full_series_minus_common_prefix_delta_kappa_W_m_K"]))
                    self.assertTrue(all(value is None for value in
                                        report["full_series_minus_2b_block_mean_delta_kappa_W_m_K"]))

    def test_analysis_work_and_memory_limits_apply_before_reading_rows(self):
        self._write_header_only(10_000_000)
        with self.assertRaisesRegex(ValueError, "bounded correlation-work"):
            analyze.analyze_current(self.path, 1.0, 12, 2)
        self._write_header_only(15_000_000)
        with self.assertRaisesRegex(MemoryError, "estimated current analysis memory"):
            analyze.analyze_current(self.path, 1.0, 12, 0)

    def test_confidence_interval_crossing_margin_is_inconclusive(self):
        self.frames = 400
        self.jc = np.zeros((self.frames, 3), dtype=np.float64)
        rng = np.random.default_rng(9)
        amplitude = 1.0 + 0.5 * rng.normal(size=self.frames)
        self._write(np.repeat(amplitude[:, None], 3, axis=1))
        report, _arrays = analyze.analyze_current(self.path, 20.0, 12, 2)
        interval = report["delta_kappa_ci95_W_m_K"]["x"]
        self.assertLess(interval[0], 20.0)
        self.assertGreater(interval[1], 20.0)
        self.assertEqual(report["equivalence_status"], "INCONCLUSIVE")
        self.assertEqual(report["equivalence_reason"], "DECISION_INTERVAL_CROSSES_OR_SPANS_TOLERANCE")
        self.assertEqual(report["delta_kappa_equivalence_by_direction"]["x"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
