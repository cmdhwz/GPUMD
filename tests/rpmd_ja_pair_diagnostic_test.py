from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rpmd_ja_pair_diagnostic as diagnostic


def _write_spool(path, frames, types, cell_length=10.0, cell=None):
    n = len(frames[0])
    with path.open("wb") as stream:
        stream.write(b"GPJASMP1")
        stream.write(struct.pack("<IIiid", 1, 0x01020304, n, 32, 300.0))
        (np.eye(3, dtype="<f8") * cell_length if cell is None else np.asarray(cell, dtype="<f8")).tofile(stream)
        np.ones(n, dtype="<f8").tofile(stream)
        np.asarray(types, dtype="<i4").tofile(stream)
        for step, xyz in enumerate(frames):
            stream.write(struct.pack("<d", float(step)))
            np.asarray(xyz, dtype="<f8").T.ravel().tofile(stream)
            np.zeros((3, n), dtype="<f8").tofile(stream)


class PairDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spool = self.root / "samples.bin"
        frames = []
        for i in range(12):
            pair_x = 0.5 if i < 8 and i % 2 == 0 else 0.7 if i < 8 else 5.2
            neighbor_x = 2.0 if i < 8 else 2.5
            frames.append([[9.4, 1.0, 1.0], [pair_x, 1.0, 1.0], [neighbor_x, 1.0, 1.0]])
        _write_spool(self.spool, frames, types=[2, 2, 1])

    def test_fixed_r0_image_train_validation_and_frozen_midpoint_neighbor(self):
        original = self.spool.read_bytes()
        report = diagnostic.diagnose(self.spool, pairs=[(0, 1)], cutoff=5.0)
        json.dumps(report, allow_nan=False)
        self.assertEqual(self.spool.read_bytes(), original)
        self.assertEqual(report["sample_step_interval"], 1.0)
        pair = report["pairs"][0]
        self.assertEqual(pair["atom_types"], [2, 2])
        self.assertEqual(pair["reference_image"], [1, 0, 0])
        self.assertAlmostEqual(pair["R0_pair_vector_A"][0], 1.2)
        self.assertEqual(pair["R0_pair_vector_A"][1:], [0.0, 0.0])
        self.assertAlmostEqual(-8.8 + pair["reference_image"][0] * 10.0, pair["R0_pair_vector_A"][0])
        self.assertAlmostEqual(pair["R0_pair_distance_A"], 1.2)
        self.assertAlmostEqual(pair["R0_cutoff_margin_A"], 3.8)
        self.assertAlmostEqual(pair["training"]["mean_drift_vector_A"][0], 0.0)
        self.assertAlmostEqual(pair["training"]["centered_variance_components_A2"][0], 0.01)
        self.assertAlmostEqual(pair["validation"]["mean_drift_vector_A"][0], 4.6)
        self.assertAlmostEqual(pair["validation"]["second_moment_about_R0_A2"][0][0], 21.16)
        self.assertAlmostEqual(pair["validation"]["distance_A"]["median"], 5.8)
        self.assertEqual(len(pair["training_blocks"]), 4)
        self.assertEqual([block["frame_count"] for block in pair["training_blocks"]], [2, 2, 2, 2])
        self.assertEqual(pair["branch_warnings"]["validation"]["warning_frames"], 4)
        self.assertEqual(pair["branch_warnings"]["training"]["warning_frames"], 0)
        neighbor = pair["midpoint_neighbors_by_type"][0]
        self.assertEqual((neighbor["type_id"], neighbor["atom_id"]), (1, 2))
        self.assertEqual(neighbor["reference_image"], [1, 0, 0])
        self.assertAlmostEqual(neighbor["R0_distance_A"], 2.0)
        self.assertAlmostEqual(neighbor["training_distance_A"]["median"], 2.0)
        self.assertAlmostEqual(neighbor["validation_distance_A"]["median"], 0.2)
        joint = pair["joint_coordinate_statistics"]
        self.assertAlmostEqual(joint["validation"]["mean"][0], 4.6)
        self.assertAlmostEqual(joint["training"]["centered_covariance"][0][3], -0.005)
        self.assertEqual(joint["coordinate_labels"][0]["atom_ids"], [0, 1])
        self.assertEqual(report["quadratic_statistics_by_type_group"][0]["support_status"],
                         "SPARSE_REQUESTED_PAIR_SET")
        self.assertEqual(report["performance"]["memory_bytes"]["retained_distance_samples"], 0)

    def test_user_atoms_and_selected_column_reader(self):
        frames = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                  for _ in range(6)]
        _write_spool(self.spool, frames, types=[2, 2, 1, 2])
        spool = diagnostic.read_spool(self.spool)
        full = diagnostic._positions(spool["frames"], 0, 4, spool["n"])
        selected = diagnostic._positions(spool["frames"], 0, 4, spool["n"], np.asarray([0, 2]))
        np.testing.assert_array_equal(selected, full[:, [0, 2], :])

        report = diagnostic.diagnose(self.spool, pairs=[(0, 1)], cutoff=5.0, atoms=[3])
        pair = report["pairs"][0]
        self.assertEqual(report["user_selected_atoms"], [3])
        self.assertEqual(pair["joint_coordinate_statistics"]["dimension"], 9)
        self.assertEqual([(item["atom_id"], item["selection"]) for item in pair["midpoint_neighbors_by_type"]],
                         [(2, "midpoint_nearest_other_type"), (3, "user_selected")])

    def test_skew_cell_fixed_image_sign(self):
        cell = np.asarray([[10.0, 0.0, 0.0], [4.0, 9.0, 0.0], [0.0, 0.0, 10.0]])
        origin = np.asarray([1.0, 1.0, 1.0])
        raw = np.asarray([0.6, -0.7, 0.0]) @ cell.T
        neighbor = origin + np.asarray([0.2, 0.1, 0.0]) @ cell.T
        frames = [[origin, origin + raw, neighbor] for _ in range(6)]
        _write_spool(self.spool, frames, types=[2, 2, 1], cell=cell)
        pair = diagnostic.diagnose(self.spool, pairs=[(0, 1)], cutoff=5.0)["pairs"][0]
        self.assertEqual(pair["reference_image"], [-1, 1, 0])
        expected = raw + np.asarray(pair["reference_image"]) @ cell.T
        np.testing.assert_allclose(pair["R0_pair_vector_A"], expected)

    def test_midpoint_neighbor_branch_warning_with_stable_pair(self):
        frames = [[[9.4, 1.0, 1.0], [0.5, 1.0, 1.0],
                   [2.0 if index < 8 else 7.0, 1.0, 1.0]] for index in range(12)]
        _write_spool(self.spool, frames, types=[2, 2, 1])
        pair = diagnostic.diagnose(self.spool, pairs=[(0, 1)], cutoff=5.0)["pairs"][0]
        self.assertEqual(pair["branch_warnings"]["validation"]["warning_frames"], 0)
        neighbor = pair["midpoint_neighbors_by_type"][0]
        self.assertEqual(neighbor["atom_id"], 2)
        self.assertEqual(neighbor["branch_warnings"]["validation"]["warning_frames"], 4)
        self.assertEqual(neighbor["branch_warnings"]["training"]["warning_frames"], 0)
        self.assertIn("Fixed-R0 midpoint-neighbor linear coordinate displacement",
                      neighbor["branch_warnings"]["definition"])

    def test_combined_joint_covariance_keeps_shared_atom_cross_edge_term(self):
        frames = [[[0.0, 0.0, 0.0], [0.5 if index % 2 == 0 else 0.7, 0.0, 0.0], [2.0, 0.0, 0.0]]
                  for index in range(12)]
        _write_spool(self.spool, frames, types=[2, 2, 2])
        report = diagnostic.diagnose(self.spool, pairs=[(0, 1), (1, 2)], cutoff=5.0)
        combined = report["joint_coordinate_statistics"]
        self.assertEqual(combined["dimension"], 6)
        self.assertAlmostEqual(combined["training"]["centered_covariance"][0][3], -0.01)
        self.assertEqual(combined["coordinate_labels"][3]["requested_pair_index"], 1)
        self.assertNotAlmostEqual(report["pairs"][0]["joint_coordinate_statistics"]["training"]
                                  ["centered_covariance"][0][0],
                                  combined["training"]["centered_covariance"][0][3])

    def test_graph_support_uses_strict_cutoffs(self):
        positions = [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [5.0, 0.0, 0.0], [5.5, 0.0, 0.0]]
        frames = [positions for _ in range(6)]
        _write_spool(self.spool, frames, types=[2, 2, 2, 2], cell_length=20.0)
        group = diagnostic.diagnose(self.spool, pairs=[(0, 1), (0, 3)], cutoff=5.0)["quadratic_statistics_by_type_group"][0]
        cutoffs = group["fixed_R0_graph_support"]["cutoffs"]
        self.assertEqual([item["cutoff_A"] for item in cutoffs], [4.5, 5.0, 5.5])
        self.assertEqual([item["pair_count"] for item in cutoffs], [3, 4, 5])
        self.assertEqual([item["supported_atom_count"] for item in cutoffs], [3, 4, 4])
        self.assertEqual(group["requested_pair_count_inside_R0_graph"], 1)
        self.assertAlmostEqual(group["full_group_coverage_fraction"], 0.25)

    def test_quadratic_statistics_sum_requested_edges_and_report_group_coverage(self):
        frames = []
        for index in range(36):
            x = 0.8 if index % 2 == 0 else 1.2
            y = 0.3 if index % 2 == 0 else -0.3
            z_sign = (1, -1, 1, -1, -1, -1, 1, 1)[index % 8]
            z = 0.4 * z_sign
            frames.append([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [10.0, 0.0, 0.0],
                           [10.0 + x, y, z]])
        _write_spool(self.spool, frames, types=[2, 2, 2, 2], cell_length=20.0)
        report = diagnostic.diagnose(self.spool, pairs=[(0, 1), (2, 3)], cutoff=5.0)
        group = report["quadratic_statistics_by_type_group"][0]
        self.assertEqual(group["quadratic_components"], ["xx", "2xy", "2xz", "yy", "2yz", "zz"])
        self.assertTrue(group["requested_edge_scope"].startswith("Only user-requested pair edges"))
        self.assertEqual(group["requested_pair_count_inside_R0_graph"], 2)
        self.assertAlmostEqual(group["full_group_coverage_fraction"], 1.0)
        self.assertAlmostEqual(group["training"]["mean"][0], 0.04)
        self.assertEqual(group["training"]["temporal_status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(len(group["training"]["block_summaries"]), 3)
        self.assertAlmostEqual(group["training"]["mean"][4], 0.12)
        self.assertAlmostEqual(group["training"]["second_moment"][4][4], 0.0576)
        self.assertEqual([item["component"] for item in group["training"]["autocorrelation"]],
                         ["xx", "2xy", "2xz", "yy", "2yz", "zz"])
        self.assertEqual(group["training"]["autocorrelation"][4]["projection"]["status"],
                         "DIAGNOSTIC_ESTIMATE")
        self.assertEqual(len(group["training"]["covariance_eigenvalues"]), 6)
        self.assertGreaterEqual(group["training"]["rank_fraction"], 0.0)

    def test_output_refuses_overwrite(self):
        report_path = self.root / "report.json"
        report_path.write_text("keep me\n", encoding="utf-8")
        args = ["pair-diagnostic", str(self.spool), "--pair", "0:1", "--cutoff", "5",
                "--output", str(report_path)]
        with patch.object(sys, "argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            diagnostic.main()
        self.assertEqual(report_path.read_text(encoding="utf-8"), "keep me\n")

    def test_cli_writes_json_with_explicit_local_atoms(self):
        report_path = self.root / "new-report.json"
        args = ["pair-diagnostic", str(self.spool), "--pair", "0:1", "--atom", "2",
                "--cutoff", "5", "--output", str(report_path)]
        with patch.object(sys, "argv", args):
            diagnostic.main()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["user_selected_atoms"], [2])

    def test_block_summaries_balance_blocks_and_report_sem(self):
        values = np.arange(162.0).reshape(81, 2)
        summaries = diagnostic._block_summaries(values, step_interval=2.0)
        self.assertEqual([item["target_block_length_frames"] for item in summaries], [8, 32, 128])
        self.assertEqual(sum(summaries[0]["block_lengths_frames"]), len(values))
        self.assertLessEqual(max(summaries[0]["block_lengths_frames"])
                             - min(summaries[0]["block_lengths_frames"]), 1)
        self.assertEqual(summaries[0]["status"], "DIAGNOSTIC_ESTIMATE")
        self.assertEqual(len(summaries[0]["block_sem"]), 2)
        self.assertEqual(len(summaries[0]["mean_block_second_moment_of_coordinates"]), 2)
        self.assertEqual(len(summaries[0]["block_sem_of_squared_coordinates"]), 2)
        self.assertAlmostEqual(summaries[0]["mean_block_length_native_steps"], 16.2)
        self.assertIsNone(summaries[-1]["block_sem"])

    def test_joint_series_memory_cap_precedes_allocation(self):
        with patch.object(diagnostic, "MAX_JOINT_SERIES_BYTES", 1), self.assertRaisesRegex(ValueError, "memory cap"):
            diagnostic.diagnose(self.spool, pairs=[(0, 1)], cutoff=5.0)

    def test_autocorrelation_statuses_and_ar_estimate(self):
        constant = diagnostic._autocorrelation_summary(np.ones(64), step_interval=0.5)
        short = diagnostic._autocorrelation_summary(np.arange(12.0), step_interval=2.0)
        self.assertEqual(constant["status"], "INCONCLUSIVE_CONSTANT")
        self.assertEqual(short["status"], "INCONCLUSIVE_TOO_SHORT")

        rng = np.random.default_rng(731)
        noise = rng.normal(size=10000)
        ar = np.empty_like(noise)
        ar[0] = noise[0]
        for index in range(1, len(ar)):
            ar[index] = 0.8 * ar[index - 1] + noise[index]
        summary = diagnostic._autocorrelation_summary(ar[1000:], step_interval=0.25)
        self.assertEqual(summary["status"], "DIAGNOSTIC_ESTIMATE")
        self.assertGreater(summary["integrated_autocorrelation_time_frames"], 3.0)
        self.assertLess(summary["integrated_autocorrelation_time_frames"], 6.0)
        self.assertAlmostEqual(summary["integrated_autocorrelation_time_native_steps"],
                               summary["integrated_autocorrelation_time_frames"] * 0.25)
        squared = diagnostic._autocorrelation_summary(ar[1000:] ** 2, step_interval=0.25)
        self.assertGreater(squared["estimated_effective_sample_count"], 0)

    def test_autocorrelation_lag_cap_is_inconclusive(self):
        rng = np.random.default_rng(123)
        noise = rng.normal(size=6000)
        ar = np.empty_like(noise)
        ar[0] = noise[0]
        for index in range(1, len(ar)):
            ar[index] = 0.995 * ar[index - 1] + noise[index]
        summary = diagnostic._autocorrelation_summary(ar[1000:], step_interval=1.0)
        self.assertEqual(summary["max_lag_frames"], 256)
        self.assertEqual(summary["status"], "INCONCLUSIVE_LAG_CAP_REACHED")


if __name__ == "__main__":
    unittest.main()
