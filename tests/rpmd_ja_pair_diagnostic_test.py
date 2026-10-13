from contextlib import redirect_stderr
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rpmd_ja_pair_diagnostic as diagnostic


def _write_spool(path, frames, types, cell_length=10.0):
    n = len(frames[0])
    with path.open("wb") as stream:
        stream.write(b"GPJASMP1")
        stream.write(struct.pack("<IIiid", 1, 0x01020304, n, 32, 300.0))
        (np.eye(3, dtype="<f8") * cell_length).tofile(stream)
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

    def test_output_refuses_overwrite(self):
        report_path = self.root / "report.json"
        report_path.write_text("keep me\n", encoding="utf-8")
        args = ["pair-diagnostic", str(self.spool), "--pair", "0:1", "--cutoff", "5",
                "--output", str(report_path)]
        with patch.object(sys, "argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            diagnostic.main()
        self.assertEqual(report_path.read_text(encoding="utf-8"), "keep me\n")


if __name__ == "__main__":
    unittest.main()
