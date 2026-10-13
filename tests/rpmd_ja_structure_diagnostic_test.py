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
import rpmd_ja_structure_diagnostic as diagnostic


def _write_spool(path, frames, types=None, cell_length=10.0):
    n = len(frames[0])
    if types is None:
        types = [1, 0, 0, 0, 1, 2]
    with path.open("wb") as stream:
        stream.write(b"GPJASMP1")
        stream.write(struct.pack("<IIiid", 1, diagnostic.ENDIAN, n, 32, 300.0))
        (np.eye(3, dtype="<f8") * cell_length).tofile(stream)
        np.asarray([1.0 if atom_type == 1 else 16.0 for atom_type in types], dtype="<f8").tofile(stream)
        np.asarray(types, dtype="<i4").tofile(stream)
        for step, xyz in enumerate(frames):
            stream.write(struct.pack("<d", float(step)))
            np.asarray(xyz, dtype="<f8").T.ravel().tofile(stream)
            np.zeros((3, n), dtype="<f8").tofile(stream)


class StructureDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spool = self.root / "samples.bin"
        def frame(hx, lattice_shift, fluctuation, remote_h):
            return [[hx, 5, 5], [9.4 + lattice_shift + fluctuation, 5, 5],
                    [0.6 + lattice_shift + fluctuation, 5, 5],
                    [1.8 + lattice_shift + fluctuation, 5, 5], [remote_h, 5, 5], [0.2, 6, 5]]
        xyz = [frame(0.0, 0.0, -0.05, 5.0), frame(0.1, 0.0, 0.05, 5.0),
               frame(0.0, 0.0, -0.05, 5.0), frame(0.1, 0.0, 0.05, 5.0),
               frame(1.7, 0.2, -0.05, 0.0), frame(1.7, 0.2, 0.05, 0.0)]
        _write_spool(self.spool, xyz)

    def test_fixed_candidates_periodicity_coverage_histograms_and_branch(self):
        original = self.spool.read_bytes()
        result = diagnostic.diagnose(self.spool, hydrogen_type=1, oxygen_type=0,
                                     candidate_cutoff=3.0, block_frames=4, ion_types=[2])
        self.assertEqual(self.spool.read_bytes(), original)
        self.assertEqual(result["sample_step_interval"], 1.0)
        self.assertIn("native spool step units", result["sample_step_interval_units"])
        self.assertTrue(result["fixed_R0_definition"].startswith("unwrapped centroid arithmetic mean"))
        self.assertTrue(result["candidate_limited"])
        coverage = result["R0_candidate_coverage"]
        self.assertEqual(coverage["hydrogen_count"], 2)
        self.assertEqual(coverage["with_two_or_more_R0_candidates"], 1)
        self.assertEqual(coverage["fixed_pair_classified_H_count"], 1)
        self.assertEqual(coverage["fixed_pair_classification_fraction"], 0.5)
        self.assertEqual(coverage["fewer_than_two_R0_candidate_H_count"], 1)
        self.assertEqual(coverage["unclassified_H_atom_ids"], [4])
        self.assertEqual(coverage["truncated_H_atom_ids"], [])
        self.assertIn("NOT_EVALUATED", coverage["outside_frozen_candidates"])
        hydrogen = result["hydrogens"][0]
        self.assertEqual(hydrogen["reference_pair_O_atoms"], [2, 1])
        self.assertAlmostEqual(hydrogen["R0_pair_R_OO_A"], 1.2, places=12)
        self.assertAlmostEqual(hydrogen["R0_ion_nearest_distance_A"]["2"], np.sqrt(1.04), places=12)
        self.assertAlmostEqual(hydrogen["training_candidate_local_pair_change_fraction"], 0.0)
        self.assertAlmostEqual(hydrogen["validation_candidate_local_pair_change_fraction"], 1.0)
        self.assertEqual(hydrogen["training_R_OO_A"]["count"], 4)
        self.assertEqual(hydrogen["validation_R_OO_A"]["count"], 2)
        self.assertEqual(hydrogen["training_candidate_local_second_to_third_HO_gap_A"]["count"], 4)
        self.assertEqual(hydrogen["validation_candidate_local_second_to_third_HO_gap_A"]["count"], 2)
        train_hist = result["fixed_pair_metrics"]["training"]["R_OO_eccentricity_histogram"]
        validation_hist = result["fixed_pair_metrics"]["validation"]["R_OO_eccentricity_histogram"]
        self.assertEqual(train_hist["bin_edges_R_OO_A"], validation_hist["bin_edges_R_OO_A"])
        self.assertEqual(train_hist["bin_edges_bridge_eccentricity"], validation_hist["bin_edges_bridge_eccentricity"])
        self.assertIn("training fixed-pair samples only", validation_hist["bin_source"])
        self.assertGreater(validation_hist["groups"]["all"]["out_of_training_range_count"], 0)
        changed = validation_hist["groups"]["local_pair_change"]
        self.assertEqual(sum(map(sum, changed["counts"])) + changed["out_of_training_range_count"], 2)
        blocks = result["block_means"]
        self.assertEqual([block["phase"] for block in blocks], ["training", "validation"])
        self.assertNotEqual(blocks[0]["bridge_eccentricity_mean"], blocks[1]["bridge_eccentricity_mean"])
        drift = result["fixed_R0_mean_drift"]
        self.assertAlmostEqual(drift["training"]["type_displacements"]["oxygen"]["vector_rms_displacement_A"], 0.0, places=12)
        self.assertAlmostEqual(drift["validation"]["type_displacements"]["oxygen"]["vector_rms_displacement_A"], 0.2, places=12)
        self.assertAlmostEqual(drift["validation"]["type_displacements"]["oxygen"]["max_vector_displacement_A"], 0.2, places=12)
        self.assertEqual(drift["internal_norm_units"], "sqrt(mass_unit)*angstrom")
        self.assertGreater(drift["validation"]["mass_weighted_internal_displacement_norm_sqrt_mass_A"], 0.0)
        branch = result["fixed_R0_branch_contract"]
        self.assertEqual(branch["threshold_fractional_component"], 0.45)
        self.assertEqual(branch["validation"]["violating_frames"], 2)
        self.assertEqual(branch["validation"]["violating_atoms"], 2)
        self.assertEqual(branch["validation"]["violating_components"], 2)
        self.assertIn("no recentering", branch["definition"])
        self.assertIn("global exchange fraction", result["limitations"][0])

    def test_nonuniform_sample_steps_are_rejected(self):
        data = bytearray(self.spool.read_bytes())
        frame_bytes = 8 + 2 * 3 * 6 * 8
        spool_header_bytes = 8 + 4 * 4 + 8 + 9 * 8 + 6 * 8 + 6 * 4
        struct.pack_into("<d", data, spool_header_bytes + 3 * frame_bytes, 3.2)
        nonuniform = self.root / "nonuniform.bin"
        nonuniform.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "uniform interval"):
            diagnostic.diagnose(nonuniform, hydrogen_type=1, oxygen_type=0,
                                candidate_cutoff=3.0, block_frames=4, ion_types=[2])

    def test_output_is_exclusive_and_never_overwrites_spool_or_report(self):
        args = ["diagnostic", str(self.spool), "--hydrogen-type", "1", "--oxygen-type", "0",
                "--candidate-cutoff", "3", "--output", str(self.spool)]
        original = self.spool.read_bytes()
        with patch.object(sys, "argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            diagnostic.main()
        self.assertEqual(self.spool.read_bytes(), original)
        report = self.root / "existing.json"
        report.write_text("keep me\n", encoding="utf-8")
        with patch.object(sys, "argv", args[:-1] + [str(report)]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            diagnostic.main()
        self.assertEqual(report.read_text(encoding="utf-8"), "keep me\n")
        self.assertEqual(self.spool.read_bytes(), original)

    def test_hydrogen_with_over_cap_candidates_is_excluded_from_fixed_pairs(self):
        spool = self.root / "many_oxygen.bin"
        first_h = np.asarray([50.0, 50.0, 50.0])
        second_h = np.asarray([80.0, 80.0, 80.0])
        oxygen = [first_h + np.asarray([0.01 * (i % 4), 0.01 * ((i // 4) % 4), 0.01 * (i // 16)])
                  for i in range(65)]
        oxygen += [second_h + np.asarray([0.2, 0, 0]), second_h + np.asarray([-0.2, 0, 0]),
                   second_h + np.asarray([0, 0.2, 0])]
        frame = [first_h, second_h, *oxygen]
        _write_spool(spool, [frame, frame, frame], types=[1, 1] + [0] * len(oxygen), cell_length=100.0)
        result = diagnostic.diagnose(spool, hydrogen_type=1, oxygen_type=0,
                                     candidate_cutoff=3.0, block_frames=2)
        coverage = result["R0_candidate_coverage"]
        self.assertEqual(coverage["with_two_or_more_R0_candidates"], 2)
        self.assertEqual(coverage["candidate_lists_truncated_H_count"], 1)
        self.assertEqual(coverage["unclassified_H_atom_ids"], [0])
        self.assertEqual(coverage["truncated_H_atom_ids"], [0])
        self.assertEqual(coverage["fixed_pair_classified_H_count"], 1)
        self.assertEqual(coverage["fixed_pair_unclassified_H_count"], 1)
        self.assertEqual(result["hydrogens"][0]["atom"], 1)


if __name__ == "__main__":
    unittest.main()
