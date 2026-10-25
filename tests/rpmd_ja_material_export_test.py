import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rpmd_ja_material_export as export


class MaterialExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.directory = Path(self.temp.name)
        self.dump = self.directory / "trajectory.xyz"
        self.current = self.directory / "heat_current_rpmd_ja.out"
        self.reference = self.directory / "reference.rpja"
        self.output = self.directory / "trajectory.npz"
        self.reference.write_bytes(b"reference-package-v1\n")
        rows = []
        values = []
        for frame in range(4):
            time_fs = (frame + 1) * export.NATIVE_TIME_FS
            rows.extend((
                "2\n",
                f'Time={time_fs:.8f} pbc="T T T" Lattice="20 0 0 2 18 0 0 1 16" Properties=species:S:1:pos:R:3:vel:R:3:unwrapped_position:R:3\n',
                f"H {5 + frame * 0.01} 5 5 0.001 0.002 0.003 {5 + frame * 0.01} 5 5\n",
                f"O 7 5 5 -0.002 0.001 0.0 7 5 5\n",
            ))
            values.append((
                (frame + 1) * export.NATIVE_TIME_PS,
                1 + frame, 2 + frame, 3 + frame,
                0.25, -0.5, 0.75,
                1.25 + frame, 1.5 + frame, 3.75 + frame,
            ))
        self.dump.write_text("".join(rows), encoding="ascii")
        with self.current.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("# segment_metadata_version 2\n")
            stream.write(f"# reference_package_sha256 {export._sha256(self.reference)}\n")
            stream.write("# backend qnep-block-v3\n")
            stream.write("# mechanical_policy native_reference_transport\n")
            stream.write("# sample_frames 4\n")
            stream.write("# current_units eV*Angstrom/natural_time\n")
            stream.write("# columns time_ps Jcent_x Jcent_y Jcent_z DeltaJ_x DeltaJ_y DeltaJ_z JA_x JA_y JA_z\n")
            for row in values:
                stream.write(" ".join(f"{item:.15g}" for item in row) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def _spec(self, current=None, dump=None, reference=None, legacy=False):
        spec = {
            "format": "RPJAMAT_EXPORT1" if legacy else "RPJAMAT_EXPORT2",
            "trajectory_id": "trajectory-17",
            "xc_branch_id": "centroid-unwrap-v1",
            "ordered_atom_ids": ["H-0", "O-1"],
            "ordered_species": ["H", "O"],
            "frame_ids": [0, 1, 2, 3],
            "step_ids": [1, 2, 3, 4],
            "time_ps": [export.NATIVE_TIME_PS * i for i in (1, 2, 3, 4)],
            "position_stream_id": "trajectory-17-centroid-position-stream",
            "memory_limit_bytes": 1 << 20,
            "dump_xyz": str(dump or self.dump),
            "current_file": str(current or self.current),
            "output_npz": str(self.output),
        }
        if legacy:
            spec["allow_legacy_current_unverified"] = True
        else:
            reference = Path(reference or self.reference)
            spec["online_reference_id"] = "reference-0"
            spec["online_reference_package"] = str(reference)
            spec["online_reference_package_sha256"] = export._sha256(reference)
        path = self.directory / "export.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path

    def test_exports_synchronized_wrapped_positions_velocities_and_native_jcent(self):
        summary = export.export_trajectory(self._spec())
        with np.load(self.output) as arrays:
            self.assertEqual(arrays["wrapped_position_angstrom"].shape, (4, 2, 3))
            self.assertEqual(arrays["atom_ids"].tolist(), ["H-0", "O-1"])
            self.assertEqual(arrays["frame_ids"].tolist(), [0, 1, 2, 3])
            self.assertEqual(arrays["step_ids"].tolist(), [1, 2, 3, 4])
            self.assertAlmostEqual(float(arrays["vc_angstrom_per_ps"][0, 0, 0]), 1.0)
            self.assertAlmostEqual(float(arrays["jc_eV_angstrom_per_ps"][0, 0]), 1.0 / export.NATIVE_TIME_PS)
            np.testing.assert_allclose(arrays["online_delta_j_eV_angstrom_per_ps"][0],
                                       np.array([0.25, -0.5, 0.75]) / export.NATIVE_TIME_PS)
            np.testing.assert_allclose(arrays["online_ja_eV_angstrom_per_ps"][0],
                                       np.array([1.25, 1.5, 3.75]) / export.NATIVE_TIME_PS)
            self.assertEqual(str(arrays["online_reference_id"].item()), "reference-0")
            self.assertEqual(str(arrays["online_reference_package_sha256"].item()), export._sha256(self.reference))
            self.assertAlmostEqual(float(arrays["time_ps"][0]), export.NATIVE_TIME_PS)
            np.testing.assert_allclose(arrays["cell_angstrom"], np.array([[20.0, 2.0, 0.0],
                [0.0, 18.0, 1.0], [0.0, 0.0, 16.0]]))
            self.assertEqual(str(arrays["position_source"].item()), "dump_xyz.atom.position_per_atom")
        self.assertIn("no unique atom IDs", summary["identity_limit"])
        self.assertEqual(summary["format"], "RPJAMAT_EXPORT2_REPORT")
        self.assertEqual(summary["online_reference_id"], "reference-0")
        self.assertEqual(summary["online_reference_package_sha256"], export._sha256(self.reference))
        self.assertEqual(len(summary["trajectory_npz_sha256"]), 64)

    def test_rejects_missing_or_malformed_online_reference_identity(self):
        spec = json.loads(self._spec().read_text(encoding="utf-8"))
        del spec["online_reference_id"]
        path = self.directory / "missing-reference.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "online_reference_id"):
            export.export_trajectory(path)

        spec = json.loads(self._spec().read_text(encoding="utf-8"))
        spec["online_reference_package_sha256"] = "not-a-digest"
        path = self.directory / "bad-reference.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            export.export_trajectory(path)

        spec = json.loads(self._spec().read_text(encoding="utf-8"))
        spec["online_reference_package_sha256"] = "b" * 64
        path = self.directory / "wrong-reference-digest.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "declared reference package SHA-256"):
            export.export_trajectory(path)

    def test_rejects_time_mismatch_or_multiple_current_segments(self):
        wrong_time = self.directory / "wrong-time.xyz"
        wrong_time.write_text(self.dump.read_text(encoding="ascii").replace("Time=20.36102000", "Time=20.46102000"),
                              encoding="ascii")
        with self.assertRaisesRegex(ValueError, "time|synchron"):
            export.export_trajectory(self._spec(dump=wrong_time))
        with self.current.open("a", encoding="ascii") as stream:
            stream.write("# segment_metadata_version 2\n")
        with self.assertRaisesRegex(ValueError, "segment"):
            export.export_trajectory(self._spec())

    def test_rejects_wrong_reference_identity_even_when_delta_j_is_zero(self):
        other_reference = self.directory / "other-reference.rpja"
        other_reference.write_bytes(b"another-reference-package\n")
        lines = self.current.read_text(encoding="ascii").splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("#"):
                values = [float(value) for value in line.split()]
                values[4:7] = [0.0, 0.0, 0.0]
                values[7:10] = values[1:4]
                lines[index] = " ".join(f"{value:.15g}" for value in values)
        self.current.write_text("\n".join(lines) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "reference.*(SHA|digest)|digest.*reference"):
            export.export_trajectory(self._spec(reference=other_reference))

    def test_diagnostic_identity_is_copied_from_current_segment_to_export(self):
        lines = self.current.read_text(encoding="ascii").splitlines()
        policy_index = next(index for index, line in enumerate(lines) if line.startswith("# mechanical_policy"))
        lines[policy_index] = "# mechanical_policy native_reference_transport;finite_temperature_additive_v1;beads=1;derivative=1;use=DIAGNOSTIC;response=FAIL;numerical=PASS"
        lines.extend(("# reference_use_class DIAGNOSTIC", "# reference_numerical_status PASS",
                      "# reference_response_status FAIL"))
        self.current.write_text("\n".join(lines) + "\n", encoding="ascii")
        summary = export.export_trajectory(self._spec())
        self.assertEqual(summary["online_reference_use_class"], "DIAGNOSTIC")
        self.assertEqual(summary["online_reference_numerical_status"], "PASS")
        self.assertEqual(summary["online_reference_response_status"], "FAIL")
        with np.load(self.output, allow_pickle=False) as arrays:
            self.assertEqual(str(arrays["online_reference_use_class"].item()), "DIAGNOSTIC")
            self.assertEqual(str(arrays["online_reference_response_status"].item()), "FAIL")

    def test_diagnostic_policy_cannot_fall_back_or_conflict(self):
        original = self.current.read_text(encoding="ascii").splitlines()
        diagnostic = "# mechanical_policy native_reference_transport;finite_temperature_additive_v1;beads=1;derivative=1;use=DIAGNOSTIC;response=FAIL;numerical=PASS"
        self.current.write_text("\n".join([line for line in original if not line.startswith("# mechanical_policy")] +
                                          [diagnostic]) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "identity is incomplete"):
            export._read_current(self.current, None)

        conflicted = original.copy()
        policy_line = conflicted.index("# mechanical_policy native_reference_transport")
        conflicted[policy_line] = diagnostic
        conflicted.extend(("# reference_use_class DIAGNOSTIC", "# reference_numerical_status PASS",
                           "# reference_response_status PASS"))
        self.current.write_text("\n".join(conflicted) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "conflicts with mechanical_policy"):
            export._read_current(self.current, None)

    def test_unknown_policy_rejected_and_legacy_ordinary_still_reads(self):
        lines = self.current.read_text(encoding="ascii").splitlines()
        policy_line = lines.index("# mechanical_policy native_reference_transport")
        lines[policy_line] = "# mechanical_policy future_unrecognized_policy"
        self.current.write_text("\n".join(lines) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "unknown or malformed"):
            export._read_current(self.current, None)
        old_lines = [line for line in lines if not line.startswith("# mechanical_policy")]
        self.current.write_text("\n".join(old_lines) + "\n", encoding="ascii")
        _jc, _delta, _ja, _sha, identity = export._read_current(self.current, None)
        self.assertEqual(identity["reference_use_class"], "FORMAL")

    def test_legacy_current_requires_explicit_unverified_export_path(self):
        legacy = self.directory / "legacy-current.out"
        text = self.current.read_text(encoding="ascii").replace("# segment_metadata_version 2\n", "# segment_metadata_version 1\n")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("# reference_package_sha256")) + "\n"
        legacy.write_text(text, encoding="ascii")
        spec = json.loads(self._spec().read_text(encoding="utf-8"))
        spec["current_file"] = str(legacy)
        path = self.directory / "legacy-not-opted-in.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "legacy|RPJAMAT_EXPORT1"):
            export.export_trajectory(path)
        legacy_summary = export.export_trajectory(self._spec(current=legacy, legacy=True))
        self.assertEqual(legacy_summary["format"], "RPJAMAT_EXPORT1_REPORT")
        with np.load(self.output) as arrays:
            self.assertNotIn("online_reference_package_sha256", arrays.files)

    def test_rejects_missing_cell_or_wrong_atom_order(self):
        no_cell = self.directory / "no-cell.xyz"
        no_cell.write_text(self.dump.read_text(encoding="ascii").replace(
            ' Lattice="20 0 0 2 18 0 0 1 16"', ""), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "Lattice|cell"):
            export.export_trajectory(self._spec(dump=no_cell))

        wrong_order = self.directory / "wrong-order.xyz"
        text = self.dump.read_text(encoding="ascii")
        text = text.replace("H 5", "O 5").replace("O 7", "H 7")
        wrong_order.write_text(text, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "species|order"):
            export.export_trajectory(self._spec(dump=wrong_order))


if __name__ == "__main__":
    unittest.main(verbosity=2)
