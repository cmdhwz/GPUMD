import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import rpmd_ja_additive_workflow_test as fixture
import rpmd_ja_qnep_prepare as prepare
import rpmd_ja_material_evaluate as evaluate
import rpmd_ja_material_export as export


class MaterialEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        cls.directory = Path(cls.temp.name)
        cls.masses = np.array([1.0, 2.0])
        cls.cell = np.diag([20.0, 20.0, 20.0])
        cls.base = np.array([[19.995, 5.0, 5.0], [7.0, 5.0, 5.0]])
        cls.lambdas = np.array([2.0, 3.0, 4.0])
        cls.k, _ = fixture._physical_model(cls.masses, cls.lambdas)
        cls.packages = []
        cls.reference_positions = []
        for index, scale in enumerate((1.0, 1.7)):
            ht = np.empty((3, 6, 6), dtype=np.float64)
            for axis in range(3):
                rows, columns = np.indices((6, 6))
                ht[axis] = (0.08 * (axis + 1) + 0.013 * rows - 0.007 * columns +
                            0.002 * rows * columns) * scale
            raw = cls.directory / f"raw-{index}.qraw"
            reference_positions = cls.base + (0.004 * index) * np.array([[1.0, -0.5, 0.25], [-0.25, 0.5, -1.0]])
            cls.reference_positions.append(reference_positions)
            fixture._write_raw(raw, reference_positions, cls.cell, cls.masses, scale * cls.k, ht)
            package = cls.directory / f"ref-{index}.ja"
            fixture._run([ROOT / "tools/rpmd_ja_qnep_prepare.py", raw,
                           "--kernel-table", ROOT / "tests/data/rpmd_ja_kernel_U8.txt",
                           "--output", package, "--lossless"])
            cls.packages.append(package)

        frames = 12
        time_ps = np.arange(frames, dtype=np.float64) * 0.2
        t = np.arange(frames, dtype=np.float64)
        xc = cls.base[None, :, :] + np.stack((
            0.01 * np.sin(0.4 * t), 0.015 * np.cos(0.3 * t), 0.006 * np.sin(0.8 * t),
            0.008 * np.cos(0.6 * t), 0.012 * np.sin(0.2 * t + 0.4), 0.009 * np.cos(0.7 * t)), axis=1).reshape(frames, 2, 3)
        xc[:, 0, 0] = np.mod(19.997 + 0.002 * np.arange(frames), cls.cell[0, 0])
        vc = np.stack((
            0.2 * np.cos(0.4 * t), 0.1 * np.sin(0.3 * t), 0.13 * np.cos(0.8 * t),
            0.07 * np.sin(0.6 * t), 0.11 * np.cos(0.2 * t + 0.4), 0.09 * np.sin(0.7 * t)), axis=1).reshape(frames, 2, 3)
        jc = np.stack((np.sin(0.5 * t), np.cos(0.3 * t), np.sin(0.8 * t + 0.2)), axis=1)
        cls.trajectory = cls.directory / "trajectory.npz"
        np.savez(cls.trajectory,
                 frame_ids=np.arange(frames, dtype=np.int64),
                 step_ids=np.arange(frames, dtype=np.int64) * 10,
                 time_ps=time_ps,
                 atom_ids=np.array(["H-site-0", "O-site-1"]),
                 trajectory_id=np.array("trajectory-run-17"),
                 position_stream_id=np.array("position-stream-run-17"),
                 position_source=np.array("dump_xyz.atom.position_per_atom"),
                 dump_source_sha256=np.array("a" * 64),
                 current_source_sha256=np.array("b" * 64),
                 cell_angstrom=cls.cell,
                 wrapped_position_angstrom=xc,
                 vc_angstrom_per_ps=vc,
                 jc_eV_angstrom_per_ps=jc)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _manifest(self, refs=None, memory_limit=1 << 30):
        refs = refs or [
            {"id": f"reference-{i}", "package": str(path),
             "package_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "ordered_atom_ids": ["H-site-0", "O-site-1"],
             "reference_position_stream_id": "position-stream-run-17"}
            for i, path in enumerate(self.packages)
        ]
        manifest = {
            "format": "RPJAMAT1",
            "trajectory_id": "trajectory-run-17",
            "trajectory_npz": str(self.trajectory),
            "trajectory_npz_sha256": hashlib.sha256(self.trajectory.read_bytes()).hexdigest(),
            "dump_source_sha256": "a" * 64,
            "current_source_sha256": "b" * 64,
            "frame_count": 12,
            "frame_ids": list(range(12)),
            "step_ids": [10 * index for index in range(12)],
            "time_start_ps": 0.0,
            "time_interval_ps": 0.2,
            "ordered_atom_ids": ["H-site-0", "O-site-1"],
            "masses_amu": self.masses.tolist(),
            "types": [0, 1],
            "cell_angstrom": self.cell.tolist(),
            "temperature_K": 300.0,
            "position_stream_id": "position-stream-run-17",
            "position_semantics": "wrapped_position_per_atom_A",
            "branch_policy": "native_rpmd_ja_mic_v1",
            "units": {
                "position": "angstrom",
                "velocity": "angstrom/ps",
                "centroid_current": "eV*angstrom/ps",
                "time": "ps",
                "reference_wire_time": "native",
                "native_time_ps": 0.01018051
            },
            "memory_limit_bytes": memory_limit,
            "max_lag_frames": 5,
            "references": refs,
            "tolerances": {
                "delta_j_near_zero_rms_native": 1.0e-10,
                "reference_sensitivity_relative": 1.0,
                "reference_sensitivity_absolute_native": 1.0e-8
            }
        }
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _v2_manifest(self, online_reference_id="reference-0", delta_offset=0.0):
        manifest = json.loads(self._manifest().read_text(encoding="utf-8"))
        package_sha256 = manifest["references"][0]["package_sha256"]
        online_delta_native = self._dense_oracle(0)[0] + delta_offset
        with np.load(self.trajectory, allow_pickle=False) as source:
            payload = {key: source[key] for key in source.files}
        online_jc_native = payload["jc_eV_angstrom_per_ps"] * evaluate.NATIVE_TIME_PS
        payload["online_delta_j_eV_angstrom_per_ps"] = online_delta_native / evaluate.NATIVE_TIME_PS
        payload["online_ja_eV_angstrom_per_ps"] = (online_jc_native + online_delta_native) / evaluate.NATIVE_TIME_PS
        payload["online_reference_id"] = np.asarray(online_reference_id)
        payload["online_reference_package_sha256"] = np.asarray(package_sha256)
        payload["online_current_units"] = np.asarray("eV*angstrom/ps")
        trajectory = self.directory / "trajectory-rpja2.npz"
        np.savez(trajectory, **payload)
        manifest["format"] = "RPJAMAT2"
        manifest["trajectory_npz"] = str(trajectory)
        manifest["trajectory_npz_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        manifest["online_current"] = {
            "reference_id": online_reference_id,
            "reference_package_sha256": package_sha256,
            "units": "eV*angstrom/ps"
        }
        path = self.directory / "manifest-rpja2.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _dense_oracle(self, index):
        pack = fixture._read_v3(self.packages[index])
        dmat, *bt = pack["matrices"]
        w, vectors = np.linalg.eigh(dmat)
        keep = w > 1.0e-9
        eigenvalues, eigenvectors = w[keep], vectors[:, keep]
        table = prepare._kernel(ROOT / "tests/data/rpmd_ja_kernel_U8.txt")
        tau = prepare.HBAR / (prepare.KB * pack["temp"])
        lambda_scale = table["u"] ** 2 / tau ** 2
        z = 2.0 * eigenvalues / lambda_scale - 1.0
        p_modes = np.polynomial.chebyshev.chebvander(z, pack["degree"]) @ table["pvec"].reshape(pack["degree"] + 1, -1)
        q_modes = np.polynomial.chebyshev.chebvander(z, pack["degree"]) @ table["qvec"].reshape(pack["degree"] + 1, -1)
        with np.load(self.trajectory) as trajectory:
            wrapped = trajectory["wrapped_position_angstrom"]
            v_soa = (trajectory["vc_angstrom_per_ps"] * (10.18051 / 1000.0)).transpose(0, 2, 1).reshape(12, 6)
        x_soa = (self._replay_reference_branch(wrapped, self.reference_positions[index]) -
                 self.reference_positions[index][None, :, :]).transpose(0, 2, 1).reshape(12, 6)
        sqrt_mass = np.tile(np.sqrt(self.masses), 3)
        oracle = np.empty((12, 3))
        for axis, matrix in enumerate(bt):
            a = eigenvectors.T @ matrix @ eigenvectors
            p = (p_modes * table["pv"][None, :]) @ p_modes.T
            q = (q_modes * table["qv"][None, :]) @ q_modes.T
            ta = tau ** 2 * eigenvalues[:, None]
            tb = tau ** 2 * eigenvalues[None, :]
            kernel = ta * tb * p * a + tb * q * a.T
            delta_h = sqrt_mass[:, None] * (eigenvectors @ kernel.T @ eigenvectors.T) * sqrt_mass[None, :]
            oracle[:, axis] = np.einsum("fi,ij,fj->f", x_soa, delta_h, v_soa)
        return oracle, (eigenvalues, eigenvectors, bt, p_modes, q_modes, table, tau, sqrt_mass, x_soa, v_soa)

    def _replay_reference_branch(self, wrapped, reference):
        # Independent test oracle for rpmd_ja initialize/update MIC semantics.
        continuous = np.empty_like(wrapped)
        delta = wrapped[0] - reference
        fractional = delta @ np.linalg.inv(self.cell).T
        fractional -= np.rint(fractional)
        continuous[0] = reference + fractional @ self.cell.T
        previous = wrapped[0]
        for frame in range(1, len(wrapped)):
            step = wrapped[frame] - previous
            step_fractional = step @ np.linalg.inv(self.cell).T
            step_fractional -= np.rint(step_fractional)
            continuous[frame] = continuous[frame - 1] + step_fractional @ self.cell.T
            previous = wrapped[frame]
        return continuous

    def test_v3_evaluation_matches_independent_dense_delta_h_and_exports_all_hac_terms(self):
        report, arrays = evaluate.evaluate_manifest(self._manifest())
        reference = report["references"][0]
        oracle, _ = self._dense_oracle(0)
        np.testing.assert_allclose(arrays[reference["delta_j_array"]], oracle, rtol=2.0e-11, atol=2.0e-12)
        oracle_second, _ = self._dense_oracle(1)
        np.testing.assert_allclose(arrays[report["references"][1]["delta_j_array"]], oracle_second,
                                   rtol=2.0e-11, atol=2.0e-12)
        eigenvalues, eigenvectors, bt, p_modes, q_modes, table, tau, sqrt_mass, x_soa, v_soa = self._dense_oracle(0)[1]
        self.assertGreater(np.linalg.norm(bt[0] - bt[0].T), 1.0e-4)
        wrong_a = eigenvectors.T @ bt[0].T @ eigenvectors
        correct_a = eigenvectors.T @ bt[0] @ eigenvectors
        self.assertGreater(np.linalg.norm(wrong_a - correct_a), 1.0e-4)
        p = (p_modes * table["pv"][None, :]) @ p_modes.T
        q = (q_modes * table["qv"][None, :]) @ q_modes.T
        ta = tau ** 2 * eigenvalues[:, None]
        tb = tau ** 2 * eigenvalues[None, :]
        wrong_kernel = ta * tb * p * wrong_a + tb * q * wrong_a.T
        wrong_h = sqrt_mass[:, None] * (eigenvectors @ wrong_kernel.T @ eigenvectors.T) * sqrt_mass[None, :]
        wrong_current = np.einsum("fi,ij,fj->f", x_soa, wrong_h, v_soa)
        self.assertGreater(np.linalg.norm(oracle[:, 0] - wrong_current), 1.0e-8)
        self.assertEqual(arrays["reference_0_jc_dj"].shape, (6, 3, 3))
        self.assertEqual(arrays["reference_0_dj_jc"].shape, (6, 3, 3))
        self.assertEqual(arrays["reference_0_ja_ja"].shape, (6, 3, 3))
        self.assertEqual(len(reference["delta_j_variance_native_squared"]), 3)
        np.testing.assert_allclose(arrays["reference_0_ja_ja"], arrays["jc_jc"] + arrays["reference_0_jc_dj"] +
                                   arrays["reference_0_dj_jc"] + arrays["reference_0_dj_dj"])
        self.assertEqual(report["hac_integral_units"], "eV^2*angstrom^2/ps")
        self.assertEqual(len(reference["ja_correlation_decomposition_error_by_lag_all_nine_components"]), 6)
        self.assertLess(reference["ja_correlation_decomposition_max_absolute_error"], 1.0e-12)
        self.assertIn("amplitude only", report["reference_sensitivity"]["scope"])
        np.testing.assert_allclose(arrays["reference_0_ja_ja"], arrays["reference_0_ja_ja_decomposed"],
                                   rtol=1.0e-12, atol=1.0e-12)
        self.assertNotIn("thermal_conductivity", report)

    def test_v2_compares_online_stream_before_correlations_and_binds_reference_identity(self):
        report, arrays = evaluate.evaluate_manifest(self._v2_manifest())
        self.assertEqual(report["format"], "RPJAMAT2_REPORT")
        self.assertEqual(report["online_current_comparison"]["status"], "PASS")
        self.assertLess(report["online_current_comparison"]["delta_j_max_abs_error"], 1.0e-10)
        self.assertLess(report["online_current_comparison"]["ja_max_abs_error"], 1.0e-10)
        self.assertEqual(report["online_current_comparison"]["reference_id"], "reference-0")
        self.assertEqual(report["online_current_comparison"]["reference_use_class"], "FORMAL")
        self.assertEqual(report["analysis_use_class"], "FORMAL")
        self.assertEqual(arrays["online_delta_j_eV_angstrom_per_ps"].shape, (12, 3))

        with self.assertRaisesRegex(ValueError, "online.*reference|reference.*online"):
            evaluate.evaluate_manifest(self._v2_manifest(online_reference_id="unknown-reference"))

        with self.assertRaisesRegex(ValueError, "online.*DeltaJ|DeltaJ.*online"):
            evaluate.evaluate_manifest(self._v2_manifest(delta_offset=1.0e-4))

    def test_diagnostic_rpjam2_requires_identity_in_current_archive(self):
        path = self._v2_manifest()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["analysis_use_class"] = "DIAGNOSTIC"
        manifest["online_current"].update({
            "reference_use_class": "DIAGNOSTIC",
            "reference_numerical_status": "PASS",
            "reference_response_status": "FAIL",
        })
        diagnostic_path = self.directory / "manifest-rpja2-diagnostic-missing-identity.json"
        diagnostic_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "diagnostic current.*identity"):
            evaluate.evaluate_manifest(diagnostic_path)

    def test_rpjam2_current_identity_must_match_reference_package_policy(self):
        path = self._v2_manifest()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        with np.load(manifest["trajectory_npz"], allow_pickle=False) as source:
            payload = {key: source[key] for key in source.files}
        payload["online_reference_use_class"] = np.asarray("DIAGNOSTIC")
        payload["online_reference_numerical_status"] = np.asarray("PASS")
        payload["online_reference_response_status"] = np.asarray("FAIL")
        trajectory = self.directory / "trajectory-rpja2-mismatched-use.npz"
        np.savez(trajectory, **payload)
        manifest["trajectory_npz"] = str(trajectory)
        manifest["trajectory_npz_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        manifest["analysis_use_class"] = "DIAGNOSTIC"
        manifest["online_current"].update({
            "reference_use_class": "DIAGNOSTIC",
            "reference_numerical_status": "PASS",
            "reference_response_status": "FAIL",
        })
        mismatch_path = self.directory / "manifest-rpja2-package-use-mismatch.json"
        mismatch_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selected reference package"):
            evaluate.evaluate_manifest(mismatch_path)

    def test_native_current_export_then_rpjam2_evaluate_fixture(self):
        with np.load(self.trajectory, allow_pickle=False) as source:
            position = source["wrapped_position_angstrom"]
            velocity = source["vc_angstrom_per_ps"] / 1000.0
            centroid_current = source["jc_eV_angstrom_per_ps"] * evaluate.NATIVE_TIME_PS
            times = source["time_ps"]
        delta_native = self._dense_oracle(0)[0]
        dump = self.directory / "native-dump.xyz"
        with dump.open("w", encoding="ascii") as stream:
            for frame in range(len(times)):
                lattice = " ".join(f"{value:.17g}" for value in self.cell.T.reshape(-1))
                stream.write(f"2\nLattice=\"{lattice}\" pbc=\"T T T\" Time={times[frame] * 1000:.17g} "
                    "Properties=species:S:1:pos:R:3:vel:R:3\n")
                for atom, species in enumerate(("H", "O")):
                    row = np.concatenate((position[frame, atom], velocity[frame, atom]))
                    stream.write(species + " " + " ".join(f"{value:.17g}" for value in row) + "\n")
        current = self.directory / "heat_current_rpmd_ja.out"
        package_sha = hashlib.sha256(self.packages[0].read_bytes()).hexdigest()
        with current.open("w", encoding="ascii") as stream:
            stream.write("# segment_metadata_version 2\n")
            stream.write(f"# reference_package_sha256 {package_sha}\n")
            stream.write("# backend qnep-block-v3\n")
            stream.write(f"# sample_frames {len(times)}\n")
            stream.write("# current_units eV*Angstrom/natural_time\n")
            stream.write("# columns time_ps Jcent_x Jcent_y Jcent_z DeltaJ_x DeltaJ_y DeltaJ_z JA_x JA_y JA_z\n")
            for index, time in enumerate(times):
                ja_native = centroid_current[index] + delta_native[index]
                row = np.concatenate(([time], centroid_current[index], delta_native[index], ja_native))
                stream.write(" ".join(f"{value:.17g}" for value in row) + "\n")
        exported = self.directory / "native-export.npz"
        spec = {
            "format": "RPJAMAT_EXPORT2", "trajectory_id": "native-fixture-run",
            "position_stream_id": "native-position-stream", "ordered_atom_ids": ["H-site-0", "O-site-1"],
            "ordered_species": ["H", "O"], "frame_ids": list(range(len(times))),
            "step_ids": (10 * np.arange(len(times))).tolist(), "time_ps": times.tolist(),
            "memory_limit_bytes": 1 << 28, "dump_xyz": str(dump), "current_file": str(current),
            "output_npz": str(exported), "online_reference_id": "reference-0",
            "online_reference_package": str(self.packages[0]), "online_reference_package_sha256": package_sha,
        }
        spec_path = self.directory / "native-export.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        export.export_trajectory(spec_path)

        manifest = json.loads(self._manifest().read_text(encoding="utf-8"))
        manifest["format"] = "RPJAMAT2"
        manifest["trajectory_id"] = "native-fixture-run"
        manifest["position_stream_id"] = "native-position-stream"
        for reference in manifest["references"]:
            reference["reference_position_stream_id"] = "native-position-stream"
        manifest["trajectory_npz"] = str(exported)
        manifest["trajectory_npz_sha256"] = hashlib.sha256(exported.read_bytes()).hexdigest()
        manifest["dump_source_sha256"] = export._sha256(dump)
        manifest["current_source_sha256"] = export._sha256(current)
        manifest["online_current"] = {
            "reference_id": "reference-0", "reference_package_sha256": package_sha,
            "units": "eV*angstrom/ps",
        }
        manifest_path = self.directory / "native-rpjam2-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report, arrays = evaluate.evaluate_manifest(manifest_path)
        self.assertEqual(report["online_current_comparison"]["status"], "PASS")
        np.testing.assert_allclose(arrays["online_delta_j_eV_angstrom_per_ps"],
            delta_native / evaluate.NATIVE_TIME_PS, rtol=2.0e-11, atol=2.0e-12)

    def test_v1_compatibility_is_explicitly_unverified_for_online_streams(self):
        report, _ = evaluate.evaluate_manifest(self._manifest())
        self.assertEqual(report["online_current_comparison"]["status"], "NOT_AVAILABLE_LEGACY_RPJAMAT1")

    def test_rejects_atom_identity_mismatch_even_when_package_types_and_masses_match(self):
        with np.load(self.trajectory) as source:
            payload = {key: source[key] for key in source.files}
        payload["atom_ids"] = np.array(["O-site-1", "H-site-0"])
        np.savez(self.directory / "wrong-order.npz", **payload)
        manifest = json.loads(self._manifest().read_text(encoding="utf-8"))
        manifest["trajectory_npz"] = str(self.directory / "wrong-order.npz")
        manifest["trajectory_npz_sha256"] = hashlib.sha256(Path(manifest["trajectory_npz"]).read_bytes()).hexdigest()
        path = self.directory / "wrong-order.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "atom order|identity"):
            evaluate.evaluate_manifest(path)

    def test_memory_limit_is_checked_from_v3_header_before_package_arrays_are_read(self):
        manifest = self._manifest(memory_limit=1)
        with self.assertRaisesRegex(MemoryError, "estimated.*memory|memory.*estimate"):
            evaluate.evaluate_manifest(manifest)

    def test_rejects_unfrozen_tolerance_and_bad_sidecar_certificate(self):
        manifest = json.loads(self._manifest().read_text(encoding="utf-8"))
        del manifest["tolerances"]["reference_sensitivity_absolute_native"]
        no_tolerance = self.directory / "no-tolerance.json"
        no_tolerance.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tolerance"):
            evaluate.evaluate_manifest(no_tolerance)

        package = self.packages[0]
        sidecar = Path(str(package) + ".stability")
        original = sidecar.read_text(encoding="ascii")
        try:
            sidecar.write_text(original.replace("minimum_positive_eigenvalue", "unsupported_certificate"), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "certificate|sidecar"):
                evaluate.evaluate_manifest(self._manifest())
        finally:
            sidecar.write_text(original, encoding="ascii")

    def test_wrapped_position_evaluation_replays_branch_and_rejects_ambiguous_step(self):
        report, arrays = evaluate.evaluate_manifest(self._manifest())
        with np.load(self.trajectory) as data:
            wrapped = data["wrapped_position_angstrom"]
        self.assertGreater(wrapped[1, 0, 0] - wrapped[2, 0, 0], 19.0)
        self.assertEqual(report["xc_semantics"], "wrapped_position_per_atom_A; production MIC replay per reference R0")
        np.testing.assert_allclose(arrays["reference_0_delta_j_native"], self._dense_oracle(0)[0],
                                   rtol=2.0e-11, atol=2.0e-12)

        with np.load(self.trajectory) as source:
            payload = {key: source[key] for key in source.files}
        payload["wrapped_position_angstrom"] = payload["wrapped_position_angstrom"].copy()
        payload["wrapped_position_angstrom"][5, 0, 0] = 9.9
        bad = self.directory / "ambiguous-position.npz"
        np.savez(bad, **payload)
        manifest = json.loads(self._manifest().read_text(encoding="utf-8"))
        manifest["trajectory_npz"] = str(bad)
        path = self.directory / "ambiguous-position.json"
        manifest["trajectory_npz_sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ambiguous|branch|0.45"):
            evaluate.evaluate_manifest(path)

    def test_rejects_npz_shape_header_before_array_loading(self):
        bad = self.directory / "forged-shape.npz"
        with np.load(self.trajectory, allow_pickle=False) as source:
            members = {key + ".npy": source[key] for key in source.files}
        with zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, array in members.items():
                if name == "wrapped_position_angstrom.npy":
                    stream = io.BytesIO()
                    np.lib.format.write_array_header_1_0(stream, {
                        "descr": np.dtype("<f8").str, "fortran_order": False,
                        "shape": (12, 2**31, 3)})
                    archive.writestr(name, stream.getvalue())
                else:
                    stream = io.BytesIO()
                    np.lib.format.write_array(stream, array, version=(1, 0))
                    archive.writestr(name, stream.getvalue())
        manifest = json.loads(self._manifest().read_text(encoding="utf-8"))
        manifest["trajectory_npz"] = str(bad)
        manifest["trajectory_npz_sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
        path = self.directory / "forged-shape.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch.object(np, "load", side_effect=AssertionError("must reject header before loading")):
            with self.assertRaisesRegex(ValueError, "shape"):
                evaluate.evaluate_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
