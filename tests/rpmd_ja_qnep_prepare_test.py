import importlib.util
import struct
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qprep", ROOT / "tools" / "rpmd_ja_qnep_prepare.py")
qprep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qprep)


def _read_tiles(path, d):
    matrices = np.zeros((d, d))
    with path.open("rb") as stream:
        tile_size, count = struct.unpack("<iQ", stream.read(12))
        assert tile_size == qprep.TILE
        seen = set()
        for _ in range(count):
            row, col, rows, cols, rank, nl, nr = struct.unpack("<5iQQ", stream.read(36))
            left = np.fromfile(stream, dtype="<f8", count=nl)
            right = np.fromfile(stream, dtype="<f8", count=nr)
            if rank:
                a = left.reshape(rows, rank) @ right.reshape(rank, cols)
            else:
                a = left.reshape(rows, cols)
            matrices[row:row+rows, col:col+cols] = a
            seen.add((row, col))
        assert len(seen) == count
        assert stream.read() == b""
    return matrices


def test_native_transport_uses_full_virial_and_chain_force_terms():
    n, d, alpha = 3, 9, 0
    rng = np.random.default_rng(41)
    v = rng.normal(size=(d, n))
    vc = v + 0.01 * rng.normal(size=(d, n))
    c = rng.normal(size=(3, d, d))
    cc = c + 0.01 * rng.normal(size=(3, d, d))
    masses = np.array([1.0, 3.0, 7.0])
    hf = np.empty((d, d)); hc = np.empty((d, d))
    qprep.build_native_transport(v, c[alpha], vc, cc[alpha], masses, n, alpha, hf, hc)
    expected = np.empty((d, d)); expected_coarse = np.empty((d, d))
    for coord in range(d):
        a, nu = coord % n, coord // n
        for row in range(d):
            site, mu = row % n, row // n
            expected[row, coord] = c[alpha, coord, row]
            expected_coarse[row, coord] = cc[alpha, coord, row]
            if nu == alpha:
                expected[row, coord] -= v[row, a]
                expected_coarse[row, coord] -= vc[row, a]
                if site == a:
                    expected[row, coord] -= -v[row].sum()
                    expected_coarse[row, coord] -= -vc[row].sum()
    coord_mass = np.tile(masses, 3)
    scale = np.sqrt(coord_mass[:, None] * coord_mass[None, :])
    np.testing.assert_allclose(hf, expected / scale, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(hc, expected_coarse / scale, rtol=1e-14, atol=1e-14)


def test_dense_and_lowrank_tile_wire_roundtrip_and_safe_bound():
    rng = np.random.default_rng(9)
    q = rng.normal(size=(7, 7))
    dmat = q @ q.T + np.eye(7)
    bmat = rng.normal(size=(7, 7))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for index, matrix in enumerate((dmat, bmat)):
            path = directory / f"{index}.tiles"
            residual, bound = qprep._write_tiles(path, matrix, index == 0, 1e-12, False)
            roundtrip = _read_tiles(path, len(matrix))
            np.testing.assert_allclose(roundtrip, matrix, rtol=1e-12, atol=1e-12)
            assert residual < 1e-12
            assert bound + 1e-12 >= float(np.max(np.sum(np.abs(roundtrip), axis=1)))


def test_block_compression_does_not_drop_tiles_or_symmetry():
    d = 131
    rng = np.random.default_rng(18)
    q = rng.normal(size=(d, d))
    matrix = q @ q.T + np.eye(d)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "D.tiles"
        residual, _ = qprep._write_tiles(path, matrix, True, 1e-7, True)
        recovered = _read_tiles(path, d)
        assert recovered.shape == (d, d)
        np.testing.assert_allclose(recovered, recovered.T, atol=1e-13)
        assert residual == 0
        np.testing.assert_allclose(recovered, matrix, rtol=1e-13, atol=1e-13)


def test_kernel_coverage_uses_gpumd_internal_time_constants():
    common = (ROOT / "src" / "utilities" / "common.cuh").read_text(encoding="utf-8")
    assert "#define HBAR 6.465412e-2" in common
    assert "#define K_B 8.617343e-5" in common
    temperature, spectral_bound = 300.0, 4.0
    expected = 6.465412e-2 / (8.617343e-5 * temperature) * np.sqrt(spectral_bound)
    assert qprep.HBAR / (qprep.KB * temperature) * np.sqrt(spectral_bound) == expected
    assert expected < 10.0


def test_translation_complement_keeps_soft_modes_and_rejects_negative_curvature():
    n, d = 4, 12
    translation = np.full(n, 0.5)
    basis = np.zeros((d, 3))
    for axis in range(3): basis[axis*n:(axis+1)*n, axis] = translation
    relative = np.zeros(d); relative[0] = 1/np.sqrt(2); relative[1] = -1/np.sqrt(2)
    stable = np.eye(d) - basis @ basis.T - (1.0 - 1e-10) * np.outer(relative, relative)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "D.f64"
        matrix = np.memmap(path, dtype="<f8", mode="w+", shape=(d, d)); matrix[:] = stable; matrix.flush()
        eigen = qprep._eigenvalues(matrix, translation, None)
        assert 0 < eigen[0] < 2e-10
        matrix[:] = stable - 2e-10 * np.outer(relative, relative); matrix.flush()
        try:
            qprep._eigenvalues(matrix, translation, None)
        except ValueError:
            pass
        else:
            raise AssertionError("a physical -1e-10 soft mode must be rejected")
        matrix._mmap.close()


def test_loewner_certificate_bounds_softmode_relative_error():
    masses = np.array([1.0, 4.0, 9.0, 16.0])
    t = np.sqrt(masses / masses.sum())
    householder = np.eye(4) - 2.0 * np.outer((t - np.eye(4)[0]) / np.linalg.norm(t - np.eye(4)[0]),
                                               (t - np.eye(4)[0]) / np.linalg.norm(t - np.eye(4)[0]))
    np.testing.assert_allclose(householder @ t, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-14)
    physical = np.diag([1e-10, 0.2, 2.0])
    a0 = np.zeros((4, 4)); a0[1:, 1:] = physical
    ac = a0.copy(); ac[1:, 1:] = physical * np.diag([1.005, 0.997, 1.002])
    l = np.linalg.cholesky(a0[1:, 1:])
    delta = ac[1:, 1:] - a0[1:, 1:]
    f = np.linalg.solve(l, delta)
    f = np.linalg.solve(l, f.T).T
    bound = np.linalg.norm(f, "fro")
    generalized = np.linalg.eigvals(np.linalg.solve(a0[1:, 1:], ac[1:, 1:])).real
    relative_frequency_error = np.max(np.abs(np.sqrt(generalized) - 1.0))
    assert bound >= np.max(np.abs(generalized - 1.0))
    assert relative_frequency_error < 0.01
    assert bound < 0.01
    ac_bad = a0.copy(); ac_bad[1:, 1:] = physical * np.diag([1.02, 1.0, 1.0])
    delta_bad = ac_bad[1:, 1:] - a0[1:, 1:]
    f_bad = np.linalg.solve(l, delta_bad)
    f_bad = np.linalg.solve(l, f_bad.T).T
    assert np.linalg.norm(f_bad, "fro") > 0.01
    ac_negative = a0.copy(); ac_negative[1, 1] = -1e-10
    try:
        np.linalg.cholesky(ac_negative[1:, 1:])
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("a physical -1e-10 soft mode must fail strict Cholesky")


def test_householder_anchor_permutation_handles_small_n_and_unequal_masses():
    for masses in (np.array([1.0, 4.0]), np.array([1.0, 4.0, 9.0])):
        n = len(masses); d = 3 * n
        t = np.sqrt(masses / masses.sum())
        q = np.eye(d)
        for axis in range(3):
            v = t.copy(); v[0] -= 1.0
            v /= np.linalg.norm(v)
            block = np.eye(n) - 2.0 * np.outer(v, v)
            q[axis*n:(axis+1)*n, axis*n:(axis+1)*n] = block
        anchors = [0, n, 2*n]
        permutation = [i for i in range(d) if i not in anchors] + anchors
        p = np.eye(d)[:, permutation]
        relative = q.T @ p[:, :d-3]
        np.testing.assert_allclose(relative.T @ relative, np.eye(d-3), atol=2e-14)
        for axis in range(3):
            tau = np.zeros(d); tau[axis*n:(axis+1)*n] = t
            np.testing.assert_allclose(relative.T @ tau, 0.0, atol=2e-14)
        a = np.eye(d) - np.column_stack([
            np.pad(t, (axis*n, d-(axis+1)*n)) for axis in range(3)
        ]) @ np.column_stack([
            np.pad(t, (axis*n, d-(axis+1)*n)) for axis in range(3)
        ]).T
        np.testing.assert_allclose(relative.T @ a @ relative, np.eye(d-3), atol=2e-14)


def test_prepare_end_to_end_writes_v3_and_bound_sidecar():
    n, d = 4, 12
    masses = np.array([1.0, 2.0, 3.0, 5.0])
    lap = np.full((n, n), -1.0); np.fill_diagonal(lap, n - 1)
    hessian = np.kron(np.eye(3), lap)
    zeros_v = np.zeros((d, n)); zeros_c = np.zeros((3, d, d))
    cell = np.zeros(18); cell[:9] = np.diag([10.0, 11.0, 12.0]).ravel()
    cell[9:] = np.diag([0.1, 1.0/11.0, 1.0/12.0]).ravel()
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw = directory / "fixture.qraw"
        kernel = directory / "kernel.txt"
        output = directory / "fixture.rpmdja"
        with raw.open("wb") as f:
            f.write(qprep.RAW_MAGIC); f.write(struct.pack("<IIii", 1, qprep.ENDIAN, n, d))
            f.write(struct.pack("<ddQQ", 300.0, 0.01, 12345, 98765))
            f.write(struct.pack("<iid", 1, 1, 1.0)); f.write(qprep.RAW_LAYOUT)
            f.write(cell.astype("<f8").tobytes()); f.write(np.ones(3, dtype="<i4").tobytes())
            f.write(np.zeros(n, dtype="<i4").tobytes()); f.write(masses.astype("<f8").tobytes())
            f.write(np.zeros(d, dtype="<f8").tobytes()); f.write(struct.pack("<d", 0.0))
            f.write(np.zeros(n, dtype="<f8").tobytes()); f.write(np.zeros(d, dtype="<f8").tobytes())
            f.write(np.zeros(9*n, dtype="<f8").tobytes())
            f.write(zeros_v.astype("<f8").tobytes())
            f.write(zeros_c.astype("<f8").tobytes())
            f.write(hessian.T.astype("<f8").tobytes())
            f.write(zeros_v.astype("<f8").tobytes()); f.write(zeros_c.astype("<f8").tobytes())
            f.write(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01,1], dtype="<f8").tobytes())
        kernel.write_text("GPUMDJA_KERNEL 1\nU 100\ndegree 0\nP_rank 1\nQ_rank 1\nP_error 0\nQ_error 0\nP_S2 0\nQ_S2 0\nP_values\n1\nP_vectors\n1\nQ_values\n1\nQ_vectors\n1\nEND\n", encoding="ascii")
        qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2)
        assert output.exists()
        assert Path(str(output) + ".stability").exists()
        with output.open("rb") as f:
            assert f.read(8) == qprep.MAGIC
            version, endian, atoms, temperature, fd_step, model_fp = struct.unpack("<IIiddQ", f.read(36))
        assert (version, endian, atoms, temperature, fd_step, model_fp) == (3, qprep.ENDIAN, n, 300.0, 0.01, 12345)
        sidecar = Path(str(output) + ".stability").read_text(encoding="ascii")
        assert f"fingerprint {qprep._fnv64(output):016x}" in sidecar
        assert "config_fingerprint 00000000000181cd" in sidecar
        assert not Path(str(output) + ".work").exists()


if __name__ == "__main__":
    for test in (test_native_transport_uses_full_virial_and_chain_force_terms,
                 test_dense_and_lowrank_tile_wire_roundtrip_and_safe_bound,
                 test_block_compression_does_not_drop_tiles_or_symmetry,
                 test_kernel_coverage_uses_gpumd_internal_time_constants,
                 test_translation_complement_keeps_soft_modes_and_rejects_negative_curvature,
                 test_loewner_certificate_bounds_softmode_relative_error,
                 test_householder_anchor_permutation_handles_small_n_and_unequal_masses,
                 test_prepare_end_to_end_writes_v3_and_bound_sidecar):
        test()
    print("qNEP RPMD-JA prepare tests passed")
