import math
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from decimal import Decimal, localcontext

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from rpmd_ja_sparse_check import Reader, check, read_kernel_table, read_reference, rebind_kernel
from rpmd_ja_export_kernel import _direct_pq, _export_npz, _g, _write_table

UNITS = b"position:A;energy:eV;mass:amu;temperature:K;time:native\0"
LAYOUT = b"xyz_soa;csr_output_row_input_col;runtime_translation_projection;cheb_rowmajor_degree_rank;fixed_d0_edges_v1\0"


def csr(matrix):
    rows, cols = np.nonzero(matrix)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    values = matrix[rows, cols]
    offsets = np.zeros(matrix.shape[0] + 1, dtype=np.uint64)
    np.add.at(offsets, rows + 1, 1)
    np.cumsum(offsets, out=offsets)
    return offsets, cols.astype(np.int32), values.astype(np.float64)


def write_v2(path, masses, positions, dynamical, site_transpose):
    n, d = len(masses), 3 * len(masses)
    degree, p_rank, q_rank = 2, 1, 1
    values = np.asarray([0.01])
    vectors = np.asarray([1.0, 0.0, 0.0])
    with path.open("wb") as out:
        out.write(b"GPUMDJA\0")
        out.write(struct.pack("<IIiddQ", 2, 0x01020304, n, 300.0, 1.0e-4, 0))
        out.write(UNITS + LAYOUT)
        out.write(struct.pack("<9d3i", 10, 0, 0, 0, 10, 0, 0, 0, 10, 1, 1, 1))
        out.write(np.asarray(np.zeros(n), dtype="<i4").tobytes())
        out.write(np.asarray(masses, dtype="<f8").tobytes())
        out.write(np.asarray(positions, dtype="<f8").tobytes())
        out.write(struct.pack("<iQQ", 1, 3, 0x123456789abcdef0))
        for matrix in [dynamical] + list(site_transpose):
            offsets, cols, vals = csr(matrix)
            out.write(struct.pack("<Q", len(vals)))
            out.write(offsets.astype("<u8").tobytes())
            out.write(cols.astype("<i4").tobytes())
            out.write(vals.astype("<f8").tobytes())
        out.write(struct.pack("<2d", float(np.max(np.sum(np.abs(dynamical), axis=1))), 8.0))
        out.write(struct.pack("<2d", 1.0e-8, 1.0e-8))
        out.write(struct.pack("<2d", 10.0, 10.0))
        out.write(struct.pack("<3id", degree, p_rank, q_rank, 0.01))
        out.write(struct.pack("<3d", 0.01, 0.01, 0.01))
        out.write(values.astype("<f8").tobytes() * 2)
        out.write(vectors.astype("<f8").tobytes() * 2)


def orthonormal_translations(masses):
    n = len(masses)
    t = np.zeros((3 * n, 3))
    for axis in range(3):
        t[axis * n : (axis + 1) * n, axis] = np.sqrt(masses / np.sum(masses))
    return t


def image_key_orthogonal(center, neighbor, length):
    raw = neighbor - center
    displacement = raw
    if displacement < -0.5 * length:
        displacement += length
    elif displacement > 0.5 * length:
        displacement -= length
    return round((displacement - raw) / length)


def chebyshev_apply(d, vector, coeff, scale):
    def z_apply(x):
        return 2.0 * (d @ x) / scale - x
    previous = vector
    result = coeff[0] * previous
    if len(coeff) == 1:
        return result
    current = z_apply(vector)
    result = result + coeff[1] * current
    for coefficient in coeff[2:]:
        previous, current = current, 2.0 * z_apply(current) - previous
        result = result + coefficient * current
    return result


def lowrank_frame_correction(d, bt, y, velocity, tau, table):
    p_values, p_vectors = table["P"]
    q_values, q_vectors = table["Q"]
    scale = (table["U"] / tau) ** 2
    dy, du = d @ y, d @ velocity
    total = 0.0
    for rank, value in enumerate(p_values):
        left = chebyshev_apply(d, dy, p_vectors[:, rank], scale)
        right = chebyshev_apply(d, du, p_vectors[:, rank], scale)
        total += tau**4 * value * ((bt @ left) @ right)
    for rank, value in enumerate(q_values):
        left = chebyshev_apply(d, dy, q_vectors[:, rank], scale)
        right = chebyshev_apply(d, velocity, q_vectors[:, rank], scale)
        total += tau**2 * value * (left @ (bt @ right))
    return float(total)


def dense_table_frame_correction(modes, eigenvalues, bt, y, velocity, tau, table):
    u_table = table["U"]
    b_modal = modes.T @ bt.T @ modes
    y_modal, v_modal = modes.T @ y, modes.T @ velocity
    delta = np.zeros_like(b_modal)
    indices = np.arange(table["degree"] + 1)
    for a, lam_a in enumerate(eigenvalues):
        for b, lam_b in enumerate(eigenvalues):
            ua, ub = tau * math.sqrt(lam_a), tau * math.sqrt(lam_b)
            x = 2.0 * (ua / u_table) ** 2 - 1.0
            z = 2.0 * (ub / u_table) ** 2 - 1.0
            tx = np.cos(indices * math.acos(max(-1.0, min(1.0, x))))
            tz = np.cos(indices * math.acos(max(-1.0, min(1.0, z))))
            pvals, pvecs = table["P"]
            qvals, qvecs = table["Q"]
            p = float(tx @ (pvecs * pvals[None, :]) @ pvecs.T @ tz)
            q = float(tx @ (qvecs * qvals[None, :]) @ qvecs.T @ tz)
            delta[a, b] = tau**4 * lam_a * lam_b * p * b_modal[a, b] + tau**2 * lam_a * q * b_modal[b, a]
    return float(y_modal @ delta @ v_modal)


def direct_fock_frame_correction(modes, eigenvalues, bt, y, velocity, tau):
    b_modal = modes.T @ bt.T @ modes
    y_modal, v_modal = modes.T @ y, modes.T @ velocity
    delta = np.zeros_like(b_modal)
    with localcontext() as context:
        context.prec = 90
        for a, lam_a in enumerate(eigenvalues):
            for b, lam_b in enumerate(eigenvalues):
                ua = Decimal(repr(tau * math.sqrt(lam_a)))
                ub = Decimal(repr(tau * math.sqrt(lam_b)))
                p, q = _direct_pq(ua, ub)
                delta[a, b] = tau**4 * lam_a * lam_b * float(p) * b_modal[a, b] + tau**2 * lam_a * float(q) * b_modal[b, a]
    return float(y_modal @ delta @ v_modal)


def periodic_ring_csr(n, masses, channel=0, transpose=False, zero=False):
    d = 3 * n
    offsets = np.arange(d + 1, dtype=np.uint64) * 3
    columns = np.empty(3 * d, dtype=np.int32)
    values = np.empty(3 * d, dtype=np.float64)
    for axis in range(3):
        for atom in range(n):
            row = axis * n + atom
            left, right = axis * n + (atom - 1) % n, axis * n + (atom + 1) % n
            if not transpose:
                entries = [(left, -1.0 / math.sqrt(masses[atom] * masses[(atom - 1) % n])),
                           (row, 2.0 / masses[atom]),
                           (right, -1.0 / math.sqrt(masses[atom] * masses[(atom + 1) % n]))]
            else:
                entries = [(left, 0.07 + 0.003 * ((atom + channel) % 7)),
                           (row, 0.2 + 0.01 * channel),
                           (right, -0.04 - 0.002 * ((atom + 2 * channel) % 5))]
            entries.sort()
            for k, (column, value) in enumerate(entries):
                columns[3 * row + k] = column
                values[3 * row + k] = 0.0 if zero else value
    return offsets, columns, values


def csr_apply(matrix, vector):
    offsets, columns, values = matrix
    if np.all(offsets[1:] - offsets[:-1] == offsets[1] - offsets[0]):
        width = int(offsets[1] - offsets[0])
        return np.sum(values.reshape(-1, width) * vector[columns.reshape(-1, width)], axis=1)
    result = np.empty_like(vector)
    for row in range(len(vector)):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        result[row] = values[begin:end] @ vector[columns[begin:end]]
    return result


def sparse_frame_correction(d0, bt0, translations, vector_x, vector_v, tau, table):
    def project(x):
        return x - translations @ (translations.T @ x)
    def apply_d(x):
        return project(csr_apply(d0, project(x)))
    def apply_bt(x):
        return csr_apply(bt0, project(x))
    def cheb(x, coeff, scale):
        def z_apply(y): return 2.0 * apply_d(y) / scale - y
        previous = x
        result = coeff[0] * previous
        if len(coeff) == 1: return result
        current = z_apply(x)
        result = result + coeff[1] * current
        for c in coeff[2:]:
            previous, current = current, 2.0 * z_apply(current) - previous
            result = result + c * current
        return result
    p_values, p_vectors = table["P"]
    q_values, q_vectors = table["Q"]
    scale = (table["U"] / tau) ** 2
    dx, dv = apply_d(vector_x), apply_d(vector_v)
    result = 0.0
    for rank, value in enumerate(p_values):
        left = cheb(dx, p_vectors[:, rank], scale)
        right = cheb(dv, p_vectors[:, rank], scale)
        result += tau**4 * value * (apply_bt(left) @ right)
    for rank, value in enumerate(q_values):
        left = cheb(dx, q_vectors[:, rank], scale)
        right = cheb(vector_v, q_vectors[:, rank], scale)
        result += tau**2 * value * (left @ apply_bt(right))
    return float(result)


def load_table(path):
    fields = path.read_text(encoding="ascii").split()
    cursor = 0
    def take(expected):
        nonlocal cursor
        assert fields[cursor] == expected
        cursor += 1
    take("GPUMDJA_KERNEL"); assert fields[cursor] == "1"; cursor += 1
    take("U"); u = float(fields[cursor]); cursor += 1
    take("degree"); degree = int(fields[cursor]); cursor += 1
    take("P_rank"); p_rank = int(fields[cursor]); cursor += 1
    take("Q_rank"); q_rank = int(fields[cursor]); cursor += 1
    take("P_error"); p_error = float(fields[cursor]); cursor += 1
    take("Q_error"); q_error = float(fields[cursor]); cursor += 1
    take("P_S2"); p_s2 = float(fields[cursor]); cursor += 1
    take("Q_S2"); q_s2 = float(fields[cursor]); cursor += 1
    arrays = []
    for name, rank in (("P", p_rank), ("Q", q_rank)):
        take(f"{name}_values")
        values = np.asarray([float(x) for x in fields[cursor : cursor + rank]])
        cursor += rank
        take(f"{name}_vectors")
        vectors = np.asarray([float(x) for x in fields[cursor : cursor + (degree + 1) * rank]]).reshape(degree + 1, rank)
        cursor += (degree + 1) * rank
        arrays.append((values, vectors))
    take("END")
    return {"U": u, "degree": degree, "P": arrays[0], "Q": arrays[1],
            "error": np.asarray([p_error, q_error]), "s2": np.asarray([p_s2, q_s2])}


class SparseReferenceTests(unittest.TestCase):
    def test_reader_detects_short_array_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.bin"
            path.write_bytes(b"12345678")
            reader = Reader(path)
            try:
                reader.size = 16  # Simulate truncation after the initial file-size snapshot.
                with self.assertRaisesRegex(ValueError, "truncated sparse RPMD-JA file"):
                    reader.array("<f8", 2)
            finally:
                reader.stream.close()

    def test_kernel_rank_is_bounded_by_polynomial_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "oversized.txt"
            table.write_text(
                "GPUMDJA_KERNEL 1\nU 8\ndegree 1\nP_rank 3\nQ_rank 1\n"
                "P_error 0\nQ_error 0\nP_S2 0\nQ_S2 0\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "invalid kernel table bounds"):
                read_kernel_table(table)
            with self.assertRaisesRegex(ValueError, r"degree\+1"):
                _write_table(root / "bad.txt", 8.0, 1,
                             [np.ones(3), np.ones(1)], [np.ones((2, 3)), np.ones((2, 1))],
                             [0.0, 0.0], [0.0, 0.0])

            bad_npz = root / "bad.npz"
            metadata = dict(total_kernel_budget=np.zeros(2), interpolation_bounds=np.zeros(2),
                            rank_l1=np.zeros(2), arithmetic_budget=np.zeros(2))
            np.savez(bad_npz, U=8.0, degree=1, P_values=np.ones(3), P_vectors=np.ones((2, 3)),
                     Q_values=np.ones(1), Q_vectors=np.ones((2, 1)), **metadata)
            bad_output = root / "from_bad_npz.txt"
            with self.assertRaisesRegex(ValueError, r"degree\+1"):
                _export_npz(bad_npz, bad_output)
            self.assertFalse(bad_output.exists())
            bad_degree_npz = root / "bad_degree.npz"
            np.savez(bad_degree_npz, U=8.0, degree=513, P_values=np.ones(1), P_vectors=np.ones((514, 1)),
                     Q_values=np.ones(1), Q_vectors=np.ones((514, 1)), **metadata)
            with self.assertRaisesRegex(ValueError, r"degree\+1"):
                _export_npz(bad_degree_npz, root / "from_bad_degree_npz.txt")

            reference = root / "oversized.rpmdja"
            with reference.open("wb") as out:
                out.write(b"GPUMDJA\0")
                out.write(struct.pack("<IIiddQ", 2, 0x01020304, 2, 300.0, 1.0e-4, 0))
                out.write(UNITS + LAYOUT)
                out.write(struct.pack("<9d3i", *([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 1, 1, 1])))
                out.write(struct.pack("<2i", 0, 0) + struct.pack("<2d", 1.0, 1.0) + struct.pack("<6d", *([0.0] * 6)))
                out.write(struct.pack("<iQQ", 1, 0, 0))
                for _ in range(4):
                    out.write(struct.pack("<Q", 0))
                    out.write(struct.pack("<7Q", *([0] * 7)))
                out.write(struct.pack("<2d", 1.0, 8.0) + struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))
                out.write(struct.pack("<3i", 1, 3, 1))
                out.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))
            with self.assertRaisesRegex(ValueError, "invalid kernel ranks/degree"):
                read_reference(reference)

    def test_decimal_kernel_zero_axis_limits(self):
        tiny = Decimal("1e-14")
        with localcontext() as context:
            context.prec = 90
            expected_g = Decimal(1) + tiny * tiny / 24
            self.assertLess(abs(_g(tiny) - expected_g), Decimal("1e-42"))
            p0, q0 = _direct_pq(Decimal(0), Decimal(0))
            self.assertEqual(p0, Decimal(-1) / 5760)
            self.assertEqual(q0, Decimal(1) / 24)
            _, qaxis = _direct_pq(Decimal(0), Decimal("0.7"))
            self.assertTrue(qaxis.is_finite())
            ptiny, qtiny = _direct_pq(Decimal(0), Decimal("1e-24"))
            self.assertLess(abs(ptiny - Decimal(-1) / 5760), Decimal("1e-30"))
            self.assertLess(abs(qtiny - Decimal(1) / 24), Decimal("1e-30"))

    def test_unwrapped_branch_preserves_periodic_image_keys(self):
        length, step = 10.0, 1.0e-4
        self.assertEqual([image_key_orthogonal(-step, 0.4 * length, length),
                          image_key_orthogonal(step, 0.4 * length, length)], [0, 0])
        self.assertEqual([image_key_orthogonal(-step, 0.9 * length, length),
                          image_key_orthogonal(step, 0.9 * length, length)], [-1, -1])
        self.assertEqual([image_key_orthogonal(0.999 * length - step, 0.4 * length, length),
                          image_key_orthogonal(0.999 * length + step, 0.4 * length, length)], [1, 1])

    def test_45_17_lowrank_recursion_matches_dense_full_table(self):
        masses = np.asarray([1.0, 2.0, 3.0])
        n, d = len(masses), 3 * len(masses)
        t = orthonormal_translations(masses)
        rng = np.random.default_rng(4)
        complement = np.linalg.qr(np.column_stack((t, rng.normal(size=(d, d - 3)))))[0][:, 3:]
        eigenvalues = np.asarray([0.7, 1.1, 1.8, 2.4, 3.1, 4.0])
        modes = np.column_stack((t, complement))
        spectrum = np.r_[np.zeros(3), eigenvalues]
        dynamical = (modes * spectrum) @ modes.T
        projector = np.eye(d) - t @ t.T
        dynamical = projector @ dynamical @ projector
        bt = rng.normal(size=(d, d)) @ projector
        y, velocity = rng.normal(size=d), rng.normal(size=d)
        table = load_table(ROOT / "tests" / "data" / "rpmd_ja_kernel_U8.txt")
        tau = 0.25
        self.assertLess(tau * math.sqrt(np.max(np.sum(np.abs(dynamical), axis=1))), table["U"])
        actual = lowrank_frame_correction(dynamical, bt, y, velocity, tau, table)
        expected = dense_table_frame_correction(modes, spectrum, bt, y, velocity, tau, table)
        self.assertLess(abs(actual - expected), 2.0e-11 * max(1.0, abs(expected)))
        full_fock = direct_fock_frame_correction(modes, spectrum, bt, y, velocity, tau)
        self.assertLess(abs(expected - full_fock), 4.0 * max(table["error"]) * max(1.0, np.linalg.norm(y) * np.linalg.norm(velocity) * np.linalg.norm(bt, ord=2)))
        self.assertFalse(np.allclose(bt, bt.T))

    def test_v2_csr_cholesky_and_dense_oracle(self):
        masses = np.asarray([1.0, 2.0, 3.0])
        n, d = len(masses), 3 * len(masses)
        t = orthonormal_translations(masses)
        projector = np.eye(d) - t @ t.T
        laplacian = np.asarray([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
        d0 = np.zeros((d, d))
        for axis in range(3):
            ids = axis * n + np.arange(n)
            d0[np.ix_(ids, ids)] = laplacian / np.sqrt(masses[:, None] * masses[None, :])
        d0 = projector @ d0 @ projector
        rng = np.random.default_rng(13)
        bt0 = rng.normal(size=(d, d)) @ projector
        self.assertFalse(np.allclose(bt0, bt0.T))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.rpmdja"
            write_v2(path, masses, np.zeros(d), d0, (bt0, bt0 * 0.7, bt0 * -0.2))
            decoded = read_reference(path)
            offsets, columns, values = decoded["dynamical"]
            dense = np.zeros((d, d))
            for row in range(d):
                dense[row, columns[offsets[row] : offsets[row + 1]]] = values[offsets[row] : offsets[row + 1]]
            self.assertLess(np.max(np.abs(dense - d0)), 1.0e-14)
            check(path)
            self.assertTrue(path.with_name(path.name + ".stability").exists())
            rebound = Path(directory) / "small_rebound.rpmdja"
            rebind_kernel(path, ROOT / "tests" / "data" / "rpmd_ja_kernel_U8.txt", rebound)
            rebound_ref = read_reference(rebound)
            np.testing.assert_array_equal(rebound_ref["dynamical"][2], decoded["dynamical"][2])
            self.assertEqual(rebound_ref["edge_fingerprint"], decoded["edge_fingerprint"])
            check(rebound)

    def test_v2_checker_rejects_negative_complement_mode(self):
        masses = np.asarray([1.0, 2.0, 3.0])
        n, d = len(masses), 3 * len(masses)
        t = orthonormal_translations(masses)
        projector = np.eye(d) - t @ t.T
        laplacian = np.asarray([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
        d0 = np.zeros((d, d))
        for axis in range(3):
            ids = axis * n + np.arange(n)
            d0[np.ix_(ids, ids)] = laplacian / np.sqrt(masses[:, None] * masses[None, :])
        d_bad = projector @ d0 @ projector - 2.0 * projector
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.rpmdja"
            write_v2(path, masses, np.zeros(d), d_bad, (np.zeros((d, d)),) * 3)
            with self.assertRaisesRegex(ValueError, "not numerically positive definite"):
                check(path)

    def test_5896_atom_synthetic_sparse_runtime_cost(self):
        n, d = 5896, 3 * 5896
        masses = 1.0 + 0.05 * (np.arange(n) % 11)
        translations = np.zeros((d, 3))
        for axis in range(3):
            translations[axis * n : (axis + 1) * n, axis] = np.sqrt(masses / np.sum(masses))
        d0 = periodic_ring_csr(n, masses)
        bts = [periodic_ring_csr(n, masses, a, transpose=True) for a in range(3)]
        table = load_table(ROOT / "tests" / "data" / "rpmd_ja_kernel_U8.txt")
        row_bound = max(float(np.sum(np.abs(d0[2][d0[0][i]:d0[0][i + 1]]))) for i in range(d))
        tau = 0.1
        self.assertLess(tau * math.sqrt(row_bound), table["U"])
        rng = np.random.default_rng(5896)
        x, velocity = rng.normal(size=d), rng.normal(size=d)
        start = time.perf_counter()
        values = [sparse_frame_correction(d0, bt, translations, x, velocity, tau, table) for bt in bts]
        elapsed = time.perf_counter() - start
        shifted = sparse_frame_correction(d0, bts[0], translations, x + translations @ np.asarray([1.0, -0.3, 0.2]), velocity, tau, table)
        zero_flow = sparse_frame_correction(d0, periodic_ring_csr(n, masses, transpose=True, zero=True), translations, x, velocity, tau, table)
        csr_bytes = sum(a.nbytes for matrix in [d0] + bts for a in matrix)
        rank_vector_bytes = (table["P"][1].nbytes + table["Q"][1].nbytes +
                             table["P"][0].nbytes + table["Q"][0].nbytes)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertLess(abs(values[0] - shifted), 1.0e-10 * max(1.0, abs(values[0])))
        self.assertEqual(zero_flow, 0.0)
        self.assertGreater(max(abs(v) for v in values), 1.0e-12)
        self.assertLess(csr_bytes + rank_vector_bytes + translations.nbytes, 32 * 1024**2)
        print(f"N=5896 synthetic CSR: D+3Bt={csr_bytes} B, kernel vectors={rank_vector_bytes} B, runtime={elapsed:.3f}s; matrix-free, no D^2 or eigenvectors")

    def test_sparse_periodic_nonsymmetric_application_matches_dense_oracle(self):
        n = 5
        masses = np.asarray([1.0, 2.0, 1.5, 3.0, 2.5])
        d0 = periodic_ring_csr(n, masses)
        bt = periodic_ring_csr(n, masses, channel=1, transpose=True)
        def dense(matrix):
            offsets, columns, values = matrix
            result = np.zeros((3 * n, 3 * n))
            for row in range(3 * n):
                result[row, columns[offsets[row]:offsets[row + 1]]] = values[offsets[row]:offsets[row + 1]]
            return result
        d_dense, bt_dense = dense(d0), dense(bt)
        self.assertFalse(np.allclose(bt_dense, bt_dense.T))
        t = orthonormal_translations(masses)
        projector = np.eye(3 * n) - t @ t.T
        x = np.random.default_rng(52).normal(size=3 * n)
        expected_d = projector @ d_dense @ projector @ x
        expected_bt = bt_dense @ projector @ x
        actual_d = projector @ csr_apply(d0, projector @ x)
        actual_bt = csr_apply(bt, projector @ x)
        np.testing.assert_allclose(actual_d, expected_d, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual_bt, expected_bt, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
