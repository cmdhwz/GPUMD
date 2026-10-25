import importlib.util
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIT_SPEC = importlib.util.spec_from_file_location("compact_fit", ROOT / "tools" / "rpmd_ja_fit_reference.py")
fit = importlib.util.module_from_spec(FIT_SPEC)
FIT_SPEC.loader.exec_module(fit)
PREP_SPEC = importlib.util.spec_from_file_location("compact_prepare", ROOT / "tools" / "rpmd_ja_qnep_prepare.py")
prepare = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(prepare)
COMPACT_SPEC = importlib.util.spec_from_file_location("qraw_compact", ROOT / "tools" / "rpmd_ja_qraw_compact.py")
compact = importlib.util.module_from_spec(COMPACT_SPEC)
COMPACT_SPEC.loader.exec_module(compact)


def _write_v3(path, *, bad_stats=False, coarse_v_error=0.0, truncate=False):
    n, d = 3, 9
    masses = np.array([1.0, 2.5, 7.0], dtype="<f8")
    positions = np.arange(3 * n, dtype="<f8") * 0.1
    cell = np.eye(3) * 10.0
    v = np.empty((d, n), dtype="<f8")
    for row in range(d):
        for atom in range(n):
            v[row, atom] = 0.041 * (row + 1) - 0.027 * (atom + 1) + 0.003 * (row + 1) * (atom + 1)
    c = np.empty((3, d, d), dtype="<f8")
    for alpha in range(3):
        for coordinate in range(d):
            for row in range(d):
                c[alpha, coordinate, row] = 0.31 + 0.013 * alpha + 0.004 * coordinate - 0.006 * row + 0.0002 * coordinate * row
    coarse_c = c + 1.0e-4 * (np.arange(3 * d * d).reshape(3, d, d) % 7 + 1)
    coarse_v = v.copy()
    coarse_v[0, 0] += coarse_v_error
    k = np.diag(np.linspace(0.5, 1.5, d)).astype("<f8")
    stats = np.zeros(18, dtype="<f8")
    stats[16], stats[17] = 0.01, 3.0
    if bad_stats:
        stats[1] = 2.0e-4

    with Path(path).open("wb") as out:
        out.write(fit.RAW_MAGIC)
        out.write(struct.pack("<IIii", 3, fit.ENDIAN, n, d))
        out.write(struct.pack("<ddQQ", 300.0, 0.01, 0x1234, 0x5678))
        out.write(struct.pack("<iid", 1, 1, 0.5))
        out.write(fit.RAW_LAYOUT)
        header = np.zeros(18, dtype="<f8")
        header[:9] = cell.ravel()
        header[9:] = np.linalg.inv(cell).ravel()
        out.write(header.tobytes())
        out.write(np.ones(3, dtype="<i4").tobytes())
        out.write(np.arange(n, dtype="<i4").tobytes())
        out.write(masses.tobytes())
        out.write(positions.tobytes())
        out.write(struct.pack("<d", 0.0))
        out.write(np.zeros(n, dtype="<f8").tobytes())
        out.write(np.zeros(d, dtype="<f8").tobytes())
        out.write(np.zeros(9 * n, dtype="<f8").tobytes())
        out.write(v.tobytes())
        out.write(c.tobytes())
        out.write(k.tobytes())
        out.write(coarse_v.tobytes())
        out.write(coarse_c.tobytes())
        out.write(stats.tobytes())
    if truncate:
        Path(path).write_bytes(Path(path).read_bytes()[:-8])
    return v, c, coarse_v, coarse_c, k, masses


def _write_cancellation_v3(path):
    n, d = 8, 24
    masses = np.ones(n, dtype="<f8")
    positions = np.zeros((n, 3), dtype="<f8")
    cell = np.eye(3) * 10.0
    row = np.array([1.0, 1e-16, 1e-16, -1.0, 1e-16, 1e-16, 1e-16, 1e-16], dtype="<f8")
    v = np.tile(row, (d, 1))
    native_sums = np.empty(d, dtype=np.float64)
    for r in range(d):
        value = 0.0
        for site in range(n):
            value += float(v[r, site])
        native_sums[r] = value
    c = np.zeros((3, d, d), dtype="<f8")
    for coordinate in range(n):
        for r in range(d):
            c[0, coordinate, r] = v[r, coordinate]
            if r % n == coordinate:
                c[0, coordinate, r] -= native_sums[r]
    lap = np.full((n, n), -1.0)
    np.fill_diagonal(lap, n - 1)
    k = np.kron(np.eye(3), lap).astype("<f8")
    stats = np.zeros(18, dtype="<f8")
    stats[16], stats[17] = 0.5, 3.0

    with Path(path).open("wb") as out:
        out.write(fit.RAW_MAGIC)
        out.write(struct.pack("<IIii", 3, fit.ENDIAN, n, d))
        out.write(struct.pack("<ddQQ", 300.0, 0.5, 0x1234, 0x5678))
        out.write(struct.pack("<iid", 1, 1, 0.5))
        out.write(fit.RAW_LAYOUT)
        header = np.zeros(18, dtype="<f8")
        header[:9] = cell.ravel()
        header[9:] = np.linalg.inv(cell).ravel()
        out.write(header.tobytes())
        out.write(np.ones(3, dtype="<i4").tobytes())
        out.write(np.arange(n, dtype="<i4").tobytes())
        out.write(masses.tobytes())
        out.write(positions.ravel().tobytes())
        out.write(struct.pack("<d", 0.0))
        out.write(np.zeros(n, dtype="<f8").tobytes())
        out.write(np.zeros(d, dtype="<f8").tobytes())
        out.write(np.zeros(9 * n, dtype="<f8").tobytes())
        out.write(v.tobytes())
        out.write(c.tobytes())
        out.write(k.tobytes())
        out.write(v.tobytes())
        out.write(c.tobytes())
        out.write(stats.tobytes())
    return v, c, k, masses, native_sums


def _close(raw):
    compact._close_raw(raw)


def test_v3_compact_cli_preserves_matrices_and_recomputes_native_h_statistics():
    with tempfile.TemporaryDirectory() as td:
        source, destination = Path(td) / "source-v3.qraw", Path(td) / "converted-v4.qraw"
        v, c, coarse_v, coarse_c, k, masses = _write_v3(source)
        command = [sys.executable, str(ROOT / "tools" / "rpmd_ja_qraw_compact.py"), str(source), str(destination)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "derivative_version=3" in result.stdout
        old = fit.read_raw(source)
        new = fit.read_raw(destination)
        try:
            assert old["version"] == 3 and new["version"] == 4 and new["derivative_version"] == 3
            assert fit._raw_dimensions(destination) == (3, 9)
            assert old["source_fingerprint"] != new["source_fingerprint"]
            for key in ("n", "d", "temperature", "model_fingerprint", "config_fingerprint", "cell", "types", "masses", "positions"):
                np.testing.assert_array_equal(old[key], new[key])
            np.testing.assert_array_equal(old["v"], new["v"])
            np.testing.assert_array_equal(old["c"], new["c"])
            np.testing.assert_array_equal(old["k"], new["k"])
            np.testing.assert_array_equal(old["v"], coarse_v)

            expected_diff, expected_norm = [], []
            for alpha in range(3):
                hf, hc = np.empty((9, 9)), np.empty((9, 9))
                prepare.build_native_transport(v, c[alpha], coarse_v, coarse_c[alpha], masses, 3, alpha, hf, hc)
                compact_h = np.empty((9, 9))
                prepare.build_native_transport_fine(new["v"], new["c"][alpha], new["masses"], 3, alpha, compact_h)
                np.testing.assert_array_equal(hf, compact_h)
                expected_diff.append(float(np.sum((hf - hc) ** 2)))
                expected_norm.append(float(np.sum(hf ** 2)))
            np.testing.assert_allclose(new["transport_diff2"], expected_diff, rtol=2e-13, atol=1e-13)
            np.testing.assert_allclose(new["transport_norm2"], expected_norm, rtol=2e-13, atol=1e-13)
            assert np.all(new["transport_norm2"] > 0.0)
        finally:
            _close(old)
            _close(new)
        assert source.exists()
        assert source.stat().st_size == old["start"] + (2 * 9 * 3 + 7 * 9 * 9 + 18) * 8
        assert destination.stat().st_size == new["start"] + (9 * 3 + 4 * 9 * 9 + 24) * 8
        assert destination.stat().st_size < source.stat().st_size


def test_v3_compactor_rejects_bad_diagnostics_coarse_v_and_truncated_source():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, options, expected in (
            ("bad-stats", {"bad_stats": True}, "diagnostics"),
            ("bad-coarse-v", {"coarse_v_error": 1e-12}, "coarse V"),
            ("truncated", {"truncate": True}, "length"),
        ):
            source, destination = td / f"{name}.qraw", td / f"{name}.v4.qraw"
            _write_v3(source, **options)
            try:
                compact.compact_v3(source, destination)
            except (ValueError, RuntimeError) as exc:
                assert expected.lower() in str(exc).lower(), str(exc)
            else:
                raise AssertionError(f"{name} input was accepted")
            assert not destination.exists()


def test_v3_compactor_never_overwrites_destination():
    with tempfile.TemporaryDirectory() as td:
        source, destination = Path(td) / "source.qraw", Path(td) / "existing.qraw"
        _write_v3(source)
        destination.write_bytes(b"preserve me")
        try:
            compact.compact_v3(source, destination)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
        assert destination.read_bytes() == b"preserve me"


def test_compactor_uses_sha256_and_reader_default_keeps_fingerprint():
    with tempfile.TemporaryDirectory() as td:
        source, destination = Path(td) / "source.qraw", Path(td) / "compact.qraw"
        _write_v3(source)
        original_fnv = compact.fit._fnv64
        compact.fit._fnv64 = lambda _: (_ for _ in ()).throw(AssertionError("converter called Python FNV"))
        try:
            result = compact.compact_v3(source, destination)
        finally:
            compact.fit._fnv64 = original_fnv
        assert result["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert result["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
        parsed = compact.fit.read_raw(source)
        try:
            assert parsed["source_fingerprint"] == original_fnv(source)
        finally:
            compact._close_raw(parsed)
        parsed_without_hash = compact.fit.read_raw(source, compute_source_fingerprint=False)
        try:
            assert "source_fingerprint" not in parsed_without_hash
        finally:
            compact._close_raw(parsed_without_hash)


def test_compactor_rejects_source_mutation_during_conversion():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, offset_kind in (("header", "header"), ("fine-c", "fine-c")):
            source, destination = td / f"{name}.qraw", td / f"{name}.v4.qraw"
            _write_v3(source)
            raw = compact.fit.read_raw(source, compute_source_fingerprint=False)
            try:
                offset = 40 if offset_kind == "header" else raw["start"] + raw["d"] * raw["n"] * 8
            finally:
                compact._close_raw(raw)
            original_copy_prefix = compact._copy_prefix

            def mutate_then_copy(stream, output, count, source_digest, output_digest):
                with source.open("r+b") as changed:
                    changed.seek(offset)
                    old = changed.read(1)
                    changed.seek(offset)
                    changed.write(bytes([old[0] ^ 1]))
                    changed.flush()
                return original_copy_prefix(stream, output, count, source_digest, output_digest)

            compact._copy_prefix = mutate_then_copy
            try:
                try:
                    compact.compact_v3(source, destination)
                except ValueError as exc:
                    assert "changed during conversion" in str(exc).lower()
                else:
                    raise AssertionError(f"{name} source mutation was published")
            finally:
                compact._copy_prefix = original_copy_prefix
            assert source.exists() and not destination.exists()
            with source.open("rb") as changed:
                changed.seek(offset)
                assert changed.read(1)[0] != _write_v3_value_at_offset(source, offset)


def _write_v3_value_at_offset(path, offset):
    """Return the fixture's original byte by recreating it in a scratch path."""
    with tempfile.TemporaryDirectory() as td:
        original = Path(td) / "original.qraw"
        _write_v3(original)
        with original.open("rb") as source:
            source.seek(offset)
            return source.read(1)[0]


def test_v4_cancellation_uses_native_left_to_right_v_sums():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        source, compacted = td / "cancellation-v3.qraw", td / "cancellation-v4.qraw"
        v, c, k, masses, native_sums = _write_cancellation_v3(source)
        assert float(np.sum(v[0])) != native_sums[0]

        # Independent scalar oracle mirrors the native per-site accumulation and H formula.
        expected_norm2 = []
        for alpha in range(3):
            h = np.zeros((24, 24), dtype=np.float64)
            for coordinate in range(24):
                component, atom = divmod(coordinate, 8)
                for r in range(24):
                    q = float(c[alpha, coordinate, r])
                    if component == alpha and r % 8 == atom:
                        q -= -native_sums[r]
                    vv = float(v[r, atom]) if component == alpha else 0.0
                    h[r, coordinate] = (q - vv) / np.sqrt(masses[r % 8] * masses[atom])
            total = 0.0
            for value in h.ravel():
                total += float(value) * float(value)
            expected_norm2.append(total)
        assert expected_norm2[0] == 0.0

        compact.compact_v3(source, compacted)
        raw = fit.read_raw(compacted)
        try:
            np.testing.assert_allclose(raw["transport_norm2"], expected_norm2, rtol=0.0, atol=1e-30)
            rebuilt = np.empty((24, 24), dtype=np.float64)
            assert prepare.build_native_transport_fine(v, c[0], masses, 8, 0, rebuilt) == 0.0
        finally:
            _close(raw)

        kernel = td / "kernel.tbl"
        kernel.write_text("GPUMDJA_KERNEL 1\nU 100\ndegree 0\nP_rank 1\nQ_rank 1\nP_error 0\nQ_error 0\nP_S2 0\nQ_S2 0\nP_values\n1\nP_vectors\n1\nQ_values\n1\nQ_vectors\n1\nEND\n", encoding="ascii")
        output = td / "prepared.rpmdja"
        prepare._prepare(compacted, kernel, output, False, 1e-8, 1e-2)
        assert output.exists()

        data = bytearray(compacted.read_bytes())
        footer = np.frombuffer(data[-24 * 8:], dtype="<f8").copy()
        footer[21] = 1e-13
        false_norm = td / "cancellation-false-norm.qraw"
        false_norm.write_bytes(data[:-24 * 8] + footer.astype("<f8").tobytes())
        try:
            prepare._prepare(false_norm, kernel, td / "false-norm.rpmdja", False, 1e-8, 1e-2)
        except ValueError as exc:
            assert "norm" in str(exc).lower()
        else:
            raise AssertionError("v4 prepare accepted a nonzero footer norm for zero H")


if __name__ == "__main__":
    test_v3_compact_cli_preserves_matrices_and_recomputes_native_h_statistics()
    test_v3_compactor_rejects_bad_diagnostics_coarse_v_and_truncated_source()
    test_v3_compactor_never_overwrites_destination()
    test_compactor_uses_sha256_and_reader_default_keeps_fingerprint()
    test_compactor_rejects_source_mutation_during_conversion()
    test_v4_cancellation_uses_native_left_to_right_v_sums()
    print("qraw v3-to-v4 compact tests passed")
