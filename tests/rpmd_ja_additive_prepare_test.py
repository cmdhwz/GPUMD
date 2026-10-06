import importlib.util
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qprep_additive", ROOT / "tools" / "rpmd_ja_qnep_prepare.py")
qprep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qprep)


def test_additive_assembly_contract_is_available_from_prepare_module():
    fitter_spec = importlib.util.spec_from_file_location("fit_additive", ROOT / "tools" / "rpmd_ja_fit_reference.py")
    fitter = importlib.util.module_from_spec(fitter_spec)
    fitter_spec.loader.exec_module(fitter)
    n = 3
    b = np.eye(6) * 2.0
    b[0, 4] = b[4, 0] = 0.5
    pack = {
        "n": n, "beads": 8, "source_fingerprint": 123, "temperature": 300.0,
        "epsilon": 1e-6, "response_max": 0.01, "response_tolerance": 0.02,
        "training_frames": 4, "validation_frames": 2,
        "sites": [dict(i=i, neighbors=[((i+1) % n, (0,0,0)), ((i-1) % n, (0,0,0))],
                       ell=np.zeros((2,3)), B=b.copy())
                  for i in range(n)],
    }
    with tempfile.TemporaryDirectory() as name:
        pack_path = Path(name) / "fixture.gpjaadd"
        fitter.write_additive(pack_path, pack)
        loaded = fitter.read_additive(pack_path)
        result = qprep.prepare_additive(loaded, np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]), np.eye(3)*10)
    assert result["Kadd"].shape == (9, 9)
    assert result["Hadd"].shape == (3, 9, 9)
    np.testing.assert_allclose(result["Kadd"], result["Kadd"].T, atol=1e-14)
    assert np.linalg.norm(result["Kadd"]) > 0
    assert np.linalg.norm(result["Hadd"][0] - result["Hadd"][0].T) > 0
    np.testing.assert_allclose(result["linear_gradient"], 0.0, atol=1e-14)


def _read_tiles(stream, d):
    tile, count = struct.unpack("<iQ", stream.read(12))
    assert tile == qprep.TILE
    matrix = np.zeros((d, d))
    for _ in range(count):
        row, col, rows, cols, rank, nl, nr = struct.unpack("<5iQQ", stream.read(36))
        left = np.frombuffer(stream.read(nl * 8), dtype="<f8")
        right = np.frombuffer(stream.read(nr * 8), dtype="<f8")
        block = left.reshape(rows, rank) @ right.reshape(rank, cols) if rank else left.reshape(rows, cols)
        matrix[row:row+rows, col:col+cols] = block
    return matrix


def _read_prepared_v3(path, n):
    d = 3*n
    with Path(path).open("rb") as f:
        assert f.read(8) == qprep.MAGIC
        version, endian, atoms, temperature, fd_step, _ = struct.unpack("<IIiddQ", f.read(36))
        assert (version, endian, atoms) == (3, qprep.ENDIAN, n)
        f.read(len(qprep.UNITS) + len(qprep.LAYOUT) + 72 + 12 + n*4 + n*8 + d*8)
        config, charge, pppm, spacing, policy_len = struct.unpack("<Qiidi", f.read(28))
        policy = f.read(policy_len).decode("ascii")
        f.read(21*8 + 2*8 + 2*8 + 2*8)
        degree, prank, qrank = struct.unpack("<iii", f.read(12))
        f.read((prank + qrank + (degree+1)*prank + (degree+1)*qrank)*8)
        matrices = [_read_tiles(f, d)] + [_read_tiles(f, d) for _ in range(3)]
    return {"policy": policy, "matrices": matrices}


def _fixture(directory):
    fitter_spec = importlib.util.spec_from_file_location("fit_additive_fixture", ROOT / "tools" / "rpmd_ja_fit_reference.py")
    fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
    n, d = 4, 12
    masses = np.array([1., 2., 3., 5.])
    lap = np.full((n, n), -1.); np.fill_diagonal(lap, n - 1)
    hessian = np.kron(np.eye(3), lap)
    zeros_v = np.zeros((d, n)); zeros_c = np.zeros((3, d, d))
    cell = np.zeros(18); cell[:9] = np.array([[10.,.4,0.],[0.,11.,.3],[0.,0.,12.]]).ravel()
    raw, kernel, pack_path, output = (directory / name for name in ("input.qraw", "kernel.txt", "additive.gpjaadd", "out.rpmdja"))
    with raw.open("wb") as f:
        f.write(qprep.RAW_MAGIC); f.write(struct.pack("<IIii", 3, qprep.ENDIAN, n, d))
        f.write(struct.pack("<ddQQ", 300., .01, 12345, 98765)); f.write(struct.pack("<iid", 1, 1, 1.))
        f.write(qprep.RAW_LAYOUT); f.write(cell.astype("<f8").tobytes()); f.write(np.ones(3, dtype="<i4").tobytes())
        f.write(np.zeros(n, dtype="<i4").tobytes()); f.write(masses.astype("<f8").tobytes())
        positions = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        f.write(positions.T.astype("<f8").tobytes()); f.write(struct.pack("<d", 0.))
        f.write(np.zeros(n, dtype="<f8").tobytes()); f.write(np.zeros(d, dtype="<f8").tobytes())
        f.write(np.zeros(9*n, dtype="<f8").tobytes())
        f.write(zeros_v.astype("<f8").tobytes()); f.write(zeros_c.astype("<f8").tobytes())
        f.write(hessian.T.astype("<f8").tobytes())
        f.write(zeros_v.astype("<f8").tobytes()); f.write(zeros_c.astype("<f8").tobytes())
        f.write(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.01,3], dtype="<f8").tobytes())
    kernel.write_text("GPUMDJA_KERNEL 1\nU 100\ndegree 0\nP_rank 1\nQ_rank 1\nP_error 0\nQ_error 0\nP_S2 0\nQ_S2 0\nP_values\n1\nP_vectors\n1\nQ_values\n1\nQ_vectors\n1\nEND\n", encoding="ascii")
    b = np.eye(6) * 2.; b[0, 4] = b[4, 0] = .5; b[0, 5] = b[5, 0] = .25
    pack = {
        "n": n, "beads": 8, "source_fingerprint": qprep._fnv64(raw), "temperature": 300.,
        "epsilon": 1e-6, "response_max": .01, "response_tolerance": .02,
        "training_frames": 4, "validation_frames": 2,
        "sites": [dict(i=i, neighbors=[((i+1)%n,((0,1,0) if i==3 else (0,0,0))),((i-1)%n,((0,-1,0) if i==0 else (0,0,0)))], ell=np.array([[.2,0.,0.],[-.2,0.,0.]]), B=b.copy())
                  for i in range(n)],
    }
    fitter.write_additive(pack_path, pack)
    return raw, kernel, pack_path, output, pack, positions, np.diag(cell[:9].reshape(3,3))


def _write_large_canceling_gradient(raw, pack_path, pack, fitter, *, cancel):
    meta = qprep._read_raw(raw)
    v = np.zeros((meta["d"], meta["n"]))
    v[0, 0] = 1e200
    if cancel:
        v[1, 0] = -1e200
    with raw.open("r+b") as f:
        f.seek(meta["start"]); f.write(v.astype("<f8").tobytes())
        for alpha in range(3):
            c = np.zeros((meta["d"], meta["d"]))
            fe = -v.sum(axis=1)
            for cidx in range(meta["d"]):
                atom, axis = cidx % meta["n"], cidx // meta["n"]
                if axis == alpha:
                    c[cidx] = v[:, atom]
                    selected = np.arange(meta["d"]) % meta["n"] == atom
                    c[cidx, selected] += fe[selected]
            f.seek(meta["start"] + meta["vb"] + alpha*meta["d"]*meta["d"]*8)
            f.write(c.astype("<f8").tobytes())
    pack["source_fingerprint"] = qprep._fnv64(raw)
    pack_path.unlink()
    fitter.write_additive(pack_path, pack)


def test_gradient_gate_uses_stable_norm_and_accepts_large_cancellation():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        for cancel in (False, True):
            case = root / str(cancel); case.mkdir()
            raw, kernel, pack_path, output, pack, *_ = _fixture(case)
            fitter_spec = importlib.util.spec_from_file_location("fit_large_gradient", ROOT / "tools" / "rpmd_ja_fit_reference.py")
            fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
            if cancel:
                pack["sites"][0]["ell"][0] = [1e200, 0., 0.]
            _write_large_canceling_gradient(raw, pack_path, pack, fitter, cancel=cancel)
            if cancel:
                qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2, pack_path)
                assert output.exists()
            else:
                try:
                    qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2, pack_path)
                except ValueError as exc:
                    assert "does not cancel" in str(exc)
                    import gc; gc.collect()
                else:
                    raise AssertionError("uncancelled large raw gradient was accepted")
                assert not output.exists()
                assert not Path(str(output)+".stability").exists()
                assert not Path(str(output)+".work").exists()


def test_additive_assembly_rejects_overflow_from_finite_geometry():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        raw, kernel, pack_path, output, pack, *_ = _fixture(root)
        fitter_spec = importlib.util.spec_from_file_location("fit_overflow_geometry", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        meta = qprep._read_raw(raw)
        positions = np.array([[-1e308,0.,0.],[1e308,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        with raw.open("r+b") as f:
            f.seek(8+16+32+16+len(qprep.RAW_LAYOUT)+18*8+12+meta["n"]*4+meta["n"]*8)
            f.write(positions.T.astype("<f8").tobytes())
        pack["source_fingerprint"] = qprep._fnv64(raw)
        pack_path.unlink()
        fitter.write_additive(pack_path, pack)
        try:
            qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2, pack_path)
        except ValueError as exc:
            assert "non-finite K, H, or gradient" in str(exc)
        else:
            raise AssertionError("finite input geometry overflow was accepted")
        assert not output.exists()
        assert not Path(str(output)+".stability").exists()
        assert not Path(str(output)+".work").exists()


def test_symmetric_multitile_writer_changes_additive_certificate_matrix():
    n = 86; d = 3*n
    translation = np.ones(n)/np.sqrt(n)
    basis = np.zeros((d,3))
    for axis in range(3): basis[axis*n:(axis+1)*n,axis] = translation
    matrix = 1.01*(np.eye(d)-basis@basis.T)
    modes = []
    for first in (0,127,255):
        mode = np.zeros(d); mode[first:first+2] = [1/np.sqrt(2),-1/np.sqrt(2)]
        modes.append(mode)
    for i in range(3):
        for j in range(i+1,3): matrix -= .55*np.outer(modes[i],modes[j])
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        stored = np.memmap(root/"stored.f64",dtype="<f8",mode="w+",shape=(d,d))
        error, _ = qprep._write_tiles(root/"matrix.tiles",matrix,True,1e-8,True,stored)
        assert error == 0
        with (root/"matrix.tiles").open("rb") as f:
            tile, count = struct.unpack("<iQ",f.read(12))
        assert tile == qprep.TILE and count == 9
        np.testing.assert_array_equal(stored[128:256,:128],stored[:128,128:256].T)
        np.testing.assert_array_equal(stored[256:,128:256],stored[128:256,256:].T)
        raw_internal = qprep._internal_dynamical(matrix,translation)
        raw_shifted = raw_internal-np.eye(d-3)
        raw_factor = np.linalg.cholesky(raw_shifted)
        assert np.linalg.norm(raw_shifted-raw_factor@raw_factor.T,"fro") < 1.
        stored_internal = qprep._internal_dynamical(stored,translation)
        try:
            np.linalg.cholesky(stored_internal-np.eye(d-3))
        except np.linalg.LinAlgError:
            pass
        else:
            raise AssertionError("lossless symmetric tile matrix unexpectedly passed the raw-matrix certificate")
        stored._mmap.close()


def test_prepare_additive_certificate_rejects_serialized_matrix():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        raw,kernel,pack_path,output,*_ = _fixture(root)
        n=4;d=3*n;masses=np.array([1.,2.,3.,5.])
        translation=np.sqrt(masses/masses.sum())
        basis=np.zeros((d,3))
        for axis in range(3):basis[axis*n:(axis+1)*n,axis]=translation
        stored_bad=np.eye(d)-basis@basis.T
        mode=np.zeros(d);mode[0:2]=[translation[1],-translation[0]]
        mode/=np.linalg.norm(mode)
        stored_bad-=2*np.outer(mode,mode)
        try:
            np.linalg.cholesky(qprep._internal_dynamical(stored_bad,translation)-.5e-6*np.eye(d-3))
        except np.linalg.LinAlgError:
            pass
        else:
            raise AssertionError("injected serialized matrix should fail the shifted certificate")
        original=qprep._write_tiles;injected=[]
        def inject_serialized_matrix(path,matrix,symmetric,tol,lossless,reconstructed=None):
            result=original(path,matrix,symmetric,tol,lossless,reconstructed)
            if reconstructed is not None and symmetric and not injected:
                reconstructed[:]=stored_bad
                reconstructed.flush()
                injected.append(True)
            return result
        qprep._write_tiles=inject_serialized_matrix
        try:
            try:
                qprep._prepare(raw,kernel,output,False,1e-8,1e-2,pack_path)
            except np.linalg.LinAlgError:
                import gc; gc.collect()
            else:
                raise AssertionError("prepare certified its dense matrix instead of the serialized matrix")
        finally:
            qprep._write_tiles=original
        assert injected
        assert qprep._translation_residual(stored_bad,translation)<=1e-8
        assert not output.exists()
        assert not Path(str(output)+".stability").exists()
        assert not Path(str(output)+".work").exists()


def test_cli_additive_prepare_accumulates_actual_v3_k_and_h_and_certificate():
    with tempfile.TemporaryDirectory() as name:
        raw, kernel, pack_path, output, pack, positions, _ = _fixture(Path(name))
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "rpmd_ja_qnep_prepare.py"),
                              str(raw), str(pack_path), "--kernel-table", str(kernel), "--output", str(output)],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        fitter_spec = importlib.util.spec_from_file_location("fit_additive_expected", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        assembled = fitter.assemble_additive(pack, positions, np.array([[10.,.4,0.],[0.,11.,.3],[0.,0.,12.]]))
        d, n = 12, 4
        with output.open("rb") as f:
            assert f.read(8) == qprep.MAGIC
            version, endian, atoms, temperature, fd_step, _ = struct.unpack("<IIiddQ", f.read(36))
            assert (version, endian, atoms, temperature, fd_step) == (3, qprep.ENDIAN, n, 300., .01)
            f.read(len(qprep.UNITS) + len(qprep.LAYOUT) + 72 + 12 + n*4 + n*8 + d*8)
            config, charge, pppm, spacing, policy_len = struct.unpack("<Qiidi", f.read(28))
            policy = f.read(policy_len).decode("ascii")
            assert policy == "native_reference_transport;finite_temperature_additive_v1;beads=8;derivative=3"
            f.read(21*8 + 2*8 + 2*8 + 2*8)
            degree, prank, qrank = struct.unpack("<iii", f.read(12))
            assert (degree, prank, qrank) == (0,1,1)
            f.read((prank + qrank + (degree+1)*prank + (degree+1)*qrank)*8)
            got_d = _read_tiles(f, d)
            got_h = [_read_tiles(f, d) for _ in range(3)]
        baseline = np.kron(np.eye(3), np.full((n,n), -1.))
        np.fill_diagonal(baseline, 3.)
        total = baseline + assembled["Kadd"]
        mass = np.tile(np.array([1.,2.,3.,5.]),3)
        expected_d = total / np.sqrt(mass[:,None]*mass[None,:])
        np.testing.assert_allclose(got_d, expected_d, atol=2e-12, rtol=2e-12)
        for alpha in range(3):
            np.testing.assert_allclose(got_h[alpha], assembled["Hadd"][alpha].T / np.sqrt(mass[:,None]*mass[None,:]), atol=2e-12, rtol=2e-12)
        sidecar = Path(str(output)+".stability").read_text(encoding="ascii")
        assert "GPUMDJA_QNEP_STABILITY 4\n" in sidecar
        assert "certificate additive_shifted_frobenius_v1\n" in sidecar
        assert "epsilon_num 9.9999999999999995e-07\n" in sidecar
        assert "beads=8;derivative=3" in sidecar


def test_internal_mass_com_v2_pulls_back_nonzero_net_and_preserves_raw_transport():
    with tempfile.TemporaryDirectory() as name:
        raw, kernel, pack_path, output, pack, positions, _ = _fixture(Path(name))
        fitter_spec = importlib.util.spec_from_file_location("fit_additive_internal", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        pack["source_fingerprint"] = qprep._fnv64(raw)
        pack["internal_mass_com"] = True
        pack_path.unlink()
        fitter.write_additive(pack_path, pack)
        meta = qprep._read_raw(raw)
        raw_v = np.zeros((meta["d"], meta["n"]))
        assembled = fitter.assemble_additive(pack, positions, np.array([[10.,.4,0.],[0.,11.,.3],[0.,0.,12.]]))
        target = -assembled["linear_gradient"].copy()
        masses = meta["masses"]
        target += masses[:,None] / masses.sum() * np.array([.6,-.3,1.2])[None,:]
        graw = target.T.reshape(-1)
        for coordinate, value in enumerate(graw): raw_v[coordinate, coordinate % meta["n"]] = value
        raw_c = []
        for alpha in range(3):
            c = np.arange(meta["d"]**2, dtype=float).reshape(meta["d"],meta["d"])*(.001*(alpha+1))
            c[0,1] += .37
            raw_c.append(c)
        with raw.open("r+b") as f:
            f.seek(meta["start"]); f.write(raw_v.astype("<f8").tobytes())
            f.seek(meta["coarse_v"]); f.write(raw_v.astype("<f8").tobytes())
            for alpha, c in enumerate(raw_c):
                f.seek(meta["start"] + meta["vb"] + alpha*meta["d"]**2*8)
                f.write(c.astype("<f8").tobytes())
                f.seek(meta["coarse_c"] + alpha*meta["d"]**2*8)
                f.write(c.astype("<f8").tobytes())
        pack["source_fingerprint"] = qprep._fnv64(raw)
        pack_path.unlink(); fitter.write_additive(pack_path, pack)
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "rpmd_ja_qnep_prepare.py"),
                                 str(raw), str(pack_path), "--kernel-table", str(kernel), "--output", str(output)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        parsed = _read_prepared_v3(output, meta["n"])
        assert parsed["policy"] == "native_reference_transport;internal_mass_com_pullback_v1;finite_temperature_additive_v1;beads=8;derivative=3"
        d, n = meta["d"], meta["n"]
        mass_scale = np.sqrt(np.tile(masses,3)[:,None]*np.tile(masses,3)[None,:])
        translation = np.zeros((d,3))
        for axis in range(3): translation[axis*n:(axis+1)*n,axis] = np.sqrt(masses/masses.sum())
        internal = np.linalg.svd(translation.T, full_matrices=True)[2][3:].T
        rng = np.random.default_rng(219)
        q = rng.normal(size=d); v = rng.normal(size=d)
        q_internal = internal.T @ q; v_internal = internal.T @ v
        q_projected = internal @ q_internal; v_projected = internal @ v_internal
        asymmetry = 0.0
        for alpha in range(3):
            h_internal = parsed["matrices"][alpha+1]
            raw_h = raw_c[alpha].T / mass_scale
            additive_h = assembled["Hadd"][alpha].T / mass_scale
            np.testing.assert_allclose(h_internal, raw_h + additive_h, atol=2e-12, rtol=2e-12)
            raw_h_actual = h_internal - additive_h
            np.testing.assert_allclose(raw_h_actual, raw_h, atol=2e-12)
            reduced = internal.T @ h_internal @ internal
            np.testing.assert_allclose(v_projected @ h_internal @ q_projected,
                                       v_internal @ reduced @ q_internal, atol=2e-12, rtol=2e-12)
            asymmetry = max(asymmetry, np.linalg.norm(h_internal-h_internal.T))
        assert asymmetry > 0
        old_output = Path(name) / "out_v1.rpmdja"
        with pack_path.open("r+b") as f:
            f.seek(8); f.write(struct.pack("<I", 1))
        old = subprocess.run([sys.executable, str(ROOT / "tools" / "rpmd_ja_qnep_prepare.py"),
                              str(raw), str(pack_path), "--kernel-table", str(kernel), "--output", str(old_output)],
                             capture_output=True, text=True)
        assert old.returncode != 0 and "does not cancel raw reference gradient" in old.stderr
        assert not old_output.exists()
        raw_fenergy = -raw_v.sum(axis=1)
        internal_gradient = target - masses[:,None] / masses.sum() * target.sum(axis=0)[None,:]
        internal_fenergy = -internal_gradient.T.reshape(-1)
        for alpha in range(3):
            raw_formula = np.zeros((d,d)); projected_formula = np.zeros((d,d))
            for cidx in range(d):
                atom, axis = cidx % n, cidx // n
                if axis != alpha: continue
                selected = np.arange(d) % n == atom
                raw_formula[cidx] = -raw_v[:,atom]
                raw_formula[cidx,selected] -= raw_fenergy[selected]
                projected_formula[cidx] = -raw_v[:,atom]
                projected_formula[cidx,selected] -= internal_fenergy[selected]
            np.testing.assert_allclose(raw_formula, 0.0, atol=1e-14)
            assert np.linalg.norm(projected_formula) > 0


def test_additive_prepare_rejects_wrong_source_and_truncated_pack():
    with tempfile.TemporaryDirectory() as name:
        raw, kernel, pack_path, output, pack, _, _ = _fixture(Path(name))
        fitter_spec = importlib.util.spec_from_file_location("fit_additive_bad", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        bad = dict(pack); bad["source_fingerprint"] ^= 1
        mismatch = Path(name)/"mismatch.gpjaadd"; fitter.write_additive(mismatch,bad)
        try:
            qprep._prepare(raw,kernel,output,False,1e-8,1e-2,mismatch)
        except ValueError as exc:
            assert "source fingerprint" in str(exc)
        else:
            raise AssertionError("mismatched source package was accepted")
        truncated = Path(name)/"truncated.gpjaadd"; truncated.write_bytes(pack_path.read_bytes()[:-1])
        try:
            qprep._prepare(raw,kernel,output,False,1e-8,1e-2,truncated)
        except ValueError as exc:
            assert "truncated" in str(exc)
        else:
            raise AssertionError("truncated additive package was accepted")


def test_raw_versions_and_version_specific_diagnostic_gates():
    def rewrite_footer(raw, version, values):
        meta = qprep._read_raw(raw)
        values = np.asarray(values, dtype="<f8")
        values[17] = version
        with raw.open("r+b") as f:
            f.seek(8); f.write(struct.pack("<I", version))
            f.seek(meta["footer"]); f.write(values.tobytes())

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        for version in (1, 2, 3):
            sub = root / f"v{version}"; sub.mkdir()
            raw, kernel, _pack, output, *_ = _fixture(sub)
            values = np.zeros(18); values[1] = 1e-5; values[16] = .01
            values[0] = .9; values[2] = .9 if version >= 2 else .001
            if version >= 2: values[3] = 1e-5
            rewrite_footer(raw, version, values)
            qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2)
            assert output.exists()
        for version, index, value in ((1,2,.051),(2,3,1.1e-4),(3,8,.051)):
            sub = root / f"bad{version}"; sub.mkdir()
            raw, kernel, _pack, output, *_ = _fixture(sub)
            values = np.zeros(18); values[1] = 1e-5; values[16] = .01
            if version >= 2: values[3] = 1e-5
            values[index] = value
            rewrite_footer(raw, version, values)
            try:
                qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2)
            except ValueError as exc:
                assert "finite-difference checks" in str(exc)
            else:
                raise AssertionError(f"raw v{version} accepted invalid diagnostic at {index}")


def test_householder_additive_and_column_major_full_certificate_oracles():
    n = 3; d = 3*n; masses = np.array([1., 2., 5.]); mt = np.tile(masses, 3)
    vectors = []
    for axis in range(3):
        w = np.zeros(d); w[axis*n:(axis+1)*n] = np.sqrt(masses/masses.sum()); w[axis*n] -= 1.
        vectors.append(w / np.linalg.norm(w))

    def rotate_and_project(a):
        out = np.array(a, dtype=float, copy=True)
        for w in vectors:
            out = (np.eye(d)-2*np.outer(w,w)) @ out @ (np.eye(d)-2*np.outer(w,w))
        removed = [0,n,2*n]; out[removed,:] = 0.; out[:,removed] = 0.
        return out

    kadd = np.zeros((d,d));
    for i in range(n):
        j=(i+1)%n
        for a in range(3):
            kadd[a*n+i,a*n+i] += 2.; kadd[a*n+j,a*n+j] += 2.
            kadd[a*n+i,a*n+j] -= 2.; kadd[a*n+j,a*n+i] -= 2.
    kadd[0, n+1] = kadd[n+1, 0] = .1
    kadd[0, 0] -= .2; kadd[n+1,n+1] -= .2
    mass_k = kadd / np.sqrt(mt[:,None]*mt[None,:])
    base = np.diag(np.arange(1,d+1,dtype=float))
    baseline_rotated = rotate_and_project(base)
    additive_rotated = rotate_and_project(mass_k)
    combined = rotate_and_project(base + mass_k)
    np.testing.assert_allclose(combined, baseline_rotated + additive_rotated, atol=2e-13)
    assert np.linalg.norm(combined - (baseline_rotated + mass_k)) > 1e-2

    lower = np.array([[1.,0.,0.],[.2,.7,0.],[-.1,.3,.5]])
    near_soft = lower @ lower.T
    near_soft[0,0] += 5e-7
    eps = 1e-6
    shifted = near_soft - .5*eps*np.eye(3)
    factor = np.linalg.cholesky(shifted)
    storage = near_soft.astype("<f8").ravel(order="C").copy()
    for row in range(3):
        for col in range(3):
            if row >= col: storage[row + col*3] = factor[row,col]
    polluted = storage.reshape((3,3), order="F")
    assert np.linalg.norm(polluted @ polluted.T - shifted, "fro") > 1e-2
    for row in range(3):
        for col in range(3):
            if row < col: storage[row + col*3] = 0.
    clean_factor = storage.reshape((3,3), order="F")
    np.testing.assert_allclose(clean_factor, factor, atol=0., rtol=0.)
    assert np.linalg.norm(clean_factor @ clean_factor.T - shifted, "fro") < eps/2


def test_failed_prepare_closes_temporary_mappings_before_cleanup():
    with tempfile.TemporaryDirectory() as name:
        raw, kernel, _pack, output, *_ = _fixture(Path(name))
        meta = qprep._read_raw(raw); kstart = meta["start"] + meta["vb"] + meta["cb"]
        with raw.open("r+b") as f:
            f.seek(kstart + 1*8); f.write(struct.pack("<d", 100.0))
        try:
            qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2)
        except ValueError as exc:
            assert "antisymmetry" in str(exc)
        else:
            raise AssertionError("asymmetric raw Hessian was accepted")
        assert not output.exists()
        assert not Path(str(output)+".stability").exists()
        assert not Path(str(output)+".work").exists()



def test_failed_prepare_closes_first_h_mapping_if_second_open_fails():
    with tempfile.TemporaryDirectory() as name:
        raw, kernel, _pack, output, *_ = _fixture(Path(name))
        original = qprep.np.memmap; opened = []
        def fail_second_h(path, *args, **kwargs):
            if Path(path).name == "Btc0.f64":
                raise OSError("injected second H mapping failure")
            mapped = original(path, *args, **kwargs)
            if Path(path).name == "Bt0.f64": opened.append(mapped)
            return mapped
        qprep.np.memmap = fail_second_h
        try:
            try:
                qprep._prepare(raw, kernel, output, False, 1e-8, 1e-2)
            except OSError as exc:
                assert "injected" in str(exc)
            else:
                raise AssertionError("injected H mapping failure did not propagate")
        finally:
            qprep.np.memmap = original
        assert len(opened) == 1 and opened[0]._mmap.closed
        assert not Path(str(output)+".work").exists()


if __name__ == "__main__":
    test_additive_assembly_contract_is_available_from_prepare_module()
    test_cli_additive_prepare_accumulates_actual_v3_k_and_h_and_certificate()
    test_internal_mass_com_v2_pulls_back_nonzero_net_and_preserves_raw_transport()
    test_additive_prepare_rejects_wrong_source_and_truncated_pack()
    test_gradient_gate_uses_stable_norm_and_accepts_large_cancellation()
    test_additive_assembly_rejects_overflow_from_finite_geometry()
    test_symmetric_multitile_writer_changes_additive_certificate_matrix()
    test_prepare_additive_certificate_rejects_serialized_matrix()
    test_raw_versions_and_version_specific_diagnostic_gates()
    test_householder_additive_and_column_major_full_certificate_oracles()
    test_failed_prepare_closes_temporary_mappings_before_cleanup()
    test_failed_prepare_closes_first_h_mapping_if_second_open_fails()
    print("qNEP RPMD-JA additive prepare tests passed")
