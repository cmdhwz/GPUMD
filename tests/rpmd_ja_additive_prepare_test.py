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


def _fixture(directory, *, atomic_additive_write=True):
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
    if atomic_additive_write:
        fitter.write_additive(pack_path, pack)
    else:
        with pack_path.open("wb") as stream:
            fitter._write_additive_stream(stream, pack)
    return raw, kernel, pack_path, output, pack, positions, np.diag(cell[:9].reshape(3,3))


def test_sparse_additive_v5_v6_roundtrip_and_dense_legacy_compatibility():
    with tempfile.TemporaryDirectory() as name:
        raw, _kernel, pack_path, _output, pack, _positions, _cell = _fixture(Path(name))
        fitter_spec = importlib.util.spec_from_file_location("fit_sparse_compat", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        pack["source_fingerprint"] = qprep._fnv64(raw)
        pack["sites"][0]["B"][0, 0] = -0.0

        def legacy_package(path, version):
            with path.open("wb") as stream:
                stream.write(fitter.ADD_MAGIC)
                stream.write(struct.pack("<IIiiQ",version,fitter.ENDIAN,pack["n"],pack["beads"],pack["source_fingerprint"]))
                legacy_response=min(float(pack["response_max"]),float(pack["response_tolerance"]))
                stream.write(struct.pack("<ddddQQ",pack["temperature"],pack["epsilon"],legacy_response,
                                         pack["response_tolerance"],pack["training_frames"],pack["validation_frames"]))
                for site in pack["sites"]:
                    z=len(site["neighbors"]); stream.write(struct.pack("<i",z))
                    for atom,image in site["neighbors"]: stream.write(struct.pack("<4i",atom,*image))
                    np.asarray(site["ell"],dtype="<f8").tofile(stream)
                    np.asarray(site["B"],dtype="<f8").tofile(stream)

        pack_path.unlink()
        fitter.write_additive(pack_path,pack)
        assert struct.unpack_from("<I",pack_path.read_bytes(),8)[0] == 5
        dense_size=80+sum(4+16*len(site["neighbors"])+24*len(site["neighbors"])+8*(3*len(site["neighbors"]))**2
                           for site in pack["sites"])
        assert pack_path.stat().st_size < dense_size
        loaded=fitter.read_additive(pack_path)
        assert not loaded["internal_mass_com"]
        assert np.signbit(loaded["sites"][0]["B"][0,0])
        assert qprep.read_additive(pack_path,expected_n=pack["n"])["internal_mass_com"] is False

        package_bytes=pack_path.read_bytes()
        first_entry=80+4+16*len(pack["sites"][0]["neighbors"])+24*len(pack["sites"][0]["neighbors"])+4+8
        for suffix,mutate in (
            ("bad_index",lambda data: struct.pack_into("<i",data,first_entry,3*len(pack["sites"][0]["neighbors"]))),
            ("positive_zero",lambda data: struct.pack_into("<d",data,first_entry+8,0.0)),
        ):
            damaged=bytearray(package_bytes); mutate(damaged)
            bad_path=Path(name)/(suffix+".gpjaadd"); bad_path.write_bytes(damaged)
            try:
                qprep.read_additive(bad_path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"sparse additive {suffix} was accepted")
        truncated=Path(name)/"truncated_sparse.gpjaadd"; truncated.write_bytes(package_bytes[:-1])
        try:
            qprep.read_additive(truncated)
        except ValueError as exc:
            assert "truncated" in str(exc)
        else:
            raise AssertionError("truncated sparse additive package was accepted")

        pack["internal_mass_com"] = True
        pack_path.unlink(); fitter.write_additive(pack_path,pack)
        assert struct.unpack_from("<I",pack_path.read_bytes(),8)[0] == 6
        com=fitter.read_additive(pack_path)
        assert com["internal_mass_com"] and np.signbit(com["sites"][0]["B"][0,0])

        pack["diagnostic"] = True
        pack["response_status"] = "FAIL"
        pack["response_max"] = 0.5
        pack_path.unlink(); fitter.write_additive(pack_path,pack)
        assert struct.unpack_from("<I",pack_path.read_bytes(),8)[0] == 7
        diagnostic=fitter.read_additive(pack_path)
        assert diagnostic["internal_mass_com"] and diagnostic["diagnostic"]
        assert diagnostic["response_status"] == "FAIL" and diagnostic["response_max"] > diagnostic["response_tolerance"]
        prepared_additive=qprep.read_additive(pack_path,expected_n=pack["n"])
        assert prepared_additive["diagnostic"] and prepared_additive["response_status"] == "FAIL"

        for version in (1,2):
            old_path=Path(name)/f"legacy_v{version}.gpjaadd"; legacy_package(old_path,version)
            old=fitter.read_additive(old_path)
            assert old["internal_mass_com"] == (version==2)
            np.testing.assert_array_equal(old["sites"][0]["B"],pack["sites"][0]["B"])


def test_hand_authored_sparse_package_readers_without_hardlink_writer():
    fitter_spec = importlib.util.spec_from_file_location("fit_sparse_hand_authored", ROOT / "tools" / "rpmd_ja_fit_reference.py")
    fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
    matrices = [np.diag([-0.0, 2.0, 3.0]), np.diag([1.0, 4.0, 5.0])]
    matrices[0][0, 1] = matrices[0][1, 0] = 0.25

    def package_bytes(version):
        data = bytearray(fitter.ADD_MAGIC)
        data.extend(struct.pack("<IIiiQ",version,fitter.ENDIAN,2,8,1234))
        data.extend(struct.pack("<ddddQQ",300.0,0.01,0.0,0.1,4,2))
        for center in range(2):
            neighbor = 1-center
            data.extend(struct.pack("<i4i",1,neighbor,0,0,0))
            data.extend(np.zeros(3,dtype="<f8").tobytes())
            values = matrices[center].reshape(-1)
            entries = [(index//3,index%3,float(value)) for index,value in enumerate(values)
                       if value != 0.0 or np.signbit(value)]
            data.extend(struct.pack("<Q",len(entries)))
            for row,column,value in entries:
                data.extend(struct.pack("<iid",row,column,value))
        return data

    with tempfile.TemporaryDirectory() as name:
        for version in (3,4):
            path=Path(name)/f"hand_v{version}.gpjaadd"; path.write_bytes(package_bytes(version))
            for reader in (fitter.read_additive,qprep.read_additive):
                loaded=reader(path)
                assert loaded["internal_mass_com"] == (version==4)
                np.testing.assert_array_equal(loaded["sites"][0]["B"],matrices[0])
                assert np.signbit(loaded["sites"][0]["B"][0,0])

        valid=package_bytes(3)
        first_entry=80+4+16+24+8
        for suffix,mutate in (
            ("index",lambda data: struct.pack_into("<i",data,first_entry,3)),
            ("positive_zero",lambda data: struct.pack_into("<d",data,first_entry+8,0.0)),
        ):
            damaged=bytearray(valid); mutate(damaged); path=Path(name)/(suffix+".gpjaadd"); path.write_bytes(damaged)
            for reader in (fitter.read_additive,qprep.read_additive):
                try: reader(path)
                except ValueError: pass
                else: raise AssertionError(f"hand-authored sparse {suffix} was accepted")
        bad_version=bytearray(valid); struct.pack_into("<I",bad_version,8,8)
        invalid=Path(name)/"bad_version.gpjaadd"; invalid.write_bytes(bad_version)
        try: qprep.read_additive(invalid)
        except ValueError: pass
        else: raise AssertionError("unsupported hand-authored sparse version was accepted")
        truncated=Path(name)/"truncated.gpjaadd"; truncated.write_bytes(valid[:-1])
        try: qprep.read_additive(truncated)
        except ValueError as exc: assert "truncated" in str(exc)
        else: raise AssertionError("truncated hand-authored sparse package was accepted")

        fitter_spec = importlib.util.spec_from_file_location("fit_v7_status", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        with tempfile.TemporaryDirectory() as v7_dir:
            v7_pack = {
                "n": 2, "beads": 8, "source_fingerprint": 1234, "temperature": 300.0,
                "epsilon": 0.01, "response_max": 0.5, "response_tolerance": 0.02,
                "training_frames": 4, "validation_frames": 2, "diagnostic": True,
                "response_status": "FAIL", "internal_mass_com": True,
                "sites": [
                    dict(i=0, neighbors=[(1, (0, 0, 0))], ell=np.zeros((1, 3)), B=np.eye(3)),
                    dict(i=1, neighbors=[(0, (0, 0, 0))], ell=np.zeros((1, 3)), B=np.eye(3)),
                ],
            }
            package_path = Path(v7_dir) / "hand_v7.gpjaadd"
            with package_path.open("wb") as stream:
                fitter._write_additive_stream(stream, v7_pack)
            package = bytearray(package_path.read_bytes())
            for suffix, offset, value in (("numerical",84,2),("response",88,9)):
                damaged = bytearray(package); struct.pack_into("<I",damaged,offset,value)
                bad_path = Path(v7_dir) / f"bad_{suffix}.gpjaadd"; bad_path.write_bytes(damaged)
                for reader in (fitter.read_additive,qprep.read_additive):
                    try: reader(bad_path)
                    except ValueError: pass
                    else: raise AssertionError(f"invalid v7 {suffix} status was accepted")
            pass_above_budget = bytearray(package); struct.pack_into("<I",pass_above_budget,88,0)
            bad_path = Path(v7_dir) / "bad_pass_budget.gpjaadd"; bad_path.write_bytes(pass_above_budget)
            try: fitter.read_additive(bad_path)
            except ValueError: pass
            else: raise AssertionError("v7 response PASS above its frozen tolerance was accepted")
            for response_code in (2,3):
                for response_value in (float("inf"),float("-inf")):
                    infinite_response=bytearray(package)
                    struct.pack_into("<d",infinite_response,48,response_value)
                    struct.pack_into("<I",infinite_response,88,response_code)
                    bad_path=Path(v7_dir)/f"bad_infinite_response_{response_code}_{'positive' if response_value > 0 else 'negative'}.gpjaadd"
                    bad_path.write_bytes(infinite_response)
                    for reader in (fitter.read_additive,qprep.read_additive):
                        try: reader(bad_path)
                        except ValueError: pass
                        else: raise AssertionError(f"v7 response status {response_code} accepted {response_value}")


def test_prepare_binds_diagnostic_identity_into_v3_policy():
    with tempfile.TemporaryDirectory() as name:
        raw,kernel,pack_path,output,pack,_positions,_cell=_fixture(Path(name),atomic_additive_write=False)
        fitter_spec=importlib.util.spec_from_file_location("fit_diagnostic_identity",ROOT/"tools"/"rpmd_ja_fit_reference.py")
        fitter=importlib.util.module_from_spec(fitter_spec);fitter_spec.loader.exec_module(fitter)
        pack["diagnostic"]=True
        pack["response_status"]="FAIL"
        pack["response_max"]=0.5
        pack["response_tolerance"]=0.02
        pack["internal_mass_com"]=True
        pack["source_fingerprint"]=qprep._fnv64(raw)
        with pack_path.open("wb") as stream:
            fitter._write_additive_stream(stream,pack)
        result=subprocess.run([sys.executable,str(ROOT/"tools/rpmd_ja_qnep_prepare.py"),str(raw),str(pack_path),
                               "--kernel-table",str(kernel),
                               "--output",str(output),"--lossless"],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        prepared=_read_prepared_v3(output,pack["n"])
        assert ";use=DIAGNOSTIC;response=FAIL;numerical=PASS" in prepared["policy"]


def test_additive_modes_and_expansion_limits():
    with tempfile.TemporaryDirectory() as name:
        fitter_spec = importlib.util.spec_from_file_location("fit_additive_modes", ROOT / "tools" / "rpmd_ja_fit_reference.py")
        fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
        pack_path = Path(name)/"modes.gpjaadd"
        n=3
        pack={"n":n,"beads":8,"source_fingerprint":1234,"temperature":300.,"epsilon":0.01,
              "response_max":0.,"response_tolerance":0.1,"training_frames":4,"validation_frames":2,
              "sites":[dict(i=i,neighbors=[((i+1)%n,(0,0,0)),((i-1)%n,(0,0,0))],
                             ell=np.zeros((2,3)),B=np.zeros((6,6))) for i in range(n)]}
        dense = np.arange(1., 37.).reshape(6, 6); dense = dense + dense.T
        dense[0,0] = -0.0
        pack["sites"][0]["B"] = dense
        scalar = np.zeros((6, 6)); scalar[0, 1] = scalar[1, 0] = 0.25; scalar[2, 2] = -0.0
        pack["sites"][1]["B"] = scalar
        block = np.zeros((6, 6)); block[:3, :3] = [[1., 2., 3.], [2., -0.0, 4.], [3., 4., 5.]]
        cross = np.array([[.125,.25,.375],[.5,.625,.75],[.875,1.,1.125]])
        block[:3, 3:] = cross; block[3:, :3] = cross.T
        pack["sites"][2]["B"] = block
        fitter._validate_additive(pack)
        with pack_path.open("wb") as stream:
            fitter._write_additive_stream(stream,pack)
        payload = pack_path.read_bytes()
        assert struct.unpack_from("<I", payload, 8)[0] == 5
        modes = []
        mode_offsets = []
        offset = 80
        for site in pack["sites"]:
            z = len(site["neighbors"]); q = 3*z
            offset += 4 + 16*z + 24*z
            mode_offsets.append(offset)
            mode = struct.unpack_from("<I", payload, offset)[0]; modes.append(mode); offset += 4
            if mode == 0:
                offset += 8*q*q
            elif mode == 1:
                count = struct.unpack_from("<Q", payload, offset)[0]; offset += 8 + 16*count
            else:
                count = struct.unpack_from("<Q", payload, offset)[0]; offset += 8 + 80*count
        assert modes == [0, 1, 2]
        assert offset == len(payload)
        legacy_dense_size = 80 + sum(4+16*len(s["neighbors"])+24*len(s["neighbors"])+8*(3*len(s["neighbors"]))**2
                                     for s in pack["sites"])
        assert len(payload) < legacy_dense_size
        loaded = fitter.read_additive(pack_path)
        for site,expected in zip(loaded["sites"],(dense,scalar,block)):
            np.testing.assert_array_equal(site["B"].view(np.uint64), expected.view(np.uint64))
        assert np.signbit(loaded["sites"][1]["B"][2, 2])
        assert np.signbit(loaded["sites"][0]["B"][0, 0])
        assert np.signbit(loaded["sites"][2]["B"][1, 1])
        assert loaded["sites"][2]["B"][1, 4] == loaded["sites"][2]["B"][4, 1] == 0.625
        prepared = qprep.read_additive(pack_path)
        for site,expected in zip(prepared["sites"],(dense,scalar,block)):
            np.testing.assert_array_equal(site["B"].view(np.uint64), expected.view(np.uint64))

        for suffix, mutate in (
            ("bad_mode", lambda data: struct.pack_into("<I", data, mode_offsets[0], 3)),
            ("bad_block_index", lambda data: struct.pack_into("<i", data,
                mode_offsets[2]+4+8, len(pack["sites"][2]["neighbors"]))),
        ):
            damaged = bytearray(payload); mutate(damaged)
            bad_path = Path(name)/(suffix+".gpjaadd"); bad_path.write_bytes(damaged)
            for reader in (fitter.read_additive, qprep.read_additive):
                try: reader(bad_path)
                except ValueError: pass
                else: raise AssertionError(f"{suffix} additive package was accepted")
        truncated = Path(name)/"truncated_block.gpjaadd"; truncated.write_bytes(payload[:-1])
        for reader in (fitter.read_additive, qprep.read_additive):
            try: reader(truncated)
            except ValueError as exc: assert "truncated" in str(exc)
            else: raise AssertionError("truncated sparse 3x3 package was accepted")

        n = 1000
        malformed = bytearray(fitter.ADD_MAGIC)
        malformed.extend(struct.pack("<IIiiQ", 5, fitter.ENDIAN, n, 8, 1234))
        malformed.extend(struct.pack("<ddddQQ", 300., 0.01, 0., 0.1, 4, 2))
        malformed.extend(struct.pack("<i", 3*n)); malformed.extend(b"\0\0\0\0"*(n-1))
        oversized = Path(name)/"oversized_expansion.gpjaadd"; oversized.write_bytes(malformed)
        for reader in (fitter.read_additive, qprep.read_additive):
            try: reader(oversized)
            except ValueError as exc: assert "expanded local matrices" in str(exc)
            else: raise AssertionError("oversized sparse local matrix was expanded")


def test_additive_writer_rejects_reader_expansion_limit_before_temp_creation():
    fitter_spec = importlib.util.spec_from_file_location("fit_writer_expansion_limit", ROOT / "tools" / "rpmd_ja_fit_reference.py")
    fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
    n=3
    pack={"n":n,"beads":8,"source_fingerprint":1234,"temperature":300.,"epsilon":0.01,
          "response_max":0.,"response_tolerance":0.1,"training_frames":4,"validation_frames":2,
          "sites":[dict(i=i,neighbors=[((i+1)%n,(0,0,0)),((i-1)%n,(0,0,0))],
                         ell=np.zeros((2,3)),B=np.zeros((6,6))) for i in range(n)]}
    with tempfile.TemporaryDirectory() as name:
        destination=Path(name)/"over_limit.gpjaadd"
        old_limit=fitter.MAX_ADDITIVE_DENSE_ELEMENTS
        fitter.MAX_ADDITIVE_DENSE_ELEMENTS=100
        try:
            try: fitter.write_additive(destination,pack)
            except ValueError as exc:
                assert "exceed 100 elements at site 2 (6x6)" in str(exc)
            else: raise AssertionError("additive writer published a package its reader rejects")
        finally:
            fitter.MAX_ADDITIVE_DENSE_ELEMENTS=old_limit
        assert not destination.exists()
        assert not list(Path(name).glob(".over_limit.gpjaadd.*.tmp"))


def test_additive_writer_and_readers_reject_coordination_over_reader_limit():
    fitter_spec = importlib.util.spec_from_file_location("fit_writer_coordination_limit", ROOT / "tools" / "rpmd_ja_fit_reference.py")
    fitter = importlib.util.module_from_spec(fitter_spec); fitter_spec.loader.exec_module(fitter)
    n,z=2,7
    neighbors=[]
    for center in range(n):
        offsets=range(-3,4) if center==0 else range(3,-4,-1)
        neighbors.append([((1-center,(offset,0,0))) for offset in offsets])
    pack={"n":n,"beads":8,"source_fingerprint":1234,"temperature":300.,"epsilon":0.01,
          "response_max":0.,"response_tolerance":0.1,"training_frames":4,"validation_frames":2,
          "sites":[dict(i=i,neighbors=neighbors[i],ell=np.zeros((z,3)),B=np.zeros((3*z,3*z))) for i in range(n)]}
    with tempfile.TemporaryDirectory() as name:
        destination=Path(name)/"over_coordination.gpjaadd"
        try: fitter.write_additive(destination,pack)
        except ValueError as exc: assert "invalid local coordination" in str(exc)
        else: raise AssertionError("additive writer accepted coordination rejected by readers")
        assert not destination.exists()
        assert not list(Path(name).glob(".over_coordination.gpjaadd.*.tmp"))

        malformed=Path(name)/"reader_over_coordination.gpjaadd"
        with malformed.open("wb") as stream:
            stream.write(fitter.ADD_MAGIC)
            stream.write(struct.pack("<IIiiQ",5,fitter.ENDIAN,n,8,1234))
            stream.write(struct.pack("<ddddQQ",300.,0.01,0.,0.1,4,2))
            for edges in neighbors:
                stream.write(struct.pack("<i",z))
                for atom,image in edges: stream.write(struct.pack("<4i",atom,*image))
                stream.write(np.zeros((z,3),dtype="<f8").tobytes())
                stream.write(struct.pack("<I",0))
                stream.write(np.zeros((3*z,3*z),dtype="<f8").tobytes())
        for reader in (fitter.read_additive,qprep.read_additive):
            try: reader(malformed)
            except ValueError as exc: assert "local coordination" in str(exc)
            else: raise AssertionError("reader accepted coordination above 3*n")


def test_prepare_reader_accepts_structurally_valid_package_over_64_mib():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name)/"large.gpjaadd"
        n = 111111
        matrix = np.ones((6, 6),dtype="<f8")
        scalar_records = b"".join(struct.pack("<iid",row,column,float(matrix[row,column]))
                                   for row in range(6) for column in range(6))
        endpoint_records = b"".join(struct.pack("<iid",row,column,1.0) for row in range(3) for column in range(3))
        with path.open("wb") as stream:
            stream.write(qprep._FITTER.ADD_MAGIC)
            stream.write(struct.pack("<IIiiQ",3,qprep.ENDIAN,n,8,1234))
            stream.write(struct.pack("<ddddQQ",300.,0.01,0.,0.1,4,2))
            for site in range(n):
                neighbors = [site+1] if site==0 else [site-1] if site==n-1 else [site-1,site+1]
                stream.write(struct.pack("<i",len(neighbors)))
                for neighbor in neighbors: stream.write(struct.pack("<4i",neighbor,0,0,0))
                stream.write(np.zeros(3*len(neighbors),dtype="<f8").tobytes())
                records = endpoint_records if len(neighbors)==1 else scalar_records
                count = len(records)//16
                stream.write(struct.pack("<Q",count)); stream.write(records)
        assert path.stat().st_size > 64*1024*1024
        original_reader = qprep._FITTER.read_additive
        qprep._FITTER.read_additive = lambda _path: {"preflight": "passed"}
        try:
            assert qprep.read_additive(path) == {"preflight": "passed"}
        finally:
            qprep._FITTER.read_additive = original_reader


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


def test_internal_mass_com_v6_pulls_back_nonzero_net_and_preserves_raw_transport():
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
        old_output = Path(name) / "out_non_com.rpmdja"
        pack["internal_mass_com"] = False
        legacy_path = Path(name) / "legacy_v3_non_com.gpjaadd"
        with legacy_path.open("wb") as stream:
            stream.write(fitter.ADD_MAGIC)
            stream.write(struct.pack("<IIiiQ",3,fitter.ENDIAN,pack["n"],pack["beads"],pack["source_fingerprint"]))
            stream.write(struct.pack("<ddddQQ",pack["temperature"],pack["epsilon"],pack["response_max"],
                                     pack["response_tolerance"],pack["training_frames"],pack["validation_frames"]))
            for site in pack["sites"]:
                neighbors=site["neighbors"]; matrix=np.asarray(site["B"],dtype="<f8")
                stream.write(struct.pack("<i",len(neighbors)))
                for atom,image in neighbors: stream.write(struct.pack("<4i",atom,*image))
                np.asarray(site["ell"],dtype="<f8").tofile(stream)
                entries=[(row,column,float(matrix[row,column])) for row in range(matrix.shape[0])
                         for column in range(matrix.shape[1])
                         if matrix[row,column] != 0.0 or np.signbit(matrix[row,column])]
                stream.write(struct.pack("<Q",len(entries)))
                for row,column,value in entries: stream.write(struct.pack("<iid",row,column,value))
        old = subprocess.run([sys.executable, str(ROOT / "tools" / "rpmd_ja_qnep_prepare.py"),
                              str(raw), str(legacy_path), "--kernel-table", str(kernel), "--output", str(old_output)],
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
    test_sparse_additive_v5_v6_roundtrip_and_dense_legacy_compatibility()
    test_hand_authored_sparse_package_readers_without_hardlink_writer()
    test_prepare_binds_diagnostic_identity_into_v3_policy()
    test_additive_modes_and_expansion_limits()
    test_additive_writer_rejects_reader_expansion_limit_before_temp_creation()
    test_additive_writer_and_readers_reject_coordination_over_reader_limit()
    test_prepare_reader_accepts_structurally_valid_package_over_64_mib()
    test_cli_additive_prepare_accumulates_actual_v3_k_and_h_and_certificate()
    test_internal_mass_com_v6_pulls_back_nonzero_net_and_preserves_raw_transport()
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
