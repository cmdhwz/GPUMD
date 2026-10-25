#!/usr/bin/env python3
"""Evaluate J_A corrections and full current correlations from a frozen v3 reference.

The manifest is an explicit identity/unit contract; the native v3 package does
not contain atom labels or the trajectory's continuous-coordinate branch ID.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np

MAGIC = b"GPUMDJA\0"
UNITS = b"position:A;energy:eV;mass:amu;temperature:K;time:native\0"
LAYOUT = b"xyz_soa;native_reference_transport;block_tiles_128;D_then_Bt;rowmajor_output_input\0"
ENDIAN = 0x01020304
TILE = 128
NATIVE_TIME_FS = 10.18051
NATIVE_TIME_PS = 0.01018051
HBAR_EV_NATIVE = 6.465412e-2
KB_EV_K = 8.617343e-5
_FIXED_UNITS = {
    "position": "angstrom",
    "velocity": "angstrom/ps",
    "centroid_current": "eV*angstrom/ps",
    "time": "ps",
    "reference_wire_time": "native",
    "native_time_ps": NATIVE_TIME_PS,
}


def _exact(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("truncated qNEP v3 reference")
    return value


def _fnv64(path):
    value = 14695981039346656037
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            for byte in chunk:
                value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linux_cgroup_available_bytes():
    if sys.platform != "linux":
        return None
    try:
        memberships = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    candidates = []
    for line in memberships:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, relative = fields[1], fields[2].lstrip("/")
        if controllers == "":
            root = Path("/sys/fs/cgroup")
            limit_name, usage_name, unlimited = "memory.max", "memory.current", "max"
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            limit_name, usage_name, unlimited = "memory.limit_in_bytes", "memory.usage_in_bytes", None
        else:
            continue
        leaf = root / relative
        if not leaf.is_dir():
            continue
        current = leaf
        while current == leaf or root in current.parents:
            limit_path, usage_path = current / limit_name, current / usage_name
            if limit_path.is_file() and usage_path.is_file():
                try:
                    limit_text = limit_path.read_text(encoding="ascii").strip()
                    usage_text = usage_path.read_text(encoding="ascii").strip()
                    usage = int(usage_text)
                    if limit_text != unlimited:
                        limit = int(limit_text)
                        if limit > usage:
                            candidates.append(limit - usage)
                except (OSError, ValueError):
                    return None
            if current == root:
                break
            current = current.parent
    return min(candidates) if candidates else None


def _available_memory_bytes():
    physical = None
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]

        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            physical = int(status.available)
    else:
        try:
            physical = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            physical = None
    cgroup = _linux_cgroup_available_bytes()
    known = [value for value in (physical, cgroup) if value is not None]
    return min(known) if known else None


def _read_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        if _exact(stream, 8) != MAGIC:
            raise ValueError("input is not a GPUMD RPMD-JA reference")
        version, endian, n, temperature, fd_step, model_fingerprint = struct.unpack("<IIiddQ", _exact(stream, 36))
        if version != 3 or endian != ENDIAN or n <= 1 or n > (2**31 - 1) // 3:
            raise ValueError("only a valid qNEP v3 block reference is supported")
        if _exact(stream, len(UNITS)) != UNITS or _exact(stream, len(LAYOUT)) != LAYOUT:
            raise ValueError("qNEP v3 wire units or D/Bt block layout do not match")
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(fd_step) or fd_step <= 0:
            raise ValueError("qNEP v3 temperature or finite-difference step is invalid")
        data_offset = stream.tell()
        file_bytes = path.stat().st_size
        if file_bytes < data_offset + 4:
            raise ValueError("qNEP v3 reference is too short for its required stability flag")
        stream.seek(-4, 2)
        if struct.unpack("<i", _exact(stream, 4))[0] != 1:
            raise ValueError("qNEP v3 reference is not marked stability-checked")
        return {"path": path, "version": version, "n": n, "d": 3 * n, "temperature": temperature,
                "fd_step": fd_step, "model_fingerprint": model_fingerprint, "data_offset": data_offset,
                "file_bytes": file_bytes}


def _estimate_peak_bytes(n, frame_count, reference_count, max_lag, include_online_current=False):
    d, r = 3 * n, 3 * n - 3
    # Covers dense D/Bt materialization, projection/eigensolver temporaries,
    # per-reference modal matrices, retained frame arrays, and full 3x3 lag terms.
    projection_values = 2 * d * d + 3 * d * r + 12 * r * r
    frame_values = frame_count * (3 + 6 * reference_count + (6 if include_online_current else 0))
    lag_values = (max_lag + 1) * 9 * (1 + 5 * reference_count)
    return (projection_values + frame_values + lag_values) * 8 + (128 << 20) + 2 * TILE * TILE * 8


def _require_keys(value, keys, where):
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{where} is missing required fields: {', '.join(missing)}")


def _validate_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(manifest, ("format", "trajectory_id", "trajectory_npz", "frame_count", "frame_ids",
        "step_ids", "time_start_ps", "time_interval_ps", "ordered_atom_ids", "masses_amu", "types",
        "cell_angstrom", "temperature_K", "position_stream_id", "position_semantics", "branch_policy", "units", "memory_limit_bytes",
        "max_lag_frames", "references", "tolerances", "trajectory_npz_sha256",
        "dump_source_sha256", "current_source_sha256"), "manifest")
    if manifest["format"] not in ("RPJAMAT1", "RPJAMAT2"):
        raise ValueError("unsupported manifest format")
    if manifest["units"] != _FIXED_UNITS or manifest["position_semantics"] != "wrapped_position_per_atom_A" or \
       manifest["branch_policy"] != "native_rpmd_ja_mic_v1":
        raise ValueError("manifest units must explicitly use GPUMD native time (10.18051 fs) and the supported wire units")
    if not isinstance(manifest["position_stream_id"], str) or not manifest["position_stream_id"]:
        raise ValueError("manifest requires an explicit position_stream_id")
    nframe = manifest["frame_count"]
    ids = manifest["ordered_atom_ids"]
    refs = manifest["references"]
    if not isinstance(nframe, int) or nframe < 2 or not isinstance(ids, list) or len(ids) < 2 or \
       any(not isinstance(atom_id, str) or not atom_id for atom_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest frame count or ordered atom identities are invalid")
    if len(manifest["frame_ids"]) != nframe or len(manifest["step_ids"]) != nframe or \
       len(set(manifest["frame_ids"])) != nframe or len(set(manifest["step_ids"])) != nframe:
        raise ValueError("manifest must freeze unique frame_ids and step_ids for every sample")
    dt = float(manifest["time_interval_ps"])
    t0 = float(manifest["time_start_ps"])
    temp = float(manifest["temperature_K"])
    memory_limit = manifest["memory_limit_bytes"]
    max_lag = manifest["max_lag_frames"]
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(t0) or not math.isfinite(temp) or temp <= 0 or \
       not isinstance(memory_limit, int) or memory_limit <= 0 or not isinstance(max_lag, int) or \
       max_lag < 0 or max_lag >= nframe:
        raise ValueError("manifest time grid, temperature, memory limit, or maximum lag is invalid")
    masses = np.asarray(manifest["masses_amu"], dtype=np.float64)
    types = np.asarray(manifest["types"], dtype=np.int32)
    cell = np.asarray(manifest["cell_angstrom"], dtype=np.float64)
    if masses.shape != (len(ids),) or types.shape != (len(ids),) or cell.shape != (3, 3) or \
       not np.isfinite(masses).all() or np.any(masses <= 0) or not np.isfinite(cell).all() or \
       abs(float(np.linalg.det(cell))) <= 1e-12:
        raise ValueError("manifest ordered masses, types, or cell are invalid")
    tolerances = manifest["tolerances"]
    _require_keys(tolerances, ("delta_j_near_zero_rms_native", "reference_sensitivity_relative",
        "reference_sensitivity_absolute_native"), "tolerances")
    values = [float(tolerances[key]) for key in ("delta_j_near_zero_rms_native",
        "reference_sensitivity_relative", "reference_sensitivity_absolute_native")]
    if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[1] < 0 or values[2] <= 0:
        raise ValueError("frozen reference-sensitivity tolerances must be finite and nonnegative, with positive absolute thresholds")
    if not isinstance(refs, list) or not refs:
        raise ValueError("manifest must list at least one fixed reference")
    ref_ids = set()
    ref_digests = {}
    for index, ref in enumerate(refs):
        _require_keys(ref, ("id", "package", "package_sha256", "ordered_atom_ids", "reference_position_stream_id"),
                      f"references[{index}]")
        if not isinstance(ref["id"], str) or not ref["id"] or ref["id"] in ref_ids:
            raise ValueError("reference IDs must be nonempty and unique")
        ref_ids.add(ref["id"])
        if ref["ordered_atom_ids"] != ids:
            raise ValueError(f"reference {ref['id']} atom order/identity does not match the trajectory manifest")
        if ref["reference_position_stream_id"] != manifest["position_stream_id"]:
            raise ValueError(f"reference {ref['id']} wrapped-position stream identity does not match")
        if not isinstance(ref["package_sha256"], str) or len(ref["package_sha256"]) != 64:
            raise ValueError(f"reference {ref['id']} requires a frozen package SHA-256")
        ref_digests[ref["id"]] = ref["package_sha256"].lower()
        package = Path(ref["package"])
        if not package.is_absolute():
            package = path.parent / package
        ref["_resolved_package"] = package.resolve()
    if manifest["format"] == "RPJAMAT2":
        online = manifest.get("online_current")
        if not isinstance(online, dict):
            raise ValueError("RPJAMAT2 requires an online_current identity and unit contract")
        _require_keys(online, ("reference_id", "reference_package_sha256", "units"), "online_current")
        if online["reference_id"] not in ref_digests or \
           not isinstance(online["reference_package_sha256"], str) or \
           online["reference_package_sha256"].lower() != ref_digests[online["reference_id"]]:
            raise ValueError("online current reference identity or package digest does not match a frozen reference")
        if online["units"] != "eV*angstrom/ps":
            raise ValueError("RPJAMAT2 online DeltaJ and JA units must be eV*angstrom/ps")
    trajectory = Path(manifest["trajectory_npz"])
    if not trajectory.is_absolute():
        trajectory = path.parent / trajectory
    manifest["_resolved_trajectory"] = trajectory.resolve()
    for key in ("trajectory_npz_sha256", "dump_source_sha256", "current_source_sha256"):
        digest = manifest[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"manifest {key} must be a frozen SHA-256")
    manifest["_source_path"] = path
    return manifest


def _read_vector(stream, dtype, count):
    dtype = np.dtype(dtype)
    size = count * dtype.itemsize
    data = _exact(stream, size)
    return np.frombuffer(data, dtype=dtype).copy()


def _inspect_trajectory_npz(path, frames, atoms, memory_limit_bytes, trajectory_format):
    expected = {
        "frame_ids": ((frames,), "i", 8), "step_ids": ((frames,), "i", 8),
        "time_ps": ((frames,), "f", 8), "atom_ids": ((atoms,), "U", None),
        "trajectory_id": ((), "U", None), "position_stream_id": ((), "U", None),
        "position_source": ((), "U", None), "dump_source_sha256": ((), "U", None),
        "current_source_sha256": ((), "U", None), "cell_angstrom": ((3, 3), "f", 8),
        "wrapped_position_angstrom": ((frames, atoms, 3), "f", 8),
        "vc_angstrom_per_ps": ((frames, atoms, 3), "f", 8),
        "jc_eV_angstrom_per_ps": ((frames, 3), "f", 8)
    }
    if trajectory_format == "RPJAMAT2":
        expected.update({
            "online_reference_id": ((), "U", None),
            "online_reference_package_sha256": ((), "U", None),
            "online_current_units": ((), "U", None),
            "online_delta_j_eV_angstrom_per_ps": ((frames, 3), "f", 8),
            "online_ja_eV_angstrom_per_ps": ((frames, 3), "f", 8)
        })
    optional_online_identity = {
        "online_reference_use_class": ((), "U", None),
        "online_reference_numerical_status": ((), "U", None),
        "online_reference_response_status": ((), "U", None),
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = {info.filename: info for info in archive.infolist()}
            wanted = {key + ".npy" for key in expected}
            member_names = set(members)
            identity_members = {key + ".npy" for key in optional_online_identity}
            if trajectory_format == "RPJAMAT2" and member_names == wanted | identity_members:
                expected.update(optional_online_identity)
                wanted |= identity_members
            if member_names != wanted:
                raise ValueError("trajectory NPZ must contain exactly the declared synchronized arrays")
            total_uncompressed = sum(info.file_size for info in members.values())
            if total_uncompressed > memory_limit_bytes:
                raise MemoryError(f"trajectory NPZ uncompressed bytes {total_uncompressed} exceed frozen memory limit {memory_limit_bytes}")
            for key, (shape_expected, kind_expected, itemsize_expected) in expected.items():
                info = members[key + ".npy"]
                if info.file_size > memory_limit_bytes:
                    raise MemoryError(f"trajectory NPZ member {key} exceeds frozen memory limit")
                with archive.open(info, "r") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"trajectory NPZ member {key} uses unsupported NPY header version {version}")
                    payload_offset = stream.tell()
                dtype = np.dtype(dtype)
                if shape != shape_expected or dtype.kind != kind_expected or fortran_order:
                    raise ValueError(f"trajectory NPZ member {key} shape, dtype kind, or array order is invalid")
                if itemsize_expected is not None and dtype.itemsize != itemsize_expected:
                    raise ValueError(f"trajectory NPZ member {key} dtype width is invalid")
                payload_bytes = math.prod(shape) * dtype.itemsize
                if payload_offset + payload_bytes != info.file_size:
                    raise ValueError(f"trajectory NPZ member {key} byte count does not match its declared shape")
    except zipfile.BadZipFile as exc:
        raise ValueError("trajectory input is not a valid NPZ archive") from exc


def _read_block_matrix(stream, dimension):
    tile_size, count = struct.unpack("<iQ", _exact(stream, 12))
    if tile_size != TILE:
        raise ValueError("qNEP v3 block matrix does not use the production 128 tile size")
    grid = (dimension + TILE - 1) // TILE
    if count != grid * grid:
        raise ValueError("qNEP v3 block matrix tile grid is incomplete")
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    seen = set()
    for _ in range(count):
        row, column, rows, columns, rank, left_count, right_count = struct.unpack("<5iQQ", _exact(stream, 36))
        if row < 0 or column < 0 or row % TILE or column % TILE or row >= dimension or column >= dimension:
            raise ValueError("qNEP v3 block tile origin is invalid")
        expected_rows = min(TILE, dimension - row)
        expected_columns = min(TILE, dimension - column)
        key = (row // TILE, column // TILE)
        if key in seen or rows != expected_rows or columns != expected_columns or rank < 0 or \
           rank > min(rows, columns):
            raise ValueError("qNEP v3 block tile shape, rank, or uniqueness is invalid")
        seen.add(key)
        expected_left = rows * columns if rank == 0 else rows * rank
        expected_right = 0 if rank == 0 else rank * columns
        if left_count != expected_left or right_count != expected_right:
            raise ValueError("qNEP v3 block tile factor counts are invalid")
        left = _read_vector(stream, "<f8", left_count)
        right = _read_vector(stream, "<f8", right_count)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("qNEP v3 block factors contain non-finite values")
        block = left.reshape(rows, columns) if rank == 0 else left.reshape(rows, rank) @ right.reshape(rank, columns)
        matrix[row:row + rows, column:column + columns] = block
    if len(seen) != grid * grid or not np.isfinite(matrix).all():
        raise ValueError("qNEP v3 block matrix has missing or non-finite tiles")
    return matrix


def _validate_sidecar(path, header, config_fingerprint, policy):
    sidecar = Path(str(path) + ".stability")
    if not sidecar.is_file():
        raise ValueError("missing qNEP v3 stability sidecar")
    try:
        lines = sidecar.read_text(encoding="ascii").splitlines()
        values = {}
        if not lines or len(lines[0].split()) != 2:
            raise ValueError
        tag, version_text = lines[0].split()
        version = int(version_text)
        if tag != "GPUMDJA_QNEP_STABILITY" or version not in (1, 3, 4):
            raise ValueError
        for line in lines[1:]:
            key, value = line.split(maxsplit=1)
            if key in values:
                raise ValueError
            values[key] = value
        required = {"fingerprint", "config_fingerprint", "atoms", "translation_residual",
                    "reconstruction_residual", "softmode_relative_error"}
        if not required.issubset(values):
            raise ValueError
        if int(values["fingerprint"], 16) != _fnv64(path) or \
           int(values["config_fingerprint"], 16) != config_fingerprint or \
           int(values["atoms"]) != header["n"]:
            raise ValueError
        additive = "finite_temperature_additive_v1" in policy
        if additive != (version == 4):
            raise ValueError
        if version == 1:
            if set(values) != required | {"minimum_positive_eigenvalue"}:
                raise ValueError
            minimum = float(values["minimum_positive_eigenvalue"])
        elif version == 3:
            if set(values) != required | {"derivative_policy", "certificate", "minimum_cholesky_pivot",
                                           "relative_operator_bound"}:
                raise ValueError
            if values.get("derivative_policy") != policy or values.get("certificate") != "cholesky_relative_bound":
                raise ValueError
            minimum = float(values["minimum_cholesky_pivot"])
            operator_bound = float(values["relative_operator_bound"])
            if not 0 <= operator_bound <= 1e-2:
                raise ValueError
        else:
            if set(values) != required | {"derivative_policy", "certificate", "epsilon_num",
                                           "reconstruction_bound", "minimum_cholesky_pivot",
                                           "relative_operator_bound"}:
                raise ValueError
            if values.get("derivative_policy") != policy or values.get("certificate") != "additive_shifted_frobenius_v1":
                raise ValueError
            epsilon = float(values["epsilon_num"])
            bound = float(values["reconstruction_bound"])
            minimum = float(values["minimum_cholesky_pivot"])
            operator_bound = float(values["relative_operator_bound"])
            if not epsilon > 0 or not 0 <= bound < 0.5 * epsilon or operator_bound != 0:
                raise ValueError
        translation = float(values["translation_residual"])
        reconstruction = float(values["reconstruction_residual"])
        softmode = float(values["softmode_relative_error"])
        if not minimum > 0 or not all(math.isfinite(x) for x in (minimum, translation, reconstruction, softmode)) or \
           not 0 <= translation <= 1e-8 or not 0 <= reconstruction <= 1e-8 or not 0 <= softmode <= 1e-2:
            raise ValueError
        return {"version": version, "minimum_pivot_or_eigenvalue": minimum,
                "translation_residual": translation, "reconstruction_residual": reconstruction,
                "softmode_relative_error": softmode}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("qNEP v3 stability sidecar/certificate does not validate this package") from exc


def _load_reference(ref, manifest, estimate_bytes, sample):
    header = _read_header(ref["_resolved_package"])
    if header["n"] != len(manifest["ordered_atom_ids"]):
        raise ValueError(f"reference {ref['id']} atom count does not match explicit atom identities")
    if estimate_bytes > manifest["memory_limit_bytes"]:
        raise MemoryError(f"estimated host memory {estimate_bytes} bytes exceeds frozen limit {manifest['memory_limit_bytes']}")
    available = _available_memory_bytes()
    if available is not None and estimate_bytes > available:
        raise MemoryError(f"estimated host memory {estimate_bytes} bytes exceeds currently available {available}")
    if _sha256(header["path"]).lower() != ref["package_sha256"].lower():
        raise ValueError(f"reference {ref['id']} package SHA-256 changed after the manifest was frozen")

    n, d = header["n"], header["d"]
    with header["path"].open("rb") as stream:
        stream.seek(header["data_offset"])
        cell = _read_vector(stream, "<f8", 9).reshape(3, 3)
        pbc = _read_vector(stream, "<i4", 3)
        types = _read_vector(stream, "<i4", n)
        masses = _read_vector(stream, "<f8", n)
        positions = _read_vector(stream, "<f8", d).reshape(3, n).T.copy()
        config_fp, charge, pppm, spacing, policy_size = struct.unpack("<Qiidi", _exact(stream, 28))
        if policy_size <= 0 or policy_size > 192:
            raise ValueError("qNEP v3 mechanical policy length is invalid")
        policy = _exact(stream, policy_size).decode("ascii")
        diagnostic_policy=";use=DIAGNOSTIC;" in policy
        expected_use="DIAGNOSTIC" if diagnostic_policy else "FORMAL"
        diagnostic_suffix=policy[policy.find(";use="):] if ";use=" in policy else ""
        diagnostic_statuses=("PASS","FAIL","INCONCLUSIVE","NOT_MEASURED")
        policy_response=(policy.split(";response=",1)[1].split(";",1)[0]
                         if diagnostic_policy and ";response=" in policy else None)
        if diagnostic_policy and (manifest["format"] != "RPJAMAT2" or diagnostic_suffix not in tuple(
                f";use=DIAGNOSTIC;response={status};numerical=PASS" for status in diagnostic_statuses) or
                manifest.get("analysis_use_class") != "DIAGNOSTIC" or ref.get("use_class") != "DIAGNOSTIC" or
                ref.get("numerical_status") != "PASS" or
                ref.get("response_status") != policy_response):
            raise ValueError(f"diagnostic reference {ref['id']} requires an explicit matching manifest use identity")
        if ";use=" in policy and not diagnostic_policy:
            raise ValueError(f"reference {ref['id']} has an unsupported use identity")
        if not diagnostic_policy and ref.get("use_class", "FORMAL") != expected_use:
            raise ValueError(f"reference {ref['id']} manifest use class does not match its package")
        sidecar = _validate_sidecar(header["path"], header, config_fp, policy)
        diagnostics = _read_vector(stream, "<f8", 21)
        spectral_bound, kernel_u = struct.unpack("<dd", _exact(stream, 16))
        kernel_error = _read_vector(stream, "<f8", 2)
        kernel_s2 = _read_vector(stream, "<f8", 2)
        degree, p_rank, q_rank = struct.unpack("<iii", _exact(stream, 12))
        if degree < 0 or degree > 512 or p_rank <= 0 or q_rank <= 0 or \
           p_rank > degree + 1 or q_rank > degree + 1:
            raise ValueError("qNEP v3 polynomial ranks or degree are invalid")
        p_values = _read_vector(stream, "<f8", p_rank)
        q_values = _read_vector(stream, "<f8", q_rank)
        p_vectors = _read_vector(stream, "<f8", (degree + 1) * p_rank).reshape(degree + 1, p_rank)
        q_vectors = _read_vector(stream, "<f8", (degree + 1) * q_rank).reshape(degree + 1, q_rank)
        if not np.isfinite(cell).all() or not np.isfinite(masses).all() or np.any(masses <= 0) or \
           not np.isfinite(positions).all() or not np.isfinite(diagnostics).all() or \
           not np.isfinite(kernel_error).all() or not np.isfinite(kernel_s2).all() or \
           not all(np.isfinite(x).all() for x in (p_values, q_values, p_vectors, q_vectors)) or \
           np.any(pbc != 1) or np.any((types < 0) | (types > 3)) or not np.isfinite(spacing) or spacing <= 0 or \
           charge not in (1, 2) or pppm != 1 or config_fp == 0 or not policy.startswith("native_reference_transport"):
            raise ValueError("qNEP v3 geometry, types, mechanics, diagnostics, or kernel data are invalid")
        if np.any(diagnostics < 0) or diagnostics[6] > 1e-4 or diagnostics[5] > 5e-2 or \
           diagnostics[1] > 5e-2 or diagnostics[2] > 5e-2 or diagnostics[3] > 5e-2 or \
           np.any(diagnostics[17:21] > 1e-8) or np.any(diagnostics[14:17] > 5e-2):
            raise ValueError("qNEP v3 finite-difference, projection, or block checks exceed production limits")
        expected_masses = np.asarray(manifest["masses_amu"], dtype=np.float64)
        expected_types = np.asarray(manifest["types"], dtype=np.int32)
        expected_cell = np.asarray(manifest["cell_angstrom"], dtype=np.float64)
        if not np.array_equal(masses, expected_masses) or not np.array_equal(types, expected_types) or \
           not np.allclose(cell, expected_cell, rtol=0.0, atol=1e-12) or \
           not math.isclose(header["temperature"], float(manifest["temperature_K"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"reference {ref['id']} ordered geometry, mass, type, or temperature binding mismatch")

        dynamical = _read_block_matrix(stream, d)
        sqrt_mass = np.tile(np.sqrt(masses), 3)
        mass_total = float(np.sum(masses))
        internal = _translation_complement(sqrt_mass, mass_total, n)
        projected = internal.T @ dynamical @ internal
        eigenvalues, modes = np.linalg.eigh(projected, UPLO="L")
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            raise ValueError(f"reference {ref['id']} projected D is not positive definite")
        if np.max(eigenvalues) > spectral_bound * (1.0 + 32.0 * np.finfo(float).eps):
            raise ValueError(f"reference {ref['id']} spectrum exceeds its stored bound")
        if not math.isfinite(spectral_bound) or spectral_bound <= 0 or not math.isfinite(kernel_u) or kernel_u <= 0:
            raise ValueError("qNEP v3 spectrum or kernel U is invalid")
        tau = HBAR_EV_NATIVE / (KB_EV_K * header["temperature"])
        if tau * math.sqrt(spectral_bound) > kernel_u * (1.0 + 32.0 * np.finfo(float).eps):
            raise ValueError("qNEP v3 kernel table does not cover the stored spectrum")
        if np.any(kernel_error < 0) or np.any(kernel_s2 < 0):
            raise ValueError("qNEP v3 kernel error metadata is invalid")
        eigvec = internal @ modes
        del dynamical, projected, modes, internal

        xc = _replay_native_ja_branch(sample["wrapped_position_angstrom"], positions, cell, ref["id"])
        displacement = xc.transpose(0, 2, 1).reshape(manifest["frame_count"], d) - positions.T.reshape(d)[None, :]
        weighted_x = displacement * sqrt_mass[None, :]
        weighted_v = sample["vc_native_soa"] * sqrt_mass[None, :]
        modal_x = weighted_x @ eigvec
        modal_v = weighted_v @ eigvec
        lambda_scale = kernel_u * kernel_u / (tau * tau)
        z = 2.0 * eigenvalues / lambda_scale - 1.0
        p_modes = _chebyshev_modes(z, p_vectors)
        q_modes = _chebyshev_modes(z, q_vectors)
        delta_j = np.empty((manifest["frame_count"], 3), dtype=np.float64)
        for axis in range(3):
            bt = _read_block_matrix(stream, d)
            a = eigvec.T @ bt @ eigvec
            del bt
            p = (p_modes * p_values[None, :]) @ p_modes.T
            q = (q_modes * q_values[None, :]) @ q_modes.T
            ta = tau * tau * eigenvalues[:, None]
            tb = tau * tau * eigenvalues[None, :]
            kernel = ta * tb * p * a + tb * q * a.T
            delta_j[:, axis] = np.einsum("fi,ij,fj->f", modal_v, kernel, modal_x, optimize=True)
        if struct.unpack("<i", _exact(stream, 4))[0] != 1 or stream.read(1):
            raise ValueError("qNEP v3 stability flag or trailing bytes are invalid")
    return {"header": header, "positions": positions, "masses": masses, "types": types,
            "cell": cell, "delta_j_native": delta_j, "sidecar": sidecar,
            "reference_use_class": expected_use,
            "reference_numerical_status": "PASS" if diagnostic_policy else "NOT_APPLICABLE",
            "reference_response_status": policy_response if diagnostic_policy else "NOT_APPLICABLE",
            "sample": sample}


def _translation_complement(sqrt_mass, mass_total, n):
    total = math.sqrt(mass_total)
    q = np.zeros((3 * n, 3 * n - 3), dtype=np.float64)
    for axis in range(3):
        translation = sqrt_mass[:n] / total
        vector = translation.copy()
        vector[0] -= 1.0
        norm = np.linalg.norm(vector)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("cannot construct the mass-weighted translation complement")
        vector /= norm
        reflector = np.eye(n) - 2.0 * np.outer(vector, vector)
        target = np.arange(n - 1) + axis * (n - 1)
        q[axis * n:(axis + 1) * n, target] = reflector[:, 1:]
    return q


def _chebyshev_modes(z, vectors):
    result = vectors[0][None, :] + z[:, None] * vectors[1][None, :] if len(vectors) > 1 else \
        np.broadcast_to(vectors[0], (len(z), vectors.shape[1])).copy()
    if len(vectors) > 1:
        t0 = np.ones_like(z)
        t1 = z.copy()
        for degree in range(2, len(vectors)):
            t2 = 2.0 * z * t1 - t0
            result += t2[:, None] * vectors[degree][None, :]
            t0, t1 = t1, t2
    return result


def _load_trajectory(manifest, n):
    if _sha256(manifest["_resolved_trajectory"]).lower() != manifest["trajectory_npz_sha256"].lower():
        raise ValueError("trajectory NPZ SHA-256 changed after the manifest was frozen")
    required = ("frame_ids", "step_ids", "time_ps", "atom_ids", "trajectory_id", "position_stream_id",
                "position_source", "cell_angstrom", "dump_source_sha256", "current_source_sha256",
                "wrapped_position_angstrom", "vc_angstrom_per_ps", "jc_eV_angstrom_per_ps")
    if manifest["format"] == "RPJAMAT2":
        required += ("online_reference_id", "online_reference_package_sha256", "online_current_units",
                     "online_delta_j_eV_angstrom_per_ps", "online_ja_eV_angstrom_per_ps")
    _inspect_trajectory_npz(manifest["_resolved_trajectory"], manifest["frame_count"], n,
                            manifest["memory_limit_bytes"], manifest["format"])
    with np.load(manifest["_resolved_trajectory"], allow_pickle=False) as data:
        identity_keys = ("online_reference_use_class", "online_reference_numerical_status",
                         "online_reference_response_status")
        online = manifest.get("online_current", {})
        diagnostic_selected = manifest.get("analysis_use_class") == "DIAGNOSTIC" or \
            online.get("reference_use_class") == "DIAGNOSTIC"
        if manifest["format"] == "RPJAMAT2" and diagnostic_selected and \
           not all(key in data.files for key in identity_keys):
            raise ValueError("diagnostic current is missing its bound online reference-use identity")
        if manifest["format"] == "RPJAMAT2" and any(key in data.files for key in identity_keys):
            if not all(key in data.files for key in identity_keys):
                raise ValueError("trajectory NPZ has an incomplete online reference-use identity")
            required += identity_keys
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"trajectory NPZ is missing required synchronized fields: {', '.join(missing)}")
        sample = {key: data[key] for key in required}
    frames = manifest["frame_count"]
    atom_ids = [str(value) for value in sample["atom_ids"].tolist()]
    if atom_ids != manifest["ordered_atom_ids"]:
        raise ValueError("trajectory atom order/identity does not match the frozen manifest")
    if str(sample["trajectory_id"].item()) != manifest["trajectory_id"] or \
       str(sample["position_stream_id"].item()) != manifest["position_stream_id"] or \
       str(sample["position_source"].item()) != "dump_xyz.atom.position_per_atom" or \
       str(sample["dump_source_sha256"].item()).lower() != manifest["dump_source_sha256"].lower() or \
       str(sample["current_source_sha256"].item()).lower() != manifest["current_source_sha256"].lower():
        raise ValueError("trajectory ID, position stream, atom source, or original input digest does not match the manifest")
    if manifest["format"] == "RPJAMAT2":
        online = manifest["online_current"]
        if str(sample["online_reference_id"].item()) != online["reference_id"] or \
           str(sample["online_reference_package_sha256"].item()).lower() != online["reference_package_sha256"].lower() or \
           str(sample["online_current_units"].item()) != online["units"]:
            raise ValueError("online current reference identity, package digest, or units do not match the manifest")
        if "online_reference_use_class" in sample:
            current_use = str(sample["online_reference_use_class"].item())
            current_numerical = str(sample["online_reference_numerical_status"].item())
            current_response = str(sample["online_reference_response_status"].item())
            if current_use not in ("FORMAL", "DIAGNOSTIC") or \
               current_numerical not in ("PASS", "NOT_APPLICABLE") or \
               current_response not in ("PASS", "FAIL", "INCONCLUSIVE", "NOT_MEASURED", "NOT_APPLICABLE") or \
               current_use == "DIAGNOSTIC" and (manifest.get("analysis_use_class") != "DIAGNOSTIC" or
                   online.get("reference_use_class") != "DIAGNOSTIC" or
                   online.get("reference_numerical_status") != current_numerical or
                   online.get("reference_response_status") != current_response):
                raise ValueError("online current reference use identity does not match the explicit manifest selection")
            if current_use == "FORMAL" and online.get("reference_use_class", "FORMAL") != "FORMAL":
                raise ValueError("online current formal identity does not match the frozen manifest")
    if sample["wrapped_position_angstrom"].shape != (frames, n, 3) or sample["vc_angstrom_per_ps"].shape != (frames, n, 3) or \
       sample["jc_eV_angstrom_per_ps"].shape != (frames, 3):
        raise ValueError("trajectory coordinates, velocities, and centroid currents must be synchronized F-by-N arrays")
    if not np.array_equal(sample["frame_ids"], np.asarray(manifest["frame_ids"])) or \
       not np.array_equal(sample["step_ids"], np.asarray(manifest["step_ids"])):
        raise ValueError("trajectory frame/step identities differ from the frozen manifest")
    time = np.asarray(sample["time_ps"], dtype=np.float64)
    expected_time = float(manifest["time_start_ps"]) + np.arange(frames) * float(manifest["time_interval_ps"])
    if time.shape != (frames,) or not np.isfinite(time).all() or not np.allclose(time, expected_time, rtol=0, atol=1e-10):
        raise ValueError("trajectory time grid is not the frozen uniform ps grid")
    expected_cell = np.asarray(manifest["cell_angstrom"], dtype=np.float64)
    if sample["cell_angstrom"].shape != (3, 3) or \
       not np.allclose(sample["cell_angstrom"], expected_cell, rtol=0.0, atol=1e-12):
        raise ValueError("trajectory cell does not match the frozen package geometry")
    finite_keys = ["wrapped_position_angstrom", "vc_angstrom_per_ps", "jc_eV_angstrom_per_ps"]
    if manifest["format"] == "RPJAMAT2":
        finite_keys.extend(("online_delta_j_eV_angstrom_per_ps", "online_ja_eV_angstrom_per_ps"))
    for key in finite_keys:
        if not np.isfinite(sample[key]).all():
            raise ValueError(f"trajectory {key} contains non-finite values")
    ps_per_native = NATIVE_TIME_PS
    sample["vc_native_soa"] = sample["vc_angstrom_per_ps"].transpose(0, 2, 1).reshape(frames, 3 * n) * ps_per_native
    sample["jc_native"] = np.asarray(sample["jc_eV_angstrom_per_ps"], dtype=np.float64) * ps_per_native
    if manifest["format"] == "RPJAMAT2":
        sample["online_delta_j_native"] = np.asarray(
            sample["online_delta_j_eV_angstrom_per_ps"], dtype=np.float64) * ps_per_native
        sample["online_ja_native"] = np.asarray(
            sample["online_ja_eV_angstrom_per_ps"], dtype=np.float64) * ps_per_native
    sample["time_ps"] = time
    return sample


def _replay_native_ja_branch(wrapped, reference, cell, reference_id):
    inverse = np.linalg.inv(cell)
    delta = np.asarray(wrapped[0], dtype=np.float64) - reference
    fractional = delta @ inverse.T
    fractional -= np.rint(fractional)
    continuous = np.empty_like(wrapped, dtype=np.float64)
    continuous[0] = reference + fractional @ cell.T
    if np.any(np.abs(fractional) >= 0.45):
        raise ValueError(f"reference {reference_id} initial MIC lies outside the production 0.45 branch window")
    last_raw = wrapped[0]
    for frame in range(1, len(wrapped)):
        step = np.asarray(wrapped[frame], dtype=np.float64) - last_raw
        step_fractional = step @ inverse.T
        step_fractional -= np.rint(step_fractional)
        if not np.isfinite(step_fractional).all() or np.any(np.abs(step_fractional) >= 0.45):
            raise ValueError(f"reference {reference_id} frame {frame} has an ambiguous wrapped-position MIC step (0.45 limit)")
        continuous[frame] = continuous[frame - 1] + step_fractional @ cell.T
        branch = (continuous[frame] - reference) @ inverse.T
        if np.any(np.abs(branch) >= 0.45):
            raise ValueError(f"reference {reference_id} frame {frame} leaves the production 0.45 crystal branch")
        last_raw = wrapped[frame]
    return continuous


def _correlate(left, right, max_lag):
    frames = len(left)
    result = np.empty((max_lag + 1, 3, 3), dtype=np.float64)
    for lag in range(max_lag + 1):
        count = frames - lag
        result[lag] = left[:count].T @ right[lag:] / count
    return result


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def evaluate_manifest(manifest_path):
    manifest = _validate_manifest(manifest_path)
    headers = [_read_header(ref["_resolved_package"]) for ref in manifest["references"]]
    if any(header["n"] != len(manifest["ordered_atom_ids"]) for header in headers):
        raise ValueError("reference atom count does not match explicit ordered atom IDs")
    estimate_bytes = _estimate_peak_bytes(headers[0]["n"], manifest["frame_count"], len(headers),
                                         manifest["max_lag_frames"], manifest["format"] == "RPJAMAT2")
    if estimate_bytes > manifest["memory_limit_bytes"]:
        raise MemoryError(f"estimated host memory {estimate_bytes} bytes exceeds frozen limit {manifest['memory_limit_bytes']}")
    available = _available_memory_bytes()
    if available is not None and estimate_bytes > available:
        raise MemoryError(f"estimated host memory {estimate_bytes} bytes exceeds currently available {available}")
    sample = _load_trajectory(manifest, len(manifest["ordered_atom_ids"]))
    jc = sample["jc_native"]
    loaded_references = [_load_reference(ref, manifest, estimate_bytes, sample) for ref in manifest["references"]]
    if manifest["format"] == "RPJAMAT2":
        online = manifest["online_current"]
        reference_index = next(index for index, ref in enumerate(manifest["references"])
                               if ref["id"] == online["reference_id"])
        online_reference = loaded_references[reference_index]
        online_identity = (online.get("reference_use_class", "FORMAL"),
                           online.get("reference_numerical_status", "NOT_APPLICABLE"),
                           online.get("reference_response_status", "NOT_APPLICABLE"))
        package_identity = (online_reference["reference_use_class"],
                            online_reference["reference_numerical_status"],
                            online_reference["reference_response_status"])
        if online_identity != package_identity:
            raise ValueError("online current reference-use identity does not match the selected reference package")
    delta_series = [loaded["delta_j_native"] for loaded in loaded_references]
    online_comparison = {"status": "NOT_AVAILABLE_LEGACY_RPJAMAT1"}
    if manifest["format"] == "RPJAMAT2":
        online = manifest["online_current"]
        reference_index = next(index for index, ref in enumerate(manifest["references"])
                               if ref["id"] == online["reference_id"])
        reconstructed_delta = delta_series[reference_index]
        reconstructed_ja = jc + reconstructed_delta
        online_delta = sample["online_delta_j_native"]
        online_ja = sample["online_ja_native"]
        delta_error = float(np.max(np.abs(online_delta - reconstructed_delta)))
        ja_error = float(np.max(np.abs(online_ja - reconstructed_ja)))
        delta_pass = np.allclose(online_delta, reconstructed_delta, rtol=1e-10, atol=1e-9)
        ja_pass = np.allclose(online_ja, reconstructed_ja, rtol=1e-10, atol=1e-9)
        online_comparison = {"status": "PASS" if delta_pass and ja_pass else "FAIL",
            "reference_id": online["reference_id"],
            "reference_package_sha256": online["reference_package_sha256"].lower(),
            "reference_use_class": online_reference["reference_use_class"],
            "reference_numerical_status": online_reference["reference_numerical_status"],
            "reference_response_status": online_reference["reference_response_status"],
            "units_compared": "eV*angstrom/native_time",
            "criterion": "per-frame np.allclose(rtol=1e-10, atol=1e-9)",
            "delta_j_max_abs_error": delta_error, "ja_max_abs_error": ja_error}
        if not delta_pass or not ja_pass:
            raise ValueError("online DeltaJ or JA differs from the independent per-frame reconstruction")
    max_lag = manifest["max_lag_frames"]
    jc_jc = _correlate(jc, jc, max_lag)
    arrays = {"jc_native": jc, "lags_ps": np.arange(max_lag + 1) * float(manifest["time_interval_ps"]),
              "jc_jc": jc_jc}
    if manifest["format"] == "RPJAMAT2":
        arrays["online_delta_j_eV_angstrom_per_ps"] = sample["online_delta_j_eV_angstrom_per_ps"]
        arrays["online_ja_eV_angstrom_per_ps"] = sample["online_ja_eV_angstrom_per_ps"]
    report = {
        "format": "RPJAMAT2_REPORT" if manifest["format"] == "RPJAMAT2" else "RPJAMAT1_REPORT",
        "status": "EVALUATION_ONLY",
        "analysis_use_class": manifest.get("analysis_use_class", "FORMAL"),
        "trajectory_id": manifest["trajectory_id"],
        "frame_count": manifest["frame_count"],
        "frame_ids": manifest["frame_ids"],
        "step_ids": manifest["step_ids"],
        "ordered_atom_ids": manifest["ordered_atom_ids"],
        "atom_order_binding": "explicit_manifest_and_npz_identity; native_v3_has_no_atom_labels",
        "position_stream_id": manifest["position_stream_id"],
        "branch_policy": manifest["branch_policy"],
        "units": {**manifest["units"], "delta_j": "eV*angstrom/native_time"},
        "native_time_ps": NATIVE_TIME_PS,
        "xc_semantics": "wrapped_position_per_atom_A; production MIC replay per reference R0",
        "correlation_definition": "C_ab[i,j,l]=mean_t(J_a[i,t]*J_b[j,t+l]); no mean subtraction",
        "max_lag_frames": max_lag,
        "hac_integral_units": "eV^2*angstrom^2/ps",
        "hac_integral_note": "converted from native current and integrated over ps; not divided by volume or kBT and not thermal conductivity",
        "resource_estimate": {"estimated_host_peak_bytes": estimate_bytes,
                              "frozen_memory_limit_bytes": manifest["memory_limit_bytes"],
                              "currently_available_host_bytes": available,
                              "estimate_is_capacity_gate_not_runtime_guarantee": True,
                              "model_includes": "dense D/Bt, projection/eigensolver temporaries, modal matrices, retained frame arrays, all 3x3 lag terms",
                              "analysis_backend": "CPU NumPy dense internal projection/eigensolve; no speed advantage over native GPU is claimed"},
        "tolerances": manifest["tolerances"],
        "online_current_comparison": online_comparison,
        "references": [],
        "reference_sensitivity": {"status": "INCONCLUSIVE_NO_REFERENCE_SENSITIVITY", "pairs": [],
            "scope": "DeltaJ RMS amplitude only; it does not establish thermal-conductivity equivalence"}
    }
    ps_per_native = NATIVE_TIME_PS
    dt_ps = float(manifest["time_interval_ps"])
    for index, (ref, loaded) in enumerate(zip(manifest["references"], loaded_references)):
        delta_j = delta_series[index]
        prefix = f"reference_{index}"
        terms = {"jc_dj": _correlate(jc, delta_j, max_lag),
                 "dj_jc": _correlate(delta_j, jc, max_lag),
                 "dj_dj": _correlate(delta_j, delta_j, max_lag)}
        ja = jc + delta_j
        decomposed_ja_ja = jc_jc + terms["jc_dj"] + terms["dj_jc"] + terms["dj_dj"]
        terms["ja_ja"] = _correlate(ja, ja, max_lag)
        decomposition_residual = terms["ja_ja"] - decomposed_ja_ja
        for name, values in terms.items():
            arrays[f"{prefix}_{name}"] = values
        arrays[f"{prefix}_ja_ja_decomposed"] = decomposed_ja_ja
        arrays[f"{prefix}_ja_ja_decomposition_residual"] = decomposition_residual
        arrays[f"{prefix}_delta_j_native"] = delta_j
        arrays[f"{prefix}_ja_native"] = ja
        hac = np.sum(0.5 * (terms["ja_ja"][:-1] + terms["ja_ja"][1:]), axis=0) * \
            dt_ps / (ps_per_native * ps_per_native)
        error_by_lag = [{"lag_index": lag,
            "maximum_absolute_component_error": float(np.max(np.abs(decomposition_residual[lag]))),
            "residual_3x3": decomposition_residual[lag].tolist()} for lag in range(len(decomposition_residual))]
        item = {"id": ref["id"], "package": str(loaded["header"]["path"]),
                "package_sha256": ref["package_sha256"], "package_version": 3,
                "reference_use_class": loaded["reference_use_class"],
                "reference_numerical_status": loaded["reference_numerical_status"],
                "reference_response_status": loaded["reference_response_status"],
                "replayed_branch_identity": hashlib.sha256((manifest["branch_policy"] + "\0" +
                    manifest["trajectory_npz_sha256"].lower() + "\0" + ref["package_sha256"].lower()).encode("ascii")).hexdigest(),
                "branch_replay": "production initialize MIC, then wrapped-step MIC accumulation and 0.45 checks passed",
                "sidecar": loaded["sidecar"], "delta_j_array": f"{prefix}_delta_j_native",
                "ja_array": f"{prefix}_ja_native", "array_prefix": prefix,
                "delta_j_mean_native": np.mean(delta_j, axis=0).tolist(),
                "delta_j_variance_native_squared": np.var(delta_j, axis=0, ddof=1).tolist(),
                "delta_j_rms_native": [_rms(delta_j[:, axis]) for axis in range(3)],
                "hac_integral_eV2_A2_per_ps": hac.tolist(),
                "ja_correlation_decomposition_max_absolute_error": float(np.max(np.abs(decomposition_residual))),
                "ja_correlation_decomposition_error_by_lag_all_nine_components": error_by_lag}
        report["references"].append(item)

    if len(delta_series) >= 2:
        absolute = float(manifest["tolerances"]["reference_sensitivity_absolute_native"])
        relative = float(manifest["tolerances"]["reference_sensitivity_relative"])
        near_zero = float(manifest["tolerances"]["delta_j_near_zero_rms_native"])
        comparisons = []
        for left in range(len(delta_series)):
            for right in range(left + 1, len(delta_series)):
                difference = _rms(delta_series[left] - delta_series[right])
                scale = max(_rms(delta_series[left]), _rms(delta_series[right]))
                use_absolute = scale <= near_zero
                ratio = None if use_absolute else difference / scale
                passed = difference <= absolute if use_absolute else ratio <= relative
                comparisons.append({"left": manifest["references"][left]["id"],
                    "right": manifest["references"][right]["id"], "rms_difference_native": difference,
                    "metric": "DeltaJ RMS amplitude",
                    "larger_reference_rms_native": scale,
                    "criterion": "absolute_near_zero" if use_absolute else "relative",
                    "absolute_tolerance_native": absolute,
                    "relative_tolerance": relative, "relative_difference": ratio,
                    "passed": bool(passed)})
        report["reference_sensitivity"] = {"status": "PASS" if all(x["passed"] for x in comparisons) else "FAIL",
            "pairs": comparisons,
            "scope": "DeltaJ RMS amplitude only; it does not establish thermal-conductivity equivalence"}
    report["arrays"] = {"jc_jc": "jc_jc", "lag_ps": "lags_ps"}
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, arrays = evaluate_manifest(args.manifest)
        prefix = args.output_prefix.resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(prefix) + ".npz", **arrays)
        report["numeric_arrays_npz"] = str(prefix) + ".npz"
        Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"J_A material evaluation written: {prefix}.json and {prefix}.npz; status={report['status']}; "
              f"reference sensitivity={report['reference_sensitivity']['status']} (DeltaJ RMS only; not kappa equivalence)")
    except (OSError, ValueError, MemoryError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
