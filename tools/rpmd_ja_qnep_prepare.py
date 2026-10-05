#!/usr/bin/env python3
"""Prepare a validated v3 qNEP RPMD-JA block reference from raw finite differences."""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

RAW_MAGIC = b"GPJQRAW\0"
RAW_LAYOUT = b"xyz_soa;derivative_input_rows_output_columns\0"
MAGIC = b"GPUMDJA\0"
UNITS = b"position:A;energy:eV;mass:amu;temperature:K;time:native\0"
LAYOUT = b"xyz_soa;native_reference_transport;block_tiles_128;D_then_Bt;rowmajor_output_input\0"
ENDIAN = 0x01020304
TILE = 128
HBAR, KB = 6.465412e-2, 8.617343e-5
MAX_ADDITIVE_DENSE_ELEMENTS = 4_000_000

_FITTER_SPEC = importlib.util.spec_from_file_location(
    "_rpmd_ja_fit_reference", Path(__file__).with_name("rpmd_ja_fit_reference.py"))
_FITTER = importlib.util.module_from_spec(_FITTER_SPEC)
_FITTER_SPEC.loader.exec_module(_FITTER)


def read_additive(path, expected_n=None):
    path = Path(path)
    size = path.stat().st_size
    if size < 88 or size > 64 * 1024 * 1024:
        raise ValueError("GPJAADD1 package must be between 88 bytes and 64 MiB")
    with path.open("rb") as f:
        header = _exact(f, 32)
        magic, version, endian, n, beads, _source = struct.unpack("<8sIIiiQ", header)
        if magic != b"GPJAADD1" or version != 1 or endian != ENDIAN or n < 2 or beads < 1 or \
           (expected_n is not None and n != expected_n):
            raise ValueError("GPJAADD1 magic, version, dimensions, or atom count do not match")
        _exact(f, 48)
        for _ in range(n):
            (z,) = struct.unpack("<i", _exact(f, 4))
            if z < 0 or z > 3*n:
                raise ValueError("invalid GPJAADD1 local coordination")
            remaining = size - f.tell()
            need = 16*z + 8*(3*z) + 8*(3*z)*(3*z)
            if need > remaining:
                raise ValueError("truncated GPJAADD1 site block")
            f.seek(need, os.SEEK_CUR)
        if f.tell() != size:
            raise ValueError("trailing data in GPJAADD1 package")
    return _FITTER.read_additive(path)


def prepare_additive(pack, positions, cell):
    return _FITTER.assemble_additive(pack, positions, cell)


def _exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated qNEP raw file")
    return value


def _fnv64(path):
    h = 14695981039346656037
    with path.open("rb") as f:
        while data := f.read(1 << 20):
            for b in data:
                h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def _available_memory():
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]
        s = Status(); s.length = ctypes.sizeof(s)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return int(s.available)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _read_raw(path, additive=False):
    path = Path(path).resolve()
    with path.open("rb") as f:
        if _exact(f, 8) != RAW_MAGIC:
            raise ValueError("input is not a qNEP raw reference")
        ver, endian, n, d = struct.unpack("<IIii", _exact(f, 16))
        if ver not in (1, 2, 3) or endian != ENDIAN or n < 2 or d != 3 * n:
            raise ValueError("unsupported qNEP raw version or dimensions")
        if additive and d * d > MAX_ADDITIVE_DENSE_ELEMENTS:
            raise MemoryError(f"additive dense assembly refuses d*d={d*d} above {MAX_ADDITIVE_DENSE_ELEMENTS}")
        temp, step, model_fp, config_fp = struct.unpack("<ddQQ", _exact(f, 32))
        charge, pppm, spacing = struct.unpack("<iid", _exact(f, 16))
        if _exact(f, len(RAW_LAYOUT)) != RAW_LAYOUT:
            raise ValueError("unsupported qNEP raw layout")
        vb, cb, kb = d * n * 8, 3 * d * d * 8, d * d * 8
        start = f.tell() + 18 * 8 + 12 + n * 4 + n * 8 + d * 8 + 8 + n * 8 + d * 8 + 9 * n * 8
        expected_size = start + 2 * vb + 2 * cb + kb + 18 * 8
        if path.stat().st_size != expected_size:
            raise ValueError("qNEP raw file has an invalid length")
        cell = np.frombuffer(_exact(f, 18 * 8), "<f8").copy()
        pbc = np.frombuffer(_exact(f, 12), "<i4").copy()
        types = np.frombuffer(_exact(f, n * 4), "<i4").copy()
        masses = np.frombuffer(_exact(f, n * 8), "<f8").copy()
        positions = np.frombuffer(_exact(f, d * 8), "<f8").copy()
        (energy0,) = struct.unpack("<d", _exact(f, 8))
        site_energy = np.frombuffer(_exact(f, n * 8), "<f8").copy()
        force = np.frombuffer(_exact(f, d * 8), "<f8").copy()
        virial = np.frombuffer(_exact(f, 9 * n * 8), "<f8").copy()
        start = f.tell()
    coarse_v = start + vb + cb + kb
    coarse_c = coarse_v + vb
    footer = coarse_c + cb
    if charge not in (1, 2) or pppm != 1 or not np.isfinite([temp, step, spacing]).all() or \
       temp <= 0 or step <= 0 or spacing <= 0 or config_fp == 0:
        raise ValueError("this tool supports qNEP charge 1/2 with PPPM only")
    if not np.isfinite(cell).all() or not np.isfinite(positions).all() or not np.isfinite(masses).all() or \
       not np.isfinite(force).all() or not np.isfinite(virial).all() or not np.isfinite(site_energy).all() or \
       not np.isfinite(energy0) or abs(float(site_energy.sum())-energy0) > 1e-9*max(1.0,abs(energy0)) or \
       np.any(masses <= 0) or np.any(pbc != 1):
        raise ValueError("non-finite or invalid raw geometry, mass, force, or virial")
    return dict(path=path, raw_version=ver, n=n, d=d, temperature=temp, step=step, model_fp=model_fp,
                config_fp=config_fp, charge=charge, spacing=spacing, cell=cell, pbc=pbc,
                types=types, masses=masses, positions=positions, energy0=energy0,
                site_energy=site_energy, force=force, virial=virial, start=start,
                vb=vb, cb=cb, kb=kb, coarse_v=coarse_v, coarse_c=coarse_c, footer=footer)


def _kernel(path):
    t = Path(path).read_text(encoding="ascii").split(); i = iter(t)
    def field(name):
        if next(i, None) != name: raise ValueError(f"kernel table expected {name}")
    field("GPUMDJA_KERNEL")
    if int(next(i)) != 1: raise ValueError("unsupported kernel table version")
    field("U"); u = float(next(i)); field("degree"); degree = int(next(i))
    field("P_rank"); pr = int(next(i)); field("Q_rank"); qr = int(next(i))
    field("P_error"); pe = float(next(i)); field("Q_error"); qe = float(next(i))
    field("P_S2"); ps = float(next(i)); field("Q_S2"); qs = float(next(i))
    if not 0 <= degree <= 512 or not 0 < pr <= degree + 1 or not 0 < qr <= degree + 1:
        raise ValueError("invalid kernel ranks or degree")
    def array(name, count):
        field(name); return np.fromiter((float(next(i)) for _ in range(count)), np.float64, count)
    pv = array("P_values", pr); pvec = array("P_vectors", (degree + 1) * pr)
    qv = array("Q_values", qr); qvec = array("Q_vectors", (degree + 1) * qr)
    field("END")
    arrays=(pv,pvec,qv,qvec)
    if next(i, None) is not None or not np.isfinite([u, pe, qe, ps, qs]).all() or u <= 0 or \
       min(pe,qe,ps,qs)<0 or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("malformed kernel table")
    return dict(u=u, degree=degree, pr=pr, qr=qr, error=(pe, qe), s2=(ps, qs),
                pv=pv, pvec=pvec, qv=qv, qvec=qvec)


def build_native_transport(v, c, coarse_v, coarse_c, masses, n, alpha, fine_out, coarse_out):
    """Build H^T rows r, columns c from row-major input-coordinate derivatives."""
    d = 3 * n
    fe, fec = -v.sum(axis=1), -coarse_v.sum(axis=1)
    for c0 in range(0, d, TILE):
        c1 = min(c0 + TILE, d)
        hf = np.asarray(c[c0:c1]).copy()
        hc = np.asarray(coarse_c[c0:c1]).copy()
        for cidx in range(c0, c1):
            a, nu = cidx % n, cidx // n
            if nu == alpha:
                hf[cidx - c0] -= v[:, a]
                hc[cidx - c0] -= coarse_v[:, a]
                selected = (np.arange(d) % n) == a
                hf[cidx - c0, selected] -= fe[selected]
                hc[cidx - c0, selected] -= fec[selected]
        fine_out[:, c0:c1] = hf.T
        coarse_out[:, c0:c1] = hc.T
    mass_coord = np.tile(masses, 3)
    for r0 in range(0, d, TILE):
        r1 = min(r0 + TILE, d)
        scale = np.sqrt(mass_coord[r0:r1, None] * mass_coord[None, :])
        fine_out[r0:r1] /= scale
        coarse_out[r0:r1] /= scale
    if hasattr(fine_out, "flush"): fine_out.flush()
    if hasattr(coarse_out, "flush"): coarse_out.flush()
    diff2 = norm2 = 0.0
    for r0 in range(0, d, TILE):
        r1 = min(r0 + TILE, d)
        delta = np.asarray(fine_out[r0:r1]) - np.asarray(coarse_out[r0:r1])
        cur = np.asarray(fine_out[r0:r1])
        diff2 += float(np.sum(delta * delta)); norm2 += float(np.sum(cur * cur))
    return math.sqrt(diff2 / (d * d)), math.sqrt(diff2 / max(norm2, 1e-300))


def _tile_factor(a, tol, lossless, symmetric_diagonal=False):
    if lossless: return 0, np.asarray(a).ravel().copy(), np.empty(0), 0.0
    if symmetric_diagonal:
        w, q = np.linalg.eigh(0.5 * (a + a.T))
        order = np.argsort(np.abs(w))[::-1]; w, q = w[order], q[:, order]
        values = w
    else:
        q, values, vt = np.linalg.svd(a, full_matrices=False)
    norm2 = float(np.dot(values, values)); tails = np.cumsum(values[::-1] ** 2)[::-1]
    rank = len(values)
    for r in range(1, len(values) + 1):
        err = math.sqrt(float(tails[r]) if r < len(values) else 0.0)
        if err <= tol * max(math.sqrt(norm2), 1e-300): rank = r; break
    if symmetric_diagonal:
        if 2 * rank * a.shape[0] >= a.size: return 0, a.ravel().copy(), np.empty(0), 0.0
        root = np.sqrt(np.abs(values[:rank])); left = q[:, :rank] * root
        right = np.sign(values[:rank, None]) * root[:, None] * q[:, :rank].T
    else:
        if rank * (a.shape[0] + a.shape[1]) >= a.size: return 0, a.ravel().copy(), np.empty(0), 0.0
        root = np.sqrt(values[:rank]); left = q[:, :rank] * root
        right = root[:, None] * vt[:rank]
    err = math.sqrt(float(tails[rank]) if rank < len(values) else 0.0)
    return rank, left.ravel(), right.ravel(), err


def _write_tile(out, row, col, rows, cols, rank, left, right):
    left, right = np.asarray(left, dtype="<f8"), np.asarray(right, dtype="<f8")
    out.write(struct.pack("<5iQQ", row, col, rows, cols, rank, left.size, right.size))
    left.tofile(out); right.tofile(out)


def _write_tiles(path, matrix, symmetric, tol, lossless, reconstructed=None):
    d = matrix.shape[0]; grid = (d + TILE - 1) // TILE
    error2 = norm2 = 0.0
    row_bound = np.zeros(d, dtype=np.float64)
    def add_bound(r0, r1, c0, c1, rank, left, right):
        if rank == 0:
            row_bound[r0:r1] += np.sum(np.abs(left.reshape(r1-r0, c1-c0)), axis=1)
        else:
            l = left.reshape(r1-r0, rank)
            rr = right.reshape(rank, c1-c0)
            row_bound[r0:r1] += np.sum(np.abs(l) * np.sum(np.abs(rr), axis=1)[None, :], axis=1)
    with path.open("wb") as out:
        out.write(struct.pack("<iQ", TILE, grid * grid))
        for bi in range(grid):
            r0, r1 = bi * TILE, min((bi + 1) * TILE, d)
            for bj in range(bi if symmetric else 0, grid):
                c0, c1 = bj * TILE, min((bj + 1) * TILE, d)
                a = np.asarray(matrix[r0:r1, c0:c1]).copy()
                rank, left, right, error = _tile_factor(a, tol, lossless, symmetric and bi == bj)
                norm = float(np.linalg.norm(a)); multiplier = 2 if symmetric and bi != bj else 1
                error2 += multiplier * error * error; norm2 += multiplier * norm * norm
                _write_tile(out, r0, c0, r1-r0, c1-c0, rank, left, right)
                add_bound(r0, r1, c0, c1, rank, left, right)
                if reconstructed is not None:
                    dense = a if rank == 0 else left.reshape(r1-r0, rank) @ right.reshape(rank, c1-c0)
                    reconstructed[r0:r1, c0:c1] = dense
                if symmetric and bi != bj:
                    if rank == 0:
                        mirror_left, mirror_right = a.T.ravel(), np.empty(0)
                    else:
                        mirror_left = right.reshape(rank, c1-c0).T
                        mirror_right = left.reshape(r1-r0, rank).T
                    _write_tile(out, c0, r0, c1-c0, r1-r0, rank, mirror_left, mirror_right)
                    add_bound(c0, c1, r0, r1, rank, mirror_left, mirror_right)
                    if reconstructed is not None: reconstructed[c0:c1, r0:r1] = dense.T
    if reconstructed is not None: reconstructed.flush()
    return math.sqrt(error2 / max(norm2, 1e-300)), float(np.max(row_bound))


def _project_translation(dmat, t):
    d = dmat.shape[0]; n = len(t)
    basis = np.zeros((d, 3))
    for a in range(3): basis[a*n:(a+1)*n, a] = t
    dt = np.empty((d, 3))
    for r0 in range(0, d, TILE): dt[r0:r0+TILE] = dmat[r0:r0+TILE] @ basis
    center = basis.T @ dt
    changed = norm = 0.0
    for r0 in range(0, d, TILE):
        r1 = min(r0 + TILE, d); b = basis[r0:r1]
        delta = b @ dt.T + dt[r0:r1] @ basis.T - b @ center @ basis.T
        old = np.asarray(dmat[r0:r1]).copy()
        changed += float(np.sum(delta * delta)); norm += float(np.sum(old * old))
        dmat[r0:r1] = old - delta
    dmat.flush()
    return math.sqrt(changed / max(norm, 1e-300))


def _translation_residual(matrix, t):
    d=matrix.shape[0]; n=len(t); total=0.0; norm=0.0
    for r0 in range(0,d,TILE):
        r1=min(r0+TILE,d); block=np.asarray(matrix[r0:r1])
        norm+=float(np.sum(block*block))
        for alpha in range(3):
            residual=block[:,alpha*n:(alpha+1)*n]@t
            total+=float(np.dot(residual,residual))
    return math.sqrt(total/max(norm,1e-300))


def _eigenvalues(matrix, translation, available):
    d = matrix.shape[0]
    need = 3 * d * d * 8 + 256 * 2**20
    if available is not None and need > available:
        raise MemoryError(f"translation-complement eigensolver preflight needs {need/2**30:.2f} GiB, available {available/2**30:.2f} GiB")
    n = len(translation)
    # Each Householder maps the known mass-weighted translation vector to its axis' first coordinate.
    householder = np.zeros((d, 3), dtype=np.float64)
    for axis in range(3):
        w = translation.copy(); w[0] -= 1.0
        norm = np.linalg.norm(w)
        if norm == 0.0:
            continue
        householder[axis*n:(axis+1)*n, axis] = w / norm
    a = np.array(matrix, dtype=np.float64, order="C", copy=True)
    dw = np.empty((d, 3), dtype=np.float64)
    for r0 in range(0, d, TILE):
        r1 = min(r0 + TILE, d)
        dw[r0:r1] = a[r0:r1] @ householder
    middle = householder.T @ dw
    for r0 in range(0, d, TILE):
        r1 = min(r0 + TILE, d)
        w = householder[r0:r1]
        block = np.asarray(a[r0:r1]).copy()
        block -= 2.0 * (w @ dw.T) + 2.0 * (dw[r0:r1] @ householder.T)
        block += 4.0 * (w @ middle @ householder.T)
        a[r0:r1] = block
    keep = np.ones(d, dtype=bool)
    keep[[0, n, 2*n]] = False
    indices = np.flatnonzero(keep)
    for row in range(d):
        a[row, :d-3] = a[row, indices]
    for target, source in enumerate(indices):
        if target != source:
            a[target, :d-3] = a[source, :d-3]
    physical = a[:d-3, :d-3]
    values = np.linalg.eigvalsh(physical)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError(f"translation-orthogonal qNEP D has a nonpositive physical eigenvalue: {values[:5]}")
    return values


def _internal_dynamical(matrix, translation):
    d = matrix.shape[0]; n = len(translation)
    householder = np.zeros((d, 3))
    for axis in range(3):
        w = translation.copy(); w[0] -= 1.0
        norm = np.linalg.norm(w)
        if norm:
            householder[axis*n:(axis+1)*n, axis] = w / norm
    a = np.array(matrix, dtype=np.float64, copy=True)
    dw = a @ householder; middle = householder.T @ dw
    a -= 2.0 * (householder @ dw.T) + 2.0 * (dw @ householder.T)
    a += 4.0 * (householder @ middle @ householder.T)
    indices = np.delete(np.arange(d), [0, n, 2*n])
    return a[np.ix_(indices, indices)]


def _read_matrix(path, offset, shape):
    return np.memmap(path, dtype="<f8", mode="r", offset=offset, shape=shape)


def _prepare_impl(raw_path, kernel_path, output, lossless, tile_tol, soft_tol, additive_path=None):
    started = time.perf_counter()
    raw_path, kernel_path, output = map(lambda x: Path(x).resolve(), (raw_path, kernel_path, output))
    sidecar = Path(str(output) + ".stability")
    if output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite final qNEP reference or stability sidecar")
    m = _read_raw(raw_path, additive_path is not None); n, d = m["n"], m["d"]
    stats = np.fromfile(raw_path, dtype="<f8", count=18, offset=m["footer"])
    if len(stats) != 18 or not np.isfinite(stats).all():
        raise ValueError("raw reference lacks finite-difference diagnostics")
    (grad_rel, grad_abs, v_rel, v_abs, k_rel, k_abs, cx, cy, cz, cax, cay, caz,
     force_max, force_rms, second_err, second_conv, step, _) = stats
    if step != m["step"] or np.any(stats < 0):
        raise ValueError("raw diagnostic footer step or nonnegative fields do not match the raw header")
    raw_version = m["raw_version"]
    if stats[17] != raw_version or raw_version not in (1, 2, 3) or stats[1] > 1e-4 or \
       (raw_version == 1 and max(stats[i] for i in (2, 4, 6, 7, 8, 14, 15)) > 0.05) or \
       (raw_version >= 2 and (stats[3] > 1e-4 or max(stats[i] for i in (4, 6, 7, 8, 14, 15)) > 0.05)):
        raise ValueError("qNEP energy/force finite-difference checks failed")
    kernel = _kernel(kernel_path)
    additive = None
    assembled = None
    if additive_path is not None:
        if d * d > MAX_ADDITIVE_DENSE_ELEMENTS:
            raise MemoryError(f"additive dense assembly refuses d*d={d*d} above {MAX_ADDITIVE_DENSE_ELEMENTS}")
        additive = read_additive(additive_path, n)
        if additive["n"] != n or additive["source_fingerprint"] != _fnv64(raw_path):
            raise ValueError("additive package atom count or raw source fingerprint does not match")
        if additive["temperature"] != m["temperature"]:
            raise ValueError("additive package temperature does not match raw reference")
        assembled = prepare_additive(additive, m["positions"].reshape(3, n).T, m["cell"][:9].reshape(3, 3))
        raw_v = _read_matrix(raw_path, m["start"], (d,n))
        raw_gradient = np.asarray(raw_v).sum(axis=1).reshape(3,n).T
        assembled_gradient = assembled["linear_gradient"]
        residual = raw_gradient + assembled_gradient
        if not np.isfinite(raw_gradient).all() or not np.isfinite(assembled_gradient).all() or not np.isfinite(residual).all():
            raise ValueError("additive linear gradient check produced non-finite values")
        gradient_error = _FITTER._stable_norm(residual)
        raw_gradient_norm = _FITTER._stable_norm(raw_gradient)
        if not math.isfinite(gradient_error) or not math.isfinite(raw_gradient_norm) or \
           gradient_error > 1e-8 * max(1.0, raw_gradient_norm):
            raise ValueError("additive linear term does not cancel raw reference gradient")
    work = Path(str(output) + ".work"); work.mkdir(parents=True, exist_ok=False)
    work_maps = []
    try:
        k = _read_matrix(raw_path, m["start"] + m["vb"] + m["cb"], (d, d))
        dmat = np.memmap(work / "D.f64", dtype="<f8", mode="w+", shape=(d, d))
        work_maps.append(dmat)
        mc = np.tile(m["masses"], 3)
        asym2 = norm2 = 0.0
        for r0 in range(0, d, TILE):
            r1 = min(r0 + TILE, d)
            for c0 in range(0, d, TILE):
                c1 = min(c0 + TILE, d)
                a = np.asarray(k[c0:c1, r0:r1]).T
                b = np.asarray(k[r0:r1, c0:c1])
                scale = np.sqrt(mc[r0:r1, None]*mc[None,c0:c1])
                aw, bw = a / scale, b / scale
                asym2 += float(np.sum((aw-bw)**2)); norm2 += float(np.sum((0.5*(aw+bw))**2))
                dmat[r0:r1, c0:c1] = 0.5*(a+b) / scale
                if not np.isfinite(dmat[r0:r1, c0:c1]).all(): raise ValueError("mass-weighted D contains non-finite values")
        asym = math.sqrt(asym2 / max(norm2, 1e-300))
        if asym > 0.05: raise ValueError(f"force Jacobian antisymmetry {asym:.3e} exceeds 5e-2")
        translation = np.sqrt(m["masses"] / np.sum(m["masses"]))
        raw_translation_residual = _translation_residual(dmat, translation)
        if raw_translation_residual > 0.05:
            raise ValueError(f"unprojected D translation residual {raw_translation_residual:.3e} exceeds 5e-2")
        projection = _project_translation(dmat, translation)
        if projection > 0.05:
            raise ValueError(f"strict translation projection changes D by {projection:.3e} (>5e-2)")
        if assembled is not None:
            kadd = assembled["Kadd"]
            dmat[:] += kadd / np.sqrt(mc[:, None] * mc[None, :])
            dmat.flush()
            if not np.isfinite(dmat).all(): raise ValueError("additive mass-weighted D contains non-finite values")
        trans_resid = _translation_residual(dmat, translation)
        if trans_resid > 1e-8: raise ValueError(f"D translation residual {trans_resid:.3e} exceeds 1e-8")
        dense_spectral = 0.0
        for r0 in range(0,d,TILE):
            r1=min(r0+TILE,d)
            dense_spectral=max(dense_spectral,float(np.max(np.sum(np.abs(dmat[r0:r1]),axis=1))))
        available = _available_memory(); eig0 = _eigenvalues(dmat, translation, available)

        v = _read_matrix(raw_path, m["start"], (d, n))
        vc = _read_matrix(raw_path, m["coarse_v"], (d, n))
        b_fine, b_coarse, transport_abs, transport_rel = [], [], [], []
        for alpha in range(3):
            c = _read_matrix(raw_path, m["start"]+m["vb"]+alpha*d*d*8, (d,d))
            cc = _read_matrix(raw_path, m["coarse_c"]+alpha*d*d*8, (d,d))
            for r0 in range(0,d,TILE):
                r1=min(r0+TILE,d)
                if not np.isfinite(v[r0:r1]).all() or not np.isfinite(vc[r0:r1]).all() or \
                   not np.isfinite(c[r0:r1]).all() or not np.isfinite(cc[r0:r1]).all():
                    raise ValueError("raw site derivative matrices contain non-finite values")
            bf = np.memmap(work/f"Bt{alpha}.f64", dtype="<f8", mode="w+", shape=(d,d))
            work_maps.append(bf)
            bc = np.memmap(work/f"Btc{alpha}.f64", dtype="<f8", mode="w+", shape=(d,d))
            work_maps.append(bc)
            ae,re = build_native_transport(v,c,vc,cc,m["masses"],n,alpha,bf,bc)
            if not math.isfinite(re) or re > 0.05:
                raise ValueError(f"baseline native H h/h2 error exceeds tolerance on axis {alpha}: {re}")
            if assembled is not None:
                htranspose = assembled["Hadd"][alpha].T
                mass_scale = np.sqrt(np.tile(m["masses"], 3)[:, None] * np.tile(m["masses"], 3)[None, :])
                bf[:] += htranspose / mass_scale
                bc[:] += htranspose / mass_scale
                if not np.isfinite(bf).all() or not np.isfinite(bc).all():
                    raise ValueError("additive mass-weighted H contains non-finite values")
                bf.flush(); bc.flush()
            b_fine.append(bf); b_coarse.append(bc); transport_abs.append(ae); transport_rel.append(re)
        if not np.isfinite(transport_rel).all() or max(transport_rel) > 0.05:
            raise ValueError(f"native H h/h2 error exceeds tolerance: {transport_rel}")

        if assembled is not None:
            lossless = True
        dcomp = np.memmap(work/"Dcompressed.f64", dtype="<f8", mode="w+", shape=(d,d))
        work_maps.append(dcomp)
        matrices = [dmat, *b_fine]; sections=[]; residuals=[]; d_bound = dense_spectral
        for j, mat in enumerate(matrices):
            path = work/f"M{j}.tiles"; sections.append(path)
            residual, bound = _write_tiles(path,mat,j==0,tile_tol,lossless,dcomp if j==0 else None)
            residuals.append(residual)
            if j == 0: d_bound = bound
        soft_error=0.0
        if not lossless:
            try:
                compressed_translation = _translation_residual(dcomp, translation)
                if compressed_translation > 1e-8:
                    raise ValueError(f"compressed D translation residual {compressed_translation:.3e} exceeds 1e-8")
                eigc=_eigenvalues(dcomp,translation,available)
                soft_error=float(np.max(np.abs(eigc-eig0)/np.maximum(np.abs(eig0),1e-300)))
                if soft_error>soft_tol: raise ValueError(f"softmode relative error {soft_error:.3e} > {soft_tol:.3e}")
                if d_bound > dense_spectral * 4:
                    raise ValueError("compressed D factor row bound is excessively loose")
            except (ValueError,np.linalg.LinAlgError) as exc:
                print(f"compression failed stability check; returning to lossless blocks: {exc}",file=sys.stderr)
                lossless=True
                for p in sections:p.unlink()
                sections=[];residuals=[];dcomp[:]=dmat
                for j,mat in enumerate(matrices):
                    p=work/f"M{j}.tiles";sections.append(p)
                    residual,bound=_write_tiles(p,mat,j==0,tile_tol,True,dcomp if j==0 else None)
                    residuals.append(residual)
                    if j==0:d_bound=bound
                soft_error=0.0
        spectral=d_bound
        required=HBAR/(KB*m["temperature"])*math.sqrt(spectral)
        if required>kernel["u"]*(1+32*np.finfo(float).eps):
            if lossless:
                raise ValueError(f"kernel U={kernel['u']:.8g} does not cover lossless block bound {required:.8g}")
            print("compressed D factor bound exceeds kernel coverage; retrying with lossless blocks", file=sys.stderr)
            for p in sections: p.unlink()
            sections=[]; residuals=[]; dcomp[:]=dmat
            for j,mat in enumerate(matrices):
                p=work/f"M{j}.tiles"; sections.append(p)
                residual,bound=_write_tiles(p,mat,j==0,tile_tol,True,dcomp if j==0 else None)
                residuals.append(residual)
                if j==0:d_bound=bound
            lossless=True; soft_error=0.0; spectral=d_bound
            required=HBAR/(KB*m["temperature"])*math.sqrt(spectral)
            if required>kernel["u"]*(1+32*np.finfo(float).eps):
                raise ValueError(f"kernel U={kernel['u']:.8g} does not cover lossless block bound {required:.8g}")

        policy = (f"native_reference_transport;finite_temperature_additive_v1;beads={additive['beads']};derivative={raw_version}"
                  if additive is not None else "native_reference_transport")
        tmp=work/"reference.tmp"
        with tmp.open("xb") as out:
            out.write(MAGIC);out.write(struct.pack("<IIiddQ",3,ENDIAN,n,m["temperature"],step,m["model_fp"]))
            out.write(UNITS);out.write(LAYOUT);out.write(np.asarray(m["cell"][:9],"<f8").tobytes())
            out.write(np.asarray(m["pbc"],"<i4").tobytes());out.write(np.asarray(m["types"],"<i4").tobytes())
            out.write(np.asarray(m["masses"],"<f8").tobytes());out.write(np.asarray(m["positions"],"<f8").tobytes())
            policy_bytes=policy.encode("ascii")
            out.write(struct.pack("<Qiidi",m["config_fp"],m["charge"],1,m["spacing"],len(policy_bytes)));out.write(policy_bytes)
            diag=[grad_rel,k_rel,asym,max(second_err,second_conv),force_max,projection,grad_abs,k_abs,
                  cax,cay,caz,*transport_abs,*transport_rel,*residuals]
            if len(diag)!=21:raise AssertionError("v3 diagnostic layout mismatch")
            out.write(np.asarray(diag,"<f8").tobytes());out.write(struct.pack("<dd",spectral,kernel["u"]))
            out.write(np.asarray(kernel["error"],"<f8").tobytes());out.write(np.asarray(kernel["s2"],"<f8").tobytes())
            out.write(struct.pack("<iii",kernel["degree"],kernel["pr"],kernel["qr"]))
            for key in ("pv","qv","pvec","qvec"):out.write(np.asarray(kernel[key],"<f8").tobytes())
            for p in sections:
                with p.open("rb") as f:shutil.copyfileobj(f,out,4*1024*1024)
            out.write(struct.pack("<i",1));out.flush();os.fsync(out.fileno())
        min_pivot=float(eig0[0])
        eta = 0.0
        if additive is not None:
            stored_translation_residual = _translation_residual(dcomp, translation)
            if not math.isfinite(stored_translation_residual) or stored_translation_residual > 1e-8:
                raise ValueError(f"stored D translation residual {stored_translation_residual:.3e} exceeds 1e-8")
            trans_resid = stored_translation_residual
            internal = _internal_dynamical(dcomp, translation)
            shifted = internal - 0.5 * additive["epsilon"] * np.eye(d-3)
            factor = np.linalg.cholesky(shifted)
            eta = float(np.linalg.norm(shifted - factor @ factor.T, "fro"))
            if not math.isfinite(eta) or eta >= 0.5 * additive["epsilon"]:
                raise ValueError("additive shifted full-Frobenius certificate eta is not below epsilon/2")
            min_pivot = float(np.min(np.diag(factor)))
        sidecar_tmp=work/"stability.tmp"
        with sidecar_tmp.open("x",encoding="ascii",newline="\n") as f:
            f.write("GPUMDJA_QNEP_STABILITY 4\n" if additive is not None else "GPUMDJA_QNEP_STABILITY 1\n")
            f.write(f"fingerprint {_fnv64(tmp):016x}\nconfig_fingerprint {m['config_fp']:016x}\n")
            f.write(f"atoms {n}\n")
            if additive is not None:
                f.write("derivative_policy " + policy + "\n")
                f.write("certificate additive_shifted_frobenius_v1\n")
                f.write(f"epsilon_num {additive['epsilon']:.17g}\nreconstruction_bound {eta:.17g}\n")
                f.write(f"minimum_cholesky_pivot {min_pivot:.17g}\nrelative_operator_bound 0\n")
            else:
                f.write(f"minimum_positive_eigenvalue {min_pivot:.17g}\n")
            f.write(f"translation_residual {trans_resid:.17g}\n")
            f.write(f"reconstruction_residual {max(residuals):.17g}\nsoftmode_relative_error {soft_error:.17g}\n")
        os.replace(tmp,output)
        os.replace(sidecar_tmp,sidecar)
        print(f"qNEP v3 written: N={n}, D={d}, D bound={spectral:.8g}, H h/h2={transport_rel}, "
              f"D projection={projection:.3e}, softmode error={soft_error:.3e}, "
              f"storage={'lossless' if lossless else 'compressed'}")
        print(f"raw={raw_path.stat().st_size/2**30:.3f} GiB final={output.stat().st_size/2**30:.3f} GiB; "
              f"temporary scratch observed={sum(p.stat().st_size for p in work.rglob('*') if p.is_file())/2**30:.3f} GiB; "
              f"eigensolver preflight available={None if available is None else f'{available/2**30:.2f} GiB'} "
              f"(requires about {3*d*d*8/2**30+0.25:.2f} GiB); "
              f"prepare time={time.perf_counter()-started:.1f}s (observed)")
    except Exception:
        raise
    finally:
        for matrix in work_maps:
            mapping = getattr(matrix, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()


def _prepare(raw_path, kernel_path, output, lossless, tile_tol, soft_tol, additive_path=None):
    output = Path(output).resolve()
    sidecar = Path(str(output) + ".stability")
    work = Path(str(output) + ".work")
    had_output, had_sidecar, had_work = output.exists(), sidecar.exists(), work.exists()
    try:
        return _prepare_impl(raw_path, kernel_path, output, lossless, tile_tol, soft_tol, additive_path)
    except Exception:
        if not had_output:
            output.unlink(missing_ok=True)
        if not had_sidecar:
            sidecar.unlink(missing_ok=True)
        raise
    finally:
        if not had_work and work.exists():
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path);p.add_argument("additive_pack",nargs="?",type=Path)
    p.add_argument("--kernel-table",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--lossless",action="store_true")
    p.add_argument("--block-relative-tolerance",type=float,default=1e-8)
    p.add_argument("--softmode-relative-tolerance",type=float,default=1e-2)
    a=p.parse_args(argv)
    if not 0<a.block_relative_tolerance<=1e-8 or not 0<a.softmode_relative_tolerance<=1e-2:
        p.error("block tolerance must be in (0,1e-8]; softmode tolerance must be in (0,1e-2]")
    try:_prepare(a.raw,a.kernel_table,a.output,a.lossless,a.block_relative_tolerance,a.softmode_relative_tolerance,a.additive_pack)
    except (OSError,ValueError,MemoryError,np.linalg.LinAlgError) as exc:p.error(str(exc))


if __name__=="__main__":main()
