#!/usr/bin/env python3
"""Offline O(N^2) CPU positive-definiteness check for sparse RPMD-JA v2 references."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

MAGIC = b"GPUMDJA\0"
ENDIAN = 0x01020304
UNITS = b"position:A;energy:eV;mass:amu;temperature:K;time:native\0"
LAYOUT_V2 = b"xyz_soa;csr_output_row_input_col;runtime_translation_projection;cheb_rowmajor_degree_rank;fixed_d0_edges_v1\0"


class Reader:
    def __init__(self, path: Path):
        self.stream = path.open("rb", buffering=0)
        self.size = path.stat().st_size

    def read(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated sparse RPMD-JA file")
        return struct.unpack(fmt, data)

    def bytes(self, count: int) -> bytes:
        if count < 0 or count > self.size - self.stream.tell():
            raise ValueError("sparse RPMD-JA count exceeds remaining file bytes")
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("truncated sparse RPMD-JA file")
        return data

    def array(self, dtype, count: int) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        if count < 0 or count > (self.size - self.stream.tell()) // itemsize:
            raise ValueError("sparse RPMD-JA array count exceeds remaining file bytes")
        values = np.fromfile(self.stream, dtype=dtype, count=count)
        if values.size != count:
            raise ValueError("truncated sparse RPMD-JA file")
        return values

def read_reference(path: Path):
    r = Reader(path)
    try:
        if r.bytes(8) != MAGIC:
            raise ValueError("not a GPUMDJA reference")
        version, endian, n = r.read("<IIi")
        if version != 2 or endian != ENDIAN or n <= 1 or n > (2**31 - 1) // 3:
            raise ValueError("expected a valid sparse RPMD-JA v2 reference")
        temperature, fd_step, model_fingerprint = r.read("<ddQ")
        if r.bytes(len(UNITS)) != UNITS or r.bytes(len(LAYOUT_V2)) != LAYOUT_V2:
            raise ValueError("unsupported sparse RPMD-JA units or layout")
        cell = r.array("<f8", 9)
        pbc = r.array("<i4", 3)
        types = r.array("<i4", n)
        masses = r.array("<f8", n)
        positions = r.array("<f8", 3 * n)
        edge_policy_version, edge_count, edge_fingerprint = r.read("<iQQ")
        if edge_policy_version != 1:
            raise ValueError("unsupported fixed-edge policy version")
        d = 3 * n

        def sparse_matrix():
            (nnz,) = r.read("<Q")
            if nnz > (r.size - r.stream.tell()) // 12:
                raise ValueError("CSR nonzero count exceeds remaining reference bytes")
            rows = r.array("<u8", d + 1)
            columns = r.array("<i4", nnz)
            values = r.array("<f8", nnz)
            if rows[0] != 0 or rows[-1] != nnz or np.any(rows[1:] < rows[:-1]):
                raise ValueError("invalid CSR row offsets")
            for row in range(d):
                cols = columns[rows[row] : rows[row + 1]]
                if np.any(cols < 0) or np.any(cols >= d) or np.any(cols[1:] <= cols[:-1]):
                    raise ValueError("CSR columns are unsorted, repeated, or out of range")
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite CSR values")
            return rows, columns, values

        dynamical = sparse_matrix()
        bt = [sparse_matrix() for _ in range(3)]
        spectral_bound, kernel_u = r.read("<dd")
        kernel_error = r.array("<f8", 2)
        kernel_s2 = r.array("<f8", 2)
        degree, p_rank, q_rank = r.read("<iii")
        fd_relative_d = r.read("<d")[0]
        fd_relative_b = r.array("<f8", 3)
        if degree < 0 or degree > 512 or p_rank <= 0 or q_rank <= 0 or p_rank > degree + 1 or q_rank > degree + 1:
            raise ValueError("invalid kernel ranks/degree")
        p_values = r.array("<f8", p_rank)
        q_values = r.array("<f8", q_rank)
        p_vectors = r.array("<f8", (degree + 1) * p_rank)
        q_vectors = r.array("<f8", (degree + 1) * q_rank)
        if r.stream.tell() != r.size:
            raise ValueError("trailing bytes in sparse RPMD-JA reference")
        if not (np.all(np.isfinite(masses)) and np.all(masses > 0) and np.all(np.isfinite(positions))):
            raise ValueError("invalid masses or positions")
        if not all(np.all(np.isfinite(x)) for x in (cell, kernel_error, kernel_s2, fd_relative_b, p_values, q_values, p_vectors, q_vectors)):
            raise ValueError("non-finite sparse RPMD-JA metadata")
        result = {
            "n": n, "d": d, "temperature": temperature, "fd_step": fd_step,
            "model_fingerprint": model_fingerprint, "masses": masses, "dynamical": dynamical,
            "site_transpose": bt,
            "edge_count": edge_count, "edge_fingerprint": edge_fingerprint,
            "spectral_bound": spectral_bound, "kernel_u": kernel_u,
        }
        return result

    finally:
        r.stream.close()

def _available_memory() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def fnv64(path: Path) -> int:
    value = 14695981039346656037
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            for byte in block:
                value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def read_kernel_table(path: Path):
    fields = path.read_text(encoding="ascii").split()
    cursor = 0
    def take(label):
        nonlocal cursor
        if cursor >= len(fields) or fields[cursor] != label:
            raise ValueError(f"malformed kernel table; expected {label}")
        cursor += 1
    take("GPUMDJA_KERNEL")
    if fields[cursor] != "1": raise ValueError("unsupported kernel table version")
    cursor += 1
    take("U"); u = float(fields[cursor]); cursor += 1
    take("degree"); degree = int(fields[cursor]); cursor += 1
    take("P_rank"); p_rank = int(fields[cursor]); cursor += 1
    take("Q_rank"); q_rank = int(fields[cursor]); cursor += 1
    take("P_error"); error = [float(fields[cursor])]; cursor += 1
    take("Q_error"); error.append(float(fields[cursor])); cursor += 1
    take("P_S2"); s2 = [float(fields[cursor])]; cursor += 1
    take("Q_S2"); s2.append(float(fields[cursor])); cursor += 1
    if not math.isfinite(u) or u <= 0 or degree < 0 or degree > 512 or min(p_rank, q_rank) <= 0 or max(p_rank, q_rank) > degree + 1:
        raise ValueError("invalid kernel table bounds or dimensions")
    arrays = []
    for name, rank in (("P", p_rank), ("Q", q_rank)):
        take(f"{name}_values")
        values = np.asarray([float(x) for x in fields[cursor:cursor + rank]], dtype=np.float64)
        cursor += rank
        take(f"{name}_vectors")
        vectors = np.asarray([float(x) for x in fields[cursor:cursor + (degree + 1) * rank]], dtype=np.float64)
        cursor += (degree + 1) * rank
        arrays.append((values, vectors))
    take("END")
    if cursor != len(fields) or not all(np.all(np.isfinite(a)) for pair in arrays for a in pair):
        raise ValueError("kernel table has trailing or non-finite data")
    return {"u": u, "degree": degree, "ranks": [p_rank, q_rank], "error": error, "s2": s2, "arrays": arrays}


def rebind_kernel(reference_path: Path, table_path: Path, output_path: Path):
    if output_path.exists() or output_path.with_name(output_path.name + ".tmp").exists() or output_path.with_name(output_path.name + ".stability").exists():
        raise FileExistsError("rebound reference, temporary, or sidecar destination already exists")
    r = Reader(reference_path)
    if r.bytes(8) != MAGIC or r.read("<II")[0] != 2:
        raise ValueError("kernel rebinding requires a sparse RPMD-JA v2 reference")
    r.stream.seek(0)
    r.bytes(8); version, endian, n = r.read("<IIi")
    temperature, fd_step, model_fingerprint = r.read("<ddQ")
    if endian != ENDIAN or n <= 1 or r.bytes(len(UNITS)) != UNITS or r.bytes(len(LAYOUT_V2)) != LAYOUT_V2:
        raise ValueError("unsupported sparse RPMD-JA reference header")
    d = 3 * n
    r.bytes(9 * 8 + 3 * 4)
    r.bytes(n * 4); r.bytes(n * 8); r.bytes(d * 8)
    if r.read("<iQQ")[0] != 1:
        raise ValueError("unsupported fixed-edge policy")
    def skip_csr():
        nnz, = r.read("<Q")
        r.bytes((d + 1) * 8); r.bytes(nnz * 4); r.bytes(nnz * 8)
    for _ in range(4): skip_csr()
    prefix_end = r.stream.tell()
    spectral_bound, old_u = r.read("<dd")
    r.array("<f8", 2); r.array("<f8", 2)
    old_degree, old_p, old_q = r.read("<iii")
    fd_relative_d, = r.read("<d")
    fd_relative_b = r.array("<f8", 3)
    r.stream.close()
    table = read_kernel_table(table_path)
    tau = 6.465412e-2 / (8.617343e-5 * temperature)
    required_u = tau * math.sqrt(spectral_bound)
    if required_u > table["u"]:
        raise ValueError(f"replacement table U={table['u']:.17g} still below required U={required_u:.17g}")
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with reference_path.open("rb") as source, temporary.open("xb") as out:
            remaining = prefix_end
            while remaining:
                block = source.read(min(1 << 20, remaining))
                if not block: raise ValueError("reference truncated while rebinding kernel")
                out.write(block); remaining -= len(block)
            def write(fmt, *values): out.write(struct.pack(fmt, *values))
            def vector(array, dtype):
                arr = np.asarray(array, dtype=dtype)
                out.write(arr.tobytes())
            write("<dd", spectral_bound, table["u"])
            vector(table["error"], "<f8"); vector(table["s2"], "<f8")
            p_rank, q_rank = table["ranks"]
            write("<iii", table["degree"], p_rank, q_rank)
            write("<d", fd_relative_d); vector(fd_relative_b, "<f8")
            (p_values, p_vectors), (q_values, q_vectors) = table["arrays"]
            vector(p_values, "<f8"); vector(q_values, "<f8")
            vector(p_vectors, "<f8"); vector(q_vectors, "<f8")
            out.flush(); os.fsync(out.fileno())
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"rebound fixed D/B and edge metadata without NEP evaluation: {output_path}; required U={required_u:.9g}, table U={table['u']:.9g}; run this checker on the new file to bind its stability sidecar")


def check(path: Path, memory_limit_gib: float | None = None):
    ref = read_reference(path)
    n, d = ref["n"], ref["d"]
    # Anchor the same atom's x/y/z coordinates to remove all three translations.
    minor_index = np.asarray([axis * n + atom for axis in range(3) for atom in range(n - 1)], dtype=np.int64)
    dimension = len(minor_index)
    global_to_minor = np.full(d, -1, dtype=np.int64)
    global_to_minor[minor_index] = np.arange(dimension)
    one_matrix = dimension * dimension * 8
    csr_bytes = 0
    for row_offsets, columns, values in [ref["dynamical"]] + ref["site_transpose"]:
        csr_bytes += row_offsets.nbytes + columns.nbytes + values.nbytes
    estimated = 3 * one_matrix + csr_bytes + 256 * 1024**2  # A, Cholesky/LAPACK copies, CSR arrays and workspace reserve.
    print(f"N={n}, anchored minor={dimension}x{dimension}: one matrix {one_matrix / 1024**3:.2f} GiB; peak estimate {estimated / 1024**3:.2f} GiB")
    available = _available_memory()
    if memory_limit_gib is not None and estimated > memory_limit_gib * 1024**3:
        raise MemoryError("estimated sparse stability check exceeds --memory-limit-gib")
    if available is not None and estimated > 0.85 * available:
        raise MemoryError("estimated sparse stability check exceeds available physical memory")

    row_offsets, columns, values = ref["dynamical"]
    masses = ref["masses"]
    mass_sum = float(np.sum(masses))
    translation = np.zeros((d, 3), dtype=np.float64)
    for axis in range(3):
        translation[axis * n : (axis + 1) * n, axis] = np.sqrt(masses / mass_sum)
    d_times_t = np.zeros((d, 3), dtype=np.float64)
    keep_translation = translation[minor_index]
    a = np.zeros((dimension, dimension), dtype=np.float64)
    symmetry2 = 0.0
    norm2 = 0.0
    for row in range(d):
        begin, end = int(row_offsets[row]), int(row_offsets[row + 1])
        cols = columns[begin:end]
        vals = values[begin:end]
        d_times_t[row] = vals @ translation[cols]
        norm2 += float(vals @ vals)
        minor_row = int(global_to_minor[row])
        if minor_row >= 0:
            mapped = global_to_minor[cols]
            keep = mapped >= 0
            a[minor_row, mapped[keep]] = vals[keep]
    # Check symmetry through the CSR transpose lookup without creating a second dense matrix.
    for row in range(d):
        begin, end = int(row_offsets[row]), int(row_offsets[row + 1])
        for k in range(begin, end):
            col = int(columns[k])
            lo, hi = int(row_offsets[col]), int(row_offsets[col + 1])
            pos = int(np.searchsorted(columns[lo:hi], row)) + lo
            transpose = values[pos] if pos < hi and columns[pos] == row else 0.0
            delta = values[k] - transpose
            symmetry2 += delta * delta
    symmetry = math.sqrt(symmetry2 / max(norm2, 1e-300))
    if symmetry > 1e-8:
        raise ValueError(f"dynamical CSR is not symmetric: relative residual {symmetry:g}")
    translation_residual = float(np.linalg.norm(d_times_t) / max(math.sqrt(3.0 * norm2), 1e-300))
    if translation_residual > 5.0e-2:
        raise ValueError(f"mass-weighted translation residual exceeds the generator's h/h2 diagnostic gate: {translation_residual:g}")

    # Sum rows explicitly so empty CSR rows are handled as zero.
    row_sums = np.asarray([np.sum(np.abs(values[int(row_offsets[i]):int(row_offsets[i + 1])])) for i in range(d)])
    actual_bound = float(np.max(row_sums))
    if actual_bound > ref["spectral_bound"] * (1.0 + 32.0 * np.finfo(float).eps):
        raise ValueError("stored spectral bound is smaller than the recomputed CSR absolute row sum")
    tau = 6.465412e-2 / (8.617343e-5 * ref["temperature"])
    if tau * math.sqrt(actual_bound) > ref["kernel_u"] * (1.0 + 32.0 * np.finfo(float).eps):
        raise ValueError("recomputed spectral bound exceeds the kernel table U coverage")

    # Form the anchored principal minor of P D P using only rank-three row updates.
    c = translation.T @ d_times_t
    for row in range(dimension):
        correction = np.zeros(dimension, dtype=np.float64)
        for a_axis in range(3):
            correction -= d_times_t[minor_index[row], a_axis] * keep_translation[:, a_axis]
            correction -= keep_translation[row, a_axis] * d_times_t[minor_index, a_axis]
            for b_axis in range(3):
                correction += keep_translation[row, a_axis] * c[a_axis, b_axis] * keep_translation[:, b_axis]
        a[row] += correction
    try:
        chol = np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError("anchored P D P principal minor is not numerically positive definite") from exc
    pivots = np.diag(chol) ** 2
    minimum_pivot = float(np.min(pivots))
    if not math.isfinite(minimum_pivot) or minimum_pivot <= 0.0:
        raise ValueError("anchored Cholesky has a nonpositive or non-finite pivot")
    residual2 = 0.0
    for start in range(0, dimension, 64):
        stop = min(start + 64, dimension)
        rebuilt = chol[start:stop] @ chol.T
        delta = a[start:stop] - rebuilt
        residual2 += float(np.einsum("ij,ij->", delta, delta))
    reconstruction_residual = math.sqrt(residual2 / max(norm2, 1e-300))
    if not math.isfinite(reconstruction_residual) or reconstruction_residual > 1e-8:
        raise ValueError(f"Cholesky reconstruction residual is too large: {reconstruction_residual:g}")

    fingerprint = fnv64(path)
    sidecar = path.with_name(path.name + ".stability")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    if sidecar.exists() or temporary.exists():
        raise FileExistsError(f"stability sidecar or temporary already exists: {sidecar}")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("GPUMDJA_STABILITY 1\n")
            stream.write(f"fingerprint {fingerprint:016x}\n")
            stream.write(f"atoms {n}\nminimum_pivot {minimum_pivot:.17g}\n")
            stream.write(f"reconstruction_residual {reconstruction_residual:.17g}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(sidecar)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"PASS numerical Cholesky: min pivot={minimum_pivot:.9g}, raw D0 translation residual={translation_residual:.3e}, recomputed row bound={actual_bound:.9g}, reconstruction residual={reconstruction_residual:.3e}")
    print(f"wrote file-bound sidecar {sidecar}; this is an offline numerical check, not interval proof")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument("--kernel-table", type=Path, help="rebind fixed sparse D/B to another complete kernel table offline")
    parser.add_argument("--output", type=Path, help="new reference path required with --kernel-table")
    args = parser.parse_args()
    if bool(args.kernel_table) != bool(args.output):
        parser.error("--kernel-table and --output must be used together")
    if args.kernel_table:
        rebind_kernel(args.reference, args.kernel_table, args.output)
    else:
        check(args.reference, args.memory_limit_gib)


if __name__ == "__main__":
    main()
