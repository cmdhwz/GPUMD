#!/usr/bin/env python3
"""Compact a validated qNEP RPMD-JA raw v3 file to v4 without changing its source."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rpmd_ja_fit_reference as fit

COPY_BYTES = 8 * 1024 * 1024
TILE_ROWS = 128


def _close_raw(raw):
    for name in ("v", "c", "k", "coarse_v", "coarse_c"):
        value = raw.get(name)
        if value is not None and getattr(value, "_mmap", None) is not None:
            value._mmap.close()
            raw[name] = None


def _transport_statistics(raw):
    n, d = int(raw["n"]), int(raw["d"])
    v, coarse_v = raw["v"], raw["coarse_v"]
    fenergy = np.empty(d, dtype=np.float64)
    for r0 in range(0, d, TILE_ROWS):
        r1 = min(r0 + TILE_ROWS, d)
        v_rows = np.asarray(v[r0:r1])
        if not np.array_equal(v_rows, coarse_v[r0:r1]):
            raise ValueError("qraw v3 coarse V is not exactly the fine V used by v4 transport")
        # Match native std::accumulate's left-to-right site order, including cancellation.
        fenergy[r0:r1] = -np.add.accumulate(v_rows, axis=1)[:, -1]
    mass_coord = np.tile(np.asarray(raw["masses"], dtype=np.float64), 3)
    diff2 = np.zeros(3, dtype=np.float64)
    norm2 = np.zeros(3, dtype=np.float64)
    for r0 in range(0, d, TILE_ROWS):
        r1 = min(r0 + TILE_ROWS, d)
        v_rows = np.asarray(v[r0:r1])
        for alpha in range(3):
            for coordinate in range(d):
                component, atom = divmod(coordinate, n)
                fine = np.array(raw["c"][alpha, coordinate, r0:r1], dtype=np.float64, copy=True)
                coarse = np.array(raw["coarse_c"][alpha, coordinate, r0:r1], dtype=np.float64, copy=True)
                if component == alpha:
                    for local_row, row in enumerate(range(r0, r1)):
                        if row % n == atom:
                            fine[local_row] -= fenergy[row]
                            coarse[local_row] -= fenergy[row]
                    v_column = v_rows[:, atom]
                    fine -= v_column
                    coarse -= v_column
                scale = np.sqrt(mass_coord[r0:r1] * mass_coord[coordinate])
                fine /= scale
                coarse /= scale
                if not np.isfinite(fine).all() or not np.isfinite(coarse).all():
                    raise ValueError("qraw v3 native-H transport produced non-finite values")
                delta = fine - coarse
                diff2[alpha] += float(np.dot(delta, delta))
                norm2[alpha] += float(np.dot(fine, fine))
    if not np.isfinite(diff2).all() or not np.isfinite(norm2).all():
        raise ValueError("qraw v3 transported-H statistics overflow")
    relative = np.sqrt(diff2 / np.maximum(norm2, 1e-300))
    if np.any(relative > 0.05):
        raise ValueError("qraw v3 transported-H fine/coarse check exceeds 5e-2")
    return diff2, norm2


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _assert_source_identity(path, source, expected):
    current_path = _file_identity(path.stat())
    current_handle = _file_identity(os.fstat(source.fileno()))
    if current_path[:3] != expected[:3] or current_handle != expected:
        raise ValueError("qraw v3 source changed during conversion")


def _sha256_stream(source):
    digest = hashlib.sha256()
    source.seek(0)
    while block := source.read(COPY_BYTES):
        digest.update(block)
    source.seek(0)
    return digest.hexdigest()


def _copy_exact(source, destination, byte_count, source_digest, output_digest):
    remaining = int(byte_count)
    while remaining:
        block = source.read(min(COPY_BYTES, remaining))
        if not block:
            raise ValueError("qraw source ended while copying the retained v4 matrices")
        written = destination.write(block)
        if written != len(block):
            raise OSError("short write while creating compact qraw v4")
        source_digest.update(block)
        output_digest.update(block)
        remaining -= len(block)


def _copy_prefix(source, destination, byte_count, source_digest, output_digest):
    remaining, position = int(byte_count), 0
    while remaining:
        block = bytearray(source.read(min(COPY_BYTES, remaining)))
        if not block:
            raise ValueError("qraw source ended while copying its v4 header")
        source_digest.update(block)
        version_start, version_end = max(8 - position, 0), min(12 - position, len(block))
        if version_start < version_end:
            block[version_start:version_end] = struct.pack("<I", 4)[version_start + position - 8:version_end + position - 8]
        written = destination.write(block)
        if written != len(block):
            raise OSError("short write while creating compact qraw v4 header")
        output_digest.update(block)
        position += len(block)
        remaining -= len(block)
    if byte_count < 12:
        raise ValueError("qraw v3 header is too short to patch its format version")


def _hash_discarded_source(source, byte_count, source_digest):
    remaining = int(byte_count)
    while remaining:
        block = source.read(min(COPY_BYTES, remaining))
        if not block:
            raise ValueError("qraw source ended while verifying discarded v3 matrices")
        source_digest.update(block)
        remaining -= len(block)


def _publish_new_file(temporary, destination):
    if os.name == "nt":
        # Windows rename fails if destination exists, keeping the commit atomic and no-overwrite.
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)
        os.unlink(temporary)


def compact_v3(source_path, destination_path):
    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    if source_path == destination_path:
        raise ValueError("qraw v3 source and v4 destination must be different files")
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite qraw destination: {destination_path}")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(f"qraw destination directory does not exist: {destination_path.parent}")

    temporary = None
    raw = None
    try:
        path_identity = _file_identity(source_path.stat())
        with source_path.open("rb") as source:
            source_identity = _file_identity(os.fstat(source.fileno()))
            if path_identity[:3] != source_identity[:3]:
                raise ValueError("qraw v3 source was replaced before conversion")
            source_sha256 = _sha256_stream(source)
            raw = fit.read_raw(source_path, compute_source_fingerprint=False)
            if raw["version"] != 3:
                raise ValueError("only validated qraw v3 files can be compacted; v1/v2 keep their original derivative layout")
            diff2, norm2 = _transport_statistics(raw)
            stats = np.asarray(raw["diagnostics"], dtype="<f8").copy()
            if stats[17] != 3.0:
                raise ValueError("qraw v3 derivative marker changed after validation")
            footer = np.concatenate((stats, diff2.astype("<f8"), norm2.astype("<f8")))
            n, d = int(raw["n"]), int(raw["d"])
            retained_bytes = (d*n + 4*d*d) * 8
            discarded_bytes = source_identity[2] - raw["start"] - retained_bytes
            if discarded_bytes < 0:
                raise ValueError("qraw v3 source length is shorter than its retained matrix layout")
            expected_size = raw["start"] + retained_bytes + footer.nbytes
            _close_raw(raw)
            _assert_source_identity(source_path, source, source_identity)

            fd, temp_name = tempfile.mkstemp(prefix=f".{destination_path.name}.", suffix=".v4.tmp",
                                             dir=destination_path.parent)
            temporary = Path(temp_name)
            source_digest, output_digest = hashlib.sha256(), hashlib.sha256()
            with os.fdopen(fd, "w+b") as output:
                source.seek(0)
                _copy_prefix(source, output, raw["start"], source_digest, output_digest)
                _copy_exact(source, output, retained_bytes, source_digest, output_digest)
                _hash_discarded_source(source, discarded_bytes, source_digest)
                footer_bytes = footer.tobytes()
                written = output.write(footer_bytes)
                if written != len(footer_bytes):
                    raise OSError("short write while creating compact qraw v4 footer")
                output_digest.update(footer_bytes)
                output.flush()
                os.fsync(output.fileno())
            if source_digest.hexdigest() != source_sha256:
                raise ValueError("qraw v3 source content changed during conversion")
            _assert_source_identity(source_path, source, source_identity)

            compacted = fit.read_raw(temporary, compute_source_fingerprint=False)
            try:
                if compacted["version"] != 4 or compacted["derivative_version"] != 3:
                    raise ValueError("compacted qraw failed v4/derivative-v3 identity validation")
            finally:
                _close_raw(compacted)
            if temporary.stat().st_size != expected_size:
                raise ValueError("compacted qraw length does not match v4 layout")
            _assert_source_identity(source_path, source, source_identity)
            output_sha256 = output_digest.hexdigest()
            _publish_new_file(temporary, destination_path)
            temporary = None
            return {"source_bytes": source_identity[2], "output_bytes": destination_path.stat().st_size,
                    "source_sha256": source_sha256, "output_sha256": output_sha256,
                    "transport_relative": np.sqrt(diff2 / np.maximum(norm2, 1e-300)).tolist()}
    finally:
        if raw is not None:
            _close_raw(raw)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="validated qraw v3 file; retained as-is")
    parser.add_argument("destination", help="new qraw v4 path; must not already exist")
    args = parser.parse_args(argv)
    result = compact_v3(args.source, args.destination)
    print(f"qraw v4 written: {args.destination}; input={result['source_bytes']} bytes; "
          f"output={result['output_bytes']} bytes; transport_relative={result['transport_relative']}; "
          f"source_sha256={result['source_sha256']}; output_sha256={result['output_sha256']}; derivative_version=3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
