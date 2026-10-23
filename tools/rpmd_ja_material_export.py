#!/usr/bin/env python3
"""Export one synchronized GPUMD rpmd_ja XYZ/current segment for offline evaluation.

The XYZ input must be the full-system dump_xyz stream containing pos and vel.
The explicit spec binds the row order to caller-provided atom IDs and sample IDs;
dump_xyz itself does not write unique atom IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex

import numpy as np

NATIVE_TIME_FS = 10.18051
NATIVE_TIME_PS = 0.01018051
_CURRENT_COLUMNS = "time_ps Jcent_x Jcent_y Jcent_z DeltaJ_x DeltaJ_y DeltaJ_z JA_x JA_y JA_z"
_CURRENT_UNITS = "eV*Angstrom/natural_time"


def _need(mapping, keys, name):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _spec(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(value, ("format", "trajectory_id", "position_stream_id", "ordered_atom_ids",
        "ordered_species", "frame_ids", "step_ids", "time_ps", "memory_limit_bytes",
        "dump_xyz", "current_file", "output_npz"), "export spec")
    if value["format"] == "RPJAMAT_EXPORT2":
        _need(value, ("online_reference_id", "online_reference_package", "online_reference_package_sha256"),
            "RPJAMAT_EXPORT2 spec")
    elif value["format"] == "RPJAMAT_EXPORT1":
        if value.get("allow_legacy_current_unverified") is not True:
            raise ValueError("RPJAMAT_EXPORT1 requires explicit allow_legacy_current_unverified=true")
    else:
        raise ValueError("unsupported export spec format")
    for key in ("trajectory_id", "position_stream_id"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{key} must be a nonempty explicit identity")
    if value["format"] == "RPJAMAT_EXPORT2":
        if not isinstance(value["online_reference_id"], str) or not value["online_reference_id"]:
            raise ValueError("online_reference_id must be a nonempty explicit identity")
        digest = value["online_reference_package_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("online_reference_package_sha256 must be a SHA-256 digest")
    atom_ids = value["ordered_atom_ids"]
    species = value["ordered_species"]
    frame_ids = value["frame_ids"]
    step_ids = value["step_ids"]
    times = np.asarray(value["time_ps"], dtype=np.float64)
    if not isinstance(atom_ids, list) or not atom_ids or len(set(atom_ids)) != len(atom_ids) or \
       any(not isinstance(item, str) or not item for item in atom_ids):
        raise ValueError("ordered_atom_ids must contain unique, explicit nonempty IDs")
    if not isinstance(species, list) or len(species) != len(atom_ids) or \
       any(not isinstance(item, str) or not item for item in species):
        raise ValueError("ordered_species must name every ordered atom")
    if not isinstance(frame_ids, list) or not isinstance(step_ids, list) or len(frame_ids) < 2 or \
       len(step_ids) != len(frame_ids) or len(times) != len(frame_ids):
        raise ValueError("frame_ids, step_ids, and time_ps must explicitly identify at least two synchronized samples")
    if len(set(frame_ids)) != len(frame_ids) or len(set(step_ids)) != len(step_ids) or \
       any(not isinstance(item, int) or isinstance(item, bool) for item in frame_ids + step_ids):
        raise ValueError("frame_ids and step_ids must be unique integers")
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])) or \
       any(right <= left for left, right in zip(step_ids, step_ids[1:])):
        raise ValueError("frame_ids and step_ids must be strictly increasing")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("time_ps must be finite, unique, and strictly increasing")
    memory_limit = value["memory_limit_bytes"]
    if not isinstance(memory_limit, int) or isinstance(memory_limit, bool) or memory_limit <= 0:
        raise ValueError("memory_limit_bytes must be a positive integer")
    for key in ("dump_xyz", "current_file", "output_npz", "online_reference_package"):
        if key == "online_reference_package" and value["format"] != "RPJAMAT_EXPORT2":
            continue
        target = Path(value[key])
        if not target.is_absolute():
            target = path.parent / target
        value["_" + key] = target.resolve()
    value["_path"] = path
    return value


def _parse_properties(comment):
    fields = {}
    for token in shlex.split(comment):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    if "Properties" not in fields:
        raise ValueError("dump_xyz frame has no Properties schema")
    raw = fields["Properties"].split(":")
    if len(raw) % 3:
        raise ValueError("dump_xyz Properties schema is malformed")
    schema, offset = {}, 0
    for i in range(0, len(raw), 3):
        name, kind, count_text = raw[i:i + 3]
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError("dump_xyz Properties field width is invalid") from exc
        if not name or name in schema or count <= 0:
            raise ValueError("dump_xyz Properties field names or widths are invalid")
        schema[name] = (kind, count, offset)
        offset += count
    if any(name in schema for name in ("group", "group_labels")):
        raise ValueError("group-selected/reordered dump_xyz frames cannot bind the full atom order")
    for name in ("species", "pos", "vel"):
        if name not in schema:
            raise ValueError(f"dump_xyz must include {name}")
    if schema["species"][:2] != ("S", 1) or schema["pos"][:2] != ("R", 3) or schema["vel"][:2] != ("R", 3):
        raise ValueError("dump_xyz species/pos/vel types or widths do not match the required layout")
    return fields, schema, offset


def _frame_header(comment, frame_index, expected_cell):
    fields, schema, row_width = _parse_properties(comment)
    try:
        time_fs = float(fields["Time"])
        pbc = fields["pbc"].split()
        lattice_rows = np.asarray([float(item) for item in fields["Lattice"].split()], dtype=np.float64).reshape(3, 3)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"dump_xyz frame {frame_index} needs valid Time, pbc, and 3x3 Lattice") from exc
    if not math.isfinite(time_fs) or pbc != ["T", "T", "T"] or not np.isfinite(lattice_rows).all():
        raise ValueError(f"dump_xyz frame {frame_index} has invalid time, PBC, or cell")
    # dump_xyz emits Box.h as the three Lattice row vectors; native reference stores Box.h row-major.
    cell = lattice_rows.T.copy()
    if abs(float(np.linalg.det(cell))) <= 1e-12:
        raise ValueError(f"dump_xyz frame {frame_index} has a singular cell")
    if expected_cell is not None and not np.allclose(cell, expected_cell, rtol=0.0, atol=1e-12):
        raise ValueError(f"dump_xyz cell changed at frame {frame_index}")
    return time_fs, cell, schema, row_width


def _iter_xyz(path, atom_count, ordered_species, store):
    fixed_cell = None
    with Path(path).open("r", encoding="utf-8") as stream:
        frame_index = 0
        while True:
            count_line = stream.readline()
            if not count_line:
                break
            try:
                count = int(count_line.strip())
            except ValueError as exc:
                raise ValueError(f"dump_xyz frame {frame_index} has an invalid atom count") from exc
            if count != atom_count:
                raise ValueError(f"dump_xyz atom count at frame {frame_index} is not the full declared system")
            comment = stream.readline()
            if not comment:
                raise ValueError(f"dump_xyz frame {frame_index} has a truncated header")
            time_fs, cell, schema, row_width = _frame_header(comment, frame_index, fixed_cell)
            if fixed_cell is None:
                fixed_cell = cell
            positions = np.empty((atom_count, 3), dtype=np.float64) if store else None
            velocities = np.empty((atom_count, 3), dtype=np.float64) if store else None
            species = []
            for atom_index in range(atom_count):
                line = stream.readline()
                if not line:
                    raise ValueError(f"dump_xyz frame {frame_index} is truncated at atom {atom_index}")
                row = line.split()
                if len(row) != row_width:
                    raise ValueError(f"dump_xyz frame {frame_index} atom {atom_index} violates Properties width")
                symbol = row[schema["species"][2]]
                species.append(symbol)
                if symbol != ordered_species[atom_index]:
                    raise ValueError(f"dump_xyz species/order mismatch at frame {frame_index}, atom {atom_index}")
                try:
                    pos_start = schema["pos"][2]
                    vel_start = schema["vel"][2]
                    pos = [float(value) for value in row[pos_start:pos_start + 3]]
                    vel = [float(value) for value in row[vel_start:vel_start + 3]]
                except ValueError as exc:
                    raise ValueError(f"dump_xyz frame {frame_index} has nonnumeric pos/vel") from exc
                if not all(math.isfinite(value) for value in pos + vel):
                    raise ValueError(f"dump_xyz frame {frame_index} has non-finite pos/vel")
                if store:
                    positions[atom_index] = pos
                    velocities[atom_index] = vel
            yield frame_index, time_fs, cell, positions, velocities
            frame_index += 1


def _read_current(path, expected_times, allow_legacy_unverified=False):
    markers = 0
    metadata = {}
    rows = []
    in_segment = False
    segment_version = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if stripped in ("# segment_metadata_version 1", "# segment_metadata_version 2"):
                markers += 1
                in_segment = True
                segment_version = int(stripped.rsplit(" ", 1)[1])
                continue
            if stripped.startswith("# segment_metadata_version"):
                raise ValueError("heat-current segment metadata version is unsupported")
            if in_segment:
                parts = stripped[1:].strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    if key in metadata:
                        raise ValueError(f"heat-current segment repeats {key} metadata")
                    metadata[key] = value
            continue
        if not in_segment:
            raise ValueError("heat-current data before the unique segment metadata is not accepted")
        try:
            row = [float(value) for value in stripped.split()]
        except ValueError as exc:
            raise ValueError("heat-current segment contains a malformed data row") from exc
        if len(row) != 10 or not all(math.isfinite(value) for value in row):
            raise ValueError("heat-current segment rows must contain ten finite columns")
        rows.append(row)
    if markers != 1:
        raise ValueError("heat-current file must contain exactly one rpmd_ja segment")
    if segment_version == 1 and not allow_legacy_unverified:
        raise ValueError("legacy heat-current v1 requires explicit RPJAMAT_EXPORT1 unverified mode")
    if segment_version == 2 and allow_legacy_unverified:
        raise ValueError("RPJAMAT_EXPORT1 only accepts a legacy v1 current segment")
    required = {"backend": "qnep-block-v3", "current_units": _CURRENT_UNITS, "columns": _CURRENT_COLUMNS}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("heat-current segment backend, units, or columns do not match the v3 evaluator")
    reference_sha256 = metadata.get("reference_package_sha256") if segment_version == 2 else None
    if segment_version == 2 and (reference_sha256 is None or len(reference_sha256) != 64 or
       any(character not in "0123456789abcdefABCDEF" for character in reference_sha256)):
        raise ValueError("heat-current v2 segment has no valid reference package SHA-256")
    try:
        declared_count = int(metadata["sample_frames"])
    except (KeyError, ValueError) as exc:
        raise ValueError("heat-current segment has no valid sample_frames declaration") from exc
    if declared_count != len(expected_times) or len(rows) != declared_count:
        raise ValueError("heat-current sample count does not match the synchronized export spec")
    current = np.asarray(rows, dtype=np.float64)
    times = current[:, 0]
    if np.any(np.diff(times) <= 0) or not np.allclose(times, expected_times, rtol=0.0, atol=1e-10):
        raise ValueError("heat-current time is duplicate or not synchronized to the declared dump frames")
    jcent, delta, ja = current[:, 1:4], current[:, 4:7], current[:, 7:10]
    scale = np.maximum(1.0, np.maximum(np.max(np.abs(ja), axis=1), np.max(np.abs(jcent + delta), axis=1)))
    residual = np.max(np.abs(ja - (jcent + delta)), axis=1)
    if np.any(residual > 1e-12 * scale):
        raise ValueError("heat-current segment violates JA = Jcent + DeltaJ")
    return jcent, delta, ja, reference_sha256


def export_trajectory(spec_path):
    spec = _spec(spec_path)
    frames = len(spec["frame_ids"])
    atoms = len(spec["ordered_atom_ids"])
    expected_times = np.asarray(spec["time_ps"], dtype=np.float64)
    estimate = 8 * (frames * (6 * atoms + 9) + 6 * atoms) + (64 << 10)
    if estimate > spec["memory_limit_bytes"]:
        raise MemoryError(f"estimated export host memory {estimate} bytes exceeds frozen limit {spec['memory_limit_bytes']}")

    count = 0
    fixed_cell = None
    for frame_index, time_fs, cell, _, _ in _iter_xyz(
            spec["_dump_xyz"], atoms, spec["ordered_species"], store=False):
        if frame_index >= frames:
            raise ValueError("dump_xyz has more frames than the frozen synchronized sample")
        if abs(time_fs / 1000.0 - expected_times[frame_index]) > 1e-10:
            raise ValueError(f"dump_xyz frame {frame_index} time is not synchronized to the frozen ps time")
        if fixed_cell is None:
            fixed_cell = cell
        count += 1
    if count != frames:
        raise ValueError("dump_xyz frame count does not match the frozen synchronized sample")

    reference_sha256 = None
    if spec["format"] == "RPJAMAT_EXPORT2":
        reference_sha256 = _sha256(spec["_online_reference_package"])
        if reference_sha256.lower() != spec["online_reference_package_sha256"].lower():
            raise ValueError("declared reference package SHA-256 does not match the package bytes")
        jc_native, delta_j_native, ja_native, native_reference_sha256 = _read_current(
            spec["_current_file"], expected_times)
        if native_reference_sha256.lower() != reference_sha256.lower():
            raise ValueError("native heat-current reference package SHA-256 does not match the selected package digest")
    else:
        jc_native, delta_j_native, ja_native, _ = _read_current(
            spec["_current_file"], expected_times, allow_legacy_unverified=True)
    dump_sha256 = _sha256(spec["_dump_xyz"])
    current_sha256 = _sha256(spec["_current_file"])
    positions = np.empty((frames, atoms, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    for frame_index, time_fs, cell, frame_position, frame_velocity in _iter_xyz(
            spec["_dump_xyz"], atoms, spec["ordered_species"], store=True):
        if abs(time_fs / 1000.0 - expected_times[frame_index]) > 1e-10:
            raise ValueError(f"dump_xyz frame {frame_index} time changed between export passes")
        if not np.allclose(cell, fixed_cell, rtol=0.0, atol=1e-12):
            raise ValueError(f"dump_xyz cell changed between export passes at frame {frame_index}")
        positions[frame_index] = frame_position
        velocities[frame_index] = frame_velocity

    output = spec["_output_npz"]
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        frame_ids=np.asarray(spec["frame_ids"], dtype=np.int64),
        step_ids=np.asarray(spec["step_ids"], dtype=np.int64),
        time_ps=expected_times,
        atom_ids=np.asarray(spec["ordered_atom_ids"]),
        trajectory_id=np.asarray(spec["trajectory_id"]),
        position_stream_id=np.asarray(spec["position_stream_id"]),
        position_source=np.asarray("dump_xyz.atom.position_per_atom"),
        dump_source_sha256=np.asarray(dump_sha256),
        current_source_sha256=np.asarray(current_sha256),
        cell_angstrom=fixed_cell,
        wrapped_position_angstrom=positions,
        vc_angstrom_per_ps=velocities * 1000.0,
        jc_eV_angstrom_per_ps=jc_native / NATIVE_TIME_PS)
    if spec["format"] == "RPJAMAT_EXPORT2":
        arrays.update(
            online_reference_id=np.asarray(spec["online_reference_id"]),
            online_reference_package_sha256=np.asarray(reference_sha256.lower()),
            online_current_units=np.asarray("eV*angstrom/ps"),
            online_delta_j_eV_angstrom_per_ps=delta_j_native / NATIVE_TIME_PS,
            online_ja_eV_angstrom_per_ps=ja_native / NATIVE_TIME_PS)
    np.savez_compressed(output, **arrays)
    if spec["format"] == "RPJAMAT_EXPORT1":
        return {"format": "RPJAMAT_EXPORT1_REPORT", "status": "EXPORTED_UNVERIFIED_LEGACY_CURRENT",
            "trajectory_id": spec["trajectory_id"], "position_stream_id": spec["position_stream_id"],
            "online_current_identity": "NOT_AVAILABLE_LEGACY_UNVERIFIED", "frame_count": frames,
            "atom_count": atoms, "output_npz": str(output), "trajectory_npz_sha256": _sha256(output),
            "dump_source_sha256": dump_sha256, "current_source_sha256": current_sha256,
            "cell_angstrom": fixed_cell.tolist(), "estimated_host_peak_bytes": estimate,
            "memory_gate_note": "capacity estimate only; not a runtime guarantee"}
    return {"format": "RPJAMAT_EXPORT2_REPORT", "status": "EXPORTED",
        "trajectory_id": spec["trajectory_id"], "position_stream_id": spec["position_stream_id"],
        "online_reference_id": spec["online_reference_id"],
        "online_reference_package_sha256": reference_sha256.lower(),
        "frame_count": frames, "atom_count": atoms, "output_npz": str(output),
        "trajectory_npz_sha256": _sha256(output), "dump_source_sha256": dump_sha256,
        "current_source_sha256": current_sha256,
        "cell_angstrom": fixed_cell.tolist(), "estimated_host_peak_bytes": estimate,
        "memory_gate_note": "capacity estimate only; not a runtime guarantee",
        "identity_limit": "dump_xyz has no unique atom IDs; atom IDs are caller-declared row bindings checked against species order",
        "position_note": "pos and vel are atom.position_per_atom and atom.velocity_per_atom; wrapped positions are retained for production MIC replay per reference R0",
        "unit_conversion": {"velocity": "angstrom/fs * 1000 = angstrom/ps",
                            "current": "eV*angstrom/native_time / 0.01018051 = eV*angstrom/ps",
                            "online_delta_j_and_ja": "eV*angstrom/native_time / 0.01018051 = eV*angstrom/ps",
                            "native_time_ps": NATIVE_TIME_PS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = export_trajectory(args.spec)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
