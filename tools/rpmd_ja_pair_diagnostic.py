#!/usr/bin/env python3
"""Read-only fixed-pair diagnostics for GPJASMP1 centroid spools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rpmd_ja_structure_diagnostic import minimum_image, read_spool


FRAME_CHUNK = 64
BRANCH_THRESHOLD = 0.45


def _parse_pair(value):
    try:
        first, second = (int(part) for part in value.split(":"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("pair must be two zero-based atom IDs separated by ':'")
    if first == second:
        raise argparse.ArgumentTypeError("pair atom IDs must differ")
    return first, second


def _positions(frames, start, end, atom_count):
    return np.asarray(frames["position"][start:end], dtype=float).reshape(end - start, 3, atom_count).transpose(0, 2, 1)


def _distance_summary(values):
    values = np.asarray(values, dtype=float)
    q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {"count": int(values.size), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "q05": float(q05), "median": float(median), "q95": float(q95)}


def _pair_accumulator(pair_count):
    return {"count": np.zeros(pair_count, dtype=np.int64), "sum": np.zeros((pair_count, 3)),
            "second": np.zeros((pair_count, 3, 3)), "warning_frames": np.zeros(pair_count, dtype=np.int64),
            "warning_components": np.zeros(pair_count, dtype=np.int64),
            "distances": [[] for _ in range(pair_count)]}


def _add_pair_samples(accumulator, drift, distances, inverse):
    accumulator["count"] += drift.shape[0]
    accumulator["sum"] += np.sum(drift, axis=0)
    accumulator["second"] += np.einsum("fpi,fpj->pij", drift, drift)
    fractional = drift @ inverse.T
    violating = ~np.isfinite(fractional) | (np.abs(fractional) >= BRANCH_THRESHOLD)
    accumulator["warning_frames"] += np.count_nonzero(np.any(violating, axis=2), axis=0)
    accumulator["warning_components"] += np.count_nonzero(violating, axis=(0, 2))
    for pair_id in range(drift.shape[1]):
        accumulator["distances"][pair_id].append(distances[:, pair_id].copy())


def _pair_summary(accumulator, pair_id):
    count = int(accumulator["count"][pair_id])
    mean = accumulator["sum"][pair_id] / count
    second = accumulator["second"][pair_id] / count
    covariance = second - np.outer(mean, mean)
    return {"frame_count": count, "mean_drift_vector_A": mean.tolist(),
            "second_moment_about_R0_A2": second.tolist(), "centered_covariance_A2": covariance.tolist(),
            "centered_variance_components_A2": np.diag(covariance).tolist(),
            "distance_A": _distance_summary(np.concatenate(accumulator["distances"][pair_id])),
            "branch_warning_frames": int(accumulator["warning_frames"][pair_id]),
            "branch_warning_components": int(accumulator["warning_components"][pair_id])}


def _distance_accumulator():
    return []


def _add_distances(accumulator, distances):
    accumulator.append(np.asarray(distances, dtype=float).copy())


def diagnose(path, pairs, cutoff):
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    spool = read_spool(path)
    frame_count, atom_count = spool["frame_count"], spool["n"]
    if frame_count < 6:
        raise ValueError("at least six frames are required for four nonempty training blocks")
    pairs = [(int(first), int(second)) for first, second in pairs]
    if not pairs:
        raise ValueError("at least one atom pair is required")
    if len(set(tuple(sorted(pair)) for pair in pairs)) != len(pairs):
        raise ValueError("atom pairs must be unique, regardless of direction")
    if any(first < 0 or second < 0 or first >= atom_count or second >= atom_count or first == second
           for first, second in pairs):
        raise ValueError(f"pair atom IDs must be distinct zero-based indices in [0, {atom_count})")

    train_count = 2 * frame_count // 3
    r0_sum = np.zeros((atom_count, 3))
    sample_interval = None
    previous_step = None
    for start in range(0, frame_count, FRAME_CHUNK):
        end = min(start + FRAME_CHUNK, frame_count)
        steps = np.asarray(spool["frames"]["step"][start:end], dtype=float)
        if not np.isfinite(steps).all():
            raise ValueError("spool steps must be finite")
        differences = np.diff(steps)
        if previous_step is not None:
            differences = np.concatenate(([steps[0] - previous_step], differences))
        if differences.size:
            if sample_interval is None:
                sample_interval = float(differences[0])
            if np.any(differences <= 0) or np.any(~np.isfinite(differences)) or np.any(np.abs(differences - sample_interval) > 1.0e-9):
                raise ValueError("spool sample steps must be strictly increasing with a uniform interval")
        previous_step = float(steps[-1])
        if start < train_count:
            stop = min(end, train_count)
            positions = _positions(spool["frames"], start, stop, atom_count)
            if not np.isfinite(positions).all():
                raise ValueError("spool centroids must be finite")
            r0_sum += np.sum(positions, axis=0)
    r0 = r0_sum / train_count
    cell, inverse = spool["cell"], np.linalg.inv(spool["cell"])

    atom_types = spool["types"]
    pair_atoms = np.asarray(pairs, dtype=int)
    pair_raw = r0[pair_atoms[:, 1]] - r0[pair_atoms[:, 0]]
    pair_fractional = pair_raw @ inverse.T
    pair_images = -np.rint(pair_fractional).astype(np.int64)
    r0_vectors = minimum_image(pair_raw, cell, inverse)
    needed_atoms = set(pair_atoms.ravel().tolist())
    neighbor_specs = [[] for _ in pairs]
    pair_midpoints = r0[pair_atoms[:, 0]] + 0.5 * r0_vectors
    for pair_id, (first, second) in enumerate(pairs):
        pair_types = {int(atom_types[first]), int(atom_types[second])}
        for atom_type in np.unique(atom_types):
            if int(atom_type) in pair_types:
                continue
            candidate_ids = np.flatnonzero(atom_types == atom_type)
            neighbor_raw = r0[candidate_ids] - pair_midpoints[pair_id]
            neighbor_vectors = minimum_image(neighbor_raw, cell, inverse)
            neighbor_distances = np.linalg.norm(neighbor_vectors, axis=1)
            chosen = int(np.argmin(neighbor_distances))
            atom_id = int(candidate_ids[chosen])
            image = -np.rint(neighbor_raw[chosen] @ inverse.T).astype(np.int64)
            neighbor_specs[pair_id].append({"type_id": int(atom_type), "atom_id": atom_id,
                                            "image": image, "R0_distance_A": float(neighbor_distances[chosen]),
                                            "column": atom_id})
            needed_atoms.add(atom_id)

    selected_atoms = np.asarray(sorted(needed_atoms), dtype=int)
    columns = {int(atom): index for index, atom in enumerate(selected_atoms)}
    pair_columns = np.asarray([[columns[first], columns[second]] for first, second in pairs], dtype=int)
    for specs in neighbor_specs:
        for spec in specs:
            spec["column"] = columns[spec["atom_id"]]

    edges = np.linspace(0, train_count, 5, dtype=int)
    groups = [("training", 0, train_count), ("validation", train_count, frame_count)]
    groups.extend((f"training_block_{index + 1}", int(edges[index]), int(edges[index + 1]))
                  for index in range(4))
    pair_accumulators = {name: _pair_accumulator(len(pairs)) for name, _, _ in groups}
    neighbor_distances = {name: [[_distance_accumulator() for _ in specs] for specs in neighbor_specs]
                          for name, _, _ in groups}

    pair_lattice = pair_images @ cell.T
    for start in range(0, frame_count, FRAME_CHUNK):
        end = min(start + FRAME_CHUNK, frame_count)
        all_positions = _positions(spool["frames"], start, end, atom_count)
        positions = all_positions[:, selected_atoms, :]
        if not np.isfinite(positions).all():
            raise ValueError("spool centroids for requested pairs or fixed neighbors must be finite")
        first_xyz = positions[:, pair_columns[:, 0], :]
        pair_vectors = positions[:, pair_columns[:, 1], :] - first_xyz + pair_lattice[None, :, :]
        drift = pair_vectors - r0_vectors[None, :, :]
        distances = np.linalg.norm(pair_vectors, axis=2)
        for name, group_start, group_end in groups:
            lo, hi = max(start, group_start), min(end, group_end)
            if lo >= hi:
                continue
            local = slice(lo - start, hi - start)
            _add_pair_samples(pair_accumulators[name], drift[local], distances[local], inverse)
            for pair_id, specs in enumerate(neighbor_specs):
                midpoint = first_xyz[local, pair_id, :] + 0.5 * pair_vectors[local, pair_id, :]
                for neighbor_index, spec in enumerate(specs):
                    raw = positions[local, spec["column"], :] - midpoint
                    vector = raw + spec["image"] @ cell.T
                    _add_distances(neighbor_distances[name][pair_id][neighbor_index], np.linalg.norm(vector, axis=1))

    result_pairs = []
    for pair_id, pair in enumerate(pairs):
        branch = {}
        for name in ("training", "validation"):
            summary = _pair_summary(pair_accumulators[name], pair_id)
            branch[name] = {"frame_count": summary["frame_count"],
                            "warning_frames": summary["branch_warning_frames"],
                            "warning_components": summary["branch_warning_components"]}
        neighbors = []
        for neighbor_index, spec in enumerate(neighbor_specs[pair_id]):
            neighbors.append({"type_id": spec["type_id"], "atom_id": spec["atom_id"],
                              "reference_image": spec["image"].tolist(),
                              "R0_distance_A": spec["R0_distance_A"],
                              "R0_cutoff_margin_A": float(cutoff - spec["R0_distance_A"]),
                              "training_distance_A": _distance_summary(np.concatenate(neighbor_distances["training"][pair_id][neighbor_index])),
                              "validation_distance_A": _distance_summary(np.concatenate(neighbor_distances["validation"][pair_id][neighbor_index]))})
        result_pairs.append({"atoms": list(pair), "atom_types": [int(atom_types[pair[0]]), int(atom_types[pair[1]])],
                             "reference_image": pair_images[pair_id].tolist(),
                             "R0_pair_vector_A": r0_vectors[pair_id].tolist(),
                             "R0_pair_distance_A": float(np.linalg.norm(r0_vectors[pair_id])),
                             "R0_cutoff_margin_A": float(cutoff - np.linalg.norm(r0_vectors[pair_id])),
                             "training": _pair_summary(pair_accumulators["training"], pair_id),
                             "validation": _pair_summary(pair_accumulators["validation"], pair_id),
                             "training_blocks": [{"block": index + 1, "frame_start": int(edges[index]),
                                                  "frame_end_exclusive": int(edges[index + 1]),
                                                  **_pair_summary(pair_accumulators[f"training_block_{index + 1}"], pair_id)}
                                                 for index in range(4)],
                             "branch_warnings": {"threshold_fractional_component": BRANCH_THRESHOLD,
                                                 "definition": "pair-vector displacement from R0 transformed by the fixed cell inverse",
                                                 **branch},
                             "midpoint_neighbors_by_type": neighbors})
    return {"status": "DIAGNOSTIC_ONLY", "input": str(spool["path"]), "format": "GPJASMP1",
            "atom_indexing": "zero_based", "frame_count": frame_count,
            "training_frame_count": train_count, "validation_frame_count": frame_count - train_count,
            "sample_step_interval": float(sample_interval), "sample_step_interval_units": "native spool step units",
            "cutoff_A": float(cutoff), "R0_definition": "per-atom arithmetic mean over the first floor(2*frame_count/3) unwrapped centroid frames",
            "reference_image_rule": "choose the component-wise fractional image once at training R0 and reuse that integer image for every frame",
            "pairs": result_pairs,
            "limitations": ["Fixed pair IDs and R0 midpoint neighbors are followed throughout; no changing-neighbor search is performed.",
                            "Pair and neighbor distances use their R0 reference images without per-frame minimum-image reassignment; branch warnings flag pair-vector excursions at the stated fractional threshold.",
                            "This diagnostic does not detect tunneling events or estimate a tunneling or transfer rate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool")
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True, metavar="ATOM1:ATOM2",
                        help="fixed zero-based atom IDs; repeat for additional pairs")
    parser.add_argument("--cutoff", type=float, required=True, help="cutoff in angstroms for R0 cutoff margins")
    parser.add_argument("--output", required=True, help="new JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    input_path, output_path = Path(args.spool).resolve(), Path(args.output).resolve()
    if output_path == input_path:
        parser.error("output path must differ from the input spool")
    if output_path.exists():
        parser.error(f"refusing to overwrite existing report: {output_path}")
    try:
        report = json.dumps(diagnose(input_path, args.pair, args.cutoff), indent=2, sort_keys=True) + "\n"
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(report)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
