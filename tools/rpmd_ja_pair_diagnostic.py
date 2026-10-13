#!/usr/bin/env python3
"""Read-only fixed-pair diagnostics for GPJASMP1 centroid spools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rpmd_ja_structure_diagnostic import minimum_image, read_spool


FRAME_CHUNK = 64
BRANCH_THRESHOLD = 0.45
MAX_JOINT_DIMENSION = 64
MAX_JOINT_SERIES_BYTES = 128 * 1024 * 1024
MAX_AUTOCORRELATION_LAG = 256
BLOCK_LENGTHS = (8, 32, 128)
GRAPH_TILE = 128
QUADRATIC_COMPONENTS = ("xx", "2xy", "2xz", "yy", "2yz", "zz")


def _parse_pair(value):
    try:
        first, second = (int(part) for part in value.split(":"))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("pair must be two zero-based atom IDs separated by ':'")
    if first == second:
        raise argparse.ArgumentTypeError("pair atom IDs must differ")
    return first, second


def _parse_atom(value):
    try:
        atom = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("atom must be a zero-based atom ID")
    if atom < 0:
        raise argparse.ArgumentTypeError("atom ID must be nonnegative")
    return atom


def _positions(frames, start, end, atom_count, atom_ids=None):
    positions = frames["position"][start:end].reshape(end - start, 3, atom_count).transpose(0, 2, 1)
    return np.asarray(positions if atom_ids is None else positions[:, atom_ids, :], dtype=float)


def _distance_summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("distance statistics require a nonempty finite series")
    q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {"count": int(values.size), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "q05": float(q05), "median": float(median), "q95": float(q95)}


def _moment_accumulator(dimension):
    return {"count": 0, "sum": np.zeros(dimension), "second": np.zeros((dimension, dimension))}


def _add_moments(accumulator, values):
    if not np.isfinite(values).all():
        raise ValueError("derived coordinate statistics must be finite")
    accumulator["count"] += values.shape[0]
    accumulator["sum"] += np.sum(values, axis=0)
    accumulator["second"] += values.T @ values
    if not np.isfinite(accumulator["sum"]).all() or not np.isfinite(accumulator["second"]).all():
        raise ValueError("derived coordinate moments overflowed finite range")


def _moment_summary(accumulator):
    count = accumulator["count"]
    mean = accumulator["sum"] / count
    second = accumulator["second"] / count
    return {"frame_count": count, "mean": mean.tolist(), "second_moment": second.tolist(),
            "centered_covariance": (second - np.outer(mean, mean)).tolist()}


def _block_summaries(values, step_interval):
    result = []
    for target in BLOCK_LENGTHS:
        block_count = max(1, len(values) // target)
        edges = np.linspace(0, len(values), block_count + 1, dtype=int)
        means = np.asarray([np.mean(values[edges[i]:edges[i + 1]], axis=0) for i in range(block_count)])
        with np.errstate(over="ignore", invalid="ignore"):
            second_moments = np.asarray([np.mean(values[edges[i]:edges[i + 1]] ** 2, axis=0)
                                         for i in range(block_count)])
        sem = None if block_count < 2 else np.std(means, axis=0, ddof=1) / np.sqrt(block_count)
        second_sem = None if block_count < 2 else np.std(second_moments, axis=0, ddof=1) / np.sqrt(block_count)
        if (not np.isfinite(means).all() or not np.isfinite(second_moments).all()
                or (sem is not None and not np.isfinite(sem).all())
                or (second_sem is not None and not np.isfinite(second_sem).all())):
            raise ValueError("block statistics overflowed finite range")
        result.append({"target_block_length_frames": target,
                       "mean_block_length_native_steps": float(len(values) * step_interval / block_count),
                       "block_count": block_count, "block_lengths_frames": np.diff(edges).tolist(),
                       "mean_of_block_means": np.mean(means, axis=0).tolist(),
                       "block_sem": None if sem is None else sem.tolist(),
                       "mean_block_second_moment_of_coordinates": np.mean(second_moments, axis=0).tolist(),
                       "block_sem_of_squared_coordinates": None if second_sem is None else second_sem.tolist(),
                       "status": "INCONCLUSIVE_TOO_FEW_BLOCKS" if block_count < 2 else "DIAGNOSTIC_ESTIMATE"})
    return result


def _autocorrelation_summary(values, step_interval):
    values = np.asarray(values, dtype=float)
    count = values.size
    if not count or not np.isfinite(values).all():
        raise ValueError("autocorrelation requires a nonempty finite series")
    lag_cap = min(count // 4, MAX_AUTOCORRELATION_LAG)
    result = {"sample_count": int(count), "max_lag_frames": int(lag_cap),
              "configured_lag_cap_frames": MAX_AUTOCORRELATION_LAG, "status": None,
              "integrated_autocorrelation_time_frames": None,
              "integrated_autocorrelation_time_native_steps": None,
              "estimated_effective_sample_count": None}
    if count < 16 or lag_cap < 1:
        result["status"] = "INCONCLUSIVE_TOO_SHORT"
        return result
    centered = values - np.mean(values)
    variance = float(np.mean(centered * centered))
    if not np.isfinite(variance):
        raise ValueError("autocorrelation variance overflowed finite range")
    with np.errstate(over="ignore", invalid="ignore"):
        mean_square = float(np.mean(values * values))
    if not np.isfinite(mean_square):
        raise ValueError("autocorrelation squared projection overflowed finite range")
    if variance <= np.finfo(float).eps * max(np.finfo(float).tiny, mean_square):
        result["status"] = "INCONCLUSIVE_CONSTANT"
        return result
    rho = np.empty(lag_cap, dtype=float)
    for lag in range(1, lag_cap + 1):
        rho[lag - 1] = np.mean(centered[:-lag] * centered[lag:]) / variance
    if not np.isfinite(rho).all():
        raise ValueError("autocorrelation estimate overflowed finite range")
    nonpositive = np.flatnonzero(rho <= 0)
    stopped = int(nonpositive[0]) if nonpositive.size else len(rho)
    tau = max(0.5, 0.5 + float(np.sum(rho[:stopped])))
    if not np.isfinite(tau) or not np.isfinite(tau * step_interval):
        raise ValueError("integrated autocorrelation time overflowed finite range")
    capped = not nonpositive.size
    result.update({"status": "INCONCLUSIVE_LAG_CAP_REACHED" if capped else "DIAGNOSTIC_ESTIMATE",
                   "integrated_autocorrelation_time_frames": tau,
                   "integrated_autocorrelation_time_native_steps": tau * step_interval,
                   "estimated_effective_sample_count": min(float(count), count / (2.0 * tau)),
                   "first_nonpositive_lag_frames": None if not nonpositive.size else int(nonpositive[0] + 1)})
    return result


def _graph_support(atom_types, r0, cell, inverse, type_groups, requested_pairs, cutoff):
    radii = [max(0.0, cutoff - 0.5), cutoff, cutoff + 0.5]
    results = {}
    for group in type_groups:
        first_ids = np.flatnonzero(atom_types == group[0])
        second_ids = np.flatnonzero(atom_types == group[1])
        same_type = group[0] == group[1]
        stats = [{"count": 0, "sum_margin": 0.0, "min_margin": np.inf, "max_margin": -np.inf,
                  "supported": np.zeros(len(atom_types), dtype=bool)} for _ in radii]
        if same_type:
            tiles = ((first_ids[i:i + GRAPH_TILE], first_ids[j:j + GRAPH_TILE], i == j)
                     for i in range(0, len(first_ids), GRAPH_TILE)
                     for j in range(i, len(first_ids), GRAPH_TILE))
        else:
            tiles = ((first_ids[i:i + GRAPH_TILE], second_ids[j:j + GRAPH_TILE], False)
                     for i in range(0, len(first_ids), GRAPH_TILE)
                     for j in range(0, len(second_ids), GRAPH_TILE))
        for left, right, diagonal in tiles:
            delta = r0[right][None, :, :] - r0[left][:, None, :]
            fractional = delta @ inverse.T
            vectors = delta - np.rint(fractional) @ cell.T
            distances = np.linalg.norm(vectors, axis=2)
            if not np.isfinite(fractional).all() or not np.isfinite(distances).all():
                raise ValueError("fixed-R0 graph distances are not finite")
            if diagonal:
                rows, columns = np.triu_indices(len(left), k=1)
                edge_left, edge_right, edge_distances = left[rows], right[columns], distances[rows, columns]
            else:
                edge_left = np.repeat(left, len(right))
                edge_right = np.tile(right, len(left))
                edge_distances = distances.ravel()
            for radius, state in zip(radii, stats):
                selected = edge_distances < radius
                if not np.any(selected):
                    continue
                margins = radius - edge_distances[selected]
                state["count"] += int(np.count_nonzero(selected))
                state["sum_margin"] += float(np.sum(margins))
                state["min_margin"] = min(state["min_margin"], float(np.min(margins)))
                state["max_margin"] = max(state["max_margin"], float(np.max(margins)))
                state["supported"][edge_left[selected]] = True
                state["supported"][edge_right[selected]] = True
        requested = [(first, second) for first, second in requested_pairs
                     if tuple(sorted((int(atom_types[first]), int(atom_types[second])))) == group]
        requested_inside = sum(np.linalg.norm(minimum_image(r0[second] - r0[first], cell, inverse)) < cutoff
                               for first, second in requested)
        cutoff_state = stats[1]
        results[group] = {"atom_types": list(group), "requested_pair_count": len(requested),
                          "requested_pair_count_inside_cutoff": int(requested_inside),
                          "full_group_coverage_fraction": None if not cutoff_state["count"] else
                          float(requested_inside / cutoff_state["count"]),
                          "cutoffs": [
            {"cutoff_A": float(radius), "pair_count": state["count"],
             "supported_atom_count": int(np.count_nonzero(state["supported"])),
             "margin_A": None if not state["count"] else {
                 "minimum": state["min_margin"], "mean": state["sum_margin"] / state["count"],
                 "maximum": state["max_margin"]}}
            for radius, state in zip(radii, stats)]}
    return list(results.values())


def _pair_accumulator(pair_count):
    return {"count": np.zeros(pair_count, dtype=np.int64), "sum": np.zeros((pair_count, 3)),
            "second": np.zeros((pair_count, 3, 3)), "warning_frames": np.zeros(pair_count, dtype=np.int64),
            "warning_components": np.zeros(pair_count, dtype=np.int64)}


def _add_pair_samples(accumulator, drift, inverse):
    if not np.isfinite(drift).all():
        raise ValueError("fixed-image pair displacements are not finite")
    accumulator["count"] += drift.shape[0]
    accumulator["sum"] += np.sum(drift, axis=0)
    accumulator["second"] += np.einsum("fpi,fpj->pij", drift, drift)
    fractional = drift @ inverse.T
    violating = ~np.isfinite(fractional) | (np.abs(fractional) >= BRANCH_THRESHOLD)
    accumulator["warning_frames"] += np.count_nonzero(np.any(violating, axis=2), axis=0)
    accumulator["warning_components"] += np.count_nonzero(violating, axis=(0, 2))
    if not np.isfinite(accumulator["sum"]).all() or not np.isfinite(accumulator["second"]).all():
        raise ValueError("pair moments overflowed finite range")


def _pair_summary(accumulator, pair_id, distances):
    count = int(accumulator["count"][pair_id])
    mean = accumulator["sum"][pair_id] / count
    second = accumulator["second"][pair_id] / count
    covariance = second - np.outer(mean, mean)
    return {"frame_count": count, "mean_drift_vector_A": mean.tolist(),
            "second_moment_about_R0_A2": second.tolist(), "centered_covariance_A2": covariance.tolist(),
            "centered_variance_components_A2": np.diag(covariance).tolist(),
            "distance_A": _distance_summary(distances),
            "branch_warning_frames": int(accumulator["warning_frames"][pair_id]),
            "branch_warning_components": int(accumulator["warning_components"][pair_id])}

def diagnose(path, pairs, cutoff, atoms=()):
    started = time.perf_counter()
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
    atoms = [int(atom) for atom in atoms]
    if len(set(atoms)) != len(atoms) or any(atom < 0 or atom >= atom_count for atom in atoms):
        raise ValueError(f"--atom IDs must be unique zero-based indices in [0, {atom_count})")
    if set(atoms) & set(np.asarray(pairs, dtype=int).ravel().tolist()):
        raise ValueError("--atom IDs must not be endpoints of requested pairs")

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
    r0_sum /= train_count
    r0 = r0_sum
    if not np.isfinite(r0).all():
        raise ValueError("training R0 centroid mean is not finite")
    r0_seconds = time.perf_counter() - started
    reference_started = time.perf_counter()
    cell, inverse = spool["cell"], np.linalg.inv(spool["cell"])

    atom_types = spool["types"]
    pair_atoms = np.asarray(pairs, dtype=int)
    pair_raw = r0[pair_atoms[:, 1]] - r0[pair_atoms[:, 0]]
    pair_fractional = pair_raw @ inverse.T
    if not np.isfinite(pair_raw).all() or not np.isfinite(pair_fractional).all():
        raise ValueError("requested-pair R0 vectors are not finite")
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
                                            "R0_vector_A": neighbor_vectors[chosen], "column": atom_id,
                                            "selection": "midpoint_nearest_other_type"})
            needed_atoms.add(atom_id)
        for atom_id in atoms:
            existing = next((spec for spec in neighbor_specs[pair_id] if spec["atom_id"] == atom_id), None)
            if existing is not None:
                existing["selection"] += "+user_selected"
                continue
            raw = r0[atom_id] - pair_midpoints[pair_id]
            image = -np.rint(raw @ inverse.T).astype(np.int64)
            vector = raw + image @ cell.T
            neighbor_specs[pair_id].append({"type_id": int(atom_types[atom_id]), "atom_id": atom_id,
                                            "image": image, "R0_distance_A": float(np.linalg.norm(vector)),
                                            "R0_vector_A": vector, "column": atom_id,
                                            "selection": "user_selected"})
            needed_atoms.add(atom_id)

    selected_atoms = np.asarray(sorted(needed_atoms), dtype=int)
    columns = {int(atom): index for index, atom in enumerate(selected_atoms)}
    pair_columns = np.asarray([[columns[first], columns[second]] for first, second in pairs], dtype=int)
    for specs in neighbor_specs:
        for spec in specs:
            spec["column"] = columns[spec["atom_id"]]

    joint_dimensions = [3 * (1 + len(specs)) for specs in neighbor_specs]
    if sum(joint_dimensions) > MAX_JOINT_DIMENSION:
        raise ValueError(f"requested joint coordinates exceed the {MAX_JOINT_DIMENSION}-coordinate diagnostic cap")
    type_groups = sorted({tuple(sorted((int(atom_types[first]), int(atom_types[second]))))
                          for first, second in pairs})
    type_pair_indices = {group: [i for i, pair in enumerate(pairs)
                                 if tuple(sorted((int(atom_types[pair[0]]), int(atom_types[pair[1]])))) == group]
                         for group in type_groups}
    in_cutoff_pair_indices = {group: [i for i in indices if np.linalg.norm(r0_vectors[i]) < cutoff]
                              for group, indices in type_pair_indices.items()}
    joint_series_bytes = frame_count * sum(joint_dimensions) * np.dtype(float).itemsize
    quadratic_series_bytes = frame_count * 6 * np.dtype(float).itemsize if any(in_cutoff_pair_indices.values()) else 0
    if joint_series_bytes + quadratic_series_bytes > MAX_JOINT_SERIES_BYTES:
        raise ValueError(f"selected joint and group-quadratic series exceed the {MAX_JOINT_SERIES_BYTES}-byte memory cap")
    pair_offsets = np.cumsum([0] + joint_dimensions)
    joint_series = np.empty((frame_count, sum(joint_dimensions)), dtype=float)

    edges = np.linspace(0, train_count, 5, dtype=int)
    groups = [("training", 0, train_count), ("validation", train_count, frame_count)]
    groups.extend((f"training_block_{index + 1}", int(edges[index]), int(edges[index + 1]))
                  for index in range(4))
    pair_accumulators = {name: _pair_accumulator(len(pairs)) for name, _, _ in groups}

    pair_lattice = pair_images @ cell.T
    reference_setup_seconds = time.perf_counter() - reference_started
    second_pass_started = time.perf_counter()
    max_selected_chunk_bytes = 0
    for start in range(0, frame_count, FRAME_CHUNK):
        end = min(start + FRAME_CHUNK, frame_count)
        positions = _positions(spool["frames"], start, end, atom_count, selected_atoms)
        max_selected_chunk_bytes = max(max_selected_chunk_bytes, positions.nbytes)
        if not np.isfinite(positions).all():
            raise ValueError("spool centroids for requested pairs or fixed neighbors must be finite")
        first_xyz = positions[:, pair_columns[:, 0], :]
        pair_vectors = positions[:, pair_columns[:, 1], :] - first_xyz + pair_lattice[None, :, :]
        drift = pair_vectors - r0_vectors[None, :, :]
        for pair_id, specs in enumerate(neighbor_specs):
            midpoint = first_xyz[:, pair_id, :] + 0.5 * pair_vectors[:, pair_id, :]
            vectors = [drift[:, pair_id, :]]
            for spec in specs:
                raw = positions[:, spec["column"], :] - midpoint
                vector = raw + spec["image"] @ cell.T
                vectors.append(vector - spec["R0_vector_A"])
            joint_series[start:end, pair_offsets[pair_id]:pair_offsets[pair_id + 1]] = np.concatenate(vectors, axis=1)
        for name, group_start, group_end in groups:
            lo, hi = max(start, group_start), min(end, group_end)
            if lo >= hi:
                continue
            local = slice(lo - start, hi - start)
            _add_pair_samples(pair_accumulators[name], drift[local], inverse)

    selected_statistics_seconds = time.perf_counter() - second_pass_started
    if not np.isfinite(joint_series).all():
        raise ValueError("derived fixed-image pair or neighbor coordinates are not finite")
    graph_started = time.perf_counter()
    graph_support = _graph_support(atom_types, r0, cell, inverse, type_groups, pairs, cutoff)
    graph_seconds = time.perf_counter() - graph_started

    result_pairs = []
    for pair_id, pair in enumerate(pairs):
        local_series = joint_series[:, pair_offsets[pair_id]:pair_offsets[pair_id + 1]]
        pair_distances = np.linalg.norm(local_series[:, :3] + r0_vectors[pair_id], axis=1)
        branch = {}
        for name in ("training", "validation"):
            selected_distances = pair_distances[:train_count] if name == "training" else pair_distances[train_count:]
            summary = _pair_summary(pair_accumulators[name], pair_id, selected_distances)
            branch[name] = {"frame_count": summary["frame_count"],
                            "warning_frames": summary["branch_warning_frames"],
                            "warning_components": summary["branch_warning_components"]}
        neighbors = []
        for neighbor_index, spec in enumerate(neighbor_specs[pair_id]):
            coordinate_start = 3 + 3 * neighbor_index
            neighbor_drift = local_series[:, coordinate_start:coordinate_start + 3]
            neighbor_vectors = neighbor_drift + spec["R0_vector_A"]
            neighbor_distance = np.linalg.norm(neighbor_vectors, axis=1)
            neighbor_fractional = neighbor_drift @ inverse.T
            neighbor_violating = (~np.isfinite(neighbor_fractional)
                                 | (np.abs(neighbor_fractional) >= BRANCH_THRESHOLD))
            neighbor_branch = {}
            for name, start, end in (("training", 0, train_count),
                                     ("validation", train_count, frame_count)):
                violating = neighbor_violating[start:end]
                neighbor_branch[name] = {"frame_count": end - start,
                                         "warning_frames": int(np.count_nonzero(np.any(violating, axis=1))),
                                         "warning_components": int(np.count_nonzero(violating))}
            neighbors.append({"type_id": spec["type_id"], "atom_id": spec["atom_id"],
                              "reference_image": spec["image"].tolist(),
                              "selection": spec["selection"],
                              "R0_distance_A": spec["R0_distance_A"],
                              "R0_cutoff_margin_A": float(cutoff - spec["R0_distance_A"]),
                              "training_distance_A": _distance_summary(neighbor_distance[:train_count]),
                              "validation_distance_A": _distance_summary(neighbor_distance[train_count:]),
                              "branch_warnings": {"threshold_fractional_component": BRANCH_THRESHOLD,
                                                  "definition": "Fixed-R0 midpoint-neighbor linear coordinate displacement, transformed by the fixed cell inverse; no per-frame image reassignment is performed.",
                                                  **neighbor_branch}})
        coordinate_labels = []
        for component in "xyz":
            coordinate_labels.append({"vector": "pair_relative_displacement_from_R0", "atom_ids": list(pair),
                                      "reference_image": pair_images[pair_id].tolist(), "component": component,
                                      "units": "angstrom"})
        for spec in neighbor_specs[pair_id]:
            for component in "xyz":
                coordinate_labels.append({"vector": "midpoint_neighbor_relative_displacement_from_R0",
                                          "atom_id": spec["atom_id"], "type_id": spec["type_id"],
                                          "reference_image": spec["image"].tolist(),
                                          "selection": spec["selection"], "component": component,
                                          "units": "angstrom"})
        joint_train = local_series[:train_count]
        joint_validation = local_series[train_count:]
        balanced_blocks = {"training": _block_summaries(joint_train, sample_interval),
                           "validation": _block_summaries(joint_validation, sample_interval)}
        autocorrelation = {"training": [], "validation": []}
        for name, values in (("training", joint_train), ("validation", joint_validation)):
            for coordinate_id, label in enumerate(coordinate_labels):
                autocorrelation[name].append({"coordinate": label,
                    "projection": _autocorrelation_summary(values[:, coordinate_id], sample_interval),
                    "squared_projection": _autocorrelation_summary(values[:, coordinate_id] ** 2, sample_interval)})
        joint = {"dimension": int(local_series.shape[1]), "coordinate_labels": coordinate_labels,
                 "coordinate_definition": "Concatenate pair-vector displacement from its fixed-R0 vector and each selected atom's displacement from the moving pair midpoint relative to its fixed-R0 midpoint vector; all integer images are chosen once at R0.",
                 "mean_units": "angstrom", "second_moment_units": "angstrom^2",
                 "centered_covariance_units": "angstrom^2",
                 "training": _moment_summary(_series_moments(joint_train)),
                 "validation": _moment_summary(_series_moments(joint_validation)),
                 "balanced_block_summaries": balanced_blocks,
                 "autocorrelation_definition": "For each Cartesian coordinate and its square, tau = 0.5 + the sum of positive-lag autocorrelations through the first nonpositive lag; effective sample count = frame_count / (2*tau). Estimates are censored when no nonpositive lag is found before min(frame_count/4, 256).",
                 "autocorrelation": autocorrelation}
        result_pairs.append({"atoms": list(pair), "atom_types": [int(atom_types[pair[0]]), int(atom_types[pair[1]])],
                             "reference_image": pair_images[pair_id].tolist(),
                             "R0_pair_vector_A": r0_vectors[pair_id].tolist(),
                             "R0_pair_distance_A": float(np.linalg.norm(r0_vectors[pair_id])),
                             "R0_cutoff_margin_A": float(cutoff - np.linalg.norm(r0_vectors[pair_id])),
                             "training": _pair_summary(pair_accumulators["training"], pair_id, pair_distances[:train_count]),
                             "validation": _pair_summary(pair_accumulators["validation"], pair_id, pair_distances[train_count:]),
                             "training_blocks": [{"block": index + 1, "frame_start": int(edges[index]),
                                                  "frame_end_exclusive": int(edges[index + 1]),
                                                  **_pair_summary(pair_accumulators[f"training_block_{index + 1}"], pair_id,
                                                                  pair_distances[edges[index]:edges[index + 1]])}
                                                 for index in range(4)],
                             "branch_warnings": {"threshold_fractional_component": BRANCH_THRESHOLD,
                                                 "definition": "pair-vector displacement from R0 transformed by the fixed cell inverse",
                                                 **branch},
                             "joint_coordinate_statistics": joint,
                             "midpoint_neighbors_by_type": neighbors})
    group_quadratics = []
    graph_lookup = {tuple(item["atom_types"]): item for item in graph_support}
    for group in type_groups:
        support = graph_lookup[group]
        in_cutoff_pairs = in_cutoff_pair_indices[group]
        item = {"atom_types": list(group), "requested_pair_count": len(type_pair_indices[group]),
                "requested_pair_count_inside_R0_graph": support["requested_pair_count_inside_cutoff"],
                "full_group_coverage_fraction": support["full_group_coverage_fraction"],
                "quadratic_components": list(QUADRATIC_COMPONENTS),
                "definition": "Frame-wise sum of the six quadratic products of R0-relative pair-vector components across requested pair edges in this unordered type group that lie strictly inside the fixed-R0 graph cutoff.",
                "requested_edge_scope": "Only user-requested pair edges in this type group with fixed-R0 distance strictly less than --cutoff.",
                "mean_units": "angstrom^2", "second_moment_units": "angstrom^4",
                "centered_covariance_units": "angstrom^4",
                "temporal_diagnostic_pair_count": len(in_cutoff_pairs),
                "support_status": "SPARSE_REQUESTED_PAIR_SET" if len(in_cutoff_pairs) < 2 else "REQUESTED_PAIR_SET"}
        if in_cutoff_pairs:
            quadratic_series = np.zeros((frame_count, 6), dtype=float)
            for pair_id in in_cutoff_pairs:
                coordinates = joint_series[:, pair_offsets[pair_id]:pair_offsets[pair_id] + 3]
                x, y, z = coordinates.T
                with np.errstate(over="ignore", invalid="ignore"):
                    quadratic_series[:, 0] += x * x
                    quadratic_series[:, 1] += 2 * x * y
                    quadratic_series[:, 2] += 2 * x * z
                    quadratic_series[:, 3] += y * y
                    quadratic_series[:, 4] += 2 * y * z
                    quadratic_series[:, 5] += z * z
            if not np.isfinite(quadratic_series).all():
                raise ValueError("group quadratic time series overflowed finite range")
            for name, values in (("training", quadratic_series[:train_count]),
                                 ("validation", quadratic_series[train_count:])):
                item[name] = _moment_summary(_series_moments(values))
                item[name]["block_summaries"] = _block_summaries(values, sample_interval)
                item[name]["autocorrelation"] = [
                    {"component": component,
                     "projection": _autocorrelation_summary(values[:, component_id], sample_interval),
                     "squared_projection": _autocorrelation_summary(values[:, component_id] ** 2,
                                                                      sample_interval)}
                    for component_id, component in enumerate(QUADRATIC_COMPONENTS)]
                item[name]["temporal_status"] = "DIAGNOSTIC_ONLY"
            del values
            del quadratic_series
        else:
            for name, count in (("training", train_count), ("validation", frame_count - train_count)):
                item[name] = {"frame_count": count, "mean": [0.0] * 6,
                              "second_moment": [[0.0] * 6 for _ in range(6)],
                              "centered_covariance": [[0.0] * 6 for _ in range(6)],
                              "block_summaries": [], "autocorrelation": [],
                              "temporal_status": "INCONCLUSIVE_NO_IN_CUTOFF_REQUESTED_EDGES"}
        for name in ("training", "validation"):
            covariance = np.asarray(item[name]["centered_covariance"])
            eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
            rank = int(np.linalg.matrix_rank(covariance))
            item[name]["covariance_eigenvalues"] = eigenvalues.tolist()
            item[name]["rank_fraction"] = rank / 6.0
            item[name]["covariance_rank_diagnostic"] = "Empirical rank of the six observed in-cutoff quadratic coordinates; joint coordinates may be algebraically redundant, and this is not parameter or operator identifiability."
        group_quadratics.append(item)
    for item in group_quadratics:
        item["fixed_R0_graph_support"] = graph_lookup[tuple(item["atom_types"])]
    combined_coordinate_labels = []
    for pair_id, pair_result in enumerate(result_pairs):
        combined_coordinate_labels.extend({"requested_pair_index": pair_id,
                                           "requested_pair_atoms": pair_result["atoms"], **label}
                                          for label in pair_result["joint_coordinate_statistics"]["coordinate_labels"])
    combined_joint = {"dimension": int(joint_series.shape[1]),
                      "coordinate_labels": combined_coordinate_labels,
                      "coordinate_definition": "Concatenate each requested pair's R0-relative pair vector followed by its fixed-image moving-midpoint neighbor vectors, in requested-pair order; cross-pair covariance is retained.",
                      "covariance_rank_interpretation": "Joint coordinates may be algebraically redundant; empirical covariance rank describes sampled variation, not parameter or operator identifiability.",
                      "mean_units": "angstrom", "second_moment_units": "angstrom^2",
                      "centered_covariance_units": "angstrom^2",
                      "training": _moment_summary(_series_moments(joint_series[:train_count])),
                      "validation": _moment_summary(_series_moments(joint_series[train_count:])),
                      "balanced_block_summaries": {"training": _block_summaries(joint_series[:train_count], sample_interval),
                                                    "validation": _block_summaries(joint_series[train_count:], sample_interval)}}
    summary_seconds = (time.perf_counter() - started - r0_seconds - reference_setup_seconds
                       - selected_statistics_seconds - graph_seconds)
    return {"status": "DIAGNOSTIC_ONLY", "input": str(spool["path"]), "format": "GPJASMP1",
            "atom_indexing": "zero_based", "frame_count": frame_count,
            "training_frame_count": train_count, "validation_frame_count": frame_count - train_count,
            "sample_step_interval": float(sample_interval), "sample_step_interval_units": "native spool step units",
            "cutoff_A": float(cutoff), "R0_definition": "per-atom arithmetic mean over the first floor(2*frame_count/3) unwrapped centroid frames",
            "reference_image_rule": "choose the component-wise fractional image once at training R0 and reuse that integer image for every frame",
            "user_selected_atoms": atoms,
            "joint_coordinate_dimension_cap": MAX_JOINT_DIMENSION,
            "selected_statistical_series_memory_cap_bytes": MAX_JOINT_SERIES_BYTES,
            "graph_support_cutoff_rule": "fixed-R0 periodic type-pair distances strictly less than each cutoff; radii are --cutoff plus or minus 0.5 A",
            "pairs": result_pairs,
            "joint_coordinate_statistics": combined_joint,
            "quadratic_statistics_by_type_group": group_quadratics,
            "performance": {"stage_seconds": {"R0_mean_and_step_validation": r0_seconds,
                                               "fixed_reference_setup": reference_setup_seconds,
                                               "selected_column_second_pass": selected_statistics_seconds,
                                               "fixed_R0_graph_support": graph_seconds,
                                               "summary_assembly": summary_seconds},
                            "memory_bytes": {"R0_mean": int(r0.nbytes),
                                             "maximum_selected_coordinate_chunk": int(max_selected_chunk_bytes),
                                             "estimated_full_coordinate_chunk_avoided": int(min(FRAME_CHUNK, frame_count) * atom_count * 3 * 8),
                                             "joint_selected_coordinate_series": int(joint_series.nbytes),
                                             "maximum_group_quadratic_series": int(quadratic_series_bytes),
                                             "retained_distance_samples": 0,
                                             "spool_file": int(spool["path"].stat().st_size)},
                            "selected_atom_count": int(selected_atoms.size),
                            "maximum_chunk_frames": FRAME_CHUNK},
            "limitations": ["Fixed pair IDs and R0 midpoint neighbors are followed throughout; no changing-neighbor search is performed.",
                            "Pair and neighbor distances use their R0 reference images without per-frame minimum-image reassignment; branch warnings flag pair-vector excursions at the stated fractional threshold.",
                            "Autocorrelation and block SEM are finite-series sensitivity diagnostics; lag-capped estimates can be unresolved and are not rigorous confidence intervals.",
                            "Coordinate, quadratic, and fixed-R0 graph summaries are data statistics; they do not measure operator dependence or identify a unique transport gauge.",
                            "This diagnostic does not detect tunneling events or estimate a tunneling or transfer rate."]}


def _series_moments(values):
    accumulator = _moment_accumulator(values.shape[1])
    _add_moments(accumulator, values)
    return accumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool")
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True, metavar="ATOM1:ATOM2",
                        help="fixed zero-based atom IDs; repeat for additional pairs")
    parser.add_argument("--atom", action="append", type=_parse_atom, default=[], metavar="ATOM",
                        help="include this fixed zero-based atom in each requested pair's local midpoint coordinates; repeat as needed")
    parser.add_argument("--cutoff", type=float, required=True, help="cutoff in angstroms for R0 cutoff margins")
    parser.add_argument("--output", required=True, help="new JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    input_path, output_path = Path(args.spool).resolve(), Path(args.output).resolve()
    if output_path == input_path:
        parser.error("output path must differ from the input spool")
    if output_path.exists():
        parser.error(f"refusing to overwrite existing report: {output_path}")
    try:
        report = json.dumps(diagnose(input_path, args.pair, args.cutoff, args.atom),
                            indent=2, sort_keys=True, allow_nan=False) + "\n"
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(report)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
