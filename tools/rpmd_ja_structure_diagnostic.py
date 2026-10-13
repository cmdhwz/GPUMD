#!/usr/bin/env python3
"""Read-only, candidate-limited diagnostics for GPJASMP1 centroid spools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np


MAGIC = b"GPJASMP1"
ENDIAN = 0x01020304
FRAME_BLOCK = 16
HYDROGEN_BLOCK = 256
MAX_CANDIDATES = 64
HISTOGRAM_BINS = 30


def read_spool(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("wrong GPJASMP1 magic")
        version_bytes, endian_bytes = stream.read(4), stream.read(4)
        if endian_bytes == struct.pack("<I", ENDIAN):
            order = "<"
        elif endian_bytes == struct.pack(">I", ENDIAN):
            order = ">"
        else:
            raise ValueError("unsupported GPJASMP1 byte order")
        if struct.unpack(order + "I", version_bytes)[0] != 1:
            raise ValueError("unsupported GPJASMP1 version")
        n, beads = struct.unpack(order + "ii", stream.read(8))
        temperature = struct.unpack(order + "d", stream.read(8))[0]
        if n < 2 or beads < 1 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid GPJASMP1 dimensions or temperature")
        cell = np.frombuffer(stream.read(72), dtype=order + "f8").astype(float).reshape(3, 3)
        masses = np.frombuffer(stream.read(n * 8), dtype=order + "f8").astype(float)
        types = np.frombuffer(stream.read(n * 4), dtype=order + "i4").astype(np.int32)
        offset = stream.tell()
        if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) <= np.finfo(float).tiny:
            raise ValueError("invalid GPJASMP1 cell")
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("invalid GPJASMP1 masses")
        frame_bytes = 8 + 2 * 3 * n * 8
        payload = path.stat().st_size - offset
        if payload < 0 or payload % frame_bytes:
            raise ValueError("truncated GPJASMP1 frame or trailing partial data")
        frames_count = payload // frame_bytes
        if frames_count < 3:
            raise ValueError("GPJASMP1 needs at least three frames")
    dtype = np.dtype([("step", order + "f8"), ("position", order + "f8", (3 * n,)),
                      ("force", order + "f8", (3 * n,))])
    frames = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(frames_count,))
    return dict(path=path, n=n, beads=beads, temperature=temperature, cell=cell, masses=masses, types=types,
                frames=frames, frame_count=frames_count)


def minimum_image(displacement, cell, inverse):
    fractional = displacement @ inverse.T
    return (fractional - np.rint(fractional)) @ cell.T


def fixed_r0_mean_drift(train_sum, validation_sum, train_count, validation_count, r0, types, masses,
                        hydrogen_type, oxygen_type, ion_types):
    type_groups = {"hydrogen": (hydrogen_type, np.flatnonzero(types == hydrogen_type)),
                   "oxygen": (oxygen_type, np.flatnonzero(types == oxygen_type))}
    for ion_type in ion_types:
        type_groups[f"ion_type_{ion_type}"] = (ion_type, np.flatnonzero(types == ion_type))
    named_types = {hydrogen_type, oxygen_type, *ion_types}
    for atom_type in np.unique(types):
        if int(atom_type) not in named_types:
            type_groups[f"atom_type_{int(atom_type)}"] = (int(atom_type), np.flatnonzero(types == atom_type))

    reports = {}
    for phase, phase_sum, count in (("training", train_sum, train_count), ("validation", validation_sum, validation_count)):
        delta = phase_sum / count - r0
        type_reports = {}
        for name, (atom_type, ids) in type_groups.items():
            vectors = delta[ids]
            norms = np.linalg.norm(vectors, axis=1)
            type_reports[name] = {"type_id": int(atom_type), "atom_count": int(len(ids)),
                                  "mean_displacement_vector_A": np.mean(vectors, axis=0).tolist(),
                                  "vector_rms_displacement_A": float(np.sqrt(np.mean(norms * norms))),
                                  "max_vector_displacement_A": float(np.max(norms))}
        translation = np.sum(delta * masses[:, None], axis=0) / np.sum(masses)
        internal = delta - translation[None, :]
        internal_norm = float(np.sqrt(np.sum(masses * np.sum(internal * internal, axis=1))))
        reports[phase] = {"type_displacements": type_reports,
                          "mass_weighted_mean_translation_A": translation.tolist(),
                          "mass_weighted_internal_displacement_norm_sqrt_mass_A": internal_norm,
                          "mass_weighted_internal_displacement_rms_A": float(internal_norm / np.sqrt(np.sum(masses)))}
    return {"definition": "per-atom phase-mean centroid minus the same training R0, without recentering or periodic image reassignment",
            "type_displacement_units": "angstrom",
            "internal_norm_units": "sqrt(mass_unit)*angstrom",
            "training": reports["training"], "validation": reports["validation"]}


def summary(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "q05": None, "median": None, "q95": None}
    q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "q05": float(q05), "median": float(median), "q95": float(q95)}


def fixed_metrics(xyz, h_ids, pairs, cell, inverse):
    pair_xyz = xyz[:, pairs, :]
    bridge = minimum_image(pair_xyz[:, :, 1] - pair_xyz[:, :, 0], cell, inverse)
    proton = minimum_image(xyz[:, h_ids, :] - pair_xyz[:, :, 0], cell, inverse)
    length2 = np.sum(bridge * bridge, axis=2)
    length = np.sqrt(length2)
    valid = length2 > 0.0
    projection = np.full(length.shape, np.nan)
    np.divide(np.sum(proton * bridge, axis=2), length2, out=projection, where=valid)
    perpendicular = np.linalg.norm(proton - projection[:, :, None] * bridge, axis=2)
    ecc = 2.0 * projection - 1.0
    asymmetry = np.full(length.shape, np.nan)
    np.divide(np.linalg.norm(proton, axis=2) - np.linalg.norm(proton - bridge, axis=2), length,
              out=asymmetry, where=valid)
    return {"r": length, "ecc": ecc, "asymmetry": asymmetry, "perp": perpendicular,
            "segment": valid & (projection >= 0.0) & (projection <= 1.0), "valid": valid}


def candidate_metrics(xyz, h_ids, candidate_ids, candidate_mask, candidate_counts, pairs, cell, inverse):
    result = fixed_metrics(xyz, h_ids, pairs, cell, inverse)
    oxygen = xyz[:, candidate_ids, :]
    delta = minimum_image(oxygen - xyz[:, h_ids, None, :], cell, inverse)
    distances = np.linalg.norm(delta, axis=3)
    distances = np.where(candidate_mask[None, :, :], distances, np.inf)
    order = np.argsort(distances, axis=2, kind="stable")
    ids = np.broadcast_to(candidate_ids[None, :, :], distances.shape)
    nearest = np.take_along_axis(ids, order[:, :, :2], axis=2)
    same = (((nearest[:, :, 0] == pairs[None, :, 0]) & (nearest[:, :, 1] == pairs[None, :, 1])) |
            ((nearest[:, :, 0] == pairs[None, :, 1]) & (nearest[:, :, 1] == pairs[None, :, 0])))
    margin = np.full(same.shape, np.nan)
    if candidate_ids.shape[1] >= 3:
        sorted_distances = np.take_along_axis(distances, order, axis=2)
        has_third = candidate_counts >= 3
        margin[:, has_third] = sorted_distances[:, has_third, 2] - sorted_distances[:, has_third, 1]
    result.update(match=same, margin=margin)
    return result


def make_edges(low, high, fallback):
    if not np.isfinite(low) or not np.isfinite(high):
        low, high = fallback
    if low == high:
        padding = max(1.0, abs(low)) * 1.0e-6
        low, high = low - padding, high + padding
    return np.linspace(low, high, HISTOGRAM_BINS + 1)


def histogram2d(r, ecc, mask, r_edges, ecc_edges):
    valid = mask & np.isfinite(r) & np.isfinite(ecc)
    counts = np.histogram2d(r[valid], ecc[valid], bins=(r_edges, ecc_edges))[0].astype(np.int64)
    return counts, int(np.count_nonzero(valid) - counts.sum())


def fixed_r0_branch_diagnostics(pos, r0, inverse, train):
    phase = {name: {"violating_frames": 0, "violating_atoms": 0, "violating_components": 0}
             for name in ("training", "validation")}
    max_abs_fractional = 0.0
    for f0 in range(0, len(pos), FRAME_BLOCK):
        f1 = min(f0 + FRAME_BLOCK, len(pos))
        fractional = (np.asarray(pos[f0:f1], dtype=float) - r0[None, :, :]) @ inverse.T
        violating_components = ~np.isfinite(fractional) | (np.abs(fractional) >= 0.45)
        if np.isfinite(fractional).any():
            max_abs_fractional = max(max_abs_fractional, float(np.nanmax(np.abs(fractional))))
        for phase_name, start, end in (("training", 0, train), ("validation", train, len(pos))):
            lo, hi = max(f0, start), min(f1, end)
            if lo >= hi:
                continue
            selected = violating_components[lo - f0:hi - f0]
            phase[phase_name]["violating_frames"] += int(np.count_nonzero(np.any(selected, axis=(1, 2))))
            phase[phase_name]["violating_atoms"] += int(np.count_nonzero(np.any(selected, axis=2)))
            phase[phase_name]["violating_components"] += int(np.count_nonzero(selected))
    return {"threshold_fractional_component": 0.45,
            "definition": "direct displacement from fixed training R0 through the cell inverse; no recentering or periodic branch reselection",
            "max_abs_fractional_component": max_abs_fractional,
            "training": phase["training"], "validation": phase["validation"],
            "violating_frames": phase["training"]["violating_frames"] + phase["validation"]["violating_frames"],
            "violating_atoms": phase["training"]["violating_atoms"] + phase["validation"]["violating_atoms"],
            "violating_components": phase["training"]["violating_components"] + phase["validation"]["violating_components"]}


def fixed_r0_ion_nearest_distances(r0, pairs, ion_atoms, cell, inverse):
    result = {ion_type: np.full(len(pairs), np.inf) for ion_type in ion_atoms}
    for h0 in range(0, len(pairs), 64):
        h1 = min(h0 + 64, len(pairs))
        pair = pairs[h0:h1]
        anchor = r0[pair[:, 0]]
        midpoint = anchor + 0.5 * minimum_image(r0[pair[:, 1]] - anchor, cell, inverse)
        for ion_type, ion_ids in ion_atoms.items():
            for i0 in range(0, len(ion_ids), 1024):
                ion_xyz = r0[ion_ids[i0:i0 + 1024]]
                delta = minimum_image(ion_xyz[None, :, :] - midpoint[:, None, :], cell, inverse)
                distance = np.linalg.norm(delta, axis=2)
                result[ion_type][h0:h1] = np.minimum(result[ion_type][h0:h1], np.min(distance, axis=1))
    return result


def diagnose(path, hydrogen_type, oxygen_type, candidate_cutoff, block_frames=100, ion_types=()):
    if not np.isfinite(candidate_cutoff) or candidate_cutoff <= 0 or block_frames < 1:
        raise ValueError("candidate cutoff and block size must be positive")
    spool = read_spool(path)
    n, nf = spool["n"], spool["frame_count"]
    h_all = np.flatnonzero(spool["types"] == hydrogen_type)
    o_all = np.flatnonzero(spool["types"] == oxygen_type)
    if not h_all.size or o_all.size < 2 or hydrogen_type == oxygen_type:
        raise ValueError("explicit atom types must identify at least one H and two distinct O atoms")
    ion_types = tuple(int(ion_type) for ion_type in ion_types)
    if len(set(ion_types)) != len(ion_types) or any(ion_type in (hydrogen_type, oxygen_type) for ion_type in ion_types):
        raise ValueError("ion types must be unique and distinct from the explicit hydrogen and oxygen types")
    ion_atoms = {ion_type: np.flatnonzero(spool["types"] == ion_type) for ion_type in ion_types}
    if any(not ids.size for ids in ion_atoms.values()):
        raise ValueError("each explicit ion type must identify at least one atom")
    train = 2 * nf // 3
    pos = spool["frames"]["position"].reshape(nf, 3, n).transpose(0, 2, 1)
    r0_sum = np.zeros((n, 3))
    validation_sum = np.zeros((n, 3))
    previous_step = -np.inf
    sample_interval = None
    for f0 in range(0, nf, FRAME_BLOCK):
        f1 = min(f0 + FRAME_BLOCK, nf)
        steps = np.asarray(spool["frames"]["step"][f0:f1], dtype=float)
        xyz = np.asarray(pos[f0:f1], dtype=float)
        if not np.isfinite(steps).all() or not np.isfinite(xyz).all():
            raise ValueError("spool steps and centroids must be finite")
        differences = np.diff(steps) if f0 == 0 else np.concatenate(([steps[0] - previous_step], np.diff(steps)))
        if sample_interval is None:
            sample_interval = float(differences[0])
        if np.any(differences <= 0) or np.any(~np.isfinite(differences)) or np.any(np.abs(differences - sample_interval) > 1.0e-9):
            raise ValueError("spool sample steps must be strictly increasing with a uniform interval")
        previous_step = steps[-1]
        if f0 < train:
            r0_sum += np.sum(xyz[:min(f1, train) - f0], axis=0)
        if f1 > train:
            validation_sum += np.sum(xyz[max(train - f0, 0):], axis=0)
    r0 = r0_sum / train
    mean_drift = fixed_r0_mean_drift(r0_sum, validation_sum, train, nf - train, r0,
                                    spool["types"], spool["masses"], hydrogen_type, oxygen_type, ion_types)
    cell, inverse = spool["cell"], np.linalg.inv(spool["cell"])
    branch_contract = fixed_r0_branch_diagnostics(pos, r0, inverse, train)
    candidate_lists, raw_counts, truncated, references, r0_pair, static_margin = [], [], [], [], [], []
    for h0 in range(0, len(h_all), HYDROGEN_BLOCK):
        h_ids = h_all[h0:h0 + HYDROGEN_BLOCK]
        delta = minimum_image(r0[o_all][None, :, :] - r0[h_ids, None, :], cell, inverse)
        distances = np.linalg.norm(delta, axis=2)
        raw_counts_block = np.count_nonzero(distances <= candidate_cutoff, axis=1)
        candidate_width = min(MAX_CANDIDATES, len(o_all))
        selected = np.argpartition(distances, kth=candidate_width - 1, axis=1)[:, :candidate_width]
        selected_distances = np.take_along_axis(distances, selected, axis=1)
        order = np.lexsort((selected, np.where(selected_distances <= candidate_cutoff, selected_distances, np.inf)), axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        selected_distances = np.take_along_axis(selected_distances, order, axis=1)
        selected_mask = selected_distances <= candidate_cutoff
        raw_counts.extend(raw_counts_block.astype(int).tolist())
        for row, h in enumerate(h_ids):
            is_truncated = raw_counts_block[row] > MAX_CANDIDATES
            if is_truncated:
                truncated.append(int(h))
            candidate_o = o_all[selected[row, selected_mask[row]]]
            candidate_lists.append(candidate_o)
            if len(candidate_o) >= 2 and not is_truncated:
                references.append(candidate_o[:2])
                r0_pair.append(float(np.linalg.norm(minimum_image(r0[candidate_o[1]] - r0[candidate_o[0]], cell, inverse))))
                static_margin.append(float(selected_distances[row, 2] - selected_distances[row, 1])
                                     if len(candidate_o) > 2 else np.nan)
            else:
                references.append(None)
                r0_pair.append(np.nan)
                static_margin.append(np.nan)
    covered_rows = np.array([i for i, pair in enumerate(references) if pair is not None], dtype=int)
    unclassified_h = h_all[np.asarray([i for i, pair in enumerate(references) if pair is None], dtype=int)]
    covered_h = h_all[covered_rows]
    if not len(covered_h):
        raise ValueError("no hydrogen has two or more R0 oxygen candidates within candidate cutoff")
    pairs_all = np.asarray([references[i] for i in covered_rows], dtype=int)
    candidate_lists = [candidate_lists[i] for i in covered_rows]
    candidate_counts_all = np.array([len(c) for c in candidate_lists], dtype=int)
    max_k = max(candidate_counts_all)
    candidate_ids_all = np.zeros((len(covered_h), max_k), dtype=int)
    candidate_mask_all = np.zeros((len(covered_h), max_k), dtype=bool)
    for i, c in enumerate(candidate_lists):
        candidate_ids_all[i, :len(c)] = c
        candidate_mask_all[i, :len(c)] = True
    ion_distances = fixed_r0_ion_nearest_distances(r0, pairs_all, ion_atoms, cell, inverse)

    low_r, high_r, low_e, high_e = np.inf, -np.inf, np.inf, -np.inf
    for f0 in range(0, train, FRAME_BLOCK):
        xyz = np.asarray(pos[f0:min(f0 + FRAME_BLOCK, train)], dtype=float)
        for h0 in range(0, len(covered_h), HYDROGEN_BLOCK):
            h1 = min(h0 + HYDROGEN_BLOCK, len(covered_h))
            m = fixed_metrics(xyz, covered_h[h0:h1], pairs_all[h0:h1], cell, inverse)
            if np.isfinite(m["r"]).any():
                low_r, high_r = min(low_r, float(np.nanmin(m["r"]))), max(high_r, float(np.nanmax(m["r"])))
            if np.isfinite(m["ecc"]).any():
                low_e, high_e = min(low_e, float(np.nanmin(m["ecc"]))), max(high_e, float(np.nanmax(m["ecc"])))
    r_edges, e_edges = make_edges(low_r, high_r, (0.0, 1.0)), make_edges(low_e, high_e, (-1.0, 1.0))
    groups = ("all", "local_pair_match", "local_pair_change", "projection_in_segment", "projection_outside_segment")
    phases = {name: {"sums": np.zeros(4), "counts": np.zeros(4, dtype=np.int64), "changes": 0, "pairs": 0,
                     "segments": 0, "valid_projection": 0, "margin_sum": 0.0, "margin_count": 0,
                     "hist": {key: np.zeros((HISTOGRAM_BINS, HISTOGRAM_BINS), dtype=np.int64) for key in groups},
                     "outside": {key: 0 for key in groups}} for name in ("training", "validation")}
    blocks = []
    for phase, start, end in (("training", 0, train), ("validation", train, nf)):
        for b0 in range(start, end, block_frames):
            blocks.append({"phase": phase, "start_frame": b0, "end_frame_exclusive": min(b0 + block_frames, end),
                           "sums": np.zeros(2), "counts": np.zeros(2, dtype=np.int64),
                           "changes": 0, "pairs": 0, "segments": 0, "valid_projection": 0})
    per_h = []
    for h0 in range(0, len(covered_h), HYDROGEN_BLOCK):
        h1 = min(h0 + HYDROGEN_BLOCK, len(covered_h))
        h_ids, pairs = covered_h[h0:h1], pairs_all[h0:h1]
        candidate_ids, candidate_mask = candidate_ids_all[h0:h1], candidate_mask_all[h0:h1]
        candidate_counts = candidate_counts_all[h0:h1]
        local_r, local_e, local_gap = (np.full((nf, len(h_ids)), np.nan) for _ in range(3))
        local_change = np.zeros((2, len(h_ids)), dtype=np.int64)
        local_pairs = np.zeros((2, len(h_ids)), dtype=np.int64)
        local_segment = np.zeros((2, len(h_ids)), dtype=np.int64)
        local_projection = np.zeros((2, len(h_ids)), dtype=np.int64)
        for phase_id, (phase, p0, p1) in enumerate((("training", 0, train), ("validation", train, nf))):
            for block in (b for b in blocks if b["phase"] == phase):
                for f0 in range(block["start_frame"], block["end_frame_exclusive"], FRAME_BLOCK):
                    f1 = min(f0 + FRAME_BLOCK, block["end_frame_exclusive"])
                    xyz = np.asarray(pos[f0:f1], dtype=float)
                    m = candidate_metrics(xyz, h_ids, candidate_ids, candidate_mask, candidate_counts, pairs, cell, inverse)
                    local_r[f0:f1], local_e[f0:f1] = m["r"], m["ecc"]
                    local_gap[f0:f1] = m["margin"]
                    state, r, e, valid = phases[phase], m["r"], m["ecc"], m["valid"]
                    segment, match = m["segment"], m["match"]
                    for k, value in enumerate((r, e, m["perp"], m["asymmetry"])):
                        finite = np.isfinite(value)
                        state["sums"][k] += float(np.nansum(value))
                        state["counts"][k] += int(np.count_nonzero(finite))
                    state["changes"] += int(np.count_nonzero(~match));state["pairs"] += match.size
                    state["segments"] += int(np.count_nonzero(segment));state["valid_projection"] += int(np.count_nonzero(valid))
                    margin_finite = np.isfinite(m["margin"])
                    state["margin_sum"] += float(np.nansum(m["margin"]))
                    state["margin_count"] += int(np.count_nonzero(margin_finite))
                    block["sums"][0] += float(np.nansum(r));block["counts"][0] += int(np.count_nonzero(np.isfinite(r)))
                    block["sums"][1] += float(np.nansum(e));block["counts"][1] += int(np.count_nonzero(np.isfinite(e)))
                    block["changes"] += int(np.count_nonzero(~match));block["pairs"] += match.size
                    block["segments"] += int(np.count_nonzero(segment));block["valid_projection"] += int(np.count_nonzero(valid))
                    for key, mask in (("all", valid), ("local_pair_match", match & valid), ("local_pair_change", ~match & valid),
                                      ("projection_in_segment", segment), ("projection_outside_segment", valid & ~segment)):
                        hist, outside = histogram2d(r, e, mask, r_edges, e_edges)
                        state["hist"][key] += hist
                        state["outside"][key] += outside
                    for j in range(len(h_ids)):
                        local_change[phase_id, j] += int(np.count_nonzero(~match[:, j]))
                        local_pairs[phase_id, j] += match.shape[0]
                        local_segment[phase_id, j] += int(np.count_nonzero(segment[:, j]))
                        local_projection[phase_id, j] += int(np.count_nonzero(valid[:, j]))
        for j, h in enumerate(h_ids):
            col = h0 + j
            per_h.append({"atom": int(h), "reference_pair_O_atoms": pairs[j].tolist(),
                          "R0_candidate_O_atoms": candidate_ids_all[col, :candidate_counts_all[col]].tolist(),
                          "R0_candidate_count": int(candidate_counts_all[col]), "R0_pair_R_OO_A": float(r0_pair[covered_rows[col]]),
                          "R0_ion_nearest_distance_A": {str(ion_type): float(values[col]) for ion_type, values in ion_distances.items()},
                          "R0_second_to_third_HO_gap_A": float(static_margin[covered_rows[col]]) if np.isfinite(static_margin[covered_rows[col]]) else None,
                          "training_R_OO_A": summary(local_r[:train, j]), "validation_R_OO_A": summary(local_r[train:, j]),
                          "training_bridge_eccentricity": summary(local_e[:train, j]),
                          "validation_bridge_eccentricity": summary(local_e[train:, j]),
                          "training_candidate_local_second_to_third_HO_gap_A": summary(local_gap[:train, j]),
                          "validation_candidate_local_second_to_third_HO_gap_A": summary(local_gap[train:, j]),
                          "training_candidate_local_pair_change_fraction": float(local_change[0, j] / local_pairs[0, j]),
                          "validation_candidate_local_pair_change_fraction": float(local_change[1, j] / local_pairs[1, j]),
                          "training_projection_in_segment_fraction": float(local_segment[0, j] / max(1, local_projection[0, j])),
                          "validation_projection_in_segment_fraction": float(local_segment[1, j] / max(1, local_projection[1, j]))})
    block_reports = []
    for b in blocks:
        s, c = b["sums"], b["counts"]
        block_reports.append({"phase": b["phase"], "start_frame": b["start_frame"], "end_frame_exclusive": b["end_frame_exclusive"],
                              "R_OO_A_mean": float(s[0] / c[0]) if c[0] else None,
                              "bridge_eccentricity_mean": float(s[1] / c[1]) if c[1] else None,
                              "projection_in_segment_fraction": float(b["segments"] / max(1, b["valid_projection"])),
                              "candidate_local_pair_change_fraction": float(b["changes"] / max(1, b["pairs"]))})
    phase_reports = {}
    for phase, state in phases.items():
        sums, counts = state["sums"], state["counts"]
        phase_reports[phase] = {"R_OO_A_mean": float(sums[0] / counts[0]) if counts[0] else None,
                                "bridge_eccentricity_mean": float(sums[1] / counts[1]) if counts[1] else None,
                                "perpendicular_distance_A_mean": float(sums[2] / counts[2]) if counts[2] else None,
                                "distance_asymmetry_mean": float(sums[3] / counts[3]) if counts[3] else None,
                                "projection_in_segment_fraction": float(state["segments"] / max(1, state["valid_projection"])),
                                "candidate_local_pair_change_fraction": float(state["changes"] / max(1, state["pairs"])),
                                "candidate_local_second_to_third_HO_gap_A_mean": float(state["margin_sum"] / state["margin_count"]) if state["margin_count"] else None,
                                "R_OO_eccentricity_histogram": {
                                    "bin_edges_R_OO_A": r_edges.tolist(), "bin_edges_bridge_eccentricity": e_edges.tolist(),
                                    "bin_source": "training fixed-pair samples only; same bins used for validation",
                                    "groups": {key: {"counts": value.tolist(), "out_of_training_range_count": state["outside"][key]}
                                               for key, value in state["hist"].items()}}}
    return {"status": "DIAGNOSTIC_ONLY", "input": str(spool["path"]), "format": "GPJASMP1",
            "frame_count": nf, "training_frames": train, "validation_frames": nf - train,
            "sample_step_interval": sample_interval,
            "sample_step_interval_units": "native spool step units; no time-unit conversion applied",
            "beads": spool["beads"],
            "temperature_K": spool["temperature"], "hydrogen_type": int(hydrogen_type), "oxygen_type": int(oxygen_type),
            "ion_types": list(ion_types),
            "fixed_R0_ion_distance_definition": "minimum-image distance from each fixed O-O midpoint to the nearest atom of each explicit ion type, evaluated once at training R0",
            "candidate_cutoff_A": float(candidate_cutoff), "reported_block_frames": int(block_frames),
            "frame_chunk": FRAME_BLOCK, "hydrogen_chunk": HYDROGEN_BLOCK, "candidate_cap_per_H": MAX_CANDIDATES,
            "fixed_R0_definition": "unwrapped centroid arithmetic mean over the first floor(2F/3) frames",
            "fixed_R0_mean_drift": mean_drift,
            "periodic_image": "component-wise fractional minimum image in the fixed spool cell",
            "candidate_scope": "R0 cutoff neighborhood, capped at the 64 nearest O atoms per H; frames search only that frozen list",
            "candidate_limited": True,
            "R0_candidate_coverage": {"hydrogen_count": int(len(h_all)),
                                   "with_two_or_more_R0_candidates": int(np.count_nonzero(np.asarray(raw_counts) >= 2)),
                                   "fixed_pair_classified_H_count": int(len(covered_h)),
                                   "fixed_pair_classification_fraction": float(len(covered_h) / len(h_all)),
                                   "fixed_pair_unclassified_H_count": int(len(h_all) - len(covered_h)),
                                   "fewer_than_two_R0_candidate_H_count": int(np.count_nonzero(np.asarray(raw_counts) < 2)),
                                   "candidate_lists_truncated_H_count": int(len(truncated)),
                                   "unclassified_H_atom_ids": unclassified_h.tolist(),
                                   "truncated_H_atom_ids": truncated,
                                   "candidate_count_per_H": summary(raw_counts),
                                   "outside_frozen_candidates": "NOT_EVALUATED; per-frame oxygen search is restricted to frozen R0 candidates"},
            "fixed_R0_branch_contract": branch_contract,
            "fixed_pair_metrics": phase_reports, "hydrogens": per_h, "block_means": block_reports,
            "limitations": ["candidate_local_pair_change_fraction is not a global exchange fraction",
                            "centroid-only spool cannot measure bead spread or tunneling rates",
                            "fixed candidates are based on R0 and may miss oxygens outside the cutoff or 64-neighbor cap"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool")
    parser.add_argument("--hydrogen-type", type=int, required=True)
    parser.add_argument("--oxygen-type", type=int, required=True)
    parser.add_argument("--ion-types", type=int, nargs="+", default=[])
    parser.add_argument("--candidate-cutoff", type=float, required=True)
    parser.add_argument("--block-frames", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output:
        input_path, output_path = Path(args.spool).resolve(), Path(args.output).resolve()
        if output_path == input_path:
            parser.error("output path must differ from the input spool")
        if output_path.exists():
            parser.error(f"refusing to overwrite existing report: {output_path}")
    report = json.dumps(diagnose(args.spool, args.hydrogen_type, args.oxygen_type,
                                 args.candidate_cutoff, args.block_frames, args.ion_types), indent=2, sort_keys=True) + "\n"
    if args.output:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(report)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
