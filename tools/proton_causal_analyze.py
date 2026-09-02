#!/usr/bin/env python3
"""Offline reconstruction of the proton defect-causal network.

The input is a raw GPUMD proton_observer.nc.  Candidate matching uses
per-oxygen time indexes and binary-search windows; null statistics are updated
online, so the number of shifts does not multiply memory usage.
"""

from __future__ import annotations

import argparse
import bisect
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
try:
    from netCDF4 import Dataset
except ImportError as error:
    raise SystemExit("proton_causal_analyze.py requires numpy and netCDF4-python") from error


def text_attr(value, default=""):
    if value is None:
        return default
    return value.decode() if isinstance(value, bytes) else str(value)


def float_list(value):
    if value is None:
        return []
    return [float(item) for item in text_attr(value).replace(";", ",").split(",") if item.strip()]


def fields(variable):
    return [item.strip() for item in text_attr(variable.getncattr("field_names"), "").split(",")
            if item.strip()]


def values(group, name, length):
    if group is None or name not in group.variables:
        return {}, np.empty((length, 0), dtype=float)
    variable = group.variables[name]
    array = np.asarray(variable[:], dtype=float).reshape(length, -1)
    return {name: index for index, name in enumerate(fields(variable))}, array


def scalar(group, name, length, dtype, default):
    if group is not None and name in group.variables:
        return np.asarray(group.variables[name][:], dtype=dtype)
    return np.full(length, default, dtype=dtype)


class Event:
    __slots__ = ("row", "start", "end", "first", "commit", "confirm", "h", "low", "high",
                 "source", "target", "dx", "dy", "dz", "fdx", "fdy", "fdz", "fvalid",
                 "nb", "nt", "na", "nc", "bead_valid", "two_well", "strict", "multi", "group")

    def __init__(self, row, start, end, first, commit, confirm, h, low, high, source, target,
                 dx, dy, dz, fdx, fdy, fdz, fvalid, counts, bead):
        self.row = row
        self.start, self.end, self.first = start, end, first
        self.commit, self.confirm = commit, confirm
        self.h, self.low, self.high = h, low, high
        self.source, self.target = source, target
        self.dx, self.dy, self.dz = dx, dy, dz
        self.fdx, self.fdy, self.fdz, self.fvalid = fdx, fdy, fdz, fvalid
        self.nb = counts.get("nH_from_before", 0)
        self.nt = counts.get("nH_to_before", 0)
        self.na = counts.get("nH_from_after", 0)
        self.nc = counts.get("nH_to_after", 0)
        self.bead_valid, self.two_well, self.strict, self.multi = bead
        self.group = -1


def read_events(source):
    group = source.groups.get("attempt")
    if group is None or "time_fs" not in group.variables:
        raise RuntimeError("input NetCDF must contain /attempt/time_fs")
    times = np.asarray(group.variables["time_fs"][:], dtype=float).reshape(-1, 2)
    n = len(times)
    h = scalar(group, "hydrogen", n, np.int32, -1)
    edge_ids = scalar(group, "edge_id", n, np.int32, -1)
    sources = scalar(group, "oxygen_from", n, np.int32, -1)
    targets = scalar(group, "oxygen_target", n, np.int32, -1)
    transferred = scalar(group, "has_transfer", n, np.uint8, 0)
    fvalid = scalar(group, "fractional_step_valid", n, np.uint8, 0)
    names, data = values(group, "value", n)

    def column(name, default=np.nan):
        return data[:, names[name]] if name in names else np.full(n, default)

    transfer = source.groups.get("transfer")
    transfer_rows = {}
    if transfer is not None and "attempt_index" in transfer.variables:
        indices = np.asarray(transfer.variables["attempt_index"][:], dtype=np.int64)
        vnames, vdata = values(transfer, "value", len(indices))
        cnames, cdata = values(transfer, "count", len(indices))
        for row, index in enumerate(indices.tolist()):
            transfer_rows[int(index)] = (vnames, vdata[row] if vdata.size else np.empty(0),
                                         cnames, cdata[row] if cdata.size else np.empty(0))

    bead = source.groups.get("bead")
    bead_valid = np.zeros((n, 2), dtype=np.uint8)
    bead_flags = np.zeros((n, 2, 6), dtype=np.uint8)
    if bead is not None:
        if "valid" in bead.variables:
            bead_valid = np.asarray(bead.variables["valid"][:], dtype=np.uint8).reshape(n, 2)
        if "flag" in bead.variables:
            bead_flags = np.asarray(bead.variables["flag"][:], dtype=np.uint8).reshape(n, 2, -1)
    edge_group = source.groups.get("edge")
    edge_oxygen = (np.asarray(edge_group.variables["oxygen"][:], dtype=np.int64)
                   if edge_group is not None and "oxygen" in edge_group.variables else None)

    events = []
    for row in range(n):
        if not transferred[row] or sources[row] < 0 or targets[row] < 0:
            continue
        first = column("time_first_opposite_fs")[row]
        if not np.isfinite(first):
            first = times[row, 0]
        transfer_row = transfer_rows.get(row)
        counts = {}
        dx = dy = dz = 0.0
        if transfer_row is not None:
            vnames, vdata, cnames, cdata = transfer_row
            counts = {name: int(round(cdata[index])) for name, index in cnames.items()}
            dx = float(vdata[vnames["dx"]]) if "dx" in vnames else 0.0
            dy = float(vdata[vnames["dy"]]) if "dy" in vnames else 0.0
            dz = float(vdata[vnames["dz"]]) if "dz" in vnames else 0.0
        if edge_oxygen is not None and 0 <= edge_ids[row] < len(edge_oxygen):
            low, high = sorted(map(int, edge_oxygen[edge_ids[row]]))
        else:
            low, high = sorted((int(sources[row]), int(targets[row])))
        flag = bead_flags[row]
        bead_summary = (int(np.any(bead_valid[row])), int(np.any(flag[:, 0])) if flag.shape[1] > 0 else 0,
                        int(np.any(flag[:, 4])) if flag.shape[1] > 4 else 0,
                        int(np.any(flag[:, 5])) if flag.shape[1] > 5 else 0)
        events.append(Event(
            row, float(times[row, 0]), float(times[row, 1]), float(first),
            float(column("time_commit_fs")[row]), float(column("time_confirm_fs")[row]),
            int(h[row]), low, high, int(sources[row]), int(targets[row]), dx, dy, dz,
            float(column("fractional_dx")[row]) if np.isfinite(column("fractional_dx")[row]) else 0.0,
            float(column("fractional_dy")[row]) if np.isfinite(column("fractional_dy")[row]) else 0.0,
            float(column("fractional_dz")[row]) if np.isfinite(column("fractional_dz")[row]) else 0.0,
            int(fvalid[row]), counts, bead_summary))
    return events


def concerted_groups(events, sync):
    ordered = sorted(events, key=lambda item: (item.first, item.row))
    groups, next_id, begin = [], 1, 0
    while begin < len(ordered):
        end = begin + 1
        reference = ordered[begin].first
        while end < len(ordered) and ordered[end].first - reference <= sync:
            end += 1
        bucket, parent = ordered[begin:end], list(range(end - begin))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        first_by_oxygen = {}
        for index, event in enumerate(bucket):
            for oxygen in (event.source, event.target):
                if oxygen in first_by_oxygen:
                    other = first_by_oxygen[oxygen]
                    left, right = root(other), root(index)
                    if left != right:
                        parent[right] = left
                else:
                    first_by_oxygen[oxygen] = index
        components = defaultdict(list)
        for index, event in enumerate(bucket):
            components[root(index)].append(event)
        for component in components.values():
            for event in component:
                event.group = next_id
            adjacency = defaultdict(list)
            for event in component:
                adjacency[event.source].append(event.target)
            closed = 0
            for start in adjacency:
                stack, visited = [start], set()
                while stack and not closed:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    for target in adjacency.get(current, ()):
                        if target == start:
                            closed = 1
                            break
                        stack.append(target)
            oxygens = {x for event in component for x in (event.source, event.target)}
            groups.append((next_id, min(e.first for e in component),
                           max(e.first for e in component) - min(e.first for e in component),
                           len(component), len({e.h for e in component}), len(oxygens),
                           len({(e.low, e.high) for e in component}), closed))
            next_id += 1
        begin = end
    return groups, {event.row: event.group for event in events}


def has_continuity(parent, child, carrier):
    if carrier == 0:
        return (parent.nt - 2 > 0 and child.nb - 2 > 0 and child.na < child.nb)
    return (parent.na - 2 < 0 and child.nt - 2 < 0 and child.nc > child.nt)


def build_links(events, search, sync, event_groups):
    by_row = {event.row: event for event in events}
    by_source, by_target = defaultdict(list), defaultdict(list)
    for event in events:
        by_source[event.source].append((event.first, event.row))
        by_target[event.target].append((event.first, event.row))
    for stream in list(by_source.values()) + list(by_target.values()):
        stream.sort()
    source_times = {oxygen: [item[0] for item in stream]
                    for oxygen, stream in by_source.items()}
    target_times = {oxygen: [item[0] for item in stream]
                    for oxygen, stream in by_target.items()}
    links, reversal_seen = [], set()

    def append(parent, child, requested, shared):
        if parent.row == child.row:
            return
        lag = child.first - parent.first
        if not np.isfinite(lag) or abs(lag) > search:
            return
        reversal = parent.source == child.target and parent.target == child.source
        carrier = 2 if reversal else requested
        if reversal and (parent.row, child.row) in reversal_seen:
            return
        if reversal:
            reversal_seen.add((parent.row, child.row))
        temporal = 0 if lag > sync else (1 if lag >= -sync else 2)
        link = {
            "parent": parent.row, "child": child.row, "shared": shared, "carrier": carrier,
            "temporal": temporal, "lag": child.start - parent.first, "lag_first": lag,
            "lag_commit": child.commit - parent.commit, "lag_confirm": child.confirm - parent.confirm,
            "continuity": -1 if reversal else int(has_continuity(parent, child, requested)),
            "same_h": int(parent.h == child.h),
            "same_edge": int(parent.low == child.low and parent.high == child.high),
            "parent_group": event_groups.get(parent.row, -1),
            "child_group": event_groups.get(child.row, -1),
            "primary": 0,
        }
        link["valid"] = int(carrier != 2 and temporal != 2 and link["continuity"] == 1 and
                             not link["same_h"] and not link["same_edge"])
        links.append(link)

    def in_window(parent, requested, shared, candidates):
        times = source_times.get(shared) if requested == 0 else target_times.get(shared)
        if times is None:
            return
        begin = bisect.bisect_left(times, parent.first - search)
        end = bisect.bisect_right(times, parent.first + search)
        for _, row in candidates[begin:end]:
            append(parent, by_row[row], requested, shared)

    for parent in events:
        in_window(parent, 0, parent.target, by_source.get(parent.target, ()))
        in_window(parent, 1, parent.source, by_target.get(parent.source, ()))

    parent_count, child_count = defaultdict(int), defaultdict(int)
    for link in links:
        if link["valid"]:
            parent_count[(link["parent"], link["carrier"])] += 1
            child_count[(link["child"], link["carrier"])] += 1
    for link in links:
        if link["valid"]:
            link["alt_child"] = max(0, parent_count[(link["parent"], link["carrier"])] - 1)
            link["alt_parent"] = max(0, child_count[(link["child"], link["carrier"])] - 1)
        else:
            link["alt_child"] = link["alt_parent"] = 0

    primary = {}
    for index, link in enumerate(links):
        if not (link["valid"] and link["temporal"] == 0 and
                link["parent_group"] != link["child_group"]):
            continue
        key = (link["parent"], link["carrier"])
        old = primary.get(key)
        if old is None or (link["lag_first"], link["lag_commit"], link["child"]) < \
                (links[old]["lag_first"], links[old]["lag_commit"], links[old]["child"]):
            if old is not None:
                links[old]["primary"] = 0
            primary[key] = index
            link["primary"] = 1
    return links, by_row


def build_chains(events, links, groups, event_groups, thresholds, sync):
    by_row = {event.row: event for event in events}
    positions = {event.row: index for index, event in enumerate(events)}
    parent = list(range(len(events)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def unite(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    for link in links:
        if link["valid"]:
            unite(positions[link["parent"]], positions[link["child"]])
    first_group = {}
    for event in events:
        if event.group >= 0:
            if event.group not in first_group:
                first_group[event.group] = positions[event.row]
            else:
                unite(first_group[event.group], positions[event.row])
    episode = {}
    for event in events:
        r = root(positions[event.row])
        episode[r] = min(episode.get(r, event.row + 1), event.row + 1)
    reversal_by_parent = defaultdict(list)
    for index, link in enumerate(links):
        if link["carrier"] == 2:
            reversal_by_parent[link["parent"]].append(index)

    chains, chain_events, next_chain = [], [], 1
    for threshold in thresholds:
        candidates = defaultdict(list)
        parent_count, child_count = defaultdict(int), defaultdict(int)
        for index, link in enumerate(links):
            if (link["valid"] and link["temporal"] == 0 and
                    link["parent_group"] != link["child_group"] and
                    link["lag_first"] > sync and link["lag_first"] <= threshold):
                key = (link["parent"], link["carrier"])
                candidates[key].append(index)
                parent_count[key] += 1
                child_count[(link["child"], link["carrier"])] += 1
        outgoing, alternatives = {}, {}
        for key, indexes in candidates.items():
            best = min(indexes, key=lambda index: (links[index]["lag_first"],
                                                    links[index]["lag_commit"], links[index]["child"]))
            outgoing[key] = best
            selected = links[best]
            alternatives[best] = (
                max(0, child_count[(selected["child"], key[1])] - 1),
                max(0, parent_count[key] - 1))
        incoming = {(links[index]["child"], carrier)
                    for (row, carrier), index in outgoing.items()}
        for carrier in (0, 1):
            roots = [event.row for event in events if (event.row, carrier) not in incoming]
            roots += [event.row for event in events if (event.row, carrier) in incoming]
            used = set()
            for root_row in roots:
                if root_row in used:
                    continue
                path, current = [], root_row
                while current not in used:
                    used.add(current)
                    path.append(current)
                    next_link = outgoing.get((current, carrier))
                    if next_link is None:
                        break
                    current = links[next_link]["child"]
                if not path:
                    continue
                path_edges = {(by_row[row].low, by_row[row].high) for row in path}
                edge_rattling = any(
                    (by_row[links[index]["child"]].low, by_row[links[index]["child"]].high) in path_edges
                    for row in path for index in reversal_by_parent.get(row, ()))
                record = {
                    "id": next_chain, "episode": episode[root(positions[path[0]])],
                    "carrier": carrier, "class": 0, "threshold": threshold,
                    "start": by_row[path[0]].start, "end": by_row[path[-1]].confirm,
                    "n_events": len(path),
                    "start_O": by_row[path[0]].target if carrier == 0 else by_row[path[0]].source,
                    "end_O": by_row[path[-1]].target if carrier == 0 else by_row[path[-1]].source,
                }
                if not np.isfinite(record["end"]):
                    record["end"] = by_row[path[-1]].end
                used_groups = {by_row[row].group for row in path if by_row[row].group >= 0}
                net = np.zeros(3); frac = np.zeros(3); position = np.zeros(3); gaps = []
                path_length = max_span = 0.0
                quantum = two_well = strict = multi = 0
                alternative_parent = alternative_child = 0
                for row in path:
                    event = by_row[row]
                    if event.bead_valid:
                        quantum += 1; two_well += event.two_well
                        strict += event.strict; multi += event.multi
                for left, right in zip(path, path[1:]):
                    index = outgoing.get((left, carrier))
                    if index is None:
                        continue
                    link, child = links[index], by_row[right]
                    sign = 1.0 if carrier == 0 else -1.0
                    displacement = sign * np.array([child.dx, child.dy, child.dz])
                    net += displacement; position += displacement
                    path_length += float(np.linalg.norm(displacement))
                    max_span = max(max_span, float(np.linalg.norm(position)))
                    if child.fvalid:
                        frac += sign * np.array([child.fdx, child.fdy, child.fdz])
                    gaps.append(link["lag_first"])
                    alt = alternatives.get(index, (0, 0))
                    alternative_parent += alt[0]; alternative_child += alt[1]
                rounded = np.rint(frac)
                residual = float(np.linalg.norm(frac - rounded))
                winding_valid = int(all(by_row[row].fvalid for row in path) and residual < 1.0e-5)
                opposite = by_row[path[0]].source if carrier == 0 else by_row[path[0]].target
                closed = int(len(path) > 1 and record["end_O"] == opposite)
                if alternative_parent or alternative_child:
                    record["class"] = 3
                elif edge_rattling:
                    record["class"] = 4
                elif closed and winding_valid and np.any(rounded != 0):
                    record["class"] = 2
                elif closed:
                    record["class"] = 1
                record.update({
                    "path_length": path_length, "net_displacement": float(np.linalg.norm(net)),
                    "net_dx": float(net[0]), "net_dy": float(net[1]), "net_dz": float(net[2]),
                    "max_span": max_span, "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
                    "max_gap": max(gaps) if gaps else 0.0, "n_groups": len(used_groups),
                    "n_quantum": quantum, "fraction_two_well": two_well / quantum if quantum else 0.0,
                    "fraction_strict": strict / quantum if quantum else 0.0,
                    "fraction_multi": multi / quantum if quantum else 0.0,
                    "alt_parent": alternative_parent, "alt_child": alternative_child,
                    "closed_O": closed, "frac_x": float(frac[0]), "frac_y": float(frac[1]),
                    "frac_z": float(frac[2]), "winding_x": int(rounded[0]),
                    "winding_y": int(rounded[1]), "winding_z": int(rounded[2]),
                    "residual": residual, "winding_valid": winding_valid})
                chains.append(record)
                chain_events.extend((next_chain, index, row) for index, row in enumerate(path))
                next_chain += 1
    return chains, chain_events


def bin_index(edges, lag):
    if lag < edges[0] or lag > edges[-1]:
        return -1
    index = bisect.bisect_right(edges, lag) - 1
    if index == len(edges) - 1:
        index -= 1
    return index if 0 <= index < len(edges) - 1 else -1


def lag_histogram(events, links, edges, search, null_shifts, seed):
    number_of_bins = len(edges) - 1
    real = np.zeros((2, number_of_bins), dtype=np.int64)
    for link in links:
        if link["valid"] and link["carrier"] < 2:
            index = bin_index(edges, link["lag_first"])
            if index >= 0:
                real[link["carrier"], index] += 1
    mean = np.zeros((2, number_of_bins), dtype=float)
    m2 = np.zeros((2, number_of_bins), dtype=float)
    completed = 0
    if null_shifts > 0 and events:
        low = min(event.first for event in events)
        high = max(event.first for event in events)
        period = high - low
        if period > 0.0 and np.isfinite(period):
            rng = np.random.default_rng(seed)
            by_row = {event.row: event for event in events}
            for shift_index in range(null_shifts):
                streams = defaultdict(list)
                shifts = {}
                for event in events:
                    for carrier in (0, 1):
                        for role in (0, 1):
                            oxygen = (event.target if carrier == 0 and role == 0 else
                                      event.source if carrier == 0 else
                                      event.source if role == 0 else event.target)
                            key = (carrier, role, oxygen)
                            if key not in shifts:
                                shifts[key] = float(rng.uniform(0.0, period))
                            shifted = (event.first - low + shifts[key]) % period + low
                            streams[key].append((shifted, event.row))
                for stream in streams.values():
                    stream.sort()
                stream_times = {key: [item[0] for item in stream]
                                for key, stream in streams.items()}
                count = np.zeros((2, number_of_bins), dtype=np.int64)
                for carrier in (0, 1):
                    for (stream_carrier, role, oxygen), parents in streams.items():
                        if stream_carrier != carrier or role != 0:
                            continue
                        children = streams.get((carrier, 1, oxygen), ())
                        child_times = stream_times.get((carrier, 1, oxygen), ())
                        for parent_time, parent_row in parents:
                            begin = bisect.bisect_left(child_times, parent_time)
                            end = bisect.bisect_right(child_times, parent_time + search)
                            parent_event = by_row[parent_row]
                            for child_time, child_row in children[begin:end]:
                                child_event = by_row[child_row]
                                if (parent_row == child_row or parent_event.h == child_event.h or
                                        (parent_event.low == child_event.low and
                                         parent_event.high == child_event.high) or
                                        not has_continuity(parent_event, child_event, carrier)):
                                    continue
                                index = bin_index(edges, child_time - parent_time)
                                if index >= 0:
                                    count[carrier, index] += 1
                completed += 1
                delta = count - mean
                mean += delta / completed
                m2 += delta * (count - mean)
                if ((shift_index + 1) % max(1, null_shifts // 10) == 0 or
                        shift_index + 1 == null_shifts):
                    print(f"  causal null: completed shifts = {shift_index + 1}/{null_shifts}", flush=True)
    rows = []
    for carrier in (0, 1):
        for index in range(number_of_bins):
            if completed:
                null_mean = mean[carrier, index]
                null_std = math.sqrt(m2[carrier, index] / max(1, completed - 1))
                causal_g = real[carrier, index] / null_mean if null_mean > 0.0 else math.nan
                standard_error = (causal_g * null_std /
                                  (math.sqrt(completed) * null_mean)
                                  if null_mean > 0.0 else math.nan)
            else:
                null_mean = null_std = causal_g = standard_error = math.nan
            rows.append((carrier, edges[index], edges[index + 1], int(real[carrier, index]),
                         null_mean, null_std, causal_g, standard_error, completed))
    return rows


def group_rows(dataset, name, length):
    group = dataset.createGroup(name)
    group.createDimension("row", length)
    return group


def one_dim(group, name, dtype, data):
    variable = group.createVariable(name, dtype, ("row",), zlib=True, complevel=4)
    if len(data):
        variable[:] = data
    return variable


def two_dim(group, name, dtype, data, dimension, length):
    group.createDimension(dimension, length)
    variable = group.createVariable(name, dtype, ("row", dimension), zlib=True, complevel=4)
    if len(data):
        variable[:] = data
    return variable


def write_output(path, source_path, source, events, groups, links, chains, chain_events, histogram,
                 search, sync, thresholds, bins, null_shifts, seed):
    with Dataset(str(path), "w", format="NETCDF4") as output:
        output.setncattr("format", "gpumd_proton_causal_offline")
        output.setncattr("format_version", "1")
        output.setncattr("causal_analysis_state", "complete")
        output.setncattr("causal_mode", "offline")
        output.setncattr("source_file", str(Path(source_path).resolve()))
        output.setncattr("causal_search_max_fs", float(search))
        output.setncattr("causal_sync_fs", float(sync))
        output.setncattr("causal_gap_thresholds_fs", ",".join(map(str, thresholds)))
        output.setncattr("causal_lag_bin_edges_fs", ",".join(map(str, bins)))
        output.setncattr("causal_null_shifts", int(null_shifts))
        output.setncattr("causal_null_seed", int(seed))
        edge = source.groups.get("edge")
        output.setncattr("source_edge_count", len(edge.dimensions["edge"]) if edge else 0)

        group = group_rows(output, "concerted_group", len(groups))
        one_dim(group, "group_id", "i8", [row[0] for row in groups])
        one_dim(group, "reference_time_fs", "f8", [row[1] for row in groups])
        one_dim(group, "time_span_fs", "f8", [row[2] for row in groups])
        for name, position in (("n_events", 3), ("n_unique_H", 4), ("n_unique_O", 5),
                               ("n_unique_edges", 6), ("has_closed_oxygen_cycle", 7)):
            one_dim(group, name, "u1" if position == 7 else "i4", [row[position] for row in groups])

        members = [(event.group, event.row) for event in events]
        group = group_rows(output, "concerted_member", len(members))
        one_dim(group, "group_id", "i8", [row[0] for row in members])
        one_dim(group, "attempt_index", "i8", [row[1] for row in members])

        group = group_rows(output, "causal_link", len(links))
        for name, key in (("parent_attempt_index", "parent"), ("child_attempt_index", "child")):
            one_dim(group, name, "i8", [link[key] for link in links])
        one_dim(group, "shared_oxygen", "i4", [link["shared"] for link in links])
        one_dim(group, "carrier_type", "u1", [link["carrier"] for link in links])
        one_dim(group, "temporal_type", "u1", [link["temporal"] for link in links])
        variable = two_dim(group, "value", "f8", np.asarray(
            [[link["lag"], link["lag_first"], link["lag_commit"], link["lag_confirm"]]
             for link in links], dtype=float), "value", 4)
        variable.setncattr("field_names", "causal_lag_fs,lag_first_opposite_fs,lag_commit_fs,lag_confirm_fs")
        variable = two_dim(group, "flag", "u1", np.asarray(
            [[2 if link["continuity"] < 0 else link["continuity"], link["same_h"],
              link["same_edge"], link["valid"], link["primary"]] for link in links], dtype=np.uint8),
            "flag", 5)
        variable.setncattr("field_names", "defect_continuity,same_hydrogen,same_edge,valid_relay,primary_link")
        variable = two_dim(group, "alternative", "i4", np.asarray(
            [[link["alt_parent"], link["alt_child"]] for link in links], dtype=np.int32),
            "alternative", 2)
        variable.setncattr("field_names", "alternative_parent_count,alternative_child_count")
        two_dim(group, "concerted_group_id", "i8", np.asarray(
            [[link["parent_group"], link["child_group"]] for link in links], dtype=np.int64),
            "endpoint", 2)

        group = group_rows(output, "chain", len(chains))
        for name, key in (("chain_id", "id"), ("episode_id", "episode")):
            one_dim(group, name, "i8", [chain[key] for chain in chains])
        one_dim(group, "carrier_type", "u1", [chain["carrier"] for chain in chains])
        one_dim(group, "chain_class", "u1", [chain["class"] for chain in chains])
        value_names = ("lag_threshold_fs,start_time_fs,end_time_fs,path_length_A,net_displacement_A,"
                       "net_dx,net_dy,net_dz,max_span_A,mean_gap_fs,max_gap_fs,fraction_two_well,"
                       "fraction_strict_tunneling_like,fraction_multi_kink,fractional_net_x,"
                       "fractional_net_y,fractional_net_z,winding_residual")
        variable = two_dim(group, "value", "f8", np.asarray([
            [chain["threshold"], chain["start"], chain["end"], chain["path_length"],
             chain["net_displacement"], chain["net_dx"], chain["net_dy"], chain["net_dz"],
             chain["max_span"], chain["mean_gap"], chain["max_gap"], chain["fraction_two_well"],
             chain["fraction_strict"], chain["fraction_multi"], chain["frac_x"], chain["frac_y"],
             chain["frac_z"], chain["residual"]] for chain in chains], dtype=float), "value", 18)
        variable.setncattr("field_names", value_names)
        count_names = ("n_events,n_concerted_groups,start_O,end_O,n_quantum_valid,"
                       "alternative_parent_count,alternative_child_count,closed_by_oxygen_id,"
                       "winding_x,winding_y,winding_z,winding_valid")
        variable = two_dim(group, "count", "i4", np.asarray([
            [chain["n_events"], chain["n_groups"], chain["start_O"], chain["end_O"], chain["n_quantum"],
             chain["alt_parent"], chain["alt_child"], chain["closed_O"], chain["winding_x"],
             chain["winding_y"], chain["winding_z"], chain["winding_valid"]] for chain in chains], dtype=np.int32),
            "count", 12)
        variable.setncattr("field_names", count_names)

        group = group_rows(output, "chain_event", len(chain_events))
        one_dim(group, "chain_id", "i8", [row[0] for row in chain_events])
        one_dim(group, "position", "i8", [row[1] for row in chain_events])
        one_dim(group, "attempt_index", "i8", [row[2] for row in chain_events])

        group = group_rows(output, "causal_lag_histogram", len(histogram))
        one_dim(group, "carrier_type", "u1", [row[0] for row in histogram])
        variable = two_dim(group, "value", "f8", np.asarray(
            [list(row[1:3]) + list(row[4:8]) for row in histogram], dtype=float), "value", 6)
        variable.setncattr("field_names", "lag_bin_low_fs,lag_bin_high_fs,null_mean_count,null_std_count,g_causal,g_causal_standard_error")
        one_dim(group, "real_count", "i8", [row[3] for row in histogram])
        one_dim(group, "n_null_shifts", "i4", [row[8] for row in histogram])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--search", type=float, default=None)
    parser.add_argument("--sync", type=float, default=None)
    parser.add_argument("--gaps", default=None, help="comma-separated chain gap thresholds in fs")
    parser.add_argument("--lag-bins", default=None, help="comma-separated lag-bin edges in fs")
    parser.add_argument("--null-shifts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_name("proton_causal.nc")
    with Dataset(str(args.input), "r") as source:
        def source_attr(name, default):
            return source.getncattr(name) if name in source.ncattrs() else default

        search = args.search if args.search is not None else float(source_attr("causal_search_max_fs", 200.0))
        sync = args.sync if args.sync is not None else float(source_attr("causal_sync_fs", 2.0))
        gaps = float_list(args.gaps) if args.gaps is not None else float_list(
            source_attr("causal_gap_thresholds_fs", ""))
        bins = float_list(args.lag_bins) if args.lag_bins is not None else float_list(
            source_attr("causal_lag_bin_edges_fs", ""))
        null_shifts = args.null_shifts if args.null_shifts is not None else int(
            source_attr("causal_null_shifts", 0))
        seed = args.seed if args.seed is not None else int(source_attr("causal_null_seed", 1))
        if not gaps:
            gaps = [search]
        if not bins:
            bins = [0.0, search]
        bins = sorted(set(bins))
        if bins[-1] < search:
            bins.append(search)
        if (len(bins) < 2 or bins[0] < 0.0 or
                any(right <= left for left, right in zip(bins, bins[1:]))):
            raise ValueError("lag-bin edges must be strictly increasing and non-negative")
        if search <= 0.0 or sync < 0.0 or sync > search:
            raise ValueError("require search > 0 and 0 <= sync <= search")
        events = read_events(source)
        print(f"raw events with transfers = {len(events)}", flush=True)
        groups, event_groups = concerted_groups(events, sync)
        links, _ = build_links(events, search, sync, event_groups)
        chains, chain_events = build_chains(events, links, groups, event_groups, gaps, sync)
        histogram = lag_histogram(events, links, bins, search, max(0, null_shifts), seed)
        write_output(output, args.input, source, events, groups, links, chains, chain_events,
                     histogram, search, sync, gaps, bins, max(0, null_shifts), seed)
        print(f"wrote {output}: {len(groups)} groups, {len(links)} links, "
              f"{len(chains)} chains", flush=True)


if __name__ == "__main__":
    main()
