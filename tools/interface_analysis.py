#!/usr/bin/env python3
"""Block-averaged NEMD interface resistance and conductivity analysis."""

import argparse
import html
import json
import math
import statistics
from pathlib import Path


FLUX_W_M2_PER_EV_PS_A2 = 1.602176634e13
PROFILE_COLUMNS_V1 = (
    'step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A '
    'coordinate_center_A material_id atom_count kinetic_energy_eV '
    'potential_energy_eV temperature_K dof samples bath_source_eV bath_sink_eV'
).split()
PROFILE_COLUMNS = (
    'step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A '
    'coordinate_center_A material_id atom_count kinetic_energy_eV '
    'potential_energy_eV temperature_K dof samples temperature_samples '
    'bath_source_eV bath_sink_eV'
).split()


def read_profile(path):
    metadata = {}
    columns = None
    frames = {}
    version = None
    previous_key = None
    previous_time = None
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                comment = stripped[1:].strip()
                parts = comment.split(maxsplit=1)
                if comment.startswith('compute_chunk interface_profile version '):
                    try:
                        version = int(comment.rsplit(maxsplit=1)[1])
                    except ValueError as error:
                        raise ValueError(f'{path}:{line_number}: invalid profile version') from error
                    continue
                header = parts
                if len(header) == 2 and header[0] == 'columns':
                    columns = header[1].split()
                elif len(header) == 2:
                    metadata[header[0]] = header[1]
                continue
            values = stripped.split()
            if version not in (1, 2):
                raise ValueError(f'{path}:{line_number}: expected interface profile version 1 or 2')
            expected_columns = PROFILE_COLUMNS_V1 if version == 1 else PROFILE_COLUMNS
            if columns != expected_columns:
                raise ValueError(f'{path}: columns do not match profile version {version}')
            if len(values) != len(columns):
                raise ValueError(f'{path}:{line_number}: expected {len(columns)} columns')
            row = dict(zip(columns, (float(value) for value in values)))
            for column in ('step', 'bin_id', 'material_id', 'samples'):
                value = row[column]
                if not math.isfinite(value) or not value.is_integer():
                    raise ValueError(f'{path}:{line_number}: {column} must be an integer')
                row[column] = int(value)
            if version == 2:
                value = row['temperature_samples']
                if not math.isfinite(value) or not value.is_integer():
                    raise ValueError(
                        f'{path}:{line_number}: temperature_samples must be an integer')
                row['temperature_samples'] = int(value)
            if not all(math.isfinite(row[key]) for key in (
                    'time_ps', 'block_start_ps', 'coordinate_min_A', 'coordinate_max_A',
                    'coordinate_center_A', 'atom_count', 'kinetic_energy_eV',
                    'potential_energy_eV', 'dof')):
                raise ValueError(f'{path}:{line_number}: non-finite profile value')
            if (row['samples'] <= 0 or row['atom_count'] < 0.0 or row['dof'] < 0.0 or
                    row['kinetic_energy_eV'] < 0.0 or row['time_ps'] < row['block_start_ps']):
                raise ValueError(f'{path}:{line_number}: invalid count, energy, or block time')
            if not (math.isnan(row['bath_source_eV']) and math.isnan(row['bath_sink_eV'])):
                if not (math.isfinite(row['bath_source_eV']) and
                        math.isfinite(row['bath_sink_eV'])):
                    raise ValueError(f'{path}:{line_number}: bath counters must both be finite or nan')
            if version == 2:
                if not 0 <= row['temperature_samples'] <= row['samples']:
                    raise ValueError(f'{path}:{line_number}: invalid temperature_samples')
                if row['temperature_samples'] == 0:
                    if not math.isnan(row['temperature_K']):
                        raise ValueError(
                            f'{path}:{line_number}: temperature must be nan without valid samples')
                elif (not math.isfinite(row['temperature_K']) or row['temperature_K'] < 0.0 or
                      row['atom_count'] <= 0.0 or row['dof'] <= 0.0):
                    raise ValueError(f'{path}:{line_number}: invalid sampled temperature')
            elif (row['atom_count'] <= 0.0 or row['dof'] <= 0.0 or
                  not math.isfinite(row['temperature_K']) or row['temperature_K'] < 0.0):
                # Version 1 cannot identify how many temperature samples had positive DOF.
                row['temperature_K'] = math.nan
            key = (row['step'], row['time_ps'])
            frame = frames.get(key)
            if frame is None:
                if previous_time is not None and row['time_ps'] <= previous_time:
                    raise ValueError(f'{path}:{line_number}: frame times must be strictly increasing')
                if key in frames:
                    raise ValueError(f'{path}:{line_number}: profile frame rows are not contiguous')
                frame = {
                    'step': row['step'],
                    'time_ps': row['time_ps'],
                    'block_start_ps': row['block_start_ps'],
                    'samples': row['samples'],
                    'bath_source_eV': row['bath_source_eV'],
                    'bath_sink_eV': row['bath_sink_eV'],
                    'bins': [],
                }
                frames[key] = frame
                previous_key = key
                previous_time = row['time_ps']
            elif key != previous_key:
                raise ValueError(f'{path}:{line_number}: profile frame rows are not contiguous')
            elif (not _same_number(row['block_start_ps'], frame['block_start_ps']) or
                  row['samples'] != frame['bins'][0]['samples'] or
                  not _same_number(row['bath_source_eV'], frame['bath_source_eV']) or
                  not _same_number(row['bath_sink_eV'], frame['bath_sink_eV'])):
                raise ValueError(f'{path}:{line_number}: block metadata differs within a frame')
            if any(existing['bin_id'] == row['bin_id'] for existing in frame['bins']):
                raise ValueError(f'{path}:{line_number}: duplicate bin in profile frame')
            frame['bins'].append(row)
    if not frames:
        raise ValueError(f'{path}: no profile data')
    if version not in (1, 2):
        raise ValueError(f'{path}: missing interface profile version')
    required_metadata = (
        'axis', 'timestep_fs', 'sample_interval', 'output_interval_samples',
        'bin_width_A', 'box_thickness_A', 'cross_section_A2', 'interface_coordinate_A',
        'material_lower_id', 'material_upper_id', 'bath_energy_sign')
    for key in required_metadata:
        if key not in metadata:
            raise ValueError(f'{path}: missing profile metadata {key}')
    if metadata['bath_energy_sign'] != 'system_to_bath':
        raise ValueError(f'{path}: unsupported bath_energy_sign')
    if metadata['axis'] not in ('x', 'y', 'z'):
        raise ValueError(f'{path}: invalid profile axis')
    try:
        width = float(metadata['bin_width_A'])
        thickness = float(metadata['box_thickness_A'])
        area = float(metadata['cross_section_A2'])
        timestep_fs = float(metadata['timestep_fs'])
        sample_interval = int(metadata['sample_interval'])
        interface_coordinate = float(metadata['interface_coordinate_A'])
        output_samples = int(metadata['output_interval_samples'])
        lower_id = int(metadata['material_lower_id'])
        upper_id = int(metadata['material_upper_id'])
        expected_bin_count = int(metadata['bin_count']) if version == 2 else math.ceil(thickness / width)
    except (KeyError, ValueError) as error:
        raise ValueError(f'{path}: invalid profile geometry metadata') from error
    if (not all(math.isfinite(value) for value in
                (width, thickness, area, timestep_fs, interface_coordinate)) or
            width <= 0.0 or thickness <= 0.0 or area <= 0.0 or output_samples <= 0 or
            timestep_fs <= 0.0 or sample_interval <= 0 or expected_bin_count <= 0 or
            expected_bin_count != math.ceil(thickness / width) or
            not 0.0 < interface_coordinate < thickness or lower_id < 0 or upper_id < 0 or
            lower_id == upper_id):
        raise ValueError(f'{path}: invalid profile geometry metadata')
    for index, frame in enumerate(frames.values()):
        if frame['samples'] > output_samples:
            raise ValueError(f'{path}: frame sample count exceeds output_interval_samples')
        frame['partial'] = frame['samples'] < output_samples
        if frame['partial'] and index != len(frames) - 1:
            raise ValueError(f'{path}: only the final profile frame may be partial')
    if version == 2:
        if metadata.get('geometry_valid') != 'orthogonal_static_runtime_checked':
            raise ValueError(f'{path}: profile geometry was not validated')
        pbc = metadata.get('pbc', '').split()
        if len(pbc) != 3 or any(value not in ('0', '1') for value in pbc):
            raise ValueError(f'{path}: invalid PBC metadata')
    expected_bins = set(range(expected_bin_count))
    ordered = list(frames.values())
    reference_geometry = None
    for frame in ordered:
        frame['bins'].sort(key=lambda row: row['bin_id'])
        if {row['bin_id'] for row in frame['bins']} != expected_bins:
            raise ValueError(f'{path}: incomplete bin set at time {frame["time_ps"]}')
        geometry = []
        for row in frame['bins']:
            if row['material_id'] not in (-1, lower_id, upper_id):
                raise ValueError(f'{path}: unexpected material ID {row["material_id"]}')
            expected_min = row['bin_id'] * width
            expected_max = min((row['bin_id'] + 1) * width, thickness)
            if not all(math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)
                       for actual, expected in ((row['coordinate_min_A'], expected_min),
                                                (row['coordinate_max_A'], expected_max),
                                                (row['coordinate_center_A'],
                                                 0.5 * (expected_min + expected_max)))):
                raise ValueError(f'{path}: bin geometry differs from profile metadata')
            geometry.append((row['bin_id'], row['coordinate_min_A'], row['coordinate_max_A'],
                             row['coordinate_center_A'], row['material_id']))
        if reference_geometry is None:
            reference_geometry = geometry
        elif geometry != reference_geometry:
            raise ValueError(f'{path}: bin geometry or material IDs changed between frames')
    return {'metadata': metadata, 'frames': ordered, 'version': version}


def _slope(x, y):
    if len(x) < 2 or len(x) != len(y):
        raise ValueError('a slope requires at least two matching samples')
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0.0:
        raise ValueError('sample times must not all be equal')
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator


def _same_number(left, right):
    return (left == right) or (math.isnan(left) and math.isnan(right))


def _standard_error(values):
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))


def _window_frames(frames, time_window):
    start, end = time_window
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError('steady-state time window must have finite increasing endpoints')
    return [frame for frame in frames
            if frame['block_start_ps'] >= start and frame['time_ps'] <= end]


def _complete_blocks(frames, time_window, block_frames):
    if block_frames < 2:
        raise ValueError('block_frames must be >= 2')
    selected = [frame for frame in _window_frames(frames, time_window)
                if not frame.get('partial', False)]
    count = len(selected) // block_frames
    if count < 2:
        raise ValueError('steady-state window needs at least two complete time blocks')
    selected = selected[:count * block_frames]
    return [selected[i:i + block_frames] for i in range(0, len(selected), block_frames)]


def _bath_power_for_frames(frames):
    if len(frames) < 2:
        raise ValueError('a bath-power estimate requires at least two profile frames')
    source = [frame['bath_source_eV'] for frame in frames]
    sink = [frame['bath_sink_eV'] for frame in frames]
    if not all(math.isfinite(value) for value in source + sink):
        raise ValueError('profile has no finite heat-bath counters; use heat_lan, heat_nhc, or heat_bdp')
    times = [frame['time_ps'] for frame in frames]
    hot = -_slope(times, source)
    cold = -_slope(times, sink)
    power = 0.5 * (hot - cold)
    denominator = abs(hot) + abs(cold)
    balance = abs(hot + cold) / denominator if denominator > 0.0 else math.nan
    return {
        'hot_eV_ps': hot, 'cold_eV_ps': cold,
        'power_eV_ps': power, 'balance_fraction': balance,
    }


def estimate_bath_power(frames, time_window, block_frames=10):
    blocks = _complete_blocks(frames, time_window, block_frames)
    selected = _window_frames(frames, time_window)
    power = _bath_power_for_frames(selected)
    block_power = [_bath_power_for_frames(block)['power_eV_ps'] for block in blocks]
    return {
        **power,
        'power_error_eV_ps': _standard_error(block_power),
        'block_power_eV_ps': block_power,
        'frames': len(selected),
    }


def _profile_means(frames):
    by_bin = {}
    for frame in frames:
        for row in frame['bins']:
            by_bin.setdefault(row['bin_id'], []).append(row)

    result = []
    for bin_id, samples in sorted(by_bin.items()):
        base = samples[0]
        frame_weights = [row['samples'] for row in samples]
        has_temperature_counts = 'temperature_samples' in base
        valid_temperature_rows = [row for row in samples
                                  if math.isfinite(row['temperature_K']) and
                                  (not has_temperature_counts or row['temperature_samples'] > 0)]
        temperature_weights = ([row['temperature_samples'] for row in valid_temperature_rows]
                               if has_temperature_counts else
                               [1 for _ in valid_temperature_rows])
        temperature_total = sum(temperature_weights)
        result.append({
            'bin_id': bin_id,
            'coordinate_min_A': base['coordinate_min_A'],
            'coordinate_max_A': base['coordinate_max_A'],
            'coordinate_center_A': base['coordinate_center_A'],
            'material_id': base['material_id'],
            'atom_count': sum(row['atom_count'] * weight
                              for row, weight in zip(samples, frame_weights)) /
                          sum(frame_weights),
            'kinetic_energy_eV': sum(row['kinetic_energy_eV'] * weight
                                     for row, weight in zip(samples, frame_weights)) /
                                 sum(frame_weights),
            'potential_energy_eV': sum(row['potential_energy_eV'] * weight
                                       for row, weight in zip(samples, frame_weights)) /
                                   sum(frame_weights),
            'temperature_K': (sum(row['temperature_K'] * weight for row, weight in
                                  zip(valid_temperature_rows, temperature_weights)) /
                              temperature_total if temperature_total else math.nan),
            'temperature_error_K': math.nan,
            'temperature_samples': (sum(row['temperature_samples'] for row in samples)
                                    if 'temperature_samples' in base else math.nan),
            'dof': sum(row['dof'] * weight for row, weight in zip(samples, frame_weights)) /
                   sum(frame_weights),
            'samples': sum(frame_weights),
        })
    return result


def _profile_statistics(frames, time_window, block_frames, blocks=None):
    blocks = blocks or _complete_blocks(frames, time_window, block_frames)
    selected = _window_frames(frames, time_window)
    if (selected and 'temperature_samples' not in selected[0]['bins'][0] and
            len({frame['samples'] for frame in selected}) > 1):
        raise ValueError('version 1 temperatures cannot be weighted across unequal frame samples')
    result = _profile_means(selected)
    block_profiles = [_profile_means(block) for block in blocks]
    for row_index, row in enumerate(result):
        row['temperature_error_K'] = _standard_error(
            [profile[row_index]['temperature_K'] for profile in block_profiles])
    return result


def _fit_line(rows, reference_coordinate):
    valid = [row for row in rows if math.isfinite(row['temperature_K'])]
    if len(valid) < 2:
        raise ValueError('temperature fit requires at least two bins')
    x = [row['coordinate_center_A'] - reference_coordinate for row in valid]
    y = [row['temperature_K'] for row in valid]
    errors = [row['temperature_error_K'] for row in valid]
    use_errors = all(math.isfinite(error) and error > 0.0 for error in errors)
    weights = [1.0 / (error * error) for error in errors] if use_errors else [1.0] * len(valid)
    total = sum(weights)
    sx = sum(w * value for w, value in zip(weights, x))
    sy = sum(w * value for w, value in zip(weights, y))
    sxx = sum(w * value * value for w, value in zip(weights, x))
    sxy = sum(w * a * b for w, a, b in zip(weights, x, y))
    determinant = total * sxx - sx * sx
    if determinant <= 0.0:
        raise ValueError('temperature fit coordinates must be distinct')
    slope = (total * sxy - sx * sy) / determinant
    intercept = (sxx * sy - sx * sxy) / determinant
    residual_sum = sum(w * (value - intercept - slope * position) ** 2
                       for w, position, value in zip(weights, x, y))
    mean = sy / total
    total_sum = sum(w * (value - mean) ** 2 for w, value in zip(weights, y))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    scale = 1.0 if use_errors else (
        residual_sum / (len(valid) - 2) if len(valid) > 2 else math.nan)
    slope_error = math.sqrt(scale * total / determinant) if math.isfinite(scale) else math.nan
    intercept_error = math.sqrt(scale * sxx / determinant) if math.isfinite(scale) else math.nan
    return {
        'slope_K_A': slope,
        'intercept_K': intercept,
        'slope_error_K_A': slope_error,
        'intercept_error_K': intercept_error,
        'r_squared': r_squared,
        'bins': len(valid),
    }


def _mean_temperature(rows):
    valid = [row for row in rows if math.isfinite(row['temperature_K']) and
             row['atom_count'] > 0.0]
    total = sum(row['atom_count'] for row in valid)
    return (sum(row['temperature_K'] * row['atom_count'] for row in valid) / total
            if total else math.nan)


def _interpolate(rows, coordinate, material_id):
    candidates = sorted(
        (row for row in rows if row['material_id'] == material_id),
        key=lambda row: row['coordinate_center_A'])
    for row in candidates:
        if math.isclose(row['coordinate_center_A'], coordinate, abs_tol=1e-8):
            return row['temperature_K'], row['temperature_error_K']
    for left, right in zip(candidates, candidates[1:]):
        x0, x1 = left['coordinate_center_A'], right['coordinate_center_A']
        if x0 <= coordinate <= x1:
            fraction = (coordinate - x0) / (x1 - x0)
            value = left['temperature_K'] * (1.0 - fraction) + right['temperature_K'] * fraction
            error = math.hypot(
                left['temperature_error_K'] * (1.0 - fraction),
                right['temperature_error_K'] * fraction)
            return value, error
    return None


def _rows_in_region(rows, material_id, region):
    low, high = region
    return [row for row in rows if row['material_id'] == material_id and
            low <= row['coordinate_center_A'] <= high]


def _cumulative_rows(rows, fit, interface_coordinate, side, material, flux, flux_error,
                     interface_exclusion, max_distance):
    result = []
    for row in rows:
        coordinate = row['coordinate_center_A']
        distance = abs(coordinate - interface_coordinate)
        is_wrong_side = (coordinate >= interface_coordinate if side == 'left'
                         else coordinate <= interface_coordinate)
        if distance <= interface_exclusion or distance > max_distance or is_wrong_side:
            continue
        delta_t = abs(fit['intercept_K'] - row['temperature_K'])
        if delta_t <= 0.0:
            continue
        delta_t_error = math.hypot(fit['intercept_error_K'], row['temperature_error_K'])
        resistance = delta_t / abs(flux)
        resistance_error = (
            math.hypot(delta_t_error / abs(flux),
                       delta_t * flux_error / (abs(flux) ** 2))
            if math.isfinite(flux_error) else math.nan)
        kappa = abs(flux) * distance * 1e-10 / delta_t
        kappa_error = (
            abs(kappa) * math.hypot(flux_error / abs(flux), delta_t_error / delta_t)
            if math.isfinite(flux_error) and abs(flux) > 0.0 else math.nan)
        result.append({
            'side': side, 'material': material, 'distance_A': distance,
            'coordinate_A': coordinate, 'temperature_K': row['temperature_K'],
            'temperature_error_K': row['temperature_error_K'],
            'delta_temperature_K': delta_t, 'delta_temperature_error_K': delta_t_error,
            'resistance_m2K_W': resistance, 'resistance_error_m2K_W': resistance_error,
            'kappa_eff_W_mK': kappa, 'kappa_eff_error_W_mK': kappa_error,
        })
    return sorted(result, key=lambda row: row['distance_A'])


def _local_rows(rows, region, material, side, source_side, flux, flux_error, window_bins):
    low, high = region
    valid = [row for row in rows if row['material_id'] == material['id'] and
             low <= row['coordinate_center_A'] <= high]
    result = []
    half = window_bins // 2
    flux_signed = abs(flux) * (1.0 if source_side == 'lower' else -1.0)
    for i, row in enumerate(valid):
        start = max(0, min(i - half, len(valid) - window_bins))
        window = valid[start:start + window_bins]
        if len(window) < 2:
            continue
        fit = _fit_line(window, row['coordinate_center_A'])
        gradient = fit['slope_K_A']
        kappa = -flux_signed * 1e-10 / gradient if gradient != 0.0 else math.nan
        error = (abs(kappa) * math.hypot(flux_error / abs(flux),
                                         fit['slope_error_K_A'] / abs(gradient))
                 if math.isfinite(kappa) and math.isfinite(flux_error) and
                 math.isfinite(fit['slope_error_K_A']) and gradient != 0.0 else math.nan)
        result.append({
            'side': side, 'material': material['name'],
            'distance_A': abs(row['coordinate_center_A'] - material['interface_A']),
            'coordinate_A': row['coordinate_center_A'],
            'gradient_K_A': gradient, 'gradient_error_K_A': fit['slope_error_K_A'],
            'kappa_local_W_mK': kappa, 'kappa_local_error_W_mK': error,
            'r_squared': fit['r_squared'], 'window_bins': len(window),
        })
    return result


def _intervals(value, run_name):
    result = []
    for interval in value or []:
        if (len(interval) != 2 or not all(math.isfinite(float(x)) for x in interval) or
                float(interval[0]) >= float(interval[1])):
            raise ValueError(f'{run_name}: bath_regions_A must contain increasing finite pairs')
        result.append((float(interval[0]), float(interval[1])))
    return result


def _overlaps(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


def _reject_bath_overlap(run_name, region, bath_regions, context):
    if any(_overlaps(region, bath) for bath in bath_regions):
        raise ValueError(f'{run_name}: {context} overlaps a configured bath region')


def _outside_baths(rows, bath_regions):
    return [row for row in rows if not any(
        _overlaps((row['coordinate_min_A'], row['coordinate_max_A']), bath)
        for bath in bath_regions)]


def _format_intervals(intervals):
    return (';'.join(f'{low:g}-{high:g}' for low, high in intervals)
            if intervals else 'none_configured')


def _interface_metrics(run, interface, interface_coordinate, source_side, window_bins,
                       interface_exclusion, max_distance):
    sides = {}
    cumulative = {}
    local = []
    for side in ('left', 'right'):
        material = interface[side]
        material_id = int(material['id'])
        rows = _outside_baths(
            _rows_in_region(run['rows'], material_id, material['region_A']),
            run['bath_regions_A'])
        fit_rows = _outside_baths(
            _rows_in_region(run['rows'], material_id, material['fit_A']),
            run['bath_regions_A'])
        fit = _fit_line(fit_rows, interface_coordinate)
        sides[side] = {'material': material, 'fit': fit, 'rows': rows}
        cumulative[side] = _cumulative_rows(
            rows, fit, interface_coordinate, side, material['name'], run['flux_W_m2'],
            run['flux_error_W_m2'], interface_exclusion, max_distance)
        local_material = {**material, 'interface_A': interface_coordinate}
        local.extend(_local_rows(
            rows, material['region_A'], local_material, side, source_side,
            run['flux_W_m2'], run['flux_error_W_m2'], window_bins))

    left_t = sides['left']['fit']['intercept_K']
    right_t = sides['right']['fit']['intercept_K']
    delta_t = abs(left_t - right_t)
    flux_abs = abs(run['flux_W_m2'])
    resistance = delta_t / flux_abs if flux_abs > 0.0 else math.nan
    return {
        'sides': sides, 'cumulative': cumulative, 'local': local,
        'interface': {
            'T_lower_interface_K': left_t,
            'T_upper_interface_K': right_t,
            'delta_T_interface_K': delta_t,
            'R_interface_m2K_W': resistance,
        },
    }


def _segment_result(run, segment):
    material_id = int(segment['material_id'])
    rows = _outside_baths(run['rows'], run['bath_regions_A'])
    rows = [row for row in rows if row['material_id'] == material_id]
    coordinate1 = float(segment['coordinate1_A'])
    coordinate2 = float(segment['coordinate2_A'])
    t1 = _interpolate(rows, coordinate1, material_id)
    t2 = _interpolate(rows, coordinate2, material_id)
    if t1 is None or t2 is None:
        raise ValueError(f"segment {segment['run']}:{coordinate1}-{coordinate2} "
                         'is outside populated bins for its material')
    delta_t = abs(t2[0] - t1[0])
    flux = abs(run['flux_W_m2'])
    resistance = delta_t / flux if flux > 0.0 else math.nan
    kappa = (flux * abs(coordinate2 - coordinate1) * 1e-10 / delta_t
             if delta_t > 0.0 else math.nan)
    return {
        'run': segment['run'], 'side': segment.get('side', 'unspecified'),
        'material_id': material_id, 'coordinate1_A': coordinate1,
        'coordinate2_A': coordinate2, 'T1_K': t1[0], 'T2_K': t2[0],
        'delta_T_K': delta_t, 'flux_W_m2': run['flux_W_m2'],
        'R_seg_m2K_W': resistance, 'kappa_seg_W_mK': kappa,
        'mean_temperature_K': run['mean_temperature_K'],
        'boundary_condition': run['boundary_condition'],
    }


def _pure_reference_profile(run, reference):
    material_id = int(reference['material_id'])
    rows = _outside_baths(run['rows'], run['bath_regions_A'])
    fit_rows = _rows_in_region(rows, material_id, reference['fit_A'])
    fit = _fit_line(fit_rows, float(reference['reference_coordinate_A']))
    return {
        'rows': rows, 'fit': fit,
        'mean_temperature_K': _mean_temperature(_rows_in_region(
            rows, material_id, reference['fit_A'])),
    }


def _pure_point(run, reference, profile, side, distance):
    sign = -1.0 if side == 'left' else 1.0
    target = float(reference['reference_coordinate_A']) + sign * distance
    material_id = int(reference['material_id'])
    interpolated = _interpolate(profile['rows'], target, material_id)
    if interpolated is None:
        return None
    temperature = interpolated[0]
    delta = abs(profile['fit']['intercept_K'] - temperature)
    flux = abs(run['flux_W_m2'])
    if delta <= 0.0 or flux <= 0.0:
        return None
    return {
        'resistance_m2K_W': delta / flux,
        'kappa_eff_W_mK': flux * distance * 1e-10 / delta,
    }


def _comparison_row(side, material, distance, hetero_point, hetero, pure, pure_profile,
                    reference, hetero_region, temperature_tolerance):
    pure_point = _pure_point(pure, reference, pure_profile, side, distance)
    if pure_point is None:
        return None
    r_hetero = hetero_point['resistance_m2K_W']
    r_pure = pure_point['resistance_m2K_W']
    k_hetero = hetero_point['kappa_eff_W_mK']
    k_pure = pure_point['kappa_eff_W_mK']
    delta_temperature = abs(hetero['mean_temperature_K'] -
                            pure_profile['mean_temperature_K'])
    status = ('NOT_CHECKED' if temperature_tolerance is None else
              ('MATCHED' if delta_temperature <= temperature_tolerance else 'NOT_MATCHED'))
    return {
        'side': side, 'material': material, 'distance_A': distance,
        'comparison_hetero_min_A': hetero_region[0],
        'comparison_hetero_max_A': hetero_region[1],
        'comparison_pure_min_A': float(reference['fit_A'][0]),
        'comparison_pure_max_A': float(reference['fit_A'][1]),
        'bath_regions_hetero_A': _format_intervals(hetero['bath_regions_A']),
        'bath_regions_pure_A': _format_intervals(pure['bath_regions_A']),
        'temperature_match': status,
        'T_mean_hetero_K': hetero['mean_temperature_K'],
        'T_mean_pure_K': pure_profile['mean_temperature_K'],
        'power_hetero_eV_ps': hetero['power']['power_eV_ps'],
        'power_pure_eV_ps': pure['power']['power_eV_ps'],
        'area_hetero_A2': hetero['area_A2'], 'area_pure_A2': pure['area_A2'],
        'power_fraction_hetero': hetero['power_fraction'],
        'power_fraction_pure': pure['power_fraction'],
        'boundary_hetero': hetero['boundary_condition'],
        'boundary_pure': pure['boundary_condition'],
        'R_hetero_m2K_W': r_hetero,
        'R_pure_m2K_W': r_pure,
        'delta_R_m2K_W': r_hetero - r_pure,
        'kappa_hetero_W_mK': k_hetero, 'kappa_pure_W_mK': k_pure,
        'eta': k_hetero / k_pure if k_pure else math.nan,
    }


def _set_block_errors(rows, block_rows, key_columns, metrics):
    by_block = [{tuple(row[column] for column in key_columns): row for row in sample}
                for sample in block_rows]
    for row in rows:
        key = tuple(row[column] for column in key_columns)
        paired = [sample.get(key, {}) for sample in by_block]
        for value_column, error_column in metrics:
            row[error_column] = _standard_error([
                sample.get(value_column, math.nan) for sample in paired])


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_table(path, columns, rows, metadata):
    with path.open('x', encoding='utf-8') as stream:
        for key, value in metadata.items():
            stream.write(f'# {key} {value}\n')
        stream.write('# columns ' + ' '.join(columns) + '\n')
        for row in rows:
            stream.write(' '.join(str(row.get(column, math.nan)) for column in columns) + '\n')


def analyze(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    metadata = config['metadata']
    for key in ('source_revision', 'nep_model', 'target_temperature_K',
                'source_group', 'sink_group', 'boundary_condition'):
        if key not in metadata:
            raise ValueError(f'metadata.{key} is required')
    base = config_path.parent
    out_dir = _resolve(base, config.get('output_dir', '.'))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [out_dir / name for name in (
        'interface_resistance.out', 'interface_kappa_profile.out',
        'interface_comparison.out', 'interface_analysis.svg',
        'interface_segment_resistance.out')]
    if any(path.exists() for path in output_paths):
        raise FileExistsError('analysis outputs already exist; choose a new output_dir')

    steady = config['steady_state_ps']
    block_frames = int(config['block_frames'])
    if block_frames < 2:
        raise ValueError('block_frames must be >= 2 for block uncertainty')
    window_bins = int(config['window_bins'])
    if window_bins < 2:
        raise ValueError('window_bins must be >= 2')
    if config['source_side'] not in ('lower', 'upper'):
        raise ValueError('source_side must be lower or upper')
    temperature_tolerance = config.get('comparison_temperature_tolerance_K')
    if temperature_tolerance is not None:
        temperature_tolerance = float(temperature_tolerance)
        if not math.isfinite(temperature_tolerance) or temperature_tolerance < 0.0:
            raise ValueError('comparison_temperature_tolerance_K must be finite and nonnegative')
    interface_z = float(config['interface_coordinate_A'])
    expected_ids = (int(config['interface']['left']['id']),
                    int(config['interface']['right']['id']))
    runs = {}
    for name, specification in config['runs'].items():
        profile = read_profile(_resolve(base, specification['profile']))
        profile_metadata = profile['metadata']
        try:
            profile_area = float(profile_metadata['cross_section_A2'])
            profile_interface = float(profile_metadata['interface_coordinate_A'])
            profile_ids = (int(profile_metadata['material_lower_id']),
                           int(profile_metadata['material_upper_id']))
        except (KeyError, ValueError) as error:
            raise ValueError(f'{name}: profile metadata is missing interface geometry') from error
        if not math.isclose(profile_interface, interface_z, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f'{name}: interface coordinate differs from analysis config')
        if profile_ids != expected_ids:
            raise ValueError(f'{name}: material IDs differ from analysis config')
        if (metadata.get('heat_direction') and
                profile_metadata.get('axis') != metadata['heat_direction']):
            raise ValueError(f'{name}: profile axis differs from metadata.heat_direction')
        frames = profile['frames']
        blocks = _complete_blocks(frames, steady, block_frames)
        selected = _window_frames(frames, steady)
        power = _bath_power_for_frames(selected)
        block_power = [_bath_power_for_frames(block)['power_eV_ps'] for block in blocks]
        power['power_error_eV_ps'] = _standard_error(block_power)
        power['block_power_eV_ps'] = block_power
        power['frames'] = len(selected)
        area = float(specification.get('area_A2', profile_area))
        fraction = float(specification['power_fraction'])
        if (not math.isfinite(profile_area) or profile_area <= 0.0 or
                not math.isclose(area, profile_area, rel_tol=1e-8, abs_tol=1e-8)):
            raise ValueError(f'{name}: area_A2 must match profile cross_section_A2')
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(f'{name}: power_fraction must be greater than 0 and at most 1')
        flux = power['power_eV_ps'] * fraction / area * FLUX_W_M2_PER_EV_PS_A2
        flux_error = power['power_error_eV_ps'] * fraction / area * FLUX_W_M2_PER_EV_PS_A2
        if not math.isfinite(flux) or flux == 0.0:
            raise ValueError(f'{name}: fitted net heat flux is zero or non-finite')
        rows = _profile_statistics(frames, steady, block_frames, blocks)
        runs[name] = {
            'metadata': profile['metadata'], 'frames': frames, 'rows': rows,
            'power': power, 'area_A2': area, 'power_fraction': fraction,
            'flux_W_m2': flux, 'flux_error_W_m2': flux_error,
            'mean_temperature_K': _mean_temperature(rows),
            'blocks': blocks,
            'bath_regions_A': _intervals(specification.get('bath_regions_A'), name),
            'used_time_ps': (selected[0]['block_start_ps'], selected[-1]['time_ps']),
            'frames_outside_complete_blocks': len(selected) -
                                              sum(len(block) for block in blocks),
            'axis': profile_metadata.get('axis', 'z'),
            'boundary_condition': str(specification.get(
                'boundary_condition', metadata['boundary_condition'])).replace(' ', '_'),
            'source_group': specification.get('source_group', metadata['source_group']),
            'sink_group': specification.get('sink_group', metadata['sink_group']),
        }
        runs[name]['block_runs'] = []
        for block in blocks:
            block_power_summary = _bath_power_for_frames(block)
            block_flux = (block_power_summary['power_eV_ps'] * fraction / area *
                          FLUX_W_M2_PER_EV_PS_A2)
            if not math.isfinite(block_flux) or block_flux == 0.0:
                raise ValueError(f'{name}: a complete time block has zero or invalid heat flux')
            block_run = {key: value for key, value in runs[name].items()
                         if key not in ('blocks', 'block_runs')}
            block_run.update({
                'rows': _profile_means(block), 'power': block_power_summary,
                'flux_W_m2': block_flux, 'flux_error_W_m2': math.nan,
                'mean_temperature_K': math.nan,
            })
            block_run['mean_temperature_K'] = _mean_temperature(block_run['rows'])
            runs[name]['block_runs'].append(block_run)

    hetero_name = config['heterostructure_run']
    hetero = runs[hetero_name]
    if any(run['axis'] != hetero['axis'] for run in runs.values()):
        raise ValueError('all compared profiles must use the same axis')
    interface = config['interface']
    max_distance = float(config['max_distance_A'])
    exclusion = float(config.get('interface_exclusion_A', 0.0))
    for side in ('left', 'right'):
        _reject_bath_overlap(hetero_name, interface[side]['fit_A'],
                             hetero['bath_regions_A'], f'{side} fit_A')
    hetero_metrics = _interface_metrics(
        hetero, interface, interface_z, config['source_side'], window_bins, exclusion,
        max_distance)
    sides = hetero_metrics['sides']
    cumulative = hetero_metrics['cumulative']
    local_output = hetero_metrics['local']
    block_hetero_metrics = [
        _interface_metrics(block_run, interface, interface_z, config['source_side'],
                           window_bins, exclusion, max_distance)
        for block_run in hetero['block_runs']]
    for side in ('left', 'right'):
        for error_column, value_column in (
                ('slope_error_K_A', 'slope_K_A'), ('intercept_error_K', 'intercept_K')):
            sides[side]['fit'][error_column] = _standard_error(
                [block['sides'][side]['fit'][value_column] for block in block_hetero_metrics])
        _set_block_errors(
            cumulative[side], [block['cumulative'][side] for block in block_hetero_metrics],
            ('coordinate_A',), (
                ('temperature_K', 'temperature_error_K'),
                ('delta_temperature_K', 'delta_temperature_error_K'),
                ('resistance_m2K_W', 'resistance_error_m2K_W'),
                ('kappa_eff_W_mK', 'kappa_eff_error_W_mK')))
    _set_block_errors(
        local_output, [block['local'] for block in block_hetero_metrics],
        ('side', 'coordinate_A'), (
            ('gradient_K_A', 'gradient_error_K_A'),
            ('kappa_local_W_mK', 'kappa_local_error_W_mK')))

    interface_values = hetero_metrics['interface']
    interface_delta_error = _standard_error(
        [block['interface']['delta_T_interface_K'] for block in block_hetero_metrics])
    interface_resistance_error = _standard_error(
        [block['interface']['R_interface_m2K_W'] for block in block_hetero_metrics])
    resistance_rows = [{
        'run': hetero_name, 'mean_temperature_K': hetero['mean_temperature_K'],
        'area_A2': hetero['area_A2'], 'power_fraction': hetero['power_fraction'],
        'power_eV_ps': hetero['power']['power_eV_ps'],
        'power_error_eV_ps': hetero['power']['power_error_eV_ps'],
        'hot_power_eV_ps': hetero['power']['hot_eV_ps'],
        'cold_power_eV_ps': hetero['power']['cold_eV_ps'],
        'energy_balance_fraction': hetero['power']['balance_fraction'],
        'flux_W_m2': hetero['flux_W_m2'],
        'T_lower_interface_K': interface_values['T_lower_interface_K'],
        'T_lower_error_K': sides['left']['fit']['intercept_error_K'],
        'T_upper_interface_K': interface_values['T_upper_interface_K'],
        'T_upper_error_K': sides['right']['fit']['intercept_error_K'],
        'delta_T_interface_K': interface_values['delta_T_interface_K'],
        'delta_T_interface_error_K': interface_delta_error,
        'R_interface_m2K_W': interface_values['R_interface_m2K_W'],
        'R_interface_error_m2K_W': interface_resistance_error,
    }]

    segment_rows = []
    for segment in config.get('segments', []):
        run = runs[segment['run']]
        region = (min(float(segment['coordinate1_A']), float(segment['coordinate2_A'])),
                  max(float(segment['coordinate1_A']), float(segment['coordinate2_A'])))
        _reject_bath_overlap(segment['run'], region, run['bath_regions_A'], 'segment')
        segment_rows.append(_segment_result(run, segment))
    for run_name in dict.fromkeys(segment['run'] for segment in config.get('segments', [])):
        run_indices = [i for i, segment in enumerate(config['segments'])
                       if segment['run'] == run_name]
        run_segments = [config['segments'][i] for i in run_indices]
        block_segment_rows = [[_segment_result(block_run, segment)
                               for segment in run_segments]
                              for block_run in runs[run_name]['block_runs']]
        _set_block_errors(
            [segment_rows[i] for i in run_indices], block_segment_rows,
            ('run', 'material_id', 'coordinate1_A', 'coordinate2_A'), (
                ('T1_K', 'T1_error_K'), ('T2_K', 'T2_error_K'),
                ('delta_T_K', 'delta_T_error_K'), ('R_seg_m2K_W', 'R_seg_error_m2K_W'),
                ('kappa_seg_W_mK', 'kappa_seg_error_W_mK')))
    for row, segment in zip(segment_rows, config.get('segments', [])):
        run = runs[segment['run']]
        row['flux_error_W_m2'] = run['flux_error_W_m2']
        row['mean_temperature_K'] = run['mean_temperature_K']

    comparison_rows = []
    pure_curves = {}
    for side in ('left', 'right'):
        reference = config['pure_references'][side]
        pure_name = reference['run']
        pure = runs[pure_name]
        _reject_bath_overlap(pure_name, reference['fit_A'], pure['bath_regions_A'], 'fit_A')
        pure_profile = _pure_reference_profile(pure, reference)
        pure_block_profiles = [_pure_reference_profile(block_run, reference)
                               for block_run in pure['block_runs']]
        hetero_region = interface[side]['fit_A']
        material_id = int(interface[side]['id'])
        hetero_region_rows = _outside_baths(
            _rows_in_region(hetero['rows'], material_id, hetero_region),
            hetero['bath_regions_A'])
        hetero_compare = {**hetero, 'mean_temperature_K':
                          _mean_temperature(hetero_region_rows)}
        pure_curve = []
        for point in cumulative[side]:
            comparison = _comparison_row(
                side, interface[side]['material']['name'], point['distance_A'], point,
                hetero_compare, pure, pure_profile, reference, hetero_region,
                temperature_tolerance)
            if comparison is None:
                continue
            pure_block_points = [
                _pure_point(block_run, reference, block_profile, side, point['distance_A'])
                for block_run, block_profile in zip(pure['block_runs'], pure_block_profiles)]
            hetero_block_means = [
                _mean_temperature(_outside_baths(
                    _rows_in_region(block_run['rows'], material_id, hetero_region),
                    block_run['bath_regions_A']))
                for block_run in hetero['block_runs']]
            pure_block_means = [profile['mean_temperature_K']
                                for profile in pure_block_profiles]
            comparison.update({
                'T_mean_hetero_error_K': _standard_error(hetero_block_means),
                'T_mean_pure_error_K': _standard_error(pure_block_means),
                'R_hetero_error_m2K_W': point['resistance_error_m2K_W'],
                'R_pure_error_m2K_W': _standard_error([
                    sample['resistance_m2K_W'] if sample else math.nan
                    for sample in pure_block_points]),
                'kappa_hetero_error_W_mK': point['kappa_eff_error_W_mK'],
                'kappa_pure_error_W_mK': _standard_error([
                    sample['kappa_eff_W_mK'] if sample else math.nan
                    for sample in pure_block_points]),
            })
            comparison['delta_R_error_m2K_W'] = (
                math.hypot(comparison['R_hetero_error_m2K_W'],
                           comparison['R_pure_error_m2K_W'])
                if math.isfinite(comparison['R_hetero_error_m2K_W']) and
                math.isfinite(comparison['R_pure_error_m2K_W']) else math.nan)
            comparison['eta_error'] = (
                abs(comparison['eta']) * math.hypot(
                    comparison['kappa_hetero_error_W_mK'] /
                    comparison['kappa_hetero_W_mK'],
                    comparison['kappa_pure_error_W_mK'] /
                    comparison['kappa_pure_W_mK'])
                if comparison['eta'] and comparison['kappa_hetero_W_mK'] and
                comparison['kappa_pure_W_mK'] and
                math.isfinite(comparison['kappa_hetero_error_W_mK']) and
                math.isfinite(comparison['kappa_pure_error_W_mK']) else math.nan)
            comparison_rows.append(comparison)
            pure_curve.append({
                'distance_A': point['distance_A'],
                'resistance_m2K_W': comparison['R_pure_m2K_W'],
                'resistance_error_m2K_W': comparison['R_pure_error_m2K_W'],
                'kappa_eff_W_mK': comparison['kappa_pure_W_mK'],
                'kappa_eff_error_W_mK': comparison['kappa_pure_error_W_mK'],
            })
        pure_curves[side] = pure_curve

    output_metadata = {
        'source_revision': metadata['source_revision'],
        'nep_model': metadata['nep_model'],
        'target_temperature_K': metadata['target_temperature_K'],
        'steady_state_ps': f'{steady[0]} {steady[1]}',
        'complete_blocks_used': len(hetero['blocks']),
        'frames_used_for_statistics': hetero['power']['frames'],
        'frames_outside_complete_error_blocks': hetero['frames_outside_complete_blocks'],
        'actual_statistics_interval_ps': ' '.join(map(str, hetero['used_time_ps'])),
        'profile_frame_filter': 'block_start_ps >= window_start and time_ps <= window_end',
        'profile_mean_weighting': 'samples; temperature uses temperature_samples in version 2',
        'uncertainty_method': 'paired complete time blocks; independent runs combine with hypot',
        'comparison_temperature_tolerance_K': ('unset' if temperature_tolerance is None
                                               else temperature_tolerance),
        'heat_direction': metadata.get('heat_direction', 'unspecified'),
        'source_group': metadata['source_group'],
        'sink_group': metadata['sink_group'],
        'boundary_condition': metadata['boundary_condition'],
        'axis': hetero['axis'],
        'sample_timestep_fs': hetero['metadata'].get('timestep_fs', 'unknown'),
        'sample_interval_steps': hetero['metadata'].get('sample_interval', 'unknown'),
        'output_interval_samples': hetero['metadata'].get(
            'output_interval_samples', 'unknown'),
        'block_frames_for_errors': block_frames,
        'interface_coordinate_A': interface_z,
        'left_fit_A': ' '.join(map(str, interface['left']['fit_A'])),
        'right_fit_A': ' '.join(map(str, interface['right']['fit_A'])),
        'bath_energy_sign': 'system_to_bath; source slope is negated for injected power',
        'interface_resistance_formula': 'abs(T_left_interface-T_right_interface)/abs(J)',
        'local_kappa_formula': '-J/(dT/dcoordinate)',
        'flux_conversion': '1 eV/(ps A2) = 1.602176634e13 W/m2',
        'conductivity_conversion': '1 eV/(ps A K) = 1602.176634 W/(m K)',
    }
    for run_name, run in runs.items():
        output_metadata[f'bath_regions_A_{run_name}'] = _format_intervals(run['bath_regions_A'])
    _write_table(output_paths[0], (
        'run mean_temperature_K area_A2 power_fraction power_eV_ps power_error_eV_ps '
        'hot_power_eV_ps cold_power_eV_ps energy_balance_fraction flux_W_m2 '
        'T_lower_interface_K T_lower_error_K T_upper_interface_K T_upper_error_K '
        'delta_T_interface_K delta_T_interface_error_K R_interface_m2K_W '
        'R_interface_error_m2K_W').split(),
        resistance_rows, output_metadata)

    kappa_rows = []
    for side in ('left', 'right'):
        material = sides[side]['material']
        local_by_coordinate = {
            row['coordinate_A']: row for row in local_output if row['side'] == side}
        for point in cumulative[side]:
            local = min(local_by_coordinate.values(),
                        key=lambda row: abs(row['coordinate_A'] - point['coordinate_A']),
                        default={})
            kappa_rows.append({**point, **{
                'gradient_K_A': local.get('gradient_K_A', math.nan),
                'gradient_error_K_A': local.get('gradient_error_K_A', math.nan),
                'kappa_local_W_mK': local.get('kappa_local_W_mK', math.nan),
                'kappa_local_error_W_mK': local.get('kappa_local_error_W_mK', math.nan),
                'fit_r_squared': local.get('r_squared', math.nan),
                'fit_window_bins': local.get('window_bins', window_bins),
            }})
    _write_table(output_paths[1], (
        'side material distance_A coordinate_A temperature_K temperature_error_K '
        'delta_temperature_K delta_temperature_error_K resistance_m2K_W '
        'resistance_error_m2K_W kappa_eff_W_mK kappa_eff_error_W_mK gradient_K_A '
        'gradient_error_K_A kappa_local_W_mK kappa_local_error_W_mK fit_r_squared '
        'fit_window_bins').split(), kappa_rows, output_metadata)
    _write_table(output_paths[2], (
        'side material distance_A comparison_hetero_min_A comparison_hetero_max_A '
        'comparison_pure_min_A comparison_pure_max_A bath_regions_hetero_A '
        'bath_regions_pure_A temperature_match T_mean_hetero_K T_mean_hetero_error_K '
        'T_mean_pure_K T_mean_pure_error_K '
        'power_hetero_eV_ps power_pure_eV_ps area_hetero_A2 area_pure_A2 '
        'power_fraction_hetero power_fraction_pure boundary_hetero boundary_pure '
        'R_hetero_m2K_W R_hetero_error_m2K_W R_pure_m2K_W R_pure_error_m2K_W '
        'delta_R_m2K_W delta_R_error_m2K_W kappa_hetero_W_mK '
        'kappa_hetero_error_W_mK kappa_pure_W_mK kappa_pure_error_W_mK eta eta_error').split(),
        comparison_rows, output_metadata)
    _write_table(output_paths[4], (
        'run side material_id coordinate1_A coordinate2_A T1_K T1_error_K '
        'T2_K T2_error_K delta_T_K '
        'delta_T_error_K flux_W_m2 flux_error_W_m2 R_seg_m2K_W R_seg_error_m2K_W '
        'kappa_seg_W_mK kappa_seg_error_W_mK mean_temperature_K boundary_condition').split(),
        segment_rows, {**output_metadata,
                       'uncertainty': 'paired complete time-block estimates'})

    _plot_results(output_paths[3], hetero, sides, cumulative, pure_curves,
                  comparison_rows, local_output, interface_z)
    return output_paths


def _plot_results(path, hetero, sides, cumulative, pure_curves, comparison, local, interface_z):
    colors = {'left': '#2369a8', 'right': '#c04b38'}
    panels = [
        (f'T({hetero["axis"]})', f'Coordinate {hetero["axis"]} (A)', 'Temperature (K)', []),
        ('Local gradient', 'Distance from interface (A)',
         f'dT/d{hetero["axis"]} (K/A)', []),
        ('Cumulative resistance', 'Distance from interface (A)', "R'' (m2 K/W)", []),
        ('Excess resistance', 'Distance from interface (A)', "Delta R'' (m2 K/W)", []),
        ('Effective conductivity', 'Distance from interface (A)', 'kappa (W/m K)', []),
        ('Hetero / pure', 'Distance from interface (A)', 'eta', []),
    ]
    for side in ('left', 'right'):
        rows = sides[side]['rows']
        rows = [row for row in rows if row['material_id'] == sides[side]['material']['id']]
        rows.sort(key=lambda row: row['coordinate_center_A'])
        panels[0][3].append(_series(side, colors[side], rows,
                                    'coordinate_center_A', 'temperature_K',
                                    'temperature_error_K'))
        lr = [r for r in local if r['side'] == side]
        panels[1][3].append(_series(side, colors[side], lr, 'distance_A',
                                    'gradient_K_A', 'gradient_error_K_A'))
        cr = cumulative[side]
        panels[2][3].append(_series(f'{side} hetero', colors[side], cr,
                                    'distance_A', 'resistance_m2K_W',
                                    'resistance_error_m2K_W'))
        pc = pure_curves.get(side, [])
        if pc:
            panels[2][3].append(_series(f'{side} pure', colors[side], pc,
                                        'distance_A', 'resistance_m2K_W',
                                        'resistance_error_m2K_W', dash=True))
        panels[4][3].append(_series(f'{side} hetero', colors[side], cr,
                                    'distance_A', 'kappa_eff_W_mK',
                                    'kappa_eff_error_W_mK'))
        if pc:
            panels[4][3].append(_series(f'{side} pure', colors[side], pc,
                                        'distance_A', 'kappa_eff_W_mK',
                                        'kappa_eff_error_W_mK', dash=True))
        comp = [r for r in comparison if r['side'] == side]
        panels[3][3].append(_series(side, colors[side], comp, 'distance_A',
                                    'delta_R_m2K_W', 'delta_R_error_m2K_W'))
        panels[5][3].append(_series(side, colors[side], comp, 'distance_A', 'eta', 'eta_error'))
    _write_svg(path, panels, interface_z)


def _series(label, color, rows, x_key, y_key, error_key=None, dash=False):
    return {
        'label': label, 'color': color, 'dash': dash,
        'points': [(row[x_key], row[y_key], row.get(error_key, math.nan)
                    if error_key else math.nan) for row in rows],
    }


def _write_svg(path, panels, interface_z):
    width, height = 1200, 760
    parts = [f'<svg id="interface-analysis" xmlns="http://www.w3.org/2000/svg" '
             f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>']
    for index, (title, xlabel, ylabel, series) in enumerate(panels):
        col, row = index % 3, index // 3
        x0, y0 = 22 + col * 395, 18 + row * 370
        px, py, pw, ph = x0 + 62, y0 + 40, 305, 230
        points = [(x, y, e) for item in series for x, y, e in item['points']
                  if math.isfinite(x) and math.isfinite(y)]
        if not points:
            continue
        xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
        y_values = [y for _, y, _ in points]
        ymin = min([y - e for _, y, e in points if math.isfinite(e) and e >= 0.0] + y_values)
        ymax = max([y + e for _, y, e in points if math.isfinite(e) and e >= 0.0] + y_values)
        dx, dy = xmax - xmin, ymax - ymin
        dx = dx if dx > 0 else 1.0
        dy = dy if dy > 0 else max(abs(ymin), 1.0)
        xmin -= 0.04 * dx
        xmax += 0.04 * dx
        ymin -= 0.08 * dy
        ymax += 0.08 * dy
        sx = lambda value: px + (value - xmin) * pw / (xmax - xmin)
        sy = lambda value: py + ph - (value - ymin) * ph / (ymax - ymin)
        parts.append(f'<text x="{x0 + 190}" y="{y0 + 19}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="15">{html.escape(title)}</text>')
        for tick in range(5):
            fraction = tick / 4
            y = py + ph * fraction
            value = ymax - (ymax - ymin) * fraction
            parts.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" '
                         f'y2="{y:.1f}" stroke="#dddddd"/>')
            parts.append(f'<text x="{px - 6}" y="{y + 4:.1f}" text-anchor="end" '
                         f'font-family="sans-serif" font-size="10">{value:.3g}</text>')
            x = px + pw * fraction
            value = xmin + (xmax - xmin) * fraction
            parts.append(f'<line x1="{x:.1f}" y1="{py}" x2="{x:.1f}" '
                         f'y2="{py + ph}" stroke="#eeeeee"/>')
            parts.append(f'<text x="{x:.1f}" y="{py + ph + 13}" text-anchor="middle" '
                         f'font-family="sans-serif" font-size="10">{value:.3g}</text>')
        parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" '
                     'fill="none" stroke="#444"/>')
        if index == 0 and xmin < interface_z < xmax:
            ix = sx(interface_z)
            parts.append(f'<line x1="{ix:.1f}" y1="{py}" x2="{ix:.1f}" '
                         f'y2="{py + ph}" stroke="#222" stroke-dasharray="4 3"/>')
        for item in series:
            ordered = sorted((p for p in item['points'] if math.isfinite(p[0]) and
                              math.isfinite(p[1])), key=lambda p: p[0])
            color = item['color']
            dash = ' stroke-dasharray="5 3"' if item['dash'] else ''
            coords = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y, _ in ordered)
            if coords:
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                             f'stroke-width="1.4"{dash}/>')
            for x, y, error in ordered:
                cx, cy = sx(x), sy(y)
                if math.isfinite(error) and error >= 0.0:
                    low, high = sy(y - error), sy(y + error)
                    parts.append(f'<line x1="{cx:.1f}" y1="{low:.1f}" '
                                 f'x2="{cx:.1f}" y2="{high:.1f}" stroke="{color}"/>')
                    parts.append(f'<line x1="{cx - 3:.1f}" y1="{low:.1f}" '
                                 f'x2="{cx + 3:.1f}" y2="{low:.1f}" stroke="{color}"/>')
                    parts.append(f'<line x1="{cx - 3:.1f}" y1="{high:.1f}" '
                                 f'x2="{cx + 3:.1f}" y2="{high:.1f}" stroke="{color}"/>')
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="{color}"/>')
        parts.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 27}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11">{html.escape(xlabel)}</text>')
        parts.append(f'<text x="{x0 + 12}" y="{py + ph / 2:.1f}" text-anchor="middle" '
                     f'transform="rotate(-90 {x0 + 12} {py + ph / 2:.1f})" '
                     f'font-family="sans-serif" font-size="11">{html.escape(ylabel)}</text>')
        legend_x, legend_y = px, py + ph + 49
        for legend, item in enumerate(series):
            lx = legend_x + (legend % 2) * 145
            ly = legend_y + (legend // 2) * 14
            parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" '
                         f'stroke="{item["color"]}" stroke-width="2"/>')
            parts.append(f'<text x="{lx + 23}" y="{ly + 4}" font-family="sans-serif" '
                         f'font-size="10">{html.escape(item["label"])}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='JSON file describing the NEMD runs and fit regions')
    args = parser.parse_args()
    for path in analyze(args.config):
        print(path)


if __name__ == '__main__':
    main()
