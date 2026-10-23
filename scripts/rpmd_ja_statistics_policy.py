#!/usr/bin/env python3
"""Versioned CPU reference for the RPMD-JA statistical admission rules."""

from math import ceil, isfinite, log, sqrt
from statistics import mean
from types import MappingProxyType


POLICY_VERSION = "RPJASTAT2"
FROZEN = MappingProxyType({
    "autocorrelation_rule": "FIXED_TRAINING_PHYSICAL_AND_PARAMETER_CHANNEL_SOKAL_WINDOW_V2",
    "max_lag_fraction": 0.25,
    "window_c": 5.0,
    "min_samples_per_tau": 50.0,
    "block_factor": 2.0,
    "min_blocks": 4,
    "adjacent_se_relative_tolerance": 0.25,
    "mean_drift_z_tolerance": 2.0,
    "mean_drift_family_alpha": 0.05,
    "mean_drift_threshold_rule": "MAX_Z_SUBGAUSSIAN_UNION_ENGINEERING_V1",
    "block_boundary_rule": "COMMON_2B_PREFIX_FOR_B_AND_2B_V1",
    "constant_relative_tolerance": 1.0e-12,
    "selection_z_multiplier": 2.0,
})


def frozen_contract():
    return {"version": POLICY_VERSION, **FROZEN}


def _checked_channels(channels):
    if not channels:
        raise ValueError("at least one named channel is required")
    names = list(channels)
    values = [list(channels[name]) for name in names]
    count = len(values[0])
    if count < 2 or any(len(series) != count for series in values):
        raise ValueError("channel series must have equal length of at least two")
    if any(not isfinite(value) for series in values for value in series):
        raise ValueError("channel series must contain only finite values")
    return names, values, count


def _moments(values):
    center = mean(values)
    centered = [value - center for value in values]
    covariance = sum(value * value for value in centered) / (len(values) - 1)
    return center, covariance


def _channel_state(values):
    center, variance = _moments(values)
    if variance == 0.0:
        return ("CONSTANT_ZERO" if center == 0.0 else "CONSTANT_NONZERO"), center, variance
    scale = max(abs(value) for value in values)
    span = max(values) - min(values)
    if scale > 0.0 and span / scale <= FROZEN["constant_relative_tolerance"]:
        return "NEAR_CONSTANT", center, variance
    return "VARYING", center, variance


def _autocorrelation(values, lag, variance):
    center = mean(values)
    count = len(values)
    covariance = sum(
        (values[index] - center) * (values[index + lag] - center)
        for index in range(count - lag)
    ) / (count - lag)
    return covariance / variance


def _block_mean_se(values, block_length, frame_limit=None):
    usable = len(values) if frame_limit is None else frame_limit
    block_count = usable // block_length
    block_means = [
        mean(values[offset : offset + block_length])
        for offset in range(0, usable, block_length)
    ]
    center = mean(block_means)
    sample_variance = sum((value - center) ** 2 for value in block_means) / (block_count - 1)
    return sqrt(sample_variance / block_count), block_count


def plan_training_blocks(training_channels):
    """Choose a conservative block plan from training-only fixed physical channels.

    This is an engineering screen, not a universal confidence guarantee. The
    Sokal window is only estimated when it closes before maxlag and the series
    contains at least 50 estimated correlation times; emcee's documentation
    likewise treats IAT as a heuristic and rejects chains too short to estimate.
    See https://emcee.readthedocs.io/en/stable/user/autocorr/.
    """
    names, series, count = _checked_channels(training_channels)
    max_lag = int(count * FROZEN["max_lag_fraction"])
    channel_reports = []
    tau_frames = 0.0
    window_lags = []
    varying_count = 0
    if max_lag == 0:
        return {
            "status": "DATA_INCONCLUSIVE",
            "policy_version": POLICY_VERSION,
            "reason": "TOO_FEW_FRAMES_FOR_MAX_LAG",
            "frame_count": count,
            "max_lag": max_lag,
            "channels": channel_reports,
        }
    for name, values in zip(names, series):
        state, center, variance = _channel_state(values)
        report = {"name": name, "state": state, "mean": center, "variance": variance}
        channel_reports.append(report)
        if state != "VARYING":
            continue
        varying_count += 1
        half_tau = 0.5
        window = None
        for lag in range(1, max_lag + 1):
            rho = _autocorrelation(values, lag, variance)
            if not isfinite(rho):
                break
            half_tau += rho
            current_tau = 2.0 * half_tau
            if current_tau > 0.0 and lag >= FROZEN["window_c"] * current_tau:
                window = lag
                break
        if window is None or half_tau <= 0.0:
            return {
                "status": "DATA_INCONCLUSIVE",
                "policy_version": POLICY_VERSION,
                "reason": "SOKAL_WINDOW_UNREACHED",
                "frame_count": count,
                "max_lag": max_lag,
                "channels": channel_reports,
            }
        current_tau = 2.0 * half_tau
        if count < FROZEN["min_samples_per_tau"] * current_tau:
            return {
                "status": "DATA_INCONCLUSIVE",
                "policy_version": POLICY_VERSION,
                "reason": "INSUFFICIENT_SAMPLES_PER_TAU",
                "frame_count": count,
                "max_lag": max_lag,
                "tau_frames": current_tau,
                "channels": channel_reports,
            }
        tau_frames = max(tau_frames, current_tau)
        window_lags.append(window)
        report.update({"tau_frames": current_tau, "window_lag": window})
    if varying_count == 0:
        return {
            "status": "DATA_INCONCLUSIVE",
            "policy_version": POLICY_VERSION,
            "reason": "NO_VARYING_TRAINING_CHANNEL",
            "frame_count": count,
            "max_lag": max_lag,
            "channels": channel_reports,
        }

    block_length = max(1, int(ceil(FROZEN["block_factor"] * tau_frames)))
    adjacent_length = 2 * block_length
    common_prefix_frames = (count // adjacent_length) * adjacent_length
    block_count = common_prefix_frames // block_length
    adjacent_count = common_prefix_frames // adjacent_length
    result = {
        "status": "PASS",
        "policy_version": POLICY_VERSION,
        "reason": "",
        "frame_count": count,
        "max_lag": max_lag,
        "tau_frames": tau_frames,
        "window_lag_max": max(window_lags),
        "block_length": block_length,
        "adjacent_block_length": adjacent_length,
        "block_count": block_count,
        "adjacent_block_count": adjacent_count,
        "common_prefix_frames": common_prefix_frames,
        "discarded_tail_frames": count - common_prefix_frames,
        "block_boundary_rule": FROZEN["block_boundary_rule"],
        "channels": channel_reports,
    }
    if block_count < FROZEN["min_blocks"] or adjacent_count < FROZEN["min_blocks"]:
        result.update({"status": "DATA_INCONCLUSIVE", "reason": "TOO_FEW_ADJACENT_BLOCKS"})
        return result

    sensitivity = []
    for report, values in zip(channel_reports, series):
        if report["state"] != "VARYING":
            continue
        se_base, _ = _block_mean_se(values, block_length, common_prefix_frames)
        se_adjacent, _ = _block_mean_se(values, adjacent_length, common_prefix_frames)
        denominator = max(se_base, se_adjacent)
        relative_change = abs(se_adjacent - se_base) / denominator if denominator else 0.0
        report.update({
            "se_block": se_base,
            "se_adjacent_block": se_adjacent,
            "adjacent_se_relative_change": relative_change,
        })
        sensitivity.append(relative_change)
    result["max_adjacent_se_relative_change"] = max(sensitivity, default=0.0)
    if result["max_adjacent_se_relative_change"] > FROZEN["adjacent_se_relative_tolerance"]:
        result.update({"status": "DATA_INCONCLUSIVE", "reason": "ADJACENT_BLOCK_SENSITIVE"})
    return result


def _block_means(channels, block_length, frame_limit=None):
    names, series, frame_count = _checked_channels(channels)
    usable_frames = frame_count if frame_limit is None else frame_limit
    block_count = usable_frames // block_length
    return names, [
        [mean(values[offset : offset + block_length]) for offset in range(0, usable_frames, block_length)]
        for values in series
    ], block_count


def _sample_covariance_matrix(values):
    centers = [mean(channel) for channel in values]
    count = len(values[0])
    matrix = []
    for row in range(len(values)):
        row_values = []
        for column in range(len(values)):
            covariance = sum(
                (values[row][index] - centers[row])
                * (values[column][index] - centers[column])
                for index in range(count)
            ) / (count - 1)
            row_values.append(covariance)
        matrix.append(row_values)
    return centers, matrix


def _z_score(value, standard_error):
    if standard_error == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value) / standard_error


def assess_fixed_r0_means(training_channels, development_channels, block_plan,
                          expected_zero_mean_channels=None):
    """Check fixed-zero means using the frozen training B and its adjacent 2B.

    Each segment's full cross-covariance of non-overlapping block means is
    divided by that segment's block count. The segments' covariance estimates
    are added for the mean difference; no universal multivariate coverage is
    claimed, and the development data never select a new block length. A
    frozen max-studentized threshold applies a sub-Gaussian union-bound style
    dimension correction as an engineering screen only; estimated block SEs
    and PIMD dependence do not establish the assumptions needed for coverage.
    """
    train_names, train_values, train_count = _checked_channels(training_channels)
    dev_names, dev_values, dev_count = _checked_channels(development_channels)
    if train_names != dev_names:
        raise ValueError("training/development physical channel names or order do not match")
    zero_mean_channels = set(train_names if expected_zero_mean_channels is None else expected_zero_mean_channels)
    if not zero_mean_channels.issubset(set(train_names)):
        raise ValueError("expected-zero mean channels must be a subset of declared channels")
    if block_plan.get("status") != "PASS" or block_plan.get("block_length", 0) <= 0:
        return {
            "status": "DATA_INCONCLUSIVE",
            "policy_version": POLICY_VERSION,
            "reason": "TRAINING_BLOCK_PLAN_UNRESOLVED",
            "dimension_count": len(train_names),
            "scope": "DECLARED_PHYSICAL_CHANNELS_ONLY",
            "channels": [
                {"name": name, "state": _channel_state(training_channels[name])[0]}
                for name in train_names
            ],
        }
    block_length = block_plan["block_length"]
    adjacent_length = block_plan["adjacent_block_length"]
    train_common_prefix = (train_count // adjacent_length) * adjacent_length
    dev_common_prefix = (dev_count // adjacent_length) * adjacent_length
    full_train_means = [mean(values) for values in train_values]
    full_dev_means = [mean(values) for values in dev_values]
    scales = []
    for current_length in (block_length, adjacent_length):
        _, train_blocks, train_block_count = _block_means(training_channels, current_length, train_common_prefix)
        _, dev_blocks, dev_block_count = _block_means(development_channels, current_length, dev_common_prefix)
        if train_block_count < FROZEN["min_blocks"] or dev_block_count < FROZEN["min_blocks"]:
            return {
                "status": "DATA_INCONCLUSIVE",
                "policy_version": POLICY_VERSION,
                "reason": "TOO_FEW_TRAINING_OR_DEVELOPMENT_BLOCKS",
                "dimension_count": len(train_names),
                "scope": "DECLARED_PHYSICAL_CHANNELS_ONLY",
                "block_length": current_length,
                "block_count_train": train_block_count,
                "block_count_development": dev_block_count,
            }
        train_means, train_block_cov = _sample_covariance_matrix(train_blocks)
        dev_means, dev_block_cov = _sample_covariance_matrix(dev_blocks)
        cov_train_mean = [[value / train_block_count for value in row] for row in train_block_cov]
        cov_dev_mean = [[value / dev_block_count for value in row] for row in dev_block_cov]
        cov_delta = [
            [cov_train_mean[i][j] + cov_dev_mean[i][j] for j in range(len(train_names))]
            for i in range(len(train_names))
        ]
        channels = []
        max_z = 0.0
        for index, name in enumerate(train_names):
            train_se = sqrt(cov_train_mean[index][index])
            dev_se = sqrt(cov_dev_mean[index][index])
            delta = full_dev_means[index] - full_train_means[index]
            delta_se = sqrt(cov_delta[index][index])
            train_z = _z_score(full_train_means[index], train_se) if name in zero_mean_channels else 0.0
            dev_z = _z_score(full_dev_means[index], dev_se) if name in zero_mean_channels else 0.0
            delta_z = _z_score(delta, delta_se)
            max_z = max(max_z, train_z, dev_z, delta_z)
            state_train = _channel_state(training_channels[name])[0]
            state_dev = _channel_state(development_channels[name])[0]
            constant_states = ("CONSTANT_ZERO", "CONSTANT_NONZERO")
            state = "CONSTANT_ZERO" if state_train == state_dev == "CONSTANT_ZERO" else \
                "CONSTANT_NONZERO" if state_train == state_dev == "CONSTANT_NONZERO" else \
                "CONSTANT_CHANGED" if state_train in constant_states and state_dev in constant_states else \
                "NEAR_CONSTANT" if "NEAR_CONSTANT" in (state_train, state_dev) else "VARYING"
            channels.append({
                "name": name,
                "state": state,
                "mean_train": full_train_means[index],
                "mean_development": full_dev_means[index],
                "mean_difference": delta,
                "common_prefix_mean_train": train_means[index],
                "common_prefix_mean_development": dev_means[index],
                "se_train": train_se,
                "se_development": dev_se,
                "se_difference": delta_se,
                "train_z": train_z,
                "development_z": dev_z,
                "delta_z": delta_z,
            })
        scales.append({
            "block_length": current_length,
            "block_count_train": train_block_count,
            "block_count_development": dev_block_count,
            "mean_train": full_train_means,
            "mean_development": full_dev_means,
            "mean_difference": [full_dev_means[i] - full_train_means[i] for i in range(len(train_names))],
            "common_prefix_frames_train": train_common_prefix,
            "common_prefix_frames_development": dev_common_prefix,
            "discarded_tail_frames_train": train_count - train_common_prefix,
            "discarded_tail_frames_development": dev_count - dev_common_prefix,
            "covariance_of_mean_train": cov_train_mean,
            "covariance_of_mean_development": cov_dev_mean,
            "covariance_of_mean_difference": cov_delta,
            "maximum_z": max_z,
            "channels": channels,
        })
    development_se_changes = []
    for index, name in enumerate(train_names):
        se_base = scales[0]["channels"][index]["se_development"]
        se_adjacent = scales[1]["channels"][index]["se_development"]
        denominator = max(se_base, se_adjacent)
        change = abs(se_adjacent - se_base) / denominator if denominator else 0.0
        scales[0]["channels"][index]["development_adjacent_se_relative_change"] = change
        scales[1]["channels"][index]["development_adjacent_se_relative_change"] = change
        development_se_changes.append(change)
    adjacent_change = max(development_se_changes, default=0.0)
    maximum_z = max(scale["maximum_z"] for scale in scales)
    comparison_count = 2 * (len(train_names) + 2 * len(zero_mean_channels))
    threshold = max(FROZEN["mean_drift_z_tolerance"], sqrt(
        2.0 * log(2.0 * comparison_count / FROZEN["mean_drift_family_alpha"])
    ))
    reason = ""
    if adjacent_change > FROZEN["adjacent_se_relative_tolerance"]:
        reason = "DEVELOPMENT_BLOCK_SENSITIVE"
    elif maximum_z > threshold:
        reason = "FIXED_R0_MEAN_OR_DRIFT"
    block = scales[0]
    return {
        "status": "PASS" if not reason else "DATA_INCONCLUSIVE",
        "policy_version": POLICY_VERSION,
        "reason": reason,
        "dimension_count": len(train_names),
        "scope": "DECLARED_PHYSICAL_CHANNELS_ONLY",
        "tau_frames": block_plan["tau_frames"],
        "block_length": block_length,
        "adjacent_block_length": adjacent_length,
        "block_boundary_rule": FROZEN["block_boundary_rule"],
        "common_prefix_frames_train": train_common_prefix,
        "common_prefix_frames_development": dev_common_prefix,
        "discarded_tail_frames_train": train_count - train_common_prefix,
        "discarded_tail_frames_development": dev_count - dev_common_prefix,
        "development_adjacent_se_relative_change": adjacent_change,
        "mean_train": block["mean_train"],
        "mean_development": block["mean_development"],
        "mean_difference": block["mean_difference"],
        "covariance_of_mean_train": block["covariance_of_mean_train"],
        "covariance_of_mean_development": block["covariance_of_mean_development"],
        "covariance_of_mean_difference": block["covariance_of_mean_difference"],
        "z_tolerance": threshold,
        "comparison_count": comparison_count,
        "family_alpha": FROZEN["mean_drift_family_alpha"],
        "threshold_method": FROZEN["mean_drift_threshold_rule"],
        "maximum_z": maximum_z,
        "channels": block["channels"],
        "adjacent_block": scales[1],
    }


def _paired_resolution(differences):
    if len(differences) < FROZEN["min_blocks"] or any(not isfinite(value) for value in differences):
        return {"resolved": False, "winner": "", "mean_difference": float("nan"), "se": float("nan")}
    center = mean(differences)
    variance = sum((value - center) ** 2 for value in differences) / (len(differences) - 1)
    se = sqrt(variance / len(differences))
    resolved = abs(center) > FROZEN["selection_z_multiplier"] * se
    winner = "A" if center < 0.0 else "B" if center > 0.0 else ""
    return {"resolved": resolved, "winner": winner if resolved else "", "mean_difference": center, "se": se}


def assess_paired_choice(differences_block, differences_adjacent):
    """Require the existing paired 2SE choice to agree at B and adjacent 2B."""
    base = _paired_resolution(differences_block)
    adjacent = _paired_resolution(differences_adjacent)
    same_winner = base["resolved"] and adjacent["resolved"] and base["winner"] == adjacent["winner"]
    return {
        "status": base["winner"] + "_WINS" if same_winner else "DATA_INCONCLUSIVE",
        "policy_version": POLICY_VERSION,
        "rule": "PAIRED_TWO_SE_SAME_WINNER_AT_B_AND_2B_V1",
        "block": base,
        "adjacent_block": adjacent,
    }
