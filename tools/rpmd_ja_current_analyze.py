"""Analyze paired Jcent, DeltaJ, and JA correlations from one native current segment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from rpmd_ja_material_export import _read_current
from rpmd_ja_material_evaluate import _correlate

KB_EV_K = 8.617343e-5
NATIVE_TIME_PS = 0.01018051
KAPPA_UNIT_CONVERSION = 1.573769e5
BOOTSTRAP_SAMPLES = 4096
MIN_BLOCKS_PER_SCALE = 8
MAX_BOOTSTRAP_BLOCKS = 4096
MAX_CORRELATION_WORK = 20_000_000
MAX_CURRENT_ANALYSIS_MEMORY_BYTES = 1 << 30
BOOTSTRAP_SEED = 0x243F6A8885A308D3


def _running_kappa(correlation, dt_native, temperature, volume):
    factor = dt_native * 0.5 / (KB_EV_K * temperature * temperature * volume) * KAPPA_UNIT_CONVERSION
    result = np.zeros_like(correlation[:, [0, 1, 2], [0, 1, 2]])
    result[1:] = np.cumsum(correlation[:-1, [0, 1, 2], [0, 1, 2]] +
                           correlation[1:, [0, 1, 2], [0, 1, 2]], axis=0) * factor
    return result


def _block_delta_kappa(jc, ja, block_frames, max_lag_frames, dt_native, temperature, volume):
    count = len(jc) // block_frames
    result = np.empty((count, 3), dtype=np.float64)
    for block in range(count):
        begin = block * block_frames
        end = begin + block_frames
        cc = _correlate(jc[begin:end], jc[begin:end], max_lag_frames)
        aa = _correlate(ja[begin:end], ja[begin:end], max_lag_frames)
        result[block] = (_running_kappa(aa, dt_native, temperature, volume)[-1] -
                         _running_kappa(cc, dt_native, temperature, volume)[-1])
    if not np.isfinite(result).all():
        raise ValueError("contiguous-block HAC integrals overflowed or became non-finite")
    return result


def _bootstrap_means(values, samples, rng):
    count = len(values)
    result = np.empty((samples, values.shape[1]), dtype=np.float64)
    for sample in range(samples):
        result[sample] = np.mean(values[rng.integers(0, count, size=count)], axis=0)
    return result


def _ci_status(ci, epsilon):
    inside = (ci[0] >= -epsilon) & (ci[1] <= epsilon)
    outside = (ci[0] > epsilon) | (ci[1] < -epsilon)
    return inside, outside


def _json_vector(values):
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def analyze_current(path, epsilon_kappa, block_frames, max_lag_frames, allow_diagnostic=False):
    if not math.isfinite(epsilon_kappa) or epsilon_kappa <= 0:
        raise ValueError("epsilon_kappa must be positive, finite, and frozen before analysis")
    if block_frames < 1 or max_lag_frames < 0:
        raise ValueError("block_frames must be positive and max_lag_frames cannot be negative")

    jc, delta, ja, reference_sha256, identity = _read_current(
        path, None, max_correlation_work=MAX_CORRELATION_WORK, max_lag_frames=max_lag_frames,
        max_memory_bytes=MAX_CURRENT_ANALYSIS_MEMORY_BYTES)
    if not reference_sha256 or len(reference_sha256) != 64:
        raise ValueError("current segment has no bound reference package SHA-256")
    if identity["reference_use_class"] == "DIAGNOSTIC" and not allow_diagnostic:
        raise ValueError("diagnostic reference requires explicit --diagnostic analysis selection")
    if identity["reference_use_class"] == "FORMAL" and identity["reference_response_status"] not in ("PASS", "NOT_APPLICABLE"):
        raise ValueError("formal reference has an incompatible response status")

    metadata = identity["metadata"]
    try:
        temperature = float(metadata["T_reference_K"])
        volume = float(metadata["cell_volume_A3"])
    except (KeyError, ValueError) as exc:
        raise ValueError("current segment must declare positive T_reference_K and cell_volume_A3") from exc
    times = identity["times_ps"]
    if len(times) < 2 or not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(volume) or volume <= 0:
        raise ValueError("current segment has insufficient frames or invalid HAC thermodynamic metadata")
    steps = np.diff(times)
    dt_ps = float(steps[0])
    if not math.isfinite(dt_ps) or dt_ps <= 0 or not np.allclose(steps, dt_ps, rtol=1e-10, atol=1e-14):
        raise ValueError("current segment time grid is not uniform")
    dt_native = dt_ps / NATIVE_TIME_PS
    frames = len(jc)
    if max_lag_frames >= frames:
        raise ValueError("max_lag_frames must be smaller than the current segment")
    if frames * (max_lag_frames + 1) > MAX_CORRELATION_WORK:
        raise ValueError("frames*max_lag_frames exceeds the bounded correlation-work limit")

    correlations = {
        "Ccc": _correlate(jc, jc, max_lag_frames),
        "CcDelta": _correlate(jc, delta, max_lag_frames),
        "CDeltac": _correlate(delta, jc, max_lag_frames),
        "CDeltaDelta": _correlate(delta, delta, max_lag_frames),
        "Caa": _correlate(ja, ja, max_lag_frames),
    }
    decomposition_residual = correlations["Caa"].copy()
    for name in ("Ccc", "CcDelta", "CDeltac", "CDeltaDelta"):
        decomposition_residual -= correlations[name]
    if any(not np.isfinite(value).all() for value in correlations.values()) or \
       not np.isfinite(decomposition_residual).all():
        raise ValueError("current correlations overflowed or became non-finite")
    decomposition_max_error = float(np.max(np.abs(decomposition_residual)))
    del decomposition_residual
    kappa = {name: _running_kappa(value, dt_native, temperature, volume)
             for name, value in correlations.items()}
    kappa["DeltaKappa"] = kappa["Caa"] - kappa["Ccc"]
    if any(not np.isfinite(value).all() for value in kappa.values()):
        raise ValueError("current HAC integrals overflowed or became non-finite")

    complete_blocks = frames // block_frames
    complete_2b_blocks = frames // (2 * block_frames)
    common_frames = complete_2b_blocks * 2 * block_frames
    omitted_frames = frames - common_frames
    minimum_block_frames = 4 * (max_lag_frames + 1)
    block_window_valid = (max_lag_frames >= 1 and block_frames >= minimum_block_frames and
                          complete_2b_blocks >= MIN_BLOCKS_PER_SCALE and
                          complete_2b_blocks <= MAX_BOOTSTRAP_BLOCKS)
    block_delta = np.empty((0, 3), dtype=np.float64)
    block_2b_delta = np.empty((0, 3), dtype=np.float64)
    bootstrap_delta = np.empty((0, 3), dtype=np.float64)
    bootstrap_b_delta = np.empty((0, 3), dtype=np.float64)
    bootstrap_scale_difference = np.empty((0, 3), dtype=np.float64)
    ci = np.full((2, 3), np.nan)
    ci_b = np.full((2, 3), np.nan)
    scale_ci = np.full((2, 3), np.nan)
    decision_ci = np.full((2, 3), np.nan)
    decision_b_ci = np.full((2, 3), np.nan)
    common_prefix_delta = np.full(3, np.nan)
    full_prefix_discrepancy = np.full(3, np.nan)
    full_b_block_mean_discrepancy = np.full(3, np.nan)
    full_block_mean_discrepancy = np.full(3, np.nan)
    if common_frames > max_lag_frames:
        prefix_corr = (_correlate(ja[:common_frames], ja[:common_frames], max_lag_frames) -
                       _correlate(jc[:common_frames], jc[:common_frames], max_lag_frames))
        common_prefix_delta = _running_kappa(prefix_corr, dt_native, temperature, volume)[-1]
        full_prefix_discrepancy = kappa["DeltaKappa"][-1] - common_prefix_delta
    direction_equivalence = ["INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE"]
    equivalence_status = "INCONCLUSIVE"
    if block_window_valid:
        block_delta = _block_delta_kappa(jc[:common_frames], ja[:common_frames], block_frames,
                                         max_lag_frames, dt_native, temperature, volume)
        block_2b_delta = _block_delta_kappa(jc[:common_frames], ja[:common_frames], 2 * block_frames,
                                            max_lag_frames, dt_native, temperature, volume)
        paired_b = 0.5 * (block_delta[0::2] + block_delta[1::2])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap_b_delta = _bootstrap_means(block_delta, BOOTSTRAP_SAMPLES, rng)
        bootstrap_delta = _bootstrap_means(block_2b_delta, BOOTSTRAP_SAMPLES, rng)
        bootstrap_scale_difference = _bootstrap_means(block_2b_delta - paired_b, BOOTSTRAP_SAMPLES, rng)
        ci_b = np.percentile(bootstrap_b_delta, [2.5, 97.5], axis=0)
        ci = np.percentile(bootstrap_delta, [2.5, 97.5], axis=0)
        scale_ci = np.percentile(bootstrap_scale_difference, [2.5, 97.5], axis=0)
        del bootstrap_b_delta, bootstrap_delta, bootstrap_scale_difference
        full_b_block_mean_discrepancy = kappa["DeltaKappa"][-1] - np.mean(block_delta, axis=0)
        full_block_mean_discrepancy = kappa["DeltaKappa"][-1] - np.mean(block_2b_delta, axis=0)
        bias_b = np.maximum(np.abs(full_prefix_discrepancy), np.abs(full_b_block_mean_discrepancy))
        bias_2b = np.maximum(np.abs(full_prefix_discrepancy), np.abs(full_block_mean_discrepancy))
        decision_b_ci[0] = ci_b[0] - bias_b
        decision_b_ci[1] = ci_b[1] + bias_b
        decision_ci[0] = ci[0] - bias_2b
        decision_ci[1] = ci[1] + bias_2b
        inside_b, outside_b = _ci_status(decision_b_ci, epsilon_kappa)
        inside_2b, outside_2b = _ci_status(decision_ci, epsilon_kappa)
        scale_inside, _ = _ci_status(scale_ci, epsilon_kappa)
        scale_ok = bool(np.all(scale_inside))
        if scale_ok:
            for axis in range(3):
                same_positive = decision_b_ci[0, axis] > epsilon_kappa and decision_ci[0, axis] > epsilon_kappa
                same_negative = decision_b_ci[1, axis] < -epsilon_kappa and decision_ci[1, axis] < -epsilon_kappa
                if inside_b[axis] and inside_2b[axis]:
                    direction_equivalence[axis] = "EQUIVALENT"
                elif outside_b[axis] and outside_2b[axis] and (same_positive or same_negative):
                    direction_equivalence[axis] = "NOT_EQUIVALENT"
            if all(value == "EQUIVALENT" for value in direction_equivalence):
                equivalence_status = "EQUIVALENT"
            elif any(value == "NOT_EQUIVALENT" for value in direction_equivalence):
                equivalence_status = "NOT_EQUIVALENT"

    if max_lag_frames == 0:
        equivalence_reason = "CORRELATION_WINDOW_HAS_NO_LAG"
    elif block_frames < minimum_block_frames:
        equivalence_reason = "BLOCK_SHORTER_THAN_FOUR_CORRELATION_WINDOWS"
    elif complete_2b_blocks < MIN_BLOCKS_PER_SCALE:
        equivalence_reason = "FEWER_THAN_EIGHT_COMPLETE_2B_BLOCKS"
    elif complete_2b_blocks > MAX_BOOTSTRAP_BLOCKS:
        equivalence_reason = "BOOTSTRAP_BLOCK_LIMIT_EXCEEDED"
    elif common_frames <= max_lag_frames:
        equivalence_reason = "COMMON_PREFIX_SHORTER_THAN_CORRELATION_WINDOW"
    elif block_window_valid and not np.all(_ci_status(scale_ci, epsilon_kappa)[0]):
        equivalence_reason = "B_2B_BLOCK_ESTIMATES_NOT_EQUIVALENT"
    elif block_window_valid and equivalence_status == "INCONCLUSIVE":
        equivalence_reason = "DECISION_INTERVAL_CROSSES_OR_SPANS_TOLERANCE"
    elif block_window_valid:
        equivalence_reason = "PAIRED_B_AND_2B_BLOCK_BOOTSTRAP_COMMON_COVERAGE"
    else:
        equivalence_reason = "INSUFFICIENT_COMMON_BLOCK_COVERAGE"
    if identity["reference_use_class"] == "DIAGNOSTIC":
        status = "DIAGNOSTIC_ONLY"
    else:
        status = equivalence_status

    arrays = {
        "lag_time_ps": np.arange(max_lag_frames + 1, dtype=np.float64) * dt_ps,
        "Ccc": correlations["Ccc"], "CcDelta": correlations["CcDelta"],
        "CDeltac": correlations["CDeltac"], "CDeltaDelta": correlations["CDeltaDelta"],
        "Caa": correlations["Caa"],
        "kappa_Ccc_W_m_K": kappa["Ccc"], "kappa_CcDelta_W_m_K": kappa["CcDelta"],
        "kappa_CDeltac_W_m_K": kappa["CDeltac"], "kappa_CDeltaDelta_W_m_K": kappa["CDeltaDelta"],
        "kappa_Caa_W_m_K": kappa["Caa"], "delta_kappa_W_m_K": kappa["DeltaKappa"],
        "block_delta_kappa_W_m_K": block_delta,
        "block_2b_delta_kappa_W_m_K": block_2b_delta,
    }
    report = {
        "format": "RPJACURRENT2_REPORT", "arrays_format": "RPJACURRENT2_NPZ", "status": status,
        "equivalence_status": equivalence_status, "equivalence_reason": equivalence_reason,
        "reference_use_class": identity["reference_use_class"],
        "reference_numerical_status": identity["reference_numerical_status"],
        "reference_response_status": identity["reference_response_status"],
        "reference_package_sha256": reference_sha256.lower(),
        "reference_fingerprint": metadata.get("reference_fingerprint"),
        "model_fingerprint": metadata.get("model_fingerprint"),
        "temperature_K": temperature, "volume_A3": volume,
        "frame_count": frames, "time_interval_ps": dt_ps,
        "max_lag_frames": max_lag_frames,
        "maximum_input_correlation_work": MAX_CORRELATION_WORK,
        "maximum_estimated_host_memory_bytes": MAX_CURRENT_ANALYSIS_MEMORY_BYTES,
        "block_frames": block_frames, "complete_blocks": complete_blocks,
        "complete_2b_blocks": complete_2b_blocks,
        "common_covered_prefix_frames": common_frames,
        "common_prefix_b_blocks": common_frames // block_frames,
        "bootstrap_tail_frames_omitted": omitted_frames,
        "bootstrap_method": "paired B and 2B resampling of precomputed contiguous block integrals over one common prefix; blocks are never concatenated",
        "minimum_block_frames": minimum_block_frames,
        "raw_bootstrap_estimands": {"B": "mean of contiguous B block DeltaKappa integrals over common_covered_prefix_frames",
                                     "2B": "mean of contiguous 2B block DeltaKappa integrals over common_covered_prefix_frames"},
        "decision_estimand": "full-series DeltaKappa; the B and 2B raw bootstrap CIs are symmetrically expanded by the largest observed offset from the common-prefix and corresponding block-mean estimates",
        "coverage_gate": "decision intervals expand each raw block bootstrap interval by the maximum observed full-series offset from the common prefix and that block mean",
        "decision_interval_scope_note": "the deterministic expansion covers observed estimand offsets for this segment; it is not an additional probability or confidence guarantee",
        "scale_stability_gate": "95% CI of paired 2B minus adjacent-pair B estimates must lie inside +/-epsilon",
        "bootstrap_samples": BOOTSTRAP_SAMPLES if block_window_valid else 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "epsilon_kappa_W_m_K": epsilon_kappa,
        "delta_kappa_ci95_W_m_K": {axis: [float(ci[0, index]) if math.isfinite(ci[0, index]) else None,
                                          float(ci[1, index]) if math.isfinite(ci[1, index]) else None]
                                   for index, axis in enumerate("xyz")},
        "delta_kappa_decision_interval_W_m_K": {axis: [float(decision_ci[0, index]) if math.isfinite(decision_ci[0, index]) else None,
                                                       float(decision_ci[1, index]) if math.isfinite(decision_ci[1, index]) else None]
                                                for index, axis in enumerate("xyz")},
        "delta_kappa_b_ci95_W_m_K": {axis: [float(ci_b[0, index]) if math.isfinite(ci_b[0, index]) else None,
                                             float(ci_b[1, index]) if math.isfinite(ci_b[1, index]) else None]
                                      for index, axis in enumerate("xyz")},
        "delta_kappa_b_decision_interval_W_m_K": {axis: [float(decision_b_ci[0, index]) if math.isfinite(decision_b_ci[0, index]) else None,
                                                          float(decision_b_ci[1, index]) if math.isfinite(decision_b_ci[1, index]) else None]
                                                   for index, axis in enumerate("xyz")},
        "b_2b_difference_ci95_W_m_K": {axis: [float(scale_ci[0, index]) if math.isfinite(scale_ci[0, index]) else None,
                                              float(scale_ci[1, index]) if math.isfinite(scale_ci[1, index]) else None]
                                       for index, axis in enumerate("xyz")},
        "full_series_minus_common_prefix_delta_kappa_W_m_K": _json_vector(full_prefix_discrepancy),
        "full_series_minus_b_block_mean_delta_kappa_W_m_K": _json_vector(full_b_block_mean_discrepancy),
        "full_series_minus_2b_block_mean_delta_kappa_W_m_K": _json_vector(full_block_mean_discrepancy),
        "decision_interval_bias_allowance_b_W_m_K": _json_vector(np.maximum(np.abs(full_prefix_discrepancy), np.abs(full_b_block_mean_discrepancy))),
        "decision_interval_bias_allowance_2b_W_m_K": _json_vector(np.maximum(np.abs(full_prefix_discrepancy), np.abs(full_block_mean_discrepancy))),
        "paired_block_delta_kappa_mean_W_m_K": block_delta.mean(axis=0).tolist()
            if len(block_delta) else [None, None, None],
        "delta_kappa_equivalence_by_direction": dict(zip("xyz", direction_equivalence)),
        "delta_j_rms_native": np.sqrt(np.einsum("ij,ij->j", delta, delta) / frames).tolist(),
        "delta_j_rms_role": "amplitude diagnostic only; it does not establish thermal-conductivity equivalence",
        "correlation_definition": "C_ab[i,j,l]=mean_t(J_a[i,t]*J_b[j,t+l]); denominator N-lag; no mean subtraction",
        "hac_integral_definition": "native rpmd_ja trapezoid: dt_native*0.5/(k_B*T_reference^2*V)*KAPPA_UNIT_CONVERSION; no mean subtraction",
        "hac_integral_units": "W/m/K",
        "native_time_ps": NATIVE_TIME_PS,
        "kappa_full_series_at_max_lag_W_m_K": {
            name: kappa[name][-1].tolist() for name in ("Ccc", "CcDelta", "CDeltac", "CDeltaDelta", "Caa", "DeltaKappa")},
        "delta_kappa_term_sum_residual_W_m_K": (kappa["DeltaKappa"][-1] -
            (kappa["CcDelta"][-1] + kappa["CDeltac"][-1] + kappa["CDeltaDelta"][-1])).tolist(),
        "ja_correlation_decomposition_max_absolute_error": decomposition_max_error,
        "arrays": {name: name for name in arrays},
    }
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path, help="one heat_current_rpmd_ja.out segment")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--epsilon-kappa", type=float, required=True,
                        help="pre-frozen equivalence margin in W/m/K")
    parser.add_argument("--block-frames", type=int, required=True)
    parser.add_argument("--max-lag-frames", type=int, required=True)
    parser.add_argument("--diagnostic", action="store_true",
                        help="explicitly analyze a DIAGNOSTIC reference segment")
    args = parser.parse_args(argv)
    try:
        report, arrays = analyze_current(args.current, args.epsilon_kappa, args.block_frames,
                                         args.max_lag_frames, args.diagnostic)
        prefix = args.output_prefix.resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
        report["arrays_npz"] = str(prefix.with_suffix(".npz"))
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        prefix.with_suffix(".json").write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, ValueError, MemoryError) as exc:
        parser.exit(2, f"rpmd_ja_current_analyze: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
