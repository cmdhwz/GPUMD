#!/usr/bin/env python3
"""Reproduce GPUMD PPPM Gopt from pppm_debug_kspace.out."""

import argparse
import math
import struct


def f32(value):
    """Round a Python number to the IEEE-754 binary32 value used by GPUMD."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


TWO_PI = f32(6.2831853)
SINC_COEFF = tuple(f32(x) for x in (
    1.0,
    -1.6666667e-1,
    8.3333333e-3,
    -1.9841270e-4,
    2.7557319e-6,
    -2.5052108e-8,
))
G_COEFF = tuple(f32(x) for x in (
    1.0000000e+00,
    -1.6666667e+00,
    7.7777778e-01,
    -8.9947090e-02,
    7.0546737e-04,
))


def _determinant(h):
    return abs(
        h[0] * (h[4] * h[8] - h[5] * h[7])
        + h[1] * (h[5] * h[6] - h[3] * h[8])
        + h[2] * (h[3] * h[7] - h[4] * h[6])
    )


def _inverse_from_h(h):
    inverse = [
        h[4] * h[8] - h[5] * h[7],
        h[2] * h[7] - h[1] * h[8],
        h[1] * h[5] - h[2] * h[4],
        h[5] * h[6] - h[3] * h[8],
        h[0] * h[8] - h[2] * h[6],
        h[2] * h[3] - h[0] * h[5],
        h[3] * h[7] - h[4] * h[6],
        h[1] * h[6] - h[0] * h[7],
        h[0] * h[4] - h[1] * h[3],
    ]
    determinant = (
        h[0] * (h[4] * h[8] - h[5] * h[7])
        + h[1] * (h[5] * h[6] - h[3] * h[8])
        + h[2] * (h[3] * h[7] - h[4] * h[6])
    )
    return [value / determinant for value in inverse]


def _box_data(box):
    """Return (double h, double inverse(h), double abs(det(h)))."""
    if isinstance(box, dict):
        h = [float(value) for value in box["h"]]
        h_inverse = [float(value) for value in box["h_inverse"]]
        return h, h_inverse, _determinant(h)

    if len(box) == 3:
        lx, ly, lz = (float(value) for value in box)
        h = [lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz]
        h_inverse = [1.0 / lx, 0.0, 0.0, 0.0, 1.0 / ly, 0.0, 0.0, 0.0, 1.0 / lz]
        return h, h_inverse, abs(lx * ly * lz)

    if len(box) == 9:
        h = [float(value) for value in box]
        return h, _inverse_from_h(h), _determinant(h)

    raise ValueError("box must contain 3 lengths, 9 h entries, or h/h_inverse")


def _gpumd_sinc(x):
    x = f32(x)
    if f32(x * x) <= f32(1.0):
        term = f32(1.0)
        y = f32(0.0)
        x_squared = f32(x * x)
        for coefficient in SINC_COEFF:
            y = f32(y + f32(coefficient * term))
            term = f32(term * x_squared)
        return y
    return f32(f32(math.sin(float(x))) / x)


def _gpumd_denominator(nk, two_pi_over_k, direction):
    argument = f32(f32(f32(0.5) * two_pi_over_k[direction]) * nk[direction])
    u = f32(math.sin(float(argument)))
    u = f32(u * u)
    t = u
    t = f32(f32(G_COEFF[4] * t) + G_COEFF[3])
    t = f32(f32(t * u) + G_COEFF[2])
    t = f32(f32(t * u) + G_COEFF[1])
    t = f32(f32(t * u) + G_COEFF[0])
    return f32(t * t)


def gpumd_gopt(k_index, mesh, box, alpha):
    """Return GPUMD's float32 Gopt for raw mesh index (ix, iy, iz).

    `box` may be three orthogonal lengths, a row-major 3x3 h matrix, or a
    dict containing the exact debug-header `box_h` and `box_h_inverse` arrays.
    `alpha` is the GPUMD inverse-length Ewald parameter before initialization.
    """
    kx_size, ky_size, kz_size = (int(value) for value in mesh)
    ix, iy, iz = (int(value) for value in k_index)
    if not (0 <= ix < kx_size and 0 <= iy < ky_size and 0 <= iz < kz_size):
        raise ValueError("k_index is outside mesh")

    h, h_inverse, volume = _box_data(box)
    del h
    alpha_f = f32(alpha)
    alpha_factor = f32(f32(0.25) / f32(alpha_f * alpha_f))
    two_pi_over_v = f32(float(TWO_PI) / volume)
    two_pi_over_k = [
        f32(float(TWO_PI) / kx_size),
        f32(float(TWO_PI) / ky_size),
        f32(float(TWO_PI) / kz_size),
    ]

    b = [
        [f32(TWO_PI * f32(h_inverse[0 + d])) for d in range(3)],
        [f32(TWO_PI * f32(h_inverse[3 + d])) for d in range(3)],
        [f32(TWO_PI * f32(h_inverse[6 + d])) for d in range(3)],
    ]

    nk = [ix, iy, iz]
    for d, size in enumerate((kx_size, ky_size, kz_size)):
        if nk[d] >= size // 2:
            nk[d] -= size

    kx = f32(
        f32(nk[0] * b[0][0])
        + f32(nk[1] * b[1][0])
        + f32(nk[2] * b[2][0])
    )
    ky = f32(
        f32(nk[0] * b[0][1])
        + f32(nk[1] * b[1][1])
        + f32(nk[2] * b[2][1])
    )
    kz = f32(
        f32(nk[0] * b[0][2])
        + f32(nk[1] * b[1][2])
        + f32(nk[2] * b[2][2])
    )
    k_squared = f32(f32(kx * kx) + f32(ky * ky) + f32(kz * kz))
    if k_squared == f32(0.0):
        return f32(0.0)

    denominator = [
        _gpumd_denominator(nk, two_pi_over_k, 0),
        _gpumd_denominator(nk, two_pi_over_k, 1),
        _gpumd_denominator(nk, two_pi_over_k, 2),
    ]
    numerator = _gpumd_sinc(f32(f32(f32(0.5) * two_pi_over_k[0]) * nk[0]))
    numerator = f32(numerator * _gpumd_sinc(f32(f32(f32(0.5) * two_pi_over_k[1]) * nk[1])))
    numerator = f32(numerator * _gpumd_sinc(f32(f32(f32(0.5) * two_pi_over_k[2]) * nk[2])))
    original_numerator = numerator
    for _ in range(4):
        numerator = f32(numerator * original_numerator)
    numerator = f32(numerator * numerator)

    exponent = f32(f32(-k_squared) * alpha_factor)
    exponential = f32(math.exp(float(exponent)))
    g_opt = f32(f32(numerator * two_pi_over_v) / k_squared)
    g_opt = f32(g_opt * exponential)
    denominator_product = f32(f32(denominator[0] * denominator[1]) * denominator[2])
    return f32(g_opt / denominator_product)


def _read_kspace(path):
    frames = []
    current = None
    with open(path, encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# pppm_frame "):
                if current is not None:
                    frames.append(current)
                current = {"frame": int(line.split()[2]), "rows": []}
                continue
            if current is None:
                continue
            if line.startswith("# pppm_mesh_size "):
                current["mesh"] = tuple(int(value) for value in line.split()[2:5])
            elif line.startswith("# source_signature "):
                current["source_signature"] = line.split(maxsplit=2)[2]
            elif line.startswith("# box_h "):
                current["h"] = [float(value) for value in line.split()[2:11]]
            elif line.startswith("# box_h_inverse "):
                current["h_inverse"] = [float(value) for value in line.split()[2:11]]
            elif not line.startswith("#"):
                values = line.split()
                if len(values) < 14:
                    raise ValueError(f"{path}: malformed k-space row: {line}")
                current["rows"].append(
                    (
                        int(values[0]),
                        int(values[1]),
                        int(values[2]),
                        int(values[3]),
                        int(values[4]),
                        int(values[5]),
                        float(values[11]),
                    )
                )
    if current is not None:
        frames.append(current)
    if not frames:
        raise ValueError(f"{path}: no PPPM k-space frame found")
    return frames


def _correlation(left, right):
    if len(left) < 2:
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return covariance / (left_norm * right_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kspace_file")
    parser.add_argument("--alpha", type=float, required=True, help="GPUMD PPPM alpha")
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--expected-signature")
    args = parser.parse_args()
    if args.tol < 0.0:
        parser.error("--tol must be non-negative")

    frames = _read_kspace(args.kspace_file)
    all_python = []
    all_gpumd = []
    for frame in frames:
        required = {"mesh", "h", "h_inverse"}
        missing = required.difference(frame)
        if missing:
            raise ValueError(f"frame {frame['frame']} missing header fields: {sorted(missing)}")
        if args.expected_signature is not None and frame.get("source_signature") != args.expected_signature:
            raise ValueError(f"frame {frame['frame']} source signature mismatch")
        expected_mode_count = frame["mesh"][0] * frame["mesh"][1] * frame["mesh"][2]
        if len(frame["rows"]) != expected_mode_count:
            raise ValueError(
                f"frame {frame['frame']} has {len(frame['rows'])} rows, "
                f"expected {expected_mode_count}"
            )

        box = {"h": frame["h"], "h_inverse": frame["h_inverse"]}
        python_values = []
        gpumd_values = []
        seen = set()
        for ix, iy, iz, nx, ny, nz, gpumd_value in frame["rows"]:
            mode = (ix, iy, iz)
            if mode in seen:
                raise ValueError(f"frame {frame['frame']} has duplicate mode {mode}")
            seen.add(mode)
            expected_signed = tuple(
                raw - size if raw >= size // 2 else raw
                for raw, size in zip(mode, frame["mesh"])
            )
            if (nx, ny, nz) != expected_signed:
                raise ValueError(
                    f"frame {frame['frame']} signed index mismatch for {mode}: "
                    f"file={(nx, ny, nz)} expected={expected_signed}"
                )
            python_value = gpumd_gopt((ix, iy, iz), frame["mesh"], box, args.alpha)
            python_values.append(python_value)
            gpumd_values.append(gpumd_value)
        differences = [a - b for a, b in zip(python_values, gpumd_values)]
        max_abs = max((abs(value) for value in differences), default=0.0)
        rms = math.sqrt(sum(value * value for value in differences) / len(differences)) if differences else 0.0
        reference_norm = math.sqrt(sum(value * value for value in gpumd_values))
        relative = math.sqrt(sum(value * value for value in differences)) / reference_norm if reference_norm else 0.0
        print(
            f"frame={frame['frame']} modes={len(differences)} "
            f"max_abs_error={max_abs:.17g} rms_error={rms:.17g} "
            f"relative_error={relative:.17g} correlation={_correlation(python_values, gpumd_values):.17g}"
        )
        all_python.extend(python_values)
        all_gpumd.extend(gpumd_values)

    differences = [a - b for a, b in zip(all_python, all_gpumd)]
    max_abs = max((abs(value) for value in differences), default=0.0)
    rms = math.sqrt(sum(value * value for value in differences) / len(differences)) if differences else 0.0
    reference_norm = math.sqrt(sum(value * value for value in all_gpumd))
    relative = math.sqrt(sum(value * value for value in differences)) / reference_norm if reference_norm else 0.0
    print(
        f"all_frames={len(frames)} modes={len(differences)} "
        f"max_abs_error={max_abs:.17g} rms_error={rms:.17g} "
        f"relative_error={relative:.17g} correlation={_correlation(all_python, all_gpumd):.17g}"
    )
    if max_abs > args.tol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
