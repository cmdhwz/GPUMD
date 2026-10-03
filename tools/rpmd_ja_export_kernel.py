#!/usr/bin/env python3
"""Export validated 45.x signed low-rank kernel tables as the GPUMD text format."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import math
from pathlib import Path

import numpy as np


def _g(x: Decimal) -> Decimal:
    if x == 0:
        return Decimal(1)
    if abs(x) < Decimal("1e-12"):
        x2 = x * x
        return Decimal(1) + x2 / 24 + x2 * x2 / 1920 + x2 * x2 * x2 / 322560
    half = x / 2
    return ((half.exp() - (-half).exp()) / 2) / half


def _h_decimal(x: Decimal) -> Decimal:
    if abs(x) < Decimal("1e-12"):
        x2 = x * x
        return Decimal(1) + x2 / 12 - x2 * x2 / 720 + x2 * x2 * x2 / 30240
    half = x / 2
    return half * (half.exp() + (-half).exp()) / (half.exp() - (-half).exp())


def _direct_pq(u: Decimal, v: Decimal) -> tuple[Decimal, Decimal]:
    if u == 0 and v == 0:
        return Decimal(-1) / 5760, Decimal(1) / 24
    if u == 0 or v == 0:
        x = v if u == 0 else u
        if abs(x) < Decimal("1e-8"):
            x2 = x * x
            return (Decimal(-1) / 5760, Decimal(1) / 24 - x2 / 1440 + x2 * x2 / 60480)
        half = x / 2
        exp = half.exp()
        coth = (exp + 1 / exp) / (exp - 1 / exp)
        log_first = coth / 2 - 1 / x
        sinh = (half.exp() - (-half).exp()) / 2
        log_second = 1 / (x * x) - 1 / (4 * sinh * sinh)
        p = (log_second / 4 - Decimal(1) / 48 + log_first * log_first / 8) / (x * x)
        q = (_h_decimal(x) - 1) / (2 * x * x)
        return p, q
    g_uv = _g(u) * _g(v)
    sp = (_g(u + v) / g_uv).sqrt()
    sm = (_g(u - v) / g_uv).sqrt()
    a = (sp + sm) / 2
    return (a - 1) / (u * u * v * v), (sp - sm) / (2 * u * v)


def _h(x: float) -> float:
    if abs(x) < 1.0e-4:
        x2 = x * x
        return 1.0 + x2 / 12.0 - x2 * x2 / 720.0 + x2 * x2 * x2 / 30240.0
    return x / (2.0 * math.tanh(x / 2.0))


def _certificate(u: float, degree: int, b: float = 0.9 * math.pi):
    eta = 2.0 * math.asinh(b / u)
    rho = math.exp(eta)
    scale = math.sqrt(_h(math.sqrt(u * u + b * b))) / math.cos(b / 2.0)
    maxima = np.array([(scale + 1.0) / b**4, scale / b**2])
    interpolation = 16.0 * maxima * rho ** (-degree - 1) / (1.0 - 1.0 / rho) ** 2
    s2_analytic = 8.0 * maxima / rho * (1.0 + 1.0 / rho) / (1.0 - 1.0 / rho) ** 4
    return rho, maxima, interpolation, s2_analytic


def _degree_for(u: float, tolerance: float) -> int:
    rho, maxima, _, _ = _certificate(u, 0)
    return max(1, math.ceil(math.log(16.0 * max(maxima) / ((1.0 - 1.0 / rho) ** 2 * tolerance)) / math.log(rho)))


def _validate_dimensions(degree: int, values, vectors):
    if degree < 0 or degree > 512 or any(
        np.asarray(value).ndim != 1 or len(value) == 0 or len(value) > degree + 1 or
        np.asarray(vector).shape != (degree + 1, len(value))
        for value, vector in zip(values, vectors)
    ) or len(values) != 2 or len(vectors) != 2:
        raise ValueError("kernel ranks must be between 1 and degree+1 with matching vector dimensions")


def _write_table(output: Path, u: float, degree: int, values, vectors, budgets, s2_values):
    _validate_dimensions(degree, values, vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("GPUMDJA_KERNEL 1\n")
        stream.write(f"U {u:.17g}\ndegree {degree}\n")
        stream.write(f"P_rank {len(values[0])}\nQ_rank {len(values[1])}\n")
        for key, value in (("P_error", budgets[0]), ("Q_error", budgets[1]),
                           ("P_S2", s2_values[0]), ("Q_S2", s2_values[1])):
            stream.write(f"{key} {value:.17g}\n")
        for name, vals, vecs in (("P", values[0], vectors[0]), ("Q", values[1], vectors[1])):
            stream.write(f"{name}_values\n")
            stream.write(" ".join(f"{x:.17g}" for x in vals) + "\n")
            stream.write(f"{name}_vectors\n")
            stream.write(" ".join(f"{x:.17g}" for x in vecs.ravel(order="C")) + "\n")
        stream.write("END\n")


def _build(u: float, output: Path, tolerance: float, rank_budget: float, arithmetic_budget: float, degree_max: int):
    degree = _degree_for(u, tolerance)
    if degree > degree_max:
        raise ValueError(f"U={u:g} needs degree {degree}, above the guarded maximum {degree_max}")
    n = degree + 1
    theta = np.pi * (np.arange(n) + 0.5) / n
    abscissae = u * np.sqrt((1.0 + np.cos(theta)) / 2.0)
    samples = [np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)]
    with localcontext() as context:
        context.prec = 80
        for i, wa in enumerate(abscissae):
            ua = Decimal(repr(float(wa)))
            for j in range(i + 1):
                pv, qv = _direct_pq(ua, Decimal(repr(float(abscissae[j]))))
                samples[0][i, j] = samples[0][j, i] = float(pv)
                samples[1][i, j] = samples[1][j, i] = float(qv)
    transform = np.cos(np.outer(np.arange(n), theta)) * np.r_[1.0 / n, np.full(n - 1, 2.0 / n)][:, None]
    compressed = []
    ranks, residuals, s2 = [], [], []
    values_list, vectors_list = [], []
    index = np.arange(n, dtype=np.float64)
    for sample in samples:
        coefficients = transform @ sample @ transform.T
        coefficients = 0.5 * (coefficients + coefficients.T)
        values, vectors = np.linalg.eigh(coefficients)
        order = np.argsort(-np.abs(values))
        values, vectors = values[order], vectors[:, order]
        for rank in range(1, n + 1):
            fitted = (vectors[:, :rank] * values[:rank]) @ vectors[:, :rank].T
            residual = float(np.sum(np.abs(fitted - coefficients)))
            if residual <= rank_budget:
                break
        else:
            raise ValueError(f"U={u:g} rank budget {rank_budget:g} is not met at degree {degree}")
        ranks.append(rank)
        residuals.append(residual)
        s2.append(float(np.sum(np.abs(fitted) * (index[:, None] ** 2 + index[None, :] ** 2))))
        values_list.append(values[:rank].copy())
        vectors_list.append(vectors[:, :rank].copy())
        compressed.append(fitted)

    rho, maxima, interpolation, _ = _certificate(u, degree)
    total_budget = interpolation + np.asarray(residuals) + arithmetic_budget
    points = ((0.0, 0.0), (0.0, 0.37), (0.37, 0.0), (1.0, 1.0),
              (0.03, 0.80), (0.40, 0.4000000001), (0.99, 0.997), (0.001, 0.70))
    max_offgrid = [0.0, 0.0]
    with localcontext() as context:
        context.prec = 80
        for ua_scale, vb_scale in points:
            ua, vb = Decimal(repr(u * ua_scale)), Decimal(repr(u * vb_scale))
            exact = _direct_pq(ua, vb)
            x = 2.0 * (float(ua) / u) ** 2 - 1.0
            y = 2.0 * (float(vb) / u) ** 2 - 1.0
            tx = np.cos(np.arange(n) * math.acos(max(-1.0, min(1.0, x))))
            ty = np.cos(np.arange(n) * math.acos(max(-1.0, min(1.0, y))))
            for k in range(2):
                approx = float(tx @ compressed[k] @ ty)
                error = abs(approx - float(exact[k]))
                max_offgrid[k] = max(max_offgrid[k], error)
                if error > total_budget[k] * 1.05:
                    raise ValueError(f"U={u:g} Decimal off-grid error {error:g} exceeded budget {total_budget[k]:g}")
    _write_table(output, u, degree, values_list, vectors_list, total_budget, s2)
    import json
    metadata = {"U": u, "degree": degree, "ranks": ranks,
                "interpolation_certificate": interpolation.tolist(), "rank_L1_residual": residuals,
                "arithmetic_reserve": [arithmetic_budget, arithmetic_budget],
                "total_kernel_budget": total_budget.tolist(), "S2_actual_compressed": s2,
                "offgrid_decimal80_max_abs": max_offgrid, "chebyshev_scale": "x=2*(u_mode/U)^2-1"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"U={u:g} degree={degree}, ranks={ranks}, interpolation={interpolation}, L1={residuals}, S2_actual={s2}")
    print(f"Decimal80 off-grid max errors={max_offgrid}, total budgets={total_budget}")
    print(f"coefficient matrices: {2 * n * n * 8 / 1024**2:.2f} MiB; dense DCT/eigensolve work remains O(degree^3)")


def _export_npz(source: Path, output: Path):
    table = np.load(source)
    u, degree = float(table["U"]), int(table["degree"])
    required = ("total_kernel_budget", "interpolation_bounds", "rank_l1", "arithmetic_budget",
                "P_values", "P_vectors", "Q_values", "Q_vectors")
    missing = [key for key in required if key not in table]
    if missing:
        raise ValueError(f"kernel NPZ missing required arrays: {missing}")
    values = [table["P_values"], table["Q_values"]]
    vectors = [table["P_vectors"], table["Q_vectors"]]
    _validate_dimensions(degree, values, vectors)
    indices = np.arange(degree + 1, dtype=np.float64)
    s2 = [float(np.sum(np.abs((vec * val[None, :]) @ vec.T) *
                       (indices[:, None] ** 2 + indices[None, :] ** 2)))
          for val, vec in zip(values, vectors)]
    _write_table(output, u, degree, values, vectors, table["total_kernel_budget"], s2)
    import json
    metadata = {"source_npz": str(source), "U": u, "degree": degree,
                "interpolation_certificate": np.asarray(table["interpolation_bounds"]).tolist(),
                "rank_L1_residual": np.asarray(table["rank_l1"]).tolist(),
                "arithmetic_reserve": np.asarray(table["arithmetic_budget"]).tolist(),
                "total_kernel_budget": np.asarray(table["total_kernel_budget"]).tolist(),
                "S2_actual_compressed": s2, "provenance": "theory NPZ exported without recomputing kernel"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"exported U={u:g}, degree={degree}, ranks={len(table['P_values'])}/{len(table['Q_values'])} to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--npz", type=Path, help="existing theory kernel NPZ (export only; does not recompute)")
    mode.add_argument("--build-u", type=float, help="build a wider table directly from 80-digit Decimal Fock weights")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--rank-budget", type=float, default=1.0e-10)
    parser.add_argument("--arithmetic-budget", type=float, default=1.0e-10)
    parser.add_argument("--degree-max", type=int, default=512, help="guard degree and dense eigensolve cost")
    args = parser.parse_args()
    if args.npz:
        _export_npz(args.npz, args.output)
    else:
        if not math.isfinite(args.build_u) or args.build_u <= 0.0:
            parser.error("--build-u must be positive and finite")
        _build(args.build_u, args.output, args.tolerance, args.rank_budget, args.arithmetic_budget, args.degree_max)


if __name__ == "__main__":
    main()
