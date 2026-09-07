#!/usr/bin/env python3
"""Check one qNEP charge-diagnostic extxyz frame using only the standard library."""

import argparse
import math
import re


def read_frame(path):
    with open(path, encoding="utf-8") as stream:
        count_line = stream.readline()
        if not count_line:
            raise ValueError(f"{path}: missing atom count")
        count = int(count_line)
        header = stream.readline()
        match = re.search(r"(?:^| )Properties=([^ ]+)", header)
        if not match:
            raise ValueError(f"{path}: missing Properties")
        schema = match.group(1).split(":")
        columns = []
        offset = 0
        for index in range(0, len(schema), 3):
            name, kind, width = schema[index], schema[index + 1], int(schema[index + 2])
            columns.append((name, kind, width, offset))
            offset += width
        rows = []
        for row_index in range(count):
            row = stream.readline().split()
            if len(row) < offset:
                raise ValueError(f"{path}: frame row {row_index + 1} is incomplete")
            rows.append(row)
    frame = {"count": count, "species": None}
    for name, kind, width, start in columns:
        if name == "species" and kind == "S" and width == 1:
            frame["species"] = [row[start] for row in rows]
        elif kind == "R":
            frame[name] = [[float(row[start + i]) for i in range(width)] for row in rows]
    if frame["species"] is None:
        raise ValueError(f"{path}: missing species property")
    return frame


def scalar(frame, name):
    return [row[0] for row in frame[name]]


def max_abs(values):
    return max(map(abs, values), default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame")
    parser.add_argument("--plus")
    parser.add_argument("--minus")
    parser.add_argument("--dt-fs", type=float)
    parser.add_argument("--charge-atol", type=float, default=1e-6)
    parser.add_argument("--rate-atol", type=float, default=1e-6)
    parser.add_argument("--virial-atol", type=float, default=1e-6)
    args = parser.parse_args()
    if min(args.charge_atol, args.rate_atol, args.virial_atol) < 0.0:
        parser.error("atol values must be non-negative")
    frame = read_frame(args.frame)

    checks = {}
    raw_q, q = scalar(frame, "charge_raw"), scalar(frame, "charge")
    raw_d, d = scalar(frame, "charge_dudq_raw"), scalar(frame, "charge_dudq")
    raw_rate, rate = scalar(frame, "charge_rate_raw"), scalar(frame, "charge_rate")
    checks["q projection"] = max_abs(
        q[i] - (raw_q[i] - sum(raw_q) / len(raw_q)) for i in range(frame["count"])
    )
    checks["D projection"] = max_abs(
        d[i] - (raw_d[i] - sum(raw_d) / len(raw_d)) for i in range(frame["count"])
    )
    checks["charge-rate projection"] = max_abs(
        rate[i] - (raw_rate[i] - sum(raw_rate) / len(raw_rate)) for i in range(frame["count"])
    )
    checks["sum(charge)"] = abs(sum(q))
    checks["sum(charge_rate)"] = abs(sum(rate))

    total = frame["virial"]
    nep = frame["virial_nep"]
    fixed = frame["virial_electrostatic_fixed"]
    dynamic = frame["virial_dynamic_charge"]
    checks["virial split"] = max_abs(
        total[i][j] - nep[i][j] - fixed[i][j] - dynamic[i][j]
        for i in range(len(total))
        for j in range(9)
    )

    if args.plus is not None or args.minus is not None or args.dt_fs is not None:
        if not (args.plus and args.minus and args.dt_fs and args.dt_fs > 0.0):
            parser.error("--plus, --minus, and positive --dt-fs must be used together")
        plus_frame = read_frame(args.plus)
        minus_frame = read_frame(args.minus)
        for name, other in (("plus", plus_frame), ("minus", minus_frame)):
            if other["count"] != frame["count"] or other["species"] != frame["species"]:
                parser.error(f"{name} frame atom count/species order does not match frame")
        plus = scalar(plus_frame, "charge_raw")
        minus = scalar(minus_frame, "charge_raw")
        checks["charge-rate finite difference"] = max_abs(
            raw_rate[i] - (plus[i] - minus[i]) / (2.0 * args.dt_fs)
            for i in range(frame["count"])
        )

    for name, error in checks.items():
        print(f"{name}: max_abs_error={error:.9g}")

    def atol_for(name):
        if name == "virial split":
            return args.virial_atol
        if name in {"charge-rate projection", "sum(charge_rate)", "charge-rate finite difference"}:
            return args.rate_atol
        return args.charge_atol

    if not all(math.isfinite(error) and error <= atol_for(name) for name, error in checks.items()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
