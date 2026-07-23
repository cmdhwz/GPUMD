#!/usr/bin/env python3
"""Reorder an XYZ/extended-XYZ water/NaCl structure for GPUMD qtip4pf.

Each oxygen is matched globally to two hydrogens using minimum-image O-H
distances.  The output contains consecutive O H H triplets followed by all
Na/Cl ions in their original order.  No virtual M atoms are written.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import shlex
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder water atoms into O H H triplets for qtip4pf."
    )
    parser.add_argument("input", type=Path, help="input XYZ or extended XYZ")
    parser.add_argument("output", type=Path, help="output XYZ")
    parser.add_argument(
        "--max-oh",
        type=float,
        default=1.30,
        metavar="ANGSTROM",
        help="maximum accepted O-H distance (default: 1.30 A)",
    )
    parser.add_argument(
        "--no-pbc",
        action="store_true",
        help="do not apply the minimum-image convention even if Lattice is present",
    )
    return parser.parse_args()


def invert_3x3(a: list[list[float]]) -> list[list[float]]:
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    if abs(det) < 1.0e-14:
        raise ValueError("Lattice matrix is singular")
    return [
        [
            (a[(j + 1) % 3][(i + 1) % 3] * a[(j + 2) % 3][(i + 2) % 3]
             - a[(j + 1) % 3][(i + 2) % 3] * a[(j + 2) % 3][(i + 1) % 3])
            / det
            for j in range(3)
        ]
        for i in range(3)
    ]


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(a[i][j] * x[j] for j in range(3)) for i in range(3)]


def lattice_from_comment(comment: str) -> list[list[float]] | None:
    match = re.search(r'(?:^|\s)Lattice=("[^"]*"|\S+)', comment)
    if not match:
        return None
    values = shlex.split(match.group(1))[0].split()
    if len(values) != 9:
        raise ValueError("Lattice must contain 9 numbers")
    v = [float(x) for x in values]
    # Extended XYZ stores lattice vectors as rows.  Cartesian = lattice^T frac.
    return [[v[0], v[3], v[6]], [v[1], v[4], v[7]], [v[2], v[5], v[8]]]


def pbc_flags_from_comment(comment: str) -> tuple[bool, bool, bool]:
    match = re.search(r'(?:^|\s)pbc=("[^"]*"|\S+)', comment, re.IGNORECASE)
    if not match:
        return True, True, True
    fields = shlex.split(match.group(1))[0].split()
    if len(fields) != 3:
        raise ValueError("pbc must contain three flags")
    return tuple(x.lower() in {"t", "true", "1"} for x in fields)  # type: ignore[return-value]


def distance(
    a: list[float],
    b: list[float],
    lattice: list[list[float]] | None,
    inverse: list[list[float]] | None,
    pbc: tuple[bool, bool, bool],
) -> float:
    delta = [b[i] - a[i] for i in range(3)]
    if lattice is not None and inverse is not None:
        frac = matvec(inverse, delta)
        for i in range(3):
            if pbc[i]:
                frac[i] -= round(frac[i])
        delta = matvec(lattice, frac)
    return math.sqrt(sum(x * x for x in delta))


def assign_hydrogens(
    atoms: list[tuple[str, list[float], str]],
    oxygen: list[int],
    hydrogen: list[int],
    lattice: list[list[float]] | None,
    inverse: list[list[float]] | None,
    pbc: tuple[bool, bool, bool],
    cutoff: float,
) -> dict[int, list[tuple[float, int]]]:
    """Assign each H to its nearest O using a cutoff-sized spatial grid."""
    if lattice is not None and inverse is not None:
        grid_position = [matvec(inverse, atom[1]) for atom in atoms]
        fractional_reach = [
            cutoff * math.sqrt(sum(inverse[i][j] ** 2 for j in range(3)))
            for i in range(3)
        ]
        for position in grid_position:
            for axis in range(3):
                if pbc[axis]:
                    position[axis] -= math.floor(position[axis])
    else:
        grid_position = [atom[1][:] for atom in atoms]
        fractional_reach = [cutoff, cutoff, cutoff]

    origins = [min(grid_position[i][axis] for i in oxygen) for axis in range(3)]
    cell_width: list[float] = []
    periodic_cells: list[int | None] = []
    for axis in range(3):
        if lattice is not None and pbc[axis]:
            number = max(1, int(math.floor(1.0 / fractional_reach[axis])))
            cell_width.append(1.0 / number)
            periodic_cells.append(number)
            origins[axis] = 0.0
        else:
            cell_width.append(max(fractional_reach[axis], 1.0e-12))
            periodic_cells.append(None)

    def cell_key(position: list[float]) -> tuple[int, int, int]:
        key = []
        for axis in range(3):
            value = int(math.floor((position[axis] - origins[axis]) / cell_width[axis]))
            number = periodic_cells[axis]
            if number is not None:
                value %= number
            key.append(value)
        return key[0], key[1], key[2]

    cells: dict[tuple[int, int, int], list[int]] = {}
    for o in oxygen:
        cells.setdefault(cell_key(grid_position[o]), []).append(o)

    assigned: dict[int, list[tuple[float, int]]] = {o: [] for o in oxygen}
    for h in hydrogen:
        center = cell_key(grid_position[h])
        neighbor_keys: set[tuple[int, int, int]] = set()
        for offset in itertools.product((-1, 0, 1), repeat=3):
            key = []
            for axis in range(3):
                value = center[axis] + offset[axis]
                number = periodic_cells[axis]
                if number is not None:
                    value %= number
                key.append(value)
            neighbor_keys.add((key[0], key[1], key[2]))

        best_o = -1
        best_distance = math.inf
        for key in neighbor_keys:
            for o in cells.get(key, []):
                oh_distance = distance(atoms[o][1], atoms[h][1], lattice, inverse, pbc)
                if oh_distance < best_distance:
                    best_distance = oh_distance
                    best_o = o
        if best_o < 0 or best_distance > cutoff:
            raise ValueError(
                f"H atom at XYZ line {h + 3} has no O within {cutoff:.6f} A; "
                "check units/box/structure"
            )
        assigned[best_o].append((best_distance, h))

    bad = [(o, len(matches)) for o, matches in assigned.items() if len(matches) != 2]
    if bad:
        details = ", ".join(f"line {o + 3}: {count} H" for o, count in bad[:8])
        if len(bad) > 8:
            details += f", ... ({len(bad)} O atoms affected)"
        raise ValueError(
            "nearest-O assignment did not produce exactly two H for every O: " + details
        )
    return assigned


def main() -> int:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")

    lines = args.input.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ file is incomplete")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("the first line must be the atom count") from exc
    if len(lines) < atom_count + 2:
        raise ValueError(f"expected {atom_count} atom lines, found {len(lines) - 2}")

    comment = lines[1]
    atoms: list[tuple[str, list[float], str]] = []
    for line_number, raw in enumerate(lines[2 : atom_count + 2], start=3):
        fields = raw.split()
        if len(fields) < 4:
            raise ValueError(f"line {line_number}: expected species x y z")
        species = fields[0]
        if species not in {"O", "H", "Na", "Cl"}:
            raise ValueError(f"line {line_number}: unsupported species {species!r}")
        try:
            position = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid Cartesian coordinates") from exc
        atoms.append((species, position, raw))

    oxygen = [i for i, atom in enumerate(atoms) if atom[0] == "O"]
    hydrogen = [i for i, atom in enumerate(atoms) if atom[0] == "H"]
    if not oxygen:
        raise ValueError("no oxygen atoms found")
    if len(hydrogen) != 2 * len(oxygen):
        raise ValueError(
            f"expected exactly 2 H per O, found {len(oxygen)} O and {len(hydrogen)} H"
        )

    lattice = None if args.no_pbc else lattice_from_comment(comment)
    inverse = invert_3x3(lattice) if lattice is not None else None
    pbc = (False, False, False) if args.no_pbc else pbc_flags_from_comment(comment)

    water_h = assign_hydrogens(
        atoms, oxygen, hydrogen, lattice, inverse, pbc, args.max_oh
    )

    worst = max(d for matches in water_h.values() for d, _ in matches)
    if worst > args.max_oh:
        raise ValueError(
            f"largest assigned O-H distance is {worst:.6f} A, exceeding "
            f"--max-oh {args.max_oh:.6f} A; check units/box/structure"
        )

    output_lines = [str(atom_count), comment]
    for o in oxygen:
        output_lines.append(atoms[o][2])
        for _, h in sorted(water_h[o]):
            output_lines.append(atoms[h][2])
    output_lines.extend(atom[2] for atom in atoms if atom[0] in {"Na", "Cl"})
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(
        f"Wrote {args.output}: {len(oxygen)} waters, "
        f"{sum(atom[0] == 'Na' for atom in atoms)} Na, "
        f"{sum(atom[0] == 'Cl' for atom in atoms)} Cl; "
        f"maximum assigned O-H distance = {worst:.6f} A"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
