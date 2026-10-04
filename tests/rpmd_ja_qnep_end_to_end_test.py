#!/usr/bin/env python3
"""Generate a qNEP v3 reference, then compare the same RPMD case with JA off/on."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )


def numeric(path: Path, columns: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise AssertionError(f"required output was not produced: {path}")
    values = np.loadtxt(path, comments="#", ndmin=2)
    if not np.isfinite(values).all():
        raise AssertionError(f"non-finite data in {path}")
    if columns is not None and values.shape[1] != columns:
        raise AssertionError(f"expected {columns} columns in {path}; found {values.shape[1]}")
    return values


def write_run_input(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_inputs(destination: Path, model_xyz: Path, qnep_model: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(model_xyz, destination / "model.xyz")
    shutil.copy2(qnep_model, destination / "nep.txt")


def compare_fresh_restart(baseline: Path, candidate: Path) -> None:
    baseline_restart = baseline / "restart_beads.xyz"
    candidate_restart = candidate / "restart_beads.xyz"
    if not baseline_restart.is_file() or not candidate_restart.is_file():
        raise AssertionError("both runs must write a fresh final restart_beads.xyz")
    if baseline_restart.read_bytes() != candidate_restart.read_bytes():
        raise AssertionError("JA off/on runs changed the RPMD restart differently")


def compare_outputs(baseline: Path, candidate: Path, rtol: float, atol: float) -> None:
    for filename in ("heat_current_centroid.out", "hac_centroid.out"):
        np.testing.assert_allclose(
            numeric(candidate / filename), numeric(baseline / filename),
            rtol=rtol, atol=atol, err_msg=filename,
        )
    baseline_thermo = numeric(baseline / "thermo.out", columns=18)
    candidate_thermo = numeric(candidate / "thermo.out", columns=18)
    if baseline_thermo.shape[0] < 2 or candidate_thermo.shape[0] < 2:
        raise AssertionError("dump_thermo must produce multiple rows for an energy-drift comparison")
    np.testing.assert_allclose(
        candidate_thermo, baseline_thermo,
        rtol=rtol, atol=atol, err_msg="thermo.out",
    )
    baseline_energy_drift = baseline_thermo[:, 1] + baseline_thermo[:, 2]
    baseline_energy_drift -= baseline_energy_drift[0]
    candidate_energy_drift = candidate_thermo[:, 1] + candidate_thermo[:, 2]
    candidate_energy_drift -= candidate_energy_drift[0]
    np.testing.assert_allclose(
        candidate_energy_drift, baseline_energy_drift,
        rtol=rtol, atol=atol, err_msg="total-energy drift (K + U)",
    )
    compare_fresh_restart(baseline, candidate)


def check_ja_current_and_hac(directory: Path, rtol: float, atol: float) -> None:
    current = numeric(directory / "heat_current_rpmd_ja.out", columns=10)
    hac = numeric(directory / "hac_rpmd_ja.out", columns=8)
    centroid = current[:, 1:4]
    delta = current[:, 4:7]
    ja = current[:, 7:10]
    np.testing.assert_allclose(ja, centroid + delta, rtol=rtol, atol=atol, err_msg="JA decomposition")

    expected_hac = np.empty((hac.shape[0], 3), dtype=float)
    for row, lag_value in enumerate(hac[:, 0]):
        lag = int(lag_value)
        if lag < 0 or lag >= current.shape[0] or lag_value != lag:
            raise AssertionError("invalid HAC lag index")
        count = current.shape[0] - lag
        cc = np.sum(centroid[:count] * centroid[lag:lag + count], axis=0) / count
        c_delta = np.sum(centroid[:count] * delta[lag:lag + count], axis=0) / count
        delta_c = np.sum(delta[:count] * centroid[lag:lag + count], axis=0) / count
        delta_delta = np.sum(delta[:count] * delta[lag:lag + count], axis=0) / count
        rebuilt = cc + c_delta + delta_c + delta_delta
        direct = np.sum(ja[:count] * ja[lag:lag + count], axis=0) / count
        np.testing.assert_allclose(rebuilt, direct, rtol=rtol, atol=atol, err_msg=f"HAC cross terms lag {lag}")
        expected_hac[row] = direct
    np.testing.assert_allclose(hac[:, 2:5], expected_hac, rtol=rtol, atol=atol, err_msg="JA HAC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpumd", required=True, type=Path)
    parser.add_argument("--model-xyz", required=True, type=Path)
    parser.add_argument("--qnep-model", required=True, type=Path)
    parser.add_argument("--kernel-table", required=True, type=Path)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--pppm-spacing", type=float, default=1.0)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--time-step", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rtol", type=float, default=2.0e-7)
    parser.add_argument("--atol", type=float, default=2.0e-9)
    args = parser.parse_args()

    for path in (args.gpumd, args.model_xyz, args.qnep_model, args.kernel_table):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if not all(math.isfinite(value) and value > 0.0 for value in (
        args.temperature, args.fd_step, args.pppm_spacing, args.time_step
    )):
        parser.error("temperature, fd-step, and PPPM spacing must be positive")
    if args.beads < 2 or args.steps < 2:
        parser.error("use at least two beads and two production steps, with a positive time step")
    if not all(math.isfinite(value) and value >= 0.0 for value in (args.rtol, args.atol)):
        parser.error("rtol and atol must be finite and non-negative")
    model_header = args.qnep_model.read_text(encoding="utf-8", errors="replace").splitlines()
    if not model_header or not model_header[0].startswith(
        ("nep4_charge1", "nep4_charge2", "nep4_zbl_charge1", "nep4_zbl_charge2")
    ):
        parser.error("qnep-model must contain a charge1 or charge2 qNEP model")
    xyz_lines = args.model_xyz.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(xyz_lines) < 2 or re.search(r"Properties=.*(?:=|:)vel(?:ocity)?(?:[:=]|$)", xyz_lines[1]):
        parser.error("model-xyz must omit velocity properties so the seed run uses the requested fixed seed")
    gpumd = args.gpumd.resolve()
    if not gpumd.is_file():
        parser.error(f"GPUMD executable does not exist: {gpumd}")

    with tempfile.TemporaryDirectory(prefix="gpumd-qnep-ja-e2e-") as temporary:
        root = Path(temporary)
        generate_dir = root / "generate"
        copy_inputs(generate_dir, args.model_xyz.resolve(), args.qnep_model.resolve())
        kernel_path = generate_dir / "kernel.tbl"
        shutil.copy2(args.kernel_table.resolve(), kernel_path)
        write_run_input(generate_dir / "run.in", [
            "potential nep.txt",
            f"kspace pppm {args.pppm_spacing:.17g}",
            f"rpmd_ja generate_sparse ja_reference.bin {args.temperature:.17g} {args.fd_step:.17g} kernel.tbl",
        ])
        run([str(gpumd)], generate_dir)
        final_reference = generate_dir / "ja_reference.bin"
        sidecar = Path(str(final_reference) + ".stability")
        if not final_reference.is_file() or not sidecar.is_file():
            raise AssertionError("generate_sparse did not produce the final v3 reference and stability sidecar")

        seed_dir = root / "seed"
        copy_inputs(seed_dir, args.model_xyz.resolve(), args.qnep_model.resolve())
        write_run_input(seed_dir / "run.in", [
            "potential nep.txt",
            f"kspace pppm {args.pppm_spacing:.17g}",
            f"velocity {args.temperature:.17g} seed {args.seed}",
            f"time_step {args.time_step:.17g}",
            f"ensemble pimd {args.beads} {args.temperature:.17g} {args.temperature:.17g} 100",
            "dump_pimd_restart 1",
            "run 1",
        ])
        run([str(gpumd)], seed_dir)
        bead_restart = seed_dir / "restart_beads.xyz"
        if not bead_restart.is_file():
            raise AssertionError("fixed-seed RPMD seed run did not produce restart_beads.xyz")

        correlation_steps = min(args.steps, 100)
        output_dirs = {}
        for mode in ("off", "on"):
            directory = root / mode
            copy_inputs(directory, args.model_xyz.resolve(), args.qnep_model.resolve())
            shutil.copy2(bead_restart, directory / "fixture_initial_restart.xyz")
            shutil.copy2(final_reference, directory / final_reference.name)
            shutil.copy2(sidecar, directory / sidecar.name)
            ja_line = "rpmd_ja off" if mode == "off" else f"rpmd_ja on {final_reference.name}"
            write_run_input(directory / "run.in", [
                "potential nep.txt",
                f"kspace pppm {args.pppm_spacing:.17g}",
                f"time_step {args.time_step:.17g}",
                f"ensemble rpmd {args.beads} {args.temperature:.17g}",
                "read_pimd_restart fixture_initial_restart.xyz",
                ja_line,
                f"compute_hac 1 {correlation_steps} 1 1 0 0",
                "dump_thermo 1",
                f"dump_pimd_restart {args.steps}",
                f"run {args.steps}",
            ])
            run([str(gpumd)], directory)
            output_dirs[mode] = directory

        compare_outputs(output_dirs["off"], output_dirs["on"], args.rtol, args.atol)
        if (output_dirs["off"] / "heat_current_rpmd_ja.out").exists():
            raise AssertionError("rpmd_ja off unexpectedly wrote JA current output")
        check_ja_current_and_hac(output_dirs["on"], args.rtol, args.atol)

    print("qNEP reference generation and JA off/on RPMD acceptance checks passed")


if __name__ == "__main__":
    main()
