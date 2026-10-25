#!/usr/bin/env python3
"""Generate a qNEP v3 reference, then compare the same RPMD case with JA off/on."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
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


def run_expect_failure(command: list[str], cwd: Path, expected: str) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode == 0 or expected not in output:
        raise AssertionError(
            f"expected failure containing {expected!r}, got {result.returncode}:\n{output}"
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


def check_sample_output(directory: Path, model: Path, backend: str, beads: int,
                        temperature: float, steps: int, sample_interval: int,
                        expected_run_ordinal: int = 0,
                        expected_ensemble: list[str] | None = None) -> np.ndarray:
    spool = directory / "train.samples.tmp"
    sidecar_path = directory / "train.samples.tmp.provenance.json"
    data = spool.read_bytes()
    if data[:8] != b"GPJASMP1":
        raise AssertionError("rpmd_ja sample did not write a GPJASMP1 spool")
    version, endian, atoms, spool_beads = struct.unpack_from("<IIii", data, 8)
    if version != 1 or endian != 0x01020304 or spool_beads != beads:
        raise AssertionError("rpmd_ja sample spool has an invalid header")
    offset = 8 + struct.calcsize("<IIii") + struct.calcsize("<d")
    offset += 9 * struct.calcsize("<d") + atoms * struct.calcsize("<d")
    offset += atoms * struct.calcsize("<i")
    frame_width = 1 + 6 * atoms
    frame_count = steps // sample_interval
    if len(data) != offset + frame_count * frame_width * struct.calcsize("<d"):
        raise AssertionError("rpmd_ja sample spool frame count or record size is invalid")
    frames = np.frombuffer(data, dtype="<f8", offset=offset).reshape(frame_count, frame_width)
    if not np.isfinite(frames).all():
        raise AssertionError("rpmd_ja sample spool contains non-finite coordinates or forces")
    expected_steps = np.arange(sample_interval, frame_count * sample_interval + 1, sample_interval)
    np.testing.assert_array_equal(frames[:, 0], expected_steps)

    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    metadata = provenance["metadata"]
    if metadata["physical_force_backend"] != backend:
        raise AssertionError("sample provenance has the wrong force backend")
    if metadata["model_sha256"] != hashlib.sha256(model.read_bytes()).hexdigest():
        raise AssertionError("sample provenance model SHA-256 does not match the active model file")
    if metadata["physical_force_source"] != "MEAN_OF_PHYSICAL_PIMD_BEAD_FORCES":
        raise AssertionError("sample provenance does not identify the physical bead-force average")
    if metadata["number_of_beads"] != beads or metadata["temperature_K"] != temperature:
        raise AssertionError("sample provenance does not match the canonical PIMD input")
    if metadata["active_run_ordinal"] != expected_run_ordinal:
        raise AssertionError("sample provenance identified the wrong run.in segment")
    if (expected_ensemble is not None and
            metadata["thermostat_configuration_tokens"] != [expected_ensemble]):
        raise AssertionError("sample provenance does not identify the active segment thermostat")
    if not metadata["configuration_stable_during_run"]:
        raise AssertionError("sample provenance could not verify a stable run configuration")
    return frames.copy()


def compare_ja_backend_outputs(native: Path, wpe: Path, rtol: float, atol: float) -> None:
    for filename in (
        "heat_current_centroid.out",
        "hac_centroid.out",
        "heat_current_rpmd_ja.out",
        "hac_rpmd_ja.out",
    ):
        np.testing.assert_allclose(
            numeric(wpe / filename), numeric(native / filename),
            rtol=rtol, atol=atol, err_msg=f"WPE/native {filename}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpumd", required=True, type=Path)
    parser.add_argument("--wpe-gpumd", type=Path,
                        help="optional gpumd-wpe executable for same-input JA and bead-force sample checks")
    parser.add_argument(
        "--model-xyz", required=True, type=Path,
        help="fixture structure, which must contain every qNEP model species so no compact copy changes the SHA-256",
    )
    parser.add_argument("--qnep-model", required=True, type=Path)
    parser.add_argument("--kernel-table", required=True, type=Path)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--pppm-spacing", type=float, default=1.0)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=20)
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
    if args.sample_interval <= 0 or args.steps // args.sample_interval < 3:
        parser.error("sample-interval must yield at least three sample frames")
    if not all(math.isfinite(value) and value >= 0.0 for value in (args.rtol, args.atol)):
        parser.error("rtol and atol must be finite and non-negative")
    model_header = args.qnep_model.read_text(encoding="utf-8", errors="replace").splitlines()
    if not model_header or not model_header[0].startswith(
        ("nep4_charge1", "nep4_charge2", "nep4_zbl_charge1", "nep4_zbl_charge2")
    ):
        parser.error("qnep-model must contain a charge1 or charge2 qNEP model")
    model_tokens = model_header[0].split()
    try:
        qnep_type_count = int(model_tokens[1])
        qnep_species = model_tokens[2 : 2 + qnep_type_count]
    except (IndexError, ValueError):
        parser.error("qnep-model has an invalid species header")
    if qnep_type_count < 1 or len(qnep_species) != qnep_type_count:
        parser.error("qnep-model has an invalid species header")
    xyz_lines = args.model_xyz.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(xyz_lines) < 2 or re.search(r"Properties=.*(?:=|:)vel(?:ocity)?(?:[:=]|$)", xyz_lines[1]):
        parser.error("model-xyz must omit velocity properties so the seed run uses the requested fixed seed")
    properties_match = re.search(r"(?:^|\s)Properties=(\S+)", xyz_lines[1], re.IGNORECASE)
    if properties_match is None:
        parser.error("model-xyz must declare its atom species in a Properties field")
    property_fields = properties_match.group(1).split(":")
    if len(property_fields) % 3 != 0:
        parser.error("model-xyz has an invalid Properties field")
    species_offset = 0
    species_column = None
    try:
        for property_index in range(0, len(property_fields), 3):
            property_name = property_fields[property_index].lower()
            property_count = int(property_fields[property_index + 2])
            if property_name == "species":
                species_column = species_offset
            species_offset += property_count
        atom_count = int(xyz_lines[0].strip())
    except ValueError:
        parser.error("model-xyz has an invalid atom count or Properties field")
    if species_column is None or atom_count < 1 or len(xyz_lines) < atom_count + 2:
        parser.error("model-xyz is missing its species column or atom records")
    atom_records = [line.split() for line in xyz_lines[2 : atom_count + 2]]
    if any(len(record) <= species_column for record in atom_records):
        parser.error("model-xyz atom record is missing its species value")
    present_species = {record[species_column] for record in atom_records}
    missing_qnep_species = sorted(set(qnep_species) - present_species)
    if missing_qnep_species:
        parser.error(
            "model-xyz must contain all qNEP model species for exact model SHA-256 verification; "
            f"missing: {', '.join(missing_qnep_species)} (otherwise GPUMD creates a compact model file)"
        )
    gpumd = args.gpumd.resolve()
    if not gpumd.is_file():
        parser.error(f"GPUMD executable does not exist: {gpumd}")
    engines = {"GPUMD_NATIVE": gpumd}
    if args.wpe_gpumd is not None:
        wpe_gpumd = args.wpe_gpumd.resolve()
        if not wpe_gpumd.is_file():
            parser.error(f"WPE executable does not exist: {wpe_gpumd}")
        engines["WPE"] = wpe_gpumd

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
        ja_outputs = {}
        for backend, executable in engines.items():
            output_dirs = {}
            for mode in ("off", "on"):
                directory = root / f"{backend.lower()}-{mode}"
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
                run([str(executable)], directory)
                output_dirs[mode] = directory

            compare_outputs(output_dirs["off"], output_dirs["on"], args.rtol, args.atol)
            if (output_dirs["off"] / "heat_current_rpmd_ja.out").exists():
                raise AssertionError(f"{backend} rpmd_ja off unexpectedly wrote JA current output")
            check_ja_current_and_hac(output_dirs["on"], args.rtol, args.atol)
            ja_outputs[backend] = output_dirs["on"]

        if "WPE" in ja_outputs:
            compare_ja_backend_outputs(
                ja_outputs["GPUMD_NATIVE"], ja_outputs["WPE"], args.rtol, args.atol)

        sample_frames = {}
        for backend, executable in engines.items():
            directory = root / f"{backend.lower()}-sample"
            copy_inputs(directory, args.model_xyz.resolve(), args.qnep_model.resolve())
            shutil.copy2(bead_restart, directory / "fixture_initial_restart.xyz")
            write_run_input(directory / "run.in", [
                "potential nep.txt",
                f"kspace pppm {args.pppm_spacing:.17g}",
                # Reset the process RNG before PIMD initializes CURAND; the restart then restores bead velocities.
                f"velocity {args.temperature:.17g} seed {args.seed}",
                f"time_step {args.time_step:.17g}",
                f"ensemble pimd {args.beads} {args.temperature:.17g} {args.temperature:.17g} 100",
                "read_pimd_restart fixture_initial_restart.xyz",
                f"rpmd_ja sample train {args.sample_interval}",
                f"run {args.steps}",
            ])
            run([str(executable)], directory)
            sample_frames[backend] = check_sample_output(
                directory, directory / "nep.txt", backend, args.beads,
                args.temperature, args.steps, args.sample_interval,
                expected_ensemble=["ensemble", "pimd", str(args.beads),
                                   f"{args.temperature:.17g}", f"{args.temperature:.17g}", "100"])

        if "WPE" in sample_frames:
            np.testing.assert_array_equal(sample_frames["WPE"][:, 0], sample_frames["GPUMD_NATIVE"][:, 0])
            np.testing.assert_allclose(
                sample_frames["WPE"][:, 1:], sample_frames["GPUMD_NATIVE"][:, 1:],
                rtol=args.rtol, atol=args.atol,
                err_msg="WPE/native sampled centroid positions and mean physical bead forces",
            )

            first_pimd_directory = root / "wpe-sample-first-pimd"
            copy_inputs(first_pimd_directory, args.model_xyz.resolve(), args.qnep_model.resolve())
            write_run_input(first_pimd_directory / "run.in", [
                "potential nep.txt",
                f"kspace pppm {args.pppm_spacing:.17g}",
                f"velocity {args.temperature:.17g} seed {args.seed}",
                f"time_step {args.time_step:.17g}",
                f"ensemble pimd {args.beads} {args.temperature:.17g} {args.temperature:.17g} 100",
                f"rpmd_ja sample train {args.sample_interval}",
                f"run {args.steps}",
            ])
            run([str(engines["WPE"])], first_pimd_directory)
            check_sample_output(
                first_pimd_directory, first_pimd_directory / "nep.txt", "WPE", args.beads,
                args.temperature, args.steps, args.sample_interval,
                expected_ensemble=["ensemble", "pimd", str(args.beads),
                                   f"{args.temperature:.17g}", f"{args.temperature:.17g}", "100"])

            for case, balance_steps in (
                ("same-run-length", args.steps),
                ("different-run-length", args.steps + 1),
            ):
                for backend, executable in engines.items():
                    directory = root / f"{backend.lower()}-sample-after-prelude-{case}"
                    copy_inputs(directory, args.model_xyz.resolve(), args.qnep_model.resolve())
                    write_run_input(directory / "run.in", [
                        "potential nep.txt",
                        f"kspace pppm {args.pppm_spacing:.17g}",
                        f"velocity {args.temperature:.17g} seed {args.seed}",
                        f"time_step {args.time_step:.17g}",
                        f"ensemble pimd {args.beads} {args.temperature:.17g} {args.temperature:.17g} 100",
                        f"dump_pimd_restart {balance_steps}",
                        f"run {balance_steps}",
                        f"ensemble pimd {args.beads} {args.temperature:.17g} {args.temperature:.17g} 200",
                        "read_pimd_restart restart_beads.xyz",
                        f"rpmd_ja sample train {args.sample_interval}",
                        f"run {args.steps}",
                    ])
                    run([str(executable)], directory)
                    check_sample_output(
                        directory, directory / "nep.txt", backend, args.beads,
                        args.temperature, args.steps, args.sample_interval,
                        expected_run_ordinal=1,
                        expected_ensemble=["ensemble", "pimd", str(args.beads),
                                          f"{args.temperature:.17g}", f"{args.temperature:.17g}", "200"])

            for case, rpmd_ja_line, extra_action, expected in (
                (
                    "wpe-fit-rejected",
                    f"rpmd_ja fit fit-output.bin {args.sample_interval} 5 1e-8 0.15 1e-4 kernel.tbl",
                    None,
                    "WPE supports rpmd_ja sample only",
                ),
                (
                    "wpe-sample-drive-rejected",
                    f"rpmd_ja sample driven {args.sample_interval}",
                    "compute_hnemd 1 1e-5 0 0",
                    "rpmd_ja sample does not support external drive/action compute_hnemd",
                ),
            ):
                directory = root / case
                copy_inputs(directory, args.model_xyz.resolve(), args.qnep_model.resolve())
                shutil.copy2(bead_restart, directory / "fixture_initial_restart.xyz")
                lines = [
                    "potential nep.txt",
                    f"kspace pppm {args.pppm_spacing:.17g}",
                    f"velocity {args.temperature:.17g} seed {args.seed}",
                    f"time_step {args.time_step:.17g}",
                    f"ensemble pimd {args.beads} {args.temperature:.17g} {args.temperature:.17g} 100",
                    "read_pimd_restart fixture_initial_restart.xyz",
                    rpmd_ja_line,
                ]
                if extra_action is not None:
                    lines.append(extra_action)
                lines.append(f"run {args.steps}")
                write_run_input(directory / "run.in", lines)
                run_expect_failure([str(engines["WPE"])], directory, expected)

    print("qNEP JA off/on checks passed; native reference generation and sample provenance verified")


if __name__ == "__main__":
    main()
