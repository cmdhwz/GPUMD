#!/usr/bin/env python3
"""Compare native qNEP centroid HAC against a baseline GPUMD executable."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


OUTPUTS = ("heat_current_centroid.out", "hac_centroid.out", "thermo.out")
INITIAL_RESTART = "fixture_initial_restart.xyz"


def run(executable: Path, case_dir: Path, work_dir: Path) -> None:
    shutil.copytree(case_dir, work_dir)
    run_input = work_dir / "run.in"
    text = run_input.read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^([ \t]*read_pimd_restart[ \t]+)\S+",
        rf"\g<1>{INITIAL_RESTART}",
        text,
    )
    if count != 1:
        raise AssertionError("expected exactly one read_pimd_restart in run.in")
    run_input.write_text(text, encoding="utf-8")
    shutil.copy2(work_dir / "restart_beads.xyz", work_dir / INITIAL_RESTART)
    for filename in (*OUTPUTS, "restart_beads.xyz"):
        (work_dir / filename).unlink(missing_ok=True)
    result = subprocess.run(
        [str(executable)], cwd=work_dir, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"{executable} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    if not (work_dir / "restart_beads.xyz").is_file():
        raise AssertionError("run did not produce a fresh final restart_beads.xyz")


def load_numeric(path: Path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if not np.isfinite(data).all():
        raise AssertionError(f"non-finite output in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-gpumd", required=True, type=Path)
    parser.add_argument("--candidate-gpumd", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=2.0e-8)
    parser.add_argument("--atol", type=float, default=2.0e-9)
    args = parser.parse_args()

    for executable in (args.baseline_gpumd, args.candidate_gpumd):
        if not executable.is_file():
            parser.error(f"executable does not exist: {executable}")
    for filename in ("run.in", "model.xyz", "nep.txt", "restart_beads.xyz"):
        if not (args.case_dir / filename).is_file():
            parser.error(f"case directory is missing {filename}")
    commands = [
        line.split("#", 1)[0].split()
        for line in (args.case_dir / "run.in").read_text(encoding="utf-8").splitlines()
    ]
    commands = [tokens for tokens in commands if tokens]
    rpmd_ja = [tokens for tokens in commands if tokens[0] == "rpmd_ja"]
    if rpmd_ja != [["rpmd_ja", "off"]]:
        parser.error("case run.in must contain only `rpmd_ja off`")
    if any(tokens[0] == "hac_current" for tokens in commands):
        parser.error("case run.in must not select a hac_current mode such as qnep_full_a")
    potentials = [tokens for tokens in commands if tokens[0] == "potential"]
    if potentials != [["potential", "nep.txt"]]:
        parser.error("case run.in must use exactly `potential nep.txt`")
    restart_reads = [tokens for tokens in commands if tokens[0] == "read_pimd_restart"]
    if len(restart_reads) != 1 or len(restart_reads[0]) != 2:
        parser.error("case run.in must read exactly one bead restart file")
    centroid_hac = [
        tokens
        for tokens in commands
        if tokens[0] == "compute_hac" and len(tokens) >= 5 and tokens[4] == "1"
    ]
    if len(centroid_hac) != 1 or sum(tokens[0] == "compute_hac" for tokens in commands) != 1:
        parser.error("case run.in must have exactly one direct centroid compute_hac")
    if sum(tokens[0] == "dump_pimd_restart" for tokens in commands) != 1:
        parser.error("case run.in must have exactly one dump_pimd_restart")
    if any(
        len(tokens) >= 6 and tokens[5] != "0"
        or len(tokens) >= 7 and tokens[6] != "0"
        for tokens in centroid_hac
    ):
        parser.error("case centroid HAC must use split=0 and deferred=0")
    run_commands = [tokens for tokens in commands if tokens[0] == "run"]
    runs = [int(tokens[1]) for tokens in run_commands if len(tokens) == 2]
    dump_intervals = [
        int(tokens[1])
        for tokens in commands
        if tokens[0] == "dump_pimd_restart" and len(tokens) >= 2
    ]
    time_steps = [float(tokens[1]) for tokens in commands if tokens[0] == "time_step" and len(tokens) >= 2]
    ensembles = [tokens for tokens in commands if tokens[0] == "ensemble"]
    if len(ensembles) != 1 or len(ensembles[0]) < 2 or ensembles[0][1] != "rpmd":
        parser.error("case run.in must use exactly one standard `ensemble rpmd` production segment")
    if len(run_commands) != 1 or len(runs) != 1 or runs[0] <= 0:
        parser.error("case run.in must have exactly one positive production `run` command")
    if (
        len(dump_intervals) != 1
        or dump_intervals[0] <= 0
        or len(time_steps) != 1
        or time_steps[0] <= 0.0
    ):
        parser.error("case run.in must have a positive time step and one restart dump interval")
    if runs[0] < dump_intervals[0] or runs[0] % dump_intervals[0] != 0:
        parser.error("the single production run must reach a final dump_pimd_restart interval")
    if not (args.case_dir / "nep.txt").read_text(encoding="utf-8").splitlines()[0].startswith(
        ("nep4_charge1", "nep4_charge2", "nep4_zbl_charge1", "nep4_zbl_charge2")
    ):
        parser.error("nep.txt must contain a qNEP charge1 or charge2 model")

    with tempfile.TemporaryDirectory(prefix="gpumd-qnep-centroid-compare-") as temporary:
        root = Path(temporary)
        baseline_dir = root / "baseline"
        candidate_dir = root / "candidate"
        run(args.baseline_gpumd.resolve(), args.case_dir.resolve(), baseline_dir)
        run(args.candidate_gpumd.resolve(), args.case_dir.resolve(), candidate_dir)

        for filename in ("heat_current_centroid.out", "hac_centroid.out"):
            baseline = load_numeric(baseline_dir / filename)
            candidate = load_numeric(candidate_dir / filename)
            np.testing.assert_allclose(
                candidate, baseline, rtol=args.rtol, atol=args.atol, err_msg=filename
            )
        baseline_thermo = (baseline_dir / "thermo.out").exists()
        candidate_thermo = (candidate_dir / "thermo.out").exists()
        if baseline_thermo != candidate_thermo:
            raise AssertionError("thermo.out was produced by only one executable")
        if baseline_thermo:
            np.testing.assert_allclose(
                load_numeric(candidate_dir / "thermo.out"),
                load_numeric(baseline_dir / "thermo.out"),
                rtol=args.rtol,
                atol=args.atol,
                err_msg="thermo.out",
            )
        baseline_restart = (baseline_dir / "restart_beads.xyz").read_bytes()
        candidate_restart = (candidate_dir / "restart_beads.xyz").read_bytes()
        if baseline_restart != candidate_restart:
            raise AssertionError("final bead restart differs between baseline and candidate")
    print("native qNEP centroid HAC and RPMD state match the baseline executable")


if __name__ == "__main__":
    main()
