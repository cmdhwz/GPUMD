import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _batch_binary():
    candidates = []
    configured = os.environ.get("GPUMD_BATCH")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / "src" / "gpumd_batch",
            ROOT / "build" / "gpumd_batch",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("build gpumd_batch first or set GPUMD_BATCH")


def _make_fake_gpumd(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    batch = bin_dir / "gpumd_batch"
    shutil.copy2(_batch_binary(), batch)

    fake_gpumd = bin_dir / "gpumd"
    fake_gpumd.write_text(
        """#!/bin/sh
set -eu
sample=$(basename "$PWD")
cp run.in seen_run.in
printf '%s\\n' "$sample" >> "$FAKE_ROOT/launched.log"
cp run.in "$FAKE_ROOT/$sample.run.in"
if [ "$sample" = "sample_1" ] && [ -f "$FAKE_ROOT/wait_for_parent_edit" ]; then
  touch "$FAKE_ROOT/sample_1_started"
  while [ ! -f "$FAKE_ROOT/release_sample_1" ]; do sleep 0.01; done
fi
if [ -f create_thermo ]; then
  printf 'fake output\\n' > thermo.out
fi
if [ -f fail ]; then exit 7; fi
""",
        encoding="utf-8",
    )
    fake_gpumd.chmod(fake_gpumd.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = os.environ.copy()
    env["FAKE_ROOT"] = str(tmp_path)
    return batch, env


def _write_run_input(root, text="ensemble nve 300\n"):
    (root / "run.in").write_text(text, encoding="utf-8")


def _make_sample(root, index, *, fail=False, model=True):
    sample = root / f"sample_{index}"
    sample.mkdir()
    if model:
        (sample / "model.xyz").write_text("1\n", encoding="utf-8")
    if fail:
        (sample / "fail").touch()
    return sample


@pytest.mark.skipif(os.name != "posix", reason="gpumd_batch behavior test is Linux/POSIX-only")
def test_batch_uses_one_run_snapshot_and_cleans_temporary_input(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    _write_run_input(root)
    _make_sample(root, 1)
    _make_sample(root, 2)
    (tmp_path / "wait_for_parent_edit").touch()
    batch, env = _make_fake_gpumd(tmp_path)

    process = subprocess.Popen(
        [str(batch), "sample_", "1", "2", "1"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(500):
            if (tmp_path / "sample_1_started").exists():
                break
            time.sleep(0.01)
        else:
            pytest.fail("fake gpumd did not start sample_1")

        (root / "run.in").write_text("ensemble nvt_nhc 300 1 1\n", encoding="utf-8")
        (tmp_path / "release_sample_1").touch()
        stdout, stderr = process.communicate(timeout=5)
    except Exception:
        process.kill()
        process.communicate()
        raise

    assert process.returncode == 0, (stdout, stderr)
    assert (root / "sample_1" / "COMPLETED").is_file()
    assert (root / "sample_2" / "COMPLETED").is_file()
    assert not (root / "sample_1" / "run.in").exists()
    assert not (root / "sample_2" / "run.in").exists()
    assert (tmp_path / "sample_1.run.in").read_text(encoding="utf-8") == "ensemble nve 300\n"
    assert (tmp_path / "sample_2.run.in").read_text(encoding="utf-8") == "ensemble nve 300\n"
    assert (tmp_path / "launched.log").read_text(encoding="utf-8").splitlines() == [
        "sample_1",
        "sample_2",
    ]


@pytest.mark.skipif(os.name != "posix", reason="gpumd_batch behavior test is Linux/POSIX-only")
def test_batch_failure_missing_model_and_completed_sample(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    _write_run_input(root)
    completed = _make_sample(root, 1)
    (completed / "COMPLETED").touch()
    failed = _make_sample(root, 2, fail=True)
    missing = _make_sample(root, 3, model=False)
    batch, env = _make_fake_gpumd(tmp_path)

    result = subprocess.run(
        [str(batch), "sample_", "1", "3", "1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode != 0
    assert (completed / "COMPLETED").is_file()
    assert not (failed / "COMPLETED").exists()
    assert not (failed / "run.in").exists()
    assert not (missing / "COMPLETED").exists()
    assert (tmp_path / "launched.log").read_text(encoding="utf-8").splitlines() == ["sample_2"]


@pytest.mark.skipif(os.name != "posix", reason="gpumd_batch behavior test is Linux/POSIX-only")
@pytest.mark.parametrize("marker_name", ["model.xyz", "run.in"])
def test_batch_rejects_input_file_marker_names(tmp_path, marker_name):
    root = tmp_path / "case"
    root.mkdir()
    _write_run_input(root)
    _make_sample(root, 1)
    batch, env = _make_fake_gpumd(tmp_path)

    result = subprocess.run(
        [str(batch), "sample_", "1", "1", "1", marker_name],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode != 0
    assert not (tmp_path / "launched.log").exists()


@pytest.mark.skipif(os.name != "posix", reason="gpumd_batch behavior test is Linux/POSIX-only")
def test_batch_rejects_nonempty_marker_and_preserves_output_collision(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    _write_run_input(root)
    invalid = _make_sample(root, 1)
    (invalid / "COMPLETED").write_text("not a completion marker", encoding="utf-8")
    collision = _make_sample(root, 2)
    (collision / "create_thermo").touch()
    batch, env = _make_fake_gpumd(tmp_path)

    invalid_result = subprocess.run(
        [str(batch), "sample_", "1", "1", "1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert invalid_result.returncode != 0
    assert not (tmp_path / "launched.log").exists()

    collision_result = subprocess.run(
        [str(batch), "sample_", "2", "2", "1", "thermo.out"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert collision_result.returncode != 0
    assert (collision / "thermo.out").read_text(encoding="utf-8") == "fake output\n"
