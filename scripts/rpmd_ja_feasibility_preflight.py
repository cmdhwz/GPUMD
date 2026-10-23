#!/usr/bin/env python3
"""Fail-closed launcher for the bounded RPMD-JA CUDA feasibility run."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PREFLIGHT_LOCKS = []

_PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    "rpmd_ja_sampling_provenance", Path(__file__).with_name("rpmd_ja_sampling_provenance.py"))
if _PROVENANCE_SPEC is None or _PROVENANCE_SPEC.loader is None:
    raise RuntimeError("cannot load the RPMD-JA sample provenance validator")
_PROVENANCE = importlib.util.module_from_spec(_PROVENANCE_SPEC)
_PROVENANCE_SPEC.loader.exec_module(_PROVENANCE)

OFFLINE_SETUP_COMMANDS = {"potential", "pppm_mesh_spacing", "pimd_fix_com", "kspace"}
CURVATURE_PRIMARY_SEED = "8d12e5b7c3a964f1"
CURVATURE_INDEPENDENT_SEED = "4f1bbcdc6762c9a5"
CURVATURE_PROBE_CAP = 128
CURVATURE_DIRECTIONAL_TOLERANCE = 0.10
CANDIDATE_COST_STOP_RULE = "MAX_COMPLETE_OR_PARTIAL_LOWER_BOUND_SECONDS_V1"
PERFORMANCE_STOP_RULE = "THREE_CURVATURE_BUILDS_GT_ONE_MAX_FULL_GRADIENT_LT_0.001_RELATIVE_OBJECTIVE_IMPROVEMENT_SAME_TAU_V1"
PERFORMANCE_STOP_OBJECTIVE_NORMALIZER = "MAX_1_ABS_WINDOW_START_OBJECTIVE"
PERFORMANCE_STOP_GRADIENT_COST_RULE = "MAX_MEASURED_FULL_EXACT_GRADIENT_SECONDS_WITHIN_SAME_TAU_WINDOW"
PERFORMANCE_STOP_CURVATURE_WINDOW = 3
PERFORMANCE_STOP_MIN_RELATIVE_IMPROVEMENT = 0.001
PERFORMANCE_STOP_CURVATURE_TO_GRADIENT_COST_RATIO = 1.0
A0_REUSE_MAGIC = "RPJAA0REUSE2"
A0_REUSE_PACKAGE_MAGIC = b"RPJAA0PACKAGE1"
A0_REUSE_QUALIFICATION = "CONVERGED_A0_WITH_COMPLETE_RESPONSE_V2"
LAMBDA_FACTORS = (1.0e-2, 1.0e-4, 1.0e-6)
MAX_DEVELOPMENT_BLOCKS = 8
MIN_DEVELOPMENT_BLOCKS = 4
STATISTICS_STORAGE_BUDGET_BYTES = 512 * 1024 * 1024
WALL_BUDGET_SECONDS = 8 * 60 * 60
MAX_EXACT_GRADIENTS = 240
MAX_A_BRANCH_GRADIENTS = 8
MAX_FIT_FAMILY_GRADIENTS = 112
MAX_FIT_FAMILY_WALL_SECONDS = 0.5 * (WALL_BUDGET_SECONDS * (1.0 - 0.25))
FROZEN_CUDA_QP_FIXTURE_SHA256 = {
    "ja_reference_resample.bin.qp_state.txt": "e2dc03d21aa1ddc16a744f5f42edd0f2973f657d0b47892fd47976870559a895",
    "ja_reference_replay_v1.bin.qp_state.txt": "011250eb6624846818b5ca92780d33409d1e63547b067eb94b86d80d126168d7",
    "ja_reference_replay_v2.bin.qp_state.txt": "6db931e9782a43d0393428f07b5c17679f8ae72c492d2670140cb00d95125ff9",
    "ja_reference_replay_v4.bin.qp_state.txt": "0c91d4d2e120bc8073a1bbddd438db4c21b93ca8551658ee8823b018cbb4dc12",
    "ja_reference_outer128_766cuts_stationarity_v1.qp_fixture.txt": "43aac0ebde38101937ec184722f48c64b01523706347bc93d465ae6a6655b9d6",
}


def fail(message: str) -> None:
    raise SystemExit(f"RPMD-JA preflight failed: {message}")


def parse_quoted_fields(path: Path, magic: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline=None) as stream:
        if stream.readline().rstrip("\r\n") != magic:
            fail(f"{path} must start with {magic}")
        for line_number, line in enumerate(stream, 2):
            if not line.strip():
                continue
            try:
                row = shlex.split(line, posix=True)
            except ValueError as error:
                fail(f"malformed quoted row at {path}:{line_number}: {error}")
            if len(row) != 2 or row[0] in fields:
                fail(f"duplicate or malformed field at {path}:{line_number}")
            fields[row[0]] = row[1]
    return fields


def required(fields: dict[str, str], name: str) -> str:
    value = fields.get(name, "")
    if not value:
        fail(f"manifest field {name!r} is required")
    return value


def resolved(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def a0_reuse_contract_sha256(fields: dict[str, str]) -> str:
    canonical = "".join(f"{key}\n{value}\n" for key, value in sorted(fields.items())
                        if key != "job_id" and not key.startswith("a0_reuse_") and
                        not key.endswith("_path"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_a0_reuse_fields(contents: bytes, magic: str) -> dict[str, str]:
    try:
        lines = contents.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("A0 reuse package is not UTF-8") from error
    if not lines or lines[0] != magic:
        raise ValueError(f"A0 reuse package must start with {magic}")
    fields: dict[str, str] = {}
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        try:
            row = shlex.split(line, posix=True)
        except ValueError as error:
            raise ValueError(f"malformed A0 reuse field at line {number}") from error
        if len(row) != 2 or row[0] in fields:
            raise ValueError(f"duplicate or malformed A0 reuse field at line {number}")
        fields[row[0]] = row[1]
    return fields


def _parse_a0_reuse_package(contents: bytes) -> tuple[bytes, bytes, bytes]:
    cursor = 0
    def line() -> bytes:
        nonlocal cursor
        end = contents.find(b"\n", cursor)
        if end < 0:
            raise ValueError("truncated A0 reuse package header")
        value = contents[cursor:end]
        cursor = end + 1
        return value.removesuffix(b"\r")
    if line() != A0_REUSE_PACKAGE_MAGIC:
        raise ValueError("A0 reuse package has an unsupported version")
    def length(key: bytes) -> int:
        row = line()
        prefix = key + b" "
        if not row.startswith(prefix):
            raise ValueError("malformed A0 reuse package length")
        value = row[len(prefix):]
        if not value or any(byte < ord("0") or byte > ord("9") for byte in value):
            raise ValueError("malformed A0 reuse package length")
        parsed = int(value)
        if parsed > len(contents) - cursor:
            raise ValueError("A0 reuse package length is out of range")
        return parsed
    manifest_size = length(b"manifest_bytes")
    state_size = length(b"state_bytes")
    record_size = length(b"record_bytes")
    if cursor + manifest_size + state_size + record_size != len(contents):
        raise ValueError("A0 reuse package payload lengths do not match the file")
    manifest_bytes = contents[cursor:cursor + manifest_size]
    cursor += manifest_size
    state_bytes = contents[cursor:cursor + state_size]
    cursor += state_size
    record_bytes = contents[cursor:cursor + record_size]
    return manifest_bytes, state_bytes, record_bytes


def load_a0_reuse_package(manifest: dict[str, str], run_directory: Path,
                          current_manifest_bytes: bytes, current_machine: str,
                          current_job_id: str) -> dict[str, str]:
    record_path_value = manifest.get("a0_reuse_package_path", "")
    record_sha256 = manifest.get("a0_reuse_package_sha256", "").lower()
    if not record_path_value and not record_sha256:
        return {"a0_reuse_status": "NOT_REQUESTED"}
    if not record_path_value or len(record_sha256) != 64 or any(c not in "0123456789abcdef" for c in record_sha256):
        raise ValueError("A0 reuse requires both a package path and a SHA-256 digest")
    record_path = resolved(record_path_value, run_directory)
    package_bytes = record_path.read_bytes()
    if hashlib.sha256(package_bytes).hexdigest() != record_sha256:
        raise ValueError("A0 reuse package SHA-256 does not match the manifest")
    source_manifest_bytes, state_bytes, record_bytes = _parse_a0_reuse_package(package_bytes)
    record = _parse_a0_reuse_fields(record_bytes, A0_REUSE_MAGIC)
    expected_keys = {"source_manifest_sha256", "contract_sha256", "state_sha256", "source_machine_fingerprint",
                     "source_job_id", "historical_origin_job_id", "qualification_status",
                     "source_optimizer_seconds", "source_fit_elapsed_seconds",
                     "development_response_complete", "development_response_pass"}
    if set(record) != expected_keys:
        raise ValueError("A0 reuse record fields differ from RPJAA0REUSE2")
    source_manifest = _parse_a0_reuse_fields(source_manifest_bytes, "RPJAFEAS9")
    state_fields = {}
    for line in state_bytes.decode("utf-8", errors="strict").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            state_fields[parts[0]] = parts[1]
    def is_sha256(value: str) -> bool:
        return len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if any(not is_sha256(record[key]) for key in ("source_manifest_sha256", "contract_sha256", "state_sha256",
                                                   "source_machine_fingerprint")):
        raise ValueError("A0 reuse record contains an invalid SHA-256")
    try:
        optimizer_seconds = float(record["source_optimizer_seconds"])
        fit_elapsed_seconds = float(record["source_fit_elapsed_seconds"])
        import_validation_seconds = float(record["current_import_validation_seconds"])
        response_seconds = float(record["current_response_seconds"])
    except ValueError as error:
        raise ValueError("A0 reuse source/current timings are invalid") from error
    if not math.isfinite(optimizer_seconds) or optimizer_seconds < 0 or \
       not math.isfinite(fit_elapsed_seconds) or fit_elapsed_seconds < 0 or \
       not math.isfinite(import_validation_seconds) or import_validation_seconds < 0 or \
       not math.isfinite(response_seconds) or response_seconds < 0:
        raise ValueError("A0 reuse source/current timings are invalid")
    if record["qualification_status"] != A0_REUSE_QUALIFICATION or \
       not record["historical_origin_job_id"] or \
       record["development_response_complete"] != "1" or record["development_response_pass"] not in {"0", "1"}:
        raise ValueError("A0 reuse source was not a converged, response-complete A0")
    if state_fields.get("state_format_version") != "4" or state_fields.get("status", "") not in {
        "COVARIANCE_OPTIMIZER_CONVERGED", "COVARIANCE_FIXED_TAU_CONVERGED"
    }:
        raise ValueError("A0 reuse state is not a converged A0 fit in state format 4")
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    current_manifest_sha256 = hashlib.sha256(current_manifest_bytes).hexdigest()
    contract = a0_reuse_contract_sha256(source_manifest)
    if source_manifest_sha256 != record["source_manifest_sha256"] or \
       hashlib.sha256(state_bytes).hexdigest() != record["state_sha256"] or \
       contract != record["contract_sha256"] or \
       a0_reuse_contract_sha256(manifest) != record["contract_sha256"] or \
       current_manifest_sha256 == source_manifest_sha256:
        raise ValueError("A0 reuse package does not match the current scientific and input contract")
    if record["source_machine_fingerprint"] != current_machine or not record["source_job_id"] or \
       not record["historical_origin_job_id"] or record["source_job_id"] == current_job_id:
        raise ValueError("A0 reuse package machine or source job identity does not match")
    return {"a0_reuse_status": "VERIFIED", "a0_reuse_package_path": record_path_value,
            "a0_reuse_resolved_package_path": str(record_path),
            "a0_reuse_package_sha256": record_sha256,
            "a0_reuse_source_manifest_sha256": source_manifest_sha256,
            "a0_reuse_contract_sha256": contract,
            "a0_reuse_state_sha256": record["state_sha256"],
            "a0_reuse_source_machine_fingerprint": record["source_machine_fingerprint"],
            "a0_reuse_source_job_id": record["source_job_id"],
            "a0_reuse_historical_origin_job_id": record["historical_origin_job_id"],
            "a0_reuse_source_optimizer_seconds": record["source_optimizer_seconds"],
            "a0_reuse_source_fit_elapsed_seconds": record["source_fit_elapsed_seconds"],
            "a0_reuse_current_import_validation_seconds": record["current_import_validation_seconds"],
            "a0_reuse_current_response_seconds": record["current_response_seconds"],
            "a0_reuse_development_response_pass": record["development_response_pass"]}


def inspect_resume_checkpoint(path: Path, expected_identity_sha256: str | None = None) -> dict[str, str]:
    try:
        contents = path.read_bytes()
    except OSError as error:
        fail(f"cannot read resume checkpoint {path}: {error}")
    checksum_row = contents.rfind(b"sha256 ")
    if checksum_row <= 0 or not contents.endswith(b"\n"):
        fail("resume checkpoint is truncated or missing its checksum")
    payload = contents[:checksum_row]
    checksum_line = contents[checksum_row:].decode("ascii", errors="replace").splitlines()
    if len(checksum_line) != 1 or len(checksum_line[0].split()) != 2 or \
       checksum_line[0].split()[0] != "sha256" or \
       hashlib.sha256(payload).hexdigest() != checksum_line[0].split()[1]:
        fail("resume checkpoint checksum mismatch")
    try:
        rows = payload.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError:
        fail("resume checkpoint is not valid ASCII")
    if not rows or rows[0] != "RPJAOUTER5":
        fail("resume checkpoint has an unsupported format")
    fields: dict[str, str] = {}
    required_fields = {"job_id", "identity_sha256", "branch", "family", "stage", "selection_state",
                       "frozen_candidate_branch", "frozen_candidate_index", "frozen_candidate_lambda",
                       "frozen_candidate_fit_state_sha256", "frozen_candidate_identity_sha256",
                       "development_response_sha256", "blind_response_sha256",
                       "publication_stage_output_sha256", "publication_stage_sidecar_sha256",
                       "published_output_sha256", "published_sidecar_sha256", "published_fit_sha256",
                       "published_trace_sha256", "common_probe_set_frozen", "common_probe_count", "candidate_count",
                       "common_probe_stage", "next_candidate_index", "active_phase", "active_candidate_started",
                       "elapsed_seconds", "operation_in_progress", "operation_start_epoch_seconds",
                       "exact_gradients", "cg_witness_attempts", "max_fit_family_gradients",
                       "max_fit_family_wall_seconds", "training_statistics_frozen",
                       "training_statistics_policy_version", "training_frame_count", "development_frame_count",
                       "training_block_length", "training_adjacent_block_length", "training_covered_frame_count",
                       "training_discarded_tail_frames", "training_channel_count"}
    required_fields.add("active_candidate_elapsed_seconds")
    training_channels: list[str] = []
    common_probes: list[tuple[str, str, list[float]]] = []
    candidate_records: list[dict[str, object]] = []

    def cpp_quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def frozen_candidate_identity(fields: dict[str, str], candidate: dict[str, object]) -> str:
        payload = (fields["identity_sha256"] + "\n" + str(candidate["branch"]) + "\n" +
                   str(candidate["index"]) + "\n" + format(float(candidate["lambda"]), ".17g") + "\n" +
                   str(candidate["fit_sha256"]) + "\n")
        payload += "".join(format(value, ".17g") + " " for value in candidate["theta"]) + "\n"
        payload += fields["common_probe_stage"] + "\n"
        for source, origin, vector in common_probes:
            payload += cpp_quote(source) + " " + cpp_quote(origin) + " "
            payload += "".join(format(value, ".17g") + " " for value in vector) + "\n"
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def parse_vector(row: list[str], position: int, label: str) -> tuple[list[float], int]:
        try:
            count = int(row[position])
            if count < 0 or count > 1_000_000 or position + 1 + count > len(row):
                raise ValueError
            values = [float(value) for value in row[position + 1:position + 1 + count]]
            if any(not math.isfinite(value) for value in values):
                raise ValueError
        except (ValueError, IndexError):
            fail(f"resume checkpoint has an invalid {label} vector")
        return values, position + 1 + count

    for line_number, line in enumerate(rows[1:], 2):
        try:
            row = shlex.split(line, posix=True)
        except ValueError as error:
            fail(f"malformed resume checkpoint row {line_number}: {error}")
        if row and row[0] == "training_channel":
            if len(row) != 2 or not row[1]:
                fail(f"malformed frozen training channel at resume checkpoint row {line_number}")
            training_channels.append(row[1])
        elif row and row[0] == "probe":
            if len(row) < 4 or not row[1]:
                fail(f"malformed frozen probe at resume checkpoint row {line_number}")
            vector, end = parse_vector(row, 3, "frozen probe")
            if end != len(row) or not vector:
                fail(f"malformed frozen probe vector at resume checkpoint row {line_number}")
            try:
                origin_values = [float(value) for value in row[2].split(",")]
                if len(origin_values) != 3 or not math.isfinite(origin_values[2]) or origin_values[2] <= 0:
                    raise ValueError
            except ValueError:
                fail(f"invalid frozen probe origin at resume checkpoint row {line_number}")
            common_probes.append((row[1], row[2], vector))
        elif row and row[0] == "candidate":
            if len(row) < 17:
                fail(f"malformed candidate at resume checkpoint row {line_number}")
            try:
                record = {"branch": row[1], "fit_sha256": row[2], "score_generation": row[3],
                          "index": int(row[4]), "lambda": float(row[5]), "fit_complete": int(row[6]),
                          "score_complete": int(row[7]), "response_replay_required": int(row[8]),
                          "converged": int(row[9]), "development_pass": int(row[10]),
                          "b_gate_pass": int(row[11]), "objective": float(row[12]), "uncertainty": float(row[13])}
                theta, position = parse_vector(row, 14, "candidate theta")
                block_b, position = parse_vector(row, position, "candidate B score")
                block_2b, position = parse_vector(row, position, "candidate 2B score")
                if position != len(row) or any(value not in {0, 1} for value in
                   (record["fit_complete"], record["score_complete"], record["response_replay_required"],
                    record["converged"], record["development_pass"], record["b_gate_pass"])) or \
                   not math.isfinite(record["lambda"]) or not math.isfinite(record["objective"]) or \
                   not math.isfinite(record["uncertainty"]):
                    raise ValueError
                record["theta"], record["block_b"], record["block_2b"] = theta, block_b, block_2b
            except (ValueError, IndexError):
                fail(f"invalid candidate at resume checkpoint row {line_number}")
            candidate_records.append(record)
        elif row and row[0] in required_fields:
            if len(row) != 2 or row[0] in fields:
                fail(f"duplicate or malformed resume checkpoint field {row[0]!r}")
            fields[row[0]] = row[1]
    missing = required_fields - fields.keys()
    if missing:
        fail("resume checkpoint is missing identity fields: " + ", ".join(sorted(missing)))
    try:
        if int(fields["common_probe_count"]) != len(common_probes) or \
           int(fields["candidate_count"]) != len(candidate_records):
            raise ValueError
        if fields["common_probe_set_frozen"] not in {"0", "1"}:
            raise ValueError
        if int(fields["common_probe_count"]) < 0 or int(fields["candidate_count"]) < 0:
            raise ValueError
        if len({source for source, _, _ in common_probes}) != len(common_probes):
            raise ValueError
    except ValueError:
        fail("resume checkpoint has inconsistent frozen probe or candidate rows")
    if not fields["job_id"] or len(fields["identity_sha256"]) != 64 or \
       any(character not in "0123456789abcdef" for character in fields["identity_sha256"]):
        fail("resume checkpoint has an invalid job or run identity")
    if fields["branch"] not in {"A0", "A1", "B0", "B1", "B2", "C0", "C1", "C2"} or \
       fields["family"] not in {"M0", "M1"}:
        fail("resume checkpoint has an invalid controller branch")
    if fields["stage"] not in {"A0", "A1", "B", "C", "FREEZE"}:
        fail("resume checkpoint has an invalid controller stage")
    branch = fields["branch"]
    expected_cursor = 0 if branch in {"A0", "A1"} else int(branch[1])
    try:
        next_candidate_index = int(fields["next_candidate_index"])
    except ValueError:
        fail("resume checkpoint has an invalid candidate cursor")
    cursor_is_consistent = next_candidate_index == expected_cursor and (
        (fields["stage"] == "A0" and branch == "A0" and fields["family"] == "M0") or
        (fields["stage"] == "A1" and branch == "A1" and fields["family"] == "M0") or
        (fields["stage"] == "B" and branch.startswith("B") and fields["family"] == "M0") or
        (fields["stage"] == "C" and branch.startswith("C") and fields["family"] == "M1") or
        (fields["stage"] == "FREEZE" and
         ((branch == "B2" and fields["family"] == "M0") or (branch == "C2" and fields["family"] == "M1"))))
    if not cursor_is_consistent:
        fail("resume checkpoint stage, family, branch, and candidate cursor disagree")
    if fields["common_probe_stage"] not in {"NONE", "B_GATE", "FINAL_M0_M1"}:
        fail("resume checkpoint has an invalid frozen common-probe stage")
    selection_state = fields["selection_state"]
    if selection_state not in {"NOT_FROZEN", "FROZEN", "BLIND_OPENED", "BLIND_PASSED", "PUBLISHING", "PUBLISHED"}:
        fail("resume checkpoint has an unsupported frozen-selection state")
    hash_fields = ("frozen_candidate_fit_state_sha256", "frozen_candidate_identity_sha256",
                   "development_response_sha256", "blind_response_sha256", "publication_stage_output_sha256",
                   "publication_stage_sidecar_sha256", "published_output_sha256", "published_sidecar_sha256",
                   "published_fit_sha256", "published_trace_sha256")
    for name in hash_fields:
        value = fields[name]
        if value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
            fail(f"resume checkpoint has an invalid {name}")
    if selection_state == "NOT_FROZEN":
        if any(fields[name] for name in hash_fields) or fields["frozen_candidate_branch"] or \
           fields["frozen_candidate_index"] != "0" or float(fields["frozen_candidate_lambda"]) != 0.0 or \
           fields["common_probe_stage"] == "FINAL_M0_M1" or fields["common_probe_set_frozen"] == "1":
            fail("unfrozen checkpoint contains frozen selection state")
    else:
        frozen_branch = fields["frozen_candidate_branch"]
        if fields["stage"] != "FREEZE" or fields["common_probe_stage"] != "FINAL_M0_M1" or \
           fields["common_probe_set_frozen"] != "1" or frozen_branch not in {"B0", "B1", "B2", "C0", "C1", "C2"}:
            fail("frozen selection checkpoint has incomplete final probe or candidate identity")
        try:
            frozen_index = int(fields["frozen_candidate_index"])
            frozen_lambda = float(fields["frozen_candidate_lambda"])
            if frozen_index != int(frozen_branch[1]) or not math.isfinite(frozen_lambda) or frozen_lambda < 0:
                raise ValueError
        except ValueError:
            fail("frozen selection checkpoint has an invalid candidate index or lambda")
        candidates = [candidate for candidate in candidate_records if candidate["branch"] == frozen_branch]
        if len(candidates) != 1:
            fail("frozen selection checkpoint does not contain exactly one selected candidate")
        selected = candidates[0]
        if selected["index"] != frozen_index or selected["lambda"] != frozen_lambda or \
           selected["fit_sha256"] != fields["frozen_candidate_fit_state_sha256"] or \
           selected["fit_complete"] != 1 or selected["converged"] != 1 or \
           selected["score_complete"] != 1 or selected["score_generation"] != "FINAL_M0_M1" or \
           selected["development_pass"] != 1 or selected["response_replay_required"] != 1 or \
           len(fields["frozen_candidate_fit_state_sha256"]) != 64 or \
           len(fields["frozen_candidate_identity_sha256"]) != 64 or \
           len(fields["development_response_sha256"]) != 64:
            fail("frozen candidate state or development response is incomplete")
        if frozen_candidate_identity(fields, selected) != fields["frozen_candidate_identity_sha256"]:
            fail("frozen candidate identity does not match its candidate theta and common probes")
        if selection_state in {"FROZEN", "BLIND_OPENED"} and any(fields[name] for name in
             ("blind_response_sha256", "publication_stage_output_sha256", "publication_stage_sidecar_sha256",
              "published_output_sha256", "published_sidecar_sha256", "published_fit_sha256", "published_trace_sha256")):
            fail("pre-blind selection checkpoint contains later-stage commit fields")
        if selection_state == "BLIND_OPENED" and fields["operation_in_progress"] != "1":
            fail("opened blind checkpoint does not have a recoverable in-progress operation")
        if selection_state == "PUBLISHING" and fields["operation_in_progress"] != "1":
            fail("publication checkpoint does not have a recoverable in-progress operation")
        if selection_state == "PUBLISHED" and fields["operation_in_progress"] != "0":
            fail("published checkpoint still has an operation in progress")
        if selection_state in {"BLIND_PASSED", "PUBLISHING", "PUBLISHED"} and not fields["blind_response_sha256"]:
            fail("blind-passed checkpoint is missing its response snapshot digest")
        stage_output, stage_sidecar = fields["publication_stage_output_sha256"], fields["publication_stage_sidecar_sha256"]
        if bool(stage_output) != bool(stage_sidecar):
            fail("publication checkpoint has only one staged artifact digest")
        if selection_state == "PUBLISHING" and fields["published_fit_sha256"] and not stage_output:
            fail("publication checkpoint commits fit diagnostics before the prepared reference pair")
        if any(fields[name] and fields[name] != stage for name, stage in
               (("published_output_sha256", stage_output), ("published_sidecar_sha256", stage_sidecar))):
            fail("published artifact digest differs from the staged artifact digest")
        if selection_state == "PUBLISHED" and (not stage_output or not fields["published_fit_sha256"] or
             not fields["published_trace_sha256"] or not fields["published_output_sha256"] or
             not fields["published_sidecar_sha256"]):
            fail("published checkpoint is missing an output or certificate identity")
    try:
        active_candidate_elapsed = float(fields["active_candidate_elapsed_seconds"])
        elapsed_seconds = float(fields["elapsed_seconds"])
        operation_in_progress = int(fields["operation_in_progress"])
        operation_start_epoch_seconds = float(fields["operation_start_epoch_seconds"])
        exact_gradients = int(fields["exact_gradients"])
        cg_witness_attempts = int(fields["cg_witness_attempts"])
        family_gradients = int(fields["max_fit_family_gradients"])
        family_wall_seconds = float(fields["max_fit_family_wall_seconds"])
        training_frames = int(fields["training_frame_count"])
        development_frames = int(fields["development_frame_count"])
        training_block_length = int(fields["training_block_length"])
        training_adjacent_block_length = int(fields["training_adjacent_block_length"])
        training_covered_frames = int(fields["training_covered_frame_count"])
        training_tail_frames = int(fields["training_discarded_tail_frames"])
        training_channel_count = int(fields["training_channel_count"])
        if fields["training_statistics_frozen"] != "1" or \
           fields["training_statistics_policy_version"] != "RPJASTAT2" or \
           training_frames <= 0 or development_frames <= 0 or training_block_length <= 0 or \
           training_adjacent_block_length != 2 * training_block_length or training_covered_frames <= 0 or \
           training_covered_frames > training_frames or training_tail_frames != training_frames-training_covered_frames or \
           training_channel_count <= 0 or training_channel_count > 1000 or \
           training_channel_count != len(training_channels) or \
           len(set(training_channels)) != len(training_channels):
            fail("resume checkpoint has an invalid frozen training statistics plan")
        if elapsed_seconds > WALL_BUDGET_SECONDS or exact_gradients > MAX_EXACT_GRADIENTS or \
           family_gradients > MAX_FIT_FAMILY_GRADIENTS or family_wall_seconds > MAX_FIT_FAMILY_WALL_SECONDS:
            fail("resume checkpoint exceeds the frozen budget")
        if not fields["cg_witness_attempts"].isascii() or not fields["cg_witness_attempts"].isdecimal() or \
           cg_witness_attempts > (1 << 64) - 1 or next_candidate_index < 0 or \
           not 0 <= int(fields["active_phase"]) < 8 or \
           elapsed_seconds < 0 or elapsed_seconds > WALL_BUDGET_SECONDS or active_candidate_elapsed < 0 or \
           active_candidate_elapsed > elapsed_seconds or exact_gradients < 0 or \
           exact_gradients > MAX_EXACT_GRADIENTS or family_gradients < 0 or \
           family_gradients > MAX_FIT_FAMILY_GRADIENTS or not math.isfinite(family_wall_seconds) or \
           family_wall_seconds < -1 or family_wall_seconds > MAX_FIT_FAMILY_WALL_SECONDS or \
           operation_in_progress not in {0, 1} or not math.isfinite(operation_start_epoch_seconds) or \
           operation_start_epoch_seconds < 0 or (operation_in_progress and operation_start_epoch_seconds <= 0) or \
           fields["active_candidate_started"] not in {"0", "1"} or \
           (fields["active_candidate_started"] == "0" and active_candidate_elapsed != 0):
            raise ValueError
    except ValueError:
        fail("resume checkpoint has invalid cursor or cumulative budget fields")
    if expected_identity_sha256 is not None and fields["identity_sha256"] != expected_identity_sha256:
        fail("resume checkpoint manifest identity does not match the selected manifest")
    fields["candidate_records"] = candidate_records
    fields["checkpoint_sha256"] = hashlib.sha256(contents).hexdigest()
    return fields


def validate_cuda_regression_fixtures(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in FROZEN_CUDA_QP_FIXTURE_SHA256.items():
        path = directory / name
        if not path.is_file():
            fail(f"native fit CUDA regression fixture is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            fail(f"SHA-256 mismatch for native fit CUDA regression fixture {name}")
        hashes[name] = actual
    return hashes


def _expected_sampling_configuration(manifest: dict[str, str], sample_name: str) -> dict[str, str]:
    result = {
        "model_sha256": required(manifest, f"{sample_name}_model_sha256"),
        "physical_force_configuration_sha256": required(manifest, f"{sample_name}_physical_force_configuration_sha256"),
        "integrator_configuration_sha256": required(manifest, f"{sample_name}_integrator_configuration_sha256"),
    }
    unknown = [value == "NOT_VERIFIABLE" for value in result.values()]
    if any(unknown) and not all(unknown):
        fail(f"manifest must declare all {sample_name} model and configuration digests or mark all NOT_VERIFIABLE")
    if all(unknown):
        return result
    for key, value in result.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            fail(f"manifest field {sample_name}_{key} is not a SHA-256 digest")
        result[key] = value.lower()
    return result


def validate_sample_provenance(sample: Path, sample_name: str,
                               expected_configuration: dict[str, str] | None = None) -> dict[str, str]:
    sidecar = Path(str(sample) + ".provenance.json")
    resolved = _PROVENANCE.resolve(sample, sidecar, expected_configuration)
    fields = {
        f"{sample_name}_provenance_status": resolved["status"],
        f"{sample_name}_integrity_status": "NOT_PRESENT" if not sidecar.is_file() else "VERIFIED",
        f"{sample_name}_configuration_match_status": resolved.get("historical_match_status", "NOT_VERIFIABLE"),
        f"{sample_name}_provenance_checksum_status": "NOT_PRESENT" if not sidecar.is_file() else "VERIFIED",
        f"{sample_name}_provenance_sample_sha256": resolved.get("sample_sha256", "NOT_AVAILABLE"),
        f"{sample_name}_provenance_configuration_sha256": resolved.get("configuration_sha256", "NOT_VERIFIABLE"),
        f"{sample_name}_model_sha256": resolved.get("metadata", {}).get("model_sha256", "NOT_VERIFIABLE"),
        f"{sample_name}_physical_force_configuration_sha256": resolved.get("metadata", {}).get("physical_force_configuration_sha256", "NOT_VERIFIABLE"),
        f"{sample_name}_integrator_configuration_sha256": resolved.get("metadata", {}).get("integrator_configuration_sha256", "NOT_VERIFIABLE"),
        f"{sample_name}_source_job_id": resolved.get("source_job_id", "NOT_AVAILABLE"),
        f"{sample_name}_source_run_id": resolved.get("source_run_id", "NOT_AVAILABLE"),
    }
    if not sidecar.is_file():
        if expected_configuration and any(value != "NOT_VERIFIABLE" for value in expected_configuration.values()):
            fail(f"{sample_name} provenance sidecar is missing but the manifest declares concrete configuration digests")
        fields[f"{sample_name}_provenance_status"] = "LEGACY_DECLARED_UNVERIFIED"
        fields[f"{sample_name}_configuration_match_status"] = "NOT_VERIFIABLE"
        return fields
    if not resolved.get("checksum_valid"):
        fail(f"{sample_name} sample provenance sidecar is invalid: {resolved['status']}")
    if resolved.get("status") != "PRODUCER_OBSERVED":
        fail(f"{sample_name} sample provenance is not complete and stable: {resolved['status']}")
    if resolved.get("historical_match_status") == "MISMATCH":
        fail(f"{sample_name} sidecar model or configuration digest does not match the frozen manifest")
    if expected_configuration:
        unknown_configuration = all(value == "NOT_VERIFIABLE" for value in expected_configuration.values())
        if not unknown_configuration and resolved.get("historical_match_status") != "MATCHED_DECLARED_VALUES":
            fail(f"{sample_name} provenance sidecar lacks configuration evidence required by the manifest")
        if unknown_configuration and resolved.get("historical_match_status") == "NOT_VERIFIABLE" and any(
            key in resolved.get("metadata", {}) for key in expected_configuration
        ):
            fail(f"{sample_name} sidecar has partial configuration evidence that cannot be declared unknown")
    return fields


def validate_training_provenance(sample: Path, expected_configuration: dict[str, str] | None = None) -> dict[str, str]:
    return validate_sample_provenance(sample, "training", expected_configuration)


def validate_blind_provenance(sample: Path, expected_configuration: dict[str, str] | None = None) -> dict[str, str]:
    return validate_sample_provenance(sample, "blind", expected_configuration)


def validate_provenance_source_relation(training: dict[str, str], blind: dict[str, str]) -> str:
    training_job = training.get("training_source_job_id", "NOT_AVAILABLE")
    blind_job = blind.get("blind_source_job_id", "NOT_AVAILABLE")
    training_run = training.get("training_source_run_id", "NOT_AVAILABLE")
    blind_run = blind.get("blind_source_run_id", "NOT_AVAILABLE")
    if training_job not in {"", "NOT_AVAILABLE"} and training_job == blind_job:
        fail("training and blind provenance sidecars name the same source job")
    if training_run not in {"", "NOT_AVAILABLE"} and training_run == blind_run:
        fail("training and blind provenance sidecars name the same source run")
    if (training_job not in {"", "NOT_AVAILABLE"} and blind_job not in {"", "NOT_AVAILABLE"} and
            training_run not in {"", "NOT_AVAILABLE"} and blind_run not in {"", "NOT_AVAILABLE"}):
        return "SIDECAR_SOURCE_IDS_DISTINCT_DECLARED"
    return "NOT_VERIFIABLE"


def cuda_regression_fixture_receipt_fields(hashes: dict[str, str]) -> dict[str, str]:
    if tuple(hashes) != tuple(FROZEN_CUDA_QP_FIXTURE_SHA256):
        fail("native fit CUDA regression fixture set differs from the frozen fixture contract")
    canonical = "".join(f"{name}\0{digest}\n" for name, digest in hashes.items()).encode("utf-8")
    fields = {
        "cuda_regression_fixture_status": "PASS",
        "cuda_regression_fixture_count": str(len(hashes)),
        "cuda_regression_fixture_set_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    for index, (name, digest) in enumerate(hashes.items(), 1):
        fields[f"cuda_regression_fixture_{index}_name"] = name
        fields[f"cuda_regression_fixture_{index}_sha256"] = digest
    return fields


def fnv64(path: Path) -> str:
    value = 14695981039346656037
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            for byte in block:
                value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def is_fnv64(value: str) -> bool:
    return len(value) == 16 and all(character in "0123456789abcdefABCDEF" for character in value)


def same_resolved_path(first: str, second: str, base: Path) -> bool:
    return os.path.samefile(resolved(first, base), resolved(second, base))


def validate_distinct_training_blind(training: Path, blind: Path,
                                    training_sha256: str, blind_sha256: str) -> None:
    try:
        same_file = os.path.samefile(training, blind)
    except OSError as error:
        fail(f"cannot establish training/blind file identity: {error}")
    if same_file:
        fail("training and blind inputs resolve to the same file")
    if training_sha256.lower() == blind_sha256.lower():
        fail("training and blind inputs have identical SHA-256 content")


def _sample_shape(path: Path) -> tuple[int, int, float, tuple[float, ...], tuple[float, ...], tuple[int, ...], int]:
    header_format = "=8sIIiid9d"
    header_bytes = struct.calcsize(header_format)
    with path.open("rb") as stream:
        prefix = stream.read(header_bytes)
    if len(prefix) != header_bytes:
        fail(f"sample header is truncated: {path}")
    magic, version, endian, atoms, beads, temperature, *cell = struct.unpack(header_format, prefix)
    if magic != b"GPJASMP1" or version != 1 or endian != 0x01020304 or atoms < 2 or atoms > 2147483647 // 3 or beads < 1 or \
       not math.isfinite(temperature) or temperature <= 0.0:
        fail(f"sample header is invalid: {path}")
    with path.open("rb") as stream:
        stream.seek(header_bytes)
        masses_bytes = stream.read(8 * atoms)
        types_bytes = stream.read(4 * atoms)
    if len(masses_bytes) != 8 * atoms or len(types_bytes) != 4 * atoms:
        fail(f"sample identity header is truncated: {path}")
    masses = struct.unpack(f"={atoms}d", masses_bytes)
    types = struct.unpack(f"={atoms}i", types_bytes)
    if not all(math.isfinite(value) and value > 0.0 for value in masses) or \
       not all(math.isfinite(value) for value in cell):
        fail(f"sample header has invalid cell or masses: {path}")
    frame_start = header_bytes + 12 * atoms
    frame_bytes = struct.calcsize("=d") * (1 + 6 * atoms)
    payload_bytes = path.stat().st_size - frame_start
    if payload_bytes < 0 or payload_bytes % frame_bytes:
        fail(f"sample payload is truncated or malformed: {path}")
    frames = payload_bytes // frame_bytes
    if frames < 3:
        fail(f"sample has fewer than three frames: {path}")
    return atoms, beads, temperature, tuple(cell), tuple(masses), tuple(types), frames


def validate_blind_probe_capacity(training: Path, blind: Path) -> int:
    (training_atoms, training_beads, training_temperature, training_cell, training_masses,
     training_types, training_frames) = _sample_shape(training)
    (blind_atoms, blind_beads, blind_temperature, blind_cell, blind_masses,
     blind_types, blind_frames) = _sample_shape(blind)
    if (training_atoms, training_beads) != (blind_atoms, blind_beads) or \
       abs(training_temperature - blind_temperature) > 1.0e-10 * training_temperature:
        fail("training and blind sample headers have different atom, bead, or temperature identities")
    if training_masses != blind_masses or training_types != blind_types:
        fail("training and blind sample headers have different mass or type identities")
    if any(abs(expected - observed) > 1.0e-9 * max(1.0, abs(expected))
           for expected, observed in zip(training_cell, blind_cell)):
        fail("training and blind sample headers have different cell identities")
    # The probe builder admits at most 8 physical + 16 random + 4 type probes,
    # and four low-Ritz modes from each of A0, three B, and three C candidates.
    fitting_frames = 2 * training_frames // 3
    training_validation_frames = training_frames - fitting_frames
    if fitting_frames < MIN_DEVELOPMENT_BLOCKS or training_validation_frames < MIN_DEVELOPMENT_BLOCKS:
        fail("training or development split has too few frames for the frozen minimum four statistical blocks")
    fixed_probe_cap = 8 + 16 + 4
    soft_probe_cap = min(4 * (1 + 3 + 3), max(0, training_validation_frames - 1))
    maximum_probe_rank = min(3 * training_atoms - 3, max(0, training_validation_frames - 1),
                             fixed_probe_cap + soft_probe_cap)
    if blind_frames <= maximum_probe_rank:
        fail(f"blind trajectory has {blind_frames} frames; frozen probe capacity requires more than "
             f"{maximum_probe_rank} frames")
    return maximum_probe_rank


def validate_native_identity_fields(manifest: dict[str, str]) -> None:
    for key in ("training_fnv64", "blind_fnv64", "declared_model_fingerprint", "declared_config_fingerprint"):
        if not is_fnv64(required(manifest, key)):
            fail(f"manifest field {key!r} must be a 64-bit hexadecimal fingerprint")


def validate_m1_minimum_support(value: str) -> int:
    try:
        count = int(value, 10)
    except ValueError:
        fail("manifest field 'm1_minimum_support_count' must be an integer from 1 through 1000")
    if str(count) != value or not 1 <= count <= 1000:
        fail("manifest field 'm1_minimum_support_count' must be an integer from 1 through 1000")
    return count


def validate_response_delta_tolerance(value: str) -> float:
    try:
        tolerance = float(value)
    except ValueError:
        fail("manifest field 'response_delta_tolerance' must be a finite positive number")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        fail("manifest field 'response_delta_tolerance' must be a finite positive number")
    return tolerance


def validate_curvature_contract(manifest: dict[str, str]) -> dict[str, str]:
    primary_seed = required(manifest, "curvature_primary_seed").lower()
    independent_seed = required(manifest, "curvature_independent_seed").lower()
    if primary_seed != CURVATURE_PRIMARY_SEED or independent_seed != CURVATURE_INDEPENDENT_SEED:
        fail("manifest curvature seeds do not match the executable's frozen RPJAFEAS9 contract")
    try:
        probe_cap = int(required(manifest, "curvature_probe_cap"), 10)
        tolerance = float(required(manifest, "curvature_directional_tolerance"))
    except ValueError:
        fail("manifest curvature probe cap or directional tolerance is invalid")
    if str(probe_cap) != manifest["curvature_probe_cap"] or probe_cap != CURVATURE_PROBE_CAP:
        fail("manifest curvature_probe_cap must equal the executable's frozen cap of 128")
    if not math.isfinite(tolerance) or tolerance != CURVATURE_DIRECTIONAL_TOLERANCE:
        fail("manifest curvature_directional_tolerance must equal the executable's frozen value 0.1")
    return {
        "curvature_primary_seed": primary_seed,
        "curvature_independent_seed": independent_seed,
        "curvature_probe_cap": str(probe_cap),
        "curvature_directional_tolerance": format(tolerance, ".17g"),
    }


def validate_candidate_cost_contract(manifest: dict[str, str]) -> dict[str, str]:
    rule = required(manifest, "candidate_cost_stop_rule")
    if rule != CANDIDATE_COST_STOP_RULE:
        fail("manifest candidate_cost_stop_rule does not match the executable's frozen rule")
    try:
        safety_factor = float(required(manifest, "candidate_cost_safety_factor"))
    except ValueError:
        fail("manifest candidate_cost_safety_factor must be finite and between 1 and 10")
    if not math.isfinite(safety_factor) or not 1.0 <= safety_factor <= 10.0:
        fail("manifest candidate_cost_safety_factor must be finite and between 1 and 10")
    return {
        "candidate_cost_stop_rule": rule,
        "candidate_cost_safety_factor": format(safety_factor, ".17g"),
    }


def validate_performance_stop_contract(manifest: dict[str, str]) -> dict[str, str]:
    rule = required(manifest, "performance_stop_rule")
    normalizer = required(manifest, "performance_stop_objective_normalizer")
    gradient_cost_rule = required(manifest, "performance_stop_gradient_cost_rule")
    try:
        window = int(required(manifest, "performance_stop_curvature_window"), 10)
        min_improvement = float(required(manifest, "performance_stop_min_relative_objective_improvement"))
        cost_ratio = float(required(manifest, "performance_stop_curvature_to_gradient_cost_ratio"))
    except ValueError:
        fail("manifest performance-stop window or threshold is invalid")
    if rule != PERFORMANCE_STOP_RULE or normalizer != PERFORMANCE_STOP_OBJECTIVE_NORMALIZER or \
       gradient_cost_rule != PERFORMANCE_STOP_GRADIENT_COST_RULE or str(window) != manifest["performance_stop_curvature_window"] or \
       window != PERFORMANCE_STOP_CURVATURE_WINDOW or not math.isfinite(min_improvement) or \
       min_improvement != PERFORMANCE_STOP_MIN_RELATIVE_IMPROVEMENT or not math.isfinite(cost_ratio) or \
       cost_ratio != PERFORMANCE_STOP_CURVATURE_TO_GRADIENT_COST_RATIO:
        fail("manifest performance-stop policy does not match the executable's frozen RPJAFEAS9 heuristic")
    return {
        "performance_stop_rule": rule,
        "performance_stop_objective_normalizer": normalizer,
        "performance_stop_gradient_cost_rule": gradient_cost_rule,
        "performance_stop_curvature_window": str(window),
        "performance_stop_min_relative_objective_improvement": format(min_improvement, ".17g"),
        "performance_stop_curvature_to_gradient_cost_ratio": format(cost_ratio, ".17g"),
    }


def machine_fingerprint(gpu_inventory: str) -> str:
    identity = "\n".join((platform.system(), platform.release(), platform.machine(), platform.processor(),
                          platform.node(), os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                          os.environ.get("CUDA_PATH", ""), gpu_inventory.strip()))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def validate_cost_baseline(path: Path, gpu_inventory: str, atom_count: int, sample_frames: int,
                           training_frames: int, development_frames: int, m0_parameter_upper_bound: int) -> dict[str, str]:
    fields = parse_quoted_fields(path, "RPJACOST1")
    expected_strings = {
        "machine_fingerprint": machine_fingerprint(gpu_inventory),
        "atom_count": str(atom_count),
        "sample_frames": str(sample_frames),
        "training_frames": str(training_frames),
        "development_frames": str(development_frames),
    }
    for key, expected in expected_strings.items():
        if required(fields, key) != expected:
            fail(f"cost baseline {key} does not match this machine or problem size")
    try:
        m0_parameter_count = int(required(fields, "m0_parameter_count"), 10)
        m1_parameter_count = int(required(fields, "m1_parameter_count"), 10)
    except ValueError:
        fail("cost baseline parameter count is invalid")
    if (str(m0_parameter_count) != fields["m0_parameter_count"] or m0_parameter_count <= 0 or
            m0_parameter_count > m0_parameter_upper_bound):
        fail("cost baseline m0_parameter_count exceeds the theoretical upper bound")
    if str(m1_parameter_count) != fields["m1_parameter_count"] or not m0_parameter_count <= m1_parameter_count <= 92:
        fail("cost baseline m1_parameter_count is outside the frozen parameter range")
    values = {}
    for key in ("candidate_seconds", "fit_seconds", "curvature_seconds", "blind_seconds", "package_write_seconds",
                "prepare_seconds", "readback_seconds"):
        try:
            value = float(required(fields, key))
        except ValueError:
            fail(f"cost baseline {key} must be finite and positive")
        if not math.isfinite(value) or value <= 0.0:
            fail(f"cost baseline {key} must be finite and positive")
        values[f"cost_baseline_{key}"] = format(value, ".17g")
    return {
        "cost_baseline_status": "VERIFIED",
        "cost_baseline_machine_fingerprint": expected_strings["machine_fingerprint"],
        "cost_baseline_atom_count": str(atom_count),
        "cost_baseline_sample_frames": str(sample_frames),
        "cost_baseline_training_frames": str(training_frames),
        "cost_baseline_development_frames": str(development_frames),
        "cost_baseline_m0_parameter_count": str(m0_parameter_count),
        "cost_baseline_m0_parameter_upper_bound": str(m0_parameter_upper_bound),
        "cost_baseline_m1_parameter_count": str(m1_parameter_count),
        **values,
    }


def validate_experiment_contract(manifest: dict[str, str]) -> dict[str, str]:
    try:
        lambda_factors = tuple(float(value) for value in required(manifest, "lambda_factors").split(","))
        wall_seconds = float(required(manifest, "wall_seconds"))
        validation_reserve = float(required(manifest, "validation_reserve_fraction"))
        a_wall_fraction = float(required(manifest, "a_wall_fraction"))
        max_exact_gradients = int(required(manifest, "max_exact_gradients"), 10)
        max_a_branch_gradients = int(required(manifest, "max_a_branch_gradients"), 10)
        max_fit_family_gradients = int(required(manifest, "max_fit_family_gradients"), 10)
        max_blocks = int(required(manifest, "development_max_blocks"), 10)
        min_blocks = int(required(manifest, "development_min_blocks"), 10)
        statistics_storage_budget_bytes = int(required(manifest, "statistics_storage_budget_bytes"), 10)
    except ValueError:
        fail("manifest experiment lambda, budget, or block contract has an invalid number")
    if len(lambda_factors) != len(LAMBDA_FACTORS) or any(not math.isfinite(x) for x in lambda_factors) or \
       lambda_factors != LAMBDA_FACTORS:
        fail("manifest lambda_factors must be the frozen sequence 0.01,0.0001,0.000001")
    if wall_seconds != WALL_BUDGET_SECONDS or validation_reserve != 0.25 or a_wall_fraction != 0.10 or \
       max_exact_gradients != MAX_EXACT_GRADIENTS or max_a_branch_gradients != MAX_A_BRANCH_GRADIENTS or \
       max_fit_family_gradients != MAX_FIT_FAMILY_GRADIENTS:
        fail("manifest budget values do not match the executable's frozen RPJAFEAS9 budget")
    if required(manifest, "family_wall_split_rule") != "EQUAL_REMAINING_FIT_TIME_V1" or \
       required(manifest, "family_gradient_split_rule") != "EQUAL_REMAINING_EXACT_GRADIENTS_V1":
        fail("manifest B/C budget split rules do not match the frozen executable contract")
    if max_blocks != MAX_DEVELOPMENT_BLOCKS or min_blocks != MIN_DEVELOPMENT_BLOCKS:
        fail("manifest development block counts do not match the frozen 8-block/4-minimum rule")
    if required(manifest, "development_block_rule") != "CONTIGUOUS_EQUAL_FRAME_COUNTS_V1":
        fail("manifest development block rule does not match the frozen executable contract")
    if str(statistics_storage_budget_bytes) != manifest["statistics_storage_budget_bytes"] or \
       statistics_storage_budget_bytes != STATISTICS_STORAGE_BUDGET_BYTES:
        fail("manifest statistics_storage_budget_bytes must equal the executable's frozen 512 MiB budget")
    return {
        "lambda_factors": ",".join(format(value, ".17g") for value in lambda_factors),
        "wall_seconds": format(wall_seconds, ".17g"),
        "max_exact_gradients": str(max_exact_gradients),
        "validation_reserve_fraction": format(validation_reserve, ".17g"),
        "max_a_branch_gradients": str(max_a_branch_gradients),
        "max_fit_family_gradients": str(max_fit_family_gradients),
        "a_wall_fraction": format(a_wall_fraction, ".17g"),
        "family_wall_split_rule": "EQUAL_REMAINING_FIT_TIME_V1",
        "family_gradient_split_rule": "EQUAL_REMAINING_EXACT_GRADIENTS_V1",
        "development_max_blocks": str(max_blocks),
        "development_min_blocks": str(min_blocks),
        "development_block_rule": "CONTIGUOUS_EQUAL_FRAME_COUNTS_V1",
        "statistics_storage_budget_bytes": str(statistics_storage_budget_bytes),
    }


def native_fit_regression_command(executable: Path, fixture_directory: Path) -> list[str]:
    return [str(executable), "--fixture-dir", str(fixture_directory)]


def input_tokens(line: str) -> list[str]:
    return line.split("#", 1)[0].split()


def find_fit_command(run_config: Path) -> tuple[list[str], str, str]:
    matches: list[list[str]] = []
    commands: list[list[str]] = []
    with run_config.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            tokens = input_tokens(line)
            if tokens:
                commands.append(tokens)
            if tokens[:2] == ["rpmd_ja", "fit_samples_covariance"]:
                matches.append(tokens)
    if len(matches) != 1:
        fail("run.in must contain exactly one rpmd_ja fit_samples_covariance command")
    tokens = matches[0]
    if any(command[0] in {"run", "ensemble"} for command in commands):
        fail("offline feasibility run.in must not contain production run or ensemble commands")
    if commands[-1] != tokens:
        fail("offline feasibility run.in must end with its rpmd_ja fit_samples_covariance command; production run/ensemble commands are not allowed")
    unsupported = sorted({command[0] for command in commands[:-1]
                          if command[0] not in OFFLINE_SETUP_COMMANDS})
    if unsupported:
        fail("offline feasibility run.in only allows potential, pppm_mesh_spacing, pimd_fix_com, and kspace before the fit; unsupported commands: "
             + ", ".join(unsupported))
    if len(tokens) < 14 or "feasibility" not in tokens or "structured_basis" not in tokens:
        fail("run.in must enable feasibility and structured_basis")
    try:
        blind_samples = tokens[tokens.index("blind_samples") + 1]
        blind_manifest = tokens[tokens.index("blind_manifest") + 1]
    except (ValueError, IndexError):
        fail("run.in must supply blind_samples and blind_manifest")
    if tokens.count("blind_samples") != 1 or tokens.count("blind_manifest") != 1:
        fail("run.in must supply each blind input exactly once")
    return tokens, blind_samples, blind_manifest


def no_existing_outputs(output: Path, resume: bool = False, checkpoint_path: Path | None = None,
                        resume_checkpoint: dict[str, object] | None = None) -> None:
    suffixes = ["", ".stability", ".qraw", ".additive.tmp", ".fit.txt", ".fit.txt.tmp",
                ".cg_witness.txt", ".qp_state.txt", ".response_state.txt", ".covariance_state.txt",
                ".failure.txt", ".tmp", ".stability.tmp", ".fit_trace.txt", ".checkpoint",
                ".development_checkpoint.txt", ".blind_checkpoint.txt", ".a0_reuse.record",
                ".a0_reuse.record.manifest", ".a0_reuse.record.state"]
    paths = [Path(str(output) + suffix) for suffix in suffixes]
    if resume:
        paths = [path for path in paths if not str(path).endswith(".a0_reuse.record")]
    paths += [Path(str(output) + ".covariance_state.txt." + label)
              for label in ("A0", "A1", "B0", "B1", "B2", "C0", "C1", "C2")]
    if not resume:
        if checkpoint_path is not None and checkpoint_path != Path(str(output) + ".checkpoint"):
            paths.append(checkpoint_path)
        existing = [str(path) for path in paths if path.exists()]
        existing += [str(path) for path in output.parent.glob(output.name + ".stage.*")]
        existing += [str(path) for path in output.parent.glob(output.name + ".a0_reuse.record.tmp.*")]
        if existing:
            fail("refusing to overwrite existing output/scratch artifacts: " + ", ".join(existing))
        return
    if checkpoint_path is None or resume_checkpoint is None:
        fail("resume requires the parsed controller checkpoint and an explicit checkpoint path")
    if checkpoint_path != Path(str(output) + ".checkpoint"):
        paths.append(Path(str(output) + ".checkpoint"))
    paths.append(Path(str(checkpoint_path) + ".solver"))

    state = str(resume_checkpoint["selection_state"])
    allowed_digests: dict[Path, str] = {}
    allowed_uncommitted: set[Path] = set()

    def allow(path: Path, digest: str = "", uncommitted: bool = False) -> None:
        if digest:
            allowed_digests[path] = digest
        elif uncommitted:
            allowed_uncommitted.add(path)

    checkpoint_digest = str(resume_checkpoint.get("checkpoint_sha256", ""))
    allow(checkpoint_path, checkpoint_digest, uncommitted=not checkpoint_digest)
    allow(Path(str(output) + ".checkpoint"), checkpoint_digest, uncommitted=not checkpoint_digest)
    allow(Path(str(checkpoint_path) + ".solver"), uncommitted=True)
    allow(Path(str(output) + ".fit_trace.txt"), str(resume_checkpoint.get("published_trace_sha256", "")),
          uncommitted=state != "PUBLISHED")
    allow(Path(str(output) + ".development_checkpoint.txt"),
          str(resume_checkpoint.get("development_response_sha256", "")), uncommitted=state == "NOT_FROZEN")
    allow(Path(str(output) + ".blind_checkpoint.txt"), str(resume_checkpoint.get("blind_response_sha256", "")),
          uncommitted=state == "BLIND_OPENED")

    stage_output_digest = str(resume_checkpoint.get("publication_stage_output_sha256", ""))
    stage_sidecar_digest = str(resume_checkpoint.get("publication_stage_sidecar_sha256", ""))
    if state in {"PUBLISHING", "PUBLISHED"}:
        allow(output, stage_output_digest)
        allow(Path(str(output) + ".stability"), stage_sidecar_digest)
        fit_digest = str(resume_checkpoint.get("published_fit_sha256", ""))
        allow(Path(str(output) + ".fit.txt"), fit_digest)
        if state == "PUBLISHING":
            allow(Path(str(output) + ".fit.txt.tmp"), fit_digest)
            allow(Path(str(output) + ".additive.tmp"), uncommitted=True)
        job_stage = Path(str(output) + ".stage." + hashlib.sha256(
            str(resume_checkpoint["job_id"]).encode("utf-8")).hexdigest()[:16])
        if state == "PUBLISHING":
            allow(job_stage, stage_output_digest, uncommitted=not stage_output_digest)
            allow(Path(str(job_stage) + ".stability"), stage_sidecar_digest,
                  uncommitted=not stage_sidecar_digest)
            allow(Path(str(job_stage) + ".tmp"), uncommitted=True)
            allow(Path(str(job_stage) + ".stability.tmp"), uncommitted=True)

    candidate_records = resume_checkpoint.get("candidate_records", [])
    active_branch = str(resume_checkpoint["branch"])
    active_candidate_started = resume_checkpoint["active_candidate_started"] == "1"
    for label in ("A0", "A1", "B0", "B1", "B2", "C0", "C1", "C2"):
        path = Path(str(output) + ".covariance_state.txt." + label)
        record = next((candidate for candidate in candidate_records if candidate.get("branch") == label), None)
        if record is not None and record.get("fit_sha256"):
            allow(path, str(record["fit_sha256"]))
        elif label == active_branch and active_candidate_started:
            allow(path, uncommitted=True)  # Native resume verifies the branch-state identity before use.

    required_paths = [checkpoint_path, Path(str(output) + ".fit_trace.txt")]
    required_alternatives: list[tuple[Path, ...]] = []
    if state != "NOT_FROZEN":
        required_paths.append(Path(str(output) + ".development_checkpoint.txt"))
        selected_state = Path(str(output) + ".covariance_state.txt." +
                              str(resume_checkpoint["frozen_candidate_branch"]))
        allow(selected_state, str(resume_checkpoint["frozen_candidate_fit_state_sha256"]))
        required_paths.append(selected_state)
    if state in {"BLIND_PASSED", "PUBLISHING", "PUBLISHED"}:
        required_paths.append(Path(str(output) + ".blind_checkpoint.txt"))
    if state == "PUBLISHING" and stage_output_digest:
        job_stage = Path(str(output) + ".stage." + hashlib.sha256(
            str(resume_checkpoint["job_id"]).encode("utf-8")).hexdigest()[:16])
        required_alternatives.extend(((output, job_stage),
                                      (Path(str(output) + ".stability"), Path(str(job_stage) + ".stability"))))
    if state == "PUBLISHING" and resume_checkpoint.get("published_fit_sha256"):
        required_alternatives.append((Path(str(output) + ".fit.txt"), Path(str(output) + ".fit.txt.tmp")))
    if state == "PUBLISHED":
        required_paths.extend((output, Path(str(output) + ".stability"), Path(str(output) + ".fit.txt")))

    if state in {"FROZEN", "BLIND_OPENED", "BLIND_PASSED"}:
        paths.extend(Path(str(output) + suffix) for suffix in (".fit.txt", ".fit.txt.tmp", ".additive.tmp"))
    path_allowlist = set(allowed_digests) | allowed_uncommitted
    existing = [path for path in paths if path.exists()]
    for path in output.parent.glob(output.name + ".stage.*"):
        if path not in existing:
            existing.append(path)
    unexpected = [str(path) for path in existing if path not in path_allowlist]
    if unexpected:
        fail("refusing to overwrite existing output/scratch artifacts: " + ", ".join(unexpected))
    missing = [str(path) for path in required_paths if not path.is_file()]
    missing += [" or ".join(str(path) for path in group) for group in required_alternatives
                if not any(path.is_file() for path in group)]
    if missing:
        fail("resume checkpoint is missing required committed artifacts: " + ", ".join(missing))
    for path, expected in allowed_digests.items():
        if path.exists() and (not path.is_file() or sha256(path) != expected):
            fail(f"resume artifact identity mismatch: {path}")
    for path in allowed_uncommitted:
        if path.exists() and not path.is_file():
            fail(f"resume scratch artifact is not a regular file: {path}")


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_cli_arguments(arguments: list[str]) -> tuple[str, str, bool]:
    resume = arguments.count("--resume") == 1
    if arguments.count("--resume") > 1:
        fail("usage: python rpmd_ja_feasibility_preflight.py [--resume] <manifest> <run-directory>")
    remaining = [value for value in arguments if value != "--resume"]
    if len(remaining) != 2 or any(value.startswith("--") for value in remaining):
        fail("usage: python rpmd_ja_feasibility_preflight.py [--resume] <manifest> <run-directory>")
    return remaining[0], remaining[1], resume


def acquire_preflight_lock(output: Path, job_id: str) -> tuple[Path, str]:
    path = Path(str(output) + ".preflight.lock")
    token = f"{job_id} {os.getpid()} {uuid.uuid4().hex}"
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ImportError):
        stream.close()
        fail(f"another feasibility launcher owns the single-job lock: {path}")
    stream.seek(0)
    stream.truncate()
    stream.write((token + "\n").encode("ascii"))
    stream.flush()
    os.fsync(stream.fileno())
    _PREFLIGHT_LOCKS.append(stream)
    return path, token


def write_receipt(path: Path, fields: dict[str, str], magic: str = "RPJAPREFLIGHT5") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(magic + "\n")
        for key, value in fields.items():
            stream.write(f"{key} {quote(value)}\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(arguments: list[str] | None = None) -> int:
    manifest_argument, run_directory_argument, resume_mode = parse_cli_arguments(
        sys.argv[1:] if arguments is None else arguments)
    manifest_path = Path(manifest_argument).resolve(strict=True)
    run_directory = Path(run_directory_argument).resolve(strict=True)
    manifest = parse_quoted_fields(manifest_path, "RPJAFEAS9")
    if required(manifest, "independence_attested") != "YES" or required(manifest, "wrapper_checksum_status") != "VERIFIED":
        fail("manifest lacks independent-data and wrapper-checksum attestations")
    if required(manifest, "cuda_regression_status") != "PASS":
        fail("manifest must declare the required CUDA regression")
    if required(manifest, "cuda_integration_status") != "PASS":
        fail("manifest must declare the required real-linked CUDA integration regression")
    validate_native_identity_fields(manifest)
    m1_minimum_support_count = validate_m1_minimum_support(required(manifest, "m1_minimum_support_count"))
    response_delta_tolerance = validate_response_delta_tolerance(required(manifest, "response_delta_tolerance"))
    curvature_contract = validate_curvature_contract(manifest)
    candidate_cost_contract = validate_candidate_cost_contract(manifest)
    performance_stop_contract = validate_performance_stop_contract(manifest)
    cost_baseline_status = manifest.get("cost_baseline_status", "NOT_PROVIDED")
    if cost_baseline_status not in {"NOT_PROVIDED", "VERIFIED"}:
        fail("manifest cost_baseline_status must be NOT_PROVIDED or VERIFIED")
    if cost_baseline_status == "VERIFIED":
        required(manifest, "cost_baseline_path")
        cost_baseline_sha256 = required(manifest, "cost_baseline_sha256").lower()
        if len(cost_baseline_sha256) != 64 or any(char not in "0123456789abcdef" for char in cost_baseline_sha256):
            fail("manifest cost_baseline_sha256 must be a SHA-256 digest")
    elif any(key in manifest for key in ("cost_baseline_path", "cost_baseline_sha256")):
        fail("manifest cost baseline path and digest require cost_baseline_status VERIFIED")
    experiment_contract = validate_experiment_contract(manifest)
    training_model_sha256 = required(manifest, "training_model_sha256")
    blind_model_sha256 = required(manifest, "blind_model_sha256")
    if (training_model_sha256 == "NOT_VERIFIABLE") != (blind_model_sha256 == "NOT_VERIFIABLE") or \
       (training_model_sha256 != "NOT_VERIFIABLE" and training_model_sha256.lower() != blind_model_sha256.lower()):
        fail("training and blind model content identities must match or both be NOT_VERIFIABLE")

    run_config = resolved(required(manifest, "run_config_path"), run_directory)
    if run_config != (run_directory / "run.in").resolve(strict=True):
        fail("run_config_path must resolve to run.in in the selected run directory")
    executable = resolved(required(manifest, "native_executable_path"), run_directory)
    regression = resolved(required(manifest, "cuda_regression_executable"), run_directory)
    integration = resolved(required(manifest, "cuda_integration_executable"), run_directory)
    fixture_directory = Path(__file__).resolve().parents[1] / "tests" / "data"
    if not fixture_directory.is_dir():
        fail(f"native fit CUDA regression fixture directory does not exist: {fixture_directory}")
    fixture_hashes_before = validate_cuda_regression_fixtures(fixture_directory)
    file_fields = [
        ("training_path", "training_sha256"), ("blind_path", "blind_sha256"),
        ("kernel_path", "kernel_sha256"), ("raw_path", "raw_sha256"),
        ("run_config_path", "run_config_sha256"),
        ("native_executable_path", "native_executable_sha256"),
        ("cuda_regression_executable", "cuda_regression_sha256"),
        ("cuda_integration_executable", "cuda_integration_sha256"),
    ]
    if cost_baseline_status == "VERIFIED":
        file_fields.append(("cost_baseline_path", "cost_baseline_sha256"))
    resolved_files: dict[str, Path] = {}
    for path_key, hash_key in file_fields:
        source = run_config if path_key == "run_config_path" else (
            executable if path_key == "native_executable_path" else
            regression if path_key == "cuda_regression_executable" else
            integration if path_key == "cuda_integration_executable" else
            resolved(required(manifest, path_key), run_directory))
        expected = required(manifest, hash_key).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            fail(f"manifest field {hash_key!r} is not a SHA-256 digest")
        if sha256(source) != expected:
            fail(f"SHA-256 mismatch for {path_key}")
        resolved_files[path_key] = source
    manifest_identity = fnv64(manifest_path)

    validate_distinct_training_blind(resolved_files["training_path"], resolved_files["blind_path"],
                                     manifest["training_sha256"], manifest["blind_sha256"])
    training_provenance = validate_training_provenance(
        resolved_files["training_path"], _expected_sampling_configuration(manifest, "training"))
    blind_provenance = validate_blind_provenance(
        resolved_files["blind_path"], _expected_sampling_configuration(manifest, "blind"))
    provenance_source_relation = validate_provenance_source_relation(training_provenance, blind_provenance)
    blind_probe_capacity = validate_blind_probe_capacity(resolved_files["training_path"],
                                                         resolved_files["blind_path"])
    if required(manifest, "training_source_id") == required(manifest, "blind_source_id"):
        fail("training and blind source IDs must differ")
    native_output_path = required(manifest, "native_output_path")
    command, command_blind_samples, command_blind_manifest = find_fit_command(run_config)
    if not same_resolved_path(command[2], manifest["training_path"], run_directory) or \
       command[3] != manifest["native_output_path"] or \
       not same_resolved_path(command[8], manifest["kernel_path"], run_directory) or \
       not same_resolved_path(command[9], manifest["raw_path"], run_directory) or \
       not same_resolved_path(command_blind_samples, manifest["blind_path"], run_directory) or \
       not same_resolved_path(command_blind_manifest, str(manifest_path), run_directory):
        fail("run.in fit inputs/output do not match the frozen manifest")
    output_arg = Path(command[3])
    output = (output_arg if output_arg.is_absolute() else run_directory / output_arg).resolve()
    if not output.parent.is_dir():
        fail(f"output parent directory does not exist: {output.parent}")
    checkpoint_arg = Path(required(manifest, "checkpoint_path"))
    checkpoint_path = (checkpoint_arg if checkpoint_arg.is_absolute() else run_directory / checkpoint_arg).resolve()
    if checkpoint_path == output or checkpoint_path in resolved_files.values():
        fail("checkpoint_path must be distinct from the output and every frozen input")
    if not checkpoint_path.parent.is_dir():
        fail(f"checkpoint parent directory does not exist: {checkpoint_path.parent}")
    resume_checkpoint = inspect_resume_checkpoint(checkpoint_path, sha256(manifest_path)) if resume_mode else None
    no_existing_outputs(output, resume=resume_mode, checkpoint_path=checkpoint_path,
                        resume_checkpoint=resume_checkpoint)
    job_id = resume_checkpoint["job_id"] if resume_checkpoint else None
    if resume_mode and not job_id:
        fail("resume checkpoint has no original job ID")
    if not resume_mode:
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    acquire_preflight_lock(output, job_id)

    sample = resolved(manifest["training_path"], run_directory)
    raw = resolved(manifest["raw_path"], run_directory)
    blind = resolved(manifest["blind_path"], run_directory)
    kernel = resolved(manifest["kernel_path"], run_directory)
    config_bytes = sum(path.stat().st_size for path in (sample, raw, blind, kernel, run_config))
    disk_required_floor = config_bytes + 1024**3
    if shutil.disk_usage(output.parent).free < disk_required_floor:
        fail(f"free disk is below the 1 GiB plus input-size reserve ({disk_required_floor} bytes)")
    if shutil.which("nvidia-smi") is None:
        fail("nvidia-smi is not available; NVIDIA device inventory cannot be recorded")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
        fail("CUDA_VISIBLE_DEVICES=-1 hides all CUDA devices")
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if gpu.returncode != 0 or not gpu.stdout.strip():
        fail("nvidia-smi did not report a visible GPU")
    try:
        a0_reuse_receipt = load_a0_reuse_package(manifest, run_directory, manifest_path.read_bytes(),
                                                  machine_fingerprint(gpu.stdout), str(job_id))
    except (OSError, ValueError) as error:
        fail(f"A0 reuse package validation failed: {error}")
    (baseline_atoms, _baseline_beads, _baseline_temperature, _baseline_cell, _baseline_masses,
     baseline_types, baseline_frames) = _sample_shape(resolved_files["training_path"])
    baseline_training_frames = 2 * baseline_frames // 3
    baseline_development_frames = baseline_frames - baseline_training_frames
    baseline_type_count = len(set(baseline_types))
    baseline_m0_parameter_count = 6 * baseline_type_count * (baseline_type_count + 1) // 2
    if cost_baseline_status == "VERIFIED":
        cost_baseline_receipt = validate_cost_baseline(resolved_files["cost_baseline_path"], gpu.stdout,
            baseline_atoms, baseline_frames, baseline_training_frames, baseline_development_frames,
            baseline_m0_parameter_count)
        cost_baseline_receipt["cost_baseline_path"] = manifest["cost_baseline_path"]
        cost_baseline_receipt["cost_baseline_resolved_path"] = str(resolved_files["cost_baseline_path"])
        cost_baseline_receipt["cost_baseline_sha256"] = cost_baseline_sha256
    else:
        cost_baseline_receipt = {
            "cost_baseline_status": "NOT_PROVIDED",
            "cost_baseline_path": "NOT_PROVIDED",
            "cost_baseline_resolved_path": "NOT_PROVIDED",
            "cost_baseline_sha256": "NOT_PROVIDED",
            "cost_baseline_machine_fingerprint": machine_fingerprint(gpu.stdout),
            "cost_baseline_atom_count": str(baseline_atoms),
            "cost_baseline_sample_frames": str(baseline_frames),
            "cost_baseline_training_frames": str(baseline_training_frames),
            "cost_baseline_development_frames": str(baseline_development_frames),
            "cost_baseline_m0_parameter_count": "NOT_AVAILABLE",
            "cost_baseline_m0_parameter_upper_bound": str(baseline_m0_parameter_count),
            "cost_baseline_m1_parameter_count": "NOT_AVAILABLE",
            **{f"cost_baseline_{key}": "NOT_AVAILABLE" for key in
               ("candidate_seconds", "fit_seconds", "curvature_seconds", "blind_seconds",
                "package_write_seconds", "prepare_seconds", "readback_seconds")},
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    receipt = run_directory / f"rpmd_ja_preflight_{run_id}.receipt"
    regression_log = run_directory / f"rpmd_ja_preflight_{run_id}.cuda.log"
    native_log = run_directory / f"rpmd_ja_preflight_{run_id}.gpumd.log"
    environment = os.environ.copy()
    environment["GPUMD_RPMD_JA_PREFLIGHT_RECEIPT"] = receipt.name
    environment["GPUMD_RPMD_JA_PREFLIGHT_MANIFEST"] = command_blind_manifest
    environment["GPUMD_RPMD_JA_JOB_ID"] = job_id
    environment["GPUMD_RPMD_JA_RESUME_MODE"] = "1" if resume_mode else "0"
    environment["GPUMD_RPMD_JA_CHECKPOINT_PATH"] = str(checkpoint_path)
    print("Running the declared numerical CUDA regression before GPUMD feasibility fitting.", flush=True)
    with regression_log.open("x", encoding="utf-8") as log:
        result = subprocess.run(native_fit_regression_command(regression, fixture_directory), cwd=run_directory, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        fail(f"CUDA regression exited {result.returncode}; see {regression_log}")
    print("Running the declared real-linked CUDA prepare/readback integration regression.", flush=True)
    integration_log = run_directory / f"rpmd_ja_preflight_{run_id}.integration.log"
    with integration_log.open("x", encoding="utf-8") as log:
        integration_result = subprocess.run([str(integration), "--integration-smoke"], cwd=run_directory, env=environment,
                                             stdout=log, stderr=subprocess.STDOUT, check=False)
    if integration_result.returncode != 0:
        fail(f"CUDA integration regression exited {integration_result.returncode}; see {integration_log}")
    if "ACTUAL_M1_V2_PREPARE_READBACK_PASS" not in integration_log.read_text(encoding="utf-8", errors="replace"):
        fail(f"CUDA integration executable did not run the actual M1 version-2 prepare/readback path; see {integration_log}")
    fixture_hashes_after = validate_cuda_regression_fixtures(fixture_directory)
    if fixture_hashes_after != fixture_hashes_before:
        fail("native fit CUDA regression fixtures changed while the CUDA regressions were running")
    if fnv64(manifest_path) != manifest_identity:
        fail("manifest changed while the CUDA regression was running")
    for path_key, hash_key in file_fields:
        if sha256(resolved_files[path_key]) != manifest[hash_key].lower():
            fail(f"{path_key} changed after the pre-run identity check")
    no_existing_outputs(output, resume=resume_mode, checkpoint_path=checkpoint_path,
                        resume_checkpoint=resume_checkpoint)

    receipt_fields = {
        "status": "PASS",
        "launcher_status": "PREFLIGHT_PASS_NATIVE_NOT_STARTED",
        "native_exit_code": "NOT_STARTED",
        "job_id": job_id,
        "resume_mode": "1" if resume_mode else "0",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": resume_checkpoint["checkpoint_sha256"] if resume_checkpoint else "NOT_STARTED",
        "resume_branch": resume_checkpoint["branch"] if resume_checkpoint else "NOT_STARTED",
        "resume_stage": resume_checkpoint["stage"] if resume_checkpoint else "NOT_STARTED",
        "resume_next_candidate_index": resume_checkpoint["next_candidate_index"] if resume_checkpoint else "NOT_STARTED",
        "manifest_path": command_blind_manifest,
        "manifest_fnv64": manifest_identity,
        "m1_minimum_support_count": str(m1_minimum_support_count),
        "response_delta_tolerance": format(response_delta_tolerance, ".17g"),
        **curvature_contract,
        **candidate_cost_contract,
        **performance_stop_contract,
        **a0_reuse_receipt,
        **cost_baseline_receipt,
        **experiment_contract,
        **cuda_regression_fixture_receipt_fields(fixture_hashes_before),
        "blind_maximum_probe_rank": str(blind_probe_capacity),
        **training_provenance,
        **blind_provenance,
        "provenance_source_relation_status": provenance_source_relation,
        "run_config_path": manifest["run_config_path"],
        "run_config_sha256": manifest["run_config_sha256"],
        "native_executable_path": manifest["native_executable_path"],
        "native_executable_sha256": manifest["native_executable_sha256"],
        "cuda_regression_executable": manifest["cuda_regression_executable"],
        "cuda_regression_sha256": manifest["cuda_regression_sha256"],
        "cuda_regression_status": "PASS",
        "cuda_regression_exit_code": str(result.returncode),
        "cuda_integration_executable": manifest["cuda_integration_executable"],
        "cuda_integration_sha256": manifest["cuda_integration_sha256"],
        "cuda_integration_status": "PASS",
        "cuda_integration_exit_code": str(integration_result.returncode),
        "native_output_path": native_output_path,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "cudacxx": os.environ.get("CUDACXX", ""),
        "cc": os.environ.get("CC", ""),
        "cxx": os.environ.get("CXX", ""),
        "netcdf_root": os.environ.get("NETCDF_ROOT", ""),
        "netcdf_home": os.environ.get("NETCDF_HOME", ""),
        "netcdf_dir": os.environ.get("NETCDF_DIR", ""),
        "python_executable": sys.executable,
        "gpu_inventory": gpu.stdout.strip().replace("\n", " | "),
        "disk_available_bytes": str(shutil.disk_usage(output.parent).free),
        "disk_required_floor_bytes": str(disk_required_floor),
        "disk_estimate_method": "input_sizes_plus_1GiB_reserve; native package size is reported by the fitter",
        "cuda_regression_log": str(regression_log),
        "cuda_integration_log": str(integration_log),
    }
    write_receipt(receipt, receipt_fields, "RPJAPREFLIGHT5")
    print(f"CUDA regression passed; receipt: {receipt}", flush=True)
    print("Launching GPUMD with the checked run.in.", flush=True)
    with native_log.open("x", encoding="utf-8") as log:
        result = subprocess.run([str(executable)], cwd=run_directory, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt_fields["native_exit_code"] = str(result.returncode)
    receipt_fields["launcher_status"] = "COMPLETE" if result.returncode == 0 else "RUN_FAILED"
    write_receipt(receipt, receipt_fields, "RPJAPREFLIGHT5")
    print(f"GPUMD exit code {result.returncode}; output: {native_log}; receipt: {receipt}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
