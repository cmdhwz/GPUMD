#!/usr/bin/env python3
"""Versioned sample-hash binding and conservative RPMD-JA provenance checks."""

import hashlib
import json
import math
from pathlib import Path


SIDECAR_SCHEMA = "RPJAPROV2"
ATTESTATION_SCHEMA = "RPJASOURCEATT1"
CONFIG_ATTESTATION_SCHEMA = "RPJACONFIGATT1"
REQUIRED_CONFIGURATION_FIELDS = (
    "integrator_type",
    "timestep_internal",
    "timestep_internal_unit",
    "timestep_fs",
    "timestep_fs_unit",
    "thermostat_configuration_tokens",
    "model_sha256",
    "run_input_sha256",
    "physical_force_configuration_sha256",
    "integrator_configuration_sha256",
    "atom_signature_sha256",
    "physical_force_source",
    "active_configuration_scope",
    "active_run_ordinal",
    "internal_mass_com",
    "sampling_com_method",
    "sampling_com_shift_final",
    "configuration_stable_during_run",
)


def sha256_bytes(contents):
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


def _finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def _configuration_missing(metadata):
    missing = []
    for field in REQUIRED_CONFIGURATION_FIELDS:
        if field not in metadata:
            missing.append(field)
    for field in ("model_sha256", "run_input_sha256", "physical_force_configuration_sha256",
                  "integrator_configuration_sha256", "atom_signature_sha256"):
        if field in metadata and not _valid_sha256(metadata[field]):
            missing.append(field)
    if "integrator_type" in metadata and not metadata["integrator_type"]:
        missing.append("integrator_type")
    if "timestep_internal" in metadata and not _finite_positive(metadata["timestep_internal"]):
        missing.append("timestep_internal")
    if "timestep_fs" in metadata and not _finite_positive(metadata["timestep_fs"]):
        missing.append("timestep_fs")
    if metadata.get("timestep_internal_unit") != "GPUMD_INTERNAL_TIME":
        missing.append("timestep_internal_unit")
    if metadata.get("timestep_fs_unit") != "fs":
        missing.append("timestep_fs_unit")
    if not isinstance(metadata.get("thermostat_configuration_tokens"), list) or not metadata.get("thermostat_configuration_tokens"):
        missing.append("thermostat_configuration_tokens")
    if metadata.get("physical_force_source") != "MEAN_OF_PHYSICAL_PIMD_BEAD_FORCES":
        missing.append("physical_force_source")
    if not isinstance(metadata.get("internal_mass_com"), bool):
        missing.append("internal_mass_com")
    if metadata.get("sampling_com_method") != "MASS_WEIGHTED_CUMULATIVE_CENTROID_DRIFT_REMOVAL":
        missing.append("sampling_com_method")
    shifts = metadata.get("sampling_com_shift_final")
    if not isinstance(shifts, list) or len(shifts) != 3 or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
        for value in shifts
    ):
        missing.append("sampling_com_shift_final")
    if metadata.get("configuration_stable_during_run") is not True:
        missing.append("configuration_stable_during_run")
    if metadata.get("active_configuration_scope") != "PREFIX_BEFORE_ACTIVE_RUN":
        missing.append("active_configuration_scope")
    if (not isinstance(metadata.get("active_run_ordinal"), int) or
            isinstance(metadata.get("active_run_ordinal"), bool) or metadata.get("active_run_ordinal", -1) < 0):
        missing.append("active_run_ordinal")
    return sorted(set(missing))


def make_sidecar(sample_bytes, sample_frames, source_job_id, source_job_id_status, metadata,
                 source_run_id="", step_start=None, step_end=None):
    if not isinstance(sample_bytes, bytes) or sample_frames <= 0 or not source_job_id:
        raise ValueError("sample bytes, positive frame count, and source job identity are required")
    if source_job_id_status not in ("SCHEDULER_DECLARED", "RUN_LOCAL_UNVERIFIED", "EXTERNAL_ATTESTED"):
        raise ValueError("unknown source job identity status")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    if (step_start is None) != (step_end is None) or (step_start is not None and
            (not isinstance(step_start, int) or not isinstance(step_end, int) or step_start < 0 or step_end < step_start)):
        raise ValueError("sample step range must be a valid inclusive integer interval")
    return {
        "schema": SIDECAR_SCHEMA,
        "sample_sha256": sha256_bytes(sample_bytes),
        "sample_bytes": len(sample_bytes),
        "sample_frames": sample_frames,
        "source_job_id": source_job_id,
        "source_job_id_status": source_job_id_status,
        "source_run_id": source_run_id,
        "step_start": step_start,
        "step_end": step_end,
        "provenance_status": "PRODUCER_OBSERVED",
        "independence_attestation": "NOT_ASSERTED",
        "metadata": metadata,
    }


def dumps(document):
    return json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"


def _result(status, checksum_valid=None, producer_metadata_status="NOT_PRESENT",
            required_configuration_fields_present=False, missing_configuration_fields=None,
            historical_match_status="NOT_VERIFIABLE", independence="NOT_ASSERTED"):
    return {
        "status": status,
        "checksum_valid": checksum_valid,
        "required_configuration_fields_present": required_configuration_fields_present,
        "missing_configuration_fields": list(missing_configuration_fields or []),
        "producer_metadata_status": producer_metadata_status,
        "historical_match_status": historical_match_status,
        "independence_attestation": independence,
    }


def resolve(sample_path, sidecar_path=None, expected_configuration=None):
    """Resolve integrity and configuration separately; neither SHA nor JSON authenticates a producer."""
    sample_path = Path(sample_path)
    sidecar_path = Path(sidecar_path) if sidecar_path else Path(str(sample_path) + ".provenance.json")
    if not sidecar_path.is_file():
        return _result("LEGACY_DECLARED_UNVERIFIED")
    try:
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sample_sha256, sample_bytes = sha256_file(sample_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _result("INVALID_SIDECAR", checksum_valid=False, producer_metadata_status="MALFORMED")
    if not isinstance(document, dict) or document.get("schema") != SIDECAR_SCHEMA:
        return _result("UNSUPPORTED_SIDECAR", checksum_valid=False, producer_metadata_status="UNSUPPORTED")
    checksum_valid = (document.get("sample_sha256") == sample_sha256 and
                      document.get("sample_bytes") == sample_bytes)
    if not checksum_valid:
        return _result("HASH_MISMATCH", checksum_valid=False, producer_metadata_status="SIDECAR_DECLARED")
    if not isinstance(document.get("sample_frames"), int) or document["sample_frames"] <= 0:
        return _result("INVALID_SIDECAR", checksum_valid=True, producer_metadata_status="MALFORMED")
    if document.get("provenance_status") != "PRODUCER_OBSERVED" or not document.get("source_job_id"):
        return _result("INVALID_SIDECAR", checksum_valid=True, producer_metadata_status="MALFORMED")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    missing = _configuration_missing(metadata)
    complete = not missing
    match_status = "NOT_VERIFIABLE"
    if expected_configuration is not None:
        if not isinstance(expected_configuration, dict) or not expected_configuration or any(
            key not in REQUIRED_CONFIGURATION_FIELDS for key in expected_configuration
        ):
            match_status = "NOT_VERIFIABLE"
        elif any(key not in metadata for key in expected_configuration):
            match_status = "NOT_VERIFIABLE"
        else:
            def matches(key):
                if key.endswith("_sha256"):
                    return metadata[key].lower() == expected_configuration[key].lower()
                return metadata[key] == expected_configuration[key]
            match_status = "MATCHED_DECLARED_VALUES" if all(matches(key) for key in expected_configuration) else "MISMATCH"
    status = ("CONFIGURATION_UNSTABLE" if metadata.get("configuration_stable_during_run") is False else
              "PRODUCER_METADATA_INCOMPLETE" if not complete else "PRODUCER_OBSERVED")
    result = _result(
        status,
        checksum_valid=True,
        producer_metadata_status="PRODUCER_OBSERVED",
        required_configuration_fields_present=complete,
        missing_configuration_fields=missing,
        historical_match_status=match_status,
    )
    result.update({
        "sample_sha256": document["sample_sha256"],
        "sample_frames": document["sample_frames"],
        "source_job_id": document["source_job_id"],
        "source_job_id_status": document.get("source_job_id_status", "RUN_LOCAL_UNVERIFIED"),
        "source_run_id": document.get("source_run_id", ""),
        "step_start": document.get("step_start"),
        "step_end": document.get("step_end"),
        "metadata": metadata,
    })
    if complete:
        canonical_configuration = json.dumps(
            {key: metadata[key] for key in REQUIRED_CONFIGURATION_FIELDS},
            sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8")
        result["configuration_sha256"] = sha256_bytes(canonical_configuration)
    return result


def validate_configuration_attestation(resolved, attestation):
    """Bind a declared external configuration record; this does not authenticate its signer."""
    if not isinstance(attestation, dict) or attestation.get("schema") != CONFIG_ATTESTATION_SCHEMA:
        return {"status": "NOT_ATTESTED", "configuration_attestation": "NOT_ASSERTED"}
    required = ("sample_sha256", "configuration_sha256", "attestor", "evidence_sha256")
    if any(not isinstance(attestation.get(key), str) or not attestation[key] for key in required):
        return {"status": "INVALID_ATTESTATION", "configuration_attestation": "NOT_ASSERTED"}
    if not resolved.get("checksum_valid") or not resolved.get("required_configuration_fields_present"):
        return {"status": "NOT_VERIFIABLE", "configuration_attestation": "NOT_ASSERTED"}
    if (attestation["sample_sha256"] != resolved.get("sample_sha256") or
        attestation["configuration_sha256"] != resolved.get("configuration_sha256")):
        return {"status": "ATTESTATION_MISMATCH", "configuration_attestation": "NOT_ASSERTED"}
    if not _valid_sha256(attestation["evidence_sha256"]):
        return {"status": "INVALID_ATTESTATION", "configuration_attestation": "NOT_ASSERTED"}
    return {
        "status": "EXTERNAL_ATTESTATION_DECLARED",
        "configuration_attestation": "EXTERNAL_ATTESTATION_DECLARED_UNAUTHENTICATED",
        "attestor": attestation["attestor"],
        "evidence_sha256": attestation["evidence_sha256"].lower(),
    }


def validate_independence_attestation(training, blind, attestation):
    """Validate declared provenance evidence; a successful shape check is unauthenticated, not proof."""
    if not isinstance(attestation, dict) or attestation.get("schema") != ATTESTATION_SCHEMA:
        return {"status": "NOT_ATTESTED", "independence_attestation": "NOT_ASSERTED"}
    required = (
        "training_sha256", "blind_sha256", "training_source_id", "blind_source_id",
        "training_job_id", "blind_job_id", "training_run_id", "blind_run_id",
        "training_step_start", "training_step_end", "blind_step_start", "blind_step_end",
        "attestor", "evidence_sha256",
    )
    if any(not isinstance(attestation.get(key), str) or not attestation[key]
           for key in ("training_sha256", "blind_sha256", "training_source_id", "blind_source_id",
                       "training_job_id", "blind_job_id", "training_run_id", "blind_run_id",
                       "attestor", "evidence_sha256")):
        return {"status": "INVALID_ATTESTATION", "independence_attestation": "NOT_ASSERTED"}
    step_keys = ("training_step_start", "training_step_end", "blind_step_start", "blind_step_end")
    if any(key not in attestation or not isinstance(attestation[key], int) for key in step_keys):
        return {"status": "INDEPENDENCE_INCONCLUSIVE", "independence_attestation": "NOT_ASSERTED"}
    if training.get("sample_sha256") != attestation["training_sha256"] or \
       blind.get("sample_sha256") != attestation["blind_sha256"]:
        return {"status": "ATTESTATION_SAMPLE_MISMATCH", "independence_attestation": "NOT_ASSERTED"}
    source_pairs = (
        (training.get("source_job_id"), attestation["training_job_id"]),
        (blind.get("source_job_id"), attestation["blind_job_id"]),
        (training.get("source_run_id"), attestation["training_run_id"]),
        (blind.get("source_run_id"), attestation["blind_run_id"]),
    )
    if any(not actual or actual != declared for actual, declared in source_pairs):
        return {"status": "INDEPENDENCE_INCONCLUSIVE", "independence_attestation": "NOT_ASSERTED"}
    if (training.get("source_run_id") != attestation["training_source_id"] or
        blind.get("source_run_id") != attestation["blind_source_id"]):
        return {"status": "INDEPENDENCE_INCONCLUSIVE", "independence_attestation": "NOT_ASSERTED"}
    if any(training.get(key) != attestation["training_" + key] for key in ("step_start", "step_end")) or \
       any(blind.get(key) != attestation["blind_" + key] for key in ("step_start", "step_end")):
        return {"status": "INDEPENDENCE_INCONCLUSIVE", "independence_attestation": "NOT_ASSERTED"}
    if (attestation["training_job_id"] == attestation["blind_job_id"] or
        attestation["training_run_id"] == attestation["blind_run_id"] or
        attestation["training_source_id"] == attestation["blind_source_id"]):
        return {"status": "ATTESTATION_SOURCE_COLLISION", "independence_attestation": "NOT_ASSERTED"}
    ranges = (attestation["training_step_start"], attestation["training_step_end"],
              attestation["blind_step_start"], attestation["blind_step_end"])
    if any(value < 0 for value in ranges) or ranges[1] < ranges[0] or ranges[3] < ranges[2]:
        return {"status": "INVALID_ATTESTATION", "independence_attestation": "NOT_ASSERTED"}
    if max(ranges[0], ranges[2]) <= min(ranges[1], ranges[3]):
        return {"status": "ATTESTATION_STEP_OVERLAP", "independence_attestation": "NOT_ASSERTED"}
    if not _valid_sha256(attestation["evidence_sha256"]):
        return {"status": "INVALID_ATTESTATION", "independence_attestation": "NOT_ASSERTED"}
    return {
        "status": "EXTERNAL_ATTESTATION_DECLARED",
        "independence_attestation": "EXTERNAL_ATTESTATION_DECLARED_UNAUTHENTICATED",
        "attestor": attestation["attestor"],
        "evidence_sha256": attestation["evidence_sha256"].lower(),
    }
