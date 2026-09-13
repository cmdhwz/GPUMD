# 4C Immediate Centroid qnep_full_a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the existing `hac_current qnep_full_a` operator to evaluate one validated centroid configuration per sampled PIMD/RPMD/TRPMD frame and feed the resulting complete three-component current into the existing HAC path.

**Architecture:** Keep the classical `qnep_full_a` path and all `legacy` paths unchanged. For immediate centroid mode, `HAC::end_of_step` requests diagnostic capture immediately before one `Force::compute` on `atom.position_per_atom` and `atom.velocity_per_atom`, then reuses the existing `compute_charge_rate`, dynamic K/R correction, full A-route assembly, component history, and `gpu_find_hac_3` path. The existing force-evaluation generation will invalidate single-frame caches at both single-frame and PIMD-batch qNEP evaluations, and an explicit current-frame diagnostic-validity marker will prevent stale bead data from being consumed.

**Tech Stack:** CUDA C++, existing `GPU_Vector`, qNEP/PPPM force APIs, existing HAC three-component correlator, pytest source-contract checks.

**Spec:** `C:\Users\Administrator\.codex\attachments\3daf69fd-ffad-4a36-8c10-76d4b147b6fe\pasted-text.txt` plus the approved方案 A clarification in the user message.

## Global Constraints

- `legacy` retains its existing behavior.
- Classical `qnep_full_a` continues to reuse the main force result.
- Immediate centroid `qnep_full_a` uses one centroid qNEP force evaluation per sampled frame and the same full-current assembly as classical mode.
- The production current is `J_conv + J_virial_existing + DeltaJ_q_total + J_A`, assembled before the existing three-component HAC history/correlator.
- Centroid diagnostic capture must be requested after the physical bead force and immediately before the centroid force; stale bead capture is invalid.
- Bead and centroid frames must not share a single-frame qNEP cache hit merely because step, time, or a buffer address matches.
- The auxiliary centroid force writes only dedicated centroid potential/force/virial buffers and must not overwrite PIMD bead force outputs.
- Both charge modes 1 and 2 and the existing fixed-cell small-box/multi-image qNEP path remain supported.
- `qnep_full_a + deferred centroid` remains rejected in this phase; existing legacy deferred centroid remains unchanged.
- No centroid batch API, bounded chunking, kernel fusion, or new electrostatic derivation is part of this phase.
- Preserve finite-value checks, component closure checks, output schemas, and existing GPU workspaces/plans.

---

### Task 1: Add the failing source-contract checks

**Files:**
- Create: `tests_pytest/test_qnep_full_a_centroid_immediate_contract.py`
- Read: `src/measure/hac.cu`, `src/force/nep_charge.cu`, `src/force/nep_charge.cuh`, `src/measure/qnep_projection.cu`

**Interfaces:**
- Consumes: the current source layout and the phase-A requirements above.
- Produces: deterministic, CUDA-free checks for immediate centroid routing and cache invalidation.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_qnep_full_a_centroid_is_immediate_and_deferred_is_still_rejected():
    source = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
    assert "qnep_full_a + deferred centroid is not implemented" in source
    assert "force.compute(" in source
    assert "request_charge_diagnostics_for_next_force()" in source


def test_qnep_force_generation_invalidates_single_frame_caches_for_batch_and_single_force():
    source = (ROOT / "src/force/nep_charge.cu").read_text(encoding="utf-8")
    assert "void NEP_Charge::begin_force_evaluation_()" in source
    assert source.count("begin_force_evaluation_();") >= 2
    assert "has_charge_diagnostics_for_current_force_frame" in (
        ROOT / "src/force/nep_charge.cuh"
    ).read_text(encoding="utf-8")


def test_full_a_assembly_rejects_stale_diagnostics():
    source = (ROOT / "src/measure/qnep_projection.cu").read_text(encoding="utf-8")
    assert "has_charge_diagnostics_for_current_force_frame" in source
```

- [ ] **Step 2: Run the focused test and verify the expected RED result**

Run: `python -m pytest tests_pytest/test_qnep_full_a_centroid_immediate_contract.py -q`

Expected: FAIL because the current source has no centroid qnep_full_a route, no shared force-generation helper, and no current-frame diagnostic validity method.

### Task 2: Make qNEP force state generation-aware

**Files:**
- Modify: `src/force/nep_charge.cuh`
- Modify: `src/force/nep_charge.cu`

**Interfaces:**
- Consumes: existing `force_evaluation_id_` cache invalidation and capture flags.
- Produces: private `NEP_Charge::begin_force_evaluation_()` used by both single-frame `compute` and `compute_pimd_batch`; public `has_charge_diagnostics_for_current_force_frame() const` used by full-current assembly.

- [ ] **Step 1: Add the state fields and declarations**

Add one private helper declaration and one validity marker tied to `force_evaluation_id_`:

```cpp
void begin_force_evaluation_();
bool charge_diagnostics_available_ = false;
unsigned long long charge_diagnostics_force_evaluation_id_ = 0;
```

Add this public accessor beside the existing raw diagnostic accessors:

```cpp
bool has_charge_diagnostics_for_current_force_frame() const
{
  return charge_diagnostics_available_ &&
    charge_diagnostics_force_evaluation_id_ == force_evaluation_id_;
}
```

- [ ] **Step 2: Implement the shared invalidation helper**

Move the existing single-frame cache invalidation from `NEP_Charge::compute` into `begin_force_evaluation_()`. Increment `force_evaluation_id_`, invalidate charge-rate, channel, full-A, and dynamic-q caches, clear the last dynamic-q validity flags, and mark current-frame diagnostics unavailable. Do not clear `charge_diagnostics_requested_`; a request must survive until the force evaluation that consumes it.

- [ ] **Step 3: Call the helper at the start of each actual qNEP force evaluation**

After argument/size validation and before qNEP work begins, call `begin_force_evaluation_()` from both `compute` and `compute_pimd_batch`. The single-frame and PIMD-batch paths therefore cannot retain a cache entry from the other evaluation kind.

- [ ] **Step 4: Mark a successful diagnostic capture with the current generation**

In both small-box and large-box single-frame electrostatic capture paths, after copying `charge_raw`, `D_raw`, and `D_projected`, set `charge_diagnostics_available_ = true` and `charge_diagnostics_force_evaluation_id_ = force_evaluation_id_`. Reset the marker in `reset_dynamic_charge_cache()` so no pre-run or prior-run buffers qualify.

- [ ] **Step 5: Run the focused test**

Run: `python -m pytest tests_pytest/test_qnep_full_a_centroid_immediate_contract.py -q`

Expected: the state-generation assertions pass; the HAC routing assertion remains RED until Task 3.

### Task 3: Route immediate centroid frames through the existing complete-current path

**Files:**
- Modify: `src/measure/hac.cu`

**Interfaces:**
- Consumes: `Force::compute`, the generation-aware qNEP capture marker, existing centroid buffers, and the existing full-current helper.
- Produces: immediate centroid `qnep_full_a` sampling with no deferred/batch fallback.

- [ ] **Step 1: Change qnep_full_a pre-run validation and storage**

Replace the unconditional classical-only rejection with these rules:

```cpp
const bool centroid_qnep_full_a = use_centroid_heat_flux_ != 0;
const bool ring_polymer_run = integrate.type >= 31 && integrate.type <= 33;
if (split_qnep_heat_by_type_ != 0) {
  PRINT_INPUT_ERROR("hac_current qnep_full_a does not support split HAC output.");
}
if (deferred_centroid_qnep_ != 0) {
  PRINT_INPUT_ERROR(
    "hac_current qnep_full_a + deferred centroid is not implemented in 4C-immediate; "
    "use immediate centroid or legacy deferred centroid.");
}
if (centroid_qnep_full_a && !ring_polymer_run) {
  PRINT_INPUT_ERROR("hac_current qnep_full_a centroid mode requires a PIMD/RPMD/TRPMD ensemble.");
}
```

Allow classical NVE/fixed-temperature NVT for the existing path and PIMD/RPMD/TRPMD for centroid mode. Keep the existing fixed-cell, orthogonal, periodic, one-qNEP, PPPM, and charge-mode 1/2 checks. When centroid mode is selected, allocate the existing dedicated centroid potential, force, and virial buffers in the qnep_full_a branch.

- [ ] **Step 2: Bind capture to the centroid evaluation**

In `HAC::pre_force`, retain the existing request for classical qnep_full_a, but skip that request when `use_centroid_heat_flux_ != 0`; the physical bead force must not be the source of the centroid diagnostic capture.

At the beginning of the qnep_full_a branch in `HAC::end_of_step`, select source pointers initialized to the classical main-force buffers. For centroid mode, increment the existing centroid counters, request charge diagnostics and per-atom virial capture immediately before one call to:

```cpp
qnep_full_a_qnep_->request_charge_diagnostics_for_next_force();
qnep_full_a_qnep_->request_peratom_virial_for_next_force();
force.compute(
  box,
  atom.position_per_atom,
  atom.type,
  group,
  centroid_potential_per_atom_,
  centroid_force_per_atom_,
  centroid_virial_per_atom_,
  atom.velocity_per_atom,
  atom.mass);
```

Check `has_charge_diagnostics_for_current_force_frame()` immediately after the call and fail the sampled HAC frame if it is false. Use `atom.position_per_atom`, `atom.unwrapped_position`, `atom.velocity_per_atom`, `centroid_potential_per_atom_`, and `centroid_virial_per_atom_` for all subsequent charge-rate, dynamic correction, full-current, and independent virial-component validation calls. The classical path must continue using `atom.potential_per_atom` and `atom.virial_per_atom` without an auxiliary force call.

- [ ] **Step 3: Preserve the existing assembly and history insertion**

Do not add a centroid correlator. Keep the current `compute_qnep_full_a_current(..., false, ...)`, component reconstruction, finite checks, closure checks, and `qnep_full_a_current_history_` writes unchanged except for the selected source buffers. This guarantees that centroid and classical currents enter the same three-component HAC history before `gpu_find_hac_3`.

- [ ] **Step 4: Add output metadata for the evaluated configuration**

In `post_run_qnep_full_a_`, add metadata identifying `current_configuration classical|centroid` and `centroid_evaluation immediate_single_frame|not_applicable`. Use a centroid-specific sampling-stage value for centroid segments while retaining the existing classical value. Keep the existing schema version and column layout unchanged.

- [ ] **Step 5: Run the focused test**

Run: `python -m pytest tests_pytest/test_qnep_full_a_centroid_immediate_contract.py -q`

Expected: PASS.

### Task 4: Reject stale full-current inputs at the shared assembly boundary

**Files:**
- Modify: `src/measure/qnep_projection.cu`

**Interfaces:**
- Consumes: `NEP_Charge::has_charge_diagnostics_for_current_force_frame() const`.
- Produces: a shared full-current guard that prevents stale raw `D`/charge-rate data from being used by HAC or diagnostic callers.

- [ ] **Step 1: Add the guard before reading raw `D` and raw charge rate**

In `compute_qnep_full_a_current`, after input-size checks and before obtaining `get_raw_D_reference()` and `get_raw_charge_rate_reference()`, return `false` when the qNEP object does not report diagnostics for its current force-evaluation generation.

- [ ] **Step 2: Run the focused test**

Run: `python -m pytest tests_pytest/test_qnep_full_a_centroid_immediate_contract.py -q`

Expected: PASS.

### Task 5: Verify the phase boundary and build status

**Files:**
- Inspect: all modified files and `CMakeLists.txt`
- Test: `tests_pytest/test_qnep_full_a_centroid_immediate_contract.py`

**Interfaces:**
- Consumes: the completed immediate-centroid source changes.
- Produces: evidence separating source checks, CUDA compilation, and runtime validation.

- [ ] **Step 1: Run source hygiene checks**

Run:

```powershell
git diff --check
python -m pytest tests_pytest/test_qnep_full_a_centroid_immediate_contract.py -q
```

Expected: both commands exit 0.

- [ ] **Step 2: Inspect the final diff and source boundaries**

Confirm that `legacy` branches, ordinary centroid HAC, legacy deferred centroid, qNEP projection PIMD rejection, and all batch force interfaces are unchanged except for the shared qNEP generation invalidation needed for isolation. Confirm no new `cudaMalloc`, FFT plan creation, or per-frame persistent intermediate storage was added.

- [ ] **Step 3: Attempt the configured CUDA build**

Use the repository's existing build command from `CMakeLists.txt`/the configured build directory. Record the exact command and exit status. A successful source test does not substitute for CUDA compilation.

- [ ] **Step 4: Report runtime gates honestly**

If a CUDA fixture is available, run short mode-1 and mode-2 fixed-cell small-box centroid cases and compare immediate centroid output against the same saved `R^c,V^c` passed through the classical single-frame entry. Also run the same bead restart with measurement disabled, immediate measurement, and legacy deferred measurement where supported; compare bead trajectories/energies to detect auxiliary-state contamination. If no CUDA fixture is available, report these as pending and do not label 4C-immediate PASS beyond source/build status.
