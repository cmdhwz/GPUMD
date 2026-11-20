# R164 phase diagnostics (development only)

This opt-in records the loaded ring-polymer phase and its native bead forces. It does not establish an R164 production flow, a physical event, a restart contract, or conductivity.

Example (paths are illustrative; run from a fresh working directory with the target qNEP and the existing GPUMD executable):

```
python tools/r164_phase_to_restart.py R161_phase.npz restart_beads.xyz --temperature 300 --model-output model.xyz
```

Use the exact matching source model and PPPM settings, then in `run.in` configure `ensemble rpmd 32 300`, `pimd_propagator exact`, `pimd_fix_com off`, `read_pimd_restart restart_beads.xyz`, a fixed `time_step`, and `r164_rpmd r164_output 1` before `run`. The directory must not exist. A fourth keyword argument `observer` opts into a separate native centroid qNEP/PPPM observer; without it this is the phase-only comparison. Check that the actual backend/model/PPPM and runtime precision match the intended calculation; this driver does not establish the model SHA or numerical agreement.

The driver permits only `dump_thermo` and `dump_pimd_restart` alongside it, and rejects other actions/properties and corrections, non-RPMD, unloaded beads, non-exact propagation, non-orthogonal/nonperiodic box, and a changed/ambiguous first-bead MIC branch. Positions are captured before the native bead force can wrap them. The observer is separate from the integration force; its convective term is an unverified diagnostic, not b. R27 output is candidate arithmetic only: full atom identity, model identity and environment are unverified. The fixed source hash and reference ID are in metadata. b, J2, real events and checkpoints are unavailable.

`metadata.txt` declares units; each committed `frame_<step>.txt` contains full-bead q, p, U, F; `scalar.txt` contains every captured step. `COMPLETE` means the diagnostic finished capturing its last step and checked output closes; it does not certify the run or any scientific gate. Absence of COMPLETE denotes incomplete capture. Convert a frame with `python tools/r164_frame_to_npz.py r164_output 0 frame0.npz`; the resulting arrays can be passed to the supplied `verify_ja_frame.py`. Compare no-observer and observer runs at identical initial state, backend, timestep, and output steps; cache independence and runtime precision are still unverified until this is done.

No CUDA compile, WPE runtime, or GPU propagation has been validated in this environment. Existing restart input checks do not establish a complete checkpoint or exact atom identity. This implementation performs no physical event detection or first-moment accounting.

The generated `restart_beads.xyz` prints mass to four decimal places in amu, while `model.xyz` retains 17-significant-digit masses. GPUMD uses model masses for dynamics; the restart reader checks the rounded mass against those model values. Reconstruct momenta using the model mass, not the rounded restart column.

Copy `examples/r164_diagnostic/run_0p2fs.in` or `run_0p1fs.in` to `run.in` in separate fresh run directories containing the converted `model.xyz`, `restart_beads.xyz`, and exact target model as `qnep.txt`. They are short diagnostic decks, not production run authorization. Add `observer` to the `r164_rpmd` line for a paired run in a separate directory. Observer relay and projection are provisional, and b and J2 are explicitly unverified candidates until independent GPU closure tests pass.

A diagnostic phase can be reinitialized by converting a committed frame to NPZ, then converting that NPZ to a fresh restart/model pair. This intentionally starts a new step/time identity; it is not a seamless chunk resume and cannot recover cache, RNG or event history. The existing bead restart output also lacks those fields.

For an explicitly fresh-cache diagnostic continuation from a committed frame, run:

```
python tools/r164_phase_relaunch.py r164_output STEP fresh_segment
```

This checks the parent frame/step/time against scalar metadata, rejects lifts outside the force wrapper's one-cell range, preserves q and p in a new NPZ plus `model.xyz` and `restart_beads.xyz`, and writes `parent_receipt.json` with SHA-256 hashes. A receipt appears only after all files are written. Prepare a new run card and model in `fresh_segment`; its step 0 is the parent frame and must be deduplicated when comparing trajectories. Caches are rebuilt; neighbor history, event history and random state are not restored. This workflow is not a seamless checkpoint and requires an uninterrupted-versus-reinitialized comparison before a continuity claim.

Sparse `force_images_<step>.txt` records the verified integer image for each bead and atom in `q_lift = q_force + cell*image`. `geometric_chart_changes.txt` is created only when adjacent actual force frames have different image indices; it counts changed bead-coordinate indices and does not localize a crossing or provide physical event energy/history. It is not a P11 event record.
