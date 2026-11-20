# R164 diagnostic readout

This opt-in development readout records full-bead phase and forces and, optionally, provisional ordinary-path centroid observables. It does not establish production acceptance, a real P11 event, a complete checkpoint, or conductivity. CUDA compilation, WPE runtime, GPU propagation, same-frame CPU comparison and ordinary-flow closure have not been verified here.

Prepare the frozen R161 phase with the matching target model, using a fresh directory:

```
python tools/r164_phase_to_restart.py R161_phase.npz restart_beads.xyz --temperature 300 --model-output model.xyz
```

Copy `examples/r164_diagnostic/run_0p2fs.in` or `run_0p1fs.in` to `run.in` in separate fresh run directories with `model.xyz`, `restart_beads.xyz`, and the exact target model as `qnep.txt`. These cards use 0.2 fs × 5 and 0.1 fs × 10, respectively; they are short diagnostics, not production authorization. The generated restart prints mass to four decimal places in amu; the matching model retains 17 significant digits. GPUMD uses model masses, so reconstruct momentum using the model mass. The existing native bead restart writer also uses four decimals for mass only.

The command `r164_rpmd r164_output 1` creates a fresh directory, with debug frames every step. A zero debug stride keeps initial and final frames only. A fourth argument `observer` creates a separate native centroid qNEP/PPPM observer; omit it for the phase-only run used in paired propagation checks. The main bead force route remains native GPUMD or WPE according to the build. The driver allows only `dump_thermo` and `dump_pimd_restart` alongside it and rejects other actions/properties or corrections, non-RPMD, unloaded beads, non-exact propagation, nonperiodic/nonorthogonal cells, and ambiguous or changed first-bead MIC branches.

`metadata.txt` records dimensions, units, R27 source identity, cell, temperature and fixed dt. Scalar time is computed as integer step × fixed dt from a zero start; it does not change GPUMD's accumulated simulation time. `source_identity.txt` records backend, model and `run.in` SHA-256 where readable, and the initial/step force-call mapping. Machine/GPU/driver identity is not recorded by the driver. The hash records source bytes at capture, not verification of the frozen target or any future run. The observer identity separately records model hash, spacing, and per-frame mesh/call identity.

Bead phase and output buffers use FP64 with 17-digit text serialization; native qNEP includes FP32 model/intermediate arrays and PPPM uses FP32 complex FFT meshes, while the indicated reductions and phase storage use FP64. These are implementation-level types, not a measured numerical error budget or a certification of WPE runtime precision.

At each whole step `scalar.txt` contains the ring energy components, total centroid momentum and four raw species number flows. The optional observer emits five separately marked unverified candidates: convection, local, mesh mechanical, charge relay and projection. It also emits source-derived local, mechanical, relay and projection image terms; the independent regular first-moment expression, regular-minus-b residual, and both site-rate minus energy-gradient power and site-rate plus native-force power. No tolerance or pass is inferred. When all five components are available, `B_UNVERIFIED` is their sum; `J2_UNVERIFIED` additionally requires the fixed R27 candidate. The R27 ID/hash, original support rows and units are recorded, but full row identity, local environment, model binding and the quantum mapping remain unverified. Unavailable values use explicit status labels, never placeholder zeroes. A successful calculation is a pointwise diagnostic, not a scientific gate pass.

Committed `frame_<step>.txt` files contain full-bead q, p, U, F. Corresponding `force_images_<step>.txt` records integer images in `q_lift = q_force + cell*image`. `geometric_chart_changes.txt` is written only for image changes between adjacent actual force frames; it counts changed bead-coordinate indices and does not localize a crossing or contain event energy/history. It is not a P11 event record. `COMPLETE` means only that diagnostic capture reached its last step and checked its output closes. Missing COMPLETE means capture is incomplete.

Convert a committed frame to a portable archive with `python tools/r164_frame_to_npz.py r164_output STEP frame.npz`; this checks the unique scalar row, final newline, step and fixed-dt time. The archive can be supplied to the accompanying `verify_ja_frame.py`. Compare observer-off and observer-on runs on the same initial phase, backend, model, PPPM, step and frame; cache isolation and same-frame force equivalence remain unverified until measured.

For fresh-cache diagnostic reinitialization from a committed frame:

```
python tools/r164_phase_relaunch.py r164_output STEP fresh_segment
```

The command preserves q/p, rejects lifts outside the force wrapper's one-cell range, creates a new NPZ and matching model/restart pair, and commits a parent receipt last. The receipt hashes the selected scalar row, the parent files at receipt creation and the new files. The scalar file can subsequently grow; the recorded parent identity does not verify a future execution. New run step 0 is the parent frame and must be deduplicated in comparisons. Caches are rebuilt, while time continuity, neighbor/event history and RNG state are not restored. Neither this workflow nor the existing bead restart is a complete seamless checkpoint; compare with an uninterrupted trajectory before claiming continuity.

Physical event localization, source-path energy transfer and event first moment remain unavailable. Full equilibrium sampling, long-run statistics, step/bead convergence, quantum-current mapping, and conductivity remain outside the validated scope.
