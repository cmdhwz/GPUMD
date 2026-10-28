.. _kw_rpmd_ja:

   single: rpmd_ja (keyword in run.in)

:attr:`rpmd_ja`
===============

Enable the optional reference-flow correction to the centroid mechanical heat current during a fixed-cell RPMD calculation.

Syntax
------

::

  rpmd_ja generate <reference_file> <temperature_K> <fd_step>
  rpmd_ja generate_sparse <reference_file> <temperature_K> <fd_step> <kernel_table>
  rpmd_ja on <reference_file>
  rpmd_ja off
  rpmd_ja sample <basename> <sample_interval>

``generate`` writes a dense reference from the current atom structure and the active single short-range NEP potential. ``generate_sparse`` uses a Chebyshev kernel table. For ordinary NEP, it writes sparse dynamical and site-gradient operators that must be rebound and checked with ``tools/rpmd_ja_sparse_check.py``. For qNEP charge mode 1 or 2 with PPPM, it directly writes the final v3 block reference and its stability sidecar. The table exporter supplies an analytic interpolation bound and a numerically checked arithmetic reserve; these are reported as approximation error budgets. Place either command after ``potential`` and before any ``ensemble`` or ``run`` command. Sparse runs report per-operator nonzero counts and GPU workspace bytes. They stop with an input error when that workspace does not fit; no dense fallback is attempted.

Build a kernel table directly with the repository exporter. A smaller ``U=8`` table may reduce the required polynomial degree only when :math:`\tau\sqrt{\text{spectral_bound}}\le 8`, where ``spectral_bound`` is the absolute CSR row-sum bound stored in the reference and used by the generator and checker. The exported table's interpolation and arithmetic error budgets must remain valid, and the stability check must pass; that check does not certify the kernel error budgets. The exporter does not automatically reduce ``U`` or shrink the kernel coverage::

  python tools/rpmd_ja_export_kernel.py --build-u 32 --output ja_kernel.tbl

When the coverage and error-budget conditions hold, explicitly select ``U=8`` with::

  python tools/rpmd_ja_export_kernel.py --build-u 8 --output ja_kernel_u8.tbl

Alternatively, an existing theory NPZ can be exported with ``python tools/rpmd_ja_export_kernel.py --npz kernel_theory.npz --output ja_kernel.tbl``.

In a reference-generation ``run.in``, put this after ``potential`` and before any ``ensemble`` or ``run`` command::

  rpmd_ja generate_sparse ja_reference_raw.bin 300 1e-4 ja_kernel.tbl

For qNEP charge mode 1 or 2 with PPPM, ``generate_sparse`` writes the ready-to-use v3 reference and its stability sidecar directly::

  rpmd_ja generate_sparse ja_reference.bin 300 1e-4 ja_kernel.tbl

The generated ``ja_reference.bin.stability`` sidecar must stay beside the reference. Use that path directly with ``rpmd_ja on ja_reference.bin``; the standalone Python preparation utility is not part of the production generation path.

For ordinary NEP sparse references, rebind the selected table and run the CPU stability check to create its certificate::

  python tools/rpmd_ja_sparse_check.py ja_reference_raw.bin --kernel-table ja_kernel.tbl --output ja_reference.bin
  python tools/rpmd_ja_sparse_check.py ja_reference.bin

With an equilibrated self-describing ``restart_beads.xyz``, use this measurement ``run.in`` order::

  potential nep.txt
  ensemble rpmd 32 300
  read_pimd_restart restart_beads.xyz
  rpmd_ja on ja_reference.bin
  compute_hac 5 1000 1 1
  run 10000

WPE supports ``rpmd_ja on`` with a previously generated native NEP/qNEP
reference when the WPE model and mechanical settings match its identity. Keep
the actual WPE ``potential``, ``kspace``, and ``pppm_mesh_spacing`` commands in
the measurement input. For WPE centroid HAC, GPUMD evaluates one private native
centroid observer per sample while WPE computes physical bead forces. Reference
generation, reference preparation, and graph fitting remain native GPUMD
operations; WPE measurement does not replace those workflows.
WPE qNEP measurement checks the active run-stage PPPM method and mesh spacing
against the private observer and selected reference. A stage, atom-count, or
bead-count identity that cannot be verified is rejected before sampling.

``rpmd_ja sample`` also works with WPE in fixed-cell canonical PIMD when the
active model is qNEP charge mode 1 or 2 with PPPM and no DFTD3 correction. The
active WPE stage and model file are checked before sampling. It records the
centroid structure and mean physical force across all beads after the physical
bead-force calculation. For example::

  rpmd_ja sample train 20

This writes ``train.samples.tmp`` and its ``.provenance.json`` sidecar. The
sidecar marks the WPE build backend and records the model-file SHA-256, run-input
fingerprint, WPE stage configuration fingerprint, and the sampled force source.
WPE ``rpmd_ja fit`` remains unavailable because it constructs a native
reference; use the separately generated native reference with ``rpmd_ja on``
instead.

Without a bead restart, first run a separate PIMD equilibration input with ``dump_pimd_restart`` to create ``restart_beads.xyz``; then start the measurement input above. Do not begin the RPMD measurement directly from the centroid-only ``model.xyz`` state.

The checker is an offline CPU positive-definiteness and certificate check. It must pass before the reference is used. A kernel-table export records its analytic interpolation budget and numerical arithmetic reserve; it does not certify that the fixed-reference approximation is accurate for arbitrary anharmonic motion.

The reference temperature must equal the constant RPMD target temperature. The option supports fixed-cell primitive RPMD with a fixed integration time step, using either an ordinary short-range NEP v1/v2 reference or a qNEP v3 reference for charge mode 1 or 2 with PPPM. Other ensembles or potentials, moving/deforming cells, adaptive time steps, DFT-D3, unverified electrostatics settings, force-modifying actions, qNEP full-A currents, and split/deferred centroid HAC are rejected. qNEP references are tied to their model and mechanical-configuration fingerprint; an ordinary NEP reference cannot be applied to qNEP. ``off`` is the default and leaves existing HAC behavior unchanged.

The centroid position is followed continuously across periodic wraps. The run stops with an error if any periodic per-step fractional jump is ambiguous (at least 0.45) or if the accumulated fractional displacement from the fixed reference reaches 0.45. This bounds the reference displacement branch and does not wrap accumulated diffusion back toward the reference.

Finite-temperature qNEP additive reference workflow
---------------------------------------------------

The optional additive workflow fits a local finite-temperature force response from complete, fixed-cell RPMD bead samples. It produces an auxiliary coefficient package; the ordinary qNEP reference generation and measurement paths remain available. The training system must contain all mobile atoms under one Hamiltonian, with the same atom ordering, masses, cell, target temperature, and bead count as the raw reference. Do not select a subset of atoms and treat it as a smaller system. The fitting data are split chronologically into training and held-out validation frames; correlated frames may make this validation optimistic, so independent blocks and material-level uncertainty remain necessary.

The native ``rpmd_ja fit_samples`` route produces a ``finite_temperature_additive`` reference, not the bare qNEP Taylor Hessian. Read its ``.fit_trace.txt`` in stage order: ``FIT_RESPONSE_PASS`` records held-out response acceptance, ``FULL_CERTIFICATE_PASS`` records the stored-matrix SPD certificate, and ``REFERENCE_WRITTEN`` confirms that the reference package and sidecar were written. ``QP_PASS`` only describes the current finite stability-cut set and does not mean the reference is ready. The ``.fit.txt`` report also records unconstrained and constrained training force residuals, the relative squared-residual cost change, design singular values, and response-ratio extrema. A failed candidate keeps diagnostic artifacts and must not be used with ``rpmd_ja on``.

For an independent one-parameter graph baseline, ``rpmd_ja graph_reference`` reuses a complete ``GPJASMP1`` trajectory and a qraw whose fixed ``R0`` matches the chronological training split. It computes the training fixed-``R0`` second moments (not a centered covariance), then solves the scalar ``mu`` stationarity equation on the stable interval. It bypasses the 60-parameter optimizer and does not require ``rho``, ridge, ``tau``, or a 60-dimensional KKT condition. Negative ``mu`` is allowed while the declared shifted matrix remains stable. A saved covariance state can supply ``baseline_mu`` and ``theta_baseline`` as a candidate using the optional ``baseline_state`` clause below. The candidate is accepted only after one fresh factorization/trace check against the current samples, qraw, COM convention, temperature, cutoff, epsilon, parameter map, and PPPM identity; a budget-exhausted final ``theta`` is never imported. The saved ``baseline_epsilon_constraint_status`` is a legacy optimizer proximity label, not evidence that the current scalar point is on the stability boundary; current feasibility and stationarity determine reuse. If the candidate fails this check, the command performs the bounded scalar search. The reported relative stationarity tolerance is fixed at ``1e-7``. The solve-residual expression is an explicitly diagnostic heuristic, not a trace-error bound and is not used to certify root or stability-bound KKT signs. If a stable-bound sign cannot be established, the calculation stops as numerically unconfirmed. A minimum Cholesky diagonal is a factorization diagnostic, not an estimate of ``lambda_min``. GPJASMP1 does not record its originating qNEP/PPPM identity; the fit trace leaves that provenance ``NOT_VERIFIABLE``. A matching qraw and active configuration are required, but cannot prove the historical sample source.

Run the command after ``potential``, ``kspace pppm``, and ``pppm_mesh_spacing`` (when the original run used a nondefault mesh), before any ensemble or run. Use the existing kernel table and qraw generated at the training ``R0``::

  rpmd_ja graph_reference samples.gpja ja_graph_diagnostic.bin 5.0 1e-8 0.15 1e-4 ja_kernel_u8.tbl ja_raw.qraw

To verify and reuse the scalar baseline candidate from an existing covariance state, append its path::

  rpmd_ja graph_reference samples.gpja ja_graph_diagnostic.bin 5.0 1e-8 0.15 1e-4 ja_kernel_u8.tbl ja_raw.qraw baseline_state ja_fit.covariance_state.txt

The command still performs the existing local additive assembly, qNEP ``prepare``, stored-``D`` SPD certificate, and readback. A numerically valid reference may be written with held-out response status ``FAIL`` or ``INCONCLUSIVE``; this is a diagnostic reference, not formal material acceptance. Use it only with explicit selection::

  rpmd_ja on ja_graph_diagnostic.bin diagnostic
  compute_hac 5 1000 1 1
  run 10000

The diagnostic identity is carried in the prepared reference, current/HAC output metadata, and synchronized material export. The ordinary ``rpmd_ja on ja_graph_diagnostic.bin`` path rejects it. GPUMD still advances beads with the physical qNEP/PPPM force; this option adds only the frozen ``x_c^T DeltaH v_c`` term to the reported current.

Analyze the resulting single ``heat_current_rpmd_ja.out`` segment without loading the reference matrices or replaying Hessian eigensolves::

  python tools/rpmd_ja_current_analyze.py heat_current_rpmd_ja.out --output-prefix ja_graph_diagnostic --epsilon-kappa 0.05 --block-frames 404 --max-lag-frames 100 --diagnostic

Choose and freeze ``epsilon_kappa`` before looking at the result. The analyzer streams the current file, checks its declared frame count against the 20-million correlation-work and 1-GiB estimated-host-memory limits, then allocates the numeric array. It writes ``RPJACURRENT2_REPORT`` JSON and an ``RPJACURRENT2_NPZ`` containing the five correlation curves, their native-normalized running integrals, and the paired B/2B block integrals. It reports the maximum ``JA`` correlation-decomposition residual but does not duplicate the derived decomposition/residual curves or bootstrap replicas in the NPZ. Its paired bootstrap resamples precomputed contiguous blocks of the base, correction, and total currents together; it never joins blocks before forming correlations. It requires ``B >= 4(max_lag+1)``, at least eight complete ``2B`` blocks, a fixed seed, and bounded correlation/bootstrap work. For each scale it preserves the raw block-bootstrap CI and separately reports a decision interval for the full-series DeltaKappa. The decision interval symmetrically expands the raw CI by the largest observed offset between the full-series estimate, common-prefix estimate, and corresponding B or 2B block mean. This deterministic segment adjustment is not an additional probability or confidence guarantee. Equivalence requires both expanded B and 2B intervals inside the frozen tolerance and a stable B/2B comparison. Non-equivalence requires both expanded intervals fully outside on the same side. Crossing or insufficient data is ``INCONCLUSIVE``; invalid-window reports keep unavailable numeric metadata as JSON ``null``. The diagnostic-reference report is always labeled ``DIAGNOSTIC_ONLY``. ``DeltaJ`` RMS is reported as an amplitude diagnostic and is not an equivalence criterion.

The separate offline ``fit_samples_covariance`` route reuses the same ``GPJASMP1`` samples, matching qraw, local additive package, ``prepare`` call, and ``J_A`` measurement chain. Its command is::

  rpmd_ja fit_samples_covariance <samples> <outfile> <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <qraw> shrinkage <rho> max_iter <N> [block_newton|joint_block|trust_region] [staged_barrier]

The bounded fixed-reference feasibility schedule has this additional form::

  rpmd_ja fit_samples_covariance <samples> <outfile> <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <qraw> shrinkage 0.1 max_iter <N> trust_region feasibility structured_basis blind_samples <blind_samples> blind_manifest <manifest>

This schedule must be launched through
``scripts/rpmd_ja_feasibility_preflight.py`` so the declared numerical CUDA
regression and real prepare/readback integration executable run before fitting.
The offline ``run.in`` may contain only ``potential``,
``pppm_mesh_spacing``, ``pimd_fix_com``, and ``kspace`` setup commands before
the fit command. Commands that perform calculations, including ``minimize``
and ``compute_phonon``, are rejected so their work cannot run outside the
declared feasibility budget. At native startup, qraw and kernel contents are
hashed in one streaming pass and both SHA-256 values are compared with the
manifest before sample traversal; the resulting FNV identities continue to
guard these files during fitting and reference preparation.
The ``RPJAFEAS9`` manifest must include ``m1_minimum_support_count`` as an
integer from 1 through 1000 and ``response_delta_tolerance`` as a finite,
positive response-change threshold. Both values are frozen before the fit,
copied into the execution receipt, and checked again by the native executable.
The same manifest also freezes ``curvature_primary_seed`` as
``8d12e5b7c3a964f1``, ``curvature_independent_seed`` as
``4f1bbcdc6762c9a5``, ``curvature_probe_cap`` as 128, and
``curvature_directional_tolerance`` as 0.1. The launcher copies these values
to the receipt, and native startup rejects a receipt or manifest that does
not match the executable's compiled curvature contract.
The manifest also declares ``candidate_cost_stop_rule`` as
``MAX_COMPLETE_OR_PARTIAL_LOWER_BOUND_SECONDS_V1`` and a finite
``candidate_cost_safety_factor`` from 1 through 10. Before each B/C candidate,
the controller compares the larger of the largest completed candidate cost and
the longest elapsed lower bound from an incomplete candidate, multiplied by
that factor, against both the remaining family allowance and the global fit
allowance after the validation reserve. Partial time never counts as a
completed cost sample. If the projection does not fit, that candidate is not
started and is reported as incomplete; this is not evidence that the reference
family failed. This estimate does not predict the final prepare/readback
duration.
The same manifest freezes ``lambda_factors`` to ``0.01,0.0001,0.000001``,
the wall and gradient limits (28800 seconds, 240 global gradients, 8 per A
branch, and 112 shared B/C gradients), the 0.25 validation reserve, and the
0.10 A wall fraction. It also freezes equal remaining B/C wall and gradient
splits. The solver's training and development uncertainty ledgers use at most
eight contiguous equal-frame-count blocks and require at least four. Candidate
selection additionally uses the training-derived common block length ``B``
and adjacent length ``2B`` over the same complete development prefix for every
candidate. Each pair is compared with paired differences at both scales; a
two-standard-error winner must agree at ``B`` and ``2B``. A disagreement or a
resolved winner at only one scale is ``DATA_INCONCLUSIVE``. If both scales are
unresolved, the frozen M0 tie preference applies. The existing frozen
max-z-adjusted drift test remains in force; only a predeclared drift threshold
exceedance rejects the data. The launcher writes these rules into the receipt;
native startup rejects any mismatch with the executable contract.
The manifest also fixes ``statistics_storage_budget_bytes`` at 536870912.
Fixed-R0 admission statistics use mass-weighted internal physical probes and
parameter channels, with B and 2B summaries sharing the same covered prefix.
The pre-allocation peak estimate includes the simultaneously live sample,
covariance, block, optimizer, and candidate-score storage. Before the training
block length is known, it bounds block means at one mean per sample frame.
Parameter covariance means are consumed incrementally rather than retained as
a second full set.
A larger input is reported as ``DATA_INCONCLUSIVE`` before fitting rather than
truncated at a hidden frame-count cap. Statistical frame and lag loops observe
the feasibility budget; exhaustion is reported as ``DATA_INCONCLUSIVE``.
For each of ``training`` and ``blind``, RPJAFEAS9 requires
``<sample>_model_sha256``, ``<sample>_physical_force_configuration_sha256``,
and ``<sample>_integrator_configuration_sha256``. The producer's RPJAPROV2
sidecar records the model-content digest alongside the declared historical
configuration digests. Native startup hashes the active NEP model contents and
binds that digest to the manifest's training and blind identities and the
qraw's recorded model identity. A matching sidecar binds the sample bytes
separately from those declared historical digests. The receipt reports byte
integrity, declared-configuration matching, and source-job/run relationship as
separate statuses. A missing sidecar is accepted only when all three manifest
identities for that sample explicitly say ``NOT_VERIFIABLE``; concrete
identities require matching evidence, and partial or downgraded declarations
fail closed. The current offline ``run.in`` is not treated as evidence of how
an existing sample was generated. Distinct sidecar identities are declarations
and do not by themselves authenticate source independence.
After the fixed M0/M1 graphs are built, native preflight performs a size-only
walk of the frozen site blocks and checks disk space before fitting. It uses
the larger exact serialized additive-package size, a qraw-sized upper bound
for the final reference, and a 1 GiB margin. Host readback and minimum prepare
GPU allocations are also estimated before fitting; the prepare stage repeats
its dynamic peak-memory check when it starts.
On Linux, the host-memory check also respects an available cgroup memory
limit. During prepare, the written reference and stability sidecar are read
back before success is reported. Three fixed vector actions are checked for
stored D and each stored ``B_alpha^T`` against the precompression operators;
failure removes the reference and sidecar.
Adjacent-lambda response change is measured as
``||P_right-P_left||_F / max(||P_left||_F, ||P_right||_F)`` using the full
predicted response matrices in the frozen common-probe order. This fixed
``FROBENIUS_RELATIVE_MAX_NORM_V1`` metric includes every cross term and does
not normalize individual entries. The formula and near-zero rule are fixed
by the native executable SHA-256 pinned in the manifest and are written to
the fit trace. If the probe orders or matrix dimensions
differ, any matrix entry or norm is non-finite, or both global matrix norms are
at most the smallest normal double value, the comparison is ``INCONCLUSIVE``.
The existing response SPD gate also rejects a predicted matrix whose minimum
eigenvalue is at most ``1e-12`` of its maximum eigenvalue. The scalar response
error difference is retained only as a diagnostic. The response-change
threshold is independent of the absolute candidate ``response_tolerance``.
If adjacent-lambda sensitivity is resolved as
``REGULARIZATION_SENSITIVE`` or remains ``INCONCLUSIVE``, the candidate is
recorded but blind data stays unopened and no reference is published. Raw
local terms below the support threshold are removed before operator
residualization. M0 remains the shared base of M1.

After A0/A1, B and C each receive an equal wall-clock cap from the time left
after the reserved blind/prepare allowance. The B/C gradient cap remains
shared as well. Development likelihood uncertainty uses at most eight
contiguous fixed-R0 blocks, requires at least four blocks, and reports a
two-standard-error block-scatter interval. Adjacent-lambda sensitivity uses
paired block differences and the manifest-frozen ``response_delta_tolerance``
as its response-change scale. Cross-family selection compares paired score
differences over the same development prefix at ``B`` and ``2B`` with a
two-standard-error rule; only a same resolved winner at both scales selects a
candidate. A scale disagreement is ``DATA_INCONCLUSIVE``; when both
comparisons are unresolved, the frozen M0 preference applies. This block
scatter is not a universal confidence interval. Converged baseline/candidate soft Ritz directions that pass the
``1e-6`` relative-residual check are unioned and all selectable candidates are
rescored on the same frozen probe set before candidate selection and blind
data access.

On a CUDA development host, configure and build the production executable and
both RPMD-JA CUDA regressions with assertions enabled::

  cmake -S . -B build -DBUILD_TESTING=ON -DBUILD_RPMD_JA_CUDA_REGRESSIONS=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build build --config Debug --target gpumd rpmd_ja_native_fit_cuda_test rpmd_ja_fit_prepare_integration_cuda_test
  ctest --test-dir build -C Debug --output-on-failure -R "rpmd_ja_(native_fit|fit_prepare_integration)_cuda_test"

The numerical regression is a standalone CUDA executable. The integration
regression links the production GPUMD sources except the application entry point
and the native-fit translation unit (which is included in the test translation
unit), so it exercises the real prepare, reference readback, and CUDA matrix
code. Point the feasibility manifest at both built test executables. Build
commands are recorded here but still require execution on the target CUDA
toolchain before their results can count as regression evidence. The preflight
receipt records executable identities and statuses; manifest text alone is not
execution evidence.

Feasibility launcher, resume, and cost baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the launcher from the repository root. Its complete command-line form is
``python scripts/rpmd_ja_feasibility_preflight.py [--resume] <manifest> <run-directory>``.
For example::

  python scripts/rpmd_ja_feasibility_preflight.py runs/job17/manifest.txt runs/job17

The launcher runs both CUDA regression executables declared by the manifest,
writes its receipt and logs in the run directory, then launches GPUMD. It has
no check-only mode. A new run requires unused output and checkpoint paths.
The manifest's ``native_output_path`` must match the fit command in ``run.in``.
Its ``checkpoint_path`` identifies the outer controller journal; a relative
value is resolved from the run directory, and the native solver checkpoint is
stored beside it with ``.solver`` appended. Preserve both checkpoints and the
candidate-state files for recovery.

Resume with the same manifest file and run directory::

  python scripts/rpmd_ja_feasibility_preflight.py --resume runs/job17/manifest.txt runs/job17

The preflight revalidates the manifest, inputs, checkpoint identity, all
resume artifacts, and both CUDA regression executables before launching
GPUMD. The controller journal must be intact and match the original manifest
and job ID. Interrupted A0/A1/B/C fitting resumes from the saved scheduler
and solver state. After selection freezes, recovery restores the exact chosen
candidate and development response. At ``FREEZE``, the journal branch/family
remains the scheduler cursor: ``B2/M0`` when no M1 candidate was added, or
``C2/M1`` after M1 candidates were evaluated. The selected candidate is recorded
separately, so an M0 winner can retain a ``C2/M1`` cursor. If blind validation
was interrupted, it may be repeated only for that candidate with the frozen
probes; after a passing blind result, the persisted response is reused without reselecting. A
``PUBLISHING`` resume accepts only reference/sidecar bytes matching the
committed stage digests and can finish either rename of that same pair. A
``PUBLISHED`` resume verifies the final reference, sidecar, fit diagnostics,
trace, and response snapshots, then exits without modifying those sealed
files. Unknown, changed, or foreign artifacts fail closed. Resume never resets
the cumulative wall-time or gradient budget. An in-flight operation is charged
from its recorded start time through restart, including stopped time; an
in-flight gradient is conservatively charged once. A long stop can exhaust
the remaining wall budget. The manifest bytes and input identities must remain
unchanged, so changing any manifest field requires a new run directory, output,
and checkpoint.

The manifest protocol is ``RPJAFEAS9`` and the execution receipt is
``RPJAPREFLIGHT5``. The outer restart journal is ``RPJAOUTER5`` and the solver
checkpoint is ``RPJACKPT4``. Older journal/checkpoint versions are rejected,
not migrated. The outer journal binds the selected candidate, final probe set,
development/blind response snapshots, publication-stage digests, and final
artifact digests. In-flight controller operations record their start time for
wall-budget reconciliation. Fit locking uses a persistent lock file with an
operating-system lock; the OS releases ownership when the process exits, so a
stale file does not prevent recovery and is never removed to guess whether
another job is live.
The journal path is exactly the manifest's ``checkpoint_path`` after resolving
it against the run directory.

An optional historical A0 import uses the paired fields
``a0_reuse_package_path`` and ``a0_reuse_package_sha256``. Both must be present
or both omitted. A completed job exports one atomic
``<native_output_path>.a0_reuse.record`` package containing the exact source
manifest, fit-state, and record byte snapshots; the manifest digest field is
the SHA-256 of the whole package. A staged package is committed without
replacing existing evidence, so a process interruption leaves no partial
final package and resume can write a fresh stage. Only a converged A0 fit with
complete development-response statistics is eligible. A failed development response
may still be imported as a diagnostic; it is recomputed for the new job.

The import requires matching input content digests, model/qraw, fixed-R0 and
sampling identity, graph, objective/scales, sampling and statistics rules,
budgets, executable, and machine fingerprint. Output/job paths and the A0
import-control fields are excluded from the canonical comparison so a new job
can consume the source package. Historical optimizer and fit costs remain
source metadata with their origin job ID. The current snapshot-export job ID is
stored separately. Current import validation, numerical replay, baseline/factor
rebuild, and response scoring are recorded separately and charged to the new
job's wall budget. Nonconverged A0 states are not
importable. Same-job resume remains the outer-journal protocol and is not an
A0 import.

Before blind validation, the manifest and receipt also freeze a performance
stop heuristic for analytic-curvature trust-region candidates. After three
successful curvature builds at unchanged ``tau``, the candidate stops as
``NUMERICAL_INCONCLUSIVE_PERFORMANCE_STOP`` when their measured host wall time
exceeds one maximum measured full exact-gradient wall time from that same
``tau`` window and relative objective improvement is below ``0.001``. Relative
improvement is normalized by ``max(1, abs(window-start objective))``. A
``tau`` change starts a new window. The normal KKT/convergence test runs first,
and this heuristic does not change the KKT gate or show that a reference family
has no solution. Timings are steady-clock host wall measurements, not GPU event
times; objective, gradient, factorization, curvature, and response costs are
recorded, including partial/failed work.

``RPJACOST1`` is an optional measured-cost baseline. With no matched prior
measurements, set ``cost_baseline_status "NOT_PROVIDED"`` (or omit that
status) and omit the baseline path and hash. Unknown costs remain unknown, not
zero, and this does not prevent startup; the hard wall/gradient budgets still
apply. The repository does not provide a baseline-generation command. For a
verified baseline, create a text file from completed measurements for the same
machine and problem dimensions, using the largest completed measurement for
each field::

  RPJACOST1
  machine_fingerprint "<64 lowercase hex characters>"
  atom_count "<integer>"
  sample_frames "<integer>"
  training_frames "<integer>"
  development_frames "<integer>"
  m0_parameter_count "<integer>"
  m1_parameter_count "<integer>"
  candidate_seconds "<positive seconds>"
  fit_seconds "<positive seconds>"
  curvature_seconds "<positive seconds>"
  blind_seconds "<positive seconds>"
  package_write_seconds "<positive seconds>"
  prepare_seconds "<positive seconds>"
  readback_seconds "<positive seconds>"

Obtain ``machine_fingerprint`` with the launcher's helper and the same visible
GPU inventory/environment used for the run::

  python -c "import importlib.util,subprocess; p='scripts/rpmd_ja_feasibility_preflight.py'; s=importlib.util.spec_from_file_location('rpmd_ja_pf',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); g=subprocess.run(['nvidia-smi','-L'],check=True,capture_output=True,text=True).stdout; print(m.machine_fingerprint(g))"

Use ``sample_frames`` from the training sample; the frozen split is
``training_frames = 2 * sample_frames // 3`` and the remainder is
``development_frames``. Use the actual atom count, type-derived M0 parameter
count ``6 * ntypes * (ntypes + 1) // 2`` (``ntypes`` is the number of unique
atom types in the training sample), and M1 parameter count for that problem,
not estimates. For timing,
use only ``MEASURED_CANDIDATE_COST`` rows with ``complete_cost_sample 1`` and
``MEASURED_STAGE_COST`` rows with ``status PASS`` in a successful
``.fit_trace.txt``. ``candidate_seconds`` is the largest completed candidate
``total_seconds``. For each complete candidate row, derive ``curvature_seconds``
as the smaller of ``total_seconds`` and ``curvature_seconds`` in that row, and
``fit_seconds`` as ``total_seconds - curvature_seconds``; use the largest
derived value for each. Use the largest passing stage duration for blind,
package write, prepare, and readback. Do not substitute partial elapsed lower
bounds or fabricate a missing measurement.

Bind the file in the manifest, with ``cost_baseline_path`` resolved relative
to the run directory and ``cost_baseline_sha256`` equal to its SHA-256::

  cost_baseline_status "VERIFIED"
  cost_baseline_path "cost_baseline.txt"
  cost_baseline_sha256 "<64 lowercase hex characters>"

From the directory containing the baseline file, compute its digest on
PowerShell with::

  (Get-FileHash .\cost_baseline.txt -Algorithm SHA256).Hash.ToLowerInvariant()

Preflight checks that digest before and after the CUDA regressions and binds the
baseline to the machine fingerprint, atom/sample/train/development sizes, and
M0/M1 parameter counts. The native executable checks those dimensions again.

Resource checks are estimates with explicit limits. The Python preflight first
requires the input-file sizes plus a 1 GiB free-disk floor. Native preflight
then reserves known simultaneous host/disk objects, including M0/M1 graph
working sets, package staging copies, generated qraw when applicable, and
reference publication copies; available host/disk capacity is rechecked before
package write, prepare, and readback. These known-object lifecycle bounds are
not complete allocator/OS/runtime certificates
(``host_allocator_peak_complete=0``). The early GPU figure is a lower bound;
prepare queries the cuSOLVER POTRF and SVD workspaces and checks its known live
GPU allocations before the main prepare allocations. Neither estimate replaces
the runtime capacity checks or constitutes a material-validation result.

The two optional tokens can appear in either order. At most one step mode may
be supplied. ``staged_barrier`` is independent of the step mode.

Without a step-mode token, covariance fitting keeps its BFGS direction. The
``block_newton`` token preserves the legacy direction: it replaces the six
coordinates of the parameter group containing the largest absolute
scaled-gradient component when that group's support is at most 32 atoms and its
conditional Hessian passes the existing numerical checks. The opt-in
``joint_block`` token ranks groups by their maximum absolute scaled-gradient
component, requires the dominant group's support to be at most 32 atoms, then
selects up to three groups while the combined atom support stays within 32. It
builds one joint Hessian, including cross-group terms, from one factor pair and
solves for the union-support RHSs. For ``DOMINANT_SUPPORT_CAP``, it retains that
selection status and alternates a committed ``GROUP_BFGS`` step with a global
``BFGS`` rescue step, beginning with ``GROUP_BFGS``. The rescue uses the shared
full-parameter history and does not update per-group histories; attribution
records its actual nonzero groups and parameters. Rejected trials do not
advance the schedule, and a ``tau`` stage change resets it. Each group keeps an
independent six-dimensional history per ``tau`` stage, seeded only by its first
curvature-valid committed pure-group secant. Joint and mixed-group steps do
not update these histories. If a finite descent direction or representable
step is unavailable, the optimizer falls back to global BFGS; non-finite and
CUDA failures remain fatal.

The opt-in ``trust_region`` token solves the quadratic subproblem over all
scaled parameters using the inverse-BFGS matrix as the inverse model. It checks
that matrix for positive definiteness and resets the scale model to a positive
identity if the spectrum is not reliable. Candidate acceptance uses actual
over predicted reduction; a rejected candidate shrinks the radius and resolves
the subproblem below the rejected step. The actual scaled step is reconstructed
from the rounded parameters and must fit inside the radius. A full candidate
gradient is evaluated only after the objective ratio passes, and the point is
committed only after the gradient solve passes. Actual objective changes use
paired log ratios from the retained Cholesky diagonals and a local arithmetic
resolution bound for those stored factors and weighted parameter terms; this
bound does not certify Cholesky accuracy. If the model decrease is resolved but
the actual change is not, the trial is rejected and the radius is reduced for a
retry. ``NUMERICAL_STAGNATION`` is reserved for an unresolved model decrease.

Without ``staged_barrier``, every barrier stage keeps the existing scaled-KKT
gate of ``1e-6``. With it, intermediate stages use
``max(1e-6, 1e-2 * tau)``; the final stage still requires ``1e-6`` and the
existing small-barrier condition. The ``tau`` reduction remains a factor of
``0.2``. The fit trace reports per-group ``xx+yy+zz`` prior-mixing diagnostics
and contiguous training-block statistics about the same fixed training ``R0``.
The reported block spread is diagnostic only, not a confidence interval, and
the program keeps the positive ``rho`` supplied by the command.

Whenever ``joint_block`` is enabled, the shared full-parameter history retains
at most eight recent curvature-valid committed secants from pure
``GROUP_BFGS``, global ``BFGS``, and small-support ``JOINT_BLOCK`` steps. Each
pair uses the full scaled step and gradient change. After a complete-gradient
candidate passes the commit gates, the inverse Hessian is rebuilt from the
newest ``gamma = sTy / yTy`` times the identity and the retained secants are
replayed oldest to newest. This refreshes the background scale in directions
untouched by those secants; it does not guarantee improved conditioning or KKT
convergence. ``bfgs_recent_pairs`` reports the retained count. Invalid
curvature and non-descent resets clear the recent pairs while retaining the
latest valid positive scale; a ``tau`` stage change clears them and restarts
from identity. Rejected candidates do not change the history. The independent
six-dimensional group histories still update only after pure ``GROUP_BFGS``
steps.

The optimizer summary, fit report, and covariance state also record
``trace_solve_batch`` (128 columns), ``trace_residual_batch`` (32 columns), and
``trace_solve_calls`` (submitted wide triangular-solve calls, including a
failing call). Residual checks still run in the existing 32-column blocks, and
``solve_rhs_columns`` counts columns from successful wide solves. The
diagnostic ``rhs_columns_checked`` counts completed 32-column residual blocks
and may be smaller if a strict check fails partway through a wide solve.
The legacy ``worst_solve_relative_residual`` remains scoped to the most recent
gradient/trace evaluation. Separate ``run_trace_residual_*`` fields accumulate
factor and checked-original residual maxima, matrix kinds and batch locations,
the number of checked and failed logical batches, and the first failure. This
run summary includes baseline, initialization, and optimizer trace solves,
including a rejected batch, but excludes diagnostic probe solves. If the
original matrix was never checked, ``original_any_checked`` is 0, its matrix
kind is ``NOT_CHECKED``, and its residual is NaN.

For ``joint_block`` runs using ``GROUP_BFGS`` or ``BFGS`` directions, bounded
objective-only expansion proposes steps up to alpha 4, then computes a full
gradient at the selected candidate's factors. A failed or worse expansion
returns to the best objective-pass proposal and re-evaluates it before the
gradient. After every accepted Armijo objective trial, all direction families
require a complete validated gradient and reject every strictly positive
candidate slope; no positive-slope candidate is committed. For joint-block
global ``BFGS`` and ``GROUP_BFGS`` directions only, a directional-secant
proposal may select the next trial alpha, clamped to ``[0.1 alpha, 0.5 alpha]``;
an invalid proposal falls back to halving. Default BFGS, legacy
``block_newton``, and spectral ``JOINT_BLOCK`` directions retain their existing
halving behavior. The trace records the rejection reason, absolute slope ratio,
and actual shrink ratio. This is not a strong-Wolfe line search. Objective
proposals alone never satisfy commit gates.
``line_search_evaluations`` and ``line_search_objective_seconds`` cover
line-search objective evaluations and time, including re-evaluations after a
failed or worse expansion. ``line_search_gradient_calls`` and
``line_search_gradient_seconds`` cover candidate full-gradient work, including
residual-rejected candidates. ``initialization_gradient_calls`` counts the
initial or selected-start gradient, and ``full_gradient_evaluations`` is the sum
of those initialization and candidate calls. Tau-stage cache recomposition does
not count as a new gradient evaluation. ``trace_solve_calls`` remains reported
separately. Group BFGS does not use the joint ray proposal and reports its
approximate model as unavailable.
All modes retain the existing full-gradient residual, Armijo, KKT, held-out
response, and final SPD acceptance gates; the BFGS spectrum and secant
diagnostics do not alter any gate or optimizer state.

Before each direction is used, the trace, covariance state, and fit report save
a ``DIRECTION_BEFORE_STEP`` diagnostic of the global parameter-space inverse
BFGS matrix after any non-descent reset. The matrix is scaled and symmetrized
for a small eigensolve; status distinguishes a reliably verified SPD spectrum,
an indefinite matrix, an unresolved near-singular spectrum, and an unverified
result. Condition is reported only for a reliable SPD spectrum. The iteration,
barrier ``tau``, matrix scale, relative asymmetry, eigen residual, and diagnostic
time identify that snapshot; ``inverse_bfgs_spectrum_total_seconds`` accumulates
the cost across snapshots. More than 60 parameters reports
``NOT_EVALUATED_PARAMETER_COUNT`` without an eigensolve. State and fit files
name the run summary ``run_trace_residual_*``; the optimizer-summary trace uses
``trace_residual_*``. The spectrum is diagnostic only: it does not gate, reset,
or modify the direction, and it adds no physical Hessian or material solve. Each
committed step also reports ``bfgs_sy``, the existing ``bfgs_s_norm2`` and
``bfgs_y_norm2`` values (respectively :math:`s^Ts` and :math:`y^Ty`), and their
normalized secant cosine.

For a successful ``JOINT_BLOCK`` direction, the optimizer also forms the small spectra of ``sqrt(V) L sqrt(V)`` for the local ``D`` and ``D-epsilon I`` inverse probes. Their exact log-determinant ray model proposes an initial alpha, capped at 90% of its estimated SPD boundary and by the existing step policy (up to 4 only from an original alpha of 1). This is only a proposal: the full objective, full-gradient residual, Armijo, directional-overshoot, and commit checks still decide every trial. Finite spectral or slope-consistency failures use the existing initial step and line search. No full-space inverse or solve is added.

The trace and covariance state record selected group IDs and supports, direction/fallback choice, factor/RHS/time costs, per-group and global gradient changes, and the last committed step's joint-model prediction ``g + H s`` at the actual accepted step. The ray proposal's eigen residuals, minimum ray eigenvalues, boundary, initial and proposed alphas, model change/slope, small-eigen and scalar-evaluation costs, and actual line outcome are recorded separately for the latest attempt and latest committed proposal. These are local optimizer diagnostics, not Hessian, response, or SPD certification. Rejected trials do not replace committed-step records. The fixed-reference objective, ``rho``, barrier schedule, and KKT threshold are unchanged.

This route is opt-in; ``fit_samples`` continues to use the force least-squares/QP method. The covariance method computes training statistics only from the first two thirds of frames and minimizes the Gaussian fluctuation objective with a graph-Laplacian baseline and a positive ``rho`` shrinkage prior. It searches the full feasible interval for the baseline parameter, including negative values when the raw reference is already stable. It uses exact Cholesky factors in the translation-complement space and analytic parameter gradients. BFGS coordinates are scaled from the baseline matrix and each parameter basis matrix; the reports include both the scaled KKT residual and raw parameter-gradient norm. When the baseline is epsilon-boundary-active, initialization first halves the original positive graph-direction step until the full objective decreases, then doubles from that point while the objective continues to decrease. The trace labels these phases ``phase=SHRINK`` and ``phase=EXPAND`` and records the stop reason, trial delta and objective, and selected start. ``theta_baseline`` remains the prior center and ``theta_start`` records the best selected start. ``initialization_status``, ``initialization_evaluations``, ``start_graph_delta``, ``start_objective``, and ``start_raw_gradient_inf_norm`` describe that search. The start report also records the full-gradient solve status, scaled KKT residual, and scaled component gradient norms. The state and fit reports expose ``bfgs_seed_gamma`` and ``bfgs_seed_source``. ``max_iter`` is the optimizer iteration budget; line-search objective evaluations and gradient-residual rejects are reported separately. The first curvature-valid secant in each fixed-``tau`` stage seeds the inverse Hessian with ``gamma = sTy / yTy`` before the inverse-BFGS update; same-stage resets reuse the latest valid positive gamma, while a ``tau`` change restarts from the identity. The ``.fit_trace.txt`` distinguishes ``SHIFTED_NOT_SPD``, ``ARMIJO_REJECT``, ``ARMIJO_ACCEPTED``, ``GRADIENT_RESIDUAL_REJECT``, and ``STEP_UNDERFLOW``. ``ARMIJO_ACCEPTED`` records only the objective test; ``STEP_COMMITTED`` is written only after full gradient solves pass and optimizer state is updated. A failed line search also records a bounded directional finite-difference check. That check compares against the analytic descent prediction only after successive halvings give two consecutive finite-difference changes within 1%; otherwise it reports an ``INCONCLUSIVE_*`` status and preserves the original line-search failure. The saved-point ``diagnose_covariance`` report uses ``cbrt(machine_epsilon) * max(1, abs(z[k]))`` for its selected coordinate, the actual scaled parameter change as the difference denominator, and one shared budget of at most 16 objective evaluations. It tries the gradient-predicted descent side first and tries the opposite side at the same step when needed for feasibility. Each ``directional_fd_probe`` line reports the side, nominal and effective steps, feasibility, objective change, finite-difference slope, stability change, and stable-halving count. ``directional_fd_one_percent_check`` separately reports whether its stabilized slope agrees with the analytic slope within 1%, while ``directional_fd_status`` retains the existing broad consistency classification. ``GRADIENT_CONSISTENCY_SUSPECT`` means a stabilized finite difference disagreed with the analytic descent prediction, and is not convergence. The numerical log-determinant barrier used to enforce ``D-epsilon I > 0`` is an optimizer parameter and is not the shrinkage strength. The optimizer's convergence status is separate from held-out response acceptance and the final stored-D certificate. Read ``baseline_mu``, ``baseline_epsilon_constraint_status``, objective terms, scaled KKT residual, raw gradient norm, barrier gap estimate, factorization count, solve RHS count, and elapsed time from the ``.fit.txt`` report. On optimizer or response failure, ``.covariance_state.txt`` records the baseline, starting, and candidate parameter vectors, optimizer rejection counts, and training statistics; the method does not write an accepted reference on that path.

The boundary-start initialization is capped at 16 full-objective evaluations and consumes the ordinary objective budget. ``initialization_evaluations`` includes the baseline evaluation, all direction-search trials, and the selected-start gradient refresh when a boundary search ran. ``line_search_evaluations`` counts objective evaluations, while ``line_search_trials`` also counts parameter updates stopped by underflow before evaluation. Every positive candidate slope is rejected after its complete validated gradient. Joint ``GROUP_BFGS`` and global ``BFGS`` directions use the bounded secant proposal only to choose a smaller next trial; all other direction families halve. The failure-only directional check uses at most 16 additional objective evaluations and reports ``INCONCLUSIVE_BUDGET`` if no budget remains.

For ``joint_block``, the shared recent history applies after fully committed
``GROUP_BFGS``, global ``BFGS``, and small-support ``JOINT_BLOCK`` steps; each
uses the full scaled step and gradient change. The first-valid-secant gamma
description above applies to standard BFGS initialization and resets, while
joint history rebuilds from its newest valid gamma and replays up to eight
retained secants.

With :math:`D(\theta)=D_t+D_{\rm add}(\theta)` and the baseline :math:`D_b=D(\theta_b)`, the fitted objective is

.. math::

   \mathcal L_\rho(D)=\frac{\beta}{2}\operatorname{Tr}(D\widehat C)-\frac{1}{2}\log\det D
   +\frac{\rho}{2}\left[\operatorname{Tr}(DD_b^{-1})-\log\det(DD_b^{-1})-n\right].

The prior term penalizes departures from the finite-temperature graph baseline when the sample covariance has weak or missing directions. It does not change the sampled trajectory or replace the later held-out response and full stored-D checks. A status is successful only after ``BASELINE_SPD_PASS``, ``COVARIANCE_OPTIMIZER_CONVERGED``, ``FIT_RESPONSE_PASS``, ``FULL_CERTIFICATE_PASS``, and ``REFERENCE_WRITTEN`` appear in order in ``.fit_trace.txt``.

First equilibrate the beads and collect aligned centroid samples and ``mean.xyz``. List every bead file explicitly; this two-bead invocation is only a short example. Real eight-decimal ``dump_beads`` cells require ``--cell-model`` with the original full-precision structure. Omitting it is suitable only when the supplied XYZ already has enough cell precision for downstream raw-reference checks::

  python tools/rpmd_ja_fit_reference.py collect --bead-files bead_0.xyz bead_1.xyz --temperature 300 --output samples.npz --mean-model mean.xyz --masses 1.008 15.999 --cell-model original_model.xyz

Replace the example mass list with every atom's actual mass in original model order. The forces in each bead dump must be that bead's physical force under the sampled Hamiltonian, not ring-spring forces and not a force recomputed at the centroid :math:`R_c`. The collector averages bead forces exactly once, :math:`P^{-1}\sum_b F_b`, while forming the centroid samples. The fitter assumes these forces are energy-consistent with the Hamiltonian used for raw derivatives; independently check qNEP/PPPM energy and force deviations against the material accuracy target. The collector uses the full-precision cell from ``--cell-model`` after checking every bead dump against the eight-decimal serialization bound. It writes ``mean.xyz`` from the first two thirds of chronological frames. In a separate raw-generation directory, copy this output as ``model.xyz`` and provide the same full-precision cell, atom types, masses, qNEP model, and PPPM settings as the bead simulation. Then generate the raw reference::

  potential nep.txt
  kspace pppm 1.0
  rpmd_ja generate_raw ja_raw.qraw 300 1e-4 ja_kernel_u8.tbl

Run GPUMD in that directory. Reuse the PPPM spacing from the original Hamiltonian (or the default 1.0 Angstrom spacing if that was used); do not change electrostatics settings. The raw ``R0`` must match the collected training mean or ``fit`` will reject it. Native generation writes qraw format v4: fine ``V/C/K`` and the original 18 diagnostics are retained, while coarse ``C`` is checked through the same mass-weighted native transport formula and stored as three difference-squared plus three fine-norm statistics. V-row sums follow native left-to-right site order and the H transform uses ``(C - fenergy) - V`` in that order, preserving cancellation-sensitive norms during prepare. Older qraw v1-v3 remain readable. Format v4 keeps the derivative policy at 3; the byte fingerprint changes, so old state files still require ordinary exact-input identity validation. Then fit and prepare the auxiliary package. Use the same kernel table for raw generation and preparation::

  python tools/rpmd_ja_fit_reference.py fit --raw ja_raw.qraw --samples samples.npz --output ja_additive.bin --cutoff 5.0 --epsilon 1e-8 --response-tolerance 0.15

For ``n`` atoms and ``d=3n``, the v3 payload is ``(2*d*n + 7*d*d + 18)*8`` bytes and the v4 payload is ``(d*n + 4*d*d + 24)*8`` bytes, excluding the common header (about 43.5% fewer payload bytes for large systems). Native preflight accounts separately for the final v4 file and the temporary-plus-V-transpose generation peak.

To compact an existing validated v3 qraw file without rerunning qNEP, choose a new output path and run::

  python tools/rpmd_ja_qraw_compact.py old_v3.qraw compact_v4.qraw

The converter rechecks the v3 diagnostics, exact fine/coarse ``V`` identity, and mass-weighted fine/coarse ``H`` transport before atomically publishing v4. It streams the matrices with bounded 128-row ``V`` blocks, preserves the source, and does not overwrite an existing output. The report uses SHA-256 for converter checksums; GPUMD's native qraw identity remains its existing FNV byte fingerprint. The new byte fingerprint requires the normal exact-input identity check for saved state; v1/v2 derivative layouts are not converted.

The fitter uses one fractional minimum-image distance per ordered distinct atom pair within the cutoff. It does not enumerate multiple periodic images or self-image edges and is not the full qNEP neighbor graph; this defines the fitted model space, whose cutoff and system-size adequacy must be checked on held-out data. Collection keeps ring unwrapping as a helper but rejects bead configurations whose resulting centroid differs from the production bead-zero minimum-image centroid. Samples must also remain within the existing 0.45 fractional fixed-reference branch. Sparse dump cadence can alias periodic motion, so choose sampling that resolves the dynamics; no cadence threshold is inferred by this utility.

For CPU-only preparation, build the final v3 reference with the fitted package::

  python tools/rpmd_ja_qnep_prepare.py ja_raw.qraw ja_additive.bin --kernel-table ja_kernel_u8.tbl --output ja_reference.bin --lossless

The equivalent GPUMD production preparation command is ``rpmd_ja prepare ja_raw.qraw ja_reference.bin ja_kernel_u8.tbl ja_additive.bin`` before any ``ensemble`` or ``run``. Both forms bind the auxiliary package to its raw source, temperature, bead count, and fixed cell. To measure, use matching reference metadata and full-system ordering, and initialize from an equilibrated bead-resolved restart within the reference's 0.45 fractional branch. The measurement need not start at the training mean position. Enable ``rpmd_ja on ja_reference.bin`` in the measurement input. A separate legacy preparation without the additive package is the native comparison. The JA current implementation is unchanged.

The collector's synthetic integration fixture uses full-precision coordinates and forces to check exact zero-addition recovery; real ``dump_beads`` coordinates, forces, and lattice are serialized to eight decimal places. The optional cell model restores only the original cell and does not recover quantized positions or forces. Quantization sensitivity must therefore be measured on real data before interpreting a material fit. The included ``U=8`` kernel fixture is a repository test input, not evidence of physical kernel accuracy. The CPU workflow tests do not run qNEP generation or the CUDA prepare path.

The fitter limits dense matrices to 6,000,000 elements, the dense parameter set to 1,600, the translation-free internal dimension to 384 (at most 129 atoms), and the uncompressed sample arrays to 256 MiB. Additive CPU preparation additionally limits dense assembly to 4,000,000 elements and the package to 64 MiB. These limits do not include all NumPy, LAPACK, eigensolver, process, or source-matrix memory; estimate host memory before larger jobs. The GPU prepare path has its own preflight and memory constraints. No larger-system performance or fit accuracy is implied.

``heat_current_rpmd_ja.out`` stores the three directional centroid current, additive correction, and their sum. ``hac_rpmd_ja.out`` stores only the three directional :math:`C_{AA}` self-correlation components and their integrals. All four correlation terms can be reconstructed offline at each lag from the three current series in ``heat_current_rpmd_ja.out``; GPUMD does not write four HAC files.

Observable and scope
--------------------

The current is :math:`J_A=J_{cent}^{mech}+\Delta J`, with :math:`\Delta J_\alpha=x_c^T\Delta H_\alpha v_c` and :math:`x_c=R_c-R_0`. :math:`J_{cent}^{mech}` is the full mechanical centroid heat current computed by a private single-centroid evaluator. For qNEP, both the baseline and reference generation retain the native charge-chain virial and :math:`q(R_c)` for each sampled centroid; they do not add the separate ``qnep_full_a`` current correction. qNEP references use the explicit ``native_reference_transport`` policy :math:`H_\alpha^T=C_\alpha-V^TL_\alpha-\Psi E_\alpha`; this is a defined transport operator, not an unconditional canonical-equivalence claim. The qNEP generation command directly creates the final v3 block reference and sidecar, and the separate Python utility remains available for offline work without being required by GPUMD generation. A qNEP reference is bound to its charge mode, electrostatics settings, model, and mechanical-configuration fingerprint; an ordinary NEP reference cannot be applied to qNEP. The fixed reference Hessian and site-flow derivative define a local quadratic correction around :math:`R_0`; it is a candidate observable for that neighborhood, not a general quantum heat current or a DC-conductivity guarantee. Harmonic quadratic calibration checks the implementation in the harmonic limit but does not establish accuracy for arbitrary anharmonic systems. This ``rpmd_ja`` option remains separate from ``hac_current qnep_full_a``.

Cost and validation
-------------------

The sparse backend shares three D-operator seed evolutions. For polynomial degree :math:`L`, and P/Q ranks :math:`R_P,R_Q`, the measured correction uses about :math:`(3L+2)` D-vector actions and :math:`3(R_P+R_Q)` site-operator vector actions per sampled frame, plus reductions. For the current implementation with both ranks nonzero, the sparse correction launches :math:`7L+23` kernels and zero-fill operations; at :math:`L=148`, that is 1059 launches. This count includes the existing translation projections and reductions. Each sparse matrix-vector kernel checks and sanitizes its own non-finite row, so separate full-vector scans are not launched. Each sample copies the latched error flag to the CPU once, after the centroid NEP evaluation and correction; a final non-sample step retains its branch check. GPU workspace bytes cover its cached CSR matrices, coefficients, and working vectors; the run separately has HAC and private NEP allocations. Runtime speed has not been benchmarked and is not implied by this launch count. Current and HAC output is buffered until ``post_run``; sampling performs no JA file writes.

The qNEP fixed-reference cache has a one-time generation and per-sample cost boundary. Its uncompressed fine/coarse :math:`V,C` and :math:`K` data volume is :math:`(23/3)D^2` doubles, or about 17.9 GiB at :math:`N=5896`, before any block compression. This is a resource-planning estimate, not the size of the final block file or evidence that a 5896-atom run fits. Generation performs about :math:`4D+13` private qNEP evaluations for the current three-direction curvature probes. The private qNEP/PPPM and stability/compression workspace peaks have not been measured; check available host memory, temporary storage, and GPU memory against the actual system before running, and do not treat 5896 as validated. During measurement initialization, source CPU tiles are uploaded in bounded batches of 32 and released after cache construction; no packed full-matrix host copy is made. The cache removes exactly the three known mass-translation modes, performs one projected :math:`D_s` eigensolve and three modal :math:`B^T` transforms at :math:`O(D^3)` setup cost, and retains the full P/Q tables and their stated error budget. The three generally non-symmetric modal matrices are transformed into physical matrices :math:`E K_\alpha E^T`, adding two dense matrix multiplications per direction during initialization. Its steady cache uses :math:`3D^2` doubles (about 6.99 GiB at :math:`N=5896`); construction uses about :math:`3D^2+2Dr+r^2` doubles (about 13.98 GiB), and the eigensolver phase and workspace can increase the maximum peak. The batch inputs and action workspace add :math:`24D` doubles; small vectors and solver workspace add to these estimates, and GPUMD's other allocations are separate. Samples are captured immediately and evaluated in batches of up to eight using three cached physical matrix-matrix actions, with :math:`O(3D^2)` work per sample and no D-order recurrence, eigensolve, or allocation. Insufficient GPU memory is an error; the former expensive recurrence path is not used as a fallback. These are estimates and have not been confirmed by CUDA measurement. Reference finite differences run only during one-time generation, and current/HAC files are written after the run rather than during sampling.

The ordinary-NEP sparse offline checker has a different cost profile: it forms and factorizes a dense anchored minor using :math:`O(D^2)` host memory and :math:`O(D^3)` Cholesky/LAPACK work, in addition to the CSR arrays. Its Python reader uses ``np.fromfile`` to avoid a full intermediate ``bytes`` object and its extra copy; this reduces transient reader memory but does not change the three-matrix Cholesky estimate. For :math:`N=5896`, one :math:`3(N-1)` square double matrix is about 2.33 GiB, so the checker's estimate is roughly :math:`3\times2.33` GiB plus CSR storage and a 256 MiB reserve. Kernel export, ordinary-NEP reference generation, kernel rebinding, and this checker are one-time preparation steps, not part of the sparse per-frame GPU cost. The full 5896-atom NEP CPU stability check remains pending on a host with sufficient physical memory; the available memory here is about 1.3 GiB, so it was not run.

The native qNEP centroid HAC evaluator has a GPU-enabled CLI comparison script. Supply a baseline executable, the candidate executable, and a case directory containing ``run.in``, ``model.xyz``, ``nep.txt``, and a self-describing ``restart_beads.xyz``. The input must use exactly ``potential nep.txt``, ``rpmd_ja off``, one ``ensemble rpmd`` and one positive ``run`` production segment, one direct centroid ``compute_hac`` with split and deferred output disabled, and one ``dump_pimd_restart`` interval reached by that run. Other ``rpmd_ja`` modes, any ``hac_current`` mode (including ``qnep_full_a``), TRPMD, and other ensemble segments are rejected. The fixture checks this native full-current/direct-centroid path; it does not test qNEP ``rpmd_ja``. Each isolated run removes prior comparison outputs, reads a preserved copy of the initial restart, and must create a fresh final bead restart. The script compares the centroid current, HAC, optional ``thermo.out``, and final bead restart::

  python tests/rpmd_ja_qnep_centroid_cli_test.py --baseline-gpumd /path/to/baseline/gpumd --candidate-gpumd /path/to/candidate/gpumd --case-dir /path/to/qnep-centroid-case

This is a production-evaluator comparison and requires working CUDA executables plus an equilibrated bead restart. It has not been run in this environment.

For an end-to-end qNEP JA acceptance run, provide one GPUMD executable, a relaxed ``model.xyz`` without velocities, a charge1/charge2 qNEP model, and a kernel table::

  python tests/rpmd_ja_qnep_end_to_end_test.py --gpumd /path/to/gpumd --model-xyz /path/to/model.xyz --qnep-model /path/to/nep.txt --kernel-table /path/to/ja_kernel.tbl --temperature 300

The script asks GPUMD to write the final v3 reference and sidecar directly, initializes a fixed-seed bead restart with a short PIMD run, and runs the same short standard-RPMD case with ``rpmd_ja off`` and ``rpmd_ja on``. It compares centroid current/HAC, the sampled 18-column ``thermo.out`` including the total-energy drift :math:`K+U`, and final bead restart; it also checks finite three-direction J_A data, :math:`J_A=J_{cent}+\Delta J`, and reconstructs every reported JA HAC lag from all four centroid/correction and cross terms. Each production run reads a separately named copy of the seed restart and must write a fresh final restart at the requested final step. The generated short restart is for software acceptance, not equilibration or production transport. No zero-correction reference is synthesized. CUDA executables are required; this end-to-end run has not been run in this environment.

Add ``--wpe-gpumd /path/to/gpumd-wpe`` to run the same reference and RPMD
inputs through WPE as well. The extended check compares native and WPE sampled
centroid/JA currents and HAC data, confirms that enabling JA leaves each
backend's bead trajectory unchanged, and runs ``rpmd_ja sample`` under
canonical PIMD to compare sampled centroid positions and mean physical bead
forces, and to check the provenance sidecar. The sample inputs set the same
``velocity ... seed`` before loading the bead restart; loading the restart
replaces those temporary velocities, while the seed fixes the PIMD thermostat
random stream initialized for that run. This requires both working CUDA
executables and has not been run in this environment.

The production CUDA fixture exercises the actual sparse workspace and v3 block GPU matvecs (dense and low-rank tiles, partial tiles, multiple vectors, and workspace reinitialization), asymmetric directional operators, signed P/Q coefficients, nonuniform masses, translation projection, zero-rank correction, multi-cell private wrapping, and non-finite error latching including sparse SpMV overflow. It does not exercise the enclosing HAC sample synchronization or the fixed-reference branch tracker. Run it where CUDA is available with::

  nvcc -std=c++14 -Isrc tests/rpmd_ja_sparse_cuda_test.cu src/measure/rpmd_ja_sparse.cu src/measure/rpmd_ja_qnep_cached.cu src/utilities/error.cu -lcublas -lcusolver -o rpmd_ja_sparse_cuda_test
  ./rpmd_ja_sparse_cuda_test

The fixture command is provided for CUDA validation; it has not been run in this environment.

Outputs
-------

``heat_current_rpmd_ja.out`` contains the centroid current, correction, and their sum in x, y, z. Current values use eV Angstrom per natural time unit. ``hac_rpmd_ja.out`` contains the separate x, y, and z self-correlations and running integrals, normalized by :math:`k_B T_{reference}^2 V`. Each HAC column is averaged over all valid time origins and has units ``(eV Angstrom/natural_time)^2``; each RTC column is in W/m/K. Sum the three RTC columns for the trace, or divide that sum by three for the isotropic average. The columns retain direction and are not averaged together. Existing centroid current and HAC files remain the comparison series. JA samples are kept in memory and both files are appended only after the run; incompatible older JA output schemas are rejected before integration starts.
