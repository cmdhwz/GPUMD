# Fixed-reference covariance root-cause diagnostic

This diagnostic replays an existing covariance state and centroid spool. It
does not write an accepted reference. Interpret numerical reliability,
optimization progress, data support, and material response separately.

## Required evidence

1. Compare training and validation moments, the baseline-prior inverse
   response, and the saved-reference inverse response on the **same vectors**.
   Split each vector's stiffness into raw-qNEP, graph stabilization, and
   fitted type-group corrections. Large signed cancellations are evidence
   to investigate, not proof that one group caused the global error.
2. Report per-RHS original and factor solve residuals and inverse-quadratic
   changes under bounded refinement. Report Ritz residuals independently.
   A reliable response on a probe does not require that probe to be an
   eigenvector; atom-localized eigenmode interpretation does.
3. Check normalized operator Gram eigenvalues and fixed-graph support.
   Distinguish operator dependence from insufficient data. A change in
   total stiffness does not uniquely determine the local energy allocation
   or transport operator.
4. Report local joint second moments, covariance and mean drift about the
   same training reference center, using frozen atom IDs and periodic images.
   Report lag and block-length sensitivity; block spread is not a confidence
   interval. Centroid spools cannot identify tunneling rates.
5. For a small supported parameter group, report the conditional Hessian
   at fixed rho, tau and all other parameters. BFGS remains the default;
   ``block_newton`` opts in to using a checked conditional direction for the
   selected six-parameter group. Neither the local Hessian nor that direction
   is a full-space certificate. Finite-difference checks must report their
   actual steps, feasibility and convergence under step reduction.
6. Record factorization count, RHS columns, sample passes and stage timing.
   New diagnostic work has explicit size/iteration caps. Exceeding a cap is
   an inconclusive diagnostic, not reference-family failure.

The bounded CUDA diagnostic uses at most four inverse-iteration vectors and
twelve block iterations, two refinement corrections per solve, and a
conditional six-parameter Hessian only when its group touches at most 32
atoms. It does not replace the full-RHS production gradient check. The
resolved Ritz residuals certify approximate eigenpairs, not that every
globally softest direction has been found. Hessian differences report both
step sensitivity and analytic agreement using a 10% diagnostic threshold;
roundoff-dominated differences are inconclusive. This does not relax the
separate production gradient/residual or reference-acceptance gates.
The offline tool limits joint coordinates to 64 and retained joint plus group
quadratic series to 128 MiB; its lag cap is min(segment_frames/4, 256).
Its 8/32/128-frame block summaries are sensitivity estimates, not confidence
intervals. The selected quadratic products are xx, 2xy, 2xz, yy, 2yz, zz;
coverage must be checked before treating them as full-group statistics.

New fits record first-order predicted decrease, actual decrease and BFGS
secant statistics; opt-in block directions also record the selected group,
conditional solve checks and extra cost. Older saved states/traces may not
contain those fields; NOT_STORED is missing evidence, not zero curvature or
optimizer failure.

The global parameter-space inverse-BFGS spectrum is recorded at the
``DIRECTION_BEFORE_STEP`` snapshot after any non-descent reset, with iteration
and tau. The scaled, symmetrized eigensolve reports matrix scale, asymmetry,
eigen residual, diagnostic time, and whether SPD is reliably verified,
indefinite, unresolved near singular, or otherwise unverified. A tiny positive
minimum eigenvalue inside the residual/roundoff uncertainty is unresolved, not
SPD; a condition number is reported only for a reliable positive spectrum.
This is read-only optimizer telemetry: it is not a direction gate, reset rule,
physical Hessian, or material solve. Committed-step ``bfgs_sy``, the existing
``bfgs_s_norm2`` and ``bfgs_y_norm2`` values (s^T s and y^T y),
and ``bfgs_secant_cosine`` report curvature and normalized secant quality.

The legacy ``worst_solve_relative_residual`` remains scoped to the latest
gradient/trace evaluation. Separate ``trace_residual_*`` summary fields retain
the run-wide factor and checked-original maxima and their matrix kind/batch,
checked-batch counts, failed logical-batch count, original-check status, and
first failure matrix kind/batch/channel. The run summary is reset once at
optimizer entry, includes baseline and initialization gradients and rejected
trace batches, and excludes probe solves. If original residuals were never
checked, the summary has ``original_any_checked=0``, matrix kind
``NOT_CHECKED``, and NaN residual, not a zero-valued pass. State and fit files
use the ``run_trace_residual_*`` prefix; the trace's optimizer-summary line
uses ``trace_residual_*``. Parameter counts above 60 report
``NOT_EVALUATED_PARAMETER_COUNT`` without an eigensolve.

The ``joint_block`` optimizer records ``bfgs_recent_pairs`` and rebuilds the
shared global inverse Hessian after every committed full-gradient step from
the newest ``gamma = sTy / yTy`` identity background, then replays up to eight
recent valid secants in chronological order. Pure ``GROUP_BFGS``, global
``BFGS``, and small-support ``JOINT_BLOCK`` commits all contribute their full
scaled secant. This can refresh the scale in directions not covered by those
pairs; it is not an exact-Hessian estimate or a guarantee of improved
conditioning or KKT convergence.
The optimizer summary and covariance state also report the 128-column trace
solve width, the unchanged 32-column residual-check width, and submitted
``trace_solve_calls`` (including a failing wide solve); total RHS columns keep
their existing meaning: ``solve_rhs_columns`` counts columns from successful
wide solves, while diagnostic ``rhs_columns_checked`` counts completed
32-column residual blocks and may be smaller on strict failure.

For comparison only, the unrestricted full-matrix optimum with epsilon=0
has C*=(C_train+rho*C_b)/(1+rho+2*tau). The restricted local parameter family
and the shifted epsilon barrier do not generally satisfy this formula.

## Decisions after one diagnostic run

| Evidence | Next action |
| --- | --- |
| Solve/refinement or independent algebra checks disagree | Repair numerics before physical interpretation. |
| Baseline response is badly mismatched on the same supported probes | Isolate baseline/prior construction in a controlled comparison. |
| Exact operator dependence is found | Fix redundant parameters and audit the local energy allocation. |
| Moments drift or remain unstable with block length | Investigate sampling support; add data only for the unsupported question. |
| Valid curvature predicts useful descent but BFGS steps poorly | Test one curvature/step modification with a short explicit budget. |
| Reliable optimization passes but independent response remains wrong | Extend frozen structural/environment basis functions. |

The final reference still requires optimizer convergence, held-out response,
consistent additive energy/transport packaging, prepare, and the stored-matrix
SPD certificate. Finite probes do not certify full-space material response.

## Server checks

From the source worktree, run the focused Python checks before a material replay:

```bash
python3 -m unittest tests/rpmd_ja_pair_diagnostic_test.py tests/rpmd_ja_structure_diagnostic_test.py
python3 tests/rpmd_ja_diagnose_source_test.py
python3 tests/rpmd_ja_fit_sampling_source_test.py
python3 tests/rpmd_ja_fit_reference_test.py
```

Build the native CUDA regression with assertions enabled and a CUDA architecture
supported by the server GPU (example command; local Windows host has no nvcc):

```bash
nvcc -std=c++14 -O2 -Isrc tests/rpmd_ja_native_fit_cuda_test.cu src/utilities/error.cu -lcublas -lcusolver -o rpmd_ja_native_fit_cuda_test
./rpmd_ja_native_fit_cuda_test
```

Rebuild gpumd using the server's existing CUDA/NetCDF settings. In the material
directory, use a new output name in the existing run.in configuration:

```text
rpmd_ja diagnose_covariance ja_reference_cov_expand25.bin.covariance_state.txt ja_cov_expand25_full_diag_fixed.txt 5 1e-6 1 0.001
```

Run the offline pair tool with the two previously attributed edges, including
fixed additional environment atoms selected from the preceding report:

```bash
python3 <worktree>/tools/rpmd_ja_pair_diagnostic.py ja_reference_sample100ps.bin.samples.tmp --pair 5844:5847 --pair 5847:5863 --atom 1168 --atom 5873 --cutoff 5.0 --output ja_pair_expand25_full_diag.json
```

Paths in the state file must resolve in the material directory. Existing
reports are not overwritten. Run diagnostic commands without a new PIMD
sampling trajectory; do not reuse a stale gpumd binary for the CUDA report.
After the CUDA regression passes, run a 10-step server comparison using the
same fixed sample and qraw, positive ``rho``, epsilon, internally fixed tau
schedule, cutoff, kernel table, response threshold, finite-difference step,
training split, and all reference acceptance gates. Vary only the output path,
iteration budget and opt-in direction token. Keep the full-gradient KKT
threshold at ``1e-6``; KKT need not decrease monotonically at every step. Do
not weaken held-out response or final stored-D SPD/package gates. IBP remains
a diagnostic and is not a ``REFERENCE_WRITTEN`` gate. Keep its threshold and
reporting policy unchanged as well. Use distinct outputs:

```text
rpmd_ja fit_samples_covariance <same_samples> ja_cov_bfgs10.bin <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <same_qraw> shrinkage <same_rho> max_iter 10
rpmd_ja fit_samples_covariance <same_samples> ja_cov_block_newton10.bin <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <same_qraw> shrinkage <same_rho> max_iter 10 block_newton
rpmd_ja fit_samples_covariance <same_samples> ja_cov_joint_block10.bin <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <same_qraw> shrinkage <same_rho> max_iter 10 joint_block
```

The ``joint_block`` trial ranks groups by maximum absolute scaled gradient and
uses up to three whose union support fits within 32 atoms. It applies a pure
selected-group direction. If the dominant support exceeds 32, the status stays
``DOMINANT_SUPPORT_CAP`` and the optimizer alternates a committed
``GROUP_BFGS`` step with the shared global ``BFGS`` direction, starting with
``GROUP_BFGS``. A rescue updates the full shared secant history but not any
per-group history; its attribution lists the nonzero groups and parameters in
the actual step. Across all ``joint_block`` directions, the shared history
retains at most eight recent curvature-valid committed secants and replays
them oldest to newest over the newest gamma identity background. Each
committed pure-group, global, or small-support joint step contributes its full
scaled step and gradient change. Rejected candidates leave the history and
schedule unchanged. Invalid curvature and non-descent resets clear the recent
secants while keeping the latest valid scale; a ``tau`` change clears them and
restarts from identity. Each group has its own six-dimensional inverse-Hessian history
for each ``tau`` stage; it seeds from the first curvature-valid committed
pure-group secant. Joint or mixed-group steps do not update these histories. A
finite direction or representability failure falls back to global BFGS. For
``joint_block`` runs using ``GROUP_BFGS`` or ``BFGS`` directions, bounded
objective-only expansion proposes alpha values up to 4, then evaluates a full
gradient at the selected point's factors. A failed or worse expansion
re-evaluates the best objective-pass point before its gradient; objective-only
proposals are never committed. After a full validated candidate gradient,
every positive directional slope is rejected. Joint
``GROUP_BFGS`` and global ``BFGS`` directions alone use a directional-secant
proposal for the next trial, clamped to ``[0.1 alpha, 0.5 alpha]``; an invalid
proposal falls back to halving. Default BFGS, legacy ``block_newton``, and
small-support spectral ``JOINT_BLOCK`` directions retain legacy halving.
The trace records rejection reason, absolute slope ratio, and actual shrink
ratio. This proposal is not a full strong-Wolfe line search. The next trial
still needs its own full gradient and all commit gates before it can be
committed. The group direction uses no extra covariance
factors, RHS columns, or joint ray proposal, and its approximate model is
reported as unavailable. ``line_search_evaluations`` and
``line_search_objective_seconds`` cover line-search objective evaluations and
time, including re-evaluations after a failed or worse expansion. Trace
line-search gradient calls and time exclude baseline, initialization, and
tau-stage gradients. For an accepted joint step, the trace compares the selected-subspace gradient change
with the undamped Hessian prediction at the actual step ``s`` using
``||(g_new-g_old)-H*s||inf / max(||g_new-g_old||inf, ||H*s||inf, 1e-300)``.
This discrepancy and the accompanying group/global gradient norms are local
optimizer diagnostics, not Hessian, response, or material certification.

The joint direction's local inverse probes also provide a bounded spectral
ray model for choosing the initial line-search alpha. It uses the exact
``log1p`` determinant change along that direction, keeps a small negative
model-slope margin near the ray minimum, and stays below 90% of the estimated
``D``/``D-epsilon I`` feasibility boundary. The actual full objective and
gradient solves still control Armijo, residual, overshoot, and commit. Trace
and state keep the latest attempted proposal separate from the latest
committed proposal, including small eigensolve and scalar-model cost; a finite
estimate that fails its checks falls back to the prior bounded line search.

The saved-point response/Ritz/residual and finite-difference diagnostic
correction can proceed in parallel with this A/B. Compare full-KKT traces,
accepted steps, group transitions and added cost; neither a 10-step run nor
finite probes imply optimizer convergence, material acceptance or that the
material KKT problem is solved. The extra bounded diagnostic work is not a
measured speedup; compare stage timings on the same GPU and inputs before
changing production batching or synchronizations.
