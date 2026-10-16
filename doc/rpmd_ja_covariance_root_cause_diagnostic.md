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
After the CUDA regression passes, run a 10-step A/B using the same fixed
sample, qraw, rho, tau schedule, cutoff and response threshold; vary only the
output path, iteration budget and opt-in token. The full-gradient KKT
threshold remains 1e-6, and need not decrease monotonically at every step.
Use distinct outputs:

```text
rpmd_ja fit_samples_covariance <same_samples> ja_cov_bfgs10.bin <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <same_qraw> shrinkage <same_rho> max_iter 10
rpmd_ja fit_samples_covariance <same_samples> ja_cov_block_newton10.bin <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <same_qraw> shrinkage <same_rho> max_iter 10 block_newton
rpmd_ja fit_samples_covariance <same_samples> ja_cov_joint_block10.bin <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <same_qraw> shrinkage <same_rho> max_iter 10 joint_block
```

The ``joint_block`` trial ranks groups by maximum absolute scaled gradient and
uses up to three whose union support fits within 32 atoms. It applies a pure
selected-group direction; larger-support curvature remains BFGS. For an
accepted joint step, the trace compares the selected-subspace gradient change
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
