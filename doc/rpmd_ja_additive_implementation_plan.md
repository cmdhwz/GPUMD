# J_A Additive Reference Implementation Plan

> For agentic workers: Use superpowers:subagent-driven-development. The user explicitly authorized execution and specified gpt-6-luna/high implementers; main agent reviews and never edits production code.

Goal: fit a finite-temperature auxiliary reference and prepare/run the existing JA using consistent K0/H0, preserving legacy references and real RPMD dynamics.
Spec: doc/rpmd_ja_finite_temperature_reference_plan.md.
Architecture: Python/NumPy offline data+fit tool writes local correction coefficients. Existing CUDA prepare optionally reads that package, analytically reconstructs Kadd/Hadd, then uses existing v3 modal cache and measurement.
Tech stack: C++/CUDA existing libraries, Python 3 + NumPy (bundled 2.3.5). SciPy absent; do not install packages.

## Global constraints
- Work only in GPUMD-branch_new_wpe-worktree, branch codex/rpmd-ja-heat-current at base c61d2a6c4d27eeece634edf4a30351e5f0c0610e.
- Preserve unrelated files and legacy inputs. No commits/pushes, no full build without toolchain, no material claims.
- Every implementer is gpt-6-luna, reasoning high, isolated context. Do not spawn children.
- Main reviews/coordinates; implementers write source/tests/docs and fix findings.
- Existing JA current and all four correlation terms are unchanged.
- Kadd and Hadd derived from SAME symmetric local Bi, fixed images and explicit ell. H is not symmetrized.
- Center term -b.x is a reference-only term, Hadd_linear=0.
- Fixed reference T/P; reject bead mismatch at runtime for additive references.
- Reject nonfinite, truncated, wrong-source, asymmetric local blocks and invalid images/dimensions/packages. Output failure leaves no usable reference/pack.
- Source matching uses the existing FNV file fingerprint once per package write/read, never per frame.
- Tests must exercise numerical values and integration, not only mirror text.

## Wire contract GPJAADD1
Little endian, no implicit padding:
8 bytes magic GPJAADD1.
u32 version=1; u32 endian=0x01020304; i32 N; i32 P; u64 source qraw FNV fingerprint.
f64 T; f64 epsilon_num; f64 max independent response relative error; f64 allowed response relative error.
u64 training frames; u64 validation frames.
For each site in original atom order: i32 z; z records (i32 neighbor, i32 image_x, i32 image_y, i32 image_z); f64 ell[3z]; f64 Bi[(3z)^2] row major.
No trailing data. N>=2, P>=1, finite T>0 and epsilon>0; frame counts positive; response nonnegative <= budget. Neighbor != site, no duplicate image keys, reverse image exists, graph connected. Matrix symmetry at 1e-12 relative scale.
Source raw cell is first9 row-major H with Cartesian column vectors; positions SoA. d_ie=Rj+h*n-Ri. Serialized Bi are Cartesian eV/A^2; ell eV/A. This is coefficient package, not independent precomputed K/H matrices.
Policy for final reference: native_reference_transport;finite_temperature_additive_v1;beads=<P>;derivative=<raw version 1|2|3>. Strict parse; preserve existing three policies.

## Task 1: offline data/fit/pack
Files: create tools/rpmd_ja_fit_reference.py, tests/rpmd_ja_fit_reference_test.py. May read existing Python prepare, not modify it (reserved task2). Write report doc/rpmd_ja_additive_task1_report.md.
- [ ] Tests first for package roundtrip/rejection, triclinic image alignment, general cross-block K/H derivative oracle, zero correction harmonic recovery, known recoverable quadratic addition, spectral-cut QP, independent response failure.
- [ ] Implement reader raw versions1/2/3 with exact layout/footer checks equivalent accepted production diagnostics. Use memmap matrix access to bound copies; data-size/memory preflight before dense allocations.
- [ ] CLI collect: --bead-files files in bead order --temperature T --output samples.npz --mean-model mean.xyz. Read actual GPUMD xyz Properties/Lattice, verify synchronized times, cells, symbols order, finite positions/forces; align beads to bead0 per frame, unwrap time continuity, remove global mass translation using explicit masses (--masses original order required if no species mass source; avoid guessing masses). Average physical force once. NPZ non-pickle contains centroid positions/forces, masses, types or species, cell, T, P, source steps. Mean xyz preserves original symbols and mass metadata for GPUMD.
- [ ] CLI fit: --raw raw --samples samples.npz --output correction.bin --cutoff A --epsilon numeric --response-tolerance numeric, training first contiguous 2/3, validation remaining 1/3 (require enough samples; manual fractions can be kept simple). Require raw R0 agrees with training aligned mean within documented mean tolerance; do not silently shift old derivatives. Default collection mean must use same training segment. Static target coordinates mass-weighted translation-free.
- [ ] Frozen local full symmetric quadratic-star basis, minimal Frobenius lift modulo EXACT algebraic kernel of T. Distinguish data rank deficiency; reject it rather than ridge. Include connected-graph quadratic direction. Build equivalent minnorm representation with SVD; preflight dense small-system gauge/QP resource cost before allocation, fail clearly for unsupported resource sizes (no claim scalable certification).
- [ ] Fit unregularized least squares; if needed closed epsilon PSD constraint via cumulative low-eigenvector cuts and convex QP. NumPy-only solver: whiten full-rank LS objective by SVD then dual projected coordinate descent for finite linear inequalities; verify primal feasibility/KKT and fail on nonconvergence. No arbitrary local PSD restrictions. Final full internal eigencheck + shifted Cholesky/reconstruction bound.
- [ ] Compute ell from unit-weight connected graph solve L zeta=b from raw V sum; require net gradient translation within tolerance.
- [ ] Independent response/IBP and fit residual report, raw-source numeric diagnostics retained. Check response probes containing full internal directions for small certification, output only when pass, atomic write + refuse overwrite. Preserve baseline diagnostics separately from artificial-reference quantities.
- [ ] Export pack using frozen wire contract. Public importable read_raw, read_additive, write_additive, assemble_additive (dicts acceptable). Return unweighted Cartesian Kadd, Hadd shape (3,D,D) row displacement col velocity and force-gradient ell contraction.
- [ ] Test commands with bundled python; report red/green evidence and exact limitations.

## Task 2: prepare/reader/runtime and CPU reference integration
Files: tools/rpmd_ja_qnep_prepare.py; src/main_gpumd/run.cu; src/measure/rpmd_ja_qnep_prepare.cu/.cuh; src/measure/rpmd_ja_reference.cu/.cuh; src/measure/rpmd_ja.cu; new small host helper src/measure/rpmd_ja_additive.cuh if appropriate; tests/rpmd_ja_additive_prepare_test.py; tests/rpmd_ja_qnep_prepare_cuda_test.cu as needed. report doc/rpmd_ja_additive_task2_report.md.
- [ ] Test-first numerical CPU prepare with v3 raw+pack, legacy regressions, wrong P/source and truncation policy rejection.
- [ ] CLI prepare <raw> <outfile> <kernel> [<additive-pack>], old five-token form unchanged. Function optional additive_path after validator/default existing callers.
- [ ] Add generate_raw <rawfile> <T> <fd_step> <kernel> qNEP-only initialization command by exposing existing raw writer without decomposition/deletion. No forced failed generate_sparse to obtain data. Same validation as existing generator.
- [ ] C++ validate pack source and coefficient metadata and assemble local K/H increments; derive/check ell cancels b but no need materialize full Cadd. No new force calculation.
- [ ] Apply Cartesian Kadd before mass weighting/projection and mass-weighted Hadd TRANSPOSE increments before compression (existing blocks are H^T velocity-output rows/displacement-input cols). Preserve baseline fine/coarse diagnostics; adding exact increment to both must not erase/mislabel baseline error.
- [ ] Additive shifted stability check with epsilon/2 numerical error bound; actual stored D stays unshifted. Do not call true-potential mode validator to attribute additive-reference failure to material curvature.
- [ ] Update accepted policies in prefix writer/reader/sidecar/runtime. Encode P in strict policy and reject atom.number_of_beads mismatch. Maintain config/T/mass/cell checks and old format.
- [ ] Python prepare updates for raw1/2/3 accepted diagnostic policies, optional pack exact same assembly and final policy wire compatible with C++.
- [ ] CUDA test source uses actual kernels if available, reports unrun otherwise. No simulated CUDA success.

## Task 3: integration tests and user workflow documentation
Files: tests/rpmd_ja_additive_workflow_test.py; doc/gpumd/input_parameters/rpmd_ja.rst; tools/rpmd_ja_fit_reference.md; small fixes only as specifically assigned. report doc/rpmd_ja_additive_task3_report.md.
- [ ] Actual CLI collect/fit/prepare synthetic workflow with good/corrupt/response-rejected inputs.
- [ ] Independent unequal-mass nonsymmetric H oracle and zero harmonic recovery to existing mapping; reference sensitivity on same validation series.
- [ ] Document commands, force/position conventions, data split/R0 generation order, full resource limits and new T/P binding, raw diagnostics vs auxiliary reference.
- [ ] Run relevant existing Python suites, compile syntax, diff hygiene. CUDA/source fixtures remain separate from runtime evidence.

## Review focus
1. Raw storage is derivative input rows/output cols; H tiles already mass weighted. Test nonsymmetric H and unequal masses.
2. Wrong R0/data mean or stale raw source must fail, not relabel derivatives.
3. Singular QP/gauge vs undersampled data must be distinguished and no nonconverged package emitted.
4. Bead cross-boundary and triclinic alignment, time synchronization, means from train only.
5. Effective-reference stability cannot suppress true raw derivative diagnostics or enter original-potential negative-curvature validator.

Execution method already authorized by user. Ruling: NumPy-only verified QP rather than installing SciPy (absent). Ruling: coefficient pack optional prepare input preserves legacy raw/version3 reference layout via explicit new policy; exact math reconstructed before factorization. Scope resource limit is honest preflight, not a silent fallback to different reference.


