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


## Additional audit rulings before integration
- Existing Cholesky certificate checks three probe vectors only. For additive epsilon certification this is insufficient: compute a full Frobenius reconstruction upper bound eta for D_internal-epsilon/2 I minus L L^T; require eta<epsilon/2 (prefer epsilon/4 margin), then retain unshifted D in stored tiles. A full-entry CUDA reduction or guarded host dense calculation is acceptable; no eigenvalue claims from probes.
- Raw cell is row-major H; dump_beads extended-XYZ Lattice is H.T flattened. GPUMD K_B=8.617343e-5.
- Keep raw derivative diagnostics before adding increments and label them baseline; don't let enlarged denominators disguise baseline error.
- C++ package and runtime policy parser must check integer overflow and exact trailing syntax. New additive stability sidecar certificate may use a distinct strict token if it conveys the shifted full-bound semantics; Python writer and C++ reader must agree. Legacy certificate remains accepted only for legacy policies.
- CUDA orientation for full Cholesky bound: compact_physical writes row-major symmetric D; cuSOLVER POTRF LOWER interprets column-major. Factor L(i,k) is physical[k*r+i], not physical[i*r+k]. Reconstruction LL^T must use that orientation. Existing saved probe products use unshifted D; either subtract epsilon/2*probe before comparison or keep legacy probe verification only in legacy branch. Never compare shifted LL^T directly against unshifted D products.
- Baseline validity includes original K asymmetry and original mass-weighted translation projection change (existing 0.05 limits), not only raw footer fine/coarse stats. Check/report these against baseline before adding Kadd; otherwise a large symmetric increment can dilute an unacceptable baseline error. Total reference projection still checked separately. Fitter must make the same baseline projection decision so offline fit and prepare agree.
- Relevant existing tests include tests/rpmd_ja_qnep_prepare_test.py and tests/rpmd_ja_measurement_test.py. If replacing the literal policy check with a shared strict helper makes a source-contract assertion stale, you may minimally update that specific assertion to the new contract; do not rewrite numerical measurement tests or weaken behavior checks.
- Python import must still work when legacy tests load rpmd_ja_qnep_prepare.py using importlib.spec_from_file_location, and when CLI runs from another directory. Resolve the sibling fitter without assuming cwd is repository root.
- Any new host dense K/H reconstruction must have an explicit element/byte preflight before allocation. The approved first implementation is small-system certification; a clear additive-only resource refusal is acceptable (document exact cap), while legacy large-reference preparation remains unchanged. Do not silently allocate four 3N-by-3N host matrices for the current 5,896-atom material. Local scatter/tile integration is preferable if equally simple; no speculative sparse infrastructure.
- Exact C++ baseline asymmetry metric uses MASS-WEIGHTED unsymmetrized K: asymmetry_sums divides each entry by sqrt(m_i*m_j) first (lines459-462). Existing Python prepare historically measures Cartesian rawK asymmetry; for the new integration make its decision agree with C++ and fitter, with an unequal-mass asymmetric-noise test. Projection metric remains Frobenius relative change of mass-weighted symmetric D.
