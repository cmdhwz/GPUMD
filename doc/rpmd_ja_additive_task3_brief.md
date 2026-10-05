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

## Task 3: integration tests and user workflow documentation
Files: tests/rpmd_ja_additive_workflow_test.py; doc/gpumd/input_parameters/rpmd_ja.rst; tools/rpmd_ja_fit_reference.md; small fixes only as specifically assigned. report doc/rpmd_ja_additive_task3_report.md.
- [ ] Actual CLI collect/fit/prepare synthetic workflow with good/corrupt/response-rejected inputs.
- [ ] Independent unequal-mass nonsymmetric H oracle and zero harmonic recovery to existing mapping; reference sensitivity on same validation series.
- [ ] Document commands, force/position conventions, data split/R0 generation order, full resource limits and new T/P binding, raw diagnostics vs auxiliary reference.
- [ ] Run relevant existing Python suites, compile syntax, diff hygiene. CUDA/source fixtures remain separate from runtime evidence.

## Required independent numerical checks
- Use actual subprocess CLI collect -> fit -> Python prepare, including stable harmonic zero correction and nonzero recoverable correction; exact synthetic validation covariance and unequal masses. Test bad raw source / corrupted pack / response refusal leaves no output. Kernel fixture must have honest test-only description; an artificial constant table does not validate physical kernel accuracy.
- Build a separate local energy/gradient/virial oracle for general cross-neighbor B and nonzero ell. Finite-difference V and C at x=0; assemble native H^T=C-V^T L-Psi_E independently. Compare to actual fitter assemble_additive in SoA and verify ell contribution cancels to Hadd=0 while total center gradient is canceled. Do not just call assembler twice or restate its frozen edge contraction.
- Inspect actual prepared v3 matrix tiles: mass-weighted D and H^T have correct unequal-mass factors and orientation. Python equality to a standalone matrix is useful evidence, but not a compiled CUDA result.
- For the unchanged harmonic mapping use the independent Fock formula in tests/rpmd_ja_reference_math_test.cpp (sp/sm sum/difference weights; preserve omega[a]/omega[b] and H asymmetry). Verify zero addition restores original K/H/DeltaH and current on the same x/v series. Verify a second legal local gauge with same K can change DeltaJ; don't claim classical gauge invariance survives mapping. Use all four correlation terms for reconstruction. Label these CPU oracle checks.
- Documentation must distinguish static software/synthetic checks, actual CUDA/build validation (unrun here), and real-data/quantum/material checks. Fixed branch, winding and sampling-rate assumptions stay explicit. No added arbitrary spread/dt threshold.

## Additional source-grounded workflow repair assigned by main
- `dump_beads` serializes Lattice with only 8 decimal places while JA runtime matches H at 1e-10 relative scale. Using a rounded dump header to create mean.xyz can therefore reject the reference against the original simulation cell.
- Task 3 may minimally extend tools/rpmd_ja_fit_reference.py and its focused tests with optional collect `--cell-model original_model.xyz` (function optional cell_model=None). Read full precision Lattice/full PBC, N and symbols from the model; validate dump cell entries against its 8-decimal serialization error (5.1e-9 absolute plus floating precision bound), not an arbitrary relaxed runtime tolerance. Use the original cell for all unwrapping/sample cell/mean.xyz. Preserve old CLI/function behavior without the option. Reject incompatible source model and leave no outputs.
- Document that real GPUMD dump workflows should supply the actual fixed-cell input model. Add a nontrivial tilted cell rounding regression and a mismatch refusal test. Preserve production runtime cell tolerance.
- Additional interface audit: fit currently accepts sample/raw T within 1e-8 relative, then writes sample T, while prepare requires exact package/raw T. Task 3 may make package T canonical raw temperature and use that same temperature in response/IBP calculations after validating sample T at production 1e-10 relative tolerance. Add small-roundoff accepted/clearly-different refused regression. Avoid generating a fit package that the prepare chain refuses solely because metadata uses the noncanonical sample representation.
- Additional confirmed production-branch gate assigned by main: ensemble_pimd.cu:885 first MICs each bead to bead0, then gpu_average:922 averages; force.cu centroid routine uses bead0 MIC too. Collector adjacent-link ring unwrap can give a different centroid even at zero winding ([0,.3,.6,.3] -> .3L vs production .05L). Keep align_bead_ring correct; in collect, before time unwrap/COM removal, independently construct bead0-MIC centroid and require equality with ring centroid at floating operation roundoff tolerance. Reject incompatible branch clearly, no adjustable physical spread threshold and no change to integrator/current. Include that specific zero-winding mismatch test plus accepted wrapped narrow ring. Document production-compatible branch requirement and normal sampling interval ambiguity separately.
- State graph/model-space boundary accurately: current fitter picks one fractional MIC image per ordered distinct atom pair within cutoff; it does not enumerate multiple periodic images, does not include i==j images, and is not the full qNEP interaction graph. This is a specified reference parameter space whose accuracy needs held-out/cutoff/size checks; do not claim all physical cutoff images are included or add unrelated graph infrastructure this round.
- Additional strict validation fix assigned: _validate_additive local B symmetry currently sums unscaled squares and can overflow/underflow, allowing relative asymmetry to pass. Use max-absolute-entry scaled Frobenius symmetry comparison (zero B passes). Match C++ prepare helper which Task2 implementer is correcting. Add large/tiny finite asymmetric block refusal with normal symmetric roundtrip retained. Limit this repair to local coefficient validation, no unrelated norm rewrite.
- Production branch integration check assigned: before fitting, compare collected centroid displacement relative to actual raw R0 in fractional coordinates with existing JA branch limit 0.45 (rpmd_ja.cu). Reject samples outside this fixed-reference branch; do not introduce a new empirical spread threshold. Raw R0 vs training mean check stays. Add a refusal regression and document that sparse output cadence can alias periodic motion and requires independently adequate sampling; do not invent a dt limit. This applies after collector COM/time alignment, before LS allocation.
