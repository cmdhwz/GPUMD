# RPMD-JA finite-temperature additive reference: Task 3 report

Task 3 adds the synthetic CLI workflow acceptance test and user-facing procedure for the optional qNEP additive reference. It keeps the existing JA current implementation and legacy inputs intact. The fitter's assigned robustness changes cover full-precision fixed-cell restoration from an optional `--cell-model`, canonical raw-reference temperature binding, production-compatible bead centroid and fixed-reference branch gates, and scale-safe local-B symmetry validation. The cell-model reader honors GPUMD's default full periodicity when `pbc` is absent and accepts case-insensitive full-periodic tokens.

The integration test drives the actual Python CLI through collect → fit → CPU prepare. Its synthetic source contains unequal masses, a tilted fixed cell, full-precision coordinates and forces, and a real checked-in U=8 kernel table. It tests a zero-addition reference and a nonzero recoverable correction, compares against a separately prepared no-pack native reference, and checks actual serialized v3 matrices, directional H/mass weighting, the stored Chebyshev values/vectors, independent Fock mapping against the factorized cache, mapped current and gauge sensitivity, and reconstruction of all four correlation terms from time series. A separate local energy/virial finite-difference oracle includes cross-neighbor B terms and a nonzero periodic image; it checks virial derivatives, force derivatives, total gradient cancellation, and the additive Hessian. Failure cases cover a wrong raw source, corrupt package, short/malformed source, significant temperature mismatch, and held-out response rejection with no usable output.

The test fixture rounds only the dump cell field while retaining 17-digit synthetic position and force values. This enables a strict harmonic zero-recovery check and tests the eight-decimal cell restoration path. It is not a model of full GPUMD dump quantization. Production `dump_beads` also rounds positions and forces to eight decimals; their effect on the fit and zero-recovery tolerance needs a separate real-data sensitivity study. The kernel fixture is used as a numerical test input and does not establish physical kernel accuracy.

The user workflow is ordered as bead equilibration and collection → raw generation from collected `mean.xyz` in a separate directory → fit → preparation. This makes raw `R0` equal to the fitter's training mean. A production measurement uses matching metadata and must start within the reference branch; its instantaneous positions need not equal the training mean. The docs specify the all-atom fixed-Hamiltonian, fixed-cell, fixed-temperature, fixed-bead-count workflow, cell-model requirement, ring-centroid and branch constraints, one-MIC-image graph boundary, resource limits, output conventions, and incomplete validation boundaries. Real eight-decimal dumps require `--cell-model`; examples enumerate bead paths explicitly and retain the original PPPM spacing. The three HAC components currently written are directional `CAA` terms in `hac_rpmd_ja.out`; the four correlation terms are reconstructed offline from the three current streams in `heat_current_rpmd_ja.out`.

## Verification

The following CPU tests are the verification scope for this task:

```text
python tests/rpmd_ja_fit_reference_test.py
python tests/rpmd_ja_additive_workflow_test.py
python tests/rpmd_ja_additive_prepare_test.py
python tests/rpmd_ja_qnep_prepare_test.py
python tests/rpmd_ja_measurement_test.py
```

The primary workflow and fitter suites were run in this worktree after the latest cell-model token fix. The additive prepare, qNEP prepare, and measurement Python suites were separately run by the main reviewer and passed. The C++ source fixture `tests/rpmd_ja_reference_math_test.cpp` was not compiled or run. Python syntax compilation and `git diff --check` passed in the review runs. CUDA compilation and GPU runtime were not performed because no CUDA toolchain is available in this environment. GPUMD `generate_raw`, production C++/CUDA `prepare`, material trajectories, real dump coordinate/force quantization, and independent-block uncertainty have not been validated here.
