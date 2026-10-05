# Task 1: additive J_A reference fitter

Status: the offline collector, force fitter, and GPJAADD1 coefficient writer are implemented in `tools/rpmd_ja_fit_reference.py`. The tests cover synthetic numerical behavior. This does not include qNEP material results, CUDA preparation, production integration, or transport claims.

## Commands and validation

```powershell
C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile tools/rpmd_ja_fit_reference.py
C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe tests/rpmd_ja_fit_reference_test.py
```

CLI forms:

```powershell
python tools/rpmd_ja_fit_reference.py collect --bead-files bead0.xyz bead1.xyz --temperature 300 --masses 1.0 3.0 --output samples.npz --mean-model mean.xyz
python tools/rpmd_ja_fit_reference.py fit --raw reference.qraw --samples samples.npz --output correction.bin --cutoff 4.0 --epsilon 0.01 --response-tolerance 0.05
```

The first test run failed because the fitter module did not yet exist. Round-2 regression tests initially failed on the scale-imbalanced response metric and, after adding exact NPZ payload-length checks, caught a closed-stream `tell()` bug; both were fixed. The final syntax check and test command pass. The integrated synthetic case uses two atoms with masses 1 and 3, two beads, 24 training frames, 12 held-out frames, and a known quadratic correction. It runs GPUMD-style extended XYZ with velocity and force columns through collection, exercises raw source versions 1, 2, and 3, performs the fit, reads the resulting package, and recovers the target Cartesian correction. The final run reports training force residual `6.47e-16`, held-out force residual `6.60e-16`, full internal held-out response operator error `1.81e-15`, IBP error `1.10e-15`, and shifted-Cholesky reconstruction residual `6.8e-16`. The spectral-cut fixture forces one cut and verifies its lower bound. Additional regressions cover a single bad response direction in 384 modes, full off-diagonal covariance, mass-weighted asymmetry, a separate baseline projection gate, preserving an existing `.tmp`, rolling back the first installed collect output if the second install fails, and rejecting oversized raw dimensions before reading samples. These are exact synthetic fixtures, not statistical material validation.

## Collection and data contract

`collect` accepts extended XYZ trajectories in bead order. Frames must declare `Time` or `Step`, three periodic directions, a nonsingular nine-value lattice, and `species`, `pos`, and physical `force`/`forces` properties. It respects the `Properties` column order, including optional velocity columns. Bead files must have equal frame counts, strictly increasing synchronized times, fixed cells, and stable symbol order. Masses are explicit and positive; the collector does not infer element masses.

Periodic bead coordinates are unwrapped along adjacent links in the triclinic cell. The collector rejects half-cell link ties and nonzero ring winding. It then keeps the centroid continuous across frames and removes changes in global mass translation relative to the first frame. Forces are averaged once. The non-pickle NPZ stores centroid positions, averaged forces, masses, symbols, cell, temperature, bead count, and steps. `mean.xyz` uses the first contiguous two-thirds of collected frames, matching the fit split, and preserves symbols and explicit masses.

`fit` verifies raw/sample temperature, cell, per-atom masses, atom count, sample frame shape, finite strictly increasing steps, and type-to-symbol partition. It compares raw `R0` with the training centroid mean and rejects differences above `2e-5 Å`; it does not translate finite-difference derivatives. It checks source-specific raw footer tags and diagnostic thresholds. The raw Hessian is mass weighted first; its relative asymmetry must be no greater than `0.05`, then it is symmetrized. The translation projection has a separate relative Frobenius change limit of `0.05` measured against the unprojected baseline alone, so an additive correction cannot dilute this gate. Raw diagnostics and both baseline gates are reported separately.

Dense matrix coordinates use Cartesian SoA order throughout: flattened coordinate `(axis, atom)` is `axis*N + atom`. Positions and forces are passed as `N x 3`. `assemble_additive` returns unweighted Cartesian `Kadd` and `Hadd[axis, displacement-SoA, velocity-SoA]`; `Hadd` is not symmetrized. Raw qraw cell data uses row-major `H` with lattice vectors in columns. GPUMD extended XYZ emits column lattice vectors and is transposed on read.

The package follows the Task 1 `GPJAADD1` little-endian wire sequence exactly. Each site stores ordered `(neighbor, image)` records, `ell` in eV/Å, and symmetric local `B` in eV/Å². The reader checks dimensions, finite values, symmetry, connected graph, reverse image edges, positive frame counts, response budget, and no trailing bytes. The writer refuses overwrite and installs the finished file atomically. `read_raw`, `read_additive`, `write_additive`, and `assemble_additive` are importable APIs.

## Fit method and limits

The local basis includes all symmetric cross-blocks within each fixed cutoff star. SVD removes the exact algebraic kernel of local-to-global stiffness assembly and chooses the minimum local Frobenius norm gauge. The force design uses mass-weighted internal coordinates and projected physical forces. A rank-deficient training design is rejected without ridge regularization. The reference center term is fixed from the raw total gradient via a unit-weight graph Laplacian; edge coefficients are split between sites.

The unregularized least-squares fit is checked against the full internal spectrum. If it violates `epsilon`, the solver accumulates low-eigenvector constraints, solves the whitened SVD quadratic program by dual projected coordinate descent, checks feasibility and complementarity, and repeats the full eigensolve. The held-out response check compares the complete internal covariance matrix, including off-diagonal terms, with `k_B*T*D0^-1`, using GPUMD's `k_B = 8.617343e-5 eV/K`. Its acceptance metric is the largest absolute eigenvalue of the response-whitened error `P^-1/2 C P^-1/2-I`; a Frobenius diagnostic is informational only. Force-position IBP uses the largest singular value of `(cross+k_B*T*I)/(k_B*T)`. The held-out actual-force residual uses the uncentered displacement from the training reference, while covariance and IBP use centered fluctuations. The package is written only when response and IBP errors fit the requested tolerance and shifted Cholesky reconstruction `eta` is below `epsilon/2`.

This is a dense small-system implementation. It preflights raw internal dimension before scanning the qraw derivative payload or opening the sample archive, then preflights local parameter count, gauge-map size, and force-design size. Current limits are 384 internal coordinates, 1,600 local parameters, 6,000,000 dense scalar elements, and 256 MiB of required uncompressed NPZ members. NPZ inputs use NumPy's public NPY 1.0/2.0 header readers; other NPY versions are rejected. Header-declared shapes and payload byte counts are checked before loading, and unrelated archive members are not loaded. qraw length is checked before variable-size vectors are read. These are conservative code guards, not a promise of acceptable runtime near the limit. Collection currently loads each entire bead trajectory into memory; large trajectories should first be tested with short slices. The tool does not attempt 5,896-atom fitting or sparse certification.

Both output writers use unique exclusive temporary files and no-overwrite installation. Collection installs the NPZ and mean model as a pair and removes outputs installed by the current call if the second installation fails; pre-existing outputs are never removed. No additional trajectory spread or time-step rejection threshold is imposed; the median sampling interval is reported.

Still outside Task 1 are CUDA/C++ prepare integration, runtime bead-count checks, real-data convergence, finite-size validation, DeltaJ correlation tests, and material claims.
