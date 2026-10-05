# Finite-temperature qNEP RPMD-JA reference fit

This utility fits an auxiliary local finite-temperature force-response model from full-system RPMD bead dumps. It does not replace qNEP/PPPM raw reference generation or the production JA current. The coefficient package is consumed by CPU or GPUMD v3 preparation and is bound to the original raw-reference file, temperature, bead count, cell, atom types, and atom order.

## Workflow

Use one fixed-cell Hamiltonian, all mobile atoms, one target temperature, and one bead count throughout. First equilibrate the beads and collect aligned samples plus `mean.xyz`. Then generate the raw reference in a separate directory using that `mean.xyz` as `model.xyz`, with the same full-precision cell, atom types, masses, qNEP model, and PPPM configuration. This ordering is required because `fit` checks that the raw reference position `R0` agrees with the training mean. A production measurement need not start exactly at `mean.xyz`; it must use matching reference metadata and initialize within the fixed-reference branch.

```text
python tools/rpmd_ja_fit_reference.py collect --bead-files bead_0.xyz bead_1.xyz --temperature 300 --output samples.npz --mean-model mean.xyz --masses 1.008 15.999 --cell-model model.xyz
```

List the actual file path for every bead in `--bead-files`; the two shown here are only an illustrative two-bead example. Real eight-decimal `dump_beads` cells require `--cell-model` pointing to the original full-precision model. Omitting that option is only appropriate when the supplied XYZ cell already has sufficient precision for the downstream raw-reference checks. The collector writes `mean.xyz` from the first two thirds of the chronological sample frames.

In a separate raw-generation directory, place this `mean.xyz` under the input name `model.xyz`, and place the same qNEP model and U=8 kernel table there. Reuse the PPPM spacing from the original Hamiltonian (or the default 1.0 Angstrom spacing when that is the original setting); do not change electrostatics settings for raw generation. `generate_raw` must appear after `potential` and `kspace pppm` and before any `ensemble` or `run` command:

```text
potential nep.txt
kspace pppm 1.0
rpmd_ja generate_raw ja_raw.qraw 300 1e-4 ja_kernel_u8.tbl
```

Run GPUMD in that directory to create `ja_raw.qraw`. The command evaluates qNEP and PPPM on the GPU. Then fit the coefficient package:

```text
python tools/rpmd_ja_fit_reference.py fit --raw ja_raw.qraw --samples samples.npz --output ja_additive.bin --cutoff 5.0 --epsilon 1e-8 --response-tolerance 0.15
```

Supply one positive mass per atom, in the same order as the original model and the bead dumps. The two mass values shown are for an illustrative hydrogen/oxygen system; replace them with the complete actual mass list. GPUMD `dump_beads` writes its Lattice values to eight decimal places. With `--cell-model`, the collector checks every dump frame against that rounded serialization bound, then uses the original cell for alignment, sample metadata, and `mean.xyz`. An omitted `pbc` field in the model follows GPUMD's all-periodic default; an explicit field must be fully periodic (tokens are case-insensitive). The model must have the same atom count and ordered symbols as every dump.

Prepare the final v3 reference using the CPU utility:

```text
python tools/rpmd_ja_qnep_prepare.py ja_raw.qraw ja_additive.bin --kernel-table ja_kernel_u8.tbl --output ja_reference.bin --lossless
```

For production GPU preparation, put this before any `ensemble` or `run`:

```text
rpmd_ja prepare ja_raw.qraw ja_reference.bin ja_kernel_u8.tbl ja_additive.bin
```

Use `rpmd_ja on ja_reference.bin` in the fixed-cell RPMD measurement run, with matching reference metadata, full-system ordering, and an equilibrated bead-resolved restart initialized within the reference's 0.45 fractional branch. The measurement's instantaneous positions need not equal the training mean. A legacy native comparison can be prepared from the same raw reference and kernel table without an additive package. Existing current calculation is unchanged.

## Collection and fit conditions

Each input XYZ frame must contain atom positions and forces and the complete bead set at a common step. The force on each bead must be the physical force from that bead's sampled Hamiltonian; do not supply ring-spring forces or a force recomputed at the centroid `R_c`. The collector averages the bead forces once, as `sum_b F_b/P`, alongside the centroid construction. The fit assumes force and energy derivatives are consistent with the same Hamiltonian used to generate the raw reference. Independently assess qNEP/PPPM energy and force deviations against the accuracy target for the material before interpreting the fit. The collector aligns adjacent beads around the ring, then checks that its centroid equals the production bead-zero minimum-image centroid to floating-point roundoff. It keeps the ring-alignment helper general, but rejects branches unsupported by the current production centroid path. The sampled centroid is time-unwrapped and COM aligned. The fitter checks its continuous fractional displacement from raw `R0` against the existing 0.45 branch bound without wrapping integer-cell drift back to zero.

The chronological split reserves the later third of frames for held-out validation. This is a frame split, not independent-block uncertainty estimation; RPMD time correlation may reduce the effective validation sample size. Choose dump cadence to resolve the motion and avoid periodic aliasing. The code does not infer or impose a new timestep/spread threshold. Independent-block uncertainty and real-material sampling validation remain future checks.

The graph contains one fractional minimum-image per ordered distinct atom pair within the supplied cutoff. It excludes self-image edges and does not enumerate multiple images of the same atom pair. This graph defines a restricted coefficient model space; it is not the complete qNEP neighbor graph and does not guarantee that the selected cutoff or system size is sufficient. Assess response and force residuals on held-out configurations and perform cutoff/size checks before physical interpretation.

The local model stores symmetric Cartesian `B_i` blocks and a linear coefficient `ell_i`, with fixed image vectors. `B_i` determines additive Hessian and JA flow corrections. The total linear center term is canceled by the reference-gradient term, so the linear contribution to additive `H` is zero. The public assembler's `H` is oriented with displacement coordinates on rows and velocity coordinates on columns, as used in `x.T @ H @ v`; it is not symmetrized. In the v3 wire representation, `block_site_transpose` stores the mass-weighted transpose, `M^(-1/2) @ H.T @ M^(-1/2)`, in the site's ordered coordinates.

## Limits and output interpretation

The fitter refuses dense matrices over 6,000,000 elements, more than 1,600 dense parameters, translation-free internal dimensions over 384 (equivalent to at most 129 atoms), or sample arrays over 256 MiB uncompressed. CPU additive preparation separately refuses dense assembly above 4,000,000 elements and packages above 64 MiB. These are hard code limits, not recommended production sizes; NumPy, LAPACK, source arrays, and solver workspace increase actual host memory. GPU prepare has a separate preflight and memory use. No large-system runtime or scaling claim is made.

The workflow fixture uses synthetic full-precision coordinates and forces, while it rounds the cell field to exercise `--cell-model`. That gives a strict zero-addition recovery oracle. Real GPUMD dumps serialize lattice, coordinates, and forces to eight decimals. `--cell-model` recovers the original lattice only; position/force quantization sensitivity must be assessed separately on real dumps. The fixture's U=8 coefficient table is the checked-in `tests/data/rpmd_ja_kernel_U8.txt`; passing these tests does not validate physical interpolation accuracy of the kernel table.

`heat_current_rpmd_ja.out` contains the three directional centroid current, additive correction, and sum. `hac_rpmd_ja.out` contains the three directional `CAA` self-correlation components and their running integrals. The four terms `Ccc`, `CcΔ`, `CΔc`, and `CΔΔ` can be reconstructed for each lag offline from the three time series in `heat_current_rpmd_ja.out`; production does not write four HAC files.

The CPU integration test exercises CLI collect → fit → CPU prepare on synthetic raw data, including zero and nonzero additions, response refusal, source/package corruption refusal, independent energy/virial finite differences, stored v3 table/matrix conventions, mapped-current checks, gauge sensitivity, and all-four-term correlation reconstruction. It does not run GPUMD `generate_raw`, the C++/CUDA `rpmd_ja prepare` command, CUDA kernels, or a material trajectory. Real qNEP generation and preparation, CUDA compilation/runtime, eight-decimal coordinate/force sensitivity, independent-block uncertainty, and material validity are unverified.
