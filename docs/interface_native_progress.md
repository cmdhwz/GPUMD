# Interface native implementation progress

## Completed pass: M3 SHC sampling and output buffering

- Preserved the sample pairing: NEP force derivatives are accumulated during the sampled force evaluation, then paired with the velocity after the same integration step's second half-step.
- Reused one GPU plane-flux buffer of `block_samples * plane_count` values. Each sample has a separate row; the host receives rows only at a complete block boundary or for the final tail.
- Kept the atom coefficient/velocity ring bounded by `selected_plane_atoms * Nc`. Per-sample capture no longer copies flux or error flags to the host and does not write result rows.
- Made displacement, missing-slot, non-finite derivative/coefficient/velocity failures sticky through the buffered block. The first force-side failing sample step is retained; the block is rejected at its boundary, or at `post_run` for a tail. Aggregate plane-flux and correlation overflow is checked after block D2H and reports the affected sample row or block/plane/lag without adding per-lag atomic-result checks.
- Retained all sampled flux metadata/rows and each complete block's raw positive/negative correlations in preallocated host arrays; block D2H copies land directly in their final run-cache spans, and a tail copies only its valid rows. Fourier transforms, Welford updates, and output rows now run in `post_run` in block order. Tail flux is output, while tail correlations remain discarded.
- Added checked run-sized cache calculations and reports for GPU and host cache bytes. A stop before `post_run` can leave files with headers but without buffered results.
- Added host regressions for direct cache spans across multiple blocks and a tail, spectrum omission for the tail, the GPU fill index limit, and sticky first-error/reset behavior. The existing empty-plane runtime verifier still checks zero output with no selected plane atoms.

## Completed pass: M1 interface-profile GPU block accumulation

- Kept the regular `compute_chunk` sampling path unchanged. Interface modes now reuse the raw GPU count, kinetic-energy, potential, and optional mass/momentum reductions, then apply the production COM/DOF/temperature helper per bin and per sample on the GPU.
- Accumulated wide atom-count totals, kinetic/potential energy, DOF, temperature sum, and valid-temperature sample count in one preallocated GPU block buffer. A GPU sticky error records the first invalid step/bin and its raw kinetic, mass, momentum, and potential values; blocks are checked before host caching.
- Replaced interface per-sample count/value/momentum D2H copies and CPU vectors with one block-boundary D2H copy into a run-sized per-bin cache. Complete blocks and the final partial block keep their own sample/time/bath metadata; `post_run` writes all interface rows. An interruption before `post_run` can leave a header-only profile.
- Precomputed the run's complete-plus-tail cache capacity, checked size arithmetic, vector capacity, and every interface GPU fill buffer against the existing `int` fill-kernel index limit, and printed the CPU/GPU allocation sizes.
- Extended the M1 native host/device-helper test for different per-sample COM momenta, variable count/DOF, empty and zero-temperature bins, error/reset behavior, wide count sums, and block/tail sizing. The C++ test and CUDA path are not compiled or run in this environment.

## Completed pass: M2 VACF/VDOS block cache and M3 block spectrum output

- M2 keeps the per-atom velocity ring and VACF accumulation kernel, then normalizes each raw lag sum on the GPU by its own origin count and selected bin atom count. The VDOS kernel consumes that raw normalized VACF directly; the raw-sum D2H/normalized-value H2D roundtrip and per-block output writes are removed.
- M2 copies only complete-block raw VACF/VDOS into preallocated final CPU caches. Block times/sample counts are cached, incomplete tails remain dropped, and all existing VACF-v1/VDOS-v2 rows and integrals are written from `post_run`. Empty-bin NaNs and populated zero-motion zero-spectra are preserved; non-finite populated-bin results fail at the block boundary.
- M2 estimates run-sized CPU cache and fixed GPU buffers with checked arithmetic, fill-index and vector-capacity limits, then reports both sizes. Its per-atom history remains bounded by `selected_atoms * Nc`.
- M3 adds `prefix_shc_blocks.out`, reusing the already computed per-block spectrum, `C0`, integrals, and same-block cached mean flux. The six data columns in the correlation file remain unchanged; its header now identifies geometry, area, and sampling parameters.
- Added a C++14 host-only CMake/CTest entry for M0 geometry, M1 math, M2 VACF normalization, and M3 math/time/statistics. M2 tests cover lag-specific origins, separate block normalization, empty/zero channels, and non-finite populated data. M3 tests check two block spectra/fluxes against cumulative means; runtime verifiers check the new 11-column block file.
- Native tests, CUDA compilation, and GPUMD runtime fixtures have not been compiled or run in this environment.

No Python files are changed in this pass. Further GPU atomic reduction or VACF-kernel changes are deferred until a matched performance/correctness measurement is available.

Review artifacts: `.native_review/perf-r1-baseline/` contains the before snapshot for the SHC buffer refinement; `.native_review/perf-r3-baseline/` contains and verifies the 70-file before snapshot for this pass. `.native_review/perf-r2-baseline/` was absent after the preceding pass, so no M1 before snapshot is claimed. Keep `.native_review/` out of release packages; the directory is not gitignored and was not staged.

## Current pass: M3 sparse candidate map and capture specialization

- Replaced the dense `N * plane_count` atom-to-slot table with `candidate_offsets[N+1]` and `candidate_planes[S]`, where `S` is the existing sum of compact per-plane selected atoms. Per-plane atom ordering and configured plane IDs are retained; the receiving atom's compact slot is found by binary search in that plane's sorted atom slice.
- Built sparse candidate arrays from the same selected slots used by coefficient and velocity sampling. GPU and host mapping-byte reports now count the sparse offsets/plane IDs; the old dense map and `N * plane_count` int-index rejection are removed.
- Specialized the regular large-box radial, angular, and ZBL kernels on `CapturePlaneFlux`. Ordinary NEP calls dispatch once per force evaluation to the enabled or disabled specialization; temperature large-box calls explicitly use the disabled specialization. Force, energy, virial, and atomic reduction formulas are unchanged.
- Added C++ assertions comparing sparse lookup/candidate coefficient results to the dense reference for unsorted plane IDs, periodic seams, both directions, exact boundaries, overlapping candidates, far atoms, and an empty plane. The real-NEP CUDA consistency fixture now constructs and consumes the production sparse mapping.
- M3 correlation and block-spectrum headers now use the validated nominal sample interval instead of the later-reset runtime sample-time member. The C++ runtime-output verifier reads `run.in` and checks both headers for positive `dt_ps == time_step * TIME_UNIT_CONVERSION * sample_interval / 1000`.
- Force-kernel and host assertions, CUDA build, and runtime fixtures remain unrun because no CUDA/C++ compiler or GPUMD executable is available. Existing force and correlation atomic reduction layouts are unchanged pending GPU timing.

Review artifact: `.native_review/perf-r4-baseline/` contains the verified before snapshot for this pass (25 files) at the existing feature branch and HEAD. Keep `.native_review/` out of release packages; it is not gitignored and was not staged.

## M4 local selected-atom snapshot follow-up

- `dump_interface_atoms` now enables only the 3N periodic image counters and reconstructs selected-atom unwrapped coordinates from the wrapped coordinates, image counts, and the fixed orthogonal M0 box. It does not allocate the full-system unwrapped array or launch its per-step update kernel. A later standard unwrapped-position consumer reuses the same image history.
- GPU staging is capped at the smaller of the configured block size and scheduled frame count. The block-to-run-cache copy checks both its source and destination spans before D2H.
- The runtime fixture separates a full-system `dump_xyz` baseline from a local-only first run, compares the 17-column rows across a periodic seam, and checks late unwrapped-position enablement in a second run. The host pack assertions cover positive and negative image counts on all axes.
- This pass has a before snapshot at `.native_review/native-r6-baseline/`. Native host/runtime tests remain unbuilt and unrun because the environment has no available C++/CUDA toolchain or GPUMD executable.

## Native closure pass: independent sampling planes, SHC verification, M4-M6 contracts

- `InterfaceGeometry` now keeps required physical `interfaces` separate from optional `sampling_planes`. Each list uses the same PBC wrap/sort/range/duplicate validation; omitted sampling planes copy the sorted physical interfaces, while an explicit empty list is valid for geometry and rejected only by SHC. Version-2 JSON preserves `interfaces_A`, adds `sampling_planes_A` and the sorted zero-based SHC ID rule. M1 bin crossing remains based only on physical interfaces; all SHC mapping, counts, and output plane IDs use the sampling list.
- Host assertions cover list separation, fallback, independent normalization, wrapped duplicates, and explicit emptiness. The geometry-only and shared-geometry runtime verifiers check the serialized metadata. All four SHC output headers identify the geometry file and sorted sampling-plane coordinates.
- The real-NEP CUDA consistency program now checks view-on/off energy/force/virial, finite-difference coefficient signs on both sides of a nonperiodic plane and on both sides of each periodic slab boundary, the periodic slab energy-rate identity, and a two-step-size velocity-Verlet convergence check using the production sparse candidate map. The periodic test wraps its displaced reference positions before mapping and differentiates the energy partition opposite the atom's side of each boundary.
- Host CMake now defines native fixture preparation and verification utilities without a GPUMD link. When an external `GPUMD_EXECUTABLE` is supplied, CTest registers a short carbon NVE/profile/SHC run, explicit-empty-SHC rejection, physical-versus-sampling geometry split, and unsorted legacy-interface fallback. The 20,000-step heat_lan check is opt-in. Fixtures use fixed mode directories under the build runtime tree and reference the repository carbon model without copying it. Their thresholds are source-defined software checks and remain uncalibrated.
- Documentation now distinguishes M4 local snapshots from offline mode projections, records the M5/FCP precision, displacement, and `F_ref` limitations, and specifies same-configuration IFC2 residual analysis and halo needs for M6. It records the absence of a t=0 local-dump frame, warns that M0 reference coordinates are not an equilibrium harmonic `R0`, and documents post_run timing and CPU/GPU buffer scaling.
- `run.cu` still reports the existing MD-loop time/speed, and now also reports post_run duration and timed-loop-plus-post_run total duration/speed.
- Static source review and whitespace checks are the only available verification. No host test, CUDA target, GPUMD runtime fixture, or GPU/material acceptance has been run because C++/CUDA/CMake/GPUMD tools are unavailable. Python sources and tests were not changed.

## Directed closure review pass

- Periodic-seam coefficient finite differences now use the boundary-specific sign: the seam is the slab's left boundary and the internal plane is its right boundary. The periodic view-on evaluation is compared with same-configuration view-off energy/force/virial through the shared assertion helper.
- The two-dt NVE test now checks endpoint and maximum running slab-energy ledger residuals in eV against `1e-3 eV + 10% of signed integrated boundary exchange`. It requires nonzero exchange; the 0.1 fs residual must improve over 0.2 fs unless both are below the 1e-3 eV floor. The test prints delta E, integral, absolute exchange activity, endpoint/max error, and tolerance.
- Native runtime fixtures use the full Nyquist bound for `dt=0.5 fs` and retain 65 exported frequency points. The verifier reads run settings and output metadata, checks sample/plane/block/frequency ordering and both endpoints, and independently trapezoids the written rows against the same-block mean flux and C0 with a rounding-aware absolute-plus-relative tolerance.
- Added actual GPUMD `geometry_split` and `geometry_legacy` modes. The split case puts the physical interface inside bin 1 and its sampling plane inside bin 4, requiring only bin 1's M1 crossing flag while SHC reports the independent plane. The legacy case omits `sampling_planes`, supplies `interfaces 2 7 3`, and verifies both serialized lists sort to `[3, 7]` with actual M1 and SHC outputs.
- Heat verification now uses 10,000 warm-up steps followed by candidate production at reported steps 10001-20000, checks two 5,000-step halves, per-signal block variability, paired source/sink and plane/bath residuals, and blockwise total-system-energy slopes. Excess residual-SE or unstable half windows fail as statistically insufficient; adjacent blocks are not described as independent confidence intervals. The acceptance thresholds remain uncalibrated.
- Runtime cleanup is restricted to the normalized fixed mode child directly under a configured build directory's `runtime` folder; symlink/reparse escapes are rejected before removal. CMake rejects a nonempty invalid `GPUMD_EXECUTABLE`, and the script requires `VERIFY` even for the expected-error empty-plane case.
- JSON/groups/dump headers now state `reference_positions_role=geometry_mapping_only`, `equilibrium_reference=external_required`, and the absence of a t=0 atom-dump frame. Value columns are unchanged. The physical geometry test remains all-carbon software-path validation, not a heterogeneous-material model.
- The shared parser handles `interfaces` and `sampling_planes` through one count/coordinate-reading branch while keeping distinct errors. No test/build can run here; the runtime modes and CUDA finite-difference tests remain source-only.

## Final directed verifier corrections

- Updated fixture expectations to match the production `step + 1` sample index for profile and SHC rows, including block start times and the 10001-20000 heat production window after 10000 warm-up steps.
- The exported-spectrum trapezoid check now accumulates absolute spectral area and uses a `5e-11 * absolute_trapezoid` allowance for 12-significant-digit output quantization. It still checks C0 against the block mean flux with the existing stricter comparison and verifies nonzero NVE exchange.
- Corrected the NEP test description: the checked-in single-carbon NEP4 fixture covers ordinary radial/angular coefficients; it does not enable ZBL or establish a multi-element interface result.
- M2 and M3 docs now distinguish nonoverlapping blocks with reset accumulators from statistical independence; adjacent blocks are not guaranteed independent.
- These source/test/document corrections remain uncompiled and unrun because this checkout environment has no C++/CUDA compiler, CMake, or GPUMD executable.
