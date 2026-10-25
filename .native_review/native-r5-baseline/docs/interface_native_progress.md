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
