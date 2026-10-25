# Interface native implementation progress

## Completed pass: M3 SHC sampling and output buffering

- Preserved the sample pairing: NEP force derivatives are accumulated during the sampled force evaluation, then paired with the velocity after the same integration step's second half-step.
- Reused one GPU plane-flux buffer of `block_samples * plane_count` values. Each sample has a separate row; the host receives rows only at a complete block boundary or for the final tail.
- Kept the atom coefficient/velocity ring bounded by `selected_plane_atoms * Nc`. Per-sample capture no longer copies flux or error flags to the host and does not write result rows.
- Made displacement, missing-slot, non-finite derivative/coefficient/velocity failures sticky through the buffered block. The first force-side failing sample step is retained; the block is rejected at its boundary, or at `post_run` for a tail. Aggregate plane-flux and correlation overflow is checked after block D2H and reports the affected sample row or block/plane/lag without adding per-lag atomic-result checks.
- Retained all sampled flux metadata/rows and each complete block's raw positive/negative correlations in preallocated host arrays; block D2H copies land directly in their final run-cache spans, and a tail copies only its valid rows. Fourier transforms, Welford updates, and output rows now run in `post_run` in block order. Tail flux is output, while tail correlations remain discarded.
- Added checked run-sized cache calculations and reports for GPU and host cache bytes. A stop before `post_run` can leave files with headers but without buffered results.
- Added host regressions for direct cache spans across multiple blocks and a tail, spectrum omission for the tail, the GPU fill index limit, and sticky first-error/reset behavior. The existing empty-plane runtime verifier still checks zero output with no selected plane atoms.

## Current pass: M1 interface-profile GPU block accumulation

- Kept the regular `compute_chunk` sampling path unchanged. Interface modes now reuse the raw GPU count, kinetic-energy, potential, and optional mass/momentum reductions, then apply the production COM/DOF/temperature helper per bin and per sample on the GPU.
- Accumulated wide atom-count totals, kinetic/potential energy, DOF, temperature sum, and valid-temperature sample count in one preallocated GPU block buffer. A GPU sticky error records the first invalid step/bin and its raw kinetic, mass, momentum, and potential values; blocks are checked before host caching.
- Replaced interface per-sample count/value/momentum D2H copies and CPU vectors with one block-boundary D2H copy into a run-sized per-bin cache. Complete blocks and the final partial block keep their own sample/time/bath metadata; `post_run` writes all interface rows. An interruption before `post_run` can leave a header-only profile.
- Precomputed the run's complete-plus-tail cache capacity, checked size arithmetic, vector capacity, and every interface GPU fill buffer against the existing `int` fill-kernel index limit, and printed the CPU/GPU allocation sizes.
- Extended the M1 native host/device-helper test for different per-sample COM momenta, variable count/DOF, empty and zero-temperature bins, error/reset behavior, wide count sums, and block/tail sizing. The C++ test and CUDA path are not compiled or run in this environment.

## Follow-up candidates, not implemented in this pass

- M2: VACF origin accumulation and per-block output. Preserve independent origin counts by lag and block-local reset while reviewing whether the current GPU accumulation causes excess atomics or synchronization.
- Reduction kernels: consider plane/bin-local aggregation only after a matched correctness reference exists; do not change atomic ordering or block statistics solely for speed.

No M2 optimization or Python changes are included in this pass.

Review artifact: `.native_review/perf-r1-baseline/` contains the before snapshot for the SHC buffer refinement. The requested `.native_review/perf-r2-baseline/` directory was absent when checked after implementation, so no M1 before snapshot is claimed. Keep `.native_review/` out of release packages; the directory is not gitignored and was not staged.
