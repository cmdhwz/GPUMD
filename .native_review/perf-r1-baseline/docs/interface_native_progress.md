# Interface native implementation progress

## Current pass: M3 SHC sampling and output buffering

- Preserved the sample pairing: NEP force derivatives are accumulated during the sampled force evaluation, then paired with the velocity after the same integration step's second half-step.
- Reused one GPU plane-flux buffer of `block_samples * plane_count` values. Each sample has a separate row; the host receives rows only at a complete block boundary or for the final tail.
- Kept the atom coefficient/velocity ring bounded by `selected_plane_atoms * Nc`. Per-sample capture no longer copies flux or error flags to the host and does not write result rows.
- Made displacement, missing-slot, non-finite derivative/coefficient/velocity failures sticky through the buffered block. The first failing sample step is retained; the block is rejected at its boundary, or at `post_run` for a tail.
- Retained all sampled flux metadata/rows and each complete block's raw positive/negative correlations in preallocated host arrays. Fourier transforms, Welford updates, and output rows now run in `post_run` in block order. Tail flux is output, while tail correlations remain discarded.
- Added checked run-sized cache calculations and reports for GPU and host cache bytes. A stop before `post_run` can leave files with headers but without buffered results.
- Added host regressions for block ordering, full-block plus tail indexing, spectrum omission for the tail, and sticky first-error/reset behavior. The existing empty-plane runtime verifier still checks zero output with no selected plane atoms.

## Follow-up candidates, not implemented in this pass

- M1: profile accumulation and output reduction costs. First measure the per-sample host transfers and reduction synchronization after the current profile correctness path is stable; batch only if profiling shows they matter.
- M2: VACF origin accumulation and per-block output. Preserve independent origin counts by lag and block-local reset while reviewing whether the current GPU accumulation causes excess atomics or synchronization.
- Reduction kernels: consider plane/bin-local aggregation only after a matched correctness reference exists; do not change atomic ordering or block statistics solely for speed.

No M1/M2 optimization or Python changes are included in this pass.
