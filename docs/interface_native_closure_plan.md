# Interface native closure plan

## Scope and assumptions

- Work only on the current `feature/interface-phonon-dynamics` checkout. Preserve existing physics and file contracts except the explicitly requested sampling-plane metadata addition and run timing lines.
- `interfaces` remains the set of physical material boundaries used by profile geometry. `sampling_planes` is an independent SHC measurement set; omitted configuration copies the normalized physical interfaces, while an explicit count of zero is valid geometry and invalid only for SHC.
- Keep the existing selected-atom GPU history, block D2H, and `post_run` output model. Do not add Python, dependencies, unbounded GPU history, streaming, checkpoint, or force-kernel formula changes.
- The workspace does not provide a C++/CUDA compiler, CMake, or GPUMD executable. Native build and physical fixture results will therefore be reported as unrun; static/source checks are not represented as runtime validation.

## Implementation and acceptance checklist

1. Trace all geometry, interface-crossing, plane-view, SHC output, and sampling-time callers. Split physical interfaces from sampling planes with one shared validation/normalization path; retain sorted zero-based plane IDs and the legacy fallback. Verify JSON/output headers, profile interface-crossing labels, zero-plane behavior, and invalid/duplicate coordinates with native host tests and focused fixtures.
2. Extend the real-NEP CUDA consistency program rather than copying force derivatives: preserve the existing invariance/finite-difference/single-plane checks, add periodic multi-plane region-energy balance and finite-time NVE convergence checks using production sparse mappings. Add opt-in GPUMD NVE and `heat_lan` fixtures plus a host verifier/CMake entry that does not register absent-executable tests as passing.
3. Update M4-M6 input/output boundaries, FCP limitations, cache growth examples, and run timing. Preserve `post_run` output and existing MD-loop speed meaning while reporting `post_run` and full-run totals.
4. Check all changed callers, schema columns, assertions under Release/NDEBUG, Python-file status, `git diff --check`, and whitespace in added files. Report every native build/runtime/physical acceptance check that cannot run in this environment.

## Directed review corrections

1. Verify periodic seam and internal-boundary coefficient signs separately; compare the periodic view-on evaluation against same-configuration view-off energy/force/virial. Bound endpoint and running energy-ledger residuals by an absolute eV floor plus a relative exchanged-energy term; compare equal-duration NVE trajectories at both timesteps.
2. Make the runtime SHC fixture use the full Nyquist endpoint, check output step/plane/frequency order, and compare independently integrated exported rows with same-block flux and C0 using output-rounding-aware tolerances.
3. Add real Action parser fixtures for a nonempty physical interface distinct from a sampling plane, and for omitted sampling planes with unsorted physical coordinates. Preserve the homogeneous carbon fixture label as software validation only.
4. Make the heat-lan check reject unresolved block statistics, compare paired block residuals, inspect two candidate steady halves and total-energy drift, and avoid treating adjacent blocks as independent confidence intervals.
5. Bound runtime fixture deletion to a canonical direct child of the configured CMake build tree; reject invalid supplied GPUMD paths. Add machine-readable metadata for geometry-only reference coordinates and the absent initial dump frame, and share the coordinate-key parser branch without changing distinct diagnostics.
