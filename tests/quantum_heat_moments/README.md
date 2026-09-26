QHM regression inputs are ready but not executed (`ADDED_NOT_RUN`). Each directory uses the existing Si_2022_NEP4_5body model and a compact diamond-Si geometry. Run `gpumd` with that directory as the working directory.

- `edgecheck_unique`: orthogonal 10.86 A cell; `2 rc < L`, with radial and angular NEP contributions. The fixed-coordinate checker writes h, h/2, h/4, h/8 first-derivative and local-energy third-directional-derivative stencils, then repeats those stencils at the selected type-pair radial cutoff `rc_pair ± max(4h, 1e-4 rc_pair)`. Side probes are skipped with explicit rows if no unique non-self pair is available or if the probe stencil reaches half the shortest lattice vector.
- `edgecheck_triclinic`: sheared cell for the Cartesian closest-image link diagnostic and local-edge derivative check.
- `edgecheck_small_box`: 5.43 A periodic cell with multiple images inside the 5 A NEP cutoff; cutoff-side probes emit explicit skip rows when no unique non-self pair has a stencil wholly inside the nearest-image radius.
- `edgecheck_self_image`: 4.5 A cell; periodic self-image edges lie inside the 5 A cutoff and are counted as `not_coordinate_fd_testable` rather than compared with a physical-atom coordinate derivative.
- `edgecheck_explicit_supercell`: 3x3x3 explicit replication of the 4.5 A cell in a 13.5 A box (`2 rc < L`). Former self-images are distinct atom IDs with a unique periodic image; the checker selects the non-self radial edge nearest its type-pair cutoff as `cutoff_reference_unique_edge`, providing the independent coordinate-FD comparison.

The profile row is a pre-candidate static snapshot: it includes A0/current work and enabled edgecheck work, while its no-cache estimate covers only exact μ0/μ2/μ4 requests. Candidate work is reported separately in metadata and overlaps the action-wide NEP and wall-time totals.

The CPU math regression target is `quantum_heat_moments_cpu_test` under `BUILD_TESTING`; it is also `ADDED_NOT_RUN`. These inputs are source assets only and are not registered with CTest.
