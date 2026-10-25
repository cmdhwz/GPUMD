# Interface NEMD analysis

## Native implementation status

The native path now has M0 shared fixed-reference geometry, M1 fixed-mapping profile output, M2 per-atom local VACF/VDOS, and M3 plane-resolved SHC for a single regular NEP large-box potential. M2 caches complete-block raw VACF/VDOS on the host and writes rows from `post_run`; its v1/v2 data columns are unchanged. M3 adds `*_shc_blocks.out` with each block's spectrum and mean flux while retaining cumulative v3 `*_shc.out`. M3 is still pending compilation and real NEP finite-difference/runtime validation. The separate legacy `compute_chunk ... interface ...` output remains version 2 with 17 columns. `compute_interface_profile` writes version 3 with 19 columns; see [the native input/output contract](../doc/gpumd/input_parameters/interface_geometry.rst).

Python post-processing has not been updated or reviewed for the version 3 profile and VACF/VDOS outputs. `tools/interface_analysis.py`, its tests, and its example configuration remain pending and must be treated as incompatible with the new native profile. The original `compute.out` and legacy `compute_chunk.out` paths are unchanged.

## Online profile

In `run.in`, the opt-in command is:

```text
compute_chunk <sample_steps> <samples_per_block> bin/1d <x|y|z> lower <bin_width_A> interface <interface_A> <lower_id> <upper_id> <remove_bin_drift_yes_no> <new_output_file>
```

It writes one row per bin per output block from `post_run`. Each sample's raw bin reductions are finalized and accumulated on the GPU, with one host copy for each completed or residual block. The profile cache is bounded by the configured run length and contains only per-block/per-bin sums and metadata; an interruption before `post_run` can leave only the file header. It assigns bins by current coordinates, labels bins that straddle the interface `-1`, excludes atoms in active `fix` and `move` groups, and writes block averages of atom count, thermal kinetic energy, site potential energy, temperature, and degrees of freedom. `yes` subtracts each bin's mass-weighted drift and removes three degrees of freedom for each populated bin. Temperature is averaged only over samples with positive degrees of freedom; version 2 adds `temperature_samples` and writes `nan` when no temperature sample was valid. Other quantities still average over every sample. Bath energies use GPUMD's system-to-bath sign: source energy decreases while the source supplies heat, and sink energy increases while the sink removes heat. The counters are snapshotted at each sampling step, so a final partial block reports counters corresponding to its own `time_ps`. This output is available for `heat_lan`, `heat_nhc`, and `heat_bdp`; other ensembles write `nan` counters.

The output filename must not exist. Each legacy version 2 profile begins with `#` metadata, including the timestep, sample/block stride, bin geometry, cross-sectional area, PBC flags, static orthogonal geometry validation, material IDs, drift choice, time/energy sign convention, and column names. Interface profiles require an orthogonal, unchanged box and unchanged PBC for the run. Length is Å, time ps, energy eV, temperature K, and area Å². Time is converted from GPUMD natural time using its source constant. Version 2 has 17 columns and includes `temperature_samples`; the version 3 shared fixed-mapping profile has 19 columns and adds the geometry-derived `bath_mask` and `interface_crossing` fields.

## Python post-processing draft (not validated for this native output)

The following describes an unfinished analysis path. Do not use it as a supported workflow: the script and Python tests have not been updated for the shared version 3 profile or the new local VACF/VDOS files.

Run:

```text
python tools/interface_analysis.py examples/interface_nemd/analysis.json
```

The JSON file lists the heterostructure and pure-material profiles, the steady-state time window, block length, material-side fit regions, pure-reference coordinates, and a `power_fraction` for each trajectory. The transverse area is read from the profile metadata; an optional `area_A2` value is checked against it. `power_fraction=0.5` represents an explicitly configured symmetric two-path periodic geometry; use `1.0` for a single path or the physically justified path fraction for another geometry. The script does not infer periodic path multiplicity. Optional per-run `bath_regions_A` entries are coordinate intervals supplied by the user; overlapping fit or segment intervals are rejected, and bath-overlapping profile bins are excluded from spatial estimates. The script does not infer bath coordinates from group IDs.

The script writes:

* `interface_resistance.out`: fitted interface-side temperatures, bath powers, energy-balance diagnostic, flux, and interface resistance.
* `interface_segment_resistance.out`: optional arbitrary same-material endpoint intervals from the `segments` list, with propagated temperature and flux errors.
* `interface_kappa_profile.out`: cumulative endpoint resistance/conductivity, sliding weighted-gradient fit, local conductivity, fit quality, and block errors.
* `interface_comparison.out`: pure-material versus heterostructure excess resistance and conductivity ratio, with local mean temperatures from the configured material-side `fit_A` intervals, the intervals and bath exclusions used, powers, areas, power fractions, and boundary labels for each run. Different powers are allowed. Without `comparison_temperature_tolerance_K`, each row says `NOT_CHECKED`; with it, each row says `MATCHED` or `NOT_MATCHED` based on the local mean-temperature difference.
* `interface_analysis.svg`: six-panel vector plot with time-block error bars.

The cumulative estimate uses the fitted temperature extrapolation at the interface separately on each side. Local conductivity uses a sliding weighted line fit. Segment conductivity uses endpoint temperatures. These are distinct estimators and are not averaged together. Fit regions must exclude thermostat-contact bins and any interface-straddling bins; configured bath intervals are checked explicitly.

The bath-power convention is:

```text
s_source = slope(E_source_raw, t)     # system-to-bath, eV/ps
s_sink   = slope(E_sink_raw, t)       # system-to-bath, eV/ps
P_hot    = -s_source
P_cold   = -s_sink
P        = (P_hot - P_cold)/2
balance  = abs(P_hot + P_cold)/(abs(P_hot) + abs(P_cold))
J        = P * power_fraction / area
```

Flux uses `1 eV/(ps Å²) = 1.602176634e13 W/m²`; conductivity uses `1 eV/(ps Å K) = 1602.176634 W/(m K)`. The actual target temperature, run labels, boundary conditions, area, and path fraction must be recorded in the configuration. The script reports `eta = kappa_hetero/kappa_pure`; a ratio above one is a matched finite-size comparison, not evidence by itself of a changed intrinsic IFC.

Frames are intended to enter a steady-state window only when `block_start_ps >= start` and `time_ps <= end`. The unverified Python draft uses `temperature_samples` for the valid-temperature weighting rule; it has not been adapted to the additional version 3 geometry columns or validated for the new fixed mapping.

Within each complete block the analyzer refits bath power, builds that block's temperature profile, and recomputes interface, cumulative, segment, and local conductivity results. Their standard errors come from those paired block results, so shared temperature/flux variation stays paired. Pure-reference trajectories are independent and their errors are combined with `hypot`. A single complete block is rejected; an invalid or zero local gradient yields `nan` conductivity/error rather than a finite precision estimate. Choose a block length longer than the relevant correlation time and repeat with longer blocks before interpreting precision.

## Scope and validation boundary

The M3 SHC view observes the existing regular NEP radial, angular, and optional ZBL derivative kernels without changing force/energy/virial accumulation. It accepts only NVE and the supported heat-lan/NHC/BDP ensembles, and rejects small-box expansion, temperature-dependent or charge NEP, DFTD3, multi-potential, external force, and driven-work paths. Its reference-image guarantee depends on the configured displacement bound; this is not a general diffusive-trajectory flux method. M4 mode projections, M5 wavepackets, M6 anharmonic modal power, M7 mean-free-path analysis, and M8 mass-scaling mismatch control-variable analysis remain unimplemented.

Native C++ assertion and GPUMD runtime fixtures are present but were not run because no CUDA/C++ compiler or GPUMD executable is available in this environment. No Python tests were run in this pass. The bath-counter sign and geometry behavior still require execution on a provisioned GPU build.
