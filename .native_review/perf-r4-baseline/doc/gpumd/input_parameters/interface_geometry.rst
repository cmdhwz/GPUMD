.. _kw_interface_geometry:

.. index::
   single: interface_geometry (keyword in run.in)
   single: compute_interface_profile (keyword in run.in)
   single: compute_interface_dos (keyword in run.in)
   single: compute_interface_shc (keyword in run.in)

Shared interface geometry, profile, and local VDOS
==================================================

These opt-in commands use one fixed atom-to-bin mapping for local profile and velocity-autocorrelation outputs. They do not change :ref:`compute_chunk <kw_compute_chunk>` dynamic binning or :ref:`compute_dos <kw_compute_dos>`.

Syntax
------

Declare the geometry before either dependent command::

  interface_geometry <config_file> <output_prefix>
  compute_interface_profile <sample_steps> <samples_per_block> <remove_bin_drift> <output_file>
  compute_interface_dos <sample_interval> <Nc> <block_samples> <num_omega> <max_omega_rad_ps> <bin_lo> <bin_hi> <type_id|-1> <prefix>

One ``interface_geometry`` declaration is allowed per run. The shared geometry is initialized in ``Measure::pre_run`` after that run's integration initialization. Its mappings are released after all actions finish; declare it again for a later run. The generated geometry files and action outputs must not already exist.

Configuration file
------------------

The configuration is whitespace-delimited text. Blank lines and text after ``#`` are ignored. Each key must appear once, except ``bin_group`` is optional::

  axis z
  bin_width_A 5.0
  interfaces 2 3.0 7.0
  material_group 1
  bath_groups 0 0 1
  bin_group 2

* ``axis`` selects the binning coordinate.
* ``bin_width_A`` is finite and positive. Bins start at zero; the final bin may be shorter.
* ``interfaces`` gives a count followed by that many finite coordinates. Periodic-axis coordinates are wrapped into ``[0,L)`` at pre-run and sorted; duplicates after wrapping are rejected. Nonperiodic-axis coordinates must already lie in ``[0,L)``. A plane exactly on a bin edge does not mark either adjacent bin as crossing it.
* ``material_group method`` takes material IDs from the indicated ``model.xyz`` group-label column. It never infers material identity from position. Every atom, including bath atoms, retains this material ID. A bin with multiple material IDs, or no atoms, has ``material_id=-1``.
* ``bath_groups method source_id sink_id`` explicitly identifies source and sink labels. They must be distinct IDs in the selected method. Bath membership is stored independently from material identity. With ``heat_lan``, ``heat_nhc``, or ``heat_bdp``, the geometry must use grouping method 0 and exactly match the ensemble's ordered source and sink group IDs. Fractional-region ``heat_lan`` is rejected because the current mapping cannot identify those atoms from fixed group labels. For NVE and other unsupported bath-counter ensembles, the labels remain explicit metadata and are not checked against a thermostat.
* Optional ``bin_group method`` reuses fixed reference-layer labels from ``model.xyz``. Each label must equal the zero-based geometric bin ID calculated at pre-run.

Unknown, repeated, malformed, or missing required keys are errors. Group IDs and methods refer to the exact ``model.xyz`` group-label columns; the source/sink IDs should match the heat-bath groups configured for the run.

Geometry definition and outputs
-------------------------------

The current implementation accepts only a finite, positive-length orthogonal static box. Every periodic reference coordinate is wrapped into ``[0,L)`` before assigning bins. A nonperiodic reference coordinate outside ``[0,L)`` is an error; it is never clamped. ``deform`` or any other box/PBC change is rejected by a run-time guard. The geometry is frozen from wrapped atom positions at pre-run, after minimization and the run's integration initialization. This is a reference-layer mapping; atoms do not migrate between bins in this mode, and these outputs are not Eulerian cross-plane flux measurements.

The command creates these files once, refusing any existing path::

  <output_prefix>_geometry.json
  <output_prefix>_groups.dat
  <output_prefix>_bins.dat

The JSON metadata records the algorithm version, box matrix and PBC, selected axis, thickness, area, bin edges, interface planes, group methods/IDs, mapping rule, and units. ``groups.dat`` maps every zero-based atom ID to its bin, material and bath labels, active fixed/moving status, wrapped reference position, mass, and type. ``bins.dat`` records boundaries, volume, reference atom count, material ID, bath bit mask (source=1, sink=2), and interface-crossing flag.

Shared fixed-mapping temperature/energy profile
-----------------------------------------------

``compute_interface_profile`` reuses the ``compute_chunk`` GPU count, kinetic-energy, site-potential, and optional mass/momentum reductions. A GPU per-bin finalize step applies the same COM correction and temperature formula to each individual sample, then accumulates the block statistics on the GPU. It does not average momenta before COM correction. ``sample_steps`` is the sampling interval and ``samples_per_block`` is the number of samples per output block. ``remove_bin_drift`` is ``yes`` or ``no``; ``yes`` removes each bin's mass-weighted drift and subtracts three DOF for a populated bin. Atoms in the active ``fix`` and ``move`` groups are excluded from the profile. Thermostat atoms remain in its counts and energies, with ``bath_mask`` marking their reference bins so analysis can exclude them explicitly.

For this interface-profile mode, non-finite accumulated kinetic energy, mass/momentum sums, or COM correction fails closed. A negative corrected kinetic energy is clamped only within the count-scaled floating-point error envelope based on total and COM kinetic energy; a larger negative value is an error. Zero-DOF empty and single-atom COM-removed bins remain excluded from valid-temperature samples, while finite zero kinetic energy with positive DOF is a valid 0 K sample. The ordinary ``compute_chunk`` path is unchanged.

The file is version 3 and has 19 columns::

  step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A coordinate_center_A material_id bath_mask interface_crossing atom_count kinetic_energy_eV potential_energy_eV temperature_K dof samples temperature_samples bath_source_eV bath_sink_eV

Counts, energies, and DOF average over every sample. Temperature averages only samples with positive DOF, with that count in ``temperature_samples``; it is ``nan`` if there was no valid temperature sample. Bath counters are snapshots from the last sample in that output block, including a residual block at run end. Supported counter sources are ``heat_lan``, ``heat_nhc``, and ``heat_bdp``; other ensembles write ``nan``. Bath energy sign is ``system_to_bath``. Each complete block is reduced to one set of per-bin sums and copied to a preallocated host cache; the final incomplete block is also retained. All profile rows are written from ``post_run`` after the cached blocks are available. The action sizes that cache from the run step/sample settings and reports its CPU/GPU memory. A GPU data error is sticky within a block and is checked before caching; an invalid block stops the run with the first failing step/bin and its kinetic/momentum values. The legacy dynamic ``compute_chunk ... interface ...`` profile remains version 2 with its existing 17 columns and ordinary ``compute_chunk.out`` behavior is unchanged.

Local VACF and VDOS
-------------------

``bin_lo`` and ``bin_hi`` select a zero-based half-open range ``[bin_lo, bin_hi)``. ``type_id=-1`` selects every atom type; otherwise select the zero-based type ID. Atoms assigned outside the chosen bins, in either bath group, or in active fixed/moving groups are excluded. The GPU ring stores each selected atom's three velocity components; it does not replace atom velocities with a bin center-of-mass velocity. Memory is bounded by the selected atoms and ``Nc`` history samples, plus per-bin correlation/spectrum buffers; the allocated GPU byte count is printed.

``block_samples`` must be at least ``2*Nc``. Every complete block is independent and clears the VACF accumulator; an incomplete tail at run end is dropped and reported. At block completion, the GPU divides each raw VACF lag sum by its own origin count and selected atom count, then uses that raw VACF for the VDOS kernel. Empty bins keep raw VACF/VDOS as ``nan``; a populated zero-velocity direction keeps raw VACF/VDOS at zero and normalized VACF at ``nan``. Non-finite data in a populated bin aborts at the block boundary before any result rows are written. The time step must remain constant throughout the run, and ``max_omega_rad_ps`` must not exceed the angular Nyquist limit ``pi/dt``. Frequency is angular frequency in rad/ps.

The action copies only the normalized raw VACF and VDOS for each complete block into preallocated run-sized CPU caches. It writes the VACF, VDOS and integral rows in ``post_run``; a process interruption before then can leave only the output headers. The CPU cache scales with ``complete_blocks * bins * directions * (Nc + num_omega)``. GPU memory remains bounded by selected atoms times ``Nc`` plus per-bin VACF/VDOS work arrays. Both CPU and GPU sizes are checked and reported before the run.

``<prefix>_vacf.out`` contains block times, samples, per-lag origin count, bin, selected type, direction, lag time, raw VACF, and its lag-zero-normalized value. Velocities are converted to A/ps. An empty bin is ``nan``. Non-finite data in a populated bin aborts at the block boundary instead of being written as a number. A populated direction with exactly zero velocity has raw VACF and VDOS equal to zero and zero spectrum integrals; only its lag-zero-normalized VACF is ``nan`` because ``C(0)=0``. Negative finite ``C(0)`` is retained in the raw VACF/VDOS and is not normalized. ``<prefix>_vdos.out`` contains the raw lag-zero correlation, the analytic integral of the Hann-windowed one-sided spectrum over the requested range (divided by pi), a trapezoid integral of the actual exported frequency rows (also divided by pi), and ``S(omega)=dt*[C(0)+2*sum(w_l*C(t)*cos(omega*t))]``. The full-Nyquist normalization is ``integral_0^Nyquist S(omega) d(omega)/pi = C(0)``. ``frequency_grid_underresolved`` marks grids with ``dOmega*tau_max > pi``; this is reported rather than silently treating the written rows as a resolved spectrum.

``compute_interface_dos`` and the shared ``compute_interface_profile`` consumer accept classical MD only. The DOS and SHC actions compare the actual ``time_step`` supplied to the force hook against the configured run step on every integration step and reject adaptive or changing steps. Their sample interval is calculated as that fixed step times ``sample_interval``. The reported global times are checked for finite, increasing values, but accumulated global-time differences are not used to infer or validate the step size.

Plane-resolved NEP SHC
----------------------

.. _kw_compute_interface_shc:

The optional action uses the same frozen reference geometry and every configured interface plane::

  compute_interface_shc <sample_interval> <Nc> <block_samples> <numomega> <max_omega_rad_ps> <max_reference_displacement_A> <prefix>

It requires ``block_samples >= 2*Nc``, a constant step and sampling interval, and at least one interface plane. The sampled coefficient is accumulated in the existing regular NEP radial, angular, and optional ZBL derivative kernels; the force, potential, virial, and integrator formulas are unchanged. Each plane is counted independently, including a periodic seam plane. The supported path is one ordinary potential-only NEP model, without DFTD3 or another potential, using the regular large-box kernel. Only ``nve``, ``heat_lan``, ``heat_nhc``, and ``heat_bdp`` ensembles are accepted. Small-box expansion, temperature-dependent/charge NEP, other ensembles, and external driving or force actions are rejected. A periodic box must satisfy the exact NEP large-box criterion and ``rc + 2*max_reference_displacement_A < L/2`` in every periodic direction. Any non-finite coordinate/derivative, atom moving beyond the configured MIC displacement limit, or crossing bond missing its compact atom map, fails the run.

The correlation is ``C(+lag)=sum_j a_j(t) dot v_j(t+lag)`` and ``C(-lag)=sum_j a_j(t+lag) dot v_j(t)``. The instantaneous flux sign follows the configured ``+axis`` crossing convention; it is not automatically oriented from source to sink. Complete blocks are independent; incomplete tails are dropped. For each plane the spectrum is ``j(omega)=dt/pi*[C0+sum_lag>0(C(+lag)+C(-lag))*cos(omega*lag*dt)]``. Its full-Nyquist analytic integral equals ``C0``. The action rejects a frequency grid whose spacing fails ``dOmega*(Nc-1)*dt <= pi``. It outputs both the analytic integral and the trapezoid integral of the actual frequency rows; for a partial range the integrals refer only to ``[0,max_omega_rad_ps]`` and are marked ``band_limited``. Spectral errors are standard errors over complete paired time blocks and are ``nan`` until at least two blocks exist.

The action writes ``<prefix>_plane_flux.out`` (sampled plane flux and raw bath counters), ``<prefix>_shc_correlation.out`` (both correlation signs and origin counts), version 3 cumulative ``<prefix>_shc.out``, and ``<prefix>_shc_blocks.out`` (one spectrum row per complete block). The correlation file keeps its six data columns and records the geometry source, area, sample interval, ``dt_ps``, ``Nc``, and ``block_samples`` in its header. The per-block spectrum columns are::

  block block_start_ps block_end_ps block_samples plane_id omega_rad_ps C0_W_m2 mean_flux_W_m2 analytic_integral_0_max_W_m2 exported_grid_trapezoid_0_max_W_m2 jomega_W_m2_per_rad_ps

Each per-block row reports the spectrum and mean plane flux from that same complete block. The existing cumulative spectrum output and its 18 columns are unchanged. Its spectrum columns are::

  block block_start_ps block_end_ps block_samples accumulation_start_ps accumulation_end_ps accumulation_samples block_count plane_id omega_rad_ps C0_mean_W_m2 C0_SE_W_m2 analytic_integral_0_max_W_m2 analytic_integral_SE_W_m2 exported_grid_trapezoid_0_max_W_m2 exported_grid_trapezoid_SE_W_m2 jomega_mean_W_m2_per_rad_ps jomega_SE_W_m2_per_rad_ps

The ``block_*`` fields describe only the current complete block. The ``accumulation_*`` fields and the 64-bit ``accumulation_samples`` describe all complete blocks represented by each cumulative mean and standard error. Do not filter cumulative rows by the current block time to remove an earlier transient; recompute from the per-block correlation file or use an independent run. Bath counters use the ``system_to_bath`` energy sign and are finite for ``heat_lan``, ``heat_nhc``, and ``heat_bdp``; other ensembles write ``nan``. They are not converted into a pass/fail heat-current comparison. Geometry and atom-to-slot mappings remain fixed to pre-run reference coordinates; this output is a reference-plane diagnostic and does not re-bin atoms by instantaneous positions.

During the MD loop, the action keeps per-sample plane fluxes in a preallocated GPU block buffer and does not copy results to the host or write result rows. At each complete block it checks the sticky error flags, copies the flux block and block correlation arrays to preallocated host caches, and continues with the next block. The incomplete tail is checked and its flux rows are retained at run end, while its SHC correlation is discarded. All result rows, transforms, and cumulative statistics are emitted from ``post_run``. A process interruption before ``post_run`` can leave the already-created files with only headers. The GPU plane-flux cache scales with ``block_samples * plane_count``; the host cache scales with ``sample_count * plane_count`` plus ``complete_blocks * plane_count * Nc * 2`` correlation values. The action checks element/byte arithmetic before allocation and prints its GPU and host cache sizes.

NEP displacement, missing-slot, non-finite derivative/coefficient/velocity errors remain sticky for one sampled block and are checked at its end; for a residual tail they are checked in ``post_run``. A detected force-side error aborts the run and reports the first sampled step. Non-finite accumulated plane flux or raw block correlation is checked after block D2H and reported as a block-result overflow; those reductions do not retain a first contributing step. Both cases abort before result rows are emitted.

Scope
-----

This native implementation covers M0 fixed geometry, M1 fixed-mapping online profile, M2 local VACF/VDOS, and the regular-NEP large-box M3 plane-resolved SHC path. M4 mode projections, M5 wavepackets, M6 anharmonic modal power, M7 mean-free-path analysis, and M8 mass-scaling mismatch control-variable analysis are not implemented. ``tools/interface_analysis.py`` and its tests have not been updated to read the native profile version 3 or these VACF/VDOS/SHC files and remain incompatible.
