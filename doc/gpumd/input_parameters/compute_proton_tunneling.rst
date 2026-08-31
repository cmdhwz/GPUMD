.. _kw_compute_proton_tunneling:
.. index::
   single: compute_proton_tunneling (keyword in run.in)

:attr:`compute_proton_tunneling`
================================

This keyword enables a centroid-geometry observer for proton-state bias and persistent
O-H-O state changes. It does not modify the potential, forces, qNEP charges, or heat current.
For PIMD/RPMD it is called after the integrator has updated the centroid positions.

Syntax
------

::

  compute_proton_tunneling <sample_interval> <window_samples> <delta_cutoff> <hold_samples> <dOO_min> <dOO_max> <rperp_max> [O_symbol H_symbol] [oho_angle angle_deg] [ion_field ion1_symbol ion1_charge ion2_symbol ion2_charge cutoff] [local_environment ion1_cutoff ion2_cutoff H-Cl_cutoff H-Cl_angle_min] [bead_diagnostic [f_min span_min [center_max centroid_max]]]

For example::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H

To require a more linear O-H-O geometry, add the optional angle setting::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H oho_angle 150

With a nominal Na/Cl ion-field proxy enabled, use one physical input line::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H ion_field Na 1.0 Cl -1.0 8.0

To record a local Na/Cl environment for every valid O-H-O observation, append
``local_environment``. The first two cutoffs are for the two species configured by
``ion_field``; the third is the H--ion2 cutoff for the continuous O-H...ion2 angles,
and the last is the threshold for the accompanying ``hcl_like`` fractions::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H ion_field Na 1.0 Cl -1.0 8.0 local_environment 3.5 3.5 3.0 150

For RPMD/PIMD, enable the lightweight bead-resolved tunneling-like diagnostic with the
strict default thresholds :math:`f_{\min}=0.25`,
:math:`f_{\mathrm{center,max}}=0.30`, and
:math:`|\Delta_{\mathrm{centroid}}|_{\max}=0.10\,\mathrm{\AA}`, together with
the legacy diagnostic :math:`\text{span}_{\min}=2\,\text{delta\_cutoff}`:

::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H bead_diagnostic

The thresholds can be supplied explicitly as the minimum bead fraction in each signed
well, the legacy minimum bead-coordinate span, the maximum center-bead fraction, and the
maximum absolute centroid displacement, in Angstrom. The span is reported as
``two_well_span`` but is not required by the final strict tunneling label:

::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H bead_diagnostic 0.25 0.25 0.30 0.10

The strict classification is layered. ``two_well_delocalized`` requires both signed wells,
the center-fraction limit, exactly two signed-well interfaces after center beads are skipped,
and at most four complete cyclic state domains with at most two center domains. The legacy
``two_well_span`` flag additionally records the minimum span. To reject multi-domain patterns such as
``LL00LLRR00RR``, the implementation also requires at most four complete cyclic state domains.
``barrier_centered_tunneling_like`` additionally requires
:math:`|\Delta_{\mathrm{centroid}}|\le0.10\,\mathrm{\AA}` by default. Thus the latter is
the strict default label, while the former remains useful for detecting asymmetric salt-induced
two-well delocalization without forcing its centroid to the barrier center. Multi-kink or
multi-domain configurations are labeled ``multi_kink_or_multi_domain`` rather than being
called noise or strict tunneling.

Each finalized attempt keeps two representative bead configurations. ``centroid_best`` is
the sample with the smallest :math:`|\Delta_{\rm centroid}|`. ``delocalization_best`` is
selected lexicographically by largest :math:`\min(f_-,f_+)`, smallest :math:`f_0`, passing
``simple_two_domain_path``, largest ``robust_span`` (the 20--80 percentile span), and finally
smallest :math:`|\Delta_{\rm centroid}|`. The two selections can occur at different times in
an asymmetric salt-ice potential.

For each hydrogen, the nearest oxygen is used as an anchor and the second endpoint is selected
from the anchor's top-8 oxygen shell with a mutual-neighbor check. The accepted pair is ordered
by atom index and its state variable is

.. math::

   \Delta = d(\mathrm{H},\mathrm{O}_{\rm low}) - d(\mathrm{H},\mathrm{O}_{\rm high}),

where the selected oxygen endpoints are ordered by atom index. A state is positive when
:math:`\Delta > \text{delta\_cutoff}`, negative when it is below the negative cutoff, and
unassigned inside the dead band. A state change must persist for ``hold_samples`` sampled
frames before it is written as a transfer event.

When a hydrogen is in a stable state and enters the dead band, an attempt is opened. It is
closed as ``success`` after the opposite state is held for ``hold_samples`` frames, as
``return`` when the original state is recovered, as ``geometry_lost`` when the O-H-O pair is
lost or replaced, or as ``run_end`` if the simulation ends first. If the sampling interval
jumps directly from one signed state to the other, the first opposite-state frame starts an
attempt so that the event is not silently discarded; a small sampling interval is still
recommended when measuring barrier recrossing.

The optional ``oho_angle`` parameter is a minimum angle in degrees and defaults to 120. It must
be between 0 and 180 degrees. The O-O range and perpendicular-distance cutoff are only the
first geometric filter. The observer uses the nearest O atom to H as an anchor, considers only its eight nearest O
neighbors, and requires the candidate pair to be mutual top-8 neighbors. It also requires H to
lie between the two O atoms, the configured minimum O-H-O angle (120 degrees by default), and
both O-H distances below 1.60 Å. If multiple candidates are nearly tied, or multiple H atoms
are assigned to one O-O pair in the same frame, those samples are marked as assignment-ambiguous
and skipped; they are not counted as physical geometry loss. The numerical O-O range should be
calibrated from the ranked O-O distributions at the target pressure. For a 30 GPa structure,
2.65 Å is a reasonable starting upper bound, but it should be kept identical for pure and
salt ice during comparison.

Output
------

The observer appends six files by default, plus ``proton_bead_event.out`` when the optional
bead diagnostic is enabled. All records are accumulated during the run and these files are
opened and written once during ``postprocess``; no proton-observer ``fprintf`` or ``fflush`` is
performed during sampled ``process`` frames. If the run terminates abnormally before
``postprocess``, the deferred records are not written. The records are held in host memory
until the end of the run, so memory use grows with the number of attempts, defect changes,
and completed edge windows.

* ``proton_bias.out`` contains one line per sampled window. ``B_mean`` is the mean of
  :math:`|A_j|`, where :math:`A_j=(N_{j,+}-N_{j,-})/(N_{j,+}+N_{j,-})`; ``F_A_gt_0.2`` and
  ``F_A_gt_0.4`` are the fractions of active O-O pairs above the corresponding bias.
  ``mean_abs_DeltaF_over_kBT`` is the mean of :math:`|\ln(N_+/N_-)|` over pairs that sampled
  both states. Defect counts use nearest-O coordination relative to two hydrogens per oxygen.
  The final ``assignment_ambiguous_samples`` and ``pair_conflict_samples`` columns report
  strict-assignment samples skipped in that window.
* ``proton_attempt.out`` contains every attempt, including ``return``, ``geometry_lost``, and
  ``run_end``. ``time_start_fs`` is the first dead-band frame and ``time_end_fs`` is the
  confirmation, return, geometry-loss, or run-end frame. ``E_parallel_start`` and
  ``E_parallel_end`` are the nominal-ion electric field projected along ``O_low`` to
  ``O_high``; ``nearest_ion_id`` and ``nearest_ion_distance`` refer to the event-end frame.
  ``delta_phi_start`` and ``delta_phi_end`` are the nominal-ion potential differences
  :math:`\phi(O_{\rm high})-\phi(O_{\rm low})` at the attempt start and end. The final two
  columns are ``delta_d_<ion1>_start`` and ``delta_d_<ion2>_start``, where
  :math:`\Delta d_{\rm ion}=d(\mathrm{ion},O_{\rm high})-d(\mathrm{ion},O_{\rm low})`;
  the species names come from ``ion_field``. These start-frame quantities are kept separate
  from the event-end field because the local environment can change during a transfer attempt.
  If ``ion_field`` is omitted, these fields are written as ``nan`` and ``-1``. Frames rejected
  only because of strict-assignment ambiguity do not close an active attempt.
* ``proton_transfer.out`` contains sparse, hold-confirmed hydrogen transfer events with the
  attempt start time and the confirmation time. Its ``event_id`` is the corresponding successful
  attempt ID. ``dx dy dz`` is the minimum-image vector from ``O_from`` to ``O_to``. Atom indices
  are zero-based, matching the internal GPUMD atom ordering. The ``nH_*_before`` values are the
  event-local counts before the inferred transfer, and the ``nH_*_after`` values apply the
  ``O_from`` to ``O_to`` operation; ``q_defect=nH-2`` is reported for all four values.
* ``proton_defect.out`` writes the complete initial oxygen defect state once, then only oxygen
  records whose nearest-H count changes between sampled frames. ``cause_event_id`` is the
  successful transfer ID when that event touched the oxygen, ``-1`` for an unassigned count
  change, and ``0`` for the initial state. This sparse stream is intended to reconstruct
  time-ordered defect propagation chains, including open chains and branches; GPUMD does not
  impose a strict closed-ring criterion or assign chain IDs.
* ``proton_edge_window.out`` contains one record for every O-O edge observed so far in each
  completed window. ``geometry_occupancy`` is the number of valid O-H-O observations divided
  by the number of sampled frames; it is normally between zero and one when one hydrogen is
  associated with an edge. ``success_probability`` uses only successes and returns, as
  :math:`N_{\rm success}/(N_{\rm success}+N_{\rm return})`; geometry loss and run-end are
  reported separately. When ``ion_field`` is enabled, the appended ``mean_E_parallel``,
  ``std_E_parallel``, ``corr_delta_E_parallel``, ``mean_E_success``, and ``mean_E_return``
  columns summarize the nominal-ion field. The first two distance columns are the mean nearest
  distances to the two configured ion species measured from the O--O midpoint; their names are
  generated from the input symbols. The appended signed-population columns are
  ``log_population_ratio=ln(N_+/N_-)``,
  ``beta_DeltaF_high_minus_low=-ln(N_+/N_-)``, and its absolute value. Thus a positive
  ``beta_DeltaF_high_minus_low`` means that the ``O_high`` side has the higher inferred free
  energy. The next three columns are the mean, standard deviation, and correlation of
  ``delta_phi_ion`` with ``delta``. The remaining six columns give the mean ion1/ion2 distances
  to ``O_low`` and ``O_high`` and their high-minus-low differences; these nearest-endpoint
  distances scan the configured species, independently of the midpoint cutoff. The ion names
  are generated from the input symbols. All signed-potential and side-resolved quantities are ``nan`` when
  ``ion_field`` is disabled or the edge has no valid geometry samples.
* ``proton_local_environment_window.out`` is written only when ``local_environment`` is enabled.
  It contains one record per observed O-O edge and window. It reports mean O-H-O geometry,
  endpoint hydrogen counts, donor/acceptor edge counts, the first three ion distances and
  coordination numbers for each endpoint, endpoint distance differences, nearest ion2-to-H
  distance, continuous O-H...ion2 angles, ``hcl_like`` fractions, and the nominal field and
  potential-difference descriptors. Missing neighbors are excluded from the corresponding
  distance mean and are reported as ``nan``.
* ``proton_local_environment_event.out`` is written only when ``local_environment`` is enabled.
  Each finalized attempt has ``start``, ``end``, and ``last_valid`` snapshots. When
  ``bead_diagnostic`` is also enabled, valid ``centroid_best`` and ``delocalization_best``
  snapshots are appended, binding the local environment to the dynamical outcome and the
  independent quantum-geometry label.
* ``proton_bond.out`` contains accumulated per-pair geometry, state, and transition counts
  after the run.
* ``proton_bead_event.out`` is written only when ``bead_diagnostic`` is enabled. It contains
  two records for each finalized attempt, selected by ``selection_kind=centroid_best`` and
  ``selection_kind=delocalization_best``, with the dynamical ``outcome`` and independent
  ``quantum_class`` label. ``probe_time_fs`` is the time of the selected probe;
  ``f_minus``, ``f_zero``, and ``f_plus`` are the fractions of beads below, inside, and above
  the centroid dead band. ``sigma_delta``, ``delta_min``, ``delta_max``, and ``span`` describe
  the bead distribution. ``kink_count`` counts signed-well changes after center beads are
  skipped; ``center_domain_count`` counts cyclic contiguous center runs; and
  ``total_state_domain_count`` counts all cyclic domains of the three-state sequence.
  ``two_well_occupied``, ``two_well_span``, ``simple_two_domain_path``,
  ``barrier_centered``, ``strict_tunneling_like``, and ``multi_kink_or_multi_domain`` expose
  the individual strict filters as integer flags.
  ``channel_valid_count`` and ``f_channel_valid`` report how many beads satisfy the same
  fixed-pair O-H-O geometry filters; this is diagnostic only and is not a strict-label filter.
  ``rms_neighbor_delta_jump`` and ``max_neighbor_delta_jump`` quantify bead-to-bead coordinate
  jumps. ``quantum_class`` is ``two_well_delocalized``,
  ``barrier_centered_tunneling_like``, ``compact_single_domain``, or
  ``multi_kink_or_multi_domain`` when the corresponding criteria pass; other cases remain
  ``ambiguous``. For a multi-bead run where bead
  coordinates are unavailable, the record is ``ambiguous`` with invalid floating-point fields.
  The file is intended for RPMD/PIMD diagnostics and is absent when the optional diagnostic is
  not enabled; an explicitly enabled one-bead run is labeled ``classical_only``.

Chain-level labels such as ``recombined`` and ``pinned`` require temporal matching of the
transfer and defect streams and are therefore left to offline analysis. The observer's
``geometry_lost`` attempt outcome is the raw signal for a geometry-blocked propagation; it is
not by itself proof that a defect chain was interrupted.

The optional ``ion_field`` is only a nominal point-charge proxy. For each valid O-H-O edge it
uses the midpoint of the minimum-image O-O vector and evaluates

.. math::

   E_{\rm ion,parallel}=K_C\sum_i q_i
   \frac{(\mathbf r_m-\mathbf r_i)\cdot\hat{\mathbf e}_{OO}}
   {|\mathbf r_m-\mathbf r_i|^3},

including only configured ions within the cutoff. The sign is positive along ``O_low`` to
``O_high`` and the numerical unit is V/Å. This is not the full local electric field and does
not use qNEP dynamic charges; it is an observer-side mechanism proxy and does not alter forces,
PIMD/RPMD integration, qNEP, or HAC.

The same configured ions also provide the signed potential-difference proxy

.. math::

   \Delta\phi_{\rm ion}=\phi(O_{\rm high})-\phi(O_{\rm low})
   =K_C\sum_i q_i\left(\frac{1}{r_{i,\rm high}}-\frac{1}{r_{i,\rm low}}\right).

Only ions whose distance from the O--O midpoint is within ``ion_field``'s cutoff contribute,
so this quantity uses the same truncated nominal-ion environment as ``E_parallel``. It is a
signed mechanism proxy, not a qNEP electrostatic potential or a rigorous free-energy difference.
With ``A>0`` denoting a population preference for ``O_high``, the simple point-charge picture
predicts an anticorrelation between ``A`` and ``delta_phi_ion`` for a positive proton.

When ``bead_diagnostic`` is enabled, each active attempt is probed at its smallest observed
:math:`|\Delta_{\rm centroid}|`. The O--O pair is inherited from the centroid assignment and
is not reselected independently for each bead. For bead :math:`s`, the diagnostic evaluates
:math:`\Delta_s=d(H_s,O_{\rm low,s})-d(H_s,O_{\rm high,s})`, using the same minimum-image
distance convention as the centroid observer. Beads are classified with the existing
``delta_cutoff`` dead band. ``two_well_delocalized`` requires at least ``f_min`` of the beads in
each signed well, no more than ``center_max`` center beads, exactly two cyclic sign changes
after dead-band beads are skipped, and no more than four complete cyclic state domains. The
legacy ``two_well_span`` flag additionally requires a span of at least ``span_min``. The strict
path also rejects
multi-domain patterns such as ``LL00LLRR00RR``. A compact distribution dominated by one signed
well or the dead band with no cyclic sign change is labeled ``compact_single_domain``; a
multi-kink or multi-domain configuration is labeled ``multi_kink_or_multi_domain``. Other
multi-bead cases are ``ambiguous``. A one-bead run is always labeled ``classical_only``.

These labels identify tunneling-like ring-polymer geometry, not a rigorous quantum observable
or proof that a particular RPMD trajectory tunneled. They should be checked against bead-number
convergence, H/D isotope comparisons, and matched pure/salt-ice conditions.

The bead coordinates are packed on the GPU into one staging buffer and copied to the CPU with
one synchronous transfer per sampled frame that needs a bead probe. This preserves the
diagnostic result while avoiding one synchronization and device-to-host transfer per bead.
At the end of the run, the observer prints the sampled-frame count, bead-probe-frame count,
packed-copy count, copied bytes, pack-plus-copy wall time, bead-analysis wall time, geometry-kernel
wall time, compact geometry D2H/host-copy wall time, CPU state-machine wall time, and total observer
wall time. The copy timing
includes the packing kernel and the synchronous device-to-host copy; it does not include the force
or integrator paths. The geometry shell used by the GPU observer is built once from the initial
centroid structure; this fixed-topology optimization is intended for fixed-volume RPMD/NVE
trajectories. Strongly deforming NPT or structure-search runs require a shell-rebuild policy
before using this optimization.

The optional ``local_environment`` observer is a nominal structural/mechanism proxy. It uses
centroid positions and the same configured ion species as ``ion_field``. Its distances and
coordination numbers are continuous descriptors, not fixed chemical coordination assignments.
``hcl_like_low`` and ``hcl_like_high`` are thresholded summaries of the reported continuous
angles. This implementation does not use qNEP dynamic charges, BECs, or polarizabilities;
qNEP environment descriptors require a separate bead-batch charge interface and are intentionally
not enabled by this option.

Use ``compute_proton_tunneling`` first in a short validation run and compare its pair/state
assignment with the existing Python trajectory script. The transfer and defect streams are
intentionally kept simple so that defect-chain lifetime, branching, recombination, spatial
extent, and propagation failures can be reconstructed and refined offline without coupling
those choices to the force calculation.
