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

  compute_proton_tunneling <sample_interval> <window_samples> <delta_cutoff> <hold_samples> <dOO_min> <dOO_max> <rperp_max> [O_symbol H_symbol] [oho_angle angle_deg] [ion_field ion1_symbol ion1_charge ion2_symbol ion2_charge cutoff]

For example::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H

To require a more linear O-H-O geometry, add the optional angle setting::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H oho_angle 150

With a nominal Na/Cl ion-field proxy enabled, use one physical input line::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.65 0.80 O H ion_field Na 1.0 Cl -1.0 8.0

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

The observer appends six files:

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
  columns summarize the nominal-ion field. The final two distance columns are the mean nearest
  distances to the two configured ion species; their names are generated from the input symbols.
* ``proton_bond.out`` contains accumulated per-pair geometry, state, and transition counts
  after the run.

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

Use ``compute_proton_tunneling`` first in a short validation run and compare its pair/state
assignment with the existing Python trajectory script. The transfer and defect streams are
intentionally kept simple so that defect-chain lifetime, branching, recombination, spatial
extent, and propagation failures can be reconstructed and refined offline without coupling
those choices to the force calculation.
