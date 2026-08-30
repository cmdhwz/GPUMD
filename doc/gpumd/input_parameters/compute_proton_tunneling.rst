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

  compute_proton_tunneling <sample_interval> <window_samples> <delta_cutoff> <hold_samples> <dOO_min> <dOO_max> <rperp_max> [O_symbol H_symbol]

For example::

  compute_proton_tunneling 5 1000 0.10 2 2.20 2.60 0.80 O H

The state variable for each hydrogen is

.. math::

   \Delta = d(\mathrm{H},\mathrm{O}_{\rm low}) - d(\mathrm{H},\mathrm{O}_{\rm high}),

where the two nearest oxygen atoms are ordered by atom index. A state is positive when
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

The O-O range and perpendicular-distance cutoff are only the first geometric filter. The
observer also requires an O-H-O angle above 120 degrees and both O-H distances below 1.60 Å;
these fixed checks reduce false pairs involving the other interpenetrating ice-VII network.
The numerical range should still be calibrated from the O-O RDF and joint O-O/perpendicular
distance distribution at the target pressure.

Output
------

The observer appends five files:

* ``proton_bias.out`` contains one line per sampled window. ``B_mean`` is the mean of
  :math:`|A_j|`, where :math:`A_j=(N_{j,+}-N_{j,-})/(N_{j,+}+N_{j,-})`; ``F_A_gt_0.2`` and
  ``F_A_gt_0.4`` are the fractions of active O-O pairs above the corresponding bias.
  ``mean_abs_DeltaF_over_kBT`` is the mean of :math:`|\ln(N_+/N_-)|` over pairs that sampled
  both states. Defect counts use nearest-O coordination relative to two hydrogens per oxygen.
* ``proton_attempt.out`` contains every attempt, including ``return``, ``geometry_lost``, and
  ``run_end``. ``time_start_fs`` is the first dead-band frame and ``time_end_fs`` is the
  confirmation, return, geometry-loss, or run-end frame.
* ``proton_transfer.out`` contains sparse, hold-confirmed hydrogen transfer events with the
  attempt start time and the confirmation time. Its ``event_id`` is the corresponding successful
  attempt ID. ``dx dy dz`` is the minimum-image vector from ``O_from`` to ``O_to``. Atom indices
  are zero-based, matching the internal GPUMD atom ordering.
* ``proton_edge_window.out`` contains one record for every O-O edge observed so far in each
  completed window. ``geometry_occupancy`` is the number of valid O-H-O observations divided
  by the number of sampled frames; it is normally between zero and one when one hydrogen is
  associated with an edge. ``success_probability`` uses only successes and returns, as
  :math:`N_{\rm success}/(N_{\rm success}+N_{\rm return})`; geometry loss and run-end are
  reported separately.
* ``proton_bond.out`` contains accumulated per-pair geometry, state, and transition counts
  after the run.

Use ``compute_proton_tunneling`` first in a short validation run and compare its pair/state
assignment with the existing Python trajectory script. The transfer stream is intentionally
kept simple so that defect-loop lifetime, radius, winding, and event-chain statistics can be
reconstructed and refined offline without coupling those choices to the force calculation.
