.. _kw_pimd_controls:

PIMD propagation and constraint controls
========================================

The following optional keywords apply to :attr:`pimd`, :attr:`rpmd`, and
:attr:`trpmd` runs. They must be placed in ``run.in`` before the corresponding
run is executed.


``pimd_propagator``
-------------------

Syntax::

    pimd_propagator exact|cayley

``exact`` uses the exact harmonic normal-mode rotation for the free
ring-polymer part of the Trotter-split propagator. It is the default and is
consistent with the default propagator in i-PI. ``cayley`` uses the previous
GPUMD Cayley approximation and can be selected for comparison or for runs
that deliberately use a larger time step.

The normal-mode coefficients are updated only when the temperature or time
step changes and are reused by the GPU kernels. Selecting ``exact`` therefore
does not perform trigonometric evaluation once per atom and mode.


``pimd_pile_scale``
-------------------

Syntax::

    pimd_pile_scale <scale>

The default is ``2.0``. GPUMD applies two Langevin half steps and uses
:math:`\exp[-\mathrm{scale}\,\Delta t\,\omega_k/2]` in each half step, so
``2.0`` gives the standard PILE-L friction :math:`\gamma_k=2\omega_k`, as in
i-PI with its default PILE scale. The historical GPUMD coefficient is
recovered with ``1.0``. Larger values damp internal normal modes more
strongly; smaller positive values damp them more weakly.

This parameter has no effect in a pure :attr:`rpmd` run because RPMD does not
apply the internal-mode Langevin thermostat.


``pimd_fix_com``
----------------

Syntax::

    pimd_fix_com on|off

The default is ``on``. It removes the three global center-of-mass momentum
components of the complete ring polymer, matching the physical meaning of a
center-of-mass constraint. It does not independently remove the center-of-mass
momentum of every bead. Set it to ``off`` when matching i-PI's default
``fixcom=False`` behavior. This option changes the sampled velocities and can
therefore affect a heat-current trajectory, but it does not change the
centroid heat-flux operator itself.

In the current implementation the COM correction is coupled to the Langevin
step, so this parameter has no effect in a pure :attr:`rpmd` run.


PIMD bead batching
------------------

Syntax::

    pimd_bead_batch on|off

This enables or disables the batched force kernels for PIMD, RPMD, and TRPMD
runs. The active potential is detected automatically: standard NEP models use
the NEP bead-batch path, qNEP models use the qNEP bead-batch path, and DP uses
the DP path when built with DeePMD support and loaded from a compact canonical
graph ``.pt2`` model. DP batching also requires the large-box minimum-image
condition: every periodic box thickness must exceed twice the sum of the DP
cutoff and skin distance. Otherwise the force calculation falls back to serial
bead evaluation.
The default is ``off``. The command must appear before the corresponding
``run``.

DP batch edge-fill experiments
------------------------------

These controls apply only to the batched DP canonical-graph path. Both
optimizations are default-off and can be tested independently.

Syntax::

    pimd_dp_batch_source_count off|check|on
    pimd_dp_batch_edge_fill_4_threads on|off

``pimd_dp_batch_source_count off`` keeps the existing per-edge atomic source
count. Use ``check`` first: GPUMD compares the atomic incoming-edge count with
``NN_local`` on every batch force call and stops if any node differs. This mode
copies the mismatch count back to the host each call, so use it only to check a
short trajectory that includes periodic crossings and neighbor-list reuse; do
not use its timing. After the check passes for the tested trajectory,
``pimd_dp_batch_source_count on`` removes the edge-fill atomics and uses
``NN_local`` as the source-count scan input.

``pimd_dp_batch_edge_fill_4_threads on`` assigns four threads to each atom's
batch edge-fill work; ``off`` retains one thread per atom. It does not change
single-frame DP calculations. Compare each option separately with the same
restart, and check energy, per-atom forces, and all nine virial components
before comparing timings with ``pimd_dp_batch_profile on``.

PIMD bead neighbor-list rebuilding
----------------------------------

Syntax::

    pimd_bead_neighbor_rebuild auto|always

``always`` is the default and rebuilds every bead's neighbor list on each force
call; use it for variable-box runs. ``auto`` reuses lists until atoms move more
than half the skin distance and is intended for fixed-box performance tests,
because box changes alone do not invalidate the cached list.

The older ``pimd_nep_bead_batch`` and ``pimd_qnep_bead_batch`` keywords remain
accepted as compatibility aliases, but new input files should use
``pimd_bead_batch``. The qNEP-only BEC control
``pimd_qnep_batch_bec`` remains separate.


Centroid heat flux
------------------

These controls do not alter the centroid heat-flux implementation selected by
``compute_hac ... use_centroid_heat_flux 1``. The NEP and qNEP force/virial
paths used to construct the centroid heat flux remain active.

For a fixed-box qNEP PIMD/RPMD/TRPMD run with ``pimd_bead_batch on``, the
centroid qNEP force and virial are evaluated in the same bead batch as one
auxiliary, non-integrated lane on HAC sampling force calls. The batch capacity
is allocated once for ``P+1`` lanes, but non-sampling force calls activate only
the ``P`` physical lanes, so the auxiliary centroid calculation is not
performed at every integration step. This optimization is enabled
automatically by ``compute_hac`` when its centroid flag is 1; physical bead
forces and the ring-polymer propagation are unchanged. Variable-box runs, SCR
barostats, unsupported potentials, and non-batched runs retain the serial
centroid-force fallback.

To remove centroid qNEP work from the dynamics loop, add the optional
``deferred_centroid_qnep`` flag to ``compute_hac``. With this flag set to 1,
fixed-box qNEP PIMD/RPMD/TRPMD runs cache centroid positions and velocities at
HAC samples and evaluate them later in fixed 32-frame qNEP batches. This mode
is default-off, supports ``split_qnep_heat_by_type 0`` only, and falls back to
the sampled auxiliary-lane or serial path when its requirements are not met.
