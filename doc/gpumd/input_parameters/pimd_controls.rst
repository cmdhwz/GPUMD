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
consistent with i-PI's default propagator. The full integration still has
finite-step splitting error, and ``exact`` is not universally more accurate
than ``cayley``; check time-step convergence for the property of interest.
``cayley`` uses the Cayley approximation and can be selected for comparison.

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
strongly; smaller positive values damp them more weakly. The value must be
finite and greater than zero.

This parameter has no effect in a pure :attr:`rpmd` run because RPMD does not
apply the internal-mode Langevin thermostat.


``pimd_fix_com``
----------------

Syntax::

    pimd_fix_com on|off

The default is ``on``. It removes the three global center-of-mass momentum
components of the complete ring polymer by subtracting the same mass-weighted
center-of-mass velocity from every bead. Internal bead velocities are
unchanged; the center-of-mass momentum of each bead is not constrained
separately. The correction is applied after initialization and after each
integration velocity update, also in a pure :attr:`rpmd` run. This matches the
default ``fixcom=True`` in i-PI's input configuration. Set it to ``off`` to
disable the correction. It changes the sampled velocities and can therefore
affect a heat-current trajectory, but it does not change the centroid
heat-flux operator itself. The projected bead-mean velocity is also written
to the centroid velocity buffer.

The ``correct_velocity`` directive is rejected for PIMD, RPMD, and TRPMD. It
modifies only the centroid velocity buffer, which is overwritten by the bead
average and therefore cannot correct the ring-polymer bead velocities. Use
``pimd_fix_com on`` for global center-of-mass removal; group and angular
velocity corrections of bead modes are unsupported.


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
call; it is the conservative choice for variable-box runs. ``auto`` reuses
lists until atoms move more than half the skin distance. The NEP and qNEP
bead-batch paths also rebuild when the cell matrix or PBC state changes; this
box-change handling is specific to those batch paths and is not guaranteed for
other force backends.

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
physical bead batch remains at ``P`` lanes. With the centroid flag set to 1,
``compute_hac`` evaluates the centroid qNEP potential, force, and complete
per-atom virial separately at each HAC sample. It does not automatically add
a ``P+1`` centroid probe lane. Physical bead forces and ring-polymer
propagation are unchanged.

To remove centroid qNEP work from the dynamics loop, add the optional
``deferred_centroid_qnep`` flag to ``compute_hac``. With this flag set to 1,
fixed-box qNEP PIMD/RPMD/TRPMD runs cache centroid positions and velocities at
HAC samples and evaluate them later in fixed 32-frame qNEP batches. This mode
is default-off, supports ``split_qnep_heat_by_type 0`` only, and falls back to
the normal sampled centroid path when its requirements are not met.
