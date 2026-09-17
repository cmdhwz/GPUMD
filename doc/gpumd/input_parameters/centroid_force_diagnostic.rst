.. _kw_centroid_force_diagnostic:
.. index::
   single: centroid_force_diagnostic (keyword in run.in)

:attr:`centroid_force_diagnostic`
=================================

This keyword records the mismatch between the physical force averaged over all
ring-polymer beads and the physical force evaluated on the centroid
configuration. It is an observer only: it does not change RPMD/PIMD dynamics,
the existing heat current, or any heat-current correction.

Syntax
------

::

  centroid_force_diagnostic <sample_interval>

The ``sample_interval`` must be the same as the ``compute_hac`` sampling
interval. The diagnostic requires ``compute_hac`` with its fourth parameter
``use_centroid_heat_flux`` set to 1, and it reuses that action's immediate
centroid force evaluation. It is not available with deferred centroid HAC or
without centroid HAC. It also requires a fixed-cell ensemble without HNEMD or
HNEMDEC driving. It cannot be combined with ``compute_es``, ``active``,
``dump_observer``, or ``plumed`` because those actions may rewrite the force
buffer after the bead-force average has been produced.

Output
------

The data are held in memory during the run and appended to
``centroid_force_diagnostic.out`` after the run completes. No diagnostic rows
are written during dynamics. The first line of each run segment is a column
header with units. Force values are in
``eV/Angstrom``, time is in ``fs``, centroid energy is in ``eV``, and
``P_delta``, ``P_Fbar``, and ``P_Fc`` are in ``eV/fs``. The internal centroid
velocity used in the power calculation is in ``Angstrom/natural_time``.

The cache payload is approximately
``(55 * sizeof(double) + sizeof(double) + sizeof(int)) *
floor(number_of_steps / sample_interval)`` bytes, or about 452 bytes per
sampled frame, excluding allocator overhead. The estimate is printed before
dynamics. If a run terminates before postprocessing completes, cached
diagnostic samples from that run are not written and cannot be recovered.

The initial global force and power columns end with ``P_delta``: they are
``deltaF_rms``, ``deltaF_mean_abs``, ``deltaF_max``, ``Fbar_rms``, ``Fc_rms``,
``relative_deltaF_rms``, and ``P_delta``. Fixed H, O, Na, and Cl column groups
follow these columns. Each group contains ``N_type``, the force statistics,
and its own ``P_delta``; the groups do not contain ``P_Fbar`` or ``P_Fc``.
A missing species has ``N_type = 0`` and ``nan`` for its floating-point
statistics. The final five columns are ``K_centroid``, ``U_centroid``,
``E_centroid``, global ``P_Fbar``, and global ``P_Fc``.

Example
-------

::

  ensemble rpmd 32 300
  time_step 0.5
  compute_hac 5 1000 1 1 0
  centroid_force_diagnostic 5
  run 5000

Here the fourth ``compute_hac`` parameter enables immediate centroid HAC, and
both observers sample every five MD steps. ``P_delta`` is only an energy-power
consistency diagnostic; it is not added to HAC.
