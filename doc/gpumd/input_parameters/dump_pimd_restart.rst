.. _kw_dump_pimd_restart:

:attr:`dump_pimd_restart`
=========================

Write the centroid and all bead positions and velocities to a single restart file.

Syntax
------

::

    dump_pimd_restart <interval>

The output file is ``restart_beads.xyz``. The current format stores the bead count, box,
species, mass, positions, velocities, and ring-polymer temperature. Positions and velocities
are written with double-precision decimal output. The keyword is valid only for PIMD-related
runs.
