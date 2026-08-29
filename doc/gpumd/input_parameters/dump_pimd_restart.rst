.. _kw_dump_pimd_restart:

:attr:`dump_pimd_restart`
=========================

Write the centroid and all bead positions and velocities to a single restart file.

Syntax
------

::

    dump_pimd_restart <interval> [backup]

The output file is ``restart_beads.xyz``. The current format stores the bead count, box,
species, mass, positions, velocities, and ring-polymer temperature. Positions and velocities
are written with double-precision decimal output. The keyword is valid only for PIMD-related
runs.

If the optional ``backup`` parameter is given, each output is also saved as a separate file
under ``restart_backups``. For example, a restart written after step 10000 is saved as
``restart_backups/restart_beads_step_0000010000.xyz``. The root-level
``restart_beads.xyz`` remains the latest restart file.
