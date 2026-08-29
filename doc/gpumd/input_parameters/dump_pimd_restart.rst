.. _kw_dump_pimd_restart:

:attr:`dump_pimd_restart`
=========================

Write the centroid and all bead positions and velocities to a single restart file.

Syntax
------

::

    dump_pimd_restart <interval> [backup_directory]

The output file is ``restart_beads.xyz``. The current format stores the bead count, box,
species, mass, positions, velocities, and ring-polymer temperature. Positions and velocities
are written with double-precision decimal output. The keyword is valid only for PIMD-related
runs.

If the optional ``backup_directory`` parameter is given, each output is also saved as a separate
file under that directory. For example, ``dump_pimd_restart 100000 backup_test`` saves a restart
written after step 100000 as ``backup_test/restart_beads_step_0000100000.xyz``. The root-level
``restart_beads.xyz`` remains the latest restart file. The directory name must be a single
directory name without path separators.
