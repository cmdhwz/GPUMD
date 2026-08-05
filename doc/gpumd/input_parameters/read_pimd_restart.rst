.. _kw_read_pimd_restart:

:attr:`read_pimd_restart`
=========================

Read the centroid and all bead positions and velocities from ``restart_beads.xyz``.
The keyword must follow a PIMD-related ``ensemble`` declaration with the same number of beads.

Syntax
------

::

    read_pimd_restart <filename>

Restart files written by the current ``dump_pimd_restart`` implementation also contain the
ring-polymer temperature. When that metadata is present it is restored automatically. Older
restart files without temperature metadata require either a preceding ``ensemble pimd``
declaration or an explicit temperature in ``ensemble rpmd``/``ensemble trpmd``.

For a PIMD-to-RPMD continuation, do not run a zero-time-step PIMD step just to set the
temperature. Use a parse-only setup followed by the RPMD declaration, for example::

    ensemble pimd 32 300 300 100
    read_pimd_restart restart_beads.xyz
    ensemble rpmd 32

or start directly from a self-describing restart file::

    ensemble rpmd 32 300
    read_pimd_restart restart_beads.xyz
