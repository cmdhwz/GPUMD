.. _kw_dump_restart:
.. index::
   single: dump_restart (keyword in run.in)

:attr:`dump_restart`
====================

Write a restart file.

Syntax
------

The first parameter is the output interval (number of steps) of updating the restart file. An
optional single directory name saves a versioned backup at every output while keeping the root
``restart.xyz`` as the latest restart file::

  dump_restart <interval> [backup_directory]

Example
-------

To update the restart file every 100000 steps for a run, one can add::

  dump_restart 100000

To also keep versioned restart files in ``restart_backups``::

  dump_restart 100000 restart_backups

The backup written after step 100000 is named
``restart_backups/restart_step_0000100000.xyz``. The directory name must be a single directory
name without path separators.

before the :ref:`run keyword <kw_run>`.


Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.
