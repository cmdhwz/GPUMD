.. _kw_md_nep_timing:

md_nep_timing
==============

Syntax
------

  md_nep_timing on|off

Default
-------

  off

Description
-----------

Enable GPU-event timing for the main single-GPU NEP and qNEP force stages.
Timing is disabled by default. When enabled, each measured stage is followed
by a synchronization, so this command is intended for profiling and should
not be used for production performance measurements.

Reports are appended to ``md_nep_timing.out`` every 100 force evaluations and
when the potential is destroyed. The report contains the number of samples,
the accumulated time in milliseconds, and the average time for each stage.

This command is independent of :ref:`md_nep_fine_parallel <kw_md_nep_fine_parallel>`.
It can be placed before or after ``potential``;
newly parsed potentials inherit the current setting. Non-NEP potentials and
multi-GPU potentials ignore the command.
