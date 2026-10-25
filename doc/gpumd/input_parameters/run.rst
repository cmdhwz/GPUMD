.. _kw_run:
.. index::
   single: run (keyword in run.in)

:attr:`run`
===========

Run a number of :term:`MD` steps according to the settings specified for the current run.

Syntax
------
This keyword only requires a single parameter, which is the number of steps to be run::

  run <number_of_steps>

Example
-------

To run one million steps, just write::

  run 1000000

Caveats
-------
* The number of steps should be compatible with some parameters in some other keywords. 
* We can regard a run as a block of commands from the :ref:`ensemble keyword <kw_ensemble>` to :attr:`run`::
    
    # the first run
    ensemble ...
    # other commands
    run ...

    # the second run
    ensemble ...
    # other commands
    run ...

    # the third run
    ensemble ...
    # other commands
    run ...

At run completion, ``Speed of this run`` reports the existing MD-loop atom-step rate. ``Post-run time`` reports the time spent in output/statistics actions after the loop, and ``Total speed including post_run`` uses the timed MD loop plus that post-run work. These additional lines expose deferred output cost without changing the MD-loop speed definition.
