.. _restart_xyz:
.. index::
   single: restart.xyz (output file)

``restart.xyz``
===============

This is the restart file.
It is generated when invoking the :ref:`dump_restart keyword <kw_dump_restart>`.

File format
-----------
This file has the same format as the :ref:`simulation model input file <model_xyz>`.

* The output mode for this file is overwrite.
* By renaming (or copying) this file to ``model.xyz`` one can restart a simulation.

When :ref:`dump_restart <kw_dump_restart>` is given an optional backup directory, the root
``restart.xyz`` remains the latest restart file and each scheduled output is also written as a
separate ``restart_step_<step>.xyz`` file in that directory.
