.. _kw_ensemble_pimd:

:attr:`ensemble` (PIMD)
=======================

The :attr:`ensemble` keyword is used to set up an integration method (an integrator).
The integrators described on this page enable one to carry out path integral molecular dynamics (:term:`PIMD`) simulations and thereby to incorporate quantum dynamical effects.


Syntax
------

:attr:`pimd`
^^^^^^^^^^^^
If the first parameter is :attr:`pimd`, it means that the current run will use path-integral molecular dynamics (:term:`PIMD`).

It can be used in the following ways::

    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> 
    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> {<pressure_control_parameters>}

In both cases, :attr:`num_beads` is the number of beads in the ring polymer, which should be a positive even integer no larger than 128.
The first case is similar to the NVT ensemble with :attr:`nvt_lan` as the Langevin thermostat is used for both the internal and the centroid modes [Ceriotti2010]_. 
The second case is similar to the NPT ensemble with :attr:`npt_ber`, where a Berendsen barostat is added compared to the first case.
Note that :attr:`pimd` (that is, not :attr:`rpmd` or :attr:`trpmd` described below) must be the first run that requires to set :attr:`num_beads` and one cannot change :attr:`num_beads` from run to run.

:attr:`rpmd`
^^^^^^^^^^^^
If the first parameter is :attr:`rpmd`, it means that the current run will use ring-polymer molecular dynamics (:term:`RPMD`) [Craig2004]_.

It can be used as follows::

    ensemble rpmd <num_beads> {<temperature>}

This can be understood as the NVE version of :term:`PIMD`, where no thermostat is applied.
The optional temperature is used to set the ring-polymer frequency when RPMD is started in a
fresh run. If it is omitted, a temperature must have been supplied by a preceding PIMD
declaration or restored from :attr:`read_pimd_restart`.

:attr:`trpmd`
^^^^^^^^^^^^^
If the first parameter is :attr:`trpmd`, it means that the current run will use thermostatted ring-polymer molecular dynamics (:term:`TRPMD`) [Rossi2014]_.

It can be used as follows::

    ensemble trpmd <num_beads> {<temperature>}

This is similar to :term:`RPMD`, but the Langevin thermosat is applied to the internal modes.
The optional temperature follows the same rule as for :attr:`rpmd`.
