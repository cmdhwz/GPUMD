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
The pressure target is fixed during a run unless the pressure-ramp form below is used.
Note that :attr:`pimd` or :attr:`pimd_scr` (that is, not :attr:`rpmd` or :attr:`trpmd` described below) must be the first run that requires to set :attr:`num_beads`. By default one cannot change :attr:`num_beads` from run to run. A continuous PIMD sequence may explicitly discard the old internal modes and rebuild a new ring polymer from its current centroid with :attr:`pimd_reseed_from_centroid`; this exception is not available for RPMD or TRPMD.

One-time centroid reseeding
---------------------------

For a continuous PIMD sequence, first complete a PIMD run and then select a different
even bead count. Add the standalone keyword before the next :attr:`run`::

    ensemble pimd 8 300 300 100
    run 10000
    ensemble pimd 32 300 300 100
    pimd_reseed_from_centroid
    run 10000

The current centroid position and velocity are copied to every new bead. All old
ring-polymer internal modes, forces, potentials, and virials are discarded; the first
force evaluation of the new run recomputes the force data. The target bead count must
differ from the current count, and the command cannot be combined with
:attr:`read_pimd_restart` in either order. ``read_pimd_restart`` remains strict and
requires an exact bead-count match.

:attr:`pimd_scr`
^^^^^^^^^^^^^^^^^
If the first parameter is :attr:`pimd_scr`, it means that the current run will use path-integral molecular dynamics (:term:`PIMD`) with stochastic cell rescaling (:attr:`npt_scr`) for pressure control.

It can be used in the following ways::

    ensemble pimd_scr <num_beads> <T_1> <T_2> <T_coup> {<pressure_control_parameters>}

The pressure-control parameters have the same meaning and support the same isotropic, orthogonal, and triclinic forms as in the pressure-controlled :attr:`pimd` case.
The difference is that :attr:`pimd` uses the Berendsen barostat, whereas :attr:`pimd_scr` uses stochastic cell rescaling.


Pressure ramp
-------------

For either :attr:`pimd` or :attr:`pimd_scr`, the pressure target can be ramped linearly over the current run by supplying an initial pressure, a final pressure, the elastic modulus, and the pressure-coupling time::

    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> <P_start> <P_stop> <C> <tau_p>
    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> <Pxx_start> <Pyy_start> <Pzz_start> <Pxx_stop> <Pyy_stop> <Pzz_stop> <Cxx> <Cyy> <Czz> <tau_p>
    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> <Pxx_start> <Pyy_start> <Pzz_start> <Pyz_start> <Pxz_start> <Pxy_start> <Pxx_stop> <Pyy_stop> <Pzz_stop> <Pyz_stop> <Pxz_stop> <Pxy_stop> <Cxx> <Cyy> <Czz> <Cyz> <Cxz> <Cxy> <tau_p>

These three forms control isotropic, orthogonal, and triclinic cells, respectively.  Pressure and elastic-modulus values are in GPa; the pressure is interpolated from ``*_start`` to ``*_stop`` using the progress of the current :attr:`run`.  The fixed-pressure forms remain available by omitting the ``*_stop`` values.  A pressure ramp is intended for compression or equilibration; use a separate fixed-pressure run for stationary production data such as thermal conductivity.

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
