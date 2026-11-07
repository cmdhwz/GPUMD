.. _kw_ensemble_pimd:

:attr:`ensemble` (PIMD)
=======================

The :attr:`ensemble` keyword is used to set up an integration method (an integrator).
The integrators described on this page enable one to carry out path integral
molecular dynamics (:term:`PIMD`) simulations and thereby to incorporate
quantum nuclear statistical effects. Standard PIMD samples equilibrium
statistical distributions; it does not represent real-time nuclear dynamics.
RPMD and TRPMD are approximate dynamical methods based on the ring-polymer
representation.


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
The pressure target is fixed during a run unless a pressure-ramp form below is used.
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

For either :attr:`pimd` or :attr:`pimd_scr`, the target pressure can be ramped linearly during the current run for isotropic or orthogonal-cell control::

    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> <P_start> <P_stop> <C> <tau_p>
    ensemble pimd <num_beads> <T_1> <T_2> <T_coup> <Pxx_start> <Pyy_start> <Pzz_start> <Pxx_stop> <Pyy_stop> <Pzz_stop> <Cxx> <Cyy> <Czz> <tau_p>

The :attr:`pimd_scr` keyword can be substituted for :attr:`pimd` in these forms. The first form controls isotropic pressure; the second controls the three orthogonal cell directions independently and requires an orthogonal box. The pressure and elastic-modulus values are in GPa. The target pressure interpolates from ``*_start`` to ``*_stop`` over the progress of the current :attr:`run`, using ``step / number_of_steps`` with zero-based ``step``. Thus the first barostat update uses the start pressure; for :math:`N` steps, the last update uses :math:`P_\mathrm{start} + (P_\mathrm{stop} - P_\mathrm{start})(N-1)/N`, and ``run 1`` uses only the start pressure. The stop pressure is the run-end boundary value. A newly declared ramp in the next run starts from that declaration's start pressure. Existing fixed-pressure forms remain available. Triclinic pressure ramps are not supported.

For example, the following ramps isotropic pressure from 0 to 50 GPa over 100000 steps::

    ensemble pimd 32 300 300 100 0 50 100 1000
    run 100000

This form ramps the three orthogonal pressure components independently::

    ensemble pimd 32 300 300 100 38 38 38 50 50 50 140 140 140 1000

PIMD runs through the GPUMD-WPE backend use the same pressure-ramp inputs and integrator. WPE supplies the bead forces and virials; the shared PIMD integrator evaluates pressure and applies the cell update.

For both pressure-controlled forms, the barostat transforms the centroid
position with the updated cell and translates every bead by the same centroid
displacement, preserving each bead's displacement from the centroid. In each
``compute1`` step, the current centroid-virial pressure is evaluated after the
first physical-force half kick, followed by the cell update and free
ring-polymer propagation. The regular force evaluation later in the step uses
the updated coordinates and cell. This is a finite-step operator splitting of
the existing weak-coupling or stochastic cell-rescaling scheme; it does not
imply strict unbiased NPT sampling or equivalence to every detail of i-PI's
barostat algorithms.

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
