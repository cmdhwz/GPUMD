.. _kw_pppm_mesh_spacing:
.. index::
   single: pppm_mesh_spacing (keyword in run.in)

:attr:`pppm_mesh_spacing`
=========================

This keyword sets the target PPPM mesh spacing in Angstrom. The default is
``1.0``. The mesh size in each box direction is still rounded up using the
existing internal FFT-friendly rule.

Syntax
------

This keyword is used as follows::

  pppm_mesh_spacing <spacing>

where ``<spacing>`` must be greater than zero.

Example
-------

To request a target mesh spacing of 0.5 Angstrom use::

  pppm_mesh_spacing 0.5

This keyword changes only the mesh density. The PPPM FFT, charge deposition,
field interpolation, and virial calculations are otherwise unchanged.
