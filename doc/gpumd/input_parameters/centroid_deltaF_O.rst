.. _kw_centroid_deltaF_O:
.. index::
   single: centroid_deltaF_O (keyword in run.in)

:attr:`centroid_deltaF_O`
==========================

This keyword records the per-oxygen-atom difference between the physical force
averaged over the ring-polymer beads and the physical force evaluated on the
centroid configuration. It is a diagnostic only and does not modify RPMD/PIMD
dynamics, HAC, qNEP, or any force buffer used by the simulation.

Syntax
------

::

  centroid_deltaF_O <sample_interval>

It must be used together with :attr:`centroid_force_diagnostic`, and both
sample intervals must be identical. The command reuses the immediate centroid
force evaluation from :attr:`compute_hac`, so the same fixed-cell
PIMD/RPMD/TRPMD and immediate-centroid-HAC restrictions apply. A GPUMD build
with NetCDF support is required.

Output
------

After the run completes, GPUMD writes ``centroid_deltaF_O.nc`` as a NetCDF4/
HDF5 file. The file is created in ``postprocess`` and is not written during
dynamics. An existing file with the same name is rejected to avoid mixing
different runs.

The dimensions are ``frame`` (unlimited), ``oxygen``, ``xyz`` (3), and
``cell`` (3). The variables are:

``time_fs[frame]``
  Float64 time in fs.
``step[frame]``
  Int64 MD step.
``box_matrix[cell, xyz]``
  Float64 box matrix in Angstrom, with rows ``a``, ``b``, and ``c``.
``O_atom_id[oxygen]``
  Int32 0-based original GPUMD atom indices in fixed oxygen order.
``O_reference_position[oxygen, xyz]``
  Float64 centroid positions from the first diagnostic frame, in Angstrom.
``deltaF_O[frame, oxygen, xyz]``
  Float32 ``Fbar - Fc`` values in eV/Angstrom.

No bead forces, spring forces, hydrogen/sodium/chlorine data, or centroid
position time series are written.

Example
-------

::

  ensemble rpmd 32 300
  time_step 0.5
  compute_hac 5 1000 1 1 0
  centroid_force_diagnostic 5
  centroid_deltaF_O 5
  run 5000
