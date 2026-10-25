.. _kw_compute_chunk:
.. index::
   single: compute_chunk (keyword in run.in)

:attr:`compute_chunk`
=====================

This keyword computes space- and time-averaged per-chunk quantities based on dynamic spatial binning.
Unlike the :ref:`compute keyword <kw_compute>`, which uses static group assignments defined in the model file, :attr:`compute_chunk` assigns atoms to spatial bins (chunks) based on their real-time coordinates at each sampling step.
This is particularly useful for systems where atoms diffuse across spatial regions during the simulation, such as low-density, porous, or amorphous materials in NEMD simulations.

The results are written to the ``compute_chunk.out`` output file.


Syntax
------

For 1D binning::

  compute_chunk <sample_interval> <output_interval> bin/1d <dim> <origin> <delta> {<quantity>}

For 2D binning::

  compute_chunk <sample_interval> <output_interval> bin/2d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2> {<quantity>}

For 3D binning::

  compute_chunk <sample_interval> <output_interval> bin/3d <dim1> <origin1> <delta1> <dim2> <origin2> <delta2> <dim3> <origin3> <delta3> {<quantity>}

For a dedicated 1D interface profile::

  compute_chunk <sample_interval> <output_interval> bin/1d <dim> <origin> <delta> interface <interface_coordinate> <lower_material_id> <upper_material_id> <remove_bin_drift> <output_file>

The parameters are defined as follows:

* :attr:`sample_interval`: Sampling interval in time steps. The quantities are sampled every this many steps.

* :attr:`output_interval`: Number of samples to average before producing one output. This is consistent with the :ref:`compute keyword <kw_compute>`: one output is produced every :math:`\text{sample\_interval} \times \text{output\_interval}` time steps.

* :attr:`bin/1d`, :attr:`bin/2d`, :attr:`bin/3d`: The binning style, specifying 1D, 2D, or 3D spatial binning respectively.

* :attr:`dim` (or :attr:`dim1`, :attr:`dim2`, :attr:`dim3`): The axis along which to bin. Must be :attr:`x`, :attr:`y`, or :attr:`z`. For :attr:`bin/2d` and :attr:`bin/3d`, the specified axes must be distinct.

* :attr:`origin` (or :attr:`origin1`, :attr:`origin2`, :attr:`origin3`): The bin origin. Currently only :attr:`lower` is supported, meaning bins start from the lower boundary of the simulation box (coordinate 0).

* :attr:`delta` (or :attr:`delta1`, :attr:`delta2`, :attr:`delta3`): The bin width in Ångströms. The number of bins along each axis is automatically calculated as :math:`\lceil L / \delta \rceil`, where :math:`L` is the box length along that axis. The last bin may be narrower than :math:`\delta` if :math:`L` is not an integer multiple of :math:`\delta`; its volume and center coordinate are computed using the actual width.

Starting after the binning parameters, one can list the quantities to be computed.
The allowed names for :attr:`quantity` are:

* :attr:`temperature`, which yields the temperature of each chunk
* :attr:`density/number`, which yields the number density (atoms per volume) of each chunk
* :attr:`density/mass`, which yields the mass density of each chunk
* :attr:`vx`, which yields the average velocity in the x direction
* :attr:`vy`, which yields the average velocity in the y direction
* :attr:`vz`, which yields the average velocity in the z direction
* :attr:`fx`, which yields the average force in the x direction
* :attr:`fy`, which yields the average force in the y direction
* :attr:`fz`, which yields the average force in the z direction

One can write one or more (distinct) names in any order.

The optional :attr:`interface` form is a separate output mode for block-averaged NEMD profiles.
It is limited to 1D binning and replaces the quantity list. Its parameters are:

* :attr:`interface_coordinate`: interface coordinate in Å, within the binning range.
* :attr:`lower_material_id`, :attr:`upper_material_id`: distinct nonnegative labels assigned to bins wholly below or above the interface. A bin crossing the interface receives material ID ``-1``.
* :attr:`remove_bin_drift`: ``yes`` subtracts each bin's mass-weighted center-of-mass velocity before calculating temperature and removes three degrees of freedom per populated bin; ``no`` uses all three translational components.
* :attr:`output_file`: a new output filename. Existing files are rejected to avoid mixing or overwriting runs.

This mode excludes atoms in the active ``fix`` and ``move`` groups from its counts, kinetic energy, potential energy, and temperature. The remaining per-bin degrees of freedom are :math:`3N` or :math:`3N-3`, depending on :attr:`remove_bin_drift`. Potential energy uses GPUMD's per-atom site-energy allocation. For :attr:`heat_lan`, :attr:`heat_nhc`, and :attr:`heat_bdp`, the source and sink cumulative energy counters are included as system-to-bath quantities; heating makes the source counter decrease and cooling makes the sink counter increase. Other ensembles write ``nan`` in those two columns and cannot be used by the bath-power analyzer.

The shared fixed-reference geometry commands ``interface_geometry`` and ``compute_interface_profile`` are documented separately in :ref:`interface_geometry <kw_interface_geometry>`. They do not change the dynamic binning or output of the legacy forms described here.

For triclinic (non-orthogonal) simulation boxes, the box length along each axis is computed as the geometric thickness :math:`L = V / A`, where :math:`V` is the box volume and :math:`A` is the cross-sectional area perpendicular to that axis. This is consistent with GPUMD's internal geometry calculations.


Example
-------

Example 1: 1D temperature profile along z
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  compute_chunk 10 100 bin/1d z lower 1.0 temperature

This means:

* sample the temperature every 10 time steps
* average over 100 samples before writing output (one output every :math:`10 \times 100 = 1000` steps)
* bin atoms along the z axis with 1.0 Å bin width, starting from the lower boundary
* compute the temperature of each bin

Example 2: 1D temperature and density profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  compute_chunk 10 100 bin/1d z lower 1.0 temperature density/number

Same as above, but also computes the number density of each bin.

Example 3: 2D velocity field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  compute_chunk 10 100 bin/2d x lower 2.0 z lower 2.0 vx vz

This creates a 2D grid of bins in the x-z plane with 2.0 Å bin width in each direction, and computes the average x and z velocity components in each bin.

Example 4: NEMD temperature profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A typical use case for NEMD simulations::

  ensemble heat_lan 300 300 50 0 1
  compute_chunk 10 100 bin/1d z lower 1.0 temperature density/number
  run 1000000

This computes the temperature and density profiles along z during a Langevin heat bath NEMD simulation.

Example 5: interface profile with block metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  ensemble heat_lan 300 100 50 0 1
  compute_chunk 100 10 bin/1d z lower 20.0 interface 5000.0 1 2 yes interface_temperature.out
  run 1000000

The input ``model.xyz`` must define source and sink labels 0 and 1, plus the remaining atoms (for example label 2), in grouping method 0; the output filename must not already exist. This records one 10-sample block every 1000 MD steps. Each row includes profile timing, site-energy sum, COM-corrected temperature, degrees of freedom, total sample count, valid-temperature sample count, and the bath-counter snapshot from the last sample in that block. Per-sample bin reductions and corrected temperature accumulation run on the GPU; complete and final partial blocks are cached and written together from ``post_run``. The Python resistance analyzer has not been updated for the new version 3 shared profile; see :ref:`interface_geometry <kw_interface_geometry>` for that format.


.. _compute_chunk_out:

Output file
-----------

The results are written to ``compute_chunk.out``.
The file is appended to if it already exists.
For each output time, there are :math:`N_\text{chunk}` consecutive lines (one per chunk), with the following format per line::

  <chunk_id> <coord1> [<coord2>] [<coord3>] <avg_count> <value1> [<value2>] ...

The columns are:

* :attr:`chunk_id`: the zero-based index of the chunk (dimensionless integer).
* :attr:`coord1` (and :attr:`coord2`, :attr:`coord3` for 2D/3D binning): the center coordinates (in units of Å) of the chunk along the corresponding binning axis.
* :attr:`avg_count`: the time-averaged number of atoms in the chunk (dimensionless).
* :attr:`value1`, :attr:`value2`, ...: the time-averaged values of the requested quantities, in the same order as specified in the command. The units for each quantity are:

  - :attr:`temperature`: K
  - :attr:`density/number`: Å\ :sup:`-3` (number of atoms per volume)
  - :attr:`density/mass`: amu/Å\ :sup:`3` (total atomic mass per volume)
  - :attr:`vx`, :attr:`vy`, :attr:`vz`: Å/fs
  - :attr:`fx`, :attr:`fy`, :attr:`fz`: eV/Å

The temperature is computed from the kinetic energy of the atoms in each chunk as :math:`T = 2 E_\text{k} / (3 k_\text{B} N)`, where :math:`E_\text{k}` is the total kinetic energy, :math:`N` is the atom count, and :math:`k_\text{B}` is the Boltzmann constant.
The velocity and force values are per-atom averages within each chunk.

For example, the command::

  compute_chunk 10 100 bin/1d z lower 2.0 temperature density/number

produces output lines of the form::

  0 1.000000 52.0 3.0012345678e+02 2.5000000000e-02
  1 3.000000 48.0 2.9876543210e+02 2.4000000000e-02
  ...

Here, each line contains: chunk index, bin center along z (Å), average atom count, temperature (K), and number density (Å\ :sup:`-3`).
The output blocks for successive time points are written consecutively with no blank line separator; each block contains exactly :math:`N_\text{chunk}` lines.

For the :attr:`interface` form, the selected :attr:`output_file` instead starts with metadata and a column header. Version 2 has one row per bin and these 17 columns::

  step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A coordinate_center_A material_id atom_count kinetic_energy_eV potential_energy_eV temperature_K dof samples temperature_samples bath_source_eV bath_sink_eV

The atom count, kinetic and potential energies, and degrees of freedom average every sample. Temperature averages only samples with positive degrees of freedom; ``temperature_samples`` records that count, and ``temperature_K`` is ``nan`` when no sample was valid. Bath energies are cumulative counters captured at each sample; the row contains the counters from the sample at ``time_ps``, including a final partial block. Other ensembles write ``nan`` for both counters. ``time_ps`` is calculated from GPUMD's internal time using its natural-time to fs conversion. Interface profiles reject non-orthogonal boxes and stop if the box matrix or PBC changes during the run. Profile rows are buffered until ``post_run``; an interruption before then can leave only the header. The legacy ``compute_chunk.out`` command, format, and temperature convention are unchanged.


Caveats
-------

* The :attr:`origin` parameter currently only accepts :attr:`lower`. Numeric origin values are not supported.
* This keyword addresses the measurement/analysis side of dynamic spatial binning. The NEMD heat baths (:attr:`ensemble heat_*` source/sink regions) still use static group assignments. Dynamic heat bath regions would require separate modifications to the integrator.
* Multiple :attr:`compute_chunk` commands can be used in the same run.
