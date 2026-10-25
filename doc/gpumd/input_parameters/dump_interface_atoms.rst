.. _kw_dump_interface_atoms:

.. index::
   single: dump_interface_atoms (keyword in run.in)

``dump_interface_atoms``
========================

This action writes position, velocity, force, and site potential snapshots for atoms in a selected fixed-reference geometry window. It requires an earlier :ref:`interface_geometry <kw_interface_geometry>` declaration and supports classical MD only.

Syntax
------

::

   dump_interface_atoms <sample_steps> <samples_per_block> <bin_lo> <bin_hi> <prefix>

The zero-based bin range is half-open, ``[bin_lo, bin_hi)``. ``sample_steps`` is the MD-step sampling interval and ``samples_per_block`` controls the GPU staging block; it does not affect the output cadence. Multiple actions are allowed in one run for different windows or output prefixes.

Every atom mapped into the selected reference bins is included, including atoms marked as bath, fixed, or moving. Atoms are written in global atom-ID order. Atom IDs join directly to the zero-based rows in the geometry's ``<output_prefix>_groups.dat`` file. An empty window or a run with no scheduled sample creates a header-only file.

Output
------

The action creates ``<prefix>_atoms.out`` and refuses an existing file. Its 17 columns are:

::

   frame step time_ps atom_id x_A y_A z_A xu_A yu_A zu_A vx_A_ps vy_A_ps vz_A_ps fx_eV_A fy_eV_A fz_eV_A potential_eV

``frame`` is zero-based; ``step`` is the one-based completed MD step; ``atom_id`` is the zero-based full-system atom index. Positions are in A, velocities in A/ps, forces in eV/A, and site potentials in eV. Values are printed with 17 significant digits. The velocity is sampled after ``compute2`` on that MD step. Force is the current ``atom.force_per_atom`` value, including contributions already added by actions; it is not a pure-NEP force residual. Unwrapped coordinates are reconstructed for selected atoms from wrapped position plus the periodic image count times the fixed orthogonal box length. The action allocates image counters but does not enable the standard all-atom unwrapped-position update.

The action gathers only selected atom data on the GPU. Full and final partial blocks are retained and formatted together in ``post_run``. If a sampled value is non-finite, the sticky error records the first failing step and atom; the block is rejected before it enters the CPU cache. A run interrupted before ``post_run`` can leave the initialized file with only its header.

The GPU staging memory scales with ``min(samples_per_block, scheduled_frames) * selected_atoms``. The final CPU cache scales with the number of scheduled frames times the selected atom count. This output is intended for short local windows; it does not store an inexpensive full-system trajectory. Periodic image counts remain on ``Atom`` after a run; if a later run enables the standard all-atom unwrapped-position path, it continues from those counts.
