.. _interface_geometry_outputs:

Shared interface geometry, profile, VACF, and VDOS outputs
=========================================================

The :ref:`interface_geometry <kw_interface_geometry>` command writes the following prefix-based reference files and refuses to overwrite an existing path:

``<output_prefix>_geometry.json``
  Version 2 metadata for the orthogonal box/PBC, area, bin edges, physical ``interfaces_A``, independent ``sampling_planes_A``, group methods/IDs, mapping rule, and units. SHC ``plane_id`` is the zero-based sorted-coordinate index into ``sampling_planes_A``.

``<output_prefix>_groups.dat``
  One row per atom, including zero-based atom ID, fixed bin/material/bath assignment, fixed/moving flags, wrapped reference coordinates, mass, and type. The first comment identifies these coordinates as geometry-mapping-only and says an equilibrium reference is external. Bath membership does not remove the material label.

``<output_prefix>_bins.dat``
  One row per reference bin with geometric edges/volume, atom count, material ID, source/sink bath bit mask, and physical-interface crossing flag. Sampling planes do not change this label.

The output filename from ``compute_interface_profile`` contains one row per bin per output block. Its shared fixed-mapping format is version 3 with 19 columns, including ``temperature_samples``, ``bath_mask``, and ``interface_crossing``. The legacy ``compute_chunk ... interface ...`` output remains version 2 with 17 columns. Per-bin temperature correction and block accumulation run on the GPU; each block's final bin sums are copied once into a run-sized host cache. Rows are written together during ``post_run``, including an incomplete final block. An interrupted process can therefore leave the profile file with only its header. See :ref:`the input description <kw_interface_geometry>` for column names and averaging rules.

For each complete block, ``compute_interface_dos`` caches GPU-normalized raw VACF and VDOS rows in preallocated CPU arrays, then writes ``<prefix>_vacf.out`` and ``<prefix>_vdos.out`` together during ``post_run``. VACF rows include an origin count, raw correlation, and lag-zero-normalized value for every lag. The one-sided cosine spectrum retains raw ``C0``, the analytic integral, and a trapezoid integral of the actual exported frequency rows. ``frequency_grid_underresolved`` reports whether ``dOmega*tau_max > pi``; a range below Nyquist is marked ``band_limited``. Incomplete tail blocks are omitted. The CPU cache grows with the number of complete blocks; an interruption before ``post_run`` can leave both files with only their headers. Non-finite data in a populated bin is rejected at its block boundary, while an empty bin remains ``nan``.

``compute_interface_shc`` writes ``<prefix>_plane_flux.out``, ``<prefix>_shc_correlation.out``, version 3 ``<prefix>_shc.out``, and ``<prefix>_shc_blocks.out``. The first contains one row per sample and configured plane, the second contains separate positive and negative correlation lags with their origins, the third contains the paired-block cumulative mean spectrum and errors, and the new fourth file contains the spectrum, ``C0``, analytic/trapezoid integrals, and mean flux for each individual complete block. Its ``block_*`` columns describe only the current block; ``accumulation_*`` columns in ``_shc.out`` give the full time/sample range represented by the cumulative mean. It rejects underresolved frequency grids. The regular NEP large-box path and restrictions are documented under :ref:`compute_interface_shc <kw_compute_interface_shc>`.

The SHC result rows are written after the run. If the process stops before ``post_run``, the output files may contain only their headers. Flux rows from a final incomplete sample block are retained; that tail does not contribute to the SHC spectrum.

``dump_interface_atoms`` writes ``<prefix>_atoms.out`` after ``post_run`` with 17 data columns. Its header identifies the geometry source, the mapping-only role of reference coordinates, and the lack of a t=0 row; force values are the current per-atom values and include enabled action contributions. The initial structure and any equilibrium ``R0`` must be retained separately.
