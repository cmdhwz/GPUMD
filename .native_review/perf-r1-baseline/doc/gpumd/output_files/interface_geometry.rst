.. _interface_geometry_outputs:

Shared interface geometry, profile, VACF, and VDOS outputs
=========================================================

The :ref:`interface_geometry <kw_interface_geometry>` command writes the following prefix-based reference files and refuses to overwrite an existing path:

``<output_prefix>_geometry.json``
  Versioned algorithm, orthogonal box/PBC, area, bin edges, interfaces, group methods/IDs, mapping rule, and unit metadata.

``<output_prefix>_groups.dat``
  One row per atom, including zero-based atom ID, fixed bin/material/bath assignment, fixed/moving flags, wrapped reference coordinates, mass, and type. Bath membership does not remove the material label.

``<output_prefix>_bins.dat``
  One row per reference bin with geometric edges/volume, atom count, material ID, source/sink bath bit mask, and interface-crossing flag.

The output filename from ``compute_interface_profile`` contains one row per bin per output block. Its shared fixed-mapping format is version 3 with 19 columns, including ``temperature_samples``, ``bath_mask``, and ``interface_crossing``. The legacy ``compute_chunk ... interface ...`` output remains version 2 with 17 columns. See :ref:`the input description <kw_interface_geometry>` for column names and averaging rules.

For each complete block, ``compute_interface_dos`` writes ``<prefix>_vacf.out`` and ``<prefix>_vdos.out``. VACF rows include an origin count, raw correlation, and lag-zero-normalized value for every lag. The one-sided cosine spectrum retains raw ``C0``, the analytic integral, and a trapezoid integral of the actual exported frequency rows. ``frequency_grid_underresolved`` reports whether ``dOmega*tau_max > pi``; a range below Nyquist is marked ``band_limited``. Incomplete tail blocks are omitted.

``compute_interface_shc`` writes ``<prefix>_plane_flux.out``, ``<prefix>_shc_correlation.out``, and version 3 ``<prefix>_shc.out``. The first contains one row per sample and configured plane, the second contains separate positive and negative correlation lags with their origins, and the third contains the paired-block cumulative mean spectrum, standard error, analytic integral, and trapezoid integral of the exported rows. Its ``block_*`` columns describe only the current block; ``accumulation_*`` columns give the full time/sample range represented by the cumulative mean. It rejects underresolved frequency grids. The regular NEP large-box path and restrictions are documented under :ref:`compute_interface_shc <kw_compute_interface_shc>`.

The SHC result rows are written after the run. If the process stops before ``post_run``, the output files may contain only their headers. Flux rows from a final incomplete sample block are retained; that tail does not contribute to the SHC spectrum.
