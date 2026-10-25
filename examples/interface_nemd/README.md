# Short interface NEMD example

`run.in` is a short, 100 Å example configuration for an AlAs/GaAs interface and its two pure-material references. Run it in three separate directories (`hetero`, `pure_alas`, `pure_gaas`) so each simulation writes its own `interface_temperature.out`. Use a 100 Å z cell and 20 × 20 Å transverse cell for the illustrative `analysis.json` values, or replace the interface, fit regions, binning, time window, and path fractions to match the actual structures. The analysis reads the area from each profile header and checks the optional configured `area_A2` against it.

Each `model.xyz` must contain a grouping method 0 with label 0 for the hot bath, 1 for the cold bath, and 2 for the rest of the atoms. The profile assigns material labels by coordinate: 1 below 50 Å, 2 above 50 Å. For the pure-material runs this is a virtual split used only to select the matched half-profile. Set `source_side` and the power fraction to match the heat-flow direction and periodic topology. The example uses `power_fraction=0.5` for two symmetric periodic paths.

This checkout contains no trained AlAs/GaAs NEP model or matching structures, so the sample is executable after those user-specific files are supplied; it is not a self-contained physical regression. Replace `SET_TO_GPUMD_COMMIT` with the revision recorded for each run. Use distinct run folders and remove or rename output files before repeating a run; the profile and analyzer intentionally reject existing outputs.

After the three profiles are written, run `python tools/interface_analysis.py examples/interface_nemd/analysis.json` from the repository root. It writes tables and an errorbar SVG into `examples/interface_nemd/analysis/`.
