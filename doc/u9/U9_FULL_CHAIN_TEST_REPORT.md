# Final evidence matrix

| Gate | Final state | Evidence and boundary |
|---|---|---|
| Static source and call-chain review | COMPLETE | Parent reviewed the implemented diagnostic and finite-model interfaces and performed the final read and tracked `git diff --check`; this is source-only acceptance. |
| Compilation and software runtime | NOT RUN | User directed static-only work; native parser, CUDA, WPE, I/O, GPU parity, restart, multiple runs, append semantics and matching old-output regression were not executed on final source. |
| Mathematical numerical validation | INCONCLUSIVE | Test source covers derivatives, metric, reference mapping, spectra and RPMD; current revision tests were not run. |
| Quantum validation | NOT VALIDATED | Independent spectrum and sampling convergence, uncertainty and mapping acceptance remain unverified. |
| Material validation | NOT VALIDATED | Real material parameters and same-source Cartesian transport remain missing. |
| Candidate | BLOCKED | Candidate mode rejects activation; finite scalar diagnostics do not define a bulk current. |

## Historical evidence — earlier source only

Earlier snapshots produced ten external OHO test passes, three HAC static source-contract passes, and six selected Python controls; none verifies the final revision. Earlier HO/DVR and P/dt outputs are retained without overwriting; `commutator-*` physical-sector interpretations are **SUPERSEDED** because they reassigned finite-representation commutator error to J1. `results/source-sha256.txt` is also an earlier snapshot. The tracked diff check is static hygiene only, not numerical evidence.

## Current source and interfaces

Edited or added source: `src/main_gpumd/run.cu`, `src/measure/{measure.cu,hac.cu,hac.cuh,u9_oho.cu,u9_oho.cuh,u9_oho_math.cuh}`, `tools/u9_oho/{bridge.py,spectrum.py,u9_rpmd.py,run_controls.py,analyze.py}`. Test and fixture source: `tests/u9_oho/{test_native_math.cpp,test_native.py,test_bridge.py,test_spectrum.py,test_rpmd.py,test_analysis.py,synthetic.cfg,general_u9_example.json}`. All eleven required markdown documents are in this directory.

For future execution, set `U9_THEORY_ROOT` and `PYTHONDONTWRITEBYTECODE=1`, then use `python -B -m unittest discover -s tests/u9_oho -v`; a quick control is `python -B tools/u9_oho/run_controls.py --theory-root <root> --output <new-directory> --quick`. The directory must not exist. These commands were not run against final source. Historical `--frames` commands belong to an older removed interface. The user waived local execution, not the scientific evidence required for quantum or material acceptance.
