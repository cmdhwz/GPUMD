# U9 implementation plan

Baseline: codex/rpmd-ja-heat-current at 5e16110099831b60b574813db769c74b25846cd7, clean on entry. Preserve external projects and existing outputs.

1. Establish independent U9 derivative and full-G canonical mass adapter with finite-difference, canonical and instability tests. Reuse external theory modules through explicit source path; verify old special-case against them.
2. Build frozen same-source reference mapping and compare separately legacy-site and moving-flow definitions. Verify both against existing OHO and avoid claiming equivalence.
3. Integrate default-off native diagnostic only when fixed-reference, model identity, PBC, lifecycle and sampling can be verified end to end; candidate remains blocked absent Cartesian operators.
4. Execute CPU, old regression and available build controls, record command outputs and all missing quantum, GPU and material gates. Parent reviews changes; same implementer fixes findings.

Assumptions: fixed A and axis, three stable identities, no dynamics changes, no local scalar treated as bulk flux. Model reference definitions must be source matched. CUDA unavailability must not be described as a pass.

## Coverage against requested gates

Sections 4–7: U9 derivative and fixed G finite-model references implemented; old legacy mapping controls partial; complete old J/quantum equivalence and mode degeneracy remain. Sections 8 and 10: HO/DVR and RPMD software with short runs, not converged or statistically accepted. Sections 9 and 11: source diagnostic, no material candidate and no bulk U9 HAC. Section 12: parser/branch source and native C++ test exist but compiler/GPU/runtime checks are NOT RUN; default-off old outputs require matched executable run. Sections 13–14: parent review round 2 completed and corrections implemented; final independent source review completed. Section 15: all named documents exist, evidence and open items are recorded. Real parameter provenance and 3D transport remain externally unavailable; absence of CUDA compiler prevents CUDA execution, not source work.

## Current implementation state

The static source for the native diagnostic, finite-model bridge, spectral/RPMD runner and paired analysis has been reached, with source tests written but unrun under the user's instruction. The frozen six-case suite remains in the runner; the optional general U9 case does not replace it. The parent completed source-only review and accepted the static scope. Physical acceptance still requires material Cartesian operators and independently converged numerical evidence; the user waived local execution, not those scientific requirements.

