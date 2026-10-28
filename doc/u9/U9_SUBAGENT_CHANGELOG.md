# Implementer change log

Round 1: independent derivatives, full-G mass modes and distinct legacy-site/moving-flow finite-model references, with numerical controls. Parent found the first round incomplete.

Round 2: added source-level opt-in diagnostic action, strict config and mathematical validation, selected centroid gather and continuous branch tracking, HAC ordering/identity/clock gates, append-only segmented output, synthetic config and C++ functional test. Candidate hard-fails. Compilers are unavailable, so native source and C++ test remain uncompiled and unrun; parent review was pending at that round. No changes to the existing HAC numerical algorithm or force/integrator paths were intended.

Round 3 after parent source review: fixed nonfinite tolerance and explicit HAC freshness; split local/global time and added bead metadata; implemented WPE observer identity checks; added confinement and canonical image checks, extracted tracker/schema helpers and C++ tests; implemented HO/DVR and RPMD control runners. Frozen controls and refinements were actually run but are INCONCLUSIVE, with large commutator/basis discrepancies. Native build remains NOT RUN and parent review remained pending at that round.

Round 4: user directed static-only work. Implementer changed native reference tolerance, file size/read errors, GPU-array and element validation, finite mesh check and reusable projection; corrected spectral J1/J3 definition; added configurable full-window runner and group analysis with matching isotope pairs; wrote additional regression tests and documentation. No tests or numerical controls were run in this round. Previously generated commutator-sector results are superseded. Parent review remained pending at that round.

Round 5 static-only response: validated optional numeric arguments and separate frozen HMC settings; preserved one shared labelled trajectory across left/right references; added contract-bound raw flows and whole-group paired bootstrap; added general U9 JSON CLI; wrote generalized-metric, degenerate-rotation, old-polynomial and old-Weyl control code. Corrected the exact finite-model instantaneous Weyl/Moyal statement. These tests and implementations were not run. Parent acceptance remained pending at that round.

Round 6: parent requested finite symmetric positive-definite metric and strict rotation checks, use of `map_transport` by the actual moving reference, preflight confinement/minima before spectral allocation, bounded JSON types and corresponding unrun test code. Changes were made under the static-only instruction; parent source-only review verdict is documented separately. No final-source tests or runtime were performed.

Final documentation correction: parent completed the final source read and tracked diff hygiene, with source-only acceptance. Final-source tests and runtime remain NOT RUN; quantum/material remain unvalidated and candidate BLOCKED.

