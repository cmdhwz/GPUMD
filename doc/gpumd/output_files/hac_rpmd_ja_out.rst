.. _hac_rpmd_ja_out:

``hac_rpmd_ja.out``
===================

Produced when ``rpmd_ja on`` is enabled, and appended after the run. The columns are lag index, lag time in ps, the x/y/z self-correlations of the total JA current, and their x/y/z running integrals. Each directional correlation uses all valid time origins and denominator ``sampled_frames - lag`` without mean subtraction. The HAC units are ``(eV Angstrom/natural_time)^2``. RTC is normalized by ``k_B T_reference^2 V`` and is in W/m/K; summing RTC_x/y/z gives the trace, and dividing the sum by three gives the isotropic average. No cross-direction correlations are included. Sparse segments also record the kernel error bounds, coefficient ranks and degree, CSR nonzero counts, and GPU workspace bytes.
