.. _kw_rpmd_ja:

   single: rpmd_ja (keyword in run.in)

:attr:`rpmd_ja`
===============

Enable the optional reference-flow correction to the centroid mechanical heat current during a fixed-cell RPMD calculation.

Syntax
------

::

  rpmd_ja generate <reference_file> <temperature_K> <fd_step>
  rpmd_ja generate_sparse <reference_file> <temperature_K> <fd_step> <kernel_table>
  rpmd_ja on <reference_file>
  rpmd_ja off

``generate`` writes a dense reference from the current atom structure and the active single short-range NEP potential. ``generate_sparse`` instead writes fixed sparse dynamical and site-gradient operators with a Chebyshev kernel table. The table exporter supplies an analytic interpolation bound and a numerically checked arithmetic reserve; these are reported as approximation error budgets. Place either command after ``potential`` and before any ``ensemble`` or ``run`` command. Export the P and Q approximants with ``tools/rpmd_ja_export_kernel.py``, then validate the generated reference with ``tools/rpmd_ja_sparse_check.py`` before measurement. Sparse runs require the verified stability certificate and report per-operator nonzero counts and GPU workspace bytes. They stop with an input error when that workspace does not fit; no dense fallback is attempted.

Build a kernel table directly with the repository exporter. A smaller ``U=8`` table may reduce the required polynomial degree only when :math:`\tau\sqrt{\text{spectral_bound}}\le 8`, where ``spectral_bound`` is the absolute CSR row-sum bound stored in the reference and used by the generator and checker. The exported table's interpolation and arithmetic error budgets must remain valid, and the stability check must pass; that check does not certify the kernel error budgets. The exporter does not automatically reduce ``U`` or shrink the kernel coverage::

  python tools/rpmd_ja_export_kernel.py --build-u 32 --output ja_kernel.tbl

When the coverage and error-budget conditions hold, explicitly select ``U=8`` with::

  python tools/rpmd_ja_export_kernel.py --build-u 8 --output ja_kernel_u8.tbl

Alternatively, an existing theory NPZ can be exported with ``python tools/rpmd_ja_export_kernel.py --npz kernel_theory.npz --output ja_kernel.tbl``.

In a reference-generation ``run.in``, put this after ``potential`` and before any ``ensemble`` or ``run`` command::

  rpmd_ja generate_sparse ja_reference_raw.bin 300 1e-4 ja_kernel.tbl

After GPUMD writes the reference, rebind the selected table, then run the CPU stability check on the rebound file to create its certificate::

  python tools/rpmd_ja_sparse_check.py ja_reference_raw.bin --kernel-table ja_kernel.tbl --output ja_reference.bin
  python tools/rpmd_ja_sparse_check.py ja_reference.bin

Then use this measurement ``run.in`` order, again after ``potential``::

  rpmd_ja on ja_reference.bin
  compute_hac 5 1000 1 1
  ensemble rpmd 32 300
  run 10000

The checker is an offline CPU positive-definiteness and certificate check. It must pass before the reference is used. A kernel-table export records its analytic interpolation budget and numerical arithmetic reserve; it does not certify that the fixed-reference approximation is accurate for arbitrary anharmonic motion.

The reference temperature must equal the constant RPMD target temperature. The option supports fixed-cell primitive RPMD with one supported short-range NEP model and a fixed integration time step. It reports an input error for other ensembles or potentials, moving/deforming cells, adaptive time steps, force-modifying actions, qNEP full-A currents, or split/deferred centroid HAC. ``off`` is the default and leaves existing HAC behavior unchanged.

The centroid position is followed continuously across periodic wraps. The run stops with an error if any periodic per-step fractional jump is ambiguous (at least 0.45) or if the accumulated fractional displacement from the fixed reference reaches 0.45. This bounds the reference displacement branch and does not wrap accumulated diffusion back toward the reference.

Observable and scope
--------------------

The current is :math:`J_A=J_{cent}^{mech}+\Delta J`, with :math:`\Delta J_\alpha=x_c^T\Delta H_\alpha v_c` and :math:`x_c=R_c-R_0`. :math:`J_{cent}^{mech}` is the full mechanical centroid heat current computed from a private single-centroid NEP energy, force, and virial evaluation. The fixed reference Hessian and site-flow derivative define a local quadratic correction around :math:`R_0`; it is a candidate observable for that neighborhood, not a general quantum heat current or a DC-conductivity guarantee. Harmonic quadratic calibration checks the implementation in the harmonic limit but does not establish accuracy for arbitrary anharmonic systems. This ``rpmd_ja`` option is separate from ``hac_current qnep_full_a`` despite both using an A label in their respective theory contexts.

Cost and validation
-------------------

The sparse backend shares three D-operator seed evolutions. For polynomial degree :math:`L`, and P/Q ranks :math:`R_P,R_Q`, the measured correction uses about :math:`(3L+2)` D-vector actions and :math:`3(R_P+R_Q)` site-operator vector actions per sampled frame, plus reductions. For the current implementation with both ranks nonzero, the sparse correction launches :math:`7L+23` kernels and zero-fill operations; at :math:`L=148`, that is 1059 launches. This count includes the existing translation projections and reductions. Each sparse matrix-vector kernel checks and sanitizes its own non-finite row, so separate full-vector scans are not launched. Each sample copies the latched error flag to the CPU once, after the centroid NEP evaluation and correction; a final non-sample step retains its branch check. GPU workspace bytes cover its cached CSR matrices, coefficients, and working vectors; the run separately has HAC and private NEP allocations. Runtime speed has not been benchmarked and is not implied by this launch count. Current and HAC output is buffered until ``post_run``; sampling performs no JA file writes.

The offline stability checker has a different cost profile: it forms and factorizes a dense anchored minor using :math:`O(D^2)` host memory and :math:`O(D^3)` Cholesky/LAPACK work, in addition to the CSR arrays. The reference reader now uses ``np.fromfile`` to avoid a full intermediate ``bytes`` object and its extra copy; this reduces transient reader memory but does not change the three-matrix Cholesky estimate. For :math:`N=5896`, one :math:`3(N-1)` square double matrix is about 2.33 GiB, so the checker's estimate is roughly :math:`3\times2.33` GiB plus CSR storage and a 256 MiB reserve. Kernel export, reference generation, kernel rebinding, and this checker are one-time preparation steps, not part of the sparse per-frame GPU cost. The full 5896-atom CPU stability check remains pending on a host with sufficient physical memory; the available memory here is about 1.3 GiB, so it was not run.

The production CUDA fixture exercises the actual sparse workspace, asymmetric directional operators, signed P/Q coefficients, nonuniform masses, translation projection, zero-rank correction, multi-cell private wrapping, and non-finite error latching including sparse SpMV overflow. It does not exercise the enclosing HAC sample synchronization or the fixed-reference branch tracker. Run it where CUDA is available with::

  nvcc -std=c++17 -Isrc tests/rpmd_ja_sparse_cuda_test.cu src/measure/rpmd_ja_sparse.cu src/utilities/error.cu -o rpmd_ja_sparse_cuda_test
  ./rpmd_ja_sparse_cuda_test

The fixture command is provided for CUDA validation; it has not been run in this environment.

Outputs
-------

``heat_current_rpmd_ja.out`` contains the centroid current, correction, and their sum in x, y, z. Current values use eV Angstrom per natural time unit. ``hac_rpmd_ja.out`` contains the separate x, y, and z self-correlations and running integrals, normalized by :math:`k_B T_{reference}^2 V`. Each HAC column is averaged over all valid time origins and has units ``(eV Angstrom/natural_time)^2``; each RTC column is in W/m/K. Sum the three RTC columns for the trace, or divide that sum by three for the isotropic average. The columns retain direction and are not averaged together. Existing centroid current and HAC files remain the comparison series. JA samples are kept in memory and both files are appended only after the run; incompatible older JA output schemas are rejected before integration starts.
