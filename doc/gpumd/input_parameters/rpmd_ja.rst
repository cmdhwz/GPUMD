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

``generate`` writes a dense reference from the current atom structure and the active single short-range NEP potential. ``generate_sparse`` uses a Chebyshev kernel table. For ordinary NEP, it writes sparse dynamical and site-gradient operators that must be rebound and checked with ``tools/rpmd_ja_sparse_check.py``. For qNEP charge mode 1 or 2 with PPPM, it directly writes the final v3 block reference and its stability sidecar. The table exporter supplies an analytic interpolation bound and a numerically checked arithmetic reserve; these are reported as approximation error budgets. Place either command after ``potential`` and before any ``ensemble`` or ``run`` command. Sparse runs report per-operator nonzero counts and GPU workspace bytes. They stop with an input error when that workspace does not fit; no dense fallback is attempted.

Build a kernel table directly with the repository exporter. A smaller ``U=8`` table may reduce the required polynomial degree only when :math:`\tau\sqrt{\text{spectral_bound}}\le 8`, where ``spectral_bound`` is the absolute CSR row-sum bound stored in the reference and used by the generator and checker. The exported table's interpolation and arithmetic error budgets must remain valid, and the stability check must pass; that check does not certify the kernel error budgets. The exporter does not automatically reduce ``U`` or shrink the kernel coverage::

  python tools/rpmd_ja_export_kernel.py --build-u 32 --output ja_kernel.tbl

When the coverage and error-budget conditions hold, explicitly select ``U=8`` with::

  python tools/rpmd_ja_export_kernel.py --build-u 8 --output ja_kernel_u8.tbl

Alternatively, an existing theory NPZ can be exported with ``python tools/rpmd_ja_export_kernel.py --npz kernel_theory.npz --output ja_kernel.tbl``.

In a reference-generation ``run.in``, put this after ``potential`` and before any ``ensemble`` or ``run`` command::

  rpmd_ja generate_sparse ja_reference_raw.bin 300 1e-4 ja_kernel.tbl

For qNEP charge mode 1 or 2 with PPPM, ``generate_sparse`` writes the ready-to-use v3 reference and its stability sidecar directly::

  rpmd_ja generate_sparse ja_reference.bin 300 1e-4 ja_kernel.tbl

The generated ``ja_reference.bin.stability`` sidecar must stay beside the reference. Use that path directly with ``rpmd_ja on ja_reference.bin``; the standalone Python preparation utility is not part of the production generation path.

For ordinary NEP sparse references, rebind the selected table and run the CPU stability check to create its certificate::

  python tools/rpmd_ja_sparse_check.py ja_reference_raw.bin --kernel-table ja_kernel.tbl --output ja_reference.bin
  python tools/rpmd_ja_sparse_check.py ja_reference.bin

With an equilibrated self-describing ``restart_beads.xyz``, use this measurement ``run.in`` order::

  potential nep.txt
  ensemble rpmd 32 300
  read_pimd_restart restart_beads.xyz
  rpmd_ja on ja_reference.bin
  compute_hac 5 1000 1 1
  run 10000

Without a bead restart, first run a separate PIMD equilibration input with ``dump_pimd_restart`` to create ``restart_beads.xyz``; then start the measurement input above. Do not begin the RPMD measurement directly from the centroid-only ``model.xyz`` state.

The checker is an offline CPU positive-definiteness and certificate check. It must pass before the reference is used. A kernel-table export records its analytic interpolation budget and numerical arithmetic reserve; it does not certify that the fixed-reference approximation is accurate for arbitrary anharmonic motion.

The reference temperature must equal the constant RPMD target temperature. The option supports fixed-cell primitive RPMD with a fixed integration time step, using either an ordinary short-range NEP v1/v2 reference or a qNEP v3 reference for charge mode 1 or 2 with PPPM. Other ensembles or potentials, moving/deforming cells, adaptive time steps, DFT-D3, unverified electrostatics settings, force-modifying actions, qNEP full-A currents, and split/deferred centroid HAC are rejected. qNEP references are tied to their model and mechanical-configuration fingerprint; an ordinary NEP reference cannot be applied to qNEP. ``off`` is the default and leaves existing HAC behavior unchanged.

The centroid position is followed continuously across periodic wraps. The run stops with an error if any periodic per-step fractional jump is ambiguous (at least 0.45) or if the accumulated fractional displacement from the fixed reference reaches 0.45. This bounds the reference displacement branch and does not wrap accumulated diffusion back toward the reference.

Finite-temperature qNEP additive reference workflow
---------------------------------------------------

The optional additive workflow fits a local finite-temperature force response from complete, fixed-cell RPMD bead samples. It produces an auxiliary coefficient package; the ordinary qNEP reference generation and measurement paths remain available. The training system must contain all mobile atoms under one Hamiltonian, with the same atom ordering, masses, cell, target temperature, and bead count as the raw reference. Do not select a subset of atoms and treat it as a smaller system. The fitting data are split chronologically into training and held-out validation frames; correlated frames may make this validation optimistic, so independent blocks and material-level uncertainty remain necessary.

The native ``rpmd_ja fit_samples`` route produces a ``finite_temperature_additive`` reference, not the bare qNEP Taylor Hessian. Read its ``.fit_trace.txt`` in stage order: ``FIT_RESPONSE_PASS`` records held-out response acceptance, ``FULL_CERTIFICATE_PASS`` records the stored-matrix SPD certificate, and ``REFERENCE_WRITTEN`` confirms that the reference package and sidecar were written. ``QP_PASS`` only describes the current finite stability-cut set and does not mean the reference is ready. The ``.fit.txt`` report also records unconstrained and constrained training force residuals, the relative squared-residual cost change, design singular values, and response-ratio extrema. A failed candidate keeps diagnostic artifacts and must not be used with ``rpmd_ja on``.

The separate offline ``fit_samples_covariance`` route reuses the same ``GPJASMP1`` samples, matching qraw, local additive package, ``prepare`` call, and ``J_A`` measurement chain. Its command is::

  rpmd_ja fit_samples_covariance <samples> <outfile> <cutoff> <epsilon> <response_tolerance> <fd_step> <kernel_table> <qraw> shrinkage <rho> max_iter <N>

This route is opt-in; ``fit_samples`` continues to use the force least-squares/QP method. The covariance method computes training statistics only from the first two thirds of frames and minimizes the Gaussian fluctuation objective with a graph-Laplacian baseline and a positive ``rho`` shrinkage prior. It searches the full feasible interval for the baseline parameter, including negative values when the raw reference is already stable. It uses exact Cholesky factors in the translation-complement space and analytic parameter gradients. BFGS coordinates are scaled from the baseline matrix and each parameter basis matrix; the reports include both the scaled KKT residual and raw parameter-gradient norm. ``max_iter`` is the BFGS iteration budget; line-search objective evaluations are reported separately. The numerical log-determinant barrier used to enforce ``D-epsilon I > 0`` is an optimizer parameter and is not the shrinkage strength. The optimizer's convergence status is separate from held-out response acceptance and the final stored-D certificate. Read ``baseline_mu``, ``baseline_epsilon_constraint_status``, objective terms, scaled KKT residual, raw gradient norm, barrier gap estimate, factorization count, solve RHS count, and elapsed time from the ``.fit.txt`` report. On optimizer or response failure, ``.covariance_state.txt`` records the baseline and candidate parameter vectors and training statistics; the method does not write an accepted reference on that path.

With :math:`D(\theta)=D_t+D_{\rm add}(\theta)` and the baseline :math:`D_b=D(\theta_b)`, the fitted objective is

.. math::

   \mathcal L_\rho(D)=\frac{\beta}{2}\operatorname{Tr}(D\widehat C)-\frac{1}{2}\log\det D
   +\frac{\rho}{2}\left[\operatorname{Tr}(DD_b^{-1})-\log\det(DD_b^{-1})-n\right].

The prior term penalizes departures from the finite-temperature graph baseline when the sample covariance has weak or missing directions. It does not change the sampled trajectory or replace the later held-out response and full stored-D checks. A status is successful only after ``BASELINE_SPD_PASS``, ``COVARIANCE_OPTIMIZER_CONVERGED``, ``FIT_RESPONSE_PASS``, ``FULL_CERTIFICATE_PASS``, and ``REFERENCE_WRITTEN`` appear in order in ``.fit_trace.txt``.

First equilibrate the beads and collect aligned centroid samples and ``mean.xyz``. List every bead file explicitly; this two-bead invocation is only a short example. Real eight-decimal ``dump_beads`` cells require ``--cell-model`` with the original full-precision structure. Omitting it is suitable only when the supplied XYZ already has enough cell precision for downstream raw-reference checks::

  python tools/rpmd_ja_fit_reference.py collect --bead-files bead_0.xyz bead_1.xyz --temperature 300 --output samples.npz --mean-model mean.xyz --masses 1.008 15.999 --cell-model original_model.xyz

Replace the example mass list with every atom's actual mass in original model order. The forces in each bead dump must be that bead's physical force under the sampled Hamiltonian, not ring-spring forces and not a force recomputed at the centroid :math:`R_c`. The collector averages bead forces exactly once, :math:`P^{-1}\sum_b F_b`, while forming the centroid samples. The fitter assumes these forces are energy-consistent with the Hamiltonian used for raw derivatives; independently check qNEP/PPPM energy and force deviations against the material accuracy target. The collector uses the full-precision cell from ``--cell-model`` after checking every bead dump against the eight-decimal serialization bound. It writes ``mean.xyz`` from the first two thirds of chronological frames. In a separate raw-generation directory, copy this output as ``model.xyz`` and provide the same full-precision cell, atom types, masses, qNEP model, and PPPM settings as the bead simulation. Then generate the raw reference::

  potential nep.txt
  kspace pppm 1.0
  rpmd_ja generate_raw ja_raw.qraw 300 1e-4 ja_kernel_u8.tbl

Run GPUMD in that directory. Reuse the PPPM spacing from the original Hamiltonian (or the default 1.0 Angstrom spacing if that was used); do not change electrostatics settings. The raw ``R0`` must match the collected training mean or ``fit`` will reject it. Then fit and prepare the auxiliary package. Use the same kernel table for raw generation and preparation::

  python tools/rpmd_ja_fit_reference.py fit --raw ja_raw.qraw --samples samples.npz --output ja_additive.bin --cutoff 5.0 --epsilon 1e-8 --response-tolerance 0.15

The fitter uses one fractional minimum-image distance per ordered distinct atom pair within the cutoff. It does not enumerate multiple periodic images or self-image edges and is not the full qNEP neighbor graph; this defines the fitted model space, whose cutoff and system-size adequacy must be checked on held-out data. Collection keeps ring unwrapping as a helper but rejects bead configurations whose resulting centroid differs from the production bead-zero minimum-image centroid. Samples must also remain within the existing 0.45 fractional fixed-reference branch. Sparse dump cadence can alias periodic motion, so choose sampling that resolves the dynamics; no cadence threshold is inferred by this utility.

For CPU-only preparation, build the final v3 reference with the fitted package::

  python tools/rpmd_ja_qnep_prepare.py ja_raw.qraw ja_additive.bin --kernel-table ja_kernel_u8.tbl --output ja_reference.bin --lossless

The equivalent GPUMD production preparation command is ``rpmd_ja prepare ja_raw.qraw ja_reference.bin ja_kernel_u8.tbl ja_additive.bin`` before any ``ensemble`` or ``run``. Both forms bind the auxiliary package to its raw source, temperature, bead count, and fixed cell. To measure, use matching reference metadata and full-system ordering, and initialize from an equilibrated bead-resolved restart within the reference's 0.45 fractional branch. The measurement need not start at the training mean position. Enable ``rpmd_ja on ja_reference.bin`` in the measurement input. A separate legacy preparation without the additive package is the native comparison. The JA current implementation is unchanged.

The collector's synthetic integration fixture uses full-precision coordinates and forces to check exact zero-addition recovery; real ``dump_beads`` coordinates, forces, and lattice are serialized to eight decimal places. The optional cell model restores only the original cell and does not recover quantized positions or forces. Quantization sensitivity must therefore be measured on real data before interpreting a material fit. The included ``U=8`` kernel fixture is a repository test input, not evidence of physical kernel accuracy. The CPU workflow tests do not run qNEP generation or the CUDA prepare path.

The fitter limits dense matrices to 6,000,000 elements, the dense parameter set to 1,600, the translation-free internal dimension to 384 (at most 129 atoms), and the uncompressed sample arrays to 256 MiB. Additive CPU preparation additionally limits dense assembly to 4,000,000 elements and the package to 64 MiB. These limits do not include all NumPy, LAPACK, eigensolver, process, or source-matrix memory; estimate host memory before larger jobs. The GPU prepare path has its own preflight and memory constraints. No larger-system performance or fit accuracy is implied.

``heat_current_rpmd_ja.out`` stores the three directional centroid current, additive correction, and their sum. ``hac_rpmd_ja.out`` stores only the three directional :math:`C_{AA}` self-correlation components and their integrals. All four correlation terms can be reconstructed offline at each lag from the three current series in ``heat_current_rpmd_ja.out``; GPUMD does not write four HAC files.

Observable and scope
--------------------

The current is :math:`J_A=J_{cent}^{mech}+\Delta J`, with :math:`\Delta J_\alpha=x_c^T\Delta H_\alpha v_c` and :math:`x_c=R_c-R_0`. :math:`J_{cent}^{mech}` is the full mechanical centroid heat current computed by a private single-centroid evaluator. For qNEP, both the baseline and reference generation retain the native charge-chain virial and :math:`q(R_c)` for each sampled centroid; they do not add the separate ``qnep_full_a`` current correction. qNEP references use the explicit ``native_reference_transport`` policy :math:`H_\alpha^T=C_\alpha-V^TL_\alpha-\Psi E_\alpha`; this is a defined transport operator, not an unconditional canonical-equivalence claim. The qNEP generation command directly creates the final v3 block reference and sidecar, and the separate Python utility remains available for offline work without being required by GPUMD generation. A qNEP reference is bound to its charge mode, electrostatics settings, model, and mechanical-configuration fingerprint; an ordinary NEP reference cannot be applied to qNEP. The fixed reference Hessian and site-flow derivative define a local quadratic correction around :math:`R_0`; it is a candidate observable for that neighborhood, not a general quantum heat current or a DC-conductivity guarantee. Harmonic quadratic calibration checks the implementation in the harmonic limit but does not establish accuracy for arbitrary anharmonic systems. This ``rpmd_ja`` option remains separate from ``hac_current qnep_full_a``.

Cost and validation
-------------------

The sparse backend shares three D-operator seed evolutions. For polynomial degree :math:`L`, and P/Q ranks :math:`R_P,R_Q`, the measured correction uses about :math:`(3L+2)` D-vector actions and :math:`3(R_P+R_Q)` site-operator vector actions per sampled frame, plus reductions. For the current implementation with both ranks nonzero, the sparse correction launches :math:`7L+23` kernels and zero-fill operations; at :math:`L=148`, that is 1059 launches. This count includes the existing translation projections and reductions. Each sparse matrix-vector kernel checks and sanitizes its own non-finite row, so separate full-vector scans are not launched. Each sample copies the latched error flag to the CPU once, after the centroid NEP evaluation and correction; a final non-sample step retains its branch check. GPU workspace bytes cover its cached CSR matrices, coefficients, and working vectors; the run separately has HAC and private NEP allocations. Runtime speed has not been benchmarked and is not implied by this launch count. Current and HAC output is buffered until ``post_run``; sampling performs no JA file writes.

The qNEP fixed-reference modal cache has a one-time generation and per-sample cost boundary. Its uncompressed fine/coarse :math:`V,C` and :math:`K` data volume is :math:`(23/3)D^2` doubles, or about 17.9 GiB at :math:`N=5896`, before any block compression. This is a resource-planning estimate, not the size of the final block file or evidence that a 5896-atom run fits. Generation performs about :math:`4D+13` private qNEP evaluations for the current three-direction curvature probes. The private qNEP/PPPM and stability/compression workspace peaks have not been measured; check available host memory, temporary storage, and GPU memory against the actual system before running, and do not treat 5896 as validated. During measurement initialization, source CPU tiles are uploaded one tile at a time and released after cache construction; no packed full-matrix host copy is made. The cache removes exactly the three known mass-translation modes, performs one projected :math:`D_s` eigensolve and three modal :math:`B^T` transforms at :math:`O(D^3)` setup cost, and retains the full P/Q tables and their stated error budget. Its steady cache uses :math:`Dr+3r^2` doubles (about 9.32 GiB at :math:`N=5896`); construction uses about :math:`2Dr+D^2+3r^2` doubles (about 13.98 GiB), with the eigensolver phase and its workspace determining the maximum peak. Small vectors and solver workspace add to these estimates, and GPUMD's other allocations are separate. Each sample performs two modal projections and three cached matrix-vector actions, with :math:`O(Dr+3r^2)` work and no D-order recurrence, eigensolve, or allocation. Insufficient GPU memory is an error; the former expensive recurrence path is not used as a fallback. These are estimates and have not been confirmed by CUDA measurement. Reference finite differences run only during one-time generation, and current/HAC files are written after the run rather than during sampling.

The ordinary-NEP sparse offline checker has a different cost profile: it forms and factorizes a dense anchored minor using :math:`O(D^2)` host memory and :math:`O(D^3)` Cholesky/LAPACK work, in addition to the CSR arrays. Its Python reader uses ``np.fromfile`` to avoid a full intermediate ``bytes`` object and its extra copy; this reduces transient reader memory but does not change the three-matrix Cholesky estimate. For :math:`N=5896`, one :math:`3(N-1)` square double matrix is about 2.33 GiB, so the checker's estimate is roughly :math:`3\times2.33` GiB plus CSR storage and a 256 MiB reserve. Kernel export, ordinary-NEP reference generation, kernel rebinding, and this checker are one-time preparation steps, not part of the sparse per-frame GPU cost. The full 5896-atom NEP CPU stability check remains pending on a host with sufficient physical memory; the available memory here is about 1.3 GiB, so it was not run.

The native qNEP centroid HAC evaluator has a GPU-enabled CLI comparison script. Supply a baseline executable, the candidate executable, and a case directory containing ``run.in``, ``model.xyz``, ``nep.txt``, and a self-describing ``restart_beads.xyz``. The input must use exactly ``potential nep.txt``, ``rpmd_ja off``, one ``ensemble rpmd`` and one positive ``run`` production segment, one direct centroid ``compute_hac`` with split and deferred output disabled, and one ``dump_pimd_restart`` interval reached by that run. Other ``rpmd_ja`` modes, any ``hac_current`` mode (including ``qnep_full_a``), TRPMD, and other ensemble segments are rejected. The fixture checks this native full-current/direct-centroid path; it does not test qNEP ``rpmd_ja``. Each isolated run removes prior comparison outputs, reads a preserved copy of the initial restart, and must create a fresh final bead restart. The script compares the centroid current, HAC, optional ``thermo.out``, and final bead restart::

  python tests/rpmd_ja_qnep_centroid_cli_test.py --baseline-gpumd /path/to/baseline/gpumd --candidate-gpumd /path/to/candidate/gpumd --case-dir /path/to/qnep-centroid-case

This is a production-evaluator comparison and requires working CUDA executables plus an equilibrated bead restart. It has not been run in this environment.

For an end-to-end qNEP JA acceptance run, provide one GPUMD executable, a relaxed ``model.xyz`` without velocities, a charge1/charge2 qNEP model, and a kernel table::

  python tests/rpmd_ja_qnep_end_to_end_test.py --gpumd /path/to/gpumd --model-xyz /path/to/model.xyz --qnep-model /path/to/nep.txt --kernel-table /path/to/ja_kernel.tbl --temperature 300

The script asks GPUMD to write the final v3 reference and sidecar directly, initializes a fixed-seed bead restart with a short PIMD run, and runs the same short standard-RPMD case with ``rpmd_ja off`` and ``rpmd_ja on``. It compares centroid current/HAC, the sampled 18-column ``thermo.out`` including the total-energy drift :math:`K+U`, and final bead restart; it also checks finite three-direction J_A data, :math:`J_A=J_{cent}+\Delta J`, and reconstructs every reported JA HAC lag from all four centroid/correction and cross terms. Each production run reads a separately named copy of the seed restart and must write a fresh final restart at the requested final step. The generated short restart is for software acceptance, not equilibration or production transport. No zero-correction reference is synthesized. CUDA executables are required; this end-to-end run has not been run in this environment.

The production CUDA fixture exercises the actual sparse workspace and v3 block GPU matvecs (dense and low-rank tiles, partial tiles, multiple vectors, and workspace reinitialization), asymmetric directional operators, signed P/Q coefficients, nonuniform masses, translation projection, zero-rank correction, multi-cell private wrapping, and non-finite error latching including sparse SpMV overflow. It does not exercise the enclosing HAC sample synchronization or the fixed-reference branch tracker. Run it where CUDA is available with::

  nvcc -std=c++14 -Isrc tests/rpmd_ja_sparse_cuda_test.cu src/measure/rpmd_ja_sparse.cu src/measure/rpmd_ja_qnep_cached.cu src/utilities/error.cu -lcublas -lcusolver -o rpmd_ja_sparse_cuda_test
  ./rpmd_ja_sparse_cuda_test

The fixture command is provided for CUDA validation; it has not been run in this environment.

Outputs
-------

``heat_current_rpmd_ja.out`` contains the centroid current, correction, and their sum in x, y, z. Current values use eV Angstrom per natural time unit. ``hac_rpmd_ja.out`` contains the separate x, y, and z self-correlations and running integrals, normalized by :math:`k_B T_{reference}^2 V`. Each HAC column is averaged over all valid time origins and has units ``(eV Angstrom/natural_time)^2``; each RTC column is in W/m/K. Sum the three RTC columns for the trace, or divide that sum by three for the isotropic average. The columns retain direction and are not averaged together. Existing centroid current and HAC files remain the comparison series. JA samples are kept in memory and both files are appended only after the run; incompatible older JA output schemas are rejected before integration starts.
