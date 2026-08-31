.. _kw_compute_hac:
.. index::
   single: compute_hac (keyword in run.in)

:attr:`compute_hac`
===================

This keyword can be used to calculate the heat current autocorrelation (:term:`HAC`) and running thermal conductivity (:term:`RTC`) using the :ref:`Green-Kubo method <green_kubo_method>`.
The results will be written to the :ref:`hac.out output file <hac_out>`.

Syntax
------
This keyword has 3 required parameters and up to 4 optional flags::

  compute_hac <sampling_interval> <correlation_steps> <output_interval> [use_centroid_heat_flux] [split_qnep_heat_by_type] [deferred_centroid_qnep] [deferred_diagnostic_stage]

The first parameter is the sampling interval for the heat current data.
The second parameter is the maximum correlations steps.
The third parameter is the output interval of the :term:`HAC` and :term:`RTC` data.

The optional fourth parameter :attr:`use_centroid_heat_flux` can be set to 1 to evaluate the
full classical heat-flux operator on the instantaneous centroid structure during PIMD-related runs.
If omitted, the default is 0.

For a fixed-box PIMD, RPMD, or TRPMD run using a single qNEP potential and
``pimd_bead_batch on``, the centroid potential and virial are evaluated as one
auxiliary lane of the existing qNEP bead batch. This is the default behavior
when ``deferred_centroid_qnep`` is 0, and avoids an additional
single-configuration qNEP force call at every HAC sample. The auxiliary lane
is not integrated and does not change the physical bead forces or ring-polymer
dynamics. If these conditions are not met, the original single-configuration
centroid calculation is used as a safe fallback.

The optional fifth parameter :attr:`split_qnep_heat_by_type` can be set to 1 to export an
additional type-resolved heat-current file with the electrostatic and non-electrostatic
contributions separated for qNEP models.
If omitted, the default is 0.

The optional sixth parameter ``deferred_centroid_qnep`` enables delayed centroid
heat-flux evaluation for fixed-box PIMD/RPMD/TRPMD runs with a single qNEP
potential, ``pimd_bead_batch on``, and no qNEP type split. Set it to 1 to save
the centroid position and velocity at each HAC sample during dynamics, then
evaluate the saved frames in fixed batches of 32 during ``postprocess``. This
keeps centroid qNEP work out of the dynamics loop. The saved trajectory uses
host memory proportional to the number of HAC frames and is padded only inside
the final 32-frame postprocessing batch. Unsupported runs, or a nonzero
``split_qnep_heat_by_type``, fall back to the normal sampled centroid path.

For example, with a 5-step sampling interval and 2000 HAC frames::

  compute_hac 5 2000 1 1 0 1

The optional seventh parameter ``deferred_diagnostic_stage`` is a development
diagnostic control. It does not change the meaning of the preceding
``deferred_centroid_qnep`` flag, and is only valid when
``use_centroid_heat_flux=1``, ``split_qnep_heat_by_type=0``, and
``deferred_centroid_qnep=1``. A nonzero stage produces no HAC, thermal
conductivity, or heat-current output file. The stages are:

Stages ``2``--``4`` additionally require a fixed-box PIMD/RPMD/TRPMD run so
that the production deferred cache can be allocated. Stage ``1`` only counts
sampling points and does not require that cache.

* ``1``: count HAC sampling points only; do not allocate HAC, centroid, or
  deferred trajectory buffers.
* ``2``: allocate the same deferred CPU/GPU cache as production, but do not
  run the snapshot store kernel.
* ``3``: allocate the cache and run the GPU snapshot store, but do not perform
  D2H copies.
* ``4``: allocate, store, and perform chunked D2H copies, but do not evaluate
  qNEP, HAC, or thermal conductivity in ``postprocess``.

Stage ``0`` is the default production deferred mode. For example::

  compute_hac 5 2000 1 1 0 1 1  # stage 1: sampling counter only
  compute_hac 5 2000 1 1 0 1 2  # stage 2: cache allocation
  compute_hac 5 2000 1 1 0 1 3  # stage 3: cache allocation + GPU store
  compute_hac 5 2000 1 1 0 1 4  # stage 4: cache allocation + store + D2H
  compute_hac 5 2000 1 1 0 1 0  # stage 0: complete deferred HAC

These diagnostic stages are intended only to isolate performance costs and
should not be used for production conductivity calculations.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   time_step 1
   compute_hac 10 100000 1
   run 10000000

This means that

* You want to calculate the thermal conductivity using the :ref:`Green-Kubo method <green_kubo_method>` (the :term:`EMD` method) in this run, which contains 10 milillion steps with a time step of 1 fs.
* The heat current data will be recorded every 10 steps.
  Therefore, there will be 1 million heat current data in each direction.
* The maximum number of correlation steps is :math:`10^5`, which is one tenth of the number of heat current data.
  This is a very sound choice.
  The maximum correlation time will be :math:`10^5 \times 10=10^6` time steps, i.e., 1 ns.
* The :term:`HAC`/:term:`RTC` data will not be averaged before outputting, generating :math:`10^5` rows of data in the output file.

Example 2
^^^^^^^^^

.. code::

   compute_hac 10 100000 10

This is similar to the above example but with one expection:
The :term:`HAC`/:term:`RTC` data will be averaged for every 10 data before outputing, generating :math:`10^4` rows of data in the output file.

Example 3
^^^^^^^^^

.. code::

   compute_hac 50 4000 10 1 1

This enables both optional flags:

* The centroid heat-flux operator is used.
* For qNEP models, an additional type-resolved heat-current file is written with the electrostatic
  and non-electrostatic contributions separated.
  The file name is ``heat_current_type_resolved_qnep_split_centroid.out`` when the centroid flag is 1,
  and ``heat_current_type_resolved_qnep_split.out`` otherwise.

At the end of a run with the centroid flag enabled, the log reports the
number of centroid HAC samples served by the qNEP batch cache and the number
that used the serial fallback. For fixed-box qNEP PIMD/RPMD/TRPMD runs, the
auxiliary lane is allocated as part of a fixed ``P+1``-lane batch capacity but
is active only on HAC sampling force calls; non-sampling force calls use only
the ``P`` physical lanes. The log also reports the physical batch-call count
(including the initial force evaluation) and the active/inactive auxiliary-lane
counts. The split qNEP option still performs its separate non-electrostatic
evaluation at sampled frames.
