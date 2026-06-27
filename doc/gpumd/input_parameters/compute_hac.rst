.. _kw_compute_hac:
.. index::
   single: compute_hac (keyword in run.in)

:attr:`compute_hac`
===================

This keyword can be used to calculate the heat current autocorrelation (:term:`HAC`) and running thermal conductivity (:term:`RTC`) using the :ref:`Green-Kubo method <green_kubo_method>`.
The results will be written to the :ref:`hac.out output file <hac_out>`.

Syntax
------
This keyword has 3 required parameters and up to 2 optional flags::

  compute_hac <sampling_interval> <correlation_steps> <output_interval> [use_centroid_heat_flux] [split_qnep_heat_by_type]

The first parameter is the sampling interval for the heat current data.
The second parameter is the maximum correlations steps.
The third parameter is the output interval of the :term:`HAC` and :term:`RTC` data.

The optional fourth parameter :attr:`use_centroid_heat_flux` can be set to 1 to evaluate the
full classical heat-flux operator on the instantaneous centroid structure during PIMD-related runs.
If omitted, the default is 0.

The optional fifth parameter :attr:`split_qnep_heat_by_type` can be set to 1 to export an
additional type-resolved heat-current file with the electrostatic and non-electrostatic
contributions separated for qNEP models.
If omitted, the default is 0.

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
