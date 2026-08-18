.. _kw_md_qnep_bec:

Classical MD qNEP BEC evaluation
================================

The optional keyword controls the Born effective charge (BEC) kernels in
ordinary classical MD runs using a qNEP potential::

    md_qnep_bec auto|on|off

``auto`` is the default. It skips the per-force BEC kernels unless a command
needs BEC values, such as ``compute_dpdt``, ``dump_xyz`` with the ``bec``
quantity, or ``add_efield ... bec``. ``on`` evaluates BEC at every force
call. ``off`` skips BEC evaluation and is only valid when no BEC-dependent
command is used.

This setting does not disable qNEP charge neutrality, PPPM/Ewald, real-space
electrostatics, forces, or virials. The existing ``pimd_qnep_batch_bec``
setting remains independent and controls PIMD/RPMD/TRPMD bead-batched BEC
evaluation.
