.. _kw_md_nep_fine_parallel:

md_nep_fine_parallel
====================

Syntax
------

::

  md_nep_fine_parallel on|off

Description
-----------

This is an experimental, opt-in optimization for a single-GPU classical MD
run using a standard NEP or qNEP potential. The default is ``off``. The
keyword may appear before or after the ``potential`` keyword.

The first implementation stage builds a reverse-edge map for the filtered
large-box angular neighbor list and uses it during many-body force and virial
assembly. This removes the repeated reverse-neighbor search from the force
kernel. The descriptor and neural-network kernels are otherwise unchanged in
this stage.

The optimized path is used only for the supported single-GPU NEP/qNEP
potentials with a large box. Small-box calculations automatically retain the
original path because periodic image information cannot be represented by an
atom index alone. Other potential types and multi-GPU NEP also retain their
original behavior.

Examples
--------

::

  potential nep.txt
  md_nep_fine_parallel on
  ensemble nve
  run 10000

Omit the keyword, or set it to ``off``, to retain the original calculation
path.
