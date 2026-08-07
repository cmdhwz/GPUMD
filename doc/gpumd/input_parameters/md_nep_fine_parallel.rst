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

For the current large-box NEP path, the implementation builds a reverse-edge
map for the filtered angular neighbor list, evaluates radial and angular force
contributions per edge, and uses block-per-atom segmented reductions for force
and virial assembly. Standard NEP descriptor accumulation also uses one block
per center atom, and the hidden-neuron ANN evaluation is parallelized within
that block.

The optimized path is used only for the supported single-GPU NEP/qNEP
potentials with a large box. For qNEP, the large-box path additionally uses
segmented real-space charge-force assembly, a parallel reduction for the
zero-mean ``D_real`` correction, and edge-parallel radial/angular force and BEC
assembly. qNEP now uses the same block-per-atom descriptor and parallel
hidden-neuron ANN evaluation. For large-box qNEP, PPPM charge spreading and
field interpolation also use atom/stencil parallel work with block reduction;
the FFT sequence itself is unchanged. After charge neutralization, BEC and
PPPM execute on separate streams and are joined before the charge-chain-rule
force stage. Real-space charge force uses a separate temporary force/
``D_real`` buffer and runs on a third stream; its contribution is merged after
the reciprocal and real-space streams complete. The qNEP local force chain
that depends on the combined ``D_real`` is still joined after this point.
Small-box calculations automatically retain the original path
because periodic image information cannot be represented by an atom index
alone. Other potential types and multi-GPU NEP also retain their original
behavior.

Examples
--------

::

  potential nep.txt
  md_nep_fine_parallel on
  ensemble nve
  run 10000

Omit the keyword, or set it to ``off``, to retain the original calculation
path.
