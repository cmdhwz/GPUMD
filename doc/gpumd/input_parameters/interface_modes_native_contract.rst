.. _interface_modes_native_contract:

Native inputs available for later interface analyses
====================================================

The native programs provide data and general simulation primitives for later user-written interface analysis. The items below describe available input paths; they do not mean that the corresponding modal, wavepacket, power-transfer, lifetime, or mass-mismatch analyses have been implemented.

M4: modal projections
---------------------

``dump_interface_atoms`` selects atoms through the frozen M0 bin map and writes wrapped and unwrapped positions, velocity, force, and per-atom potential. Its global zero-based atom IDs join to ``<output_prefix>_groups.dat`` for reference position, type, mass, material, bath, fixed, moving, and bin labels. The existing :ref:`dump_xyz <kw_dump_xyz>` also supports user-defined group selection and velocity/force/potential/unwrapped-position quantities, but gathers the full system before filtering on the host. Neither dumper projects onto modes or parses Phonopy data. Relevant source paths are ``src/measure/interface_atoms.cu::DumpInterfaceAtoms::end_of_step``, ``src/measure/dump_xyz.cu::Dump_XYZ::end_of_step``, and ``src/measure/interface_geometry.cu::InterfaceGeometry::write_outputs``.

The geometry JSON exposes ``reference_positions_role: geometry_mapping_only`` and ``equilibrium_reference: external_required``; the first ``groups.dat`` comment says the same. These are wrapped pre-run mapping positions, not guaranteed equilibrated harmonic ``R0`` coordinates. If ``model.xyz`` already contains a wavepacket displacement, those shifted positions become the mapping reference and must not be reused as the equilibrium crystal positions. ``dump_interface_atoms`` starts at its first scheduled post-``compute2`` sample and has no t=0 frame, which its output header states explicitly. A later offline projection therefore needs separately saved initial positions plus the matching equilibrium ``R0``, masses, global atom IDs, unit-cell indices, and mode phase/eigenvector metadata. The output rows keep IDs so the external analysis can join these inputs without adding mode-specific native formats.

M5: wavepacket setup and harmonic reference
--------------------------------------------

Velocity initialization from ``model.xyz`` and NVE integration already exist in ``src/model/read_xyz.cu::read_xyz`` and ``src/integrate/ensemble_nve.cu``. The existing FCP can read order-2 force constants, their cluster list, and ``r0.in`` from the directory named by the FCP potential file. ``src/force/fcp.cu::FCP::read_fc2`` reads ``fcs_order2.in`` and ``clusters_order2.in``; ``FCP::compute`` evaluates the potential and force, then subtracts the total force divided equally among atoms in ``gpu_correct_force``. There is no separate input for an arbitrary reference-force vector ``F_ref`` in this FCP path.

FCP stores ``r0``, displacement ``u``, and force constants as ``float``. ``gpu_get_u`` evaluates coordinate minus reference coordinate directly, without MIC wrapping; the order-2 cluster reference bond itself is MIC-adjusted while reading its geometry. A harmonic comparison therefore needs small displacements that do not cross the periodic boundary, a relaxed reference configuration, and force constants with the intended acoustic-sum-rule behavior. Independently validate force/energy agreement before interpreting a wavepacket result. This contract records existing behavior; it does not modify FCP physics.

For a short reference check, evaluate the same nearly relaxed structure at ``R0`` and at deterministic small, non-crossing displacements (for example, a few amplitudes between 0.001 and 0.01 Å). Compare the order-2 force and quadratic energy change with finite differences and check the remaining net force after FCP's equal mean-force subtraction. Repeat for a small set of normalized modes and verify that restoring the displacement sign restores the force sign. This is a check procedure, not an unconditional validation of arbitrary IFC files. The current FCP path has no general input for an arbitrary constant ``F_ref`` offset.

M6: anharmonic modal power
--------------------------

NEP forces, per-atom potential, and selected local atom snapshots are available from the existing ``compute_force`` path in ``src/main_gpumd/run.cu`` and the ``dump_interface_atoms`` action. The geometry mapping provides fixed reference IDs and labels. These data can support a user-written second-order IFC residual and subsequent projections, but the native interface actions do not compute IFC residuals or project anharmonic power into modes. Evaluate ``F_NEP(R)`` and ``F_harm(R)=F_ref-Phi*u`` on the same instantaneous configuration; do not subtract forces from two separately advanced trajectories frame by frame. The selected analysis core must include an IFC2 interaction halo large enough to supply displacements from every force-constant neighbor, even if only the inner core is projected. Snapshot forces are the current ``atom.force_per_atom`` and include contributions from any enabled force actions, so residual analysis should use a controlled pure-NEP NVE run.

M7: mean-free-path analysis
---------------------------

No native interface lifetime, recovery-length, or mean-free-path estimator is added here. Users can consume the local snapshots and spectra using their own offline workflow; neither an MFP nor a recovery length is inferred by the native dump.

M8: mass-scaling mismatch control variable
-------------------------------------------

Per-atom mass input already exists in ``model.xyz``. ``src/model/read_xyz.cu::read_xyz`` reads the optional mass property (or supplies the element-table mass when absent), and the resulting ``atom.mass`` array is passed to integrators such as ``src/integrate/ensemble_nve.cu``. Existing modal-analysis code also mass-scales velocities in ``src/measure/modal_analysis.cu``. There is no native interface mass-scaling mismatch diagnostic or control-variable analysis in this change.
