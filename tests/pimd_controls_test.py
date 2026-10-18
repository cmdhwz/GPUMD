from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/integrate/ensemble_pimd.cu").read_text(encoding="utf-8")
RUN_SOURCE = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf-8")


def function_body(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {marker}")


def project_global_com(velocities, masses, enabled):
    if not enabled:
        return velocities.copy()
    momentum = np.einsum("i,kid->d", masses, velocities)
    return velocities - momentum / (len(velocities) * masses.sum())


def test_global_com_projection():
    masses = np.array([1.0, 16.0, 23.0])
    velocities = np.array(
        [
            [[1.0, -2.0, 0.5], [3.0, 0.5, -1.0], [-2.0, 1.0, 4.0]],
            [[-0.5, 2.0, 1.5], [1.0, -3.0, 0.0], [2.5, 0.0, -2.0]],
            [[4.0, 1.0, -1.0], [-2.0, 2.5, 3.0], [0.0, -1.0, 2.0]],
            [[-3.0, 0.5, 2.0], [1.5, -2.0, 0.5], [4.0, 2.0, -1.5]],
        ]
    )
    internal_before = velocities - velocities.mean(axis=0, keepdims=True)
    stale_centroid_velocity = np.array(
        [[9.0, 2.0, -3.0], [-4.0, 5.0, 8.0], [2.0, -6.0, 1.0]]
    )

    projected = project_global_com(velocities, masses, enabled=True)
    com_velocity = np.einsum("i,kid->d", masses, velocities) / (
        len(velocities) * masses.sum()
    )
    projected_centroid_velocity = projected.mean(axis=0)
    np.testing.assert_allclose(
        np.einsum("i,kid->d", masses, projected), 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        projected - projected.mean(axis=0, keepdims=True), internal_before, atol=1e-13
    )
    np.testing.assert_allclose(
        projected_centroid_velocity,
        velocities.mean(axis=0) - com_velocity,
        atol=1e-13,
    )
    assert not np.allclose(projected_centroid_velocity, stale_centroid_velocity - com_velocity)
    np.testing.assert_array_equal(
        project_global_com(velocities, masses, enabled=False), velocities
    )


def test_global_com_projection_bead_counts():
    rng = np.random.default_rng(20261018)
    masses = np.exp(rng.uniform(np.log(1.0), np.log(180.0), size=12))

    for bead_count in (2, 4, 32, 128):
        velocities = rng.normal(size=(bead_count, len(masses), 3))
        internal_before = velocities - velocities.mean(axis=0, keepdims=True)

        projected = project_global_com(velocities, masses, enabled=True)
        projected_twice = project_global_com(projected, masses, enabled=True)

        np.testing.assert_allclose(
            np.einsum("i,kid->d", masses, projected)
            / (bead_count * masses.sum()),
            0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            projected - projected.mean(axis=0, keepdims=True),
            internal_before,
            atol=1e-12,
        )
        np.testing.assert_allclose(projected_twice, projected, atol=1e-12)


def test_centroid_only_cell_transforms():
    beads = np.array(
        [
            [[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]],
            [[-1.0, 0.5, 2.0], [0.0, 1.5, -3.0]],
            [[2.0, -1.0, 0.0], [-2.0, 0.0, 1.0]],
            [[0.5, 3.0, 1.0], [1.0, 2.0, 2.5]],
        ]
    )
    transforms = (
        np.eye(3) * 1.03,
        np.diag([0.97, 1.02, 1.05]),
        np.array([[1.02, 0.01, -0.02], [0.01, 0.98, 0.015], [-0.02, 0.015, 1.04]]),
    )
    centroid_before = beads.mean(axis=0)
    internal_before = beads - centroid_before

    for matrix in transforms:
        centroid_after = centroid_before @ matrix.T
        translated = beads + (centroid_after - centroid_before)[None, :, :]
        np.testing.assert_allclose(translated.mean(axis=0), centroid_after, atol=1e-13)
        np.testing.assert_allclose(
            translated - translated.mean(axis=0), internal_before, atol=1e-13
        )


def test_source_call_order_and_kernel_contracts():
    header = (ROOT / "src/integrate/ensemble_pimd.cuh").read_text(encoding="utf-8")
    helper = function_body(SOURCE, "void Ensemble_PIMD::remove_center_of_mass_velocity_")
    assert "if (!fix_com_)" in helper
    assert "gpu_find_momentum_beads" in helper
    assert "gpu_correct_momentum_beads" in helper
    assert "nonham_work_per_atom_" in helper
    assert "atom.velocity_per_atom.data()" in helper
    assert "remove_center_of_mass_velocity_(Atom& atom);" in header

    correction_signature = SOURCE[
        SOURCE.index("static __global__ void gpu_correct_momentum_beads(") :
        SOURCE.index("{", SOURCE.index("static __global__ void gpu_correct_momentum_beads("))
    ]
    correction = function_body(SOURCE, "static __global__ void gpu_correct_momentum_beads(")
    assert "double* g_average_velocity" in correction_signature
    assert "g_velocity[k][i + d * number_of_atoms] = corrected_velocity;" in correction
    assert "average_velocity[d] += corrected_velocity;" in correction
    assert "g_average_velocity[i + d * number_of_atoms] = average_velocity[d] / number_of_beads;" in correction

    initialize = function_body(SOURCE, "void Ensemble_PIMD::initialize_run(")
    assert initialize.index("initialize(atom);") < initialize.index(
        "remove_center_of_mass_velocity_(atom);"
    )

    compute1 = function_body(SOURCE, "void Ensemble_PIMD::compute1(")
    half_kick = compute1.index("gpu_half_kick<<<")
    com_fix = compute1.index("remove_center_of_mass_velocity_(atom);")
    thermo_update = compute1.index("update_thermo_(box, atom, thermo);")
    pressure_update = compute1.index("apply_barostat_(box, atom, thermo);")
    box_refresh = compute1.index("box.set_is_orthogonal();")
    free_step = compute1.index("gpu_nve_forward<<<")
    assert half_kick < com_fix < thermo_update < pressure_update < box_refresh < free_step
    assert "if (num_target_pressure_components > 0)" in compute1

    compute2 = function_body(SOURCE, "void Ensemble_PIMD::compute2(")
    assert compute2.index("gpu_half_kick<<<") < compute2.index("langevin(time_step, atom);")
    assert compute2.index("langevin(time_step, atom);") < compute2.index(
        "remove_center_of_mass_velocity_(atom);"
    )
    assert compute2.index("remove_center_of_mass_velocity_(atom);") < compute2.index(
        "gpu_apply_pbc<<<"
    )
    assert compute2.index("gpu_apply_pbc<<<") < compute2.index(
        "update_thermo_(box, atom, thermo);"
    )
    assert "apply_barostat_" not in compute2
    assert "gpu_pressure_" not in compute2

    langevin = function_body(SOURCE, "void Ensemble_PIMD::langevin(")
    assert "gpu_find_momentum_beads" not in langevin
    assert "gpu_correct_momentum_beads" not in langevin
    assert "device_momentum_beads[0][3] * number_of_beads" in SOURCE
    assert "total_momentum[d] * inverse_of_ring_polymer_mass" in SOURCE

    update_thermo = function_body(SOURCE, "void Ensemble_PIMD::update_thermo_(")
    thermo_kernels = (
        "gpu_average<<<",
        "gpu_find_kinetic_energy_virial_part<<<",
        "gpu_find_sum_1024<<<",
        "gpu_find_thermo<<<",
    )
    assert [update_thermo.index(kernel) for kernel in thermo_kernels] == sorted(
        update_thermo.index(kernel) for kernel in thermo_kernels
    )

    apply_barostat = function_body(SOURCE, "void Ensemble_PIMD::apply_barostat_(")
    for operation in (
        "cpu_pressure_isotropic(",
        "cpu_pressure_orthogonal(",
        "cpu_pressure_triclinic(",
        "gpu_pressure_isotropic<<<",
        "gpu_pressure_orthogonal<<<",
        "gpu_pressure_triclinic<<<",
    ):
        assert operation in apply_barostat
    assert "compute_pimd_beads" not in apply_barostat

    for name in ("orthogonal", "isotropic", "triclinic"):
        kernel = function_body(SOURCE, f"static __global__ void gpu_pressure_{name}(")
        assert "g_average_position" in kernel
        assert "g_beads_position[k][" in kernel
        assert "g_beads_position[k][index] *=" not in kernel
    assert "g_beads_position[k][index] += delta;" in function_body(
        SOURCE, "static __global__ void gpu_pressure_isotropic("
    )
    assert "g_beads_position[k][index] += delta;" in function_body(
        SOURCE, "static __global__ void gpu_pressure_orthogonal("
    )
    triclinic = function_body(SOURCE, "static __global__ void gpu_pressure_triclinic(")
    assert "g_average_position[i] = x_new;" in triclinic
    assert "g_beads_position[k][i] += dx;" in triclinic
    assert "g_beads_position[k][i + number_of_particles] += dy;" in triclinic
    assert "g_beads_position[k][i + number_of_particles * 2] += dz;" in triclinic

    pile_scale_parser = function_body(RUN_SOURCE, "void Run::parse_pimd_pile_scale(")
    assert "std::isfinite(pile_scale)" in pile_scale_parser
    assert "pile_scale <= 0.0" in pile_scale_parser

    perform_a_run = function_body(RUN_SOURCE, "void Run::perform_a_run(")
    pimd_guard = perform_a_run.index("if (is_pimd(integrate.get_type()))")
    correction_rejection = perform_a_run.index("if (velocity.do_velocity_correction)")
    initialize = perform_a_run.index("integrate.initialize(")
    assert pimd_guard < correction_rejection < initialize
    assert "correct_velocity does not update ring-polymer bead velocities." in perform_a_run


if __name__ == "__main__":
    test_global_com_projection()
    test_global_com_projection_bead_counts()
    test_centroid_only_cell_transforms()
    test_source_call_order_and_kernel_contracts()
