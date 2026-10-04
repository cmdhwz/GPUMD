import math
import re
import unittest
from pathlib import Path


def local_current(x1, x2, k, split):
    d = x2 - x1
    return (-(1.0 - split) * k * d * d, -split * k * d * d)


def fixed_reference_current(x1, x2, k, split, d0):
    d = x2 - x1
    return (-(1.0 - split) * k * d0 * d, -split * k * d0 * d)


def central_x1_derivative(function, x1, step=1.0e-6):
    plus = function(x1 + step)
    minus = function(x1 - step)
    return tuple((a - b) / (2.0 * step) for a, b in zip(plus, minus))


def three_site_energy(positions, k):
    return 0.5 * k * (positions[1] - positions[0]) ** 2


def three_site_site_energy(positions, k, center):
    del center
    return three_site_energy(positions, k) / 3.0


def three_site_edge_gradient(positions, k, center, neighbor, step=1.0e-5):
    plus = list(positions)
    minus = list(positions)
    plus[neighbor] += step
    minus[neighbor] -= step
    return (
        three_site_site_energy(plus, k, center)
        - three_site_site_energy(minus, k, center)
    ) / (2.0 * step)


def three_site_local_current(positions, k):
    flow = []
    for neighbor in range(3):
        value = 0.0
        for center in range(3):
            if center != neighbor:
                value -= (positions[neighbor] - positions[center]) * (
                    three_site_edge_gradient(positions, k, center, neighbor)
                )
        flow.append(value)
    return tuple(flow)


def three_site_fixed_reference_current(positions, reference, k):
    flow = []
    for neighbor in range(3):
        value = 0.0
        for center in range(3):
            if center != neighbor:
                value -= (reference[neighbor] - reference[center]) * (
                    three_site_edge_gradient(positions, k, center, neighbor)
                )
        flow.append(value)
    return tuple(flow)


def three_site_uniform_virial(positions, k):
    step = 1.0e-5
    gradient = []
    for atom in range(3):
        plus = list(positions)
        minus = list(positions)
        plus[atom] += step
        minus[atom] -= step
        gradient.append(
            (three_site_energy(plus, k) - three_site_energy(minus, k))
            / (2.0 * step)
        )
    total_virial = -sum(r * g for r, g in zip(positions, gradient))
    return (total_virial / 3.0,) * 3


def three_site_site_gradient(positions, k, center):
    step = 1.0e-5
    gradient = []
    for atom in range(3):
        plus = list(positions)
        minus = list(positions)
        plus[atom] += step
        minus[atom] -= step
        gradient.append(
            (
                three_site_site_energy(plus, k, center)
                - three_site_site_energy(minus, k, center)
            )
            / (2.0 * step)
        )
    return tuple(gradient)


def three_site_force(positions, k):
    step = 1.0e-5
    force = []
    for atom in range(3):
        plus = list(positions)
        minus = list(positions)
        plus[atom] += step
        minus[atom] -= step
        force.append(
            -(
                three_site_energy(plus, k) - three_site_energy(minus, k)
            )
            / (2.0 * step)
        )
    return tuple(force)


class StationOperatorContractTest(unittest.TestCase):
    def test_reference_edge_weight_is_fixed_in_reference_source(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "measure"
            / "rpmd_ja_reference.cu"
        ).read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(
                r"ref\.displacement\[alpha\]\s*\*\s*it->second\.derivative\[mu\]"
            ),
        )
        self.assertNotRegex(source, r"it->second\.displacement\[alpha\]")

    def test_virial_derivative_and_fixed_edge_operator_are_distinct(self):
        k, d0, x1, x2, split = 2.0, 1.0, 0.0, 1.0, 1.0
        current_derivative = central_x1_derivative(
            lambda moved_x1: local_current(moved_x1, x2, k, split), x1
        )
        fixed_edge_derivative = central_x1_derivative(
            lambda moved_x1: fixed_reference_current(
                moved_x1, x2, k, split, d0
            ),
            x1,
        )
        self.assertAlmostEqual(current_derivative[1], 4.0, places=8)
        self.assertAlmostEqual(fixed_edge_derivative[1], 2.0, places=8)

        # For U1=E,U2=0 and x=(1,0), Eq. 44 removes the current-virial
        # geometry and force terms to recover the fixed-reference operator.
        d = x2 - x1
        v_transpose_l = (-k * d, k * d)
        force = (k * d, -k * d)
        psi = (force[0], 0.0)
        recovered = tuple(
            current_derivative[j] - v_transpose_l[j] - psi[j] for j in range(2)
        )
        self.assertAlmostEqual(recovered[0], fixed_edge_derivative[0], places=8)
        self.assertAlmostEqual(recovered[1], fixed_edge_derivative[1], places=8)

        uniform_virial_derivative = central_x1_derivative(
            lambda moved_x1: tuple(
                -0.5 * k * (x2 - moved_x1) ** 2 for _ in range(2)
            ),
            x1,
        )
        uniform_recovered = tuple(
            uniform_virial_derivative[j] - v_transpose_l[j] - psi[j]
            for j in range(2)
        )
        self.assertAlmostEqual(uniform_recovered[0], 2.0, places=8)
        self.assertAlmostEqual(uniform_recovered[1], 0.0, places=8)
        self.assertGreater(
            math.dist(uniform_recovered, fixed_edge_derivative), 1.0
        )

    def test_equal_energy_and_uniform_virial_partition_can_match(self):
        # Splitting the same pair energy equally gives the same equal-site
        # virial as the uniform assignment in this two-site model.
        k, d0, x1, x2, split = 2.0, 1.0, 0.0, 1.0, 0.5
        current_derivative = central_x1_derivative(
            lambda moved_x1: local_current(moved_x1, x2, k, split), x1
        )
        uniform_virial_derivative = central_x1_derivative(
            lambda moved_x1: tuple(
                -0.5 * k * (x2 - moved_x1) ** 2 for _ in range(2)
            ),
            x1,
        )
        fixed_edge_derivative = central_x1_derivative(
            lambda moved_x1: fixed_reference_current(
                moved_x1, x2, k, split, d0
            ),
            x1,
        )
        self.assertAlmostEqual(current_derivative[0], uniform_virial_derivative[0])
        self.assertAlmostEqual(current_derivative[1], uniform_virial_derivative[1])
        d = x2 - x1
        v_transpose_l = (-0.5 * k * d, 0.5 * k * d)
        force = (k * d, -k * d)
        psi = (force[0], 0.0)
        recovered = tuple(
            current_derivative[j] - v_transpose_l[j] - psi[j] for j in range(2)
        )
        self.assertAlmostEqual(recovered[0], fixed_edge_derivative[0], places=8)
        self.assertAlmostEqual(recovered[1], fixed_edge_derivative[1], places=8)

    def test_three_site_equal_energy_and_uniform_virial_need_not_match(self):
        positions = (0.0, 1.0, 3.0)
        k = 2.0
        local_c = central_x1_derivative(
            lambda moved_x1: three_site_local_current(
                (moved_x1, positions[1], positions[2]), k
            ),
            positions[0],
            step=1.0e-4,
        )
        fixed_h = central_x1_derivative(
            lambda moved_x1: three_site_fixed_reference_current(
                (moved_x1, positions[1], positions[2]), positions, k
            ),
            positions[0],
            step=1.0e-4,
        )
        uniform_c = central_x1_derivative(
            lambda moved_x1: three_site_uniform_virial(
                (moved_x1, positions[1], positions[2]), k
            ),
            positions[0],
            step=1.0e-4,
        )
        v_transpose_l = three_site_site_gradient(positions, k, center=0)
        force = three_site_force(positions, k)
        psi = (force[0], 0.0, 0.0)
        local_recovered = tuple(
            local_c[j] - v_transpose_l[j] - psi[j] for j in range(3)
        )
        uniform_recovered = tuple(
            uniform_c[j] - v_transpose_l[j] - psi[j] for j in range(3)
        )

        for actual, expected in zip(local_c, (4.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(v_transpose_l, (-2.0 / 3.0, 2.0 / 3.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(force, (2.0, -2.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(psi, (2.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(fixed_h, (8.0 / 3.0, -2.0 / 3.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(local_recovered, fixed_h):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(uniform_c, (4.0 / 3.0,) * 3):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(uniform_recovered, (0.0, 2.0 / 3.0, 4.0 / 3.0)):
            self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
