"""CPU formula checks against a local i-PI checkout, not a CUDA runtime test.

Run: python -B tests/pimd_ipi_reference_test.py [path/to/i-pi]
Only NumPy and the reference i-PI source are needed; nothing is installed.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    ipi_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "i-pi-main"
    if not (ipi_root / "ipi/engine/normalmodes.py").is_file():
        raise SystemExit("Pass the path to a local i-PI source checkout.")
    sys.path.insert(0, str(ipi_root.resolve()))
    from ipi.engine.normalmodes import NormalModes
    from ipi.engine.thermostats import ThermoPILE_L
    from ipi.inputs.motion.motion import InputMotion
    from ipi.inputs.normalmodes import InputNormalModes
    from ipi.inputs.thermostats import InputThermo

    assert InputMotion().fixcom.fetch() is True
    assert InputNormalModes().propagator.fetch() == "exact"
    assert InputThermo().pile_lambda.fetch() == 1.0
    # Tie the independent reference calculation to the formulas being audited.
    source = (ROOT / "src/integrate/ensemble_pimd.cu").read_text(encoding="utf-8")
    for expression in (
        "sin_factor = cayley * omega_k * time_step;",
        "temp_velocity[d] = cos_factor * old_velocity - sin_factor_times_omega * old_position;",
        "temp_position[d] = sin_factor_over_omega * old_velocity + cos_factor * old_position;",
        "-0.5 * pile_scale * time_step *",
        "(1 - c1 * c1) * K_B * temperature * number_of_beads / g_mass[n]",
    ):
        assert expression in source, expression

    errors = dict(transform=0.0, exact=0.0, cayley=0.0, pile=0.0)
    # Dimensionless units: both codes receive identical omega and dt.
    for beads in (2, 4, 8, 16, 32, 64, 128):
        omega = 2 * np.sin(np.arange(beads) * np.pi / beads)
        j = np.arange(1, beads + 1)
        transform = np.empty((beads, beads))
        transform[:, 0] = 1 / np.sqrt(beads)
        for k in range(1, beads):
            if k < beads // 2:
                transform[:, k] = np.sqrt(2 / beads) * np.cos(2 * np.pi * j * k / beads)
            elif k == beads // 2:
                transform[:, k] = (-1.0) ** j / np.sqrt(beads)
            else:
                transform[:, k] = np.sqrt(2 / beads) * np.sin(2 * np.pi * j * k / beads)
        errors["transform"] = max(
            errors["transform"], np.max(np.abs(transform.T @ transform - np.eye(beads)))
        )
        for dt in (0.01, 0.2, 0.8):
            for propagator in ("exact", "cayley"):
                nm = SimpleNamespace(
                    dt=dt, nbeads=beads, nm_factor=np.ones(beads),
                    omegak=omega, propagator=propagator,
                )
                reference = NormalModes.get_prop_pq(nm)
                gp = np.zeros_like(reference)
                gp[0] = [[1, 0], [dt, 1]]  # i-PI order is (mass-scaled p, q).
                w = omega[1:]
                if propagator == "exact":
                    cosine, sine = np.cos(w * dt), np.sin(w * dt)
                else:
                    factor = 1 / (1 + (w * dt / 2) ** 2)
                    cosine = factor * (1 - (w * dt / 2) ** 2)
                    sine = factor * w * dt
                gp[1:, 0, 0] = gp[1:, 1, 1] = cosine
                gp[1:, 0, 1] = -sine * w
                gp[1:, 1, 0] = sine / w
                errors[propagator] = max(errors[propagator], np.max(np.abs(gp - reference)))
            thermostat = SimpleNamespace(
                pilescale=1.0, nm=SimpleNamespace(dynomegak=omega), _thermos=[None] * beads
            )
            ipi_damping = np.exp(-0.5 * dt / ThermoPILE_L.get_tauk(thermostat))
            gp_damping = np.exp(-0.5 * 2.0 * dt * omega[1:])
            errors["pile"] = max(errors["pile"], np.max(np.abs(ipi_damping - gp_damping)))

    assert errors["transform"] < 1e-12, errors
    assert max(errors[k] for k in ("exact", "cayley", "pile")) < 1e-13, errors
    print("i-PI reference:", ipi_root.resolve())
    print("Maximum absolute errors:", errors)
    print("PASS: formula-level checks; full finite-step splitting and CUDA are not tested.")


if __name__ == "__main__":
    main()
