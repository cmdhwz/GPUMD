import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_centroid_force_diagnostic_reuses_immediate_centroid_hac_data():
    run_source = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf-8")
    measure_source = (ROOT / "src/measure/measure.cu").read_text(encoding="utf-8")
    hac_header = (ROOT / "src/measure/hac.cuh").read_text(encoding="utf-8")
    hac_source = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
    diagnostic_header = (ROOT / "src/measure/centroid_force_diagnostic.cuh").read_text(
        encoding="utf-8"
    )
    diagnostic_source = (ROOT / "src/measure/centroid_force_diagnostic.cu").read_text(
        encoding="utf-8"
    )

    assert '#include "measure/centroid_force_diagnostic.cuh"' in run_source
    assert 'strcmp(param[0], "centroid_force_diagnostic")' in run_source
    assert "new Centroid_Force_Diagnostic" in run_source
    assert "centroid_force_diagnostic->set_hac(centroid_force_hac)" in run_source
    assert "centroid_force_ready" in hac_header
    assert "centroid_force_per_atom" in hac_header
    assert "centroid_potential_per_atom" in hac_header
    assert "centroid_force_step_" in hac_header
    assert hac_source.count("centroid_force_step_ = step + 1") == 2

    assert "class Centroid_Force_Diagnostic : public Property" in diagnostic_header
    assert "gpu_reduce_centroid_force_diagnostic" in diagnostic_source
    assert "atom.force_per_atom" in diagnostic_source
    assert "atom.velocity_per_atom" in diagnostic_source
    assert "hac_->centroid_force_per_atom()" in diagnostic_source
    assert "hac_->centroid_potential_per_atom()" in diagnostic_source
    assert "force.compute(" not in diagnostic_source
    assert "TIME_UNIT_CONVERSION" in diagnostic_source
    assert "# columns step time_fs" in diagnostic_source
    assert "force.compute_hnemd_" in diagnostic_source
    assert "force.compute_hnemdec_" in diagnostic_source
    assert "num_target_pressure_components" in diagnostic_source
    assert "use_scr_barostat" in diagnostic_source
    assert "const double delta_max = std::isfinite(delta_rms)" in diagnostic_source
    assert (
        "const double delta_species_max = std::isfinite(delta_species_rms)"
        in diagnostic_source
    )
    for column in (
        "deltaF_rms",
        "deltaF_mean_abs",
        "deltaF_max",
        "Fbar_rms",
        "Fc_rms",
        "relative_deltaF_rms",
        "P_delta",
        "N_type",
        "K_centroid",
        "U_centroid",
        "E_centroid",
    ):
        assert column in diagnostic_source
    for species in ('"H"', '"O"', '"Na"', '"Cl"'):
        assert species in diagnostic_source

    end_of_step = measure_source[measure_source.index("void Measure::end_of_step") :]
    action_loop = end_of_step.index("for (auto& action : actions)")
    property_loop = end_of_step.index("for (auto& property : properties)", action_loop)
    assert property_loop > action_loop


def test_centroid_force_diagnostic_requires_an_immediate_centroid_source():
    source = (ROOT / "src/measure/centroid_force_diagnostic.cu").read_text(encoding="utf-8")
    docs = (ROOT / "doc/gpumd/input_parameters/centroid_force_diagnostic.rst").read_text(
        encoding="utf-8"
    )
    assert "PIMD/RPMD/TRPMD" in source
    assert "immediate centroid HAC" in source
    assert "deferred centroid" in source
    assert "PRINT_INPUT_ERROR" in source
    assert "ensemble rpmd 32 300" in docs
    assert "ensemble nve" not in docs


def test_centroid_force_diagnostic_rejects_force_buffer_overwriters():
    run_source = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf-8")
    for action_name in ("compute_es", "active", "dump_observer", "plumed"):
        assert action_name in run_source
    assert "centroid_force_diagnostic cannot be combined" in run_source


def test_centroid_force_diagnostic_reports_force_power_components():
    source = (ROOT / "src/measure/centroid_force_diagnostic.cu").read_text(
        encoding="utf-8"
    )

    constants = {
        name: int(value)
        for name, value in re.findall(
            r"constexpr int (DIAGNOSTIC_[A-Z_]+) = (\d+);", source
        )
    }
    assert constants["DIAGNOSTIC_POWER"] == 6
    assert constants["DIAGNOSTIC_KINETIC"] == 7
    assert constants["DIAGNOSTIC_POTENTIAL"] == 8
    assert constants["DIAGNOSTIC_POWER_FBAR"] == 9
    assert constants["DIAGNOSTIC_POWER_FC"] == 10
    assert constants["DIAGNOSTIC_STATISTICS"] == 11
    statistic_names = (
        "DIAGNOSTIC_COUNT",
        "DIAGNOSTIC_DELTA_SQUARED",
        "DIAGNOSTIC_DELTA_ABS",
        "DIAGNOSTIC_DELTA_MAX",
        "DIAGNOSTIC_FBAR_SQUARED",
        "DIAGNOSTIC_FC_SQUARED",
        "DIAGNOSTIC_POWER",
        "DIAGNOSTIC_KINETIC",
        "DIAGNOSTIC_POTENTIAL",
        "DIAGNOSTIC_POWER_FBAR",
        "DIAGNOSTIC_POWER_FC",
    )
    max_statistic_index = max(
        constants[name] for name in statistic_names
    )
    assert max_statistic_index < constants["DIAGNOSTIC_STATISTICS"]
    assert (
        "const double power_fbar = vc_x * fbar_x + vc_y * fbar_y + vc_z * fbar_z;"
        in source
    )
    assert (
        "const double power_fc = vc_x * fc_x + vc_y * fc_y + vc_z * fc_z;" in source
    )
    assert (
        "const double power = vc_x * delta_x + vc_y * delta_y + vc_z * delta_z;"
        in source
    )
    assert "s_statistics[DIAGNOSTIC_POWER_FBAR][tid] += power_fbar;" in source
    assert "s_statistics[DIAGNOSTIC_POWER_FC][tid] += power_fc;" in source
    assert source.count("P_Fbar[eV/fs]") == 1
    assert source.count("P_Fc[eV/fs]") == 1
    header_source = source[
        source.index("void Centroid_Force_Diagnostic::write_header_()") : source.index(
            "void Centroid_Force_Diagnostic::process"
        )
    ]
    assert (
        "K_centroid[eV] U_centroid[eV] E_centroid[eV] P_Fbar[eV/fs] P_Fc[eV/fs]"
        in header_source
    )
    assert header_source.count(
        "K_centroid[eV] U_centroid[eV] E_centroid[eV] P_Fbar[eV/fs] P_Fc[eV/fs]"
    ) == 1
    assert "%s_P_Fbar" not in header_source
    assert "%s_P_Fc" not in header_source
    assert 9 + 4 * 8 + 5 == 46

    row_source = source[source.index("void Centroid_Force_Diagnostic::write_row_") :]
    assert (
        "centroid_kinetic,\n"
        "    centroid_potential,\n"
        "    centroid_kinetic + centroid_potential,\n"
        "    global[DIAGNOSTIC_POWER_FBAR] / TIME_UNIT_CONVERSION,\n"
        "    global[DIAGNOSTIC_POWER_FC] / TIME_UNIT_CONVERSION);"
        in row_source
    )
    assert "global[DIAGNOSTIC_POWER_FBAR] / TIME_UNIT_CONVERSION" in source
    assert "global[DIAGNOSTIC_POWER_FC] / TIME_UNIT_CONVERSION" in source

    # Algebra sanity only; this does not exercise the CUDA reduction or output parsing.
    samples = [
        ((0.3, -0.2, 0.5), (1.1, -0.7, 0.2), (0.4, -0.1, 0.8)),
        ((-0.6, 0.4, 0.1), (0.2, 0.9, -0.3), (-0.5, 0.3, 0.7)),
    ]
    errors = []
    for velocity, fbar, fc in samples:
        p_delta = sum(v * (fb - c) for v, fb, c in zip(velocity, fbar, fc))
        p_fbar = sum(v * fb for v, fb in zip(velocity, fbar))
        p_fc = sum(v * c for v, c in zip(velocity, fc))
        errors.append(p_delta - (p_fbar - p_fc))
    max_error = max(abs(error) for error in errors)
    rms_error = (sum(error * error for error in errors) / len(errors)) ** 0.5
    assert max_error < 1.0e-15
    assert rms_error < 1.0e-15


def test_rpmd_parse_resets_pressure_control_state():
    source = (ROOT / "src/integrate/integrate.cu").read_text(encoding="utf-8")
    parse_start = source.index("void Integrate::parse_ensemble")
    setup_end = source.index("  // 1. Determine the integration method", parse_start)
    assert "num_target_pressure_components = 0;" in source[parse_start:setup_end]
