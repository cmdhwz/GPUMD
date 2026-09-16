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
