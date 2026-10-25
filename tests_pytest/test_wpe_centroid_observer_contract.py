from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_wpe_pimd_centroid_hac_uses_an_isolated_native_observer():
    force_source = (ROOT / "src/force/force.cu").read_text(encoding="utf-8")
    force_start = force_source.index("void Force::parse_potential(")
    wpe_start = force_source.index("#ifdef GPUMD_WPE_ENABLED", force_start)
    force_factory = force_source[force_start : force_source.index("#else", wpe_start)]
    assert "number_of_atoms_ = number_of_atoms;" in force_factory
    assert "run_input_ = &run_input;" in force_factory
    assert "primary_nep_model_path_ =" not in force_factory

    source = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
    assert '#include "utilities/compact_nep.cuh"' in source
    pre_run = source[source.index("void HAC::pre_run(") : source.index("static __global__ void")]
    assert "wpe_centroid_observer_.reset();" in pre_run
    centroid_start = pre_run.index(
        "if (use_centroid_heat_flux_) {", pre_run.index("heat_all.resize(")
    )
    centroid_end = pre_run.index("if (split_qnep_heat_by_type_)", centroid_start)
    centroid_setup = pre_run[centroid_start:centroid_end]
    init_start = centroid_setup.index("if (is_ring_polymer_run)")
    init_end = centroid_setup.index("#endif", init_start)
    init = centroid_setup[init_start:init_end]
    assert centroid_start < pre_run.index(init) < centroid_end
    assert "split_qnep_heat_by_type_ != 0" in init
    assert 'model_type != "nep4" && model_type != "nep4_zbl"' in init
    for model in (
        "nep4_charge1",
        "nep4_zbl_charge1",
        "nep4_charge2",
        "nep4_zbl_charge2",
    ):
        assert f'model_type == "{model}"' in init
    assert "get_wpe_stage_snapshot()" in init
    assert "stage.qnep_kspace_method" in init
    assert "stage.qnep_pppm_spacing" in init
    assert "configure_mechanical_observer()" in init
    assert "set_pppm_mesh_spacing(pppm_spacing)" in init
    assert 'find_first("kspace")' not in init
    assert 'find_last("pppm_mesh_spacing")' not in init
    assert 'observer_lines{"potential " + model_path}' in init
    assert "stage.dftd3_functional" in init
    assert 'WPE centroid HAC uses a private native %s observer.' in init

    start = source.index("} else if (wpe_centroid_observer_) {")
    end = source.index("#endif", start)
    observer = source[start:end]
    order = (
        "box.set_is_orthogonal()",
        "rpmd_ja_wrap_positions(",
        "centroid_qnep_observer_error_.copy_to_host",
        "centroid_potential_per_atom_.fill(0.0)",
        "centroid_force_per_atom_.fill(0.0)",
        "centroid_virial_per_atom_.fill(0.0)",
        "request_peratom_virial_for_next_force()",
        "wpe_centroid_observer_->compute(",
    )
    assert [observer.index(item) for item in order] == sorted(
        observer.index(item) for item in order
    )
    compute_args = observer.split("wpe_centroid_observer_->compute(", 1)[1].split(
        ");", 1
    )[0]
    for arg in (
        "box",
        "atom.type",
        "centroid_position_work_",
        "centroid_potential_per_atom_",
        "centroid_force_per_atom_",
        "centroid_virial_per_atom_",
    ):
        assert arg in compute_args
    assert "force.compute(" not in observer
    assert "WPE's PIMD context must receive only physical bead evaluations." in observer
    assert "WPE PIMD centroid HAC does not support potential model " in init
    assert "PRINT_INPUT_ERROR(error.c_str());" in init


def test_wpe_rpmd_ja_reuses_centroid_observer_for_the_sampled_current():
    force = (ROOT / "src/force/force.cu").read_text(encoding="utf-8")
    force_header = (ROOT / "src/force/force.cuh").read_text(encoding="utf-8")
    assert "WpeStageSnapshot" in force_header
    assert "wpe_stage_snapshot_" in force
    assert "tokens[0] == \"run\"" in force
    assert "wpe_stage_snapshot_.qnep_pppm_spacing = stage.qnep_pppm_spacing" in force
    assert "wpe_stage_snapshot_.dftd3_functional =" in force
    assert "wpe_stage_snapshot_.dftd3_rc_potential = stage.dftd3_rc_potential" in force

    hac = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
    pre_run = hac[hac.index("void HAC::pre_run(") : hac.index("static __global__ void")]
    setup_start = pre_run.index("#ifdef GPUMD_WPE_ENABLED", pre_run.index("if (use_centroid_heat_flux_)"))
    setup_end = pre_run.index("#endif", setup_start)
    setup = pre_run[setup_start:setup_end]
    assert "if (is_ring_polymer_run)" in setup
    assert "!rpmd_ja_enabled_" not in setup
    assert "wpe_centroid_model_path_" in setup
    assert setup.index("wpe_centroid_observer_ = std::move(observer)") < pre_run.index(
        "pre_run_rpmd_ja_(number_of_frames"
    )

    sample_start = hac.index("if (rpmd_ja_enabled_ &&", hac.index("void HAC::pre_force("))
    sample_end = hac.index("centroid_force_step_ = step + 1;", sample_start)
    sample_path = hac[sample_start:sample_end]
    native_observer_guard = sample_path[sample_path.index("if (rpmd_ja_enabled_ &&") :]
    assert "!wpe_centroid_observer_" in native_observer_guard
    assert sample_path.count("wpe_centroid_observer_->compute(") == 1
    assert "atom.force_beads" not in sample_path
    wpe_start = sample_path.index("} else if (wpe_centroid_observer_) {")
    wpe_end = sample_path.index("#endif", wpe_start)
    assert "force.compute(" not in sample_path[wpe_start:wpe_end]

    ja = (ROOT / "src/measure/rpmd_ja.cu").read_text(encoding="utf-8")
    pre_run_ja = ja[ja.index("void HAC::pre_run_rpmd_ja_(") : ja.index("void HAC::update_rpmd_ja_centroid_(")]
    assert "wpe_centroid_observer_" in pre_run_ja
    assert "wpe_centroid_model_path_" in pre_run_ja
    assert "get_wpe_stage_snapshot()" in pre_run_ja
    assert "qnep_kspace_method" in pre_run_ja
    assert "qnep_pppm_spacing" in pre_run_ja
    assert "stage.number_of_atoms != atom.number_of_atoms" in pre_run_ja
    assert "stage.pimd_beads != atom.number_of_beads" in pre_run_ja
    assert "WPE_STAGE_RUN" in pre_run_ja
    assert "WPE_STAGE_PIMD_ECO_CONFIGURATION" in pre_run_ja
    assert "WPE_QNEP_KSPACE_PPPM" in pre_run_ja
    assert "wpe_stage.dftd3_enabled != 0" in pre_run_ja
    assert "rpmd_ja WPE qNEP PPPM settings do not match the active stage." in pre_run_ja
    assert "rpmd_ja_model_fingerprint(model_path)" in pre_run_ja
    assert "rpmd_ja_qnep_config_fingerprint(" in pre_run_ja
    assert "wpe_stage.dftd3_enabled != 0" in pre_run_ja
    assert "rpmd_ja_reference_.mechanical_config_fingerprint" in pre_run_ja
    for identity_check in (
        "rpmd_ja_reference_.model_fingerprint",
        "rpmd_ja_reference_.q_charge_mode != active_qnep->get_charge_mode()",
        "rpmd_ja_reference_.q_mesh_spacing - active_mesh_spacing",
        "rpmd_ja_reference_policy_beads(rpmd_ja_reference_.mechanical_policy)",
        "rpmd_ja_reference_policy_requires_pimd_fix_com",
        "rpmd_ja_reference_policy_use_matches",
    ):
        assert identity_check in pre_run_ja


def test_wpe_sample_uses_bead_forces_and_records_model_identity_without_enabling_fit():
    run = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf-8")
    assert run.index("compute_force();", run.index("for (int step")) < run.index(
        "measure.post_force(step", run.index("for (int step")
    )

    sampler = (ROOT / "src/measure/rpmd_ja_fit.cu").read_text(encoding="utf-8")
    post_force = sampler[sampler.index("void RpmdJA_Fit::post_force(") : sampler.index("void RpmdJA_Fit::post_run(")]
    sampler_header = (ROOT / "src/measure/rpmd_ja_fit.cuh").read_text(encoding="utf-8")
    assert "atom.position_beads[bead].data()" in post_force
    assert "atom.force_beads[bead].data()" in post_force
    assert "mean_force_" not in sampler and "mean_force_" not in sampler_header
    assert "!std::isfinite(sample_output_[coordinates + i + d * number_of_atoms_])" in post_force
    assert "std::copy(sample_output_.begin() + coordinates, sample_output_.begin() + 2 * coordinates,\n    frame_buffer_.begin() + 1 + coordinates);" in post_force
    sampler_pre_run = sampler[sampler.index("void RpmdJA_Fit::pre_run(") : sampler.index("void RpmdJA_Fit::post_force(")]
    assert "frame_buffer_.resize(1 + 2 * coordinates);" in sampler_pre_run
    post_run = sampler[sampler.index("void RpmdJA_Fit::post_run(") :]
    assert post_run.index("if (sample_only_) {") < post_run.index("fit_rpmd_ja_native_reference(")

    force = (ROOT / "src/force/force.cu").read_text(encoding="utf-8")
    bead_force = force[force.index("void Force::compute_pimd_beads(") :]
    assert "force_beads[k]" in bead_force
    assert "virial_beads[k]" in bead_force

    provenance = (ROOT / "src/measure/rpmd_ja_sample_provenance.h").read_text(encoding="utf-8")
    assert '#include "utilities/compact_nep.cuh"' in provenance
    assert "get_first_potential_filename(input)" in provenance
    assert "result.model_sha256.empty()" in provenance
    assert "get_wpe_stage_snapshot()" in provenance
    assert "wpe_qnep_kspace_method" in provenance
    assert "wpe_qnep_pppm_spacing" in provenance
    assert "wpe_dftd3_functional" in provenance
    assert "physical_force_backend" in sampler

    fit = sampler[sampler.index("void RpmdJA_Fit::pre_run(") : sampler.index("void RpmdJA_Fit::post_force(")]
    assert "if (!sample_only_)" in fit
    assert "WPE supports rpmd_ja sample only; native fit/reference construction is unavailable." in fit
    assert "get_wpe_stage_snapshot()" in fit
    assert "stage.number_of_atoms != atom.number_of_atoms" in fit
    assert "stage.pimd_beads != atom.number_of_beads" in fit
    assert fit.index("number_of_atoms_ = atom.number_of_atoms;") < fit.index(
        "rpmd_ja_sample_provenance::snapshot("
    )
    assert fit.index("number_of_beads_ = atom.number_of_beads;") < fit.index(
        "rpmd_ja_sample_provenance::snapshot("
    )

    run_source = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf-8")
    run_start = run_source.index("void Run::perform_a_run(")
    entry = run_source[run_start : run_source.index("if (is_pimd(integrate.get_type()))", run_start)]
    assert "const unsigned current_run_ordinal = run_ordinal_++;" in run_source[run_start:]
    assert "rpmd_ja_fit->set_run_ordinal(current_run_ordinal);" in entry
    assert "const bool sample_only = rpmd_ja_fit->sample_only();" in entry
    assert "if (!sample_only)" in entry
    assert "WPE supports rpmd_ja sample only; native fit/reference construction is unavailable." in entry
    assert "get_wpe_stage_snapshot()" in entry
    assert "WPE_QNEP_KSPACE_PPPM" in entry
    assert "charge2" in entry and "charge1" in entry
    assert "stage.pimd_beads != integrate.get_number_of_beads()" in entry
    assert "stage.pimd_beads != atom.number_of_beads" not in entry
    assert run_source.index("stage.pimd_beads != integrate.get_number_of_beads()", run_start) < run_source.index(
        "integrate.initialize(time_step", run_start
    )
    wpe_entry = entry[entry.index("#ifdef GPUMD_WPE_ENABLED") : entry.index("#else", entry.index("#ifdef GPUMD_WPE_ENABLED"))]
    native_entry = entry[entry.index("#else", entry.index("#ifdef GPUMD_WPE_ENABLED")) :]
    assert "dynamic_cast<NEP_Charge" not in wpe_entry
    assert "rpmd_ja_qnep_config_fingerprint(force)" in native_entry
    assert 'action->modifies_force()' in entry
    assert 'name == "compute_hnemd"' in entry
    assert 'name == "plumed"' in entry
    wpe_end = entry.index("#endif", entry.index("#ifdef GPUMD_WPE_ENABLED"))
    assert wpe_end < entry.index("for (const auto& action : actions)", wpe_end)

    runtime_check = (ROOT / "tests/rpmd_ja_qnep_end_to_end_test.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--wpe-gpumd"' in runtime_check
    assert "qnep_species = model_tokens[2 : 2 + qnep_type_count]" in runtime_check
    assert "missing_qnep_species" in runtime_check
    assert "model-xyz must contain all qNEP model species for exact model SHA-256 verification" in runtime_check
    assert 'expected_run_ordinal: int = 0' in runtime_check
    assert 'metadata["active_run_ordinal"] != expected_run_ordinal' in runtime_check
    assert 'metadata["thermostat_configuration_tokens"] != [expected_ensemble]' in runtime_check
    assert 'f"rpmd_ja sample train {args.sample_interval}"' in runtime_check
    assert '"wpe-fit-rejected"' in runtime_check
    assert '"wpe-sample-drive-rejected"' in runtime_check
    assert "run_expect_failure" in runtime_check
    assert "configuration_stable_during_run" in runtime_check
    assert 'f"velocity {args.temperature:.17g} seed {args.seed}"' in runtime_check
    assert runtime_check.index('f"velocity {args.temperature:.17g} seed {args.seed}"') < runtime_check.index(
        '"read_pimd_restart fixture_initial_restart.xyz"',
        runtime_check.index('directory = root / f"{backend.lower()}-sample"'),
    )
    assert 'directory = root / "wpe-sample-first-pimd"' in runtime_check
    startup_sample = runtime_check[
        runtime_check.index('directory = root / "wpe-sample-first-pimd"') :
        runtime_check.index('for case, balance_steps in (')
    ]
    assert "read_pimd_restart" not in startup_sample
    assert "check_sample_output(" in startup_sample
    assert '"same-run-length"' in runtime_check
    assert '"different-run-length"' in runtime_check
    assert "expected_run_ordinal=1" in runtime_check
    assert '"compute_hnemd 1 1e-5 0 0"' in runtime_check
    assert '"rpmd_ja sample does not support external drive/action compute_hnemd"' in runtime_check

    provenance = (ROOT / "src/measure/rpmd_ja_sample_provenance.h").read_text(encoding="utf-8")
    assert "effective_thermostat_configuration_lines(active_lines)" in provenance
    assert "result.active_run_ordinal=run_ordinal" in provenance

