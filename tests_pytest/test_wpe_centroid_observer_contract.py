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
    init_start = centroid_setup.index("if (is_ring_polymer_run && !rpmd_ja_enabled_)")
    init_end = centroid_setup.index("#endif", init_start)
    init = centroid_setup[init_start:init_end]
    assert centroid_start < pre_run.index(init) < centroid_end
    assert "split_qnep_heat_by_type_ != 0" in init
    assert 'model_type == "nep4" || model_type == "nep4_zbl"' in init
    for model in (
        "nep4_charge1",
        "nep4_zbl_charge1",
        "nep4_charge2",
        "nep4_zbl_charge2",
    ):
        assert f'model_type == "{model}"' in init
    assert "force.get_run_input()" in init
    assert "configure_mechanical_observer()" in init
    assert "set_pppm_mesh_spacing" not in init
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

