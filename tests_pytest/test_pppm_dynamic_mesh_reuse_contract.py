from pathlib import Path


ROOT = Path(__file__).parents[1]


def _function_body(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for position in range(brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise AssertionError(f"unclosed function: {signature}")


def _block_body(source, marker, occurrence=0):
    search_start = 0
    for _ in range(occurrence + 1):
        start = source.index(marker, search_start)
        search_start = start + len(marker)
    brace = source.index("{", start)
    depth = 0
    for position in range(brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : position + 1]
    raise AssertionError(f"unclosed block: {marker}")


def test_production_dynamic_q_reuses_current_force_mesh_only_on_a_valid_frame():
    header = (ROOT / "src/force/pppm.cuh").read_text(encoding="utf-8")
    source = (ROOT / "src/force/pppm.cu").read_text(encoding="utf-8")
    nep_source = (ROOT / "src/force/nep_charge.cu").read_text(encoding="utf-8")
    dynamic = _function_body(source, "bool PPPM::compute_dynamic_charge_correction(")
    force = _function_body(source, "void PPPM::find_force(")
    nep_dynamic = _function_body(
        nep_source, "bool NEP_Charge::compute_dynamic_charge_correction_impl("
    )
    fallback_forward = _block_body(dynamic, "if (!reuse_current_force_mesh) {")
    fallback_inverse = _block_body(dynamic, "if (!reuse_current_force_mesh) {", 1)

    assert "const unsigned long long force_evaluation_id = 0" in header
    assert "current_force_mesh_matches" in header
    assert "invalidate_current_force_mesh" in header
    assert "current_force_mesh_valid_ = false;" in force
    assert "current_force_mesh_valid_ = true;" in force
    assert "current_force_mesh_matches" in dynamic
    assert "reuse_current_force_mesh" in dynamic
    assert "N1 != 0 || N2 != N" in source
    assert "reuse_current_force_mesh ? nullptr : dynamic_Q_.data()" in dynamic
    assert "reuse_current_force_mesh ? mesh.data() : dynamic_Q_.data()" in dynamic
    assert "reuse_current_force_mesh ? mesh_G.data() : dynamic_Q_.data()" in dynamic
    assert "if (!reuse_current_force_mesh)" in dynamic
    assert "dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_FORWARD" in fallback_forward
    assert "dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_INVERSE" in fallback_inverse
    assert fallback_forward.index("GPUFFT_FORWARD") < dynamic.index("reduce_dynamic_mesh_current")
    assert dynamic.index("dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_INVERSE") < dynamic.index(
        "dynamic_S_.data(), dynamic_S_.data(), GPUFFT_INVERSE"
    )
    assert "dynamic_q_cache_force_evaluation_id_ == force_evaluation_id_" in nep_dynamic
    assert "dynamic_q_cache_charge_rate_generation_ == charge_rate_generation_" in nep_dynamic
    assert "position.data() == dynamic_q_cache_position_" in nep_dynamic
    assert "pppm.invalidate_current_force_mesh();" in nep_source


def test_dynamic_mesh_kernel_can_skip_q_deposition_without_changing_diagnostic_path():
    source = (ROOT / "src/force/pppm.cu").read_text(encoding="utf-8")
    kernel = _function_body(source, "__global__ void find_dynamic_mesh(")
    production = _function_body(source, "bool PPPM::compute_dynamic_charge_correction(")
    diagnostic = _function_body(source, "bool PPPM::diagnose_dynamic_charge(")

    assert "if (g_Q != nullptr)" in kernel
    assert "find_dynamic_mesh" in production
    assert "find_dynamic_mesh" in diagnostic
    assert "dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_FORWARD" in production
    assert "dynamic_Q_.data(), dynamic_Q_.data(), GPUFFT_INVERSE" in production
    assert "Q.data(), Q.data(), GPUFFT_FORWARD" in diagnostic


def test_pppm_merge_preserves_spacing_virial_and_fft_lifecycle_contracts():
    pppm_header = (ROOT / "src/force/pppm.cuh").read_text(encoding="utf-8")
    pppm_source = (ROOT / "src/force/pppm.cu").read_text(encoding="utf-8")
    nep_source = (ROOT / "src/force/nep_charge.cu").read_text(encoding="utf-8")
    nep_header = (ROOT / "src/force/nep_charge.cuh").read_text(encoding="utf-8")
    force_source = (ROOT / "src/force/force.cu").read_text(encoding="utf-8")
    force_header = (ROOT / "src/force/force.cuh").read_text(encoding="utf-8")
    find_para = _function_body(pppm_source, "void PPPM::find_para(")
    parse_potential = _function_body(force_source, "void Force::parse_potential(")
    kspace_parser = _function_body(nep_source, "void NEP_Charge::check_ewald_pppm(")

    assert "const bool need_peratom_virial_every_batch_input,\n    const double mesh_spacing_input" in pppm_header
    assert "Para para = {};" in pppm_header
    assert "pppm.initialize(\n      charge_para.alpha,\n      virial_requirements.anywhere,\n      virial_requirements.every_batch,\n      pppm_spacing);" in nep_source
    assert "tokens.size() < 2 || tokens.size() > 3" in kspace_parser
    assert "kspace ewald does not accept spacing" in kspace_parser
    assert "tokens.size() == 3" in kspace_parser
    assert "!is_valid_real(tokens[2], &pppm_spacing)" in kspace_parser
    assert "pppm_spacing >= 0.2 && pppm_spacing <= 2.0" in kspace_parser
    assert "bool use_pppm = true;" in nep_header
    assert "bool pppm_mesh_spacing_explicit_ = false;" in force_header
    assert "pppm_mesh_spacing_explicit_ = true;" in force_source
    assert "if (!pppm_mesh_spacing_explicit_)" in parse_potential
    assert "qnep->get_pppm_mesh_spacing()" in parse_potential
    assert parse_potential.index("qnep->get_pppm_mesh_spacing()") < parse_potential.index(
        "potential->set_pppm_mesh_spacing(pppm_mesh_spacing_);"
    )
    assert "void PPPM::destroy_plans()" in pppm_source
    assert "plan_batch" in _function_body(pppm_source, "void PPPM::destroy_plans()")
    assert "plan_inverse_batch" in _function_body(pppm_source, "void PPPM::destroy_plans()")
    assert "plan_virial_batch" in _function_body(pppm_source, "void PPPM::destroy_plans()")
    assert "batch_capacity = 0;" in _function_body(pppm_source, "void PPPM::destroy_plans()")
    assert "is_good_K" in pppm_source
    assert "if (required > para.K[d])" in find_para
    assert "if (number_of_points > max_mesh_points)" in find_para
    assert find_para.index("if (number_of_points > max_mesh_points)") < find_para.index(
        "static_cast<int>(number_of_points)"
    )
    assert "current_force_mesh_valid_ = false;" in find_para
