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


def _qnep_full_a_branch(source):
    start = source.index("if (qnep_full_a_) {", source.index("void HAC::end_of_step"))
    end = source.index("\n  if ((step + 1) % sample_interval != 0)", start)
    return source[start:end]


def test_qnep_full_a_centroid_uses_one_isolated_immediate_force_and_shared_assembly():
    source = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
    branch = _qnep_full_a_branch(source)
    force_position = branch.index("force.compute(")

    assert "qnep_full_a + deferred centroid is not implemented" in source
    assert "const bool centroid_qnep_full_a = use_centroid_heat_flux_ != 0;" in source
    assert "centroid_position_work_.resize" in source
    assert "centroid_charge_backup_.resize" in source
    assert "centroid_bec_backup_.resize" in source
    assert branch.count("force.compute(") == 1
    assert "const GPU_Vector<double>* current_position = &atom.position_per_atom;" in branch
    assert "const GPU_Vector<double>* current_unwrapped_position = &atom.unwrapped_position;" in branch
    assert "centroid_position_work_.copy_from_device" in branch
    assert "centroid_charge_backup_.copy_from_device" in branch
    assert "centroid_bec_backup_.copy_from_device" in branch
    assert "current_position = &centroid_position_work_;" in branch
    assert "qnep_full_a_qnep_->request_charge_diagnostics_for_next_force();" in branch
    assert "qnep_full_a_qnep_->request_peratom_virial_for_next_force();" in branch
    assert branch.index("centroid_position_work_.copy_from_device") < force_position
    assert branch.index("centroid_charge_backup_.copy_from_device") < force_position
    assert branch.index("qnep_full_a_qnep_->request_charge_diagnostics_for_next_force();") < force_position
    assert "centroid_position_work_" in branch[force_position : force_position + 700]
    assert "centroid_potential_per_atom_" in branch[force_position : force_position + 700]
    assert "centroid_virial_per_atom_" in branch[force_position : force_position + 700]
    restore_position = branch.index("get_charge_reference().copy_from_device")
    assert "centroid_charge_backup_.data()" in branch[restore_position : restore_position + 120]
    assert "get_bec_reference().copy_from_device(centroid_bec_backup_.data())" in branch
    assert restore_position < branch.index("invalidate_current_force_caches()")
    assert "mark_single_frame_neighbor_reference_pending()" in branch
    assert "invalidate_single_frame_neighbor_reference();" not in branch
    assert "qnep_full_a_current_history_[nd + Nd * d] = j_full_a[d];" in branch


def test_centroid_contract_detects_removing_the_centroid_force_call():
    source = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
    branch = _qnep_full_a_branch(source)
    assert "force.compute(" in branch
    removed_force = branch.replace("force.compute(", "force.compute_disabled(", 1)
    assert "force.compute(" not in removed_force


def test_qnep_force_generation_covers_single_and_batch_force_entries():
    source = (ROOT / "src/force/nep_charge.cu").read_text(encoding="utf-8")
    single = _function_body(source, "void NEP_Charge::compute(")
    batch = _function_body(source, "bool NEP_Charge::compute_pimd_batch(")
    assert "void NEP_Charge::begin_force_evaluation_()" in source
    assert "begin_force_evaluation_();" in single
    assert "begin_force_evaluation_();" in batch

    header = (ROOT / "src/force/nep_charge.cuh").read_text(encoding="utf-8")
    assert "has_charge_diagnostics_for_current_force_frame" in header
    assert "mark_single_frame_neighbor_reference_pending" in header
    assert "consume_single_frame_neighbor_reference_invalidation" in header
    assert "invalidate_current_force_caches" in header
    assert "md_qnep_bec_enabled() const" in header
    assert "invalidate_reference_positions" in (
        ROOT / "src/force/neighbor.cuh"
    ).read_text(encoding="utf-8")


def test_serial_bead_fallback_consumes_auxiliary_neighbor_invalidation_after_batch_attempts():
    source = (ROOT / "src/force/force.cu").read_text(encoding="utf-8")
    body = _function_body(source, "void Force::compute_pimd_beads(")
    consume = body.index("consume_single_frame_neighbor_reference_invalidation")
    serial_loop = body.index("for (int k = 0; k < position_beads.size(); ++k)")
    assert body.index("try_compute_pimd_qnep_batch_") < consume
    assert body.index("try_compute_pimd_nep_batch_") < consume
    assert consume < serial_loop


def test_full_a_assembly_rejects_stale_diagnostics():
    source = (ROOT / "src/measure/qnep_projection.cu").read_text(encoding="utf-8")
    assert "has_charge_diagnostics_for_current_force_frame" in source
