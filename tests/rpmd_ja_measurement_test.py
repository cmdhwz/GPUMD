from pathlib import Path

import numpy as np


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def bilinear(x, matrix, velocity):
    return dot(x, matvec(matrix, velocity))


def unwrap(raw, previous, box):
    delta = raw - previous
    if delta > box / 2:
        delta -= box
    elif delta < -box / 2:
        delta += box
    return previous + delta


def check_hac_decomposition():
    centroid = np.array(
        [[3.0, 5.0, -8.0, 5.0, -1.0, 0.0], [2.0, -4.0, 8.0, -8.0, -4.0, -2.0], [1.0, -2.0, -6.0, -8.0, -8.0, -8.0]]
    )
    delta = np.array(
        [[-6.0, 8.0, -5.0, 3.0, 4.0, -5.0], [-4.0, -1.0, -4.0, 8.0, -5.0, 7.0], [5.0, 6.0, -7.0, -2.0, 2.0, 0.0]]
    )
    number_of_frames = centroid.shape[1]
    ja = centroid + delta

    def correlation(left, right, lag):
        return np.correlate(left, right, mode="full")[number_of_frames - 1 + lag] / (
            number_of_frames - lag
        )

    for lag in range(4):
        c_cc = np.array([correlation(c, c, lag) for c in centroid])
        c_cdelta = np.array([correlation(d, c, lag) for c, d in zip(centroid, delta)])
        c_deltac = np.array([correlation(c, d, lag) for c, d in zip(centroid, delta)])
        c_deltadelta = np.array([correlation(d, d, lag) for d in delta])
        c_ja = np.array([correlation(j, j, lag) for j in ja])
        assert np.allclose(c_ja, c_cc + c_cdelta + c_deltac + c_deltadelta)

    cross_forward = np.array(
        [[correlation(d, c, lag) for lag in range(1, 4)] for c, d in zip(centroid, delta)]
    )
    cross_reverse = np.array(
        [[correlation(c, d, lag) for lag in range(1, 4)] for c, d in zip(centroid, delta)]
    )
    assert np.all(np.abs(cross_forward - cross_reverse) > 1.0e-12)


def demo():
    check_hac_decomposition()
    # A non-symmetric matrix catches accidental transposition of DeltaH.
    matrix = [[0.0, 2.0], [-3.0, 1.0]]
    x, velocity = [2.0, -1.0], [4.0, 5.0]
    value = bilinear(x, matrix, velocity)
    assert value == 27.0
    assert value != bilinear(x, [list(row) for row in zip(*matrix)], velocity)
    assert bilinear(x, [[0.0, 0.0], [0.0, 0.0]], velocity) == 0.0

    # Wrapped centroids crossing a face remain continuous on the tracked branch.
    tracked = unwrap(0.1, 9.8, 10.0)
    tracked = unwrap(0.4, tracked, 10.0)
    assert abs(tracked - 10.4) < 1.0e-12

    # Autocorrelation of Jcent + DeltaJ contains both cross terms.
    jcent, delta = [1.0, 2.0], [3.0, -1.0]
    ja = [a + b for a, b in zip(jcent, delta)]
    expanded = sum(
        left * right
        for left, right in zip(jcent, jcent)
    ) + sum(left * right for left, right in zip(jcent, delta))
    expanded += sum(left * right for left, right in zip(delta, jcent))
    expanded += sum(left * right for left, right in zip(delta, delta))
    assert sum(value * value for value in ja) == expanded

    root = Path(__file__).resolve().parents[1]
    hac = (root / "src/measure/hac.cu").read_text(encoding="utf-8")
    hac_header = (root / "src/measure/hac.cuh").read_text(encoding="utf-8")
    qnep_source = (root / "src/force/nep_charge.cu").read_text(encoding="utf-8")
    ja_source = (root / "src/measure/rpmd_ja.cu").read_text(encoding="utf-8")
    run_source = (root / "src/main_gpumd/run.cu").read_text(encoding="utf-8")
    ja_doc = (root / "doc/gpumd/input_parameters/rpmd_ja.rst").read_text(encoding="utf-8")
    assert "compute_rpmd_ja_current_(nd, atom, box)" in hac
    assert "gpu_find_hac_3<<<Nc, 128>>>(Nc, Nd, rpmd_ja_current_[2].data(), hac_gpu.data())" in hac
    sparse = (root / "src/measure/rpmd_ja_sparse.cu").read_text(encoding="utf-8")
    assert "static __global__ void compute_hac" not in ja_source
    assert "q_accumulators[static_cast<std::size_t>(rank) * D + i]" in sparse
    assert "b_results[static_cast<std::size_t>(p_rank + rank) * D + i]" in sparse
    assert "HAC_x HAC_y HAC_z RTC_x RTC_y RTC_z" in hac
    assert "rpmd_ja_integrate_hac(Nc, number_of_components, factor, hac, rtc);" in hac
    assert "centroid_potential_per_atom_.fill(0.0)" in hac
    assert "centroid_force_per_atom_.fill(0.0)" in hac
    assert "centroid_virial_per_atom_.fill(0.0)" in hac
    assert "std::unique_ptr<NEP_Charge> centroid_qnep_observer_" in hac_header
    assert "split_qnep_heat_by_type_ == 0" in hac
    assert "deferred_centroid_qnep_ == 0" in hac
    assert "centroid_qnep_observer_->configure_mechanical_observer()" in hac
    assert "centroid_qnep_observer_->request_peratom_virial_for_next_force()" in hac
    assert "centroid_qnep_observer_->compute(" in hac
    assert "# centroid_operator qnep_native_full_mechanical" in hac
    assert "# charge_snapshot q_of_centroid_configuration" in hac
    assert "# additional_qnep_full_a_correction 0" in hac
    assert 'rpmd_ja_reference_.backend != 2' in ja_source
    assert 'rpmd_ja_reference_.mechanical_policy != "native_reference_transport"' in ja_source
    assert "rpmd_ja_reference_policy_requires_pimd_fix_com" in ja_source
    assert "!integrate.get_pimd_fix_com()" in ja_source
    assert 'rpmd_ja_qnep_config_fingerprint(force)' in ja_source
    assert 'generate_rpmd_ja_qnep_reference(' in run_source
    assert 'rpmd_ja_reference_.backend == 2' in ja_source
    block_release = ja_source.split("rpmd_ja_sparse_workspace_.initialize", 1)[1].split(
        "if (rpmd_ja_reference_.backend == 1)", 1
    )[0]
    assert 'if (rpmd_ja_reference_.backend == 2)' in block_release
    assert 'std::vector<RpmdJAMatrixTile>().swap(matrix.tiles)' in block_release
    assert 'rpmd_ja_sparse_workspace_.dynamical_block_count()' in ja_source
    assert 'rpmd_ja_sparse_workspace_.site_block_count(0)' in ja_source
    assert 'native_reference_transport' in ja_doc
    assert 'if (rpmd_ja_enabled_ && !centroid_qnep_observer_)' in hac
    assert 'rpmd_ja_enabled_ && !qnep_full_a_ && split_qnep_heat_by_type_ == 0' in hac
    assert 'block_apply_right<<<' in sparse
    assert 'block_apply_rows<<<' in sparse
    assert 'qnep_cached_.initialize(reference, masses)' in sparse
    assert 'mechanical_config_fingerprint' in hac
    observer_body = hac.split("} else if (centroid_qnep_observer_) {", 1)[1].split("} else {", 1)[0]
    compute_at = observer_body.index("centroid_qnep_observer_->compute(")
    assert observer_body.index("centroid_potential_per_atom_.fill(0.0)") < compute_at
    assert observer_body.index("centroid_force_per_atom_.fill(0.0)") < compute_at
    assert observer_body.index("centroid_virial_per_atom_.fill(0.0)") < compute_at
    assert observer_body.index("rpmd_ja_wrap_positions(") < compute_at
    assert "atom.position_per_atom" in observer_body
    observer_config = qnep_source.split("void NEP_Charge::configure_mechanical_observer()", 1)[1].split("void NEP_Charge::enable_charge_diagnostics()", 1)[0]
    assert "need_bec = false" in observer_config
    assert "md_qnep_bec_enabled_ = false" in observer_config
    assert "pimd_batch_bec_enabled_ = false" in observer_config
    cli_test = (root / "tests/rpmd_ja_qnep_centroid_cli_test.py").read_text(encoding="utf-8")
    assert "--baseline-gpumd" in cli_test and "--candidate-gpumd" in cli_test
    assert '"heat_current_centroid.out", "hac_centroid.out"' in cli_test
    assert "for (int d = 0; d < 3; ++d)" in ja_source
    assert "branch_limit = 0.45" in ja_source
    current_body = ja_source.split("void HAC::compute_rpmd_ja_current_", 1)[1]
    assert current_body.index("store_current<<<") < current_body.index("copy_to_host")
    assert current_body.count("copy_to_host") == 1
    assert "RPMD_JA_ERROR_BRANCH" in current_body
    assert "RPMD_JA_ERROR_NUMERICAL" in current_body
    assert "atomicOr(error, RPMD_JA_ERROR_NUMERICAL)" in sparse
    assert "check_sparse_finite" not in sparse
    assert "step + 1 == number_of_steps && (step + 1) % sample_interval != 0" in hac


if __name__ == "__main__":
    demo()
