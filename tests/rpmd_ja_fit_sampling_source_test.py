from pathlib import Path
from math import isclose


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


run = source("src/main_gpumd/run.cu")
measure = source("src/measure/measure.cu")
sampler = source("src/measure/rpmd_ja_fit.cu")
sampling = source("src/measure/rpmd_ja_fit_sampling.cuh")
native_header = source("src/measure/rpmd_ja_native_fit.cuh")
native_fit = source("src/measure/rpmd_ja_native_fit.cu")
sampler_fit = source("src/measure/rpmd_ja_fit.cu")

assert 'if (tokens[1] == "fit")' in run
assert "tokens, number_of_types, integrate, group, atom, box, force, first_potential_filename_" in run
assert "rpmd_ja fit requires a fixed integration time step" in run
assert "rpmd_ja fit currently supports exactly one qNEP potential in charge mode 1 or 2 with PPPM" in run
assert 'tokens[1] == "fit"' in measure and "new RpmdJA_Fit(tokens)" in measure

force_call = run.index("compute_force();", run.index("for (int step = 0; step < number_of_steps"))
sample_call = run.index("measure.post_force(", force_call)
integrate2_call = run.index("integrate.compute2(", sample_call)
assert force_call < sample_call < integrate2_call
assert "if ((step + 1) % sample_interval_ != 0) return;" in sampler
post_force_start = sampler.index("void RpmdJA_Fit::post_force(")
post_run_start = sampler.index("void RpmdJA_Fit::post_run(", post_force_start)
post_force = sampler[post_force_start:post_run_start]
assert post_force.index("if ((step + 1) % sample_interval_ != 0) return;") < post_force.index("rpmd_ja_fit_sampling_kernel<<<")
assert "sample_output_gpu_.copy_to_host(sample_output_.data());" in post_force
assert post_force.count("copy_to_host(") == 1
assert "position != bead_position_ptrs_[bead] || force != bead_force_ptrs_[bead]" in post_force
assert "if (refresh_pointers)" in post_force
assert "sample_output_.resize(static_cast<std::size_t>(10) * number_of_atoms_);" in sampler
assert "coordinate_scale = std::max(coordinate_scale, sample_output_[6 * atom_count + atom_id]);" in post_force
assert "const double ring_tolerance = 128.0 * std::numeric_limits<double>::epsilon()" in post_force
assert "coordinate_scale > 1.0e12 * box_scale" in post_force
assert "previous_centroid_ = centroid_;" in post_force
assert "write_or_throw(spool_, frame_buffer_.data(), frame_buffer_.size() * sizeof(double));" in post_force
assert "bead_position_." not in sampler and "bead_force_." not in sampler

assert "'G', 'P', 'J', 'A', 'S', 'M', 'P', '1'" in sampler
assert "ring path with nonzero periodic winding" in sampler
assert "ring-unwrapped centroid differs from the bead-0 MIC centroid" in sampler
assert "previous_centroid_ = centroid_;" in sampler
assert "atom.number_of_atoms > INT_MAX / 3" in sampler
assert "pbc_[0] != 1 || pbc_[1] != 1 || pbc_[2] != 1" in sampler
assert "size_stream.close();" in sampler
assert "stream limits" in sampler
assert "fit_rpmd_ja_native_reference(options, spool_path_, frame_count_, atom, box, *force_)" in sampler
assert "struct RpmdJANativeFitOptions" in native_header
assert "std::uint64_t frame_count" in native_header
assert 'if (tokens[1] == "fit_samples")' in run
assert "fit_rpmd_ja_native_reference_from_samples(options, tokens[2], atom, box, force)" in run
assert "derived.temperature=header.temperature" in native_fit and "derived.sample_interval=static_cast<int>(std::llround(sample_interval))" in native_fit
assert "fit_rpmd_ja_native_reference_checked(derived,spool_path,header.frame_count,atom,box,force,true)" in native_fit
assert "raw_model!=expected_model||raw_config!=expected_config" in native_fit
assert "raw_types[i]==header.types[i]&&raw_masses[i]==header.masses[i]" in native_fit
assert "raw_positions[i]-r0[i]" in native_fit
assert "generate_raw&&std::remove(generated_raw_path.c_str())" in native_fit
assert "options.raw_input_path.empty()?generated_raw_path:options.raw_input_path" in native_fit
assert '#include "force/nep_charge.cuh"' in native_fit
assert 'record_search("FIXED_INDEPENDENT"' in native_fit
assert 'lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,steps,4)' in native_fit
assert "write_response_stats(trace,response_progress)" in native_fit
assert "observed_cov_condition" in native_fit and "predicted_cov_condition" in native_fit
assert "candidate_direction_original_mass_weighted" in source("src/measure/rpmd_ja_qnep_prepare.cu")
assert "pack_owned=false;return true;" in native_fit and "no automatic stability cut was added" in native_fit
assert 'preserve_candidate_package_on_full_spd_failure(e.what(),own_scratch.pack_owned)' in native_fit
assert 'qNEP translation-complement Hessian is not positive definite' in source("src/measure/rpmd_ja_qnep_prepare.cu")
assert 'qNEP translation-complement Hessian is not positive definite' in native_fit
assert 'std::sort(modes.begin(),modes.end()' in native_fit
assert 'low_mode_probes<4' in native_fit and 'if(add(soft[i].vector))++low_mode_probes' in native_fit
assert 'observed_cov_status=' in native_fit and 'predicted_cov_status=' in native_fit
assert 'response_ibp_status=' in native_fit and 'NOT_COMPUTED' in native_fit
spectral_start = native_fit.index("std::vector<double> spectral_row(")
spectral_end = native_fit.index("void apply_total(", spectral_start)
assert "make_design(" not in native_fit[spectral_start:spectral_end]
assert "block_variance_relative_delta" in native_fit
assert "test_fit_samples_entry_preserves_inputs" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_full_spd_failure_keeps_candidate_pack" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_probe_selection_skips_duplicate_low_modes" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_response_uncomputed_values_are_explicit" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "test_active_set_qp_snapshot_and_rejection" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "tests/data/ja_reference_resample.bin.qp_state.txt" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "solve_active_equalities" in native_fit and "cusolverDnDgesvd(solver,'S','S',p,n" in native_fit
assert "ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT" in native_fit and "ACTIVE_EQUALITY_UNRELIABLE" in native_fit
active_svd_start = native_fit.index("bool solve_active_equalities(")
active_svd_end = native_fit.index("ActiveSetPolishResult polish_cut_qp(", active_svd_start)
active_svd = native_fit[active_svd_start:active_svd_end]
assert "if(q>static_cast<std::size_t>(p)){failure_reason=\"ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT\";return false;}" in active_svd
assert "if(da){cudaFree(da);da=nullptr;}" in active_svd and "if(info){cudaFree(info);info=nullptr;}" in active_svd
assert "rounded_alpha" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "rounded_exit.accepted" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "dimension_rank_failure.failure_reason" in source("tests/rpmd_ja_native_fit_cuda_test.cu")
assert "first_polish_iteration=4096" in native_fit and "max_iterations=200000" in native_fit
assert "result.polish_attempts<2" in native_fit and "4*(m+static_cast<std::size_t>(svd.p))" in native_fit
assert 'method="active_set_polish"' in native_fit and '<<" polish_attempts="' in native_fit
assert "polish_failure_reason" in native_fit and "ACTIVE_FULL_KKT_NOT_ACCEPTED" in native_fit
assert (ROOT / "tests/data/ja_reference_resample.bin.qp_state.txt").stat().st_size > 1_000_000
assert "POTRF_diagonal_shift_eV_per_A2_per_amu" in source("src/measure/rpmd_ja_qnep_prepare.cu")

pre_run_start = sampler_fit.index("void RpmdJA_Fit::pre_run(")
pre_run_end = sampler_fit.index("void RpmdJA_Fit::post_force(", pre_run_start)
pre_run_body = sampler_fit[pre_run_start:pre_run_end]
assert 'lock_path_ = output_path_ + ".fit.lock";' in sampler_fit
assert 'output_path_ + ".fit.txt.tmp"' in pre_run_body
assert pre_run_body.index("if (file_exists(path))") < pre_run_body.index('std::fopen(lock_path_.c_str(), "wx")')
assert pre_run_body.index('std::fopen(lock_path_.c_str(), "wx")') < pre_run_body.index("spool_.open(")

release_start = sampler_fit.index("void RpmdJA_Fit::release_lock_()")
release_end = sampler_fit.index("void RpmdJA_Fit::pre_run(", release_start)
release_body = sampler_fit[release_start:release_end]
assert "std::fclose(lock_file_);" in release_body
assert "lock_file_ = nullptr;" in release_body
assert "std::remove(lock_path_.c_str());" in release_body
assert "RpmdJA_Fit::~RpmdJA_Fit() { release_lock_(); }" in sampler_fit

post_run_start = sampler_fit.index("void RpmdJA_Fit::post_run(")
post_run_body = sampler_fit[post_run_start:]
assert post_run_body.index("catch (...) {") < post_run_body.index("release_lock_();") < post_run_body.index("throw;")
assert post_run_body.rindex("release_lock_();") > post_run_body.index("fit_rpmd_ja_native_reference(")

assert "__host__ __device__ inline bool rpmd_ja_fit_mic(" in sampling
assert "nearbyint(sx)" in sampling and "nearbyint(sy)" in sampling and "nearbyint(sz)" in sampling
assert "return isfinite(x) && isfinite(y) && isfinite(z);" in sampling
assert "output[static_cast<std::size_t>(9) * atoms + atom] = static_cast<double>(error);" in sampling
assert "rpmd_ja_fit_mic(box.cpu_h + 9, box.cpu_h, x, y, z)" in sampler
assert sampler.count("apply_fit_mic(box,") == 1


# Algebra-only multi-cell check; the C++ helper remains covered by source-contract checks above.
def matvec(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(3)) for row in matrix]


def determinant(h):
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    return a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)


def inverse(h):
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    det = determinant(h)
    return [[(e * j - f * i) / det, (c * i - b * j) / det, (b * f - c * e) / det],
            [(f * g - d * j) / det, (a * j - c * g) / det, (c * d - a * f) / det],
            [(d * i - e * g) / det, (b * g - a * i) / det, (a * e - b * d) / det]]


def check_multicell_mic(h, delta, image):
    lattice_shift = matvec(h, image)
    wrapped_input = [delta[i] + lattice_shift[i] for i in range(3)]
    fractional = matvec(inverse(h), wrapped_input)
    wrapped_fractional = [fractional[i] - round(fractional[i]) for i in range(3)]
    actual = matvec(h, wrapped_fractional)
    assert all(isclose(actual[i], delta[i], rel_tol=1e-12, abs_tol=1e-12) for i in range(3))


check_multicell_mic([[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]],
                    [0.2, -0.4, 1.1], [3.0, -2.0, 4.0])
check_multicell_mic([[4.0, 0.7, -0.3], [0.0, 3.5, 0.4], [0.2, 0.0, 5.0]],
                    [0.15, 0.2, -0.3], [-3.0, 4.0, 2.0])
