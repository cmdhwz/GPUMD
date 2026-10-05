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
