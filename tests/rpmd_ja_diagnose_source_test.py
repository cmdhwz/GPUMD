from pathlib import Path


source = Path(__file__).parents[1] / "src/measure/rpmd_ja_reference.cu"
text = source.read_text(encoding="utf-8")
root = Path(__file__).parents[1]
cli = (root / "src/main_gpumd/run.cu").read_text(encoding="utf-8")
native = (root / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")
header = (root / "src/measure/rpmd_ja_reference.cuh").read_text(encoding="utf-8")

# Keep the diagnostic entry point stable for the saved-spool caller.
assert "void diagnose_rpmd_ja_qnep_reference(const double fd_step, Atom& atom, Box& box, Force& force, const bool full)" in text
assert "for (const double step_scale : {0.5, 1.0, 2.0, 5.0, 10.0})" in text
assert "analytic-gradient Hessian sampled D4 norm" not in text
assert "relative-to-D4" in text and "native-force K D4 norm" in text
assert "FAIL=" in text and "qNEP rpmd_ja sampled stencil-step consistency check failed" in text
assert "const int component[3][3] = {{0, 3, 4}, {6, 1, 5}, {7, 8, 2}}" in text
assert "for (const double factor : {0.75, 0.5})" in text
assert 'tokens[1] == "diagnose_samples"' in cli
assert 'tokens[1] == "diagnose"' in cli and 'tokens[3] == "full"' in cli
assert 'tokens[4] == "full"' in cli
assert "diagnose_rpmd_ja_qnep_reference(fd_step, atom, box, force, full)" in cli
assert "diagnose_rpmd_ja_native_fit_samples(tokens[2], fd_step, atom, box, force, full)" in cli
assert "bool full = false" in header
assert "const SampleHeader header = read_header(in, 0, atom, box, 0.0, true)" in native
assert "struct RestorePosition" in native and "~RestorePosition()" in native
sample_diagnostic = native[native.index("void diagnose_rpmd_ja_native_fit_samples("):native.index("#else", native.index("void diagnose_rpmd_ja_native_fit_samples("))]
assert "bool full)" in sample_diagnostic
assert "diagnose_rpmd_ja_qnep_reference(fd_step, atom, box, force, full)" in sample_diagnostic
assert "bool full = false" in (root / "src/measure/rpmd_ja_native_fit.cuh").read_text(encoding="utf-8")
assert "diagnose_rpmd_ja_qnep_reference(double, Atom& atom, Box&, Force&, bool)" in (root / "tests/rpmd_ja_native_fit_cuda_test.cu").read_text(encoding="utf-8")
fit_wrapper = native[native.index("void fit_rpmd_ja_native_reference(const RpmdJANativeFitOptions& options"):native.index("void diagnose_rpmd_ja_native_fit_samples(")]
assert "diagnose_rpmd_ja_qnep_reference" not in fit_wrapper
assert "const bool precheck_failed" in text and "throw std::runtime_error(failure.str())" in text
precheck_table = text[text.index("void print_qnep_precheck_table("):text.index("std::uint64_t qnep_config_fingerprint(")]
assert "100.0 * k" in precheck_table and "100.0 * c[alpha]" in precheck_table
generation = text[text.index("static void generate_rpmd_ja_qnep_raw_reference("):text.index("void diagnose_rpmd_ja_qnep_reference(")]
assert generation.index("print_qnep_precheck_table(") < generation.index("if (precheck_failed)") < generation.index("throw std::runtime_error(failure.str())")
diagnose = text[text.index("void diagnose_rpmd_ja_qnep_reference("):]
assert "requested_h" in diagnose
assert "rpmd_ja_reference_math::central_difference_4th(g[0][r], g[1][r], g[2][r], g[3][r], h)" in text
assert "std::fflush(stdout)" in text
assert 'uniform physical translation JVP direction=1' in text
assert 'mass-weighted translation direction=1/sqrt(total_mass)' in text
assert 'proxies, not the full-column V sum' in text
assert "if (full) {" in diagnose and "for (int coordinate = 0; coordinate < d; ++coordinate)" in diagnose
assert "site_values = evaluator.analytic_site_jvp(direction_host)" in diagnose
assert "axis_site_columns[(coordinate / n) * n + site] += value" in diagnose
assert "QNEPReferenceTangentDiagnostics tangent" in diagnose
assert "compensated_running_sum" in diagnose and "long double extended" in diagnose
assert "diagnose_reference_translation_energy(" in diagnose
for label in ("CODE_INVARIANT", "EXACT_V_NET", "translation oracle"):
    assert label in diagnose
assert "p.fd_uncertainty + p.roundoff > report.precision_target" in diagnose
assert "std::abs(C.analytic_derivative) > report.precision_target" in diagnose
assert "for (int axis = 0; axis < 3; ++axis)" in diagnose and "for (int phase = 0; phase < 2; ++phase)" in diagnose
assert "mesh_oracle, mesh_gradient_net_for_target.limit" in diagnose
assert "full V columns" in diagnose
for heading in ("[1/4] reference/net + exact V", "[2/4] stencil compact/short",
                "[3/4] mesh translation oracle", "[4/4] final summary"):
    assert heading in diagnose
assert "h table (5 rows)" in diagnose and "shortmode=diagnostic label" in diagnose
assert "sampled PASS is not reference acceptance" in diagnose
assert "Shared algebraic identities are not independent oracle proof" in diagnose
assert "source_protocol_version" in diagnose and "source_log_confirmation" in diagnose
assert "native-A(host/FP32 proxy)" in diagnose
assert "A-B" in diagnose and "B-C" in diagnose and "C-residual" in diagnose
assert "std::abs(component_residual) <= kForceTolerance" not in diagnose
assert "full V columns=%d" in diagnose and "evaluation counter excludes the 3N JVPs" in diagnose
assert ".qraw" not in diagnose and "std::ofstream" not in diagnose
raw_generation = text[text.index("static void generate_rpmd_ja_qnep_raw_reference("):text.index("void generate_rpmd_ja_qnep_raw(")]
assert raw_generation.index("exact full-column V gradient net xyz=") < raw_generation.index("if (require_zero_net_gradient && !net_stats.within_limit)") < raw_generation.index("const auto kc_phase_start")
assert "bool require_zero_net_gradient = false" in header
fit_impl = native[native.index("static void fit_rpmd_ja_native_reference_impl("):native.index("void fit_rpmd_ja_native_reference(")]
assert "force,!options.internal_mass_com);" in fit_impl
assert "rpmd_ja_reference_math::net_force_stats(raw_gradient,n)" in native


def central2(plus, minus, h):
    return (plus - minus) / (2.0 * h)


def central4(plus_h, minus_h, plus_2h, minus_2h, h):
    return (8.0 * (plus_h - minus_h) - (plus_2h - minus_2h)) / (12.0 * h)


def neumaier(values):
    running = correction = 0.0
    for value in values:
        next_sum = running + value
        correction += ((running - next_sum) + value if abs(running) >= abs(value)
                       else (value - next_sum) + running)
        running = next_sum
    return running + correction


# Cancellation verifies the compensated path recovers the lost unit.
cancelled = [1.0e16, 1.0, -1.0e16]
ordered_cancelled = 0.0
for value in cancelled:
    ordered_cancelled += value
assert ordered_cancelled == 0.0
assert neumaier(cancelled) == 1.0


def resolved_nonzero(derivative, plateau_error, target):
    return abs(derivative) - 3.0 * plateau_error > target


# The fixed-grid derivative is called resolved only when its conservative
# plateau bound stays above the caller's tolerance.
assert resolved_nonzero(2.0e-5, 2.0e-6, 1.0e-5)
assert not resolved_nonzero(2.0e-5, 4.0e-6, 1.0e-5)


# For a quartic g, D4 is exact at zero; D2(h)-D2(2h) is -6 h^2.
g = lambda x: x**4 + 2.0 * x**3 - 3.0 * x**2 + 4.0 * x + 1.0
h = 0.2
d4 = central4(g(h), g(-h), g(2.0 * h), g(-2.0 * h), h)
d2_delta = central2(g(h), g(-h), h) - central2(g(2.0 * h), g(-2.0 * h), 2.0 * h)
assert abs(d4 - 4.0) < 1.0e-12
assert abs(d2_delta + 6.0 * h * h) < 1.0e-12
assert abs(abs(d2_delta) / abs(d4) - 1.5 * h * h) < 1.0e-12

# A multi-column aggregate checks the same D4 normalization independently.
columns = [(0.5, 2.0), (-1.25, -3.0), (2.0, 0.75)]
measured_d4, measured_delta = [], []
for cubic, linear in columns:
    f = lambda x, a=cubic, b=linear: 0.5 * x**4 + b * x + a * x**3
    measured_d4.append(central4(f(h), f(-h), f(2.0 * h), f(-2.0 * h), h))
    measured_delta.append(central2(f(h), f(-h), h) - central2(f(2.0 * h), f(-2.0 * h), 2.0 * h))
expected_d4_norm = sum(linear * linear for _, linear in columns) ** 0.5
expected_delta_norm = sum((3.0 * cubic * h * h) ** 2 for cubic, _ in columns) ** 0.5
assert abs(sum(x * x for x in measured_d4) ** 0.5 - expected_d4_norm) < 1.0e-12
assert abs(sum(x * x for x in measured_delta) ** 0.5 - expected_delta_norm) < 1.0e-12
assert abs((sum(x * x for x in measured_delta) / sum(x * x for x in measured_d4)) ** 0.5 -
           expected_delta_norm / expected_d4_norm) < 1.0e-12

print("rpmd_ja sampled diagnostic source and stencil oracle: PASS")
