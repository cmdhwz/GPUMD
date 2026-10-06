from fractions import Fraction
from math import pi, sin
from pathlib import Path


root = Path(__file__).parents[1]
source = (root / "src/force/pppm.cu").read_text(encoding="utf-8")
header = (root / "src/force/pppm.cuh").read_text(encoding="utf-8")
fixture = (root / "tests/pppm_reference_tangent_cuda_test.cu").read_text(encoding="utf-8")

# Exact quartic assignment rows retain partition of unity and zero summed slope.
w = [
    [Fraction(1, 384), Fraction(-1, 48), Fraction(1, 16), Fraction(-1, 12), Fraction(1, 24)],
    [Fraction(19, 96), Fraction(-11, 24), Fraction(1, 4), Fraction(1, 6), Fraction(-1, 6)],
    [Fraction(115, 192), Fraction(0), Fraction(-5, 8), Fraction(0), Fraction(1, 4)],
    [Fraction(19, 96), Fraction(11, 24), Fraction(1, 4), Fraction(-1, 6), Fraction(-1, 6)],
    [Fraction(1, 384), Fraction(1, 48), Fraction(1, 16), Fraction(1, 12), Fraction(1, 24)],
]
start = source.index("constexpr double exact_W[5][5] = {")
end = source.index("\n  };", start)
rows = source[start:end].splitlines()[1:]
actual_w = []
for line in rows:
    row = line[line.index("{") + 1:line.index("}")]
    values = []
    for expression in row.split(","):
        pieces = expression.strip().split("/")
        value = Fraction(pieces[0].strip())
        if len(pieces) == 2:
            value /= Fraction(pieces[1].strip())
        values.append(value)
    actual_w.append(values)
assert actual_w == w
for delta in (Fraction(-47, 100), Fraction(-13, 100), Fraction(0), Fraction(29, 100), Fraction(47, 100)):
    values = [sum(c * delta**i for i, c in enumerate(row)) for row in actual_w]
    slopes = [sum(i * row[i] * delta ** (i - 1) for i in range(1, 5)) for row in actual_w]
    assert sum(values) == 1
    assert sum(slopes) == 0

# Verify the rational G polynomial against the cardinal-spline image sum.
g_coeff = (1.0, -5.0 / 3.0, 7.0 / 9.0, -17.0 / 189.0, 2.0 / 2835.0)
for x in (0.1, 0.7, 1.3):
    z = sin(x) ** 2
    polynomial = sum(c * z**i for i, c in enumerate(g_coeff))
    image_sum = sum((sin(x + pi * m) / (x + pi * m)) ** 10 for m in range(-200, 201))
    assert abs(polynomial - image_sum) < 2e-14

assert 'sinc_fifth * sinc_fifth' in source
assert 'const double sinc_fifth = sinc_product * sinc_product * sinc_product * sinc_product * sinc_product;' in source
assert 'Source source[3]' in header
assert 'bool signal_resolved = false' in header
assert 'result.analytic_fd_difference <= total_uncertainty' in source
assert 'result.valid = finite;' in source
assert 'result.signal_resolved = finite && std::abs(result.fd_derivative) > 3.0 * total_uncertainty;' in source
assert 'pppm_reference_weight_derivative' not in source[source.index('auto exact_G_at'):source.index('double charge_scale =')]
assert 'source[source].phase[phase]' in fixture and 'source_log_confirmation' in fixture
