# Hamiltonian contract

U=c0+c1 q+c2 q²+c3 q³+c4 q⁴+K Q²/2+(lambda0+lambda1 q+lambda2 q²)Q; d0 and geometry are separate. Uq=c1+2c2q+3c3q²+4c4q³+(lambda1+2lambda2q)Q and UQ=KQ+lambda0+lambda1q+lambda2q². Hessian entries are Kqq=2c2+6c3q+12c4q²+2lambda2Q, KqQ=lambda1+2lambda2q and KQQ=K. Uqqq=6c3+24c4q, UqqQ=2lambda2, Uqqqq=24c4; all other independent higher components through fourth order vanish. These are implemented in `bridge.derivatives`; native observer implements U, first and second derivatives separately. Finite difference and external local_model comparisons cover selected points, not every parameter domain.

H=p^T C p/2+U, zdot=Cp and pdot=-grad U. For old OHO: c=[a b⁴,delta,-2a b²,0,a], lambda=[-lambda b²,0,lambda]. Native parser proves confinement only for positive relaxed quartic c4-lambda2²/(2K), or the exact positive quadratic boundary case; it conservatively rejects other potentially bounded degenerate cases. Reference stationarity tolerance uses separate absolute derivative-term sums as documented below; positive determinant is required relative to diagonal product at 1e-12. This scale-dependent numerical rule is not a physical stability certificate.

## Gauge-independent stationarity rule

For each gradient component, the acceptance tolerance is 1e-8 times max(1, the sum of the absolute values of that component's explicit derivative terms). The additive c0 and Hessian do not set this scale. Hessian stability additionally requires positive diagonal entries and determinant greater than 1e-12 times their product. The same gradient-term scaling is applied in `bridge.reference`. A native test now encodes both large c0 with valid references and large c0 with a nonstationary left reference; it was not run.

