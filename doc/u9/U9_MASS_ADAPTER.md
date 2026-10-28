# Full-G mass adapter

With W=G^(-1/2), eigenvectors E of W K W and T=W E, coordinates and momenta are z=z*+T xi, p=T^(-T) Pi. Thus T^T G T=I, T^T K T=diag(omega^2) and p^T dz=Pi^T dxi. Row-displacement/column-velocity transport transforms as T^T H T. `normal_modes` checks stability without taking absolute values of negative curvature. Test source covers a non-diagonal metric, a fixed spatial embedding with unequal endpoint masses and H/D substitution; current tests are NOT RUN. Units are caller responsibility.

## Transform and units

Mass whitening is y=W^-1(z-z*), pi=W^T p with W=G^-1/2, so p=W^-T pi and p^T dz=pi^T dy. The mass-weighted curvature is W^T K W. If E diagonalizes it, T=W E and xi=E^T y, Pi=E^T pi, giving z=z*+T xi and p=T^-T Pi. A row-displacement/column-velocity matrix obeys H_modal=T^T H T. For coordinates in Å, masses in amu and energies in eV, G is amu, C is inverse amu, K is eV/Å², and H has eV/Å when its bilinear current is in eV Å per natural time. Other independent-model units must be declared explicitly by their input; the benchmark uses its own dimensionless units.

