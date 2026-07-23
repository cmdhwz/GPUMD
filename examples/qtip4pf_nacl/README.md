# q-TIP4P/F water + NaCl

## Build the modified GPUMD

This potential adds a new compiled force class, so an existing GPUMD binary
cannot use it. Configure a new build directory from the repository root and
build the `gpumd` target:

```powershell
cmake -S . -B build-qtip4pf
cmake --build build-qtip4pf --config Release --target gpumd
```

Re-running the CMake configure command is required because the project uses a
source-file glob and must discover `src/force/qtip4pf.cu`. With a multi-config
Windows generator the executable is normally
`build-qtip4pf/Release/gpumd.exe`; with a single-config generator it is normally
`build-qtip4pf/gpumd.exe`. The original GPUMD executable can be kept unchanged.

## Use in run.in

Use the potential with:

```text
potential ../../potentials/qtip4pf_nacl.txt
```

The supplied `run.in` performs a one-step static force check. A conservative
starting point for classical finite-temperature dynamics is:

```text
potential ../../potentials/qtip4pf_nacl.txt
velocity 250
ensemble nvt_lan 250 250 100
time_step 0.1
dump_thermo 100
run 10000
```

q-TIP4P/F contains flexible high-frequency O-H stretches, so do not begin with
the 1 fs step commonly used for constrained water. After confirming NVE energy
conservation, a larger step such as 0.25 fs can be tested. For PIMD, replace the
ensemble line as appropriate, for example
`ensemble pimd 32 250 250 100`, while retaining the small initial time step.

`model.xyz` contains only real atoms. Do not add an `M` atom. The species and
potential type order are fixed as `O H Na Cl`. Every water molecule must be a
consecutive `O H H` triplet; Na and Cl atoms may occur between different water
triplets. Charge values in `model.xyz` are ignored by this potential because
the fixed charges are read from the potential file. For example:

```text
O  xO  yO  zO
H  xH1 yH1 zH1
H  xH2 yH2 zH2
Na xNa yNa zNa
Cl xCl yCl zCl
```

The potential-file format after the first line is:

```text
kspace_method  lj_cutoff_A  coulomb_real_cutoff_A
wO wH1 wH2
qO qH qM qNa qCl
bond_r0_A bond_D_eV bond_a_A^-1
angle_theta0_rad angle_k_eV_rad^-2
epsilon_O_eV  sigma_O_A
epsilon_H_eV  sigma_H_A
epsilon_Na_eV sigma_Na_A
epsilon_Cl_eV sigma_Cl_A
number_of_explicit_LJ_pair_overrides
type_i type_j epsilon_ij_eV sigma_ij_A
... one line for each override
```

`kspace_method` can be `pppm` or `ewald`. Lennard-Jones cross interactions use
Lorentz-Berthelot mixing unless an explicit pair override is supplied. The
number of overrides can be zero. All supplied numbers use GPUMD units
(angstrom and eV).

The default `qtip4pf_nacl.txt` uses Madrid-2019 NaCl parameters: ion charges
are +0.85/-0.85 e and the O-Na, O-Cl, and Na-Cl LJ pairs are explicitly
specified rather than mixed. This is the recommended starting point for salt
ice because Madrid-2019 was parameterized with TIP4P/2005, from which the
q-TIP4P/F intermolecular water parameters were derived, and it has been used in
published NaCl/ice studies. This combination still requires validation because
Madrid-2019 was not fitted directly to flexible, quantum q-TIP4P/F water.

`../../potentials/qtip4pf_nacl_jc_tip4pew.txt` preserves the original full-charge
Joung-Cheatham/TIP4P-Ew values for comparison. Those values are a defensible
control model but should not be described as q-TIP4P/F-specific parameters.

## Repulsive interactions

The implementation includes the same repulsive interactions as the supplied
OpenMM `NonbondedForce`:

```text
U_LJ(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6]
U_C(r)  = k_e q_i q_j/r
```

The positive `r^-12` LJ term supplies the short-range core repulsion. The
nonzero LJ cores are O-O, O-Na, O-Cl, Na-Na, Na-Cl, and Cl-Cl. H and the virtual
M site have zero LJ epsilon, exactly as in the water XML, so every LJ pair
involving H or M is zero. Their nonbonded interaction with ions and other waters
is electrostatic instead. Like-signed charge pairs repel and opposite-signed
pairs attract.

All Coulomb pairs belonging to the same water molecule are removed from the
Ewald/PPPM result. In particular, intramolecular H-H Coulomb repulsion is not
kept; the flexible O-H bond and H-O-H angle potentials define the molecular
geometry. No extra Born-Mayer or Buckingham repulsion is added because no such
term exists in the source XML.

The M position is rebuilt every force call as

```text
rM = 0.73612 rO + 0.13194 rH1 + 0.13194 rH2
```

using minimum-image O-H vectors. Its force, potential energy, and per-site
virial are redistributed to O, H1, and H2 with the same weights. All Coulomb
interactions within one water molecule are excluded after the Ewald/PPPM sum.

The current implementation requires a charge-neutral periodic system and does
not couple the virtual M site to GPUMD's external-electric-field feature. The
force and total thermodynamic quantities are the primary implementation target;
heat-current/HNEMD calculations with the chosen per-atom energy and virial
partition have not yet been validated against OpenMM or another reference code.

The included `run.in` and `model.xyz` form a one-step force-evaluation smoke
test. They are not an equilibrated salt-ice configuration.

Parameter references:

- Madrid-2019: Zeron, Abascal, and Vega, J. Chem. Phys. 151, 134504 (2019),
  DOI 10.1063/1.5121392.
- Original full-charge control: Joung and Cheatham, J. Phys. Chem. B 112,
  9020-9041 (2008), DOI 10.1021/jp8001614.
