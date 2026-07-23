/*
    q-TIP4P/F water with fixed-charge NaCl for GPUMD.

    Real atoms in model.xyz are O, H, Na, and Cl.  Each water molecule must
    occur as a consecutive O H H triplet.  The massless M charge site is built
    internally every force call and its force is redistributed to O, H1, H2.
*/

#include "qtip4pf.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstring>
#include <vector>

namespace
{

constexpr int BLOCK_SIZE = 128;

__global__ void build_charge_sites(
  const int N,
  const int number_of_waters,
  const Box box,
  const QTIP4PF_Para para,
  const int* water_O,
  const int* water_H1,
  const int* water_H2,
  const double* x,
  const double* y,
  const double* z,
  double* site_x,
  double* site_y,
  double* site_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    site_x[n] = x[n];
    site_y[n] = y[n];
    site_z[n] = z[n];
  }
  if (n < number_of_waters) {
    const int iO = water_O[n];
    const int iH1 = water_H1[n];
    const int iH2 = water_H2[n];

    double d1x = x[iH1] - x[iO];
    double d1y = y[iH1] - y[iO];
    double d1z = z[iH1] - z[iO];
    double d2x = x[iH2] - x[iO];
    double d2y = y[iH2] - y[iO];
    double d2z = z[iH2] - z[iO];
    apply_mic(box, d1x, d1y, d1z);
    apply_mic(box, d2x, d2y, d2z);

    const int iM = N + n;
    site_x[iM] = para.weight[0] * x[iO] +
                 para.weight[1] * (x[iO] + d1x) +
                 para.weight[2] * (x[iO] + d2x);
    site_y[iM] = para.weight[0] * y[iO] +
                 para.weight[1] * (y[iO] + d1y) +
                 para.weight[2] * (y[iO] + d2y);
    site_z[iM] = para.weight[0] * z[iO] +
                 para.weight[1] * (z[iO] + d1z) +
                 para.weight[2] * (z[iO] + d2z);
  }
}

__global__ void find_lj_force(
  const int N,
  const Box box,
  const QTIP4PF_Para para,
  const int* NN,
  const int* NL,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  double* fx,
  double* fy,
  double* fz,
  double* virial,
  double* potential)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  const int type1 = type[n1];
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  float sum_f[3] = {0.0f, 0.0f, 0.0f};
  float sum_virial[9] = {0.0f};
  float sum_potential = 0.0f;

  for (int k = 0; k < NN[n1]; ++k) {
    const int n2 = NL[n1 + static_cast<size_t>(N) * k];
    const int type2 = type[n2];
    double dx = x[n2] - x1;
    double dy = y[n2] - y1;
    double dz = z[n2] - z1;
    apply_mic(box, dx, dy, dz);
    const float r2 = float(dx * dx + dy * dy + dz * dz);
    if (r2 >= para.lj_cutoff_square || r2 <= 0.0f) {
      continue;
    }

    const float inv_r2 = 1.0f / r2;
    const float inv_r6 = inv_r2 * inv_r2 * inv_r2;
    const float s6e4 = para.lj_s6e4[type1][type2];
    const float s12e4 = para.lj_s12e4[type1][type2];
    const float pair_energy = s12e4 * inv_r6 * inv_r6 - s6e4 * inv_r6;
    const float force_over_r =
      6.0f * (s6e4 * inv_r6 - 2.0f * s12e4 * inv_r6 * inv_r6) * inv_r2;
    const float pair_fx = force_over_r * float(dx);
    const float pair_fy = force_over_r * float(dy);
    const float pair_fz = force_over_r * float(dz);

    sum_f[0] += pair_fx;
    sum_f[1] += pair_fy;
    sum_f[2] += pair_fz;
    sum_potential += 0.5f * pair_energy;
    sum_virial[0] -= 0.5f * float(dx) * pair_fx;
    sum_virial[1] -= 0.5f * float(dy) * pair_fy;
    sum_virial[2] -= 0.5f * float(dz) * pair_fz;
    sum_virial[3] -= 0.5f * float(dx) * pair_fy;
    sum_virial[4] -= 0.5f * float(dx) * pair_fz;
    sum_virial[5] -= 0.5f * float(dy) * pair_fz;
    sum_virial[6] -= 0.5f * float(dy) * pair_fx;
    sum_virial[7] -= 0.5f * float(dz) * pair_fx;
    sum_virial[8] -= 0.5f * float(dz) * pair_fy;
  }

  fx[n1] += sum_f[0];
  fy[n1] += sum_f[1];
  fz[n1] += sum_f[2];
  potential[n1] += sum_potential;
  for (int c = 0; c < 9; ++c) {
    virial[n1 + static_cast<size_t>(N) * c] += sum_virial[c];
  }
}

__global__ void find_coulomb_real_space(
  const int N,
  const Box box,
  const QTIP4PF_Para para,
  const int* NN,
  const int* NL,
  const float* charge,
  const double* x,
  const double* y,
  const double* z,
  double* fx,
  double* fy,
  double* fz,
  double* virial,
  double* potential)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  const float q1 = charge[n1];
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  float sum_f[3] = {0.0f, 0.0f, 0.0f};
  float sum_virial[9] = {0.0f};
  float sum_potential = -0.5f * K_C_SP * para.two_alpha_over_sqrt_pi * q1 * q1;

  for (int k = 0; k < NN[n1]; ++k) {
    const int n2 = NL[n1 + static_cast<size_t>(N) * k];
    double dx = x[n2] - x1;
    double dy = y[n2] - y1;
    double dz = z[n2] - z1;
    apply_mic(box, dx, dy, dz);
    const float r2 = float(dx * dx + dy * dy + dz * dz);
    if (r2 >= para.coulomb_cutoff_square || r2 <= 0.0f) {
      continue;
    }

    const float q2 = charge[n2];
    const float qq = q1 * q2;
    if (qq == 0.0f) {
      continue;
    }
    const float r = sqrtf(r2);
    const float inv_r = 1.0f / r;
    const float erfc_r = erfcf(para.alpha * r) * inv_r;
    const float gaussian =
      para.two_alpha_over_sqrt_pi * expf(-para.alpha * para.alpha * r2);
    const float force_over_r = -K_C_SP * qq * (erfc_r + gaussian) / r2;
    const float pair_fx = force_over_r * float(dx);
    const float pair_fy = force_over_r * float(dy);
    const float pair_fz = force_over_r * float(dz);

    sum_f[0] += pair_fx;
    sum_f[1] += pair_fy;
    sum_f[2] += pair_fz;
    sum_potential += 0.5f * K_C_SP * qq * erfc_r;
    sum_virial[0] -= 0.5f * float(dx) * pair_fx;
    sum_virial[1] -= 0.5f * float(dy) * pair_fy;
    sum_virial[2] -= 0.5f * float(dz) * pair_fz;
    sum_virial[3] -= 0.5f * float(dx) * pair_fy;
    sum_virial[4] -= 0.5f * float(dx) * pair_fz;
    sum_virial[5] -= 0.5f * float(dy) * pair_fz;
    sum_virial[6] -= 0.5f * float(dy) * pair_fx;
    sum_virial[7] -= 0.5f * float(dz) * pair_fx;
    sum_virial[8] -= 0.5f * float(dz) * pair_fy;
  }

  fx[n1] += sum_f[0];
  fy[n1] += sum_f[1];
  fz[n1] += sum_f[2];
  potential[n1] += sum_potential;
  for (int c = 0; c < 9; ++c) {
    virial[n1 + static_cast<size_t>(N) * c] += sum_virial[c];
  }
}

__device__ void subtract_intramolecular_pair(
  const int N,
  const Box box,
  const int i,
  const int j,
  const float qi,
  const float qj,
  const double* x,
  const double* y,
  const double* z,
  double* fx,
  double* fy,
  double* fz,
  double* virial,
  double* potential)
{
  const float qq = qi * qj;
  if (qq == 0.0f) {
    return;
  }
  double dx = x[j] - x[i];
  double dy = y[j] - y[i];
  double dz = z[j] - z[i];
  apply_mic(box, dx, dy, dz);
  const float r2 = float(dx * dx + dy * dy + dz * dz);
  if (r2 <= 0.0f) {
    return;
  }
  const float inv_r = rsqrtf(r2);
  const float correction_energy = -K_C_SP * qq * inv_r;
  const float force_over_r = K_C_SP * qq * inv_r * inv_r * inv_r;
  const float pair_fx = force_over_r * float(dx);
  const float pair_fy = force_over_r * float(dy);
  const float pair_fz = force_over_r * float(dz);

  fx[i] += pair_fx;
  fy[i] += pair_fy;
  fz[i] += pair_fz;
  fx[j] -= pair_fx;
  fy[j] -= pair_fy;
  fz[j] -= pair_fz;
  potential[i] += 0.5f * correction_energy;
  potential[j] += 0.5f * correction_energy;

  const float v[9] = {
    -0.5f * float(dx) * pair_fx,
    -0.5f * float(dy) * pair_fy,
    -0.5f * float(dz) * pair_fz,
    -0.5f * float(dx) * pair_fy,
    -0.5f * float(dx) * pair_fz,
    -0.5f * float(dy) * pair_fz,
    -0.5f * float(dy) * pair_fx,
    -0.5f * float(dz) * pair_fx,
    -0.5f * float(dz) * pair_fy};
  for (int c = 0; c < 9; ++c) {
    virial[i + static_cast<size_t>(N) * c] += v[c];
    virial[j + static_cast<size_t>(N) * c] += v[c];
  }
}

__global__ void subtract_intramolecular_coulomb(
  const int number_of_atoms,
  const int number_of_sites,
  const int number_of_waters,
  const Box box,
  const QTIP4PF_Para para,
  const int* water_O,
  const int* water_H1,
  const int* water_H2,
  const double* x,
  const double* y,
  const double* z,
  double* fx,
  double* fy,
  double* fz,
  double* virial,
  double* potential)
{
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= number_of_waters) {
    return;
  }
  const int site[4] = {water_O[w], water_H1[w], water_H2[w], number_of_atoms + w};
  const float q[4] = {para.charge[0], para.charge[1], para.charge[1], para.charge[4]};
  for (int a = 0; a < 4; ++a) {
    for (int b = a + 1; b < 4; ++b) {
      subtract_intramolecular_pair(
        number_of_sites,
        box,
        site[a],
        site[b],
        q[a],
        q[b],
        x,
        y,
        z,
        fx,
        fy,
        fz,
        virial,
        potential);
    }
  }
}

__device__ void add_outer_product(float scale, const float r[3], const float f[3], float v[9])
{
  v[0] += scale * r[0] * f[0];
  v[1] += scale * r[1] * f[1];
  v[2] += scale * r[2] * f[2];
  v[3] += scale * r[0] * f[1];
  v[4] += scale * r[0] * f[2];
  v[5] += scale * r[1] * f[2];
  v[6] += scale * r[1] * f[0];
  v[7] += scale * r[2] * f[0];
  v[8] += scale * r[2] * f[1];
}

__global__ void find_intra_water_force(
  const int N,
  const int number_of_waters,
  const Box box,
  const QTIP4PF_Para para,
  const int* water_O,
  const int* water_H1,
  const int* water_H2,
  const double* x,
  const double* y,
  const double* z,
  double* fx,
  double* fy,
  double* fz,
  double* virial,
  double* potential)
{
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= number_of_waters) {
    return;
  }

  const int index[3] = {water_O[w], water_H1[w], water_H2[w]};
  float f[3][3] = {{0.0f}};
  float pe[3] = {0.0f, 0.0f, 0.0f};
  float v[3][9] = {{0.0f}};
  float rOH[2][3];

  for (int h = 0; h < 2; ++h) {
    const int iH = index[h + 1];
    double dx = x[iH] - x[index[0]];
    double dy = y[iH] - y[index[0]];
    double dz = z[iH] - z[index[0]];
    apply_mic(box, dx, dy, dz);
    rOH[h][0] = float(dx);
    rOH[h][1] = float(dy);
    rOH[h][2] = float(dz);
    const float r = sqrtf(rOH[h][0] * rOH[h][0] + rOH[h][1] * rOH[h][1] +
                          rOH[h][2] * rOH[h][2]);
    if (r <= 0.0f) {
      continue;
    }
    const float dr = r - para.bond_r0;
    const float a2 = para.bond_a * para.bond_a;
    const float a3 = a2 * para.bond_a;
    const float a4 = a3 * para.bond_a;
    const float energy = para.bond_D *
      (a2 * dr * dr - a3 * dr * dr * dr + (7.0f / 12.0f) * a4 * dr * dr * dr * dr);
    const float dUdr = para.bond_D *
      (2.0f * a2 * dr - 3.0f * a3 * dr * dr + (7.0f / 3.0f) * a4 * dr * dr * dr);
    float force_O[3];
    float force_H[3];
    for (int d = 0; d < 3; ++d) {
      force_O[d] = dUdr * rOH[h][d] / r;
      force_H[d] = -force_O[d];
      f[0][d] += force_O[d];
      f[h + 1][d] += force_H[d];
    }
    pe[0] += 0.5f * energy;
    pe[h + 1] += 0.5f * energy;
    add_outer_product(0.5f, rOH[h], force_H, v[0]);
    add_outer_product(0.5f, rOH[h], force_H, v[h + 1]);
  }

  const float r1 = sqrtf(rOH[0][0] * rOH[0][0] + rOH[0][1] * rOH[0][1] +
                         rOH[0][2] * rOH[0][2]);
  const float r2 = sqrtf(rOH[1][0] * rOH[1][0] + rOH[1][1] * rOH[1][1] +
                         rOH[1][2] * rOH[1][2]);
  if (r1 > 0.0f && r2 > 0.0f) {
    float cosine = (rOH[0][0] * rOH[1][0] + rOH[0][1] * rOH[1][1] +
                    rOH[0][2] * rOH[1][2]) / (r1 * r2);
    cosine = fminf(1.0f, fmaxf(-1.0f, cosine));
    const float theta = acosf(cosine);
    const float delta = theta - para.angle_theta0;
    const float energy = 0.5f * para.angle_k * delta * delta;
    const float sine = sqrtf(fmaxf(1.0e-12f, 1.0f - cosine * cosine));
    const float prefactor = para.angle_k * delta / sine;
    float force_H1[3];
    float force_H2[3];
    for (int d = 0; d < 3; ++d) {
      force_H1[d] = prefactor *
        (rOH[1][d] / (r1 * r2) - cosine * rOH[0][d] / (r1 * r1));
      force_H2[d] = prefactor *
        (rOH[0][d] / (r1 * r2) - cosine * rOH[1][d] / (r2 * r2));
      f[1][d] += force_H1[d];
      f[2][d] += force_H2[d];
      f[0][d] -= force_H1[d] + force_H2[d];
    }
    for (int a = 0; a < 3; ++a) {
      pe[a] += energy / 3.0f;
    }
    add_outer_product(0.5f, rOH[0], force_H1, v[0]);
    add_outer_product(0.5f, rOH[1], force_H2, v[0]);
    add_outer_product(0.5f, rOH[0], force_H1, v[1]);
    add_outer_product(0.5f, rOH[1], force_H2, v[2]);
  }

  for (int a = 0; a < 3; ++a) {
    const int i = index[a];
    fx[i] += f[a][0];
    fy[i] += f[a][1];
    fz[i] += f[a][2];
    potential[i] += pe[a];
    for (int c = 0; c < 9; ++c) {
      virial[i + static_cast<size_t>(N) * c] += v[a][c];
    }
  }
}

__global__ void scatter_real_charge_sites(
  const int N,
  const int number_of_sites,
  const double* site_force,
  const double* site_virial,
  const double* site_potential,
  double* force,
  double* virial,
  double* potential)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }
  for (int d = 0; d < 3; ++d) {
    force[n + static_cast<size_t>(N) * d] +=
      site_force[n + static_cast<size_t>(number_of_sites) * d];
  }
  potential[n] += site_potential[n];
  for (int c = 0; c < 9; ++c) {
    virial[n + static_cast<size_t>(N) * c] +=
      site_virial[n + static_cast<size_t>(number_of_sites) * c];
  }
}

__global__ void scatter_virtual_sites(
  const int N,
  const int number_of_sites,
  const int number_of_waters,
  const QTIP4PF_Para para,
  const int* water_O,
  const int* water_H1,
  const int* water_H2,
  const double* site_force,
  const double* site_virial,
  const double* site_potential,
  double* force,
  double* virial,
  double* potential)
{
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= number_of_waters) {
    return;
  }
  const int iM = N + w;
  const int parent[3] = {water_O[w], water_H1[w], water_H2[w]};
  for (int a = 0; a < 3; ++a) {
    const int i = parent[a];
    const double weight = para.weight[a];
    for (int d = 0; d < 3; ++d) {
      force[i + static_cast<size_t>(N) * d] +=
        weight * site_force[iM + static_cast<size_t>(number_of_sites) * d];
    }
    potential[i] += weight * site_potential[iM];
    for (int c = 0; c < 9; ++c) {
      virial[i + static_cast<size_t>(N) * c] +=
        weight * site_virial[iM + static_cast<size_t>(number_of_sites) * c];
    }
  }
}

} // namespace

QTIP4PF::QTIP4PF(FILE* fid, int num_types, int num_atoms) : number_of_atoms_(num_atoms)
{
  if (num_types != QTIP4PF_NUM_REAL_TYPES) {
    PRINT_INPUT_ERROR("qtip4pf requires exactly four real atom types: O H Na Cl.\n");
  }

  const char* expected[QTIP4PF_NUM_REAL_TYPES] = {"O", "H", "Na", "Cl"};
  for (int t = 0; t < QTIP4PF_NUM_REAL_TYPES; ++t) {
    char symbol[16];
    const int count = fscanf(fid, "%15s", symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading atom symbols for qtip4pf.");
    if (strcmp(symbol, expected[t]) != 0) {
      PRINT_INPUT_ERROR("The qtip4pf atom type order must be: O H Na Cl.\n");
    }
  }

  char kspace_method[16];
  int count = fscanf(fid, "%15s%lf%lf", kspace_method, &lj_cutoff_, &coulomb_cutoff_);
  PRINT_SCANF_ERROR(count, 3, "Reading qtip4pf kspace method and cutoffs.");
  if (strcmp(kspace_method, "pppm") == 0) {
    use_pppm_ = true;
  } else if (strcmp(kspace_method, "ewald") == 0) {
    use_pppm_ = false;
  } else {
    PRINT_INPUT_ERROR("qtip4pf kspace method must be pppm or ewald.\n");
  }
  if (lj_cutoff_ <= 1.0 || coulomb_cutoff_ <= 1.0) {
    PRINT_INPUT_ERROR("qtip4pf cutoffs must be larger than 1 angstrom.\n");
  }

  count = fscanf(fid, "%f%f%f", &para_.weight[0], &para_.weight[1], &para_.weight[2]);
  PRINT_SCANF_ERROR(count, 3, "Reading qtip4pf virtual-site weights.");
  const float weight_sum = para_.weight[0] + para_.weight[1] + para_.weight[2];
  if (fabsf(weight_sum - 1.0f) > 1.0e-5f) {
    PRINT_INPUT_ERROR("qtip4pf virtual-site weights must sum to one.\n");
  }

  count = fscanf(
    fid,
    "%f%f%f%f%f",
    &para_.charge[0],
    &para_.charge[1],
    &para_.charge[4],
    &para_.charge[2],
    &para_.charge[3]);
  PRINT_SCANF_ERROR(count, 5, "Reading qtip4pf charges (O H M Na Cl).");

  count = fscanf(fid, "%f%f%f", &para_.bond_r0, &para_.bond_D, &para_.bond_a);
  PRINT_SCANF_ERROR(count, 3, "Reading qtip4pf bond parameters.");
  count = fscanf(fid, "%f%f", &para_.angle_theta0, &para_.angle_k);
  PRINT_SCANF_ERROR(count, 2, "Reading qtip4pf angle parameters.");

  float epsilon[QTIP4PF_NUM_REAL_TYPES];
  float sigma[QTIP4PF_NUM_REAL_TYPES];
  for (int t = 0; t < QTIP4PF_NUM_REAL_TYPES; ++t) {
    count = fscanf(fid, "%f%f", &epsilon[t], &sigma[t]);
    PRINT_SCANF_ERROR(count, 2, "Reading qtip4pf LJ epsilon and sigma.");
    if (epsilon[t] < 0.0f || sigma[t] <= 0.0f) {
      PRINT_INPUT_ERROR("qtip4pf LJ epsilon must be nonnegative and sigma positive.\n");
    }
  }

  float pair_epsilon[QTIP4PF_NUM_REAL_TYPES][QTIP4PF_NUM_REAL_TYPES];
  float pair_sigma[QTIP4PF_NUM_REAL_TYPES][QTIP4PF_NUM_REAL_TYPES];
  for (int i = 0; i < QTIP4PF_NUM_REAL_TYPES; ++i) {
    for (int j = 0; j < QTIP4PF_NUM_REAL_TYPES; ++j) {
      pair_epsilon[i][j] = sqrtf(epsilon[i] * epsilon[j]);
      pair_sigma[i][j] = 0.5f * (sigma[i] + sigma[j]);
    }
  }

  int number_of_pair_overrides = 0;
  count = fscanf(fid, "%d", &number_of_pair_overrides);
  if (count == EOF) {
    number_of_pair_overrides = 0; // backward compatibility with the first file format
  } else {
    PRINT_SCANF_ERROR(count, 1, "Reading number of qtip4pf LJ pair overrides.");
  }
  if (number_of_pair_overrides < 0 || number_of_pair_overrides > 10) {
    PRINT_INPUT_ERROR("The number of qtip4pf LJ pair overrides must be between 0 and 10.\n");
  }
  for (int p = 0; p < number_of_pair_overrides; ++p) {
    char symbol_i[16];
    char symbol_j[16];
    float epsilon_ij;
    float sigma_ij;
    count = fscanf(fid, "%15s%15s%f%f", symbol_i, symbol_j, &epsilon_ij, &sigma_ij);
    PRINT_SCANF_ERROR(count, 4, "Reading a qtip4pf LJ pair override.");
    if (epsilon_ij < 0.0f || sigma_ij <= 0.0f) {
      PRINT_INPUT_ERROR("A qtip4pf pair epsilon must be nonnegative and sigma positive.\n");
    }
    int type_i = -1;
    int type_j = -1;
    for (int t = 0; t < QTIP4PF_NUM_REAL_TYPES; ++t) {
      if (strcmp(symbol_i, expected[t]) == 0) {
        type_i = t;
      }
      if (strcmp(symbol_j, expected[t]) == 0) {
        type_j = t;
      }
    }
    if (type_i < 0 || type_j < 0) {
      PRINT_INPUT_ERROR("Unknown atom symbol in a qtip4pf LJ pair override.\n");
    }
    pair_epsilon[type_i][type_j] = pair_epsilon[type_j][type_i] = epsilon_ij;
    pair_sigma[type_i][type_j] = pair_sigma[type_j][type_i] = sigma_ij;
  }

  for (int i = 0; i < QTIP4PF_NUM_REAL_TYPES; ++i) {
    for (int j = 0; j < QTIP4PF_NUM_REAL_TYPES; ++j) {
      para_.lj_s6e4[i][j] =
        4.0f * pair_epsilon[i][j] * powf(pair_sigma[i][j], 6.0f);
      para_.lj_s12e4[i][j] =
        4.0f * pair_epsilon[i][j] * powf(pair_sigma[i][j], 12.0f);
    }
  }

  para_.lj_cutoff_square = float(lj_cutoff_ * lj_cutoff_);
  para_.coulomb_cutoff_square = float(coulomb_cutoff_ * coulomb_cutoff_);
  para_.alpha = float(3.14159265358979323846 / coulomb_cutoff_);
  para_.two_alpha_over_sqrt_pi = 2.0f * para_.alpha / sqrtf(3.14159265358979323846f);
  rc = lj_cutoff_ > coulomb_cutoff_ ? lj_cutoff_ : coulomb_cutoff_;

  if (use_pppm_) {
    pppm_.reset(new PPPM());
    pppm_->initialize(para_.alpha);
  } else {
    ewald_.reset(new Ewald());
    ewald_->initialize(para_.alpha);
  }

  printf("Use q-TIP4P/F water + fixed-charge NaCl potential.\n");
  printf("    real atom order: O H Na Cl\n");
  printf("    kspace: %s, LJ cutoff: %g A, Coulomb cutoff: %g A\n",
         kspace_method,
         lj_cutoff_,
         coulomb_cutoff_);
  printf("    short-range repulsion: Lennard-Jones 4*epsilon*(sigma/r)^12\n");
  printf("    explicit LJ pair overrides: %d\n", number_of_pair_overrides);
  printf("    LJ pairs with nonzero repulsive cores:\n");
  for (int i = 0; i < QTIP4PF_NUM_REAL_TYPES; ++i) {
    for (int j = i; j < QTIP4PF_NUM_REAL_TYPES; ++j) {
      if (pair_epsilon[i][j] > 0.0f) {
        printf(
          "        %s-%s: epsilon=%g eV, sigma=%g A\n",
          expected[i],
          expected[j],
          pair_epsilon[i][j],
          pair_sigma[i][j]);
      }
    }
  }
  printf("    like-signed charge pairs also repel through Ewald/PPPM Coulomb forces.\n");
  printf("    M sites are generated internally and are not atoms in model.xyz.\n");
}

QTIP4PF::~QTIP4PF(void) = default;

void QTIP4PF::initialize_topology(const GPU_Vector<int>& type)
{
  if (int(type.size()) != number_of_atoms_) {
    PRINT_INPUT_ERROR("Atom count changed after qtip4pf potential construction.\n");
  }
  std::vector<int> cpu_type(number_of_atoms_);
  CHECK(gpuMemcpy(
    cpu_type.data(),
    type.data(),
    sizeof(int) * number_of_atoms_,
    gpuMemcpyDeviceToHost));

  std::vector<int> water_O;
  std::vector<int> water_H1;
  std::vector<int> water_H2;
  for (int n = 0; n < number_of_atoms_;) {
    if (cpu_type[n] == 0) {
      if (n + 2 >= number_of_atoms_ || cpu_type[n + 1] != 1 || cpu_type[n + 2] != 1) {
        PRINT_INPUT_ERROR(
          "Each qtip4pf water must be a consecutive O H H triplet in model.xyz.\n");
      }
      water_O.push_back(n);
      water_H1.push_back(n + 1);
      water_H2.push_back(n + 2);
      n += 3;
    } else if (cpu_type[n] == 1) {
      PRINT_INPUT_ERROR("An H atom was found outside an O H H water triplet.\n");
    } else if (cpu_type[n] == 2 || cpu_type[n] == 3) {
      ++n;
    } else {
      PRINT_INPUT_ERROR("qtip4pf supports only O, H, Na, and Cl real atoms.\n");
    }
  }
  if (water_O.empty()) {
    PRINT_INPUT_ERROR("No O H H water triplet was found for qtip4pf.\n");
  }

  number_of_waters_ = int(water_O.size());
  number_of_sites_ = number_of_atoms_ + number_of_waters_;
  water_O_.resize(number_of_waters_);
  water_H1_.resize(number_of_waters_);
  water_H2_.resize(number_of_waters_);
  water_O_.copy_from_host(water_O.data());
  water_H1_.copy_from_host(water_H1.data());
  water_H2_.copy_from_host(water_H2.data());

  std::vector<int> cpu_site_type(number_of_sites_, 0);
  std::vector<float> cpu_site_charge(number_of_sites_, para_.charge[4]);
  double total_charge = 0.0;
  for (int n = 0; n < number_of_atoms_; ++n) {
    cpu_site_type[n] = cpu_type[n];
    cpu_site_charge[n] = para_.charge[cpu_type[n]];
    total_charge += cpu_site_charge[n];
  }
  total_charge += number_of_waters_ * para_.charge[4];
  if (fabs(total_charge) > 1.0e-4) {
    PRINT_INPUT_ERROR("qtip4pf Ewald/PPPM currently requires a charge-neutral system.\n");
  }

  site_type_.resize(number_of_sites_);
  site_charge_.resize(number_of_sites_);
  site_D_real_.resize(number_of_sites_);
  site_position_.resize(static_cast<size_t>(number_of_sites_) * 3);
  site_potential_.resize(number_of_sites_);
  site_force_.resize(static_cast<size_t>(number_of_sites_) * 3);
  site_virial_.resize(static_cast<size_t>(number_of_sites_) * 9);
  site_type_.copy_from_host(cpu_site_type.data());
  site_charge_.copy_from_host(cpu_site_charge.data());

  // Neighbor::find_neighbor_global sorts with one CUDA block per atom, so the
  // allocated maximum (including its 1 A skin) must not exceed 1024 threads.
  const auto safe_neighbor_capacity = [](const double cutoff, const int requested) {
    const double ratio = cutoff / (cutoff + 1.0);
    const int cuda_limit = int(1024.0 * ratio * ratio * ratio);
    return requested < cuda_limit ? requested : cuda_limit;
  };
  lj_neighbor_.initialize(
    lj_cutoff_, number_of_atoms_, safe_neighbor_capacity(lj_cutoff_, 700));
  charge_neighbor_.initialize(
    coulomb_cutoff_, number_of_sites_, safe_neighbor_capacity(coulomb_cutoff_, 750));
  topology_initialized_ = true;
  printf("    detected %d water molecules and %d internal charge sites.\n",
         number_of_waters_,
         number_of_sites_);
}

void QTIP4PF::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  if (!topology_initialized_) {
    initialize_topology(type);
  }

  const int real_grid = (number_of_atoms_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int water_grid = (number_of_waters_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int site_grid = (number_of_sites_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

  build_charge_sites<<<real_grid, BLOCK_SIZE>>>(
    number_of_atoms_,
    number_of_waters_,
    box,
    para_,
    water_O_.data(),
    water_H1_.data(),
    water_H2_.data(),
    position.data(),
    position.data() + number_of_atoms_,
    position.data() + 2 * number_of_atoms_,
    site_position_.data(),
    site_position_.data() + number_of_sites_,
    site_position_.data() + 2 * number_of_sites_);
  GPU_CHECK_KERNEL

  lj_neighbor_.find_neighbor_global(lj_cutoff_, box, type, position);
  find_lj_force<<<real_grid, BLOCK_SIZE>>>(
    number_of_atoms_,
    box,
    para_,
    lj_neighbor_.NN.data(),
    lj_neighbor_.NL.data(),
    type.data(),
    position.data(),
    position.data() + number_of_atoms_,
    position.data() + 2 * number_of_atoms_,
    force.data(),
    force.data() + number_of_atoms_,
    force.data() + 2 * number_of_atoms_,
    virial.data(),
    potential.data());
  GPU_CHECK_KERNEL

  find_intra_water_force<<<water_grid, BLOCK_SIZE>>>(
    number_of_atoms_,
    number_of_waters_,
    box,
    para_,
    water_O_.data(),
    water_H1_.data(),
    water_H2_.data(),
    position.data(),
    position.data() + number_of_atoms_,
    position.data() + 2 * number_of_atoms_,
    force.data(),
    force.data() + number_of_atoms_,
    force.data() + 2 * number_of_atoms_,
    virial.data(),
    potential.data());
  GPU_CHECK_KERNEL

  site_potential_.fill(0.0);
  site_force_.fill(0.0);
  site_virial_.fill(0.0);
  site_D_real_.fill(0.0f);
  charge_neighbor_.find_neighbor_global(coulomb_cutoff_, box, site_type_, site_position_);

  if (use_pppm_) {
    pppm_->find_force(
      number_of_sites_,
      0,
      number_of_sites_,
      box,
      site_charge_,
      site_position_,
      site_D_real_,
      site_force_,
      site_virial_,
      site_potential_);
  } else {
    ewald_->find_force(
      number_of_sites_,
      0,
      number_of_sites_,
      box.cpu_h,
      site_charge_,
      site_position_,
      site_D_real_,
      site_force_,
      site_virial_,
      site_potential_);
  }

  find_coulomb_real_space<<<site_grid, BLOCK_SIZE>>>(
    number_of_sites_,
    box,
    para_,
    charge_neighbor_.NN.data(),
    charge_neighbor_.NL.data(),
    site_charge_.data(),
    site_position_.data(),
    site_position_.data() + number_of_sites_,
    site_position_.data() + 2 * number_of_sites_,
    site_force_.data(),
    site_force_.data() + number_of_sites_,
    site_force_.data() + 2 * number_of_sites_,
    site_virial_.data(),
    site_potential_.data());
  GPU_CHECK_KERNEL

  subtract_intramolecular_coulomb<<<water_grid, BLOCK_SIZE>>>(
    number_of_atoms_,
    number_of_sites_,
    number_of_waters_,
    box,
    para_,
    water_O_.data(),
    water_H1_.data(),
    water_H2_.data(),
    site_position_.data(),
    site_position_.data() + number_of_sites_,
    site_position_.data() + 2 * number_of_sites_,
    site_force_.data(),
    site_force_.data() + number_of_sites_,
    site_force_.data() + 2 * number_of_sites_,
    site_virial_.data(),
    site_potential_.data());
  GPU_CHECK_KERNEL

  scatter_real_charge_sites<<<real_grid, BLOCK_SIZE>>>(
    number_of_atoms_,
    number_of_sites_,
    site_force_.data(),
    site_virial_.data(),
    site_potential_.data(),
    force.data(),
    virial.data(),
    potential.data());
  GPU_CHECK_KERNEL
  scatter_virtual_sites<<<water_grid, BLOCK_SIZE>>>(
    number_of_atoms_,
    number_of_sites_,
    number_of_waters_,
    para_,
    water_O_.data(),
    water_H1_.data(),
    water_H2_.data(),
    site_force_.data(),
    site_virial_.data(),
    site_potential_.data(),
    force.data(),
    virial.data(),
    potential.data());
  GPU_CHECK_KERNEL
}
