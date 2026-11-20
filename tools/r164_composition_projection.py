#!/usr/bin/env python3
"""Finite-window composition projection; does not compute conductivity."""
import argparse
import json
from pathlib import Path
import numpy as np
from r164_provisional_correlations import analyze, digest, fields

SPECIES = ('O', 'H', 'Na', 'Cl')
MASS_FACTOR = 10.18051**2


def composition(directory, hashes):
    directory = Path(directory)
    metadata, scalar = directory/'metadata.txt', directory/'scalar.txt'
    if digest(metadata) != hashes['metadata.txt'] or digest(scalar) != hashes['scalar.txt']:
        raise ValueError('source hash mismatch')
    meta = fields(metadata)
    rows = [line.split() for line in metadata.read_text(encoding='utf8').splitlines() if line.startswith('atom ')]
    if len(rows) != int(meta['N']):
        raise ValueError('atom metadata count mismatch')
    atoms, masses, counts = [], {}, {s: 0 for s in SPECIES}
    for index, row in enumerate(rows):
        if len(row) != 4 or row[1] != str(index) or row[2] not in SPECIES:
            raise ValueError('invalid atom order or species')
        mass = float(row[3])
        if not np.isfinite(mass) or mass <= 0 or (row[2] in masses and masses[row[2]] != mass):
            raise ValueError('invalid or mixed species masses')
        masses[row[2]] = mass
        counts[row[2]] += 1
        atoms.append((row[2], mass))
    if any(counts[s] == 0 for s in SPECIES):
        raise ValueError('all four species required')
    species_mass = np.array([masses[s] * MASS_FACTOR for s in SPECIES])
    maximum = 0.0
    count = 0
    with scalar.open(encoding='utf8', newline='') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            if not line.endswith('\n'):
                raise ValueError('partial scalar row')
            tokens = line.split()
            if len(tokens) < 21 or int(tokens[0]) != count:
                raise ValueError('invalid scalar row')
            p = np.asarray(tokens[6:9], dtype=float)
            n = np.asarray(tokens[9:21], dtype=float).reshape(4, 3)
            if not np.isfinite(p).all() or not np.isfinite(n).all():
                raise ValueError('nonfinite momentum or number flow')
            reconstructed = species_mass @ n
            residual = reconstructed - p
            maximum = max(maximum, float(np.max(np.abs(residual))))
            if not np.allclose(reconstructed, p, rtol=2e-13, atol=1e-10):
                raise ValueError('mass-flow momentum mismatch')
            count += 1
    if digest(metadata) != hashes['metadata.txt'] or digest(scalar) != hashes['scalar.txt']:
        raise ValueError('source changed during verification')
    return atoms, counts, masses, count, maximum


def transform(counts, masses):
    m = np.array([masses[s]*MASS_FACTOR for s in SPECIES])
    n = np.array([counts[s] for s in SPECIES])
    total = float(m @ n)
    t = np.zeros((4,4))
    t[0] = m/total
    for i in range(1,4):
        t[i] = -n[i]*t[0]
        t[i,i] += 1
    a = np.eye(18)
    a[6:,6:] = np.kron(t, np.eye(3))
    return a, total


def matrix_diagnostic(raw, a, rcond, null_rtol):
    if raw is None:
        return {'status':'UNAVAILABLE', 'reason':'source matrix unavailable'}
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (18,18) or not np.isfinite(raw).all():
        raise ValueError('invalid source matrix')
    ordered = a @ raw @ a.T
    sym = (ordered + ordered.T)*0.5
    anti = (ordered - ordered.T)*0.5
    yy, xy = sym[6:,6:], sym[:6,6:]
    eig, vec = np.linalg.eigh(yy)
    scale = float(np.max(np.abs(eig)))
    scan = []
    ranks = []
    for factor in (0.1, 1.0, 10.0):
        cutoff = rcond*factor*scale
        support = eig > cutoff
        null = vec[:,~support]
        checks = {}
        for name, block in (('b',xy[:3]), ('d2',xy[3:]), ('J2',xy[:3]+xy[3:]), ('combined',xy)):
            denominator = float(np.linalg.norm(block))
            numerator = float(np.linalg.norm(block @ null))
            checks[name] = {'null_cross_norm':numerator, 'cross_norm':denominator,
                            'null_cross_ratio':numerator/denominator if denominator else 0.0}
        ranks.append(int(np.count_nonzero(support)))
        scan.append({'factor':factor, 'threshold':float(cutoff), 'rank':ranks[-1], 'null_cross':checks})
    direct_map = np.concatenate((np.eye(3),np.eye(3)),axis=1)
    j_unprojected = direct_map @ sym[:6,:6] @ direct_map.T
    result = {'ordered':ordered.tolist(), 'symmetric':sym.tolist(), 'antisymmetric':anti.tolist(),
              'eigenvalues_YY':eig.tolist(), 'eigenvalue_scale':scale,
              'threshold_units':'angstrom^2/fs; no conductivity normalization', 'threshold_scan':scan,
              'unprojected_6x6':sym[:6,:6].tolist(),
              'unprojected':{'b':sym[:3,:3].tolist(), 'J2':j_unprojected.tolist(),
                             'J2_minus_b':(j_unprojected-sym[:3,:3]).tolist()}}
    reasons = []
    if eig[0] < -rcond*scale:
        reasons.append('NEGATIVE_YY_SPECTRUM')
    if len(set(ranks)) != 1:
        reasons.append('SUPPORT_RANK_UNSTABLE')
    if max(scan[1]['null_cross'][name]['null_cross_ratio'] for name in ('b','d2','J2')) > null_rtol:
        reasons.append('NULL_CROSS_INCOMPATIBLE')
    if reasons:
        result.update(status='UNRESOLVED', reasons=reasons, projected=None)
        return result
    supported = eig > rcond*scale
    z = (xy @ vec[:,supported]) / np.sqrt(eig[supported])
    projected = sym[:6,:6] - z @ z.T
    z_j = direct_map @ z
    j_projected = direct_map @ sym[:6,:6] @ direct_map.T - z_j @ z_j.T
    blocks = {'bb':projected[:3,:3], 'bd':projected[:3,3:], 'db':projected[3:,:3], 'dd':projected[3:,3:]}
    block_sum = sum(blocks.values(), np.zeros((3,3)))
    result.update(status='FINITE_SAMPLE_SUPPORTED_ONLY', reasons=[],
                  projected={'Lperp_6x6':projected.tolist(), 'blocks':{k:v.tolist() for k,v in blocks.items()},
                             'b_projected':blocks['bb'].tolist(),
                             'J2_projected_direct':j_projected.tolist(),
                             'J2_projected_block_sum':block_sum.tolist(),
                             'block_identity_max_abs':float(np.max(np.abs(block_sum-j_projected))),
                             'J2_minus_b_block_identity_max_abs':float(np.max(np.abs((j_projected-blocks['bb'])-(blocks['bd']+blocks['db']+blocks['dd'])))),
                             'J2_minus_b_projected':(j_projected-blocks['bb']).tolist(),
                             'projection_change_in_J2_minus_b':((j_projected-blocks['bb'])-(j_unprojected-sym[:3,:3])).tolist()})
    return result


def run(directories, max_lag, rcond, null_rtol, number_intervals='trapezoid'):
    if not np.isfinite(rcond) or not 0 < rcond < 0.1 or not np.isfinite(null_rtol) or not 0 <= null_rtol <= 1:
        raise ValueError('invalid support rcond or null-cross tolerance')
    names = ('metadata.txt','source_identity.txt','observer.txt','scalar.txt','COMPLETE')
    initial_hashes = [{name:digest(Path(d)/name) for name in names} for d in directories]
    raw = analyze(directories, max_lag, number_intervals)
    if any(before != item['sha256'] for before,item in zip(initial_hashes,raw['inputs'])):
        raise ValueError('source changed during correlation calculation')
    checked = [composition(d, item['sha256']) for d,item in zip(directories,raw['inputs'])]
    if any(x[0] != checked[0][0] for x in checked[1:]):
        raise ValueError('capture atom identity or mass differs')
    if sum(x[3] for x in checked) != raw['sample_counts'][0]:
        raise ValueError('scalar sample domain mismatch')
    atoms, counts, masses = checked[0][:3]
    a, total_mass = transform(counts,masses)
    eh = raw['eh_regular_only']
    matrices = {'GK':raw['integrated_full_18x18_trapezoid'][-1],
                'EH':eh['eh_covariance_over_2t'], 'triangular':eh['triangular_correlation_integral']}
    return {'schema':'r164_composition_projection_v1',
            'status':'FINITE_WINDOW_DIAGNOSTIC_NOT_CONDUCTIVITY',
            'scope':'B_UNVERIFIED is a pointwise candidate, B_REGULAR_UNVERIFIED is an independent chart/image reconstruction; b_path, statistical support, physical events and P13 remain uncertified',
            'production_release':'NOT_PASSED', 'window_fs':max_lag*raw['dt_fs'],
            'window_scope':'GK uses point samples; EH and triangular use common interval-bin data and actual contiguous origins as recorded in raw_report', 'support_rcond':rcond, 'null_cross_rtol':null_rtol,
            'species_order':SPECIES, 'counts':counts, 'mass_amu':masses, 'total_mass_eV_fs2_per_A2':total_mass,
            'mass_flow_max_abs_eV_fs_per_A':max(x[4] for x in checked),
            'mass_flow_tolerance':'rtol 2e-13, atol 1e-10 eV fs/angstrom',
            'transform_Y_from_species':a[6:,6:].tolist(),
            'matrices':{name:matrix_diagnostic(value,a,rcond,null_rtol) for name,value in matrices.items()},
            'raw_report':raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--max-lag',required=True,type=int)
    parser.add_argument('--support-rcond',required=True,type=float)
    parser.add_argument('--null-cross-rtol',required=True,type=float)
    parser.add_argument('--number-intervals',choices=('trapezoid','lift'),default='trapezoid')
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    report = run(args.directories,args.max_lag,args.support_rcond,args.null_cross_rtol,args.number_intervals)
    encoded = json.dumps(report,indent=2,allow_nan=False)+'\n'
    with args.output.open('x',encoding='utf8',newline='\n') as output:
        output.write(encoded)


if __name__ == '__main__':
    main()
