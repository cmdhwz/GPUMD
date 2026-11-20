#!/usr/bin/env python3
"""Draw independent Maxwell bead momenta for a diagnostic position archive."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

K_B = 8.617343e-5
TIME_UNIT_CONVERSION = 10.18051


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sample(phase, expected_sha, temperature, seed, trajectory_id, destination):
    phase = Path(phase).resolve()
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha) or digest(phase) != expected_sha.lower():
        raise ValueError('position archive SHA mismatch')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not isinstance(seed, int) or not 0 <= seed < 2**128:
        raise ValueError('seed must be a nonnegative 128-bit integer')
    if not trajectory_id or not trajectory_id.strip():
        raise ValueError('trajectory identity is required')
    with np.load(phase, allow_pickle=False) as data:
        raw_q, raw_mass, raw_cell = data['q_A'], data['mass'], data['cell']
        if any(x.dtype.kind not in ('f','i','u') for x in (raw_q,raw_mass,raw_cell)):
            raise ValueError('position, mass and cell must be real numeric arrays')
        q = np.asarray(raw_q, dtype=np.float64)
        mass = np.asarray(raw_mass, dtype=np.float64)
        species = np.asarray(data['species'])
        cell = np.asarray(raw_cell, dtype=np.float64)
        if any(not np.array_equal(converted.astype(original.dtype), original)
               for converted,original in ((q,raw_q),(mass,raw_mass),(cell,raw_cell))):
            raise ValueError('input values cannot be represented exactly in binary64')
        atom_rows = np.asarray(data['atom_row_zero_based'])
        bead_rows = np.asarray(data['bead_order_zero_based'])
    if q.ndim != 3 or q.shape[2] != 3:
        raise ValueError('invalid q shape')
    P, N, _ = q.shape
    if (P < 2 or P > 128 or P % 2 != 0 or N < 1 or P*N*3 > 2147483647 or mass.shape != (N,) or species.shape != (N,) or cell.shape != (3,3)
            or atom_rows.dtype.kind not in ('i','u') or bead_rows.dtype.kind not in ('i','u')
            or not np.array_equal(atom_rows,np.arange(N)) or not np.array_equal(bead_rows,np.arange(P))
            or not np.isfinite(q).all() or not np.isfinite(mass).all() or np.any(mass <= 0)
            or not np.isfinite(cell).all()):
        raise ValueError('invalid position, mass, cell or row order')
    if species.dtype.kind not in ('U','S'):
        raise ValueError('species must be strings')
    species = species.astype(str)
    if any(x not in ('O','H','Na','Cl') for x in species):
        raise ValueError('unsupported species')
    length = np.diag(cell)
    if not np.array_equal(cell,np.diag(length)) or np.any(length<=0):
        raise ValueError('requires positive axis-aligned orthogonal cell')
    if np.any(q < -length) or np.any(q > 2*length):
        raise ValueError('position outside one-wrap input range')
    if np.any(np.abs(q-q[0:1]) >= length/2-1e-9):
        raise ValueError('ambiguous first-bead MIC branch')
    amu = mass/TIME_UNIT_CONVERSION**2
    if not np.isfinite(amu).all() or any(float(format(x,'.4f')) <= 0 for x in amu):
        raise ValueError('mass cannot be represented by restart format')
    sigma = np.sqrt(mass * P * K_B * temperature)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('nonfinite Maxwell scale')
    momentum = np.random.Generator(np.random.PCG64(seed)).standard_normal((P,N,3))*sigma[None,:,None]
    if not np.isfinite(momentum).all():
        raise ValueError('nonfinite sampled momentum')
    kinetic = float(np.sum(momentum*momentum/(2*mass[None,:,None])))
    total_momentum = np.sum(momentum,axis=(0,1))/P
    beta_p = 1/(P*K_B*temperature)
    theory_k = 1.5*N*P*P*K_B*temperature
    momentum_std = np.sqrt(np.sum(mass)*K_B*temperature)
    if not np.isfinite(kinetic) or not np.isfinite(beta_p) or not np.isfinite(theory_k) or not np.isfinite(total_momentum).all() or not np.isfinite(momentum_std):
        raise ValueError('nonfinite diagnostic moment')
    if digest(phase) != expected_sha.lower():
        raise ValueError('position archive changed during read')
    destination = Path(destination).resolve()
    destination.mkdir()
    output = destination/'phase.npz'
    np.savez(output,q_A=q,p_eV_fs_per_A=momentum,mass=mass,species=species,cell=cell,
             atom_row_zero_based=atom_rows,bead_order_zero_based=bead_rows)
    subprocess.run([sys.executable,str(Path(__file__).with_name('r164_phase_to_restart.py')),
                    str(output),str(destination/'restart_beads.xyz'),'--temperature',str(temperature),
                    '--model-output',str(destination/'model.xyz')],check=True)
    if digest(phase) != expected_sha.lower():
        raise ValueError('position archive changed before receipt')
    receipt = {
        'schema':'r164_maxwell_diagnostic_v1',
        'scope':'development_only_no_equilibrium_independence_production_or_run_card',
        'position_identity':{'archive_path':str(phase),'archive_sha256':expected_sha.lower(),
                             'P':P,'N':N,'cell_angstrom':cell.tolist(),
                             'atom_row_zero_based':atom_rows.tolist(),'bead_order_zero_based':bead_rows.tolist(),
                             'species':species.tolist(),'mass_eV_fs2_per_A2':mass.tolist(),
                             'qualification':'UNVERIFIED_P12_POSITION_AND_INDEPENDENCE'},
        'sampling':{'generator':'numpy.random.Generator(PCG64)', 'numpy_version':np.__version__,
                    'seed_decimal':str(seed),'trajectory_id':trajectory_id,'temperature_K':temperature,
                    'k_B_eV_per_K':K_B,'beta_P_per_eV':beta_p,'formula':'p=sqrt(m*P*k_B*T)*standard_normal',
                    'constraints':'none; no COM projection or kinetic-energy rescaling',
                    'source_momentum':'not_read'},
        'kinetic':{'K_P_eV':kinetic,'K_P_over_P_eV':kinetic/P,
                   'expected_K_P_eV':theory_k,'expected_K_P_over_P_eV':theory_k/P,
                   'P_cm_eV_fs_per_A':total_momentum.tolist(),
                   'P_cm_component_expected_std_eV_fs_per_A':float(momentum_std)},
        'output_sha256':{name:digest(destination/name) for name in ('phase.npz','model.xyz','restart_beads.xyz')},
        'future_execution':'NOT_VERIFIED',
    }
    temporary=destination/'receipt.json.tmp'
    temporary.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(destination/'receipt.json')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    parser.add_argument('--phase-sha256',required=True)
    parser.add_argument('--temperature',required=True,type=float)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--trajectory-id',required=True)
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    sample(args.phase,args.phase_sha256,args.temperature,args.seed,args.trajectory_id,args.destination)


if __name__=='__main__':
    main()
