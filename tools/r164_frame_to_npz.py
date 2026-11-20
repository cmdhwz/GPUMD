#!/usr/bin/env python3
"""Convert an R164 debug frame and metadata into an R161-style NPZ."""
import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('step', type=int)
    ap.add_argument('output')
    a = ap.parse_args()
    lines = (a.directory / 'metadata.txt').read_text(encoding='utf-8').splitlines()
    fields = {line.split()[0]: line.split()[1:] for line in lines if not line.startswith('atom ')}
    atoms = [line.split()[1:] for line in lines if line.startswith('atom ')]
    P, N = int(fields['P'][0]), int(fields['N'][0])
    if len(atoms) != N or [int(x[0]) for x in atoms] != list(range(N)):
        raise ValueError('atom metadata ordering mismatch')
    q = np.empty((P, N, 3)); p = np.empty_like(q); f = np.empty_like(q); u = np.empty(P)
    with (a.directory / f'frame_{a.step}.txt').open(encoding='utf-8') as source:
        for s in range(P):
            header=source.readline().split()
            if len(header)!=3 or header[:2] != ['bead',str(s)]: raise ValueError('invalid bead header')
            u[s]=float(header[2])
            for i in range(N):
                row=source.readline().split()
                if len(row)!=10 or int(row[0])!=i: raise ValueError('invalid atom row')
                values=np.array(row[1:],dtype=float)
                q[s,i],p[s,i],f[s,i]=values[:3],values[3:6],values[6:]
        if source.read().strip(): raise ValueError('trailing frame data')
    if not all(np.isfinite(x).all() for x in (q,p,f,u)): raise ValueError('nonfinite frame')
    matching=[]
    with (a.directory / 'scalar.txt').open(encoding='utf-8') as scalar:
        for line in scalar:
            if not line.endswith('\n'): raise ValueError('partial scalar row without newline')
            if line.startswith('#'): continue
            columns=line.split()
            if columns and int(columns[0]) == a.step: matching.append(columns)
    if len(matching) != 1: raise ValueError('missing or duplicate scalar frame identity')
    time_fs=float(matching[0][1]); dt_fs=float(fields['dt_fs'][0])
    if (not np.isfinite(time_fs) or not np.isfinite(dt_fs) or dt_fs <= 0
        or not np.isclose(time_fs, a.step * dt_fs, rtol=1e-12, atol=1e-10)):
        raise ValueError('invalid or aliased frame time/step')
    with open(a.output,'xb') as target:
        np.savez_compressed(target,q_A=q,p_eV_fs_per_A=p,F_eV_per_A=f,U_eV=u,
            mass=np.array([float(x[2]) for x in atoms])*10.18051**2,
            time_fs=time_fs,dt_fs=dt_fs,step=a.step,
            species=np.array([x[1] for x in atoms]),cell=np.diag(np.array(fields['cell_angstrom'],float)),
            atom_row_zero_based=np.arange(N),bead_order_zero_based=np.arange(P))
if __name__=='__main__': main()
