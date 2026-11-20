#!/usr/bin/env python3
"""Integrate the observer projection under a piecewise-linear endpoint-c model."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from r164_projection_readout import readout
from r164_frame_to_npz import read_frame
from r164_verify_short_runs import keyfile, load_run, sha


def eta(x, cell):
    n = len(x)
    result = np.empty_like(x)
    for axis in range(3):
        values = x[:, axis]
        ordered = np.sort(values)
        left = np.searchsorted(ordered, values-cell[axis]/2, side='left')
        right = np.searchsorted(ordered, values+cell[axis]/2, side='right')
        result[:, axis] = values - sum(float(v)/n for v in values) + cell[axis]*(n-right-left)/n
    return result


def integrate_interval(x0, x1, r0, r1, c0, c1, cell, h, eps=1e-10):
    n = len(x0)
    e0, e1 = eta(x0, cell), eta(x1, cell)
    displacement = r1-r0
    correction = np.zeros(3)
    predicted_jump = np.zeros_like(x0)
    roots = []
    unresolved = []
    for axis, L in enumerate(cell):
        delta = displacement[:, axis]
        max_delta = np.max(np.abs(delta))
        if 2*max_delta+eps >= L/2:
            unresolved.append({'axis':axis, 'reason':'displacement exceeds bounded candidate search'})
            continue
        values = x0[:, axis]
        extended = np.concatenate([values-L, values, values+L, values+2*L])
        indices = np.tile(np.arange(n), 4)
        order = np.argsort(extended)
        extended, indices = extended[order], indices[order]
        for i in range(n):
            width = abs(delta[i])+max_delta+eps
            target = values[i]+L/2
            lo = np.searchsorted(extended, target-width, side='left')
            hi = np.searchsorted(extended, target+width, side='right')
            for j in set(int(v) for v in indices[lo:hi] if v > i):
                start = r0[j, axis]-r0[i, axis]
                end = r1[j, axis]-r1[i, axis]
                first = int(np.floor(min(start,end)/L-0.5))-1
                last = int(np.ceil(max(start,end)/L-0.5))+1
                for k in range(first,last+1):
                    face = (k+0.5)*L
                    if min(abs(start-face),abs(end-face)) <= eps:
                        unresolved.append({'i':i,'j':j,'axis':axis,'face_A':face,'reason':'endpoint near face'})
                    elif (start-face)*(end-face) < 0:
                        fraction = (face-start)/(end-start)
                        sign = 1 if end>start else -1
                        jump = L/n*sign
                        predicted_jump[i,axis] += jump
                        predicted_jump[j,axis] -= jump
                        dc0 = c0[i]-c0[j]
                        dc = (c1[i]-c0[i])-(c1[j]-c0[j])
                        # Subtracting the eta chord leaves jump * (H(u-r) - u).
                        correction[axis] += h*jump*((0.5-fraction)*dc0+(1/6-fraction*fraction/2)*dc)
                        roots.append({'i':i,'j':j,'axis':'xyz'[axis],'face_lift_A':face,'fraction':fraction,'direction':sign})
    actual_jump = e1-e0-displacement+np.mean(displacement,axis=0)
    residual = actual_jump-predicted_jump
    net_error = float(np.max(np.abs(residual)))
    if net_error > 1e-8:
        unresolved.append({'reason':'endpoint eta net-jump mismatch','max_A':net_error})
    baseline = h/6*np.sum(2*e0*c0[:,None]+e0*c1[:,None]+e1*c0[:,None]+2*e1*c1[:,None],axis=0)
    return {'status':'unavailable' if unresolved else 'valid', 'root_count':len(roots), 'roots':roots,
            'unresolved':unresolved, 'eta_net_jump_max_residual_A':net_error,
            'eta_chord_integral_eV_A':baseline.tolist(), 'root_correction_eV_A':correction.tolist(),
            'linear_c_model_integral_eV_A':None if unresolved else (baseline+correction).tolist()}


def analyze(directory, run_in):
    directory = Path(directory).resolve()
    run_in = Path(run_in).resolve()
    meta = keyfile(directory/'metadata.txt')
    dt = float(meta['dt_fs'])
    marker = (directory/'COMPLETE').read_text(encoding='utf8').split()
    if len(marker)!=3 or marker[:2]!=['r164_diagnostic_v1','final_step']:
        raise ValueError('incomplete capture')
    steps = int(marker[2])
    if steps<1 or not np.isfinite(dt) or dt<=0: raise ValueError('invalid interval count or time step')
    run = load_run(directory,run_in,dt,steps,True)
    commands = {row[0]:row for row in run['config']}
    lines = [line.split('#',1)[0].split() for line in run_in.read_text(encoding='utf8').splitlines()]
    r164 = next(line for line in lines if line and line[0]=='r164_rpmd')
    if (run_in.parent/r164[1]).resolve()!=directory: raise ValueError('capture path differs from run card')
    src=run['source']
    for path,key in ((run_in.parent/commands['potential'][1],'model_sha256'),
                     (run_in.parent/'model.xyz','model_xyz_sha256'),
                     (run_in.parent/commands['read_pimd_restart'][1],'restart_input_sha256')):
        if sha(path).lower()!=src[key].lower(): raise ValueError('input SHA mismatch: '+key)
    with (run_in.parent/'model.xyz').open(encoding='utf8') as model_file:
        model_file.readline()
        model_header=model_file.readline()
    lattice=re.search(r'Lattice="([^"]+)"',model_header)
    pbc=re.search(r'pbc="([^"]+)"',model_header)
    if not lattice or not pbc or pbc.group(1).split()!=['T','T','T']:
        raise ValueError('requires fully periodic orthogonal cell')
    matrix=np.array(lattice.group(1).split(),float).reshape(3,3)
    if not np.array_equal(matrix,np.diag(run['cell'])):
        raise ValueError('requires matching axis-aligned cell')
    observer=keyfile(directory/'observer.txt')
    if observer.get('run_in_sha256','').lower()!=src['run_in_sha256'].lower():
        raise ValueError('observer run.in SHA mismatch')
    mass=run['mass']; cell=run['cell']; intervals=[]; endpoints=[]
    previous=None
    for step in range(steps+1):
        path=directory/f'observer_projection_{step}.txt'
        if path.exists():
            result,data=readout(directory,step,_return_data=True)
        else:
            result={'step':step,'status':'missing_projection_file'}; data=None
        endpoints.append(result)
        if 'bound_source_sha256' in result:
            bound=result['bound_source_sha256']
            if (bound['frame_sha256']!=run['frame_sha256'][step] or
                    bound['metadata_sha256']!=run['inputs']['metadata.txt'] or
                    bound['source_identity_sha256']!=run['inputs']['source_identity.txt'] or
                    bound['observer_sha256']!=run['observer_sha256']):
                raise ValueError('endpoint provenance changed during analysis')
        if data is None:
            q,p,F,_=read_frame(run['frames'][step],32,5896)
            data={}
        else:
            q,p,F=data.pop('q'),data.pop('p'),data.pop('F')
            if sha(run['frames'][step])!=bound['frame_sha256']:
                raise ValueError('frame changed while reading endpoint')
        data['lift']=np.mean(q,axis=0)
        data['drift']=np.mean((p+dt*F/2)/mass[None,:,None],axis=0)
        current=(result,data)
        if previous is not None:
            before,old=previous
            record={'step':step,'start_step':step-1,'time_start_fs':(step-1)*dt,'time_end_fs':step*dt}
            drift_error=float(np.max(np.abs(data['lift']-old['lift']-dt*old['drift'])))
            record['centroid_free_drift_max_error_A']=drift_error
            if not np.isfinite(drift_error) or drift_error>=1e-8: raise ValueError('centroid free drift mismatch')
            if before['status']!='valid' or result['status']!='valid':
                record.update(status='unavailable',reason='missing or invalid endpoint projection')
            else:
                record.update(integrate_interval(old['x'],data['x'],old['lift'],data['lift'],old['c'],data['c'],cell,dt))
                record['native_scalar_trapezoid_eV_A']=(dt/2*(np.array(before['recorded_projection'])+np.array(result['recorded_projection']))).tolist()
            intervals.append(record)
        previous=current
    complete=all(x['status']=='valid' for x in intervals)
    total=None
    if complete:
        keys=('native_scalar_trapezoid_eV_A','eta_chord_integral_eV_A',
              'root_correction_eV_A','linear_c_model_integral_eV_A')
        total={key:np.sum([row[key] for row in intervals],axis=0).tolist() for key in keys}
        total['model_minus_trapezoid_eV_A']=(np.array(total[keys[3]])-np.array(total[keys[0]])).tolist()
    return {'schema':'r164_projection_interval_v1','scope':'observer projection only; endpoint c linear model, not measured continuous c, impulse or physical event first moment',
            'capture':str(directory),'run_in':str(run_in),'input_sha256':run['inputs'],
            'potential_sha256':src['model_sha256'],'model_xyz_sha256':src['model_xyz_sha256'],
            'restart_sha256':src['restart_input_sha256'],
            'backend':src['bead_force_backend'],'frame_sha256':run['frame_sha256'],
            'force_image_sha256':run['image_sha256'],'dt_fs':dt,'steps':steps,'endpoints':endpoints,
            'intervals':intervals,'complete_window_totals':total,
            'summation_scope':'only supplied when every interval is valid'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--run-in',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.capture,args.run_in)
    with args.output.open('x',encoding='utf8',newline='\n') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')


if __name__=='__main__':
    main()
