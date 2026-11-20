#!/usr/bin/env python3
"""Replay one private-observer projection readout; no physical-event inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from r164_frame_to_npz import read_frame
from r164_verify_short_runs import keyfile, sha, labelled

TIME_UNIT=10.18051
GUARD=1e-9


def replay_projection(x, raw_D, raw_rate, cell, stop_on_guard=True):
    n=len(raw_D)
    mean_D=sum(float(v) for v in (raw_D/np.float32(n)))
    mean_rate=sum(float(v) for v in (raw_rate/np.float32(n)))
    c=mean_D*raw_rate.astype(float)-raw_D.astype(float)*mean_rate
    projection=np.zeros(3); image=np.zeros(3); first_guard=None
    for d in range(3):
        ordered=np.sort(x[:,d]); mean_x=sum(float(v)/n for v in x[:,d]); L=cell[d]
        for i in range(n):
            left=x[i,d]-L/2; right=x[i,d]+L/2
            li=np.searchsorted(ordered,left,side='left'); ri=np.searchsorted(ordered,right,side='right')
            left_near=np.searchsorted(ordered,left-GUARD,side='left')
            right_near=np.searchsorted(ordered,right-GUARD,side='left')
            if ((left_near<n and ordered[left_near]<=left+GUARD) or
                    (right_near<n and ordered[right_near]<=right+GUARD)):
                if first_guard is None: first_guard=(d,i)
                if stop_on_guard: break
            image_term=L*(n-ri-li)/n
            projection[d]+=(x[i,d]-mean_x+image_term)*c[i]/TIME_UNIT
            image[d]+=image_term*c[i]/TIME_UNIT
        if first_guard is not None and stop_on_guard: break
    return projection,image,c,first_guard


def readout(directory,step,pair=None):
    if step<0: raise ValueError('step must be nonnegative')
    d=Path(directory).resolve(); path=d/f'observer_projection_{step}.txt'
    header={}; rows=[]
    for line in path.read_text(encoding='utf8').splitlines():
        if line.startswith('#'):
            parts=line[1:].strip().split(maxsplit=1)
            if len(parts)==2:
                if parts[0] in header: raise ValueError('duplicate projection header')
                header[parts[0]]=parts[1]
        elif line.strip():
            rows.append(line.split())
    if (header.get('schema')!='r164_observer_projection_v1' or header.get('step')!=str(step)
            or header.get('observer_force_call')!=str(step+1)
            or header.get('columns')!='atom centroid_canonical_xyz_A raw_D_fp32_eV_per_e raw_rate_fp32_e_per_natural_time'):
        raise ValueError('projection identity mismatch')
    meta=keyfile(d/'metadata.txt'); P=int(meta['P']); N=int(meta['N']); dt=float(meta['dt_fs']); cell=np.array(meta['cell_angstrom'].split(),float)
    if (meta.get('schema')!='r164_diagnostic_v1' or N<=0 or P<2 or cell.shape!=(3,) or not np.isfinite(cell).all() or np.any(cell<=0)
            or not np.isfinite(dt) or dt<=0 or int(header['N'])!=N
            or float(header['time_unit_conversion'])!=TIME_UNIT or float(header['half_box_guard_A'])!=GUARD
            or not np.isclose(float(header['time_fs']),step*dt,rtol=1e-12,atol=1e-10)):
        raise ValueError('invalid projection dimensions, units or time')
    for name,file in (('metadata','metadata.txt'),('source_identity','source_identity.txt'),('observer','observer.txt'),
                      ('frame',f'frame_{step}.txt'),('observer_frame',f'observer_frame_{step}.txt')):
        if sha(d/file)!=header.get(name+'_sha256'): raise ValueError('projection source SHA mismatch: '+name)
    if (d/'observer.txt').read_text(encoding='utf8').splitlines()[0]!='native_private_qnep_pppm_observer':
        raise ValueError('unsupported observer backend')
    observer_line=(d/f'observer_frame_{step}.txt').read_text(encoding='utf8').splitlines()[0].split()
    if len(observer_line)<6 or observer_line[:6]!=['#','step',str(step),'force_call',str(step+1),'mesh']:
        raise ValueError('observer frame call identity mismatch')
    scalar=[]
    with (d/'scalar.txt').open('rb') as source:
        for line in source:
            if line.startswith(b'#'): continue
            if not line.endswith(b'\n'): raise ValueError('partial scalar row')
            fields=line.split()
            if fields and int(fields[0])==step: scalar.append(line.rstrip(b'\r\n')+b'\n')
    if len(scalar)!=1 or hashlib.sha256(scalar[0]).hexdigest()!=header.get('scalar_line_sha256'):
        raise ValueError('projection scalar SHA mismatch')
    tokens=scalar[0].decode('utf8').split()
    if len(tokens)<2 or not np.isfinite(float(tokens[1])) or float(tokens[1])!=float(header['time_fs']):
        raise ValueError('scalar and projection time mismatch')
    value=labelled(tokens,'PROJECTION_UNVERIFIED',3)
    image=labelled(tokens,'PROJECTION_IMAGE_UNVERIFIED',3)
    status=header.get('status')
    if status not in ('valid','half_box_guard','nonfinite','charge_diagnostics_unavailable'):
        raise ValueError('invalid projection status')
    if status=='valid':
        if value is None or image is None or 'PROJECTION_UNAVAILABLE' in tokens or 'PROJECTION_IMAGE_UNAVAILABLE' in tokens:
            raise ValueError('projection status and scalar disagree')
    elif (value is not None or image is not None or tokens.count('PROJECTION_UNAVAILABLE')!=1
          or tokens.count('PROJECTION_IMAGE_UNAVAILABLE')!=1):
        raise ValueError('projection unavailable status and scalar disagree')
    if status!='half_box_guard' and (int(header['guard_axis']),int(header['guard_index']))!=(-1,-1):
        raise ValueError('unexpected guard metadata')
    count=int(header['rows'])
    if count not in (0,N) or len(rows)!=count or (status=='charge_diagnostics_unavailable')!=(count==0):
        raise ValueError('projection row count/status mismatch')
    result={'schema':'r164_projection_readout_v1','scope':'private observer projection diagnostic only; no impulse or physical event first moment',
            'step':step,'time_fs':float(header['time_fs']),'status':status,'projection_file_sha256':sha(path),
            'bound_source_sha256':{key:value for key,value in header.items() if key.endswith('_sha256')},
            'rows':count,'recorded_projection':None if value is None else value.tolist(),
            'recorded_projection_image':None if image is None else image.tolist()}
    if not count:
        if pair is not None: raise ValueError('pair comparison requires raw rows')
        return result
    if any(len(row)!=6 or int(row[0])!=i for i,row in enumerate(rows)):
        raise ValueError('invalid projection row order')
    data=np.array([row[1:] for row in rows],float)
    x=data[:,:3]; raw=data[:,3:]
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x>cell):
        raise ValueError('invalid canonical centroid')
    q,_,_,_=read_frame(d/f'frame_{step}.txt',P,N)
    centroid=np.zeros((N,3))
    for bead in range(P): centroid+=q[bead]/P
    centroid-=np.floor(centroid/cell)*cell
    if not np.allclose(x,centroid,rtol=0,atol=2e-12): raise ValueError('projection centroid differs from frame')
    if status=='nonfinite':
        if pair is not None: raise ValueError('pair comparison requires finite values')
        return result
    if not np.isfinite(raw).all() or not np.array_equal(raw,raw.astype(np.float32).astype(float)):
        raise ValueError('raw values are not finite float32 source values')
    calculated,calculated_image,c,guard=replay_projection(x,raw[:,0].astype(np.float32),raw[:,1].astype(np.float32),cell)
    if status=='half_box_guard':
        if guard!=(int(header['guard_axis']),int(header['guard_index'])): raise ValueError('guard location mismatch')
        result['guard']={'axis':'xyz'[guard[0]],'center_index':guard[1]}
        branch,branch_image,_,_=replay_projection(x,raw[:,0].astype(np.float32),raw[:,1].astype(np.float32),cell,False)
        result['unprotected_strict_branch_candidate']=branch.tolist()
        result['unprotected_strict_branch_image_candidate']=branch_image.tolist()
        result['unprotected_scope']='computed ignoring guard; does not replace unavailable scalar'
    elif guard is not None or (int(header['guard_axis']),int(header['guard_index']))!=(-1,-1):
        raise ValueError('unexpected projection guard')
    else:
        if not np.allclose(calculated,value,rtol=1e-10,atol=1e-12) or not np.allclose(calculated_image,image,rtol=1e-10,atol=1e-12):
            raise ValueError('projection replay residual exceeds arithmetic tolerance')
        result['arithmetic_replay_tolerance']='rtol 1e-10, atol 1e-12 eV angstrom/fs; not a physical threshold'
        result['replayed_projection']=calculated.tolist(); result['replayed_projection_image']=calculated_image.tolist()
        result['projection_residual']=(calculated-value).tolist(); result['projection_image_residual']=(calculated_image-image).tolist()
    if pair is not None:
        i,j,axis=pair; axis='xyz'.index(axis)
        if i==j or not (0<=i<N and 0<=j<N): raise ValueError('invalid projection pair')
        delta=x[j,axis]-x[i,axis]
        if abs(abs(delta)-cell[axis]/2)>GUARD: raise ValueError('pair is not at the guarded half-box face')
        result['frozen_pair_jump']={'i':i,'j':j,'axis':'xyz'[axis], 'raw_separation_A':float(delta),
                                    'for_increasing_xj_minus_xi_eV_A_per_fs':float(cell[axis]/N*(c[i]-c[j])/TIME_UNIT),
                                    'scope':'fixed q and raw charge tangent projection branch only; not an impulse or physical event'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path); p.add_argument('--step',required=True,type=int)
    p.add_argument('--pair',nargs=3,metavar=('I','J','AXIS')); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); pair=None
    if a.pair:
        if a.pair[2] not in ('x','y','z'): p.error('axis must be x, y or z')
        pair=(int(a.pair[0]),int(a.pair[1]),a.pair[2])
    result=readout(a.directory,a.step,pair)
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    with a.output.open('x',encoding='utf8',newline='\n') as out: out.write(encoded)

if __name__=='__main__': main()
