#!/usr/bin/env python3
"""Locate a native PPPM assignment-center change on one recorded RPMD drift."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from r164_frame_to_npz import read_frame
from r164_verify_short_runs import card, keyfile, read_scalars, read_images, sha, validate_frame_scalars, check_observer_mode

C=10.18051
KB=8.617343e-5
HBAR=6.465412e-2


def normal_matrix(P):
    T=np.empty((P,P))
    for j in range(1,P+1):
        for k in range(P):
            if k==0: T[j-1,k]=np.sqrt(1/P)
            elif k<P//2: T[j-1,k]=np.sqrt(2/P)*np.cos(2*np.pi*j*k/P)
            elif k==P//2: T[j-1,k]=np.sqrt(1/P)*((-1)**j)
            else: T[j-1,k]=np.sqrt(2/P)*np.sin(2*np.pi*j*k/P)
    return T


def center(x):
    return int(np.float32(np.float32(x)+np.float32(0.5)))


def fp32_boundary(lower_index):
    if lower_index < 0: raise ValueError('unsupported negative canonical mesh center')
    x=np.float32(lower_index+0.5)
    for _ in range(8):
        if center(x)<=lower_index: break
        x=np.nextafter(x,np.float32(-np.inf),dtype=np.float32)
    for _ in range(8):
        y=np.nextafter(x,np.float32(np.inf),dtype=np.float32)
        if center(x)==lower_index and center(y)==lower_index+1:
            return float((float(x)+float(y))/2),float(x),float(y)
        x=y
    raise ValueError('cannot resolve adjacent FP32 transition')


def lifted_label(q,image,L,K):
    canonical=q-image*L
    if canonical < -1e-8 or canonical > L+1e-8: raise ValueError('noncanonical force coordinate')
    s=np.float32((1.0/L)*canonical*K)
    return K*image+center(s)


def mode_state(Q,V,omega,t):
    wt=omega*t; co=np.cos(wt); si=np.sin(wt)
    factor=np.empty_like(omega); factor[0]=t; factor[1:]=si[1:]/omega[1:]
    return co[:,None,None]*Q+factor[:,None,None]*V, -omega[:,None,None]*si[:,None,None]*Q+co[:,None,None]*V


def build_modes(q,p,F,mass,dt,T,temperature):
    P=q.shape[0]; velocity=(p+dt*F/2)/mass[None,:,None]
    Q=np.einsum('sk,sia->kia',T,q); V=np.einsum('sk,sia->kia',T,velocity)
    omega=2*P*KB*temperature/(HBAR*C)*np.sin(np.arange(P)*np.pi/P); omega[0]=0
    return Q,V,omega


def reconstruct(q,p,F0,F1,mass,dt,T,temperature):
    Q,V,omega=build_modes(q,p,F0,mass,dt,T,temperature)
    qt,vt=mode_state(Q,V,omega,dt)
    q1=np.einsum('sk,kia->sia',T,qt)
    p1=np.einsum('sk,kia->sia',T,vt)*mass[None,:,None]+dt*F1/2
    return Q,V,omega,q1,p1


def validate_endpoint(qhat,phat,q1,p1):
    qerr=float(np.max(np.abs(qhat-q1))); perr=float(np.max(np.abs(phat-p1)))
    if not np.isfinite(qerr+perr) or qerr>1e-8 or perr>1e-8:
        raise ValueError('full-array endpoint replay exceeds arithmetic tolerance')
    return qerr,perr


def by_filename(lines,keyword):
    return next(line[1] for line in lines if line[0]==keyword)

def validate_inputs(source,run_directory,lines):
    if source.get('bead_force_backend')!='native_GPUMD': raise ValueError('only native GPUMD mesh precision is supported')
    by={x[0]:x for x in lines}
    for field,path in [('model_sha256',run_directory/by['potential'][1]),('model_xyz_sha256',run_directory/'model.xyz'),('restart_input_sha256',run_directory/by['read_pimd_restart'][1])]:
        if not re.fullmatch('[0-9a-fA-F]{64}',source.get(field,'')) or sha(path).lower()!=source[field].lower():
            raise ValueError('source input SHA mismatch: '+field)

def locate_one(Q,V,omega,T,bead,atom,axis,dt,L,K,image0,image1):
    weight=T[bead]; qk=Q[:,atom,axis]; vk=V[:,atom,axis]
    def state(t):
        wt=omega*t; co=np.cos(wt); si=np.sin(wt)
        factor=np.empty_like(omega); factor[0]=t; factor[1:]=si[1:]/omega[1:]
        return float(np.dot(weight,co*qk+factor*vk)),float(np.dot(weight,-omega*si*qk+co*vk))
    q0,v0=state(0); q1,_=state(dt)
    label0=lifted_label(q0,image0,L,K); label1=lifted_label(q1,image1,L,K)
    if abs(label1-label0)!=1: raise ValueError('endpoint centers do not differ by exactly one')
    bound=float(np.sum(np.abs(weight)*(np.abs(qk)*omega**2+np.abs(vk)*omega)))
    if abs(v0)<=bound*dt: raise ValueError('monotonicity cannot be proved on this interval')
    direction=1 if v0>0 else -1
    if direction*(q1-q0)<=0 or label1-label0!=direction: raise ValueError('center change inconsistent with monotonic direction')
    low=min(label0,label1)-K*image0
    threshold,lo,hi=fp32_boundary(low)
    edge=threshold*L/K+image0*L; nominal=(low+0.5)*L/K+image0*L
    if direction*(q0-edge)>=0 or direction*(q1-edge)<=0: raise ValueError('FP32 boundary not strictly bracketed')
    left,right=0.,dt
    for _ in range(80):
        mid=(left+right)/2
        if direction*(state(mid)[0]-edge)<0: left=mid
        else: right=mid
    root=(left+right)/2; side=min(root,dt-root)/4
    if not side>0: raise ValueError('no room for two-sided sample')
    samples=[]
    for t in (root-side,root,root+side):
        q,v=state(t)
        # Endpoint chart is locally unchanged for this root; reject any wrap ambiguity.
        label=lifted_label(q,image0,L,K)
        samples.append({'time_from_step_fs':t,'q_A':q,'v_A_per_fs':v,'lifted_center':label})
    if samples[0]['lifted_center']!=label0 or samples[2]['lifted_center']!=label1:
        raise ValueError('two-sided center identity failed')
    return {'initial_center':label0,'final_center':label1,'direction':direction,
            'fp32_low':lo,'fp32_high':hi,'threshold_scaled':threshold,'threshold_lift_A':edge,
            'nominal_half_grid_lift_A':nominal,'threshold_minus_nominal_A':edge-nominal,
            'acceleration_bound_A_per_fs2':bound,'initial_velocity_A_per_fs':v0,
            'velocity_lower_bound_A_per_fs':abs(v0)-bound*dt,'root_bracket_fs':[left,right],
            'root_from_step_fs':root,'side_delta_fs':side,'samples':samples}


def stdout_mesh(path,dt):
    text=path.read_text(encoding='utf8',errors='strict')
    matches=re.findall(r'PPPM mesh:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*\(target spacing\s+([0-9.eE+-]+)\s+A;',text)
    parsed={(int(a),int(b),int(c),float(spacing)) for a,b,c,spacing in matches}
    if len(parsed)!=1: raise ValueError('missing or inconsistent stdout PPPM mesh')
    a,b,c,spacing=next(iter(parsed))
    if spacing!=0.5 or 'Number of atoms is 5896.' not in text or 'number of beads is 32.' not in text or f'Time step for this run is {dt:g} fs.' not in text or 'Finished running GPUMD.' not in text:
        raise ValueError('stdout does not match required run facts')
    return [int(a),int(b),int(c)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path); ap.add_argument('--run-in',required=True,type=Path); ap.add_argument('--stdout',required=True,type=Path)
    ap.add_argument('--step',required=True,type=int); ap.add_argument('--bead',required=True,type=int); ap.add_argument('--atom',required=True,type=int)
    ap.add_argument('--axis',required=True,choices=('x','y','z')); ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args(); directory=a.capture.resolve(); runin=a.run_in.resolve(); stdout=a.stdout.resolve()
    if directory.parent!=runin.parent or stdout.parent!=runin.parent: raise ValueError('capture, run.in and stdout must be from the same run directory')
    meta=keyfile(directory/'metadata.txt'); source=keyfile(directory/'source_identity.txt')
    if meta.get('schema')!='r164_diagnostic_v1' or meta.get('time_convention')!='step_times_fixed_dt_start_zero': raise ValueError('invalid capture schema')
    marker=(directory/'COMPLETE').read_text(encoding='utf8').split()
    if len(marker)!=3 or marker[:2]!=['r164_diagnostic_v1','final_step']: raise ValueError('incomplete capture')
    final=int(marker[2]); dt=float(meta['dt_fs']); temperature=float(meta['temperature_K']); P=int(meta['P']); N=int(meta['N'])
    if (P,N,temperature)!=(32,5896,300.) or (dt,final) not in ((0.1,10),(0.2,5)) or not (0<=a.step<final and 0<=a.bead<P and 0<=a.atom<N): raise ValueError('unsupported capture or indices')
    lines=[line.split('#',1)[0].split() for line in runin.read_text(encoding='utf8').splitlines()]; lines=[x for x in lines if x]
    observer=any(x[0]=='r164_rpmd' and len(x)==4 and x[3]=='observer' for x in lines)
    card(runin,source,dt,final,observer); records=read_scalars(directory/'scalar.txt',dt,final)
    check_observer_mode(directory,records,observer)
    mesh=stdout_mesh(stdout,dt)
    if any(x<=0 for x in mesh): raise ValueError('invalid mesh dimensions')
    validate_inputs(source,runin.parent,lines)
    atom_rows=[line.split()[1:] for line in (directory/'metadata.txt').read_text(encoding='utf8').splitlines() if line.startswith('atom ')]
    if len(atom_rows)!=N or [int(x[0]) for x in atom_rows]!=list(range(N)): raise ValueError('invalid atom order')
    mass=np.array([float(x[2])*C*C for x in atom_rows]); species=np.array([x[1] for x in atom_rows]); cell=np.array(meta['cell_angstrom'].split(),float)
    if not np.isfinite(mass).all() or np.any(mass<=0) or cell.shape!=(3,) or not np.allclose(cell,29.3104,atol=1e-8,rtol=0): raise ValueError('invalid masses or cell')
    files=[directory/f'frame_{i}.txt' for i in (a.step,a.step+1)]; images=[directory/f'force_images_{i}.txt' for i in (a.step,a.step+1)]
    q0,p0,F0,U0=read_frame(files[0],P,N); q1,p1,F1,U1=read_frame(files[1],P,N)
    validate_frame_scalars(q0,p0,U0,mass,species,temperature,records[a.step])
    validate_frame_scalars(q1,p1,U1,mass,species,temperature,records[a.step+1])
    img0=read_images(images[0],q0,cell); img1=read_images(images[1],q1,cell)
    for q in (q0,q1):
        if np.any(np.abs(q-q[0:1])>=cell[None,None,:]/2-1e-9): raise ValueError('ambiguous first-bead lift branch')
    T=normal_matrix(P); Q,V,omega,qhat,phat=reconstruct(q0,p0,F0,F1,mass,dt,T,temperature)
    qerr,perr=validate_endpoint(qhat,phat,q1,p1)
    d='xyz'.index(a.axis); info=locate_one(Q,V,omega,T,a.bead,a.atom,d,dt,cell[d],mesh[d],int(img0[a.bead,a.atom,d]),int(img1[a.bead,a.atom,d]))
    info.update(schema='r164_native_assignment_crossing_v1',scope='recorded trajectory numerical assignment-center crossing; no P11 physical event, impulse, energy transfer, heat current or winding',
        stdout_binding='same run directory and matching printed facts; stdout not cryptographically bound to capture',step=a.step,bead=a.bead,atom=a.atom,axis=a.axis,mesh=mesh,dt_fs=dt,
        full_q_endpoint_max_A=qerr,full_p_endpoint_max_eV_fs_per_A=perr,endpoint_tolerance=1e-8,
        absolute_root_time_fs=a.step*dt+info['root_from_step_fs'],
        monotonicity_scope='selected coordinate only; other coordinates not certified',
        momentum_scope='post-first-kick free-ring substep momentum; no impulse inserted',
        input_sha256={str(p):sha(p) for p in [runin,stdout,directory/'metadata.txt',directory/'source_identity.txt',directory/'COMPLETE',directory/'scalar.txt',*files,*images,runin.parent/by_filename(lines,'potential'),runin.parent/'model.xyz',runin.parent/by_filename(lines,'read_pimd_restart')]})
    arrays={}
    for tag,t in [('left',info['samples'][0]['time_from_step_fs']),('root',info['root_from_step_fs']),('right',info['samples'][2]['time_from_step_fs'])]:
        qt,vt=mode_state(Q,V,omega,t)
        arrays['q_'+tag+'_A']=np.einsum('k,kia->ia',T[a.bead],qt)
        arrays['p_'+tag+'_eV_fs_per_A']=np.einsum('k,kia->ia',T[a.bead],vt)*mass[:,None]
    with a.output.open('xb') as output: np.savez_compressed(output,metadata_json=json.dumps(info,allow_nan=False),**arrays)

if __name__=='__main__': main()
