#!/usr/bin/env python3
"""Provisional R27 D/E replay from contiguous full-bead R164 frame NPZ files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED='1d46365935f48e390d08e69e8564a3adc692d439b9472b4250914b22144a2eaa'

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024),b''): value.update(block)
    return value.hexdigest()

def evaluate(c,y,p):
    c0,lin,sq,sp,t,v=(c[k] for k in ('c','gamma','S_Q','S_P','T','V'))
    val=c0+lin@y+0.5*np.einsum('aij,i,j->a',sq,y,y)+np.einsum('aijk,i,j,k->a',t,y,y,y)/6
    w=sp+np.einsum('aijk,i->ajk',v,y)
    val+=0.5*np.einsum('aij,i,j->a',w,p,p)
    dy=lin+sq@y+0.5*np.einsum('aijk,j,k->ai',t,y,y)+0.5*np.einsum('aijk,j,k->ai',v,p,p)
    return val,dy,w@p

def replay(Q,Pi,a,time,omega,q0,c):
    y=Q-q0; affine=-y*omega**2; delta=a-affine
    pol,dy,dp=[np.stack(z) for z in zip(*(evaluate(c,r,p) for r,p in zip(y,Pi)))]
    jg=np.einsum('tai,ti->ta',dy,Pi)+np.einsum('tai,ti->ta',dp,a)
    jh=np.einsum('tai,ti->ta',dy,Pi)+np.einsum('tai,ti->ta',dp,affine)
    feedback=np.einsum('tai,ti->ta',dp,delta)
    before=[]; drift=[]; after=[]; qerr=[]; perr=[]
    for i,h in enumerate(np.diff(time)):
        mid=Pi[i]+h*a[i]/2; dq=Q[i+1]-Q[i]; first=mid-Pi[i]; last=Pi[i+1]-mid
        qerr.append(dq-h*mid); perr.append(last-h*a[i+1]/2)
        before.append(evaluate(c,y[i],(Pi[i]+mid)/2)[2]@first)
        after.append(evaluate(c,y[i+1],(Pi[i+1]+mid)/2)[2]@last)
        g=[evaluate(c,z,mid)[1] for z in (y[i],(y[i]+y[i+1])/2,y[i+1])]
        drift.append((g[0]+4*g[1]+g[2])@dq/6)
    split=np.array(before)+np.array(drift)+np.array(after)
    return {'Q':Q.tolist(),'Pi':Pi.tolist(),'bead_mean_projected_force':a.tolist(),
            'affine_force':affine.tolist(),'force_residual':delta.tolist(),
            'JG':jg.tolist(),'JH':jh.tolist(),'feedback':feedback.tolist(),
            'D_identity_residual':(jg-jh-feedback).tolist(),
            'Q_drift_residual':np.asarray(qerr).tolist(),'last_kick_residual':np.asarray(perr).tolist(),
            'first_kick_integral':np.asarray(before).tolist(),'drift_integral':np.asarray(drift).tolist(),
            'last_kick_integral':np.asarray(after).tolist(),
            'split_minus_endpoint':(split-np.diff(pol,axis=0)).tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parameters',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('frames',nargs='+',type=Path)
    args=ap.parse_args()
    payload=args.parameters.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=EXPECTED:
        raise ValueError('R27 parameter SHA mismatch')
    par=json.loads(payload)
    if par['id']!='R27_fixed_2OH_R157' or len(args.frames)<2:
        raise ValueError('requires fixed R27 reference and at least two contiguous frames')
    support=par['support']; indices=[r['atom_id'] for r in support]
    mass=np.array([r['mass_eV_fs2_per_A2'] for r in support]); basis=np.array([r['B_cartesian_by_mode'] for r in support]); ref=np.array([r['reference_lift_A'] for r in support])
    qlist=[]; plist=[]; flist=[]; times=[]; steps=[]; dt0=None; inputs=[]; first_shape=None; first_species=None; first_mass=None
    for path in args.frames:
        with np.load(path,allow_pickle=False) as z:
            q=z['q_A']; p=z['p_eV_fs_per_A']; f=z['F_eV_per_A']; t=float(z['time_fs']); dt=float(z['dt_fs']); step=int(z['step'])
            if q.ndim!=3 or q.shape[0]<1 or q.shape[1:]!=(5896,3) or q.shape!=p.shape or q.shape!=f.shape or not all(np.isfinite(v).all() for v in (q,p,f)) or not np.isfinite(t) or not np.isfinite(dt) or dt<=0:
                raise ValueError('invalid full-bead frame')
            species=z['species']; masses=z['mass']; cell=z['cell']
            if species.shape!=(5896,) or masses.shape!=(5896,) or cell.shape!=(3,3) or not np.allclose(cell,np.eye(3)*29.3104,rtol=0,atol=1e-8):
                raise ValueError('invalid atom metadata or cell')
            if first_shape is not None and (q.shape!=first_shape or not np.array_equal(species,first_species) or not np.array_equal(masses,first_mass)):
                raise ValueError('frame bead count or atom metadata differs')
            if first_shape is None: first_shape=q.shape; first_species=species.copy(); first_mass=masses.copy()
            if not np.array_equal(species[indices],[r['species'] for r in support]) or not np.allclose(masses[indices],mass,rtol=1e-10,atol=0):
                raise ValueError('R27 support species or mass mismatch')
            if not np.isclose(t,step*dt,rtol=1e-12,atol=1e-10) or (steps and (step!=steps[-1]+1 or dt!=dt0)):
                raise ValueError('frames must be contiguous with matching dt and time')
            dt0=dt; times.append(t); steps.append(step); qlist.append(q[:,indices,:].mean(axis=0)); plist.append(p[:,indices,:].mean(axis=0)); flist.append(f[:,indices,:].mean(axis=0))
        inputs.append({'path':str(path),'sha256':digest(path)})
    xc=np.array(qlist); pc=np.array(plist); fc=np.array(flist)
    Q=np.einsum('i,iaj,tia->tj',mass,basis,xc-ref); Pi=np.einsum('iaj,tia->tj',basis,pc); a=np.einsum('iaj,tia->tj',basis,fc)
    groups=par['diagnostic_delta_G_affine_centered_by_Cartesian_component']; c={k:np.array([r[k] for r in groups]) for k in ('c','gamma','S_Q','S_P','T','V')}
    result={'schema':'r164_provisional_de_v1','scope':'diagnostic arithmetic on supplied whole-step frames; no event or physical release',
            'static_C':'inherited reference coefficients only; no independent C revalidation',
            'full_hessian_and_same_source_backend_residual':'UNAVAILABLE',
            'reference_sha256':EXPECTED,'frame_inputs':inputs,'time_fs':times,
            'units':{'Q':'sqrt(eV) fs','Pi':'sqrt(eV)','force':'sqrt(eV)/fs','polarization':'eV angstrom'},
            'replay':replay(Q,Pi,a,np.array(times),np.array(par['mode_omega_per_fs']),np.array(par['diagnostic_q0_sqrt_eV_fs']),c)}
    with args.output.open('x',encoding='utf8') as out:
        json.dump(result,out,indent=2,allow_nan=False); out.write('\n')

if __name__=='__main__': main()
