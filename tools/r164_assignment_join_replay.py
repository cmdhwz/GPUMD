#!/usr/bin/env python3
"""Replay two assignment branches at one captured native force frame."""
import argparse
import json
from pathlib import Path
import numpy as np
from r164_audit_force_split import audit, COEFF, metrics
from r164_verify_short_runs import sha


def fused_weights(delta):
    x=np.float32(delta)
    out=np.empty(5,dtype=np.float32)
    for i in range(5):
        value=np.float32(COEFF[i,4])
        for j in (3,2,1,0):
            value=np.float32(float(value)*float(x)+COEFF[i,j])
        out[i]=value
    return out


def adjacent_coordinates(lower_center):
    upper=np.float32(lower_center+0.5)
    lower=np.nextafter(upper,np.float32(-np.inf))
    while int(np.float32(lower+np.float32(.5)))!=lower_center:
        lower=np.nextafter(lower,np.float32(-np.inf))
    upper=np.nextafter(lower,np.float32(np.inf))
    if int(np.float32(upper+np.float32(.5)))!=lower_center+1:
        raise ValueError('failed to locate adjacent native center transition')
    return lower,upper


def branch_state(A, centers, weights, K, G, wavevectors, kc):
    n=len(A); shape=tuple(K[::-1]); size=int(np.prod(K))
    offsets=np.indices((5,5,5)).reshape(3,-1).T-2
    linear=((centers[:,2,None]+offsets[None,:,2])%K[2]*K[1]+(centers[:,1,None]+offsets[None,:,1])%K[1])*K[0]+(centers[:,0,None]+offsets[None,:,0])%K[0]
    w=weights.astype(np.float32)
    w0=w[:,0,offsets[:,0]+2]; w1=w[:,1,offsets[:,1]+2]; w2=w[:,2,offsets[:,2]+2]
    product=((w0*w1).astype(np.float32)*w2).astype(np.float32)
    deposited=(A[:,1,None].astype(np.float32)*product).astype(np.float32)
    rho=np.zeros(size)
    np.add.at(rho,linear.ravel(),deposited.ravel())
    rho=rho.reshape(shape)
    spectrum=np.fft.fftn(rho)
    phi=np.fft.ifftn(spectrum*G).real*size
    D=2*kc*np.sum(product*phi.ravel()[linear],axis=1)
    force=np.empty((n,3))
    for axis in range(3):
        field=np.fft.ifftn(-1j*wavevectors[:,axis].reshape(shape)*G*spectrum).real*size
        force[:,axis]=2*kc*A[:,1]*np.sum(product*field.ravel()[linear],axis=1)
    return rho,phi,D,force,linear,deposited


def compare_states(lo,hi,G,kc,drho=None):
    rho0,phi0=lo[:2]; rho1,phi1=hi[:2]
    if drho is None: drho=rho1-rho0
    dphi=np.fft.ifftn(np.fft.fftn(drho)*G).real*drho.size
    paired=kc*np.sum(drho*(2*phi0+dphi))
    node=kc*(drho*phi0+(rho0+drho)*dphi)
    direct=kc*(rho1*phi1-rho0*phi0)
    return {'paired_delta_U_eV':float(paired),'node_delta_U_eV':float(np.sum(node)),
            'energy_identity_residual_eV':float(np.sum(node)-paired),
            'direct_endpoint_subtraction_residual_eV':float(np.sum(direct)-paired),
            'delta_rho_L1_e':float(np.abs(drho).sum()),'changed_nodes':int(np.count_nonzero(drho)),
            'node_delta_energy_max_abs_eV':float(np.max(np.abs(node))),
            'node_delta_energy_L1_eV':float(np.sum(np.abs(node))),
            'delta_phi_reconstruction_max':float(np.max(np.abs(phi1-phi0-dphi)))}


def charge_feedback(edge, dg, n):
    force=np.zeros((n,3))
    value=dg[edge['center'],None]*edge['g']
    np.add.at(force,edge['center'],value)
    np.add.at(force,edge['atom'],-value)
    return force


def replay(capture,step,atom,axis,lower_center,native_adjacent=False):
    report,data=audit(capture,step,_return_data=True)
    directory=Path(capture).resolve()
    lines=[line.split('#',1)[0].split() for line in (directory.parent/'run.in').read_text(encoding='utf8').splitlines()]
    restart=next(line[1] for line in lines if line and line[0]=='read_pimd_restart')
    restart_sha=sha(directory.parent/restart)
    if restart_sha!=report['source_identity'].get('restart_input_sha256'):
        raise ValueError('restart input SHA mismatch')
    A=data['A']; n=len(A); K=data['K']; cell=data['cell']; ax='xyz'.index(axis)
    if not 0<=atom<n or not 0<=lower_center<int(K[ax]):
        raise ValueError('target atom or adjacent center pair out of range')
    centers=A[:,8:11].astype(int); actual=int(centers[atom,ax]); delta=A[atom,11+ax]
    if actual not in (lower_center,lower_center+1): raise ValueError('captured center is not in requested pair')
    captured_s=np.float32(A[atom,5+ax])
    if (float(captured_s)!=A[atom,5+ax] or int(np.float32(captured_s+np.float32(.5)))!=actual
            or float(np.float32(captured_s-np.float32(actual)))!=delta):
        raise ValueError('captured FP32 grid coordinate, center and delta disagree')
    lower_s,upper_s=adjacent_coordinates(lower_center)
    face=(lower_center+0.5)*cell[ax]/K[ax]
    distance=float(A[atom,2+ax]-face)
    tolerance=max(abs(float(lower_s)-(lower_center+.5)),abs(float(upper_s)-(lower_center+.5)))*cell[ax]/K[ax]+1e-12
    if abs(distance)>tolerance: raise ValueError('captured coordinate is not at the assignment face')
    captured=A[atom,14:].reshape(3,5).astype(np.float32)
    reconstructed=np.stack([fused_weights(A[atom,11+d]) for d in range(3)])
    if not np.array_equal(captured,reconstructed): raise ValueError('captured target weights differ from FP32 FMA replay')
    if native_adjacent:
        offsets=(float(np.float32(lower_s-np.float32(lower_center))),
                 float(np.float32(upper_s-np.float32(lower_center+1))))
        branch_coordinates=[float(lower_s),float(upper_s)]
    else:
        offsets=(0.5,-0.5)
        branch_coordinates=None
    states=[]
    for center,offset in zip((lower_center,lower_center+1),offsets):
        c=centers.copy(); c[atom,ax]=center
        w=A[:,14:].reshape(n,3,5).copy(); w[atom,ax]=fused_weights(offset)
        states.append(branch_state(A,c,w,K,data['G'],data['wavevectors'],data['kc']))
    lo,hi=states
    delta_rho=np.zeros(int(np.prod(K)))
    np.add.at(delta_rho,hi[4][atom],hi[5][atom].astype(float))
    np.add.at(delta_rho,lo[4][atom],-lo[5][atom].astype(float))
    delta_rho=delta_rho.reshape(tuple(K[::-1]))
    mesh=compare_states(lo,hi,data['G'],data['kc'],delta_rho)
    dg=hi[2]-lo[2]; dg-=dg.mean()
    feedback=charge_feedback(data['edge'],dg,n)
    ik=hi[3]-lo[3]
    return {'schema':'r164_assignment_join_replay_v1',
            'scope':'same-frame CPU FP64 FFT, FP32 native-form assignment; fixed short-range ANN graph and charge; no GPU two-sided force, impulse, winding, transport or propagation bound',
            'capture':report['capture'],'step':step,'bead':report['bead'],'force_evaluation_id':report['force_evaluation_id'],
            'input_sha256':{**report['input_sha256'], 'restart':restart_sha},'receipt_sha256':report['receipt_sha256'],
            'target':{'atom':atom,'axis':axis,'lower_center':lower_center,'captured_center':actual,
                      'captured_delta':float(delta),'captured_grid_coordinate':float(captured_s),
                      'face_A':float(face),'distance_A':distance,'face_tolerance_A':float(tolerance),
                      'captured_weights_match_FP32_fused_Horner_all_axes':True,
                      'branch_delta_lower':offsets[0],'branch_delta_upper':offsets[1],
                      'branch_grid_coordinates':branch_coordinates,
                      'branch_mode':'native_adjacent_FP32' if native_adjacent else 'nominal_half_grid'},
            'mesh':mesh,'projected_D_difference_max_eV_per_e':float(np.max(np.abs(dg))),
            'force_difference_eV_per_A':{'reciprocal_ik':metrics(ik,data['velocity']),
                 'ANN_charge_feedback':metrics(feedback,data['velocity']),
                 'total':metrics(ik+feedback,data['velocity'])}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path); ap.add_argument('--step',required=True,type=int)
    ap.add_argument('--atom',required=True,type=int); ap.add_argument('--axis',required=True,choices='xyz')
    ap.add_argument('--lower-center',required=True,type=int); ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--native-adjacent',action='store_true')
    args=ap.parse_args(); result=replay(args.capture,args.step,args.atom,args.axis,args.lower_center,args.native_adjacent)
    with args.output.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')


if __name__=='__main__': main()
