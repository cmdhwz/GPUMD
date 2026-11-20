#!/usr/bin/env python3
"""Audit the fixed-branch native all-center qNEP/PPPM force split."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from r164_frame_to_npz import read_frame
from r164_verify_short_runs import keyfile, read_images, sha

COEFF=np.array([[.0026041667,-.020833333,.0625,-.083333333,.041666667],[.19791667,-.45833333,.25,.16666667,-.16666667],[.59895833,0,-.625,0,.25],[.19791667,.45833333,.25,-.16666667,-.16666667],[.0026041667,.020833333,.0625,.083333333,.041666667]],dtype=np.float32).astype(float)
EDGE=np.dtype([('kind','<i4'),('center','<i4'),('slot','<i4'),('atom','<i4'),('image','<i4',(3,)),('r','<f4',(3,)),('B','<f4',(3,)),('e','<f4',(3,)),('g','<f4',(3,))],align=False)

def headers(path):
    out={}
    with path.open(encoding='utf8') as f:
        for line in f:
            if not line.startswith('#'): break
            parts=line[1:].strip().split(maxsplit=1)
            if len(parts)==2 and parts[0] not in out: out[parts[0]]=parts[1]
    return out

def loadtxt(path,columns):
    x=np.loadtxt(path,comments='#',ndmin=2)
    if x.shape[1]!=columns or not np.isfinite(x).all(): raise ValueError('invalid columns or nonfinite '+str(path))
    return x

def metrics(x,velocity):
    k=np.unravel_index(np.argmax(np.abs(x)),x.shape)
    return {'max_abs':float(abs(x[k])),'atom':int(k[0]),'axis':'xyz'[k[1]],'signed_value':float(x[k]),'rms':float(np.sqrt(np.mean(x*x))),'net_vector':x.sum(axis=0).tolist(),'power_eV_per_fs':float(np.sum(x*velocity))}

def read_edges(path, counts):
    raw=path.read_bytes()
    if len(raw)!=int(counts.sum())*76: raise ValueError('edge length mismatch')
    return np.frombuffer(raw,dtype=EDGE)


def audit(directory,step):
    d=Path(directory).resolve(); rec={}
    for line in (d/f'pppm_trace_step_{step}_receipt.txt').read_text(encoding='utf8').splitlines():
        parts=line.split()
        if len(parts)!=2 or parts[0] in rec: raise ValueError('malformed receipt')
        rec[parts[0]]=parts[1]
    if rec.get('schema')!='r164_pppm_trace_all_v1' or rec.get('atom')!='all' or int(rec['step'])!=step or rec.get('physical_event')!='unavailable' or rec.get('neighbor_history')!='all_centers_selected_bead_endpoint': raise ValueError('not an all-center trace receipt')
    bead=int(rec['bead']); fid=int(rec['force_evaluation_id']); prefix=f'pppm_trace_step_{step}_bead_{bead}'
    files={'metadata':d/'metadata.txt','source_identity':d/'source_identity.txt','frame':d/f'frame_{step}.txt','force_images':d/f'force_images_{step}.txt'}
    for name,suffix in [('mesh','mesh.out'),('kspace','kspace.out'),('atom','atom.out'),('assignment_atom','assignment_atom.out'),('assignment_stencil','assignment_stencil_atom0.out'),('pre_electro','pre_electro.out'),('all_meta','all_meta.out'),('all_edges','all_edges.bin')]: files[name]=d/f'{prefix}_{suffix}'
    hashes={}
    for name,path in files.items():
        hashes[name]=sha(path)
        if hashes[name]!=rec.get(name+'_sha256'): raise ValueError('receipt hash mismatch: '+name)
    meta=keyfile(files['metadata']); src=keyfile(files['source_identity'])
    if src.get('bead_force_backend')!='native_GPUMD' or meta.get('schema')!='r164_diagnostic_v1': raise ValueError('requires native R164 capture')
    run=d.parent/'run.in'; model=d.parent/'model.xyz'
    if sha(run)!=src.get('run_in_sha256') or sha(model)!=src.get('model_xyz_sha256'): raise ValueError('input provenance mismatch')
    lines=[line.split('#',1)[0].split() for line in run.read_text(encoding='utf8').splitlines()]; lines=[line for line in lines if line]
    if [x[0] for x in lines]!=['potential','kspace','time_step','ensemble','pimd_propagator','pimd_fix_com','read_pimd_restart','r164_rpmd','run']: raise ValueError('requires fixed nine-command native card')
    by={x[0]:x for x in lines}
    if len(by['potential'])!=2 or len(by['ensemble'])!=4: raise ValueError('invalid force card')
    potential=(d.parent/by['potential'][1]).resolve()
    if len(by['potential'])!=2 or sha(potential)!=src.get('model_sha256') or len(by['kspace'])!=3 or by['kspace'][1]!='pppm' or by['pimd_propagator']!=['pimd_propagator','exact'] or by['pimd_fix_com']!=['pimd_fix_com','off'] or by['ensemble'][1]!='rpmd': raise ValueError('unsupported force configuration')
    first=potential.read_text(encoding='utf8').splitlines()[0].split()[0]
    if 'charge2' not in first or 'zbl' in first.lower(): raise ValueError('requires ANN-only charge2 potential')
    if not re.search(r'pbc\s*=\s*"T\s+T\s+T"',model.read_text(encoding='utf8').splitlines()[1],re.I): raise ValueError('requires full periodicity')
    rows=[]
    with (d/'scalar.txt').open('rb') as f:
        for line in f:
            if line.startswith(b'#'): continue
            if not line.endswith(b'\n'): raise ValueError('partial scalar row')
            if int(line.split()[0])==step: rows.append(line.rstrip(b'\r\n')+b'\n')
    if len(rows)!=1 or hashlib.sha256(rows[0]).hexdigest()!=rec['scalar_line_sha256']: raise ValueError('scalar row mismatch')
    N=int(meta['N']); P=int(meta['P']); cell=np.array(meta['cell_angstrom'].split(),float); T=float(meta['temperature_K'])
    if fid!=step*P+bead+1: raise ValueError('unexpected native force call identity')
    if not 0<=bead<P or N<1 or cell.shape!=(3,) or not np.isfinite(cell).all() or np.any(cell<=0): raise ValueError('invalid capture dimensions')
    atoms=[line.split() for line in files['metadata'].read_text(encoding='utf8').splitlines() if line.startswith('atom ')]
    if len(atoms)!=N or [int(x[1]) for x in atoms]!=list(range(N)): raise ValueError('invalid atom order')
    mass=np.array([float(x[3])*10.18051**2 for x in atoms]);
    if not np.isfinite(mass).all() or np.any(mass<=0): raise ValueError('invalid masses')
    q,p,F,U=read_frame(files['frame'],P,N); images=read_images(files['force_images'],q,cell); velocity=p[bead]/mass[:,None]
    for name in ('mesh','kspace','atom','assignment_atom','assignment_stencil','pre_electro','all_meta'):
        h=headers(files[name])
        if int(h.get('force_evaluation_id','0'))!=fid: raise ValueError('force ID mismatch: '+name)
    ah=headers(files['all_meta'])
    if ah.get('schema')!='r164_all_edges_v1' or int(ah.get('step','-1'))!=step or int(ah.get('bead','-1'))!=bead or ah.get('binary')!='little_endian record_bytes 76 order kind_slot_center' or 'ANN_only_ZBL_excluded' not in ah.get('units',''): raise ValueError('all-edge schema mismatch')
    ph=headers(files['pre_electro'])
    if int(ph.get('step','-1'))!=step or ph.get('atom')!='all': raise ValueError('pre-electro source mismatch')
    h=headers(files['assignment_atom']); branch=h.get('reciprocal_energy_branch')
    if branch not in ('global_uniform','per_atom') or any(headers(files[name]).get('reciprocal_energy_branch')!=branch for name in ('mesh','kspace','atom','assignment_stencil')): raise ValueError('inconsistent reciprocal energy branch')
    H=np.array(h['box_h'].split(),float).reshape(3,3)
    if not np.allclose(H,np.diag(cell),atol=1e-10,rtol=0): raise ValueError('requires orthogonal fixed cell')
    K=np.array(h['pppm_mesh_size'].split(),int); M=int(np.prod(K)); kc=float(h['K_C_SP'].split()[0]); shape=tuple(K[::-1]);
    if K.shape!=(3,) or np.any(K<5) or not np.isfinite(kc) or int(h['pppm_N'])!=N or int(h['pppm_N1'])!=0 or int(h['pppm_N2'])!=N: raise ValueError('invalid mesh or atom range')
    A=loadtxt(files['assignment_atom'],29); atom=loadtxt(files['atom'],7); m=loadtxt(files['mesh'],7); k=loadtxt(files['kspace'],14); md=loadtxt(files['all_meta'],4)
    if A.shape[0]!=N or atom.shape[0]!=N or md.shape[0]!=N or m.shape[0]!=M or k.shape[0]!=M or not np.array_equal(A[:,0],np.arange(N)) or not np.array_equal(atom[:,0],np.arange(N)) or not np.array_equal(md[:,0],np.arange(N)): raise ValueError('row order/dimensions mismatch')
    grid=np.indices(shape).reshape(3,-1).T[:,::-1]
    if not np.array_equal(m[:,:3],grid) or not np.array_equal(k[:,:3],grid): raise ValueError('mesh ordering mismatch')
    if not np.array_equal(A[:,1],atom[:,4]) or not np.array_equal(A[:,2:5],atom[:,1:4]) or not np.allclose(A[:,2:5],q[bead]-images[bead]*cell,atol=1e-8,rtol=0): raise ValueError('assignment/source mismatch')
    expected_projection=(atom[:,5].astype(np.float32)-np.float32(np.mean(atom[:,5].astype(np.float32),dtype=np.float64))).astype(np.float32)
    if not np.array_equal(expected_projection,md[:,3].astype(np.float32)): raise ValueError('projected D mismatch')
    counts=md[:,1:3].astype(int)
    if np.any(counts<0) or not np.array_equal(counts,md[:,1:3]): raise ValueError('invalid counts')
    edge=read_edges(files['all_edges'],counts); n=len(edge)
    if (np.any(edge['kind']<0) or np.any(edge['kind']>1) or np.any(edge['center']<0) or np.any(edge['center']>=N) or np.any(edge['atom']<0) or np.any(edge['atom']>=N) or not all(np.isfinite(edge[x]).all() for x in ('r','B','e','g'))): raise ValueError('invalid edges')
    if n:
        order=np.lexsort((edge['center'],edge['slot'],edge['kind']))
        if not np.array_equal(order,np.arange(n)): raise ValueError('edge ordering mismatch')
        index=edge['kind']*N+edge['center']; actual=np.bincount(index,minlength=2*N).reshape(2,N).T
        if not np.array_equal(actual,counts): raise ValueError('edge counts mismatch')
        for typ in range(2):
            z=edge[edge['kind']==typ]; idx=np.lexsort((z['slot'],z['center'])); z=z[idx]
            if not np.array_equal(z['slot'],np.arange(len(z))-np.repeat(np.cumsum(np.r_[0,counts[:-1,typ]]),counts[:,typ])): raise ValueError('edge slots mismatch')
    if n:
        geometric=A[edge['atom'],2:5]-A[edge['center'],2:5]+edge['image']*cell
        if np.any(np.abs(geometric-edge['r'])>2e-5*(1+np.abs(geometric))): raise ValueError('edge image/displacement mismatch')
    centers=A[:,8:11].astype(int); delta=A[:,11:14]; weights=A[:,14:].reshape(N,3,5); charge=A[:,1]; gn=md[:,3];
    if not np.array_equal(centers,A[:,8:11]) or np.any(np.abs(delta)>.500001) or not np.isfinite(weights).all(): raise ValueError('invalid assignment')
    stencil=loadtxt(files['assignment_stencil'],9)
    if stencil.shape!=(125,9): raise ValueError('invalid captured stencil')
    pre=loadtxt(files['pre_electro'],3)
    if pre.shape!=(N,3) or not np.array_equal(pre[:,0],np.arange(N)): raise ValueError('invalid raw source rows')
    offsets=np.indices((5,5,5)).reshape(3,-1).T-2
    if not np.array_equal(stencil[:,:3],offsets) or not np.array_equal(stencil[:,3:6],(centers[0]+offsets)%K): raise ValueError('assignment stencil identity mismatch')
    linear=(((centers[:,2,None]+offsets[None,:,2])%K[2])*K[1]+(centers[:,1,None]+offsets[None,:,1])%K[1])*K[0]+(centers[:,0,None]+offsets[None,:,0])%K[0]
    wf=weights.astype(np.float32); w0=wf[:,0,offsets[:,0]+2]; w1=wf[:,1,offsets[:,1]+2]; w2=wf[:,2,offsets[:,2]+2]
    W32=((w0*w1).astype(np.float32)*w2).astype(np.float32)
    if not np.array_equal(stencil[:,6].astype(int),linear[0]) or not np.array_equal(stencil[:,7].astype(np.float32),W32[0]) or not np.array_equal(stencil[:,8].astype(np.float32),(np.float32(charge[0])*W32[0]).astype(np.float32)): raise ValueError('captured stencil weight mismatch')
    derivatives=np.empty((N,3,5))
    for z in range(5): derivatives[:,:,z]=COEFF[z,1]+2*COEFF[z,2]*delta+3*COEFF[z,3]*delta**2+4*COEFF[z,4]*delta**3
    dw=np.empty((N,125,3)); factors=(w0.astype(float),w1.astype(float),w2.astype(float))
    for axis in range(3):
        others=[j for j in range(3) if j!=axis]
        dw[:,:,axis]=derivatives[:,axis,offsets[:,axis]+2]*(K[axis]/cell[axis])*factors[others[0]]*factors[others[1]]
    Q=np.zeros(M); np.add.at(Q,linear.ravel(),(charge[:,None]*W32).ravel()); Q=Q.reshape(shape)
    phi=m[:,5].reshape(shape); S=(k[:,9]+1j*k[:,10]).reshape(shape); G=k[:,11].reshape(shape)
    inverse=np.ix_(*[(-np.arange(n))%n for n in shape])
    if G.flat[0]!=0 or np.any(G<0) or not np.allclose(G,G[inverse],rtol=1e-7,atol=1e-15): raise ValueError('mesh kernel is not even, nonnegative and zero-mode-free')
    field=np.stack([np.fft.ifftn(-1j*k[:,6+a].reshape(shape)*S*G).real*M for a in range(3)])
    native_phi=phi.ravel()[linear]; phi64=np.fft.ifftn(np.fft.fftn(Q)*G).real*M
    ik=2*kc*charge[:,None]*np.stack([(W32*field[a].ravel()[linear]).sum(axis=1) for a in range(3)],axis=1)
    fw=-2*kc*charge[:,None]*np.sum(dw*native_phi[:,:,None],axis=1)
    potential64=phi64.ravel()[linear]
    fw64=-2*kc*charge[:,None]*np.sum(dw*potential64[:,:,None],axis=1)
    D64=2*kc*np.sum(W32*potential64,axis=1); G64=D64-D64.mean()
    local=[]
    for v in (edge['B'],edge['e']+gn[edge['center'],None]*edge['g'],edge['e']+G64[edge['center'],None]*edge['g']):
        f=np.zeros((N,3)); np.add.at(f,edge['center'],v); np.add.at(f,edge['atom'],-v); local.append(f)
    stages=[F[bead],local[0]+ik,local[1]+ik,local[2]+ik,local[2]+fw64]
    differences={f'{a}_minus_{b}':metrics(stages[i]-stages[i+1],velocity) for i,(a,b) in enumerate([('native','local_B_ik'),('local_B_ik','local_e_Gnative_ik'),('local_e_Gnative_ik','local_e_G64_ik'),('local_e_G64_ik','local_e_G64_FW64')])}
    differences['FW_native_phi_minus_ik']=metrics(fw-ik,velocity); differences['FW64_minus_FW_native_phi']=metrics(fw64-fw,velocity)
    force_q=A[:,2:5]
    def shifted_Q(selection,axis,shift):
        indices=np.arange(N) if selection is None else np.array([selection])
        ds=shift*K[axis]/cell[axis]; d0=delta[indices,axis]
        def poly(x):
            return np.stack([sum(COEFF[z,p]*x**p for p in range(5)) for z in range(5)],axis=1)
        replacement=weights[indices,axis,:]+poly(d0+ds)-poly(d0)
        z=offsets[:,axis]+2
        dw_axis=replacement[:,z]-weights[indices,axis,:][:,z]
        others=[j for j in range(3) if j!=axis]
        change=charge[indices,None]*dw_axis*factors[others[0]][indices]*factors[others[1]][indices]
        Qdiff=np.zeros(M); np.add.at(Qdiff,linear[indices].ravel(),change.ravel())
        return Qdiff.reshape(shape)
    def finite_difference(selection,axis):
        ids=np.arange(N) if selection is None else np.array([selection])
        chart=np.minimum(.5-delta[ids,axis],.5+delta[ids,axis])*cell[axis]/K[axis]
        face=np.minimum(force_q[ids,axis],cell[axis]-force_q[ids,axis])
        limit=float(min(np.min(chart),np.min(face)))
        if limit<=1e-12: return {'status':'UNAVAILABLE','reason':'no symmetric fixed-chart interval','margin_A':limit}
        base=min(0.004,limit/4); entries=[]
        for h in (base,base/2,base/4,base/8):
            changes=[]
            for sign in (1,-1):
                dq=shifted_Q(selection,axis,sign*h)
                dphi=np.fft.ifftn(np.fft.fftn(dq)*G).real*M
                changes.append(kc*np.sum(dq*(2*phi64+dphi)))
            fd=-(changes[0]-changes[1])/(2*h)
            fw_ref=float(fw64[:,axis].sum() if selection is None else fw64[selection,axis])
            ik_ref=float(ik[:,axis].sum() if selection is None else ik[selection,axis])
            entries.append({'h_A':h,'negative_energy_derivative_eV_per_A':float(fd),'FW64_eV_per_A':fw_ref,'ik_eV_per_A':ik_ref,'FD_minus_FW64_eV_per_A':float(fd-fw_ref),'FD_minus_ik_eV_per_A':float(fd-ik_ref)})
        return {'status':'COMPUTED_FIXED_BRANCH','margin_A':limit,'values':entries}
    maxcoord=np.unravel_index(np.argmax(np.abs(fw64-ik)),fw64.shape)
    fd_coordinate=finite_difference(int(maxcoord[0]),int(maxcoord[1]))
    fd_translation={axis:finite_difference(None,i) for i,axis in enumerate('xyz')}
    return {'schema':'r164_force_split_audit_v1','scope':'fixed_descriptor_edges_assignment_branch_ANN_only_native_no_event_or_transport_claim','capture':str(d),'receipt_sha256':sha(d/f'pppm_trace_step_{step}_receipt.txt'),'scalar_line_sha256':rec['scalar_line_sha256'],'step':step,'bead':bead,'force_evaluation_id':fid,'N':N,'P':P,'temperature_K':T,'reciprocal_energy_branch':branch,'energy_scope':'global_uniform is total-bead energy; no site-energy allocation claimed','source_identity':src,'input_sha256':{**hashes,'run.in':src['run_in_sha256'],'model.xyz':src['model_xyz_sha256'],'potential':src['model_sha256']},'units':{'force':'eV/angstrom','power':'eV/fs'},'force_stages':{name:metrics(value,velocity) for name,value in zip(('native','local_B_ik','local_e_Gnative_ik','local_e_G64_ik','local_e_G64_FW64'),stages)},'force_differences':differences,'mesh_charge_FP64_minus_native_max':float(np.max(np.abs(Q-m[:,3].reshape(shape)))),'branch_finite_difference':{'atom':int(maxcoord[0]),'axis':'xyz'[maxcoord[1]],'result':fd_coordinate},'translation_finite_difference':fd_translation,'total_F_regular_minus_native':metrics(stages[-1]-stages[0],velocity),'chain_closure':metrics((stages[0]-stages[-1])-sum((stages[i]-stages[i+1] for i in range(4)),np.zeros_like(stages[0])),velocity)}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('capture',type=Path); ap.add_argument('step',type=int); ap.add_argument('--output',type=Path); a=ap.parse_args()
    result=json.dumps(audit(a.capture,a.step),indent=2,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf8',newline='\n') as f: f.write(result)
    else: print(result,end='')
if __name__=='__main__': main()
