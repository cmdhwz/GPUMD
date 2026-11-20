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
    if rec.get('schema')!='r164_pppm_trace_all_v1' or rec.get('atom')!='all' or int(rec['step'])!=step or rec.get('physical_event')!='unavailable': raise ValueError('not an all-center trace receipt')
    bead=int(rec['bead']); fid=int(rec['force_evaluation_id']); prefix=f'pppm_trace_step_{step}_bead_{bead}'
    files={'metadata':d/'metadata.txt','source_identity':d/'source_identity.txt','frame':d/f'frame_{step}.txt','force_images':d/f'force_images_{step}.txt'}
    for name,suffix in [('mesh','mesh.out'),('kspace','kspace.out'),('atom','atom.out'),('assignment_atom','assignment_atom.out'),('assignment_stencil','assignment_stencil_atom0.out'),('pre_electro','pre_electro.out'),('all_meta','all_meta.out'),('all_edges','all_edges.bin')]: files[name]=d/f'{prefix}_{suffix}'
    for name,path in files.items():
        if sha(path)!=rec.get(name+'_sha256'): raise ValueError('receipt hash mismatch: '+name)
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
    if ah.get('schema')!='r164_all_edges_v1' or int(ah.get('step','-1'))!=step or int(ah.get('bead','-1'))!=bead or 'record_bytes 76 order kind_slot_center' not in ah.get('binary','') or 'ANN_only_ZBL_excluded' not in ah.get('units',''): raise ValueError('all-edge schema mismatch')
    ph=headers(files['pre_electro'])
    if int(ph.get('step','-1'))!=step or ph.get('atom')!='all': raise ValueError('pre-electro source mismatch')
    h=headers(files['assignment_atom']); H=np.array(h['box_h'].split(),float).reshape(3,3)
    if not np.allclose(H,np.diag(cell),atol=1e-10,rtol=0): raise ValueError('requires orthogonal fixed cell')
    K=np.array(h['pppm_mesh_size'].split(),int); M=int(np.prod(K)); kc=float(h['K_C_SP'].split()[0]); shape=tuple(K[::-1]);
    if np.any(K<5) or not np.isfinite(kc) or int(h['pppm_N'])!=N or int(h['pppm_N1'])!=0 or int(h['pppm_N2'])!=N: raise ValueError('invalid mesh or atom range')
    A=loadtxt(files['assignment_atom'],29); atom=loadtxt(files['atom'],7); m=loadtxt(files['mesh'],7); k=loadtxt(files['kspace'],14); md=loadtxt(files['all_meta'],4)
    if A.shape[0]!=N or atom.shape[0]!=N or md.shape[0]!=N or m.shape[0]!=M or k.shape[0]!=M or not np.array_equal(A[:,0],np.arange(N)) or not np.array_equal(atom[:,0],np.arange(N)) or not np.array_equal(md[:,0],np.arange(N)): raise ValueError('row order/dimensions mismatch')
    grid=np.indices(shape).reshape(3,-1).T[:,::-1]
    if not np.array_equal(m[:,:3],grid) or not np.array_equal(k[:,:3],grid): raise ValueError('mesh ordering mismatch')
    if not np.array_equal(A[:,1],atom[:,4]) or not np.allclose(A[:,2:5],q[bead]-images[bead]*cell,atol=1e-8,rtol=0): raise ValueError('assignment/source mismatch')
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
    centers=A[:,8:11].astype(int); delta=A[:,11:14]; weights=A[:,14:].reshape(N,3,5); charge=A[:,1]; gn=md[:,3];
    if not np.array_equal(centers,A[:,8:11]) or np.any(np.abs(delta)>.500001) or not np.isfinite(weights).all(): raise ValueError('invalid assignment')
    stencil=loadtxt(files['assignment_stencil'],9)
    if stencil.shape!=(125,9): raise ValueError('invalid captured stencil')
    pre=loadtxt(files['pre_electro'],3)
    if pre.shape!=(N,3) or not np.array_equal(pre[:,0],np.arange(N)): raise ValueError('invalid raw source rows')
    offsets=np.indices((5,5,5)).reshape(3,-1).T-2
    if not np.array_equal(stencil[:,:3],offsets) or not np.array_equal(stencil[:,3:6],(centers[0]+offsets)%K): raise ValueError('assignment stencil identity mismatch')
    Q=np.zeros(shape); phi=m[:,5].reshape(shape); S=(k[:,9]+1j*k[:,10]).reshape(shape); G=k[:,11].reshape(shape)
    if int(ah.get('N','-1'))!=N: raise ValueError('all-edge N mismatch')
    field=[np.fft.ifftn(-1j*k[:,6+a].reshape(shape)*S*G).real*M for a in range(3)]
    derivatives=np.empty((N,3,5));
    for z in range(5): derivatives[:,:,z]=COEFF[z,1]+2*COEFF[z,2]*delta+3*COEFF[z,3]*delta**2+4*COEFF[z,4]*delta**3
    ik=np.zeros((N,3)); fw=np.zeros((N,3)); D=np.zeros(N)
    def traverse(phi_input,scatter=False):
        fw0=np.zeros((N,3)); D0=np.zeros(N)
        for a in range(5):
          ix=(centers[:,0]+a-2)%K[0]
          for b in range(5):
           iy=(centers[:,1]+b-2)%K[1]
           for c in range(5):
            iz=(centers[:,2]+c-2)%K[2]; w=weights[:,0,a]*weights[:,1,b]*weights[:,2,c]; index=(iz,iy,ix)
            if scatter: np.add.at(Q,index,charge*w)
            else:
             v=phi_input[index]; D0+=2*kc*w*v
             for axis in range(3):
              dw=derivatives[:,axis,(a,b,c)[axis]]*K[axis]/cell[axis]
              for j,z in enumerate((a,b,c)):
               if j!=axis: dw*=weights[:,j,z]
              fw0[:,axis]-=2*kc*charge*dw*v
        return fw0,D0
    traverse(phi,True); phi64=np.fft.ifftn(np.fft.fftn(Q)*G).real*M; fw,D=traverse(phi); fw64,D64=traverse(phi64); G64=D64-D64.mean()
    for a in range(5):
      ix=(centers[:,0]+a-2)%K[0]
      for b in range(5):
       iy=(centers[:,1]+b-2)%K[1]
       for c in range(5):
        iz=(centers[:,2]+c-2)%K[2]; w=weights[:,0,a]*weights[:,1,b]*weights[:,2,c]
        for axis in range(3): ik[:,axis]+=2*kc*charge*w*field[axis][iz,iy,ix]
    local=[]
    for v in (edge['B'],edge['e']+gn[edge['center'],None]*edge['g'],edge['e']+G64[edge['center'],None]*edge['g']):
        f=np.zeros((N,3)); np.add.at(f,edge['center'],v); np.add.at(f,edge['atom'],-v); local.append(f)
    stages=[F[bead],local[0]+ik,local[1]+ik,local[2]+ik,local[2]+fw64]
    differences={f'{a}_minus_{b}':metrics(stages[i]-stages[i+1],velocity) for i,(a,b) in enumerate([('native','local_B_ik'),('local_B_ik','local_e_Gnative_ik'),('local_e_Gnative_ik','local_e_G64_ik'),('local_e_G64_ik','local_e_G64_FW64')])}
    differences['FW_native_phi_minus_ik']=metrics(fw-ik,velocity); differences['FW64_minus_FW_native_phi']=metrics(fw64-fw,velocity)
    force_q=A[:,2:5]
    def shifted_Q(selection,axis,shift):
        indices=np.arange(N) if selection is None else np.array([selection])
        Qdiff=np.zeros(shape); ds=shift*K[axis]/cell[axis]
        d0=delta[indices,axis]; w0=weights[indices,axis,:]
        def poly(x):
            return np.stack([sum(COEFF[z,p]*x**p for p in range(5)) for z in range(5)],axis=1)
        replacement=w0+poly(d0+ds)-poly(d0)
        for a in range(5):
          ix=(centers[indices,0]+a-2)%K[0]
          for b in range(5):
           iy=(centers[indices,1]+b-2)%K[1]
           for c in range(5):
            iz=(centers[indices,2]+c-2)%K[2]; z=(a,b,c)
            dw=replacement[:,z[axis]]-w0[:,z[axis]]
            for j in range(3):
             if j!=axis: dw=dw*weights[indices,j,z[j]]
            np.add.at(Qdiff,(iz,iy,ix),charge[indices]*dw)
        return Qdiff
    def finite_difference(selection,axis):
        ids=np.arange(N) if selection is None else np.array([selection])
        chart=np.minimum(.5-delta[ids,axis],.5+delta[ids,axis])*cell[axis]/K[axis]
        face=np.minimum(force_q[ids,axis],cell[axis]-force_q[ids,axis])
        limit=float(min(np.min(chart),np.min(face)))
        if limit<=1e-12: return {'status':'UNAVAILABLE','reason':'no symmetric fixed-chart interval','margin_A':limit}
        base=min(0.004,limit/4); entries=[]
        for h in (base,base/2,base/4):
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
    return {'schema':'r164_force_split_audit_v1','scope':'fixed_descriptor_edges_assignment_branch_ANN_only_native_no_event_or_transport_claim','capture':str(d),'step':step,'bead':bead,'force_evaluation_id':fid,'N':N,'P':P,'temperature_K':T,'source_identity':src,'input_sha256':{**{x:sha(y) for x,y in files.items()},'run.in':sha(run),'model.xyz':sha(model),'potential':sha(potential)},'units':{'force':'eV/angstrom','power':'eV/fs'},'force_stages':{name:metrics(value,velocity) for name,value in zip(('native','local_B_ik','local_e_Gnative_ik','local_e_G64_ik','local_e_G64_FW64'),stages)},'force_differences':differences,'mesh_charge_FP64_minus_native_max':float(np.max(np.abs(Q-m[:,3].reshape(shape)))),'branch_finite_difference':{'atom':int(maxcoord[0]),'axis':'xyz'[maxcoord[1]],'result':fd_coordinate},'translation_finite_difference':fd_translation}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('capture',type=Path); ap.add_argument('step',type=int); ap.add_argument('--output',type=Path); a=ap.parse_args()
    result=json.dumps(audit(a.capture,a.step),indent=2,allow_nan=False)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf8',newline='\n') as f: f.write(result)
    else: print(result,end='')
if __name__=='__main__': main()
