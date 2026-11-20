#!/usr/bin/env python3
"""Algebraic coarse/fine exact-RPMD split ledger for two recorded traces."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from r164_verify_short_runs import load_run, frame, keyfile, sha
from r164_locate_native_crossing import normal_matrix, validate_inputs, stdout_mesh, C, KB, HBAR, PI


def propagate(z, omega, h):
    angle=omega*h
    c=np.cos(angle)[:,None,None]
    s=np.empty_like(omega)
    s[0]=h
    s[1:]=np.sin(angle[1:])/omega[1:]
    s=s[:,None,None]
    w2=omega[:,None,None]**2
    return np.stack((c*z[0]+s*z[1],-w2*s*z[0]+c*z[1]))


def kick(f):
    return np.stack((np.zeros_like(f),f))


def source_terms(f0, fmid, f1, omega, h):
    d=h/2
    average=(f0+f1)/2
    halfdelta=(f1-f0)/2
    angle=omega*d
    cosine_minus_one=-2*np.sin(angle/2)**2
    sinc=np.empty_like(omega)
    sinc[0]=d
    sinc[1:]=np.sin(angle[1:])/omega[1:]
    curvature=(h/2)*propagate(kick(average-fmid),omega,d)
    coupling=(h/2)*propagate(kick(cosine_minus_one[:,None,None]*average),omega,d)
    slope=(h/2)*propagate(np.stack((-sinc[:,None,None]*halfdelta,np.zeros_like(f0))),omega,d)
    direct=(h/4)*(propagate(kick(f0),omega,h)+kick(f1)-2*propagate(kick(fmid),omega,d))
    return curvature,coupling,slope,direct


def summary(z):
    def stats(v):
        return {'max_abs':float(np.max(np.abs(v))), 'rms':float(np.sqrt(np.mean(v*v)))}
    out={}
    for name,part in (('all',z),('centroid',z[:,0:1]),('internal',z[:,1:])):
        out[name]={'Q':stats(part[0]),'P':stats(part[1])}
    return out


def physical_summary(z,T,mass):
    root=np.sqrt(mass)[None,:,None]
    q=np.einsum('sk,kia->sia',T,z[0])/root
    p=np.einsum('sk,kia->sia',T,z[1])*root
    return {'q_max_A':float(np.max(np.abs(q))), 'q_rms_A':float(np.sqrt(np.mean(q*q))),
            'p_max_eV_fs_per_A':float(np.max(np.abs(p))), 'p_rms_eV_fs_per_A':float(np.sqrt(np.mean(p*p))),
            'centroid_q_max_A':float(np.max(np.abs(q.mean(axis=0)))),
            'centroid_p_max_eV_fs_per_A':float(np.max(np.abs(p.mean(axis=0))))}


def component_hash(array):
    return hashlib.sha256(np.asarray(array,dtype='<f8',order='C').tobytes(order='C')).hexdigest()


def load_verified(directory, runin, dt, steps):
    directory=Path(directory).resolve(); runin=Path(runin).resolve()
    lines=[line.split('#',1)[0].split() for line in runin.read_text(encoding='utf8').splitlines()]
    lines=[line for line in lines if line]
    by={line[0]:line for line in lines}
    if runin.parent != directory.parent or (runin.parent/by['r164_rpmd'][1]).resolve()!=directory:
        raise ValueError('capture path does not match run card')
    run=load_run(directory,runin,dt,steps,True)
    validate_inputs(run['source'],runin.parent,lines)
    observer_path=directory/'observer.txt'
    if observer_path.read_text(encoding='utf8').splitlines()[0] != 'native_private_qnep_pppm_observer':
        raise ValueError('observer backend identity mismatch')
    observer=keyfile(observer_path)
    if observer.get('run_in_sha256') != run['source']['run_in_sha256']:
        raise ValueError('observer run identity mismatch')
    potential=runin.parent/by['potential'][1]
    if potential.read_text(encoding='utf8').splitlines()[0].split()[0]!='nep4_charge2':
        raise ValueError('requires native qNEP mode2 potential')
    if run['source']['bead_force_backend']!='native_GPUMD':
        raise ValueError('requires native force route')
    model=(runin.parent/'model.xyz').read_text(encoding='utf8').splitlines()
    if len(model)<2 or not re.search(r'pbc\s*=\s*"T\s+T\s+T"',model[1],re.I):
        raise ValueError('requires three periodic axes')
    stdout=runin.parent/'stdout.log'
    mesh=stdout_mesh(stdout,dt,steps)
    if tuple(mesh)!=(60,60,60): raise ValueError('unexpected PPPM mesh')
    run['stdout_sha256']=sha(stdout)
    return run


def load_frame(run,index):
    path=run['frames'][index]
    if sha(path)!=run['frame_sha256'][index]: raise ValueError('frame changed after validation')
    q,p,F,U=frame(run,index)
    if sha(path)!=run['frame_sha256'][index]: raise ValueError('frame changed during read')
    if sha(run['images'][index])!=run['image_sha256'][index]: raise ValueError('image changed after validation')
    receipts=[]
    for filename in (f'checkpoint_{index}.txt', f'pppm_trace_step_{index}_receipt.txt'):
        receipt_path=Path(run['directory'])/filename
        if not receipt_path.exists(): continue
        data=keyfile(receipt_path)
        allowed=('r164_checkpoint_v1',) if filename.startswith('checkpoint_') else (
            'r164_pppm_trace_v1','r164_pppm_trace_v2','r164_pppm_trace_v3','r164_pppm_trace_all_v1')
        if data.get('schema') not in allowed: raise ValueError('unsupported frame receipt schema')
        if (data.get('step')!=str(index) or data.get('frame_sha256')!=run['frame_sha256'][index]
                or data.get('force_images_sha256')!=run['image_sha256'][index]
                or data.get('metadata_sha256')!=run['inputs']['metadata.txt']
                or data.get('source_identity_sha256')!=run['inputs']['source_identity.txt']):
            raise ValueError('frame receipt identity mismatch')
        rows=[]
        with (Path(run['directory'])/'scalar.txt').open('rb') as source:
            for line in source:
                if line.startswith(b'#'): continue
                if line.split() and int(line.split()[0])==index and line.endswith(b'\n'):
                    rows.append(line.rstrip(b'\r\n')+b'\n')
        if len(rows)!=1 or hashlib.sha256(rows[0]).hexdigest()!=data.get('scalar_line_sha256'):
            raise ValueError('frame receipt scalar mismatch')
        receipts.append({'file':filename,'sha256':sha(receipt_path)})
    record={'step':index,'time_fs':index*run['dt'],'frame_sha256':run['frame_sha256'][index],
            'image_sha256':run['image_sha256'][index],
            'array_sha256_binary64_little_endian':{'q':component_hash(q),'p':component_hash(p),'F':component_hash(F),'U':component_hash(U)},
            'frame_provenance':'frame_image_scalar_line_receipt_bound' if receipts else 'posthoc_hash_only_no_receipt',
            'receipts':receipts}
    return (q,p,F,U),record


def modes(state,T,mass):
    q,p,F,_=state
    root=np.sqrt(mass)[None,:,None]
    return np.stack((np.einsum('sk,sia->kia',T,q*root),np.einsum('sk,sia->kia',T,p/root))),np.einsum('sk,sia->kia',T,F/root)


def execution_evidence(path, coarse, fine, binary_override):
    if path is None:
        if binary_override is not None: raise ValueError('binary verification requires execution sidecar')
        return {'status':'UNAVAILABLE_NOT_BOUND_TO_BINARY'}
    path=Path(path).resolve()
    if coarse.parent.parent!=fine.parent.parent or path.parent!=coarse.parent.parent:
        raise ValueError('execution sidecar and runs must share campaign directory')
    evidence=json.loads(path.read_text(encoding='utf8'))
    names={x.get('case') for x in evidence.get('cases',[]) if x.get('returncode')==0 and x.get('marker_written') is True}
    digest=evidence.get('sha256','')
    if coarse.parent.name not in names or fine.parent.name not in names or not re.fullmatch('[0-9a-fA-F]{64}',digest):
        raise ValueError('execution sidecar does not name both completed runs and a binary SHA')
    declared=Path(evidence['binary'])
    binary=Path(binary_override).resolve() if binary_override is not None else declared
    if binary_override is not None and not binary.is_file(): raise ValueError('binary verification path missing')
    actual=sha(binary) if binary.is_file() else None
    if actual is not None and actual.lower()!=digest.lower(): raise ValueError('execution sidecar binary SHA mismatch')
    return {'status':'SIDECAR_NOT_FRAME_RECEIPT','path':str(path),'sha256':sha(path),
            'binary_path_declared':str(declared),'binary_path_verified':str(binary) if actual is not None else None,
            'binary_sha256_declared':digest.lower(),'binary_sha256_current':actual}


def ledger(coarse, coarse_runin, fine, fine_runin, execution=None, binary=None):
    coarse=Path(coarse).resolve(); fine=Path(fine).resolve()
    c=load_verified(coarse,coarse_runin,0.2,5); f=load_verified(fine,fine_runin,0.1,10)
    for key in ('bead_force_backend','model_sha256','model_xyz_sha256','restart_input_sha256','build_git_commit','build_git_dirty'):
        if c['source'].get(key)!=f['source'].get(key): raise ValueError('source identity differs: '+key)
    if (c['config']!=f['config'] or not np.array_equal(c['species'],f['species'])
            or not np.array_equal(c['mass'],f['mass']) or not np.array_equal(c['cell'],f['cell'])):
        raise ValueError('propagation, atom or cell identity differs')
    h=c['dt']; d=f['dt']
    if h!=2*d: raise ValueError('time step ratio is not two')
    beads=int(c['meta']['P']); T=normal_matrix(beads)
    omega=2*beads*KB*float(c['meta']['temperature_K'])/(HBAR*C)*np.sin(np.arange(beads)*PI/beads); omega[0]=0
    cs,cr=load_frame(c,0); fs,fr=load_frame(f,0)
    initial_q=float(np.max(np.abs(cs[0]-fs[0]))); initial_p=float(np.max(np.abs(cs[1]-fs[1])))
    if initial_q>1e-8 or initial_p>1e-8: raise ValueError('initial lifted phase mismatch')
    cz,cf=modes(cs,T,c['mass']); fz,ff=modes(fs,T,c['mass'])
    initial=cz-fz
    cumulative={name:np.zeros_like(initial) for name in ('endpoint_force_difference','curvature','coupling','slope','closure')}
    propagated_initial=initial.copy(); steps=[]; coarse_records=[cr]; fine_records=[fr]
    for i in range(5):
        cm,cmrec=load_frame(c,i+1); fm,fmrec=load_frame(f,2*i+1); fe,ferec=load_frame(f,2*i+2)
        c1,cf1=modes(cm,T,c['mass']); fmid,ffmid=modes(fm,T,c['mass']); f1,ff1=modes(fe,T,c['mass'])
        actual=c1-f1
        feedback=(h/2)*(propagate(kick(cf-ff),omega,h)+kick(cf1-ff1))
        curvature,coupling,slope,direct=source_terms(ff,ffmid,ff1,omega,h)
        residual=actual-(propagate(cz-fz,omega,h)+feedback+direct)
        coarse_residual=c1-(propagate(cz,omega,h)+(h/2)*(propagate(kick(cf),omega,h)+kick(cf1)))
        fine_first=fmid-(propagate(fz,omega,d)+(d/2)*(propagate(kick(ff),omega,d)+kick(ffmid)))
        fine_second=f1-(propagate(fmid,omega,d)+(d/2)*(propagate(kick(ffmid),omega,d)+kick(ff1)))
        physical_maps={name:physical_summary(value,T,c['mass']) for name,value in
                       (('coarse',coarse_residual),('fine_first',fine_first),('fine_second',fine_second))}
        if any(x['q_max_A']>1e-8 or x['p_max_eV_fs_per_A']>1e-8 for x in physical_maps.values()):
            raise ValueError('single-step reconstructed map exceeds 1e-8 arithmetic tolerance')
        terms={'endpoint_force_difference':feedback,'curvature':curvature,'coupling':coupling,'slope':slope,'closure':residual}
        propagated_initial=propagate(propagated_initial,omega,h)
        for name in cumulative: cumulative[name]=propagate(cumulative[name],omega,h)+terms[name]
        assembled=propagated_initial+sum(cumulative.values(),np.zeros_like(initial))
        steps.append({'coarse_step':i+1,'time_fs':(i+1)*h,'actual_difference':summary(actual),
                      'terms':{name:summary(value) for name,value in terms.items()},
                      'three_part_identity_residual':summary(curvature+coupling+slope-direct),
                      'map_residual_physical':physical_maps,
                      'closure_vs_separate_map_residual':summary(residual-(coarse_residual-propagate(fine_first,omega,d)-fine_second)),
                      'coarse_map_residual':summary(coarse_residual),
                      'fine_first_map_residual':summary(fine_first), 'fine_second_map_residual':summary(fine_second),
                      'cumulative_reconstruction_residual':summary(actual-assembled),
                      'cumulative_terms':{name:summary(value) for name,value in cumulative.items()}})
        cz,cf,fz,ff=c1,cf1,f1,ff1
        coarse_records.append(cmrec); fine_records.extend((fmrec,ferec))
    return {'schema':'r164_split_refinement_ledger_v1',
            'scope':'observed finite paired trajectories only; no exact-solution bound, event attribution or P11 acceptance',
            'endpoint_force_difference_scope':'endpoint force differences include trajectory response and native force variation; not independently separated',
            'units':{'Q':'sqrt(eV) fs','P':'sqrt(eV)','time':'fs','mode_force':'sqrt(eV)/fs'},
            'source_identity':c['source'],'normalized_config':[list(x) for x in c['config']],
            'binary_evidence':execution_evidence(execution,coarse,fine,binary),
            'capture_source':{'coarse':c['inputs'],'fine':f['inputs']},
            'observer_sha256':{'coarse':c['observer_sha256'],'fine':f['observer_sha256']},
            'stdout_sha256':{'coarse':c['stdout_sha256'],'fine':f['stdout_sha256']},
            'pppm_mesh':[60,60,60],
            'frame_identity':{'coarse':coarse_records,'fine':fine_records},
            'initial_q_lift_max_A':initial_q,'initial_p_max_eV_fs_per_A':initial_p,
            'initial_mass_weighted_difference':summary(initial),'steps':steps,
            'final_physical_difference':physical_summary(actual,T,c['mass']),
            'cumulative_physical_terms':{name:physical_summary(value,T,c['mass']) for name,value in cumulative.items()},
            'single_step_arithmetic_tolerance':'1e-8 angstrom and 1e-8 eV fs/angstrom; not a physical threshold',
            'cumulative_initial':summary(propagated_initial),
            'cumulative_terms':{name:summary(value) for name,value in cumulative.items()},
            'observed_trajectory_triangle_bound_rms_only':{key:sum(summary(value)['all'][key]['rms'] for value in [propagated_initial,*cumulative.values(),actual-assembled]) for key in ('Q','P')}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse',required=True,type=Path); p.add_argument('--coarse-run-in',required=True,type=Path)
    p.add_argument('--fine',required=True,type=Path); p.add_argument('--fine-run-in',required=True,type=Path)
    p.add_argument('--execution',type=Path); p.add_argument('--binary',type=Path); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); result=ledger(a.coarse,a.coarse_run_in,a.fine,a.fine_run_in,a.execution,a.binary)
    data=json.dumps(result,indent=2,allow_nan=False)+'\n'
    with a.output.open('x',encoding='utf8',newline='\n') as out: out.write(data)

if __name__=='__main__': main()
