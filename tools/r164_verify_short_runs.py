#!/usr/bin/env python3
"""Inspect completed R164 short runs; report differences without acceptance thresholds."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from r164_frame_to_npz import read_frame


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def keyfile(path):
    data={}
    for line in path.read_text(encoding='utf8').splitlines():
        parts=line.split(maxsplit=1)
        if len(parts)==2 and parts[0]!='atom':
            if parts[0] in data: raise ValueError(f'duplicate {parts[0]}')
            data[parts[0]]=parts[1]
    return data


def card(path,source,dt,steps,observer):
    if sha(path)!=source.get('run_in_sha256'): raise ValueError('run.in SHA mismatch')
    lines=[]
    for line in path.read_text(encoding='utf8').splitlines():
        text=line.split('#',1)[0].strip()
        if text: lines.append(text.split())
    keys=['potential','kspace','time_step','ensemble','pimd_propagator','pimd_fix_com','read_pimd_restart','r164_rpmd','run']
    if [x[0] for x in lines]!=keys: raise ValueError('only frozen nine-command diagnostic cards are supported')
    by={x[0]:x for x in lines}
    if (by['kspace']!=['kspace','pppm','0.5'] or by['ensemble']!=['ensemble','rpmd','32','300'] or
        by['pimd_propagator']!=['pimd_propagator','exact'] or by['pimd_fix_com']!=['pimd_fix_com','off'] or
        len(by['potential'])!=2 or len(by['read_pimd_restart'])!=2 or len(by['time_step'])!=2 or
        float(by['time_step'][1])!=dt or by['run']!=['run',str(steps)]):
        raise ValueError('card differs from required thermostat-free short-run contract')
    r=by['r164_rpmd']
    if len(r) not in (3,4) or r[2]!='1' or (len(r)==4)!=(observer) or (observer and r[3]!='observer'):
        raise ValueError('requires stride 1 and requested observer mode')
    return [tuple(x) for x in lines if x[0] not in ('r164_rpmd','time_step','run')]


def labelled(tokens,label,count):
    if label not in tokens: return None
    if tokens.count(label)!=1: raise ValueError('duplicate scalar label')
    pos=tokens.index(label)+1
    try: values=np.array(tokens[pos:pos+count],dtype=float)
    except ValueError as e: raise ValueError('invalid scalar label values') from e
    if values.shape!=(count,) or not np.isfinite(values).all(): raise ValueError('invalid scalar values')
    return values


def read_scalars(path,dt,steps):
    records=[]
    with Path(path).open(encoding='utf8',newline='') as f:
        for line in f:
            if not line.endswith('\n'): raise ValueError('partial scalar line')
            if line.startswith('#'): continue
            t=line.split(); i=len(records)
            if len(t)<21 or int(t[0])!=i or not np.isclose(float(t[1]),i*dt,rtol=1e-12,atol=1e-10): raise ValueError('missing, duplicate or mistimed scalar')
            base=np.array(t[2:21],float)
            if not np.isfinite(base).all(): raise ValueError('nonfinite scalar')
            b=labelled(t,'B_UNVERIFIED',3); j=labelled(t,'J2_UNVERIFIED',3); r=labelled(t,'R27_UNVERIFIED',7)
            if j is not None and (b is None or r is None or not np.allclose(j,b+r[4:],atol=2e-14,rtol=2e-14)): raise ValueError('J2 identity mismatch')
            components={name:labelled(t,label,3) for name,label in [('conv','OBSERVER_CONV_UNVERIFIED'),('local','LOCAL_UNVERIFIED'),('mechanical','MECHANICAL_UNVERIFIED'),('relay','RELAY_UNVERIFIED'),('projection','PROJECTION_UNVERIFIED')]}
            records.append({'base':base,'b':b,'j2':j,'d2':None if r is None else r[4:],'components':components})
    if len(records)!=steps+1: raise ValueError('scalar length mismatch')
    return records


def load_run(directory,run_in,dt,steps,observer):
    directory=Path(directory); meta=keyfile(directory/'metadata.txt'); src=keyfile(directory/'source_identity.txt')
    marker=(directory/'COMPLETE').read_text(encoding='utf8').split()
    if marker!=['r164_diagnostic_v1','final_step',str(steps)] or meta.get('schema')!='r164_diagnostic_v1': raise ValueError('incomplete run or schema')
    if meta.get('time_convention')!='step_times_fixed_dt_start_zero' or float(meta['dt_fs'])!=dt: raise ValueError('time convention or dt mismatch')
    if meta.get('reference_id')!='R27_fixed_2OH_R157' or meta.get('reference_source_sha256','').lower()!='1d46365935f48e390d08e69e8564a3adc692d439b9472b4250914b22144a2eaa': raise ValueError('R27 reference identity mismatch')
    if int(meta['P'])!=32 or int(meta['N'])!=5896 or float(meta['temperature_K'])!=300: raise ValueError('wrong material dimensions or temperature')
    for k in ('model_sha256','model_xyz_sha256','restart_input_sha256','run_in_sha256'):
        if not re.fullmatch('[0-9a-fA-F]{64}',src.get(k,'')): raise ValueError(f'missing {k}')
    if src.get('bead_force_backend') not in ('native_GPUMD','WPE'): raise ValueError('unknown backend')
    cfg=card(run_in,src,dt,steps,observer)
    if observer:
        obs=keyfile(directory/'observer.txt')
        if obs.get('model_sha256','').lower()!=src['model_sha256'].lower() or float(obs['pppm_spacing'])!=0.5: raise ValueError('observer identity mismatch')
    atoms=[]
    for line in (directory/'metadata.txt').read_text(encoding='utf8').splitlines():
        if line.startswith('atom '): atoms.append(line.split()[1:])
    if len(atoms)!=5896 or [int(a[0]) for a in atoms]!=list(range(5896)): raise ValueError('atom order metadata mismatch')
    species=np.array([a[1] for a in atoms]); mass=np.array([float(a[2])*10.18051**2 for a in atoms]); cell=np.array(meta['cell_angstrom'].split(),float)
    if (cell.shape!=(3,) or not np.allclose(cell,29.3104,atol=1e-8,rtol=0) or not np.isfinite(mass).all() or np.any(mass<=0) or
        set(species)!={'O','H','Na','Cl'} or np.count_nonzero(species=='Na')!=26 or np.count_nonzero(species=='Cl')!=26): raise ValueError('invalid cell/mass')
    records=read_scalars(directory/'scalar.txt',dt,steps)
    frames=[directory/f'frame_{i}.txt' for i in range(steps+1)]
    images=[directory/f'force_images_{i}.txt' for i in range(steps+1)]
    for frame_path,image_path in zip(frames,images):
        q,_,_,_=read_frame(frame_path,32,5896)
        read_images(image_path,q,cell)
    return {'directory':str(directory),'source':src,'config':cfg,'meta':meta,'species':species,'mass':mass,'cell':cell,'records':records,'frames':frames,'images':images,'dt':dt,'observer_requested':observer,
            'inputs':{p.name:sha(p) for p in (directory/'metadata.txt',directory/'source_identity.txt',directory/'scalar.txt',directory/'COMPLETE',Path(run_in))},
            'frame_sha256':[sha(p) for p in frames], 'image_sha256':[sha(p) for p in images],
            'observer_sha256':sha(directory/'observer.txt') if observer else None}


def read_images(path,q,cell):
    images=np.empty(q.shape,dtype=np.int64)
    with Path(path).open(encoding='utf8') as f:
        for bead in range(q.shape[0]):
            for atom in range(q.shape[1]):
                line=f.readline()
                if bead==0 and atom==0 and line.startswith('#'): line=f.readline()
                row=line.split()
                if len(row)!=5 or int(row[0])!=bead or int(row[1])!=atom: raise ValueError('invalid force-image ordering')
                try: images[bead,atom]=[int(x) for x in row[2:]]
                except (ValueError,OverflowError) as e: raise ValueError('invalid force image') from e
        if f.read().strip(): raise ValueError('trailing force images')
    canonical=q-images*cell
    if not np.isfinite(canonical).all() or np.any(canonical < -1e-8) or np.any(canonical > cell+1e-8):
        raise ValueError('force image is not canonical for frame')
    return images


def frame(run,i):
    source=run['frames'][i]
    return read_frame(source,32,5896) if isinstance(source,Path) else source

def maximum(a): return float(np.max(np.abs(a)))
def rms(a): return float(np.sqrt(np.mean(np.square(a))))
def compare(a,b,pairs):
    out=[]
    for i,j in pairs:
        qa,pa,fa,ua=frame(a,i); qb,pb,fb,ub=frame(b,j)
        raw=qa-qb; mic=raw-a['cell']*np.rint(raw/a['cell'])
        entry={'step_a':i,'step_b':j,'time_fs':i*a['dt'],'q_lift_max_A':maximum(raw),'q_MIC_max_A':maximum(mic),
               'p_max_eV_fs_per_A':maximum(pa-pb),'F_max_eV_per_A':maximum(fa-fb),'F_rms_eV_per_A':rms(fa-fb),'U_bead_max_eV':maximum(ua-ub),
               'N_max_A_per_fs':maximum(a['records'][i]['base'][7:19]-b['records'][j]['base'][7:19]),
               'energy_momentum_scalar_difference':(a['records'][i]['base'][:7]-b['records'][j]['base'][:7]).tolist()}
        if 'images' in a and 'images' in b:
            ia=read_images(a['images'][i],qa,a['cell']); ib=read_images(b['images'][j],qb,b['cell'])
            entry['force_image_changed_components']=int(np.count_nonzero(ia!=ib))
        entry['observer_components_max_abs']={}
        for name in ('conv','local','mechanical','relay','projection'):
            x=a['records'][i]['components'][name]; y=b['records'][j]['components'][name]
            entry['observer_components_max_abs'][name]=None if x is None or y is None else maximum(x-y)
        for key in ('b','d2','j2'):
            x=a['records'][i][key]; y=b['records'][j][key]
            entry[key+'_max_abs']=None if x is None or y is None else maximum(x-y)
        out.append(entry)
    return out


def energy(run):
    h=np.array([x['base'][3] for x in run['records']]); delta=h-h[0]; index=int(np.argmax(np.abs(delta)))
    return {'H_per_PN_initial_eV':float(h[0]),'H_per_PN_by_step_eV':h.tolist(),'delta_H_per_PN_by_step_eV':delta.tolist(),'max_abs_drift_eV':float(abs(delta[index])),'max_drift_step':index,'max_drift_time_fs':index*run['dt']}


def cpu_reference(run,phase_path,force_path):
    with np.load(phase_path,allow_pickle=False) as ref, np.load(force_path,allow_pickle=False) as f:
        q,p,F,U=frame(run,0); rq=ref['q_A']; rp=ref['p_eV_fs_per_A']; rF=f['F_eV_per_A']; rU=f['U_eV']
        if q.shape!=rq.shape or p.shape!=rp.shape or F.shape!=rF.shape or U.shape!=rU.shape or not all(np.isfinite(x).all() for x in (rq,rp,rF,rU)): raise ValueError('CPU reference shape mismatch')
        if not np.array_equal(run['species'],ref['species']) or not np.allclose(run['mass'],ref['mass'],rtol=0,atol=1e-10) or not np.array_equal(ref['atom_row_zero_based'],np.arange(5896)) or not np.array_equal(ref['bead_order_zero_based'],np.arange(32)) or not np.allclose(ref['cell'],np.diag(run['cell']),atol=1e-10,rtol=0): raise ValueError('CPU reference identity mismatch')
        diff=q-rq; qerr=maximum(diff-run['cell']*np.rint(diff/run['cell'])); perr=maximum(p-rp)
        result={'phase_q_MIC_max_A':qerr,'phase_p_max_eV_fs_per_A':perr,'same_phase_for_force_comparison':qerr<=1e-8 and perr<=1e-8,
                'phase_sha256':sha(phase_path),'force_sha256':sha(force_path),'force_energy_comparison':None}
        result['reference_identity']=('FROZEN_HASHES_MATCH' if result['phase_sha256']=='584363648f470f9a53d05fde068cf7ba76dcf4c27e5b549448a5dd7db3159f25' and result['force_sha256']=='e0c4bee3a1d5936b171584012587349fb1173b324fb0e96e4dcf079061d927d4' and run['source']['model_sha256'].lower()=='fb5b6526ac2b244dba04fea86f09d0344e56b404d413503edad167cf2c39430c' else 'UNVERIFIED')
        if result['same_phase_for_force_comparison']:
            du=U-rU; df=F-rF; species={}
            for symbol in ('O','H','Na','Cl'):
                x=df[:,run['species']==symbol,:]; species[symbol]={'max_abs':maximum(x),'rms':rms(x)}
            result['force_energy_comparison']={'U_mean_offset_eV':float(du.mean()),'U_delta_by_bead_eV':du.tolist(),
                'U_centered_max_eV':maximum(du-du.mean()),'F_max_eV_per_A':maximum(df),'F_rms_eV_per_A':rms(df),'F_by_species_eV_per_A':species}
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('coarse','fine','observer-off'):
        ap.add_argument('--'+name,required=True,type=Path); ap.add_argument('--'+name+'-run-in',required=True,type=Path)
    ap.add_argument('--phase',required=True,type=Path); ap.add_argument('--cpu-forces',required=True,type=Path); ap.add_argument('--output',required=True,type=Path)
    x=ap.parse_args()
    runs=[load_run(x.coarse,x.coarse_run_in,0.2,5,True),load_run(x.fine,x.fine_run_in,0.1,10,True),load_run(x.observer_off,x.observer_off_run_in,0.1,10,False)]
    signature=lambda r:(r['source']['bead_force_backend'],r['source']['model_sha256'].lower(),r['source']['model_xyz_sha256'].lower(),r['source']['restart_input_sha256'].lower(),r['config'],r['meta']['reference_id'],r['meta']['reference_source_sha256'])
    if any(signature(r)!=signature(runs[0]) or not np.array_equal(r['species'],runs[0]['species']) or not np.array_equal(r['mass'],runs[0]['mass']) or not np.array_equal(r['cell'],runs[0]['cell']) for r in runs[1:]): raise ValueError('source/configuration/atom/cell identities differ')
    builds=[(r['source'].get('build_git_commit','UNAVAILABLE'),r['source'].get('build_git_dirty','UNAVAILABLE')) for r in runs]
    known={x for x in builds if x[0] not in ('UNAVAILABLE','unknown','') and x[1] not in ('UNAVAILABLE','unknown','')}
    if len(known)>1: raise ValueError('known build identities differ')
    result={'normalized_propagation_config':[list(x) for x in runs[0]['config']], 'build_identity':'PARTIAL_OR_UNAVAILABLE' if len(known)==0 or any(x[0] in ('UNAVAILABLE','unknown','') or x[1] in ('UNAVAILABLE','unknown','') for x in builds) else 'METADATA_MATCH_UNVERIFIED_BINARY', 'schema':'r164_short_run_verification_v1','status':'STRUCTURE_CHECKED_NO_PHYSICAL_GATE_PASS','thresholds':'NOT_FROZEN',
            'scope':'finite short-run comparisons; no event, production or conductivity acceptance',
            'runs':{name:{'directory':r['directory'],'source':r['source'],'files':r['inputs'],'frame_sha256':r['frame_sha256'],'image_sha256':r['image_sha256'],'observer_sha256':r['observer_sha256'],
                    'observer_component_available_steps':{k:sum(x['components'][k] is not None for x in r['records']) for k in ('conv','local','mechanical','relay','projection')}}
                    for name,r in zip(('coarse_on','fine_on','fine_off'),runs)},
            'units':{'q':'angstrom','p':'eV fs/angstrom','F':'eV/angstrom','U':'eV/bead','H_per_PN':'eV','b_d2_J2':'eV angstrom/fs','N':'angstrom/fs'},
            'gpu_and_runtime':'UNVERIFIED_FROM_CAPTURE',
            'observer_candidate_coverage':('COMPLETE_UNVERIFIED' if all(all(c is not None for c in row['components'].values()) for run in runs[:2] for row in run['records']) else 'INCOMPLETE_UNAVAILABLE'), 'energy_full_window':[energy(r) for r in runs],
            'coarse_fine_common_times':compare(runs[0],runs[1],[(i,2*i) for i in range(6)]),
            'observer_on_off_same_step':compare(runs[1],runs[2],[(i,i) for i in range(11)]),
            'image_scope':'geometric native force-frame endpoints only; no physical event inference',
            'cpu_initial':[cpu_reference(r,x.phase,x.cpu_forces) for r in runs]}
    serialized=json.dumps(result,indent=2,allow_nan=False)+'\n'
    with x.output.open('x',encoding='utf8') as f: f.write(serialized)
if __name__=='__main__': main()
