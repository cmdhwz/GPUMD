#!/usr/bin/env python3
"""Prepare and verify fresh-cache links between committed R164 checkpoints."""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from r164_frame_to_npz import read_frame
from r164_verify_short_runs import sha, keyfile, read_images, labelled, validate_frame_scalars, r164_options

KEYS={'schema','step','time_fs','metadata_sha256','source_identity_sha256','frame_sha256','force_images_sha256','scalar_line_sha256','scalar_prefix_sha256','cache_policy','physical_event_history'}
HEX=re.compile(r'^[0-9a-f]{64}$')
C=10.18051

def scalar_prefix(path,step,dt):
    rows=[]
    with path.open('rb') as f:
        for line in f:
            if line.startswith(b'#'): continue
            if len(rows)>step: break
            if not line.endswith(b'\n'): raise ValueError('partial required scalar row')
            raw=line.rstrip(b'\r\n')+b'\n'; t=raw.decode('utf8').split()
            if len(t)<21 or int(t[0])!=len(rows) or not math.isclose(float(t[1]),len(rows)*dt,rel_tol=1e-12,abs_tol=1e-10): raise ValueError('missing or mistimed scalar row')
            base=np.array(t[2:21],float)
            if not np.isfinite(base).all(): raise ValueError('nonfinite scalar')
            b=labelled(t,'B_UNVERIFIED',3); j=labelled(t,'J2_UNVERIFIED',3); r=labelled(t,'R27_UNVERIFIED',7)
            if j is not None and (b is None or r is None or not np.allclose(j,b+r[4:],rtol=2e-14,atol=2e-14)): raise ValueError('J2 identity mismatch')
            rows.append((raw,{'base':base,'b':b,'j2':j,'d2':None if r is None else r[4:]}))
    if len(rows)!=step+1: raise ValueError('missing required scalar row')
    return rows

def simple(name):
    if not name or Path(name).name!=name or name in ('.','..') or '/' in name or '\\' in name or ':' in name: raise ValueError('input must be a simple filename')
    return name

def config(capture,meta,source):
    root=capture.parent; path=root/'run.in'
    if sha(path)!=source.get('run_in_sha256'): raise ValueError('run.in SHA mismatch')
    lines=[x.split('#',1)[0].split() for x in path.read_text(encoding='utf8').splitlines()]; lines=[x for x in lines if x]
    keys=['potential','kspace','time_step','ensemble','pimd_propagator','pimd_fix_com','read_pimd_restart','r164_rpmd','run']
    if [x[0] for x in lines]!=keys: raise ValueError('requires nine-command card')
    b={x[0]:x for x in lines}; r=b['r164_rpmd']
    if (len(b['potential'])!=2 or len(b['read_pimd_restart'])!=2 or len(b['kspace'])!=3 or b['kspace'][1]!='pppm' or float(b['kspace'][2])<=0 or len(b['time_step'])!=2 or not math.isclose(float(b['time_step'][1]),float(meta['dt_fs']),rel_tol=4.5e-16,abs_tol=0.0) or len(b['ensemble'])!=4 or b['ensemble'][1]!='rpmd' or int(b['ensemble'][2])!=int(meta['P']) or float(b['ensemble'][3])!=float(meta['temperature_K']) or b['pimd_propagator']!=['pimd_propagator','exact'] or b['pimd_fix_com']!=['pimd_fix_com','off'] or len(b['run'])!=2 or not 0<int(b['run'][1])<=2147483647): raise ValueError('unsupported run configuration')
    observer,checkpoint=r164_options(r)
    if r[1]!='r164_output' or (root/r[1]).resolve()!=capture.resolve() or checkpoint is None or checkpoint!=int(meta['checkpoint_stride']): raise ValueError('checkpoint card mismatch')
    for name,field in ((simple(b['potential'][1]),'model_sha256'),('model.xyz','model_xyz_sha256'),(simple(b['read_pimd_restart'][1]),'restart_input_sha256')):
        if not HEX.fullmatch(source.get(field,'')) or sha(root/name)!=source[field]: raise ValueError('input identity mismatch: '+field)
    if source.get('bead_force_backend') not in ('native_GPUMD','WPE'): raise ValueError('unknown backend')
    return lines,b

def load(capture,step):
    capture=Path(capture).resolve(); meta=keyfile(capture/'metadata.txt'); source=keyfile(capture/'source_identity.txt')
    if meta.get('schema')!='r164_diagnostic_v1' or meta.get('time_convention')!='step_times_fixed_dt_start_zero' or meta.get('checkpoint')!='FRESH_REBUILD_ONLY': raise ValueError('unsupported metadata')
    if not meta.get('reference_id') or not HEX.fullmatch(meta.get('reference_source_sha256','')): raise ValueError('missing reference identity')
    if not re.fullmatch('[0-9a-fA-F]{40}',source.get('build_git_commit','')) or source.get('build_git_dirty')!='no': raise ValueError('clean Git build identity required')
    lines,b=config(capture,meta,source); dt=float(meta['dt_fs']); P=int(meta['P']); N=int(meta['N']); temp=float(meta['temperature_K']); stride=int(meta['checkpoint_stride']); final=int(b['run'][1])
    if not math.isfinite(dt) or dt<=0 or not math.isfinite(temp) or temp<=0 or P<2 or N<1 or not 0<=step<=final or (step not in (0,final) and step%stride): raise ValueError('invalid checkpoint index or metadata')
    receipt=capture/f'checkpoint_{step}.txt'; data={}
    for line in receipt.read_text(encoding='utf8').splitlines():
        fields=line.split()
        if len(fields)!=2 or fields[0] in data: raise ValueError('malformed checkpoint receipt')
        data[fields[0]]=fields[1]
    if set(data)!=KEYS or data['schema']!='r164_checkpoint_v1' or data['step']!=str(step) or data['cache_policy']!='fresh_rebuild_only' or data['physical_event_history']!='unavailable' or not math.isclose(float(data['time_fs']),step*dt,rel_tol=1e-12,abs_tol=1e-10): raise ValueError('invalid checkpoint receipt')
    for field,path in [('metadata_sha256',capture/'metadata.txt'),('source_identity_sha256',capture/'source_identity.txt'),('frame_sha256',capture/f'frame_{step}.txt'),('force_images_sha256',capture/f'force_images_{step}.txt')]:
        if not HEX.fullmatch(data[field]) or sha(path)!=data[field]: raise ValueError('checkpoint SHA mismatch: '+field)
    rows=scalar_prefix(capture/'scalar.txt',step,dt)
    if not HEX.fullmatch(data['scalar_line_sha256']) or hashlib.sha256(rows[-1][0]).hexdigest()!=data['scalar_line_sha256']: raise ValueError('scalar SHA mismatch')
    prefix=hashlib.sha256()
    for raw,_ in rows: prefix.update(raw)
    if not HEX.fullmatch(data['scalar_prefix_sha256']) or prefix.hexdigest()!=data['scalar_prefix_sha256']: raise ValueError('scalar prefix SHA mismatch')
    observer,_=r164_options(b['r164_rpmd'])
    observer_present=(capture/'observer.txt').exists()
    observer_active='OBSERVER_CONV_UNVERIFIED' in rows[-1][0].decode('utf8').split()
    if not observer and (observer_present or observer_active): raise ValueError('observer-off capture contains observer output')
    if observer_active and not observer_present: raise ValueError('missing observer identity')
    if observer_present:
        obs=keyfile(capture/'observer.txt')
        if obs.get('model_sha256')!=source.get('model_sha256') or obs.get('run_in_sha256')!=source.get('run_in_sha256') or float(obs.get('pppm_spacing','nan'))!=float(b['kspace'][2]): raise ValueError('observer identity mismatch')
    atoms=[x.split()[1:] for x in (capture/'metadata.txt').read_text(encoding='utf8').splitlines() if x.startswith('atom ')]
    if len(atoms)!=N or [int(x[0]) for x in atoms]!=list(range(N)): raise ValueError('atom order mismatch')
    species=np.array([x[1] for x in atoms]); mass=np.array([float(x[2])*C*C for x in atoms]); cell=np.array(meta['cell_angstrom'].split(),float)
    if cell.shape!=(3,) or not np.isfinite(cell).all() or np.any(cell<=0) or not np.isfinite(mass).all() or np.any(mass<=0): raise ValueError('invalid cell or mass')
    q,p,F,U=read_frame(capture/f'frame_{step}.txt',P,N); images=read_images(capture/f'force_images_{step}.txt',q,cell)
    validate_frame_scalars(q,p,U,mass,species,temp,rows[-1][1])
    return {'capture':capture,'step':step,'meta':meta,'source':source,'dt':dt,'P':P,'N':N,'temp':temp,'final':final,'lines':lines,'b':b,'receipt':receipt,'rows':rows,'atoms':atoms,'species':species,'mass':mass,'cell':cell,'q':q,'p':p,'F':F,'U':U,'images':images,'observer':observer,'observer_present':observer_present,'observer_active':observer_active}

def prepare(parent,step,dest,steps):
    if not 0<steps<=2147483647: raise ValueError('steps must be a positive int')
    a=load(parent,step); q=a['q']; cell=a['cell']
    if np.any(q < -cell) or np.any(q > 2*cell): raise ValueError('lift outside one-wrap range')
    potential=a['b']['potential'][1]
    if potential in ('run.in','model.xyz','restart_beads.xyz','parent_receipt.json','phase.npz'): raise ValueError('conflicting potential filename')
    dest=Path(dest).resolve(); dest.mkdir(); phase=dest/'phase.npz'
    np.savez(phase,q_A=q,p_eV_fs_per_A=a['p'],mass=a['mass'],species=a['species'],cell=np.diag(cell),atom_row_zero_based=np.arange(a['N']),bead_order_zero_based=np.arange(a['P']))
    subprocess.run([sys.executable,str(Path(__file__).with_name('r164_phase_to_restart.py')),str(phase),str(dest/'restart_beads.xyz'),'--temperature',str(a['temp']),'--model-output',str(dest/'model.xyz')],check=True); phase.unlink()
    shutil.copyfile(a['capture'].parent/potential,dest/potential)
    lines=[x[:] for x in a['lines']]
    for line in lines:
        if line[0]=='read_pimd_restart': line[1]='restart_beads.xyz'
        if line[0]=='run': line[1]=str(steps)
    (dest/'run.in').write_text(''.join(' '.join(x)+'\n' for x in lines),encoding='utf8',newline='\n')
    receipt={'schema':'r164_checkpoint_parent_v1','parent_capture':str(a['capture']),'parent_step':step,'parent_checkpoint_sha256':sha(a['receipt']),'child_input_sha256':{name:sha(dest/name) for name in ('run.in','model.xyz','restart_beads.xyz',potential)},'scope':'intent_only_fresh_cache_event_history_unavailable'}
    temporary=dest/'parent_receipt.json.tmp'
    temporary.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    temporary.replace(dest/'parent_receipt.json')

def maximum(x): return float(np.max(np.abs(x)))

def verify(captures,output):
    captures=[Path(x).resolve() for x in captures]; links=[]; segments=[]; allrows=[]; absolute=0; previous=None
    if len(set(captures))!=len(captures): raise ValueError('duplicate capture')
    if (captures[0].parent/'parent_receipt.json').exists(): raise ValueError('child capture cannot be used as root')
    for index,capture in enumerate(captures):
        if index+1<len(captures):
            manifest=json.loads((captures[index+1].parent/'parent_receipt.json').read_text(encoding='utf8')); end=int(manifest['parent_step'])
        else:
            marker=(capture/'COMPLETE').read_text(encoding='utf8').split()
            if len(marker)!=3 or marker[:2]!=['r164_diagnostic_v1','final_step']: raise ValueError('final capture incomplete')
            end=int(marker[2])
        run=load(capture,end)
        if index==len(captures)-1:
            if end!=run['final']: raise ValueError('final marker does not match card')
            with (capture/'scalar.txt').open('rb') as f:
                count=0
                for line in f:
                    if line.startswith(b'#'): continue
                    count+=1
                    if count>end+1 or not line.endswith(b'\n'): raise ValueError('final scalar contains extra or partial data')
            if count!=end+1: raise ValueError('final scalar length mismatch')
        if index:
            manifest=json.loads((capture.parent/'parent_receipt.json').read_text(encoding='utf8'))
            if set(manifest)!={'schema','parent_capture','parent_step','parent_checkpoint_sha256','child_input_sha256','scope'} or manifest['schema']!='r164_checkpoint_parent_v1' or manifest['scope']!='intent_only_fresh_cache_event_history_unavailable' or Path(manifest['parent_capture']).resolve()!=previous['capture'] or manifest['parent_step']!=previous['step'] or manifest['parent_checkpoint_sha256']!=sha(previous['receipt']): raise ValueError('parent link mismatch')
            if run['b']['read_pimd_restart'][1]!='restart_beads.xyz': raise ValueError('child restart input differs from receipt')
            expected={name:sha(capture.parent/name) for name in ('run.in','model.xyz','restart_beads.xyz',run['b']['potential'][1])}
            if manifest['child_input_sha256']!=expected: raise ValueError('child input receipt mismatch')
            for key in ('bead_force_backend','model_sha256','build_git_commit','build_git_dirty'):
                if run['source'].get(key)!=previous['source'].get(key): raise ValueError('source identity mismatch: '+key)
            for key in ('P','N','dt_fs','temperature_K','cell_angstrom','reference_id','reference_source_sha256','checkpoint_stride'):
                if run['meta'].get(key)!=previous['meta'].get(key): raise ValueError('metadata mismatch: '+key)
            if (not np.array_equal(run['species'],previous['species']) or not np.allclose(run['mass'],previous['mass'],rtol=2e-15,atol=0) or any(run['b'][key]!=previous['b'][key] for key in ('potential','kspace','time_step','ensemble','pimd_propagator','pimd_fix_com','r164_rpmd'))): raise ValueError('configuration mismatch')
            start=load(capture,0)
            if (start['observer'],start['observer_present'],start['observer_active'])!=(previous['observer'],previous['observer_present'],previous['observer_active']): raise ValueError('observer state mismatch')
            dq=maximum(start['q']-previous['q']); dp=maximum(start['p']-previous['p']); images=int(np.count_nonzero(start['images']!=previous['images']))
            if dq>1e-8 or dp>1e-8 or images: raise ValueError('checkpoint seam mismatch')
            diffs={}
            for name in ('b','d2','j2'):
                x=previous['rows'][-1][1][name]; y=start['rows'][0][1][name]; diffs[name]=None if x is None or y is None else maximum(x-y)
            links.append({'q_lift_max_A':dq,'p_max_eV_fs_per_A':dp,'force_image_changed_components':images,'F_max_eV_per_A':maximum(start['F']-previous['F']),'U_bead_max_eV':maximum(start['U']-previous['U']),'observer_max_abs':diffs})
        segments.append({'capture':str(capture),'end_step':end,'absolute_step_start':absolute,'absolute_step_end':absolute+end,'boundary_truncated':end<run['final'],'checkpoint_sha256':sha(run['receipt']),'metadata_sha256':sha(capture/'metadata.txt'),'source_identity_sha256':sha(capture/'source_identity.txt'),'source_identity':run['source'],'build_binding':'clean_git_label_not_binary_hash','observer_state':{'requested':run['observer'],'identity_present':run['observer_present'],'active_at_endpoint':run['observer_active']}})
        for local,(raw,_) in enumerate(run['rows']):
            if index and local==0: continue
            allrows.append({'absolute_step':absolute+local,'absolute_time_fs':(absolute+local)*run['dt'],'capture':str(capture),'local_step':local,'scalar_line':raw.decode('utf8').rstrip('\n')})
        absolute+=end; previous=run
    if [x['absolute_step'] for x in allrows]!=list(range(absolute+1)): raise ValueError('noncontiguous chain')
    report={'schema':'r164_checkpoint_chain_v1','scope':'fresh_cache_diagnostic_no_transport_or_event_history','cache_history':'unavailable','physical_event_history':'unavailable','links':links,'segments':segments,'rows':allrows}
    with Path(output).open('x',encoding='utf8',newline='\n') as f: f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('parent',type=Path); p.add_argument('step',type=int); p.add_argument('destination',type=Path); p.add_argument('--steps',type=int,required=True)
    v=sub.add_parser('verify'); v.add_argument('captures',nargs='+',type=Path); v.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.command=='prepare': prepare(a.parent,a.step,a.destination,a.steps)
    else:
        if len(a.captures)<2: ap.error('verify requires root and child captures')
        verify(a.captures,a.output)
if __name__=='__main__': main()
