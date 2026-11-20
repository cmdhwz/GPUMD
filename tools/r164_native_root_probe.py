#!/usr/bin/env python3
"""Prepare and review fresh-cache native probes of a reconstructed geometric root."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from r164_all_bead_geometric_replay import inspect_inputs, read_endpoint, endpoint_labels
from r164_locate_native_crossing import normal_matrix, reconstruct, mode_state, validate_endpoint
from r164_verify_short_runs import sha

OFFSET=2e-9


def prepare(geometry,expected_sha,step,bead,atom,axis,kind,root_index,destination):
    geometry=Path(geometry).resolve()
    if sha(geometry)!=expected_sha.lower(): raise ValueError('geometry SHA mismatch')
    doc=json.loads(geometry.read_text(encoding='utf8'))
    if doc.get('schema')!='r164_all_bead_geometric_replay_v1': raise ValueError('unsupported geometry schema')
    capture=Path(doc['capture']).resolve(); inputs=doc['input_sha256']
    def input_path(name):
        matches=[Path(x) for x in inputs if Path(x).name==name]
        if len(matches)!=1: raise ValueError('ambiguous geometry input: '+name)
        return matches[0]
    for path,expected in inputs.items():
        if sha(Path(path))!=expected: raise ValueError('geometry source hash mismatch: '+path)
    info=inspect_inputs(capture,input_path('run.in'),input_path('stdout.log'),step,step+1)
    if info['source']!=doc['source_identity'] or info['dt']!=doc['dt_fs']: raise ValueError('geometry source identity mismatch')
    steps=[x for x in doc['steps'] if x['step']==step]
    if len(steps)!=1: raise ValueError('missing or duplicate geometry step')
    entry=steps[0]; selected=[x for x in entry['roots'] if (x['bead'],x['atom'],x['axis'],x['kind'])==(bead,atom,axis,kind)]
    if not 0<=root_index<len(selected): raise ValueError('root index outside selected roots')
    root=selected[root_index]
    if not 0<=bead<info['P'] or not 0<=atom<info['N'] or axis not in 'xyz' or root['sign'] not in (-1,1): raise ValueError('invalid root identity')
    if entry.get('unresolved') or any(entry['endpoint_net_mismatches'].values()): raise ValueError('geometry step is incomplete or unresolved')
    window=root['arithmetic_time_interval_fs']; speed=abs(float(root['velocity_A_per_fs']))
    if speed<=0 or len(window)!=2 or window[0]>root['relative_time_fs'] or window[1]<root['relative_time_fs']: raise ValueError('invalid root time bound')
    half=OFFSET/speed
    if window[0]-half<=0 or window[1]+half>=info['dt']: raise ValueError('root probe window exceeds step')
    for other in entry['roots']:
        if other is root: continue
        a,b=other['arithmetic_time_interval_fs']
        if a<=window[1]+half and b>=window[0]-half: raise ValueError('root probe time window overlaps another crossing')
    info['identity_hashes']={name:sha(capture/name) for name in ('metadata.txt','source_identity.txt')}
    left=read_endpoint(info,step); right=read_endpoint(info,step+1)
    for endpoint in (left,right):
        for name,digest in endpoint['hashes'].items():
            if entry['input_sha256'].get(name)!=digest: raise ValueError('geometry endpoint hash mismatch: '+name)
    q0,p0,F0,_=left['frame']; q1,p1,F1,_=right['frame']
    T=normal_matrix(info['P']); Q,V,omega,qhat,phat=reconstruct(q0,p0,F0,F1,info['mass'],info['dt'],T,info['temp'])
    validate_endpoint(qhat,phat,q1,p1)
    root_time=float(root['relative_time_fs']); qt,vt=mode_state(Q,V,omega,root_time)
    bracket=root['bracket_fs']
    if len(bracket)!=2 or not bracket[0]<=root_time<=bracket[1]: raise ValueError('root time outside bisection bracket')
    q=np.einsum('sk,kia->sia',T,qt); p=np.einsum('sk,kia->sia',T,vt)*info['mass'][None,:,None]
    ax='xyz'.index(axis)
    actual_velocity=float(np.einsum('k,k->',T[bead],vt[:,atom,ax]))
    if not np.isfinite(actual_velocity) or actual_velocity*root['sign']<=0 or abs(actual_velocity-root['velocity_A_per_fs'])>1e-10: raise ValueError('root velocity mismatch')
    face=float(root['lift_boundary_A']); tolerance=max(2e-10,2*float(root['coordinate_cushion_A']))
    if not np.isfinite(q).all() or not np.isfinite(p).all() or abs(q[bead,atom,ax]-face)>tolerance: raise ValueError('root coordinate does not bind to reconstructed face')
    samples={}
    for label,direction in (('before',-1),('after',1),('before-repeat',-1)):
        state=q.copy(); state[bead,atom,ax]=face+direction*root['sign']*OFFSET
        if np.any(state < -info['cell']) or np.any(state > 2*info['cell']): raise ValueError('root phase outside one-wrap restart range')
        images=np.floor(state/info['cell']).astype(np.int64)
        if np.any(images < -1) or np.any(images>1): raise ValueError('root image outside one-wrap range')
        labels=endpoint_labels(state,images,info['cell'],info['mesh'])
        samples[label]=(state,images,labels)
    first=samples['before']; second=samples['after']; target=(bead,atom,ax)
    di=second[1]-first[1]; dl=second[2]-first[2]
    if kind=='assignment_center':
        if np.count_nonzero(di)!=0 or np.count_nonzero(dl)!=1 or dl[target]!=root['sign']: raise ValueError('assignment probe changes unintended branch')
    elif kind=='periodic_chart':
        if np.count_nonzero(di)!=1 or di[target]!=root['sign'] or np.count_nonzero(dl)!=0: raise ValueError('periodic probe changes unintended branch')
    else: raise ValueError('unsupported root kind')
    if np.any(np.abs(q-q[0:1])>=info['cell'][None,None,:]/2-1e-9): raise ValueError('ambiguous root bead branch')
    target=Path(destination).resolve(); target.mkdir(); potential=next(x[1] for x in info['lines'] if x[0]=='potential')
    if Path(potential).name!=potential or potential in ('run.in','model.xyz','restart_beads.xyz','phase.npz'): raise ValueError('unsupported potential filename')
    decks={}
    for label,(phase,_,_) in samples.items():
        folder=target/label; folder.mkdir(); phase_path=folder/'phase.npz'
        np.savez(phase_path,q_A=phase,p_eV_fs_per_A=p,mass=info['mass'],species=info['species'],cell=np.diag(info['cell']),atom_row_zero_based=np.arange(info['N']),bead_order_zero_based=np.arange(info['P']))
        subprocess.run([sys.executable,str(Path(__file__).with_name('r164_phase_to_restart.py')),str(phase_path),str(folder/'restart_beads.xyz'),'--temperature',str(info['temp']),'--model-output',str(folder/'model.xyz')],check=True)
        shutil.copyfile(info['runin'].parent/potential,folder/potential)
        card=[['potential',potential],['kspace','pppm','0.5'],['time_step',str(info['dt'])],['ensemble','rpmd',str(info['P']),str(info['temp'])],['pimd_propagator','exact'],['pimd_fix_com','off'],['read_pimd_restart','restart_beads.xyz'],['r164_rpmd','trace_out','0','pppm_trace','0',str(bead),str(atom)],['run','1']]
        (folder/'run.in').write_text(''.join(' '.join(x)+'\n' for x in card),encoding='utf8',newline='\n')
        decks[label]={name:sha(folder/name) for name in ('phase.npz','model.xyz','restart_beads.xyz',potential,'run.in')}
    manifest={'schema':'r164_native_root_probe_v1','scope':'fresh_cache_geometric_root_probe_no_impulse_event_or_transport_claim', 'geometry_path':str(geometry),'geometry_sha256':expected_sha.lower(),'source_capture':str(capture),'source_inputs':inputs,'source_identity':info['source'],'selected_step':step,'selected_root':root,'root_index':root_index,'offset_A':OFFSET,'root_coordinate_reconstruction_error_A':float(q[bead,atom,ax]-face),'decks':decks,'potential_name':potential,'required_review':'independent actual capture receipts and same-side noise comparison'}
    temp=target/'manifest.json.tmp'; temp.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf8'); temp.replace(target/'manifest.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('geometry',type=Path); p.add_argument('--geometry-sha256',required=True); p.add_argument('--step',required=True,type=int); p.add_argument('--bead',required=True,type=int); p.add_argument('--atom',required=True,type=int); p.add_argument('--axis',required=True,choices='xyz'); p.add_argument('--kind',required=True,choices=('assignment_center','periodic_chart')); p.add_argument('--root-index',type=int,default=0); p.add_argument('--destination',required=True,type=Path)
    a=ap.parse_args(); prepare(a.geometry,a.geometry_sha256,a.step,a.bead,a.atom,a.axis,a.kind,a.root_index,a.destination)
if __name__=='__main__': main()

