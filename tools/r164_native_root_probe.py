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
    v_min=float(root['velocity_lower_bound_A_per_fs'])
    if not np.isfinite(v_min) or v_min<=0 or v_min>speed: raise ValueError('invalid monotonic velocity bound')
    half=OFFSET/v_min
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
        if np.count_nonzero(di)!=0 or np.count_nonzero(dl)!=1 or dl[target]!=root['sign'] or min(first[2][target],second[2][target])!=root['face_index']: raise ValueError('assignment probe changes unintended branch or face')
    elif kind=='periodic_chart':
        if np.count_nonzero(di)!=1 or di[target]!=root['sign'] or np.count_nonzero(dl)!=0 or max(first[1][target],second[1][target])!=root['face_index']: raise ValueError('periodic probe changes unintended branch or face')
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



def review(directory, expected_manifest_sha, output):
    import hashlib
    from r164_frame_to_npz import read_frame
    from r164_verify_short_runs import keyfile, read_images
    from r164_audit_force_split import headers, loadtxt

    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    if sha(manifest_path) != expected_manifest_sha.lower():
        raise ValueError('manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('schema') != 'r164_native_root_probe_v1':
        raise ValueError('invalid probe manifest')
    if sha(Path(manifest['geometry_path'])) != manifest['geometry_sha256']:
        raise ValueError('geometry SHA mismatch')
    for path, digest in manifest['source_inputs'].items():
        if sha(Path(path)) != digest:
            raise ValueError('source input SHA mismatch')
    selected = manifest['selected_root']
    bead, atom = selected['bead'], selected['atom']
    results = {}
    for side in ('before', 'after', 'before-repeat'):
        folder = root / side
        for name, digest in manifest['decks'][side].items():
            if sha(folder / name) != digest:
                raise ValueError('deck SHA mismatch: ' + side + '/' + name)
        capture = folder / 'trace_out'
        if (capture / 'COMPLETE').read_text(encoding='utf8').strip() != 'r164_diagnostic_v1 final_step 1':
            raise ValueError('incomplete probe capture')
        receipt_path = capture / 'pppm_trace_step_0_receipt.txt'
        receipt = {}
        for line in receipt_path.read_text(encoding='utf8').splitlines():
            parts = line.split()
            if len(parts) != 2 or parts[0] in receipt:
                raise ValueError('invalid trace receipt')
            receipt[parts[0]] = parts[1]
        if (receipt.get('schema') != 'r164_pppm_trace_v3' or receipt.get('step') != '0'
                or receipt.get('bead') != str(bead) or receipt.get('atom') != str(atom)
                or receipt.get('force_evaluation_id') != str(bead + 1)
                or receipt.get('neighbor_history') != 'partial_target_descriptor_only'
                or receipt.get('physical_event') != 'unavailable'):
            raise ValueError('trace receipt identity mismatch')
        prefix = f'pppm_trace_step_0_bead_{bead}'
        files = {'metadata': capture/'metadata.txt', 'source_identity': capture/'source_identity.txt',
                 'frame': capture/'frame_0.txt', 'force_images': capture/'force_images_0.txt'}
        for name, suffix in (('mesh','mesh.out'), ('kspace','kspace.out'), ('atom','atom.out'),
                             ('assignment_atom','assignment_atom.out'),
                             ('assignment_stencil',f'assignment_stencil_atom{atom}.out'),
                             ('pre_electro','pre_electro.out'), ('descriptor_neighbors','descriptor_neighbors.out'),
                             ('local_edge_derivatives','local_edge_derivatives.out')):
            files[name] = capture / f'{prefix}_{suffix}'
        hashes = {name: sha(path) for name, path in files.items()}
        for name, digest in hashes.items():
            if receipt.get(name + '_sha256') != digest:
                raise ValueError('trace SHA mismatch: ' + name)
        scalar_lines = (capture/'scalar.txt').read_bytes().splitlines(keepends=True)
        scalar_rows = [line for line in scalar_lines if line and not line.startswith(b'#')]
        if len(scalar_rows) != 2 or any(not line.endswith(b'\n') for line in scalar_rows):
            raise ValueError('invalid complete scalar rows')
        if [line.split()[0] for line in scalar_rows] != [b'0', b'1']:
            raise ValueError('invalid scalar steps')
        canonical = scalar_rows[0].rstrip(b'\r\n') + b'\n'
        if hashlib.sha256(canonical).hexdigest() != receipt.get('scalar_line_sha256'):
            raise ValueError('scalar receipt mismatch')
        meta, source = keyfile(files['metadata']), keyfile(files['source_identity'])
        if source.get('bead_force_backend') != 'native_GPUMD':
            raise ValueError('probe is not native GPUMD')
        identities = {'run_in_sha256':'run.in', 'model_xyz_sha256':'model.xyz',
                      'restart_input_sha256':'restart_beads.xyz', 'model_sha256':manifest['potential_name']}
        for field, name in identities.items():
            if source.get(field) != manifest['decks'][side][name]:
                raise ValueError('capture input identity mismatch: ' + field)
        P, N = int(meta['P']), int(meta['N'])
        cell = np.array(meta['cell_angstrom'].split(), dtype=float)
        q, momentum, force, energy = read_frame(files['frame'], P, N)
        images = read_images(files['force_images'], q, cell)
        with np.load(folder/'phase.npz', allow_pickle=False) as phase:
            if (phase['q_A'].shape != q.shape or phase['p_eV_fs_per_A'].shape != momentum.shape
                    or not np.isfinite(phase['q_A']).all() or not np.isfinite(phase['p_eV_fs_per_A']).all()
                    or np.max(np.abs(q - phase['q_A'])) > 3e-11
                    or np.max(np.abs(momentum - phase['p_eV_fs_per_A'])) > 1e-8):
                raise ValueError('capture phase differs from prepared phase')
        for name in ('mesh', 'kspace', 'atom', 'assignment_atom', 'assignment_stencil',
                     'pre_electro', 'descriptor_neighbors', 'local_edge_derivatives'):
            h = headers(files[name])
            if h.get('force_evaluation_id') != str(bead + 1):
                raise ValueError('force evaluation id mismatch: ' + name)
        atom_data = loadtxt(files['atom'], 7)
        pre = loadtxt(files['pre_electro'], 3)
        assignment = loadtxt(files['assignment_atom'], 29)
        stencil = loadtxt(files['assignment_stencil'], 9)
        if (atom_data.shape != (N,7) or pre.shape != (N,3)
                or not np.array_equal(atom_data[:,0],np.arange(N))
                or not np.array_equal(pre[:,0],np.arange(N))
                or assignment.shape != (1,29) or assignment[0,0] != atom
                or stencil.shape != (125,9)):
            raise ValueError('invalid native PPPM rows')
        if (not np.array_equal(assignment[0,2:5], atom_data[atom,1:4])
                or assignment[0,1] != atom_data[atom,4]):
            raise ValueError('target assignment does not match native atom')
        if np.max(np.abs(atom_data[:,1:4] - (q[bead] - images[bead]*cell))) > 2e-12:
            raise ValueError('PPPM atom positions mismatch')
        def edges(path, values):
            rows = []
            for line in path.read_text(encoding='utf8').splitlines():
                if not line or line.startswith('#'):
                    continue
                fields = line.split()
                if fields[0] not in ('radial','angular') or len(fields) != values:
                    raise ValueError('invalid neighbor row')
                row = [0 if fields[0]=='radial' else 1] + [float(x) for x in fields[1:]]
                if not np.isfinite(row).all():
                    raise ValueError('nonfinite neighbor row')
                rows.append(row)
            return np.asarray(rows, dtype=float).reshape((-1, values))
        neighbors = edges(files['descriptor_neighbors'], 9)
        derivatives = edges(files['local_edge_derivatives'], 18)
        if len(neighbors) != len(derivatives) or not np.array_equal(neighbors, derivatives[:,:9]):
            raise ValueError('neighbor and derivative slots differ')
        for name in ('descriptor_neighbors','local_edge_derivatives'):
            h = headers(files[name])
            if h.get('step') != '0' or h.get('bead') != str(bead) or h.get('atom') != str(atom):
                raise ValueError('neighbor source identity mismatch')
            for kind_id, key in ((0,'radial_count'),(1,'angular_count')):
                slots = neighbors[neighbors[:,0] == kind_id,1]
                if int(h[key]) != len(slots) or not np.array_equal(slots,np.arange(len(slots))):
                    raise ValueError('neighbor slot order/count mismatch')
        if len(neighbors):
            indexes = neighbors[:,2].astype(int)
            if np.any(indexes<0) or np.any(indexes>=N) or not np.array_equal(indexes, neighbors[:,2]):
                raise ValueError('invalid neighbor atom')
            lifted = neighbors[:,:6].copy()
            lifted[:,3:6] += images[bead,atom] - images[bead,indexes]
        else:
            lifted = neighbors[:,:6].copy()
        mesh = np.array(headers(files['assignment_atom'])['pppm_mesh_size'].split(), dtype=int)
        if mesh.shape != (3,) or np.any(mesh <= 0): raise ValueError('invalid mesh')
        if source.get('build_git_dirty') != 'no' or len(source.get('build_git_commit','')) != 40:
            raise ValueError('probe requires known clean build identity')
        results[side] = dict(cell=cell, mesh=mesh, q=q, p=momentum, F=force, U=energy, images=images,
                             projected_q=atom_data[:,4], raw_q=pre[:,1], local_energy=pre[:,2],
                             assignment=assignment, stencil=stencil, neighbors=neighbors,
                             derivatives=derivatives, lifted_edges=lifted, hashes=hashes,
                             receipt_sha256=sha(receipt_path), source=source)
    target = (bead, atom, 'xyz'.index(selected['axis']))
    a, b, repeat = (results[x] for x in ('before','after','before-repeat'))
    for state in (b,repeat):
        if (state['q'].shape != a['q'].shape or not np.array_equal(state['cell'],a['cell'])
                or not np.array_equal(state['mesh'],a['mesh'])
                or state['source'].get('build_git_commit') != a['source'].get('build_git_commit')
                or state['source'].get('build_git_dirty') != a['source'].get('build_git_dirty')):
            raise ValueError('probe capture geometry or build mismatch')
    face, sign = selected['lift_boundary_A'], selected['sign']
    for side, state, direction in (('before',a,-1),('after',b,1),('before-repeat',repeat,-1)):
        if abs((state['q'][target]-face)-direction*sign*OFFSET) > 3e-11:
            raise ValueError('actual capture is on wrong probe side: ' + side)
    for other in (b,repeat):
        delta = other['q'] - a['q']; delta[target] = 0
        if np.max(np.abs(delta)) > 3e-11 or np.max(np.abs(other['p']-a['p'])) > 1e-8:
            raise ValueError('non-target phase differs across probes')
    image_delta = b['images'] - a['images']
    label_a = a['assignment'][0,8:11] + a['images'][bead,atom]*a['mesh']
    label_b = b['assignment'][0,8:11] + b['images'][bead,atom]*b['mesh']
    label_delta = label_b-label_a; axis = target[2]
    if selected['kind'] == 'assignment_center':
        if (np.count_nonzero(image_delta) or np.count_nonzero(label_delta)!=1 or label_delta[axis]!=sign
                or min(label_a[axis],label_b[axis])!=selected['face_index']):
            raise ValueError('actual assignment probe branch mismatch')
    elif selected['kind'] == 'periodic_chart':
        if (np.count_nonzero(image_delta)!=1 or image_delta[target]!=sign or np.count_nonzero(label_delta)
                or max(a['images'][target],b['images'][target])!=selected['face_index']):
            raise ValueError('actual periodic probe branch mismatch')
    else: raise ValueError('unsupported probe kind')
    if not np.array_equal(a['images'],repeat['images']) or not np.array_equal(a['assignment'][0,8:11],repeat['assignment'][0,8:11]):
        raise ValueError('actual before-repeat branch mismatch')
    def metric(a, b):
        delta = b-a
        if delta.size == 0:
            return {'max_abs':0.0, 'count_changed':0}
        index = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        return {'max_abs':float(abs(delta[index])), 'signed_at_max':float(delta[index]),
                'index':[int(x) for x in index], 'count_changed':int(np.count_nonzero(delta))}
    comparisons = {}
    for other in ('after','before-repeat'):
        a, b = results['before'], results[other]
        item = {name:metric(a[name],b[name]) for name in
                ('q','p','F','U','images','projected_q','raw_q','local_energy','assignment','stencil')}
        item['selected_bead_F'] = metric(a['F'][bead],b['F'][bead])
        item['selected_bead_U'] = metric(a['U'][bead:bead+1],b['U'][bead:bead+1])
        item['edge_comparison_scope'] = 'lifted directed edge multiset; derivatives only by unchanged raw slot'
        item['raw_neighbor_slots_equal'] = bool(np.array_equal(a['neighbors'][:,:6],b['neighbors'][:,:6]))
        edge_a = sorted(map(tuple,a['lifted_edges'][:,[0,2,3,4,5]]))
        edge_b = sorted(map(tuple,b['lifted_edges'][:,[0,2,3,4,5]]))
        item['lifted_directed_edges_equal'] = edge_a == edge_b
        item['derivatives_comparable_by_raw_slot'] = item['raw_neighbor_slots_equal']
        if item['raw_neighbor_slots_equal']:
            item['edge_derivative_difference'] = metric(a['derivatives'][:,9:],b['derivatives'][:,9:])
        comparisons[other] = item
    report = {'schema':'r164_native_root_probe_review_v1', 'manifest_sha256':sha(manifest_path),
              'scope':'fresh-cache numerical comparison; no inferred impulse, winding, physical transfer or transport',
              'noise_interpretation':'single same-side repeat is an observed comparison, not a noise bound; no inference above observed noise', 'selected_root':selected,
              'captures':{side:{'receipt_sha256':r['receipt_sha256'],'file_sha256':r['hashes'],
                                'force_evaluation_id':bead+1, 'source_identity':r['source']}
                          for side,r in results.items()}, 'comparisons':comparisons}
    encoded = json.dumps(report,indent=2,allow_nan=False) + '\n'
    with Path(output).open('x',encoding='utf8',newline='\n') as f:
        f.write(encoded)

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('geometry',type=Path); p.add_argument('--geometry-sha256',required=True); p.add_argument('--step',required=True,type=int); p.add_argument('--bead',required=True,type=int); p.add_argument('--atom',required=True,type=int); p.add_argument('--axis',required=True,choices='xyz'); p.add_argument('--kind',required=True,choices=('assignment_center','periodic_chart')); p.add_argument('--root-index',type=int,default=0); p.add_argument('--destination',required=True,type=Path)
    r=sub.add_parser('review'); r.add_argument('directory',type=Path); r.add_argument('--manifest-sha256',required=True); r.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.command=='prepare': prepare(a.geometry,a.geometry_sha256,a.step,a.bead,a.atom,a.axis,a.kind,a.root_index,a.destination)
    else: review(a.directory,a.manifest_sha256,a.output)
if __name__=='__main__': main()

