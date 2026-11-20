#!/usr/bin/env python3
"""Replay descriptor-cutoff geometry on one recorded native bead drift."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from r164_all_bead_geometric_replay import inspect_inputs, read_endpoint
from r164_locate_native_crossing import normal_matrix, reconstruct, validate_endpoint
from r164_audit_force_split import headers, loadtxt, read_edges
from r164_verify_short_runs import sha

BLOCK = 128
MAX_DEPTH = 18


def scan_pair(state, acceleration, velocity_bound, distance_bound, cutoff, dt, cushion):
    """Bounded floating replay: f is squared distance, cushion is in angstrom."""
    f_cushion = 2*cutoff*cushion + cushion*cushion
    second = 2*(velocity_bound**2 + distance_bound*acceleration)
    roots, unresolved = [], []
    def evaluate(t):
        d,v = state(t)
        return float(d@d-cutoff**2),float(2*d@v)
    def visit(t0,t1,f0,f1,depth):
        width=t1-t0
        chord=second*width*width/8
        if min(f0,f1)-chord>f_cushion or max(f0,f1)+chord < -f_cushion:
            return
        if min(f0,f1)-chord >= -f_cushion and max(f0,f1)+chord <= f_cushion:
            unresolved.append({'time_interval_fs':[t0,t1],'reason':'entire enclosure within cutoff cushion'})
            return
        mid=(t0+t1)/2
        fm,derivative=evaluate(mid)
        floor=abs(derivative)-second*width/2
        if floor > f_cushion/max(dt,1e-30):
            sign=1 if derivative>0 else -1
            a,b=sign*f0,sign*f1
            if a < -f_cushion and b > f_cushion:
                lo,hi=t0,t1
                for _ in range(60):
                    t=(lo+hi)/2
                    if sign*evaluate(t)[0]<0: lo=t
                    else: hi=t
                t=(lo+hi)/2
                roots.append({'relative_time_fs':t,'bracket_fs':[lo,hi],
                              'direction':sign,'fprime_A2_per_fs':evaluate(t)[1],
                              'fprime_lower_bound_A2_per_fs':floor,
                              'arithmetic_time_interval_fs':[max(t0,t-f_cushion/floor),min(t1,t+f_cushion/floor)]})
            elif not (a>f_cushion or b < -f_cushion):
                unresolved.append({'time_interval_fs':[t0,t1],'reason':'endpoint within cutoff cushion'})
            return
        if depth>=MAX_DEPTH:
            unresolved.append({'time_interval_fs':[t0,t1],'reason':'nonmonotonic, tangent or clustered roots'})
            return
        visit(t0,mid,f0,fm,depth+1)
        visit(mid,t1,fm,f1,depth+1)
    f0,_=evaluate(0); f1,_=evaluate(dt)
    visit(0.,dt,f0,f1,0)
    return roots,unresolved


def load_graph(info, endpoint, step, bead):
    d=info['capture']; receipt_path=d/f'pppm_trace_step_{step}_receipt.txt'
    receipt={}
    for line in receipt_path.read_text(encoding='utf8').splitlines():
        parts=line.split()
        if len(parts)!=2 or parts[0] in receipt: raise ValueError('malformed graph receipt')
        receipt[parts[0]]=parts[1]
    if (receipt.get('schema')!='r164_pppm_trace_all_v1' or receipt.get('atom')!='all'
            or receipt.get('bead')!=str(bead) or receipt.get('step')!=str(step)
            or receipt.get('force_evaluation_id')!=str(step*info['P']+bead+1)
            or receipt.get('neighbor_history')!='all_centers_selected_bead_endpoint'
            or receipt.get('physical_event')!='unavailable'):
        raise ValueError('all graph receipt identity mismatch')
    prefix=f'pppm_trace_step_{step}_bead_{bead}'
    meta=d/f'{prefix}_all_meta.out'; binary=d/f'{prefix}_all_edges.bin'
    hashes={'receipt':sha(receipt_path)}
    for name,path in (('all_meta',meta),('all_edges',binary)):
        hashes[name]=sha(path)
        if hashes[name]!=receipt.get(name+'_sha256'): raise ValueError('all graph SHA mismatch: '+name)
    with (d/'scalar.txt').open('rb') as source:
        rows=[line.rstrip(b'\r\n')+b'\n' for line in source if not line.startswith(b'#') and line.split() and int(line.split()[0])==step and line.endswith(b'\n')]
    if len(rows)!=1 or hashlib.sha256(rows[0]).hexdigest()!=receipt.get('scalar_line_sha256'):
        raise ValueError('graph scalar receipt mismatch')
    head=headers(meta); N=info['N']
    if (head.get('schema')!='r164_all_edges_v1' or head.get('path')!='large_descriptor_mic'
            or head.get('binary')!='little_endian record_bytes 76 order kind_slot_center'
            or head.get('step')!=str(step) or head.get('bead')!=str(bead)
            or head.get('force_evaluation_id')!=receipt['force_evaluation_id'] or head.get('N')!=str(N)):
        raise ValueError('unsupported all graph header')
    before=int(head.get('global_generation_before','-1')); after=int(head.get('global_generation_after','-1'))
    if before<0 or after<before or after-before>1: raise ValueError('invalid global cache generation')
    md=loadtxt(meta,4)
    if md.shape!=(N,4) or not np.array_equal(md[:,0],np.arange(N)):
        raise ValueError('invalid graph metadata')
    counts=md[:,1:3].astype(int)
    if not np.array_equal(counts,md[:,1:3]) or np.any(counts<0): raise ValueError('invalid graph counts')
    edge=read_edges(binary,counts)
    if (np.any(edge['kind']<0) or np.any(edge['kind']>1) or np.any(edge['center']<0)
            or np.any(edge['center']>=N) or np.any(edge['atom']<0) or np.any(edge['atom']>=N)
            or np.any(edge['atom']==edge['center']) or not all(np.isfinite(edge[x]).all() for x in ('r','B','e','g'))):
        raise ValueError('invalid all graph edges')
    order=np.lexsort((edge['center'],edge['slot'],edge['kind']))
    if not np.array_equal(order,np.arange(len(edge))): raise ValueError('graph slot ordering mismatch')
    index=edge['kind']*N+edge['center']
    actual=np.bincount(index,minlength=2*N).reshape(2,N).T
    if not np.array_equal(actual,counts): raise ValueError('graph edge counts mismatch')
    for kind in (0,1):
        z=edge[edge['kind']==kind]; z=z[np.lexsort((z['slot'],z['center']))]
        expected=np.arange(len(z))-np.repeat(np.cumsum(np.r_[0,counts[:-1,kind]]),counts[:,kind])
        if not np.array_equal(z['slot'],expected): raise ValueError('invalid graph slots')
        keys=z['center'].astype(np.int64)*N+z['atom']
        if len(np.unique(keys))!=len(keys): raise ValueError('duplicate directed graph edge')
    image=endpoint['image'][bead]; q=endpoint['frame'][0][bead]; cell=info['cell']
    lift=edge['image'].astype(np.int64)-image[edge['atom']]+image[edge['center']]
    delta=q[edge['atom']]-q[edge['center']]+lift*cell
    residual=np.linalg.norm(delta-edge['r'],axis=1)
    return edge,lift,hashes,{'step':step,'generation_before':before,'generation_after':after,
                              'max_native_lift_displacement_residual_A':float(np.max(residual)) if len(residual) else 0.0}


def replay(capture, runin, stdout, step, bead):
    info=inspect_inputs(Path(capture),Path(runin),Path(stdout),step,step+1)
    if not 0<=bead<info['P']: raise ValueError('invalid bead')
    info['identity_hashes']={x:sha(info['capture']/x) for x in ('metadata.txt','source_identity.txt')}
    left=read_endpoint(info,step); right=read_endpoint(info,step+1)
    edges=[]; graph_hashes=[]; graph_metadata=[]
    for frame,at in ((left,step),(right,step+1)):
        edge,lift,hashes,metadata=load_graph(info,frame,at,bead); edges.append((edge,lift)); graph_hashes.append(hashes); graph_metadata.append(metadata)
    model=next(info['runin'].parent/line[1] for line in info['lines'] if line[0]=='potential')
    lines=[x.split() for x in model.read_text(encoding='utf8').splitlines()[:3]]
    if len(lines)<2 or lines[0][0]!='nep4_charge2' or lines[1][0]!='cutoff' or len(lines[1])!=5:
        raise ValueError('unsupported mode2 cutoff model')
    cutoffs=np.array(lines[1][1:3],dtype=float)
    if not np.isfinite(cutoffs).all() or np.any(cutoffs<=0) or cutoffs[1]>cutoffs[0]: raise ValueError('invalid cutoffs')
    q0,p0,F0,_=left['frame']; q1,p1,F1,_=right['frame']; T=normal_matrix(info['P']); dt=info['dt']; cell=info['cell']; N=info['N']
    Q,V,omega,qhat,phat=reconstruct(q0,p0,F0,F1,info['mass'],dt,T,info['temp'])
    qerr,perr=validate_endpoint(qhat,phat,q1,p1)
    weights=T[bead,:,None,None]; qk=Q*weights; vk=V*weights
    start=np.sum(qk,axis=0); velocity=np.sum(vk,axis=0)
    acceleration_axes=np.sum(np.abs(qk)*omega[:,None,None]**2+np.abs(vk)*omega[:,None,None],axis=0)
    acceleration=np.linalg.norm(acceleration_axes,axis=1)
    displacement=np.linalg.norm(qhat[bead]-start,axis=1)+acceleration*dt*dt/8
    position_cushion=max(1e-10,float(np.max(np.linalg.norm(start-q0[bead],axis=1))),float(np.max(np.linalg.norm(qhat[bead]-q1[bead],axis=1))))
    geometry=[ [[],[]], [[],[]] ]; endpoint_ambiguous=[[[],[]],[[],[]]]; candidates=[[],[]]; pair_count=0
    for i0 in range(0,N,BLOCK):
        i=np.arange(i0,min(i0+BLOCK,N))[:,None]; j=np.arange(N)[None,:]; mask=j>i
        pair_count+=int(np.count_nonzero(mask))
        for endpoint_index,positions in enumerate((q0[bead],q1[bead])):
            raw=positions[None,:,:]-positions[i]
            delta=raw-np.rint(raw/cell)*cell
            distances=np.linalg.norm(delta,axis=2)
            for kind,cutoff in enumerate(cutoffs):
                a,b=np.where(mask & (distances<cutoff))
                geometry[endpoint_index][kind].extend(((i0+a)*N+b).tolist())
                a,b=np.where(mask & (np.abs(distances-cutoff)<=2*position_cushion))
                endpoint_ambiguous[endpoint_index][kind].extend((int(i0+x),int(y)) for x,y in zip(a,b))
        raw=start[None,:,:]-start[i]
        image=-np.rint(raw/cell).astype(np.int64); delta=raw+image*cell
        distances=np.linalg.norm(delta,axis=2)
        bound=displacement[i]+displacement[None,:]+2*position_cushion
        for kind,cutoff in enumerate(cutoffs):
            a,b=np.where(mask & (np.abs(distances-cutoff)<=bound))
            if np.any(distances[a,b]+bound[a,b]>=np.min(cell)/2): raise ValueError('candidate fixed MIC image not bounded')
            candidates[kind].extend((int(i0+x),int(y),image[x,y].tolist(),float(distances[x,y]),float(bound[x,y])) for x,y in zip(a,b))
    if pair_count!=N*(N-1)//2: raise ValueError('pair enumeration count mismatch')
    graph_reports=[]; fp_tolerance=float(8*float(np.finfo(np.float32).eps)*float(np.max(cell)))
    for endpoint_index,(edge,lift) in enumerate(edges):
        positions=(q0,q1)[endpoint_index][bead]; reports=[]
        if graph_metadata[endpoint_index]['max_native_lift_displacement_residual_A']>fp_tolerance:
            raise ValueError('native graph displacement exceeds float comparison cushion')
        for kind,cutoff in enumerate(cutoffs):
            z=edge[edge['kind']==kind]; images=lift[edge['kind']==kind]
            keys=z['center'].astype(np.int64)*N+z['atom']; reverse=z['atom'].astype(np.int64)*N+z['center']
            undirected=np.asarray(geometry[endpoint_index][kind],dtype=np.int64)
            expected=np.sort(np.concatenate((undirected,(undirected%N)*N+undirected//N)))
            extra=np.setdiff1d(keys,expected); missing=np.setdiff1d(expected,keys)
            def classify(keys):
                if not len(keys): return {'within_float_cushion':0,'outside_float_cushion':0,'entries':[]}
                i=keys//N; j=keys%N; delta=positions[j]-positions[i]; delta-=np.rint(delta/cell)*cell
                margin=np.linalg.norm(delta,axis=1)-cutoff
                return {'within_float_cushion':int(np.count_nonzero(np.abs(margin)<=fp_tolerance)),
                        'outside_float_cushion':int(np.count_nonzero(np.abs(margin)>fp_tolerance)),
                        'entries':[{'center':int(a),'atom':int(b),'signed_margin_A':float(m)} for a,b,m in zip(i,j,margin)]}
            image_expected=-np.rint((positions[z['atom']]-positions[z['center']])/cell).astype(np.int64)
            if not np.array_equal(images,image_expected): raise ValueError('native graph lift image differs from MIC')
            native_radius=np.linalg.norm(z['r'].astype(float),axis=1)
            if np.any(native_radius>cutoff+fp_tolerance): raise ValueError('native edge outside cutoff cushion')
            reports.append({'step':step+endpoint_index,'kind':('radial','angular')[kind],
                            'directed_edges':len(keys),'geometric_directed_edges':len(expected),
                            'reverse_missing':len(np.setdiff1d(reverse,keys)),
                            'native_only':classify(extra),'geometry_only':classify(missing)})
        graph_reports.append(reports)
    roots=[]; unresolved=[]
    for kind,entries in enumerate(candidates):
        for i,j,image,r0,bound in entries:
            dq=qk[:,j]-qk[:,i]; dv=vk[:,j]-vk[:,i]; img=np.array(image)*cell
            def state(t):
                wt=omega*t; co=np.cos(wt); si=np.sin(wt); fac=np.empty_like(omega); fac[0]=t; fac[1:]=si[1:]/omega[1:]
                return np.sum(co[:,None]*dq+fac[:,None]*dv,axis=0)+img,np.sum(-omega[:,None]*si[:,None]*dq+co[:,None]*dv,axis=0)
            acc=acceleration[i]+acceleration[j]; vbound=np.linalg.norm(velocity[j]-velocity[i])+acc*dt
            found,unknown=scan_pair(state,acc,vbound,r0+bound,float(cutoffs[kind]),dt,2*position_cushion)
            for item in found:
                item.update(kind=('radial','angular')[kind],i=i,j=j,lift_image=image,absolute_time_fs=step*dt+item['relative_time_fs'])
                roots.append(item)
            for item in unknown:
                item.update(kind=('radial','angular')[kind],i=i,j=j,lift_image=image)
                unresolved.append(item)
    reconciliations=[]
    for kind in range(2):
        start_keys=set(geometry[0][kind]); end_keys=set(geometry[1][kind]); root_net={}; root_count={}
        unresolved_keys=set()
        for item in roots:
            if item['kind']==('radial','angular')[kind]:
                key=item['i']*N+item['j']; root_net[key]=root_net.get(key,0)+item['direction']; root_count[key]=root_count.get(key,0)+1
        for item in unresolved:
            if item['kind']==('radial','angular')[kind]: unresolved_keys.add(item['i']*N+item['j'])
        candidate_keys={i*N+j for i,j,_,_,_ in candidates[kind]}
        changed=start_keys^end_keys
        missing_candidates=sorted(changed-candidate_keys)
        mismatch=[]
        for key in sorted(changed | set(root_net)):
            expected=int(key in start_keys)-int(key in end_keys); observed=root_net.get(key,0)
            if expected!=observed:
                mismatch.append({'i':key//N,'j':key%N,'expected_net':expected,'root_net':observed,'has_unresolved':key in unresolved_keys})
        reconciliations.append({'kind':('radial','angular')[kind], 'endpoint_changed_pairs':len(changed),
                                'changed_pairs_without_candidate':[{'i':x//N,'j':x%N} for x in missing_candidates],
                                'net_mismatches':mismatch, 'same_side_multiroot_pairs':sum(count>1 and key not in changed for key,count in root_count.items()),
                                'unresolved_pair_count':len(unresolved_keys)})
    incomplete=bool(unresolved or any(endpoint_ambiguous[t][k] for t in range(2) for k in range(2)) or any(x['net_mismatches'] or x['changed_pairs_without_candidate'] for x in reconciliations)
                    or any(z['native_only']['outside_float_cushion'] or z['geometry_only']['outside_float_cushion'] for frame in graph_reports for z in frame))
    return {'schema':'r164_descriptor_cutoff_replay_v1','scope':'single-bead numerical geometry; no physical event, impulse, event first moment or P11/P12 release',
            'capture':str(info['capture']),'source_identity':info['source'],'run_in_sha256':sha(info['runin']),
            'stdout_sha256':sha(info['stdout']),'model_sha256':sha(model),'endpoint_frame_image_sha256':[left['hashes'],right['hashes']],
            'graph_sha256':graph_hashes,'graph_metadata':graph_metadata,'pair_count':pair_count,
            'status':'INCOMPLETE' if incomplete else 'GEOMETRY_REPLAY_COMPLETE_NO_PHYSICAL_ACCEPTANCE',
            'pair_net_reconciliation':reconciliations,
            'endpoint_within_position_cushion':[[[{'i':i,'j':j} for i,j in group] for group in frame] for frame in endpoint_ambiguous],
            'step':step,'bead':bead,'force_ids':[step*info['P']+bead+1,(step+1)*info['P']+bead+1],
            'dt_fs':dt,'cutoff_A':{'radial':float(cutoffs[0]),'angular':float(cutoffs[1])},
            'coordinate_cushion_A':position_cushion,'native_float_cutoff_comparison_cushion_A':fp_tolerance,
            'cushion_basis':'8 float32 eps times maximum cell length; diagnostic rounding envelope, not a rigorous interval or physical threshold',
            'root_time_scope':'arithmetic_time_interval_fs bounds only reconstructed floating trajectory resolution; does not bound native FP32 cutoff switching time',
            'endpoint_replay_max_q_A':qerr,'endpoint_replay_max_p_eV_fs_per_A':perr,
            'candidate_count':{'radial':len(candidates[0]),'angular':len(candidates[1])},
            'endpoint_graph_comparison':graph_reports,'roots':roots,'unresolved':unresolved}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path); p.add_argument('runin',type=Path); p.add_argument('stdout',type=Path)
    p.add_argument('--step',required=True,type=int); p.add_argument('--bead',required=True,type=int); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); result=replay(a.capture,a.runin,a.stdout,a.step,a.bead)
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    with a.output.open('x',encoding='utf8',newline='\n') as output: output.write(encoded)

if __name__=='__main__': main()
