#!/usr/bin/env python3
"""Replay all geometric assignment and periodic-chart crossings in native R161 steps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from r164_locate_native_crossing import (C, fp32_boundary, normal_matrix, reconstruct, validate_endpoint,
                                          validate_inputs, validate_capture_path, stdout_mesh)
from r164_frame_to_npz import read_frame
from r164_verify_short_runs import (keyfile, sha, read_images, read_scalars, validate_frame_scalars,
                                    check_observer_mode, r164_options)

CUSHION=1e-10
MAX_DEPTH=18


def boundaries(length, count, low, high):
    result=[]
    for image in range(int(np.floor(low/length))-2,int(np.ceil(high/length))+3):
        result.append((float(image*length),'periodic_chart',image))
        for cut in range(count):
            edge,_,_=fp32_boundary(cut)
            result.append((float(image*length+edge*length/count),'assignment_center',image*count+cut))
    result.sort()
    return np.array([x[0] for x in result]),result


def scan_coordinate(state,accel,dt,faces,coordinate,step,cushion=CUSHION):
    values,descriptions=faces
    roots=[]
    unresolved=[]
    def scan(t0,t1,x0,x1,depth):
        width=t1-t0
        error=accel*width*width/8+cushion
        left=np.searchsorted(values,min(x0,x1)-error)
        right=np.searchsorted(values,max(x0,x1)+error,side='right')
        if left==right:
            return
        mid=(t0+t1)/2
        _,vm=state(mid)
        velocity_floor=abs(vm)-accel*width/2
        if velocity_floor>cushion/max(dt,1e-12):
            sign=1 if vm>0 else -1
            ambiguous=[]
            for face in descriptions[left:right]:
                edge,kind,label=face
                a,b=sign*(x0-edge),sign*(x1-edge)
                if a < -cushion and b > cushion:
                    lo,hi=t0,t1
                    for _ in range(60):
                        probe=(lo+hi)/2
                        if sign*(state(probe)[0]-edge)<0: lo=probe
                        else: hi=probe
                    root=(lo+hi)/2
                    roots.append({'bead':coordinate[0],'atom':coordinate[1],'axis':'xyz'[coordinate[2]],
                                  'kind':kind,'face_index':int(label),'lift_boundary_A':edge,'sign':sign,
                                  'relative_time_fs':root,'absolute_time_fs':step*dt+root,'bracket_fs':[lo,hi],
                                  'velocity_A_per_fs':state(root)[1],'velocity_lower_bound_A_per_fs':velocity_floor,
                                  'coordinate_cushion_A':cushion,'timing_resolution_proxy_fs':cushion/velocity_floor})
                elif not (a>cushion or b<-cushion):
                    ambiguous.append({'kind':kind,'face_index':int(label),'lift_boundary_A':edge})
            if ambiguous:
                unresolved.append({'bead':coordinate[0],'atom':coordinate[1],'axis':'xyz'[coordinate[2]],
                                   'time_interval_fs':[t0,t1],'faces':ambiguous,'reason':'endpoint overlaps coordinate cushion'})
            return
        if depth>=MAX_DEPTH:
            unresolved.append({'bead':coordinate[0],'atom':coordinate[1],'axis':'xyz'[coordinate[2]],
                               'time_interval_fs':[t0,t1],
                               'faces':[{'kind':x[1],'face_index':int(x[2]),'lift_boundary_A':x[0]} for x in descriptions[left:right]],
                               'reason':'nonmonotonic, tangential or unresolved close roots'})
            return
        xm,_=state(mid)
        scan(t0,mid,x0,xm,depth+1)
        scan(mid,t1,xm,x1,depth+1)
    x0,_=state(0)
    x1,_=state(dt)
    scan(0.,dt,x0,x1,0)
    return roots,unresolved


def endpoint_labels(q,image,cell,mesh):
    scaled=((q-image*cell)*(1./cell)*mesh).astype(np.float32)
    return image*mesh+np.trunc(scaled+np.float32(.5)).astype(np.int64)


def inspect_inputs(capture,runin,stdout,start,end):
    capture=capture.resolve(); runin=runin.resolve(); stdout=stdout.resolve()
    if capture.parent!=runin.parent or stdout.parent!=runin.parent: raise ValueError('capture, run.in and stdout must share run directory')
    meta=keyfile(capture/'metadata.txt'); source=keyfile(capture/'source_identity.txt')
    if source.get('model_sha256','').lower()!='fb5b6526ac2b244dba04fea86f09d0344e56b404d413503edad167cf2c39430c': raise ValueError('unsupported frozen R161 model')
    if meta.get('schema')!='r164_diagnostic_v1' or meta.get('time_convention')!='step_times_fixed_dt_start_zero': raise ValueError('invalid capture schema')
    marker=(capture/'COMPLETE').read_text(encoding='utf8').split()
    if len(marker)!=3 or marker[:2]!=['r164_diagnostic_v1','final_step']: raise ValueError('incomplete capture')
    total=int(marker[2]); dt=float(meta['dt_fs']); P=int(meta['P']); N=int(meta['N']); temp=float(meta['temperature_K'])
    if (P,N,temp)!=(32,5896,300.) or (dt,total) not in ((.1,1),(.1,10),(.2,5)) or not 0<=start<end<=total: raise ValueError('unsupported R161 card or step range')
    lines=[line.split('#',1)[0].split() for line in runin.read_text(encoding='utf8').splitlines()]; lines=[x for x in lines if x]
    keys=['potential','kspace','time_step','ensemble','pimd_propagator','pimd_fix_com','read_pimd_restart','r164_rpmd','run']
    if [x[0] for x in lines]!=keys or sha(runin)!=source.get('run_in_sha256'): raise ValueError('card or run.in identity mismatch')
    by={x[0]:x for x in lines}; r=by['r164_rpmd']; observer,_=r164_options(r)
    if (by['kspace']!=['kspace','pppm','0.5'] or by['ensemble']!=['ensemble','rpmd','32','300'] or
        by['pimd_propagator']!=['pimd_propagator','exact'] or by['pimd_fix_com']!=['pimd_fix_com','off'] or
        len(by['time_step'])!=2 or float(by['time_step'][1])!=dt or by['run']!=['run',str(total)]): raise ValueError('unsupported run configuration')
    validate_capture_path(lines,runin.parent,capture); validate_inputs(source,runin.parent,lines)
    mesh=np.array(stdout_mesh(stdout,dt,total)); records=read_scalars(capture/'scalar.txt',dt,total)
    check_observer_mode(capture,records,observer)
    atom_rows=[line.split()[1:] for line in (capture/'metadata.txt').read_text(encoding='utf8').splitlines() if line.startswith('atom ')]
    if len(atom_rows)!=N or [int(x[0]) for x in atom_rows]!=list(range(N)): raise ValueError('invalid atom order')
    mass=np.array([float(x[2])*C*C for x in atom_rows]); species=np.array([x[1] for x in atom_rows]); cell=np.array(meta['cell_angstrom'].split(),float)
    if not np.isfinite(mass).all() or np.any(mass<=0) or cell.shape!=(3,) or not np.allclose(cell,29.3104,atol=1e-8,rtol=0): raise ValueError('invalid masses or cell')
    return {'capture':capture,'runin':runin,'stdout':stdout,'source':source,'lines':lines,'dt':dt,'P':P,'N':N,'temp':temp,'mesh':mesh,'records':records,'mass':mass,'species':species,'cell':cell,'total':total}


def run_step(info,step,T):
    d=info['capture']; P=info['P']; N=info['N']; cell=info['cell']; mesh=info['mesh']; dt=info['dt']
    frames=[]; images=[]; inputs={}
    for i in (step,step+1):
        f=d/f'frame_{i}.txt'; image=d/f'force_images_{i}.txt'
        frame=read_frame(f,P,N); img=read_images(image,frame[0],cell)
        validate_frame_scalars(frame[0],frame[1],frame[3],info['mass'],info['species'],info['temp'],info['records'][i])
        if np.any(np.abs(frame[0]-frame[0][0:1])>=cell[None,None,:]/2-1e-9): raise ValueError('ambiguous first-bead lift branch')
        frames.append(frame); images.append(img); inputs[f.name]=sha(f); inputs[image.name]=sha(image)
    q,p,F,_=frames[0]; q1,p1,F1,_=frames[1]
    Q,V,omega,qhat,phat=reconstruct(q,p,F,F1,info['mass'],dt,T,info['temp'])
    qerr,perr=validate_endpoint(qhat,phat,q1,p1)
    qstart=np.einsum('sk,kia->sia',T,Q)
    initial_error=float(np.max(np.abs(qstart-q)))
    if initial_error>1e-8: raise ValueError('initial reconstruction exceeds arithmetic tolerance')
    cushion=CUSHION+max(initial_error,qerr)
    accel=np.einsum('sk,kia->sia',np.abs(T),np.abs(Q)*omega[:,None,None]**2+np.abs(V)*omega[:,None,None])
    seen=[]; unresolved=[]; candidates=0
    for axis,(length,count) in enumerate(zip(cell,mesh)):
        extra=accel[:,:,axis]*dt*dt/8+cushion
        face=boundaries(length,int(count),float(np.min(np.minimum(qstart[:,:,axis],qhat[:,:,axis])-extra)),float(np.max(np.maximum(qstart[:,:,axis],qhat[:,:,axis])+extra)))
        lower=np.searchsorted(face[0],np.minimum(qstart[:,:,axis],qhat[:,:,axis])-extra)
        upper=np.searchsorted(face[0],np.maximum(qstart[:,:,axis],qhat[:,:,axis])+extra,side='right')
        coords=np.argwhere(upper>lower); candidates+=len(coords)
        for bead,atom in coords:
            weights=T[bead]; qm=Q[:,atom,axis]; vm=V[:,atom,axis]
            def state(t):
                wt=omega*t; co=np.cos(wt); si=np.sin(wt); factor=np.r_[t,si[1:]/omega[1:]]
                return float(weights@(co*qm+factor*vm)),float(weights@(-omega*si*qm+co*vm))
            roots,issues=scan_coordinate(state,float(accel[bead,atom,axis]),dt,face,(int(bead),int(atom),axis),step,cushion)
            seen.extend(roots); unresolved.extend(issues)
    seen.sort(key=lambda x:(x['relative_time_fs'],x['bead'],x['atom'],x['axis'],x['kind']))
    counted={k:np.zeros(q.shape,dtype=np.int64) for k in ('periodic_chart','assignment_center')}
    for x in seen: counted[x['kind']][x['bead'],x['atom'],'xyz'.index(x['axis'])]+=x['sign']
    expected={'periodic_chart':images[1]-images[0], 'assignment_center':endpoint_labels(q1,images[1],cell,mesh)-endpoint_labels(q,images[0],cell,mesh)}
    checks={}
    for kind in counted:
        bad=np.argwhere(counted[kind]!=expected[kind]); checks[kind]=[{'bead':int(a),'atom':int(b),'axis':'xyz'[c],'recorded_net':int(expected[kind][a,b,c]),'replayed_net':int(counted[kind][a,b,c])} for a,b,c in bad]
    rootgroups=Counter((x['bead'],x['atom'],x['axis'],x['kind']) for x in seen)
    roundtrips=[{'bead':a,'atom':b,'axis':c,'kind':kind,'roots':count} for (a,b,c,kind),count in rootgroups.items() if count>abs(int(expected[kind][a,b,'xyz'.index(c)]))]
    return {'step':step,'absolute_interval_fs':[step*dt,(step+1)*dt], 'reconstructed_initial_q_error_A':initial_error,'endpoint_q_error_A':qerr,'endpoint_p_error_eV_fs_per_A':perr,'max_acceleration_bound_A_per_fs2':float(accel.max()),'coordinate_cushion_A':cushion,'screened_candidate_paths':candidates,'counts':dict(Counter(x['kind'] for x in seen)), 'endpoint_net_mismatches':checks,'roundtrip_coordinates':roundtrips,'unresolved':unresolved,'input_sha256':inputs,'roots':seen}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('capture',type=Path); ap.add_argument('--run-in',type=Path,required=True); ap.add_argument('--stdout',type=Path,required=True); ap.add_argument('--start-step',type=int,required=True); ap.add_argument('--end-step',type=int,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    info=inspect_inputs(a.capture,a.run_in,a.stdout,a.start_step,a.end_step); T=normal_matrix(info['P'])
    runs=[run_step(info,step,T) for step in range(a.start_step,a.end_step)]
    paths=[info['runin'],info['stdout'],info['capture']/'metadata.txt',info['capture']/'source_identity.txt',info['capture']/'scalar.txt',info['capture']/'COMPLETE',info['runin'].parent/'model.xyz']
    for x in info['lines']:
        if x[0] in ('potential','read_pimd_restart'): paths.append(info['runin'].parent/x[1])
    report={'schema':'r164_all_bead_geometric_replay_v1','scope':'bounded floating-point reconstruction of geometric assignment and periodic chart surfaces; no substep force evaluation, physical event, full MIC or neighbor cutoff coverage; not a rigorous mathematical certificate', 'source_identity':info['source'],'dt_fs':info['dt'],'P':info['P'],'N':info['N'],'capture':str(info['capture']),'stdout_binding':'same directory and matching printed facts, not cryptographically bound to run','minimum_coordinate_cushion_A':CUSHION,'endpoint_replay_tolerance':1e-8,'physical_event_history':'unavailable','accumulated_propagation_error':'unknown','input_sha256':{str(p):sha(p) for p in paths},'steps':runs}
    with a.output.open('x',encoding='utf8',newline='\n') as f: json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
if __name__=='__main__': main()
