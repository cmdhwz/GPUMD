"""Run frozen OHO and optionally a general nine-coefficient fixed-A finite model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import bridge, u9_rpmd as rpmd, spectrum


def choose(value, fallback):
    return fallback if value is None else value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--theory-root',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--u9-config',help='additional general U9 JSON input')
    parser.add_argument('--quick',action='store_true'); parser.add_argument('--seed',type=int,default=20261027)
    for name in ('beads','groups','burn','origins','stride','hmc-steps'): parser.add_argument('--'+name,type=int)
    for name in ('dt','hmc-dt','time-end','time-step'): parser.add_argument('--'+name,type=float)
    parser.add_argument('--basis',type=int,nargs=2,action='append',metavar=('NQ','NQ2'))
    parser.add_argument('--dvr',type=int,nargs=2,action='append',metavar=('NQ','NQ2'))
    parser.add_argument('--box',type=float,nargs=2,default=(8.,8.)); parser.add_argument('--frequencies',type=float,nargs=2)
    parser.add_argument('--rotation',type=float,default=0.,help='orthogonal coordinate rotation in radians')
    args=parser.parse_args(); root=Path(args.theory_root).resolve()
    case_path=root/'quantum_benchmark_OHO/frozen_cases.json'
    cases=json.loads(case_path.read_text(encoding='utf-8')); frozen=cases['rpmd']; grid=cases['time_grid']
    P=choose(args.beads,8 if args.quick else frozen['P']); dt=choose(args.dt,.02 if args.quick else frozen['dt'])
    groups=choose(args.groups,8 if args.quick else frozen['groups']); burn=choose(args.burn,40 if args.quick else frozen['burnin'])
    origins=choose(args.origins,2 if args.quick else frozen['origins_per_group']); stride=choose(args.stride,frozen['stride'])
    hmc_dt=choose(args.hmc_dt,frozen['hmc_dt']); hmc_steps=choose(args.hmc_steps,frozen['hmc_steps'])
    end=choose(args.time_end,.3 if args.quick else grid['end']); spacing=choose(args.time_step,.02 if args.quick else grid['step'])
    frequencies=choose(args.frequencies,cases['basis_frequencies'])
    floats=np.array([dt,hmc_dt,end,spacing,*args.box,*frequencies,args.rotation],float)
    if not np.isfinite(floats).all() or min(dt,hmc_dt,end,spacing,*args.box,*frequencies)<=0 or args.seed<0:
        parser.error('finite positive times, basis scales and nonnegative seed required')
    if min(P,groups,burn,origins,stride,hmc_steps)<1 or P<2 or groups<2: parser.error('invalid integer dimensions')
    if not np.isfinite(spacing/dt) or not np.isfinite(end/spacing): parser.error('time-grid ratios overflow')
    substeps=round(spacing/dt); frames=round(end/spacing)+1
    if substeps<1 or frames<2 or abs(substeps*dt-spacing)>1e-10 or abs((frames-1)*spacing-end)>1e-10:
        parser.error('exact real-time and output-grid alignment required')
    basis=choose(args.basis,[(6,6),(8,8)] if args.quick else cases['basis_sequence'])
    dvr=choose(args.dvr,[(9,9),(13,13)] if args.quick else [(25,25),(33,33)])
    if any(len(d)!=2 or min(d)<2 for d in basis) or any(len(d)!=2 or min(d)<3 or any(n%2==0 for n in d) for d in dvr): parser.error('basis sizes >=2 and odd DVR sizes >=3 required')
    modules=spectrum.load_modules(root); oho=modules['oho']; ce=modules['embedding']
    jobs=[]
    for entry in cases['cases']:
        case=cases['common'] | entry
        c=({'c':[0.,0.,case['kq']/2,0.,0.],'K':case['K_lat'],'lambda':[0.,0.,0.]} if case.get('harmonic') else
           {'c':[case['a']*case['b']**4,case['delta'],-2*case['a']*case['b']**2,0.,case['a']],'K':case['K_lat'],'lambda':[-case['lambda']*case['b']**2,0.,case['lambda']]})
        ref={'ML':case['M'],'m':case['m'],'MR':case['M']}
        for variant,masses in [('historical',[case['M'],case['m'],case['M']]),('unequal_H',[case['M'],1.,case['M']+1.]),('unequal_D',[case['M'],2.,case['M']+1.])]:
            jobs.append((case['name']+'_'+variant,c,ref,masses,case['d0'],case['beta'],case['hbar'],cases['units'],'frozen_cases'))
    general=None
    if args.u9_config:
        general=json.loads(Path(args.u9_config).read_text(encoding='utf-8'))
        required={'c','K','lambda','reference_masses','isotope_masses','d0','beta','hbar','units','source'}
        if not isinstance(general,dict) or set(general)!=required:
            parser.error('general U9 JSON fields invalid')
        if any(not isinstance(general[k],list) for k in ('c','lambda','isotope_masses')) or not isinstance(general['reference_masses'],dict):
            parser.error('general U9 coefficients and masses must be arrays and an object')
        if len(general['c'])!=5 or len(general['lambda'])!=3 or len(general['isotope_masses'])!=3 or set(general['reference_masses'])!={'ML','m','MR'} or not isinstance(general['units'],str) or not general['units'] or not isinstance(general['source'],str) or not general['source']:
            parser.error('general U9 JSON fields or lengths invalid')
        raw_numbers=[*general['c'],general['K'],*general['lambda'],*general['reference_masses'].values(),*general['isotope_masses'],general['d0'],general['beta'],general['hbar']]
        if any(isinstance(value,bool) or not isinstance(value,(int,float)) for value in raw_numbers):
            parser.error('general U9 numerical fields must be numbers, not booleans or objects')
        try:
            numbers=np.asarray(raw_numbers,float)
        except (OverflowError,ValueError):
            parser.error('general U9 numerical fields are out of range')
        if not np.isfinite(numbers).all() or min(general['K'],general['d0'],general['beta'],general['hbar'],*general['reference_masses'].values(),*general['isotope_masses'])<=0:
            parser.error('general U9 requires finite parameters and positive masses, K, d0, beta and hbar')
        jobs.append(('general_u9',{'c':general['c'],'K':general['K'],'lambda':general['lambda']},general['reference_masses'],general['isotope_masses'],general['d0'],general['beta'],general['hbar'],general['units'],general['source']))
    import local_model
    prepared=[]
    for key,c,refmass,masses,d0,beta,hbar,units,source in jobs:
        points=local_model.stationary_points(c); stability=points['stability']['state']
        effective=np.asarray(points['stability']['effective_coefficients'],float)
        if not np.isfinite(effective).all(): raise ValueError('nonfinite relaxed potential for '+key)
        if stability!='strict_sufficient_confinement' and not (effective[4]==0 and effective[3]==0 and effective[2]>0):
            raise ValueError('confinement not established for '+key)
        centers=[np.array([r['q'],r['Q']],float) for r in points['roots'] if r['classification']=='minimum']
        if not centers or not all(np.isfinite(center).all() for center in centers): raise ValueError('no finite stable minimum for '+key)
        model=ce.build_polynomials(c,refmass,masses,d0,hbar)
        refs={'left':bridge.reference(model,centers[0],beta,oho)}
        if len(centers)>1: refs['right']=bridge.reference(model,centers[-1],beta,oho)
        prepared.append((model,centers,refs))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    hashes=dict(modules['hashes']); files=[case_path,*Path(__file__).resolve().parent.glob('*.py')]
    if args.u9_config: files.append(Path(args.u9_config).resolve())
    for path in files: hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    angle=args.rotation; rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    settings={'P':P,'dt':dt,'hmc_dt':hmc_dt,'hmc_steps':hmc_steps,'groups':groups,'burn':burn,'origins':origins,'stride':stride,'times':[0.,end,spacing],'basis':basis,'dvr':dvr,'box':args.box,'frequencies':frequencies,'rotation':angle,'seed':args.seed}
    report={'settings':settings,'source_sha256':hashes,'cases':{},'status':'QUICK_SMOKE' if args.quick else 'INCONCLUSIVE_PENDING_CONVERGENCE','frozen_cases':cases,'general_input':general}
    for index,(key,c,refmass,masses,d0,beta,hbar,units,source) in enumerate(jobs):
        model,centers,refs=prepared[index]; times=np.arange(frames)*spacing; spectra={}
        contract=hashlib.sha256(json.dumps({'settings':settings,'hashes':hashes,'case':key,'coefficients':c,'reference_masses':refmass,'isotope_masses':masses,'d0':d0,'beta':beta,'hbar':hbar,'units':units,'source':source},sort_keys=True).encode('utf-8')).hexdigest()
        for method,dimensions in [('ho',basis),('dvr',dvr)]:
            for dims in dimensions:
                result=spectrum.solve(model,beta,times,tuple(dims),method,modules,args.box,frequencies,rotation)
                label=method+'x'.join(map(str,dims)); spectra[label]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items() if k not in ('blocks','commutator_curve')}
                np.savez(out/(key+'_'+label+'_quantum.npz'),times=times,blocks=result['blocks'],commutator_curve=result['commutator_curve'],mu2=result['mu2'],contract=contract)
        g,V=np.linalg.eigh(model['geometry']['G']); W=(V/np.sqrt(g))@V.T
        x=np.empty((groups,P,2)); invW=np.linalg.inv(W)
        for group in range(groups): x[group]=centers[group%len(centers)]@invW.T
        x,acceptance=rpmd.sample(x,W,c,beta,hbar,args.seed+index,burn,hmc_dt,hmc_steps)
        names=['J1','J3']; polys=[model['J1'],model['J3']]
        for side in refs:
            names.extend(['delta_legacy_'+side,'delta_moving_'+side])
            polys.extend([oho.add(refs[side]['legacy_JA'],oho.scale(model['J'],-1)),oho.add(refs[side]['moving_JA'],oho.scale(model['J'],-1))])
        flows_by_origin=[]; occupancy=[]
        for origin in range(origins):
            x,_=rpmd.sample(x,W,c,beta,hbar,args.seed+100000+index*origins+origin,stride,hmc_dt,hmc_steps)
            occupancy.append(((x@W.T).mean(axis=1)[:,0]>0).tolist())
            flows_by_origin.append(rpmd.trajectory(x.copy(),W,c,beta,hbar,args.seed+200000+index*origins+origin,dt,frames,polys,oho,substeps))
        flows=np.stack(flows_by_origin,axis=1)
        np.savez(out/(key+'_flows.npz'),flows=flows,times=times,channels=np.asarray(names),contract=contract)
        for side in refs:
            selection=[0,1,names.index('delta_legacy_'+side),names.index('delta_moving_'+side)]
            grouped=np.mean([rpmd.correlations(flows[:,origin,:,:][:,:,selection]) for origin in range(origins)],axis=0)
            np.savez(out/(key+'_'+side+'_rpmd.npz'),group_correlations=grouped,times=times,channels=np.asarray([names[i] for i in selection]),contract=contract)
        frozen_refs={side:{name:ref[name].tolist() for name in ('K','legacy_H','legacy_affine','moving_H','moving_delta_H')} for side,ref in refs.items()}
        report['cases'][key]={'contract':contract,'G':model['geometry']['G'].tolist(),'coefficients':c,'reference_masses':refmass,'isotope_masses':masses,'units':units,'source':source,'spectra':spectra,'hmc_burn_acceptance':acceptance,'positive_q_occupancy_by_origin':occupancy,'reference_centers':{'left':centers[0].tolist(),**({'right':centers[-1].tolist()} if 'right' in refs else {})},'references':frozen_refs,'mixing_status':'NOT_ESTABLISHED'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__': main()
