"""Read saved finite-model correlations; descriptive bootstrap is not acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
import u9_rpmd


def summarize(group, target, times, delta):
    c=group[:,:,:2,:2].sum(axis=(-1,-2))
    if delta is None:
        cd=np.zeros_like(c); dc=np.zeros_like(c); dd=np.zeros_like(c)
        s11=group[:,:,0,0]; s13=group[:,:,0,1]; s31=group[:,:,1,0]
    else:
        cd=group[:,:,:2,delta].sum(axis=-1); dc=group[:,:,delta,:2].sum(axis=-1); dd=group[:,:,delta,delta]
        s11=group[:,:,0,0]+group[:,:,0,delta]+group[:,:,delta,0]+dd
        s13=group[:,:,0,1]+group[:,:,delta,1]; s31=group[:,:,1,0]+group[:,:,1,delta]
    s33=group[:,:,1,1]; total=c+cd+dc+dd
    closure=np.max(np.abs(total-(s11+s13+s31+s33)))
    dt=times[1]-times[0]
    return {'sectors':np.stack([s11,s13,s31,s33],axis=-1),'total':total,
            'correction_blocks':np.stack([c,cd,dc,dd],axis=-1),'closure_max':float(closure),
            'finite_integrals':np.trapezoid(total,times,axis=1),'mu0':total[:,0],
            'mu2_forward':-(2*total[:,0]-5*total[:,1]+4*total[:,2]-total[:,3])/(dt*dt) if len(times)>=4 else None}


def paired_bootstrap(candidate,centroid,target,counts):
    def rms(curves): return np.sqrt(np.mean((curves-target)**2,axis=-1))
    difference=float(rms(candidate.mean(axis=0))-rms(centroid.mean(axis=0)))
    boot=rms(candidate[counts].mean(axis=1))-rms(centroid[counts].mean(axis=1))
    return difference,np.percentile(boot,[2.5,97.5]),float(boot.std(ddof=1))


def main():
    p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('case'); p.add_argument('--quantum',required=True); p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int); p.add_argument('--seed',type=int)
    args=p.parse_args(); root=Path(args.directory); output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    case=report['cases'][args.case]; contract=case['contract']; frozen=report['frozen_cases']['rpmd']
    count=frozen['bootstrap'] if args.bootstrap is None else args.bootstrap
    seed=frozen['seed_bootstrap'] if args.seed is None else args.seed
    if count<2 or seed<0: raise ValueError('bootstrap count must be >=2 and seed nonnegative')
    if args.quantum not in case['spectra'] or not report['source_sha256']: raise ValueError('missing basis or source provenance')
    with np.load(root/(args.case+'_'+args.quantum+'_quantum.npz'),allow_pickle=False) as q:
        times=q['times']; quantum=q['blocks']; qcontract=str(q['contract'])
    if qcontract!=contract or times.ndim!=1 or times.size<2 or quantum.shape!=(times.size,2,2) or not np.isfinite(times).all() or not np.isfinite(quantum).all() or times[0]!=0 or np.any(np.diff(times)<=0): raise ValueError('invalid quantum data or contract')
    dt=np.diff(times)
    if not np.allclose(dt,dt[0],rtol=1e-12,atol=1e-12): raise ValueError('time grid must be uniform')
    with np.load(root/(args.case+'_flows.npz'),allow_pickle=False) as raw:
        flows=raw['flows']; names=raw['channels'].tolist()
        if str(raw['contract'])!=contract or not np.array_equal(times,raw['times']): raise ValueError('raw flow contract or clock mismatch')
    expected=['J1','J3','delta_legacy_left','delta_moving_left']
    if 'right' in case['references']: expected+=['delta_legacy_right','delta_moving_right']
    if names!=expected or flows.ndim!=4 or flows.shape[0]!=report['settings']['groups'] or flows.shape[1]!=report['settings']['origins'] or flows.shape[2:]!=(len(times),len(names)) or not np.isfinite(flows).all(): raise ValueError('invalid raw flows or channel labels')
    result={'status':'INCONCLUSIVE: convergence and mixing gates not established','case':args.case,'contract':contract,'quantum_basis':args.quantum,'bootstrap_count':count,'bootstrap_seed':seed,'candidates':{}}
    arrays={}; centroid=None; target=quantum.sum(axis=(1,2)); groups=flows.shape[0]
    counts=np.random.default_rng(seed).integers(groups,size=(count,groups))
    for side in case['references']:
        labels=['J1','J3','delta_legacy_'+side,'delta_moving_'+side]
        with np.load(root/(args.case+'_'+side+'_rpmd.npz'),allow_pickle=False) as data:
            if str(data['contract'])!=contract or not np.array_equal(times,data['times']) or data['channels'].tolist()!=labels: raise ValueError('RPMD contract, labels or times mismatch')
            group=data['group_correlations']
        selection=[names.index(label) for label in labels]
        reconstructed=np.mean([u9_rpmd.correlations(flows[:,origin,:,:][:,:,selection]) for origin in range(flows.shape[1])],axis=0)
        if group.shape!=(groups,len(times),4,4) or not np.isfinite(group).all() or not np.allclose(group,reconstructed,rtol=1e-12,atol=1e-12): raise ValueError('RPMD group correlation does not match raw flows')
        for name,delta in [('centroid',None),('legacy_'+side,2),('moving_'+side,3)]:
            if name=='centroid' and centroid is not None: continue
            summary=summarize(group,quantum,times,delta)
            if name=='centroid': centroid=summary['total']
            difference,interval,se=paired_bootstrap(summary['total'],centroid,target,counts)
            arrays[name+'_sectors']=summary['sectors']; arrays[name+'_correction_blocks']=summary['correction_blocks']
            result['candidates'][name]={'closure_max':summary['closure_max'],'paired_rms_difference':difference,'paired_interval_95':interval.tolist(),'bootstrap_se':se,'finite_integral_mean':float(summary['finite_integrals'].mean()),'mu0_mean':float(summary['mu0'].mean()),'mu2_forward_mean':None if summary['mu2_forward'] is None else float(summary['mu2_forward'].mean()),'ranking':'UNKNOWN: basis, box, P, dt and mixing gates not established'}
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'decomposition.npz',times=times,quantum_blocks=quantum,**arrays)
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__': main()
