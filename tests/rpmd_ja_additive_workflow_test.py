#!/usr/bin/env python3
"""CPU oracles and real CLI workflow checks for finite-temperature RPMD J_A references."""
from pathlib import Path
import math
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import rpmd_ja_fit_reference as fit
import rpmd_ja_qnep_prepare as prep


def _run(args,ok=True):
    result=subprocess.run([sys.executable,*map(str,args)],capture_output=True,text=True)
    if ok and result.returncode: raise AssertionError(result.stdout+result.stderr)
    if not ok and result.returncode==0: raise AssertionError("CLI unexpectedly accepted invalid input")
    return result


def _model(path,cell,positions,symbols):
    rows="\n".join(f"{s} {r[0]:.14g} {r[1]:.14g} {r[2]:.14g}" for s,r in zip(symbols,positions))
    path.write_text(f'{len(symbols)}\npbc="T T T" Lattice="{" ".join(map(str,cell.T.ravel()))}" Properties=species:S:1:pos:R:3\n{rows}\n')


def _write_raw(path,positions,cell,masses,k,ht,temperature=300.):
    n=len(masses); d=3*n; h=np.zeros(18); h[:9]=cell.ravel(); h[9:]=np.linalg.inv(cell).ravel()
    v=np.zeros((d,n)); c=np.asarray(ht); stats=np.zeros(18); stats[16]=.01; stats[17]=3.
    with path.open("wb") as f:
        f.write(fit.RAW_MAGIC); f.write(struct.pack("<IIii",3,fit.ENDIAN,n,d))
        f.write(struct.pack("<ddQQ",temperature,.01,7,19)); f.write(struct.pack("<iid",1,1,1.))
        f.write(fit.RAW_LAYOUT); f.write(h.astype("<f8").tobytes()); f.write(np.ones(3,dtype="<i4").tobytes())
        f.write(np.arange(n,dtype="<i4").tobytes()); f.write(np.asarray(masses,"<f8").tobytes())
        f.write(np.asarray(positions,"<f8").T.reshape(-1).tobytes()); f.write(struct.pack("<d",0.))
        f.write(np.zeros(n,dtype="<f8").tobytes()); f.write(np.zeros(d,dtype="<f8").tobytes())
        f.write(np.zeros(9*n,dtype="<f8").tobytes()); f.write(v.astype("<f8").tobytes())
        f.write(c.astype("<f8").tobytes()); f.write(np.asarray(k,"<f8").tobytes())
        f.write(v.astype("<f8").tobytes()); f.write(c.astype("<f8").tobytes()); f.write(stats.astype("<f8").tobytes())


def _physical_model(masses,lambdas):
    n=len(masses); d=3*n; e=np.zeros((d,3)); sqrtm=np.tile(np.sqrt(masses),3)
    for a in range(3):
        e[a*n,a]=np.sqrt(masses[1]/sum(masses)); e[a*n+1,a]=-np.sqrt(masses[0]/sum(masses))
    k=sqrtm[:,None]*e@np.diag(lambdas)@e.T*sqrtm[None,:]
    return k,e


def _tetrahedral_samples(base,masses,k,e,lambdas,cell,blocks=3):
    signs=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]],float)
    kbt=fit.KB_EV*300.; sqrtm=np.tile(np.sqrt(masses),3); positions=[]; forces=[]
    for _ in range(blocks):
        for s in signs:
            qmode=np.sqrt(kbt/np.asarray(lambdas))*s
            qcart=(e@qmode/sqrtm).reshape(3,len(masses)).T
            positions.append(base+qcart); forces.append(-(k@(qcart.T.reshape(-1))))
    return np.asarray(positions),np.asarray(forces).reshape(-1,3,len(masses)).transpose(0,2,1)


def _write_beads(directory,positions,forces,cell,symbols,beads=2):
    files=[]; rounded=np.round(cell,8)
    for b in range(beads):
        path=directory/f"beads_{b}.xyz"; files.append(path)
        with path.open("w") as f:
            for t,(pos,force) in enumerate(zip(positions,forces)):
                # A whole-cell image on bead one must not change the centroid.
                p=pos+(cell[:,0][None,:] if b else 0.)
                f.write(f'{len(symbols)}\nTime={t} pbc="T T T" Lattice="{" ".join(map(str,rounded.T.ravel()))}" Properties=species:S:1:pos:R:3:force:R:3\n')
                for s,r,g in zip(symbols,p,force): f.write(f"{s} {r[0]:.17g} {r[1]:.17g} {r[2]:.17g} {g[0]:.17g} {g[1]:.17g} {g[2]:.17g}\n")
    return files


def _read_block(stream,d):
    tile,count=struct.unpack("<iQ",stream.read(12)); assert tile==prep.TILE
    out=np.zeros((d,d))
    for _ in range(count):
        row,col,nr,nc,rank,nl,nright=struct.unpack("<5iQQ",stream.read(36))
        left=np.frombuffer(stream.read(8*nl),"<f8"); right=np.frombuffer(stream.read(8*nright),"<f8")
        block=left.reshape(nr,rank)@right.reshape(rank,nc) if rank else left.reshape(nr,nc)
        out[row:row+nr,col:col+nc]=block
    return out


def _read_v3(path):
    with path.open("rb") as f:
        assert f.read(8)==prep.MAGIC
        version,endian,n,temp,step,model_fp=struct.unpack("<IIiddQ",f.read(36)); assert version==3 and endian==fit.ENDIAN
        assert f.read(len(prep.UNITS))==prep.UNITS and f.read(len(prep.LAYOUT))==prep.LAYOUT
        cell=np.frombuffer(f.read(9*8),"<f8").copy().reshape(3,3); pbc=np.frombuffer(f.read(12),"<i4").copy()
        f.read(4*n+8*n+8*3*n)
        config,charge,pppm,spacing,plen=struct.unpack("<Qiidi",f.read(28)); policy=f.read(plen).decode()
        f.read(21*8); spectral,u=struct.unpack("<dd",f.read(16)); f.read(2*8); f.read(2*8)
        degree,pr,qr=struct.unpack("<iii",f.read(12))
        pbuf=f.read(pr*8)
        if len(pbuf)!=pr*8: raise AssertionError(f"bad v3 p values offset={f.tell()} degree={degree} p_rank={pr} q_rank={qr} bytes={len(pbuf)}")
        pval=np.frombuffer(pbuf,"<f8").copy(); qval=np.frombuffer(f.read(qr*8),"<f8").copy()
        pv=np.frombuffer(f.read((degree+1)*pr*8),"<f8").copy(); qv=np.frombuffer(f.read((degree+1)*qr*8),"<f8").copy()
        d=3*n; matrices=[_read_block(f,d) for _ in range(4)]
        assert struct.unpack("<i",f.read(4))[0]==1 and f.read()==b""
        return dict(n=n,temp=temp,step=step,cell=cell,pbc=pbc,policy=policy,spectral=spectral,u=u,degree=degree,
                    pval=pval,qval=qval,pv=pv,qv=qv,matrices=matrices,config=config,charge=charge,pppm=pppm,spacing=spacing)


def _local_eval(pack,positions,cell,x):
    n=len(positions); cart=x.reshape(3,n).T; r=positions+cart; energy=np.zeros(n); virial=np.zeros((3,3*n))
    for s in pack["sites"]:
        i=s["i"]; edges=s["neighbors"]; z=len(edges); y=np.zeros(3*z); geom=[]
        for e,(j,img) in enumerate(edges):
            d0=positions[j]+cell@np.asarray(img)-positions[i]; dr=r[j]+cell@np.asarray(img)-r[i]
            geom.append(dr); y[3*e:3*e+3]=dr-d0
        a=s["ell"].reshape(-1)+s["B"]@y; energy[i]=s["ell"].reshape(-1)@y+.5*y@s["B"]@y
        for e,(j,_) in enumerate(edges):
            for mu in range(3): virial[:,mu*n+j]-=np.asarray(geom[e])*a[3*e+mu]
    return energy,virial


def _fd_native_h(pack,positions,cell,alpha,step=2e-5):
    n=len(positions); d=3*n; v=np.zeros((n,d)); c=np.zeros((d,d)); zero=np.zeros(d)
    for q in range(d):
        dx=np.zeros(d); dx[q]=step
        ep,wp=_local_eval(pack,positions,cell,dx); em,wm=_local_eval(pack,positions,cell,-dx)
        v[:,q]=(ep-em)/(2*step); c[:,q]=(wp[alpha]-wm[alpha])/(2*step)
    l=np.zeros((n,d))
    for i in range(n): l[i,alpha*n+i]=1.
    ht=c-v.T@l
    fe=-v.sum(axis=0)
    for out in range(d):
        for inp in range(d):
            if inp//n==alpha and inp%n==out%n: ht[out,inp]-=fe[out]
    return ht,fe


def _fd_energy_k(pack,positions,cell,step=2e-4):
    d=3*len(positions); zero=np.zeros(d); e0=_local_eval(pack,positions,cell,zero)[0].sum(); k=np.zeros((d,d))
    for i in range(d):
        ei=np.zeros(d); ei[i]=step
        k[i,i]=(_local_eval(pack,positions,cell,ei)[0].sum()-2*e0+_local_eval(pack,positions,cell,-ei)[0].sum())/step**2
        for j in range(i):
            ej=np.zeros(d); ej[j]=step
            value=(_local_eval(pack,positions,cell,ei+ej)[0].sum()-_local_eval(pack,positions,cell,ei-ej)[0].sum()
                  -_local_eval(pack,positions,cell,-ei+ej)[0].sum()+_local_eval(pack,positions,cell,-ei-ej)[0].sum())/(4*step**2)
            k[i,j]=k[j,i]=value
    return k


def test_independent_local_energy_virial_finite_difference_oracle():
    n=3; cell=np.array([[5.,.8,.2],[0.,4.7,.5],[0.,0.,5.2]]); r=np.array([[.3,.4,.2],[1.2,.5,.6],[.4,1.4,.9]])
    sites=[]; rng=np.random.default_rng(112)
    neighbor_data=([(1,(1,0,0)),(2,(0,0,0))],[(0,(-1,0,0)),(2,(0,0,0))],[(0,(0,0,0)),(1,(0,0,0))])
    for i,neighbors in enumerate(neighbor_data):
        b=rng.normal(size=(6,6)); b=(b+b.T)*.5
        sites.append(dict(i=i,neighbors=neighbors,ell=rng.normal(scale=.03,size=(2,3)),B=b))
    bforce=np.array([[.1,-.2,.05],[-.04,.08,-.02],[-.06,.12,-.03]])
    fit._set_linear_balance(sites,bforce)
    pack=dict(n=n,beads=4,source_fingerprint=1,temperature=300.,epsilon=.01,response_max=0.,response_tolerance=.1,
              training_frames=4,validation_frames=2,sites=sites)
    got=fit.assemble_additive(pack,r,cell); fd=[]
    np.testing.assert_allclose(_fd_energy_k(pack,r,cell),got["Kadd"],rtol=2e-7,atol=2e-8)
    grad=np.zeros(9); h=1e-5
    for q in range(9):
        dx=np.zeros(9); dx[q]=h
        grad[q]=(_local_eval(pack,r,cell,dx)[0].sum()-_local_eval(pack,r,cell,-dx)[0].sum())/(2*h)
    np.testing.assert_allclose(grad.reshape(3,n).T,-bforce,atol=2e-9)
    for a in range(3):
        ht,fe=_fd_native_h(pack,r,cell,a); fd.append(ht)
        np.testing.assert_allclose(ht,got["Hadd"][a].T,rtol=2e-7,atol=2e-8)
        np.testing.assert_allclose(got["linear_gradient"],-bforce,atol=1e-12)
    # With B removed, independent energy/virial differences cancel every linear ell term in H.
    linear=[dict(s, B=np.zeros_like(s["B"])) for s in sites]; lp=dict(pack,sites=linear)
    for a in range(3):
        ht,_=_fd_native_h(lp,r,cell,a)
        np.testing.assert_allclose(ht,0.,atol=3e-8)


def test_cli_collect_fit_prepare_zero_and_nonzero_addition():
    masses=np.array([1.,3.]); cell=np.array([[20.123456789,1.2,.3],[0.,19.234567891,.6],[0.,0.,21.345678912]])
    base=np.array([[5.,5.,5.],[7.,5.,5.]]); symbols=["H","O"]; lambdas=np.array([2.,3.,4.]); ktrue,e=_physical_model(masses,lambdas)
    x,f=_tetrahedral_samples(base,masses,ktrue,e,lambdas,cell)
    with tempfile.TemporaryDirectory() as tmp:
        td=Path(tmp); model=td/"model.xyz"; _model(model,cell,base,symbols); beads=_write_beads(td,x,f,cell,symbols)
        samples=td/"samples.npz"; mean=td/"mean.xyz"
        _run([ROOT/"tools/rpmd_ja_fit_reference.py","collect","--bead-files",*beads,"--temperature","300","--output",samples,"--mean-model",mean,"--masses",*masses,"--cell-model",model])
        with np.load(samples) as z: np.testing.assert_array_equal(z["cell"],cell)
        step_beads=[]
        for i,path in enumerate(beads):
            step_path=td/f"step_beads_{i}.xyz"
            step_path.write_text(path.read_text(encoding="utf-8").replace("Time=","Step="),encoding="utf-8")
            step_beads.append(step_path)
        step_samples=td/"step_samples.npz"; step_mean=td/"step_mean.xyz"
        _run([ROOT/"tools/rpmd_ja_fit_reference.py","collect","--bead-files",*step_beads,"--temperature","300","--output",step_samples,"--mean-model",step_mean,"--masses",*masses,"--cell-model",model])
        with np.load(step_samples) as z:
            assert z["steps"].dtype.kind=="f"
            with np.load(samples) as ztime: np.testing.assert_array_equal(z["positions"],ztime["positions"])
        tables=ROOT/"tests/data/rpmd_ja_kernel_U8.txt"
        outputs=[]; successful_fit=None; zero_additive=None; zero_raw=None; zero_ref=None; native_baseline=None
        for mode,scale in (("zero",1.),("recoverable",.5)):
            raw=td/f"{mode}.qraw"; ht=np.zeros((3,6,6))
            for a in range(3):
                ht[a]=np.arange(36,dtype=float).reshape(6,6)*(a+1)/31.
                ht[a,0,1]+=0.37
            _write_raw(raw,base,cell,masses,scale*ktrue,ht)
            pack=td/f"{mode}.add"; ref=td/f"{mode}.ja"
            fit_result=_run([ROOT/"tools/rpmd_ja_fit_reference.py","fit","--raw",raw,"--samples",samples,"--output",pack,
                             "--cutoff","3.0","--epsilon","0.01","--response-tolerance","2e-7"])
            if mode=="zero":
                step_pack=td/"step_only.add"
                _run([ROOT/"tools/rpmd_ja_fit_reference.py","fit","--raw",raw,"--samples",step_samples,"--output",step_pack,
                      "--cutoff","3.0","--epsilon","0.01","--response-tolerance","2e-7"])
                assert fit.read_additive(step_pack)["source_fingerprint"]==fit.read_additive(pack)["source_fingerprint"]
            if mode=="zero": successful_fit=fit_result; zero_additive=fit.read_additive(pack); zero_raw=raw; zero_ref=ref
            cli=_run([ROOT/"tools/rpmd_ja_qnep_prepare.py",raw,pack,"--kernel-table",tables,"--output",ref,"--lossless"])
            parsed=_read_v3(ref); assert "finite_temperature_additive_v1;beads=2;derivative=3" in parsed["policy"]
            additive=fit.read_additive(pack); assembled=fit.assemble_additive(additive,base,cell)
            mass=np.tile(masses,3); qtrans=np.zeros((6,3))
            for a in range(3): qtrans[a*2:(a+1)*2,a]=np.sqrt(masses)/np.sqrt(masses.sum())
            projector=np.eye(6)-qtrans@qtrans.T
            expected_d=projector@((scale*ktrue+assembled["Kadd"])/(np.sqrt(mass[:,None]*mass[None,:])))@projector
            np.testing.assert_allclose(parsed["matrices"][0],expected_d,rtol=2e-9,atol=2e-10)
            expected_h=np.array([(ht[a].T+assembled["Hadd"][a].T)/np.sqrt(mass[:,None]*mass[None,:]) for a in range(3)])
            for a in range(3): np.testing.assert_allclose(parsed["matrices"][a+1],expected_h[a],rtol=2e-10,atol=2e-11)
            if mode=="zero":
                assert np.linalg.norm(assembled["Kadd"])<1e-10 and np.linalg.norm(assembled["Hadd"])<1e-10
                native_ref=td/"zero_native.ja"
                _run([ROOT/"tools/rpmd_ja_qnep_prepare.py",raw,"--kernel-table",tables,"--output",native_ref,"--lossless"])
                native_baseline=_read_v3(native_ref)
                np.testing.assert_allclose(parsed["matrices"][0],native_baseline["matrices"][0],rtol=2e-12,atol=2e-12)
                for a in range(3): np.testing.assert_allclose(parsed["matrices"][a+1],native_baseline["matrices"][a+1],rtol=2e-12,atol=2e-12)
            else:
                np.testing.assert_allclose(assembled["Kadd"],.5*ktrue,rtol=2e-5,atol=2e-7)
            outputs.append((parsed,expected_h,assembled,ht,native_baseline))
        zero=outputs[0][0]; ref_d=zero["matrices"][0]; w,evec=np.linalg.eigh(ref_d)
        # Three mass translations are exactly zero; use only the positive internal eigenspace.
        keep=w>1e-9; omega=np.sqrt(w[keep]); evec=evec[:,keep]; assert len(omega)==3
        hmodal=np.array([evec.T@zero["matrices"][a+1]@evec for a in range(3)])
        table=prep._kernel(tables)
        for key,value in (("degree","degree"),("u","u"),("pv","pval"),("qv","qval"),("pvec","pv"),("qvec","qv")):
            np.testing.assert_array_equal(zero[value],table[key])
        # Replace input text coefficients with the coefficients read back from the prepared v3 file.
        table.update(degree=zero["degree"],u=zero["u"],pv=zero["pval"],qv=zero["qval"],pvec=zero["pv"],qvec=zero["qv"])
        tau=prep.HBAR/(prep.KB*300.); lam_scale=table["u"]**2/tau**2
        with np.load(samples) as z:
            xval=z["positions"][16:]
        sqrtm=np.tile(np.sqrt(masses),3); xcart=(xval-base).transpose(0,2,1).reshape(len(xval),-1)
        xmodal=(xcart*sqrtm)@evec
        vmodal=np.random.default_rng(381).normal(scale=.2,size=xmodal.shape)
        caches=[]; deltacurrent=np.zeros((len(xmodal),3)); zero_native_deltacurrent=np.zeros_like(deltacurrent)
        for a in range(3):
            cache=_cached_delta(hmodal[a],omega,tau,lam_scale,table); direct=_direct_fock_delta(hmodal[a].T,omega,tau)
            for t in range(len(xmodal)):
                actual=vmodal[t]@cache@xmodal[t]; expected=xmodal[t]@direct@vmodal[t]
                budget=_cache_error_bound(hmodal[a],omega,tau,table,xmodal[t],vmodal[t])
                assert abs(actual-expected)<=budget+2e-10*max(1.,abs(expected)), (actual,expected,budget)
                deltacurrent[t,a]=actual
            caches.append(cache)
        native=outputs[0][4]
        hnative=np.array([evec.T@native["matrices"][a+1]@evec for a in range(3)])
        for a in range(3):
            cache_native=_cached_delta(hnative[a],omega,tau,lam_scale,table)
            for t in range(len(xmodal)): zero_native_deltacurrent[t,a]=vmodal[t]@cache_native@xmodal[t]
        np.testing.assert_allclose(deltacurrent,zero_native_deltacurrent,rtol=2e-11,atol=2e-11)
        jc=np.random.default_rng(54).normal(size=(len(xmodal),3)); ja=jc+deltacurrent
        def corr(left,right):
            return np.array([sum(float(left[t]@right[t+lag]) for t in range(len(left)-lag))/(len(left)-lag) for lag in range(len(left))])
        for lag in range(len(jc)):
            cc=corr(jc,jc)[lag]; cd=corr(jc,deltacurrent)[lag]; dc=corr(deltacurrent,jc)[lag]; dd=corr(deltacurrent,deltacurrent)[lag]
            np.testing.assert_allclose(corr(ja,ja)[lag],cc+cd+dc+dd,rtol=2e-14,atol=2e-14)
        # A legal null-K redistribution has an explicit same-K assertion and a measured mapped-current change.
        zero_pack=zero_additive; gauge_sites=[dict(s,ell=s["ell"].copy(),B=s["B"].copy()) for s in zero_pack["sites"]]
        qg=np.eye(3)*.2; gauge_sites[0]["B"]+=qg; gauge_sites[1]["B"]-=qg
        gauge_pack=dict(zero_pack,sites=gauge_sites)
        ka=fit.assemble_additive(zero_pack,base,cell); kg=fit.assemble_additive(gauge_pack,base,cell)
        np.testing.assert_allclose(kg["Kadd"],ka["Kadd"],rtol=0.,atol=2e-14)
        mass_scale=np.sqrt(np.tile(masses,3)[:,None]*np.tile(masses,3)[None,:]); hphys_t=outputs[0][3]
        delta_gauge=np.zeros_like(deltacurrent)
        for a in range(3):
            ht_g=(hphys_t[a].T+kg["Hadd"][a].T)/mass_scale
            at_g=evec.T@ht_g@evec; cache_g=_cached_delta(at_g,omega,tau,lam_scale,table)
            for t in range(len(xmodal)): delta_gauge[t,a]=vmodal[t]@cache_g@xmodal[t]
        assert np.linalg.norm(delta_gauge-deltacurrent)>1e-7
        # A response-budget refusal and input-integrity refusals must leave no package/reference behind.
        def save_npz(path,changes):
            with np.load(samples) as z: arrays={k:z[k].copy() for k in z.files}
            arrays.update(changes); np.savez(path,**arrays)
        rounded_samples=td/"rounded_temperature.npz"; save_npz(rounded_samples,{"temperature":np.array(300.*(1.+5e-11))})
        rounded_pack=td/"rounded_temperature.add"
        _run([ROOT/"tools/rpmd_ja_fit_reference.py","fit","--raw",zero_raw,"--samples",rounded_samples,"--output",rounded_pack,
              "--cutoff","3.0","--epsilon","0.01","--response-tolerance","2e-7"])
        assert fit.read_additive(rounded_pack)["temperature"]==300.
        different_samples=td/"different_temperature.npz"; save_npz(different_samples,{"temperature":np.array(300.*(1.+2e-10))})
        different_pack=td/"different_temperature.add"
        _run([ROOT/"tools/rpmd_ja_fit_reference.py","fit","--raw",zero_raw,"--samples",different_samples,"--output",different_pack,
              "--cutoff","3.0","--epsilon","0.01","--response-tolerance","2e-7"],ok=False)
        assert not different_pack.exists()
        with np.load(samples) as z: arrays={k:z[k].copy() for k in z.files}
        train=2*len(arrays["positions"])//3; r0=fit.read_raw(zero_raw)["positions"]
        arrays["positions"][train:]=r0+1.2*(arrays["positions"][train:]-r0)
        arrays["forces"][train:]/=1.2
        response_samples=td/"response_failure.npz"; np.savez(response_samples,**arrays)
        reject_fit=td/"response_rejected.add"
        refused=_run([ROOT/"tools/rpmd_ja_fit_reference.py","fit","--raw",zero_raw,"--samples",response_samples,"--output",reject_fit,
                      "--cutoff","3.0","--epsilon","0.01","--response-tolerance","0.1"],ok=False)
        assert "independent response error" in refused.stderr.lower() and not reject_fit.exists()
        short=td/"bad.qraw"; short.write_bytes(b"GPJQRAW")
        badfit=td/"bad.add"
        _run([ROOT/"tools/rpmd_ja_fit_reference.py","fit","--raw",short,"--samples",samples,"--output",badfit,
              "--cutoff","3.0","--epsilon","0.01","--response-tolerance","2e-7"],ok=False)
        assert not badfit.exists()
        corrupt=td/"corrupt.add"; corrupt.write_bytes((td/"zero.add").read_bytes()+b"x")
        out=td/"corrupt.ja"
        _run([ROOT/"tools/rpmd_ja_qnep_prepare.py",zero_raw,corrupt,"--kernel-table",tables,"--output",out],ok=False)
        assert not out.exists() and not Path(str(out)+".stability").exists()
        wrong=td/"wrong_source.qraw"; rawbytes=bytearray(zero_raw.read_bytes()); rawbytes[40]^=1; wrong.write_bytes(rawbytes)
        out=td/"wrong_source.ja"
        _run([ROOT/"tools/rpmd_ja_qnep_prepare.py",wrong,td/"zero.add","--kernel-table",tables,"--output",out],ok=False)
        assert not out.exists() and not Path(str(out)+".stability").exists()


def _cheb_values(z,degree):
    vals=np.empty(degree+1); vals[0]=1.
    if degree: vals[1]=z
    for k in range(2,degree+1): vals[k]=2*z*vals[k-1]-vals[k-2]
    return vals


def _cached_delta(at,omega,tau,lambda_scale,table):
    modes=[]
    for arrays,rank in ((table["pvec"],len(table["pv"])),(table["qvec"],len(table["qv"]))):
        poly=np.empty((rank,len(omega)))
        for a,la in enumerate(omega**2):
            basis=_cheb_values(2*la/lambda_scale-1,table["degree"])
            poly[:,a]=arrays.reshape(table["degree"]+1,rank).T@basis
        modes.append(poly)
    pvec,qvec=modes; p=np.einsum("r,ra,rb->ab",table["pv"],pvec,pvec); q=np.einsum("r,ra,rb->ab",table["qv"],qvec,qvec)
    la=omega**2; ta=tau**2*la
    return ta[:,None]*ta[None,:]*p*at + ta[None,:]*q*at.T


def _g(x):
    return math.sinh(x)/x if abs(x)>1e-7 else 1+x*x/6+x**4/120


def _direct_fock_delta(at,omega,tau):
    out=np.empty_like(at); a=tau*omega
    for i in range(len(a)):
        for j in range(len(a)):
            sp=math.sqrt(_g((a[i]+a[j])/2)/(_g(a[i]/2)*_g(a[j]/2)))
            sm=math.sqrt(_g((a[i]-a[j])/2)/(_g(a[i]/2)*_g(a[j]/2)))
            out[i,j]=(.5*(sp+sm)-1)*at[i,j]+.5*(sp-sm)*(omega[i]/omega[j])*at[j,i]
    return out


def _cache_error_bound(at,omega,tau,table,x,y):
    la=omega**2; t=tau**2*la; pe,qe=table["error"]
    return float(np.sum(np.abs(y[:,None]*x[None,:])*(t[:,None]*t[None,:]*pe*np.abs(at)+t[None,:]*qe*np.abs(at.T))))


if __name__=="__main__":
    for name,value in list(globals().items()):
        if name.startswith("test_"): value()
    print("RPMD-JA additive workflow tests passed")
