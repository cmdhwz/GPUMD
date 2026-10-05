#!/usr/bin/env python3
"""Offline finite-temperature additive reference fitter for RPMD J_A."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import zipfile
import numpy as np

RAW_MAGIC = b"GPJQRAW\0"
RAW_LAYOUT = b"xyz_soa;derivative_input_rows_output_columns\0"
ADD_MAGIC = b"GPJAADD1"


def _stable_norm(values):
    return float(np.hypot.reduce(np.asarray(values, dtype=float).ravel()))
ENDIAN = 0x01020304
KB_EV = 8.617343e-5
# ponytail: Dense gauge/QP certification is deliberately limited to small systems;
# use a sparse solver in a separately validated implementation before raising this.
MAX_DENSE_ELEMENTS = 6_000_000
MAX_PARAMETERS = 1600
MAX_INTERNAL_DIM = 384
MAX_SAMPLE_BYTES = 256 * 1024 * 1024
MEAN_TOLERANCE = 2e-5


def _fnv64(path):
    h = 14695981039346656037
    with Path(path).open("rb") as f:
        while data := f.read(1 << 20):
            for b in data:
                h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def _exact(f, size):
    b = f.read(size)
    if len(b) != size:
        raise ValueError("truncated qNEP raw file")
    return b


def read_raw(path):
    """Read and validate the original v1/v2/v3 qraw layout; matrices stay memmapped."""
    path = Path(path).resolve()
    with path.open("rb") as f:
        if _exact(f, 8) != RAW_MAGIC:
            raise ValueError("wrong qNEP raw magic")
        version, endian, n, d = struct.unpack("<IIii", _exact(f, 16))
        if version not in (1, 2, 3) or endian != ENDIAN or n < 2 or d != 3*n:
            raise ValueError("unsupported raw version or dimensions")
        start_expected = 8+16+32+16+len(RAW_LAYOUT)+18*8+12 + n*4+n*8+d*8+8+n*8+d*8+9*n*8
        file_expected = start_expected + (2*d*n+7*d*d+18)*8
        if path.stat().st_size != file_expected:
            raise ValueError("invalid qraw length/footer before variable-size header read")
        temp, step, model_fp, config_fp = struct.unpack("<ddQQ", _exact(f, 32))
        charge, pppm, spacing = struct.unpack("<iid", _exact(f, 16))
        if _exact(f, len(RAW_LAYOUT)) != RAW_LAYOUT:
            raise ValueError("unsupported qraw layout")
        cell = np.frombuffer(_exact(f, 18*8), "<f8").copy()
        pbc = np.frombuffer(_exact(f, 12), "<i4").copy()
        types = np.frombuffer(_exact(f, n*4), "<i4").copy()
        masses = np.frombuffer(_exact(f, n*8), "<f8").copy()
        positions = np.frombuffer(_exact(f, d*8), "<f8").copy()
        energy = struct.unpack("<d", _exact(f, 8))[0]
        site_energy = np.frombuffer(_exact(f, n*8), "<f8").copy()
        force = np.frombuffer(_exact(f, d*8), "<f8").copy()
        virial = np.frombuffer(_exact(f, 9*n*8), "<f8").copy()
        start = f.tell()
    vb, cb, kb = d*n*8, 3*d*d*8, d*d*8
    coarse_v, coarse_c = start+vb+cb+kb, start+2*vb+cb+kb
    footer = start+2*vb+2*cb+kb
    if path.stat().st_size != footer+18*8:
        raise ValueError("invalid qraw length/footer")
    stats = np.memmap(path, mode="r", dtype="<f8", offset=footer, shape=(18,))
    common_bad = (not np.isfinite(stats).all() or stats[16] != step or stats[17] != float(version)
                 or np.any(stats < 0) or stats[1] > 1e-4)
    old_bad = max(stats[2],stats[4],stats[6],stats[7],stats[8],stats[14],stats[15]) > .05
    new_bad = stats[3] > 1e-4 or max(stats[4],stats[6],stats[7],stats[8],stats[14],stats[15]) > .05
    if common_bad or (version == 1 and old_bad) or (version in (2,3) and new_bad):
        raise ValueError("qraw diagnostics fail accepted generation thresholds")
    if (not np.isfinite([temp, step, spacing, energy]).all() or temp <= 0 or step <= 0 or spacing <= 0
        or charge not in (1,2) or pppm != 1 or config_fp == 0 or not np.isfinite(cell).all()
        or not np.isfinite(masses).all() or not np.isfinite(positions).all() or not np.isfinite(force).all() or not np.isfinite(virial).all()
        or not np.isfinite(site_energy).all() or np.any(masses <= 0) or np.any(pbc != 1)
        or abs(float(site_energy.sum())-energy) > 1e-9*max(1.,abs(energy))):
        raise ValueError("invalid qraw source geometry or diagnostics")
    # Scan files in bounded chunks so NaNs cannot hide in memmapped derivative arrays.
    for off, count in ((start,d*n),(start+d*n*8,d*d*3),(start+(d*n+d*d*3)*8,d*d),
                       (coarse_v,d*n),(coarse_c,3*d*d)):
        a = np.memmap(path, mode="r", dtype="<f8", offset=off, shape=(count,))
        for i in range(0,count,1<<18):
            if not np.isfinite(a[i:i+(1<<18)]).all():
                raise ValueError("non-finite qraw derivative matrix")
    def matrix(offset, shape):
        return np.memmap(path, mode="r", dtype="<f8", offset=offset, shape=shape)
    return dict(path=path, version=version, n=n, d=d, temperature=temp, step=step,
                model_fingerprint=model_fp, config_fingerprint=config_fp, cell=cell[:9].reshape(3,3),
                types=types, masses=masses, positions=positions.reshape(3,n).T, force=force.reshape(3,n).T,
                site_energy=site_energy, energy=energy, start=start, footer=footer,
                v=matrix(start,(d,n)), c=matrix(start+vb,(3,d,d)), k=matrix(start+vb+cb,(d,d)),
                coarse_v=matrix(coarse_v,(d,n)), coarse_c=matrix(coarse_c,(3,d,d)), diagnostics=stats.copy(),
                source_fingerprint=_fnv64(path))


def _validate_additive(a):
    n, p = int(a["n"]), int(a["beads"])
    if n < 2 or p < 1: raise ValueError("invalid additive dimensions")
    vals = [a[k] for k in ("temperature","epsilon","response_max","response_tolerance")]
    if not np.isfinite(vals).all() or vals[0] <= 0 or vals[1] <= 0 or vals[2] < 0 or vals[3] < 0 or vals[2] > vals[3]:
        raise ValueError("invalid additive temperature, epsilon, or response budget")
    if int(a["training_frames"]) < 1 or int(a["validation_frames"]) < 1:
        raise ValueError("frame counts must be positive")
    sites = a["sites"]
    if len(sites) != n: raise ValueError("wrong site count")
    edges = set()
    for i,s in enumerate(sites):
        if int(s["i"]) != i: raise ValueError("sites must be in original atom order")
        es = s["neighbors"]
        keys = [(int(j),tuple(map(int,img))) for j,img in es]
        if len(keys) != len(set(keys)) or any(j < 0 or j >= n or j == i or len(img)!=3 for j,img in keys):
            raise ValueError("invalid or duplicate neighbor image key")
        z=len(keys); ell=np.asarray(s["ell"],float); b=np.asarray(s["B"],float)
        if ell.shape != (z,3) or b.shape != (3*z,3*z) or not np.isfinite(ell).all() or not np.isfinite(b).all():
            raise ValueError("invalid local coefficient dimensions or values")
        scale=float(np.max(np.abs(b))) if b.size else 0.0
        if scale and np.linalg.norm(b/scale-(b/scale).T) > 1e-12*np.linalg.norm(b/scale): raise ValueError("asymmetric local B")
        edges.update((i,j,img) for j,img in keys)
    for i,j,img in edges:
        if (j,i,tuple(-x for x in img)) not in edges: raise ValueError("missing reverse image edge")
    seen={0}; todo=[0]
    while todo:
        i=todo.pop()
        for j,img in sites[i]["neighbors"]:
            if j not in seen: seen.add(j); todo.append(j)
    if len(seen)!=n: raise ValueError("neighbor graph is disconnected")
    return True


def _unique_temp_path(destination):
    destination=Path(destination)
    fd,name=tempfile.mkstemp(prefix=f".{destination.name}.",suffix=".tmp",dir=destination.parent)
    os.close(fd)
    return Path(name)


def write_additive(path, data):
    _validate_additive(data); path=Path(path)
    if path.exists(): raise FileExistsError(path)
    tmp=_unique_temp_path(path)
    try:
        with tmp.open("wb") as f:
            f.write(ADD_MAGIC)
            f.write(struct.pack("<IIiiQ",1,ENDIAN,data["n"],data["beads"],data["source_fingerprint"]))
            f.write(struct.pack("<ddddQQ",data["temperature"],data["epsilon"],data["response_max"],data["response_tolerance"],data["training_frames"],data["validation_frames"]))
            for s in data["sites"]:
                z=len(s["neighbors"]); f.write(struct.pack("<i",z))
                for j,img in s["neighbors"]: f.write(struct.pack("<4i",j,*img))
                np.asarray(s["ell"],dtype="<f8").tofile(f); np.asarray(s["B"],dtype="<f8").tofile(f)
            f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
        tmp.unlink()
    finally:
        if tmp.exists(): tmp.unlink()


def read_additive(path):
    with Path(path).open("rb") as f:
        if _exact(f,8)!=ADD_MAGIC: raise ValueError("wrong GPJAADD1 magic")
        version,endian,n,p,fp=struct.unpack("<IIiiQ",_exact(f,24))
        if (version,endian)!=(1,ENDIAN): raise ValueError("unsupported additive package version")
        temp,eps,rmax,rtol,ntrain,nval=struct.unpack("<ddddQQ",_exact(f,48))
        sites=[]
        for i in range(n):
            (z,)=struct.unpack("<i",_exact(f,4))
            if z<0 or z>3*n: raise ValueError("invalid local coordination")
            neigh=[struct.unpack("<4i",_exact(f,16)) for _ in range(z)]
            ell=np.frombuffer(_exact(f,24*z),"<f8").copy().reshape(z,3)
            b=np.frombuffer(_exact(f,8*(3*z)**2),"<f8").copy().reshape(3*z,3*z)
            sites.append(dict(i=i,neighbors=[(x[0],tuple(x[1:])) for x in neigh],ell=ell,B=b))
        if f.read(1): raise ValueError("trailing data in additive package")
    a=dict(n=n,beads=p,source_fingerprint=fp,temperature=temp,epsilon=eps,response_max=rmax,
           response_tolerance=rtol,training_frames=ntrain,validation_frames=nval,sites=sites)
    _validate_additive(a); return a


def assemble_additive(pack, positions, cell):
    """Return unweighted Cartesian Kadd and Hadd[axis, displacement-SoA, velocity-SoA]."""
    _validate_additive(pack); x=np.asarray(positions,float); h=np.asarray(cell,float); n=pack["n"]; d=3*n
    if x.shape!=(n,3) or h.shape!=(3,3) or not np.isfinite(x).all() or not np.isfinite(h).all(): raise ValueError("geometry dimensions mismatch")
    k=np.zeros((d,d)); hh=np.zeros((3,d,d)); grad=np.zeros((n,3))
    for s in pack["sites"]:
        i=s["i"]; edges=s["neighbors"]; z=len(edges); b=np.asarray(s["B"]); ell=np.asarray(s["ell"])
        sm=np.zeros((3*z,d)); geom=[]
        for e,(j,img) in enumerate(edges):
            for a in range(3): sm[3*e+a,a*n+j]+=1.; sm[3*e+a,a*n+i]-=1.
            geom.append(x[j]+h@np.asarray(img)-x[i])
        k += sm.T@b@sm
        grad += (sm.T@ell.reshape(-1)).reshape(3,n).T
        for e,(j,img) in enumerate(edges):
            for alpha in range(3):
                # Formula gives H^T rows (velocity), columns (displacement).
                hh[alpha,np.arange(3)*n+j,:] -= geom[e][alpha]*b[3*e:3*e+3,:]@sm
    hh=hh.transpose(0,2,1)
    if not np.isfinite(k).all() or not np.isfinite(hh).all() or not np.isfinite(grad).all():
        raise ValueError("additive assembly produced non-finite K, H, or gradient")
    return {"Kadd":k,"Hadd":hh,"linear_gradient":grad}


def align_bead_frame(reference, bead, cell, previous=None):
    h=np.asarray(cell,float); inv=np.linalg.inv(h); ref=np.asarray(reference,float); b=np.asarray(bead,float)
    df=(b-ref)@inv.T; df-=np.rint(df); aligned=ref+df@h.T
    return aligned, aligned.copy() if previous is None else previous


def align_bead_ring(beads, cell):
    """Unwrap adjacent ring-polymer links and reject winding or half-cell ties."""
    h=np.asarray(cell,float); inv=np.linalg.inv(h); raw=[np.asarray(b,float) for b in beads]
    if not raw: raise ValueError("empty bead ring")
    jumps=[]
    for i in range(len(raw)):
        df=(raw[(i+1)%len(raw)]-raw[i])@inv.T; jump=df-np.rint(df)
        if np.any(np.abs(np.abs(jump)-.5)<1e-10): raise ValueError("bead ring has a half-cell branch ambiguity")
        jumps.append(jump)
    if np.max(np.abs(np.sum(jumps,axis=0)))>1e-7: raise ValueError("bead ring has unresolved periodic winding")
    out=[raw[0].copy()]
    for i in range(1,len(raw)): out.append(out[-1]+jumps[i-1]@h.T)
    return out


def unwrap_centroid_frame(centroid, cell, previous):
    c=np.asarray(centroid,float)
    if previous is None:return c.copy()
    df=(c-previous)@np.linalg.inv(cell).T; df-=np.rint(df)
    return previous+df@cell.T


def _fit_design(a,y):
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    if len(s)==0 or s[-1]<=s[0]*1e-11: raise ValueError("force design is rank deficient")
    theta=vt.T@((u.T@y)/s); residual=a@theta-y
    return {"theta":theta,"relative_residual":float(np.linalg.norm(residual)/max(np.linalg.norm(y),1e-300)),"design_rank":len(s)}


def check_baseline_projection(cartesian_hessian, masses, tolerance=.05):
    """Check mass-weighted Hessian asymmetry and native translation projection."""
    k=np.asarray(cartesian_hessian,float); m=np.asarray(masses,float)
    if k.shape!=(3*len(m),3*len(m)) or not np.isfinite(k).all() or not np.isfinite(m).all() or np.any(m<=0):
        raise ValueError("invalid baseline Hessian or masses")
    sqrtm=np.tile(np.sqrt(m),3); raw=k/(sqrtm[:,None]*sqrtm[None,:])
    sym=.5*(raw+raw.T); asym=float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(sym),1e-300))
    projector=_translation_projector(m); projected=projector@sym@projector
    change=float(np.linalg.norm(projected-sym)/max(np.linalg.norm(sym),1e-300))
    return {"asymmetry":asym,"relative_change":change,"passed":bool(asym<=tolerance and change<=tolerance),"D":sym,"projected_D":projected}


def _raw_dimensions(path):
    with Path(path).open("rb") as f:
        if _exact(f,8)!=RAW_MAGIC: raise ValueError("wrong qNEP raw magic")
        version,endian,n,d=struct.unpack("<IIii",_exact(f,16))
    if version not in (1,2,3) or endian!=ENDIAN or n<2 or d!=3*n: raise ValueError("unsupported raw version or dimensions")
    return n,d


def fit_force_least_squares(q, force, dt, modes, masses):
    q=np.asarray(q,float); force=np.asarray(force,float); mass=np.asarray(masses,float)
    # q is Cartesian displacement N x (3N) in mass-weighted coordinates.
    sqrtm=np.tile(np.sqrt(mass),3); designs=[]; target=[]
    for x,f in zip(q,force):
        qw=x*sqrtm; fw=f.reshape(-1)/sqrtm
        designs.append(np.column_stack([m@qw for m in modes])); target.append(-fw-dt@qw)
    a=np.concatenate(designs); y=np.concatenate(target)
    return _fit_design(a,y)


def solve_psd_qp(hessian, gradient, dt, modes, epsilon, max_cuts=40, *, design=None, target=None):
    """Fit-space-whitened spectral QP; the compatibility path whitens a supplied Hessian."""
    h0=np.asarray(dt,float); mats=[np.asarray(m,float) for m in modes]; p=len(mats)
    if p>MAX_PARAMETERS or h0.size>MAX_DENSE_ELEMENTS: raise ValueError("dense QP resource limit exceeded")
    if design is not None:
        a=np.asarray(design,float); y=np.asarray(target,float)
        u,s,vt=np.linalg.svd(a,full_matrices=False)
        if len(s)<p or s[-1]<=s[0]*1e-11: raise ValueError("QP least-squares design is rank deficient")
        transform=vt.T/s[None,:]
        gw=u.T@y
        whiten_h=np.eye(p)
    else:
        hessian=np.asarray(hessian,float); gradient=np.asarray(gradient,float)
        if hessian.shape!=(p,p) or not np.isfinite(hessian).all(): raise ValueError("invalid QP objective")
        l=np.linalg.cholesky(hessian)
        transform=np.linalg.solve(l.T,np.eye(p))
        gw=np.linalg.solve(l,gradient); whiten_h=np.eye(p)
    eta=gw.copy(); cuts=0; rows=[]; bounds=[]
    for _ in range(max_cuts):
        theta=transform@eta
        eig,vec=np.linalg.eigh(h0+sum(theta[r]*mats[r] for r in range(p)))
        if eig[0]>=epsilon*(1-1e-9):
            return dict(theta=theta,min_eigenvalue=float(eig[0]),cuts=cuts)
        u0=vec[:,0]; row=np.array([u0@m@u0 for m in mats])@transform; rhs=epsilon-u0@h0@u0
        rows.append(row); bounds.append(rhs); cuts+=1
        eta=_dual_qp(whiten_h,gw,np.asarray(rows),np.asarray(bounds))
    raise RuntimeError("spectral-cut QP did not converge")


def _dual_qp(h,g,a,b):
    # Minimize .5*x'Hx-g'x subject to A x >= b by cyclic exact dual coordinate updates.
    hinva=np.linalg.solve(h,a.T); gram=a@hinva; rhs=b-a@np.linalg.solve(h,g)
    lam=np.zeros(len(b)); diag=np.diag(gram)
    if np.any(diag<=1e-20): raise RuntimeError("infeasible or singular spectral QP cut")
    for _ in range(200000):
        old=lam.copy()
        for i in range(len(lam)):
            lam[i]=max(0.,lam[i]+(rhs[i]-gram[i]@lam)/diag[i])
        if np.max(np.abs(lam-old))<1e-11 and np.min(gram@lam-rhs)>-1e-9: break
    else: raise RuntimeError("dual QP coordinate solver did not converge")
    x=np.linalg.solve(h,g)+hinva@lam
    if np.min(a@x-b)<-1e-8: raise RuntimeError("QP primal feasibility failure")
    if np.max(np.abs(lam*(a@x-b)))>1e-7: raise RuntimeError("QP KKT complementarity failure")
    return x


def check_independent_response(reference, data, probes, tolerance, temperature=1.):
    r=np.asarray(reference,float); c=np.asarray(data,float)
    w=np.column_stack(probes) if isinstance(probes,(list,tuple)) else np.asarray(probes,float)
    if w.ndim==1: w=w[:,None]
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or c.shape!=r.shape or w.ndim!=2 or w.shape[0]!=r.shape[0]:
        raise ValueError("response matrices and probes have inconsistent shapes")
    if not np.isfinite([temperature,tolerance]).all() or temperature<=0 or tolerance<0 or not np.isfinite(r).all() or not np.isfinite(c).all() or not np.isfinite(w).all():
        raise ValueError("response inputs must be finite and temperature positive")
    if np.linalg.matrix_rank(w)<min(w.shape): raise ValueError("response probes are rank deficient")
    predicted=KB_EV*temperature*(w.T@np.linalg.solve(r,w)); observed=w.T@c@w
    predicted=.5*(predicted+predicted.T); observed=.5*(observed+observed.T)
    pe,pv=np.linalg.eigh(predicted); ce=np.linalg.eigvalsh(observed)
    if pe[0]<=0 or not np.isfinite(pe).all(): raise ValueError("predicted response is not positive definite")
    if ce[0]<=np.finfo(float).eps*max(ce[-1],1e-300)*len(ce): raise ValueError("held-out response covariance is rank deficient")
    invroot=(pv/np.sqrt(pe))@pv.T
    whitened=invroot@observed@invroot-np.eye(len(pe))
    error=float(np.max(np.abs(np.linalg.eigvalsh(.5*(whitened+whitened.T)))))
    frobenius=float(np.linalg.norm(observed-predicted)/max(np.linalg.norm(predicted),1e-300))
    return {"relative_errors":[error],"max_relative_error":error,"frobenius_relative_error":frobenius,"passed":bool(error<=tolerance)}


def check_integration_by_parts(q, force, temperature):
    q=np.asarray(q,float); f=np.asarray(force,float); kb_t=KB_EV*temperature
    if q.ndim!=2 or f.shape!=q.shape or len(q)<1 or not np.isfinite([temperature]).all() or temperature<=0 or not np.isfinite(q).all() or not np.isfinite(f).all():
        raise ValueError("invalid IBP arrays or temperature")
    cross=(f.T@q)/len(q); target=-kb_t*np.eye(q.shape[1])
    err=float(np.linalg.svd((cross-target)/kb_t,compute_uv=False)[0])
    return {"relative_error":err,"cross":cross,"passed":bool(np.isfinite(err))}


def _heldout_force_relative_residual(q, force, stiffness):
    residual=force+q@stiffness.T
    return float(np.linalg.norm(residual)/max(np.linalg.norm(force),1e-300))


def _check_reference_branch(positions, reference, cell):
    inv=np.linalg.inv(cell)
    for frame in positions:
        if np.max(np.abs((frame-reference)@inv.T))>=0.45:
            raise ValueError("sample centroid lies outside the fixed-reference JA branch limit 0.45")


def read_xyz_frames(path):
    """Read extended XYZ with arbitrary Properties order and GPUMD bead columns."""
    frames=[]
    with Path(path).open(encoding="utf-8") as f:
        while True:
            line=f.readline()
            if not line: break
            if not line.strip(): continue
            n=int(line); header=f.readline().strip(); props=re.search(r"(?:^| )Properties=([^ ]+)",header)
            lat=re.search(r'(?:^| )Lattice="([^"]+)"',header)
            if props is None or lat is None: raise ValueError("XYZ needs Properties and Lattice")
            fields=props.group(1).split(":"); names={}; col=0; offset=0
            if len(fields)%3: raise ValueError("malformed XYZ Properties descriptor")
            while col<len(fields):
                name,typ,count=fields[col],fields[col+1],int(fields[col+2])
                if name in names: raise ValueError("duplicate XYZ property")
                names[name]=(offset,count,typ); offset+=count; col+=3
            if "species" not in names or "pos" not in names or not ("force" in names or "forces" in names): raise ValueError("XYZ missing species/pos/force")
            arr=[f.readline().split() for _ in range(n)]
            width=sum(v[1] for v in names.values())
            if any(len(row)!=width for row in arr): raise ValueError("truncated or malformed XYZ atom row")
            sy,sp=names["species"][0],names["pos"][0]; fk=names.get("force",names.get("forces"))[0]
            sym=[r[sy] for r in arr]
            pos=np.array([[float(x) for x in r[sp:sp+3]] for r in arr])
            force=np.array([[float(x) for x in r[fk:fk+3]] for r in arr])
            time_m=re.search(r"(?:^| )Time=([^ ]+)",header); step_m=re.search(r"(?:^| )Step=([^ ]+)",header)
            if time_m is None and step_m is None: raise ValueError("XYZ frames require Time or Step")
            step=float(time_m.group(1)) if time_m else int(step_m.group(1))
            if not math.isfinite(step): raise ValueError("non-finite XYZ time")
            pbc=re.search(r'(?:^| )pbc="([^"]+)"',header)
            if pbc is None or pbc.group(1).split()!=["T","T","T"]: raise ValueError("XYZ must declare three-dimensional periodic boundaries")
            for key, count, typ in names.values():
                if count<=0 or typ not in ("S","R","I","L"): raise ValueError("invalid XYZ Properties descriptor")
            if names["species"][1:]!=(1,"S") or names["pos"][1:]!=(3,"R") or names.get("force",names.get("forces"))[1:]!=(3,"R"):
                raise ValueError("unsupported XYZ species/position/force property types")
            # GPUMD writes column lattice vectors in the order h00,h10,h20,...
            lattice=[float(x) for x in lat.group(1).split()]
            if len(lattice)!=9: raise ValueError("XYZ Lattice must contain nine values")
            cell=np.array(lattice).reshape(3,3).T
            if not np.isfinite(pos).all() or not np.isfinite(force).all() or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1e-12: raise ValueError("non-finite or singular XYZ frame")
            if frames and step<=frames[-1]["step"]: raise ValueError("XYZ Time/Step must increase strictly")
            if frames and (sym!=frames[0]["symbols"] or not np.allclose(cell,frames[0]["cell"],atol=1e-10,rtol=1e-10)):
                raise ValueError("XYZ symbols or fixed cell changed")
            frames.append(dict(symbols=sym,positions=pos,forces=force,cell=cell,step=step,header=header))
    return frames


def _read_cell_model(path):
    with Path(path).open(encoding="utf-8") as f:
        try: n=int(f.readline())
        except ValueError as exc: raise ValueError("cell model must be an extended XYZ structure") from exc
        header=f.readline().strip(); lat=re.search(r'(?:^| )Lattice="([^\"]+)"',header)
        if n<1 or lat is None: raise ValueError("cell model requires atom count and Lattice")
        lattice=np.fromstring(lat.group(1),sep=" ")
        if lattice.size!=9 or not np.isfinite(lattice).all(): raise ValueError("invalid cell model lattice")
        pbc=re.search(r'(?:^| )pbc="([^\"]+)"',header)
        if pbc is not None and [token.upper() for token in pbc.group(1).split()]!=["T","T","T"]: raise ValueError("cell model must be fully periodic")
        props=re.search(r"(?:^| )Properties=([^ ]+)",header)
        species_col=0
        if props:
            fields=props.group(1).split(":"); col=0; found=False
            if len(fields)%3: raise ValueError("malformed cell model Properties")
            for name,typ,count in zip(fields[::3],fields[1::3],fields[2::3]):
                count=int(count)
                if name=="species":
                    if typ!="S" or count!=1: raise ValueError("cell model species must be one string")
                    species_col=col; found=True
                col+=count
            if not found: raise ValueError("cell model lacks species property")
        symbols=[]
        for _ in range(n):
            row=f.readline().split()
            if len(row)<=species_col: raise ValueError("truncated cell model atom rows")
            symbols.append(row[species_col])
    if len(set(symbols))==0 or any(not x for x in symbols): raise ValueError("invalid cell model symbols")
    cell=lattice.reshape(3,3).T
    if abs(np.linalg.det(cell))<1e-12: raise ValueError("singular cell model lattice")
    return n,symbols,cell


def collect(bead_files, temperature, output, mean_model, masses, cell_model=None):
    if not math.isfinite(float(temperature)) or temperature<=0: raise ValueError("temperature must be finite and positive")
    output=Path(output); mean_model=Path(mean_model)
    if output.resolve()==mean_model.resolve(): raise ValueError("samples and mean-model paths must differ")
    if output.exists() or mean_model.exists(): raise FileExistsError("refusing to overwrite collected samples or mean model")
    model=None if cell_model is None else _read_cell_model(cell_model)
    trajectories=[read_xyz_frames(p) for p in bead_files]; p=len(trajectories)
    if p<1 or not trajectories[0]: raise ValueError("no bead frames")
    if any(len(t)!=len(trajectories[0]) for t in trajectories): raise ValueError("bead frame counts differ")
    n=len(trajectories[0][0]["symbols"]); m=np.asarray(masses,float)
    if model is not None and (model[0]!=n or model[1]!=trajectories[0][0]["symbols"]): raise ValueError("cell model atom count/order does not match bead dump")
    if m.shape!=(n,) or np.any(m<=0) or not np.isfinite(m).all(): raise ValueError("explicit positive masses required")
    frames=[]; previous=None; first_com=None
    for fi in range(len(trajectories[0])):
        base=trajectories[0][fi]; cell=base["cell"] if model is None else model[2]
        if model is not None and np.max(np.abs(base["cell"]-cell))>5.1e-9+8*np.finfo(float).eps*max(1.,float(np.max(np.abs(cell)))):
            raise ValueError("bead dump cell differs from full-precision cell model by more than 8-decimal serialization error")
        bead_positions=[base["positions"]]; fsum=base["forces"].copy()
        for traj in trajectories[1:]:
            b=traj[fi]
            if b["step"]!=base["step"] or b["symbols"]!=base["symbols"] or not np.allclose(b["cell"],base["cell"],atol=1e-10,rtol=1e-10): raise ValueError("bead frames are not synchronized")
            if model is not None and np.max(np.abs(b["cell"]-cell))>5.1e-9+8*np.finfo(float).eps*max(1.,float(np.max(np.abs(cell)))):
                raise ValueError("bead dump cell differs from full-precision cell model by more than 8-decimal serialization error")
            bead_positions.append(b["positions"]); fsum+=b["forces"]
        aligned=align_bead_ring(bead_positions,cell); cent=np.mean(aligned,axis=0)
        bead0_cent=np.mean([align_bead_frame(bead_positions[0],b,cell)[0] for b in bead_positions],axis=0)
        scale=max(1.,float(np.max(np.abs(cell))),float(np.max(np.abs(bead_positions[0]))))
        if np.max(np.abs(cent-bead0_cent))>16*p*np.finfo(float).eps*scale:
            raise ValueError("adjacent-link ring centroid differs from GPUMD bead0-MIC production centroid; this ring branch is unsupported")
        cent=unwrap_centroid_frame(cent,cell,previous); previous=cent.copy()
        com=np.sum(m[:,None]*cent,axis=0)/m.sum()
        if first_com is None: first_com=com.copy()
        cent-=com-first_com; frames.append((base["step"],cent,fsum/p,cell,base["symbols"]))
    train=max(1,2*len(frames)//3); avg=np.mean([x[1] for x in frames[:train]],axis=0)
    arrays=dict(positions=np.array([x[1] for x in frames]),forces=np.array([x[2] for x in frames]),masses=m,
                cell=frames[0][3],temperature=float(temperature),beads=p,steps=np.array([x[0] for x in frames],dtype=float),symbols=np.array(frames[0][4]))
    sample_tmp=None; mean_tmp=None; installed=[]
    try:
        sample_tmp=_unique_temp_path(output); mean_tmp=_unique_temp_path(mean_model)
        with sample_tmp.open("wb") as f: np.savez(f,**arrays)
        with mean_tmp.open("w",encoding="utf-8") as f:
            f.write(f"{n}\nLattice=\"{' '.join(map(str,frames[0][3].T.ravel()))}\" Properties=species:S:1:pos:R:3:mass:R:1 pbc=\"T T T\"\n")
            for sym,pos,mass in zip(frames[0][4],avg,m): f.write(f"{sym} {pos[0]:.14g} {pos[1]:.14g} {pos[2]:.14g} {mass:.14g}\n")
        os.link(sample_tmp,output); installed.append(output)
        os.link(mean_tmp,mean_model); installed.append(mean_model)
    except Exception:
        for destination in installed: destination.unlink(missing_ok=True)
        raise
    finally:
        if sample_tmp is not None: sample_tmp.unlink(missing_ok=True)
        if mean_tmp is not None: mean_tmp.unlink(missing_ok=True)


def cutoff_graph(positions, cell, cutoff, neighbor_lists=None):
    n=len(positions); sites=[]
    if neighbor_lists is None: neighbor_lists=_cutoff_neighbors(positions,cell,cutoff)
    for i,neigh in enumerate(neighbor_lists):
        sites.append(dict(i=i,neighbors=neigh,ell=np.zeros((len(neigh),3)),B=np.zeros((3*len(neigh),3*len(neigh)))))
    return sites


def _cutoff_neighbors(positions, cell, cutoff):
    n=len(positions); inv=np.linalg.inv(cell); result=[]
    for i in range(n):
        neigh=[]
        for j in range(n):
            if i==j: continue
            df=(positions[j]-positions[i])@inv.T; im=-np.rint(df).astype(int)
            if np.linalg.norm(positions[j]+cell@im-positions[i])<=cutoff:
                neigh.append((j,tuple(map(int,im))))
        result.append(neigh)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="command",required=True)
    c=sub.add_parser("collect"); c.add_argument("--bead-files",nargs="+",required=True); c.add_argument("--temperature",type=float,required=True); c.add_argument("--output",required=True); c.add_argument("--mean-model",required=True); c.add_argument("--masses",nargs="+",type=float,required=True); c.add_argument("--cell-model")
    f=sub.add_parser("fit"); f.add_argument("--raw",required=True); f.add_argument("--samples",required=True); f.add_argument("--output",required=True); f.add_argument("--cutoff",type=float,required=True); f.add_argument("--epsilon",type=float,required=True); f.add_argument("--response-tolerance",type=float,required=True)
    a=ap.parse_args()
    if a.command=="collect": collect(a.bead_files,a.temperature,a.output,a.mean_model,a.masses,a.cell_model)
    else: fit_cli(a)


def _load_samples_npz(path, expected_n):
    required={"positions","forces","masses","cell","temperature","beads","steps","symbols"}
    with zipfile.ZipFile(path) as zf:
        members={}
        for info in zf.infolist():
            key=Path(info.filename).stem if info.filename.endswith(".npy") else None
            if key in required:
                if key in members: raise ValueError(f"duplicate sample array {key}")
                members[key]=info
        if required-set(members): raise ValueError(f"sample archive lacks arrays: {sorted(required-set(members))}")
        if sum(members[k].file_size for k in required)>MAX_SAMPLE_BYTES:
            raise ValueError("sample arrays exceed the 256 MiB uncompressed input limit")
        headers={}
        for key,info in members.items():
            with zf.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else: raise ValueError("unsupported sample NPY version")
                payload_bytes=info.file_size-stream.tell()
            if dtype.hasobject: raise ValueError("sample arrays may not use pickle/object dtypes")
            expected_bytes=math.prod(shape)*dtype.itemsize
            if payload_bytes!=expected_bytes: raise ValueError(f"sample {key} NPY payload length does not match its declared shape")
            headers[key]=(shape,dtype)
        shape,xdtype=headers["positions"]
        if len(shape)!=3 or shape[1:]!=(expected_n,3) or shape[0]<3 or xdtype.kind!="f":
            raise ValueError("sample position shape must be frames x raw atoms x 3")
        frames=shape[0]
        expected={"forces":(frames,expected_n,3),"masses":(expected_n,),"cell":(3,3),
                  "temperature":(),"beads":(),"steps":(frames,),"symbols":(expected_n,)}
        for key,want in expected.items():
            got,dtype=headers[key]
            if got!=want: raise ValueError(f"invalid sample {key} shape")
            if key in ("forces","masses","cell","temperature","steps") and dtype.kind!="f": raise ValueError(f"sample {key} must be floating point")
        if headers["symbols"][1].kind not in ("U","S"): raise ValueError("sample symbols must be strings")
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key] for key in required}


def fit_cli(args):
    raw_n,raw_d=_raw_dimensions(args.raw)
    if raw_d-3>MAX_INTERNAL_DIM or raw_d**2>MAX_DENSE_ELEMENTS:
        raise ValueError("dense fit dimension limit exceeded before qraw scan and sample loading")
    raw=read_raw(args.raw)
    sample=_load_samples_npz(args.samples,raw_n)
    x=sample["positions"]; force=sample["forces"]; masses=sample["masses"]; n=len(masses)
    temp=float(sample["temperature"]); pvalue=float(sample["beads"])
    if not math.isfinite(temp) or temp<=0 or not math.isfinite(pvalue) or pvalue<1 or pvalue!=int(pvalue): raise ValueError("invalid sample temperature or bead count")
    if not math.isfinite(args.epsilon) or args.epsilon<=0 or not math.isfinite(args.response_tolerance) or args.response_tolerance<0: raise ValueError("invalid numerical or response tolerance")
    train=max(1,2*len(x)//3); val=len(x)-train
    if val<1 or raw["n"]!=n: raise ValueError("insufficient samples or raw/sample atom mismatch")
    if x.ndim!=3 or x.shape!=(len(sample["steps"]),n,3) or force.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(force).all(): raise ValueError("invalid sample position/force arrays")
    steps=np.asarray(sample["steps"],float)
    if steps.shape!=(len(x),) or not np.isfinite(steps).all() or np.any(np.diff(steps)<=0): raise ValueError("sample steps must be finite and strictly increasing")
    if sample["cell"].shape!=(3,3) or not np.isfinite(sample["cell"]).all(): raise ValueError("invalid sample cell")
    if masses.shape!=(n,) or not np.isfinite(masses).all() or np.any(masses<=0): raise ValueError("invalid sample masses")
    if not np.allclose(masses,raw["masses"],atol=1e-10,rtol=1e-10): raise ValueError("sample/raw masses differ")
    if int(sample["beads"])<1 or sample["positions"].shape[0]<3: raise ValueError("invalid bead count or insufficient samples")
    syms=[str(z) for z in sample["symbols"]]
    if len(syms)!=n or any(not v for v in syms): raise ValueError("sample symbols shape mismatch")
    for i in range(n):
        for j in range(n):
            if (syms[i]==syms[j]) != (raw["types"][i]==raw["types"][j]): raise ValueError("sample symbols do not partition-match raw types")
    if abs(temp-raw["temperature"])>1e-10*raw["temperature"]: raise ValueError("sample/raw temperatures differ")
    temp=float(raw["temperature"])
    if (3*n)**2>MAX_DENSE_ELEMENTS: raise ValueError("dense fit dimension limit exceeded before allocation")
    if not np.allclose(sample["cell"],raw["cell"],atol=1e-9,rtol=1e-9): raise ValueError("sample/raw cells differ")
    training_mean=np.mean(x[:train],axis=0)
    if np.max(np.abs(raw["positions"]-training_mean))>MEAN_TOLERANCE: raise ValueError(f"raw reference position differs from training mean by {np.max(np.abs(raw['positions']-training_mean)):.3g} A")
    r0=raw["positions"]
    _check_reference_branch(x,raw["positions"],raw["cell"])
    d=3*n
    if d-3>MAX_INTERNAL_DIM or d*d>MAX_DENSE_ELEMENTS: raise ValueError("dense fit dimension limit exceeded before allocation")
    raw_k=np.asarray(raw["k"])
    baseline=check_baseline_projection(raw_k,masses,.05)
    if baseline["asymmetry"]>.05: raise ValueError(f"mass-weighted native Hessian antisymmetry {baseline['asymmetry']:.3g} exceeds 0.05")
    if baseline["relative_change"]>.05: raise ValueError(f"native baseline translation projection change {baseline['relative_change']:.3g} exceeds 0.05")
    if not math.isfinite(args.cutoff) or args.cutoff<=0: raise ValueError("cutoff must be finite and positive")
    neighbors=_cutoff_neighbors(r0,raw["cell"],args.cutoff)
    estimate=sum((3*len(v))*(3*len(v)+1)//2 for v in neighbors)
    if estimate>MAX_PARAMETERS or d*d*estimate>MAX_DENSE_ELEMENTS: raise ValueError("dense gauge resource limit exceeded before allocation")
    sites=cutoff_graph(r0,raw["cell"],args.cutoff,neighbors)
    b=(np.asarray(raw["v"])@np.ones(n)).reshape(3,n).T
    # Center gradient must have no net translation component.
    net_gradient=b.sum(axis=0)
    if not np.isfinite(b).all() or not np.isfinite(net_gradient).all(): raise ValueError("raw gradient is non-finite")
    net_norm=_stable_norm(net_gradient); gradient_norm=_stable_norm(b)
    if not math.isfinite(net_norm) or not math.isfinite(gradient_norm): raise ValueError("raw gradient norm is non-finite")
    if net_norm>1e-8*max(1.,gradient_norm): raise ValueError("raw gradient has nonzero translation component")
    _set_linear_balance(sites,b)
    # Dense full-star basis and exact algebraic gauge are deliberately small-system only.
    basis, mapmat=_local_basis(sites,n)
    # The right singular vectors span the exact algebraic complement of ker(T).
    _,sv,vh=np.linalg.svd(mapmat,full_matrices=False)
    gauge_tol=np.finfo(float).eps*max(mapmat.shape)*sv[0]*10 if len(sv) else 0.
    rank=int(np.sum(sv>gauge_tol)) if len(sv) else 0
    q=vh[:rank].T
    reduced=[]
    for r in range(rank):
        local=[np.zeros_like(s["B"]) for s in sites]
        for j,(si,b,_) in enumerate(basis): local[si] += q[j,r]*b
        reduced.append(local)
    if len(reduced)>MAX_PARAMETERS: raise ValueError("dense fit parameter limit exceeded")
    sqrtm=np.tile(np.sqrt(masses),3); internal=_translation_projector(masses)
    trans=np.zeros((3*n,3))
    for axis in range(3): trans[axis*n:(axis+1)*n,axis]=np.sqrt(masses)/np.linalg.norm(np.sqrt(masses))
    z=np.linalg.svd(trans,full_matrices=True)[0][:,3:]
    smodes=[]
    for blocks in reduced:
        kval=np.zeros((3*n,3*n))
        for site,b in zip(sites,blocks):
            smat=np.zeros((3*len(site["neighbors"]),3*n)); i=site["i"]
            for e,(j,img) in enumerate(site["neighbors"]):
                for xyz in range(3): smat[3*e+xyz,xyz*n+j]+=1.;smat[3*e+xyz,xyz*n+i]-=1.
            kval+=smat.T@b@smat
        smodes.append(internal@(kval/(sqrtm[:,None]*sqrtm[None,:]))@internal)
    dt=baseline["projected_D"]
    dtint=z.T@dt@z; modeint=[z.T@m@z for m in smodes]
    qtrain=(x[:train]-r0).transpose(0,2,1).reshape(train,-1)
    if train*(3*n-3)*rank>MAX_DENSE_ELEMENTS: raise ValueError("dense force design resource limit exceeded before allocation")
    ftrain=force[:train].transpose(0,2,1).reshape(train,-1)
    designs=[]; targets=[]
    for qv,fv in zip(qtrain,ftrain):
        qw=z.T@(qv*sqrtm); fw=z.T@(fv/sqrtm)
        designs.append(np.column_stack([m@qw for m in modeint])); targets.append(-fw-dtint@qw)
    amat=np.concatenate(designs); yvec=np.concatenate(targets)
    forcefit=_fit_design(amat,yvec)
    qp=solve_psd_qp(None,None,dtint,modeint,args.epsilon,design=amat,target=yvec)
    theta=qp["theta"]
    for coef,blocks in zip(theta,reduced):
        for site,bblock in zip(sites,blocks): site["B"] += coef*bblock
    # The held-out covariance checks every orthonormal internal direction.
    qval_uncentered=(x[train:]-r0).transpose(0,2,1).reshape(val,-1)*sqrtm
    qval=qval_uncentered-qval_uncentered.mean(axis=0)
    cov=qval.T@qval/val
    d0=dt+sum(theta[r]*smodes[r] for r in range(len(theta)))
    dint=z.T@d0@z; eig,vec=np.linalg.eigh(0.5*(dint+dint.T))
    if eig[0]<args.epsilon*(1-1e-9): raise ValueError("full internal spectrum failed epsilon check")
    shifted=dint-0.5*args.epsilon*np.eye(3*n-3)
    chol=np.linalg.cholesky(shifted); rebuild=chol@chol.T
    eta=float(np.linalg.norm(rebuild-shifted))
    if eta>=0.5*args.epsilon: raise ValueError("shifted Cholesky reconstruction bound eta>=epsilon/2")
    probes=[vec[:,j] for j in range(3*n-3)]
    covint=z.T@cov@z
    response=check_independent_response(dint,covint,probes,args.response_tolerance,temp)
    fval=force[train:].transpose(0,2,1).reshape(val,-1)/sqrtm
    qval_int=qval@z; qval_uncentered_int=qval_uncentered@z; fval_int=fval@z
    ibp=check_integration_by_parts(qval_int,fval_int,temp)
    train_force_int=(ftrain/sqrtm)@z
    train_relative=float(np.linalg.norm(amat@theta-yvec)/max(np.linalg.norm(train_force_int),1e-300))
    heldout_relative=_heldout_force_relative_residual(qval_uncentered_int,fval_int,dint)
    if not ibp["passed"] or ibp["relative_error"]>args.response_tolerance: raise ValueError(f"held-out force-position IBP error {ibp['relative_error']:.3g} exceeds budget")
    if response["max_relative_error"]>args.response_tolerance: raise ValueError(f"independent response error {response['max_relative_error']:.3g} exceeds budget; theta={theta}; evals={eig}")
    pack=dict(n=n,beads=int(sample["beads"]),source_fingerprint=raw["source_fingerprint"],temperature=temp,epsilon=args.epsilon,response_max=response["max_relative_error"],response_tolerance=args.response_tolerance,training_frames=train,validation_frames=val,sites=sites)
    _validate_additive(pack); write_additive(args.output,pack)
    print(f"raw_diagnostics={raw['diagnostics']} baseline_mass_weighted_asymmetry={baseline['asymmetry']:.6g} baseline_projection_relative_change={baseline['relative_change']:.6g} sampling_interval={float(np.median(np.diff(steps))):.9g} training={train} validation={val} force_relative_residual={train_relative:.6g} heldout_force_relative_residual={heldout_relative:.6g} response_relative_error={response['max_relative_error']:.6g} ibp_relative_error={ibp['relative_error']:.6g} parameters={rank} spectral_cuts={qp['cuts']} certificate_eta={eta:.3g}")


def _translation_projector(masses):
    n=len(masses); q=np.zeros((3*n,3))
    for a in range(3): q[a*n:(a+1)*n,a]=np.sqrt(masses)/np.linalg.norm(np.sqrt(masses))
    return np.eye(3*n)-q@q.T


def _local_basis(sites,n):
    """Full symmetric local-star coordinates and their Cartesian assembly map."""
    d=3*n; basis=[]; local_shapes=[(3*len(s["neighbors"]),)*2 for s in sites]
    for si,s in enumerate(sites):
        size=local_shapes[si][0]; sm=np.zeros((size,d)); i=s["i"]
        for e,(j,img) in enumerate(s["neighbors"]):
            for xyz in range(3): sm[3*e+xyz,xyz*n+j]+=1.; sm[3*e+xyz,xyz*n+i]-=1.
        for a in range(size):
            b=np.zeros((size,size)); b[a,a]=1.; basis.append((si,b,sm.T@b@sm))
            for c in range(a+1,size):
                b=np.zeros((size,size)); b[a,c]=b[c,a]=1/math.sqrt(2); basis.append((si,b,sm.T@b@sm))
    mat=np.column_stack([x[2].ravel() for x in basis]) if basis else np.empty((d*d,0))
    return basis,mat


def _set_linear_balance(sites,b):
    n=len(sites); edges={}
    for s in sites:
        i=s["i"]
        for j,img in s["neighbors"]:
            key=(min(i,j),max(i,j),tuple(img if i<j else (-x for x in img)))
            edges[key]=None
    lap=np.zeros((n,n)); es=[]
    for i,j,img in sorted(edges):
        lap[i,i]+=1;lap[j,j]+=1;lap[i,j]-=1;lap[j,i]-=1;es.append((i,j,img))
    zeta=np.zeros((n,3)); zeta[1:]=np.linalg.solve(lap[1:,1:],b[1:])
    coeff={(i,j,img):-(zeta[j]-zeta[i]) for i,j,img in es}
    for s in sites:
        i=s["i"]
        for e,(j,img) in enumerate(s["neighbors"]):
            if i<j: c=coeff[(i,j,img)]
            else: c=-coeff[(j,i,tuple(-x for x in img))]
            s["ell"][e]=.5*c


if __name__=="__main__": main()

