"""Full-G finite-model HO and independent Fourier DVR spectral controls."""
from pathlib import Path
import hashlib
import sys
import numpy as np


def load_modules(root):
    root=Path(root).resolve()
    material=root/'quantum_material_hamiltonian_v1'/'src'
    legacy=root/'quantum_benchmark_OHO'/'src'
    for path in (legacy,material):
        if not path.is_dir(): raise ValueError('theory root lacks '+str(path))
        if str(path) not in sys.path: sys.path.insert(0,str(path))
    import oho, constrained_embedding, quantum, round2_quantum
    for module,expected in ((oho,legacy/'oho.py'),(constrained_embedding,material/'constrained_embedding.py'),(quantum,legacy/'quantum.py'),(round2_quantum,legacy/'round2_quantum.py')):
        if Path(module.__file__).resolve()!=expected.resolve():
            raise RuntimeError('an incompatible theory module is already loaded: '+module.__name__)
    paths=[material/'local_model.py',material/'constrained_embedding.py',material/'quantum_adapter.py',legacy/'oho.py',legacy/'quantum.py',legacy/'round2_quantum.py']
    return {'oho':oho,'embedding':constrained_embedding,'quantum':quantum,'dvr':round2_quantum,
            'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def whitened_polynomials(model,oho,rotation=None):
    G=np.asarray(model['geometry']['G'],float)
    if G.shape!=(2,2) or not np.isfinite(G).all() or not np.allclose(G,G.T,rtol=0,atol=1e-12):
        raise ValueError('mass metric must be finite symmetric 2x2')
    g,V=np.linalg.eigh(G)
    if g.min()<=max(g.max(),1.)*1e-12: raise ValueError('mass metric is not safely positive definite')
    W=(V/np.sqrt(g))@V.T
    if rotation is not None:
        rotation=np.asarray(rotation,float)
        if rotation.shape!=(2,2) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(2),rtol=0,atol=1e-12):
            raise ValueError('basis rotation must be orthogonal')
        W=W@rotation
    Winv=np.linalg.inv(W)
    forward=[oho.add(*(oho.scale(oho.VARS[j],W[i,j]) for j in range(2))) for i in range(2)]
    forward += [oho.add(*(oho.scale(oho.VARS[j+2],Winv[j,i]) for j in range(2))) for i in range(2)]
    return {key:oho.substitute(model[key],forward) for key in ('H','PE','J1','J3')}


def solve(model,beta,times,dims,method,modules,half_box=(8.,8.),frequencies=(1.,1.),rotation=None):
    hbar=float(model['hbar']); beta=float(beta); times=np.asarray(times,float)
    dims=tuple(dims)
    if (not np.isfinite(beta) or beta<=0 or not np.isfinite(hbar) or hbar<=0 or
        len(dims)!=2 or any(not isinstance(n,(int,np.integer)) or n<2 for n in dims) or
        times.ndim!=1 or times.size<2 or not np.isfinite(times).all() or
        times[0]!=0 or np.any(np.diff(times)<=0)):
        raise ValueError('invalid beta, hbar, dimensions or time grid')
    oho=modules['oho']; pol=whitened_polynomials(model,oho,rotation)
    half_box=np.broadcast_to(np.asarray(half_box,float),(2,))
    frequencies=np.asarray(frequencies,float)
    if not np.isfinite(half_box).all() or np.any(half_box<=0) or frequencies.shape!=(2,) or not np.isfinite(frequencies).all() or np.any(frequencies<=0):
        raise ValueError('positive basis frequencies and DVR box widths required')
    if method=='ho':
        matrices=[oho.weyl_matrix(pol[k],dims,np.ones(2),frequencies,hbar) for k in ('H','PE','J1','J3')]
    elif method=='dvr':
        dvr=modules['dvr']; axes=[dvr.fourier_axis(n,box,hbar) for n,box in zip(dims,half_box)]
        matrices=[dvr.dvr_matrix(pol[k],[a[0] for a in axes],[a[1] for a in axes]) for k in ('H','PE','J1','J3')]
    else: raise ValueError('method must be ho or dvr')
    H,P,J1,J3=matrices
    E,V=np.linalg.eigh(H)
    comm=1j/hbar*(H@P-P@H)
    operators=[V.conj().T@J@V for J in (J1,J3)]
    weight,rho=oho.kubo_weights(E,beta)
    blocks,mu2,_=modules['quantum'].correlation(E,operators,weight,np.asarray(times),hbar)
    comm_eigen=V.conj().T@comm@V
    comm_blocks,comm_mu2,_=modules['quantum'].correlation(E,[comm_eigen,np.zeros_like(comm_eigen)],weight,np.asarray(times),hbar)
    defect=V.conj().T@(J1+J3-comm)@V
    denominator=np.sum(weight*abs(operators[0]+operators[1])**2)
    defect_rel=float(np.sqrt(np.sum(weight*abs(defect)**2)/denominator)) if denominator>0 else 0.
    edge=None
    if method=='dvr':
        density=(abs(V)**2@rho).reshape(dims)
        mask=np.zeros(dims,bool); mask[0,:]=mask[-1,:]=True; mask[:,0]=mask[:,-1]=True
        edge=float(density[mask].sum())
    if method=='ho':
        axes=[range(max(1,n//2)) for n in dims]
    else:
        axes=[range(n//4,n-n//4) for n in dims]
    interior=np.array([i*dims[1]+j for i in axes[0] for j in axes[1]])
    raw_defect=J1+J3-comm
    interior_norm=np.linalg.norm((J1+J3)[np.ix_(interior,interior)])
    return {'energy':E,'blocks':blocks,'mu2':mu2,'times':np.asarray(times),
            'commutator_curve':comm_blocks[:,0,0],'commutator_mu2':comm_mu2[0,0],
            'commutator_full_relative':float(np.linalg.norm(raw_defect)/max(np.linalg.norm(J1+J3),1e-300)),
            'commutator_interior_relative':float(np.linalg.norm(raw_defect[np.ix_(interior,interior)])/max(interior_norm,1e-300)),
            'finite_integral':np.trapezoid(blocks.sum(axis=(1,2)),times),
            'commutator_relative':defect_rel,'edge_probability':edge,
            'hermiticity':max(float(np.max(abs(x-x.conj().T))) for x in matrices)}

