"""Finite-P unit-mass whitened RPMD controls; no bulk material interpretation."""
import numpy as np


def potential_gradient(x,W,c):
    z=x@W.T; q=z[...,0]; Q=z[...,1]; a=np.asarray(c['c']); l=np.asarray(c['lambda'])
    U=a[0]+q*(a[1]+q*(a[2]+q*(a[3]+q*a[4])))+.5*c['K']*Q**2+(l[0]+l[1]*q+l[2]*q*q)*Q
    g=np.stack((a[1]+2*a[2]*q+3*a[3]*q*q+4*a[4]*q**3+(l[1]+2*l[2]*q)*Q,c['K']*Q+l[0]+l[1]*q+l[2]*q*q),axis=-1)
    return U,g@W


def step(x,p,dt,W,c,beta,hbar):
    P=x.shape[1]; _,grad=potential_gradient(x,W,c); p=p-.5*dt*grad
    omega=2*P/(beta*hbar)*np.sin(np.pi*np.fft.fftfreq(P))
    X=np.fft.fft(x,axis=1,norm='ortho'); V=np.fft.fft(p,axis=1,norm='ortho')
    cs=np.cos(omega*dt)[None,:,None]; sn=np.empty(P); sn[0]=dt; sn[1:]=np.sin(omega[1:]*dt)/omega[1:]; sn=sn[None,:,None]
    newX=cs*X+sn*V; newV=cs*V-(omega[None,:,None]**2)*sn*X
    x=np.fft.ifft(newX,axis=1,norm='ortho').real; p=np.fft.ifft(newV,axis=1,norm='ortho').real
    _,grad=potential_gradient(x,W,c); return x,p-.5*dt*grad


def energy(x,p,W,c,beta,hbar):
    P=x.shape[1]; U,_=potential_gradient(x,W,c); spring=(P/(beta*hbar))**2
    return (np.sum(.5*p*p,axis=(1,2))+.5*spring*np.sum((x-np.roll(x,1,axis=1))**2,axis=(1,2))+U.sum(axis=1))


def sample(x,W,c,beta,hbar,seed,steps,dt,leapfrog):
    rng=np.random.default_rng(seed); P=x.shape[1]; accepted=0
    for _ in range(steps):
        p=rng.normal(size=x.shape)*np.sqrt(P/beta)
        old=energy(x,p,W,c,beta,hbar); newx=x.copy(); newp=p.copy()
        for _ in range(leapfrog): newx,newp=step(newx,newp,dt,W,c,beta,hbar)
        delta=(energy(newx,newp,W,c,beta,hbar)-old)*beta/P
        ok=np.log(rng.random(x.shape[0])) < -delta
        x[ok]=newx[ok]; accepted+=int(ok.sum())
    return x,accepted/(steps*x.shape[0])


def trajectory(x,W,c,beta,hbar,seed,dt,frames,polynomials,oho,substeps=1):
    rng=np.random.default_rng(seed); P=x.shape[1]; p=rng.normal(size=x.shape)*np.sqrt(P/beta)
    invW=np.linalg.inv(W); curves=[]
    for _ in range(frames):
        z=(x@W.T).mean(axis=1); momentum=(p@invW).mean(axis=1)
        state=np.concatenate((z,momentum),axis=1)
        curves.append(np.stack([oho.evaluate(poly,state).real for poly in polynomials],axis=-1))
        for _ in range(substeps): x,p=step(x,p,dt,W,c,beta,hbar)
    return np.stack(curves,axis=1)


def correlations(curves):
    groups,frames,channels=curves.shape
    out=np.empty((groups,frames,channels,channels))
    for lag in range(frames):
        out[:,lag]=np.einsum('gti,gtj->gij',curves[:,:frames-lag],curves[:,lag:])/(frames-lag)
    return out
