"""Independent finite-model controls; no material Cartesian current is implied."""
import numpy as np


def derivatives(coefficients, z):
    c=np.asarray(coefficients['c'],float); l=np.asarray(coefficients['lambda'],float)
    z=np.asarray(z,float); k=float(coefficients['K'])
    if c.shape != (5,) or l.shape != (3,) or z.shape != (2,) or not np.isfinite(np.r_[c,l,z,k]).all():
        raise ValueError('invalid U9 parameters or coordinates')
    q,Q=z
    U=sum(c[i]*q**i for i in range(5))+.5*k*Q**2+(l[0]+l[1]*q+l[2]*q*q)*Q
    g=np.array([sum(i*c[i]*q**(i-1) for i in range(1,5))+(l[1]+2*l[2]*q)*Q,k*Q+l[0]+l[1]*q+l[2]*q*q])
    H=np.array([[2*c[2]+6*c[3]*q+12*c[4]*q*q+2*l[2]*Q,l[1]+2*l[2]*q],[l[1]+2*l[2]*q,k]])
    third=np.zeros((2,2,2)); third[0,0,0]=6*c[3]+24*c[4]*q
    third[0,0,1]=third[0,1,0]=third[1,0,0]=2*l[2]
    fourth=np.zeros((2,2,2,2)); fourth[0,0,0,0]=24*c[4]
    return U,g,H,third,fourth


def normal_modes(G,K):
    G=np.asarray(G,float); K=np.asarray(K,float)
    if G.shape != (2,2) or K.shape != (2,2) or not np.isfinite(G).all() or not np.isfinite(K).all() or not np.allclose(G,G.T,rtol=1e-12,atol=1e-12) or not np.allclose(K,K.T,rtol=1e-12,atol=1e-12):
        raise ValueError('G and K must be finite symmetric 2x2 matrices')
    g,V=np.linalg.eigh(G)
    if g.min() <= max(g.max(),1.)*1e-12: raise ValueError('G is not safely positive definite')
    W=(V / np.sqrt(g))@V.T
    w2,E=np.linalg.eigh(W@K@W)
    if w2.min() <= max(abs(w2).max(),1.)*1e-12: raise ValueError('reference is not stable')
    return W@E,np.sqrt(w2)


def map_transport(H,G,K,beta,hbar,oho):
    """Map a homogeneous row-displacement/column-velocity matrix."""
    H=np.asarray(H,float); G=np.asarray(G,float)
    if H.shape!=(2,2) or not np.isfinite(H).all() or not np.isfinite([beta,hbar]).all() or min(beta,hbar)<=0:
        raise ValueError('invalid transport or temperature')
    T,omega=normal_modes(G,K); C=np.linalg.inv(G); coefficients=H@C
    poly=oho.add(*(oho.scale(oho.mul(oho.VARS[i],oho.VARS[j+2]),coefficients[i,j]) for i in range(2) for j in range(2)))
    momentum=np.linalg.inv(T).T
    forward=[oho.add(*(oho.scale(oho.VARS[j],T[i,j]) for j in range(2))) for i in range(2)]
    forward += [oho.add(*(oho.scale(oho.VARS[j+2],momentum[i,j]) for j in range(2))) for i in range(2)]
    invT=np.linalg.inv(T)
    backward=[oho.add(*(oho.scale(oho.VARS[j],invT[i,j]) for j in range(2))) for i in range(2)]
    backward += [oho.add(*(oho.scale(oho.VARS[j+2],T.T[i,j]) for j in range(2))) for i in range(2)]
    mapped=oho.substitute(oho.harmonic_map(oho.substitute(poly,forward),omega,beta,hbar),backward)
    B=np.array([[mapped.get(tuple(int(k==i)+int(k==j+2) for k in range(4)),0.) for j in range(2)] for i in range(2)])
    return B@G


def reference(model, center, beta, oho):
    """Return distinct legacy-site and moving-flow frozen reference candidates."""
    center=np.asarray(center,float); G=model['geometry']['G']; C=model['geometry']['C']
    at=np.r_[center,0.,0.]
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(beta) or beta <= 0:
        raise ValueError('invalid reference center or beta')
    grad=np.array([oho.evaluate(oho.deriv(model['U'],i),at) for i in range(2)],float)
    K=np.array([[oho.evaluate(oho.deriv(oho.deriv(model['U'],i),j),at) for j in range(2)] for i in range(2)],float)
    scales=[]
    for i in range(2):
        terms=oho.deriv(model['U'],i)
        scales.append(max(1.,sum(abs(float(coef)*np.prod(at**np.array(key))) for key,coef in terms.items())))
    if not np.isfinite(grad).all() or not np.isfinite(scales).all() or np.any(abs(grad)>1e-8*np.asarray(scales)): raise ValueError('reference is not stationary')
    T,omega=normal_modes(G,K)
    ti=[oho.add(h,oho.scale(u,-1)) for h,u in zip(model['hsite'],model['site_potential'])]
    ui=[oho.taylor(u,center) for u in model['site_potential']]
    hi=[oho.add(t,u) for t,u in zip(ti,ui)]
    H=oho.add(*hi); r=model['position_polynomials']
    fixed=oho.add(*(oho.scale(h,float(oho.evaluate(x,at))) for x,h in zip(r,hi)))
    S=oho.bracket(fixed,H)
    J=oho.bracket(oho.add(*(oho.mul(x,h) for x,h in zip(r,hi))),H)
    shifted=[oho.add(oho.VARS[i],oho.constant(center[i])) for i in range(2)]+oho.VARS[2:]
    def bilinear(poly):
        pp=oho.substitute(poly,shifted)
        B=np.array([[pp.get(tuple(int(k==i)+int(k==j+2) for k in range(4)),0.) for j in range(2)] for i in range(2)])
        a=np.array([pp.get(tuple(int(k==j+2) for k in range(4)),0.) for j in range(2)])
        return B@G,a@G
    def bilinear_poly(B):
        return oho.add(*(oho.scale(oho.mul(oho.add(oho.VARS[i],oho.constant(-center[i])),oho.VARS[j+2]),(B@C)[i,j]) for i in range(2) for j in range(2)))
    legacy_H,legacy_affine=bilinear(S); moving_H,_=bilinear(J)
    momentum=np.linalg.inv(T).T
    forward=[oho.add(oho.constant(center[i]),*(oho.scale(oho.VARS[j],T[i,j]) for j in range(2))) for i in range(2)]
    forward += [oho.add(*(oho.scale(oho.VARS[j+2],momentum[i,j]) for j in range(2))) for i in range(2)]
    invT=np.linalg.inv(T)
    backward=[oho.add(*(oho.scale(oho.add(oho.VARS[j],oho.constant(-center[j])),invT[i,j]) for j in range(2))) for i in range(2)]
    backward += [oho.add(*(oho.scale(oho.VARS[j+2],T.T[i,j]) for j in range(2))) for i in range(2)]
    def mapped(poly):
        return oho.substitute(oho.harmonic_map(oho.substitute(poly,forward),omega,beta,model['hbar']),backward)
    delta_H=map_transport(moving_H,G,K,beta,model['hbar'],oho)-moving_H
    return {'K':K,'omega':omega,'T':T,'legacy_H':legacy_H,'legacy_affine':legacy_affine,'moving_H':moving_H,'moving_delta_H':delta_H,'legacy_JA':oho.add(model['J'],mapped(S),oho.scale(S,-1)),'moving_JA':oho.add(model['J'],bilinear_poly(delta_H))}
