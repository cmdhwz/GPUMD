import unittest, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'tools/u9_oho'))
class RPMDControls(unittest.TestCase):
    def test_old_diagonal_step_and_second_order(self):
        import u9_rpmd as new, spectrum
        modules=spectrum.load_modules(Path(os.environ['U9_THEORY_ROOT']))
        import rpmd as old
        case={'M':16.,'m':1.,'a':1.,'b':1.,'K_lat':4.,'lambda':.3,'delta':0.,'d0':4.,'beta':2.,'hbar':.8}
        mod=modules['oho'].model(case)
        c={'c':[1.,0.,-2.,0.,1.],'K':4.,'lambda':[-.3,0.,.3]}
        W=np.diag(1/np.sqrt(mod['masses'])); rng=np.random.default_rng(3)
        x=rng.normal(size=(2,8,2)); p=rng.normal(size=x.shape); z=x@W.T; physical=p@np.linalg.inv(W)
        dt=.001; z1,p1=old.step(mod,z,physical,dt,old.spring_coefficients(mod,8,dt)); x1,pi1=new.step(x,p,dt,W,c,2.,.8)
        np.testing.assert_allclose(z1,x1@W.T,atol=1e-12)
        np.testing.assert_allclose(p1,pi1@np.linalg.inv(W),atol=1e-12)
        def integrate(step,n):
            xx,pp=x.copy(),p.copy()
            for _ in range(n): xx,pp=new.step(xx,pp,step,W,c,2.,.8)
            return xx,pp
        coarse=integrate(.04,2); fine=integrate(.02,4); reference=integrate(.005,16)
        e1=sum(np.linalg.norm(a-b) for a,b in zip(coarse,reference)); e2=sum(np.linalg.norm(a-b) for a,b in zip(fine,reference))
        self.assertGreater(e1/e2,2.5)
    def test_full_g_reversibility(self):
        import u9_rpmd as rpmd
        W=np.array([[1.,.1],[.1,.8]]); c={'c':[1.,0.,-2.,0.,1.],'K':4.,'lambda':[0.,0.,.2]}
        rng=np.random.default_rng(4); x=rng.normal(size=(3,8,2)); p=rng.normal(size=x.shape)
        a,b=rpmd.step(x,p,.001,W,c,2.,1.)
        aa,bb=rpmd.step(a,-b,.001,W,c,2.,1.)
        np.testing.assert_allclose(aa,x,atol=2e-13); np.testing.assert_allclose(bb,-p,atol=2e-13)

if __name__=='__main__': unittest.main()

