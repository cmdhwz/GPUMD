import unittest, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/u9_oho'))
class SpectralControls(unittest.TestCase):
    def test_invalid_mass_and_rotation_rejected(self):
        import spectrum
        data=spectrum.load_modules(Path(os.environ['U9_THEORY_ROOT']))
        model=data['embedding'].build_polynomials({'c':[0.,0.,1.,0.,0.],'K':3.,'lambda':[0.,0.,0.]},{'ML':16.,'m':1.,'MR':16.},[16.,1.,16.],4.,.8)
        with self.assertRaises(ValueError): spectrum.whitened_polynomials(model,data['oho'],np.eye(2)*1.000001)
        invalid=dict(model); invalid['geometry']=dict(model['geometry']); invalid['geometry']['G']=np.array([[np.nan,0.],[0.,1.]])
        with self.assertRaises(ValueError): spectrum.whitened_polynomials(invalid,data['oho'])
    def test_padded_interior_exact_instantaneous_weyl_current(self):
        import spectrum
        data=spectrum.load_modules(Path(os.environ['U9_THEORY_ROOT'])); oho=data['oho']
        c={'c':[1.,.1,-2.,.05,1.], 'K':4., 'lambda':[-.2,.1,.2]}
        model=data['embedding'].build_polynomials(c,{'ML':16.,'m':1.,'MR':17.},[16.,2.,17.],4.,.8)
        polys=spectrum.whitened_polynomials(model,oho); dims=(14,14)
        H,P,J1,J3=[oho.weyl_matrix(polys[k],dims,np.ones(2),np.ones(2),.8) for k in ('H','PE','J1','J3')]
        interior=np.array([i*dims[1]+j for i in range(3) for j in range(3)])
        defect=J1+J3-1j/.8*(H@P-P@H)
        np.testing.assert_allclose(defect[np.ix_(interior,interior)],0.,atol=1e-9)
    def test_old_diagonal_weyl_basis_matches(self):
        import spectrum
        data=spectrum.load_modules(Path(os.environ['U9_THEORY_ROOT']))
        oho,ce=data['oho'],data['embedding']
        case={'M':16.,'m':1.,'a':1.,'b':1.,'K_lat':4.,'lambda':.3,'delta':.1,'d0':4.,'beta':2.,'hbar':.8}
        old=oho.model(case)
        c={'c':[1.,.1,-2.,0.,1.],'K':4.,'lambda':[-.3,0.,.3]}
        model=ce.build_polynomials(c,{'ML':16.,'m':1.,'MR':16.},[16.,1.,16.],4.,.8)
        transformed=spectrum.whitened_polynomials(model,oho)
        for new,legacy in [(transformed['H'],old['H']),(transformed['J1'],old['sectors'][0]),(transformed['J3'],old['sectors'][1])]:
            a=oho.weyl_matrix(new,(6,6),np.ones(2),[1.5,.7],.8)
            b=oho.weyl_matrix(legacy,(6,6),old['masses'],[1.5,.7],.8)
            np.testing.assert_allclose(a,b,atol=1e-10)
    def test_full_g_harmonic_spectrum_and_kubo(self):
        import spectrum
        theory=Path(os.environ['U9_THEORY_ROOT'])
        data=spectrum.load_modules(theory)
        oho,ce=data['oho'],data['embedding']
        c={'c':[0,0,1,0,0], 'K':3., 'lambda':[0,0,0]}
        model=ce.build_polynomials(c,{'ML':16.,'m':1.,'MR':17.},[16.,2.,17.],2.,1.)
        times=np.linspace(0,.2,3)
        ho=spectrum.solve(model,2.,times,(9,9),'ho',data)
        dvr=spectrum.solve(model,2.,times,(15,15),'dvr',data,half_box=8.)
        self.assertLess(abs(ho['energy'][0]-dvr['energy'][0]),.08)
        from scipy.linalg import eigh
        omega=np.sqrt(eigh(np.diag([2.,3.]),model['geometry']['G'],eigvals_only=True))
        self.assertAlmostEqual(ho['energy'][0],.5*np.sum(omega),places=3)
        self.assertLess(ho['hermiticity'],1e-9)
        np.testing.assert_allclose(ho['blocks'],ho['blocks'].transpose(0,2,1),atol=1e-8)
        self.assertTrue(np.isfinite(dvr['mu2']).all())

    def test_primary_blocks_are_direct_weyl_sectors(self):
        import spectrum
        data=spectrum.load_modules(Path(os.environ['U9_THEORY_ROOT']))
        oho,ce=data['oho'],data['embedding']
        c={'c':[1.,.1,-2.,0.,1.], 'K':4., 'lambda':[-.2,0.,.2]}
        model=ce.build_polynomials(c,{'ML':16.,'m':1.,'MR':16.},[16.,2.,17.],4.,.8)
        dims=(6,6); times=np.array([0.,.05,.1]); beta=2.
        result=spectrum.solve(model,beta,times,dims,'ho',data)
        poly=spectrum.whitened_polynomials(model,oho)
        H,J1,J3=[oho.weyl_matrix(poly[k],dims,np.ones(2),np.ones(2),.8) for k in ('H','J1','J3')]
        E,V=np.linalg.eigh(H); weights,_=oho.kubo_weights(E,beta)
        direct=[V.conj().T@J@V for J in (J1,J3)]
        gap=(E[:,None]-E[None,:])/.8
        expected=np.empty_like(result['blocks'])
        for t,time in enumerate(times):
            for a in range(2):
                for b in range(2): expected[t,a,b]=np.sum((weights*direct[a]*direct[b].T).real*np.cos(time*gap))
        np.testing.assert_allclose(result['blocks'],expected,atol=1e-10)
        np.testing.assert_allclose(result['blocks'][0],expected[0],atol=1e-10)
        expected_mu2=np.array([[np.sum((weights*direct[a]*direct[b].T).real*gap**2) for b in range(2)] for a in range(2)])
        np.testing.assert_allclose(result['mu2'],expected_mu2,atol=1e-10)

if __name__=='__main__': unittest.main()
