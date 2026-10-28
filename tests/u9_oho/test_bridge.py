import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools' / 'u9_oho'))
import bridge

class Controls(unittest.TestCase):
    def test_derivatives(self):
        c = {'c':[.2,-.3,.4,.1,.7], 'K':2., 'lambda':[.1,.2,-.3]}
        z=np.array([.37,-.24]); h=1e-5
        f,g,H,t,u=bridge.derivatives(c,z)
        gn=np.array([(bridge.derivatives(c,z+np.eye(2)[i]*h)[0]-bridge.derivatives(c,z-np.eye(2)[i]*h)[0])/(2*h) for i in range(2)])
        Hn=np.column_stack([(bridge.derivatives(c,z+np.eye(2)[i]*h)[1]-bridge.derivatives(c,z-np.eye(2)[i]*h)[1])/(2*h) for i in range(2)])
        tn=(bridge.derivatives(c,z+np.array([h,0]))[2]-bridge.derivatives(c,z-np.array([h,0]))[2])/(2*h)
        np.testing.assert_allclose(g,gn,rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(H,Hn,rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(t[0],tn,rtol=1e-8,atol=1e-9)
        self.assertAlmostEqual(u[0,0,0,0],24*c['c'][4])
    def test_full_metric_canonical_and_unstable(self):
        from scipy.linalg import eigh
        G=np.array([[2.,.4],[.4,1.3]]); K=np.array([[3.,.7],[.7,2.]])
        T,w=bridge.normal_modes(G,K)
        np.testing.assert_allclose(T.T@G@T,np.eye(2),atol=1e-12)
        np.testing.assert_allclose(T.T@K@T,np.diag(w*w),atol=1e-12)
        np.testing.assert_allclose(w*w,eigh(K,G,eigvals_only=True),atol=1e-12)
        x=np.array([.4,-.2]); p=np.array([.5,.9]); dx=T@x; P=np.linalg.solve(T.T,p)
        self.assertAlmostEqual(P@dx,p@x)
        with self.assertRaises(ValueError): bridge.normal_modes(G,np.diag([-1.,2.]))
        with self.assertRaises(ValueError): bridge.normal_modes(np.diag([-1.,2.]),K)


class ReferenceControls(unittest.TestCase):
    def test_degenerate_nonsymmetric_rotation_covariance(self):
        import os
        sys.path.insert(0,str(Path(os.environ['U9_THEORY_ROOT'])/'quantum_benchmark_OHO'/'src'))
        import oho
        H=np.array([[.7,.4],[-.2,1.1]]); angle=.47
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        mapped=bridge.map_transport(H,np.eye(2),2*np.eye(2),2.,.8,oho)
        rotated=bridge.map_transport(R.T@H@R,np.eye(2),2*np.eye(2),2.,.8,oho)
        np.testing.assert_allclose(rotated,R.T@mapped@R,atol=1e-10)
        np.testing.assert_allclose(bridge.map_transport(H,np.eye(2),2*np.eye(2),1e-7,.8,oho),H,atol=1e-10)
    def test_legacy_and_moving_are_distinct_and_old_recovers(self):
        import os, json
        root=Path(os.environ['U9_THEORY_ROOT'])
        sys.path.insert(0,str(root/'quantum_benchmark_OHO'/'src'))
        sys.path.insert(0,str(root/'quantum_material_hamiltonian_v1'/'src'))
        import oho, constrained_embedding as ce
        cases=json.loads((root/'quantum_benchmark_OHO'/'frozen_cases.json').read_text(encoding='utf-8'))
        for case_data in cases['cases']:
            case=cases['common'] | case_data
            old=oho.model(case)
            if case.get('harmonic',False):
                c={'c':[0,0,case['kq']/2,0,0], 'K':case['K_lat'], 'lambda':[0,0,0]}
            else:
                c={'c':[case['a']*case['b']**4,case['delta'],-2*case['a']*case['b']**2,0,case['a']], 'K':case['K_lat'], 'lambda':[-case['lambda']*case['b']**2,0,case['lambda']]}
            masses={'ML':case['M'],'m':case['m'],'MR':case['M']}
            model=ce.build_polynomials(c,masses,[case['M'],case['m'],case['M']],case['d0'],case['hbar'])
            for new,previous in [(model['U'],old['U']),(model['H'],old['H']),
                                 (model['J1'],old['sectors'][0]),(model['J3'],old['sectors'][1]),
                                 *zip(model['hsite'],old['hi'])]:
                for term in set(new)|set(previous):
                    self.assertAlmostEqual(float(new.get(term,0.)),float(previous.get(term,0.)),places=10)
            for state in ([.2,-.1,.3,.4],[-.7,.3,-.2,.5]):
                for new,previous in [(model['U'],old['U']),(model['H'],old['H']),
                                     (model['J1'],old['sectors'][0]),(model['J3'],old['sectors'][1]),
                                     *zip(model['hsite'],old['hi'])]:
                    self.assertAlmostEqual(float(oho.evaluate(new,state)),float(oho.evaluate(previous,state)),places=10)
                for axis in range(2):
                    self.assertAlmostEqual(float(oho.evaluate(oho.deriv(model['U'],axis),state)),float(oho.evaluate(oho.deriv(old['U'],axis),state)),places=10)
            for center in oho.minima(old):
                result=bridge.reference(model,center,case['beta'],oho)
                expected=oho.reference(old,center)
                mapped=bridge.map_transport(result['moving_H'],model['geometry']['G'],result['K'],case['beta'],case['hbar'],oho)
                np.testing.assert_allclose(result['moving_delta_H'],mapped-result['moving_H'],atol=1e-11)
                for state in ([.2,-.1,.3,.4],[-.7,.3,-.2,.5]):
                    self.assertAlmostEqual(float(oho.evaluate(result['legacy_JA'],state)),float(oho.evaluate(expected['JA'],state)),places=10)
                    state=np.asarray(state)
                    correction=float(oho.evaluate(result['moving_JA'],state)-oho.evaluate(model['J'],state))
                    self.assertAlmostEqual(correction,float((state[:2]-center)@result['moving_delta_H']@model['geometry']['C']@state[2:]),places=10)
                if case['delta'] != 0:
                    self.assertTrue(np.linalg.norm(result['moving_H']-result['legacy_H'])>1e-6)
                    self.assertTrue(np.linalg.norm(result['legacy_affine'])>1e-6)

    def test_isotope_full_metric_current_and_classical_limit(self):
        import os
        root=Path(os.environ['U9_THEORY_ROOT'])
        sys.path.insert(0,str(root/'quantum_benchmark_OHO'/'src'))
        sys.path.insert(0,str(root/'quantum_material_hamiltonian_v1'/'src'))
        import oho, constrained_embedding as ce, local_model
        c={'c':[1.,0.,-2.,0.,1.], 'K':4., 'lambda':[-.3,0.,.3]}
        refmass={'ML':16.,'m':1.,'MR':17.}; masses=[16.,2.,17.]
        model=ce.build_polynomials(c,refmass,masses,2.4,1.)
        protium=ce.build_polynomials(c,refmass,[16.,1.,17.],2.4,1.)
        np.testing.assert_allclose(model['geometry']['A0'],protium['geometry']['A0'],atol=0)
        self.assertGreater(np.linalg.norm(model['geometry']['G']-protium['geometry']['G']),.1)
        G=model['geometry']['G']; self.assertGreater(abs(G[0,1]),1e-5)
        z=np.array([.42,-.3]); p=np.array([.7,-.2]); u,du=ce.site_potentials(c,z)
        evaluated=ce.evaluate_point(model['geometry'],z,p,u,du)
        state=np.r_[z,p]
        self.assertAlmostEqual(float(oho.evaluate(model['J'],state)),float(evaluated['J']),places=11)
        self.assertAlmostEqual(float(oho.evaluate(model['U'],state)),float(local_model.potential(c,*z)),places=12)
        np.testing.assert_allclose(bridge.derivatives(c,z)[1],local_model.potential_gradient(c,*z),atol=1e-12)
        r=bridge.reference(model,np.array([1.,0.]),1e-7,oho)
        self.assertLess(np.linalg.norm(r['moving_delta_H']),1e-10)

if __name__ == '__main__': unittest.main()

