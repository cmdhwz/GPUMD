import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from r164_assignment_join_replay import fused_weights, compare_states, charge_feedback, adjacent_coordinates

class JoinTest(unittest.TestCase):
    def test_weight_reflection_at_face(self):
        np.testing.assert_array_equal(fused_weights(.5),fused_weights(-.5)[::-1])

    def test_zero_center_float_tie(self):
        lower,upper=adjacent_coordinates(0)
        self.assertEqual(int(np.float32(lower+np.float32(.5))),0)
        self.assertEqual(int(np.float32(upper+np.float32(.5))),1)
        self.assertEqual(np.nextafter(lower,np.float32(np.inf)),upper)
        self.assertLess(float(upper),.5)

    def test_last_center_pair(self):
        lower,upper=adjacent_coordinates(59)
        self.assertEqual(int(np.float32(lower+np.float32(.5))),59)
        self.assertEqual(int(np.float32(upper+np.float32(.5))),60)

    def test_pair_energy_identity(self):
        G=np.array([0.,1.,2.,1.]); rho0=np.array([1.,0.,0.,0.]); rho1=np.array([0.,2.,0.,0.])
        phi0=np.fft.ifft(np.fft.fft(rho0)*G).real*4
        phi1=np.fft.ifft(np.fft.fft(rho1)*G).real*4
        result=compare_states((rho0,phi0,None,None),(rho1,phi1,None,None),G,0.5)
        self.assertAlmostEqual(result['paired_delta_U_eV'],6.)
        self.assertAlmostEqual(result['energy_identity_residual_eV'],0.)

    def test_directed_edge_force_sign(self):
        edge=np.zeros(1,dtype=[('center',int),('atom',int),('g',float,(3,))])
        edge['center']=0; edge['atom']=1; edge['g'][0]=[1.,2.,-1.]
        np.testing.assert_array_equal(charge_feedback(edge,np.array([2.,0.]),2),[[2.,4.,-2.],[-2.,-4.,2.]])

if __name__=='__main__': unittest.main()
