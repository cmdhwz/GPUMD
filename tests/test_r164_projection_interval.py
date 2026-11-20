import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from r164_projection_interval import integrate_interval


class ProjectionIntervalTest(unittest.TestCase):
    def test_crossing_and_chord(self):
        x=np.array([[1.,1.,1.],[5.9,2.,2.],[3.,3.,3.]])
        y=x.copy(); y[1,0]+=0.4
        c=np.array([1.,-1.,0.])
        result=integrate_interval(x,y,x,y,c,c,np.array([10.,10.,10.]),1.)
        self.assertEqual(result['status'],'valid')
        self.assertEqual(result['root_count'],1)
        self.assertAlmostEqual(result['roots'][0]['fraction'],0.25)
        self.assertAlmostEqual(result['root_correction_eV_A'][0],10/3*0.25*2)
        self.assertLess(result['eta_net_jump_max_residual_A'],1e-12)

    def test_crossing_with_changing_coefficient(self):
        x=np.array([[1.,1.,1.],[5.9,2.,2.],[3.,3.,3.]])
        y=x.copy(); y[1,0]+=.4
        result=integrate_interval(x,y,x,y,np.array([1.,-1.,0.]),
                                  np.array([2.,-2.,0.]),np.full(3,10.),1.)
        expected=10/3*((.5-.25)*2+(1/6-.25**2/2)*2)
        self.assertAlmostEqual(result['root_correction_eV_A'][0],expected)

    def test_boundary_unavailable(self):
        x=np.array([[1.,1.,1.],[6.,2.,2.]])
        y=x.copy(); y[1,0]+=.1
        result=integrate_interval(x,y,x,y,np.ones(2),np.ones(2),np.full(3,10.),1.)
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(result['linear_c_model_integral_eV_A'])

    def test_net_jump_mismatch_is_unavailable(self):
        x=np.array([[1.,1.,1.],[2.,2.,2.]])
        y=x.copy(); y[1,0]+=.1
        with patch('r164_projection_interval.eta',side_effect=[np.zeros((2,3)),np.ones((2,3))]):
            result=integrate_interval(x,y,x,y,np.ones(2),np.ones(2),np.full(3,10.),1.)
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(result['linear_c_model_integral_eV_A'])

    def test_no_crossing_linear_product(self):
        x=np.array([[1.,1.,1.],[2.,2.,2.]])
        y=x.copy(); y[1,0]+=.1
        result=integrate_interval(x,y,x,y,np.array([1.,-1.]),np.array([2.,-2.]),np.full(3,10.),1.)
        self.assertEqual(result['root_count'],0)
        self.assertAlmostEqual(result['linear_c_model_integral_eV_A'][0],-19/12)

if __name__=='__main__': unittest.main()
