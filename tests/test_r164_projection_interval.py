import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from r164_projection_interval import integrate_interval, read_drift_charge


class ProjectionIntervalTest(unittest.TestCase):
    def test_drift_attachment_binding_and_side(self):
        header=('# schema r164_observer_drift_charge_v1\n# step 0\n# time_fs 0\n'
                '# observer_force_call 1\n# N 2\n# time_unit_conversion 10.18051\n'
                '# arrival_status absent_boundary\n# departure_status valid\n'
                '# columns side atom raw_charge_rate_fp32_e_per_natural_time\n'
                '# observer_projection_sha256 abc\n')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'drift.txt'
            path.write_text(header+'departure 0 0.5\ndeparture 1 -0.25\n',encoding='utf8')
            data=read_drift_charge(path,0,2,2,0.,'abc')
            np.testing.assert_array_equal(data['rates']['departure'],[.5,-.25])
            with self.assertRaises(ValueError): read_drift_charge(path,0,2,2,0.,'tampered')
            path.write_text(header+'arrival 0 0.5\ndeparture 1 -0.25\n',encoding='utf8')
            with self.assertRaises(ValueError): read_drift_charge(path,0,2,2,0.,'abc')
            self.assertIsNone(read_drift_charge(Path(tmp)/'missing',0,2,2,0.,'abc'))
            mid=header.replace('# step 0', '# step 1').replace('# time_fs 0', '# time_fs 0.1').replace('# observer_force_call 1', '# observer_force_call 2').replace('# arrival_status absent_boundary', '# arrival_status valid')
            path.write_text(mid+'arrival 0 0.125\narrival 1 0.25\ndeparture 0 0.5\ndeparture 1 -0.25\n',encoding='utf8')
            data=read_drift_charge(path,1,2,2,0.1,'abc')
            np.testing.assert_array_equal(data['rates']['arrival'],[.125,.25])
            np.testing.assert_array_equal(data['rates']['departure'],[.5,-.25])

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
