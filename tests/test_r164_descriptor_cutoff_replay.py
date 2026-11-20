import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from r164_descriptor_cutoff_replay import scan_pair

class CutoffScanTests(unittest.TestCase):
    def test_crossing(self):
        def state(t): return np.array([.8+.4*t,0,0]),np.array([.4,0,0])
        roots,unresolved=scan_pair(state,0,.4,1.2,1.,1.,1e-10)
        self.assertEqual(len(roots),1); self.assertEqual(roots[0]['direction'],1)
        self.assertAlmostEqual(roots[0]['relative_time_fs'],.5,places=10); self.assertFalse(unresolved)
    def test_same_side_two_roots(self):
        def state(t): return np.array([1+(t-.27)*(t-.73),0,0]),np.array([2*t-1,0,0])
        roots,unresolved=scan_pair(state,2,1,2,1.,1.,1e-10)
        self.assertEqual([x['direction'] for x in roots],[-1,1])
        self.assertAlmostEqual(roots[0]['relative_time_fs'],.27,places=9)
        self.assertAlmostEqual(roots[1]['relative_time_fs'],.73,places=9)
    def test_tangent_and_endpoint_are_unresolved(self):
        def tangent(t): return np.array([1+(t-.5)**2,0,0]),np.array([2*(t-.5),0,0])
        roots,unknown=scan_pair(tangent,2,1,2,1.,1.,1e-10)
        self.assertFalse(roots); self.assertTrue(unknown)
        def dyadic(t): return np.array([1+(t-.25)*(t-.75),0,0]),np.array([2*t-1,0,0])
        roots,unknown=scan_pair(dyadic,2,1,2,1.,1.,1e-10)
        self.assertTrue(unknown)
        def endpoint(t): return np.array([1+t,0,0]),np.array([1.,0,0])
        roots,unknown=scan_pair(endpoint,0,1,2,1.,1.,1e-10)
        self.assertFalse(roots); self.assertTrue(unknown)

if __name__=='__main__': unittest.main()
