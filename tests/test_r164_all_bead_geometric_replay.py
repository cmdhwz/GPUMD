import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_all_bead_geometric_replay as replay

class GeometryTests(unittest.TestCase):
    def test_roundtrip_resolved(self):
        def state(t): return .09-(t-.5)**2, -2*(t-.5)
        face=(np.array([0.]),[(0.,'periodic_chart',0)])
        roots,issues=replay.scan_coordinate(state,2.,1.,face,(0,0,0),0)
        self.assertFalse(issues)
        self.assertEqual([x['sign'] for x in roots],[1,-1])
        np.testing.assert_allclose([x['relative_time_fs'] for x in roots],[.2,.8],atol=1e-12)
    def test_boundary_cushion_is_unresolved(self):
        def state(t): return t-1e-11,1.
        face=(np.array([0.]),[(0.,'assignment_center',0)])
        roots,issues=replay.scan_coordinate(state,0.,.1,face,(0,0,0),0)
        self.assertFalse(roots)
        self.assertTrue(issues)
    def test_mismatched_source_directory(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            with self.assertRaisesRegex(ValueError,'share run directory'):
                replay.inspect_inputs(d/'capture',d/'other'/'run.in',d/'stdout',0,1)
if __name__=='__main__': unittest.main()
