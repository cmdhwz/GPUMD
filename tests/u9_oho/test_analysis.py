import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/u9_oho'))
import analyze

class AnalysisControls(unittest.TestCase):
    def test_ordered_lag_and_identical_bootstrap(self):
        import u9_rpmd
        flow=np.array([[[1.,0.,2.,0.],[3.,0.,5.,0.],[7.,0.,11.,0.]]])
        lag=u9_rpmd.correlations(flow)
        self.assertEqual(lag[0,1,0,2],(1*5+3*11)/2)
        self.assertEqual(lag[0,1,2,0],(2*3+5*7)/2)
        self.assertNotEqual(lag[0,1,0,2],lag[0,1,2,0])
        curve=np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]])
        counts=np.array([[0,1,2],[2,2,0],[1,0,1]])
        difference,interval,se=analyze.paired_bootstrap(curve,curve,np.zeros(3),counts)
        self.assertEqual(difference,0.); np.testing.assert_array_equal(interval,[0.,0.]); self.assertEqual(se,0.)
    def test_sector_and_correction_closure(self):
        rng=np.random.default_rng(128)
        signals=rng.normal(size=(4,6,4))
        matrices=np.einsum('gti,gtj->gtij',signals,signals)
        times=np.arange(6)*.1
        target=np.zeros((6,2,2))
        for delta in (None,2,3):
            out=analyze.summarize(matrices,target,times,delta)
            np.testing.assert_allclose(out['sectors'].sum(axis=-1),out['correction_blocks'].sum(axis=-1))
            self.assertLess(out['closure_max'],1e-12)

if __name__=='__main__': unittest.main()
