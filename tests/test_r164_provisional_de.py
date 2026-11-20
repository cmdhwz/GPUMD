import importlib.util
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('de',ROOT/'tools/r164_provisional_de.py')
de=importlib.util.module_from_spec(spec); spec.loader.exec_module(de)
class ReplayTests(unittest.TestCase):
    def test_generator_and_split(self):
        c={'c':np.zeros(3),'gamma':np.array([[1.,2.],[2.,1.],[1.,-1.]]),
           'S_Q':np.tile(np.eye(2),(3,1,1)),'S_P':np.tile(np.eye(2),(3,1,1)),
           'T':np.zeros((3,2,2,2)),'V':np.zeros((3,2,2,2))}
        h=0.1; q=np.array([[1.,2.],[1.11,2.22]]); pi=np.array([[1.,2.],[1.2,2.4]]); force=np.array([[2.,4.],[2.,4.]])
        result=de.replay(q,pi,force,np.array([0.,h]),np.ones(2),np.zeros(2),c)
        np.testing.assert_allclose(result['D_identity_residual'],0,atol=1e-14)
        np.testing.assert_allclose(result['Q_drift_residual'],0,atol=1e-14)
        np.testing.assert_allclose(result['last_kick_residual'],0,atol=1e-14)
        np.testing.assert_allclose(result['split_minus_endpoint'],0,atol=1e-14)
if __name__=='__main__': unittest.main()
