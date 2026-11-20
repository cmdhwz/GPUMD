import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_audit_force_split as audit

class ForceAuditTests(unittest.TestCase):
    def test_empty_and_truncated_edges(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'edges'; p.write_bytes(b'')
            self.assertEqual(len(audit.read_edges(p,np.zeros((2,2),int))),0)
            p.write_bytes(b'\0'*75)
            with self.assertRaisesRegex(ValueError,'edge length'): audit.read_edges(p,np.array([[1,0]]))
    def test_bad_receipt_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'metadata.txt').write_text('bad')
            (p/'pppm_trace_step_0_receipt.txt').write_text('schema r164_pppm_trace_all_v1\nstep 0\nbead 0\natom all\nforce_evaluation_id 1\nneighbor_history all_centers_selected_bead_endpoint\nphysical_event unavailable\nmetadata_sha256 '+ '0'*64 +'\n')
            with self.assertRaisesRegex(ValueError,'receipt hash mismatch'): audit.audit(p,0)
    def test_raw_charge_1024_lane_reduction(self):
        raw=np.array([((i%13)-6)*0.0037 for i in range(2050)],dtype=np.float32)
        lanes=[np.float32(0) for _ in range(1024)]
        for i,value in enumerate(raw):
            lane=i%1024
            lanes[lane]=np.float32(lanes[lane]+value)
        for offset in (512,256,128,64,32,16,8,4,2,1):
            for i in range(offset):
                lanes[i]=np.float32(lanes[i]+lanes[i+offset])
        mean=np.float32(lanes[0]/np.float32(len(raw)))
        expected=np.array([np.float32(value-mean) for value in raw])
        np.testing.assert_array_equal(audit.project_raw_charge(raw),expected)

    def test_signed_max_and_power(self):
        result=audit.metrics(np.array([[1.,-3.,2.],[0.,0.,0.]]),np.ones((2,3)))
        self.assertEqual((result['atom'],result['axis'],result['signed_value']),(0,'y',-3.))
        self.assertEqual(result['power_eV_per_fs'],0.)
if __name__=='__main__': unittest.main()
