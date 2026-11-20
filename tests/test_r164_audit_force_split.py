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
            (p/'pppm_trace_step_0_receipt.txt').write_text('schema r164_pppm_trace_all_v1\nstep 0\nbead 0\natom all\nforce_evaluation_id 1\nphysical_event unavailable\nmetadata_sha256 '+ '0'*64 +'\n')
            with self.assertRaisesRegex(ValueError,'receipt hash mismatch'): audit.audit(p,0)
    def test_signed_max_and_power(self):
        result=audit.metrics(np.array([[1.,-3.,2.],[0.,0.,0.]]),np.ones((2,3)))
        self.assertEqual((result['atom'],result['axis'],result['signed_value']),(0,'y',-3.))
        self.assertEqual(result['power_eV_per_fs'],0.)
if __name__=='__main__': unittest.main()
