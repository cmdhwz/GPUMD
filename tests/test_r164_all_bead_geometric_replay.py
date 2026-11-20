import sys
import tempfile
import unittest
from unittest.mock import patch
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
    def test_constant_on_face_is_one_unresolved_interval(self):
        def state(t): return 0.,0.
        face=(np.array([0.]),[(0.,'periodic_chart',0)])
        roots,issues=replay.scan_coordinate(state,0.,1.,face,(0,0,0),0)
        self.assertFalse(roots)
        self.assertEqual(len(issues),1)
        self.assertEqual(issues[0]['time_interval_fs'],[0.,1.])

    def test_boundary_cushion_is_unresolved(self):
        def state(t): return t-1e-11,1.
        face=(np.array([0.]),[(0.,'assignment_center',0)])
        roots,issues=replay.scan_coordinate(state,0.,.1,face,(0,0,0),0)
        self.assertFalse(roots)
        self.assertTrue(issues)
    def test_receipt_checks_frame_digest(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            for name in ('metadata.txt','source_identity.txt'):
                (d/name).write_text(name)
            (d/'frame_0.txt').write_text('bead 0 0\n0 0 0 0 0 0 0 0 0 0\nbead 1 0\n0 0 0 0 0 0 0 0 0 0\n')
            (d/'force_images_0.txt').write_text('0 0 0 0 0\n1 0 0 0 0\n')
            info={'capture':d,'P':2,'N':1,'cell':np.array([10.,10.,10.]),'mass':np.array([1.]),
                  'species':np.array(['O']),'temp':300.,'records':[{}],
                  'identity_hashes':{n:replay.sha(d/n) for n in ('metadata.txt','source_identity.txt')}}
            fields={'schema':'r164_checkpoint_v1','step':'0','metadata_sha256':info['identity_hashes']['metadata.txt'],
                    'source_identity_sha256':info['identity_hashes']['source_identity.txt'],
                    'frame_sha256':'0'*64,'force_images_sha256':replay.sha(d/'force_images_0.txt')}
            (d/'checkpoint_0.txt').write_text(''.join(k+' '+v+'\n' for k,v in fields.items()))
            with patch.object(replay,'validate_frame_scalars'):
                with self.assertRaisesRegex(ValueError,'frame receipt digest mismatch'):
                    replay.read_endpoint(info,0)
                fields['frame_sha256']=replay.sha(d/'frame_0.txt')
                (d/'checkpoint_0.txt').write_text(''.join(k+' '+v+'\n' for k,v in fields.items()))
                self.assertEqual(replay.read_endpoint(info,0)['identity_scope'],'receipt_verified')

    def test_mismatched_source_directory(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            with self.assertRaisesRegex(ValueError,'share run directory'):
                replay.inspect_inputs(d/'capture',d/'other'/'run.in',d/'stdout',0,1)
if __name__=='__main__': unittest.main()
