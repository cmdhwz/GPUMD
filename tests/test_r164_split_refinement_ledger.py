import sys
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_split_refinement_ledger as ledger
from r164_split_refinement_ledger import propagate, source_terms


class SplitLedgerTests(unittest.TestCase):
    def setUp(self):
        self.omega=np.array([0.,2.]); self.h=.2
        self.ones=np.ones((2,1,1)); self.zeros=np.zeros_like(self.ones)
    def test_frozen_short_card_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            c=Path(temp)/'coarse.in'; f=Path(temp)/'fine.in'
            def card(path,dt,n): path.write_text(f'time_step {dt}\nrun {n}\n',encoding='utf8')
            card(c,.2,5); card(f,.1,10)
            self.assertEqual(ledger.short_card_pair(c,f),((.2,5),(.1,10)))
            card(c,.1,10); card(f,.05,20)
            self.assertEqual(ledger.short_card_pair(c,f),((.1,10),(.05,20)))
            card(f,.025,40)
            with self.assertRaisesRegex(ValueError,'unsupported frozen'): ledger.short_card_pair(c,f)
            card(f,.05,20); f.write_text(f.read_text()+'run 20\n')
            with self.assertRaisesRegex(ValueError,'unique'): ledger.short_card_pair(c,f)

    def test_execution_case_hashes_optional_but_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cases=[]
            for name in ('coarse','fine'):
                folder=root/name; folder.mkdir(); (folder/'r164_output').mkdir()
                (folder/'run.in').write_text(name); (folder/'stdout.log').write_text('log'+name)
                cases.append({'case':name,'returncode':0,'marker_written':True})
            path=root/'execution.json'
            def write(): path.write_text(json.dumps({'binary':'missing','sha256':'a'*64,'cases':cases}))
            verified=[{'run_in_sha256':ledger.sha(root/name/'run.in'),'stdout_sha256':ledger.sha(root/name/'stdout.log')} for name in ('coarse','fine')]
            write(); args=(path,root/'coarse/r164_output',root/'fine/r164_output',None,verified)
            self.assertEqual(ledger.execution_evidence(*args)['status'],'SIDECAR_NOT_FRAME_RECEIPT')
            cases[1]['run_in_sha256']=ledger.sha(root/'fine/run.in')
            cases[1]['stdout_sha256']=ledger.sha(root/'fine/stdout.log'); write()
            ledger.execution_evidence(*args)
            cases[1]['stdout_sha256']='0'*64; write()
            with self.assertRaisesRegex(ValueError,'input SHA mismatch'): ledger.execution_evidence(*args)

    def test_three_sources_and_zero_mode(self):
        curvature,coupling,slope,direct=source_terms(self.zeros,self.ones,self.zeros,self.omega,self.h)
        np.testing.assert_allclose(coupling,0,atol=1e-15); np.testing.assert_allclose(slope,0,atol=1e-15)
        np.testing.assert_allclose(curvature,direct,atol=1e-15)
        curvature,coupling,slope,direct=source_terms(self.ones,self.ones,self.ones,self.omega,self.h)
        np.testing.assert_allclose(curvature,0,atol=1e-15); np.testing.assert_allclose(slope,0,atol=1e-15)
        self.assertEqual(coupling[0,0,0,0],0); self.assertNotEqual(coupling[0,1,0,0],0)
        np.testing.assert_allclose(coupling,direct,atol=1e-15)
        curvature,coupling,slope,direct=source_terms(-self.ones,self.zeros,self.ones,self.omega,self.h)
        np.testing.assert_allclose(curvature,0,atol=1e-15); np.testing.assert_allclose(coupling,0,atol=1e-15)
        self.assertAlmostEqual(slope[0,0,0,0],-self.h**2/4)
        np.testing.assert_allclose(slope,direct,atol=1e-15)
    def test_frame_receipt_schema_checked(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp); (d/'frame_0.txt').write_text('frame'); (d/'force_images_0.txt').write_text('image')
            (d/'scalar.txt').write_bytes(b'0 0\n')
            frame_sha=ledger.sha(d/'frame_0.txt'); image_sha=ledger.sha(d/'force_images_0.txt')
            receipt={'schema':'unknown','step':'0','frame_sha256':frame_sha,'force_images_sha256':image_sha,
                     'metadata_sha256':'a','source_identity_sha256':'b',
                     'scalar_line_sha256':hashlib.sha256(b'0 0\n').hexdigest()}
            path=d/'checkpoint_0.txt'
            def write(): path.write_text(''.join(f'{k} {v}\n' for k,v in receipt.items()))
            write()
            run={'directory':str(d),'frames':[d/'frame_0.txt'],'images':[d/'force_images_0.txt'],
                 'frame_sha256':[frame_sha],'image_sha256':[image_sha],
                 'inputs':{'metadata.txt':'a','source_identity.txt':'b'},'dt':.1}
            state=(np.zeros((2,1,3)),np.zeros((2,1,3)),np.zeros((2,1,3)),np.zeros(2))
            with patch.object(ledger,'frame',return_value=state):
                with self.assertRaisesRegex(ValueError,'unsupported frame receipt schema'): ledger.load_frame(run,0)
                receipt['schema']='r164_checkpoint_v1'; write()
                _,record=ledger.load_frame(run,0)
                self.assertEqual(record['frame_provenance'],'frame_image_scalar_line_receipt_bound')

    def test_multistep_accumulation(self):
        z=np.zeros((2,2,1,1)); components=[z.copy(),z.copy(),z.copy()]
        for k in range(3):
            values=source_terms(self.ones*k,self.ones*(k+.2),self.ones*(k+1),self.omega,self.h)
            z=propagate(z,self.omega,self.h)+values[3]
            components=[propagate(x,self.omega,self.h)+v for x,v in zip(components,values[:3])]
        np.testing.assert_allclose(sum(components),z,rtol=1e-12,atol=1e-14)


if __name__=='__main__': unittest.main()
