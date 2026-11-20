import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_checkpoint_chain as chain
import r164_provisional_correlations as corr


def fixture(missing=None):
    segments=[]; records=[]; absolute=0
    for index,values in enumerate(([0.,1.],[10.,20.,30.])):
        rows=[]
        for step,value in enumerate(values):
            base=[step,step*.1]+[0]*19
            suffix=['NUMBER_INCREMENT_INITIAL_BOUNDARY'] if step==0 else ['NUMBER_INCREMENT_LIFT_A']+[str(value)]*12
            if missing==(index,step): suffix=[]
            fields=base+['B_UNVERIFIED',value,0,0,'R27_UNVERIFIED',0,0,0,0,0,0,0,'J2_UNVERIFIED',value,0,0]+suffix
            rows.append((' '.join(map(str,fields)).encode()+b'\n',{}))
        meta={'reference_id':'R27_fixed_2OH_R157','reference_source_sha256':'1d46365935f48e390d08e69e8564a3adc692d439b9472b4250914b22144a2eaa','number_increment':'lift_angstrom O_H_Na_Cl_xyz right_closed_initial_boundary_no_increment'}
        segments.append({'capture':Path(f'capture{index}'),'source':{'bead_force_backend':'native_GPUMD'},'meta':meta,'dt':.1,'temp':300.,'cell':np.array([10.,10.,10.]),'observer':True,'observer_present':True,'rows':rows})
        end=len(values)-1
        records.append({'end_step':end,'absolute_step_start':absolute,'checkpoint_sha256':'a'*64,'metadata_sha256':'b'*64,'source_identity_sha256':'c'*64})
        absolute+=end
    return {'segments':records,'rows':[],'links':[]},segments


class CheckpointTransportTests(unittest.TestCase):
    def test_seam_owned_bins_and_cross_seam_window(self):
        with patch.object(chain,'inspect',side_effect=lambda _:fixture()):
            out=corr.analyze(['one','two'],2,'lift',True)
        self.assertEqual(out['sample_counts'],[4,3,2])
        self.assertEqual(out['eh_regular_only']['origin_count'],2)
        expected=np.array([.5,15.,25.])
        self.assertAlmostEqual(out['eh_regular_only']['interval_mean'][0],float(expected.mean()))
        self.assertEqual(out['checkpoint_chain']['segments'][1]['point_local_range'],[1,2])
        self.assertEqual(out['checkpoint_chain']['segments'][1]['seam_absolute_step'],1)
        self.assertAlmostEqual(out['mean_shared_all_samples'][0],(0+1+20+30)/4)
        self.assertAlmostEqual(out['eh_regular_only']['interval_mean'][6],(10+200+300)/3)

    def test_bad_boundary_and_missing_increment_rejected(self):
        for bad in ((1,0),(1,1)):
            with patch.object(chain,'inspect',side_effect=lambda _,bad=bad:fixture(bad)):
                with self.assertRaises(ValueError): corr.analyze(['one','two'],1,'lift',True)


if __name__=='__main__': unittest.main()
