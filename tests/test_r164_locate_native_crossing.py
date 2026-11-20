import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('cross',ROOT/'tools/r164_locate_native_crossing.py'); cross=importlib.util.module_from_spec(spec); spec.loader.exec_module(cross)

class CrossingTests(unittest.TestCase):
    def state(self,q,v):
        T=cross.normal_matrix(2); positions=np.zeros((2,1,3)); velocities=np.zeros_like(positions)
        positions[:,:,0]=q; velocities[:,:,0]=v
        Q=np.einsum('sk,sia->kia',T,positions); V=np.einsum('sk,sia->kia',T,velocities)
        return Q,V,np.array([0.,1.]),T
    def test_both_directions_and_fp32_boundary(self):
        for q,v,direction in [(4.4,1.,1),(4.6,-1.,-1)]:
            Q,V,w,T=self.state(q,v)
            result=cross.locate_one(Q,V,w,T,0,0,0,.2,10.,10,0,0,q,q+.2*v)
            self.assertEqual(result['direction'],direction)
            self.assertGreater(result['velocity_lower_bound_A_per_fs'],0)
            self.assertEqual(result['samples'][0]['lifted_center'],4 if direction==1 else 5)
            self.assertEqual(result['samples'][2]['lifted_center'],5 if direction==1 else 4)
        threshold,lo,hi=cross.fp32_boundary(58)
        self.assertNotEqual(threshold,58.5); self.assertEqual(cross.center(np.float32(lo)),58); self.assertEqual(cross.center(np.float32(hi)),59)
    def test_reject_bad_endpoint_nonmonotonic_and_multigrid(self):
        a=np.zeros((2,1,3)); cross.validate_endpoint(a,a,a,a)
        with self.assertRaises(ValueError): cross.validate_endpoint(a,a,a+1e-7,a)
        Q,V,w,T=self.state(4.,20.)
        with self.assertRaises(ValueError): cross.locate_one(Q,V,w,T,0,0,0,.2,10.,10,0,0,4.,8.)
        Q,V,w,T=self.state(4.4,1.)
        # The endpoints cross one center, but the global acceleration bound is too large.
        w[1]=5.; Q[1,0,0]=1.; Q[0,0,0]=(4.4-T[0,1]*Q[1,0,0])/T[0,0]
        with self.assertRaisesRegex(ValueError,'monotonicity'): cross.locate_one(Q,V,w,T,0,0,0,.2,10.,10,0,0,4.4,4.6)
    def test_endpoint_labels_use_recorded_coordinates(self):
        threshold,_,_=cross.fp32_boundary(4)
        Q,V,w,T=self.state(threshold-1e-9,1.)
        with self.assertRaisesRegex(ValueError, 'recorded and replay endpoint centers differ'):
            cross.locate_one(Q,V,w,T,0,0,0,.2,10.,10,0,0,threshold+1e-9,threshold+.2)

    def test_stdout_requires_exact_card_grid_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'stdout'
            facts='Number of atoms is 5896.\nnumber of beads is 32.\nTime step for this run is 0.1 fs.\nFinished running GPUMD.\n'
            grid='PPPM mesh: 60 x 60 x 60 (target spacing 0.5 A;\n'
            log.write_text(facts+'Run 10 steps.\n'+grid*2)
            self.assertEqual(cross.stdout_mesh(log,.1,10),[60,60,60])
            log.write_text(facts+'Run 9 steps.\n'+grid)
            with self.assertRaises(ValueError): cross.stdout_mesh(log,.1,10)
            log.write_text(facts+'Run 10 steps.\n'+grid.replace('60 x 60 x 60','59 x 60 x 60'))
            with self.assertRaises(ValueError): cross.stdout_mesh(log,.1,10)
    def test_capture_path_must_match_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); lines=[['r164_rpmd','r164_output','1']]
            cross.validate_capture_path(lines,d,d/'r164_output')
            with self.assertRaisesRegex(ValueError,'capture path mismatch'):
                cross.validate_capture_path(lines,d,d/'other')
    def test_reject_mismatched_source_and_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            for name in ('potential','model.xyz','restart'): (d/name).write_text(name,encoding='utf8')
            lines=[['potential','potential'],['read_pimd_restart','restart']]
            source={'bead_force_backend':'native_GPUMD','model_sha256':cross.sha(d/'potential'),
                    'model_xyz_sha256':cross.sha(d/'model.xyz'),'restart_input_sha256':cross.sha(d/'restart')}
            cross.validate_inputs(source,d,lines)
            source['model_sha256']='0'*64
            with self.assertRaises(ValueError): cross.validate_inputs(source,d,lines)
            source['bead_force_backend']='WPE'
            with self.assertRaises(ValueError): cross.validate_inputs(source,d,lines)
            log=d/'stdout'; log.write_text('Number of atoms is 5896.\nnumber of beads is 32.\nTime step for this run is 0.1 fs.\nFinished running GPUMD.\nPPPM mesh: 60 x 60 x 60 (target spacing 0.5 A;\nPPPM mesh: 59 x 60 x 60 (target spacing 0.5 A;\n')
            with self.assertRaises(ValueError): cross.stdout_mesh(log,.1,10)
if __name__=='__main__': unittest.main()
