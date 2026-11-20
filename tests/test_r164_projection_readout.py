import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_projection_readout as proj


class ProjectionReadoutTests(unittest.TestCase):
    def test_sorted_against_all_pairs(self):
        x=np.array([[.5,1.,2.],[2.,4.,6.],[8.,8.,10.],[4.,9.,12.]])
        cell=np.array([11.,13.,17.]); D=np.array([1.,-2.,3.,4.],np.float32); rate=np.array([.3,.7,-.2,.1],np.float32)
        actual,image,c,guard=proj.replay_projection(x,D,rate,cell)
        self.assertIsNone(guard)
        expected=np.zeros(3); expected_image=np.zeros(3)
        for d in range(3):
            for i in range(len(x)):
                left=sum(x[j,d]-x[i,d] < -cell[d]/2 for j in range(len(x)))
                right=sum(x[j,d]-x[i,d] > cell[d]/2 for j in range(len(x)))
                image_term=cell[d]*(right-left)/len(x)
                expected[d]+=(x[i,d]-sum(x[:,d]/len(x))+image_term)*c[i]/proj.TIME_UNIT
                expected_image[d]+=image_term*c[i]/proj.TIME_UNIT
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
        np.testing.assert_allclose(image,expected_image,rtol=0,atol=1e-15)
    def test_guard_and_signed_jump(self):
        cell=np.array([10.,12.,14.]); D=np.array([1.,-2.,3.],np.float32); rate=np.array([.3,.7,-.2],np.float32)
        for xi,xj in ((1.,6.),(6.,1.)):
            x=np.array([[xi,1.,1.],[xj,2.,2.],[3.,3.,3.]])
            _,_,c,guard=proj.replay_projection(x,D,rate,cell)
            self.assertEqual(guard,(0,0))
            epsilon=1e-6; left=x.copy(); right=x.copy(); left[1,0]-=epsilon; right[1,0]+=epsilon
            a,_,_,_=proj.replay_projection(left,D,rate,cell); b,_,_,_=proj.replay_projection(right,D,rate,cell)
            jump=cell[0]/len(x)*(c[0]-c[1])/proj.TIME_UNIT
            smooth=(2*epsilon)*(c[1]-sum(c)/len(c))/proj.TIME_UNIT
            self.assertAlmostEqual(b[0]-a[0]-smooth,jump,places=13)
    def test_valid_and_guard_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); cell=np.array([10.,12.,14.]); P=2; N=3
            (d/'metadata.txt').write_text('schema r164_diagnostic_v1\nP 2\nN 3\ndt_fs 0.1\ncell_angstrom 10 12 14\n')
            (d/'source_identity.txt').write_text('backend native\n'); (d/'observer.txt').write_text('native_private_qnep_pppm_observer\n')
            (d/'observer_frame_0.txt').write_text('# step 0 force_call 1 mesh 60 60 60\n')
            D=np.array([1.,-2.,3.],np.float32); rate=np.array([.3,.7,-.2],np.float32)
            for guarded in (False,True):
                x=np.array([[1.,1.,1.],[6. if guarded else 3.,2.,2.],[4.,3.,4.]])
                frame=[]
                for bead in range(P):
                    frame.append(f'bead {bead} 0\n')
                    for i in range(N): frame.append(str(i)+' '+' '.join(map(str,x[i]))+' 0 0 0 0 0 0\n')
                (d/'frame_0.txt').write_text(''.join(frame))
                value,image,_,guard=proj.replay_projection(x,D,rate,cell)
                if guarded: scalar=b'0 0 PROJECTION_UNAVAILABLE PROJECTION_IMAGE_UNAVAILABLE\n'
                else: scalar=('0 0 PROJECTION_UNVERIFIED '+' '.join(map(str,value))+' PROJECTION_IMAGE_UNVERIFIED '+' '.join(map(str,image))+'\n').encode()
                (d/'scalar.txt').write_bytes(scalar)
                header={'schema':'r164_observer_projection_v1','step':'0','time_fs':'0','observer_force_call':'1','N':'3',
                        'status':'half_box_guard' if guarded else 'valid','guard_axis':'0' if guarded else '-1',
                        'guard_index':'0' if guarded else '-1','rows':'3','time_unit_conversion':'10.18051',
                        'half_box_guard_A':'1e-9','columns':'atom centroid_canonical_xyz_A raw_D_fp32_eV_per_e raw_rate_fp32_e_per_natural_time','scalar_line_sha256':hashlib.sha256(scalar).hexdigest()}
                for name,file in (('metadata','metadata.txt'),('source_identity','source_identity.txt'),('observer','observer.txt'),('frame','frame_0.txt'),('observer_frame','observer_frame_0.txt')): header[name+'_sha256']=proj.sha(d/file)
                lines=['# '+k+' '+v+'\n' for k,v in header.items()]
                for i in range(N): lines.append(' '.join([str(i),*[format(v,'.17g') for v in x[i]],format(float(D[i]),'.17g'),format(float(rate[i]),'.17g')])+'\n')
                (d/'observer_projection_0.txt').write_text(''.join(lines))
                result=proj.readout(d,0,(0,1,'x') if guarded else None)
                if guarded:
                    self.assertIsNone(result['recorded_projection']); self.assertEqual(result['guard']['center_index'],0)
                    self.assertIn('frozen_pair_jump',result)
                else:
                    np.testing.assert_allclose(result['projection_residual'],0,atol=1e-15)
                    lines[-1]=lines[-1].replace(' 3 -0.20000000298023224',' 4 -0.20000000298023224')
                    (d/'observer_projection_0.txt').write_text(''.join(lines))
                    with self.assertRaisesRegex(ValueError,'replay residual'): proj.readout(d,0)

    def test_negative_step_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonnegative'): proj.readout('.',-1)

    def test_source_sha_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); (d/'metadata.txt').write_text('schema r164_diagnostic_v1\nP 2\nN 1\ndt_fs 0.1\ncell_angstrom 10 10 10\n')
            for name in ('source_identity.txt','observer.txt','frame_0.txt','observer_frame_0.txt'): (d/name).write_text('original')
            scalar=b'0 0 PROJECTION_UNAVAILABLE PROJECTION_IMAGE_UNAVAILABLE\n'; (d/'scalar.txt').write_bytes(scalar)
            header={'schema':'r164_observer_projection_v1','step':'0','time_fs':'0','observer_force_call':'1','N':'1','status':'charge_diagnostics_unavailable','guard_axis':'-1','guard_index':'-1','rows':'0','time_unit_conversion':'10.18051','half_box_guard_A':'1e-9','columns':'atom centroid_canonical_xyz_A raw_D_fp32_eV_per_e raw_rate_fp32_e_per_natural_time','scalar_line_sha256':hashlib.sha256(scalar).hexdigest()}
            for name,file in (('metadata','metadata.txt'),('source_identity','source_identity.txt'),('observer','observer.txt'),('frame','frame_0.txt'),('observer_frame','observer_frame_0.txt')): header[name+'_sha256']=proj.sha(d/file)
            (d/'observer_projection_0.txt').write_text(''.join('# '+k+' '+v+'\n' for k,v in header.items()))
            (d/'observer_frame_0.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'source SHA mismatch'): proj.readout(d,0)

if __name__=='__main__': unittest.main()
