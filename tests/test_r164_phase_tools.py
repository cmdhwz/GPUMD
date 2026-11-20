import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class PhaseToolsTest(unittest.TestCase):
    def test_conversion_preserves_order_and_state(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); phase=d/'phase.npz'; restart=d/'restart.xyz'; model=d/'model.xyz'
            q=np.array([[[1.,2.,3.],[2.,3.,4.]],[[1.1,2.,3.],[2.,3.1,4.]]])
            p=np.arange(12,dtype=float).reshape(2,2,3); mass=np.array([1658.,104.])
            np.savez(phase,q_A=q,p_eV_fs_per_A=p,mass=mass,species=np.array(['O','H']),cell=np.eye(3)*10,
                     atom_row_zero_based=np.arange(2),bead_order_zero_based=np.arange(2))
            subprocess.run([sys.executable,str(ROOT/'tools/r164_phase_to_restart.py'),str(phase),str(restart),
                            '--temperature','300','--model-output',str(model)],check=True)
            lines=restart.read_text(encoding='utf-8').splitlines()
            for s in range(2):
                for i in range(2):
                    row=lines[(s+1)*4+2+i].split()
                    self.assertRegex(row[4], r'^\d+\.\d{4}$')
                    self.assertEqual(row[0],['O','H'][i]); np.testing.assert_allclose(np.array(row[1:4],float),q[s,i],rtol=0,atol=0)
                    np.testing.assert_allclose(np.array(row[5:8],float)*mass[i],p[s,i],atol=1e-14)
            self.assertIn('mass:R:1',model.read_text(encoding='utf-8'))
            result=subprocess.run([sys.executable,str(ROOT/'tools/r164_phase_to_restart.py'),str(phase),str(restart), '--temperature','300'],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_frame_identity_and_time(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            (d/'metadata.txt').write_text('P 2\nN 1\ndt_fs 0.1\ncell_angstrom 10 10 10\natom 0 O 15.999\n',encoding='utf8')
            (d/'frame_1.txt').write_text('bead 0 -1\n0 1 2 3 4 5 6 7 8 9\nbead 1 -2\n0 2 3 4 5 6 7 8 9 10\n',encoding='utf8')
            (d/'frame_0.txt').write_text('initial frame',encoding='utf8')
            (d/'source_identity.txt').write_text('model_sha256 example\n',encoding='utf8')
            scalar=d/'scalar.txt'; scalar.write_text('# header\n1 0.1\n',encoding='utf8')
            cmd=[sys.executable,str(ROOT/'tools/r164_frame_to_npz.py'),str(d),'1',str(d/'out.npz')]
            subprocess.run(cmd,check=True)
            with np.load(d/'out.npz') as out:
                self.assertEqual(out['q_A'].shape,(2,1,3)); self.assertEqual(out['time_fs'],0.1); self.assertEqual(out['dt_fs'],0.1); self.assertEqual(len(str(out['capture_id_sha256'])),64)
            scalar.write_text('1 0.2\n',encoding='utf8')
            cmd[-1]=str(d/'bad.npz')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            scalar.write_text('1 0.1',encoding='utf8')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

    def test_relaunch_receipt_and_one_wrap_guard(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); parent=d/'parent'; parent.mkdir()
            (parent/'metadata.txt').write_text('P 2\nN 1\ndt_fs 0.1\ntemperature_K 300\ncell_angstrom 10 10 10\natom 0 O 15.999\n',encoding='utf8')
            (parent/'source_identity.txt').write_text('model_sha256 abc\n',encoding='utf8')
            (parent/'scalar.txt').write_text('1 0.1\n',encoding='utf8')
            (parent/'frame_1.txt').write_text('bead 0 -1\n0 1 2 3 4 5 6 7 8 9\nbead 1 -2\n0 2 3 4 5 6 7 8 9 10\n',encoding='utf8')
            child=d/'child'
            cmd=[sys.executable,str(ROOT/'tools/r164_phase_relaunch.py'),str(parent),'1',str(child)]
            subprocess.run(cmd,check=True)
            self.assertTrue((child/'parent_receipt.json').is_file())
            self.assertTrue((child/'restart_beads.xyz').is_file())
            self.assertTrue((child/'model.xyz').is_file())
            (parent/'frame_1.txt').write_text('bead 0 -1\n0 21 2 3 4 5 6 7 8 9\nbead 1 -2\n0 2 3 4 5 6 7 8 9 10\n',encoding='utf8')
            cmd[-1]=str(d/'bad')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertFalse((d/'bad'/'parent_receipt.json').exists())

if __name__=='__main__': unittest.main()
