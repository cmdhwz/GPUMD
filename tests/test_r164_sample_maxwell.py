import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_sample_maxwell as sampler


class MaxwellSamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.q=np.array([[[1.,2.,3.],[4.,5.,6.]],[[1.2,2.,3.],[4.,5.1,6.]]])
        self.mass=np.array([1658.,104.])
        self.data=dict(q_A=self.q,mass=self.mass,species=np.array(['O','H']),cell=np.diag([10.,12.,14.]),atom_row_zero_based=np.arange(2),bead_order_zero_based=np.arange(2))
    def tearDown(self): self.tmp.cleanup()
    def archive(self,name,**changes):
        data=dict(self.data); data.update(changes); path=self.root/name; np.savez(path,**data); return path
    def draw(self,source,dest,seed=42):
        return sampler.sample(source,sampler.digest(source),300.,seed,'diagnostic',self.root/dest)
    def test_formula_seed_old_momentum_and_readback(self):
        a=self.archive('a.npz',p_eV_fs_per_A=np.ones((2,2,3)))
        b=self.archive('b.npz',p_eV_fs_per_A=np.full((2,2,3),999.))
        c=self.archive('c.npz')
        self.draw(a,'out1'); self.draw(b,'out2'); receipt=self.draw(c,'out3'); self.draw(c,'out4',43)
        with np.load(self.root/'out1/phase.npz') as out, np.load(self.root/'out2/phase.npz') as out2, np.load(self.root/'out3/phase.npz') as out3, np.load(self.root/'out4/phase.npz') as out4:
            expected=np.random.Generator(np.random.PCG64(42)).standard_normal((2,2,3))*np.sqrt(self.mass*2*sampler.K_B*300)[None,:,None]
            np.testing.assert_array_equal(out['p_eV_fs_per_A'],expected)
            np.testing.assert_array_equal(out['p_eV_fs_per_A'],out2['p_eV_fs_per_A'])
            np.testing.assert_array_equal(out['p_eV_fs_per_A'],out3['p_eV_fs_per_A'])
            self.assertFalse(np.array_equal(out['p_eV_fs_per_A'],out4['p_eV_fs_per_A']))
            np.testing.assert_array_equal(out['q_A'],self.q); np.testing.assert_array_equal(out['mass'],self.mass)
            np.testing.assert_array_equal(out['atom_row_zero_based'],np.arange(2))
        restart=(self.root/'out1/restart_beads.xyz').read_text().splitlines()
        model=(self.root/'out1/model.xyz').read_text().splitlines()
        for bead in range(2):
            for atom in range(2):
                row=restart[(bead+1)*4+2+atom].split()
                self.assertRegex(row[4],r'^\d+\.\d{4}$')
                np.testing.assert_array_equal(np.array(row[1:4],float),self.q[bead,atom])
                np.testing.assert_allclose(np.array(row[5:8],float)*self.mass[atom],expected[bead,atom],rtol=0,atol=1e-14)
        for atom in range(2):
            self.assertAlmostEqual(float(model[2+atom].split()[4])*sampler.TIME_UNIT_CONVERSION**2,self.mass[atom],places=11)
        self.assertEqual(receipt['sampling']['source_momentum'],'not_read')
        self.assertAlmostEqual(receipt['kinetic']['expected_K_P_eV'],1.5*2*2*2*sampler.K_B*300)
        self.assertAlmostEqual(receipt['kinetic']['expected_K_P_over_P_eV'],1.5*2*2*sampler.K_B*300)
        self.assertAlmostEqual(receipt['kinetic']['P_cm_component_expected_std_eV_fs_per_A'],np.sqrt(self.mass.sum()*sampler.K_B*300))
        self.assertFalse((self.root/'out1/run.in').exists())
        with self.assertRaises(FileExistsError): self.draw(a,'out1')
    def test_invalid_inputs_rejected(self):
        bad=[dict(q_A=np.repeat(self.q[:1],3,axis=0),bead_order_zero_based=np.arange(3)),dict(q_A=self.q.astype(complex)+1j),dict(mass=np.array([1.,-1.])),
             dict(atom_row_zero_based=np.array([0.,1.])),dict(species=np.array(['O','X'])),
             dict(cell=np.array([[10.,1.,0.],[0.,12.,0.],[0.,0.,14.]])),
             dict(q_A=self.q+100),dict(q_A=np.array([[[1.,2.,3.],[4.,5.,6.]],[[6.,2.,3.],[4.,5.,6.]]]))]
        for i,changes in enumerate(bad):
            path=self.archive(f'bad{i}.npz',**changes)
            with self.assertRaises(ValueError): self.draw(path,f'badout{i}')
            self.assertFalse((self.root/f'badout{i}').exists())
        path=self.archive('valid.npz')
        for seed in (-1,2**128):
            with self.assertRaises(ValueError): self.draw(path,f'seed{seed}',seed)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            sampler.sample(path,'0'*64,300,1,'id',self.root/'badsha')


if __name__=='__main__': unittest.main()
