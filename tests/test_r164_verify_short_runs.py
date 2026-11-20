import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('verify',ROOT/'tools/r164_verify_short_runs.py'); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

class ShortRunTests(unittest.TestCase):
    def test_cards_reject_wrong_hash_and_configuration(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'run.in'; path.write_text((ROOT/'examples/r164_diagnostic/run_0p1fs.in').read_text().replace('r164_rpmd r164_output 1','r164_rpmd r164_output 1 observer'),encoding='utf8')
            identity={'run_in_sha256':v.sha(path)}; self.assertEqual(len(v.card(path,identity,0.1,10,True)),6)
            with self.assertRaises(ValueError): v.card(path,{'run_in_sha256':'0'*64},0.1,10,True)
            path.write_text(path.read_text().replace('pimd_fix_com off','pimd_fix_com on'),encoding='utf8'); identity['run_in_sha256']=v.sha(path)
            with self.assertRaises(ValueError): v.card(path,identity,0.1,10,True)
    def test_scalar_holes_times_and_midwindow_drift(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'scalar.txt'; lines=[]
            for i,h in enumerate([1.,1.4,1.0]):
                lines.append(' '.join(map(str,[i,i*.1,0,0,0,h,0,0,0]+[0]*12))+'\n')
            path.write_text(''.join(lines)); rows=v.read_scalars(path,.1,2)
            result=v.energy({'records':rows,'dt':.1}); self.assertAlmostEqual(result['max_abs_drift_eV'],.4); self.assertEqual(result['max_drift_step'],1)
            path.write_text(''.join([lines[0],lines[2]]))
            with self.assertRaises(ValueError): v.read_scalars(path,.1,2)
            lines[1]=lines[1].replace('0.1 ','0.15 ',1); path.write_text(''.join(lines))
            with self.assertRaises(ValueError): v.read_scalars(path,.1,2)
    def test_actual_image_rows_and_component_comparison(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'images.txt'; path.write_text('# images\n0 0 1 0 -1\n',encoding='utf8')
            q=np.array([[[11.,2.,-7.]]]); image=v.read_images(path,q,np.array([10.,10.,10.]))
            np.testing.assert_array_equal(image,[[[1,0,-1]]])
            path.write_text('# images\n0 1 1 0 -1\n',encoding='utf8')
            with self.assertRaises(ValueError): v.read_images(path,q,np.array([10.,10.,10.]))
            path.write_text('# images\n0 0 0 0 -1\n',encoding='utf8')
            with self.assertRaises(ValueError): v.read_images(path,q,np.array([10.,10.,10.]))

    def test_comparison_keeps_phase_and_scalar_differences(self):
        q=np.zeros((1,1,3)); p=q.copy(); f=q.copy(); u=np.zeros(1)
        zero={k:np.zeros(3) for k in ('conv','local','mechanical','relay','projection')}
        one={k:np.ones(3) for k in zero}
        a={'frames':[(q,p,f,u)],'cell':np.ones(3)*10,'dt':.1,'records':[{'base':np.zeros(19),'b':np.zeros(3),'d2':np.zeros(3),'j2':np.zeros(3),'components':zero}]}
        b={'frames':[(q+.2,p+.3,f+.4,u+.5)],'cell':np.ones(3)*10,'dt':.1,'records':[{'base':np.ones(19),'b':np.ones(3),'d2':None,'j2':None,'components':one}]}
        item=v.compare(a,b,[(0,0)])[0]
        self.assertAlmostEqual(item['q_lift_max_A'],.2); self.assertAlmostEqual(item['F_max_eV_per_A'],.4)
        self.assertEqual(item['observer_components_max_abs']['relay'],1.); self.assertIsNone(item['d2_max_abs'])
        self.assertEqual(item['energy_momentum_scalar_difference'],[-1.]*7)

    def test_frame_scalar_ledger_and_observer_contradictions(self):
        q=np.zeros((2,2,3)); q[1,:,0]=[1.,2.]
        p=np.zeros_like(q); p[0,0,0]=2.; p[1,0,0]=4.; p[0,1,1]=4.; p[1,1,1]=8.
        mass=np.array([2.,4.]); species=np.array(['O','H']); U=np.array([1.,2.]); T=300.
        omega=2*8.617343e-5*T/(6.465412e-2*10.18051)
        # Hand sums: K=5+10, spring=(2+16)*omega^2, centroid p=(3,6,0), species v=(1.5,0,0)/(0,1.5,0).
        kinetic=15.; spring=18.*omega**2
        base=np.array([3.,kinetic,spring,(3.+kinetic+spring)/4.,3.,6.,0.,1.5,0,0,0,1.5,0,0,0,0,0,0,0])
        record={'base':base,'components':{k:None for k in ('conv','local','mechanical','relay','projection')}}
        result=v.validate_frame_scalars(q,p,U,mass,species,T,record)
        self.assertLess(max(map(abs,result['residual'])),1e-12)
        bad=dict(record,base=base.copy()); bad['base'][1]+=0.01
        with self.assertRaises(ValueError): v.validate_frame_scalars(q,p,U,mass,species,T,bad)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); v.check_observer_mode(d,[record],False)
            (d/'observer.txt').write_text('unexpected',encoding='utf8')
            with self.assertRaises(ValueError): v.check_observer_mode(d,[record],False)
            (d/'observer.txt').unlink(); bad_components=dict(record,components=dict(record['components'],relay=np.ones(3)))
            with self.assertRaises(ValueError): v.check_observer_mode(d,[bad_components],False)

    def test_cpu_initial_energy_force_delta_and_phase_gate(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t); N=5896; P=32; shape=(P,N,3); q=np.zeros(shape); p=np.zeros(shape); force=np.zeros(shape); U=np.arange(P,dtype=float)
            mass=np.ones(N); species=np.full(N,'H',dtype='<U2'); species[0]='O'; species[1]='Na'; species[2]='Cl'; cell=np.array([29.3104]*3)
            np.savez(d/'phase.npz',q_A=q,p_eV_fs_per_A=p,mass=mass,species=species,cell=np.diag(cell),atom_row_zero_based=np.arange(N),bead_order_zero_based=np.arange(P))
            np.savez(d/'forces.npz',F_eV_per_A=force,U_eV=U)
            actual=force.copy(); actual[:,0,0]=2.; run={'frames':[(q,p,actual,U+3)],'species':species,'mass':mass,'cell':cell,'source':{'model_sha256':'0'*64}}
            result=v.cpu_reference(run,d/'phase.npz',d/'forces.npz'); self.assertTrue(result['same_phase_for_force_comparison']); self.assertEqual(result['reference_identity'],'UNVERIFIED')
            self.assertEqual(result['force_energy_comparison']['U_mean_offset_eV'],3.); self.assertEqual(result['force_energy_comparison']['F_by_species_eV_per_A']['O']['max_abs'],2.)
            badp=p.copy(); badp[0,0,0]=1e-7; run['frames']=[(q,badp,actual,U+3)]
            self.assertIsNone(v.cpu_reference(run,d/'phase.npz',d/'forces.npz')['force_energy_comparison'])
if __name__=='__main__': unittest.main()
