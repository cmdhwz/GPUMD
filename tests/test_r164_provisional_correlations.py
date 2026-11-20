import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corr',ROOT/'tools/r164_provisional_correlations.py')
corr=importlib.util.module_from_spec(spec); spec.loader.exec_module(corr)

class CorrelationTests(unittest.TestCase):
    def fixture(self, d, n=8):
        h='a'*64
        (d/'metadata.txt').write_text(f'schema r164_diagnostic_v1\ntime_convention step_times_fixed_dt_start_zero\nP 32\nN 5896\ntemperature_K 300\ndt_fs 0.1\ncell_angstrom 29 29 29\nreference_id R27_fixed_2OH_R157\nreference_source_sha256 1d46365935f48e390d08e69e8564a3adc692d439b9472b4250914b22144a2eaa\n',encoding='utf8')
        (d/'source_identity.txt').write_text(f'bead_force_backend native_GPUMD\nmodel_sha256 {h}\nrun_in_sha256 {h}\n',encoding='utf8')
        (d/'observer.txt').write_text(f'model_sha256 {h}\npppm_spacing 0.5\n',encoding='utf8')
        (d/'COMPLETE').write_text(f'r164_diagnostic_v1 final_step {n-1}\n',encoding='utf8')
        lines=['# header\n']
        for i in range(n):
            b=np.array([i,i*i,(-1)**i],float); v=np.array([i**3,2*i,1.0],float); j=b+v
            base=[i,i*0.1,0,0,0,0,0,0,0]+[i+k for k in range(12)]
            rows=[*base,'B_UNVERIFIED',*b,'J2_UNVERIFIED',*j,'R27_UNVERIFIED',0,0,0,0,*v]
            lines.append(' '.join(map(str,rows))+'\n')
        (d/'scalar.txt').write_text(''.join(lines),encoding='utf8')
    def test_order_and_common_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.fixture(d); result=corr.analyze([d],2)
            x,_,_,_,_,_=corr.load_trace(d); mean=x.mean(axis=0)
            expected=(x[:-1]-mean).T@(x[1:]-mean)/(len(x)-1)
            np.testing.assert_allclose(result['C_full_18x18'][1],expected)
            np.testing.assert_allclose(result['C_bd'][1],expected[:3,3:6])
            np.testing.assert_allclose(result['C_db'][1],expected[3:6,:3])
            self.assertFalse(np.allclose(result['C_bd'][1],result['C_db'][1]))
            self.assertLess(result['J2_identity_max_abs'],1e-12)
            self.assertEqual(result['sample_counts'],[8,7,6]); self.assertEqual(result['paired_block_count'],2)
            self.assertEqual(np.asarray(result['integrated_full_18x18_trapezoid']).shape,(3,18,18))
            with self.assertRaises(ValueError): corr.analyze([d,d],2)
            copy=d/'copy'; copy.mkdir(); self.fixture(copy)
            with self.assertRaises(ValueError): corr.analyze([d,copy],2)
    def test_eh_one_interval_identity(self):
        x=np.arange(108,dtype=float).reshape(6,18)**1.1
        result=corr.eh_diagnostic([x],0.2,1)
        self.assertEqual(result['origin_count'],5)
        np.testing.assert_allclose(result['eh_covariance_over_2t'],result['triangular_correlation_integral'],rtol=1e-12,atol=1e-12)
        self.assertLess(np.max(np.abs(result['finite_sample_residual'])),1e-12)
        self.assertIsNone(corr.eh_diagnostic([x[:2]],0.2,1)['eh_covariance_over_2t'])

    def test_reject_missing_hole_and_bad_j(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.fixture(d); scalar=d/'scalar.txt'; original=scalar.read_text(encoding='utf8')
            (d/'COMPLETE').unlink()
            with self.assertRaises(FileNotFoundError): corr.analyze([d],1)
            self.fixture(d); lines=original.splitlines(True); scalar.write_text(''.join(lines[:3]+lines[4:]),encoding='utf8')
            with self.assertRaises(ValueError): corr.analyze([d],1)
            self.fixture(d); scalar.write_text(original.replace('J2_UNVERIFIED 0.0', 'J2_UNVERIFIED 9.0',1),encoding='utf8')
            with self.assertRaises(ValueError): corr.analyze([d],1)
            self.fixture(d); scalar.write_text(original[:-1],encoding='utf8')
            with self.assertRaises(ValueError): corr.analyze([d],1)

if __name__=='__main__': unittest.main()
