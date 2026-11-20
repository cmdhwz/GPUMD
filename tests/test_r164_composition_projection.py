import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import r164_composition_projection as projection
from r164_provisional_correlations import digest


class ProjectionTests(unittest.TestCase):
    def test_full_rank_and_common_j2_schur(self):
        rng = np.random.default_rng(23)
        z = rng.normal(size=(18,18))
        matrix = z @ z.T + np.eye(18)
        got = projection.matrix_diagnostic(matrix,np.eye(18),1e-8,1e-8)
        self.assertEqual(got['status'],'FINITE_SAMPLE_SUPPORTED_ONLY')
        expected = matrix[:6,:6] - matrix[:6,6:] @ np.linalg.inv(matrix[6:,6:]) @ matrix[6:,:6]
        np.testing.assert_allclose(got['projected']['Lperp_6x6'], expected, rtol=1e-12, atol=1e-12)
        self.assertLess(got['projected']['block_identity_max_abs'],1e-12)

    def test_structural_null_and_coupled_null(self):
        matrix = np.diag([1.0]*6 + [1.0]*9 + [0.0]*3)
        got = projection.matrix_diagnostic(matrix,np.eye(18),1e-8,1e-8)
        self.assertEqual([x['rank'] for x in got['threshold_scan']],[9,9,9])
        self.assertEqual(got['status'],'FINITE_SAMPLE_SUPPORTED_ONLY')
        matrix[3,17] = matrix[17,3] = 1.0
        got = projection.matrix_diagnostic(matrix,np.eye(18),1e-8,1e-8)
        self.assertIn('NULL_CROSS_INCOMPATIBLE',got['reasons'])
        self.assertEqual(got['threshold_scan'][1]['null_cross']['d2']['null_cross_ratio'],1)
        self.assertIsNone(got['projected'])

    def test_negative_and_unstable_rank(self):
        matrix = np.eye(18); matrix[6,6] = -0.1
        self.assertIn('NEGATIVE_YY_SPECTRUM',projection.matrix_diagnostic(matrix,np.eye(18),1e-8,1e-5)['reasons'])
        matrix[6,6] = 1e-8
        got = projection.matrix_diagnostic(matrix,np.eye(18),1e-8,1e-5)
        self.assertIn('SUPPORT_RANK_UNSTABLE',got['reasons'])
        self.assertIsNone(got['projected'])

    def test_species_transform_round_trip(self):
        counts = dict(zip(projection.SPECIES, (4, 8, 1, 1)))
        masses = dict(zip(projection.SPECIES, (16., 1., 23., 35.)))
        a, total = projection.transform(counts, masses)
        flows = np.arange(12, dtype=float).reshape(4, 3) / 10
        y = (a[6:,6:] @ flows.ravel()).reshape(4, 3)
        oxygen_relative = -(masses['H']*y[1] + masses['Na']*y[2] + masses['Cl']*y[3])/masses['O']
        reconstructed = np.vstack((oxygen_relative, y[1:])) + np.array(list(counts.values()))[:,None]*y[0]
        np.testing.assert_allclose(reconstructed, flows, rtol=1e-13, atol=1e-13)
        self.assertGreater(total, 0)

    def test_mass_flow_and_bad_momentum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'metadata.txt').write_text('N 4\n'+''.join(f'atom {i} {s} {i+1}\n' for i,s in enumerate(projection.SPECIES)),encoding='utf8')
            n = np.arange(12,dtype=float).reshape(4,3)
            p = np.arange(1,5) @ n * projection.MASS_FACTOR
            def write(momentum):
                values = [0,0,0,0,0,0,*momentum,*n.ravel()]
                (root/'scalar.txt').write_text(' '.join(map(str,values))+'\n',encoding='utf8')
            write(p)
            hashes = {name:digest(root/name) for name in ('metadata.txt','scalar.txt')}
            self.assertEqual(projection.composition(root,hashes)[3],1)
            write(p+1)
            hashes['scalar.txt'] = digest(root/'scalar.txt')
            with self.assertRaisesRegex(ValueError,'momentum mismatch'):
                projection.composition(root,hashes)


if __name__ == '__main__':
    unittest.main()
