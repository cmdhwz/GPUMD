import sys
import hashlib
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from r164_native_root_probe import prepare, review, validate_assignment_join


class RootProbeIdentityTest(unittest.TestCase):
    def test_optional_assignment_join_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prefix='trace'
            self.assertIsNone(validate_assignment_join(root,prefix,{},0,3,4))
            path=root/(prefix+'_assignment_join.out')
            path.write_text('# schema r164_assignment_join_v1\n# step 0\n# atom 3\n# force_evaluation_id 4\n',encoding='utf8')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(validate_assignment_join(root,prefix,{'assignment_join_sha256':digest},0,3,4),digest)
            with self.assertRaises(ValueError): validate_assignment_join(root,prefix,{'assignment_join_sha256':'0'*64},0,3,4)
            with self.assertRaises(ValueError): validate_assignment_join(root,prefix,{'assignment_join_sha256':digest},0,2,4)

    def test_prepare_rejects_wrong_geometry_sha_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = root / 'geometry.json'
            geometry.write_text('{}', encoding='utf8')
            destination = root / 'decks'
            with self.assertRaisesRegex(ValueError, 'geometry SHA mismatch'):
                prepare(geometry, '0' * 64, 0, 0, 0, 'x', 'assignment_center', 0, destination)
            self.assertFalse(destination.exists())

    def test_review_rejects_wrong_manifest_sha_before_creating_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text('{}', encoding='utf8')
            output = root / 'review.json'
            with self.assertRaisesRegex(ValueError, 'manifest SHA mismatch'):
                review(root, '0' * 64, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
