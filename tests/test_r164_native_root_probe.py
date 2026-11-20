import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from r164_native_root_probe import prepare, review


class RootProbeIdentityTest(unittest.TestCase):
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
