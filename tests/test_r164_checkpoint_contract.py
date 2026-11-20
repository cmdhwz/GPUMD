"""Source-level contracts pending native GPU checkpoint execution."""
import hashlib
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "src/measure/r164_rpmd.cu").read_text(encoding="utf8")

class CheckpointContract(unittest.TestCase):
    def test_parser_accepts_only_ordered_positive_checkpoint(self):
        parse = SOURCE.split("void R164_RPMD::parse(", 1)[1].split("void R164_RPMD::read_positions", 1)[0]
        self.assertLess(parse.index('t[next]=="observer"'), parse.index('t[next]=="checkpoint"'))
        for guard in ('next+1>=t.size()', "c<'0' || c>'9'", 'std::numeric_limits<int>::max()', 'checkpoint_stride_<=0'):
            self.assertIn(guard, parse)

    def test_commit_order_and_opt_in_cadence(self):
        output = SOURCE.split('void R164_RPMD::output(', 1)[1]
        for text in ('checkpoint_stride_ && (step==0 || step==total || step%checkpoint_stride_==0)',
                     'snapshot || (!buffered_ && (step==0 || step==total || checkpoint || trace_here))'):
            self.assertIn(text, output)
        points = [output.index(x) for x in ('atomic_commit_new_file(frame_path', 'atomic_commit_new_file(image_path',
                  'crossings.close()', 'scalar_file.close()', 'sha256_text(scalar_line)',
                  'PRINT_INPUT_ERROR("r164_rpmd failed committing checkpoint.")', 'if (step == total) write_complete(step)')]
        self.assertEqual(points, sorted(points))
        self.assertIn('if (checkpoint) {', output)
        complete=SOURCE.split('void R164_RPMD::write_complete(',1)[1].split('void R164_RPMD::output(',1)[0]
        self.assertIn('atomic_commit_new_file(marker+".tmp",marker)',complete)

    def test_canonical_line_hash_and_required_fields(self):
        lf = b'12 1.2 3.4\n'
        self.assertEqual(hashlib.sha256(lf).digest(), hashlib.sha256(b'12 1.2 3.4\r\n'.rstrip(b'\r\n') + b'\n').digest())
        for field in ('metadata_sha256', 'source_identity_sha256', 'frame_sha256', 'force_images_sha256',
                      'scalar_line_sha256', 'scalar_prefix_sha256', 'cache_policy fresh_rebuild_only', 'physical_event_history unavailable'):
            self.assertIn(field, SOURCE)
        self.assertIn('metadata_hash.size()!=64', SOURCE)
        self.assertIn('scalar_hash.size()!=64', SOURCE)

if __name__ == '__main__': unittest.main()
