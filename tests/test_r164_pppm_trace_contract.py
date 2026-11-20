"""Source-level trace contracts; native execution is verified separately."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class TraceContract(unittest.TestCase):
    def test_existing_bead_force_and_no_extra_compute(self):
        text=(ROOT/'src/force/force.cu').read_text(encoding='utf8')
        body=text.split('void Force::compute_pimd_beads(',1)[1].split('void Force::compute_pimd_bead_range_on_device(',1)[0]
        self.assertLess(body.index('request_pppm_debug_for_next_force'),body.index('      compute('))
        self.assertIn('trace && k == trace->bead',body)
    def test_trace_receipt_after_scalar_and_uses_raw_files(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        output=text.split('void R164_RPMD::output(',1)[1]
        self.assertLess(output.index('scalar_file.close()'),output.index('if (trace_here) {'))
        for label in ('metadata','source_identity','frame','force_images','assignment_stencil','pre_electro','force_evaluation_id'):
            self.assertIn(label,output)
    def test_both_descriptor_paths_capture_before_normalization(self):
        text=(ROOT/'src/force/nep_charge.cu').read_text(encoding='utf8')
        self.assertEqual(text.count('write_r164_pre_electro_trace(N, potential_per_atom);'),2)
        for part in text.split('write_r164_pre_electro_trace(N, potential_per_atom);')[1:]:
            self.assertLess(part.index('zero_total_charge<<<'),part.index('pppm.find_force('))
if __name__=='__main__': unittest.main()
