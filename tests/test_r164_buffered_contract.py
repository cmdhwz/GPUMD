from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class BufferedContract(unittest.TestCase):
    def test_skipped_step_preserves_sequence_before_transfer(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        body=text.split('void R164_RPMD::output(',1)[1]
        self.assertLess(body.index('offset_[j]=q_[j]-current[j]'),body.index('if (buffered_ && !sample && !snapshot)'))
        self.assertLess(body.index('if (buffered_ && !sample && !snapshot)'),body.index('atom.velocity_beads[s].copy_to_host'))
        self.assertIn('drift_pending_=false; ++next_output_step_',body)
        self.assertIn('if (observer_ && sample)',body)
        self.assertIn('if (need_force) atom.force_beads',body)
    def test_finish_orders_series_before_complete(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        body=text.split('void R164_RPMD::finish_buffered(',1)[1].split('void R164_RPMD::write_complete(',1)[0]
        self.assertLess(body.index('atomic_commit_new_file(path+".tmp",path)'),body.index('write_complete(buffered_total_)'))
        self.assertIn('unavailable values nan',body)
        self.assertIn('sample_stride_',body)
        self.assertLess(body.index('atomic_commit_new_file(binary_path+".tmp",binary_path)'),body.index('write_complete(buffered_total_)'))
        self.assertIn('if (!snapshots_.empty())',body)
        run=(ROOT/'src/main_gpumd/run.cu').read_text(encoding='utf8')
        self.assertLess(run.index('const auto time_finish'),run.index('r164_rpmd.finish_buffered(atom)'))
        self.assertLess(run.index('r164_rpmd.finish_buffered(atom)'),run.index('measure.post_run('))
        output=text.split('void R164_RPMD::output(',1)[1]
        self.assertIn('} else write_snapshot(step,q_,v,f,energy,force_images_,atom)',output)
        self.assertNotIn('finish_buffered(',output)
        self.assertIn('frame.v=std::move(v); frame.f=std::move(f)',output)
    def test_option_excludes_other_diagnostics(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        self.assertIn('buffered_ && next<t.size()',text)
        self.assertIn('!decimal(t[next+1],sample_stride_)',text)
        self.assertIn('sample_stride_<=0',text)

if __name__=='__main__': unittest.main()
