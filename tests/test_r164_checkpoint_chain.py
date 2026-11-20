import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import r164_checkpoint_chain as chain

class OptionTests(unittest.TestCase):
    def test_legacy_and_extended(self):
        for args,expected in [(['r164_rpmd','r164_output','1'],(False,None)),(['r164_rpmd','r164_output','1','observer'],(True,None)),(['r164_rpmd','r164_output','1','observer','checkpoint','2'],(True,2)),(['r164_rpmd','r164_output','1','checkpoint','2'],(False,2)),(['r164_rpmd','r164_output','1','observer','checkpoint','2','pppm_trace','0','0','5609'],(True,2)),(['r164_rpmd','r164_output','1','pppm_trace','0','0','1'],(False,None)),(['r164_rpmd','r164_output','1','pppm_trace','0','0','all'],(False,None))]:
            self.assertEqual(chain.r164_options(args),expected)
        for rest in (['checkpoint','0'],['checkpoint','4294967297'],['checkpoint','2','observer'],['checkpoint','+2']):
            with self.assertRaises(ValueError): chain.r164_options(['r164_rpmd','r164_output','1']+rest)

class ChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)/'root'; self.root.mkdir()
        (self.root/'qnep.txt').write_text('potential')
        (self.root/'model.xyz').write_text('model')
        (self.root/'restart_beads.xyz').write_text('restart')
        (self.root/'run.in').write_text('potential qnep.txt\nkspace pppm 0.5\ntime_step 0.1\nensemble rpmd 2 300\npimd_propagator exact\npimd_fix_com off\nread_pimd_restart restart_beads.xyz\nr164_rpmd r164_output 1 checkpoint 1\nrun 1\n')
        self.capture=self.make_capture(self.root)
    def tearDown(self): self.tmp.cleanup()
    def make_capture(self,root,q=0):
        d=root/'r164_output'; d.mkdir()
        meta='schema r164_diagnostic_v1\ntime_convention step_times_fixed_dt_start_zero\ncheckpoint FRESH_REBUILD_ONLY\ncheckpoint_stride 1\nP 2\nN 1\ntemperature_K 300\ndt_fs 0.1\ncell_angstrom 10 10 10\nreference_id R27\nreference_source_sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\natom 0 O 1\n'
        (d/'metadata.txt').write_text(meta)
        source='bead_force_backend native_GPUMD\nbuild_git_commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\nbuild_git_dirty no\n'+''.join(k+' '+chain.sha(root/n)+'\n' for k,n in [('model_sha256','qnep.txt'),('model_xyz_sha256','model.xyz'),('restart_input_sha256','restart_beads.xyz'),('run_in_sha256','run.in')])
        (d/'source_identity.txt').write_text(source)
        rows=[]
        for i in range(2):
            row=f'{i} {i*.1} '+'0 '*19+'OBSERVER_UNAVAILABLE B_UNAVAILABLE J2_UNAVAILABLE R27_NOT_APPLICABLE\n'; rows.append(row)
            (d/f'frame_{i}.txt').write_text(f'bead 0 0\n0 {q} 0 0 0 0 0 0 0 0\nbead 1 0\n0 {q} 0 0 0 0 0 0 0 0\n')
            (d/f'force_images_{i}.txt').write_text('0 0 0 0 0\n1 0 0 0 0\n')
            data={'schema':'r164_checkpoint_v1','step':str(i),'time_fs':str(i*.1),'metadata_sha256':chain.sha(d/'metadata.txt'),'source_identity_sha256':chain.sha(d/'source_identity.txt'),'frame_sha256':chain.sha(d/f'frame_{i}.txt'),'force_images_sha256':chain.sha(d/f'force_images_{i}.txt'),'scalar_line_sha256':hashlib.sha256(row.encode()).hexdigest(),'scalar_prefix_sha256':hashlib.sha256(''.join(rows).encode()).hexdigest(),'cache_policy':'fresh_rebuild_only','physical_event_history':'unavailable'}
            (d/f'checkpoint_{i}.txt').write_text(''.join(k+' '+v+'\n' for k,v in data.items()))
        (d/'scalar.txt').write_bytes((''.join(rows)+'2 partial').encode()); (d/'COMPLETE').write_text('r164_diagnostic_v1 final_step 1\n')
        return d
    def test_dirty_and_unknown_build_rejected(self):
        source=self.capture/'source_identity.txt'; original=source.read_text()
        for identity in ('build_git_dirty yes', 'build_git_dirty unknown'):
            source.write_text(original.replace('build_git_dirty no',identity))
            with self.assertRaisesRegex(ValueError,'clean Git build identity'): chain.load(self.capture,1)
    def test_timestep_rounding_and_step_bounds(self):
        meta=chain.keyfile(self.capture/'metadata.txt'); source=chain.keyfile(self.capture/'source_identity.txt')
        meta['dt_fs']=format(__import__('math').nextafter(.1,1),'.17g')
        chain.config(self.capture,meta,source)
        with self.assertRaisesRegex(ValueError,'positive int'): chain.prepare(self.capture,1,self.root.parent/'new',2147483648)
    def test_observer_identity_rejected_if_mismatched(self):
        (self.capture/'observer.txt').write_text('model_sha256 wrong\nrun_in_sha256 wrong\npppm_spacing 0.5\n')
        with self.assertRaisesRegex(ValueError,'observer-off'): chain.load(self.capture,1)
        card=self.root/'run.in'; card.write_text(card.read_text().replace('1 checkpoint','1 observer checkpoint'))
        src=self.capture/'source_identity.txt'; src.write_text(src.read_text().replace(chain.keyfile(src)['run_in_sha256'],chain.sha(card)))
        receipt=self.capture/'checkpoint_1.txt'; receipt.write_text(receipt.read_text().replace(chain.keyfile(receipt)['source_identity_sha256'],chain.sha(src)))
        with self.assertRaisesRegex(ValueError,'observer identity mismatch'): chain.load(self.capture,1)
    def test_prefix_and_missing_receipt(self):
        chain.load(self.capture,1)
        (self.capture/'checkpoint_1.txt').unlink()
        with self.assertRaises(FileNotFoundError): chain.load(self.capture,1)
        with self.assertRaises(ValueError): chain.scalar_prefix(self.capture/'scalar.txt',2,.1)
    def test_hash_corruption(self):
        (self.capture/'frame_1.txt').write_text('bad')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'): chain.load(self.capture,1)
    def test_prepare_verify_and_bad_seam(self):
        child=self.root.parent/'child'; chain.prepare(self.capture,1,child,1)
        d=self.make_capture(child)
        (d/'scalar.txt').write_bytes((d/'scalar.txt').read_bytes().removesuffix(b'2 partial'))
        report=self.root.parent/'report.json'; chain.verify([self.capture,d],report)
        self.assertEqual(len(json.loads(report.read_text())['rows']),3)
        for i in (0,1):
            f=d/f'frame_{i}.txt'; f.write_text(f.read_text().replace('0 0 0 0 0 0 0 0 0 0','0 0.01 0 0 0 0 0 0 0 0'))
            r=d/f'checkpoint_{i}.txt'; s=r.read_text().splitlines(); r.write_text('\n'.join('frame_sha256 '+chain.sha(f) if x.startswith('frame_sha256 ') else x for x in s)+'\n')
        with self.assertRaisesRegex(ValueError,'seam mismatch'): chain.verify([self.capture,d],report)
    def test_prefix_tampering_and_false_root(self):
        f=self.capture/'scalar.txt'; f.write_bytes(f.read_bytes().replace(b'0 0.0 ',b'0 0.00 ',1))
        with self.assertRaisesRegex(ValueError,'prefix SHA mismatch'): chain.load(self.capture,1)
        (self.root/'parent_receipt.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'cannot be used as root'): chain.verify([self.capture,self.root.parent/'other'],self.root/'x.json')
if __name__=='__main__': unittest.main()
