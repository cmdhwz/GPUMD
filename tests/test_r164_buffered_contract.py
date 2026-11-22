from pathlib import Path
import unittest
import bisect
import math
import random
ROOT=Path(__file__).resolve().parents[1]

class BufferedContract(unittest.TestCase):
    def test_skipped_step_preserves_sequence_before_transfer(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        body=text.split('void R164_RPMD::output(',1)[1]
        self.assertLess(body.index('offset_[j]=q_[j]-current[j]'),body.index('if (buffered_ && !sample && !snapshot)'))
        self.assertLess(body.index('if (buffered_ && !sample && !snapshot)'),body.index('r164_gather_sample<<<'))
        self.assertIn('drift_pending_=false; ++next_output_step_',body)
        self.assertIn('if (observer_ && sample)',body)
        self.assertIn('copy_to_host(atom.force_beads[bead]',body)
        self.assertLess(body.index('shadow_stage(atom,SHADOW_POST_KICK)'),body.index('if (buffered_ && !sample && !snapshot)'))
        self.assertIn('sh.staging.copy_to_host(sh.staging_host.data(),size)',body)
        self.assertIn('pointers!=sh.sample_pointers_host',body)
        stage=text.split('void R164_RPMD::shadow_stage(',1)[1].split('void R164_RPMD::read_positions(',1)[0]
        self.assertIn('if (phase==SHADOW_DRIFT) shadow_->error.fill(0)',stage)
        self.assertIn('if (phase==SHADOW_FORCE) return',stage)
        self.assertEqual(stage.count('copy_to_host(shadow_->error,&error)'),1)
        for code in range(1,6): self.assertIn(f'atomicOr(error,1<<{code})',text)
    def test_projection_sweep_matches_binary_search(self):
        def reference(values, L):
            ordered=sorted(values)
            out=[]
            for x in values:
                left=x-L/2; right=x+L/2
                ln=bisect.bisect_left(ordered,left-1e-9)
                rn=bisect.bisect_left(ordered,right-1e-9)
                guard=(ln<len(ordered) and ordered[ln]<=left+1e-9) or (rn<len(ordered) and ordered[rn]<=right+1e-9)
                out.append((bisect.bisect_left(ordered,left),len(ordered)-bisect.bisect_right(ordered,right),guard))
            return out
        def sweep(values,L):
            ordered=sorted(values); result=[None]*len(values); li=ri=ln=rn=0
            for i in sorted(range(len(values)),key=values.__getitem__):
                x=values[i]; left=x-L/2; right=x+L/2
                while li<len(ordered) and ordered[li]<left: li+=1
                while ri<len(ordered) and not right<ordered[ri]: ri+=1
                while ln<len(ordered) and ordered[ln]<left-1e-9: ln+=1
                while rn<len(ordered) and ordered[rn]<right-1e-9: rn+=1
                guard=(ln<len(ordered) and ordered[ln]<=left+1e-9) or (rn<len(ordered) and ordered[rn]<=right+1e-9)
                result[i]=(li,len(ordered)-ri,guard)
            return result
        rng=random.Random(164)
        cases=[[0.,0.,5.,5.,10.], [0.,5.-1e-9,5.,5.+1e-9,10.],
               [0.,math.nextafter(5.,0.),math.nextafter(5.,10.),10.]]
        cases += [[rng.choice([0.,2.5,5.,7.5,10.])+rng.choice([0.,-1e-9,1e-9]) for _ in range(30)] for _ in range(100)]
        for values in cases: self.assertEqual(reference(values,10.),sweep(values,10.))
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        self.assertIn('std::sort(ordered.begin(),ordered.end()',text)
        self.assertIn('if (guarded[i])',text)
        self.assertIn('ordered[ln].first<left-1e-9',text)
    def test_finish_orders_series_before_complete(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        body=text.split('void R164_RPMD::finish_buffered(',1)[1].split('void R164_RPMD::write_complete(',1)[0]
        self.assertLess(body.index('atomic_commit_new_file(path+".tmp",path)'),body.index('write_complete(buffered_total_)'))
        self.assertIn('unavailable values nan',body)
        self.assertIn('sample_stride_',body)
        self.assertLess(body.index('atomic_commit_new_file(binary_path+".tmp",binary_path)'),body.index('write_complete(buffered_total_)'))
        self.assertIn('if (!snapshots_.empty() || compressed_)',body)
        run=(ROOT/'src/main_gpumd/run.cu').read_text(encoding='utf8')
        self.assertLess(run.index('const auto time_finish'),run.index('r164_rpmd.finish_buffered(atom)'))
        self.assertLess(run.index('r164_rpmd.finish_buffered(atom)'),run.index('measure.post_run('))
        output=text.split('void R164_RPMD::output(',1)[1]
        self.assertIn('} else write_snapshot(step,q_,v_storage,f_storage,energy,force_images_,atom)',output)
        self.assertNotIn('finish_buffered(',output)
        self.assertIn('frame.v.assign(v,v+count*3); frame.f.assign(f,f+count*3)',output)
    def test_centroid_native_layout_and_sampling(self):
        import re
        source=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        hac=(ROOT/'src/measure/hac.cu').read_text(encoding='utf8')
        kernel=source.split('static __global__ void centroid_virial_sum',1)[1].split('void R164_RPMD::prepare_buffered',1)[0]
        offsets=[int(x) for x in re.search(r'offsets\[9\]=\{([^}]+)\}',kernel).group(1).split(',')]
        call=hac.split('static void compute_full_heat_per_atom(',1)[1].split('velocity_per_atom.data(),',1)[0]
        hac_offsets=[0]+[int(x) for x in re.findall(r'virial_per_atom.data\(\) \+ N \* (\d+)',call)]
        self.assertEqual(offsets,hac_offsets)
        matrix=[2,3,5,7,11,13,17,19,23]; velocity=[-2,3,5]
        self.assertEqual([sum(matrix[offsets[3*d+k]]*velocity[k] for k in range(3)) for d in range(3)], [72,40,56])
        self.assertIn('output[3*sample+d]',kernel)
        self.assertIn('centroid_virial_current_->data(),series_.size()',source)
        self.assertLess(source.index('atomic_commit_new_file(current_path+".tmp",current_path)'),source.index('write_complete(buffered_total_)'))

    def test_option_excludes_other_diagnostics(self):
        text=(ROOT/'src/measure/r164_rpmd.cu').read_text(encoding='utf8')
        self.assertIn('buffered_ && next<t.size()',text)
        self.assertIn('!decimal(t[next],sample_stride_)',text)
        self.assertIn('sample_stride_<=0',text)

if __name__=='__main__': unittest.main()
