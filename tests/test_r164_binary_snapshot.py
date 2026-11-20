import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from r164_frame_to_npz import read_binary_frame, binary_lineage, main

try:
    import zstandard as zstd
except ImportError:
    zstd=None

class BinarySnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.fields={'dt_fs':['0.1'],'buffered_frame_stride':['1'],'buffered_snapshot_format':['binary_v1'],
                     'cell_angstrom':['10','10','10']}
        (self.root/'COMPLETE').write_text('r164_diagnostic_v1 final_step 2\n')
        (self.root/'metadata.txt').write_text('metadata\n')
        (self.root/'source_identity.txt').write_text('source\n')
    def tearDown(self): self.tmp.cleanup()
    def make_file(self,width,values):
        path=self.root/'snapshots.bin'; code={1:'b',2:'h',4:'i',8:'q'}[width]
        with path.open('wb') as out:
            out.write(struct.pack('<8sIIII',b'R164SNAP',1,1,2,3))
            for step in range(3):
                out.write(struct.pack('<II',step,width))
                out.write(struct.pack('<d',step+1.))
                q=np.array(values,dtype=float)*10
                out.write(q.astype('<f8').tobytes())
                out.write(np.arange(6,dtype='<f8').tobytes())
                out.write((np.arange(6,dtype='<f8')+10).tobytes())
                out.write(struct.pack('<6'+code,*values))
        return path
    def test_widths_and_off_grid_step(self):
        for width,extreme in ((1,127),(2,32767),(4,2147483647),(8,9223372036854775807)):
            values=[-extreme-1,extreme,0,1,-1,0]
            self.make_file(width,values)
            q,p,f,u,images,header,first,dt=read_binary_frame(self.root,1,1,2,self.fields)
            np.testing.assert_array_equal(images.reshape(1,2,3).transpose(0,2,1).ravel(),values)
            self.assertEqual(u[0],2.); self.assertEqual(dt,.1)
            identity=binary_lineage(self.root,header,first)['capture_id_sha256']
            other=read_binary_frame(self.root,2,1,2,self.fields)
            self.assertEqual(identity,binary_lineage(self.root,other[5],other[6])['capture_id_sha256'])
    def test_cli_exports_off_sample_grid(self):
        self.make_file(1,[0]*6)
        (self.root/'metadata.txt').write_text('P 1\nN 2\ndt_fs 0.1\nbuffered_frame_stride 1\nbuffered_sample_stride 3\nbuffered_snapshot_format binary_v1\ncell_angstrom 10 10 10\natom 0 O 16\natom 1 H 1\n')
        output=self.root/'phase.npz'
        with patch.object(sys,'argv',['r164_frame_to_npz',str(self.root),'1',str(output)]): main()
        with np.load(output,allow_pickle=False) as phase:
            self.assertEqual(int(phase['step']),1)
            self.assertAlmostEqual(float(phase['time_fs']),.1)
            np.testing.assert_array_equal(phase['force_images'],np.zeros((1,2,3),dtype=np.int64))

    @unittest.skipIf(zstd is None,'optional Python zstandard unavailable')
    def test_compressed_random_access_and_corruption(self):
        self.fields['buffered_snapshot_format']=['zstd_shuffle_v2']
        widths=[1,2,4]; values=[[-3,4,0,1,-1,0],[-129,128,0,1,-1,0],[-32769,32768,0,1,-1,0]]
        records=[]
        for step,(width,images) in enumerate(zip(widths,values)):
            code={1:'b',2:'h',4:'i'}[width]
            floats=np.r_[np.array([step+1.]),np.array(images,dtype=float)*10,np.arange(6.),np.arange(6.)+10].astype('<f8')
            shuffled=np.frombuffer(floats.tobytes(),dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
            raw=shuffled+struct.pack('<6'+code,*images)
            payload=zstd.ZstdCompressor(level=1,write_checksum=True).compress(raw)
            records.append(struct.pack('<IIQQ',step,width,len(raw),len(payload))+payload)
        offsets=[]; cursor=24+8*len(records)
        for record in records:
            offsets.append(cursor); cursor+=len(record)
        path=self.root/'snapshots.bin'
        original=struct.pack('<8sIIII',b'R164SNAP',2,1,2,3)+struct.pack('<3Q',*offsets)+b''.join(records)
        path.write_bytes(original)
        result=read_binary_frame(self.root,1,1,2,self.fields)
        np.testing.assert_array_equal(result[4].transpose(0,2,1).ravel(),values[1])
        self.assertEqual(result[3][0],2.)
        for broken in (original[:-1], original[:24]+struct.pack('<Q',0)+original[32:]):
            path.write_bytes(broken)
            with self.assertRaises(ValueError): read_binary_frame(self.root,1,1,2,self.fields)
        broken=bytearray(original); broken[offsets[1]+24+len(records[1])-25]^=1
        path.write_bytes(broken)
        with self.assertRaises(ValueError): read_binary_frame(self.root,1,1,2,self.fields)

    @unittest.skipIf(zstd is None,'optional Python zstandard unavailable')
    def test_compressed_index_skips_unrelated_headers(self):
        self.fields['buffered_snapshot_format']=['zstd_shuffle_v2']
        (self.root/'COMPLETE').write_text('r164_diagnostic_v1 final_step 6\n')
        floats=np.zeros(19,dtype='<f8')
        shuffled=np.frombuffer(floats.tobytes(),dtype=np.uint8).reshape(-1,8).T.copy().tobytes()
        raw=shuffled+bytes(6)
        payload=zstd.ZstdCompressor(write_checksum=True).compress(raw)
        records=[struct.pack('<IIQQ',i,1,len(raw),len(payload))+payload for i in range(7)]
        offsets=[]; cursor=24+8*len(records)
        for record in records:
            offsets.append(cursor); cursor+=len(record)
        path=self.root/'snapshots.bin'
        path.write_bytes(struct.pack('<8sIIII',b'R164SNAP',2,1,2,7)+struct.pack('<7Q',*offsets)+b''.join(records))
        seeks=[]; original=Path.open
        class Logged:
            def __init__(self,wrapped): self.wrapped=wrapped
            def __enter__(self): return self
            def __exit__(self,*args): self.wrapped.close()
            def read(self,*args): return self.wrapped.read(*args)
            def seek(self,*args): seeks.append(args[0]); return self.wrapped.seek(*args)
        def opening(path,*args,**kwargs):
            result=original(path,*args,**kwargs)
            return Logged(result) if path.name=='snapshots.bin' else result
        with patch.object(Path,'open',opening): read_binary_frame(self.root,3,1,2,self.fields)
        for i in (1,2,4,5): self.assertNotIn(offsets[i],seeks)

    def test_canonical_edge_tolerance(self):
        path=self.make_file(1,[0]*6)
        data=bytearray(path.read_bytes())
        second_q=24+(8+8+3*6*8+6)+8+8
        struct.pack_into('<d',data,second_q,10+1.05e-8)
        path.write_bytes(data)
        read_binary_frame(self.root,1,1,2,self.fields)
        struct.pack_into('<d',data,second_q,10+2e-8)
        path.write_bytes(data)
        with self.assertRaises(ValueError): read_binary_frame(self.root,1,1,2,self.fields)

    def test_rejects_corrupt_width_truncation_and_step(self):
        path=self.make_file(1,[0]*6); original=path.read_bytes()
        for data in (original[:-1],original+b'X',original[:28]+struct.pack('<I',3)+original[32:],
                     original[:24]+struct.pack('<I',1)+original[28:]):
            path.write_bytes(data)
            with self.assertRaises(ValueError): read_binary_frame(self.root,1,1,2,self.fields)

if __name__=='__main__': unittest.main()
