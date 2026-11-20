#!/usr/bin/env python3
"""Convert an R164 debug frame and metadata into an R161-style NPZ."""
import argparse
import hashlib
import struct
from pathlib import Path
import numpy as np


def read_frame(path, P, N):
    q = np.empty((P, N, 3)); p = np.empty_like(q); f = np.empty_like(q); u = np.empty(P)
    with (path).open(encoding='utf-8') as source:
        for s in range(P):
            header=source.readline().split()
            if len(header)!=3 or header[:2] != ['bead',str(s)]: raise ValueError('invalid bead header')
            u[s]=float(header[2])
            for i in range(N):
                row=source.readline().split()
                if len(row)!=10 or int(row[0])!=i: raise ValueError('invalid atom row')
                values=np.array(row[1:],dtype=float)
                q[s,i],p[s,i],f[s,i]=values[:3],values[3:6],values[6:]
        if source.read().strip(): raise ValueError('trailing frame data')
    if not all(np.isfinite(x).all() for x in (q,p,f,u)): raise ValueError('nonfinite frame')
    return q,p,f,u

def read_binary_frame(directory, step, P, N, fields):
    if step < 0 or fields.get('buffered_snapshot_format') not in (['binary_v1'],['zstd_shuffle_v2']):
        raise ValueError('invalid binary snapshot metadata')
    dt = float(fields['dt_fs'][0])
    stride = int(fields['buffered_frame_stride'][0])
    marker = (directory/'COMPLETE').read_text(encoding='utf8').split()
    if len(marker)!=3 or marker[:2]!=['r164_diagnostic_v1','final_step']:
        raise ValueError('incomplete binary snapshot capture')
    final=int(marker[2])
    if not np.isfinite(dt) or dt<=0 or stride<=0 or final<0 or step>final:
        raise ValueError('invalid binary snapshot cadence')
    count=1+final//stride+(final%stride!=0)
    path=directory/'snapshots.bin'
    size=path.stat().st_size
    first=None; selected=None
    with path.open('rb') as source:
        header=source.read(24)
        if len(header)!=24:
            raise ValueError('truncated binary snapshot header')
        magic,version,beads,atoms,records=struct.unpack('<8sIIII',header)
        expected_version=1 if fields['buffered_snapshot_format']==['binary_v1'] else 2
        if (magic!=b'R164SNAP' or version!=expected_version or beads!=P or atoms!=N or
                P<1 or N<1 or records!=count):
            raise ValueError('binary snapshot header mismatch')
        if version==2:
            if 8*records>size-24: raise ValueError('truncated binary snapshot index')
            index_bytes=source.read(8*records)
            if len(index_bytes)!=8*records: raise ValueError('truncated binary snapshot index')
            offsets=np.frombuffer(index_bytes,dtype='<u8')
            index_end=24+8*records
            if (int(offsets[0])!=index_end or np.any(offsets>size-24)
                    or np.any(offsets[1:]<=offsets[:-1])
                    or np.any(offsets[1:]-offsets[:-1]<24)):
                raise ValueError('invalid binary snapshot offset index')
            if step%stride==0:
                selected_index=step//stride
            elif step==final:
                selected_index=records-1
            else:
                raise ValueError('requested step is not a stored snapshot')
            for i in sorted({0,selected_index,records-1}):
                pos=int(offsets[i]); source.seek(pos); record=source.read(24)
                if len(record)!=24: raise ValueError('truncated snapshot record header')
                current,width,stored_raw,length=struct.unpack('<IIQQ',record)
                expected=final if i==records-1 else i*stride
                if current!=expected or width not in (1,2,4,8):
                    raise ValueError('binary snapshot step or image width mismatch')
                raw_length=8*P+3*(3*P*N)*8+(3*P*N)*width
                bound=raw_length+(raw_length>>8)+(((128<<10)-raw_length)>>11 if raw_length<(128<<10) else 0)
                if stored_raw!=raw_length or length<1 or length>bound:
                    raise ValueError('invalid compressed snapshot lengths')
                end=int(offsets[i+1]) if i+1<records else size
                if pos+24+length!=end: raise ValueError('compressed snapshot boundary mismatch')
                entry=(pos,current,width,length,24)
                if i==0: first=entry
                if i==selected_index: selected=entry
        else:
            cursor=24
            for i in range(records):
                source.seek(cursor); record=source.read(8)
                if len(record)!=8: raise ValueError('truncated snapshot record header')
                current,width=struct.unpack('<II',record)
                expected=final if i==records-1 else i*stride
                if current!=expected or width not in (1,2,4,8):
                    raise ValueError('binary snapshot step or image width mismatch')
                length=8*P+3*(3*P*N)*8+(3*P*N)*width
                if cursor+8+length>size: raise ValueError('truncated snapshot record')
                entry=(cursor,current,width,length,8)
                if i==0: first=entry
                if current==step: selected=entry
                cursor+=8+length
            if cursor!=size: raise ValueError('trailing binary snapshot data')
        if selected is None: raise ValueError('requested step is not a stored snapshot')
        pos,_,width,length,record_header=selected
        source.seek(pos+record_header)
        data=source.read(length)
        if len(data)!=length: raise ValueError('truncated snapshot payload')
    if version==2:
        try:
            import zstandard as zstd
        except ImportError as e:
            raise ValueError('compressed snapshots require Python zstandard') from e
        raw_length=8*P+3*(3*P*N)*8+(3*P*N)*width
        try:
            if zstd.frame_content_size(data)!=raw_length:
                raise ValueError('compressed frame content size mismatch')
            decoded=zstd.ZstdDecompressor().decompress(data,max_output_size=raw_length,allow_extra_data=False)
        except zstd.ZstdError as e:
            raise ValueError('invalid compressed snapshot or checksum') from e
        if len(decoded)!=raw_length: raise ValueError('compressed snapshot size mismatch')
        double_bytes=8*(P+9*P*N)
        shuffled=np.frombuffer(decoded[:double_bytes],dtype=np.uint8).reshape(8,-1)
        data=shuffled.T.copy().tobytes()+decoded[double_bytes:]
    offset=0
    def take(dtype, count):
        nonlocal offset
        width_bytes=np.dtype(dtype).itemsize
        value=np.frombuffer(data,dtype=dtype,count=count,offset=offset)
        offset+=width_bytes*count
        return value
    u=take('<f8',P).copy()
    def xyz(dtype):
        return take(dtype,3*P*N).reshape(P,3,N).transpose(0,2,1).copy()
    q,p,f=xyz('<f8'),xyz('<f8'),xyz('<f8')
    images=xyz({1:'i1',2:'<i2',4:'<i4',8:'<i8'}[width]).astype(np.int64)
    if not all(np.isfinite(value).all() for value in (q,p,f,u)):
        raise ValueError('nonfinite binary phase')
    cell=np.array(fields['cell_angstrom'],float)
    if cell.shape!=(3,) or not np.isfinite(cell).all() or np.any(cell<=0):
        raise ValueError('invalid binary cell')
    image_offset=images*cell
    canonical=q-image_offset
    tolerance=1.1e-8+4*np.finfo(float).eps*(np.abs(q)+np.abs(image_offset))
    if not np.isfinite(canonical).all() or np.any(canonical < -tolerance) or np.any(canonical>cell+tolerance):
        raise ValueError('binary force image is not canonical')
    return q,p,f,u,images,header,first,dt


def binary_lineage(directory, header, first):
    identity=hashlib.sha256()
    identity.update(str(directory.resolve()).encode('utf8'))
    for path in (directory/'metadata.txt',directory/'source_identity.txt'):
        identity.update(b'\0'+path.name.encode('utf8')+b'\0')
        with path.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''): identity.update(block)
    identity.update(b'\0snapshots.bin:first_record\0'+header)
    with (directory/'snapshots.bin').open('rb') as source:
        source.seek(first[0]); remaining=first[4]+first[3]
        while remaining:
            block=source.read(min(1024*1024,remaining))
            if not block: raise ValueError('truncated first snapshot during identity')
            identity.update(block); remaining-=len(block)
    return {'capture_id_sha256':identity.hexdigest(),'capture_source_directory':str(directory.resolve()),
            'capture_identity_scope':'metadata, source identity, binary header and first record; not full binary SHA'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('step', type=int)
    ap.add_argument('output')
    a = ap.parse_args()
    lines = (a.directory / 'metadata.txt').read_text(encoding='utf-8').splitlines()
    fields = {line.split()[0]: line.split()[1:] for line in lines if not line.startswith('atom ')}
    atoms = [line.split()[1:] for line in lines if line.startswith('atom ')]
    P, N = int(fields['P'][0]), int(fields['N'][0])
    if len(atoms) != N or [int(x[0]) for x in atoms] != list(range(N)):
        raise ValueError('atom metadata ordering mismatch')
    binary=(a.directory/'snapshots.bin').is_file()
    if not binary and fields.get('buffered_snapshot_format') in (['binary_v1'],['zstd_shuffle_v2']):
        raise ValueError('buffered capture has no binary snapshots')
    extra={}
    if binary:
        q,p,f,u,images,header,first,dt_fs=read_binary_frame(a.directory,a.step,P,N,fields)
        time_fs=a.step*dt_fs
        if not np.isfinite(time_fs): raise ValueError('invalid binary frame time')
        extra={'force_images':images,**binary_lineage(a.directory,header,first)}
    else:
        q,p,f,u = read_frame(a.directory / f'frame_{a.step}.txt',P,N)
        matching=[]
        with (a.directory / 'scalar.txt').open(encoding='utf-8') as scalar:
            for line in scalar:
                if not line.endswith('\n'): raise ValueError('partial scalar row without newline')
                if line.startswith('#'): continue
                columns=line.split()
                if columns and int(columns[0]) == a.step: matching.append(columns)
        if len(matching) != 1: raise ValueError('missing or duplicate scalar frame identity')
        time_fs=float(matching[0][1]); dt_fs=float(fields['dt_fs'][0])
        if (not np.isfinite(time_fs) or not np.isfinite(dt_fs) or dt_fs <= 0
            or not np.isclose(time_fs, a.step * dt_fs, rtol=1e-12, atol=1e-10)):
            raise ValueError('invalid or aliased frame time/step')
        source_paths=[a.directory/'metadata.txt',a.directory/'source_identity.txt',a.directory/'frame_0.txt']
        if all(path.is_file() for path in source_paths):
            identity=hashlib.sha256()
            directory=str(a.directory.resolve())
            identity.update(directory.encode('utf8'))
            for path in source_paths:
                identity.update(b'\0'+path.name.encode('utf8')+b'\0')
                with path.open('rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''): identity.update(block)
            extra={'capture_id_sha256':identity.hexdigest(),'capture_source_directory':directory}
    with open(a.output,'xb') as target:
        np.savez_compressed(target,q_A=q,p_eV_fs_per_A=p,F_eV_per_A=f,U_eV=u,
            mass=np.array([float(x[2]) for x in atoms])*10.18051**2,
            time_fs=time_fs,dt_fs=dt_fs,step=a.step,
            species=np.array([x[1] for x in atoms]),cell=np.diag(np.array(fields['cell_angstrom'],float)),
            atom_row_zero_based=np.arange(N),bead_order_zero_based=np.arange(P),**extra)
if __name__=='__main__': main()
