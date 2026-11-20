#!/usr/bin/env python3
"""Finite-time connected R164 tensor diagnostics; not a conductivity calculation."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fields(path):
    result = {}
    for line in path.read_text(encoding='utf8').splitlines():
        if line.startswith('atom '):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            if parts[0] in result:
                raise ValueError(f'duplicate {parts[0]} in {path}')
            result[parts[0]] = parts[1]
    return result


def numeric(tokens, label, count):
    if tokens.count(label) != 1:
        raise ValueError(f'missing or duplicated {label}')
    at = tokens.index(label) + 1
    values = np.asarray(tokens[at:at + count], dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f'invalid {label}')
    return values


def load_trace(directory):
    directory = Path(directory)
    meta_path, identity_path = directory/'metadata.txt', directory/'source_identity.txt'
    scalar_path, complete_path = directory/'scalar.txt', directory/'COMPLETE'
    observer_path = directory/'observer.txt'
    meta, identity, observer = fields(meta_path), fields(identity_path), fields(observer_path)
    if meta.get('schema') != 'r164_diagnostic_v1' or meta.get('time_convention') != 'step_times_fixed_dt_start_zero':
        raise ValueError('unsupported metadata schema or time convention')
    marker = complete_path.read_text(encoding='utf8').split()
    if len(marker) != 3 or marker[:2] != ['r164_diagnostic_v1', 'final_step']:
        raise ValueError('invalid completion marker')
    final = int(marker[2]); dt = float(meta['dt_fs']); temperature = float(meta['temperature_K'])
    cell = np.asarray(meta['cell_angstrom'].split(), dtype=float)
    if final < 1 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(temperature) or temperature <= 0 or cell.shape != (3,) or not np.isfinite(cell).all() or np.any(cell <= 0):
        raise ValueError('invalid completion or thermodynamic metadata')
    for key in ('model_sha256', 'run_in_sha256'):
        if not re.fullmatch(r'[0-9a-fA-F]{64}', identity.get(key, '')):
            raise ValueError(f'missing source {key}')
    if identity.get('bead_force_backend') not in ('native_GPUMD', 'WPE'):
        raise ValueError('unidentified bead-force backend')
    if observer.get('model_sha256', '').lower() != identity['model_sha256'].lower():
        raise ValueError('observer model identity differs')
    if (meta.get('reference_id') != 'R27_fixed_2OH_R157' or
        meta.get('reference_source_sha256', '').lower() != '1d46365935f48e390d08e69e8564a3adc692d439b9472b4250914b22144a2eaa'):
        raise ValueError('unsupported fixed reference identity')
    signature = (identity['bead_force_backend'], identity['model_sha256'].lower(), identity['run_in_sha256'].lower(),
                 meta.get('reference_id'), meta['reference_source_sha256'].lower(), meta.get('P'), meta.get('N'),
                 meta['temperature_K'], meta['dt_fs'], meta['cell_angstrom'], observer.get('pppm_spacing'))
    if not signature[3] or not signature[-1]:
        raise ValueError('incomplete reference or observer identity')
    rows = []
    with scalar_path.open(encoding='utf8', newline='') as scalar:
        for line in scalar:
            if not line.endswith('\n'):
                raise ValueError('partial scalar line')
            if line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 21:
                raise ValueError('short scalar line')
            step = int(tokens[0]); time = float(tokens[1])
            if step != len(rows) or not np.isfinite(time) or not np.isclose(time, step*dt, rtol=1e-12, atol=1e-10):
                raise ValueError('noncontiguous or mistimed scalar')
            species = np.asarray(tokens[9:21], dtype=float)
            b = numeric(tokens, 'B_UNVERIFIED', 3)
            r27 = numeric(tokens, 'R27_UNVERIFIED', 7)
            j = numeric(tokens, 'J2_UNVERIFIED', 3)
            d = r27[4:7]
            if not np.isfinite(species).all() or not np.allclose(j, b+d, rtol=2e-14, atol=2e-14):
                raise ValueError('nonfinite species or J2 identity mismatch')
            rows.append(np.concatenate((b, d, species)))
    if len(rows) != final + 1:
        raise ValueError('scalar length does not match COMPLETE')
    paths = (meta_path, identity_path, observer_path, scalar_path, complete_path)
    return np.asarray(rows), signature, {p.name: digest(p) for p in paths}, dt, temperature, cell


def correlations(traces, mean, max_lag):
    result = []
    counts = []
    for lag in range(max_lag + 1):
        total = np.zeros((18, 18)); count = 0
        for trace in traces:
            n = len(trace) - lag
            if n <= 0:
                raise ValueError('max lag must be less than each trace length')
            a = trace[:n] - mean; b = trace[lag:lag+n] - mean
            total += a.T @ b; count += n
        result.append(total/count); counts.append(count)
    return np.asarray(result), counts


def integral(c, dt):
    out = np.zeros_like(c)
    out[1:] = np.cumsum((c[:-1]+c[1:])*(0.5*dt), axis=0)
    return out


def eh_diagnostic(traces, dt, window):
    bins=[(trace[:-1]+trace[1:])*0.5 for trace in traces]
    total_bins=sum(len(x) for x in bins)
    mean=sum((x.sum(axis=0) for x in bins),np.zeros(18))/total_bins
    origin_count=sum(max(0,len(x)-window+1) for x in bins)
    result={'window_intervals':window, 'window_fs':window*dt, 'interval_bin_count':total_bins,
            'origin_count':origin_count, 'interval_mean':mean.tolist(),
            'units':'outer product of vector units multiplied by fs; no volume or temperature normalization',
            'scope':'regular-only; no event increments, no cross-trace windows, no conductivity',
            'centering':'EH increments centered over actual origins; triangular correlation uses common interval-bin mean',
            'eh_covariance_over_2t':None, 'triangular_correlation_integral':None, 'finite_sample_residual':None}
    if origin_count < 2 or any(len(x)<window for x in bins):
        result['support']='INSUFFICIENT_CONTIGUOUS_ORIGINS'
        return result
    increments=[]
    for x in bins:
        prefix=np.vstack((np.zeros((1,18)),np.cumsum(x*dt,axis=0)))
        increments.append(prefix[window:]-prefix[:-window])
    increments=np.concatenate(increments,axis=0)
    centered=increments-increments.mean(axis=0)
    eh=centered.T@centered/(origin_count*2*window*dt)
    numerator=np.zeros((18,18)); counts=[]
    for lag in range(window):
        term=np.zeros((18,18)); count=0
        for x in bins:
            n=len(x)-lag; z=x-mean
            term+=z[:n].T@z[lag:lag+n]; count+=n
        c=term/count; counts.append(count)
        numerator += (0.5*c if lag==0 else (1-lag/window)*(c+c.T)*0.5)
    triangular=dt*numerator
    result.update(support='FINITE_SAMPLE_ONLY', interval_sample_counts=counts,
                  eh_covariance_over_2t=eh.tolist(), triangular_correlation_integral=triangular.tolist(),
                  finite_sample_residual=(eh-triangular).tolist())
    return result

def analyze(directories, max_lag):
    if max_lag <= 0:
        raise ValueError('max lag must be positive')
    if len({Path(d).resolve() for d in directories}) != len(directories):
        raise ValueError('duplicate input directory')
    loaded = [load_trace(d) for d in directories]
    scalar_hashes=[item[2]['scalar.txt'] for item in loaded]
    if len(set(scalar_hashes)) != len(scalar_hashes):
        raise ValueError('identical scalar inputs would be double counted')
    if any(item[1] != loaded[0][1] for item in loaded[1:]):
        raise ValueError('backend, source/configuration, reference or thermodynamic identity mismatch')
    traces = [item[0] for item in loaded]
    if any(len(x) <= max_lag for x in traces):
        raise ValueError('max lag must be shorter than every trace')
    mean = sum((x.sum(axis=0) for x in traces), np.zeros(18))/sum(map(len, traces))
    c, counts = correlations(traces, mean, max_lag)
    bb, bd, db, dd = c[:, :3, :3], c[:, :3, 3:6], c[:, 3:6, :3], c[:, 3:6, 3:6]
    j_direct = []
    for lag in range(max_lag+1):
        numerator = np.zeros((3,3)); count = 0
        for trace in traces:
            z = trace[:,:3] + trace[:,3:6] - mean[:3] - mean[3:6]
            n = len(z)-lag; numerator += z[:n].T @ z[lag:lag+n]; count += n
        j_direct.append(numerator/count)
    j_direct = np.asarray(j_direct); blocks = np.stack((bb,bd,db,dd),axis=1)
    block_vectors = []
    for trace in traces:
        half = len(trace)//2
        if half > max_lag:
            for part in (trace[:half], trace[half:2*half]):
                cb, _ = correlations([part], mean, max_lag)
                four = np.stack((cb[:,:3,:3],cb[:,:3,3:6],cb[:,3:6,:3],cb[:,3:6,3:6]),axis=1)
                block_vectors.append(integral(four, loaded[0][3])[-1].reshape(36))
    covariance = np.cov(np.asarray(block_vectors),rowvar=False,ddof=1).tolist() if len(block_vectors)>=2 else None
    return {
        'schema': 'r164_provisional_correlations_v1',
        'status': 'FINITE_TIME_DIAGNOSTIC_NOT_CONDUCTIVITY',
        'scope': 'regular-path candidate; no event increments inferred; physical release and complete event history required',
        'axes': 'C[lag,i,j] uses time-zero i and later-time j; vector order b_xyz,d2_xyz,N_O_xyz,N_H_xyz,N_Na_xyz,N_Cl_xyz',
        'units': {'b_d2':'eV angstrom/fs','species':'angstrom/fs','correlation':'outer product of vector units',
                  'integral':'correlation units multiplied by fs; no volume or temperature normalization'},
        'eh_regular_only':eh_diagnostic(traces,loaded[0][3],max_lag),
        'dt_fs': loaded[0][3], 'temperature_K': loaded[0][4], 'cell_angstrom': loaded[0][5].tolist(),
        'max_lag': max_lag, 'sample_counts': counts, 'mean_shared_all_samples': mean.tolist(),
        'C_full_18x18': c.tolist(), 'C_bb':bb.tolist(), 'C_bd':bd.tolist(), 'C_db':db.tolist(), 'C_dd':dd.tolist(),
        'C_J2_direct':j_direct.tolist(), 'J2_identity_max_abs':float(np.max(np.abs(j_direct-bb-bd-db-dd))),
        'integrated_full_18x18_trapezoid':integral(c,loaded[0][3]).tolist(),
        'integrated_four_blocks_trapezoid':integral(blocks,loaded[0][3]).tolist(),
        'paired_block_covariance_36x36':covariance, 'paired_block_count':len(block_vectors),
        'paired_block_note':'two nonoverlapping equal blocks per eligible trace, common global mean; blocks not asserted independent',
        'inputs':[{'directory':str(d),'sha256':entry[2]} for d,entry in zip(directories,loaded)],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--max-lag',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.directories,args.max_lag)
    with args.output.open('x',encoding='utf8') as output:
        json.dump(result,output,indent=2,allow_nan=False); output.write('\n')

if __name__=='__main__': main()
