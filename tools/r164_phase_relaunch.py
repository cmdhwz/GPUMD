#!/usr/bin/env python3
"""Reinitialize a diagnostic phase from a committed R164 frame, with parent receipt."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('parent', type=Path)
    ap.add_argument('step', type=int)
    ap.add_argument('destination', type=Path)
    args = ap.parse_args()
    if args.step < 0:
        ap.error('step must be nonnegative')
    meta = args.parent / 'metadata.txt'
    frame = args.parent / f'frame_{args.step}.txt'
    scalar = args.parent / 'scalar.txt'
    identity = args.parent / 'source_identity.txt'
    for path in (meta, frame, scalar, identity):
        if not path.is_file():
            raise ValueError(f'missing committed parent file: {path}')
    fields = {line.split()[0]: line.split()[1:] for line in meta.read_text(encoding='utf8').splitlines() if not line.startswith('atom ')}
    temperature = float(fields['temperature_K'][0])
    args.destination.mkdir()
    phase = args.destination / 'parent_phase.npz'
    subprocess.run([sys.executable, str(HERE/'r164_frame_to_npz.py'), str(args.parent), str(args.step), str(phase)], check=True)
    with np.load(phase, allow_pickle=False) as data:
        q = data['q_A']; cell = data['cell']; length = np.diag(cell)
        if not np.array_equal(cell, np.diag(length)) or np.any(q < -length) or np.any(q > 2*length):
            raise ValueError('lift outside one-wrap range; cannot reinitialize safely without a verified image-aware input path')
        time_fs = float(data['time_fs']); dt_fs = float(data['dt_fs'])
    model = args.destination/'model.xyz'; restart = args.destination/'restart_beads.xyz'
    subprocess.run([sys.executable, str(HERE/'r164_phase_to_restart.py'), str(phase), str(restart),
                    '--temperature', str(temperature), '--model-output', str(model)], check=True)
    receipt = {
        'schema': 'r164_diagnostic_phase_relaunch_v1', 'scope': 'fresh_run_reinitialization_not_seamless_restart',
        'parent_step': args.step, 'parent_time_fs': time_fs, 'parent_dt_fs': dt_fs,
        'parent_files_sha256': {path.name: digest(path) for path in (meta, frame, scalar, identity)},
        'child_files_sha256': {path.name: digest(path) for path in (phase, model, restart)},
        'parent_source_identity': identity.read_text(encoding='utf8').splitlines(),
        'missing': ['continuing_step_time_identity', 'neighbor_and_pppm_cache_history', 'event_history', 'rng_and_thermostat_history'],
        'required_check': 'compare uninterrupted and fresh-cache trajectories before any continuity claim',
    }
    temporary = args.destination/'parent_receipt.json.tmp'
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(args.destination/'parent_receipt.json')

if __name__ == '__main__':
    main()
