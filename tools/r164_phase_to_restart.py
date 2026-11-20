#!/usr/bin/env python3
"""Convert the R161 full-bead NPZ into a GPUMD read_pimd_restart file."""
import argparse
import numpy as np

TIME_UNIT_CONVERSION = 10.18051

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_npz')
    parser.add_argument('output_restart')
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--model-output', help='create matching model.xyz with original mass and row order')
    args = parser.parse_args()
    with np.load(args.input_npz, allow_pickle=False) as data:
        q = np.asarray(data['q_A'], dtype=np.float64)
        p = np.asarray(data['p_eV_fs_per_A'], dtype=np.float64)
        mass = np.asarray(data['mass'], dtype=np.float64)
        species = np.asarray(data['species'])
        cell = np.asarray(data['cell'], dtype=np.float64)
        beads, atoms, _ = q.shape
        if (q.shape != (beads, atoms, 3) or p.shape != q.shape or mass.shape != (atoms,)
            or species.shape != (atoms,) or cell.shape != (3, 3)
            or not np.array_equal(data['atom_row_zero_based'], np.arange(atoms))
            or not np.array_equal(data['bead_order_zero_based'], np.arange(beads))
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p))
            or not np.all(np.isfinite(mass)) or np.any(mass <= 0)
            or not np.all(np.isfinite(cell)) or not np.isfinite(args.temperature) or args.temperature <= 0):
            raise ValueError('invalid R161 phase archive')
        velocity = p / mass[None, :, None]
        amu = mass / TIME_UNIT_CONVERSION**2
        lattice = ' '.join(format(x, '.17g') for x in cell.T.flat)
        if args.model_output:
            q_centroid, v_centroid = q.mean(axis=0), velocity.mean(axis=0)
            with open(args.model_output, 'x', encoding='utf-8', newline='\n') as model:
                model.write(f'{atoms}\npbc="T T T" Lattice="{lattice}" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n')
                for i in range(atoms):
                    values = (*q_centroid[i], amu[i], *v_centroid[i])
                    model.write(str(species[i]) + ' ' + ' '.join(format(x, '.17g') for x in values) + '\n')
        with open(args.output_restart, 'x', encoding='utf-8', newline='\n') as out:
            for s in range(-1, beads):
                positions = q.mean(axis=0) if s == -1 else q[s]
                velocities = velocity.mean(axis=0) if s == -1 else velocity[s]
                role = 'centroid' if s == -1 else 'bead'
                out.write(f'{atoms}\npimd_restart=1 num_beads={beads} bead={s} role={role} temperature={args.temperature:.17g} pbc="T T T" Lattice="{lattice}" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n')
                for i in range(atoms):
                    pos = ' '.join(format(x, '.17g') for x in positions[i])
                    vel = ' '.join(format(x, '.17g') for x in velocities[i])
                    out.write(f'{species[i]} {pos} {amu[i]:.4f} {vel}\n')
if __name__ == '__main__':
    main()
