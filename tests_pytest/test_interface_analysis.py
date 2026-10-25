import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from interface_analysis import analyze, estimate_bath_power, read_profile, _local_rows


class InterfaceAnalysisTests(unittest.TestCase):
    def test_bath_slopes_use_system_to_bath_signs(self):
        frames = [
            {'time_ps': float(t), 'bath_source_eV': -2.0 * t, 'bath_sink_eV': 2.0 * t}
            for t in range(6)
        ]
        result = estimate_bath_power(frames, (0.0, 5.0), block_frames=2)
        self.assertAlmostEqual(result['hot_eV_ps'], 2.0)
        self.assertAlmostEqual(result['cold_eV_ps'], -2.0)
        self.assertAlmostEqual(result['power_eV_ps'], 2.0)
        self.assertAlmostEqual(result['balance_fraction'], 0.0)

    def test_profile_groups_bin_rows_into_time_frames(self):
        contents = (
            '# columns step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A '
            'coordinate_center_A material_id atom_count kinetic_energy_eV potential_energy_eV '
            'temperature_K dof samples bath_source_eV bath_sink_eV\n'
            '10 0.1 0.0 0 0 1 0.5 1 2 0.1 -1 300 6 10 -0.2 0.2\n'
            '10 0.1 0.0 1 1 2 1.5 2 2 0.1 -1 300 6 10 -0.2 0.2\n'
            '20 0.2 0.1 0 0 1 0.5 1 2 0.1 -1 300 6 10 -0.4 0.4\n'
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'interface_temperature.out'
            path.write_text(contents, encoding='utf-8')
            profile = read_profile(path)
        self.assertEqual(len(profile['frames']), 2)
        self.assertEqual(len(profile['frames'][0]['bins']), 2)
        self.assertAlmostEqual(profile['frames'][1]['bath_source_eV'], -0.4)

    def test_local_conductivity_converts_gradient_from_per_angstrom(self):
        rows = [
            {'material_id': 1, 'coordinate_center_A': float(x),
             'temperature_K': 300.0 - 0.01 * x, 'temperature_error_K': 1.0}
            for x in range(3)
        ]
        result = _local_rows(
            rows, (0.0, 2.0), {'id': 1, 'name': 'AlAs', 'interface_A': 3.0},
            'left', 'lower', 1.0e9, 0.0, 3)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1]['kappa_local_W_mK'], 10.0)

    def test_analyze_writes_tables_and_dependency_free_errorbar_svg(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            header = (
                '# cross_section_A2 100\n'
                '# axis z\n'
                '# interface_coordinate_A 5\n'
                '# material_lower_id 1\n'
                '# material_upper_id 2\n'
                '# columns step time_ps block_start_ps bin_id coordinate_min_A '
                'coordinate_max_A coordinate_center_A material_id atom_count '
                'kinetic_energy_eV potential_energy_eV temperature_K dof samples '
                'bath_source_eV bath_sink_eV\n')
            for name, materials, temperature in (
                ('hetero', lambda x: 1 if x < 5 else 2,
                 lambda x: 400 - 10 * x if x < 5 else 330 - 5 * (x - 5)),
                ('pure_left', lambda x: 1, lambda x: 350 - 5 * (x - 5)),
                ('pure_right', lambda x: 2, lambda x: 330 - 10 * (x - 5)),
            ):
                lines = [header]
                for step in range(6):
                    for bin_id in range(10):
                        center = bin_id + 0.5
                        lines.append(
                            f'{step} {step} {step} {bin_id} {bin_id} {bin_id + 1} '
                            f'{center} {materials(center)} 10 1 -10 '
                            f'{temperature(center)} 30 10 {-2 * step} {2 * step}\n')
                (root / f'{name}.out').write_text(''.join(lines), encoding='utf-8')

            config = {
                'metadata': {
                    'source_revision': 'test', 'nep_model': 'test.nep',
                    'target_temperature_K': 350, 'source_group': 1, 'sink_group': 2,
                    'boundary_condition': 'periodic',
                },
                'output_dir': 'analysis', 'steady_state_ps': [0, 5],
                'block_frames': 2, 'window_bins': 3, 'source_side': 'lower',
                'interface_coordinate_A': 5, 'max_distance_A': 5,
                'interface_exclusion_A': 0,
                'heterostructure_run': 'hetero',
                'runs': {
                    name: {'profile': f'{name}.out', 'area_A2': 100, 'power_fraction': 1}
                    for name in ('hetero', 'pure_left', 'pure_right')
                },
                'interface': {
                    'left': {'id': 1, 'name': 'AlAs', 'region_A': [0, 5],
                             'fit_A': [1.5, 3.5]},
                    'right': {'id': 2, 'name': 'GaAs', 'region_A': [5, 10],
                              'fit_A': [6.5, 8.5]},
                },
                'pure_references': {
                    'left': {'run': 'pure_left', 'material_id': 1,
                             'fit_A': [1.5, 3.5], 'reference_coordinate_A': 5},
                    'right': {'run': 'pure_right', 'material_id': 2,
                              'fit_A': [6.5, 8.5], 'reference_coordinate_A': 5},
                },
                'segments': [
                    {'run': 'hetero', 'side': 'left', 'material_id': 1,
                     'coordinate1_A': 1.5, 'coordinate2_A': 3.5},
                    {'run': 'hetero', 'side': 'right', 'material_id': 2,
                     'coordinate1_A': 6.5, 'coordinate2_A': 8.5},
                ],
            }
            config_path = root / 'analysis.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            outputs = analyze(config_path)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.exists() for path in outputs))
            segment_data = outputs[-1].read_text(encoding='utf-8')
            self.assertIn('R_seg_m2K_W', segment_data)
            self.assertEqual(len([line for line in segment_data.splitlines()
                                  if line and not line.startswith('#')]), 2)
            for table in (*outputs[:3], outputs[-1]):
                lines = table.read_text(encoding='utf-8').splitlines()
                columns = next(line for line in lines if line.startswith('# columns ')).split()[2:]
                data = [line.split() for line in lines if line and not line.startswith('#')]
                self.assertTrue(all(len(row) == len(columns) for row in data), table.name)
            svg = outputs[3].read_text(encoding='utf-8')
            self.assertIn('<svg', svg)
            self.assertIn('interface-analysis', svg)
            self.assertIn('Excess resistance', svg)
            self.assertIn('stroke="#2369a8"', svg)


if __name__ == '__main__':
    unittest.main()
