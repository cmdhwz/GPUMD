"""CPU/source checks for HAC output timing and fixed-step input protection."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAC = (ROOT / "src/measure/hac.cu").read_text(encoding="utf-8")
RUN = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf-8")


class HacOutputContractTest(unittest.TestCase):
    def test_output_time_is_mean_of_included_lags(self):
        output = HAC[HAC.index("// major data") :]
        match = re.search(r'fprintf\(fid, "%25\.15e", (.+)\);', output)
        self.assertIsNotNone(match)
        expression = match.group(1)
        for interval in (1, 10, 100):
            for first in (0, interval, 30 * interval):
                for sample_dt in (0.00025, 0.005):
                    with self.subTest(interval=interval, first=first, dt=sample_dt):
                        actual = eval(expression, {"__builtins__": {}}, {
                            "nc": first,
                            "output_interval": interval,
                            "dt_in_ps": sample_dt,
                        })
                        expected = sum(range(first, first + interval)) / interval * sample_dt
                        self.assertAlmostEqual(actual, expected, places=14)

    def test_adaptive_step_is_rejected_before_initialization(self):
        perform = RUN[RUN.index("void Run::perform_a_run(") :]
        condition = "centroid_force_hac != nullptr && max_distance_per_step > 0.0"
        guard = f"if ({condition})"
        self.assertTrue(guard in perform, "missing fixed-step HAC guard")
        start = perform.index(guard)
        self.assertLess(perform.index("centroid_force_hac = dynamic_cast<HAC*>"), start)
        self.assertLess(start, perform.index("integrate.initialize("))
        self.assertLess(start, perform.index("measure.pre_run("))
        body = perform[start : perform.index("}", start)]
        self.assertIn("PRINT_INPUT_ERROR", body)
        self.assertIn("compute_hac requires a fixed integration time step", body)
        expression = condition.replace("nullptr", "None").replace("&&", "and")
        for has_hac, adaptive, expected in (
            (True, 0.0, False),
            (True, 0.01, True),
            (False, 0.0, False),
            (False, 0.01, False),
        ):
            with self.subTest(has_hac=has_hac, adaptive=adaptive):
                self.assertEqual(eval(expression, {"__builtins__": {}}, {
                    "centroid_force_hac": object() if has_hac else None,
                    "max_distance_per_step": adaptive,
                }), expected)

    def test_full_resolution_integration_and_virial_mapping_remain_intact(self):
        output = HAC[HAC.index("// major data") :]
        self.assertLess(output.index("gpu_find_hac<<<"), output.index("find_rtc("))
        self.assertLess(output.index("find_rtc("), output.index("double hac_ave["))
        self.assertIn("hac_ave[k] += hac_cpu[count];", output)
        self.assertIn("rtc_ave[k] += rtc[count];", output)
        self.assertIn("hac_ave[m] /= output_interval;", output)
        self.assertIn("rtc_ave[m] /= output_interval;", output)
        start = HAC.rindex("static void compute_full_heat_per_atom(")
        wrapper = HAC[start : HAC.index("GPU_CHECK_KERNEL", start)]
        offsets = re.findall(
            r"virial_per_atom\.data\(\)(?:\s*\+\s*N\s*\*\s*(\d+))?\s*,",
            wrapper,
        )
        self.assertEqual([int(x) if x else 0 for x in offsets], [0, 3, 4, 6, 1, 5, 7, 8, 2])


if __name__ == "__main__":
    unittest.main()
