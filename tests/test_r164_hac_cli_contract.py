from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class HacCliContract(unittest.TestCase):
    def test_resolves_stride_before_run_in_either_command_order(self):
        source = (ROOT / "src/measure/r164_rpmd.cu").read_text(encoding="utf8")
        run = (ROOT / "src/main_gpumd/run.cu").read_text(encoding="utf8")
        parse = source.split("void R164_RPMD::parse_hac(", 1)[1].split("void R164_RPMD::parse(", 1)[0]
        self.assertIn("count!=3 && count!=4", parse)
        self.assertIn("sample_stride_!=hac_sample_stride_", parse)
        self.assertIn("if (!sample_stride_) sample_stride_=hac_sample_stride_", parse)
        self.assertIn("if (!sample_stride_)", parse)
        self.assertLess(run.index("r164_rpmd.validate_hac(number_of_steps)"), run.index("r164_rpmd.prepare_buffered(number_of_steps"))

if __name__ == "__main__": unittest.main()
