from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples/r164_ja_reference/ja_reference_r27.txt"

class ReferenceParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        existing = os.environ.get("R164_JA_TEST_DRIVER")
        if existing:
            cls.exe = Path(existing)
            return
        compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.exe = Path(cls.temp.name) / ("driver.exe" if os.name == "nt" else "driver")
        subprocess.run([compiler, "-std=c++14", str(ROOT / "tests/r164_ja_reference_driver.cpp"), "-o", str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_values_bitwise(self):
        path = Path(self.temp.name) / "reference.txt"
        contents = TEMPLATE.read_bytes()
        path.write_bytes(contents)
        subprocess.run([str(self.exe), str(path), "compare", hashlib.sha256(contents).hexdigest()], check=True)

    def test_bad_inputs_rejected(self):
        text = TEMPLATE.read_text(encoding="utf8")
        bad = [text.replace("r164_ja_reference 1", "r164_ja_reference 2"),
               text.replace("rows 715 3382 5674", "rows 715 715 5674"),
               text.replace("rows 715 3382 5674", "rows 715 3382 5896"),
               text.replace("temperature_K 300", "temperature_K -1"),
               text.replace("cell_A 29.3104", "cell_A nan", 1),
               text.replace("potential_sha256 fb5", "potential_sha256 Zb5"),
               text.replace("masses_eV_fs2_A2 1658", "masses_eV_fs2_A2 -1658"),
               text.replace("basis 0.0035724334610606951", "basis 1", 1),
               text.rsplit(" ", 1)[0], text + "extra\n"]
        for i, content in enumerate(bad):
            path = Path(self.temp.name) / f"bad{i}.txt"
            path.write_text(content, encoding="utf8")
            with self.subTest(case=i):
                self.assertNotEqual(subprocess.run([str(self.exe), str(path)], capture_output=True).returncode, 0)

if __name__ == "__main__":
    unittest.main()
