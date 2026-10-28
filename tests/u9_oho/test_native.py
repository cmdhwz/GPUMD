import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from rpmd_ja_cpp_test_support import find_cpp_compiler, compile_command

class NativeMathTest(unittest.TestCase):
    def test_cpp_config_and_math(self):
        compiler = find_cpp_compiler()
        if not compiler:
            self.skipTest('C++ compiler not found; native parser and math have not run')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests' / 'u9_oho') as work:
            binary = Path(work) / ('u9_test.exe' if os.name == 'nt' else 'u9_test')
            subprocess.run(compile_command(compiler, 17, ROOT/'tests/u9_oho/test_native_math.cpp', binary), cwd=work, check=True)
            subprocess.run([str(binary), str(ROOT/'tests/u9_oho/synthetic.cfg')], check=True)

if __name__ == '__main__': unittest.main()
