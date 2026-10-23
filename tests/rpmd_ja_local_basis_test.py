from pathlib import Path
import subprocess
import tempfile
import unittest

if __package__:
    from . import rpmd_ja_cpp_test_support as cpp_test_support
else:
    import rpmd_ja_cpp_test_support as cpp_test_support


ROOT = Path(__file__).resolve().parents[1]
CPP_TEST = ROOT / "tests" / "rpmd_ja_local_basis_test.cpp"
COMPILER = cpp_test_support.find_cpp_compiler()


class LocalBasisTests(unittest.TestCase):
    def test_cpp_runtime(self):
        if COMPILER is None:
            self.skipTest("C++ runtime test skipped: c++, g++, clang++, and cl are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "rpmd_ja_local_basis_test.exe"
            build = subprocess.run(cpp_test_support.compile_command(COMPILER, 11, CPP_TEST, executable),
                                   cwd=directory, capture_output=True, text=True, check=False)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], cwd=directory, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
