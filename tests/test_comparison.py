import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check_results.py"


class CheckTests(unittest.TestCase):
    def run_check(self, actual):
        return subprocess.run([sys.executable, str(CHECK), "--actual", str(actual)], capture_output=True, text=True)

    def test_shipped_results_reproduce_every_printed_number(self):
        result = self.run_check(ROOT / "results/expected")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_changed_value_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual = Path(tmp) / "actual"
            shutil.copytree(ROOT / "results/expected", actual)
            path = actual / "twibot22_bag_of_words.csv"
            path.write_text(path.read_text().replace("0.6644", "0.6700"))
            self.assertNotEqual(self.run_check(actual).returncode, 0)


if __name__ == "__main__":
    unittest.main()
