from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from analysis.common import output_directory

ROOT = Path(__file__).resolve().parents[1]


class OutputPathTests(unittest.TestCase):
    def test_saved_evidence_and_nested_outputs_are_protected(self):
        for name in ('results', 'results/expected', 'results/expected/new-run'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Saved results'):
                output_directory(ROOT / name)

    def test_symlink_cannot_bypass_saved_evidence_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / 'alias'
            link.symlink_to(ROOT / 'results/expected', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Saved results'):
                output_directory(link)

    def test_new_run_rejects_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = output_directory(Path(tmp) / 'run', fresh=True)
            sentinel = path / 'draws.csv'
            sentinel.write_text('prior result\n')
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                output_directory(path, fresh=True)
            self.assertEqual(sentinel.read_text(), 'prior result\n')
            # The non-IO reproduction runner intentionally shares its output directory.
            self.assertEqual(output_directory(path), path)

    def test_phrase_dependency_fails_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-m', 'analysis.phrase_removal_f1',
                '--data-root', tmp, '--output-dir', str(Path(tmp) / 'output'),
                '--cohort', str(Path(tmp) / 'absent.parquet')], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Run analysis.baseline_f1', result.stderr)
