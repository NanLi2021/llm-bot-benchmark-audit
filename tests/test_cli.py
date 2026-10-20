"""Exercise the real command path, including serialization and CPU scoring."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_sampling import campaign
from analysis.io_matching import prepare, pair_pool

ROOT=Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_original_pairing_sequence(self):
        golden=json.loads((ROOT/'tests/fixtures/io_historical.json').read_text())
        d,h,_=prepare(campaign())
        for row in golden['selections']:
            posts=pair_pool(d,h,row['draw']).postid.tolist()
            self.assertEqual(hashlib.sha256(json.dumps(posts).encode()).hexdigest(),row['sha256'])

    def test_io_command_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'input/campaigns/Synthetic'
            source.mkdir(parents=True)
            campaign().to_parquet(source/'posts.parquet',index=False)
            result=subprocess.run([sys.executable,'-m','analysis.io_matching','--data-root',str(root/'input'),
                '--output-dir',str(root/'output'),'--protocol','paired','--draws','1','--permutations','1','--score'],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            import pandas as pd
            scores=pd.read_csv(root/'output/draws.csv')
            self.assertEqual(set(scores.rung),{'M0','M1','M2','M3'})
            self.assertTrue((scores.valid_folds==5).all())
            audit=pd.read_csv(root/'output/balance.csv')
            self.assertTrue((audit[audit.stage.isin(['M2_final','M3_final'])].cell_l1==0).all())
            original = (root/'output/draws.csv').read_bytes()
            retry = subprocess.run([sys.executable,'-m','analysis.io_matching','--data-root',str(root/'input'),
                '--output-dir',str(root/'output'),'--protocol','paired','--draws','1'],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(retry.returncode, 2)
            self.assertIn('must be empty', retry.stderr)
            self.assertEqual((root/'output/draws.csv').read_bytes(), original)
