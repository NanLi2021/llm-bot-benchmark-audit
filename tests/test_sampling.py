"""Synthetic checks of scientific invariants; no real users or experiment claims."""
import unittest
import numpy as np
import pandas as pd
from analysis.cohorts import normalize, select
from analysis.io_matching import balance, capped, construct, pair_pool, prepare, sample_pairs


def campaign():
    rows = []
    for label in (1, 0):
        for i in range(140):
            # Two posts in different cells share cleaned text on only the IO side.
            text = ('copied text sufficiently long for the experiment' if label and i < 2
                    else f'original document label{label} unique{i} contains several words')
            rows.append(dict(postid=f'{label}-{i}', accountid=f'account-{label}-{i}',
                             post_text=text, is_control=not bool(label), is_repost=False,
                             post_language='en', post_time='2020-01-01', hashtags=[f'cell{i//10}']))
    return pd.DataFrame(rows)


class SamplingTests(unittest.TestCase):
    def test_cohort_order_and_filtering(self):
        corpus = {'A': {1: {'a': ['@name https://site.test short', 'a long retained message 111', 'a long retained message 222'],
                            'b': ['too short']}, 0: {'c': ['human retained message 111', 'human retained message 222']}}}
        result = select(corpus, messages=2, per_class=1)
        self.assertEqual(result.account.tolist(), ['a', 'a', 'c', 'c'])
        self.assertEqual(result.idx.tolist(), [0, 1, 0, 1])
        self.assertEqual(normalize('@name @other hello https://site.test'), 'hello')

    def test_historical_sampling_can_break_pairs(self):
        pool = pd.DataFrame([dict(pair_id=i, cell=str(i), io=label, accountid=f'{i}-{label}')
                             for label in (1, 0) for i in range(20)])
        result = capped(pool, 8, 0)
        self.assertGreater(balance(result)['cell_l1'], 0)
        self.assertEqual(balance(result)['io_posts'], 8)
        self.assertEqual(balance(result)['control_posts'], 8)

    def test_pair_sampling_rejects_incomplete_pairs(self):
        broken = pd.DataFrame([dict(pair_id=0, cell='a', io=1, accountid='one')])
        with self.assertRaises(ValueError):
            sample_pairs(broken, 1, 0)

    def test_pair_survival_through_text_exclusion(self):
        d, h, _ = prepare(campaign())
        for seed in range(4):
            sets, stages, n, eligible, _ = construct(d, h, seed, 'paired')
            self.assertTrue(eligible)
            self.assertEqual(n, 138)
            self.assertEqual(len(stages['M3_after_text_exclusions']), 278)
            self.assertEqual(len(stages['M3_pool']), 276)
            for rung in ('M2', 'M3'):
                info = balance(sets[rung])
                self.assertEqual(info['cell_l1'], 0)
                self.assertEqual(info['incomplete_pairs'], 0)
                self.assertEqual(info['io_posts'], info['control_posts'])
                self.assertTrue(sets[rung].accountid.value_counts().max() <= 5)

    def test_multiple_hashtags_do_not_duplicate_posts(self):
        raw = campaign()
        raw['hashtags'] = [['shared', *tags] for tags in raw.hashtags]
        d, h, _ = prepare(raw)
        pool = pair_pool(d, h, 0)
        self.assertFalse(pool.postid.duplicated().any())
        self.assertEqual(balance(pool)['incomplete_pairs'], 0)
        self.assertTrue(pd.crosstab(pool.cell, pool.io).max().max() <= 20)

    def test_conflicting_account_labels_rejected(self):
        raw = campaign()
        raw.loc[140, 'accountid'] = raw.loc[0, 'accountid']
        with self.assertRaises(ValueError):
            prepare(raw)


if __name__ == '__main__':
    unittest.main()
