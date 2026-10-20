"""Audit historical IO sampling or evaluate a protocol that preserves matched pairs.

Default: audit only. --score fits CPU classifiers; --protocol paired retains only
complete pairs after M3 exclusions and samples pair IDs. No raw text is exported.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from .common import record_run, output_directory

COLS = ['postid', 'post_text', 'is_control', 'accountid', 'is_repost',
        'post_language', 'post_time', 'hashtags']
PER_PAIR, PER_ACCOUNT, CAP, MIN_N, MIN_ACC = 20, 5, 2000, 100, 20
STRIP = re.compile(r'https?://\S+|t\.co/\S+|[@#]\w+')


def cleaned(text):
    return re.sub(r'\s+', ' ', STRIP.sub(' ', text)).strip().lower()


def resolve_account_conflicts(frame, policy):
    """Fail by default, or explicitly exclude entire cross-class accounts.

    Applied after language selection but before post deduplication, so duplicate
    post IDs cannot conceal inconsistent account labels. Never relabel records.
    """
    if policy not in ('error', 'exclude'):
        raise ValueError('Unknown account conflict policy')
    missing = int(frame.accountid.isna().sum())
    classes = frame.groupby('accountid').io.nunique()
    conflicts = classes[classes > 1].index
    mask = frame.accountid.isin(conflicts)
    audit = dict(account_conflict_policy=policy, conflicting_accounts=len(conflicts),
                 excluded_accounts=len(conflicts) if policy == 'exclude' else 0,
                 excluded_posts=int(mask.sum()) if policy == 'exclude' else 0,
                 excluded_io_posts=int((mask & (frame.io == 1)).sum()) if policy == 'exclude' else 0,
                 excluded_control_posts=int((mask & (frame.io == 0)).sum()) if policy == 'exclude' else 0)
    if missing or (len(conflicts) and policy == 'error'):
        raise ValueError(f'Account consistency: {missing} posts with missing account IDs; '
                         f'{len(conflicts)} account IDs with both classes')
    return frame[~mask].copy() if policy == 'exclude' else frame.copy(), audit


def prepare(frame, account_conflicts='error'):
    d = frame[frame.is_repost != True].copy()
    d['post_text'] = d.post_text.astype(str)
    d = d[d.post_text.str.len() >= 20].copy()
    if d.empty or d.is_control.isna().any() or not set(d.is_control.unique()) <= {True, False}:
        raise ValueError('Missing, empty or non-boolean class labels')
    if d.is_control.nunique() < 2:
        return d, pd.DataFrame(), None
    d['io'] = (~d.is_control.astype(bool)).astype(int)
    language = d[d.io == 1].post_language.fillna('und').value_counts().index[0]
    # Preserve historical handling: missing language is counted as und for choosing
    # the mode but does not equal the literal string und in the filtering step.
    d, exclusion = resolve_account_conflicts(d[d.post_language == language], account_conflicts)
    d = d.drop_duplicates('postid').reset_index(drop=True)
    d.attrs['input_exclusion'] = exclusion
    d['date'] = pd.to_datetime(d.post_time, errors='coerce', utc=True).dt.date
    h = d[['io', 'date', 'hashtags', 'accountid']].explode('hashtags').dropna(subset=['hashtags', 'date'])
    h['pair'] = list(zip(h.hashtags.astype(str).str.lower(), h.date))
    counts = h.groupby('pair').io.nunique()
    h = h[h.pair.isin(set(counts[counts == 2].index))]
    return d, h, language


def pair_pool(d, h, seed):
    """Original greedy assignment and random order, plus explicit pair/cell IDs."""
    hs = h.sample(frac=1.0, random_state=seed)
    used, per_acc, take, assignments = set(), Counter(), {0: [], 1: []}, {}
    pair_id = 0
    for cell, group in hs.groupby('pair', sort=False):
        def ok(i, account):
            return i not in used and per_acc[account] < PER_ACCOUNT
        a = [(i, acc) for i, acc in zip(group.index[group.io == 1], group.accountid[group.io == 1]) if ok(i, acc)]
        b = [(i, acc) for i, acc in zip(group.index[group.io == 0], group.accountid[group.io == 0]) if ok(i, acc)]
        matched = 0
        for (i, ai), (j, aj) in zip(a, b):
            if matched >= PER_PAIR:
                break
            if not ok(i, ai) or not ok(j, aj):
                continue
            take[1].append(i); take[0].append(j)
            used.update((i, j)); per_acc[ai] += 1; per_acc[aj] += 1; matched += 1
            assignments[i] = assignments[j] = (pair_id, json.dumps([cell[0], str(cell[1])]))
            pair_id += 1
    pool = d.loc[take[1] + take[0]].copy()
    pool['pair_id'] = [assignments[i][0] for i in pool.index]
    pool['cell'] = [assignments[i][1] for i in pool.index]
    pool['clean'] = pool.post_text.map(cleaned)
    return pool


def capped(frame, n, seed):
    f = frame.sample(frac=1.0, random_state=seed)
    f = f[f.groupby('accountid').cumcount() < PER_ACCOUNT]
    return pd.concat([f[f.io == c].head(n) for c in (0, 1)])


def sample_pairs(frame, n, seed):
    groups = frame.groupby('pair_id').io.agg(['size', 'nunique'])
    if not ((groups['size'] == 2) & (groups['nunique'] == 2)).all():
        raise ValueError('Pair sampling requires exactly one post of each class per pair')
    ids = groups.index.to_series().sample(n=n, random_state=seed)
    return frame[frame.pair_id.isin(ids)].copy()


def balance(frame):
    if frame.empty:
        return dict(posts=0, io_posts=0, control_posts=0, io_accounts=0, control_accounts=0,
                    cells=0, imbalanced_cells=0, cell_l1=0, incomplete_pairs=0)
    table = pd.crosstab(frame.cell, frame.io).reindex(columns=[0, 1], fill_value=0)
    delta = (table[1] - table[0]).abs()
    return dict(posts=len(frame), io_posts=int(frame.io.sum()), control_posts=int((frame.io == 0).sum()),
                io_accounts=frame[frame.io == 1].accountid.nunique(),
                control_accounts=frame[frame.io == 0].accountid.nunique(), cells=len(table),
                imbalanced_cells=int((delta > 0).sum()), cell_l1=int(delta.sum()),
                incomplete_pairs=int((frame.groupby('pair_id').size() != 2).sum()))


def construct(d, h, seed, protocol):
    m2 = pair_pool(d, h, seed)
    valid = (m2.clean.str.len() >= 20) & ~m2.clean.duplicated(keep=False)
    filtered = m2[valid].copy()
    m3 = filtered
    if protocol == 'paired':
        complete = m3.groupby('pair_id').size()
        m3 = m3[m3.pair_id.isin(complete[complete == 2].index)].copy()
    n = min(int((m3.io == 1).sum()), int((m3.io == 0).sum()), CAP)
    sample = sample_pairs if protocol == 'paired' else capped
    sets = dict(M0=capped(d, n, seed), M1=capped(d.loc[h.index.unique()], n, seed),
                M2=sample(m2, n, seed), M3=sample(m3, n, seed).assign(post_text=lambda f: f.clean))
    stages = {'M2_pool': m2, 'M3_after_text_exclusions': filtered, 'M3_pool': m3,
              'M2_final': sets['M2'], 'M3_final': sets['M3']}
    eligible = n >= MIN_N and all(
        len(f[f.io == c]) == n and f[f.io == c].accountid.nunique() >= MIN_ACC
        for f in sets.values() for c in (0, 1))
    # Historical eligibility is recorded separately; stricter final checks prevent
    # claiming requested sizes or account counts that capping failed to produce.
    historical_eligible = n >= MIN_N and all(m3[m3.io == c].accountid.nunique() >= MIN_ACC for c in (0, 1))
    if protocol == 'paired':
        for rung in ('M2', 'M3'):
            if balance(sets[rung])['cell_l1'] or balance(sets[rung])['incomplete_pairs']:
                raise AssertionError('Matched sampling lost a pair')
    return sets, stages, n, eligible, historical_eligible


def evaluate(frame, seed, labels=None):
    x = frame.post_text.to_numpy(dtype=object)
    y = frame.io.to_numpy() if labels is None else labels
    groups = frame.accountid.to_numpy()
    scores = []
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('Account leakage across folds')
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        pipe = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=200000),
                             LogisticRegression(max_iter=2000))
        scores.append(roc_auc_score(y[test], pipe.fit(x[train], y[train]).predict_proba(x[test])[:, 1]))
    if not scores:
        raise ValueError('No valid folds; refusing to report NaN as a result')
    return float(np.mean(scores)), len(scores)


def score_draw(frame, seed, permutations):
    auc, folds = evaluate(frame, seed)
    accounts = frame.groupby('accountid').io.first()
    rng = np.random.default_rng(1000 + seed)
    nulls, counts = [], []
    for _ in range(permutations):
        mapping = dict(zip(accounts.index, rng.permutation(accounts.to_numpy())))
        value, count = evaluate(frame, seed, frame.accountid.map(mapping).to_numpy())
        nulls.append(value); counts.append(count)
    return dict(auc=auc, valid_folds=folds, null=float(np.mean(nulls)),
                null_lo=float(np.percentile(nulls, 2.5)), null_hi=float(np.percentile(nulls, 97.5)),
                null_valid_folds_min=min(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', choices=['historical', 'paired'], default='historical')
    parser.add_argument('--account-conflicts', choices=['error', 'exclude'], default='error',
                        help='Explicit sensitivity policy: exclude whole cross-class accounts before deduplication')
    parser.add_argument('--campaign', action='append', help='Repeat for selected campaigns; default all directories')
    parser.add_argument('--draws', type=int, default=10)
    parser.add_argument('--permutations', type=int, default=20)
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--save-membership', action='store_true', help='Private IDs/cell assignments, never raw text')
    args = parser.parse_args()
    if args.draws < 1 or args.permutations < 1:
        parser.error('draws and permutations must be positive')
    root = args.data_root / 'campaigns'
    if not root.is_dir():
        parser.error('data-root must contain campaigns/')
    try:
        out = output_directory(args.output_dir, fresh=True)
    except ValueError as error:
        parser.error(str(error))
    campaigns = sorted(p for p in root.iterdir() if p.is_dir() and (not args.campaign or p.name in args.campaign))
    if not campaigns or (args.campaign and set(args.campaign) - {p.name for p in campaigns}):
        parser.error('No campaigns or unknown campaign selection')
    audit, scores, inventory = [], [], []
    inputs = []
    exclusions = []
    for directory in campaigns:
        files = sorted(directory.glob('*.parquet'))
        if not files:
            inventory.append(dict(campaign=directory.name, status='no_parquet'))
            continue
        inputs.extend(dict(campaign=directory.name, name=p.name, bytes=p.stat().st_size) for p in files)
        try:
            raw = pd.concat([pd.read_parquet(p, columns=COLS) for p in files], ignore_index=True)
            d, h, language = prepare(raw, args.account_conflicts)
        except (ValueError, KeyError) as error:
            inventory.append(dict(campaign=directory.name, status='input_error', reason=str(error)))
            print(f'{directory.name} INPUT ERROR: {error}', flush=True)
            continue
        exclusion = dict(campaign=directory.name, **d.attrs.get('input_exclusion', {}))
        exclusions.append(exclusion)
        pd.DataFrame(exclusions).to_csv(out / 'input_exclusions.csv', index=False)
        if exclusion.get('excluded_accounts', 0):
            print(f"{directory.name}: excluded {exclusion['excluded_accounts']} conflicting accounts, "
                  f"{exclusion['excluded_io_posts']} IO and {exclusion['excluded_control_posts']} control posts", flush=True)
        if h.empty:
            inventory.append(dict(campaign=directory.name, status='no_shared_cells_or_two_classes', language=language))
            continue
        draws, all_eligible = [], True
        for seed in range(args.draws):
            sets, stages, n, eligible, historical = construct(d, h, seed, args.protocol)
            all_eligible &= eligible
            for stage, frame in stages.items():
                audit.append(dict(campaign=directory.name, draw=seed, protocol=args.protocol,
                                  account_conflict_policy=args.account_conflicts, stage=stage,
                                  target=n, eligible=eligible, historical_eligible=historical, **balance(frame)))
            if args.save_membership:
                target = out / 'private-membership' / directory.name
                target.mkdir(parents=True, exist_ok=True)
                for rung, frame in sets.items():
                    cols = [c for c in ('postid','accountid','io','pair_id','cell') if c in frame]
                    frame[cols].to_csv(target / f'{rung}-{seed}.csv.gz', index=False)
            draws.append(sets)
        inventory.append(dict(campaign=directory.name, status='eligible' if all_eligible else 'threshold_or_final_count_failure', language=language))
        if all_eligible and args.score:
            for seed, sets in enumerate(draws):
                for rung, frame in sets.items():
                    print(f'{directory.name} {args.protocol} draw {seed} {rung}', flush=True)
                    scores.append(dict(campaign=directory.name, rung=rung, draw=seed, language=language,
                                       protocol=args.protocol, account_conflict_policy=args.account_conflicts, n_per_class=int((frame.io == 1).sum()),
                                       accounts_IO=frame[frame.io == 1].accountid.nunique(),
                                       accounts_control=frame[frame.io == 0].accountid.nunique(),
                                       **score_draw(frame, seed, args.permutations)))
                    pd.DataFrame(scores).to_csv(out / 'draws.csv', index=False)
        pd.DataFrame(audit).to_csv(out / 'balance.csv', index=False)
        pd.DataFrame(inventory).to_csv(out / 'campaigns.csv', index=False)
        if scores:
            pd.DataFrame(scores).to_csv(out / 'draws.csv', index=False)
        print(directory.name, inventory[-1]['status'], flush=True)
    pd.DataFrame(audit).to_csv(out / 'balance.csv', index=False)
    pd.DataFrame(inventory).to_csv(out / 'campaigns.csv', index=False)
    if scores:
        rows = []
        for (campaign, rung), group in pd.DataFrame(scores).groupby(['campaign','rung'], sort=False):
            rows.append(dict(campaign=campaign, rung=rung, protocol=args.protocol,
                             account_conflict_policy=args.account_conflicts, draws=len(group),
                             auc=group.auc.mean(), lo=group.auc.quantile(.025), hi=group.auc.quantile(.975),
                             null=group['null'].mean(), n_per_class=group.n_per_class.mean(),
                             accounts_IO=group.accounts_IO.mean(), accounts_control=group.accounts_control.mean(),
                             valid_folds_min=group.valid_folds.min(), null_valid_folds_min=group.null_valid_folds_min.min()))
        pd.DataFrame(rows).to_csv(out / 'io_topic_matched.csv', index=False)
    errors = [r for r in inventory if r['status'] == 'input_error']
    record_run(out, 'io_matching', status='incomplete' if errors else 'completed',
               input_errors=errors, input_exclusions=exclusions, account_conflict_policy=args.account_conflicts,
               protocol=args.protocol, draws=args.draws, score=args.score,
               permutations=args.permutations, input_inventory=inputs,
               input_hash_status='Use scripts/input_manifest.py for content hashes; this record contains sizes only')
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
