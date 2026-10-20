"""Extract the Fox8/BotSim message cohort without loading any detector models."""
import gzip
import json
import re
import numpy as np
import pandas as pd
from .common import arguments, record_run

_LEAD = re.compile(r'^(?:\s*@\w+)+')
_URL = re.compile(r'https?://\S+|t\.co/\S+|pic\.twitter\.com/\S+')


def normalize(text):
    return _URL.sub(' ', _LEAD.sub('', str(text))).strip()


def load(data_root):
    corpora = {}
    groups = {1: {}, 0: {}}
    with gzip.open(data_root / 'fox8_23_dataset.ndjson.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            groups[int(row['label'] == 'bot')][row['user_id']] = [
                t.get('text') or '' for t in row.get('user_tweets') or []]
    corpora['Fox8-23'] = groups
    base = data_root / 'BotSim-24-Dataset'
    users = pd.read_csv(base / 'Users.csv')
    if len(users) < 1907:
        raise ValueError('BotSim input must contain the historical 1907 human rows')
    with (base / 'user_post_comment.json').open(encoding='utf-8') as handle:
        posts = json.load(handle)
    groups = {1: {}, 0: {}}
    for index, uid in enumerate(users.user_id):
        rec = posts.get(uid) or {}
        messages = [str(x.get('posts') or '') for x in rec.get('posts') or []]
        for key in ('comment_1', 'comment_2'):
            messages += [str(x.get('comment_body') or '') for x in rec.get(key) or []]
        groups[int(index >= 1907)][uid] = messages
    corpora['BotSim-24'] = groups
    return corpora


def select(corpora, seed=0, messages=50, per_class=500):
    # One RNG across corpora, bots before humans. Changing either changes BotSim selection.
    rng = np.random.default_rng(seed)
    rows = []
    for corpus, groups in corpora.items():
        for label, accounts in groups.items():
            keys = list(accounts)
            rng.shuffle(keys)
            selected = 0
            for account in keys:
                texts = [t for t in map(normalize, accounts[account]) if len(t) >= 20][:messages]
                if len(texts) >= messages:
                    rows.extend(dict(corpus=corpus, label=label, account=str(account), idx=i, text=t)
                                for i, t in enumerate(texts))
                    selected += 1
                if selected >= per_class:
                    break
    return pd.DataFrame(rows, columns=['corpus', 'label', 'account', 'idx', 'text'])


def main():
    args = arguments(__doc__)
    data = select(load(args.data_root))
    data.to_parquet(args.output_dir / 'cohort.parquet', index=False)
    counts = data.groupby(['corpus', 'label']).account.nunique().reset_index(name='accounts')
    counts.to_csv(args.output_dir / 'cohort_counts.csv', index=False)
    record_run(args.output_dir, 'cohorts', [args.data_root / p for p in (
        'fox8_23_dataset.ndjson.gz', 'BotSim-24-Dataset/Users.csv',
        'BotSim-24-Dataset/user_post_comment.json')], seed=0, messages=50, per_class_cap=500)
    print(counts.to_string(index=False))


if __name__ == '__main__':
    main()
