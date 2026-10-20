"""Bag of words fitted on one LLM-era corpus and scored on the other (Fox8-23 <-> BotSim-24); the Limitations.

Historical producer: transfer_fox8_botsim.py, with explicit paths. Population: the text cohort written by
analysis.cohorts (Fox8-23 500/500, BotSim-24 329/408, 50 messages joined per account). Model: Table 1's bag of
words, fitted on all accounts of one corpus and scored on all of the other, so no test account is seen in
training. Rows: as released; "as an AI language model" deleted from both; document length alone; shuffled
training labels (20). Interval: bootstrap over test accounts, 1,000 resamples, seed 0.
Out: transfer_fox8_botsim.csv
"""
import re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from .common import arguments, record_run
args = arguments(__doc__, cohort=True)
rng = np.random.default_rng(0)
PIPE = lambda: make_pipeline(TfidfVectorizer(min_df=2, max_features=50_000, sublinear_tf=True),
                             LogisticRegression(max_iter=2000))
d = pd.read_parquet(args.cohort, columns=["corpus", "label", "account", "idx", "text"])
C = {}
for corpus in ("Fox8-23", "BotSim-24"):
    g = d[d.corpus == corpus].sort_values(["account", "idx"]).groupby("account", sort=False)
    C[corpus] = (g.text.apply(" ".join).values, g.label.first().values.astype(int))


def boot(y, s):
    v = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)


strip = lambda docs: np.array([re.sub(r"as an ai language model", " ", x, flags=re.I) for x in docs], dtype=object)
rows = []
for a, b in (("Fox8-23", "BotSim-24"), ("BotSim-24", "Fox8-23")):
    (Xa, ya), (Xb, yb) = C[a], C[b]
    for cond, fa, fb in (("as released", np.asarray(Xa, dtype=object), np.asarray(Xb, dtype=object)),
                         ("selection phrase deleted", strip(Xa), strip(Xb))):
        s = PIPE().fit(fa, ya).predict_proba(fb)[:, 1]
        lo, hi = boot(yb, s)
        rows.append(dict(fit_on=a, scored_on=b, condition=cond, auc=round(roc_auc_score(yb, s), 4), lo=lo, hi=hi,
                         n_fit=len(ya), n_scored=len(yb)))
    La = np.array([len(x) for x in Xa], float).reshape(-1, 1)
    Lb = np.array([len(x) for x in Xb], float).reshape(-1, 1)
    s = LogisticRegression(max_iter=1000).fit(np.log1p(La), ya).predict_proba(np.log1p(Lb))[:, 1]
    lo, hi = boot(yb, s)
    rows.append(dict(fit_on=a, scored_on=b, condition="document length alone", auc=round(roc_auc_score(yb, s), 4),
                     lo=lo, hi=hi, n_fit=len(ya), n_scored=len(yb)))
    nl = [roc_auc_score(yb, PIPE().fit(np.asarray(Xa, dtype=object), rng.permutation(ya))
                        .predict_proba(np.asarray(Xb, dtype=object))[:, 1]) for _ in range(20)]
    rows.append(dict(fit_on=a, scored_on=b, condition="shuffled training labels (20)", auc=round(float(np.mean(nl)), 4),
                     lo=round(float(np.percentile(nl, 2.5)), 4), hi=round(float(np.percentile(nl, 97.5)), 4),
                     n_fit=len(ya), n_scored=len(yb)))
R = pd.DataFrame(rows)
R.to_csv(args.output_dir / "transfer_fox8_botsim.csv", index=False)
print(R.to_string(index=False))
record_run(args.output_dir, "transfer", [args.cohort])
