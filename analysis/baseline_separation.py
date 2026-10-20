"""Is the bag-of-words F1 of 1.0000 an artefact of the 0.5 threshold? (historical baseline_separation.py)

On the 50-message condition the out-of-fold probabilities are perfectly separated, every bot above every
human, so every threshold in the gap gives F1 = 1 and the AUC is exactly 1.0. This records the gap: the
lowest bot score and the highest human score per corpus. Same cohort, model and folds as baseline_f1.py.
Out: baseline_separation.csv
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .common import arguments, record_run
args = arguments(__doc__, cohort=True)
PIPE = lambda: make_pipeline(TfidfVectorizer(min_df=2, max_features=50_000, sublinear_tf=True),
                             LogisticRegression(max_iter=2000))
d = pd.read_parquet(args.cohort, columns=["corpus", "label", "account", "idx", "text"])

rows = []
for corpus in ("Fox8-23", "BotSim-24"):
    sub = d[d.corpus == corpus].sort_values(["account", "idx"])
    g = sub.groupby("account", sort=False)
    docs = np.asarray(g.text.apply(lambda s: " ".join(s)).values, dtype=object)
    y = g.label.first().values.astype(int)
    p = cross_val_predict(PIPE(), docs, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    lo_bot, hi_human = float(p[y == 1].min()), float(p[y == 0].max())
    rows.append(dict(corpus=corpus, accounts=len(y), bots=int(y.sum()), auc=roc_auc_score(y, p),
                     auc_is_exactly_one=roc_auc_score(y, p) == 1.0, lowest_bot=round(lo_bot, 4),
                     highest_human=round(hi_human, 4), separated=lo_bot > hi_human,
                     half_is_inside_the_gap=hi_human < 0.5 < lo_bot))
    print(f"{corpus}: AUC {roc_auc_score(y, p)!r}, lowest bot {lo_bot:.4f} > highest human "
          f"{hi_human:.4f}, 0.5 inside the gap: {hi_human < 0.5 < lo_bot}", flush=True)

out = pd.DataFrame(rows)
assert out.separated.all() and out.half_is_inside_the_gap.all(), out
out.to_csv(args.output_dir / "baseline_separation.csv", index=False)
record_run(args.output_dir, "baseline_separation", [args.cohort])
