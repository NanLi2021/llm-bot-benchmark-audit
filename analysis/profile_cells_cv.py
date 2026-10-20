"""BotSim attribute diagnostic extracted from the historical v1 producer."""
import gzip, json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from .common import arguments, record_run, feature_counts
args = arguments(__doc__)
OUT = args.output_dir
BD = args.data_root
rng = np.random.default_rng(0)
NPERM = 10


def cv_auc(v, y):
    ok = np.isfinite(v)
    if ok.sum() < 50 or len(np.unique(v[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    pipe = make_pipeline(QuantileTransformer(output_distribution="normal",
                                             n_quantiles=min(1000, int(ok.sum()))),
                         LogisticRegression(max_iter=1000))
    p = cross_val_predict(pipe, v[ok].reshape(-1, 1), y[ok], cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y[ok], p)


def best_field(F, y):
    """Select the maximum pooled out-of-fold AUC; selection is not nested."""
    scores = {k: cv_auc(v, y) for k, v in F.items()}
    scores = {k: s for k, s in scores.items() if np.isfinite(s)}
    k = max(scores, key=scores.get)
    return k, scores[k]


def cell(name, F, y):
    k, a = best_field(F, y)
    nulls = [best_field(F, rng.permutation(y))[1] for _ in range(NPERM)]
    print(f"{name:<12} {k:<22} CV AUC {a:.4f}   null {np.mean(nulls):.4f}   n={len(y)}", flush=True)
    return dict(corpus=name, field=k, cv_auc=round(a, 4), null=round(float(np.mean(nulls)), 4),
                n=len(y), bots=int(y.sum()))


rows = []
num = lambda s: pd.to_numeric(pd.Series(s), errors="coerce").values.astype(float)

# ---- BotSim-24 ----------------------------------------------------------------------------
u = pd.read_csv(BD / "BotSim-24-Dataset/Users.csv")
y = np.r_[np.zeros(1907), np.ones(len(u) - 1907)].astype(int)   # the release's own ordering
F = {"submission_num": num(u.submission_num), "comment_num": num(u.comment_num),
     "has_description": u.description.notna().astype(float).values,
     "description_len": u.description.fillna("").astype(str).str.len().values.astype(float),
     "n_subreddits": u.subreddit.fillna("[]").astype(str).str.count(",").values.astype(float) + 1}
rows.append(cell("BotSim-24", F, y))


pd.DataFrame(rows).to_csv(OUT / "profile_cells_cv.csv", index=False)
pd.DataFrame(feature_counts(F, y, "BotSim-24")).to_csv(OUT / "profile_cells_cv_counts.csv", index=False)

record_run(args.output_dir, "profile_cells_cv", [BD / "BotSim-24-Dataset/Users.csv"])
