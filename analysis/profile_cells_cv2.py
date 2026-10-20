"""Fox8 attribute diagnostic extracted from v2, which reads nested tweet user profiles."""
import glob, gzip, json, pathlib
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
num = lambda s: pd.to_numeric(pd.Series(s), errors="coerce").values.astype(float)


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
    s = {k: cv_auc(v, y) for k, v in F.items()}
    s = {k: x for k, x in s.items() if np.isfinite(x)}
    if not s:
        return None, np.nan
    k = max(s, key=s.get)
    return k, s[k]


def cell(name, F, y, quiet=False):
    k, a = best_field(F, y)
    nulls = [best_field(F, rng.permutation(y))[1] for _ in range(NPERM)]
    if not quiet:
        print(f"{name:<14} {str(k):<20} CV AUC {a:.4f}   null {np.nanmean(nulls):.4f}   n={len(y)}", flush=True)
    return dict(corpus=name, field=k, cv_auc=round(a, 4), null=round(float(np.nanmean(nulls)), 4),
                n=len(y), bots=int(y.sum()))


rows = []

# ---- Fox8-23: the user object lives inside each tweet --------------------------------------
prof, lab = [], []
with gzip.open(BD / "fox8_23_dataset.ndjson.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        tws = r.get("user_tweets") or []
        u = next((t["user"] for t in reversed(tws) if isinstance(t.get("user"), dict)), None)
        if u is None:
            continue
        prof.append(u); lab.append(1 if str(r["label"]).lower().startswith("bot") else 0)
d = pd.DataFrame(prof); y = np.array(lab)
F = {k: num(d[k]) for k in ("followers_count", "friends_count", "statuses_count",
                            "favourites_count", "listed_count") if k in d}
F["ff_ratio"] = F["followers_count"] / (F["friends_count"] + 1)
F["description_len"] = d.description.fillna("").astype(str).str.len().values.astype(float)
F["verified"] = d.verified.fillna(False).astype(float).values if "verified" in d else np.full(len(d), np.nan)
rows.append(cell("Fox8-23", F, y))


pd.DataFrame(rows).to_csv(OUT / "profile_cells_cv2.csv", index=False)
pd.DataFrame(feature_counts(F, y, "Fox8-23")).to_csv(OUT / "profile_cells_cv2_counts.csv", index=False)

record_run(args.output_dir, "profile_cells_cv2", [BD / "fox8_23_dataset.ndjson.gz"])
