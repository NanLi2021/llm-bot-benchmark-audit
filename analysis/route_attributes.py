"""Historical human-source route_attributes diagnostic. Selection is not nested."""
import gzip, itertools, json, pathlib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from .common import arguments, record_run, feature_counts
args = arguments(__doc__)
OUT = args.output_dir
rng = np.random.default_rng(0)
NPERM = 10
num = lambda s: pd.to_numeric(pd.Series(s), errors="coerce").values.astype(float)


def cv_auc(v, y):
    ok = np.isfinite(v)
    if ok.sum() < 50 or len(np.unique(v[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    pipe = make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, int(ok.sum()))),
                         LogisticRegression(max_iter=1000))
    p = cross_val_predict(pipe, v[ok].reshape(-1, 1), y[ok], cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y[ok], p)


def best_field(F, y):
    s = {k: cv_auc(v, y) for k, v in F.items()}
    s = {k: a for k, a in s.items() if np.isfinite(a)}
    k = max(s, key=s.get)
    return k, s[k]


prof, src = [], []
with gzip.open(args.data_root / "fox8_23_dataset.ndjson.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        tws = r.get("user_tweets") or []          # Fox8-23 keeps the user object inside each tweet
        u = next((t["user"] for t in reversed(tws) if isinstance(t.get("user"), dict)), None)  # as profile_cells_cv2.py
        if u and not str(r.get("label", "")).lower().startswith("bot"):
            prof.append(u); src.append(r.get("dataset"))
d = pd.DataFrame(prof); src = np.array(src)
print(pd.Series(src).value_counts().to_string())
F = {k: num(d[k]) for k in ("followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count")}
F["ff_ratio"] = F["followers_count"] / (F["friends_count"] + 1)
F["description_len"] = d.description.fillna("").astype(str).str.len().values.astype(float)

rows = []
counts = []
for a, b in itertools.combinations(sorted(set(src)), 2):
    m = np.isin(src, [a, b]); y = (src[m] == b).astype(int)
    Fm = {k: v[m] for k, v in F.items()}
    counts.extend(feature_counts(Fm, y, f"{a} vs {b}"))
    k, auc = best_field(Fm, y)
    null = float(np.mean([best_field(Fm, rng.permutation(y))[1] for _ in range(NPERM)]))
    rows.append(dict(study_a=a, study_b=b, n=int(m.sum()), field=k, cv_auc=round(auc, 4), null=round(null, 4)))
    print(rows[-1], flush=True)
pd.DataFrame(counts).rename(columns={"finite_bots":"finite_study_b", "finite_humans":"finite_study_a"}).to_csv(OUT / "route_attribute_counts.csv", index=False)
R = pd.DataFrame(rows); R.to_csv(OUT / "route_attributes.csv", index=False)
print(f"\nmedian pairwise AUC {R.cv_auc.median():.4f}  range {R.cv_auc.min():.4f} to {R.cv_auc.max():.4f}  median null {R.null.median():.4f}")
record_run(args.output_dir, "route_attributes", [args.data_root / "fox8_23_dataset.ndjson.gz"])
