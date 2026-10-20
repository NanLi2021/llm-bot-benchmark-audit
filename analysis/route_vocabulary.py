"""Historical human-source route_vocabulary diagnostic. Selection is not nested."""
import gzip, itertools, json, pathlib
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .common import arguments, record_run
args = arguments(__doc__)
D = args.output_dir
rng = np.random.default_rng(0)

rows = []
with gzip.open(args.data_root / "fox8_23_dataset.ndjson.gz", "rt") as fh:
    for line in fh:
        o = json.loads(line)
        if o["label"] != "human":
            continue
        msgs = [(t.get("text") or "").strip() for t in (o.get("user_tweets") or [])]
        msgs = [m for m in msgs if len(m) >= 20]
        if len(msgs) < 5:
            continue
        pick = rng.choice(len(msgs), size=min(20, len(msgs)), replace=False)
        words = " ".join(msgs[i] for i in pick).split()
        if len(words) >= 8:
            rows.append({"route": o.get("dataset"), "text": " ".join(words),
                         "text25": " ".join(words[:25])})

d = pd.DataFrame(rows)
mk = lambda: make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                                           sublinear_tf=True),
                           LogisticRegression(max_iter=2000, C=1.0))
cv = StratifiedKFold(5, shuffle=True, random_state=0)

out = []
# four-way: which of the four studies did this human account come from?
Y = np.array([str(x) for x in d.route.tolist()], dtype=object)
COLS = {"full text, 20 messages": "text", "the same 25 words": "text25"}
for _lab, _c in COLS.items():
    X = np.array([str(x) for x in d[_c].tolist()], dtype=object)
    p = cross_val_predict(mk(), X, Y, cv=cv)
    out.append({"input": _lab, "comparison": "four-way, which study", "n": len(d),
                "accuracy": round(accuracy_score(Y, p), 4),
                "chance": round(float(pd.Series(Y).value_counts(normalize=True).max()), 4),
                "auc": ""})
# and pairwise, so the number is on the same scale as every other AUC in the paper
    for a, b in itertools.combinations(sorted(set(Y)), 2):
        m = np.isin(Y, [a, b])
        xs, ys = X[m], (Y[m] == b).astype(int)
        pr = cross_val_predict(mk(), xs, ys, cv=cv, method="predict_proba")[:, 1]
        out.append({"input": _lab, "comparison": f"{a} vs {b}", "n": int(m.sum()),
                    "accuracy": "", "chance": "", "auc": round(roc_auc_score(ys, pr), 4)})

o = pd.DataFrame(out)
o.to_csv(D / "route_vocabulary.csv", index=False)
print(o.to_string(index=False))
for _lab in COLS:
    _a = [x for x in o[o.input == _lab].auc if x != ""]
    print(f"{_lab}: pairwise AUC {min(_a):.3f} to {max(_a):.3f}, median {np.median(_a):.3f}")
record_run(args.output_dir, "route_vocabulary", [args.data_root / "fox8_23_dataset.ndjson.gz"])
