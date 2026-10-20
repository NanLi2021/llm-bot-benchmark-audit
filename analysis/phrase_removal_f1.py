"""Historical phrase_removal_f1 estimator, with explicit input/output paths. See docs/methods.md."""
import gzip
import json
import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .common import arguments, record_run
args = arguments(__doc__, cohort=True)
D = args.output_dir
if not (D / 'baseline_f1.csv').is_file():
    raise SystemExit('Run analysis.baseline_f1 with the same --cohort and --output-dir before phrase removal.')
FOX8 = args.data_root / "fox8_23_dataset.ndjson.gz"
SEED, BOOT = 0, 1000
PHRASE = r"as an ai language model"
TELL = re.compile(PHRASE + r"|i[’']?m an ai\b|i am an ai\b|i cannot fulfill|as a language model|"
                  r"i don[’']?t have personal|openai|chatgpt", re.I)
EXACT = re.compile(PHRASE, re.I)
PIPE = lambda: make_pipeline(TfidfVectorizer(min_df=2, max_features=50_000, sublinear_tf=True),
                             LogisticRegression(max_iter=2000))

d = pd.read_parquet(args.cohort, columns=["corpus", "label", "account", "idx", "text"])
sub = d[d.corpus == "Fox8-23"].sort_values(["account", "idx"])
g = sub.groupby("account", sort=False)
acc = pd.DataFrame({"account": list(g.groups), "y": g.label.first().values.astype(int),
                    "doc": g.text.apply(lambda s: " ".join(s)).values})
assert len(acc) == 1000 and acc.y.sum() == 500

full = {}
with gzip.open(FOX8, "rt") as f:
    for line in f:
        r = json.loads(line)
        uid = str(r["user_id"])
        if uid in set(acc.account):
            full[uid] = " ".join((t.get("text") or "") if isinstance(t, dict) else str(t)
                                 for t in (r.get("user_tweets") or []))
assert len(full) == 1000

acc["tell_pool"] = acc.doc.map(lambda s: bool(TELL.search(s)))
acc["exact_pool"] = acc.doc.map(lambda s: bool(EXACT.search(s)))
acc["tell_full"] = acc.account.map(lambda a: bool(TELL.search(full[a])))
acc["exact_full"] = acc.account.map(lambda a: bool(EXACT.search(full[a])))


def score(docs, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X = np.asarray(docs, dtype=object)
    p = cross_val_predict(PIPE(), X, y, cv=cv, method="predict_proba")[:, 1]
    ys = np.random.default_rng(SEED).permutation(y)
    ps = cross_val_predict(PIPE(), X, ys, cv=cv, method="predict_proba")[:, 1]
    rng = np.random.default_rng(SEED)
    b, h = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = [roc_auc_score(y[i], p[i]) for i in
            (np.r_[rng.choice(b, len(b)), rng.choice(h, len(h))] for _ in range(BOOT))]
    return dict(auc=roc_auc_score(y, p), ci_lo=np.quantile(boot, 0.025),
                ci_hi=np.quantile(boot, 0.975), f1_bot=f1_score(y, (p >= 0.5).astype(int)),
                f1_macro=f1_score(y, (p >= 0.5).astype(int), average="macro"),
                errors=int(((p >= 0.5).astype(int) != y).sum()), null_auc=roc_auc_score(ys, ps))


arms = [("A0", "nothing changed", acc, acc.doc),
        ("A1", "self-revealing phrases deleted from every message", acc,
         acc.doc.map(lambda s: TELL.sub(" ", s))),
        ("A2", "accounts dropped whose 50 scored messages contain a phrase",
         acc[~acc.tell_pool], acc[~acc.tell_pool].doc),
        ("A2-strict", "accounts dropped whose full history contains a phrase",
         acc[~acc.tell_full], acc[~acc.tell_full].doc)]
rows = []
for arm, rule, a, docs in arms:
    y = a.y.values
    s = score(docs.values, y)
    assert s["null_auc"] < 0.6, f"{arm}: shuffled labels do not fall to chance"
    rows.append(dict(arm=arm, rule=rule, bots=int(y.sum()), humans=int((y == 0).sum()),
                     dropped_bots=500 - int(y.sum()), dropped_humans=500 - int((y == 0).sum()),
                     **{k: round(float(v), 4) if isinstance(v, float) else v for k, v in s.items()},
                     fold_seed=SEED, boot=BOOT, pattern=TELL.pattern))
    print(f"{arm:10s} bots {y.sum():3d} humans {(y == 0).sum():3d}  AUC {s['auc']:.4f} "
          f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]  F1 {s['f1_bot']:.4f}  errors {s['errors']}  "
          f"null {s['null_auc']:.4f}", flush=True)

# A2-strict keeps 31 bots against 500 humans. The ranking (AUC) is unaffected by that imbalance, but
# a 0.5 cut is not: the first run gave AUC 1.0000 with F1 0.0000, every bot ranked above every human
# and none above 0.5. So the same arm is also scored against 31 humans drawn at random, 100 times.
strict = acc[~acc.tell_full]
bots, hums = strict[strict.y == 1], strict[strict.y == 0]
rng = np.random.default_rng(SEED)
bal = []
for _ in range(100):
    k = pd.concat([bots, hums.iloc[rng.choice(len(hums), len(bots), replace=False)]])
    y = k.y.values
    p = cross_val_predict(PIPE(), np.asarray(k.doc.values, dtype=object), y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method="predict_proba")[:, 1]
    bal.append((roc_auc_score(y, p), f1_score(y, (p >= 0.5).astype(int))))
bal = np.array(bal)
rows.append(dict(arm="A2-strict, balanced", rule="A2-strict with humans drawn to equal the bots, "
                 "100 draws; mean, and 2.5-97.5% of draws", bots=len(bots), humans=len(bots),
                 dropped_bots=500 - len(bots), dropped_humans=500 - len(bots),
                 auc=round(float(bal[:, 0].mean()), 4), ci_lo=round(float(np.quantile(bal[:, 0], .025)), 4),
                 ci_hi=round(float(np.quantile(bal[:, 0], .975)), 4),
                 f1_bot=round(float(bal[:, 1].mean()), 4), f1_macro=None, errors=None, null_auc=None,
                 fold_seed=SEED, boot=100, pattern=TELL.pattern))
print(f"A2-strict balanced, {len(bots)} v {len(bots)}, 100 draws: AUC {bal[:, 0].mean():.4f} "
      f"[{np.quantile(bal[:, 0], .025):.4f}, {np.quantile(bal[:, 0], .975):.4f}], "
      f"F1 {bal[:, 1].mean():.4f} [{np.quantile(bal[:, 1], .025):.4f}, {np.quantile(bal[:, 1], .975):.4f}]",
      flush=True)

base = pd.read_csv(D / "baseline_f1.csv")
cell = base[(base.corpus == "Fox8-23") & (base.condition == "all 50 messages")].iloc[0]
assert abs(rows[0]["auc"] - cell.auc) < 1e-4 and abs(rows[0]["f1_bot"] - cell.f1_bot) < 1e-4, \
    "A0 does not reproduce the ceiling cell"

for w in ("pool", "full"):
    for c, n in ((1, "bots"), (0, "humans")):
        k = acc[acc.y == c]
        print(f"  {w:4s} {n:6s}: any phrase {k[f'tell_{w}'].sum():3d} of {len(k)}, "
              f"exact search phrase {k[f'exact_{w}'].sum():3d}")
pd.DataFrame(rows).to_csv(D / "phrase_removal_f1.csv", index=False)
drop = []
for w in ("pool", "full"):
    for r in acc[acc[f"tell_{w}"]].itertuples():
        text = r.doc if w == "pool" else full[r.account]
        drop.append(dict(window=w, account=r.account, label=r.y, exact=getattr(r, f"exact_{w}"),
                         matched=";".join(sorted({m.lower() for m in TELL.findall(text)}))))
pd.DataFrame(drop).to_csv(D / "phrase_removal_dropped.csv", index=False)
print("wrote phrase_removal_f1.csv, phrase_removal_dropped.csv")
record_run(args.output_dir, "phrase_removal_f1", [args.cohort] + [FOX8])
