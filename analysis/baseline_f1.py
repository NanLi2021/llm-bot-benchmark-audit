"""Historical baseline_f1 estimator, with explicit input/output paths. See docs/methods.md."""
import pathlib, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline

from .common import arguments, record_run
args = arguments(__doc__, cohort=True)
D = args.output_dir
DRAWS, WORDS, NMSG, SEED = 10, 25, 20, 0
PIPE = lambda: make_pipeline(TfidfVectorizer(min_df=2, max_features=50_000, sublinear_tf=True),
                             LogisticRegression(max_iter=2000))
d = pd.read_parquet(args.cohort, columns=["corpus", "label", "account", "idx", "text"])
rows = []
for corpus in ("Fox8-23", "BotSim-24"):
    sub = d[d.corpus == corpus].sort_values(["account", "idx"])
    # both arrays must come from the SAME grouping, or documents and labels are shuffled apart:
    # the first version grouped a sorted frame for the text and an unsorted one for the labels,
    # which put Fox8-23 at AUC 0.473 instead of 1.000.
    g = sub.groupby("account", sort=False)
    docs = g.text.apply(lambda s: " ".join(s)).values
    y = g.label.first().values.astype(int)
    p = cross_val_predict(PIPE(), np.asarray(docs, dtype=object), y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    f1 = f1_score(y, (p >= 0.5).astype(int))
    f1m = f1_score(y, (p >= 0.5).astype(int), average="macro")
    rows.append(dict(corpus=corpus, condition="all 50 messages", accounts=len(y),
                     bots=int(y.sum()), auc=round(auc, 4), f1_bot=round(f1, 4),
                     f1_macro=round(f1m, 4)))
    print(f"{corpus}: {len(y)} accounts, AUC {auc:.4f}, F1 (bot class) {f1:.4f}, macro F1 {f1m:.4f}",
          flush=True)

    # the truncated condition \S2.4 also quotes: 20 messages per account, joined and cut to 25
    # words, averaged over draws so one unlucky sample does not decide the number
    per = {a: list(v) for a, v in g.text}
    rng = np.random.default_rng(SEED)
    aucs, f1s = [], []
    for _ in range(DRAWS):
        docs_t = [" ".join(" ".join(rng.choice(per[a], min(NMSG, len(per[a])), replace=False))
                           .split()[:WORDS]) for a in g.groups]
        pt = cross_val_predict(PIPE(), np.asarray(docs_t, dtype=object), y,
                               cv=StratifiedKFold(5, shuffle=True, random_state=0),
                               method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, pt)); f1s.append(f1_score(y, (pt >= 0.5).astype(int)))
    # A perfect score has to survive being attacked before it is printed. Shuffled labels must
    # fall to chance, the result must not depend on the fold seed, no document may appear twice or
    # in both classes, and removing the phrase Fox8-23's bots were selected by must change nothing.
    yshuf = np.random.default_rng(SEED).permutation(y)
    ps = cross_val_predict(PIPE(), np.asarray(docs, dtype=object), yshuf,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    null = roc_auc_score(yshuf, ps)
    p7 = cross_val_predict(PIPE(), np.asarray(docs, dtype=object), y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=7),
                           method="predict_proba")[:, 1]
    seed7 = roc_auc_score(y, p7)
    dups = int(pd.Series(docs).duplicated().sum())
    shared = len(set(docs[y == 1]) & set(docs[y == 0]))
    nophrase = np.array([re.sub(r"as an ai language model", " ", x, flags=re.I) for x in docs],
                        dtype=object)
    pn = cross_val_predict(PIPE(), nophrase, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    stripped = roc_auc_score(y, pn)
    print(f"  checks: shuffled-label null {null:.4f}, fold seed 7 {seed7:.4f}, duplicate docs "
          f"{dups}, shared across classes {shared}, phrase removed {stripped:.4f}", flush=True)
    assert null < 0.6 and dups == 0 and shared == 0, "the perfect score is an artefact"
    rows.append(dict(corpus=corpus, condition="checks", accounts=len(y), bots=int(y.sum()),
                     auc=round(float(seed7), 4), f1_bot=None, f1_macro=None,
                     shuffled_null=round(float(null), 4), duplicate_docs=dups,
                     shared_docs=shared, auc_phrase_removed=round(float(stripped), 4)))

    rows.append(dict(corpus=corpus, condition=f"{NMSG} messages cut to {WORDS} words",
                     accounts=len(y), bots=int(y.sum()), auc=round(float(np.mean(aucs)), 4),
                     f1_bot=round(float(np.mean(f1s)), 4), f1_macro=None))
    print(f"  cut to {WORDS} words: AUC {np.mean(aucs):.4f}, F1 {np.mean(f1s):.4f} "
          f"over {DRAWS} draws", flush=True)
pd.DataFrame(rows).to_csv(D / "baseline_f1.csv", index=False)
print("wrote baseline_f1.csv")
record_run(args.output_dir, "baseline_f1", [args.cohort])
