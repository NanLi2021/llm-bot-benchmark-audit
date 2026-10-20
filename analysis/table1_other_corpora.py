"""Table 1's baseline cells for the corpora that are not candidates: TwiBot-20, TwiBot-22, MGTAB and Twitter IO.

Each block is the historical producer's code with explicit input paths; no protocol is changed.
  TwiBot-20  bag of words                  historical fill_table_dashes.py, block 1
  TwiBot-20  best single account attribute historical profile_cells_cv.py, TwiBot-20 block
  MGTAB      best single account attribute historical profile_cells_cv.py, MGTAB block (needs torch)
  TwiBot-22  best single account attribute historical profile_cells_cv2.py, TwiBot-22 block (needs ijson)
  Twitter IO best single account attribute historical profile_cells_cv2.py, IO block (range over campaigns)
  Twitter IO bag of words                  historical io_all_campaigns_gauge.py (range over campaigns)
The AUCs are deterministic (unshuffled or seeded folds; the quantile transformer is seeded, unlike the
historical runs, which changes TwiBot-22's cell from 0.7736 to the seeded value); shuffled-label nulls use one generator per block,
seeded 0, so a null can differ from the historical value where the historical script shared its generator
across blocks. TwiBot-22's bag-of-words cell is in twibot22_bag_of_words.py.

Inputs under DATA_ROOT: TwiBot-20/Twibot-20.zip, MGTAB/MGTAB/{features,labels_bot}.pt,
TwiBot-22/{label.csv,user.json}, campaigns/*/*.parquet.
Out: table1_other_corpora.csv, io_profile_field_cv.csv, io_all_campaigns_gauge.csv
"""
import glob, json, pathlib, re, zipfile
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from .common import arguments, record_run
args = arguments(__doc__)
OUT, BD = args.output_dir, args.data_root
TB20 = BD / "TwiBot-20/Twibot-20.zip"
NPERM = 10
num = lambda s: pd.to_numeric(pd.Series(s), errors="coerce").values.astype(float)
rows = []


def twibot20_records():
    with zipfile.ZipFile(TB20) as z:
        for split in ("train", "dev", "test"):
            name = next(n for n in z.namelist() if n.endswith(f"/{split}.json") or n == f"{split}.json")
            for r in json.loads(z.read(name)):
                yield r


# ---- single-attribute protocol (profile_cells_cv.py and profile_cells_cv2.py) ---------------
def cv_auc(v, y):
    ok = np.isfinite(v)
    if ok.sum() < 50 or len(np.unique(v[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    # random_state=0: above 10,000 accounts (TwiBot-20, MGTAB, TwiBot-22) the transformer estimates its
    # quantiles on a random subsample; the historical runs left it unseeded
    pipe = make_pipeline(QuantileTransformer(output_distribution="normal", random_state=0,
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


def attribute_cell(name, F, y, rng, quiet=False):
    k, a = best_field(F, y)
    nulls = [best_field(F, rng.permutation(y))[1] for _ in range(NPERM)]
    if not quiet:
        print(f"{name:<14} best attribute {str(k):<20} CV AUC {a:.4f}   null {np.nanmean(nulls):.4f}   "
              f"n={len(y)}", flush=True)
    return dict(corpus=name, field=k, cv_auc=round(a, 4), null=round(float(np.nanmean(nulls)), 4),
                n=len(y), bots=int(y.sum()))


# ---- 1. TwiBot-20, bag of words over each account's shipped tweets (fill_table_dashes.py) ----
rng = np.random.default_rng(0)
_URL = re.compile(r"https?://\S+|t\.co/\S+|pic\.twitter\.com/\S+")
clean = lambda s: re.sub(r"\s+", " ", _URL.sub(" ", str(s))).strip()
docs, lab = [], []
for r in twibot20_records():
    d = clean(" ".join(r.get("tweet") or []))
    if len(d.split()) >= 5:
        docs.append(d); lab.append(int(r["label"]))
X, y = np.asarray(docs, dtype=object), np.array(lab)
pipe = make_pipeline(TfidfVectorizer(min_df=2, max_features=50_000, sublinear_tf=True),
                     LogisticRegression(max_iter=2000, C=1.0))
auc = roc_auc_score(y, cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1])
ysh = rng.permutation(y)
null = roc_auc_score(ysh, cross_val_predict(pipe, X, ysh, cv=5, method="predict_proba")[:, 1])
print(f"TwiBot-20      bag of words {auc:.4f}   null {null:.4f}   n={len(y)}", flush=True)
rows.append(dict(corpus="TwiBot-20", cell="bag of words", field="", value=round(auc, 4),
                 null=round(null, 4), n=len(y)))

# ---- 2. TwiBot-20 and MGTAB, best single attribute (profile_cells_cv.py) ---------------------
rng = np.random.default_rng(0)
recs, lab = [], []
for r in twibot20_records():
    recs.append(r.get("profile") or {}); lab.append(int(r["label"]))
d, y = pd.DataFrame(recs), np.array(lab)
F = {k: num(d[k].astype(str).str.strip()) for k in
     ("followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count") if k in d}
F["ff_ratio"] = F["followers_count"] / (F["friends_count"] + 1)
if "description" in d:
    F["description_len"] = d.description.fillna("").astype(str).str.len().values.astype(float)
c = attribute_cell("TwiBot-20", F, y, rng)
rows.append(dict(corpus="TwiBot-20", cell="best attribute", field=c["field"], value=c["cv_auc"],
                 null=c["null"], n=c["n"]))

try:
    import torch
    X = torch.load(BD / "MGTAB/MGTAB/features.pt", map_location="cpu", weights_only=True).numpy()
    ym = torch.load(BD / "MGTAB/MGTAB/labels_bot.pt", map_location="cpu", weights_only=True).numpy()
    c = attribute_cell("MGTAB", {f"col{j}": X[:, j].astype(float) for j in range(20)}, ym, rng)
    rows.append(dict(corpus="MGTAB", cell="best attribute", field=c["field"], value=c["cv_auc"],
                     null=c["null"], n=c["n"]))
except ImportError:
    print("MGTAB skipped: its features ship as torch tensors; install requirements-mgtab.txt", flush=True)

# ---- 3. TwiBot-22, best single attribute (profile_cells_cv2.py) ------------------------------
import ijson
rng = np.random.default_rng(0)
lab_df = pd.read_csv(BD / "TwiBot-22/label.csv")
want = dict(zip(lab_df.id.astype(str), (lab_df.label == "bot").astype(int)))
keep = {}
with open(BD / "TwiBot-22/user.json", "rb") as fh:
    for u in ijson.items(fh, "item"):
        uid = str(u.get("id"))
        if uid in want:
            pm = u.get("public_metrics") or {}
            keep[uid] = dict(
                followers=pm.get("followers_count"), following=pm.get("following_count"),
                tweets=pm.get("tweet_count"), listed=pm.get("listed_count"),
                desc_len=len(str(u.get("description") or "")),
                verified=1.0 if u.get("verified") else 0.0,
                name_digits=sum(ch.isdigit() for ch in str(u.get("username") or "")))
        if len(keep) == len(want):
            break
k22 = pd.DataFrame(keep).T
y22 = np.array([want[i] for i in k22.index])
F = {col: num(k22[col]) for col in k22.columns}
F["ff_ratio"] = F["followers"] / (F["following"] + 1)
c = attribute_cell("TwiBot-22", F, y22, rng)
rows.append(dict(corpus="TwiBot-22", cell="best attribute", field=c["field"], value=c["cv_auc"],
                 null=c["null"], n=c["n"]))

# ---- 4. Twitter IO, best single attribute per campaign (profile_cells_cv2.py, IO block) ------
per = []
for cdir in sorted(p for p in glob.glob(str(BD / "campaigns/*")) if pathlib.Path(p).is_dir()):
    files = sorted(glob.glob(f"{cdir}/*.parquet"))
    if not files:
        continue
    name = pathlib.Path(cdir).name
    cols = ["accountid", "is_control", "follower_count", "following_count",
            "account_creation_date", "account_profile_description"]
    try:
        g = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                      ignore_index=True).groupby("accountid").first()
    except Exception as e:
        print(f"  {name}: skipped ({e})"); continue
    yy = (~g.is_control.astype(bool)).astype(int).values
    if min((yy == 1).sum(), (yy == 0).sum()) < 20:
        continue
    F = {"follower_count": num(g.follower_count), "following_count": num(g.following_count)}
    F["ff_ratio"] = F["follower_count"] / (F["following_count"] + 1)
    F["account_age_days"] = (pd.Timestamp("2020-01-01") -
                             pd.to_datetime(g.account_creation_date, errors="coerce")).dt.days.values.astype(float)
    F["description_len"] = g.account_profile_description.fillna("").astype(str).str.len().values.astype(float)
    per.append(attribute_cell(name, F, yy, rng, quiet=True))
pc = pd.DataFrame(per)
pc.to_csv(OUT / "io_profile_field_cv.csv", index=False)
print(f"Twitter IO     best attribute {pc.cv_auc.min():.2f} to {pc.cv_auc.max():.2f} over {len(pc)} campaigns",
      flush=True)
rows.append(dict(corpus="Twitter IO", cell="best attribute", field="varies",
                 value=f"{pc.cv_auc.min():.2f} to {pc.cv_auc.max():.2f}", null=round(pc.null.median(), 4),
                 n=len(pc)))

# ---- 5. Twitter IO, bag of words per campaign (io_all_campaigns_gauge.py) -------------------
POSTS_PER_ACCT, ACCTS_PER_CLASS, SEED = 200, 4000, 0
rng = np.random.default_rng(SEED)
gauge = []
for camp in sorted(p.name for p in (BD / "campaigns").iterdir() if p.is_dir()):
    fs = sorted(glob.glob(str(BD / "campaigns" / camp / "*.parquet")))
    if not fs:
        gauge.append(dict(campaign=camp, note="no parquet")); continue
    d = pd.concat([pd.read_parquet(f, columns=["post_text", "is_control", "accountid", "is_repost"])
                   for f in fs], ignore_index=True)
    d = d[d.is_repost != True]
    d["post_text"] = d.post_text.astype(str)
    d = d[d.post_text.str.len() >= 20]
    if "is_control" not in d or d.is_control.nunique() < 2:
        gauge.append(dict(campaign=camp, note="no is_control / single class")); continue
    acc = (d.groupby(["accountid", "is_control"]).post_text
             .apply(lambda s: " ".join(s.iloc[:POSTS_PER_ACCT])).reset_index())
    acc["y"] = (~acc.is_control.astype(bool)).astype(int)
    n_io, n_ct = int(acc.y.sum()), int((1 - acc.y).sum())
    if min(n_io, n_ct) < 20:
        gauge.append(dict(campaign=camp, note="fewer than 20 accounts in a class", n_io=n_io, n_control=n_ct))
        continue
    kept = pd.concat([grp.sample(min(len(grp), ACCTS_PER_CLASS), random_state=SEED)
                      for _, grp in acc.groupby("y")], ignore_index=True)
    txt, yy = np.asarray(list(kept.post_text), dtype=object), kept.y.to_numpy()
    cv = StratifiedKFold(min(5, min(np.bincount(yy))), shuffle=True, random_state=SEED)
    pipe = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=200000),
                         LogisticRegression(max_iter=2000))
    p = cross_val_predict(pipe, txt, yy, cv=cv, method="predict_proba")[:, 1]
    L = np.array([len(t) for t in txt], dtype=float)         # length-only reference column
    gauge.append(dict(campaign=camp, n_io=int(yy.sum()), n_control=int((1 - yy).sum()),
                      bag_of_words=round(roc_auc_score(yy, p), 4), length=round(roc_auc_score(yy, L), 4),
                      null=round(roc_auc_score(rng.permutation(yy), p), 4), note=""))
ga = pd.DataFrame(gauge)
ga.to_csv(OUT / "io_all_campaigns_gauge.csv", index=False)
s = ga[ga.bag_of_words.notna()] if "bag_of_words" in ga else ga.iloc[0:0]
print(f"Twitter IO     bag of words {s.bag_of_words.min():.2f} to {s.bag_of_words.max():.2f} over {len(s)} campaigns",
      flush=True)
rows.append(dict(corpus="Twitter IO", cell="bag of words", field="",
                 value=f"{s.bag_of_words.min():.2f} to {s.bag_of_words.max():.2f}",
                 null=round(s.null.median(), 4), n=len(s)))

pd.DataFrame(rows).to_csv(OUT / "table1_other_corpora.csv", index=False)
record_run(OUT, "table1_other_corpora", [TB20, BD / "TwiBot-22/label.csv"])
