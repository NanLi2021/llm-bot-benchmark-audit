"""Table 1's TwiBot-22 bag-of-words cell, from the reply pairs written by analysis.twibot22_reply_pairs.

Historical chain, with the one step that had no script now written out:
  1. analysis.twibot22_reply_pairs     replied_to edges + tweet shards -> (reply, parent) pairs   [historical]
  2. keep pairs where reply and parent are English and the reply is not to its author's own thread
                                       [had no script; this definition reproduces the historical
                                        file exactly, 582,558 rows, and is the one the historical
                                        extractor's --audit mode prints]
  3. one document per labelled account: its kept replies joined, cut to 20,000 characters, in the order of
     the labelled accounts in user.json                                                       [historical]
  4. TF-IDF over 1-2 grams (min_df 3, sublinear tf, 300k features) + logistic regression, stratified 5-fold
     CV with shuffle seed 0; shuffled-label null with generator seed 0                           [historical]
Reported on the accounts that have at least one kept reply (6,920), the subset Table 1's caption states.
Inputs: --pairs (twibot22_reply_parent.parquet), DATA_ROOT/TwiBot-22/{label.csv,user.json}.
Out: twibot22_bag_of_words.csv
"""
import argparse
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .common import output_directory, record_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--pairs", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
OUT = output_directory(args.output_dir)
TB = args.data_root.resolve(strict=True) / "TwiBot-22"

# step 2: English replies to English parents, not self-threads
pairs = pd.read_parquet(args.pairs, columns=["reply_author", "reply_text", "reply_lang", "parent_lang", "self_thread"])
kept = pairs[(pairs.reply_lang == "en") & (pairs.parent_lang == "en") & (~pairs.self_thread)]
print(f"pairs {len(pairs):,} -> English, not self-thread {len(kept):,}", flush=True)

# step 3: labelled accounts in user.json order, each with its kept replies
lab = pd.read_csv(TB / "label.csv")
labelled = set(lab.id)
order = []
with open(TB / "user.json", "rb") as fh:
    for u in ijson.items(fh, "item"):
        if u.get("id") in labelled:
            order.append(u["id"])
df = pd.DataFrame({"id": order}).merge(lab, on="id", how="inner")
text = kept.groupby("reply_author").reply_text.apply(lambda v: " ".join(v.astype(str))[:20000])
df["tweet_text"] = df.id.map(text).fillna("")
df = df[df.tweet_text.str.strip().str.len() > 0]

# step 4: bag of words
X = np.asarray(df.tweet_text.tolist(), dtype=object)
y = (df.label == "bot").astype(int).values
mk = lambda: make_pipeline(
    TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True, max_features=300_000),
    LogisticRegression(max_iter=2000))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
auc = roc_auc_score(y, cross_val_predict(mk(), X, y, cv=cv, method="predict_proba")[:, 1])
ys = np.random.default_rng(0).permutation(y)
null = roc_auc_score(ys, cross_val_predict(mk(), X, ys, cv=cv, method="predict_proba")[:, 1])
pd.DataFrame([dict(corpus="TwiBot-22", pairs_kept=len(kept), accounts=len(y), bots=int(y.sum()),
                   bag_of_words=round(auc, 4), null=round(null, 4))]).to_csv(OUT / "twibot22_bag_of_words.csv", index=False)
print(f"accounts {len(y)} | bots {int(y.sum())} | bag of words {auc:.4f} | null {null:.4f}", flush=True)
record_run(OUT, "twibot22_bag_of_words", [args.pairs, TB / "label.csv"])
