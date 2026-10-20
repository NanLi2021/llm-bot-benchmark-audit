"""Masquerade-rebuilt, two constructions (historical content_signal/masquerade_rebuilt.py and
masquerade_rebuilt_table1_fields.py).

1. The two-platform construction behind Table 1's cell: Chirper accounts against TwiBot-20 humans.
2. The detector papers' recipe (the appendix's Masquerade table): all labelled TwiBot-20 accounts, with
   3,000 (BotDMM, tweets and bio) or 3,500 (LLM-BotGuard, tweets only) bot accounts carrying a Chirper
   account's posts and relabelled LLM-driven; three classes; split 70/20/10. The bag of words is scored
   three-way and on two binary readings; the attribute baseline on Table 1's TwiBot-20 field set.
Out: masquerade_rebuilt_baselines.csv, masquerade_rebuilt_table1_fields.csv
"""
import json, pathlib, re, zipfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from .common import arguments, record_run
args = arguments(__doc__)
PROBES = args.output_dir
CHIRPER = args.data_root / 'Masquerade-23/SDAR/chirper_user_en_cleaned.json'
TB20 = args.data_root / 'TwiBot-20/Twibot-20.zip'
CORPUS = 'Masquerade-rebuilt'
TWO_PLATFORM = 'Chirper accounts vs TwiBot-20 humans'
_URL = re.compile(r"https?://\S+|t\.co/\S+|pic\.twitter\.com/\S+")
_LEAD = re.compile(r"^(?:\s*@\w+)+")
clean = lambda s: re.sub(r"\s+", " ", _URL.sub(" ", str(s))).strip()      # masquerade_two_platforms.py
norm = lambda t: _URL.sub(" ", _LEAD.sub("", str(t))).strip()             # kc_dump.py
# BotDMM's cleaning: "removal of control characters, emoji symbols, and URL artifacts"
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d\U0001F1E6-\U0001F1FF]")
clean_recipe = lambda s: re.sub(r"\s+", " ", _EMOJI.sub(" ", _CTRL.sub(" ", _URL.sub(" ", str(s))))).strip()
RECIPES = {"BotDMM": dict(n_replace=3000, replace_bio=True),       # BotEvo follows this one
           "LLM-BotGuard": dict(n_replace=3500, replace_bio=False)}
CLS = {0: "human", 1: "traditional bot", 2: "LLM-driven bot"}
# Table 1's TwiBot-20 attribute field set (profile_cells_cv.py), for masquerade_rebuilt_table1_fields
TABLE1_FIELDS = ["followers", "friends", "statuses", "favourites", "listed", "ff_ratio", "bio_len"]
NPC, KMAX, DRAWS, WORDS, NMSG = 500, 50, 10, 25, 20
PIPE = lambda: make_pipeline(TfidfVectorizer(min_df=2, max_features=50_000, sublinear_tf=True),
                             LogisticRegression(max_iter=2000))
CV = lambda seed: StratifiedKFold(5, shuffle=True, random_state=seed)
NJOBS = 1
# profile_cells_cv.py
NPERM = 10


r4 = lambda d: {k: round(float(v), 4) for k, v in d.items()}

def _num(s):
    """'12345 ', '1.2K', None -> float. Chirper prints a few follower counts with a K suffix."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KkMm])?", str(s).strip().replace(",", ""))
    if not m:
        return np.nan
    return float(m.group(1)) * {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1.0)

def load_sdar():
    """Every Chirper.ai account of the English channel, in file order, posts kept as a list."""
    out = []
    with open(CHIRPER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            bio = str(r.get("user_bio") or "").strip()
            out.append(dict(id=str(r.get("user_herf")), name=str(r.get("user_name") or ""),
                            handle=str(r.get("user_herf") or "").rstrip("/").rsplit("/", 1)[-1],
                            followers=_num(r.get("follower_num")), post_count=_num(r.get("tweet_num")),
                            description=bio, posts=[str(x) for x in (r.get("tweet_content") or [])]))
    return out

def load_twibot20():
    """Every labelled TwiBot-20 account (train, dev, test of the release zip), in file order."""
    out = []
    with zipfile.ZipFile(TB20) as z:
        for split in ("train", "dev", "test"):
            for o in json.load(z.open(f"Twibot-20/{split}.json")):
                lab = o.get("label")
                if lab not in ("0", "1", 0, 1):
                    continue
                p = o.get("profile") or {}
                bio = str(p.get("description") or "").strip()
                bio = "" if bio == "None" else bio
                out.append(dict(id=str(o.get("ID")), y=int(lab), name=str(p.get("name") or "").strip(),
                                handle=str(p.get("screen_name") or "").strip(),
                                followers=_num(p.get("followers_count")), friends=_num(p.get("friends_count")),
                                statuses=_num(p.get("statuses_count")), favourites=_num(p.get("favourites_count")),
                                listed=_num(p.get("listed_count")),
                                verified=float(str(p.get("verified")).strip().lower() == "true"),
                                description=bio, posts=[str(x) for x in (o.get("tweet") or [])]))
    return out

def load_two_platform(seed=0):
    """Chirper accounts against TwiBot-20 humans, at least five words each, balanced at 5,158 per class:
    the rule, order of draws and seed of masquerade_two_platforms.py."""
    rng = np.random.default_rng(seed)
    bots = [dict(r, source="Chirper.ai", label="bot", y=1) for r in load_sdar()
            if len(clean(" ".join(r["posts"])).split()) >= 5]
    hums = [dict(r, source="TwiBot-20", label="human", y=0, post_count=r["statuses"]) for r in load_twibot20()
            if r["y"] == 0 and len(clean(" ".join(r["posts"])).split()) >= 5]
    k = min(len(bots), len(hums))
    ib = rng.choice(len(bots), k, replace=False)
    ih = rng.choice(len(hums), k, replace=False)
    cols = ["id", "source", "label", "y", "name", "handle", "followers", "post_count", "description", "posts"]
    df = pd.DataFrame([{c: bots[i][c] for c in cols} for i in ib] + [{c: hums[i][c] for c in cols} for i in ih])
    df["has_bio"] = (df.description.str.len() > 0).astype(float)
    df["bio_len"] = df.description.str.len().astype(float)
    print(f"{CORPUS}, two-platform: Chirper {len(bots):,} eligible, TwiBot-20 humans {len(hums):,} eligible, "
          f"balanced at {k:,} per class (seed {seed})", flush=True)
    return df

def cv_auc(v, y, seed=None):
    ok = np.isfinite(v)
    if ok.sum() < 50 or len(np.unique(v[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    pipe = make_pipeline(QuantileTransformer(output_distribution="normal",
                                             n_quantiles=min(1000, int(ok.sum())), random_state=seed),
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

def oof(docs, y, seed=0, method="predict_proba"):
    p = cross_val_predict(PIPE(), np.asarray(docs, dtype=object), y, cv=CV(seed), method=method,
                          n_jobs=NJOBS)
    return p[:, 1] if method == "predict_proba" else p

def binary(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p), f1_bot=f1_score(y, yhat),
                f1_macro=f1_score(y, yhat, average="macro"), acc=accuracy_score(y, yhat))

def two_platform_baselines(df, seed, rows):
    row = lambda **kw: rows.append(dict(construction=TWO_PLATFORM, corpus=CORPUS, **kw))
    rng = np.random.default_rng(seed)

    # bag of words, the kc_dump cohort rule: 500 per class with 50 posts each
    usable = df.posts.map(lambda ps: [t for t in (norm(x) for x in ps) if len(t) >= 20])
    df["n_usable_posts"] = usable.map(len)
    picked = []
    for cls in (1, 0):                                   # bots first, then humans, as kc_dump.py
        keys = list(df.index[df.y == cls]); rng.shuffle(keys); n = 0
        for i in keys:
            if len(usable[i]) >= KMAX:
                picked.append(i); n += 1
            if n >= NPC:
                break
    idx = np.array(picked)
    y = df.y.values[idx]
    assert (y == 1).sum() == NPC and (y == 0).sum() == NPC, "the 500/500 cohort is short"
    per = {i: usable[i][:KMAX] for i in idx}
    docs = np.asarray([" ".join(per[i]) for i in idx], dtype=object)
    p50 = oof(docs, y, seed)
    s = binary(y, p50)
    print(f"  bag of words, all 50 messages: {len(y)} accounts, AUC {s['auc']:.4f}, F1 (bot class) "
          f"{s['f1_bot']:.4f}, macro F1 {s['f1_macro']:.4f}, accuracy {s['acc']:.4f}", flush=True)
    row(baseline="bag of words", task="LLM bot vs human", condition="all 50 messages", field=None,
        accounts=len(y), llm=int(y.sum()), humans=int((1 - y).sum()), **r4(s))

    # the checks baseline_f1.py runs before a perfect score is printed
    yshuf = np.random.default_rng(seed).permutation(y)
    null = roc_auc_score(yshuf, oof(docs, yshuf, seed))
    seed7 = roc_auc_score(y, oof(docs, y, 7))
    dups = int(pd.Series(docs).duplicated().sum())
    shared = len(set(docs[y == 1]) & set(docs[y == 0]))
    nophrase = [re.sub(r"as an ai language model", " ", x, flags=re.I) for x in docs]
    stripped = roc_auc_score(y, oof(nophrase, y, seed))
    print(f"    checks: shuffled-label null {null:.4f}, fold seed 7 {seed7:.4f}, duplicate docs {dups}, "
          f"shared across classes {shared}, phrase removed {stripped:.4f}", flush=True)
    assert null < 0.6 and dups == 0 and shared == 0, "the score is an artefact"
    row(baseline="bag of words", task="LLM bot vs human", condition="checks", field=None, accounts=len(y),
        auc=round(float(seed7), 4), null_auc=round(float(null), 4), duplicate_docs=dups,
        shared_docs=shared, auc_phrase_removed=round(float(stripped), 4))

    # the truncated condition: 20 posts per account, joined and cut to 25 words, averaged over draws
    cut = []
    for _ in range(DRAWS):
        dt = [" ".join(" ".join(rng.choice(per[i], min(NMSG, len(per[i])), replace=False))
                       .split()[:WORDS]) for i in idx]
        cut.append(binary(y, oof(dt, y, seed)))
    m = {k: np.mean([c[k] for c in cut]) for k in cut[0]}
    print(f"    cut to {WORDS} words: AUC {m['auc']:.4f}, F1 {m['f1_bot']:.4f}, accuracy {m['acc']:.4f} "
          f"over {DRAWS} draws", flush=True)
    row(baseline="bag of words", task="LLM bot vs human", condition=f"{NMSG} messages cut to {WORDS} words",
        field=None, accounts=len(y), **r4(m))

    # Table 1's cohort: the whole history of all 5,158 per class
    full = np.asarray([clean(" ".join(ps)) for ps in df.posts], dtype=object)
    yf = df.y.values
    pfull = oof(full, yf, seed)
    s = binary(yf, pfull)
    yshuf = np.random.default_rng(seed).permutation(yf)
    nullf = roc_auc_score(yshuf, oof(full, yshuf, seed))
    print(f"  bag of words, whole history (Table 1 cohort): {len(yf)} accounts, AUC {s['auc']:.4f}, "
          f"F1 {s['f1_bot']:.4f}, accuracy {s['acc']:.4f}, null {nullf:.4f}", flush=True)
    row(baseline="bag of words", task="LLM bot vs human", condition="whole history, Table 1 cohort",
        field=None, accounts=len(yf), llm=int(yf.sum()), humans=int((1 - yf).sum()),
        null_auc=round(float(nullf), 4), **r4(s))

    # one account attribute, the four fields both platforms ship, on Table 1's cohort
    F = {k: df[k].values.astype(float) for k in ("followers", "post_count", "has_bio", "bio_len")}
    per_field = {k: cv_auc(v, yf) for k, v in F.items()}
    for k, v in per_field.items():
        row(baseline="attribute", task="LLM bot vs human", condition="one field", field=k,
            accounts=int(np.isfinite(F[k]).sum()), auc=round(float(v), 4))
    k, a = best_field(F, yf)
    nulls = [best_field(F, rng.permutation(yf))[1] for _ in range(NPERM)]
    print("  attribute, one field each: " + ", ".join(f"{f} {v:.4f}" for f, v in per_field.items()))
    print(f"  attribute, best field: {k} CV AUC {a:.4f}   null {np.nanmean(nulls):.4f}   n={len(yf)}",
          flush=True)
    row(baseline="attribute", task="LLM bot vs human", condition="best field", field=k, accounts=len(yf),
        auc=round(float(a), 4), null_auc=round(float(np.nanmean(nulls)), 4))

    coh = df[["id", "source", "label", "y", "n_usable_posts", "followers", "post_count", "has_bio",
              "bio_len"]].copy()
    coh["in_50msg_cohort"] = coh.index.isin(idx)
    coh["p_oof_50msg"] = np.nan
    coh.loc[idx, "p_oof_50msg"] = p50
    coh["p_oof_whole_history"] = pfull
    coh.to_csv(PROBES / "masquerade_rebuilt_cohort.csv", index=False)

def build_recipe(tb20, sdar, n_replace, replace_bio, seed=0):
    """TwiBot-20 with n_replace of its bot accounts carrying a Chirper account's posts (and bio)."""
    rng = np.random.default_rng(seed)
    pool = [i for i, r in enumerate(sdar) if any(p.strip() for p in r["posts"])]
    bots = [i for i, r in enumerate(tb20) if r["y"] == 1]
    swap = dict(zip(rng.choice(bots, n_replace, replace=False), rng.choice(pool, n_replace, replace=False)))
    rows = []
    for i, r in enumerate(tb20):
        a = {k: v for k, v in r.items() if k != "posts"}
        posts, cls = r["posts"], r["y"]
        if i in swap:
            s = sdar[swap[i]]
            posts, cls = s["posts"], 2
            if replace_bio:
                a["description"] = s["description"]
        a.update(cls=cls, label=CLS[cls], text=clean_recipe(" ".join(posts)))
        rows.append(a)
    df = pd.DataFrame(rows)
    df["bio_len"] = df.description.str.len().astype(float)
    df["ff_ratio"] = df.followers / (df.friends + 1)
    # BotDMM: 70% training, 20% validation, 10% testing; BotEvo: stratified
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.3, stratify=df.cls, random_state=seed)
    va, te = train_test_split(rest, test_size=1 / 3, stratify=df.cls.values[rest], random_state=seed)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    return df

def threeway(c, pred):
    return dict(acc=accuracy_score(c, pred), f1_macro=f1_score(c, pred, average="macro"),
                f1_weighted=f1_score(c, pred, average="weighted"), mcc=matthews_corrcoef(c, pred))

def recipe_baselines(df, name, seed, rows):
    """The bag of words on the recipe corpus: three-way, then two binary readings."""
    row = lambda **kw: rows.append(dict(construction=f"{name} recipe", corpus=CORPUS, **kw))
    X, c = np.asarray(df.text.values, dtype=object), df.cls.values
    tr, te = (df.split == "train").values, (df.split == "test").values
    n = dict(accounts=len(df), humans=int((c == 0).sum()), traditional=int((c == 1).sum()),
             llm=int((c == 2).sum()))

    # three-way, their split: fit on the 70%, report on the 10%
    s = threeway(c[te], PIPE().fit(X[tr], c[tr]).predict(X[te]))
    snull = threeway(c[te], PIPE().fit(X[tr], np.random.default_rng(seed).permutation(c[tr])).predict(X[te]))
    print(f"  {name}, three-way, their split: acc {s['acc']:.4f}, macro F1 {s['f1_macro']:.4f}, "
          f"MCC {s['mcc']:.4f} (null macro F1 {snull['f1_macro']:.4f})", flush=True)
    row(baseline="bag of words", task="three-way", condition="their 70/20/10 split, test", field=None,
        test_accounts=int(te.sum()), null_acc=round(snull["acc"], 4),
        null_f1_macro=round(snull["f1_macro"], 4), **n, **r4(s))

    # three-way, five-fold CV over every account
    s = threeway(c, oof(X, c, seed, method="predict"))
    cs = np.random.default_rng(seed).permutation(c)
    snull = threeway(cs, oof(X, cs, seed, method="predict"))
    print(f"  {name}, three-way, five-fold CV: acc {s['acc']:.4f}, macro F1 {s['f1_macro']:.4f}, "
          f"MCC {s['mcc']:.4f} (null macro F1 {snull['f1_macro']:.4f})", flush=True)
    row(baseline="bag of words", task="three-way", condition="five-fold CV", field=None,
        null_acc=round(snull["acc"], 4), null_f1_macro=round(snull["f1_macro"], 4), **n, **r4(s))

    # binary readings: LLM-driven bot against human, all bots against humans
    for task, keep, pos in (("LLM bot vs human", np.isin(c, [0, 2]), 2),
                            ("all bots vs human", np.ones(len(c), bool), None)):
        y = ((c == pos) if pos is not None else (c > 0)).astype(int)
        ktr, kte = keep & tr, keep & te
        s = binary(y[kte], PIPE().fit(X[ktr], y[ktr]).predict_proba(X[kte])[:, 1])
        print(f"  {name}, {task}, their split: AUC {s['auc']:.4f}, F1 {s['f1_bot']:.4f}", flush=True)
        row(baseline="bag of words", task=task, condition="their 70/20/10 split, test", field=None,
            test_accounts=int(kte.sum()), **n, **r4(s))
        s = binary(y[keep], oof(X[keep], y[keep], seed))
        ys = np.random.default_rng(seed).permutation(y[keep])
        null = roc_auc_score(ys, oof(X[keep], ys, seed))
        print(f"  {name}, {task}, five-fold CV: AUC {s['auc']:.4f}, F1 {s['f1_bot']:.4f}, null {null:.4f}",
              flush=True)
        row(baseline="bag of words", task=task, condition="five-fold CV", field=None,
            null_auc=round(float(null), 4), **n, **r4(s))

def table1_fields(df, name, rng, rows):
    """The best single attribute from Table 1's TwiBot-20 field set; the null repeats the choice."""
    def best(F, y):
        s = {k: cv_auc(v, y, seed=0) for k, v in F.items()}
        k = max(s, key=s.get)
        return k, s[k], s
    for task, keep, pos in (("LLM bot vs human", df.cls.isin([0, 2]), 2), ("all bots vs human", df.cls >= 0, None),
                            ("LLM bot vs traditional bot", df.cls.isin([1, 2]), 2)):
        s = df[keep]
        y = ((s.cls == pos) if pos is not None else (s.cls > 0)).astype(int).values
        F = {k: s[k].values.astype(float) for k in TABLE1_FIELDS}
        k, a, per = best(F, y)
        null = np.nanmean([best(F, rng.permutation(y))[1] for _ in range(NPERM)])
        print(f"  {name}, {task}: best {k} {a:.4f}, null {null:.4f}, n={len(y)}", flush=True)
        rows.append(dict(recipe=name, task=task, field=k, cv_auc=round(a, 4), null=round(float(null), 4),
                         n=len(y), **{f"auc_{f}": round(v, 4) for f, v in per.items()}))

rows, fields = [], []
tb20, sdar = load_twibot20(), load_sdar()
frng = np.random.default_rng(0)
for name, kw in RECIPES.items():
    df = build_recipe(tb20, sdar, seed=0, **kw)
    print(f"{CORPUS}, {name} recipe: {len(df):,} accounts, {df.label.value_counts().to_dict()} (seed 0)", flush=True)
    recipe_baselines(df, name, 0, rows)
    table1_fields(df, name, frng, fields)
del tb20, sdar
two_platform_baselines(load_two_platform(0), 0, rows)
pd.DataFrame(rows).to_csv(PROBES / "masquerade_rebuilt_baselines.csv", index=False)
pd.DataFrame(fields).to_csv(PROBES / "masquerade_rebuilt_table1_fields.csv", index=False)
record_run(PROBES, "masquerade", [CHIRPER, TB20], seed=0)
