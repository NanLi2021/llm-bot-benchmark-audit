"""Reconstruct route contingency counts from authorized source releases."""
import glob, gzip, json, os, re, zipfile
import numpy as np
import pandas as pd

from .common import arguments, record_run
args = arguments(__doc__)
ROWS = []
CT_ROWS = []


def cramers_v(ct):
    o = ct.to_numpy(dtype=float)
    n = o.sum()
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    chi2 = ((o - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (n * (min(o.shape) - 1))))


def summarise(name, df):
    """df: one row per account, columns `route` and `label`."""
    n = len(df)
    ct = pd.crosstab(df.route, df.label)
    for route, row in ct.iterrows():
        for label, count in row.items():
            CT_ROWS.append(dict(corpus=name, route=route, label=label, accounts=int(count)))
    single = ct[(ct > 0).sum(axis=1) == 1]
    acc_in_single = int(ct.loc[single.index].to_numpy().sum())
    route_only = int(ct.max(axis=1).sum()) / n
    base = df.label.value_counts().max() / n
    ROWS.append(dict(corpus=name, accounts=n, routes=len(ct),
                     single_class_routes=f"{len(single)}/{len(ct)}",
                     pct_accounts_in_single_class_routes=round(100 * acc_in_single / n, 1),
                     route_only_accuracy=round(route_only, 4),
                     majority_base_rate=round(base, 4),
                     gain_over_base=round(route_only - base, 4),
                     cramers_v=round(cramers_v(ct), 4)))
    print(f"\n=== {name}  ({n} accounts, {len(ct)} routes) ===")
    print(ct.to_string())
    print(f"  accounts in single-class routes : {acc_in_single}/{n} ({100*acc_in_single/n:.1f}%)")
    print(f"  route-only accuracy             : {route_only:.4f}   (base rate {base:.4f})")
    print(f"  Cramer's V                      : {ROWS[-1]['cramers_v']:.4f}")


# ---------------------------------------------------------------- TwiBot-20
# Its release records each account's seed domains (a combination of politics, business, entertainment
# and sports; 15 combinations occur), the seed groups it was crawled from, so they are a route by the
# definition above. Read from the release zip.
TB20 = args.data_root / "TwiBot-20/Twibot-20.zip"
_URL = re.compile(r"https?://\S+|t\.co/\S+|pic\.twitter\.com/\S+")
clean = lambda s: re.sub(r"\s+", " ", _URL.sub(" ", str(s))).strip()
rows, tb_h = [], 0
with zipfile.ZipFile(TB20) as z:
    for split in ("train", "dev", "test"):
        for o in json.load(z.open(f"Twibot-20/{split}.json")):
            lab = o.get("label")
            dom = o.get("domain")
            if lab in ("0", "1", 0, 1) and dom:
                rows.append((str(dom).strip(), int(lab)))
            # the human side of Masquerade-rebuilt: TwiBot-20 humans with at least five words of tweets
            if lab in ("0", 0) and len(clean(" ".join(o.get("tweet") or [])).split()) >= 5:
                tb_h += 1
summarise("TwiBot-20", pd.DataFrame(rows, columns=["route", "label"]))

# ---------------------------------------------------------------- Fox8-23
FOX8 = args.data_root / "fox8_23_dataset.ndjson.gz"
rows = []
with gzip.open(FOX8, "rt") as fh:
    for line in fh:
        o = json.loads(line)
        rows.append((o.get("dataset"), o["label"]))
summarise("Fox8-23", pd.DataFrame(rows, columns=["route", "label"]))

# ---------------------------------------------------------------- BotSim-24 (by construction)
u = pd.read_csv(args.data_root / "BotSim-24-Dataset/Users.csv")
route = ["scraped from Reddit"] * 1907 + ["generated for the corpus"] * (len(u) - 1907)
label = ["human"] * 1907 + ["bot"] * (len(u) - 1907)
summarise("BotSim-24", pd.DataFrame({"route": route, "label": label}))

# ---------------------------------------------------------------- Masquerade-rebuilt (by construction)
# The cohort masquerade_two_platforms.py scores: Chirper accounts with at least five words of text
# against as many TwiBot-20 humans, balanced at min(chirper, tb_h) per class.
chirper = 0
with open(args.data_root / "Masquerade-23/SDAR/chirper_user_en_cleaned.json") as f:
    for line in f:
        line = line.strip()
        if line and len(clean(" ".join(json.loads(line).get("tweet_content") or [])).split()) >= 5:
            chirper += 1
k = min(chirper, tb_h)
print(f"\nMasquerade-rebuilt cohort: Chirper {chirper}, TwiBot-20 humans {tb_h}, balanced at {k} per class")
summarise("Masquerade-rebuilt", pd.DataFrame({"route": ["Chirper.ai"] * k + ["TwiBot-20"] * k,
                                              "label": ["bot"] * k + ["human"] * k}))

# ---------------------------------------------------------------- Meng et al. 2026 (Bluesky)
MENG = args.data_root / "bluesky/meng2026_dataset_label_raw.csv"
m = pd.read_csv(MENG)
summarise("Meng 2026", m.rename(columns={"Source": "route", "Label": "label"})[["route", "label"]])
# the appendix's "460 of its 4,215 labelled bots are marked as generative" (operation_label 1)
pd.DataFrame([dict(corpus="Meng 2026", bots=int((m.Label == "bot").sum()), humans=int((m.Label == "human").sum()),
                   bots_generative=int(((m.Label == "bot") & (m.operation_label == 1)).sum()))]
             ).to_csv(args.output_dir / "bluesky_label_counts.csv", index=False)

# ---------------------------------------------------------------- Twitter IO campaigns
CAMP = args.data_root / "campaigns"
rows = []
skipped = []
for d in sorted(os.listdir(CAMP)):
    files = sorted(glob.glob(f"{CAMP}/{d}/*.parquet"))
    if not files:
        continue
    try:
        c = pd.concat([pd.read_parquet(f, columns=["accountid", "is_control"]) for f in files],
                      ignore_index=True)
    except Exception as e:                      # campaign without the is_control column
        skipped.append((d, type(e).__name__))
        continue
    c = c.drop_duplicates("accountid")
    c["route"] = d
    c["label"] = c.is_control.map(lambda v: "control" if v is True else "io")
    rows.append(c[["route", "label"]])
print(f"\ncampaigns without an is_control column, skipped: {skipped}")
io = pd.concat(rows, ignore_index=True)
summarise("Twitter IO campaigns", io)                      # diagnostic: the campaign alone
summarise("Twitter IO", io.assign(route=io.label.map(
    {"io": "Twitter's takedown archive", "control": "a hashtag and day search"})))

pd.DataFrame(skipped, columns=["campaign", "error_type"]).to_csv(args.output_dir / "route_skipped.csv", index=False)
out = pd.DataFrame(ROWS)
print("\n" + out.to_string(index=False))
out.to_csv(args.output_dir / "route_provenance_table.csv", index=False)
print("\nwrote route_provenance_table.csv in the requested output directory")
pd.DataFrame(CT_ROWS).to_csv(args.output_dir / "route_provenance_crosstab.csv", index=False)
print("wrote route_provenance_crosstab.csv in the requested output directory")
record_run(args.output_dir, "route_provenance_table", [TB20, FOX8, MENG,
    args.data_root / "BotSim-24-Dataset/Users.csv",
    args.data_root / "Masquerade-23/SDAR/chirper_user_en_cleaned.json"],
    campaign_hashes="Use input_manifest.py --hash for campaign content hashes")
