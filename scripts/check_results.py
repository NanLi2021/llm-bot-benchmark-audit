"""Check a run against results/expected/ and against every number the paper prints.

1. Row by row: each output CSV that has an expected counterpart must have the same rows (by key) and the same
   values, numbers within 5e-5.
2. Printed values: each number printed in the paper is recomputed from the outputs and compared at its printed
   precision.
Usage: python scripts/check_results.py --actual build/run [--expected results/expected] [--output report.json]
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

KEYS = {
    "baseline_f1": ["corpus", "condition"], "phrase_removal_f1": ["arm"],
    "profile_cells_cv": ["corpus"], "profile_cells_cv2": ["corpus"],
    "route_vocabulary": ["input", "comparison"], "route_attributes": ["study_a", "study_b"],
    "route_provenance_table": ["corpus"], "route_provenance_crosstab": ["corpus", "route", "label"],
    "masquerade_rebuilt_baselines": ["construction", "baseline", "task", "condition", "field"],
    "table1_other_corpora": ["corpus", "cell"], "io_profile_field_cv": ["corpus"],
    "io_all_campaigns_gauge": ["campaign"], "twibot22_bag_of_words": ["corpus"],
    "transfer_fox8_botsim": ["fit_on", "scored_on", "condition"],
    "io_topic_matched": ["campaign", "rung"], "io_matching_balance": ["campaign", "draw", "stage"],
    "io_matching_exclusions": ["campaign"],
    "baseline_separation": ["corpus"], "botsim_bio_share": ["rule", "cls"],
    "masquerade_rebuilt_table1_fields": ["recipe", "task"],
    "bluesky_label_counts": ["corpus"],
}


def med(s):
    return float(s.median())


# (label, file, function of the DataFrame, printed value, decimals)
PRINTED = [
    ("bag of words AUC, Fox8-23", "baseline_f1",
     lambda d: d.query("corpus == 'Fox8-23' and condition == 'all 50 messages'").auc.iloc[0], 1.0, 4),
    ("bag of words AUC, BotSim-24", "baseline_f1",
     lambda d: d.query("corpus == 'BotSim-24' and condition == 'all 50 messages'").auc.iloc[0], 1.0, 4),
    ("bag of words AUC, Masquerade-rebuilt", "masquerade_rebuilt_baselines",
     lambda d: d[d.condition.astype(str).str.startswith("whole history") & (d.baseline == "bag of words")].auc.iloc[0],
     0.9996, 4),
    ("phrases deleted, Fox8-23 AUC", "phrase_removal_f1", lambda d: d.query("arm == 'A1'").auc.iloc[0], 1.0, 4),
    ("phrase accounts dropped, Fox8-23 AUC", "phrase_removal_f1", lambda d: d.query("arm == 'A2'").auc.iloc[0], 1.0, 4),
    ("post count, Fox8-23", "profile_cells_cv2", lambda d: d.cv_auc.iloc[0], 0.9553, 4),
    ("has a bio, BotSim-24", "profile_cells_cv", lambda d: d.cv_auc.iloc[0], 0.9222, 4),
    ("route-only accuracy, Fox8-23", "route_provenance_table",
     lambda d: d.query("corpus == 'Fox8-23'").route_only_accuracy.iloc[0], 1.0, 3),
    ("route-only accuracy, TwiBot-20", "route_provenance_table",
     lambda d: d.query("corpus == 'TwiBot-20'").route_only_accuracy.iloc[0], 0.655, 3),
    ("majority rate, TwiBot-20", "route_provenance_table",
     lambda d: d.query("corpus == 'TwiBot-20'").majority_base_rate.iloc[0], 0.557, 3),
    ("Cramer's V, TwiBot-20", "route_provenance_table",
     lambda d: d.query("corpus == 'TwiBot-20'").cramers_v.iloc[0], 0.37, 2),
    ("Cramer's V, Bluesky", "route_provenance_table",
     lambda d: d.query("corpus == 'Meng 2026'").cramers_v.iloc[0], 0.99, 2),
    ("humans only, bag of words (median of six pairs)", "route_vocabulary",
     lambda d: med(d[(d.input == "full text, 20 messages") & d.auc.notna()].auc), 0.882, 3),
    ("humans only, best attribute (median of six pairs)", "route_attributes", lambda d: med(d.cv_auc), 0.775, 3),
    ("Twitter IO, two routes (median M0)", "io_topic_matched", lambda d: med(d.query("rung == 'M0'").auc), 0.801, 3),
    ("Twitter IO, one route (median M2)", "io_topic_matched", lambda d: med(d.query("rung == 'M2'").auc), 0.621, 3),
    ("TwiBot-20 bag of words", "table1_other_corpora",
     lambda d: float(d.query("corpus == 'TwiBot-20' and cell == 'bag of words'").value.iloc[0]), 0.8204, 4),
    ("TwiBot-20 best attribute", "table1_other_corpora",
     lambda d: float(d.query("corpus == 'TwiBot-20' and cell == 'best attribute'").value.iloc[0]), 0.7485, 4),
    ("MGTAB best attribute", "table1_other_corpora",
     lambda d: float(d.query("corpus == 'MGTAB' and cell == 'best attribute'").value.iloc[0]), 0.8622, 4),
    ("TwiBot-22 best attribute", "table1_other_corpora",
     lambda d: float(d.query("corpus == 'TwiBot-22' and cell == 'best attribute'").value.iloc[0]), 0.7733, 4),
    ("TwiBot-22 bag of words", "twibot22_bag_of_words", lambda d: d.bag_of_words.iloc[0], 0.6644, 4),
    ("Twitter IO bag of words per account (median of 25 campaigns)", "io_all_campaigns_gauge",
     lambda d: med(d.bag_of_words.dropna()), 0.97, 2),
    ("transfer, Fox8-23 -> BotSim-24", "transfer_fox8_botsim",
     lambda d: d.query("fit_on == 'Fox8-23' and condition == 'as released'").auc.iloc[0], 0.998, 3),
    ("transfer, BotSim-24 -> Fox8-23", "transfer_fox8_botsim",
     lambda d: d.query("fit_on == 'BotSim-24' and condition == 'as released'").auc.iloc[0], 0.890, 3),
]
RANGES = [("Twitter IO bag of words range", "table1_other_corpora", "bag of words", "0.81 to 0.99"),
          ("Twitter IO best attribute range", "table1_other_corpora", "best attribute", "0.54 to 0.99")]


def close(a, b):
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if math.isnan(x) and math.isnan(y):
        return True
    return math.isfinite(x) and math.isfinite(y) and abs(x - y) <= 0.00005001


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--expected", type=Path, default=Path(__file__).resolve().parents[1] / "results/expected")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    diffs, missing, compared = [], [], 0
    for name, keys in KEYS.items():
        exp, act = a.expected / f"{name}.csv", a.actual / f"{name}.csv"
        if not act.exists():
            missing.append(name); continue
        e, g = pd.read_csv(exp, dtype=str, keep_default_na=False), pd.read_csv(act, dtype=str, keep_default_na=False)
        e_idx, g_idx = e.set_index(keys), g.set_index(keys)
        if set(e_idx.index) != set(g_idx.index) or e_idx.index.duplicated().any():
            diffs.append({"file": name, "issue": "different or duplicate row keys"}); continue
        for k, row in e_idx.iterrows():
            for col, val in row.items():
                if col not in g_idx.columns:
                    diffs.append({"file": name, "key": str(k), "field": col, "issue": "missing column"}); continue
                if not close(val, g_idx.at[k, col]):
                    diffs.append({"file": name, "key": str(k), "field": col, "expected": val, "actual": g_idx.at[k, col]})
            compared += 1
    printed = []
    for label, name, f, value, dec in PRINTED:
        path = a.actual / f"{name}.csv"
        if not path.exists():
            continue
        got = round(float(f(pd.read_csv(path))), dec)
        ok = abs(got - value) < 10 ** (-dec) / 2
        printed.append({"number": label, "printed": value, "recomputed": got, "match": ok})
    for label, name, cell, value in RANGES:
        path = a.actual / f"{name}.csv"
        if path.exists():
            d = pd.read_csv(path)
            got = d.query("corpus == 'Twitter IO' and cell == @cell").value.iloc[0]
            printed.append({"number": label, "printed": value, "recomputed": got, "match": got == value})
    bad = [r for r in printed if not r["match"]]
    report = {"rows_compared": compared, "missing_outputs": missing, "row_differences": diffs,
              "printed_numbers": printed, "passed": not diffs and not bad}
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    for r in printed:
        print(f"{'ok ' if r['match'] else 'BAD'} {r['number']:<50} printed {r['printed']}  recomputed {r['recomputed']}")
    print(f"{compared} rows compared; {len(diffs)} row differences; {len(bad)} printed numbers off; "
          f"outputs not present: {', '.join(missing) or 'none'}")
    if diffs or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
