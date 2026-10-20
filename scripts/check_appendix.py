"""Check that every decimal, percentage and four-digit-or-larger integer in paper/appendix.tex has a source.

A number passes if it appears in a results CSV (as stored, rounded to 1-4 decimals, as a thousands-separated
integer, or as a share in per cent), or if it is listed in NOT_RESULTS with what it is.
Usage: python scripts/check_appendix.py [--results results/expected]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOT_RESULTS = {
    # quoted from the detector papers (BotDMM, LLM-BotGuard, BotEvo)
    "0.9542": "published", "0.9362": "published", "0.9025": "published", "0.8850": "published",
    "0.8851": "published", "0.8212": "published", "0.8817": "published", "0.8930": "published",
    # parameters of the methods and of the published constructions
    "10,000": "quantile-transform subsample", "20,000": "characters per TwiBot-22 account",
    "3,000": "BotDMM recipe", "3,500": "LLM-BotGuard recipe",
    "70%": "the papers' 70/20/10 split", "10%": "the papers' 70/20/10 split",
    # section numbers of the paper
    "2.1": "section", "2.2": "section", "2.3": "section",
    # medians recomputed by scripts/check_results.py
    "0.801": "median, io_topic_matched M0", "0.621": "median, io_topic_matched M2",
}


def variants(v):
    out = {v.strip()}
    try:
        x = float(v)
    except ValueError:
        return out
    if x != x or abs(x) == float("inf"):
        return out
    for r in (1, 2, 3, 4):
        out.add(f"{x:.{r}f}")
    if x == int(x):
        out.add(f"{int(x):,}")
    if 0 <= x <= 1:
        out.add(f"{100 * x:.1f}%")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results/expected")
    a = p.parse_args()
    tex = (ROOT / "paper/appendix.tex").read_text()
    body = tex[tex.index("\\begin{document}"):]
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\(cite[pt]?|ref|label)\{[^}]*\}", "", body).replace("\\%", "%")
    numbers = set(re.findall(r"(?<![\w.])(\d+\.\d+%?|\d{1,3}(?:,\d{3})+|\d+%)(?![\w.])", body))
    found = set()
    for f in a.results.glob("*.csv"):
        for col in pd.read_csv(f, dtype=str).astype(str).values.T:
            for v in set(col):
                found |= variants(v)
    missing = sorted(n for n in numbers if n not in found and n not in NOT_RESULTS)
    print(f"{len(numbers)} numbers in the appendix; {len(numbers) - len(missing)} sourced")
    for n in missing:
        i = body.find(n)
        print(f"  no source: {n}   ...{' '.join(body[max(0, i - 60):i + 10].split())}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
