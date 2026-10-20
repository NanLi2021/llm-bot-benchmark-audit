# Benchmark corpora for detecting LLM-era bots: code and online appendix

Anonymous companion repository of our paper. It holds the code behind every number the paper
prints, the aggregate results those runs produced, and the [online appendix](paper/appendix.pdf). It holds no
dataset: see [data access](docs/data-access.md).

## Check without any data

```sh
python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.lock
make test     # synthetic and regression tests
make check    # every number in the paper and the appendix, against results/expected/
make figure   # Figure 1, from results/expected/ (written to build/figures/)
```

## Rerun everything

Obtain the corpora from their owners and arrange them under one directory as listed in
[data/sources.csv](data/sources.csv), then:

```sh
.venv/bin/python -m pip install -r requirements-mgtab.txt   # optional: only the MGTAB cell of Table 1
scripts/run_all.sh /path/to/inputs build/run
```

The script runs each analysis, redraws Figure 1 and ends with `scripts/check_results.py`, which compares every
output row with `results/expected/` and every printed number with its recomputed value. CPU only; the longest steps
are the TwiBot-22 reply extraction (about 100 GB of tweets) and the Twitter IO matching.

## Where each printed number comes from

| printed in the paper | value | produced by | output file |
|---|---|---|---|
| bag of words, Fox8-23 / BotSim-24 (Sect. 2.2, Table 1) | AUC and F1 1.0000 | `analysis.cohorts` then `analysis.baseline_f1` | `baseline_f1.csv` |
| shuffled-label nulls | 0.51, 0.50 | `analysis.baseline_f1`, `analysis.masquerade` | `baseline_f1.csv`, `masquerade_rebuilt_baselines.csv` |
| self-revealing phrases deleted or their accounts dropped | AUC 1.0000 | `analysis.phrase_removal_f1` | `phrase_removal_f1.csv` |
| bag of words, Masquerade-rebuilt | 0.9996 | `analysis.masquerade` | `masquerade_rebuilt_baselines.csv` |
| best single attribute, Fox8-23 / BotSim-24 | 0.9553 / 0.9222 | `analysis.profile_cells_cv2` / `analysis.profile_cells_cv` | `profile_cells_cv2.csv` / `profile_cells_cv.csv` |
| route-only accuracy, majority rate, Cramér's V (Table 1, Figure 1) | e.g. 1.000; TwiBot-20 0.655 vs 0.557, V 0.37 | `analysis.route_provenance_table` | `route_provenance_table.csv` |
| humans only: bag of words / best attribute (Sect. 2.3, Figure 1) | median 0.882 / 0.775 | `analysis.route_vocabulary` / `analysis.route_attributes` | `route_vocabulary.csv` / `route_attributes.csv` |
| Twitter IO, two routes and one route (Sect. 2.3, Figure 1) | median 0.801 and 0.621 | `analysis.io_matching --protocol paired --account-conflicts exclude --score` | `io_topic_matched.csv` |
| TwiBot-20, MGTAB, TwiBot-22, Twitter IO baseline cells (Table 1) | 0.8204, 0.7485, 0.8622, 0.7733, 0.81–0.99, 0.54–0.99 | `analysis.table1_other_corpora` | `table1_other_corpora.csv` |
| Twitter IO bag of words per account (Sect. 2.3) | median 0.97 over 25 campaigns | `analysis.table1_other_corpora` | `io_all_campaigns_gauge.csv` |
| TwiBot-22 bag of words (Table 1) | 0.6644 on 6,920 accounts | `analysis.twibot22_reply_pairs` then `analysis.twibot22_bag_of_words` | `twibot22_bag_of_words.csv` |
| Online appendix: bag of words fitted on one corpus, scored on the other | 0.998 and 0.890 | `analysis.transfer` | `transfer_fox8_botsim.csv` |
| Online appendix: score gap at the 0.5 threshold | 0.737 vs 0.471; 0.704 vs 0.240 | `analysis.baseline_separation` | `baseline_separation.csv` |
| Online appendix: BotSim-24 humans with a bio | 278 of 1,907 | `analysis.botsim_bio_share` | `botsim_bio_share.csv` |
| Online appendix: Masquerade-rebuilt by the detector papers' recipe | its table; 0.7495 | `analysis.masquerade` | `masquerade_rebuilt_baselines.csv`, `masquerade_rebuilt_table1_fields.csv` |
| Online appendix: Twitter IO per campaign, with the length column | its table | `analysis.table1_other_corpora` | `io_all_campaigns_gauge.csv` |
| Online appendix: Bluesky bots marked generative | 460 of 4,215 | `analysis.route_provenance_table` | `bluesky_label_counts.csv` |
| Figure 1 | — | `figures/fig_route_column.py` | `fig_route_column.pdf` |

Published scores of other detectors (TRACE-Bot, SAHG, BotDMM, LLM-BotGuard, BotEvo) are quoted from their papers.

## Layout

```text
analysis/          one module per analysis; each takes --data-root and --output-dir
figures/           Figure 1
scripts/           run_all.sh (full rerun) and check_results.py (row and printed-number checks)
results/expected/  aggregate outputs of the runs behind the paper (no text, no identifiers)
paper/             online appendix: appendix.pdf, appendix.tex, references.bib
data/              where the inputs go; sources.csv lists paths and official access
tests/             synthetic and regression tests
```

## Licence

Code: MIT ([LICENSE](LICENSE)). Appendix, figures, documentation and aggregate results: CC BY 4.0
([LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt)). The datasets remain under their owners' terms
([details](docs/licensing.md)).
