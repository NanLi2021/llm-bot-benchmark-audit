# Data access

The repository contains code and aggregate results only: no posts, no account or post identifiers, no source files.
Each corpus stays under its owners' terms; the code and documentation licences do not cover them.

| corpus | how to obtain it | notes |
|---|---|---|
| Fox8-23 | Zenodo record 8035290 (`fox8_23_dataset.ndjson.gz`) | |
| BotSim-24 | the authors' GitHub repository (`BotSim-24-Dataset`) | labels follow the release's row order: the first 1,907 users are human |
| Masquerade-23 | the authors' GitHub repository, which links the data | English channel (`SDAR/chirper_user_en_cleaned.json`) |
| TwiBot-20 | apply to the authors | the release zip is read as shipped |
| TwiBot-22 | apply to the authors | `user.json` is streamed; the tweet shards are about 100 GB |
| MGTAB | the authors' GitHub repository | features ship as torch tensors; install `requirements-mgtab.txt` for that cell |
| Twitter IO | Zenodo record 14189193 and the per-campaign records | one directory of parquet files per campaign |
| Bluesky | OSF project qscf8 (`dataset_label_raw.csv`) | account labels and sources, no posts |
| CMV, Yard Signs | not needed | Table 1 describes them from their own papers |

The aggregate tables in `results/expected/` hold counts and scores per corpus, campaign, route or draw; none holds
text or identifiers.
