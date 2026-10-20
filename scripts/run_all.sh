#!/usr/bin/env bash
# Rerun every analysis behind the paper's printed numbers, then check them.
# Usage: scripts/run_all.sh DATA_ROOT [OUTPUT_DIR]     (DATA_ROOT layout: data/sources.csv)
# CPU only. The longest steps are the TwiBot-22 reply extraction (about 100 GB of tweets) and the
# Twitter IO matching (ten draws, twenty permutations per rung).
set -euo pipefail
# One BLAS/OpenMP thread: with hundreds of threads the fourth decimal can move (Iran_5 0.9938 vs 0.9937).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATA=$(cd "${1:?usage: scripts/run_all.sh DATA_ROOT [OUTPUT_DIR]}" && pwd)
OUT=${2:-build/run}
PY=${PYTHON:-.venv/bin/python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
run() { echo "== $*"; "$PY" -m "$@"; }

run analysis.cohorts            --data-root "$DATA" --output-dir "$OUT"
run analysis.baseline_f1        --data-root "$DATA" --output-dir "$OUT" --cohort "$OUT/cohort.parquet"
run analysis.baseline_separation --data-root "$DATA" --output-dir "$OUT" --cohort "$OUT/cohort.parquet"
run analysis.phrase_removal_f1  --data-root "$DATA" --output-dir "$OUT" --cohort "$OUT/cohort.parquet"
run analysis.transfer           --data-root "$DATA" --output-dir "$OUT" --cohort "$OUT/cohort.parquet"
for m in profile_cells_cv profile_cells_cv2 botsim_bio_share route_vocabulary route_attributes route_provenance_table \
         masquerade table1_other_corpora twibot22_reply_pairs; do
  run "analysis.$m" --data-root "$DATA" --output-dir "$OUT"
done
run analysis.twibot22_bag_of_words --data-root "$DATA" --output-dir "$OUT" --pairs "$OUT/twibot22_reply_parent.parquet"

# Twitter IO matching needs an empty directory of its own; its aggregate tables are copied next to the rest.
IO="$OUT/io-matching"
rm -rf "$IO"
run analysis.io_matching --data-root "$DATA" --output-dir "$IO" --protocol paired --account-conflicts exclude --score
cp "$IO/io_topic_matched.csv" "$OUT/io_topic_matched.csv"
cp "$IO/balance.csv" "$OUT/io_matching_balance.csv"
cp "$IO/input_exclusions.csv" "$OUT/io_matching_exclusions.csv"

"$PY" figures/fig_route_column.py --results "$OUT" --output-dir "$OUT/figures"
"$PY" scripts/check_results.py --actual "$OUT" --output "$OUT/check.json"
