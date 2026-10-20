"""BotSim-24: what share of each class has a profile description ("bio")? (historical botsim_bio_share.py)

Table 1's BotSim-24 attribute cell is `has_description` (profile_cells_cv.py); this gives the counts behind
it. Population: every row of Users.csv, labelled by the release's own ordering (first 1,907 rows human, the
rest bot), as profile_cells_cv.py. A bio is a non-missing `description`; the second rule counts only
non-empty text after stripping whitespace.
Out: botsim_bio_share.csv
"""
import pandas as pd

from .common import arguments, record_run
args = arguments(__doc__)
USERS = args.data_root / "BotSim-24-Dataset/Users.csv"
u = pd.read_csv(USERS)
u["cls"] = ["human"] * 1907 + ["bot"] * (len(u) - 1907)
rows = []
for rule, has in (("description not missing", u.description.notna()),
                  ("description non-empty after strip", u.description.fillna("").astype(str).str.strip().ne(""))):
    for c, g in has.groupby(u.cls):
        rows.append(dict(rule=rule, cls=c, accounts=len(g), with_bio=int(g.sum()), share=round(float(g.mean()), 4)))
r = pd.DataFrame(rows)
r.to_csv(args.output_dir / "botsim_bio_share.csv", index=False)
print(r.to_string(index=False))
record_run(args.output_dir, "botsim_bio_share", [USERS])
