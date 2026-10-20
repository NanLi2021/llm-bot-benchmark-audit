"""TwiBot-22 (reply, parent) text pairs with labels; the input of analysis.twibot22_bag_of_words.

Historical producer: extract_twibot22_pairs.py, with explicit paths. It reads the replied_to edges and the post
edges from edge.csv, then streams the nine tweet_*.json shards (about 100 GB) in 256 MB chunks. Each shard is one
JSON array whose records all begin with {"attachments": and carry exactly one t-prefixed id, so splitting on that
key is safe; a chunk that yields no id is counted, never dropped silently.
Inputs: DATA_ROOT/TwiBot-22/{label.csv, edge.csv, tweet_0.json ... tweet_8.json}.
Out: twibot22_reply_parent.parquet (1,114,980 pairs in the historical run).
"""
import csv
import json

import pandas as pd

from .common import arguments, record_run
args = arguments(__doc__)
TB = args.data_root / "TwiBot-22"
OUT = args.output_dir / "twibot22_reply_parent.parquet"
DELIM = '{"attachments":'
CHUNK = 1 << 28  # 256 MB

lab = {}
with (TB / "label.csv").open() as f:
    for r in csv.reader(f):
        if r[0] != "id":
            lab[r[0]] = r[1]
print(f"labels: {len(lab):,}", flush=True)

pairs, need = [], set()
with (TB / "edge.csv").open() as f:
    for line in f:
        if ",replied_to," in line:
            s, _, t = line.rstrip("\n").split(",")
            pairs.append((s, t))
            need.add(s); need.add(t)
print(f"replied_to edges: {len(pairs):,} | distinct tweets needed: {len(need):,}", flush=True)

author = {}
with (TB / "edge.csv").open() as f:
    for line in f:
        if ",post," not in line:
            continue
        u, _, t = line.rstrip("\n").split(",")
        if t in need:
            author[t] = u
print(f"authors resolved: {len(author):,} of {len(need):,}", flush=True)

keep = {}
n_obj = n_noid = 0


def handle(obj_str):
    """obj_str is a full JSON object WITHOUT its leading delimiter."""
    global n_obj, n_noid
    n_obj += 1
    i = obj_str.find('"id": "t')
    if i < 0:
        n_noid += 1
        return
    j = obj_str.find('"', i + 8)
    tid = obj_str[i + 7:j]
    if tid not in need:
        return
    try:
        o = json.loads(DELIM + obj_str)
    except Exception:
        n_noid += 1
        return
    keep[tid] = (str(o.get("text") or ""), o.get("lang"), str(o.get("created_at") or ""),
                 str(o.get("conversation_id") or ""))


for fi in range(9):
    p = TB / f"tweet_{fi}.json"
    tail = ""
    with p.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            buf = tail + buf
            parts = buf.split(DELIM)
            tail = parts.pop()
            for seg in parts:
                seg = seg.strip()
                if not seg:
                    continue
                seg = seg.rstrip()
                if seg.endswith(","):
                    seg = seg[:-1]
                if seg.endswith("]"):
                    seg = seg[:-1].rstrip().rstrip(",")
                handle(seg)
    if tail.strip():
        seg = tail.strip().rstrip()
        if seg.endswith("]"):
            seg = seg[:-1].rstrip().rstrip(",")
        if seg.endswith(","):
            seg = seg[:-1]
        handle(seg)
    print(f"  {p.name}: cumulative kept {len(keep):,} / {len(need):,}", flush=True)

print(f"objects scanned {n_obj:,} | unparseable/no-id {n_noid:,}", flush=True)
rows = []
for rid, pid in pairs:
    r, q = keep.get(rid), keep.get(pid)
    if not r or not q:
        continue
    ra, pa = author.get(rid), author.get(pid)
    rows.append({"reply_id": rid, "parent_id": pid, "reply_author": ra, "parent_author": pa,
                 "reply_label": lab.get(ra), "parent_label": lab.get(pa),
                 "self_thread": ra is not None and ra == pa,
                 "reply_text": r[0], "reply_lang": r[1], "reply_time": r[2], "conv": r[3],
                 "parent_text": q[0], "parent_lang": q[1], "parent_time": q[2]})
d = pd.DataFrame(rows)
d.to_parquet(OUT)
print(f"wrote {OUT.name}  rows={len(d):,}", flush=True)
record_run(args.output_dir, "twibot22_reply_pairs", [TB / "label.csv"])
