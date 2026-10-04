"""Split one big Stage 2b JSONL (rows of v3_alldates.csv) back into per-cell, per-date stage2b_<date>.jsonl."""
import json, sys, collections, pandas as pd
V3, csv_path, jsonl_path = sys.argv[1:4]
c=pd.read_csv(csv_path,usecols=["cell_id","date"],dtype=str)
buf=collections.defaultdict(list)
for line in open(jsonl_path):
    r=json.loads(line); i=r["sample_index"]; key=(c.cell_id[i],c.date[i]); buf[key].append(r)
for (cell,date),rows in buf.items():
    rows.sort(key=lambda r:r["sample_index"]); base=rows[0]["sample_index"]
    with open(f"{V3}/{cell}/dates/{date}/stage2b_{date}.jsonl","w") as f:
        for r in rows: r["sample_index"]-=base; f.write(json.dumps(r)+"\n")
print(len(buf),"cell-date files written")
