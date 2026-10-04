import json, os, numpy as np, pandas as pd
os.chdir("/media/data/building_instance_tamu/la_fire_2025/validation")
P=pd.read_csv("exp1_persistence_per_date.csv"); v=P[P.valid].sort_values("date")
ids=set(v.gt_index)
m=pd.read_csv("matched_v2.csv"); m=m[m.dins.notna()&m.gt_index.isin(ids)].copy(); m["dins"]=m.dins.astype(int)
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
rows=[]
for cell in sorted(m.cell_id.unique()):
    c=f"{R2}/{cell}"
    for dt in sorted(os.listdir(c+"/dates")):
        q=json.load(open(f"{c}/dates/{dt}/quality_metrics.json")); f=f"{c}/dates/{dt}/stage2b_{dt}.jsonl"
        if not os.path.exists(f) or not q.get("tile_quality_ok",True): continue
        for line in open(f):
            r=json.loads(line); rows.append((cell,r["bldg_uid"],dt,r["y_pred_ensemble"]))
S2=pd.DataFrame(rows,columns=["cell_id","bldg_uid","date","pred"]).merge(m[["cell_id","bldg_uid","gt_index","dins","incident","m2b"]],on=["cell_id","bldg_uid"])
# keep only (building, date) pairs where the footprint has valid post-fire pixels, as in experiment 1
S2["date"]=S2.date.astype(int); vv=v[["gt_index","date"]].assign(date=v.date.astype(int))
S2=S2.merge(vv,on=["gt_index","date"])
def prf(y,p):
    tp=(y&p).sum(); pr=tp/max(p.sum(),1); rc=tp/max(y.sum(),1); return {"precision":round(pr,3),"recall":round(rc,3),"f1":round(2*pr*rc/max(pr+rc,1e-9),3)}
out={}
for name,g in [("stage2b_first",S2.sort_values("date").groupby("gt_index").first()),("stage2b_last",S2.sort_values("date").groupby("gt_index").last())]:
    out[name]={"all":prf((g.dins==3).values,(g.pred==3).values),"n":int(len(g))}
    for inc,h in g.groupby("incident"): out[name][inc]=prf((h.dins==3).values,(h.pred==3).values)
mm=m[m.m2b>=0].drop_duplicates("gt_index")
out["stage2b_m2b"]={"all":prf((mm.dins==3).values,(mm.m2b==3).values),"n":int(len(mm))}
for inc,h in mm.groupby("incident"): out["stage2b_m2b"][inc]=prf((h.dins==3).values,(h.m2b==3).values)
for name,g in [("first",S2.sort_values("date").groupby("gt_index").first()),("last",S2.sort_values("date").groupby("gt_index").last())]: out["counts_"+name]={"tp":int(((g.dins==3)&(g.pred==3)).sum()),"pred_destroyed":int((g.pred==3).sum()),"dins_destroyed":int((g.dins==3).sum())}
json.dump(out,open("exp_common_stage2b.json","w"),indent=1); print(json.dumps(out,indent=1))
