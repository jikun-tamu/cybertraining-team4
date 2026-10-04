"""Experiment 5 (Stage 2b): single post date vs multi-date vote, against DINS. Also confidence under shift."""
import json, os, glob, numpy as np, pandas as pd
V="/media/data/building_instance_tamu/la_fire_2025/validation"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
m=pd.read_csv(V+"/matched_v2.csv"); m=m[m.dins.notna()].copy(); m["dins"]=m.dins.astype(int)
rows=[]
for cell in sorted(m.cell_id.unique()):
    c=f"{R2}/{cell}"
    for dt in sorted(os.listdir(c+"/dates")):
        q=json.load(open(f"{c}/dates/{dt}/quality_metrics.json")); f=f"{c}/dates/{dt}/stage2b_{dt}.jsonl"
        if not os.path.exists(f): continue
        for line in open(f):
            r=json.loads(line); rows.append((cell,r["bldg_uid"],dt,bool(q.get("tile_quality_ok",True)),r["y_pred_ensemble"],r["pmax"],r["entropy"]))
P=pd.DataFrame(rows,columns=["cell_id","bldg_uid","date","ok","pred","pmax","entropy"])
P=P.merge(m[["cell_id","bldg_uid","gt_index","dins","incident"]],on=["cell_id","bldg_uid"])
def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); out={}
    for name,yt,pt in [("destroyed",y==3,p==3),("any_damage",y>=1,p>=1)]:
        tp=(yt&pt).sum(); pr=tp/max(pt.sum(),1); rc=tp/max(yt.sum(),1)
        out[name]={"precision":round(pr,3),"recall":round(rc,3),"f1":round(2*pr*rc/max(pr+rc,1e-9),3)}
    f1s=[]
    for k in range(4):
        tp=((y==k)&(p==k)).sum(); pr=tp/max((p==k).sum(),1); rc=tp/max((y==k).sum(),1); f1s.append(2*pr*rc/max(pr+rc,1e-9))
    out["macro_f1_4class"]=round(float(np.mean(f1s)),3); out["accuracy"]=round(float((y==p).mean()),3); out["n"]=int(len(y))
    return out
ok=P[P.ok]
first=ok.sort_values("date").groupby("gt_index").first().reset_index()  # one row per DINS building
last=ok.sort_values("date").groupby("gt_index").last().reset_index()
res={"stage2b_first_valid_date":metrics(first.dins,first.pred),"stage2b_last_valid_date":metrics(last.dins,last.pred),
     "stage2b_m2b_multidate":metrics(m[m.m2b>=0].dins,m[m.m2b>=0].m2b)}
res["per_date"]={dt:metrics(g.dins,g.pred) for dt,g in ok.groupby("date") if len(g)>=200}
# confidence under shift: DINS-destroyed buildings (all predicted not destroyed) vs DINS no-damage
lastc=last.copy()
res["confidence_last_date"]={lab:{"mean_pmax":round(float(g.pmax.mean()),3),"median_pmax":round(float(g.pmax.median()),3),
      "share_pmax_over_0.8":round(float((g.pmax>0.8).mean()),3),"mean_entropy":round(float(g.entropy.mean()),3),"n":int(len(g))}
      for lab,g in [("dins_destroyed",lastc[lastc.dins==3]),("dins_no_damage",lastc[lastc.dins==0])]}
json.dump(res,open(V+"/exp5_stage2b_results.json","w"),indent=1)
print(json.dumps({k:v for k,v in res.items() if k!="per_date"},indent=1))
for dt,v in res["per_date"].items(): print(dt, v["n"], "destroyed recall",v["destroyed"]["recall"],"any-dmg F1",v["any_damage"]["f1"])
