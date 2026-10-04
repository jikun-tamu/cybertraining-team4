"""Experiment 2 (Stage 2b): flood-trained Stage 2b on oracle LARIAC footprints vs DINS."""
import csv, glob, json, os, numpy as np, pandas as pd
V="/media/data/building_instance_tamu/la_fire_2025/validation"
O="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_oracle"
L=open(O+"/oracle_run.log").read()
fails=L.count("FAILED")
rows=[]
for c in sorted(glob.glob(O+"/cell_*")):
    for f in glob.glob(c+"/dates/*/stage2b_*.jsonl"):
        dt=os.path.basename(f)[8:16]
        for line in open(f):
            r=json.loads(line); rows.append((int(r["bldg_uid"].split("_")[1]),int(dt),r["y_pred_ensemble"],r["pmax"]))
S2=pd.DataFrame(rows,columns=["gt_index","date","pred","pmax"])
OP=pd.read_csv(V+"/exp2_oracle_persistence_per_date.csv")
v=OP[OP.valid & OP.dins.notna()][["gt_index","date","dins","incident"]]
S2=S2.merge(v,on=["gt_index","date"]).sort_values("date")
agg=[]
for c in sorted(glob.glob(O+"/cell_*/aggregated_predictions.csv")):
    for r in csv.DictReader(open(c)): agg.append((int(r["bldg_uid"].split("_")[1]),int(r["m2b_coverage_vote_class"])))
A=pd.DataFrame(agg,columns=["gt_index","m2b"]).merge(v.drop_duplicates("gt_index")[["gt_index","dins","incident"]],on="gt_index")
def prf(y,p):
    tp=int((y&p).sum()); pr=tp/max(int(p.sum()),1); rc=tp/max(int(y.sum()),1)
    return {"precision":round(pr,3),"recall":round(rc,3),"f1":round(2*pr*rc/max(pr+rc,1e-9),3),"tp":tp,"pred_pos":int(p.sum()),"n_pos":int(y.sum())}
res={"failed_steps":fails}
for name,g in [("first",S2.groupby("gt_index").first()),("last",S2.groupby("gt_index").last())]:
    res[name]={"n":int(len(g)),"destroyed":prf((g.dins==3).values,(g.pred==3).values),"any_damage":prf((g.dins>=1).values,(g.pred>=1).values),
               "pred_class_share":{int(k):round(float(x),3) for k,x in g.pred.value_counts(normalize=True).sort_index().items()}}
A=A[A.m2b>=0]
res["m2b"]={"n":int(len(A)),"destroyed":prf((A.dins==3).values,(A.m2b==3).values),"any_damage":prf((A.dins>=1).values,(A.m2b>=1).values)}
json.dump(res,open(V+"/exp2_stage2b_results.json","w"),indent=1); print(json.dumps(res,indent=1))
