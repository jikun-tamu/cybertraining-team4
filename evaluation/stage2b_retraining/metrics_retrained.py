"""Metrics for retrained Stage 2b models: xView2 test by hazard, LA vs DINS (oracle / SAM 3 footprints)."""
import json, os, numpy as np, pandas as pd
O="/media/data/building_instance_tamu/stage2_training_data"; V="/media/data/building_instance_tamu/la_fire_2025/validation"
MODELS=["flood_run019","A_all","B_fire","C_nofire"]
def load(m,d):
    p=f"{O}/eval/{m}__{d}.jsonl"
    if not os.path.exists(p): return None
    j=pd.DataFrame([json.loads(l) for l in open(p)])[["sample_index","y_pred_ensemble","pmax"]]
    c=pd.read_csv(f"{O}/csv/{d}.csv",dtype=str)
    return c.iloc[j.sample_index.values].reset_index(drop=True).assign(pred=j.y_pred_ensemble.values,pmax=j.pmax.values)
def prf(y,p):
    tp=int((y&p).sum()); pr=tp/max(int(p.sum()),1); rc=tp/max(int(y.sum()),1); return round(2*pr*rc/max(pr+rc,1e-9),3),round(pr,3),round(rc,3)
def macro(y,p): return round(float(np.mean([prf(y==k,p==k)[0] for k in range(4)])),3)
res={}
# xView2 test
for m in MODELS:
    t=load(m,"test_all")
    if t is None: continue
    y=t.damage_class.astype(int).values; p=t.pred.values; r={"all":{"macro_f1":macro(y,p),"destroyed_f1":prf(y==3,p==3)[0]}}
    for hz,g in t.groupby("hazard_type"):
        yy=g.damage_class.astype(int).values; pp=g.pred.values; r[hz]={"macro_f1":macro(yy,pp),"destroyed_f1":prf(yy==3,pp==3)[0],"n":len(g)}
    res.setdefault(m,{})["xview2_test"]=r
# LA: valid (building, date) pairs and DINS labels from experiments 1/2
OPv=pd.read_csv(V+"/exp2_oracle_persistence_per_date.csv"); OPv=OPv[OPv.valid&OPv.dins.notna()][["gt_index","date","dins","incident"]]
SPv=pd.read_csv(V+"/exp1_persistence_per_date.csv"); SPv=SPv[SPv.valid&SPv.dins.notna()][["cell_id","bldg_uid","gt_index","date"]]
denom=OPv.drop_duplicates("gt_index").set_index("gt_index")[["dins","incident"]]
def la_agg(df):
    df=df.sort_values("date"); g=df.groupby("gt_index")
    return {"last":g.pred.last(),"first":g.pred.first(),"max":g.pred.max()}
for m in MODELS:
    o=load(m,"la_oracle_alldates"); s=load(m,"la_sam3_alldates")
    if o is None or s is None: continue
    o["gt_index"]=o.bldg_uid.str.split("_").str[1].astype(int); o["date"]=o.date.astype(int)
    o=o.merge(OPv[["gt_index","date"]],on=["gt_index","date"])
    s["date"]=s.date.astype(int); s=s.merge(SPv,on=["cell_id","bldg_uid","date"])
    oa=la_agg(o); sa=la_agg(s); r={}
    for how in ["first","last","max"]:
        y=(denom.dins==3); po=oa[how].reindex(denom.index); ps=sa[how].reindex(denom.index)
        det=ps.notna()
        r[how]={"oracle":prf(y.values,(po==3).fillna(False).values),
                "sam3_detected_only":prf(y[det].values,(ps[det]==3).values),
                "sam3_end_to_end":prf(y.values,(ps==3).fillna(False).values),
                "oracle_any_damage":prf((denom.dins>=1).values,(po>=1).fillna(False).values),
                "oracle_by_fire":{inc:prf(y[k].values,(po[k]==3).fillna(False).values)[0] for inc,k in [(i,denom.incident==i) for i in ["Eaton","Palisades"]]}}
    res.setdefault(m,{})["la"]=r
json.dump(res,open(f"{O}/eval/metrics_retrained.json","w"),indent=1)
print("F1 tuples are (f1, precision, recall)")
for m,r in res.items():
    x=r.get("xview2_test",{}); print(f"\n== {m}")
    if x: print("  xView2 test: all",x["all"]," fire",x.get("fire")," flooding",x.get("flooding"))
    if "la" in r:
        for how in ["first","last","max"]: print(f"  LA {how}: oracle {r['la'][how]['oracle']}  sam3 e2e {r['la'][how]['sam3_end_to_end']}  by fire {r['la'][how]['oracle_by_fire']}")
