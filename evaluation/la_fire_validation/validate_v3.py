"""Validate the v3 LA product (model A) against DINS using the v2 footprint matches."""
import json, numpy as np, pandas as pd, geopandas as gpd
V="/media/data/building_instance_tamu/la_fire_2025/validation"
B="/media/data/building_instance_tamu/la_fire_2025/stage2_damage"
m=pd.read_csv(V+"/matched_v2.csv"); m=m[m.dins.notna()].copy(); m["dins"]=m.dins.astype(int)
out={}
for run in ["multidate_full_run_v2","multidate_full_run_v3"]:
    p=pd.read_csv(f"{B}/{run}/building_damage_all_cells.csv")
    cols=[c for c in ["m1_damage_class","m1b_damage_class","m2_majority_class","m2b_damage_class","m3_quality_filtered_class"] if c in p]
    j=m.merge(p[["cell_id","bldg_uid"]+cols],on=["cell_id","bldg_uid"])
    r={"pairs":int(len(j)),"class_counts_all_buildings":{int(k):int(v) for k,v in p.m2b_damage_class.value_counts().sort_index().items()}}
    for c in cols:
        g=j[j[c]>=0]; y=(g.dins==3).values; pr_=(g[c]==3).values; tp=int((y&pr_).sum())
        P=tp/max(pr_.sum(),1); R=tp/max(y.sum(),1)
        anyy=(g.dins>=1).values; anyp=(g[c]>=1).values; tpa=int((anyy&anyp).sum())
        r[c]={"n":int(len(g)),"destroyed_P":round(P,3),"destroyed_R":round(R,3),"destroyed_F1":round(2*P*R/max(P+R,1e-9),3),
              "any_damage_F1":round(2*tpa/max(anyy.sum()+anyp.sum(),1),3),
              "by_fire_destroyed_F1":{inc:round(2*int(((h.dins==3)&(h[c]==3)).sum())/max(int((h.dins==3).sum()+(h[c]==3).sum()),1),3) for inc,h in g.groupby("incident")}}
    out[run]=r
json.dump(out,open(V+"/v3_product_validation.json","w"),indent=1); print(json.dumps(out,indent=1))
