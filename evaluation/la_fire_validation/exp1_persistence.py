"""Experiment 1: training-free damage from building persistence.
For each pre-fire footprint, coverage = share of its pixels labelled building by SAM 3 on a post-fire
image. Low coverage -> destroyed. Evaluated against DINS (destroyed vs not), threshold chosen on one
fire and tested on the other."""
import csv, glob, json, os, re, sys, numpy as np, pandas as pd, rasterio
from rasterio import features
from shapely.geometry import Polygon
V="/media/data/building_instance_tamu/la_fire_2025/validation"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
B="/media/data/building_instance_tamu/la_fire_2025/postfire_sam3"
CH="/media/data/building_instance_tamu/la_fire_2025/chips"
m=pd.read_csv(V+"/matched_v2.csv"); m=m[m.dins.notna()].copy(); m["dins"]=m.dins.astype(int)
want=set(zip(m.cell_id,m.bldg_uid))
def poly(w):
    v=[float(x) for x in re.findall(r"-?\d+(?:\.\d+)?",w)]; return Polygon(list(zip(v[0::2],v[1::2])))
rows=[]
for cell in sorted(m.cell_id.unique()):
    sb=f"{R2}/{cell}/shared_base/shared_instance_samples.csv"
    if not os.path.exists(sb): continue
    fps=[(r["bldg_uid"],poly(r["polygon_wkt_xy_pre"])) for r in csv.DictReader(open(sb)) if (cell,r["bldg_uid"]) in want]
    if not fps: continue
    for post in sorted(glob.glob(f"{CH}/{cell}/post/*.tif")):
        dt=re.search(r"(\d{8})\.tif$",post).group(1); stem=f"{cell}__{dt}"
        pj=[p for p in glob.glob(f"{B}/part*/predictions/{stem}_prediction.json")]
        if not pj: continue                      # not processed (yet)
        q=f"{R2}/{cell}/dates/{dt}/quality_metrics.json"
        ok=json.load(open(q)).get("tile_quality_ok",True) if os.path.exists(q) else True
        mk=glob.glob(f"{B}/part*/masks/{stem}.tif")
        with rasterio.open(post) as s: H,W=s.height,s.width; img=s.read(1)
        lab=rasterio.open(mk[0]).read(1) if mk else np.zeros((H,W),np.int32)
        for uid,g in fps:
            x0,y0,x1,y1=[int(v) for v in g.bounds]; x0=max(x0,0); y0=max(y0,0); x1=min(x1+1,W); y1=min(y1+1,H)
            if x1<=x0 or y1<=y0: continue
            fp=features.rasterize([(g,1)],out_shape=(y1-y0,x1-x0),transform=rasterio.Affine(1,0,x0,0,1,y0)).astype(bool)
            if fp.sum()<5: continue
            valid=(img[y0:y1,x0:x1][fp]>0).mean()
            cov=(lab[y0:y1,x0:x1][fp]>0).mean()
            rows.append((cell,uid,dt,ok and valid>=0.5,float(cov)))
P=pd.DataFrame(rows,columns=["cell_id","bldg_uid","date","valid","coverage"]).merge(m[["cell_id","bldg_uid","gt_index","dins","incident"]],on=["cell_id","bldg_uid"])
P.to_csv(V+"/exp1_persistence_per_date.csv",index=False)
v=P[P.valid].sort_values("date")
agg={"first_date":v.groupby("gt_index").first(),"last_date":v.groupby("gt_index").last()}  # one row per DINS building
mx=v.groupby("gt_index").agg(coverage=("coverage","max"),dins=("dins","first"),incident=("incident","first"))
agg["multidate_max"]=mx
def auc(score,y):
    o=np.argsort(score); r=np.empty(len(score)); r[o]=np.arange(1,len(score)+1)
    n1=y.sum(); n0=len(y)-n1; return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
def prf(y,p):
    tp=(y&p).sum(); pr=tp/max(p.sum(),1); rc=tp/max(y.sum(),1); return pr,rc,2*pr*rc/max(pr+rc,1e-9)
res={}
for name,a in agg.items():
    a=a.reset_index(); y=(a.dins==3).values; s=1-a.coverage.values
    out={"n":int(len(a)),"auc_destroyed":round(auc(s,y),3) if auc(s,y) else None}
    taus=np.linspace(0.05,0.95,19)
    for tr,te in [("Eaton","Palisades"),("Palisades","Eaton")]:
        A=a[a.incident==tr]; Bt=a[a.incident==te]
        best=max(taus,key=lambda t: prf((A.dins==3).values,(A.coverage<t).values)[2])
        pr,rc,f1=prf((Bt.dins==3).values,(Bt.coverage<best).values)
        out[f"tune_{tr}_test_{te}"]={"tau":round(float(best),2),"precision":round(pr,3),"recall":round(rc,3),"f1":round(f1,3),"n_test":int(len(Bt))}
    pr,rc,f1=prf(y,a.coverage.values<0.5); out["tau_0.5_all"]={"precision":round(pr,3),"recall":round(rc,3),"f1":round(f1,3)}
    out["median_coverage"]={"dins_destroyed":round(float(a[a.dins==3].coverage.median()),3),"dins_no_damage":round(float(a[a.dins==0].coverage.median()),3)}
    res[name]=out
json.dump(res,open(V+"/exp1_persistence_results.json","w"),indent=1); print(json.dumps(res,indent=1))
