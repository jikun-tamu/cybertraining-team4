"""Experiment 2 (persistence): oracle LARIAC footprints vs SAM 3 footprints, same buildings and dates."""
import glob, json, re, os, numpy as np, pandas as pd, rasterio
from rasterio import features
from shapely.geometry import Polygon
V="/media/data/building_instance_tamu/la_fire_2025/validation"
O="/media/data/building_instance_tamu/la_fire_2025/oracle_footprints"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
B="/media/data/building_instance_tamu/la_fire_2025/postfire_sam3"; CH="/media/data/building_instance_tamu/la_fire_2025/chips"
import geopandas as gpd
gt=gpd.read_file(V+"/lariac_dins_in_region.gpkg"); gt["gt_index"]=range(len(gt))
lab=pd.DataFrame({"gt_index":gt.gt_index,"dins":gt.dins,"incident":gt.Incident_Name,"area":gt.area})
rows=[]
for d in sorted(glob.glob(O+"/la_fire_cell_*")):
    cell=os.path.basename(d)[len("la_fire_"):]; doc=json.load(open(glob.glob(d+"/stage1/labels/*.json")[0]))
    fps=[(int(i["uid"].split("_")[1]),Polygon(i["polygon"])) for i in doc["instances"]]
    for post in sorted(glob.glob(f"{CH}/{cell}/post/*.tif")):
        dt=re.search(r"(\d{8})\.tif$",post).group(1); stem=f"{cell}__{dt}"
        q=f"{R2}/{cell}/dates/{dt}/quality_metrics.json"
        ok=json.load(open(q)).get("tile_quality_ok",True) if os.path.exists(q) else True
        mk=glob.glob(f"{B}/part*/masks/{stem}.tif")
        with rasterio.open(post) as s: H,W=s.height,s.width; img=s.read(1)
        L=rasterio.open(mk[0]).read(1) if mk else np.zeros((H,W),np.int32)
        for gi,g in fps:
            x0,y0,x1,y1=[int(v) for v in g.bounds]; x0=max(x0,0); y0=max(y0,0); x1=min(x1+1,W); y1=min(y1+1,H)
            if x1<=x0 or y1<=y0: continue
            fp=features.rasterize([(g,1)],out_shape=(y1-y0,x1-x0),transform=rasterio.Affine(1,0,x0,0,1,y0)).astype(bool)
            if fp.sum()<5: continue
            rows.append((gi,dt,ok and (img[y0:y1,x0:x1][fp]>0).mean()>=0.5,float((L[y0:y1,x0:x1][fp]>0).mean())))
OP=pd.DataFrame(rows,columns=["gt_index","date","valid","coverage"]).merge(lab,on="gt_index")
OP.to_csv(V+"/exp2_oracle_persistence_per_date.csv",index=False)
SP=pd.read_csv(V+"/exp1_persistence_per_date.csv")   # SAM 3 footprints, keyed by gt_index
def agg(P,how):
    v=P[P.valid & P.dins.notna()].sort_values("date")
    if how=="first": a=v.groupby("gt_index").first()
    elif how=="last": a=v.groupby("gt_index").last()
    else: a=v.groupby("gt_index").agg(coverage=("coverage","max"),dins=("dins","first"),incident=("incident","first"))
    return a[["coverage","dins","incident"]]
def prf(y,p):
    tp=int((y&p).sum()); pr=tp/max(int(p.sum()),1); rc=tp/max(int(y.sum()),1)
    return {"precision":round(pr,3),"recall":round(rc,3),"f1":round(2*pr*rc/max(pr+rc,1e-9),3),"tp":tp,"n_destroyed":int(y.sum())}
res={}
for how in ["first","last","max"]:
    o=agg(OP,how)                     # every DINS building with a valid post date: the common denominator
    s=agg(SP,how).reindex(o.index)    # SAM footprints: NaN where Stage 1 missed the building
    y=(o.dins==3).values
    r={"n_buildings":int(len(o)),"stage1_detected_share":round(float(s.coverage.notna().mean()),3),
       "oracle_footprints":prf(y,(o.coverage<0.5).values),
       "sam3_footprints_end_to_end":prf(y,(s.coverage<0.5).fillna(False).values),
       "sam3_footprints_detected_only":prf(y[s.coverage.notna().values],(s.coverage<0.5).values[s.coverage.notna().values])}
    for inc in ["Eaton","Palisades"]:
        k=(o.incident==inc).values
        r[inc]={"oracle":prf(y[k],(o.coverage<0.5).values[k]),"sam3_end_to_end":prf(y[k],(s.coverage<0.5).fillna(False).values[k])}
    # recall of destroyed buildings by Stage 1 (were they detected at all?)
    r["stage1_detected_share_destroyed"]=round(float(s.coverage.notna().values[y].mean()),3)
    r["stage1_detected_share_not_destroyed"]=round(float(s.coverage.notna().values[~y].mean()),3)
    res[how]=r
json.dump(res,open(V+"/exp2_persistence_results.json","w"),indent=1); print(json.dumps(res,indent=1))
