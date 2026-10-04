"""Validate LA fire 2025 Stage 1 footprints and M2b damage against LARIAC footprints + CAL FIRE DINS."""
import glob, json, os, subprocess, collections
import numpy as np, geopandas as gpd, pandas as pd, rasterio
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union
V="/media/data/building_instance_tamu/la_fire_2025/validation"; RAW=V+"/raw"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
REPO="/media/gisense/xihan/archive/250812_tamu_cybertraining_team4"
UTM=32611
# ---- ground truth
gt=pd.concat([gpd.read_file(f) for f in sorted(glob.glob(RAW+"/lariac_dins_*.geojson"))],ignore_index=True)
gt=gpd.GeoDataFrame(gt,geometry="geometry",crs=4326).to_crs(UTM)
gt=gt[gt.Incident_Name.isin(["Eaton","Palisades"])].copy()
gt["geometry"]=gt.geometry.buffer(0)
per=gpd.read_file(RAW+"/perimeters_2025.geojson").to_crs(UTM)
per=per[per.poly_GISAcres>1000]
per_buf=unary_union(per.geometry.buffer(100))
# ---- valid imagery footprint of the 295 pre images (1/8 resolution)
valid=[]
for c in sorted(glob.glob(R2+"/cell_*")):
    p=glob.glob(c+"/stage1/predictions/*.json")
    if not p: continue
    img=json.load(open(p[0]))["image"]["path"]
    with rasterio.open(img) as s:
        a=s.read(out_shape=(s.count,s.height//8,s.width//8)); t=s.transform*s.transform.scale(8,8)
        m=(a.max(0)>0).astype("uint8")
        valid+= [shape(g) for g,v in features.shapes(m,mask=m.astype(bool),transform=t) if v==1]
        crs=s.crs
valid=gpd.GeoSeries(valid,crs=crs).to_crs(UTM)
region=unary_union(valid.buffer(0)).intersection(per_buf)
print(f"evaluation region: {region.area/1e6:.1f} km2")
def inside(g): return g[g.geometry.representative_point().within(region)].copy()
gt_r=inside(gt); print("GT buildings in region:",len(gt_r))
# ---- predictions: v2 and v1 (from git history)
v2=gpd.read_file(R2+"/building_damage_all_cells.gpkg").to_crs(UTM)
v1p="/tmp/xihan_v1.gpkg"
subprocess.run(f"cd {REPO} && git show 0dc7630:results/final_product/building_damage_all_cells.gpkg > {v1p}",shell=True,check=True)
v1=gpd.read_file(v1p).to_crs(UTM); os.remove(v1p)
DMAP={"No Damage":0,"Affected (1-9%)":1,"Minor (10-25%)":1,"Major (26-50%)":2,"Destroyed (>50%)":3}
gt_r["dins"]=gt_r.DINS_MaximumDamage.map(DMAP)
gt_r["area"]=gt_r.area
def evaluate(pred, tag):
    pred=inside(pred); pred=pred[pred.is_valid | True].copy(); pred["geometry"]=pred.geometry.buffer(0)
    pg=pred.geometry.values; gg=gt_r.geometry.values
    j=gpd.sjoin(gpd.GeoDataFrame({"pi":range(len(pred))},geometry=pg,crs=UTM),
                gpd.GeoDataFrame({"gi":range(len(gt_r))},geometry=gg,crs=UTM),predicate="intersects")
    inter=np.array([pg[a].intersection(gg[b]).area for a,b in zip(j.pi,j.gi)])
    iou=inter/np.array([pg[a].area+gg[b].area for a,b in zip(j.pi,j.gi)]-inter)
    j=j.assign(iou=iou).sort_values("iou",ascending=False)
    mp,mg,pairs=set(),set(),[]
    for a,b,s in zip(j.pi,j.gi,j.iou):
        if s<0.5: break
        if a in mp or b in mg: continue
        mp.add(a); mg.add(b); pairs.append((a,b))
    tp=len(pairs); P=tp/len(pred); Rc=tp/len(gt_r)
    # loose: GT representative point inside any prediction
    pts=gpd.GeoDataFrame(geometry=gt_r.geometry.representative_point().values,crs=UTM)
    hit=gpd.sjoin(pts,gpd.GeoDataFrame({"pi":range(len(pred))},geometry=pg,crs=UTM),predicate="within")
    covered=set(hit.index); loose_r=len(covered)/len(gt_r)
    pred_hits=gpd.sjoin(gpd.GeoDataFrame({"pi":range(len(pred))},geometry=pg,crs=UTM),
                        gpd.GeoDataFrame(geometry=gg,crs=UTM),predicate="intersects")
    loose_p=pred_hits.pi.nunique()/len(pred)
    out={"run":tag,"pred":len(pred),"gt":len(gt_r),"tp_iou50":tp,"precision_iou50":round(P,3),"recall_iou50":round(Rc,3),
         "f1_iou50":round(2*P*Rc/(P+Rc),3),"recall_loose_centroid":round(loose_r,3),"precision_loose_overlap":round(loose_p,3),
         "median_iou_matched":round(float(np.median([s for s in j.iou if s>=0.5])),3) if tp else None}
    # recall by DINS class and size (loose)
    cov=np.zeros(len(gt_r),bool); cov[list(covered)]=True
    gt_r[f"cov_{tag}"]=cov
    out["recall_by_dins"]={k:round(float(cov[(gt_r.DINS_MaximumDamage==k).values].mean()),3) for k in ["Destroyed (>50%)","No Damage","Affected (1-9%)","Minor (10-25%)","Major (26-50%)",""]}
    bins=[0,50,100,200,400,1e9]; lab=["<50","50-100","100-200","200-400",">400"]
    sz=pd.cut(gt_r.area,bins,labels=lab)
    out["recall_by_area_m2"]={l:round(float(cov[(sz==l).values].mean()),3) for l in lab}
    out["gt_by_area_m2"]={l:int((sz==l).sum()) for l in lab}
    # damage agreement for loose matches: GT point inside prediction
    if "m2b_damage_class" in pred:
        h=hit.drop_duplicates(subset=None).copy(); h["gi"]=h.index
        h=h.drop_duplicates("gi")
        h["pred_cls"]=pred.m2b_damage_class.values[h.pi]; h["dins"]=gt_r.dins.values[h.gi]
        h=h[h.dins.notna() & (h.pred_cls>=0)]
        cm=pd.crosstab(h.dins.astype(int),h.pred_cls.astype(int)).reindex(index=[0,1,2,3],columns=[0,1,2,3],fill_value=0)
        out["confusion_dins_rows_pred_cols"]=cm.values.tolist()
        d=(h.dins==3); pr=(h.pred_cls==3)
        tpd=int((d&pr).sum()); out["destroyed"]={"dins_destroyed":int(d.sum()),"pred_destroyed":int(pr.sum()),"tp":tpd,
            "recall":round(tpd/max(d.sum(),1),3),"precision":round(tpd/max(pr.sum(),1),3)}
        dmg=(h.dins>=1); prd=(h.pred_cls>=1)
        out["any_damage_binary"]={"dins_damaged":int(dmg.sum()),"recall":round(float((dmg&prd).sum()/max(dmg.sum(),1)),3),
                                  "precision":round(float((dmg&prd).sum()/max(prd.sum(),1)),3)}
        out["n_damage_pairs"]=int(len(h))
    return out
def coregister(pred):
    """Per-cell translation that aligns Maxar-derived footprints to LARIAC (median centroid offset
    of loosely matched pairs, cells with >= 30 pairs; others use the global median)."""
    from shapely.affinity import translate
    p=inside(pred).reset_index(drop=True)
    pts=gpd.GeoDataFrame(geometry=gt_r.geometry.representative_point().values,crs=UTM)
    h=gpd.sjoin(pts,gpd.GeoDataFrame({"pi":range(len(p))},geometry=p.geometry.values,crs=UTM),predicate="within")
    h=h[~h.index.duplicated()]
    pc=p.geometry.centroid.values[h.pi.values]; gc=gt_r.geometry.centroid.values[h.index.values]
    d=pd.DataFrame({"cell":p.cell_id.values[h.pi.values],"dx":[a.x-b.x for a,b in zip(pc,gc)],"dy":[a.y-b.y for a,b in zip(pc,gc)]})
    g=d.groupby("cell").agg(n=("dx","size"),dx=("dx","median"),dy=("dy","median"))
    gx,gy=d.dx.median(),d.dy.median()
    sh={c:((r.dx,r.dy) if r.n>=30 else (gx,gy)) for c,r in g.iterrows()}
    p["geometry"]=[translate(geom,-sh.get(c,(gx,gy))[0],-sh.get(c,(gx,gy))[1]) for geom,c in zip(p.geometry,p.cell_id)]
    return p, {"global_dx_m":round(gx,2),"global_dy_m":round(gy,2),"cells_with_own_shift":int((g.n>=30).sum())}
res=[evaluate(v1,"v1_april"),evaluate(v2,"v2_october")]
for pred,tag in [(v1,"v1_april_coregistered"),(v2,"v2_october_coregistered")]:
    pc,info=coregister(pred); r=evaluate(pc,tag); r["coregistration"]=info; res.append(r)
res.append({"region_km2":round(region.area/1e6,2),"gt_dins_counts_in_region":gt_r.DINS_MaximumDamage.value_counts().to_dict()})
json.dump(res,open(V+"/validation_results.json","w"),indent=1)
gpd.GeoDataFrame({"geometry":[region]},crs=UTM).to_file(V+"/evaluation_region.gpkg")
gt_r.drop(columns=[c for c in gt_r.columns if c.startswith("cov_")]).to_file(V+"/lariac_dins_in_region.gpkg")
print(json.dumps(res,indent=1))
