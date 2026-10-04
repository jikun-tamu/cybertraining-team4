"""Experiment 2: write LARIAC footprints (shifted onto the Maxar grid) as stage1-style prediction JSONs,
one per cell, so the LA pipeline can run with oracle footprints."""
import glob, json, os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely.affinity import translate
from shapely.geometry import box
V="/media/data/building_instance_tamu/la_fire_2025/validation"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
O="/media/data/building_instance_tamu/la_fire_2025/oracle_footprints"
gt=gpd.read_file(V+"/lariac_dins_in_region.gpkg"); gt["gt_index"]=range(len(gt))
m=pd.read_csv(V+"/matched_v2.csv")
shift=m.groupby("cell_id")[["shift_dx_m","shift_dy_m"]].first()
gx,gy=m.shift_dx_m.median(),m.shift_dy_m.median()
cells=[l.strip() for l in open(V+"/postfire_cells.txt") if l.strip()]
rep=gt.geometry.representative_point(); n_total=0; assigned=set()
for cell in cells:
    pj=glob.glob(f"{R2}/{cell}/stage1/predictions/*_prediction.json")[0]
    doc=json.load(open(pj)); img=doc["image"]
    with rasterio.open(img["path"]) as s: T=s.transform; W,H=s.width,s.height; bb=box(*s.bounds)
    dx,dy=(shift.loc[cell].values if cell in shift.index else (gx,gy))
    sel=gt[rep.within(bb).values & ~gt.gt_index.isin(assigned)]
    inv=~T; insts=[]
    for gi,geom in zip(sel.gt_index,sel.geometry):
        g=translate(geom,dx,dy)
        if g.geom_type=="MultiPolygon": g=max(g.geoms,key=lambda q:q.area)
        px=[inv*(x,y) for x,y in g.exterior.coords]
        xs=[p[0] for p in px]; ys=[p[1] for p in px]
        if max(xs)<0 or max(ys)<0 or min(xs)>W or min(ys)>H: continue
        insts.append({"id":len(insts)+1,"uid":f"lariac_{gi}","bbox_xyxy":[round(min(xs)),round(min(ys)),round(max(xs)),round(max(ys))],
                      "polygon":[[round(x,2),round(y,2)] for x,y in px],"area_px":round(g.area/(T.a**2),2),"confidence":1.0})
        assigned.add(gi)
    out=dict(doc); out["instances"]=insts; out["summary"]={"num_instances":len(insts),"status":"ok","source":"LARIAC oracle, shifted"}
    d=f"{O}/la_fire_{cell}/stage1/labels"; os.makedirs(d,exist_ok=True)
    json.dump(out,open(f"{d}/{os.path.basename(pj)}","w")); n_total+=len(insts)
print(len(cells),"cells,",n_total,"oracle footprints of",len(gt),"LARIAC buildings in region")
