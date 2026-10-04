"""Pair DINS-labelled LARIAC buildings with stage1 v0.2 footprints (per-cell co-registration,
GT representative point inside the shifted prediction). Writes validation/matched_v2.csv."""
import numpy as np, pandas as pd, geopandas as gpd
from shapely.affinity import translate
V="/media/data/building_instance_tamu/la_fire_2025/validation"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
gt=gpd.read_file(V+"/lariac_dins_in_region.gpkg"); reg=gpd.read_file(V+"/evaluation_region.gpkg").geometry[0]
pr=gpd.read_file(R2+"/building_damage_all_cells.gpkg").to_crs(gt.crs)
pr=pr[pr.geometry.representative_point().within(reg)].reset_index(drop=True)
def pair(p):
    pts=gpd.GeoDataFrame({"gi":range(len(gt))},geometry=gt.geometry.representative_point().values,crs=gt.crs)
    h=gpd.sjoin(pts,gpd.GeoDataFrame({"pi":range(len(p))},geometry=p.geometry.values,crs=gt.crs),predicate="within")
    return h.drop_duplicates("gi")
h=pair(pr)
d=pd.DataFrame({"cell":pr.cell_id.values[h.pi],"dx":[a.x-b.x for a,b in zip(pr.geometry.centroid.values[h.pi],gt.geometry.centroid.values[h.gi])],
                "dy":[a.y-b.y for a,b in zip(pr.geometry.centroid.values[h.pi],gt.geometry.centroid.values[h.gi])]})
g=d.groupby("cell").agg(n=("dx","size"),dx=("dx","median"),dy=("dy","median")); gx,gy=d.dx.median(),d.dy.median()
sh={c:((r.dx,r.dy) if r.n>=30 else (gx,gy)) for c,r in g.iterrows()}
ps=pr.copy(); ps["geometry"]=[translate(x,-sh.get(c,(gx,gy))[0],-sh.get(c,(gx,gy))[1]) for x,c in zip(pr.geometry,pr.cell_id)]
h=pair(ps)
out=pd.DataFrame({"gt_index":h.gi.values,"lariac_bld_id":gt.BLD_ID.values[h.gi],"incident":gt.Incident_Name.values[h.gi],
    "dins_label":gt.DINS_MaximumDamage.values[h.gi],"dins":gt.dins.values[h.gi],"gt_area_m2":gt.area.values[h.gi],
    "cell_id":ps.cell_id.values[h.pi],"bldg_uid":ps.bldg_uid.values[h.pi],"m2b":ps.m2b_damage_class.values[h.pi],
    "shift_dx_m":[sh.get(c,(gx,gy))[0] for c in ps.cell_id.values[h.pi]],"shift_dy_m":[sh.get(c,(gx,gy))[1] for c in ps.cell_id.values[h.pi]]})
out.to_csv(V+"/matched_v2.csv",index=False)
print(len(out),"pairs;", out.dins.notna().sum(),"with DINS class;", out.groupby("incident").size().to_dict())
print(out.dins_label.value_counts().to_dict())
