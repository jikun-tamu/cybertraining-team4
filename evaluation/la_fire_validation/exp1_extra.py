import csv, glob, json, re, os, numpy as np, pandas as pd, rasterio
from PIL import Image, ImageDraw, ImageFont
V="/media/data/building_instance_tamu/la_fire_2025/validation"
R2="/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v2"
B="/media/data/building_instance_tamu/la_fire_2025/postfire_sam3"; CH="/media/data/building_instance_tamu/la_fire_2025/chips"
P=pd.read_csv(V+"/exp1_persistence_per_date.csv"); m=pd.read_csv(V+"/matched_v2.csv")
v=P[P.valid].sort_values("date")
first=v.groupby("gt_index").first().reset_index()
mx=v.groupby("gt_index").agg(coverage=("coverage","max"),dins=("dins","first"),incident=("incident","first")).reset_index()
def prf(y,p):
    tp=(y&p).sum(); pr=tp/max(p.sum(),1); rc=tp/max(y.sum(),1); return round(pr,3),round(rc,3),round(2*pr*rc/max(pr+rc,1e-9),3)
out={}
for name,a in [("first_date",first),("multidate_max",mx)]:
    out[name]={inc:dict(zip(["precision","recall","f1"],prf((g.dins==3).values,(g.coverage<0.5).values)))|{"n":int(len(g)),"n_destroyed":int((g.dins==3).sum())} for inc,g in a.groupby("incident")}
    out[name]["by_dins_class_share_coverage_below_0.5"]={int(k):round(float((g.coverage<0.5).mean()),3) for k,g in a.groupby("dins")}
# matched DINS buildings with no valid date at all
allm=m[m.dins.notna()]; have=set(v.gt_index)
out["dins_pairs_without_valid_post_date"]=int(sum(g not in have for g in allm.gt_index))
json.dump(out,open(V+"/exp1_extra.json","w"),indent=1); print(json.dumps(out,indent=1))
# ---- example figure: TP / FP / FN on first valid date
F=ImageFont.load_default(size=19); S=220
def poly(w):
    vv=[float(x) for x in re.findall(r"-?\d+(?:\.\d+)?",w)]; return list(zip(vv[0::2],vv[1::2]))
sb_cache={}
def footprint(cell,uid):
    if cell not in sb_cache:
        sb_cache[cell]={r["bldg_uid"]:poly(r["polygon_wkt_xy_pre"]) for r in csv.DictReader(open(f"{R2}/{cell}/shared_base/shared_instance_samples.csv"))}
    return sb_cache[cell][uid]
def rgb(p):
    with rasterio.open(p) as s: return np.moveaxis(s.read([1,2,3]),0,-1)
def panel(r,kind):
    pts=footprint(r.cell_id,r.bldg_uid); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; h=110
    pre=rgb(glob.glob(f"{CH}/{r.cell_id}/pre/*.tif")[0]); post=rgb(glob.glob(f"{CH}/{r.cell_id}/post/*{r.date}.tif")[0])
    H,W=pre.shape[:2]; x0=int(min(max(cx-h,0),W-2*h)); y0=int(min(max(cy-h,0),H-2*h))
    stem=f"{r.cell_id}__{r.date}"; mk=glob.glob(f"{B}/part*/masks/{stem}.tif")
    lab=rasterio.open(mk[0]).read(1)[y0:y0+2*h,x0:x0+2*h] if mk else np.zeros((2*h,2*h),int)
    ims=[]
    for arr,det in [(pre,False),(post,True)]:
        im=Image.fromarray(arr[y0:y0+2*h,x0:x0+2*h]).convert("RGBA"); lay=Image.new("RGBA",im.size,(0,0,0,0)); d=ImageDraw.Draw(lay)
        if det:
            ov=np.zeros((2*h,2*h,4),np.uint8); ov[lab>0]=(255,0,255,90); lay=Image.alpha_composite(lay,Image.fromarray(ov))
            d=ImageDraw.Draw(lay)
        d.polygon([(x-x0,y-y0) for x,y in pts],outline=(0,230,255,255),width=3)
        ims.append(Image.alpha_composite(im,lay).convert("RGB").resize((S,S)))
    t=Image.new("RGB",(2*S+4,S+52),(255,255,255)); t.paste(ims[0],(0,52)); t.paste(ims[1],(S+4,52))
    d=ImageDraw.Draw(t); d.text((4,4),f"{kind}: DINS {r.dins_label}",fill=(0,0,0),font=F); d.text((4,27),f"coverage {r.coverage:.2f}  ({r.date})",fill=(80,80,80),font=F)
    return t
fd=first.merge(m[["gt_index","dins_label"]],on="gt_index")
rs=np.random.RandomState(11)
sel=[("Correct, destroyed",fd[(fd.dins==3)&(fd.coverage<0.5)]),("Correct, intact",fd[(fd.dins==0)&(fd.coverage>=0.5)]),
     ("False alarm",fd[(fd.dins==0)&(fd.coverage<0.5)]),("Missed",fd[(fd.dins==3)&(fd.coverage>=0.5)])]
rowsimg=[]
for kind,g in sel:
    pick=g.iloc[rs.choice(len(g),2,replace=False)]
    rowsimg.append([panel(r,kind) for r in pick.itertuples()])
w,hh=rowsimg[0][0].size; G=Image.new("RGB",(2*w+20,4*hh+30),(255,255,255))
for i,row in enumerate(rowsimg):
    for j,t in enumerate(row): G.paste(t,(j*(w+20),i*(hh+10)))
G.save("/tmp/xihan_persist.jpg",quality=86); print("fig ok", {k:len(g) for k,g in sel})
