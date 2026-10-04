"""Write xView2 (tier1) ground-truth building polygons as stage1-format prediction JSONs, so the
inference crop generator (generate_shared_instance_subimages.py) produces the training crops.
Polygons come from the pre-disaster label (pre_then_post anchoring), damage from the post label.
Also writes labels.csv: bldg_uid, tile_id, event_id, hazard_type, damage_class."""
import csv, glob, json, os, sys
from shapely import wkt
D="/media/data/building_instance_tamu"; OUT=D+"/stage2_training_data"
CLS={"no-damage":0,"minor-damage":1,"major-damage":2,"destroyed":3}
for split in ["train","test"]:
    od=f"{OUT}/{split}/stage1_labels"; os.makedirs(od,exist_ok=True)
    rows=[]; skipped=0
    for post_f in sorted(glob.glob(f"{D}/{split}/labels/*_post_disaster.json")):
        base=os.path.basename(post_f)[:-len("_post_disaster.json")]
        pre_f=f"{D}/{split}/labels/{base}_pre_disaster.json"
        post=json.load(open(post_f)); pre=json.load(open(pre_f)); md=post["metadata"]
        dmg={f["properties"]["uid"]:f["properties"].get("subtype") for f in post["features"]["xy"]}
        insts=[]
        for f in pre["features"]["xy"]:
            uid=f["properties"]["uid"]; c=CLS.get(dmg.get(uid))
            if c is None: skipped+=1; continue
            g=wkt.loads(f["wkt"])
            if g.is_empty or g.area<4: skipped+=1; continue
            insts.append({"id":len(insts)+1,"uid":uid,"polygon":[[round(x,2),round(y,2)] for x,y in g.exterior.coords],"confidence":1.0})
            rows.append((uid,f"{base}",md["disaster"],md["disaster_type"],c))
        doc={"image":{"path":f"{D}/{split}/images/{base}_pre_disaster.png","stem":f"{base}_pre_disaster",
                      "width":md.get("width",1024),"height":md.get("height",1024),"disaster_type":"pre"},
             "instances":insts,"summary":{"num_instances":len(insts),"status":"ok","source":"xView2 ground truth"}}
        json.dump(doc,open(f"{od}/{base}_pre_disaster_prediction.json","w"))
    with open(f"{OUT}/{split}/labels.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["bldg_uid","tile_id","event_id","hazard_type","damage_class"]); w.writerows(rows)
    print(split, len(rows), "buildings,", skipped, "skipped (un-classified or degenerate)")
