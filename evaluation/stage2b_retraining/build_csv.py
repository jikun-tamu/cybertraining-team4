"""Join generated crops with xView2 damage labels and write training/test CSVs for train_stage2.py."""
import os, pandas as pd
OUT="/media/data/building_instance_tamu/stage2_training_data"; os.makedirs(OUT+"/csv",exist_ok=True)
for split in ["train","test"]:
    c=pd.read_csv(f"{OUT}/{split}/crops/shared_instance_samples.csv",dtype=str)
    l=pd.read_csv(f"{OUT}/{split}/labels.csv",dtype=str)
    assert l.bldg_uid.is_unique, "uid not unique"
    c=c.drop(columns=["event_id","hazard_type","damage_class"]).merge(l[["bldg_uid","event_id","hazard_type","damage_class"]],on="bldg_uid",how="inner")
    c=c[[x for x in pd.read_csv(f"{OUT}/{split}/crops/shared_instance_samples.csv",nrows=0).columns]]
    print(split,len(c),"rows;",c.groupby("hazard_type").size().to_dict())
    if split=="train":
        c.to_csv(f"{OUT}/csv/train_all.csv",index=False)
        c[c.hazard_type=="fire"].to_csv(f"{OUT}/csv/train_fire.csv",index=False)
        c[c.hazard_type!="fire"].to_csv(f"{OUT}/csv/train_nofire.csv",index=False)
    else:
        c.to_csv(f"{OUT}/csv/test_all.csv",index=False)
for f in sorted(os.listdir(OUT+"/csv")):
    d=pd.read_csv(f"{OUT}/csv/{f}",usecols=["damage_class"]); print(f,len(d),d.damage_class.value_counts().sort_index().to_dict())
