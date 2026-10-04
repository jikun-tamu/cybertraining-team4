"""Concatenate per-cell, per-date Stage 2b input CSVs for the LA run (SAM 3 and oracle footprints)."""
import glob, os, pandas as pd
B="/media/data/building_instance_tamu/la_fire_2025/stage2_damage"
OUT="/media/data/building_instance_tamu/stage2_training_data/csv"
cells=[l.strip() for l in open("/media/data/building_instance_tamu/la_fire_2025/validation/postfire_cells.txt") if l.strip()]
for name,run in [("la_sam3","multidate_full_run_v2"),("la_oracle","multidate_oracle")]:
    parts=[]
    for cell in cells:
        for f in sorted(glob.glob(f"{B}/{run}/{cell}/dates/*/shared_for_date.csv")):
            d=pd.read_csv(f,dtype=str); d["cell_id"]=cell; d["date"]=f.split("/")[-2]; parts.append(d)
    df=pd.concat(parts,ignore_index=True); df["damage_class"]=""
    df.to_csv(f"{OUT}/{name}_alldates.csv",index=False); print(name,len(df),"rows,",df.cell_id.nunique(),"cells")
