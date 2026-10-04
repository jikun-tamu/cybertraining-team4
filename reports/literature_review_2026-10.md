# Literature Review and Positioning — October 2026

Context: Stage 1 (SAM 3) was rewritten in September 2026 (`reports/sam3_audit_2026-09.md`) and the
LA fire 2025 output was validated against LARIAC footprints with CAL FIRE DINS attributes
(`/media/data/building_instance_tamu/la_fire_2025/validation/`). This review collects related work
to decide how to position a paper.

Verification: titles, authors, years and venues were checked against arXiv, publisher pages or
Crossref by the reviewer. Six key items were re-checked by hand on 2026-10-02 (SAM 3, Remote
SAMsing, BRIGHT 2026 challenge report, Damage-TriageFormer, HASTE, Benson & Ecker 2020). Many
2026 items are unrefereed arXiv preprints. Re-check every citation before it goes into a paper.

## Our validation numbers (for reference)

- Evaluation region: imaged area ∩ (Eaton + Palisades perimeters + 100 m), 83.3 km², 14,369 LARIAC buildings.
- Maxar vs LARIAC offset: median 4.3 m, per 600 m cell -4.7 to +0.7 m (x); corrected per cell.
- Stage 1 v0.2 (co-registered): P 0.606 / R 0.591 / F1 0.598 at IoU 0.5; centroid recall 0.790,
  overlap precision 0.881. Recall ≥ 0.84 for buildings > 100 m², 0.60 for < 50 m². v1 (April): F1 0.563.
- Stage 2b (flood-trained) vs DINS on 9,135 matched buildings: 0 of 4,068 DINS-destroyed predicted
  destroyed; 3,665 predicted no damage; any-damage recall 0.11. Pre/post co-registration is fine
  (median shift < 1 px); per-date post crops use the correct dates.

## A. Annotated bibliography

### 1. Foundation-model / zero-shot building extraction
1. Kirillov et al. 2023, Segment Anything, ICCV 2023. doi:10.1109/ICCV51070.2023.00371.
2. Ravi et al. 2025, SAM 2: Segment Anything in Images and Videos, ICLR 2025.
3. Carion et al. 2026, SAM 3: Segment Anything with Concepts, ICLR 2026. arXiv:2511.16719 (checked). The model we use.
4. Wu & Osco 2023, samgeo, JOSS 8(89):5663. doi:10.21105/joss.05663. Cite as the tool.
5. Osco et al. 2023, The Segment Anything Model for Remote Sensing Applications: From Zero to One Shot, IJAEOG 124. arXiv:2306.16623. Text prompts weak; one-shot helps.
6. de Carvalho et al. 2026, Remote SAMsing: From Segment Anything to Segment Everything, arXiv:2605.00256 (checked). SAM 2 tiling with contextual padding and best-match merge across tiles; tile 1000→250 px raises Det@0.5 from 56% to 85%. Anticipates our tiling finding for SAM 2.
7. Li et al. 2025/26, SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images, arXiv:2512.08730. SAM 3 on 20+ RS datasets (building datasets not verified).
8. Dabaja & Celik 2026, Promptable Concept Segmentation from Above: Evaluating SAM 3's Zero-Shot and One-Shot Capabilities in Remote Sensing, arXiv:2607.09583. Text prompts carry ground-level bias; exemplar prompts do better.
9. Akbulut, Özdemir & Karslı 2025, Comparative Analysis of Vision Foundation Models for Building Segmentation in Aerial Imagery, ISPRS Archives XLVIII-M-6-2025. doi:10.5194/isprs-archives-XLVIII-M-6-2025-23-2025.
10. Yao et al. 2025, RemoteSAM: Towards Segment Anything for Earth Observation, arXiv:2505.18022.

### 2. Damage benchmarks and state of the art
11. Gupta et al. 2019, Creating xBD, CVPR Workshops. arXiv:1911.09296.
12. Weber & Kané 2020, Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion, ICLR AI4Earth. arXiv:2004.05525.
13. Zheng et al. 2021, ChangeOS, Remote Sensing of Environment 265:112636. Object-based localise-then-classify, end to end.
14. Chen et al. 2022, DamFormer, IGARSS. arXiv:2201.10953.
15. Chen et al. 2024, ChangeMamba, IEEE TGRS. doi:10.1109/TGRS.2024.3417253.
16. Chen et al. 2025, BRIGHT, Earth System Science Data 17:6217. arXiv:2501.06019.
17. Chen et al. 2026, Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 BRIGHT Challenge, arXiv:2607.22746 (checked). Winners 0.182 / 0.181 test mAP vs 0.513 in-domain; cross-event generalisation is the main open problem; leading solutions separated building localisation from damage recognition.
18. Wang et al. 2025, DisasterM3, NeurIPS 2025 Datasets & Benchmarks. arXiv:2505.21089.
19. Tehrani et al. 2026, DisasterInsight, arXiv:2601.18493.
20. Zhang & Wang 2024, Good at captioning, bad at counting: Benchmarking GPT-4V on Earth observation data, arXiv:2401.17600.
21. Dietrich et al. 2026, GeBDA: Building Damage Assessment as Text-Based Sequence Prediction, arXiv:2608.28567.
22. Wang, Zhong & He 2026, CogVis, arXiv:2608.06150.

### 3. Cross-hazard and domain generalisation
23. Benson & Ecker 2020, Assessing out-of-domain generalization for robust building damage detection, NeurIPS AI+HADR. arXiv:2011.10328 (checked). IID performance does not predict OOD performance.
24. Gerard, Borne-Pons & Sullivan 2024, A simple, strong baseline for building damage detection on the xBD dataset, arXiv:2401.17271.
25. Ahn et al. 2025, DAVI: Generalizable Disaster Damage Assessment via Change Detection with Vision Foundation Model, AAAI 2025. arXiv:2406.08020. Includes wildfire.
26. Gençoğlu & Ekenel 2026, Improved MambaBDA Framework for Robust Building Damage Assessment Across Disaster Domains, arXiv:2603.01116.
27. Mouradi & Kshirsagar 2026, Robust Building Damage Detection in Cross-Disaster Settings Using Domain Adaptation, arXiv:2603.14694.
28. Li et al. 2026, Smart Transfer, arXiv:2604.02627.
29. Dietrich et al. 2025/26, xBD-S12: The Potential of Copernicus Satellites for Disaster Response, arXiv:2511.05461.

### 4. Wildfire damage mapping and the 2025 LA fires
30. Galanis et al. 2021, DamageMap: A post-wildfire damaged buildings classifier, IJDRR 65. doi:10.1016/j.ijdrr.2021.102540.
31. Kasraee, Hawbaker & Radeloff 2023, Identifying building locations in the WUI before and after fires with CNNs, Int. J. Wildland Fire 32(4). doi:10.1071/WF22181. Missed buildings under trees bias destruction rates.
32. Trivedi et al. 2025, Advancing Wildfire Damage Assessment with Aerial Thermal Remote Sensing and AI: Applications to the 2025 Eaton and Palisades Fires, Remote Sensing 17:3962. doi:10.3390/rs17243962.
33. Du & Feng 2025, Post-wildfire damage assessment of buildings in the 2025 Palisades fire based on InSAR, IJDRR. doi:10.1016/j.ijdrr.2025.105809.
34. Antoine 2026, Assessing Structure Collapse and Vegetation Loss After the 2025 Eaton Fire Using Optical Remote Sensing, Earth and Space Science. doi:10.1029/2025EA004583.
35. Esparza et al. 2025/26, Automated Wildfire Damage Assessment from Multi-view Ground-level Imagery via Vision Language Models, arXiv:2509.01895.
36. Xiao, Ho, Thasma, Ma & Mostafavi 2026, Damage-TriageFormer, arXiv:2606.12248 (checked). Post-event only, footprint-conditioned, NOAA aerial imagery incl. 2025 LA wildfires; macro-F1 0.62.
37. Farajpoor & Narimani 2026, The spatial anatomy of urban wildfire vulnerability (Palisades), arXiv:2608.22293. 12,081 DINS inspections; AUC 0.92 random CV vs 0.75 spatial-block CV.
38. Sener et al. 2026, CalFireSegNet, Scientific Reports. doi:10.1038/s41598-026-67452-7. Footprint disappearance as loss proxy.
39. Sabir & Khati 2026, Forest fire damage assessment using Sentinel-1 dual-pol SAR, EGU26-16366 (abstract).

### 5. Training-free, label-light and operational systems
40. Robinson et al. 2023, Rapid building damage assessment workflow (Rolling Fork tornado), ICCV HADR. arXiv:2306.12589.
41. Robinson et al. 2026, HASTE: A Platform for Rapid Post-Disaster Building Damage Assessment, arXiv:2607.11838 (checked). Foundation-model embeddings + logistic regression with few in-event labels; 30+ deployments incl. wildfires.

### 6. Benchmark vs field data
42. Vescovo et al. 2025, The 2024 Noto Peninsula earthquake building damage dataset, ESSD 17:5259. doi:10.5194/essd-17-5259-2025. Image annotation vs field survey F1 0.94 (survived vs destroyed).

Not verified: SegEarth-OV3 building datasets; whether Trivedi et al. used DINS; xBD-S12 Palisades case; Du & Feng validation details; whether BRIGHT contains wildfire events.

## B. What already exists

- SAM tiling + cross-tile merging (Remote SAMsing, SAM 2). Ours is a SAM 3 measurement on xView2.
- SAM 3 remote-sensing evaluations (SegEarth-OV3, Dabaja & Celik), without per-hazard xView2 instance F1.
- Cross-hazard failure of damage models (Benson & Ecker; Gerard et al.; BRIGHT 2026). Our flood→fire collapse confirms it.
- Two-stage localise-then-classify is standard (ChangeOS, DamageMap, BRIGHT winners). Not novel.
- LA 2025 already studied with thermal ML, InSAR, stereo elevation, ground VLMs, DINOv3 (Damage-TriageFormer), DINS risk models.
- Closest competitors: Damage-TriageFormer, Trivedi et al., HASTE, DAVI, DamageMap.

No paper found that evaluates a fully open-weight zero-shot pipeline on free VHR satellite imagery
of both LA fires per structure against the full DINS set, with multiple post-event dates.

## C. Gaps

1. Per-structure validation of a zero-shot / transferred pipeline against DINS for both fires.
2. Decomposing end-to-end error into footprint vs classification error on real data (oracle-footprint ablation).
3. Quantifying a calibrated-but-wrong hazard shift (temperature-calibrated flood model on fire) and testing cheap fixes (fire-trained model, footprint persistence, few-label probe).
4. Value of multiple post-event dates against field truth.
5. Hazard-dependent recall of SAM 3 footprints (when zero-shot footprints are safe to use).
6. Geolocation offset between VHR imagery and cadastral footprints (~4 m here) as an evaluation pitfall.

## D. Positioning options

1. (Recommended) Evaluation / operational-lessons paper: how far zero-shot and transferred models get
   against field inspections for the 2025 LA fires. Needs DINS agreement for both fires, spatial-block
   reporting, footprint-source ablation (SAM 3 vs LARIAC/Microsoft), at least one cheap damage
   alternative, multi-date vs single-date. Venues: IJDRR, Natural Hazards Review, Fire Technology,
   IJAEOG, Science of Remote Sensing; workshops AI+HADR, CVPR EarthVision.
2. Failure analysis: calibrated but wrong under hazard-type shift, validated with DINS. Needs
   calibration/uncertainty analysis under shift, an xBD-wildfire-trained reference, one label-light
   adaptation. Venues: AI+HADR, EarthVision, IEEE JSTARS/GRSL.
3. Short technical note on SAM 3 as a zero-shot footprint source (tiling, threshold, per-hazard
   recall). Weakest novelty; better as a section of option 1.

Note on the internal BRIGHT result (two-stage 0.09 vs Mask R-CNN 0.244): the 2026 challenge
winners were decoupled two-stage designs at ~0.18 test mAP, so treat the internal number as an
implementation issue, not evidence against two-stage designs.
