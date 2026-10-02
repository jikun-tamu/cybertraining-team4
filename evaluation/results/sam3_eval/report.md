# SAM3 Building Segmentation — Evaluation Report

**IoU threshold**: 0.5 (detection match criterion)

**Metrics definition**: TP = predicted building matched to a GT building (IoU ≥ threshold); FP = unmatched prediction; FN = unmatched GT building.

---

## Split: TEST

| Metric | Value |
|--------|-------|
| Images evaluated | 933 |
| GT buildings | 54862 |
| Predicted buildings | 42087 |
| Avg pred / image | 45.1 |
| Avg GT / image | 58.8 |
| True Positives (TP) | 31016 |
| False Positives (FP) | 11071 |
| False Negatives (FN) | 23846 |
| **Precision** | **0.7369** |
| **Recall** | **0.5653** |
| **F1** | **0.6398** |
| **Mean IoU (matched pairs)** | **0.7559** |
| Matched pairs | 31016 |

### Per-Disaster Breakdown

| Disaster | Images | GT | Pred | Precision | Recall | F1 | Mean IoU |
|----------|-------:|---:|-----:|----------:|-------:|---:|---------:|
| guatemala-volcano | 5 | 32 | 34 | 0.647 | 0.688 | 0.667 | 0.812 |
| hurricane-florence | 108 | 2268 | 2254 | 0.855 | 0.850 | 0.853 | 0.798 |
| hurricane-harvey | 108 | 7715 | 8052 | 0.731 | 0.762 | 0.746 | 0.759 |
| hurricane-matthew | 73 | 4189 | 2772 | 0.752 | 0.497 | 0.599 | 0.733 |
| hurricane-michael | 98 | 5657 | 5869 | 0.700 | 0.727 | 0.713 | 0.737 |
| mexico-earthquake | 38 | 11411 | 4627 | 0.654 | 0.265 | 0.377 | 0.734 |
| midwest-flooding | 80 | 2532 | 2260 | 0.718 | 0.641 | 0.677 | 0.734 |
| palu-tsunami | 42 | 12560 | 6867 | 0.805 | 0.440 | 0.569 | 0.763 |
| santa-rosa-wildfire | 74 | 4226 | 4764 | 0.769 | 0.868 | 0.816 | 0.783 |
| socal-fire | 307 | 4272 | 4588 | 0.687 | 0.737 | 0.711 | 0.752 |
