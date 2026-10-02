# SAM3 Prompt Comparison

**Dataset**: xView2 test set, pre-disaster (933 images)  
**Match criterion**: IoU >= 0.5  
**Stage-1 config**: `{"model_id": "facebook/sam3", "confidence_threshold": 0.4, "tile_size": 512, "tile_overlap": 64, "merge_iou": 0.5, "min_size": 100, "min_polygon_area": 100.0, "polygon_epsilon": 2.0, "code_version": null}`

| Prompt | Precision | Recall | F1 | Mean IoU | Predicted | Images w/o pred |
|--------|----------:|-------:|---:|---------:|----------:|----------------:|
| `building` | 0.7369 | 0.5653 | 0.6398 | 0.7559 | 42087 | 227 |
| `house` | 0.7083 | 0.5929 | 0.6455 | 0.7516 | 45922 | 224 |

## F1 per disaster

| Disaster | building | house |
|----------| -----:| -----:|
| guatemala-volcano | 0.667 | 0.597 |
| hurricane-florence | 0.853 | 0.839 |
| hurricane-harvey | 0.746 | 0.716 |
| hurricane-matthew | 0.599 | 0.619 |
| hurricane-michael | 0.713 | 0.697 |
| mexico-earthquake | 0.377 | 0.427 |
| midwest-flooding | 0.677 | 0.665 |
| palu-tsunami | 0.569 | 0.618 |
| santa-rosa-wildfire | 0.816 | 0.799 |
| socal-fire | 0.711 | 0.693 |
