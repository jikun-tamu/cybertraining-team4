# SAM3 Prompt Strategy Comparison

**Dataset**: xView2 test set (933 images, 10 disaster types)  
**IoU threshold**: 0.5

---

## Overall Metrics

| Prompt | Precision | Recall | F1 | Mean IoU | Images w/o pred |
|--------|----------:|-------:|---:|---------:|----------------:|
| `building` | 0.6820 | 0.2837 | 0.4008 | 0.7589 | 276 |
| `house` | 0.6979 | 0.3180 | 0.4369 | 0.7501 | 285 |
| `rooftop` | 0.8451 | 0.0570 | 0.1068 | 0.8061 | 704 |
| `building rooftop` | 0.8043 | 0.0746 | 0.1365 | 0.8035 | 633 |
| `structure` | 0.7410 | 0.1318 | 0.2238 | 0.7734 | 371 |

## F1 per Disaster

| Disaster | building | house | rooftop | building rooftop | structure |
|----------| -----:| -----:| -----:| -----:| -----:|
| guatemala-volcano | 0.630 | 0.356 | 0.000 | 0.000 | 0.360 |
| hurricane-florence | 0.751 | 0.757 | 0.268 | 0.482 | 0.696 |
| hurricane-harvey | 0.421 | 0.537 | 0.093 | 0.111 | 0.182 |
| hurricane-matthew | 0.286 | 0.377 | 0.018 | 0.028 | 0.127 |
| hurricane-michael | 0.656 | 0.664 | 0.184 | 0.205 | 0.447 |
| mexico-earthquake | 0.057 | 0.062 | 0.010 | 0.013 | 0.027 |
| midwest-flooding | 0.482 | 0.521 | 0.066 | 0.093 | 0.337 |
| palu-tsunami | 0.151 | 0.188 | 0.004 | 0.010 | 0.033 |
| santa-rosa-wildfire | 0.655 | 0.673 | 0.430 | 0.458 | 0.366 |
| socal-fire | 0.586 | 0.592 | 0.164 | 0.228 | 0.387 |
