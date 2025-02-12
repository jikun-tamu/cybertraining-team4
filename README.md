# Personality and Urban Environment Analysis

This project analyzes the relationship between personality traits and urban environmental characteristics using street view imagery and survey data.

## Prerequisites

- Python 3.8+
- Required Python packages (install using `pip install -r requirements.txt`)

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data directories:
   ```bash
   mkdir -p data/{raw,processed}
   ```

## Data Structure

The script expects the following data structure:

```
/media/data/personality/
├── OOS Master Dataset Sept 2022/
│   └── 00_OOS_MASTER DATASET_2022_08_22.sav
└── cb_2016_us_zcta510_500k/
    └── cb_2016_us_zcta510_500k.shp

./data/
├── raw/
│   ├── city1/
│   │   └── gsv_pids.csv
│   ├── city2/
│   │   └── gsv_pids.csv
│   └── ...
└── processed/
    ├── city1/
    │   ├── batch_1/
    │   │   ├── label_counts.csv
    │   │   └── pixel_ratios.csv
    │   └── ...
    └── ...
```

## Usage

Run the data preparation script:
```bash
python code/prepare_dataset.py
```

This will:
1. Create a SQLite database
2. Load and process all data sources
3. Perform spatial join between GSV points and zipcodes
4. Aggregate environmental features by zipcode
5. Save all data to the database

## Output

The script creates a SQLite database (`data/personality.db`) with the following structure:

- Tables:
  - `survey_data`: Complete personality survey responses
  - `gsv_panoramas`: Street view image locations with zipcode information
  - `segmentation_results`: Raw image segmentation results with both pixel ratios and label counts
  - `segment_stats_by_zipcode`: Environmental features aggregated by zipcode

The `segment_stats_by_zipcode` table includes:
- `zipcode`: ZIP Code Tabulation Area (ZCTA)
- `image_count`: Number of street view images in the zipcode
- Average values for all segmentation metrics:
  - Pixel ratios (suffix: `_ratios`)
  - Label counts (suffix: `_counts`)

## Analysis

After running the preparation script, you can query the database for analysis. Example queries:

```sql
-- Get environmental characteristics for a participant's youth residence
SELECT s.*, g.*
FROM survey_data s
JOIN segment_stats_by_zipcode g ON s.youth_zip = g.zipcode;

-- Compare youth and current environment characteristics
SELECT 
    s.record_id,
    y.building_ratios as youth_building,
    c.building_ratios as current_building,
    y.vegetation_ratios as youth_vegetation,
    c.vegetation_ratios as current_vegetation
FROM survey_data s
JOIN segment_stats_by_zipcode y ON s.youth_zip = y.zipcode
JOIN segment_stats_by_zipcode c ON s.now_zip = c.zipcode;
``` 