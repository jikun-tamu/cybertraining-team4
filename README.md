<<<<<<< HEAD
# Assessing Disaster Impact with Multimodal Geospatial Data

This repository contains the code and analysis for the project "Assessing Disaster Impact Through Streamlining Multimodal Geospatial Data with Building Damage Prediction and Demographic Attributes".

## Project Structure

```
your-project/
├─ README.md                 # What/why/how; quickstart; data access notes
├─ pyproject.toml            # Dependencies + tooling (ruff, black, pytest)
├─ environment.yml           # Or requirements.txt; pin major versions
├─ .gitignore                # Ignore data/, artifacts/, checkpoints/, etc.
├─ data/                     # NOT tracked in git
│  ├─ external/              # Third-party (licenses!)
│  ├─ raw/                   # Immutable raw dumps
│  ├─ interim/               # Cleaned/merged but not final
│  └─ processed/             # Final analysis-ready
├─ notebooks/                # Exploratory work; 1 topic = 1 notebook
├─ src/your_project/         # Importable package (functions live here)
├─ scripts/                  # Thin CLIs calling src/ (zero logic here)
├─ configs/                  # YAML/TOML configs (Hydra/OmegaConf friendly)
├─ experiments/              # One folder per run (auto-created)
├─ results/                  # Aggregated tables/figures for the paper
├─ tests/                    # pytest unit tests (fast!)
└─ CITATION.cff              # How to cite your work
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jikun-tamu/cybertraining-team4.git
   cd cybertraining-team4
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate cybertraining-team4
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Explore the notebooks:**
   - `notebooks/stage1.ipynb`
   - `notebooks/validation_case.ipynb`
   - `notebooks/validation_LA_fire.ipynb`

## Natural Disaster Case Studies for Model Testing and Validation

This repository examines recent U.S. disasters to test and validate model performance across varied physical processes and socio-demographic contexts.

### Case Studies

#### 1. 2025 Southern California Wildfires
- Among the most destructive in U.S. history
- Severe wildland–urban interface losses and unprecedented damages

#### 2. 2024 Hurricane Helene (Northwestern North Carolina)
- Catastrophic inland flooding in the Appalachians
- Most fatalities occurred far from the coast

#### 3. 2023 Midwest Tornado Outbreak (Greenfield, Iowa)
- Multi-day outbreak of high-intensity tornadoes
- Greenfield tornado recorded near-historic wind speeds

