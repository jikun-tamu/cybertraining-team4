# Assessing Disaster Impact with Multimodal Geospatial Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

> **⚠️ PROJECT CONCLUDED (2026-07-10).** This project is no longer under active
> development. **Start with [PROJECT_CONCLUSION.md](PROJECT_CONCLUSION.md)** — it records
> final results, why the project ended, a full asset inventory (including which model
> checkpoints exist only on local disk), how to re-run everything, and the recommended
> continuation path (CAL FIRE DINS validation) if anyone picks this up.

This repository contains the source code for the project **"Assessing Disaster Impact Through Streamlining Multimodal Geospatial Data with Building Damage Prediction and Demographic Attributes"**. Our goal is to develop a robust pipeline for rapidly assessing building damage and socio-economic impacts following natural disasters, using a combination of satellite imagery and demographic data.

## Mission
To provide a streamlined, open-source framework for multimodal geospatial analysis that empowers researchers, first responders, and policymakers to better understand and mitigate the impacts of natural disasters.

## Project Structure
The repository is organized following modern data science best practices to ensure clarity, reproducibility, and scalability.

```
cybertraining-team4/
├─ README.md                 # Project overview (this file)
├─ PROJECT_CONCLUSION.md     # ★ Final status, results, asset inventory, resumption guide
├─ CLAUDE.md                 # Working notes: environments, commands, parameters
├─ stage1/                   # SAM3 zero-shot building detection package (production)
├─ pipeline/                 # Combined Stage 1 + Stage 2 pipeline (models, configs, docs)
├─ evaluation/               # xView2 benchmark script + results
├─ reports/                  # M2b validation + I-GUIDE competition audit
├─ results/                  # LA fire figures, prompt experiment outputs
├─ notebooks/                # EDA and validation notebooks
├─ src/cybertraining_team4/  # Early training code + collaborator's Stage 2 handoff
├─ data/                     # (Not tracked by git) LA fire data
├─ archive/                  # Exploration work: Mask R-CNN, PolyWorld, early SAM3 (not production)
└─ II_package/               # Local clone of collaborator's I-GUIDE submission repo (not tracked)
```

---

## Getting Started

### 1. Prerequisites
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) package manager
- Git

### 2. Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jikun-tamu/cybertraining-team4.git
    cd cybertraining-team4
    ```

2.  **Create and activate the Conda environment:**
    This command sets up an isolated environment with all the necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate cybertraining-team4
    ```

3.  **Install the project package:**
    This installs the `src/cybertraining_team4` directory as a package, making your code importable across the project.
    ```bash
    pip install -e .
    ```

---

## Project Workflow
The project is structured as a pipeline from data ingestion to final analysis.

### 1. Data Preparation
- **Raw data** is stored in `data/raw/`. This data should be considered immutable.
- **Processing scripts** (e.g., `src/cybertraining_team4/process_chips_600m.py`) transform raw data into analysis-ready formats.
- **Intermediate data** is saved in `data/interim/`.
- **Final datasets** for modeling are stored in `data/processed/`.
- See the `data/README.md` for more details on the data structure.

### 2. Model Training
- The core training logic is located in `src/cybertraining_team4/`. For example, `stage1_train.py` handles the first stage of model training.
- Experiments can be launched via scripts in the `scripts/` directory.

### 3. Evaluation
- Evaluation scripts in `src/cybertraining_team4/` assess model performance.
- Results from individual runs are saved in the `experiments/` directory.

### 4. Exploration and Analysis
- The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis (EDA), visualization, and validation of results. Key notebooks include:
  - `notebooks/xview2_eda_unet.ipynb` — xView2 EDA and UNet training prototype
  - `notebooks/validation_case.ipynb` — LA fire AOI setup, grid creation, Maxar download

---

## Case Studies
The pipeline was deployed end-to-end on one real event:

- **2025 Southern California Wildfires (completed)**: 295 grid cells, 21,797 building
  instances with multi-date damage predictions. Not validated against ground truth —
  see PROJECT_CONCLUSION.md §4 for the recommended DINS validation path.

Hurricane Helene (2024) and the 2023 Midwest tornado outbreak were originally planned
as additional case studies but were not pursued.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code or our findings in your research, please cite us as follows:
```
(Citation details to be added here)
```

