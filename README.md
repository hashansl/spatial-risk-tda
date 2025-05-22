# Spatial‑Risk‑TDA

*A Topological Data Analysis (TDA) workflow for measuring un‑measured spatial risk and fitting Besag–York–Mollié (BYM‑T) models to opioid‑related health outcomes.*

This repository contains the full research pipeline—from raw county‑level data to Bayesian spatial models—used in our study of opioid vulnerability across U.S. counties. The core stages are:

1. **Data pre‑processing** – Clean Social Vulnerability Index (SVI) shapefiles, overdose mortality counts, and census populations.
2. **TDA summary generation** – Build adjacency‑based simplicial complexes and extract persistence summaries that capture latent spatial structure.
3. **Bayesian modeling** – Re‑parameterise the BYM model to include the TDA effect (BYM‑T) and fit it in selected states.

---

## 📁 Repository layout

```text
SPATIAL-RISK-TDA/
├── data/
│   ├── raw_data/                # original SVI & overdose data (not included)
│   └── processed_data/          # cleaned & merged data ready for analysis
│
├── data_processing/
│   ├── main_data_process.ipynb  # processes SVI and overdose data(scaling per state/ data cleaning)
│   ├── svi_treat_null.py        # helper script to impute / treat SVI nulls/ scale
│   └── smr.py                   # utilities for Standard Mortality Ratio (SMR)
│
├── simulations/                 # synthetic experiments
│   ├── controlled_ac.ipynb      # vary autocorrelation strength
│   ├── controlled_avg.ipynb     # vary mean SVI level
│   └── controlled_grid.ipynb    # vary spatial grid patterns
│
├── results/                     # model outputs, figures & tables
│
├── utills/                      # adjacency method, tda summaries
│
├── generate_tda_summaries.ipynb # tda summaries for all counties in a state
├── bym_modeling_tn.ipynb        # BYM case study – Tennessee
├── bym_modeling_va.ipynb        # BYM case study – Virginia
│
├── requirements.txt             # reproducible Python environment
├── .gitignore                   # ignore large / sensitive files
└── README.md                    # you are here
```

> **Note**: Raw data live in `data/raw_data/` and are *not* under version control. Add them manually following the instructions below.

---

## 🚀 Quick‑start

### 1. Clone & create environment

```bash
git clone https://github.com/<your‑org>/spatial-risk-tda.git
cd spatial-risk-tda
conda create -n spatial-tda-env python=3.11
conda activate spatial-tda-env
pip install -r requirements.txt
```

### 2. Supply raw inputs

Place the following files in **`data/raw_data/`**:

| Dataset                                    | Source | Example filename          |
| ------------------------------------------ | ------ | ------------------------- |
| CDC WONDER overdose mortality counts       | NVSS   | `mort_2018.csv`           |
| Social Vulnerability Index (SVI)           | CDC    | `SVI_2018_US.shp`         |

### 3. Run the pre‑processing pipeline

Open **`data_processing/main_data_process.ipynb`** and execute all cells. Edit the data paths to point to your local copies of the raw data. Change the year and the variable names depending on the SVI version you are using.

To scale the SVI variables per state, add state abbr to the `state_abbr` variable in the notebook. The notebook will then create a scaled data file for each state in the `data/processed_data/` folder.

SMR for each study area is calculated using the `smr.py` script. The script takes the cleaned data and calculates the SMR for each county. 

Cleaned outputs will be written to **`data/processed_data/`**.

### 4. Generate TDA summaries

```bash
jupyter nbconvert --execute generate_tda_summaries.ipynb
```

This notebook constructs adjacency‑based simplicial complexes for each county‑level snapshot and generates TDA summaries for each county. The summaries are saved in **`data/processed_data/`**.


### 5. Fit BYM‑T models

Select a state‑specific notebook (e.g. **`bym_modeling_tn.ipynb`**) and run the full workflow to:

1. Load processed data and TDA summaries.
2. Compose the BYM2/BYM-T hierarchical model in PyMC.
3. Sample with NUTS, diagnose convergence, and save posterior draws.
4. Produce comparative WAIC/LOO metrics.

Outputs (trace files, figures) are saved under **`results/`**.

### 6. (Optional) Synthetic simulations

Use notebooks in **`simulations/`** to evaluate how well TDA summaries capture known spatial structures under controlled settings.

---

## 📜 Citation

If you use this code, please cite:

<!-- ```bibtex
@article{your2025tda,
  title   = {},
  author  = {},
  journal = {},
  year    = {}
}
``` -->

---

## 📝 License

MIT – see `LICENSE` for details.

---