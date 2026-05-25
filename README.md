# Life Course Epidemiology Deep Learning Architecture Study

> **Final Year Project, University of Bristol**
> Supervised by **Dr Juliette Unwin** (University of Bristol) and **Helen Coupland** (Imperial College). Parts of this work were completed in collaboration with Jeanelle Leong (University of Bristol) and Isobelle Clemmens (University of Bristol).

This project builds upon the research and code from:
> Coupland et al. (2024). *Deep learning for life course epidemiology.*
> GitHub: [hcoupland/LifeCourseDLSimulationStudy](https://github.com/hcoupland/LifeCourseDLSimulationStudy)

All simulation methodology, Life Course Pattern (LCP) rule definitions, and the core modelling framework originate from that work.

## Quickstart

### 1. Install

```bash
git clone https://github.com/<your-username>/life-course-epidemiology.git
cd life-course-epidemiology
pip install -e ".[dev]"
```

### 2. Simulate data (R)

```r
# In RStudio or from the terminal:
Rscript Data_simulation/Data_generation_DAG.R
```

Outputs `data/data_X.npy` and `data/data_{LCP}_Y.npy` for all 14 LCPs.

### 3. Run a model

```bash
# Logistic Regression across all data sizes
lce --model-name LR

# ResNet with Optuna hyperparameter search
lce --model-name ResNet --run-hyper-opt --num-optuna-trials 20

# BasicCNN
python -m lce.models.cnn
```

### 4. Visualise results

```bash
# All models — interactive by LCP and metric
python visualisations/plot_results.py --csv output/all_results.csv

# CNN results only
python visualisations/plot_results.py --csv metrics_results.csv --model CNN

# AUC gap between CNN and baseline
python visualisations/plot_results.py --csv metrics_diff.csv --diff
```

### 5. Collate results (R)

```r
source("Results_collation/collate_results.R")
```

## Handling Class Imbalance

Given the severe label imbalance inherent in life course data:

- **DL models** — `FocalLoss` (α, γ tuned via Optuna)
- **BasicCNN** — weighted `CrossEntropyLoss` (minority class weight = 25×)
- **XGBoost** — `scale_pos_weight` auto-computed per fold
- **Logistic Regression** — `class_weight="balanced"` + L1 penalty
- All DL models undergo **probability calibration** (quantile strategy via tsai)
- XGBoost and LR use **Platt scaling** (`CalibratedClassifierCV`)
