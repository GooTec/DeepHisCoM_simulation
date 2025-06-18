# DeepHisCoM Simulation & Permutation Testing

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Simulation Data Generation](#simulation-data-generation)
  - [Scenarios & Settings](#scenarios--settings)
  - [Preprocessing](#preprocessing)
  - [Running the Simulation](#running-the-simulation)
- [Permutation Testing with DeepHisCoM](#permutation-testing-with-deephiscom)
  - [Model & Data](#model--data)
  - [Configuration](#configuration)
  - [Running Permutations](#running-permutations)
- [Dependencies](#dependencies)
- [Example Usage](#example-usage)
- [License](#license)

---

## Overview

This repository provides tools to:

1. **Generate simulated binary outcomes** from real metabolite data under three effect scenarios (linear, interaction, quadratic).
2. **Train DeepHisCoM** on each simulated dataset, with optional phenotype permutation tests to assess pathway–disease associations.

---

## Directory Structure

.
├── 181_metabolite_clinical.csv # Raw clinical + metabolite data  
├── layerinfo.csv # Node counts & layer depths  
├── annot.csv # Metabolite-to-pathway annotations  
├── simulation/ # Generated simulation outputs  
│ ├── linear_beta_0.1.csv  
│ ├── linear_beta_0.2.csv  
│ ├── …  
│ ├── interaction_w_0.1.csv  
│ ├── …  
│ └── quadratic_beta_0.6.csv  
├── DeepHisCoM_simulation.ipynb # Notebook: loads simulations & runs model  
└── README.md # This file

---

## Simulation Data Generation

### Scenarios & Settings

| Scenario        | Description                                            | Parameters                                                        |
| --------------- | ------------------------------------------------------ | ----------------------------------------------------------------- |
| **Linear**      | Sum of metabolites in two pathways                     | weight w = 1.5; β ∈ {0.1, 0.2, 0.3, 0.4}                          |
| **Interaction** | Linear effects + pairwise interactions within pathways | linear w_i = 1; interaction w_int ∈ {0.1, 0.2, 0.3, 0.4}; β = 0.4 |
| **Quadratic**   | Sum of squared metabolites (alternating signs)         | w_i = (–1)^{i+1}; β ∈ {0.6, 0.7, 0.8, 0.9}                        |

### Preprocessing

1. **Log-transform** metabolites  
   X_log = np.log(raw_metabolites + 1e-6)
2. **Standardize** to mean 0, variance 1  
   from sklearn.preprocessing import StandardScaler  
   X_scaled = StandardScaler().fit_transform(X_log)
3. **Pathway features**
   - map00400 : [Phenylalanine, Tyrosine, Tryptophan]
   - map00860 : [Heme, Protoporphyrin, …]

### Running the Simulation

jupyter nbconvert --execute DeepHisCoM_simulation.ipynb

Outputs (saved in _simulation/_):

- linear*beta*{β}.csv
- interaction*w*{w_int}.csv
- quadratic*beta*{β}.csv

Each CSV contains  
• Columns 1–14 : original clinical covariates  
• Last column : simulated binary phenotype

---

## Permutation Testing with DeepHisCoM

### Model & Data

- **Model** : DeepHisCoM (hierarchical pathway blocks → sigmoid)
- **Input** : pathway-summarized metabolites (no extra covariates)
- **Loss** : BCELoss or MSELoss
- **Regularization** : optional L1/L2 on pathway-disease & bio-pathway weights

### Configuration (CLI flags)

--dir Base directory path  
--scenario Path to simulation CSV  
--seed Random seed (default 100)  
--perm Number of phenotype permutations (default 1000)  
--batch_size 0 = full-batch  
--learning_rate …  
--activation tanh | relu | leakyrelu | identity  
--loss BCELoss | MSELoss  
--reg_type l1 | l2  
--reg_const_pathway_disease  
--reg_const_bio_pathway  
--stop_type Early-stopping rule  
--divide_rate Train/test split ratio  
--count_lim Patience for stopping

### Running Permutations

jupyter nbconvert --execute DeepHisCoM_simulation.ipynb \
 --ExecutePreprocessor.timeout=600 \
 --to notebook --output results.ipynb \
 --argv="--dir . --scenario simulation/linear_beta_0.1.csv --perm 1000 --batch_size 0"

Procedure

1. Load simulated data
2. Standardize features & phenotype
3. For each permutation (if perm ≠ 0):
   - Shuffle phenotype
   - Train DeepHisCoM
   - Track AUC / loss
   - Save best weights to `exp/tmp/param{perm}.txt`
   - For the original data (perm=0) also write validation AUC to
     `exp/tmp/val_auc.txt`

---

## Dependencies

• Python >= 3.7  
• numpy, pandas, scikit-learn, scipy  
• torch >= 1.7  
• jupyter

Installation  
pip install numpy pandas scikit-learn scipy torch notebook

---

## Example Usage

Generate all simulations  
jupyter nbconvert --execute DeepHisCoM_simulation.ipynb

Run DeepHisCoM on one scenario
jupyter nbconvert --execute DeepHisCoM_simulation.ipynb \
 --argv="--dir . --scenario simulation/linear_beta_0.1.csv --perm 1000"

### Empirical Power & FDR

After computing permutation p-values with `compute_pvalues.py`, you can
summarize empirical power and the false discovery rate:

```bash
python analyze_power_fdr.py
# The script will prompt for the result directory (e.g. `exp`)
```

---

## License

MIT © 2025
