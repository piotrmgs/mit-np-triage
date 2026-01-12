# mit-np-triage

Code to reproduce the experiments in:

> **Auditing High-Confidence ECG Predictions with a Newton–Puiseux Onset Score**  

This repository implements a **post-hoc audit / triage pipeline** for ECG classification, with a focus on **silent failures** (high-confidence misclassifications inside an accept set defined by a confidence threshold). The main contribution is the **Newton–Puiseux onset score** `r_dom` (NP-onset), used **only as a ranking signal** to allocate a limited audit budget within accepted predictions.

---

## Repository layout

- `up_real/`  
  Training + calibration + export of full-test predictions for MIT–BIH (Normal vs PVC) with record-wise (patient-wise) splits.

- `post_processing_real/`  
  Cohort construction (uncertain vs high-confidence accept), per-record balanced sampling, local surrogate fitting, Newton–Puiseux coefficient extraction, and per-score triage evaluation inputs.

- `NP-analysis_real/`  
  Aggregates post-processing outputs into paper-ready metrics (AUPRC, PR curves, review-budget curves) and bootstrap sensitivity analyses.

- `mit_bih_pre/`  
  MIT–BIH preprocessing utilities (reads PhysioNet-format records, builds beat-aligned windows).

- `src/`  
  Shared utilities (data handling, models, calibration, fitting, plotting).

---


## Project Structure
```
├── mit‑bih/                      # Raw MIT‑BIH dataset
├── mit_bih_pre/                  # Preprocessing scripts for MIT‑BIH
│   └── pre_pro.py                # Signal filtering and feature extraction
├── src/                          # Core library modules
│   ├── post_processing.py        # Post‑processing (common)
│   ├── find_up_synthetic.py      # Uncertainty mining on synthetic data
│   ├── find_up_real.py           # Uncertainty mining on MIT‑BIH data
│   ├── local_analysis.py         # Local surrogate + Puiseux wrapper
│   └── puiseux.py                # Newton‑Puiseux solver
├── up_synth/                     # Synthetic dataset training and evaluation
│   └── up_synthetic.py
├── up_real/                      # MIT‑BIH CVNN training and evaluation
│   └── up_real.py
├── post_processing_real/         # Post‑processing for MIT‑BIH data
│   └── post_processing_real.py
├── NP-analysis_real/             # Newton–Puiseux evidence & triage (MIT-BIH)
│   └── NP-analysis_real.py
└── README.md                     # This file
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/piotrmgs/mit-bih-triagle.git
   cd puiseux-cvnn
   ```   
2. Create a virtual environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate mit-bih-triagle
   ```

---

## Data: MIT‑BIH Arrhythmia (PhysioNet) ⚠️

**MIT-BIH Arrhythmia Database is *not* stored in this repository.**

**Manual download**  
   * Go to <https://physionet.org/content/mitdb/1.0.0/>  
   * Download all files and unzip them into `mit-bih/`.

**Licence notice:**  
MIT-BIH data are released under the PhysioNet open-access license.  
By downloading the files you agree to its terms.

---

## Reproducing the BSPC paper results

All commands below are meant to be executed **from the repository root**.

### 1) Train + export full-test predictions (per seed)

This step trains the classifier with **record-wise 10-fold splitting**, applies post-hoc calibration, and writes a *full-test fold* prediction CSV used by the downstream pipeline.

Example for one seed (replace `SEED` and output folder):

```bash
SEED=11111

python -m up_real.up_real \
  --output_folder results/bspc_seed${SEED} \
  --seed ${SEED} \
  --epochs 20 \
  --lr 1e-3 \
  --batch_size 128 \
  --folds 10 \
  --calibration platt \
  --calibs platt,beta,vector,temperature,isotonic,none \
  --review_budget 50 \
  --budget_fracs 0.005,0.01,0.02,0.05,0.10 \
  --auto-full-folds \
  --embed_method pca \
  --embed_max_points 5000 \
  --num_workers 0 \
  --pin_memory \
  --save-arrays \
  --save-pred-csv
```

**Key outputs (per seed):**
- `results/bspc_seedSEED/full_test_predictions_ext.csv` (anchors + predictions used by post-processing)
- `results/bspc_seedSEED/best_model_full.pt` (trained model checkpoint)
- `results/bspc_seedSEED/run_args.json`, `run_meta.json` (provenance)

### 2) Post-processing: uncertain cohort (per seed)

Constructs an **uncertainty-enriched review cohort** and runs local analysis on a **per-record balanced** sample (`n=1000` in the paper setting).

```bash
SEED=11111

python -m post_processing_real.post_processing_real \
  --output_folder results/bspc_seed${SEED} \
  --out_dir results/bspc_pp_uncertain_perrec_seed${SEED}_n1000 \
  --anchors_csv full_test_predictions_ext.csv \
  --selection per_record_mixed \
  --rank_by margin \
  --max_points 1000 \
  --attack_radius 0.05 \
  --triage_bootstrap 5000 \
  --seed ${SEED} \
  --skip_lime --skip_shap --skip_contours --skip_benchmark --skip_multiplicity \
  --paper
```

### 3) Post-processing: high-confidence accept cohort (per seed)

Constructs the **accept-set audit cohort** by filtering to `pmax >= 0.95` (calibrated probabilities), sampling uniformly per record (`n=1000` in the paper setting), and running local analysis with a larger neighborhood radius.

```bash
SEED=11111

python -m post_processing_real.post_processing_real \
  --output_folder results/bspc_seed${SEED} \
  --out_dir results/bspc_pp_highconf_accept_p95_perrec_seed${SEED}_n1000 \
  --anchors_csv full_test_predictions_ext.csv \
  --filter_accepted_only \
  --filter_pmax_min 0.95 \
  --selection per_record_random \
  --max_points 1000 \
  --attack_radius 0.15 \
  --robust_steps 80 \
  --robust_num_random 40 \
  --triage_bootstrap 5000 \
  --seed ${SEED} \
  --skip_lime --skip_shap --skip_contours --skip_benchmark --skip_multiplicity \
  --paper
```

**Key outputs (per cohort / per seed):**
- `selected_uncertain_points.csv` (the evaluated cohort; includes labels and computed scores)
- `dominant_ratio_summary.csv` (NP coefficients / onset summaries)
- `local_fit_summary.csv` (surrogate fidelity diagnostics)
- `fig/` (cohort-level plots generated by the post-processing stage)
- `paper_summary.md` + `run_command.txt` (paper-oriented summary + provenance)

### 4) NP-analysis: compute AUPRC + PR + budget curves (per seed)

This stage reads the two cohort directories produced above and writes the **paper-ready** tables and figures (AUPRC, PR curves, review-budget curves). Two bootstrap modes are used in the paper workflow:

#### 4a) Stratified bootstrap
```bash
SEED=11111

python NP-analysis_real/NP-analysis_real.py \
  --mode bspc_error \
  --pp_dirs results/bspc_pp_uncertain_perrec_seed${SEED}_n1000,results/bspc_pp_highconf_accept_p95_perrec_seed${SEED}_n1000 \
  --cohort_names uncertain,highconf_accept_p95 \
  --out_dir results/NP_bspc_seed${SEED}_n1000 \
  --bootstrap 5000 \
  --bootstrap_kind stratified \
  --budget_fracs 0.005,0.01,0.02,0.05,0.10 \
  --seed ${SEED}
```

#### 4b) Record-level cluster bootstrap (sensitivity analysis)
```bash
SEED=11111

python NP-analysis_real/NP-analysis_real.py \
  --mode bspc_error \
  --pp_dirs results/bspc_pp_uncertain_perrec_seed${SEED}_n1000,results/bspc_pp_highconf_accept_p95_perrec_seed${SEED}_n1000 \
  --cohort_names uncertain,highconf_accept_p95 \
  --out_dir results/NP_bspc_seed${SEED}_n1000_clusterboot \
  --bootstrap 5000 \
  --bootstrap_kind cluster_record \
  --bootstrap_cluster_col record \
  --budget_fracs 0.005,0.01,0.02,0.05,0.10 \
  --seed ${SEED}
```

**Key outputs (per seed):**
- `all_cohorts__auprc.csv` (per-score AUPRC)
- `uncertain__budget_curve.csv`, `highconf_accept_p95__budget_curve.csv`
- `fig/pr__uncertain.png`, `fig/pr__highconf_accept_p95.png`
- `fig/capture__*.png`, `fig/risk__*.png`
- additional bootstrap/sensitivity artifacts in the `_clusterboot` directory

### 5) Aggregate across seeds (paper tables)

After running the pipeline for all seeds (`11111, 22222, 33333, 44444, 55555`), aggregate AUPRC across seeds:

```bash
python - << 'PY'
import pandas as pd
from pathlib import Path
import re

seed_dirs = [
    "results/NP_bspc_seed11111_n1000",
    "results/NP_bspc_seed22222_n1000",
    "results/NP_bspc_seed33333_n1000",
    "results/NP_bspc_seed44444_n1000",
    "results/NP_bspc_seed55555_n1000",
]

dfs = []
for d in seed_dirs:
    d = Path(d)
    f = d / "all_cohorts__auprc.csv"
    df = pd.read_csv(f)

    need = {"cohort","score","auprc"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{f} missing {miss}; has {list(df.columns)}")

    m = re.search(r"seed(\\d+)", d.name)
    seed = m.group(1) if m else d.name
    df["seed"] = seed
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

agg = (all_df.groupby(["cohort","score"])
       .agg(mean_auprc=("auprc","mean"),
            sd_auprc=("auprc","std"),
            n=("auprc","count"),
            mean_delta_vs_best_conf=("delta_vs_best_conf","mean"),
            mean_delta_vs_1margin=("delta_vs_1-margin","mean"))
       .reset_index()
       .sort_values(["cohort","mean_auprc"], ascending=[True, False]))

out = Path("results/NP_bspc_across_seeds_n1000")
out.mkdir(parents=True, exist_ok=True)
agg.to_csv(out / "auprc_across_seeds.csv", index=False)

print("Wrote:", out / "auprc_across_seeds.csv")
print(agg.to_string(index=False))
PY
```

### 6) Optional: small “resource benchmark” (paper appendix)

The paper reports a small compute-cost benchmark using `max_points=30` (single seed).

```bash
python -m post_processing_real.post_processing_real \
  --output_folder results/bspc_seed11111 \
  --out_dir results/bspc_pp_uncertain_bench30_seed11111_v2 \
  --anchors_csv full_test_predictions_ext.csv \
  --selection worst_margin \
  --rank_by margin \
  --max_points 30 \
  --attack_radius 0.05 \
  --seed 11111 \
  --skip_lime --skip_shap --skip_contours --skip_multiplicity \
  --paper

python -m post_processing_real.post_processing_real \
  --output_folder results/bspc_seed11111 \
  --out_dir results/bspc_pp_highconf_accept_p95_bench30_seed11111_v2 \
  --anchors_csv full_test_predictions_ext.csv \
  --filter_accepted_only \
  --filter_pmax_min 0.95 \
  --selection random \
  --max_points 30 \
  --attack_radius 0.15 \
  --robust_steps 80 \
  --robust_num_random 40 \
  --seed 11111 \
  --skip_lime --skip_shap --skip_contours --skip_multiplicity \
  --paper
```

---

## Where to find the “paper numbers” after running

- Across-seed AUPRC table:  
  `results/NP_bspc_across_seeds_n1000/auprc_across_seeds.csv`

- Per-seed AUPRC + PR curves + budget curves:  
  `results/NP_bspc_seedSEED_n1000/` (and `..._clusterboot/` for sensitivity)

- Cohort construction + per-anchor NP outputs (inputs to NP-analysis):  
  `results/bspc_pp_*_seedSEED_n1000/`

---

## Notes on reproducibility

- Randomness is controlled by `--seed` and the code uses record-wise splitting.  
- For determinism, the paper runs used `--num_workers 0` (DataLoader) and fixed seeds.
- Local analysis is compute-heavy; the repository positions NP-onset for **offline / periodic audits**, not real-time routing.

---

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**MIT-BIH Arrhythmia Database**  
ECG recordings are redistributed under the PhysioNet open-access license.  
Please ensure compliance with the original terms: https://physionet.org/content/mitdb/1.0.0/

---

## Contact
For questions or contributions, please open an issue or contact Piotr Migus at migus.piotr@gmail.com.
