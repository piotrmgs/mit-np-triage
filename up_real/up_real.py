# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
Main script for training and evaluating a complex-valued neural network (CVNN) 
on the MIT-BIH Arrhythmia Dataset using K-fold cross-validation across patients.

This script performs the following:
- Loads and preprocesses ECG windows from MIT-BIH records.
- Transforms real-valued time series into complex-valued representations.
- Trains a complex-valued neural network using cross-patient K-fold CV.
- Calibrates the model using temperature scaling.
- Collects predictions for all folds and performs global analyses:
    - Training curves (loss and accuracy)
    - Calibration curve (reliability diagram)
    - Uncertainty histogram
    - Complex PCA scatter plot
    - Ablation support
- Extracts and saves uncertain predictions to CSV.
- Optionally retrains the model on the full dataset and saves the best weights.

Key Features:
-------------
- Complex feature extraction using Hilbert-based analytics or PCA.
- Model architecture: SimpleComplexNet (custom CVNN)
- Cross-validation ensures generalization across patients.
- Visual tools for model interpretability and calibration evaluation.
- CLI arguments for easy experiment control.
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import logging
import pickle
import csv
import psutil
import json
import platform
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
import time
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy import stats


# ───────────────────────────────────────────────────────────
#  Project imports
# ───────────────────────────────────────────────────────────
from src.find_up_synthetic import (
    SimpleComplexNet,
    complex_modulus_to_logits,
    find_uncertain_points,
)
from mit_bih_pre.pre_pro import load_mitbih_data

from src.find_up_real import (
    prepare_complex_input,
    create_dataloaders,
    create_train_val_test_loaders,
    train_model as train_real,
    save_plots,
    tune_temperature,
    fit_isotonic_on_val,          
    save_confusion_roc,
    expected_calibration_error,
    save_overall_history,
    save_calibration_curve,
    save_uncertainty_hist,
    save_complex_pca_scatter,
    save_feature_embedding_2d,    
    select_thresholds_budget_count,
    save_ablation_barplot,
    sensitivity_analysis,
    save_sensitivity_heatmaps,
    save_capture_abstain_curve,   
    negative_log_likelihood,
    brier_score,
    mean_ci,
    seed_everything,
    WINDOW_SIZE,
    PRE_SAMPLES,
    FS,
    get_calibrator,
)


# ───────────────────────────────────────────────────────────


def parse_args():
    """Parse CLI arguments controlling data paths, training hyperparameters, CV, and analysis options."""
    parser = argparse.ArgumentParser(
        description="Complex-Valued NN on MIT-BIH with visualizations"
    )
    parser.add_argument("--data_folder", type=str, default="mit-bih")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--output_folder",
        type=str,
        default=script_dir,
        help="Where to save models and figures",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5001)
    parser.add_argument("--cpu", action="store_true")

    # ▼▼▼ Grid-based (tau/delta) sensitivity analysis + heatmaps — ENABLED by default ▼▼▼
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        default=True,
        help="Enable tau/delta grid analysis (ON by default)."
    )
    # Optional switch to disable the analysis if needed:
    parser.add_argument(
        "--no-sensitivity",
        dest="sensitivity",
        action="store_false",
        help="Disable tau/delta grid analysis."
    )

    parser.add_argument("--capture_target", type=float, default=0.80,
                        help="Target error-capture for mode='capture' (default 0.80).")
    parser.add_argument("--select_mode", type=str, default="budget",
                        choices=["capture","budget","risk","knee"],
                        help="Selection criterion for (tau, delta); default 'budget'.")
    parser.add_argument("--max_abstain", type=float, default=0.20,
                        help="Review budget (fraction) for mode='budget' (default 0.20).")
    parser.add_argument("--target_risk", type=float, default=None,
                        help="Target risk among accepted for mode='risk' (optional).")

    parser.add_argument("--review_budget", type=int, default=10,
                        help="Exact-count review budget on VAL when selecting (tau, delta).")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Global seed for full reproducibility")
    parser.add_argument("--calibration", type=str, default="temperature",
                        choices=["temperature", "isotonic", "none"],
                        help="Calibration method fitted on VALIDATION (default: temperature).")
    parser.add_argument(
        "--calibs",
        type=str,
        default="temperature,isotonic,platt,beta,vector,none",
        help=("Comma-separated list of calibration methods to evaluate in one run. "
              "Supported: temperature,isotonic,platt,beta,vector,none")
    )


    parser.add_argument("--budget_fracs", type=str, default="",
                        help="Optional comma-separated list of fractional review budgets, e.g. '0.005,0.02'.")
    
    parser.add_argument(
        "--taus",
        type=str,
        default="",
        help=("Optional explicit tau grid as comma-separated floats, e.g. "
              "'0.50,0.55,0.60,0.70,0.80,0.90,0.95,0.97,0.99'. "
              "If empty, a strong default non-uniform grid is used.")
    )
    parser.add_argument(
        "--deltas",
        type=str,
        default="",
        help=("Optional explicit delta grid as comma-separated floats, e.g. "
              "'0.00,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.60'. "
              "If empty, a strong default non-uniform grid is used.")
    )
    
    
    
    parser.add_argument("--embed_method", type=str, default="tsne",
                        choices=["tsne", "umap"],
                        help="2D projection method for penultimate features (default: tsne).")


    # ---- Paper-grade calibration metric config ----
    parser.add_argument(
        "--ece_bins",
        type=int,
        default=15,
        help="Number of bins for ECE (default 15)."
    )
    parser.add_argument(
        "--ece_strategy",
        type=str,
        default="quantile",
        choices=["quantile", "uniform"],
        help="ECE binning strategy (default quantile = equal-frequency)."
    )

    # ---- FULL model split (patient-wise, record-level) ----
    parser.add_argument(
        "--full_test_fold",
        type=int,
        default=1,
        help="Which fold (1..K) to use as TEST records for the full-model retrain (patient-wise)."
    )
    parser.add_argument(
        "--full_val_fold",
        type=int,
        default=2,
        help="Which fold (1..K) to use as VAL records for the full-model retrain (patient-wise)."
    )
    parser.add_argument(
        "--save_full_predictions",
        action="store_true",
        default=True,
        help="Save full-model VAL/TEST predictions to CSV (ON by default)."
    )
    parser.add_argument(
        "--no-save_full_predictions",
        dest="save_full_predictions",
        action="store_false",
        help="Disable saving full-model VAL/TEST predictions CSVs."
    )


    return parser.parse_args()



record_names = [
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
        "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
        "222", "223", "228", "230", "231", "232", "233", "234",
    ]

def _parse_float_csv(s: str):
    """Parse comma-separated floats safely (empty -> [])."""
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _default_tau_delta_grid():
    """
    Strong default grid:
    - tau: coarse in [0.50..0.90], fine in [0.90..0.95]
    - delta: dense near 0 (most action happens there), then medium, then coarse tail
    """
    taus = np.unique(np.concatenate([
        np.linspace(0.50, 0.90, 9),     # 0.50,0.55,...,0.90
        np.linspace(0.90, 0.95, 6),     # 0.90,0.91,...,0.95
    ])).astype(float)

    deltas = np.unique(np.concatenate([
        np.linspace(0.00, 0.10, 11),    # 0.00..0.10 step 0.01
        np.linspace(0.12, 0.30, 10),    # 0.12..0.30
        np.linspace(0.35, 0.60, 6),     # 0.35..0.60
    ])).astype(float)

    return taus, deltas



def _safe_write_json(path: str, payload: Dict):
    """Write JSON with a hard fallback (never crash the run for metadata)."""
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
    except Exception as e:
        logging.warning("Could not write json=%s: %s", path, e)


def _mean_ci_t(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    """Student-t CI half-width (matches paper-style CI)."""
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    if n == 1:
        return m, 0.0
    sd = float(np.std(vals, ddof=1))
    tcrit = float(stats.t.ppf(1.0 - alpha/2.0, df=n-1))
    hw = tcrit * sd / (n ** 0.5)
    return m, float(hw)


def _ece_binary(y_true: np.ndarray, y_prob_pos: np.ndarray, n_bins: int = 15, strategy: str = "quantile") -> float:
    """
    Binary ECE for P(y=1):
      ECE = sum_b | mean(y) - mean(p) | * (n_b / n)
    with either uniform or quantile (equal-frequency) bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob_pos = np.asarray(y_prob_pos, dtype=float)
    n = int(y_true.size)
    if n == 0:
        return float("nan")

    p = np.clip(y_prob_pos, 0.0, 1.0)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # de-duplicate / enforce strictly increasing edges
        eps = 1e-12
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = min(1.0, edges[i-1] + eps)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        acc_b = float(y_true[mask].mean())
        conf_b = float(p[mask].mean())
        ece += abs(acc_b - conf_b) * (nb / n)
    return float(ece)


def _binary_clf_metrics(y_true: List[int], y_pred: List[int], y_prob_pos: List[float]) -> Dict[str, float]:
    """Common paper metrics for binary arrhythmia detection."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    ps = np.asarray(y_prob_pos, dtype=float)
    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(yt, yp)) if yt.size else float("nan")
    out["bacc"] = float(balanced_accuracy_score(yt, yp)) if yt.size else float("nan")
    out["f1"] = float(f1_score(yt, yp, zero_division=0)) if yt.size else float("nan")
    out["precision"] = float(precision_score(yt, yp, zero_division=0)) if yt.size else float("nan")
    out["recall"] = float(recall_score(yt, yp, zero_division=0)) if yt.size else float("nan")
    # specificity from confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    except Exception:
        out["specificity"] = float("nan")

    # AUROC/AUPRC can fail if only one class in yt
    try:
        out["auroc"] = float(roc_auc_score(yt, ps))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(yt, ps))
    except Exception:
        out["auprc"] = float("nan")
    return out


def _collect_probs_on_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    calibration: str,
    T: Optional[torch.Tensor],
    iso_cal: Optional[callable],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (y_true, probs_raw, probs_cal, X_scaled) in the *dataset order*.
    probs_* are numpy arrays of shape [N,2].
    """
    ys, pr_list, pc_list, xs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xs.append(xb.cpu().numpy())
            ys.append(yb.cpu().numpy())

            xb_d = xb.to(device)
            logits = complex_modulus_to_logits(model(xb_d))
            probs_raw = nn.Softmax(dim=1)(logits)

            if calibration == "temperature" and T is not None:
                probs_cal = nn.Softmax(dim=1)(logits / T.to(device))
            elif calibration == "isotonic" and iso_cal is not None:
                probs_cal = torch.from_numpy(iso_cal(probs_raw.cpu().numpy())).to(dtype=torch.float32)
            else:
                probs_cal = probs_raw

            pr_list.append(probs_raw.cpu().numpy())
            pc_list.append(probs_cal.cpu().numpy())

    y = np.concatenate(ys, axis=0).astype(int)
    probs_r = np.concatenate(pr_list, axis=0).astype(float)
    probs_c = np.concatenate(pc_list, axis=0).astype(float)
    X = np.concatenate(xs, axis=0).astype(float)
    return y, probs_r, probs_c, X



def _selective_metrics_binary(y_true: np.ndarray, probs: np.ndarray, tau: float, delta: float):
    """
    Compute selective prediction / triage metrics on a binary task.

    Decision rule:
      ACCEPT (auto) iff pmax >= tau AND margin >= delta
      REVIEW otherwise.

    Returns:
      coverage, abstain, risk_accept, capture, precision_review,
      plus clinically-relevant auto FN/FP rates wrt the *whole* population.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)

    n_total = int(y_true.size)
    if n_total == 0:
        return {}

    pred = probs.argmax(axis=1)
    pmax = probs.max(axis=1)

    # Binary margin = |p1 - p0|
    if probs.shape[1] == 2:
        margin = np.abs(probs[:, 1] - probs[:, 0])
    else:
        # General fallback: top-2 difference
        part = np.partition(probs, -2, axis=1)[:, -2:]
        margin = np.abs(part[:, 1] - part[:, 0])

    accept = (pmax >= float(tau)) & (margin >= float(delta))
    review = ~accept

    err = (pred != y_true)
    n_accept = int(accept.sum())
    n_review = n_total - n_accept

    coverage = n_accept / n_total
    abstain = 1.0 - coverage

    risk_accept = float(err[accept].mean()) if n_accept > 0 else float("nan")

    n_err = int(err.sum())
    capture = float((err & review).sum() / n_err) if n_err > 0 else float("nan")
    precision_review = float((err & review).sum() / n_review) if n_review > 0 else float("nan")

    # Clinical-style: how many positives/negatives would be *mis-handled automatically*
    pos = (y_true == 1)
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    fn_auto = int((pos & accept & (pred == 0)).sum())
    fp_auto = int((neg & accept & (pred == 1)).sum())
    tp_auto = int((pos & accept & (pred == 1)).sum())
    tn_auto = int((neg & accept & (pred == 0)).sum())

    fn_auto_rate = (fn_auto / n_pos) if n_pos > 0 else float("nan")  # missed PVC among ALL PVC
    fp_auto_rate = (fp_auto / n_neg) if n_neg > 0 else float("nan")  # false alarms among ALL normal
    tp_auto_rate = (tp_auto / n_pos) if n_pos > 0 else float("nan")
    tn_auto_rate = (tn_auto / n_neg) if n_neg > 0 else float("nan")

    mean_pmax_accept = float(pmax[accept].mean()) if n_accept > 0 else float("nan")
    mean_margin_accept = float(margin[accept].mean()) if n_accept > 0 else float("nan")

    return {
        "n_total": n_total,
        "n_accept": n_accept,
        "n_review": n_review,
        "coverage": coverage,
        "abstain": abstain,
        "risk_accept": risk_accept,
        "capture": capture,
        "precision_review": precision_review,
        "fn_auto_rate": fn_auto_rate,
        "fp_auto_rate": fp_auto_rate,
        "tp_auto_rate": tp_auto_rate,
        "tn_auto_rate": tn_auto_rate,
        "mean_pmax_accept": mean_pmax_accept,
        "mean_margin_accept": mean_margin_accept,
    }


# ───────────────────────────────────────────────────────────
#  main
# ───────────────────────────────────────────────────────────
def main():
    """Entry point: run cross-patient K-fold CV, collect metrics/plots, then retrain on full data."""
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Hard reset files that are appended fold-by-fold, to avoid mixing runs.
    multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
    if os.path.exists(multi_path):
        os.remove(multi_path)

    sum_multi_path = os.path.join(args.output_folder, "cv_metrics_summary_multi.csv")
    if os.path.exists(sum_multi_path):
        os.remove(sum_multi_path)

    
    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, "run.log"), mode="w"),
            logging.StreamHandler(),
        ],
        force=True,  # ensure reconfiguration even if logging was set elsewhere
    )


    logger = logging.getLogger()

    # Reproducibility: seed Python/NumPy/PyTorch (CPU/CUDA)
    seed_everything(args.seed)


    # Determinism & numerical stability (match post_processing_real)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s | seed=%d", device, args.seed)
    
    # Save run metadata (useful for reproducibility in rebuttals/appendices)
    meta = {
        "seed": args.seed,
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "args": vars(args),
        "record_names": record_names,
    }
    _safe_write_json(os.path.join(args.output_folder, "run_meta.json"), meta)
    _safe_write_json(os.path.join(args.output_folder, "run_args.json"), vars(args))

    # ── Global collectors across folds ─────────────────────────────────    
    histories_all_folds = []
    # Per-fold predictions (for plots)
    y_true_all, y_pred_all = [], []
    y_prob_all_pos = []   # P(y=1) — for calibration curve
    y_conf_all_max = []   # max softmax — for uncertainty histogram
    y_conf_all = []       # (kept for compatibility, not used)
    all_uncertain = []

    # RAW baseline collectors for calibration (no temperature scaling)
    y_true_all_raw = []
    y_prob_all_pos_raw = []
    y_pred_all_raw = []
    y_conf_all_max_raw = []

    y_margin_all = []        
    pred_rows_all = []        

    # K-fold CV over patients/records
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Per-fold metrics (RAW vs CAL)
    ece_raw_folds, ece_ts_folds = [], []
    nll_raw_folds, nll_ts_folds = [], []
    br_raw_folds,  br_ts_folds  = [], []
    acc_ts_folds, auc_ts_folds  = [], []

    acc_raw_folds, auc_raw_folds = [], []

    # Classification metrics (paper-friendly)
    f1_raw_folds, f1_ts_folds = [], []
    bacc_raw_folds, bacc_ts_folds = [], []
    ap_raw_folds, ap_ts_folds = [], []
    prec_raw_folds, prec_ts_folds = [], []
    rec_raw_folds, rec_ts_folds = [], []
    spec_raw_folds, spec_ts_folds = [], []

    # Fold manifests for reproducibility
    fold_manifest = []
    fold_counts = []

    # CV triage (pick tau/delta on VAL; evaluate on TEST)
    triage_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(record_names), 1):
        train_recs = [record_names[i] for i in train_idx]
        test_recs = [record_names[i] for i in test_idx]
        logger.info("Fold %d  train=%s  test=%s", fold, train_recs, test_recs)
        fold_manifest.append({"fold": fold, "train_records": train_recs, "test_records": test_recs})
 

        # ── Load raw (real-valued) windows ─────────────────
        X_train, y_train = load_mitbih_data(
            args.data_folder, train_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )
        X_test, y_test = load_mitbih_data(
            args.data_folder, test_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )

        # Basic split stats for paper appendix
        bc_tr = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
        bc_te = np.bincount(np.asarray(y_test, dtype=int), minlength=2)
        fold_counts.append({
            "fold": fold,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_n0": int(bc_tr[0]), "train_n1": int(bc_tr[1]),
            "test_n0": int(bc_te[0]),  "test_n1": int(bc_te[1]),
        })

        # ───────────────────────────────────────────────────
        #  CVNN + complex_stats (main article path)
        # ───────────────────────────────────────────────────
        X_tr_s = prepare_complex_input(X_train, method='complex_stats')
        X_te_s = prepare_complex_input(X_test,  method='complex_stats')

        # Split TRAIN → TRAIN/VAL (stratified)
        X_tr_s_tr, X_tr_s_val, y_train_tr, y_train_val = train_test_split(
            X_tr_s, y_train, test_size=0.2, stratify=y_train, random_state=args.seed + 1000 + fold
        )

        # Build loaders: train/val/test (scaler fit on TRAIN only)
        tr_ld, val_ld, te_ld, scaler = create_train_val_test_loaders(
            X_tr_s_tr, y_train_tr,
            X_tr_s_val, y_train_val,
            X_te_s,     y_test,
            batch_size=args.batch_size,
            seed=args.seed
        )
        # Zapisz scaler dla tego folda (przyda się w postprocessingu)
        with open(os.path.join(args.output_folder, f"scaler_fold{fold}.pkl"), "wb") as f:
            pickle.dump(scaler, f)


        # Model
        model_s = SimpleComplexNet(
            in_features=X_tr_s.shape[1] // 2,
            hidden_features=64,
            out_features=2,
            bias=0.1,
        )
        t0 = time.perf_counter()
        history, best_s = train_real(
            model_s, tr_ld, val_ld, epochs=args.epochs, lr=args.lr, device=device
        )
        train_time_s = (time.perf_counter() - t0) / len(history["train_loss"])
        histories_all_folds.append(history)
        
        save_plots(history, args.output_folder, fold)

        # Resource snapshot (CPU RSS, optional GPU peak) per fold
        try:
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024**2)
            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                torch.cuda.reset_peak_memory_stats(device)
                logging.info(f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | CPU_RSS={rss_mb:.1f}MB | GPU_peak={peak_mb:.1f}MB")
            else:
                logging.info(f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | CPU_RSS={rss_mb:.1f}MB")
        except Exception as _e:
            logging.warning(f"[Fold {fold}] memory logging failed: {_e}")

        # Load best weights (by val acc)
        model_s.load_state_dict(best_s)

        # --- Choose calibration on VALIDATION (temperature / isotonic / none) ---
        T_fold = None
        iso_cal = None
        is_binary = False
        if args.calibration == "temperature":
            T_fold = tune_temperature(model_s, val_ld, device=device)
            torch.save(T_fold, os.path.join(args.output_folder, f"T_calib_fold{fold}.pt"))
            logging.info(f"[Fold {fold}] Learned temperature T={T_fold.item():.3f}")
        elif args.calibration == "isotonic":
            iso_cal, is_binary = fit_isotonic_on_val(model_s, val_ld, device=device)
            if not is_binary:
                logging.warning("[Fold %d] Isotonic calibration skipped (multi-class).", fold)
        else:
            logging.info("[Fold %d] No calibration ('none').", fold)


        # --- Per-fold evaluation: RAW (no T) vs TS (with T) on TEST ---
        model_s.eval()
        y_true_raw, y_prob_pos_raw, y_max_raw = [], [], []
        y_true_ts,  y_prob_pos_ts,  y_max_ts  = [], [], []
        y_pred_ts = []
        y_pred_raw = []
        probs_ts_all = []
        probs_raw_all = []
        yb_all = []


        
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                logits = complex_modulus_to_logits(model_s(xb))

                # RAW probabilities (always computed)
                probs_raw = nn.Softmax(dim=1)(logits)
                probs_raw_all.append(probs_raw.cpu())
                y_pred_raw.extend(probs_raw.argmax(dim=1).cpu().tolist())

                # Calibrated probabilities according to --calibration
                if args.calibration == "temperature" and T_fold is not None:
                    probs_ts = nn.Softmax(dim=1)(logits / T_fold.to(device))
                elif args.calibration == "isotonic" and iso_cal is not None:
                    probs_ts = torch.from_numpy(iso_cal(probs_raw.cpu().numpy())).to(dtype=torch.float32)
                else:
                    probs_ts = probs_raw


                probs_ts_all.append(probs_ts.cpu())
                yb_all.append(yb.cpu())

                # RAW collectors
                y_true_raw.extend(yb.tolist())
                y_prob_pos_raw.extend(probs_raw[:, 1].cpu().tolist())
                y_max_raw.extend(probs_raw.max(dim=1).values.cpu().tolist())

                # CALIBRATED collectors (keep *_ts variable names to avoid touching the rest)
                y_true_ts.extend(yb.tolist())
                y_prob_pos_ts.extend(probs_ts[:, 1].cpu().tolist())
                y_max_ts.extend(probs_ts.max(dim=1).values.cpu().tolist())
                y_pred_ts.extend(probs_ts.argmax(dim=1).cpu().tolist())

         
        # Acc/AUC (CAL) for this fold
        acc_ts = float((np.array(y_true_ts) == np.array(y_pred_ts)).mean())
        try:
            auc_ts = float(roc_auc_score(np.array(y_true_ts), np.array(y_prob_pos_ts)))
        except Exception:
            auc_ts = float('nan')
        acc_ts_folds.append(acc_ts)
        auc_ts_folds.append(auc_ts)

        # Acc/AUC (RAW) for this fold
        acc_raw = float((np.array(y_true_raw) == np.array(y_pred_raw)).mean())
        try:
            auc_raw = float(roc_auc_score(np.array(y_true_raw), np.array(y_prob_pos_raw)))
        except Exception:
            auc_raw = float("nan")
        acc_raw_folds.append(acc_raw)
        auc_raw_folds.append(auc_raw)

        # Classification metrics (RAW / CAL)
        m_raw = _binary_clf_metrics(y_true_raw, y_pred_raw, y_prob_pos_raw)
        m_ts  = _binary_clf_metrics(y_true_ts,  y_pred_ts,  y_prob_pos_ts)
        f1_raw_folds.append(m_raw["f1"]); f1_ts_folds.append(m_ts["f1"])
        bacc_raw_folds.append(m_raw["bacc"]); bacc_ts_folds.append(m_ts["bacc"])
        ap_raw_folds.append(m_raw["auprc"]); ap_ts_folds.append(m_ts["auprc"])
        prec_raw_folds.append(m_raw["precision"]); prec_ts_folds.append(m_ts["precision"])
        rec_raw_folds.append(m_raw["recall"]); rec_ts_folds.append(m_ts["recall"])
        spec_raw_folds.append(m_raw["specificity"]); spec_ts_folds.append(m_ts["specificity"])
        
        roc_auc = save_confusion_roc(y_true_ts, y_pred_ts, y_prob_pos_ts, args.output_folder, fold)
        logging.info(f"[Fold {fold}] ROC AUC (TS) = {roc_auc:.4f}")


        # --- Per-sample rows for CSV + margin (TS) ---
        probs_ts_all  = torch.cat(probs_ts_all, dim=0)   # [N_test, 2]
        probs_raw_all = torch.cat(probs_raw_all, dim=0)  # [N_test, 2]
        yb_all        = torch.cat(yb_all, dim=0)         # [N_test]


        top2 = torch.topk(probs_ts_all, k=2, dim=1).values
        margin_batch = (top2[:, 0] - top2[:, 1]).abs().tolist()
        y_margin_all.extend(margin_batch)

        pmax_batch  = probs_ts_all.max(dim=1).values.tolist()
        p1_batch    = probs_ts_all[:, 0].tolist()
        p2_batch    = probs_ts_all[:, 1].tolist()
        pred_batch  = probs_ts_all.argmax(dim=1).tolist()
        true_batch  = yb_all.tolist()

        # RAW derived (for free)
        top2_raw = torch.topk(probs_raw_all, k=2, dim=1).values
        margin_raw_batch = (top2_raw[:, 0] - top2_raw[:, 1]).abs().tolist()
        pmax_raw_batch   = probs_raw_all.max(dim=1).values.tolist()
        p1_raw_batch     = probs_raw_all[:, 0].tolist()
        p2_raw_batch     = probs_raw_all[:, 1].tolist()
        pred_raw_batch   = probs_raw_all.argmax(dim=1).tolist()

        cal_suffix_fold = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "NONE")
        T_fold_val = float(T_fold.item()) if T_fold is not None else float("nan")

        base_row_id = len(pred_rows_all)
        for i in range(len(true_batch)):
            pred_rows_all.append({
                "fold": fold,
                "row_global": base_row_id + i,
                "cal_method": cal_suffix_fold,
                "T_fold": T_fold_val,

                "true": int(true_batch[i]),

                # CAL (kept for compatibility with your downstream naming)
                "pred_TS": int(pred_batch[i]),
                "p1_TS": float(p1_batch[i]),
                "p2_TS": float(p2_batch[i]),
                "pmax_TS": float(pmax_batch[i]),
                "margin_TS": float(margin_batch[i]),

                # RAW (new)
                "pred_RAW": int(pred_raw_batch[i]),
                "p1_RAW": float(p1_raw_batch[i]),
                "p2_RAW": float(p2_raw_batch[i]),
                "pmax_RAW": float(pmax_raw_batch[i]),
                "margin_RAW": float(margin_raw_batch[i]),
            })



        # Global collectors (TS for figures)
        y_true_all.extend(y_true_ts)
        y_pred_all.extend(y_pred_ts)
        y_prob_all_pos.extend(y_prob_pos_ts)
        y_conf_all_max.extend(y_max_ts)
        
        # Global collectors (RAW for baseline reliability curve)
        y_true_all_raw.extend(y_true_raw)
        y_prob_all_pos_raw.extend(y_prob_pos_raw)
        y_pred_all_raw.extend(y_pred_raw)
        y_conf_all_max_raw.extend(y_max_raw)

        # Per-fold metrics (RAW vs TS)
        ece_raw = _ece_binary(np.array(y_true_raw), np.array(y_prob_pos_raw),
                              n_bins=args.ece_bins, strategy=args.ece_strategy)
        ece_ts  = _ece_binary(np.array(y_true_ts),  np.array(y_prob_pos_ts),
                              n_bins=args.ece_bins, strategy=args.ece_strategy)
        nll_raw = negative_log_likelihood(np.array(y_true_raw), np.array(y_prob_pos_raw))
        nll_ts  = negative_log_likelihood(np.array(y_true_ts),  np.array(y_prob_pos_ts))
        br_raw  = brier_score(np.array(y_true_raw), np.array(y_prob_pos_raw))
        br_ts   = brier_score(np.array(y_true_ts),  np.array(y_prob_pos_ts))

        ece_raw_folds.append(ece_raw); ece_ts_folds.append(ece_ts)
        nll_raw_folds.append(nll_raw); nll_ts_folds.append(nll_ts)
        br_raw_folds.append(br_raw);   br_ts_folds.append(br_ts)

        cal_suffix = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "CAL")
        logger.info(
            "[Fold %d] epoch_avg=%.2fs | params=%d | "
            "ECE(%s,%db) raw=%.4f→%.4f | NLL raw=%.4f→%.4f | Brier raw=%.4f→%.4f | "
            "Acc raw=%.4f→%.4f | AUROC raw=%.4f→%.4f | F1 raw=%.4f→%.4f",
            fold, train_time_s, sum(p.numel() for p in model_s.parameters()),
            args.ece_strategy, args.ece_bins,
            ece_raw, ece_ts,
            nll_raw, nll_ts, br_raw, br_ts,
            acc_raw, acc_ts, auc_raw, auc_ts,
            m_raw["f1"], m_ts["f1"]
        )


        # ---- CV triage: choose (tau,delta) on VAL, evaluate on TEST ----
        try:
            yv, pr_v, pc_v, Xv = _collect_probs_on_loader(model_s, val_ld, device, args.calibration, T_fold, iso_cal)
            budget_count = int(min(args.review_budget, len(yv)))
            chosen_fold = select_thresholds_budget_count(y_true=yv, probs=pc_v, X=Xv, budget_count=budget_count)
            tau_f = float(chosen_fold["tau"]); delta_f = float(chosen_fold["delta"])
            tri_val = _selective_metrics_binary(yv, pc_v, tau=tau_f, delta=delta_f)
            tri_te  = _selective_metrics_binary(np.asarray(y_true_ts, dtype=int),
                                                probs_ts_all.numpy(), tau=tau_f, delta=delta_f)
            triage_rows.append({
                "fold": fold,
                "budget_count_val": budget_count,
                "tau": tau_f,
                "delta": delta_f,
                "val_abstain": float(chosen_fold.get("abstain", np.nan)),
                "val_capture": float(chosen_fold.get("capture", np.nan)),
                "val_risk_accept": float(chosen_fold.get("risk_accept", np.nan)),
                "test_coverage": float(tri_te.get("coverage", np.nan)),
                "test_abstain": float(tri_te.get("abstain", np.nan)),
                "test_risk_accept": float(tri_te.get("risk_accept", np.nan)),
                "test_capture": float(tri_te.get("capture", np.nan)),
                "test_precision_review": float(tri_te.get("precision_review", np.nan)),
                "test_fn_auto_rate": float(tri_te.get("fn_auto_rate", np.nan)),
                "test_fp_auto_rate": float(tri_te.get("fp_auto_rate", np.nan)),
            })
        except Exception as e:
            logging.warning("[Fold %d] triage selection/eval failed: %s", fold, e)

        # ========== NEW: multi-calibration block (runs in addition to RAW vs selected --calibration) ==========
        methods = [m.strip().lower() for m in args.calibs.split(",") if m.strip()]

        # 1) Cache TEST logits once (to avoid re-running the model per method)
        logits_list_mc, y_list_mc = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                lo = complex_modulus_to_logits(model_s(xb))
                logits_list_mc.append(lo.cpu().numpy())
                y_list_mc.extend(yb.cpu().numpy().tolist())
        logits_np_mc = np.vstack(logits_list_mc)          # [N_test, C]
        y_np_mc = np.asarray(y_list_mc, dtype=int)        # [N_test]

        # Softmax for RAW (shared by ISO/BETA)
        z_mc = logits_np_mc - logits_np_mc.max(axis=1, keepdims=True)
        expz_mc = np.exp(z_mc)
        probs_raw_np_mc = expz_mc / expz_mc.sum(axis=1, keepdims=True)

        fold_rows = []  # rows for this fold across all methods

        # 2) Fit calibrator on VALIDATION per method, then apply to TEST
        for method in methods:
            apply_fn, tag = get_calibrator(model_s, val_ld, method=method, device=device)
            if apply_fn is None:
                logging.info(f"[Fold {fold}] Skipping method={method} (not applicable).")
                continue

            # Apply on TEST
            if method in ("temperature", "vector", "platt"):
                probs_np = apply_fn(logits_np_mc)                # expects logits
            elif method in ("isotonic", "beta"):
                probs_np = apply_fn(probs_np=probs_raw_np_mc)    # expects probs
            elif method == "none":
                probs_np = probs_raw_np_mc
            else:
                probs_np = probs_raw_np_mc  # safe fallback

            # Metrics (binary-friendly; AUC/ECE/NLL/Brier only if C==2)
            preds = probs_np.argmax(axis=1)
            acc   = float((preds == y_np_mc).mean())
            if probs_np.shape[1] == 2:
                try:
                    auc   = float(roc_auc_score(y_np_mc, probs_np[:, 1]))
                except Exception:
                    auc = float("nan")
                ece   = _ece_binary(y_np_mc, probs_np[:, 1], n_bins=args.ece_bins, strategy=args.ece_strategy)
                nll   = negative_log_likelihood(y_np_mc, probs_np[:, 1])
                brier = brier_score(y_np_mc, probs_np[:, 1])
            else:
                auc = ece = nll = brier = float("nan")

            fold_rows.append({
                "fold": fold, "method": method.upper(), "tag": tag,
                "ECE": ece, "NLL": nll, "Brier": brier, "Acc": acc, "AUC": auc
            })

        # 3) Append to a fold-wise CSV (one file for the whole run)
        multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
        write_header = not os.path.exists(multi_path)
        with open(multi_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold","method","tag","ECE","NLL","Brier","Acc","AUC"])
            if write_header:
                w.writeheader()
            w.writerows(fold_rows)
        logging.info(f"[Fold {fold}] Wrote multi-calibration rows to {multi_path}")
        # ========== /NEW ==========


    if pred_rows_all:
        pred_csv_path = os.path.join(args.output_folder, "predictions_all_folds.csv")
        with open(pred_csv_path, "w", newline="") as fcsv:
            cw = csv.DictWriter(fcsv, fieldnames=list(pred_rows_all[0].keys()))
            cw.writeheader()
            cw.writerows(pred_rows_all)
        logging.info(f"Saved predictions with margins to {pred_csv_path}")

    # --- Save per-fold metrics for transparency (place right after the K-Fold loop) ---
    per_fold_path = os.path.join(args.output_folder, "cv_metrics_per_fold.csv")
    with open(per_fold_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fold",
            f"ECE_raw({args.ece_strategy},{args.ece_bins})", f"ECE_cal({args.ece_strategy},{args.ece_bins})",
            "NLL_raw","NLL_cal","Brier_raw","Brier_cal",
            "Acc_raw","Acc_cal","AUROC_raw","AUROC_cal",
            "F1_raw","F1_cal","BalAcc_raw","BalAcc_cal",
            "AUPRC_raw","AUPRC_cal",
            "Precision_raw","Precision_cal",
            "Recall_raw","Recall_cal",
            "Specificity_raw","Specificity_cal",
        ])
        for i in range(len(ece_raw_folds)):
            w.writerow([
                i + 1,
                f"{ece_raw_folds[i]:.6f}", f"{ece_ts_folds[i]:.6f}",
                f"{nll_raw_folds[i]:.6f}", f"{nll_ts_folds[i]:.6f}",
                f"{br_raw_folds[i]:.6f}",  f"{br_ts_folds[i]:.6f}",
                f"{acc_raw_folds[i]:.6f}", f"{acc_ts_folds[i]:.6f}",
                f"{auc_raw_folds[i]:.6f}", f"{auc_ts_folds[i]:.6f}",
                f"{f1_raw_folds[i]:.6f}",  f"{f1_ts_folds[i]:.6f}",
                f"{bacc_raw_folds[i]:.6f}", f"{bacc_ts_folds[i]:.6f}",
                f"{ap_raw_folds[i]:.6f}",  f"{ap_ts_folds[i]:.6f}",
                f"{prec_raw_folds[i]:.6f}", f"{prec_ts_folds[i]:.6f}",
                f"{rec_raw_folds[i]:.6f}",  f"{rec_ts_folds[i]:.6f}",
                f"{spec_raw_folds[i]:.6f}", f"{spec_ts_folds[i]:.6f}",
            ])

    logger.info("Saved per-fold CV metrics to %s", per_fold_path)


    # Save fold manifest + basic class counts
    _safe_write_json(os.path.join(args.output_folder, "cv_folds.json"), {"folds": fold_manifest})
    counts_path = os.path.join(args.output_folder, "cv_fold_counts.csv")
    with open(counts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_counts[0].keys()) if fold_counts else [])
        if fold_counts:
            w.writeheader()
            w.writerows(fold_counts)
    logger.info("Saved CV fold class counts to %s", counts_path)

    # Save CV triage per-fold (if computed)
    if triage_rows:
        tri_path = os.path.join(args.output_folder, "cv_triage_per_fold.csv")
        with open(tri_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(triage_rows[0].keys()))
            w.writeheader()
            w.writerows(triage_rows)
        logger.info("Saved CV triage per-fold to %s", tri_path)

        # Summary (t-CI) for key triage metrics
        def _summ(metric: str) -> Tuple[float, float]:
            return _mean_ci_t([r.get(metric, float("nan")) for r in triage_rows])
        tri_sum_path = os.path.join(args.output_folder, "cv_triage_summary.csv")
        with open(tri_sum_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric","Mean","CI95"])
            for metric in ["test_coverage","test_abstain","test_risk_accept","test_capture",
                           "test_precision_review","test_fn_auto_rate","test_fp_auto_rate"]:
                m, ci = _summ(metric)
                w.writerow([metric, f"{m:.6f}", f"{ci:.6f}"])
        logger.info("Saved CV triage summary to %s", tri_sum_path)

    save_overall_history(histories_all_folds, args.output_folder)

    # Save reliability diagrams for baseline (RAW) and calibrated (TS)
    save_calibration_curve(y_true_all_raw, y_prob_all_pos_raw, args.output_folder, suffix="RAW")
    cal_suffix = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "CAL")
    save_calibration_curve(y_true_all, y_prob_all_pos, args.output_folder, suffix=cal_suffix)


    def _row(name, raw_list, ts_list):
        """Helper: summary row with mean, 95% CI half-width, and relative drop from RAW to TS."""
        m_raw, ci_raw = _mean_ci_t(raw_list)
        m_ts,  ci_ts  = _mean_ci_t(ts_list)
        rel_drop = (m_raw - m_ts) / max(m_raw, 1e-12)
        return [name, m_raw, ci_raw, m_ts, ci_ts, rel_drop]

    table = [
        _row("ECE", ece_raw_folds, ece_ts_folds),
        _row("NLL", nll_raw_folds, nll_ts_folds),
        _row("Brier", br_raw_folds, br_ts_folds),
    ]

    with open(os.path.join(args.output_folder, "cv_metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "RAW_mean", "RAW_CI95", "TS_mean", "TS_CI95", "Relative_drop"])
        w.writerows(table)

    logger.info("Saved CV metrics summary with 95%% CI to %s",
                os.path.join(args.output_folder, "cv_metrics_summary.csv"))

    # ========== NEW: global summary for multi-calibration ==========
    multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
    if os.path.exists(multi_path):
        import pandas as pd
        dfm = pd.read_csv(multi_path)
        out_rows = []
        for m in sorted(dfm["method"].unique()):
            sub = dfm[dfm["method"] == m]
            def _ci95(a):
                a = np.asarray([x for x in a if not np.isnan(x)], dtype=float)
                if a.size == 0:
                    return (float("nan"), float("nan"))
                mean = float(a.mean())
                if a.size == 1:
                    return (mean, 0.0)
                sd = float(a.std(ddof=1))
                tcrit = float(stats.t.ppf(0.975, df=a.size - 1))
                hw = tcrit * sd / (a.size ** 0.5)
                return (mean, hw)
            for name in ["ECE","NLL","Brier","Acc","AUC"]:
                mean, hw = _ci95(sub[name])
                out_rows.append([m, name, mean, hw])

        path_sum_multi = os.path.join(args.output_folder, "cv_metrics_summary_multi.csv")
        with open(path_sum_multi, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Method","Metric","Mean","CI95"])
            w.writerows(out_rows)
        logging.info("Saved multi-calibration summary to %s", path_sum_multi)



    # Keep scatter consistent with features used during CV (complex_stats)
    X_full, y_full = load_mitbih_data(args.data_folder, record_names, WINDOW_SIZE, PRE_SAMPLES, FS)
    save_complex_pca_scatter(
        prepare_complex_input(X_full, method='complex_stats'),
        y_full, args.output_folder
    )
    
    # Retrain on full dataset with patient-wise (record-level) train/val/test split (no leakage)
    # (paper protocol emphasizes patient-wise split)  :contentReference[oaicite:3]{index=3}
    logger.info("Retraining FULL model with patient-wise record split…")
    if args.full_test_fold == args.full_val_fold:
        raise ValueError("full_test_fold and full_val_fold must be different.")
    if not (1 <= args.full_test_fold <= args.folds and 1 <= args.full_val_fold <= args.folds):
        raise ValueError("full_test_fold/full_val_fold must be within 1..folds.")

    # Build K partitions of record indices (test_idx of each fold)
    parts = []
    for _, te_idx in kf.split(record_names):
        parts.append(list(te_idx))
    te_set = set(parts[args.full_test_fold - 1])
    va_set = set(parts[args.full_val_fold - 1])
    tr_set = set(range(len(record_names))) - te_set - va_set

    train_recs_full = [record_names[i] for i in sorted(tr_set)]
    val_recs_full   = [record_names[i] for i in sorted(va_set)]
    test_recs_full  = [record_names[i] for i in sorted(te_set)]

    _safe_write_json(os.path.join(args.output_folder, "full_split_records.json"), {
        "folds": args.folds,
        "seed": args.seed,
        "full_test_fold": args.full_test_fold,
        "full_val_fold": args.full_val_fold,
        "train_records": train_recs_full,
        "val_records": val_recs_full,
        "test_records": test_recs_full,
    })

    X_tr_raw, y_tr = load_mitbih_data(args.data_folder, train_recs_full, WINDOW_SIZE, PRE_SAMPLES, FS)
    X_va_raw, y_va = load_mitbih_data(args.data_folder, val_recs_full,   WINDOW_SIZE, PRE_SAMPLES, FS)
    X_te_raw, y_te = load_mitbih_data(args.data_folder, test_recs_full,  WINDOW_SIZE, PRE_SAMPLES, FS)

    X_tr = prepare_complex_input(X_tr_raw, method='complex_stats')
    X_va = prepare_complex_input(X_va_raw, method='complex_stats')
    X_te = prepare_complex_input(X_te_raw, method='complex_stats')
    
    tr_all, va_all, te_all, scaler_full = create_train_val_test_loaders(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        batch_size=args.batch_size,
        seed=args.seed
    )
    with open(os.path.join(args.output_folder, "scaler_full.pkl"), "wb") as f:
        pickle.dump(scaler_full, f)


    model_all = SimpleComplexNet(
        in_features=X_tr.shape[1] // 2, hidden_features=64, out_features=2, bias=0.1
    )
    hist_all, best_all = train_real(model_all, tr_all, va_all, epochs=args.epochs, lr=args.lr, device=device)
    torch.save(best_all, os.path.join(args.output_folder, 'best_model_full.pt'))
    save_plots(hist_all, args.output_folder, 'full')
    logger.info('Saved full-model and plots')

    # Temperature calibration on VALIDATION

    model_all.load_state_dict(best_all)
    T_full = None
    iso_full = None
    is_binary = False
    if args.calibration == "temperature":
        T_full = tune_temperature(model_all, va_all, device=device)
        torch.save(T_full, os.path.join(args.output_folder, 'T_calib.pt'))
        logger.info(f"[INFO] Full-model temperature T={T_full.item():.3f} saved to {args.output_folder}")
    elif args.calibration == "isotonic":
        iso_full, is_binary = fit_isotonic_on_val(model_all, va_all, device=device)
        logger.info("[INFO] Full-model isotonic calibration %s", "enabled" if is_binary else "skipped (multi-class)")
    else:
        logger.info("[INFO] No calibration ('none') on full model.")

    # Collect calibrated probabilities on VALIDATION (needed to pick tau*, delta*)
    yva, pva_raw, pva, Xva = _collect_probs_on_loader(model_all, va_all, device, args.calibration, T_full, iso_full)

    kink_star = float('nan')  # set later if grid was computed
    if args.sensitivity:
        taus_def, deltas_def = _default_tau_delta_grid()
        taus_user = _parse_float_csv(args.taus)
        deltas_user = _parse_float_csv(args.deltas)

        taus = np.array(taus_user, dtype=float) if taus_user else taus_def
        deltas = np.array(deltas_user, dtype=float) if deltas_user else deltas_def

        taus = np.unique(np.clip(taus, 0.0, 0.999999))
        deltas = np.unique(np.clip(deltas, 0.0, 0.999999))

        logging.info(f"[SENS] tau grid: n={len(taus)} min={taus.min():.3f} max={taus.max():.3f}")
        logging.info(f"[SENS] delta grid: n={len(deltas)} min={deltas.min():.3f} max={deltas.max():.3f}")


        sens = sensitivity_analysis(
            y_true=yva, probs=pva, X=Xva,
            taus=taus, deltas=deltas,
            target_capture=args.capture_target,
            max_abstain=args.max_abstain,
            target_risk=args.target_risk,
            mode=args.select_mode
        )
        grid = sens["grid"]

        # "Knee" score proxy: normalized (capture - abstain)
        abst = grid[:, 2]; capt = grid[:, 3]
        a = (abst - abst.min()) / (abst.max() - abst.min() + 1e-12)
        c = (capt - capt.min()) / (capt.max() - capt.min() + 1e-12)
        kink = c - a

        # Save the full grid with knee score
        sens_csv = os.path.join(args.output_folder, "sens_grid.csv")
        with open(sens_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau","delta","abstain","capture","precision","dispersion","risk_accept","kink"])
            for row, k in zip(grid, kink):
                w.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}"] + [f"{x:.6f}" for x in row[2:]] + [f"{float(k):.6f}"])
        logging.info("Saved sensitivity grid (with kink) to %s", sens_csv)
        
        # === NEW: enriched grid with coverage + exact counts + clinically-relevant auto FN/FP ===
        sens_csv_ext = os.path.join(args.output_folder, "sens_grid_ext.csv")
        with open(sens_csv_ext, "w", newline="") as f:
            fieldnames = [
                "tau","delta",
                # original grid metrics (as returned)
                "abstain_grid","capture_grid","precision_grid","dispersion_grid","risk_accept_grid","kink",
                # exact recompute (robust for paper tables)
                "n_total","n_accept","n_review","coverage_exact","abstain_exact",
                "risk_accept_exact","capture_exact","precision_review_exact",
                # clinical-style safety for PVC
                "fn_auto_rate","fp_auto_rate","tp_auto_rate","tn_auto_rate",
                # sanity summaries
                "mean_pmax_accept","mean_margin_accept",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for row, k in zip(grid, kink):
                tau = float(row[0]); delta = float(row[1])

                extra = _selective_metrics_binary(yva, pva, tau=tau, delta=delta)

                # Guard: extra can be {} only if yva empty (should not happen)
                if not extra:
                    continue

                # Compare against grid (optional sanity log)
                try:
                    risk_grid = float(row[6])
                    if (not np.isnan(extra["risk_accept"])) and abs(extra["risk_accept"] - risk_grid) > 1e-6:
                        logging.warning(f"[SENS] risk_accept mismatch tau={tau:.4f} delta={delta:.4f}: "
                                        f"grid={risk_grid:.6f} vs exact={extra['risk_accept']:.6f}")
                except Exception:
                    pass

                w.writerow({
                    "tau": f"{tau:.6f}",
                    "delta": f"{delta:.6f}",
                    "abstain_grid": f"{float(row[2]):.6f}",
                    "capture_grid": f"{float(row[3]):.6f}",
                    "precision_grid": f"{float(row[4]):.6f}",
                    "dispersion_grid": f"{float(row[5]):.6f}",
                    "risk_accept_grid": f"{float(row[6]):.6f}",
                    "kink": f"{float(k):.6f}",

                    "n_total": extra["n_total"],
                    "n_accept": extra["n_accept"],
                    "n_review": extra["n_review"],
                    "coverage_exact": f"{extra['coverage']:.6f}",
                    "abstain_exact": f"{extra['abstain']:.6f}",
                    "risk_accept_exact": f"{extra['risk_accept']:.6f}" if not np.isnan(extra["risk_accept"]) else "",
                    "capture_exact": f"{extra['capture']:.6f}" if not np.isnan(extra["capture"]) else "",
                    "precision_review_exact": f"{extra['precision_review']:.6f}" if not np.isnan(extra["precision_review"]) else "",

                    "fn_auto_rate": f"{extra['fn_auto_rate']:.6f}" if not np.isnan(extra["fn_auto_rate"]) else "",
                    "fp_auto_rate": f"{extra['fp_auto_rate']:.6f}" if not np.isnan(extra["fp_auto_rate"]) else "",
                    "tp_auto_rate": f"{extra['tp_auto_rate']:.6f}" if not np.isnan(extra["tp_auto_rate"]) else "",
                    "tn_auto_rate": f"{extra['tn_auto_rate']:.6f}" if not np.isnan(extra["tn_auto_rate"]) else "",

                    "mean_pmax_accept": f"{extra['mean_pmax_accept']:.6f}" if not np.isnan(extra["mean_pmax_accept"]) else "",
                    "mean_margin_accept": f"{extra['mean_margin_accept']:.6f}" if not np.isnan(extra["mean_margin_accept"]) else "",
                })

        logging.info("Saved enriched sensitivity grid to %s", sens_csv_ext)


        # Heatmaps
        save_sensitivity_heatmaps(grid, args.output_folder, prefix="sens_full")

    # --- Choose (tau, delta) on VALIDATION of the full split using exact-count budget ---
    chosen = select_thresholds_budget_count(
        y_true=yva, probs=pva, X=Xva, budget_count=args.review_budget
    )
    tau_star   = float(chosen['tau'])
    delta_star = float(chosen['delta'])

    # Extract knee score for (tau*, delta*) if grid was computed
    kink_star = float('nan')
    if args.sensitivity:
        if args.sensitivity and 'kink' in locals():
            dist = np.abs(grid[:, 0] - tau_star) + np.abs(grid[:, 1] - delta_star)
            idx = int(np.argmin(dist))
            kink_star = float(kink[idx])


    with open(os.path.join(args.output_folder, "sens_full.csv"), "w") as f:
        f.write("tau,delta,abstain,capture,precision,dispersion,risk_accept,kink_score\n")
        f.write(f"{tau_star},{delta_star},{chosen['abstain']},{chosen['capture']},"
                f"{chosen['precision']},{chosen['dispersion']},{chosen['risk_accept']},{kink_star}\n")

    logger.info(
        f"[FULL] chosen (tau*, delta*) = ({tau_star:.6f}, {delta_star:.6f}); "
        f"val_abstain≈{chosen['abstain']:.6f} "
        f"({int(round(chosen['abstain']*len(yva)))} samples), "
        f"capture≈{chosen['capture']:.3f}, risk_accept≈{chosen['risk_accept']:.3f}"
    )


    
    # === NEW: thresholds for multiple review budgets (fractional) ===
    budget_fracs = _parse_float_csv(args.budget_fracs)
    if budget_fracs:
        out_path = os.path.join(args.output_folder, "sens_full_multi.csv")
        with open(out_path, "w", newline="") as f:
            fieldnames = [
                "budget_frac","budget_count",
                "tau","delta",
                "coverage","abstain",
                "risk_accept","capture","precision_review",
                "fn_auto_rate","fp_auto_rate",
                "kink_score_nearest",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            n_val = int(len(yva))
            for bf in budget_fracs:
                bf = float(bf)
                bc = int(round(bf * n_val))

                # Handle bc=0 explicitly: accept all => tau=0, delta=0
                if bc <= 0:
                    tau_b, delta_b = 0.0, 0.0
                    chosen_b = {"tau": tau_b, "delta": delta_b}
                else:
                    chosen_b = select_thresholds_budget_count(
                        y_true=yva, probs=pva, X=Xva, budget_count=bc
                    )
                    tau_b = float(chosen_b["tau"])
                    delta_b = float(chosen_b["delta"])

                extra_b = _selective_metrics_binary(yva, pva, tau=tau_b, delta=delta_b)

                kink_b = float("nan")
                if args.sensitivity and 'kink' in locals():
                    dist = np.abs(grid[:, 0] - tau_b) + np.abs(grid[:, 1] - delta_b)
                    kink_b = float(kink[int(np.argmin(dist))])

                w.writerow({
                    "budget_frac": bf,
                    "budget_count": bc,
                    "tau": f"{tau_b:.6f}",
                    "delta": f"{delta_b:.6f}",
                    "coverage": f"{extra_b['coverage']:.6f}",
                    "abstain": f"{extra_b['abstain']:.6f}",
                    "risk_accept": f"{extra_b['risk_accept']:.6f}" if not np.isnan(extra_b["risk_accept"]) else "",
                    "capture": f"{extra_b['capture']:.6f}" if not np.isnan(extra_b["capture"]) else "",
                    "precision_review": f"{extra_b['precision_review']:.6f}" if not np.isnan(extra_b["precision_review"]) else "",
                    "fn_auto_rate": f"{extra_b['fn_auto_rate']:.6f}" if not np.isnan(extra_b["fn_auto_rate"]) else "",
                    "fp_auto_rate": f"{extra_b['fp_auto_rate']:.6f}" if not np.isnan(extra_b["fp_auto_rate"]) else "",
                    "kink_score_nearest": f"{kink_b:.6f}" if not np.isnan(kink_b) else "",
                })

        logging.info("Saved multi-budget (tau*,delta*) table to %s", out_path)


    # --- Uncertainty histogram on TEST set (uses tau_star picked on VAL) ---
    yte, pte_raw, pte, Xte = _collect_probs_on_loader(model_all, te_all, device, args.calibration, T_full, iso_full)
    y_conf_test = pte.max(axis=1).tolist()
    y_margin_test = np.abs(pte[:, 1] - pte[:, 0]).tolist()

    label = None
    if tau_star <= 1e-9:
        label = "tau* not used — only the condition margin < delta* is active"
    save_uncertainty_hist(y_conf_test, float(tau_star), args.output_folder, label_override=label)

    # NEW: margin histogram with a vertical line at delta*
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(y_margin_test, bins=30, alpha=0.7)
    ax.axvline(float(delta_star), linestyle="--", color="red", alpha=0.8, label=f"delta* = {delta_star:.4f}")
    cal_suffix = "TS" if args.calibration == "temperature" else ("ISO" if args.calibration == "isotonic" else "CAL")
    ax.set_xlabel(f"Margin |p1 - p2| ({cal_suffix})")
    ax.set_ylabel("Number of samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "uncertainty_margin_hist.png"), dpi=300)
    plt.close(fig)
    logging.info("Saved uncertainty margin histogram")


    # --- Save full-model predictions (VAL/TEST) for stage 2.4 and the paper appendix ---
    if args.save_full_predictions:
        def _dump_pred_csv(path: str, y: np.ndarray, pr: np.ndarray, pc: np.ndarray):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "row","true",
                    "pred_RAW","p0_RAW","p1_RAW","pmax_RAW","margin_RAW",
                    "pred_CAL","p0_CAL","p1_CAL","pmax_CAL","margin_CAL",
                ])
                for i in range(len(y)):
                    p0r, p1r = float(pr[i,0]), float(pr[i,1])
                    p0c, p1c = float(pc[i,0]), float(pc[i,1])
                    w.writerow([
                        i, int(y[i]),
                        int(np.argmax(pr[i])), p0r, p1r, float(max(p0r,p1r)), float(abs(p1r-p0r)),
                        int(np.argmax(pc[i])), p0c, p1c, float(max(p0c,p1c)), float(abs(p1c-p0c)),
                    ])
        _dump_pred_csv(os.path.join(args.output_folder, "predictions_full_val.csv"), yva, pva_raw, pva)
        _dump_pred_csv(os.path.join(args.output_folder, "predictions_full_test.csv"), yte, pte_raw, pte)
        logging.info("Saved full-model prediction CSVs (VAL/TEST).")

    # --- Detect uncertain points on TEST using the chosen (tau*, delta*) on calibrated probs ---
    pmax = pte.max(axis=1)
    margin = np.abs(pte[:, 1] - pte[:, 0])
    mask_uncertain = (pmax <= float(tau_star) + 1e-12) | (margin <= float(delta_star) + 1e-12)
    idxs = np.where(mask_uncertain)[0].tolist()
    logger.info(f"[FULL] Uncertain flagged on TEST: {len(idxs)} samples.")

    # --- Save uncertain points CSV ---
    # NOTE: post_processing_real expects at least: index, X, true_label, p1, p2  :contentReference[oaicite:4]{index=4}
    csv_path = os.path.join(args.output_folder, 'uncertain_full.csv')
    csv_path = os.path.join(args.output_folder, 'uncertain_full.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','X','true_label','pred','p1','p2','pmax','margin'])
        writer.writeheader()
        # X must be in model-input space (already scaled in te_all.dataset.tensors)
        X_te_scaled_t, y_te_t = te_all.dataset.tensors
        X_te_scaled_t = X_te_scaled_t.cpu().numpy()
        for i in idxs:
            writer.writerow({
                "index": int(i),
                "X": X_te_scaled_t[i].tolist(),
                "true_label": int(yte[i]),
                "pred": int(np.argmax(pte[i])),
                "p1": float(pte[i,0]),
                "p2": float(pte[i,1]),
                "pmax": float(pmax[i]),
                "margin": float(margin[i]),
            })


    logger.info(f"[INFO] Saved full-model uncertain points to {csv_path}")
    # Make sure all logs are flushed to disk
    logging.shutdown()
    
if __name__ == "__main__":
    main()
