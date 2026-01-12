# Copyright (c) 2026 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
Main script for training and evaluating a complex-valued neural network (CVNN) 
on the MIT-BIH Arrhythmia Dataset using K-fold cross-validation across patients.

This script performs the following:
- Loads and preprocesses ECG windows from MIT-BIH records.
- Transforms real-valued time series into complex-valued representations.
- Trains a complex-valued neural network using cross-patient K-fold CV.
- Calibrates predicted probabilities on the VALIDATION split using post-hoc calibration
  (default: Platt scaling; optionally Beta/Vector/Isotonic/Temperature/None).
- Collects predictions for all folds and performs global analyses:
    - Training curves (loss and accuracy)
    - Calibration curve (reliability diagram)
    - Uncertainty histogram
    - Complex PCA scatter plot
    - Ablation support
- Extracts and saves uncertain predictions to CSV.
- Exports triage/review anchors (IDs, inputs, calibrated probabilities) for local Puiseux analysis.
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
import math
import csv
import json
import logging
import warnings
import faulthandler
import pickle
import platform
import sys
import glob
import hashlib
import inspect
import bisect
from datetime import datetime, timezone

try:
    import psutil
except Exception:
    psutil = None

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA


# Use a non-interactive backend for full reproducibility / headless servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Ensure PDF text is editable (avoid Type 3 fonts)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ───────────────────────────────────────────────────────────
#  Project imports
# ───────────────────────────────────────────────────────────
from src.find_up_synthetic import (
    SimpleComplexNet,
    complex_modulus_to_logits,
)
from mit_bih_pre.pre_pro import load_mitbih_data

from src.find_up_real import (
    prepare_complex_input,
    create_train_val_test_loaders,
    train_model as train_real,
    save_plots,
    fit_isotonic_on_val,          
    save_confusion_roc,
    expected_calibration_error,
    save_overall_history,
    save_calibration_curve,
    select_thresholds_budget_count,
    sensitivity_analysis,
    save_sensitivity_heatmaps,  
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
#  Debug / safety helpers
# ───────────────────────────────────────────────────────────

def _maybe_enable_faulthandler(out_dir: str, timeout_s: int, logger: logging.Logger):
    """
    If timeout_s > 0, periodically dumps Python stack traces to faulthandler.log.
    Returns an open file handle that MUST stay alive for the duration of the run.
    """
    if not timeout_s or int(timeout_s) <= 0:
        return None
    try:
        path = os.path.join(out_dir, "faulthandler.log")
        fh = open(path, "w")
        faulthandler.enable(file=fh, all_threads=True)
        faulthandler.dump_traceback_later(int(timeout_s), repeat=True, file=fh)
        logger.info("[DEBUG] faulthandler enabled: traceback every %ds -> %s", int(timeout_s), path)
        return fh
    except Exception as e:
        logger.warning("[DEBUG] Could not enable faulthandler: %s", e)
        return None


def _subsample_xy(X: np.ndarray, y: np.ndarray, max_points: int, seed: int):
    """
    Deterministically subsample (X,y) to at most max_points for expensive embeddings (t-SNE/UMAP).
    Returns (X_sub, y_sub, idx_or_None).
    """
    try:
        n = int(len(y))
    except Exception:
        return X, y, None
    if max_points is None or int(max_points) <= 0 or n <= int(max_points):
        return X, y, None
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    idx = np.sort(idx)
    return X[idx], y[idx], idx


def _create_train_val_test_loaders_compat(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Compatibility wrapper: passes num_workers/pin_memory only if
    create_train_val_test_loaders supports them (or accepts **kwargs).
    """
    kwargs = {"batch_size": batch_size, "seed": seed}
    try:
        sig = inspect.signature(create_train_val_test_loaders)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_varkw or ("num_workers" in sig.parameters):
            kwargs["num_workers"] = int(num_workers)
        if has_varkw or ("pin_memory" in sig.parameters):
            kwargs["pin_memory"] = bool(pin_memory)
    except Exception:
        # If signature inspection fails, fall back without extra args
        pass

    return create_train_val_test_loaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        **kwargs,
    )


def _get_calibrator_compat(
    model: nn.Module,
    val_loader,
    *,
    method: str,
    device: torch.device,
    seed: int,
    logger: logging.Logger = None,
):
    """
    Compatibility + determinism wrapper for get_calibrator():
    - passes seed/random_state iff the underlying implementation supports it (or accepts **kwargs).
    This makes Platt/Beta/Vector/Temperature calibrators reproducible across environments.
    """
    kwargs = {"method": method, "device": device}
    try:
        sig = inspect.signature(get_calibrator)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for name in ("seed", "random_state"):
            if has_varkw or (name in sig.parameters):
                kwargs[name] = int(seed)
    except Exception as e:
        if logger:
            logger.debug("Could not inspect get_calibrator signature (%s) -> calling without seed kwargs.", e)

    try:
        return get_calibrator(model, val_loader, **kwargs)
    except TypeError:
        # Absolute fallback: original call signature
        return get_calibrator(model, val_loader, method=method, device=device)


def _safe_train_val_split(X, y, test_size: float, seed: int, logger: logging.Logger = None, tag: str = "train/val"):
    """
    Try stratified train/val split, but if it is impossible (single-class, too few positives, etc.),
    fall back to an unstratified split 
    """
    try:
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )
    except Exception as e:
        if logger is not None:
            logger.warning("[%s] Stratified split failed (%s) -> using unstratified split.", tag, e)
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=None,
            random_state=seed,
        )


def _ensure_embed_method_available(embed_method: str, logger: logging.Logger) -> str:
    """
    If user requests UMAP but umap-learn is missing, fall back to TSNE with a clear warning.
    """
    m = str(embed_method).strip().lower()
    if m == "umap":
        try:
            import umap  # noqa: F401
        except Exception as e:
            logger.warning("UMAP requested but 'umap-learn' is not installed (%s). Falling back to TSNE.", e)
            return "tsne"
    return m


def _save_joint_pca_scatter(X: np.ndarray, y: np.ndarray, out_dir: str, *, fname: str = "complex_PCA_scatter.png", save_pdf: bool = True):
    """
    replacement for the old 'real PC1 vs imag PC1' plot:
    joint PCA on the full feature vector (Re+Im) -> PC1 vs PC2.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2 or X.shape[0] == 0:
        return

    # standardize
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for lab in sorted(np.unique(y).tolist()):
        m = (y == lab)
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.35, label=f"class {lab} (n={int(m.sum())})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ev = pca.explained_variance_ratio_
    ax.set_title(f"Joint PCA of complex features (EV: {ev[0]:.2f}, {ev[1]:.2f})")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_png = os.path.join(out_dir, fname)
    fig.savefig(out_png, dpi=300)
    if save_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)



def _save_2d_scatter(
    Z: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    *,
    fname: str,
    title: str,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    save_pdf: bool = True,
):
    """Generic paper-style 2D scatter (PNG + optional PDF)."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=int)
    if Z.ndim != 2 or Z.shape[1] != 2 or Z.shape[0] == 0:
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for lab in sorted(np.unique(y).tolist()):
        m = (y == lab)
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.35, label=f"class {lab} (n={int(m.sum())})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_png = os.path.join(out_dir, fname)
    fig.savefig(out_png, dpi=300)
    if save_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)


def _save_embedding_scatter(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    *,
    method: str,
    seed: int,
    save_pdf: bool,
    logger: logging.Logger = None,
):
    """
    Optional qualitative 2D embeddings for appendices:
      - method='tsne' (sklearn)
      - method='umap' (umap-learn, optional)
    Always standardizes X first.
    """
    m = str(method).strip().lower()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2 or X.shape[0] < 5:
        if logger:
            logger.warning("[EMBED] %s skipped (need >=5 points).", m.upper())
        return

    # standardize
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd

    if m == "tsne":
        from sklearn.manifold import TSNE
        n = int(Xs.shape[0])
        # TSNE constraint: perplexity < n_samples
        perp = float(min(30.0, max(5.0, (n - 1) / 3.0)))
        perp = float(min(perp, max(2.0, n - 1.0)))
        tsne = TSNE(
            n_components=2,
            random_state=int(seed),
            init="pca",
            learning_rate="auto",
            perplexity=perp,
        )
        Z = tsne.fit_transform(Xs)
        _save_2d_scatter(
            Z, y, out_dir,
            fname="complex_TSNE_scatter.png",
            title=f"t-SNE of complex features (n={n}, perplexity={perp:.1f})",
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
            save_pdf=save_pdf,
        )
        return

    if m == "umap":
        import umap
        n = int(Xs.shape[0])
        emb = umap.UMAP(n_components=2, random_state=int(seed))
        Z = emb.fit_transform(Xs)
        _save_2d_scatter(
            Z, y, out_dir,
            fname="complex_UMAP_scatter.png",
            title=f"UMAP of complex features (n={n})",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            save_pdf=save_pdf,
        )
        return

    if logger:
        logger.warning("[EMBED] Unknown method=%s -> skipped.", method)


# ───────────────────────────────────────────────────────────
#  Post-hoc calibration utilities
#  NOTE:
#   - Default calibration for paper runs is **Platt scaling** (or **Beta scaling**),
#     fitted on the per-fold VALIDATION split.
#   - This is intentional: training uses class rebalancing/oversampling, which can
#     shift the effective class prior and distort raw probabilities.
#   - Calibrators with an intercept (Platt/Beta/Vector) are typically more robust
#     under such base-rate shift. Temperature scaling remains available mainly
#     for ablations/comparison.
# ───────────────────────────────────────────────────────────

def _collect_logits_and_labels(model: nn.Module, loader, device: torch.device):
    """
    Collect logits and labels from a loader in a single forward pass.
    IMPORTANT: logits are computed via complex_modulus_to_logits(model(x)),
    to stay consistent with the rest of this script (CVNN output handling).
    Returns (logits_cpu [N,C], y_cpu [N]).
    """
    was_training = bool(model.training)
    model.eval()
    logits_chunks = []
    y_chunks = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = complex_modulus_to_logits(model(xb))
            logits_chunks.append(logits.detach().cpu())
            y_chunks.append(yb.detach().cpu())

    if was_training:
        model.train()

    if not logits_chunks:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    logits_all = torch.cat(logits_chunks, dim=0).to(dtype=torch.float32, device="cpu")
    y_all = torch.cat(y_chunks, dim=0).to(dtype=torch.long, device="cpu")
    return logits_all, y_all


def _fit_temperature_scaling_from_logits(
    logits_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    *,
    init_temp: float = 1.0,
    min_temp: float = 0.05,
    max_temp: float = 10.0,
    max_iter: int = 50,
    logger: logging.Logger = None,
    tag: str = "",
) -> torch.Tensor:
    """
    Robust temperature scaling on fixed logits (CPU), minimizing NLL (cross-entropy).
    - Ensures strictly positive T via log-parameterization.
    - Clamps T to [min_temp, max_temp] to avoid pathological values.
    - If VAL is single-class or anything goes wrong -> returns T=1.0 (no-op).

    Returns: torch.Tensor([T]) on CPU (float32).
    """
    # Basic shape safety
    if logits_cpu is None or y_cpu is None:
        if logger:
            logger.warning("[%s] TempScaling: logits/y is None -> using T=1.0", tag)
        return torch.tensor(1.0, dtype=torch.float32)

    logits = logits_cpu.detach().to(dtype=torch.float32, device="cpu")
    y = y_cpu.detach().to(dtype=torch.long, device="cpu")

    n = int(y.numel())
    if n <= 0 or logits.ndim != 2 or logits.shape[0] != n:
        if logger:
            logger.warning(
                "[%s] TempScaling: invalid shapes logits=%s y=%s -> using T=1.0",
                tag, tuple(getattr(logits, "shape", ())), tuple(getattr(y, "shape", ()))
            )
        return torch.tensor(1.0, dtype=torch.float32)

    # If only one class is present, temperature scaling is ill-posed for calibration.
    # For paper stability: do a no-op (T=1.0) and log.
    try:
        if int(torch.unique(y).numel()) < 2:
            if logger:
                logger.warning("[%s] TempScaling: VAL single-class -> using T=1.0 (no-op).", tag)
            return torch.tensor(1.0, dtype=torch.float32)
    except Exception:
        # If unique() fails for any reason, continue best-effort.
        pass

    # Clamp init within bounds (avoid log(0) or absurd start)
    try:
        init_temp = float(init_temp)
    except Exception:
        init_temp = 1.0
    init_temp = float(np.clip(init_temp, float(min_temp), float(max_temp)))

    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        nll_T1 = float(ce(logits, y).item())

    # Optimize log(T) with LBFGS (fast for 1 parameter, stable, standard for temp scaling)
    log_T = torch.tensor([math.log(max(init_temp, 1e-12))], dtype=torch.float32, requires_grad=True)

    # Slight regularizer to prevent extreme drift in degenerate/low-signal cases
    reg = 1e-6

    opt = torch.optim.LBFGS(
        [log_T],
        lr=0.1,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        loss = ce(logits / T, y) + reg * (log_T ** 2).sum()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception as e:
        if logger:
            logger.warning("[%s] TempScaling: LBFGS failed (%s) -> using T=1.0", tag, e)
        return torch.tensor(1.0, dtype=torch.float32)

    with torch.no_grad():
        T_raw = torch.exp(log_T.detach()).to(dtype=torch.float32, device="cpu")
        if (not torch.isfinite(T_raw).all()) or float(T_raw.item()) <= 0.0:
            if logger:
                logger.warning("[%s] TempScaling: non-finite or non-positive T -> using T=1.0", tag)
            return torch.tensor(1.0, dtype=torch.float32)

        T_clamped = torch.clamp(T_raw, min=float(min_temp), max=float(max_temp))

        # Evaluate NLL for candidates and pick the best (never worsen vs T=1)
        candidates = [
            torch.tensor(1.0, dtype=torch.float32),
            T_clamped,
        ]

        best_T = candidates[0]
        best_nll = nll_T1
        for Tc in candidates:
            try:
                nll = float(ce(logits / Tc, y).item())
            except Exception:
                continue
            if np.isfinite(nll) and (nll <= best_nll + 1e-12):
                best_nll = nll
                best_T = Tc

        if logger:
            logger.info(
                "[%s] TempScaling: N=%d | NLL(T=1)=%.6f -> NLL=%.6f | T=%.4f (raw=%.4f, clamp=[%.3f,%.3f])",
                tag, n, nll_T1, best_nll,
                float(best_T.item()), float(T_raw.item()),
                float(min_temp), float(max_temp)
            )

        return best_T.detach().to(dtype=torch.float32, device="cpu")


def _tune_temperature_safe(
    model: nn.Module,
    val_loader,
    *,
    device: torch.device,
    logger: logging.Logger,
    init_temp: float,
    min_temp: float,
    max_temp: float,
    max_iter: int,
    tag: str,
) -> torch.Tensor:
    """
    One-pass logits collection + safe temperature fit.
    Always returns a finite positive scalar tensor on CPU.
    """
    logits_cpu, y_cpu = _collect_logits_and_labels(model, val_loader, device=device)
    return _fit_temperature_scaling_from_logits(
        logits_cpu,
        y_cpu,
        init_temp=init_temp,
        min_temp=min_temp,
        max_temp=max_temp,
        max_iter=max_iter,
        logger=logger,
        tag=tag,
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

    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help=("Optional subfolder name inside --output_folder for this run "
              "(recommended for paper-ready experiments; avoids mixing artifacts)."),
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)

    # Model hyperparameter: modReLU bias (critical for nonlinearity / kink prevalence)
    parser.add_argument(
        "--modrelu_bias",
        type=float,
        default=0.1,
        help="Bias term used in modReLU activation (SimpleComplexNet). "
             "Positive => always-active regime; negative => introduces inactive/kink regions.",
    )


    parser.add_argument("--folds", type=int, default=10)
    # NOTE: kept for backwards compatibility; predictions use argmax for binary.
    # (We log a warning if threshold != 0.5 to avoid confusion in paper appendices.)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")


    # DataLoader controls (reproducibility + avoiding deadlocks on some systems)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader num_workers. 0 is safest/deterministic; >0 may speed up but can hang on some setups."
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Pin memory in DataLoaders (useful on GPU)."
    )

    # Debug: periodic traceback dumps to diagnose 'hangs'
    parser.add_argument(
        "--hang_timeout",
        type=int,
        default=0,
        help="If >0: dump Python stack traces every N seconds to faulthandler.log (great for diagnosing hangs)."
    )


    # Paper-control knobs
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        help="Number of bins for Expected Calibration Error (ECE). Default: 15.",
    )
    parser.add_argument(
        "--save-pdf",
        dest="save_pdf",
        action="store_true",
        default=True,
        help="Save PDF copies of paper figures (ON by default).",
    )
    parser.add_argument(
        "--no-save-pdf",
        dest="save_pdf",
        action="store_false",
        help="Disable saving PDF copies of figures.",
    )

    # Grid-based (tau/delta) sensitivity analysis + heatmaps
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
    parser.add_argument(
        "--calibration",
        type=str,
        default="platt",
        choices=["platt", "beta", "vector", "temperature", "isotonic", "none"],
        help=(
            "Calibration method fitted on VALIDATION. "
            "Recommended with oversampling: platt/beta/vector (have intercept). "
            "Default: platt."
        ),
    )


    # Temperature scaling controls (paper-stable defaults + clamping)
    parser.add_argument(
        "--temp_init",
        type=float,
        default=1.0,
        help="Initial temperature for scaling optimization (default 1.0).",
    )
    parser.add_argument(
        "--temp_min",
        type=float,
        default=0.05,
        help="Minimum clamp for temperature (default 0.05).",
    )
    parser.add_argument(
        "--temp_max",
        type=float,
        default=10.0,
        help="Maximum clamp for temperature (default 10.0).",
    )
    parser.add_argument(
        "--temp_max_iter",
        type=int,
        default=50,
        help="Max LBFGS iterations for temperature scaling (default 50).",
    )

    parser.add_argument(
        "--calibs",
        type=str,
        default="platt,beta,vector,temperature,isotonic,none",
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
    
    
    
    parser.add_argument("--embed_method", type=str, default="pca",
                        choices=["pca", "tsne", "umap"],
                        help="2D projection method for feature scatter (default: pca; deterministic and fast).")


    parser.add_argument(
        "--embed_max_points",
        type=int,
        default=5000,
        help="Max number of points used for TSNE/UMAP scatter. If exceeded, subsample deterministically."
    )


    # Full model retrain + triage stage (enabled by default; can be disabled for quick CV-only runs)
    parser.add_argument(
        "--full-retrain",
        dest="full_retrain",
        action="store_true",
        default=True,
        help="Run full model retrain + triage selection after CV (ON by default).",
    )
    parser.add_argument(
        "--no-full-retrain",
        dest="full_retrain",
        action="store_false",
        help="Skip full model retrain/triage stage (CV-only run).",
    )
    parser.add_argument(
        "--full-test-fold",
        type=int,
        default=1,
        help="Which CV fold partition to use as TEST records for the full retrain split (1-indexed).",
    )
    parser.add_argument(
        "--full-val-fold",
        type=int,
        default=2,
        help="Which CV fold partition to use as VAL records for the full retrain split (1-indexed).",
    )


    parser.add_argument(
        "--auto-full-folds",
        action="store_true",
        default=False,
        help=(
            "Auto-select --full-test-fold and --full-val-fold using ONLY per-fold label counts "
            "(no model performance), to avoid extreme class-prior shift between FULL VAL and FULL TEST. "
            "Useful for paper-ready FULL split stability."
        ),
    )


    # Optional: persist lightweight arrays (.npz) for downstream analysis / debugging
    parser.add_argument(
        "--save-arrays",
        dest="save_arrays",
        action="store_true",
        default=True,
        help="Save lightweight .npz arrays (predictions/probabilities) (ON by default).",
    )

    # Per-sample CV predictions CSV
    parser.add_argument(
        "--save-pred-csv",
        dest="save_pred_csv",
        action="store_true",
        default=False,
        help="Save per-sample predictions_all_folds.csv (can be huge / slow). OFF by default."
    )
    parser.add_argument(
        "--no-save-pred-csv",
        dest="save_pred_csv",
        action="store_false",
        help="Disable per-sample predictions CSV (default)."
    )


    parser.add_argument(
        "--no-save-arrays",
        dest="save_arrays",
        action="store_false",
        help="Disable saving .npz arrays.",
    )


    parser.add_argument(
        "--save-env",
        dest="save_env",
        action="store_true",
        default=True,
        help="Save pip-freeze snapshot to pip_freeze.txt (ON by default).",
    )
    parser.add_argument(
        "--no-save-env",
        dest="save_env",
        action="store_false",
        help="Disable pip-freeze snapshot.",
    )
    parser.add_argument(
        "--hash-manifest",
        dest="hash_manifest",
        action="store_true",
        default=False,
        help=("If set, artifact_manifest.json includes SHA256 per file (slower; useful for archival)."),
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


def _float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _calib_suffix(method: str) -> str:
    """
    Short, filesystem-safe label for calibration method.
    Used for figure/CSV suffixes.
    """
    m = str(method).strip().lower()
    if m == "temperature":
        return "TS"
    if m == "isotonic":
        return "ISO"
    if m == "platt":
        return "PLATT"
    if m == "beta":
        return "BETA"
    if m == "vector":
        return "VECTOR"
    if m in ("none", "raw"):
        return "RAW"
    return m.upper()


def _sanitize_run_id(run_id: str) -> str:
    """
    Make run_id filesystem-safe and avoid accidental nested paths.
    Allowed chars: [A-Za-z0-9._-] ; everything else becomes '_'.
    """
    rid = str(run_id or "").strip()
    if not rid:
        return ""
    # replace path separators explicitly
    rid = rid.replace(os.sep, "_")
    if os.altsep:
        rid = rid.replace(os.altsep, "_")
    out = []
    for c in rid:
        if c.isalnum() or c in ("-", "_", "."):
            out.append(c)
        else:
            out.append("_")
    rid2 = "".join(out).strip("._")
    return rid2


def _ensure_prob_matrix_np(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Ensure probabilities are a valid NxC matrix on the simplex.
    Supports binary outputs shaped as (N,), (N,1), or (N,2).
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    elif p.ndim == 2 and p.shape[1] == 1:
        p = np.concatenate([1.0 - p, p], axis=1)
    elif p.ndim != 2:
        raise ValueError(f"Invalid prob shape: {p.shape}")
    p = np.clip(p, eps, 1.0)
    denom = np.clip(p.sum(axis=1, keepdims=True), eps, np.inf)
    p = p / denom
    return p


def _apply_calibrator_probs_compat(cal_fn, probs_np):
    """
    Compatibility wrapper for calibrators that may accept either:
      - apply_fn(probs_np=<np.ndarray>)
      - apply_fn(<np.ndarray>)
    """
    try:
        return cal_fn(probs_np=probs_np)
    except TypeError:
        return cal_fn(probs_np)



def _ensure_prob_matrix_torch(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Torch version of _ensure_prob_matrix_np.
    Supports binary outputs shaped as (N,), (N,1), or (N,2).
    """
    p = probs
    if not torch.is_tensor(p):
        p = torch.as_tensor(p)
    if p.ndim == 1:
        p = torch.stack([1.0 - p, p], dim=1)
    elif p.ndim == 2 and p.shape[1] == 1:
        p = torch.cat([1.0 - p, p], dim=1)
    if p.ndim != 2:
        raise ValueError(f"Invalid prob tensor shape: {tuple(p.shape)}")
    p = p.to(dtype=torch.float32)
    p = torch.clamp(p, min=eps, max=1.0)
    denom = torch.clamp(p.sum(dim=1, keepdim=True), min=eps)
    p = p / denom
    return p


def _calibrate_probs_torch(
    logits: torch.Tensor,
    probs_raw: torch.Tensor,
    *,
    method: str,
    T: torch.Tensor = None,
    iso_cal=None,
    cal_fn=None,
) -> torch.Tensor:
    """
    Single source of truth: apply post-hoc calibration and return a valid prob matrix.

    Parameters
    ----------
    logits:
        Torch tensor [N,C] (any device) produced by complex_modulus_to_logits.
    probs_raw:
        Torch tensor [N,C] = softmax(logits) (any device).
    method:
        'temperature' | 'isotonic' | 'platt' | 'beta' | 'vector' | 'none'
    T / iso_cal / cal_fn:
        Fitted calibration objects for the selected method.

    Notes
    -----
    - Numpy-based calibrators (isotonic / platt / beta / vector) run on CPU.
    - Output is always normalized via _ensure_prob_matrix_torch.
    """
    m = str(method).strip().lower()

    # Always treat probs_raw as a well-defined simplex matrix
    probs_raw = _ensure_prob_matrix_torch(probs_raw)

    if m == "temperature" and T is not None:
        # Keep temperature scaling on logits device (fast on GPU, stable on CPU)
        try:
            T_dev = T.to(device=logits.device, dtype=logits.dtype)
        except Exception:
            T_dev = T
        probs = torch.softmax(logits / T_dev, dim=1)
        return _ensure_prob_matrix_torch(probs)

    if m == "isotonic" and iso_cal is not None:
        probs_np = iso_cal(probs_raw.detach().cpu().numpy())
        return _ensure_prob_matrix_torch(torch.from_numpy(np.asarray(probs_np)))

    if m in ("platt", "vector") and cal_fn is not None:
        probs_np = cal_fn(logits.detach().cpu().numpy())
        return _ensure_prob_matrix_torch(torch.from_numpy(np.asarray(probs_np)))

    if m == "beta" and cal_fn is not None:
        probs_np = _apply_calibrator_probs_compat(cal_fn, probs_raw.detach().cpu().numpy())
        return _ensure_prob_matrix_torch(torch.from_numpy(np.asarray(probs_np)))

    # 'none' or missing calibrator => RAW
    return probs_raw


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_write_pip_freeze(out_dir: str, logger: logging.Logger):
    """Best-effort environment capture for paper reproducibility."""
    try:
        import subprocess
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        path = os.path.join(out_dir, "pip_freeze.txt")
        with open(path, "w") as f:
            f.write(freeze)
        logger.info("Saved pip freeze -> %s", path)
        return path
    except Exception as e:
        logger.warning("Could not capture pip freeze: %s", e)
        return None


_DEFAULT_ARTIFACT_PATTERNS = [
    "*.csv", "*.json", "*.png", "*.pdf",
    "*.npz", "*.npy",
    "*.pt", "*.pkl",
    "*.txt", "*.log",
    "*.md",
]



def _list_existing_artifacts(out_dir: str):
    """Return a sorted list of existing artifact files in out_dir (used to avoid mixing runs)."""
    files = []
    for pat in _DEFAULT_ARTIFACT_PATTERNS:
        files.extend(glob.glob(os.path.join(out_dir, pat)))
    files = [p for p in files if os.path.isfile(p)]
    return sorted(set(files))


def _write_artifact_manifest(out_dir: str, logger: logging.Logger, hash_files: bool = False):
    """Write artifact_manifest.json for auditability (what was produced in this run)."""
    files = []
    for pat in _DEFAULT_ARTIFACT_PATTERNS:
        for p in sorted(glob.glob(os.path.join(out_dir, pat))):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, out_dir)
            item = {"path": rel, "bytes": int(os.path.getsize(p))}
            if hash_files:
                try:
                    item["sha256"] = _sha256_file(p)
                except Exception:
                    item["sha256"] = None
            files.append(item)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": os.path.abspath(out_dir),
        "n_files": int(len(files)),
        "files": files,
    }
    out_path = os.path.join(out_dir, "artifact_manifest.json")
    try:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved artifact manifest -> %s (n=%d)", out_path, len(files))
    except Exception as e:
        logger.warning("Failed to write artifact manifest: %s", e)

def _write_artifact_readme(out_dir: str, args, logger: logging.Logger):

    try:
        key_files = [
            ("run_args.json", "Exact CLI arguments for this run."),
            ("run_meta.json", "Environment + git + script hash + device metadata."),
            ("pip_freeze.txt", "Dependency snapshot (best-effort)."),
            ("cv_record_splits.json", "Record-level CV partitions (train/test per fold)."),
            ("record_window_stats.csv", "Per-record window counts and positives (dataset audit)."),
            ("cv_metrics_per_fold_extended.csv", "Per-fold performance + calibration metrics (paper table source)."),
            ("cv_metrics_summary_perf.csv", "Mean±CI95 for performance metrics across folds."),
            ("cv_metrics_summary.csv", "Mean±CI95 for calibration metrics (RAW vs CAL)."),
            ("cv_metrics_per_fold_multi.csv", "Per-fold metrics for multiple calibration methods (temperature/isotonic/platt/beta/vector/none)."),
            ("cv_metrics_summary_multi.csv", "Mean±CI95 for multi-calibration comparison across folds."),
            ("cv_test_metrics.txt", "Pooled CV TEST metrics (incl. pooled calibration)."),
            ("confusion_roc_cv.png", "Pooled CV: confusion + ROC + PR (PNG)."),
            ("confusion_roc_cv.pdf", "Pooled CV: confusion + ROC + PR (PDF)."),
            ("uncertainty_hist_cv.png", "Pooled CV: pmax histogram."),
            ("uncertainty_margin_hist_cv.png", "Pooled CV: margin histogram."),
            ("predictions_all_folds.csv", "Per-sample predictions/probabilities/margins (all folds)."),
            ("cv_test_arrays.npz", "Lightweight pooled CV arrays for downstream analysis."),
            ("full_split_records.json", "Record-level FULL train/val/test split."),
            ("sens_grid_ext.csv", "FULL-VAL tau/delta grid (enriched exact counts + clinical FN/FP auto)."),
            ("sens_full.csv", "Chosen (tau*,delta*) on FULL-VAL + knee score."),
            ("sens_grid.csv", "FULL-VAL tau/delta grid (with knee score)."),
            ("sens_full_multi.csv", "FULL-VAL thresholds for multiple fractional review budgets (tau/delta per budget)."),
            ("full_test_triage_curve.csv", "FULL-TEST triage curve (evaluate the FULL-VAL budget thresholds on TEST)."),
            ("full_test_risk_coverage.png", "FULL-TEST selective risk–coverage plot (PNG)."),
            ("full_test_risk_coverage.pdf", "FULL-TEST selective risk–coverage plot (PDF)."),
            ("full_test_capture_abstain.png", "FULL-TEST capture–abstain plot (PNG)."),
            ("full_test_capture_abstain.pdf", "FULL-TEST capture–abstain plot (PDF)."),
            ("uncertainty_hist.png", "FULL-TEST pmax histogram (PNG)."),
            ("uncertainty_hist.pdf", "FULL-TEST pmax histogram (PDF)."),
            ("uncertainty_margin_hist.png", "FULL-TEST margin histogram (PNG)."),
            ("uncertainty_margin_hist.pdf", "FULL-TEST margin histogram (PDF)."),
            ("full_test_metrics.txt", "FULL-TEST metrics + selective triage summary."),
            ("full_test_selective_metrics.csv", "FULL-TEST selective metrics (exact counts)."),
            ("confusion_roc_full.png", "FULL-TEST confusion + ROC + PR (PNG)."),
            ("confusion_roc_full.pdf", "FULL-TEST confusion + ROC + PR (PDF)."),
            ("uncertain_full.csv", "FULL-TEST review set (legacy format)."),
            ("uncertain_full_ext.csv", "FULL-TEST review set (with pmax/margin/reason)."),
            ("artifact_manifest.json", "Machine-readable list of all produced artifacts."),
            ("cv_fold_window_stats.csv", "Per-fold TEST window counts + pos_frac (helps sanity-check FULL split choice)."),
            ("cv_selective_budget_count_per_fold.csv", "CV selective triage (thresholds chosen on VAL count budget; evaluated on TEST) per fold."),
            ("cv_test_triage_curve_per_fold.csv", "CV triage curve across budget_fracs (thresholds from VAL; metrics on TEST) per fold."),
            ("cv_test_triage_curve_summary.csv", "CV triage curve aggregated across folds (mean ± CI95)."),
            ("cv_risk_coverage_mean_ci.png", "CV selective risk–coverage (mean ± CI95) (PNG)."),
            ("cv_risk_coverage_mean_ci.pdf", "CV selective risk–coverage (mean ± CI95) (PDF)."),
            ("cv_capture_abstain_mean_ci.png", "CV capture–abstain (mean ± CI95) (PNG)."),
            ("cv_capture_abstain_mean_ci.pdf", "CV capture–abstain (mean ± CI95) (PDF)."),
        ]

        lines = []
        lines.append("# Artifact index\n")
        lines.append(f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        lines.append(f"Output dir: `{os.path.abspath(out_dir)}`\n")
        lines.append(f"run_id: `{getattr(args, 'run_id', '')}`  |  seed: `{args.seed}`  |  folds: `{args.folds}`  |  calibration: `{args.calibration}`\n")
        lines.append("\n## Key files\n")
        for fn, desc in key_files:
            p = os.path.join(out_dir, fn)
            if os.path.exists(p):
                lines.append(f"- **{fn}** ({os.path.getsize(p)} bytes) — {desc}")
            else:
                lines.append(f"- {fn} — {desc} *(missing)*")

        lines.append("\n## Conventions\n")
        lines.append("- `RAW` = uncalibrated softmax.\n")
        lines.append("- `CAL` = output of `--calibration` (platt / beta / vector / temperature / isotonic / none).\n")
        lines.append("- Selective rule: **ACCEPT iff** `pmax >= tau` AND `margin >= delta`; otherwise **REVIEW**.\n")
        lines.append("- Binary note (C=2): `margin=|p1-p0|=2*pmax-1`, so τ/δ are redundant; we use a canonical encoding (often τ=0, δ>0) for consistency.\n")

        out_path = os.path.join(out_dir, "ARTIFACTS.md")
        with open(out_path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        logger.info("Wrote artifact index -> %s", out_path)
    except Exception as e:
        logger.warning("Failed to write ARTIFACTS.md: %s", e)

def _default_tau_delta_grid():
    """
    Strong default grid:
    - tau: coarse in [0.50..0.90], fine in [0.90..0.99]
    - delta: dense near 0 (most action happens there), then medium, then coarse tail
    """
    taus = np.unique(np.concatenate([
        np.linspace(0.50, 0.90, 9),     # 0.50,0.55,...,0.90
        np.linspace(0.90, 0.99, 10),    # 0.90,0.91,...,0.99
    ])).astype(float)

    deltas = np.unique(np.concatenate([
        np.linspace(0.00, 0.10, 11),    # 0.00..0.10 step 0.01
        np.linspace(0.12, 0.30, 10),    # 0.12..0.30
        np.linspace(0.35, 0.60, 6),     # 0.35..0.60
    ])).astype(float)

    return taus, deltas


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

    fn_auto_rate = (fn_auto / n_pos) if n_pos > 0 else float("nan")  
    fp_auto_rate = (fp_auto / n_neg) if n_neg > 0 else float("nan")  
    tp_auto_rate = (tp_auto / n_pos) if n_pos > 0 else float("nan")
    tn_auto_rate = (tn_auto / n_neg) if n_neg > 0 else float("nan")

    mean_pmax_accept = float(pmax[accept].mean()) if n_accept > 0 else float("nan")
    mean_margin_accept = float(margin[accept].mean()) if n_accept > 0 else float("nan")

    return {
        "n_total": n_total,
        "n_accept": n_accept,
        "n_review": n_review,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_err": int(n_err),
        "coverage": coverage,
        "abstain": abstain,
        "risk_accept": risk_accept,
        "capture": capture,
        "precision_review": precision_review,
        "fn_auto": int(fn_auto),
        "fp_auto": int(fp_auto),
        "tp_auto": int(tp_auto),
        "tn_auto": int(tn_auto),
        "fn_auto_rate": fn_auto_rate,
        "fp_auto_rate": fp_auto_rate,
        "tp_auto_rate": tp_auto_rate,
        "tn_auto_rate": tn_auto_rate,
        "mean_pmax_accept": mean_pmax_accept,
        "mean_margin_accept": mean_margin_accept,
    }



def _select_thresholds_budget_count_fast(y_true, probs, budget_count, X=None):
    """
    Fast exact-count threshold selection for the rule:
      ACCEPT iff pmax >= tau AND margin >= delta

    Canonical binary encoding (paper-friendly):
    ------------------------------------------
    For C==2, margin = |p1 - p0| = 2*pmax - 1, hence (tau, delta) are redundant.
    To preserve the *margin* narrative (and make delta* meaningful in figures/tables),
    we return a canonical pair:
        tau = 0.0
        delta chosen to match the desired review budget (via an equivalent pmax threshold).

    This yields the SAME accept/review partition as pmax-thresholding, but expresses the
    operating point through delta (as in the Puiseux / uncertainty-mining framing).

    Returns a dict compatible with the old output: tau, delta, abstain, capture, precision,
    dispersion, risk_accept.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = _ensure_prob_matrix_np(np.asarray(probs, dtype=float))
    n = int(y_true.size)

    # sanitize budget_count
    bc = int(budget_count) if budget_count is not None else 0
    bc = max(0, min(bc, n))

    if n == 0:
        return {
            "tau": 0.0,
            "delta": 0.0,
            "abstain": float("nan"),
            "capture": float("nan"),
            "precision": float("nan"),
            "dispersion": float("nan"),
            "risk_accept": float("nan"),
        }

    # If not binary, fall back to the original (may be slower).
    if probs.shape[1] != 2:
        return select_thresholds_budget_count(y_true=y_true, probs=probs, X=X, budget_count=bc)

    pmax = probs.max(axis=1)

    # We choose a pmax-threshold that yields ~bc REVIEW samples (pmax < p_thr),
    # then encode it as (tau=0, delta=2*p_thr - 1).
    if bc == 0:
        tau = 0.0
        delta = 0.0  # accept all
    elif bc >= n:
        tau = 0.0
        # review all: force accept_mask to be always False (margin<=1.0)
        delta = float(np.nextafter(1.0, 2.0))
    else:
        p_sorted = np.sort(pmax)
        lo = float(p_sorted[bc - 1])
        hi = float(p_sorted[bc])  # bc < n here

        if hi > lo:
            p_thr = 0.5 * (lo + hi)
        else:
            # Ties: strict threshold cannot guarantee exact bc.
            # Choose p_thr=lo so we do NOT exceed the review capacity (may underuse by a few in tie-pathological cases).
            p_thr = lo

        tau = 0.0
        delta = 2.0 * float(p_thr) - 1.0
        if delta < 0.0:
            delta = 0.0

    # Compute metrics (paper-friendly)
    m = _selective_metrics_binary(y_true, probs, tau=float(tau), delta=float(delta))

    # Cheap dispersion proxy on REVIEW set only (won't explode to NxN)
    dispersion = float("nan")
    try:
        if X is not None:
            X = np.asarray(X, dtype=float)
            margin = np.abs(probs[:, 1] - probs[:, 0])  # binary margin
            accept = (pmax >= float(tau)) & (margin >= float(delta))
            Xr = X[~accept]
            if Xr.shape[0] >= 2:
                ctr = Xr.mean(axis=0)
                dispersion = float(np.linalg.norm(Xr - ctr, axis=1).mean())
            elif Xr.shape[0] == 1:
                dispersion = 0.0
    except Exception:
        dispersion = float("nan")

    return {
        "tau": float(tau),
        "delta": float(delta),
        "abstain": float(m.get("abstain", float("nan"))),
        "capture": float(m.get("capture", float("nan"))),
        "precision": float(m.get("precision_review", float("nan"))),
        "dispersion": float(dispersion),
        "risk_accept": float(m.get("risk_accept", float("nan"))),
    }




def _mean_ci95_ignore_nan(values):
    """Mean and 95% CI half-width, ignoring NaNs (paper tables)."""
    a = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    mean = float(a.mean())
    if a.size == 1:
        return mean, 0.0
    hw = 1.96 * float(a.std(ddof=1)) / (a.size ** 0.5)
    return mean, hw


def _log_label_stats(logger: logging.Logger, tag: str, y):
    """Log and return basic label statistics (binary)."""
    y = np.asarray(y, dtype=int)
    n = int(y.size)
    pos = int((y == 1).sum())
    neg = n - pos
    frac = 100.0 * pos / max(n, 1)
    logger.info("[%s] N=%d | pos=%d (%.2f%%) | neg=%d", tag, n, pos, frac, neg)
    return {"n": n, "pos": pos, "neg": neg, "pos_frac": pos / max(n, 1)}


def _make_pos_frac_strata(pos_frac: np.ndarray, n_splits: int, max_bins: int = 4):
    """
    Build record-level strata from pos_frac using quantile binning.
    Tries up to max_bins bins; if any bin is too small for n_splits, reduces bins.
    Falls back to has_pos (2 bins) if needed.
    Returns: (strata[int], desc[str])
    """
    x = np.asarray(pos_frac, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Try quantile bins (max_bins -> 2)
    for nb in range(int(max_bins), 1, -1):
        qs = np.linspace(0.0, 1.0, nb + 1)
        edges = np.unique(np.quantile(x, qs))
        if edges.size < 3:
            continue  # would produce <2 bins
        strata = np.digitize(x, edges[1:-1], right=False).astype(int)
        counts = np.bincount(strata)
        if (counts.size >= 2) and (counts.min() >= int(n_splits)):
            return strata, f"pos_frac_quantile_bins{int(counts.size)}"
    # Fallback: has_pos (2 bins)
    strata = (x > 0.0).astype(int)
    return strata, "has_pos_fallback"



def _compute_binary_metrics_and_curves(y_true, y_pred, y_score=None):
    """
    Compute binary classification metrics + curves for plotting.
    Returns: (metrics_dict, cm, (fpr,tpr), (recall,precision))
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = (int(x) for x in cm.ravel())
    else:
        tn = fp = fn = tp = 0
    n_total = int(cm.sum())

    acc = (tp + tn) / max(n_total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = 0.5 * (recall + specificity)

    # ROC / PR only if both classes are present
    if y_score is not None and len(np.unique(y_true)) > 1:
        roc_auc_val = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)

        ap_val = float(average_precision_score(y_true, y_score))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    else:
        roc_auc_val = float("nan")
        fpr, tpr = np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
        ap_val = float("nan")
        precision_curve, recall_curve = np.asarray([1.0]), np.asarray([0.0])

    metrics = {
        "N": n_total,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
        "BalancedAcc": float(bal_acc),
        "ROC_AUC": float(roc_auc_val),
        "PR_AUC(AP)": float(ap_val),
    }
    return metrics, cm, (fpr, tpr), (recall_curve, precision_curve)


def _save_confusion_roc_pr_figure(cm, roc, pr, metrics, out_png: str, title_prefix: str, save_pdf: bool = True):
    """Create a 1×3 figure: confusion matrix + ROC + PR (paper-ready)."""
    fpr, tpr = roc
    rec, prec = pr  # x=recall, y=precision

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))

    # Confusion matrix
    ax0 = axes[0]
    ax0.imshow(cm, interpolation="nearest")
    ax0.set_title(f"{title_prefix}: Confusion")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    for (r, c), v in np.ndenumerate(cm):
        ax0.text(c, r, str(int(v)), ha="center", va="center")

    # ROC
    ax1 = axes[1]
    ax1.plot(fpr, tpr, label=f"ROC AUC={metrics.get('ROC_AUC', float('nan')):.3f}")
    ax1.plot([0, 1], [0, 1], "--", alpha=0.6)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"{title_prefix}: ROC")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # PR
    ax2 = axes[2]
    ax2.plot(rec, prec, label=f"AP={metrics.get('PR_AUC(AP)', float('nan')):.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix}: Precision–Recall")
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if save_pdf:
        fig.savefig(os.path.splitext(out_png)[0] + ".pdf")
    plt.close(fig)


def _write_metrics_txt(path: str, header: str, metrics: dict, extra_lines=None):
    """Write a compact TXT report (nice for paper artifact bundles)."""
    with open(path, "w") as f:
        f.write(f"=== {header} ===\n")
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")
        if extra_lines:
            f.write("\n")
            if isinstance(extra_lines, str):
                f.write(extra_lines)
            else:
                for line in extra_lines:
                    f.write(str(line).rstrip() + "\n")


def _safe_save_calibration_curve(y_true, y_prob_pos, out_dir: str, *, suffix: str, logger: logging.Logger):
    """
    Defensive wrapper around save_calibration_curve:
    - skips empty / size-mismatch / single-class label sets
    - never crashes the run (paper safety)
    """
    try:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_prob_pos, dtype=float)
    except Exception as e:
        logger.warning("[CALCURVE] %s skipped (could not coerce arrays): %s", suffix, e)
        return

    if yt.size == 0 or yp.size == 0:
        logger.warning("[CALCURVE] %s skipped (empty arrays).", suffix)
        return
    if yt.shape[0] != yp.shape[0]:
        logger.warning("[CALCURVE] %s skipped (size mismatch y_true=%d y_prob=%d).", suffix, int(yt.size), int(yp.size))
        return
    try:
        if int(np.unique(yt).size) < 2:
            logger.warning("[CALCURVE] %s skipped (single-class labels).", suffix)
            return
    except Exception:
        pass

    try:
        save_calibration_curve(yt.tolist(), yp.tolist(), out_dir, suffix=suffix)
    except Exception as e:
        logger.warning("[CALCURVE] %s failed: %s", suffix, e)


# ───────────────────────────────────────────────────────────
#  main
# ───────────────────────────────────────────────────────────
def main():
    """Entry point: run cross-patient K-fold CV, collect metrics/plots, then retrain on full data."""
    args = parse_args()
    run_id_original = str(getattr(args, "run_id", "") or "")
    args.run_id = _sanitize_run_id(run_id_original)
    # Output dir safety: if user did not pass --run_id and base output already contains artifacts,
    # auto-create a timestamped run_id to prevent accidental mixing.
    base_out = os.path.abspath(args.output_folder)
    os.makedirs(base_out, exist_ok=True)
    auto_run_id_applied = False
    if not args.run_id:
        existing = _list_existing_artifacts(base_out)
        if existing:
            auto_run_id_applied = True
            args.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%SZ")
    if args.run_id:
        args.output_folder = os.path.join(base_out, args.run_id)
    else:
        args.output_folder = base_out
    os.makedirs(args.output_folder, exist_ok=True)

    # Hard reset files that are appended fold-by-fold, to avoid mixing runs.
    multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
    if os.path.exists(multi_path):
        os.remove(multi_path)

    sum_multi_path = os.path.join(args.output_folder, "cv_metrics_summary_multi.csv")
    if os.path.exists(sum_multi_path):
        os.remove(sum_multi_path)

    

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_folder, "run.log"), mode="w"),
            logging.StreamHandler(),
        ],
        force=True,  
    )


    logger = logging.getLogger()

    if run_id_original and args.run_id != run_id_original:
        logger.warning("Sanitized run_id: %r -> %r", run_id_original, args.run_id)


    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Enable periodic traceback dumps (if requested)
    fh_faulthandler = _maybe_enable_faulthandler(args.output_folder, int(getattr(args, "hang_timeout", 0)), logger)

    if auto_run_id_applied:
        logger.warning(
            "Base output folder already contained artifacts; auto-created run_id=%s to avoid mixing runs.",
            args.run_id
        )

    if abs(float(args.threshold) - 0.5) > 1e-12:
        logger.warning("NOTE: --threshold is not used for prediction in this script (argmax rule). "
                       "For paper reproducibility, keep it at 0.5 to avoid confusion.")


    # Ensure embedding backend exists (UMAP is an optional dependency)
    args.embed_method = _ensure_embed_method_available(args.embed_method, logger)

    # Sanitize loader knobs (paper runs should never use negative workers)
    if int(args.num_workers) < 0:
        logger.warning("num_workers < 0 is invalid -> forcing num_workers=0")
        args.num_workers = 0


    # Persist CLI args for exact reproducibility
    try:
        with open(os.path.join(args.output_folder, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    except Exception as _e:
        logger.warning("Could not write run_args.json: %s", _e)
    

    # Reproducibility: seed Python/NumPy/PyTorch (CPU/CUDA)
    seed_everything(args.seed)


    # Parse budget_fracs once (used both in CV triage + FULL triage)
    budget_fracs = _parse_float_csv(args.budget_fracs)
    if budget_fracs:
        logger.info("[BUDGET] Using budget_fracs=%s", budget_fracs)
    else:
        logger.info("[BUDGET] budget_fracs not provided -> CV/FULL triage curves (multi-budget) will be skipped.")


    # Strong determinism defaults (best-effort; does not crash on non-deterministic ops)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Avoid TF32 numerical drift across GPUs (paper reproducibility).
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _e:
        logger.warning("Determinism flags could not be fully enabled: %s", _e)    
    
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Using device: %s | seed=%d", device, args.seed)
    
    # Save run metadata (useful for reproducibility in rebuttals/appendices)
    try:
        meta = {
            "run_id": getattr(args, "run_id", ""),
            "auto_run_id_applied": bool(auto_run_id_applied),
            "seed": args.seed,
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "psutil": getattr(psutil, "__version__", None),
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "command": " ".join([sys.executable] + sys.argv),
            "script_path": os.path.abspath(__file__),
            "script_sha256": _sha256_file(os.path.abspath(__file__)),
        }

        # sklearn version (best-effort)
        try:
            import sklearn
            meta["sklearn"] = sklearn.__version__
        except Exception:
            meta["sklearn"] = None

        if torch.cuda.is_available():
            try:
                meta["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                meta["gpu_name"] = None

        # Git state (best-effort; ignored if repo is not a git checkout)
        try:
            import subprocess
            meta["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
            ).strip()
            meta["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    text=True,
                ).strip()
            )
        except Exception:
            meta["git_commit"] = None
            meta["git_dirty"] = None


        with open(os.path.join(args.output_folder, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as _e:
        logger.warning("Could not write run_meta.json: %s", _e)

    # Environment snapshot
    if args.save_env:
        _try_write_pip_freeze(args.output_folder, logger)

    # ── Global collectors across folds ─────────────────────────────────    
    histories_all_folds = []
    # Per-fold predictions (for plots)
    y_true_all, y_pred_all = [], []
    y_prob_all_pos = []   # P(y=1) — for calibration curve
    y_conf_all_max = []   # max softmax — for uncertainty histogram
    y_conf_all = []       # (kept for compatibility, not used)
    all_uncertain = []

    # RAW baseline collectors for calibration (uncalibrated softmax)
    y_true_all_raw = []
    y_prob_all_pos_raw = []

    y_margin_all = []        


    # Optional streamed per-sample predictions CSV (can be huge; OFF by default unless explicitly enabled)
    pred_csv_fh = None
    pred_csv_writer = None
    pred_row_global = 0
    pred_csv_path = os.path.join(args.output_folder, "predictions_all_folds.csv")
    pred_csv_fields = [
        "fold","row_global","cal_method","T_fold","true",
        "pred_CAL","p1_CAL","p2_CAL","pmax_CAL","margin_CAL",
        "pred_RAW","p1_RAW","p2_RAW","pmax_RAW","margin_RAW",
    ]
    # Backwards-compatible behavior: keep CSV generation available, but guarded by a flag.
    # We default it to OFF because per-sample CSV can dominate runtime and appear as a "hang".
    if getattr(args, "save_pred_csv", False):
        pred_csv_fh = open(pred_csv_path, "w", newline="")
        pred_csv_writer = csv.writer(pred_csv_fh)
        pred_csv_writer.writerow(pred_csv_fields)
        logger.info("[PRED] Streaming per-sample predictions -> %s", pred_csv_path)
    else:
        logger.info("[PRED] Per-sample predictions CSV disabled (enable via --save-pred-csv).")



    # Per-fold metric collectors (paper tables)
    acc_ts_folds, auc_ts_folds = [], []
    ece_raw_folds, ece_ts_folds = [], []
    nll_raw_folds, nll_ts_folds = [], []
    br_raw_folds, br_ts_folds = [], []
    prec_ts_folds, rec_ts_folds, spec_ts_folds = [], [], []
    f1_ts_folds, balacc_ts_folds, ap_ts_folds = [], [], []
    fold_metrics_ext_rows = []
   

    # CV selective triage collectors (per-fold; thresholds selected on VAL, evaluated on TEST)
    cv_sel_budget_count_rows = []
    cv_triage_curve_rows = []


    # K-fold CV over patients/records (record-level)
    records = list(record_names)  # local copy; may be filtered (e.g., records with 0 windows)
    record_has_pos = []
    record_window_stats = []  # per-record window/positive counts for auditability
    for rec in records:
        try:
            _Xr, yr = load_mitbih_data(args.data_folder, [rec], WINDOW_SIZE, PRE_SAMPLES, FS)
            yr = np.asarray(yr, dtype=int)
            record_has_pos.append(int(np.any(yr == 1)))
            record_window_stats.append({
                "record": str(rec),
                "n_windows": int(len(yr)),
                "pos_windows": int((yr == 1).sum()),
                "neg_windows": int((yr == 0).sum()),
                "pos_frac": float((yr == 1).mean()) if len(yr) > 0 else float("nan"),
                "has_pos": int(np.any(yr == 1)),
            })
        except Exception as e:
            logger.warning("[CV] Failed to inspect record=%s for stratification (%s). Using has_pos=0.", rec, e)
            record_has_pos.append(0)
            record_window_stats.append({
                "record": str(rec),
                "n_windows": 0,
                "pos_windows": 0,
                "neg_windows": 0,
                "pos_frac": float("nan"),
                "has_pos": 0,
            })

    record_has_pos = np.asarray(record_has_pos, dtype=int)

    # drop records that yielded zero windows (prevents degenerate folds & confusing splits)
    try:
        valid_mask = np.asarray([int(s.get("n_windows", 0)) > 0 for s in record_window_stats], dtype=bool)
        if not bool(valid_mask.all()):
            dropped = [records[i] for i, ok in enumerate(valid_mask) if not ok]
            logger.warning("[DATA] Excluding %d record(s) with n_windows==0: %s", len(dropped), dropped)
            # Filter in-place
            records = [r for r, ok in zip(records, valid_mask) if ok]
            record_has_pos = record_has_pos[valid_mask]
            record_window_stats = [s for s, ok in zip(record_window_stats, valid_mask) if ok]
    except Exception as _e:
        logger.warning("[DATA] Could not filter empty records (continuing as-is): %s", _e)


    # Hard sanity checks (paper runs should fail fast with a readable error)
    if len(records) == 0:
        raise RuntimeError(
            "No records produced any windows after filtering (n_windows==0). "
            "Check --data_folder path and preprocessing."
        )
    if int(args.folds) > len(records):
        raise ValueError(
            f"Not enough non-empty records ({len(records)}) for --folds={args.folds}. "
            "Reduce --folds or fix dataset loading."
        )


    # stratify by pos_frac bins (reduces fold-to-fold prior shift vs has_pos)
    record_pos_frac = np.asarray([s.get("pos_frac", 0.0) for s in record_window_stats], dtype=float)
    record_strata, strat_desc = _make_pos_frac_strata(record_pos_frac, n_splits=int(args.folds), max_bins=4)

    # Store stratum id in stats CSV
    try:
        for s, b in zip(record_window_stats, record_strata.tolist()):
            s["strat_bin"] = int(b)
    except Exception:
        pass

    use_strat = (len(np.unique(record_strata)) > 1 and np.min(np.bincount(record_strata)) >= args.folds)

    if use_strat:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(records, record_strata)
        logger.info("[CV] Using StratifiedKFold over records; strat=%s counts=%s", strat_desc, np.bincount(record_strata).tolist())
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(records)
        logger.warning("[CV] Using plain KFold (stratification not possible) -> AUC may be NaN in some folds.")


    cv_record_splits = []

    for fold, (train_idx, test_idx) in enumerate(split_iter, 1):
        train_recs = [records[i] for i in train_idx]
        test_recs  = [records[i] for i in test_idx]
        logger.info("Fold %d  train=%s  test=%s", fold, train_recs, test_recs)

        cv_record_splits.append({
            "fold": int(fold),
            "train_records": train_recs,
            "test_records": test_recs,
            "n_train_records": int(len(train_recs)),
            "n_test_records": int(len(test_recs)),
        })

        # ── Load raw (real-valued) windows ─────────────────
        X_train, y_train = load_mitbih_data(
            args.data_folder, train_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )
        X_test, y_test = load_mitbih_data(
            args.data_folder, test_recs, WINDOW_SIZE, PRE_SAMPLES, FS
        )

        # Record sample-level stats in the split manifest
        try:
            stats_tr = _log_label_stats(logger, f"Fold {fold} TRAIN(raw)", y_train)
            stats_te = _log_label_stats(logger, f"Fold {fold} TEST(raw)",  y_test)
            cv_record_splits[-1].update({
                "n_train_windows": int(stats_tr["n"]),
                "n_test_windows": int(stats_te["n"]),
                "pos_train_windows": int(stats_tr["pos"]),
                "pos_test_windows": int(stats_te["pos"]),
            })
        except Exception as _e:
            logger.warning("[Fold %d] Could not log split label stats: %s", fold, _e)

        # ───────────────────────────────────────────────────
        #  CVNN + complex_stats (main article path)
        # ───────────────────────────────────────────────────
        X_tr_s = prepare_complex_input(X_train, method='complex_stats')
        X_te_s = prepare_complex_input(X_test,  method='complex_stats')

        # Split TRAIN → TRAIN/VAL (stratified)
        X_tr_s_tr, X_tr_s_val, y_train_tr, y_train_val = _safe_train_val_split(
            X_tr_s,
            y_train,
            test_size=0.2,
            seed=args.seed + 1000 + fold,
            logger=logger,
            tag=f"Fold {fold} TRAIN→VAL",
        )


        _log_label_stats(logger, f"Fold {fold} TRAIN(split)", y_train_tr)
        _log_label_stats(logger, f"Fold {fold} VAL(split)",   y_train_val)

        try:
            cv_record_splits[-1].update({
                "n_train_split": int(len(y_train_tr)),
                "n_val_split": int(len(y_train_val)),
                "pos_train_split": int((np.asarray(y_train_tr, dtype=int) == 1).sum()),
                "pos_val_split": int((np.asarray(y_train_val, dtype=int) == 1).sum()),
            })
        except Exception:
            pass

        # Build loaders: train/val/test (scaler fit on TRAIN only)
        tr_ld, val_ld, te_ld, scaler = _create_train_val_test_loaders_compat(
            X_tr_s_tr, y_train_tr,
            X_tr_s_val, y_train_val,
            X_te_s,     y_test,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        
        with open(os.path.join(args.output_folder, f"scaler_fold{fold}.pkl"), "wb") as f:
            pickle.dump(scaler, f)


        # Model
        model_s = SimpleComplexNet(
            in_features=X_tr_s.shape[1] // 2,
            hidden_features=64,
            out_features=2,
            bias=float(args.modrelu_bias),
        )
        model_s = model_s.to(device)
        t0 = time.perf_counter()
        history, best_s = train_real(
            model_s, tr_ld, val_ld, epochs=args.epochs, lr=args.lr, device=device
        )
        denom = max(len(history.get("train_loss", [])), 1)
        train_time_s = (time.perf_counter() - t0) / denom
        histories_all_folds.append(history)
        
        save_plots(history, args.output_folder, fold)

        # Save best weights per fold (paper auditability + enables post-hoc checks)
        try:
            best_cpu = {k: v.detach().cpu() for k, v in best_s.items()}
            torch.save(best_cpu, os.path.join(args.output_folder, f"best_model_fold{fold}.pt"))
        except Exception as _e:
            logger.warning("[Fold %d] Could not save best_model_fold weights: %s", fold, _e)

        # Resource snapshot (CPU RSS, optional GPU peak) per fold
        if psutil is not None:
            try:
                proc = psutil.Process(os.getpid())
                rss_mb = proc.memory_info().rss / (1024**2)
                if device.type == "cuda":
                    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                    torch.cuda.reset_peak_memory_stats(device)
                    logging.info(
                        f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | CPU_RSS={rss_mb:.1f}MB | GPU_peak={peak_mb:.1f}MB"
                    )
                else:
                    logging.info(f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | CPU_RSS={rss_mb:.1f}MB")
            except Exception as _e:
                logging.warning(f"[Fold {fold}] memory logging failed: {_e}")
        else:
            logging.info(f"[Fold {fold}] avg_epoch_time={train_time_s:.3f}s | psutil not installed -> skipping RSS stats")


        # Load best weights (by val acc)
        model_s.load_state_dict(best_s)

        # --- Choose calibration on VALIDATION (temperature / isotonic / platt / beta / vector / none) ---
        T_fold = None
        iso_cal = None
        cal_fn = None          # platt/beta/vector apply function from get_calibrator
        cal_kind = None        # "logits" or "probs"
        cal_tag = ""           # tag string from get_calibrator 
        is_binary = False

        main_cal = str(args.calibration).strip().lower()


        # Single-class VAL can happen in record-wise CV with extreme imbalance -> don't crash.
        try:
            _val_unique = np.unique(np.asarray(y_train_val, dtype=int))
            val_has_both = (_val_unique.size > 1)
        except Exception:
            val_has_both = True  # best-effort; let calibrator try

        if main_cal == "temperature":
            # temperature scaling: fast (logits cached), safe (positive + clamped), CVNN-correct.
            try:
                if not val_has_both:
                    logger.warning("[Fold %d] VAL has a single class -> using T=1.0 (no-op temperature).", fold)
                T_fold = _tune_temperature_safe(
                    model_s,
                    val_ld,
                    device=device,
                    logger=logger,
                    init_temp=float(getattr(args, "temp_init", 1.0)),
                    min_temp=float(getattr(args, "temp_min", 0.05)),
                    max_temp=float(getattr(args, "temp_max", 10.0)),
                    max_iter=int(getattr(args, "temp_max_iter", 50)),
                    tag=f"Fold {fold}",
                )
                torch.save(T_fold, os.path.join(args.output_folder, f"T_calib_fold{fold}.pt"))
                logging.info(f"[Fold {fold}] Using temperature T={float(T_fold.item()):.3f}")
            except Exception as e:
                logger.warning("[Fold %d] Safe temperature scaling failed (%s) -> using T=1.0.", fold, e)
                T_fold = torch.tensor(1.0, dtype=torch.float32)

        elif main_cal == "isotonic":
            if not val_has_both:
                logger.warning("[Fold %d] VAL has a single class -> isotonic calibration skipped; using RAW probs.", fold)
                iso_cal = None
            else:
                try:
                    iso_cal, is_binary = fit_isotonic_on_val(model_s, val_ld, device=device)
                    if not is_binary:
                        logging.warning("[Fold %d] Isotonic calibration skipped (multi-class).", fold)
                        iso_cal = None
                except Exception as e:
                    logger.warning("[Fold %d] Isotonic calibration failed (%s) -> using RAW probs.", fold, e)
                    iso_cal = None
        elif main_cal in ("platt", "beta", "vector"):
            if not val_has_both:
                logger.warning(
                    "[Fold %d] VAL has a single class -> %s calibration skipped; using RAW probs.",
                    fold, main_cal
                )
                cal_fn = None
            else:
                try:
                    cal_fn, cal_tag = _get_calibrator_compat(
                        model_s, val_ld, method=main_cal, device=device, seed=int(args.seed), logger=logger
                    )
                    if cal_fn is None:
                        logger.warning("[Fold %d] %s calibration unavailable -> using RAW probs.", fold, main_cal)
                    else:
                        cal_kind = "logits" if main_cal in ("platt", "vector") else "probs"
                        logger.info("[Fold %d] Using %s calibration (%s).", fold, main_cal, cal_tag)
                except Exception as e:
                    logger.warning("[Fold %d] %s calibration failed (%s) -> using RAW probs.", fold, main_cal, e)
                    cal_fn = None
        else:
            logging.info("[Fold %d] No calibration ('none').", fold)



        # --- Derived labels used later (CSV + fold tables) ---
        # MUST be defined for every fold (even when --save-pred-csv is OFF),
        # otherwise fold_metrics_ext_rows below will crash with UnboundLocalError.
        if main_cal == "temperature" and T_fold is not None:
            cal_suffix_fold = "TS"
            T_fold_val = float(T_fold.detach().cpu().item())
        elif main_cal == "isotonic" and iso_cal is not None:
            cal_suffix_fold = "ISO"
            T_fold_val = float("nan")
        elif main_cal == "platt" and cal_fn is not None:
            cal_suffix_fold = "PLATT"
            T_fold_val = float("nan")
        elif main_cal == "beta" and cal_fn is not None:
            cal_suffix_fold = "BETA"
            T_fold_val = float("nan")
        elif main_cal == "vector" and cal_fn is not None:
            cal_suffix_fold = "VECTOR"
            T_fold_val = float("nan")
        elif main_cal == "none":
            cal_suffix_fold = "NONE"
            T_fold_val = float("nan")
        else:
            # requested method but calibrator not available -> we effectively used RAW
            cal_suffix_fold = "RAW"
            T_fold_val = float("nan")



        # --- Per-fold evaluation: RAW vs CAL on TEST (CAL = method selected by --calibration) ---
        model_s.eval()
        y_true_raw, y_prob_pos_raw, y_max_raw = [], [], []
        y_true_ts,  y_prob_pos_ts,  y_max_ts  = [], [], []
        y_pred_ts = []
        probs_ts_all = []
        probs_raw_all = []
        yb_all = []
        logits_test_all = []   # cached TEST logits for multi-calibration (avoid second forward pass)


        
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(device)
                logits = complex_modulus_to_logits(model_s(xb))
                logits_cpu = logits.detach().cpu()
                logits_test_all.append(logits_cpu)

                # RAW probabilities (always computed)
                probs_raw = _ensure_prob_matrix_torch(torch.softmax(logits, dim=1))
                probs_raw_all.append(probs_raw.cpu())

                # CAL probabilities (single source of truth)
                probs_ts = _calibrate_probs_torch(
                    logits,
                    probs_raw,
                    method=main_cal,
                    T=T_fold,
                    iso_cal=iso_cal,
                    cal_fn=cal_fn,
                )

                probs_ts_all.append(probs_ts.cpu())
                yb_all.append(yb.cpu())

                # RAW collectors
                y_true_raw.extend(yb.tolist())
                y_prob_pos_raw.extend(probs_raw[:, 1].cpu().tolist())
                y_max_raw.extend(probs_raw.max(dim=1).values.cpu().tolist())

                # CAL collectors (historical *_ts variable names kept to minimize invasive refactors)
                y_true_ts.extend(yb.tolist())
                y_prob_pos_ts.extend(probs_ts[:, 1].cpu().tolist())
                y_max_ts.extend(probs_ts.max(dim=1).values.cpu().tolist())
                y_pred_ts.extend(probs_ts.argmax(dim=1).cpu().tolist())

         
        # Fold-level performance metrics (CAL probabilities; method selected by --calibration)
        fold_perf, _, _, _ = _compute_binary_metrics_and_curves(
            np.array(y_true_ts), np.array(y_pred_ts), np.array(y_prob_pos_ts)
        )
        acc_ts = float(fold_perf["Accuracy"])
        auc_ts = float(fold_perf["ROC_AUC"])
        ap_ts  = float(fold_perf["PR_AUC(AP)"])
        acc_ts_folds.append(acc_ts)
        auc_ts_folds.append(auc_ts)
        ap_ts_folds.append(ap_ts)
        prec_ts_folds.append(float(fold_perf["Precision"]))
        rec_ts_folds.append(float(fold_perf["Recall"]))
        spec_ts_folds.append(float(fold_perf["Specificity"]))
        f1_ts_folds.append(float(fold_perf["F1"]))
        balacc_ts_folds.append(float(fold_perf["BalancedAcc"]))
        
        try:
            roc_auc = save_confusion_roc(y_true_ts, y_pred_ts, y_prob_pos_ts, args.output_folder, fold)
            logging.info(f"[Fold {fold}] ROC AUC ({cal_suffix_fold}) = {roc_auc:.4f}")
        except Exception as e:
            logging.warning("[Fold %d] save_confusion_roc failed (likely single-class TEST): %s", fold, e)
            # Fallback: generate robust fold diagnostic (handles single-class gracefully)
            try:
                m_f, cm_f, roc_f, pr_f = _compute_binary_metrics_and_curves(
                    np.asarray(y_true_ts, dtype=int),
                    np.asarray(y_pred_ts, dtype=int),
                    np.asarray(y_prob_pos_ts, dtype=float),
                )
                out_png = os.path.join(args.output_folder, f"confusion_roc_fold{fold}.png")
                _save_confusion_roc_pr_figure(
                    cm_f, roc_f, pr_f, m_f, out_png,
                    title_prefix=f"Fold {fold} TEST",
                    save_pdf=bool(args.save_pdf),
                )
            except Exception as e2:
                logging.warning("[Fold %d] fallback confusion/ROC/PR failed: %s", fold, e2)


        # --- Per-sample rows for CSV + margin (TS) ---
        probs_ts_all  = torch.cat(probs_ts_all, dim=0)   # [N_test, 2]
        probs_raw_all = torch.cat(probs_raw_all, dim=0)  # [N_test, 2]
        yb_all        = torch.cat(yb_all, dim=0)         # [N_test]
        logits_test_all = torch.cat(logits_test_all, dim=0)  # [N_test, 2]


        # Compute margin in vectorized form (fast) and store for pooled histograms
        try:
            if probs_ts_all.shape[1] == 2:
                margin_np = (probs_ts_all[:, 1] - probs_ts_all[:, 0]).abs().numpy()
            else:
                top2 = torch.topk(probs_ts_all, k=2, dim=1).values
                margin_np = (top2[:, 0] - top2[:, 1]).abs().numpy()
            y_margin_all.extend(margin_np.tolist())
        except Exception:
            pass

        # Optional per-sample CSV export (STREAMING, no giant in-memory dicts)
        if pred_csv_writer is not None:
            t_write0 = time.perf_counter()
            probs_ts_np = probs_ts_all.numpy()
            probs_raw_np = probs_raw_all.numpy()
            y_np = yb_all.numpy().astype(int)

            pred_ts_np = probs_ts_np.argmax(axis=1).astype(int)
            pred_raw_np = probs_raw_np.argmax(axis=1).astype(int)

            pmax_ts_np = probs_ts_np.max(axis=1).astype(float)
            pmax_raw_np = probs_raw_np.max(axis=1).astype(float)

            if probs_ts_np.shape[1] == 2:
                margin_ts_np = np.abs(probs_ts_np[:, 1] - probs_ts_np[:, 0]).astype(float)
                margin_raw_np = np.abs(probs_raw_np[:, 1] - probs_raw_np[:, 0]).astype(float)
            else:
                part_ts = np.partition(probs_ts_np, -2, axis=1)[:, -2:]
                margin_ts_np = np.abs(part_ts[:, 1] - part_ts[:, 0]).astype(float)
                part_rw = np.partition(probs_raw_np, -2, axis=1)[:, -2:]
                margin_raw_np = np.abs(part_rw[:, 1] - part_rw[:, 0]).astype(float)

            n_rows = int(len(y_np))
            logger.info("[Fold %d] Writing predictions CSV: n=%d ...", fold, n_rows)
            # Chunked writes for responsiveness
            chunk = 200000
            for start in range(0, n_rows, chunk):
                end = min(start + chunk, n_rows)
                rows = []
                for i in range(start, end):
                    rg = pred_row_global + i
                    rows.append([
                        int(fold), int(rg), cal_suffix_fold, f"{T_fold_val:.6f}", int(y_np[i]),
                        int(pred_ts_np[i]),
                        f"{float(probs_ts_np[i, 0]):.12g}", f"{float(probs_ts_np[i, 1]):.12g}",
                        f"{float(pmax_ts_np[i]):.12g}", f"{float(margin_ts_np[i]):.12g}",
                        int(pred_raw_np[i]),
                        f"{float(probs_raw_np[i, 0]):.12g}", f"{float(probs_raw_np[i, 1]):.12g}",
                        f"{float(pmax_raw_np[i]):.12g}", f"{float(margin_raw_np[i]):.12g}",
                    ])
                pred_csv_writer.writerows(rows)
                pred_csv_fh.flush()
            pred_row_global += n_rows
            logger.info("[Fold %d] predictions CSV write done in %.2fs", fold, time.perf_counter() - t_write0)



        # Global collectors (CAL for figures)
        y_true_all.extend(y_true_ts)
        y_pred_all.extend(y_pred_ts)
        y_prob_all_pos.extend(y_prob_pos_ts)
        y_conf_all_max.extend(y_max_ts)
        
        # Global collectors (RAW for baseline reliability curve)
        y_true_all_raw.extend(y_true_raw)
        y_prob_all_pos_raw.extend(y_prob_pos_raw)

        # Per-fold metrics (RAW vs TS)
        ece_raw = expected_calibration_error(np.array(y_true_raw), np.array(y_prob_pos_raw), n_bins=int(args.ece_bins))
        ece_ts  = expected_calibration_error(np.array(y_true_ts),  np.array(y_prob_pos_ts),  n_bins=int(args.ece_bins))
        nll_raw = negative_log_likelihood(np.array(y_true_raw), np.array(y_prob_pos_raw))
        nll_ts  = negative_log_likelihood(np.array(y_true_ts),  np.array(y_prob_pos_ts))
        br_raw  = brier_score(np.array(y_true_raw), np.array(y_prob_pos_raw))
        br_ts   = brier_score(np.array(y_true_ts),  np.array(y_prob_pos_ts))

        ece_raw_folds.append(ece_raw); ece_ts_folds.append(ece_ts)
        nll_raw_folds.append(nll_raw); nll_ts_folds.append(nll_ts)
        br_raw_folds.append(br_raw);   br_ts_folds.append(br_ts)

        # ─────────────────────────────────────────────────────────────
        # CV selective triage (thresholds selected on VAL, evaluated on TEST)
        # ─────────────────────────────────────────────────────────────
        try:
            # Compute calibrated probabilities on VAL (same calibration as TEST in this fold)
            # and collect aligned X/y in the SAME order as probs (robust even if loader shuffles).
            probs_val_list = []
            x_val_list = []
            y_val_list = []
            model_s.eval()
            with torch.no_grad():
                for xb, yb in val_ld:
                    xb = xb.to(device)
                    x_val_list.append(xb.detach().cpu())
                    y_val_list.append(yb.detach().cpu())
                    logits_v = complex_modulus_to_logits(model_s(xb))
                    probs_raw_v = _ensure_prob_matrix_torch(torch.softmax(logits_v, dim=1))
                    probs_cal_v = _calibrate_probs_torch(
                        logits_v,
                        probs_raw_v,
                        method=main_cal,
                        T=T_fold,
                        iso_cal=iso_cal,
                        cal_fn=cal_fn,
                    )
                    probs_val_list.append(probs_cal_v.cpu())

            probs_val_np = torch.cat(probs_val_list, dim=0).numpy()
            X_val_np = torch.cat(x_val_list, dim=0).numpy()
            y_val_np = torch.cat(y_val_list, dim=0).numpy().astype(int)

            # Fixed exact-count budget on VAL (args.review_budget)
            bc = int(args.review_budget)
            if bc > 0:
                chosen_cv = _select_thresholds_budget_count_fast(
                    y_true=y_val_np, probs=probs_val_np, X=X_val_np, budget_count=bc
                )
                tau_cv = float(chosen_cv["tau"]); delta_cv = float(chosen_cv["delta"])

                sel_test = _selective_metrics_binary(
                    np.asarray(y_true_ts, dtype=int),
                    probs_ts_all.numpy(),
                    tau=tau_cv, delta=delta_cv
                )
                cv_sel_budget_count_rows.append({
                    "fold": int(fold),
                    "budget_count_val": int(bc),
                    "n_val": int(len(y_val_np)),
                    "tau": float(tau_cv),
                    "delta": float(delta_cv),
                    "val_abstain": float(chosen_cv.get("abstain", float("nan"))),
                    "val_capture": float(chosen_cv.get("capture", float("nan"))),
                    "val_risk_accept": float(chosen_cv.get("risk_accept", float("nan"))),
                    "test_coverage": float(sel_test.get("coverage", float("nan"))),
                    "test_abstain": float(sel_test.get("abstain", float("nan"))),
                    "test_risk_accept": float(sel_test.get("risk_accept", float("nan"))),
                    "test_capture": float(sel_test.get("capture", float("nan"))),
                    "test_precision_review": float(sel_test.get("precision_review", float("nan"))),
                    "test_fn_auto_rate": float(sel_test.get("fn_auto_rate", float("nan"))),
                    "test_fp_auto_rate": float(sel_test.get("fp_auto_rate", float("nan"))),
                })

            # Multi-budget triage curve (fractions on VAL; thresholds chosen on VAL; evaluated on TEST)
            if budget_fracs:
                n_val = int(len(y_val_np))
                for bf in budget_fracs:
                    bf = float(bf)
                    bc_bf = int(round(bf * n_val))
                    if bc_bf <= 0:
                        tau_b, delta_b = 0.0, 0.0
                    else:
                        chosen_b = _select_thresholds_budget_count_fast(
                            y_true=y_val_np, probs=probs_val_np, X=X_val_np, budget_count=bc_bf
                        )
                        tau_b = float(chosen_b["tau"]); delta_b = float(chosen_b["delta"])

                    sel_b = _selective_metrics_binary(
                        np.asarray(y_true_ts, dtype=int),
                        probs_ts_all.numpy(),
                        tau=float(tau_b), delta=float(delta_b)
                    )
                    cv_triage_curve_rows.append({
                        "fold": int(fold),
                        "budget_frac_val": float(bf),
                        "budget_count_val": int(bc_bf),
                        "tau": float(tau_b),
                        "delta": float(delta_b),
                        "test_coverage": float(sel_b.get("coverage", float("nan"))),
                        "test_abstain": float(sel_b.get("abstain", float("nan"))),
                        "test_risk_accept": float(sel_b.get("risk_accept", float("nan"))),
                        "test_capture": float(sel_b.get("capture", float("nan"))),
                        "test_precision_review": float(sel_b.get("precision_review", float("nan"))),
                        "test_fn_auto_rate": float(sel_b.get("fn_auto_rate", float("nan"))),
                        "test_fp_auto_rate": float(sel_b.get("fp_auto_rate", float("nan"))),
                    })
        except Exception as _e:
            logger.warning("[Fold %d] CV selective triage computation failed: %s", fold, _e)

        cal_suffix = _calib_suffix(args.calibration)
        logger.info(
            "[Fold %d] epoch_avg=%.2fs | params=%d | ECE raw=%.4f→%.4f %s | NLL raw=%.4f→%.4f | Brier raw=%.4f→%.4f",
            fold, train_time_s, sum(p.numel() for p in model_s.parameters()),
            ece_raw, ece_ts, cal_suffix, nll_raw, nll_ts, br_raw, br_ts
        )


        # Extended per-fold row 
        fold_metrics_ext_rows.append({
            "fold": int(fold),
            "calibration": cal_suffix_fold,
            "T_fold": float(T_fold_val),
            "n_test": int(fold_perf["N"]),
            "TN": int(fold_perf["TN"]),
            "FP": int(fold_perf["FP"]),
            "FN": int(fold_perf["FN"]),
            "TP": int(fold_perf["TP"]),
            "Acc": float(fold_perf["Accuracy"]),
            "AUC": float(fold_perf["ROC_AUC"]),
            "AP": float(fold_perf["PR_AUC(AP)"]),
            "Precision": float(fold_perf["Precision"]),
            "Recall": float(fold_perf["Recall"]),
            "Specificity": float(fold_perf["Specificity"]),
            "F1": float(fold_perf["F1"]),
            "BalancedAcc": float(fold_perf["BalancedAcc"]),
            "ECE_raw": float(ece_raw),
            "ECE_cal": float(ece_ts),
            "NLL_raw": float(nll_raw),
            "NLL_cal": float(nll_ts),
            "Brier_raw": float(br_raw),
            "Brier_cal": float(br_ts),
        })        

        # ========== multi-calibration block (runs in addition to RAW vs selected --calibration) ==========

        allowed_methods = {"platt", "beta", "vector", "temperature", "isotonic", "none", "raw"}
        methods = []
        for _m in str(args.calibs).split(","):
            _m = _m.strip().lower()
            if not _m:
                continue
            if _m not in allowed_methods:
                logger.warning("[Fold %d] Multi-calib: unknown method '%s' -> skipping.", fold, _m)
                continue
            if _m not in methods:
                methods.append(_m)

        # 1) Reuse cached TEST logits from the main TEST eval (avoid 2nd forward pass)
        logits_np_mc = logits_test_all.numpy()            # [N_test, C]
        y_np_mc = yb_all.numpy().astype(int)              # [N_test]
        # RAW probs already computed during eval
        probs_raw_np_mc = probs_raw_all.numpy()

        fold_rows = []  # rows for this fold across all methods

        # 2) Fit calibrator on VALIDATION per method, then apply to TEST
        for method in methods:
            logger.info("[Fold %d] Multi-calib: fitting method=%s ...", fold, method)
            t_fit = time.perf_counter()

            # 'none' / 'raw' = uncalibrated softmax (no fitting required).
            if method in ("none", "raw"):
                try:
                    probs_np = _ensure_prob_matrix_np(probs_raw_np_mc)
                    preds = probs_np.argmax(axis=1)
                    acc = float((preds == y_np_mc).mean())

                    ece = nll = brier = auc = float("nan")
                    if probs_np.shape[1] == 2:
                        try:
                            auc = float(roc_auc_score(y_np_mc, probs_np[:, 1]))
                        except Exception:
                            auc = float("nan")
                        try:
                            ece = expected_calibration_error(y_np_mc, probs_np[:, 1], n_bins=int(args.ece_bins))
                            nll = negative_log_likelihood(y_np_mc, probs_np[:, 1])
                            brier = brier_score(y_np_mc, probs_np[:, 1])
                        except Exception:
                            pass

                    recall_m = f1_m = ap_m = float("nan")
                    try:
                        if probs_np.shape[1] == 2:
                            m_perf, _, _, _ = _compute_binary_metrics_and_curves(y_np_mc, preds, probs_np[:, 1])
                            recall_m = float(m_perf.get("Recall", float("nan")))
                            f1_m = float(m_perf.get("F1", float("nan")))
                            ap_m = float(m_perf.get("PR_AUC(AP)", float("nan")))
                    except Exception:
                        pass

                    fold_rows.append({
                        "fold": fold,
                        "method": method.upper(),
                        "tag": "no_calibration",
                        "ECE": ece,
                        "NLL": nll,
                        "Brier": brier,
                        "Acc": acc,
                        "AUC": auc,
                        "Recall": recall_m,
                        "F1": f1_m,
                        "AP": ap_m,
                    })
                    logger.info(
                        "[Fold %d] Multi-calib: method=%s (raw) done in %.2fs",
                        fold, method, time.perf_counter() - t_fit
                    )
                except Exception as e:
                    logger.warning("[Fold %d] Multi-calib: RAW metrics failed: %s", fold, e)
                continue

            try:
                # Reuse already-fitted calibrators when possible (speed + identical results)
                apply_fn, tag = None, "reuse"
                # If this fold already fitted the SAME calibrator as --calibration, reuse it
                if method == main_cal and method in ("platt", "beta", "vector") and cal_fn is not None:
                    apply_fn = cal_fn
                    tag = f"reuse_main({cal_suffix_fold}:{cal_tag or 'cal'})"

                if method == "temperature" and T_fold is not None:
                    T_val = float(T_fold.detach().cpu().item())

                    def _apply_temp(logits_np, T=T_val):
                        z = logits_np / max(T, 1e-12)
                        z = z - z.max(axis=1, keepdims=True)
                        ez = np.exp(z)
                        return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, np.inf)

                    apply_fn, tag = _apply_temp, f"reuse_T={T_val:.4f}"

                elif method == "isotonic" and iso_cal is not None:

                    def _apply_iso(_logits_np=None, probs_np=None):
                        return iso_cal(probs_np)

                    apply_fn, tag = _apply_iso, "reuse_isotonic"

                elif method == "temperature":
                    # Fit a fresh temperature on VAL using the same safe routine as the main path
                    T_tmp = _tune_temperature_safe(
                        model_s,
                        val_ld,
                        device=device,
                        logger=logger,
                        init_temp=float(getattr(args, "temp_init", 1.0)),
                        min_temp=float(getattr(args, "temp_min", 0.05)),
                        max_temp=float(getattr(args, "temp_max", 10.0)),
                        max_iter=int(getattr(args, "temp_max_iter", 50)),
                        tag=f"Fold {fold} (multi-calib)",
                    )
                    T_val = float(T_tmp.detach().cpu().item())

                    def _apply_temp(logits_np, T=T_val):
                        z = logits_np / max(T, 1e-12)
                        z = z - z.max(axis=1, keepdims=True)
                        ez = np.exp(z)
                        return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, np.inf)

                    apply_fn, tag = _apply_temp, f"safe_T={T_val:.4f}"


                elif apply_fn is None:
                    apply_fn, tag = _get_calibrator_compat(
                        model_s, val_ld, method=method, device=device, seed=int(args.seed), logger=logger
                    )

            except Exception as e:
                logger.warning("[Fold %d] Multi-calib: method=%s fit failed: %s", fold, method, e)
                continue

            if apply_fn is None:
                logger.info("[Fold %d] Skipping method=%s (not applicable).", fold, method)
                continue

            logger.info(
                "[Fold %d] Multi-calib: method=%s fit done in %.2fs (%s)",
                fold, method, time.perf_counter() - t_fit, tag
            )

            # Apply on TEST + compute metrics (guard everything)
            try:
                if method in ("temperature", "vector", "platt"):
                    probs_np = apply_fn(logits_np_mc)  # expects logits
                elif method in ("isotonic", "beta"):
                    try:
                        probs_np = apply_fn(probs_np=probs_raw_np_mc)
                    except TypeError:
                        probs_np = apply_fn(probs_raw_np_mc)
                elif method == "none":
                    probs_np = probs_raw_np_mc
                else:
                    probs_np = probs_raw_np_mc  # safe fallback

                # Normalize / shape-safety
                try:
                    probs_np = _ensure_prob_matrix_np(probs_np)
                except Exception:
                    probs_np = _ensure_prob_matrix_np(probs_raw_np_mc)

                preds = probs_np.argmax(axis=1)
                acc = float((preds == y_np_mc).mean())

                # calibration metrics
                ece = nll = brier = auc = float("nan")
                if probs_np.shape[1] == 2:
                    try:
                        auc = float(roc_auc_score(y_np_mc, probs_np[:, 1]))
                    except Exception:
                        auc = float("nan")
                    try:
                        ece = expected_calibration_error(y_np_mc, probs_np[:, 1], n_bins=int(args.ece_bins))
                        nll = negative_log_likelihood(y_np_mc, probs_np[:, 1])
                        brier = brier_score(y_np_mc, probs_np[:, 1])
                    except Exception:
                        pass


                # performance metrics that matter under imbalance
                recall_m = f1_m = ap_m = float("nan")
                try:
                    if probs_np.shape[1] == 2:
                        m_perf, _, _, _ = _compute_binary_metrics_and_curves(y_np_mc, preds, probs_np[:, 1])
                        recall_m = float(m_perf.get("Recall", float("nan")))
                        f1_m = float(m_perf.get("F1", float("nan")))
                        ap_m = float(m_perf.get("PR_AUC(AP)", float("nan")))
                except Exception:
                    pass

                fold_rows.append({
                    "fold": fold,
                    "method": method.upper(),
                    "tag": tag,
                    "ECE": ece,
                    "NLL": nll,
                    "Brier": brier,
                    "Acc": acc,
                    "AUC": auc,
                    "Recall": recall_m,
                    "F1": f1_m,
                    "AP": ap_m,
                })


            except Exception as e:
                logger.warning("[Fold %d] Multi-calib: method=%s apply/metrics failed: %s", fold, method, e)
                continue


        # 3) Append to a fold-wise CSV (one file for the whole run)
        multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
        write_header = not os.path.exists(multi_path)
        with open(multi_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold","method","tag","ECE","NLL","Brier","Acc","AUC","Recall","F1","AP"])
            if write_header:
                w.writeheader()
            w.writerows(fold_rows)
        logging.info(f"[Fold {fold}] Wrote multi-calibration rows to {multi_path}")
        


        # Best-effort cleanup between folds
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Save record-level split manifest (for reproducibility / debugging)
    try:
        splits_path = os.path.join(args.output_folder, "cv_record_splits.json")
        with open(splits_path, "w") as f:
            json.dump(cv_record_splits, f, indent=2)
        logger.info("[CV] Saved record-level splits to %s", splits_path)
    except Exception as e:
        logger.warning("[CV] Failed to save cv_record_splits.json: %s", e)


    # fold-level window distribution table
    try:
        fold_stats_path = os.path.join(args.output_folder, "cv_fold_window_stats.csv")
        with open(fold_stats_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold","n_test_windows","pos_test_windows","pos_frac_test","test_records"])
            for s in cv_record_splits:
                n = int(s.get("n_test_windows", 0) or 0)
                p = int(s.get("pos_test_windows", 0) or 0)
                frac = (p / n) if n > 0 else float("nan")
                w.writerow([int(s["fold"]), n, p, f"{frac:.6f}", ",".join([str(x) for x in s.get("test_records", [])])])
        logger.info("[CV] Saved cv_fold_window_stats.csv -> %s", fold_stats_path)
    except Exception as e:
        logger.warning("[CV] Failed to save cv_fold_window_stats.csv: %s", e)

    # save CV selective triage artifacts (per-fold + summary)
    try:
        if cv_sel_budget_count_rows:
            p = os.path.join(args.output_folder, "cv_selective_budget_count_per_fold.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cv_sel_budget_count_rows[0].keys()))
                w.writeheader()
                w.writerows(cv_sel_budget_count_rows)
            logger.info("[CV] Saved %s", p)
        if cv_triage_curve_rows:
            p = os.path.join(args.output_folder, "cv_test_triage_curve_per_fold.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cv_triage_curve_rows[0].keys()))
                w.writeheader()
                w.writerows(cv_triage_curve_rows)
            logger.info("[CV] Saved %s", p)

        # aggregate CV triage curve across folds (mean ± CI95)
        try:
            if cv_triage_curve_rows:
                # group by budget_frac_val
                by_bf = {}
                for r in cv_triage_curve_rows:
                    bf = float(r.get("budget_frac_val", float("nan")))
                    if np.isnan(bf):
                        continue
                    by_bf.setdefault(bf, []).append(r)

                summary_rows = []
                for bf in sorted(by_bf.keys()):
                    rows = by_bf[bf]

                    cov_vals  = [_float_or_nan(rr.get("test_coverage", "nan")) for rr in rows]
                    risk_vals = [_float_or_nan(rr.get("test_risk_accept", "nan")) for rr in rows]
                    abst_vals = [_float_or_nan(rr.get("test_abstain", "nan")) for rr in rows]
                    capt_vals = [_float_or_nan(rr.get("test_capture", "nan")) for rr in rows]

                    cov_m,  cov_ci  = _mean_ci95_ignore_nan(cov_vals)
                    risk_m, risk_ci = _mean_ci95_ignore_nan(risk_vals)
                    abst_m, abst_ci = _mean_ci95_ignore_nan(abst_vals)
                    capt_m, capt_ci = _mean_ci95_ignore_nan(capt_vals)

                    summary_rows.append({
                        "budget_frac_val": bf,
                        "coverage_mean": cov_m, "coverage_CI95": cov_ci,
                        "risk_mean": risk_m, "risk_CI95": risk_ci,
                        "abstain_mean": abst_m, "abstain_CI95": abst_ci,
                        "capture_mean": capt_m, "capture_CI95": capt_ci,
                    })

                # write summary CSV
                if summary_rows:
                    out_sum = os.path.join(args.output_folder, "cv_test_triage_curve_summary.csv")
                    with open(out_sum, "w", newline="") as f:
                        fieldnames = list(summary_rows[0].keys())
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(summary_rows)
                    logger.info("[CV] Saved aggregated triage summary -> %s", out_sum)
                else:
                    logger.warning("[CV] Aggregated triage summary is empty -> skipping cv_test_triage_curve_summary.csv and plots.")


                # plots (risk-coverage, capture-abstain)
                if summary_rows:
                    # risk-coverage: x=coverage, y=risk
                    pts = [(r["coverage_mean"], r["risk_mean"], r["coverage_CI95"], r["risk_CI95"]) for r in summary_rows]
                    pts = [p for p in pts if not np.isnan(p[0]) and not np.isnan(p[1])]
                    if pts:
                        pts.sort(key=lambda x: x[0])
                        covs  = np.array([p[0] for p in pts], float)
                        risks = np.array([p[1] for p in pts], float)
                        xerr  = np.array([p[2] for p in pts], float)
                        yerr  = np.array([p[3] for p in pts], float)

                        fig, ax = plt.subplots(figsize=(5.2, 3.2))
                        ax.errorbar(covs, risks, xerr=xerr, yerr=yerr, marker="o", linewidth=1.2, capsize=3)
                        ax.set_xlabel("Coverage (CV mean)")
                        ax.set_ylabel("Risk among accepted (CV mean)")
                        ax.set_title("Selective risk–coverage (CV mean±CI95)")
                        fig.tight_layout()
                        p = os.path.join(args.output_folder, "cv_risk_coverage_mean_ci.png")
                        fig.savefig(p, dpi=300)
                        if args.save_pdf:
                            fig.savefig(os.path.splitext(p)[0] + ".pdf")
                        plt.close(fig)

                    # capture-abstain: x=abstain, y=capture
                    pts = [(r["abstain_mean"], r["capture_mean"], r["abstain_CI95"], r["capture_CI95"]) for r in summary_rows]
                    pts = [p for p in pts if not np.isnan(p[0]) and not np.isnan(p[1])]
                    if pts:
                        pts.sort(key=lambda x: x[0])
                        absts = np.array([p[0] for p in pts], float)
                        caps  = np.array([p[1] for p in pts], float)
                        xerr  = np.array([p[2] for p in pts], float)
                        yerr  = np.array([p[3] for p in pts], float)

                        fig, ax = plt.subplots(figsize=(5.2, 3.2))
                        ax.errorbar(absts, caps, xerr=xerr, yerr=yerr, marker="o", linewidth=1.2, capsize=3)
                        ax.set_xlabel("Abstain (CV mean)")
                        ax.set_ylabel("Error capture (CV mean)")
                        ax.set_title("Capture–abstain (CV mean±CI95)")
                        fig.tight_layout()
                        p = os.path.join(args.output_folder, "cv_capture_abstain_mean_ci.png")
                        fig.savefig(p, dpi=300)
                        if args.save_pdf:
                            fig.savefig(os.path.splitext(p)[0] + ".pdf")
                        plt.close(fig)

                    logger.info("[CV] Saved aggregated CV triage plots.")
        except Exception as e:
            logger.warning("[CV] Failed to aggregate CV triage curve/plots: %s", e)

    except Exception as e:
        logger.warning("[CV] Failed to save CV triage artifacts: %s", e)

    # Save per-record window stats (dataset auditability)
    try:
        rec_stats_path = os.path.join(args.output_folder, "record_window_stats.csv")
        with open(rec_stats_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(record_window_stats[0].keys()))
            w.writeheader()
            w.writerows(record_window_stats)
        logger.info("[CV] Saved record_window_stats.csv -> %s", rec_stats_path)
    except Exception as e:
        logger.warning("[CV] Failed to save record_window_stats.csv: %s", e)

    # Close streamed predictions CSV (if enabled)
    try:
        if pred_csv_fh is not None:
            pred_csv_fh.close()
            logger.info("[PRED] Closed predictions CSV (%d rows) -> %s", int(pred_row_global), pred_csv_path)
    except Exception as _e:
        logger.warning("[PRED] Could not close predictions CSV: %s", _e)

    # --- Save per-fold metrics for transparency (place right after the K-Fold loop) ---
    per_fold_path = os.path.join(args.output_folder, "cv_metrics_per_fold.csv")
    with open(per_fold_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold","ECE_raw","ECE_cal","NLL_raw","NLL_cal","Brier_raw","Brier_cal","Acc_cal","AUC_cal"])
        for i in range(len(ece_raw_folds)):
            w.writerow([i + 1,
                        f"{ece_raw_folds[i]:.6f}", f"{ece_ts_folds[i]:.6f}",
                        f"{nll_raw_folds[i]:.6f}", f"{nll_ts_folds[i]:.6f}",
                        f"{br_raw_folds[i]:.6f}",  f"{br_ts_folds[i]:.6f}",
                        f"{acc_ts_folds[i]:.6f}",  f"{auc_ts_folds[i]:.6f}"])

    logger.info("Saved per-fold CV metrics to %s", per_fold_path)


    # extended per-fold metrics (performance + calibration)
    if fold_metrics_ext_rows:
        ext_path = os.path.join(args.output_folder, "cv_metrics_per_fold_extended.csv")
        with open(ext_path, "w", newline="") as f:
            fieldnames = list(fold_metrics_ext_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(fold_metrics_ext_rows)
        logger.info("Saved extended per-fold metrics to %s", ext_path)

        # performance summary (mean ± CI95)
        perf_sum_path = os.path.join(args.output_folder, "cv_metrics_summary_perf.csv")
        with open(perf_sum_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Mean", "CI95"])
            for name, vals in [
                ("Acc", acc_ts_folds),
                ("AUC", auc_ts_folds),
                ("AP", ap_ts_folds),
                ("Precision", prec_ts_folds),
                ("Recall", rec_ts_folds),
                ("Specificity", spec_ts_folds),
                ("F1", f1_ts_folds),
                ("BalancedAcc", balacc_ts_folds),
            ]:
                mean, hw = _mean_ci95_ignore_nan(vals)
                w.writerow([name, f"{mean:.6f}", f"{hw:.6f}"])
        logger.info("Saved performance summary (mean±CI95) to %s", perf_sum_path)



    save_overall_history(histories_all_folds, args.output_folder)

    # Save reliability diagrams for baseline (RAW) and calibrated (CAL)
    _safe_save_calibration_curve(y_true_all_raw, y_prob_all_pos_raw, args.output_folder, suffix="RAW", logger=logger)
    # Avoid misleading duplicate "CAL" curves when calibration='none' (RAW already saved).
    if args.calibration != "none":
        _safe_save_calibration_curve(
            y_true_all,
            y_prob_all_pos,
            args.output_folder,
            suffix=_calib_suffix(args.calibration),
            logger=logger,
        )

    else:
        logger.info("Calibration='none' -> skipping second calibration curve (RAW already saved).")

    # pooled CV TEST diagnostics 
    try:
        metrics_cv, cm_cv, roc_cv, pr_cv = _compute_binary_metrics_and_curves(
            np.asarray(y_true_all, dtype=int),
            np.asarray(y_pred_all, dtype=int),
            np.asarray(y_prob_all_pos, dtype=float),
        )

        # Add pooled calibration metrics (RAW vs CAL) into the TXT
        try:
            ece_pool_raw = expected_calibration_error(np.asarray(y_true_all_raw, dtype=int), np.asarray(y_prob_all_pos_raw, dtype=float), n_bins=int(args.ece_bins))
            ece_pool_cal = expected_calibration_error(np.asarray(y_true_all, dtype=int), np.asarray(y_prob_all_pos, dtype=float), n_bins=int(args.ece_bins))
            nll_pool_raw = negative_log_likelihood(np.asarray(y_true_all_raw, dtype=int), np.asarray(y_prob_all_pos_raw, dtype=float))
            nll_pool_cal = negative_log_likelihood(np.asarray(y_true_all, dtype=int), np.asarray(y_prob_all_pos, dtype=float))
            br_pool_raw  = brier_score(np.asarray(y_true_all_raw, dtype=int), np.asarray(y_prob_all_pos_raw, dtype=float))
            br_pool_cal  = brier_score(np.asarray(y_true_all, dtype=int), np.asarray(y_prob_all_pos, dtype=float))
            cal_lines = [
                "--- pooled calibration (CV TEST pooled) ---",
                f"ECE_raw={ece_pool_raw:.6f}",
                f"ECE_cal={ece_pool_cal:.6f}",
                f"NLL_raw={nll_pool_raw:.6f}",
                f"NLL_cal={nll_pool_cal:.6f}",
                f"Brier_raw={br_pool_raw:.6f}",
                f"Brier_cal={br_pool_cal:.6f}",
            ]
        except Exception as _e:
            cal_lines = [f"(pooled calibration metrics unavailable: {_e})"]


        cv_txt = os.path.join(args.output_folder, "cv_test_metrics.txt")
        _write_metrics_txt(
            cv_txt,
            header=f"CV pooled TEST metrics ({args.folds}-fold) [calibration={args.calibration}]",
            metrics=metrics_cv,
            extra_lines=[
                f"seed={args.seed}",
                f"device={device}",
                *cal_lines,
            ],
        )
        cv_png = os.path.join(args.output_folder, "confusion_roc_cv.png")
        _save_confusion_roc_pr_figure(cm_cv, roc_cv, pr_cv, metrics_cv, cv_png, title_prefix="CV pooled TEST", save_pdf=bool(args.save_pdf))
        logger.info("Saved pooled CV diagnostics: %s and %s", cv_txt, cv_png)
    except Exception as e:
        logger.warning("Failed to save pooled CV diagnostics: %s", e)

    # pooled CV uncertainty histograms (pmax + margin)
    try:
        # pmax
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.hist(np.asarray(y_conf_all_max, dtype=float), bins=30, alpha=0.75)
        cal_suffix = _calib_suffix(args.calibration)
        ax.set_ylabel("Count")
        ax.set_xlabel(f"max softmax probability pmax (CV pooled, {cal_suffix})")
        ax.set_xlim(0.5, 1.0)
        fig.tight_layout()
        outp = os.path.join(args.output_folder, "uncertainty_hist_cv.png")
        fig.savefig(outp, dpi=300)
        if args.save_pdf:
            fig.savefig(os.path.splitext(outp)[0] + ".pdf")
        plt.close(fig)

        # margin
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.hist(np.asarray(y_margin_all, dtype=float), bins=30, alpha=0.75)
        ax.set_xlabel(f"margin |p1-p2| (CV pooled, {cal_suffix})")
        ax.set_ylabel("Count")
        fig.tight_layout()
        outm = os.path.join(args.output_folder, "uncertainty_margin_hist_cv.png")
        fig.savefig(outm, dpi=300)
        if args.save_pdf:
            fig.savefig(os.path.splitext(outm)[0] + ".pdf")
        plt.close(fig)
        logger.info("Saved pooled CV uncertainty histograms.")
    except Exception as e:
        logger.warning("Failed to save pooled CV uncertainty histograms: %s", e)

    # persist lightweight arrays for downstream analysis
    if args.save_arrays:
        try:
            np.savez_compressed(
                os.path.join(args.output_folder, "cv_test_arrays.npz"),
                y_true=np.asarray(y_true_all, dtype=int),
                y_pred=np.asarray(y_pred_all, dtype=int),
                prob_pos=np.asarray(y_prob_all_pos, dtype=float),
                pmax=np.asarray(y_conf_all_max, dtype=float),
                margin=np.asarray(y_margin_all, dtype=float),
                y_true_raw=np.asarray(y_true_all_raw, dtype=int),
                prob_pos_raw=np.asarray(y_prob_all_pos_raw, dtype=float),
                calibration=args.calibration,
                seed=int(args.seed),
                folds=int(args.folds),
            )
            logger.info("Saved cv_test_arrays.npz")
        except Exception as e:
            logger.warning("Failed to save cv_test_arrays.npz: %s", e)    


    def _row(name, raw_list, ts_list):
        """Helper: summary row with mean, 95% CI half-width, and relative drop from RAW to CAL."""
        m_raw, ci_raw = mean_ci(raw_list)
        m_ts,  ci_ts  = mean_ci(ts_list)
        rel_drop = (m_raw - m_ts) / max(m_raw, 1e-12)
        return [name, m_raw, ci_raw, m_ts, ci_ts, rel_drop]

    table = [
        _row("ECE", ece_raw_folds, ece_ts_folds),
        _row("NLL", nll_raw_folds, nll_ts_folds),
        _row("Brier", br_raw_folds, br_ts_folds),
    ]

    with open(os.path.join(args.output_folder, "cv_metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "RAW_mean", "RAW_CI95", "CAL_mean", "CAL_CI95", "Relative_drop"])
        w.writerows(table)

    logger.info("Saved CV metrics summary with 95%% CI to %s",
                os.path.join(args.output_folder, "cv_metrics_summary.csv"))

    # ========== global summary for multi-calibration ==========
    multi_path = os.path.join(args.output_folder, "cv_metrics_per_fold_multi.csv")
    if os.path.exists(multi_path):
        try:
            rows = []
            with open(multi_path, "r", newline="") as f:
                r = csv.DictReader(f)
                rows = list(r)

            metrics_names = ["ECE","NLL","Brier","Acc","AUC","Recall","F1","AP"]
            by_method = {}
            for rr in rows:
                m = rr.get("method", "UNKNOWN")
                if m not in by_method:
                    by_method[m] = {k: [] for k in metrics_names}
                for k in metrics_names:
                    by_method[m][k].append(_float_or_nan(rr.get(k, "nan")))

            out_rows = []
            for m in sorted(by_method.keys()):
                for k in metrics_names:
                    mean, hw = _mean_ci95_ignore_nan(by_method[m][k])
                    out_rows.append([m, k, mean, hw])

            path_sum_multi = os.path.join(args.output_folder, "cv_metrics_summary_multi.csv")
            with open(path_sum_multi, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Method","Metric","Mean","CI95"])
                w.writerows(out_rows)
            logging.info("Saved multi-calibration summary to %s", path_sum_multi)
        except Exception as e:
            logging.warning("Failed multi-calibration summary: %s", e)



    # Keep scatter consistent with features used during CV (complex_stats)
    X_full, y_full = load_mitbih_data(args.data_folder, records, WINDOW_SIZE, PRE_SAMPLES, FS)
    # Subsample BEFORE feature extraction to avoid multi-hour TSNE/UMAP on huge datasets
    X_full, y_full, idx_embed = _subsample_xy(np.asarray(X_full), np.asarray(y_full), int(args.embed_max_points), int(args.seed))
    if idx_embed is not None:
        try:
            np.save(os.path.join(args.output_folder, "embed_subset_indices.npy"), idx_embed)
            logger.info("[EMBED] Subsampled embedding points: n=%d (saved embed_subset_indices.npy)", int(len(idx_embed)))
        except Exception:
            logger.info("[EMBED] Subsampled embedding points: n=%d", int(len(idx_embed)))
    X_full_c = prepare_complex_input(X_full, method='complex_stats')
    # Always save a deterministic PCA scatter
    try:
        _save_joint_pca_scatter(X_full_c, y_full, args.output_folder, save_pdf=bool(args.save_pdf))
        logger.info("[EMBED] Saved joint PCA scatter -> complex_PCA_scatter.png")
    except Exception as e:
        logger.warning("Joint PCA scatter failed: %s", e)

    # Optional: additional qualitative embedding for appendices.
    if str(args.embed_method).strip().lower() in ("tsne", "umap"):
        try:
            _save_embedding_scatter(
                X_full_c, y_full, args.output_folder,
                method=str(args.embed_method).strip().lower(),
                seed=int(args.seed),
                save_pdf=bool(args.save_pdf),
                logger=logger,
            )
            logger.info("[EMBED] Saved %s scatter.", str(args.embed_method).upper())
        except Exception as e:
            logger.warning("[EMBED] %s scatter failed: %s", str(args.embed_method).upper(), e)

    if not args.full_retrain:
        logger.info("Skipping FULL retrain stage (--no-full-retrain).")

        try:
            _write_artifact_readme(args.output_folder, args, logger)
        except Exception as _e:
            logger.warning("Failed to write ARTIFACTS.md (CV-only): %s", _e)

        try:
            _write_artifact_manifest(args.output_folder, logger, hash_files=bool(args.hash_manifest))
        except Exception as _e:
            logger.warning("Failed to write artifact_manifest.json (CV-only): %s", _e)

        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        try:
            if fh_faulthandler is not None:
                fh_faulthandler.close()
        except Exception:
            pass
        logging.shutdown()
        return    
    
    # Retrain on full dataset with proper train/val/test split
    # =========================
    # FULL model retrain (patient-wise record split, no leakage)
    # =========================
    logger.info("Retraining FULL model with patient-wise record split (no leakage)…")

    # Reconstruct the same record partitions used in CV (deterministic)
    full_test_fold = int(args.full_test_fold)
    full_val_fold  = int(args.full_val_fold)


    # optionally auto-select FULL folds to avoid extreme prior shift (label-only heuristic)
    if getattr(args, "auto_full_folds", False):
        try:
            # per-fold test pos_frac (from CV splits)
            fold_frac = {}
            total_n = 0
            total_p = 0
            for s in cv_record_splits:
                f = int(s["fold"])
                n = int(s.get("n_test_windows", 0) or 0)
                p = int(s.get("pos_test_windows", 0) or 0)
                total_n += n
                total_p += p
                fold_frac[f] = (p / n) if n > 0 else float("nan")

            overall = (total_p / total_n) if total_n > 0 else float("nan")
            best = None
            for i in range(1, args.folds + 1):
                for j in range(1, args.folds + 1):
                    if i == j:
                        continue
                    fi = fold_frac.get(i, float("nan"))
                    fj = fold_frac.get(j, float("nan"))
                    if np.isnan(fi) or np.isnan(fj) or np.isnan(overall):
                        continue
                    score = abs(fi - overall) + abs(fj - overall) + abs(fi - fj)
                    if best is None or score < best[0]:
                        best = (score, i, j, fi, fj, overall)

            if best is not None:
                _, full_test_fold, full_val_fold, fi, fj, ov = best
                logger.warning(
                    "[FULL] --auto-full-folds enabled. Selected test_fold=%d (pos_frac=%.4f), "
                    "val_fold=%d (pos_frac=%.4f), overall_pos_frac≈%.4f",
                    full_test_fold, fi, full_val_fold, fj, ov
                )
                args.full_test_fold = int(full_test_fold)
                args.full_val_fold = int(full_val_fold)
        except Exception as e:
            logger.warning("[FULL] auto fold selection failed: %s", e)

    if args.folds < 3:
        raise ValueError("For a disjoint train/val/test record split in FULL retrain, set --folds >= 3.")
    if full_test_fold == full_val_fold:
        raise ValueError("--full-test-fold and --full-val-fold must be different.")
    if not (1 <= full_test_fold <= args.folds) or not (1 <= full_val_fold <= args.folds):
        raise ValueError(f"Full split fold indices must be in [1, {args.folds}]. Got test={full_test_fold}, val={full_val_fold}.")

    # Build fold partitions again
    parts = []
    if use_strat:
        splitter2 = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for _, te_idx in splitter2.split(records, record_strata):
            parts.append(list(te_idx))
    else:
        splitter2 = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for _, te_idx in splitter2.split(records):
            parts.append(list(te_idx))

    te_set = set(parts[full_test_fold - 1])
    va_set = set(parts[full_val_fold - 1])
    tr_set = set(range(len(records))) - te_set - va_set

    train_recs_full = [records[i] for i in sorted(tr_set)]
    val_recs_full   = [records[i] for i in sorted(va_set)]
    test_recs_full  = [records[i] for i in sorted(te_set)]

    # Save manifest
    try:
        with open(os.path.join(args.output_folder, "full_split_records.json"), "w") as f:
            json.dump({
                "seed": args.seed,
                "folds": args.folds,
                "full_test_fold": full_test_fold,
                "full_val_fold": full_val_fold,
                "train_records": train_recs_full,
                "val_records": val_recs_full,
                "test_records": test_recs_full,
            }, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write full_split_records.json: %s", e)

    # Load windows record-wise
    X_tr_raw, y_tr = load_mitbih_data(args.data_folder, train_recs_full, WINDOW_SIZE, PRE_SAMPLES, FS)
    X_va_raw, y_va = load_mitbih_data(args.data_folder, val_recs_full,   WINDOW_SIZE, PRE_SAMPLES, FS)
    X_te_raw, y_te = load_mitbih_data(args.data_folder, test_recs_full,  WINDOW_SIZE, PRE_SAMPLES, FS)
    _log_label_stats(logger, "FULL TRAIN(raw)", y_tr)
    _log_label_stats(logger, "FULL VAL(raw)",   y_va)
    _log_label_stats(logger, "FULL TEST(raw)",  y_te)

    # Feature extraction
    X_tr = prepare_complex_input(X_tr_raw, method='complex_stats')
    X_va = prepare_complex_input(X_va_raw, method='complex_stats')
    X_te = prepare_complex_input(X_te_raw, method='complex_stats')

    tr_all, va_all, te_all, scaler_full = _create_train_val_test_loaders_compat(
        X_tr, y_tr,
        X_va, y_va,
        X_te, y_te,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    with open(os.path.join(args.output_folder, "scaler_full.pkl"), "wb") as f:
        pickle.dump(scaler_full, f)



    model_all = SimpleComplexNet(
        in_features=X_tr.shape[1] // 2, hidden_features=64, out_features=2, bias=float(args.modrelu_bias)
    )
    model_all = model_all.to(device)
    hist_all, best_all = train_real(model_all, tr_all, va_all, epochs=args.epochs, lr=args.lr, device=device)
    try:
        best_all_cpu = {k: v.detach().cpu() for k, v in best_all.items()}
        torch.save(best_all_cpu, os.path.join(args.output_folder, 'best_model_full.pt'))
    except Exception:
        torch.save(best_all, os.path.join(args.output_folder, 'best_model_full.pt'))
    save_plots(hist_all, args.output_folder, 'full')
    logger.info('Saved full-model and plots')

    # Calibration on VALIDATION (default: Platt scaling; see --calibration)
    model_all.load_state_dict(best_all)
    T_full = None
    iso_full = None
    is_binary = False

    cal_full_fn = None
    cal_full_kind = None
    cal_full_tag = ""
    main_cal = str(args.calibration).strip().lower()


    try:
        _val_unique_full = np.unique(np.asarray(y_va, dtype=int))
        val_has_both_full = (_val_unique_full.size > 1)
    except Exception:
        val_has_both_full = True

    if main_cal == "temperature":
        try:
            if not val_has_both_full:
                logger.warning("[FULL] VAL has a single class -> using T=1.0 (no-op temperature).")
            T_full = _tune_temperature_safe(
                model_all,
                va_all,
                device=device,
                logger=logger,
                init_temp=float(getattr(args, "temp_init", 1.0)),
                min_temp=float(getattr(args, "temp_min", 0.05)),
                max_temp=float(getattr(args, "temp_max", 10.0)),
                max_iter=int(getattr(args, "temp_max_iter", 50)),
                tag="FULL",
            )
            torch.save(T_full, os.path.join(args.output_folder, "T_calib.pt"))
            logger.info(f"[INFO] Full-model temperature T={float(T_full.item()):.3f} saved to {args.output_folder}")
        except Exception as e:
            logger.warning("[FULL] Safe temperature scaling failed (%s) -> using T=1.0.", e)
            T_full = torch.tensor(1.0, dtype=torch.float32)

    elif main_cal == "isotonic":
        if not val_has_both_full:
            logger.warning("[FULL] VAL has a single class -> isotonic calibration skipped; using RAW probs.")
            iso_full = None
        else:
            try:
                iso_full, is_binary = fit_isotonic_on_val(model_all, va_all, device=device)
                logger.info("[INFO] Full-model isotonic calibration %s", "enabled" if is_binary else "skipped (multi-class)")
                if not is_binary:
                    iso_full = None
            except Exception as e:
                logger.warning("[FULL] Isotonic calibration failed (%s) -> using RAW probs.", e)
                iso_full = None

    elif main_cal in ("platt", "beta", "vector"):
        if not val_has_both_full:
            logger.warning("[FULL] VAL has a single class -> %s calibration skipped; using RAW probs.", main_cal)
            cal_full_fn = None
        else:
            try:
                cal_full_fn, cal_full_tag = _get_calibrator_compat(
                    model_all, va_all, method=main_cal, device=device, seed=int(args.seed), logger=logger
                )
                if cal_full_fn is None:
                    logger.warning("[FULL] %s calibration unavailable -> using RAW probs.", main_cal)
                else:
                    cal_full_kind = "logits" if main_cal in ("platt", "vector") else "probs"
                    logger.info("[FULL] Using %s calibration (%s).", main_cal, cal_full_tag)
            except Exception as e:
                logger.warning("[FULL] %s calibration failed (%s) -> using RAW probs.", main_cal, e)
                cal_full_fn = None

    else:
        logger.info("[INFO] No calibration ('none') on full model.")



    # Collect calibrated probabilities on VALIDATION (needed to pick tau*, delta*) in a vectorized way
    yva_chunks, pva_chunks, Xva_chunks = [], [], []
    pva_raw_chunks = []
    lva_chunks = []

    model_all.eval()
    with torch.no_grad():
        for xb, yb in va_all:
            logits = complex_modulus_to_logits(model_all(xb.to(device)))
            lva_chunks.append(logits.detach().cpu())

            probs_raw = _ensure_prob_matrix_torch(torch.softmax(logits, dim=1))
            pva_raw_chunks.append(probs_raw.detach().cpu())

            probs = _calibrate_probs_torch(
                logits,
                probs_raw,
                method=main_cal,
                T=T_full,
                iso_cal=iso_full,
                cal_fn=cal_full_fn,
            )


            yva_chunks.append(yb.detach().cpu())
            pva_chunks.append(probs.detach().cpu())
            Xva_chunks.append(xb.detach().cpu())

    yva = torch.cat(yva_chunks, dim=0).numpy().astype(int)
    pva = torch.cat(pva_chunks, dim=0).numpy().astype(float)
    Xva = torch.cat(Xva_chunks, dim=0).numpy().astype(float)

    pva_raw = torch.cat(pva_raw_chunks, dim=0).numpy().astype(np.float32)
    lva = torch.cat(lva_chunks, dim=0).numpy().astype(np.float32)



    if args.save_arrays:
        try:
            np.savez_compressed(
                os.path.join(args.output_folder, "full_val_arrays.npz"),
                y_true=yva.astype(int),
                probs=pva.astype(np.float32),
                probs_raw=pva_raw,
                logits_raw=lva,
                calibration=args.calibration,
                seed=int(args.seed),
            )
            logger.info("Saved full_val_arrays.npz")
        except Exception as e:
            logger.warning("Failed to save full_val_arrays.npz: %s", e)    

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
        
        # === enriched grid with coverage + exact counts + clinically-relevant auto FN/FP ===
        sens_csv_ext = os.path.join(args.output_folder, "sens_grid_ext.csv")
        with open(sens_csv_ext, "w", newline="") as f:
            fieldnames = [
                "tau","delta",
                # original grid metrics (as returned)
                "abstain_grid","capture_grid","precision_grid","dispersion_grid","risk_accept_grid","kink",
                # exact recompute 
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
    chosen = _select_thresholds_budget_count_fast(
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


    
    # === thresholds for multiple review budgets (fractional) ===
    budget_fracs = _parse_float_csv(args.budget_fracs)
    budget_plans = []  # store tau/delta for later TEST evaluation
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
                    chosen_b = _select_thresholds_budget_count_fast(
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

                budget_plans.append({
                    "budget_frac": float(bf),
                    "budget_count": int(bc),
                    "tau": float(tau_b),
                    "delta": float(delta_b),
                    "kink_score_nearest": float(kink_b) if not np.isnan(kink_b) else float("nan"),
                })

        logging.info("Saved multi-budget (tau*,delta*) table to %s", out_path)

    # --- FULL TEST: calibrated probabilities in a stable order (also used for triage/uncertainty) ---
    X_te_scaled_t, y_te_t = te_all.dataset.tensors
    X_te_scaled_t = X_te_scaled_t.detach().cpu()
    y_te_t = y_te_t.detach().cpu()
    X_te_np = X_te_scaled_t.numpy()

    logits_te = []
    model_all.eval()
    with torch.no_grad():
        for start in range(0, len(X_te_scaled_t), args.batch_size):
            xb = X_te_scaled_t[start:start + args.batch_size].to(device)
            lo = complex_modulus_to_logits(model_all(xb))
            logits_te.append(lo.detach().cpu())
    logits_te = torch.cat(logits_te, dim=0)  # [N_test, 2]

    probs_raw_te = torch.softmax(logits_te, dim=1)
    probs_raw_te = _ensure_prob_matrix_torch(probs_raw_te)
    probs_te = _calibrate_probs_torch(
        logits_te,
        probs_raw_te,
        method=main_cal,
        T=T_full,
        iso_cal=iso_full,
        cal_fn=cal_full_fn,
    )
    if main_cal == "isotonic" and (iso_full is None):
        logger.warning("[FULL] Isotonic selected but calibrator unavailable -> using RAW probabilities.")

    probs_te_np = probs_te.numpy()
    y_true_test = y_te_t.numpy().astype(int)
    y_pred_test = probs_te_np.argmax(axis=1).astype(int)
    y_prob_pos_test = probs_te_np[:, 1].astype(float) if probs_te_np.shape[1] >= 2 else probs_te_np[:, 0].astype(float)

    y_conf_test = probs_te_np.max(axis=1).astype(float)
    if probs_te_np.shape[1] == 2:
        y_margin_test = np.abs(probs_te_np[:, 1] - probs_te_np[:, 0]).astype(float)
    else:
        part = np.partition(probs_te_np, -2, axis=1)[:, -2:]
        y_margin_test = np.abs(part[:, 1] - part[:, 0]).astype(float)
    n_classes = int(probs_te_np.shape[1])


    # Optional
    test_rec_order = None
    test_cum_ends = None
    try:
        rec2nwin = {str(s.get("record")): int(s.get("n_windows", 0) or 0) for s in record_window_stats}
        cum = 0
        ends = []
        order = []
        for rec in test_recs_full:
            nrec = int(rec2nwin.get(str(rec), 0))
            cum += nrec
            ends.append(int(cum))
            order.append(str(rec))

        if cum == int(len(y_true_test)):
            test_rec_order = order
            test_cum_ends = ends
            map_path = os.path.join(args.output_folder, "full_test_index_map.json")
            with open(map_path, "w") as f:
                json.dump(
                    {"test_records": order, "cum_ends": ends, "n_total_test": int(len(y_true_test))},
                    f,
                    indent=2
                )
            logger.info("[FULL] Saved index→record map -> %s", map_path)

            # Convenience export: row-wise TEST table (join-friendly: record/window/index + probs + features).
            # Redundant with full_test_arrays.npz
            try:
                rec_list: List[int] = []
                win_list: List[int] = []
                prev_end = 0
                for rec, end in zip(test_rec_order, test_cum_ends):
                    n = int(end - prev_end)
                    rec_list.extend([int(rec)] * n)
                    win_list.extend(list(range(n)))
                    prev_end = end
                if len(rec_list) != int(X_te_np.shape[0]):
                    raise RuntimeError(f"Record/window map length mismatch: {len(rec_list)} != {int(X_te_np.shape[0])}")

                full_test_df = pd.DataFrame(
                    {
                        "run_id": str(run_id),
                        "seed": int(args.seed),
                        "calibration": str(args.calibration),
                        "index": np.arange(int(X_te_np.shape[0]), dtype=int),
                        "record": rec_list,
                        "window_in_record": win_list,
                        "true_label": y_te_np.astype(int),
                        "pred": y_pred_test.astype(int),
                        "is_error": (y_pred_test != y_te_np).astype(int),
                        "p1": probs_te_np[:, 0],
                        "p2": probs_te_np[:, 1],
                        "pmax": y_conf_test,
                        "margin": y_margin_test,
                        "x0": X_te_np[:, 0],
                        "x1": X_te_np[:, 1],
                        "x2": X_te_np[:, 2],
                        "x3": X_te_np[:, 3],
                    }
                )
                out_csv = os.path.join(args.output_folder, "full_test_predictions_ext.csv")
                full_test_df.to_csv(out_csv, index=False)
                logger.info("[FULL] Wrote %s", out_csv)
            except Exception as e:
                logger.warning("[FULL] Could not write full_test_predictions_ext.csv: %s", e)

        else:
            logger.warning(
                "[FULL] index→record map not saved: sum(n_windows on test records)=%d != N_test=%d",
                cum, int(len(y_true_test))
            )
    except Exception as _e:
        logger.warning("[FULL] index→record map build failed: %s", _e)


    if args.save_arrays:
        try:
            np.savez_compressed(
                os.path.join(args.output_folder, "full_test_arrays.npz"),
                y_true=y_true_test.astype(int),
                probs=probs_te_np.astype(float),
                pmax=y_conf_test.astype(float),
                margin=y_margin_test.astype(float),
                tau_star=float(tau_star),
                delta_star=float(delta_star),
                calibration=args.calibration,
                seed=int(args.seed),
                logits_raw=logits_te.numpy().astype(np.float32),
                probs_raw=probs_raw_te.numpy().astype(np.float32),
            )
            logger.info("Saved full_test_arrays.npz")
        except Exception as e:
            logger.warning("Failed to save full_test_arrays.npz: %s", e)

    # File label for histograms
    cal_suffix = _calib_suffix(args.calibration)


    # 1) pmax histogram (article-friendly: show the *effective* pmax threshold implied by delta*)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(y_conf_test, bins=30, alpha=0.75)
    ax.set_xlabel(f"max softmax probability pmax ({cal_suffix})")
    ax.set_ylabel("Count")
    ax.set_xlim(0.5, 1.0)  # binary classifier => pmax ∈ [0.5, 1]

    # tau line (only if active)
    if float(tau_star) > 1e-9:
        ax.axvline(float(tau_star), linestyle="--", alpha=0.9, label=f"tau* = {float(tau_star):.4f}")

    # delta line converted to pmax threshold for binary (margin = |p1 - p2| = 2*pmax - 1)
    if float(delta_star) > 0 and (n_classes == 2):
        p_thr = 0.5 + 0.5 * float(delta_star)
        ax.axvline(p_thr, linestyle="--", alpha=0.9, label=f"pmax_thr(δ*) = {p_thr:.4f}")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    p_hist_path = os.path.join(args.output_folder, "uncertainty_hist.png")
    fig.savefig(p_hist_path, dpi=300)
    if args.save_pdf:
        fig.savefig(os.path.splitext(p_hist_path)[0] + ".pdf")
    plt.close(fig)
    logging.info("Saved uncertainty histogram -> %s", p_hist_path)

    # 2) margin histogram
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(y_margin_test, bins=30, alpha=0.75)


    if float(delta_star) > 1e-9:
        ax.axvline(float(delta_star), linestyle="--", alpha=0.9, label=f"delta* = {float(delta_star):.4f}")

    ax.set_xlabel(f"margin |p1 - p2| ({cal_suffix})")
    ax.set_ylabel("Count")

    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    m_hist_path = os.path.join(args.output_folder, "uncertainty_margin_hist.png")
    fig.savefig(m_hist_path, dpi=300)
    if args.save_pdf:
        fig.savefig(os.path.splitext(m_hist_path)[0] + ".pdf")
    plt.close(fig)
    logging.info("Saved uncertainty margin histogram -> %s", m_hist_path)

    # 3) Full TEST confusion + ROC + PR + selective triage metrics
    try:
        metrics_full, cm_full, roc_full, pr_full = _compute_binary_metrics_and_curves(
            y_true_test, y_pred_test, y_prob_pos_test
        )

        # Full TEST calibration metrics (RAW vs CAL)
        try:
            probs_raw_np_full = probs_raw_te.numpy()
            y_prob_pos_raw_full = probs_raw_np_full[:, 1].astype(float) if probs_raw_np_full.shape[1] == 2 else probs_raw_np_full[:, 0].astype(float)
            ece_full_raw = expected_calibration_error(y_true_test, y_prob_pos_raw_full, n_bins=int(args.ece_bins))
            ece_full_cal = expected_calibration_error(y_true_test, y_prob_pos_test, n_bins=int(args.ece_bins))
            nll_full_raw = negative_log_likelihood(y_true_test, y_prob_pos_raw_full)
            nll_full_cal = negative_log_likelihood(y_true_test, y_prob_pos_test)
            br_full_raw  = brier_score(y_true_test, y_prob_pos_raw_full)
            br_full_cal  = brier_score(y_true_test, y_prob_pos_test)

            cal_path = os.path.join(args.output_folder, "full_test_calibration_metrics.csv")
            with open(cal_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Metric","RAW","CAL"])
                w.writerow(["ECE", f"{ece_full_raw:.6f}", f"{ece_full_cal:.6f}"])
                w.writerow(["NLL", f"{nll_full_raw:.6f}", f"{nll_full_cal:.6f}"])
                w.writerow(["Brier", f"{br_full_raw:.6f}", f"{br_full_cal:.6f}"])
            logger.info("Saved full_test_calibration_metrics.csv -> %s", cal_path)
        except Exception as _e:
            cal_path = None
            logger.warning("Could not compute full TEST calibration metrics: %s", _e)

        sel = _selective_metrics_binary(y_true_test, probs_te_np, tau=float(tau_star), delta=float(delta_star))
        sel_path = os.path.join(args.output_folder, "full_test_selective_metrics.csv")
        with open(sel_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sel.keys()))
            w.writeheader()
            w.writerow(sel)

        metrics_txt = os.path.join(args.output_folder, "full_test_metrics.txt")
        _write_metrics_txt(
            metrics_txt,
            header="Full TEST metrics",
            metrics=metrics_full,
            extra_lines=[
                f"calibration={args.calibration}",
                f"tau*={float(tau_star):.6f}, delta*={float(delta_star):.6f}",
                f"full_test_calibration_metrics_csv={cal_path}" if cal_path else "full_test_calibration_metrics_csv=None",
                "--- selective triage (tau*,delta*) ---",
                f"coverage={sel.get('coverage')}",
                f"abstain={sel.get('abstain')}",
                f"risk_accept={sel.get('risk_accept')}",
                f"capture={sel.get('capture')}",
                f"precision_review={sel.get('precision_review')}",
                f"fn_auto_rate={sel.get('fn_auto_rate')}",
                f"fp_auto_rate={sel.get('fp_auto_rate')}",
            ],
        )
        logging.info("Saved %s", metrics_txt)

        out_png = os.path.join(args.output_folder, "confusion_roc_full.png")
        _save_confusion_roc_pr_figure(cm_full, roc_full, pr_full, metrics_full, out_png, title_prefix="TEST", save_pdf=bool(args.save_pdf))
        logging.info("Saved confusion+ROC/PR -> %s", out_png)
        logging.info("Saved selective triage metrics -> %s", sel_path)

    except Exception as e:
        logging.warning("Failed to build full TEST diagnostics: %s", e)

    # --- Uncertain/Review set on FULL TEST: must match the same calibrated probs used above ---
    # Review rule MUST match the selective decision used elsewhere:
    #   ACCEPT iff pmax >= tau* AND margin >= delta*
    tau_eff = float(tau_star) if np.isfinite(float(tau_star)) else 0.0
    delta_eff = float(delta_star) if np.isfinite(float(delta_star)) else 0.0
    accept_mask = (y_conf_test >= tau_eff) & (y_margin_test >= delta_eff)
    review_idx = np.where(~accept_mask)[0].astype(int)
    logger.info(f"[FULL] Review flagged on TEST: {len(review_idx)} / {len(y_true_test)} samples.")

    uncertain_full = []
    for idx in review_idx.tolist():
        uncertain_full.append({
            "index": int(idx),
            "X": X_te_np[int(idx)].tolist(),
            "true_label": int(y_true_test[int(idx)]),
            "prob": [float(probs_te_np[int(idx), 0]), float(probs_te_np[int(idx), 1])],
        })

    # --- Save uncertain points CSV ---
    csv_path = os.path.join(args.output_folder, 'uncertain_full.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','X','true_label','p1','p2'])
        writer.writeheader()
        for u in uncertain_full:
            p1, p2 = u['prob']
            writer.writerow({
                'index':      u['index'],
                'X':          u['X'],
                'true_label': u['true_label'],
                'p1':         p1,
                'p2':         p2
            })

    # Extended uncertain CSV (safe to add; downstream can ignore)
    try:
        ext_path = os.path.join(args.output_folder, "uncertain_full_ext.csv")
        with open(ext_path, "w", newline="") as f:
            fieldnames = [
                "run_id","seed","calibration","tau_star","delta_star",
                "record","window_in_record",
                "index","true_label","pred","is_error","p1","p2","pmax","margin","reason"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for idx in review_idx.tolist():
                idx = int(idx)
                p1 = float(probs_te_np[idx, 0]); p2 = float(probs_te_np[idx, 1])
                pmax = float(y_conf_test[idx]); marg = float(y_margin_test[idx])
                pred = int(y_pred_test[idx])
                reason = []
                if pmax < float(tau_eff): reason.append("pmax")
                if marg < float(delta_eff): reason.append("margin")

                rec = ""
                win_in_rec = ""
                if test_rec_order is not None and test_cum_ends is not None:
                    j = int(bisect.bisect_right(test_cum_ends, idx))
                    if 0 <= j < len(test_rec_order):
                        rec = str(test_rec_order[j])
                        start = int(test_cum_ends[j-1]) if j > 0 else 0
                        win_in_rec = int(idx - start)

                w.writerow({
                    "index": idx,
                    "true_label": int(y_true_test[idx]),
                    "pred": pred,
                    "is_error": int(pred != int(y_true_test[idx])),
                    "p1": p1, "p2": p2,
                    "pmax": pmax,
                    "margin": marg,
                    "reason": "+".join(reason) if reason else "review",
                    "run_id": str(getattr(args, "run_id", "")),
                    "seed": int(args.seed),
                    "calibration": str(args.calibration),
                    "tau_star": float(tau_eff),
                    "delta_star": float(delta_eff),
                    "record": rec,
                    "window_in_record": win_in_rec,
                })
        logger.info("Saved extended uncertain CSV -> %s", ext_path)
    except Exception as _e:
        logger.warning("Failed to write uncertain_full_ext.csv: %s", _e)

    # --- export FULL TEST predictions (all samples) for post-processing on non-uncertain subsets ---
    # This enables analyses like: "high-confidence PVC predictions that are still fragile / error-prone".
    try:
        full_pred_path = os.path.join(args.output_folder, "full_test_predictions_ext.csv")
        fieldnames = [
            "run_id","seed","calibration","tau_star","delta_star",
            "index","record","window_in_record",
            "true_label","pred","is_error",
            "p1","p2","pmax","margin",
            "accepted","review_reason",
            "x0","x1","x2","x3","X",
        ]
        rec_stats = {}

        with open(full_pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for idx in range(len(y_true_test)):
                idx = int(idx)
                p1 = float(probs_te_np[idx, 0]); p2 = float(probs_te_np[idx, 1])
                pmax = float(y_conf_test[idx]); marg = float(y_margin_test[idx])
                pred = int(y_pred_test[idx])
                true = int(y_true_test[idx])
                is_err = int(pred != true)
                accepted = bool(accept_mask[idx])

                # record mapping (requires deterministic concatenation order saved in full_test_index_map.json)
                rec = ""
                win_in_rec = ""
                if test_rec_order is not None and test_cum_ends is not None:
                    j = int(bisect.bisect_right(test_cum_ends, idx))
                    if 0 <= j < len(test_rec_order):
                        rec = str(test_rec_order[j])
                        start = int(test_cum_ends[j-1]) if j > 0 else 0
                        win_in_rec = int(idx - start)

                rr = []
                if not accepted:
                    if pmax < float(tau_eff): rr.append("pmax")
                    if marg < float(delta_eff): rr.append("margin")
                review_reason = "+".join(rr) if rr else ""

                x0, x1, x2, x3 = [float(v) for v in X_te_np[idx].reshape(-1).tolist()]

                w.writerow({
                    "run_id": str(getattr(args, "run_id", "")),
                    "seed": int(args.seed),
                    "calibration": str(args.calibration),
                    "tau_star": float(tau_eff),
                    "delta_star": float(delta_eff),

                    "index": idx,
                    "record": rec,
                    "window_in_record": win_in_rec,
                    "true_label": true,
                    "pred": pred,
                    "is_error": is_err,
                    "p1": p1, "p2": p2,
                    "pmax": pmax,
                    "margin": marg,
                    "accepted": int(accepted),
                    "review_reason": review_reason,
                    "x0": x0, "x1": x1, "x2": x2, "x3": x3,
                    # NOTE: keep 'X' as a Python-list string for compatibility with load_uncertain_points()
                    "X": [x0, x1, x2, x3],
                })

                # Per-record rollup
                if rec:
                    st = rec_stats.setdefault(rec, {
                        "record": rec,
                        "n_total": 0,
                        "n_error": 0,
                        "n_accept": 0,
                        "n_accept_error": 0,
                        "n_review": 0,
                        "n_review_error": 0,
                        "n_pred1": 0,
                        "n_pred1_error": 0,
                        "n_pred1_accept": 0,
                        "n_pred1_accept_error": 0,
                        "n_pred1_review": 0,
                        "n_pred1_review_error": 0,
                    })

                    st["n_total"] += 1
                    st["n_error"] += int(is_err)

                    if accepted:
                        st["n_accept"] += 1
                        st["n_accept_error"] += int(is_err)
                    else:
                        st["n_review"] += 1
                        st["n_review_error"] += int(is_err)

                    if pred == 1:
                        st["n_pred1"] += 1
                        st["n_pred1_error"] += int(is_err)
                        if accepted:
                            st["n_pred1_accept"] += 1
                            st["n_pred1_accept_error"] += int(is_err)
                        else:
                            st["n_pred1_review"] += 1
                            st["n_pred1_review_error"] += int(is_err)

        logger.info("Saved full TEST predictions -> %s", full_pred_path)

        # Per-record summary for deployment / shift narrative.
        if rec_stats:
            rec_path = os.path.join(args.output_folder, "full_test_per_record_summary.csv")
            with open(rec_path, "w", newline="") as f:
                fieldnames = [
                    "record",
                    "n_total","n_error","error_rate",
                    "n_accept","risk_accept",
                    "n_review","capture_review",
                    "n_pred1","n_pred1_error","error_rate_pred1",
                    "n_pred1_accept","risk_accept_pred1",
                    "n_pred1_review","capture_review_pred1",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for rec in sorted(rec_stats.keys()):
                    st = rec_stats[rec]
                    n_total = int(st["n_total"])
                    n_error = int(st["n_error"])
                    n_accept = int(st["n_accept"])
                    n_accept_err = int(st["n_accept_error"])
                    n_review = int(st["n_review"])
                    n_review_err = int(st["n_review_error"])

                    risk_accept = (n_accept_err / n_accept) if n_accept > 0 else float("nan")
                    capture_review = (n_review_err / n_error) if n_error > 0 else float("nan")

                    n_pred1 = int(st["n_pred1"])
                    n_pred1_err = int(st["n_pred1_error"])
                    n_pred1_accept = int(st["n_pred1_accept"])
                    n_pred1_accept_err = int(st["n_pred1_accept_error"])
                    n_pred1_review = int(st["n_pred1_review"])
                    n_pred1_review_err = int(st["n_pred1_review_error"])

                    risk_accept_pred1 = (n_pred1_accept_err / n_pred1_accept) if n_pred1_accept > 0 else float("nan")
                    capture_review_pred1 = (n_pred1_review_err / n_pred1_err) if n_pred1_err > 0 else float("nan")

                    w.writerow({
                        "record": rec,
                        "n_total": n_total,
                        "n_error": n_error,
                        "error_rate": (n_error / n_total) if n_total > 0 else float("nan"),
                        "n_accept": n_accept,
                        "risk_accept": risk_accept,
                        "n_review": n_review,
                        "capture_review": capture_review,

                        "n_pred1": n_pred1,
                        "n_pred1_error": n_pred1_err,
                        "error_rate_pred1": (n_pred1_err / n_pred1) if n_pred1 > 0 else float("nan"),
                        "n_pred1_accept": n_pred1_accept,
                        "risk_accept_pred1": risk_accept_pred1,
                        "n_pred1_review": n_pred1_review,
                        "capture_review_pred1": capture_review_pred1,
                    })

            logger.info("Saved full TEST per-record summary -> %s", rec_path)

    except Exception as _e:
        logger.warning("Failed to export full_test_predictions_ext.csv: %s", _e)
        

    # If multiple budget plans were computed on VAL, evaluate them on TEST to get a real triage curve
    if budget_plans:
        try:
            out_curve = os.path.join(args.output_folder, "full_test_triage_curve.csv")
            with open(out_curve, "w", newline="") as f:
                fieldnames = [
                    "budget_frac_val","budget_count_val",
                    "tau","delta","kink_score_nearest",
                    "n_total_test","n_accept_test","n_review_test",
                    "coverage_test","abstain_test","risk_accept_test","capture_test",
                    "precision_review_test","fn_auto_rate_test","fp_auto_rate_test",
                ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for bp in budget_plans:
                    tau_b = float(bp["tau"]); delta_b = float(bp["delta"])
                    sel_b = _selective_metrics_binary(y_true_test, probs_te_np, tau=tau_b, delta=delta_b)
                    w.writerow({
                        "budget_frac_val": float(bp["budget_frac"]),
                        "budget_count_val": int(bp["budget_count"]),
                        "tau": f"{tau_b:.6f}",
                        "delta": f"{delta_b:.6f}",
                        "kink_score_nearest": f"{float(bp.get('kink_score_nearest', float('nan'))):.6f}" if not np.isnan(bp.get("kink_score_nearest", float("nan"))) else "",
                        "n_total_test": int(sel_b["n_total"]),
                        "n_accept_test": int(sel_b["n_accept"]),
                        "n_review_test": int(sel_b["n_review"]),
                        "coverage_test": f"{sel_b['coverage']:.6f}",
                        "abstain_test": f"{sel_b['abstain']:.6f}",
                        "risk_accept_test": f"{sel_b['risk_accept']:.6f}" if not np.isnan(sel_b["risk_accept"]) else "",
                        "capture_test": f"{sel_b['capture']:.6f}" if not np.isnan(sel_b["capture"]) else "",
                        "precision_review_test": f"{sel_b['precision_review']:.6f}" if not np.isnan(sel_b["precision_review"]) else "",
                        "fn_auto_rate_test": f"{sel_b['fn_auto_rate']:.6f}" if not np.isnan(sel_b["fn_auto_rate"]) else "",
                        "fp_auto_rate_test": f"{sel_b['fp_auto_rate']:.6f}" if not np.isnan(sel_b["fp_auto_rate"]) else "",
                    })
            logger.info("Saved TEST triage curve -> %s", out_curve)

            # Simple plots: risk vs coverage and capture vs abstain
            data = []
            with open(out_curve, "r", newline="") as f:
                rr = csv.DictReader(f)
                for row in rr:
                    cov = _float_or_nan(row.get("coverage_test","nan"))
                    abst = _float_or_nan(row.get("abstain_test","nan"))
                    risk = _float_or_nan(row.get("risk_accept_test","nan"))
                    capt = _float_or_nan(row.get("capture_test","nan"))
                    data.append((cov, abst, risk, capt))
            data = [d for d in data if not np.isnan(d[0]) and not np.isnan(d[1])]
            if data:
                covs = np.array([d[0] for d in data], float)
                absts = np.array([d[1] for d in data], float)
                risks = np.array([d[2] for d in data], float)
                caps = np.array([d[3] for d in data], float)

                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                ax.plot(covs, risks, marker="o", linewidth=1.2)
                ax.set_xlabel("Coverage on TEST")
                ax.set_ylabel("Risk among accepted (error rate)")
                ax.set_title("Selective risk–coverage (TEST)")
                fig.tight_layout()
                p = os.path.join(args.output_folder, "full_test_risk_coverage.png")
                fig.savefig(p, dpi=300)
                if args.save_pdf:
                    fig.savefig(os.path.splitext(p)[0] + ".pdf")
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                ax.plot(absts, caps, marker="o", linewidth=1.2)
                ax.set_xlabel("Abstain (review rate) on TEST")
                ax.set_ylabel("Error capture on TEST")
                ax.set_title("Capture–abstain (TEST)")
                fig.tight_layout()
                p = os.path.join(args.output_folder, "full_test_capture_abstain.png")
                fig.savefig(p, dpi=300)
                if args.save_pdf:
                    fig.savefig(os.path.splitext(p)[0] + ".pdf")
                plt.close(fig)
                logger.info("Saved TEST triage plots (risk-coverage, capture-abstain).")
        except Exception as _e:
            logger.warning("Failed to build TEST triage curve/plots: %s", _e)


    logger.info(f"[INFO] Saved full-model uncertain points to {csv_path}")

    # Human-readable artifact index
    _write_artifact_readme(args.output_folder, args, logger)

    # Final artifact manifest (quick audit + easy packaging)
    _write_artifact_manifest(args.output_folder, logger, hash_files=bool(args.hash_manifest))

    # Disable faulthandler periodic dumps and close its file handle
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
    try:
        if fh_faulthandler is not None:
            fh_faulthandler.close()
    except Exception:
        pass

    
if __name__ == "__main__":
    try:
        main()
    finally:
        logging.shutdown()
