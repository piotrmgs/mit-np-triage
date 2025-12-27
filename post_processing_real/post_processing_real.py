# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.
"""
Post-processing for MIT-BIH experiment:
- Loads trained CVNN and uncertain anchors.
- Fits local complex polynomial surrogates with robustness safeguards.
- Computes Puiseux expansions + interpretation.
- Robustness along adversarial directions.
- LIME & SHAP explanations with optional temperature scaling.
- Sensitivity analysis summary (tau, delta).
- Comparative calibration table with 95% CI (+ Wilcoxon & win-rate).
- Resource benchmark: Puiseux vs gradient saliency.
"""

import os
import sys
import csv
import json
import ast
import argparse
import glob
import time
import hashlib
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import random
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import sympy
import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

import re
import shutil


def _parse_X_value(x: Any, anchor: Optional[Dict[str, Any]] = None) -> Optional[List[float]]:
    """Parse X that may be stored as a Python list, numpy array, or a string like '[0.1, -0.2, ...]'.
    Returns list[float] or None if parsing fails.
    """
    if x is None:
        x_list = None
    elif isinstance(x, (list, tuple, np.ndarray)):
        try:
            x_list = [float(v) for v in list(x)]
        except Exception:
            x_list = None
    elif isinstance(x, str):
        s = x.strip()
        # Common case: literal list string
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    x_list = [float(t) for t in v]
                else:
                    x_list = None
            except Exception:
                x_list = None
        else:
            x_list = None
    else:
        x_list = None

    # Fallback: use explicit columns if present (x0..x3)
    if (x_list is None or len(x_list) < 4) and isinstance(anchor, dict):
        vals = []
        for k in ("x0", "x1", "x2", "x3"):
            if k in anchor and anchor[k] is not None:
                try:
                    vals.append(float(anchor[k]))
                except Exception:
                    vals = []
                    break
        if len(vals) == 4:
            x_list = vals

    if x_list is None:
        return None



    # Ensure exactly 4 floats (C^2 -> R^4)
    if len(x_list) > 4:
        x_list = x_list[:4]
    return x_list


def _anchor_to_xstar(up: Dict[str, Any]) -> np.ndarray:
    """Convert an anchor dict to an R^4 numpy vector."""
    x_list = _parse_X_value(up.get("X", None), anchor=up)
    if x_list is None or len(x_list) != 4:
        raise ValueError(f"Invalid X for anchor (expected 4 floats). Got: {up.get('X')}")
    # Also normalize the dict so downstream code sees a parsed list, not a string.
    up["X"] = x_list
    return np.asarray(x_list, dtype=np.float32)

# Optional deps (allow running with --skip_shap/--skip_lime even if libs are missing)
try:
    import shap  # type: ignore  # noqa: F401
except Exception:
    shap = None  # noqa: F401

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore  # noqa: F401
except Exception:
    LimeTabularExplainer = None  # noqa: F401


# =========================
# Model & data
# =========================
from src.find_up_synthetic import SimpleComplexNet, complex_modulus_to_logits  # noqa: F401
from mit_bih_pre.pre_pro import load_mitbih_data  # constants are imported below from find_up_real

# CLI & record list (robust import)
# Prefer importing record_names via 'src'; do NOT import up_real.parse_args here.
# Post-processing should have a stable CLI independent from training CLI.
try:
    from src.up_real import record_names  # type: ignore
except Exception:
    try:
        from up_real.up_real import record_names  # type: ignore
    except Exception:
        # Final fallback: hard-coded record list.
        record_names = [
            "100","101","102","103","104","105","106","107","108","109","111","112","113","114","115","116","117","118","119","121",
            "122","123","124","200","201","202","203","205","207","208","209","210","212","213","214","215","217","219","220","221",
            "222","223","228","230","231","232","233","234",
        ]

def parse_pp_args():
    """
    Minimal, stable CLI for post-processing.
    - output_folder: folder with up_real artifacts (may be a run folder or a parent folder)
    - out_dir: where to write post-processing outputs (default: <IN_DIR>/post_processing_real_<run_id>)
    - max_points: how many uncertain anchors to analyze deeply (default: 30; 0 = all)
    - selection: how to choose anchors when there are many (default: per_record_mixed)
    - point_indices: explicit comma-separated list of `index` values from uncertain_full(_ext).csv
    - skip_*: speed toggles
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, default="mit-bih")
    p.add_argument("--output_folder", type=str, default="up_real")  # points to up_real outputs

    p.add_argument(
        "--anchors_csv",
        type=str,
        default="",
        help="Optional anchors CSV to analyze (relative to input_dir or absolute). "
             "If empty, uses the default uncertain_full.csv produced by up_real.",
    )    
    p.add_argument("--out_dir", type=str, default="")               # optional override
    p.add_argument("--seed", type=int, default=12345)

    # Anchor selection (critical for paper workflow)
    p.add_argument("--max_points", type=int, default=30)
    p.add_argument("--selection", type=str, default="per_record_mixed",
                   choices=["per_record_mixed", "worst_margin", "random"])
    p.add_argument("--point_indices", type=str, default="")

    # Optional pre-filtering (lets you analyze e.g. high-confidence PVC predictions)
    p.add_argument(
        "--filter_pred",
        type=int,
        default=-1,
        help="Filter anchors by predicted class (requires 'pred' column in metadata). -1 disables.",
    )
    p.add_argument(
        "--filter_true",
        type=int,
        default=-1,
        help="Filter anchors by true label (requires 'true_label' column). -1 disables.",
    )
    p.add_argument(
        "--filter_pmax_min",
        type=float,
        default=float("nan"),
        help="Keep anchors with pmax >= this value (requires 'pmax' column). NaN disables.",
    )
    p.add_argument(
        "--filter_pmax_max",
        type=float,
        default=float("nan"),
        help="Keep anchors with pmax <= this value (requires 'pmax' column). NaN disables.",
    )
    p.add_argument(
        "--filter_accepted_only",
        action="store_true",
        help="Keep only 'accepted' predictions (requires 'accepted' column). Useful for silent-failure analysis.",
    )    

    # How to rank "most uncertain" points when selection uses ordering.
    # NOTE: lower = more uncertain for all supported keys.
    p.add_argument("--rank_by", type=str, default="margin",
                   choices=["margin", "pmax", "abs_logit", "abs_logit_cal"],
                   help="Ranking key for uncertainty ordering (lower = more uncertain).")

    # Paper workflow toggles
    p.add_argument("--paper", action="store_true",
                   help="Write paper-ready summaries (markdown + tables) using up_real artifacts.")
    p.add_argument("--copy_figures", action="store_true",
                   help="Copy key up_real figures/csv into OUT_DIR (self-contained bundle).")
    p.add_argument("--use_review_budget", action="store_true",
                   help="Override --max_points/--selection to analyze exactly review_budget "
                        "points (from run_args.json), ordered by --rank_by.")


    # Speed toggles
    p.add_argument("--skip_lime", action="store_true")
    p.add_argument("--skip_shap", action="store_true")
    p.add_argument("--skip_contours", action="store_true")
    p.add_argument("--skip_benchmark", action="store_true")
    p.add_argument(
        "--skip_multiplicity",
        action="store_true",
        help="Skip multiplicity (m) sensitivity sweep (can be slow).",
    )


    # Local surrogate knobs (keep defaults consistent with current script)
    p.add_argument("--local_delta", type=float, default=0.05)
    p.add_argument("--local_degree", type=int, default=4)
    p.add_argument("--local_samples", type=int, default=600)
    p.add_argument("--quality_samples", type=int, default=200)
    p.add_argument("--kink_samples", type=int, default=1000)


    # Robustness probe controls (affects flip-radius censoring + runtime)
    p.add_argument(
        "--attack_radius",
        type=float,
        default=0.05,
        help="Max radius for robustness tracing / flip search (r_max). Increase (e.g., 0.15) to reduce censoring.",
    )
    p.add_argument(
        "--robust_steps",
        type=int,
        default=60,
        help="Number of steps for robustness tracing along a direction.",
    )
    p.add_argument(
        "--robust_num_random",
        type=int,
        default=20,
        help="Number of random directions added on top of Puiseux-guided candidates.",
    )
    p.add_argument(
        "--robust_dir_radius",
        type=float,
        default=0.01,
        help="Small probe radius used inside find_adversarial_directions().",
    )    

    # Paper-ready robustness knobs (auto-refit local surrogate when fidelity is low)
    p.add_argument(
        "--min_corr",
        type=float,
        default=0.50,
        help="Minimum Pearson corr for accepting local surrogate (else refit).",
    )
    p.add_argument(
        "--min_sign_agree",
        type=float,
        default=0.75,
        help="Minimum sign agreement for accepting local surrogate (else refit).",
    )
    p.add_argument(
        "--max_refit_attempts",
        type=int,
        default=2,
        help="Max number of surrogate refit attempts when quality is low.",
    )
    p.add_argument(
        "--refit_samples_mult",
        type=float,
        default=2.0,
        help="Multiply local_samples by this factor on each refit attempt.",
    )
    p.add_argument(
        "--refit_kink_eps_mult",
        type=float,
        default=5.0,
        help="Multiply exclude_kink_eps by this factor on each refit attempt.",
    )

    # Bundle / packaging helpers (paper workflow)
    p.add_argument(
        "--zip_bundle",
        action="store_true",
        help="Create a zip archive of OUT_DIR at the end (for paper supplements).",
    )
    p.add_argument(
        "--zip_name",
        type=str,
        default="",
        help="Optional zip filename (default: <OUT_DIR>.zip in parent dir).",
    )


    # Error-triage evaluation (quantify the practical gain of Puiseux-guided probes)
    p.add_argument(
        "--skip_triage_eval",
        action="store_true",
        help="Skip error-triage evaluation (Puiseux vs uncertainty baselines).",
    )
    p.add_argument(
        "--triage_bootstrap",
        type=int,
        default=2000,
        help="Bootstrap replicates for AUPRC CIs in triage eval (0 disables bootstrap).",
    )
    p.add_argument(
        "--triage_pr_max_curves",
        type=int,
        default=4,
        help="Max number of score curves to overlay in the PR plot.",
    )

    p.add_argument(
        "--triage_only_eval",
        action="store_true",
        help="Only run the triage evaluation step using existing CSVs in OUT_DIR, then exit.",
    )


    return p.parse_args()



# Local surrogate & utilities
from src.local_analysis import (
    local_poly_approx_complex,
    puiseux_uncertain_point,
    load_uncertain_points,
    evaluate_poly_approx_quality,
    estimate_nonholomorphic_fraction,      # NEW
    benchmark_local_poly_approx_and_puiseux  # NEW
)
from src.puiseux import puiseux_expansions  # for benchmark timing
from src.find_up_real import compress_to_C2, WINDOW_SIZE, PRE_SAMPLES, FS  # constants + C^2 compression

# Post-processing helpers (accept temperature T where relevant)
from src.post_processing import (
    interpret_puiseux_expansions,
    find_adversarial_directions,
    scalarize,
    time_gradient_saliency,
    kink_score
)


def mean_ci95(values):
    """Return (mean, 95% CI) for a list of floats; handles n=0/1."""
    vals = [float(v) for v in values if v is not None and str(v) != "nan"]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = var ** 0.5
    ci = 1.96 * sd / (n ** 0.5)
    return m, ci



def write_paper_summary_md(
    *,
    out_dir: str,
    in_dir: str,
    run_id: str,
    calib_used: str,
    run_meta: Dict[str, Any],
    run_args: Dict[str, Any],
) -> None:
    """
    Build a single, paper-friendly markdown summary from up_real artifacts.
    Copy-pasteable into appendix/supplement.
    """

    def _f(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _md(df: Optional[pd.DataFrame]) -> str:
        """
        Render dataframe as a markdown table when possible; fallback to fixed-width text.
        (Keeps paper_summary.md copy-paste friendly.)
        """
        if df is None or df.empty:
            return ""
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)

    def _fmt_mean_ci(mean: float, ci: float, *, digits: int = 6) -> str:
        if not np.isfinite(mean):
            return "nan"
        if not np.isfinite(ci):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ±{ci:.{digits}f}"

    def _get_metric_row(df: Optional[pd.DataFrame], metric: str) -> Optional[pd.Series]:
        if df is None or df.empty or "Metric" not in df.columns:
            return None
        sub = df[df["Metric"].astype(str) == str(metric)]
        return sub.iloc[0] if len(sub) else None


    def _get_any(r: pd.Series, *names: str, default: Any = np.nan) -> Any:
        """
        Backward/forward compatible getter for columns that changed names between up_real versions.
        """
        for n in names:
            try:
                if n in r and r.get(n) is not None:
                    return r.get(n)
            except Exception:
                pass
        return default



    lines: List[str] = []
    err_rate_full_test = None
    lines.append(f"# Paper summary (post_processing_real) — run_id={run_id}")
    lines.append("")

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    lines.append("## Provenance")
    lines.append(f"- IN_DIR: `{in_dir}`")
    lines.append(f"- OUT_DIR: `{out_dir}`")
    lines.append(f"- calibration_used (up_real): **{calib_used}**")

    if run_meta:
        git = run_meta.get("git_commit", run_meta.get("git_hash"))
        dirty = run_meta.get("git_dirty", None)
        lines.append(f"- git_commit: `{git}` (dirty={dirty})")

        lines.append(f"- script_sha256: `{run_meta.get('script_sha256')}`")

        ts = run_meta.get("generated_at_utc", run_meta.get("timestamp"))
        if ts is not None:
            lines.append(f"- generated_at_utc: `{ts}`")


    if run_args:
        lines.append(
            f"- folds: {run_args.get('folds')}, epochs: {run_args.get('epochs')}, "
            f"lr: {run_args.get('lr')}, batch_size: {run_args.get('batch_size')}"
        )
        lines.append(
            f"- review_budget: {run_args.get('review_budget')}, budget_fracs: {run_args.get('budget_fracs')}"
        )
        lines.append(
            f"- embed_method: {run_args.get('embed_method')}, embed_max_points: {run_args.get('embed_max_points')}"
        )

    # Full split records (important for shift / budget drift narrative)
    split = _read_json_optional(os.path.join(in_dir, "full_split_records.json"))
    if split:
        # up_real writes: train_records / val_records / test_records
        lines.append(f"- full_train_records: {split.get('train_records')}")
        lines.append(f"- full_val_records: {split.get('val_records')}")
        lines.append(f"- full_test_records: {split.get('test_records')}")
    lines.append("")



    # ------------------------------------------------------------------
    # Key takeaways (for manuscript claims)
    # ------------------------------------------------------------------
    lines.append("## Key takeaways (sanity-check vs manuscript claims)")
    lines.append("")


    # --- CV worst-case fold (discrimination) ---
    df_pf_ext = safe_read_csv(os.path.join(in_dir, "cv_metrics_per_fold_extended.csv"))
    if df_pf_ext is not None and len(df_pf_ext):
        try:
            tmp = df_pf_ext.copy()
            tmp["err_rate"] = (pd.to_numeric(tmp["FP"], errors="coerce") + pd.to_numeric(tmp["FN"], errors="coerce")) / pd.to_numeric(tmp["n_test"], errors="coerce")
            tmp["AUC_num"] = pd.to_numeric(tmp["AUC"], errors="coerce")
            worst = tmp.loc[tmp["AUC_num"].idxmin()]
            lines.append(
                f"- **CV worst fold (by AUC):** fold={int(worst['fold'])}, "
                f"AUC={float(worst['AUC_num']):.3f}, "
                f"BalancedAcc={_f(worst.get('BalancedAcc', np.nan)):.3f}, "
                f"err_rate={_f(worst.get('err_rate', np.nan)):.3f}."
            )
        except Exception:
            pass

    # --- CV triage worst-case at max budget ---
    df_tri_pf = safe_read_csv(os.path.join(in_dir, "cv_test_triage_curve_per_fold.csv"))
    if df_tri_pf is not None and len(df_tri_pf) and "budget_frac_val" in df_tri_pf.columns:
        try:
            bmax = float(pd.to_numeric(df_tri_pf["budget_frac_val"], errors="coerce").max())
            sub = df_tri_pf[pd.to_numeric(df_tri_pf["budget_frac_val"], errors="coerce") == bmax].copy()
            if len(sub) and "test_capture" in sub.columns and "test_risk_accept" in sub.columns:
                sub["cap"] = pd.to_numeric(sub["test_capture"], errors="coerce")
                sub["risk"] = pd.to_numeric(sub["test_risk_accept"], errors="coerce")
                worst_cap = sub.loc[sub["cap"].idxmin()]
                worst_risk = sub.loc[sub["risk"].idxmax()]
                lines.append(
                    f"- **CV triage worst-case @ budget={bmax:.3f}:** "
                    f"min capture={_f(worst_cap.get('test_capture', np.nan)):.3f} (fold={int(worst_cap['fold'])}, abstain={_f(worst_cap.get('test_abstain', np.nan)):.3f}); "
                    f"max risk_accept={_f(worst_risk.get('test_risk_accept', np.nan)):.3f} (fold={int(worst_risk['fold'])}, abstain={_f(worst_risk.get('test_abstain', np.nan)):.3f})."
                )
        except Exception:
            pass


    # --- CV discrimination ---
    df_perf_q = safe_read_csv(os.path.join(in_dir, "cv_metrics_summary_perf.csv"))
    r_auc = _get_metric_row(df_perf_q, "AUC")
    r_acc = _get_metric_row(df_perf_q, "Acc")
    r_ap  = _get_metric_row(df_perf_q, "AP")

    parts = []
    if r_auc is not None:
        parts.append(f"AUC={_fmt_mean_ci(_f(r_auc.get('Mean', np.nan)), _f(r_auc.get('CI95', np.nan)))}")
    if r_acc is not None:
        parts.append(f"Acc={_fmt_mean_ci(_f(r_acc.get('Mean', np.nan)), _f(r_acc.get('CI95', np.nan)))}")
    if r_ap is not None:
        parts.append(f"AP={_fmt_mean_ci(_f(r_ap.get('Mean', np.nan)), _f(r_ap.get('CI95', np.nan)))}")
    if parts:
        lines.append(f"- **CV discrimination:** " + ", ".join(parts))

    # --- CV calibration improvement (raw -> calibrated) ---
    df_cal_q = safe_read_csv(os.path.join(in_dir, "cv_metrics_summary.csv"))
    r_ece = _get_metric_row(df_cal_q, "ECE")
    r_nll = _get_metric_row(df_cal_q, "NLL")
    r_bri = _get_metric_row(df_cal_q, "Brier")

    def _fmt_raw_cal(r: Optional[pd.Series]) -> str:
        if r is None:
            return "N/A"

        # Support both:
        # - Mean_raw/CI95_raw/Mean_cal/CI95_cal/RelativeDrop  (newer)
        # - RAW_mean/RAW_CI95/CAL_mean/CAL_CI95/Relative_drop (current up_real artifacts)
        m_raw = _f(_get_any(r, "Mean_raw", "RAW_mean"))
        ci_raw = _f(_get_any(r, "CI95_raw", "RAW_CI95"))
        m_cal = _f(_get_any(r, "Mean_cal", "CAL_mean"))
        ci_cal = _f(_get_any(r, "CI95_cal", "CAL_CI95"))
        rel = _f(_get_any(r, "RelativeDrop", "Relative_drop"))

        if np.isfinite(rel):
            return f"{_fmt_mean_ci(m_raw, ci_raw)} → {_fmt_mean_ci(m_cal, ci_cal)} (drop={100.0*rel:.1f}%)"
        return f"{_fmt_mean_ci(m_raw, ci_raw)} → {_fmt_mean_ci(m_cal, ci_cal)}"


    if df_cal_q is not None and len(df_cal_q):
        lines.append(f"- **CV calibration (method={calib_used}):** "
                     f"ECE { _fmt_raw_cal(r_ece) }; "
                     f"NLL { _fmt_raw_cal(r_nll) }; "
                     f"Brier { _fmt_raw_cal(r_bri) }")

    # --- Calibration vs NONE (win-rate + Wilcoxon) ---
    df_pf_q = safe_read_csv(os.path.join(in_dir, "cv_metrics_per_fold_multi.csv"))
    if df_pf_q is not None and len(df_pf_q) and "method" in df_pf_q.columns:
        used = str(calib_used).upper()
        try:
            import numpy as _np
            from scipy.stats import wilcoxon  # type: ignore

            def _wilc(metric: str, alt: str):
                base = df_pf_q[df_pf_q["method"] == "NONE"][metric].astype(float).values
                sub  = df_pf_q[df_pf_q["method"] == used][metric].astype(float).values
                n = min(len(base), len(sub))
                if n < 5:
                    return None, None, None
                wins = int((_np.asarray(sub[:n]) < _np.asarray(base[:n])).sum()) if alt == "less" else int((_np.asarray(sub[:n]) > _np.asarray(base[:n])).sum())
                p = wilcoxon(sub[:n], base[:n], alternative=alt).pvalue
                return wins, n, float(p)

            for met, alt in [("ECE","less"), ("NLL","less"), ("Brier","less")]:
                if met in df_pf_q.columns and used in set(df_pf_q["method"].unique()) and "NONE" in set(df_pf_q["method"].unique()):
                    wins, n, p = _wilc(met, alt)
                    if wins is not None:
                        lines.append(f"- **{used} vs NONE (CV):** {met} wins {wins}/{n} folds (Wilcoxon one-sided p={p:.3e}).")
        except Exception:
            # keep summary robust even if scipy is missing
            pass

    # --- CV triage headline number (use the largest budget_frac in the summary) ---
    df_tri_q = safe_read_csv(os.path.join(in_dir, "cv_test_triage_curve_summary.csv"))
    if df_tri_q is not None and len(df_tri_q) and "budget_frac_val" in df_tri_q.columns:
        r = df_tri_q.sort_values("budget_frac_val").iloc[-1]
        b = _f(r.get("budget_frac_val", np.nan))
        a = _f(r.get("abstain_mean", np.nan))
        risk = _f(r.get("risk_mean", np.nan))
        cap = _f(r.get("capture_mean", np.nan))
        eff = cap / a if np.isfinite(cap) and np.isfinite(a) and a > 0 else float("nan")
        lines.append(f"- **CV triage (test, mean across folds):** target budget_frac={b:.3f}, "
                     f"observed abstain={a:.4f}, risk={risk:.4f}, capture={cap:.4f}, "
                     f"capture/abstain≈{eff:.2f}× (vs random baseline).")

    # --- Full test base + calibration + selective (if present) ---
    txt_full_q = read_text_optional(os.path.join(in_dir, "full_test_metrics.txt"))
    if txt_full_q:
        kv = parse_kv_from_text(txt_full_q)
        if kv and all(k in kv for k in ["TN","FP","FN","TP"]):
            n_total = float(kv["TN"] + kv["FP"] + kv["FN"] + kv["TP"])
            err_rate = float((kv["FP"] + kv["FN"]) / max(n_total, 1.0))
            err_rate_full_test = err_rate
            acc = kv.get("Accuracy", kv.get("acc", float("nan")))
            auc = kv.get("ROC_AUC", kv.get("auc", float("nan")))
            ap  = kv.get("PR_AUC(AP)", kv.get("ap", float("nan")))
            lines.append(f"- **Full test (base):** acc={acc:.4f}, auc={auc:.4f}, ap={ap:.4f}, error_rate={err_rate:.4f}.")

    df_full_cal_q = safe_read_csv(os.path.join(in_dir, "full_test_calibration_metrics.csv"))
    if df_full_cal_q is not None and len(df_full_cal_q):
        try:
            r = df_full_cal_q[df_full_cal_q["Metric"].astype(str) == "ECE"].iloc[0].to_dict()
            lines.append(
                f"- **Full test calibration:** ECE raw={_f(r.get('RAW', r.get('raw', np.nan))):.6f} → "
                f"cal={_f(r.get('CAL', r.get('cal', np.nan))):.6f}."
            )

        except Exception:
            pass

    df_full_sel_q = safe_read_csv(os.path.join(in_dir, "full_test_selective_metrics.csv"))
    if df_full_sel_q is not None and len(df_full_sel_q):
        rr = df_full_sel_q.iloc[0].to_dict()
        lines.append(f"- **Full test selective (chosen on validation):** "
                     f"n_review={rr.get('n_review')}, abstain={_f(rr.get('abstain', np.nan)):.4f}, "
                     f"risk_accept={_f(rr.get('risk_accept', np.nan)):.4f}, capture={_f(rr.get('capture', np.nan)):.4f}, "
                     f"precision_review={_f(rr.get('precision_review', np.nan)):.3f}.")

        # drift vs validation choice (review_budget selection)
        df_sens_full_q = safe_read_csv(os.path.join(in_dir, "sens_full.csv"))
        if df_sens_full_q is not None and len(df_sens_full_q):
            val_abst = _f(df_sens_full_q.iloc[0].get("abstain", np.nan))
            test_abst = _f(rr.get("abstain", np.nan))
            if np.isfinite(val_abst) and val_abst > 0 and np.isfinite(test_abst):
                lines.append(f"- **Budget drift (val→test, fixed threshold):** abstain {val_abst:.4f} → {test_abst:.4f} "
                             f"({test_abst/val_abst:.1f}×).")

    # --- Uncertain set concentration (records) ---
    df_unc_ext_q = safe_read_csv(os.path.join(in_dir, "uncertain_full_ext.csv"))
    if df_unc_ext_q is not None and len(df_unc_ext_q) and "record" in df_unc_ext_q.columns:
        vc = df_unc_ext_q["record"].astype(str).value_counts()
        top = vc.head(3)
        share = float(top.sum() / max(len(df_unc_ext_q), 1))

        top_info = []
        if "pred" in df_unc_ext_q.columns and "true_label" in df_unc_ext_q.columns:
            try:
                tmp = df_unc_ext_q.copy()
                tmp["_is_error"] = (tmp["pred"].astype(int) != tmp["true_label"].astype(int))
                for rec in list(top.index):
                    sub = tmp[tmp["record"].astype(str) == str(rec)]
                    er = float(sub["_is_error"].mean()) if len(sub) else float("nan")
                    top_info.append(f"{rec} (n={len(sub)}, err={er:.1%})")
            except Exception:
                top_info = [f"{r} (n={int(c)})" for r, c in top.items()]
        else:
            top_info = [f"{r} (n={int(c)})" for r, c in top.items()]

        lines.append(
            f"- **Uncertain set concentration:** top records={top_info} "
            f"cover {share:.1%} of uncertain points (n_uncertain={len(df_unc_ext_q)})."
        )


    lines.append("")


    # ------------------------------------------------------------------
    # Discrimination performance (CV)
    # ------------------------------------------------------------------
    df_perf = safe_read_csv(os.path.join(in_dir, "cv_metrics_summary_perf.csv"))
    if df_perf is not None and len(df_perf):
        lines.append("## Discrimination performance (CV, mean ±95% CI)")
        lines.append("")
        wanted = ["Acc", "AUC", "AP", "F1", "Precision", "Recall", "Spec", "BalancedAcc"]
        try:
            df_show = df_perf[df_perf["Metric"].isin(wanted)].copy()
            if df_show.empty:
                df_show = df_perf.copy()
            lines.append(_md(df_show))
        except Exception:
            lines.append(df_perf.to_string(index=False))
        lines.append("")

    # ------------------------------------------------------------------
    # Calibration summary (raw vs calibrated)
    # ------------------------------------------------------------------
    df_cal = safe_read_csv(os.path.join(in_dir, "cv_metrics_summary.csv"))
    if df_cal is not None and len(df_cal):
        lines.append("## Calibration (CV: raw vs calibrated)")
        lines.append("")
        for _, r in df_cal.iterrows():
            metric = str(r.get("Metric", ""))

            m_raw = _f(_get_any(r, "Mean_raw", "RAW_mean"))
            ci_raw = _f(_get_any(r, "CI95_raw", "RAW_CI95"))
            m_cal = _f(_get_any(r, "Mean_cal", "CAL_mean"))
            ci_cal = _f(_get_any(r, "CI95_cal", "CAL_CI95"))
            rel = _f(_get_any(r, "RelativeDrop", "Relative_drop"))

            lines.append(
                f"- **{metric}**: raw={m_raw:.6f} ±{ci_raw:.6f} → "
                f"cal={m_cal:.6f} ±{ci_cal:.6f} | rel_drop={rel:.3f}"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Calibration methods comparison (CV)
    # ------------------------------------------------------------------
    df_multi = safe_read_csv(os.path.join(in_dir, "cv_metrics_summary_multi.csv"))
    if df_multi is not None and len(df_multi):
        lines.append("## Calibration methods comparison (CV)")
        lines.append("")
        try:
            top = df_multi[df_multi["Metric"] == "ECE"].sort_values("Mean").head(6)
            lines.append("Top by ECE (lower is better):")
            for _, r in top.iterrows():
                lines.append(f"- {r['Method']}: ECE={_f(r['Mean']):.6f} ±{_f(r['CI95']):.6f}")
            lines.append("")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Win-rate vs NONE (ECE) for selected calibration
    # ------------------------------------------------------------------
    df_pf = safe_read_csv(os.path.join(in_dir, "cv_metrics_per_fold_multi.csv"))
    if df_pf is not None and len(df_pf):
        try:
            used = str(calib_used).upper()
            base = df_pf[df_pf["method"] == "NONE"][["fold", "ECE"]].rename(columns={"ECE": "ECE_NONE"})
            sub = df_pf[df_pf["method"] == used][["fold", "ECE"]].rename(columns={"ECE": f"ECE_{used}"})
            j = base.merge(sub, on="fold", how="inner")
            if len(j):
                wins = int((j[f"ECE_{used}"] < j["ECE_NONE"]).sum())
                lines.append(f"**ECE win-rate vs NONE:** {used} wins in **{wins}/{len(j)} folds**.")
                lines.append("")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Triage curve (CV test)
    # ------------------------------------------------------------------
    df_tri = safe_read_csv(os.path.join(in_dir, "cv_test_triage_curve_summary.csv"))
    if df_tri is not None and len(df_tri):
        lines.append("## Triage curve (CV test, mean across folds)")
        lines.append("")
        cols = [c for c in ["budget_frac_val", "abstain_mean", "coverage_mean", "risk_mean", "capture_mean"] if c in df_tri.columns]
        if cols:
            lines.append(_md(df_tri[cols]))
            lines.append("")

    # ------------------------------------------------------------------
    # Full test (base metrics)
    # ------------------------------------------------------------------
    txt_full = read_text_optional(os.path.join(in_dir, "full_test_metrics.txt"))
    if txt_full:
        kv = parse_kv_from_text(txt_full)
        if kv:
            lines.append("## Full test (base classifier metrics)")
            lines.append("")
            if all(k in kv for k in ["TN", "FP", "FN", "TP"]):
                lines.append(
                    f"- confusion: TN={int(kv['TN'])}, FP={int(kv['FP'])}, FN={int(kv['FN'])}, TP={int(kv['TP'])}"
                )
            key_map = {
                "acc": ["Accuracy", "acc"],
                "precision": ["Precision", "precision"],
                "recall": ["Recall", "recall"],
                "f1": ["F1", "f1"],
                "auc": ["ROC_AUC", "auc"],
                "ap": ["PR_AUC(AP)", "ap"],
                "specificity": ["Specificity", "specificity"],
                "balanced_acc": ["BalancedAcc", "balanced_acc"],
            }
            for out_k, cand in key_map.items():
                for kk in cand:
                    if kk in kv:
                        lines.append(f"- {out_k}: {kv[kk]:.6f}")
                        break

            lines.append("")

    # ------------------------------------------------------------------
    # Full test calibration (raw vs calibrated) — IMPORTANT for narrative
    # ------------------------------------------------------------------
    df_full_calib = safe_read_csv(os.path.join(in_dir, "full_test_calibration_metrics.csv"))
    if df_full_calib is not None and len(df_full_calib):
        lines.append("## Full test calibration (raw vs calibrated)")
        lines.append("")
        lines.append(_md(df_full_calib))
        lines.append("")

    # ------------------------------------------------------------------
    # Full test triage curve
    # ------------------------------------------------------------------
    df_full_tri = safe_read_csv(os.path.join(in_dir, "full_test_triage_curve.csv"))
    if df_full_tri is not None and len(df_full_tri):
        lines.append("## Full test triage curve (sweep over budget_fracs)")
        lines.append("")
        cols = [c for c in ["budget_frac_val", "abstain_test", "coverage_test", "risk_accept_test", "capture_test", "precision_review_test"] if c in df_full_tri.columns]
        if cols:
            lines.append(_md(df_full_tri[cols]))
            lines.append("")


        # Budget drift table (critical: budget_frac is target on validation; abstain_test is observed on test)
        try:
            if "budget_frac_val" in df_full_tri.columns and "abstain_test" in df_full_tri.columns:
                df_drift = df_full_tri.copy()
                df_drift["budget_frac_val"] = pd.to_numeric(df_drift["budget_frac_val"], errors="coerce")
                df_drift["abstain_test"] = pd.to_numeric(df_drift["abstain_test"], errors="coerce")
                df_drift["drift_factor"] = df_drift["abstain_test"] / df_drift["budget_frac_val"]

                cols = [c for c in [
                    "budget_frac_val", "abstain_test", "drift_factor",
                    "risk_accept_test", "capture_test", "precision_review_test"
                ] if c in df_drift.columns]

                lines.append("## Budget drift (validation-chosen threshold → observed test abstain)")
                lines.append("")
                lines.append(
                    "Important: `budget_frac_val` is the *target on validation* used to pick a fixed threshold; "
                    "`abstain_test` is the *observed* abstain on test. Under record-level shift, abstain can drift "
                    "by large factors. For deployment with strict review capacity, prefer top-K enforcement."
                )
                lines.append("")
                if cols:
                    lines.append(_md(df_drift[cols]))
                    lines.append("")
        except Exception:
            pass


    # ------------------------------------------------------------------
    # Full test (final model): selective metrics @ chosen (tau*,delta*)
    # ------------------------------------------------------------------
    df_full_sel = safe_read_csv(os.path.join(in_dir, "full_test_selective_metrics.csv"))
    if df_full_sel is not None and len(df_full_sel):
        lines.append("## Full test (final): selective metrics @ chosen (tau*,delta*)")
        lines.append("")
        r = df_full_sel.iloc[0].to_dict()
        for k in ["n_total", "n_review", "abstain", "coverage", "risk_accept", "capture", "precision_review", "delta", "tau"]:
            if k in r:
                lines.append(f"- {k}: {r[k]}")
        lines.append("")

    # ------------------------------------------------------------------
    # Budget drift (validation → full test) for the chosen tau*,delta*
    # ------------------------------------------------------------------
    df_sens_full = safe_read_csv(os.path.join(in_dir, "sens_full.csv"))
    if df_sens_full is not None and len(df_sens_full) and df_full_sel is not None and len(df_full_sel):
        try:
            val_tau = _f(df_sens_full.iloc[0].get("tau", float("nan")))
            val_delta = _f(df_sens_full.iloc[0].get("delta", float("nan")))
            val_abst = _f(df_sens_full.iloc[0].get("abstain", float("nan")))
            test_abst = _f(df_full_sel.iloc[0].get("abstain", float("nan")))

            lines.append("## Budget drift note (validation → full test)")
            lines.append("")
            lines.append(f"- tau* (val)  = {val_tau:.6f}")
            lines.append(f"- delta* (val)= {val_delta:.6f}")
            if np.isfinite(val_delta):
                lines.append(f"- implied pmax threshold (binary): {(val_delta + 1.0)/2.0:.6f}")
            lines.append(f"- abstain_val (chosen)    = {val_abst:.6f}")
            lines.append(f"- abstain_test (observed) = {test_abst:.6f}")
            if np.isfinite(val_abst) and val_abst > 0 and np.isfinite(test_abst):
                lines.append(f"- drift factor (test/val) = {test_abst/val_abst:.2f}×")
            lines.append(
                "Interpretation: `sens_full.csv` corresponds to the **absolute review_budget selection** "
                "(budget_count=review_budget) on validation; `sens_full_multi.csv` + `full_test_triage_curve.csv` "
                "is the separate sweep over `budget_fracs`. In all cases thresholds are chosen on validation, "
                "so under distribution shift the *observed* test abstain/coverage can drift (sometimes by orders of magnitude). "
                "For reporting: always cite **observed test coverage & risk**. "
                "If you need a strict budget in deployment, enforce it via top-K / quantile thresholding per batch."
            )
            lines.append("")
        except Exception:
            pass

    # --- Shift sanity-check: validation base risk vs full-test base error rate ---
    try:
        if err_rate_full_test is not None:
            df_sens0 = safe_read_csv(os.path.join(in_dir, "sens_grid_ext.csv"))
            if df_sens0 is None:
                df_sens0 = safe_read_csv(os.path.join(in_dir, "sens_grid.csv"))

            if df_sens0 is not None and len(df_sens0):
                risk_col = None
                for cand in ["risk_accept_exact", "risk_accept", "risk_accept_grid"]:
                    if cand in df_sens0.columns:
                        risk_col = cand
                        break
                if risk_col is None:
                    raise KeyError("No risk column found in sens grid (expected risk_accept_exact or risk_accept).")

                # Prefer abstain_exact==0 if present; else minimal abstain.
                if "abstain_exact" in df_sens0.columns:
                    sub0 = df_sens0[df_sens0["abstain_exact"] == 0.0]
                    if len(sub0) == 0:
                        sub0 = df_sens0.sort_values("abstain_exact").head(1)
                else:
                    sub0 = df_sens0.head(1)

                val_risk = float(pd.to_numeric(sub0.iloc[0].get(risk_col, np.nan), errors="coerce"))

                if np.isfinite(val_risk) and val_risk > 0:
                    ratio = float(err_rate_full_test) / val_risk
                    lines.append(
                        f"- **Shift sanity-check:** base error_rate val≈{val_risk:.4%} "
                        f"vs full-test≈{float(err_rate_full_test):.4%} (×{ratio:.1f})."
                    )
    except Exception:
        pass


    # ------------------------------------------------------------------
    # Strict-budget triage on FULL test (Top-K by uncertainty on test)
    # ------------------------------------------------------------------
    try:
        npz_path = os.path.join(in_dir, "full_test_arrays.npz")
        if os.path.isfile(npz_path):
            arr = np.load(npz_path)

            y = np.asarray(arr["y_true"], dtype=int)
            probs = np.asarray(arr["probs"], dtype=np.float64)  # calibrated probs
            pmax = np.asarray(arr["pmax"], dtype=np.float64)

            y_pred = (probs[:, 1] >= 0.5).astype(int)
            is_err = (y_pred != y).astype(int)

            N = int(len(y))
            total_err = int(is_err.sum())

            budgets = _parse_float_list((run_args or {}).get("budget_fracs", ""))
            if not budgets:
                # fallback: use any budgets present in full_test_triage_curve.csv
                if df_full_tri is not None and "budget_frac_val" in df_full_tri.columns:
                    budgets = sorted(set(pd.to_numeric(df_full_tri["budget_frac_val"], errors="coerce").dropna().astype(float).tolist()))
            budgets = [b for b in budgets if np.isfinite(b) and b > 0 and b < 1.0]

            if budgets:
                order = np.argsort(pmax)  # lower pmax => more uncertain
                rows = []
                for b in budgets:
                    k = max(1, int(round(float(b) * N)))
                    mask_review = np.zeros(N, dtype=bool)
                    mask_review[order[:k]] = True
                    mask_accept = ~mask_review

                    n_review = int(mask_review.sum())
                    n_accept = int(mask_accept.sum())
                    abstain = float(n_review / max(N, 1))

                    err_review = int(is_err[mask_review].sum())
                    err_accept = int(is_err[mask_accept].sum())

                    risk_accept = float(err_accept / max(n_accept, 1))
                    capture = float(err_review / max(total_err, 1))
                    precision_review = float(err_review / max(n_review, 1))

                    rows.append({
                        "budget_frac_target": float(b),
                        "n_review_topk": n_review,
                        "abstain_topk": abstain,
                        "risk_accept_topk": risk_accept,
                        "capture_topk": capture,
                        "precision_review_topk": precision_review,
                    })

                df_topk = pd.DataFrame(rows)

                lines.append("## Full test strict-budget triage (Top-K by uncertainty on test)")
                lines.append("")
                lines.append(
                    "This table enforces the review fraction *exactly on test* by taking the top-K most uncertain points "
                    "(lowest `pmax`). It complements the fixed-threshold results, which can drift under shift."
                )
                lines.append("")
                lines.append(_md(df_topk))
                lines.append("")
    except Exception:
        pass



    # ------------------------------------------------------------------
    # Puiseux gain: error-triage among the analyzed uncertain points
    # ------------------------------------------------------------------
    triage_auc_path = os.path.join(out_dir, "triage_error_auc_compare.csv")
    if os.path.isfile(triage_auc_path):
        try:
            df_tri = pd.read_csv(triage_auc_path)

            lines.append("## Puiseux-guided error triage among analyzed uncertain points")
            lines.append("")
            lines.append(
                "We treat *misclassification* as the target event and ask: which ranking score best prioritizes the cases that are actually wrong?"
            )
            lines.append("")

            # Small narrative hook (top-2 comparison if available)
            try:
                base_row = df_tri.loc[df_tri["score_name"] == "uncert_margin"].head(1)
                pu_row = df_tri.loc[df_tri["score_name"] == "puiseux_inv_rflip"].head(1)
                if len(base_row) == 1 and len(pu_row) == 1:
                    ap_base = float(base_row["auprc"].iloc[0])
                    ap_pu = float(pu_row["auprc"].iloc[0])
                    lines.append(
                        f"**Key result:** Puiseux flip-radius probe (score=1/r_flip) achieves AUPRC={ap_pu:.3f} versus margin-based uncertainty AUPRC={ap_base:.3f} on the analyzed set."
                    )
                    lines.append("")
            except Exception:
                pass

            # Main table
            lines.append(df_tri.head(8).to_markdown(index=False))
            lines.append("")

            diff_path = os.path.join(out_dir, "triage_error_auc_diff_vs_uncert_margin.csv")
            if os.path.isfile(diff_path):
                try:
                    df_diff = pd.read_csv(diff_path)
                    lines.append("**Bootstrap AUPRC deltas vs margin-uncertainty:**")
                    lines.append("")
                    lines.append(df_diff.head(8).to_markdown(index=False))
                    lines.append("")
                except Exception:
                    pass

            pr_fig = os.path.join(out_dir, "triage_error_pr_curve.png")
            if os.path.isfile(pr_fig):
                lines.append(f"![]({os.path.basename(pr_fig)})")
                lines.append("")
        except Exception as e:
            lines.append(f"(Failed to summarize triage_error_auc_compare.csv: {e})")
           


    out_path = os.path.join(out_dir, "paper_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))



def _normalize_shap_output(shap_values):
    """
    Always return a list [arr_class0, arr_class1], each of shape [M, P].

    Supports variants returned by different SHAP versions:
    - a list of arrays (use first two),
    - a single 2D array (sum-to-zero convention) -> reconstruct the second class via negation,
    - a 3D array (K, M, P) -> take [0] and [1].

    If shape or rank is unexpected, returns [arr_2d, None] where arr_2d is coerced to shape (1, P).
    """
    import numpy as np
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return [np.asarray(shap_values[0]), np.asarray(shap_values[1])]
        # Single element on the list — treat as sum-to-zero.
        a0 = np.asarray(shap_values[0])
        return [a0, -a0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return [arr[0], arr[1]]
    elif arr.ndim == 2:
        return [arr, -arr]
    else:
        # Fallback: coerce to (1, P); second class unknown.
        arr = arr.reshape(1, -1)
        return [arr, None]


def _read_json_optional(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def sanitize_run_id(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "run"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] if s else "run")


def infer_run_id(in_dir: str, run_meta: Dict[str, Any], run_args: Dict[str, Any]) -> str:
    """
    Make run_id stable & unique even if run_meta/run_args has empty run_id.
    Preference order:
      1) run_meta.run_id / run_args.run_id
      2) basename(IN_DIR)
      3) add mtime suffix when still generic
    """
    rid = str((run_meta or {}).get("run_id") or (run_args or {}).get("run_id") or "").strip()
    if not rid:
        rid = os.path.basename(os.path.abspath(in_dir)).strip()

    rid = sanitize_run_id(rid)
    # If still generic, add a suffix to avoid collisions.
    if rid in {"run", "up_real"}:
        try:
            rid = sanitize_run_id(f"{rid}_{int(os.path.getmtime(in_dir))}")
        except Exception:
            rid = sanitize_run_id(rid)
    return rid


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_text_optional(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def parse_kv_from_text(text: str) -> Dict[str, float]:
    """
    Parse lines like:
      key = 0.123
    into a dict {key: 0.123}.
    Accept keys containing parentheses etc. (e.g. "PR_AUC(AP)").
    """
    out: Dict[str, float] = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out



def predict_logits_and_proba(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    *,
    cal: Optional["Calibrator"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      logits_raw_np: (N,2)
      probs_raw_np:  (N,2)
      probs_cal_np:  (N,2)  -- calibrated via `cal` if provided, else = raw
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    x_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits_raw = complex_modulus_to_logits(model(x_t))  # (N,2)

        probs_raw = torch.softmax(logits_raw, dim=1)

        if cal is None or (cal.method or "NONE").upper() == "NONE":
            probs_cal = probs_raw
        else:
            logits_cal = cal.apply_to_logits(logits_raw)
            probs_cal = torch.softmax(logits_cal, dim=1)

    return (
        logits_raw.detach().cpu().numpy(),
        probs_raw.detach().cpu().numpy(),
        probs_cal.detach().cpu().numpy(),
    )


def enforce_puiseux_ready_poly(
    expr: sympy.Expr,
    x_sym: sympy.Symbol,
    y_sym: sympy.Symbol,
    *,
    tol: float = 1e-10,
) -> Tuple[sympy.Expr, Dict[str, Any]]:
    """
    Defensive cleanup for Newton–Puiseux:
      - enforce F(0,0)=0 (remove constant if leaked numerically)
      - enforce no linear terms in x or y (remove if leaked numerically)
    Returns (expr_fixed, diag).
    """
    diag: Dict[str, Any] = {"puiseux_enforced": False}
    try:
        P = sympy.Poly(expr, x_sym, y_sym)
    except Exception as e:
        diag.update({"puiseux_enforced": False, "puiseux_enforce_error": str(e)})
        return expr, diag

    c0 = P.coeff_monomial(1)
    cx = P.coeff_monomial(x_sym)
    cy = P.coeff_monomial(y_sym)

    def _mag(z) -> float:
        try:
            return float(abs(complex(sympy.N(z))))
        except Exception:
            try:
                return float(abs(z))
            except Exception:
                return float("nan")

    diag.update({"c0_abs": _mag(c0), "cx_abs": _mag(cx), "cy_abs": _mag(cy)})

    expr2 = expr
    changed = False
    if np.isfinite(diag["c0_abs"]) and diag["c0_abs"] > tol:
        expr2 = expr2 - c0
        changed = True
    if np.isfinite(diag["cx_abs"]) and diag["cx_abs"] > tol:
        expr2 = expr2 - cx * x_sym
        changed = True
    if np.isfinite(diag["cy_abs"]) and diag["cy_abs"] > tol:
        expr2 = expr2 - cy * y_sym
        changed = True

    diag["puiseux_enforced"] = bool(changed)
    return sympy.expand(expr2), diag


def time_gradient_saliency_calibrated(
    model: torch.nn.Module,
    xstar: np.ndarray,
    device: torch.device,
    *,
    cal: Optional["Calibrator"],
    repeat: int = 5,
) -> Dict[str, Any]:
    """
    Saliency timing that matches the *effective* calibrated decision system.
    Returns a dict compatible with the existing resource report keys.
    """
    x0 = np.asarray(xstar, dtype=np.float32).reshape(1, -1)
    times_ms: List[float] = []
    grad_norm = float("nan")

    for _ in range(int(max(1, repeat))):
        model.zero_grad(set_to_none=True)
        xt = torch.from_numpy(x0).to(device).requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits_raw = complex_modulus_to_logits(model(xt))
        logits_eff = cal.apply_to_logits(logits_raw) if cal is not None else logits_raw
        d_eff = logits_eff[0, 1] - logits_eff[0, 0]
        d_eff.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append(1000.0 * (t1 - t0))

        g = xt.grad.detach().cpu().numpy().reshape(-1)
        grad_norm = float(np.linalg.norm(g))

    return {
        "time_ms": float(np.mean(times_ms)) if times_ms else float("nan"),
        "grad_norm": float(grad_norm),
        "cpu_rss_mb_delta": float("nan"),
        "gpu_peak_mb": float("nan"),
    }


# ============================================================
# Paper-ready bundle helpers: README + manifest + zip + checks
# + calibrated robustness/contour plotting (PLATT shifts boundary!)
# ============================================================

_FEAT_NAMES_R4 = ["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def zip_directory(src_dir: str, zip_path: str) -> str:
    """
    Zip whole directory. Does NOT include the zip itself (if zip_path is inside src_dir).
    Returns zip_path.
    """
    src_dir = os.path.abspath(src_dir)
    zip_path = os.path.abspath(zip_path)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                if os.path.abspath(full) == zip_path:
                    continue
                arc = os.path.relpath(full, src_dir)
                z.write(full, arcname=arc)

    return zip_path


def build_post_processing_manifest(
    *,
    out_dir: str,
    in_dir: str,
    run_id: str,
    calib_used: str,
    run_meta: Dict[str, Any],
    run_args: Dict[str, Any],
    pp_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a reproducibility manifest with SHA256 hashes for all files in OUT_DIR.
    Writes: OUT_DIR/post_processing_manifest.json
    """
    out_dir = os.path.abspath(out_dir)
    files: List[Dict[str, Any]] = []
    for root, _, fnames in os.walk(out_dir):
        for fn in sorted(fnames):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, out_dir).replace("\\", "/")
            try:
                st = os.stat(p)
                files.append(
                    {
                        "path": rel,
                        "bytes": int(st.st_size),
                        "sha256": _sha256_file(p),
                    }
                )
            except Exception:
                files.append({"path": rel, "bytes": None, "sha256": None})

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _now_utc_iso(),
        "run_id": run_id,
        "in_dir": os.path.abspath(in_dir),
        "out_dir": out_dir,
        "calibration_used": str(calib_used),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "libs": {
            "torch": getattr(torch, "__version__", "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "unknown"),
        },
        "run_meta_snapshot": run_meta or {},
        "run_args_snapshot": run_args or {},
        "post_processing_args": pp_args or {},
        "files": files,
    }

    out_path = os.path.join(out_dir, "post_processing_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def write_post_processing_readme_md(
    *,
    out_dir: str,
    in_dir: str,
    run_id: str,
    calib_used: str,
    manifest: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes: OUT_DIR/README_post_processing_real.md
    """
    out_dir = os.path.abspath(out_dir)
    rel = lambda p: os.path.relpath(p, out_dir).replace("\\", "/")

    def _exists(name: str) -> bool:
        return os.path.isfile(os.path.join(out_dir, name))

    key_files = [
        ("paper_summary.md", "Paper-ready summary (tables + key takeaways)."),
        ("README_post_processing_real.md", "This file."),
        ("post_processing_manifest.json", "SHA256 manifest of all bundle files."),
        ("run_command.txt", "Exact CLI + versions used to generate this bundle."),
        ("selected_uncertain_points.csv", "Which anchors were analyzed deeply (pp_id → index/record/etc)."),
        ("uncertain_points_overview.txt", "Global uncertain-set overview."),
        ("uncertain_points_by_record.csv", "Uncertain-set concentration by record."),
        ("uncertain_points_calibration_shift.csv", "up_real vs post_processing calibration consistency."),
        ("calibration_shift_summary.txt", "Calibration shift summary stats."),
        ("comparative_table.csv", "Per-method CV calibration CI table (mean ± 95% CI)."),
        ("calibration_ci_table.csv", "Same: record-level CV from up_real."),
        ("calibration_stats_tests.csv", "Wilcoxon pairwise tests (if scipy available)."),
        ("calibration_winrate_vs_none.csv", "ECE win-rate vs NONE."),
        ("sensitivity_summary.txt", "tau/delta selection summary."),
        ("kink_summary.csv", "Kink prevalence per anchor."),
        ("local_fit_summary.csv", "Local surrogate fidelity per anchor (+ residual stats)."),
        ("dominant_ratio_summary.csv", "r_dom vs observed flip + axis-baseline flips."),
        ("triage_error_auc_compare.csv", "AUPRC comparison for error detection among analyzed uncertain points."),
        ("triage_error_auc_diff_vs_uncert_margin.csv", "Bootstrap AUPRC deltas (score vs margin baseline)."),
        ("triage_error_budget_curve.csv", "Capture/risk vs review budget curves for multiple scores."),
        ("triage_error_scores_joined.csv", "Joined per-point table with all scores + labels used in triage eval."),
        ("triage_error_pr_curve.png", "Precision–Recall curves (error triage)."),
        ("resource_summary.csv", "Runtime summary per anchor (Puiseux vs saliency)."),
        ("kink_global_summary.txt", "Global kink summary across anchors."),
        ("missing_outputs_report.txt", "Missing per-point outputs audit (should be empty)."),
    ]

    lines: List[str] = []
    lines.append(f"# post_processing_real — paper bundle")
    lines.append("")
    lines.append(f"- run_id: **{run_id}**")
    lines.append(f"- calibration_used (up_real): **{calib_used}**")
    lines.append(f"- IN_DIR: `{os.path.abspath(in_dir)}`")
    lines.append(f"- OUT_DIR: `{out_dir}`")
    lines.append("")
    lines.append("## Key outputs (most important first)")
    lines.append("")
    for fn, desc in key_files:
        if _exists(fn):
            lines.append(f"- `{fn}` — {desc}")
    lines.append("")
    lines.append("## Per-point outputs")
    lines.append("")
    lines.append("Each analyzed anchor `pp_id=i` produces:")
    lines.append("- `benchmark_point{i}.txt` (main report)")
    lines.append("- `resource_point{i}.txt` (timings)")
    lines.append("- `local_surrogate_diag_point{i}.json` (fit quality + refit attempts)")
    lines.append("- `robustness_traces_point{i}.csv` + `robustness_summary_point{i}.csv` (calibrated traces)")
    lines.append("- `contour_point{i}_fix_dim=[1,3].png` and `contour_point{i}_fix_dim=[0,2].png` (calibrated boundary)")
    lines.append("")
    if manifest is not None:
        try:
            lines.append("## Manifest")
            lines.append("")
            lines.append(f"- files_in_manifest: {len(manifest.get('files', []))}")
            lines.append(f"- created_at_utc: `{manifest.get('created_at_utc')}`")
            lines.append("")
        except Exception:
            pass

    out_path = os.path.join(out_dir, "README_post_processing_real.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_missing_outputs_report(out_dir: str, *, n_points: int) -> None:
    """
    Writes a simple audit file listing missing expected per-point outputs.
    """
    out_dir = os.path.abspath(out_dir)
    patterns = {
        "benchmark_report": "benchmark_point{i}.txt",
        "resource_report": "resource_point{i}.txt",
        "local_surrogate_diag": "local_surrogate_diag_point{i}.json",
        "robustness_traces": "robustness_traces_point{i}.csv",
        "robustness_summary": "robustness_summary_point{i}.csv",
    }

    missing: Dict[str, List[int]] = {k: [] for k in patterns}
    for i in range(int(n_points)):
        for key, pat in patterns.items():
            p = os.path.join(out_dir, pat.format(i=i))
            if not os.path.isfile(p):
                missing[key].append(i)

    lines = []
    lines.append("=== Missing outputs audit (post_processing_real) ===")
    lines.append(f"n_points_expected = {int(n_points)}")
    lines.append("")
    any_missing = False
    for key, idxs in missing.items():
        if idxs:
            any_missing = True
            lines.append(f"- {key}: missing for points: {idxs}")
        else:
            lines.append(f"- {key}: OK (none missing)")
    lines.append("")
    if not any_missing:
        lines.append("All expected per-point outputs are present.")

    with open(os.path.join(out_dir, "missing_outputs_report.txt"), "w") as f:
        f.write("\n".join(lines))


# -------------------------
# Calibrated robustness & contour helpers (PLATT shifts decision boundary)
# -------------------------
def dir_radians_to_r4(dir_radians: Tuple[float, float]) -> np.ndarray:
    thx, thy = float(dir_radians[0]), float(dir_radians[1])
    # direction in C^2: (e^{i thx}, e^{i thy}) -> in R^4: [cos thx, cos thy, sin thx, sin thy]
    return np.asarray([np.cos(thx), np.cos(thy), np.sin(thx), np.sin(thy)], dtype=np.float32)


def robustness_trace_along_dir_calibrated(
    *,
    model: torch.nn.Module,
    xstar: np.ndarray,
    dir_radians: Tuple[float, float],
    radius: float,
    steps: int,
    device: torch.device,
    cal: Optional["Calibrator"],
    direction_id: int = 0,
    phase: float = float("nan"),
) -> Tuple[List[Dict[str, Any]], bool, Optional[float]]:
    x0 = np.asarray(xstar, dtype=np.float32).reshape(-1)
    v = dir_radians_to_r4(dir_radians).reshape(1, -1)

    rs = np.linspace(0.0, float(radius), int(steps) + 1, dtype=np.float32)
    X = x0.reshape(1, -1) + rs.reshape(-1, 1) * v

    _, _, probs_cal = predict_logits_and_proba(model, X, device, cal=cal)
    preds = np.argmax(probs_cal, axis=1).astype(int)
    y0 = int(preds[0])

    idx = np.where(preds != y0)[0]
    changed_radius = float(rs[idx[0]]) if len(idx) else None
    changed_class = bool(len(idx))

    rows: List[Dict[str, Any]] = []
    for j in range(len(rs)):
        p0 = float(probs_cal[j, 0])
        p1 = float(probs_cal[j, 1])
        pmax = float(max(p0, p1))
        rows.append(
            {
                "direction_id": int(direction_id),
                "phase": float(phase),
                "theta_x": float(dir_radians[0]),
                "theta_y": float(dir_radians[1]),
                "step": int(j),
                "radius": float(rs[j]),
                "p0_cal": p0,
                "p1_cal": p1,
                "pmax_cal": pmax,
                "margin_cal": float(2.0 * pmax - 1.0),
                "pred_cal": int(preds[j]),
                "changed": bool(preds[j] != y0),
            }
        )

    return rows, changed_class, changed_radius


def plot_robustness_traces_calibrated(traces_rows: List[Dict[str, Any]], save_path: str) -> bool:
    if not traces_rows:
        return False

    by_dir: Dict[int, List[Dict[str, Any]]] = {}
    for r in traces_rows:
        try:
            did = int(r.get("direction_id", -1))
        except Exception:
            did = -1
        by_dir.setdefault(did, []).append(r)

    plt.figure()
    for did, rows in sorted(by_dir.items(), key=lambda kv: kv[0]):
        rows_sorted = sorted(rows, key=lambda rr: float(rr.get("radius", 0.0)))
        xs = [float(rr.get("radius", 0.0)) for rr in rows_sorted]
        ys = [float(rr.get("p1_cal", np.nan)) for rr in rows_sorted]
        plt.plot(xs, ys, marker="o", markersize=2, linewidth=1, label=f"dir{did}")

    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("radius")
    plt.ylabel("P(class1) (calibrated)")
    plt.title("Robustness traces along candidate directions")
    if len(by_dir) <= 8:
        plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def plot_local_contour_2d_calibrated(
    model: torch.nn.Module,
    xstar: np.ndarray,
    *,
    fix_dims: Tuple[int, int],
    delta: float,
    steps: int,
    device: torch.device,
    cal: Optional["Calibrator"],
    save_path: str,
) -> bool:
    """
    2D contour of calibrated P(class1) in the 2 dims NOT in fix_dims.
    """
    x0 = np.asarray(xstar, dtype=np.float32).reshape(-1)
    if x0.shape[0] != 4:
        raise ValueError("xstar must be shape (4,)")

    fix_set = set(int(d) for d in fix_dims)
    vary = [d for d in range(4) if d not in fix_set]
    if len(vary) != 2:
        raise ValueError(f"fix_dims must fix exactly 2 dims; got {fix_dims} -> vary={vary}")

    d0, d1 = vary[0], vary[1]
    xs = np.linspace(float(x0[d0] - delta), float(x0[d0] + delta), int(steps))
    ys = np.linspace(float(x0[d1] - delta), float(x0[d1] + delta), int(steps))
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

    grid = np.tile(x0.reshape(1, 4), (Xg.size, 1))
    grid[:, d0] = Xg.reshape(-1)
    grid[:, d1] = Yg.reshape(-1)

    _, _, probs_cal = predict_logits_and_proba(model, grid.astype(np.float32), device, cal=cal)
    P = probs_cal[:, 1].reshape(int(steps), int(steps))

    plt.figure()
    plt.contourf(Xg, Yg, P, levels=20)
    plt.contour(Xg, Yg, P, levels=[0.5], linewidths=1)

    plt.scatter([x0[d0]], [x0[d1]], marker="x")
    title = (
        f"Calibrated P(class1) contour | vary=({_FEAT_NAMES_R4[d0]}, {_FEAT_NAMES_R4[d1]}) "
        f"| fixed {fix_dims[0]}={x0[fix_dims[0]]:.3f}, {fix_dims[1]}={x0[fix_dims[1]]:.3f}"
    )
    plt.title(title)
    plt.xlabel(_FEAT_NAMES_R4[d0])
    plt.ylabel(_FEAT_NAMES_R4[d1])
    plt.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True



@dataclass
class Calibrator:
    method: str

    # TEMPERATURE
    T: Optional[float] = None

    # PLATT: p = sigmoid(A*(logit1-logit0) + B)
    A: Optional[float] = None
    B: Optional[float] = None

    # BETA: p = sigmoid(a*log(p_raw) + b*log(1-p_raw) + c)
    beta_a: Optional[float] = None
    beta_b: Optional[float] = None
    beta_c: Optional[float] = None

    # VECTOR (binary): p = sigmoid(w0*logit0 + w1*logit1 + b)
    vec_w0: Optional[float] = None
    vec_w1: Optional[float] = None
    vec_b: Optional[float] = None

    # ISOTONIC on p_raw for class1
    iso: Optional[Any] = None


    def apply_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (N,2) raw logits from complex_modulus_to_logits(model(x))
        Returns logits' (N,2) such that softmax(logits') = calibrated probs.
        """
        m = (self.method or "NONE").upper()

        if m == "NONE":
            return logits

        if m == "TEMPERATURE":
            if self.T is None:
                return logits
            return logits / float(self.T)

        if m == "PLATT":
            if self.A is None or self.B is None:
                raise RuntimeError("PLATT requested but (A,B) not set.")
            d = logits[:, 1] - logits[:, 0]
            s = float(self.A) * d + float(self.B)
            z0 = torch.zeros_like(s)
            z1 = s
            return torch.stack([z0, z1], dim=1)

        if m == "VECTOR":
            # Binary vector scaling implemented as logistic regression on [logit0, logit1].
            if self.vec_w0 is None or self.vec_w1 is None or self.vec_b is None:
                raise RuntimeError("VECTOR requested but (vec_w0, vec_w1, vec_b) not set.")
            s = float(self.vec_w0) * logits[:, 0] + float(self.vec_w1) * logits[:, 1] + float(self.vec_b)
            z0 = torch.zeros_like(s)
            z1 = s
            return torch.stack([z0, z1], dim=1)

        if m in {"BETA", "ISOTONIC"}:
            # These are probability-domain calibrators; we map p_raw -> p_cal -> logits.
            probs_raw = torch.softmax(logits, dim=1)
            p = probs_raw[:, 1].clamp(1e-6, 1.0 - 1e-6)

            if m == "BETA":
                if self.beta_a is None or self.beta_b is None or self.beta_c is None:
                    raise RuntimeError("BETA requested but (beta_a, beta_b, beta_c) not set.")
                lp = torch.log(p)
                lq = torch.log(1.0 - p)
                s = float(self.beta_a) * lp + float(self.beta_b) * lq + float(self.beta_c)
                z0 = torch.zeros_like(s)
                z1 = s
                return torch.stack([z0, z1], dim=1)

            # ISOTONIC
            if self.iso is None:
                raise RuntimeError("ISOTONIC requested but iso model not set.")
            p_np = p.detach().cpu().numpy()
            p_cal_np = self.iso.predict(p_np)
            p_cal_np = np.clip(p_cal_np, 1e-6, 1.0 - 1e-6)
            s = torch.from_numpy(np.log(p_cal_np / (1.0 - p_cal_np))).to(logits.device, dtype=logits.dtype)
            z0 = torch.zeros_like(s)
            z1 = s
            return torch.stack([z0, z1], dim=1)



        raise NotImplementedError(f"Unsupported calibration method in post_processing: {m}")



def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Numerically safe logit. Works for scalar/array.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))



def fit_platt_from_full_val_npz(in_dir: str) -> Tuple[float, float]:
    full_val_npz = os.path.join(in_dir, "full_val_arrays.npz")
    if not os.path.exists(full_val_npz):
        raise FileNotFoundError(f"Missing {full_val_npz}")

    arr = np.load(full_val_npz)
    logits_raw = np.asarray(arr["logits_raw"], dtype=float)
    d = (logits_raw[:, 1] - logits_raw[:, 0]).reshape(-1)

    # Preferred: recover (A,B) from the calibrated probabilities produced by up_real (arr["probs"])
    # If up_real used Platt scaling, then logit(p_cal) = A*d + B exactly (up to clipping).
    if "probs" in arr.files:
        probs_cal = np.asarray(arr["probs"], dtype=float)
        if probs_cal.ndim == 2 and probs_cal.shape[1] >= 2:
            t = _safe_logit(probs_cal[:, 1], eps=1e-6)
            X = np.column_stack([d, np.ones_like(d)])
            A, B = np.linalg.lstsq(X, t, rcond=None)[0]
            return float(A), float(B)

    # Fallback (older artifacts): refit from labels
    y_true = np.asarray(arr["y_true"]).astype(int)
    X = d.reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, y_true)
    A = float(lr.coef_[0][0])
    B = float(lr.intercept_[0])
    return A, B



def fit_beta_from_full_val_npz(in_dir: str) -> Tuple[float, float, float]:
    full_val_npz = os.path.join(in_dir, "full_val_arrays.npz")
    if not os.path.exists(full_val_npz):
        raise FileNotFoundError(f"Missing {full_val_npz}")

    arr = np.load(full_val_npz)
    probs_raw = np.asarray(arr["probs_raw"], dtype=float)
    p_raw = np.clip(probs_raw[:, 1], 1e-6, 1.0 - 1e-6)

    # Preferred: match up_real's calibrated probs
    if "probs" in arr.files:
        probs_cal = np.asarray(arr["probs"], dtype=float)
        if probs_cal.ndim == 2 and probs_cal.shape[1] >= 2:
            t = _safe_logit(probs_cal[:, 1], eps=1e-6)
            X = np.column_stack([np.log(p_raw), np.log1p(-p_raw), np.ones_like(p_raw)])
            a, b, c = np.linalg.lstsq(X, t, rcond=None)[0]
            return float(a), float(b), float(c)

    # Fallback: refit from labels
    y_true = np.asarray(arr["y_true"]).astype(int)
    X = np.column_stack([np.log(p_raw), np.log1p(-p_raw)])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(X, y_true)
    a = float(lr.coef_[0][0])
    b = float(lr.coef_[0][1])
    c = float(lr.intercept_[0])
    return a, b, c



def fit_vector_from_full_val_npz(in_dir: str) -> Tuple[float, float, float]:
    full_val_npz = os.path.join(in_dir, "full_val_arrays.npz")
    if not os.path.exists(full_val_npz):
        raise FileNotFoundError(f"Missing {full_val_npz}")

    arr = np.load(full_val_npz)
    logits_raw = np.asarray(arr["logits_raw"], dtype=float)
    L0 = logits_raw[:, 0].reshape(-1)
    L1 = logits_raw[:, 1].reshape(-1)

    # Preferred: match up_real's calibrated probs
    if "probs" in arr.files:
        probs_cal = np.asarray(arr["probs"], dtype=float)
        if probs_cal.ndim == 2 and probs_cal.shape[1] >= 2:
            t = _safe_logit(probs_cal[:, 1], eps=1e-6)
            X = np.column_stack([L0, L1, np.ones_like(L0)])
            w0, w1, b = np.linalg.lstsq(X, t, rcond=None)[0]
            return float(w0), float(w1), float(b)

    # Fallback: refit from labels
    y_true = np.asarray(arr["y_true"]).astype(int)
    X = np.column_stack([L0, L1])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(X, y_true)
    w0 = float(lr.coef_[0][0])
    w1 = float(lr.coef_[0][1])
    b = float(lr.intercept_[0])
    return w0, w1, b



def fit_isotonic_from_full_val_npz(in_dir: str) -> IsotonicRegression:
    full_val_npz = os.path.join(in_dir, "full_val_arrays.npz")
    if not os.path.exists(full_val_npz):
        raise FileNotFoundError(f"Missing {full_val_npz}")

    arr = np.load(full_val_npz)
    probs_raw = np.asarray(arr["probs_raw"], dtype=float)
    p_raw = probs_raw[:, 1].reshape(-1)

    # Preferred: match up_real's calibrated probs
    if "probs" in arr.files:
        probs_cal = np.asarray(arr["probs"], dtype=float)
        if probs_cal.ndim == 2 and probs_cal.shape[1] >= 2:
            p_cal = probs_cal[:, 1].reshape(-1)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_raw, p_cal)
            return iso

    # Fallback: classic isotonic on labels
    y_true = np.asarray(arr["y_true"]).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_true)
    return iso




def plot_xy_with_ci(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    yerr_col: str = "",
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
) -> bool:
    """
    Minimal helper to generate paper convenience plots from CSV summaries.
    Returns True if plot was created.
    """
    if df is None or df.empty:
        return False
    if x_col not in df.columns or y_col not in df.columns:
        return False

    try:
        x = pd.to_numeric(df[x_col], errors="coerce").values
        y = pd.to_numeric(df[y_col], errors="coerce").values

        plt.figure()
        if yerr_col and yerr_col in df.columns:
            yerr = pd.to_numeric(df[yerr_col], errors="coerce").values
            plt.errorbar(x, y, yerr=yerr, fmt="o-")
        else:
            plt.plot(x, y, marker="o")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return False



def copy_if_exists(src_path: str, dst_dir: str, *, new_name: str = "") -> bool:
    """
    Copy file if it exists. Returns True if copied.
    """
    if not src_path or not os.path.isfile(src_path):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, new_name if new_name else os.path.basename(src_path))
    try:
        shutil.copy2(src_path, dst)
        return True
    except Exception:
        return False


def add_uncertainty_ranks(
    df_ext: pd.DataFrame,
    *,
    rank_by: str,
    review_budget: int,
) -> pd.DataFrame:
    """
    Adds:
      - rank_uncertainty: 0 = most uncertain (ascending by rank_by)
      - rank_metric: name of ranking key
      - in_review_budget: rank_uncertainty < review_budget
    """
    df = df_ext.copy()
    if rank_by not in df.columns:
        return df

    s = pd.to_numeric(df[rank_by], errors="coerce")
    # Lower score => more uncertain for: margin, pmax, abs_logit, abs_logit_cal
    r = s.rank(method="first", ascending=True, na_option="bottom").astype(int) - 1
    df["rank_uncertainty"] = r
    df["rank_metric"] = str(rank_by)

    rb = int(review_budget) if review_budget else 0
    df["in_review_budget"] = bool(rb) and (df["rank_uncertainty"] < rb)
    return df


def resolve_in_dir(output_folder: str) -> str:
    """
    Resolve actual directory with up_real artifacts.
    Works if `output_folder` is:
      - the run directory itself, OR
      - a parent directory containing run subfolders.
    """
    base = os.path.abspath(os.path.expanduser(output_folder))
    required = ["best_model_full.pt", "uncertain_full.csv"]

    def is_run_dir(d: str) -> bool:
        return all(os.path.isfile(os.path.join(d, r)) for r in required)

    if is_run_dir(base):
        return base

    candidates: List[str] = []
    # common layouts: <base>/*  and  <base>/runs/*
    for pat in [os.path.join(base, "*"), os.path.join(base, "runs", "*")]:
        for d in glob.glob(pat):
            if os.path.isdir(d) and is_run_dir(d):
                candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate up_real artifact directory.\n"
            f"Base: {base}\n"
            f"Expected at least: {required}\n"
            "Tried: base itself, base/*, base/runs/*"
        )

    def score(d: str) -> float:
        meta = _read_json_optional(os.path.join(d, "run_meta.json"))
        if meta and isinstance(meta.get("timestamp"), (int, float)):
            return float(meta["timestamp"])
        return float(os.path.getmtime(d))

    return max(candidates, key=score)

def _attach_uncertain_meta(up_list: List[Dict[str, Any]], df_ext: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Enrich uncertain points loaded from uncertain_full.csv (has X)
    with metadata from uncertain_full_ext.csv (record, window_in_record, pmax, margin, pred...).
    Join key: `index`.
    """
    if "index" not in df_ext.columns:
        return up_list

    # Build map: index -> row dict
    meta_map: Dict[int, Dict[str, Any]] = {}
    for _, r in df_ext.iterrows():
        try:
            idx = int(r["index"])
        except Exception:
            continue
        meta_map[idx] = r.to_dict()

    for u in up_list:
        try:
            idx = int(u.get("index"))
        except Exception:
            continue
        if idx in meta_map:
            u.update(meta_map[idx])
            # normalize a few keys
            if "record" in u:
                u["record"] = str(u["record"])
    return up_list

def _parse_point_indices(s: str) -> List[int]:
    """
    Parse comma-separated list like: "12,15, 200" -> [12,15,200]
    """
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out

def _parse_float_list(v: Any) -> List[float]:
    """
    Parse:
      - "0.005,0.01,0.02" -> [0.005,0.01,0.02]
      - [0.005, 0.01]     -> [0.005,0.01]
      - None/""           -> []
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out: List[float] = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out
    if isinstance(v, (int, float)):
        return [float(v)]
    s = str(v).strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            pass
    return out


def _select_uncertain_subset(
    up_list: List[Dict[str, Any]],
    *,
    max_points: int,
    seed: int,
    selection: str,
    explicit_indices: List[int],
    rank_by: str = "margin",
) -> List[Dict[str, Any]]:

    """
    Choose a manageable subset for deep Puiseux/LIME/SHAP analysis.
    - If explicit_indices provided -> keep those (in that order).
    - Else if len(up_list) <= max_points or max_points==0 -> keep all.
    - Else:
        per_record_mixed: stratify by record, prefer low margin, include errors if possible.
        worst_margin: global lowest margin first.
        random: random sample (seeded).
    """
    if explicit_indices:
        idx_set = set(explicit_indices)
        picked = [u for u in up_list if int(u.get("index", -1)) in idx_set]
        # preserve user order where possible
        order = {idx: k for k, idx in enumerate(explicit_indices)}
        picked.sort(key=lambda u: order.get(int(u.get("index", -1)), 10**9))
        return picked

    if max_points == 0 or len(up_list) <= max_points:
        return up_list

    rng = random.Random(int(seed))

    def score_key(u):
        # Prefer stable precomputed rank if present
        if "rank_uncertainty" in u:
            try:
                return float(u.get("rank_uncertainty", float("inf")))
            except Exception:
                pass
        # Otherwise: rank directly by chosen key
        try:
            v = u.get(rank_by, None)
            if v is None or str(v) == "nan":
                # fallback
                v = u.get("margin", None)
            return float(v)
        except Exception:
            return float("inf")


    if selection == "worst_margin":
        return sorted(up_list, key=score_key)[:max_points]

    if selection == "random":
        return rng.sample(up_list, k=max_points)

    # default: per_record_mixed
    by_rec: Dict[str, List[Dict[str, Any]]] = {}
    for u in up_list:
        rec = str(u.get("record", "NA"))
        by_rec.setdefault(rec, []).append(u)

    recs = sorted(by_rec.keys())
    k_base = max_points // max(1, len(recs))
    k_rem  = max_points % max(1, len(recs))

    selected: List[Dict[str, Any]] = []
    used_idx: set = set()

    for j, rec in enumerate(recs):
        lst = by_rec[rec]
        lst_sorted = sorted(lst, key=score_key)

        # split into errors / correct if pred/true_label exist
        err = []
        ok  = []
        for u in lst_sorted:
            try:
                is_err = int(u.get("pred", -999)) != int(u.get("true_label", -999))
            except Exception:
                is_err = False
            (err if is_err else ok).append(u)

        k = k_base + (1 if j < k_rem else 0)
        k_err = min(len(err), k // 2)
        k_ok  = min(len(ok),  k - k_err)

        chosen = err[:k_err] + ok[:k_ok]

        # if still short (e.g., not enough ok/err), top up from remaining in order
        if len(chosen) < k:
            for u in lst_sorted:
                if u in chosen:
                    continue
                chosen.append(u)
                if len(chosen) >= k:
                    break

        for u in chosen:
            idx = int(u.get("index", -1))
            if idx in used_idx:
                continue
            used_idx.add(idx)
            selected.append(u)

    # If still short (due to duplicates/empty rec), fill globally by worst margin
    if len(selected) < max_points:
        rest = [u for u in sorted(up_list, key=score_key) if int(u.get("index", -1)) not in used_idx]
        selected.extend(rest[: (max_points - len(selected))])

    return selected[:max_points]

def _bootstrap_auprc_ci(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for AUPRC.

    Returns (auprc, ci_low, ci_high). If bootstrap is disabled or degenerate,
    CI bounds are returned as NaN.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score):
        raise ValueError('y and score must be 1D arrays of the same length')
    if len(y) == 0 or y.sum() == 0:
        return float('nan'), float('nan'), float('nan')

    ap = float(average_precision_score(y, score))
    if n_boot <= 0:
        return ap, float('nan'), float('nan')

    rng = np.random.default_rng(int(seed))
    n = len(y)
    aps = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        # Skip degenerate resamples
        if y_b.sum() == 0:
            continue
        aps.append(float(average_precision_score(y_b, score[idx])))

    if len(aps) < 10:
        return ap, float('nan'), float('nan')

    lo, hi = np.percentile(aps, [2.5, 97.5])
    return ap, float(lo), float(hi)


def _bootstrap_auprc_diff_ci(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for ΔAUPRC = AUPRC(a) - AUPRC(b)."""
    y = np.asarray(y).astype(int)
    score_a = np.asarray(score_a).astype(float)
    score_b = np.asarray(score_b).astype(float)
    if len(y) == 0 or y.sum() == 0:
        return float('nan'), float('nan'), float('nan')

    base = float(average_precision_score(y, score_a) - average_precision_score(y, score_b))
    if n_boot <= 0:
        return base, float('nan'), float('nan')

    rng = np.random.default_rng(int(seed))
    n = len(y)
    diffs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() == 0:
            continue
        diffs.append(
            float(
                average_precision_score(y_b, score_a[idx])
                - average_precision_score(y_b, score_b[idx])
            )
        )
    if len(diffs) < 10:
        return base, float('nan'), float('nan')

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, float(lo), float(hi)


def _triage_budget_curve(
    y: np.ndarray,
    score: np.ndarray,
    *,
    max_k: Optional[int] = None,
) -> pd.DataFrame:
    """Compute capture / risk_accept / precision_review vs top-k review budget."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    n = len(y)
    if max_k is None:
        max_k = n
    max_k = int(min(max_k, n))

    order = np.argsort(-score)  # higher score => review earlier
    total_err = int(y.sum())

    rows = []
    for k in range(1, max_k + 1):
        reviewed = order[:k]
        accepted = order[k:]
        err_review = int(y[reviewed].sum())
        err_accept = int(y[accepted].sum())
        n_accept = int(len(accepted))
        rows.append(
            {
                'k_review': k,
                'review_frac': k / n if n > 0 else float('nan'),
                'capture': (err_review / total_err) if total_err > 0 else float('nan'),
                'precision_review': (err_review / k) if k > 0 else float('nan'),
                'risk_accept': (err_accept / n_accept) if n_accept > 0 else float('nan'),
                'n_accept': n_accept,
                'total_err': total_err,
            }
        )
    return pd.DataFrame(rows)


def run_puiseux_error_triage_eval(
    out_dir: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    pr_max_curves: int = 4,
) -> Optional[pd.DataFrame]:
    """Quantify the *practical* gain of Puiseux on real data.

    We treat the analyzed (selected) uncertain points as an "audit set".
    Task: prioritize *misclassifications* (is_error=1) for human review.

    Inputs:
      - selected_uncertain_points.csv (contains is_error, pmax, margin)
      - dominant_ratio_summary.csv (contains r_flip_cens, r_dom, etc.)

    Outputs (written to out_dir):
      - triage_error_auc_compare.csv
      - triage_error_auc_diff_vs_uncert_margin.csv
      - triage_error_budget_curve.csv
      - triage_error_pr_curve.png
    """
    dom_path = os.path.join(out_dir, 'dominant_ratio_summary.csv')
    sel_path = os.path.join(out_dir, 'selected_uncertain_points.csv')
    if not (os.path.isfile(dom_path) and os.path.isfile(sel_path)):
        return None

    dom = pd.read_csv(dom_path)
    sel = pd.read_csv(sel_path)
    df = dom.merge(sel, left_on='point', right_on='pp_id', how='inner')
    if 'is_error' not in df.columns:
        return None

    y = df['is_error'].astype(int).values
    if y.sum() == 0:
        logger.warning('[TRIAGE] No errors in selected_uncertain_points -> skipping triage eval')
        return None

    # Effective Puiseux flip radius (censored at the scan budget when no flip is observed)
    r_flip_puiseux = df['r_flip_cens'].astype(float).fillna(0.05).values
    score_puiseux = 1.0 / np.maximum(r_flip_puiseux, 1e-12)

    # Baselines
    score_margin = (1.0 - df['margin'].astype(float)).values
    score_pmax = (1.0 - df['pmax'].astype(float)).values
    score_grad_norm = df.get('saliency_grad_norm', pd.Series(np.zeros(len(df)))).astype(float).values

    # XAI-ray baseline (min flip radius over {grad, LIME-axis, SHAP-axis})
    xai_cols = [c for c in ['flip_grad', 'flip_lime', 'flip_shap'] if c in df.columns]
    if len(xai_cols) > 0:
        r_flip_xai = df[xai_cols].min(axis=1, skipna=True).astype(float).fillna(0.05).values
        score_xai = 1.0 / np.maximum(r_flip_xai, 1e-12)
    else:
        score_xai = None

    # Puiseux dominance proxy (smaller r_dom => more quartic-dominant locally)
    if 'r_dom' in df.columns:
        score_inv_r_dom = 1.0 / np.maximum(df['r_dom'].astype(float).values, 1e-12)
    else:
        score_inv_r_dom = None

    scores: Dict[str, np.ndarray] = {
        'puiseux_inv_rflip': score_puiseux,
        'uncert_margin': score_margin,
        'uncert_pmax': score_pmax,
        'grad_norm': score_grad_norm,
    }
    if score_xai is not None:
        scores['xai_inv_rflip'] = score_xai
    if score_inv_r_dom is not None:
        scores['puiseux_inv_rdom'] = score_inv_r_dom

    # AUPRC table + bootstrap CI
    auc_rows = []
    for name, s in scores.items():
        ap, lo, hi = _bootstrap_auprc_ci(y, s, n_boot=n_boot, seed=seed)
        auc_rows.append({'score': name, 'auprc': ap, 'ci_low': lo, 'ci_high': hi})
    auc_df = pd.DataFrame(auc_rows).sort_values('auprc', ascending=False)
    auc_df.to_csv(os.path.join(out_dir, 'triage_error_auc_compare.csv'), index=False)

    # Pairwise ΔAUPRC vs a strong, standard baseline (margin)
    if 'uncert_margin' in scores:
        diff_rows = []
        for name, s in scores.items():
            if name == 'uncert_margin':
                continue
            d, lo, hi = _bootstrap_auprc_diff_ci(y, s, scores['uncert_margin'], n_boot=n_boot, seed=seed)
            diff_rows.append({'score': name, 'delta_auprc_vs_margin': d, 'ci_low': lo, 'ci_high': hi})
        pd.DataFrame(diff_rows).sort_values('delta_auprc_vs_margin', ascending=False).to_csv(
            os.path.join(out_dir, 'triage_error_auc_diff_vs_uncert_margin.csv'), index=False
        )

    # Budget curve
    curves = []
    max_k = len(y)
    for name, s in scores.items():
        c = _triage_budget_curve(y, s, max_k=max_k)
        c.insert(0, 'score', name)
        curves.append(c)
    curve_df = pd.concat(curves, ignore_index=True)
    curve_df.to_csv(os.path.join(out_dir, 'triage_error_budget_curve.csv'), index=False)

    # PR plot (top-k curves by AUPRC)
    try:
        top_names = auc_df['score'].head(int(pr_max_curves)).tolist()
        plt.figure()
        for name in top_names:
            s = scores[name]
            prec, rec, _ = precision_recall_curve(y, s)
            ap = float(average_precision_score(y, s))
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        base = float(y.mean())
        plt.hlines(base, 0, 1, linestyles='dashed', label=f"prevalence={base:.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Error triage on selected uncertain points (audit set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'triage_error_pr_curve.png'), dpi=200)
        plt.close()
    except Exception as e:
        logger.warning('[TRIAGE] Could not create PR curve plot: %s', e)

    logger.info('[TRIAGE] Wrote triage_error_auc_compare.csv + triage_error_budget_curve.csv')
    return auc_df



########################################################################
# MAIN SCRIPT
########################################################################
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 0) Parse args, determinism & device
    # ----------------------------------------------------------------------
    args = parse_pp_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # Determinism and numerical stability.
    random.seed(getattr(args, "seed", 12345))
    np.random.seed(getattr(args, "seed", 12345))
    torch.manual_seed(getattr(args, "seed", 12345))
    torch.cuda.manual_seed_all(getattr(args, "seed", 12345))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)  # katalog nadrzędny projektu
    DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "post_processing_real")

    OUT_DIR = os.path.abspath(args.out_dir) if getattr(args, "out_dir", "") else DEFAULT_OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    # Resolve actual up_real artifact dir (can be a run dir or a parent dir).
    IN_DIR = resolve_in_dir(getattr(args, "output_folder", "up_real"))

    # Read run metadata if present (used later for calibration/provenance).
    _run_meta = _read_json_optional(os.path.join(IN_DIR, "run_meta.json")) or {}
    _run_args = _read_json_optional(os.path.join(IN_DIR, "run_args.json")) or {}

    run_id = infer_run_id(IN_DIR, _run_meta, _run_args)



    # == Logging: to file + console ==
    log_path = os.path.join(OUT_DIR, "post_processing_real.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("pp_real")
    logger.info("Log file: %s", log_path)


    logger.info("Resolved IN_DIR: %s", IN_DIR)
    logger.info("Resolved OUT_DIR: %s", OUT_DIR)

    if args.triage_only_eval:
        if args.skip_triage_eval:
            logger.warning("--triage_only_eval set, but --skip_triage_eval is also set; nothing to do.")
            sys.exit(0)

        try:
            run_puiseux_error_triage_eval(
                out_dir=OUT_DIR,
                n_boot=int(args.triage_bootstrap),
                seed=int(args.seed),
                pr_max_curves=int(args.triage_pr_max_curves),
            )
        except Exception as e:
            logger.warning("Puiseux error-triage eval failed: %s", e)

        sys.exit(0)

    # Persist run args/meta snapshot for paper reproducibility
    try:
        if _run_meta:
            with open(os.path.join(OUT_DIR, "run_meta.snapshot.json"), "w") as f:
                json.dump(_run_meta, f, indent=2, sort_keys=True)
        if _run_args:
            with open(os.path.join(OUT_DIR, "run_args.snapshot.json"), "w") as f:
                json.dump(_run_args, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Failed to write run snapshot JSONs: %s", e)

    # Copy key provenance files into OUT_DIR (paper appendix convenience)
    copy_if_exists(os.path.join(IN_DIR, "artifact_manifest.json"), OUT_DIR, new_name="artifact_manifest.snapshot.json")
    copy_if_exists(os.path.join(IN_DIR, "run.log"), OUT_DIR, new_name="run.snapshot.log")
    copy_if_exists(os.path.join(IN_DIR, "pip_freeze.txt"), OUT_DIR, new_name="pip_freeze.snapshot.txt")


    # Paper reproducibility: exact CLI + versions + this script snapshot
    try:
        with open(os.path.join(OUT_DIR, "run_command.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
            f.write(f"python_executable: {sys.executable}\n")
            f.write(f"python_version: {sys.version}\n")
            f.write(f"torch_version: {torch.__version__}\n")
            f.write(f"numpy_version: {np.__version__}\n")
            f.write(f"pandas_version: {pd.__version__}\n")
    except Exception as e:
        logger.warning("Failed to write run_command.txt: %s", e)

    try:
        copy_if_exists(os.path.abspath(__file__), OUT_DIR, new_name="post_processing_real.snapshot.py")
    except Exception as e:
        logger.warning("Failed to snapshot script into OUT_DIR: %s", e)



    # ----------------------------------------------------------------------
    # 1) Load the trained model, temperature and scaler
    # ----------------------------------------------------------------------
    model_path = os.path.join(IN_DIR, 'best_model_full.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SimpleComplexNet(
        in_features=2,
        hidden_features=64,
        out_features=2,
        bias=0.1
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Calibration used in up_real (main run).
    # We reconstruct a compatible calibrator so that post-processing analyzes the SAME decision system.
    calib_used = str((_run_args or {}).get("calibration", "UNKNOWN")).upper()
    cal = Calibrator(method=calib_used)

    # Backward-compat: keep T variable (used by some helper functions), but primary is `cal`.
    T = None

    if cal.method == "TEMPERATURE":
        T_path = os.path.join(IN_DIR, "T_calib.pt")
        if os.path.isfile(T_path):
            T_tensor = torch.load(T_path, map_location="cpu")
            cal.T = float(T_tensor.item())
            T = T_tensor.to(device)
            logger.info("Loaded TEMPERATURE T=%.6f from %s", cal.T, T_path)
            print(f"[INFO] Calibration= TEMPERATURE, loaded T={cal.T:.3f}")
        else:
            logger.warning("Calibration=TEMPERATURE but T_calib.pt not found -> falling back to NONE.")
            cal.method = "NONE"
            print("[WARN] Calibration=TEMPERATURE but T_calib.pt not found; proceeding with NONE.")
    elif cal.method == "PLATT":
        try:
            A, B = fit_platt_from_full_val_npz(IN_DIR)
            cal.A, cal.B = A, B
            thr = (-B / A) if abs(A) > 1e-12 else float("nan")

            # Save params for reproducibility
            with open(os.path.join(OUT_DIR, "platt_params.json"), "w") as f:
                json.dump({"A": A, "B": B, "threshold_logit_diff_at_p05": thr}, f, indent=2)

            logger.info("Fitted PLATT from full_val_arrays.npz: A=%.6f B=%.6f thr(d)@p=0.5=%.6f", A, B, thr)
            print(f"[INFO] Calibration used in up_real: PLATT (fitted A,B from full_val_arrays.npz)")
        except Exception as e:
            logger.warning("Failed to fit PLATT -> falling back to NONE: %s", e)
            cal.method = "NONE"
            print("[WARN] Calibration=PLATT but fitting failed; proceeding with NONE.")
    elif cal.method == "BETA":
        try:
            a, b, c = fit_beta_from_full_val_npz(IN_DIR)
            cal.beta_a, cal.beta_b, cal.beta_c = a, b, c
            with open(os.path.join(OUT_DIR, "beta_params.json"), "w") as f:
                json.dump({"a": a, "b": b, "c": c}, f, indent=2)
            logger.info("Fitted BETA from full_val_arrays.npz: a=%.6f b=%.6f c=%.6f", a, b, c)
            print("[INFO] Calibration used in up_real: BETA (fitted a,b,c from full_val_arrays.npz)")
        except Exception as e:
            logger.warning("Failed to fit BETA -> falling back to NONE: %s", e)
            cal.method = "NONE"
            print("[WARN] Calibration=BETA but fitting failed; proceeding with NONE.")

    elif cal.method == "VECTOR":
        try:
            w0, w1, b = fit_vector_from_full_val_npz(IN_DIR)
            cal.vec_w0, cal.vec_w1, cal.vec_b = w0, w1, b
            with open(os.path.join(OUT_DIR, "vector_params.json"), "w") as f:
                json.dump({"w0": w0, "w1": w1, "b": b}, f, indent=2)
            logger.info("Fitted VECTOR from full_val_arrays.npz: w0=%.6f w1=%.6f b=%.6f", w0, w1, b)
            print("[INFO] Calibration used in up_real: VECTOR (fitted w0,w1,b from full_val_arrays.npz)")
        except Exception as e:
            logger.warning("Failed to fit VECTOR -> falling back to NONE: %s", e)
            cal.method = "NONE"
            print("[WARN] Calibration=VECTOR but fitting failed; proceeding with NONE.")

    elif cal.method == "ISOTONIC":
        try:
            iso = fit_isotonic_from_full_val_npz(IN_DIR)
            cal.iso = iso
            # Persist for reproducibility
            with open(os.path.join(OUT_DIR, "isotonic_calibrator.pkl"), "wb") as f:
                pickle.dump(iso, f)
            logger.info("Fitted ISOTONIC from full_val_arrays.npz")
            print("[INFO] Calibration used in up_real: ISOTONIC (fitted iso from full_val_arrays.npz)")
        except Exception as e:
            logger.warning("Failed to fit ISOTONIC -> falling back to NONE: %s", e)
            cal.method = "NONE"
            print("[WARN] Calibration=ISOTONIC but fitting failed; proceeding with NONE.")

    else:
        logger.warning("Calibration %s not supported in post_processing -> using NONE", cal.method)
        print(f"[INFO] Calibration used in up_real: {cal.method} (not supported here -> using NONE)")
        cal.method = "NONE"




    # Scaler (keeps feature space consistent for LIME/SHAP and CV).
    scaler_path = os.path.join(IN_DIR, "scaler_full.pkl")
    if os.path.isfile(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler_full = pickle.load(f)
        print("[INFO] Loaded scaler_full.pkl")
    else:
        scaler_full = None
        print("[WARN] scaler_full.pkl not found – explanations/CV will run on unscaled features.")

    model.eval()
    print(f"[INFO] Loaded MIT-BIH parameters from {model_path}")

    # ----------------------------------------------------------------------
    # 2) Load anchors (default: uncertain_full.csv, but can override via --anchors_csv)
    # ----------------------------------------------------------------------
    anchors_csv_arg = str(getattr(args, "anchors_csv", "") or "").strip()

    if anchors_csv_arg:
        unc_csv = anchors_csv_arg
        if not os.path.isabs(unc_csv):
            unc_csv = os.path.join(IN_DIR, anchors_csv_arg)
    else:
        unc_csv = os.path.join(IN_DIR, "uncertain_full.csv")
    if not os.path.isfile(unc_csv):
        raise FileNotFoundError(f"Anchors CSV not found: {unc_csv}")

    up_list = load_uncertain_points(unc_csv)


    # BSPC PATCH: normalize X to a numeric list in case it was saved as a string in CSV
    n_bad_x = 0
    for _u in up_list:
        try:
            _x = _parse_X_value(_u.get("X", None), anchor=_u)
            if _x is None:
                n_bad_x += 1
            else:
                _u["X"] = _x
        except Exception:
            n_bad_x += 1
    if n_bad_x > 0:
        logger.warning("Parsed anchors: %d/%d have invalid/unparsed X; they may fail later.", n_bad_x, len(up_list))
    logger.info("Loaded %d anchors from %s", len(up_list), unc_csv)

    # Attach metadata if available:
    #  - default uncertain run: use uncertain_full_ext.csv
    #  - otherwise: try to use the anchors CSV itself (if it contains pmax/margin/pred/accepted/is_error/etc.)
    review_budget = int((_run_args or {}).get("review_budget", 0) or 0)
    rank_by = str(getattr(args, "rank_by", "margin"))

    df_unc_ext = None
    unc_ext_csv = os.path.join(IN_DIR, "uncertain_full_ext.csv")
    use_default_uncertain = (os.path.basename(unc_csv) == "uncertain_full.csv")

    if os.path.isfile(unc_ext_csv) and use_default_uncertain:
        try:
            df_unc_ext = pd.read_csv(unc_ext_csv)
            df_unc_ext = add_uncertainty_ranks(df_unc_ext, rank_by=rank_by, review_budget=review_budget)
            up_list = _attach_uncertain_meta(up_list, df_unc_ext)
            logger.info("Loaded uncertain_full_ext.csv and attached metadata (rows=%d).", len(df_unc_ext))
        except Exception as e:
            df_unc_ext = None
            logger.warning("Failed to load/attach uncertain_full_ext.csv: %s", e)


    if df_unc_ext is None:
        # Fallback: use anchors CSV itself as metadata (works for full_test_predictions_ext.csv etc.)
        try:
            df_unc_ext = pd.read_csv(unc_csv)
            if "index" in df_unc_ext.columns:
                if ("pmax" in df_unc_ext.columns) or ("margin" in df_unc_ext.columns):
                    df_unc_ext = add_uncertainty_ranks(df_unc_ext, rank_by=rank_by, review_budget=review_budget)
                up_list = _attach_uncertain_meta(up_list, df_unc_ext)
                logger.info("Attached metadata from anchors CSV itself (rows=%d).", len(df_unc_ext))
            else:
                logger.warning("Anchors CSV has no 'index' column; skipping meta attach.")
                df_unc_ext = None
        except Exception as e:
            df_unc_ext = None
            logger.warning("No metadata attached (failed to read anchors CSV as metadata): %s", e)

    # Optional anchor filtering (pred / pmax / accepted) — useful for “silent-failure” experiments.
    f_pred = int(getattr(args, "filter_pred", -1) or -1)
    f_true = int(getattr(args, "filter_true", -1) or -1)
    f_pmax_min = float(getattr(args, "filter_pmax_min", float("nan")))
    f_pmax_max = float(getattr(args, "filter_pmax_max", float("nan")))
    f_accept_only = bool(getattr(args, "filter_accepted_only", False))


    def _safe_int(v, default=None):
        try:
            return int(v)
        except Exception:
            return default

    def _safe_float(v, default=np.nan):
        try:
            return float(v)
        except Exception:
            return default

    up_list_before = len(up_list)
    up_list_filt = []
    for u in up_list:
        if f_pred != -1:
            if _safe_int(u.get("pred", None), default=10**9) != f_pred:
                continue
        if f_true != -1:
            if _safe_int(u.get("true_label", None), default=10**9) != f_true:
                continue
        if np.isfinite(f_pmax_min):
            if _safe_float(u.get("pmax", np.nan)) < f_pmax_min:
                continue
        if np.isfinite(f_pmax_max):
            if _safe_float(u.get("pmax", np.nan)) > f_pmax_max:
                continue
        if f_accept_only:
            acc = u.get("accepted", 0)
            try:
                acc = int(acc)
            except Exception:
                acc = 1 if bool(acc) else 0
            if acc != 1:
                continue
        up_list_filt.append(u)

    if len(up_list_filt) != up_list_before:
        logger.info("Anchor filters applied: %d -> %d", up_list_before, len(up_list_filt))
    up_list = up_list_filt

    # If requested: analyze exactly review_budget points (top-k by rank_by).
    # This only makes semantic sense for uncertain_full.csv (VAL-selected review set).
    if bool(getattr(args, "use_review_budget", False)) and use_default_uncertain:
        if review_budget <= 0:
            logger.warning("review_budget not found in run_args; ignoring --use_review_budget.")
        else:
            logger.info("Using review_budget=%d (exact count) for point selection.", review_budget)
            args.max_points = int(review_budget)
            args.selection = "worst_margin" if (rank_by == "margin") else "worst_pmax"
            logger.info("Auto-set: --max_points=%d, --selection=%s", args.max_points, args.selection)

    # ----------------------------------------------------------------------
    # 2b) Calibration-shift diagnostics (up_real probs vs raw model softmax)
    # ----------------------------------------------------------------------
    try:
        X_unc = np.asarray([u.get("X") for u in up_list], dtype=np.float32)

        logits_np, probs_raw_np, probs_cal_np = predict_logits_and_proba(model, X_unc, device, cal=cal)

        for u, lg, pr_raw, pr_cal in zip(up_list, logits_np, probs_raw_np, probs_cal_np):
            u["logit0_raw"] = float(lg[0])
            u["logit1_raw"] = float(lg[1])
            u["logit_diff_raw"] = float(lg[1] - lg[0])

            # RAW
            u["p1_raw"] = float(pr_raw[0])
            u["p2_raw"] = float(pr_raw[1])
            u["pmax_raw"] = float(max(pr_raw[0], pr_raw[1]))
            u["margin_raw"] = float(2.0 * u["pmax_raw"] - 1.0)

            # CAL (PLATT or TEMPERATURE)
            u["p1_cal"] = float(pr_cal[0])
            u["p2_cal"] = float(pr_cal[1])
            u["pmax_cal"] = float(max(pr_cal[0], pr_cal[1]))
            u["margin_cal"] = float(2.0 * u["pmax_cal"] - 1.0)

            # Effective calibrated logit-diff (log-odds) for ranking / debugging
            try:
                p0c = float(u["p1_cal"])
                p1c = float(u["p2_cal"])
                p0c = max(p0c, 1e-12)
                p1c = max(p1c, 1e-12)
                u["logit_diff_cal"] = float(np.log(p1c / p0c))
            except Exception:
                u["logit_diff_cal"] = float("nan")

            u["abs_logit"] = float(abs(u.get("logit_diff_raw", float("nan"))))
            u["abs_logit_cal"] = float(abs(u.get("logit_diff_cal", float("nan"))))


        rows_shift = []
        for u in up_list:
            # up_real stores calibrated p1/p2 in uncertain_full.csv
            p1_up = float(u.get("p1", np.nan))
            p2_up = float(u.get("p2", np.nan))
            pmax_up = float(max(p1_up, p2_up)) if np.isfinite(p1_up) and np.isfinite(p2_up) else float("nan")
            margin_up = float(2.0 * pmax_up - 1.0) if np.isfinite(pmax_up) else float("nan")

            rows_shift.append({
                "index": u.get("index"),
                "record": u.get("record", "NA"),
                "window_in_record": u.get("window_in_record", np.nan),
                "true_label": u.get("true_label", np.nan),
                "pred": u.get("pred", np.nan),

                "calibration_used": calib_used,

                "p1_up": p1_up,
                "p2_up": p2_up,
                "pmax_up": pmax_up,
                "margin_up": margin_up,

                "p1_raw": float(u.get("p1_raw", np.nan)),
                "p2_raw": float(u.get("p2_raw", np.nan)),
                "pmax_raw": float(u.get("pmax_raw", np.nan)),
                "margin_raw": float(u.get("margin_raw", np.nan)),

                "p1_cal": float(u.get("p1_cal", np.nan)),
                "p2_cal": float(u.get("p2_cal", np.nan)),
                "pmax_cal": float(u.get("pmax_cal", np.nan)),
                "margin_cal": float(u.get("margin_cal", np.nan)),

                "logit_diff_raw": float(u.get("logit_diff_raw", np.nan)),
                "logit_diff_cal": float(u.get("logit_diff_cal", np.nan)),
                "abs_logit": float(u.get("abs_logit", np.nan)),
                "abs_logit_cal": float(u.get("abs_logit_cal", np.nan)),
            })

        df_shift = pd.DataFrame(rows_shift)
        df_shift.to_csv(os.path.join(OUT_DIR, "uncertain_points_calibration_shift.csv"), index=False)

        # Summary txt
        d_up_vs_raw = pd.to_numeric(df_shift["p2_up"], errors="coerce") - pd.to_numeric(df_shift["p2_raw"], errors="coerce")
        d_up_vs_cal = pd.to_numeric(df_shift["p2_up"], errors="coerce") - pd.to_numeric(df_shift["p2_cal"], errors="coerce")

        with open(os.path.join(OUT_DIR, "calibration_shift_summary.txt"), "w") as f:
            f.write("=== Calibration shift diagnostics (up_real vs post_processing) ===\n")
            f.write(f"calibration_used(up_real) = {calib_used}\n")
            f.write(f"post_processing_cal = {cal.method}\n")
            f.write(f"n_uncertain = {len(df_shift)}\n\n")

            f.write("Δp2(up - raw): p2_up - p2_raw\n")
            f.write(d_up_vs_raw.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).to_string())
            f.write("\n\n")

            f.write("Δp2(up - cal): p2_up - p2_cal  (should be ~0 if we reconstructed calibration correctly)\n")
            f.write(d_up_vs_cal.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).to_string())
            f.write("\n")


        logger.info("Saved calibration shift -> uncertain_points_calibration_shift.csv + calibration_shift_summary.txt")
    except Exception as e:
        logger.warning("Calibration shift diagnostics failed: %s", e)



    # Paper-friendly overview of uncertain set
    try:
        n_all = len(up_list)
        n_err = 0
        margins = []
        pmaxs = []
        rec_counts = {}
        for u in up_list:
            try:
                if int(u.get("pred", -999)) != int(u.get("true_label", -999)):
                    n_err += 1
            except Exception:
                pass
            try:
                margins.append(float(u.get("margin")))
            except Exception:
                pass
            try:
                pmaxs.append(float(u.get("pmax")))
            except Exception:
                pass
            rec = str(u.get("record", "NA"))
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

        with open(os.path.join(OUT_DIR, "uncertain_points_overview.txt"), "w") as f:
            f.write("=== Uncertain points overview (from up_real) ===\n")
            f.write(f"run_id: {run_id}\n")
            f.write(f"calibration_used: {calib_used}\n")
            f.write(f"n_uncertain: {n_all}\n")
            if n_all > 0:
                f.write(f"error_rate_in_uncertain: {n_err / n_all:.4f} (n_err={n_err})\n")
            if margins:
                f.write(f"margin: min={np.min(margins):.4f}, q25={np.quantile(margins,0.25):.4f}, "
                        f"median={np.median(margins):.4f}, q75={np.quantile(margins,0.75):.4f}, max={np.max(margins):.4f}\n")
            if pmaxs:
                f.write(f"pmax:   min={np.min(pmaxs):.4f}, q25={np.quantile(pmaxs,0.25):.4f}, "
                        f"median={np.median(pmaxs):.4f}, q75={np.quantile(pmaxs,0.75):.4f}, max={np.max(pmaxs):.4f}\n")
            # Top records
            top = sorted(rec_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            f.write("\nTop records in uncertain set:\n")
            for rec, cnt in top:
                f.write(f"  record {rec}: {cnt}\n")

            # Also save per-record breakdown as CSV (count + error rate + median pmax/margin)
            try:
                rec_err = {}
                rec_pmax = {}
                rec_margin = {}

                # rebuild stats from up_list (all uncertain points)
                for u in up_list:
                    rec = str(u.get("record", "NA"))
                    rec_err.setdefault(rec, 0)
                    rec_pmax.setdefault(rec, [])
                    rec_margin.setdefault(rec, [])

                    try:
                        if int(u.get("pred", -999)) != int(u.get("true_label", -999)):
                            rec_err[rec] += 1
                    except Exception:
                        pass

                    try:
                        rec_pmax[rec].append(float(u.get("pmax")))
                    except Exception:
                        pass
                    try:
                        rec_margin[rec].append(float(u.get("margin")))
                    except Exception:
                        pass

                rows_rec = []
                for rec, cnt in sorted(rec_counts.items(), key=lambda kv: kv[1], reverse=True):
                    n_err = int(rec_err.get(rec, 0))
                    er = (n_err / cnt) if cnt > 0 else float("nan")
                    pm = np.median(rec_pmax.get(rec, [])) if len(rec_pmax.get(rec, [])) else float("nan")
                    mm = np.median(rec_margin.get(rec, [])) if len(rec_margin.get(rec, [])) else float("nan")
                    rows_rec.append({
                        "record": rec,
                        "n_uncertain": int(cnt),
                        "n_err": int(n_err),
                        "err_rate": float(er),
                        "pmax_median": float(pm),
                        "margin_median": float(mm),
                    })

                pd.DataFrame(rows_rec).to_csv(os.path.join(OUT_DIR, "uncertain_points_by_record.csv"), index=False)
            except Exception as e:
                logger.warning("Failed to write uncertain_points_by_record.csv: %s", e)



    except Exception as e:
        logger.warning("Failed to write uncertain_points_overview.txt: %s", e)

    # Select a manageable subset for deep analysis (Puiseux/LIME/SHAP/benchmark).
    explicit_idx = _parse_point_indices(getattr(args, "point_indices", ""))
    up_list_all = up_list
    up_list = _select_uncertain_subset(
        up_list_all,
        max_points=int(getattr(args, "max_points", 30)),
        seed=int(getattr(args, "seed", 12345)),
        selection=str(getattr(args, "selection", "per_record_mixed")),
        explicit_indices=explicit_idx,
        rank_by=str(getattr(args, "rank_by", "margin")),
    )

    logger.info("Uncertain points: total=%d, selected_for_deep=%d", len(up_list_all), len(up_list))

    # Persist selected points index for reproducibility
    try:
        rows = []
        for k, u in enumerate(up_list):
            rows.append({
                "pp_id": k,
                "index": u.get("index"),
                "record": u.get("record"),
                "window_in_record": u.get("window_in_record"),
                "true_label": u.get("true_label"),
                "pred": u.get("pred"),
                "pmax": u.get("pmax"),
                "margin": u.get("margin"),
                "rank_metric": u.get("rank_metric"),
                "rank_uncertainty": u.get("rank_uncertainty"),
                "in_review_budget": u.get("in_review_budget"),
                "is_error": (int(u.get("pred", -999)) != int(u.get("true_label", -999))) if ("pred" in u and "true_label" in u) else None,
                "p1_up": u.get("p1"),
                "p2_up": u.get("p2"),
                "p1_raw": u.get("p1_raw"),
                "p2_raw": u.get("p2_raw"),
                "pmax_raw": u.get("pmax_raw"),
                "margin_raw": u.get("margin_raw"),
                "logit_diff_raw": u.get("logit_diff_raw"),

            })
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "selected_uncertain_points.csv"), index=False)
    except Exception as e:
        logger.warning("Failed to write selected_uncertain_points.csv: %s", e)

    print(f"[INFO] Deep analysis will run on {len(up_list)} anchors (out of {len(up_list_all)}).")


    # Training data for background (LIME/SHAP) + optional global sweeps.
    need_bg = (not getattr(args, "skip_lime", False)) or (not getattr(args, "skip_shap", False))
    need_m  = not getattr(args, "skip_multiplicity", False)

    X_full = None
    y_full = None
    X_train = None

    if need_bg or need_m:
        X_full, y_full = load_mitbih_data(
            args.data_folder,
            record_names,
            WINDOW_SIZE,
            PRE_SAMPLES,
            FS
        )

        if need_bg:
            bg_size = min(512, len(X_full))
            idx_bg = np.random.choice(len(X_full), size=bg_size, replace=False)
            X_train_full = X_full[idx_bg]
            X_train_c2 = compress_to_C2(X_train_full)  # shape (B, 4)
            X_train = scaler_full.transform(X_train_c2) if scaler_full is not None else X_train_c2
        else:
            logger.info("LIME/SHAP skipped -> background not built.")
    else:
        logger.info("Skipped loading MIT-BIH dataset (LIME/SHAP + multiplicity sweep disabled).")


    # Symbols for expansions.
    x_sym, y_sym = sympy.symbols('x y')

    # ----------------------------------------------------------------------
    # 3) Sensitivity (tau, delta) summary (prefer extended grid if available)
    # ----------------------------------------------------------------------
    sens_grid_ext_path  = os.path.join(IN_DIR, "sens_grid_ext.csv")
    sens_grid_legacy    = os.path.join(IN_DIR, "sens_grid.csv")
    sens_full_path      = os.path.join(IN_DIR, "sens_full.csv")
    sens_full_multi_path = os.path.join(IN_DIR, "sens_full_multi.csv")

    df_sens = None
    if os.path.isfile(sens_grid_ext_path):
        df_sens = pd.read_csv(sens_grid_ext_path)
        df_sens.to_csv(os.path.join(OUT_DIR, "sensitivity_detailed.csv"), index=False)
        logger.info("Loaded EXT sensitivity grid: %s (rows=%d)", sens_grid_ext_path, len(df_sens))
    elif os.path.isfile(sens_grid_legacy):
        df_sens = pd.read_csv(sens_grid_legacy)
        # legacy file has fewer columns; just copy it as detailed for backward compat
        df_sens.to_csv(os.path.join(OUT_DIR, "sensitivity_detailed.csv"), index=False)
        logger.info("Loaded LEGACY sensitivity grid: %s (rows=%d)", sens_grid_legacy, len(df_sens))
    else:
        logger.warning("No sensitivity grid found (sens_grid_ext.csv nor sens_grid.csv).")

    df_multi = None
    if os.path.isfile(sens_full_multi_path):
        df_multi = pd.read_csv(sens_full_multi_path)
        df_multi.to_csv(os.path.join(OUT_DIR, "sensitivity_multi.csv"), index=False)
        logger.info("Loaded multi-budget sensitivity summary: %s (rows=%d)", sens_full_multi_path, len(df_multi))

    chosen = None
    if os.path.isfile(sens_full_path):
        try:
            tmp = pd.read_csv(sens_full_path)
            if len(tmp):
                chosen = tmp.iloc[0].to_dict()
                logger.info("Loaded chosen (tau*,delta*) from sens_full.csv: tau=%.6f delta=%.6f",
                            float(chosen.get("tau", float("nan"))), float(chosen.get("delta", float("nan"))))
        except Exception as e:
            logger.warning("Failed to parse sens_full.csv: %s", e)

    # Write one coherent summary (no append/overwrite chaos)
    sum_path = os.path.join(OUT_DIR, "sensitivity_summary.txt")
    with open(sum_path, "w") as f:
        f.write("=== Sensitivity summary (post_processing_real) ===\n")
        f.write("NOTE: sens_grid*.csv is typically computed on the FULL-model validation split "
                "(used to choose tau*, delta*). Do NOT report it as test performance. "
                "Use cv_test_triage_curve_summary.csv / full_test_* for test-side reporting.\n\n")
        
        f.write(f"Anchors (uncertain points): {len(up_list)}\n")
        if df_sens is None or df_sens.empty:
            f.write("Grid: N/A\n")
        else:
            f.write(f"Grid rows: {len(df_sens)}\n")
            for col in ["tau", "delta"]:
                if col in df_sens.columns:
                    f.write(f"{col} range: [{df_sens[col].min():.6f}, {df_sens[col].max():.6f}]\n")
            # Prefer exact columns if present
            if "abstain_exact" in df_sens.columns:
                f.write(f"abstain_exact median: {df_sens['abstain_exact'].median():.6f}\n")
            if "risk_accept_exact" in df_sens.columns:
                f.write(f"risk_accept_exact median: {df_sens['risk_accept_exact'].median():.6f}\n")
            if "capture_exact" in df_sens.columns:
                f.write(f"capture_exact median: {df_sens['capture_exact'].median():.6f}\n")

            # Simple “best under budget” views if the columns exist
            if "abstain_exact" in df_sens.columns and "risk_accept_exact" in df_sens.columns:
                for b in [0.01, 0.02, 0.05, 0.10, 0.20]:
                    sub = df_sens[df_sens["abstain_exact"] <= b]
                    if len(sub):
                        best = sub.loc[sub["risk_accept_exact"].idxmin()]
                        f.write(f"\nBest risk under abstain<= {b:.2f}:\n")
                        f.write(f"  tau={best['tau']:.6f}, delta={best['delta']:.6f}, "
                                f"risk={best['risk_accept_exact']:.6f}, "
                                f"coverage={best.get('coverage_exact', float('nan')):.6f}, "
                                f"capture={best.get('capture_exact', float('nan')):.6f}\n")

        if chosen is not None:
            f.write("\nChosen (tau*,delta*) from sens_full.csv:\n")
            f.write(f"  tau*={chosen.get('tau')}, delta*={chosen.get('delta')}\n")
            for k in ["abstain", "capture", "precision", "risk_accept", "kink_score"]:
                if k in chosen:
                    f.write(f"  {k}={chosen.get(k)}\n")

        if df_multi is not None and len(df_multi):
            f.write("\nMulti-budget table (sens_full_multi.csv):\n")
            f.write(df_multi.to_string(index=False))
            f.write("\n")

    logger.info("Saved sensitivity summary -> %s", sum_path)


    # ----------------------------------------------------------------------
    # 4) Comparative calibration table with 95% CI (+ Wilcoxon, win-rate)
    # ----------------------------------------------------------------------
    cv_multi = os.path.join(IN_DIR, 'cv_metrics_per_fold_multi.csv')
    if os.path.isfile(cv_multi):
        by_method = {}
        with open(cv_multi, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                m = r.get("method", "UNKNOWN")
                by_method.setdefault(m, {"ECE": [], "NLL": [], "Brier": [], "Acc": [], "AUC": []})
                for k in by_method[m].keys():
                    try:
                        by_method[m][k].append(float(r.get(k, "nan")))
                    except Exception:
                        pass
        comp_path = os.path.join(OUT_DIR, "comparative_table.csv")
        with open(comp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Method", "Metric", "Mean", "CI95"])
            for m in sorted(by_method.keys()):
                for metric in ["ECE", "NLL", "Brier", "Acc", "AUC"]:
                    mean, ci = mean_ci95(by_method[m][metric])
                    w.writerow([m, metric, f"{mean:.6f}", f"{ci:.6f}"])
        print(f"[INFO] Comparative table saved to {comp_path}")

        # Relative ECE reduction vs NONE for the calibration used in this run (if present).
        base_ece_list = by_method.get("NONE", {}).get("ECE", [])
        used_key = str(calib_used).upper()
        used_ece_list = by_method.get(used_key, {}).get("ECE", [])

        if base_ece_list and used_ece_list:
            base_ece_mean, _ = mean_ci95(base_ece_list)
            used_ece_mean, _ = mean_ci95(used_ece_list)
            rel_drop = (base_ece_mean - used_ece_mean) / max(base_ece_mean, 1e-12)

            with open(os.path.join(OUT_DIR, "calibration_rel_drop_report.txt"), "w") as f:
                f.write(f"Method used (run): {used_key}\n")
                f.write(f"ECE(base=NONE)      = {base_ece_mean:.6f}\n")
                f.write(f"ECE({used_key})     = {used_ece_mean:.6f}\n")
                f.write(f"Relative drop (ECE) = {100*rel_drop:.2f}%\n")


        # --- Statistical comparisons & win-rate ---
        try:
            import itertools
            from scipy.stats import wilcoxon
            dfm = pd.read_csv(cv_multi)
            methods_present = sorted(dfm["method"].unique())
            with open(os.path.join(OUT_DIR, "calibration_stats_tests.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["Metric","MethodA","MethodB","p_value"])
                for metric in ["ECE","NLL","Brier","Acc","AUC"]:
                    for a,b in itertools.combinations(methods_present, 2):
                        sa = dfm[dfm["method"]==a][metric].astype(float).values
                        sb = dfm[dfm["method"]==b][metric].astype(float).values
                        n = min(len(sa), len(sb))
                        if n >= 5:
                            # For metrics with "lower is better" run one-sided 'less', else 'greater'.
                            alt = "less" if metric in ["ECE","NLL","Brier"] else "greater"
                            diff = sa[:n] - sb[:n]
                            if np.allclose(diff, 0.0):
                                p = 1.0  # identical distributions -> no difference
                            else:
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    p = wilcoxon(sa[:n], sb[:n], alternative=alt).pvalue
                            w.writerow([metric, a, b, f"{p:.3e}"])
            base = dfm[dfm["method"]=="NONE"][["fold","ECE"]].rename(columns={"ECE":"ECE_NONE"})
            rows_wr = []
            for m in methods_present:
                if m == "NONE": 
                    continue
                sub = dfm[dfm["method"]==m][["fold","ECE"]].rename(columns={"ECE":f"ECE_{m}"})
                j = base.merge(sub, on="fold", how="inner")
                wins = int((j[f"ECE_{m}"] < j["ECE_NONE"]).sum())
                rows_wr.append([m, wins, int(j.shape[0])])
            with open(os.path.join(OUT_DIR, "calibration_winrate_vs_none.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["Method","Wins_vs_NONE","N_folds"])
                w.writerows(rows_wr)
        except Exception as e:
            logger.warning("Statistical comparison failed: %s", e)

    else:
        print("[WARN] cv_metrics_per_fold_multi.csv not found; skipping comparative CI table.")


    # ----------------------------------------------------------------------
    # 4b) Paper bundle (optional): copy key figures + write a single markdown summary
    # ----------------------------------------------------------------------
    if bool(getattr(args, "copy_figures", False) or getattr(args, "paper", False)):
        # Copy key up_real figures into OUT_DIR so the post_processing output is self-contained
        figs = [
            # Core plots
            "calibration_curve_PLATT.png",
            "calibration_curve_RAW.png",
            "cv_risk_coverage_mean_ci.png",
            "cv_capture_abstain_mean_ci.png",
            "full_test_risk_coverage.png",
            "full_test_capture_abstain.png",

            # Convenience plots that we may generate below if missing:
            "cv_risk_abstain_mean_ci.png",
            "full_test_risk_abstain.png",

            # Sensitivity / selection diagnostics
            "sens_full_abstain.png",
            "sens_full_capture.png",
            "uncertainty_hist.png",
            "complex_PCA_scatter.png",

            # Other run visuals (optional)
            "budget_curve.png",
            "triage_curve.png",
            "overall_history.png",
            "fold_full_history.png",
        ]

        copied = 0
        for fn in figs:
            copied += int(copy_if_exists(os.path.join(IN_DIR, fn), OUT_DIR))
        logger.info("Copied %d figure(s) from IN_DIR to OUT_DIR.", copied)

        # Also copy the key CSV summaries
        csvs = [
            # CV summaries
            "cv_metrics_summary.csv",
            "cv_metrics_summary_multi.csv",
            "cv_metrics_summary_perf.csv",
            "cv_metrics_per_fold_multi.csv",
            "cv_metrics_per_fold.csv",
            "cv_metrics_per_fold_extended.csv",
            "cv_test_triage_curve_summary.csv",
            "cv_test_triage_curve_per_fold.csv",
            "cv_selective_budget_count_per_fold.csv",

            # Full test summaries
            "full_test_metrics.txt",
            "full_test_selective_metrics.csv",
            "full_test_triage_curve.csv",
            "full_test_calibration_metrics.csv",

            # Sensitivity grids (validation-side selection)
            "sens_full.csv",
            "sens_full_multi.csv",
            "sens_grid_ext.csv",
            "sens_grid.csv",

            # Uncertain points
            "uncertain_full.csv",
            "uncertain_full_ext.csv",
        ]

        copied_csv = 0
        for fn in csvs:
            copied_csv += int(copy_if_exists(os.path.join(IN_DIR, fn), OUT_DIR))
        logger.info("Copied %d CSV summary file(s) from IN_DIR to OUT_DIR.", copied_csv)


        # Generate missing convenience plots (up_real may not emit these).
        try:
            # CV: risk vs abstain (mean ± CI)
            cv_dst = os.path.join(OUT_DIR, "cv_risk_abstain_mean_ci.png")
            if not os.path.isfile(cv_dst):
                df_src = safe_read_csv(os.path.join(OUT_DIR, "cv_test_triage_curve_summary.csv"))
                if df_src is None:
                    df_src = safe_read_csv(os.path.join(IN_DIR, "cv_test_triage_curve_summary.csv"))
                if df_src is not None:
                    ok = plot_xy_with_ci(
                        df_src,
                        x_col="abstain_mean",
                        y_col="risk_mean",
                        yerr_col="risk_CI95",
                        title="CV risk vs abstain (mean ± 95% CI)",
                        xlabel="abstain rate",
                        ylabel="risk on accepted",
                        out_path=cv_dst,
                    )
                    if ok:
                        logger.info("Generated %s", cv_dst)

            # Full test: risk vs abstain
            full_dst = os.path.join(OUT_DIR, "full_test_risk_abstain.png")
            if not os.path.isfile(full_dst):
                df_src = safe_read_csv(os.path.join(OUT_DIR, "full_test_triage_curve.csv"))
                if df_src is None:
                    df_src = safe_read_csv(os.path.join(IN_DIR, "full_test_triage_curve.csv"))
                if df_src is not None:
                    ok = plot_xy_with_ci(
                        df_src,
                        x_col="abstain_test",
                        y_col="risk_accept_test",
                        yerr_col="",
                        title="Full test risk vs abstain",
                        xlabel="abstain rate (test)",
                        ylabel="risk on accepted (test)",
                        out_path=full_dst,
                    )
                    if ok:
                        logger.info("Generated %s", full_dst)

        except Exception as e:
            logger.warning("Failed to generate convenience plots: %s", e)



    if bool(getattr(args, "paper", False)):
        try:
            write_paper_summary_md(
                out_dir=OUT_DIR,
                in_dir=IN_DIR,
                run_id=run_id,
                calib_used=calib_used,
                run_meta=_run_meta,
                run_args=_run_args,
            )
            logger.info("Wrote paper_summary.md")
        except Exception as e:
            logger.warning("Failed to write paper_summary.md: %s", e)


    # ----------------------------------------------------------------------
    # 5) Process each uncertain point
    # ----------------------------------------------------------------------
    # Collectors for aggregate reports.
    kink_rows = []
    res_rows = []
    fit_rows = []
    dom_rows = []

    for i, up in enumerate(up_list):
        # Local-analysis knobs (paper reproducibility)
        DELTA_LOCAL = float(getattr(args, "local_delta", 0.05))
        DEG_LOCAL   = int(getattr(args, "local_degree", 4))
        NS_LOCAL    = int(getattr(args, "local_samples", 600))
        NS_QUAL     = int(getattr(args, "quality_samples", 200))
        NS_KINK     = int(getattr(args, "kink_samples", 1000))

        ROBUST_STEPS  = int(getattr(args, "robust_steps", 60))

        RADIUS_ATTACK = float(getattr(args, "attack_radius", max(float(DELTA_LOCAL), 0.05)))
        if (not np.isfinite(RADIUS_ATTACK)) or (RADIUS_ATTACK <= 0.0):
            RADIUS_ATTACK = max(float(DELTA_LOCAL), 0.05)
        if RADIUS_ATTACK < float(DELTA_LOCAL):
            logger.warning(
                "attack_radius (%.4f) < local_delta (%.4f); using local_delta.",
                RADIUS_ATTACK,
                float(DELTA_LOCAL),
            )
            RADIUS_ATTACK = float(DELTA_LOCAL)

        NUM_RANDOM_DIRS  = int(getattr(args, "robust_num_random", 20))
        DIR_PROBE_RADIUS = float(getattr(args, "robust_dir_radius", 0.01))


        print(f"\n=== POINT # {i} ===")
        print("[DATA]", up)

        # Base anchor in R^4 = (Re(z1), Re(z2), Im(z1), Im(z2))
        xstar = _anchor_to_xstar(up)

        # Non-holomorphicity heuristic.
        try:
            ks = kink_score(model, xstar, radius=0.01, samples=24, device=device)
            logger.info("[KINK] angular-std=%.3f rad (%s)", ks,
                        "suspected non-holomorphic sector" if ks > 0.5 else "smooth")
        except Exception as e:
            logger.warning("kink_score failed: %s", e)

        # --- Fraction of 'kink' (modReLU) in the neighborhood ---
        try:
            kdiag = estimate_nonholomorphic_fraction(
                model, xstar, delta=DELTA_LOCAL, n_samples=NS_KINK, kink_eps=1e-6, device=device
            )
        except Exception as e:
            logger.warning("estimate_nonholomorphic_fraction failed for point %d: %s", i, e)
            kdiag = {
                "frac_kink": float("nan"),
                "frac_active": float("nan"),
                "frac_inactive": float("nan"),
                "n_samples": int(NS_KINK),
                "error": str(e),
            }
        
        # --- optional sweep over kink_eps (diagnostic; OFF by default) ---
        DO_KINK_SWEEP = bool(int(os.environ.get("PP_KINK_SWEEP", "0")))
        if DO_KINK_SWEEP:
            kink_sweep_eps = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
            sweep_n = int(os.environ.get("PP_KINK_SWEEP_NSAMPLES", "2000"))
            rows_k = []
            for eps in kink_sweep_eps:
                ksweep = estimate_nonholomorphic_fraction(
                    model,
                    xstar,
                    delta=DELTA_LOCAL,
                    n_samples=sweep_n,
                    kink_eps=float(eps),
                    device=device,
                )
                rows_k.append(
                    {
                        "idx": i,
                        "kink_eps": float(eps),
                        "frac_kink": float(ksweep.get("frac_kink", np.nan)),
                        "mean_kink_score": float(ksweep.get("mean_kink_score", np.nan)),
                    }
                )

            sweep_path = os.path.join(OUT_DIR, f"kink_sweep_point{i}.csv")
            pd.DataFrame(rows_k).to_csv(sweep_path, index=False)
            logger.info("Saved kink sweep -> %s", sweep_path)
        print(f"[KINK] frac_kink={float(kdiag.get('frac_kink', float('nan'))):.3f} | "
              f"active={float(kdiag.get('frac_active', float('nan'))):.3f} | "
              f"inactive={float(kdiag.get('frac_inactive', float('nan'))):.3f} | "
              f"n={int(kdiag.get('n_samples', 0) or 0)}")


        # ------------------------------------------------------------------
        # (A) Local polynomial approximation (robust, with auto-refit for fidelity)
        # ------------------------------------------------------------------
        exclude_kink_eps = float(os.environ.get("PP_EXCLUDE_KINK_EPS", "1e-6"))

        min_corr = float(getattr(args, "min_corr", 0.50))
        min_sign = float(getattr(args, "min_sign_agree", 0.75))
        max_refit = int(getattr(args, "max_refit_attempts", 2))
        refit_mult = float(getattr(args, "refit_samples_mult", 2.0))
        refit_kink_mult = float(getattr(args, "refit_kink_eps_mult", 5.0))

        degree_used = int(DEG_LOCAL)
        n_samples_used = int(NS_LOCAL)
        kink_eps_used = float(exclude_kink_eps)

        refit_attempts = 0
        fit_ok = False
        metrics = {}
        fit_diag = {}
        F_expr_zero = None

        while True:
            try:
                F_expr_zero, fit_diag = local_poly_approx_complex(
                    model, xstar,
                    delta=DELTA_LOCAL, degree=degree_used, n_samples=n_samples_used,
                    device=device,
                    calibrator=cal,            # <<< NEW
                    remove_linear=True,
                    exclude_kink_eps=kink_eps_used, weight_by_distance=True,
                    return_diag=True
                )
            except Exception as e:
                F_expr_zero = None
                fit_diag = {"fit_error": str(e)}
                metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                           "corr_pearson": float("nan"), "sign_agreement": float("nan"),
                           "quality_error": str(e)}
                fit_ok = False
                logger.warning("[FIT] local_poly_approx_complex failed for point %d: %s", i, e)
                break

            # Defensive Puiseux preconditions (F(0,0)=0 + no linear leakage)
            try:
                F_expr_zero, pdiag = enforce_puiseux_ready_poly(F_expr_zero, x_sym, y_sym, tol=1e-10)
                if isinstance(fit_diag, dict):
                    fit_diag.update(pdiag)
            except Exception as e:
                if isinstance(fit_diag, dict):
                    fit_diag["puiseux_enforce_error"] = str(e)

            print("[FIT] kept={}/{} ({:.1f}%), cond={:.2e}, rank={}/{} | deg={} | n_samples={} | kink_eps={:.1e}".format(
                fit_diag.get('n_kept', -1), fit_diag.get('n_total', -1), 100 * float(fit_diag.get('kept_ratio', 0.0)),
                float(fit_diag.get('cond', float('nan'))),
                int(fit_diag.get('rank', -1)), int(fit_diag.get('n_monomials', -1)),
                degree_used, n_samples_used, kink_eps_used
            ))

            try:
                metrics = evaluate_poly_approx_quality(
                    model=model,
                    poly_expr=F_expr_zero,
                    xstar=xstar,
                    delta=DELTA_LOCAL,
                    n_samples=NS_QUAL,
                    device=device,
                    calibrator=cal,            # <<< NEW
                    fit_diag=fit_diag,         # <<< NEW
                    include_linear_and_const=True  # <<< NEW
                )

            except Exception as e:
                logger.warning("[FIT] evaluate_poly_approx_quality failed for point %d: %s", i, e)
                metrics = {"RMSE": float("nan"), "MAE": float("nan"),
                           "corr_pearson": float("nan"), "sign_agreement": float("nan"),
                           "quality_error": str(e)}

            print("[Approx Quality] RMSE={:.3f}, MAE={:.3f}, corr={:.3f}, sign_agree={:.3f}".format(
                float(metrics.get('RMSE', float('nan'))),
                float(metrics.get('MAE', float('nan'))),
                float(metrics.get('corr_pearson', float('nan'))),
                float(metrics.get('sign_agreement', float('nan')))
            ))

            fit_ok = (
                np.isfinite(float(metrics.get("corr_pearson", float("nan")))) and
                np.isfinite(float(metrics.get("sign_agreement", float("nan")))) and
                float(metrics.get("corr_pearson", -1.0)) >= min_corr and
                float(metrics.get("sign_agreement", -1.0)) >= min_sign
            )

            if fit_ok:
                break

            if refit_attempts >= max_refit:
                logger.warning(
                    "[FIT] Point %d: surrogate quality below thresholds after %d attempt(s): corr=%.3f (min=%.3f), sign=%.3f (min=%.3f). Proceeding but marking fit_ok=FALSE.",
                    i, refit_attempts,
                    float(metrics.get("corr_pearson", float("nan"))), min_corr,
                    float(metrics.get("sign_agreement", float("nan"))), min_sign,
                )
                break

            # Refit with more samples + stricter kink exclusion.
            # Degree selection is already handled inside local_poly_approx_complex (it falls back if unstable),
            # so do not force degree down here.
            refit_attempts += 1
            n_samples_used = int(max(n_samples_used + 1, round(n_samples_used * refit_mult)))
            kink_eps_used = float(kink_eps_used * refit_kink_mult)


            logger.warning(
                "[FIT] Refitting point %d (attempt %d/%d): n_samples=%d, degree=%d, exclude_kink_eps=%.1e",
                i, refit_attempts, max_refit, n_samples_used, degree_used, kink_eps_used
            )

        # Persist diagnostics for this point (paper reproducibility)
        try:
            diag_out = {
                "point": int(i),
                "index": up.get("index"),
                "record": up.get("record"),
                "window_in_record": up.get("window_in_record"),
                "delta_local": float(DELTA_LOCAL),
                "degree_used": int(degree_used),
                "n_samples_used": int(n_samples_used),
                "exclude_kink_eps_used": float(kink_eps_used),
                "refit_attempts": int(refit_attempts),
                "fit_ok": bool(fit_ok),
                "fit_diag": (fit_diag or {}),
                "quality": (metrics or {}),
            }
            with open(os.path.join(OUT_DIR, f"local_surrogate_diag_point{i}.json"), "w") as fdiag:
                json.dump(diag_out, fdiag, indent=2)
        except Exception as e:
            logger.warning("Failed to write local_surrogate_diag_point%d.json: %s", i, e)

        # Ensure downstream summaries have stable keys
        fit_diag["degree_used"] = int(degree_used)
        fit_diag["n_samples_used"] = int(n_samples_used)
        fit_diag["exclude_kink_eps_used"] = float(kink_eps_used)
        fit_diag["refit_attempts"] = int(refit_attempts)
        fit_diag["fit_ok"] = bool(fit_ok)


        # ------------------------------------------------------------------
        # (B) Puiseux expansions + interpretation (ONLY if surrogate is trustworthy)
        # ------------------------------------------------------------------
        if (not fit_ok) or (F_expr_zero is None):
            expansions_np = []
            interpret_results = []
            print("\n[PUISEUX] skipped (fit_ok=FALSE: local surrogate below thresholds).")
            logger.warning("[PUISEUX] Point %d skipped due to fit_ok=FALSE.", i)
        else:
            puiseux_error = None
            try:
                expansions_np = puiseux_uncertain_point(F_expr_zero, prec=4, base_point=xstar)
                interpret_results = interpret_puiseux_expansions(expansions_np, x_sym, y_sym)
            except Exception as e:
                puiseux_error = str(e)
                expansions_np = []
                interpret_results = []
                logger.warning("[PUISEUX] Point %d failed: %s", i, e)

            print("\n[PUISEUX EXPANSIONS & INTERPRETATION]")
            for idx_e, ir in enumerate(interpret_results):
                print(f"  Expansion {idx_e}:")
                print("    ", ir["puiseux_expr"])
                print("    =>", ir["comment"])


        # ------------------------------------------------------------------
        # (C) Robustness: adversarial directions (CALIBRATED decision system) + trace export
        # ------------------------------------------------------------------
        results_table = []
        traces_rows: List[Dict[str, Any]] = []
        robustness_skip_reason = ""

        if (not fit_ok) or (F_expr_zero is None):
            robustness_skip_reason = "fit_ok=FALSE"
            logger.warning("[ROBUSTNESS] skipped for point %d (fit_ok=FALSE).", i)
        else:
            # Reconstruct FULL score surrogate for direction search:
            # score_hat(x,y) = score0 + cx*x + cy*y + F_nl(x,y)
            polynom_full = F_expr_zero
            try:
                score0 = float(fit_diag.get("score0", 0.0))
                cx_re = float(fit_diag.get("lin_cx_re", 0.0))
                cx_im = float(fit_diag.get("lin_cx_im", 0.0))
                cy_re = float(fit_diag.get("lin_cy_re", 0.0))
                cy_im = float(fit_diag.get("lin_cy_im", 0.0))

                cx_sym = sympy.Float(cx_re) + sympy.I * sympy.Float(cx_im)
                cy_sym = sympy.Float(cy_re) + sympy.I * sympy.Float(cy_im)

                polynom_full = sympy.expand(sympy.Float(score0) + cx_sym * x_sym + cy_sym * y_sym + F_expr_zero)
            except Exception as e:
                logger.warning("[ROBUSTNESS] Failed to reconstruct full polynomial (point %d): %s", i, e)
                polynom_full = F_expr_zero  # fallback

            try:
                best_dirs_info = find_adversarial_directions(
                    polynom_full,
                    x_sym,
                    y_sym,
                    num_random=NUM_RANDOM_DIRS,
                    radius=DIR_PROBE_RADIUS,
                )
            except Exception as e:
                best_dirs_info = []
                robustness_skip_reason = f"find_adversarial_directions_failed: {e}"
                logger.warning("[ROBUSTNESS] find_adversarial_directions failed for point %d: %s", i, e)

            # Baseline prediction at anchor (calibrated)
            _, _, p0 = predict_logits_and_proba(model, xstar.reshape(1, -1), device, cal=cal)
            y0 = int(np.argmax(p0[0]))

            for d_id, (dir_radians, phase_val) in enumerate(best_dirs_info):
                rows_dir, changed_class, changed_radius = robustness_trace_along_dir_calibrated(
                    model=model,
                    xstar=xstar,
                    dir_radians=dir_radians,
                    radius=RADIUS_ATTACK,
                    steps=ROBUST_STEPS,
                    device=device,
                    cal=cal,
                    direction_id=int(d_id),
                    phase=float(phase_val),
                )
                traces_rows.extend(rows_dir)

                results_table.append({
                    "direction_id": int(d_id),
                    "direction_radians": dir_radians,
                    "phase": float(phase_val),
                    "baseline_pred": int(y0),
                    "changed_class": bool(changed_class),
                    "changed_radius": (float(changed_radius) if changed_radius is not None else None),
                    "skipped": False,
                    "reason": "",
                })

        # Export traces + summary — ALWAYS
        try:
            df_tr = pd.DataFrame(traces_rows)
            if df_tr.empty:
                df_tr = pd.DataFrame(columns=[
                    "direction_id","phase","theta_x","theta_y","step","radius",
                    "p0_cal","p1_cal","pmax_cal","margin_cal","pred_cal","changed"
                ])
            df_tr.to_csv(os.path.join(OUT_DIR, f"robustness_traces_point{i}.csv"), index=False)

            if not results_table:
                results_table = [{
                    "direction_id": -1,
                    "direction_radians": (float("nan"), float("nan")),
                    "phase": float("nan"),
                    "baseline_pred": None,
                    "changed_class": False,
                    "changed_radius": None,
                    "skipped": True,
                    "reason": robustness_skip_reason,
                }]

            pd.DataFrame(results_table).to_csv(os.path.join(OUT_DIR, f"robustness_summary_point{i}.csv"), index=False)

        except Exception as e:
            logger.warning("Failed to write robustness CSVs for point %d: %s", i, e)

        # Plot — ALWAYS attempt
        try:
            robustness_plot_path = os.path.join(OUT_DIR, f"robustness_curves_point{i}.png")
            plot_robustness_traces_calibrated(traces_rows, save_path=robustness_plot_path)
        except Exception as e:
            logger.warning("Robustness plot failed for point %d: %s", i, e)


        
        # ------------------------------------------------------------------
        # (D) LIME & SHAP with temperature (apply scaler consistently)
        # ------------------------------------------------------------------
        # NOTE: uncertain_full.csv stores X already in *model input space* (scaled by scaler_full in up_real.py).
        # Therefore: DO NOT transform xstar again here.
        xstar_model = xstar.reshape(1, -1).astype(np.float32)

        def predict_proba_np(X_np: np.ndarray) -> np.ndarray:
            _, _, pcal = predict_logits_and_proba(model, X_np, device, cal=cal)
            return pcal

        # LIME (calibrated predictor)
        lime_list = []
        if getattr(args, "skip_lime", False):
            logger.info("LIME skipped by --skip_lime (point %d).", i)
        elif X_train is None:
            logger.warning("LIME requested but X_train background is unavailable; skipping (point %d).", i)
        elif LimeTabularExplainer is None:
            logger.warning("LIME library not available; skipping (point %d).", i)
        else:
            try:
                explainer = LimeTabularExplainer(
                    X_train,
                    feature_names=["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"],
                    class_names=["class0", "class1"],
                    mode="classification",
                    discretize_continuous=False,
                )
                lime_exp = explainer.explain_instance(
                    xstar_model[0],
                    predict_proba_np,
                    num_features=4,
                    num_samples=4000,
                )
                lime_list = lime_exp.as_list()
                print("\n[LIME Explanation] (calibrated)")
                for feat, val in lime_list:
                    print(f" {feat}: {val:.3f}")
            except Exception as e:
                logger.warning("LIME failed for point %d: %s", i, e)
                lime_list = []

        # SHAP (calibrated predictor)
        shap_vals = None
        shap_class0 = None
        shap_class1 = None

        if getattr(args, "skip_shap", False):
            logger.info("SHAP skipped by --skip_shap (point %d).", i)
        elif X_train is None:
            logger.warning("SHAP requested but X_train background is unavailable; skipping (point %d).", i)
        elif shap is None:
            logger.warning("SHAP library not available; skipping (point %d).", i)
        else:
            try:
                bg = X_train[:10]
                expl = shap.KernelExplainer(predict_proba_np, bg)
                shap_vals = expl.shap_values(xstar_model, nsamples=100)

                arr0, arr1 = _normalize_shap_output(shap_vals)
                arr0 = np.asarray(arr0)
                arr1 = None if (arr1 is None) else np.asarray(arr1)

                shap_class0 = arr0[0] if arr0.ndim == 2 else arr0.reshape(-1)
                if arr1 is not None:
                    shap_class1 = arr1[0] if arr1.ndim == 2 else arr1.reshape(-1)

                print("\n[SHAP Explanation] (calibrated) per feature:")
                for fid, feat_name in enumerate(["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]):
                    if shap_class1 is None:
                        print("  {} => shap: {:.3f}".format(feat_name, scalarize(shap_class0[fid])))
                    else:
                        print("  {} => shap0: {:.3f}, shap1: {:.3f}".format(
                            feat_name, scalarize(shap_class0[fid]), scalarize(shap_class1[fid])
                        ))
            except Exception as e:
                logger.warning("SHAP failed for point %d: %s", i, e)
                shap_class0 = None
                shap_class1 = None
                print("\n[SHAP] skipped/failed.")



        # ==== AXIS-BASELINE RAY SWEEPS (add below SHAP block) ====
        def _flip_radius_along_vector(model, x0, v, device, cal, r_max, steps=ROBUST_STEPS):
            v = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n < 1e-12:
                return None
            v = v / n

            def _pred_label(x_np: np.ndarray) -> int:
                x_t = torch.from_numpy(x_np.astype(np.float32)).to(device).unsqueeze(0)
                with torch.no_grad():
                    logits_raw = complex_modulus_to_logits(model(x_t))
                    logits_eff = cal.apply_to_logits(logits_raw) if cal is not None else logits_raw
                    p = torch.softmax(logits_eff, dim=1)
                    return int(p.argmax(dim=1).item())

            y0 = _pred_label(x0)

            # coarse scan
            rs = np.linspace(0.0, float(r_max), int(steps) + 1)[1:]
            y_first = None
            lo = 0.0
            hi = None
            for r in rs:
                y = _pred_label(x0 + r * v)
                if y != y0:
                    y_first = y
                    hi = float(r)
                    lo = float(r - (rs[1]-rs[0]))
                    break
            if hi is None:
                return None

            # binary search refine
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                ymid = _pred_label(x0 + mid * v)
                if ymid != y0:
                    hi = mid
                else:
                    lo = mid
            return float(hi)


        def _idx_from_featname(name: str):
            # Map feature label to index in R^4 = [Re(z1), Re(z2), Im(z1), Im(z2)]
            name = name.replace(" ", "").lower()
            if "re(z1)" in name or "rex1" in name: return 0
            if "re(z2)" in name or "rex2" in name: return 1
            if "im(z1)" in name or "imx1" in name: return 2
            if "im(z2)" in name or "imx2" in name: return 3
            return None

        def _best_axis_from_lime(lime_list):
            # lime_list: [(feat_name, weight), ...] descending by |weight|
            for feat, w in sorted(lime_list, key=lambda kv: abs(kv[1]), reverse=True):
                j = _idx_from_featname(feat)
                if j is not None:
                    return j
            return None

        def _best_axis_from_shap(shap_vec):
            # shap_vec: ndarray shape (4,)
            import numpy as np
            j = int(np.nanargmax(np.abs(shap_vec)))
            return j

        # pick top axis for LIME/SHAP
        axis_lime = _best_axis_from_lime(lime_list)
        axis_shap = None
        if shap_class0 is not None:
            axis_shap = _best_axis_from_shap(shap_class1 if shap_class1 is not None else shap_class0)

        # gradient (saliency) direction in R^4
        # IMPORTANT: use the same decision system as up_real (calibrated logits)
        model.zero_grad()
        x_t = torch.from_numpy(xstar.astype(np.float32)).to(device).unsqueeze(0).requires_grad_(True)

        logits_raw = complex_modulus_to_logits(model(x_t))
        logits_eff = cal.apply_to_logits(logits_raw) if cal is not None else logits_raw

        # Use binary decision logit difference for boundary geometry
        d_eff = logits_eff[0, 1] - logits_eff[0, 0]
        d_eff.backward()


        g = x_t.grad.detach().cpu().numpy().reshape(-1)
        grad_dir = g if np.linalg.norm(g) > 1e-12 else None

        E = np.eye(4, dtype=np.float32)
        def _min_flip_two_sides(vec):
            r_max_eff = float(RADIUS_ATTACK) if "RADIUS_ATTACK" in locals() else max(float(DELTA_LOCAL), 0.05)
            r1 = _flip_radius_along_vector(model, xstar, vec, device, cal, r_max=r_max_eff, steps=ROBUST_STEPS)
            r2 = _flip_radius_along_vector(model, xstar, -vec, device, cal, r_max=r_max_eff, steps=60)

            vals = [r for r in [r1, r2] if r is not None]
            return (min(vals) if vals else None)

        flip_grad = _min_flip_two_sides(grad_dir) if grad_dir is not None else None
        flip_lime = _min_flip_two_sides(E[axis_lime]) if axis_lime is not None else None
        def _clamp_index(idx, n):
            if idx is None or n <= 0:
                return None
            idx = int(idx)
            if idx < 0:
                return 0
            if idx >= n:
                return n - 1
            return idx

        idx_shap = _clamp_index(axis_shap, len(E))
        flip_shap = _min_flip_two_sides(E[idx_shap]) if idx_shap is not None else None



        # 7b. Axis-baseline ray sweeps 
        axis_report_str = (
            f"AXIS-BASELINE RAY SWEEPS (r_max={float(RADIUS_ATTACK):.3f}, steps={int(ROBUST_STEPS)})\n"
            f"   flip_grad = {('N/A' if flip_grad is None else f'{float(flip_grad):.6f}')}\n"
            f"   lime_axis = {('N/A' if axis_lime is None else f'{axis_lime} ({_FEAT_NAMES_R4[axis_lime]})')} | "
            f"flip_lime = {('N/A' if flip_lime is None else f'{float(flip_lime):.6f}')}\n"
            f"   shap_axis = {('N/A' if idx_shap is None else f'{idx_shap} ({_FEAT_NAMES_R4[idx_shap]})')} | "
            f"flip_shap = {('N/A' if flip_shap is None else f'{float(flip_shap):.6f}')}\n\n"
        )


                
        # ------------------------------------------------------------------
        # (E) 2D local decision contour (fix pairs of dims)
        # ------------------------------------------------------------------
        if getattr(args, "skip_contours", False):
            logger.info("Contours skipped by --skip_contours (point %d).", i)
        else:
            try:
                save_dim_1_3 = os.path.join(OUT_DIR, f"contour_point{i}_fix_dim=[1,3].png")
                plot_local_contour_2d_calibrated(
                    model, xstar,
                    fix_dims=(1, 3), delta=DELTA_LOCAL, steps=50,
                    device=device, cal=cal, save_path=save_dim_1_3
                )

                save_dim_0_2 = os.path.join(OUT_DIR, f"contour_point{i}_fix_dim=[0,2].png")
                plot_local_contour_2d_calibrated(
                    model, xstar,
                    fix_dims=(0, 2), delta=DELTA_LOCAL, steps=50,
                    device=device, cal=cal, save_path=save_dim_0_2
                )

            except Exception as e:
                logger.warning("Contours failed for point %d: %s", i, e)


        # ------------------------------------------------------------------
        # (F) Resource benchmark (Puiseux vs gradient saliency)
        # ------------------------------------------------------------------

        # Defaults (so the script remains robust when benchmarks are skipped)
        times_pp = {
            "time_sampling": float("nan"),
            "time_lstsq": float("nan"),
            "time_factor": float("nan"),
            "time_simplify": float("nan"),
            "time_puiseux": float("nan"),
            "time_total": float("nan"),
        }
        sal = {
            "time_ms": float("nan"),
            "grad_norm": float("nan"),
            "cpu_rss_mb_delta": float("nan"),
            "gpu_peak_mb": float("nan"),
        }

        # (F) Resource benchmark (Puiseux vs gradient saliency)
        if getattr(args, "skip_benchmark", False):
            logger.info("Benchmark skipped by --skip_benchmark (point %d).", i)
        else:
            try:
                local_poly_func_bench = lambda **kw: local_poly_approx_complex(
                    calibrator=cal,
                    **kw
                )

                times_pp, _, _ = benchmark_local_poly_approx_and_puiseux(
                    model=model,
                    xstar=xstar,
                    local_poly_func=local_poly_func_bench,
                    puiseux_func=puiseux_expansions,
                    delta=DELTA_LOCAL,
                    degree=DEG_LOCAL,
                    n_samples=300,
                    device=device,
                    do_factor=True,
                    do_simplify=True,
                    puiseux_prec=4
                )

            except Exception as e:
                logger.warning("Puiseux benchmark failed for point %d: %s", i, e)

            try:
                if (cal is not None) and (cal.method or "NONE").upper() != "TEMPERATURE":
                    sal = time_gradient_saliency_calibrated(model, xstar, device, cal=cal, repeat=5)
                else:
                    sal = time_gradient_saliency(model, xstar, device, T=T, repeat=5)
            except Exception as e:
                logger.warning("Saliency benchmark failed for point %d: %s", i, e)

        # Always write a resource file (paper bundles like having it consistently)
        res_txt = os.path.join(OUT_DIR, f"resource_point{i}.txt")
        try:
            with open(res_txt, "w") as fz:
                fz.write("=== Resource benchmark (Puiseux vs. gradient saliency) ===\n")
                fz.write(
                    f"Puiseux: sample={times_pp.get('time_sampling', float('nan')):.2f}s, "
                    f"lsq={times_pp.get('time_lstsq', float('nan')):.2f}s, "
                    f"factor={times_pp.get('time_factor', float('nan')):.2f}s, "
                    f"simplify={times_pp.get('time_simplify', float('nan')):.2f}s, "
                    f"expansion={times_pp.get('time_puiseux', float('nan')):.2f}s, "
                    f"total={times_pp.get('time_total', float('nan')):.2f}s\n"
                )
                fz.write(
                    f"Saliency: time={sal.get('time_ms', float('nan')):.2f} ms, "
                    f"grad_norm={sal.get('grad_norm', float('nan')):.3e}, "
                    f"cpu_dRSS={sal.get('cpu_rss_mb_delta', float('nan')):.1f} MB, "
                    f"gpu_peak={sal.get('gpu_peak_mb', float('nan')):.1f} MB\n"
                )
            print(f"[INFO] Resource benchmark saved to {res_txt}")
        except Exception as e:
            logger.warning("Failed to write resource_point file for point %d: %s", i, e)


        # Append to aggregate reports.
        kink_rows.append([i, kdiag['frac_kink'], kdiag['frac_active'], kdiag['frac_inactive'], kdiag['n_samples']])

        res_rows.append([
            i,
            float(times_pp.get("time_total", float("nan"))),
            float(times_pp.get("time_sampling", float("nan"))),
            float(times_pp.get("time_lstsq", float("nan"))),
            float(times_pp.get("time_factor", float("nan"))),
            float(times_pp.get("time_simplify", float("nan"))),
            float(times_pp.get("time_puiseux", float("nan"))),
            float(times_pp.get("cpu_rss_mb", float("nan"))),
            float(times_pp.get("gpu_peak_mb", float("nan"))),
            float(sal.get("time_ms", float("nan"))),
            float(sal.get("grad_norm", float("nan"))),
            float(sal.get("cpu_rss_mb_delta", float("nan"))),
            float(sal.get("gpu_peak_mb", float("nan"))),
        ])

        fit_rows.append([
            i,
            bool(fit_diag.get("fit_ok", False)),
            int(fit_diag.get("refit_attempts", 0)),
            float(fit_diag.get("kept_ratio", float("nan"))),
            float(fit_diag.get("cond", float("nan"))),
            int(fit_diag.get("degree_used", DEG_LOCAL)),
            int(fit_diag.get("n_samples_used", NS_LOCAL)),
            float(fit_diag.get("exclude_kink_eps_used", float("nan"))),
            float(metrics.get("RMSE", float("nan"))),
            float(metrics.get("MAE", float("nan"))),
            float(metrics.get("corr_pearson", float("nan"))),
            float(metrics.get("sign_agreement", float("nan"))),
            float(fit_diag.get("resid_mean", float("nan"))),
            float(fit_diag.get("resid_std", float("nan"))),
            float(fit_diag.get("resid_skew", float("nan"))),
            float(fit_diag.get("resid_kurt", float("nan"))),
        ])


        
        # ------------------------------------------------------------------
        # (G) Save a comprehensive per-point report
        # ------------------------------------------------------------------
        out_txt_path = os.path.join(OUT_DIR, f'benchmark_point{i}.txt')
        with open(out_txt_path, "w") as f_out:
            f_out.write("=" * 80 + "\n")
            f_out.write(f"Local Analysis Report for Uncertain Point #{i}\n")
            f_out.write("=" * 80 + "\n\n")

            f_out.write("0. Kink diagnostics (modReLU neighborhood):\n")
            f_out.write(f"   frac_kink       : {float(kdiag.get('frac_kink', float('nan'))):.3f}\n")
            f_out.write(f"   frac_active     : {float(kdiag.get('frac_active', float('nan'))):.3f}\n")
            f_out.write(f"   frac_inactive   : {float(kdiag.get('frac_inactive', float('nan'))):.3f}\n")
            f_out.write(f"   samples_checked : {int(kdiag.get('n_samples', 0) or 0)}\n\n")

            f_out.write("1. Base Point (xstar):\n")
            f_out.write(f"   {xstar.tolist()}\n\n")

            # Meta (if uncertain_full_ext.csv was available)
            f_out.write("1b. Anchor metadata (from uncertain_full_ext.csv if present):\n")
            f_out.write(f"   index           : {up.get('index', 'N/A')}\n")
            f_out.write(f"   record          : {up.get('record', 'N/A')}\n")
            f_out.write(f"   window_in_record: {up.get('window_in_record', 'N/A')}\n")
            f_out.write(f"   true_label      : {up.get('true_label', 'N/A')}\n")
            f_out.write(f"   pred            : {up.get('pred', 'N/A')}\n")
            f_out.write(f"   pmax            : {up.get('pmax', 'N/A')}\n")
            f_out.write(f"   margin          : {up.get('margin', 'N/A')}\n")
            f_out.write(f"   rank_metric     : {up.get('rank_metric', 'N/A')}\n")
            f_out.write(f"   rank_uncertainty: {up.get('rank_uncertainty', 'N/A')}\n")
            f_out.write(f"   in_review_budget: {up.get('in_review_budget', 'N/A')}\n")

            try:
                is_err = int(up.get("pred", -999)) != int(up.get("true_label", -999))
                f_out.write(f"   is_error        : {is_err}\n\n")
            except Exception:
                f_out.write("   is_error        : N/A\n\n")


            f_out.write("2. Local polynomial fit (robust):\n")
            n_kept = int((fit_diag or {}).get("n_kept", 0) or 0)
            n_total = int((fit_diag or {}).get("n_total", 0) or 0)
            kept_ratio = float((fit_diag or {}).get("kept_ratio", float("nan")))
            kr_pct = (100.0 * kept_ratio) if np.isfinite(kept_ratio) else float("nan")
            condA = float((fit_diag or {}).get("cond", float("nan")))
            rankA = int((fit_diag or {}).get("rank", -1) or -1)
            n_mono = int((fit_diag or {}).get("n_monomials", -1) or -1)

            f_out.write(f"   kept / total    : {n_kept} / {n_total} ({kr_pct:.1f}%)\n")
            f_out.write(f"   cond(A)         : {condA:.3e}\n")
            f_out.write(f"   rank / monomials: {rankA} / {n_mono}\n\n")

            f_out.write("3. Approximation Quality Metrics:\n")
            rmse = float((metrics or {}).get("RMSE", float("nan")))
            mae = float((metrics or {}).get("MAE", float("nan")))
            corr = float((metrics or {}).get("corr_pearson", float("nan")))
            sag = float((metrics or {}).get("sign_agreement", float("nan")))
            f_out.write(f"   RMSE            : {rmse:.3f}\n")
            f_out.write(f"   MAE             : {mae:.3f}\n")
            f_out.write(f"   Pearson Corr    : {corr:.3f}\n")
            f_out.write(f"   Sign Agreement  : {sag:.3f}\n\n")

            f_out.write("4. Puiseux Expansions and Interpretation:\n")
            if not interpret_results:
                f_out.write("   Puiseux: skipped/empty.\n")
                try:
                    if 'puiseux_error' in locals() and puiseux_error:
                        f_out.write(f"   Puiseux error: {puiseux_error}\n")
                except Exception:
                    pass
                f_out.write("\n")
            else:
                for idx_e, ir in enumerate(interpret_results):
                    f_out.write(f"   >> Expansion {idx_e}:\n")
                    f_out.write(f"      Puiseux Expression: {ir['puiseux_expr']}\n")
                    f_out.write(f"      Interpretation    : {ir['comment']}\n")
                f_out.write("\n")


            
            # --- Dominant-ratio heuristic (r_dom) and observed flip radius ---
            # We estimate when quartic terms overtake quadratic curvature: 
            # r_dom ≈ sqrt(max|c2| / max|c4|), where c2 and c4 are coefficients of x^2 and x^4
            # across all Puiseux branches. We also extract the minimal observed flip radius r_flip
            # from the robustness table above.
            try:
                # 1) Parse Puiseux expressions into SymPy and collect coefficients
                exprs = []
                for ir in interpret_results:
                    expr_repr = ir.get("puiseux_expr")
                    if isinstance(expr_repr, sympy.Expr):
                        e = sympy.expand(expr_repr)
                    else:
                        # robust to string formatting
                        e = sympy.expand(sympy.sympify(str(expr_repr)))
                    exprs.append(e)

                max_abs_c2 = 0.0
                max_abs_c4 = 0.0
                for e in exprs:
                    c2 = e.coeff(x_sym, 2)
                    c4 = e.coeff(x_sym, 4)

                    # Convert possibly-complex SymPy numbers to Python complex and take magnitude
                    if c2 != 0:
                        max_abs_c2 = max(max_abs_c2, float(abs(complex(c2.evalf()))))
                    if c4 != 0:
                        max_abs_c4 = max(max_abs_c4, float(abs(complex(c4.evalf()))))

                if (max_abs_c2 > 0.0) and (max_abs_c4 > 0.0):
                    r_dom = float(np.sqrt(max_abs_c2 / max(max_abs_c4, 1e-12)))
                else:
                    r_dom = float("nan")
            except Exception as e:
                logger.warning("Failed to compute r_dom for point %d: %s", i, e)
                max_abs_c2 = float("nan")
                max_abs_c4 = float("nan")
                r_dom = float("nan")

            # 2) Observed minimal flip radius from robustness results_table
            try:
                r_flip_candidates = [row["changed_radius"] for row in results_table
                                     if row.get("changed_radius") is not None]
                flip_found = int(len(r_flip_candidates) > 0)
                r_flip = float(min(r_flip_candidates)) if flip_found else float("nan")
                r_flip_cens = float(r_flip) if flip_found else float(RADIUS_ATTACK)
            except Exception as e:
                logger.warning("Failed to compute r_flip for point %d: %s", i, e)
                flip_found = 0
                r_flip = float("nan")
                r_flip_cens = float(RADIUS_ATTACK)

            # 3) Persist both numbers to the TXT so external parsers can consume them
            f_out.write("   Dominant-ratio heuristic and flip comparison:\n")
            f_out.write(f"      max|c2| = {max_abs_c2:.6g}, max|c4| = {max_abs_c4:.6g}\n")
            f_out.write(f"      Predicted onset radius r_dom ≈ sqrt(|c2|/|c4|) = {r_dom:.6f}\n")
            f_out.write("      Observed min flip radius r_flip = "
                        f"{('N/A' if np.isnan(r_flip) else f'{r_flip:.6f}')}\n\n")

            # 4) Collect for a CSV summary across all anchors
            dom_rows.append([
                i,
                max_abs_c2,
                max_abs_c4,
                r_dom,
                flip_found,
                r_flip,          # r_flip_obs = observed min flip radius (NaN if no flip found <= RADIUS_ATTACK)
                r_flip_cens,     # r_flip_cens = censored at RADIUS_ATTACK when no flip found
                flip_grad if flip_grad is not None else np.nan,
                flip_lime if flip_lime is not None else np.nan,
                flip_shap if flip_shap is not None else np.nan,
                float(sal.get("grad_norm", np.nan)),
                float(kdiag.get("frac_kink", np.nan)),
            ])


            
            
            
            f_out.write("5. Robustness Analysis Results:\n")
            f_out.write("-" * 80 + "\n")
            f_out.write("{:<10s} {:<20s} {:<10s} {:<18s} {:<15s}\n".format(
                "Dir. ID", "(thx, thy)", "Phase", "Class Change", "Change Radius"))
            f_out.write("-" * 80 + "\n")
            for row in results_table:
                change_radius_str = f"{row['changed_radius']:.4f}" if row["changed_radius"] is not None else "N/A"
                f_out.write("{:<10d} ({:<6.3f}, {:<6.3f})    {:<10.3f} {:<18s} {:<15s}\n".format(
                    row["direction_id"],
                    row["direction_radians"][0], row["direction_radians"][1],
                    row["phase"],
                    "YES" if row["changed_class"] else "NO",
                    change_radius_str
                ))
            f_out.write("\n")

            f_out.write("6. LIME Explanation (Local Feature Importance):\n")
            for feat, val in lime_list:
                f_out.write(f"   {feat}: {val:.3f}\n")
            f_out.write("\n")

            f_out.write("7. SHAP Explanation (Feature Contributions per Class):\n")
            if shap_class0 is None:
                f_out.write("   SHAP: not available (skipped or failed)\n\n")
            else:
                for fid, feat_name in enumerate(["Re(z1)", "Re(z2)", "Im(z1)", "Im(z2)"]):
                    if shap_class1 is None:
                        f_out.write(f"  {feat_name} => Class: {scalarize(shap_class0[fid]):.3f}\n")
                    else:
                        f_out.write(f"  {feat_name} => Class 0: {scalarize(shap_class0[fid]):.3f}, "
                                    f"Class 1: {scalarize(shap_class1[fid]):.3f}\n")
                f_out.write("\n")


            # 7b. Axis-baseline ray sweeps
            f_out.write(axis_report_str)
            
            f_out.write("8. Resource benchmark (Puiseux vs gradient saliency):\n")
            f_out.write(f"   Puiseux times   : sample={times_pp['time_sampling']:.2f}s, lsq={times_pp['time_lstsq']:.2f}s, "
                        f"factor={times_pp['time_factor']:.2f}s, simplify={times_pp['time_simplify']:.2f}s, "
                        f"expansion={times_pp['time_puiseux']:.2f}s, total={times_pp['time_total']:.2f}s\n")
            f_out.write(f"   Saliency (1x avg): time={sal['time_ms']:.2f} ms, grad_norm={sal['grad_norm']:.3e}, "
                        f"cpu_dRSS={sal['cpu_rss_mb_delta']:.1f} MB, gpu_peak={sal['gpu_peak_mb']:.1f} MB\n")

            f_out.write("\n" + "=" * 80 + "\n")
            f_out.write("End of Report\n")
            f_out.write("=" * 80 + "\n")

    # ==================================================================
    # Calibration CI table (paper-grade): use record-level CV results from up_real
    # ==================================================================
    try:
        cv_multi = os.path.join(IN_DIR, "cv_metrics_per_fold_multi.csv")
        if os.path.isfile(cv_multi):
            dfm = pd.read_csv(cv_multi)

            rows = []
            for method in sorted(dfm["method"].unique()):
                sub = dfm[dfm["method"] == method]
                for metric in ["ECE", "NLL", "Brier", "Acc", "AUC"]:
                    vals = sub[metric].astype(float).values
                    m, ci = mean_ci95(vals)
                    rows.append([method, metric, m, ci])

            out_csv = os.path.join(OUT_DIR, "calibration_ci_table.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Method", "Metric", "Mean", "CI95"])
                for r in rows:
                    w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"])

            out_txt = os.path.join(OUT_DIR, "calibration_ci_report.txt")
            with open(out_txt, "w") as f:
                f.write("=== Calibration (record-level CV from up_real) ===\n")
                f.write(pd.DataFrame(rows, columns=["Method","Metric","Mean","CI95"]).to_string(index=False))
                f.write("\n")

            logger.info("Saved calibration_ci_table.csv and calibration_ci_report.txt from record-level CV.")
        else:
            logger.warning("cv_metrics_per_fold_multi.csv not found in IN_DIR -> skipping calibration_ci_table.")
    except Exception as e:
        logger.warning("Failed to build calibration CI table from record-level CV: %s", e)



    # ==================================================================
    # Aggregate reports across anchors + T-sweep
    # ==================================================================        
    try:
        pd.DataFrame(kink_rows, columns=["point","frac_kink","frac_active","frac_inactive","n"]).to_csv(
            os.path.join(OUT_DIR,"kink_summary.csv"), index=False)

        pd.DataFrame(res_rows, columns=[
            "point",
            "puiseux_time_total_s",
            "puiseux_time_sampling_s",
            "puiseux_time_lstsq_s",
            "puiseux_time_factor_s",
            "puiseux_time_simplify_s",
            "puiseux_time_expand_s",
            "puiseux_cpu_rss_mb",
            "puiseux_gpu_peak_mb",
            "saliency_time_ms",
            "saliency_grad_norm",
            "saliency_cpu_rss_mb_delta",
            "saliency_gpu_peak_mb",
        ]).to_csv(os.path.join(OUT_DIR,"resource_summary.csv"), index=False)

        pd.DataFrame(fit_rows, columns=[
            "point",
            "fit_ok",
            "refit_attempts",
            "kept_ratio",
            "cond",
            "degree_used",
            "n_samples_used",
            "exclude_kink_eps_used",
            "RMSE",
            "MAE",
            "corr_pearson",
            "sign_agree",
            "resid_mean",
            "resid_std",
            "resid_skew",
            "resid_kurt",
        ]).to_csv(os.path.join(OUT_DIR,"local_fit_summary.csv"), index=False)

        pd.DataFrame(dom_rows, columns=[
            "point",
            "c2_max_abs",
            "c4_max_abs",
            "r_dom",
            "flip_found",
            "r_flip_obs",
            "r_flip_cens",
            "flip_grad",
            "flip_lime",
            "flip_shap",
            "saliency_grad_norm",
            "frac_kink",
        ]).to_csv(os.path.join(OUT_DIR, "dominant_ratio_summary.csv"), index=False)



        # --- GLOBAL kink report ---
        ks = pd.DataFrame(kink_rows, columns=["point","frac_kink","frac_active","frac_inactive","n"])
        lf_full = pd.DataFrame(
            fit_rows,
            columns=[
                "point", "fit_ok", "refit_attempts", "kept_ratio", "cond", "degree_used",
                "n_samples_used", "exclude_kink_eps_used", "RMSE", "MAE",
                "corr_pearson", "sign_agree", "resid_mean", "resid_std", "resid_skew", "resid_kurt",
            ],
        )
        lf = lf_full[
            [
                "point", "kept_ratio", "cond", "degree_used", "RMSE", "sign_agree",
                "resid_mean", "resid_std", "resid_skew", "resid_kurt",
            ]
        ]

        rep = []
        for thr in (0.01, 0.05, 0.10):
            mask = ks["frac_kink"] >= thr
            share = float(mask.mean())
            n_hi = int(mask.sum())
            med = float(ks.loc[mask, "frac_kink"].median()) if n_hi > 0 else float("nan")
            rep.append((thr, share, med, n_hi))

        with open(os.path.join(OUT_DIR, "kink_global_summary.txt"), "w") as f:
            f.write("=== Kink prevalence across anchors ===\n")
            for thr, share, med, n_hi in rep:
                if n_hi > 0 and np.isfinite(med):
                    f.write(f"frac_kink >= {thr:.2%}: share={share:.3f}, median(frac_kink)={med:.3f} (n={n_hi})\n")
                else:
                    f.write(f"frac_kink >= {thr:.2%}: share={share:.3f}, median(frac_kink)=N/A (n=0)\n")

            # Effect of kinks on quality and residuals:
            cols = ["RMSE", "sign_agree", "resid_mean", "resid_std", "resid_skew", "resid_kurt"]
            for thr in (0.01, 0.05, 0.10):
                mask = ks["frac_kink"] >= thr
                lo_df = lf.loc[~mask, cols]
                hi_df = lf.loc[mask, cols]

                lo = lo_df.mean() if len(lo_df) > 0 else None
                hi = hi_df.mean() if len(hi_df) > 0 else None

                f.write(f"\n-- threshold {thr:.2%} --\n")
                f.write(f"LOW-KINK mean: {lo.to_dict() if lo is not None else 'N/A'}\n")
                f.write(f"HIGH-KINK mean: {hi.to_dict() if hi is not None else 'N/A (no high-kink points)'}\n")
    except Exception as e:
        logger.warning("Saving aggregate summaries failed: %s", e)



    # ------------------------------------------------------------
    # Error-triage evaluation (quantifies the *practical* gain of Puiseux-guided probes)
    # ------------------------------------------------------------
    if not args.skip_triage_eval:
        try:
            run_puiseux_error_triage_eval(
                out_dir=OUT_DIR,
                n_boot=int(args.triage_bootstrap),
                seed=int(args.seed),
                pr_max_curves=int(args.triage_pr_max_curves),
            )
        except Exception as e:
            logger.warning("Puiseux error-triage eval failed: %s", e)        

    # ECE sensitivity to branch multiplicity error m:
    # Assume T ∝ m^{-1/2}. If m_est = m_true*(1+eps),
    # then T_mult = (m_true/m_est)^{1/2} = (1+eps)^{-1/2}.
    if getattr(args, "skip_multiplicity", False) or (cal.method != "TEMPERATURE") or (T is None):
        logger.info("Multiplicity-sensitivity sweep skipped (requires TEMPERATURE + T).")
    else:
        try:
            if X_full is None or y_full is None:
                logger.warning("Multiplicity-sensitivity sweep requested but X_full/y_full not loaded; skipping.")
            else:
                X_full_c2 = compress_to_C2(X_full)
                X_full_c2 = scaler_full.transform(X_full_c2) if scaler_full is not None else X_full_c2

                rel_errs = (-0.5, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)
                from src.post_processing import sweep_multiplicity_misestimation
                res = sweep_multiplicity_misestimation(
                    model, X_full, y_full,
                    compress_fn=compress_to_C2, scaler=scaler_full,
                    device=device, T_base=T,
                    rel_errors=rel_errs, gamma=0.5
                )
                df_m = pd.DataFrame(res, columns=["rel_err_m", "ECE"])
                df_m.to_csv(os.path.join(OUT_DIR, "branch_multiplicity_sensitivity.csv"), index=False)
                logger.info("Saved branch_multiplicity_sensitivity.csv")
        except Exception as e:
            logger.warning("Multiplicity-sensitivity sweep failed: %s", e)



    print("\n[INFO] Completed analysis for all uncertain points.")


    # ----------------------------------------------------------------------
    # FINAL (paper-ready): always write summary + README + manifest + optional zip
    # ----------------------------------------------------------------------
    try:
        write_paper_summary_md(
            out_dir=OUT_DIR,
            in_dir=IN_DIR,
            run_id=run_id,
            calib_used=calib_used,
            run_meta=_run_meta,
            run_args=_run_args,
        )
        logger.info("Wrote paper_summary.md (final).")
    except Exception as e:
        logger.warning("Failed to write paper_summary.md (final): %s", e)

    try:
        write_missing_outputs_report(OUT_DIR, n_points=len(up_list))
        logger.info("Wrote missing_outputs_report.txt")
    except Exception as e:
        logger.warning("Failed to write missing_outputs_report.txt: %s", e)

    try:
        manifest = build_post_processing_manifest(
            out_dir=OUT_DIR,
            in_dir=IN_DIR,
            run_id=run_id,
            calib_used=calib_used,
            run_meta=_run_meta,
            run_args=_run_args,
            pp_args=vars(args),
        )
        write_post_processing_readme_md(
            out_dir=OUT_DIR,
            in_dir=IN_DIR,
            run_id=run_id,
            calib_used=calib_used,
            manifest=manifest,
        )
        logger.info("Wrote README_post_processing_real.md + post_processing_manifest.json")
    except Exception as e:
        logger.warning("Failed to write README/manifest: %s", e)

    if bool(getattr(args, "zip_bundle", False)):
        try:
            parent = os.path.dirname(os.path.abspath(OUT_DIR))
            base = os.path.basename(os.path.abspath(OUT_DIR)).rstrip(os.sep)
            zip_name = (getattr(args, "zip_name", "") or "").strip()
            if not zip_name:
                zip_name = f"{base}.zip"
            zip_path = os.path.join(parent, zip_name)
            zip_directory(OUT_DIR, zip_path)
            logger.info("Created zip bundle: %s", zip_path)
            print(f"[INFO] Zip bundle created: {zip_path}")
        except Exception as e:
            logger.warning("Failed to create zip bundle: %s", e)


    # Make sure the log file is flushed/closed (prevents empty post_processing_real.log in some environments)
    try:
        logging.shutdown()
    except Exception:
        pass
