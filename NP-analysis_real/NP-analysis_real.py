# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.

"""
Build Newton–Puiseux evidence by joining anchors, benchmark TXT, and
dominant-ratio CSV produced by post_processing_*.
"""

import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


import json
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from typing import Dict, Any, List, Tuple, Optional



def _first_existing(pp_dir: Path, candidates):
    for name in candidates:
        p = pp_dir / name
        if p.exists():
            try:
                if p.stat().st_size == 0:
                    continue
            except Exception:
                pass
            return p
    return None


def load_sensitivity_outputs(pp_dir: Path):
    """
    Load sensitivity artifacts produced by post_processing_real.
    Tolerates historical typos: 'sensivity_*' vs 'sensitivity_*'.
    """
    p_det = _first_existing(pp_dir, ["sensitivity_detailed.csv", "sensivity_detailed.csv"])
    p_mul = _first_existing(pp_dir, ["sensitivity_multi.csv", "sensivity_multi.csv"])
    p_sum = _first_existing(pp_dir, ["sensitivity_summary.txt", "sensivity_summary.txt"])

    df_det = None
    df_mul = None
    txt_sum = None

    if p_det is not None:
        try:
            df_det = pd.read_csv(p_det)
        except Exception as e:
            print(f"[WARN] Failed to read {p_det.name}: {e}")

    if p_mul is not None:
        try:
            df_mul = pd.read_csv(p_mul)
        except Exception as e:
            print(f"[WARN] Failed to read {p_mul.name}: {e}")

    if p_sum is not None:
        try:
            txt_sum = p_sum.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Failed to read {p_sum.name}: {e}")

    return df_det, df_mul, txt_sum



# ---------- PR tools (AUPRC for ranking by a score) ----------
def precision_recall_from_scores(y_true: np.ndarray, scores: np.ndarray):
    """
    Compute a precision–recall curve and step-based AP for a ranking signal.

    Parameters
    ----------
    y_true : np.ndarray
        1D array of {0, 1} where 1 = 'fragile' (flip within budget).
    scores : np.ndarray
        1D array of floats; higher = 'more fragile' (e.g., |c4|).

    Returns
    -------
    rec : np.ndarray
        Recall points (starts at 0).
    prec : np.ndarray
        Precision points (starts at 1 by convention).
    ap_step : float
        Average precision via step integration.
    df_curve : pandas.DataFrame
        Per-prefix table with k, threshold(score), tp, fp, fn, tn, precision, recall.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)

    # Keep only finite scores.
    valid = np.isfinite(s)
    y = y[valid]
    s = s[valid]

    # Stable sort by descending score.
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    s = s[order]

    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    P = int(tp_cum[-1])            # positives
    N = int(fp_cum[-1])            # negatives

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / max(P, 1)

    # Prepend (recall=0, precision=1) for a conventional start.
    rec = np.concatenate(([0.0], recall))
    prec = np.concatenate(([1.0], precision))

    # Step-wise AP (area under the PR curve).
    ap_step = np.sum((rec[1:] - rec[:-1]) * prec[1:])

    k = np.arange(1, len(y) + 1)
    df_curve = pd.DataFrame({
        "k": k,
        "threshold": s,                   # score threshold at prefix k
        "tp": tp_cum,
        "fp": fp_cum,
        "fn": P - tp_cum,
        "tn": N - fp_cum,
        "precision": precision,
        "recall": recall
    })

    return rec, prec, float(ap_step), df_curve


# ---------- Robust float regex ----------
FLOAT_RE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

# ---------- TXT parser ----------
def parse_benchmark_file(txt_path: str) -> dict:
    """
    Parse a single post_processing_*/benchmark_point*.txt file and extract:
    - kink diagnostics (frac_kink/active/inactive, samples_checked),
    - local-fit metrics (kept_ratio, cond, rank, n_monomials, degree_used, retry),
    - approximation metrics (RMSE/MAE/Pearson/Sign_Agreement, residual moments),
    - robustness table (flip radii and minimum flip radius),
    - Puiseux and saliency timings (including CPU/GPU memory and grad_norm),
    - optional r_dom prediction and axis-baseline sweep flips (grad/LIME/SHAP).

    Returns
    -------
    dict
        A dictionary with parsed scalars and lists ready to be joined downstream.
    """
    p = Path(txt_path)
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    out = {
        "point": None,
        "frac_kink": np.nan,
        "frac_active": np.nan,
        "frac_inactive": np.nan,
        "samples_checked": np.nan,

        "kept_ratio": np.nan,
        "cond": np.nan,
        "rank": np.nan,
        "n_monomials": np.nan,
        "degree_used": np.nan,   # NEW
        "retry": np.nan,         # NEW

        "RMSE": np.nan,
        "MAE": np.nan,
        "Pearson": np.nan,
        "Sign_Agreement": np.nan,
        "resid_mean": np.nan,    # NEW
        "resid_std": np.nan,     # NEW
        "resid_skew": np.nan,    # NEW
        "resid_kurt": np.nan,    # NEW

        "flip_radii": [],
        "min_flip_radius": np.nan,

        "puiseux_time_s": np.nan,
        "saliency_ms": np.nan,
        "saliency_cpu_dRSS_MB": np.nan,
        "saliency_gpu_peak_MB": np.nan,
        "saliency_grad_norm": np.nan,  # NOTE: the trailing comma was needed in the log line

        "r_dom_pred": np.nan,
        "flip_grad": np.nan,     # NEW default
        "flip_lime": np.nan,     # NEW default
        "flip_shap": np.nan,     # NEW default
    }

    m_pt = re.search(r'point(\d+)', p.name)
    if m_pt:
        out["point"] = int(m_pt.group(1))

    # Scalars, timings, optional r_dom_pred.
    for raw in lines:
        s = raw.strip()
        s_norm = s.replace("—", "-")  # normalize em-dash

        # Kink diagnostics.
        if "frac_kink" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["frac_kink"] = float(m.group()) if m else np.nan
        elif "frac_active" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["frac_active"] = float(m.group()) if m else np.nan
        elif "frac_inactive" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["frac_inactive"] = float(m.group()) if m else np.nan
        elif "samples_checked" in s_norm:
            m = re.search(r'\d+', s_norm);    out["samples_checked"] = int(m.group()) if m else np.nan

        # Local-fit section.
        elif "kept / total" in s_norm:
            m = re.search(r'(\d+)\s*/\s*(\d+).*?\(\s*(' + FLOAT_RE + r')\s*%\)', s_norm)
            if m: out["kept_ratio"] = float(m.group(3)) / 100.0
        elif "cond(A)" in s_norm:
            m = re.search(r'cond\(A\)\s*:\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["cond"] = float(m.group(1))
        elif "rank / monomials" in s_norm:
            m = re.search(r'(\d+)\s*/\s*(\d+)', s_norm)
            if m: out["rank"], out["n_monomials"] = int(m.group(1)), int(m.group(2))
        elif "degree_used" in s_norm:
            m = re.search(r'degree_used\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["degree_used"] = float(m.group(1))
        elif "retry" in s_norm:
            m = re.search(r'retry\s*[:=]\s*(\d+)', s_norm)
            if m: out["retry"] = int(m.group(1))

        # Approximation quality.
        elif s_norm.startswith("RMSE"):
            m = re.search(FLOAT_RE, s_norm);  out["RMSE"] = float(m.group()) if m else np.nan
        elif s_norm.startswith("MAE"):
            m = re.search(FLOAT_RE, s_norm);  out["MAE"] = float(m.group()) if m else np.nan
        elif "Pearson" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["Pearson"] = float(m.group()) if m else np.nan
        elif "Sign Agreement" in s_norm:
            m = re.search(FLOAT_RE, s_norm);  out["Sign_Agreement"] = float(m.group()) if m else np.nan
        elif "resid_mean" in s_norm:
            m = re.search(r'resid_mean\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_mean"] = float(m.group(1))
        elif "resid_std" in s_norm:
            m = re.search(r'resid_std\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_std"] = float(m.group(1))
        elif "resid_skew" in s_norm:
            m = re.search(r'resid_skew\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_skew"] = float(m.group(1))
        elif "resid_kurt" in s_norm:
            m = re.search(r'resid_kurt\s*[:=]\s*(' + FLOAT_RE + r')', s_norm)
            if m: out["resid_kurt"] = float(m.group(1))

        # Puiseux timing.
        elif s_norm.startswith("Puiseux times"):
            mt = re.search(r'total=('+FLOAT_RE+')s', s_norm)
            if mt: out["puiseux_time_s"] = float(mt.group(1))

        # Saliency timing + resources.
        elif s_norm.startswith("Saliency"):
            mt = re.search(r'time=(' + FLOAT_RE + r')\s*ms', s_norm)
            if mt: out["saliency_ms"] = float(mt.group(1))
            mr = re.search(r'cpu_dRSS=(' + FLOAT_RE + r')\s*MB', s_norm)
            if mr: out["saliency_cpu_dRSS_MB"] = float(mr.group(1))
            mp = re.search(r'gpu_peak=(' + FLOAT_RE + r')\s*MB', s_norm)
            if mp: out["saliency_gpu_peak_MB"] = float(mp.group(1))
            mg = re.search(r'grad_norm=(' + FLOAT_RE + r')', s_norm)
            if mg: out["saliency_grad_norm"] = float(mg.group(1))

        # Optional r_dom/onset radius line.
        elif re.search(r'(r_dom|onset radius)', s_norm, flags=re.IGNORECASE):
            m = re.search(r'(?:r_dom|onset radius)[^=]*=\s*(' + FLOAT_RE + r')',
                          s_norm, flags=re.IGNORECASE)
            if m:
                out["r_dom_pred"] = float(m.group(1))

        # AXIS-BASELINE header -> initialize flip_* fields.
        elif s_norm.startswith("AXIS-BASELINE RAY SWEEPS"):
            out["flip_grad"] = np.nan
            out["flip_lime"] = np.nan
            out["flip_shap"] = np.nan
        elif s_norm.startswith("flip_grad"):
            m = re.search(FLOAT_RE, s_norm);  out["flip_grad"] = float(m.group()) if m else np.nan
        elif s_norm.startswith("flip_lime"):
            m = re.search(FLOAT_RE, s_norm);  out["flip_lime"] = float(m.group()) if m else np.nan
        elif s_norm.startswith("flip_shap"):
            m = re.search(FLOAT_RE, s_norm);  out["flip_shap"] = float(m.group()) if m else np.nan

    # Parse robustness results table -> collect flip radii.
    in_table = False
    for raw in lines:
        s = raw.strip()
        s_norm = s.replace("—", "-")

        if s_norm.startswith("Dir. ID"):
            in_table = True
            continue

        if in_table:
            # End of table: blank line OR a new numbered section header, e.g. "6. Puiseux times ..."
            if not s_norm:
                in_table = False
                continue
            if re.match(r'^\d+\.\s*$', s_norm) and not re.search(r'\b(YES|NO)\b', s_norm, flags=re.IGNORECASE):
                in_table = False
                continue
            if re.match(r'^\d+\.\s+\D', s_norm) and not re.search(r'\b(YES|NO)\b', s_norm, flags=re.IGNORECASE):
                in_table = False
                continue

            # Skip separators
            if set(s_norm) <= {"-", " "}:
                continue

            # Keep only directions where a flip was actually found.
            if re.search(r'\bYES\b', s_norm, flags=re.IGNORECASE):
                m = re.search(r'(' + FLOAT_RE + r')\s*$', s_norm)
                if m:
                    try:
                        out["flip_radii"].append(float(m.group(1)))
                    except ValueError:
                        pass  # e.g., "N/A"


    if out["flip_radii"]:
        out["min_flip_radius"] = min(out["flip_radii"])

    return out



def collect_benchmarks(pp_dir: Path) -> pd.DataFrame:
    """
    Read all benchmark_point*.txt files in a post_processing_* directory
    and assemble a normalized DataFrame of the parsed contents.

    Parameters
    ----------
    pp_dir : pathlib.Path
        Path to the post-processing directory.

    Returns
    -------
    pandas.DataFrame
        Subset of standardized columns useful for joining and analysis.
    """
    paths = sorted(glob.glob(str(pp_dir / "benchmark_point*.txt")))
    if not paths:
        print(f"[WARN] No benchmark_point*.txt found under: {pp_dir}")
        return pd.DataFrame()
    rows = [parse_benchmark_file(p) for p in paths]
    df = pd.DataFrame(rows)
    keep = [
        "point", "min_flip_radius", "flip_radii",
        "frac_kink", "frac_active", "frac_inactive",
        "kept_ratio", "cond", "rank", "n_monomials",
        "degree_used", "retry",
        "RMSE", "MAE", "Pearson", "Sign_Agreement",
        "resid_mean", "resid_std", "resid_skew", "resid_kurt",
        "puiseux_time_s", "saliency_ms", "saliency_cpu_dRSS_MB", "saliency_gpu_peak_MB",
        "saliency_grad_norm","r_dom_pred", "flip_grad", "flip_lime", "flip_shap"
    ]
    return df[[c for c in keep if c in df.columns]]


# ---------- Loaders ----------
def load_anchors(up_dir: Path) -> pd.DataFrame:
    """
    Load uncertain anchors and derive convenience columns.

    - Adds a 'point' index [0..N-1] for consistent joining.
    - Renames 'index' to 'anchor_index' if present (for clarity).
    - Computes pmax = max(p1, p2) and margin = |p1 - p2| when available.
    """
    # Prefer the newer *_ext export if present (it contains extra metadata).
    path = _first_existing(up_dir, ["uncertain_full_ext.csv", "uncertain_full.csv"])
    if path is None:
        raise FileNotFoundError(
            f"Could not find anchors under: {up_dir}. Expected one of: uncertain_full_ext.csv, uncertain_full.csv"
        )
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["point"] = np.arange(len(df), dtype=int)
    if "index" in df.columns:
        df = df.rename(columns={"index": "anchor_index"})
    if {"p1", "p2"}.issubset(df.columns):
        df["pmax"] = df[["p1", "p2"]].max(axis=1)
        df["margin"] = (df["p1"] - df["p2"]).abs()
    return df


def load_dom_ratio(pp_dir: Path) -> pd.DataFrame:
    """
    Load dominant-ratio summary produced by post_processing_real.

    Supports both old and new column variants, normalizes names, and returns
    a DataFrame for joining on 'point'.

    Expected (new) columns (subset; the file may contain more):
      point, c2_max_abs, c4_max_abs, r_dom,
      r_flip_obs, r_flip_cens, flip_found, r_flip_eff,
      flip_grad, flip_lime, flip_shap, saliency_grad_norm, frac_kink
    """
    path = pp_dir / "dominant_ratio_summary.csv"
    if not path.exists():
        print(f"[WARN] Missing {path.name}; will rely on r_dom_pred from TXT if available.")
        return pd.DataFrame()

    # Guard: empty file
    try:
        if path.stat().st_size == 0:
            print(f"[WARN] {path.name} is empty; skipping.")
            return pd.DataFrame()
    except Exception:
        pass

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return pd.DataFrame()

    # Normalize common variants
    df = df.rename(columns={
        "point_id": "point",
        "max_abs_c2": "c2_max_abs",
        "max_abs_c4": "c4_max_abs",
        "r_flip": "r_flip_obs",  # old naming -> new canonical
    })


    # Ensure numeric types where expected
    for c in [
        "point", "c2_max_abs", "c4_max_abs", "r_dom",
        "r_flip_obs", "r_flip_cens", "flip_found", "r_flip_eff",
        "flip_grad", "flip_lime", "flip_shap",
        "saliency_grad_norm", "frac_kink",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Provide an effective flip radius that respects censoring.
    # If flip_found==1 -> use r_flip_obs; else use r_flip_cens.
    if "r_flip_eff" not in df.columns:
        if "flip_found" in df.columns and "r_flip_cens" in df.columns:
            ff = pd.to_numeric(df["flip_found"], errors="coerce").fillna(0).astype(int)
            r_obs = pd.to_numeric(df["r_flip_obs"], errors="coerce") if "r_flip_obs" in df.columns else pd.Series(np.nan, index=df.index)
            r_cens = pd.to_numeric(df["r_flip_cens"], errors="coerce")
            df["r_flip_eff"] = np.where(ff.values == 1, r_obs.values, r_cens.values)
        elif "r_flip_obs" in df.columns:
            df["r_flip_eff"] = pd.to_numeric(df["r_flip_obs"], errors="coerce")

    keep = [c for c in [
        "point", "c2_max_abs", "c4_max_abs", "r_dom",
        "r_flip_obs", "r_flip_cens", "flip_found", "r_flip_eff",
        "flip_grad", "flip_lime", "flip_shap",
        "saliency_grad_norm", "frac_kink",
    ] if c in df.columns]
    return df[keep]


def parse_args():
    p = argparse.ArgumentParser()


    # -------------------------
    # New BSPC-focused analysis
    # -------------------------
    p.add_argument(
        "--mode",
        type=str,
        default="classic",
        choices=["classic", "bspc_error"],
        help=(
            "Which analysis to run. "
            "'bspc_error' reads post_processing artefacts (selected_uncertain_points.csv, "
            "dominant_ratio_summary.csv, local_fit_summary.csv) and produces publication-ready "
            "error-triage plots/tables. "
            "'classic' keeps the legacy behaviour."
        ),
    )
    p.add_argument(
        "--pp_dirs",
        type=str,
        default="",
        help=(
            "Comma-separated list of post_processing output dirs. Each must contain "
            "selected_uncertain_points.csv (and ideally dominant_ratio_summary.csv + local_fit_summary.csv). "
            "Used in --mode bspc_error."
        ),
    )
    p.add_argument(
        "--cohort_names",
        type=str,
        default="",
        help="Optional comma-separated names aligned with --pp_dirs. If empty, dir basenames are used.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/NP-analysis_real_bspc",
        help="Output directory for figures/tables in --mode bspc_error.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for AUPRC/AUROC confidence intervals (bspc_error mode).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for bootstrapping (bspc_error mode).",
    )

    p.add_argument(
        "--plots_only",
        action="store_true",
        help=(
            "bspc_error: only (re)generate figures. "
            "Skips expensive bootstrap computations and does NOT overwrite CSV summaries "
            "if they already exist in --out_dir."
        ),
    )

    p.add_argument(
        "--bootstrap_kind",
        type=str,
        default="iid",
        choices=["iid", "stratified", "cluster_record"],
        help=(
            "Bootstrap scheme for CI estimation in bspc_error mode. "
            "'iid' resamples points, 'stratified' resamples positives/negatives separately "
            "(recommended for rare positives), 'cluster_record' resamples records/clusters with replacement."
        ),
    )
    p.add_argument(
        "--bootstrap_cluster_col",
        type=str,
        default="record",
        help="Column used as cluster id when --bootstrap_kind=cluster_record (default: record).",
    )


    p.add_argument(
        "--budget_fracs",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.3",
        help="Comma-separated review budget fractions for budget-risk curves (bspc_error mode).",
    )
    p.add_argument(
        "--label_col",
        type=str,
        default="is_error",
        help="Binary label column to use in bspc_error mode (default: is_error).",
    )

    # -------------------------
    # Legacy (classic) analysis
    # -------------------------

    p.add_argument("--radius_budget", type=float, default=0.02,
                   help="Primary flip-radius budget used for the main report/figures (classic mode).")
    p.add_argument("--radius_budget_sweep", type=str, default="0.002,0.005,0.01,0.02",
                   help="Comma-separated flip-radius budgets for AUPRC-vs-budget sweep. Empty disables (classic mode).")
    return p.parse_args()


def _parse_float_list(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            pass
    return out


# ---------- error-triage analysis ----------

def _parse_fracs(s: str) -> List[float]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _to_int01(series: pd.Series) -> pd.Series:
    # Accept bool/0-1/int/float strings
    if series.dtype == bool:
        return series.astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def _finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def load_pp_cohort(pp_dir: str) -> pd.DataFrame:
    pp_dir_p = Path(pp_dir)
    sel_path = pp_dir_p / "selected_uncertain_points.csv"
    if not sel_path.is_file():
        raise FileNotFoundError(f"Missing selected_uncertain_points.csv in: {pp_dir_p}")

    df = pd.read_csv(sel_path)

    # -----------------------------
    # Normalise point id (join key)
    # -----------------------------
    if "pp_id" in df.columns and "point" not in df.columns:
        df = df.rename(columns={"pp_id": "point"})

    if "point" not in df.columns:
        # Try common alternative ids first (keeps joins consistent with dominant_ratio_summary)
        if "anchor_index" in df.columns:
            df["point"] = pd.to_numeric(df["anchor_index"], errors="coerce")
        elif "index" in df.columns:
            df["point"] = pd.to_numeric(df["index"], errors="coerce")
        else:
            df["point"] = np.arange(len(df), dtype=int)

    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    if df["point"].isna().any():
        print(f"[WARN] Some 'point' values are NaN in {pp_dir_p}; using row-order fallback point ids (joins may be unreliable).")
        df["point"] = np.arange(len(df), dtype=int)

    df["point"] = df["point"].astype(int)

    if df["point"].duplicated().any():
        print(f"[WARN] Duplicate 'point' ids detected in {sel_path} (n_dup={int(df['point'].duplicated().sum())}).")

    # -----------------------------
    # Label normalisation
    # -----------------------------
    if "is_error" in df.columns:
        df["is_error"] = _to_int01(df["is_error"])
    else:
        # Optional fallback: derive is_error if standard truth/pred columns are present.
        derived = False
        for tcol, pcol in [
            ("y_true", "y_pred"),
            ("true", "pred"),
            ("true_label", "pred_label"),
            ("label", "pred_label"),
            ("target", "pred"),
        ]:
            if {tcol, pcol}.issubset(df.columns):
                yt = pd.to_numeric(df[tcol], errors="coerce")
                yp = pd.to_numeric(df[pcol], errors="coerce")
                ok = yt.notna() & yp.notna()
                df["is_error"] = ((ok) & (yp != yt)).astype(int)
                df["is_error"] = _to_int01(df["is_error"])
                print(f"[WARN] 'is_error' missing in {pp_dir_p}; derived from '{pcol}' vs '{tcol}'.")
                derived = True
                break
        if not derived:
            # Leave missing; downstream will raise if args.label_col expects it.
            pass

    if "accepted" in df.columns:
        df["accepted"] = _to_int01(df["accepted"])


    # Robust: derive pmax/margin if missing (some runs may only store p1/p2)
    if ("pmax" not in df.columns or "margin" not in df.columns) and ("p1" in df.columns and "p2" in df.columns):
        p1 = pd.to_numeric(df["p1"], errors="coerce")
        p2 = pd.to_numeric(df["p2"], errors="coerce")
        if "pmax" not in df.columns:
            df["pmax"] = pd.concat([p1, p2], axis=1).max(axis=1)
        if "margin" not in df.columns:
            df["margin"] = (p1 - p2).abs()
 
    # Coerce common numeric cols early (prevents dropping everything due to strings)
    for c in ["pmax", "margin", "r_dom", "c2_max_abs", "c4_max_abs", "saliency_grad_norm", "flip_grad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")        

    # Merge Puiseux dominant ratio summary (if available)
    dom_path = pp_dir_p / "dominant_ratio_summary.csv"
    if dom_path.is_file():
        df_dom = pd.read_csv(dom_path)
        if "point" not in df_dom.columns and "pp_id" in df_dom.columns:
            df_dom = df_dom.rename(columns={"pp_id": "point"})
        df = df.merge(df_dom, on="point", how="left", suffixes=("", "_dom"))
        # If the same numeric signal exists in both sources, prefer the dom summary as a fill-in.
        for col in [
            "r_dom", "c2_max_abs", "c4_max_abs",
            "r_flip_obs", "r_flip_cens", "flip_found", "r_flip_eff",
            "saliency_grad_norm", "frac_kink",
            "flip_grad", "flip_lime", "flip_shap",
        ]:
            dom_col = f"{col}_dom"
            if dom_col in df.columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").combine_first(
                        pd.to_numeric(df[dom_col], errors="coerce")
                    )
                else:
                    df[col] = pd.to_numeric(df[dom_col], errors="coerce")
                df = df.drop(columns=[dom_col])

        # Ensure r_flip_eff exists (needed for censored flip radii).
        if "r_flip_eff" not in df.columns:
            if "flip_found" in df.columns and "r_flip_cens" in df.columns:
                ff = pd.to_numeric(df["flip_found"], errors="coerce").fillna(0).astype(int)
                r_obs = pd.to_numeric(df["r_flip_obs"], errors="coerce") if "r_flip_obs" in df.columns else pd.Series(np.nan, index=df.index)
                r_cens = pd.to_numeric(df["r_flip_cens"], errors="coerce")
                df["r_flip_eff"] = np.where(ff.values == 1, r_obs.values, r_cens.values)
            elif "r_flip_obs" in df.columns:
                df["r_flip_eff"] = pd.to_numeric(df["r_flip_obs"], errors="coerce")

    # Merge local fit summary (if available)
    fit_path = pp_dir_p / "local_fit_summary.csv"
    if fit_path.is_file():
        df_fit = pd.read_csv(fit_path)
        if "point" not in df_fit.columns and "pp_id" in df_fit.columns:
            df_fit = df_fit.rename(columns={"pp_id": "point"})
        # Avoid column collisions
        if "RMSE" in df_fit.columns and "RMSE" in df.columns:
            df_fit = df_fit.rename(columns={"RMSE": "fit_RMSE"})
        df = df.merge(df_fit, on="point", how="left", suffixes=("", "_fit"))

    return df


def _bootstrap_indices(
    rng: np.random.Generator,
    y: np.ndarray,
    kind: str,
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a bootstrap resample index array of length n=len(y).

    kind:
      - 'iid'            : sample points with replacement
      - 'stratified'     : sample positives and negatives separately (same class counts as original)
      - 'cluster_record' : sample clusters/records with replacement, then take ~n points
    """
    y = np.asarray(y).astype(int)
    n = int(len(y))
    if n <= 0:
        return np.array([], dtype=int)

    if kind == "iid":
        return rng.integers(0, n, size=n)

    if kind == "stratified":
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        # Fallback to iid if one class is missing
        if len(pos) == 0 or len(neg) == 0:
            return rng.integers(0, n, size=n)
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        rng.shuffle(idx)
        return idx

    if kind == "cluster_record":
        if cluster_ids is None:
            return rng.integers(0, n, size=n)
        cid = np.asarray(cluster_ids)
        if len(cid) != n:
            # Defensive fallback
            return rng.integers(0, n, size=n)

        # Build mapping cluster -> indices
        clusters: Dict[Any, List[int]] = {}
        for i, c in enumerate(cid):
            clusters.setdefault(c, []).append(i)
        keys = list(clusters.keys())
        if len(keys) == 0:
            return rng.integers(0, n, size=n)

        idx_out: List[int] = []
        # Sample clusters with replacement until we have >= n points, then trim.
        while len(idx_out) < n:
            j = int(rng.integers(0, len(keys)))
            k = keys[j]
            idx_out.extend(clusters[k])
        idx = np.asarray(idx_out, dtype=int)
        rng.shuffle(idx)
        return idx[:n]

    # Unknown kind -> iid
    return rng.integers(0, n, size=n)


def auprc_with_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
    bootstrap_kind: str = "iid",
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Returns (point_estimate, lo95, hi95) for AUPRC."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    m = _finite_mask(score)
    y = y[m]
    score = score[m]
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[m]    
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = float(average_precision_score(y, score))

    rng = np.random.default_rng(seed)
    n = len(y)
    aps = []
    for _ in range(int(n_boot)):
        idx = _bootstrap_indices(rng, y, bootstrap_kind, cluster_ids=cluster_ids)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aps.append(float(average_precision_score(yb, score[idx])))
    if len(aps) < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(aps, [2.5, 97.5])
    return point, float(lo), float(hi)


def auroc_with_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    seed: int,
    bootstrap_kind: str = "iid",
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Returns (point_estimate, lo95, hi95) for AUROC."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    m = _finite_mask(score)
    y = y[m]
    score = score[m]
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")

    point = float(roc_auc_score(y, score))

    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(int(n_boot)):
        idx = _bootstrap_indices(rng, y, bootstrap_kind, cluster_ids=cluster_ids)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(float(roc_auc_score(yb, score[idx])))
    if len(aucs) < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, float(lo), float(hi)


def delta_auprc_with_ci(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    seed: int,
    bootstrap_kind: str = "iid",
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, int]:
    """
    Paired bootstrap CI for ΔAUPRC = AUPRC(score_a) - AUPRC(score_b),
    computed on the *intersection* of finite values for both scores.

    Returns: (delta_point, lo95, hi95, n_used)
    """
    y = np.asarray(y).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    y = y[m]
    a = a[m]
    b = b[m]
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)[m]
    n_used = int(len(y))
    if n_used == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan"), n_used

    delta_point = float(average_precision_score(y, a) - average_precision_score(y, b))

    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    for _ in range(int(n_boot)):
        idx = _bootstrap_indices(rng, y, bootstrap_kind, cluster_ids=cluster_ids)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(float(average_precision_score(yb, a[idx]) - average_precision_score(yb, b[idx])))
    if len(deltas) < 10:
        return delta_point, float("nan"), float("nan"), n_used
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta_point, float(lo), float(hi), n_used    


def budget_curve(y: np.ndarray, score: np.ndarray, fracs: List[float]) -> pd.DataFrame:
    """Compute capture rate & residual risk vs review budget."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    m = _finite_mask(score)
    y = y[m]
    score = score[m]
    n = len(y)
    if n == 0:
        return pd.DataFrame()

    order = np.argsort(-score)  # higher score = more risky
    y_sorted = y[order]

    total_pos = float(y_sorted.sum())
    rows = []
    for f in fracs:
        f = float(f)
        k = int(np.ceil(f * n))
        k = max(0, min(k, n))
        top = y_sorted[:k]
        rest = y_sorted[k:]

        captured = float(top.sum())
        capture_rate = captured / total_pos if total_pos > 0 else float("nan")
        precision_top = captured / float(k) if k > 0 else float("nan")
        risk_rest = float(rest.sum()) / float(len(rest)) if len(rest) > 0 else float("nan")

        rows.append({
            "review_frac": f,
            "review_n": k,
            "precision_top": precision_top,
            "capture_rate": capture_rate,
            "residual_risk": risk_rest,
        })

    return pd.DataFrame(rows)


def two_sided_confidence_budget_curve(y: np.ndarray, pmax: np.ndarray, fracs: List[float]) -> pd.DataFrame:
    """
    Two-sided confidence policy (fair silent-failure baseline):

    For each budget fraction f (=> k samples), review:
      - floor(k/2) samples with LOWEST pmax  (low confidence)
      - remaining samples with HIGHEST pmax (overconfidence)
    Returns the same columns as budget_curve(): review_frac, review_n, precision_top, capture_rate, residual_risk.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(pmax, dtype=float)

    m = np.isfinite(p)
    y = y[m]
    p = p[m]
    n = len(y)
    if n == 0:
        return pd.DataFrame()

    # ascending order of pmax (lowest first)
    order = np.argsort(p, kind="mergesort")
    total_pos = float(y.sum())

    rows = []
    for f in fracs:
        f = float(f)
        k = int(np.ceil(f * n))
        k = max(0, min(k, n))

        k_low = k // 2
        k_high = k - k_low

        review = np.zeros(n, dtype=bool)
        if k_low > 0:
            review[order[:k_low]] = True
        if k_high > 0:
            review[order[-k_high:]] = True

        top = y[review]
        rest = y[~review]

        captured = float(top.sum())
        precision_top = captured / float(k) if k > 0 else float("nan")
        capture_rate = captured / total_pos if total_pos > 0 else float("nan")
        residual_risk = float(rest.sum()) / float(len(rest)) if len(rest) > 0 else float("nan")

        rows.append({
            "review_frac": float(review.mean()),
            "review_n": int(k),
            "precision_top": precision_top,
            "capture_rate": capture_rate,
            "residual_risk": residual_risk,
        })

    return pd.DataFrame(rows)


def plot_pr_curves(
    y: np.ndarray,
    score_dict: Dict[str, np.ndarray],
    out_png: Path,
    title: str,
    mode: str = "per_score",
    add_details: bool = True,
    min_points: int = 50,
) -> bool:
    """
    Plot PR curves for multiple scores.

    Parameters
    ----------
    mode:
      - 'per_score'    : each score uses its own finite subset (best for debugging/coverage)
      - 'intersection' : all scores evaluated on a common subset (intersection of finite values)

    add_details:
      If True, legend includes AP and n_used.

    Returns
    -------
    bool
      True if a figure was written, False otherwise.
    """
    y = np.asarray(y).astype(int)
    if len(y) == 0 or not score_dict:
        return False

    styles = ["-", "--", "-.", ":"]
    # For PR curves markers usually clutter; keep None.
    # (We avoid explicit colors; matplotlib handles them.)

    def _safe_ap(yy: np.ndarray, ss: np.ndarray) -> float:
        yy = np.asarray(yy).astype(int)
        ss = np.asarray(ss, dtype=float)
        if len(yy) < 2 or len(np.unique(yy)) < 2:
            return float("nan")
        try:
            return float(average_precision_score(yy, ss))
        except Exception:
            return float("nan")

    plt.figure(figsize=(6.2, 4.6))

    plotted = 0

    if mode == "intersection":
        # Intersection mask across all valid (length-matching) scores
        mask = np.ones(len(y), dtype=bool)
        valid_scores: List[Tuple[str, np.ndarray]] = []

        for name, sc in score_dict.items():
            sc = np.asarray(sc, dtype=float)
            if len(sc) != len(y):
                print(f"[WARN] plot_pr_curves(intersection): '{name}' len={len(sc)} != len(y)={len(y)}; skipping.")
                continue
            valid_scores.append((name, sc))
            mask &= np.isfinite(sc)

        if len(valid_scores) == 0:
            plt.close()
            return False

        yy = y[mask]
        if len(yy) < max(2, int(min_points)) or len(np.unique(yy)) < 2:
            print(f"[WARN] plot_pr_curves(intersection): not enough usable points for '{title}' (n={len(yy)}).")
            plt.close()
            return False

        base_rate = float(np.mean(yy))
        for i, (name, sc) in enumerate(valid_scores):
            ss = sc[mask]
            prec, rec, _ = precision_recall_curve(yy, ss)
            ap = _safe_ap(yy, ss)
            label = name
            if add_details:
                label = f"{name} (AP={ap:.3f}, n={int(len(yy))})"
            ls = styles[i % len(styles)]
            plt.plot(rec, prec, linestyle=ls, label=label)
            plotted += 1

        if np.isfinite(base_rate):
            plt.hlines(
                base_rate, 0.0, 1.0,
                linestyles="dashed", linewidth=1.0,
                label=f"base rate={base_rate:.3f} (common)"
            )

    elif mode == "per_score":
        # Per-score finite subset (most robust; avoids missing-curve confusion).
        base_rate = float(np.mean(y)) if len(y) else float("nan")
        if np.isfinite(base_rate):
            plt.hlines(
                base_rate, 0.0, 1.0,
                linestyles="dashed", linewidth=1.0,
                label=f"base rate={base_rate:.3f} (all)"
            )

        for i, (name, sc) in enumerate(score_dict.items()):
            sc = np.asarray(sc, dtype=float)
            if len(sc) != len(y):
                print(f"[WARN] plot_pr_curves(per_score): '{name}' len={len(sc)} != len(y)={len(y)}; skipping.")
                continue
            m = np.isfinite(sc)
            yy = y[m]
            ss = sc[m]
            if len(yy) < max(2, int(min_points)) or len(np.unique(yy)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(yy, ss)
            ap = _safe_ap(yy, ss)
            label = name
            if add_details:
                label = f"{name} (AP={ap:.3f}, n={int(m.sum())})"
            ls = styles[i % len(styles)]
            plt.plot(rec, prec, linestyle=ls, label=label)
            plotted += 1

    else:
        plt.close()
        raise ValueError("plot_pr_curves: mode must be 'per_score' or 'intersection'")

    if plotted == 0:
        plt.close()
        return False

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    # Many labels -> smaller font helps for paper screenshots too.
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()
    return True




def plot_budget_risk(
    curves: Dict[str, pd.DataFrame],
    out_png: Path,
    title: str,
    base_rate: Optional[float] = None,
) -> bool:
    """
    Plot residual risk vs review fraction.

    base_rate:
      If provided, draws a dashed horizontal line (no-review baseline).
    """
    if not curves:
        return False

    styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "P", "X", ">", "<"]

    plt.figure(figsize=(6.2, 4.6))
    plotted = 0
    xmax = 0.0

    for i, (name, dfc) in enumerate(curves.items()):
        if dfc is None or dfc.empty:
            continue
        if ("review_frac" not in dfc.columns) or ("residual_risk" not in dfc.columns):
            continue

        dfp = dfc.copy().sort_values("review_frac")
        x = dfp["review_frac"].to_numpy()
        y = dfp["residual_risk"].to_numpy()

        xmax = float(max(xmax, np.nanmax(x))) if len(x) else xmax
        ls = styles[i % len(styles)]
        mk = markers[i % len(markers)]
        plt.plot(x, y, marker=mk, linestyle=ls, label=name)
        plotted += 1

    if base_rate is not None and np.isfinite(base_rate) and plotted > 0:
        plt.hlines(
            float(base_rate), 0.0, max(0.0, xmax),
            linestyles="dashed", linewidth=1.0,
            label=f"base rate={float(base_rate):.3f}"
        )

    if plotted == 0:
        plt.close()
        return False

    plt.xlabel("Review fraction")
    plt.ylabel("Residual risk (error rate among NOT-reviewed)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()
    return True



def plot_budget_capture(
    curves: Dict[str, pd.DataFrame],
    out_png: Path,
    title: str,
) -> bool:
    """
    Plot capture rate vs review fraction.

    Adds a dashed diagonal y=x baseline (random review expectation).
    """
    if not curves:
        return False

    styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "v", "D", "P", "X", ">", "<"]

    plt.figure(figsize=(6.2, 4.6))
    plotted = 0
    xmax = 0.0

    for i, (name, dfc) in enumerate(curves.items()):
        if dfc is None or dfc.empty:
            continue
        if ("review_frac" not in dfc.columns) or ("capture_rate" not in dfc.columns):
            continue

        dfp = dfc.copy().sort_values("review_frac")
        x = dfp["review_frac"].to_numpy()
        y = dfp["capture_rate"].to_numpy()

        xmax = float(max(xmax, np.nanmax(x))) if len(x) else xmax
        ls = styles[i % len(styles)]
        mk = markers[i % len(markers)]
        plt.plot(x, y, marker=mk, linestyle=ls, label=name)
        plotted += 1

    # Random baseline: expected capture ~= review_frac
    if plotted > 0 and xmax > 0:
        xg = np.linspace(0.0, float(xmax), 50)
        plt.plot(xg, xg, linestyle="dashed", linewidth=1.0, label="random baseline (capture=review_frac)")

    if plotted == 0:
        plt.close()
        return False

    plt.xlabel("Review fraction")
    plt.ylabel("Capture rate (share of all errors caught in reviewed set)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()
    return True




def run_bspc_error_triage(args):
    pp_dirs = [p.strip() for p in str(args.pp_dirs).split(",") if p.strip()]
    if not pp_dirs:
        raise ValueError("In --mode bspc_error you must pass --pp_dirs (comma-separated).")

    cohort_names = [c.strip() for c in str(args.cohort_names).split(",") if c.strip()]
    if cohort_names and len(cohort_names) != len(pp_dirs):
        raise ValueError("--cohort_names must have the same number of entries as --pp_dirs (or be empty).")
    if not cohort_names:
        cohort_names = [Path(p).name for p in pp_dirs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fracs = _parse_fracs(args.budget_fracs)
    if not fracs:
        fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

    summary_rows = []
    md_lines = []
    md_lines.append("# BSPC error-triage summary")
    md_lines.append("")
    md_lines.append(f"- bootstraps: {int(args.bootstrap)}")
    md_lines.append(f"- budgets: {', '.join([str(f) for f in fracs])}")
    md_lines.append("")
    plots_only = bool(getattr(args, "plots_only", False))
    if plots_only:
        print("[INFO] plots_only=True: regeneruję tylko wykresy (bez bootstrap CI) i NIE nadpisuję CSV w --out_dir.")
    

    for cohort_name, pp_dir in zip(cohort_names, pp_dirs):
        df = load_pp_cohort(pp_dir)
        if args.label_col not in df.columns:
            raise ValueError(f"Label column '{args.label_col}' not found in {pp_dir}. Available: {list(df.columns)}")

        y = _to_int01(df[args.label_col]).values
        base_rate = float(np.mean(y)) if len(y) else float("nan")
        n = int(len(y))
        n_pos = int(np.sum(y))


        # --- record / cluster diagnostics (paper-risk mitigation) ---
        if "record" in df.columns:
            try:
                vc = df["record"].astype(str).value_counts()
                n_records = int(vc.size)
                top_record = str(vc.index[0]) if n_records > 0 else ""
                top_share = float(vc.iloc[0] / max(n, 1)) if n_records > 0 else float("nan")
                # Save record mix to disk for auditing
                rec_df = vc.rename("n").reset_index().rename(columns={"index": "record"})
                rec_df.to_csv(out_dir / f"{cohort_name}__record_counts.csv", index=False)
            except Exception:
                n_records, top_record, top_share = None, "", float("nan")
        else:
            n_records, top_record, top_share = None, "", float("nan")

        # Bootstrap settings (default iid; recommend stratified for rare positives).
        bootstrap_kind = getattr(args, "bootstrap_kind", "iid")
        cluster_ids = None
        if bootstrap_kind == "cluster_record":
            col = getattr(args, "bootstrap_cluster_col", "record")
            if col in df.columns:
                cluster_ids = df[col].values
            else:
                print(f"[WARN] bootstrap_kind=cluster_record but column '{col}' not found; falling back to iid.")
                bootstrap_kind = "iid"        

        # Candidate scores (higher = more risky)
        scores = {}

        def _maybe_add_score(name: str, arr: np.ndarray, allow_constant: bool = False):
            arr = np.asarray(arr, dtype=float)
            finite = arr[np.isfinite(arr)]
            # must have enough signal and not be constant
            if finite.size < 10:
                return
            if (not allow_constant) and float(np.nanstd(finite)) < 1e-12:
                return
            scores[name] = arr

        if "pmax" in df.columns:
            _maybe_add_score("1-pmax", 1.0 - pd.to_numeric(df["pmax"], errors="coerce").values, allow_constant=True)

        if "pmax" in df.columns:
            # Optional but fair: review by highest pmax (overconfidence ranking)
            _maybe_add_score("Overconf:pmax", pd.to_numeric(df["pmax"], errors="coerce").values, allow_constant=True)

        # Fair silent-failure baseline as a *ranking* (complements the two-sided *policy* curve below):
        # score is high near both extremes of binary pmax in [0.5, 1.0], i.e., near 0.5 (uncertain)
        # and near 1.0 (overconfident). Midpoint is 0.75.
        if "pmax" in df.columns:
            pmax_vals = pd.to_numeric(df["pmax"], errors="coerce").values
            _maybe_add_score("TwoSidedRank:|pmax-0.75|", np.abs(pmax_vals - 0.75), allow_constant=True)

        if "margin" in df.columns:
            _maybe_add_score("1-margin", 1.0 - pd.to_numeric(df["margin"], errors="coerce").values, allow_constant=True)


        # -----------------------------
        # Puiseux geometric proxies
        # -----------------------------
        # IMPORTANT (BSPC/ECG error triage):
        # For label=is_error we empirically see higher risk for *larger* r_dom (flat-but-wrong / silent failures),
        # so we must include r_dom in that direction.
        if "r_dom" in df.columns:
            rdom = pd.to_numeric(df["r_dom"], errors="coerce").values
            _maybe_add_score("Puiseux:r_dom", rdom)
            # keep inverse as directionality control / fragility proxy
            _maybe_add_score("Puiseux:inv_r_dom", 1.0 / (rdom + 1e-12))


        # |c4| – duże |c4| => mniejsze r_dom (bo r_dom ~ sqrt(|c2|/|c4|)) => większe ryzyko
        c2 = None
        c4 = None
        if "c2_max_abs" in df.columns:
            c2 = np.abs(pd.to_numeric(df["c2_max_abs"], errors="coerce").values)
        if "c4_max_abs" in df.columns:
            c4 = np.abs(pd.to_numeric(df["c4_max_abs"], errors="coerce").values)
            _maybe_add_score("Puiseux:|c4|", c4)

        # Ratio aligned with r_dom: |c2|/|c4| ~ r_dom^2 (higher => riskier for is_error)
        if (c2 is not None) and (c4 is not None):
            _maybe_add_score("Puiseux:|c2|/|c4|", c2 / (c4 + 1e-12))
            # optional anti-direction control (~1/r_dom^2)
            _maybe_add_score("Puiseux:|c4|/|c2|", c4 / (c2 + 1e-12))


        # prefer fit_RMSE if both exist
        if "fit_RMSE" in df.columns:
            _maybe_add_score("LocalFit:RMSE", pd.to_numeric(df["fit_RMSE"], errors="coerce").values)
        elif "RMSE" in df.columns:
            _maybe_add_score("LocalFit:RMSE", pd.to_numeric(df["RMSE"], errors="coerce").values)

        if "saliency_grad_norm" in df.columns:
            _maybe_add_score("Grad:||∇||", pd.to_numeric(df["saliency_grad_norm"], errors="coerce").values)

        # If flip_grad is censored and constant (common in highconf cohort), it will be auto-dropped above.
        if "flip_grad" in df.columns:
            _maybe_add_score("Grad:1/flip_r", 1.0 / (pd.to_numeric(df["flip_grad"], errors="coerce").values + 1e-12))


        # -----------------------------
        # AUPRC table (bootstrap OR reuse)
        # -----------------------------
        df_ap = None

        # If plots_only: try to reuse existing AUPRC table (so we avoid expensive bootstraps).
        ap_path = out_dir / f"{cohort_name}__auprc.csv"
        if plots_only and ap_path.is_file():
            try:
                df_ap = pd.read_csv(ap_path)
                # minimal sanity
                if not {"score", "auprc"}.issubset(df_ap.columns):
                    print(f"[WARN] plots_only: {ap_path} missing required columns; will compute point-estimates.")
                    df_ap = None
            except Exception as e:
                print(f"[WARN] plots_only: failed to read {ap_path}: {e}")
                df_ap = None

        # If no reusable df_ap -> compute (with CI if not plots_only, without CI if plots_only)
        if df_ap is None:
            rows = []
            for sname, svals in scores.items():
                s_arr = np.asarray(svals, dtype=float)
                m_fin = np.isfinite(s_arr)
                n_used = int(m_fin.sum())
                n_pos_used = int(y[m_fin].sum()) if n_used > 0 else 0
                base_rate_used = float(np.mean(y[m_fin])) if n_used > 0 else float("nan")
                missing_frac = 1.0 - (n_used / max(n, 1))

                if n_used < 2 or len(np.unique(y[m_fin])) < 2:
                    ap = float("nan")
                    lo = float("nan")
                    hi = float("nan")
                else:
                    if plots_only:
                        # fast point estimate only
                        ap = float(average_precision_score(y[m_fin], s_arr[m_fin]))
                        lo = float("nan")
                        hi = float("nan")
                    else:
                        ap, lo, hi = auprc_with_ci(
                            y, s_arr,
                            n_boot=args.bootstrap,
                            seed=args.seed,
                            bootstrap_kind=bootstrap_kind,
                            cluster_ids=cluster_ids,
                        )

                rows.append({
                    "cohort": cohort_name,
                    "score": sname,
                    "auprc": ap,
                    "auprc_lo95": lo,
                    "auprc_hi95": hi,
                    "base_rate": base_rate,
                    "n": n,
                    "n_pos": n_pos,
                    "n_used": n_used,
                    "n_pos_used": n_pos_used,
                    "base_rate_used": base_rate_used,
                    "missing_frac": missing_frac,
                })

            if len(rows) == 0:
                df_ap = pd.DataFrame([{
                    "cohort": cohort_name,
                    "score": "NO_USABLE_SCORES",
                    "auprc": float("nan"),
                    "auprc_lo95": float("nan"),
                    "auprc_hi95": float("nan"),
                    "base_rate": base_rate,
                    "n": n,
                    "n_pos": n_pos,
                    "n_used": 0,
                    "n_pos_used": 0,
                    "base_rate_used": float("nan"),
                    "missing_frac": 1.0,
                }])
            else:
                df_ap = pd.DataFrame(rows)
                df_ap["auprc"] = pd.to_numeric(df_ap["auprc"], errors="coerce")
                df_ap = df_ap.sort_values("auprc", ascending=False)

        # --- deltas vs confidence baselines ---
        conf_candidates = ["1-margin", "1-pmax", "Overconf:pmax", "TwoSidedRank:|pmax-0.75|"]
        conf_sub = df_ap[df_ap["score"].isin(conf_candidates)].copy()

        if len(conf_sub):
            conf_sub["auprc"] = pd.to_numeric(conf_sub["auprc"], errors="coerce")
            i_best = conf_sub["auprc"].idxmax()
            best_conf_name = str(conf_sub.loc[i_best, "score"])
            best_conf_auprc = float(conf_sub.loc[i_best, "auprc"])
        else:
            best_conf_name = ""
            best_conf_auprc = float("nan")

        if (df_ap["score"] == "1-margin").any():
            base_1margin = float(df_ap.loc[df_ap["score"] == "1-margin", "auprc"].iloc[0])
        else:
            base_1margin = float("nan")

        df_ap["best_conf_name"] = best_conf_name
        df_ap["best_conf_auprc"] = best_conf_auprc
        df_ap["delta_vs_best_conf"] = pd.to_numeric(df_ap["auprc"], errors="coerce") - best_conf_auprc
        df_ap["delta_vs_1-margin"] = pd.to_numeric(df_ap["auprc"], errors="coerce") - base_1margin

        # --- paired-bootstrap CIs for deltas (skip in plots_only) ---
        df_ap["delta_vs_best_conf_lo95"] = float("nan")
        df_ap["delta_vs_best_conf_hi95"] = float("nan")
        df_ap["delta_vs_best_conf_n"] = 0
        df_ap["delta_vs_1-margin_lo95"] = float("nan")
        df_ap["delta_vs_1-margin_hi95"] = float("nan")
        df_ap["delta_vs_1-margin_n"] = 0

        if (not plots_only) and best_conf_name and best_conf_name in scores:
            base_sc = scores[best_conf_name]
            for sname, svals in scores.items():
                dpt, dlo, dhi, dn_used = delta_auprc_with_ci(
                    y, svals, base_sc,
                    n_boot=args.bootstrap,
                    seed=args.seed,
                    bootstrap_kind=bootstrap_kind,
                    cluster_ids=cluster_ids,
                )
                df_ap.loc[df_ap["score"] == sname, "delta_vs_best_conf"] = dpt
                df_ap.loc[df_ap["score"] == sname, "delta_vs_best_conf_lo95"] = dlo
                df_ap.loc[df_ap["score"] == sname, "delta_vs_best_conf_hi95"] = dhi
                df_ap.loc[df_ap["score"] == sname, "delta_vs_best_conf_n"] = int(dn_used)

        if (not plots_only) and ("1-margin" in scores):
            base_sc = scores["1-margin"]
            for sname, svals in scores.items():
                dpt, dlo, dhi, dn_used = delta_auprc_with_ci(
                    y, svals, base_sc,
                    n_boot=args.bootstrap,
                    seed=args.seed,
                    bootstrap_kind=bootstrap_kind,
                    cluster_ids=cluster_ids,
                )
                df_ap.loc[df_ap["score"] == sname, "delta_vs_1-margin"] = dpt
                df_ap.loc[df_ap["score"] == sname, "delta_vs_1-margin_lo95"] = dlo
                df_ap.loc[df_ap["score"] == sname, "delta_vs_1-margin_hi95"] = dhi
                df_ap.loc[df_ap["score"] == sname, "delta_vs_1-margin_n"] = int(dn_used)

        # Save AUPRC table only in full mode (avoid overwriting in plots_only)
        if not plots_only:
            df_ap.to_csv(out_dir / f"{cohort_name}__auprc.csv", index=False)


        # Budget curves
        curves = {}
        for sname, svals in scores.items():
            curves[sname] = budget_curve(y, svals, fracs)
            if not curves[sname].empty:
                curves[sname].insert(0, "score", sname)
                curves[sname].insert(0, "cohort", cohort_name)
        # Two-sided confidence baseline (policy; not a simple ranking score)
        if "pmax" in df.columns:
            pmax_vals = pd.to_numeric(df["pmax"], errors="coerce").values
            df_two = two_sided_confidence_budget_curve(y, pmax_vals, fracs)
            if df_two is not None and not df_two.empty:
                df_two.insert(0, "score", "TwoSidedConf:pmax")
                df_two.insert(0, "cohort", cohort_name)
                curves["TwoSidedConf:pmax"] = df_two

        df_budget = pd.concat([c for c in curves.values() if c is not None and not c.empty], ignore_index=True) if any((c is not None and not c.empty) for c in curves.values()) else pd.DataFrame()
        if (not plots_only) and (not df_budget.empty):
            df_budget.to_csv(out_dir / f"{cohort_name}__budget_curve.csv", index=False)


        # -----------------------------
        # Plots (paper-ready, less confusing)
        # -----------------------------
        # 1) we keep a small set -> avoids "missing curve" illusion due to exact overlaps (e.g., r_dom vs |c2|/|c4|)
        # 2) main PR plot uses per_score mode (always shows every method)
        # 3) we also write a common-set PR plot for fairness

        # Build missingness map (fallback-safe)
        miss_map = {}
        if "missing_frac" in df_ap.columns:
            try:
                miss_map = (df_ap.set_index("score")["missing_frac"]).to_dict()
            except Exception:
                miss_map = {}

        def _miss(sname: str) -> float:
            v = miss_map.get(sname, np.nan)
            try:
                return float(v)
            except Exception:
                return float("nan")

        # Choose a compact set of curves to show
        plot_keys: List[str] = []

        # Always show best confidence baseline for this cohort (if available)
        if best_conf_name and (best_conf_name in scores):
            plot_keys.append(best_conf_name)

        # Classic confidence baselines (avoid duplicates)
        for k in ["1-margin", "1-pmax", "TwoSidedRank:|pmax-0.75|", "Overconf:pmax"]:
            if k in scores and k not in plot_keys:
                plot_keys.append(k)

        # Puiseux: show the main hypothesis + direction control
        for k in ["Puiseux:r_dom", "Puiseux:inv_r_dom", "Puiseux:|c4|"]:
            if k in scores and k not in plot_keys:
                plot_keys.append(k)

        # Optional extras only if they have decent coverage (so they don't ruin common-set plots)
        for k in ["LocalFit:RMSE", "Grad:||∇||"]:
            if k in scores and k not in plot_keys:
                mf = _miss(k)
                if (not np.isfinite(mf)) or (mf <= 0.25):
                    plot_keys.append(k)

        # Hard cap to keep the plot readable
        if len(plot_keys) > 8:
            plot_keys = plot_keys[:8]

        pr_scores = {k: scores[k] for k in plot_keys if k in scores}

        # PR plots
        if pr_scores:
            # Per-score PR (robust; fixes the "missing curve" complaint)
            plot_pr_curves(
                y,
                pr_scores,
                out_dir / "fig" / f"pr__{cohort_name}.png",
                title=f"PR (error triage): {cohort_name}",
                mode="per_score",
                add_details=True,
                min_points=50,
            )

            # Common-set PR (fair comparison)
            plot_pr_curves(
                y,
                pr_scores,
                out_dir / "fig" / f"pr__{cohort_name}__common.png",
                title=f"PR (common eval set): {cohort_name}",
                mode="intersection",
                add_details=True,
                min_points=50,
            )

            # Budget plots on a common set (if possible)
            # Use intersection across the plotted scores (excluding very-missing ones)
            keys_common = []
            for k in plot_keys:
                if k not in pr_scores:
                    continue
                mf = _miss(k)
                if (not np.isfinite(mf)) or (mf <= 0.25):
                    keys_common.append(k)

            mask_common = np.ones(len(y), dtype=bool)
            for k in keys_common:
                arr = np.asarray(pr_scores[k], dtype=float)
                if len(arr) != len(y):
                    continue
                mask_common &= np.isfinite(arr)

            plot_curves_common = {}
            if int(mask_common.sum()) >= 50 and len(np.unique(y[mask_common])) >= 2:
                y_common = y[mask_common]
                base_rate_common = float(np.mean(y_common))

                for k in keys_common:
                    arr = np.asarray(pr_scores[k], dtype=float)
                    plot_curves_common[k] = budget_curve(y_common, arr[mask_common], fracs)

                # Two-sided confidence policy baseline on the same common set (requires pmax)
                if "pmax" in df.columns:
                    pmax_vals = pd.to_numeric(df["pmax"], errors="coerce").values
                    mask_pmax = mask_common & np.isfinite(pmax_vals)
                    if int(mask_pmax.sum()) >= 50 and len(np.unique(y[mask_pmax])) >= 2:
                        plot_curves_common["TwoSidedConf:pmax"] = two_sided_confidence_budget_curve(
                            y[mask_pmax], pmax_vals[mask_pmax], fracs
                        )

                if plot_curves_common:
                    plot_budget_risk(
                        plot_curves_common,
                        out_dir / "fig" / f"risk__{cohort_name}.png",
                        title=f"Residual risk vs review (common set): {cohort_name}",
                        base_rate=base_rate_common,
                    )
                    plot_budget_capture(
                        plot_curves_common,
                        out_dir / "fig" / f"capture__{cohort_name}.png",
                        title=f"Capture rate vs review (common set): {cohort_name}",
                    )
            else:
                print(f"[WARN] Common-set mask too small for budget plots in cohort '{cohort_name}'.")



        # Optional: record-level AUROC diagnostic (esp. record108 silent failures)
        rec_diag = ""
        if (not plots_only) and ("record" in df.columns) and (df["record"].astype(str) == "108").any():
            df108 = df[df["record"].astype(str) == "108"].copy()
            y108 = _to_int01(df108[args.label_col]).values
            if len(np.unique(y108)) >= 2 and ("r_dom" in df108.columns):
                r108 = pd.to_numeric(df108["r_dom"], errors="coerce").values
                s108 = 1.0 / (r108 + 1e-12)  # inv_r_dom: większe = bardziej ryzykowne
                auc, lo, hi = auroc_with_ci(
                    y108, s108,
                    n_boot=max(500, int(args.bootstrap // 2)),
                    seed=args.seed
                )
                rec_diag = (
                    f"record108 AUROC(Puiseux:inv_r_dom)={auc:.3f} "
                    f"[{lo:.3f},{hi:.3f}] (n={len(y108)}, pos={int(y108.sum())})"
                )


        # Markdown summary line
        best_row = df_ap.iloc[0].to_dict() if not df_ap.empty else {}
        md_lines.append(f"## {cohort_name}")
        md_lines.append(f"- n={n}, positives={n_pos}, base_rate={base_rate:.3f}")
        if n_records is not None:
            md_lines.append(f"- records={n_records}, top_record={top_record}, top_share={top_share:.2f}")

        if best_conf_name:
            md_lines.append(f"- best confidence baseline: {best_conf_name} = {best_conf_auprc:.3f}")

        if best_row:
            try:
                md_lines.append(
                    f"- best AUPRC: {best_row['score']} = {float(best_row['auprc']):.3f} "
                    f"[{float(best_row['auprc_lo95']):.3f},{float(best_row['auprc_hi95']):.3f}] "
                    f"(n_used={int(best_row.get('n_used', n))}, pos_used={int(best_row.get('n_pos_used', n_pos))})"
                )
            except Exception:
                md_lines.append(f"- best AUPRC: {best_row.get('score','?')} = {best_row.get('auprc')}")

            # Δ vs best confidence baseline (if available)
            if best_conf_name:
                try:
                    d = float(best_row.get("delta_vs_best_conf", np.nan))
                    dlo = float(best_row.get("delta_vs_best_conf_lo95", np.nan))
                    dhi = float(best_row.get("delta_vs_best_conf_hi95", np.nan))
                    dn = int(best_row.get("delta_vs_best_conf_n", 0))
                    if np.isfinite(d):
                        md_lines.append(f"- ΔAUPRC(best - {best_conf_name}) = {d:.3f} [{dlo:.3f},{dhi:.3f}] (paired_n={dn})")
                except Exception:
                    pass

        if rec_diag:
            md_lines.append(f"- {rec_diag}")
        md_lines.append("")


        summary_rows.extend(df_ap.to_dict(orient="records"))

    # Combined CSV + markdown
    if not plots_only:
        pd.DataFrame(summary_rows).to_csv(out_dir / "all_cohorts__auprc.csv", index=False)
        (out_dir / "paper_summary_bspc.md").write_text("\n".join(md_lines), encoding="utf-8")
        print(f"[DONE] Wrote BSPC analysis to: {out_dir}")
    else:
        print(f"[DONE] Regenerated figures (plots_only) in: {out_dir / 'fig'}")



    print(f"[DONE] Wrote BSPC analysis to: {out_dir}")


# ---------- Main ----------
def main():
    """
    Orchestrate loading of inputs, joining evidence, running triage analyses,
    computing PR/AUPRC summaries, writing figures and markdown, and exporting
    compact CSVs for downstream use.
    """
    args = parse_args()

    if getattr(args, "mode", "classic") == "bspc_error":
        run_bspc_error_triage(args)
        return

    EPS = 1e-9

    BUDGET = float(args.radius_budget)
    BUDGET_SWEEP = sorted(set(_parse_float_list(args.radius_budget_sweep)))
    if BUDGET not in BUDGET_SWEEP:
        BUDGET_SWEEP.append(BUDGET)
    BUDGET_SWEEP = sorted(BUDGET_SWEEP)

    BASE = Path(__file__).resolve().parent  # build_np_evidence/

    # Auto-detect REAL vs RADIO (prefer *_real if it exists, else *_radio).
    UP_DIR = (BASE.parent / "up_real") if (BASE.parent / "up_real").exists() else (BASE.parent / "up_radio")
    PP_DIR = (BASE.parent / "post_processing_real") if (BASE.parent / "post_processing_real").exists() else (BASE.parent / "post_processing_radio")
    OUT_DIR = BASE
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using UP_DIR = {UP_DIR.name}, PP_DIR = {PP_DIR.name}")

    # 1) Load sources.
    df_anchors = load_anchors(UP_DIR)
    df_bench = collect_benchmarks(PP_DIR)
    df_dom = load_dom_ratio(PP_DIR)

    df_sens_det, df_sens_mul, sens_txt = load_sensitivity_outputs(PP_DIR)

    # Copy the budget-policy table to NP-analysis outputs for paper packaging
    if df_sens_mul is not None and len(df_sens_mul):
        out_pol = OUT_DIR / "triage_policy_multi_budget.csv"
        df_sens_mul.to_csv(out_pol, index=False)
        print(f"[INFO] Saved triage policy table -> {out_pol}")

        # Figure: risk_accept vs abstain with budget annotations
        if {"abstain", "risk_accept", "budget_frac"}.issubset(df_sens_mul.columns):
            plt.figure(figsize=(6, 5))
            plt.plot(df_sens_mul["abstain"], df_sens_mul["risk_accept"], marker="o")
            for _, r in df_sens_mul.iterrows():
                try:
                    plt.text(float(r["abstain"]), float(r["risk_accept"]),
                            f"{float(r['budget_frac']):.2f}", fontsize=8)
                except Exception:
                    pass
            plt.xlabel("Abstain fraction (review load)")
            plt.ylabel("Risk on accept set (risk_accept)")
            plt.title("Triage policy: risk vs review budget (labels = budget_frac)")
            plt.tight_layout()
            figp = OUT_DIR / "figures" / "triage_risk_vs_abstain_multi.png"
            plt.savefig(figp, dpi=160)
            plt.close()
            print(f"[INFO] Saved figure -> {figp}")

    # Optional: Pareto-front from detailed grid (risk vs abstain)
    if df_sens_det is not None and len(df_sens_det):
        if {"abstain_exact", "risk_accept_exact"}.issubset(df_sens_det.columns):
            tmp = df_sens_det[["abstain_exact", "risk_accept_exact"]].dropna().sort_values("abstain_exact")
            if len(tmp):
                # Pareto-ish: cumulative min risk as abstain increases
                tmp["risk_cummin"] = tmp["risk_accept_exact"].cummin()
                out_pf = OUT_DIR / "sensitivity_pareto_front.csv"
                tmp[["abstain_exact", "risk_cummin"]].drop_duplicates().to_csv(out_pf, index=False)

                plt.figure(figsize=(6, 5))
                plt.plot(tmp["abstain_exact"], tmp["risk_cummin"])
                plt.xlabel("Abstain (exact)")
                plt.ylabel("Best achievable risk_accept (cummin over grid)")
                plt.title("Sensitivity grid: Pareto-style front (risk vs abstain)")
                plt.tight_layout()
                figpf = OUT_DIR / "figures" / "sensitivity_pareto_risk_vs_abstain.png"
                plt.savefig(figpf, dpi=160)
                plt.close()
                print(f"[INFO] Saved Pareto front -> {out_pf}")
                print(f"[INFO] Saved figure -> {figpf}")


    # Sanity check: are TXT points indexed consistently with anchors?
    if not df_bench.empty and df_bench["point"].max() >= len(df_anchors):
        print("[WARN] TXT points index > #anchors; check enumeration 0..N-1!")

    # 2) Join on 'point'.
    df = df_anchors.merge(df_bench, on="point", how="left", suffixes=("", "_bench"))
    if not df_dom.empty:
        before = df["point"].nunique()
        df = df.merge(df_dom, on="point", how="left", suffixes=("", "_dom"))

        # Prefer dom-ratio values if present; fall back to TXT-derived ones
        if "r_dom_dom" in df.columns:
            df["r_dom"] = df["r_dom_dom"].combine_first(df.get("r_dom"))
        if "r_flip_obs_dom" in df.columns:
            df["r_flip_obs"] = df["r_flip_obs_dom"].combine_first(df.get("r_flip_obs"))

        # If dom provides flip_* / grad_norm / frac_kink, use them only as fill-ins
        for col in ["flip_grad", "flip_lime", "flip_shap", "saliency_grad_norm", "frac_kink"]:
            dom_col = f"{col}_dom"
            if dom_col in df.columns:
                if col in df.columns:
                    df[col] = df[col].combine_first(df[dom_col])
                else:
                    df[col] = df[dom_col]

        matched = int(pd.to_numeric(df.get("r_dom"), errors="coerce").notna().sum())
        print(f"[INFO] Dominant-ratio matched rows: {matched}/{before}")
    else:
        if "r_dom_pred" in df.columns:
            df["r_dom"] = df["r_dom_pred"]


    # 3) Observed flip radius.
    if "r_flip" in df.columns:
        df["r_flip_obs"] = df["r_flip"]
    if "r_flip_obs" not in df.columns or df["r_flip_obs"].isna().all():
        if "min_flip_radius" in df.columns:
            df["r_flip_obs"] = df["min_flip_radius"]


    # Effective radius (respects censoring when flip was not found within r_max).
    if "r_flip_eff" not in df.columns:
        if "flip_found" in df.columns and "r_flip_cens" in df.columns:
            ff = pd.to_numeric(df["flip_found"], errors="coerce").fillna(0).astype(int)
            r_obs = pd.to_numeric(df["r_flip_obs"], errors="coerce") if "r_flip_obs" in df.columns else pd.Series(np.nan, index=df.index)
            r_cens = pd.to_numeric(df["r_flip_cens"], errors="coerce")
            df["r_flip_eff"] = np.where(ff.values == 1, r_obs.values, r_cens.values)
        elif "r_flip_obs" in df.columns:
            df["r_flip_eff"] = pd.to_numeric(df["r_flip_obs"], errors="coerce")            
            

    def _hit_rate_col(df, name, BUDGET=0.02):
        """Share of anchors with r <= BUDGET for a given radius-like column."""
        if name not in df.columns:
            return float("nan")
        s = pd.to_numeric(df[name], errors="coerce")
        hits = (s <= BUDGET).fillna(False).sum()
        n = int(df["point"].nunique())
        return float(hits) / max(n, 1)

    def _col(df, name):
        """Convenience accessor that returns a float Series or empty Series."""
        return df[name] if name in df.columns else pd.Series(dtype=float)

    summary = {
        "n_anchors": df["point"].nunique(),
        "hit_puiseux": _hit_rate_col(df, "r_flip_obs"),
        "hit_grad":    _hit_rate_col(df, "flip_grad"),
        "hit_lime":    _hit_rate_col(df, "flip_lime"),
        "hit_shap":    _hit_rate_col(df, "flip_shap"),
        "med_r_puiseux": float(pd.to_numeric(_col(df, "r_flip_obs"), errors="coerce").dropna().median()) if "r_flip_obs" in df else float("nan"),
        "med_r_grad":    float(pd.to_numeric(_col(df, "flip_grad"), errors="coerce").dropna().median()) if "flip_grad" in df.columns else float("nan"),
        "med_r_lime":    float(pd.to_numeric(_col(df, "flip_lime"), errors="coerce").dropna().median()) if "flip_lime" in df.columns else float("nan"),
        "med_r_shap":    float(pd.to_numeric(_col(df, "flip_shap"), errors="coerce").dropna().median()) if "flip_shap" in df.columns else float("nan"),
    }

    pd.DataFrame([summary]).to_csv(OUT_DIR / "xai_vs_puiseux_summary.csv", index=False)
    print("[INFO] Saved head-to-head flip summary ->", OUT_DIR / "xai_vs_puiseux_summary.csv")

    # 4) Compute r_dom if missing but c2/c4 available.
    if {"r_dom", "c2_max_abs", "c4_max_abs"}.issubset(df.columns):
        mask = df["r_dom"].isna() & df["c2_max_abs"].notna() & df["c4_max_abs"].notna() & (df["c4_max_abs"] > 0)
        if mask.any():
            df.loc[mask, "r_dom"] = np.sqrt(df.loc[mask, "c2_max_abs"] / df.loc[mask, "c4_max_abs"])

    # 5) Save joined anchors.
    out_csv = OUT_DIR / "evidence_anchors_joined.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved joined anchors -> {out_csv}")

    # 6) Correlations and summary stats.
    sub = df[["r_dom", "r_flip_obs"]].dropna()
    if len(sub) >= 2:
        pear = float(sub.corr(method="pearson").iloc[0, 1])
        spear = float(sub.corr(method="spearman").iloc[0, 1])
        mae = float(np.mean(np.abs(sub["r_dom"] - sub["r_flip_obs"])))
    else:
        pear = spear = mae = float("nan")

    # Full triage curve by |c4| → PR curve + AUPRC.
    auprc_c4 = np.nan
    pr_at_topk = re_at_topk = np.nan
    f1_max_c4 = np.nan
    thr_at_f1max = np.nan

    y_true = None

    if {"c4_max_abs", "r_flip_eff"}.issubset(df.columns) or {"c4_max_abs", "r_flip_obs"}.issubset(df.columns):
        # Label: fragile=1 if a flip is observed within the budget (r <= BUDGET).
        # Missing flips (NaN) are treated as 0.
        r_for_label = df["r_flip_eff"] if "r_flip_eff" in df.columns else df["r_flip_obs"]
        y_true = ((pd.to_numeric(r_for_label, errors="coerce").fillna(np.inf)) <= BUDGET).astype(int).to_numpy()
        scores = pd.to_numeric(df["c4_max_abs"], errors="coerce").to_numpy()

        if np.isfinite(scores).any() and y_true.sum() > 0:
            rec, prec, auprc_c4, df_curve = precision_recall_from_scores(y_true, scores)

            # F1 along the curve (skip the initial (1, 0) point).
            with np.errstate(divide="ignore", invalid="ignore"):
                f1 = 2 * prec[1:] * rec[1:] / np.clip(prec[1:] + rec[1:], 1e-12, None)
            if len(f1) > 0:
                imax = int(np.nanargmax(f1))
                f1_max_c4 = float(f1[imax])
                thr_at_f1max = float(df_curve.loc[imax, "threshold"])

            # Legacy point: top-K where K = #positives (kept for comparability).
            K = int(y_true.sum())
            if K > 0:
                tp_at_k = int(df_curve.loc[min(K, len(df_curve)) - 1, "tp"])
                pr_at_topk = tp_at_k / K
                re_at_topk = tp_at_k / K  # at top-K with K=#pos, precision == recall

            # Save CSV and the PR plot.
            curve_csv = OUT_DIR / "pr_by_abs_c4.csv"
            df_curve.to_csv(curve_csv, index=False)

            plt.figure(figsize=(6, 5))
            plt.step(rec, prec, where="post")
            pos_rate = y_true.mean()
            plt.hlines(pos_rate, 0.0, 1.0, linestyles="--", linewidth=1.0, label=f"baseline={pos_rate:.2f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Triage by |c4|: Precision–Recall")
            plt.legend()
            pr_fig = OUT_DIR / "figures" / "pr_curve_by_abs_c4.png"
            plt.tight_layout()
            plt.savefig(pr_fig, dpi=160)
            plt.close()
            print(f"[INFO] Saved PR curve -> {pr_fig}")
            print(f"[INFO] Saved PR table -> {curve_csv}")

    def _auprc_for_score(name, scores, y_true, out_prefix):
        """
        Helper: compute AUPRC for an arbitrary score, save its PR curve (CSV + PNG),
        and return the AP value.
        """
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(scores)
        if not valid.any() or y_true.sum() == 0:
            return np.nan
        rec2, prec2, ap2, df_curve2 = precision_recall_from_scores(y_true, scores)
        # Save curve table and figure
        (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
        df_curve2.to_csv(OUT_DIR / f"pr_by_{out_prefix}.csv", index=False)
        plt.figure(figsize=(6,5))
        plt.step(rec2, prec2, where="post", label=name)
        pos_rate2 = y_true.mean()
        plt.hlines(pos_rate2, 0.0, 1.0, linestyles="--", linewidth=1.0, label=f"baseline={pos_rate2:.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR: {name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT_DIR / "figures" / f"pr_by_{out_prefix}.png", dpi=160)
        plt.close()
        return float(ap2)

    results_pr = []
    if y_true is not None and int(np.sum(y_true)) > 0:
        # Include already-computed |c4| in the comparison table.
        if np.isfinite(auprc_c4):
            results_pr.append(("|c4|", float(auprc_c4)))


        # 1/r_dom as a fragility score (smaller r_dom => more fragile)
        if "r_dom" in df.columns:
            rdom = pd.to_numeric(df["r_dom"], errors="coerce").to_numpy()
            score = 1.0 / np.where(np.isfinite(rdom), rdom + EPS, np.inf)
            ap = _auprc_for_score("1/r_dom", score, y_true, "inv_r_dom")
            results_pr.append(("1/r_dom", ap))

        # 1/r scores for radii parsed from TXT.
        for col, label, pref in [
            ("flip_grad", "1/r_grad", "per_grad"),
            ("flip_lime", "1/r_lime", "per_lime"),
            ("flip_shap", "1/r_shap", "per_shap"),
        ]:
            if col in df.columns:
                r = pd.to_numeric(df[col], errors="coerce").to_numpy()
                score = 1.0 / np.where(np.isfinite(r), r + EPS, np.inf)
                ap = _auprc_for_score(label, score, y_true, pref)
                results_pr.append((label, ap))

        # grad_norm from Saliency logs (if available).
        if "saliency_grad_norm" in df.columns:
            g = pd.to_numeric(df["saliency_grad_norm"], errors="coerce").fillna(0.0).to_numpy()
            ap = _auprc_for_score("grad_norm", g, y_true, "grad_norm")
            results_pr.append(("grad_norm", ap))

        # Save the AUPRC comparison summary.
        pd.DataFrame(results_pr, columns=["score","AUPRC"]).to_csv(
            OUT_DIR / "triage_compare_summary.csv", index=False
        )
    else:
        print("[WARN] Skipping triage_compare_summary.csv: could not build y_true (fragility labels).")


            
    def _safe_ap(y_true_vec, score_vec):
        y_true_vec = np.asarray(y_true_vec).astype(int)
        if y_true_vec.sum() == 0 or y_true_vec.sum() == len(y_true_vec):
            return np.nan
        _, _, apv, _ = precision_recall_from_scores(y_true_vec, score_vec)
        return float(apv)

    # Build score bank once
    score_bank = {}
    if "c4_max_abs" in df.columns:
        score_bank["|c4|"] = pd.to_numeric(df["c4_max_abs"], errors="coerce").to_numpy()

    if "r_dom" in df.columns:
        rdom = pd.to_numeric(df["r_dom"], errors="coerce").to_numpy()
        score_bank["1/r_dom"] = 1.0 / np.where(np.isfinite(rdom), rdom + EPS, np.inf)

    if "saliency_grad_norm" in df.columns:
        score_bank["grad_norm"] = pd.to_numeric(df["saliency_grad_norm"], errors="coerce").fillna(0.0).to_numpy()

    if "frac_kink" in df.columns:
        score_bank["frac_kink"] = pd.to_numeric(df["frac_kink"], errors="coerce").fillna(0.0).to_numpy()

    for col, lab in [("flip_grad", "1/r_grad"), ("flip_lime", "1/r_lime"), ("flip_shap", "1/r_shap")]:
        if col in df.columns:
            rr = pd.to_numeric(df[col], errors="coerce").to_numpy()
            score_bank[lab] = 1.0 / np.where(np.isfinite(rr), rr + EPS, np.inf)

    rows_sweep = []
    if ("r_flip_eff" in df.columns or "r_flip_obs" in df.columns) and len(BUDGET_SWEEP):
        rcol = df["r_flip_eff"] if "r_flip_eff" in df.columns else df["r_flip_obs"]
        rflip = pd.to_numeric(rcol, errors="coerce").fillna(np.inf).to_numpy()
        for b in BUDGET_SWEEP:
            yb = (rflip <= float(b)).astype(int)
            base = float(yb.mean())
            for name, sc in score_bank.items():
                apv = _safe_ap(yb, sc)
                rows_sweep.append({
                    "radius_budget": float(b),
                    "score": name,
                    "pos_rate": base,
                    "AUPRC": apv
                })

        df_sweep = pd.DataFrame(rows_sweep)
        out_sweep = OUT_DIR / "triage_auprc_vs_radius_budget.csv"
        df_sweep.to_csv(out_sweep, index=False)
        print(f"[INFO] Saved sweep table -> {out_sweep}")

        # Figure: AUPRC vs radius_budget for each score
        plt.figure(figsize=(7, 5))
        for name in sorted(df_sweep["score"].unique()):
            subp = df_sweep[df_sweep["score"] == name].sort_values("radius_budget")
            plt.plot(subp["radius_budget"], subp["AUPRC"], marker="o", label=name)
        plt.xlabel("Flip-radius budget")
        plt.ylabel("AUPRC (fragile=flip within budget)")
        plt.title("Triage AUPRC vs flip-radius budget")
        plt.legend()
        plt.tight_layout()
        fig_sweep = OUT_DIR / "figures" / "triage_auprc_vs_radius_budget.png"
        plt.savefig(fig_sweep, dpi=160)
        plt.close()
        print(f"[INFO] Saved figure -> {fig_sweep}")


    corr_df = pd.DataFrame({
        "anchors_total": [df["point"].nunique()],
        "anchors_with_flip": [int((pd.to_numeric(df["r_flip_obs"], errors="coerce") <= BUDGET).sum()) if "r_flip_obs" in df.columns else 0],
        "mean_r_dom": [float(np.nanmean(pd.to_numeric(df["r_dom"], errors="coerce"))) if "r_dom" in df.columns else float("nan")],
        "mean_r_flip": [float(np.nanmean(pd.to_numeric(df["r_flip_obs"], errors="coerce"))) if "r_flip_obs" in df.columns else float("nan")],
        "mae_abs(r_dom,r_flip)": [mae],
        "pearson_r": [pear],
        "spearman_r": [spear],
        "precision_topk_by_c4": [pr_at_topk],
        "recall_topk_by_c4": [re_at_topk],
        "AUPRC_by_c4": [auprc_c4],
        "F1_max_by_c4": [f1_max_c4],
        "threshold_at_F1max_by_c4": [thr_at_f1max],
    })

    corr_df.to_csv(OUT_DIR / "corr_summary.csv", index=False)
    print(f"[INFO] Saved correlation summary -> {OUT_DIR / 'corr_summary.csv'}")

    # 7) Figures
    # 7a) r_dom vs r_flip with a regression line (and y=x reference).
    plt.figure(figsize=(6, 5))
    if len(sub) > 0:
        plt.scatter(sub["r_dom"], sub["r_flip_obs"], alpha=0.85)
        # Use standard least squares for a simple slope (not constrained through origin).
        try:
            X = sub["r_dom"].to_numpy().reshape(-1,1)
            y = sub["r_flip_obs"].to_numpy()
            A = (X.T @ X + 1e-12*np.eye(1))
            b = (X.T @ y)
            coef = np.linalg.solve(A, b).reshape(-1)   # slope in coef[0]
            mmax = float(np.max([sub["r_dom"].max(), sub["r_flip_obs"].max()]))
            xg = np.linspace(0.0, mmax, 100)
            plt.plot(xg, float(coef[0]) * xg, linestyle="--", label=f"slope≈{float(coef[0]):.2f}")
            plt.legend()
        except Exception:
            pass
        mmax = float(max(sub["r_dom"].max(), sub["r_flip_obs"].max()))
        plt.plot([0, mmax], [0, mmax], color="grey", linewidth=1)
    plt.xlabel("Predicted onset radius r_dom ≈ sqrt(|c2|/|c4|)")
    plt.ylabel("Observed min flip radius")
    plt.title("Puiseux prediction vs observed flip")
    plt.tight_layout()
    fig1 = OUT_DIR / "figures" / "scatter_rdom_vs_rflip.png"
    plt.savefig(fig1, dpi=160)
    plt.close()
    print(f"[INFO] Saved {fig1}")

    # 7b) Kink fraction vs flip radius.
    if "frac_kink" in df.columns:
        s2 = df[["frac_kink", "r_flip_obs"]].dropna()
        plt.figure(figsize=(6, 5))
        if len(s2) > 0:
            plt.scatter(s2["frac_kink"], s2["r_flip_obs"], alpha=0.85)
        plt.xlabel("Kink fraction in neighborhood")
        plt.ylabel("Observed min flip radius")
        plt.title("Non-holomorphicity vs fragility")
        plt.tight_layout()
        fig2 = OUT_DIR / "figures" / "scatter_kink_vs_rflip.png"
        plt.savefig(fig2, dpi=160)
        plt.close()
        print(f"[INFO] Saved {fig2}")

    # 8) Markdown report.
    md = OUT_DIR / "np_evidence_report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Newton–Puiseux Evidence Report\n")
        f.write(f"- Anchors analyzed: **{df['point'].nunique()}**\n")
        n_flips = int((pd.to_numeric(df.get('r_flip_obs'), errors='coerce') < BUDGET).sum())
        f.write(f"- Anchors with observed class flip (r < {BUDGET:.3f}): **{n_flips}**\n")
        mean_rdom = np.nanmean(pd.to_numeric(df.get('r_dom'), errors='coerce'))
        mean_rflip = np.nanmean(pd.to_numeric(df.get('r_flip_obs'), errors='coerce'))
        f.write(f"- Mean predicted onset radius r_dom: **{mean_rdom:.6f}**\n")
        f.write(f"- Mean observed min flip radius: **{mean_rflip:.6f}**\n")
        f.write(f"- MAE |r_dom - r_flip|: **{mae if np.isfinite(mae) else float('nan'):.6f}**\n")
        f.write(f"- Pearson(r_dom, r_flip): **{pear if np.isfinite(pear) else 'nan'}**\n")
        f.write(f"- Spearman(r_dom, r_flip): **{spear if np.isfinite(spear) else 'nan'}**\n")
        if np.isfinite(pr_at_topk) and np.isfinite(re_at_topk):
            f.write(f"- Triage @top-K by |c4| → precision≈**{pr_at_topk:.2f}**, recall≈**{re_at_topk:.2f}**\n")
        f.write("\n## Key Figures\n")
        f.write("- Puiseux prediction vs observed flip: `figures/scatter_rdom_vs_rflip.png`\n")
        if (OUT_DIR / 'figures' / 'scatter_kink_vs_rflip.png').exists():
            f.write("- Kink fraction vs flip radius: `figures/scatter_kink_vs_rflip.png`\n")
        if np.isfinite(auprc_c4):
            f.write(f"- Triage by |c4| → AUPRC=**{auprc_c4:.3f}**, PR curve: `figures/pr_curve_by_abs_c4.png`\n")
    print(f"[INFO] Wrote report -> {md}")



if __name__ == "__main__":
    main()
