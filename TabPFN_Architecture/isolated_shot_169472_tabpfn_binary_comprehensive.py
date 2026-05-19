#!/usr/bin/env python3
"""
Smoothed prediction visualization for TabPFN HighPerf holdout shot 169472.

Trains (or loads) bagged TabPFN, runs inference on held-out shot 169472,
applies rolling-mean smoothing to P(ELMy), and plots identical layout to
LSTM_50_2_Shot_169472_Predictions_Smoothed.py:
  (a) Observed plasma states
  (b) Smoothed P(ELMy) with thresholded regions (twin y-axis)
  (c) Raw vs smoothed prediction accuracy

Outputs:
  Isolated Shot Comprehensive Analysis/isolated_shot_169472_tabpfn_binary_comprehensive_analysis.png
  TabPFN_Architecture/tabpfn_binary_highperf_holdout_169472.pkl

Usage:
  MPLBACKEND=Agg TabPFN_Architecture/.venv/bin/python \\
      TabPFN_Architecture/isolated_shot_169472_tabpfn_binary_comprehensive.py [data_csv]

  MPLBACKEND=Agg TabPFN_Architecture/.venv/bin/python \\
      TabPFN_Architecture/isolated_shot_169472_tabpfn_binary_comprehensive.py --skip-train

  # A100: larger inference batches (default 8192 on CUDA; override if needed):
  TABPFN_PREDICT_BATCH_SIZE=16384 MPLBACKEND=Agg ...
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import TabPFN_Center_Point_Binary_HighPerf as hp  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HOLDOUT_SHOT = 169472
PROBLEM_SHOT = hp.PROBLEM_SHOT
DEFAULT_CSV = os.path.join(_WORKSPACE, "combined_database.csv")
OUTPUT_DIR = os.path.join(_WORKSPACE, "Isolated Shot Comprehensive Analysis")
OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    f"isolated_shot_{HOLDOUT_SHOT}_tabpfn_binary_comprehensive_analysis.png",
)
BUNDLE_PATH = os.path.join(SCRIPT_DIR, f"tabpfn_binary_highperf_holdout_{HOLDOUT_SHOT}.pkl")

STATE_COLORS = {0: "#2E8B57", 1: "#DC143C"}
STATE_NAMES = {0: "Suppressed", 1: "ELMy"}

WINDOW_SIZE = hp.WINDOW_SIZE
PREDICTION_HORIZON_MS = hp.PREDICTION_HORIZON_MS
TRAIN_RATIO = hp.TRAIN_RATIO
VAL_RATIO = hp.VAL_RATIO
SHOT_SPLIT_SEED = hp.SHOT_SPLIT_SEED

SMOOTH_WINDOW_MS = float(os.environ.get("SMOOTH_WINDOW_MS", "40"))
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.3"))
VAL_TUNE_MAX_SAMPLES = int(os.environ.get("TABPFN_VAL_TUNE_MAX", "20000"))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _subsample_for_threshold(val_tab, val_y, max_n: int, seed: int = 42):
    if len(val_tab) <= max_n:
        return val_tab, val_y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(val_tab), size=max_n, replace=False)
    print(
        f"  Threshold tuning: using {max_n:,} / {len(val_tab):,} val windows "
        f"(set TABPFN_VAL_TUNE_MAX to change)"
    )
    return val_tab[idx], val_y[idx]


def _feature_list_for_csv(df: pd.DataFrame, data_path: str) -> list[str]:
    feats = list(hp.IMPORTANT_FEATURES)
    if "combined_database" in os.path.basename(data_path):
        if "fs_sum_past_max_smoothed" in df.columns:
            feats = [
                "fs_sum_past_max_smoothed" if f == "fs_sum_max_smoothed" else f
                for f in feats
            ]
    return [f for f in feats if f in df.columns]


def map_values_to_future_times(time_arr, window_end_times, values, horizon_ms):
    """Average window values that map to the same future timestep index."""
    mapped = np.full(len(time_arr), np.nan, dtype=np.float64)
    counts = np.zeros(len(time_arr), dtype=np.int32)
    for value, window_end_time in zip(values, window_end_times):
        future_time = window_end_time + horizon_ms
        time_idx = int(np.argmin(np.abs(time_arr - future_time)))
        if time_idx >= len(mapped):
            continue
        if counts[time_idx] == 0:
            mapped[time_idx] = float(value)
        else:
            mapped[time_idx] += float(value)
        counts[time_idx] += 1
    has_data = counts > 0
    mapped[has_data] /= counts[has_data]
    return mapped, has_data


def _interpolate_nans(series):
    out = series.astype(np.float64).copy()
    valid = ~np.isnan(out)
    if not valid.any() or valid.all():
        return out
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def smooth_timeline_series(series, time_arr, window_ms=SMOOTH_WINDOW_MS):
    filled = _interpolate_nans(series)
    dt = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 1.0
    if dt <= 0:
        dt = 1.0
    window_pts = max(3, int(round(window_ms / dt)))
    if window_pts % 2 == 0:
        window_pts += 1
    smoothed = (
        pd.Series(filled)
        .rolling(window=window_pts, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return smoothed, window_pts, dt


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_and_prepare_data(data_path: str, holdout_shot: int = HOLDOUT_SHOT):
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df[df["shot"] != PROBLEM_SHOT].copy()

    selected_features = _feature_list_for_csv(df, data_path)
    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    if "state_binary" in df_sorted.columns:
        state_as_float = df_sorted["state_binary"].replace(-1, np.nan)
    else:
        state_as_float = df_sorted["state"].replace(-1, np.nan)
    df_sorted = df_sorted.assign(state=state_as_float)
    df_sorted["state"] = (
        df_sorted.groupby("shot", group_keys=False)["state"]
        .transform(lambda s: s.ffill().bfill())
    )

    df_filtered = df_sorted.dropna(subset=["state"]).copy()
    df_filtered["state"] = df_filtered["state"].astype(int)
    df_filtered["state_binary"] = np.where(
        np.isin(df_filtered["state"].values, list(hp.SUPPRESSED_STATES)),
        0,
        1,
    ).astype(np.int64)

    feat_mask = ~df_filtered[selected_features].isna().any(axis=1)
    df_filtered = df_filtered.loc[feat_mask].copy()

    if not (df_filtered["shot"] == holdout_shot).any():
        raise ValueError(f"Shot {holdout_shot} not found in {data_path} after cleaning.")

    train_fit_mask = df_filtered["shot"] != holdout_shot
    scaler = StandardScaler()
    scaler.fit(df_filtered.loc[train_fit_mask, selected_features].values)

    X = scaler.transform(df_filtered[selected_features].values).astype(np.float32)
    y = df_filtered["state_binary"].values.astype(np.int64)
    shots = df_filtered["shot"].values
    times = df_filtered["time"].values.astype(np.float64)

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Binary labels: {Counter(y.tolist())}")
    print(f"  Holdout shot {holdout_shot} rows: {(shots == holdout_shot).sum():,}")

    return df_filtered, X, y, shots, times, selected_features, scaler


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------
def create_train_val_windows_excluding_holdout(
    X, y, times, shots, holdout_shot: int = HOLDOUT_SHOT,
):
    print(
        f"Creating causal windows (size {WINDOW_SIZE}, {PREDICTION_HORIZON_MS} ms ahead); "
        f"holdout {holdout_shot} excluded..."
    )
    unique_shots = np.unique(shots)
    if int(holdout_shot) not in {int(s) for s in unique_shots}:
        raise ValueError(f"Shot {holdout_shot} missing after filtering.")

    trainable = unique_shots[unique_shots != holdout_shot]
    n_shots = len(trainable)
    print(f"Trainable shots: {n_shots}")

    rng = np.random.RandomState(SHOT_SPLIT_SEED)
    shuffled = rng.permutation(trainable)
    train_end = int(TRAIN_RATIO * n_shots)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n_shots)
    train_shots = shuffled[:train_end]
    val_shots = shuffled[train_end:val_end]

    train_w, train_y = hp.create_windows_for_shots(
        X, y, times, shots, train_shots.tolist(), WINDOW_SIZE, PREDICTION_HORIZON_MS
    )
    val_w, val_y = hp.create_windows_for_shots(
        X, y, times, shots, val_shots.tolist(), WINDOW_SIZE, PREDICTION_HORIZON_MS
    )
    print(f"Windows: train={len(train_w):,} | val={len(val_w):,}")
    return train_w, train_y, val_w, val_y


def build_holdout_windows(X, y, times, shots, holdout_shot: int = HOLDOUT_SHOT):
    holdout_w, holdout_y = hp.create_windows_for_shots(
        X, y, times, shots, [holdout_shot], WINDOW_SIZE, PREDICTION_HORIZON_MS
    )
    if len(holdout_w) == 0:
        raise ValueError(f"Shot {holdout_shot}: no valid 50ms-ahead windows.")

    shot_mask = shots == holdout_shot
    t_sub = times[shot_mask]
    X_sub = X[shot_mask]
    window_size = WINDOW_SIZE
    window_end_times = []
    for i in range(len(X_sub) - window_size + 1):
        window_end_t = t_sub[i + window_size - 1]
        target_t = window_end_t + PREDICTION_HORIZON_MS
        future_idx = np.searchsorted(t_sub, target_t)
        if future_idx >= len(t_sub):
            continue
        window = X_sub[i : i + window_size]
        if np.isnan(window).any() or np.isinf(window).any():
            continue
        window_end_times.append(float(window_end_t))

    if len(window_end_times) != len(holdout_w):
        print(
            f"  Warning: {len(window_end_times)} end-times vs {len(holdout_w)} windows; "
            "using min length for plotting."
        )
        n = min(len(window_end_times), len(holdout_w))
        return holdout_w[:n], holdout_y[:n], np.array(window_end_times[:n], dtype=np.float64)

    return holdout_w, holdout_y, np.array(window_end_times, dtype=np.float64)


def load_holdout_plot_frame(df: pd.DataFrame, holdout_shot: int = HOLDOUT_SHOT):
    shot_df = df[df["shot"] == holdout_shot].copy()
    shot_df = shot_df.sort_values("time").reset_index(drop=True)
    if "fs04" not in shot_df.columns:
        raise ValueError("Column 'fs04' missing; cannot plot.")
    return shot_df


# -----------------------------------------------------------------------------
# Plotting (matches LSTM_50_2_Shot_169472_Predictions_Smoothed.py exactly)
# -----------------------------------------------------------------------------
def plot_smoothed_predictions(
    shot_df: pd.DataFrame,
    raw_probs: np.ndarray,
    smoothed_probs: np.ndarray,
    labels: np.ndarray,
    window_end_times: np.ndarray,
    raw_preds: np.ndarray,
    holdout_shot: int = HOLDOUT_SHOT,
    output_path: str = OUTPUT_PNG,
    threshold: float = PREDICTION_THRESHOLD,
):
    """Three-panel plot: actual states, smoothed probability, smoothed accuracy."""
    plt.style.use("seaborn-v0_8")
    time_arr = shot_df["time"].values
    fs04 = shot_df["fs04"].values
    actual_binary = shot_df["state_binary"].values.astype(int)
    horizon = PREDICTION_HORIZON_MS

    true_mapped, _ = map_values_to_future_times(time_arr, window_end_times, labels, horizon)
    raw_prob_mapped, prob_valid = map_values_to_future_times(
        time_arr, window_end_times, raw_probs, horizon,
    )
    raw_pred_mapped, _ = map_values_to_future_times(time_arr, window_end_times, raw_preds, horizon)

    smoothed_pred = (smoothed_probs >= threshold).astype(int)
    valid_eval = prob_valid & ~np.isnan(true_mapped)
    smoothed_acc = (
        accuracy_score(true_mapped[valid_eval].astype(int), smoothed_pred[valid_eval])
        if valid_eval.any() else float("nan")
    )
    raw_acc = accuracy_score(labels, raw_preds) if len(labels) else float("nan")

    y_lo, y_hi = fs04.min(), fs04.max()
    xlim = (time_arr.min(), time_arr.max())

    fig, axes = plt.subplots(3, 1, figsize=(24, 20))
    fig.suptitle(
        f"Plasma State Predictions (Smoothed) — Shot {holdout_shot} — TabPFN 50ms 2-state",
        fontsize=20, fontweight="bold", y=0.995,
    )

    # (a) Actual states
    ax1 = axes[0]
    ax1.plot(time_arr, fs04, "k-", linewidth=1, alpha=0.7, label="fs04")
    for state in (0, 1):
        mask = actual_binary == state
        if mask.any():
            ax1.fill_between(
                time_arr, y_lo, y_hi, where=mask, alpha=0.3,
                color=STATE_COLORS[state], label=f"Actual: {STATE_NAMES[state]}",
            )
    ax1.set_ylabel("fs04", fontsize=16)
    ax1.set_title("(a) Observed Plasma States", fontsize=18, fontweight="bold", pad=15)
    ax1.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12,
        frameon=True, fancybox=True, shadow=True,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)

    # (b) Smoothed P(class=1) with thresholded regions
    ax2 = axes[1]
    ax2.plot(time_arr, fs04, "k-", linewidth=1, alpha=0.5, label="fs04")
    ax2b = ax2.twinx()
    if prob_valid.any():
        ax2b.plot(
            time_arr[prob_valid], raw_prob_mapped[prob_valid],
            color="#9E9E9E", linewidth=0.8, alpha=0.55, label="Raw P(ELMy)",
        )
    ax2b.plot(time_arr, smoothed_probs, color="#1f77b4", linewidth=2.2, label="Smoothed P(ELMy)")
    ax2b.axhline(
        threshold, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"Threshold ({threshold:.2f})",
    )
    ax2b.set_ylim(-0.02, 1.02)
    ax2b.set_ylabel("P(ELMy)", fontsize=16, color="#1f77b4")
    ax2b.tick_params(axis="y", labelcolor="#1f77b4")

    for state in (0, 1):
        mask = smoothed_pred == state
        if mask.any():
            ax2.fill_between(
                time_arr, y_lo, y_hi, where=mask, alpha=0.28,
                color=STATE_COLORS[state],
                label=f"Smoothed pred: {STATE_NAMES[state]}",
            )
    ax2.set_ylabel("fs04", fontsize=16)
    ax2.set_title(
        f"(b) Smoothed Predictions (rolling {SMOOTH_WINDOW_MS:.0f} ms)",
        fontsize=18, fontweight="bold",
    )
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2, labels1 + labels2,
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11,
        frameon=True, fancybox=True, shadow=True,
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)

    # (c) Raw vs smoothed accuracy
    ax3 = axes[2]
    ax3.plot(time_arr, fs04, "k-", linewidth=2, alpha=0.8, label="fs04")

    raw_valid = prob_valid & ~np.isnan(raw_pred_mapped) & ~np.isnan(true_mapped)
    raw_correct = raw_valid & (raw_pred_mapped == true_mapped)
    raw_incorrect = raw_valid & (raw_pred_mapped != true_mapped)
    smooth_correct = valid_eval & (smoothed_pred == true_mapped.astype(int))
    smooth_incorrect = valid_eval & (smoothed_pred != true_mapped.astype(int))

    ax3.fill_between(time_arr, y_lo, y_hi, where=raw_correct, alpha=0.2,
                     color="#81C784", label="Raw correct")
    ax3.fill_between(time_arr, y_lo, y_hi, where=raw_incorrect, alpha=0.2,
                     color="#E57373", label="Raw incorrect")
    ax3.fill_between(time_arr, y_lo, y_hi, where=smooth_correct, alpha=0.45,
                     color="green", label="Smoothed correct")
    ax3.fill_between(time_arr, y_lo, y_hi, where=smooth_incorrect, alpha=0.45,
                     color="red", label="Smoothed incorrect")

    ax3.set_xlabel("Time (ms)", fontsize=16)
    ax3.set_ylabel("fs04", fontsize=16)
    ax3.set_title("(c) Raw vs Smoothed Prediction Accuracy", fontsize=18, fontweight="bold")
    ax3.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11,
        frameon=True, fancybox=True, shadow=True,
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(xlim)
    ax3.text(
        0.02, 0.98,
        f"Window argmax accuracy: {raw_acc:.3f}\n"
        f"Smoothed ({SMOOTH_WINDOW_MS:.0f} ms) accuracy: {smoothed_acc:.3f}",
        transform=ax3.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.subplots_adjust(hspace=0.3, bottom=0.06, top=0.92)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    max_px = 12_000
    save_dpi = min(800, max(100, int(max_px / max(w_in, h_in))))
    fig.savefig(
        output_path,
        format="png",
        dpi=save_dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
        pil_kwargs={"optimize": True},
    )
    plt.close(fig)
    print(f"  Saved at dpi={save_dpi} (~{int(w_in * save_dpi)}x{int(h_in * save_dpi)} px)")
    print(f"Smoothed plot saved to '{output_path}'")
    return raw_acc, smoothed_acc


# -----------------------------------------------------------------------------
# Training bundle
# -----------------------------------------------------------------------------
def train_bundle(train_tab, train_y, val_tab, val_y, features, tab_names, scaler, device, n_bags):
    print(
        f"\nFitting bagged TabPFN ({n_bags} bags x {hp.BAG_TRAIN_SIZE}, "
        f"n_estimators={hp.TABPFN_N_ESTIMATORS})..."
    )
    models, used_sizes = hp.fit_bagged_tabpfn(
        train_tab, train_y,
        n_bags=n_bags,
        bag_size=hp.BAG_TRAIN_SIZE,
        n_estimators=hp.TABPFN_N_ESTIMATORS,
        device=device,
    )
    if len(val_tab) == 0:
        best_t, best_val_acc = 0.5, float("nan")
    else:
        val_tune_tab, val_tune_y = _subsample_for_threshold(
            val_tab, val_y, VAL_TUNE_MAX_SAMPLES
        )
        val_probs = hp.bagged_predict_proba(
            models, val_tune_tab, batch_label="Validation (tune subset)"
        )
        best_t, best_val_acc, _, _ = hp.tune_threshold_for_accuracy(val_probs, val_tune_y)
        print(f"Validation accuracy at threshold {best_t:.4f}: {best_val_acc:.4f}")

    bundle = {
        "models": models,
        "threshold": best_t,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": features,
        "tab_feature_names": tab_names,
        "window_size": WINDOW_SIZE,
        "holdout_shot": HOLDOUT_SHOT,
        "prediction_horizon_ms": PREDICTION_HORIZON_MS,
        "bagging": {
            "n_bags": n_bags,
            "bag_train_size": hp.BAG_TRAIN_SIZE,
            "n_estimators": hp.TABPFN_N_ESTIMATORS,
            "used_sizes": used_sizes,
        },
        "val_accuracy_tuned": best_val_acc,
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved bundle: {BUNDLE_PATH}")
    return bundle


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TabPFN HighPerf isolated-shot 169472 smoothed predictions plot"
    )
    parser.add_argument(
        "data_csv", nargs="?", default=DEFAULT_CSV,
        help=f"Training CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-bags", type=int, default=None)
    parser.add_argument("--holdout-shot", type=int, default=HOLDOUT_SHOT)
    args = parser.parse_args()

    holdout = args.holdout_shot
    n_bags = args.n_bags if args.n_bags is not None else hp.N_BAGS
    output_png = os.path.join(
        OUTPUT_DIR,
        f"isolated_shot_{holdout}_tabpfn_binary_comprehensive_analysis.png",
    )

    print("=" * 72)
    print(f"TabPFN HighPerf isolated-shot analysis — shot {holdout}")
    print("=" * 72)
    print(f"  CSV: {args.data_csv}")
    print(f"  Smooth window: {SMOOTH_WINDOW_MS} ms")
    print(f"  Threshold: {PREDICTION_THRESHOLD}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("TABPFN_PREDICT_BATCH_SIZE"):
        hp.PREDICT_BATCH_SIZE = int(os.environ["TABPFN_PREDICT_BATCH_SIZE"])
    elif device == "cuda":
        hp.PREDICT_BATCH_SIZE = max(hp.PREDICT_BATCH_SIZE, 8192)
    print(f"Device: {device} (predict batch_size={hp.PREDICT_BATCH_SIZE})")

    df, X, y, shots, times, features, scaler = load_and_prepare_data(
        args.data_csv, holdout_shot=holdout
    )
    holdout_w, holdout_y, holdout_end_times = build_holdout_windows(
        X, y, times, shots, holdout_shot=holdout
    )
    print(f"Holdout windows: {len(holdout_w):,}")

    if args.skip_train:
        if not os.path.isfile(BUNDLE_PATH):
            raise FileNotFoundError(f"--skip-train but missing: {BUNDLE_PATH}")
        with open(BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        print(f"Loaded bundle: {BUNDLE_PATH} (skip-train)")
    else:
        train_w, train_y, val_w, val_y = create_train_val_windows_excluding_holdout(
            X, y, times, shots, holdout_shot=holdout
        )
        print("\nExtracting tabular features (train + val + holdout)...")
        t0 = time.time()
        train_tab, tab_names = hp.windows_to_tabular_features(train_w, features)
        val_tab, _ = hp.windows_to_tabular_features(val_w, features)
        holdout_tab, _ = hp.windows_to_tabular_features(holdout_w, features)
        print(f"  Done in {time.time() - t0:.1f}s")
        print(f"  train={train_tab.shape}, val={val_tab.shape}, holdout={holdout_tab.shape}")
        if len(train_tab) == 0:
            raise RuntimeError("Empty training set after excluding holdout.")
        bundle = train_bundle(
            train_tab, train_y, val_tab, val_y, features, tab_names, scaler, device, n_bags
        )

    print("\nExtracting holdout tabular features...")
    t0 = time.time()
    holdout_tab, _ = hp.windows_to_tabular_features(holdout_w, features)
    print(f"  Done in {time.time() - t0:.1f}s, holdout={holdout_tab.shape}")

    models = bundle["models"]
    threshold = PREDICTION_THRESHOLD
    holdout_probs = hp.bagged_predict_proba(
        models, holdout_tab, batch_label=f"Holdout {holdout}"
    )
    raw_probs_pos = holdout_probs[:, hp.POS_CLASS_INDEX]
    raw_preds = (raw_probs_pos >= threshold).astype(int)
    holdout_acc = accuracy_score(holdout_y, raw_preds)
    print(f"\nHoldout {PREDICTION_HORIZON_MS}ms-ahead accuracy (raw): {holdout_acc:.4f}")

    if len(np.unique(holdout_y)) > 1:
        auc = roc_auc_score(holdout_y, raw_probs_pos)
        print(f"Holdout ROC AUC: {auc:.4f}")

    print(classification_report(holdout_y, raw_preds, target_names=hp.CLASS_NAMES, digits=4))

    shot_df = load_holdout_plot_frame(df, holdout_shot=holdout)
    time_arr = shot_df["time"].values
    print(
        f"Plot rows: {len(shot_df)}, time {time_arr.min():.0f}–{time_arr.max():.0f} ms"
    )

    raw_prob_mapped, prob_valid = map_values_to_future_times(
        time_arr, holdout_end_times, raw_probs_pos, PREDICTION_HORIZON_MS,
    )
    smoothed_probs, window_pts, dt = smooth_timeline_series(
        np.where(prob_valid, raw_prob_mapped, np.nan), time_arr,
    )
    print(f"Smoothing: median dt={dt:.2f} ms, window={window_pts} samples")

    raw_acc_plot, smoothed_acc_plot = plot_smoothed_predictions(
        shot_df,
        raw_probs_pos,
        smoothed_probs,
        holdout_y,
        holdout_end_times,
        raw_preds,
        holdout_shot=holdout,
        output_path=output_png,
        threshold=threshold,
    )
    print(f"\nRaw accuracy: {raw_acc_plot:.4f}")
    print(f"Smoothed accuracy: {smoothed_acc_plot:.4f}")
    print("Done.")
    print(f"  Plot: {output_png}")


if __name__ == "__main__":
    main()
