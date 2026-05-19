#!/usr/bin/env python3
"""
Train TabPFN binary (HighPerf config) on all shots except 169472, tune threshold on
validation, run 50ms-ahead inference on shot 169472 only, and save a three-panel
fs04 comprehensive plot (no confidence panel).

Based on TabPFN_Center_Point_Binary_HighPerf.py; plotting pattern from
isolated_shot_169472_bilstm_binary_comprehensive.py / LSTM_50_2_Shot_169472_Predictions.py.

Outputs:
  - Pickle: TabPFN_Architecture/tabpfn_binary_highperf_shot_169472_holdout.pkl
  - Plot: Isolated Shot Comprehensive Analysis/isolated_shot_169472_tabpfn_comprehensive_analysis.png

Usage:
  MPLBACKEND=Agg TabPFN_Architecture/.venv/bin/python TabPFN_Shot_169472_Predictions.py [data_csv] [run_tag]

  # Reuse saved bundle (plot only):
  MPLBACKEND=Agg TabPFN_Architecture/.venv/bin/python TabPFN_Shot_169472_Predictions.py --skip-train
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import TabPFN_Center_Point_Binary_HighPerf as base  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HOLDOUT_SHOT = 169472
PROBLEM_SHOT = base.PROBLEM_SHOT
DATA_CSV = (
    sys.argv[1]
    if len(sys.argv) > 1
    else base.DATA_PATH
)
RUN_TAG = sys.argv[2] if len(sys.argv) > 2 else "shot_169472_holdout"
# Optional env overrides for long runs (defaults match parent HighPerf).
N_BAGS = int(os.environ.get("TABPFN_N_BAGS", base.N_BAGS))
BAG_TRAIN_SIZE = int(os.environ.get("TABPFN_BAG_TRAIN_SIZE", base.BAG_TRAIN_SIZE))
WINDOW_SIZE = base.WINDOW_SIZE
PREDICTION_HORIZON_MS = base.PREDICTION_HORIZON_MS
TRAIN_RATIO = base.TRAIN_RATIO
VAL_RATIO = base.VAL_RATIO
SHOT_SPLIT_SEED = base.SHOT_SPLIT_SEED

SAVE_DIR = SCRIPT_DIR
OUTPUT_DIR = "/mnt/homes/sr4240/my_folder/Isolated Shot Comprehensive Analysis"
PLOT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    f"isolated_shot_{HOLDOUT_SHOT}_tabpfn_comprehensive_analysis.png",
)
CHECKPOINT_PATH = os.path.join(
    SAVE_DIR, f"tabpfn_binary_highperf_{RUN_TAG}.pkl"
)

BINARY_COLORS = {0: "#2E8B57", 1: "#DC143C"}
BINARY_NAMES = {0: "Suppressed", 1: "ELMy"}


def _make_tabpfn_classifier(device: str, seed: int):
    """Support both tabpfn<3 (N_ensemble_configurations) and tabpfn>=3 APIs."""
    import inspect
    from tabpfn import TabPFNClassifier

    params = inspect.signature(TabPFNClassifier.__init__).parameters
    if "n_estimators" in params:
        return TabPFNClassifier(
            device=device,
            n_estimators=base.TABPFN_N_ESTIMATORS,
            balance_probabilities=True,
            ignore_pretraining_limits=False,
            random_state=seed,
        )
    return TabPFNClassifier(
        device=device,
        N_ensemble_configurations=base.TABPFN_N_ESTIMATORS,
        seed=seed,
    )


def fit_bagged_tabpfn_compat(train_X, train_y, n_bags=N_BAGS, bag_size=BAG_TRAIN_SIZE, device="cpu"):
    """Same bagging loop as parent, with TabPFN version compatibility."""
    models = []
    used_sizes = []
    for k in range(n_bags):
        seed = base.RANDOM_SEED + k
        Xb, yb, _ = base.stratified_subsample(train_X, train_y, bag_size, seed=seed)
        used_sizes.append(len(Xb))
        print(
            f"  Bag {k+1}/{n_bags}: subsample size {len(Xb):>6} "
            f"(seed={seed}, dist={Counter(yb.tolist())})"
        )
        clf = _make_tabpfn_classifier(device=device, seed=seed)
        t0 = time.time()
        clf.fit(Xb, yb)
        print(f"    fit_time={time.time() - t0:.1f}s")
        models.append(clf)
    return models, used_sizes


def load_and_prepare_data_holdout(data_path: str, holdout_shot: int = HOLDOUT_SHOT):
    """Same cleaning/labels as parent; scaler fit excluding holdout shot."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df[df["shot"] != PROBLEM_SHOT].copy()

    selected_features = [f for f in base.IMPORTANT_FEATURES if f in df.columns]
    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    pre_dist = Counter(df_sorted["state"].values.tolist())
    n_unknown_pre = int((df_sorted["state"] == -1).sum())
    print(f"Raw state distribution (incl. -1 edges): {pre_dist}")
    print(f"  Unknown (-1) frames before label padding: {n_unknown_pre:,}")

    state_as_float = df_sorted["state"].replace(-1, np.nan)
    df_sorted = df_sorted.assign(state=state_as_float)
    df_sorted["state"] = (
        df_sorted.groupby("shot", group_keys=False)["state"]
        .transform(lambda s: s.ffill().bfill())
    )

    n_remaining = int(df_sorted["state"].isna().sum())
    n_filled = n_unknown_pre - n_remaining
    print(f"  -1 frames filled by per-shot label padding: {n_filled:,}")
    if n_remaining:
        print(f"  -1 frames remaining (entire-shot unlabelled, dropped): {n_remaining:,}")

    df_filtered = df_sorted.dropna(subset=["state"]).copy()
    df_filtered["state"] = df_filtered["state"].astype(int)
    df_filtered["state_binary"] = np.where(
        np.isin(df_filtered["state"].values, list(base.SUPPRESSED_STATES)),
        0,
        1,
    ).astype(np.int64)

    feat_mask = ~df_filtered[selected_features].isna().any(axis=1)
    df_filtered = df_filtered.loc[feat_mask].copy()

    if not (df_filtered["shot"] == holdout_shot).any():
        raise ValueError(
            f"Shot {holdout_shot} not found in {data_path} after cleaning."
        )

    train_fit_mask = df_filtered["shot"] != holdout_shot
    scaler = StandardScaler()
    scaler.fit(df_filtered.loc[train_fit_mask, selected_features].values)

    X = scaler.transform(df_filtered[selected_features].values)
    y = df_filtered["state_binary"].values.astype(np.int64)
    shots = df_filtered["shot"].values
    times = df_filtered["time"].values.astype(np.float64)

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Binary label distribution: {Counter(y.tolist())}")
    print(f"  Holdout shot {holdout_shot} rows: {(shots == holdout_shot).sum():,}")

    return df_filtered, X.astype(np.float32), y, shots, times, selected_features, scaler


def create_windows_excluding_holdout(
    X, y, times, shots, holdout_shot: int = HOLDOUT_SHOT,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
):
    """Shot-based 70/15/15 split on all shots except holdout."""
    print(
        f"Creating causal windows of size {window_size} "
        f"({PREDICTION_HORIZON_MS} ms ahead; holdout {holdout_shot} excluded)..."
    )
    unique_shots = np.unique(shots)
    if int(holdout_shot) not in {int(s) for s in unique_shots}:
        raise ValueError(f"Shot {holdout_shot} missing after filtering.")

    trainable_shots = unique_shots[unique_shots != holdout_shot]
    n_shots = len(trainable_shots)
    print(f"Total unique shots (excl. holdout): {n_shots}")

    rng = np.random.RandomState(SHOT_SPLIT_SEED)
    shuffled = rng.permutation(trainable_shots)
    train_end = int(train_ratio * n_shots)
    val_end = int((train_ratio + val_ratio) * n_shots)

    train_shots = shuffled[:train_end]
    val_shots = shuffled[train_end:val_end]

    assert holdout_shot not in train_shots
    assert holdout_shot not in val_shots

    print(
        f"Shot split: train {len(train_shots)} | val {len(val_shots)} "
        f"| holdout eval only: {holdout_shot}"
    )

    train_w, train_y = base.create_windows_for_shots(
        X, y, times, shots, train_shots.tolist(), window_size, PREDICTION_HORIZON_MS
    )
    val_w, val_y = base.create_windows_for_shots(
        X, y, times, shots, val_shots.tolist(), window_size, PREDICTION_HORIZON_MS
    )

    print(f"\nWindows: train={len(train_w)} | val={len(val_w)}")
    print(f"Train labels: {Counter(train_y.tolist())}")
    print(f"Val   labels: {Counter(val_y.tolist())}")
    return train_w, train_y, val_w, val_y


def build_holdout_shot_windows(
    X, y, times, shots, holdout_shot: int = HOLDOUT_SHOT,
    window_size: int = WINDOW_SIZE,
    prediction_horizon_ms: int = PREDICTION_HORIZON_MS,
):
    """Causal 50ms-ahead windows for the holdout shot only (matches parent)."""
    shot_mask = shots == holdout_shot
    if not shot_mask.any():
        raise ValueError(f"Shot {holdout_shot} not found after cleaning.")

    shot_indices = np.where(shot_mask)[0]
    if len(shot_indices) < window_size:
        raise ValueError(
            f"Shot {holdout_shot} has only {len(shot_indices)} rows; "
            f"need at least {window_size}."
        )

    X_sub = X[shot_mask]
    y_sub = y[shot_mask].astype(int)
    t_sub = times[shot_mask]
    valid_labels = {0, 1}

    windows, labels, window_end_times = [], [], []
    for i in range(len(shot_indices) - window_size + 1):
        window = X_sub[i : i + window_size]
        window_end_t = t_sub[i + window_size - 1]
        target_t = window_end_t + prediction_horizon_ms
        future_idx = np.searchsorted(t_sub, target_t)
        if future_idx >= len(t_sub):
            continue
        future_label = int(y_sub[future_idx])
        if future_label not in valid_labels:
            continue
        if np.isnan(window).any() or np.isinf(window).any():
            continue
        windows.append(window)
        labels.append(future_label)
        window_end_times.append(float(window_end_t))

    if len(windows) == 0:
        raise ValueError(f"Shot {holdout_shot}: no valid windows after horizon filtering.")

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=int),
        np.array(window_end_times, dtype=np.float64),
    )


def load_holdout_shot_plot_frame(df, holdout_shot: int = HOLDOUT_SHOT):
    shot_df = df[df["shot"] == holdout_shot].copy()
    if shot_df.empty:
        raise ValueError(f"Shot {holdout_shot} not found for plotting.")
    shot_df = shot_df.sort_values("time").reset_index(drop=True)
    if "fs04" not in shot_df.columns:
        raise ValueError("Column 'fs04' missing; cannot plot comprehensive analysis.")
    return shot_df


def plot_comprehensive_analysis(
    shot_df, predictions, labels, window_end_times,
    holdout_acc, holdout_shot=HOLDOUT_SHOT, output_path=PLOT_OUTPUT,
    prediction_horizon_ms: int = PREDICTION_HORIZON_MS,
):
    """Three-panel fs04 plot (actual / predicted / accuracy); no confidence panel."""
    plt.style.use("seaborn-v0_8")

    time = shot_df["time"].values
    fs04 = shot_df["fs04"].values
    actual_binary = shot_df["state_binary"].values.astype(int)

    pred_at_time = np.full(len(shot_df), -1, dtype=float)
    true_at_time = np.full(len(shot_df), -1, dtype=float)

    for pred, true, window_end_time in zip(predictions, labels, window_end_times):
        future_time = window_end_time + prediction_horizon_ms
        time_idx = int(np.argmin(np.abs(time - future_time)))
        if time_idx < len(pred_at_time):
            pred_at_time[time_idx] = pred
            true_at_time[time_idx] = true

    y_lo, y_hi = fs04.min(), fs04.max()
    xlim = (time.min(), time.max())

    fig, axes = plt.subplots(3, 1, figsize=(24, 18))
    fig.suptitle(
        f"Plasma State Classification Analysis for Shot {holdout_shot} - "
        f"TabPFN (Split Based on Shot)",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    ax1 = axes[0]
    ax1.plot(time, fs04, "k-", linewidth=1, alpha=0.7, label="fs04")
    for state in (0, 1):
        mask = actual_binary == state
        if mask.any():
            ax1.fill_between(
                time, y_lo, y_hi, where=mask, alpha=0.3,
                color=BINARY_COLORS[state],
                label=f"Actual: {BINARY_NAMES[state]}",
            )
    ax1.set_ylabel("fs04 Signal (a.u.)", fontsize=16)
    ax1.set_title("(a) Observed Plasma States", fontsize=18, fontweight="bold", pad=15)
    ax1.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12,
        frameon=True, fancybox=True, shadow=True,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)

    ax2 = axes[1]
    ax2.plot(time, fs04, "k-", linewidth=1, alpha=0.7, label="fs04")
    for state in (0, 1):
        mask = pred_at_time == state
        if mask.any():
            ax2.fill_between(
                time, y_lo, y_hi, where=mask, alpha=0.3,
                color=BINARY_COLORS[state],
                label=f"Predicted: {BINARY_NAMES[state]}",
            )
    ax2.set_ylabel("fs04 Signal (a.u.)", fontsize=16)
    ax2.set_title("(b) Predicted Plasma States", fontsize=18, fontweight="bold")
    ax2.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12,
        frameon=True, fancybox=True, shadow=True,
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)

    ax3 = axes[2]
    ax3.plot(time, fs04, "k-", linewidth=2, alpha=0.8, label="fs04")
    valid = (pred_at_time >= 0) & (true_at_time >= 0)
    correct = valid & (pred_at_time == true_at_time)
    incorrect = valid & (pred_at_time != true_at_time)
    if correct.any():
        ax3.fill_between(
            time, y_lo, y_hi, where=correct, alpha=0.4,
            color="green", label="Correct Predictions",
        )
    if incorrect.any():
        ax3.fill_between(
            time, y_lo, y_hi, where=incorrect, alpha=0.4,
            color="red", label="Incorrect Predictions",
        )
    ax3.set_xlabel("Time (ms)", fontsize=16)
    ax3.set_ylabel("fs04 Signal (a.u.)", fontsize=16)
    ax3.set_title("(c) Prediction Accuracy Analysis", fontsize=18, fontweight="bold")
    ax3.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12,
        frameon=True, fancybox=True, shadow=True,
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(xlim)

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.subplots_adjust(hspace=0.3, bottom=0.06, top=0.92)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print(f"Comprehensive plot saved to '{output_path}'")


def main():
    print("=" * 72)
    print(f"TabPFN HighPerf — holdout shot {HOLDOUT_SHOT}")
    print("=" * 72)
    print(f"  CSV: {DATA_CSV}")
    print(f"  RUN_TAG: {RUN_TAG}")
    print(f"  Holdout: {HOLDOUT_SHOT} (excluded from train/val/threshold-fit shots)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Smaller inference batches when GPU is shared (parent default 2048 can OOM).
    if os.environ.get("TABPFN_PREDICT_BATCH_SIZE"):
        base.PREDICT_BATCH_SIZE = int(os.environ["TABPFN_PREDICT_BATCH_SIZE"])
    elif base.PREDICT_BATCH_SIZE > 512:
        base.PREDICT_BATCH_SIZE = 512
    print(f"Using device: {device} (predict batch_size={base.PREDICT_BATCH_SIZE})")

    df, X, y, shots, times, features, scaler = load_and_prepare_data_holdout(DATA_CSV)

    train_w, train_y, val_w, val_y = create_windows_excluding_holdout(
        X, y, times, shots
    )
    if len(train_w) == 0 or len(val_w) == 0:
        raise RuntimeError("Empty train or val set after holdout exclusion.")

    holdout_w, holdout_y, holdout_end_times = build_holdout_shot_windows(
        X, y, times, shots
    )
    print(f"Holdout windows for shot {HOLDOUT_SHOT}: {len(holdout_w):,}")

    print("\nExtracting tabular features (expanded set)...")
    t0 = time.time()
    train_tab, tab_feature_names = base.windows_to_tabular_features(train_w, features)
    val_tab, _ = base.windows_to_tabular_features(val_w, features)
    holdout_tab, _ = base.windows_to_tabular_features(holdout_w, features)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(
        f"  Tabular shapes: train={train_tab.shape}, val={val_tab.shape}, "
        f"holdout={holdout_tab.shape}"
    )

    print(
        f"\nFitting bagged TabPFN ({N_BAGS} bags x {BAG_TRAIN_SIZE} train each)..."
    )
    t0 = time.time()
    models, used_sizes = fit_bagged_tabpfn_compat(
        train_tab,
        train_y,
        device=device,
    )
    print(f"Total bagged fit time: {time.time() - t0:.1f}s")

    val_probs = base.bagged_predict_proba(models, val_tab, batch_label="Validation")
    holdout_probs = base.bagged_predict_proba(
        models, holdout_tab, batch_label=f"Holdout shot {HOLDOUT_SHOT}"
    )

    best_t, best_val_acc, _, _ = base.tune_threshold_for_accuracy(val_probs, val_y)
    print(
        f"\nTuned threshold from validation: {best_t:.4f} (val acc {best_val_acc:.4f})"
    )

    holdout_preds = (holdout_probs[:, base.POS_CLASS_INDEX] >= best_t).astype(int)
    holdout_acc = accuracy_score(holdout_y, holdout_preds)
    print(
        f"Holdout shot {HOLDOUT_SHOT} {PREDICTION_HORIZON_MS}ms-ahead accuracy: "
        f"{holdout_acc:.4f}"
    )

    shot_df = load_holdout_shot_plot_frame(df)
    print(
        f"Holdout shot plot rows: {len(shot_df)}, "
        f"time {shot_df['time'].min():.0f}–{shot_df['time'].max():.0f} ms"
    )
    print(f"state_binary distribution: {shot_df['state_binary'].value_counts().to_dict()}")

    plot_comprehensive_analysis(
        shot_df, holdout_preds, holdout_y, holdout_end_times, holdout_acc
    )

    bundle = {
        "models": models,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": features,
        "tab_feature_names": tab_feature_names,
        "window_size": WINDOW_SIZE,
        "class_names": base.CLASS_NAMES,
        "holdout_shot": HOLDOUT_SHOT,
        "run_tag": RUN_TAG,
        "data_csv": DATA_CSV,
        "bagging": {
            "n_bags": N_BAGS,
            "bag_train_size": BAG_TRAIN_SIZE,
            "n_estimators_per_bag": base.TABPFN_N_ESTIMATORS,
            "used_sizes": used_sizes,
        },
        "threshold_tuned": best_t,
        "val_accuracy_tuned": best_val_acc,
        "prediction_horizon_ms": PREDICTION_HORIZON_MS,
        "holdout_accuracy": holdout_acc,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved TabPFN bundle to: {CHECKPOINT_PATH}")

    print("\nDone.")
    print(f"  Script: {os.path.abspath(__file__)}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Plot: {PLOT_OUTPUT}")
    print(f"  Holdout accuracy: {holdout_acc:.4f}")


if __name__ == "__main__":
    main()
