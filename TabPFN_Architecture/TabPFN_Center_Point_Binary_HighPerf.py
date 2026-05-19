"""
TabPFN_Center_Point_Binary_HighPerf.py

High-performance binary TabPFN for plasma state classification.

Same label handling as `TabPFN_Center_Point_Binary.py`:
    - per-shot ffill+bfill of state == -1
    - shot-based 70/15/15 split with the same seed

Windowing (causal, 50 ms ahead — matches `LSTM_50_2_Binary_Transitions_eval.py`):
    - 150-step window uses only past+present plasma data (oldest -> newest)
    - label is binary state at window_end_time + 50 ms (no future features)

What this script changes for performance:
    1. Expanded tabular features (~162 vs the previous 78):
         * existing 13 stats per raw feature
         * first-difference statistics: mean(|diff|), std(diff), max(|diff|),
           and 90th percentile of absolute deviation from the window mean
         * multi-scale stats on the trailing 50-step and 25-step sub-windows
           (most recent segment of the causal window)
         * trend stats: last_quarter - first_quarter, last - mean(first_half),
           slope of the last quarter
    2. Bagged ensemble of K TabPFN classifiers (different stratified train
       subsamples, different seeds). Predictions are averaged probabilities.
    3. n_estimators = 8 (TabPFN's max-quality default) inside each bag.
    4. balance_probabilities = True (post-correction for class prior).
    5. Decision-threshold tuning on the validation set. The threshold that
       maximizes binary accuracy on validation is then applied to test.
"""

import inspect
import time
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")

try:
    from tabpfn import TabPFNClassifier
except ImportError as e:
    raise ImportError(
        "TabPFN is not installed in this venv. Install with:\n"
        "    pip install 'tabpfn<3.0'\n"
    ) from e


# ---------------------------------------------------------------------------
# Reproducibility & constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 43
SHOT_SPLIT_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

DATA_PATH = "/mnt/homes/sr4240/my_folder/plasma_data.csv"
SAVE_DIR = "/mnt/homes/sr4240/my_folder/TabPFN_Architecture"
PROBLEM_SHOT = 191675
IMPORTANT_FEATURES = [
    "iln3iamp",
    "betan",
    "density",
    "li",
    "tritop",
    "fs_sum_max_smoothed",
]

VALID_STATES = [0, 1, 2, 3]
SUPPRESSED_STATES = {0}
ELMY_STATES = {1, 2, 3}
CLASS_NAMES = ["Suppressed", "ELMy"]
POS_CLASS_INDEX = 1

WINDOW_SIZE = 150
PREDICTION_HORIZON_MS = 50
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Bagging configuration. Each bag is a fully-fit TabPFNClassifier with
# n_estimators internal estimators, fit on a different stratified subsample.
N_BAGS = 5
BAG_TRAIN_SIZE = 10_000        # in-pretraining-distribution per bag
TABPFN_N_ESTIMATORS = 8        # max-quality default per bag
# Chunk predict_proba to avoid CUDA OOM when val/test are large or GPU is shared.
PREDICT_BATCH_SIZE = 2048


# ---------------------------------------------------------------------------
# Data loading (matches BiLSTM_NN_Center_Point_Binary.py exactly)
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """Load plasma data, fill -1 labels per-shot via ffill+bfill, binary remap."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["shot"] != PROBLEM_SHOT].copy()

    selected_features = [f for f in IMPORTANT_FEATURES if f in df.columns]
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

    X = df_filtered[selected_features].values
    y_multi = df_filtered["state"].values.astype(int)
    shots = df_filtered["shot"].values

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_multi = y_multi[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Multi-class label distribution after padding: {Counter(y_multi.tolist())}")

    y = np.where(np.isin(y_multi, list(SUPPRESSED_STATES)), 0, 1).astype(np.int64)
    print(f"Binary label distribution (0=Suppressed, 1=ELMy): {Counter(y.tolist())}")

    times = df_filtered["time"].values[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, times, shots, selected_features, scaler


# ---------------------------------------------------------------------------
# Causal shot windowing: 150 past steps -> label at window_end + 50 ms
# ---------------------------------------------------------------------------
def create_windows_for_shots(
    X,
    y,
    times,
    shots,
    shot_list,
    window_size=WINDOW_SIZE,
    prediction_horizon_ms=PREDICTION_HORIZON_MS,
):
    """Build causal windows: features from [t-(W-1)..t], label at t + horizon_ms."""
    windows, labels = [], []
    valid_labels = {0, 1}
    skipped_no_future = 0
    skipped_invalid = 0

    for shot_id in shot_list:
        shot_indices = np.where(shots == shot_id)[0]
        if len(shot_indices) < window_size:
            continue

        X_sub = X[shot_indices]
        y_sub = y[shot_indices].astype(int)
        t_sub = times[shot_indices]

        for i in range(len(shot_indices) - window_size + 1):
            window = X_sub[i : i + window_size]
            window_end_t = t_sub[i + window_size - 1]
            target_t = window_end_t + prediction_horizon_ms

            future_idx = np.searchsorted(t_sub, target_t)
            if future_idx >= len(t_sub):
                skipped_no_future += 1
                continue

            future_label = int(y_sub[future_idx])
            if future_label not in valid_labels:
                skipped_invalid += 1
                continue

            if np.isnan(window).any() or np.isinf(window).any():
                continue

            windows.append(window)
            labels.append(future_label)

    if skipped_no_future or skipped_invalid:
        print(
            f"  Shot {shot_list[0] if len(shot_list) == 1 else 'batch'}: "
            f"skipped no_future={skipped_no_future:,} invalid_label={skipped_invalid:,}"
        )

    if not windows:
        return np.array([]), np.array([])
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def create_windows_with_shot_split(
    X,
    y,
    times,
    shots,
    window_size=WINDOW_SIZE,
    prediction_horizon_ms=PREDICTION_HORIZON_MS,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
):
    print(
        f"Creating causal windows of size {window_size} "
        f"(predicting {prediction_horizon_ms} ms ahead) with SHOT-BASED split..."
    )
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total number of unique shots: {n_shots}")

    rng = np.random.RandomState(SHOT_SPLIT_SEED)
    shuffled_shots = rng.permutation(unique_shots)
    train_end = int(train_ratio * n_shots)
    val_end = int((train_ratio + val_ratio) * n_shots)

    train_shots = shuffled_shots[:train_end]
    val_shots = shuffled_shots[train_end:val_end]
    test_shots = shuffled_shots[val_end:]
    print(f"\nShot split: train {len(train_shots)} | val {len(val_shots)} | test {len(test_shots)}")

    print("\nCreating windows for each split...")
    train_w, train_y = create_windows_for_shots(
        X, y, times, shots, train_shots, window_size, prediction_horizon_ms
    )
    val_w, val_y = create_windows_for_shots(
        X, y, times, shots, val_shots, window_size, prediction_horizon_ms
    )
    test_w, test_y = create_windows_for_shots(
        X, y, times, shots, test_shots, window_size, prediction_horizon_ms
    )

    print(f"\nWindows: train={len(train_w)} | val={len(val_w)} | test={len(test_w)}")
    print(f"Train labels: {Counter(train_y)}")
    print(f"Val   labels: {Counter(val_y)}")
    print(f"Test  labels: {Counter(test_y)}")
    return train_w, train_y, val_w, val_y, test_w, test_y


# ---------------------------------------------------------------------------
# Expanded window -> tabular feature engineering
# ---------------------------------------------------------------------------
def _vectorized_slopes(windows):
    """Linear slope per (window, feature) along the time axis."""
    n_windows, win_size, n_feat = windows.shape
    if win_size < 2:
        return np.zeros((n_windows, n_feat), dtype=np.float32)
    t = np.arange(win_size, dtype=np.float32)
    t_centered = t - t.mean()
    var_t = float((t_centered ** 2).mean())
    if var_t == 0:
        return np.zeros((n_windows, n_feat), dtype=np.float32)
    means = windows.mean(axis=1, keepdims=True)
    x_centered = windows - means
    cov_tx = (t_centered[None, :, None] * x_centered).mean(axis=1)
    return (cov_tx / var_t).astype(np.float32)


def windows_to_tabular_features(windows, feature_names):
    """Compact but information-dense per-window summary (~162 features).

    Includes the original 13 stats plus first-difference statistics, two
    multi-scale center sub-window summaries, and trend-around-center stats.
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected windows of shape (N, T, F); got {windows.shape}")

    n_windows, win_size, n_features = windows.shape
    # Causal window: index 0 = oldest, win_size-1 = most recent ("now").
    now_idx = win_size - 1

    blocks = {}

    # ---- Global stats (full window) ----
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    blocks["mean"] = means
    blocks["std"] = stds
    blocks["min"] = mins
    blocks["max"] = maxs
    blocks["median"] = np.median(windows, axis=1)
    blocks["q25"] = np.percentile(windows, 25, axis=1)
    blocks["q75"] = np.percentile(windows, 75, axis=1)
    blocks["first"] = windows[:, 0, :]
    blocks["last"] = windows[:, -1, :]
    blocks["now"] = windows[:, now_idx, :]
    blocks["range"] = maxs - mins
    blocks["delta"] = blocks["last"] - blocks["first"]
    blocks["slope"] = _vectorized_slopes(windows)

    # ---- First-difference / spikiness statistics ----
    diffs = np.diff(windows, axis=1)
    abs_diffs = np.abs(diffs)
    blocks["diff_abs_mean"] = abs_diffs.mean(axis=1)
    blocks["diff_std"] = diffs.std(axis=1)
    blocks["diff_abs_max"] = abs_diffs.max(axis=1)
    blocks["abs_dev_q90"] = np.percentile(np.abs(windows - means[:, None, :]), 90, axis=1)

    # ---- Multi-scale stats on trailing 50 and 25 steps (most recent segment) ----
    for sub_name, sub_size in [("t50", 50), ("t25", 25)]:
        sub = windows[:, -sub_size:, :]
        blocks[f"{sub_name}_mean"] = sub.mean(axis=1)
        blocks[f"{sub_name}_std"] = sub.std(axis=1)
        blocks[f"{sub_name}_slope"] = _vectorized_slopes(sub)

    # ---- Trend stats (oldest vs most recent quarters of causal window) ----
    q = win_size // 4
    fq = windows[:, :q, :]
    lq = windows[:, -q:, :]
    fh = windows[:, : 2 * q, :]
    blocks["lq_minus_fq_mean"] = lq.mean(axis=1) - fq.mean(axis=1)
    blocks["now_minus_fh_mean"] = blocks["now"] - fh.mean(axis=1)
    blocks["lq_minus_fq_std"] = lq.std(axis=1) - fq.std(axis=1)
    blocks["lq_slope"] = _vectorized_slopes(lq)

    out_blocks, names = [], []
    for stat_name, arr in blocks.items():
        out_blocks.append(arr.astype(np.float32))
        names.extend([f"{f}__{stat_name}" for f in feature_names])

    return np.concatenate(out_blocks, axis=1), names


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------
def stratified_subsample(X, y, max_samples, seed):
    if max_samples is None or len(X) <= max_samples:
        return X, y, np.arange(len(X))

    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()

    quotas = np.maximum(1, np.round(proportions * max_samples).astype(int))
    while quotas.sum() > max_samples:
        quotas[quotas.argmax()] -= 1
    while quotas.sum() < max_samples and (quotas < counts).any():
        room = counts - quotas
        quotas[room.argmax()] += 1

    chosen = []
    for cls, quota in zip(classes, quotas):
        idx = np.where(y == cls)[0]
        take = int(min(quota, len(idx)))
        chosen.append(rng.choice(idx, size=take, replace=False))
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    return X[chosen], y[chosen], chosen


# ---------------------------------------------------------------------------
# Bagged TabPFN
# ---------------------------------------------------------------------------
def _make_tabpfn_classifier(device, seed, n_estimators):
    """tabpfn 2.x (n_estimators) vs 1.x (N_ensemble_configurations, seed)."""
    params = inspect.signature(TabPFNClassifier.__init__).parameters
    if "n_estimators" in params:
        return TabPFNClassifier(
            device=device,
            n_estimators=n_estimators,
            balance_probabilities=True,
            ignore_pretraining_limits=False,
            random_state=seed,
        )
    print(
        "  NOTE: tabpfn 1.x detected — use TabPFN_Architecture/.venv/bin/python "
        "for tabpfn 2.x (balance_probabilities, batched inference)."
    )
    return TabPFNClassifier(
        device=device,
        N_ensemble_configurations=n_estimators,
        seed=seed,
    )


def fit_bagged_tabpfn(train_X, train_y, n_bags, bag_size, n_estimators, device):
    """Fit `n_bags` TabPFN models on different stratified subsamples."""
    models = []
    used_sizes = []
    for k in range(n_bags):
        seed = RANDOM_SEED + k
        Xb, yb, _ = stratified_subsample(train_X, train_y, bag_size, seed=seed)
        used_sizes.append(len(Xb))
        print(
            f"  Bag {k+1}/{n_bags}: subsample size {len(Xb):>6} "
            f"(seed={seed}, dist={Counter(yb.tolist())})"
        )
        clf = _make_tabpfn_classifier(device, seed, n_estimators)
        t0 = time.time()
        clf.fit(Xb, yb)
        print(f"    fit_time={time.time() - t0:.1f}s")
        models.append(clf)
    return models, used_sizes


def predict_proba_batched(clf, X, batch_size=PREDICT_BATCH_SIZE):
    """predict_proba in chunks so large val/test sets fit in GPU memory."""
    n = len(X)
    if n <= batch_size:
        return clf.predict_proba(X)
    parts = []
    for start in range(0, n, batch_size):
        parts.append(clf.predict_proba(X[start : start + batch_size]))
    return np.vstack(parts)


def bagged_predict_proba(models, X, batch_label="X"):
    """Average predict_proba across all bags."""
    print(
        f"\nBagged predict_proba on {batch_label} ({len(X)} samples) "
        f"over {len(models)} bags (batch_size={PREDICT_BATCH_SIZE})..."
    )
    probs_sum = None
    total_t = 0.0
    for k, clf in enumerate(models):
        t0 = time.time()
        p = predict_proba_batched(clf, X)
        elapsed = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        total_t += elapsed
        print(f"  Bag {k+1}/{len(models)}: {elapsed:.1f}s ({1000*elapsed/max(len(X),1):.2f} ms/sample)")
        probs_sum = p if probs_sum is None else probs_sum + p
    avg = probs_sum / len(models)
    print(f"  Total inference: {total_t:.1f}s")
    return avg


# ---------------------------------------------------------------------------
# Threshold tuning on val for binary accuracy
# ---------------------------------------------------------------------------
def tune_threshold_for_accuracy(val_probs, val_y, n_steps=1001):
    """Sweep thresholds on positive-class probability; return the one that
    maximises binary accuracy on val."""
    pos = val_probs[:, POS_CLASS_INDEX]
    thresholds = np.linspace(0.0, 1.0, n_steps)
    accs = np.empty_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = (pos >= t).astype(int)
        accs[i] = accuracy_score(val_y, pred)
    best_i = int(np.argmax(accs))
    return float(thresholds[best_i]), float(accs[best_i]), thresholds, accs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_split(probs, y_true, threshold, class_names, split_name):
    pos = probs[:, POS_CLASS_INDEX]
    preds = (pos >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    auc = (
        roc_auc_score(y_true, pos)
        if len(np.unique(y_true)) > 1 and probs.shape[1] >= 2
        else float("nan")
    )
    print(f"\n{split_name} (threshold={threshold:.4f})")
    print(classification_report(y_true, preds, target_names=class_names, digits=4))
    print(f"  ROC AUC (positive='{class_names[POS_CLASS_INDEX]}'): {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    return preds, acc, auc


def plot_results(test_preds, test_y, class_names, n_train_per_bag, n_bags, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_norm = confusion_matrix(test_y, test_preds, normalize="true")
    sns.heatmap(
        cm_norm, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[0],
    )
    axes[0].set_title(
        f"Normalized Confusion Matrix\n"
        f"(TabPFN HighPerf, {n_bags} bags x {n_train_per_bag} train, n_est=8)"
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    cm_counts = confusion_matrix(test_y, test_preds)
    sns.heatmap(
        cm_counts, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (counts)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResults plot saved to: {save_path}")


def print_final_classification_report(y_true, preds, class_names, roc_auc):
    """Sklearn-style test report — printed last so it stays visible in the terminal."""
    print("\nClassification Report:", flush=True)
    print(
        classification_report(y_true, preds, target_names=class_names, digits=4),
        flush=True,
    )
    print("\nROC AUC Score:", flush=True)
    if len(np.unique(y_true)) > 1:
        print(f"  Binary (ELMy vs Suppressed): {roc_auc:.4f}", flush=True)
    else:
        print("  Only one class present in test labels; skipping AUC.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("TabPFN Plasma BINARY Classification (HIGH-PERFORMANCE config)")
    print("=" * 72)
    print("Architecture:    Bagged TabPFN (Prior-Fitted Network)")
    print(f"Bagging:         {N_BAGS} bags x {BAG_TRAIN_SIZE} stratified train per bag")
    print(f"Per-bag config:  n_estimators={TABPFN_N_ESTIMATORS}, balance_probabilities=True")
    print("Features:        ~162 tabular summary stats per 150-step causal window")
    print(f"Horizon:         {PREDICTION_HORIZON_MS} ms ahead (label at window_end + horizon)")
    print("Threshold:       Tuned on validation set for max accuracy")
    print("Split:           SHOT-BASED (70/15/15 of shots, seed 42)")
    print("Classes:         0 = Suppressed   (state == 0)")
    print("                 1 = ELMy         (state in {1, 2, 3})")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Data + windowing.
    X, y, times, shots, features, scaler = load_and_prepare_data()
    train_w, train_y, val_w, val_y, test_w, test_y = create_windows_with_shot_split(
        X, y, times, shots,
    )

    # 2. Expanded tabular features.
    print("\nExtracting tabular features (expanded set)...")
    t0 = time.time()
    train_tab, tab_feature_names = windows_to_tabular_features(train_w, features)
    val_tab, _ = windows_to_tabular_features(val_w, features)
    test_tab, _ = windows_to_tabular_features(test_w, features)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(
        f"  Tabular shapes: train={train_tab.shape}, val={val_tab.shape}, "
        f"test={test_tab.shape}"
    )
    print(f"  Tabular feature count: {len(tab_feature_names)}")

    # 3. Bagged TabPFN fit.
    print(f"\nFitting bagged TabPFN ({N_BAGS} bags x {BAG_TRAIN_SIZE} train each)...")
    t0 = time.time()
    models, used_sizes = fit_bagged_tabpfn(
        train_tab, train_y,
        n_bags=N_BAGS,
        bag_size=BAG_TRAIN_SIZE,
        n_estimators=TABPFN_N_ESTIMATORS,
        device=device,
    )
    print(f"Total bagged fit time: {time.time() - t0:.1f}s")

    # 4. Bagged predict_proba on val + test.
    val_probs = bagged_predict_proba(models, val_tab, batch_label="Validation")
    test_probs = bagged_predict_proba(models, test_tab, batch_label="Test")

    # 5. Three reports for full visibility:
    #    - Default 0.5 threshold on val and test.
    #    - Tuned threshold (on val) applied to val (sanity) and test (real).
    print("\n" + "=" * 72)
    print("Results: default threshold = 0.5 (no tuning)")
    print("=" * 72)
    _, val_acc_05, val_auc_05 = report_split(val_probs, val_y, 0.5, CLASS_NAMES, "Validation")
    _, test_acc_05, test_auc_05 = report_split(test_probs, test_y, 0.5, CLASS_NAMES, "Test")

    best_t, best_val_acc, _, _ = tune_threshold_for_accuracy(val_probs, val_y)
    print("\n" + "=" * 72)
    print(f"Results: tuned threshold from validation = {best_t:.4f} "
          f"(val acc {best_val_acc:.4f})")
    print("=" * 72)
    _, val_acc, val_auc = report_split(val_probs, val_y, best_t, CLASS_NAMES, "Validation")
    test_preds, test_acc, test_auc = report_split(
        test_probs, test_y, best_t, CLASS_NAMES, "Test"
    )

    # 6. Plot + save (before final terminal report so nothing blocks stdout).
    plot_path = f"{SAVE_DIR}/tabpfn_binary_highperf_results.png"
    plot_results(test_preds, test_y, CLASS_NAMES, BAG_TRAIN_SIZE, N_BAGS, plot_path)

    save_path = f"{SAVE_DIR}/tabpfn_binary_highperf_complete_model.pkl"
    bundle = {
        "models": models,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": features,
        "tab_feature_names": tab_feature_names,
        "window_size": WINDOW_SIZE,
        "prediction_horizon_ms": PREDICTION_HORIZON_MS,
        "windowing": "causal_past_only_label_at_end_plus_horizon",
        "class_names": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES),
        "label_mapping": {
            "suppressed_states": sorted(SUPPRESSED_STATES),
            "elmy_states": sorted(ELMY_STATES),
            "unknown_handling": "per-shot ffill+bfill of state==-1 (label padding)",
        },
        "bagging": {
            "n_bags": N_BAGS,
            "bag_train_size": BAG_TRAIN_SIZE,
            "n_estimators_per_bag": TABPFN_N_ESTIMATORS,
            "used_sizes": used_sizes,
            "balance_probabilities": True,
            "ignore_pretraining_limits": False,
        },
        "threshold_tuned": best_t,
        "val_accuracy_at_05": val_acc_05,
        "val_accuracy_tuned": val_acc,
        "test_accuracy_at_05": test_acc_05,
        "test_accuracy_tuned": test_acc,
        "val_roc_auc": val_auc,
        "test_roc_auc": test_auc,
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved bagged TabPFN bundle to: {save_path}")

    # 7. Final test report — always last in terminal (validation-tuned threshold).
    print_final_classification_report(test_y, test_preds, CLASS_NAMES, test_auc)


if __name__ == "__main__":
    main()
