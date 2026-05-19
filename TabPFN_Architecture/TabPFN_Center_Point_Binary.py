"""
TabPFN_Center_Point_Binary.py

Binary plasma-state classification with TabPFN. Companion to
`TabPFN_Center_Point_Shot_Split.py`; same data, same windowing, same
shot-based split, same window->tabular feature extraction, but the 4-way
target is collapsed to a binary one:

    0 = Suppressed   (multi-class state == 0)
    1 = ELMy         (multi-class state in {1, 2, 3}: Dithering,
                      Mitigated, ELMing)

state == -1 (unknown) is excluded, matching the multi-class script.

There is no gradient training: TabPFN does in-context learning; .fit()
just caches and preprocesses the train set, .predict() is one forward
pass conditioned on it.

Install (in the project venv):
    pip install tabpfn
"""

import time
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch  # only used for device detection
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
        "TabPFN is not installed. Install it with:\n"
        "    pip install tabpfn\n"
        "See https://github.com/PriorLabs/TabPFN for details."
    ) from e


# ---------------------------------------------------------------------------
# Reproducibility & constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 43
SHOT_SPLIT_SEED = 42  # match the multi-class shot-split script

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

# Multi-class -> binary mapping
# 0 (Suppressed)               -> 0  (Suppressed)
# 1 (Dithering), 2 (Mitigated), 3 (ELMing) -> 1 (ELMy)
# -1 (unknown edge frames)     -> filled per-shot via ffill+bfill of nearest
#                                 valid label (matches the fixed
#                                 BiLSTM_NN_Center_Point_Binary.py).
VALID_STATES = [0, 1, 2, 3]
SUPPRESSED_STATES = {0}                 # original states that mean "Suppressed"
ELMY_STATES = {1, 2, 3}                 # original states that get merged into ELMy
CLASS_NAMES = ["Suppressed", "ELMy"]
POS_CLASS_INDEX = 1                     # ELMy is the positive class for AUC

WINDOW_SIZE = 150
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# TabPFN v2 supports up to 10,000 train rows out-of-the-box. Inference cost
# scales with this number, so we cap and stratified-subsample if necessary.
TABPFN_MAX_TRAIN_SAMPLES = 10_000
TABPFN_N_ESTIMATORS = 4  # 8 = max-quality default, 4 = ~2x faster


# ---------------------------------------------------------------------------
# Data loading + binary remap
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """Load plasma data and produce binary labels (0=Suppressed, 1=ELMy).

    Matches the fixed BiLSTM_NN_Center_Point_Binary.py:
      - sort by (shot, time)
      - replace state == -1 with the nearest valid state within each shot
        (per-shot ffill then bfill on the time-sorted rows)
      - drop rows whose entire shot is unlabelled (all -1)
      - binary remap: 0 -> 0 (Suppressed); 1, 2, 3 -> 1 (ELMy).
    """
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

    n_remaining_unknown = int(df_sorted["state"].isna().sum())
    n_filled = n_unknown_pre - n_remaining_unknown
    print(f"  -1 frames filled by per-shot label padding: {n_filled:,}")
    if n_remaining_unknown:
        print(
            f"  -1 frames remaining (entire-shot unlabelled, dropped): "
            f"{n_remaining_unknown:,}"
        )

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, selected_features, scaler


# ---------------------------------------------------------------------------
# Shot-based windowing with edge-replicate padding
# ---------------------------------------------------------------------------
# Mirrors the windowing in `BiLSTM_NN_Center_Point_Binary.py` so a TabPFN
# vs BiLSTM head-to-head sees the same (X_window, y_center) samples:
#
#   - Replicate the first/last row of each shot to extend it by
#     `center_idx` on the left and `window_size - center_idx - 1` on the
#     right. Every original timestep is then a valid window center.
#   - If an entire shot is shorter than `window_size`, first expand it to
#     length `window_size` by replicating its endpoints, then run the
#     same edge-replicate procedure.
def edge_pad_shot_features(X_sub, window_size):
    """Replicate first/last rows so every original timestep can be a center.

    Returns
    -------
    X_pad : np.ndarray, shape (L + window_size - 1, n_features)
    pad_left : int
        Number of replicated rows prepended to the front of `X_sub`.
    """
    X_sub = np.asarray(X_sub, dtype=np.float64)
    L, _ = X_sub.shape
    center_idx = window_size // 2
    pad_left = center_idx
    pad_right = window_size - center_idx - 1
    if L == 0:
        return X_sub.astype(np.float32), pad_left
    left = np.repeat(X_sub[:1], pad_left, axis=0)
    right = np.repeat(X_sub[-1:], pad_right, axis=0)
    X_pad = np.vstack([left, X_sub, right]).astype(np.float32)
    return X_pad, pad_left


def create_windows_for_shots(X, y, shots, shot_list, window_size=WINDOW_SIZE):
    """Edge-padded center-labelled windows for the given list of shots.

    One window is produced per original timestep in each shot, so the count
    matches `BiLSTM_NN_Center_Point_Binary.py` exactly.
    """
    windows, labels = [], []
    center_idx = window_size // 2

    for shot_id in shot_list:
        shot_indices = np.where(shots == shot_id)[0]
        if len(shot_indices) == 0:
            continue

        X_sub = X[shot_indices]
        y_sub = y[shot_indices]
        L = len(shot_indices)

        if L < window_size:
            pad_extra = window_size - L
            left_extra = pad_extra // 2
            right_extra = pad_extra - left_extra
            first, last = X_sub[:1], X_sub[-1:]
            X_sub = np.vstack(
                [
                    np.repeat(first, left_extra, axis=0),
                    X_sub,
                    np.repeat(last, right_extra, axis=0),
                ]
            )
            y_sub = np.concatenate(
                [
                    np.repeat(y_sub[:1], left_extra),
                    y_sub,
                    np.repeat(y_sub[-1:], right_extra),
                ]
            )
            L = len(X_sub)

        X_pad, pad_left = edge_pad_shot_features(X_sub, window_size)

        for t in range(L):
            start = pad_left + t - center_idx
            window = X_pad[start : start + window_size]
            center_label = y_sub[t]

            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(center_label)

    if len(windows) == 0:
        return np.array([]), np.array([])

    return np.asarray(windows, dtype=np.float32), np.asarray(labels)


def create_windows_with_shot_split(
    X, y, shots, window_size=WINDOW_SIZE, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
):
    """Create windows with a shot-based train/val/test split (no leakage)."""
    print(f"Creating windows of size {window_size} with SHOT-BASED split...")

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

    print("\nShot split:")
    print(f"  Train shots: {len(train_shots)}")
    print(f"  Val shots:   {len(val_shots)}")
    print(f"  Test shots:  {len(test_shots)}")

    print("\nCreating windows for each split...")
    train_w, train_y = create_windows_for_shots(X, y, shots, train_shots, window_size)
    val_w, val_y = create_windows_for_shots(X, y, shots, val_shots, window_size)
    test_w, test_y = create_windows_for_shots(X, y, shots, test_shots, window_size)

    print("\nWindows created:")
    print(f"  Train: {len(train_w)}")
    print(f"  Val:   {len(val_w)}")
    print(f"  Test:  {len(test_w)}")

    print("\nLabel distributions:")
    print(f"  Train: {Counter(train_y)}")
    print(f"  Val:   {Counter(val_y)}")
    print(f"  Test:  {Counter(test_y)}")

    return train_w, train_y, val_w, val_y, test_w, test_y


# ---------------------------------------------------------------------------
# Window -> tabular feature engineering for TabPFN
# ---------------------------------------------------------------------------
def windows_to_tabular_features(windows, feature_names):
    """Summarise each (window_size, n_features) window as a flat feature vector.

    13 stats per raw feature: mean, std, min, max, median, q25, q75, first,
    last, center, range, delta (last-first), linear slope. With 6 raw
    features this yields 78 tabular features.

    The center value is included on purpose because the label is the
    state at the center of the window.
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected windows of shape (N, T, F); got {windows.shape}")

    n_windows, win_size, _ = windows.shape
    center = win_size // 2

    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    medians = np.median(windows, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    q75 = np.percentile(windows, 75, axis=1)
    firsts = windows[:, 0, :]
    lasts = windows[:, -1, :]
    centers = windows[:, center, :]
    ranges = maxs - mins
    deltas = lasts - firsts

    # Linear slope per feature across time, computed analytically.
    t = np.arange(win_size, dtype=np.float32)
    t_centered = t - t.mean()
    var_t = float((t_centered ** 2).mean())
    x_centered = windows - means[:, None, :]
    cov_tx = (t_centered[None, :, None] * x_centered).mean(axis=1)
    slopes = cov_tx / var_t

    stat_blocks = {
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "median": medians,
        "q25": q25,
        "q75": q75,
        "first": firsts,
        "last": lasts,
        "center": centers,
        "range": ranges,
        "delta": deltas,
        "slope": slopes,
    }

    blocks, names = [], []
    for stat_name, arr in stat_blocks.items():
        blocks.append(arr.astype(np.float32))
        names.extend([f"{f}__{stat_name}" for f in feature_names])

    return np.concatenate(blocks, axis=1), names


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------
def stratified_subsample(X, y, max_samples, seed=RANDOM_SEED):
    """Return a class-balanced random subsample of (X, y) of size <= max_samples."""
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
# Evaluation (binary)
# ---------------------------------------------------------------------------
def evaluate_split(model, X, y_true, class_names, split_name):
    """Run predict_proba and report binary classification metrics."""
    print(f"\nPredicting on {split_name} set ({len(X)} samples)...")
    t0 = time.time()
    probs = model.predict_proba(X)
    elapsed = time.time() - t0
    preds = np.argmax(probs, axis=1)
    print(f"  Inference time: {elapsed:.1f}s ({1000 * elapsed / max(len(X), 1):.2f} ms/sample)")

    print(f"\n{split_name} Classification Report:")
    print(classification_report(y_true, preds, target_names=class_names, digits=4))

    # Single ROC-AUC for binary, with ELMy (1) as positive class.
    if len(np.unique(y_true)) > 1 and probs.shape[1] >= 2:
        auc = roc_auc_score(y_true, probs[:, POS_CLASS_INDEX])
        print(f"{split_name} ROC AUC (positive class = '{class_names[POS_CLASS_INDEX]}'): {auc:.4f}")
    else:
        auc = float("nan")
        print(f"{split_name} ROC AUC: undefined (only one class present)")

    acc = accuracy_score(y_true, preds)
    print(f"{split_name} Accuracy: {acc:.4f}")
    return preds, probs, acc, auc


def plot_results(test_preds, test_y, class_names, n_train_used, save_path):
    """Plot normalised + count confusion matrices for the binary test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_norm = confusion_matrix(test_y, test_preds, normalize="true")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Normalized Confusion Matrix\n(TabPFN binary, {n_train_used} train samples)"
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_counts = confusion_matrix(test_y, test_preds)
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (counts)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nResults plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TabPFN Plasma BINARY Classification (shot-based split)")
    print("=" * 60)
    print("Architecture: TabPFN (Prior-Fitted Network, in-context learning)")
    print("Window: 150 timesteps -> 78 tabular summary-statistics features")
    print("Split:  SHOT-BASED (no shot appears in more than one split)")
    print("Classes: 0 = Suppressed (state == 0)")
    print("         1 = ELMy       (state in {1=Dithering, 2=Mitigated, 3=ELMing})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X, y, shots, features, scaler = load_and_prepare_data()

    train_w, train_y, val_w, val_y, test_w, test_y = create_windows_with_shot_split(
        X, y, shots,
        window_size=WINDOW_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    print("\nConverting windows to tabular features...")
    t0 = time.time()
    train_tab, tab_feature_names = windows_to_tabular_features(train_w, features)
    val_tab, _ = windows_to_tabular_features(val_w, features)
    test_tab, _ = windows_to_tabular_features(test_w, features)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(f"  Tabular shape: train={train_tab.shape}, val={val_tab.shape}, test={test_tab.shape}")
    print(f"  Tabular feature count: {len(tab_feature_names)}")

    n_full_train = len(train_tab)
    train_tab_used, train_y_used, _ = stratified_subsample(
        train_tab, train_y, TABPFN_MAX_TRAIN_SAMPLES
    )
    if len(train_tab_used) < n_full_train:
        print(
            f"\nSubsampled training data: {n_full_train} -> {len(train_tab_used)} "
            f"(stratified, capped at TABPFN_MAX_TRAIN_SAMPLES={TABPFN_MAX_TRAIN_SAMPLES})"
        )
    else:
        print(f"\nUsing full training set: {len(train_tab_used)} samples")
    print(f"  Used label distribution: {Counter(train_y_used)}")

    print("\nFitting TabPFN (in-context learning, no gradient descent)...")
    t0 = time.time()
    model = TabPFNClassifier(
        device=device,
        n_estimators=TABPFN_N_ESTIMATORS,
        ignore_pretraining_limits=False,
        random_state=RANDOM_SEED,
    )
    model.fit(train_tab_used, train_y_used)
    print(f"  Fit time: {time.time() - t0:.1f}s")

    val_preds, val_probs, val_acc, val_auc = evaluate_split(
        model, val_tab, val_y, CLASS_NAMES, "Validation"
    )
    test_preds, test_probs, test_acc, test_auc = evaluate_split(
        model, test_tab, test_y, CLASS_NAMES, "Test"
    )

    plot_path = f"{SAVE_DIR}/tabpfn_binary_shot_split_results.png"
    plot_results(test_preds, test_y, CLASS_NAMES, len(train_tab_used), plot_path)

    save_path = f"{SAVE_DIR}/tabpfn_binary_complete_model.pkl"
    bundle = {
        "model": model,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": features,
        "tab_feature_names": tab_feature_names,
        "window_size": WINDOW_SIZE,
        "class_names": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES),
        "label_mapping": {
            "suppressed_states": sorted(SUPPRESSED_STATES),
            "elmy_states": sorted(ELMY_STATES),
        },
        "n_train_used": len(train_tab_used),
        "n_estimators": TABPFN_N_ESTIMATORS,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "val_roc_auc": val_auc,
        "test_roc_auc": test_auc,
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved TabPFN binary bundle to: {save_path}")

    print("\n" + "=" * 60)
    print(f"Final Test Accuracy (TabPFN binary): {test_acc:.4f}")
    print(f"Final Test ROC AUC  (TabPFN binary): {test_auc:.4f}")
    print(f"Final Val  Accuracy (TabPFN binary): {val_acc:.4f}")
    print(f"Final Val  ROC AUC  (TabPFN binary): {val_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
