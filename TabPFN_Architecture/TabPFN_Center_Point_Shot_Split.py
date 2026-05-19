"""
TabPFN_Center_Point_Shot_Split.py

Plasma state classification with TabPFN, mirroring the pipeline of
`BiLSTM_NN_Architecture/BiLSTM_NN_Center_Point_Shot_Split.py`.

Goal (same as the BiLSTM script):
    Predict the plasma state (0=Suppressed, 1=Dithering, 2=Mitigated, 3=ELMing)
    at the center of a sliding window of length 150 over a fixed set of plasma
    diagnostics, with a shot-based train/val/test split so that no shot appears
    in more than one split.

Difference from the BiLSTM script:
    - Architecture: TabPFN (a pre-trained Prior-Fitted Network for tabular data)
      instead of BiLSTM + NN with attention.
    - TabPFN does not consume sequences. Each window is summarised into a
      compact tabular feature vector via per-feature summary statistics
      (mean, std, min, max, percentiles, first/last/center values, range,
      delta, linear slope). This preserves global trend, variability, and the
      center-point value used for labelling.
    - There is no gradient-descent training. TabPFN does in-context learning:
      `.fit(X, y)` caches the train set, and predictions are produced by a
      single forward pass conditioned on it.

Install:
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

import torch  # only used for device detection (TabPFN handles the rest)
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
# Reproducibility & constants (kept consistent with the BiLSTM script)
# ---------------------------------------------------------------------------
RANDOM_SEED = 43
SHOT_SPLIT_SEED = 42  # the BiLSTM script reseeds before shuffling shots

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
CLASS_NAMES = ["Suppressed", "Dithering", "Mitigated", "ELMing"]

WINDOW_SIZE = 150
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# TabPFN v2.5 supports up to 50,000 train rows. We default to a smaller cap
# because inference cost scales with train-set size; raise this for max accuracy.
TABPFN_MAX_TRAIN_SAMPLES = 10_000
TABPFN_N_ESTIMATORS = 4  # 8 = max-quality default, 4 = ~2x faster


# ---------------------------------------------------------------------------
# Data loading and shot-based windowing (identical logic to the BiLSTM script)
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """Load and preprocess the plasma data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    df = df[df["shot"] != PROBLEM_SHOT].copy()

    selected_features = [f for f in IMPORTANT_FEATURES if f in df.columns]
    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted["state"].isin(VALID_STATES)].copy()

    X = df_filtered[selected_features].values
    y = df_filtered["state"].values
    shots = df_filtered["shot"].values

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {Counter(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, selected_features, scaler


def create_windows_for_shots(X, y, shots, shot_list, window_size=WINDOW_SIZE):
    """Create center-point-labelled windows for the given list of shots."""
    windows, labels = [], []
    center_idx = window_size // 2

    for shot_id in shot_list:
        shot_indices = np.where(shots == shot_id)[0]
        if len(shot_indices) < window_size:
            continue

        for i in range(len(shot_indices) - window_size + 1):
            start = shot_indices[i]
            end = start + window_size
            if end > shot_indices[-1] + 1:
                break

            window = X[start:end]
            center_label = y[start + center_idx]

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

    For every raw feature we extract: mean, std, min, max, median, q25, q75,
    first, last, center, range (max-min), delta (last-first), and linear slope
    over the window. With 6 raw features this yields ~78 tabular features.

    The center value is included because the BiLSTM script labels each window
    by its center point; it is the strongest single predictor and we keep it.
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

    # Linear slope per feature across time, computed analytically:
    # slope = mean_t[(t - mean_t) * (x - mean_x)] / mean_t[(t - mean_t)^2]
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

    tab_X = np.concatenate(blocks, axis=1)
    return tab_X, names


# ---------------------------------------------------------------------------
# Subsampling (TabPFN inference cost grows with the train set)
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
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_split(model, X, y_true, class_names, split_name):
    """Run predict_proba and report classification metrics."""
    print(f"\nPredicting on {split_name} set ({len(X)} samples)...")
    t0 = time.time()
    probs = model.predict_proba(X)
    elapsed = time.time() - t0
    preds = np.argmax(probs, axis=1)
    print(f"  Inference time: {elapsed:.1f}s ({1000 * elapsed / max(len(X), 1):.2f} ms/sample)")

    print(f"\n{split_name} Classification Report:")
    print(classification_report(y_true, preds, target_names=class_names, digits=4))

    print(f"{split_name} ROC AUC (one-vs-rest):")
    for i, name in enumerate(class_names):
        if i < probs.shape[1]:
            cls_y = (y_true == i).astype(int)
            if len(np.unique(cls_y)) > 1:
                auc = roc_auc_score(cls_y, probs[:, i])
                print(f"  {name}: {auc:.4f}")

    acc = accuracy_score(y_true, preds)
    print(f"\n{split_name} Accuracy: {acc:.4f}")
    return preds, probs, acc


def plot_results(test_preds, test_y, class_names, n_train_used, save_path):
    """Plot normalized + counts confusion matrices for the test set."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cm_norm = confusion_matrix(test_y, test_preds, normalize="true")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(f"Normalized Confusion Matrix\n(TabPFN, {n_train_used} train samples)")
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
    print("TabPFN Plasma Classification (shot-based split)")
    print("=" * 60)
    print("Architecture: TabPFN (Prior-Fitted Network, in-context learning)")
    print("Window: 150 timesteps -> tabular summary-statistics features")
    print("Split:  SHOT-BASED (no shot appears in more than one split)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Same data loading + scaling as the BiLSTM script.
    X, y, shots, features, scaler = load_and_prepare_data()

    # 2. Same shot-based windowing.
    train_w, train_y, val_w, val_y, test_w, test_y = create_windows_with_shot_split(
        X, y, shots,
        window_size=WINDOW_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # 3. Convert sequence windows -> tabular features.
    print("\nConverting windows to tabular features...")
    t0 = time.time()
    train_tab, tab_feature_names = windows_to_tabular_features(train_w, features)
    val_tab, _ = windows_to_tabular_features(val_w, features)
    test_tab, _ = windows_to_tabular_features(test_w, features)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(f"  Tabular shape: train={train_tab.shape}, val={val_tab.shape}, test={test_tab.shape}")
    print(f"  Tabular feature count: {len(tab_feature_names)}")

    # 4. Stratified subsample of training data (TabPFN train-size budget).
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

    # 5. Fit TabPFN. There is no gradient training here; .fit caches the data
    #    and the actual prediction is a single conditioned forward pass.
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

    # 6. Evaluate.
    val_preds, val_probs, val_acc = evaluate_split(
        model, val_tab, val_y, CLASS_NAMES, "Validation"
    )
    test_preds, test_probs, test_acc = evaluate_split(
        model, test_tab, test_y, CLASS_NAMES, "Test"
    )

    # 7. Plot test results.
    plot_path = f"{SAVE_DIR}/tabpfn_shot_split_results.png"
    plot_results(test_preds, test_y, CLASS_NAMES, len(train_tab_used), plot_path)

    # 8. Save the full bundle (model + preprocessing + metadata).
    save_path = f"{SAVE_DIR}/tabpfn_complete_model.pkl"
    bundle = {
        "model": model,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": features,
        "tab_feature_names": tab_feature_names,
        "window_size": WINDOW_SIZE,
        "class_names": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES),
        "n_train_used": len(train_tab_used),
        "n_estimators": TABPFN_N_ESTIMATORS,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved TabPFN bundle to: {save_path}")

    print("\n" + "=" * 60)
    print(f"Final Test Accuracy (TabPFN): {test_acc:.4f}")
    print(f"Final Val  Accuracy (TabPFN): {val_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
