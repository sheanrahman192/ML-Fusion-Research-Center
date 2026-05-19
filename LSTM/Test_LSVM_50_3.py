import argparse
import warnings
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore")

# K-fold CV (shot-level; windows never cross shots)
N_FOLDS_DEFAULT = 5
CHECKPOINT_BASENAME = "best_test_sgdc_50_3"
RESULTS_PLOT_BASENAME = "test_sgdc_50_3_results"

# Set random seeds for reproducibility
np.random.seed(42)

# Window configuration (time-based, assuming ~1 ms per sample)
WINDOW_MS = 200
BRANCH1_MS = 150

WINDOW_SIZE = WINDOW_MS
BRANCH1_LEN = BRANCH1_MS

# 3-state classification: Suppressed, Dithering/Mitigated (combined), ELMing
N_CLASSES = 3


def load_and_prepare_data():
    """
    Load and preprocess plasma data for SGDClassifier.

    Short branch (6 features, 150 ms): iln3iamp, betan, density, li,
        fs_sum_past_max_smoothed, n_eped
    Long branch (9 features, 200 ms): pinj, tijnj, echpwrc, I_ECCD, tritop,
        tribot, Ip, bt, gasa
    """
    print("Loading data for SGDClassifier (dual-window flattened features)...")
    df = pd.read_csv("/mnt/homes/sr4240/my_folder/plasma_data.csv")

    df = df[df["shot"] != 191675].copy()

    branch_short_features = [
        "iln3iamp",
        "betan",
        "density",
        "li",
        "fs_sum_past_max_smoothed",
        "n_eped",
    ]
    branch_long_features = [
        "pinj",
        "tijnj",
        "echpwrc",
        "I_ECCD",
        "tritop",
        "tribot",
        "Ip",
        "bt",
        "gasa",
    ]

    missing_s = [f for f in branch_short_features if f not in df.columns]
    missing_l = [f for f in branch_long_features if f not in df.columns]
    if missing_s or missing_l:
        raise ValueError(
            f"Missing required columns in plasma_data.csv. "
            f"Short branch missing: {missing_s}, Long branch missing: {missing_l}"
        )

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    Xs = df_sorted[branch_short_features].values
    Xl = df_sorted[branch_long_features].values
    y = df_sorted["state"].values
    times = df_sorted["time"].values
    shots = df_sorted["shot"].values

    valid_mask = ~np.isnan(Xs).any(axis=1) & ~np.isnan(y) & ~np.isnan(times)
    Xs = Xs[valid_mask]
    Xl = Xl[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    Xl_imputed = Xl.copy()
    for j in range(Xl_imputed.shape[1]):
        col = Xl_imputed[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[nan_mask] = 0.0
        elif nan_mask.any():
            col[nan_mask] = np.nanmean(col)
        Xl_imputed[:, j] = col
    Xl = Xl_imputed

    print("Data shape after cleaning (short valid mask + long imputation):")
    print(f"  Short branch: {Xs.shape}")
    print(f"  Long branch:  {Xl.shape}")
    print(f"  Labels: {y.shape}")
    print(f"\nRaw label distribution (4-state): {Counter(y)}")

    # Unscaled; StandardScaler is fit per CV fold (or single split) on train shots only.
    return (
        Xs.astype(np.float32),
        Xl.astype(np.float32),
        y,
        times,
        shots,
        branch_short_features,
        branch_long_features,
    )


def scale_features_from_train_shots(Xs, Xl, shots, train_shots):
    """Fit StandardScalers on training-shot rows only; transform all rows."""
    train_shots = set(train_shots)
    mask = np.isin(shots, list(train_shots))
    if not np.any(mask):
        raise ValueError("No rows from train_shots; cannot fit scalers.")
    scaler_s = StandardScaler()
    scaler_l = StandardScaler()
    scaler_s.fit(Xs[mask])
    scaler_l.fit(Xl[mask])
    Xs_scaled = scaler_s.transform(Xs).astype(np.float32)
    Xl_scaled = scaler_l.transform(Xl).astype(np.float32)
    return Xs_scaled, Xl_scaled


def create_dual_windows_by_shot(
    Xs,
    Xl,
    y,
    times,
    shots,
    train_shots,
    val_shots,
    test_shots,
    window_size: int = WINDOW_SIZE,
    branch1_len: int = BRANCH1_LEN,
):
    """
    Build separate tensors per window:
      - x_short: (150, n_short) — first 150 steps in the 200-step context
      - x_long:  (200, n_long) — full window on long features

    Label: state at the end of the 200-step window (3-state mapping).
    """
    train_shots = set(train_shots)
    val_shots = set(val_shots)
    test_shots = set(test_shots)

    n_short = Xs.shape[1]
    n_long = Xl.shape[1]
    print(
        f"\nCreating dual-input windows (long window={window_size} steps, "
        f"short branch length={branch1_len})..."
    )
    print(
        f"Short: {n_short} features x {branch1_len} ms; "
        f"Long: {n_long} features x {window_size} ms"
    )
    print("Split is BY SHOT NUMBER (not individual data points).")

    unique_shots = np.unique(shots)
    print(f"Total unique shots in data: {len(unique_shots)}")
    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")

    label_mapping = {0: 0, 1: 1, 2: 1, 3: 2}
    valid_raw_labels = {0, 1, 2, 3}

    train_xs, train_xl, train_y = [], [], []
    val_xs, val_xl, val_y = [], [], []
    test_xs, test_xl, test_y = [], [], []

    windows_created = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        if shot_id in train_shots:
            txs, txl, ty = train_xs, train_xl, train_y
        elif shot_id in val_shots:
            txs, txl, ty = val_xs, val_xl, val_y
        elif shot_id in test_shots:
            txs, txl, ty = test_xs, test_xl, test_y
        else:
            continue

        shot_Xs = Xs[shot_indices]
        shot_Xl = Xl[shot_indices]
        shot_y = y[shot_indices]

        for end_idx in range(window_size - 1, len(shot_indices)):
            start_idx = end_idx - window_size + 1
            raw_label = int(shot_y[end_idx])
            if raw_label not in valid_raw_labels:
                continue

            mapped_label = label_mapping[raw_label]
            w_short = shot_Xs[start_idx : start_idx + branch1_len].astype(
                np.float32, copy=False
            )
            w_long = shot_Xl[start_idx : end_idx + 1].astype(np.float32, copy=False)

            txs.append(w_short)
            txl.append(w_long)
            ty.append(mapped_label)
            windows_created += 1

    train_xs = np.array(train_xs, dtype=np.float32)
    train_xl = np.array(train_xl, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int64)
    val_xs = np.array(val_xs, dtype=np.float32)
    val_xl = np.array(val_xl, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.int64)
    test_xs = np.array(test_xs, dtype=np.float32)
    test_xl = np.array(test_xl, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int64)

    print("\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    if len(train_xs) > 0:
        print(f"  Short window shape: {train_xs.shape[1:]}")
        print(f"  Long window shape:  {train_xl.shape[1:]}")
    print("\nCreated windows:")
    print(f"  Train: {len(train_xs)}")
    print(f"  Val:   {len(val_xs)}")
    print(f"  Test:  {len(test_xs)}")

    print("\nLabel distribution (3-state):")
    print(f"  Train: {Counter(train_y)}")
    print(f"  Val:   {Counter(val_y)}")
    print(f"  Test:  {Counter(test_y)}")

    return (
        train_xs,
        train_xl,
        train_y,
        val_xs,
        val_xl,
        val_y,
        test_xs,
        test_xl,
        test_y,
    )


def flatten_dual_windows(x_short, x_long):
    """Flatten and concatenate short and long windows for linear classifier input."""
    if len(x_short) == 0 or len(x_long) == 0:
        return np.empty((0, BRANCH1_LEN * 6 + WINDOW_SIZE * 9), dtype=np.float32)
    short_flat = x_short.reshape(x_short.shape[0], -1)
    long_flat = x_long.reshape(x_long.shape[0], -1)
    return np.concatenate([short_flat, long_flat], axis=1).astype(np.float32, copy=False)


def fit_sgdc_with_validation(train_x, train_y, val_x, val_y, alpha_values, max_iter, tol):
    """Select alpha using validation accuracy and return best trained SGDClassifier."""
    best_model = None
    best_alpha = None
    best_val_acc = -np.inf
    per_alpha_metrics = []

    for alpha in alpha_values:
        model = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=float(alpha),
            max_iter=max_iter,
            tol=tol,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        val_preds = model.predict(val_x)
        val_acc = float(accuracy_score(val_y, val_preds))
        per_alpha_metrics.append((float(alpha), val_acc))
        print(f"  alpha={alpha}: val accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_alpha = float(alpha)
            best_model = model

    print(f"Best alpha from validation: {best_alpha} (val accuracy={best_val_acc:.4f})")
    return best_model, best_alpha, per_alpha_metrics


def evaluate_model(model, test_x, test_y, class_names):
    preds = model.predict(test_x)
    decision_scores = model.decision_function(test_x)

    print("\nClassification Report (3-state):")
    print(
        classification_report(
            test_y,
            preds,
            target_names=class_names,
            labels=[0, 1, 2],
            digits=4,
            zero_division=0,
        )
    )

    test_acc = float(accuracy_score(test_y, preds))
    print(f"\nTest Accuracy: {test_acc:.4f}")

    print("\nROC AUC Scores:")
    if decision_scores.ndim == 1:
        # Binary fallback; not expected for this 3-class setup.
        decision_scores = np.column_stack([-decision_scores, decision_scores])

    for i, class_name in enumerate(class_names):
        if i < decision_scores.shape[1]:
            binary_labels = (test_y == i).astype(int)
            if len(np.unique(binary_labels)) > 1:
                auc = roc_auc_score(binary_labels, decision_scores[:, i])
                print(f"  {class_name}: {auc:.4f}")
            else:
                print(f"  {class_name}: N/A (single class in test labels)")

    return preds, test_y, decision_scores, test_acc


def plot_results(all_preds, all_labels, class_names, alpha_metrics, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    if alpha_metrics:
        alphas = [str(a) for a, _ in alpha_metrics]
        vals = [acc for _, acc in alpha_metrics]
        axes[0, 0].bar(alphas, vals, color="steelblue")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_xlabel("alpha")
        axes[0, 0].set_ylabel("Validation Accuracy")
        axes[0, 0].set_title("SGDClassifier Validation Accuracy by alpha")
        axes[0, 0].grid(True, axis="y", alpha=0.3)
    else:
        axes[0, 0].axis("off")

    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics_text = (
        f"Accuracy: {report['accuracy']:.4f}\n\n"
        f"Macro Precision: {report['macro avg']['precision']:.4f}\n"
        f"Macro Recall: {report['macro avg']['recall']:.4f}\n"
        f"Macro F1: {report['macro avg']['f1-score']:.4f}"
    )
    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.05,
        0.95,
        metrics_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    axes[0, 1].set_title("SGDClassifier Summary Metrics")

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2], normalize="true")
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Normalized Confusion Matrix (3-state)")
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")

    cm_counts = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Confusion Matrix (Counts, 3-state)")
    axes[1, 1].set_ylabel("True Label")
    axes[1, 1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Results saved to '{save_path}'")


def single_split_shot_sets(unique_shots, seed: int = 42):
    """Original 70% / 15% / 15% random shot split."""
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_shots)
    n = len(shuffled)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    train_shots = set(shuffled[:train_size])
    val_shots = set(shuffled[train_size : train_size + val_size])
    test_shots = set(shuffled[train_size + val_size :])
    return train_shots, val_shots, test_shots


def run_training_pipeline(
    Xs,
    Xl,
    y,
    times,
    shots,
    train_shots,
    val_shots,
    test_shots,
    checkpoint_path: str,
    plot_path: str,
    alpha_values,
    max_iter: int,
    tol: float,
):
    Xs_s, Xl_s = scale_features_from_train_shots(Xs, Xl, shots, train_shots)
    (
        train_xs,
        train_xl,
        train_y,
        val_xs,
        val_xl,
        val_y,
        test_xs,
        test_xl,
        test_y,
    ) = create_dual_windows_by_shot(
        Xs_s,
        Xl_s,
        y,
        times,
        shots,
        train_shots,
        val_shots,
        test_shots,
        window_size=WINDOW_SIZE,
        branch1_len=BRANCH1_LEN,
    )

    print("\nFinal dataset sizes:")
    print(f"  Train: {len(train_xs)} samples")
    print(f"  Val:   {len(val_xs)} samples")
    print(f"  Test:  {len(test_xs)} samples")

    train_x = flatten_dual_windows(train_xs, train_xl)
    val_x = flatten_dual_windows(val_xs, val_xl)
    test_x = flatten_dual_windows(test_xs, test_xl)

    print(f"Flattened feature size per window: {train_x.shape[1] if len(train_x) else 0}")

    print("\nTraining SGDClassifier (hinge) with validation-based alpha selection...")
    model, best_alpha, alpha_metrics = fit_sgdc_with_validation(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        alpha_values=alpha_values,
        max_iter=max_iter,
        tol=tol,
    )

    artifact = {
        "model": model,
        "best_alpha": best_alpha,
        "window_ms": WINDOW_MS,
        "branch1_ms": BRANCH1_MS,
        "n_classes": N_CLASSES,
        "class_names": ["Suppressed", "Dithering/Mitigated", "ELMing"],
    }
    joblib.dump(artifact, checkpoint_path)
    print(f"Saved SGDClassifier checkpoint to '{checkpoint_path}'")

    class_names = artifact["class_names"]
    all_preds, all_labels, _, final_acc = evaluate_model(model, test_x, test_y, class_names)
    plot_results(
        all_preds=all_preds,
        all_labels=all_labels,
        class_names=class_names,
        alpha_metrics=alpha_metrics,
        save_path=plot_path,
    )
    return final_acc


def parse_args():
    p = argparse.ArgumentParser(
        description="Dual-window SGDClassifier 3-state plasma classifier (shot-level CV or single split)."
    )
    p.add_argument(
        "--single-split",
        action="store_true",
        help="Use one random 70%%/15%%/15%% shot split instead of K-fold CV.",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS_DEFAULT,
        help=f"Number of K-folds (shot-level). Default: {N_FOLDS_DEFAULT}.",
    )
    p.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[1e-5, 3e-5, 1e-4, 3e-4],
        help="Candidate alpha values for SGDClassifier validation selection.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for SGDClassifier.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for SGDClassifier.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("SGDClassifier Model for Plasma State Classification (3-state)")
    print("=" * 60)
    print("Prediction target: CURRENT time's state (no future horizon)")
    print(
        f"Input windows: short {BRANCH1_MS} ms x 6 features, "
        f"long {WINDOW_MS} ms x 9 features"
    )
    print("Model: flatten(short,long) -> SGDClassifier(loss='hinge', class_weight=balanced)")
    print("Split: BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    (
        Xs,
        Xl,
        y,
        times,
        shots,
        branch_short_features,
        branch_long_features,
    ) = load_and_prepare_data()

    print(
        f"Feature groups: short={len(branch_short_features)} ({branch_short_features}), "
        f"long={len(branch_long_features)} ({branch_long_features})"
    )

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    if args.single_split:
        train_shots, val_shots, test_shots = single_split_shot_sets(unique_shots, seed=42)
        final_acc = run_training_pipeline(
            Xs,
            Xl,
            y,
            times,
            shots,
            train_shots,
            val_shots,
            test_shots,
            checkpoint_path=f"{CHECKPOINT_BASENAME}.pkl",
            plot_path=f"{RESULTS_PLOT_BASENAME}.png",
            alpha_values=args.alpha_values,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        print("\n" + "=" * 60)
        print(
            f"Training Complete! SGDClassifier (current state, {WINDOW_MS} ms context) - "
            f"Test Accuracy: {final_acc:.4f}"
        )
        print("=" * 60)
        return

    n_folds = max(2, args.folds)
    if n_shots < n_folds:
        raise ValueError(
            f"Need at least {n_folds} unique shots for {n_folds}-fold CV; got {n_shots}."
        )

    print(f"\nK-fold cross-validation: {n_folds} folds, shot-level.")
    print(
        "Per fold: test = one fold of shots; remaining shots split ~87.5% train / "
        "~12.5% val for alpha selection."
    )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(shuffled_shots)):
        test_shots = set(shuffled_shots[test_idx])
        remaining = shuffled_shots[train_val_idx]
        n_rem = len(remaining)
        train_size = int(0.875 * n_rem)
        if train_size < 1:
            train_size = 1
        if n_rem > 1 and train_size >= n_rem:
            train_size = n_rem - 1

        train_shots = set(remaining[:train_size])
        val_shots = set(remaining[train_size:])
        if len(val_shots) == 0 and len(train_shots) > 1:
            move = next(iter(train_shots))
            train_shots.remove(move)
            val_shots.add(move)

        print("\n" + "=" * 60)
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(
            f"  Shots - Train: {len(train_shots)}, Val: {len(val_shots)}, Test: {len(test_shots)}"
        )
        print("=" * 60)

        ckpt = f"{CHECKPOINT_BASENAME}_fold{fold_idx + 1}.pkl"
        plot_path = f"{RESULTS_PLOT_BASENAME}_fold{fold_idx + 1}.png"

        fold_acc = run_training_pipeline(
            Xs,
            Xl,
            y,
            times,
            shots,
            train_shots,
            val_shots,
            test_shots,
            checkpoint_path=ckpt,
            plot_path=plot_path,
            alpha_values=args.alpha_values,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        fold_accs.append(fold_acc)
        print(f"Fold {fold_idx + 1} test accuracy: {fold_acc:.4f}")

    fold_accs = np.asarray(fold_accs, dtype=np.float64)
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0

    print("\n" + "=" * 60)
    print(f"Cross-validation complete ({n_folds} folds, shot-level)")
    print(f"  Test accuracy per fold: {fold_accs}")
    print(f"  Mean +- std: {mean_acc:.4f} +- {std_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
