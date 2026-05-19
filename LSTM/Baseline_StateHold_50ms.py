import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PREDICTION_HORIZON_MS = 50
WINDOW_SIZE = 150
RANDOM_SEED = 42


def _get_task_spec(task: str):
    task = task.lower().strip()
    if task in {"2", "2state", "binary_supp_vs_other", "supp_vs_other"}:
        # Matches `LSTM_50_2_Random_Shot.py`
        label_mapping = {0: 0, 1: 1, 2: 1, 3: 1}
        class_names = ["Suppressed", "Other"]
        labels = [0, 1]
        title = "2-state (Suppressed vs Other)"
        out_suffix = "2state"
        return label_mapping, class_names, labels, title, out_suffix

    if task in {"3", "3state", "tri"}:
        # Matches `LSTM_50_3_Random_Shot.py`
        label_mapping = {0: 0, 1: 1, 2: 1, 3: 2}
        class_names = ["Suppressed", "Dithering/Mitigated", "ELMing"]
        labels = [0, 1, 2]
        title = "3-state (Suppressed vs Dithering/Mitigated vs ELMing)"
        out_suffix = "3state"
        return label_mapping, class_names, labels, title, out_suffix

    if task in {"4", "4state", "raw"}:
        # Matches `LSTM_50_Random_Shot.py`
        label_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        class_names = ["Suppressed", "Dithering", "Mitigated", "ELMing"]
        labels = [0, 1, 2, 3]
        title = "4-state (raw states)"
        out_suffix = "4state"
        return label_mapping, class_names, labels, title, out_suffix

    raise ValueError(
        f"Unknown task '{task}'. Use one of: 2state, 3state, 4state."
    )


def load_state_time_shot(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df["shot"] != 191675].copy()
    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    y = df_sorted["state"].values
    times = df_sorted["time"].values
    shots = df_sorted["shot"].values

    valid_mask = ~np.isnan(y) & ~np.isnan(times) & ~np.isnan(shots)
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    return y, times, shots


def make_random_shot_split(shots: np.ndarray, seed: int = RANDOM_SEED):
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_shots)

    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)

    train_shots = set(shuffled[:train_size])
    val_shots = set(shuffled[train_size : train_size + val_size])
    test_shots = set(shuffled[train_size + val_size :])

    return unique_shots, train_shots, val_shots, test_shots


def baseline_predictions_for_shot(
    shot_times: np.ndarray,
    shot_labels: np.ndarray,
    *,
    window_size: int,
    horizon_ms: int,
    label_mapping: dict,
):
    """
    Vectorized baseline over all windows in a shot:
    - current state = state at end of window
    - prediction = current (mapped)
    - target = state at (end_time + horizon_ms) (mapped)
    """
    n = len(shot_times)
    if n < window_size:
        return np.array([], dtype=int), np.array([], dtype=int), 0, 0

    end_idxs = np.arange(window_size - 1, n, dtype=int)
    end_times = shot_times[end_idxs]
    target_times = end_times + horizon_ms

    future_idxs = np.searchsorted(shot_times, target_times, side="left")
    has_future = future_idxs < n
    if not np.any(has_future):
        return np.array([], dtype=int), np.array([], dtype=int), len(end_idxs), 0

    end_idxs = end_idxs[has_future]
    future_idxs = future_idxs[has_future]

    current_raw = shot_labels[end_idxs].astype(int)
    future_raw = shot_labels[future_idxs].astype(int)

    # Must be mappable for this task (e.g. exclude -1 / other invalid labels)
    valid = np.isin(current_raw, list(label_mapping.keys())) & np.isin(
        future_raw, list(label_mapping.keys())
    )
    skipped_invalid = int((~valid).sum())
    current_raw = current_raw[valid]
    future_raw = future_raw[valid]

    preds = np.vectorize(label_mapping.get, otypes=[int])(current_raw)
    targets = np.vectorize(label_mapping.get, otypes=[int])(future_raw)

    skipped_no_future = int((~has_future).sum())
    return preds.astype(int), targets.astype(int), skipped_no_future, skipped_invalid


def summarize_and_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list,
    class_names: list,
    title: str,
    out_png: str,
):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Samples: {len(y_true):,}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Transition rate (1 - acc): {1.0 - acc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_cnt = confusion_matrix(y_true, y_pred, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Normalized confusion matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_cnt,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion matrix (counts)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Persistence baseline: predict state at t+50ms as the current state at t "
            "(where t is the end of the 150-point window), using the same random-by-shot "
            "split logic as the LSTM scripts."
        )
    )
    parser.add_argument(
        "--task",
        default="2state",
        help="Which label mapping to use: 2state, 3state, or 4state (default: 2state).",
    )
    parser.add_argument(
        "--csv",
        default="/mnt/homes/sr4240/my_folder/plasma_data.csv",
        help="Path to plasma_data.csv",
    )
    parser.add_argument(
        "--horizon_ms",
        type=int,
        default=PREDICTION_HORIZON_MS,
        help="Prediction horizon in ms (default: 50).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=WINDOW_SIZE,
        help="Window size in points (default: 150).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for shot split (default: 42).",
    )
    args = parser.parse_args()

    label_mapping, class_names, labels, task_title, out_suffix = _get_task_spec(args.task)

    print("=" * 70)
    print("State-hold baseline (no ML)")
    print("=" * 70)
    print(f"Task: {task_title}")
    print(f"Horizon: {args.horizon_ms} ms")
    print(f"Window size: {args.window_size} points")
    print("Prediction rule: predict(t + horizon) := state(t)")
    print("Split: random by shot number (70/15/15)")
    print("=" * 70)

    print("Loading state/time/shot columns...")
    y, times, shots = load_state_time_shot(args.csv)
    print(f"Rows: {len(y):,}")
    print(f"Raw label distribution: {Counter(y.astype(int))}")

    unique_shots, train_shots, val_shots, test_shots = make_random_shot_split(
        shots, seed=args.seed
    )
    print(
        f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}"
    )

    # Collect preds/targets for each split
    split_store = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    skipped_no_future_total = 0
    skipped_invalid_total = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        idxs = np.where(shot_mask)[0]
        shot_times = times[idxs]
        shot_labels = y[idxs]

        preds, targets, skipped_no_future, skipped_invalid = baseline_predictions_for_shot(
            shot_times,
            shot_labels,
            window_size=args.window_size,
            horizon_ms=args.horizon_ms,
            label_mapping=label_mapping,
        )
        skipped_no_future_total += skipped_no_future
        skipped_invalid_total += skipped_invalid

        if len(targets) == 0:
            continue

        if shot_id in train_shots:
            key = "train"
        elif shot_id in val_shots:
            key = "val"
        else:
            key = "test"

        split_store[key][0].append(preds)
        split_store[key][1].append(targets)

    def _concat(list_of_arrays):
        if not list_of_arrays:
            return np.array([], dtype=int)
        return np.concatenate(list_of_arrays).astype(int)

    train_pred = _concat(split_store["train"][0])
    train_true = _concat(split_store["train"][1])
    val_pred = _concat(split_store["val"][0])
    val_true = _concat(split_store["val"][1])
    test_pred = _concat(split_store["test"][0])
    test_true = _concat(split_store["test"][1])

    print("\nWindow matching statistics:")
    print(f"  Skipped (no future point available): {skipped_no_future_total:,}")
    print(f"  Skipped (invalid current/future label): {skipped_invalid_total:,}")
    print("\nSplit sample counts (after skipping):")
    print(f"  Train: {len(train_true):,}")
    print(f"  Val:   {len(val_true):,}")
    print(f"  Test:  {len(test_true):,}")

    if len(test_true) == 0:
        raise RuntimeError("No test samples created. Check CSV contents and label mapping.")

    out_png = f"baseline_statehold_{args.horizon_ms}ms_{out_suffix}_random_shot_results.png"
    summarize_and_plot(
        test_true,
        test_pred,
        labels=labels,
        class_names=class_names,
        title=f"State-hold baseline ({args.horizon_ms}ms) — {task_title}",
        out_png=out_png,
    )


if __name__ == "__main__":
    main()

