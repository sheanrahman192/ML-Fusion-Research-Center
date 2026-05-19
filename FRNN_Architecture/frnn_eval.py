"""
Evaluation utilities for the FRNN-TCN binary plasma-state classifier.

Reproduces (and slightly tightens) the metrics from
LSTM/LSTM_50_Binary_Transitions.py:
  * classification_report and confusion_matrix
  * binary ROC AUC (positive class = ELMy)
  * threshold tuning on validation set (max F1)
  * transition-effectiveness analysis: metrics restricted to windows where
    current_state != future_state, broken down by transition type.

All evaluation runs with model.eval() and torch.no_grad().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for evaluation utilities") from exc


@dataclass
class EvalOutputs:
    probs: np.ndarray
    preds: np.ndarray
    labels: np.ndarray
    current_states: Optional[np.ndarray] = None


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_current_states: bool = False,
) -> EvalOutputs:
    """Run the model over a loader and return softmax probs / argmax preds / labels."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_current: List[np.ndarray] = []

    for batch in loader:
        if return_current_states:
            x, y, curr = batch
            all_current.append(curr.numpy())
        else:
            x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.numpy())

    return EvalOutputs(
        probs=np.concatenate(all_probs),
        preds=np.concatenate(all_preds),
        labels=np.concatenate(all_labels),
        current_states=np.concatenate(all_current) if all_current else None,
    )


def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: Sequence[float] = tuple(np.linspace(0.1, 0.9, 81).tolist()),
    average: str = "weighted",
) -> Tuple[float, float]:
    """Sweep the decision threshold on (positive-class) prob and pick the F1 max.

    Only valid for binary problems. Returns (best_threshold, best_f1).
    """
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("find_optimal_threshold expects 2-class probabilities.")
    pos_probs = probs[:, 1]
    best_t = 0.5
    best_f1 = -np.inf
    for t in grid:
        p = (pos_probs >= t).astype(np.int64)
        if len(np.unique(p)) < 2:
            continue
        f1 = f1_score(labels, p, average=average, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("predict_with_threshold is binary-only.")
    return (probs[:, 1] >= threshold).astype(np.int64)


def report_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: Optional[np.ndarray],
    class_names: Sequence[str],
    title: str = "Metrics",
) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(classification_report(labels, preds, target_names=list(class_names), digits=4, zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)
    if probs is not None and probs.shape[1] == 2 and len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs[:, 1])
        print(f"ROC AUC (positive = {class_names[1]}): {auc:.4f}")
    elif probs is not None and probs.shape[1] > 2 and len(np.unique(labels)) == probs.shape[1]:
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            print(f"Macro OVR ROC AUC: {auc:.4f}")
        except ValueError:
            pass


def transition_analysis(
    preds: np.ndarray,
    labels: np.ndarray,
    current_states: np.ndarray,
    probs: Optional[np.ndarray],
    class_names: Sequence[str],
    horizon_ms: int,
) -> None:
    """Quality of predictions on windows where the future state != current state."""
    print("\n" + "=" * 60)
    print(f"Transition effectiveness ({horizon_ms}ms horizon)")
    print("=" * 60)
    transition_mask = current_states != labels
    n_trans = int(np.sum(transition_mask))
    n_total = int(len(labels))
    print(f"transition windows: {n_trans:,} / {n_total:,} ({100 * n_trans / max(n_total, 1):.2f}%)")
    if n_trans == 0:
        print("  No transitions in this set; skipping.")
        return

    tp = preds[transition_mask]
    tl = labels[transition_mask]
    print(f"  acc       : {accuracy_score(tl, tp):.4f}")
    print(f"  precision : {precision_score(tl, tp, average='weighted', zero_division=0):.4f}")
    print(f"  recall    : {recall_score(tl, tp, average='weighted', zero_division=0):.4f}")
    print(f"  f1        : {f1_score(tl, tp, average='weighted', zero_division=0):.4f}")
    if probs is not None and probs.shape[1] == 2 and len(np.unique(tl)) > 1:
        print(f"  ROC AUC   : {roc_auc_score(tl, probs[transition_mask, 1]):.4f}")

    print(f"\nTransition-only classification report:")
    print(classification_report(tl, tp, target_names=list(class_names), digits=4, zero_division=0))

    print("Breakdown by transition type:")
    for src in range(len(class_names)):
        for dst in range(len(class_names)):
            if src == dst:
                continue
            mask = (current_states == src) & (labels == dst)
            n = int(np.sum(mask))
            if n == 0:
                continue
            acc = accuracy_score(labels[mask], preds[mask])
            print(f"  {class_names[src]:>30s} -> {class_names[dst]:<30s}  "
                  f"n={n:>6,}  acc={acc:.4f}")


def plot_curves(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    train_accs: Iterable[float],
    val_accs: Iterable[float],
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
) -> None:
    """Save a 2x2 plot: loss curves, acc curves, normalized CM, count CM."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover
        print(f"  (skipping plot: {exc})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].plot(list(train_losses), label="train")
    axes[0, 0].plot(list(val_losses), label="val")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(list(train_accs), label="train")
    axes[0, 1].plot(list(val_accs), label="val")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("acc")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    cm_norm = confusion_matrix(labels, preds, labels=list(range(len(class_names))), normalize="true")
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=list(class_names), yticklabels=list(class_names),
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Confusion matrix (normalized)")
    axes[1, 0].set_xlabel("predicted")
    axes[1, 0].set_ylabel("true")

    cm_count = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    sns.heatmap(
        cm_count, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(class_names), yticklabels=list(class_names),
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Confusion matrix (counts)")
    axes[1, 1].set_xlabel("predicted")
    axes[1, 1].set_ylabel("true")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
