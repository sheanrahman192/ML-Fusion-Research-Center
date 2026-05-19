"""
Data pipeline for the FRNN-TCN binary plasma-state classifier.

Mirrors the inputs of LSTM/LSTM_50_Binary_Transitions.py but tightens leakage:
  * The 6-signal feature set is identical (iln3iamp, betan, density, li, tritop,
    fs04_past_max_smoothed).
  * Per-signal z-score (mean/std) is fit on TRAIN shots only and applied to
    val/test. This is more correct than the LSTM script which fits the scaler
    on the full dataset before splitting.
  * Windowing is strictly causal: a window of `window_size` past 1ms-sampled
    samples ending at time t is used to predict the discrete state at time
    t + prediction_horizon_ms.
  * Shot-level random split (70/15/15) with the same numpy seed as the LSTM
    script for reproducibility.
  * Binary label mapping: state 1 (Suppressed) -> 0; states {2, 3, 4}
    (Dithering / ELMing / Mitigated, "ELMy") -> 1. Set ``n_classes=4`` and
    pass ``binary=False`` to keep states 1..4 mapped to 0..3.
  * Returns "current state" (state at the end of each window) so train/eval
    can perform transition-quality analysis.

Usage:
    from frnn_data import load_plasma_data, build_window_splits

    X, y, t, s, feats = load_plasma_data()
    splits = build_window_splits(
        X, y, t, s,
        window_size=150,
        prediction_horizon_ms=150,
        binary=True,
        oversample_transitions=True,
    )
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_DATA_PATH = "/mnt/homes/sr4240/my_folder/plasma_data.csv"
DEFAULT_FEATURES: Tuple[str, ...] = (
    "iln3iamp",
    "betan",
    "density",
    "li",
    "tritop",
    "fs04_past_max_smoothed",
)
EXCLUDED_SHOTS: Tuple[int, ...] = (191675,)


@dataclass
class WindowSplit:
    """Container for one (train | val | test) split."""

    windows: np.ndarray
    labels: np.ndarray
    current_states: np.ndarray
    shots: np.ndarray
    times: np.ndarray

    def __len__(self) -> int:
        return len(self.windows)

    def label_distribution(self) -> Counter:
        return Counter(self.labels.tolist())


@dataclass
class WindowSplits:
    train: WindowSplit
    val: WindowSplit
    test: WindowSplit
    feature_names: Tuple[str, ...]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    n_classes: int


def load_plasma_data(
    data_path: str = DEFAULT_DATA_PATH,
    features: Sequence[str] = DEFAULT_FEATURES,
    excluded_shots: Sequence[int] = EXCLUDED_SHOTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Load DIII-D plasma data and return (X_raw, y, times, shots, feature_names).

    NOTE: features are returned RAW (no normalization). Z-score fit happens in
    `build_window_splits` on the train split only.
    """
    print(f"Loading plasma data from {data_path} ...")
    df = pd.read_csv(data_path)

    if excluded_shots:
        df = df[~df["shot"].isin(excluded_shots)].copy()

    feature_names = tuple(f for f in features if f in df.columns)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing requested features in CSV: {missing}")

    df = df.sort_values(["shot", "time"]).reset_index(drop=True)
    X = df[list(feature_names)].values.astype(np.float32)
    y = df["state"].values.astype(np.int64)
    t = df["time"].values.astype(np.int64)
    s = df["shot"].values.astype(np.int64)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(t)
    X, y, t, s = X[valid], y[valid], t[valid], s[valid]

    print(f"  rows: {X.shape[0]:,}, signals: {X.shape[1]} -> {list(feature_names)}")
    print(f"  shots: {len(np.unique(s)):,}")
    print(f"  raw state distribution: {dict(Counter(y.tolist()))}")
    return X, y, t, s, feature_names


def _shot_split(
    unique_shots: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_shots)
    n = len(unique_shots)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = set(shuffled[:n_train].tolist())
    val = set(shuffled[n_train : n_train + n_val].tolist())
    test = set(shuffled[n_train + n_val :].tolist())
    return train, val, test


def _per_signal_zscore(
    X: np.ndarray,
    train_mask: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-signal z-score fit on TRAIN rows only, applied everywhere."""
    means = X[train_mask].mean(axis=0)
    stds = X[train_mask].std(axis=0)
    stds = np.where(stds < eps, 1.0, stds)
    return (X - means) / stds, means.astype(np.float32), stds.astype(np.float32)


def _label_map(binary: bool) -> Dict[int, int]:
    if binary:
        return {1: 0, 2: 1, 3: 1, 4: 1}
    return {1: 0, 2: 1, 3: 2, 4: 3}


def _make_windows_for_shot(
    X_shot: np.ndarray,
    y_shot: np.ndarray,
    t_shot: np.ndarray,
    window_size: int,
    horizon_ms: int,
    label_map: Dict[int, int],
) -> Tuple[List[np.ndarray], List[int], List[int], List[int]]:
    out_w: List[np.ndarray] = []
    out_y: List[int] = []
    out_curr: List[int] = []
    out_t: List[int] = []
    n = X_shot.shape[0]
    if n < window_size:
        return out_w, out_y, out_curr, out_t

    for i in range(n - window_size + 1):
        end_idx = i + window_size - 1
        end_time = int(t_shot[end_idx])
        target_time = end_time + horizon_ms

        future_idx = int(np.searchsorted(t_shot, target_time))
        if future_idx >= n:
            continue

        curr = int(y_shot[end_idx])
        future = int(y_shot[future_idx])
        if curr not in label_map or future not in label_map:
            continue

        win = X_shot[i : i + window_size]
        if not (np.isnan(win).any() or np.isinf(win).any()):
            out_w.append(win)
            out_y.append(label_map[future])
            out_curr.append(label_map[curr])
            out_t.append(end_time)
    return out_w, out_y, out_curr, out_t


def _oversample_transitions(
    split: WindowSplit,
    transition_multiplier: int = 3,
    problematic_multiplier: int = 5,
    problematic_pair: Tuple[int, int] = (0, 1),
) -> WindowSplit:
    """Replicate the LSTM script's transition-aware oversampling on the train set."""
    cur = split.current_states
    fut = split.labels
    transition = cur != fut
    problematic = (cur == problematic_pair[0]) & (fut == problematic_pair[1])

    keep_idx = np.where(~transition)[0]
    regular_idx = np.where(transition & ~problematic)[0]
    problematic_idx = np.where(problematic)[0]

    parts: List[np.ndarray] = [keep_idx]
    parts += [regular_idx] * transition_multiplier
    parts += [problematic_idx] * problematic_multiplier
    new_idx = np.concatenate(parts)

    print(f"  oversampling: {len(split):,} -> {len(new_idx):,} samples")
    print(f"  problematic ({problematic_pair[0]}->{problematic_pair[1]}): "
          f"{len(problematic_idx):,} originals * {problematic_multiplier}x")

    return WindowSplit(
        windows=split.windows[new_idx],
        labels=split.labels[new_idx],
        current_states=split.current_states[new_idx],
        shots=split.shots[new_idx],
        times=split.times[new_idx],
    )


def build_window_splits(
    X_raw: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    shots: np.ndarray,
    window_size: int = 150,
    prediction_horizon_ms: int = 150,
    binary: bool = True,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    oversample_transitions: bool = True,
) -> WindowSplits:
    """Build train/val/test WindowSplits with strictly causal sliding windows.

    Per-signal z-score is fit on TRAIN rows only.
    """
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("train_frac/val_frac invalid")

    label_map = _label_map(binary)
    n_classes = 2 if binary else 4
    unique_shots = np.unique(shots)
    print(f"\nSplitting {len(unique_shots):,} shots randomly with seed={seed} "
          f"({train_frac:.0%}/{val_frac:.0%}/{1 - train_frac - val_frac:.0%})")
    train_shots, val_shots, test_shots = _shot_split(
        unique_shots, train_frac, val_frac, seed
    )
    print(f"  train_shots={len(train_shots)}, val_shots={len(val_shots)}, test_shots={len(test_shots)}")

    train_mask = np.isin(shots, list(train_shots))
    X, means, stds = _per_signal_zscore(X_raw, train_mask)
    print("Per-signal z-score (fit on TRAIN rows only):")
    for i, (m, s) in enumerate(zip(means, stds)):
        print(f"  signal {i}: mean={m:+.3e}, std={s:.3e}")

    print(f"\nWindowing: window_size={window_size} (causal past), horizon={prediction_horizon_ms}ms")
    splits: Dict[str, Tuple[List[np.ndarray], List[int], List[int], List[int], List[int]]] = {
        name: ([], [], [], [], []) for name in ("train", "val", "test")
    }

    for shot_id in unique_shots:
        if shot_id in train_shots:
            split_name = "train"
        elif shot_id in val_shots:
            split_name = "val"
        else:
            split_name = "test"
        idx = np.where(shots == shot_id)[0]
        if idx.size < window_size:
            continue
        sw, sy, sc, st = _make_windows_for_shot(
            X[idx],
            y[idx],
            times[idx],
            window_size=window_size,
            horizon_ms=prediction_horizon_ms,
            label_map=label_map,
        )
        if not sw:
            continue
        bucket = splits[split_name]
        bucket[0].extend(sw)
        bucket[1].extend(sy)
        bucket[2].extend(sc)
        bucket[3].extend(st)
        bucket[4].extend([int(shot_id)] * len(sw))

    def _pack(name: str) -> WindowSplit:
        ws, ys, cs, ts, sids = splits[name]
        return WindowSplit(
            windows=np.asarray(ws, dtype=np.float32) if ws else np.zeros((0, window_size, X.shape[1]), dtype=np.float32),
            labels=np.asarray(ys, dtype=np.int64),
            current_states=np.asarray(cs, dtype=np.int64),
            shots=np.asarray(sids, dtype=np.int64),
            times=np.asarray(ts, dtype=np.int64),
        )

    train = _pack("train")
    val = _pack("val")
    test = _pack("test")

    print(f"\nWindow counts: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    for name, sp in [("train", train), ("val", val), ("test", test)]:
        n_trans = int(np.sum(sp.current_states != sp.labels))
        print(f"  {name}: labels={dict(sp.label_distribution())}  transitions={n_trans:,}"
              f" ({100 * n_trans / max(len(sp), 1):.1f}%)")

    if oversample_transitions and len(train) > 0:
        print("\nOversampling transitions in train split:")
        train = _oversample_transitions(train)
        print(f"  post-oversample train labels: {dict(train.label_distribution())}")

    return WindowSplits(
        train=train,
        val=val,
        test=test,
        feature_names=tuple(),
        feature_means=means,
        feature_stds=stds,
        n_classes=n_classes,
    )


class PlasmaWindowDataset(Dataset):
    """Yields (x: (n_features, T), y: int) tensors. Matches LSTM script's expected layout."""

    def __init__(self, split: WindowSplit, also_return_current_state: bool = False):
        self.windows = torch.from_numpy(split.windows)
        self.labels = torch.from_numpy(split.labels)
        self.current_states = torch.from_numpy(split.current_states)
        self.also_return_current_state = also_return_current_state

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        win = self.windows[idx].transpose(0, 1).contiguous()
        if self.also_return_current_state:
            return win, self.labels[idx], self.current_states[idx]
        return win, self.labels[idx]


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """sklearn-style balanced class weights, normalized to sum to n_classes."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (n_classes * counts)
    return (w / w.sum() * n_classes).astype(np.float32)


def _smoke_test() -> None:
    """End-to-end smoke test: load, window, build a DataLoader, and check shapes."""
    print("\n" + "=" * 60)
    print("frnn_data smoke test")
    print("=" * 60)
    X, y, t, s, feats = load_plasma_data()

    splits = build_window_splits(
        X, y, t, s,
        window_size=150,
        prediction_horizon_ms=150,
        binary=True,
        oversample_transitions=True,
    )
    splits.feature_names = feats

    ds = PlasmaWindowDataset(splits.train)
    print(f"\nTrain dataset length: {len(ds):,}")

    win, lbl = ds[0]
    print(f"sample window shape: {tuple(win.shape)}  (expected: ({len(feats)}, 150))")
    print(f"sample label: {int(lbl)}  (binary in {{0, 1}})")
    assert win.shape == (len(feats), 150)
    assert int(lbl) in (0, 1)

    cw = class_weights(splits.train.labels, n_classes=splits.n_classes)
    print(f"class weights: {cw.tolist()}")

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    _smoke_test()
