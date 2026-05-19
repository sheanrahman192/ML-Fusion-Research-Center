#!/usr/bin/env python3
"""
Shot-level K-fold cross-validation for binary BiLSTM and TabPFN (same preprocessing).

- Unique-shot splits only (no leakage). StratifiedKFold on shots using majority
  binary label per shot when feasible; else KFold (same seed).
- StandardScaler fit **only on training-fold frame rows** before windowing.
- BiLSTM: edge-padded windows match BiLSTM_NN_Center_Point_Binary.py; validation
  shots carved from train fold for early stopping (same proportions as main script).
- TabPFN: matches TabPFN_Center_Point_Binary.py (tabular features, subsample cap).

Run from anywhere:
    TabPFN_Architecture/.venv/bin/python BiLSTM_NN_Architecture/run_cv_binary.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# Repo paths — BiLSTM script lives next to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_SCRIPT_DIR)
_TABPFN_DIR = os.path.join(_WORKSPACE, "TabPFN_Architecture")
for _p in (_SCRIPT_DIR, _TABPFN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# TabPFN reads pydantic settings at import time — set CPU allowance before any tabpfn import.
_tabpfn_dev_early = os.environ.get("TABPFN_DEVICE", "").strip().lower()
_cv_force_cpu = os.environ.get("CV_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
if _cv_force_cpu:
    os.environ["TABPFN_DEVICE"] = "cpu"
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
elif _tabpfn_dev_early == "cpu":
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

# TabPFN binary script (same windowing / TabPFN hyperparameters)
import TabPFN_Center_Point_Binary as tabpfn_bin  # noqa: E402

import torch.nn as nn
import torch.optim as optim

from BiLSTM_NN_Center_Point_Binary import (  # noqa: E402
    LSTMFirstNN,
    PlasmaDataset,
)

# --- Reproducibility ---
RNG_SEED = 42
CV_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RNG_SEED)

DATA_PATH = tabpfn_bin.DATA_PATH
PROBLEM_SHOT = tabpfn_bin.PROBLEM_SHOT
IMPORTANT_FEATURES = tabpfn_bin.IMPORTANT_FEATURES
WINDOW_SIZE = tabpfn_bin.WINDOW_SIZE
TABPFN_MAX_TRAIN_SAMPLES = tabpfn_bin.TABPFN_MAX_TRAIN_SAMPLES
TABPFN_N_ESTIMATORS = int(
    os.environ.get("TABPFN_N_ESTIMATORS", str(tabpfn_bin.TABPFN_N_ESTIMATORS))
)
TABPFN_RANDOM_STATE = tabpfn_bin.RANDOM_SEED

# Inside train fold: validation shot fraction (matches 0.15 val in main 70/15/15 split)
VAL_FRAC_WITHIN_TRAIN = 0.15
VAL_SPLIT_SEED = 42

# BiLSTM CV training (overridden by --fast; main script uses n_epochs=50)
BILSTM_CV_EPOCHS = int(os.environ.get("BILSTM_CV_EPOCHS", "50"))
BILSTM_CV_PATIENCE = int(os.environ.get("BILSTM_CV_PATIENCE", "20"))
BILSTM_CV_BATCH_SIZE = int(os.environ.get("BILSTM_CV_BATCH_SIZE", "256"))
RESULTS_JSON = os.path.join(_SCRIPT_DIR, "run_cv_binary_results.json")


def load_frames_no_scaler():
    """Same cleaning / binary mapping as BiLSTM & TabPFN binaries; returns unscaled X."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["shot"] != PROBLEM_SHOT].copy()
    selected_features = [f for f in IMPORTANT_FEATURES if f in df.columns]

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    state_as_float = df_sorted["state"].replace(-1, np.nan)
    df_sorted = df_sorted.assign(state=state_as_float)
    df_sorted["state"] = (
        df_sorted.groupby("shot", group_keys=False)["state"]
        .transform(lambda s: s.ffill().bfill())
    )

    df_filtered = df_sorted.dropna(subset=["state"]).copy()
    df_filtered["state"] = df_filtered["state"].astype(int)

    X = df_filtered[selected_features].values.astype(np.float64)
    y_multi = df_filtered["state"].values.astype(int)
    shots = df_filtered["shot"].values

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_multi = y_multi[valid_mask]
    shots = shots[valid_mask]

    y_binary = np.where(y_multi == 0, 0, 1).astype(np.int64)
    return X, y_binary, shots, selected_features


def majority_shot_labels(unique_shots: np.ndarray, shots: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One binary label per shot (majority of frame labels) for stratification."""
    out = np.empty(len(unique_shots), dtype=np.int64)
    for i, s in enumerate(unique_shots):
        ys = y[shots == s]
        n0 = int(np.sum(ys == 0))
        n1 = int(np.sum(ys == 1))
        out[i] = 0 if n0 >= n1 else 1
    return out


def scale_features_train_only(
    X: np.ndarray, shots: np.ndarray, train_shots: np.ndarray
) -> np.ndarray:
    train_mask = np.isin(shots, train_shots)
    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    return scaler.transform(X).astype(np.float32)


def split_train_into_train_val_shots(
    train_shots: np.ndarray, y_shot_majority: dict, seed: int = VAL_SPLIT_SEED
):
    """Split CV train shots into subtrain / val shots (stratified when possible)."""
    ts = np.asarray(train_shots)
    if len(ts) <= 1:
        return ts, np.array([], dtype=ts.dtype)

    labels = np.array([y_shot_majority[int(s)] for s in ts])
    stratify = labels if len(np.unique(labels)) > 1 else None
    try:
        subtrain, val = train_test_split(
            ts,
            test_size=VAL_FRAC_WITHIN_TRAIN,
            stratify=stratify,
            random_state=seed,
        )
    except ValueError:
        subtrain, val = train_test_split(ts, test_size=VAL_FRAC_WITHIN_TRAIN, random_state=seed)

    if len(subtrain) == 0:
        return ts, np.array([], dtype=ts.dtype)
    return np.asarray(subtrain, dtype=ts.dtype), np.asarray(val, dtype=ts.dtype)


def build_shot_stratified_folds(unique_shots: np.ndarray, y_maj: np.ndarray, k: int, seed: int):
    """Return list of (train_idx, test_idx) arrays into unique_shots."""
    min_per_class = min(np.sum(y_maj == 0), np.sum(y_maj == 1))
    use_stratified = min_per_class >= k

    X_dummy = np.zeros((len(unique_shots), 1))
    if use_stratified:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(cv.split(X_dummy, y_maj))
        print(f"Using StratifiedKFold (min class count among shots = {min_per_class} >= K={k}).")
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(cv.split(X_dummy))
        print(
            f"Using KFold (stratified requires >= K shots per class; "
            f"min class count = {min_per_class})."
        )
    return splits, use_stratified


def train_model_cv(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs: int,
    max_patience: int,
    fold_label: str = "",
):
    """Compact BiLSTM training loop for CV (early stop, minimal logging)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    prefix = f"[{fold_label}] " if fold_label else ""

    for epoch in range(n_epochs):
        model.train()
        train_preds, train_labels = [], []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(batch_y.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        n_val = len(val_loader.dataset)
        if n_val > 0:
            val_acc = accuracy_score(val_labels_list, val_preds)
        else:
            val_acc = train_acc

        scheduler.step(val_acc)
        monitor = val_acc if n_val > 0 else train_acc

        improved = monitor > best_val_acc
        if improved:
            best_val_acc = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            star = " *best"
        else:
            patience_counter += 1
            star = ""

        if (epoch + 1) % 5 == 0 or improved or epoch == 0 or patience_counter >= max_patience:
            print(
                f"{prefix}BiLSTM epoch {epoch + 1}/{n_epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}{star}",
                flush=True,
            )

        if patience_counter >= max_patience:
            print(f"{prefix}BiLSTM early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc


def run_one_fold_bilstm(
    device,
    n_features,
    train_windows,
    train_labels,
    val_windows,
    val_labels,
    test_windows,
    test_labels,
    batch_size: int | None = None,
    fold_label: str = "",
):
    if batch_size is None:
        batch_size = BILSTM_CV_BATCH_SIZE
    model = LSTMFirstNN(n_features=n_features, n_classes=2).to(device)

    train_ds = PlasmaDataset(train_windows, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if len(val_windows) > 0:
        val_ds = PlasmaDataset(val_windows, val_labels)
    else:
        empty_w = np.zeros(
            (0, train_windows.shape[1], train_windows.shape[2]), dtype=np.float32
        )
        val_ds = PlasmaDataset(empty_w, np.zeros(0, dtype=np.int64))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_model_cv(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=BILSTM_CV_EPOCHS,
        max_patience=BILSTM_CV_PATIENCE,
        fold_label=fold_label,
    )

    test_ds = PlasmaDataset(test_windows, test_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            pr = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pr)
            labels.extend(batch_y.numpy())
    return accuracy_score(labels, preds)


def _tabpfn_predict_proba_batched(model, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Chunked predict_proba; use large chunks (TabPFN retrains context each call)."""
    n = len(X)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
  # One shot when small enough (much faster than thousands of tiny batches).
    if n <= batch_size:
        print(f"  TabPFN: single predict_proba call ({n} rows)...", flush=True)
        return model.predict_proba(X)

    parts = []
    n_chunks = (n + batch_size - 1) // batch_size
    for ci, start in enumerate(range(0, n, batch_size), start=1):
        end = min(start + batch_size, n)
        print(
            f"  TabPFN: predict chunk {ci}/{n_chunks} rows {start}:{end}...",
            flush=True,
        )
        parts.append(model.predict_proba(X[start:end]))
    return np.vstack(parts)


def run_one_fold_tabpfn(test_tab, test_y, train_tab, train_y, device: str, infer_batch_size: int = 512):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ImportError("Install tabpfn in the TabPFN venv.") from e

    if device == "cpu":
        # TabPFN blocks CPU fit with >1k train rows unless explicitly allowed.
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    train_tab_used, train_y_used, _ = tabpfn_bin.stratified_subsample(
        train_tab, train_y, TABPFN_MAX_TRAIN_SAMPLES, seed=TABPFN_RANDOM_STATE
    )
    model = TabPFNClassifier(
        device=device,
        n_estimators=TABPFN_N_ESTIMATORS,
        ignore_pretraining_limits=False,
        random_state=TABPFN_RANDOM_STATE,
    )
    print(f"  TabPFN: fitting on {len(train_tab_used)} subsampled train rows...", flush=True)
    t_fit = time.time()
    model.fit(train_tab_used, train_y_used)
    print(f"  TabPFN: fit finished in {time.time() - t_fit:.1f}s", flush=True)
    print(f"  TabPFN: predicting on {len(test_tab)} test rows (batch={infer_batch_size})...", flush=True)
    t_pred = time.time()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    probs = None
    batch_try = infer_batch_size
    for attempt in range(4):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            probs = _tabpfn_predict_proba_batched(model, test_tab, batch_size=batch_try)
            break
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" not in err and "cuda" not in err:
                raise
            batch_try = max(512, batch_try // 2)
            print(
                f"  TabPFN: CUDA OOM — retry predict with batch={batch_try} "
                f"(attempt {attempt + 1}/4)...",
                flush=True,
            )
    if probs is None:
        print(
            "  TabPFN: GPU retries failed; CPU fallback (last resort)...",
            flush=True,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_cpu = TabPFNClassifier(
            device="cpu",
            n_estimators=TABPFN_N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=TABPFN_RANDOM_STATE,
        )
        model_cpu.fit(train_tab_used, train_y_used)
        probs = _tabpfn_predict_proba_batched(
            model_cpu, test_tab, batch_size=4096
        )
    print(f"  TabPFN: predict finished in {time.time() - t_pred:.1f}s", flush=True)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(test_y, preds)
    try:
        del model
    except NameError:
        pass
    try:
        del model_cpu
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc


def main():
    global BILSTM_CV_EPOCHS, BILSTM_CV_PATIENCE, TABPFN_N_ESTIMATORS

    parser = argparse.ArgumentParser(description="Shot-level K-fold CV for BiLSTM + TabPFN binary.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default 5).")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="If set, stop after this many folds (e.g. quick smoke test).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Faster CV: BiLSTM 15 epochs / patience 8, TabPFN n_estimators=4 (default on).",
    )
    parser.add_argument(
        "--no-fast",
        action="store_false",
        dest="fast",
        help="Use full settings (BiLSTM 50 epochs unless env set, TabPFN n_estimators from script).",
    )
    parser.add_argument("--skip-bilstm", action="store_true")
    parser.add_argument("--skip-tabpfn", action="store_true")
    args = parser.parse_args()
    k = args.k

    if args.fast:
        if "BILSTM_CV_EPOCHS" not in os.environ:
            BILSTM_CV_EPOCHS = 15
        if "BILSTM_CV_PATIENCE" not in os.environ:
            BILSTM_CV_PATIENCE = 8
        if "TABPFN_N_ESTIMATORS" not in os.environ:
            TABPFN_N_ESTIMATORS = 4

    print("=" * 70, flush=True)
    print("Shot-level K-fold CV — Binary BiLSTM & TabPFN", flush=True)
    print(f"  K={k}, scaler fit on train-fold rows only, window_size={WINDOW_SIZE}", flush=True)
    print(f"  Fast mode: {args.fast}", flush=True)
    print(f"  BiLSTM: epochs={BILSTM_CV_EPOCHS}, patience={BILSTM_CV_PATIENCE}, batch={BILSTM_CV_BATCH_SIZE}", flush=True)
    print(f"  TabPFN: n_estimators={TABPFN_N_ESTIMATORS}, train_cap={TABPFN_MAX_TRAIN_SAMPLES}", flush=True)
    print("=" * 70, flush=True)

    X_raw, y, shots, features = load_frames_no_scaler()
    unique_shots = np.unique(shots)
    y_maj = majority_shot_labels(unique_shots, shots, y)
    y_maj_map = {int(s): int(y_maj[i]) for i, s in enumerate(unique_shots)}

    print(f"Frames: {len(y):,}, unique shots: {len(unique_shots)}")
    print(f"Frame-level binary distribution: {Counter(y.tolist())}")
    print(f"Shot majority labels: {Counter(y_maj.tolist())}")

    splits, strat = build_shot_stratified_folds(unique_shots, y_maj, k, CV_SEED)

    # BiLSTM device: CUDA when available unless CV_FORCE_CPU=1 or cuDNN issues.
    if os.environ.get("CV_FORCE_CPU", "").strip() in ("1", "true", "yes"):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TabPFN: default to CUDA when available; override with TABPFN_DEVICE=cpu if GPU errors.
    tabpfn_device = os.environ.get(
        "TABPFN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if os.environ.get("CV_FORCE_CPU", "").strip() in ("1", "true", "yes"):
        tabpfn_device = "cpu"
    if tabpfn_device == "cpu":
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    # Avoid cuDNN init failures on some drivers / GPUs (LSTM still runs on CUDA).
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    # Large chunks: each predict_proba re-runs in-context attention on the train set.
    _default_ib = "4096" if tabpfn_device == "cuda" else "4096"
    infer_bs = int(os.environ.get("TABPFN_INFER_BATCH", _default_ib))
    print(f"PyTorch (BiLSTM) device: {device}")
    print(f"TabPFN device: {tabpfn_device} (infer batch={infer_bs})")

    bilstm_accs = []
    tabpfn_accs = []
    fold_shot_counts = []

    t0_all = time.time()

    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        if args.max_folds is not None and fold_id >= args.max_folds:
            print(f"\nStopping early (--max-folds={args.max_folds}).")
            break
        train_shots = unique_shots[tr_idx]
        test_shots = unique_shots[te_idx]
        fold_shot_counts.append(len(test_shots))

        print("\n" + "-" * 70, flush=True)
        fold_tag = f"Fold {fold_id + 1}/{k}"
        print(
            f"UPDATE {fold_tag}: train shots={len(train_shots)}, "
            f"test shots={len(test_shots)}",
            flush=True,
        )

        X_scaled = scale_features_train_only(X_raw, shots, train_shots)

        train_sub, val_shots = split_train_into_train_val_shots(train_shots, y_maj_map)

        train_w, train_y = tabpfn_bin.create_windows_for_shots(
            X_scaled, y, shots, train_sub, WINDOW_SIZE
        )
        val_w, val_y = tabpfn_bin.create_windows_for_shots(
            X_scaled, y, shots, val_shots, WINDOW_SIZE
        )
        test_w, test_y = tabpfn_bin.create_windows_for_shots(
            X_scaled, y, shots, test_shots, WINDOW_SIZE
        )

        print(
            f"  Windows: train={len(train_w)}, val={len(val_w)}, test={len(test_w)}",
            flush=True,
        )

        if not args.skip_tabpfn:
            print(f"  UPDATE {fold_tag}: TabPFN starting...", flush=True)
            tr_tab, _ = tabpfn_bin.windows_to_tabular_features(train_w, features)
            te_tab, _ = tabpfn_bin.windows_to_tabular_features(test_w, features)
            t0 = time.time()
            acc_t = run_one_fold_tabpfn(
                te_tab, test_y, tr_tab, train_y, tabpfn_device, infer_batch_size=infer_bs
            )
            tabpfn_accs.append(acc_t)
            print(
                f"  UPDATE {fold_tag}: TabPFN test acc={acc_t:.4f} ({time.time()-t0:.1f}s)",
                flush=True,
            )
        else:
            print("  TabPFN skipped.", flush=True)

        if not args.skip_bilstm:
            print(f"  UPDATE {fold_tag}: BiLSTM training...", flush=True)
            if len(train_w) == 0 or len(test_w) == 0:
                print("  BiLSTM skipped (empty train or test windows).")
                bilstm_accs.append(float("nan"))
            else:
                t0 = time.time()
                acc_b = run_one_fold_bilstm(
                    device,
                    len(features),
                    train_w,
                    train_y,
                    val_w,
                    val_y,
                    test_w,
                    test_y,
                    fold_label=fold_tag,
                )
                bilstm_accs.append(acc_b)
                print(
                    f"  UPDATE {fold_tag}: BiLSTM test acc={acc_b:.4f} ({time.time()-t0:.1f}s)",
                    flush=True,
                )
        else:
            print("  BiLSTM skipped.")

    elapsed = time.time() - t0_all

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Shots per test fold (#shots held out): {fold_shot_counts}")
    print(f"Total elapsed: {elapsed:.1f}s")

    def mean_std(arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return float("nan"), float("nan")
        return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    if bilstm_accs:
        m, s = mean_std(bilstm_accs)
        print(f"\nBiLSTM binary — mean ± std accuracy (K={k}): {m:.4f} ± {s:.4f}")
        print(f"  Per-fold: {[float(f'{x:.4f}') for x in bilstm_accs]}")
    if tabpfn_accs:
        m, s = mean_std(tabpfn_accs)
        print(f"\nTabPFN binary — mean ± std accuracy (K={k}): {m:.4f} ± {s:.4f}", flush=True)
        print(f"  Per-fold: {[float(f'{x:.4f}') for x in tabpfn_accs]}", flush=True)

    results = {
        "k": k,
        "fast_mode": args.fast,
        "bilstm_epochs": BILSTM_CV_EPOCHS,
        "bilstm_patience": BILSTM_CV_PATIENCE,
        "tabpfn_n_estimators": TABPFN_N_ESTIMATORS,
        "stratified_folds": bool(strat),
        "fold_test_shot_counts": fold_shot_counts,
        "bilstm_fold_accuracy": [float(x) for x in bilstm_accs],
        "tabpfn_fold_accuracy": [float(x) for x in tabpfn_accs],
        "elapsed_seconds": elapsed,
    }
    if bilstm_accs:
        m, s = mean_std(bilstm_accs)
        results["bilstm_mean_accuracy"] = m
        results["bilstm_std_accuracy"] = s
    if tabpfn_accs:
        m, s = mean_std(tabpfn_accs)
        results["tabpfn_mean_accuracy"] = m
        results["tabpfn_std_accuracy"] = s

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON: {RESULTS_JSON}", flush=True)

    print("\nNotes:", flush=True)
    print(f"  - Stratified folds: {strat}", flush=True)
    print(f"  - BiLSTM epochs cap: {BILSTM_CV_EPOCHS} (early stopping patience {BILSTM_CV_PATIENCE})", flush=True)
    print(f"  - TabPFN train cap: {TABPFN_MAX_TRAIN_SAMPLES} (stratified subsample if exceeded)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
