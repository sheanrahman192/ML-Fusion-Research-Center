#!/usr/bin/env python3
"""
Train LSTM 50ms / 2-state (same settings as LSTM_50_2_Random_Shot.py) on combined_database.csv
while holding out shot 169472 from train and validation. After training, run inference on
shot 169472 only and save a three-panel comprehensive analysis plot (no confidence panel).

Outputs:
  - Checkpoint: LSTM/best_lstm_50ms_2state_random_shot_{RUN_TAG}.pth
  - Plot: Isolated Shot Comprehensive Analysis/isolated_shot_169472_lstm_comprehensive_analysis.png

Usage:
  MPLBACKEND=Agg python LSTM_50_2_Shot_169472_Predictions.py [data_csv] [run_tag]
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import LSTM_50_2_Random_Shot as base

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HOLDOUT_SHOT = 169472
DATA_CSV = sys.argv[1] if len(sys.argv) > 1 else '/mnt/homes/sr4240/my_folder/combined_database.csv'
RUN_TAG = sys.argv[2] if len(sys.argv) > 2 else 'shot_169472_holdout'
WINDOW_SIZE = 150
PLOT_OUTPUT = os.path.join(
    '/mnt/homes/sr4240/my_folder/Isolated Shot Comprehensive Analysis',
    f'isolated_shot_{HOLDOUT_SHOT}_lstm_comprehensive_analysis.png',
)
CHECKPOINT_NAME = f'best_lstm_50ms_2state_random_shot_{RUN_TAG}.pth'
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, CHECKPOINT_NAME)

base.DATA_CSV = DATA_CSV
base.RUN_TAG = RUN_TAG
# Smaller batch if GPU is contended (parent default 4096 can OOM on shared GPUs).
if os.environ.get('LSTM_BATCH_SIZE'):
    base.BATCH_SIZE = int(os.environ['LSTM_BATCH_SIZE'])
elif base.BATCH_SIZE > 1024:
    base.BATCH_SIZE = 1024


def create_windows_excluding_holdout(X, y, times, shots, holdout_shot=HOLDOUT_SHOT,
                                     window_size=WINDOW_SIZE,
                                     prediction_horizon_ms=base.PREDICTION_HORIZON_MS):
    """Same shot-based split as parent, but holdout shot never enters train/val/test."""
    print(f"Creating windows (holdout shot {holdout_shot} excluded from all splits)...")

    unique_shots = np.unique(shots)
    holdout_present = int(holdout_shot) in {int(s) for s in unique_shots}
    if not holdout_present:
        raise ValueError(
            f"Shot {holdout_shot} has no rows after NaN filtering. "
            "Cannot run holdout evaluation."
        )

    trainable_shots = unique_shots[unique_shots != holdout_shot]
    n_shots = len(trainable_shots)
    print(f"Total unique shots (excl. holdout): {n_shots}")

    valid_set = {int(s) for s in trainable_shots}
    requested_test = {int(s) for s in base.TEST_SHOTS}
    test_shots = requested_test & valid_set
    remaining = np.array([s for s in trainable_shots if int(s) not in test_shots])
    np.random.seed(42)
    shuffled_remaining = np.random.permutation(remaining)
    train_size = int(0.9 * len(shuffled_remaining))
    train_shots = set(shuffled_remaining[:train_size].astype(int).tolist())
    val_shots = set(shuffled_remaining[train_size:].astype(int).tolist())

    assert holdout_shot not in train_shots
    assert holdout_shot not in val_shots
    assert holdout_shot not in test_shots

    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")
    print(f"  Holdout (eval only): {holdout_shot}")

    train_windows, train_labels, train_current = [], [], []
    val_windows, val_labels, val_current = [], [], []
    test_windows, test_labels, test_current = [], [], []

    label_mapping = {0: 0, 1: 1}
    valid_raw_labels = {0, 1}
    windows_created = 0

    for shot_id in trainable_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        if len(shot_indices) < window_size:
            continue

        sid_int = int(shot_id)
        if sid_int in train_shots:
            target_windows, target_labels, target_current = train_windows, train_labels, train_current
            split_kind = 'train'
        elif sid_int in val_shots:
            target_windows, target_labels, target_current = val_windows, val_labels, val_current
            split_kind = 'val'
        elif sid_int in test_shots:
            target_windows, target_labels, target_current = test_windows, test_labels, test_current
            split_kind = 'test'
        else:
            continue

        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]
        stride = 1 if split_kind == 'test' else base.TRAIN_STRIDE

        for i in range(0, len(shot_indices) - window_size + 1, stride):
            window = shot_X[i:i + window_size]
            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms
            future_local_idx = np.searchsorted(shot_times, target_time)
            if future_local_idx >= len(shot_times):
                continue
            future_label = shot_labels[future_local_idx]
            current_label = shot_labels[i + window_size - 1]
            if int(future_label) not in valid_raw_labels or int(current_label) not in valid_raw_labels:
                continue
            if not np.isnan(window).any() and not np.isinf(window).any():
                target_windows.append(window)
                target_labels.append(label_mapping[int(future_label)])
                target_current.append(label_mapping[int(current_label)])
                windows_created += 1

    arrays = (
        np.array(train_windows, dtype=np.float32), np.array(train_labels),
        np.array(train_current),
        np.array(val_windows, dtype=np.float32), np.array(val_labels), np.array(val_current),
        np.array(test_windows, dtype=np.float32), np.array(test_labels), np.array(test_current),
    )
    print(f"Windows created (train/val/test): {len(arrays[0])} / {len(arrays[3])} / {len(arrays[6])}")
    return arrays


def build_holdout_shot_windows(X, y, times, shots, shot_id=HOLDOUT_SHOT,
                               window_size=WINDOW_SIZE,
                               prediction_horizon_ms=base.PREDICTION_HORIZON_MS):
    """Build stride-1 windows for a single holdout shot (50ms-ahead labels)."""
    shot_mask = shots == shot_id
    if not shot_mask.any():
        raise ValueError(f"Shot {shot_id} has no rows after NaN filtering.")

    shot_indices = np.where(shot_mask)[0]
    if len(shot_indices) < window_size:
        raise ValueError(
            f"Shot {shot_id} has only {len(shot_indices)} rows after filtering; "
            f"need at least {window_size} for windowing."
        )

    shot_times = times[shot_mask]
    shot_labels = y[shot_mask]
    shot_X = X[shot_mask]

    windows, labels, window_end_times = [], [], []
    label_mapping = {0: 0, 1: 1}
    valid_raw_labels = {0, 1}

    for i in range(len(shot_indices) - window_size + 1):
        window = shot_X[i:i + window_size]
        window_end_time = shot_times[i + window_size - 1]
        target_time = window_end_time + prediction_horizon_ms
        future_local_idx = np.searchsorted(shot_times, target_time)
        if future_local_idx >= len(shot_times):
            continue
        future_label = shot_labels[future_local_idx]
        current_label = shot_labels[i + window_size - 1]
        if int(future_label) not in valid_raw_labels or int(current_label) not in valid_raw_labels:
            continue
        if not np.isnan(window).any() and not np.isinf(window).any():
            windows.append(window)
            labels.append(label_mapping[int(future_label)])
            window_end_times.append(float(window_end_time))

    if len(windows) == 0:
        raise ValueError(
            f"Shot {shot_id}: no valid windows after horizon/label filtering."
        )

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=int),
        np.array(window_end_times, dtype=np.float64),
    )


def predict_windows(model, windows, labels, device, batch_size=4096):
    """Run model on windows; return predictions and labels."""
    dataset = base.PlasmaDataset(windows, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            batch_preds = torch.argmax(outputs, dim=1)
            preds.extend(batch_preds.cpu().numpy())
    return np.array(preds, dtype=int), labels


def load_holdout_shot_dataframe():
    """Load holdout shot rows (post same cleaning as training) for fs04 / state_binary plotting."""
    df = pd.read_csv(DATA_CSV)
    df = df[df['shot'] != 191675].copy()
    shot_df = df[df['shot'] == HOLDOUT_SHOT].copy()
    if shot_df.empty:
        raise ValueError(f"Shot {HOLDOUT_SHOT} not found in {DATA_CSV}.")

    if 'state_binary' in shot_df.columns:
        shot_df = shot_df[~shot_df['state_binary'].isna()].copy()
    else:
        raw = shot_df['state'].values.astype(np.float64)
        mapped = np.where(raw == 0, 0.0, np.where(np.isin(raw, [1, 2, 3]), 1.0, np.nan))
        shot_df = shot_df.assign(state_binary=mapped)
        shot_df = shot_df[~shot_df['state_binary'].isna()].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li', 'tritop', 'fs_sum_past_max_smoothed']
    feat_cols = [f for f in important_features if f in shot_df.columns]
    shot_df = shot_df.dropna(subset=feat_cols + ['time']).copy()
    shot_df = shot_df.sort_values('time').reset_index(drop=True)

    if shot_df.empty:
        raise ValueError(f"Shot {HOLDOUT_SHOT} has no rows after NaN filtering.")

    if 'fs04' not in shot_df.columns:
        raise ValueError(f"Shot {HOLDOUT_SHOT}: column 'fs04' missing; cannot plot.")

    return shot_df


def plot_isolated_shot_comprehensive(shot_df, predictions, labels, window_end_times,
                                     shot_number=HOLDOUT_SHOT, output_path=PLOT_OUTPUT):
    """Three-panel fs04 plot (actual / predicted / accuracy), binary 2-state, no confidence."""
    plt.style.use('seaborn-v0_8')
    state_colors = {0: '#2E8B57', 1: '#DC143C'}
    state_names = {0: 'Suppressed', 1: 'Other (Dithering/Mitigated/ELMing)'}

    time = shot_df['time'].values
    fs04 = shot_df['fs04'].values
    actual_binary = shot_df['state_binary'].values.astype(int)

    pred_at_time = np.full(len(shot_df), -1, dtype=float)
    true_at_time = np.full(len(shot_df), -1, dtype=float)
    horizon = base.PREDICTION_HORIZON_MS

    for pred, true, window_end_time in zip(predictions, labels, window_end_times):
        future_time = window_end_time + horizon
        time_idx = int(np.argmin(np.abs(time - future_time)))
        if time_idx < len(pred_at_time):
            pred_at_time[time_idx] = pred
            true_at_time[time_idx] = true

    fig, axes = plt.subplots(3, 1, figsize=(24, 20))
    fig.suptitle(
        f'Plasma State Classification Analysis for Shot {shot_number} - LSTM (Split Based on Shot)',
        fontsize=20, fontweight='bold', y=0.995,
    )

    y_lo, y_hi = fs04.min(), fs04.max()
    xlim = (time.min(), time.max())

    ax1 = axes[0]
    ax1.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    for state in (0, 1):
        mask = actual_binary == state
        if mask.any():
            ax1.fill_between(time, y_lo, y_hi, where=mask, alpha=0.3,
                             color=state_colors[state], label=f'Actual: {state_names[state]}')
    ax1.set_ylabel('fs04', fontsize=16)
    ax1.set_title('(a) Observed Plasma States', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)

    ax2 = axes[1]
    ax2.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    for state in (0, 1):
        mask = pred_at_time == state
        if mask.any():
            ax2.fill_between(time, y_lo, y_hi, where=mask, alpha=0.3,
                             color=state_colors[state], label=f'Predicted: {state_names[state]}')
    ax2.set_ylabel('fs04', fontsize=16)
    ax2.set_title('(b) Predicted Plasma States', fontsize=18, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)

    ax3 = axes[2]
    ax3.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')
    valid = (pred_at_time >= 0) & (true_at_time >= 0)
    correct = valid & (pred_at_time == true_at_time)
    incorrect = valid & (pred_at_time != true_at_time)
    ax3.fill_between(time, y_lo, y_hi, where=correct, alpha=0.4,
                     color='green', label='Correct Predictions')
    ax3.fill_between(time, y_lo, y_hi, where=incorrect, alpha=0.4,
                     color='red', label='Incorrect Predictions')
    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('(c) Prediction Accuracy Analysis', fontsize=18, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(xlim)

    acc = accuracy_score(labels, predictions)
    ax3.text(0.02, 0.98, f'50ms-ahead window accuracy: {acc:.3f}',
             transform=ax3.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.subplots_adjust(hspace=0.3, bottom=0.06, top=0.92)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=800, bbox_inches='tight')
    plt.close(fig)
    print(f"Comprehensive plot saved to '{output_path}'")
    return acc


def main():
    os.chdir(SCRIPT_DIR)
    print("=" * 60)
    print("LSTM 50ms 2-state — holdout shot 169472")
    print("=" * 60)
    print(f"  CSV: {DATA_CSV}")
    print(f"  RUN_TAG: {RUN_TAG}")
    print(f"  Holdout shot: {HOLDOUT_SHOT} (excluded from train/val)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, scaler = base.load_and_prepare_data()

    (train_X, train_y, _,
     val_X, val_y, _,
     test_X, test_y, _) = create_windows_excluding_holdout(X, y, times, shots)

    if len(train_X) == 0 or len(val_X) == 0:
        raise RuntimeError("Empty train or val set after holdout split.")

    train_loader = DataLoader(
        base.PlasmaDataset(train_X, train_y),
        batch_size=base.BATCH_SIZE, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        base.PlasmaDataset(val_X, val_y),
        batch_size=base.BATCH_SIZE, shuffle=False, pin_memory=True,
    )

    model = base.LSTMFirstNN(n_features=len(features), n_classes=base.N_CLASSES).to(device)

    print("\nStarting training (20 epochs, early stopping)...")
    base.train_model(model, train_loader, val_loader, device, n_epochs=20)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Expected checkpoint not found: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    holdout_windows, holdout_labels, holdout_end_times = build_holdout_shot_windows(
        X, y, times, shots,
    )
    print(f"Holdout windows for shot {HOLDOUT_SHOT}: {len(holdout_windows)}")

    holdout_preds, _ = predict_windows(model, holdout_windows, holdout_labels, device)
    holdout_acc = accuracy_score(holdout_labels, holdout_preds)
    print(f"Holdout shot {HOLDOUT_SHOT} 50ms-ahead accuracy: {holdout_acc:.4f}")

    shot_df = load_holdout_shot_dataframe()
    print(f"Holdout shot plot rows: {len(shot_df)}, time {shot_df['time'].min():.0f}–{shot_df['time'].max():.0f} ms")
    print(f"state_binary distribution: {shot_df['state_binary'].value_counts().to_dict()}")

    plot_isolated_shot_comprehensive(
        shot_df, holdout_preds, holdout_labels, holdout_end_times,
    )

    print("\nDone.")
    print(f"  Script checkpoint: {CHECKPOINT_PATH}")
    print(f"  Plot: {PLOT_OUTPUT}")


if __name__ == '__main__':
    main()
