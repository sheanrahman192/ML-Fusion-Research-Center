#!/usr/bin/env python3
"""
Isolated-shot comprehensive visualization for BiLSTM binary (center-point windows).

Trains BiLSTM_NN_Center_Point_Binary on all shots except 169472, runs inference on
shot 169472 only, and saves a three-panel fs04 plot matching the Random Forest
isolated-shot layout (without prediction-confidence scatter or colorbar).

Output:
  Isolated Shot Comprehensive Analysis/isolated_shot_169472_bilstm_binary_comprehensive_analysis.png

Usage:
  MPLBACKEND=Agg python isolated_shot_169472_bilstm_binary_comprehensive.py
  MPLBACKEND=Agg python isolated_shot_169472_bilstm_binary_comprehensive.py --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import BiLSTM_NN_Center_Point_Binary as bilstm  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HOLDOUT_SHOT = 169472
WINDOW_SIZE = 150
PLASMA_CSV = '/mnt/homes/sr4240/my_folder/plasma_data.csv'
OUTPUT_DIR = '/mnt/homes/sr4240/my_folder/Isolated Shot Comprehensive Analysis'
OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    f'isolated_shot_{HOLDOUT_SHOT}_bilstm_binary_comprehensive_analysis.png',
)
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, f'best_bilstm_binary_holdout_{HOLDOUT_SHOT}.pth')

# Binary states (state_binary: 0 = Suppressed, 1 = ELMy)
BINARY_COLORS = {0: '#2E8B57', 1: '#DC143C'}
BINARY_NAMES = {0: 'Suppressed', 1: 'ELMy'}


def load_plasma_frame():
    """Load plasma_data.csv with the same cleaning/label padding as the binary trainer."""
    df = pd.read_csv(PLASMA_CSV)
    df = df[df['shot'] != 191675].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li', 'tritop', 'fs_sum_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    state_as_float = df_sorted['state'].replace(-1, np.nan)
    df_sorted = df_sorted.assign(state=state_as_float)
    df_sorted['state'] = (
        df_sorted.groupby('shot', group_keys=False)['state']
        .transform(lambda s: s.ffill().bfill())
    )

    df_filtered = df_sorted.dropna(subset=['state']).copy()
    df_filtered['state'] = df_filtered['state'].astype(int)
    df_filtered['state_binary'] = np.where(df_filtered['state'] == 0, 0, 1).astype(np.int64)

    feat_mask = ~df_filtered[selected_features].isna().any(axis=1)
    df_filtered = df_filtered.loc[feat_mask].copy()

    return df_filtered, selected_features


def scale_features_by_shot(df, features, holdout_shot=HOLDOUT_SHOT):
    """Fit StandardScaler on all rows except the holdout shot; transform everyone."""
    train_mask = df['shot'] != holdout_shot
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, features].values)

    X = scaler.transform(df[features].values).astype(np.float32)
    y = df['state_binary'].values.astype(np.int64)
    shots = df['shot'].values
    times = df['time'].values.astype(np.float64)
    return X, y, shots, times, scaler


def _windows_for_shot_block(X_sub, y_sub, shot_id, store_indices=False):
    """Center-point windows for one contiguous shot block (edge-padded)."""
    L = len(X_sub)
    if L < WINDOW_SIZE:
        pad_extra = WINDOW_SIZE - L
        left_extra = pad_extra // 2
        right_extra = pad_extra - left_extra
        first, last = X_sub[:1], X_sub[-1:]
        X_sub = np.vstack(
            [np.repeat(first, left_extra, axis=0), X_sub, np.repeat(last, right_extra, axis=0)]
        )
        y_sub = np.concatenate(
            [np.repeat(y_sub[:1], left_extra), y_sub, np.repeat(y_sub[-1:], right_extra)]
        )
        L = len(X_sub)
        row_offset = -left_extra
    else:
        row_offset = 0

    X_pad, pad_left = bilstm.edge_pad_shot_features(X_sub, WINDOW_SIZE)
    center_idx = WINDOW_SIZE // 2

    windows, labels, row_indices = [], [], []
    for t in range(L):
        start = pad_left + t - center_idx
        window = X_pad[start : start + WINDOW_SIZE]
        center_label = int(y_sub[t])
        if np.isnan(window).any() or np.isinf(window).any():
            continue
        windows.append(window)
        labels.append(center_label)
        if store_indices:
            row_indices.append(row_offset + t)

    meta = {'shot': shot_id}
    if store_indices:
        meta['row_indices'] = np.array(row_indices, dtype=int)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=int), meta


def create_train_val_windows(X, y, shots, holdout_shot=HOLDOUT_SHOT,
                             train_frac=0.7, val_frac=0.15):
    """Build train/val windows from all shots except holdout."""
    windows, labels, window_shots = [], [], []

    for shot_id in np.unique(shots):
        if int(shot_id) == int(holdout_shot):
            continue
        shot_mask = shots == shot_id
        X_sub = X[shot_mask]
        y_sub = y[shot_mask]
        w, lab, _ = _windows_for_shot_block(X_sub, y_sub, shot_id)
        if len(w) == 0:
            continue
        windows.append(w)
        labels.append(lab)
        window_shots.extend([shot_id] * len(w))

    windows = np.concatenate(windows, axis=0)
    labels = np.concatenate(labels, axis=0)
    window_shots = np.array(window_shots)

    rng = np.random.RandomState(42)
    unique_shots = rng.permutation(np.unique(window_shots))
    n_shots = len(unique_shots)
    train_end = int(train_frac * n_shots)
    val_end = int((train_frac + val_frac) * n_shots)

    train_list = unique_shots[:train_end].tolist()
    val_list = unique_shots[train_end:val_end].tolist()
    if n_shots >= 3:
        if len(val_list) == 0 and len(train_list) > 1:
            val_list.append(train_list.pop())
    elif n_shots == 2:
        train_list = [unique_shots[0]]
        val_list = []
    else:
        train_list = unique_shots.tolist()
        val_list = []

    train_mask = np.isin(window_shots, train_list)
    val_mask = np.isin(window_shots, val_list)
    print(
        f"Train/val windows: {train_mask.sum():,} / {val_mask.sum():,} "
        f"(shots train={len(train_list)}, val={len(val_list)})"
    )
    return windows[train_mask], labels[train_mask], windows[val_mask], labels[val_mask]


def build_holdout_shot_windows(X, y, shots, holdout_shot=HOLDOUT_SHOT):
    """Windows and row indices for the holdout shot only."""
    shot_mask = shots == holdout_shot
    if not shot_mask.any():
        raise ValueError(f"Shot {holdout_shot} not found after cleaning.")

    X_sub = X[shot_mask]
    y_sub = y[shot_mask]
    windows, labels, meta = _windows_for_shot_block(
        X_sub, y_sub, holdout_shot, store_indices=True,
    )
    if len(windows) == 0:
        raise ValueError(f"Shot {holdout_shot}: no valid windows.")
    return windows, labels, meta['row_indices']


def predict_windows(model, windows, labels, device, batch_size=128):
    loader = DataLoader(
        bilstm.PlasmaDataset(windows, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    return np.array(preds, dtype=int)


def load_holdout_shot_plot_frame(df, holdout_shot=HOLDOUT_SHOT):
    """Per-timestep frame for shot 169472 with fs04 and binary labels."""
    shot_df = df[df['shot'] == holdout_shot].copy()
    if shot_df.empty:
        raise ValueError(f"Shot {holdout_shot} not found.")
    shot_df = shot_df.sort_values('time').reset_index(drop=True)
    if 'fs04' not in shot_df.columns:
        raise ValueError("Column 'fs04' missing; cannot plot comprehensive analysis.")
    return shot_df


def assign_predictions_to_shot(shot_df, row_indices, predictions, labels):
    """Map window-center predictions onto shot rows."""
    n = len(shot_df)
    pred_col = np.full(n, np.nan)
    for idx, pred in zip(row_indices, predictions):
        if 0 <= idx < n:
            pred_col[idx] = pred

    out = shot_df.copy()
    out['predicted_state'] = pred_col
    valid = ~np.isnan(out['predicted_state'])
    pred_int = out['predicted_state'].fillna(-1).astype(int)
    out['prediction_correct'] = np.where(
        valid,
        out['state_binary'].values == pred_int.values,
        np.nan,
    )
    return out


def create_comprehensive_visualization(shot_data, holdout_shot=HOLDOUT_SHOT):
    """
    Three-panel fs04 plot (actual / predicted / accuracy).
    No confidence scatter or colorbar on the accuracy panel.
    """
    plt.style.use('seaborn-v0_8')

    time = shot_data['time'].values
    fs04 = shot_data['fs04'].values
    actual_binary = shot_data['state_binary'].values.astype(int)
    predicted = shot_data['predicted_state'].astype(float).values
    correct = shot_data['prediction_correct'].values

    y_lo, y_hi = fs04.min(), fs04.max()
    xlim = (time.min(), time.max())

    fig, axes = plt.subplots(3, 1, figsize=(24, 18))
    fig.suptitle(
        f'Plasma State Classification Analysis for Shot {holdout_shot} - BiLSTM (Split Based on Shot)',
        fontsize=20,
        fontweight='bold',
        y=0.995,
    )

    # (a) Observed plasma states (state_binary ground truth)
    ax1 = axes[0]
    ax1.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    for state in (0, 1):
        mask = actual_binary == state
        if mask.any():
            ax1.fill_between(
                time, y_lo, y_hi, where=mask, alpha=0.3,
                color=BINARY_COLORS[state],
                label=f'Actual: {BINARY_NAMES[state]}',
            )
    ax1.set_ylabel('fs04 Signal (a.u.)', fontsize=16)
    ax1.set_title('(a) Observed Plasma States', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)

    # (b) Predicted plasma states (binary BiLSTM)
    ax2 = axes[1]
    ax2.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    for state in (0, 1):
        mask = predicted == state
        if mask.any():
            ax2.fill_between(
                time, y_lo, y_hi, where=mask, alpha=0.3,
                color=BINARY_COLORS[state],
                label=f'Predicted: {BINARY_NAMES[state]}',
            )
    ax2.set_ylabel('fs04 Signal (a.u.)', fontsize=16)
    ax2.set_title('(b) Predicted Plasma States', fontsize=18, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)

    # (c) Prediction accuracy (no confidence coloring)
    ax3 = axes[2]
    ax3.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')

    valid = ~np.isnan(correct)
    correct_mask = valid & (correct == True)  # noqa: E712
    incorrect_mask = valid & (correct == False)  # noqa: E712

    if correct_mask.any():
        ax3.fill_between(time, y_lo, y_hi, where=correct_mask, alpha=0.4,
                         color='green', label='Correct Predictions')
    if incorrect_mask.any():
        ax3.fill_between(time, y_lo, y_hi, where=incorrect_mask, alpha=0.4,
                         color='red', label='Incorrect Predictions')

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04 Signal (a.u.)', fontsize=16)
    ax3.set_title('(c) Prediction Accuracy Analysis', fontsize=18, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(xlim)

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.subplots_adjust(hspace=0.3, bottom=0.06, top=0.92)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=800, bbox_inches='tight')
    plt.close(fig)
    print(f"Comprehensive plot saved to '{OUTPUT_PNG}'")


def train_holdout_model(model, train_loader, val_loader, device, n_epochs, checkpoint_path):
    """Train with early stopping; save best weights to checkpoint_path."""
    orig_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    try:
        bilstm.train_model(model, train_loader, val_loader, device, n_epochs=n_epochs)
        # train_model writes best_lstm_first_nn_binary.pth in SCRIPT_DIR
        default_ckpt = os.path.join(SCRIPT_DIR, 'best_lstm_first_nn_binary.pth')
        if os.path.isfile(default_ckpt):
            import shutil
            shutil.copy2(default_ckpt, checkpoint_path)
    finally:
        os.chdir(orig_cwd)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not created: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='BiLSTM binary isolated shot 169472 plot')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--skip-train', action='store_true',
                        help='Load existing holdout checkpoint and only plot')
    args = parser.parse_args()

    print('=' * 70)
    print(f'BiLSTM binary isolated-shot analysis — shot {HOLDOUT_SHOT}')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    df, features = load_plasma_frame()
    if not (df['shot'] == HOLDOUT_SHOT).any():
        raise ValueError(f'Shot {HOLDOUT_SHOT} not in {PLASMA_CSV}')

    X, y, shots, times, scaler = scale_features_by_shot(df, features)
    train_X, train_y, val_X, val_y = create_train_val_windows(X, y, shots)
    holdout_windows, holdout_labels, row_indices = build_holdout_shot_windows(X, y, shots)

    print(f'Holdout shot {HOLDOUT_SHOT}: {len(holdout_windows):,} windows')

    model = bilstm.LSTMFirstNN(n_features=len(features), n_classes=2).to(device)

    if args.skip_train:
        if not os.path.isfile(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f'--skip-train set but checkpoint missing: {CHECKPOINT_PATH}'
            )
        print(f'Loading checkpoint: {CHECKPOINT_PATH}')
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        if len(train_X) == 0:
            raise RuntimeError('Empty training set after excluding holdout shot.')
        train_loader = DataLoader(
            bilstm.PlasmaDataset(train_X, train_y),
            batch_size=args.batch_size, shuffle=True,
        )
        if len(val_X) > 0:
            val_loader = DataLoader(
                bilstm.PlasmaDataset(val_X, val_y),
                batch_size=args.batch_size, shuffle=False,
            )
        else:
            val_loader = DataLoader(
                bilstm.PlasmaDataset(train_X[: min(128, len(train_X))], train_y[: min(128, len(train_y))]),
                batch_size=args.batch_size, shuffle=False,
            )
            print('  (No val shots in split; using small train subset for early stopping.)')

        print(f'\nTraining (max {args.epochs} epochs, holdout excluded)...')
        train_holdout_model(
            model, train_loader, val_loader, device,
            n_epochs=args.epochs, checkpoint_path=CHECKPOINT_PATH,
        )
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    holdout_preds = predict_windows(model, holdout_windows, holdout_labels, device)
    holdout_acc = accuracy_score(holdout_labels, holdout_preds)
    print(f'Holdout center-point accuracy: {holdout_acc:.4f}')

    shot_df = load_holdout_shot_plot_frame(df)
    shot_plot = assign_predictions_to_shot(shot_df, row_indices, holdout_preds, holdout_labels)

    valid = ~shot_plot['prediction_correct'].isna()
    row_acc = shot_plot.loc[valid, 'prediction_correct'].mean()
    print(f'Row-aligned accuracy ({valid.sum():,} points): {row_acc:.4f}')

    create_comprehensive_visualization(shot_plot)
    print('Done.')


if __name__ == '__main__':
    main()
