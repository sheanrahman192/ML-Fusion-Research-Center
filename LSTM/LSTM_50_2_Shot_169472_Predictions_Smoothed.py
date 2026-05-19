#!/usr/bin/env python3
"""
Smoothed prediction visualization for LSTM 50ms / 2-state holdout shot 169472.

Same training, holdout split, and inference as LSTM_50_2_Shot_169472_Predictions.py, but
maps class-1 probabilities onto the shot timeline, applies a rolling mean, and plots
smoothed probability plus thresholded states (reduces window-to-window flicker).

Outputs:
  - Checkpoint: LSTM/best_lstm_50ms_2state_random_shot_{RUN_TAG}.pth (shared with parent)
  - Plot: Isolated Shot Comprehensive Analysis/isolated_shot_169472_lstm_smoothed_predictions.png

Usage:
  MPLBACKEND=Agg python LSTM_50_2_Shot_169472_Predictions_Smoothed.py [data_csv] [run_tag]
  MPLBACKEND=Agg python LSTM_50_2_Shot_169472_Predictions_Smoothed.py --skip-train
  SMOOTH_WINDOW_MS=40 MPLBACKEND=Agg python LSTM_50_2_Shot_169472_Predictions_Smoothed.py
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
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

_DEFAULT_CSV = '/mnt/homes/sr4240/my_folder/combined_database.csv'
_DEFAULT_RUN_TAG = 'shot_169472_holdout'


def _parse_cli_early():
    """Parse CLI before importing parent (parent reads sys.argv at import)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('data_csv', nargs='?', default=_DEFAULT_CSV)
    parser.add_argument('run_tag', nargs='?', default=_DEFAULT_RUN_TAG)
    return parser.parse_known_args()[0]


_EARLY_ARGS = _parse_cli_early()
_argv_backup = sys.argv
sys.argv = [sys.argv[0], _EARLY_ARGS.data_csv, _EARLY_ARGS.run_tag]

import LSTM_50_2_Random_Shot as base  # noqa: E402
import LSTM_50_2_Shot_169472_Predictions as parent  # noqa: E402

sys.argv = _argv_backup

HOLDOUT_SHOT = parent.HOLDOUT_SHOT


def _checkpoint_path(run_tag: str) -> str:
    return os.path.join(SCRIPT_DIR, f'best_lstm_50ms_2state_random_shot_{run_tag}.pth')
PLOT_OUTPUT = os.path.join(
    '/mnt/homes/sr4240/my_folder/Isolated Shot Comprehensive Analysis',
    f'isolated_shot_{HOLDOUT_SHOT}_lstm_smoothed_predictions.png',
)
SMOOTH_WINDOW_MS = float(os.environ.get('SMOOTH_WINDOW_MS', '40'))  # ±20 ms at ~1 ms sampling
PREDICTION_THRESHOLD = float(os.environ.get('PREDICTION_THRESHOLD', '0.5'))
STATE_COLORS = {0: '#2E8B57', 1: '#DC143C'}
STATE_NAMES = {0: 'Suppressed', 1: 'Other (Dithering/Mitigated/ELMing)'}


def predict_windows_with_probs(model, windows, labels, device, batch_size=4096):
    """Return argmax labels and P(class=1) for each window."""
    dataset = base.PlasmaDataset(windows, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    preds, probs_pos = [], []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            batch_probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(outputs, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            probs_pos.extend(batch_probs[:, 1].cpu().numpy())
    return (
        np.array(preds, dtype=int),
        np.array(probs_pos, dtype=np.float64),
        labels,
    )


def map_values_to_future_times(time, window_end_times, values, horizon_ms):
    """Average window values that map to the same future timestep index."""
    mapped = np.full(len(time), np.nan, dtype=np.float64)
    counts = np.zeros(len(time), dtype=np.int32)
    for value, window_end_time in zip(values, window_end_times):
        future_time = window_end_time + horizon_ms
        time_idx = int(np.argmin(np.abs(time - future_time)))
        if time_idx >= len(mapped):
            continue
        if counts[time_idx] == 0:
            mapped[time_idx] = float(value)
        else:
            mapped[time_idx] += float(value)
        counts[time_idx] += 1
    has_data = counts > 0
    mapped[has_data] /= counts[has_data]
    return mapped, has_data


def _interpolate_nans(series):
    """Linear interpolate NaNs; edge NaNs filled from first/last valid sample."""
    out = series.astype(np.float64).copy()
    valid = ~np.isnan(out)
    if not valid.any():
        return out
    if valid.all():
        return out
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def smooth_timeline_series(series, time, window_ms=SMOOTH_WINDOW_MS):
    """Rolling-mean smooth after filling gaps on the shot timeline."""
    filled = _interpolate_nans(series)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    if dt <= 0:
        dt = 1.0
    window_pts = max(3, int(round(window_ms / dt)))
    if window_pts % 2 == 0:
        window_pts += 1
    smoothed = (
        pd.Series(filled)
        .rolling(window=window_pts, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return smoothed, window_pts, dt


def plot_smoothed_predictions(
    shot_df,
    raw_probs,
    smoothed_probs,
    labels,
    window_end_times,
    raw_preds,
    shot_number=HOLDOUT_SHOT,
    output_path=PLOT_OUTPUT,
    threshold=PREDICTION_THRESHOLD,
):
    """Three-panel plot: actual states, smoothed probability, smoothed accuracy."""
    plt.style.use('seaborn-v0_8')
    time = shot_df['time'].values
    fs04 = shot_df['fs04'].values
    actual_binary = shot_df['state_binary'].values.astype(int)
    horizon = base.PREDICTION_HORIZON_MS

    true_mapped, _ = map_values_to_future_times(time, window_end_times, labels, horizon)
    raw_prob_mapped, prob_valid = map_values_to_future_times(
        time, window_end_times, raw_probs, horizon,
    )
    raw_pred_mapped, _ = map_values_to_future_times(time, window_end_times, raw_preds, horizon)

    smoothed_pred = (smoothed_probs >= threshold).astype(int)
    valid_eval = prob_valid & ~np.isnan(true_mapped)
    smoothed_acc = (
        accuracy_score(true_mapped[valid_eval].astype(int), smoothed_pred[valid_eval])
        if valid_eval.any() else float('nan')
    )
    raw_acc = (
        accuracy_score(labels, raw_preds)
        if len(labels) else float('nan')
    )

    y_lo, y_hi = fs04.min(), fs04.max()
    xlim = (time.min(), time.max())

    fig, axes = plt.subplots(3, 1, figsize=(24, 20))
    fig.suptitle(
        f'Plasma State Predictions (Smoothed) — Shot {shot_number} — LSTM 50ms 2-state',
        fontsize=20, fontweight='bold', y=0.995,
    )

    # (a) Actual states
    ax1 = axes[0]
    ax1.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    for state in (0, 1):
        mask = actual_binary == state
        if mask.any():
            ax1.fill_between(
                time, y_lo, y_hi, where=mask, alpha=0.3,
                color=STATE_COLORS[state], label=f'Actual: {STATE_NAMES[state]}',
            )
    ax1.set_ylabel('fs04', fontsize=16)
    ax1.set_title('(a) Observed Plasma States', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
               frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(xlim)

    # (b) Smoothed P(class=1) with thresholded regions
    ax2 = axes[1]
    ax2.plot(time, fs04, 'k-', linewidth=1, alpha=0.5, label='fs04')
    ax2b = ax2.twinx()
    if prob_valid.any():
        ax2b.plot(
            time[prob_valid], raw_prob_mapped[prob_valid],
            color='#9E9E9E', linewidth=0.8, alpha=0.55, label='Raw P(Other)',
        )
    ax2b.plot(time, smoothed_probs, color='#1f77b4', linewidth=2.2, label='Smoothed P(Other)')
    ax2b.axhline(threshold, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
                 label=f'Threshold ({threshold:.2f})')
    ax2b.set_ylim(-0.02, 1.02)
    ax2b.set_ylabel('P(Other)', fontsize=16, color='#1f77b4')
    ax2b.tick_params(axis='y', labelcolor='#1f77b4')

    for state in (0, 1):
        mask = smoothed_pred == state
        if mask.any():
            ax2.fill_between(
                time, y_lo, y_hi, where=mask, alpha=0.28,
                color=STATE_COLORS[state],
                label=f'Smoothed pred: {STATE_NAMES[state]}',
            )
    ax2.set_ylabel('fs04', fontsize=16)
    ax2.set_title(
        f'(b) Smoothed Predictions (rolling {SMOOTH_WINDOW_MS:.0f} ms)',
        fontsize=18, fontweight='bold',
    )
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2, labels1 + labels2,
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11,
        frameon=True, fancybox=True, shadow=True,
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xlim)

    # (c) Accuracy: raw argmax vs smoothed threshold
    ax3 = axes[2]
    ax3.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')

    raw_valid = prob_valid & ~np.isnan(raw_pred_mapped) & ~np.isnan(true_mapped)
    raw_correct = raw_valid & (raw_pred_mapped == true_mapped)
    raw_incorrect = raw_valid & (raw_pred_mapped != true_mapped)
    smooth_correct = valid_eval & (smoothed_pred == true_mapped.astype(int))
    smooth_incorrect = valid_eval & (smoothed_pred != true_mapped.astype(int))

    ax3.fill_between(time, y_lo, y_hi, where=raw_correct, alpha=0.2,
                     color='#81C784', label='Raw correct')
    ax3.fill_between(time, y_lo, y_hi, where=raw_incorrect, alpha=0.2,
                     color='#E57373', label='Raw incorrect')
    ax3.fill_between(time, y_lo, y_hi, where=smooth_correct, alpha=0.45,
                     color='green', label='Smoothed correct')
    ax3.fill_between(time, y_lo, y_hi, where=smooth_incorrect, alpha=0.45,
                     color='red', label='Smoothed incorrect')

    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('(c) Raw vs Smoothed Prediction Accuracy', fontsize=18, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11,
               frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(xlim)
    ax3.text(
        0.02, 0.98,
        f'Window argmax accuracy: {raw_acc:.3f}\n'
        f'Smoothed ({SMOOTH_WINDOW_MS:.0f} ms) accuracy: {smoothed_acc:.3f}',
        transform=ax3.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.subplots_adjust(hspace=0.3, bottom=0.06, top=0.92)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Cap pixel count for IDE/browser viewing (~10k px wide); dpi=2000 created 42k×39k PNGs.
    w_in, h_in = fig.get_size_inches()
    max_px = 12_000
    save_dpi = min(800, max(100, int(max_px / max(w_in, h_in))))
    fig.savefig(
        output_path,
        format='png',
        dpi=save_dpi,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.05,
        pil_kwargs={'optimize': True},
    )
    plt.close(fig)
    print(f"  Saved at dpi={save_dpi} (~{int(w_in * save_dpi)}×{int(h_in * save_dpi)} px before tight bbox)")
    print(f"Smoothed plot saved to '{output_path}'")
    return raw_acc, smoothed_acc


def train_if_needed(model, train_loader, val_loader, device, checkpoint_path, skip_train=False):
    if skip_train:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"--skip-train set but checkpoint not found: {checkpoint_path}"
            )
        print(f"Skipping training; loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return

    print("\nStarting training (20 epochs, early stopping)...")
    base.train_model(model, train_loader, val_loader, device, n_epochs=20)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")


def main():
    args = _EARLY_ARGS
    checkpoint_path = _checkpoint_path(args.run_tag)
    parent.DATA_CSV = args.data_csv
    parent.RUN_TAG = args.run_tag
    base.DATA_CSV = args.data_csv
    base.RUN_TAG = args.run_tag

    os.chdir(SCRIPT_DIR)
    print("=" * 60)
    print("LSTM 50ms 2-state — smoothed predictions for holdout shot 169472")
    print("=" * 60)
    print(f"  CSV: {args.data_csv}")
    print(f"  RUN_TAG: {args.run_tag}")
    print(f"  Smooth window: {SMOOTH_WINDOW_MS} ms")
    print(f"  Threshold: {PREDICTION_THRESHOLD}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, _ = base.load_and_prepare_data()

    (train_X, train_y, _, val_X, val_y, _, _, _, _) = (
        parent.create_windows_excluding_holdout(X, y, times, shots)
    )

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
    train_if_needed(
        model, train_loader, val_loader, device, checkpoint_path,
        skip_train=args.skip_train,
    )

    holdout_windows, holdout_labels, holdout_end_times = parent.build_holdout_shot_windows(
        X, y, times, shots,
    )
    print(f"Holdout windows for shot {HOLDOUT_SHOT}: {len(holdout_windows)}")

    holdout_preds, holdout_probs, _ = predict_windows_with_probs(
        model, holdout_windows, holdout_labels, device,
    )
    print(f"Holdout window argmax accuracy: {accuracy_score(holdout_labels, holdout_preds):.4f}")

    shot_df = parent.load_holdout_shot_dataframe()
    time = shot_df['time'].values
    raw_prob_mapped, prob_valid = map_values_to_future_times(
        time, holdout_end_times, holdout_probs, base.PREDICTION_HORIZON_MS,
    )
    smoothed_probs, window_pts, dt = smooth_timeline_series(
        np.where(prob_valid, raw_prob_mapped, np.nan), time,
    )
    print(f"Smoothing: median dt={dt:.2f} ms, window={window_pts} samples")

    plot_smoothed_predictions(
        shot_df,
        holdout_probs,
        smoothed_probs,
        holdout_labels,
        holdout_end_times,
        holdout_preds,
    )

    print("\nDone.")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Plot: {PLOT_OUTPUT}")


if __name__ == '__main__':
    main()
