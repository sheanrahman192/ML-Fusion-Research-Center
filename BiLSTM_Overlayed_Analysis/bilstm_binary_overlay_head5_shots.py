#!/usr/bin/env python3
"""
Overlay visualization for the binary LSTM-First-NN (BiLSTM_NN_Center_Point_Binary).

Selects a fixed set of shots and verifies that all model features impute cleanly
(ffill/bfill with no remaining NaNs — same gate as a valid propagate_labels run).

Saves:
  - one composite PNG (4 rows × N columns)
  - one comprehensive PNG per shot

Each shot shows only fs04 and fs_sum_max_smoothed:
  rows 1–2: fs04 — predicted binary, then accuracy + confidence
  rows 3–4: fs_sum_max_smoothed — predicted binary, then accuracy + confidence

Requires: BiLSTM_NN_Architecture/bilstm_nn_binary_complete_model.pth and
          best_lstm_first_nn_binary.pth (see propagate_labels_binary).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

NN_DIR = Path(__file__).resolve().parent.parent / "BiLSTM_NN_Architecture"
sys.path.insert(0, str(NN_DIR))
from propagate_labels_binary import load_binary_model, predict_shot_labels_binary  # noqa: E402

DATA_CSV = "/mnt/homes/sr4240/my_folder/LABEL_PROPAGATED_DATABASE.csv"
OUT_DIR = Path(__file__).resolve().parent
TARGET_SHOTS = [186202, 173988, 173984, 173983, 174740]
REMOVE_SHOT = 191675  # matches BiLSTM_NN_Center_Point_Binary.load_and_prepare_data


def first_n_unique_shots_with_usable_features(
    df: pd.DataFrame, features: list[str], n: int
) -> list:
    """
    Walk shots in order of first CSV appearance; keep the first n shots whose feature
    matrix has no NaNs after forward/backward fill (matches inference preconditions).
    """
    out: list[int] = []
    for sid in df["shot"].drop_duplicates(keep="first"):
        sub = df[df["shot"] == sid].sort_values("time")
        if len(sub) == 0:
            continue
        X = sub[features].values
        X2 = pd.DataFrame(X, columns=features).ffill().bfill().values
        if np.isnan(X2).any():
            continue
        out.append(int(sid))
        if len(out) >= n:
            break
    return out


def state_to_binary(states: np.ndarray) -> np.ndarray:
    """Training remap: Suppressed=0 (state 1), ELMy=1 (states 2,3,4); NaN for state 0."""
    s = np.asarray(states)
    out = np.full(s.shape, np.nan, dtype=np.float64)
    out[s == 1] = 0.0
    out[np.isin(s, (2, 3, 4))] = 1.0
    return out


def _signal_y_limits(y: np.ndarray) -> tuple[float, float]:
    lo = np.nanmin(y)
    hi = np.nanmax(y)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0, 1.0
    return float(lo), float(hi)


def _plot_predicted_binary_panel(
    ax,
    time: np.ndarray,
    y_sig: np.ndarray,
    y_pred: np.ndarray,
    ylabel: str,
    title: str,
    shot_title: str | None = None,
) -> None:
    colors = {0: "#2E8B57", 1: "#DC143C"}
    names = {0: "Suppressed", 1: "ELMy"}
    lo, hi = _signal_y_limits(y_sig)

    ax.plot(time, y_sig, "k-", lw=0.8, alpha=0.75)
    for lab in (0, 1):
        m = y_pred == lab
        if m.any():
            ax.fill_between(
                time,
                lo,
                hi,
                where=m,
                alpha=0.35,
                color=colors[lab],
                label=f"Pred: {names[lab]}",
            )
    ax.set_ylabel(ylabel)
    if shot_title:
        ax.set_title(f"{shot_title}\n{title}")
    else:
        ax.set_title(title)
    h, _ = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_accuracy_confidence_panel(
    ax,
    time: np.ndarray,
    y_sig: np.ndarray,
    shot_data: pd.DataFrame,
    ylabel: str,
    title: str,
    show_xlabel: bool,
) -> None:
    lo, hi = _signal_y_limits(y_sig)
    conf = shot_data["prediction_confidence"].to_numpy()
    ax.plot(time, y_sig, "k-", lw=1.0, alpha=0.8)

    pc = shot_data["prediction_correct"]
    correct = pc == True
    incorrect = pc == False
    if correct.any():
        ax.fill_between(
            time,
            lo,
            hi,
            where=correct.to_numpy(),
            alpha=0.4,
            color="green",
            label="Correct",
        )
    if incorrect.any():
        ax.fill_between(
            time,
            lo,
            hi,
            where=incorrect.to_numpy(),
            alpha=0.4,
            color="red",
            label="Incorrect",
        )

    vm = np.isfinite(conf)
    if vm.any():
        sc = ax.scatter(
            time[vm],
            y_sig[vm],
            c=conf[vm],
            cmap="viridis",
            s=10,
            alpha=0.85,
            edgecolors="none",
        )
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="confidence")

    if show_xlabel:
        ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    h3, _ = ax.get_legend_handles_labels()
    if h3:
        ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_shot_panels(axes: np.ndarray, shot_data: pd.DataFrame, shot_id: int) -> None:
    """Fill a column of 4 axes: fs04/fs_sum predicted + accuracy; no observed-binary panel."""
    time = shot_data["time"].to_numpy()
    fs04 = shot_data["fs04"].to_numpy()
    fs_sum = shot_data["fs_sum_max_smoothed"].to_numpy()
    y_pred = shot_data["predicted_binary"].to_numpy()

    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

    shot_hdr = f"Shot {shot_id}"
    _plot_predicted_binary_panel(
        ax1, time, fs04, y_pred, "fs04", "Predicted binary", shot_title=shot_hdr
    )
    _plot_accuracy_confidence_panel(
        ax2, time, fs04, shot_data, "fs04", "Accuracy + confidence", show_xlabel=False
    )
    _plot_predicted_binary_panel(
        ax3, time, fs_sum, y_pred, "fs_sum_max_smoothed", "Predicted binary"
    )
    _plot_accuracy_confidence_panel(
        ax4,
        time,
        fs_sum,
        shot_data,
        "fs_sum_max_smoothed",
        "Accuracy + confidence",
        show_xlabel=True,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV)
    df = df[df["shot"] != REMOVE_SHOT].copy()

    model, scaler, ckpt = load_binary_model(device)
    features = ckpt["features"]
    window_size = ckpt["window_size"]

    selected_shots = TARGET_SHOTS
    missing_shots = [sid for sid in selected_shots if sid not in set(df["shot"].unique())]
    if missing_shots:
        raise RuntimeError(f"Requested shots not found in {DATA_CSV}: {missing_shots}")
    print(f"Selected shots: {selected_shots}")

    shot_frames: list[tuple[int, pd.DataFrame]] = []
    for sid in selected_shots:
        shot_df = df[df["shot"] == sid].sort_values("time").reset_index(drop=True)
        required_cols = list(features) + ["fs04", "fs_sum_max_smoothed"]
        missing = [c for c in required_cols if c not in shot_df.columns]
        if missing:
            raise RuntimeError(f"Shot {sid} missing columns: {missing}")
        X = shot_df[features].values
        X2 = pd.DataFrame(X, columns=features).ffill().bfill().values
        if np.isnan(X2).any():
            raise RuntimeError(f"Shot {sid} has model feature NaNs after ffill/bfill.")

        preds, confidence = predict_shot_labels_binary(
            model,
            shot_df,
            scaler,
            features,
            window_size,
            device,
            return_confidence=True,
        )

        shot_df["predicted_binary"] = preds
        shot_df["prediction_confidence"] = confidence
        if "state_binary" in shot_df.columns:
            shot_df["actual_binary"] = shot_df["state_binary"].replace(-1, np.nan)
        else:
            shot_df["actual_binary"] = state_to_binary(shot_df["state"].values)

        eval_rows = shot_df["actual_binary"].notna() & (shot_df["predicted_binary"] >= 0)
        shot_df["prediction_correct"] = np.nan
        shot_df.loc[eval_rows, "prediction_correct"] = (
            shot_df.loc[eval_rows, "predicted_binary"].values
            == shot_df.loc[eval_rows, "actual_binary"].values
        )

        if eval_rows.any():
            acc = shot_df.loc[eval_rows, "prediction_correct"].mean()
            print(f"Shot {sid}: accuracy on labeled timesteps (excl. state 0): {acc:.4f}")
        else:
            print(f"Shot {sid}: no comparable labeled rows")

        shot_frames.append((sid, shot_df))

    n = len(selected_shots)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 18), squeeze=False)
    fig.suptitle(
        "Binary LSTM-First-NN — selected high-density/high-Ti suppressed shots\n"
        "Training model: BiLSTM_NN_Center_Point_Binary / propagate_labels_binary",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    for col, (sid, sdf) in enumerate(shot_frames):
        plot_shot_panels(axes[:, col], sdf, sid)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    composite = OUT_DIR / "bilstm_binary_overlay_selected_suppressed_shots.png"
    fig.savefig(composite, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved composite: {composite}")

    for sid, sdf in shot_frames:
        fig2, axes2 = plt.subplots(4, 1, figsize=(14, 16), squeeze=False)
        plot_shot_panels(axes2[:, 0], sdf, sid)
        fig2.suptitle(f"Binary LSTM overlay — shot {sid}", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        one = OUT_DIR / f"bilstm_binary_overlay_selected_shot_{sid}_comprehensive.png"
        fig2.savefig(one, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {one}")


if __name__ == "__main__":
    main()
