"""
Binary label propagation using the trained BiLSTM-NN from BiLSTM_NN_Center_Point_Binary.py.

Uses the same edge padding as training: replicate first/last rows so each timestep is a
window center; shots shorter than window_size are expanded to window_size first.

Weights: best_lstm_first_nn_binary.pth (saved during training).
Scaler / features / window: bilstm_nn_binary_complete_model.pth; falls back to its
model_state_dict if best weights file is missing.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
# Training saves best_lstm_first_nn_binary.pth to CWD; try script dir then repo root.
METADATA_PATH = SCRIPT_DIR / "bilstm_nn_binary_complete_model.pth"


def _resolve_weights_path():
    for p in (
        SCRIPT_DIR / "best_lstm_first_nn_binary.pth",
        WORKSPACE_ROOT / "best_lstm_first_nn_binary.pth",
    ):
        if p.is_file():
            return p
    return SCRIPT_DIR / "best_lstm_first_nn_binary.pth"


class LSTMFirstNN(nn.Module):
    """Must match BiLSTM_NN_Center_Point_Binary.py (binary: n_classes=2)."""

    def __init__(self, n_features, n_classes=2, lstm_hidden=128, nn_hidden_sizes=(256, 128)):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        lstm_output_size = lstm_hidden * 2
        nn_layers = []
        input_dim = lstm_output_size
        for hidden_size in nn_hidden_sizes:
            nn_layers.extend(
                [
                    nn.Linear(input_dim, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.25),
                ]
            )
            input_dim = hidden_size
        self.nn_layers = nn.Sequential(*nn_layers)

        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        lstm_output, _ = self.lstm(x)
        attention = self.attention_weights(lstm_output)
        attended_features = torch.sum(lstm_output * attention, dim=1)
        final_hidden = lstm_output[:, -1, :]
        nn_features = self.nn_layers(final_hidden)
        combined = torch.cat([nn_features, attended_features], dim=1)
        return self.classifier(combined)


class InferenceDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.FloatTensor(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].T


def _is_full_checkpoint(obj):
    return isinstance(obj, dict) and "model_state_dict" in obj


def load_binary_model(device):
    """
    Load n_classes=2 model. Prefer weights from best_lstm_first_nn_binary.pth;
    scaler and feature list from bilstm_nn_binary_complete_model.pth.
    """
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing metadata checkpoint: {METADATA_PATH}\n"
            "Run BiLSTM_NN_Center_Point_Binary.py once to produce it (scaler + features)."
        )

    print(f"Loading metadata from: {METADATA_PATH}")
    try:
        meta = torch.load(METADATA_PATH, map_location=device, weights_only=False)
    except TypeError:
        meta = torch.load(METADATA_PATH, map_location=device)

    n_features = meta["n_features"]
    n_classes = meta.get("n_classes", 2)
    if n_classes != 2:
        raise ValueError(f"Expected binary checkpoint n_classes=2, got {n_classes}")

    lstm_hidden = meta["lstm_hidden"]
    nn_hidden_sizes = meta["nn_hidden_sizes"]
    features = meta["features"]

    model = LSTMFirstNN(
        n_features=n_features,
        n_classes=n_classes,
        lstm_hidden=lstm_hidden,
        nn_hidden_sizes=nn_hidden_sizes,
    ).to(device)

    weights_path = _resolve_weights_path()
    if weights_path.is_file():
        print(f"Loading weights from: {weights_path}")
        try:
            w = torch.load(weights_path, map_location=device, weights_only=False)
        except TypeError:
            w = torch.load(weights_path, map_location=device)
        if _is_full_checkpoint(w):
            model.load_state_dict(w["model_state_dict"])
        else:
            model.load_state_dict(w)
    else:
        print(
            "Warning: best_lstm_first_nn_binary.pth not found under script dir or repo root; "
            "using model_state_dict from metadata file."
        )
        model.load_state_dict(meta["model_state_dict"])

    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = meta["scaler_mean"]
    scaler.scale_ = meta["scaler_scale"]
    scaler.n_features_in_ = n_features

    print("Model loaded successfully!")
    print(f"  Features: {features}")
    print(f"  Window size: {meta['window_size']}")
    print(f"  Classes: {meta.get('class_names', ['Suppressed', 'ELMy'])}")
    if "test_accuracy" in meta:
        print(f"  Saved test accuracy (full run): {meta['test_accuracy']:.4f}")

    return model, scaler, meta


def _edge_pad_scaled(X_scaled, window_size):
    """Replicate first/last rows; matches BiLSTM_NN_Center_Point_Binary.edge_pad_shot_features."""
    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    n_samples, _ = X_scaled.shape
    center_idx = window_size // 2
    pad_left = center_idx
    pad_right = window_size - center_idx - 1
    if n_samples == 0:
        return X_scaled.astype(np.float32), pad_left
    left = np.repeat(X_scaled[:1], pad_left, axis=0)
    right = np.repeat(X_scaled[-1:], pad_right, axis=0)
    X_pad = np.vstack([left, X_scaled, right]).astype(np.float32)
    return X_pad, pad_left


def predict_shot_labels_binary(model, shot_data, scaler, features, window_size, device, batch_size=256,
                               return_confidence=False):
    """
    Edge replicate padding so every original timestep is a valid window center (same as training).
    Short shots are expanded to window_size with replicated edges first, then padded.
    -1 only if NaNs remain after fill, no rows, or a timestep's window is invalid (NaN/Inf).
    If return_confidence is True, returns (predictions, confidence) where confidence is NaN for -1 preds.
    """
    n_orig = len(shot_data)
    X = shot_data[features].values
    xdf = pd.DataFrame(X, columns=features)
    X = xdf.ffill().bfill().values

    if np.isnan(X).any():
        bad = np.full(n_orig, -1)
        if return_confidence:
            return bad, np.full(n_orig, np.nan)
        return bad

    X_scaled = (X - scaler.mean_) / scaler.scale_
    if n_orig == 0:
        empty = np.array([], dtype=np.int64)
        if return_confidence:
            return empty, np.array([], dtype=np.float64)
        return empty

    center_idx = window_size // 2
    # Index in working series -> original row index (training short-shot rule)
    left_extra = 0
    if n_orig < window_size:
        pad_extra = window_size - n_orig
        left_extra = pad_extra // 2
        right_extra = pad_extra - left_extra
        first = X_scaled[:1]
        last = X_scaled[-1:]
        X_scaled = np.vstack(
            [np.repeat(first, left_extra, axis=0), X_scaled, np.repeat(last, right_extra, axis=0)]
        ).astype(np.float32)

    n_work = len(X_scaled)
    X_pad, pad_left = _edge_pad_scaled(X_scaled, window_size)

    windows = []
    for t in range(n_work):
        start = pad_left + t - center_idx
        window = X_pad[start : start + window_size]
        if not np.isnan(window).any() and not np.isinf(window).any():
            windows.append(window)
        else:
            windows.append(None)

    valid_idx = [i for i, w in enumerate(windows) if w is not None]
    if not valid_idx:
        bad = np.full(n_orig, -1)
        if return_confidence:
            return bad, np.full(n_orig, np.nan)
        return bad

    win_arr = np.stack([windows[i] for i in valid_idx], axis=0).astype(np.float32)
    loader = DataLoader(InferenceDataset(win_arr), batch_size=batch_size, shuffle=False)

    preds_list = []
    conf_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            conf_list.extend(probs.max(dim=1).values.cpu().numpy())

    full_predictions = np.full(n_orig, -1, dtype=np.int64)
    full_confidence = np.full(n_orig, np.nan, dtype=np.float64)
    for j, t in enumerate(valid_idx):
        orig_k = t - left_extra
        if 0 <= orig_k < n_orig:
            full_predictions[orig_k] = int(preds_list[j])
            full_confidence[orig_k] = float(conf_list[j])

    if return_confidence:
        return full_predictions, full_confidence
    return full_predictions


def main():
    print("=" * 60)
    print("Binary label propagation (Suppressed=0, ELMy=1)")
    print("=" * 60)

    input_csv = "/mnt/homes/sr4240/my_folder/LABEL_PROPAGATED_DATABASE.csv"
    output_csv = input_csv
    output_column = "state_binary"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, scaler, checkpoint = load_binary_model(device)
    features = checkpoint["features"]
    window_size = checkpoint["window_size"]
    class_names = checkpoint.get("class_names", ["Suppressed", "ELMy"])

    print(f"\nLoading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} rows")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing features in data: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        return

    unique_shots = df["shot"].unique()
    print(f"Total unique shots: {len(unique_shots)}")

    df[output_column] = -1

    shot_stats = {"processed": 0, "skipped": 0, "too_short": 0}

    for shot_id in tqdm(unique_shots, desc="Processing shots"):
        shot_mask = df["shot"] == shot_id
        shot_data = df.loc[shot_mask].copy()
        shot_data = shot_data.sort_values("time")
        shot_indices = shot_data.index

        predictions = predict_shot_labels_binary(
            model, shot_data, scaler, features, window_size, device
        )

        if predictions is not None and len(predictions) == len(shot_indices):
            df.loc[shot_indices, output_column] = predictions
            if (predictions == -1).all():
                shot_stats["too_short"] += 1
            else:
                shot_stats["processed"] += 1
        else:
            shot_stats["skipped"] += 1

    print("\n" + "=" * 60)
    print("Prediction statistics")
    print("=" * 60)
    print(f"Shots processed: {shot_stats['processed']}")
    print(f"Shots too short (< {window_size}): {shot_stats['too_short']}")
    print(f"Shots skipped: {shot_stats['skipped']}")

    valid = df[output_column][df[output_column] >= 0]
    print(f"\nPredicted distribution ({output_column}):")
    for label in sorted(valid.unique()):
        count = (valid == label).sum()
        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        pct = count / len(valid) * 100 if len(valid) else 0
        print(f"  {int(label)} ({name}): {count:,} ({pct:.1f}%)")

    unpredicted = (df[output_column] == -1).sum()
    print(f"\nUnpredicted (-1): {unpredicted:,} ({unpredicted / len(df) * 100:.1f}%)")

    cols = df.columns.tolist()
    cols.remove(output_column)
    insert_at = min(2, len(cols))
    cols.insert(insert_at, output_column)
    df = df[cols]

    print(f"\nSaving to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Done.")

    print("\n" + "=" * 60)
    print("Binary propagation complete.")
    print(f"Column {output_column}: -1 = unknown edge/short, 0 = Suppressed, 1 = ELMy")
    print("=" * 60)


if __name__ == "__main__":
    main()
