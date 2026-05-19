"""
LSTM (architecture from LSTM_50_2_Random_Shot.py) evaluated on combined_database.csv
with a deterministic train/test split by shot for cross-model comparison with
PatchTST and iTransformer Binary-Transitions evaluations.

Key differences vs the original LSTM_50_2_Random_Shot.py:
  - Data source     : /mnt/homes/sr4240/my_folder/combined_database.csv
  - Target          : `state_binary` column directly (0=Suppressed, 1=Other),
                      rows with state_binary == -1 (uncertain) are dropped.
                      Equivalent to iTransformer/PatchTST binary target.
  - Split           : Deterministic by shot. 42 listed shots = TEST.
                      Every other shot in the CSV = TRAIN. A 10% slice of the
                      train shots is held out as VAL purely for early stopping
                      / LR scheduling so the original training procedure stays
                      intact (no leakage into TEST).
  - Metrics         : Accuracy, per-class & macro Precision/Recall/F1,
                      confusion matrix, ROC-AUC, per-shot accuracy.
  - Architecture / loss / optimizer / batch / epochs UNCHANGED.
"""

import os
import time as time_module
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_recall_fscore_support,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ----- Configuration -----------------------------------------------------
PREDICTION_HORIZON_MS = 50
WINDOW_SIZE = 150
N_CLASSES = 2

DATA_CSV = '/mnt/homes/sr4240/my_folder/combined_database.csv'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'best_lstm_50ms_2state_binary_transitions_eval.pth')
RESULTS_PNG_PATH = os.path.join(SCRIPT_DIR, 'lstm_50ms_2state_binary_transitions_eval_results.png')

# 42 held-out test shots (per evaluation spec). Every other shot in the CSV
# becomes the training set.
TEST_SHOTS = [
    169449, 169457, 169460, 169463, 169467, 169470, 169473, 169476, 169479,
    169500, 169503, 169506, 169959, 169966, 170063, 170070, 170090, 170114,
    175673, 175682, 175686, 175695, 175701, 179350, 179363, 183153, 183185,
    190276, 190284, 190733, 191662, 191665, 191674, 191683, 191686, 191689,
    191975, 191978, 191981, 191984, 191987, 191990,
]

VAL_FRACTION_OF_TRAIN_SHOTS = 0.10  # held out from train shots only


# ----- Model (UNCHANGED from LSTM_50_2_Random_Shot.py) -------------------

class LSTMFirstNN(nn.Module):
    """LSTM-First-NN classifier - unidirectional LSTM -> NN -> classifier."""

    def __init__(self, n_features, n_classes=2, lstm_hidden=64,
                 nn_hidden_sizes=[128, 64]):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.4,
        )
        lstm_output_size = lstm_hidden

        nn_layers = []
        input_dim = lstm_output_size
        for hidden_size in nn_hidden_sizes:
            nn_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.45),
            ])
            input_dim = hidden_size
        self.nn_layers = nn.Sequential(*nn_layers)

        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "=" * 60)
        print("LSTM-First-NN Model Parameter Count (Binary Transitions Eval):")
        print("=" * 60)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print("Architecture: LSTM (unidirectional) -> NN -> Classifier")

    def forward(self, x):
        # x: (B, n_features, seq_len) -- transpose to (B, seq_len, n_features)
        x = x.transpose(1, 2)
        lstm_output, _ = self.lstm(x)

        attention = self.attention_weights(lstm_output)
        attended = torch.sum(lstm_output * attention, dim=1)

        final_hidden = lstm_output[:, -1, :]
        nn_features = self.nn_layers(final_hidden)

        combined = torch.cat([nn_features, attended], dim=1)
        return self.classifier(combined)


class PlasmaDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].T, self.labels[idx]


# ----- Data loading & windowing -----------------------------------------

def load_and_prepare_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df['shot'].nunique():,} unique shots")

    # Match the original feature set used by LSTM_50_2_Random_Shot.py
    important_features = ['iln3iamp', 'betan', 'density', 'li',
                          'tritop', 'fs_sum_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]
    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Use state_binary (already 0/1, with -1 = uncertain) as the classification
    # target. Equivalent to the iTransformer/PatchTST binary mapping.
    if 'state_binary' not in df_sorted.columns:
        raise RuntimeError("state_binary column not found in CSV")

    X = df_sorted[selected_features].values
    y = df_sorted['state_binary'].values  # 0, 1, or -1 (invalid)
    times = df_sorted['time'].values
    shots = df_sorted['shot'].values

    valid_mask = (
        ~np.isnan(X).any(axis=1)
        & ~np.isnan(y)
        & ~np.isnan(times)
    )
    X = X[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    print(f"After cleaning: X={X.shape}, label distribution={dict(Counter(y.astype(int)))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, times, shots, selected_features, scaler


def split_shots(shots_array, test_shots_list, val_fraction):
    unique = np.unique(shots_array).astype(np.int64)
    unique_set = set(int(s) for s in unique.tolist())
    requested = [int(s) for s in test_shots_list]
    test_shots = sorted(s for s in requested if s in unique_set)
    missing_test = sorted(set(requested) - set(test_shots))
    if missing_test:
        print(f"  NOTE: {len(missing_test)} requested test shots have no usable data "
              f"after NaN filtering and are dropped from the test set.")
        print(f"  Dropped: {missing_test}")

    train_pool = sorted(s for s in unique if s not in set(test_shots))

    rng = np.random.RandomState(42)
    train_pool_arr = np.array(train_pool)
    rng.shuffle(train_pool_arr)
    n_val = int(len(train_pool_arr) * val_fraction)
    val_shots = sorted(train_pool_arr[:n_val].tolist())
    train_shots = sorted(train_pool_arr[n_val:].tolist())

    print(f"\nShot split (deterministic, no leakage):")
    print(f"  Train shots: {len(train_shots)}")
    print(f"  Val   shots: {len(val_shots)} (10% of non-test pool, for early stopping only)")
    print(f"  Test  shots: {len(test_shots)} (held-out evaluation set)")
    return set(train_shots), set(val_shots), set(test_shots)


def create_windows(X, y, times, shots, train_shots, val_shots, test_shots,
                   window_size=WINDOW_SIZE,
                   prediction_horizon_ms=PREDICTION_HORIZON_MS):
    print(f"\nCreating windows of size {window_size} "
          f"(predicting {prediction_horizon_ms}ms ahead)...")
    print("Target: state_binary (Suppressed=0, Dithering/Mitigated/ELMing=1)")

    train_w, train_l, train_s = [], [], []
    val_w, val_l, val_s = [], [], []
    test_w, test_l, test_s = [], [], []

    valid_labels = {0, 1}
    skipped_no_future = 0
    skipped_invalid = 0
    created = 0

    unique_shots = np.unique(shots)
    for shot_id in unique_shots:
        idx_mask = np.where(shots == shot_id)[0]
        if len(idx_mask) < window_size:
            continue

        if shot_id in train_shots:
            tw, tl, ts = train_w, train_l, train_s
        elif shot_id in val_shots:
            tw, tl, ts = val_w, val_l, val_s
        elif shot_id in test_shots:
            tw, tl, ts = test_w, test_l, test_s
        else:
            continue  # shot not in any split

        shot_X = X[idx_mask]
        shot_y = y[idx_mask].astype(int)
        shot_t = times[idx_mask]

        for i in range(len(idx_mask) - window_size + 1):
            window = shot_X[i:i + window_size]
            window_end_t = shot_t[i + window_size - 1]
            target_t = window_end_t + prediction_horizon_ms

            future_local_idx = np.searchsorted(shot_t, target_t)
            if future_local_idx >= len(shot_t):
                skipped_no_future += 1
                continue

            future_label = int(shot_y[future_local_idx])
            if future_label not in valid_labels:
                skipped_invalid += 1
                continue

            if np.isnan(window).any() or np.isinf(window).any():
                continue

            tw.append(window)
            tl.append(future_label)
            ts.append(int(shot_id))
            created += 1

    train_w = np.array(train_w, dtype=np.float32)
    val_w = np.array(val_w, dtype=np.float32)
    test_w = np.array(test_w, dtype=np.float32)
    train_l = np.array(train_l, dtype=np.int64)
    val_l = np.array(val_l, dtype=np.int64)
    test_l = np.array(test_l, dtype=np.int64)
    train_s = np.array(train_s, dtype=np.int64)
    val_s = np.array(val_s, dtype=np.int64)
    test_s = np.array(test_s, dtype=np.int64)

    print(f"\nWindow creation statistics:")
    print(f"  Created:                {created:,}")
    print(f"  Skipped (no future):    {skipped_no_future:,}")
    print(f"  Skipped (invalid label): {skipped_invalid:,}")

    print(f"\nWindow counts:")
    print(f"  Train: {len(train_w):,} from {len(train_shots)} shots, "
          f"label dist={dict(Counter(train_l.tolist()))}")
    print(f"  Val:   {len(val_w):,} from {len(val_shots)} shots, "
          f"label dist={dict(Counter(val_l.tolist()))}")
    print(f"  Test:  {len(test_w):,} from {len(test_shots)} shots, "
          f"label dist={dict(Counter(test_l.tolist()))}")

    return (train_w, train_l, train_s,
            val_w, val_l, val_s,
            test_w, test_l, test_s)


# ----- Training (UNCHANGED hyperparameters) ------------------------------

def train_model(model, train_loader, val_loader, device, n_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        tp, tl = [], []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            tp.extend(preds.cpu().numpy())
            tl.extend(batch_y.cpu().numpy())

        model.eval()
        val_loss = 0.0
        vp, vl = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                vp.extend(preds.cpu().numpy())
                vl.extend(batch_y.cpu().numpy())

        train_acc = accuracy_score(tl, tp)
        val_acc = accuracy_score(vl, vp)
        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}  "
              f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")

        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"  New best val_acc={val_acc:.4f}, model saved.")
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    return train_losses, val_losses, train_accs, val_accs


# ----- Evaluation -------------------------------------------------------

def evaluate_model(model, test_loader, test_shots_arr, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    inf_start = time_module.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    inference_time_s = time_module.time() - inf_start

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    overall_acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', labels=[0, 1], zero_division=0
    )
    p_per, r_per, f_per, sup_per = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    auc = None
    if len(np.unique(all_labels)) > 1 and all_probs.shape[1] >= 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"\nROC AUC Score: {auc:.4f}")

    rows = []
    for s in sorted(np.unique(test_shots_arr)):
        m = test_shots_arr == s
        if not m.any():
            continue
        a = accuracy_score(all_labels[m], all_preds[m])
        rows.append((int(s), int(m.sum()),
                     int(all_preds[m].sum()),
                     int(all_labels[m].sum()), a))

    return all_preds, all_labels, all_probs, {
        'accuracy': overall_acc,
        'macro_p': p_macro, 'macro_r': r_macro, 'macro_f1': f_macro,
        'p_per_class': p_per, 'r_per_class': r_per, 'f1_per_class': f_per,
        'support_per_class': sup_per,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'per_shot': rows,
        'inference_time_s': inference_time_s,
    }


def plot_results(train_losses, val_losses, train_accs, val_accs,
                 all_preds, all_labels, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train', color='blue')
    axes[0, 0].plot(val_losses, label='Val', color='red')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train', color='blue')
    axes[0, 1].plot(val_accs, label='Val', color='red')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1], normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion matrix (normalized)')
    axes[1, 0].set_xlabel('Pred'); axes[1, 0].set_ylabel('True')

    cm_counts = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion matrix (counts)')
    axes[1, 1].set_xlabel('Pred'); axes[1, 1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(RESULTS_PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nResults figure saved to: {RESULTS_PNG_PATH}")


# ----- Main --------------------------------------------------------------

def main():
    print("=" * 60)
    print("LSTM Binary-Transitions Eval (combined_database.csv, 50ms horizon)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, _ = load_and_prepare_data(DATA_CSV)

    train_shots, val_shots, test_shots = split_shots(
        shots, TEST_SHOTS, VAL_FRACTION_OF_TRAIN_SHOTS
    )

    (train_X, train_y, train_s,
     val_X, val_y, val_s,
     test_X, test_y, test_s) = create_windows(
        X, y, times, shots, train_shots, val_shots, test_shots
    )

    if len(test_X) == 0:
        raise RuntimeError("No test windows created -- check shot list / horizon.")

    train_loader = DataLoader(PlasmaDataset(train_X, train_y),
                              batch_size=2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(PlasmaDataset(val_X, val_y),
                            batch_size=2048, shuffle=False, num_workers=0)
    test_loader = DataLoader(PlasmaDataset(test_X, test_y),
                             batch_size=2048, shuffle=False, num_workers=0)

    model = LSTMFirstNN(n_features=len(features), n_classes=N_CLASSES).to(device)

    # Forward-pass smoke check
    sample, _ = next(iter(train_loader))
    sample = sample.to(device)
    t0 = time_module.time()
    with torch.no_grad():
        _ = model(sample)
    print(f"Forward pass for batch of {sample.shape[0]}: {time_module.time() - t0:.3f}s")

    print("\n" + "=" * 60)
    print(f"Training (epochs=50, batch=2048, Adam lr=1e-3 -- unchanged)")
    print("=" * 60)
    train_t0 = time_module.time()
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=50,
    )
    train_time_s = time_module.time() - train_t0
    print(f"\nTotal training wall-clock: {train_time_s:.1f} s "
          f"({train_time_s/60.0:.2f} min)")

    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    class_names = ['Suppressed', 'ELMy']
    all_preds, all_labels, all_probs, metrics = evaluate_model(
        model, test_loader, test_s, device, class_names
    )

    plot_results(train_losses, val_losses, train_accs, val_accs,
                 all_preds, all_labels, class_names)

    print("\n" + "=" * 60)
    print("CROSS-MODEL SUMMARY (machine-readable)")
    print("=" * 60)
    print(f"model_name,LSTM_50_2_BinaryTransitions_eval")
    print(f"data_csv,{DATA_CSV}")
    print(f"target,state_binary (Suppressed=0 vs Other=1)")
    print(f"window,{WINDOW_SIZE}")
    print(f"horizon_ms,{PREDICTION_HORIZON_MS}")
    print(f"n_train_shots,{len(train_shots)}")
    print(f"n_val_shots,{len(val_shots)}")
    print(f"n_test_shots,{len(test_shots)}")
    print(f"n_train_samples,{len(train_X)}")
    print(f"n_val_samples,{len(val_X)}")
    print(f"n_test_samples,{len(test_X)}")
    print(f"accuracy,{metrics['accuracy']:.6f}")
    print(f"macro_precision,{metrics['macro_p']:.6f}")
    print(f"macro_recall,{metrics['macro_r']:.6f}")
    print(f"macro_f1,{metrics['macro_f1']:.6f}")
    for i, cn in enumerate(class_names):
        print(f"class{i}_{cn}_precision,{metrics['p_per_class'][i]:.6f}")
        print(f"class{i}_{cn}_recall,{metrics['r_per_class'][i]:.6f}")
        print(f"class{i}_{cn}_f1,{metrics['f1_per_class'][i]:.6f}")
        print(f"class{i}_{cn}_support,{int(metrics['support_per_class'][i])}")
    if metrics['roc_auc'] is not None:
        print(f"roc_auc,{metrics['roc_auc']:.6f}")
    print(f"train_wallclock_s,{train_time_s:.2f}")
    print(f"inference_wallclock_s,{metrics['inference_time_s']:.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
