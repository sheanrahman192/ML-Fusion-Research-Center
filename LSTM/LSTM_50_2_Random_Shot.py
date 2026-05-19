import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, cohen_kappa_score,
                             precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

DATA_CSV = sys.argv[1] if len(sys.argv) > 1 else '/mnt/homes/sr4240/my_folder/combined_database.csv'
RUN_TAG = sys.argv[2] if len(sys.argv) > 2 else 'combined_db_fixed_test'

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Prediction horizon in milliseconds
PREDICTION_HORIZON_MS = 50

# 2-state classification: Suppressed vs Other (Dithering + Mitigated + ELMing)
N_CLASSES = 2

# Held-out test shots (combined_database fixed evaluation set)
TEST_SHOTS = {
    169449, 169457, 169460, 169463, 169467, 169470, 169473, 169476, 169479,
    169500, 169503, 169506, 169959, 169966, 170063, 170070, 170090, 170114,
    175673, 175682, 175686, 175695, 175701, 179350, 179363, 183153, 183185,
    190276, 190284, 190733, 191662, 191665, 191674, 191683, 191686, 191689,
    191975, 191978, 191981, 191984, 191987, 191990,
}

# Speed knobs:
# - TRAIN_STRIDE > 1 subsamples training/val windows by stride (test always stride=1).
# - BATCH_SIZE: bigger batches use the A100 better.
TRAIN_STRIDE = 5
BATCH_SIZE = 4096

class LSTMFirstNN(nn.Module):
    """
    A hybrid model with LSTM processing FIRST (for temporal patterns)
    followed by NN layers (for feature transformation).
    Uses 150 datapoints BEFORE the classification point.
    Predicts state 50ms into the future.
    Unidirectional LSTM only (not bidirectional).
    2-state classification: Suppressed vs Other (Dithering/Mitigated/ELMing combined).
    """
    def __init__(self, n_features, n_classes=2, lstm_hidden=64, nn_hidden_sizes=[128, 64]):
        super(LSTMFirstNN, self).__init__()

        # LSTM processes the raw temporal data FIRST
        # Unidirectional for future prediction
        self.lstm = nn.LSTM(
            input_size=n_features,  # Direct input of raw features
            hidden_size=lstm_hidden,
            num_layers=2,  # Deeper LSTM for better temporal learning
            batch_first=True,
            bidirectional=False,  # Unidirectional for forward-in-time prediction
            dropout=0.4
        )

        # After LSTM, we have temporal features
        lstm_output_size = lstm_hidden  # Unidirectional

        # NN layers process the LSTM output
        nn_layers = []
        input_dim = lstm_output_size

        for hidden_size in nn_hidden_sizes:
            nn_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.45)
            ])
            input_dim = hidden_size

        self.nn_layers = nn.Sequential(*nn_layers)

        # Feature aggregation from sequence
        self.attention_weights = nn.Sequential(
            nn.Linear(lstm_output_size, 1),
            nn.Softmax(dim=1)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

        # Print detailed model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        lstm_params = sum(p.numel() for name, p in self.named_parameters() if 'lstm' in name)
        nn_params = sum(p.numel() for name, p in self.named_parameters() if 'nn_layers' in name)
        attention_params = sum(p.numel() for name, p in self.named_parameters() if 'attention' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)

        print(f"\n{'='*60}")
        print(f"LSTM-First-NN Model Parameter Count (2-state):")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - LSTM layers: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
        print(f"  - NN layers: {nn_params:,} ({nn_params/total_params*100:.1f}%)")
        print(f"  - Attention: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: LSTM (unidirectional) → NN → Classifier")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # Transpose for LSTM: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)

        # STEP 1: LSTM processes the temporal sequence
        lstm_output, (hidden, cell) = self.lstm(x)
        # lstm_output shape: (batch_size, seq_len, lstm_hidden)

        # STEP 2: Apply attention to aggregate temporal information
        attention = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attended_features = torch.sum(lstm_output * attention, dim=1)  # (batch_size, lstm_hidden)

        # STEP 3: Process the final LSTM hidden state through NN
        # Take the last hidden state (for future prediction)
        final_hidden = lstm_output[:, -1, :]  # (batch_size, lstm_hidden)

        # Process through NN layers
        nn_features = self.nn_layers(final_hidden)  # (batch_size, nn_hidden[-1])

        # STEP 4: Combine attended features with NN features
        combined = torch.cat([nn_features, attended_features], dim=1)

        # STEP 5: Final classification
        output = self.classifier(combined)

        return output

class PlasmaDataset(Dataset):
    """Dataset class for plasma data windows"""
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Transpose to get (n_features, sequence_length) format
        return self.windows[idx].T, self.labels[idx]

def load_and_prepare_data():
    """Load and preprocess the plasma data - includes time column for future prediction"""
    print("Loading data...")
    print(f"  CSV: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select only the specified 7 features
    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs_sum_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    print(f"Using {len(selected_features)} features: {selected_features}")

    # Sort by shot and time
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Keep ALL data (including state=0 and state=-1) for temporal context
    # We'll filter invalid labels only when creating prediction targets

    # Use state_binary (already 0/1) when present so propagated labels are picked up;
    # fall back to deriving from raw 4-state `state` for older CSVs.
    if 'state_binary' in df_sorted.columns:
        y = df_sorted['state_binary'].values.astype(np.float64)
        label_source = 'state_binary'
    else:
        raw = df_sorted['state'].values.astype(np.float64)
        y = np.where(raw == 0, 0.0, np.where(np.isin(raw, [1, 2, 3]), 1.0, np.nan))
        label_source = 'state (mapped 0->0, {1,2,3}->1)'
    print(f"Label source: {label_source}")

    X = df_sorted[selected_features].values
    times = df_sorted['time'].values
    shots = df_sorted['shot'].values

    # Drop rows where features, labels, or times are missing.
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(times)
    X = X[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution (binary): {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, times, shots, selected_features, scaler

def create_windows_with_random_shot_split(X, y, times, shots, window_size=150, prediction_horizon_ms=50):
    """Create windows and perform random split BY SHOT - predicting state at future time

    2-state classification: 0=Suppressed, 1=Other (Dithering + Mitigated + ELMing combined).
    Raw states 1, 2, 3 are all mapped to class 1.
    """
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by SHOT NUMBER (not individual data points)")
    print("2-state labels: Suppressed=0, Other (Dithering/Mitigated/ELMing)=1")

    # Get unique shots
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots: {n_shots}")

    # Test set = explicit TEST_SHOTS that are present in the valid data
    # Remaining shots → 90/10 train/val
    valid_set = {int(s) for s in unique_shots}
    requested_test = {int(s) for s in TEST_SHOTS}
    test_shots = requested_test & valid_set
    missing_test = sorted(requested_test - valid_set)
    remaining = np.array([s for s in unique_shots if int(s) not in test_shots])
    np.random.seed(42)
    shuffled_remaining = np.random.permutation(remaining)
    train_size = int(0.9 * len(shuffled_remaining))
    train_shots = set(shuffled_remaining[:train_size].astype(int).tolist())
    val_shots = set(shuffled_remaining[train_size:].astype(int).tolist())

    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")
    print(f"  Test shots requested: {len(requested_test)}, present in data: {len(test_shots)}")
    if missing_test:
        print(f"  Missing (filtered by NaN): {missing_test}")
    print(f"  Test shot IDs (present): {sorted(test_shots)}")

    # Create windows for each split
    train_windows, train_labels, train_current = [], [], []
    val_windows, val_labels, val_current = [], [], []
    test_windows, test_labels, test_current = [], [], []
    test_shot_ids, test_window_end_times = [], []

    # Labels are already binary (0=Suppressed, 1=Other/ELMy). -1 = uncertain (skip).
    label_mapping = {0: 0, 1: 1}
    valid_raw_labels = {0, 1}

    # Track statistics
    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0

    # Create windows per shot and assign to appropriate split
    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        # Determine which split this shot belongs to
        sid_int = int(shot_id)
        if sid_int in train_shots:
            target_windows = train_windows
            target_labels = train_labels
            target_current = train_current
            split_kind = 'train'
        elif sid_int in val_shots:
            target_windows = val_windows
            target_labels = val_labels
            target_current = val_current
            split_kind = 'val'
        elif sid_int in test_shots:
            target_windows = test_windows
            target_labels = test_labels
            target_current = test_current
            split_kind = 'test'
        else:
            continue

        # OPTIMIZATION: Extract shot data ONCE before the inner loop
        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        # Create windows for this shot. Stride>1 for train/val to speed up training;
        # test always uses stride 1 to preserve full evaluation density.
        stride = 1 if split_kind == 'test' else TRAIN_STRIDE
        for i in range(0, len(shot_indices) - window_size + 1, stride):
            window = shot_X[i:i + window_size]

            # Get the time at the end of the window
            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms

            # OPTIMIZATION: Use binary search O(log n) instead of full array scan O(n)
            future_local_idx = np.searchsorted(shot_times, target_time)

            if future_local_idx >= len(shot_times):
                # No future data available for this window
                windows_skipped_no_future += 1
                continue

            # Get the label at the future time point
            future_label = shot_labels[future_local_idx]
            current_label = shot_labels[i + window_size - 1]

            # Only create training example if target + current labels are valid
            if int(future_label) not in valid_raw_labels or int(current_label) not in valid_raw_labels:
                windows_skipped_invalid_label += 1
                continue

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                target_windows.append(window)
                target_labels.append(label_mapping[int(future_label)])
                target_current.append(label_mapping[int(current_label)])
                if split_kind == 'test':
                    test_shot_ids.append(sid_int)
                    test_window_end_times.append(float(window_end_time))
                windows_created += 1

    # Convert to numpy arrays
    train_windows = np.array(train_windows, dtype=np.float32)
    train_labels = np.array(train_labels)
    train_current = np.array(train_current)
    val_windows = np.array(val_windows, dtype=np.float32)
    val_labels = np.array(val_labels)
    val_current = np.array(val_current)
    test_windows = np.array(test_windows, dtype=np.float32)
    test_labels = np.array(test_labels)
    test_current = np.array(test_current)
    test_shot_ids = np.array(test_shot_ids, dtype=np.int64)
    test_window_end_times = np.array(test_window_end_times, dtype=np.float64)

    print(f"\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Skipped (no future data): {windows_skipped_no_future:,}")
    print(f"  Skipped (invalid label): {windows_skipped_invalid_label:,}")

    print(f"\nCreated windows:")
    print(f"  Train: {len(train_windows)} windows from {len(train_shots)} shots")
    print(f"  Val: {len(val_windows)} windows from {len(val_shots)} shots")
    print(f"  Test: {len(test_windows)} windows from {len(test_shots)} shots")

    print(f"\nLabel distribution (2-state):")
    print(f"  Train: {Counter(train_labels)}")
    print(f"  Val: {Counter(val_labels)}")
    print(f"  Test: {Counter(test_labels)}")

    n_trans_train = int(np.sum(train_current != train_labels))
    n_trans_val = int(np.sum(val_current != val_labels))
    n_trans_test = int(np.sum(test_current != test_labels))
    print(f"\nTransition counts (current != future):")
    print(f"  Train: {n_trans_train:,}  ({n_trans_train/max(len(train_labels),1)*100:.2f}%)")
    print(f"  Val:   {n_trans_val:,}  ({n_trans_val/max(len(val_labels),1)*100:.2f}%)")
    print(f"  Test:  {n_trans_test:,}  ({n_trans_test/max(len(test_labels),1)*100:.2f}%)")

    return (train_windows, train_labels, train_current,
            val_windows, val_labels, val_current,
            test_windows, test_labels, test_current,
            test_shot_ids, test_window_end_times)

def train_model(model, train_loader, val_loader, device, n_epochs=50):
    """Train the model"""
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Store predictions
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels_list, val_preds)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_lstm_50ms_2state_random_shot_{RUN_TAG}.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set with detailed metrics."""
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 60)
    print("OVERALL TEST METRICS")
    print("=" * 60)
    print("\nClassification Report (2-state):")
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=[0, 1], digits=4))

    test_acc = accuracy_score(all_labels, all_preds)
    test_bal_acc = (recall_score(all_labels, all_preds, pos_label=0, zero_division=0) +
                    recall_score(all_labels, all_preds, pos_label=1, zero_division=0)) / 2.0
    test_prec_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_rec_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_mcc = matthews_corrcoef(all_labels, all_preds)
    test_kappa = cohen_kappa_score(all_labels, all_preds)
    auc_pos = roc_auc_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else float('nan')
    ap_pos = average_precision_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else float('nan')

    print(f"\nAccuracy:           {test_acc:.4f}")
    print(f"Balanced accuracy:  {test_bal_acc:.4f}")
    print(f"Precision (wgt):    {test_prec_w:.4f}")
    print(f"Recall (wgt):       {test_rec_w:.4f}")
    print(f"F1 (weighted):      {test_f1_w:.4f}")
    print(f"F1 (macro):         {test_f1_macro:.4f}")
    print(f"ROC AUC (pos=1):    {auc_pos:.4f}")
    print(f"Avg Precision (PR): {ap_pos:.4f}")
    print(f"Matthews corr:      {test_mcc:.4f}")
    print(f"Cohen's kappa:      {test_kappa:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    print(f"\nConfusion matrix [rows=true, cols=pred] (counts):")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    cm_norm = confusion_matrix(all_labels, all_preds, labels=[0, 1], normalize='true')
    print(f"\nConfusion matrix (row-normalized):")
    print(f"  {cm_norm[0]}")
    print(f"  {cm_norm[1]}")

    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        prec_i = precision_score(all_labels, all_preds, pos_label=i, zero_division=0)
        rec_i = recall_score(all_labels, all_preds, pos_label=i, zero_division=0)
        f1_i = f1_score(all_labels, all_preds, pos_label=i, zero_division=0)
        sup_i = int(np.sum(all_labels == i))
        ovr = (all_labels == i).astype(int)
        auc_i = roc_auc_score(ovr, all_probs[:, i]) if len(np.unique(ovr)) > 1 else float('nan')
        print(f"  {name:>12s}: precision={prec_i:.4f} recall={rec_i:.4f} f1={f1_i:.4f} support={sup_i} auc={auc_i:.4f}")

    metrics_summary = {
        'accuracy': float(test_acc),
        'balanced_accuracy': float(test_bal_acc),
        'precision_weighted': float(test_prec_w),
        'recall_weighted': float(test_rec_w),
        'f1_weighted': float(test_f1_w),
        'f1_macro': float(test_f1_macro),
        'roc_auc': float(auc_pos),
        'avg_precision': float(ap_pos),
        'mcc': float(test_mcc),
        'cohens_kappa': float(test_kappa),
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'support_0_suppressed': int(np.sum(all_labels == 0)),
        'support_1_other': int(np.sum(all_labels == 1)),
        'n_test_windows': int(len(all_labels)),
    }

    return all_preds, all_labels, all_probs, metrics_summary


def analyze_transitions(all_preds, all_labels, current_states, all_probs, class_names):
    """Stratified metrics on points where current state != future state."""
    print("\n" + "=" * 60)
    print("TRANSITION-CASE ANALYSIS")
    print("=" * 60)
    transition_mask = current_states != all_labels
    n_t = int(np.sum(transition_mask))
    n_total = len(all_labels)
    print(f"  Total test windows: {n_total:,}")
    print(f"  Transition cases:   {n_t:,} ({n_t/n_total*100:.2f}%)")
    print(f"  Stationary cases:   {n_total - n_t:,} ({(n_total - n_t)/n_total*100:.2f}%)")

    summary = {'transition_count': n_t, 'stationary_count': n_total - n_t}
    if n_t == 0:
        print("  No transitions in test set — skipping stratified metrics.")
        return summary

    t_p = all_preds[transition_mask]; t_l = all_labels[transition_mask]
    t_pr = all_probs[transition_mask] if len(all_probs.shape) > 1 else None
    summary['transition_accuracy'] = float(accuracy_score(t_l, t_p))
    summary['transition_f1_weighted'] = float(f1_score(t_l, t_p, average='weighted', zero_division=0))
    summary['transition_recall_weighted'] = float(recall_score(t_l, t_p, average='weighted', zero_division=0))
    summary['transition_precision_weighted'] = float(precision_score(t_l, t_p, average='weighted', zero_division=0))
    if t_pr is not None and len(np.unique(t_l)) > 1:
        summary['transition_roc_auc'] = float(roc_auc_score(t_l, t_pr[:, 1]))
    print(f"\nTransition metrics:")
    for k in ['transition_accuracy', 'transition_precision_weighted', 'transition_recall_weighted',
              'transition_f1_weighted', 'transition_roc_auc']:
        if k in summary:
            print(f"  {k:>30s}: {summary[k]:.4f}")

    s_p = all_preds[~transition_mask]; s_l = all_labels[~transition_mask]
    summary['stationary_accuracy'] = float(accuracy_score(s_l, s_p))
    print(f"\nStationary-case accuracy: {summary['stationary_accuracy']:.4f}")

    print("\nTransition type breakdown:")
    types = {
        '0->1 (Suppressed -> Other)': (current_states == 0) & (all_labels == 1),
        '1->0 (Other -> Suppressed)': (current_states == 1) & (all_labels == 0),
    }
    for name, mask in types.items():
        n = int(np.sum(mask))
        if n == 0:
            continue
        acc = accuracy_score(all_labels[mask], all_preds[mask])
        print(f"  {name}: n={n} accuracy={acc:.4f}")
        summary[f'transition_{name.split()[0]}_count'] = n
        summary[f'transition_{name.split()[0]}_accuracy'] = float(acc)
    return summary


def per_shot_breakdown(all_preds, all_labels, test_shot_ids, top_k=5):
    """Print per-shot accuracy for the test set; useful for spotting outlier shots."""
    print("\n" + "=" * 60)
    print("PER-SHOT TEST METRICS")
    print("=" * 60)
    rows = []
    for sid in np.unique(test_shot_ids):
        mask = test_shot_ids == sid
        if mask.sum() < 10:
            continue
        rows.append((int(sid), int(mask.sum()),
                     float(accuracy_score(all_labels[mask], all_preds[mask])),
                     float(f1_score(all_labels[mask], all_preds[mask], average='weighted', zero_division=0))))
    rows.sort(key=lambda r: r[2])
    print(f"  {'shot':>10s}  {'n':>6s}  {'acc':>7s}  {'f1':>7s}")
    print(f"  worst {min(top_k, len(rows))}:")
    for sid, n, acc, f1 in rows[:top_k]:
        print(f"  {sid:>10d}  {n:>6d}  {acc:>7.4f}  {f1:>7.4f}")
    print(f"  best {min(top_k, len(rows))}:")
    for sid, n, acc, f1 in rows[-top_k:][::-1]:
        print(f"  {sid:>10d}  {n:>6d}  {acc:>7.4f}  {f1:>7.4f}")
    accs = np.array([r[2] for r in rows])
    print(f"\n  Per-shot accuracy: mean={accs.mean():.4f} std={accs.std():.4f} "
          f"min={accs.min():.4f} max={accs.max():.4f} (n_shots={len(rows)})")
    return [{'shot': sid, 'n': n, 'accuracy': acc, 'f1': f1} for sid, n, acc, f1 in rows]

def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names):
    """Plot training curves and confusion matrix"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Validation Loss ({PREDICTION_HORIZON_MS}ms, 2-state)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training accuracy
    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Training and Validation Accuracy ({PREDICTION_HORIZON_MS}ms, 2-state)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot confusion matrix (normalized) - 2 classes
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1], normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix (2-state)')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    # Plot confusion matrix (counts) - 2 classes
    cm_counts = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Counts, 2-state)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plot_path = f'lstm_{PREDICTION_HORIZON_MS}ms_2_random_shot_results_{RUN_TAG}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Results saved to '{plot_path}'")

import time

def main():
    """Main training pipeline - 2-state classification (Suppressed vs Other)"""
    print("=" * 60)
    print("LSTM-NN Model for Plasma Classification (2-state)")
    print("=" * 60)
    print("Architecture: Unidirectional LSTM → NN → Classifier")
    print("Window: 150 datapoints BEFORE current time")
    print(f"Prediction: {PREDICTION_HORIZON_MS}ms INTO THE FUTURE")
    print("Classes: Suppressed (0), Other (Dithering/Mitigated/ELMing combined) (1)")
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data (now includes times)
    X, y, times, shots, features, scaler = load_and_prepare_data()

    # Create windows and split BY SHOT (2-state labels)
    (train_X, train_y, train_current,
     val_X, val_y, val_current,
     test_X, test_y, test_current,
     test_shot_ids, test_window_end_times) = create_windows_with_random_shot_split(
        X, y, times, shots, prediction_horizon_ms=PREDICTION_HORIZON_MS
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Create data loaders
    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Create model - 2 classes
    model = LSTMFirstNN(n_features=len(features), n_classes=N_CLASSES).to(device)

    # Test forward pass speed
    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)

    start_time = time.time()
    with torch.no_grad():
        _ = model(test_batch)
    forward_time = time.time() - start_time
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {forward_time:.3f} seconds")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=20
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(f'best_lstm_50ms_2state_random_shot_{RUN_TAG}.pth'))

    # Evaluate on test set
    class_names = ['Suppressed', 'Other']
    all_preds, all_labels, all_probs, metrics_summary = evaluate_model(
        model, test_loader, device, class_names
    )

    # Transition + per-shot stratification
    transition_summary = analyze_transitions(all_preds, all_labels, test_current, all_probs, class_names)
    per_shot = per_shot_breakdown(all_preds, all_labels, test_shot_ids)

    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    # Final test accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    summary = {
        'run_tag': RUN_TAG,
        'data_csv': DATA_CSV,
        'overall': metrics_summary,
        'transition': transition_summary,
        'per_shot_test': per_shot,
        'split': {
            'n_train_windows': int(len(train_X)),
            'n_val_windows': int(len(val_X)),
            'n_test_windows': int(len(test_X)),
        },
        'training_curves': {
            'train_losses': [float(v) for v in train_losses],
            'val_losses': [float(v) for v in val_losses],
            'train_accs': [float(v) for v in train_accs],
            'val_accs': [float(v) for v in val_accs],
            'best_val_acc': float(max(val_accs)) if val_accs else None,
            'epochs_completed': int(len(train_losses)),
        },
    }
    summary_path = f'lstm_{PREDICTION_HORIZON_MS}ms_2_random_shot_metrics_{RUN_TAG}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics JSON saved to '{summary_path}'")

    print("\n" + "=" * 60)
    print(f"Training Complete! (2-state, predicting {PREDICTION_HORIZON_MS}ms into the future)")
    print("=" * 60)

if __name__ == "__main__":
    main()
