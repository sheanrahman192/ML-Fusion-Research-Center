# CNN-GRU hybrid variant of LSTM_50_Binary_Transitions.py
# Same input data, same binary classification target, same training pipeline.
# Architecture: 1D CNN feature extractor -> Unidirectional GRU -> Attention + NN -> Classifier

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(48)
torch.manual_seed(48)
if torch.cuda.is_available():
    torch.cuda.manual_seed(48)

# Prediction horizon in milliseconds
PREDICTION_HORIZON_MS = 50
# Variant suffix for saves (Suppressed vs Dithering/ELMing/Mitigated)
VARIANT_SUFFIX = '_supp_vs_dem'


class CNNGRUNet(nn.Module):
    """
    CNN-GRU hybrid for binary plasma state classification.

    Stage 1 (CNN): three 1D convolutional blocks extract local temporal patterns
                   directly from the (n_features, seq_len) input. Each block
                   uses Conv1d -> BatchNorm -> ReLU -> Dropout, with the second
                   block downsampling via stride=2 to compress the sequence.
    Stage 2 (GRU): a 2-layer unidirectional GRU consumes the CNN feature
                   sequence and models longer-range temporal dependencies. GRU
                   is chosen over LSTM for fewer parameters and faster training
                   while retaining gated memory.
    Stage 3 (Aggregation): attention pooling over the GRU output sequence
                           combined with the final GRU hidden state.
    Stage 4 (Classifier): NN head with BatchNorm + Dropout for regularization.

    Input  shape: (batch_size, n_features, sequence_length)  [same as LSTM script]
    Output shape: (batch_size, n_classes)
    """
    def __init__(self, n_features, n_classes=2,
                 cnn_channels=(32, 64, 64),
                 gru_hidden=64,
                 nn_hidden_sizes=(128, 64)):
        super(CNNGRUNet, self).__init__()

        # ---- 1D CNN feature extractor ----
        # Input is already (batch, channels=n_features, seq_len), perfect for Conv1d.
        c1, c2, c3 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),  # downsample seq_len /2
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ---- GRU temporal model ----
        # Unidirectional to preserve forward-in-time semantics (predicting future state).
        self.gru = nn.GRU(
            input_size=c3,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.4,
        )

        gru_output_size = gru_hidden  # unidirectional

        # ---- Attention pooling over GRU output sequence ----
        self.attention_weights = nn.Sequential(
            nn.Linear(gru_output_size, 1),
            nn.Softmax(dim=1),
        )

        # ---- NN head on top of last GRU hidden state ----
        nn_layers = []
        input_dim = gru_output_size
        for hidden_size in nn_hidden_sizes:
            nn_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.45),
            ])
            input_dim = hidden_size
        self.nn_layers = nn.Sequential(*nn_layers)

        # ---- Final classifier (combines NN features + attention-pooled features) ----
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

        # Parameter accounting
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for name, p in self.named_parameters() if name.startswith('cnn'))
        gru_params = sum(p.numel() for name, p in self.named_parameters() if name.startswith('gru'))
        attention_params = sum(p.numel() for name, p in self.named_parameters() if 'attention' in name)
        nn_params = sum(p.numel() for name, p in self.named_parameters() if name.startswith('nn_layers'))
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if name.startswith('classifier'))

        print(f"\n{'='*60}")
        print(f"CNN-GRU Model Parameter Count (Binary Classification):")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - CNN layers:    {cnn_params:,} ({cnn_params/total_params*100:.1f}%)")
        print(f"  - GRU layers:    {gru_params:,} ({gru_params/total_params*100:.1f}%)")
        print(f"  - Attention:     {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - NN head:       {nn_params:,} ({nn_params/total_params*100:.1f}%)")
        print(f"  - Classifier:    {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: Conv1D x3 -> GRU (unidirectional, 2-layer) -> Attention + NN -> Classifier")
        print(f"Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)

        # STAGE 1: CNN feature extraction (still channel-first)
        cnn_out = self.cnn(x)  # (batch_size, c3, seq_len/2)

        # Transpose for GRU: (batch_size, seq_len/2, c3)
        gru_in = cnn_out.transpose(1, 2)

        # STAGE 2: GRU temporal modeling
        gru_out, hidden = self.gru(gru_in)
        # gru_out:  (batch_size, seq_len/2, gru_hidden)
        # hidden:   (num_layers, batch_size, gru_hidden)

        # STAGE 3: Attention-pooled features over the GRU sequence
        attention = self.attention_weights(gru_out)            # (batch_size, seq_len/2, 1)
        attended_features = torch.sum(gru_out * attention, dim=1)  # (batch_size, gru_hidden)

        # Last GRU hidden state (top layer) for forward-in-time prediction
        final_hidden = gru_out[:, -1, :]  # (batch_size, gru_hidden)

        # STAGE 4: NN head on the last hidden state
        nn_features = self.nn_layers(final_hidden)

        # Combine NN features with attention-pooled features and classify
        combined = torch.cat([nn_features, attended_features], dim=1)
        output = self.classifier(combined)
        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, (float, int)):
            if isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha)
            else:
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PlasmaDataset(Dataset):
    """Dataset class for plasma data windows (returns channel-first tensors)."""
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Transpose to (n_features, sequence_length) — Conv1d/GRU upstream expect this layout
        return self.windows[idx].T, self.labels[idx]


def load_and_prepare_data():
    """Load and preprocess the plasma data — identical to the LSTM script."""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li',
                          'tritop', 'fs04_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # fs04 rate of change per shot (kept for parity with LSTM script's preprocessing)
    if 'fs04' in df_sorted.columns:
        fs04_values = df_sorted['fs04'].values
        times_temp = df_sorted['time'].values
        shots_temp = df_sorted['shot'].values

        fs04_rate_of_change = np.zeros(len(df_sorted))

        for shot_id in df_sorted['shot'].unique():
            shot_mask = shots_temp == shot_id
            shot_indices = np.where(shot_mask)[0]
            if len(shot_indices) > 1:
                fs04_diff = np.diff(fs04_values[shot_indices])
                time_diff = np.diff(times_temp[shot_indices])
                time_diff_safe = np.where(time_diff == 0, 1, time_diff)
                rate = fs04_diff / time_diff_safe
                fs04_rate_of_change[shot_indices[0]] = 0.0
                fs04_rate_of_change[shot_indices[1:]] = rate

        df_sorted['fs04_rate_of_change'] = fs04_rate_of_change

    print(f"Using {len(selected_features)} features: {selected_features}")

    X = df_sorted[selected_features].values
    y = df_sorted['state'].values
    times = df_sorted['time'].values
    shots = df_sorted['shot'].values

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(times)
    X = X[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {Counter(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, times, shots, selected_features, scaler


def create_windows_with_random_shot_split(X, y, times, shots, window_size=150, prediction_horizon_ms=50):
    """Create windows split by shot number; predict state at future time."""
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by SHOT NUMBER (not individual data points)")
    print("Binary classification: Suppressed (0) vs Dithering/ELMing/Mitigated (1)")

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots: {n_shots}")

    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)

    train_shots = set(shuffled_shots[:train_size])
    val_shots = set(shuffled_shots[train_size:train_size + val_size])
    test_shots = set(shuffled_shots[train_size + val_size:])

    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")

    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []

    train_current_states = []
    val_current_states = []
    test_current_states = []

    binary_label_mapping = {1: 0, 2: 1, 3: 1, 4: 1}

    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        if shot_id in train_shots:
            target_windows = train_windows
            target_labels = train_labels
            target_current_states = train_current_states
        elif shot_id in val_shots:
            target_windows = val_windows
            target_labels = val_labels
            target_current_states = val_current_states
        else:
            target_windows = test_windows
            target_labels = test_labels
            target_current_states = test_current_states

        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        for i in range(len(shot_indices) - window_size + 1):
            window = shot_X[i:i + window_size]

            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms

            current_label = shot_labels[i + window_size - 1]

            future_local_idx = np.searchsorted(shot_times, target_time)

            if future_local_idx >= len(shot_times):
                windows_skipped_no_future += 1
                continue

            future_label = shot_labels[future_local_idx]

            if int(current_label) not in binary_label_mapping or int(future_label) not in binary_label_mapping:
                windows_skipped_invalid_label += 1
                continue

            if not np.isnan(window).any() and not np.isinf(window).any():
                target_windows.append(window)
                target_labels.append(binary_label_mapping[int(future_label)])
                target_current_states.append(binary_label_mapping[int(current_label)])
                windows_created += 1

    train_windows = np.array(train_windows, dtype=np.float32)
    train_labels = np.array(train_labels)
    train_current_states = np.array(train_current_states)
    val_windows = np.array(val_windows, dtype=np.float32)
    val_labels = np.array(val_labels)
    val_current_states = np.array(val_current_states)
    test_windows = np.array(test_windows, dtype=np.float32)
    test_labels = np.array(test_labels)
    test_current_states = np.array(test_current_states)

    print(f"\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Skipped (no future data): {windows_skipped_no_future:,}")
    print(f"  Skipped (invalid label): {windows_skipped_invalid_label:,}")

    print(f"\nCreated windows:")
    print(f"  Train: {len(train_windows)} windows from {len(train_shots)} shots")
    print(f"  Val: {len(val_windows)} windows from {len(val_shots)} shots")
    print(f"  Test: {len(test_windows)} windows from {len(test_shots)} shots")

    print(f"\nLabel distribution (binary):")
    print(f"  Train: {Counter(train_labels)}")
    print(f"  Val: {Counter(val_labels)}")
    print(f"  Test: {Counter(test_labels)}")

    print(f"\nTransition statistics:")
    train_transitions = np.sum(train_current_states != train_labels)
    val_transitions = np.sum(val_current_states != val_labels)
    test_transitions = np.sum(test_current_states != test_labels)
    print(f"  Train: {train_transitions:,} transitions ({train_transitions/len(train_labels)*100:.1f}%)")
    print(f"  Val: {val_transitions:,} transitions ({val_transitions/len(val_labels)*100:.1f}%)")
    print(f"  Test: {test_transitions:,} transitions ({test_transitions/len(test_labels)*100:.1f}%)")

    print(f"\nOversampling transition cases in training set...")
    train_windows, train_labels, train_current_states = oversample_transitions(
        train_windows, train_labels, train_current_states
    )

    print(f"After oversampling:")
    print(f"  Train: {len(train_windows)} windows")
    print(f"  Label distribution: {Counter(train_labels)}")
    train_transitions_after = np.sum(train_current_states != train_labels)
    print(f"  Transitions: {train_transitions_after:,} ({train_transitions_after/len(train_labels)*100:.1f}%)")

    return (train_windows, train_labels, train_current_states,
            val_windows, val_labels, val_current_states,
            test_windows, test_labels, test_current_states)


def oversample_transitions(windows, labels, current_states, transition_multiplier=3, problematic_multiplier=5):
    """Oversample transition cases, especially Suppressed -> Dithering/ELMing/Mitigated."""
    transition_mask = current_states != labels
    problematic_mask = (current_states == 0) & (labels == 1)

    transition_indices = np.where(transition_mask)[0]
    problematic_indices = np.where(problematic_mask)[0]
    non_transition_indices = np.where(~transition_mask)[0]

    print(f"  Before oversampling:")
    print(f"    Total samples: {len(windows)}")
    print(f"    Transition cases: {len(transition_indices)}")
    print(f"    Problematic transitions (0->1): {len(problematic_indices)}")
    print(f"    Non-transition cases: {len(non_transition_indices)}")

    oversampled_windows = [windows[i] for i in non_transition_indices]
    oversampled_labels = [labels[i] for i in non_transition_indices]
    oversampled_current_states = [current_states[i] for i in non_transition_indices]

    regular_transition_indices = transition_indices[~np.isin(transition_indices, problematic_indices)]
    for idx in regular_transition_indices:
        oversampled_windows.append(windows[idx])
        oversampled_labels.append(labels[idx])
        oversampled_current_states.append(current_states[idx])
        for _ in range(transition_multiplier - 1):
            oversampled_windows.append(windows[idx])
            oversampled_labels.append(labels[idx])
            oversampled_current_states.append(current_states[idx])

    for idx in problematic_indices:
        oversampled_windows.append(windows[idx])
        oversampled_labels.append(labels[idx])
        oversampled_current_states.append(current_states[idx])
        for _ in range(problematic_multiplier - 1):
            oversampled_windows.append(windows[idx])
            oversampled_labels.append(labels[idx])
            oversampled_current_states.append(current_states[idx])

    oversampled_windows = np.array(oversampled_windows, dtype=np.float32)
    oversampled_labels = np.array(oversampled_labels)
    oversampled_current_states = np.array(oversampled_current_states)

    print(f"  After oversampling:")
    print(f"    Total samples: {len(oversampled_windows)}")
    print(f"    Problematic transitions (0->1): {np.sum((oversampled_current_states == 0) & (oversampled_labels == 1))}")

    return oversampled_windows, oversampled_labels, oversampled_current_states


def train_model(model, train_loader, val_loader, device, class_weights_tensor, n_epochs=50):
    """Train the model with Focal Loss and class weights."""
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 25

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

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

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_cnn_gru_50ms_binary_transitions{VARIANT_SUFFIX}.pth')
            patience_counter = 0
            print(f"  New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def find_optimal_threshold(model, val_loader, device):
    """Find optimal decision threshold on validation set."""
    model.eval()
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            val_probs.extend(probs[:, 1].cpu().numpy())
            val_labels.extend(batch_y.numpy())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= threshold).astype(int)
        if len(np.unique(preds)) > 1:
            f1 = f1_score(val_labels, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"\nOptimal threshold found: {best_threshold:.4f} (F1 score: {best_f1:.4f})")
    return best_threshold


def evaluate_model(model, test_loader, device, class_names, threshold=0.5):
    """Evaluate the model on test set using threshold-based prediction."""
    print(f"\nEvaluating with threshold: {threshold:.4f}")
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)

            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)

            pos_class_probs = probs[:, 1].cpu().numpy()
            preds = (pos_class_probs >= threshold).astype(int)

            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"\nROC AUC Score: {auc:.4f}")

    return all_preds, all_labels, all_probs


def analyze_transition_effectiveness(all_preds, all_labels, all_current_states, all_probs, class_names):
    """Analyze model effectiveness on state transitions."""
    print("\n" + "="*60)
    print("TRANSITION EFFECTIVENESS ANALYSIS")
    print("="*60)
    print(f"Analyzing predictions for points where future state ({PREDICTION_HORIZON_MS}ms) differs from current state")
    print("="*60)

    transition_mask = all_current_states != all_labels
    n_transitions = np.sum(transition_mask)
    n_total = len(all_labels)

    print(f"\nTransition Statistics:")
    print(f"  Total test samples: {n_total:,}")
    print(f"  Transition cases: {n_transitions:,} ({n_transitions/n_total*100:.2f}%)")
    print(f"  Non-transition cases: {n_total - n_transitions:,} ({(n_total - n_transitions)/n_total*100:.2f}%)")

    if n_transitions == 0:
        print("\n  No transitions found in test set. Cannot perform transition analysis.")
        return

    transition_preds = all_preds[transition_mask]
    transition_labels = all_labels[transition_mask]
    transition_probs = all_probs[transition_mask] if len(all_probs.shape) > 1 else None

    transition_acc = accuracy_score(transition_labels, transition_preds)
    transition_precision = precision_score(transition_labels, transition_preds, average='weighted', zero_division=0)
    transition_recall = recall_score(transition_labels, transition_preds, average='weighted', zero_division=0)
    transition_f1 = f1_score(transition_labels, transition_preds, average='weighted', zero_division=0)

    print(f"\nTransition Case Metrics:")
    print(f"  Accuracy: {transition_acc:.4f}")
    print(f"  Precision (weighted): {transition_precision:.4f}")
    print(f"  Recall (weighted): {transition_recall:.4f}")
    print(f"  F1-Score (weighted): {transition_f1:.4f}")

    if transition_probs is not None and len(np.unique(transition_labels)) > 1:
        transition_auc = roc_auc_score(transition_labels, transition_probs[:, 1])
        print(f"  ROC AUC: {transition_auc:.4f}")

    print(f"\nTransition Case Classification Report:")
    print(classification_report(transition_labels, transition_preds, target_names=class_names, digits=4))

    transition_cm = confusion_matrix(transition_labels, transition_preds)
    print(f"\nTransition Case Confusion Matrix:")
    print(f"  Predicted ->")
    print(f"  Actual v")
    print(f"  {transition_cm}")

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\nComparison with Overall Performance:")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Transition accuracy: {transition_acc:.4f}")
    print(f"  Difference: {transition_acc - overall_acc:.4f} ({((transition_acc - overall_acc)/overall_acc*100):.2f}%)")

    print(f"\nTransition Type Breakdown:")
    transition_types = {
        'Suppressed -> Dithering/ELMing/Mitigated': (all_current_states == 0) & (all_labels == 1),
        'Dithering/ELMing/Mitigated -> Suppressed': (all_current_states == 1) & (all_labels == 0),
    }

    for trans_type, mask in transition_types.items():
        n_type = np.sum(mask)
        if n_type > 0:
            type_preds = all_preds[mask]
            type_labels = all_labels[mask]
            type_acc = accuracy_score(type_labels, type_preds)
            print(f"  {trans_type}: {n_type:,} cases, Accuracy: {type_acc:.4f}")

    print("="*60)


def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names):
    """Plot training curves and confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'CNN-GRU Training and Validation Loss ({PREDICTION_HORIZON_MS}ms - Binary)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'CNN-GRU Training and Validation Accuracy ({PREDICTION_HORIZON_MS}ms - Binary)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix (Binary)')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    cm_counts = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Counts - Binary)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(f'cnn_gru_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Results saved to 'cnn_gru_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png'")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("CNN-GRU Hybrid Model for Binary Plasma Classification")
    print("=" * 60)
    print("Architecture: Conv1D x3 -> GRU (unidirectional) -> Attention + NN -> Classifier")
    print("Window: 150 datapoints BEFORE current time")
    print(f"Prediction: {PREDICTION_HORIZON_MS}ms INTO THE FUTURE")
    print("Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, scaler = load_and_prepare_data()

    train_X, train_y, train_current_states, val_X, val_y, val_current_states, test_X, test_y, test_current_states = create_windows_with_random_shot_split(
        X, y, times, shots, prediction_horizon_ms=PREDICTION_HORIZON_MS
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    print("\nCalculating class weights from training data...")
    class_counts = np.bincount(train_y, minlength=2)
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights: {dict(zip(range(len(class_weights)), class_weights))}")

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model = CNNGRUNet(n_features=len(features), n_classes=2).to(device)

    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)

    import time
    start_time = time.time()
    with torch.no_grad():
        _ = model(test_batch)
    forward_time = time.time() - start_time
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {forward_time:.3f} seconds")

    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, class_weights_tensor, n_epochs=50
    )

    print("\nLoading best model...")
    model.load_state_dict(torch.load(f'best_cnn_gru_50ms_binary_transitions{VARIANT_SUFFIX}.pth'))

    optimal_threshold = find_optimal_threshold(model, val_loader, device)

    class_names = ['Suppressed', 'Dithering/ELMing/Mitigated']
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names, threshold=optimal_threshold
    )

    analyze_transition_effectiveness(all_preds, all_labels, test_current_states, all_probs, class_names)

    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training Complete! (CNN-GRU, predicting {PREDICTION_HORIZON_MS}ms into the future - Binary)")
    print("=" * 60)


if __name__ == "__main__":
    main()
