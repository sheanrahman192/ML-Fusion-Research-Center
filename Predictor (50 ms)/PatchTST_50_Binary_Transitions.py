#PatchTST variant of LSTM_50_Binary_Transitions.py - same inputs/outputs, different architecture

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
VARIANT_SUFFIX = '_combined_db_fixed_test'
DATA_CSV = '/mnt/homes/sr4240/my_folder/combined_database.csv'

# Held-out test shots (combined_database evaluation)
TEST_SHOTS = {
    169449, 169457, 169460, 169463, 169467, 169470, 169473, 169476, 169479,
    169500, 169503, 169506, 169959, 169966, 170063, 170070, 170090, 170114,
    175673, 175682, 175686, 175695, 175701, 179350, 179363, 183153, 183185,
    190276, 190284, 190733, 191662, 191665, 191674, 191683, 191686, 191689,
    191975, 191978, 191981, 191984, 191987, 191990,
}

# PatchTST hyperparameters
WINDOW_SIZE = 150
PATCH_LEN = 10
PATCH_STRIDE = 10  # non-overlapping patches → 15 patches per channel
D_MODEL = 96
N_HEADS = 6
N_ENCODER_LAYERS = 3
D_FF = 192
PATCH_DROPOUT = 0.3


class PatchTST(nn.Module):
    """
    PatchTST for binary plasma classification.

    Architecture:
      1. Split each channel's time series into patches (channel-independent).
      2. Linearly embed each patch into d_model.
      3. Add learnable positional encoding.
      4. Pass each channel's patch sequence through a shared Transformer encoder.
      5. Mean-pool patches per channel, concatenate channels, classify.

    Channel-independent processing (the key PatchTST idea) treats each feature's
    time series as an independent token sequence, sharing transformer weights
    across channels. This regularizes well with limited data and captures
    long-range temporal dependencies more effectively than recurrent models.
    """

    def __init__(self, n_features, n_classes=2, seq_len=WINDOW_SIZE,
                 patch_len=PATCH_LEN, stride=PATCH_STRIDE,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_ENCODER_LAYERS,
                 d_ff=D_FF, dropout=PATCH_DROPOUT):
        super(PatchTST, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        # Per-feature instance norm for non-stationarity (RevIN-lite)
        self.instance_norm = nn.InstanceNorm1d(n_features, affine=True)

        # Channel-independent patch embedding (shared weights across channels)
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Learnable positional encoding for patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        self.embed_dropout = nn.Dropout(dropout)

        # Shared Transformer encoder (channel-independent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Cross-channel attention to mix information after channel-independent encoding
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # Classification head: per-channel attention-weighted aggregation
        head_input = n_features * d_model
        self.classifier = nn.Sequential(
            nn.Linear(head_input, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout + 0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout + 0.15),
            nn.Linear(64, n_classes)
        )

        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed_params = sum(p.numel() for n, p in self.named_parameters() if 'patch_embedding' in n or 'pos_embedding' in n)
        transformer_params = sum(p.numel() for n, p in self.named_parameters() if 'transformer' in n or 'encoder_norm' in n)
        attention_params = sum(p.numel() for n, p in self.named_parameters() if 'channel_attention' in n)
        classifier_params = sum(p.numel() for n, p in self.named_parameters() if 'classifier' in n)

        print(f"\n{'='*60}")
        print(f"PatchTST Model Parameter Count (Binary Classification):")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - Patch embedding + pos: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
        print(f"  - Transformer encoder: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
        print(f"  - Channel attention: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: PatchTST (channel-independent) → Channel attn → Classifier")
        print(f"Patches: {self.num_patches} per channel, patch_len={patch_len}, stride={stride}")
        print(f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        print(f"Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        B, C, L = x.shape

        # Instance normalize each channel's time series
        x = self.instance_norm(x)

        # Create patches: (B, C, num_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Channel-independent: collapse batch and channel dims
        # (B, C, P, patch_len) → (B*C, P, patch_len)
        patches = patches.reshape(B * C, self.num_patches, self.patch_len)

        # Embed each patch
        emb = self.patch_embedding(patches)  # (B*C, P, d_model)

        # Add positional encoding
        emb = emb + self.pos_embedding
        emb = self.embed_dropout(emb)

        # Transformer encoder (shared across channels)
        out = self.transformer(emb)  # (B*C, P, d_model)
        out = self.encoder_norm(out)

        # Attention pool over patches (per channel)
        attn = self.channel_attention(out)  # (B*C, P, 1)
        pooled = torch.sum(out * attn, dim=1)  # (B*C, d_model)

        # Restore channel dim and concatenate channels
        pooled = pooled.reshape(B, C * self.d_model)

        # Classify
        logits = self.classifier(pooled)
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
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

    fs_col = ('fs_sum_past_max_smoothed' if 'combined_database' in DATA_CSV
              else 'fs04_past_max_smoothed')
    important_features = ['iln3iamp', 'betan', 'density', 'li', 'tritop', fs_col]
    selected_features = [f for f in important_features if f in df.columns]

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Calculate fs04 rate of change (per-shot)
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
    if 'state_binary' in df_sorted.columns:
        y = df_sorted['state_binary'].values.astype(np.float64)
        print("Label source: state_binary")
    else:
        y = df_sorted['state'].values
        print("Label source: state")
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


def create_windows_with_random_shot_split(X, y, times, shots, window_size=WINDOW_SIZE, prediction_horizon_ms=50):
    """Create windows and perform random split BY SHOT - predicting state at future time"""
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by SHOT NUMBER (not individual data points)")
    print("Binary classification: Suppressed (0) vs Dithering/ELMing/Mitigated (1)")

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots: {n_shots}")

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
        print(f"  Missing (no valid rows after filtering): {missing_test}")

    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []

    train_current_states = []
    val_current_states = []
    test_current_states = []

    # combined_database encoding: 0=Suppressed, 1=Dithering, 2=ELMing, 3=Mitigated; -1 invalid
    binary_label_mapping = {0: 0, 1: 1, 2: 1, 3: 1}

    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        sid_int = int(shot_id)
        if sid_int in train_shots:
            target_windows = train_windows
            target_labels = train_labels
            target_current_states = train_current_states
        elif sid_int in val_shots:
            target_windows = val_windows
            target_labels = val_labels
            target_current_states = val_current_states
        elif sid_int in test_shots:
            target_windows = test_windows
            target_labels = test_labels
            target_current_states = test_current_states
        else:
            continue

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
    """Oversample transition cases, especially Suppressed → Dithering/ELMing/Mitigated"""
    transition_mask = current_states != labels
    problematic_mask = (current_states == 0) & (labels == 1)

    transition_indices = np.where(transition_mask)[0]
    problematic_indices = np.where(problematic_mask)[0]
    non_transition_indices = np.where(~transition_mask)[0]

    print(f"  Before oversampling:")
    print(f"    Total samples: {len(windows)}")
    print(f"    Transition cases: {len(transition_indices)}")
    print(f"    Problematic transitions (0→1): {len(problematic_indices)}")
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
    print(f"    Problematic transitions (0→1): {np.sum((oversampled_current_states == 0) & (oversampled_labels == 1))}")

    return oversampled_windows, oversampled_labels, oversampled_current_states


def train_model(model, train_loader, val_loader, device, class_weights_tensor, n_epochs=50):
    """Train the PatchTST model with Focal Loss + class weights, AdamW + cosine schedule"""
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction='mean')

    # AdamW with weight decay works better for transformers than Adam
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # Warmup + cosine schedule helps transformer convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

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
            # Gradient clipping for transformer stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_patchtst_50ms_binary_transitions{VARIANT_SUFFIX}.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def find_optimal_threshold(model, val_loader, device):
    """Find optimal decision threshold on validation set"""
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
    """Evaluate the model on test set using threshold-based prediction"""
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
    """Analyze model effectiveness on state transitions"""
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
    print(f"  Predicted →")
    print(f"  Actual ↓")
    print(f"  {transition_cm}")

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\nComparison with Overall Performance:")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Transition accuracy: {transition_acc:.4f}")
    print(f"  Difference: {transition_acc - overall_acc:.4f} ({((transition_acc - overall_acc)/overall_acc*100):.2f}%)")

    print(f"\nTransition Type Breakdown:")
    transition_types = {
        'Suppressed → Dithering/ELMing/Mitigated': (all_current_states == 0) & (all_labels == 1),
        'Dithering/ELMing/Mitigated → Suppressed': (all_current_states == 1) & (all_labels == 0)
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
    """Plot training curves and confusion matrix"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Validation Loss ({PREDICTION_HORIZON_MS}ms PatchTST - Binary)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Training and Validation Accuracy ({PREDICTION_HORIZON_MS}ms PatchTST - Binary)')
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
    plt.savefig(f'patchtst_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Results saved to 'patchtst_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png'")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("PatchTST Model for Binary Plasma Classification")
    print("=" * 60)
    print("Architecture: Channel-Independent PatchTST → Channel attn → Classifier")
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

    model = PatchTST(n_features=len(features), n_classes=2).to(device)

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
        model, train_loader, val_loader, device, class_weights_tensor, n_epochs=20
    )

    print("\nLoading best model...")
    model.load_state_dict(torch.load(f'best_patchtst_50ms_binary_transitions{VARIANT_SUFFIX}.pth'))

    optimal_threshold = find_optimal_threshold(model, val_loader, device)

    class_names = ['Suppressed', 'Dithering/ELMing/Mitigated']
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names, threshold=optimal_threshold)

    analyze_transition_effectiveness(all_preds, all_labels, test_current_states, all_probs, class_names)

    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training Complete! (Predicting {PREDICTION_HORIZON_MS}ms into the future - Binary Classification - PatchTST)")
    print("=" * 60)


if __name__ == "__main__":
    main()
