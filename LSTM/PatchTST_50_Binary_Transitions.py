#PatchTST script for fall 2025 work - same I/O as LSTM_50_Binary_Transitions.py with transformer architecture

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


class PatchTST(nn.Module):
    """
    PatchTST: Patch Time Series Transformer for binary classification.

    Architecture (Nie et al., 2023, "A Time Series is Worth 64 Words"):
      1. Split each channel of the input sequence into overlapping patches.
      2. Project patches into d_model via a shared linear embedding.
      3. Add learnable positional encoding.
      4. Process patches with a Transformer encoder applied independently per
         channel (channel-independence with shared weights).
      5. Attention-pool across patches per channel to a single vector.
      6. Concatenate channel embeddings and classify.

    Channel-independence (each variable processed separately by shared
    transformer weights) is the core trick that makes PatchTST generalize
    well on multivariate time series compared to channel-mixing models.

    Uses 150 datapoints BEFORE the classification point.
    Predicts state 50ms into the future.
    Binary classification: Suppressed (0) vs Dithering/ELMing/Mitigated (1).
    """
    def __init__(self, n_features, n_classes=2, seq_len=150,
                 patch_len=15, stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256,
                 dropout=0.2, head_dropout=0.3):
        super(PatchTST, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        # Patch embedding: project each patch (patch_len-dim vector) to d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Learnable positional encoding (one vector per patch position)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder (shared across channels - channel independence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Attention pooling across patches (aggregates the patch sequence per channel)
        self.patch_attention = nn.Linear(d_model, 1)

        # Classification head: aggregates pooled features across all channels
        head_input_dim = n_features * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, n_classes)
        )

        # Print detailed model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embed_params = sum(p.numel() for n, p in self.named_parameters()
                           if 'patch_embedding' in n or 'pos_embedding' in n)
        transformer_params = sum(p.numel() for n, p in self.named_parameters()
                                 if 'transformer' in n or 'encoder_norm' in n)
        attention_params = sum(p.numel() for n, p in self.named_parameters() if 'patch_attention' in n)
        head_params = sum(p.numel() for n, p in self.named_parameters() if 'head' in n)

        print(f"\n{'='*60}")
        print(f"PatchTST Model Parameter Count (Binary Classification):")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - Patch + positional embedding: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
        print(f"  - Transformer encoder ({n_layers} layers): {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
        print(f"  - Attention pooling: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - Classification head: {head_params:,} ({head_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: Patching -> Channel-Independent Transformer -> Attn Pool -> Classifier")
        print(f"Patches: {self.n_patches} (patch_len={patch_len}, stride={stride})")
        print(f"d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, dropout={dropout}")
        print(f"Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        batch_size, n_features, seq_len = x.shape

        # STEP 1: Patch the sequence with a sliding window
        # (B, F, L) -> (B, F, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # STEP 2: Merge batch and feature dims for channel-independent processing
        # (B, F, n_patches, patch_len) -> (B*F, n_patches, patch_len)
        x = x.reshape(batch_size * n_features, self.n_patches, self.patch_len)

        # STEP 3: Project each patch into the model dimension
        x = self.patch_embedding(x)  # (B*F, n_patches, d_model)

        # STEP 4: Add learnable positional encoding (broadcast over B*F)
        x = x + self.pos_embedding
        x = self.embed_dropout(x)

        # STEP 5: Transformer encoder
        x = self.transformer(x)        # (B*F, n_patches, d_model)
        x = self.encoder_norm(x)

        # STEP 6: Attention-pool across patches (per channel)
        attn_logits = self.patch_attention(x)               # (B*F, n_patches, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = torch.sum(x * attn_weights, dim=1)              # (B*F, d_model)

        # STEP 7: Reshape so each sample's channels are concatenated, then classify
        x = x.reshape(batch_size, n_features * self.d_model)
        output = self.head(x)

        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focal loss focuses learning on hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for rare class (tensor, list, or float)
        gamma: Focusing parameter (gamma > 0 reduces the loss for well-classified examples)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, (float, int)):
            # Convert to tensor and register as buffer
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
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class

        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Index alpha tensor with targets
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
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select only the specified features
    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs04_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    # Sort by shot and time
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Calculate fs04 rate of change (change per millisecond)
    # Rate of change = (fs04[t] - fs04[t-1]) / (time[t] - time[t-1])
    if 'fs04' in df_sorted.columns:
        fs04_values = df_sorted['fs04'].values
        times_temp = df_sorted['time'].values
        shots_temp = df_sorted['shot'].values

        # Initialize rate of change array
        fs04_rate_of_change = np.zeros(len(df_sorted))

        # Calculate rate of change per shot (reset at shot boundaries)
        for shot_id in df_sorted['shot'].unique():
            shot_mask = shots_temp == shot_id
            shot_indices = np.where(shot_mask)[0]

            if len(shot_indices) > 1:
                # Calculate differences
                fs04_diff = np.diff(fs04_values[shot_indices])
                time_diff = np.diff(times_temp[shot_indices])

                # Avoid division by zero (set to 0 if time_diff is 0)
                time_diff_safe = np.where(time_diff == 0, 1, time_diff)
                rate = fs04_diff / time_diff_safe

                # Set first point in shot to 0 (no previous point), rest to calculated rate
                fs04_rate_of_change[shot_indices[0]] = 0.0
                fs04_rate_of_change[shot_indices[1:]] = rate

        # Add rate of change as a new column
        df_sorted['fs04_rate_of_change'] = fs04_rate_of_change
        # selected_features.append('fs04_rate_of_change')  # Removed from input features

    print(f"Using {len(selected_features)} features: {selected_features}")

    # Keep ALL data (including state=0 and state=-1) for temporal context
    # We'll filter invalid labels only when creating prediction targets

    # Extract features, labels, times, and shots
    X = df_sorted[selected_features].values
    y = df_sorted['state'].values
    times = df_sorted['time'].values  # Time in milliseconds
    shots = df_sorted['shot'].values

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(times)
    X = X[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, times, shots, selected_features, scaler


def create_windows_with_random_shot_split(X, y, times, shots, window_size=150, prediction_horizon_ms=50):
    """Create windows and perform random split BY SHOT - predicting state at future time

    This function splits data by shot number, ensuring all windows from the same
    shot end up in the same split (train, val, or test).

    The label is taken from the point that is prediction_horizon_ms in the future
    from the end of the window.

    Returns current states for transition analysis:
    - Binary mapping: Suppressed (1) -> 0, Dithering/ELMing/Mitigated (2,3,4) -> 1
    - Also returns current states (at end of window) for transition analysis
    """
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by SHOT NUMBER (not individual data points)")
    print("Binary classification: Suppressed (0) vs Dithering/ELMing/Mitigated (1)")

    # Get unique shots
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots: {n_shots}")

    # Randomly shuffle shots
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    # Split shots into train/val/test (70/15/15)
    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)

    train_shots = set(shuffled_shots[:train_size])
    val_shots = set(shuffled_shots[train_size:train_size + val_size])
    test_shots = set(shuffled_shots[train_size + val_size:])

    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")

    # Create windows for each split
    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []

    # Also track current states for transition analysis
    train_current_states = []
    val_current_states = []
    test_current_states = []

    # Binary label mapping: Suppressed (1) -> 0, Dithering/ELMing/Mitigated (2,3,4) -> 1
    # Also map current states for transition analysis
    binary_label_mapping = {1: 0, 2: 1, 3: 1, 4: 1}

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

        # OPTIMIZATION: Extract shot data ONCE before the inner loop
        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        # Create windows for this shot
        for i in range(len(shot_indices) - window_size + 1):
            window = shot_X[i:i + window_size]

            # Get the time at the end of the window
            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms

            # Get the current state (at the end of the window)
            current_label = shot_labels[i + window_size - 1]

            # OPTIMIZATION: Use binary search O(log n) instead of full array scan O(n)
            future_local_idx = np.searchsorted(shot_times, target_time)

            if future_local_idx >= len(shot_times):
                # No future data available for this window
                windows_skipped_no_future += 1
                continue

            # Get the label at the future time point
            future_label = shot_labels[future_local_idx]

            # Only create training example if both current and future labels are valid (1, 2, 3, 4)
            # Skip state=0 (unknown) and state=-1 (uncertain from label propagation)
            if int(current_label) not in binary_label_mapping or int(future_label) not in binary_label_mapping:
                windows_skipped_invalid_label += 1
                continue

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                target_windows.append(window)
                target_labels.append(binary_label_mapping[int(future_label)])
                target_current_states.append(binary_label_mapping[int(current_label)])
                windows_created += 1

    # Convert to numpy arrays
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

    # Oversample transition cases in training set, especially problematic transitions
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
    """Oversample transition cases, especially problematic 'Suppressed -> Dithering/ELMing/Mitigated' transitions

    Args:
        windows: Array of windows
        labels: Array of labels (future states)
        current_states: Array of current states
        transition_multiplier: How many times to duplicate general transition cases
        problematic_multiplier: How many times to duplicate problematic transition cases

    Returns:
        Oversampled windows, labels, and current_states
    """
    # Identify transition cases
    transition_mask = current_states != labels

    # Identify problematic transition: Suppressed (0) -> Dithering/ELMing/Mitigated (1)
    problematic_mask = (current_states == 0) & (labels == 1)

    # Get indices
    transition_indices = np.where(transition_mask)[0]
    problematic_indices = np.where(problematic_mask)[0]
    non_transition_indices = np.where(~transition_mask)[0]

    print(f"  Before oversampling:")
    print(f"    Total samples: {len(windows)}")
    print(f"    Transition cases: {len(transition_indices)}")
    print(f"    Problematic transitions (0->1): {len(problematic_indices)}")
    print(f"    Non-transition cases: {len(non_transition_indices)}")

    # Create oversampled arrays
    oversampled_windows = [windows[i] for i in non_transition_indices]
    oversampled_labels = [labels[i] for i in non_transition_indices]
    oversampled_current_states = [current_states[i] for i in non_transition_indices]

    # Add transition cases (excluding problematic ones, they'll be added separately)
    regular_transition_indices = transition_indices[~np.isin(transition_indices, problematic_indices)]
    for idx in regular_transition_indices:
        oversampled_windows.append(windows[idx])
        oversampled_labels.append(labels[idx])
        oversampled_current_states.append(current_states[idx])
        # Duplicate transition_multiplier times
        for _ in range(transition_multiplier - 1):
            oversampled_windows.append(windows[idx])
            oversampled_labels.append(labels[idx])
            oversampled_current_states.append(current_states[idx])

    # Add problematic transition cases with higher multiplier
    for idx in problematic_indices:
        oversampled_windows.append(windows[idx])
        oversampled_labels.append(labels[idx])
        oversampled_current_states.append(current_states[idx])
        # Duplicate problematic_multiplier times
        for _ in range(problematic_multiplier - 1):
            oversampled_windows.append(windows[idx])
            oversampled_labels.append(labels[idx])
            oversampled_current_states.append(current_states[idx])

    # Convert back to numpy arrays
    oversampled_windows = np.array(oversampled_windows, dtype=np.float32)
    oversampled_labels = np.array(oversampled_labels)
    oversampled_current_states = np.array(oversampled_current_states)

    print(f"  After oversampling:")
    print(f"    Total samples: {len(oversampled_windows)}")
    print(f"    Problematic transitions (0->1): {np.sum((oversampled_current_states == 0) & (oversampled_labels == 1))}")

    return oversampled_windows, oversampled_labels, oversampled_current_states


def train_model(model, train_loader, val_loader, device, class_weights_tensor, n_epochs=50):
    """Train the model with Focal Loss and class weights for imbalanced data"""
    # Use Focal Loss with class weights (alpha parameter)
    # Focal loss focuses on hard examples, class weights penalize minority class
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction='mean')

    # AdamW with weight decay - standard pairing for transformer models
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, verbose=True, min_lr=1e-6)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 25  # Increased patience to allow more training epochs

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
            # Gradient clipping for transformer training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            torch.save(model.state_dict(), f'best_patchtst_50ms_binary_transitions{VARIANT_SUFFIX}.pth')
            patience_counter = 0
            print(f"  New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def find_optimal_threshold(model, val_loader, device):
    """Find optimal decision threshold on validation set for better minority class prediction"""
    model.eval()
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            val_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            val_labels.extend(batch_y.numpy())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # Try different thresholds and find the one with best F1 score
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.linspace(0.1, 0.9, 81):  # Try thresholds from 0.1 to 0.9
        preds = (val_probs >= threshold).astype(int)
        if len(np.unique(preds)) > 1:  # Only if both classes are present
            f1 = f1_score(val_labels, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"\nOptimal threshold found: {best_threshold:.4f} (F1 score: {best_f1:.4f})")
    return best_threshold


def evaluate_model(model, test_loader, device, class_names, threshold=0.5):
    """Evaluate the model on test set using threshold-based prediction

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run on
        class_names: List of class names
        threshold: Decision threshold for positive class (default 0.5)
    """
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

            # Use threshold-based prediction instead of argmax for better minority class handling
            # For binary classification, use probability of positive class (class 1)
            pos_class_probs = probs[:, 1].cpu().numpy()
            preds = (pos_class_probs >= threshold).astype(int)

            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Calculate ROC AUC
    if len(np.unique(all_labels)) > 1:
        # For binary classification, use the positive class probabilities
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"\nROC AUC Score: {auc:.4f}")

    return all_preds, all_labels, all_probs


def analyze_transition_effectiveness(all_preds, all_labels, all_current_states, all_probs, class_names):
    """Analyze model effectiveness on state transitions (where current_state != future_state)

    This function evaluates how well the model predicts future states when there is
    a state transition (i.e., when the state 50ms in the future is different from
    the current state at the end of the window).
    """
    print("\n" + "="*60)
    print("TRANSITION EFFECTIVENESS ANALYSIS")
    print("="*60)
    print(f"Analyzing predictions for points where future state ({PREDICTION_HORIZON_MS}ms) differs from current state")
    print("="*60)

    # Identify transition cases (where current_state != future_state)
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

    # Extract predictions and labels for transition cases only
    transition_preds = all_preds[transition_mask]
    transition_labels = all_labels[transition_mask]
    transition_probs = all_probs[transition_mask] if len(all_probs.shape) > 1 else None

    # Calculate metrics for transition cases
    transition_acc = accuracy_score(transition_labels, transition_preds)
    transition_precision = precision_score(transition_labels, transition_preds, average='weighted', zero_division=0)
    transition_recall = recall_score(transition_labels, transition_preds, average='weighted', zero_division=0)
    transition_f1 = f1_score(transition_labels, transition_preds, average='weighted', zero_division=0)

    print(f"\nTransition Case Metrics:")
    print(f"  Accuracy: {transition_acc:.4f}")
    print(f"  Precision (weighted): {transition_precision:.4f}")
    print(f"  Recall (weighted): {transition_recall:.4f}")
    print(f"  F1-Score (weighted): {transition_f1:.4f}")

    # Calculate ROC AUC for transitions if binary
    if transition_probs is not None and len(np.unique(transition_labels)) > 1:
        transition_auc = roc_auc_score(transition_labels, transition_probs[:, 1])
        print(f"  ROC AUC: {transition_auc:.4f}")

    # Detailed classification report for transitions
    print(f"\nTransition Case Classification Report:")
    print(classification_report(transition_labels, transition_preds, target_names=class_names, digits=4))

    # Confusion matrix for transitions
    transition_cm = confusion_matrix(transition_labels, transition_preds)
    print(f"\nTransition Case Confusion Matrix:")
    print(f"  Predicted ->")
    print(f"  Actual v")
    print(f"  {transition_cm}")

    # Compare with overall performance
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\nComparison with Overall Performance:")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Transition accuracy: {transition_acc:.4f}")
    print(f"  Difference: {transition_acc - overall_acc:.4f} ({((transition_acc - overall_acc)/overall_acc*100):.2f}%)")

    # Breakdown by transition type
    print(f"\nTransition Type Breakdown:")
    transition_types = {
        'Suppressed -> Dithering/ELMing/Mitigated': (all_current_states == 0) & (all_labels == 1),
        'Dithering/ELMing/Mitigated -> Suppressed': (all_current_states == 1) & (all_labels == 0)
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

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Validation Loss ({PREDICTION_HORIZON_MS}ms Prediction - PatchTST Binary)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training accuracy
    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Training and Validation Accuracy ({PREDICTION_HORIZON_MS}ms Prediction - PatchTST Binary)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot confusion matrix (normalized)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix (PatchTST Binary)')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    # Plot confusion matrix (counts)
    cm_counts = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Counts - PatchTST Binary)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(f'patchtst_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Results saved to 'patchtst_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png'")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("PatchTST Model for Binary Plasma Classification")
    print("=" * 60)
    print("Architecture: Patching -> Channel-Independent Transformer -> Attn Pool -> Classifier")
    print("Window: 150 datapoints BEFORE current time")
    print(f"Prediction: {PREDICTION_HORIZON_MS}ms INTO THE FUTURE")
    print("Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data (now includes times)
    X, y, times, shots, features, scaler = load_and_prepare_data()

    # Create windows and split BY SHOT (also returns current states for transition analysis)
    train_X, train_y, train_current_states, val_X, val_y, val_current_states, test_X, test_y, test_current_states = create_windows_with_random_shot_split(
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

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # Calculate class weights from training data to penalize minority class misclassification
    print("\nCalculating class weights from training data...")
    class_counts = np.bincount(train_y, minlength=2)
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    # Normalize weights to sum to number of classes
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights: {dict(zip(range(len(class_weights)), class_weights))}")

    # Convert to tensor and move to device
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Create PatchTST model (binary classification: 2 classes)
    model = PatchTST(
        n_features=len(features),
        n_classes=2,
        seq_len=150,
        patch_len=15,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
        head_dropout=0.3,
    ).to(device)

    # Test forward pass speed
    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)

    import time
    start_time = time.time()
    with torch.no_grad():
        _ = model(test_batch)
    forward_time = time.time() - start_time
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {forward_time:.3f} seconds")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, class_weights_tensor, n_epochs=50
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(f'best_patchtst_50ms_binary_transitions{VARIANT_SUFFIX}.pth'))

    # Find optimal threshold on validation set
    optimal_threshold = find_optimal_threshold(model, val_loader, device)

    # Evaluate on test set with optimal threshold
    class_names = ['Suppressed', 'Dithering/ELMing/Mitigated']
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names, threshold=optimal_threshold)

    # Analyze transition effectiveness
    analyze_transition_effectiveness(all_preds, all_labels, test_current_states, all_probs, class_names)

    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    # Final test accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training Complete! (Predicting {PREDICTION_HORIZON_MS}ms into the future - PatchTST Binary)")
    print("=" * 60)


if __name__ == "__main__":
    main()
