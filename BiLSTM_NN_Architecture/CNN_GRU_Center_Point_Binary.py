import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class CNNGRU(nn.Module):
    """
    CNN -> GRU hybrid for binary plasma-state classification.

    Pipeline:
      1) 1D CNN stack extracts local temporal features over the window
         (multi-scale receptive fields via stacked conv layers).
      2) Bidirectional GRU integrates long-range temporal dependencies on
         top of the CNN feature maps (cheaper than LSTM, often as accurate).
      3) Additive attention pools the GRU output across time.
      4) MLP head combines attention pool + last-step hidden -> logits.
    """
    def __init__(
        self,
        n_features,
        n_classes=2,
        cnn_channels=(64, 128, 128),
        cnn_kernel_sizes=(7, 5, 3),
        gru_hidden=128,
        gru_layers=2,
        fc_hidden=(256, 128),
        dropout=0.25,
    ):
        super(CNNGRU, self).__init__()

        # ----- CNN feature extractor (operates on (batch, n_features, seq)) -----
        cnn_blocks = []
        in_ch = n_features
        for out_ch, k in zip(cnn_channels, cnn_kernel_sizes):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        cnn_out_ch = cnn_channels[-1]

        # ----- Bidirectional GRU on top of the CNN feature stream -----
        self.gru = nn.GRU(
            input_size=cnn_out_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * 2  # bidirectional

        # ----- Additive (Bahdanau-style) attention pooling over time -----
        self.attn_proj = nn.Linear(gru_out_dim, gru_out_dim)
        self.attn_score = nn.Linear(gru_out_dim, 1)

        # ----- MLP classifier head -----
        head_in = gru_out_dim * 2  # [attn_pool, last_step]
        head_layers = []
        prev = head_in
        for h in fc_hidden:
            head_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout + 0.05),
            ])
            prev = h
        head_layers.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*head_layers)

        # ----- Parameter accounting (matches the BiLSTM-NN script style) -----
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('cnn'))
        gru_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('gru'))
        attn_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('attn'))
        cls_p = sum(p.numel() for n, p in self.named_parameters() if n.startswith('classifier'))

        print(f"\n{'='*60}")
        print(f"CNN-GRU Model Parameter Count (Binary):")
        print(f"{'='*60}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"\nParameters by component:")
        print(f"  - CNN stack:    {cnn_p:,} ({cnn_p/total*100:.1f}%)")
        print(f"  - GRU layers:   {gru_p:,} ({gru_p/total*100:.1f}%)")
        print(f"  - Attention:    {attn_p:,} ({attn_p/total*100:.1f}%)")
        print(f"  - Classifier:   {cls_p:,} ({cls_p/total*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: 1D-CNN (local features) -> BiGRU (long-range) -> Attention -> MLP")

    def forward(self, x):
        # x: (batch, n_features, seq_len) -- already in conv-friendly layout
        feats = self.cnn(x)              # (batch, cnn_out_ch, seq_len)
        feats = feats.transpose(1, 2)    # (batch, seq_len, cnn_out_ch) for GRU

        gru_out, _ = self.gru(feats)     # (batch, seq_len, gru_hidden*2)

        # Additive attention over time
        # u_t = tanh(W h_t); a_t = softmax(v u_t)
        u = torch.tanh(self.attn_proj(gru_out))            # (batch, seq, dim)
        scores = self.attn_score(u)                        # (batch, seq, 1)
        attn = torch.softmax(scores, dim=1)                # (batch, seq, 1)
        pooled = torch.sum(gru_out * attn, dim=1)          # (batch, dim)

        last_step = gru_out[:, -1, :]                      # (batch, dim)
        combined = torch.cat([pooled, last_step], dim=1)   # (batch, 2*dim)

        return self.classifier(combined)


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
    """Load and preprocess the plasma data for binary classification.

    Multi-class state convention (matches BiLSTM_NN_Center_Point_Shot_Split.py):
        0 = Suppressed
        1 = Dithering
        2 = Mitigated
        3 = ELMing
       -1 = Unclassified edge/short (filled here by per-shot label padding).

    -1 occurs because the upstream label-propagation pipeline left some shot
    edges unlabelled. We treat -1 as missing and fill it with the nearest
    valid state within the same shot (forward-fill then back-fill on time-
    sorted rows). After this, every retained row has state in {0, 1, 2, 3}.

    Binary mapping applied below:
        state == 0          -> 0 (Suppressed)
        state in {1, 2, 3}  -> 1 (ELMy = Dithering + Mitigated + ELMing)
    """
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot (matches the multi-class script).
    df = df[df['shot'] != 191675].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs_sum_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]
    print(f"Using {len(selected_features)} features: {selected_features}")

    # Sort by (shot, time) so per-shot ffill/bfill respects time order.
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    pre_dist = Counter(df_sorted['state'].values.tolist())
    n_unknown_pre = int((df_sorted['state'] == -1).sum())
    print(f"Raw state distribution (incl. -1 edges): {pre_dist}")
    print(f"  Unknown (-1) frames before padding: {n_unknown_pre:,}")

    # Label "padding": replace -1 with the nearest valid state within each
    # shot using forward-fill then back-fill. This mirrors the edge-replicate
    # padding already used for the *features* and eliminates the -1 frames
    # that exist because labels weren't propagated all the way to shot edges.
    state_as_float = df_sorted['state'].replace(-1, np.nan)
    df_sorted = df_sorted.assign(state=state_as_float)
    df_sorted['state'] = (
        df_sorted.groupby('shot', group_keys=False)['state']
                 .transform(lambda s: s.ffill().bfill())
    )

    n_remaining_unknown = int(df_sorted['state'].isna().sum())
    n_filled = n_unknown_pre - n_remaining_unknown
    print(f"  -1 frames filled by per-shot label padding: {n_filled:,}")
    if n_remaining_unknown:
        print(
            f"  -1 frames remaining (entire-shot unlabelled, dropped): "
            f"{n_remaining_unknown:,}"
        )

    df_filtered = df_sorted.dropna(subset=['state']).copy()
    df_filtered['state'] = df_filtered['state'].astype(int)

    X = df_filtered[selected_features].values
    y = df_filtered['state'].values.astype(int)
    shots = df_filtered['shot'].values

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Multi-class label distribution after padding: {Counter(y.tolist())}")

    # Binary remap: 0 -> 0 (Suppressed); 1, 2, 3 -> 1 (ELMy).
    y_binary = np.where(y == 0, 0, 1).astype(np.int64)
    print(f"Binary label distribution (0=Suppressed, 1=ELMy): {Counter(y_binary.tolist())}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_binary, shots, selected_features, scaler


def edge_pad_shot_features(X_sub, window_size):
    """
    Replicate first/last rows so every original timestep can be the center of a length-window_size window.
    Returns X_pad (L + window_size - 1, n_features) with original rows in the middle segment.
    """
    X_sub = np.asarray(X_sub, dtype=np.float64)
    L, n_feat = X_sub.shape
    center_idx = window_size // 2
    pad_left = center_idx
    pad_right = window_size - center_idx - 1
    if L == 0:
        return X_sub, pad_left
    left = np.repeat(X_sub[:1], pad_left, axis=0)
    right = np.repeat(X_sub[-1:], pad_right, axis=0)
    X_pad = np.vstack([left, X_sub, right])
    return X_pad.astype(np.float32), pad_left


def create_windows_with_shot_split(X, y, shots, window_size=150, train_frac=0.7, val_frac=0.15):
    """Create windows (edge-padded per shot) and split by shot: all windows from a shot share the same fold."""
    print(f"Creating windows of size {window_size} (edge replicate padding)...")
    print(f"Train/val/test split by shot: {train_frac:.0%} / {val_frac:.0%} / {1 - train_frac - val_frac:.0%}")

    windows, labels, window_shots = [], [], []
    center_idx = window_size // 2

    # Create windows per shot
    for shot_id in np.unique(shots):
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) == 0:
            continue

        X_sub = X[shot_indices]
        y_sub = y[shot_indices]
        L = len(shot_indices)

        if L < window_size:
            # Pad short shots to length window_size (replicate edges), then edge_pad below
            pad_extra = window_size - L
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

        X_pad, pad_left = edge_pad_shot_features(X_sub, window_size)

        for t in range(L):
            start = pad_left + t - center_idx
            window = X_pad[start : start + window_size]
            center_label = y_sub[t]

            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(center_label)
                window_shots.append(shot_id)

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels)
    window_shots = np.array(window_shots)

    print(f"Created {len(windows)} valid windows")
    print(f"Label distribution: {Counter(labels)}")

    rng = np.random.RandomState(42)
    unique_shots = rng.permutation(np.unique(window_shots))
    n_shots = len(unique_shots)

    train_end = int(train_frac * n_shots)
    val_end = int((train_frac + val_frac) * n_shots)

    train_list = unique_shots[:train_end].tolist()
    val_list = unique_shots[train_end:val_end].tolist()
    test_list = unique_shots[val_end:].tolist()

    # Integer boundaries can leave val empty (e.g. n_shots=3); rebalance when possible.
    if n_shots >= 3:
        if len(val_list) == 0 and len(train_list) > 1:
            val_list.append(train_list.pop())
        if len(test_list) == 0 and len(val_list) > 1:
            test_list.append(val_list.pop())
    elif n_shots == 2:
        train_list = [unique_shots[0]]
        val_list = []
        test_list = [unique_shots[1]]
    else:
        train_list = unique_shots.tolist()
        val_list = []
        test_list = []

    train_shot_set = set(train_list)
    val_shot_set = set(val_list)
    test_shot_set = set(test_list)

    train_mask = np.isin(window_shots, list(train_shot_set))
    val_mask = np.isin(window_shots, list(val_shot_set))
    test_mask = np.isin(window_shots, list(test_shot_set))

    print(f"Shots per split — train: {len(train_shot_set)}, val: {len(val_shot_set)}, test: {len(test_shot_set)} (total unique shots: {n_shots})")

    return (windows[train_mask], labels[train_mask],
            windows[val_mask], labels[val_mask],
            windows[test_mask], labels[test_mask])


def train_model(model, train_loader, val_loader, device, n_epochs=100,
                class_weights=None, lr=1e-3, weight_decay=1e-4, grad_clip=1.0):
    """Train the CNN-GRU model.

    Adds class weighting + gradient clipping + AdamW + cosine LR schedule on top
    of the original BiLSTM-NN training loop, while keeping the same early-
    stopping / best-checkpoint behavior so results are directly comparable.
    """
    if class_weights is not None:
        class_weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        print(f"Using class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Plateau scheduler keeps parity with the BiLSTM-NN baseline's behavior.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 20

    print("\nStarting training...")
    for epoch in range(n_epochs):
        # ----- Training phase -----
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # ----- Validation phase -----
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
        val_acc = accuracy_score(val_labels_list, val_preds) if len(val_labels_list) else float("nan")

        avg_train_loss = train_loss / len(train_loader)
        n_val_batches = len(val_loader)
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float("nan")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if n_val_batches > 0:
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print("  Val: skipped (no validation shots)")

        scheduler.step(val_acc if n_val_batches > 0 else train_acc)

        monitor_acc = val_acc if n_val_batches > 0 else train_acc
        if monitor_acc > best_val_acc:
            best_val_acc = monitor_acc
            torch.save(model.state_dict(), 'best_cnn_gru_binary.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set (binary)"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

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

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("\nROC AUC Score:")
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"  Binary (ELMy vs Suppressed): {auc:.4f}")
    else:
        print("  Only one class present in test labels; skipping AUC.")

    return all_preds, all_labels, all_probs


def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, all_probs, class_names):
    """Plot training curves, confusion matrix, and ROC curve"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('CNN-GRU: Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('CNN-GRU: Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('CNN-GRU: Normalized Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    if len(np.unique(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
        axes[1, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('CNN-GRU: ROC Curve (ELMy vs Suppressed)')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC unavailable\n(only one class in test)',
                        ha='center', va='center')
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig('cnn_gru_binary_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Results saved to 'cnn_gru_binary_results.png'")


import time


def main():
    """Main training pipeline for CNN-GRU binary classification"""
    print("=" * 50)
    print("CNN-GRU Model for Plasma Binary Classification")
    print("=" * 50)
    print("Classes: Suppressed (0) vs ELMy (1)")
    print("Architecture: 1D-CNN (local features) -> BiGRU (long-range) -> Attention -> MLP")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, shots, features, scaler = load_and_prepare_data()

    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_shot_split(X, y, shots)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Compute class weights from the training fold to counteract class imbalance.
    if len(train_y) > 0 and len(np.unique(train_y)) > 1:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1], dtype=np.int64),
            y=train_y,
        )
    else:
        class_weights = None

    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = CNNGRU(n_features=len(features), n_classes=2).to(device)

    # Forward-pass timing sanity check (mirrors the BiLSTM-NN script).
    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)

    start_time = time.time()
    with torch.no_grad():
        _ = model(test_batch)
    forward_time = time.time() - start_time
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {forward_time:.3f} seconds")

    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device,
        n_epochs=50,
        class_weights=class_weights,
    )

    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_cnn_gru_binary.pth'))

    class_names = ['Suppressed', 'ELMy']
    if len(test_X) == 0:
        print("\nNo test windows after shot-level split; skipping test evaluation and plots.")
        all_preds = all_labels = np.array([])
        all_probs = np.empty((0, 2))
        test_acc = float("nan")
    else:
        all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names)
        plot_results(train_losses, val_losses, train_accs, val_accs,
                     all_preds, all_labels, all_probs, class_names)
        test_acc = accuracy_score(all_labels, all_preds)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    save_path = '/mnt/homes/sr4240/my_folder/BiLSTM_NN_Architecture/cnn_gru_binary_complete_model.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features,
        'n_features': len(features),
        'n_classes': 2,
        'cnn_channels': [64, 128, 128],
        'cnn_kernel_sizes': [7, 5, 3],
        'gru_hidden': 128,
        'gru_layers': 2,
        'fc_hidden': [256, 128],
        'window_size': 150,
        'class_names': class_names,
        'label_mapping': {
            'Suppressed': 0,
            'ELMy': 1,
            'source_states': {'Suppressed': [0], 'ELMy': [1, 2, 3]},
            'unknown_handling': 'per-shot ffill+bfill of state==-1 (label padding)',
        },
        'class_weights': class_weights.tolist() if class_weights is not None else None,
        'test_accuracy': float(test_acc) if test_acc == test_acc else None,
    }
    torch.save(checkpoint, save_path)
    print(f"\n✓ Complete model checkpoint saved to: {save_path}")
    print("  Includes: model weights, scaler, features, label mappings, hyperparams")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
