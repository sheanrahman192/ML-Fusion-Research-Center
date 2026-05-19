"""
Binary iTransformer for plasma classification (Suppressed vs ELMy).

This script is a controlled architecture comparison against
BiLSTM_NN_Center_Point_Binary.py. The DATA PIPELINE is copied verbatim
from that file (load_and_prepare_data, edge_pad_shot_features,
create_windows_with_shot_split, PlasmaDataset, evaluate_model,
plot_results) so the only thing that differs is the model + the tuned
training recipe carried over from the 4-class iTransformer tuning
(Round 4 winner: class weights, label smoothing, AdamW wd=3e-2,
grad-clip, warmup+cosine, macro-F1 model selection, shrunk/regularized
flatten head).
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility (match the binary BiLSTM script)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class iTransformer(nn.Module):
    """
    iTransformer (Inverted Transformer) — Round 4 tuned config.

    Each variate's whole time series is embedded into one token; self-
    attention runs ACROSS the variate tokens. Round 4 head: flatten the
    variate tokens (preserves per-variate structure) but keep it small
    and heavily regularized (768->128, dropout 0.4), since overfitting
    concentrated in the head in the 4-class tuning.
    """
    def __init__(self, n_features, seq_len=150, n_classes=2,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2):
        super(iTransformer, self).__init__()

        self.n_features = n_features
        self.seq_len = seq_len

        # INVERTED EMBEDDING: each variate's whole series -> one token
        self.variate_embedding = nn.Linear(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Attention OVER variate tokens; no positional encoding (variates
        # have no inherent order) — defining iTransformer choice.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # ROUND 4 head: shrunk + heavily regularized flatten head.
        self.classifier = nn.Sequential(
            nn.Linear(n_features * d_model, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed_params = sum(p.numel() for name, p in self.named_parameters() if 'variate_embedding' in name)
        encoder_params = sum(p.numel() for name, p in self.named_parameters() if 'encoder' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)

        print(f"\n{'='*60}")
        print(f"iTransformer Model Parameter Count (Binary):")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nParameters by component:")
        print(f"  - Inverted embedding: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
        print(f"  - Transformer encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
        print(f"  - Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print(f"Architecture: Inverted embedding (variate->token) -> "
              f"attention across {n_features} variates -> classifier")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length) — already the
        # inverted layout, so embed along the time dimension directly.
        batch_size, n_features, seq_len = x.shape
        tokens = self.variate_embedding(x)
        tokens = self.embed_dropout(tokens)
        encoded = self.encoder(tokens)
        encoded = self.encoder_norm(encoded)
        flat = encoded.reshape(batch_size, -1)
        output = self.classifier(flat)
        return output


# ===========================================================================
# DATA PIPELINE — copied verbatim from BiLSTM_NN_Center_Point_Binary.py so the
# comparison isolates the model. Do not modify these without mirroring there.
# ===========================================================================

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

    Binary mapping:
        state == 0          -> 0 (Suppressed)
        state in {1, 2, 3}  -> 1 (ELMy = Dithering + Mitigated + ELMing)
    -1 edges are filled by per-shot ffill+bfill (label padding).
    """
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    df = df[df['shot'] != 191675].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li',
                         'tritop', 'fs_sum_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]
    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    pre_dist = Counter(df_sorted['state'].values.tolist())
    n_unknown_pre = int((df_sorted['state'] == -1).sum())
    print(f"Raw state distribution (incl. -1 edges): {pre_dist}")
    print(f"  Unknown (-1) frames before padding: {n_unknown_pre:,}")

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

    y_binary = np.where(y == 0, 0, 1).astype(np.int64)
    print(f"Binary label distribution (0=Suppressed, 1=ELMy): {Counter(y_binary.tolist())}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_binary, shots, selected_features, scaler


def edge_pad_shot_features(X_sub, window_size):
    """Replicate first/last rows so every timestep can be a window center."""
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
    """Create windows (edge-padded per shot) and split by shot."""
    print(f"Creating windows of size {window_size} (edge replicate padding)...")
    print(f"Train/val/test split by shot: {train_frac:.0%} / {val_frac:.0%} / {1 - train_frac - val_frac:.0%}")

    windows, labels, window_shots = [], [], []
    center_idx = window_size // 2

    for shot_id in np.unique(shots):
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) == 0:
            continue

        X_sub = X[shot_indices]
        y_sub = y[shot_indices]
        L = len(shot_indices)

        if L < window_size:
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


# ===========================================================================
# TRAINING — tuned recipe carried over from the 4-class iTransformer Round 4.
# ===========================================================================

def train_model(model, train_loader, val_loader, device, n_epochs=50, class_weights=None):
    """Train the model (tuned iTransformer recipe)."""
    if class_weights is not None:
        class_weights = class_weights.to(device)
    # Weighted loss + label smoothing (R1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # AdamW; R4 weight decay 3e-2 (R1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-2)
    # Warmup + cosine (R3)
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_metric = 0.0
    patience_counter = 0
    max_patience = 15

    print("\nStarting training...")
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
        n_val_batches = len(val_loader)
        if n_val_batches > 0:
            val_acc = accuracy_score(val_labels_list, val_preds)
            val_macro_f1 = f1_score(val_labels_list, val_preds, average='macro')
        else:
            val_acc = float('nan')
            val_macro_f1 = float('nan')

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float("nan")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if n_val_batches > 0:
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}")
        else:
            print("  Val: skipped (no validation shots)")

        scheduler.step()

        # Select on macro-F1 (train acc fallback if no val fold)
        monitor = val_macro_f1 if n_val_batches > 0 else train_acc
        if monitor > best_metric:
            best_metric = monitor
            torch.save(model.state_dict(), 'best_itransformer_binary.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved! (macro-F1={monitor:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set (binary) — copied from binary BiLSTM."""
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
    """Plot training curves, confusion matrix, ROC — copied from binary BiLSTM."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix')
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
        axes[1, 1].set_title('ROC Curve (ELMy vs Suppressed)')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC unavailable\n(only one class in test)',
                        ha='center', va='center')
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig('itransformer_binary_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Results saved to 'itransformer_binary_results.png'")


import time

def main():
    print("=" * 50)
    print("iTransformer Model for Plasma Binary Classification")
    print("=" * 50)
    print("Classes: Suppressed (0) vs ELMy (1)")
    print("Pipeline: identical to BiLSTM_NN_Center_Point_Binary.py")
    print("Recipe: tuned iTransformer Round 4")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, shots, features, scaler = load_and_prepare_data()

    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_shot_split(X, y, shots)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Balanced class weights (binary is imbalanced too)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
    class_weights = torch.FloatTensor(cw)
    print(f"\nClass weights (balanced): {dict(enumerate(np.round(cw, 4)))}")

    window_size = 150
    d_model = 128
    n_heads = 8
    n_layers = 3
    d_ff = 256
    dropout = 0.2

    model = iTransformer(
        n_features=len(features), seq_len=window_size, n_classes=2,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, dropout=dropout
    ).to(device)

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
        model, train_loader, val_loader, device, n_epochs=50, class_weights=class_weights
    )

    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_itransformer_binary.pth'))

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

    save_path = '/mnt/homes/sr4240/my_folder/BiLSTM_NN_Architecture/iTransformer/itransformer_binary_complete_model.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features,
        'n_features': len(features),
        'n_classes': 2,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'window_size': window_size,
        'class_names': class_names,
        'label_mapping': {
            'Suppressed': 0,
            'ELMy': 1,
            'source_states': {'Suppressed': [0], 'ELMy': [1, 2, 3]},
            'unknown_handling': 'per-shot ffill+bfill of state==-1 (label padding)',
        },
        'test_accuracy': float(test_acc) if test_acc == test_acc else None
    }
    torch.save(checkpoint, save_path)
    print(f"\n✓ Complete model checkpoint saved to: {save_path}")

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
