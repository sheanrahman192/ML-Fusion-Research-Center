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

# Set random seeds for reproducibility
np.random.seed(43)
torch.manual_seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed(43)

class iTransformer(nn.Module):
    """
    iTransformer (Inverted Transformer) for plasma window classification.

    Unlike a vanilla Transformer that tokenizes each time step, the
    iTransformer INVERTS the embedding: each variate's entire time series
    is embedded into a single token. Self-attention is then applied ACROSS
    variates (capturing multivariate correlations) while a feed-forward
    network learns temporal representations within each variate token.

    Reference: Liu et al., "iTransformer: Inverted Transformers Are
    Effective for Time Series Forecasting", ICLR 2024.
    """
    def __init__(self, n_features, seq_len=150, n_classes=4,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2):
        super(iTransformer, self).__init__()

        self.n_features = n_features
        self.seq_len = seq_len

        # INVERTED EMBEDDING: map each variate's whole time series -> one token
        # Input variate series of length seq_len -> token of dim d_model
        self.variate_embedding = nn.Linear(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder: attention is computed OVER variate tokens.
        # No positional encoding is used because variates have no inherent
        # ordering (this is a defining choice of the iTransformer).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Classification head: flatten all variate tokens -> class logits
        self.classifier = nn.Sequential(
            nn.Linear(n_features * d_model, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

        # Print detailed model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        embed_params = sum(p.numel() for name, p in self.named_parameters() if 'variate_embedding' in name)
        encoder_params = sum(p.numel() for name, p in self.named_parameters() if 'encoder' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)

        print(f"\n{'='*60}")
        print(f"iTransformer Model Parameter Count:")
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
        # x shape: (batch_size, n_features, sequence_length)
        # This is ALREADY in the inverted layout: each variate is a row,
        # so we embed along the time dimension directly.
        batch_size, n_features, seq_len = x.shape

        # INVERTED EMBEDDING: each variate's series -> a token of dim d_model
        # (batch_size, n_features, seq_len) -> (batch_size, n_features, d_model)
        tokens = self.variate_embedding(x)
        tokens = self.embed_dropout(tokens)

        # Self-attention ACROSS the n_features variate tokens
        encoded = self.encoder(tokens)            # (batch, n_features, d_model)
        encoded = self.encoder_norm(encoded)

        # Flatten all variate tokens and classify
        flat = encoded.reshape(batch_size, -1)    # (batch, n_features * d_model)
        output = self.classifier(flat)

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
    """Load and preprocess the plasma data"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    # Select only the specified 7 features
    important_features = ['iln3iamp', 'betan', 'density', 'li', 'tritop', 'fs_sum_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    print(f"Using {len(selected_features)} features: {selected_features}")

    # Sort by shot and time
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Valid labels are already 0-indexed:
    # 0=Suppressed, 1=Dithering, 2=Mitigated, 3=ELMing. Exclude unknown values like -1.
    valid_states = [0, 1, 2, 3]
    df_filtered = df_sorted[df_sorted['state'].isin(valid_states)].copy()

    # Extract features and labels
    X = df_filtered[selected_features].values
    y = df_filtered['state'].values
    shots = df_filtered['shot'].values

    # Remove NaN values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    shots = shots[valid_mask]

    y = y.astype(int)

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {Counter(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, shots, selected_features, scaler

def create_windows_for_shots(X, y, shots, shot_list, window_size=150):
    """Create windows for a specific list of shots"""
    windows, labels = [], []
    center_idx = window_size // 2

    for shot_id in shot_list:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        for i in range(len(shot_indices) - window_size + 1):
            start = shot_indices[i]
            end = start + window_size

            if end > shot_indices[-1] + 1:
                break

            window = X[start:end]
            center_label = y[start + center_idx]

            # Check window validity
            if not np.isnan(window).any() and not np.isinf(window).any():
                windows.append(window)
                labels.append(center_label)

    if len(windows) == 0:
        return np.array([]), np.array([])

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels)

    return windows, labels

def create_windows_with_shot_split(X, y, shots, window_size=150, train_ratio=0.7, val_ratio=0.15):
    """Create windows and perform shot-based split"""
    print(f"Creating windows of size {window_size} with SHOT-BASED split...")

    # Get unique shots
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total number of unique shots: {n_shots}")

    # Shuffle shots for random assignment
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    # Calculate split indices
    train_end = int(train_ratio * n_shots)
    val_end = int((train_ratio + val_ratio) * n_shots)

    # Split shots into train/val/test
    train_shots = shuffled_shots[:train_end]
    val_shots = shuffled_shots[train_end:val_end]
    test_shots = shuffled_shots[val_end:]

    print(f"\nShot split:")
    print(f"  Train shots: {len(train_shots)}")
    print(f"  Val shots: {len(val_shots)}")
    print(f"  Test shots: {len(test_shots)}")

    # Create windows for each split
    print("\nCreating windows for each split...")
    train_windows, train_labels = create_windows_for_shots(X, y, shots, train_shots, window_size)
    val_windows, val_labels = create_windows_for_shots(X, y, shots, val_shots, window_size)
    test_windows, test_labels = create_windows_for_shots(X, y, shots, test_shots, window_size)

    print(f"\nWindows created:")
    print(f"  Train: {len(train_windows)} windows")
    print(f"  Val: {len(val_windows)} windows")
    print(f"  Test: {len(test_windows)} windows")

    print(f"\nLabel distributions:")
    print(f"  Train: {Counter(train_labels)}")
    print(f"  Val: {Counter(val_labels)}")
    print(f"  Test: {Counter(test_labels)}")

    return (train_windows, train_labels,
            val_windows, val_labels,
            test_windows, test_labels)

def train_model(model, train_loader, val_loader, device, n_epochs=50, class_weights=None):
    """Train the model"""
    # Loss and optimizer
    # ROUND 1: weighted loss (fights class imbalance) + label smoothing
    # (fights overconfident wrong predictions / exploding val loss)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # ROUND 1: AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    # ROUND 1: schedule / select on macro-F1 (correct objective under imbalance)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    print("\nStarting training...")
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
            # ROUND 1: gradient clipping for transformer training stability
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
        # ROUND 1: macro-F1 is the true objective under heavy class imbalance
        val_macro_f1 = f1_score(val_labels_list, val_preds, average='macro')

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_macro_f1:.4f}")

        # Learning rate scheduling on macro-F1
        scheduler.step(val_macro_f1)

        # Early stopping on macro-F1 (best_val_acc now tracks best macro-F1)
        if val_macro_f1 > best_val_acc:
            best_val_acc = val_macro_f1
            torch.save(model.state_dict(), 'best_itransformer_shot_split.pth')
            patience_counter = 0
            print(f"  ✓ New best model saved! (macro-F1={val_macro_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set"""
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

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Calculate ROC AUC for each class
    print("\nROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        if i < all_probs.shape[1]:
            class_labels = (all_labels == i).astype(int)
            if len(np.unique(class_labels)) > 1:
                auc = roc_auc_score(class_labels, all_probs[:, i])
                print(f"  {class_name}: {auc:.4f}")

    return all_preds, all_labels, all_probs

def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names):
    """Plot training curves and confusion matrix"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot training loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training accuracy
    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot confusion matrix (normalized)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    # Plot confusion matrix (counts)
    cm_counts = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Counts)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('itransformer_shot_split_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Results saved to 'itransformer_shot_split_results.png'")

import time

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("iTransformer Model for Plasma Classification")
    print("=" * 60)
    print("Architecture: Inverted embedding (each variate -> token) →")
    print("              self-attention across variates → classifier")
    print("Split Method: SHOT-BASED (no data leakage between train/val/test)")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X, y, shots, features, scaler = load_and_prepare_data()

    # Create windows and split BY SHOT
    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_shot_split(X, y, shots)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    # Create data loaders
    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # ROUND 1: balanced class weights to counter heavy class imbalance
    cw = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=train_y)
    class_weights = torch.FloatTensor(cw)
    print(f"\nClass weights (balanced): {dict(enumerate(np.round(cw, 4)))}")

    # iTransformer hyperparameters
    window_size = 150
    d_model = 128
    n_heads = 8
    n_layers = 3
    d_ff = 256
    dropout = 0.2

    # Create model
    model = iTransformer(
        n_features=len(features),
        seq_len=window_size,
        n_classes=4,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

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
        model, train_loader, val_loader, device, n_epochs=50, class_weights=class_weights
    )

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_itransformer_shot_split.pth'))

    # Evaluate on test set
    class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names)

    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    # Final test accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Save complete model checkpoint for later use
    save_path = '/mnt/homes/sr4240/my_folder/BiLSTM_NN_Architecture/iTransformer/itransformer_complete_model.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features,
        'n_features': len(features),
        'n_classes': 4,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'window_size': window_size,
        'class_names': class_names,
        'test_accuracy': test_acc
    }
    torch.save(checkpoint, save_path)
    print(f"\n✓ Complete model checkpoint saved to: {save_path}")
    print("  Includes: model weights, scaler, features, label mappings")

    # Also save as best_itransformer.pth
    torch.save(checkpoint, 'best_itransformer.pth')
    print(f"✓ Also saved to: best_itransformer.pth")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("Note: Shot-based split ensures no data leakage between splits")
    print("=" * 60)

if __name__ == "__main__":
    main()
