import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(43)
torch.manual_seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed(43)

# Prediction horizon in milliseconds
PREDICTION_HORIZON_MS = 50

# Shots to evaluate for generalizability (must be excluded from train/val/test)
GENERALIZABILITY_SHOTS = [191986, 191992, 169505, 169500, 169472]


class LSTMFirstNN(nn.Module):
    """
    A hybrid model with LSTM processing FIRST (for temporal patterns)
    followed by NN layers (for feature transformation).
    Uses 150 datapoints BEFORE the classification point.
    Predicts state 50ms into the future.
    Unidirectional LSTM only (not bidirectional).
    """

    def __init__(self, n_features, n_classes=4, lstm_hidden=64, nn_hidden_sizes=[128, 64]):
        super(LSTMFirstNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.4,
        )

        lstm_output_size = lstm_hidden  # unidirectional

        nn_layers = []
        input_dim = lstm_output_size
        for hidden_size in nn_hidden_sizes:
            nn_layers.extend(
                [
                    nn.Linear(input_dim, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.45),
                ]
            )
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

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lstm_params = sum(p.numel() for name, p in self.named_parameters() if 'lstm' in name)
        nn_params = sum(p.numel() for name, p in self.named_parameters() if 'nn_layers' in name)
        attention_params = sum(p.numel() for name, p in self.named_parameters() if 'attention' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)

        print(f"\n{'='*60}")
        print("LSTM-First-NN Model Parameter Count:")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("\nParameters by component:")
        print(f"  - LSTM layers: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")
        print(f"  - NN layers: {nn_params:,} ({nn_params/total_params*100:.1f}%)")
        print(f"  - Attention: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
        print(f"  - Classifier: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")
        print(f"{'='*60}")
        print("Architecture: LSTM (unidirectional) → NN → Classifier")

    def forward(self, x):
        # x shape: (batch_size, n_features, sequence_length)
        x = x.transpose(1, 2)  # (batch, seq_len, n_features)

        lstm_output, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)

        attention = self.attention_weights(lstm_output)  # (batch, seq_len, 1)
        attended_features = torch.sum(lstm_output * attention, dim=1)  # (batch, lstm_hidden)

        final_hidden = lstm_output[:, -1, :]  # (batch, lstm_hidden)
        nn_features = self.nn_layers(final_hidden)

        combined = torch.cat([nn_features, attended_features], dim=1)
        return self.classifier(combined)


class PlasmaDataset(Dataset):
    """Dataset class for plasma data windows"""

    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].T, self.labels[idx]


def load_and_prepare_data():
    """Load and preprocess the plasma data - includes time column for future prediction"""
    print("Loading data...")
    df = pd.read_csv('/mnt/homes/sr4240/my_folder/combined_database.csv')

    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()

    important_features = [
        'iln3iamp',
        'betan',
        'density',
        'li',
        'tritop',
        'fs04_past_max_smoothed',
    ]
    selected_features = [f for f in important_features if f in df.columns]
    print(f"Using {len(selected_features)} features: {selected_features}")

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    X = df_sorted[selected_features].values
    y = df_sorted['state'].values
    times = df_sorted['time'].values  # ms
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


def _random_window_split_indices(n_samples, train_frac=0.7, val_frac=0.15, seed=42):
    if n_samples <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def create_windows_with_random_data_point_split(
    X,
    y,
    times,
    shots,
    window_size=150,
    prediction_horizon_ms=50,
    exclude_shots=None,
):
    """Create windows, then split RANDOMLY BY WINDOW (data point), not by shot.
    
    Windows are still constructed within-shot (no cross-shot windows), but once windows
    exist, they are mixed across train/val/test even if they originate from the same shot.
    
    The label is taken from the point that is prediction_horizon_ms in the future
    from the end of the window.
    
    Args:
        exclude_shots: List of shot IDs to exclude from train/val/test splits
    """
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by INDIVIDUAL WINDOWS (mixed across shots)")
    
    if exclude_shots:
        exclude_shots_set = set(exclude_shots)
        print(f"Excluding windows from shots: {exclude_shots}")
    else:
        exclude_shots_set = set()

    unique_shots = np.unique(shots)
    print(f"Total unique shots: {len(unique_shots)}")

    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}

    all_windows = []
    all_labels = []
    all_window_shots = []

    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0
    windows_excluded = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        if len(shot_indices) < window_size:
            continue

        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        for i in range(len(shot_indices) - window_size + 1):
            window = shot_X[i : i + window_size]

            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms

            future_local_idx = np.searchsorted(shot_times, target_time)
            if future_local_idx >= len(shot_times):
                windows_skipped_no_future += 1
                continue

            future_label = shot_labels[future_local_idx]
            if int(future_label) not in label_mapping:
                windows_skipped_invalid_label += 1
                continue

            # Exclude windows from specified shots
            if shot_id in exclude_shots_set:
                windows_excluded += 1
                continue

            if not np.isnan(window).any() and not np.isinf(window).any():
                all_windows.append(window)
                all_labels.append(label_mapping[int(future_label)])
                all_window_shots.append(int(shot_id))
                windows_created += 1

    all_windows = np.array(all_windows, dtype=np.float32)
    all_labels = np.array(all_labels)
    all_window_shots = np.array(all_window_shots)

    print("\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Skipped (no future data): {windows_skipped_no_future:,}")
    print(f"  Skipped (invalid label): {windows_skipped_invalid_label:,}")
    print(f"  Excluded (generalizability shots): {windows_excluded:,}")

    train_idx, val_idx, test_idx = _random_window_split_indices(
        len(all_windows), train_frac=0.7, val_frac=0.15, seed=42
    )

    train_windows = all_windows[train_idx]
    train_labels = all_labels[train_idx]
    val_windows = all_windows[val_idx]
    val_labels = all_labels[val_idx]
    test_windows = all_windows[test_idx]
    test_labels = all_labels[test_idx]

    train_shots = set(all_window_shots[train_idx].tolist())
    val_shots = set(all_window_shots[val_idx].tolist())
    test_shots = set(all_window_shots[test_idx].tolist())

    print(f"\nSplit sizes (windows): Train={len(train_windows)}, Val={len(val_windows)}, Test={len(test_windows)}")
    print(f"Unique shots represented:")
    print(f"  Train: {len(train_shots)} shots")
    print(f"  Val: {len(val_shots)} shots")
    print(f"  Test: {len(test_shots)} shots")
    print("Note: the same shot may appear in multiple splits (intentionally).")

    # Verify excluded shots are not in any split
    if exclude_shots_set:
        train_overlap = train_shots & exclude_shots_set
        val_overlap = val_shots & exclude_shots_set
        test_overlap = test_shots & exclude_shots_set
        if train_overlap or val_overlap or test_overlap:
            print(f"\nWARNING: Found excluded shots in splits!")
            print(f"  Train overlap: {train_overlap}")
            print(f"  Val overlap: {val_overlap}")
            print(f"  Test overlap: {test_overlap}")
        else:
            print(f"\n✓ Verified: Excluded shots are not in train/val/test splits")

    overlaps = {
        "train∩val": len(train_shots & val_shots),
        "train∩test": len(train_shots & test_shots),
        "val∩test": len(val_shots & test_shots),
    }
    print(f"Shot overlap counts: {overlaps}")

    print("\nLabel distribution:")
    print(f"  Train: {Counter(train_labels)}")
    print(f"  Val: {Counter(val_labels)}")
    print(f"  Test: {Counter(test_labels)}")

    return (
        train_windows,
        train_labels,
        val_windows,
        val_labels,
        test_windows,
        test_labels,
    )


def create_windows_for_specific_shots(
    X,
    y,
    times,
    shots,
    target_shots,
    window_size=150,
    prediction_horizon_ms=50,
):
    """Create windows specifically for the target shots (for generalizability evaluation)"""
    print(f"\nCreating windows for generalizability shots: {target_shots}")
    
    target_shots_set = set(target_shots)
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}

    all_windows = []
    all_labels = []
    all_window_shots = []
    all_window_times = []

    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0

    for shot_id in target_shots_set:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        
        if len(shot_indices) < window_size:
            print(f"  Shot {shot_id}: Insufficient data points ({len(shot_indices)} < {window_size})")
            continue

        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        shot_windows = 0
        for i in range(len(shot_indices) - window_size + 1):
            window = shot_X[i : i + window_size]

            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms

            future_local_idx = np.searchsorted(shot_times, target_time)
            if future_local_idx >= len(shot_times):
                windows_skipped_no_future += 1
                continue

            future_label = shot_labels[future_local_idx]
            if int(future_label) not in label_mapping:
                windows_skipped_invalid_label += 1
                continue

            if not np.isnan(window).any() and not np.isinf(window).any():
                all_windows.append(window)
                all_labels.append(label_mapping[int(future_label)])
                all_window_shots.append(int(shot_id))
                all_window_times.append(window_end_time)
                windows_created += 1
                shot_windows += 1
        
        print(f"  Shot {shot_id}: Created {shot_windows} windows")

    if len(all_windows) == 0:
        print("  WARNING: No windows created for generalizability shots!")
        return None, None, None

    all_windows = np.array(all_windows, dtype=np.float32)
    all_labels = np.array(all_labels)
    all_window_shots = np.array(all_window_shots)
    all_window_times = np.array(all_window_times)

    print(f"\nGeneralizability window statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Skipped (no future data): {windows_skipped_no_future:,}")
    print(f"  Skipped (invalid label): {windows_skipped_invalid_label:,}")
    print(f"\nLabel distribution: {Counter(all_labels)}")

    return all_windows, all_labels, all_window_shots


def train_model(model, train_loader, val_loader, device, n_epochs=7):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 3

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels_list.extend(batch_y.detach().cpu().numpy())

        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list2 = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list2.extend(batch_y.cpu().numpy())

        train_acc = accuracy_score(train_labels_list, train_preds)
        val_acc = accuracy_score(val_labels_list2, val_preds)

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))

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
            torch.save(model.state_dict(), 'best_lstm_50ms_random_data_generalizability.pth')
            patience_counter = 0
            print("  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device, class_names, dataset_name="Test"):
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

    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Classification Report:")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{dataset_name} Accuracy: {test_acc:.4f}")

    print(f"\n{dataset_name} ROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        if i < all_probs.shape[1]:
            class_labels = (all_labels == i).astype(int)
            if len(np.unique(class_labels)) > 1:
                auc = roc_auc_score(class_labels, all_probs[:, i])
                print(f"  {class_name}: {auc:.4f}")

    return all_preds, all_labels, all_probs


def evaluate_by_shot(model, windows, labels, window_shots, device, class_names):
    """Evaluate model performance broken down by shot"""
    print(f"\n{'='*60}")
    print("Generalizability Evaluation by Shot:")
    print(f"{'='*60}")
    
    unique_shots = np.unique(window_shots)
    shot_results = {}
    
    model.eval()
    
    for shot_id in unique_shots:
        shot_mask = window_shots == shot_id
        shot_windows = windows[shot_mask]
        shot_labels = labels[shot_mask]
        
        if len(shot_windows) == 0:
            continue
        
        shot_dataset = PlasmaDataset(shot_windows, shot_labels)
        shot_loader = DataLoader(shot_dataset, batch_size=2048, shuffle=False)
        
        shot_preds = []
        shot_labels_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in shot_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                
                shot_preds.extend(preds.cpu().numpy())
                shot_labels_list.extend(batch_y.numpy())
        
        shot_preds = np.array(shot_preds)
        shot_labels_list = np.array(shot_labels_list)
        
        shot_acc = accuracy_score(shot_labels_list, shot_preds)
        shot_results[shot_id] = {
            'accuracy': shot_acc,
            'n_samples': len(shot_labels_list),
            'label_distribution': Counter(shot_labels_list),
        }
        
        print(f"\nShot {shot_id}:")
        print(f"  Accuracy: {shot_acc:.4f}")
        print(f"  Samples: {shot_results[shot_id]['n_samples']}")
        print(f"  Label distribution: {shot_results[shot_id]['label_distribution']}")
    
    return shot_results


def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names):
    """Plot training curves and confusion matrix"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Validation Loss ({PREDICTION_HORIZON_MS}ms Prediction)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Training and Validation Accuracy ({PREDICTION_HORIZON_MS}ms Prediction)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title('Normalized Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    cm_counts = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title('Confusion Matrix (Counts)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(
        f'lstm_{PREDICTION_HORIZON_MS}ms_random_data_generalizability_results.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
    print(f"Results saved to 'lstm_{PREDICTION_HORIZON_MS}ms_random_data_generalizability_results.png'")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("LSTM-NN Model for Plasma Classification - Generalizability Study")
    print("=" * 60)
    print("Architecture: Unidirectional LSTM → NN → Classifier")
    print("Window: 150 datapoints BEFORE current time")
    print(f"Prediction: {PREDICTION_HORIZON_MS}ms INTO THE FUTURE")
    print("Split: RANDOM BY WINDOW / DATA POINT (NOT by shot)")
    print(f"Generalizability shots: {GENERALIZABILITY_SHOTS}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, scaler = load_and_prepare_data()

    # Create windows excluding generalizability shots from train/val/test
    train_X, train_y, val_X, val_y, test_X, test_y = create_windows_with_random_data_point_split(
        X, y, times, shots, prediction_horizon_ms=PREDICTION_HORIZON_MS,
        exclude_shots=GENERALIZABILITY_SHOTS
    )

    print("\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val: {len(val_X)} samples")
    print(f"  Test: {len(test_X)} samples")

    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    model = LSTMFirstNN(n_features=len(features), n_classes=4).to(device)

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
        model, train_loader, val_loader, device, n_epochs=7
    )

    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_lstm_50ms_random_data_generalizability.pth'))

    class_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
    
    # Evaluate on test set
    all_preds, all_labels, _all_probs = evaluate_model(model, test_loader, device, class_names, "Test")

    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Now evaluate on generalizability shots
    print("\n" + "=" * 60)
    print("Evaluating on Generalizability Shots")
    print("=" * 60)
    
    gen_windows, gen_labels, gen_window_shots = create_windows_for_specific_shots(
        X, y, times, shots, GENERALIZABILITY_SHOTS, 
        prediction_horizon_ms=PREDICTION_HORIZON_MS
    )
    
    if gen_windows is not None and len(gen_windows) > 0:
        gen_dataset = PlasmaDataset(gen_windows, gen_labels)
        gen_loader = DataLoader(gen_dataset, batch_size=2048, shuffle=False)
        
        gen_preds, gen_labels_eval, gen_probs = evaluate_model(
            model, gen_loader, device, class_names, "Generalizability"
        )
        
        gen_acc = accuracy_score(gen_labels_eval, gen_preds)
        print(f"\nOverall Generalizability Accuracy: {gen_acc:.4f}")
        
        # Evaluate by individual shot
        shot_results = evaluate_by_shot(
            model, gen_windows, gen_labels, gen_window_shots, device, class_names
        )
        
        print(f"\n{'='*60}")
        print("Summary of Generalizability Results:")
        print(f"{'='*60}")
        for shot_id in sorted(shot_results.keys()):
            result = shot_results[shot_id]
            print(f"Shot {shot_id}: Accuracy = {result['accuracy']:.4f}, Samples = {result['n_samples']}")
    else:
        print("WARNING: Could not create windows for generalizability shots!")
    
    print("\n" + "=" * 60)
    print(f"Training and Evaluation Complete! (Predicting {PREDICTION_HORIZON_MS}ms into the future)")
    print("=" * 60)


if __name__ == "__main__":
    main()
