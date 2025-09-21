import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings('ignore')

# Speedups
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Reproducibility
np.random.seed(44)
torch.manual_seed(44)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal = alpha_t * focal
            else:
                focal = self.alpha * focal
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def load_and_preprocess_data():
    print("Loading CSV dataset...")
    df = pd.read_csv('plasma_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("Missing values per column:")
    print(df.isnull().sum())
    # Remove problematic shot if present for parity with CNN script
    if 'shot' in df.columns:
        df = df[df['shot'] != 191675].copy()
        print(f"Removed rows with shot 191675. New shape: {df.shape}")
    return df


def select_features(df):
    columns_to_remove = ['ELMPMARKER', 'zeff', 'rotation_edge', 'rotation_core']
    important_features = [
        'n_eped', 'n_e', 'betan', 'li', 'q95', 'Ip',
        'bt0', 'kappa', 'tribot', 'tritop', 'dR_sep', 'fs04', 'fs04_max_avg', 'fs04_max_smoothed'
    ]
    available_features = [col for col in df.columns if col not in columns_to_remove]
    selected_features = [f for f in important_features if f in available_features]
    print(f"Selected {len(selected_features)} features: {selected_features}")
    return selected_features


def prepare_data(df, selected_features):
    print("Preparing data with 60/20/20 shot-based split (random shot split with seed=42)...")
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    # Remove N/A states (state=0)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()
    print(f"Data after filtering N/A states: {df_filtered.shape}")
    print(f"Original state distribution: {df_filtered['state'].value_counts().sort_index()}")

    unique_shots = df_filtered['shot'].unique()
    num_shots = len(unique_shots)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    train_count = int(np.floor(0.60 * num_shots))
    val_count = int(np.floor(0.20 * num_shots))
    train_shots = shuffled_shots[:train_count]
    val_shots = shuffled_shots[train_count:train_count + val_count]
    test_shots = shuffled_shots[train_count + val_count:]

    print(f"Total shots: {num_shots} | Train shots: {len(train_shots)} | Val shots: {len(val_shots)} | Test shots: {len(test_shots)}")

    train_df = df_filtered[df_filtered['shot'].isin(train_shots)]
    val_df = df_filtered[df_filtered['shot'].isin(val_shots)]
    test_df = df_filtered[df_filtered['shot'].isin(test_shots)]
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

    def engineer(sub_df):
        X = sub_df[selected_features].values
        y = sub_df['state'].values
        shots = sub_df['shot'].values
        engineered = []
        for shot_id in np.unique(shots):
            idx = np.where(shots == shot_id)[0]
            Xs = X[idx]
            # per-column diff
            X_diff = np.vstack([np.zeros((1, Xs.shape[1])), np.diff(Xs, axis=0)])
            # per-column moving average (window=5)
            k = 5
            kernel = np.ones(k) / k
            X_ma = np.zeros_like(Xs)
            for c in range(Xs.shape[1]):
                X_ma[:, c] = np.convolve(Xs[:, c], kernel, mode='same')
            engineered.append(np.concatenate([Xs, X_diff, X_ma], axis=1))
        X_engineered = np.vstack(engineered)
        return X_engineered, y, shots

    X_train, y_train, shots_train = engineer(train_df)
    X_val, y_val, shots_val = engineer(val_df)
    X_test, y_test, shots_test = engineer(test_df)

    print(f"Training set size: {len(X_train)} (shots: {len(train_shots)})")
    print(f"Validation set size: {len(X_val)} (shots: {len(val_shots)})")
    print(f"Test set size: {len(X_test)} (shots: {len(test_shots)})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, shots_train, shots_val, shots_test


class PlasmaRNNWindowDataset(Dataset):
    def __init__(self, features, labels, shots, window_size=100):
        self.features = features
        self.labels = labels
        self.shots = shots
        self.window_size = window_size
        self.windows = []
        self.window_labels = []

        unique_shots = np.unique(self.shots)
        for shot_id in unique_shots:
            shot_indices = np.where(self.shots == shot_id)[0]
            if len(shot_indices) < window_size:
                continue
            start_idx = shot_indices[0]
            end_idx = shot_indices[-1]
            for i in range(start_idx, end_idx - window_size + 2):
                if (i + window_size - 1) > end_idx:
                    break
                window_slice = slice(i, i + window_size)
                window_data = self.features[window_slice]
                window_labels = self.labels[window_slice]
                # Majority label with at least 70% dominance and not 0
                label_counts = Counter(window_labels)
                majority_label, count = Counter(window_labels).most_common(1)[0]
                if (count >= window_size * 0.7) and (majority_label != 0):
                    self.windows.append(window_data)
                    self.window_labels.append(majority_label)

        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)

        # Map 1,2,3 -> 0 and 4 -> 1
        label_mapping = {1: 0, 2: 0, 3: 0, 4: 1}
        self.window_labels = np.array([label_mapping[int(label)] for label in self.window_labels])

        print(f"Created {len(self.windows)} windows with window size {window_size}ms (±{window_size//2}ms context)")
        print(f"Label distribution: {Counter(self.window_labels)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Return shape (seq_len, n_features) for RNNs
        features = torch.FloatTensor(self.windows[idx])
        label = torch.LongTensor([self.window_labels[idx]])
        return features, label.squeeze()


class PlasmaBinaryBiRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, n_classes=2, dropout=0.5):
        super(PlasmaBinaryBiRNN, self).__init__()
        # Single bidirectional RNN layer with further reduced hidden size
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # No dropout between layers since only 1 layer
        )
        
        # Use mean pooling instead of attention for sequence aggregation
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Even smaller classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        outputs, h_n = self.rnn(x)
        # outputs: (batch, seq_len, hidden_dim * 2)
        
        # Use mean pooling across sequence length instead of attention
        # Transpose to (batch, hidden_dim * 2, seq_len) for pooling
        pooled = self.pooling(outputs.transpose(1, 2))
        # Squeeze to (batch, hidden_dim * 2)
        pooled = pooled.squeeze(-1)
        
        # Classify
        logits = self.classifier(pooled)
        return logits


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    shots_train,
    shots_val,
    shots_test,
    input_dim,
    window_size=100,
):
    print("Creating datasets...")
    train_dataset = PlasmaRNNWindowDataset(X_train, y_train, shots_train, window_size)
    val_dataset = PlasmaRNNWindowDataset(X_val, y_val, shots_val, window_size)
    test_dataset = PlasmaRNNWindowDataset(X_test, y_test, shots_test, window_size)

    num_workers = min(8, os.cpu_count() or 4)
    use_cuda = torch.cuda.is_available()

    # Weighted sampler for balanced batches
    class_counts = Counter(train_dataset.window_labels)
    num_classes = 2
    class_weights_list = [0.0] * num_classes
    for c in range(num_classes):
        class_weights_list[c] = 1.0 / max(1, class_counts.get(c, 0))
    sample_weights = np.array([class_weights_list[label] for label in train_dataset.window_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    common_loader_kwargs = dict(
        batch_size=128,  # Smaller batch size to reduce overfitting
        shuffle=False,
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=use_cuda,
    )

    train_loader = DataLoader(train_dataset, sampler=sampler, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, **common_loader_kwargs)

    print(f"Training batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Class weights for focal loss
    all_train_labels = train_dataset.window_labels
    n_classes = 2
    present_classes = np.unique(all_train_labels)
    present_weights = compute_class_weight('balanced', classes=present_classes, y=all_train_labels)
    class_weights = np.ones(n_classes, dtype=np.float32)
    for cls, w in zip(present_classes, present_weights):
        class_weights[int(cls)] = w
    print(f"Class weights (for classes 0-1): {class_weights}")
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    model = PlasmaBinaryBiRNN(input_dim=input_dim, n_classes=n_classes).to(device)
    
    # Conservative optimizer settings to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2, betas=(0.9, 0.999))

    n_epochs = 30  # Reduced epochs
    patience = 8   # Reduced patience for faster early stopping
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Longer warmup
        anneal_strategy='cos',
        div_factor=50.0,  # Much lower initial LR
        final_div_factor=100.0,
    )

    patience_counter = 0
    train_losses = []
    val_losses = []
    best_val_accuracy = 0.0

    print("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)

        print(f"\nEpoch {epoch+1}/{n_epochs}")
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            # (B, T, F)
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            # Very aggressive gradient clipping to prevent overfitting
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            progress = (batch_idx + 1) / total_batches * 100
            print(f"\rTraining progress: {progress:.1f}%", end="", flush=True)

        print()

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        val_probabilities = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
                val_probabilities.extend(probabilities[:, 1].cpu().numpy())
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_plasma_binary_birnn_50ms_anti_overfitting.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model and tune decision threshold on validation set
    model.load_state_dict(torch.load('best_plasma_binary_birnn_50ms_anti_overfitting.pth'))
    model.eval()
    val_probs = []
    val_true = []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(probabilities.cpu().numpy())
            val_true.extend(batch_labels.cpu().numpy())
    val_probs = np.array(val_probs)
    val_true = np.array(val_true)
    best_thr = 0.5
    best_acc = 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (val_probs >= thr).astype(int)
        acc = accuracy_score(val_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    print(f"Selected probability threshold on validation: {best_thr:.2f} (Val Acc: {best_acc:.4f})")

    # Evaluate on test with tuned threshold
    all_predictions = []
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = (probabilities >= best_thr).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    state_names = ['Suppressed', 'ELMing']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=state_names))

    # ROC AUC
    auc_score = roc_auc_score(all_labels, all_probabilities)
    print(f"ROC AUC Score: {auc_score:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - BiRNN (±50ms) - Anti-Overfitting')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve_birnn_50ms_anti_overfitting.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=state_names, yticklabels=state_names)
    plt.title(f'Normalized Confusion Matrix - Accuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_binary_birnn_50ms_anti_overfitting.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss - BiRNN (±50ms) - Anti-Overfitting')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves_binary_birnn_50ms_anti_overfitting.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model


def main():
    print("=== Plasma Binary State Classification BiRNN (±50ms) - Optimized ===")
    print("States: Suppressed (1,2,3) vs ELMing (4)")
    print("Architecture: Single BiRNN + Attention (~15K params) - Anti-Overfitting")
    df = load_and_preprocess_data()
    selected_features = select_features(df)

    # 50 ms in both directions -> total 100 ms window
    context_ms = 50
    window_size = context_ms * 2

    X_train, X_val, X_test, y_train, y_val, y_test, shots_train, shots_val, shots_test = prepare_data(df, selected_features)
    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        shots_train,
        shots_val,
        shots_test,
        input_dim=X_train.shape[1],
        window_size=window_size,
    )
    print("\n=== Training Complete ===")
    print("Model saved as 'best_plasma_binary_birnn_50ms_anti_overfitting.pth'")
    print("ROC curve, confusion matrix and training curves saved as images")


if __name__ == "__main__":
    main()


