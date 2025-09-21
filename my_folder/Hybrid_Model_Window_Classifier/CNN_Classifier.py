import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Enable cuDNN benchmarking and allow TF32 on supported GPUs for speedups
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    # Optional but usually safe and faster on Ampere+: set False if you need strict FP32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_output = torch.sum(attention_weights * x, dim=1)
        return attended_output

class PlasmaDataset(Dataset):
    def __init__(self, features, labels, shots, window_size=150):
        self.features = features.astype(np.float32)  # Convert to float32 early
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
                window_label = Counter(window_labels).most_common(1)[0][0]
                label_counts = Counter(window_labels)
                if (max(label_counts.values()) >= window_size * 0.7) and (window_label != 0):
                    self.windows.append(window_data)
                    self.window_labels.append(window_label)
        self.windows = np.array(self.windows, dtype=np.float32)
        self.window_labels = np.array(self.window_labels)
        # Map states: 1->0, 2->1, 3->2, 4->3
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        self.window_labels = np.array([label_mapping[int(label)] for label in self.window_labels])
        print(f"Created {len(self.windows)} windows with window size {window_size}ms")
        print(f"Label distribution: {Counter(self.window_labels)})")
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.windows[idx]).T
        label = torch.LongTensor([self.window_labels[idx]])
        return features, label.squeeze()

class PlasmaCNNBiLSTMAttention(nn.Module):
    def __init__(self, n_features, n_classes=4, window_size=150):
        super(PlasmaCNNBiLSTMAttention, self).__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4)
        )
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, 
                              batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = AttentionModule(256)  # 128*2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    def forward(self, x):
        x = self.feature_conv(x)  # (B, 256, T)
        x = x.permute(0, 2, 1)    # (B, T, 256)
        x, _ = self.bilstm(x)     # (B, T, 256)
        x = self.attention(x)      # (B, 256)
        x = self.classifier(x)
        return x

def load_and_preprocess_data():
    print("Loading CSV dataset...")
    df = pd.read_csv('plasma_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    # Remove rows with shot number 191675
    df = df[df['shot'] != 191675].copy()
    print(f"Removed rows with shot 191675. New shape: {df.shape}")
    return df

def select_features(df):
    columns_to_remove = ['ELMPMARKER', 'zeff', 'rotation_edge', 'rotation_core']
    important_features = [
        'n_eped', 'li', 'q95', 'bt0', 'kappa', 'tribot', 'tritop', 'dR_sep', 'thin_fs04_max_smoothed',

    ]
    available_features = [col for col in df.columns if col not in columns_to_remove]
    selected_features = [f for f in important_features if f in available_features]
    print(f"Selected {len(selected_features)} features: {selected_features}")
    return selected_features

def prepare_data(df, selected_features):
    print("Preparing data with 70/10/20 shot-based split (random shot split with seed=42)...")
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    # Remove N/A states (state=0)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()
    print(f"Data after filtering N/A states: {df_filtered.shape}")
    print(f"State distribution: {df_filtered['state'].value_counts().sort_index()}")

    # Random shot split identical to Binary_Random_Forest.py
    unique_shots = df_filtered['shot'].unique()
    num_shots = len(unique_shots)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    train_count = int(np.floor(0.70 * num_shots))
    val_count = int(np.floor(0.10 * num_shots))
    train_shots = shuffled_shots[:train_count]
    val_shots = shuffled_shots[train_count:train_count + val_count]
    test_shots = shuffled_shots[train_count + val_count:]

    print(f"Total shots: {num_shots} | Train shots: {len(train_shots)} | Val shots: {len(val_shots)} | Test shots: {len(test_shots)}")

    # Split data by shot
    train_df = df_filtered[df_filtered['shot'].isin(train_shots)]
    val_df = df_filtered[df_filtered['shot'].isin(val_shots)]
    test_df = df_filtered[df_filtered['shot'].isin(test_shots)]
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")
    
    # Process training data
    X_train = train_df[selected_features].values
    y_train = train_df['state'].values
    
    # Process validation data
    X_val = val_df[selected_features].values
    y_val = val_df['state'].values
    
    # Process test data
    X_test = test_df[selected_features].values
    y_test = test_df['state'].values
    
    # Feature engineering (done separately for train and test to avoid leakage)
    # Training features
    X_train_diff = np.diff(X_train, axis=0, prepend=X_train[0:1])
    X_train_ma = np.convolve(X_train.flatten(), np.ones(5)/5, mode='same').reshape(X_train.shape)
    X_train = np.concatenate([X_train, X_train_diff, X_train_ma], axis=1)
    
    # Validation features (using same prepend value as training for consistency)
    X_val_diff = np.diff(X_val, axis=0, prepend=X_val[0:1])
    X_val_ma = np.convolve(X_val.flatten(), np.ones(5)/5, mode='same').reshape(X_val.shape)
    X_val = np.concatenate([X_val, X_val_diff, X_val_ma], axis=1)
    
    # Test features (using same prepend value as training for consistency)
    X_test_diff = np.diff(X_test, axis=0, prepend=X_test[0:1])
    X_test_ma = np.convolve(X_test.flatten(), np.ones(5)/5, mode='same').reshape(X_test.shape)
    X_test = np.concatenate([X_test, X_test_diff, X_test_ma], axis=1)
    
    print(f"Training set size: {len(X_train)} (shots: {len(train_shots)})")
    print(f"Validation set size: {len(X_val)} (shots: {len(val_shots)})")
    print(f"Test set size: {len(X_test)} (shots: {len(test_shots)})")
    
    # Scaling (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE oversampling for minority classes (only on training data)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {Counter(y_train_bal)})")
    
    return X_train_bal, X_val_scaled, X_test_scaled, y_train_bal, y_val, y_test

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, n_features, window_size = 150):
    print("Creating datasets...")
    train_dataset = PlasmaDataset(X_train, y_train, window_size)
    val_dataset = PlasmaDataset(X_val, y_val, window_size)
    test_dataset = PlasmaDataset(X_test, y_test, window_size)
    num_workers = min(8, os.cpu_count() or 4)
    use_cuda = torch.cuda.is_available()

    if num_workers > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=2048,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=2048,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )
    print(f"Training batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compute class weights for imbalanced classes (now only 4 classes)
    all_train_labels = train_dataset.window_labels
    n_classes = 4
    present_classes = np.unique(all_train_labels)
    present_weights = compute_class_weight('balanced', classes=present_classes, y=all_train_labels)
    class_weights = np.ones(n_classes, dtype=np.float32)
    for cls, w in zip(present_classes, present_weights):
        class_weights[int(cls)] = w
    print(f"Class weights (for classes 0-3): {class_weights}")
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Use Focal Loss for better handling of class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)

    model = PlasmaCNNBiLSTMAttention(n_features=n_features, n_classes=n_classes, window_size=window_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Cosine annealing scheduler with warmup
    def warmup_cosine_schedule(epoch):
        if epoch < 5:
            return epoch / 5
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / 45))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    n_epochs = 50
    patience = 10
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
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            # Progress indicator
            progress = (batch_idx + 1) / total_batches * 100
            print(f"\rTraining progress: {progress:.1f}%", end="", flush=True)
        
        print()  # New line after progress bar
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        val_accuracy = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print results after every epoch
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_plasma_cnn.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load('best_plasma_cnn.pth'))
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    state_names = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=state_names))
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=state_names, yticklabels=state_names)
    plt.title(f'Normalized Confusion Matrix - Accuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
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
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    return model

def main():
    print("=== Plasma State Classification CNN ===")
    df = load_and_preprocess_data()
    selected_features = select_features(df)
    window_size = 150
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, selected_features)
    model = train_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                               n_features=X_train.shape[1], window_size=window_size)
    print("\n=== Training Complete ===")
    print("Model saved as 'best_plasma_cnn.pth'")
    print("Confusion matrix and training curves saved as images")

if __name__ == "__main__":
    main()