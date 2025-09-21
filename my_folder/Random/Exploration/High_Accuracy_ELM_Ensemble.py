import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import optuna
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Speedups
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

class AdvancedFocalLoss(nn.Module):
    """Advanced Focal Loss with label smoothing and adaptive alpha"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.label_smoothing)(inputs, targets)
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

class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for capturing different temporal patterns"""
    def __init__(self, hidden_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_scales)
        ])
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        # Multi-scale attention
        attention_outputs = []
        for i, attention in enumerate(self.attention_modules):
            # Apply attention at different scales
            if i == 0:  # Full sequence
                scale_x = x
            elif i == 1:  # Half sequence
                mid = seq_len // 2
                scale_x = x[:, mid//2:mid+mid//2, :]
            else:  # Quarter sequence
                quarter = seq_len // 4
                scale_x = x[:, quarter:quarter*3, :]
            
            # Compute attention weights
            attention_weights = torch.softmax(attention(scale_x), dim=1)
            weighted_output = torch.sum(attention_weights * scale_x, dim=1)
            attention_outputs.append(weighted_output)
        
        # Weighted combination of different scales
        scale_weights = torch.softmax(self.scale_weights, dim=0)
        final_output = sum(w * out for w, out in zip(scale_weights, attention_outputs))
        
        return final_output

class AdvancedCNN(nn.Module):
    """Advanced CNN with multi-scale convolutions and residual connections"""
    def __init__(self, n_features, n_classes=2, dropout=0.3):
        super().__init__()
        
        # Multi-scale feature extraction
        self.conv1_3 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.conv1_7 = nn.Conv1d(n_features, 32, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(96)  # 32*3 = 96
        self.conv2 = nn.Conv1d(96, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Residual connections
        self.residual_conv = nn.Conv1d(n_features, 128, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, features, seq_len)
        residual = self.residual_conv(x)
        
        # Multi-scale convolutions
        conv1_3 = self.relu(self.conv1_3(x))
        conv1_5 = self.relu(self.conv1_5(x))
        conv1_7 = self.relu(self.conv1_7(x))
        
        # Concatenate multi-scale features
        x = torch.cat([conv1_3, conv1_5, conv1_7], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Add residual connection
        x = x + residual
        
        # Global pooling and classification
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x

class AdvancedTCN(nn.Module):
    """Advanced Temporal Convolutional Network with dilated convolutions"""
    def __init__(self, n_features, n_classes=2, dropout=0.3):
        super().__init__()
        
        self.input_conv = nn.Conv1d(n_features, 64, kernel_size=1)
        
        # Dilated convolutions with increasing dilation
        self.tcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, dilation=2**i, padding=2**i),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(4)
        ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=1) for _ in range(4)
        ])
        
        self.output_conv = nn.Conv1d(64, 128, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, features, seq_len)
        x = self.input_conv(x)
        
        # TCN layers with skip connections
        for i, (tcn_layer, skip_conv) in enumerate(zip(self.tcn_layers, self.skip_connections)):
            residual = skip_conv(x)
            x = tcn_layer(x)
            x = x + residual
            x = torch.relu(x)
        
        x = self.output_conv(x)
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x

class AdvancedBiRNN(nn.Module):
    """Advanced Bidirectional RNN with LSTM and GRU combination"""
    def __init__(self, n_features, n_classes=2, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size // 2,  # Bidirectional will double this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-scale attention
        self.attention = MultiScaleAttention(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size = x.size(0)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # GRU processing
        gru_out, _ = self.gru(lstm_out)
        gru_out = self.dropout(gru_out)
        
        # Multi-scale attention
        attended = self.attention(gru_out)
        
        # Classification
        output = self.classifier(attended)
        
        return output

class EnsembleModel:
    """Ensemble of CNN, TCN, BiRNN, and Random Forest"""
    def __init__(self, n_features, n_classes=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Initialize models
        self.cnn = AdvancedCNN(n_features, n_classes).to(device)
        self.tcn = AdvancedTCN(n_features, n_classes).to(device)
        self.birnn = AdvancedBiRNN(n_features, n_classes).to(device)
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        
        # Ensemble weights (will be optimized)
        self.ensemble_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
    def forward(self, x, x_flat=None):
        """Forward pass through ensemble"""
        # Deep learning models
        cnn_out = torch.softmax(self.cnn(x), dim=1).detach().cpu().numpy()
        tcn_out = torch.softmax(self.tcn(x), dim=1).detach().cpu().numpy()
        birnn_out = torch.softmax(self.birnn(x), dim=1).detach().cpu().numpy()
        
        # Random Forest (if x_flat is provided)
        if x_flat is not None and hasattr(self.rf, 'predict_proba'):
            rf_out = self.rf.predict_proba(x_flat)
        else:
            rf_out = np.zeros_like(cnn_out)
        
        # Weighted ensemble
        ensemble_out = (self.ensemble_weights[0] * cnn_out + 
                       self.ensemble_weights[1] * tcn_out + 
                       self.ensemble_weights[2] * birnn_out + 
                       self.ensemble_weights[3] * rf_out)
        
        return ensemble_out

class AdvancedPlasmaDataset(Dataset):
    """Advanced dataset with multiple window sizes and data augmentation"""
    def __init__(self, features, labels, shots, window_size=150, augment=True):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.shots = shots
        self.window_size = window_size
        self.augment = augment
        self.windows, self.window_labels = self._create_windows()
        
        # Convert to binary classification
        label_mapping = {1: 0, 2: 0, 3: 0, 4: 1}  # Suppressed/Dithering/Mitigated -> 0, ELMing -> 1
        self.window_labels = np.array([label_mapping[int(label)] for label in self.window_labels])
        
        print(f"Created {len(self.windows)} windows | Label distribution: {Counter(self.window_labels)}")
    
    def _create_windows(self):
        windows, window_labels = [], []
        
        for shot_id in np.unique(self.shots):
            shot_indices = np.where(self.shots == shot_id)[0]
            if len(shot_indices) < self.window_size:
                continue
                
            start_idx, end_idx = shot_indices[0], shot_indices[-1]
            
            for i in range(start_idx, end_idx - self.window_size + 2):
                if (i + self.window_size - 1) > end_idx:
                    break
                    
                window_slice = slice(i, i + self.window_size)
                window_data = self.features[window_slice]
                window_labels_slice = self.labels[window_slice]
                
                # Majority label with at least 70% dominance and not 0
                label_counts = Counter(window_labels_slice)
                majority_label, count = label_counts.most_common(1)[0]
                
                if (count >= self.window_size * 0.7) and (majority_label != 0):
                    windows.append(window_data)
                    window_labels.append(majority_label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window_data = self.windows[idx]
        label = self.window_labels[idx]
        
        if self.augment and np.random.random() < 0.3:
            # Data augmentation: add small noise
            noise = np.random.normal(0, 0.01, window_data.shape)
            window_data = window_data + noise
            
            # Time shifting (small random shifts)
            if np.random.random() < 0.2:
                shift = np.random.randint(-2, 3)
                if shift > 0:
                    window_data = np.vstack([window_data[shift:], window_data[-shift:]])
                elif shift < 0:
                    window_data = np.vstack([window_data[:shift], window_data[:-shift]])
        
        return torch.FloatTensor(window_data).T, torch.LongTensor([label]).squeeze()

def load_and_preprocess_data():
    """Load and preprocess the plasma dataset"""
    print("=== Loading Plasma Dataset ===")
    
    # Load CSV dataset
    df = pd.read_csv('plasma_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove problematic shot if present
    if 'shot' in df.columns:
        df = df[df['shot'] != 191675].copy()
        print(f"Removed rows with shot 191675. New shape: {df.shape}")
    
    return df

def select_features(df):
    """Select only the most essential plasma physics features"""
    # Only keep the most important features for ELM detection
    essential_features = [
        'n_eped',      # Pedestal density
        'betan',       # Beta normalized
        'li',          # Internal inductance
        'q95',         # Safety factor
        'Ip',          # Plasma current
        'bt0',         # Toroidal magnetic field
        'kappa',       # Elongation
        'dR_sep',      # Distance to separatrix
        'fs04',        # Fluctuation signal
        'fs04_max_smoothed',  # Smoothed fluctuation max
        'fs_up_sum',   # Fluctuation sum up
        'fs_sum'       # Fluctuation sum
    ]
    
    # Select only features that exist in the dataset
    selected_features = [f for f in essential_features if f in df.columns]
    
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Selected {len(selected_features)} essential features: {selected_features}")
    return selected_features

def engineer_features(df, selected_features):
    """Advanced feature engineering with multiple techniques"""
    print("=== Advanced Feature Engineering ===")
    
    X = df[selected_features].values
    y = df['state'].values
    shots = df['shot'].values
    
    engineered_features = []
    
    # Progress bar for shot processing
    unique_shots = np.unique(shots)
    print(f"Processing {len(unique_shots)} shots...")
    
    for shot_id in tqdm(unique_shots, desc="Engineering features", unit="shot"):
        idx = np.where(shots == shot_id)[0]
        if len(idx) < 10:  # Skip very short shots
            continue
            
        Xs = X[idx]
        
        # 1. Original features
        X_orig = Xs.copy()
        
        # 2. First-order differences
        X_diff = np.vstack([np.zeros((1, Xs.shape[1])), np.diff(Xs, axis=0)])
        
        # 3. Second-order differences
        X_diff2 = np.vstack([np.zeros((2, Xs.shape[1])), np.diff(Xs, n=2, axis=0)])
        
        # 4. Moving averages with multiple windows
        windows = [3, 5, 7]
        X_ma = np.zeros((Xs.shape[0], len(windows) * Xs.shape[1]))
        
        for i, window in enumerate(windows):
            kernel = np.ones(window) / window
            for j in range(Xs.shape[1]):
                X_ma[:, i * Xs.shape[1] + j] = np.convolve(Xs[:, j], kernel, mode='same')
        
        # 5. Rolling statistics
        X_rolling = np.zeros((Xs.shape[0], 2 * Xs.shape[1]))  # mean and std
        for j in range(Xs.shape[1]):
            for k in range(Xs.shape[0]):
                start_idx = max(0, k - 3)
                end_idx = min(Xs.shape[0], k + 4)
                X_rolling[k, j * 2] = np.mean(Xs[start_idx:end_idx, j])
                X_rolling[k, j * 2 + 1] = np.std(Xs[start_idx:end_idx, j])
        
        # 6. Rate of change
        X_rate = np.zeros_like(Xs)
        for j in range(Xs.shape[1]):
            for k in range(1, Xs.shape[0]):
                if Xs[k-1, j] != 0:
                    X_rate[k, j] = (Xs[k, j] - Xs[k-1, j]) / Xs[k-1, j]
        
        # 7. Handle NaN and infinite values
        X_combined = np.concatenate([
            X_orig, X_diff, X_diff2, X_ma, X_rolling, X_rate
        ], axis=1)
        
        # Replace NaN and infinite values with 0
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
        
        engineered_features.append(X_combined)
        
        # X_combined is already created above with NaN handling
    
    print("Stacking engineered features...")
    X_engineered = np.vstack(engineered_features)
    
    # Update labels and shots to match engineered features
    y_engineered = []
    shots_engineered = []
    
    for shot_id in tqdm(unique_shots, desc="Updating labels", unit="shot"):
        idx = np.where(shots == shot_id)[0]
        if len(idx) >= 10:
            y_engineered.extend(y[idx])
            shots_engineered.extend(shots[idx])
    
    y_engineered = np.array(y_engineered)
    shots_engineered = np.array(shots_engineered)
    
    print(f"Feature engineering complete:")
    print(f"  Original features: {len(selected_features)}")
    print(f"  Engineered features: {X_engineered.shape[1]}")
    print(f"  Total samples: {len(X_engineered)}")
    
    return X_engineered, y_engineered, shots_engineered

def prepare_data(df, selected_features):
    """Prepare data with advanced preprocessing and shot-based splitting"""
    print("=== Data Preparation ===")
    
    # Engineer features
    X_engineered, y_engineered, shots_engineered = engineer_features(df, selected_features)
    
    # Remove N/A states (state=0)
    valid_mask = y_engineered != 0
    X_filtered = X_engineered[valid_mask]
    y_filtered = y_engineered[valid_mask]
    shots_filtered = shots_engineered[valid_mask]
    
    print(f"Data after filtering N/A states: {X_filtered.shape}")
    print(f"State distribution: {Counter(y_filtered)}")
    
    # Shot-based splitting (60/20/20)
    unique_shots = np.unique(shots_filtered)
    num_shots = len(unique_shots)
    
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    
    train_count = int(np.floor(0.60 * num_shots))
    val_count = int(np.floor(0.20 * num_shots))
    
    train_shots = shuffled_shots[:train_count]
    val_shots = shuffled_shots[train_count:train_count + val_count]
    test_shots = shuffled_shots[train_count + val_count:]
    
    print(f"Total shots: {num_shots}")
    print(f"Train shots: {len(train_shots)} | Val shots: {len(val_shots)} | Test shots: {len(test_shots)}")
    
    # Split data by shots
    train_mask = np.isin(shots_filtered, train_shots)
    val_mask = np.isin(shots_filtered, val_shots)
    test_mask = np.isin(shots_filtered, test_shots)
    
    X_train = X_filtered[train_mask]
    y_train = y_filtered[train_mask]
    shots_train = shots_filtered[train_mask]
    
    X_val = X_filtered[val_mask]
    y_val = y_filtered[val_mask]
    shots_val = shots_filtered[val_mask]
    
    X_test = X_filtered[test_mask]
    y_test = y_filtered[test_mask]
    shots_test = shots_filtered[test_mask]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Advanced scaling with RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'ensemble_scaler.pkl')
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, 
            shots_train, shots_val, shots_test, scaler)

def train_ensemble_model(X_train, y_train, X_val, y_val, shots_train, shots_val, scaler):
    """Train the ensemble model with hyperparameter optimization"""
    print("=== Training Ensemble Model ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = AdvancedPlasmaDataset(X_train, y_train, shots_train, window_size=150, augment=True)
    val_dataset = AdvancedPlasmaDataset(X_val, y_val, shots_val, window_size=150, augment=False)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    
    # Initialize ensemble
    n_features = X_train.shape[1]
    ensemble = EnsembleModel(n_features, n_classes=2, device=device)
    
    # Train Random Forest first (on flattened data)
    print("\n" + "="*50)
    print("🔄 Training Random Forest...")
    print("="*50)
    
    rf_X_train = X_train.reshape(X_train.shape[0], -1)
    rf_X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Handle any remaining NaN values for Random Forest
    print("Handling NaN values for Random Forest...")
    rf_X_train = np.nan_to_num(rf_X_train, nan=0.0, posinf=0.0, neginf=0.0)
    rf_X_val = np.nan_to_num(rf_X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Random Forest input shapes: Train {rf_X_train.shape}, Val {rf_X_val.shape}")
    print(f"NaN check - Train: {np.isnan(rf_X_train).sum()}, Val: {np.isnan(rf_X_val).sum()}")
    
    ensemble.rf.fit(rf_X_train, y_train)
    rf_val_pred = ensemble.rf.predict(rf_X_val)
    rf_accuracy = accuracy_score(y_val, rf_val_pred)
    print(f"✅ Random Forest validation accuracy: {rf_accuracy:.4f}")
    
    # Train deep learning models
    models = [ensemble.cnn, ensemble.tcn, ensemble.birnn]
    model_names = ['CNN', 'TCN', 'BiRNN']
    
    for model_idx, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n" + "="*50)
        print(f"🔄 Training {name} ({model_idx + 1}/{len(models)})...")
        print("="*50)
        
        # Loss function with class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = AdvancedFocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
        
        # Optimizer with different learning rates for different components
        optimizer = optim.AdamW([
            {'params': model.parameters(), 'lr': 0.001},
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Training loop - reduced epochs for quick testing
        best_val_acc = 0
        patience_counter = 0
        max_patience = 5  # Reduced patience for quick testing
        
        # Progress bar for epochs - reduced to 5 epochs
        epoch_pbar = tqdm(range(5), desc=f"Training {name}", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Progress bar for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False)
            for batch_features, batch_labels in train_pbar:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
                
                # Update training progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_correct/train_total:.4f}'
                })
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False)
            with torch.no_grad():
                for batch_features, batch_labels in val_pbar:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
                    
                    # Update validation progress bar
                    val_pbar.set_postfix({
                        'Acc': f'{val_correct/val_total:.4f}'
                    })
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step(val_acc)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Acc': f'{train_acc:.4f}',
                'Val Acc': f'{val_acc:.4f}',
                'Best Val': f'{best_val_acc:.4f}',
                'Patience': f'{patience_counter}/{max_patience}'
            })
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_ensemble_{name.lower()}.pth')
                epoch_pbar.set_postfix({
                    'Train Acc': f'{train_acc:.4f}',
                    'Val Acc': f'{val_acc:.4f}',
                    'Best Val': f'{best_val_acc:.4f} ✨',
                    'Patience': f'{patience_counter}/{max_patience}'
                })
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"\n🛑 Early stopping for {name} at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_ensemble_{name.lower()}.pth'))
        print(f"✅ {name} best validation accuracy: {best_val_acc:.4f}")
        
        # Store model performance for comparison
        model_performances[name] = best_val_acc
    
    # Show individual model performance summary
    print(f"\n" + "="*50)
    print("📊 Individual Model Performance Summary")
    print("="*50)
    for name, acc in model_performances.items():
        print(f"{name:>6}: {acc:.4f}")
    print(f"Random Forest: {rf_accuracy:.4f}")
    
    # Optimize ensemble weights
    print(f"\n" + "="*50)
    print("🔄 Optimizing ensemble weights...")
    print("="*50)
    
    best_weights = optimize_ensemble_weights(ensemble, X_val, y_val, shots_val, device)
    ensemble.ensemble_weights = best_weights
    
    print(f"✅ Final ensemble weights: {best_weights}")
    
    # Save ensemble model
    print("\n💾 Saving ensemble model...")
    torch.save({
        'cnn_state_dict': ensemble.cnn.state_dict(),
        'tcn_state_dict': ensemble.tcn.state_dict(),
        'birnn_state_dict': ensemble.birnn.state_dict(),
        'ensemble_weights': ensemble.ensemble_weights,
        'scaler': scaler
    }, 'best_ensemble_model.pth')
    
    print("✅ Ensemble model saved successfully!")
    
    return ensemble

def optimize_ensemble_weights(ensemble, X_val, y_val, shots_val, device):
    """Optimize ensemble weights using validation data"""
    
    def objective(trial):
        weights = [
            trial.suggest_float('w1', 0.0, 1.0),
            trial.suggest_float('w2', 0.0, 1.0),
            trial.suggest_float('w3', 0.0, 1.0),
            trial.suggest_float('w4', 0.0, 1.0)
        ]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Set weights
        ensemble.ensemble_weights = weights
        
        # Evaluate
        val_dataset = AdvancedPlasmaDataset(X_val, y_val, shots_val, window_size=150, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        ensemble.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                # Get ensemble predictions
                outputs = ensemble.forward(batch_features)
                preds = np.argmax(outputs, axis=1)
                
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy
    
    # Run optimization with progress bar - reduced trials for quick testing
    print("Running Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    
    # Create progress bar for trials - reduced to 10 trials
    with tqdm(total=10, desc="Optimizing weights", unit="trial") as pbar:
        def objective_with_progress(trial):
            result = objective(trial)
            pbar.update(1)
            pbar.set_postfix({'Best Acc': f'{study.best_value:.4f}' if study.best_value else 'N/A'})
            return result
        
        study.optimize(objective_with_progress, n_trials=10)
    
    best_weights = [
        study.best_params['w1'],
        study.best_params['w2'],
        study.best_params['w3'],
        study.best_params['w4']
    ]
    
    best_weights = np.array(best_weights)
    best_weights = best_weights / best_weights.sum()
    
    print(f"Best validation accuracy: {study.best_value:.4f}")
    
    return best_weights

def evaluate_ensemble(ensemble, X_test, y_test, shots_test, scaler):
    """Evaluate the ensemble model on test data"""
    print("=== Ensemble Evaluation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare test dataset
    test_dataset = AdvancedPlasmaDataset(X_test, y_test, shots_test, window_size=150, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    # Evaluation with progress bar
    ensemble.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(test_loader, desc="Evaluating ensemble", unit="batch"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Get ensemble predictions
            outputs = ensemble.forward(batch_features)
            probs = outputs
            preds = np.argmax(outputs, axis=1)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Convert to binary classification
    binary_labels = np.array([0 if label in [1, 2, 3] else 1 for label in all_labels])
    binary_preds = np.array([0 if label in [1, 2, 3] else 1 for label in all_preds])
    binary_probs = all_probs[:, 1]  # Probability of ELMing class
    
    # Metrics
    accuracy = accuracy_score(binary_labels, binary_preds)
    precision = precision_score(binary_labels, binary_preds)
    recall = recall_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds)
    auc = roc_auc_score(binary_labels, binary_probs)
    
    print(f"\nTest Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(binary_labels, binary_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Suppressed', 'ELMing'],
                yticklabels=['Suppressed', 'ELMing'])
    plt.title('Ensemble Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(binary_labels, binary_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Ensemble (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble Model ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ensemble_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(binary_labels, binary_preds, 
                              target_names=['Suppressed', 'ELMing']))
    
    return accuracy, precision, recall, f1, auc

def main():
    """Main execution function"""
    print("=== High Accuracy ELM Ensemble Model ===")
    print("This script creates an ensemble of CNN, TCN, BiRNN, and Random Forest")
    print("for extremely high accuracy in detecting ELM states.")
    
    # Overall progress tracking
    print("\n" + "="*60)
    print("🚀 Starting ELM Ensemble Training Pipeline")
    print("="*60)
    
    # Load and preprocess data
    print("\n📊 Step 1/4: Loading and preprocessing data...")
    df = load_and_preprocess_data()
    selected_features = select_features(df)
    
    # Prepare data
    print("\n🔧 Step 2/4: Preparing data and engineering features...")
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     shots_train, shots_val, shots_test, scaler) = prepare_data(df, selected_features)
    
    # Train ensemble model
    print("\n🧠 Step 3/4: Training ensemble model...")
    ensemble = train_ensemble_model(X_train, y_train, X_val, y_val, shots_train, shots_val, scaler)
    
    # Evaluate ensemble
    print("\n📈 Step 4/4: Evaluating ensemble model...")
    accuracy, precision, recall, f1, auc = evaluate_ensemble(ensemble, X_test, y_test, shots_test, scaler)
    
    print(f"\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"Ensemble Model achieved {accuracy:.4f} accuracy on test set!")
    print(f"Model saved as 'best_ensemble_model.pth'")
    print(f"Scaler saved as 'ensemble_scaler.pkl'")
    
    # Save final metrics
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'ensemble_weights': ensemble.ensemble_weights.tolist()
    }
    
    import json
    with open('ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to 'ensemble_results.json'")
    print(f"\n🎯 Final Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

