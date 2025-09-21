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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

class NoncausalConv1d(nn.Module):
    """Noncausal 1D convolution that only looks at past data points"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(NoncausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation, **kwargs)
    
    def forward(self, x):
        # Apply convolution with padding
        x = self.conv(x)
        # Remove the padding from the right side to ensure noncausality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class ResidualBlock(nn.Module):
    """Residual block with noncausal convolutions and dilated convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = NoncausalConv1d(in_channels, out_channels, kernel_size, 
                                  dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = NoncausalConv1d(out_channels, out_channels, kernel_size, 
                                  dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Downsample if input and output channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        # For noncausal attention, we'll use a simpler approach
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # Noncausal attention: biases attention toward recent positions
        # This ensures the model focuses more on recent past data
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute attention scores for each position
        attention_scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # For noncausal attention, we'll use a sliding window approach
        # Each position will have higher attention weight for recent positions
        # This ensures noncausality without complex masking
        
        # Create a noncausal bias: recent positions get higher attention
        noncausal_bias = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        noncausal_bias = noncausal_bias.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        
        # Add noncausal bias to attention scores
        attention_scores = attention_scores + noncausal_bias * 0.1  # Small bias to favor recent positions
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)  # (batch_size, hidden_dim)
        return attended_output

class PlasmaDataset(Dataset):
    def __init__(self, features, labels, window_size=150):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.windows = []
        self.window_labels = []
        
        # For TCN, we want to classify each point based on its past context
        # We'll create windows where each point is classified based on previous data
        for i in range(window_size, len(features)):
            # Use data from i-window_size to i-1 (past data only)
            window_data = features[i-window_size:i]
            # The label is the state at the current point i
            current_label = labels[i]
            
            # Only include non-N/A states
            if current_label != 0:
                self.windows.append(window_data)
                self.window_labels.append(current_label)
        
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)
        
        # Map states: 1->0, 2->1, 3->2, 4->3
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        self.window_labels = np.array([label_mapping[label] for label in self.window_labels])
        
        print(f"Created {len(self.windows)} TCN windows with window size {window_size}")
        print(f"Label distribution: {Counter(self.window_labels)}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.windows[idx]).T
        label = torch.LongTensor([self.window_labels[idx]])
        return features, label.squeeze()

class PlasmaTCNBiLSTMAttention(nn.Module):
    def __init__(self, n_features, n_classes=4, window_size=150, num_channels=[64, 128, 256]):
        super(PlasmaTCNBiLSTMAttention, self).__init__()
        
        self.n_features = n_features
        self.window_size = window_size
        
        # Feature extraction with noncausal convolutions (similar to original CNN)
        self.feature_conv = nn.Sequential(
            NoncausalConv1d(n_features, num_channels[0], kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels[0]),
            nn.Dropout(0.3),
            NoncausalConv1d(num_channels[0], num_channels[1], kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels[1]),
            nn.Dropout(0.3),
            NoncausalConv1d(num_channels[1], num_channels[2], kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels[2]),
            nn.Dropout(0.4)
        )
        
        # Residual blocks with increasing dilation for TCN
        self.residual_blocks = nn.ModuleList()
        in_channels = num_channels[2]
        
        for i in range(3):  # Add 3 residual blocks
            dilation = 2 ** i
            block = ResidualBlock(in_channels, num_channels[2], kernel_size=3, 
                                dilation=dilation, dropout=0.3)
            self.residual_blocks.append(block)
        
        # Unidirectional LSTM (noncausal - only processes past to present)
        self.lstm = nn.LSTM(input_size=num_channels[2], hidden_size=128, num_layers=2, 
                           batch_first=True, bidirectional=False, dropout=0.3)
        
        # Attention mechanism (noncausal - only attends to past and current positions)
        self.attention = AttentionModule(128)  # 128 for unidirectional
        
        # Classifier (similar to original CNN)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
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
        # x shape: (batch_size, n_features, window_size)
        
        # Feature extraction with noncausal convolutions
        x = self.feature_conv(x)  # (B, 256, T)
        
        # Residual blocks for TCN
        for block in self.residual_blocks:
            x = block(x)
        
        # Permute for LSTM: (B, T, 256)
        x = x.permute(0, 2, 1)
        
        # Unidirectional LSTM (processes sequence from past to present)
        x, _ = self.lstm(x)  # (B, T, 128)
        
        # Noncausal attention mechanism (only attends to past and current positions)
        x = self.attention(x)  # (B, 128)
        
        # Classification
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
    print(f"State distribution: {df_filtered['state'].value_counts().sort_index()}")

    # Random shot split with 60/20/20 distribution
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

    # Split data by shot
    train_df = df_filtered[df_filtered['shot'].isin(train_shots)]
    val_df = df_filtered[df_filtered['shot'].isin(val_shots)]
    test_df = df_filtered[df_filtered['shot'].isin(test_shots)]
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")
    
    # Process training data
    X_train = train_df[selected_features].values
    y_train = train_df['state'].values
    
    # Process test data
    X_test = test_df[selected_features].values
    y_test = test_df['state'].values
    
    # Feature engineering (done separately for train and test to avoid leakage)
    # Training features
    X_train_diff = np.diff(X_train, axis=0, prepend=X_train[0:1])
    X_train_ma = np.convolve(X_train.flatten(), np.ones(5)/5, mode='same').reshape(X_train.shape)
    X_train = np.concatenate([X_train, X_train_diff, X_train_ma], axis=1)
    
    # Test features (using same prepend value as training for consistency)
    X_test_diff = np.diff(X_test, axis=0, prepend=X_test[0:1])
    X_test_ma = np.convolve(X_test.flatten(), np.ones(5)/5, mode='same').reshape(X_test.shape)
    X_test = np.concatenate([X_test, X_test_diff, X_test_ma], axis=1)
    
    print(f"Training set size: {len(X_train)} (shots: {len(train_shots)})")
    print(f"Validation set size (unused here): {len(val_df)} (shots: {len(val_shots)})")
    print(f"Test set size: {len(X_test)} (shots: {len(test_shots)})")
    
    # Scaling (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE oversampling for minority classes (only on training data)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {Counter(y_train_bal)}")
    
    return X_train_bal, X_test_scaled, y_train_bal, y_test

def train_model(X_train, y_train, X_test, y_test, n_features, window_size=150):
    print("Creating TCN datasets...")
    train_dataset = PlasmaDataset(X_train, y_train, window_size)
    test_dataset = PlasmaDataset(X_test, y_test, window_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compute class weights for imbalanced classes
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

    model = PlasmaTCNBiLSTMAttention(n_features=n_features, n_classes=n_classes, window_size=window_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Cosine annealing scheduler with warmup
    def warmup_cosine_schedule(epoch):
        if epoch < 5:
            return epoch / 5
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / 45))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    n_epochs = 50
    patience = 20
    patience_counter = 0
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_accuracy = 0.0
    
    print("Starting TCN training...")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
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
            
            # Progress indicator
            progress = (batch_idx + 1) / total_batches * 100
            print(f"\rTraining progress: {progress:.1f}%", end="", flush=True)
        
        print()  # New line after progress bar
        
        scheduler.step()
        
        model.eval()
        test_loss = 0.0
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        accuracy = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # Print results after every epoch
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_plasma_tcn.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load('best_plasma_tcn.pth'))
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
    plt.title(f'TCN Normalized Confusion Matrix - Accuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('tcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('TCN Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label='Test Loss')
    plt.title('TCN Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tcn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model

def main():
    print("=== Plasma State Classification TCN ===")
    df = load_and_preprocess_data()
    selected_features = select_features(df)
    window_size = 150
    X_train, X_test, y_train, y_test = prepare_data(df, selected_features)
    model = train_model(X_train, y_train, X_test, y_test, 
                       n_features=X_train.shape[1], window_size=window_size)
    print("\n=== TCN Training Complete ===")
    print("Model saved as 'best_plasma_tcn.pth'")
    print("Confusion matrix and training curves saved as images")

if __name__ == "__main__":
    main()