import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
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
        return focal.mean()

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

class PlasmaDataset(Dataset):
    def __init__(self, features, labels, shots, window_size=1000):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.shots = shots
        self.window_size = window_size
        self.windows, self.window_labels = self._create_windows()
        label_mapping = {1: 0, 2: 0, 3: 0, 4: 1}
        self.window_labels = np.array([label_mapping[int(label)] for label in self.window_labels])
        print(f"Created {len(self.windows)} windows | Label distribution: {Counter(self.window_labels)}")
    
    def _create_windows(self):
        windows, window_labels = [], []
        half = self.window_size // 2
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
                center_label = window_labels_slice[half]
                windows.append(window_data)
                window_labels.append(center_label)
        return np.array(windows, dtype=np.float32), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]).T, torch.LongTensor([self.window_labels[idx]]).squeeze()

class PlasmaBinaryCNNBiLSTMAttention(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv1d(n_features, 8, kernel_size=7, stride=10, padding=3), nn.ReLU(), nn.BatchNorm1d(8), nn.Dropout(0.3),
            nn.Conv1d(8, 16, kernel_size=5, stride=10, padding=2), nn.ReLU(), nn.BatchNorm1d(16), nn.Dropout(0.3)
        )
        self.bilstm = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = AttentionModule(16)
        self.classifier = nn.Sequential(
            nn.Linear(16, n_classes)
        )
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        x = self.feature_conv(x).permute(0, 2, 1)
        x, _ = self.bilstm(x)
        return self.classifier(self.attention(x))

def load_and_prepare_data():
    print("Loading and preparing data...")
    df = pd.read_csv('plasma_data.csv')
    df = df[df['shot'] != 191675].copy()
    
    # Feature selection
    columns_to_remove = ['ELMPMARKER', 'zeff', 'rotation_edge', 'rotation_core']
    important_features = ['n_eped', 'li', 'q95', 'bt0', 'kappa', 'tribot', 'tritop', 'dR_sep', 'fs04_max_smoothed', 'iln3iamp', 'iun3iamp']
    selected_features = [f for f in important_features if f in df.columns]
    
    # Data preparation
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()
    
    # Shot-based split
    unique_shots = df_filtered['shot'].unique()
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    train_count, val_count = int(0.55 * len(unique_shots)), int(0.30 * len(unique_shots))
    train_shots = shuffled_shots[:train_count]
    val_shots = shuffled_shots[train_count:train_count + val_count]
    test_shots = shuffled_shots[train_count + val_count:]
    
    # Split data
    train_df = df_filtered[df_filtered['shot'].isin(train_shots)]
    val_df = df_filtered[df_filtered['shot'].isin(val_shots)]
    test_df = df_filtered[df_filtered['shot'].isin(test_shots)]
    
    # Prepare features without engineering
    X_train = train_df[selected_features].values
    y_train = train_df['state'].values
    shots_train = train_df['shot'].values
    
    X_val = val_df[selected_features].values
    y_val = val_df['state'].values
    shots_val = val_df['shot'].values
    
    X_test = test_df[selected_features].values
    y_test = test_df['state'].values
    shots_test = test_df['shot'].values

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, shots_train, shots_val, shots_test

def create_data_loaders(X_train, y_train, shots_train, X_val, y_val, shots_val, X_test, y_test, shots_test, window_size=150):
    train_dataset = PlasmaDataset(X_train, y_train, shots_train, window_size)
    val_dataset = PlasmaDataset(X_val, y_val, shots_val, window_size)
    test_dataset = PlasmaDataset(X_test, y_test, shots_test, window_size)
    
    # Weighted sampler for training
    class_counts = Counter(train_dataset.window_labels)
    class_weights_list = [1.0 / max(1, class_counts.get(c, 0)) for c in range(2)]
    sample_weights = np.array([class_weights_list[label] for label in train_dataset.window_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_and_evaluate(train_loader, val_loader, test_loader, n_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    model = PlasmaBinaryCNNBiLSTMAttention(n_features=n_features, n_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Class weights
    all_train_labels = train_loader.dataset.window_labels
    classes_present = np.unique(all_train_labels)
    class_weights = compute_class_weight('balanced', classes=classes_present, y=all_train_labels)
    # Ensure weights align to class indices [0,1]
    full_weights = np.zeros(2, dtype=np.float32)
    for cls, w in zip(classes_present, class_weights):
        full_weights[int(cls)] = w
    class_weights_t = torch.tensor(full_weights, dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=class_weights_t, gamma=2.0)
    
    # Training
    n_epochs, patience = 15, 5
    best_val_f1, patience_counter = 0.0, 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, val_predictions, val_labels, val_probabilities = 0.0, [], [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
                val_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_plasma_binary_cnn_centerlabel.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_plasma_binary_cnn_centerlabel.pth'))
    model.eval()
    
    # Threshold tuning on validation
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            val_true.extend(batch_labels.numpy())
    
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (np.array(val_probs) >= thr).astype(int)
        f1 = f1_score(val_true, preds, average='weighted')
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    
    # Test evaluation
    all_predictions, all_labels, all_probabilities = [], [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = (probabilities >= best_thr).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Results
    print(f"\nThreshold: {best_thr:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Suppressed', 'ELMing']))
    
    auc_score = roc_auc_score(all_labels, all_probabilities)
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Plots
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Center Label)')
    plt.legend()
    plt.savefig('roc_curve_centerlabel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Suppressed', 'ELMing'], yticklabels=['Suppressed', 'ELMing'])
    plt.title('Normalized Confusion Matrix (Center Label)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_binary_centerlabel.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Plasma Binary State Classification CNN (Center Label) ===")
    X_train, X_val, X_test, y_train, y_val, y_test, shots_train, shots_val, shots_test = load_and_prepare_data()
    train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, shots_train, X_val, y_val, shots_val, X_test, y_test, shots_test)
    train_and_evaluate(train_loader, val_loader, test_loader, n_features=X_train.shape[1])
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()


