#!/usr/bin/env python3

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("=== Simple Ensemble Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

class SimpleCNN(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

class SimpleDataset(Dataset):
    def __init__(self, features, labels, window_size=50):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.window_size = window_size
        self.windows, self.window_labels = self._create_windows()
        
        # Convert to binary classification
        label_mapping = {1: 0, 2: 0, 3: 0, 4: 1}
        self.window_labels = np.array([label_mapping[int(label)] for label in self.window_labels])
        
        print(f"Created {len(self.windows)} windows | Label distribution: {Counter(self.window_labels)}")
    
    def _create_windows(self):
        windows, window_labels = [], []
        
        for i in range(0, len(self.features) - self.window_size + 1, self.window_size // 2):
            window_data = self.features[i:i + self.window_size]
            window_labels_slice = self.labels[i:i + self.window_size]
            
            # Majority label
            label_counts = Counter(window_labels_slice)
            majority_label, count = label_counts.most_common(1)[0]
            
            if majority_label != 0:  # Skip N/A states
                windows.append(window_data)
                window_labels.append(majority_label)
        
        return np.array(windows, dtype=np.float32), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]).T, torch.LongTensor([self.window_labels[idx]]).squeeze()

def load_data():
    print("Loading data...")
    df = pd.read_csv('plasma_data.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Remove problematic shot
    if 'shot' in df.columns:
        df = df[df['shot'] != 191675].copy()
        print(f"After removing shot 191675: {df.shape}")
    
    return df

def prepare_simple_data(df):
    print("Preparing data...")
    
    # Select features
    features = ['n_eped', 'n_e', 'betan', 'li', 'q95', 'Ip', 'bt0', 'kappa', 'tribot', 'tritop', 'dR_sep']
    available_features = [f for f in features if f in df.columns]
    print(f"Using features: {available_features}")
    
    # Filter data
    df_filtered = df[df['state'] != 0].copy()
    X = df_filtered[available_features].values
    y = df_filtered['state'].values
    
    print(f"Data shape: {X.shape}")
    print(f"State distribution: {Counter(y)}")
    
    # Simple train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_simple_model(X_train, y_train, X_test, y_test):
    print("Training simple model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SimpleDataset(X_train, y_train, window_size=50)
    test_dataset = SimpleDataset(X_test, y_test, window_size=50)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model
    n_features = X_train.shape[1]
    model = SimpleCNN(n_features, n_classes=2).to(device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Acc = {train_acc:.4f}")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(test_loader, desc="Evaluating"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Suppressed', 'ELMing'],
                yticklabels=['Suppressed', 'ELMing'])
    plt.title('Simple Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('simple_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def main():
    print("Starting simple ensemble test...")
    
    # Load data
    df = load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_simple_data(df)
    
    # Train and evaluate
    accuracy = train_simple_model(X_train, y_train, X_test, y_test)
    
    print(f"\nFinal accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

