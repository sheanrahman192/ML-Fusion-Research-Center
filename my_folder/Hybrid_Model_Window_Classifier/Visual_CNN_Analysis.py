import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# cuDNN fast paths (safe for inference)
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

STATE_NAMES = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']


def load_and_preprocess_data():
    print('=== Loading Plasma Dataset from CSV ===')
    df = pd.read_csv('plasma_data.csv')
    print(f"Original dataset shape: {df.shape}")
    df = df[df['shot'] != 191675].copy()
    print(f"After removing shot 191675: {df.shape}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def select_features(df):
    columns_to_remove = ['ELMPMARKER', 'zeff', 'rotation_edge', 'rotation_core']
    important_features = [
        'n_eped', 'n_e', 'betan', 'li', 'q95', 'Ip',
        'bt0', 'kappa', 'tribot', 'tritop', 'dR_sep', 'fs04', 'fs04_max_avg', 'fs04_max_smoothed'
    ]
    available_features = [col for col in df.columns if col not in columns_to_remove]
    selected = [f for f in important_features if f in available_features]
    print(f"Selected {len(selected)} features: {selected}")
    return selected


def prepare_data(df, selected_features):
    print('=== Preparing data with 70/10/20 shot-based split (seed=42) ===')
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()

    unique_shots = df_filtered['shot'].unique()
    np.random.seed(42)
    shuffled = np.random.permutation(unique_shots)
    n = len(shuffled)
    n_train = int(np.floor(0.70 * n))
    n_val = int(np.floor(0.10 * n))
    train_shots = shuffled[:n_train]
    val_shots = shuffled[n_train:n_train + n_val]
    test_shots = shuffled[n_train + n_val:]

    train_df = df_filtered[df_filtered['shot'].isin(train_shots)]
    test_df = df_filtered[df_filtered['shot'].isin(test_shots)]

    X_train = train_df[selected_features].values
    y_train = train_df['state'].values
    X_test = test_df[selected_features].values
    y_test = test_df['state'].values

    # Feature engineering identical to training
    X_train_diff = np.diff(X_train, axis=0, prepend=X_train[0:1])
    X_train_ma = np.convolve(X_train.flatten(), np.ones(5) / 5, mode='same').reshape(X_train.shape)
    X_train_fg = np.concatenate([X_train, X_train_diff, X_train_ma], axis=1)

    X_test_diff = np.diff(X_test, axis=0, prepend=X_test[0:1])
    X_test_ma = np.convolve(X_test.flatten(), np.ones(5) / 5, mode='same').reshape(X_test.shape)
    X_test_fg = np.concatenate([X_test, X_test_diff, X_test_ma], axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fg)
    X_test_scaled = scaler.transform(X_test_fg)

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Feature dim after engineering: {X_test_scaled.shape[1]}")
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


def compute_feature_medians_and_range(df, selected_features):
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()
    medians = df_filtered[selected_features].median()
    n_min = df_filtered['n_eped'].min() if 'n_eped' in df_filtered.columns else None
    n_max = df_filtered['n_eped'].max() if 'n_eped' in df_filtered.columns else None
    return medians, n_min, n_max


def build_augmented_window_from_profile(profile_vector, window_size, selected_features):
    # profile_vector: pd.Series or np.array of base selected features in training order
    base = np.array([profile_vector[f] for f in selected_features], dtype=np.float32)
    # Repeat across time
    base_window = np.tile(base, (window_size, 1))  # (T, F)
    diff_window = np.zeros_like(base_window)       # constant profile → diffs are zero
    ma_window = base_window.copy()                 # moving average equals constant profile
    augmented = np.concatenate([base_window, diff_window, ma_window], axis=1)  # (T, 3F)
    return augmented


 


def plot_probabilities_vs_feature(model, device, scaler, df, selected_features, feature_name, window_size=150, num_points=120, threshold=0.5, out_dir='cnn_pdp'):
    if feature_name not in selected_features:
        return False
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()
    if feature_name not in df_filtered.columns:
        return False
    medians = df_filtered[selected_features].median()
    f_min = df_filtered[feature_name].min()
    f_max = df_filtered[feature_name].max()
    if pd.isna(f_min) or pd.isna(f_max):
        return False
    feature_values = np.linspace(f_min, f_max, num_points)

    probs = []
    model.eval()
    with torch.no_grad():
        for v in feature_values:
            prof = medians.copy()
            prof[feature_name] = v
            augmented = build_augmented_window_from_profile(prof, window_size, selected_features)
            augmented_scaled = scaler.transform(augmented)
            tensor = torch.from_numpy(augmented_scaled.astype(np.float32)).T.unsqueeze(0).to(device)
            logits = model(tensor)
            prob = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            probs.append(prob)
    probs = np.stack(probs, axis=0)  # (N, C)
    max_change = float(np.max(np.ptp(probs, axis=0)))
    if max_change <= threshold:
        return False

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for c in range(probs.shape[1]):
        plt.plot(feature_values, probs[:, c], label=STATE_NAMES[c], linewidth=2)
    plt.xlabel(feature_name, fontsize=14)
    plt.ylabel('Predicted Probability', fontsize=14)
    plt.title(f'CNN Probabilities vs {feature_name} (others at median)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"cnn_pdp_{feature_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path} (max class Δ={max_change:.3f})")
    return True


class PlasmaDataset(Dataset):
    def __init__(self, features, labels, window_size=150):
        self.window_size = window_size
        self.windows = []
        self.window_labels = []
        for i in range(len(features) - window_size + 1):
            window_data = features[i:i + window_size]
            window_label = Counter(labels[i:i + window_size]).most_common(1)[0][0]
            label_counts = Counter(labels[i:i + window_size])
            if (max(label_counts.values()) >= window_size * 0.7) and (window_label != 0):
                self.windows.append(window_data)
                self.window_labels.append(window_label)
        self.windows = np.array(self.windows)
        # Map states: 1->0, 2->1, 3->2, 4->3
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        self.window_labels = np.array([label_mapping[l] for l in self.window_labels])
        print(f"Created {len(self.windows)} windows (size {window_size}) | Label dist: {Counter(self.window_labels)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.windows[idx]).T  # (F, T)
        label = torch.LongTensor([self.window_labels[idx]])
        return features, label.squeeze()


class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        att_w = torch.softmax(self.attention(x), dim=1)
        return torch.sum(att_w * x, dim=1)


class PlasmaCNNBiLSTMAttention(nn.Module):
    def __init__(self, n_features, n_classes=4):
        super().__init__()
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
            nn.Dropout(0.4),
        )
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = AttentionModule(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.feature_conv(x)         # (B, 256, T)
        x = x.permute(0, 2, 1)          # (B, T, 256)
        x, _ = self.bilstm(x)           # (B, T, 256)
        x = self.attention(x)           # (B, 256)
        x = self.classifier(x)          # (B, C)
        return x


def build_test_loader(X_test, y_test, window_size=150, batch_size=2048):
    dataset = PlasmaDataset(X_test, y_test, window_size)
    use_cuda = torch.cuda.is_available()
    num_workers = min(8, os.cpu_count() or 4)
    if num_workers > 0:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=True, prefetch_factor=4,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=use_cuda,
        )
    return loader, dataset


def load_cnn(n_features, weights_path='best_plasma_cnn.pth', n_classes=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlasmaCNNBiLSTMAttention(n_features=n_features, n_classes=n_classes).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file '{weights_path}' not found. Train the CNN first.")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def evaluate(model, device, test_loader):
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    avg_loss = test_loss / max(1, len(test_loader))
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    report = classification_report(all_labels, all_preds, target_names=STATE_NAMES)
    return avg_loss, acc, cm, report


def plot_confusion_matrix(cm, acc, out_path='cnn_confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=STATE_NAMES, yticklabels=STATE_NAMES)
    plt.title(f'CNN Normalized Confusion Matrix - Accuracy: {acc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


    


def main():
    print('=== Visual CNN Analysis ===')
    df = load_and_preprocess_data()
    selected = select_features(df)
    X_train, y_train, X_test, y_test, scaler = prepare_data(df, selected)
    n_features = X_test.shape[1]

    test_loader, test_dataset = build_test_loader(X_test, y_test, window_size=150, batch_size=2048)
    if len(test_dataset) == 0:
        raise RuntimeError('No valid test windows after filtering. Try adjusting windowing or data filters.')

    model, device = load_cnn(n_features=n_features, weights_path='best_plasma_cnn.pth', n_classes=4)
    avg_loss, acc, cm, report = evaluate(model, device, test_loader)

    print('\nClassification Report:')
    print(report)
    print(f"Avg test loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    plot_confusion_matrix(cm, acc, out_path='cnn_confusion_matrix.png')
    # Generalized PDPs: save only if any class Δprob > 0.3
    saved_count = 0
    for feat in selected:
        did_save = plot_probabilities_vs_feature(model, device, scaler, df, selected, feat, window_size=150, num_points=120, threshold=0.3, out_dir='cnn_pdp')
        if did_save:
            saved_count += 1
    print(f"PDPs saved: {saved_count}")


if __name__ == '__main__':
    main()


