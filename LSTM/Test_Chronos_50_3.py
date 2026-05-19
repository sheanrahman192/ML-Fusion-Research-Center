"""
Test_Chronos_50_3.py: Same task as Test_LSTM_50_3 using the Chronos foundation model (Chronos-2).

Predicts the CURRENT time's state (3-state: Suppressed, Dithering/Mitigated, ELMing) using:
- Same inputs: plasma_data.csv, 200 ms window, Branch 1 = first 150 ms of 5 features,
  Branch 2 = full 200 ms of 9 features.
- Chronos-2 (amazon/chronos-2) embeds each *multivariate* branch window in one pass;
  the last-step embedding is pooled and concatenated, then a small MLP classifier
  is trained on top (Chronos-2 frozen).

Requires: pip install chronos-forecasting
        (or: pip install "chronos-forecasting>=2.0" / install from GitHub if no wheel)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import functools
import warnings

warnings.filterwarnings("ignore")

# Force unbuffered output so progress prints show up immediately
print = functools.partial(print, flush=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Window configuration (same as Test_LSTM_50_3)
WINDOW_MS = 200
BRANCH1_MS = 150
WINDOW_SIZE = WINDOW_MS
BRANCH1_LEN = BRANCH1_MS
N_CLASSES = 3

# Chronos-2 supports multivariate inputs + long contexts (CPU ok, GPU optional)
CHRONOS_MODEL_ID = "amazon/chronos-2"
EMBED_BATCH_SIZE = 16

# Data path (same as Test_LSTM_50_3)
DATA_PATH = "/mnt/homes/sr4240/my_folder/plasma_data.csv"


def load_and_prepare_data():
    """
    Same as Test_LSTM_50_3: load plasma_data.csv, drop shot 191675,
    Branch 1 (5 features), Branch 2 (9 features), scale and clean.
    """
    print("Loading data (same as Test_LSTM_50_3)...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["shot"] != 191675].copy()

    branch1_features = [
        "iln3iamp",
        "betan",
        "density",
        "li",
        "fs_sum_past_max_smoothed",
    ]
    branch2_features = [
        "pinj",
        "tijnj",
        "echpwrc",
        "I_ECCD",
        "tritop",
        "tribot",
        "Ip",
        "bt",
        "gasa",
    ]

    missing1 = [f for f in branch1_features if f not in df.columns]
    missing2 = [f for f in branch2_features if f not in df.columns]
    if missing1 or missing2:
        raise ValueError(
            f"Missing columns. Branch1: {missing1}, Branch2: {missing2}"
        )

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)
    X1 = df_sorted[branch1_features].values
    X2 = df_sorted[branch2_features].values
    y = df_sorted["state"].values
    times = df_sorted["time"].values
    shots = df_sorted["shot"].values

    valid_mask = (
        ~np.isnan(X1).any(axis=1)
        & ~np.isnan(y)
        & ~np.isnan(times)
    )
    X1 = X1[valid_mask]
    X2 = X2[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    X2_imputed = X2.copy()
    for j in range(X2_imputed.shape[1]):
        col = X2_imputed[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[nan_mask] = 0.0
        elif nan_mask.any():
            col[nan_mask] = np.nanmean(col)
        X2_imputed[:, j] = col
    X2 = X2_imputed

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X1 = scaler1.fit_transform(X1)
    X2 = scaler2.fit_transform(X2)

    return (
        X1, X2, y, times, shots,
        branch1_features, branch2_features,
        scaler1, scaler2,
    )


def create_dual_branch_windows_by_shot(
    X1, X2, y, times, shots,
    window_size: int = WINDOW_SIZE,
    branch1_len: int = BRANCH1_LEN,
):
    """Same as Test_LSTM_50_3: dual-branch windows, label at current time, 70/15/15 by shot."""
    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)
    train_shots = set(shuffled_shots[:train_size])
    val_shots = set(shuffled_shots[train_size : train_size + val_size])
    test_shots = set(shuffled_shots[train_size + val_size :])

    label_mapping = {0: 0, 1: 1, 2: 1, 3: 2}
    valid_raw_labels = {0, 1, 2, 3}

    train_x1, train_x2, train_y = [], [], []
    val_x1, val_x2, val_y = [], [], []
    test_x1, test_x2, test_y = [], [], []

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        if len(shot_indices) < window_size:
            continue

        if shot_id in train_shots:
            tx1, tx2, ty = train_x1, train_x2, train_y
        elif shot_id in val_shots:
            tx1, tx2, ty = val_x1, val_x2, val_y
        else:
            tx1, tx2, ty = test_x1, test_x2, test_y

        shot_X1 = X1[shot_indices]
        shot_X2 = X2[shot_indices]
        shot_y = y[shot_indices]

        for end_idx in range(window_size - 1, len(shot_indices)):
            start_idx = end_idx - window_size + 1
            window1 = shot_X1[start_idx : end_idx + 1][:branch1_len]
            window2 = shot_X2[start_idx : end_idx + 1]
            raw_label = int(shot_y[end_idx])
            if raw_label not in valid_raw_labels:
                continue
            mapped_label = label_mapping[raw_label]
            tx1.append(window1)
            tx2.append(window2)
            ty.append(mapped_label)

    train_x1 = np.array(train_x1, dtype=np.float32)
    train_x2 = np.array(train_x2, dtype=np.float32)
    train_y = np.array(train_y)
    val_x1 = np.array(val_x1, dtype=np.float32)
    val_x2 = np.array(val_x2, dtype=np.float32)
    val_y = np.array(val_y)
    test_x1 = np.array(test_x1, dtype=np.float32)
    test_x2 = np.array(test_x2, dtype=np.float32)
    test_y = np.array(test_y)

    print(f"Windows: Train={len(train_x1)}, Val={len(val_x1)}, Test={len(test_x1)}")
    return (
        train_x1, train_x2, train_y,
        val_x1, val_x2, val_y,
        test_x1, test_x2, test_y,
    )


def compute_chronos2_embeddings(pipeline, windows, context_length: int):
    """
    windows: (N, T, F) numpy float32 (multivariate window)
    Returns: (N, embed_dim) numpy (one embedding per window)
    """
    N = windows.shape[0]
    pooled = []

    # Process in chunks so we never materialize embeddings for all N windows at once.
    for start in range(0, N, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, N)
        # Chronos-2 expects (batch, n_variates, history_length)
        # Our windows are (batch, T, F) so we transpose to (batch, F, T).
        batch = np.transpose(windows[start:end], (0, 2, 1))
        with torch.no_grad():
            emb_list, _meta = pipeline.embed(
                batch,
                batch_size=end - start,
                context_length=context_length,
            )
        for e in emb_list:
            # e: (n_variates, num_tokens, d_model)
            # Pool across variates then take the [REG] token embedding (index 0).
            pooled.append(e.mean(dim=0)[0].cpu().numpy())

        if (end == N) or (start % (EMBED_BATCH_SIZE * 5) == 0):
            print(f"  Embedded {end:,}/{N:,} windows (context_length={context_length})")

    return np.stack(pooled, axis=0)


class ChronosClassifier(nn.Module):
    """MLP on top of concatenated Chronos embeddings (branch1_emb | branch2_emb) -> 3 classes."""

    def __init__(self, input_dim: int, n_classes: int = N_CLASSES, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.4),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"ChronosClassifier parameters: {total:,}")

    def forward(self, x):
        return self.mlp(x)


class EmbeddingDataset(Dataset):
    """Dataset of precomputed (emb_branch1 | emb_branch2) and labels."""

    def __init__(self, emb_branch1, emb_branch2, labels):
        self.X = np.concatenate([emb_branch1, emb_branch2], axis=1).astype(np.float32)
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def train_model(model, train_loader, val_loader, device, n_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_train_preds.append(preds.cpu().numpy())
            all_train_labels.append(batch_y.cpu().numpy())

        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)

        model.eval()
        epoch_val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(batch_y.cpu().numpy())

        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_labels = np.concatenate(all_val_labels, axis=0)

        train_acc = float(accuracy_score(all_train_labels, all_train_preds))
        val_acc = float(accuracy_score(all_val_labels, all_val_preds))
        avg_train_loss = epoch_train_loss / max(len(train_loader), 1)
        avg_val_loss = epoch_val_loss / max(len(val_loader), 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{n_epochs}  Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.4f}  Val Loss: {avg_val_loss:.4f}  Val Acc: {val_acc:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_test_chronos_50_3.pth")
            patience_counter = 0
            print("  ✓ New best model saved")
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())
            all_probs.append(probs.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    print("\nClassification Report (3-state):")
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=[0, 1, 2], digits=4))
    print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nROC AUC Scores:")
    for i, name in enumerate(class_names):
        if i < all_probs.shape[1]:
            binary = (all_labels == i).astype(int)
            if len(np.unique(binary)) > 1:
                print(f"  {name}: {roc_auc_score(binary, all_probs[:, i]):.4f}")
    return all_preds, all_labels, all_probs


def plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].plot(train_losses, label="Train Loss", color="blue")
    axes[0, 0].plot(val_losses, label="Val Loss", color="red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Chronos + Classifier: Loss (3-state)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label="Train Acc", color="blue")
    axes[0, 1].plot(val_accs, label="Val Acc", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Chronos + Classifier: Accuracy (3-state)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2], normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title("Normalized Confusion Matrix (3-state)")
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")

    cm_counts = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    sns.heatmap(cm_counts, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title("Confusion Matrix Counts (3-state)")
    axes[1, 1].set_ylabel("True Label")
    axes[1, 1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig("test_chronos_50_3_results.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Results saved to 'test_chronos_50_3_results.png'")


def main():
    print("=" * 60)
    print("Chronos foundation model for Plasma State Classification (3-state)")
    print("Same task & inputs as Test_LSTM_50_3")
    print("=" * 60)
    print(f"Window: {WINDOW_MS} ms, Branch1: first {BRANCH1_MS} ms of 5 features, Branch2: full 200 ms of 9 features")
    print("Chronos model:", CHRONOS_MODEL_ID)
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data and create windows (same as Test_LSTM_50_3)
    (
        X1, X2, y, times, shots,
        branch1_features, branch2_features,
        scaler1, scaler2,
    ) = load_and_prepare_data()

    (
        train_x1, train_x2, train_y,
        val_x1, val_x2, val_y,
        test_x1, test_x2, test_y,
    ) = create_dual_branch_windows_by_shot(X1, X2, y, times, shots)

    # Load Chronos and precompute embeddings
    try:
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise ImportError(
            "Chronos not found. Install with: pip install chronos-forecasting "
            "(or pip install 'chronos-forecasting>=2.0'). If no wheel for your platform, "
            "try: pip install git+https://github.com/amazon-science/chronos-forecasting.git"
        ) from e

    print("\nLoading Chronos pipeline...")
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        dtype=torch.float32,
        device_map="cuda" if device.type == "cuda" else "cpu",
    )

    print("Computing Chronos-2 embeddings for Branch 1 (train/val/test)...")
    train_emb1 = compute_chronos2_embeddings(pipeline, train_x1, context_length=BRANCH1_LEN)
    val_emb1 = compute_chronos2_embeddings(pipeline, val_x1, context_length=BRANCH1_LEN)
    test_emb1 = compute_chronos2_embeddings(pipeline, test_x1, context_length=BRANCH1_LEN)
    print("Computing Chronos-2 embeddings for Branch 2 (train/val/test)...")
    train_emb2 = compute_chronos2_embeddings(pipeline, train_x2, context_length=WINDOW_SIZE)
    val_emb2 = compute_chronos2_embeddings(pipeline, val_x2, context_length=WINDOW_SIZE)
    test_emb2 = compute_chronos2_embeddings(pipeline, test_x2, context_length=WINDOW_SIZE)

    embed_dim = train_emb1.shape[1]
    input_dim = embed_dim + train_emb2.shape[1]
    print(f"Embedding dim: {embed_dim}, Classifier input dim: {input_dim}")

    # Datasets and loaders
    train_ds = EmbeddingDataset(train_emb1, train_emb2, train_y)
    val_ds = EmbeddingDataset(val_emb1, val_emb2, val_y)
    test_ds = EmbeddingDataset(test_emb1, test_emb2, test_y)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=0)

    model = ChronosClassifier(input_dim=input_dim, n_classes=N_CLASSES).to(device)

    print("\nTraining classifier on Chronos embeddings...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=50
    )

    print("\nLoading best checkpoint...")
    model.load_state_dict(torch.load("best_test_chronos_50_3.pth", map_location=device))

    class_names = ["Suppressed", "Dithering/Mitigated", "ELMing"]
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, class_names)
    plot_results(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names)

    print("\n" + "=" * 60)
    print(f"Chronos (current state, window={WINDOW_MS} ms) - Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
