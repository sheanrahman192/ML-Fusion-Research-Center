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
import warnings
import time

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Window configuration (time-based, assuming ~1 ms per sample)
WINDOW_MS = 200          # full window length
BRANCH1_MS = 150         # first 150 ms of the 200 ms window

# For now we treat 1 timestep ≈ 1 ms
WINDOW_SIZE = WINDOW_MS
BRANCH1_LEN = BRANCH1_MS

# 3-state classification: Suppressed, Dithering/Mitigated (combined), ELMing
N_CLASSES = 3


class SingleBranchLSTM(nn.Module):
    """
    Single LSTM model over all 14 features for the full 200 ms window.
    Branch 1 features (first 5) are zero-padded for the last 50 ms,
    Branch 2 features (last 9) span the full 200 ms.

    Input shape: (batch, 200, 14)
      - t=0..149:  all 14 features have real values
      - t=150..199: first 5 features are 0, last 9 have real values
    """

    def __init__(
        self,
        n_features: int = 14,
        lstm_hidden: int = 64,
        lstm_num_layers: int = 1,
        nn_hidden_sizes=None,
        classifier_hidden: int = 32,
        n_classes: int = N_CLASSES,
    ):
        super(SingleBranchLSTM, self).__init__()

        if nn_hidden_sizes is None:
            nn_hidden_sizes = [64, 32]

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.3 if lstm_num_layers > 1 else 0.0,
        )

        # Fully connected head
        layers = []
        in_dim = lstm_hidden
        for h in nn_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(0.4),
                ]
            )
            in_dim = h

        self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(classifier_hidden, n_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("Single-Branch LSTM Model Parameter Count (3-state):")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60)
        print("Architecture: Single LSTM (200 ms, 14 features, Branch1 zero-padded last 50 ms) → MLP → Classifier")

    def forward(self, x):
        """
        x: (batch, 200, 14)  - combined features, Branch1 zeroed for t=150..199
        """
        out, (h, c) = self.lstm(x)
        h_last = h[-1]  # (batch, lstm_hidden)

        features = self.mlp(h_last)
        logits = self.classifier(features)
        return logits


class CombinedDataset(Dataset):
    """Dataset for combined temporal windows (200 ms, 14 features)."""

    def __init__(self, windows, labels):
        self.x = np.ascontiguousarray(windows, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def load_and_prepare_data():
    """
    Load and preprocess plasma data for dual-branch LSTM.

    Branch 1: first 150 ms of 5 features:
        ['iln3iamp', 'betan', 'density', 'li', 'fs_sum_past_max_smoothed']
    Branch 2: full 200 ms of 9 features:
        ['pinj', 'tijnj', 'echpwrc', 'I_ECCD', 'tritop', 'tribot', 'Ip', 'bt', 'gasa']
    """
    print("Loading data for dual-branch LSTM...")
    df = pd.read_csv("/mnt/homes/sr4240/my_folder/plasma_data.csv")

    # Remove problematic shot (for consistency with other scripts)
    df = df[df["shot"] != 191675].copy()

    # Define feature lists
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

    # Ensure all columns exist
    missing1 = [f for f in branch1_features if f not in df.columns]
    missing2 = [f for f in branch2_features if f not in df.columns]
    if missing1 or missing2:
        raise ValueError(
            f"Missing required columns in plasma_data.csv. "
            f"Branch1 missing: {missing1}, Branch2 missing: {missing2}"
        )

    # Sort by shot and time
    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    X1 = df_sorted[branch1_features].values
    X2 = df_sorted[branch2_features].values
    y = df_sorted["state"].values
    times = df_sorted["time"].values
    shots = df_sorted["shot"].values

    # First, reproduce the baseline cleaning used in existing LSTM scripts:
    # require valid labels/times and non-NaN branch1 features.
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

    # Impute NaNs in branch2 features (some, like tijnj, may be entirely NaN)
    # so that we do not discard all rows.
    X2_imputed = X2.copy()
    for j in range(X2_imputed.shape[1]):
        col = X2_imputed[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            # If the entire column is NaN (e.g., tijnj), fill with 0.
            col[nan_mask] = 0.0
        elif nan_mask.any():
            # Otherwise fill with column mean.
            mean_val = np.nanmean(col)
            col[nan_mask] = mean_val
        X2_imputed[:, j] = col

    X2 = X2_imputed

    print("Data shape after cleaning (baseline mask + branch2 imputation):")
    print(f"  Branch1 features: {X1.shape}")
    print(f"  Branch2 features: {X2.shape}")
    print(f"  Labels: {y.shape}")

    print(f"\nRaw label distribution (4-state): {Counter(y)}")

    # Standardize features separately per branch
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    X2_scaled = scaler2.fit_transform(X2)

    return (
        X1_scaled,
        X2_scaled,
        y,
        times,
        shots,
        branch1_features,
        branch2_features,
        scaler1,
        scaler2,
    )


def create_combined_windows_by_shot(
    X1,
    X2,
    y,
    times,
    shots,
    window_size: int = WINDOW_SIZE,
    branch1_len: int = BRANCH1_LEN,
):
    """
    Create combined windows (14 features, 200 timesteps) with Branch 1
    zero-padded for the last 50 ms.

    Each window has shape (200, 14):
      - Columns 0..4  = Branch 1 features (real values for t=0..149, zeros for t=150..199)
      - Columns 5..13 = Branch 2 features (real values for all 200 timesteps)

    Label: state at the CURRENT time (end of the 200-timestep window).

    3-state mapping:
      raw 0 -> 0 (Suppressed)
      raw 1,2 -> 1 (Dithering/Mitigated)
      raw 3 -> 2 (ELMing)
    """
    n_feat1 = X1.shape[1]
    n_feat2 = X2.shape[1]
    print(
        f"\nCreating combined windows (window={window_size} steps, "
        f"Branch1 zeroed after t={branch1_len})..."
    )
    print(f"Combined features: {n_feat1} (Branch1) + {n_feat2} (Branch2) = {n_feat1 + n_feat2}")
    print("Split is RANDOM BY SHOT NUMBER (not individual data points).")

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots: {n_shots}")

    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)

    train_shots = set(shuffled_shots[:train_size])
    val_shots = set(shuffled_shots[train_size : train_size + val_size])
    test_shots = set(shuffled_shots[train_size + val_size :])

    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")

    # 3-state mapping
    label_mapping = {0: 0, 1: 1, 2: 1, 3: 2}
    valid_raw_labels = {0, 1, 2, 3}

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    windows_created = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        # Assign split
        if shot_id in train_shots:
            tx, ty = train_x, train_y
        elif shot_id in val_shots:
            tx, ty = val_x, val_y
        else:
            tx, ty = test_x, test_y

        shot_X1 = X1[shot_indices]
        shot_X2 = X2[shot_indices]
        shot_y = y[shot_indices]

        # Sliding windows within this shot
        for end_idx in range(window_size - 1, len(shot_indices)):
            start_idx = end_idx - window_size + 1

            # Label at CURRENT time (end of window)
            raw_label = int(shot_y[end_idx])
            if raw_label not in valid_raw_labels:
                continue
            mapped_label = label_mapping[raw_label]

            # Build combined window: (window_size, n_feat1 + n_feat2)
            window = np.zeros((window_size, n_feat1 + n_feat2), dtype=np.float32)

            # Branch 1: only first branch1_len timesteps, rest stays 0
            window[:branch1_len, :n_feat1] = shot_X1[start_idx : start_idx + branch1_len]

            # Branch 2: full window
            window[:, n_feat1:] = shot_X2[start_idx : end_idx + 1]

            tx.append(window)
            ty.append(mapped_label)
            windows_created += 1

    # Convert to numpy arrays
    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y)
    val_x = np.array(val_x, dtype=np.float32)
    val_y = np.array(val_y)
    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y)

    print(f"\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Window shape: {train_x.shape[1:]}  (timesteps, features)")
    print(f"\nCreated windows:")
    print(f"  Train: {len(train_x)}")
    print(f"  Val:   {len(val_x)}")
    print(f"  Test:  {len(test_x)}")

    print(f"\nLabel distribution (3-state):")
    print(f"  Train: {Counter(train_y)}")
    print(f"  Val:   {Counter(val_y)}")
    print(f"  Test:  {Counter(test_y)}")

    return train_x, train_y, val_x, val_y, test_x, test_y


def train_model(model, train_loader, val_loader, device, n_epochs: int = 50, use_amp: bool = True):
    """Train dual-branch LSTM model with optional mixed-precision (AMP)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
        verbose=True,
    )
    use_amp = use_amp and device.type == "cuda"
    if use_amp:
        try:
            scaler = torch.cuda.amp.GradScaler()
            autocast_ctx = torch.cuda.amp.autocast
        except AttributeError:
            use_amp = False
            scaler = None
            autocast_ctx = None
    else:
        scaler = None
        autocast_ctx = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with autocast_ctx():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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

        # Validation
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

        # Metrics
        train_acc = float(accuracy_score(all_train_labels, all_train_preds))
        val_acc = float(accuracy_score(all_val_labels, all_val_preds))
        avg_train_loss = epoch_train_loss / max(len(train_loader), 1)
        avg_val_loss = epoch_val_loss / max(len(val_loader), 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # Step scheduler on validation LOSS (what it expects by default)
        scheduler.step(avg_val_loss)

        # Early stopping on validation loss (save best model on lowest loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_test_lstm_50_3.pth")
            patience_counter = 0
            print("  ✓ New best model saved (val loss improved)!")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on the test set."""
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
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, labels=[0, 1, 2], digits=4
        )
    )

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    print("\nROC AUC Scores:")
    for i, class_name in enumerate(class_names):
        if i < all_probs.shape[1]:
            binary_labels = (all_labels == i).astype(int)
            if len(np.unique(binary_labels)) > 1:
                auc = roc_auc_score(binary_labels, all_probs[:, i])
                print(f"  {class_name}: {auc:.4f}")

    return all_preds, all_labels, all_probs


def plot_results(
    train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names
):
    """Plot training curves and confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss curves
    axes[0, 0].plot(train_losses, label="Train Loss", color="blue")
    axes[0, 0].plot(val_losses, label="Val Loss", color="red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss (Single LSTM, 3-state)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(train_accs, label="Train Acc", color="blue")
    axes[0, 1].plot(val_accs, label="Val Acc", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy (Single LSTM, 3-state)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion matrix (normalized)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2], normalize="true")
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Normalized Confusion Matrix (3-state)")
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")

    # Confusion matrix (counts)
    cm_counts = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    sns.heatmap(
        cm_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Confusion Matrix (Counts, 3-state)")
    axes[1, 1].set_ylabel("True Label")
    axes[1, 1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(
        "test_lstm_50_3_dual_branch_results.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("Results saved to 'test_lstm_50_3_dual_branch_results.png'")


def main():
    print("=" * 60)
    print("Single-Branch LSTM Model for Plasma State Classification (3-state)")
    print("=" * 60)
    print("Prediction target: CURRENT time's state (no future horizon)")
    print(f"Window: {WINDOW_MS} ms of past data (≈ {WINDOW_SIZE} samples)")
    print(
        f"All 14 features combined into one LSTM. "
        f"Branch 1 (5 features) zero-padded for last {WINDOW_MS - BRANCH1_MS} ms."
    )
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and scale data
    (
        X1,
        X2,
        y,
        times,
        shots,
        branch1_features,
        branch2_features,
        scaler1,
        scaler2,
    ) = load_and_prepare_data()

    # Create windows
    (
        train_x, train_y,
        val_x, val_y,
        test_x, test_y,
    ) = create_combined_windows_by_shot(
        X1, X2, y, times, shots, window_size=WINDOW_SIZE, branch1_len=BRANCH1_LEN
    )

    print("\nFinal dataset sizes:")
    print(f"  Train: {len(train_x)} samples")
    print(f"  Val:   {len(val_x)} samples")
    print(f"  Test:  {len(test_x)} samples")

    # Build datasets and loaders
    train_dataset = CombinedDataset(train_x, train_y)
    val_dataset = CombinedDataset(val_x, val_y)
    test_dataset = CombinedDataset(test_x, test_y)

    _num_workers = 4 if torch.cuda.is_available() else 0
    _pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )

    # Model
    n_features = len(branch1_features) + len(branch2_features)
    model = SingleBranchLSTM(
        n_features=n_features,
        n_classes=N_CLASSES,
    ).to(device)

    # Quick forward-pass timing
    print("\nTesting forward pass speed...")
    test_batch_x, _ = next(iter(train_loader))
    test_batch_x = test_batch_x.to(device)

    start_t = time.time()
    with torch.no_grad():
        _ = model(test_batch_x)
    elapsed = time.time() - start_t
    print(
        f"Forward pass time for batch of {test_batch_x.shape[0]}: {elapsed:.3f} seconds"
    )

    # Training
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, n_epochs=50
    )

    # Load best model
    print("\nLoading best model from 'best_test_lstm_50_3.pth' ...")
    model.load_state_dict(torch.load("best_test_lstm_50_3.pth", map_location=device))

    # Evaluate
    class_names = ["Suppressed", "Dithering/Mitigated", "ELMing"]
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names
    )

    # Plot results
    plot_results(
        train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names
    )

    final_acc = accuracy_score(all_labels, all_preds)
    print("\n" + "=" * 60)
    print(
        f"Training Complete! Single LSTM (current state, window={WINDOW_MS} ms) - Test Accuracy: {final_acc:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

