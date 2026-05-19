import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import KFold
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
WINDOW_MS = 200          # full window length (long branch)
BRANCH1_MS = 150         # first 150 ms for short branch

WINDOW_SIZE = WINDOW_MS
BRANCH1_LEN = BRANCH1_MS

# 3-state classification: Suppressed, Dithering/Mitigated (combined), ELMing
N_CLASSES = 3

TCN_KERNEL_SIZE = 3
TCN_CHANNELS = 32
TCN_DROPOUT = 0.3
MLP_DROPOUT = 0.4
BATCH_SIZE = 512


def tcn_receptive_field(num_blocks: int, kernel_size: int = TCN_KERNEL_SIZE) -> int:
    """Receptive field along the main conv path (two dilated convs per block)."""
    rf = 1
    for b in range(num_blocks):
        d = 2**b
        rf += (kernel_size - 1) * d
        rf += (kernel_size - 1) * d
    return rf


def min_tcn_blocks_for_window(window_len: int, kernel_size: int = TCN_KERNEL_SIZE) -> int:
    n = 0
    while tcn_receptive_field(n, kernel_size) < window_len:
        n += 1
    return max(n, 1)


class CausalConv1d(nn.Module):
    """Left padding only so output length equals input length (causal)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """Dilated causal Conv1d → ReLU → Dropout ×2 + residual (+ 1x1 if channels differ)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBranch(nn.Module):
    """Stack of TCN blocks; input (B, L, F) → last timestep (B, num_channels)."""

    def __init__(
        self,
        in_features: int,
        window_len: int,
        num_channels: int = TCN_CHANNELS,
        kernel_size: int = TCN_KERNEL_SIZE,
        dropout: float = TCN_DROPOUT,
    ):
        super().__init__()
        self.window_len = window_len
        self.num_blocks = min_tcn_blocks_for_window(window_len, kernel_size)
        self.receptive_field = tcn_receptive_field(self.num_blocks, kernel_size)

        blocks = []
        in_ch = in_features
        for b in range(self.num_blocks):
            d = 2**b
            blocks.append(
                TCNResidualBlock(
                    in_ch, num_channels, kernel_size, d, dropout
                )
            )
            in_ch = num_channels
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.network(x)
        return h[:, :, -1]


class DualBranchTCN(nn.Module):
    """
    Two parallel TCN branches:
      - Short: first 150 ms, n_short features
      - Long: full 200 ms, n_long features
    Last timestep per branch → concat → MLP → classifier.
    """

    def __init__(
        self,
        n_short: int,
        n_long: int,
        short_len: int = BRANCH1_LEN,
        long_len: int = WINDOW_SIZE,
        num_channels: int = TCN_CHANNELS,
        kernel_size: int = TCN_KERNEL_SIZE,
        tcn_dropout: float = TCN_DROPOUT,
        nn_hidden_sizes=None,
        classifier_hidden: int = 32,
        n_classes: int = N_CLASSES,
        mlp_dropout: float = MLP_DROPOUT,
    ):
        super().__init__()

        if nn_hidden_sizes is None:
            nn_hidden_sizes = [32]

        self.tcn_short = TCNBranch(
            n_short, short_len, num_channels, kernel_size, tcn_dropout
        )
        self.tcn_long = TCNBranch(
            n_long, long_len, num_channels, kernel_size, tcn_dropout
        )

        layers = []
        in_dim = 2 * num_channels
        for h in nn_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(mlp_dropout),
                ]
            )
            in_dim = h

        self.mlp = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(classifier_hidden, n_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("Dual-Branch TCN Model (3-state)")
        print("=" * 60)
        print(f"Short branch: window={short_len}, receptive field={self.tcn_short.receptive_field}")
        print(
            f"  TCN blocks={self.tcn_short.num_blocks}, "
            f"channels={num_channels}, kernel={kernel_size}"
        )
        print(f"Long branch:  window={long_len}, receptive field={self.tcn_long.receptive_field}")
        print(
            f"  TCN blocks={self.tcn_long.num_blocks}, "
            f"channels={num_channels}, kernel={kernel_size}"
        )
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60)
        print(
            f"Architecture: TCN1 ({BRANCH1_MS} ms, {n_short} feat) + "
            f"TCN2 ({WINDOW_MS} ms, {n_long} feat) → Concat → MLP → Classifier"
        )

    def forward(self, x_short, x_long):
        """
        x_short: (batch, 150, n_short)
        x_long:  (batch, 200, n_long)
        """
        h_s = self.tcn_short(x_short)
        h_l = self.tcn_long(x_long)
        fused = torch.cat([h_s, h_l], dim=1)
        features = self.mlp(fused)
        logits = self.classifier(features)
        return logits


class DualInputDataset(Dataset):
    """Dataset with separate short-window and long-window tensors."""

    def __init__(self, x_short, x_long, labels):
        self.x_short = np.ascontiguousarray(x_short, dtype=np.float32)
        self.x_long = np.ascontiguousarray(x_long, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_short[idx]),
            torch.from_numpy(self.x_long[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def load_and_prepare_data():
    """
    Load and preprocess plasma data for dual-branch architecture.

    Short branch (6 features, 150 ms): iln3iamp, betan, density, li,
        fs_sum_past_max_smoothed, n_eped
    Long branch (9 features, 200 ms): pinj, tijnj, echpwrc, I_ECCD, tritop,
        tribot, Ip, bt, gasa
    """
    print("Loading data for dual-TCN (2 parallel TCN branches)...")
    df = pd.read_csv("/mnt/homes/sr4240/my_folder/plasma_data.csv")

    df = df[df["shot"] != 191675].copy()

    branch_short_features = [
        "iln3iamp",
        "betan",
        "density",
        "li",
        "fs_sum_past_max_smoothed",
        "n_eped",
    ]
    branch_long_features = [
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

    missing_s = [f for f in branch_short_features if f not in df.columns]
    missing_l = [f for f in branch_long_features if f not in df.columns]
    if missing_s or missing_l:
        raise ValueError(
            f"Missing required columns in plasma_data.csv. "
            f"Short branch missing: {missing_s}, Long branch missing: {missing_l}"
        )

    df_sorted = df.sort_values(["shot", "time"]).reset_index(drop=True)

    Xs = df_sorted[branch_short_features].values
    Xl = df_sorted[branch_long_features].values
    y = df_sorted["state"].values
    times = df_sorted["time"].values
    shots = df_sorted["shot"].values

    valid_mask = (
        ~np.isnan(Xs).any(axis=1)
        & ~np.isnan(y)
        & ~np.isnan(times)
    )
    Xs = Xs[valid_mask]
    Xl = Xl[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    Xl_imputed = Xl.copy()
    for j in range(Xl_imputed.shape[1]):
        col = Xl_imputed[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[nan_mask] = 0.0
        elif nan_mask.any():
            mean_val = np.nanmean(col)
            col[nan_mask] = mean_val
        Xl_imputed[:, j] = col
    Xl = Xl_imputed

    print("Data shape after cleaning (short valid mask + long imputation):")
    print(f"  Short branch: {Xs.shape}")
    print(f"  Long branch:  {Xl.shape}")
    print(f"  Labels: {y.shape}")

    print(f"\nRaw label distribution (4-state): {Counter(y)}")

    scaler_s = StandardScaler()
    scaler_l = StandardScaler()
    Xs_scaled = scaler_s.fit_transform(Xs)
    Xl_scaled = scaler_l.fit_transform(Xl)

    return (
        Xs_scaled,
        Xl_scaled,
        y,
        times,
        shots,
        branch_short_features,
        branch_long_features,
        scaler_s,
        scaler_l,
    )


def create_dual_windows_by_shot(
    Xs,
    Xl,
    y,
    times,
    shots,
    window_size: int = WINDOW_SIZE,
    branch1_len: int = BRANCH1_LEN,
):
    """
    Build separate tensors per window:
      - x_short: (150, n_short) — same global time span as first 150 steps of window
      - x_long:  (200, n_long) — full window on long features

    Label: state at the end of the 200-step window (3-state mapping).
    """
    n_short = Xs.shape[1]
    n_long = Xl.shape[1]
    print(
        f"\nCreating dual-input windows (long window={window_size} steps, "
        f"short branch length={branch1_len})..."
    )
    print(f"Short: {n_short} features × {branch1_len} ms; Long: {n_long} features × {window_size} ms")
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

    label_mapping = {0: 0, 1: 1, 2: 1, 3: 2}
    valid_raw_labels = {0, 1, 2, 3}

    train_xs, train_xl, train_y = [], [], []
    val_xs, val_xl, val_y = [], [], []
    test_xs, test_xl, test_y = [], [], []

    windows_created = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]

        if len(shot_indices) < window_size:
            continue

        if shot_id in train_shots:
            txs, txl, ty = train_xs, train_xl, train_y
        elif shot_id in val_shots:
            txs, txl, ty = val_xs, val_xl, val_y
        else:
            txs, txl, ty = test_xs, test_xl, test_y

        shot_Xs = Xs[shot_indices]
        shot_Xl = Xl[shot_indices]
        shot_y = y[shot_indices]

        for end_idx in range(window_size - 1, len(shot_indices)):
            start_idx = end_idx - window_size + 1

            raw_label = int(shot_y[end_idx])
            if raw_label not in valid_raw_labels:
                continue
            mapped_label = label_mapping[raw_label]

            w_short = shot_Xs[start_idx : start_idx + branch1_len].astype(
                np.float32, copy=False
            )
            w_long = shot_Xl[start_idx : end_idx + 1].astype(np.float32, copy=False)

            txs.append(w_short)
            txl.append(w_long)
            ty.append(mapped_label)
            windows_created += 1

    train_xs = np.array(train_xs, dtype=np.float32)
    train_xl = np.array(train_xl, dtype=np.float32)
    train_y = np.array(train_y)
    val_xs = np.array(val_xs, dtype=np.float32)
    val_xl = np.array(val_xl, dtype=np.float32)
    val_y = np.array(val_y)
    test_xs = np.array(test_xs, dtype=np.float32)
    test_xl = np.array(test_xl, dtype=np.float32)
    test_y = np.array(test_y)

    print(f"\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Short window shape: {train_xs.shape[1:]}")
    print(f"  Long window shape:  {train_xl.shape[1:]}")
    print(f"\nCreated windows:")
    print(f"  Train: {len(train_xs)}")
    print(f"  Val:   {len(val_xs)}")
    print(f"  Test:  {len(test_xs)}")

    print(f"\nLabel distribution (3-state):")
    print(f"  Train: {Counter(train_y)}")
    print(f"  Val:   {Counter(val_y)}")
    print(f"  Test:  {Counter(test_y)}")

    return (
        train_xs, train_xl, train_y,
        val_xs, val_xl, val_y,
        test_xs, test_xl, test_y,
    )


def compute_class_weights(train_y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = torch.tensor(inv, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    class_weights: torch.Tensor,
    n_epochs: int = 50,
    use_amp: bool = True,
):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=7,
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
    max_patience = 15

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for batch_xs, batch_xl, batch_y in train_loader:
            batch_xs = batch_xs.to(device, non_blocking=True)
            batch_xl = batch_xl.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with autocast_ctx():
                    outputs = model(batch_xs, batch_xl)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_xs, batch_xl)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            for batch_xs, batch_xl, batch_y in val_loader:
                batch_xs = batch_xs.to(device, non_blocking=True)
                batch_xl = batch_xl.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                outputs = model(batch_xs, batch_xl)
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

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_test_tcn_50_3_dual_tcn.pth")
            patience_counter = 0
            print("  ✓ New best model saved (val loss improved)!")
        else:
            if epoch >= 5:
                patience_counter += 1

        if epoch >= 5 and patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.inference_mode():
        for batch_xs, batch_xl, batch_y in test_loader:
            batch_xs = batch_xs.to(device, non_blocking=True)
            batch_xl = batch_xl.to(device, non_blocking=True)

            outputs = model(batch_xs, batch_xl)
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label="Train Loss", color="blue")
    axes[0, 0].plot(val_losses, label="Val Loss", color="red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss (Dual TCN, 3-state)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label="Train Acc", color="blue")
    axes[0, 1].plot(val_accs, label="Val Acc", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy (Dual TCN, 3-state)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

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
        "test_tcn_50_3_dual_tcn_results.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("Results saved to 'test_tcn_50_3_dual_tcn_results.png'")


def main():
    print("=" * 60)
    print("Dual-TCN Model for Plasma State Classification (3-state)")
    print("=" * 60)
    print("Prediction target: CURRENT time's state (no future horizon)")
    print(
        f"TCN 1: first {BRANCH1_MS} ms, 6 short features | "
        f"TCN 2: full {WINDOW_MS} ms, 9 long features"
    )
    print("Merge: Concat(TCN1 last step, TCN2 last step) → MLP → Classifier")
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (
        Xs,
        Xl,
        y,
        times,
        shots,
        branch_short_features,
        branch_long_features,
        _scaler_s,
        _scaler_l,
    ) = load_and_prepare_data()

    (
        train_xs, train_xl, train_y,
        val_xs, val_xl, val_y,
        test_xs, test_xl, test_y,
    ) = create_dual_windows_by_shot(
        Xs, Xl, y, times, shots, window_size=WINDOW_SIZE, branch1_len=BRANCH1_LEN
    )

    print("\nFinal dataset sizes:")
    print(f"  Train: {len(train_xs)} samples")
    print(f"  Val:   {len(val_xs)} samples")
    print(f"  Test:  {len(test_xs)} samples")

    class_weights = compute_class_weights(train_y, N_CLASSES)
    print(f"\nClass weights (inverse frequency, mean-normalized): {class_weights.numpy()}")

    train_dataset = DualInputDataset(train_xs, train_xl, train_y)
    val_dataset = DualInputDataset(val_xs, val_xl, val_y)
    test_dataset = DualInputDataset(test_xs, test_xl, test_y)

    _num_workers = 4 if torch.cuda.is_available() else 0
    _pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
    )

    model = DualBranchTCN(
        n_short=len(branch_short_features),
        n_long=len(branch_long_features),
        nn_hidden_sizes=[32],
        n_classes=N_CLASSES,
    ).to(device)

    print("\nTesting forward pass speed...")
    batch_xs, batch_xl, _ = next(iter(train_loader))
    batch_xs = batch_xs.to(device)
    batch_xl = batch_xl.to(device)

    start_t = time.time()
    with torch.no_grad():
        _ = model(batch_xs, batch_xl)
    elapsed = time.time() - start_t
    print(
        f"Forward pass time for batch of {batch_xs.shape[0]}: {elapsed:.3f} seconds"
    )

    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model,
        train_loader,
        val_loader,
        device,
        class_weights,
        n_epochs=50,
    )

    print("\nLoading best model from 'best_test_tcn_50_3_dual_tcn.pth' ...")
    model.load_state_dict(
        torch.load("best_test_tcn_50_3_dual_tcn.pth", map_location=device)
    )

    class_names = ["Suppressed", "Dithering/Mitigated", "ELMing"]
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names
    )

    plot_results(
        train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, class_names
    )

    final_acc = accuracy_score(all_labels, all_preds)
    print("\n" + "=" * 60)
    print(
        f"Training Complete! Dual TCN (current state, {WINDOW_MS} ms context) - "
        f"Test Accuracy: {final_acc:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
