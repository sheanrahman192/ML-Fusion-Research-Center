import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
import argparse

warnings.filterwarnings("ignore")

# K-fold CV (shot-level; windows never cross shots)
N_FOLDS_DEFAULT = 5
CHECKPOINT_BASENAME = "best_test_lstm_50_3_dual_lstm"
RESULTS_PLOT_BASENAME = "test_lstm_50_3_dual_lstm_results"

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


class DualBranchLSTM(nn.Module):
    """
    Two parallel LSTMs per diagram:
      - LSTM 1: short features, first 150 ms of the 200 ms context
      - LSTM 2: long features, full 200 ms window
    Concatenate final hidden states → MLP → classifier.
    """

    def __init__(
        self,
        n_short: int,
        n_long: int,
        lstm_hidden: int = 64,
        lstm_num_layers: int = 1,
        nn_hidden_sizes=None,
        classifier_hidden: int = 32,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        if nn_hidden_sizes is None:
            nn_hidden_sizes = [64, 32]

        self.lstm_short = nn.LSTM(
            input_size=n_short,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.3 if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_long = nn.LSTM(
            input_size=n_long,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.3 if lstm_num_layers > 1 else 0.0,
        )

        layers = []
        in_dim = 2 * lstm_hidden
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
        print("Dual-Branch LSTM Model Parameter Count (3-state):")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60)
        print(
            f"Architecture: LSTM1 ({BRANCH1_MS} ms, {n_short} feat) + "
            f"LSTM2 ({WINDOW_MS} ms, {n_long} feat) → Concat → MLP → Classifier"
        )

    def forward(self, x_short, x_long):
        """
        x_short: (batch, 150, n_short)
        x_long:  (batch, 200, n_long)
        """
        _, (h_s, _) = self.lstm_short(x_short)
        _, (h_l, _) = self.lstm_long(x_long)
        h_last_s = h_s[-1]
        h_last_l = h_l[-1]
        fused = torch.cat([h_last_s, h_last_l], dim=1)
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
    Load and preprocess plasma data for dual-LSTM architecture.

    Short branch (6 features, 150 ms): iln3iamp, betan, density, li,
        fs_sum_past_max_smoothed, n_eped
    Long branch (9 features, 200 ms): pinj, tijnj, echpwrc, I_ECCD, tritop,
        tribot, Ip, bt, gasa
    """
    print("Loading data for dual-LSTM (2 parallel LSTMs)...")
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

    # Unscaled; StandardScaler is fit per CV fold (or single split) on train shots only.
    return (
        Xs.astype(np.float32),
        Xl.astype(np.float32),
        y,
        times,
        shots,
        branch_short_features,
        branch_long_features,
    )


def scale_features_from_train_shots(Xs, Xl, shots, train_shots):
    """Fit StandardScalers on training-shot rows only; transform all rows."""
    train_shots = set(train_shots)
    mask = np.isin(shots, list(train_shots))
    if not np.any(mask):
        raise ValueError("No rows from train_shots; cannot fit scalers.")
    scaler_s = StandardScaler()
    scaler_l = StandardScaler()
    scaler_s.fit(Xs[mask])
    scaler_l.fit(Xl[mask])
    Xs_scaled = scaler_s.transform(Xs).astype(np.float32)
    Xl_scaled = scaler_l.transform(Xl).astype(np.float32)
    return Xs_scaled, Xl_scaled


def create_dual_windows_by_shot(
    Xs,
    Xl,
    y,
    times,
    shots,
    train_shots,
    val_shots,
    test_shots,
    window_size: int = WINDOW_SIZE,
    branch1_len: int = BRANCH1_LEN,
):
    """
    Build separate tensors per window:
      - x_short: (150, n_short) — same global time span as first 150 steps of window
      - x_long:  (200, n_long) — full window on long features

    Label: state at the end of the 200-step window (3-state mapping).

    train_shots, val_shots, test_shots: disjoint sets of shot IDs (by-shot split).
    """
    train_shots = set(train_shots)
    val_shots = set(val_shots)
    test_shots = set(test_shots)

    n_short = Xs.shape[1]
    n_long = Xl.shape[1]
    print(
        f"\nCreating dual-input windows (long window={window_size} steps, "
        f"short branch length={branch1_len})..."
    )
    print(f"Short: {n_short} features × {branch1_len} ms; Long: {n_long} features × {window_size} ms")
    print("Split is BY SHOT NUMBER (not individual data points).")

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots in data: {n_shots}")
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


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs: int = 50,
    use_amp: bool = True,
    checkpoint_path: str = None,
):
    if checkpoint_path is None:
        checkpoint_path = f"{CHECKPOINT_BASENAME}.pth"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
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
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_xs, batch_xl)
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
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            print("  ✓ New best model saved (val loss improved)!")
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
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    all_preds,
    all_labels,
    class_names,
    save_path: str = None,
):
    if save_path is None:
        save_path = f"{RESULTS_PLOT_BASENAME}.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label="Train Loss", color="blue")
    axes[0, 0].plot(val_losses, label="Val Loss", color="red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss (Dual LSTM, 3-state)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label="Train Acc", color="blue")
    axes[0, 1].plot(val_accs, label="Val Acc", color="red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy (Dual LSTM, 3-state)")
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Results saved to '{save_path}'")


def single_split_shot_sets(unique_shots, seed: int = 42):
    """Original 70% / 15% / 15% random shot split."""
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_shots)
    n = len(shuffled)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    train_shots = set(shuffled[:train_size])
    val_shots = set(shuffled[train_size : train_size + val_size])
    test_shots = set(shuffled[train_size + val_size :])
    return train_shots, val_shots, test_shots


def run_training_pipeline(
    Xs,
    Xl,
    y,
    times,
    shots,
    branch_short_features,
    branch_long_features,
    train_shots,
    val_shots,
    test_shots,
    device,
    checkpoint_path: str,
    plot_path: str,
    n_epochs: int = 50,
):
    Xs_s, Xl_s = scale_features_from_train_shots(Xs, Xl, shots, train_shots)
    (
        train_xs,
        train_xl,
        train_y,
        val_xs,
        val_xl,
        val_y,
        test_xs,
        test_xl,
        test_y,
    ) = create_dual_windows_by_shot(
        Xs_s,
        Xl_s,
        y,
        times,
        shots,
        train_shots,
        val_shots,
        test_shots,
        window_size=WINDOW_SIZE,
        branch1_len=BRANCH1_LEN,
    )

    print("\nFinal dataset sizes:")
    print(f"  Train: {len(train_xs)} samples")
    print(f"  Val:   {len(val_xs)} samples")
    print(f"  Test:  {len(test_xs)} samples")

    train_dataset = DualInputDataset(train_xs, train_xl, train_y)
    val_dataset = DualInputDataset(val_xs, val_xl, val_y)
    test_dataset = DualInputDataset(test_xs, test_xl, test_y)

    if len(train_dataset) < 2:
        raise ValueError(
            f"Training set has only {len(train_dataset)} sample(s); "
            "BatchNorm requires at least 2 training samples."
        )

    train_batch_size = min(2048, len(train_dataset))
    # Avoid singleton final batch, which breaks BatchNorm during training.
    train_drop_last = (len(train_dataset) % train_batch_size) == 1

    _num_workers = 4 if torch.cuda.is_available() else 0
    _pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=train_drop_last,
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

    model = DualBranchLSTM(
        n_short=len(branch_short_features),
        n_long=len(branch_long_features),
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
        n_epochs=n_epochs,
        checkpoint_path=checkpoint_path,
    )

    print(f"\nLoading best model from '{checkpoint_path}' ...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    class_names = ["Suppressed", "Dithering/Mitigated", "ELMing"]
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names
    )

    plot_results(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        all_preds,
        all_labels,
        class_names,
        save_path=plot_path,
    )

    final_acc = accuracy_score(all_labels, all_preds)
    return final_acc, train_losses, val_losses


def parse_args():
    p = argparse.ArgumentParser(
        description="Dual-LSTM 3-state plasma classifier (shot-level CV or single split)."
    )
    p.add_argument(
        "--single-split",
        action="store_true",
        help="Use one random 70%%/15%%/15%% shot split (legacy) instead of K-fold CV.",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS_DEFAULT,
        help=f"Number of K-folds (shot-level). Default: {N_FOLDS_DEFAULT}.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per fold (or single run).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Dual-LSTM Model for Plasma State Classification (3-state)")
    print("=" * 60)
    print("Prediction target: CURRENT time's state (no future horizon)")
    print(
        f"LSTM 1: first {BRANCH1_MS} ms, 6 short features | "
        f"LSTM 2: full {WINDOW_MS} ms, 9 long features"
    )
    print("Merge: Concat(LSTM1 hidden, LSTM2 hidden) → MLP → Classifier")
    print("Split: BY SHOT NUMBER (not individual data points)")
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
    ) = load_and_prepare_data()

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    if args.single_split:
        train_shots, val_shots, test_shots = single_split_shot_sets(unique_shots, seed=42)
        final_acc, _, _ = run_training_pipeline(
            Xs,
            Xl,
            y,
            times,
            shots,
            branch_short_features,
            branch_long_features,
            train_shots,
            val_shots,
            test_shots,
            device,
            checkpoint_path=f"{CHECKPOINT_BASENAME}.pth",
            plot_path=f"{RESULTS_PLOT_BASENAME}.png",
            n_epochs=args.epochs,
        )
        print("\n" + "=" * 60)
        print(
            f"Training Complete! Dual LSTM (current state, {WINDOW_MS} ms context) — "
            f"Test Accuracy: {final_acc:.4f}"
        )
        print("=" * 60)
        return

    n_folds = max(2, args.folds)
    if n_shots < n_folds:
        raise ValueError(
            f"Need at least {n_folds} unique shots for {n_folds}-fold CV; got {n_shots}."
        )

    print(
        f"\nK-fold cross-validation: {n_folds} folds, shot-level (no windows crossing shots)."
    )
    print(
        "Per fold: test = one fold of shots; remaining shots split ~87.5%% train / "
        "~12.5%% val for early stopping (same ratio as 70/15 of all shots)."
    )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(shuffled_shots)):
        test_shots = set(shuffled_shots[test_idx])
        remaining = shuffled_shots[train_val_idx]
        n_rem = len(remaining)
        train_size = int(0.875 * n_rem)
        if train_size < 1:
            train_size = 1
        if n_rem > 1 and train_size >= n_rem:
            train_size = n_rem - 1
        train_shots = set(remaining[:train_size])
        val_shots = set(remaining[train_size:])
        if len(val_shots) == 0 and len(train_shots) > 1:
            move = next(iter(train_shots))
            train_shots = train_shots - {move}
            val_shots = {move}

        print("\n" + "=" * 60)
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(
            f"  Shots — Train: {len(train_shots)}, Val: {len(val_shots)}, Test: {len(test_shots)}"
        )
        print("=" * 60)

        ckpt = f"{CHECKPOINT_BASENAME}_fold{fold_idx + 1}.pth"
        plot_path = f"{RESULTS_PLOT_BASENAME}_fold{fold_idx + 1}.png"

        final_acc, _, _ = run_training_pipeline(
            Xs,
            Xl,
            y,
            times,
            shots,
            branch_short_features,
            branch_long_features,
            train_shots,
            val_shots,
            test_shots,
            device,
            checkpoint_path=ckpt,
            plot_path=plot_path,
            n_epochs=args.epochs,
        )
        fold_accs.append(final_acc)
        print(f"Fold {fold_idx + 1} test accuracy: {final_acc:.4f}")

    fold_accs = np.asarray(fold_accs)
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0

    print("\n" + "=" * 60)
    print(f"Cross-validation complete ({n_folds} folds, shot-level)")
    print(f"  Test accuracy per fold: {fold_accs}")
    print(f"  Mean ± std: {mean_acc:.4f} ± {std_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
