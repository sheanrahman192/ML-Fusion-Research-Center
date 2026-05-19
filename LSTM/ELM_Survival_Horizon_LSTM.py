"""
ELM prediction in the spirit of:
  Perry et al., "Risk-Aware Framework Development for Disruption Prediction:
  Alcator C-Mod and DIII-D Survival Analysis", J. Fusion Energy (2024),
  https://doi.org/10.1007/s10894-024-00413-y

That work trains survival-style models (Cox, Deep Cox, DSM) so that at each time
one can ask P(event within [t, t + Δt_horizon]). Here we predict *ELM onset*
from the same 200 ms causal windows and 14-channel setup as Test_LSTM_50_3.py:
  - Event: first transition into raw state 3 (ELMing) strictly after the window end.
  - Horizons: configurable ms windows; labels are binary "ELM within horizon"
    with masking when the shot ends before the horizon (right-censoring).

Architecture: LSTM encoder (same role as deep nets in Deep Cox / DSM) +
multi-head sigmoid outputs, one per horizon (marginal P(within H_k)).

This is not a full Auton-Survival DSM/Cox implementation; it matches the paper's
*operational* output (risk within horizon) on your plasma_data.csv pipeline.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
import time
from typing import Optional

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Align with Test_LSTM_50_3.py
WINDOW_MS = 200
BRANCH1_MS = 150
WINDOW_SIZE = WINDOW_MS
BRANCH1_LEN = BRANCH1_MS
DT_MS = 1.0  # 1 sample ≈ 1 ms, same as Test_LSTM_50_3

# Raw plasma_data "state": 3 == ELMing (see Test_LSTM_50_3.py)
RAW_ELM = 3

# Horizons (ms) for P(ELM within H | history) — analogous to paper's Δt_horizon
DEFAULT_HORIZONS_MS = (20, 50, 100, 200)


class SurvivalHorizonLSTM(nn.Module):
    """
    LSTM over (200, 14) windows → one logit per horizon (multi-task survival-style).
    """

    def __init__(
        self,
        n_features: int = 14,
        lstm_hidden: int = 32,
        lstm_num_layers: int = 1,
        nn_hidden_sizes=None,
        classifier_hidden: int = 24,
        horizons: tuple = DEFAULT_HORIZONS_MS,
        dropout_lstm_out: float = 0.35,
        dropout_mlp: float = 0.55,
        dropout_head: float = 0.45,
    ):
        super().__init__()
        self.horizons = horizons
        n_out = len(horizons)

        if nn_hidden_sizes is None:
            nn_hidden_sizes = [32]

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.3 if lstm_num_layers > 1 else 0.0,
        )
        self.dropout_lstm = nn.Dropout(dropout_lstm_out)

        layers = []
        in_dim = lstm_hidden
        for h in nn_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.LayerNorm(h),
                    nn.ReLU(),
                    nn.Dropout(dropout_mlp),
                ]
            )
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        self.heads = nn.Sequential(
            nn.Linear(in_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(classifier_hidden, n_out),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print("\n" + "=" * 60)
        print("Survival-horizon LSTM (ELM within Δt)")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Horizons (ms): {horizons}")
        print("=" * 60)

    def forward(self, x):
        # x: (batch, 200, 14)
        _, (h, _) = self.lstm(x)
        h_last = self.dropout_lstm(h[-1])
        z = self.mlp(h_last)
        logits = self.heads(z)
        return logits


class SurvivalHorizonDataset(Dataset):
    def __init__(self, windows, labels, masks):
        self.x = np.ascontiguousarray(windows, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.float32)  # (N, n_horizons)
        self.mask = np.asarray(masks, dtype=np.float32)  # (N, n_horizons), 1 = use in loss

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.mask[idx]),
        )


def load_and_prepare_data():
    """Same as Test_LSTM_50_3.py (paths, branches, scaling)."""
    print("Loading data for survival-horizon ELM model...")
    df = pd.read_csv("/mnt/homes/sr4240/my_folder/plasma_data.csv")
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
    )


def create_survival_horizon_windows(
    X1,
    X2,
    y_raw,
    times,
    shots,
    horizons_ms,
    window_size: int = WINDOW_SIZE,
    branch1_len: int = BRANCH1_LEN,
):
    """
    Build the same combined windows as Test_LSTM_50_3, but labels are vectors:
      y[k] = 1 iff first ELM (state==3) occurs in (t_end, t_end + H_k] (in time steps),
      with mask[k]=0 if the shot ends before we could observe the full horizon
      (right-censored for that H without an ELM — excluded from loss for that head).

    If an ELM occurs at time t_e > t_end: duration_steps = t_e - t_end (in samples).
    Censoring: no ELM in remainder of shot → event=0, time_to_censor = last_idx - end_idx.
    """
    n_feat1 = X1.shape[1]
    n_feat2 = X2.shape[1]
    n_h = len(horizons_ms)
    horizon_steps = [int(round(h / DT_MS)) for h in horizons_ms]

    print(
        f"\nCreating survival-horizon windows (window={window_size}, "
        f"horizons_ms={horizons_ms})..."
    )

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    np.random.seed(42)
    shuffled = np.random.permutation(unique_shots)

    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)
    train_shots = set(shuffled[:train_size])
    val_shots = set(shuffled[train_size : train_size + val_size])
    test_shots = set(shuffled[train_size + val_size :])

    print(f"Shot split: train={len(train_shots)}, val={len(val_shots)}, test={len(test_shots)}")

    train_x, train_y, train_m = [], [], []
    val_x, val_y, val_m = [], [], []
    test_x, test_y, test_m = [], [], []

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        if len(shot_indices) < window_size:
            continue

        if shot_id in train_shots:
            tx, ty, tm = train_x, train_y, train_m
        elif shot_id in val_shots:
            tx, ty, tm = val_x, val_y, val_m
        else:
            tx, ty, tm = test_x, test_y, test_m

        shot_X1 = X1[shot_indices]
        shot_X2 = X2[shot_indices]
        shot_y = y_raw[shot_indices]
        n_shot = len(shot_y)

        for end_local in range(window_size - 1, n_shot):
            start_local = end_local - window_size + 1

            window = np.zeros((window_size, n_feat1 + n_feat2), dtype=np.float32)
            window[:branch1_len, :n_feat1] = shot_X1[start_local : start_local + branch1_len]
            window[:, n_feat1:] = shot_X2[start_local : end_local + 1]

            state_end = shot_y[end_local]
            e_after = None
            for j in range(end_local + 1, n_shot):
                if shot_y[j] == RAW_ELM:
                    e_after = j
                    break

            # Samples strictly after window end until shot end (for censoring visibility)
            remain_steps = n_shot - 1 - end_local

            y_vec = np.zeros(n_h, dtype=np.float32)
            m_vec = np.zeros(n_h, dtype=np.float32)

            for ki, Hs in enumerate(horizon_steps):
                if state_end == RAW_ELM:
                    # Already ELMing at window end — do not train "upcoming ELM" heads
                    m_vec[ki] = 0.0
                    continue
                if e_after is not None:
                    dur = e_after - end_local
                    m_vec[ki] = 1.0
                    y_vec[ki] = 1.0 if dur <= Hs else 0.0
                else:
                    if remain_steps >= Hs:
                        m_vec[ki] = 1.0
                        y_vec[ki] = 0.0
                    else:
                        m_vec[ki] = 0.0

            tx.append(window)
            ty.append(y_vec)
            tm.append(m_vec)

    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)
    train_m = np.array(train_m, dtype=np.float32)
    val_x = np.array(val_x, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.float32)
    val_m = np.array(val_m, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)
    test_m = np.array(test_m, dtype=np.float32)

    print(f"Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")
    for i, h in enumerate(horizons_ms):
        print(
            f"  H={h}ms: train labeled count={np.sum(train_m[:, i])}, "
            f"positives={np.sum((train_y[:, i] > 0.5) & (train_m[:, i] > 0.5))}"
        )

    return train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m


def compute_pos_weights(train_y: np.ndarray, train_m: np.ndarray, n_h: int, cap: float = 40.0):
    """Per-horizon neg/pos ratio for BCE pos_weight (handles class imbalance)."""
    w = np.ones(n_h, dtype=np.float32)
    for i in range(n_h):
        sel = train_m[:, i] > 0.5
        if sel.sum() < 2:
            continue
        y = train_y[sel, i]
        pos = float(np.sum(y > 0.5))
        neg = float(np.sum(y <= 0.5))
        if pos < 1e-6:
            continue
        w[i] = min(neg / pos, cap)
    return torch.tensor(w, dtype=torch.float32)


def masked_bce_loss(logits, targets, mask, pos_weight: Optional[torch.Tensor] = None):
    """BCE with logits, averaged only over mask==1; optional per-head pos_weight."""
    if mask.sum() < 1:
        return logits.sum() * 0.0
    kwargs = {"reduction": "none"}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight.to(logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, **kwargs)
    loss = loss * mask
    return loss.sum() / mask.sum()


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    pos_weight: torch.Tensor,
    n_epochs=50,
    use_amp=True,
    input_noise_std: float = 0.02,
    max_grad_norm: float = 1.0,
):
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5, verbose=True
    )
    use_amp = use_amp and device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast_ctx = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast_ctx = None

    best_val = float("inf")
    patience_c = 0
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y, batch_m in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_m = batch_m.to(device, non_blocking=True)

            if model.training and input_noise_std > 0:
                batch_x = batch_x + torch.randn_like(batch_x) * input_noise_std

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with autocast_ctx():
                    logits = model(batch_x)
                    loss = masked_bce_loss(logits, batch_y, batch_m, pos_weight)
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x)
                loss = masked_bce_loss(logits, batch_y, batch_m, pos_weight)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        vb = 0
        with torch.inference_mode():
            for batch_x, batch_y, batch_m in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_m = batch_m.to(device, non_blocking=True)
                logits = model(batch_x)
                val_loss += masked_bce_loss(logits, batch_y, batch_m, pos_weight).item()
                vb += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val = val_loss / max(vb, 1)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1}/{n_epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience_c = 0
            torch.save(model.state_dict(), "best_elm_survival_horizon_lstm.pth")
            print("  ✓ Saved best model")
        else:
            patience_c += 1
        if patience_c >= 12:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses


def evaluate_horizons(model, test_loader, device, horizons_ms):
    model.eval()
    all_logits, all_y, all_m = [], [], []
    with torch.inference_mode():
        for batch_x, batch_y, batch_m in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
            all_y.append(batch_y.numpy())
            all_m.append(batch_m.numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = np.concatenate(all_y, axis=0)
    m = np.concatenate(all_m, axis=0)

    print("\n--- Per-horizon (masked samples) ---")
    for i, h in enumerate(horizons_ms):
        sel = m[:, i] > 0.5
        if sel.sum() < 2 or len(np.unique(y[sel, i])) < 2:
            print(f"  H={h}ms: insufficient data for AUROC")
            continue
        auc = roc_auc_score(y[sel, i], probs[sel, i])
        pred = (probs[sel, i] >= 0.5).astype(int)
        acc = accuracy_score(y[sel, i], pred)
        print(f"  H={h}ms: AUROC={auc:.4f}  Acc@0.5={acc:.4f}  n={sel.sum()}")

    return probs, y, m


def plot_results(train_losses, val_losses, horizons_ms):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(train_losses, label="Train")
    ax[0].plot(val_losses, label="Val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Masked BCE")
    ax[0].set_title("ELM survival-horizon LSTM loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].axis("off")
    ax[1].text(
        0.05,
        0.85,
        "Targets: P(first ELM within H ms | 200 ms history)\n"
        + "Mask: full horizon observable in shot (censoring).\n"
        + f"Horizons (ms): {horizons_ms}",
        fontsize=11,
        family="monospace",
        va="top",
    )
    plt.tight_layout()
    out = "elm_survival_horizon_lstm_results.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out}")


def main():
    print("=" * 60)
    print("ELM survival-horizon LSTM (Perry et al. 2024-style outputs on DIII-D labels)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    horizons_ms = DEFAULT_HORIZONS_MS

    (
        X1,
        X2,
        y,
        times,
        shots,
        branch1_features,
        branch2_features,
    ) = load_and_prepare_data()

    train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m = (
        create_survival_horizon_windows(
            X1, X2, y, times, shots, horizons_ms=horizons_ms
        )
    )

    n_h = len(horizons_ms)
    pos_weight = compute_pos_weights(train_y, train_m, n_h)
    print(f"Per-horizon pos_weight (capped): {pos_weight.numpy().round(3)}")

    train_ds = SurvivalHorizonDataset(train_x, train_y, train_m)
    val_ds = SurvivalHorizonDataset(val_x, val_y, val_m)
    test_ds = SurvivalHorizonDataset(test_x, test_y, test_m)

    nw = 4 if torch.cuda.is_available() else 0
    pm = device.type == "cuda"
    batch_size = 512
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=nw > 0,
    )

    n_features = len(branch1_features) + len(branch2_features)
    model = SurvivalHorizonLSTM(
        n_features=n_features,
        horizons=horizons_ms,
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        bx, _, _ = next(iter(train_loader))
        _ = model(bx.to(device))
    print(f"Sanity forward: {time.time()-t0:.3f}s")

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
        pos_weight=pos_weight.to(device),
        n_epochs=50,
        input_noise_std=0.02,
        max_grad_norm=1.0,
    )

    model.load_state_dict(
        torch.load("best_elm_survival_horizon_lstm.pth", map_location=device)
    )
    evaluate_horizons(model, test_loader, device, horizons_ms)
    plot_results(train_losses, val_losses, horizons_ms)

    print("\nDone. Weights: best_elm_survival_horizon_lstm.pth")


if __name__ == "__main__":
    main()
