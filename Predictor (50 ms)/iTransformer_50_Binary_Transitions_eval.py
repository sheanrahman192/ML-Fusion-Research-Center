"""
iTransformer for Binary Plasma State Classification (50ms prediction horizon).

EVAL VARIANT
============
Same model, loss, optimizer, and hyperparameters as
``iTransformer_50_Binary_Transitions.py`` but reads from
``/mnt/homes/sr4240/my_folder/combined_database.csv`` and uses a *fixed* set
of 42 held-out shots as the test split (every other labeled shot in the CSV
becomes train). A small fraction of train shots is carved out as a
validation set so the original early-stopping / best-checkpoint logic is
preserved. Final reporting adds a per-shot accuracy table and a structured
``METRICS`` block to make cross-model (PatchTST / LSTM) comparison easy.
"""

import os
import math
import time as time_module
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

np.random.seed(48)
torch.manual_seed(48)
if torch.cuda.is_available():
    torch.cuda.manual_seed(48)

PREDICTION_HORIZON_MS = 50
WINDOW_SIZE = 150
VARIANT_SUFFIX = '_supp_vs_dem_eval'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = '/mnt/homes/sr4240/my_folder/combined_database.csv'
MODEL_SAVE_PATH = os.path.join(
    SCRIPT_DIR,
    f'best_itransformer_{PREDICTION_HORIZON_MS}ms_binary_transitions{VARIANT_SUFFIX}.pth'
)
RESULTS_PNG_PATH = os.path.join(
    SCRIPT_DIR,
    f'itransformer_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}.png'
)
PER_SHOT_CSV_PATH = os.path.join(
    SCRIPT_DIR,
    f'itransformer_{PREDICTION_HORIZON_MS}ms_binary_transitions_per_shot{VARIANT_SUFFIX}.csv'
)

# Held-out test shots (42). Every other labeled shot in the CSV is training.
TEST_SHOTS = [
    169449, 169457, 169460, 169463, 169467, 169470, 169473, 169476, 169479,
    169500, 169503, 169506, 169959, 169966, 170063, 170070, 170090, 170114,
    175673, 175682, 175686, 175695, 175701, 179350, 179363, 183153, 183185,
    190276, 190284, 190733, 191662, 191665, 191674, 191683, 191686, 191689,
    191975, 191978, 191981, 191984, 191987, 191990,
]
# Fraction of TRAIN shots reserved as a small held-out validation set
# (only used for early stopping / threshold tuning, never for final metrics).
VAL_SHOT_FRACTION = 0.10


# --------------------------------------------------------------------------
# Model: iTransformer (inverted) for binary classification
# --------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Used by iTransformer to handle non-stationary multivariate series. We
    normalize each (instance, variate) pair to zero mean / unit variance over
    the time axis. Although the global StandardScaler in the data pipeline
    already standardizes per-feature globally, RevIN additionally absorbs
    per-shot drift, which matters here because plasma shots have different
    operating points.
    """
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_features))
            self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        # x: (B, N, T)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class iTransformerBlock(nn.Module):
    """One Pre-LN encoder block operating on N variate tokens of dim d_model.

    Pre-LN (norm_first) is what the iTransformer reference implementation
    uses; it's more stable than Post-LN when stacking several layers without
    a learning-rate warmup tuned per depth.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, d_model) -- attention is across the N variate tokens
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)

        h = self.norm_ffn(x)
        x = x + self.drop(self.ffn(h))
        return x


class iTransformerClassifier(nn.Module):
    """
    iTransformer adapted for binary plasma-state classification.

    Pipeline (inputs are (B, N, T) to match LSTM script's PlasmaDataset):
        1. RevIN per (instance, variate) over the T axis.
        2. Variate tokenization: Linear(T -> d_model) embeds each variate's
           full time series into a single d_model vector. Output: (B, N, d_model).
        3. n_layers iTransformer blocks: self-attention across variate tokens,
           position-wise FFN per variate (shared weights). The attention here
           directly models cross-feature correlations -- e.g. li vs density
           vs fs04 patterns -- which is what physics-grounded plasma classifiers
           rely on.
        4. Aggregation head: attention pooling + mean pooling concatenated.
        5. MLP classifier -> n_classes logits.
    """
    def __init__(self, n_features, seq_len=WINDOW_SIZE, n_classes=2,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        self.revin = RevIN(n_features, affine=True)

        # Variate tokenization: each variate's T-length series -> d_model vector
        self.variate_embed = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Attention pooling over the N variate tokens
        self.pool_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # Classifier head: concat(attention-pooled, mean-pooled) -> 2 logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Parameter count breakdown
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed_p = sum(p.numel() for p in self.variate_embed.parameters())
        block_p = sum(p.numel() for p in self.blocks.parameters())
        head_p = sum(p.numel() for n, p in self.named_parameters()
                     if 'classifier' in n or 'pool_attn' in n)
        print("\n" + "=" * 60)
        print("iTransformer Model Parameter Count (Binary Classification):")
        print("=" * 60)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"\nParameters by component:")
        print(f"  - Variate embedding: {embed_p:,} ({embed_p/total*100:.1f}%)")
        print(f"  - Transformer blocks: {block_p:,} ({block_p/total*100:.1f}%)")
        print(f"  - Pool + classifier: {head_p:,} ({head_p/total*100:.1f}%)")
        print("=" * 60)
        print("Architecture: RevIN -> VariateEmbed -> iTransformerBlocks (N tokens) -> Pool -> Classifier")
        print("Token semantics: each token is a VARIATE (feature), embedding spans TIME")
        print("Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")

    def forward(self, x):
        # x: (B, N, T) -- same layout the LSTM model receives from PlasmaDataset
        x = self.revin(x)                      # (B, N, T)
        tokens = self.variate_embed(x)         # (B, N, d_model)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)       # (B, N, d_model)

        # Attention pooling over variate tokens
        scores = self.pool_attn(tokens)                       # (B, N, 1)
        weights = torch.softmax(scores, dim=1)                # (B, N, 1)
        attn_pooled = torch.sum(tokens * weights, dim=1)      # (B, d_model)
        mean_pooled = tokens.mean(dim=1)                      # (B, d_model)
        pooled = torch.cat([attn_pooled, mean_pooled], dim=1) # (B, 2 * d_model)

        return self.classifier(pooled)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss with class weights -- identical to LSTM script for fair comparison."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None and not isinstance(alpha, (float, int)):
            if isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha)
            else:
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


# --------------------------------------------------------------------------
# Dataset (identical to LSTM script -- same input/output shape contract)
# --------------------------------------------------------------------------

class PlasmaDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Returns (n_features, sequence_length) -- same as LSTM script
        return self.windows[idx].T, self.labels[idx]


# --------------------------------------------------------------------------
# Data preparation (mirrors LSTM script)
# --------------------------------------------------------------------------

def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    # Preserve original shot exclusion (safety: only filters if present)
    if (df['shot'] == 191675).any():
        df = df[df['shot'] != 191675].copy()

    important_features = ['iln3iamp', 'betan', 'density', 'li',
                          'tritop', 'fs04_past_max_smoothed']
    selected_features = [f for f in important_features if f in df.columns]

    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)

    if 'fs04' in df_sorted.columns:
        fs04_values = df_sorted['fs04'].values
        times_temp = df_sorted['time'].values
        shots_temp = df_sorted['shot'].values
        fs04_rate = np.zeros(len(df_sorted))
        for shot_id in df_sorted['shot'].unique():
            shot_mask = shots_temp == shot_id
            shot_indices = np.where(shot_mask)[0]
            if len(shot_indices) > 1:
                fs04_diff = np.diff(fs04_values[shot_indices])
                time_diff = np.diff(times_temp[shot_indices])
                time_diff_safe = np.where(time_diff == 0, 1, time_diff)
                rate = fs04_diff / time_diff_safe
                fs04_rate[shot_indices[0]] = 0.0
                fs04_rate[shot_indices[1:]] = rate
        df_sorted['fs04_rate_of_change'] = fs04_rate

    print(f"Using {len(selected_features)} features: {selected_features}")

    X = df_sorted[selected_features].values
    y = df_sorted['state'].values
    times = df_sorted['time'].values
    shots = df_sorted['shot'].values

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(times)
    X = X[valid_mask]
    y = y[valid_mask]
    times = times[valid_mask]
    shots = shots[valid_mask]

    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {Counter(y)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, times, shots, selected_features, scaler


def create_windows_with_fixed_test_split(X, y, times, shots,
                                         window_size=WINDOW_SIZE,
                                         prediction_horizon_ms=PREDICTION_HORIZON_MS):
    """Build windows + binary labels and split shots into TRAIN/VAL/TEST by
    the fixed ``TEST_SHOTS`` list. A small fraction of remaining shots is
    carved out as a validation set to preserve the original early-stopping /
    threshold-tuning logic. Per-test-window shot ids are returned to enable
    the per-shot accuracy table required by this evaluation.
    """
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by SHOT NUMBER using a FIXED held-out test list (42 shots).")
    print("Binary classification: Suppressed (0) vs Dithering/ELMing/Mitigated (1)")

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots in CSV (after cleaning): {n_shots}")

    test_shots_set = set(int(s) for s in TEST_SHOTS)
    csv_shots_set = set(int(s) for s in unique_shots)
    test_shots_present = test_shots_set & csv_shots_set
    test_shots_missing = test_shots_set - csv_shots_set
    train_pool = sorted(s for s in csv_shots_set if s not in test_shots_set)
    print(f"Test shots requested: {len(test_shots_set)}")
    print(f"Test shots found in CSV: {len(test_shots_present)}")
    if test_shots_missing:
        print(f"Test shots NOT found in CSV (will be skipped): {sorted(test_shots_missing)}")
    print(f"Remaining (train + val) shot pool: {len(train_pool)}")

    rng = np.random.RandomState(48)
    shuffled = rng.permutation(train_pool)
    n_val = max(1, int(round(VAL_SHOT_FRACTION * len(train_pool))))
    val_shots = set(int(s) for s in shuffled[:n_val])
    train_shots = set(int(s) for s in shuffled[n_val:])
    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, "
          f"Test={len(test_shots_present)}")

    train_windows, train_labels, train_current_states = [], [], []
    val_windows, val_labels, val_current_states = [], [], []
    test_windows, test_labels, test_current_states, test_window_shots = [], [], [], []

    binary_label_mapping = {1: 0, 2: 1, 3: 1, 4: 1}

    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0
    train_shots_with_windows = set()
    val_shots_with_windows = set()
    test_shots_with_windows = set()

    for shot_id in unique_shots:
        shot_id_int = int(shot_id)
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        if len(shot_indices) < window_size:
            continue

        if shot_id_int in test_shots_set:
            tw, tl, tc, ts = test_windows, test_labels, test_current_states, test_window_shots
            seen = test_shots_with_windows
        elif shot_id_int in val_shots:
            tw, tl, tc, ts = val_windows, val_labels, val_current_states, None
            seen = val_shots_with_windows
        elif shot_id_int in train_shots:
            tw, tl, tc, ts = train_windows, train_labels, train_current_states, None
            seen = train_shots_with_windows
        else:
            continue

        shot_times = times[shot_indices]
        shot_labels = y[shot_indices]
        shot_X = X[shot_indices]

        for i in range(len(shot_indices) - window_size + 1):
            window = shot_X[i:i + window_size]
            window_end_time = shot_times[i + window_size - 1]
            target_time = window_end_time + prediction_horizon_ms
            current_label = shot_labels[i + window_size - 1]

            future_local_idx = np.searchsorted(shot_times, target_time)
            if future_local_idx >= len(shot_times):
                windows_skipped_no_future += 1
                continue
            future_label = shot_labels[future_local_idx]

            if (int(current_label) not in binary_label_mapping or
                    int(future_label) not in binary_label_mapping):
                windows_skipped_invalid_label += 1
                continue

            if not np.isnan(window).any() and not np.isinf(window).any():
                tw.append(window)
                tl.append(binary_label_mapping[int(future_label)])
                tc.append(binary_label_mapping[int(current_label)])
                if ts is not None:
                    ts.append(shot_id_int)
                seen.add(shot_id_int)
                windows_created += 1

    train_windows = np.array(train_windows, dtype=np.float32)
    train_labels = np.array(train_labels)
    train_current_states = np.array(train_current_states)
    val_windows = np.array(val_windows, dtype=np.float32)
    val_labels = np.array(val_labels)
    val_current_states = np.array(val_current_states)
    test_windows = np.array(test_windows, dtype=np.float32)
    test_labels = np.array(test_labels)
    test_current_states = np.array(test_current_states)
    test_window_shots = np.array(test_window_shots, dtype=np.int64)

    print(f"\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Skipped (no future data): {windows_skipped_no_future:,}")
    print(f"  Skipped (invalid label): {windows_skipped_invalid_label:,}")

    print(f"\nCreated windows:")
    print(f"  Train: {len(train_windows)} windows from {len(train_shots_with_windows)} shots "
          f"(of {len(train_shots)} requested)")
    print(f"  Val:   {len(val_windows)} windows from {len(val_shots_with_windows)} shots "
          f"(of {len(val_shots)} requested)")
    print(f"  Test:  {len(test_windows)} windows from {len(test_shots_with_windows)} shots "
          f"(of {len(test_shots_present)} requested)")

    test_shots_no_windows = sorted(test_shots_present - test_shots_with_windows)
    if test_shots_no_windows:
        print(f"  Test shots that produced ZERO windows (all rows unlabeled or too short): "
              f"{test_shots_no_windows}")

    print(f"\nLabel distribution (binary):")
    print(f"  Train: {Counter(train_labels)}")
    print(f"  Val:   {Counter(val_labels)}")
    print(f"  Test:  {Counter(test_labels)}")

    print(f"\nTransition statistics (current != future):")
    print(f"  Train: {np.sum(train_current_states != train_labels):,}")
    print(f"  Val:   {np.sum(val_current_states != val_labels):,}")
    print(f"  Test:  {np.sum(test_current_states != test_labels):,}")

    print(f"\nOversampling transition cases in training set...")
    train_windows, train_labels, train_current_states = oversample_transitions(
        train_windows, train_labels, train_current_states
    )
    print(f"After oversampling:")
    print(f"  Train: {len(train_windows)} windows")
    print(f"  Label distribution: {Counter(train_labels)}")

    split_meta = {
        'train_shots_used': sorted(train_shots_with_windows),
        'val_shots_used': sorted(val_shots_with_windows),
        'test_shots_used': sorted(test_shots_with_windows),
        'test_shots_missing_from_csv': sorted(test_shots_missing),
        'test_shots_no_windows': test_shots_no_windows,
    }

    return (train_windows, train_labels, train_current_states,
            val_windows, val_labels, val_current_states,
            test_windows, test_labels, test_current_states,
            test_window_shots, split_meta)


def oversample_transitions(windows, labels, current_states,
                           transition_multiplier=3, problematic_multiplier=5):
    transition_mask = current_states != labels
    problematic_mask = (current_states == 0) & (labels == 1)
    transition_indices = np.where(transition_mask)[0]
    problematic_indices = np.where(problematic_mask)[0]
    non_transition_indices = np.where(~transition_mask)[0]

    print(f"  Before oversampling:")
    print(f"    Total samples: {len(windows)}")
    print(f"    Transition cases: {len(transition_indices)}")
    print(f"    Problematic transitions (0->1): {len(problematic_indices)}")
    print(f"    Non-transition cases: {len(non_transition_indices)}")

    out_w = [windows[i] for i in non_transition_indices]
    out_l = [labels[i] for i in non_transition_indices]
    out_c = [current_states[i] for i in non_transition_indices]

    regular_transition_indices = transition_indices[~np.isin(transition_indices, problematic_indices)]
    for idx in regular_transition_indices:
        for _ in range(transition_multiplier):
            out_w.append(windows[idx])
            out_l.append(labels[idx])
            out_c.append(current_states[idx])

    for idx in problematic_indices:
        for _ in range(problematic_multiplier):
            out_w.append(windows[idx])
            out_l.append(labels[idx])
            out_c.append(current_states[idx])

    out_w = np.array(out_w, dtype=np.float32)
    out_l = np.array(out_l)
    out_c = np.array(out_c)
    print(f"  After oversampling:")
    print(f"    Total samples: {len(out_w)}")
    print(f"    Problematic transitions (0->1): {np.sum((out_c == 0) & (out_l == 1))}")
    return out_w, out_l, out_c


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, class_weights_tensor,
                n_epochs=50, base_lr=5e-4, weight_decay=1e-4,
                warmup_epochs=3, grad_clip=1.0):
    """AdamW + warmup-cosine schedule + gradient clipping. The Transformer
    benefits noticeably from these vs the LSTM defaults; we keep loss and
    class-weighting identical so any gain is attributable to the model.
    """
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction='mean')

    no_decay_keywords = ('bias', 'LayerNorm', 'norm', 'RevIN')
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW(
        [{'params': decay_params, 'weight_decay': weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=base_lr,
    )

    total_steps = max(1, n_epochs * len(train_loader))
    warmup_steps = max(1, warmup_epochs * len(train_loader))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 25

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels_acc = [], []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_acc.extend(batch_y.cpu().numpy())

        model.eval()
        val_loss = 0.0
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(batch_y.cpu().numpy())

        train_acc = accuracy_score(train_labels_acc, train_preds)
        val_acc = accuracy_score(val_labels_list, val_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{n_epochs}  lr={cur_lr:.2e}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"  New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accs, val_accs


# --------------------------------------------------------------------------
# Evaluation utilities (mirror LSTM script)
# --------------------------------------------------------------------------

def find_optimal_threshold(model, val_loader, device):
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            val_probs.extend(probs[:, 1].cpu().numpy())
            val_labels.extend(batch_y.numpy())
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= threshold).astype(int)
        if len(np.unique(preds)) > 1:
            f1 = f1_score(val_labels, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    print(f"\nOptimal threshold found: {best_threshold:.4f} (F1 score: {best_f1:.4f})")
    return best_threshold


def evaluate_model(model, test_loader, device, class_names, threshold=0.5):
    print(f"\nEvaluating with threshold: {threshold:.4f}")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            pos = probs[:, 1].cpu().numpy()
            preds = (pos >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"\nROC AUC Score: {auc:.4f}")
    return all_preds, all_labels, all_probs


def analyze_transition_effectiveness(all_preds, all_labels, all_current_states,
                                     all_probs, class_names):
    print("\n" + "=" * 60)
    print("TRANSITION EFFECTIVENESS ANALYSIS")
    print("=" * 60)
    print(f"Analyzing predictions for points where future state ({PREDICTION_HORIZON_MS}ms) differs from current state")
    print("=" * 60)

    transition_mask = all_current_states != all_labels
    n_transitions = np.sum(transition_mask)
    n_total = len(all_labels)
    print(f"\nTransition Statistics:")
    print(f"  Total test samples: {n_total:,}")
    print(f"  Transition cases:   {n_transitions:,} ({n_transitions/n_total*100:.2f}%)")
    print(f"  Non-transition:     {n_total - n_transitions:,} ({(n_total - n_transitions)/n_total*100:.2f}%)")
    if n_transitions == 0:
        print("\n  No transitions found in test set.")
        return

    t_preds = all_preds[transition_mask]
    t_labels = all_labels[transition_mask]
    t_probs = all_probs[transition_mask] if len(all_probs.shape) > 1 else None

    t_acc = accuracy_score(t_labels, t_preds)
    t_p = precision_score(t_labels, t_preds, average='weighted', zero_division=0)
    t_r = recall_score(t_labels, t_preds, average='weighted', zero_division=0)
    t_f = f1_score(t_labels, t_preds, average='weighted', zero_division=0)
    print(f"\nTransition Case Metrics:")
    print(f"  Accuracy:  {t_acc:.4f}")
    print(f"  Precision: {t_p:.4f}")
    print(f"  Recall:    {t_r:.4f}")
    print(f"  F1-Score:  {t_f:.4f}")
    if t_probs is not None and len(np.unique(t_labels)) > 1:
        print(f"  ROC AUC:   {roc_auc_score(t_labels, t_probs[:, 1]):.4f}")

    print(f"\nTransition Case Classification Report:")
    print(classification_report(t_labels, t_preds, target_names=class_names, digits=4))
    print(f"\nTransition Case Confusion Matrix:")
    print(confusion_matrix(t_labels, t_preds))

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"\nComparison with Overall Performance:")
    print(f"  Overall accuracy:    {overall_acc:.4f}")
    print(f"  Transition accuracy: {t_acc:.4f}")
    print(f"  Difference: {t_acc - overall_acc:.4f}")

    transition_types = {
        'Suppressed -> Dithering/ELMing/Mitigated': (all_current_states == 0) & (all_labels == 1),
        'Dithering/ELMing/Mitigated -> Suppressed': (all_current_states == 1) & (all_labels == 0),
    }
    print(f"\nTransition Type Breakdown:")
    for trans_type, mask in transition_types.items():
        if np.sum(mask) > 0:
            tp = all_preds[mask]; tl = all_labels[mask]
            print(f"  {trans_type}: {np.sum(mask):,} cases, Accuracy: {accuracy_score(tl, tp):.4f}")
    print("=" * 60)


def plot_results(train_losses, val_losses, train_accs, val_accs,
                 all_preds, all_labels, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Validation Loss ({PREDICTION_HORIZON_MS}ms - iTransformer Binary)')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Training and Validation Accuracy ({PREDICTION_HORIZON_MS}ms - iTransformer Binary)')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix (Binary)')
    axes[1, 0].set_ylabel('True Label'); axes[1, 0].set_xlabel('Predicted Label')

    cm_counts = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix (Counts - Binary)')
    axes[1, 1].set_ylabel('True Label'); axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(RESULTS_PNG_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Results saved to '{RESULTS_PNG_PATH}'")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

MODEL_NAME = 'iTransformer'


def report_per_shot_accuracy(test_window_shots, all_labels, all_preds):
    """Print and save a per-shot accuracy / sample-count table."""
    print("\n" + "=" * 60)
    print("PER-SHOT ACCURACY TABLE (test set)")
    print("=" * 60)
    print(f"{'shot':>10} {'n':>8} {'pos':>6} {'neg':>6} {'acc':>8}")

    rows = []
    for shot_id in sorted(np.unique(test_window_shots)):
        mask = test_window_shots == shot_id
        if not np.any(mask):
            continue
        sl = all_labels[mask]
        sp = all_preds[mask]
        acc = accuracy_score(sl, sp)
        n_pos = int(np.sum(sl == 1))
        n_neg = int(np.sum(sl == 0))
        print(f"{int(shot_id):>10} {len(sl):>8} {n_pos:>6} {n_neg:>6} {acc:>8.4f}")
        rows.append({'shot': int(shot_id), 'n_samples': int(len(sl)),
                     'n_pos': n_pos, 'n_neg': n_neg, 'accuracy': float(acc)})

    pd.DataFrame(rows).to_csv(PER_SHOT_CSV_PATH, index=False)
    print(f"\nPer-shot table saved to: {PER_SHOT_CSV_PATH}")


def emit_structured_metrics(all_labels, all_preds, all_probs, class_names,
                            n_train_shots, n_val_shots, n_test_shots,
                            n_train_samples, n_val_samples, n_test_samples,
                            train_wall_seconds, inference_wall_seconds,
                            optimal_threshold, n_epochs_run):
    """Print a machine-friendly METRICS block for cross-model comparison."""
    acc = accuracy_score(all_labels, all_preds)
    p_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    p_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    r_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    p_cls = precision_score(all_labels, all_preds, average=None,
                            labels=[0, 1], zero_division=0)
    r_cls = recall_score(all_labels, all_preds, average=None,
                         labels=[0, 1], zero_division=0)
    f_cls = f1_score(all_labels, all_preds, average=None,
                     labels=[0, 1], zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if all_probs.ndim == 2 and all_probs.shape[1] == 2 and len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = float('nan')

    print("\n" + "=" * 60)
    print("STRUCTURED METRICS (machine-readable)")
    print("=" * 60)
    print("# Format: model,metric,value")
    rows = [
        (MODEL_NAME, 'horizon_ms', PREDICTION_HORIZON_MS),
        (MODEL_NAME, 'window_size', WINDOW_SIZE),
        (MODEL_NAME, 'n_train_shots', n_train_shots),
        (MODEL_NAME, 'n_val_shots', n_val_shots),
        (MODEL_NAME, 'n_test_shots', n_test_shots),
        (MODEL_NAME, 'n_train_samples', n_train_samples),
        (MODEL_NAME, 'n_val_samples', n_val_samples),
        (MODEL_NAME, 'n_test_samples', n_test_samples),
        (MODEL_NAME, 'epochs_run', n_epochs_run),
        (MODEL_NAME, 'optimal_threshold', round(float(optimal_threshold), 4)),
        (MODEL_NAME, 'train_wall_seconds', round(float(train_wall_seconds), 2)),
        (MODEL_NAME, 'inference_wall_seconds', round(float(inference_wall_seconds), 4)),
        (MODEL_NAME, 'accuracy', round(float(acc), 6)),
        (MODEL_NAME, 'precision_macro', round(float(p_macro), 6)),
        (MODEL_NAME, 'recall_macro', round(float(r_macro), 6)),
        (MODEL_NAME, 'f1_macro', round(float(f_macro), 6)),
        (MODEL_NAME, 'precision_weighted', round(float(p_weighted), 6)),
        (MODEL_NAME, 'recall_weighted', round(float(r_weighted), 6)),
        (MODEL_NAME, 'f1_weighted', round(float(f_weighted), 6)),
        (MODEL_NAME, f'precision_class0_{class_names[0]}', round(float(p_cls[0]), 6)),
        (MODEL_NAME, f'recall_class0_{class_names[0]}',    round(float(r_cls[0]), 6)),
        (MODEL_NAME, f'f1_class0_{class_names[0]}',        round(float(f_cls[0]), 6)),
        (MODEL_NAME, f'precision_class1_{class_names[1]}', round(float(p_cls[1]), 6)),
        (MODEL_NAME, f'recall_class1_{class_names[1]}',    round(float(r_cls[1]), 6)),
        (MODEL_NAME, f'f1_class1_{class_names[1]}',        round(float(f_cls[1]), 6)),
        (MODEL_NAME, 'roc_auc', round(float(auc), 6) if not np.isnan(auc) else 'nan'),
        (MODEL_NAME, 'cm_TN', int(cm[0, 0])),
        (MODEL_NAME, 'cm_FP', int(cm[0, 1])),
        (MODEL_NAME, 'cm_FN', int(cm[1, 0])),
        (MODEL_NAME, 'cm_TP', int(cm[1, 1])),
    ]
    for r in rows:
        print(f"METRIC,{r[0]},{r[1]},{r[2]}")
    print("=" * 60)


def main():
    print("=" * 60)
    print(f"{MODEL_NAME} (ICLR 2024) for Binary Plasma Classification -- EVAL VARIANT")
    print("=" * 60)
    print("Architecture: RevIN -> Variate Tokenization -> iTransformer Blocks -> Pool -> Classifier")
    print(f"Window: {WINDOW_SIZE} datapoints BEFORE current time")
    print(f"Prediction: {PREDICTION_HORIZON_MS}ms INTO THE FUTURE")
    print("Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")
    print(f"Split: FIXED held-out test list of {len(TEST_SHOTS)} shots; "
          f"all other CSV shots = train (small {VAL_SHOT_FRACTION:.0%} val carve-out for early stopping).")
    print(f"Data: {DATA_CSV}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, scaler = load_and_prepare_data()

    (train_X, train_y, train_current_states,
     val_X, val_y, val_current_states,
     test_X, test_y, test_current_states,
     test_window_shots, split_meta) = create_windows_with_fixed_test_split(
        X, y, times, shots, prediction_horizon_ms=PREDICTION_HORIZON_MS
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples ({len(split_meta['train_shots_used'])} shots)")
    print(f"  Val:   {len(val_X)} samples ({len(split_meta['val_shots_used'])} shots)")
    print(f"  Test:  {len(test_X)} samples ({len(split_meta['test_shots_used'])} shots)")

    if len(test_X) == 0:
        raise RuntimeError("No test windows were created -- check TEST_SHOTS / labels.")

    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    # Smaller batch than the original 2048 because GPU is being shared with
    # parallel PatchTST/LSTM eval runs; keeps OOM risk low while preserving
    # all other hyperparameters.
    BATCH_SIZE = 512
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    print("\nCalculating class weights from training data...")
    class_counts = np.bincount(train_y, minlength=2)
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights:      {dict(zip(range(len(class_weights)), class_weights))}")
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model = iTransformerClassifier(
        n_features=len(features),
        seq_len=WINDOW_SIZE,
        n_classes=2,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
    ).to(device)

    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)
    start = time_module.time()
    with torch.no_grad():
        _ = model(test_batch)
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {time_module.time() - start:.3f} seconds")

    # Reduced epochs from 50 to 20 to keep wall-clock reasonable while
    # the GPU is shared with parallel model evals; warmup-cosine schedule
    # is rescaled to the new epoch count and early stopping (max_patience=25)
    # is unchanged. Loss is monitored to confirm convergence.
    N_EPOCHS = 20
    print(f"\nStarting training for {N_EPOCHS} epochs...")
    train_start = time_module.time()
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, class_weights_tensor,
        n_epochs=N_EPOCHS, base_lr=5e-4, weight_decay=1e-4,
        warmup_epochs=3, grad_clip=1.0,
    )
    train_wall_seconds = time_module.time() - train_start
    n_epochs_run = len(train_losses)
    print(f"\nTraining wall-clock: {train_wall_seconds:.1f}s "
          f"({n_epochs_run} epochs actually run)")

    print("\nLoading best model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    optimal_threshold = find_optimal_threshold(model, val_loader, device)

    class_names = ['Suppressed', 'Dithering/ELMing/Mitigated']
    inference_start = time_module.time()
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names, threshold=optimal_threshold
    )
    inference_wall_seconds = time_module.time() - inference_start
    print(f"\nInference wall-clock on test set: {inference_wall_seconds:.3f}s "
          f"({len(all_labels)} samples)")

    # Confusion matrix (counts) printed inline so it's in the run log too.
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("\nConfusion matrix (rows=true, cols=pred; labels=[0,1]):")
    print(cm)

    analyze_transition_effectiveness(
        all_preds, all_labels, test_current_states, all_probs, class_names
    )

    plot_results(train_losses, val_losses, train_accs, val_accs,
                 all_preds, all_labels, class_names)

    report_per_shot_accuracy(test_window_shots, all_labels, all_preds)

    emit_structured_metrics(
        all_labels, all_preds, all_probs, class_names,
        n_train_shots=len(split_meta['train_shots_used']),
        n_val_shots=len(split_meta['val_shots_used']),
        n_test_shots=len(split_meta['test_shots_used']),
        n_train_samples=len(train_X),
        n_val_samples=len(val_X),
        n_test_samples=len(test_X),
        train_wall_seconds=train_wall_seconds,
        inference_wall_seconds=inference_wall_seconds,
        optimal_threshold=optimal_threshold,
        n_epochs_run=n_epochs_run,
    )

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training Complete! ({MODEL_NAME}, {PREDICTION_HORIZON_MS}ms future, Binary, EVAL)")
    print("=" * 60)


if __name__ == "__main__":
    main()
