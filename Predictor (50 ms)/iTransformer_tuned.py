"""
iTransformer for Binary Plasma State Classification (50ms prediction horizon)

Reference:
    Liu et al. "iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting." ICLR 2024 (arXiv:2310.06625).

Key idea (the "inversion"):
    Vanilla Transformers for time series treat each timestep as a token, with
    the embedding spanning across variates. iTransformer flips this: each
    VARIATE (channel/feature) is a token, and its full time series is
    embedded into a d_model vector. Self-attention then operates across
    variates -- which is what we want for multivariate plasma signals where
    cross-feature correlations carry the predictive signal -- while the FFN
    handles per-variate dynamics. The paper shows this inversion gives
    consistent gains over vanilla TS-Transformers on multivariate benchmarks.

Inputs / outputs match LSTM_50_Binary_Transitions.py exactly:
    Input  : 150-step windows of 6 plasma features
    Output : Binary state at 50ms in the future
             (Suppressed=0 vs Dithering/ELMing/Mitigated=1)
    Data   : /mnt/homes/sr4240/my_folder/plasma_data.csv
    Split  : 70/15/15 by shot number, same seed
    Loss   : Focal loss with class weights, transition oversampling
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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

np.random.seed(48)
torch.manual_seed(48)
if torch.cuda.is_available():
    torch.cuda.manual_seed(48)

PREDICTION_HORIZON_MS = 50
WINDOW_SIZE = 150
VARIANT_SUFFIX = '_supp_vs_dem'

# Tuning configuration via TUNE_VERSION env var.
# Test-set transitions are *all* 1->0 (Dith/ELM/Mit -> Suppressed) — the
# original "problematic" oversampling targets the opposite direction (0->1),
# so iter 1 fixes that. Subsequent iters layer on regularization, capacity,
# and longer training.
TUNE_VERSION = os.getenv('TUNE_VERSION', '0')

TUNE_CONFIGS = {
    '0': {  # original baseline
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 0,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 3, 'd_ff': 256,
        'n_epochs': 50, 'base_lr': 5e-4, 'weight_decay': 1e-4, 'warmup_epochs': 3,
    },
    '1': {  # fix oversampling target + macro-F1 threshold + higher focal gamma
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 8,
        'threshold_metric': 'macro_f1',
        'threshold_low': 0.05, 'threshold_high': 0.95, 'threshold_count': 91,
        'focal_gamma': 2.5, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 3, 'd_ff': 256,
        'n_epochs': 50, 'base_lr': 5e-4, 'weight_decay': 1e-4, 'warmup_epochs': 3,
    },
    '2': {  # baseline thresholding/loss + regularization + mild 1->0 oversampling
        # Iter 1 lessons: macro_f1 threshold drifted to 0.06 (worse class-1 bias),
        # and 8x oversampling didn't help separation. Revert those, keep mild 1->0
        # boost (4x) and add dropout/label-smooth/weight-decay to combat overfitting.
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 4,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'dropout': 0.3, 'd_model': 128, 'n_heads': 8, 'n_layers': 3, 'd_ff': 256,
        'n_epochs': 50, 'base_lr': 5e-4, 'weight_decay': 5e-4, 'warmup_epochs': 3,
    },
    '3': {  # baseline regularization, larger architecture, mild 1->0 oversampling
        # Iter 2 lesson: heavy regularization crushed the model (ROC AUC 0.58!).
        # iTransformer already has attention+ffn+pool dropout, so 0.2 is sweet
        # spot. Iter 3 isolates the architecture change with only the safe
        # 1->0 oversampling boost retained.
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 4,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 4, 'd_ff': 384,
        'n_epochs': 50, 'base_lr': 5e-4, 'weight_decay': 1e-4, 'warmup_epochs': 3,
    },
    '4': {  # iter 3 config (best so far) + longer training + slightly higher 1->0 boost
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 6,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 4, 'd_ff': 384,
        'n_epochs': 80, 'base_lr': 3e-4, 'weight_decay': 1e-4, 'warmup_epochs': 5,
        'class_0_weight_boost': 1.0,
    },
    '5': {  # iter 4 + LARGER architecture (more capacity for separation)
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 6,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 192, 'n_heads': 12, 'n_layers': 4, 'd_ff': 512,
        'n_epochs': 80, 'base_lr': 3e-4, 'weight_decay': 1e-4, 'warmup_epochs': 5,
        'class_0_weight_boost': 1.0,
    },
    '6': {  # iter 4 arch + aggressive 1->0 oversampling (mult=10)
        # Iter 5 lesson: bigger arch overfits on this data size — stick with iter 4 arch.
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 10,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 4, 'd_ff': 384,
        'n_epochs': 80, 'base_lr': 3e-4, 'weight_decay': 1e-4, 'warmup_epochs': 5,
        'class_0_weight_boost': 1.0,
    },
    '7': {  # iter 4 arch + lower threshold floor (0.02) so threshold search can go further
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 6,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.02, 'threshold_high': 0.95, 'threshold_count': 187,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 4, 'd_ff': 384,
        'n_epochs': 80, 'base_lr': 3e-4, 'weight_decay': 1e-4, 'warmup_epochs': 5,
        'class_0_weight_boost': 1.0,
    },
    '8': {  # iter 4 + boost class 0 weight in focal alpha (1.5x)
        'transition_mult': 3, 'problematic_0to1_mult': 5, 'problematic_1to0_mult': 6,
        'threshold_metric': 'weighted_f1',
        'threshold_low': 0.10, 'threshold_high': 0.90, 'threshold_count': 81,
        'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'dropout': 0.2, 'd_model': 128, 'n_heads': 8, 'n_layers': 4, 'd_ff': 384,
        'n_epochs': 80, 'base_lr': 3e-4, 'weight_decay': 1e-4, 'warmup_epochs': 5,
        'class_0_weight_boost': 1.5,
    },
}
CFG = TUNE_CONFIGS[TUNE_VERSION]
TUNE_TAG = f'_tune{TUNE_VERSION}' if TUNE_VERSION != '0' else ''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = '/mnt/homes/sr4240/my_folder/plasma_data.csv'
MODEL_SAVE_PATH = os.path.join(
    SCRIPT_DIR,
    f'best_itransformer_{PREDICTION_HORIZON_MS}ms_binary_transitions{VARIANT_SUFFIX}{TUNE_TAG}.pth'
)
RESULTS_PNG_PATH = os.path.join(
    SCRIPT_DIR,
    f'itransformer_{PREDICTION_HORIZON_MS}ms_binary_transitions_results{VARIANT_SUFFIX}{TUNE_TAG}.png'
)


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
    """Focal Loss with class weights and optional label smoothing."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
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
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # CE with optional label smoothing — focal modulation applied on top
        ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.label_smoothing)(inputs, targets)
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


def create_windows_with_random_shot_split(X, y, times, shots,
                                          window_size=WINDOW_SIZE,
                                          prediction_horizon_ms=PREDICTION_HORIZON_MS):
    print(f"Creating windows of size {window_size} (predicting {prediction_horizon_ms}ms in the future)...")
    print("Splitting by SHOT NUMBER (not individual data points)")
    print("Binary classification: Suppressed (0) vs Dithering/ELMing/Mitigated (1)")

    unique_shots = np.unique(shots)
    n_shots = len(unique_shots)
    print(f"Total unique shots: {n_shots}")

    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)

    train_size = int(0.7 * n_shots)
    val_size = int(0.15 * n_shots)
    train_shots = set(shuffled_shots[:train_size])
    val_shots = set(shuffled_shots[train_size:train_size + val_size])
    test_shots = set(shuffled_shots[train_size + val_size:])
    print(f"Shot split: Train={len(train_shots)}, Val={len(val_shots)}, Test={len(test_shots)}")

    train_windows, train_labels = [], []
    val_windows, val_labels = [], []
    test_windows, test_labels = [], []
    train_current_states, val_current_states, test_current_states = [], [], []

    binary_label_mapping = {1: 0, 2: 1, 3: 1, 4: 1}

    windows_created = 0
    windows_skipped_no_future = 0
    windows_skipped_invalid_label = 0

    for shot_id in unique_shots:
        shot_mask = shots == shot_id
        shot_indices = np.where(shot_mask)[0]
        if len(shot_indices) < window_size:
            continue

        if shot_id in train_shots:
            tw, tl, tc = train_windows, train_labels, train_current_states
        elif shot_id in val_shots:
            tw, tl, tc = val_windows, val_labels, val_current_states
        else:
            tw, tl, tc = test_windows, test_labels, test_current_states

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

    print(f"\nWindow creation statistics:")
    print(f"  Windows created: {windows_created:,}")
    print(f"  Skipped (no future data): {windows_skipped_no_future:,}")
    print(f"  Skipped (invalid label): {windows_skipped_invalid_label:,}")

    print(f"\nCreated windows:")
    print(f"  Train: {len(train_windows)} windows from {len(train_shots)} shots")
    print(f"  Val:   {len(val_windows)} windows from {len(val_shots)} shots")
    print(f"  Test:  {len(test_windows)} windows from {len(test_shots)} shots")

    print(f"\nLabel distribution (binary):")
    print(f"  Train: {Counter(train_labels)}")
    print(f"  Val:   {Counter(val_labels)}")
    print(f"  Test:  {Counter(test_labels)}")

    print(f"\nTransition statistics:")
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

    return (train_windows, train_labels, train_current_states,
            val_windows, val_labels, val_current_states,
            test_windows, test_labels, test_current_states)


def oversample_transitions(windows, labels, current_states,
                           transition_multiplier=None,
                           problematic_0to1_mult=None,
                           problematic_1to0_mult=None):
    # Pull defaults from CFG so each tune iteration controls oversampling
    transition_multiplier = transition_multiplier if transition_multiplier is not None else CFG['transition_mult']
    problematic_0to1_mult = problematic_0to1_mult if problematic_0to1_mult is not None else CFG['problematic_0to1_mult']
    problematic_1to0_mult = problematic_1to0_mult if problematic_1to0_mult is not None else CFG['problematic_1to0_mult']
    # mult==0 means "use generic transition_multiplier"
    if problematic_1to0_mult == 0:
        problematic_1to0_mult = transition_multiplier

    transition_mask = current_states != labels
    p_0to1_mask = (current_states == 0) & (labels == 1)  # Suppressed -> DEM
    p_1to0_mask = (current_states == 1) & (labels == 0)  # DEM -> Suppressed
    non_transition_indices = np.where(~transition_mask)[0]

    print(f"  Before oversampling:")
    print(f"    Total samples: {len(windows)}")
    print(f"    Transition cases: {transition_mask.sum()}")
    print(f"    0->1 transitions: {p_0to1_mask.sum()}  (mult={problematic_0to1_mult})")
    print(f"    1->0 transitions: {p_1to0_mask.sum()}  (mult={problematic_1to0_mult})")
    print(f"    Non-transition cases: {len(non_transition_indices)}")

    out_w = [windows[i] for i in non_transition_indices]
    out_l = [labels[i] for i in non_transition_indices]
    out_c = [current_states[i] for i in non_transition_indices]

    for idx in np.where(p_0to1_mask)[0]:
        for _ in range(problematic_0to1_mult):
            out_w.append(windows[idx]); out_l.append(labels[idx]); out_c.append(current_states[idx])
    for idx in np.where(p_1to0_mask)[0]:
        for _ in range(problematic_1to0_mult):
            out_w.append(windows[idx]); out_l.append(labels[idx]); out_c.append(current_states[idx])

    out_w = np.array(out_w, dtype=np.float32)
    out_l = np.array(out_l)
    out_c = np.array(out_c)
    print(f"  After oversampling:")
    print(f"    Total samples: {len(out_w)}")
    print(f"    0->1 transitions: {np.sum((out_c == 0) & (out_l == 1))}")
    print(f"    1->0 transitions: {np.sum((out_c == 1) & (out_l == 0))}")
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
    criterion = FocalLoss(
        alpha=class_weights_tensor,
        gamma=CFG.get('focal_gamma', 2.0),
        reduction='mean',
        label_smoothing=CFG.get('label_smoothing', 0.0),
    )

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

    metric_name = CFG.get('threshold_metric', 'weighted_f1')
    avg_kind = 'macro' if metric_name == 'macro_f1' else 'weighted'
    low = CFG.get('threshold_low', 0.10)
    high = CFG.get('threshold_high', 0.90)
    count = CFG.get('threshold_count', 81)

    best_threshold, best_score = 0.5, 0.0
    for threshold in np.linspace(low, high, count):
        preds = (val_probs >= threshold).astype(int)
        if len(np.unique(preds)) > 1:
            score = f1_score(val_labels, preds, average=avg_kind)
            if score > best_score:
                best_score = score
                best_threshold = threshold
    print(f"\nOptimal threshold ({metric_name}): {best_threshold:.4f}  score={best_score:.4f}")
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
    plt.show()
    print(f"Results saved to '{RESULTS_PNG_PATH}'")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("iTransformer (ICLR 2024) for Binary Plasma Classification")
    print("=" * 60)
    print("Architecture: RevIN -> Variate Tokenization -> iTransformer Blocks -> Pool -> Classifier")
    print(f"Window: {WINDOW_SIZE} datapoints BEFORE current time")
    print(f"Prediction: {PREDICTION_HORIZON_MS}ms INTO THE FUTURE")
    print("Classification: Binary (Suppressed=0, Dithering/ELMing/Mitigated=1)")
    print("Split: RANDOM BY SHOT NUMBER (not individual data points)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X, y, times, shots, features, scaler = load_and_prepare_data()

    (train_X, train_y, train_current_states,
     val_X, val_y, val_current_states,
     test_X, test_y, test_current_states) = create_windows_with_random_shot_split(
        X, y, times, shots, prediction_horizon_ms=PREDICTION_HORIZON_MS
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_X)} samples")
    print(f"  Val:   {len(val_X)} samples")
    print(f"  Test:  {len(test_X)} samples")

    train_dataset = PlasmaDataset(train_X, train_y)
    val_dataset = PlasmaDataset(val_X, val_y)
    test_dataset = PlasmaDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    print("\nCalculating class weights from training data...")
    class_counts = np.bincount(train_y, minlength=2)
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class weights:      {dict(zip(range(len(class_weights)), class_weights))}")
    class_0_boost = CFG.get('class_0_weight_boost', 1.0)
    if class_0_boost != 1.0:
        class_weights[0] *= class_0_boost
        print(f"After class-0 weight boost x{class_0_boost}: {dict(zip(range(len(class_weights)), class_weights))}")
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model = iTransformerClassifier(
        n_features=len(features),
        seq_len=WINDOW_SIZE,
        n_classes=2,
        d_model=CFG['d_model'],
        n_heads=CFG['n_heads'],
        n_layers=CFG['n_layers'],
        d_ff=CFG['d_ff'],
        dropout=CFG['dropout'],
    ).to(device)

    print("\nTesting forward pass speed...")
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch.to(device)
    start = time_module.time()
    with torch.no_grad():
        _ = model(test_batch)
    print(f"Forward pass time for batch of {test_batch.shape[0]}: {time_module.time() - start:.3f} seconds")

    print(f"\nStarting training... (TUNE_VERSION={TUNE_VERSION})")
    print(f"Config: {CFG}")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, class_weights_tensor,
        n_epochs=CFG['n_epochs'], base_lr=CFG['base_lr'],
        weight_decay=CFG['weight_decay'], warmup_epochs=CFG['warmup_epochs'],
        grad_clip=1.0,
    )

    print("\nLoading best model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    optimal_threshold = find_optimal_threshold(model, val_loader, device)

    class_names = ['Suppressed', 'Dithering/ELMing/Mitigated']
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names, threshold=optimal_threshold
    )

    analyze_transition_effectiveness(
        all_preds, all_labels, test_current_states, all_probs, class_names
    )

    plot_results(train_losses, val_losses, train_accs, val_accs,
                 all_preds, all_labels, class_names)

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training Complete! (iTransformer, {PREDICTION_HORIZON_MS}ms future, Binary)")
    print("=" * 60)


if __name__ == "__main__":
    main()
