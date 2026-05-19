"""
FRNN-TCN classifier for plasma instability forecasting on DIII-D.

This is the **TCN trunk** within the Tang/PPPL Fusion Recurrent Neural Network
(FRNN) framework -- the canonical FRNN code (PPPLDeepLearning/plasma-python,
``builder.py``) makes the trunk a configurable choice between stacked LSTMs and
a packed TCN. We use the TCN variant introduced in Dong et al., 2021 because:

  * Tang et al., 2022 ("Implementation of AI/DEEP Learning Disruption Predictor
    into a Plasma Control System", arXiv:2204.01289) adds *signals and a
    sensitivity/saliency head* to the same FRNN framework -- it is not a new
    trunk architecture.
  * For 0D scalar-only inputs (no radial profiles), the FRNN spatial-conv block
    collapses, so the FRNN-LSTM trunk reduces to a ~2x200 stacked LSTM, which
    is essentially the architecture LSTM/LSTM_50_Binary_Transitions.py already
    uses; that defeats the goal of "different architecture, better performance".
  * Dilated causal TCN is a strictly different architecture from LSTM and
    typically wins on fixed-window forecasting at this scale (parallel training,
    long receptive field, no vanishing gradients).

Targets adapt FRNN's per-timestep time-to-disruption regression to a discrete
classification head over the **final causal time step** of the window:
  * Class 0 = Suppressed
  * Class 1 = Dithering / ELMing / Mitigated  ("ELMy" in the user's spec)

Inputs match LSTM/LSTM_50_Binary_Transitions.py:
  6 scalar 1ms-sampled signals  ->  (B, n_features=6, T=window_size)

Architecture (default hyper-params reproduce the user's spec):
  - Channel dropout (p=0.1) zeros entire signal channels.
  - Spatial conv block: Ns=2 layers of Conv2d with kernel (Ks=7, 1) acting only
    across the feature axis at each time step. nsf=20 filters.
  - 1x1 bottleneck flattens (nsf, n_features) into ntf=60 channels.
  - Temporal TCN: NTstack=2 stacks of Nt=8 residual blocks. Each block has two
    causal Conv1d layers (Kt=11) with exponentially-growing dilation 2**i,
    ReLU, dropout=0.05, and a residual skip connection.
  - Classification head: take the last (strictly causal) time step and project
    to n_classes logits. Use with nn.CrossEntropyLoss (implicit softmax).

NOTE on n_classes: the spec contains a tension ("2-class classification ...
4-way softmax head"). We default to n_classes=2 to match the original LSTM
script's outputs; pass n_classes=4 to recover the full {Suppressed, Dithering,
ELMing, Mitigated} softmax instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D conv with strictly-causal padding: output[t] depends only on input[<=t]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """TCN residual block (Bai et al., 2018 style) with two causal dilated convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(F.relu(self.conv1(x)))
        out = self.dropout2(F.relu(self.conv2(out)))
        return F.relu(out + residual)


class FRNN_TCN(nn.Module):
    """FRNN-TCN binary plasma-state classifier (Dong et al., 2021)."""

    def __init__(
        self,
        n_features: int = 6,
        n_classes: int = 2,
        Ns: int = 2,
        Ks: int = 7,
        nsf: int = 20,
        Nt: int = 8,
        NTstack: int = 2,
        Kt: int = 11,
        ntf: int = 60,
        dropout: float = 0.05,
        channel_dropout: float = 0.1,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self._nsf = nsf

        self.channel_dropout = nn.Dropout1d(channel_dropout)

        spatial_layers = []
        in_ch = 1
        for _ in range(Ns):
            spatial_layers += [
                nn.Conv2d(in_ch, nsf, kernel_size=(Ks, 1), padding=(Ks // 2, 0)),
                nn.ReLU(inplace=True),
            ]
            in_ch = nsf
        self.spatial_conv = nn.Sequential(*spatial_layers)

        self.bottleneck = nn.Conv1d(nsf * n_features, ntf, kernel_size=1)

        blocks = []
        for _ in range(NTstack):
            for layer_idx in range(Nt):
                blocks.append(
                    TemporalBlock(
                        in_channels=ntf,
                        out_channels=ntf,
                        kernel_size=Kt,
                        dilation=2 ** layer_idx,
                        dropout=dropout,
                    )
                )
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Linear(ntf, n_classes)

        self.receptive_field = 1 + 2 * (Kt - 1) * NTstack * (2 ** Nt - 1)

        if verbose:
            self._print_summary()

    def _print_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        spatial = sum(p.numel() for p in self.spatial_conv.parameters())
        bottle = sum(p.numel() for p in self.bottleneck.parameters())
        tcn = sum(p.numel() for p in self.tcn.parameters())
        head = sum(p.numel() for p in self.head.parameters())

        print("=" * 60)
        print("FRNN-TCN Plasma Classifier (Dong et al., 2021 architecture)")
        print("=" * 60)
        print(f"Total parameters: {total:,}")
        print(f"  Spatial conv:     {spatial:,} ({100 * spatial / total:5.2f}%)")
        print(f"  1x1 bottleneck:   {bottle:,} ({100 * bottle / total:5.2f}%)")
        print(f"  Temporal TCN:     {tcn:,} ({100 * tcn / total:5.2f}%)")
        print(f"  Classifier head:  {head:,} ({100 * head / total:5.2f}%)")
        print(f"Causal receptive field (timesteps): {self.receptive_field:,}")
        print("=" * 60)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, C, T), got shape {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {C}")

        x = self.channel_dropout(x)

        x = x.unsqueeze(1)
        x = self.spatial_conv(x)
        x = x.reshape(B, self._nsf * self.n_features, T)

        x = F.relu(self.bottleneck(x))

        x = self.tcn(x)

        last = x[:, :, -1]
        return self.head(last)


def _smoke_test() -> None:
    print("\n" + "=" * 60)
    print("FRNN-TCN forward-pass smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)
    B, C, T = 8, 6, 150
    n_classes = 2

    model = FRNN_TCN(n_features=C, n_classes=n_classes).to(device)

    x = torch.randn(B, C, T, device=device)
    print(f"\nInput shape:  {tuple(x.shape)}")

    model.eval()
    with torch.no_grad():
        logits = model(x)

    expected = (B, n_classes)
    assert logits.shape == expected, f"Expected {expected}, got {tuple(logits.shape)}"
    print(f"Output shape: {tuple(logits.shape)}  (expected {expected})")
    print(f"Output dtype: {logits.dtype}")

    probs = F.softmax(logits, dim=1)
    print(f"Softmax row sums: {probs.sum(dim=1).detach().cpu().tolist()}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(B, device=device), atol=1e-5)

    print("\nBackward-pass check (loss + grads):")
    model.train()
    target = torch.randint(0, n_classes, (B,), device=device)
    loss = F.cross_entropy(model(x), target)
    loss.backward()
    grad_present = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    grad_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  loss = {loss.item():.4f}")
    print(f"  parameters with non-zero grad: {grad_present}/{grad_total}")
    assert grad_present == grad_total, "Some parameters received no gradient."

    print("\nStrict-causality check (perturbing the future must not change the past):")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x1 = torch.randn(2, C, T, device=device)
    x2 = x1.clone()
    half = T // 2
    x2[:, :, half:] = torch.randn_like(x2[:, :, half:])

    def features(xx: torch.Tensor) -> torch.Tensor:
        h = model.spatial_conv(xx.unsqueeze(1)).reshape(xx.shape[0], -1, xx.shape[2])
        h = F.relu(model.bottleneck(h))
        return model.tcn(h)

    with torch.no_grad():
        h1, h2 = features(x1), features(x2)
    diff_past = (h1 - h2)[:, :, :half].abs().max().item()
    diff_future = (h1 - h2)[:, :, half:].abs().max().item()
    print(f"  Max |Delta| over t <  T/2: {diff_past:.2e}")
    print(f"  Max |Delta| over t >= T/2: {diff_future:.2e}")
    assert diff_past < 1e-5, "Causality violated: past features depend on the future."
    assert diff_future > 0, "Sanity check failed: future perturbation had no effect anywhere."
    print("  Strict causality verified.")

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    _smoke_test()
