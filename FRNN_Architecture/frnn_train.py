"""
Training entry-point for the FRNN-TCN binary plasma-state classifier.

Reproduces the user's spec:
  * Per-signal z-score normalization fit on TRAIN shots only (in frnn_data).
  * Strict causal windowing (in frnn_data).
  * Channel dropout p=0.1 inside the model (Dropout1d on input channels).
  * Weighted cross-entropy with sklearn-balanced class weights from the train
    split.
  * Adam, lr=9.08e-5.
  * Exponential LR decay with gamma=0.99 per epoch ("0.99 decay").
  * Best model checkpoint chosen on validation accuracy + early stopping.
  * Threshold tuning on val (max F1) and final report on the held-out test set.

Run:
    cd /mnt/homes/sr4240/my_folder
    python3 FRNN_Architecture/frnn_train.py
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from frnn_data import (
    PlasmaWindowDataset,
    build_window_splits,
    class_weights,
    load_plasma_data,
)
from frnn_eval import (
    collect_predictions,
    find_optimal_threshold,
    plot_curves,
    predict_with_threshold,
    report_metrics,
    transition_analysis,
)
from frnn_model import FRNN_TCN


SEED = 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FRNN-TCN binary plasma classifier.")
    p.add_argument("--window-size", type=int, default=150)
    p.add_argument("--horizon-ms", type=int, default=150,
                   help="forecasting horizon in milliseconds (1ms-sampled signals)")
    p.add_argument("--n-classes", type=int, default=2,
                   help="2 = binary {Suppressed, ELMy}; 4 = {Suppressed, Dithering, ELMing, Mitigated}")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=9.08e-5)
    p.add_argument("--lr-decay", type=float, default=0.99,
                   help="ExponentialLR gamma applied per epoch")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-oversample", action="store_true",
                   help="disable transition-aware oversampling on train split")
    p.add_argument("--save-dir", type=str, default="/mnt/homes/sr4240/my_folder/FRNN_Architecture")
    p.add_argument("--tag", type=str, default="frnn_tcn_binary")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"best_{args.tag}.pth")
    plot_path = os.path.join(args.save_dir, f"{args.tag}_results.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("FRNN-TCN Plasma Classifier  |  training run")
    print("=" * 60)
    print(f"window_size={args.window_size}, horizon={args.horizon_ms}ms, "
          f"n_classes={args.n_classes}")
    print(f"lr={args.lr:.3e}, gamma={args.lr_decay}, batch_size={args.batch_size}, "
          f"epochs={args.epochs}, patience={args.patience}")

    X, y, t, s, feats = load_plasma_data()
    splits = build_window_splits(
        X, y, t, s,
        window_size=args.window_size,
        prediction_horizon_ms=args.horizon_ms,
        binary=(args.n_classes == 2),
        oversample_transitions=not args.no_oversample,
    )
    splits.feature_names = feats

    train_ds = PlasmaWindowDataset(splits.train)
    val_ds = PlasmaWindowDataset(splits.val)
    test_ds = PlasmaWindowDataset(splits.test, also_return_current_state=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    cw = class_weights(splits.train.labels, n_classes=args.n_classes)
    print(f"\nClass weights (balanced): {cw.tolist()}")
    cw_tensor = torch.from_numpy(cw).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    model = FRNN_TCN(n_features=len(feats), n_classes=args.n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    print("\nWarmup forward-pass timing on a real batch ...")
    sample_x, _ = next(iter(train_loader))
    sample_x = sample_x.to(device)
    torch.cuda.synchronize() if pin else None
    t0 = time.time()
    with torch.no_grad():
        _ = model(sample_x)
    torch.cuda.synchronize() if pin else None
    print(f"  forward time for batch of {sample_x.size(0)}: {time.time() - t0:.3f}s")

    best_val_acc = -1.0
    patience = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta); val_accs.append(va)
        cur_lr = optimizer.param_groups[0]["lr"]
        flag = ""
        if va > best_val_acc:
            best_val_acc = va
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = " *best"
        else:
            patience += 1

        print(f"epoch {epoch:>3d}/{args.epochs} | "
              f"train_loss={tl:.4f} train_acc={ta:.4f} | "
              f"val_loss={vl:.4f} val_acc={va:.4f} | "
              f"lr={cur_lr:.2e} | {dt:.1f}s{flag}")

        if patience >= args.patience:
            print(f"early stop at epoch {epoch} (no val_acc gain in {patience} epochs)")
            break

    print(f"\nBest val_acc = {best_val_acc:.4f}; checkpoint -> {ckpt_path}")

    print("\nLoading best checkpoint and evaluating on TEST split ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    if args.n_classes == 2:
        val_out = collect_predictions(model, val_loader, device, return_current_states=False)
        threshold, val_f1 = find_optimal_threshold(val_out.probs, val_out.labels)
        print(f"Optimal val threshold (max F1): {threshold:.4f}  (val F1={val_f1:.4f})")
    else:
        threshold = None

    test_out = collect_predictions(model, test_loader, device, return_current_states=True)
    if threshold is not None:
        thresholded_preds = predict_with_threshold(test_out.probs, threshold)
    else:
        thresholded_preds = test_out.preds

    if args.n_classes == 2:
        class_names = ["Suppressed", "ELMy (Dith/ELM/Mit)"]
    else:
        class_names = ["Suppressed", "Dithering", "ELMing", "Mitigated"]

    report_metrics(thresholded_preds, test_out.labels, test_out.probs, class_names,
                   title="Test set metrics")
    if test_out.current_states is not None:
        transition_analysis(thresholded_preds, test_out.labels, test_out.current_states,
                            test_out.probs, class_names, horizon_ms=args.horizon_ms)

    final_acc = accuracy_score(test_out.labels, thresholded_preds)
    print(f"\nFINAL TEST ACCURACY: {final_acc:.4f}")

    plot_curves(
        train_losses, val_losses, train_accs, val_accs,
        thresholded_preds, test_out.labels, class_names, plot_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
