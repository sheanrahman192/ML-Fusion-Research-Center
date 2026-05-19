"""
Apples-to-apples comparison: evaluate the existing LSTM checkpoint and the
freshly-trained iTransformer checkpoint on the SAME test split, using the
SAME threshold-tuning procedure.

Both scripts use np.random.seed(42) for the shot shuffle, so the train/val/test
shot membership is identical -- only the model differs.
"""
import os, sys, time
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# Reproduce the same seeds as both scripts
np.random.seed(48)
torch.manual_seed(48)
if torch.cuda.is_available():
    torch.cuda.manual_seed(48)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/mnt/homes/sr4240/my_folder'

# Import the iTransformer module
spec = importlib.util.spec_from_file_location(
    'itrans', os.path.join(HERE, 'iTransformer_50_Binary_Transitions.py')
)
itrans = importlib.util.module_from_spec(spec)
spec.loader.exec_module(itrans)

# Import the LSTM module
spec = importlib.util.spec_from_file_location(
    'lstm_mod', os.path.join(ROOT, 'LSTM', 'LSTM_50_Binary_Transitions.py')
)
lstm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lstm_mod)


def predict_with_threshold(model, loader, device, threshold):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            out = model(batch_X)
            p = torch.softmax(out, dim=1)
            pos = p[:, 1].cpu().numpy()
            preds.extend((pos >= threshold).astype(int))
            labels.extend(batch_y.numpy())
            probs.extend(p.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)


def find_threshold(model, loader, device):
    model.eval()
    pos_probs, lbls = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            p = torch.softmax(model(batch_X), dim=1)
            pos_probs.extend(p[:, 1].cpu().numpy())
            lbls.extend(batch_y.numpy())
    pos_probs = np.array(pos_probs); lbls = np.array(lbls)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (pos_probs >= t).astype(int)
        if len(np.unique(pred)) > 1:
            f = f1_score(lbls, pred, average='weighted')
            if f > best_f1:
                best_f1, best_t = f, t
    return best_t


def metrics_dict(y, yhat, probs, current_states):
    pos = probs[:, 1]
    out = {
        'accuracy': accuracy_score(y, yhat),
        'precision_w': precision_score(y, yhat, average='weighted', zero_division=0),
        'recall_w': recall_score(y, yhat, average='weighted', zero_division=0),
        'f1_w': f1_score(y, yhat, average='weighted', zero_division=0),
        'precision_macro': precision_score(y, yhat, average='macro', zero_division=0),
        'recall_macro': recall_score(y, yhat, average='macro', zero_division=0),
        'f1_macro': f1_score(y, yhat, average='macro', zero_division=0),
        'roc_auc': roc_auc_score(y, pos) if len(np.unique(y)) > 1 else float('nan'),
        # Per-class
        'precision_supp': precision_score(y, yhat, pos_label=0, zero_division=0),
        'recall_supp': recall_score(y, yhat, pos_label=0, zero_division=0),
        'f1_supp': f1_score(y, yhat, pos_label=0, zero_division=0),
        'precision_dem': precision_score(y, yhat, pos_label=1, zero_division=0),
        'recall_dem': recall_score(y, yhat, pos_label=1, zero_division=0),
        'f1_dem': f1_score(y, yhat, pos_label=1, zero_division=0),
    }
    # Transition-specific
    tr_mask = current_states != y
    if tr_mask.sum() > 0:
        out['transition_n'] = int(tr_mask.sum())
        out['transition_acc'] = accuracy_score(y[tr_mask], yhat[tr_mask])
        out['transition_f1'] = f1_score(y[tr_mask], yhat[tr_mask], average='weighted', zero_division=0)
        # Problematic 0->1 transitions
        prob_mask = (current_states == 0) & (y == 1)
        if prob_mask.sum() > 0:
            out['supp_to_dem_n'] = int(prob_mask.sum())
            out['supp_to_dem_acc'] = accuracy_score(y[prob_mask], yhat[prob_mask])
        else:
            out['supp_to_dem_n'] = 0; out['supp_to_dem_acc'] = float('nan')
        # Reverse 1->0 transitions
        rev_mask = (current_states == 1) & (y == 0)
        if rev_mask.sum() > 0:
            out['dem_to_supp_n'] = int(rev_mask.sum())
            out['dem_to_supp_acc'] = accuracy_score(y[rev_mask], yhat[rev_mask])
        else:
            out['dem_to_supp_n'] = 0; out['dem_to_supp_acc'] = float('nan')
    return out


def main():
    # GPU is congested by other jobs -- evaluation forward-pass on CPU is fine
    # for ~24k windows. Use FORCE_CPU=0 env var to override if you want GPU.
    if os.environ.get('FORCE_CPU', '1') == '1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n[1/4] Building data pipeline (matches both scripts exactly)...")
    t0 = time.time()
    X, y, times, shots, features, scaler = itrans.load_and_prepare_data()
    (train_X, train_y, train_cs, val_X, val_y, val_cs,
     test_X, test_y, test_cs) = itrans.create_windows_with_random_shot_split(
        X, y, times, shots, prediction_horizon_ms=50
    )
    print(f"   Done in {time.time() - t0:.1f}s")
    print(f"   Train/Val/Test: {len(train_X)}/{len(val_X)}/{len(test_X)}")

    eval_batch = 512 if device.type == 'cpu' else 2048
    val_loader = DataLoader(itrans.PlasmaDataset(val_X, val_y), batch_size=eval_batch, shuffle=False)
    test_loader = DataLoader(itrans.PlasmaDataset(test_X, test_y), batch_size=eval_batch, shuffle=False)

    # ---- LSTM evaluation ----
    print("\n[2/4] Evaluating LSTM checkpoint...")
    lstm_path = os.path.join(ROOT, 'best_lstm_50ms_binary_transitions_supp_vs_dem.pth')
    print(f"   Loading: {lstm_path}")
    lstm = lstm_mod.LSTMFirstNN(n_features=len(features), n_classes=2).to(device)
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    t_lstm = find_threshold(lstm, val_loader, device)
    lstm_pred, lstm_y, lstm_probs = predict_with_threshold(lstm, test_loader, device, t_lstm)
    lstm_metrics = metrics_dict(lstm_y, lstm_pred, lstm_probs, test_cs)
    lstm_metrics['threshold'] = t_lstm
    lstm_metrics['n_params'] = sum(p.numel() for p in lstm.parameters())
    print(f"   LSTM threshold: {t_lstm:.4f}")
    print(f"   LSTM test accuracy: {lstm_metrics['accuracy']:.4f}")

    # ---- iTransformer evaluation ----
    print("\n[3/4] Evaluating iTransformer checkpoint...")
    itrans_path = os.path.join(HERE, 'best_itransformer_50ms_binary_transitions_supp_vs_dem.pth')
    if not os.path.exists(itrans_path):
        print(f"   ERROR: {itrans_path} not found. Did training finish?")
        return
    print(f"   Loading: {itrans_path}")
    it = itrans.iTransformerClassifier(
        n_features=len(features), seq_len=150, n_classes=2,
        d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2,
    ).to(device)
    it.load_state_dict(torch.load(itrans_path, map_location=device))
    t_it = find_threshold(it, val_loader, device)
    it_pred, it_y, it_probs = predict_with_threshold(it, test_loader, device, t_it)
    it_metrics = metrics_dict(it_y, it_pred, it_probs, test_cs)
    it_metrics['threshold'] = t_it
    it_metrics['n_params'] = sum(p.numel() for p in it.parameters())
    print(f"   iTransformer threshold: {t_it:.4f}")
    print(f"   iTransformer test accuracy: {it_metrics['accuracy']:.4f}")

    # ---- Comparison ----
    print("\n[4/4] Comparison")
    print("=" * 86)
    print(f"{'Metric':<32} {'LSTM':>14} {'iTransformer':>16} {'Δ (iT - LSTM)':>16}")
    print("-" * 86)
    keys_overall = [
        ('accuracy', 'Test accuracy'),
        ('roc_auc', 'ROC AUC'),
        ('f1_w', 'F1 (weighted)'),
        ('f1_macro', 'F1 (macro)'),
        ('precision_w', 'Precision (weighted)'),
        ('recall_w', 'Recall (weighted)'),
    ]
    keys_perclass = [
        ('precision_supp', 'Precision Suppressed'),
        ('recall_supp', 'Recall Suppressed'),
        ('f1_supp', 'F1 Suppressed'),
        ('precision_dem', 'Precision Dith/ELM/Mit'),
        ('recall_dem', 'Recall Dith/ELM/Mit'),
        ('f1_dem', 'F1 Dith/ELM/Mit'),
    ]
    keys_trans = [
        ('transition_acc', 'Transition accuracy'),
        ('transition_f1', 'Transition F1 (weighted)'),
        ('supp_to_dem_acc', '0->1 (Supp->Dith) accuracy'),
        ('dem_to_supp_acc', '1->0 (Dith->Supp) accuracy'),
    ]

    def row(label, k):
        a, b = lstm_metrics.get(k, float('nan')), it_metrics.get(k, float('nan'))
        if isinstance(a, float) and isinstance(b, float):
            d = b - a
            print(f"{label:<32} {a:>14.4f} {b:>16.4f} {d:>+16.4f}")
        else:
            print(f"{label:<32} {a!s:>14} {b!s:>16} {'':>16}")

    print("\n  Overall:")
    for k, lab in keys_overall:
        row(lab, k)

    print("\n  Per-class:")
    for k, lab in keys_perclass:
        row(lab, k)

    print("\n  Transition cases:")
    for k, lab in keys_trans:
        row(lab, k)
    print(f"  (#transition: {lstm_metrics.get('transition_n', 0)},  "
          f"#0->1: {lstm_metrics.get('supp_to_dem_n', 0)},  "
          f"#1->0: {lstm_metrics.get('dem_to_supp_n', 0)})")

    print("\n  Other:")
    print(f"  {'Decision threshold':<32} {lstm_metrics['threshold']:>14.4f} {it_metrics['threshold']:>16.4f}")
    print(f"  {'# parameters':<32} {lstm_metrics['n_params']:>14,} {it_metrics['n_params']:>16,}")
    print("=" * 86)

    # ---- Side-by-side confusion matrix figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    class_names = ['Suppressed', 'Dith/ELM/Mit']
    for ax, (name, ypred, ytrue) in zip(
        axes,
        [('LSTM', lstm_pred, lstm_y), ('iTransformer', it_pred, it_y)]
    ):
        cm = confusion_matrix(ytrue, ypred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    vmin=0, vmax=1, cbar=False)
        ax.set_title(f'{name}  (acc={accuracy_score(ytrue, ypred):.4f})')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.suptitle('Normalized Confusion Matrices - Test Set (50ms binary)')
    plt.tight_layout()
    out_png = os.path.join(HERE, 'lstm_vs_itransformer_comparison.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"\n  Comparison plot: {out_png}")

    # Also save a metrics table to CSV-like text
    out_txt = os.path.join(HERE, 'lstm_vs_itransformer_metrics.txt')
    with open(out_txt, 'w') as f:
        f.write("Metric,LSTM,iTransformer,Delta\n")
        for k, lab in keys_overall + keys_perclass + keys_trans:
            a = lstm_metrics.get(k, float('nan'))
            b = it_metrics.get(k, float('nan'))
            if isinstance(a, float) and isinstance(b, float):
                f.write(f"{lab},{a:.4f},{b:.4f},{b-a:+.4f}\n")
        f.write(f"Threshold,{lstm_metrics['threshold']:.4f},{it_metrics['threshold']:.4f},\n")
        f.write(f"Parameters,{lstm_metrics['n_params']},{it_metrics['n_params']},\n")
    print(f"  Metrics table:    {out_txt}")


if __name__ == '__main__':
    main()
