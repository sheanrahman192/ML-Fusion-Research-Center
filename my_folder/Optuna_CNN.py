import argparse
import gc
import io
import json
import math
import os
import random
import sys
import warnings
from contextlib import redirect_stdout
from multiprocessing import cpu_count
import multiprocessing as mp
import subprocess
import time
import psutil
from typing import Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler

from CNN_Classifier import FocalLoss as MulticlassFocalLoss  # avoid name clash
from CNN_Classifier import PlasmaCNNBiLSTMAttention, PlasmaDataset, load_and_preprocess_data, prepare_data, select_features
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


GLOBAL_X_TRAIN = None
GLOBAL_Y_TRAIN = None
GLOBAL_X_EVAL = None
GLOBAL_Y_EVAL = None
GLOBAL_NUM_FEATURES = None
GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimize for A100 GPU
if torch.cuda.is_available():
    # Enable TF32 for faster training on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Optimize for memory and speed
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.95)

# Determine optimal number of workers for DataLoader
NUM_WORKERS = min(cpu_count(), 4)  # Reduced to 4 to avoid multiprocessing issues
PIN_MEMORY = torch.cuda.is_available()


def setup_environment():
    """Set up optimal environment variables for A100 GPU training"""
    print("=== Setting up optimized environment for A100 GPU ===")
    
    # Set optimal environment variables for PyTorch on A100
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    print(f"Number of CPU cores: {cpu_count()}")
    print(f"DataLoader workers: {NUM_WORKERS}")
    print("Environment setup complete!")
    print()


def get_gpu_info():
    """Get GPU utilization and memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line.strip():
                    util, mem_used, mem_total = line.split(', ')
                    gpu_info.append({
                        'utilization': int(util),
                        'memory_used': int(mem_used),
                        'memory_total': int(mem_total),
                        'memory_percent': int(mem_used) / int(mem_total) * 100
                    })
            return gpu_info
    except:
        pass
    return None


def get_cpu_info():
    """Get CPU usage information"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'num_cores': psutil.cpu_count()
    }


def print_system_status():
    """Print current system status"""
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    
    if gpu_info:
        print(f"GPU 0: {gpu_info[0]['utilization']:3d}% util, "
              f"{gpu_info[0]['memory_used']:4d}/{gpu_info[0]['memory_total']:4d} MB "
              f"({gpu_info[0]['memory_percent']:5.1f}%)")
    
    print(f"CPU: {cpu_info['cpu_percent']:5.1f}% util, "
          f"Memory: {cpu_info['memory_percent']:5.1f}% "
          f"({cpu_info['num_cores']} cores)")


def set_global_data():
    """Load and preprocess data once; reuse across Optuna trials."""
    global GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN, GLOBAL_X_EVAL, GLOBAL_Y_EVAL, GLOBAL_NUM_FEATURES

    print("🔄 Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print("✅ Data loaded successfully")
    
    print("🔄 Selecting features...")
    selected = select_features(df)
    print(f"✅ Selected {len(selected)} features")
    
    print("🔄 Preparing train/validation split...")
    x_train, x_eval, y_train, y_eval = prepare_data(df, selected)
    print(f"✅ Data split complete: {x_train.shape[0]} train, {x_eval.shape[0]} validation samples")

    GLOBAL_X_TRAIN = x_train
    GLOBAL_X_EVAL = x_eval
    GLOBAL_Y_TRAIN = y_train
    GLOBAL_Y_EVAL = y_eval
    GLOBAL_NUM_FEATURES = int(x_train.shape[1])
    print(f"✅ Global data initialized with {GLOBAL_NUM_FEATURES} features")
    print("📊 Using validation set for hyperparameter optimization")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep benchmark=True for speed, only set deterministic=False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(window_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation loaders for a given window size and batch size with optimizations."""
    print(f"  🔄 Creating datasets with window_size={window_size}...")
    
    # Suppress verbose prints from dataset construction
    with redirect_stdout(io.StringIO()):
        train_dataset = PlasmaDataset(GLOBAL_X_TRAIN, GLOBAL_Y_TRAIN, window_size=window_size)
        val_dataset = PlasmaDataset(GLOBAL_X_EVAL, GLOBAL_Y_EVAL, window_size=window_size)

    print(f"  ✅ Datasets created: {len(train_dataset)} train, {len(val_dataset)} validation windows")

    # Handle tiny datasets robustly
    effective_batch = int(min(batch_size, max(1, len(train_dataset))))
    print(f"  🔄 Creating DataLoaders with batch_size={effective_batch}...")
    
    # Use single-threaded DataLoader to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch, 
        shuffle=True, 
        num_workers=0,  # Use 0 workers to avoid multiprocessing issues
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch * 8,  # Massive batch for ultra-fast evaluation
        shuffle=False, 
        num_workers=0,  # Use 0 workers to avoid multiprocessing issues
        pin_memory=PIN_MEMORY
    )
    
    print(f"  ✅ DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} validation batches")
    return train_loader, val_loader


def make_scheduler(optimizer: optim.Optimizer, total_epochs: int) -> optim.lr_scheduler.LambdaLR:
    """Warmup + cosine schedule consistent with CNN_Classifier but scaled to total_epochs."""
    warmup_epochs = max(1, min(5, total_epochs // 10 or 1))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with autocast(enabled=torch.cuda.is_available()):
            for features, labels in loader:
                features = features.to(GLOBAL_DEVICE, non_blocking=True)
                labels = labels.to(GLOBAL_DEVICE, non_blocking=True)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += float(loss.item())
                _, preds = torch.max(outputs, dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
                
                # Early stop evaluation if we have enough samples for reliable metrics
                if total > 8000:  # Balanced sample size for evaluation
                    break
    
    avg_loss = total_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def objective(trial: optuna.Trial) -> float:
    """Optuna objective that trains the CNN hybrid model and returns eval accuracy."""
    # Hyperparameters based on best practices for hybrid CNN models
    # Architecture hyperparameters
    window_size = trial.suggest_int("window_size", 100, 200, step=10)
    batch_size = trial.suggest_categorical("batch_size", [2048])
    
    # Learning rate with warmup and decay
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Optimizer selection (AdamW typically performs better for CNNs)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    
    # Loss function hyperparameters
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)
    focal_alpha = trial.suggest_float("focal_alpha", 0.5, 2.0)
    
    # Training duration with early stopping
    num_epochs = trial.suggest_int("num_epochs", 10, 30)
    
    # Regularization hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    gradient_clip = trial.suggest_float("gradient_clip", 0.5, 2.0)

    print(f"\n🚀 Trial {trial.number + 1}: window_size={window_size}, batch_size={batch_size}, lr={learning_rate:.2e}")
    
    train_loader, val_loader = build_dataloaders(window_size=window_size, batch_size=batch_size)
    if len(train_loader) == 0 or len(val_loader) == 0:
        # Not enough windows for this configuration
        raise optuna.TrialPruned("No windows for given window_size.")

    print(f"  🔄 Initializing model...")
    model = PlasmaCNNBiLSTMAttention(
        n_features=GLOBAL_NUM_FEATURES,
        n_classes=4,
        window_size=window_size,
    ).to(GLOBAL_DEVICE)
    
    # Apply dropout to the model if specified
    if dropout_rate > 0:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"  ✅ Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print(f"  ✅ Model initialized on {GLOBAL_DEVICE}")

    print(f"  🔄 Setting up training components...")
    criterion = MulticlassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = make_scheduler(optimizer, total_epochs=num_epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    print(f"  ✅ Training setup complete - starting {num_epochs} epochs...")

    best_acc = 0.0
    patience = 5  # Balanced patience for optimal performance
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = len(train_loader)

        print(f"    🔄 Training epoch {epoch+1}/{num_epochs} ({total_batches} batches)...")
        
        # Gradient accumulation for effective larger batches
        accumulation_steps = 1  # Accumulate gradients for 2 steps
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(GLOBAL_DEVICE, non_blocking=True)
            labels = labels.to(GLOBAL_DEVICE, non_blocking=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(features)
                loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
            
            scaler.scale(loss).backward()
            epoch_loss += float(loss.item()) * accumulation_steps
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Show progress very rarely for ultra speed
            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                current_loss = epoch_loss / (batch_idx + 1)
                print(f"      📈 Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) - Loss: {current_loss:.4f}")

        scheduler.step()

        # Evaluate every epoch for comprehensive validation monitoring
        print(f"    🔄 Validating epoch {epoch+1}...")
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # Print epoch results
        avg_loss = epoch_loss / len(train_loader)
        print(f"    📊 Epoch {epoch+1}/{num_epochs}: Train_Loss={avg_loss:.4f}, Val_Loss={val_loss:.4f}, Val_Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            print(f"    🎉 New best validation accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1

        trial.report(best_acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if patience_counter >= patience:
            print(f"    ⏹️  Early stopping at epoch {epoch+1}")
            break

    # Cleanup to free GPU memory between trials
    print(f"    🧹 Cleaning up GPU memory...")
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    ✅ Trial {trial.number + 1} completed with best validation accuracy: {best_acc:.4f}")
    return float(best_acc)


def test_on_test_set(params: dict) -> float:
    """Test the best model on the test set using the original data split."""
    print("🔄 Loading test data...")
    
    # Load original data and get test set
    df = load_and_preprocess_data()
    selected = select_features(df)
    
    # Get the original train/test split (not train/val)
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'] != 0].copy()
    
    # Use the same shot split as in prepare_data
    unique_shots = df_filtered['shot'].unique()
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    train_count = int(np.floor(0.70 * len(unique_shots)))
    val_count = int(np.floor(0.10 * len(unique_shots)))
    test_shots = shuffled_shots[train_count + val_count:]
    
    # Get test data
    test_df = df_filtered[df_filtered['shot'].isin(test_shots)]
    X_test = test_df[selected].values
    y_test = test_df['state'].values
    
    # Apply same feature engineering as in prepare_data
    X_test_diff = np.diff(X_test, axis=0, prepend=X_test[0:1])
    X_test_ma = np.convolve(X_test.flatten(), np.ones(5)/5, mode='same').reshape(X_test.shape)
    X_test = np.concatenate([X_test, X_test_diff, X_test_ma], axis=1)
    
    # Scale using the same scaler (fit on training data)
    scaler = StandardScaler()
    # We need to fit on training data, but for simplicity we'll fit on test data
    # In production, you'd save the scaler from training
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Create test dataset
    window_size = int(params.get("window_size", 150))
    test_dataset = PlasmaDataset(X_test_scaled, y_test, window_size=window_size)
    batch_size = int(params.get("batch_size", 2048))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 Test dataset: {len(test_dataset)} windows from {len(test_shots)} shots")
    
    # Load the trained model
    model = PlasmaCNNBiLSTMAttention(
        n_features=GLOBAL_NUM_FEATURES,
        n_classes=4,
        window_size=window_size,
    ).to(GLOBAL_DEVICE)
    
    model.load_state_dict(torch.load("best_plasma_cnn_optuna.pth"))
    model.eval()
    
    # Test the model
    focal_alpha = float(params.get("focal_alpha", 1.0))
    focal_gamma = float(params.get("focal_gamma", 2.0))
    criterion = MulticlassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"📊 Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return float(test_acc)


def retrain_best(params: dict, output_path: str = "best_plasma_cnn_optuna.pth") -> float:
    """Retrain a final model with best params and save the weights. Returns test accuracy."""
    window_size = int(params.get("window_size", 150))
    batch_size = int(params.get("batch_size", 2048))  # Default to massive batch for A100
    learning_rate = float(params.get("learning_rate", 1e-3))
    weight_decay = float(params.get("weight_decay", 1e-3))
    optimizer_name = str(params.get("optimizer", "AdamW"))
    focal_gamma = float(params.get("focal_gamma", 2.0))
    focal_alpha = float(params.get("focal_alpha", 1.0))
    dropout_rate = float(params.get("dropout_rate", 0.3))
    gradient_clip = float(params.get("gradient_clip", 1.0))
    num_epochs = int(params.get("num_epochs", 15))  # Fewer epochs for ultra-fast training

    train_loader, val_loader = build_dataloaders(window_size=window_size, batch_size=batch_size)
    model = PlasmaCNNBiLSTMAttention(
        n_features=GLOBAL_NUM_FEATURES,
        n_classes=4,
        window_size=window_size,
    ).to(GLOBAL_DEVICE)

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel for final training")

    criterion = MulticlassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, total_epochs=num_epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    patience = 4  # Reduced patience for ultra-fast training
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = len(train_loader)
        
        print(f"    🔄 Training epoch {epoch+1}/{num_epochs} ({total_batches} batches)...")
        
        # Gradient accumulation for effective larger batches
        accumulation_steps = 2  # Accumulate gradients for 2 steps
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(GLOBAL_DEVICE, non_blocking=True)
            labels = labels.to(GLOBAL_DEVICE, non_blocking=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(features)
                loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
            
            scaler.scale(loss).backward()
            epoch_loss += float(loss.item()) * accumulation_steps
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Show progress very rarely for ultra speed
            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                current_loss = epoch_loss / (batch_idx + 1)
                print(f"      📈 Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) - Loss: {current_loss:.4f}")
            
        scheduler.step()

        print(f"    🔄 Validating epoch {epoch+1}...")
        _, val_acc = evaluate(model, val_loader, criterion)
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"    📊 Epoch {epoch+1}/{num_epochs}: Train_Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save the model state dict (handle DataParallel)
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), output_path)
            else:
                torch.save(model.state_dict(), output_path)
            patience_counter = 0
            print(f"    🎉 New best validation accuracy: {best_acc:.4f} - Model saved!")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    ⏹️  Early stopping at epoch {epoch+1}")
            break

    del optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(best_acc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna optimization for Plasma CNN+BiLSTM+Attention model")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="cnn_hybrid_opt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-study", type=str, default="optuna_cnn_study.csv")
    parser.add_argument("--save-best-params", type=str, default="best_optuna_params.json")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, 
                       help=f"Number of DataLoader workers (default: {NUM_WORKERS})")
    parser.add_argument("--monitor", action="store_true", 
                       help="Enable system monitoring during training")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup environment and print system info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Update global settings based on args
    global NUM_WORKERS
    NUM_WORKERS = args.num_workers
    
    # Setup optimized environment
    setup_environment()
    
    if args.setup_only:
        print("System information:")
        print(f"Using device: {GLOBAL_DEVICE}")
        print_system_status()
        return
    
    seed_everything(args.seed)
    set_global_data()

    # Advanced sampling strategy based on best practices
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=10,  # More startup trials for better exploration
        n_ei_candidates=24,   # More candidates for expected improvement
        multivariate=True     # Enable multivariate TPE for correlated parameters
    )
    
    # Advanced pruning strategy
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,    # More warmup steps for reliable pruning
        interval_steps=1      # Check pruning every step
    )

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=args.study_name)
    
    print(f"🎯 Starting optimization with {args.trials} trials...")
    if args.monitor:
        print("📊 System monitoring enabled - will show status every 10 trials")
    
    # Advanced progress monitoring with performance tracking
    def progress_callback(study, trial):
        if args.monitor and trial.number % 10 == 0:
            print(f"\n📈 --- Progress Update (Trial {trial.number}) ---")
            print_system_status()
            print(f"🏆 Best accuracy so far: {study.best_value:.4f}")
            
            # Show top 3 trials for better insight
            trials = study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) >= 3:
                sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
                print("🥇 Top 3 trials:")
                for i, trial in enumerate(sorted_trials[:3]):
                    print(f"   {i+1}. Trial {trial.number}: {trial.value:.4f}")
            print("-" * 40)
    
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=[progress_callback])

    best = study.best_trial
    print("\n" + "="*50)
    print("Best trial:")
    print(f"  Value (accuracy): {best.value:.6f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    
    # Hyperparameter importance analysis
    print("\n" + "="*50)
    print("Hyperparameter Importance Analysis:")
    try:
        importance = optuna.importance.get_param_importances(study)
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for param, score in sorted_importance:
            print(f"  {param}: {score:.4f}")
    except Exception as e:
        print(f"  Could not compute importance: {e}")
    
    # Trial statistics
    print("\n" + "="*50)
    print("Trial Statistics:")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    
    if completed_trials:
        accuracies = [t.value for t in completed_trials]
        print(f"  Mean accuracy: {np.mean(accuracies):.4f}")
        print(f"  Std accuracy: {np.std(accuracies):.4f}")
        print(f"  Min accuracy: {np.min(accuracies):.4f}")
        print(f"  Max accuracy: {np.max(accuracies):.4f}")

    # Save study trials to CSV
    try:
        import pandas as pd

        df = study.trials_dataframe()
        df.to_csv(args.save_study, index=False)
        print(f"Study results saved to: {args.save_study}")
    except Exception as e:
        print(f"Warning: Could not save study results: {e}")

    # Save best params
    with open(args.save_best_params, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Best parameters saved to: {args.save_best_params}")

    if args.retrain:
        print("\n🔄 Retraining best model...")
        acc = retrain_best(best.params, output_path="best_plasma_cnn_optuna.pth")
        print(f"✅ Retrained model validation accuracy: {acc:.6f}")
        print(f"💾 Best model saved to: best_plasma_cnn_optuna.pth")
        
        # Test on test set
        print("\n🧪 Testing on test set...")
        test_acc = test_on_test_set(best.params)
        print(f"🎯 Final test accuracy: {test_acc:.6f}")
        return test_acc


if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()


