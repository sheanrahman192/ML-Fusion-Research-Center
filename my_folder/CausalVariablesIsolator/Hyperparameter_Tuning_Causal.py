#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_noncausal_database(csv_path: str = 'noncausal_database.csv') -> pd.DataFrame:
    print(f"Loading noncausal database from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Loaded shape: {df.shape}; columns: {len(df.columns)}")
    if 'label' not in df.columns:
        raise ValueError("Expected column 'label' not found in noncausal_database.csv")
    return df


def select_features(df: pd.DataFrame) -> List[str]:
    """Pick the noncausal feature set, keeping only those present in the file."""
    preferred = [
        'iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop',
        'fs04_max_smoothed'
    ]
    available = [f for f in preferred if f in df.columns]
    if not available:
        raise ValueError("None of the preferred features are present in the dataset.")
    print(f"Using {len(available)} features: {available}")
    return available


def prepare_data(df: pd.DataFrame, features: List[str]):
    print("Preparing data (binary: To be suppressed=0, ELMing=1)...")
    df_clean = df.dropna(subset=features, how='any').copy()
    # Map labels to binary
    df_clean['binary_state'] = np.where(df_clean['label'] == 'ELMing', 1, 0)

    X = df_clean[features]
    y = df_clean['binary_state']

    # Show distribution
    counts = y.value_counts().sort_index()
    names = {0: 'To be suppressed', 1: 'ELMing'}
    print("Class distribution (after cleaning):")
    for k, v in counts.items():
        print(f"  {names.get(k, k)} (class {k}): {v} ({v/len(y)*100:.1f}%)")

    # Split 70/10/20 (train/cv/test) via two-step split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp
    )
    print(f"Train/CV/Test sizes: {len(X_train)}/{len(X_cv)}/{len(X_test)}")
    return X_train, X_cv, X_test, y_train, y_cv, y_test, names


def evaluate_and_plot(clf: RandomForestClassifier, X_test, y_test, class_names, images_dir: str = 'noncausal_images'):
    os.makedirs(images_dir, exist_ok=True)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=[class_names[0], class_names[1]]))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[0], class_names[1]],
                yticklabels=[class_names[0], class_names[1]])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Noncausal RF)')
    out_path = os.path.join(images_dir, 'confusion_matrix_noncausal_rf.png')
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {out_path}")


def main():
    # =============================================
    # Load and prepare data
    # =============================================
    print("Loading and preparing data...")
    df = load_noncausal_database()
    features = select_features(df)
    X_train, X_cv, X_test, y_train, y_cv, y_test, class_names = prepare_data(df, features)
    print("Data loaded and prepared successfully!")

    # =============================================
    # Sequential Hyperparameter Optimization (binary accuracy based)
    # =============================================
    print("\n" + "="*60)
    print("SEQUENTIAL HYPERPARAMETER OPTIMIZATION - NONCAUSAL RF (Binary Accuracy)")
    print("="*60)

    param_ranges: Dict[str, List[Any]] = {
        'n_estimators': [50, 100, 150, 200, 300, 400, 600],
        'max_depth': [5, 10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 3, 4, 5, 6, 8, 10, 12],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 8, 10],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]
    }

    optimization_order = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']

    best_params: Dict[str, Any] = {
        'n_estimators': 400,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    }

    all_results: Dict[str, Dict[str, Any]] = {}
    optimization_history: Dict[str, Dict[str, Any]] = {}

    for step_index, param_name in enumerate(optimization_order):
        print(f"\n" + "="*40)
        print(f"STEP {step_index + 1}: Optimizing {param_name}")
        print(f"Current best params: {dict((k, v) for k, v in best_params.items() if k not in ['random_state', 'n_jobs'])}")
        print("="*40)

        candidates = param_ranges[param_name]
        cv_accs: List[float] = []
        train_accs: List[float] = []

        best_cv_for_param = -1.0
        best_value_for_param = best_params[param_name]

        for i, value in enumerate(candidates, 1):
            print(f"  Testing {param_name} = {value} ({i}/{len(candidates)})")

            current_params = best_params.copy()
            current_params[param_name] = value

            clf = RandomForestClassifier(**current_params)
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            y_cv_pred = clf.predict(X_cv)

            train_acc = accuracy_score(y_train, y_train_pred)
            cv_acc = accuracy_score(y_cv, y_cv_pred)

            train_accs.append(train_acc)
            cv_accs.append(cv_acc)

            if cv_acc > best_cv_for_param:
                best_cv_for_param = cv_acc
                best_value_for_param = value

            print(f"    -> Train Accuracy: {train_acc:.4f}")
            print(f"    -> CV Accuracy: {cv_acc:.4f}")
            print(f"    -> Best so far: {best_value_for_param} (CV Accuracy: {best_cv_for_param:.4f})")
            print()

        best_params[param_name] = best_value_for_param

        all_results[param_name] = {
            'values': candidates,
            'train_accs': train_accs,
            'cv_accs': cv_accs,
            'best_value': best_value_for_param,
            'best_acc': best_cv_for_param,
        }

        optimization_history[param_name] = best_params.copy()

        print(f"\n  BEST {param_name}: {best_value_for_param}")
        print(f"  BEST CV Accuracy: {best_cv_for_param:.4f}")
        print(f"  CV Accuracy improvement range: {max(cv_accs) - min(cv_accs):.4f}")

    # =============================================
    # Visualization of sequential optimization
    # =============================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (param_name, results) in enumerate(all_results.items()):
        ax = axes[i]
        x_indices = range(len(results['values']))
        ax.plot(x_indices, results['train_accs'], 'o-', color='green', alpha=0.7, label='Train Accuracy', linewidth=2)
        ax.plot(x_indices, results['cv_accs'], 'o-', color='blue', alpha=0.7, label='CV Accuracy', linewidth=2)

        best_idx = results['values'].index(results['best_value'])
        ax.plot(best_idx, results['cv_accs'][best_idx], 'r*', markersize=15, label=f"Best: {results['best_value']}")

        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Accuracy Score')
        ax.set_title(f"{param_name}\nBest: {results['best_value']} (CV: {results['best_acc']:.4f})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if len(results['values']) <= 10:
            ax.set_xticks(list(x_indices))
            ax.set_xticklabels([str(v) for v in results['values']], rotation=45)
        else:
            step = max(1, len(results['values']) // 8)
            tick_indices = list(x_indices)[::step]
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([str(results['values'][j]) for j in tick_indices], rotation=45)

    axes[-1].remove()

    plt.tight_layout()
    plt.suptitle('Sequential Hyperparameter Optimization - Noncausal RF (Binary Accuracy)', y=1.02, fontsize=16)
    os.makedirs('noncausal_images', exist_ok=True)
    plt.savefig(os.path.join('noncausal_images', 'sequential_optimization_noncausal.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # =============================================
    # Final model evaluation
    # =============================================
    print("\n" + "="*60)
    print("FINAL OPTIMIZED MODEL EVALUATION - NONCAUSAL RF (Binary Accuracy)")
    print("="*60)

    print("Final optimized parameters:")
    for param, value in best_params.items():
        if param not in ['random_state', 'n_jobs']:
            print(f"  {param}: {value}")

    final_clf = RandomForestClassifier(**best_params)
    final_clf.fit(X_train, y_train)

    y_pred_test = final_clf.predict(X_test)
    y_pred_cv = final_clf.predict(X_cv)
    y_pred_train = final_clf.predict(X_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_cv = accuracy_score(y_cv, y_pred_cv)
    acc_train = accuracy_score(y_train, y_pred_train)

    print("\nFinal Model Performance (Accuracy):")
    print(f"  Training Accuracy: {acc_train:.4f}")
    print(f"  CV Accuracy: {acc_cv:.4f}")
    print(f"  Test Accuracy: {acc_test:.4f}")
    print(f"  Overfitting Gap (Train-CV): {acc_train - acc_cv:.4f}")
    print(f"  Overfitting Gap (Train-Test): {acc_train - acc_test:.4f}")

    # Classification report for optimized parameters (test set)
    print("\nClassification Report (Test, optimized parameters):")
    print(classification_report(y_test, y_pred_test, target_names=[class_names[0], class_names[1]]))

    # Baselines
    baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_clf.fit(X_train, y_train)
    baseline_cv_acc = accuracy_score(y_cv, baseline_clf.predict(X_cv))
    baseline_test_acc = accuracy_score(y_test, baseline_clf.predict(X_test))

    initial_optimized_clf = RandomForestClassifier(n_estimators=all_results['n_estimators']['best_value'], random_state=42)
    initial_optimized_clf.fit(X_train, y_train)
    initial_cv_acc = accuracy_score(y_cv, initial_optimized_clf.predict(X_cv))
    initial_test_acc = accuracy_score(y_test, initial_optimized_clf.predict(X_test))

    print("\nComparison with baseline models (Accuracy):")
    print(f"  Baseline (n_estimators=100) - CV: {baseline_cv_acc:.4f}, Test: {baseline_test_acc:.4f}")
    print(f"  Only n_estimators optimized - CV: {initial_cv_acc:.4f}, Test: {initial_test_acc:.4f}")
    print(f"  Fully optimized model - CV: {acc_cv:.4f}, Test: {acc_test:.4f}")
    print(f"  Total improvement - CV: {acc_cv - baseline_cv_acc:.4f}, Test: {acc_test - baseline_test_acc:.4f}")
    print(f"  Improvement beyond n_estimators - CV: {acc_cv - initial_cv_acc:.4f}, Test: {acc_test - initial_test_acc:.4f}")

    # Confusion matrix for test set
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[0], class_names[1]],
                yticklabels=[class_names[0], class_names[1]])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test (Noncausal RF)')
    out_path = os.path.join('noncausal_images', 'confusion_matrix_test_noncausal_rf.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == '__main__':
    main()


