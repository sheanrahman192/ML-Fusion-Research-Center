#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/Test sizes: {len(X_train)}/{len(X_test)}")
    return X_train, X_test, y_train, y_test, names


def train_rf(X_train, y_train) -> RandomForestClassifier:
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


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


def print_feature_importance(clf: RandomForestClassifier, features: List[str], top_n: int = 10):
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\nTop feature importances:")
    for i in range(min(top_n, len(features))):
        idx = order[i]
        print(f"  {i+1:2d}. {features[idx]}: {importances[idx]:.4f}")


def main():
    df = load_noncausal_database()
    features = select_features(df)
    X_train, X_test, y_train, y_test, names = prepare_data(df, features)
    clf = train_rf(X_train, y_train)
    evaluate_and_plot(clf, X_test, y_test, names)
    print_feature_importance(clf, features)


if __name__ == '__main__':
    main()


