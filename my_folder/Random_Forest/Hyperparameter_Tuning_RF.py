import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from itertools import product
import warnings
warnings.filterwarnings('ignore')

plt.close('all')

def load_and_prepare_data():
    """Load and prepare the dataset from CSV"""
    df = pd.read_csv('plasma_data.csv')
    df = df[df['shot'] != 191675].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['original_index'] = df.index
    return df

def prepare_binary_classification(df):
    """Prepare data for binary classification"""
    selected_features = [
        'iln3iamp', 'iln2iamp', 'iun2iamp', 'iun3iamp', 'iln3iphase', 'iln2iphase', 'iun2iphase', 'iun3iphase',
        'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs04_max_avg'
    ]
    
    available_features = [f for f in selected_features if f in df.columns]
    df_cleaned = df.dropna(subset=available_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    
    # Map states to binary: 1-3 -> 0 (Suppressed), 4 -> 1 (ELMing)
    df_cleaned['binary_state'] = df_cleaned['state'].apply(lambda x: 0 if x in [1,2,3] else 1)
    
    columns_to_keep = ['shot', 'time', 'binary_state'] + available_features
    df_cleaned = df_cleaned[columns_to_keep].copy()
    
    return df_cleaned, available_features

def split_data_by_shots(df_cleaned, available_features):
    """Split data by unique shots: 70% train, 10% CV, 20% test"""
    unique_shots = df_cleaned['shot'].unique()
    num_shots = len(unique_shots)
    
    # Shuffle shots first, then split
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    
    # Split shuffled shots: 70% train, 10% CV, 20% test
    train_count = int(np.floor(0.70 * num_shots))
    cv_count = int(np.floor(0.10 * num_shots))
    
    train_shots = shuffled_shots[:train_count]
    cv_shots = shuffled_shots[train_count:train_count + cv_count]
    test_shots = shuffled_shots[train_count + cv_count:]
    
    # Get data for each split
    train_df = df_cleaned[df_cleaned['shot'].isin(train_shots)]
    cv_df = df_cleaned[df_cleaned['shot'].isin(cv_shots)]
    test_df = df_cleaned[df_cleaned['shot'].isin(test_shots)]
    
    X_train = train_df[available_features]
    y_train = train_df['binary_state']
    X_cv = cv_df[available_features]
    y_cv = cv_df['binary_state']
    X_test = test_df[available_features]
    y_test = test_df['binary_state']
    
    return X_train, X_cv, X_test, y_train, y_cv, y_test

# Load and prepare data
print("Loading and preparing data...")
df = load_and_prepare_data()
df_cleaned, available_features = prepare_binary_classification(df)
X_train, X_cv, X_test, y_train, y_cv, y_test = split_data_by_shots(df_cleaned, available_features)

print(f"Number of rows remaining after cleaning: {len(df_cleaned)}")
print(f"Binary state distribution:")
print(df_cleaned['binary_state'].value_counts().sort_index())
print(f"Training set size: {len(X_train)}")
print(f"Cross-validation set size: {len(X_cv)}")
print(f"Testing set size: {len(X_test)}")

# =============================================================================
# SEQUENTIAL HYPERPARAMETER OPTIMIZATION
# =============================================================================

print("\n" + "="*60)
print("SEQUENTIAL HYPERPARAMETER OPTIMIZATION - BINARY CLASSIFICATION (F1-based)")
print("="*60)

# Full parameter ranges
param_ranges = {
    'n_estimators': [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700],
    'max_depth': [3, 5, 7, 10, 12, 15, 17, 20, 25, 30, 35, None],
    'min_samples_split': [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30],
    'max_features': ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3]
}

# Define the order to optimize parameters
optimization_order = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']

# Initialize best parameters with defaults
best_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Store results for visualization
all_results = {}
optimization_history = {}

# Sequential optimization
for param_idx, param_name in enumerate(optimization_order):
    print(f"\n" + "="*40)
    print(f"STEP {param_idx + 1}: Optimizing {param_name}")
    print(f"Current best params: {dict((k,v) for k,v in best_params.items() if k not in ['random_state', 'n_jobs'])}")
    print("="*40)

    param_values = param_ranges[param_name]
    test_f1s = []
    train_f1s = []

    best_f1_for_param = 0
    best_value_for_param = best_params[param_name]

    for i, value in enumerate(param_values, 1):
        print(f"  Testing {param_name} = {value} ({i}/{len(param_values)})")

        # Create parameters dict with current value and all previously optimized values
        current_params = best_params.copy()
        current_params[param_name] = value

        # Train model
        clf = RandomForestClassifier(**current_params)
        clf.fit(X_train, y_train)

        # Evaluate using F1 score
        train_f1 = f1_score(y_train, clf.predict(X_train), average='binary', zero_division=0)
        cv_f1 = f1_score(y_cv, clf.predict(X_cv), average='binary', zero_division=0)

        train_f1s.append(train_f1)
        test_f1s.append(cv_f1)

        # Check if this is the best value so far
        if cv_f1 > best_f1_for_param:
            best_f1_for_param = cv_f1
            best_value_for_param = value

        print(f"    -> Train F1: {train_f1:.4f}")
        print(f"    -> CV F1: {cv_f1:.4f}")
        print(f"    -> Best so far: {best_value_for_param} (CV F1: {best_f1_for_param:.4f})")
        print()

    # Update best parameters with the optimal value found
    best_params[param_name] = best_value_for_param

    # Store results for plotting
    all_results[param_name] = {
        'values': param_values,
        'train_f1s': train_f1s,
        'test_f1s': test_f1s,
        'best_value': best_value_for_param,
        'best_f1': best_f1_for_param
    }

    # Store optimization history
    optimization_history[param_name] = best_params.copy()

    print(f"\n  BEST {param_name}: {best_value_for_param}")
    print(f"  BEST CV F1: {best_f1_for_param:.4f}")
    print(f"  F1 improvement range: {max(test_f1s) - min(test_f1s):.4f}")

# =============================================================================
# VISUALIZATION OF SEQUENTIAL OPTIMIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (param_name, results) in enumerate(all_results.items()):
    ax = axes[i]

    # Plot both train and test F1 scores
    x_indices = range(len(results['values']))
    ax.plot(x_indices, results['train_f1s'], 'o-', color='green', alpha=0.7, label='Train F1', linewidth=2)
    ax.plot(x_indices, results['test_f1s'], 'o-', color='blue', alpha=0.7, label='CV F1', linewidth=2)

    # Mark best point
    best_idx = results['values'].index(results['best_value'])
    ax.plot(best_idx, results['test_f1s'][best_idx], 'r*', markersize=15, label=f'Best: {results["best_value"]}')

    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'{param_name}\nBest: {results["best_value"]} (F1: {results["best_f1"]:.4f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis to show parameter values (sample some if too many)
    if len(results['values']) <= 10:
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(v) for v in results['values']], rotation=45)
    else:
        # Show every nth value to avoid crowding
        step = max(1, len(results['values']) // 8)
        tick_indices = x_indices[::step]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(results['values'][i]) for i in tick_indices], rotation=45)

# Remove unused subplot
axes[-1].remove()

plt.tight_layout()
plt.suptitle('Sequential Hyperparameter Optimization Results - Binary Classification (F1-based)', y=1.02, fontsize=16)
plt.show()

# =============================================================================
# OPTIMIZATION HISTORY TRACKING
# =============================================================================

print("\n" + "="*60)
print("OPTIMIZATION HISTORY (F1-based)")
print("="*60)

# Create a summary of how parameters evolved
history_df_data = []
for step, param_name in enumerate(optimization_order):
    params_at_step = optimization_history[param_name]
    row = {'Step': step + 1, 'Parameter_Optimized': param_name}
    for p in optimization_order:
        row[p] = params_at_step[p]
    row['Best_CV_F1'] = all_results[param_name]['best_f1']
    history_df_data.append(row)

history_df = pd.DataFrame(history_df_data)
print("\nOptimization progression (F1 scores):")
print(history_df.to_string(index=False))

# Plot F1 improvement over optimization steps
plt.figure(figsize=(10, 6))
steps = [f"Step {i+1}\n{param}" for i, param in enumerate(optimization_order)]
f1s = [all_results[param]['best_f1'] for param in optimization_order]

plt.plot(range(len(steps)), f1s, 'o-', linewidth=2, markersize=8)
plt.xlabel('Optimization Step')
plt.ylabel('Best CV F1 Score')
plt.title('Sequential Optimization Progress - Binary Classification (F1-based)')
plt.xticks(range(len(steps)), steps)
plt.grid(True, alpha=0.3)

# Annotate improvements
for i in range(1, len(f1s)):
    improvement = f1s[i] - f1s[i-1]
    if improvement > 0:
        plt.annotate(f'+{improvement:.4f}',
                    xy=(i, f1s[i]),
                    xytext=(i, f1s[i] + 0.002),
                    ha='center', color='green', fontweight='bold')
    elif improvement < 0:
        plt.annotate(f'{improvement:.4f}',
                    xy=(i, f1s[i]),
                    xytext=(i, f1s[i] - 0.002),
                    ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# FINAL MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("FINAL OPTIMIZED MODEL EVALUATION - BINARY CLASSIFICATION (F1-based)")
print("="*60)

print("Final optimized parameters:")
for param, value in best_params.items():
    if param not in ['random_state', 'n_jobs']:
        print(f"  {param}: {value}")

# Train final model with optimized parameters
final_clf = RandomForestClassifier(**best_params)
final_clf.fit(X_train, y_train)

# Predictions
y_pred_final = final_clf.predict(X_test)
y_cv_pred_final = final_clf.predict(X_cv)
y_train_pred_final = final_clf.predict(X_train)

# F1 scores
final_test_f1 = f1_score(y_test, y_pred_final, average='binary', zero_division=0)
final_cv_f1 = f1_score(y_cv, y_cv_pred_final, average='binary', zero_division=0)
final_train_f1 = f1_score(y_train, y_train_pred_final, average='binary', zero_division=0)

print(f"\nFinal Model Performance (F1 scores):")
print(f"  Training F1: {final_train_f1:.4f}")
print(f"  CV F1: {final_cv_f1:.4f}")
print(f"  Test F1: {final_test_f1:.4f}")
print(f"  Overfitting Gap (Train-CV F1): {final_train_f1 - final_cv_f1:.4f}")
print(f"  Overfitting Gap (Train-Test F1): {final_train_f1 - final_test_f1:.4f}")

# Compare with baseline models
baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_clf.fit(X_train, y_train)
baseline_cv_f1 = f1_score(y_cv, baseline_clf.predict(X_cv), average='binary', zero_division=0)
baseline_test_f1 = f1_score(y_test, baseline_clf.predict(X_test), average='binary', zero_division=0)

# Compare with initial n_estimators optimization only
initial_optimized_clf = RandomForestClassifier(n_estimators=all_results['n_estimators']['best_value'], random_state=42)
initial_optimized_clf.fit(X_train, y_train)
initial_optimized_cv_f1 = f1_score(y_cv, initial_optimized_clf.predict(X_cv), average='binary', zero_division=0)
initial_optimized_test_f1 = f1_score(y_test, initial_optimized_clf.predict(X_test), average='binary', zero_division=0)

print(f"\nComparison with baseline models (F1 scores):")
print(f"  Baseline (n_estimators=100) - CV: {baseline_cv_f1:.4f}, Test: {baseline_test_f1:.4f}")
print(f"  Only n_estimators optimized - CV: {initial_optimized_cv_f1:.4f}, Test: {initial_optimized_test_f1:.4f}")
print(f"  Fully optimized model - CV: {final_cv_f1:.4f}, Test: {final_test_f1:.4f}")
print(f"  Total improvement - CV: {final_cv_f1 - baseline_cv_f1:.4f}, Test: {final_test_f1 - baseline_test_f1:.4f}")
print(f"  Improvement beyond n_estimators - CV: {final_cv_f1 - initial_optimized_cv_f1:.4f}, Test: {final_test_f1 - initial_optimized_test_f1:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': final_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Confusion matrix for binary classification
unique_classes = sorted(y_train.unique())
class_labels = ['Suppressed', 'ELMing']  # Binary labels

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_final, labels=unique_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Binary Classification (ELMing vs Suppressed)')
plt.show()

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=class_labels))

print(f"\nBinary state mapping summary:")
print(f"  Suppressed (0): Combines original states 1, 2, 3 (Suppressed, Dithering, Mitigated)")
print(f"  ELMing (1): Original state 4 (ELMing)")
