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

def prepare_four_state_classification(df):
    """Prepare data for four-state classification"""
    selected_features = [
        'iln3iamp', 'tribot', 'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs04_max_avg','fs_up_sum', 'fs_sum'
    ]
    
    available_features = [f for f in selected_features if f in df.columns]
    df_cleaned = df.dropna(subset=available_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    
    # Keep original 4 states: 1 (Suppressed), 2 (Dithering), 3 (Mitigated), 4 (ELMing)
    df_cleaned['four_state'] = df_cleaned['state']
    
    columns_to_keep = ['shot', 'time', 'four_state'] + available_features
    df_cleaned = df_cleaned[columns_to_keep].copy()
    
    return df_cleaned, available_features

def split_data_randomly(df_cleaned, available_features):
    """Split data randomly: 70% train, 10% CV, 20% test"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get total number of samples
    total_samples = len(df_cleaned)
    
    # Create random indices for shuffling
    indices = np.random.permutation(total_samples)
    
    # Calculate split points
    train_end = int(np.floor(0.70 * total_samples))
    cv_end = int(np.floor(0.80 * total_samples))
    
    # Split indices
    train_indices = indices[:train_end]
    cv_indices = indices[train_end:cv_end]
    test_indices = indices[cv_end:]
    
    # Get data for each split
    train_df = df_cleaned.iloc[train_indices]
    cv_df = df_cleaned.iloc[cv_indices]
    test_df = df_cleaned.iloc[test_indices]
    
    X_train = train_df[available_features]
    y_train = train_df['four_state']
    X_cv = cv_df[available_features]
    y_cv = cv_df['four_state']
    X_test = test_df[available_features]
    y_test = test_df['four_state']
    
    return X_train, X_cv, X_test, y_train, y_cv, y_test



# Load and prepare data
print("Loading and preparing data...")
try:
    df = load_and_prepare_data()
    df_cleaned, available_features = prepare_four_state_classification(df)
    X_train, X_cv, X_test, y_train, y_cv, y_test = split_data_randomly(df_cleaned, available_features)
    print("Data loaded and prepared successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

print(f"Number of rows remaining after cleaning: {len(df_cleaned)}")
print(f"Four state distribution:")
print(df_cleaned['four_state'].value_counts().sort_index())
print(f"Training set size: {len(X_train)}")
print(f"Cross-validation set size: {len(X_cv)}")
print(f"Testing set size: {len(X_test)}")

# =============================================================================
# SEQUENTIAL HYPERPARAMETER OPTIMIZATION
# =============================================================================

print("\n" + "="*60)
print("SEQUENTIAL HYPERPARAMETER OPTIMIZATION - FOUR-STATE CLASSIFICATION (Overall Accuracy-based)")
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
    'n_jobs': -2
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
    test_accs = []
    train_accs = []

    best_acc_for_param = 0
    best_value_for_param = best_params[param_name]

    for i, value in enumerate(param_values, 1):
        print(f"  Testing {param_name} = {value} ({i}/{len(param_values)})")

        # Create parameters dict with current value and all previously optimized values
        current_params = best_params.copy()
        current_params[param_name] = value

        # Train model
        clf = RandomForestClassifier(**current_params)
        clf.fit(X_train, y_train)

        # Evaluate using overall accuracy
        y_train_pred = clf.predict(X_train)
        y_cv_pred = clf.predict(X_cv)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        cv_acc = accuracy_score(y_cv, y_cv_pred)

        train_accs.append(train_acc)
        test_accs.append(cv_acc)

        # Check if this is the best value so far
        if cv_acc > best_acc_for_param:
            best_acc_for_param = cv_acc
            best_value_for_param = value

        print(f"    -> Train Accuracy: {train_acc:.4f}")
        print(f"    -> CV Accuracy: {cv_acc:.4f}")
        print(f"    -> Best so far: {best_value_for_param} (CV Accuracy: {best_acc_for_param:.4f})")
        print()

    # Update best parameters with the optimal value found
    best_params[param_name] = best_value_for_param

    # Store results for plotting
    all_results[param_name] = {
        'values': param_values,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_value': best_value_for_param,
        'best_acc': best_acc_for_param
    }

    # Store optimization history
    optimization_history[param_name] = best_params.copy()

    print(f"\n  BEST {param_name}: {best_value_for_param}")
    print(f"  BEST CV Accuracy: {best_acc_for_param:.4f}")
    print(f"  CV Accuracy improvement range: {max(test_accs) - min(test_accs):.4f}")

# =============================================================================
# VISUALIZATION OF SEQUENTIAL OPTIMIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (param_name, results) in enumerate(all_results.items()):
    ax = axes[i]

    # Plot both train and CV accuracy scores
    x_indices = range(len(results['values']))
    ax.plot(x_indices, results['train_accs'], 'o-', color='green', alpha=0.7, label='Train Accuracy', linewidth=2)
    ax.plot(x_indices, results['test_accs'], 'o-', color='blue', alpha=0.7, label='CV Accuracy', linewidth=2)

    # Mark best point
    best_idx = results['values'].index(results['best_value'])
    ax.plot(best_idx, results['test_accs'][best_idx], 'r*', markersize=15, label=f'Best: {results["best_value"]}')

    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Accuracy Score')
    ax.set_title(f'{param_name}\nBest: {results["best_value"]} (CV Accuracy: {results["best_acc"]:.4f})')
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
plt.suptitle('Sequential Hyperparameter Optimization Results - Four-State Classification (Overall Accuracy-based)', y=1.02, fontsize=16)
plt.savefig('sequential_optimization_results.png', dpi=200, bbox_inches='tight')
plt.show()

# =============================================================================
# OPTIMIZATION HISTORY TRACKING
# =============================================================================

print("\n" + "="*60)
print("OPTIMIZATION HISTORY (Overall Accuracy-based)")
print("="*60)

# Create a summary of how parameters evolved
history_df_data = []
for step, param_name in enumerate(optimization_order):
    params_at_step = optimization_history[param_name]
    row = {'Step': step + 1, 'Parameter_Optimized': param_name}
    for p in optimization_order:
        row[p] = params_at_step[p]
    row['Best_CV_Accuracy'] = all_results[param_name]['best_acc']
    history_df_data.append(row)

history_df = pd.DataFrame(history_df_data)
print("\nOptimization progression (Overall Accuracy scores):")
print(history_df.to_string(index=False))

# Plot accuracy improvement over optimization steps
plt.figure(figsize=(10, 6))
steps = [f"Step {i+1}\n{param}" for i, param in enumerate(optimization_order)]
accs = [all_results[param]['best_acc'] for param in optimization_order]

plt.plot(range(len(steps)), accs, 'o-', linewidth=2, markersize=8)
plt.xlabel('Optimization Step')
plt.ylabel('Best CV Accuracy Score')
plt.title('Sequential Optimization Progress - Four-State Classification (Overall Accuracy-based)')
plt.xticks(range(len(steps)), steps)
plt.grid(True, alpha=0.3)

# Annotate improvements
for i in range(1, len(accs)):
    improvement = accs[i] - accs[i-1]
    if improvement > 0:
        plt.annotate(f'+{improvement:.4f}',
                    xy=(i, accs[i]),
                    xytext=(i, accs[i] + 0.002),
                    ha='center', color='green', fontweight='bold')
    elif improvement < 0:
        plt.annotate(f'{improvement:.4f}',
                    xy=(i, accs[i]),
                    xytext=(i, accs[i] - 0.002),
                    ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('optimization_history.png', dpi=200, bbox_inches='tight')
plt.show()

# =============================================================================
# FINAL MODEL EVALUATION
# =============================================================================

print("\n" + "="*60)
print("FINAL OPTIMIZED MODEL EVALUATION - FOUR-STATE CLASSIFICATION (Overall Accuracy-based)")
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

# Accuracy scores
final_test_acc = accuracy_score(y_test, y_pred_final)
final_cv_acc = accuracy_score(y_cv, y_cv_pred_final)
final_train_acc = accuracy_score(y_train, y_train_pred_final)



print(f"\nFinal Model Performance (Overall Accuracy scores):")
print(f"  Training Accuracy: {final_train_acc:.4f}")
print(f"  CV Accuracy: {final_cv_acc:.4f}")
print(f"  Test Accuracy: {final_test_acc:.4f}")
print(f"  Overfitting Gap (Train-CV Accuracy): {final_train_acc - final_cv_acc:.4f}")
print(f"  Overfitting Gap (Train-Test Accuracy): {final_train_acc - final_test_acc:.4f}")



# Compare with baseline models
baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_clf.fit(X_train, y_train)
baseline_cv_acc = accuracy_score(y_cv, baseline_clf.predict(X_cv))
baseline_test_acc = accuracy_score(y_test, baseline_clf.predict(X_test))


# Compare with initial n_estimators optimization only
initial_optimized_clf = RandomForestClassifier(n_estimators=all_results['n_estimators']['best_value'], random_state=42)
initial_optimized_clf.fit(X_train, y_train)
initial_optimized_cv_acc = accuracy_score(y_cv, initial_optimized_clf.predict(X_cv))
initial_optimized_test_acc = accuracy_score(y_test, initial_optimized_clf.predict(X_test))


print(f"\nComparison with baseline models (Overall Accuracy scores):")
print(f"  Baseline (n_estimators=100) - CV: {baseline_cv_acc:.4f}, Test: {baseline_test_acc:.4f}")
print(f"  Only n_estimators optimized - CV: {initial_optimized_cv_acc:.4f}, Test: {initial_optimized_test_acc:.4f}")
print(f"  Fully optimized model - CV: {final_cv_acc:.4f}, Test: {final_test_acc:.4f}")
print(f"  Total improvement - CV: {final_cv_acc - baseline_cv_acc:.4f}, Test: {final_test_acc - baseline_test_acc:.4f}")
print(f"  Improvement beyond n_estimators - CV: {final_cv_acc - initial_optimized_cv_acc:.4f}, Test: {final_test_acc - initial_optimized_test_acc:.4f}")



# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': final_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Confusion matrix for four-state classification
unique_classes = sorted(y_train.unique())
class_labels = ['Suppressed', 'Dithering', 'Mitigated', 'ELMing']  # Four-state labels

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_final, labels=unique_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Four-State Classification')

plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=class_labels))

print(f"\nFour-state mapping summary:")
print(f"  State 1: Suppressed")
print(f"  State 2: Dithering")
print(f"  State 3: Mitigated")
print(f"  State 4: ELMing")
