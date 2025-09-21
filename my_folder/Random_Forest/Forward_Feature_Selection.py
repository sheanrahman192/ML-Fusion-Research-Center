import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

plt.close('all')

# Load dataset from CSV
df = pd.read_csv('/mnt/homes/sr4240/my_folder/plasma_data.csv')

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Add an index column to keep track of original row numbers
df['original_index'] = df.index

# Select candidate features and the target variable
candidate_features = ['betan', 'bt', 'bt0', 'density', 'dR_sep', 'fs04', 'fs04_max_avg',
                    'fs04_max_smoothed', 'iln2iamp', 'iln2iphase', 'iln3iamp', 'iln3iphase',
                    'Ip', 'iun2iamp', 'iun2iphase', 'iun3iamp', 'iun3iphase', 'kappa',
                    'li', 'n_e', 'n_eped', 'p_eped', 'q95', 'rotation_core', 'rotation_edge',
                    't_eped', 'tribot', 'tritop', 'zeff', 'thin_fs04_max_smoothed']

target_column = 'state'

# Validate target column
if target_column not in df.columns:
    print(f"Error: Missing target column: {target_column}")
    exit()

# Keep only features that exist in the data
available_features = [f for f in candidate_features if f in df.columns]
missing_feature_columns = [f for f in candidate_features if f not in df.columns]
if missing_feature_columns:
    print(f"Warning: Missing feature columns (will be skipped): {missing_feature_columns}")
if len(available_features) == 0:
    print("Error: No candidate features found in dataset.")
    exit()

# Speed controls and classifier factory
train_subsample_fraction = 0.3
random_state_global = 42
outer_n_jobs = max(1, os.cpu_count() // 2)

def create_classifier(mode='selection'):
    if mode == 'final':
        n_estimators = 300
        max_depth = 14
        max_samples = 1.0
    else:
        n_estimators = 150
        max_depth = 10
        max_samples = 0.8
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        max_samples=max_samples,
        random_state=random_state_global,
        n_jobs=1
    )

# Clean the dataframe
df_cleaned = df.dropna(subset=available_features, how='any')

# Check if 'n' column exists before filtering
if 'n' in df.columns:
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]
    print("Applied filter: n == 3")
else:
    print("Warning: 'n' column not found, skipping n==3 filter")

# Remove rows where state is 'N/A' (0)
df_cleaned = df_cleaned[df_cleaned['state'] != 0]

# Map states to binary classification: ELMing vs Suppressed
# Assuming state mapping: 1=Suppressed, 2=Dithering, 3=Mitigated, 4=ELMing
# Combine states 1, 2, 3 into "Suppressed" and keep state 4 as "ELMing"
def map_states_to_binary(state):
    if state in [1, 2, 3]:  # Suppressed, Dithering, Mitigated -> Suppressed
        return 0  # Suppressed
    elif state == 4:  # ELMing
        return 1  # ELMing
    else:
        return state  # Keep other states as is (shouldn't happen after cleaning)

df_cleaned['binary_state'] = df_cleaned['state'].apply(map_states_to_binary)

# Check minimum dataset size
if len(df_cleaned) < 100:
    print(f"Error: Dataset too small after cleaning: {len(df_cleaned)} rows")
    exit()

print(f"Number of rows remaining after cleaning: {len(df_cleaned)}")
print(f"Original features count: {len(available_features)}")
print(f"Original state distribution:")
print(df_cleaned['state'].value_counts().sort_index())
print(f"Binary state distribution:")
print(df_cleaned['binary_state'].value_counts().sort_index())

# Prepare input (X) and target (y) using binary state
X = df_cleaned[available_features]
y = df_cleaned['binary_state']  # Use binary state instead of original state

# Linear/Chronological split (80/20)
split_point = int(len(df_cleaned) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

def evaluate_features(features, X_train, X_test, y_train, y_test, return_model=False, verbose=False):
    """Evaluate a set of features and return F1 scores (optionally model)."""
    if len(features) == 0:
        if return_model:
            return 0.0, 0.0, None
        else:
            return 0.0, 0.0

    # Optional subsample of training data for speed
    if 0.0 < train_subsample_fraction < 1.0:
        X_sub = X_train[features].sample(frac=train_subsample_fraction, random_state=random_state_global)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub = X_train[features]
        y_sub = y_train

    clf = create_classifier(mode='selection')
    clf.fit(X_sub, y_sub)

    # Test F1 score
    y_pred = clf.predict(X_test[features])
    test_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    # Training F1 score (on subsample when used)
    y_train_pred = clf.predict(X_sub)
    train_f1 = f1_score(y_sub, y_train_pred, average='binary', zero_division=0)

    if verbose:
        print(f"  Features: {features}")
        print(f"  Test F1: {test_f1:.4f}, Train F1: {train_f1:.4f}")

    if return_model:
        return test_f1, train_f1, clf
    else:
        return test_f1, train_f1

def forward_selection(X_train, X_test, y_train, y_test, available_features,
                      min_f1_improvement=0.001,
                      max_features=None):
    """Greedy forward selection using F1 improvement only (no importance)."""
    if max_features is None:
        max_features = len(available_features)

    selected_features = []
    remaining_features = available_features.copy()
    results = []

    print("\n=== FAST FORWARD SELECTION (F1-only) - BINARY CLASSIFICATION ===")
    print(f"{'Step':<4} {'Features':<8} {'Test F1':<10} {'Train F1':<11} {'Added Feature':<20} {'Action':<30}")
    print("-" * 120)

    step = 0
    current_test_f1 = 0.0
    current_train_f1 = 0.0

    # Log initial state
    results.append({
        'step': step,
        'features': selected_features.copy(),
        'test_f1': current_test_f1,
        'train_f1': current_train_f1,
        'action': 'Starting with empty set',
        'added_feature': None,
        'f1_improvement': 0.0
    })

    print(f"{step:<4} {len(selected_features):<8} {current_test_f1:.4f}     {current_train_f1:.4f}      {'None':<20} {'Starting with empty set':<30}")

    # Continue adding features
    while len(remaining_features) > 0 and len(selected_features) < max_features:
        step += 1

        print(f"\n  Step {step}: Evaluating {len(remaining_features)} candidate features in parallel...")

        def eval_one(candidate_feature):
            temp_features = selected_features + [candidate_feature]
            temp_test_f1, temp_train_f1 = evaluate_features(
                temp_features, X_train, X_test, y_train, y_test, return_model=False
            )
            return candidate_feature, temp_test_f1, temp_train_f1

        parallel_results = Parallel(n_jobs=outer_n_jobs, prefer='processes')(delayed(eval_one)(cf) for cf in remaining_features)

        candidate_evaluations = {}
        for i, (candidate_feature, temp_test_f1, temp_train_f1) in enumerate(parallel_results):
            candidate_evaluations[candidate_feature] = {
                'test_f1': temp_test_f1,
                'train_f1': temp_train_f1,
                'f1_improvement': temp_test_f1 - current_test_f1
            }
            print(f"    {i+1:2d}/{len(remaining_features)}: {candidate_feature:<20} - "
                  f"F1: {temp_test_f1:.4f} (+{temp_test_f1 - current_test_f1:+.4f})")

        # Find candidates that meet minimum criteria
        valid_candidates = {f: d for f, d in candidate_evaluations.items() if d['f1_improvement'] > min_f1_improvement}

        if not valid_candidates:
            # No valid candidates found
            results.append({
                'step': step,
                'features': selected_features.copy(),
                'test_f1': current_test_f1,
                'train_f1': current_train_f1,
                'action': 'No valid candidates - stopped',
                'added_feature': None,
                'f1_improvement': 0.0
            })
            print(f"{step:<4} {len(selected_features):<8} {current_test_f1:.4f}     {current_train_f1:.4f}      {'None':<20} {'No valid candidates found':<30}")
            break

        # Among valid candidates, select the one with highest test F1
        best_feature = max(valid_candidates.keys(), key=lambda f: valid_candidates[f]['test_f1'])
        best_eval = valid_candidates[best_feature]

        # Add the best feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        current_test_f1 = best_eval['test_f1']
        current_train_f1 = best_eval['train_f1']

        results.append({
            'step': step,
            'features': selected_features.copy(),
            'test_f1': current_test_f1,
            'train_f1': current_train_f1,
            'action': f'Added {best_feature} (best F1)',
            'added_feature': best_feature,
            'f1_improvement': best_eval['f1_improvement']
        })

        improvement = best_eval['f1_improvement']
        print(f"{step:<4} {len(selected_features):<8} {current_test_f1:.4f}     {current_train_f1:.4f}      {best_feature:<20} {'Added (+' + f'{improvement:.4f}' + ')':<30}")

        # Show runner-ups for context
        if len(valid_candidates) > 1:
            print("    Runner-ups:")
            sorted_candidates = sorted(valid_candidates.items(),
                                     key=lambda x: x[1]['test_f1'],
                                     reverse=True)[1:4]  # Show top 3 runner-ups
            for feat, eval_data in sorted_candidates:
                print(f"      {feat:<20} - F1: {eval_data['test_f1']:.4f}")

    return results, selected_features, current_test_f1

# Run forward selection
print("Starting fast forward selection for binary classification...")
results, final_features, final_f1 = forward_selection(
    X_train, X_test, y_train, y_test, available_features,
    min_f1_improvement=0.0005,  # Minimum F1 improvement
    max_features=15  # Limit to prevent overfitting
)

print("\n" + "="*80)
print("FINAL RESULTS - BINARY CLASSIFICATION")
print("="*80)
print(f"Final Test F1: {final_f1:.4f}")
print(f"Number of features in final subset: {len(final_features)}")
print(f"Final feature subset: {final_features}")

# Enhanced visualization
plt.figure(figsize=(16, 10))

# Extract data for plotting
steps = [r['step'] for r in results]
test_f1s = [r['test_f1'] for r in results]
train_f1s = [r['train_f1'] for r in results]
num_features = [len(r['features']) for r in results]

# Plot 1: F1 vs Number of Features
plt.subplot(2, 3, 1)
plt.plot(num_features, test_f1s, 'b-o', label='Test F1', linewidth=2, markersize=6)
plt.plot(num_features, train_f1s, 'r-s', label='Train F1', linewidth=2, markersize=6)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.title('F1 vs Number of Features\n(Binary Classification)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(steps, test_f1s, 'b-o', label='Test F1', linewidth=2, markersize=6)
plt.plot(steps, train_f1s, 'r-s', label='Train F1', linewidth=2, markersize=6)
plt.axhline(y=final_f1, color='g', linestyle='--', alpha=0.7,
            label=f'Final Test F1: {final_f1:.4f}')
plt.xlabel('Selection Step')
plt.ylabel('F1 Score')
plt.title('F1 Through Selection Process')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
improvements = [0]  # First step has no improvement
for i in range(1, len(test_f1s)):
    improvements.append(test_f1s[i] - test_f1s[i-1])
colors = ['green' if imp > 0 else 'red' for imp in improvements]
plt.bar(steps, improvements, alpha=0.7, color=colors)
plt.xlabel('Selection Step')
plt.ylabel('F1 Improvement')
plt.title('F1 Improvement per Step')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
overfitting = [train_f1 - test_f1 for train_f1, test_f1 in zip(train_f1s, test_f1s)]
plt.plot(steps, overfitting, 'orange', marker='d', linewidth=2, label='Train - Test Gap')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Warning')
plt.xlabel('Selection Step')
plt.ylabel('Overfitting (Train - Test F1)')
plt.title('Overfitting Monitor')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: F1 improvement per step
plt.subplot(2, 3, 5)
improvements = [0]  # First step has no improvement
for i in range(1, len(test_f1s)):
    improvements.append(test_f1s[i] - test_f1s[i-1])
colors = ['green' if imp > 0 else 'red' for imp in improvements]
plt.bar(steps, improvements, alpha=0.7, color=colors)
plt.xlabel('Selection Step')
plt.ylabel('F1 Improvement')
plt.title('F1 Improvement per Step')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.text(0.1, 0.9, f"Final Results:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f"• Features: {len(final_features)}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f"• Test F1: {final_f1:.4f}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f"• Selection Steps: {len(results)-1}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.45, f"Binary Classification:", fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.4, f"• Suppressed (0): States 1,2,3", fontsize=9, transform=plt.gca().transAxes)
plt.text(0.1, 0.35, f"• ELMing (1): State 4", fontsize=9, transform=plt.gca().transAxes)
plt.text(0.1, 0.25, f"• Min F1 Δ: 0.0005", fontsize=9, transform=plt.gca().transAxes)
plt.axis('off')

# Plot 7: Selection summary
plt.subplot(2, 3, 6)
plt.text(0.1, 0.9, f"Final Results:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.8, f"• Features: {len(final_features)}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f"• Test F1: {final_f1:.4f}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f"• Selection Steps: {len(results)-1}", fontsize=10, transform=plt.gca().transAxes)
plt.text(0.1, 0.45, f"Binary Classification:", fontsize=10, fontweight='bold', transform=plt.gca().transAxes)
plt.text(0.1, 0.4, f"• Suppressed (0): States 1,2,3", fontsize=9, transform=plt.gca().transAxes)
plt.text(0.1, 0.35, f"• ELMing (1): State 4", fontsize=9, transform=plt.gca().transAxes)
plt.text(0.1, 0.25, f"• Min F1 Δ: 0.0005", fontsize=9, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.suptitle('Feature Selection Results - Binary Classification (ELMing vs Suppressed)', y=1.02, fontsize=14)
plt.show()

# Final model evaluation with selected features
if len(final_features) > 0:
    print(f"\n{'='*80}")
    print("FINAL MODEL EVALUATION WITH SELECTED FEATURES - BINARY CLASSIFICATION")
    print(f"{'='*80}")

    clf_final = create_classifier(mode='final')
    clf_final.fit(X_train[final_features], y_train)

    y_pred_final = clf_final.predict(X_test[final_features])
    y_train_pred_final = clf_final.predict(X_train[final_features])

    final_test_f1 = f1_score(y_test, y_pred_final, average='binary', zero_division=0)
    final_train_f1 = f1_score(y_train, y_train_pred_final, average='binary', zero_division=0)

    print(f"Final Test F1: {final_test_f1:.4f}")
    print(f"Final Training F1: {final_train_f1:.4f}")
    print(f"Overfitting Gap (Train-Test F1): {final_train_f1 - final_test_f1:.4f}")

    # Binary classification labels
    unique_classes = sorted(y.unique())
    class_labels = ['Suppressed', 'ELMing']

    # Final Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_final, labels=unique_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Final Confusion Matrix - Binary Classification\n'
              f'Test F1: {final_test_f1:.4f} | Features: {len(final_features)}')
    plt.tight_layout()
    plt.show()

    # Classification Report
    print(f"\nFinal Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=class_labels))

print(f"\n{'='*80}")
print("SUMMARY - BINARY CLASSIFICATION")
print(f"{'='*80}")
print(f"• Algorithm: Fast Forward Selection (F1-only)")
print(f"• Classification: Binary (ELMing vs Suppressed)")
print(f"• Started with {len(available_features)} available features")
print(f"• Final subset contains {len(final_features)} features")
print(f"• Final test F1: {final_f1:.4f}")
print(f"• Selection completed in {len(results)-1} steps")

if len(final_features) > 0:
    print(f"• Final feature subset: {final_features}")

    # Show selection order
    selection_order = []
    for result in results[1:]:  # Skip step 0
        if result['added_feature']:
            selection_order.append(result['added_feature'])

    if selection_order:
        print(f"• Feature selection order: {selection_order}")

    # Show which features were NOT selected
    not_selected_features = [f for f in available_features if f not in final_features]
    if not_selected_features:
        print(f"• Features not selected: {not_selected_features[:10]}{'...' if len(not_selected_features) > 10 else ''}")
        print(f"• Total features not selected: {len(not_selected_features)}")
else:
    print("• No features were selected - none met the F1 improvement criteria")

print(f"\nBinary state mapping summary:")
print(f"  Suppressed (0): Combines original states 1, 2, 3 (Suppressed, Dithering, Mitigated)")
print(f"  ELMing (1): Original state 4 (ELMing)")
