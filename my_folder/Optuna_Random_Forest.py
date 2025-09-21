import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna
import warnings
import time
from datetime import datetime
import os
import gc
import psutil
import sys
from tqdm import tqdm

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPU libraries not available. Using CPU-only mode.")
    print("   Install with: pip install cupy-cuda11x")

warnings.filterwarnings('ignore')
plt.close('all')

# System configuration
N_CORES = min(16, os.cpu_count() - 2)  # Use 16 cores to prevent overload
print(f"Using {N_CORES} CPU cores for parallel processing")

# GPU configuration
if GPU_AVAILABLE:
    try:
        print(f"GPU Memory: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.1f} GB available")
        print(f"GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except:
        print("GPU info not available")

def memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def print_status(message, level=0):
    """Print status message with indentation"""
    indent = "  " * level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {indent}{message}")

def print_phase_header(phase_num, total_phases, phase_name):
    """Print phase header with progress"""
    print(f"\n{'='*60}")
    print(f"📊 PHASE {phase_num}/{total_phases}: {phase_name}")
    print(f"Progress: {phase_num-1}/{total_phases} phases completed ({((phase_num-1)/total_phases)*100:.1f}%)")
    print(f"Memory usage: {memory_usage():.1f} MB")
    print(f"{'='*60}")

def load_and_prepare_data():
    """Load and prepare the dataset from CSV with enhanced preprocessing"""
    print_status("Loading and preparing data...")
    print_status(f"Memory usage: {memory_usage():.1f} MB", level=1)
    
    print_status("Reading CSV file...", level=1)
    df = pd.read_csv('plasma_data.csv')
    print_status(f"Loaded {len(df)} rows, {len(df.columns)} columns", level=2)
    
    print_status("Cleaning data...", level=1)
    df = df[df['shot'] != 191675].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['original_index'] = df.index
    print_status(f"After cleaning: {len(df)} rows", level=2)
    
    # Add engineered features
    print_status("Adding engineered features...", level=1)
    features_to_add = [
        ('shot_duration', lambda x: x.groupby('shot')['time'].transform('max') - x.groupby('shot')['time'].transform('min')),
        ('time_normalized', lambda x: x.groupby('shot')['time'].transform(lambda g: (g - g.min()) / (g.max() - g.min()))),
        ('time_std', lambda x: x.groupby('shot')['time'].transform('std')),
        ('time_mean', lambda x: x.groupby('shot')['time'].transform('mean')),
        ('time_skew', lambda x: x.groupby('shot')['time'].transform(lambda g: g.skew() if len(g) > 2 else 0))
    ]
    
    for feature_name, feature_func in features_to_add:
        print_status(f"Adding {feature_name}...", level=2)
        df[feature_name] = feature_func(df)
    
    print_status(f"Memory usage after feature engineering: {memory_usage():.1f} MB", level=1)
    return df

def prepare_binary_classification(df):
    """Prepare data for binary classification with feature engineering"""
    print_status("Preparing binary classification...")
    
    selected_features = [
        'iln3iamp', 'iln2iamp', 'iun2iamp', 'iun3iamp', 'iln3iphase', 'iln2iphase', 'iun2iphase', 'iun3iphase',
        'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs04_max_avg'
    ]
    
    # Add engineered features
    engineered_features = ['shot_duration', 'time_normalized', 'time_std', 'time_mean', 'time_skew']
    all_features = selected_features + engineered_features
    
    print_status("Checking feature availability...", level=1)
    available_features = [f for f in all_features if f in df.columns]
    print_status(f"Available features: {len(available_features)}/{len(all_features)}", level=2)
    
    print_status("Removing rows with missing values...", level=1)
    df_cleaned = df.dropna(subset=available_features, how='any')
    print_status(f"After removing NaN: {len(df_cleaned)} rows", level=2)
    
    print_status("Filtering valid states...", level=1)
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    print_status(f"After filtering states: {len(df_cleaned)} rows", level=2)
    
    # Map states to binary: 1-3 -> 0 (Suppressed), 4 -> 1 (ELMing)
    print_status("Creating binary state labels...", level=1)
    df_cleaned['binary_state'] = df_cleaned['state'].apply(lambda x: 0 if x in [1,2,3] else 1)
    
    columns_to_keep = ['shot', 'time', 'binary_state'] + available_features
    df_cleaned = df_cleaned[columns_to_keep].copy()
    
    print_status(f"Memory usage after cleaning: {memory_usage():.1f} MB", level=1)
    return df_cleaned, available_features

def split_data_by_shots(df_cleaned, available_features):
    """Split data by unique shots: 60% train, 20% CV, 20% test"""
    print_status("Splitting data by shots...")
    
    unique_shots = df_cleaned['shot'].unique()
    num_shots = len(unique_shots)
    print_status(f"Total unique shots: {num_shots}", level=1)
    
    # Shuffle shots first, then split
    print_status("Shuffling shots...", level=1)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    
    # Split shuffled shots: 60% train, 20% CV, 20% test
    train_count = int(np.floor(0.60 * num_shots))
    cv_count = int(np.floor(0.20 * num_shots))
    
    train_shots = shuffled_shots[:train_count]
    cv_shots = shuffled_shots[train_count:train_count + cv_count]
    test_shots = shuffled_shots[train_count + cv_count:]
    
    print_status(f"Split sizes - Train: {len(train_shots)}, CV: {len(cv_shots)}, Test: {len(test_shots)}", level=1)
    
    # Get data for each split
    print_status("Creating data splits...", level=1)
    train_df = df_cleaned[df_cleaned['shot'].isin(train_shots)]
    cv_df = df_cleaned[df_cleaned['shot'].isin(cv_shots)]
    test_df = df_cleaned[df_cleaned['shot'].isin(test_shots)]
    
    X_train = train_df[available_features]
    y_train = train_df['binary_state']
    X_cv = cv_df[available_features]
    y_cv = cv_df['binary_state']
    X_test = test_df[available_features]
    y_test = test_df['binary_state']
    
    print_status(f"Final split sizes - Train: {len(X_train)}, CV: {len(X_cv)}, Test: {len(X_test)}", level=1)
    
    # Clear memory
    del train_df, cv_df, test_df
    gc.collect()
    
    return X_train, X_cv, X_test, y_train, y_cv, y_test

def feature_selection_analysis(X_train, y_train, available_features):
    """Perform comprehensive feature selection analysis"""
    print_status("=== Feature Selection Analysis ===")
    
    # Multiple feature selection methods
    print_status("Running F-test feature selection...", level=1)
    f_scores, f_p_values = f_classif(X_train, y_train)
    print_status("F-test completed", level=2)
    
    print_status("Running mutual information feature selection...", level=1)
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    print_status("Mutual information completed", level=2)
    
    # Handle infinite values in log_p_value
    print_status("Processing feature scores...", level=1)
    log_p_values = -np.log10(f_p_values)
    log_p_values = np.where(np.isinf(log_p_values), 100, log_p_values)  # Cap at 100
    
    feature_analysis = pd.DataFrame({
        'feature': available_features,
        'f_score': f_scores,
        'f_p_value': f_p_values,
        'log_p_value': log_p_values,
        'mutual_info': mi_scores
    })
    
    # Combined score (normalized and weighted)
    feature_analysis['f_score_norm'] = (feature_analysis['f_score'] - feature_analysis['f_score'].min()) / (feature_analysis['f_score'].max() - feature_analysis['f_score'].min())
    feature_analysis['mi_score_norm'] = (feature_analysis['mutual_info'] - feature_analysis['mutual_info'].min()) / (feature_analysis['mutual_info'].max() - feature_analysis['mutual_info'].min())
    feature_analysis['combined_score'] = 0.6 * feature_analysis['f_score_norm'] + 0.4 * feature_analysis['mi_score_norm']
    
    feature_analysis = feature_analysis.sort_values('combined_score', ascending=False)
    
    print_status("Top 10 features by combined score:", level=1)
    print(feature_analysis[['feature', 'f_score', 'mutual_info', 'combined_score']].head(10).to_string(index=False))
    
    # Select top features (keep at least 8, at most 15)
    n_features = min(max(8, len(available_features) // 2), 15)
    top_features = feature_analysis.head(n_features)['feature'].tolist()
    
    print_status(f"Selected {len(top_features)} features: {top_features}", level=1)
    
    return top_features, feature_analysis

def objective(trial, X_train, y_train, X_cv, y_cv):
    """Optuna objective function for hyperparameter optimization"""
    try:
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 800, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 50) if trial.suggest_categorical('use_max_depth', [True, False]) else None,
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42,
            'n_jobs': 1
        }
        
        # GPU acceleration for data transfer
        if GPU_AVAILABLE:
            try:
                # Transfer data to GPU
                X_train_gpu = cp.asarray(X_train.values if hasattr(X_train, 'values') else X_train)
                y_train_gpu = cp.asarray(y_train.values if hasattr(y_train, 'values') else y_train)
                X_cv_gpu = cp.asarray(X_cv.values if hasattr(X_cv, 'values') else X_cv)
                y_cv_gpu = cp.asarray(y_cv.values if hasattr(y_cv, 'values') else y_cv)
                
                # Transfer back to CPU for sklearn
                X_train_cpu = cp.asnumpy(X_train_gpu)
                y_train_cpu = cp.asnumpy(y_train_gpu)
                X_cv_cpu = cp.asnumpy(X_cv_gpu)
                y_cv_cpu = cp.asnumpy(y_cv_gpu)
                
                # Clear GPU memory
                del X_train_gpu, y_train_gpu, X_cv_gpu, y_cv_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
                gpu_used = True
            except Exception as e:
                X_train_cpu, y_train_cpu, X_cv_cpu, y_cv_cpu = X_train, y_train, X_cv, y_cv
                gpu_used = False
        else:
            X_train_cpu, y_train_cpu, X_cv_cpu, y_cv_cpu = X_train, y_train, X_cv, y_cv
            gpu_used = False
        
        # Train model
        clf = RandomForestClassifier(**params)
        clf.fit(X_train_cpu, y_train_cpu)
        
        # Cross-validation on training data
        cv_scores = cross_val_score(clf, X_train_cpu, y_train_cpu, cv=3, scoring='accuracy', n_jobs=1)
        
        # Direct CV evaluation
        cv_pred = clf.predict(X_cv_cpu)
        cv_accuracy = accuracy_score(y_cv_cpu, cv_pred)
        
        # Return CV accuracy as objective (Optuna maximizes by default)
        return cv_accuracy
        
    except Exception as e:
        print(f"Error in trial: {e}")
        return 0.0

def optimize_with_optuna(X_train, y_train, X_cv, y_cv, n_trials=100):
    """Optimize hyperparameters using Optuna"""
    print_status("Starting Optuna optimization...")
    print_status(f"Number of trials: {n_trials}", level=1)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_cv, y_cv),
        n_trials=n_trials,
        n_jobs=1,  # Use single job to avoid conflicts
        show_progress_bar=True
    )
    
    print_status(f"Best CV accuracy: {study.best_value:.4f}", level=1)
    print_status(f"Best parameters: {study.best_params}", level=1)
    
    return study

def final_model_evaluation(best_params, X_train, y_train, X_cv, y_cv, X_test, y_test, top_features):
    """Evaluate final model with best parameters"""
    print_status("Training final model with best parameters...")
    
    # Add default parameters
    final_params = best_params.copy()
    final_params['random_state'] = 42
    final_params['n_jobs'] = N_CORES
    
    # GPU acceleration for final training
    if GPU_AVAILABLE:
        try:
            print_status("Using GPU acceleration for final training...", level=1)
            X_train_gpu = cp.asarray(X_train.values if hasattr(X_train, 'values') else X_train)
            y_train_gpu = cp.asarray(y_train.values if hasattr(y_train, 'values') else y_train)
            X_cv_gpu = cp.asarray(X_cv.values if hasattr(X_cv, 'values') else X_cv)
            y_cv_gpu = cp.asarray(y_cv.values if hasattr(y_cv, 'values') else y_cv)
            X_test_gpu = cp.asarray(X_test.values if hasattr(X_test, 'values') else X_test)
            y_test_gpu = cp.asarray(y_test.values if hasattr(y_test, 'values') else y_test)
            
            # Transfer back to CPU for sklearn
            X_train_cpu = cp.asnumpy(X_train_gpu)
            y_train_cpu = cp.asnumpy(y_train_gpu)
            X_cv_cpu = cp.asnumpy(X_cv_gpu)
            y_cv_cpu = cp.asnumpy(y_cv_gpu)
            X_test_cpu = cp.asnumpy(X_test_gpu)
            y_test_cpu = cp.asnumpy(y_test_gpu)
            
            # Clear GPU memory
            del X_train_gpu, y_train_gpu, X_cv_gpu, y_cv_gpu, X_test_gpu, y_test_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print_status(f"GPU transfer failed: {e}, using CPU", level=1)
            X_train_cpu, y_train_cpu, X_cv_cpu, y_cv_cpu, X_test_cpu, y_test_cpu = X_train, y_train, X_cv, y_cv, X_test, y_test
    else:
        X_train_cpu, y_train_cpu, X_cv_cpu, y_cv_cpu, X_test_cpu, y_test_cpu = X_train, y_train, X_cv, y_cv, X_test, y_test
    
    # Train final model
    final_clf = RandomForestClassifier(**final_params)
    final_clf.fit(X_train_cpu, y_train_cpu)
    
    # Evaluate on all sets
    train_pred = final_clf.predict(X_train_cpu)
    cv_pred = final_clf.predict(X_cv_cpu)
    test_pred = final_clf.predict(X_test_cpu)
    
    train_acc = accuracy_score(y_train_cpu, train_pred)
    cv_acc = accuracy_score(y_cv_cpu, cv_pred)
    test_acc = accuracy_score(y_test_cpu, test_pred)
    
    print_status(f"Final Results:", level=1)
    print_status(f"  Train Accuracy: {train_acc:.4f}", level=2)
    print_status(f"  CV Accuracy: {cv_acc:.4f}", level=2)
    print_status(f"  Test Accuracy: {test_acc:.4f}", level=2)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': top_features,
        'importance': final_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print_status("Top 5 features by importance:", level=1)
    print(feature_importance.head().to_string(index=False))
    
    return final_clf, feature_importance

def plot_results(study, feature_importance, X_test, y_test, final_clf):
    """Plot optimization results and model performance"""
    print_status("Generating plots...")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"optuna_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Optimization history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    
    plt.subplot(2, 2, 2)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importances')
    
    plt.subplot(2, 2, 3)
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title('Parallel Coordinate Plot')
    
    plt.subplot(2, 2, 4)
    feature_importance.head(10).plot(x='feature', y='importance', kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/optimization_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Confusion Matrix
    plt.figure(figsize=(15, 5))
    
    # Test set confusion matrix
    plt.subplot(1, 3, 1)
    cm_test = confusion_matrix(y_test, final_clf.predict(X_test))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve
    plt.subplot(1, 3, 2)
    y_test_proba = final_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    
    # Feature importance
    plt.subplot(1, 3, 3)
    feature_importance.head(10).plot(x='feature', y='importance', kind='barh')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/model_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Plots saved to: {results_dir}", level=1)
    return results_dir

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("🚀 Starting Optuna-based Random Forest Hyperparameter Optimization")
    print("="*60)
    print(f"Initial memory usage: {memory_usage():.1f} MB")
    
    # Check if we should use simplified mode
    import sys
    n_trials = 50 if len(sys.argv) > 1 and sys.argv[1] == '--quick' else 100
    
    # Load and prepare data
    print_phase_header(1, 4, "Loading and preparing data")
    df = load_and_prepare_data()
    df_cleaned, available_features = prepare_binary_classification(df)
    X_train, X_cv, X_test, y_train, y_cv, y_test = split_data_by_shots(df_cleaned, available_features)
    
    print_status(f"Dataset sizes:")
    print_status(f"  Training: {len(X_train)} samples", level=1)
    print_status(f"  CV: {len(X_cv)} samples", level=1)
    print_status(f"  Test: {len(X_test)} samples", level=1)
    print_status(f"  Features: {len(available_features)}", level=1)
    
    # Feature selection
    print_phase_header(2, 4, "Feature selection analysis")
    top_features, feature_analysis = feature_selection_analysis(X_train, y_train, available_features)
    
    # Use selected features
    print_status("Selecting top features for modeling...", level=1)
    X_train_selected = X_train[top_features]
    X_cv_selected = X_cv[top_features]
    X_test_selected = X_test[top_features]
    
    # Clear memory
    print_status("Clearing memory...", level=1)
    del X_train, X_cv, X_test
    gc.collect()
    
    # Optuna optimization
    print_phase_header(3, 4, "Optuna hyperparameter optimization")
    study = optimize_with_optuna(X_train_selected, y_train, X_cv_selected, y_cv, n_trials=n_trials)
    
    # Final evaluation
    print_phase_header(4, 4, "Final model evaluation")
    final_clf, feature_importance = final_model_evaluation(
        study.best_params, X_train_selected, y_train, X_cv_selected, y_cv, 
        X_test_selected, y_test, top_features
    )
    
    # Generate plots
    print_status("Generating plots and saving results...")
    results_dir = plot_results(study, feature_importance, X_test_selected, y_test, final_clf)
    
    # Save results
    print_status("Saving results...", level=1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and results
    import joblib
    joblib.dump(final_clf, f"{results_dir}/optuna_rf_model.pkl")
    feature_importance.to_csv(f"{results_dir}/feature_importance.csv", index=False)
    feature_analysis.to_csv(f"{results_dir}/feature_analysis.csv", index=False)
    
    # Save optimization results
    optimization_summary = pd.DataFrame({
        'best_value': [study.best_value],
        'best_params': [str(study.best_params)],
        'n_trials': [len(study.trials)],
        'optimization_time': [time.time() - start_time]
    })
    optimization_summary.to_csv(f"{results_dir}/optimization_summary.csv", index=False)
    
    # Save system info
    system_info = {
        'cpu_cores_used': N_CORES,
        'gpu_available': GPU_AVAILABLE,
        'total_memory_mb': memory_usage(),
        'n_trials': n_trials,
        'timestamp': timestamp
    }
    pd.DataFrame([system_info]).to_csv(f"{results_dir}/system_info.csv", index=False)
    
    total_time = (time.time() - start_time)/60
    print(f"\n✅ COMPLETED!")
    print(f"Results saved to: {results_dir}")
    print(f"Total execution time: {total_time:.2f} minutes")
    print(f"Best CV accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Final memory usage: {memory_usage():.1f} MB")
    print("="*60)

if __name__ == "__main__":
    main()
