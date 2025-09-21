import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Use the noncausal RF data utilities and trainer
from Noncausal_Random_Forest import (
    load_noncausal_database as noncausal_load_db,
    select_features as noncausal_select_features,
    prepare_data as noncausal_prepare_data,
    train_rf as noncausal_train_rf,
)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Model file paths (reuse same trained artifacts if present)
MODEL_FILE = 'trained_plasma_classifier_random_split.pkl'
FEATURES_FILE = 'model_features_random_split.pkl'
OUT_DIR = 'noncausal_upqtl_images'

# Advanced thresholds for notable changes
NOTABLE_THRESHOLD_DIFFERENCE = 0.3
NOTABLE_PROBABILITY_DIFFERENCE = 0.15
NOTABLE_MAX_SUPPRESSION_DIFFERENCE = 0.2
NOTABLE_CLUSTER_DIFFERENCE = 0.25

def analyze_parameter_distributions(df_cleaned, available_features):
    print("\n" + "="*80)
    print("PARAMETER DISTRIBUTION ANALYSIS")
    print("="*80)
    suppressed_data = df_cleaned[df_cleaned['binary_state'] == 0][available_features]
    elming_data = df_cleaned[df_cleaned['binary_state'] == 1][available_features]
    print(f"Suppressed samples: {len(suppressed_data)}")
    print(f"ELMing samples: {len(elming_data)}")
    separation_scores = {}
    notable_parameters = []
    for param in available_features:
        if param in df_cleaned.columns:
            supp_mean = suppressed_data[param].mean()
            supp_std = suppressed_data[param].std()
            elm_mean = elming_data[param].mean()
            elm_std = elming_data[param].std()
            pooled_std = np.sqrt((supp_std**2 + elm_std**2) / 2)
            separation_score = abs(elm_mean - supp_mean) / pooled_std if pooled_std > 0 else 0
            separation_scores[param] = separation_score
            if separation_score > 0.5:
                notable_parameters.append(param)
                print(f"NOTABLE: {param} - Separation score: {separation_score:.3f}")
                print(f"  Suppressed: {supp_mean:.3e} ± {supp_std:.3e}")
                print(f"  ELMing: {elm_mean:.3e} ± {elm_std:.3e}")
    sorted_params = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 parameters by separation score:")
    for i, (param, score) in enumerate(sorted_params[:5], 1):
        print(f"{i}. {param}: {score:.3f}")
    return notable_parameters, separation_scores

def perform_cluster_analysis(df_cleaned, available_features):
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS FOR PLASMA STATES")
    print("="*80)
    X_cluster = df_cleaned[available_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clustered = df_cleaned.copy()
    df_clustered['cluster'] = cluster_labels
    cluster_analysis = {}
    for cluster_id in range(4):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_analysis[cluster_id] = {
            'size': len(cluster_data),
            'suppressed_ratio': (cluster_data['binary_state'] == 0).mean(),
            'elming_ratio': (cluster_data['binary_state'] == 1).mean(),
            'mean_params': cluster_data[available_features].mean().to_dict()
        }
    notable_clusters = []
    for cluster_id, analysis in cluster_analysis.items():
        if analysis['suppressed_ratio'] > 0.7 or analysis['elming_ratio'] > 0.7:
            notable_clusters.append(cluster_id)
            print(f"NOTABLE CLUSTER {cluster_id}:")
            print(f"  Size: {analysis['size']}")
            print(f"  Suppressed ratio: {analysis['suppressed_ratio']:.3f}")
            print(f"  ELMing ratio: {analysis['elming_ratio']:.3f}")
    if notable_clusters:
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'purple']
        for cluster_id in range(4):
            mask = cluster_labels == cluster_id
            if cluster_id in notable_clusters:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=colors[cluster_id], label=f'Cluster {cluster_id} (Notable)', 
                           alpha=0.7, s=50)
            else:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=colors[cluster_id], label=f'Cluster {cluster_id}', 
                           alpha=0.3, s=30)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.title('Cluster Analysis of Plasma States (iln3iamp @ 75th in sweeps)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, 'notable_clusters.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Image saved: '{out_path}'")
    return cluster_analysis, notable_clusters

def scan_parameter_with_adaptive_ranges(clf, available_features, scenario_name, scenario_params, 
                                       parameter_name, adaptive_range, parameter_unit=""):
    print(f"\n=== {parameter_name.upper()} Scanning for {scenario_name} Scenario ===")
    suppression_probabilities = []
    elming_probabilities = []
    print(f"Scenario parameters:")
    for param, value in scenario_params.items():
        if param in available_features:
            print(f"  {param}: {value:.3e}")
    print(f"\nScanning {parameter_name} with adaptive range: {len(adaptive_range)} points")
    X_scan = []
    for param_value in adaptive_range:
        profile = scenario_params.copy()
        if parameter_name in available_features:
            profile[parameter_name] = param_value
        X_scan.append([profile.get(f, 0) for f in available_features])
    X_scan = np.array(X_scan)
    proba_matrix = clf.predict_proba(X_scan)
    suppression_probabilities = proba_matrix[:, 0].tolist()
    elming_probabilities = proba_matrix[:, 1].tolist()
    suppression_probs_array = np.array(suppression_probabilities)
    above_threshold = suppression_probs_array > 0.5
    if np.any(above_threshold):
        crossing_indices = np.where(np.diff(above_threshold.astype(int)))[0]
        if len(crossing_indices) > 0:
            suppression_threshold_idx = crossing_indices[0]
            suppression_threshold = adaptive_range[suppression_threshold_idx]
        else:
            first_above_idx = np.argmax(above_threshold)
            suppression_threshold = adaptive_range[first_above_idx]
    else:
        suppression_threshold = None
    max_suppression_prob = max(suppression_probabilities)
    max_suppression_param = adaptive_range[np.argmax(suppression_probabilities)]
    print(f"\nResults for {scenario_name} - {parameter_name}:")
    print(f"Suppression threshold: {suppression_threshold:.3e} {parameter_unit}" if suppression_threshold is not None else "No suppression threshold found")
    print(f"Maximum suppression probability: {max_suppression_prob:.3f} at {max_suppression_param:.3e} {parameter_unit}")
    print(f"Minimum ELMing probability: {min(elming_probabilities):.3f}")
    return {
        'scenario': scenario_name,
        'parameter': parameter_name,
        'parameter_range': adaptive_range,
        'suppression_probs': suppression_probabilities,
        'elming_probs': elming_probabilities,
        'threshold': suppression_threshold,
        'max_suppression': max_suppression_prob,
        'max_suppression_param': max_suppression_param,
        'unit': parameter_unit
    }

def test_adaptive_parameter_scanning(clf, available_features, df_cleaned):
    print("\n" + "="*80)
    print("ADAPTIVE PARAMETER SCANNING (iln3iamp fixed at 75th percentile)")
    print("="*80)
    # Build a single scenario: all params at median, ONLY iln3iamp at 75th percentile
    scenario_params = {}
    for param in available_features:
        scenario_params[param] = df_cleaned[param].median()
    if 'iln3iamp' in available_features:
        scenario_params['iln3iamp'] = df_cleaned['iln3iamp'].quantile(0.75)
    scenarios = {'iln3iamp_upper_quartile': scenario_params}
    # Analyze parameter distributions to get adaptive ranges
    notable_params, separation_scores = analyze_parameter_distributions(df_cleaned, available_features)
    adaptive_ranges = {}
    for param in available_features:
        if param in df_cleaned.columns:
            param_data = df_cleaned[param]
            param_min, param_max = param_data.min(), param_data.max()
            adaptive_ranges[param] = np.linspace(param_min, param_max, 1000)
    all_params = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    all_notable_findings = []
    for param_name, separation_score in all_params:
        # Ensure ONLY iln3iamp is fixed at its upper quartile during sweeps; skip sweeping iln3iamp itself
        if param_name == 'iln3iamp':
            continue
        print(f"\n" + "="*60)
        print(f"TESTING PARAMETER: {param_name} (Separation score: {separation_score:.3f}) - {all_params.index((param_name, separation_score)) + 1}/{len(all_params)}")
        print("="*60)
        results = {}
        for scenario_name, params in scenarios.items():
            results[scenario_name] = scan_parameter_with_adaptive_ranges(
                clf, available_features, scenario_name, params,
                param_name, adaptive_ranges[param_name], ""
            )
        notable_findings = []
        max_suppressions = [(name, result['max_suppression']) for name, result in results.items()]
        max_suppressions.sort(key=lambda x: x[1], reverse=True)
        best_scenario, best_suppression = max_suppressions[0]
        worst_scenario, worst_suppression = max_suppressions[-1]
        suppression_difference = best_suppression - worst_suppression
        print(f"Max suppression range: {worst_suppression:.3f} to {best_suppression:.3f} (difference: {suppression_difference:.3f})")
        if suppression_difference > NOTABLE_MAX_SUPPRESSION_DIFFERENCE:
            notable_findings.append(f"NOTABLE: Max suppression difference of {suppression_difference:.3f} between {best_scenario} and {worst_scenario}")
        param_check_points = np.percentile(adaptive_ranges[param_name], [25, 50, 75])
        for check_point in param_check_points:
            prob_differences = []
            for scenario_name, result in results.items():
                param_idx = np.argmin(np.abs(result['parameter_range'] - check_point))
                suppression_prob = result['suppression_probs'][param_idx]
                prob_differences.append((scenario_name, suppression_prob))
            prob_differences.sort(key=lambda x: x[1])
            min_prob = prob_differences[0][1]
            max_prob = prob_differences[-1][1]
            prob_diff = max_prob - min_prob
            if prob_diff > NOTABLE_PROBABILITY_DIFFERENCE:
                notable_findings.append(f"NOTABLE: At {param_name}={check_point:.3e}, suppression probability varies from {min_prob:.3f} to {max_prob:.3f} (difference: {prob_diff:.3f})")
        plt.figure(figsize=(12, 8))
        label = 'Only iln3iamp at 75th percentile'
        result = list(results.values())[0]
        plt.plot(result['parameter_range'], result['suppression_probs'], label=label, linewidth=2.5, linestyle='-', color='blue')
        if param_name == 'n_eped':
            plt.xlabel('Electron Density', fontsize=14)
            plt.title('Suppression Probability of Electron Density (iln3iamp 75th)', fontsize=16, fontweight='bold')
        else:
            plt.xlabel(f'{param_name}', fontsize=14)
            plt.title(f'{param_name}-Dependent Suppression Probability (iln3iamp 75th)', fontsize=16, fontweight='bold')
        plt.ylabel('Suppression Probability', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(OUT_DIR, exist_ok=True)
        out_file = os.path.join(OUT_DIR, f'upqtl_{param_name}_sweep.png')
        plt.savefig(out_file, dpi=1200, bbox_inches='tight')
        print(f"📊 Image saved: '{out_file}'")
        all_notable_findings.extend(notable_findings)
    return all_notable_findings

def perform_decision_boundary_analysis(clf, available_features, df_cleaned):
    print("\n" + "="*80)
    print("DECISION BOUNDARY ANALYSIS")
    print("="*80)
    feature_importance = clf.feature_importances_
    feature_importance_pairs = list(zip(available_features, feature_importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in feature_importance_pairs[:2]]
    print(f"Top 2 features by importance: {top_features}")
    x_feature, y_feature = top_features[0], top_features[1]
    x_min, x_max = df_cleaned[x_feature].min(), df_cleaned[x_feature].max()
    y_min, y_max = df_cleaned[y_feature].min(), df_cleaned[y_feature].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    test_points = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = {}
            for feature in available_features:
                if feature == x_feature:
                    point[feature] = xx[i, j]
                elif feature == y_feature:
                    point[feature] = yy[i, j]
                else:
                    point[feature] = df_cleaned[feature].mean()
            test_points.append([point.get(f, 0) for f in available_features])
    test_array = np.array(test_points)
    probabilities = clf.predict_proba(test_array)[:, 0]
    probabilities = probabilities.reshape(xx.shape)
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(xx, yy, probabilities, levels=20, cmap='RdYlBu_r')
    plt.colorbar(contour, label='Suppression Probability')
    for state in [0, 1]:
        mask = df_cleaned['binary_state'] == state
        color = 'blue' if state == 0 else 'red'
        label = 'Suppressed' if state == 0 else 'ELMing'
        plt.scatter(df_cleaned[mask][x_feature], df_cleaned[mask][y_feature], 
                   c=color, label=label, alpha=0.6, s=20)
    plt.xlabel(x_feature, fontsize=14)
    plt.ylabel(y_feature, fontsize=14)
    plt.title('Decision Boundary Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'decision_boundary_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📊 Image saved: '{out_path}'")
    return top_features

def main():
    print("=== Noncausal Plasma Parameter Visualization (iln3iamp at upper quartile during sweeps) ===")
    print("Using noncausal_database.csv and the noncausal feature list\n")
    # Load noncausal database and select noncausal features via Noncausal_Random_Forest utilities
    df = noncausal_load_db()
    features = noncausal_select_features(df)
    # Prepare data (creates binary_state and splits); also reconstruct cleaned df for analysis
    X_train, X_test, y_train, y_test, class_names = noncausal_prepare_data(df, features)
    df_cleaned = df.dropna(subset=features, how='any').copy()
    df_cleaned['binary_state'] = np.where(df_cleaned['label'] == 'ELMing', 1, 0)
    available_features = features
    # Train the RandomForest using the same trainer as Noncausal_Random_Forest
    clf = noncausal_train_rf(X_train, y_train)
    print("\n" + "="*80)
    print("RUNNING ANALYSIS WITH iln3iamp FIXED AT 75th PERCENTILE")
    print("="*80)
    all_notable_findings = []
    notable_params, separation_scores = analyze_parameter_distributions(df_cleaned, available_features)
    if notable_params:
        all_notable_findings.append(f"NOTABLE: Found {len(notable_params)} parameters with significant separation between states")
    cluster_analysis, notable_clusters = perform_cluster_analysis(df_cleaned, available_features)
    if notable_clusters:
        all_notable_findings.append(f"NOTABLE: Found {len(notable_clusters)} clusters with distinct state characteristics")
    adaptive_findings = test_adaptive_parameter_scanning(clf, available_features, df_cleaned)
    all_notable_findings.extend(adaptive_findings)
    top_features = perform_decision_boundary_analysis(clf, available_features, df_cleaned)
    all_notable_findings.append(f"NOTABLE: Decision boundary analysis reveals {top_features[0]} and {top_features[1]} as key discriminators")
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY (iln3iamp @ 75th during sweeps)")
    print("="*80)
    if all_notable_findings:
        print("🎯 NOTABLE FINDINGS:")
        for i, finding in enumerate(all_notable_findings, 1):
            print(f"{i}. {finding}")
        print(f"\n📊 Generated visualizations in '{OUT_DIR}'")
    else:
        print("❌ No notable differences found with analysis")
    return all_notable_findings

if __name__ == "__main__":
    main()


