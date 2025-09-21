import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Model file paths
MODEL_FILE = 'trained_plasma_classifier.pkl'
FEATURES_FILE = 'model_features.pkl'

# Advanced thresholds for notable changes
NOTABLE_THRESHOLD_DIFFERENCE = 0.3  # Reduced threshold for more sensitive detection
NOTABLE_PROBABILITY_DIFFERENCE = 0.15  # 15% difference in suppression probability
NOTABLE_MAX_SUPPRESSION_DIFFERENCE = 0.2  # 20% difference in max suppression
NOTABLE_CLUSTER_DIFFERENCE = 0.25  # 25% difference between clusters

def load_and_prepare_data():
    """
    Load and prepare the dataset from CSV
    """
    print("=== Loading Plasma Dataset from CSV ===")
    
    # Load the CSV dataset
    df = pd.read_csv('plasma_data.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()
    print(f"After removing shot 191675: {df.shape}")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Add an index column to keep track of original row numbers
    df['original_index'] = df.index
    
    return df

def select_features_and_clean(df):
    """
    Select features and clean the dataset for binary classification
    """
    print("\n=== Feature Selection and Data Cleaning ===")
    
    # Select features based on the provided example
    selected_features = [
        'iln3iamp', 'tribot',
        'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs04_max_avg', 'fs_up_sum', 'fs_sum'
    ]
    
    # Check which features are available
    available_features = [f for f in selected_features if f in df.columns]
    print(f"Available features: {available_features}")
    print(f"Features not found: {[f for f in selected_features if f not in df.columns]}")
    
    # Clean the dataframe - remove rows with missing values in selected features
    df_cleaned = df.dropna(subset=available_features, how='any')
    print(f"After removing rows with missing values: {len(df_cleaned)}")
    
    # Remove rows where state is 'N/A' (0)
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    print(f"After removing N/A states: {len(df_cleaned)}")
    
    # Map states to binary classification
    def map_states_to_binary(state):
        if state in [1, 2, 3]:  # Suppressed, Dithering, Mitigated -> Suppressed
            return 0
        elif state == 4:  # ELMing
            return 1
        else:
            return state
    
    df_cleaned['binary_state'] = df_cleaned['state'].apply(map_states_to_binary)
    
    # Check binary state distribution
    binary_state_counts = df_cleaned['binary_state'].value_counts().sort_index()
    binary_state_names = {0: 'Suppressed', 1: 'ELMing'}
    
    print(f"\nBinary state distribution:")
    for state, count in binary_state_counts.items():
        if state in binary_state_names:
            print(f"  {binary_state_names[state]} (State {state}): {count:6d} records ({count/len(df_cleaned)*100:.1f}%)")
        else:
            print(f"  State {state}: {count:6d} records ({count/len(df_cleaned)*100:.1f}%)")
    
    return df_cleaned, available_features, binary_state_names

def prepare_chronological_splits(df_cleaned, available_features):
    """
    Prepare training and testing data with chronological split (last 20% as test)
    """
    print("\n=== Data Splitting (Chronological) ===")
    
    # Prepare input (X) and target (y)
    X = df_cleaned[available_features]
    y = df_cleaned['binary_state']
    
    # Create chronological splits: last 20% as test, first 80% as training
    split_point = int(len(df_cleaned) * 0.999)
    
    # Split the data
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    # Get original indices for reference
    idx_train = df_cleaned['original_index'].iloc[:split_point]
    idx_test = df_cleaned['original_index'].iloc[split_point:]
    
    print(f"Training set size: {len(X_train)} (first 80%)")
    print(f"Testing set size: {len(X_test)} (last 20%)")
    print(f"Training data covers original indices: {idx_train.min()} to {idx_train.max()}")
    print(f"Testing data covers original indices: {idx_test.min()} to {idx_test.max()}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier for binary classification
    """
    print("\n=== Training Random Forest (Binary Classification) ===")
    
    # Train a Random Forest Classifier with parameters from the example
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    clf.fit(X_train, y_train)
    
    print("Random Forest training completed!")
    
    return clf

def save_model(clf, available_features):
    """
    Save the trained model and features for later use
    """
    print(f"\n=== Saving Model ===")
    
    # Save the trained model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    
    # Save the feature names
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(available_features, f)
    
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Features saved to: {FEATURES_FILE}")

def load_model():
    """
    Load the trained model and features
    """
    print(f"\n=== Loading Model ===")
    
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} not found. Please train the model first.")
    
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file {FEATURES_FILE} not found. Please train the model first.")
    
    # Load the trained model
    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)
    
    # Load the feature names
    with open(FEATURES_FILE, 'rb') as f:
        available_features = pickle.load(f)
    
    print(f"Model loaded from: {MODEL_FILE}")
    print(f"Features loaded from: {FEATURES_FILE}")
    print(f"Number of features: {len(available_features)}")
    
    return clf, available_features

def analyze_parameter_distributions(df_cleaned, available_features):
    """
    Analyze parameter distributions to identify potential separation between suppressed and ELMing states
    """
    print("\n" + "="*80)
    print("PARAMETER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Separate data by state
    suppressed_data = df_cleaned[df_cleaned['binary_state'] == 0][available_features]
    elming_data = df_cleaned[df_cleaned['binary_state'] == 1][available_features]
    
    print(f"Suppressed samples: {len(suppressed_data)}")
    print(f"ELMing samples: {len(elming_data)}")
    
    # Calculate statistics for each parameter
    separation_scores = {}
    notable_parameters = []
    
    for param in available_features:
        if param in df_cleaned.columns:
            # Calculate means and standard deviations
            supp_mean = suppressed_data[param].mean()
            supp_std = suppressed_data[param].std()
            elm_mean = elming_data[param].mean()
            elm_std = elming_data[param].std()
            
            # Calculate separation score (difference in means normalized by pooled std)
            pooled_std = np.sqrt((supp_std**2 + elm_std**2) / 2)
            separation_score = abs(elm_mean - supp_mean) / pooled_std if pooled_std > 0 else 0
            
            separation_scores[param] = separation_score
            
            # Check if this parameter shows notable separation
            if separation_score > 0.5:  # Moderate separation threshold
                notable_parameters.append(param)
                print(f"NOTABLE: {param} - Separation score: {separation_score:.3f}")
                print(f"  Suppressed: {supp_mean:.3e} ± {supp_std:.3e}")
                print(f"  ELMing: {elm_mean:.3e} ± {elm_std:.3e}")
    
    # Sort parameters by separation score
    sorted_params = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 parameters by separation score:")
    for i, (param, score) in enumerate(sorted_params[:5], 1):
        print(f"{i}. {param}: {score:.3f}")
    
    return notable_parameters, separation_scores

def create_adaptive_parameter_scenarios(clf, available_features, df_cleaned):
    """
    Create adaptive parameter scenarios based on actual data distributions
    """
    print("\n" + "="*80)
    print("CREATING ADAPTIVE PARAMETER SCENARIOS")
    print("="*80)
    
    # Analyze parameter distributions
    notable_params, separation_scores = analyze_parameter_distributions(df_cleaned, available_features)
    
    # Get parameter ranges from actual data
    param_ranges = {}
    for param in available_features:
        if param in df_cleaned.columns:
            param_ranges[param] = {
                'min': df_cleaned[param].min(),
                'max': df_cleaned[param].max(),
                'mean': df_cleaned[param].mean(),
                'std': df_cleaned[param].std()
            }
    
    # Create scenarios based on actual data characteristics
    scenarios = {}
    
    # Scenario 1: Balanced scenario (all parameters at their median values)
    scenarios['balanced_importance'] = {}
    for param in available_features:
        # Use median values for all parameters for a balanced baseline
        scenarios['balanced_importance'][param] = df_cleaned[param].median()
    
    # Scenario 2: Model-optimized scenario (based on model predictions)
    scenarios['model_optimized'] = {}
    # Start with mean values
    for param in available_features:
        scenarios['model_optimized'][param] = param_ranges[param]['mean']
    
    # Optimize important parameters
    for param in notable_params[:2]:
        # Test both extremes and choose the one with higher suppression probability
        test_low = scenarios['model_optimized'].copy()
        test_low[param] = param_ranges[param]['min']
        test_high = scenarios['model_optimized'].copy()
        test_high[param] = param_ranges[param]['max']
        
        # Predict suppression probabilities
        X_test_low = np.array([[test_low.get(f, 0) for f in available_features]])
        X_test_high = np.array([[test_high.get(f, 0) for f in available_features]])
        
        prob_low = clf.predict_proba(X_test_low)[0][0]  # Suppression probability
        prob_high = clf.predict_proba(X_test_high)[0][0]
        
        # Choose the value that gives higher suppression probability
        if prob_high > prob_low:
            scenarios['model_optimized'][param] = param_ranges[param]['max']
        else:
            scenarios['model_optimized'][param] = param_ranges[param]['min']
    
    print(f"Created {len(scenarios)} adaptive scenarios:")
    for scenario_name, params in scenarios.items():
        print(f"  {scenario_name}: {len(params)} parameters")
    
    return scenarios

def perform_cluster_analysis(df_cleaned, available_features):
    """
    Perform cluster analysis to identify distinct plasma states
    """
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS FOR PLASMA STATES")
    print("="*80)
    
    # Prepare data for clustering
    X_cluster = df_cleaned[available_features].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster characteristics
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
    
    # Find clusters with notable differences
    notable_clusters = []
    for cluster_id, analysis in cluster_analysis.items():
        if analysis['suppressed_ratio'] > 0.7 or analysis['elming_ratio'] > 0.7:
            notable_clusters.append(cluster_id)
            print(f"NOTABLE CLUSTER {cluster_id}:")
            print(f"  Size: {analysis['size']}")
            print(f"  Suppressed ratio: {analysis['suppressed_ratio']:.3f}")
            print(f"  ELMing ratio: {analysis['elming_ratio']:.3f}")
    
    # Create visualization if notable clusters found
    if notable_clusters:
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
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
        plt.title('Cluster Analysis of Plasma States', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('notable_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Image saved: 'notable_clusters.png'")
    
    return cluster_analysis, notable_clusters

def scan_parameter_with_adaptive_ranges(clf, available_features, scenario_name, scenario_params, 
                                       parameter_name, adaptive_range, parameter_unit=""):
    """
    Scan parameter with adaptive ranges based on actual data distribution
    """
    print(f"\n=== {parameter_name.upper()} Scanning for {scenario_name} Scenario ===")
    
    # Initialize results storage
    suppression_probabilities = []
    elming_probabilities = []
    
    print(f"Scenario parameters:")
    for param, value in scenario_params.items():
        if param in available_features:
            print(f"  {param}: {value:.3e}")
    
    print(f"\nScanning {parameter_name} with adaptive range: {len(adaptive_range)} points")
    
    # Scan through parameter values
    for param_value in adaptive_range:
        # Create a test profile with current parameter value
        test_profile = scenario_params.copy()
        
        # Set the parameter being scanned
        if parameter_name in available_features:
            test_profile[parameter_name] = param_value
        
        # Reshape for prediction (single sample)
        X_test_single = np.array([[test_profile.get(f, 0) for f in available_features]])
        
        # Get prediction probabilities
        proba = clf.predict_proba(X_test_single)[0]
        suppression_prob = proba[0]  # Probability of suppression (class 0)
        elming_prob = proba[1]      # Probability of ELMing (class 1)
        
        suppression_probabilities.append(suppression_prob)
        elming_probabilities.append(elming_prob)
    
    # Find suppression threshold (where suppression probability crosses 0.5)
    suppression_probs_array = np.array(suppression_probabilities)
    above_threshold = suppression_probs_array > 0.5
    
    # Check if there are any values above threshold
    if np.any(above_threshold):
        # Find the first crossing point
        crossing_indices = np.where(np.diff(above_threshold.astype(int)))[0]
        if len(crossing_indices) > 0:
            # Use the first crossing point
            suppression_threshold_idx = crossing_indices[0]
            suppression_threshold = adaptive_range[suppression_threshold_idx]
        else:
            # If no crossing but some values are above threshold, use the first above-threshold value
            first_above_idx = np.argmax(above_threshold)
            suppression_threshold = adaptive_range[first_above_idx]
    else:
        # No values above threshold
        suppression_threshold = None
    
    # Find maximum suppression probability
    max_suppression_prob = max(suppression_probabilities)
    max_suppression_param = adaptive_range[np.argmax(suppression_probabilities)]
    
    # Print results
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
    """
    Test adaptive parameter scanning based on actual data characteristics
    """
    print("\n" + "="*80)
    print("ADAPTIVE PARAMETER SCANNING")
    print("="*80)
    
    # Create adaptive scenarios
    scenarios = create_adaptive_parameter_scenarios(clf, available_features, df_cleaned)
    
    # Analyze parameter distributions to get adaptive ranges
    notable_params, separation_scores = analyze_parameter_distributions(df_cleaned, available_features)
    
    # Create adaptive ranges for top parameters
    adaptive_ranges = {}
    for param in available_features:
        if param in df_cleaned.columns:
            param_data = df_cleaned[param]
            param_min, param_max = param_data.min(), param_data.max()
            
            # Create adaptive range based on data distribution
            if param in notable_params:
                # For notable parameters, use finer sampling around the range
                adaptive_ranges[param] = np.linspace(param_min, param_max, 350)
            else:
                # For other parameters, use coarser sampling
                adaptive_ranges[param] = np.linspace(param_min, param_max, 350)
    
    # Test top 3 parameters with highest separation scores
    top_params = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    all_notable_findings = []
    
    for param_name, separation_score in top_params:
        print(f"\n" + "="*60)
        print(f"TESTING PARAMETER: {param_name} (Separation score: {separation_score:.3f})")
        print("="*60)
        
        # Scan this parameter for each scenario
        results = {}
        for scenario_name, scenario_params in scenarios.items():
            results[scenario_name] = scan_parameter_with_adaptive_ranges(
                clf, available_features, scenario_name, scenario_params,
                param_name, adaptive_ranges[param_name], ""
            )
        
        # Analyze for notable differences
        notable_findings = []
        
        # Find scenarios with highest and lowest max suppression
        max_suppressions = [(name, result['max_suppression']) for name, result in results.items()]
        max_suppressions.sort(key=lambda x: x[1], reverse=True)
        
        best_scenario, best_suppression = max_suppressions[0]
        worst_scenario, worst_suppression = max_suppressions[-1]
        
        suppression_difference = best_suppression - worst_suppression
        print(f"Max suppression range: {worst_suppression:.3f} to {best_suppression:.3f} (difference: {suppression_difference:.3f})")
        
        if suppression_difference > NOTABLE_MAX_SUPPRESSION_DIFFERENCE:
            notable_findings.append(f"NOTABLE: Max suppression difference of {suppression_difference:.3f} between {best_scenario} and {worst_scenario}")
        
        # Check for notable probability differences at specific parameter values
        param_check_points = np.percentile(adaptive_ranges[param_name], [25, 50, 75])
        for check_point in param_check_points:
            prob_differences = []
            for scenario_name, result in results.items():
                # Find closest parameter value in the range
                param_idx = np.argmin(np.abs(result['parameter_range'] - check_point))
                suppression_prob = result['suppression_probs'][param_idx]
                prob_differences.append((scenario_name, suppression_prob))
            
            prob_differences.sort(key=lambda x: x[1])
            min_prob = prob_differences[0][1]
            max_prob = prob_differences[-1][1]
            prob_diff = max_prob - min_prob
            
            if prob_diff > NOTABLE_PROBABILITY_DIFFERENCE:
                notable_findings.append(f"NOTABLE: At {param_name}={check_point:.3e}, suppression probability varies from {min_prob:.3f} to {max_prob:.3f} (difference: {prob_diff:.3f})")
        
        # Only generate plot if there are notable findings
        if notable_findings:
            print(f"\n🎯 NOTABLE FINDINGS DETECTED! Generating visualization...")
            
            # Create publication-quality plot
            plt.figure(figsize=(12, 8))
            
            # Plot suppression probabilities
            line_styles = ['-', '--', '-.', ':']
            colors = ['blue', 'red', 'green', 'purple']
            
            # Remove 'model_optimized' scenario and rename 'balanced_importance' to 'Median Value Parameters'
            plot_results = {}
            for scenario_name, result in results.items():
                if scenario_name == 'model_optimized':
                    continue  # Skip model optimized
                label = 'Median Value Parameters' if scenario_name == 'balanced_importance' else scenario_name
                plot_results[label] = result
            
            for i, (label, result) in enumerate(plot_results.items()):
                plt.plot(result['parameter_range'], result['suppression_probs'], 
                        label=label, linewidth=2.5, 
                        linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)])
            
            # Custom title and xlabel for n_eped
            if param_name == 'n_eped':
                plt.xlabel('Electron Density', fontsize=14)
                plt.title('Suppression Probability of Electron Density Values from Random Forest Predictions', fontsize=16, fontweight='bold')
            else:
                plt.xlabel(f'{param_name}', fontsize=14)
                plt.title(f'Notable Differences in {param_name}-Dependent Suppression', fontsize=16, fontweight='bold')
            plt.ylabel('Suppression Probability', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save with high quality for publication
            plt.savefig(f'notable_{param_name}_differences.png', dpi=1200, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Image saved: 'notable_{param_name}_differences.png'")
        else:
            print("❌ No notable differences found in this parameter scanning")
        
        all_notable_findings.extend(notable_findings)
    
    return all_notable_findings

def perform_decision_boundary_analysis(clf, available_features, df_cleaned):
    """
    Perform decision boundary analysis to understand model behavior
    """
    print("\n" + "="*80)
    print("DECISION BOUNDARY ANALYSIS")
    print("="*80)
    
    # Get the two most important features
    feature_importance = clf.feature_importances_
    feature_importance_pairs = list(zip(available_features, feature_importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    top_features = [f[0] for f in feature_importance_pairs[:2]]
    print(f"Top 2 features by importance: {top_features}")
    
    # Create grid for decision boundary
    x_feature, y_feature = top_features[0], top_features[1]
    
    # Get data ranges
    x_min, x_max = df_cleaned[x_feature].min(), df_cleaned[x_feature].max()
    y_min, y_max = df_cleaned[y_feature].min(), df_cleaned[y_feature].max()
    
    # Create meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create test points
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
                    # Use mean value for other features
                    point[feature] = df_cleaned[feature].mean()
            test_points.append([point.get(f, 0) for f in available_features])
    
    # Predict probabilities
    test_array = np.array(test_points)
    probabilities = clf.predict_proba(test_array)[:, 0]  # Suppression probabilities
    probabilities = probabilities.reshape(xx.shape)
    
    # Create decision boundary plot
    plt.figure(figsize=(12, 8))
    
    # Plot decision boundary
    contour = plt.contourf(xx, yy, probabilities, levels=20, cmap='RdYlBu_r')
    plt.colorbar(contour, label='Suppression Probability')
    
    # Plot actual data points
    for state in [0, 1]:  # Suppressed and ELMing
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
    
    plt.savefig('decision_boundary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Image saved: 'decision_boundary_analysis.png'")
    
    return top_features

def main():
    """
    Main execution function for advanced plasma parameter analysis
    """
    print("=== Advanced Plasma Parameter Space Exploration ===")
    print("Using ML-assisted sampling and adaptive scenarios\n")
    
    # Check if model exists, otherwise train it
    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
        print("Training new model...")
        
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Select features and clean data
        df_cleaned, available_features, binary_state_names = select_features_and_clean(df)
        
        # Prepare chronological data splits
        X_train, X_test, y_train, y_test = prepare_chronological_splits(df_cleaned, available_features)
        
        # Train Random Forest
        clf = train_random_forest(X_train, y_train)
        
        # Save model
        save_model(clf, available_features)
    else:
        # Load existing model
        clf, available_features = load_model()
        
        # Load data for analysis
        df = load_and_prepare_data()
        df_cleaned, _, _ = select_features_and_clean(df)
    
    # Run advanced analysis
    print("\n" + "="*80)
    print("RUNNING ADVANCED ANALYSIS")
    print("="*80)
    
    all_notable_findings = []
    
    # Analysis 1: Parameter distribution analysis
    notable_params, separation_scores = analyze_parameter_distributions(df_cleaned, available_features)
    if notable_params:
        all_notable_findings.append(f"NOTABLE: Found {len(notable_params)} parameters with significant separation between states")
    
    # Analysis 2: Cluster analysis
    cluster_analysis, notable_clusters = perform_cluster_analysis(df_cleaned, available_features)
    if notable_clusters:
        all_notable_findings.append(f"NOTABLE: Found {len(notable_clusters)} clusters with distinct state characteristics")
    
    # Analysis 3: Adaptive parameter scanning
    adaptive_findings = test_adaptive_parameter_scanning(clf, available_features, df_cleaned)
    all_notable_findings.extend(adaptive_findings)
    
    # Analysis 4: Decision boundary analysis
    top_features = perform_decision_boundary_analysis(clf, available_features, df_cleaned)
    all_notable_findings.append(f"NOTABLE: Decision boundary analysis reveals {top_features[0]} and {top_features[1]} as key discriminators")
    
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS SUMMARY")
    print("="*80)
    
    if all_notable_findings:
        print("🎯 NOTABLE FINDINGS FOR RESEARCH PAPER:")
        for i, finding in enumerate(all_notable_findings, 1):
            print(f"{i}. {finding}")
        
        print(f"\n📊 Generated multiple visualizations for publication")
    else:
        print("❌ No notable differences found with advanced analysis")
    
    return all_notable_findings

if __name__ == "__main__":
    main() 