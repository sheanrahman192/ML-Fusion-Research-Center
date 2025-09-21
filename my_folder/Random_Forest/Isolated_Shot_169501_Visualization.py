"""
Isolated Shot Visualization Script for Random Forest Analysis

This script visualizes Random Forest predictions for shot 169501 while ensuring that
shot 169501 is completely excluded from the training data. This provides a more
realistic assessment of the model's performance on truly unseen data.

HOW TO USE:
1. Change the SHOT_NUMBER variable below to the shot you want to analyze
2. Run the script: python Isolated_Shot_169501_Visualization.py
3. The script will generate two visualization files with the shot number in the filename

The script will automatically check if the shot exists and provide helpful error messages
if the shot is not found in the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import os

plt.close('all')

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# =============================================================================

# Shot number to analyze (change this to any shot number you want to visualize)
# Examples: 169501, 169472, 169503, etc.
SHOT_NUMBER = 169501

# Output file prefix (will be automatically updated based on shot number)
OUTPUT_PREFIX = f"isolated_shot_{SHOT_NUMBER}"

# =============================================================================

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_train_model_excluding_shot(exclude_shot_number=169501):
    """Load data and train the Random Forest model while excluding the specified shot from training"""
    print("Loading plasma data...")
    
    # Try multiple possible paths for the plasma data file
    possible_paths = [
        '../plasma_data.csv',  # Relative path from Random_Forest directory
        'plasma_data.csv',     # Current directory
        '/mnt/homes/sr4240/my_folder/plasma_data.csv'  # Absolute path
    ]
    
    df = None
    for path in possible_paths:
        try:
            print(f"Trying to load from: {path}")
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            print(f"Loaded {len(df)} rows")
            break
        except FileNotFoundError:
            print(f"File not found at: {path}")
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find plasma_data.csv in any of the expected locations")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Add an index column to keep track of original row numbers
    df['original_index'] = df.index
    
    # Select features and the target variable (same as original script)
    selected_features = ['iln3iamp', 'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop',
                        'fs04_max_smoothed', 'fs_sum', 'fs_up_sum']
    target_column = 'state'
    
    # Clean the dataframe (same as original script)
    df_cleaned = df.dropna(subset=selected_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['n'] == 3]
    
    # CRITICAL: Exclude the target shot from training data
    print(f"Excluding shot {exclude_shot_number} from training data...")
    df_training = df_cleaned[df_cleaned['shot'] != exclude_shot_number].copy()
    df_target_shot = df_cleaned[df_cleaned['shot'] == exclude_shot_number].copy()
    
    print(f"Training data: {len(df_training)} rows (excluding shot {exclude_shot_number})")
    print(f"Target shot data: {len(df_target_shot)} rows (shot {exclude_shot_number})")
    
    # Prepare input (X) and target (y) for training (excluding target shot)
    X_train = df_training[selected_features]
    y_train = df_training[target_column]
    
    # Split training data into train/test for model validation
    X_train_split, X_test_split, y_train_split, y_test_split, idx_train_split, idx_test_split = train_test_split(
        X_train, y_train, df_training['original_index'], test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier (reduced estimators for faster training)
    print("Training Random Forest model on data excluding target shot...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=35, min_samples_split=2, 
                                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
    clf.fit(X_train_split, y_train_split)
    
    # Calculate accuracy on the held-out training test set
    y_pred_split = clf.predict(X_test_split)
    accuracy = accuracy_score(y_test_split, y_pred_split)
    print(f"Model accuracy on training test set: {accuracy:.3f}")
    
    return clf, selected_features, df_cleaned, df_target_shot

def check_shot_exists(df, shot_number):
    """Check if the specified shot exists in the dataset"""
    shot_exists = (df['shot'] == shot_number).any()
    if not shot_exists:
        available_shots = sorted(df['shot'].unique())
        print(f"ERROR: Shot {shot_number} not found in the dataset!")
        print(f"Available shots: {available_shots[:10]}..." if len(available_shots) > 10 else f"Available shots: {available_shots}")
        return False
    return True

def extract_shot_data(df, shot_number=169501):
    """Extract all time series data for the specified shot"""
    print(f"Extracting data for shot {shot_number}...")
    
    # Check if shot exists first
    if not check_shot_exists(df, shot_number):
        raise ValueError(f"Shot {shot_number} not found in dataset")
    
    # Filter data for the specific shot
    shot_data = df[df['shot'] == shot_number].copy()
    
    # Sort by time to ensure proper time series order
    shot_data = shot_data.sort_values('time').reset_index(drop=True)
    
    print(f"Found {len(shot_data)} time points for shot {shot_number}")
    print(f"Time range: {shot_data['time'].min()} to {shot_data['time'].max()}")
    print(f"State distribution: {shot_data['state'].value_counts().to_dict()}")
    
    return shot_data

def make_predictions_on_shot(clf, shot_data, selected_features):
    """Make predictions on the shot data"""
    print("Making predictions on isolated shot data...")
    
    # Prepare features for prediction
    X_shot = shot_data[selected_features]
    
    # Make predictions
    predictions = clf.predict(X_shot)
    prediction_proba = clf.predict_proba(X_shot)
    
    # Add predictions to shot data
    shot_data = shot_data.copy()
    shot_data['predicted_state'] = predictions
    shot_data['prediction_confidence'] = np.max(prediction_proba, axis=1)
    
    # Determine if predictions are correct
    shot_data['prediction_correct'] = (shot_data['state'] == shot_data['predicted_state'])
    
    # Calculate accuracy for this shot
    shot_accuracy = shot_data['prediction_correct'].mean()
    print(f"Prediction accuracy for isolated shot {shot_data['shot'].iloc[0]}: {shot_accuracy:.3f}")
    
    return shot_data

def create_comprehensive_visualization(shot_data):
    """Create a comprehensive visualization of the shot data with predictions"""
    print("Creating visualization...")
    
    # Create figure with subplots - make the figure larger and adjust layout
    fig, axes = plt.subplots(3, 1, figsize=(24, 20))
    fig.suptitle(f'Isolated Shot {shot_data["shot"].iloc[0]} Analysis: Random Forest Predictions vs Actual States\n(Shot {shot_data["shot"].iloc[0]} was EXCLUDED from training)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors for states
    state_colors = {1: '#2E8B57', 2: '#FFD700', 3: '#FF6347', 4: '#DC143C'}  # Green, Gold, Tomato, Crimson
    state_names = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}
    
    # Plot 1: fs04 time series with actual states
    ax1 = axes[0]
    time = shot_data['time']
    fs04 = shot_data['fs04']
    states = shot_data['state']
    
    # Plot fs04 time series
    ax1.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    
    # Color background by actual state
    for state in [1, 2, 3, 4]:
        state_mask = states == state
        if state_mask.any():
            ax1.fill_between(time, fs04.min(), fs04.max(), 
                           where=state_mask, alpha=0.3, 
                           color=state_colors[state], 
                           label=f'Actual: {state_names[state]}')
    
    ax1.set_ylabel('fs04', fontsize=16)
    ax1.set_title('fs04 Time Series with Actual States', fontsize=18, fontweight='bold')
    # Position legend outside the plot area
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: fs04 time series with predicted states
    ax2 = axes[1]
    predicted_states = shot_data['predicted_state']
    
    # Plot fs04 time series
    ax2.plot(time, fs04, 'k-', linewidth=1, alpha=0.7, label='fs04')
    
    # Color background by predicted state
    for state in [1, 2, 3, 4]:
        state_mask = predicted_states == state
        if state_mask.any():
            ax2.fill_between(time, fs04.min(), fs04.max(), 
                           where=state_mask, alpha=0.3, 
                           color=state_colors[state], 
                           label=f'Predicted: {state_names[state]}')
    
    ax2.set_ylabel('fs04', fontsize=16)
    ax2.set_title('fs04 Time Series with Predicted States', fontsize=18, fontweight='bold')
    # Position legend outside the plot area
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: fs04 time series with correct/incorrect predictions overlay (MAKE THIS LARGER)
    ax3 = axes[2]
    
    # Plot fs04 time series with thicker line
    ax3.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')
    
    # Overlay correct predictions in green
    correct_mask = shot_data['prediction_correct']
    ax3.fill_between(time, fs04.min(), fs04.max(), 
                    where=correct_mask, alpha=0.4, 
                    color='green', label='Correct Predictions')
    
    # Overlay incorrect predictions in red
    incorrect_mask = ~correct_mask
    ax3.fill_between(time, fs04.min(), fs04.max(), 
                    where=incorrect_mask, alpha=0.4, 
                    color='red', label='Incorrect Predictions')
    
    # Add markers for prediction confidence with larger size
    confidence = shot_data['prediction_confidence']
    scatter = ax3.scatter(time, fs04, c=confidence, cmap='viridis', 
                         s=40, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Add colorbar for confidence - position it to avoid overlap
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8, aspect=25, pad=0.02)
    cbar.set_label('Prediction Confidence', fontsize=14, rotation=270, labelpad=25)
    
    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('fs04', fontsize=16)
    ax3.set_title('fs04 Time Series with Prediction Accuracy Overlay (ENLARGED) - ISOLATED SHOT', fontsize=18, fontweight='bold')
    # Position legend outside the plot area to avoid overlap with colorbar
    ax3.legend(bbox_to_anchor=(1.12, 1), loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap - give more space for legends and ensure bottom is visible
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, hspace=0.3, bottom=0.08, top=0.95)  # More space for legends and ensure bottom is visible
    
    return fig

def create_detailed_analysis_plot(shot_data):
    """Create a detailed analysis plot showing state transitions and prediction errors"""
    print("Creating detailed analysis plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    fig.suptitle(f'Isolated Shot {shot_data["shot"].iloc[0]} Detailed Analysis\n(Shot {shot_data["shot"].iloc[0]} was EXCLUDED from training)', 
                 fontsize=20, fontweight='bold')
    
    time = shot_data['time']
    fs04 = shot_data['fs04']
    actual_states = shot_data['state']
    predicted_states = shot_data['predicted_state']
    correct_predictions = shot_data['prediction_correct']
    
    # Plot 1: State transitions over time
    ax1 = axes[0]
    
    # Plot actual states
    ax1.step(time, actual_states, 'b-', linewidth=2, label='Actual State', where='post')
    
    # Plot predicted states
    ax1.step(time, predicted_states, 'r--', linewidth=2, label='Predicted State', where='post')
    
    # Highlight incorrect predictions
    incorrect_times = time[~correct_predictions]
    incorrect_actual = actual_states[~correct_predictions]
    ax1.scatter(incorrect_times, incorrect_actual, color='red', s=60, 
               marker='x', label='Incorrect Predictions', zorder=5)
    
    ax1.set_ylabel('State', fontsize=16)
    ax1.set_title('State Transitions: Actual vs Predicted (ISOLATED SHOT)', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 4.5)
    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels(['Suppressed', 'Dithering', 'Mitigated', 'ELMing'])
    
    # Plot 2: fs04 with prediction accuracy markers
    ax2 = axes[1]
    
    # Plot fs04
    ax2.plot(time, fs04, 'k-', linewidth=2, alpha=0.8, label='fs04')
    
    # Mark correct predictions
    correct_times = time[correct_predictions]
    correct_fs04 = fs04[correct_predictions]
    ax2.scatter(correct_times, correct_fs04, color='green', s=40, 
               alpha=0.8, label='Correct Predictions', marker='o')
    
    # Mark incorrect predictions
    incorrect_fs04 = fs04[~correct_predictions]
    ax2.scatter(incorrect_times, incorrect_fs04, color='red', s=60, 
               alpha=0.8, label='Incorrect Predictions', marker='x')
    
    ax2.set_xlabel('Time (ms)', fontsize=16)
    ax2.set_ylabel('fs04', fontsize=16)
    ax2.set_title('fs04 with Prediction Accuracy Markers (ISOLATED SHOT)', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=14, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Ensure proper layout and spacing to prevent cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.93, hspace=0.3)  # Make room for title and subplot spacing
    
    return fig

def print_detailed_statistics(shot_data):
    """Print detailed statistics about the predictions"""
    print("\n" + "="*70)
    print(f"ISOLATED SHOT ANALYSIS FOR SHOT {shot_data['shot'].iloc[0]}")
    print(f"(Shot {shot_data['shot'].iloc[0]} was EXCLUDED from training data)")
    print("="*70)
    
    # Overall statistics
    total_points = len(shot_data)
    correct_predictions = shot_data['prediction_correct'].sum()
    accuracy = correct_predictions / total_points
    
    print(f"Total time points: {total_points}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_points - correct_predictions}")
    print(f"Overall accuracy: {accuracy:.3f}")
    
    # Statistics by actual state
    print(f"\nAccuracy by Actual State:")
    for state in [1, 2, 3, 4]:
        state_mask = shot_data['state'] == state
        if state_mask.any():
            state_correct = shot_data.loc[state_mask, 'prediction_correct'].sum()
            state_total = state_mask.sum()
            state_accuracy = state_correct / state_total
            state_name = {1: 'Suppressed', 2: 'Dithering', 3: 'Mitigated', 4: 'ELMing'}[state]
            print(f"  {state_name} (State {state}): {state_correct}/{state_total} ({state_accuracy:.3f})")
    
    # Prediction confidence statistics
    print(f"\nPrediction Confidence Statistics:")
    print(f"  Mean confidence: {shot_data['prediction_confidence'].mean():.3f}")
    print(f"  Std confidence: {shot_data['prediction_confidence'].std():.3f}")
    print(f"  Min confidence: {shot_data['prediction_confidence'].min():.3f}")
    print(f"  Max confidence: {shot_data['prediction_confidence'].max():.3f}")
    
    # Confidence by prediction accuracy
    correct_confidence = shot_data[shot_data['prediction_correct']]['prediction_confidence']
    incorrect_confidence = shot_data[~shot_data['prediction_correct']]['prediction_confidence']
    
    print(f"\nConfidence by Prediction Accuracy:")
    print(f"  Correct predictions - Mean: {correct_confidence.mean():.3f}, Std: {correct_confidence.std():.3f}")
    print(f"  Incorrect predictions - Mean: {incorrect_confidence.mean():.3f}, Std: {incorrect_confidence.std():.3f}")

def list_available_shots(df, limit=20):
    """List available shots in the dataset"""
    available_shots = sorted(df['shot'].unique())
    print(f"\nAvailable shots in dataset (showing first {min(limit, len(available_shots))}):")
    for i, shot in enumerate(available_shots[:limit]):
        shot_count = len(df[df['shot'] == shot])
        print(f"  {shot} ({shot_count} time points)")
    if len(available_shots) > limit:
        print(f"  ... and {len(available_shots) - limit} more shots")
    print(f"Total unique shots: {len(available_shots)}")
    return available_shots

def main():
    """Main function to run the complete analysis"""
    print(f"Starting Isolated Shot {SHOT_NUMBER} Visualization Analysis...")
    print("="*70)
    print(f"IMPORTANT: Shot {SHOT_NUMBER} will be EXCLUDED from training data")
    print("="*70)
    
    # Load data and train model (excluding target shot)
    clf, selected_features, df_cleaned, df_target_shot = load_and_train_model_excluding_shot(SHOT_NUMBER)
    
    # Check if shot exists and provide helpful information
    if not check_shot_exists(df_cleaned, SHOT_NUMBER):
        print("\nTo change the shot number, edit the SHOT_NUMBER variable at the top of this script.")
        list_available_shots(df_cleaned)
        return None
    
    # Extract shot data
    shot_data = extract_shot_data(df_cleaned, shot_number=SHOT_NUMBER)
    
    # Make predictions
    shot_data = make_predictions_on_shot(clf, shot_data, selected_features)
    
    # Create visualizations
    fig1 = create_comprehensive_visualization(shot_data)
    fig2 = create_detailed_analysis_plot(shot_data)
    
    # Print detailed statistics
    print_detailed_statistics(shot_data)
    
    # Create prediction_visualizations directory if it doesn't exist
    output_dir = 'prediction_visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save plots in the prediction_visualizations folder
    comprehensive_filename = os.path.join(output_dir, f'{OUTPUT_PREFIX}_comprehensive_analysis.png')
    detailed_filename = os.path.join(output_dir, f'{OUTPUT_PREFIX}_detailed_analysis.png')
    
    fig1.savefig(comprehensive_filename, dpi=300, bbox_inches='tight')
    fig2.savefig(detailed_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved as:")
    print(f"  - {comprehensive_filename}")
    print(f"  - {detailed_filename}")
    
    return shot_data

if __name__ == "__main__":
    shot_data = main()
