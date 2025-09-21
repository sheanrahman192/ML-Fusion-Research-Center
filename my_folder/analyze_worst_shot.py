import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Binary_BiRNN_50ms import PlasmaBinaryBiRNN, PlasmaRNNWindowDataset, load_and_preprocess_data, select_features, prepare_data
import seaborn as sns

def analyze_worst_shot():
    print("=== Analyzing Worst Performing Shot ===")
    
    # Load data and prepare
    df = load_and_preprocess_data()
    selected_features = select_features(df)
    
    # Filter to states 1, 2, and 4 (as in your current setup)
    df_sorted = df.sort_values(['shot', 'time']).reset_index(drop=True)
    df_filtered = df_sorted[df_sorted['state'].isin([1, 2, 4])].copy()
    
    # Prepare data with same splits
    X_train, X_val, X_test, y_train, y_val, y_test, shots_train, shots_val, shots_test = prepare_data(df, selected_features)
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlasmaBinaryBiRNN(input_dim=X_train.shape[1], n_classes=2).to(device)
    model.load_state_dict(torch.load('best_plasma_binary_birnn_50ms_anti_overfitting.pth'))
    model.eval()
    
    # Create test dataset
    window_size = 100
    test_dataset = PlasmaRNNWindowDataset(X_test, y_test, shots_test, window_size)
    
    # Get predictions for each window
    predictions = []
    true_labels = []
    shot_ids = []
    window_starts = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            features, true_label = test_dataset[i]
            features = features.unsqueeze(0).to(device)  # Add batch dimension
            
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = (probabilities[:, 1] >= 0.45).long()  # Using threshold from training
            
            predictions.append(prediction.item())
            true_labels.append(true_label.item())
            
            # Get shot ID and window start for this sample
            shot_id = shots_test[i * window_size] if i * window_size < len(shots_test) else shots_test[-1]
            shot_ids.append(shot_id)
            window_starts.append(i * window_size)
    
    # Convert to arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    shot_ids = np.array(shot_ids)
    
    # Find misclassifications
    misclassifications = predictions != true_labels
    
    # Count misclassifications per shot
    unique_shots = np.unique(shot_ids)
    shot_error_counts = {}
    
    for shot in unique_shots:
        shot_mask = shot_ids == shot
        error_count = np.sum(misclassifications[shot_mask])
        total_windows = np.sum(shot_mask)
        error_rate = error_count / total_windows if total_windows > 0 else 0
        shot_error_counts[shot] = {
            'error_count': error_count,
            'total_windows': total_windows,
            'error_rate': error_rate
        }
    
    # Find worst shot
    worst_shot = max(shot_error_counts.keys(), key=lambda x: shot_error_counts[x]['error_count'])
    worst_stats = shot_error_counts[worst_shot]
    
    print(f"Worst performing shot: {worst_shot}")
    print(f"Total windows: {worst_stats['total_windows']}")
    print(f"Misclassifications: {worst_stats['error_count']}")
    print(f"Error rate: {worst_stats['error_rate']:.3f}")
    
    # Plot the worst shot
    plot_shot_analysis(df_filtered, worst_shot, window_size, predictions, true_labels, shot_ids)
    
    return worst_shot, worst_stats

def plot_shot_analysis(df, shot_id, window_size, predictions, true_labels, shot_ids):
    """Plot the shot with predictions vs true labels and fs04 overlay"""
    
    # Get shot data
    shot_data = df[df['shot'] == shot_id].copy()
    shot_data = shot_data.sort_values('time')
    
    # Get predictions for this shot
    shot_mask = shot_ids == shot_id
    shot_preds = predictions[shot_mask]
    shot_true = true_labels[shot_mask]
    
    # Create time indices for predictions
    pred_times = []
    for i in range(len(shot_preds)):
        start_idx = i * window_size
        if start_idx < len(shot_data):
            pred_times.append(shot_data.iloc[start_idx]['time'])
    
    # Create single plot with dual y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot fs04 and fs04_max_smoothed on primary y-axis
    ax1.plot(shot_data['time'], shot_data['fs04'], 'b-', linewidth=1, alpha=0.6, label='fs04 (raw)', color='lightblue')
    ax1.plot(shot_data['time'], shot_data['fs04_max_smoothed'], 'b-', linewidth=2, alpha=0.8, label='fs04_max_smoothed', color='blue')
    ax1.set_ylabel('fs04 Signal', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary y-axis for predictions
    ax2 = ax1.twinx()
    
    # Create continuous time series for predictions
    pred_series = []
    true_series = []
    time_series = []
    
    for i, (pred, true, time) in enumerate(zip(shot_preds, shot_true, pred_times)):
        # Extend prediction for the entire window
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(shot_data))
        
        for j in range(start_idx, end_idx):
            if j < len(shot_data):
                pred_series.append(pred)
                true_series.append(true)
                time_series.append(shot_data.iloc[j]['time'])
    
    # Plot only the difference between predicted and true (highlight misclassifications)
    ax2.plot(time_series, pred_series, 'r-', linewidth=3, alpha=0.9, label='Predicted State', color='red')
    
    # Highlight misclassifications by showing where predictions differ from true
    misclass_mask = np.array(pred_series) != np.array(true_series)
    if np.any(misclass_mask):
        misclass_times = np.array(time_series)[misclass_mask]
        misclass_preds = np.array(pred_series)[misclass_mask]
        
        # Plot misclassifications with larger markers
        ax2.scatter(misclass_times, misclass_preds, c='red', s=150, alpha=1.0, marker='x', 
                   linewidth=4, label='Misclassifications', zorder=5, edgecolors='black')
    
    ax2.set_ylabel('ELMing State\n(0=Suppressed, 1=ELMing)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.1, 1.1)
    
    # Add state information as background shading (simplified)
    state_colors = {1: 'lightblue', 2: 'orange', 4: 'red'}
    state_labels = {1: 'Suppressed', 2: 'Intermediate', 4: 'ELMing'}
    
    # Only show ELMing state (4) as background to avoid clutter
    state_mask = shot_data['state'] == 4
    if np.any(state_mask):
        state_times = shot_data.loc[state_mask, 'time']
        if len(state_times) > 1:
            for i in range(len(state_times) - 1):
                ax1.axvspan(state_times.iloc[i], state_times.iloc[i+1], 
                           alpha=0.3, color='red', 
                           label='True ELMing Periods' if i == 0 else "")
    
    # Set up the plot
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_title(f'Shot {shot_id} Analysis - fs04 Signal vs Predictions\n'
                  f'Red line = Predicted ELMing, Red shading = True ELMing periods, X marks = Errors', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Remove duplicate labels
    unique_labels = []
    unique_lines = []
    for line, label in zip(lines1 + lines2, labels1 + labels2):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_lines.append(line)
    
    ax1.legend(unique_lines, unique_labels, loc='upper right', fontsize=10, 
              bbox_to_anchor=(1.15, 1.0))
    
    plt.tight_layout()
    plt.savefig(f'worst_shot_{shot_id}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Analysis for Shot {shot_id}:")
    print(f"Total time points: {len(shot_data)}")
    print(f"Total windows: {len(shot_preds)}")
    print(f"Correct predictions: {np.sum(shot_preds == shot_true)}")
    print(f"Misclassifications: {np.sum(shot_preds != shot_true)}")
    print(f"Accuracy: {np.mean(shot_preds == shot_true):.3f}")
    
    # State distribution
    print(f"\nState distribution in shot:")
    state_counts = shot_data['state'].value_counts().sort_index()
    for state, count in state_counts.items():
        print(f"  State {state}: {count} time points")

if __name__ == "__main__":
    worst_shot, stats = analyze_worst_shot()
