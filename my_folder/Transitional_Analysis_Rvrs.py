import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from collections import defaultdict
import pickle

plt.close('all')

# Load the pickled dataset
print("Loading pickled dataset...")
with open('OMFITpickled_8dab1ca6e8', 'rb') as f:
    data = pickle.load(f)

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Add an index column to keep track of original row numbers
df['original_index'] = df.index

# Clean the dataframe - match your existing pipeline
df_cleaned = df.dropna(subset=['state'], how='any')
df_cleaned = df_cleaned[df_cleaned['n'] == 3]  # Only n=3 RMP
df_cleaned = df_cleaned[df_cleaned['state'] != 0]  # Remove N/A states

print(f"Data after cleaning: {df_cleaned.shape}")
print(f"State distribution: {df_cleaned['state'].value_counts().sort_index()}")

# Sort by shot and time for transition detection
df_cleaned = df_cleaned.sort_values(['shot', 'time'])

def find_transition_borders(df, from_state=4, to_state=1, time_window=100):
    """
    Find borders between ELMing (4) and suppressed (1) states

    Args:
        df: DataFrame with 'shot', 'time', 'state' columns
        from_state: Initial state (4 = ELMing)
        to_state: Final state (1 = suppressed)
        time_window: Minimum time between transitions (ms) not really necessary

    Returns:
        List of dictionaries with transition info
    """
    transitions = []

    # Group by shot to analyze each discharge separately
    for shot_id in df['shot'].unique():
        shot_df = df[df['shot'] == shot_id].sort_values('time')

        if len(shot_df) < 2:
            continue

        # Find state changes
        state_changes = shot_df['state'].diff() != 0
        change_indices = shot_df[state_changes].index

        for i, idx in enumerate(change_indices):
            current_state = shot_df.loc[idx, 'state']
            if idx == shot_df.index[0]:  # Skip first point
                continue

            prev_idx = shot_df.index[shot_df.index.get_loc(idx) - 1]
            prev_state = shot_df.loc[prev_idx, 'state']

            # Check if this is an ELMing → suppressed transition
            if prev_state == from_state and current_state == to_state:
                transition_time = shot_df.loc[idx, 'time']

                # Check minimum time separation
                if transitions and transitions[-1]['shot'] == shot_id:
                    if transition_time - transitions[-1]['time'] < time_window:
                        continue

                transitions.append({
                    'shot': shot_id,
                    'time': transition_time,
                    'index': idx,
                    'prev_state': prev_state,
                    'current_state': current_state
                })

    return transitions

def calculate_variable_deltas(df, transitions, variables, time_window=60):
    """
    Calculate percent deltas for variables around transition borders

    Args:
        df: DataFrame with time series data
        transitions: List of transition dictionaries
        variables: List of variable names to analyze
        time_window: Time window in milliseconds (60ms default)

    Returns:
        DataFrame with delta analysis results
    """
    results = []

    for transition in transitions:
        shot_id = transition['shot']
        transition_time = transition['time']

        # Get data for this shot
        shot_df = df[df['shot'] == shot_id].sort_values('time')

        # Define time windows (time is in ms in your database)
        before_start = transition_time - time_window
        before_end = transition_time
        after_start = transition_time
        after_end = transition_time + time_window

        # Get data in time windows
        before_data = shot_df[(shot_df['time'] >= before_start) & (shot_df['time'] < before_end)]
        after_data = shot_df[(shot_df['time'] >= after_start) & (shot_df['time'] < after_end)]

        if len(before_data) == 0 or len(after_data) == 0:
            continue

        # Calculate amplitudes and percent deltas for each variable
        for var in variables:
            if var in shot_df.columns:
                # Calculate amplitude (RMS for better noise handling)
                before_amp = np.sqrt(np.mean(before_data[var]**2)) if len(before_data) > 0 else np.nan
                after_amp = np.sqrt(np.mean(after_data[var]**2)) if len(after_data) > 0 else np.nan

                # Calculate percent delta
                percent_delta = ((after_amp - before_amp) / before_amp * 100) if before_amp != 0 else np.nan
                absolute_delta = after_amp - before_amp

                results.append({
                    'shot': shot_id,
                    'transition_time': transition_time,
                    'variable': var,
                    'before_amplitude': before_amp,
                    'after_amplitude': after_amp,
                    'percent_delta': percent_delta,
                    'absolute_delta': absolute_delta,
                    'n_points_before': len(before_data),
                    'n_points_after': len(after_data)
                })

    return pd.DataFrame(results)

# Variables to analyze - using the actual parameter names from your database
analysis_vars = ['betan', 'bt', 'bt0', 'density', 'dR_sep', 'fs04', 'iln2iamp',
                 'iln2iphase', 'iln3iamp', 'iln3iphase', 'Ip', 'iun2iamp',
                 'iun2iphase', 'iun3iamp', 'iun3iphase', 'kappa', 'li', 'n_e',
                 'n_eped', 'p_eped', 'q95', 'rotation_core', 'rotation_edge',
                 't_eped', 'tribot', 'tritop', 'zeff']

# Filter to only variables that exist in the database
available_vars = [var for var in analysis_vars if var in df_cleaned.columns]
print(f"Available variables for analysis: {available_vars}")

# Find transitions (ELMing → suppressed)
print("Finding ELMing → suppressed transitions...")
transitions = find_transition_borders(df_cleaned, from_state=4, to_state=1)
print(f"Found {len(transitions)} transitions")

# Show transition details
if len(transitions) > 0:
    print("\nTransition summary:")
    for i, trans in enumerate(transitions[:10]):  # Show first 10
        print(f"  Shot {trans['shot']}: t={trans['time']}ms")
    if len(transitions) > 10:
        print(f"  ... and {len(transitions)-10} more")

# Calculate percent deltas
delta_results = calculate_variable_deltas(df_cleaned, transitions, available_vars)

# Analysis and visualization
if len(delta_results) > 0:
    # Summary statistics
    var_summary = delta_results.groupby('variable').agg({
        'percent_delta': ['mean', 'std', 'count'],
        'absolute_delta': ['mean', 'std']
    }).round(4)

    var_summary.columns = ['percent_delta_mean', 'percent_delta_std', 'n_transitions', 'abs_delta_mean', 'abs_delta_std']
    var_summary['abs_percent_delta_mean'] = np.abs(var_summary['percent_delta_mean'])
    var_summary = var_summary.sort_values('abs_percent_delta_mean', ascending=False)

    print("\n=== VARIABLE SUMMARY ===")
    print(var_summary)

else:
    print("No transitions found or no valid delta calculations possible")

# Categorize variables by noncausal significance
print("\n=== NONCAUSAL ANALYSIS ===")
if len(delta_results) > 0:
    # Define threshold for noncausal significance
    noncausal_threshold = 0.9  # +/- 1%

    # Categorize variables
    noncausal_vars = []
    non_noncausal_vars = []

    for var in var_summary.index:
        abs_percent_change = abs(var_summary.loc[var, 'percent_delta_mean'])
        if abs_percent_change > noncausal_threshold:
            noncausal_vars.append({
                'variable': var,
                'percent_change': var_summary.loc[var, 'percent_delta_mean'],
                'abs_percent_change': abs_percent_change
            })
        else:
            non_noncausal_vars.append({
                'variable': var,
                'percent_change': var_summary.loc[var, 'percent_delta_mean'],
                'abs_percent_change': abs_percent_change
            })

    # Sort by absolute percent change
    noncausal_vars.sort(key=lambda x: x['abs_percent_change'], reverse=True)
    non_noncausal_vars.sort(key=lambda x: x['abs_percent_change'], reverse=True)

    print(f"\nNONCAUSAL VARIABLES (>±{noncausal_threshold}% change):")
    if noncausal_vars:
        for var_info in noncausal_vars:
            print(f"  {var_info['variable']}: {var_info['percent_change']:+.1f}%")
    else:
        print("  No variables show noncausal significance")

    print(f"\nNON-NONCAUSAL VARIABLES (≤±{noncausal_threshold}% change):")
    if non_noncausal_vars:
        for var_info in non_noncausal_vars:
            print(f"  {var_info['variable']}: {var_info['percent_change']:+.1f}%")
    else:
        print("  No variables in non-noncausal range")

    print(f"\nSUMMARY:")
    print(f"  Noncausal variables ({len(noncausal_vars)}): {', '.join([var['variable'] for var in noncausal_vars])}")
    print(f"  Non-noncausal variables ({len(non_noncausal_vars)}): {', '.join([var['variable'] for var in non_noncausal_vars])}")
    print(f"  Total variables analyzed: {len(var_summary)}")

print("\n=== ANALYSIS COMPLETE ===")
