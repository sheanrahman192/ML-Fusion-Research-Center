#!/usr/bin/env python3
"""
NoncausalDatabaseCreator.py

This script creates a new database from plasma_data.csv that focuses on transitions
from ELMing (state 4) to Suppressed (state 1). For each transition, it extracts:
1. The last 60 milliseconds (60 points) before the transition, labeled as "To be suppressed"
2. All points before the 60-point window that are in ELMing state (4), labeled as "ELMing"

The script processes all unique shots and identifies all 4->1 transitions.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict

def find_transitions(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """
    Find all transitions from state 4 (ELMing) to state 1 (Suppressed) in the dataset.
    
    Args:
        df: DataFrame containing the plasma data
        
    Returns:
        List of tuples (shot, transition_time) for each 4->1 transition
    """
    transitions = []
    
    for shot in df['shot'].unique():
        shot_data = df[df['shot'] == shot].sort_values('time')
        states = shot_data['state'].values
        
        for i in range(len(states) - 1):
            if states[i] == 4 and states[i+1] == 1:
                transition_time = shot_data.iloc[i+1]['time']
                transitions.append((shot, transition_time))
    
    return transitions

def extract_transition_data(df: pd.DataFrame, shot: int, transition_time: float) -> pd.DataFrame:
    """
    Extract data for a specific transition from ELMing to Suppressed.
    
    Args:
        df: Full DataFrame
        shot: Shot number
        transition_time: Time of the transition
        
    Returns:
        DataFrame with extracted data and labels
    """
    shot_data = df[df['shot'] == shot].sort_values('time')
    
    # Find the transition point
    transition_idx = shot_data[shot_data['time'] == transition_time].index[0]
    transition_row = shot_data.loc[transition_idx]
    
    # Get the index position in the sorted shot data
    shot_data_reset = shot_data.reset_index(drop=True)
    transition_pos = shot_data_reset[shot_data_reset['time'] == transition_time].index[0]
    
    # Extract the last 60 points before transition (labeled as "To be suppressed")
    start_idx = max(0, transition_pos - 60)
    end_idx = transition_pos
    
    # Get the 60 points before transition
    transition_window = shot_data_reset.iloc[start_idx:end_idx].copy()
    transition_window['label'] = 'To be suppressed'
    transition_window['transition_id'] = f"{shot}_{int(transition_time)}"
    
    # Extract ELMing data (state 4) before the transition window
    elming_data = shot_data_reset.iloc[:start_idx].copy()
    elming_data = elming_data[elming_data['state'] == 4].copy()
    elming_data['label'] = 'ELMing'
    elming_data['transition_id'] = f"{shot}_{int(transition_time)}"
    
    # Combine the data
    result_data = pd.concat([elming_data, transition_window], ignore_index=True)
    
    return result_data

def create_noncausal_database(input_file: str, output_file: str) -> None:
    """
    Create the noncausal database from plasma_data.csv.
    
    Args:
        input_file: Path to the input plasma_data.csv file
        output_file: Path to save the output noncausal database
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {len(df['shot'].unique())} unique shots")
    
    # Find all transitions
    print("Finding transitions from ELMing (4) to Suppressed (1)...")
    transitions = find_transitions(df)
    print(f"Found {len(transitions)} transitions")
    
    # Process each transition
    all_extracted_data = []
    
    for i, (shot, transition_time) in enumerate(transitions):
        print(f"Processing transition {i+1}/{len(transitions)}: Shot {shot}, Time {transition_time}")
        
        try:
            extracted_data = extract_transition_data(df, shot, transition_time)
            all_extracted_data.append(extracted_data)
            
            print(f"  - Extracted {len(extracted_data)} data points")
            print(f"  - ELMing points: {len(extracted_data[extracted_data['label'] == 'ELMing'])}")
            print(f"  - To be suppressed points: {len(extracted_data[extracted_data['label'] == 'To be suppressed'])}")
            
        except Exception as e:
            print(f"  - Error processing transition: {e}")
            continue
    
    # Combine all extracted data
    if all_extracted_data:
        final_database = pd.concat(all_extracted_data, ignore_index=True)
        
        # Add some metadata columns
        final_database['data_source'] = 'plasma_data.csv'
        final_database['extraction_timestamp'] = pd.Timestamp.now()
        
        # Save the database
        print(f"\nSaving noncausal database to {output_file}...")
        final_database.to_csv(output_file, index=False)
        
        print(f"Database created successfully!")
        print(f"Total data points: {len(final_database)}")
        print(f"ELMing points: {len(final_database[final_database['label'] == 'ELMing'])}")
        print(f"To be suppressed points: {len(final_database[final_database['label'] == 'To be suppressed'])}")
        print(f"Unique transitions: {final_database['transition_id'].nunique()}")
        print(f"Unique shots: {final_database['shot'].nunique()}")
        
        # Display sample of the data
        print("\nSample of the created database:")
        print(final_database[['shot', 'time', 'state', 'label', 'transition_id']].head(10))
        
    else:
        print("No data was extracted. Please check the input file and data format.")

def main():
    """Main function to run the noncausal database creation."""
    input_file = "plasma_data.csv"
    output_file = "noncausal_database.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print("=" * 60)
    print("Noncausal Database Creator")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    create_noncausal_database(input_file, output_file)
    
    print("=" * 60)
    print("Process completed!")

if __name__ == "__main__":
    main()
