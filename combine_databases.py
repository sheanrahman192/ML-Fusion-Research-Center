#!/usr/bin/env python3
"""
Script to combine LABEL_PROPAGATED_DATABASE.csv and plasma_data.csv.
When both databases have the same shot number, data from plasma_data.csv takes priority.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def combine_databases(
    label_propagated_path: str = "LABEL_PROPAGATED_DATABASE.csv",
    plasma_data_path: str = "plasma_data.csv",
    output_path: str = "combined_database.csv",
    key_columns: list = None
):
    """
    Combine two databases, prioritizing plasma_data.csv for overlapping shots.
    
    Args:
        label_propagated_path: Path to LABEL_PROPAGATED_DATABASE.csv
        plasma_data_path: Path to plasma_data.csv
        output_path: Path for the output combined CSV
        key_columns: Columns to use as merge key (default: ['shot', 'time'])
    """
    if key_columns is None:
        key_columns = ['shot', 'time']
    
    print(f"Loading {label_propagated_path}...")
    df_label = pd.read_csv(label_propagated_path)
    print(f"  Loaded {len(df_label):,} rows, {len(df_label.columns)} columns")
    print(f"  Unique shots: {df_label['shot'].nunique():,}")
    
    print(f"\nLoading {plasma_data_path}...")
    df_plasma = pd.read_csv(plasma_data_path)
    print(f"  Loaded {len(df_plasma):,} rows, {len(df_plasma.columns)} columns")
    print(f"  Unique shots: {df_plasma['shot'].nunique():,}")
    
    # Find overlapping and unique shots
    label_shots = set(zip(df_label['shot'], df_label['time']))
    plasma_shots = set(zip(df_plasma['shot'], df_plasma['time']))
    
    overlapping = label_shots & plasma_shots
    only_in_label = label_shots - plasma_shots
    only_in_plasma = plasma_shots - label_shots
    
    print(f"\n--- Overlap Analysis (by shot+time) ---")
    print(f"  Rows in both databases: {len(overlapping):,}")
    print(f"  Rows only in LABEL_PROPAGATED: {len(only_in_label):,}")
    print(f"  Rows only in plasma_data: {len(only_in_plasma):,}")
    
    # Get all unique columns from both dataframes
    all_columns = list(df_label.columns)
    for col in df_plasma.columns:
        if col not in all_columns:
            all_columns.append(col)
    
    print(f"\n--- Column Analysis ---")
    label_only_cols = set(df_label.columns) - set(df_plasma.columns)
    plasma_only_cols = set(df_plasma.columns) - set(df_label.columns)
    common_cols = set(df_label.columns) & set(df_plasma.columns)
    
    print(f"  Columns in both: {len(common_cols)}")
    print(f"  Columns only in LABEL_PROPAGATED: {label_only_cols if label_only_cols else 'None'}")
    print(f"  Columns only in plasma_data: {plasma_only_cols if plasma_only_cols else 'None'}")
    
    # Create the combined dataframe
    # Strategy: Use plasma_data.csv for overlapping shots, 
    #           add remaining rows from LABEL_PROPAGATED_DATABASE.csv
    
    # Mark rows in label_propagated that DON'T exist in plasma_data
    df_label['_merge_key'] = list(zip(df_label['shot'], df_label['time']))
    df_label_unique = df_label[~df_label['_merge_key'].isin(plasma_shots)].drop(columns=['_merge_key'])
    
    print(f"\n--- Combining Databases ---")
    print(f"  Using all {len(df_plasma):,} rows from plasma_data.csv")
    print(f"  Adding {len(df_label_unique):,} unique rows from LABEL_PROPAGATED_DATABASE.csv")
    
    # Combine: all of plasma_data + unique rows from label_propagated
    df_combined = pd.concat([df_plasma, df_label_unique], ignore_index=True)
    
    # Merge state_binary (predicted by propagate_labels_binary.py) into state_binary column where missing
    if 'state_binary' in df_label.columns:
        n_missing_before = df_combined['state_binary'].isna().sum() if 'state_binary' in df_combined.columns else len(df_combined)
        if 'state_binary' not in df_combined.columns:
            df_combined['state_binary'] = np.nan
        # Build a (shot,time)-keyed map of propagated labels and fill where state_binary is NaN
        prop = df_label.set_index(['shot', 'time'])['state_binary']
        idx = pd.MultiIndex.from_arrays([df_combined['shot'].values, df_combined['time'].values])
        propagated = prop.reindex(idx).values
        # Treat -1 (uncertain) from propagation as still-unknown
        propagated = pd.Series(propagated)
        propagated = propagated.where(propagated != -1, np.nan)
        df_combined['state_binary'] = df_combined['state_binary'].fillna(pd.Series(propagated.values, index=df_combined.index))
        n_missing_after = df_combined['state_binary'].isna().sum()
        print(f"  Filled {n_missing_before - n_missing_after:,} missing 'state_binary' values from propagated labels")
    else:
        print("  WARNING: LABEL_PROPAGATED_DATABASE.csv has no 'state_binary' column — run propagate_labels_binary.py first.")
    
    # Sort by shot and time for consistency
    df_combined = df_combined.sort_values(['shot', 'time']).reset_index(drop=True)
    
    print(f"\n--- Combined Database ---")
    print(f"  Total rows: {len(df_combined):,}")
    print(f"  Total columns: {len(df_combined.columns)}")
    print(f"  Unique shots: {df_combined['shot'].nunique():,}")
    
    # Save to output file
    print(f"\nSaving to {output_path}...")
    df_combined.to_csv(output_path, index=False)
    print(f"Done! Combined database saved to {output_path}")
    
    return df_combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine LABEL_PROPAGATED_DATABASE.csv and plasma_data.csv"
    )
    parser.add_argument(
        "--label-propagated", "-l",
        default="LABEL_PROPAGATED_DATABASE.csv",
        help="Path to LABEL_PROPAGATED_DATABASE.csv (default: LABEL_PROPAGATED_DATABASE.csv)"
    )
    parser.add_argument(
        "--plasma-data", "-p",
        default="plasma_data.csv",
        help="Path to plasma_data.csv (default: plasma_data.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="combined_database.csv",
        help="Output path for combined CSV (default: combined_database.csv)"
    )
    
    args = parser.parse_args()
    
    combine_databases(
        label_propagated_path=args.label_propagated,
        plasma_data_path=args.plasma_data,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

