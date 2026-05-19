#!/usr/bin/env python3
"""
Reproduce Figure 2(a) from Hu et al., "RMP ELM control unveils high ion 
temperature with ITB in the DIII-D tokamak."

Shows core ion temperature vs. electron density with NBI power as color.
Each point is a 200 ms time-averaged value from RMP experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Load data
print("Loading data...")
df = pd.read_csv('/mnt/homes/sr4240/my_folder/LABEL_PROPAGATED_DATABASE.csv')

# Select relevant columns
cols = ['shot', 'time', 'cerqtit1', 'density', 'pinj']
df = df[cols].copy()

# Drop rows with missing values in key columns
print(f"Original rows: {len(df)}")
df = df.dropna(subset=['cerqtit1', 'density', 'pinj'])
print(f"Rows after dropping NaN: {len(df)}")

# Apply unit conversions
# - density: multiply by 1e6 to get m^-3, then divide by 1e19 to get 10^19 m^-3
# - cerqtit1: convert from eV to keV (divide by 1000)
# - pinj: convert from kW to MW (divide by 1000)
df['density_1e19'] = df['density'] * 1e6 / 1e19
df['Ti_keV'] = df['cerqtit1'] / 1000
df['P_NBI_MW'] = df['pinj'] / 1000

# Print data ranges for verification
print("\nData ranges after conversion:")
print(f"  density (10^19 m^-3): {df['density_1e19'].min():.2f} - {df['density_1e19'].max():.2f}")
print(f"  Ti (keV): {df['Ti_keV'].min():.2f} - {df['Ti_keV'].max():.2f}")
print(f"  P_NBI (MW): {df['P_NBI_MW'].min():.2f} - {df['P_NBI_MW'].max():.2f}")

BIN_MS = 200

# Create 200 ms time bins (non-overlapping)
# Time is in milliseconds, so BIN_MS ms bins
df['time_bin'] = (df['time'] // BIN_MS) * BIN_MS

# Group by shot and time bin, then average
print(f"\nApplying {BIN_MS} ms time averaging...")
df_avg = df.groupby(['shot', 'time_bin']).agg({
    'density_1e19': 'mean',
    'Ti_keV': 'mean',
    'P_NBI_MW': 'mean'
}).reset_index()

print(f"Data points after {BIN_MS} ms averaging: {len(df_avg)}")

# Filter data to match the figure's axis ranges
# x-axis: 0-10 (ne in 10^19 m^-3)
# y-axis: 0-20 (Ti in keV)
# color: 4-10 (P_NBI in MW)
df_plot = df_avg[
    (df_avg['density_1e19'] >= 0) & (df_avg['density_1e19'] <= 10) &
    (df_avg['Ti_keV'] >= 0) & (df_avg['Ti_keV'] <= 20) &
    (df_avg['P_NBI_MW'] >= 4) & (df_avg['P_NBI_MW'] <= 10)
].copy()

print(f"Data points after filtering to figure range: {len(df_plot)}")

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Set up colormap - use a similar colormap to the original (viridis-like)
cmap = plt.cm.viridis
norm = Normalize(vmin=4, vmax=10)

# Create scatter plot
scatter = ax.scatter(
    df_plot['density_1e19'],
    df_plot['Ti_keV'],
    c=df_plot['P_NBI_MW'],
    cmap=cmap,
    norm=norm,
    s=15,
    alpha=0.8,
    edgecolors='none'
)

# Add colorbar
cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label(r'$P_{\mathrm{NBI}}$ (MW)', fontsize=12)
cbar.set_ticks([4, 5, 6, 7, 8, 9, 10])

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)

# Set axis labels with proper formatting
ax.set_xlabel(r'$n_{e}$ ($10^{19}$m$^{-3}$)', fontsize=12)
ax.set_ylabel(r'$T_{i,0}$ (keV)', fontsize=12)

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=10, direction='in', 
               top=True, right=True)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks([0, 5, 10, 15, 20])

# Tight layout and save
plt.tight_layout()
plt.savefig('/mnt/homes/sr4240/my_folder/figure_2a_reproduction.png', dpi=300, bbox_inches='tight')
print("\nFigure saved as 'figure_2a_reproduction.png'")

plt.show()
