import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

CSV_FILE = 'plasma_data.csv'
SELECTED_FEATURES = [
    'iln3iamp', 'tribot',
    'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop',
    'fs04_max_smoothed', 'fs04_max_avg', 'fs_up_sum', 'fs_sum'
]


def load_and_prepare_data():
    """
    Load CSV, apply the same cleaning and binary state mapping as Visual_Data_Analysis.py
    """
    df = pd.read_csv(CSV_FILE)

    # Remove problematic shot as in the original script
    df = df[df['shot'] != 191675].copy()

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Coerce state to numeric and drop rows with missing state
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])

    # Map states to binary: {0: Suppressed group (1,2,3), 1: ELMing (4)}
    def map_states_to_binary(state):
        if state in [1]:
            return 0  # Suppressed
        elif state == 4:
            return 1  # ELMing
        else:
            return np.nan

    df['binary_state'] = df['state'].apply(map_states_to_binary)
    df = df.dropna(subset=['binary_state'])
    df['binary_state'] = df['binary_state'].astype(int)

    return df


def get_available_features(df):
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    return available


def plot_all_feature_distributions(df, available_features, bins=100):
    """
    Create one figure with subplots for all available features.
    Each subplot shows normalized (density) histograms for Suppressed (blue) and ELMing (red).
    Saves a single PNG: all_feature_distributions.png
    """
    num_features = len(available_features)
    if num_features == 0:
        return

    # Layout: up to 4 columns
    ncols = 4 if num_features >= 4 else num_features
    nrows = int(np.ceil(num_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols + 2, 3.2 * nrows + 1), squeeze=False)
    fig.suptitle('Normalized Distributions by State (Suppressed vs ELMing)', fontsize=16, fontweight='bold')

    green = 'green'
    red = 'red'

    legend_handles = None
    legend_labels = None

    for idx, feature in enumerate(available_features):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        data = df[[feature, 'binary_state']].dropna()
        supp = data[data['binary_state'] == 0][feature].values
        elm = data[data['binary_state'] == 1][feature].values

        if len(supp) == 0 or len(elm) == 0:
            ax.set_visible(False)
            continue

        finite_vals = np.concatenate([supp[np.isfinite(supp)], elm[np.isfinite(elm)]])
        if finite_vals.size == 0:
            ax.set_visible(False)
            continue
        # Normalize x-axis using combined mean/std (z-score)
        mean_val = np.nanmean(finite_vals)
        std_val = np.nanstd(finite_vals)
        if not np.isfinite(std_val) or std_val == 0:
            ax.set_visible(False)
            continue
        supp_norm = (supp - mean_val) / std_val
        elm_norm = (elm - mean_val) / std_val

        # Determine common bin edges in normalized space
        finite_norm = np.concatenate([supp_norm[np.isfinite(supp_norm)], elm_norm[np.isfinite(elm_norm)]])
        if finite_norm.size == 0:
            ax.set_visible(False)
            continue
        min_n, max_n = np.nanmin(finite_norm), np.nanmax(finite_norm)
        if not np.isfinite(min_n) or not np.isfinite(max_n) or min_n == max_n:
            ax.set_visible(False)
            continue
        bin_edges = np.linspace(min_n, max_n, bins + 1)

        h1 = ax.hist(supp_norm, bins=bin_edges, alpha=0.5, label='Suppressed', color=green, density=True)
        h2 = ax.hist(elm_norm, bins=bin_edges, alpha=0.5, label='ELMing', color=red, density=True)

        ax.set_title(feature, fontsize=11)
        ax.set_xlabel('Standardized Value (z-score)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Capture legend handles from the first valid subplot
        if legend_handles is None or legend_labels is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                legend_handles, legend_labels = handles, labels

    # Hide any unused subplots
    total_axes = nrows * ncols
    for j in range(num_features, total_axes):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_visible(False)

    # Single legend (fallback if none captured)
    if not legend_handles or not legend_labels:
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=green, alpha=0.5), Patch(facecolor=red, alpha=0.5)]
        legend_labels = ['Suppressed', 'ELMing']
    fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])

    plt.savefig('all_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=== State Distribution Histogram ===")
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print("Selecting available features and generating distributions...")
    available_features = get_available_features(df)
    if not available_features:
        print("No selected features found in the dataset.")
        return
    print(f"Features: {', '.join(available_features)}")
    plot_all_feature_distributions(df, available_features)
    print("Saved one figure: 'all_feature_distributions.png'")


if __name__ == "__main__":
    main()


