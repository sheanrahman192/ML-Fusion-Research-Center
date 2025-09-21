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

CSV_FILE = 'noncausal_database.csv'
SELECTED_FEATURES = [
    'iln3iamp', 'tribot',
    'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop',
    'fs04_max_smoothed', 'fs04_max_avg', 'fs_up_sum', 'fs_sum'
]


def load_and_prepare_data():
    """
    Load CSV, apply cleaning and binary transition_id mapping
    """
    df = pd.read_csv(CSV_FILE)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Clean the label column and create binary mapping
    df['label'] = df['label'].str.strip()  # Remove any whitespace
    
    # Map transition_id categories to binary: {0: To be suppressed, 1: ELMing}
    def map_transition_to_binary(label):
        if label == 'To be suppressed':
            return 0  # To be suppressed
        elif label == 'ELMing':
            return 1  # ELMing
        else:
            return np.nan

    df['binary_transition'] = df['label'].apply(map_transition_to_binary)
    df = df.dropna(subset=['binary_transition'])
    df['binary_transition'] = df['binary_transition'].astype(int)

    return df


def get_available_features(df):
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    return available


def plot_all_feature_distributions(df, available_features, bins=100):
    """
    Create one figure with subplots for all available features.
    Each subplot shows normalized (density) histograms for To be suppressed (blue) and ELMing (red).
    Saves a single PNG: transition_id_feature_distributions.png
    """
    num_features = len(available_features)
    if num_features == 0:
        return

    # Layout: 3 columns for better readability
    ncols = 3
    nrows = int(np.ceil(num_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols + 2, 4 * nrows + 1), squeeze=False)
    fig.suptitle('Normalized Distributions by Transition ID (To be suppressed vs ELMing)', fontsize=18, fontweight='bold')

    blue = 'blue'
    red = 'red'

    legend_handles = None
    legend_labels = None

    for idx, feature in enumerate(available_features):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        data = df[[feature, 'binary_transition']].dropna()
        suppressed = data[data['binary_transition'] == 0][feature].values
        elming = data[data['binary_transition'] == 1][feature].values

        if len(suppressed) == 0 or len(elming) == 0:
            ax.set_visible(False)
            continue

        finite_vals = np.concatenate([suppressed[np.isfinite(suppressed)], elming[np.isfinite(elming)]])
        if finite_vals.size == 0:
            ax.set_visible(False)
            continue
            
        # Normalize x-axis using combined mean/std (z-score)
        mean_val = np.nanmean(finite_vals)
        std_val = np.nanstd(finite_vals)
        if not np.isfinite(std_val) or std_val == 0:
            ax.set_visible(False)
            continue
        suppressed_norm = (suppressed - mean_val) / std_val
        elming_norm = (elming - mean_val) / std_val

        # Determine common bin edges in normalized space
        finite_norm = np.concatenate([suppressed_norm[np.isfinite(suppressed_norm)], elming_norm[np.isfinite(elming_norm)]])
        if finite_norm.size == 0:
            ax.set_visible(False)
            continue
        min_n, max_n = np.nanmin(finite_norm), np.nanmax(finite_norm)
        if not np.isfinite(min_n) or not np.isfinite(max_n) or min_n == max_n:
            ax.set_visible(False)
            continue
        bin_edges = np.linspace(min_n, max_n, bins + 1)

        h1 = ax.hist(suppressed_norm, bins=bin_edges, alpha=0.5, label='To be suppressed', color=blue, density=True)
        h2 = ax.hist(elming_norm, bins=bin_edges, alpha=0.5, label='ELMing', color=red, density=True)

        ax.set_title(feature, fontsize=13, fontweight='bold')
        ax.set_xlabel('Standardized Value (z-score)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)

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
        legend_handles = [Patch(facecolor=blue, alpha=0.5), Patch(facecolor=red, alpha=0.5)]
        legend_labels = ['To be suppressed', 'ELMing']
    fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])

    plt.savefig('all_transition_histograms_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_individual_feature_distributions(df, available_features, bins=100):
    """
    Create individual histogram plots for each feature.
    Saves separate PNG files for each feature.
    """
    for feature in available_features:
        data = df[[feature, 'binary_transition']].dropna()
        suppressed = data[data['binary_transition'] == 0][feature].values
        elming = data[data['binary_transition'] == 1][feature].values

        if len(suppressed) == 0 or len(elming) == 0:
            print(f"Skipping {feature}: insufficient data")
            continue

        finite_vals = np.concatenate([suppressed[np.isfinite(suppressed)], elming[np.isfinite(elming)]])
        if finite_vals.size == 0:
            print(f"Skipping {feature}: no finite values")
            continue
            
        # Normalize x-axis using combined mean/std (z-score)
        mean_val = np.nanmean(finite_vals)
        std_val = np.nanstd(finite_vals)
        if not np.isfinite(std_val) or std_val == 0:
            print(f"Skipping {feature}: invalid statistics")
            continue
            
        suppressed_norm = (suppressed - mean_val) / std_val
        elming_norm = (elming - mean_val) / std_val

        # Determine common bin edges in normalized space
        finite_norm = np.concatenate([suppressed_norm[np.isfinite(suppressed_norm)], elming_norm[np.isfinite(elming_norm)]])
        if finite_norm.size == 0:
            print(f"Skipping {feature}: no finite normalized values")
            continue
            
        min_n, max_n = np.nanmin(finite_norm), np.nanmax(finite_norm)
        if not np.isfinite(min_n) or not np.isfinite(max_n) or min_n == max_n:
            print(f"Skipping {feature}: invalid normalized range")
            continue
            
        bin_edges = np.linspace(min_n, max_n, bins + 1)

        # Create individual plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(suppressed_norm, bins=bin_edges, alpha=0.5, label='To be suppressed', color='blue', density=True)
        ax.hist(elming_norm, bins=bin_edges, alpha=0.5, label='ELMing', color='red', density=True)

        ax.set_title(f'Feature Distribution: {feature}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Standardized Value (z-score)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{feature}_transition_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {feature}_transition_histogram.png")


def main():
    print("=== Transition ID Feature Distribution Histogram ===")
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print(f"Data loaded: {len(df)} rows")
    print(f"To be suppressed: {len(df[df['binary_transition'] == 0])} rows")
    print(f"ELMing: {len(df[df['binary_transition'] == 1])} rows")
    
    print("Selecting available features and generating distributions...")
    available_features = get_available_features(df)
    if not available_features:
        print("No selected features found in the dataset.")
        return
    print(f"Features: {', '.join(available_features)}")
    
    # Generate combined histogram
    plot_all_feature_distributions(df, available_features)
    print("Saved combined figure: 'all_transition_histograms_combined.png'")
    
    # Optionally generate individual histograms (commented out to focus on combined image)
    # print("Generating individual feature histograms...")
    # plot_individual_feature_distributions(df, available_features)
    # print("Individual histograms completed!")


if __name__ == "__main__":
    main()
