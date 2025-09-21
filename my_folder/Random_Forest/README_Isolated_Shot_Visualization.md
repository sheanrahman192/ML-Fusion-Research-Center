# Isolated Shot Visualization Script for Random Forest Analysis

## Overview
This script (`Isolated_Shot_169501_Visualization.py`) provides a comprehensive visualization and analysis of Random Forest predictions for shot 169501 while ensuring that shot 169501 is **completely excluded from the training data**. This provides a more realistic assessment of the model's performance on truly unseen data.

## Key Differences from Regular Shot Visualization

### 1. **True Isolation Testing**
- **Excludes target shot from training**: Shot 169501 is completely removed from the training dataset
- **Realistic performance assessment**: Shows how the model performs on truly unseen data
- **More conservative accuracy**: Typically shows lower accuracy than when the shot is included in training

### 2. **Training Data Reduction**
- **Reduced training data**: Uses 203,717 rows instead of full dataset (excluding shot 169501)
- **Faster training**: Uses 100 estimators instead of 500 for quicker execution
- **Same model parameters**: Maintains same Random Forest configuration for consistency

### 3. **Enhanced Visualizations**
- **Clear isolation indicators**: All plots clearly indicate that the shot was excluded from training
- **Realistic performance metrics**: Shows true generalization capability
- **Isolation warnings**: Multiple reminders that the shot was excluded from training

## Key Improvements Made

### 1. **Legend Positioning**
- **Fixed legend overlap issues** by positioning all legends outside the plot area
- **Enhanced legend styling** with frames, shadows, and better positioning
- **Separated colorbar and legend** to prevent visual conflicts
- **Increased figure size** to accommodate legends without overlap

### 2. **Enlarged fs04 Time Series Plot**
- **Made the prediction accuracy overlay plot significantly larger** (24x18 figure size)
- **Increased line thickness** for better visibility
- **Larger scatter points** for confidence markers (size 40)
- **Enhanced title** to indicate the enlarged nature and isolation status

### 3. **Improved Layout**
- **Better subplot spacing** with `hspace=0.3` for clear separation
- **Optimized right margin** (88% of figure width) to accommodate legends
- **Enhanced font sizes** throughout for better readability
- **Professional styling** with frames and shadows on legends

## Features

### 1. Model Training (Isolated)
- Loads the plasma data from `../plasma_data.csv`
- **Excludes shot 169501 from training data** (203,717 training rows)
- Trains a Random Forest classifier with:
  - 100 estimators (reduced for faster training)
  - Max depth of 35
  - Random state 42 for reproducibility
- Uses the same feature set: `['iln3iamp', 'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs_sum', 'fs_up_sum']`

### 2. Shot Analysis
- Extracts all time series data for shot 169501 (4,330 time points)
- Makes predictions on the isolated shot time series
- Calculates prediction accuracy and confidence metrics
- **Configurable shot number** - easily change `SHOT_NUMBER` variable at the top

### 3. Visualizations

#### Comprehensive Analysis Plot (`isolated_shot_{SHOT_NUMBER}_comprehensive_analysis.png`)
- **Top Panel**: fs04 time series with actual states color-coded
- **Middle Panel**: fs04 time series with predicted states color-coded  
- **Bottom Panel**: **ENLARGED** fs04 time series with correct/incorrect prediction overlays and confidence markers
- **All legends positioned outside plot area** to prevent overlap
- **Clear isolation indicators** in titles

#### Detailed Analysis Plot (`isolated_shot_{SHOT_NUMBER}_detailed_analysis.png`)
- **Top Panel**: State transitions over time (actual vs predicted)
- **Bottom Panel**: fs04 with prediction accuracy markers
- **Legends positioned in upper right** to avoid overlap
- **Isolation status clearly indicated**

### 4. Statistical Analysis
The script provides detailed statistics including:
- Overall prediction accuracy (typically lower than non-isolated version)
- Accuracy breakdown by actual state
- Prediction confidence statistics
- Confidence comparison between correct and incorrect predictions

## Results for Isolated Shot 169501

- **Total time points**: 4,330
- **Overall accuracy**: 88.2% (3,821 correct, 509 incorrect)
- **State distribution**: 
  - Suppressed (State 1): 2,258 points (87.6% accuracy)
  - Mitigated (State 3): 196 points (0.0% accuracy - model struggles with this state)
  - ELMing (State 4): 1,876 points (98.2% accuracy)
- **Mean prediction confidence**: 88.1%

## Comparison with Non-Isolated Version

| Metric | Non-Isolated | Isolated | Difference |
|--------|-------------|----------|------------|
| Overall Accuracy | 99.9% | 88.2% | -11.7% |
| Suppressed Accuracy | 99.9% | 87.6% | -12.3% |
| Mitigated Accuracy | 100.0% | 0.0% | -100.0% |
| ELMing Accuracy | 99.9% | 98.2% | -1.7% |
| Mean Confidence | 98.4% | 88.1% | -10.3% |

## Usage

```bash
cd /mnt/homes/sr4240/my_folder/Random_Forest
python Isolated_Shot_169501_Visualization.py
```

To analyze a different shot, simply change the `SHOT_NUMBER` variable at the top of the script.

## Output Files
All visualization files are automatically saved in the `prediction_visualizations/` folder:
- `prediction_visualizations/isolated_shot_{SHOT_NUMBER}_comprehensive_analysis.png`: Main visualization with fs04 overlay
- `prediction_visualizations/isolated_shot_{SHOT_NUMBER}_detailed_analysis.png`: Detailed state transition analysis

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Notes
- The script automatically creates a `prediction_visualizations/` folder and saves all plots there
- The script automatically saves the plots as high-resolution PNG files (300 DPI)
- All visualizations include proper legends, labels, and color coding
- **Legends are positioned outside plot areas** to prevent overlap
- **The fs04 prediction accuracy plot is significantly enlarged** for better visibility
- **Shot 169501 is completely excluded from training data** for realistic performance assessment
- The isolated model shows more realistic performance metrics (88.2% vs 99.9% accuracy)
- The model particularly struggles with the "Mitigated" state when the shot is isolated
