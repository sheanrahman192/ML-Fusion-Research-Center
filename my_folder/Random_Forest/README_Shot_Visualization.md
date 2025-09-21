# Shot Visualization Script for Random Forest Analysis

## Overview
This script (`Shot_169501_Visualization.py`) provides a comprehensive visualization and analysis of Random Forest predictions for any shot in the plasma database. It overlays the predictions with the fs04 time series to show which portions were correctly and incorrectly identified.

## Key Improvements Made

### 1. Legend Positioning
- **Fixed legend overlap issues** by positioning all legends outside the plot area
- **Enhanced legend styling** with frames, shadows, and better positioning
- **Separated colorbar and legend** to prevent visual conflicts
- **Increased figure size** to accommodate legends without overlap

### 2. Enlarged fs04 Time Series Plot
- **Made the prediction accuracy overlay plot significantly larger** (24x18 figure size)
- **Increased line thickness** for better visibility
- **Larger scatter points** for confidence markers (size 40)
- **Enhanced title** to indicate the enlarged nature of the plot
- **Better spacing** between subplots for improved readability

### 3. Improved Layout
- **Better subplot spacing** with `hspace=0.3` for clear separation
- **Optimized right margin** (88% of figure width) to accommodate legends
- **Enhanced font sizes** throughout for better readability
- **Professional styling** with frames and shadows on legends

## Features

### 1. Model Training
- Loads the plasma data from `../plasma_data.csv`
- Trains a Random Forest classifier using the same parameters as the original script:
  - 500 estimators
  - Max depth of 35
  - Random state 42 for reproducibility
- Uses the same feature set: `['iln3iamp', 'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs_sum', 'fs_up_sum']`

### 2. Shot Analysis
- Extracts all time series data for the specified shot
- Makes predictions on the entire shot time series
- Calculates prediction accuracy and confidence metrics
- **Configurable shot number** - easily change `SHOT_NUMBER` variable at the top

### 3. Visualizations

#### Comprehensive Analysis Plot (`shot_{SHOT_NUMBER}_comprehensive_analysis.png`)
- **Top Panel**: fs04 time series with actual states color-coded
- **Middle Panel**: fs04 time series with predicted states color-coded  
- **Bottom Panel**: **ENLARGED** fs04 time series with correct/incorrect prediction overlays and confidence markers
- **All legends positioned outside plot area** to prevent overlap

#### Detailed Analysis Plot (`shot_{SHOT_NUMBER}_detailed_analysis.png`)
- **Top Panel**: State transitions over time (actual vs predicted)
- **Bottom Panel**: fs04 with prediction accuracy markers
- **Legends positioned in upper right** to avoid overlap

### 4. Statistical Analysis
The script provides detailed statistics including:
- Overall prediction accuracy
- Accuracy breakdown by actual state
- Prediction confidence statistics
- Confidence comparison between correct and incorrect predictions

## Results for Shot 169501

- **Total time points**: 4,330
- **Overall accuracy**: 99.9% (4,326 correct, 4 incorrect)
- **State distribution**: 
  - Suppressed (State 1): 2,258 points (99.9% accuracy)
  - Mitigated (State 3): 196 points (100% accuracy)
  - ELMing (State 4): 1,876 points (99.9% accuracy)
- **Mean prediction confidence**: 98.4%

## Usage

```bash
cd /mnt/homes/sr4240/my_folder/Random_Forest
python Shot_169501_Visualization.py
```

To analyze a different shot, simply change the `SHOT_NUMBER` variable at the top of the script.

## Output Files
All visualization files are automatically saved in the `prediction_visualizations/` folder:
- `prediction_visualizations/shot_{SHOT_NUMBER}_comprehensive_analysis.png`: Main visualization with fs04 overlay
- `prediction_visualizations/shot_{SHOT_NUMBER}_detailed_analysis.png`: Detailed state transition analysis

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
- The Random Forest model achieves excellent performance with 99.9% accuracy
- Only 4 out of 4,330 predictions were incorrect, demonstrating the model's reliability