# Plasma State Classification and Analysis

A comprehensive machine learning project for classifying plasma states in fusion experiments using multiple approaches including Random Forest, CNN, and hybrid models.

## 🎯 Project Overview

This repository contains a complete machine learning pipeline for plasma state classification, focusing on identifying different plasma states (Suppressed, Mitigated, ELMing) in fusion experiments. The project includes extensive data analysis, model training, hyperparameter optimization, and visualization tools.

## 📊 Dataset

- **Primary Dataset**: `plasma_data.csv` - Contains plasma diagnostic measurements
- **Non-causal Dataset**: `noncausal_database.csv` - Alternative dataset for comparison
- **Features**: 10 key plasma parameters including `iln3iamp`, `betan`, `dR_sep`, `density`, `n_eped`, `li`, `tritop`, `fs04_max_smoothed`, `fs_sum`, `fs_up_sum`
- **Target States**: 4-state classification (Suppressed, Mitigated, ELMing, etc.)

## 🏗️ Project Structure

### Core Analysis Scripts
- **`Random_Forest.py`** - Main Random Forest implementation
- **`Optuna_Random_Forest.py`** - Hyperparameter optimization for Random Forest
- **`Optuna_CNN.py`** - CNN hyperparameter optimization with Optuna
- **`elm_classifier.py`** - Extreme Learning Machine implementation
- **`Transitional_Analysis.py`** - Analysis of plasma state transitions

### Specialized Modules

#### Random Forest Analysis (`Random_Forest/`)
- **`Shot_169501_Visualization.py`** - Comprehensive shot visualization with prediction overlays
- **`Isolated_Shot_169501_Visualization.py`** - True isolation testing (shot excluded from training)
- **`Binary_Random_Forest.py`** - Binary classification implementation
- **`Hyperparameter_Tuning_RF.py`** - RF hyperparameter optimization
- **`Forward_Feature_Selection.py`** - Feature selection algorithms
- **`Visual_Data_Analysis.py`** - Data visualization and analysis tools

#### CNN and Hybrid Models (`Hybrid_Model_*/`)
- **`CNN_Classifier.py`** - Convolutional Neural Network implementation
- **`Binary_CNN_Classifier.py`** - Binary CNN classification
- **`TCN_Classifier.py`** - Temporal Convolutional Network
- **`Binary_TCN_Classifier.py`** - Binary TCN implementation
- **`CNN_Center_Point_Random_Split.py`** - Center-point CNN with random splits

#### Causal Analysis (`CausalVariablesIsolator/`)
- **`Causal_Random_Forest.py`** - Causal variable analysis
- **`CausalDatabaseCreator.py`** - Database creation for causal analysis
- **`Hyperparameter_Tuning_Causal.py`** - Causal model optimization

#### Non-causal Analysis (`NoncausalVariablesIsolator/`)
- **`Noncausal_Random_Forest.py`** - Non-causal variable analysis
- **`NoncausalDatabaseCreator.py`** - Non-causal database creation
- **`Noncausal_Visual.py`** - Visualization for non-causal analysis

#### Advanced Models (`Random/Exploration/`)
- **`Binary_BiRNN_50ms.py`** - Bidirectional RNN implementation
- **`High_Accuracy_ELM_Ensemble.py`** - Ensemble ELM models
- **`Unsupervised_Learning.py`** - Unsupervised learning approaches

## 🚀 Key Features

### Model Performance
- **Random Forest**: 99.9% accuracy on full dataset, 88.2% on isolated shots
- **CNN Models**: Advanced deep learning approaches with attention mechanisms
- **Hybrid Models**: Combining CNN with center-point classification
- **Ensemble Methods**: Multiple model combinations for improved accuracy

### Visualization Tools
- **Shot Analysis**: Comprehensive visualization of individual shots with prediction overlays
- **Confusion Matrices**: Detailed performance analysis across all models
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Feature Importance**: Analysis of which plasma parameters are most predictive
- **State Transitions**: Visualization of plasma state changes over time

### Advanced Analysis
- **Isolation Testing**: True performance assessment by excluding test shots from training
- **Causal vs Non-causal**: Comparison of different variable sets
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Cross-validation**: Robust model evaluation with stratified splits

## 📈 Results Summary

### Random Forest Performance (Shot 169501)
| Metric | Full Dataset | Isolated Testing |
|--------|-------------|------------------|
| Overall Accuracy | 99.9% | 88.2% |
| Suppressed State | 99.9% | 87.6% |
| Mitigated State | 100.0% | 0.0% |
| ELMing State | 99.9% | 98.2% |

### Model Comparison
- **Random Forest**: Best overall performance, fast training
- **CNN**: Good for complex patterns, requires more data
- **ELM**: Fast training, good for real-time applications
- **Hybrid Models**: Balanced performance across different scenarios

## 🛠️ Installation & Usage

### Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision  # For CNN models
pip install optuna  # For hyperparameter optimization
pip install cupy-cuda11x  # For GPU acceleration (optional)
```

### Quick Start
```bash
# Run Random Forest analysis
python Random_Forest.py

# Visualize specific shot
cd Random_Forest/
python Shot_169501_Visualization.py

# Run hyperparameter optimization
python Optuna_Random_Forest.py
```

### Configuration
- Modify `SHOT_NUMBER` in visualization scripts to analyze different shots
- Adjust hyperparameters in optimization scripts
- Configure GPU settings in CNN scripts for acceleration

## 📁 Output Files

### Visualizations
- **Confusion Matrices**: `confusion_matrix_*.png`
- **ROC Curves**: `roc_curve*.png`
- **Feature Importance**: `feature_importance*.png`
- **Shot Analysis**: `shot_*_analysis.png`
- **State Distributions**: `state_distribution_histogram.png`

### Model Files
- **Trained Models**: `trained_*.pkl`
- **Feature Data**: `model_features*.pkl`
- **CNN Checkpoints**: `best_plasma_*.pth`

### Analysis Results
- **State Probabilities**: `state_probabilities_results.csv`
- **Optimization History**: `optimization_history.png`
- **Sequential Results**: `sequential_optimization_results.png`

## 🔬 Research Applications

This project is designed for:
- **Fusion Research**: Plasma state classification in tokamak experiments
- **Real-time Monitoring**: Fast classification for plasma control systems
- **Predictive Analysis**: Early detection of plasma instabilities
- **Machine Learning Research**: Comparative analysis of different ML approaches

## 📊 Data Analysis Features

- **Missing Value Analysis**: Comprehensive handling of incomplete data
- **Feature Distribution Analysis**: Statistical analysis of plasma parameters
- **Transition Analysis**: Study of plasma state changes
- **Cluster Analysis**: Unsupervised learning for pattern discovery
- **Decision Boundary Analysis**: Understanding model decision regions

## 🤝 Contributing

This is a research project focused on plasma physics and machine learning. Contributions are welcome for:
- New model architectures
- Additional visualization tools
- Performance optimizations
- Documentation improvements

## 📄 License

This project is for research purposes. Please cite appropriately if used in academic work.

## 🔗 Related Work

- Plasma state classification in fusion experiments
- Machine learning applications in plasma physics
- Real-time plasma monitoring systems
- Predictive control for fusion reactors

---

*This repository contains a comprehensive suite of tools for plasma state classification, combining traditional machine learning approaches with modern deep learning techniques for robust and accurate plasma state prediction.*
