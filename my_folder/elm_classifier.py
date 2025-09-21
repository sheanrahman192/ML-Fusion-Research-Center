#!/usr/bin/env python3
"""
ELM Classifier for Plasma Data
Classifies ELMing (state 4) vs Suppressed (states 1,2,3) states
Trains on some shots, tests on others, but classifies every millisecond
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ELMClassifier:
    def __init__(self):
        self.best_model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_names = None
        self.feature_importance = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the plasma data"""
        print("Loading plasma data...")
        df = pd.read_csv(file_path)
        
        # Remove header row if it exists
        if df.iloc[0, 2] == 'state':
            df = df.iloc[1:].reset_index(drop=True)
        
        # Convert state to numeric
        df['state'] = pd.to_numeric(df['state'], errors='coerce')
        
        # Create binary classification: ELMing (4) vs Suppressed (1,2,3)
        df['is_elming'] = (df['state'] == 4).astype(int)
        
        # Select relevant features based on the image description
        # fs04 is the key indicator (red spiky line)
        # betan and density are also important (purple and green lines)
        feature_columns = [
            'fs04', 'betan', 'density', 'iln3iamp',  # Key features from image
            'n', 'n_eped', 't_eped', 'p_eped',      # Density-related features
            'li', 'q95', 'Ip', 'bt0', 'bt',         # Plasma parameters
            'kappa', 'tribot', 'tritop', 'dR_sep',  # Geometry parameters
            'zeff', 'rotation_edge', 'rotation_core', # Additional plasma properties
            'fs04_max_smoothed', 'fs04_max_avg', 'thin_fs04_max_smoothed', 'fs_sum', 'fs_up_sum'  # fs04 derivatives
        ]
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Available features: {available_features}")
        
        # Create feature matrix and target
        X = df[available_features].copy()
        y = df['is_elming']
        shot_ids = df['shot']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Create engineered features
        X = self.create_engineered_features(X)
        
        self.feature_names = list(X.columns)
        
        return X, y, shot_ids, df
    
    def create_engineered_features(self, X):
        """Create engineered features to improve classification"""
        print("Creating engineered features...")
        X_eng = X.copy()
        
        # fs04-based features (key indicator from image)
        if 'fs04' in X.columns:
            print("  Adding fs04-based features...")
            # Rolling statistics for fs04
            X_eng['fs04_rolling_mean_5'] = X['fs04'].rolling(window=5, min_periods=1).mean()
            X_eng['fs04_rolling_std_5'] = X['fs04'].rolling(window=5, min_periods=1).std()
            X_eng['fs04_rolling_max_10'] = X['fs04'].rolling(window=10, min_periods=1).max()
            X_eng['fs04_rolling_min_10'] = X['fs04'].rolling(window=10, min_periods=1).min()
            X_eng['fs04_range_10'] = X_eng['fs04_rolling_max_10'] - X_eng['fs04_rolling_min_10']
            
            # fs04 change rate
            X_eng['fs04_diff'] = X['fs04'].diff().fillna(0)
            X_eng['fs04_diff_abs'] = np.abs(X_eng['fs04_diff'])
            
            # fs04 threshold features
            X_eng['fs04_high_activity'] = (X['fs04'] > X['fs04'].quantile(0.8)).astype(int)
            X_eng['fs04_low_activity'] = (X['fs04'] < X['fs04'].quantile(0.2)).astype(int)
        
        # Density-based features
        if 'density' in X.columns:
            print("  Adding density-based features...")
            X_eng['density_rolling_mean_5'] = X['density'].rolling(window=5, min_periods=1).mean()
            X_eng['density_rolling_std_5'] = X['density'].rolling(window=5, min_periods=1).std()
            X_eng['density_diff'] = X['density'].diff().fillna(0)
        
        # betan-based features
        if 'betan' in X.columns:
            print("  Adding betan-based features...")
            X_eng['betan_rolling_mean_5'] = X['betan'].rolling(window=5, min_periods=1).mean()
            X_eng['betan_rolling_std_5'] = X['betan'].rolling(window=5, min_periods=1).std()
            X_eng['betan_diff'] = X['betan'].diff().fillna(0)
        
        # Interaction features
        if all(col in X.columns for col in ['fs04', 'density', 'betan']):
            print("  Adding interaction features...")
            X_eng['fs04_density_ratio'] = X['fs04'] / (X['density'] + 1e-8)
            X_eng['fs04_betan_ratio'] = X['fs04'] / (X['betan'] + 1e-8)
            X_eng['density_betan_ratio'] = X['density'] / (X['betan'] + 1e-8)
        
        # Plasma stability features
        if all(col in X.columns for col in ['li', 'q95']):
            print("  Adding plasma stability features...")
            X_eng['li_q95_ratio'] = X['li'] / (X['q95'] + 1e-8)
            X_eng['stability_index'] = X['li'] * X['q95']
        
        # Handle any new NaN values
        X_eng = X_eng.fillna(X_eng.median())
        
        print(f"  Total features after engineering: {X_eng.shape[1]}")
        return X_eng
    
    def create_shot_based_split(self, X, y, shot_ids, test_size=0.3):
        """Split data by shot IDs, ensuring no shot appears in both train and test"""
        unique_shots = shot_ids.unique()
        train_shots, test_shots = train_test_split(
            unique_shots, test_size=test_size, random_state=42
        )
        
        train_mask = shot_ids.isin(train_shots)
        test_mask = shot_ids.isin(test_shots)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"Training on {len(train_shots)} shots ({len(X_train)} samples)")
        print(f"Testing on {len(test_shots)} shots ({len(X_test)} samples)")
        print(f"Train shots: {sorted(train_shots)[:10]}...")
        print(f"Test shots: {sorted(test_shots)[:10]}...")
        
        return X_train, X_test, y_train, y_test, train_shots, test_shots
    
    def train_model(self, X_train, y_train):
        """Train the ELM classifier with multiple models and hyperparameter tuning"""
        print("Training ELM classifier with advanced models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Feature selection using RFE
        base_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        self.feature_selector = RFE(estimator=base_rf, n_features_to_select=min(20, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        
        # Get selected feature names
        selected_features = self.feature_selector.get_support()
        self.selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_features[i]]
        
        print(f"Selected {len(self.selected_feature_names)} features: {self.selected_feature_names[:10]}...")
        
        # Define multiple models to try
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Define parameter grids for each model
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ExtraTrees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced']
            }
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        
        # Try each model with hyperparameter tuning
        total_models = len(models)
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"[{i}/{total_models}] Tuning {model_name}...")
            try:
                grid_search = GridSearchCV(
                    model, param_grids[model_name], 
                    cv=3, scoring='accuracy', n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train_selected, y_train)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name
                    print(f"  New best model found!")
                    
                print(f"  {model_name} best CV score: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                continue
        
        self.best_model = best_model
        print(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
        
        # Get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_[0])
        else:
            self.feature_importance = None
        
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        return self.best_model.predict(X_selected)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        return self.best_model.predict_proba(X_selected)
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if self.feature_importance is not None and self.selected_feature_names is not None:
            importance_dict = dict(zip(self.selected_feature_names, self.feature_importance))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return None
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        print("Evaluating model...")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba[:, 1])
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Suppressed', 'ELMing']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print feature importance
        importance = self.get_feature_importance()
        if importance:
            print("\nTop 10 Most Important Features:")
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                print(f"{i+1:2d}. {feature}: {imp:.4f}")
        
        return accuracy, y_pred, y_proba
    
    def analyze_shot_predictions(self, df, y_pred, y_proba, test_shots):
        """Analyze predictions for specific shots"""
        # Add predictions to dataframe
        df_test = df[df['shot'].isin(test_shots)].copy()
        df_test['predicted_elming'] = y_pred
        df_test['elming_probability'] = y_proba[:, 1]
        
        # Print summary statistics
        print(f"\nPrediction Summary for Test Shots:")
        print(f"Total test samples: {len(df_test)}")
        print(f"Predicted ELMing: {df_test['predicted_elming'].sum()} ({df_test['predicted_elming'].mean()*100:.1f}%)")
        print(f"Actual ELMing: {(df_test['state'] == 4).sum()} ({(df_test['state'] == 4).mean()*100:.1f}%)")
        
        # Analyze per-shot accuracy
        shot_accuracies = []
        for shot in test_shots:
            shot_data = df_test[df_test['shot'] == shot]
            if len(shot_data) > 0:
                actual_elming = (shot_data['state'] == 4).astype(int)
                predicted_elming = shot_data['predicted_elming']
                shot_acc = accuracy_score(actual_elming, predicted_elming)
                shot_accuracies.append((shot, shot_acc, len(shot_data)))
        
        shot_accuracies.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 Most Accurate Test Shots:")
        for shot, acc, n_samples in shot_accuracies[:5]:
            print(f"Shot {shot}: {acc:.4f} ({n_samples} samples)")
        
        print(f"\nBottom 5 Least Accurate Test Shots:")
        for shot, acc, n_samples in shot_accuracies[-5:]:
            print(f"Shot {shot}: {acc:.4f} ({n_samples} samples)")
        
        return df_test

def main():
    """Main function to run the ELM classifier"""
    print("=== ELM Classifier for Plasma Data ===")
    
    # Initialize classifier
    classifier = ELMClassifier()
    
    # Load and preprocess data
    X, y, shot_ids, df = classifier.load_and_preprocess_data('plasma_data.csv')
    
    print(f"Total samples: {len(X)}")
    print(f"ELMing samples: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Suppressed samples: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    
    # Create shot-based train/test split
    X_train, X_test, y_train, y_test, train_shots, test_shots = classifier.create_shot_based_split(
        X, y, shot_ids
    )
    
    # Train model
    classifier.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, y_pred, y_proba = classifier.evaluate_model(X_test, y_test)
    
    # Analyze predictions for specific shots
    df_test_with_predictions = classifier.analyze_shot_predictions(df, y_pred, y_proba, test_shots)
    
    print("\n=== Analysis Complete ===")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
    
    # Check if we need to improve further
    if accuracy < 0.95:
        print(f"\nWarning: Accuracy {accuracy:.4f} is below 95%. Consider:")
        print("1. Adding more engineered features")
        print("2. Trying different model architectures")
        print("3. Adjusting the train/test split ratio")
        print("4. Using ensemble methods")
    else:
        print(f"\nSuccess: Target accuracy of 95% achieved! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
