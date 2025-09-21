import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Load and prepare the dataset from CSV
    """
    print("=== Loading Plasma Dataset from CSV ===")
    
    # Load the CSV dataset
    df = pd.read_csv('plasma_data.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove problematic shot
    df = df[df['shot'] != 191675].copy()
    print(f"After removing shot 191675: {df.shape}")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Add an index column to keep track of original row numbers
    df['original_index'] = df.index
    
    return df

def select_features_and_clean(df):
    """
    Select features and clean the dataset for binary classification
    """
    print("\n=== Feature Selection and Data Cleaning ===")
    
    # Select features based on the provided example
    selected_features = [
        'iln3iamp', 'iln2iamp', 'iun2iamp', 'iun3iamp', 'iln3iphase', 'iln2iphase', 'iun2iphase', 'iun3iphase',
        'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04_max_smoothed', 'fs04_max_avg'
    ]
    
    # Check which features are available
    available_features = [f for f in selected_features if f in df.columns]
    print(f"Available features: {available_features}")
    print(f"Features not found: {[f for f in selected_features if f not in df.columns]}")
    
    # Clean the dataframe - remove rows with missing values in selected features
    df_cleaned = df.dropna(subset=available_features, how='any')
    print(f"After removing rows with missing values: {len(df_cleaned)}")
    
    # Remove rows where state is 'N/A' (0)
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    print(f"After removing N/A states: {len(df_cleaned)}")
    
    # Map states to binary classification
    def map_states_to_binary(state):
        if state in [1, 2, 3]:  # Suppressed, Dithering, Mitigated -> Suppressed
            return 0
        elif state == 4:  # ELMing
            return 1
        else:
            return state
    
    df_cleaned['binary_state'] = df_cleaned['state'].apply(map_states_to_binary)
    
    # Strictly keep only the selected features plus columns required for splitting/label
    columns_to_keep = ['shot', 'time', 'binary_state'] + available_features
    df_cleaned = df_cleaned[columns_to_keep].copy()
    print(f"Columns kept for modeling (strict): {['shot', 'time', 'binary_state'] + available_features}")
    
    # Check binary state distribution
    binary_state_counts = df_cleaned['binary_state'].value_counts().sort_index()
    binary_state_names = {0: 'Suppressed', 1: 'ELMing'}
    
    print(f"\nBinary state distribution:")
    for state, count in binary_state_counts.items():
        if state in binary_state_names:
            print(f"  {binary_state_names[state]} (State {state}): {count:6d} records ({count/len(df_cleaned)*100:.1f}%)")
        else:
            print(f"  State {state}: {count:6d} records ({count/len(df_cleaned)*100:.1f}%)")
    
    return df_cleaned, available_features, binary_state_names

def prepare_chronological_splits(df_cleaned, available_features):
    """
    Prepare training/testing data with a 70/10/20 shot-based split using a random
    permutation of shots (seed=42), identical to Binary_Random_Forest.py.
    Validation shots are reported but not returned.
    """
    print("\n=== Data Splitting (70/10/20 by shot; random shot split seed=42) ===")

    # Random shot split identical to Binary_Random_Forest.py
    unique_shots = df_cleaned['shot'].unique()
    num_shots = len(unique_shots)
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    train_count = int(np.floor(0.70 * num_shots))
    val_count = int(np.floor(0.10 * num_shots))
    train_shots = shuffled_shots[:train_count]
    val_shots = shuffled_shots[train_count:train_count + val_count]
    test_shots = shuffled_shots[train_count + val_count:]

    print(f"Total shots: {num_shots} | Train shots: {len(train_shots)} | Val shots: {len(val_shots)} | Test shots: {len(test_shots)}")

    # Prepare input (X) and target (y)
    train_df = df_cleaned[df_cleaned['shot'].isin(train_shots)]
    val_df = df_cleaned[df_cleaned['shot'].isin(val_shots)]
    test_df = df_cleaned[df_cleaned['shot'].isin(test_shots)]

    X_train = train_df[available_features]
    y_train = train_df['binary_state']
    X_test = test_df[available_features]
    y_test = test_df['binary_state']

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier for binary classification
    """
    print("\n=== Training Random Forest (Binary Classification) ===")
    
    # Train a Random Forest Classifier with parameters from the example
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    clf.fit(X_train, y_train)
    
    print("Random Forest training completed!")
    
    return clf

def evaluate_model(clf, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model
    """
    print("\n=== Model Evaluation ===")
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (last 20% data): {accuracy:.4f}")
    
    # Evaluate on training set
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Check for overfitting
    overfitting = train_accuracy - accuracy
    print(f"Overfitting (Train - Test accuracy): {overfitting:.4f}")
    
    return y_pred, y_train_pred

def plot_confusion_matrix(y_test, y_pred, binary_state_names):
    """
    Plot confusion matrix with accuracy in title
    """
    print("\n=== Confusion Matrix ===")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get unique classes and create proper labels
    unique_classes = sorted(y_test.unique())
    class_labels = [binary_state_names.get(cls, f'Class {cls}') for cls in unique_classes]
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Normalized) - Accuracy: {accuracy:.4f}')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(clf, X_test, y_test, binary_state_names):
    """
    Plot ROC curve for binary classification
    """
    print("\n=== ROC Curve Analysis ===")
    
    # Get prediction probabilities for the positive class (ELMing)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (ELMing)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    # Plot ROC curve with white background and black grid
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('white')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    legend = plt.legend(loc="lower right")
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    plt.grid(True, color='black', alpha=0.3)
    plt.savefig('roc_curve_random_forest.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ROC AUC Score: {auc_score:.4f}")
    print(f"ROC curve saved as 'roc_curve_random_forest.png'")

def print_classification_report(y_test, y_pred, binary_state_names):
    """
    Print detailed classification report
    """
    print("\n=== Classification Report ===")
    
    # Get unique classes and create proper labels
    unique_classes = sorted(y_test.unique())
    class_labels = [binary_state_names.get(cls, f'Class {cls}') for cls in unique_classes]
    
    # Classification Report
    print(classification_report(y_test, y_pred, target_names=class_labels))

def analyze_feature_importance(clf, available_features):
    """
    Analyze and plot feature importance
    """
    print("\n=== Feature Importance Analysis ===")
    
    # Get feature importance
    feature_importances = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    
    print("Feature Importances:")
    print(feature_importances)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importances)))
    bars = plt.barh(range(len(feature_importances)), feature_importances.values, color=colors)
    plt.yticks(range(len(feature_importances)), feature_importances.index)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importances for Binary Plasma State Classification', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(feature_importances.head(10).items(), 1):
        print(f"{i:2d}. {feature}: {importance:.4f}")

def print_model_parameters(clf):
    """
    Print the Random Forest parameters
    """
    print(f"\n=== Random Forest Parameters ===")
    print(f"n_estimators: {clf.n_estimators}")
    print(f"max_depth: {clf.max_depth}")
    print(f"min_samples_split: {clf.min_samples_split}")
    print(f"min_samples_leaf: {clf.min_samples_leaf}")
    print(f"max_features: {clf.max_features}")
    print(f"random_state: {clf.random_state}")

def main():
    """
    Main execution function
    """
    print("=== Binary Plasma State Classification with Random Forest ===\n")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Select features and clean data
    df_cleaned, available_features, binary_state_names = select_features_and_clean(df)
    
    # Prepare chronological data splits
    X_train, X_test, y_train, y_test = prepare_chronological_splits(df_cleaned, available_features)
    
    # Train Random Forest
    clf = train_random_forest(X_train, y_train)
    
    # Evaluate model
    y_pred, y_train_pred = evaluate_model(clf, X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, binary_state_names)
    
    # Plot ROC curve
    plot_roc_curve(clf, X_test, y_test, binary_state_names)
    
    # Print classification report
    print_classification_report(y_test, y_pred, binary_state_names)
    
    # Analyze feature importance
    analyze_feature_importance(clf, available_features)
    
    # Print model parameters
    print_model_parameters(clf)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 