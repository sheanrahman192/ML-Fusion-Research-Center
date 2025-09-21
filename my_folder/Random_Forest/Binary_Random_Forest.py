import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('plasma_data.csv')
    df = df[df['shot'] != 191675].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['original_index'] = df.index
    return df

def prepare_binary_classification(df):
    """Prepare data for binary classification"""
    selected_features = [
        'fs04_max_smoothed', 'fs04_max_avg', 'bt0', 'rotation_core', 'dR_sep'
        'iln3iamp', 'kappa','p_eped','li', 'bt', 'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop', 'fs04', 'tribot'
    ]
    
    available_features = [f for f in selected_features if f in df.columns]
    df_cleaned = df.dropna(subset=available_features, how='any')
    df_cleaned = df_cleaned[df_cleaned['state'] != 0]
    
    # Map states to binary: 1-3 -> 0 (Suppressed), 4 -> 1 (ELMing)
    df_cleaned['binary_state'] = df_cleaned['state'].apply(lambda x: 0 if x in [1,2,3] else 1)
    
    columns_to_keep = ['shot', 'time', 'binary_state'] + available_features
    df_cleaned = df_cleaned[columns_to_keep].copy()
    
    return df_cleaned, available_features

def split_data_by_shots(df_cleaned, available_features):
    """Split data by unique shots: 70% train, 10% CV, 20% test"""
    unique_shots = df_cleaned['shot'].unique()
    num_shots = len(unique_shots)
    
    # Shuffle shots first, then split
    np.random.seed(42)
    shuffled_shots = np.random.permutation(unique_shots)
    
    # Split shuffled shots: 70% train, 10% CV, 20% test
    train_count = int(np.floor(0.70 * num_shots))
    cv_count = int(np.floor(0.10 * num_shots))
    
    train_shots = shuffled_shots[:train_count]
    cv_shots = shuffled_shots[train_count:train_count + cv_count]
    test_shots = shuffled_shots[train_count + cv_count:]
    
    # Get data for each split
    train_df = df_cleaned[df_cleaned['shot'].isin(train_shots)]
    test_df = df_cleaned[df_cleaned['shot'].isin(test_shots)]
    
    X_train = train_df[available_features]
    y_train = train_df['binary_state']
    X_test = test_df[available_features]
    y_test = test_df['binary_state']
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Suppressed', 'ELMing'],
                yticklabels=['Suppressed', 'ELMing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.4f}')
    plt.tight_layout()
    plt.show()
    #plt.savefig('confusion_matrix_random_forest.png', dpi=1200, bbox_inches='tight')

def plot_roc_curve(clf, X_test, y_test):
    """Plot ROC curve"""
    y_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    #plt.savefig('roc_curve_random_forest.png', dpi=1200, bbox_inches='tight')

def train_and_evaluate():
    """Main function to train and evaluate the model"""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    df_cleaned, available_features = prepare_binary_classification(df)
    X_train, X_test, y_train, y_test = split_data_by_shots(df_cleaned, available_features)
    
    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=7, min_samples_split=3,
        min_samples_leaf=12, max_features=0.5, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # ROC curve
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc_score:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(clf, X_test, y_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Suppressed', 'ELMing']))
    
    return clf, X_test, y_test, y_pred

if __name__ == "__main__":
    clf, X_test, y_test, y_pred = train_and_evaluate() 