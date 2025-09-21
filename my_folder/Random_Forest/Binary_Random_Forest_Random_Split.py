import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

plt.close('all')

# Load the plasma data from CSV file
df = pd.read_csv('plasma_data.csv')

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Add an index column to keep track of original row numbers
df['original_index'] = df.index

# Select features and the target variable
selected_features = ['iln3iamp', 'betan', 'dR_sep', 'density', 'n_eped', 'li', 'tritop','fs04_max_smoothed', 'fs_sum', 'fs_up_sum']
target_column = 'state'

# Clean the dataframe
df_cleaned = df.dropna(subset=selected_features, how='any')
df_cleaned = df_cleaned[df_cleaned['n'] == 3]

# Create binary classification: states 1, 2, 3 -> 0 (Suppressed), state 4 -> 1 (ELMing)
df_cleaned['binary_state'] = np.where(df_cleaned['state'].isin([1, 2, 3]), 0, 1)

# Prepare input (X) and target (y) for binary classification
X = df_cleaned[selected_features]
y = df_cleaned['binary_state']

# Split into training and testing sets
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df_cleaned['original_index'], test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=500, max_depth=35, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Binary Classification Accuracy: {accuracy:.2f}")

# Confusion Matrix
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Suppressed (States 1,2,3)', 'ELMing (State 4)'],
            yticklabels=['Suppressed (States 1,2,3)', 'ELMing (State 4)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Binary Classification Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Suppressed (States 1,2,3)', 'ELMing (State 4)']))

# Add the predictions to the DataFrame
df_test = X_test.copy()
df_test['actual_binary_state'] = y_test
df_test['pred_binary_state'] = y_pred

# Create a column to classify the prediction results for binary classification
df_test['condition'] = np.where(df_test['pred_binary_state'] == df_test['actual_binary_state'], 'Correct',
                                'Incorrect')

# Add detailed prediction analysis for each binary class
print("\nDetailed Prediction Analysis by Binary Class:")
for binary_state, state_name in [(0, 'Suppressed (States 1,2,3)'), (1, 'ELMing (State 4)')]:
    state_mask = df_test['actual_binary_state'] == binary_state
    if state_mask.sum() > 0:
        correct_predictions = (df_test.loc[state_mask, 'pred_binary_state'] == binary_state).sum()
        total_predictions = state_mask.sum()
        accuracy = correct_predictions / total_predictions
        print(f"{state_name}: {correct_predictions}/{total_predictions} correct ({accuracy:.3f})")

# Print summary statistics
print(f"\nSummary:")
print(f"Final binary classification accuracy: {accuracy:.3f}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Number of features: {len(selected_features)}")

# Show class distribution
print(f"\nClass Distribution:")
print(f"Suppressed (States 1,2,3): {(y == 0).sum()} samples")
print(f"ELMing (State 4): {(y == 1).sum()} samples")
