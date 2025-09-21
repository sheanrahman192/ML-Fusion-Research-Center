import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the plasma data from CSV file
df = pd.read_csv('plasma_data.csv')

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Add an index column to keep track of original row numbers
df['original_index'] = df.index

# Select features and the target variable
selected_features = [
    'iln3iamp', 'betan', 'density', 'n_eped', 'li', 'tritop',
    'fs04_max_smoothed', 'fs_sum', 'fs_up_sum'
]
target_column = 'state'

# Clean the dataframe
df_cleaned = df.dropna(subset=selected_features, how='any')
df_cleaned = df_cleaned[df_cleaned['n'] == 3]

# Prepare input (X) and target (y)
X = df_cleaned[selected_features]
y = df_cleaned[target_column]

# Split into training and testing sets (capture indices to reuse the same split)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df_cleaned['original_index'], test_size=0.2, random_state=42
)

current_features = list(selected_features)

iteration = 0
while len(current_features) >= 1:
    print(f"Features: {current_features}")

    # Align to the same split each time
    X_train_iter = X_train[current_features]
    X_test_iter = X_test[current_features]

    clf_iter = RandomForestClassifier(
        n_estimators=500,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    clf_iter.fit(X_train_iter, y_train)

    y_pred_iter = clf_iter.predict(X_test_iter)
    acc_iter = accuracy_score(y_test, y_pred_iter)
    mitigated_mask = (y_test == 3)
    if mitigated_mask.any():
        mitigated_acc = (y_pred_iter[mitigated_mask] == 3).mean()
    else:
        mitigated_acc = float('nan')

    print(f"Overall accuracy: {acc_iter:.4f}")
    print(f"Mitigated accuracy: {mitigated_acc:.4f}")

    # Stop if only one feature remains (cannot drop further)
    if len(current_features) == 1:
        break

    # Compute importances among current features and drop the least important
    importances_series = pd.Series(clf_iter.feature_importances_, index=current_features)
    least_feature = importances_series.idxmin()
    current_features = [f for f in current_features if f != least_feature]
    iteration += 1
