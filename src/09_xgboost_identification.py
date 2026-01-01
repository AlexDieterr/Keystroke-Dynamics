import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

y = df["subject"]
X = df.drop(columns=["subject"])

# Encode labels as integers for XGBoost
label_map = {label: i for i, label in enumerate(y.unique())}
y_encoded = y.map(label_map)

# Prevent session leakage
groups = df["subject"].astype(str) + "_sess" + df["sessionIndex"].astype(str)

splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

accuracies = []

for i, (train_idx, test_idx) in enumerate(splitter.split(X, y_encoded, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(label_map),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)
    print(f"Split {i+1} accuracy: {acc:.3f}")

accuracies = np.array(accuracies)
print("\nMean XGBoost accuracy:", accuracies.mean())
print("Std dev XGBoost accuracy:", accuracies.std())