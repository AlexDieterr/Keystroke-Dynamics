import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

y = df["subject"]
X = df.drop(columns=["subject"])

# Group by subject-session to prevent leakage
groups = df["subject"].astype(str) + "_sess" + df["sessionIndex"].astype(str)

splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

accuracies = []

for i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

    print(f"Split {i+1} accuracy: {acc:.3f}")

accuracies = np.array(accuracies)

print("\nMean RF accuracy:", accuracies.mean())
print("Std dev RF accuracy:", accuracies.std())