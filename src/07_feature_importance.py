import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

y = df["subject"]
X = df.drop(columns=["subject"])
feature_names = X.columns.tolist()

# Prevent session leakage
groups = df["subject"].astype(str) + "_sess" + df["sessionIndex"].astype(str)

splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

importance_matrix = []

for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=split_id
    )

    rf.fit(X_train, y_train)
    importance_matrix.append(rf.feature_importances_)

importance_df = pd.DataFrame(
    importance_matrix,
    columns=feature_names
)

summary = pd.DataFrame({
    "mean_importance": importance_df.mean(),
    "std_importance": importance_df.std()
}).sort_values("mean_importance", ascending=False)

print("Top 10 features by mean importance:")
print(summary.head(10))

print("\nBottom 10 features by mean importance:")
print(summary.tail(10))