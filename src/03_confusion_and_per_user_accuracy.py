import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

y = df["subject"]
X = df.drop(columns=["subject"])

groups = df["subject"].astype(str) + "_sess" + df["sessionIndex"].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Per-user accuracy
per_user_accuracy = {}
for i, user in enumerate(labels):
    correct = cm[i, i]
    total = cm[i].sum()
    per_user_accuracy[user] = correct / total

acc_df = (
    pd.DataFrame.from_dict(per_user_accuracy, orient="index", columns=["accuracy"])
    .sort_values("accuracy")
)

print("Lowest 10 per-user accuracies:")
print(acc_df.head(10))

print("\nHighest 10 per-user accuracies:")
print(acc_df.tail(10))

print("\nSummary statistics:")
print(acc_df.describe())