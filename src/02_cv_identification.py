import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

y = df["subject"]
X = df.drop(columns=["subject"])

groups = df["subject"].astype(str) + "_sess" + df["sessionIndex"].astype(str)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000))
])

splitter = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

scores = []

for i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    scores.append(acc)
    print(f"Split {i+1} accuracy: {acc:.3f}")

scores = np.array(scores)
print("\nMean accuracy:", scores.mean())
print("Std dev:", scores.std())