import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

feature_cols = [c for c in df.columns if c not in ["subject", "sessionIndex", "rep"]]

# Build pairwise dataset
pairs_X = []
pairs_y = []

rng = np.random.default_rng(42)
subjects = df["subject"].unique()

for subj in subjects:
    user_data = df[df["subject"] == subj][feature_cols].values
    other_data = df[df["subject"] != subj][feature_cols].values

    # same-user pairs
    for _ in range(100):
        i, j = rng.choice(len(user_data), size=2, replace=False)
        pairs_X.append(np.abs(user_data[i] - user_data[j]))
        pairs_y.append(1)

    # different-user pairs
    for _ in range(100):
        i = rng.integers(len(user_data))
        j = rng.integers(len(other_data))
        pairs_X.append(np.abs(user_data[i] - other_data[j]))
        pairs_y.append(0)

X = np.array(pairs_X)
y = np.array(pairs_y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Verification accuracy:", accuracy_score(y_test, pred))
print("Verification ROC AUC:", roc_auc_score(y_test, proba))