import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

# Select only timing features (drop identifiers)
feature_cols = [c for c in df.columns if c not in ["subject", "sessionIndex", "rep"]]

# Compute within-user variance for each feature
within_var = (
    df
    .groupby("subject")[feature_cols]
    .var()
    .mean(axis=1)   # average variance across features
)

within_var = within_var.to_frame(name="mean_within_variance")

# Load per-user accuracy from previous step
# Recompute quickly here to avoid file passing
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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

labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)

per_user_acc = {
    user: cm[i, i] / cm[i].sum()
    for i, user in enumerate(labels)
}

acc_df = pd.Series(per_user_acc, name="accuracy")

# Merge
analysis_df = pd.concat([acc_df, within_var], axis=1).dropna()

# Correlation
corr = analysis_df["accuracy"].corr(analysis_df["mean_within_variance"])

print("Correlation between accuracy and within-user variance:", corr)

print("\nLowest accuracy users:")
print(analysis_df.sort_values("accuracy").head(5))

print("\nHighest accuracy users:")
print(analysis_df.sort_values("accuracy").tail(5))