import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Load data
df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

feature_cols = [c for c in df.columns if c not in ["subject", "sessionIndex", "rep"]]

# Scale features (important for distance calculations)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
df_scaled["subject"] = df["subject"].values

# ---- 1. Compute centroids per user ----
centroids = (
    df_scaled
    .groupby("subject")[feature_cols]
    .mean()
)

# ---- 2. Compute within-user spread ----
within_spread = {}

for subject in centroids.index:
    user_points = df_scaled[df_scaled["subject"] == subject][feature_cols].values
    centroid = centroids.loc[subject].values.reshape(1, -1)

    # average distance to centroid
    dists = pairwise_distances(user_points, centroid)
    within_spread[subject] = dists.mean()

within_spread = pd.Series(within_spread, name="within_spread")

# ---- 3. Compute between-user distance ----
centroid_dist_matrix = pairwise_distances(centroids.values)

between_dist = {}
subjects = centroids.index.tolist()

for i, subject in enumerate(subjects):
    # mean distance to all other centroids
    other_dists = np.delete(centroid_dist_matrix[i], i)
    between_dist[subject] = other_dists.mean()

between_dist = pd.Series(between_dist, name="between_distance")

# ---- 4. Separability score ----
separability = (between_dist / within_spread).rename("separability")

# ---- Load per-user accuracy (recomputed cleanly) ----
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
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

accuracy = {
    user: cm[i, i] / cm[i].sum()
    for i, user in enumerate(labels)
}

accuracy = pd.Series(accuracy, name="accuracy")

# ---- Combine everything ----
analysis_df = pd.concat(
    [accuracy, within_spread, between_dist, separability],
    axis=1
).dropna()

# ---- Results ----
corr = analysis_df["accuracy"].corr(analysis_df["separability"])

print("Correlation between separability and accuracy:", corr)

print("\nLowest separability users:")
print(analysis_df.sort_values("separability").head(5))

print("\nHighest separability users:")
print(analysis_df.sort_values("separability").tail(5))