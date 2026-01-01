import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/raw/DSL-StrongPasswordData.csv")

y = df["subject"]
X = df.drop(columns=["subject"])

groups = df["subject"].astype(str) + "_sess" + df["sessionIndex"].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Train size:", len(train_idx), "Test size:", len(test_idx))
print("Unique subjects in train:", y_train.nunique(), "test:", y_test.nunique())
print("Test accuracy:", accuracy_score(y_test, pred))