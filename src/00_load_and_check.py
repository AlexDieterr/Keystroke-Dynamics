import pandas as pd

PATH = "data/raw/DSL-StrongPasswordData.csv"

df = pd.read_csv(PATH)

print("Shape:", df.shape)
print("Columns:", len(df.columns))
print(df.head(3))

assert {"subject", "sessionIndex", "rep"}.issubset(df.columns)

print("\nUnique subjects:", df["subject"].nunique())
print("Sessions per subject (unique sessionIndex):")
print(df.groupby("subject")["sessionIndex"].nunique().describe())

missing = df.isna().mean().sort_values(ascending=False).head(10)
print("\nTop missingness columns:\n", missing)