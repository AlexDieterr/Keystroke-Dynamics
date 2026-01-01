import os
import pandas as pd
import matplotlib.pyplot as plt

# Path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Separability and accuracy values from analysis
data = {
    "s049": (0.98, 0.91),
    "s047": (0.64, 1.04),
    "s032": (0.60, 1.05),
    "s046": (0.93, 1.07),
    "s020": (0.39, 1.07),
    "s024": (0.90, 2.10),
    "s055": (0.96, 2.13),
    "s010": (0.90, 2.30),
    "s028": (0.88, 2.41),
    "s017": (0.92, 3.15),
}

df = pd.DataFrame.from_dict(
    data, orient="index", columns=["accuracy", "separability"]
)

import numpy as np

# Plot points
plt.figure(figsize=(6, 4))
plt.scatter(df["separability"], df["accuracy"])

# Trend line (simple linear fit)
x = df["separability"].values
y = df["accuracy"].values
coef = np.polyfit(x, y, 1)
trend = np.poly1d(coef)

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, trend(x_line), linestyle="--")

plt.xlabel("Separability")
plt.ylabel("Identification accuracy")
plt.title("Separability vs Identification Accuracy")

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "separability_vs_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()