import os
import pandas as pd
import matplotlib.pyplot as plt

# Path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Per-user accuracy values from analysis
per_user_accuracy = {
    "s011": 0.27, "s020": 0.39, "s054": 0.50, "s032": 0.60, "s029": 0.62,
    "s018": 0.62, "s041": 0.63, "s047": 0.64, "s003": 0.66, "s002": 0.69,
    "s027": 0.95, "s055": 0.96, "s051": 0.96, "s052": 0.97, "s022": 0.97,
    "s053": 0.98, "s043": 0.98, "s049": 0.98, "s004": 0.98, "s036": 1.00
}

acc_df = pd.DataFrame.from_dict(
    per_user_accuracy, orient="index", columns=["accuracy"]
)

# Plot
plt.figure(figsize=(6, 4))
plt.hist(acc_df["accuracy"], bins=8, edgecolor="black")

plt.xlabel("Per-user identification accuracy")
plt.ylabel("Number of users")
plt.title("Distribution of Identification Accuracy Across Users")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "per_user_accuracy_distribution.png"),
            dpi=300, bbox_inches="tight")
plt.close()