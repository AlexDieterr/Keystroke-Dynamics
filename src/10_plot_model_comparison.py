import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)



# Model accuracies from prior analysis
models = ["Logistic Regression", "Random Forest", "XGBoost"]
accuracies = [0.78, 0.856, 0.857]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Identification Accuracy by Model")

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()