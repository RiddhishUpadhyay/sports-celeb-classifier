import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------------
# Load Evaluation Results
# ----------------------------------

df = pd.read_csv("evaluation_results_35.csv")

y_true = df["ground_truth"]
y_pred = df["prediction"]

# ----------------------------------
# Metrics
# ----------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("=" * 50)
print("Evaluation Metrics")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print("=" * 50)

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

# ----------------------------------
# Confusion Matrix
# ----------------------------------

labels = sorted(
    list(
        set(y_true) | set(y_pred)
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45
)

# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()