from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import numpy as np

# Simulated ground truth (1 = positive class, 0 = negative class)
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])

# Simulated model scores (confidence or probability for class 1)
y_scores = np.array([0.1, 0.9, 0.8, 0.4, 0.95, 0.2, 0.85, 0.05, 0.3, 0.99])

# Convert scores to binary predictions using a threshold (e.g. 0.5)
y_pred = (y_scores >= 0.5).astype(int)

# Evaluation metrics
ap = average_precision_score(y_true, y_scores)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_scores)

# Display results
print(f"Average Precision (AP): {ap:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
