from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Given data
y_true = [0, 1, 1, 0, 1]
y_scores = [0.1, 0.8, 0.6, 0.3, 0.9]

# Convert scores to binary predictions (threshold = 0.5)
y_pred = [1 if score >= 0.5 else 0 for score in y_scores]

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
average_precision = average_precision_score(y_true, y_scores)

# Display the results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Average Precision (AP): {average_precision:.4f}")
