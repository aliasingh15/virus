# Given values
TP = 60  # True Positives
FP = 30  # False Positives
FN = 20  # False Negatives

# Calculate Precision
precision = TP / (TP + FP)

# Calculate Recall
recall = TP / (TP + FN)

# Calculate F-Measure (F1 Score)
f_measure = 2 * (precision * recall) / (precision + recall)

# Print the results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-Measure (F1 Score): {f_measure:.4f}")
