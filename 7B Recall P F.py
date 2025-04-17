# Given values
TP = 20  # True Positives
FP = 10  # False Positives
FN = 30  # False Negatives

# Calculating Precision
Precision = TP / (TP + FP)

# Calculating Recall
Recall = TP / (TP + FN)

# Calculating F-score (F-measure)
F_score = 2 * (Precision * Recall) / (Precision + Recall)

# Output the results
print(f"Precision: {Precision:.2f}")
print(f"Recall: {Recall:.2f}")
print(f"F-measure: {F_score:.2f}")
