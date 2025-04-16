#Implement a learning to rank algorithm (e.g., RankSVM or RankBoost)on a given dataset.


import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Step 1: Sample dataset (3 documents with features, relevance given)
X = np.array([
    [0.1, 0.2, 0.3],  # doc1
    [0.4, 0.2, 0.1],  # doc2
    [0.2, 0.1, 0.4],  # doc3
])
y = np.array([2, 1, 3])  # relevance scores (higher is more relevant)

# Step 2: Generate pairwise data (Xi - Xj and label = sign(yi - yj))
X_pairs = []
y_pairs = []

for i in range(len(X)):
    for j in range(len(X)):
        if y[i] > y[j]:
            X_pairs.append(X[i] - X[j])
            y_pairs.append(1)
        elif y[i] < y[j]:
            X_pairs.append(X[j] - X[i])
            y_pairs.append(0)

X_pairs = np.array(X_pairs)
y_pairs = np.array(y_pairs)

# Step 3: Train RankSVM model using LinearSVC
scaler = StandardScaler()
X_pairs = scaler.fit_transform(X_pairs)

model = LinearSVC()
model.fit(X_pairs, y_pairs)

# Step 4: Use model to rank new documents
# We'll score documents by dot product of weight vector and features
weights = model.coef_.flatten()
scores = X @ weights  # score = w·x

# Step 5: Sort by score (higher is better)
ranked_indices = np.argsort(scores)[::-1]
print("Ranking of documents (best to worst):", ranked_indices + 1)  # +1 for doc numbers
print("Scores:", scores)
