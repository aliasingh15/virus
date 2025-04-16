#Train the ranking model using labeled data and evaluate its effectiveness.import numpy as np

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.utils import shuffle

# Step 1: Create mock dataset
# Format: [query_id, features..., relevance_score]
data = np.array([
    [1, 0.2, 0.1, 0.4, 3],
    [1, 0.3, 0.2, 0.3, 2],
    [1, 0.1, 0.4, 0.3, 1],
    [2, 0.5, 0.4, 0.2, 1],
    [2, 0.6, 0.1, 0.3, 2],
    [2, 0.7, 0.2, 0.5, 3],
])

query_ids = data[:, 0].astype(int)
X = data[:, 1:-1]
y = data[:, -1].astype(int)

# Step 2: Generate pairwise differences
def generate_pairwise_data(X, y, qid):
    X_pairs = []
    y_pairs = []
    for q in np.unique(qid):
        q_idx = np.where(qid == q)[0]
        for i in q_idx:
            for j in q_idx:
                if y[i] > y[j]:
                    X_pairs.append(X[i] - X[j])
                    y_pairs.append(1)
                elif y[i] < y[j]:
                    X_pairs.append(X[j] - X[i])
                    y_pairs.append(0)
    return np.array(X_pairs), np.array(y_pairs)

X_pairs, y_pairs = generate_pairwise_data(X, y, query_ids)

# Step 3: Train RankSVM (pairwise SVM)
X_pairs, y_pairs = shuffle(X_pairs, y_pairs)
scaler = StandardScaler()
X_pairs_scaled = scaler.fit_transform(X_pairs)

model = LinearSVC()
model.fit(X_pairs_scaled, y_pairs)

# Step 4: Rank documents per query using learned weights
def rank_documents(X, qid, model, scaler):
    weights = model.coef_.flatten()
    scores = (X @ weights)
    rankings = {}
    for q in np.unique(qid):
        indices = np.where(qid == q)[0]
        q_scores = scores[indices]
        q_rels = y[indices]
        rankings[q] = (q_scores, q_rels)
    return rankings

rankings = rank_documents(X, query_ids, model, scaler)

# Step 5: Evaluate using MAP and NDCG
def dcg(scores):
    return np.sum([(2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores)])

def ndcg(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    ideal = sorted(y_true, reverse=True)
    return dcg(np.array(y_true)[order]) / dcg(ideal)

map_scores = []
ndcg_scores = []

for q, (scores, rels) in rankings.items():
    order = np.argsort(scores)[::-1]
    y_true = rels
    y_pred = scores

    # Average Precision (for MAP)
    binarized = [1 if r > 1 else 0 for r in y_true]  # Assume rel > 1 is relevant
    map_scores.append(average_precision_score(binarized, y_pred))

    # NDCG
    ndcg_scores.append(ndcg(y_true, y_pred))

print(f"\nMean Average Precision (MAP): {np.mean(map_scores):.4f}")
print(f"Normalized DCG (NDCG): {np.mean(ndcg_scores):.4f}")
