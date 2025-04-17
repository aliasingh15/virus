#Implement a clustering algorithm (e.g., K-means or hierarchicalclustering). Also, apply the clustering algorithm to a set of documentsand evaluate the clustering results.

#output:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Sample document set
documents = [
    "The sun is shining and the weather is sweet.",
    "It is a bright sunny day today.",
    "We should go for a walk in the sun.",
    "The stock market crashed and investors are worried.",
    "Investing in stocks can be risky.",
    "The economy is in recession and jobs are at risk.",
    "Football is a great sport played worldwide.",
    "Many people watch the football world cup.",
    "Cristiano Ronaldo is a famous football player."
]

# Step 1: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 2: Apply K-means clustering
k = 3  # Try 3 clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)
labels = model.labels_

# Step 3: Evaluate using Silhouette Score
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")

# Step 4: Show document clusters
print("\n--- Document Clusters ---")
for i, doc in enumerate(documents):
    print(f"Cluster {labels[i]}: {doc}")

# Optional: Visualize clusters in 2D using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(8, 5))
for i in range(k):
    plt.scatter(
        reduced[labels == i, 0], reduced[labels == i, 1],
        label=f'Cluster {i}'
    )
plt.legend()
plt.title("Document Clusters (PCA Visualization)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


