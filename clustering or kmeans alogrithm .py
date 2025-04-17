import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = [
    "Machine learning is great for analyzing data.",
    "Data science involves programming, statistics, and analysis.",
    "Python is a great programming language for data science.",
    "Artificial intelligence is a branch of computer science.",
    "Neural networks are a part of machine learning algorithms.",
    "Clustering algorithms are used to group similar data points together.",
    "Data visualization is a key component of data science.",
    "Deep learning is a subfield of machine learning.",
    "Statistics is important for analyzing and interpreting data."
]

# Step 1: Text Preprocessing (remove stopwords, tokenize, etc.)
stop_words = stopwords.words('english')

# Tokenization and stopwords removal
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Preprocess the documents
processed_docs = [preprocess(doc) for doc in documents]

# Step 2: Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_docs)

# Step 3: Apply K-means Clustering
# Let's set k = 2 for simplicity (You can experiment with different values of k)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Step 4: Cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Step 5: Evaluate the clustering results
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Step 6: Visualize the clustering (Optional)
# We can use the first two dimensions of the document vectors for visualization (PCA or TSNE)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X.toarray())

# Plotting the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="Set1", s=100, alpha=0.7)
plt.title("Document Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Output the document clusters and their corresponding labels
clustered_docs = pd.DataFrame({'Document': documents, 'Cluster Label': labels})
print("\nClustered Documents:")
print(clustered_docs)
