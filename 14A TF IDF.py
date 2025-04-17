import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Corpus of documents
documents = [
    "The sun is the star at the center of the solar system.",
    "She wore a beautiful dress to the party last night.",
    "The book on the table caught my attention immediately."
]

# Query
query = "solar system"

# Step 1: Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Step 2: Combine the documents and the query into one list for fitting the model
all_text = documents + [query]

# Step 3: Fit the model and transform the documents and query into TF-IDF matrices
tfidf_matrix = vectorizer.fit_transform(all_text)

# Step 4: Extract the query vector and document vectors
query_vector = tfidf_matrix[-1]  # The last vector corresponds to the query
document_vectors = tfidf_matrix[:-1]  # The first vectors correspond to the documents

# Step 5: Calculate cosine similarity between the query and each document
cosine_similarities = cosine_similarity(query_vector, document_vectors)

# Step 6: Output the cosine similarity results
for idx, similarity in enumerate(cosine_similarities[0]):
    print(f"Cosine similarity between query and Document {idx+1}: {similarity:.4f}")
