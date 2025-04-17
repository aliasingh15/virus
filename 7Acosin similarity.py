import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input query and document
query = "python programming"
document = "Document about python programming language and data analysis."

# Step 1: Preprocessing (you can extend this step by removing stopwords, stemming, etc.)
# In this case, we directly use the texts.

# Step 2: TF-IDF Vectorization
corpus = [query, document]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Step 3: Cosine Similarity Calculation
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Output the cosine similarity
print(f"Cosine Similarity between query and document: {cosine_sim[0][0]}")
