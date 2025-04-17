#Consider the following corpus:

#"India has the second-largest population in the world.",
#" It is surrounded by oceans from three sides which are Bay Of Bengal in
#the east, the Arabian Sea in the west and Indian oceans in the south.",
#"Tiger is the national animal of India.",
#"Peacock is the national bird of India.",
#"Mango is the national fruit of India."     solve in python
#Build a question-answering system and query for "Which is th nationalbird of India?"

#Output:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Corpus
corpus = [
    "India has the second-largest population in the world.",
    "It is surrounded by oceans from three sides which are Bay Of Bengal in the east, the Arabian Sea in the west and Indian oceans in the south.",
    "Tiger is the national animal of India.",
    "Peacock is the national bird of India.",
    "Mango is the national fruit of India."
]

# Step 2: User query
query = "Which is the national animal of India?"

# Step 3: Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus + [query])  # add query as last item

# Step 4: Compute similarity
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# Step 5: Find best match
most_similar_index = cosine_similarities.argmax()
answer = corpus[most_similar_index]

print("Query:", query)
print("Answer:", answer)
