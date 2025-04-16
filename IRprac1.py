import math
from collections import defaultdict

# Define the corpus
documents = [
    "The sun is the star at the center of the solar system.",
    "She wore a beautiful dress to the party last night.",
    "The book on the table caught my attention immediately."
]

query = "solar system"

# Step 1: Preprocessing - tokenize and lowercase
def preprocess(text):
    return [word.lower().strip(".,") for word in text.split()]

# Step 2: Compute term frequencies (TF)
def compute_tf(doc):
    tf = defaultdict(int)
    for word in doc:
        tf[word] += 1
    return tf

# Step 3: Compute inverse document frequency (IDF)
def compute_idf(docs):
    idf = {}
    N = len(docs)
    all_tokens = set(token for doc in docs for token in doc)
    
    for token in all_tokens:
        df = sum(1 for doc in docs if token in doc)
        idf[token] = math.log(N / (df + 1)) + 1  # smooth IDF
    return idf

# Step 4: Compute TF-IDF vectors
def compute_tfidf(tf, idf):
    return {term: freq * idf[term] for term, freq in tf.items()}

# Step 5: Cosine similarity
def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in common)
    
    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    return numerator / denominator if denominator else 0.0

# === Main process ===

# Preprocess documents and query
processed_docs = [preprocess(doc) for doc in documents]
processed_query = preprocess(query)

# Compute TF for docs and query
tf_docs = [compute_tf(doc) for doc in processed_docs]
tf_query = compute_tf(processed_query)

# Compute IDF using all documents
idf = compute_idf(processed_docs + [processed_query])  # include query for fair IDF

# Compute TF-IDF vectors
tfidf_docs = [compute_tfidf(tf, idf) for tf in tf_docs]
tfidf_query = compute_tfidf(tf_query, idf)

# Compute cosine similarity
print("Cosine Similarity Scores:")
for i, vec in enumerate(tfidf_docs):
    sim = cosine_similarity(vec, tfidf_query)
    print(f"Document {i+1}: {sim:.4f}")
