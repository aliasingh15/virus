#Calculate the cosine similarity between the query and each document.


import math

# 1. Documents and Query
documents = [
    "The sun is the star at the center of the solar system.",
    "She wore a beautiful dress to the party last night.",
    "The book on the table caught my attention immediately."
]
query = "solar system"

# 2. Tokenization
def tokenize(text):
    return [word.lower().strip(".,") for word in text.split()]

tokenized_docs = [tokenize(doc) for doc in documents]
tokenized_query = tokenize(query)

# 3. Vocabulary
vocab = sorted(set(word for doc in tokenized_docs for word in doc))
vocab_index = {word: i for i, word in enumerate(vocab)}

# 4. Term Frequency (TF)
def term_frequency(tokens):
    tf = [0] * len(vocab)
    for word in tokens:
        if word in vocab_index:
            tf[vocab_index[word]] += 1
    return tf

tf_docs = [term_frequency(doc) for doc in tokenized_docs]
tf_query = term_frequency(tokenized_query)

# 5. Inverse Document Frequency (IDF)
def compute_idf(docs):
    N = len(docs)
    idf = [0] * len(vocab)
    for i, word in enumerate(vocab):
        df = sum(1 for doc in docs if word in doc)
        idf[i] = math.log((N + 1) / (df + 1)) + 1  # smoothing
    return idf

idf = compute_idf(tokenized_docs)

# 6. TF-IDF Calculation
def compute_tfidf(tf, idf):
    return [tf[i] * idf[i] for i in range(len(tf))]

tfidf_docs = [compute_tfidf(tf, idf) for tf in tf_docs]
tfidf_query = compute_tfidf(tf_query, idf)

# 7. Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# 8. Calculate and Print Cosine Similarity
print("Cosine Similarity between query and each document:")
for i, vec in enumerate(tfidf_docs):
    sim = cosine_similarity(vec, tfidf_query)
    print(f"Document {i + 1}: {sim:.4f}")
