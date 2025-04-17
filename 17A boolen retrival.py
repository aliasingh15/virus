import re
from collections import defaultdict

# Step 1: Documents
documents = {
    1: 'this is the first document.',
    2: 'this document is the second document.',
    3: 'And this is the third one.',
    4: 'Is this the first document?'
}

# Step 2: Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()

# Step 3: Build inverted index
inverted_index = defaultdict(set)

for doc_id, text in documents.items():
    words = preprocess(text)
    for word in words:
        inverted_index[word].add(doc_id)

# Step 4: Boolean AND query
def boolean_and_query(term1, term2):
    docs1 = inverted_index.get(term1, set())
    docs2 = inverted_index.get(term2, set())
    return docs1 & docs2  # Intersection

# Step 5: Run query
query_terms = ["first", "third"]
result_docs = boolean_and_query(query_terms[0], query_terms[1])

# Step 6: Display results
print(f"Documents matching query 'first AND third': {result_docs}")
