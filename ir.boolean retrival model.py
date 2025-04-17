import re
from collections import defaultdict

# Step 1: Corpus
documents = {
    1: 'this is the first document.',
    2: 'this document is the second document.',
    3: 'And this is the third one.',
    4: 'Is this the first document?'
}

# Step 2: Preprocess and build the inverted index
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    return text.split()

inverted_index = defaultdict(set)

for doc_id, text in documents.items():
    words = preprocess(text)
    for word in words:
        inverted_index[word].add(doc_id)

# Step 3: Boolean query evaluation
def boolean_and(set1, set2):
    return set1 & set2

# Query: "first and third"
term1 = 'first'
term2 = 'third'

result = boolean_and(inverted_index[term1], inverted_index[term2])

print("Documents matching query 'first and third':", sorted(result))
