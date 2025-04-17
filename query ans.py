import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK resources (only once)
nltk.download('punkt')

# Corpus of documents
documents = [
    "The cat is on the mat.",
    "The dog is in the yard.",
    "A bird is flying in the sky.",
    "The sun is shining brightly."
]

# Function to find the answer to the query
def answer_query(query, documents):
    query_tokens = word_tokenize(query.lower())  # Tokenize the query and convert to lowercase
    location_keywords = ['where', 'is']  # Keywords that might help in identifying the location query

    # Search for sentences that contain the keyword(s) related to the query
    for doc in documents:
        sentences = sent_tokenize(doc)
        for sentence in sentences:
            # Tokenize the sentence and convert to lowercase
            sentence_tokens = word_tokenize(sentence.lower())
            
            # Check if the query contains the word "cat" and if the sentence contains location-related keywords
            if 'cat' in query_tokens and any(keyword in sentence_tokens for keyword in location_keywords):
                # Here, we'll return the sentence that contains the location of the cat
                return sentence
    return "Sorry, I couldn't find the answer to your query."

# Example query
query = "Where is the cat?"
answer = answer_query(query, documents)

# Output the result
print(f"Query: {query}")
print(f"Answer: {answer}")
