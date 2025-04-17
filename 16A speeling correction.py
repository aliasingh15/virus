# Step 1: Dictionary of valid words
dictionary = ["spelling", "correction", "algorithm", "information", "retrieval", "system"]

# Step 2: Define the edit distance (Levenshtein distance) function
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j  # Insert all j characters
            elif j == 0:
                dp[i][j] = i  # Remove all i characters
            elif word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No change needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    return dp[m][n]

# Step 3: Spell correction using the dictionary
def correct_spelling(word, dictionary):
    distances = [(dict_word, edit_distance(word, dict_word)) for dict_word in dictionary]
    best_match = min(distances, key=lambda x: x[1])
    return best_match  # Returns (corrected_word, distance)

# Step 4: IR system that uses the corrected spelling
documents = {
    1: "This system provides accurate information.",
    2: "Spelling correction is important in a retrieval system.",
    3: "The algorithm handles user queries with efficiency."
}

def retrieve_documents(query, dictionary, documents):
    corrected_word, distance = correct_spelling(query, dictionary)
    print(f"Did you mean: '{corrected_word}' (edit distance = {distance})\n")
    
    results = {}
    for doc_id, content in documents.items():
        if corrected_word.lower() in content.lower():
            results[doc_id] = content
    return results

# Step 5: Run the query
query_word = "infmoration"
results = retrieve_documents(query_word, dictionary, documents)

# Step 6: Display the results
print("Search Results:")
for doc_id, content in results.items():
    print(f"Doc {doc_id}: {content}")
