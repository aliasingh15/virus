#Develop a spelling correction module using edit distance algorithms andfind the edit distance between strings “write” and “right”.

#output:
#distance between write and right:
import numpy as np

def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j  # Insert all
            elif j == 0:
                dp[i][j] = i  # Remove all
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Remove
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    return dp[m][n]

# Example
word1 = "write"
word2 = "right"
distance = edit_distance(word1, word2)
print(f"Edit distance between '{word1}' and '{word2}': {distance}")

#spelling correction:

def correct_word(word, dictionary):
    distances = [(dict_word, edit_distance(word, dict_word)) for dict_word in dictionary]
    closest_match = min(distances, key=lambda x: x[1])
    return closest_match[0], closest_match[1]

# Sample dictionary
dictionary = ["right", "write", "white", "rate", "riot", "night", "kite"]

# Input misspelled word
misspelled_word = "rite"
suggestion, dist = correct_word(misspelled_word, dictionary)

print(f"Suggested correction for '{misspelled_word}': '{suggestion}' (Edit distance: {dist})")


