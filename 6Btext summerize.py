import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import heapq

# Download necessary resources (if not already available)
nltk.download('punkt')
nltk.download('stopwords')

# Input text
text = """
Natural language processing (NLP) is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human (natural) languages. As such, NLP is related to the area of human–computer interaction. Many challenges in NLP involve natural language understanding, natural language generation, and machine learning.

Text summarization is the process of distilling the most important information from a source (text) to produce an abridged version for a particular user or task. Automatic text summarization methods are greatly needed to address the ever-growing amount of text data available online to both better help discover relevant information and to consume the vast amount of text data available more efficiently.
"""

# Sentence tokenization
sentences = sent_tokenize(text)

# Word frequency table
stop_words = set(stopwords.words('english'))
word_frequencies = {}

for word in word_tokenize(text.lower()):
    if word.isalnum() and word not in stop_words:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1

# Normalize frequencies (optional)
max_freq = max(word_frequencies.values())
for word in word_frequencies:
    word_frequencies[word] /= max_freq

# Score sentences based on the sum of the frequencies of words they contain
sentence_scores = {}
for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in word_frequencies:
            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

# Select the top 2 sentences with the highest scores as summary
summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)

# Output
print("📜 Original Text:\n", text.strip(), "\n")
print("📝 Summarized Text:\n", summary)
