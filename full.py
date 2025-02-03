#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:24:14 2025

@author: aayushsrivatsav
"""

#svdtext

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import string

# Preprocess text: lowercase, remove punctuation, and stopwords
def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = text.lower()  # Lowercase
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        text = " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stop words
        processed_texts.append(text)
    return processed_texts

# Construct TF-IDF Matrix
def construct_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Construct Co-occurrence Matrix
def construct_co_occurrence_matrix(texts, window_size=2):
    word_list = [text.split() for text in texts]
    co_occurrence = Counter()

    for words in word_list:
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1

    # Create a list of all unique words
    all_words = list(set([word for words in word_list for word in words]))

    # Create co-occurrence matrix as a 2D array
    co_occurrence_matrix = np.zeros((len(all_words), len(all_words)))
    word_to_index = {word: idx for idx, word in enumerate(all_words)}

    for (word1, word2), count in co_occurrence.items():
        i = word_to_index[word1]
        j = word_to_index[word2]
        co_occurrence_matrix[i, j] = count
        co_occurrence_matrix[j, i] = count  # Co-occurrence is symmetric

    return co_occurrence_matrix, all_words

# Construct PMI Matrix
def construct_pmi_matrix(texts, window_size=2):
    word_list = [text.split() for text in texts]
    word_freq = Counter(word for words in word_list for word in words)
    total_words = sum(word_freq.values())

    # Co-occurrence Matrix Calculation
    co_occurrence = Counter()
    for words in word_list:
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1

    # PMI Calculation
    pmi_matrix = {}
    for (word1, word2), count in co_occurrence.items():
        p_word1 = word_freq[word1] / total_words
        p_word2 = word_freq[word2] / total_words
        p_word1_word2 = count / total_words
        pmi = np.log(p_word1_word2 / (p_word1 * p_word2)) if p_word1_word2 > 0 else 0

        if word1 not in pmi_matrix:
            pmi_matrix[word1] = {}
        pmi_matrix[word1][word2] = pmi

    all_words = list(set(word_freq.keys()))
    return pmi_matrix, all_words

# Manually perform SVD
def manual_svd(A, k):
    # Compute A^T A and A A^T
    AtA = np.dot(A.T, A)  # Right singular vectors
    AAt = np.dot(A, A.T)  # Left singular vectors

    # Eigenvalue decomposition of A^T A
    eig_vals_V, V = np.linalg.eig(AtA)
    # Eigenvalue decomposition of A A^T
    eig_vals_U, U = np.linalg.eig(AAt)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices_V = np.argsort(eig_vals_V)[::-1]
    sorted_indices_U = np.argsort(eig_vals_U)[::-1]

    V = V[:, sorted_indices_V]
    U = U[:, sorted_indices_U]

    # Singular values (sqrt of eigenvalues)
    singular_values = np.sqrt(np.abs(eig_vals_V[sorted_indices_V]))

    # Create Sigma matrix (with top k singular values)
    Sigma = np.zeros((k, k))
    np.fill_diagonal(Sigma, singular_values[:k])

    return U[:, :k], Sigma, V.T[:k, :]

# Example usage
def main():
    # Example dataset
    texts = [
        "This is a sample document about machine learning.",
        "Another document discussing data science.",
        "Text processing and natural language.",
        "Machine learning and artificial intelligence."
    ]
    #IF CSV IS GIVEN
    # df=pd.read_csv("/content/emails.csv")
    # df=(df.head(15)) # for sample taking only first 15 rows

    # texts=df['text'].tolist()

    # Preprocess the texts
    processed_texts = preprocess_text(texts)

    # 1. Construct TF-IDF Matrix
    tfidf_matrix, vocab_tfidf = construct_tfidf_matrix(processed_texts)
    print("\nTF-IDF Matrix:")
    print(tfidf_matrix)

    # Apply SVD on TF-IDF Matrix
    k = 2  # Number of topics (components) to keep
    U_tfidf, Sigma_tfidf, VT_tfidf = manual_svd(tfidf_matrix, k)

    print("\nSVD Results (TF-IDF Matrix):")
    print("U matrix (document-topic vectors):", U_tfidf)
    print("Sigma matrix (importance of each topic):", Sigma_tfidf)
    print("VT matrix (word-topic vectors):", VT_tfidf)

    # 2. Construct Co-occurrence Matrix
    co_occurrence_matrix, vocab_cooc = construct_co_occurrence_matrix(processed_texts)
    print("\nCo-occurrence Matrix:")
    print(co_occurrence_matrix)

    # Apply SVD on Co-occurrence Matrix
    U_cooc, Sigma_cooc, VT_cooc = manual_svd(co_occurrence_matrix, k)

    print("\nSVD Results (Co-occurrence Matrix):")
    print("U matrix (document-topic vectors):", U_cooc)
    print("Sigma matrix (importance of each topic):", Sigma_cooc)
    print("VT matrix (word-topic vectors):", VT_cooc)

    # 3. Construct PMI Matrix
    pmi_matrix, vocab_pmi = construct_pmi_matrix(processed_texts)
    pmi_matrix_array = np.zeros((len(vocab_pmi), len(vocab_pmi)))

    # Convert PMI matrix to a numpy array
    word_to_index = {word: idx for idx, word in enumerate(vocab_pmi)}
    for word1 in pmi_matrix:
        for word2 in pmi_matrix[word1]:
            i = word_to_index[word1]
            j = word_to_index[word2]
            pmi_matrix_array[i, j] = pmi_matrix[word1][word2]

    print("\nPMI Matrix:")
    print(pmi_matrix_array)

    # Apply SVD on PMI Matrix
    U_pmi, Sigma_pmi, VT_pmi = manual_svd(pmi_matrix_array, k)

    print("\nSVD Results (PMI Matrix):")
    print("U matrix (document-topic vectors):", U_pmi)
    print("Sigma matrix (importance of each topic):", Sigma_pmi)
    print("VT matrix (word-topic vectors):", VT_pmi)

main()


#NEXT WORD PREDICTION AND TEXT GENERATION
import numpy as np
import pandas as pd
import string
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.util import bigrams, trigrams
from nltk.util import ngrams

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # Download the punkt_tab data package

# Load dataset
data = pd.read_csv('/content/emails.csv')

print(data.head(5))

# Initialize tokenizer and lemmatizer
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = tokenizer.tokenize(text)  # Tokenize the text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return words

# Apply preprocessing to the text data
data['clean_text'] = data['text'].apply(lambda x:' '.join(preprocess_text(x)))

# Train n-gram model with Laplacian smoothing
def train_ngram_model(texts, n=2):
    """Train an n-gram model with Laplacian smoothing."""
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)
    vocabulary = set()

    for text in texts:
        tokens = preprocess_text(text)
        vocabulary.update(tokens)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngram_counts[ngram] += 1
            context_counts[ngram[:-1]] += 1

    return ngram_counts, context_counts, len(vocabulary)

# Predict next word using the n-gram model with Laplacian smoothing
def predict_next_word(text, ngram_counts, context_counts, vocab_size, n=2, prev_words=set()):
    """Predict the next word using Laplacian smoothing while avoiding excessive repetition."""
    tokens = preprocess_text(text)

    if len(tokens) < n - 1:
        return None, 0.0

    context = tuple(tokens[-(n - 1):])  # Get the context (last n-1 words)

    word_probs = {}
    total_count = context_counts[context] + vocab_size  # Laplacian smoothing (adding vocab size to denominator)

    for (ngram, count) in ngram_counts.items():
        if ngram[:-1] == context and ngram[-1] not in prev_words:
            word_probs[ngram[-1]] = (count + 1) / total_count  # Laplacian smoothing (adding 1 to numerator)

    if word_probs:
        best_word = max(word_probs, key=word_probs.get)
        return best_word, word_probs[best_word]  # Return the word with highest probability and its probability

    return None, 0.0

# Generate a sequence of words based on an initial text
def generate_text(initial_text, ngram_counts, context_counts, vocab_size, n=2, num_words=5):
    """Generate the next sequence of words while reducing repetition."""
    result = initial_text
    prev_words = set(initial_text.split())  # Track previously used words to avoid repetition

    for _ in range(num_words):
        next_word, _ = predict_next_word(result, ngram_counts, context_counts, vocab_size, n, prev_words)
        if not next_word:
            break  # Stop if no word is predicted
        result += " " + next_word
        prev_words.add(next_word)  # Add new word to the set of used words

    return result

# Test predictions and text generation
test_texts = ['subject really']

# Train n-gram model (you can change n to 2 for bigram or 3 for trigram)
n = 2  # bigram model
ngram_counts, context_counts, vocab_size = train_ngram_model(data['clean_text'], n)

for test_text in test_texts:
    predicted_word, probability = predict_next_word(test_text, ngram_counts, context_counts, vocab_size, n)
    print(f"Bigram Test - Input: '{test_text}' → Predicted next word: '{predicted_word}' with probability: {probability:.4f}")
    print(f"Generated text: '{generate_text(test_text, ngram_counts, context_counts, vocab_size, n, 10)}'")

# Train n-gram model (you can change n to 2 for bigram or 3 for trigram)
n = 3  # Trigram model
ngram_counts, context_counts, vocab_size = train_ngram_model(data['clean_text'], n)

for test_text in test_texts:
    predicted_word, probability = predict_next_word(test_text, ngram_counts, context_counts, vocab_size, n)
    print(f"Trigram Test - Input: '{test_text}' → Predicted next word: '{predicted_word}' with probability: {probability:.4f}")
    print(f"Generated text: '{generate_text(test_text, ngram_counts, context_counts, vocab_size, n, 10)}'")



from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Function to calculate n-gram probabilities
def calculate_ngram_probabilities(text, n):
    tokens = word_tokenize(text.lower())
    ngram_counts = Counter(ngrams(tokens, n))
    total_ngrams = sum(ngram_counts.values())
    ngram_frequencies = {}
    for ngram, count in ngram_counts.items():
        ngram_frequencies[ngram] = count / total_ngrams
    return ngram_frequencies

# Classify Text based on n-gram Models
def classify_text_ngram(text, spam_ngrams, ham_ngrams, n):
    text_ngrams = calculate_ngram_probabilities(text, n)
    spam_score = sum(text_ngrams.get(ngram, 0) * spam_ngrams.get(ngram, 0) for ngram in text_ngrams)
    ham_score = sum(text_ngrams.get(ngram, 0) * ham_ngrams.get(ngram, 0) for ngram in text_ngrams)
    return "Spam" if spam_score > ham_score else "Ham"

# Train n-gram Models for Spam and Ham
def train_ngram_model(emails, n):
    ngram_counts = Counter()
    for email in emails:
        ngram_counts.update(calculate_ngram_probabilities(email, n))
    total_ngrams = sum(ngram_counts.values())
    ngram_frequencies = {}
    for ngram, count in ngram_counts.items():
        ngram_frequencies[ngram] = count / total_ngrams
    return ngram_frequencies

# Example usage for training and classification
def train_and_classify(data, n):
    spam_ngrams = train_ngram_model(data[data['spam'] == 1]['text'], n)
    ham_ngrams = train_ngram_model(data[data['spam'] == 0]['text'], n)

    test_texts = [
        "Can you send me the latest project report?",
        "Meeting scheduled for 3 PM",
        "Urgent! Your bank account has been compromised. Verify your details immediately.",
        "Exciting Offers! Click here to win an iphone"
    ]

    for text in test_texts:
        print(f"{n}-gram Classification: The text '{text}' is classified as: {classify_text_ngram(text, spam_ngrams, ham_ngrams, n)}")
        print("\n")

# Example for bigram (n=2) classification
train_and_classify(data, 2)

# Example for trigram (n=3) classification
train_and_classify(data, 3)



import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image (RGB)
image = cv2.imread('/content/test_cat.png')  # OpenCV loads in BGR format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = image.astype(np.float32)  # Convert to float

# Function to manually compute SVD for each channel
def manual_svd(A, k):
    # Compute A^T A and A A^T
    # Compute A^T A and A A^T
    AtA = np.dot(A.T, A)  # Right singular vectors
    AAt = np.dot(A, A.T)  # Left singular vectors

    # Eigenvalue decomposition of A^T A
    eig_vals_V, V = np.linalg.eig(AtA)
    # Eigenvalue decomposition of A A^T
    eig_vals_U, U = np.linalg.eig(AAt)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices_V = np.argsort(eig_vals_V)[::-1]
    sorted_indices_U = np.argsort(eig_vals_U)[::-1]

    V = V[:, sorted_indices_V]
    U = U[:, sorted_indices_U]

    # Singular values (sqrt of eigenvalues)
    singular_values = np.sqrt(np.abs(eig_vals_V[sorted_indices_V]))

    # Create Sigma matrix (with top k singular values)
    Sigma = np.zeros((k, k))
    np.fill_diagonal(Sigma, singular_values[:k])

    # Reconstruct compressed image channel
    compressed_A = np.dot(U[:, :k], np.dot(Sigma, V.T[:k, :]))
    return compressed_A

# Set k (number of singular values to keep)
k = 10 #change as per our requirement

# Apply SVD to each color channel separately
compressed_R = manual_svd(image[:, :, 0], k)
compressed_G = manual_svd(image[:, :, 1], k)
compressed_B = manual_svd(image[:, :, 2], k)

# Stack channels back into an RGB image
compressed_image = np.stack([compressed_R, compressed_G, compressed_B], axis=2)

# Normalize pixel values to 0-255 (avoid overflow)
compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

# Display original and compressed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image.astype(np.uint8))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image (k={k})")
plt.imshow(compressed_image)
plt.axis('off')

plt.show()
