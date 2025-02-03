#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:18:16 2025

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
