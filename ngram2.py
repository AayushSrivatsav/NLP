#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:37:10 2025

@author: aayushsrivatsav
"""

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
