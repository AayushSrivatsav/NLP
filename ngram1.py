#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:17:05 2025

@author: aayushsrivatsav
"""

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
