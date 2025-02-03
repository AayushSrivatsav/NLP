#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:27:55 2025

@author: aayushsrivatsav
"""

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
