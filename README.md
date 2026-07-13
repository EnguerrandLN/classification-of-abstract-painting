# Unsupervised Classification of Abstract Art

Welcome to the projet-im06 repository. This project explores the unsupervised classification of abstract paintings using a multimodal Artificial Intelligence pipeline. 

Abstract art lacks semantic objects (like faces or landscapes) that standard computer vision models rely on. To solve this, our pipeline extracts pure stylistic signatures by fusing three distinct feature spaces: Classical Geometric Descriptors, Variational Autoencoders (VAE), and Convolutional Neural Networks (CNN).

## My Core Contributions
While this was a collaborative academic project, I was personally responsible for the computer vision feature extraction, the algorithmic clustering refinement, and the data engineering. My specific contributions include:

* **CNN Feature Extraction:** Designed and implemented a custom pipeline using VGG16 to extract texture-based style signatures via Gram matrices, deliberately bypassing semantic object recognition.
* **Multimodal Fusion:** Engineered the weighted alpha-blending logic to normalize, standardize, and fuse the CNN, VAE, and geometric feature spaces into a cohesive mathematical topology.
* **Adaptive Skimming (Core Silhouette):** Developed a custom statistical filtering algorithm that purifies K-Means clusters by dynamically calculating exclusion thresholds based on the median Silhouette score of each cluster's "core."
* **Data Visualization & Pipeline Engineering:** Wrote the topological validation scripts (Inter-Artist cosine matrices, stylistic heatmaps) and built the Python pipelines to clean, format, and physically sort the dataset for the final WebGL application.

## Repository Architecture

The project has been refactored into a clean, modular structure.

### 1. cnn/ (CNN Models & Features)
*Contains the computer vision extraction pipeline.*
* **cnn.ipynb**: Extracts high-level stylistic features (Gram matrices across 4 VGG16 layers, averaged over 10 random crops per image to ensure spatial robustness).
* **cnn_features_pca.npy**: The resulting PCA-reduced feature matrix. *(Note: Raw ~1.5GB tensors are excluded for repository performance).*

### 2. classification/ (Fusion, Clustering & Final Results)
*Contains the clustering engine and statistical validation logic.*
* **melange_features_CNN.ipynb**: The core script performing multimodal fusion, K-Means clustering, and the Adaptive Skimming algorithm.
* **classification_finale_oeuvres.csv** & **results_clusters.csv**: The final sanitized data outputs formatting the clusters.
* **lavraieclassification.png**: The stylometric heatmap validating the final clustering.
* **classification_finale/**: A directory where the dataset is physically sorted into AI-determined movements (C0-C19 for main clusters, A0-A9 for re-clustered marginals).

### 3. features/ (Geometric & Raw Extraction)
* **features.ipynb**: Computes classical geometric and low-level computer vision features (color palettes, edge detection).
* **features.pkl** & **features_brutes.pkl**: Pre-computed geometric outputs.

### 4. latent_space/ (Latent Space & Autoencoders)
* **latent-space-vae.ipynb**: Trains a Variational Autoencoder (VAE) to capture structural and chromatic variations.
* **latent_vectors.npy** & **fused_vectors.npy**: The resulting latent space representations.

### 5. Root Directories
* **abstrait-v4/**: The complete, self-contained abstract art dataset.
* **visualisation et data/**: Contains the code and assets for our custom interactive WebGL (deck.gl) 2D map. Read the included tutorial file to run it locally.
* **rapports/** & **papers/**: Research papers, state-of-the-art literature, and our final academic report.

## How to Use

This repository is plug-and-play. You can clone the project and immediately explore the notebooks. The dataset is fully included, and the heavy CNN processing has been cached into lightweight `.npy` and `.pkl` files, allowing you to run the clustering and fusion engines (`classification/melange_features_CNN.ipynb`) out-of-the-box.

---
*Developed for the IM06 Project — Télécom Paris.*
