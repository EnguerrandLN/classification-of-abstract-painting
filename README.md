# Unsupervised Classification of Abstract Art

This repository contains the complete pipeline for the unsupervised classification of an abstract painting corpus (1,700 works). The project focuses on organizing abstract art by purely visual properties, without relying on artist metadata, historical labels, or biographical context.

## Project Overview

Abstract art presents a unique challenge for computer vision because it lacks the semantic objects (faces, landscapes, etc.) that characterize figurative datasets. This project addresses this by fusing three complementary approaches:
1. **Geometric Descriptors**: Extracting low-level features like color distribution, texture patterns, and compositional structure.
2. **Variational Autoencoders (VAE)**: Learning a continuous latent representation of stylistic variations.
3. **Convolutional Neural Networks (CNN)**: Utilizing pre-trained VGG16 models and Gram matrices to capture complex texture and stylistic signatures independent of spatial location.



## Repository Architecture

The project is structured into modular components:

* **`cnn/`**: Contains the computer vision extraction pipeline. Includes VGG16-based style extraction using Gram matrices and PCA-reduced feature matrices.
* **`classification/`**: The core clustering engine. Includes multimodal fusion, K-Means implementation, Adaptive Skimming (Core Silhouette) for cluster purification, and final validation metrics.
* **`features/`**: Computes classical geometric and low-level computer vision features (color palettes, edge detection).
* **`latent_space/`**: Training and implementation of the $\beta$-VAE for structural and chromatic latent representations.
* **`abstrait-v4/`**: The self-contained abstract art dataset.
* **`visualisation et data/`**: Contains the assets and code for the custom interactive WebGL visualization tool.
* **`rapports/` & `papers/`**: Research documentation and the final academic report.

## Methodology

The pipeline follows a structured three-stage approach:

1.  **Feature Extraction**: Harmonization of three disparate feature spaces (geometric, latent VAE, and CNN-Gram) through L2 normalization and standard scaling.
2.  **Multimodal Fusion**: Concatenation of the standardized feature spaces into a cohesive high-dimensional "super-vector."
3.  **Refined Clustering**:
    * **K-Means Partitioning**: Initial discretization of the latent space.
    * **Adaptive Skimming**: A custom statistical filtering algorithm that purifies heterogeneous clusters. It calculates an exclusion threshold ($\tau_k$) based on the median Silhouette score of each cluster's "core," reassigning outliers to a marginal set ($C_{-1}$) for recursive sub-clustering.



## Final Results

The final classification is available in the `classification_finale/` directory. The dataset has been physically organized into folders corresponding to the AI-identified movements:
* **C0–C19**: Main homogeneous stylistic clusters.
* **A0–A9**: Sub-clusters of marginal artworks isolated through the recursive skimming process.

## Requirements

* **Language**: Python 3.x
* **Libraries**: `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Data**: The `abstrait-v4` dataset is provided within the repository.

Developed for the IM06 Project - Télécom Paris.