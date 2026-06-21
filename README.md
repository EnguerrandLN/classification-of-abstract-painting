# projet-im06

Le notebook features.ipynb permet de calculer les features géométriques classiques. Le fichier features.pkl contient ses features déjà calculées sur les données abstrait-v4, mais on peut le supprimer pour les recalculer.

Le notebook latent-space-vae.ipynb permet de calculer les features du VAE. Le résultat est sauvegardé dans le fichier latent_vectors.npy.

Le notebook cnn.ipynb permet de calculer les features du CNN. Le résultat est sauvegardé dans le fichier CNN_features_PCA.npy.

Le notebook melange_features_CNN.ipynb fait la fusion des 3 approches et calcule le K-means, l'écrémage, ainsi que des tests pour évaluer la classification.

La visualisation des données se fait dans le dossier "visualisation et data", où un fichier tuto détaille la démarche pour la mettre en oeuvre.

Notre classification finale (décrite dans le rapport et lors de la défense orale) peut être retrouvée dans le dossier classification_finale.