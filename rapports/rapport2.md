Partie Enguerrand
# Rapport - Semaine 2 : Deep Learning, Fusion Multimodale et Features Expertes

## 1. Extraction de Caractéristiques Profondes (CNN)

Afin de pallier les limites des primitives classiques (qui tendent à noyer l'information structurelle dans la variance des HOG), nous avons exploré la construction d'un espace latent basé sur un réseau de neurones convolutif pré-entraîné.

* **Architecture** : Utilisation du modèle **VGG16** (pré-entraîné sur ImageNet), amputé de ses couches de classification finales pour l'utiliser comme pur extracteur visuel.
* **Réduction de Dimension (PCA)** : Le tenseur brut a été réduit via une PCA à **50 composantes**, isolant l'essence stylistique (nervosité du trait, abstraction) tout en conservant un temps de calcul viable.

## 2. Implémentation des "Features Expertes" (Réponse aux retours)

Suite aux retours de l'encadrement, nous avons implémenté trois métriques expertes très ciblées pour traiter les angles morts de nos premiers algorithmes :

1. **Détection du Minimalisme ("Ratio JPEG")** : Calcul du poids du fichier divisé par sa résolution. Une toile minimaliste ou monochrome (type "carré blanc") offre un taux de compression mathématiquement bien supérieur à une toile complexe, permettant de l'isoler.
2. **Mesure de Symétrie (Grille 4x4)** : Évaluation de la différence de luminosité absolue entre les colonnes symétriques de la grille pour isoler les compositions géométriques centrées.
3. **Détection de Grilles Régulières (FFT)** : Application d'une Transformée de Fourier 2D spatiale (FFT) et comptage des pics d'intensité pour détecter la présence de motifs répétitifs (Vasarely, Toroni).

## 3. La Fusion Multimodale Définitive (Espace à 103 dimensions)

Pour unifier ces approches sans créer de conflit sémantique, nous avons procédé à une concaténation pondérée (*Alpha-Blending*) : **85% pour le CNN, 15% pour les features classiques, et 40% pour les 3 features expertes**.

* **Bilan quantitatif** : Le K-Means (K=15) appliqué à cette matrice de 103 dimensions génère un score de silhouette de **0.037**. Cette très légère baisse par rapport au CNN pur s'explique par l'introduction de nos contraintes sémantiques strictes (FFT, Minimalisme). L'espace vectoriel est mathématiquement un peu moins "lisse", mais visuellement et historiquement beaucoup plus cohérent.

## 4. Validation Sémantique des Clusters (K=15)

La répartition des œuvres au sein des clusters prouve que l'algorithme hybride a réussi à reconstruire des familles stylistiques majeures :

* **Cluster 0 - Le Grand Carrefour de la Matière** : C'est le cluster dominant de notre espace (320 œuvres). Nicolas de Staël y est littéralement sanctuarisé (24 de ses œuvres y sont regroupées). Il est rejoint par Hans Hartung (14 œuvres) et Zao Wou-Ki (7). L'algorithme a parfaitement identifié les empâtements et la peinture au couteau.
* **Cluster 13 - L'Expressionnisme Chromatique** : Joan Mitchell y règne de manière spectaculaire (19 œuvres isolées). Le réseau a reconnu son chaos lumineux et gestuel caractéristique.
* **Le Cas Mark Rothko (L'évolution du style)** : L'algorithme a divisé méthodiquement les toiles de Rothko. Si un noyau dur est identifié dans le Cluster 3 (15 œuvres), le reste est réparti (Cluster 1, C0, C14). Le réseau reconnaît ses rectangles vaporeux, mais les primitives classiques forcent la ségrégation par période chromatique.
* **L'Anomalie Spatiale (Cluster 9)** : Lucio Fontana domine ce sous-groupe très spécifique avec 12 œuvres. Ses toiles monochromes fendues créent un signal géométrique unique, très probablement renforcé par notre nouvelle feature de "Ratio JPEG".

## 5. Analyse des Similarités Ciblées (Distances Cosinus)

Pour valider formellement la robustesse de notre espace face aux interrogations du jury, nous avons calculé la distance Cosinus entre les centroïdes (vecteurs moyens) d'artistes ciblés.

* **Opposition Radicale (Accardi vs Riley)** : La distance calculée entre Carla Accardi (traits libres) et Bridget Riley (Op Art géométrique) s'élève à **1.2423**. Dans cet espace vectoriel, une distance supérieure à 1.0 indique une quasi-orthogonalité. Le modèle confirme mathématiquement qu'ils n'appartiennent à aucun voisinage sémantique commun.
* **La Famille des Motifs (Vasarely / Riley / Toroni)** : À l'inverse, l'ajout de notre détecteur spectral (FFT) a porté ses fruits. Les distances se réduisent drastiquement entre les peintres de la régularité : Vasarely se rapproche fortement de Riley (**0.6962**) et de Toroni (**0.7434**). L'algorithme a su créer un sous-espace latent dédié à la géométrie répétitive, contournant la prédominance des couleurs.