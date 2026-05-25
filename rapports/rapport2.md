Partie Enguerrand
# Rapport - semaine 2

## 1. Extraction de Caractéristiques Profondes (CNN)

Afin de pallier les limites des primitives classiques (qui tendent à noyer l'information structurelle dans la variance des HOG), nous avons exploré la construction d'un espace latent basé sur un réseau de neurones convolutif pré-entraîné.

* **Architecture** : Utilisation du modèle **VGG16** (pré-entraîné sur ImageNet), amputé de ses couches de classification finales pour l'utiliser comme pur extracteur visuel.
* **Réduction de Dimension (PCA)** : Le tenseur brut a été réduit via une PCA à **50 composantes**, isolant l'essence stylistique (nervosité du trait, abstraction) tout en conservant un temps de calcul viable.

## 2. Nouvelles features

Suite à vos retours, nous avons implémenté de nouvelles métriques très ciblées pour traiter les angles morts de nos premiers algorithmes :

1. Taille de l'image : taille brute, taille normalisée par nombre de pixels, et log de la taille
2. Symétrie : symétrie gauche/droite, haut/bas, diagonale, et gradient centre/bords
3. Application d'une TF en 2D spatiale (FFT) et comptage des pics d'intensité pour détecter la présence de motifs répétitifs
4. Mesures d'entropie : entropie multi-canal (R, G, B, saturation), nombre de couleurs distinctes quantifiées, ratio de zones uniformes (std locale < seuil)
5. Approximation par filtre gaussien large (sigma=2 et sigma=5) suivi d'un gradient Sobel avec seuillage sélectif au 85e percentile
6. Transformée de Hough probabiliste (nombre de lignes longues, proportion H/V, entropie angulaire) et mesures de compacité et rectangularité des régions

Une piste que nous aurions serait de trouver au cas par cas des features très spécifiques à certains peintres si ceux-ci sont pathologiques.

## 3. La Fusion Multimodale Définitive (Espace à 103 dimensions)

Pour unifier ces approches sans créer de conflit sémantique, nous avons procédé comme précédemment à une concaténation pondérée : 85% pour le CNN, 15% pour les anciennes features, et 40% pour les nouvelles features.

* **Bilan quantitatif** : Le K-Means (K=15) appliqué à cette matrice de 103 dimensions génère un score de silhouette de **0.037**. Cette très légère baisse par rapport au CNN pur s'explique par l'introduction de nos contraintes sémantiques strictes (FFT, Minimalisme). L'espace vectoriel est mathématiquement un peu moins "lisse", mais visuellement et historiquement beaucoup plus cohérent.

## 4. Validation Sémantique des Clusters (K=15)

La répartition des œuvres au sein des clusters prouve que l'algorithme hybride a réussi à reconstruire des familles stylistiques majeures :

* **Cluster 0 : C'est le cluster dominant de notre espace (320 œuvres). Nicolas de Staël y est littéralement sanctuarisé (24 de ses œuvres y sont regroupées). Il est rejoint par Hans Hartung (14 œuvres) et Zao Wou-Ki (7). L'algorithme a parfaitement identifié les empâtements et la peinture au couteau.
* **Cluster 13 : Joan Mitchell y règne de manière spectaculaire (19 œuvres isolées). Le réseau a reconnu son chaos lumineux et gestuel caractéristique.
* **Le cas Mark Rothko** : L'algorithme a divisé méthodiquement les toiles de Rothko. Si un noyau dur est identifié dans le Cluster 3 (15 œuvres), le reste est réparti (Cluster 1, C0, C14). Le réseau reconnaît ses rectangles vaporeux, mais les primitives classiques forcent la ségrégation par période chromatique.
* **Cluster 9** : Lucio Fontana domine ce sous-groupe très spécifique avec 12 œuvres. Ses toiles monochromes fendues créent un signal géométrique unique, très probablement renforcé par notre nouvelle feature de "Ratio JPEG".

## 5. Analyse des Similarités Ciblées (Distances Cosinus)

Pour valider formellement la robustesse de notre espace face aux interrogations du jury, nous avons calculé la distance Cosinus entre les centroïdes (vecteurs moyens) d'artistes ciblés.

* **Opposition Radicale (Accardi vs Riley)** : La distance calculée entre Carla Accardi (traits libres) et Bridget Riley (Op Art géométrique) s'élève à **1.2423**. Dans cet espace vectoriel, une distance supérieure à 1.0 indique une quasi-orthogonalité. Le modèle confirme mathématiquement qu'ils n'appartiennent à aucun voisinage sémantique commun.
* **La Famille des Motifs (Vasarely / Riley / Toroni)** : À l'inverse, l'ajout de notre détecteur spectral (FFT) a porté ses fruits. Les distances se réduisent drastiquement entre les peintres de la régularité : Vasarely se rapproche fortement de Riley (**0.6962**) et de Toroni (**0.7434**). L'algorithme a su créer un sous-espace latent dédié à la géométrie répétitive, contournant la prédominance des couleurs.