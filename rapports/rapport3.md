# Rapport semaine 3 

## 1. Partie VAE et espace latent

Sur cet espace latent, deux approches sont testées : une classification par mouvement artistique (14 classes, SVM RBF) qui contourne le problème du déséquilibre, et un réseau siamois qui apprend à rapprocher les œuvres d'un même artiste sans nécessiter plusieurs exemples par classe.

### Résultats de visualisation
La comparaison t-SNE des deux espaces (Top-20 artistes) montre une transformation notable : l'espace VAE brut est un nuage uniforme sans structure, tandis que l'espace siamois fait apparaître des clusters isolés pour les artistes disposant de plusieurs œuvres. Le nuage central reste dense pour les artistes à image unique — limite intrinsèque du dataset, pas du modèle.

### Limites et suite
Cette approche est une première brique. Les pistes d'amélioration incluent la Triplet Loss (plus efficace que la perte contrastive sur les datasets déséquilibrés) et une augmentation plus aggressive. D'autres approches complémentaires sont explorées en parallèle.
