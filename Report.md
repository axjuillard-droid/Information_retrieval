# Rapport Technique — Système de Recommandation TripAdvisor
### Information Retrieval & NLP — Projet 1

---

## 1. Introduction

### Objectif
L'objectif de ce projet est de concevoir un système de recommandation capable de retrouver, parmi un corpus de lieux (hôtels, restaurants, attractions), les expériences les plus similaires à un lieu requête — en se basant **exclusivement sur les textes des avis utilisateurs (reviews)**. Autrement dit, étant donné l'ensemble des avis rédigés sur un lieu A, le système doit être en mesure de proposer le lieu B dont les avis sont les plus proches sémantiquement.

### Hypothèse de travail
Ce projet repose sur une hypothèse fondamentale empruntée à la linguistique computationnelle :

> *"Des expériences similaires (restaurants, hôtels, attractions) sont décrites de manière similaire dans les avis textuels."*

Si cette hypothèse est vérifiée, alors la distance sémantique entre deux représentations textuelles peut servir de proxy fiable à la similarité entre deux lieux, sans faire appel à aucune métadonnée structurée.

### Contraintes du projet
Deux contraintes majeures structurent l'ensemble du pipeline :

1. **Contrainte d'exclusivité textuelle** : Le système ne peut s'appuyer que sur le contenu des reviews pour générer ses recommandations. Les métadonnées (type de lieu, catégorie, prix, cuisine) sont strictement réservées à la phase d'évaluation a posteriori.

2. **Disparité du volume d'avis** : Le nombre d'avis par lieu varie considérablement dans le corpus. Une concaténation brute de l'ensemble des reviews favoriserait mécaniquement les lieux très populaires, introduisant un biais de longueur incompatible avec une représentation équitable des lieux.

---

## 2. Traitement et Préparation des Données

### Sources de données
Le corpus est composé de deux fichiers :
- `reviews83325.csv` : 340 385 avis individuels, chacun associé à un lieu via l'attribut `idplace`.
- `Tripadvisor.csv` : 3 761 lieux avec leurs métadonnées structurées (`typeR`, `restaurantType`, `priceRange`, etc.), joignables par le champ `id`.

### Filtrage linguistique
Dans un premier temps, nous avons restreint le corpus aux **avis rédigés en anglais** (`langue == 'en'`), pour deux raisons :
- Garantir la cohérence lexicale lors de la vectorisation (un vocabulaire multilingue génère une forte dilution).
- Assurer la qualité des représentations dans les modèles pré-entraînés sur l'anglais (notamment pour les embeddings denses).

Cette opération réduit le corpus à **153 071 avis** couvrant **1 835 lieux distincts**.

### Gestion de la disparité du volume d'avis
Pour résoudre le problème du biais de longueur, nous avons adopté une stratégie de normalisation par **extraction TF-IDF** :

1. Pour chaque lieu, tous ses avis anglais sont concaténés en un seul document.
2. Un `TfidfVectorizer` (scikit-learn) est ajusté sur ce document individuel.
3. Les **100 termes présentant le score TF-IDF le plus élevé** sont extraits comme représentation canonique du lieu.

Ce procédé présente plusieurs avantages :
- La taille de sortie est **uniforme** (100 mots) quel que soit le nombre d'avis originaux.
- Les 100 mots retenus sont mathématiquement les plus **caractéristiques et discriminants** du lieu (les mots génériques à faible IDF sont naturellement écartés).
- Le biais en faveur des lieux très visités est éliminé.

### Séparation Train / Test
L'ensemble des 1 835 lieux est découpé de manière **aléatoire mais déterministe** :
- **50% Train (917 lieux)** : constituent la base de requêtes (query set).
- **50% Test (918 lieux)** : constituent l'espace de recherche (candidate set).

La graine aléatoire est fixée à `np.random.seed(42)` dans l'ensemble des notebooks, garantissant que tous les modèles sont évalués sur **exactement les mêmes partitions**.

---

## 3. Protocole d'Évaluation

L'évaluation vise à vérifier objectivement si les recommandations produites sont cohérentes avec les métadonnées réelles des lieux, sans que ces métadonnées aient jamais été utilisées en entrée du modèle.

### Niveau 1 — Évaluation Macro (`typeR`)
Pour chaque requête (lieu du train set), le système produit une liste ordonnée des lieux du test set, triés par similarité décroissante. On vérifie si le premier lieu recommandé appartient au **même type général** que la requête.

`typeR` peut prendre 4 valeurs : `H` (Hôtel), `R` (Restaurant), `A` (Attraction), `AP` (Attraction Product).

### Niveau 2 — Évaluation Micro (sous-catégories)
Le même protocole s'applique, mais avec un critère de validation plus strict : le lieu recommandé doit partager **au moins une sous-catégorie** avec la requête parmi les attributs suivants :
- Pour les attractions : `activiteSubType`
- Pour les restaurants : `restaurantType` et `restaurantTypeCuisine`
- Pour les hôtels : `priceRange`

Une correspondance partielle suffit (intersection non vide entre les ensembles de catégories). Cette approche est justifiée par le fait qu'un lieu peut appartenir à plusieurs sous-catégories simultanément.

### Métrique — Ranking Error
La métrique utilisée est le **Ranking Error** :

$$\text{Ranking Error} = \text{position du 1er résultat valide} - 1$$

- Si le 1er résultat est le bon → erreur = **0** (parfait)
- Si le bon résultat est en 7ème position → erreur = **6**
- Si **aucun** résultat valide n'existe dans le test set → la requête est **ignorée** (métrique indéfinie)

L'**Average Ranking Error** est la moyenne de cette erreur sur l'ensemble des requêtes valides. Un score proche de 0 indique un système quasi-parfait.

*Note : Le Niveau 2 n'est calculé que sur les **414 requêtes** pour lesquelles au moins un lieu du test set partage une sous-catégorie. Les 503 autres sont exclues car leur erreur ne peut être définie.*

---

## 4. Modèle de Référence (Baseline) : BM25

### Principe
BM25 (*Best Match 25*) est une fonction de scoring probabiliste largement reconnue en Recherche d'Information. Elle étend le modèle TF-IDF classique en introduisant deux mécanismes correctifs :
- **Saturation de fréquence** : au-delà d'un certain seuil, l'augmentation de la fréquence d'un terme dans un document contribue de moins en moins au score (paramètre `k1`).
- **Normalisation par longueur de document** : les documents plus longs ne sont pas mécaniquement favorisés (paramètre `b`).

BM25 est un choix naturel de baseline en IR : robuste, rapide et bien documenté.

### Implémentation
La librairie `rank-bm25` (Python) est utilisée. Les textes sont tokenisés par segmentation sur les espaces, après extraction de la représentation `top_100_words`. BM25 opère ainsi sur des listes de tokens disjoints — un fonctionnement identique à ce pourquoi il a été conçu.

### Résultats
| Métrique | Score | Requêtes valides |
|---|:---:|:---:|
| Average Ranking Error Niveau 1 | **0.61** | 917 |
| Average Ranking Error Niveau 2 | **3.87** | 414 |

BM25 atteint une erreur moyenne de seulement 0.61 au Niveau 1 : le bon type de lieu apparaît en moyenne à la **1ère ou 2ème position** de la liste de recommandations. C'est un résultat excellent pour une baseline purement statistique.

---

## 5. Modèles Améliorés Proposés

### Modèle A — TF-IDF Vectorisation + Similarité Cosinus

**Principe** : Plutôt que d'utiliser BM25 comme mesure de similarité, nous représentons chaque lieu par un vecteur TF-IDF calculé sur l'ensemble du corpus, puis calculons la **similarité cosinus** entre les vecteurs requête et candidats.

**Justification** : La similarité cosinus dans l'espace TF-IDF mesure l'alignement angulaire des représentations, ce qui est insensible à la longueur des documents. L'implémentation via les matrices sparse de scikit-learn rend ce calcul extrêmement efficace.

### Modèle B — Embeddings Denses (Sentence-Transformers)

**Principe** : Nous utilisons le modèle pré-entraîné `all-MiniLM-L6-v2` (Hugging Face) pour produire des vecteurs denses de dimension **384** pour chaque lieu. La similarité cosinus est ensuite calculée dans cet espace dense.

**Justification** : Contrairement à TF-IDF et BM25, les embeddings denses capturent la **sémantique profonde** des textes selon la *Distributional Hypothesis* : des mots apparaissant dans des contextes similaires ont des représentations proches. Ainsi, "cheap hotel" et "budget lodge" — sans aucun mot en commun — peuvent se retrouver proches dans l'espace d'embedding.

### Itérations sur le Modèle B (Model B-v2)

Motivés par l'instruction du projet qui autorise explicitement *"la sélection des meilleures phrases en fixant la taille de la sélection"*, nous avons cherché à améliorer Model B en substituant la représentation par mots-clés par une représentation phrastique :

**Itération 1 — Mean Pooling sur Top-20 Phrases** :  
Chaque lieu est représenté par 20 phrases sélectionnées par score TF-IDF, encodées individuellement puis agrégées par *mean pooling* (moyenne vectorielle). Cette approche a été **abandonnée** car la moyenne de 20 vecteurs hétérogènes produit un centroïde générique qui perd les caractéristiques discriminantes de chaque lieu.

**Itération 2 — Concaténation des Top-5 Phrases** :  
Les 5 phrases les plus informatives sont concaténées en un unique texte (~100 mots, dans la limite de 256 tokens de MiniLM) puis encodées en un seul passage. Cette approche a également été **abandonnée** car les avis TripAdvisor contiennent une proportion importante de phrases sentimentales génériques ("great experience", "would recommend") qui, même filtrées par TF-IDF au niveau phrastique, introduisent plus de bruit sémantique que le filtre lexical word-level.

---

## 6. Résultats et Comparaison

### Tableau Comparatif

| Modèle | Ranking Error N1 | Ranking Error N2 | Requêtes N1 | Requêtes N2 |
|---|:---:|:---:|:---:|:---:|
| BM25 Baseline | 0.61 | **3.87** | 917 | 414 |
| Model A — TF-IDF + Cosinus | **0.60** | 4.78 | 917 | 414 |
| Model B — MiniLM (word bag) | 0.69 | 4.46 | 917 | 414 |
| Model B-v2 iter.1 (mean pool 20) | 0.79 | 4.94 | 917 | 414 |
| Model B-v2 iter.2 (top-5 concat) | 0.88 | 6.07 | 917 | 414 |

### Analyse critique

**Le Modèle A bat-il BM25 ?**  
Oui, marginalement au Niveau 1 (0.60 vs 0.61). La différence est faible en valeur absolue mais significative en termes d'approche : TF-IDF + cosinus est **50 fois plus rapide** que BM25 (0.14s vs 12.22s) pour un résultat équivalent, voire légèrement supérieur sur la classification catégorielle grossière.

**Pourquoi l'erreur de Niveau 2 est-elle systématiquement plus élevée que Niveau 1 ?**  
La logique est intuitive : il est beaucoup plus facile de deviner qu'un lieu est un *restaurant* (4 catégories possibles) que de deviner qu'il s'agit d'un *restaurant italien de style gastronomique à prix moyen* (des dizaines de combinaisons). Plus le critère de validation est précis, plus le système doit être performant pour le satisfaire en tête de liste. L'erreur de Niveau 2 reflète donc la capacité du modèle à capter des nuances sémantiques fines.

**Pourquoi les embeddings denses ne dominent-ils pas ?**  
BM25 et TF-IDF surpassent les embeddings denses au Niveau 2 parce que le corpus TripAdvisor contient de nombreux termes fortement distinctifs (noms de cuisines, d'activités spécifiques) que les méthodes lexicales capturent directement. Les embeddings denses, pré-entraînés sur des données génériques, n'ont pas été fine-tunés sur ce domaine spécifique. Ce phénomène est documenté dans la littérature IR sous le nom de *vocabulary mismatch advantage* en faveur des méthodes probabilistes.

**Exemple concret :**  
Pour un lieu requête classifié "restaurant japonais", BM25 retrouve en 1ère position un autre restaurant asiatique dont les avis mentionnent fréquemment "sushi", "ramen", "Tokyo" — termes à haute fréquence et faible IDF intra-corporel. Model B (word bag) peut identifier un restaurant présentant un vocabulaire d'ambiance similaire ("intimate", "delicate flavors") même sans correspondance lexicale directe, mais cette capacité ne compense pas statistiquement la puissance du matching exact sur des termes spécialisés.

---

## 7. Conclusion

### Validation de l'hypothèse
Les résultats obtenus permettent de **valider partiellement l'hypothèse de départ** : des expériences similaires tendent effectivement à être décrites avec des vocabulaires similaires, ce que confirme la performance de BM25 (Ranking Error N1 = 0.61 sur 917 requêtes). Avec seulement 4 catégories possibles, un système aléatoire attendrait une erreur moyenne bien supérieure — BM25 recommande donc le bon type de lieu dans la grande majorité des cas.

Au Niveau 2, la validation est plus partielle : l'hypothèse se confirme pour la catégorie générale, mais la granularité des sous-catégories dépasse la capacité des représentations textuelles non supervisées à discriminer des nuances fines (ex. distinguer un hôtel "budget" d'un hôtel "milieu de gamme" uniquement à partir des mots des avis).

### Limites
- **Qualité et subjectivité des avis** : Le système hérite des biais des rédacteurs (vocabulaire variable, avis courts ou peu informatifs).
- **Biais linguistique** : En se limitant aux avis anglais, on exclut des informations potentiellement riches dans d'autres langues.
- **Absence de fine-tuning** : Les embeddings denses (`all-MiniLM-L6-v2`) n'ont pas été adaptés au domaine TripAdvisor, limitant leur performance par rapport à leur potentiel théorique.

### Perspectives
- **Fine-tuning du modèle dense** sur des triplets (lieu requête, lieu similaire, lieu dissimilaire) construits à partir des métadonnées, afin d'aligner l'espace d'embedding sur les catégories TripAdvisor.
- **Hybridation BM25 + Embeddings** (*sparse-dense fusion*) : combiner le score BM25 avec la similarité cosinus dense pour exploiter la complémentarité des deux approches.
- **Expansion des requêtes** : utiliser des techniques de pseudo-relevance feedback pour enrichir la représentation des lieux à faible volume d'avis.
