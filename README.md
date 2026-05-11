# PolyBERT — Détection de la polysémie par embeddings contextuels

Projet final de M1 TAL — Sémantique distributionnelle  
Modèle : **CamemBERT** (`camembert-base`) | Langue : **Français** | Corpus : **Wikipedia**

---

## Objectif

Mesurer automatiquement le degré de polysémie d'un mot à partir de ses embeddings contextuels produits par CamemBERT, et valider cette mesure en la corrélant avec une ressource lexicale de référence (WordNet français via NLTK).

**Hypothèse** : un mot polysémique apparaît dans des contextes sémantiquement très variés → ses embeddings sont dispersés dans l'espace → l'écart-type des similarités cosinus entre paires d'embeddings est élevé.

---

## Score de polysémie

Pour chaque mot cible, le pipeline :

1. Extrait des phrases depuis Wikipedia contenant le mot
2. Calcule l'embedding contextuel du mot dans chaque phrase via CamemBERT (couche finale, moyenne des sous-tokens)
3. Calcule toutes les similarités cosinus paires-à-paires entre embeddings
4. Retourne l'**écart-type** de ces similarités comme score de polysémie

```
score_polysemie(mot) = std({ cos_sim(eᵢ, eⱼ) | i < j })
```

Plus le score est élevé, plus les contextes d'apparition du mot sont sémantiquement diversifiés.

---

## Structure du projet

```
polybert_sem/
│
├── polybert_functions.ipynb     # Bibliothèque de fonctions communes (modèle, corpus, scores, viz)
│
├── polybert_20mots.ipynb        # Analyse principale — 20 mots (10 polysémiques / 10 monosémiques)
├── script_polybert.ipynb        # Version exploratoire initiale (même pipeline, code non modulaire)
├── polybert_1000mots.ipynb      # Extension à grande échelle — 671 mots du vocabulaire fréquent
│
├── score_polysemie.png          # Barplot des scores pour les 20 mots
├── tsne_global.png              # t-SNE global — 20 mots, 754 embeddings
├── tsne_1000mots.png            # t-SNE — top 30 mots par volume d'embeddings (1000 mots)
├── correlation_wordnet.png      # Nuage de points : score BERT vs nb de sens WordNet (20 mots)
├── correlation_wordnet_1000.png # Idem pour l'analyse à 1000 mots
└── viz_<mot>.png                # Projections PCA 2D individuelles par mot
```

---

## Notebooks

### `polybert_functions.ipynb` — Fonctions communes

Chargé via `%run polybert_functions.ipynb` dans les autres notebooks. Contient :

| Fonction | Rôle |
|---|---|
| `extraire_phrases(texte, mot)` | Segmente un article Wikipedia et filtre les phrases contenant le mot (20–300 caractères) |
| `get_word_embedding(phrase, mot)` | Tokenise avec CamemBERT, reconstruit les sous-tokens, retourne le vecteur 768-dim du mot |
| `build_corpus(articles_par_mot)` | Construit un corpus depuis un dict `{mot: [titres Wikipedia]}` |
| `build_corpus_auto(mots)` | Corpus automatique par recherche Wikipedia sur une liste de mots |
| `extract_embeddings(corpus)` | Extrait les embeddings pour tout le corpus |
| `score_polysemie(vecteurs)` | Calcule l'écart-type cosinus (score de polysémie) |
| `compute_scores_df(embeddings)` | Agrège les scores dans un DataFrame trié |
| `viz_bar_scores(df)` | Barplot horizontal des scores |
| `viz_mot_pca(mot, ...)` | Projection PCA 2D des embeddings d'un mot |
| `viz_tsne_global(embeddings)` | t-SNE sur l'ensemble des embeddings de tous les mots |
| `nb_sens_wordnet(mot)` | Nombre de synsets WordNet (proxy du nombre de sens) |
| `viz_correlation_wordnet(df)` | Nuage de points score BERT vs WordNet + corrélations |

### `polybert_20mots.ipynb` — Analyse principale

Analyse contrôlée sur un ensemble équilibré de 20 mots français :

**10 mots polysémiques** : café, souris, langue, fraise, avocat, bois, livre, vol, action, tour  
**10 mots monosémiques** : médecin, volcan, pentathlon, atome, photosynthèse, chrysanthème, hydrogène, électron, antibiotique, alambic

Le corpus est construit manuellement : chaque mot est associé à plusieurs articles Wikipedia thématiquement distincts (ex. `"avocat"` → `["Avocat (fruit)", "Avocat (métier)"]`).

**Résultats clés :**

| Mot | Score polysémie | WordNet | Type |
|---|---|---|---|
| tour | 0.163 | 18 sens | polysémique |
| livre | 0.144 | 19 sens | polysémique |
| avocat | 0.151 | 8 sens | polysémique |
| antibiotique | 0.039 | 2 sens | monosémique |
| photosynthèse | 0.043 | 1 sens | monosémique |

Corrélation avec WordNet : **Spearman ρ = 0.629 (p = 0.003)** — significative au seuil 5%.

### `polybert_1000mots.ipynb` — Extension à grande échelle

Passage à l'échelle sur les 1000 mots les plus fréquents du français (bibliothèque `wordfreq`), filtrés à ≥ 5 caractères → **671 mots**, dont **666 scorés** (au moins 5 embeddings).

Le corpus est construit automatiquement via `build_corpus_auto` (recherche Wikipedia par mot).  
La référence lexicale est WordNet anglais (`wn.synsets(mot)` sans paramètre de langue), utilisé comme proxy.

**Observation** : les mots polysémiques (≥ 2 synsets) ont un score moyen légèrement supérieur (0.153 vs 0.147), cohérent avec l'hypothèse mais la différence s'atténue à grande échelle — les mots grammaticaux très fréquents (pronoms, conjonctions) génèrent des scores élevés artéfactuels.

---

## Pipeline complet

```
Mots cibles
    │
    ▼
Corpus Wikipedia  ──────────────────────────────────────────────┐
(phrases contenant le mot, 20–300 char)                         │
    │                                                           │
    ▼                                                           │
CamemBERT (camembert-base)                                      │
→ embedding 768-dim par occurrence                              │
    │                                                           │
    ▼                                                           │
Similarités cosinus paires-à-paires                             │
→ score = std(similarités)                                      │
    │                                                           │
    ├── Barplot des scores                                       │
    ├── PCA 2D par mot                                          │
    ├── t-SNE global                                            │
    └── Corrélation avec WordNet ◄──────────────────────────────┘
```

---

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd polybert_sem

# Créer l'environnement (uv recommandé)
uv venv --python 3.10
source .venv/bin/activate

# Dépendances
uv pip install transformers torch numpy pandas scipy scikit-learn \
               matplotlib wikipedia-api wordfreq nltk
```

Téléchargement des ressources NLTK (une seule fois) :

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Utilisation

Lancer JupyterLab depuis la racine du projet :

```bash
jupyter lab
```

Ordre d'exécution recommandé :

1. `polybert_functions.ipynb` — vérifie que le modèle se charge correctement
2. `polybert_20mots.ipynb` — analyse principale (≈ 5–10 min selon CPU)
3. `polybert_1000mots.ipynb` — analyse à grande échelle (≈ 1–2h selon CPU)

> **Note** : tous les notebooks appellent `%run polybert_functions.ipynb` en première cellule pour charger le modèle et les fonctions.

---

## Résultats visuels

| Fichier | Description |
|---|---|
| `score_polysemie.png` | Score BERT pour les 20 mots (rouge = polysémique, vert = monosémique) |
| `correlation_wordnet.png` | Corrélation score BERT × nombre de sens WordNet (20 mots) |
| `tsne_global.png` | Projection t-SNE des 754 embeddings des 20 mots |
| `correlation_wordnet_1000.png` | Corrélation à grande échelle (666 mots) |
| `tsne_1000mots.png` | t-SNE sur les 30 mots avec le plus d'embeddings (1000 mots) |
| `viz_<mot>.png` | Projection PCA 2D individuelle pour chaque mot analysé |

---

## Limites

- **CamemBERT** produit des embeddings très similaires pour des contextes proches → les scores restent dans une plage étroite (0.04–0.16 pour 20 mots).
- Les mots grammaticaux (pronoms, conjonctions) dans l'analyse à 1000 mots génèrent des scores élevés non liés à la polysémie lexicale.
- **WordNet français** (via `omw-1.4`) a une couverture incomplète ; certains mots retournent 0 synset malgré leur richesse sémantique.
- Le nombre de phrases par mot est limité par la disponibilité des articles Wikipedia.
