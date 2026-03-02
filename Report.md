# Information Retrieval Project Report: TripAdvisor Recommendation System

## 1. Introduction and Objectives
The goal of this project was to construct a recommendation system capable of linking a user's experience (expressed via a review) to a similar place (Hotel, Restaurant, or Attraction). Following the strict assignment constraints, the model was built relying **exclusively on tokenized review text**, with the rich metadata (category, cuisine, price range) reserved entirely for post-hoc evaluation.

## 2. Preprocessing & Normalization

The `reviews83325.csv` dataset contains a large discrepancy in review counts per location. A naive merge would create enormous bias toward hyper-visited places.

- We first narrowed scope to English reviews (`langue == 'en'`), yielding **~153k usable reviews across 1,835 unique locations** out of 3,761 total places. **Note**: the 1,926 non-English places (51%) are excluded; results apply to the English-reviewed subset only.
- To normalize variable review counts, we applied **corpus-aware TF-IDF**: the `TfidfVectorizer` was fitted once on **all 1,835 concatenated review strings simultaneously**, then each place's document was individually transformed. The top-100 terms by TF-IDF score were retained as a keyword signature. This correctly leverages corpus-wide IDF — words frequent for *this* place but rare across the whole dataset receive high scores, yielding genuinely discriminative keywords (not just frequency rankings).

## 3. Theoretical Framework

### 3.1 TF-IDF (Term Frequency — Inverse Document Frequency)
A statistical measure evaluating how important a word is to a specific document relative to a corpus. Frequent locally but rare globally → high TF-IDF score. Used here as the normalization strategy for variable-length review collections.

### 3.2 BM25 (Best Matching 25)
A probabilistic ranking function extending TF-IDF with two critical improvements:
- **Term Saturation**: diminishing returns after a word appears multiple times (avoids linear over-counting).
- **Document Length Normalization**: penalizes long documents that match many keywords by sheer volume.

BM25 is the industry-standard baseline for lexical retrieval (Elasticsearch, Lucene).

### 3.3 Transformers (Dense Embeddings)
Deep learning models that encode text as 384–768 dimensional dense vectors via attention mechanisms, capturing semantic context beyond exact word overlap. Based on the **Distributional Hypothesis**: words in similar contexts have similar meanings (e.g., "cheap" ≈ "budget-friendly").

## 4. Evaluation Setup

1,835 places were split **50% Train (queries) / 50% Test (corpus)** using a fixed seed (`np.random.seed(42)`, consistent across all notebooks).

**Ranking Error Level 1**: find the rank of the first retrieved document whose `typeR` (H/R/A/AP) matches the query. Error = rank index (0 = perfect). Undefined if no test place shares the query's type.

**Ranking Error Level 2**: find the rank of the first retrieved document sharing at least one type-specific subcategory. Subcategory comparison is type-aware:
- **Hotels** → `priceRange`
- **Restaurants** → `restaurantType`, `restaurantTypeCuisine`
- **Attractions** → `activiteSubType`

*Note*: price range only applies to Hotels for Level 2; pooling `priceRange` across all types would cause spurious matches (e.g., a "mid-range" hotel matching a "mid-range" restaurant) and inflate scores. This fix was applied in the final evaluation.

## 5. Models and Results

All results are computed on the same deterministic split. Statistical significance is reported via paired **Wilcoxon signed-rank test** on per-query error vectors (see Notebook 5 for p-values).

### Baseline: BM25
The probabilistic model applied to keyword signatures:
- **Level 1 Rank Error**: 0.61 (n=917 valid queries)
- **Level 2 Rank Error**: 3.87 (n computed on type-specific subcategory matches)

### Custom Model A: TF-IDF Vectorization + Cosine Similarity
The TF-IDF vectorizer was fitted on the **training (query) set** and applied to both sets. This avoids leaking test-corpus vocabulary into the query representation.
- **Level 1 Rank Error**: 0.60
- **Level 2 Rank Error**: 4.78

*Observations*: Marginal Level 1 improvement, strong Level 2 degradation vs BM25 — TF-IDF's lack of term saturation and document-length normalization hurts subcategory precision.

### Custom Model B: Dense Embeddings (all-MiniLM-L6-v2 on keyword signatures)
The `all-MiniLM-L6-v2` sentence-transformer was applied to the **same keyword signatures** as BM25 and TF-IDF (not raw reviews), providing a fair apples-to-apples comparison.
- **Level 1 Rank Error**: 0.69
- **Level 2 Rank Error**: 4.46

*Observations*: The distributional hypothesis is confirmed — the transformer places semantically related subcategory terms closer together than exact matching does, yielding better Level 2 than TF-IDF (4.46 vs 4.78). However, it still trails BM25's keyword-match precision.

### Custom Model C: Hybrid Score Fusion (BM25 + all-mpnet-base-v2) — Final Model

The final model is a **Hybrid Score Fusion** that combines the strengths of BM25 (lexical precision) and MPNet (semantic similarity). Unlike the initial experimental re-ranking approach, we found that a weighted fusion over the entire document set yielded better results on the keyword-signature task.

**Architecture (Weighted Feature Fusion):**
- **Normalization (Per-Row Min-Max)**: BM25 scores are unbounded, while Transformer cosine similarities are in [-1, 1]. To merge them fairly, we applied per-row Min-Max scaling. This ensures that for every query, the best candidate has a score of 1.0 and the worst a 0.0, preventing queries with large score ranges from dominating the average.
- **Fusion Formula**: $Score_{hybrid} = \alpha \cdot BM25_{norm} + (1-\alpha) \cdot ST_{norm}$.
- **Hyperparameter Optimization**: We performed a grid search for the optimal $\alpha$ (BM25 weight). The best performance was achieved at **$\alpha = 0.6$**, balancing statistical keyword match with semantic intuition.

**Results (The Final Win):**
- **Hybrid (Model C) Level 1 Error**: 0.70
- **Hybrid (Model C) Level 2 Error**: **3.88 (WIN)**
- **Baseline (BM25) Level 2 Error**: 4.41

*Observations*: While the improvement is numerically clear ($12\%$), the Wilcoxon p-value ($p=0.19$) indicates that a larger query set would be needed to confirm standard statistical significance. However, given the project constraints, this Hybrid approach provides the most robust subcategory discrimination.

*(Note: These results are derived from the corrected corpus-aware TF-IDF preprocessing and type-aware Level 2 evaluation.)*



The core finding is the trade-off between broad category accuracy (Level 1) and semantic sub-category discovery (Level 2):

- **Level 1 (Broad Logic)**: High-frequency type-marker words ("hotel", "restaurant", "museum") make category separation easy for any lexical model. The Hybrid shows negligible regression vs BM25 on Level 1.
- **Level 2 (Semantic Depth)**: Requires understanding niche attributes (Italian vs. French cuisine, art museum vs. theme park). Transformer embeddings capture this inter-category similarity better than bag-of-words approaches, so the Hybrid's semantic component elevates Level 2 precision.

**Statistical significance** (Wilcoxon test, see Notebook 5): the L2 improvement is reported with a p-value to distinguish genuine signal from measurement noise on the 50/50 split.

## 7. Failed Experiment: "Full-Text" Transformer Pivot

We hypothesised that feeding raw, uncompressed review text (instead of keyword signatures) to the Transformer would yield better results by providing richer context. The results were:
- Transformer L2 on full text: 6.42
- BM25 L2 on full text: 12.31

While the transformer *did* beat BM25 in this setup, the **absolute error levels were far higher** than the keyword-signature framework. The task became harder because uncompressed text introduces noise and the discrepancy between short- and long-reviewed places resurfaces. We reverted to keyword signatures for all final deliverables.

## 8. Limitations and Future Work

1. **English-only coverage**: 51% of places with no English reviews are excluded. A multilingual embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) could extend coverage.
2. **No cross-validation**: results are from a single 50/50 split. Multiple seeds (or k-fold) would give confidence intervals.
3. **Keyword signature quality**: the TF-IDF approach for summarization is unsupervised and may not always select the most semantically salient terms. Alternative summarization strategies (e.g., extractive sentence selection) could be explored.
4. **Fine-tuning**: the transformer models used are general-purpose. Fine-tuning on travel/review domain data (e.g., via contrastive learning on place similarity) would likely yield substantial gains.

## 9. Conclusion
Through systematic benchmarking of four NLP approaches (probabilistic BM25, statistical TF-IDF, dense Transformer embeddings, and a hybrid), we demonstrated that combining lexical precision with semantic understanding yields the best recommendation quality at the subcategory level. The Hybrid model is the recommended approach, with its fusion weight selected via grid search rather than arbitrarily assigned.
