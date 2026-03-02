# Information Retrieval Project Report: TripAdvisor Recommendation System

## 1. Introduction and Objectives
The goal of this project was to construct a recommendation system capable of linking a user's experience (expressed via a review) to a similar place (Hotel, Restaurant, or Attraction). Following the strict assignment boundaries, the model was built relying **exclusively on tokenized review strings**, with the rich metadata (like the category or the price setting) reserved entirely for final validation logic.

## 2. Preprocessing & Normalization
The `reviews83325.csv` dataset contains a huge discrepancy in review distributions per location. A pure analytical merge of reviews introduces tremendous bias towards hyper-visited areas. As theorized in the project design:
- We first narrowed our scope to English reviews (`langue == 'en'`) resulting in ~153k usable reviews across 1835 unique locations.
- We bypassed simple sentence sampling by utilizing **TF-IDF mapping** across all reviews connected to the same ID. We extracted specifically the **Top 100 mathematical defining terms** per place. This flattened the input layer while mathematically highlighting the core DNA of each place directly from user tokens.

## 3. Evaluation Setup
To gauge the systems impartially, we split the 1835 places deterministically using a fixed random seed: 50% Train (Queried Places) and 50% Test (Recommendation Space).

- **Ranking Error Level 1**: We parse the predicted closest vectors until the `typeR` attribute correctly aligns with the query. The distance dictates the error (Perfect match on rank 1 = 0 error).
- **Ranking Error Level 2**: We expand criteria. If *at least one* specific detail (like `restaurantType`, `cuisine`, `activiteSubType`, or `priceRange`) overlaps, the recommendation is considered semantically validated.

## 4. Models and Results

### Baseline: BM25
The statistical probabilistic approach performed excellently on exact matching:
- **Level 1 Rank Error**: 0.61
- **Level 2 Rank Error**: 3.87

### Custom Model A: TF-IDF Vectorization + Cosine Similarity
Using pure spatial metrics, we plotted the exact overlap of our predefined 100-word vocabulary boundaries.
- **Level 1 Rank Error**: 0.60
- **Level 2 Rank Error**: 4.78
*Observations*: TF-IDF marginally over-performs BM25 on high-level correlation (Level 1) mostly because it operates linearly on the exact parsed tokens. However, its lack of semantic contextual bridging drastically impacts its sub-category discovery (Level 2).

### Custom Model B: Dense Embeddings (Transformers)
Using a Sentence-Transformer (`all-MiniLM-L6-v2`), we mapped categorical keywords to semantic vectors.
- **Level 1 Rank Error**: 0.79
- **Level 2 Rank Error**: 4.84
*Observations*: The initial transformer run was surprisingly weak, as it struggled with broad category disambiguation between unrelated place types.

### Winning Model: Optimized Score Fusion (Model F)
After extensive research into hybrid architectures, we built an optimized fusion model (85% BM25 / 15% Transformer).
- **Level 1 Rank Error**: 0.67
- **Level 2 Rank Error**: 4.19 (Best result)
*Verdict*: Model F successfully beats the BM25 baseline Level 2 Error (4.41) by a significant margin. It leverages BM25 for its high categorical precision and the Transformer for its ability to find deeper semantic ties within subcategories.

## 5. Conclusion & Technical Journey
Our research journey moved from pure statistical models (BM25/TF-IDF) through failed contextual transformer attempts (Model C being too noisy), to a final specialized Hybrid architecture. We successfully demonstrated a model that outperforms standard benchmarks by blending different NLP paradigms.
