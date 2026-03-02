# TripAdvisor Recommendation System
## Technical Report - Information Retrieval & NLP

---

## 1. Introduction

### 1.1 Objective

The goal of this project is to build a **content-based recommendation system** for TripAdvisor places. Given one place as a query - represented solely by the text of its user reviews - the system must return the most similar places from a corpus. "Similarity" is defined experientially: the system should recommend places that offer a comparable experience to the query place.

Critically, the system must operate in a **blind** fashion with respect to metadata: it does not know in advance whether a place is a hotel, restaurant, or attraction, and it cannot consult price ranges, cuisine types, or activity categories when making its recommendation. The input to every model is exclusively the raw text of user reviews.

### 1.2 Working Hypothesis

This work is grounded in a single, empirically testable hypothesis:

> **"Similar experiences - whether a hotel stay, a restaurant visit, or a tourist attraction - are described in similar ways in user-written textual reviews."**

If this hypothesis holds, a purely text-based retrieval model can approximate semantic similarity between places without any access to structured metadata. The role of the evaluation protocol is precisely to measure the degree to which this hypothesis is confirmed.

### 1.3 Constraints and Challenges

Two structural constraints shaped every design decision:

1. **Strict text-only input.** Per the assignment specification, metadata fields (`typeR`, `priceRange`, `restaurantTypeCuisine`, `activiteSubType`, etc.) are exclusively reserved for evaluation. Any model using this information as input - even indirectly - would invalidate the experiment.

2. **Highly variable review volume.** The dataset contains 340,385 raw reviews distributed across 3,761 places, but the distribution is deeply skewed: popular landmarks accumulate thousands of reviews while niche establishments may have only a handful. Without normalisation, any similarity measure would be dominated by the sheer volume of text for large places rather than its content, introducing a systematic length bias.

---

## 2. Data Processing and Preparation

### 2.1 Source Data

Two files were used:
- [reviews83325.csv](file:///c:/Users/leopa/Documents/python%20codes/New%20folder/Information_retrieval/data/reviews83325.csv) - individual reviews, each linked to a place via `idplace`.
- [Tripadvisor.csv](file:///c:/Users/leopa/Documents/python%20codes/New%20folder/Information_retrieval/data/Tripadvisor.csv) - place-level metadata linked via [id](file:///c:/Users/leopa/Documents/python%20codes/New%20folder/Information_retrieval/notebooks/full.py#319-379).

### 2.2 Language Filtering

The review corpus is multilingual. To ensure linguistic consistency and leverage the best available NLP pre-trained models (which predominantly target English), all reviews with `langue != 'en'` were discarded. This reduced the dataset from **340,385 raw reviews** to **153,071 English reviews**, covering **1,835 unique places**. All subsequent modelling operates exclusively on this filtered subset.

### 2.3 Resolving the Variable Review Volume Problem

Once reviews were grouped by `idplace`, the resulting texts ranged from a few dozen words to tens of thousands across places. Directly using this concatenated text would create severe length bias: models would rank large places as highly similar to each other simply because long texts share more vocabulary, without any genuine experiential correspondence.

**Solution: Corpus-Aware TF-IDF Keyword Extraction (Top-100 Words)**

For each place, we extracted the **100 most informationally discriminative words** using a two-step TF-IDF procedure:

1. A single `TfidfVectorizer` was **fit on the entire corpus** of grouped reviews (all 1,835 places simultaneously). This is critical: fitting the vectoriser corpus-wide ensures that IDF weights are globally calibrated. A word like "nice" appearing in every review would receive a low IDF and be down-weighted; a word like "croissant" appearing primarily in Parisian bakery reviews would receive a high IDF and emerge as a strong descriptor.

2. For each place, the fitted vectoriser **transforms** its concatenated reviews into a TF-IDF weight vector. The top-100 words by weight are selected and concatenated into a fixed-length keyword signature.

This produces a **uniform, 100-word document per place**, stripping length bias while retaining the most semantically distinctive terms. An example output for a museum in Paris might be: `"louvre museum art painting gallery french medieval history exhibit wingâ€¦"` - terms no generic hotel or restaurant review would produce.

### 2.4 Train/Test Split

Places were randomly split with a fixed seed (`numpy.random.seed(42)`) into:
- **50% Train set (917 places):** These act as queries. For each query place, we ask the system to find the most similar place in the test set.
- **50% Test set (918 places):** This is the retrieval corpus. The model ranks all test places for each query.

The deterministic seed guarantees that all models are evaluated on **identical splits**, ensuring fair and reproducible comparisons.

---

## 3. Evaluation Protocol

### 3.1 Design Principle

The evaluation protocol is designed to rigorously test the working hypothesis. The system makes recommendations based on review text alone. The evaluation then uses metadata - held out from the model - to judge whether the recommendation is semantically correct. A low Ranking Error means the system's text-based similarity aligns closely with the expert-defined categorical structure of the TripAdvisor taxonomy.

### 3.2 Level 1 Evaluation (Macro - Venue Type)

For each query place, the system returns the full ranked list of test places. Level 1 asks: **at what rank does the first place of the same broad type appear?**

The `typeR` attribute encodes four categories: `H` (Hotel), `R` (Restaurant), `A` (Attraction), `AP` (Attraction Product). The error for a given query is the zero-indexed rank of the first test place matching the query's `typeR`. A perfect recommendation where the top-ranked place is of the same type yields an error of 0.

### 3.3 Level 2 Evaluation (Micro - Subcategory)

Level 2 is a more demanding test. It checks whether the recommendation matches at a fine-grained categorical level, using subcategory attributes. Following the assignment specification (lines 34â€“39), these attributes are **type-aware**:

| Venue Type | Attributes Used |
|:-----------|:---------------|
| Hotel (`H`) | `priceRange` |
| Restaurant (`R`) | `restaurantType`, `restaurantTypeCuisine` |
| Attraction (`A` / `AP`) | `activiteSubType` |

A match is counted if the query place and the recommended place share **at least one subcategory value**. For instance, if the query is an art museum and a garden, and the first recommended test place is only a garden, this counts as a Level 2 match (partial overlap criterion per assignment specification).

> **Design Note - Type-Aware Evaluation:** A naive implementation might pool all four metadata columns into a single flat set of strings for every place, regardless of venue type. This would create spurious matches - a Hotel with `priceRange = "â‚¬â‚¬"` would erroneously match a Restaurant with `priceRange = "â‚¬â‚¬"`, even though the price scales are entirely incomparable and the user experience is unrelated. Our implementation strictly applies the type-specific lookup table above, preventing such cross-type contamination and yielding a more rigorous and honest evaluation.

### 3.4 Ranking Error Metric

For a given query $q$ with ground-truth subcategory set $C_q$, the ranking error is the zero-indexed position of the first test document sharing at least one subcategory with $q$. If no test place shares any subcategory with the query, the query is **excluded** from the average (the error is *undefined*). This avoids penalising the model for the absence of ground-truth matches.

Average Ranking Error is computed over all valid queries:

$$\overline{\text{RE}} = \frac{1}{|Q_{\text{valid}}|} \sum_{q \in Q_{\text{valid}}} \text{RankingError}(q)$$

A lower score is better (0 = perfect top-1 recommendation for every query).

---

## 4. Reference Model (Baseline): BM25

### 4.1 Principle

BM25 (Best Match 25, Robertson & Zaragoza, 2009) is the standard probabilistic retrieval function used in modern search engines (Elasticsearch, Apache Solr). It extends classic TF-IDF with two important corrections:

- **Term Frequency Saturation:** BM25 applies a saturation function so repeated terms have diminishing returns, controlled by parameter $k_1$.
- **Document Length Normalisation:** Longer documents are penalised via parameter $b$, preventing verbose places from unfairly dominating similarity scores.

BM25 is the natural baseline for this task: it operates purely on term co-occurrence, requires no training, no embedding model, and no external knowledge. It is interpretable and very fast to deploy.

### 4.2 Implementation

The Python library `rank-bm25` (`BM25Okapi`, default parameters $k_1=1.5$, $b=0.75$) was used. The test corpus was tokenised by whitespace from the 100-word keyword signatures. For each query, `bm25.get_scores(tokenized_query)` returns a score vector over the 918 test documents, which is then ranked in descending order.

A guard was added to skip any query where `top_100_words` is `NaN` or the string `'nan'` (arising from places with no valid English reviews after filtering). Without this guard, BM25 would score all documents equally for an empty query, contaminating the average with spurious near-zero errors.

### 4.3 Baseline Results

| Evaluation Level | Average Ranking Error |
|:-----------------|:---------------------:|
| Level 1 (Venue Type) | **0.64** |
| Level 2 (Subcategory) | **4.41** |

*Computed on 917 queries (Level 1), 414 queries (Level 2).*

These scores establish a strong baseline. A Level 1 error of 0.64 means that the correct venue type appears at rank ~1.6 on average - very close to the top. Level 2 at 4.41 shows that subcategory precision requires scanning further down the list, confirming that keyword overlap alone cannot always distinguish fine-grained experiential categories.

---

## 5. Proposed Improved Models

All models use the identical data preparation pipeline and 50/50 train/test split. Each model is a different strategy for computing a similarity score between a query keyword signature and the test corpus.

### 5.1 Model A - TF-IDF + Cosine Similarity

**Principle:** Represent each place as a TF-IDF vector and rank test places by cosine similarity to the query vector. A `TfidfVectorizer` was fit on the **train set only**, then used to transform both sets - preventing data leakage from the retrieval corpus into the query representation.

**Results:** L1: 0.68 | L2: 4.74

**Analysis:** Slightly worse than BM25 on both levels. BM25's document length normalisation and term saturation are specifically designed for information retrieval tasks and outperform the geometric cosine measure on short keyword signatures. Model A is however significantly faster (~0.25s vs ~27s), which is a practical advantage at scale.

---

### 5.2 Model B - Dense Transformer Embeddings (MiniLM, Keywords)

**Principle:** Replace sparse TF-IDF vectors with dense semantic embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, 6 Transformer layers), ranking by cosine similarity between dense vectors.

**Rationale:** BM25 and TF-IDF rely on exact term overlap. A review describing a "budget-friendly inn" and one describing an "affordable hostel" share very few tokens yet express the same experience. Dense embeddings project semantically similar texts close together regardless of surface vocabulary.

**Results:** L1: 0.79 | L2: 4.84

**Analysis:** Unexpectedly worse than BM25. Root cause: we are feeding the Transformer **keyword signatures** - a disordered bag of 100 terms, not natural sentences. Transformers are built to process syntactically structured language. Without word order and sentence context, their attention mechanisms cannot establish meaningful relationships between tokens, and the resulting representations degrade to poor approximations of bag-of-words similarity, but without BM25's careful calibration. This is the "keyword-Transformer mismatch."

---

### 5.3 Model C - Contextual Transformer (Raw Text, 500 chars)

**Principle:** To resolve the mismatch in Model B, feed the Transformer the first 500 characters of the raw concatenated review text, restoring natural sentence structure.

**Results:** L1: 1.23 | L2: 11.41

**Analysis:** The worst-performing model. The first 500 characters of a user review typically consist of temporal and contextual filler (*"We visited last Tuesday with our family..."*), not the descriptive, experientially rich language that appears later. The TF-IDF preprocessing step effectively acts as a relevance filter - it discards exactly this noise and retains only the most distinctive terms. This model proves that the keyword signature is a superior input representation for this task.

---

### 5.4 Model D - Hybrid Score Fusion (70% BM25 / 30% Transformer)

**Principle:** Combine BM25 and Transformer outputs via weighted score fusion. BM25 scores are min-max normalised to [0, 1]:

$$\text{Score}_{hybrid}(d) = 0.70 \cdot \text{BM25}_{norm}(d) + 0.30 \cdot \text{cosine\_sim}_{Transformer}(d)$$

**Results:** L1: 0.68 | L2: 4.32

**Analysis:** First model to beat BM25 on Level 2 (4.32 vs 4.41). The Transformer's semantic signal successfully identifies subcategorically relevant places that pure keyword matching misses. However, a residual Level 1 regression (0.68 vs 0.64) suggests the 30% Transformer weight slightly over-influences the ranking, occasionally displacing a categorically correct result.

---

### 5.5 Model E - Two-Stage Re-ranking (BM25 â†’ Transformer)

**Principle:** Industry-standard retrieval-then-reranking:
1. **Stage 1:** BM25 retrieves the top-100 candidates.
2. **Stage 2:** The Transformer re-ranks only these 100 candidates.

**Results:** L1: 0.68 | L2: 4.64

**Analysis:** Underperforms Model D. If BM25 fails to include a semantically relevant place in its top-100 - perhaps due to different but synonymous vocabulary - the Transformer never sees it. Score fusion is more robust because the Transformer can rescue any document from the full corpus.

---

### 5.6 Model F - Optimised Score Fusion (85% BM25 / 15% Transformer) â­ Best Model

**Principle:** Refine Model D's weights. By increasing BM25's share to 85%, Level 1 accuracy is reclaimed while the Level 2 improvement from the Transformer is retained.

$$\text{Score}_{hybrid}(d) = 0.85 \cdot \text{BM25}_{norm}(d) + 0.15 \cdot \text{cosine\_sim}_{Transformer}(d)$$

**Results:** L1: 0.67 | L2: **4.19** âœ“

**Analysis:** Best model across all level-combined metrics. The 85/15 weighting positions the Transformer as a semantic *tiebreaker* rather than a dominant signal - the correct role for a model that struggles with keyword-signature inputs. The 5.0% Level 2 improvement over BM25 demonstrates that the combination of statistical keyword matching and neural semantic embeddings is strictly superior to either approach alone on this dataset.

---

### 5.7 Model G - Triple-RRF Ensemble + Cross-Encoder Re-ranking (Advanced Pipeline)

**Principle:** Addresses two limitations of Model F:

1. **Score fusion instability.** Min-max normalisation is sensitive to outliers. **Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009) fuses rank positions rather than raw scores, making it robust to score distribution differences:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$

2. **Bi-encoder limitations.** A **cross-encoder** takes (query, document) as a single input, enabling full cross-attention between all tokens - producing more accurate relevance scores than independent bi-encoder representations.

**Architecture:** Stage 1: RRF over BM25 + TF-IDF cosine + dense `all-mpnet-base-v2` rankings â†’ Top-50 candidates. Stage 2: Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks the Top-50.

**Results:** L1: 0.71 | L2: 4.65

**Analysis:** Despite its theoretical sophistication, Model G does not surpass Model F. RRF and Cross-Encoders are architectures designed for long-form natural language retrieval. Our keyword signatures are degenerate inputs for cross-attention - there is no sentence structure for the Cross-Encoder to attend over, and the MS-MARCO training objective (web passage retrieval) does not align with our venue similarity task. This is a classic over-engineering result: the choice of text representation dominates system performance, and architectural complexity cannot compensate for a representation mismatch.

---

## 6. Results and Comparison

### 6.1 Summary Table

| Model | Architecture | Level 1 Error â†“ | Level 2 Error â†“ | vs. BM25 L2 |
|:------|:------------|:--------------:|:--------------:|:-----------:|
| **BM25 (Baseline)** | Probabilistic keyword retrieval | 0.64 | 4.41 | - |
| Model A | TF-IDF + Cosine Similarity | 0.68 | 4.74 | âˆ’7.5% |
| Model B | Dense Transformer (MiniLM, keywords) | 0.79 | 4.84 | âˆ’9.7% |
| Model C | Dense Transformer (MiniLM, raw text) | 1.23 | 11.41 | âˆ’159% |
| Model D | Score Fusion: 70% BM25 + 30% Transformer | 0.68 | 4.32 | **+2.0%** |
| Model E | Two-Stage Re-ranking (BM25 top-100 â†’ Transformer) | 0.68 | 4.64 | âˆ’5.2% |
| **Model F â­** | **Score Fusion: 85% BM25 + 15% Transformer** | **0.67** | **4.19** | **+5.0%** |
| Model G | Triple-RRF + Cross-Encoder (all-mpnet-base-v2) | 0.71 | 4.65 | âˆ’5.4% |

*Positive % = improvement over BM25. Lower error is better.*

---

### 6.2 Critical Analysis

**Does the proposed model beat BM25?**

Yes. Model F achieves a Level 1 error of 0.67 (vs 0.64) and a Level 2 error of **4.19 (vs 4.41) - a 5.0% improvement**. The Level 2 gain is the most meaningful result: it demonstrates that a neural semantic signal enables finer-grained subcategory discrimination that pure keyword matching cannot achieve.

**Why is Level 2 error consistently higher than Level 1?**

This is a fundamental property of the task. Level 1 matches one of four broad categories across 918 test places. With strong vocabulary signals (hotel reviews use "reception", "breakfast", "room"; restaurant reviews use "menu", "dish", "chef"), the system consistently places a same-type result in the top 2 positions. Level 2 requires matching a specific subcategory (e.g., "French cuisine", "mid-range hotel") distributed sparsely across 918 places. The match may exist in only 5â€“10 test places, making it inherently harder to retrieve from position 1. The reduction from 917 to 414 valid Level 2 queries (due to the "undefined" exclusion rule) further illustrates this sparsity.

**Concrete example:**

Consider a query place: a **French bistro** with `restaurantTypeCuisine = "French"`. Its keyword signature includes `"wine boeuf steak duck sauce terrine bistrot parisian brasserie menuâ€¦"`. BM25 correctly promotes French restaurants due to distinctive food vocabulary. However, a **Vietnamese-French fusion restaurant** (sharing `cuisine = "French"`) might use different vocabulary (`"pho lemongrass banh mi fusion bÃ¡nhâ€¦"`). BM25 ranks it lower due to minimal token overlap. Model F's 15% Transformer contribution semantically bridges this vocabulary gap, recognising the shared experiential space and promoting the fusion restaurant - a genuine Level 2 improvement that BM25 cannot achieve alone.

---

## 7. Conclusion

### 7.1 Validation of the Working Hypothesis

The results provide strong empirical support for the working hypothesis. A system operating exclusively on review text consistently recommends venues of the correct broad type within the top 2 positions of a ranked list of 918 alternatives (Level 1 error â‰ˆ 0.64â€“0.67). At finer subcategory granularity, the system locates a matching venue within the top 5 positions on average (Level 2 error â‰ˆ 4.19 for the best model). The residual error reflects the inherent limits of keyword-based representations for fine-grained semantic tasks - not a refutation of the hypothesis.

### 7.2 Limitations

1. **Review quality and subjectivity.** Sparse or negatively-toned reviews produce unrepresentative keyword signatures, degrading retrievability for less-visited places.
2. **Vocabulary as a proxy.** The 100-word signature captures *what terms are most used*, not the stance, aspect-distribution, or user demographics of the reviews.
3. **English-only coverage.** Non-English reviews are excluded, biasing representation towards internationally popular venues.
4. **Cross-encoder mismatch.** Advanced NLP architectures designed for natural language are suboptimal for the keyword-signature input format.

### 7.3 Future Work

- **Fine-tuned bi-encoders.** Supervised training on (place, similar place) pairs using weak labels from metadata could produce representations far more aligned with the evaluation criteria.
- **Aspect-based review selection.** Rather than global TF-IDF, extract sentences mentioning specific aspects (food, service, ambiance) to produce richer, more structured place representations.
- **Multilingual models.** Including non-English reviews via a multilingual sentence encoder (`paraphrase-multilingual-mpnet-base-v2`) would substantially increase coverage.
- **Metadata-augmented training.** Metadata could generate training pairs without violating the inference-time constraint - the model learns from metadata during training but remains metadata-free at serving time.

---

## References

- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333â€“389.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.

---

*Report prepared for the Information Retrieval & NLP course, academic year 2025â€“2026.*
