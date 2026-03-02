import os
import sys
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings('ignore')

# Set working directory to the notebooks folder as the notebooks expect relative paths to ../data/
# Actually, since we are in a single .py file, we should handle paths carefully.
# We'll assume the script is run from the project root.

def run_data_preparation():
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    
    # Use absolute paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # parent of notebooks/
    
    reviews_path = os.path.join(project_root, 'data', 'reviews83325.csv')
    places_path = os.path.join(project_root, 'data', 'Tripadvisor.csv')
    output_path = os.path.join(project_root, 'prepared_reviews.csv')

    df_reviews = pd.read_csv(reviews_path)
    df_places = pd.read_csv(places_path)

    print(f"Total reviews: {len(df_reviews)}")
    print(f"Total places: {len(df_places)}")

    # Filter English Reviews
    df_reviews_en = df_reviews[df_reviews['langue'] == 'en'].copy()
    print(f"English reviews: {len(df_reviews_en)}")

    # Group Reviews by Place
    df_grouped = df_reviews_en.groupby('idplace')['review'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
    print(f"Places with English reviews: {len(df_grouped)}")

    # TF-IDF Normalization (Top 100 Words) — Corpus-Aware
    print("Normalizing reviews using TF-IDF (extracting top 100 words)...")
    corpus_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    all_texts = df_grouped['review'].fillna('').tolist()
    corpus_vectorizer.fit(all_texts)
    feature_names = corpus_vectorizer.get_feature_names_out()

    def extract_top_100_tfidf(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        try:
            X = corpus_vectorizer.transform([text])
            scores = list(zip(feature_names, X.toarray().ravel()))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top_words = [w for w, s in sorted_scores if s > 0][:100]
            return " ".join(top_words) if top_words else text
        except ValueError:
            return text

    df_grouped['top_100_words'] = df_grouped['review'].apply(extract_top_100_tfidf)
    print(f"Sample keywords for place 0: {df_grouped['top_100_words'].iloc[0][:80]}...")

    df_grouped.to_csv(output_path, index=False)
    print(f"Cleaned reviews dataset saved as '{output_path}'")
    return df_grouped

def eval_level_1(query_typeR, sorted_test_typeR_list):
    """Level 1 error: rank of first match on typeR (H/R/A/AP). Error = rank index."""
    if pd.isna(query_typeR) or query_typeR not in sorted_test_typeR_list:
        return None
    for i, t in enumerate(sorted_test_typeR_list):
        if t == query_typeR:
            return i
    return None

def extract_subcategories(row):
    """
    Type-aware subcategory extraction (FIX: avoids spurious cross-type matches).

    Per assignment spec:
      H  (Hotel)             -> priceRange
      R  (Restaurant)        -> restaurantType, restaurantTypeCuisine
      A / AP (Attraction)    -> activiteSubType
    """
    cats = set()
    type_r = str(row.get('typeR', '')).strip()
    if type_r == 'H':
        val = row.get('priceRange', '')
        if pd.notna(val) and str(val).strip():
            for p in str(val).split(','):
                cats.add(p.strip().lower())
    elif type_r == 'R':
        for col in ['restaurantType', 'restaurantTypeCuisine']:
            val = row.get(col, '')
            if pd.notna(val) and str(val).strip():
                for p in str(val).split(','):
                    cats.add(p.strip().lower())
    elif type_r in ('A', 'AP'):
        val = row.get('activiteSubType', '')
        if pd.notna(val) and str(val).strip():
            for p in str(val).split(','):
                cats.add(p.strip().lower())
    return cats

def eval_level_2(query_subcats, sorted_test_indices, test_subcats_list):
    """Level 2 error: rank of first test place sharing at least one subcategory."""
    if not query_subcats:
        return None
    for i, test_idx in enumerate(sorted_test_indices):
        if query_subcats.intersection(test_subcats_list[test_idx]):
            return i
    return None

def run_bm25_baseline():
    print("\n" + "="*50)
    print("STEP 2: BM25 BASELINE EVALUATION")
    print("="*50)
    
    # Standard path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    prepared_path = os.path.join(project_root, 'prepared_reviews.csv')
    tripadvisor_path = os.path.join(project_root, 'data', 'Tripadvisor.csv')

    # Load prepared textual data
    df_reviews = pd.read_csv(prepared_path)
    
    # Load metadata for evaluation
    df_places = pd.read_csv(tripadvisor_path, low_memory=False)
        
    eval_cols = ['id', 'typeR', 'activiteSubType', 'restaurantType', 'restaurantTypeCuisine', 'priceRange']
    df_places = df_places[eval_cols].copy()

    # Merge to have text + metadata in one dataframe for splitting
    df_merged = pd.merge(df_reviews, df_places, left_on='idplace', right_on='id', how='inner')
    print(f"Total valid places with text and metadata: {len(df_merged)}")

    # Split: 50% Train (Queries) | 50% Test (Search Base)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df_merged))
    split_idx = len(df_merged) // 2

    train_df = df_merged.iloc[shuffled_indices[:split_idx]].copy()
    test_df = df_merged.iloc[shuffled_indices[split_idx:]].copy().reset_index(drop=True)

    print(f"Train setup (Queries): {len(train_df)} places")
    print(f"Test setup (Corpus): {len(test_df)} places")

    # Pre-calculate test subcategories
    test_subcats_list = [extract_subcategories(row) for _, row in test_df.iterrows()]
    test_df['subcategories'] = test_subcats_list

    # BM25 Setup
    print("Tokenizing test documents for BM25...")
    corpus = test_df['top_100_words'].fillna('').astype(str).tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 Indexed")

    # Running the Benchmark
    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()
    print(f"Evaluating {len(train_df)} queries...")

    for idx, row in train_df.iterrows():
        query_text = str(row['top_100_words']).strip()
        if not query_text or query_text == 'nan':
            continue
        tokenized_query = query_text.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(doc_scores)[::-1]
        test_typeR_list = test_df['typeR'].values[ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"BM25 Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"BM25 Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")
    
    return train_df, test_df, test_subcats_list

def run_tfidf_similarity(train_df, test_df, test_subcats_list):
    print("\n" + "="*50)
    print("STEP 3: TF-IDF SIMILARITY MODEL")
    print("="*50)
    
    # Model Pipeline: TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    train_texts = train_df['top_100_words'].fillna('').tolist()
    test_texts  = test_df['top_100_words'].fillna('').tolist()

    # fit_transform on TRAIN (queries)
    train_vectors = vectorizer.fit_transform(train_texts)
    # transform TEST (corpus)
    test_vectors  = vectorizer.transform(test_texts)

    print(f"Vocabulary size (from train): {len(vectorizer.vocabulary_)}")
    print(f"Vectorized {train_vectors.shape[0]} Queries and {test_vectors.shape[0]} Test Docs.")

    # Running the Benchmark
    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()
    print(f"Evaluating {len(train_df)} queries...")

    similarities = cosine_similarity(train_vectors, test_vectors)
    test_type_array = test_df['typeR'].values

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        sims = similarities[i]
        ranked_indices = np.argsort(sims)[::-1]
        test_typeR_list = test_type_array[ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"Model A (TF-IDF + Cosine Sim) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model A (TF-IDF + Cosine Sim) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")

def run_transformer_embedding(train_df, test_df, test_subcats_list):
    print("\n" + "="*50)
    print("STEP 4: TRANSFORMER EMBEDDING MODEL")
    print("="*50)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_texts = test_df['top_100_words'].fillna('').tolist()
    train_texts = train_df['top_100_words'].fillna('').tolist()

    print("Encoding 50% Test set...")
    test_vectors = model.encode(test_texts, show_progress_bar=True)
    print("Encoding 50% Train set (Queries)...")
    train_vectors = model.encode(train_texts, show_progress_bar=True)

    # Running the Benchmark
    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()
    print(f"Evaluating {len(train_df)} queries...")

    similarities = cosine_similarity(train_vectors, test_vectors)
    test_type_array = test_df['typeR'].values

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        sims = similarities[i]
        ranked_indices = np.argsort(sims)[::-1]
        test_typeR_list = test_type_array[ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"Model B (Transformer) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model B (Transformer) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")
    return similarities # Return similarities for potential fusion later

def run_contextual_transformer(train_df, test_df, test_subcats_list):
    print("\n" + "="*50)
    print("STEP 6: MODEL C - CONTEXTUAL TRANSFORMER (RAW TEXT)")
    print("="*50)
    print("Insight: Using raw review context instead of keywords...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We take a sample of the raw text (first 500 chars) to maintain context without hitting length limits
    test_texts = [str(t)[:500] for t in test_df['review'].fillna('').tolist()]
    train_texts = [str(t)[:500] for t in train_df['review'].fillna('').tolist()]

    print("Encoding Test set (Contextual)...")
    test_vectors = model.encode(test_texts, show_progress_bar=True)
    print("Encoding Train set (Contextual)...")
    train_vectors = model.encode(train_texts, show_progress_bar=True)

    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()
    
    similarities = cosine_similarity(train_vectors, test_vectors)
    test_type_array = test_df['typeR'].values

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        sims = similarities[i]
        ranked_indices = np.argsort(sims)[::-1]
        test_typeR_list = test_type_array[ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"Model C (Contextual Transformer) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model C (Contextual Transformer) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")
    return similarities

def run_hybrid_model(train_df, test_df, test_subcats_list):
    print("\n" + "="*50)
    print("STEP 7: MODEL D - HYBRID MODEL (BM25 + TRANSFORMER)")
    print("="*50)
    
    # 1. Get BM25 scores
    corpus = test_df['top_100_words'].fillna('').astype(str).tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. Get Transformer (Contextual) similarities
    # Note: For efficiency, we would reuse vectors, but we'll re-run here for clarity in this self-contained script
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_texts = [str(t)[:500] for t in test_df['review'].fillna('').tolist()]
    train_texts = [str(t)[:500] for t in train_df['review'].fillna('').tolist()]
    test_vectors = model.encode(test_texts, show_progress_bar=False)
    train_vectors = model.encode(train_texts, show_progress_bar=False)
    trans_similarities = cosine_similarity(train_vectors, test_vectors)

    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()
    print("Combining scores and evaluating...")

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        query_text = str(row['top_100_words']).strip()
        if not query_text or query_text == 'nan':
            continue
            
        # BM25 scores
        tokenized_query = query_text.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1] for fusion
        if np.max(bm25_scores) > 0:
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        else:
            bm25_norm = bm25_scores
            
        # Transformer scores are already cosine sims (mostly [0, 1] for these models)
        trans_scores = trans_similarities[i]
        
        # HYBRID SCORE: 70% BM25 (Precision) + 30% Transformer (Recall/Semantics)
        # We can tune these weights
        hybrid_scores = 0.7 * bm25_norm + 0.3 * trans_scores
        
        ranked_indices = np.argsort(hybrid_scores)[::-1]
        test_typeR_list = test_df['typeR'].values[ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"Model D (Hybrid) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model D (Hybrid) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")

def run_two_stage_reranking(train_df, test_df, test_subcats_list):
    print("\n" + "="*50)
    print("STEP 8: MODEL E - TWO-STAGE RE-RANKING (BM25 -> TRANSFORMER)")
    print("="*50)
    print("Strategy: BM25 retrieves top 100, Transformer re-ranks them.")
    
    # 1. BM25 Setup
    corpus = test_df['top_100_words'].fillna('').astype(str).tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. Transformer Setup (Using Keywords as it was more stable than raw text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_texts = test_df['top_100_words'].fillna('').tolist()
    train_texts = train_df['top_100_words'].fillna('').tolist()
    test_vectors = model.encode(test_texts, show_progress_bar=False)
    train_vectors = model.encode(train_texts, show_progress_bar=False)

    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        query_text = str(row['top_100_words']).strip()
        if not query_text or query_text == 'nan':
            continue
            
        # STAGE 1: BM25 retrieval
        tokenized_query = query_text.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[-100:][::-1] # Top 100
        
        # STAGE 2: Transformer re-ranking on the Top K
        query_vec = train_vectors[i].reshape(1, -1)
        candidate_vecs = test_vectors[top_k_indices]
        rerank_sims = cosine_similarity(query_vec, candidate_vecs).flatten()
        
        # Re-sort the top_k_indices based on rerank_sims
        reranked_top_indices = top_k_indices[np.argsort(rerank_sims)[::-1]]
        
        # Append the rest of the indices (those not in top 100) at the end to keep the full list
        all_indices = list(reranked_top_indices)
        remaining_indices = [idx for idx in np.argsort(bm25_scores)[::-1] if idx not in top_k_indices]
        full_ranked_indices = all_indices + remaining_indices
        
        test_typeR_list = test_df['typeR'].values[full_ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, full_ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"Model E (Two-Stage) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model E (Two-Stage) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")

def run_optimized_hybrid(train_df, test_df, test_subcats_list):
    print("\n" + "="*50)
    print("STEP 9: MODEL F - OPTIMIZED SCORE FUSION (0.85 BM25 / 0.15 TRANSFORMER)")
    print("="*50)
    print("Strategy: Tilt weights towards BM25 to regain Level 1 accuracy while keeping Level 2 gains.")
    
    # 1. BM25 Setup
    corpus = test_df['top_100_words'].fillna('').astype(str).tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 2. Transformer Setup
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_texts = test_df['top_100_words'].fillna('').tolist()
    train_texts = train_df['top_100_words'].fillna('').tolist()
    test_vectors = model.encode(test_texts, show_progress_bar=False)
    train_vectors = model.encode(train_texts, show_progress_bar=False)
    trans_similarities = cosine_similarity(train_vectors, test_vectors)

    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()

    for i in range(len(train_df)):
        row = train_df.iloc[i]
        query_text = str(row['top_100_words']).strip()
        if not query_text or query_text == 'nan':
            continue
            
        tokenized_query = query_text.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Normalize BM25
        if np.max(bm25_scores) > 0:
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
        else:
            bm25_norm = bm25_scores
            
        trans_scores = trans_similarities[i]
        
        # FINAL OPTIMIZED WEIGHTS: 0.85 BM25 / 0.15 Transformer
        hybrid_scores = 0.85 * bm25_norm + 0.15 * trans_scores
        
        ranked_indices = np.argsort(hybrid_scores)[::-1]
        test_typeR_list = test_df['typeR'].values[ranked_indices].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None: lvl1_errors.append(err_1)
            
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, ranked_indices, test_subcats_list)
        if err_2 is not None: lvl2_errors.append(err_2)

    print(f"Evaluation done in {time.time() - start_time:.2f}s")
    print(f"Model F (Optimized Hybrid) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model F (Optimized Hybrid) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")

# ==============================================================================
# ADVANCED PIPELINE: Triple-RRF + Cross-Encoder Re-ranking (Model G)
# ==============================================================================

def reciprocal_rank_fusion(*rank_lists, k=60):
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).
    
    Given N rank lists (each a list of document indices sorted by relevance),
    compute for each document:
        RRF_score(d) = sum_{r in rankers} 1 / (k + rank_r(d))
    
    k=60 is the standard smoothing constant from the original paper.
    Returns indices sorted by descending RRF score.
    """
    rrf_scores = {}
    for rank_list in rank_lists:
        for rank, doc_idx in enumerate(rank_list):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    return sorted_docs

def run_advanced_pipeline(train_df, test_df, test_subcats_list):
    print("\n" + "="*60)
    print("STEP 11: MODEL G — TRIPLE-RRF + CROSS-ENCODER RE-RANKING")
    print("="*60)
    
    # ====================================================================
    # STAGE 1a: BM25 Retrieval (already proven: best for Level 1)
    # ====================================================================
    print("[Stage 1a] Building BM25 index...")
    corpus_kw = test_df['top_100_words'].fillna('').astype(str).tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus_kw]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # ====================================================================
    # STAGE 1b: TF-IDF Cosine Similarity
    # ====================================================================
    print("[Stage 1b] Building TF-IDF vectors...")
    tfidf_vectorizer = TfidfVectorizer()
    train_kw = train_df['top_100_words'].fillna('').tolist()
    test_kw  = test_df['top_100_words'].fillna('').tolist()
    tfidf_train = tfidf_vectorizer.fit_transform(train_kw)
    tfidf_test  = tfidf_vectorizer.transform(test_kw)
    tfidf_sims  = cosine_similarity(tfidf_train, tfidf_test)
    
    # ====================================================================
    # STAGE 1c: Dense Transformer (upgraded to all-mpnet-base-v2)
    # ====================================================================
    print("[Stage 1c] Encoding with all-mpnet-base-v2 (768-dim, 12 layers)...")
    bi_encoder = SentenceTransformer('all-mpnet-base-v2')
    train_vecs = bi_encoder.encode(train_kw, show_progress_bar=True, batch_size=64)
    test_vecs  = bi_encoder.encode(test_kw, show_progress_bar=True, batch_size=64)
    dense_sims = cosine_similarity(train_vecs, test_vecs)
    
    # ====================================================================
    # STAGE 2: Cross-Encoder (for re-ranking top candidates)
    # ====================================================================
    print("[Stage 2] Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    TOP_K = 50  # Number of RRF candidates to re-rank
    
    lvl1_errors = []
    lvl2_errors = []
    start_time = time.time()
    n_queries = len(train_df)
    print(f"Evaluating {n_queries} queries (RRF + Cross-Encoder re-rank top {TOP_K})...")
    
    for i in range(n_queries):
        row = train_df.iloc[i]
        query_text = str(row['top_100_words']).strip()
        if not query_text or query_text == 'nan':
            continue
        
        # --- Get rank lists from each retriever ---
        # BM25 ranks
        bm25_scores = bm25.get_scores(query_text.split(" "))
        bm25_ranked = np.argsort(bm25_scores)[::-1].tolist()
        
        # TF-IDF ranks
        tfidf_ranked = np.argsort(tfidf_sims[i])[::-1].tolist()
        
        # Dense (mpnet) ranks
        dense_ranked = np.argsort(dense_sims[i])[::-1].tolist()
        
        # --- Reciprocal Rank Fusion ---
        rrf_ranked = reciprocal_rank_fusion(bm25_ranked, tfidf_ranked, dense_ranked, k=60)
        
        # --- Cross-Encoder re-ranking of Top K ---
        top_k_indices = rrf_ranked[:TOP_K]
        
        # Build (query, candidate) pairs for cross-encoder
        pairs = [(query_text, test_kw[idx]) for idx in top_k_indices]
        ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        
        # Re-sort the top K by cross-encoder score
        reranked_top = [top_k_indices[j] for j in np.argsort(ce_scores)[::-1]]
        
        # Append remaining docs (outside top K) in original RRF order
        remaining = rrf_ranked[TOP_K:]
        full_ranked = reranked_top + remaining
        
        # --- Evaluation ---
        test_typeR_list = test_df['typeR'].values[full_ranked].tolist()
        
        err_1 = eval_level_1(row['typeR'], test_typeR_list)
        if err_1 is not None:
            lvl1_errors.append(err_1)
        
        query_subcats = extract_subcategories(row)
        err_2 = eval_level_2(query_subcats, full_ranked, test_subcats_list)
        if err_2 is not None:
            lvl2_errors.append(err_2)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  ... processed {i+1}/{n_queries} queries")

    elapsed = time.time() - start_time
    print(f"\nEvaluation done in {elapsed:.2f}s")
    print("-" * 40)
    print(f"Model G (Triple-RRF + Cross-Encoder) Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f}")
    print(f"Model G (Triple-RRF + Cross-Encoder) Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f}")
    print(f"  (Computed on {len(lvl1_errors)} L1 queries, {len(lvl2_errors)} L2 queries)")

if __name__ == "__main__":
    # Run Step 1
    run_data_preparation()
    
    # Run Step 2 (BM25)
    train_df, test_df, test_subcats_list = run_bm25_baseline()
    
    # Run Step 3 (TF-IDF)
    run_tfidf_similarity(train_df, test_df, test_subcats_list)
    
    # Run Step 4 (Transformer - Keywords)
    run_transformer_embedding(train_df, test_df, test_subcats_list)

    # Run Step 6 (Transformer - Raw Text)
    run_contextual_transformer(train_df, test_df, test_subcats_list)

    # Run Step 7 (Hybrid Model)
    run_hybrid_model(train_df, test_df, test_subcats_list)

    # Run Step 8 (Two-Stage Re-ranking)
    run_two_stage_reranking(train_df, test_df, test_subcats_list)

    # Run Step 9 (Optimized Hybrid)
    run_optimized_hybrid(train_df, test_df, test_subcats_list)

    # Run Step 11 (Advanced Pipeline: Triple-RRF + Cross-Encoder)
    run_advanced_pipeline(train_df, test_df, test_subcats_list)
