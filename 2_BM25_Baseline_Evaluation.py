#!/usr/bin/env python
# coding: utf-8

# # Step 2 & 3: Evaluation Setup & Baseline BM25
# 
# In this notebook, we:
# 1. Split the data into 50% Train (to generate queries) and 50% Test (the searchable database).
# 2. Setup Level 1 and Level 2 evaluation functions leveraging metadata (ONLY for evaluation).
# 3. Run the BM25 probabilistic model as our baseline.
# 4. Compute the average Ranking Errors.

# In[ ]:


import pandas as pd
import numpy as np
import random
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')


# ### 1. Load Data & Train/Test Split

# In[ ]:


# Load prepared textual data
df_reviews = pd.read_csv('prepared_reviews.csv')

# Load metadata for evaluation
df_places = pd.read_csv('Tripadvisor.csv', low_memory=False)
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


# ### 2. Evaluation Logic
# 
# We need to find the rank $n$ of the first correct match, from a sorted array of recommendations. The error is $n-1$.
# If no test place matches the requirement, the error is undefined (we yield `None` and ignore it in the average).

# In[ ]:


def eval_level_1(query_typeR, sorted_test_typeR_list):
    """
    Level 1: Match exactly the typeR ('H', 'R', 'A', 'AP')
    sorted_test_typeR_list is an ordered list of typeR from the ranked system output.
    """
    if pd.isna(query_typeR):
        return None
        
    if query_typeR not in sorted_test_typeR_list:
        return None # Undefined error
        
    # find the index (our system index starts at 0, which corresponds to position 1, so error is precisely the index)
    for i, t in enumerate(sorted_test_typeR_list):
        if t == query_typeR:
            return i # i = n-1
            
    return None

def extract_subcategories(row):
    """Helper to gather all subcategories into a single flat list/set of strings"""
    cats = []
    cols = ['activiteSubType', 'restaurantType', 'restaurantTypeCuisine', 'priceRange']
    for c in cols:
        val = row[c]
        if pd.notna(val):
            # Splitting by comma if it's a list string
            parts = str(val).split(',')
            for p in parts:
                cats.append(p.strip().lower())
    return set(cats)

# Pre-calculate test subcategories to speed up testing
test_subcats = [extract_subcategories(row) for _, row in test_df.iterrows()]
test_df['subcategories'] = test_subcats

def eval_level_2(query_subcats, sorted_test_indices):
    """
    Level 2: At least one metadata category overlaps.
    sorted_test_indices: indices mapping back to test_df.
    """
    if not query_subcats:
        return None
        
    for i, test_idx in enumerate(sorted_test_indices):
        test_sub = test_df.loc[test_idx, 'subcategories']
        if not test_sub:
            continue
        # Check intersection
        if len(query_subcats.intersection(test_sub)) > 0:
            return i
            
    return None


# ### 3. BM25 Baseline Setup

# In[ ]:


print("Tokenizing test documents for BM25...")
corpus = test_df['top_100_words'].fillna('').astype(str).tolist()
tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
print("BM25 Indexed")


# ### 4. Running the Benchmark

# In[ ]:


import time

lvl1_errors = []
lvl2_errors = []

start_time = time.time()
print(f"Evaluating {len(train_df)} queries...")

for idx, row in train_df.iterrows():
    query_text = str(row['top_100_words'])
    if not query_text:
        continue
        
    tokenized_query = query_text.split(" ")
    
    # Get fast scores array: index matches the index in test_df
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get argsort descending
    ranked_indices = np.argsort(doc_scores)[::-1]
    
    # Build Ranked list of typeR for Level 1
    test_typeR_list = test_df['typeR'].values[ranked_indices].tolist()
    
    # --- EVAL LEVEL 1 ---
    err_1 = eval_level_1(row['typeR'], test_typeR_list)
    if err_1 is not None:
        lvl1_errors.append(err_1)
        
    # --- EVAL LEVEL 2 ---
    query_subcats = extract_subcategories(row)
    err_2 = eval_level_2(query_subcats, ranked_indices)
    if err_2 is not None:
        lvl2_errors.append(err_2)

print(f"Evaluation done in {time.time() - start_time:.2f}s")
print("-" * 30)
print(f"BM25 Average Ranking Error Level 1: {np.mean(lvl1_errors):.2f} (Computed on {len(lvl1_errors)} valid queries)")
print(f"BM25 Average Ranking Error Level 2: {np.mean(lvl2_errors):.2f} (Computed on {len(lvl2_errors)} valid queries)")

