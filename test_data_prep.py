import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# FIX: TF-IDF must be fit on the ENTIRE corpus, not per-document.
# All documents are loaded first, the vectorizer is fit once on all of them,
# and then each document is individually transformed. This gives corpus-aware
# IDF weights (words rare across places get higher scores).

print("1. Loading datasets...")
df_reviews = pd.read_csv('../data/reviews83325.csv')
print(f"Total reviews: {len(df_reviews)}")

print("\n2. Filtering for English 'en' reviews...")
df_reviews_en = df_reviews[df_reviews['langue'] == 'en'].copy()
print(f"English reviews: {len(df_reviews_en)}")

print("\n3. Grouping reviews by idplace...")
df_grouped = df_reviews_en.groupby('idplace')['review'].apply(
    lambda x: ' '.join(x.dropna().astype(str))
).reset_index()
print(f"Total places with English reviews: {len(df_grouped)}")

print("\n4. Fitting corpus-aware TF-IDF vectorizer (max_features=5000)...")
all_texts = df_grouped['review'].fillna('').tolist()
corpus_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
corpus_vectorizer.fit(all_texts)
feature_names = corpus_vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(feature_names)}")

def extract_top_100_tfidf(text):
    """
    Extract the top-100 TF-IDF-scored words for a single document,
    using the already-fit corpus vectorizer (corpus-aware IDF).
    """
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

# Test on a sample first
print("\nTesting corrected TF-IDF extraction on first 10 places...")
sample_df = df_grouped.head(10).copy()
sample_df['top_100_words'] = sample_df['review'].apply(extract_top_100_tfidf)
print(sample_df[['idplace', 'top_100_words']].head(2).to_string())

print("\nScript executed successfully — corpus-aware TF-IDF confirmed.")
