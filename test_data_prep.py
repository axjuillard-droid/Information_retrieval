import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("1. Loading datasets...")
df_reviews = pd.read_csv('reviews83325.csv')
print(f"Total reviews: {len(df_reviews)}")

print("\n2. Filtering for English 'en' reviews...")
df_reviews_en = df_reviews[df_reviews['langue'] == 'en'].copy()
print(f"English reviews: {len(df_reviews_en)}")

print("\n3. Grouping reviews by idplace...")
df_grouped = df_reviews_en.groupby('idplace')['review'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
print(f"Total places with English reviews: {len(df_grouped)}")

print("\n4. Extracting Top 100 words based on TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

def extract_top_100_tfidf(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    try:
        X = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return " ".join([word for word, score in sorted_scores])
    except ValueError:
        pass
    return text

# Apply to a sample first to check time
print("Testing TF-IDF extraction on first 10 places...")
sample_df = df_grouped.head(10).copy()
sample_df['top_100_words'] = sample_df['review'].apply(extract_top_100_tfidf)
print(sample_df[['idplace', 'top_100_words']].head(2))

print("\nScript executed successfully!")
