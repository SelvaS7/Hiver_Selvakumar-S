import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalRetriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=100_000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.df.customer_text)

    def search(self, query: str, k: int = 5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(-scores)[:k]
        return [
            {"customer_text": self.df.iloc[i].customer_text,
             "brand_response": self.df.iloc[i].brand_response,
             "score": float(scores[i])}
            for i in idx
        ]
