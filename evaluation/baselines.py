import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.taxonomy import infer_intent


def _metrics(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, pred), "macro_precision": p, "macro_recall": r, "macro_f1": f}


def evaluate_baselines(train_df, gold_df):
    rows = []
    Xtr, ytr = train_df.customer_text, train_df.intent
    Xg, yg = gold_df.customer_text, gold_df.intent
    majority = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    rows.append({"baseline": "majority", **_metrics(yg, majority.predict(Xg))})
    rule_pred = gold_df.customer_text.map(infer_intent)
    rows.append({"baseline": "keyword_rules", **_metrics(yg, rule_pred)})
    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]).fit(Xtr, ytr)
    rows.append({"baseline": "tfidf_logreg", **_metrics(yg, tfidf.predict(Xg))})
    return pd.DataFrame(rows)
