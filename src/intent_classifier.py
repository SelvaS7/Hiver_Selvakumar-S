import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .taxonomy import infer_intent


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])


def train(df: pd.DataFrame, evaluate_holdout: bool = True):
    y = df["intent"]
    model = build_model()
    if evaluate_holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            df.customer_text, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(classification_report(y_test, pred, zero_division=0))
        # Refit on all development data for the saved production/evaluation model.
        model.fit(df.customer_text, y)
    else:
        model.fit(df.customer_text, y)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    df["intent"] = df.customer_text.map(infer_intent)
    model = train(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
