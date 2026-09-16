import argparse
from pathlib import Path
import pandas as pd
from .preprocessing import normalize_text, is_probably_english

COLS = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]


def build_pairs(input_path: str, brand: str, output_path: str, limit: int = 15000) -> pd.DataFrame:
    # First pass: IDs of customer tweets directly answered by the selected brand.
    response_ids = set()
    chunks = pd.read_csv(input_path, usecols=COLS, chunksize=100_000)
    for df in chunks:
        brand_rows = df[df.author_id.astype(str).eq(brand)]
        response_ids.update(brand_rows.in_response_to_tweet_id.dropna().map(lambda x: str(int(float(x)))))

    # Second pass: retrieve those customer tweets and their brand responses.
    parents = {}
    responses = {}
    chunks = pd.read_csv(input_path, usecols=COLS, chunksize=100_000)
    for df in chunks:
        ids = df.tweet_id.map(lambda x: str(int(float(x))))
        p = df[ids.isin(response_ids)]
        for row in p.itertuples(index=False):
            parents[str(int(float(row.tweet_id)))] = row
        b = df[df.author_id.astype(str).eq(brand)]
        for row in b.itertuples(index=False):
            parent = str(int(float(row.in_response_to_tweet_id))) if pd.notna(row.in_response_to_tweet_id) else ""
            if parent in response_ids:
                responses[parent] = row

    records = []
    for parent_id, customer in parents.items():
        response = responses.get(parent_id)
        if response is None:
            continue
        if not is_probably_english(customer.text):
            continue
        records.append({
            "customer_tweet_id": parent_id,
            "customer_text": normalize_text(customer.text),
            "customer_created_at": customer.created_at,
            "brand_response_id": str(int(float(response.tweet_id))),
            "brand_response": normalize_text(response.text),
            "brand": brand,
        })
    out = pd.DataFrame(records).drop_duplicates("customer_tweet_id")
    if limit:
        out = out.head(limit)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=15000)
    args = ap.parse_args()
    df = build_pairs(args.input, args.brand, args.output, args.limit)
    print(f"Wrote {len(df):,} paired conversations to {args.output}")

if __name__ == "__main__":
    main()
