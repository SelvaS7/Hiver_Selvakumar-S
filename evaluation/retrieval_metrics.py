import math

def recall_at_k(results, expected_ids, k=5):
    found={r.get("customer_tweet_id") for r in results[:k]}
    return int(bool(found & set(expected_ids)))

def reciprocal_rank(results, expected_ids):
    expected=set(expected_ids)
    for i,r in enumerate(results,1):
        if r.get("customer_tweet_id") in expected:
            return 1/i
    return 0.0
