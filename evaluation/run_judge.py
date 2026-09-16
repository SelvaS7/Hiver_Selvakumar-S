"""Run the LLM-as-judge on a reproducible sample of golden predictions.
Requires OPENAI_API_KEY in the local environment. Never paste the key into the repo or chat.
"""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
from src.retriever import HistoricalRetriever
from .llm_judge import judge

ap=argparse.ArgumentParser()
ap.add_argument('--predictions',default='analysis/golden_predictions.csv')
ap.add_argument('--pairs',default='data/amazon_pairs.csv')
ap.add_argument('--n',type=int,default=30)
ap.add_argument('--seed',type=int,default=42)
ap.add_argument('--out',default='analysis/llm_judge_results.csv')
args=ap.parse_args()
if not os.getenv('OPENAI_API_KEY'):
    raise SystemExit('OPENAI_API_KEY is not configured. Set it locally, then rerun this command.')

gold=pd.read_csv(args.predictions).sample(n=min(args.n,200),random_state=args.seed)
pairs=pd.read_csv(args.pairs)
retriever=HistoricalRetriever(pairs)
rows=[]
for _,r in gold.iterrows():
    evidence=retriever.search(r.customer_text,k=5)
    scores=judge(r.customer_text,r.predicted_reply,evidence)
    row={'customer_tweet_id':r.customer_tweet_id,'customer_text':r.customer_text,'reply':r.predicted_reply}
    row.update(scores)
    rows.append(row)
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out,index=False)
print(f'Wrote {out}')
