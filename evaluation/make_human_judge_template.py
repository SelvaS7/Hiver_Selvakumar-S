import argparse
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--judge',default='analysis/llm_judge_results.csv')
ap.add_argument('--out',default='analysis/human_judge_review.csv')
args=ap.parse_args()
df=pd.read_csv(args.judge)
# Human gives a single overall accept/reject label for agreement measurement.
out=df[['customer_tweet_id','customer_text','reply']].copy()
out['judge_label']=df.apply(lambda r: 'ACCEPT' if all(pd.to_numeric([r.get(k,float('nan'))],errors='coerce')[0] >= 4 for k in ['correctness','grounding','relevance','helpfulness','style_match','hallucination']) else 'REVIEW',axis=1)
out['human_label']=''
out['human_notes']=''
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(args.out,index=False)
print(f'Wrote {args.out}; fill human_label with ACCEPT or REVIEW for the same rows.')
