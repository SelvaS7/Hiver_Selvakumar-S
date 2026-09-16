"""Compute simple judge-vs-human agreement after exporting judge scores and human scores."""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True,help="CSV with judge_label and human_label columns")
args=ap.parse_args()
df=pd.read_csv(args.input).dropna(subset=["judge_label","human_label"])
print({"n":len(df),"cohen_kappa":cohen_kappa_score(df.human_label,df.judge_label)})
