import argparse
import json
from src.pipeline import SupportAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--pairs", default="data/apple_pairs.csv")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    agent = SupportAgent(args.pairs, args.model)
    print(json.dumps(agent.predict(args.message), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
