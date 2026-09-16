import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.intent_classifier import train
from src.taxonomy import infer_intent
from src.escalation import decide
from src.retriever import HistoricalRetriever
from .baselines import evaluate_baselines
from .response_metrics import basic_response_checks


def metrics(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, pred), "macro_precision": p, "macro_recall": r, "macro_f1": f}


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Hiver support agent on a held-out human-reviewed golden set.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="analysis/evaluation_results.json")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    pairs = pd.read_csv(args.pairs)
    if "review_status" in gold.columns and (gold.review_status != "human_reviewed").any():
        print("WARNING: evaluating an AI-assisted draft; final submission requires human review.")
    required = {"customer_text", "intent", "expected_action"}
    missing = required - set(gold.columns)
    if missing:
        raise SystemExit(f"Golden set missing columns: {sorted(missing)}")

    train_df = pairs.copy()
    train_df["intent"] = train_df.customer_text.map(infer_intent)
    model = joblib.load(args.model) if args.model else train(train_df, evaluate_holdout=False)
    pred_intent = model.predict(gold.customer_text.tolist())

    retriever = HistoricalRetriever(train_df)
    pred_actions, reasons, evidence_scores, replies = [], [], [], []
    from src.response_generator import generate_reply
    for msg, intent in zip(gold.customer_text, pred_intent):
        evidence = retriever.search(msg, k=5)
        action, reason = decide(msg, intent, evidence)
        pred_actions.append(action)
        reasons.append(reason)
        evidence_scores.append(max([e["score"] for e in evidence], default=0.0))
        replies.append(generate_reply(msg, intent, evidence))

    result = metrics(gold.intent, pred_intent)
    result["gold_n"] = len(gold)
    labels = sorted(gold.intent.unique())
    result["confusion_labels"] = labels
    result["confusion_matrix"] = confusion_matrix(gold.intent, pred_intent, labels=labels).tolist()
    result["escalation"] = metrics(gold.expected_action, pred_actions)
    action_labels = ["AUTO-HANDLE", "ESCALATE"]
    result["escalation_confusion_labels"] = action_labels
    result["escalation_confusion_matrix"] = confusion_matrix(gold.expected_action, pred_actions, labels=action_labels).tolist()
    result["avg_top_evidence_score"] = sum(evidence_scores) / len(evidence_scores)
    result["response_checks"] = basic_response_checks(pd.DataFrame({"customer_text": gold.customer_text, "reply": replies}))
    result["baselines"] = evaluate_baselines(train_df, gold).to_dict("records")

    output_gold = gold.copy()
    output_gold["predicted_intent"] = pred_intent
    output_gold["predicted_action"] = pred_actions
    output_gold["action_reason"] = reasons
    output_gold["top_evidence_score"] = evidence_scores
    output_gold["predicted_reply"] = replies
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = out.with_name("golden_predictions.csv")
    output_gold.to_csv(pred_path, index=False)
    result["prediction_file"] = str(pred_path)
    result["llm_judge"] = {"status": "not_run", "reason": "Requires an LLM API key; harness included for local execution."}
    result["human_agreement"] = {"status": "not_run", "reason": "Requires human ratings of the judge subset; template included."}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
