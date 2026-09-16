# Hiver SDE Intern — Take-Home Report

## 1. Problem framing

The agent is designed for **AppleSupport** customer-support conversations. Good performance means correct intent classification, replies grounded in historical AppleSupport resolutions, avoidance of unsupported claims, and escalation of account-specific or high-risk issues.

## 2. Data and brand selection

The supplied Customer Support on Twitter dataset contains 2,811,774 tweets. The supplied brand analysis lists **106,860 AppleSupport tweets**. A capped 15,000-row direct customer-to-AppleSupport response-pair corpus is included for development and reproducibility. The raw 500+ MB CSV is not committed to GitHub.

## 3. System

`message -> TF-IDF/logistic intent -> historical TF-IDF retrieval -> grounded reply -> auto-handle/escalate + reason`

The taxonomy contains 10 operational intents: `device_technical`, `apple_id_account`, `icloud`, `app_store`, `billing_payment`, `subscription`, `purchase_order`, `repair_service`, `complaint`, and `other`.

## 4. Development evaluation

A 200-example AppleSupport evaluation draft was generated to accelerate development. **It is not yet a human-labelled final golden set.** The current metrics are therefore proxy/development metrics only and must not be presented as final assignment results.

| System / baseline | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 41.0% | see `analysis/evaluation_results.json` |
| Keyword rules | 100.0%* | 1.000* |
| TF-IDF + logistic regression | 95.0%* | 0.971* |

\*These numbers are inflated because the current draft labels were bootstrapped using the same keyword taxonomy. They are **not valid evidence of generalisation**. A fresh human-labelled golden set is required.

## 5. Golden evaluation set

`evaluation/golden_set.csv` contains 200 AI-assisted candidate examples with `review_status=needs_human_review`. The candidate must review and correct the intent and expected action fields before submission, then change the status only for examples actually reviewed.

## 6. Judge protocol

The repository contains the LLM-as-judge and human-agreement harness. Run the judge only after the golden set is genuinely reviewed. Do not claim judge or human-agreement results unless they were actually produced.

## 7. What is misleading about the headline number?

The current 95% intent accuracy is misleading because the draft ground truth was generated with the same rules used during development. This creates label leakage. The number should therefore be treated as a pipeline smoke-test, not a trustworthy performance estimate. The final headline must come from independently human-labelled examples.

## 8. Next steps

1. Human-review 150–250 AppleSupport examples.
2. Re-run evaluation and replace all proxy numbers.
3. Run the LLM judge on a fixed sample and collect human agreement ratings.
4. Update failure analysis from the real confusion matrix.
5. Commit the final reproducible results.

## 9. Reproducibility

```bash
pip install -r requirements.txt
python -m src.data_loader --input twcs/twcs.csv --brand AppleSupport --output data/apple_pairs.csv --limit 15000
python -m src.intent_classifier --input data/apple_pairs.csv --output artifacts/intent_model.joblib
python -m pytest -q
python -m app.main --message "My iPhone is not connecting to Wi-Fi"
python -m evaluation.evaluate --gold evaluation/golden_set.csv --pairs data/apple_pairs.csv --out analysis/evaluation_results.json
```
