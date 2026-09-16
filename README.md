# Hiver SDE Intern — AI Support Agent Take-Home

An evaluation-first customer-support agent built on the Customer Support on Twitter dataset.

## Selected brand

**AppleSupport**. It has a large number of support tweets in the supplied dataset, giving enough historical support interactions for intent discovery, retrieval, and evaluation.

## What the system does

`customer message -> intent classification -> retrieve similar historical AppleSupport cases -> grounded reply -> auto-handle/escalate + reason`

The project also includes two baselines, automated metrics, a draft golden set, an LLM-as-judge harness, human-agreement evaluation, tests, and a short report.

## Important submission note

The included `evaluation/golden_set.csv` is an **AI-assisted draft** created from AppleSupport data. Before submission, review and correct 150–250 examples yourself and set `review_status=human_reviewed`. Hiver explicitly requires hand-labelled examples; this repository does not pretend an automated draft is human-labelled.

The headline metrics in `analysis/pilot_results.json` are **development/proxy results**, not final golden-set results. Run the evaluation again after reviewing the golden set.

## Dataset

The original assignment points to Kaggle's `thoughtvector/customer-support-on-twitter` dataset. Do not commit the full ~500 MB CSV to GitHub.

Expected file:

```text
twcs/twcs.csv
```

The repository includes a small AppleSupport sample so the pipeline can be smoke-tested without the full dataset.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Put the dataset here

```text
twcs/twcs.csv
```

### 3. Build a brand sample

```bash
python -m src.data_loader --input twcs/twcs.csv --brand AppleSupport --output data/apple_pairs.csv --limit 15000
```

### 4. Train the classifier

```bash
python -m src.intent_classifier --input data/apple_pairs.csv --output artifacts/intent_model.joblib
```

### 5. Run the agent locally

```bash
python -m app.main --message "My package says delivered but I never received it"
```

Without an LLM API key, the app uses a safe deterministic fallback response. With an API key it generates a grounded LLM draft.

### 6. Run tests

```bash
pytest -q
```

### 7. Run evaluation

After reviewing `evaluation/golden_set_draft.csv`, save the reviewed file as:

```text
evaluation/golden_set.csv
```

Then:

```bash
python -m evaluation.evaluate --gold evaluation/golden_set.csv --pairs data/apple_pairs.csv
```

## Optional LLM configuration

Copy `.env.example` to `.env` and set the provider credentials. The implementation supports OpenAI through the `OPENAI_API_KEY` environment variable. The model name is configurable with `OPENAI_MODEL`.

## Architecture

- **Pandas**: dataset loading and cleaning
- **scikit-learn**: TF-IDF + logistic-regression intent baseline/model
- **TF-IDF cosine retrieval**: lightweight historical evidence retrieval; no heavyweight vector database is required
- **LLM API**: grounded reply generation and optional judge
- **FastAPI**: optional HTTP endpoint
- **pytest**: regression tests

## Reproducibility

The project deliberately works on a capped sample rather than the full dataset. This follows the assignment's guidance that a subsample is expected and encouraged.

For a final submission, the candidate should:

1. Review all 150–250 golden examples.
2. Run the evaluation on that reviewed set.
3. Paste the resulting metrics into the report.
4. Inspect the five largest failure modes and replace generic examples with real examples from the run.
5. Run the README commands from a clean environment and verify the headline results are reproducible in under 15 minutes.
