# Final Submission Runbook

## Already completed in this repository

- 200 AI-assisted draft examples in `evaluation/golden_set.csv`; human review is still required before submission.
- The draft evaluation examples are sampled from the development corpus; create a genuinely held-out human-reviewed set before final submission.
- Three intent comparisons: majority, keyword rules, TF-IDF/logistic regression.
- Agent evaluation: intent + escalation metrics, confusion matrices, prediction export.
- LLM-as-judge rubric and scripts.
- Human-agreement script and review template.
- 4 automated regression tests.
- Six-page-style report content in `report/report.md` and 15-entry decision log.

## Required final commands

```bash
pip install -r requirements.txt
python -m pytest -q
python -m evaluation.evaluate --gold evaluation/golden_set.csv --pairs data/apple_pairs.csv --out analysis/evaluation_results.json
```

The last command writes `analysis/golden_predictions.csv` and `analysis/evaluation_results.json`.

## Optional but required for the complete judge evidence requested by Hiver

Set the API key **locally** (do not paste it into chat or commit it):

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
python -m evaluation.run_judge --n 30
python -m evaluation.make_human_judge_template
```

Open `analysis/human_judge_review.csv`, fill `human_label` with `ACCEPT` or `REVIEW` for each sampled row, then:

```bash
python -m evaluation.human_agreement --input analysis/human_judge_review.csv
```

Copy the resulting judge averages and Cohen's kappa into `report/report.md` before submitting if you run this step.

## Important

Do not claim the LLM judge or human agreement was run unless those commands actually produced results. The packaged report explicitly marks them as pending external credentials/human ratings.
