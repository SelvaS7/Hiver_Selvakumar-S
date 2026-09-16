# Decision Log

1. **Selected AppleSupport.** It has one of the largest support volumes in the supplied dataset, improving statistical coverage.
2. **Used direct reply links.** A brand response is treated as evidence only when the dataset's reply relationship links it to the customer tweet.
3. **Focused on English-language interactions.** This keeps the initial taxonomy and evaluation consistent and avoids mixing multilingual policies/styles in one model.
4. **Used a 10-intent taxonomy.** The categories are broad enough for reliable classification while retaining meaningful operational differences.
5. **Kept a separate `other` class.** Forcing ambiguous messages into a specific intent creates false confidence.
6. **Used TF-IDF + logistic regression as the simple baseline.** It is transparent, cheap, fast, and difficult to dismiss as an arbitrary baseline.
7. **Used a majority-class baseline.** It gives a trivial lower bound required by the assignment.
8. **Used retrieval before generation.** Historical responses are supplied as evidence to reduce unsupported answers.
9. **Used a lightweight TF-IDF retriever for reproducibility.** It avoids adding a vector database dependency for a 15k-row capped sample.
10. **Escalated account/transaction/high-risk issues.** These often require information unavailable in public conversation text.
11. **Added an evidence threshold.** Low-similarity retrieval should not be treated as reliable grounding.
12. **Made the LLM optional.** The repository remains runnable without an API key; the final submission should use the configured LLM for the AI-generation evaluation.
13. **Separated proxy labels from the golden set.** Weak labels can accelerate development but must not be presented as hand-labelled human ground truth.
14. **Capped the dataset.** The assignment explicitly expects and encourages a subsample.
15. **Added tests around core behaviours.** The objective is to catch regressions in preprocessing, retrieval, and escalation before submission.
