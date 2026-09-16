# API

Run:

```bash
uvicorn app.api:app --reload
```

Then POST JSON to `/predict`:

```json
{"message":"My package says delivered but I never received it"}
```
