import joblib
from typing import Optional
import pandas as pd
from .retriever import HistoricalRetriever
from .response_generator import generate_reply
from .escalation import decide

class SupportAgent:
    def __init__(self, pairs_path: str, model_path: Optional[str] = None):
        self.df = pd.read_csv(pairs_path)
        if model_path:
            self.model = joblib.load(model_path)
        else:
            # Lightweight development classifier; train on weak labels.
            from .taxonomy import infer_intent
            from .intent_classifier import train
            self.df["intent"] = self.df.customer_text.map(infer_intent)
            self.model = train(self.df)
        self.retriever = HistoricalRetriever(self.df)

    def predict(self, message: str):
        intent = self.model.predict([message])[0]
        evidence = self.retriever.search(message, k=5)
        reply = generate_reply(message, intent, evidence)
        action, reason = decide(message, intent, evidence)
        return {"intent": intent, "reply": reply, "action": action, "reason": reason, "evidence": evidence}
