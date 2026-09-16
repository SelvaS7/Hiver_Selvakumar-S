import re

HIGH_RISK = [
    r"charged twice", r"fraud", r"unauthorized", r"stolen", r"identity", r"legal", r"lawsuit",
    r"police", r"danger", r"injur", r"medical", r"account hacked", r"data breach"
]


def decide(message: str, intent: str, evidence: list[dict]):
    x = message.lower()
    if any(re.search(p, x) for p in HIGH_RISK):
        return "ESCALATE", "Potential high-risk, account-specific, safety, fraud, or legal issue."
    if intent in {"billing_payment", "apple_id_account", "repair_service"}:
        return "ESCALATE", "The issue may require account-specific verification or a transaction lookup."
    if not evidence or max(e["score"] for e in evidence) < 0.15:
        return "ESCALATE", "Insufficient historical evidence to draft a reliably grounded answer."
    if intent in {"device_technical", "icloud", "app_store", "subscription", "purchase_order", "complaint"}:
        return "AUTO-HANDLE", "A sufficiently similar historical support pattern was retrieved and the issue is not flagged as high-risk."
    return "ESCALATE", "The intent or historical evidence is too ambiguous for unattended handling."
