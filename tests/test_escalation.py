from src.escalation import decide

def test_high_risk_escalates():
    action, _ = decide("I see an unauthorized charge", "payments_billing", [{"score":.9}])
    assert action == "ESCALATE"
