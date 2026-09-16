from src.taxonomy import infer_intent

def test_taxonomy():
    assert infer_intent("My package was not delivered") == "purchase_order"
    assert infer_intent("I need a refund") == "billing_payment"
