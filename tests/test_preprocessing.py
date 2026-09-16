from src.preprocessing import normalize_text

def test_normalize():
    assert "URL" in normalize_text("@AmazonHelp see https://example.com")
