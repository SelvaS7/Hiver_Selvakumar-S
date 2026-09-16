import pandas as pd
from src.retriever import HistoricalRetriever

def test_retrieval():
    df=pd.DataFrame({"customer_text":["package is late","refund missing"],"brand_response":["check tracking","check refund"]})
    r=HistoricalRetriever(df).search("my package is late",k=1)
    assert r and "package" in r[0]["customer_text"]
