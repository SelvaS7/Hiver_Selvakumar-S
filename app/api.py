from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import SupportAgent

app = FastAPI(title='AppleSupport Support Agent')
_agent = None

class Request(BaseModel):
    message: str

def get_agent():
    global _agent
    if _agent is None:
        _agent = SupportAgent('data/apple_pairs.csv')
    return _agent

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(req: Request):
    return get_agent().predict(req.message)
