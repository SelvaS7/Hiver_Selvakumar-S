import os, json
from dotenv import load_dotenv
load_dotenv()

RUBRIC = {
    "correctness": "Does the reply address the customer's actual problem without factual errors?",
    "grounding": "Is every policy, promise, timeline, or action supported by the supplied historical evidence?",
    "relevance": "Is the reply focused and directly useful?",
    "helpfulness": "Does it give the customer an actionable next step?",
    "style_match": "Does it reasonably match the historical brand support style?",
    "hallucination": "Does it invent unsupported facts, policies, timelines, guarantees, or account details?",
}


def judge(message, reply, evidence):
    if not os.getenv("OPENAI_API_KEY"):
        return {"status":"skipped","reason":"OPENAI_API_KEY not configured"}
    from openai import OpenAI
    client=OpenAI()
    rubric_text="\n".join(f"{k}: {v}" for k,v in RUBRIC.items())
    ev="\n\n".join(f"Customer: {e['customer_text']}\nBrand: {e['brand_response']}" for e in evidence)
    prompt=f"""Evaluate this customer-support reply. Score each criterion 1-5. For hallucination, 5 means no unsupported claims. Return JSON only.\n\nCustomer: {message}\nReply: {reply}\nHistorical evidence:\n{ev}\n\nRubric:\n{rubric_text}"""
    r=client.responses.create(model=os.getenv("OPENAI_MODEL","gpt-5-mini"),input=prompt)
    text=r.output_text.strip()
    try: return json.loads(text)
    except Exception: return {"raw":text}
