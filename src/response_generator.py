import os
from dotenv import load_dotenv
load_dotenv()


def fallback_reply(message: str, intent: str, evidence: list[dict]) -> str:
    if evidence:
        return "Thanks for reaching out. We’re sorry you’re having trouble. Please share the relevant order/account details in a private message so the support team can check this for you."
    return "Thanks for reaching out. Please provide a few more details about the issue so the support team can help."


def generate_reply(message: str, intent: str, evidence: list[dict]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return fallback_reply(message, intent, evidence)
    from openai import OpenAI
    client = OpenAI(api_key=key)
    examples = "\n\n".join(
        f"Customer: {e['customer_text']}\nBrand response: {e['brand_response']}\nSimilarity: {e['score']:.3f}"
        for e in evidence
    )
    prompt = f"""You are an Apple customer-support drafting assistant.\n\nCustomer message:\n{message}\n\nPredicted intent:\n{intent}\n\nHistorical evidence from AppleSupport:\n{examples}\n\nWrite one concise reply. Match the historical support behaviour and do not invent policies, refunds, timelines, guarantees, or account facts. If the evidence is insufficient, ask for the minimum information needed. Do not mention the retrieval process or that you are an AI."""
    response = client.responses.create(model=os.getenv("OPENAI_MODEL", "gpt-5-mini"), input=prompt)
    return response.output_text.strip()
