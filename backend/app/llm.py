"""Call an LLM on one caller sentence and measure the response.

Quick test (from the evalboard folder):
    docker compose exec backend python -m app.llm
"""
import os
import time

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Instructions for the model: what to extract and the exact output format.
# Keep this identical for every model you compare, so the test is fair.
SYSTEM_PROMPT = """You extract information from food-delivery support calls.
Return ONLY a JSON object with these keys:
- "intent": one of refund_request, wrong_item, missing_item, late_order,
  cancel_order, order_status, payment_issue, speak_to_agent
- "order_id": the order number as digits (e.g. "427"), or null if not mentioned
- "item": the food item mentioned, or null
Order numbers may be spoken as words, e.g. "four two seven" means "427"."""


def call_model(model: str, text: str):
    """Send one caller sentence to a model.

    Returns (answer, latency_ms, input_tokens, output_tokens):
      answer        - the model's raw text reply (should be JSON)
      latency_ms    - how long the API call took, in milliseconds
      input_tokens  - tokens we sent (prompt + sentence)
      output_tokens - tokens the model generated
    """
    body = {
        "model": model,
        "temperature": 0,  # same input -> same answer, so runs are repeatable
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}

    start = time.perf_counter()  # start the stopwatch
    response = httpx.post(GROQ_URL, json=body, headers=headers, timeout=30)
    latency_ms = (time.perf_counter() - start) * 1000  # stop the stopwatch

    response.raise_for_status()  # raise an error if the API call failed (bad key, wrong model, etc.)
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data["usage"]
    return answer, latency_ms, usage["prompt_tokens"], usage["completion_tokens"]


if __name__ == "__main__":
    sentence = "uh my order four two seven came with the wrong burger"
    answer, latency_ms, in_tok, out_tok = call_model(DEFAULT_MODEL, sentence)
    print("Sentence:     ", sentence)
    print("Model answer: ", answer)
    print(f"Latency:       {latency_ms:.0f} ms")
    print(f"Tokens:        {in_tok} in, {out_tok} out")