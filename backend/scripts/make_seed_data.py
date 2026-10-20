"""Generate a STARTER labeled dataset for Lupi-style support calls.

This is template-generated so you can start immediately. It is NOT your final
eval set: later you'll add LLM-generated examples and hand-verify a 200-example
test set (see README, Milestone 2.5).

Each line: {"text": "...", "expected": {"intent": ..., "order_id": ..., "item": ...}}
order_id is always normalized to digits, even when spoken as words
("four two seven" -> "427"), which is a realistic speech-to-text challenge.
"""
import json
import random
from pathlib import Path

random.seed(7)

DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
ITEMS = ["burger", "fries", "pizza", "chicken wings", "salad", "milkshake", "burrito",
         "pad thai", "sushi roll", "coke", "garlic bread", "noodles", "sandwich", "tacos"]
FILLERS = ["", "", "", "uh ", "um ", "yeah ", "hi ", "so ", "okay so "]

# (intent, count, templates). {oid} = order id, {item} = item.
SPEC = [
    ("refund_request", 40, [
        "i want a refund for order {oid}", "can i get my money back for order {oid}",
        "please refund my order", "i need a refund the food was cold",
        "refund order {oid} please", "i'd like a refund for the {item}",
        "give me my money back the {item} was terrible", "can you refund order {oid} the {item} was inedible"]),
    ("wrong_item", 30, [
        "i got the wrong item in order {oid}", "my order {oid} came with the wrong {item}",
        "this isn't what i ordered i got {item} instead", "you sent me the wrong food",
        "wrong {item} in my order", "order {oid} has someone else's {item}"]),
    ("missing_item", 25, [
        "my {item} is missing from order {oid}", "the {item} wasn't in the bag",
        "order {oid} is missing the {item}", "i didn't get my {item}", "something is missing from my order"]),
    ("late_order", 30, [
        "where is my order it's been an hour", "order {oid} is really late",
        "my food still hasn't arrived", "it's been ninety minutes and order {oid} isn't here",
        "the driver is taking forever", "why is order {oid} so late"]),
    ("cancel_order", 20, [
        "cancel order {oid}", "i want to cancel my order", "please cancel order {oid} i don't need it anymore",
        "can i cancel the {item} order", "stop my order {oid}"]),
    ("order_status", 25, [
        "what's the status of order {oid}", "has order {oid} been picked up yet",
        "when will my food arrive", "is my order on the way", "track order {oid}"]),
    ("payment_issue", 15, [
        "i was charged twice for order {oid}", "my card got charged but the order failed",
        "the payment for order {oid} is wrong", "why was i charged extra", "double charge on my card"]),
    ("speak_to_agent", 15, [
        "let me talk to a human", "i want to speak to a real person", "transfer me to an agent",
        "can i talk to someone please", "get me a manager"]),
]


def spoken_digits(oid: str) -> str:
    return " ".join(DIGIT_WORDS[int(d)] for d in oid)


def make_example(intent: str, template: str) -> dict:
    oid = str(random.randint(100, 9999)) if "{oid}" in template else None
    item = random.choice(ITEMS) if "{item}" in template else None
    # ~40% of the time, say the order id as words (like speech-to-text output)
    oid_text = spoken_digits(oid) if oid and random.random() < 0.4 else oid
    text = template.format(oid=oid_text, item=item)
    text = random.choice(FILLERS) + text
    return {"text": text, "expected": {"intent": intent, "order_id": oid, "item": item}}


def main():
    rows = []
    for intent, count, templates in SPEC:
        for i in range(count):
            rows.append(make_example(intent, templates[i % len(templates)]))
    random.shuffle(rows)
    out = Path(__file__).resolve().parent.parent / "data" / "intents_seed.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()
