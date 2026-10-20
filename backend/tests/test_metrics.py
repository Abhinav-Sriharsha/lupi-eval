"""Unit tests for the grading logic, using tiny hand-made examples
where we can work out the right answer by hand."""
from app.metrics import compute_metrics, macro_f1, normalize, percentile


def test_normalize():
    assert normalize(" Burger ") == "burger"
    assert normalize(427) == "427"
    assert normalize("") is None
    assert normalize(None) is None


def test_macro_f1_perfect():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_macro_f1_counts_rare_class_equally():
    # 9 "common" right, the single "rare" one wrong.
    # Accuracy = 90%, but the rare class F1 = 0, so macro-F1 is much lower.
    expected = ["common"] * 9 + ["rare"]
    predicted = ["common"] * 10
    # common: precision 9/10, recall 1 -> F1 = 0.947..; rare: F1 = 0 -> macro = 0.473..
    assert round(macro_f1(expected, predicted), 3) == 0.474


def test_percentile():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50.5
    assert percentile([], 95) is None


def row(expected, predicted, latency=100.0, error=None):
    return {"expected": expected, "predicted": predicted, "latency_ms": latency,
            "input_tokens": 10, "output_tokens": 5, "error": error}


def test_compute_metrics():
    rows = [
        # fully correct (order id given as a number - still matches "427")
        row({"intent": "wrong_item", "order_id": "427", "item": "burger"},
            {"intent": "wrong_item", "order_id": 427, "item": "Burger"}),
        # right intent, wrong slot
        row({"intent": "late_order", "order_id": "12", "item": None},
            {"intent": "late_order", "order_id": "21", "item": None}),
        # invalid JSON from the model
        row({"intent": "cancel_order", "order_id": None, "item": None}, None),
        # API error
        row({"intent": "refund_request", "order_id": None, "item": None}, None, latency=None, error="429"),
    ]
    m = compute_metrics(rows)
    assert m["n"] == 4
    assert m["valid_json_rate"] == 0.5
    assert m["api_error_rate"] == 0.25
    assert m["intent_accuracy"] == 0.5
    assert m["slot_exact_match"] == 0.25
    assert m["fully_correct"] == 0.25
    assert m["latency_p50_ms"] == 100.0
