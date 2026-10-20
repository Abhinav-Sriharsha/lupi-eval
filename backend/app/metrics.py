"""Grade eval runs: compare each model answer with the correct answer.

Usage (from the evalboard folder):
    docker compose exec backend python -m app.metrics 1          # report for run 1
    docker compose exec backend python -m app.metrics 1 2        # compare runs 1 and 2 side by side
"""
import statistics
import sys

from sqlalchemy import select

from .db import SessionLocal
from .models import Example, Result, Run


# ---------- pure functions (no database) - easy to unit test ----------

def normalize(value):
    """Make answers comparable: '427' == 427, ' Burger ' == 'burger', missing == None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def macro_f1(expected: list[str], predicted: list[str | None]) -> float:
    """F1 computed for each intent separately, then averaged.

    Every intent counts equally, so a model can't hide bad performance on rare
    intents behind good performance on common ones (unlike plain accuracy).
    """
    labels = sorted(set(expected))
    f1_scores = []
    for label in labels:
        tp = sum(1 for e, p in zip(expected, predicted) if e == label and p == label)
        fp = sum(1 for e, p in zip(expected, predicted) if e != label and p == label)
        fn = sum(1 for e, p in zip(expected, predicted) if e == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def percentile(values: list[float], pct: int) -> float | None:
    """p50 = median (typical speed), p95 = slow tail (what unlucky users feel)."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]


def compute_metrics(rows: list[dict]) -> dict:
    """rows: one dict per example with keys
    expected (dict), predicted (dict or None), latency_ms, input_tokens, output_tokens, error.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}

    exp_intents = [normalize(r["expected"]["intent"]) for r in rows]
    pred_intents = [normalize((r["predicted"] or {}).get("intent")) for r in rows]

    intent_correct = [e == p for e, p in zip(exp_intents, pred_intents)]

    def slots_match(r):
        if r["predicted"] is None:  # invalid JSON / API error never counts as correct
            return False
        pred = r["predicted"]
        return (normalize(pred.get("order_id")) == normalize(r["expected"].get("order_id"))
                and normalize(pred.get("item")) == normalize(r["expected"].get("item")))

    slot_correct = [slots_match(r) for r in rows]
    fully_correct = [i and s for i, s in zip(intent_correct, slot_correct)]

    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    out_tokens = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    in_tokens = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]

    return {
        "n": n,
        "valid_json_rate": sum(r["predicted"] is not None for r in rows) / n,
        "api_error_rate": sum(r["error"] is not None for r in rows) / n,
        "intent_accuracy": sum(intent_correct) / n,
        "intent_macro_f1": macro_f1(exp_intents, pred_intents),
        "slot_exact_match": sum(slot_correct) / n,
        "fully_correct": sum(fully_correct) / n,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "avg_input_tokens": statistics.mean(in_tokens) if in_tokens else None,
        "avg_output_tokens": statistics.mean(out_tokens) if out_tokens else None,
    }


# ---------- database loading ----------

def load_rows(db, run_id: int) -> list[dict]:
    query = (
        select(Result, Example)
        .join(Example, Result.example_id == Example.id)
        .where(Result.run_id == run_id)
        .order_by(Example.id)
    )
    return [
        {
            "text": ex.text,
            "expected": ex.expected,
            "predicted": res.predicted,
            "raw_output": res.raw_output,
            "latency_ms": res.latency_ms,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "error": res.error,
        }
        for res, ex in db.execute(query).all()
    ]


def failures(rows: list[dict]) -> list[dict]:
    """Examples where the intent or the slots were wrong - read these to find patterns."""
    out = []
    for r in rows:
        pred = r["predicted"] or {}
        exp = r["expected"]
        wrong = [k for k in ("intent", "order_id", "item") if normalize(pred.get(k)) != normalize(exp.get(k))]
        if wrong:
            out.append({"text": r["text"], "wrong": wrong, "expected": exp, "predicted": r["predicted"],
                        "raw_output": r["raw_output"], "error": r["error"]})
    return out


# ---------- command-line report ----------

ROWS = [
    ("Examples", "n", "{:.0f}"),
    ("Valid JSON", "valid_json_rate", "{:.1%}"),
    ("API errors", "api_error_rate", "{:.1%}"),
    ("Intent accuracy", "intent_accuracy", "{:.1%}"),
    ("Intent macro-F1", "intent_macro_f1", "{:.3f}"),
    ("Slot exact match", "slot_exact_match", "{:.1%}"),
    ("Fully correct", "fully_correct", "{:.1%}"),
    ("Latency p50 (ms)", "latency_p50_ms", "{:.0f}"),
    ("Latency p95 (ms)", "latency_p95_ms", "{:.0f}"),
    ("Avg input tokens", "avg_input_tokens", "{:.0f}"),
    ("Avg output tokens", "avg_output_tokens", "{:.0f}"),
]


def fmt(value, pattern):
    return "-" if value is None else pattern.format(value)


def main(run_ids: list[int]):
    with SessionLocal() as db:
        runs, results = [], []
        for rid in run_ids:
            run = db.get(Run, rid)
            if run is None:
                raise SystemExit(f"Run {rid} not found")
            rows = load_rows(db, rid)
            runs.append(run)
            results.append((compute_metrics(rows), failures(rows)))

    # Side-by-side table
    headers = [f"run {r.id}: {r.model}" for r in runs]
    width = max(22, *(len(h) for h in headers)) + 2
    print("Metric".ljust(20) + "".join(h.rjust(width) for h in headers))
    print("-" * (20 + width * len(headers)))
    for label, key, pattern in ROWS:
        print(label.ljust(20) + "".join(fmt(m.get(key), pattern).rjust(width) for m, _ in results))

    # A few failures per run, to start spotting patterns
    for run, (_, fails) in zip(runs, results):
        print(f"\nRun {run.id} ({run.model}): {len(fails)} wrong answers. First 5:")
        for f in fails[:5]:
            print(f"  - \"{f['text']}\"")
            print(f"      wrong: {', '.join(f['wrong'])} | expected: {f['expected']} | got: {f['predicted'] or f['raw_output'] or f['error']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m app.metrics <run_id> [<run_id> ...]")
    main([int(x) for x in sys.argv[1:]])
