"""Run one model on one dataset and save every prediction.

Usage (from the evalboard folder):
    docker compose exec backend python -m app.runner 1 openai/gpt-oss-20b
                                                      ^ dataset id  ^ model id
"""
import json
import sys
import time
from datetime import datetime, timezone

from sqlalchemy import select

from .db import Base, SessionLocal, engine
from .llm import call_model
from .models import Example, Result, Run

SECONDS_BETWEEN_CALLS = 1  # stay under Groq's free-tier rate limit


def run_eval(dataset_id: int, model: str) -> int:
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        # 1. Create the run record first: every saved answer points to it.
        run = Run(dataset_id=dataset_id, model=model, status="running")
        db.add(run)
        db.commit()  # saves it; the database assigns run.id
        print(f"Started run {run.id}: model={model}, dataset={dataset_id}")

        examples = db.scalars(
            select(Example).where(Example.dataset_id == dataset_id).order_by(Example.id)
        ).all()
        if not examples:
            run.status = "failed"
            db.commit()
            raise SystemExit(f"No examples found for dataset {dataset_id}")

        failures = 0
        for i, ex in enumerate(examples, start=1):
            answer = latency_ms = in_tok = out_tok = error = None

            # 2. Ask the model. If one call fails, record why and keep going,
            #    instead of losing the whole run.
            try:
                answer, latency_ms, in_tok, out_tok = call_model(model, ex.text)
            except Exception as e:
                error = str(e)
                failures += 1

            # 3. The model returns text; we need a dict to compare with the
            #    correct answer. Invalid JSON is stored as None (counted later
            #    in the "valid JSON rate" metric).
            predicted = None
            if answer is not None:
                try:
                    predicted = json.loads(answer)
                except json.JSONDecodeError:
                    predicted = None

            # 4. Save everything: raw text (for debugging), parsed answer,
            #    speed, and token usage.
            db.add(Result(
                run_id=run.id,
                example_id=ex.id,
                raw_output=answer,
                predicted=predicted,
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                error=error,
            ))
            db.commit()  # commit each one so progress survives a crash

            status = "ERROR" if error else ("ok" if predicted is not None else "bad JSON")
            print(f"[{i}/{len(examples)}] {status:8} {ex.text[:50]}")
            time.sleep(SECONDS_BETWEEN_CALLS)

        # 5. Mark the run finished.
        run.status = "done"
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
        print(f"Finished run {run.id}: {len(examples)} examples, {failures} API errors")
        return run.id


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m app.runner <dataset_id> <model_id>")
    run_eval(int(sys.argv[1]), sys.argv[2])