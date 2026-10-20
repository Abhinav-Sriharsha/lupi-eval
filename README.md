# EvalBoard

An internal **model-evaluation platform**: register a labeled dataset, run it against any
LLM inference endpoint, and compare models on accuracy, macro-F1, latency, and cost in a
dashboard, with a drill-down into every failed prediction.

First use case: **intent + slot extraction for a real-time voice support agent** (Lupi).
Each utterance must become structured JSON for the agent's state machine:

```
"uh my order four two seven came with the wrong burger"
-> {"intent": "wrong_item", "order_id": "427", "item": "burger"}
```

```
React/TS dashboard  ->  FastAPI  ->  background eval worker  ->  model endpoints (Groq, OpenAI, Anthropic, local)
                           |
                       PostgreSQL (datasets, examples, runs, results)
```

---

# Build guide

Each milestone has a **goal**, **steps**, **done when**, and **interview points**.
Commit and push at the end of every milestone, even if it's rough.

## Milestone 1: Scaffold (provided, ~1 hour to run and understand)

**Goal:** Postgres + FastAPI running in Docker, with 200 labeled examples loaded.

1. Install Docker Desktop. Create a GitHub repo named `evalboard` and push this folder.
2. `cp .env.example .env` and add your Groq API key (free at console.groq.com).
3. Start everything:
   ```bash
   docker compose up --build
   ```
4. In a second terminal, load the data:
   ```bash
   docker compose exec backend python -m scripts.seed
   ```
5. Check it works:
   - http://localhost:8000/health returns `{"status": "ok"}`
   - http://localhost:8000/datasets shows `num_examples: 200`
   - http://localhost:8000/docs is the auto-generated API explorer
6. Run tests: `docker compose exec backend python -m pytest -q`

**Read every file** in `backend/app/` and `backend/scripts/` and be able to explain it.
Key questions to answer for yourself:
- Why does `get_db()` use `yield` and `finally`? (one session per request, always closed)
- Why does re-seeding create a new dataset **version** instead of overwriting? (old runs stay reproducible)
- Why store `raw_output` *and* `predicted`? (so you can debug invalid JSON)

**Done when:** `docker compose up` works and `/datasets` shows 200 examples.

## Milestone 2: Eval runner (you build, ~3-4 hours)

**Goal:** `POST /runs` with `{"dataset_id": 1, "model": "<groq model id>"}` (check console.groq.com for current model names) starts a run
in the background, calls the model for every example, and stores each prediction.

Steps:
1. Create `app/llm.py` with one function:
   `call_model(model, text) -> (raw_output, latency_ms, input_tokens, output_tokens)`.
   - Use `httpx` against Groq's **OpenAI-compatible** endpoint:
     `POST https://api.groq.com/openai/v1/chat/completions`.
     The same code then works for OpenAI and any OpenAI-compatible server (vLLM, Ollama).
   - Measure latency with `time.perf_counter()` around the request.
   - Read token counts from the response's `usage` field.
   - Use `temperature: 0` so runs are repeatable.
2. Write the prompt in `app/prompts.py`: list the allowed intents, the JSON format, and
   2-3 examples. Ask for **JSON only**. Keep it identical across models for a fair comparison.
3. Create `app/runner.py`: `run_eval(run_id)`:
   - set status `running` -> loop over examples -> call model -> try `json.loads`
     (on failure store `predicted=None`) -> save a `Result` -> set status `done`
     (or `failed` with the error).
4. Add `POST /runs` in `main.py` using FastAPI `BackgroundTasks`, returning the run id
   immediately. Add `GET /runs` and `GET /runs/{id}` (status + progress).
5. Handle real-world problems: request timeout (e.g. 30s), retry on HTTP 429 rate limits
   with backoff, and record per-example errors without crashing the whole run.

**Done when:** you run an ~8B and a ~70B model on all 200 examples and see results in the DB.

**Interview points:** why background jobs instead of a blocking request; how you handle
rate limits and partial failures; why temperature 0; why identical prompts across models.

## Milestone 2.5: Make the test set real (~2 hours, don't skip)

The starter data is template-generated and too easy. Before trusting any numbers:
1. Generate ~300 more varied utterances with a strong model (messier speech, two intents
   in one sentence, missing order IDs, misheard digits).
2. **Hand-check 200 of them** and fix any wrong labels. This becomes your test set.
3. Load it as a new dataset version with the seed script.

**Interview point:** "How did you build your eval set?" is the #1 deep-dive question.
Your answer: synthetic generation + human verification + versioning.

## Milestone 3: Metrics (you build, ~2-3 hours)

**Goal:** `GET /runs/{id}/metrics` returns:

| Metric | How |
|---|---|
| Intent accuracy and **macro-F1** | `sklearn.metrics.f1_score(average="macro")` |
| Per-intent precision / recall | `classification_report` |
| Slot exact-match | `order_id` and `item` both match the expected values |
| Valid-JSON rate | % of results with `predicted` not None |
| Latency p50 / p95 | `statistics.quantiles` or numpy percentile |
| Cost per 1K calls | tokens x the provider's price per token |

Also add `GET /runs/{id}/failures` returning wrong predictions (text, expected, predicted).
Write pytest tests for the metric functions with tiny hand-made inputs.

**Done when:** you can compare the 8B vs. the 70B with real numbers.

**Interview points:** why macro-F1 over accuracy (imbalanced intents); p95 vs. average
latency (tail latency is what users feel); how cost was calculated.

## Milestone 4: Dashboard (you build, ~4-5 hours)

`frontend/`: React + TypeScript via Vite. Three pages:
1. **Runs list:** model, dataset version, status, created time; a form to start a new run.
2. **Compare:** pick 2+ runs and see a table of macro-F1, slot match, JSON validity,
   p50/p95, and cost, best value highlighted.
3. **Failures:** a filterable table of wrong predictions (filter by intent). This is
   where you'll find the patterns to talk about in interviews.

Poll `GET /runs/{id}` every 2s while a run is in progress. Enable CORS in FastAPI for
`http://localhost:5173`.

## Milestone 5: Production polish + publish (~3 hours)

1. CI is already in `.github/workflows/ci.yml`; make sure it stays green.
2. Structured logging (run id, model, latency) in the runner.
3. README: screenshots, architecture, a results table, and "what I learned" from failures.
4. A 60-second demo video: start a run, watch it finish, compare models, open failures.
5. Post on LinkedIn with the results table and the repo link.

---

## Connection to the voice fine-tuning project

This harness is the evaluation layer for the next project: fine-tune a ~1B model for
intent + slot extraction and compare it here against the 8B, the 70B, GPT, and Claude.
Add a model by pointing the runner at its OpenAI-compatible endpoint (e.g. vLLM).
