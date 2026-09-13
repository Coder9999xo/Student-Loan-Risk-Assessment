# Code Flow

How the code actually executes: entry points, call order, and which parts were
written or changed by AI. Keep this file in sync whenever a function is added,
removed, or rewired — not just at session boundaries.

## Entry points

| Command | What it does |
|---|---|
| `python app/nemotron.py` | Standalone test: runs one hardcoded student (`TEST_STUDENT`) through Nemotron and prints the raw + parsed result. Useful for testing the prompt/model in isolation, without Streamlit or MongoDB. |
| `python app/mongo.py` | Standalone test: pings MongoDB and prints connection status + stored assessment count. Useful for verifying MongoDB is up before touching the rest of the app. |
| `streamlit run app/streamlit_app.py` | The real app: form → Nemotron → MongoDB → dashboard. |

## Request-level call flow (the path that matters)

This is what happens when a user fills out the form and clicks
**"Assess Credit Risk"**:

```
streamlit_app.new_assessment_tab()
  │  (collects form inputs into a student_data dict)
  ▼
nemotron.assess_student(student_data)
  │
  ├─▶ nemotron.build_messages(student_data)
  │      └─▶ nemotron.format_student_record(student_data)   # builds the "Student record: ..." prompt text
  │
  ├─▶ nemotron._call_nemotron(messages)                     # the actual NVIDIA API call
  │      └─▶ nemotron.get_client()                          # lazy-inits the OpenAI client from NVIDIA_API_KEY
  │
  └─▶ nemotron.validate_assessment(result)                  # required fields/types AND risk_score↔risk_level consistency
        (on JSON error or ValueError: retries the whole call once, then gives up)
  │
  ▼ (on success)
  result["loan_decision"] = nemotron.derive_loan_decision(result["risk_level"])
        # Low→Approve, Moderate→Investigate, High→Disapprove — pure Python, not an LLM output
  ▼
streamlit_app.new_assessment_tab()  (continued)
  │  if outcome["success"] is False → st.error(...) and stop (no misleading result shown)
  │  if True:
  ▼
mongo.test_connection()
  ▼
mongo.save_assessment(student_data, assessment, nemotron.MODEL_NAME)
  ▼
streamlit_app.render_assessment(student_data, assessment)
  ├─▶ render_decision_banner(assessment)
  ├─▶ render_score_gauge(risk_score, risk_level)
  └─▶ (renders risk_factors / protective_factors / explanation directly from Nemotron's JSON)
```

The **History** and **Analytics** tabs follow simpler, read-only paths:

```
streamlit_app.history_tab()
  └─▶ mongo.get_assessments(limit=100)   → table
  └─▶ mongo.get_assessment_by_id(id)     → streamlit_app.render_assessment(...)

streamlit_app.analytics_tab()
  └─▶ mongo.get_analytics()              → st.metric / st.bar_chart
```

## Per-file function map

### `app/nemotron.py` — prediction + explanation engine

| Function | Does | Called by |
|---|---|---|
| `format_student_record(student_data)` | Turns a student dict into the `"Student record: ..."` prompt block; raises `ValueError` if a required field is missing. | `build_messages` |
| `risk_level_for_score(risk_score)` | Pure lookup: the risk band (`Low`/`Moderate`/`High`) that a given `risk_score` implies, per the same 0–30/31–70/71–100 bands stated in the system prompt. Used only to check Nemotron's own `risk_level` against its own `risk_score` — never to override it. | `validate_assessment` |
| `derive_loan_decision(risk_level)` | Pure mapping: `Low→Approve`, `Moderate→Investigate`, `High→Disapprove`. No LLM call, no judgment — the decision policy is fixed once `risk_level` is known. | `assess_student` |
| `validate_assessment(result)` | Checks the parsed JSON has all required fields and valid `risk_score`/`risk_level` values, **and** that `risk_level` agrees with the band `risk_score` implies (via `risk_level_for_score`). Raises `ValueError` on any mismatch — Nemotron's `loan_decision` is no longer part of this schema at all. | `assess_student` |
| `get_client()` | Lazily creates (and caches) the `OpenAI` client pointed at NVIDIA's endpoint, reading `NVIDIA_API_KEY` from `.env`. | `_call_nemotron` |
| `build_messages(student_data)` | Assembles the full message list: system prompt + few-shot examples + the current student's record. | `assess_student` |
| `_call_nemotron(messages)` | Makes the actual `chat.completions.create(...)` call to `nvidia/nemotron-3.5-lightning-30b-a3b`. | `assess_student` |
| `assess_student(student_data, retries=1)` | Orchestrates the above: build messages → call → parse JSON → validate (incl. score↔level consistency); retries once on any failure; on success, sets `result["loan_decision"] = derive_loan_decision(result["risk_level"])` before returning. Never raises — always returns a `{"success": ...}` dict. | `streamlit_app.new_assessment_tab` |

### `app/mongo.py` — persistence layer only (never predicts risk)

| Function | Does | Called by |
|---|---|---|
| `get_client()` | Lazily creates (and caches) the `MongoClient`, reading `MONGODB_URI` from `.env`. | everything else in this file |
| `get_collection()` | Returns the `student_credit_risk.assessments` collection handle. | most functions below |
| `test_connection()` | Pings MongoDB; returns `True`/`False` instead of raising, so the UI can degrade gracefully. | `streamlit_app` (all three tabs) |
| `save_assessment(student_data, assessment, model_name)` | Inserts one document (student input + Nemotron output + model name + UTC timestamp). Returns the inserted id. | `streamlit_app.new_assessment_tab` |
| `get_assessments(limit=50)` | Returns the most recent assessments, newest first. | `streamlit_app.history_tab` |
| `get_assessment_by_id(assessment_id)` | Fetches one document by its Mongo `_id`. | `streamlit_app.history_tab` |
| `get_analytics()` | Aggregates total count, risk-level counts, loan-decision counts, and average risk score. | `streamlit_app.analytics_tab` |

### `app/streamlit_app.py` — UI only (validation + visualization, never prediction)

| Function | Does | Called by |
|---|---|---|
| `render_score_gauge(risk_score, risk_level)` | Draws the 0–100 risk bar (HTML/CSS). | `render_assessment` |
| `render_decision_banner(assessment)` | Draws the CREDIT / DO NOT CREDIT banner. | `render_assessment` |
| `render_assessment(student_data, assessment)` | Composes the full result view: banner, gauge, risk/protective factor lists, explanation, raw input expander. | `new_assessment_tab`, `history_tab` |
| `new_assessment_tab()` | Renders the input form; on submit, calls `nemotron.assess_student`, then `mongo.save_assessment`, then `render_assessment`. | `main` |
| `history_tab()` | Lists past assessments from MongoDB; lets the user pick one to view via `render_assessment`. | `main` |
| `analytics_tab()` | Shows aggregate stats from `mongo.get_analytics()`. | `main` |
| `main()` | Sets up the page title/tabs and dispatches to the three tab functions. | Streamlit's script runner (top-level) |

## Timing instrumentation (diagnostic, print-only)

`nemotron.py` and `streamlit_app.py` print `[NEMOTRON-TIMING]` /
`[STREAMLIT-TIMING]` lines covering: prompt assembly, `get_client()` init,
the NVIDIA API call, JSON parsing, `validate_assessment()`, each retry
attempt, `mongo.save_assessment()`, and end-to-end totals — see
`docs/Decisions.md` for what this found. `nemotron.py` also forces
`sys.stdout.reconfigure(line_buffering=True)` at import time so these
prints are visible immediately under Streamlit's subprocess (which
otherwise block-buffers stdout). None of this changes control flow,
retries, validation, or returned data — it only prints.

## What's AI-written vs. what existed before

- **New files, written by Claude this session:** `app/mongo.py`,
  `app/streamlit_app.py`, `app/.claude/launch.json`, `docs/Decisions.md`,
  `docs/Flow.md` (this file).
- **Refactored, not rewritten:** `app/nemotron.py` — same system prompt,
  few-shot examples, model name, and API parameters as the user's
  already-working version; only wrapped into callable functions
  (`assess_student`, `build_messages`, etc.) instead of running top-to-bottom
  as a script.
- **Small additive edits:** `requirements.txt` (added `pymongo`), `.env`
  (added `MONGODB_URI`), `.gitignore` (expanded to cover Python caches,
  virtual envs, packaging/test/lint caches, Jupyter checkpoints, IDE
  metadata, OS files, and this project's `.tmp_downloads/` — see
  `docs/Decisions.md`).
- **Untouched:** `SRA-Notebooks/`, `data/`, and all prior conventional-ML work
  — intentionally out of scope per the project's current phase.
