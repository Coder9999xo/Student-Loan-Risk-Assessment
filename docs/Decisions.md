# Decisions Log

Running log of meaningful project decisions, newest entry first. Each entry
records what was decided and *why*, so anyone (human or AI) picking up the
repo later doesn't have to reverse-engineer intent from the code or re-open
settled questions.

---

## 2026-09-13 — Expanded `.gitignore` to cover generated/cache/tool artifacts
**Decision:** Rewrote `.gitignore` to explicitly cover, beyond what was
already there (`.env`, `__pycache__/`, `*.pyc`, `app/.tmp_downloads/`):
virtual environments (`.venv/`, `venv/`, `env/`, `ENV/`), Python
packaging/test/lint caches (`*.egg-info/`, `build/`, `dist/`,
`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`), Jupyter checkpoints
(`.ipynb_checkpoints/`), IDE/tool metadata (`.vscode/`, `.idea/`,
`app/.claude/`), and OS-generated files (`.DS_Store`, `Thumbs.db`,
`desktop.ini`).
**Why:** None of these existed yet in the repo except `__pycache__` and the
old MongoDB installer, but they're exactly the categories of file that get
accidentally committed the first time someone opens this project in a
different editor or OS. Added proactively per an explicit request to audit
for generated/cache/environment-specific files.
**`app/.claude/` specifically:** ignored as tool metadata for the Claude
Code browser-preview tooling (`launch.json`, `settings.local.json`) — not
required for a human to run the app (`streamlit run app/streamlit_app.py`
works without it).
**Explicitly kept tracked (not ignored):** `app/.streamlit/config.toml` —
this is real, required app configuration (the Streamlit color theme), not
personal or machine-specific, so it stays in the repo.
**Verified** with `git check-ignore` on both sides (generated files ignored,
source/docs/config files not) and a `git add --dry-run -A` showing exactly
8 legitimate files would be staged: `.gitignore`, `app/nemotron.py`,
`requirements.txt`, `app/.streamlit/config.toml`, `app/mongo.py`,
`app/streamlit_app.py`, `docs/Decisions.md`, `docs/Flow.md`.

## 2026-09-13 — Latency investigation: added print-based timing instrumentation; root cause is NVIDIA API response-time variance, not Streamlit
**Decision:** Instrumented `nemotron.py` (`assess_student`, `_call_nemotron`,
`get_client`) and `streamlit_app.py` (`new_assessment_tab`) with `[NEMOTRON-
TIMING]` / `[STREAMLIT-TIMING]` print statements covering: prompt assembly,
client init, the API call itself, JSON parsing, validation, retries, Mongo
save, and end-to-end totals. Also forced `sys.stdout.reconfigure(line_buffering=True)`
in `nemotron.py`, because the Streamlit dev-server subprocess runs with
stdout not attached to a terminal, which defaults Python to full block
buffering — without this, timing prints sat in a buffer and never appeared
in the logs the tooling reads, making live diagnosis impossible.
**Why:** The user reported `python app/nemotron.py` taking ~5–10s normally,
vs. Streamlit sometimes taking ~60–70s on the first assessment.
**Findings:**
- The code does **not** contain `reasoning_budget` or `max_tokens=16384` —
  the actual call uses `max_tokens=2000`, `temperature=0.0`, `stream=False`
  (confirmed non-streaming), and no `reasoning_budget` parameter exists
  anywhere in this file. Those parameters were not contributing to anything,
  because they aren't present.
- No proxy environment variables are set in either the direct-script or the
  Streamlit-subprocess context (checked via a one-time diagnostic print in
  `get_client()`) — ruling out a proxy-related explanation.
- `get_client()`'s one-time `OpenAI()` construction cost varied 0.38s–1.8s
  across runs, in both contexts — a small, one-time-per-process cost, not
  the main story.
- The real finding: running `python app/nemotron.py` **four times back-to-
  back with no Streamlit involved at all** produced API-call times of
  26.4s, 12.1s, 18.3s, 15.8s — i.e., the same high variance historically
  blamed on Streamlit reproduces in the plain reference script when called
  repeatedly without delay. Streaming is off, so "time to first token" and
  "time to full response" are not separately observable — both collapse
  into one blocking measurement, which is itself the dominant cost every
  time (JSON parsing and validation are consistently ~0.000s).
- Conclusion: the latency lives in NVIDIA's hosted endpoint response time
  (or the network path to it from this environment), not in Streamlit's
  execution model, not in `max_tokens`/`reasoning_budget` (which don't
  exist as described), and not in stdout buffering (that only blocked our
  ability to *observe* timing live — it added no real latency). The
  existing retry-once mechanism is a real amplifier though: if one slow
  attempt (e.g. 26s) also fails JSON validation, the retry roughly doubles
  the visible wait — this is almost certainly how the historically-observed
  60–130s totals happened.
**What was proposed, not yet applied:** add a bounded request `timeout`
(e.g. 30–45s) to the `chat.completions.create()` call so a genuinely stalled
request fails fast into the existing retry/graceful-error path instead of
hanging indefinitely, without changing the risk framework, decision mapping,
or JSON validation. Not implemented yet — pending the user's go-ahead, per
their explicit instruction not to change these parameters during this
investigation.
**Timing instrumentation left in place** (print-only, no behavior change) —
safe to keep for ongoing diagnosis; can be stripped out on request.

---

## 2026-09-13 — 3-way decision (Approve/Investigate/Disapprove); Nemotron stays the classifier; app enforces score↔level consistency
**Decision:** Replaced the binary `loan_decision` (`Credit`/`Do Not Credit`,
generated by Nemotron) with a 3-way outcome — `Approve` / `Investigate` /
`Disapprove` — computed by the **application**, not the model:
`derive_loan_decision()` in `nemotron.py` maps Nemotron's own `risk_level`
(Low→Approve, Moderate→Investigate, High→Disapprove) after that `risk_level`
has passed a new consistency check.
**Why:** The user asked for a "fit non-deterministic answers within a range"
approach without taking classification away from the model. Confirmed
explicitly with the user: **Nemotron remains the classifier** — it still
outputs `risk_level` from its own reasoning over the rules; the app does not
override it. Instead, `validate_assessment()` now recomputes the risk band
implied by Nemotron's own `risk_score` (0–30/31–70/71–100) and raises
`ValueError` if it disagrees with the `risk_level` Nemotron stated — which
feeds the *existing* retry-once loop in `assess_student()`, so an
inconsistent pairing gets silently retried rather than shown to the user.
Because `loan_decision` is now a pure, no-judgment mapping from an
already-validated `risk_level`, it can never disagree with the risk level
shown, and it no longer depends on the separate, previously-inconsistent
`default_prediction` signal at all.
**Also changed:** The system prompt gained a "REASONING REQUIREMENTS" section
requiring every `key_reasons`/`risk_factors`/`protective_factors` entry to (a)
reference only the 10 defined data points — nothing outside this framework
(no collateral, credit history, cosigners, etc.) — and (b) state the actual
supplied value and interpret it per the framework's definitions (e.g. "ECS =
0.85, indicating strong enrollment continuity, which is protective") instead
of a bare label like "Strong academic continuity". The few-shot examples were
rewritten to demonstrate this format. The 10 student input fields themselves
were left completely unchanged, per the user's explicit instruction.

---

## 2026-09-13 — `.claude/launch.json` lives under `app/`, not the repo root
**Decision:** Placed the browser-preview launch config at `app/.claude/launch.json`
instead of `SRA/.claude/launch.json`.
**Why:** The active session's working directory for this project is `SRA/app`,
and the preview tool resolves `.claude/launch.json` relative to the working
directory. A copy at the repo root would never be found; a duplicate would risk
drifting out of sync.

## 2026-09-13 — Risk-score bar and decision banner use plain HTML/CSS, not a charting library
**Decision:** `streamlit_app.py` renders the risk score bar and the decision
banner with inline HTML/CSS inside `st.markdown(..., unsafe_allow_html=True)`
rather than adding Plotly or another charting dependency.
**Why:** It's a single deterministic bar driven by one number — not worth a new
dependency for a prototype. Consistent with the project rule that the UI must
never invent numbers: every value shown comes directly from Nemotron's own
structured JSON or from the student's raw input.

## 2026-09-13 — `streamlit_app.py` is one file with three tabs
**Decision:** Built the frontend as a single `streamlit_app.py` with three
tabs — New Assessment, History, Analytics — instead of splitting into multiple
pages/files.
**Why:** Matches the UI flow already planned in the project handoff doc
(form → Nemotron → MongoDB → dashboard, plus history/analytics as "future
dashboard features"). Splitting further now would be premature structure for a
prototype with three screens.

## 2026-09-13 — Added `MONGODB_URI` to `.env`
**Decision:** Added `MONGODB_URI=mongodb://localhost:27017` to `.env`,
alongside the existing `NVIDIA_API_KEY`.
**Why:** Matches the env-var plan already laid out for this project. Keeps all
secrets/config in one gitignored place; `mongo.py` reads it via
`os.getenv("MONGODB_URI", "mongodb://localhost:27017")` so the app still works
with a sane default even if the variable is ever missing.

## 2026-09-13 — Added `pymongo` to `requirements.txt`
**Decision:** Added `pymongo` (the official MongoDB Python driver).
**Why:** It's the standard, first-party driver — no alternatives were
considered for a local prototype.

## 2026-09-13 — MongoDB installed locally without administrator rights
**Decision:** Installed MongoDB Community binaries into
`%LOCALAPPDATA%\MongoDB` using `msiexec /a <installer.msi> TARGETDIR=...`
(an "administrative install" — it just unpacks the MSI's file tree; it does
not register a Windows service and does not require elevation), then started
`mongod.exe` manually, bound to `127.0.0.1:27017`, with `--dbpath` and
`--logpath` pointing at folders under that same directory.
**Why:** This sandbox session has no administrator privileges, and a normal
`msiexec /i` (full service install) failed with error 1603 because it insists
on writing to `C:\Program Files\...` and `C:\ProgramData\...` and registering a
Windows service — both of which require admin rights. `winget`'s package
source was also failing to sync in this environment, so a MongoDB-through-winget
install wasn't an option either. The administrative-install extraction was the
only path that produced a working `mongod.exe` without elevation.
**Tradeoff:** MongoDB does **not** auto-start on reboot and is not registered
as a Windows service this way — it must be started manually (see `Flow.md` /
`README` for the exact command), or the user can later run the official MSI
installer themselves with administrator rights to install it as a proper
always-on service.

## 2026-09-13 — MongoDB runs locally, not on Atlas
**Decision:** Use a local MongoDB instance (`mongodb://localhost:27017`)
instead of MongoDB Atlas.
**Why:** Decided by the user before this session (carried over from the
project handoff doc): no cloud credentials to manage, no IP allowlisting, and
the app needs to run/demo entirely offline during development.
*(Logged here for the record — the decision itself predates this session.)*

## 2026-09-13 — `assess_student()` retries once, then fails gracefully
**Decision:** `nemotron.assess_student()` retries the Nemotron call at most
once on invalid/incomplete JSON, then returns `{"success": False, "error":
"...", "raw": "..."}` instead of raising or returning a partial result.
**Why:** Nemotron is generative and occasionally returns malformed or
incomplete JSON. The project's explicit principle (from the handoff doc) is
that the app must never display a misleading assessment — so the Streamlit
layer checks `outcome["success"]` and shows an explicit error state instead of
guessing at a result.

## 2026-09-13 — Refactored `nemotron.py` into importable functions
**Decision:** Restructured `nemotron.py` from a top-to-bottom script (that ran
once against a hardcoded `TEST_EXAMPLE` on `python nemotron.py`) into a module
exposing `assess_student()`, `format_student_record()`,
`validate_assessment()`, `build_messages()`, and `get_client()`, while leaving
the original `python nemotron.py` standalone-test behavior intact under
`if __name__ == "__main__":`.
**Why:** Streamlit needs to call the assessment logic per form submission with
dynamic student data, not just run once against a fixed test record.
**Constraint honored:** The `SYSTEM_PROMPT`, `FEW_SHOT_EXAMPLES`, model name
(`nvidia/nemotron-3.5-lightning-30b-a3b`), and all API call parameters
(temperature, max_tokens, `response_format`, `extra_body`) were kept exactly as
they were in the already-working script — only the surrounding code
structure changed, per the standing rule not to rewrite a working
implementation unnecessarily.
