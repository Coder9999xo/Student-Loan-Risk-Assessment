# nemotron.py

import os
import sys
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, APIStatusError

# ============================================================
# TIMING INSTRUMENTATION (diagnostic only — no behavior change)
#
# Prints [NEMOTRON-TIMING] lines to stdout so latency can be measured
# identically whether this module is driven by `python app/nemotron.py`
# or imported by Streamlit. Safe to leave in: it only prints, it does not
# alter control flow, retries, validation, or the returned data.
#
# Streamlit's subprocess (as launched by this project's dev-server tooling)
# runs with stdout NOT attached to a terminal, which defaults Python to
# fully block-buffered output — timing prints would sit in a buffer and
# only appear after several KB accumulate or the process exits, making
# live diagnosis impossible. Force line-buffering so every print() (each
# ending in \n) flushes immediately, in both entry points.
# ============================================================

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _log_timing(label: str, elapsed: float) -> None:
    print(f"[NEMOTRON-TIMING] {label}: {elapsed:.3f}s")


# ============================================================
# 1. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a Student Credit Risk Assessment assistant for a fully
synthetic student financial-risk dataset.

Your task is to assess whether the supplied student is likely
to default and whether the loan should be credited.

Use ONLY the student information supplied in the current request
and the risk framework below.

==================================================
TARGET
==================================================

default_flag:
0 = Non-default / Paid
1 = Default

Do NOT use default_probability as an input predictor.

student_id is an identifier and must not be used as evidence.

==================================================
ENGINEERED RISK FEATURES
==================================================

ECS — Enrollment Continuity Score

ECS = semesters_enrolled / expected_total_semesters

Higher ECS indicates stronger enrollment continuity and academic
progression. Lower ECS may indicate disruption or delayed
progression.


ACR — Aid Coverage Ratio

ACR =
[aid amount + (monthly inflow mean × 4)]
/
[estimated tuition + estimated living cost]

Higher ACR indicates stronger financial coverage.
Lower ACR indicates greater financial burden.


CFRI — Cash Flow Resilience Index

CFRI = 1 - cash_flow_cv

Higher CFRI indicates more stable income.
Lower CFRI indicates more volatile income.


OPR — Obligation Punctuality Rate

OPR = bill_payment_rate

Higher OPR indicates stronger payment discipline.
Lower OPR indicates weaker payment behavior.


TGB — Time-to-Graduation Buffer

TGB represents flexibility/resilience relative to remaining
academic time and financial burden.

Higher TGB indicates greater resilience.
Lower TGB indicates greater vulnerability.


GDI — GPA Direction Index

+1 = improving
0 = stable
-1 = declining

==================================================
EDA RISK RELATIONSHIPS
==================================================

1. CASH-FLOW VOLATILITY

Higher cash_flow_cv is associated with greater default risk.

The EDA identified a volatility-discipline interaction:

- At cash_flow_cv <= 0.4, defaults are uncommon unless
  bill_payment_rate falls below approximately 20%.
- As cash_flow_cv approaches >= 1.0, defaults can appear at
  bill_payment_rate levels as high as approximately 60%.

Therefore, high cash-flow volatility combined with weak or
moderate payment compliance is particularly concerning.

Do not treat volatility alone as proof of default.


2. AID COVERAGE

Lower ACR is associated with higher default risk.
Higher ACR provides a stronger financial buffer.


3. STUDENT ECONOMIC INDEX

SEI has a strong positive relationship with ACR and a negative
relationship with synthetic_debt_load.

Use SEI as an overall economic-health indicator.


4. ACADEMIC CONTINUITY

Lower ECS is associated with greater default risk.
Higher ECS indicates stronger academic progression.


5. PAYMENT DISCIPLINE

Low bill_payment_rate / OPR is a major warning signal.

Payment behavior must be interpreted together with
cash-flow volatility.


6. ACADEMIC DIRECTION

Declining GPA (GDI = -1) is a negative signal.
Improving GPA (GDI = +1) is a protective signal.


7. TEMPORAL CONTEXT

semesters_enrolled is the primary temporal anchor.

Do not treat synthetic_age, remaining_semesters, and
expected_total_semesters as independent evidence when they
represent redundant temporal information.


8. MAJOR

Do not automatically consider STEM, Business, Humanities,
or Arts safer or riskier.

The EDA found broadly similar structural behavior across majors.

==================================================
RISK SCORE CONSISTENCY
==================================================

risk_score is an integer from 0 to 100.

Use the following score ranges:

0–30   = Low risk
31–70  = Moderate risk
71–100 = High risk

The risk score must be consistent with risk_level.

Do not change the risk_level merely because the exact score
changes.

For similar student profiles, keep the risk score close to
the same value.

Do not use arbitrary score changes between otherwise identical
assessments.
==================================================
RISK SCORE GUIDANCE
==================================================

Use the following general anchors:

LOW RISK:
Typically 0–30.

MODERATE RISK:
Typically 31–75.

HIGH RISK:
Typically 76–100.

When assigning a score, consider the combined strength of:

- ACR / financial coverage
- cash-flow volatility
- bill payment rate
- ECS / academic continuity
- GDI / academic direction
- TGB
- SEI
- synthetic debt load

High cash-flow volatility combined with weak/moderate
payment compliance should substantially increase risk.

Strong academic continuity and improving GPA can reduce,
but should not completely cancel, substantial financial
and behavioral risk.

The score should represent the overall severity of the
profile, not the number of risk factors listed.

For the same input values, produce approximately the same
risk score.
==================================================
DECISION RULE
==================================================

Assess the overall combination of financial, behavioral,
academic, and temporal characteristics.

Do not determine risk from a single feature unless it is
an exceptionally strong signal.

Classify the student as:

Low
Moderate
High

Then provide:

Likely Non-default
OR
Likely Default

DO NOT output a loan_decision field.

The application determines the final decision (Approve /
Investigate / Disapprove) automatically from your risk_level.
Your only responsibility is to produce an accurate,
well-justified risk_score and risk_level that agree with each
other under the RISK SCORE CONSISTENCY bands above.

==================================================
REASONING REQUIREMENTS
==================================================

Every entry in key_reasons, risk_factors, and
protective_factors MUST:

1. Reference ONLY the following data points, and nothing else:
   - Aid Coverage Ratio (ACR)
   - Cash-flow volatility (cash_flow_cv)
   - Bill payment rate (OPR)
   - Enrollment Continuity Score (ECS)
   - GPA Direction Index (GDI)
   - Time-to-Graduation Buffer (TGB)
   - Student Economic Index (SEI)
   - Synthetic debt load
   - Semesters enrolled
   - Major category

   Do NOT reference collateral, credit history, cosigners,
   interest rates, employment status, or any other concept not
   supplied in the student record or defined in this framework.

2. State the actual supplied value and interpret it according
   to this framework's definitions, not just name the feature.

   Example (correct): "ECS = 0.85, indicating strong enrollment
   continuity, which is protective."

   Example (too vague, do NOT do this): "Strong academic
   continuity."

==================================================
IMPORTANT SYNTHETIC DATA CAVEAT
==================================================

The dataset is synthetic.

Do not claim that these relationships are universal financial
laws or guaranteed real-world outcomes.

Predictions are assessments within this project's synthetic
framework.

Do not fabricate missing values.

==================================================
OUTPUT
==================================================

Return ONLY one complete valid JSON object.

Never output thinking, reasoning, chain-of-thought,
step-by-step analysis, or text outside the JSON.

Use EXACTLY these fields:

{
  "risk_level": "Low | Moderate | High",
  "default_prediction": "Likely Non-default | Likely Default",
  "risk_score": 0,
  "key_reasons": [],
  "protective_factors": [],
  "risk_factors": [],
  "explanation": ""
}

Rules:

- risk_score must be an integer from 0 to 100.
- risk_score must match risk_level.
- Do NOT include a loan_decision field.
- key_reasons, protective_factors, and risk_factors must
  follow the REASONING REQUIREMENTS above.
- Do not invent factors unsupported by the supplied values.
- Do not repeat the same factor in both protective_factors
  and risk_factors.
- explanation must be concise.
- Complete the entire JSON object before stopping.
"""


# ============================================================
# 2. FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = [

    {
        "user": """
Student record:

acr: 0.55
cash_flow_cv: 1.10
bill_payment_rate: 0.35
ecs: 0.50
gdi: -1
tgb: 0.25
sei: 35
synthetic_debt_load: 85000
semesters_enrolled: 4
major_category: Humanities

Assess the student's default risk and loan decision.
""",

        "assistant": """
{
  "risk_level": "High",
  "default_prediction": "Likely Default",
  "risk_score": 86,
  "key_reasons": [
    "ACR = 0.55, indicating weak financial coverage relative to costs",
    "cash_flow_cv = 1.10, indicating very high cash-flow volatility",
    "bill_payment_rate = 0.35, weak payment compliance combined with high volatility is especially concerning",
    "ECS = 0.50, indicating low enrollment continuity",
    "GDI = -1, indicating declining academic direction"
  ],
  "protective_factors": [],
  "risk_factors": [
    "ACR = 0.55 (low), weak financial coverage",
    "cash_flow_cv = 1.10 (very high), highly unstable cash flow",
    "bill_payment_rate = 0.35 (low), weak payment compliance under volatility",
    "ECS = 0.50 (low), weak academic continuity",
    "GDI = -1, declining academic direction"
  ],
  "explanation": "The combination of weak financial coverage (ACR 0.55), highly unstable cash flow (cash_flow_cv 1.10), poor payment compliance (bill_payment_rate 0.35) under that volatility, and weaker academic progression (ECS 0.50, GDI -1) indicates high default risk within the synthetic dataset framework."
}
"""
    },


    {
        "user": """
Student record:

acr: 1.35
cash_flow_cv: 0.25
bill_payment_rate: 0.95
ecs: 0.90
gdi: 1
tgb: 0.80
sei: 88
synthetic_debt_load: 45000
semesters_enrolled: 7
major_category: Business

Assess the student's default risk and loan decision.
""",

        "assistant": """
{
  "risk_level": "Low",
  "default_prediction": "Likely Non-default",
  "risk_score": 8,
  "key_reasons": [
    "ACR = 1.35, indicating strong financial coverage",
    "cash_flow_cv = 0.25, indicating low cash-flow volatility",
    "bill_payment_rate = 0.95, indicating very strong payment compliance",
    "ECS = 0.90, indicating strong enrollment continuity",
    "GDI = 1, indicating improving academic direction"
  ],
  "protective_factors": [
    "ACR = 1.35 (high), strong financial coverage",
    "cash_flow_cv = 0.25 (low), stable cash flow",
    "bill_payment_rate = 0.95 (high), strong payment discipline",
    "ECS = 0.90 (high), strong academic progression",
    "GDI = 1, improving academic direction"
  ],
  "risk_factors": [
    "synthetic_debt_load = 45000, existing debt remains a consideration despite strong coverage"
  ],
  "explanation": "The student demonstrates strong financial stability (ACR 1.35, cash_flow_cv 0.25), excellent payment behavior (bill_payment_rate 0.95), and strong academic progression (ECS 0.90, GDI 1). The combined indicators support a low-risk assessment."
}
"""
    }
]


# ============================================================
# TEST EXAMPLE (used only when this file is run directly)
# ============================================================

TEST_STUDENT = {
    "acr": 0.68,
    "cash_flow_cv": 0.85,
    "bill_payment_rate": 0.42,
    "ecs": 0.65,
    "gdi": -1,
    "tgb": 0.35,
    "sei": 48,
    "synthetic_debt_load": 78000,
    "semesters_enrolled": 5,
    "major_category": "Arts",
}


# ============================================================
# STUDENT RECORD FIELDS
# ============================================================

REQUIRED_FIELDS = [
    "acr",
    "cash_flow_cv",
    "bill_payment_rate",
    "ecs",
    "gdi",
    "tgb",
    "sei",
    "synthetic_debt_load",
    "semesters_enrolled",
    "major_category",
]

REQUIRED_ASSESSMENT_FIELDS = [
    "risk_level",
    "default_prediction",
    "risk_score",
    "key_reasons",
    "protective_factors",
    "risk_factors",
    "explanation",
]

# Must match the RISK SCORE CONSISTENCY bands in SYSTEM_PROMPT.
RISK_LEVEL_BY_SCORE_BAND = [
    (30, "Low"),
    (70, "Moderate"),
    (100, "High"),
]

LOAN_DECISION_BY_RISK_LEVEL = {
    "Low": "Approve",
    "Moderate": "Investigate",
    "High": "Disapprove",
}


def risk_level_for_score(risk_score: int) -> str:
    """The risk_level Nemotron's own score bands imply, for consistency checking."""
    for max_score, level in RISK_LEVEL_BY_SCORE_BAND:
        if risk_score <= max_score:
            return level
    raise ValueError(f"risk_score out of range: {risk_score!r}")


def derive_loan_decision(risk_level: str) -> str:
    """Pure business-policy mapping. No judgment call: Nemotron's own
    (validated-consistent) risk_level fully determines this."""
    return LOAN_DECISION_BY_RISK_LEVEL[risk_level]


def format_student_record(student_data: dict) -> str:
    """Build the 'Student record: ...' prompt block from a student dict."""
    missing = [f for f in REQUIRED_FIELDS if f not in student_data]
    if missing:
        raise ValueError(f"Missing required student fields: {missing}")

    lines = "\n".join(f"{field}: {student_data[field]}" for field in REQUIRED_FIELDS)

    return f"""
Student record:

{lines}

Assess the student's default risk and loan decision.
"""


def validate_assessment(result: dict) -> None:
    """Raise ValueError if the parsed Nemotron JSON is missing required fields."""
    missing = [f for f in REQUIRED_ASSESSMENT_FIELDS if f not in result]
    if missing:
        raise ValueError(f"Assessment missing required fields: {missing}")

    if result["risk_level"] not in ("Low", "Moderate", "High"):
        raise ValueError(f"Invalid risk_level: {result['risk_level']!r}")

    if not isinstance(result["risk_score"], int) or not (0 <= result["risk_score"] <= 100):
        raise ValueError(f"Invalid risk_score: {result['risk_score']!r}")

    expected_level = risk_level_for_score(result["risk_score"])
    if result["risk_level"] != expected_level:
        raise ValueError(
            f"risk_level {result['risk_level']!r} is inconsistent with "
            f"risk_score {result['risk_score']!r} (expected {expected_level!r})"
        )


# ============================================================
# API CONNECTION
# ============================================================

load_dotenv()

_client = None

# Per-attempt request timeout, in seconds. Applies to each individual call
# to the NVIDIA API; assess_student()'s own retry-once loop is unaffected
# and independent of this value.
REQUEST_TIMEOUT_SECONDS = 45.0


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    init_start = time.perf_counter()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found. Add it to your .env file.")

    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        # Bounded per-attempt timeout — the SDK default is a 600s read
        # timeout, which is far too long to sit behind a Streamlit spinner.
        # This does NOT change risk logic; it only bounds how long a single
        # network attempt may take before failing into the existing
        # retry-once path in assess_student().
        timeout=REQUEST_TIMEOUT_SECONDS,
        # The SDK's own default (max_retries=2) would silently retry
        # timeouts/5xx internally, on top of our own retry-once loop in
        # assess_student() — multiplying attempts in a way that isn't
        # visible in application logs. Disabling it here keeps the total
        # number of HTTP attempts equal to exactly what assess_student()'s
        # retries=1 loop makes (at most 2), not up to 2x that.
        max_retries=0,
    )
    _log_timing(
        "get_client() FIRST-TIME init (OpenAI() constructor — no network call, "
        "but on some hosts this is where proxy/env resolution happens)",
        time.perf_counter() - init_start,
    )
    return _client


MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"


def build_messages(student_data: dict) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    messages.append({"role": "user", "content": format_student_record(student_data)})
    return messages


def _call_nemotron(messages: list) -> str:
    # NOTE (diagnostic): stream=False below, so this is one blocking call —
    # "time to first token" and "time to full response" are NOT separately
    # observable at this layer; both are covered by the single
    # "NVIDIA API request" measurement. If streaming were enabled, those
    # two would become distinguishable — but streaming is NOT being changed
    # here, only reported on.
    request_start = time.perf_counter()

    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,

        # Force JSON output
        response_format={
            "type": "json_object"
        },

        # Disable visible reasoning for this structured-output stage
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },

        temperature=0.0,
        max_tokens=2000,
        stream=False,
    )

    _log_timing(
        "NVIDIA API request (request-sent -> full response received; "
        "stream=False, so this includes connect + TLS + full generation time)",
        time.perf_counter() - request_start,
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        print(
            f"[NEMOTRON-TIMING] token usage: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )

    return response.choices[0].message.content.strip()


def assess_student(student_data: dict, retries: int = 1) -> dict:
    """
    Run a Nemotron credit-risk assessment for a single student.

    Returns a dict with structured JSON on success:
        {"success": True, "assessment": {...}, "raw": "..."}

    Returns a dict describing the failure otherwise, rather than raising,
    so callers (e.g. Streamlit) can fail gracefully instead of showing a
    misleading assessment:
        {"success": False, "error": "...", "raw": "..."}
    """
    overall_start = time.perf_counter()
    print(f"[NEMOTRON-TIMING] === assess_student() entered at {time.strftime('%H:%M:%S')} ===")

    build_start = time.perf_counter()
    messages = build_messages(student_data)
    _log_timing("build_messages() (prompt assembly, no network)", time.perf_counter() - build_start)

    last_error = None
    last_raw = None

    # At most `retries + 1` attempts total (2 with the default retries=1).
    # get_client() sets max_retries=0 on the SDK client, so each iteration
    # below makes exactly one HTTP attempt — the SDK cannot silently add
    # more attempts of its own underneath this loop.
    for attempt in range(retries + 1):
        attempt_label = f"attempt {attempt + 1}/{retries + 1}"
        attempt_start = time.perf_counter()
        will_retry = attempt < retries

        try:
            raw_output = _call_nemotron(messages)

        except APITimeoutError as exc:
            _log_timing(f"{attempt_label}: API TIMEOUT (>{REQUEST_TIMEOUT_SECONDS:.0f}s) after", time.perf_counter() - attempt_start)
            last_error = f"API timeout after {REQUEST_TIMEOUT_SECONDS:.0f}s: {exc}"
            last_raw = None
            if will_retry:
                print(f"[NEMOTRON-TIMING] retrying -> attempt {attempt + 2}/{retries + 1}")
            continue

        except APIConnectionError as exc:
            _log_timing(f"{attempt_label}: API CONNECTION ERROR after", time.perf_counter() - attempt_start)
            last_error = f"API connection error: {exc}"
            last_raw = None
            if will_retry:
                print(f"[NEMOTRON-TIMING] retrying -> attempt {attempt + 2}/{retries + 1}")
            continue

        except APIStatusError as exc:
            _log_timing(
                f"{attempt_label}: API STATUS ERROR (HTTP {exc.status_code}) after",
                time.perf_counter() - attempt_start,
            )
            last_error = f"API status error (HTTP {exc.status_code}): {exc}"
            last_raw = None
            if will_retry:
                print(f"[NEMOTRON-TIMING] retrying -> attempt {attempt + 2}/{retries + 1}")
            continue

        last_raw = raw_output

        try:
            parse_start = time.perf_counter()
            result = json.loads(raw_output)
            _log_timing(f"{attempt_label}: JSON parsing (json.loads)", time.perf_counter() - parse_start)

        except json.JSONDecodeError as exc:
            _log_timing(f"{attempt_label}: INVALID/TRUNCATED JSON after", time.perf_counter() - attempt_start)
            print(f"[NEMOTRON-TIMING] {attempt_label} JSON error: {exc}")
            last_error = f"Invalid JSON: {exc}"
            if will_retry:
                print(f"[NEMOTRON-TIMING] retrying -> attempt {attempt + 2}/{retries + 1}")
            continue

        try:
            validate_start = time.perf_counter()
            validate_assessment(result)
            _log_timing(f"{attempt_label}: validate_assessment()", time.perf_counter() - validate_start)

        except ValueError as exc:
            _log_timing(f"{attempt_label}: RISK VALIDATION FAILURE after", time.perf_counter() - attempt_start)
            print(f"[NEMOTRON-TIMING] {attempt_label} validation error: {exc}")
            last_error = f"Validation failed: {exc}"
            if will_retry:
                print(f"[NEMOTRON-TIMING] retrying -> attempt {attempt + 2}/{retries + 1}")
            continue

        result["loan_decision"] = derive_loan_decision(result["risk_level"])

        _log_timing(f"{attempt_label}: SUCCESS - TOTAL (API call + parse + validate)", time.perf_counter() - attempt_start)
        _log_timing("assess_student() TOTAL end-to-end", time.perf_counter() - overall_start)
        return {"success": True, "assessment": result, "raw": raw_output}

    _log_timing("assess_student() TOTAL end-to-end (all attempts failed)", time.perf_counter() - overall_start)
    return {"success": False, "error": last_error, "raw": last_raw}


# ============================================================
# STANDALONE TEST (only runs when executing this file directly)
# ============================================================

if __name__ == "__main__":
    outcome = assess_student(TEST_STUDENT)

    print("\nRaw Nemotron Response:\n")
    print(outcome["raw"])

    if outcome["success"]:
        print("\nStructured Assessment:\n")
        print(json.dumps(outcome["assessment"], indent=2))
    else:
        print("\nERROR: Nemotron did not return a valid assessment.")
        print(f"Reason: {outcome['error']}")
