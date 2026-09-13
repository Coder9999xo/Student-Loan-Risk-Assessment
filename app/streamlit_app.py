# streamlit_app.py
#
# Streamlit frontend for the Student Credit Risk Assessment app.
#
# Pipeline (unchanged):
#   Streamlit form -> student_data dict -> nemotron.assess_student()
#       -> structured JSON -> mongo.save_assessment() -> dashboard display
#
# This file only handles UI, layout, and visualization. It does not invent
# any numbers Nemotron didn't produce or that don't come directly from the
# student's own input, and it does not alter the assessment/decision logic.

import html
import time

import altair as alt
import pandas as pd
import streamlit as st

import nemotron
import mongo


st.set_page_config(page_title="Student Risk Assessment", layout="wide")


MAJOR_CATEGORIES = ["STEM", "Business", "Humanities", "Arts"]

RISK_COLORS = {"Low": "#1E7E34", "Moderate": "#B7791F", "High": "#B3261E"}
DECISION_COLORS = {"Approve": "#1E7E34", "Investigate": "#B7791F", "Disapprove": "#B3261E"}
DECISION_ORDER = ["Approve", "Investigate", "Disapprove"]
RISK_LEVEL_ORDER = ["Low", "Moderate", "High"]

# Static reference copy only — mirrors the fixed policy already enforced by
# nemotron.derive_loan_decision(); not used to compute anything.
DECISION_POLICY = [
    ("Low", "0 – 30", "Approve"),
    ("Moderate", "31 – 70", "Investigate"),
    ("High", "71 – 100", "Disapprove"),
]


def esc(value) -> str:
    """Escape Nemotron-generated text before it is interpolated into raw HTML."""
    return html.escape(str(value), quote=True)


FIELD_HELP = {
    "acr": "Aid Coverage Ratio — (aid + estimated income) divided by estimated costs. Higher means stronger financial coverage.",
    "cash_flow_cv": "Coefficient of variation of monthly cash flow. Higher means more volatile, less predictable income.",
    "bill_payment_rate": "Share of bills paid on time. Higher means stronger payment discipline.",
    "ecs": "Enrollment Continuity Score — semesters enrolled relative to expected total. Higher means stronger academic continuity.",
    "gdi": "GPA Direction Index — whether recent academic performance is improving, stable, or declining.",
    "tgb": "Time-to-Graduation Buffer — resilience relative to remaining academic time and financial burden. Higher is more resilient.",
    "sei": "Student Economic Index — an overall economic-health indicator (0-100).",
    "synthetic_debt_load": "Total existing synthetic debt load, in dollars.",
    "semesters_enrolled": "Number of semesters the student has been enrolled so far.",
    "major_category": "Broad field of study.",
}


# ============================================================
# STYLING (native Streamlit components first; CSS only where it
# materially improves layout/hierarchy that Streamlit can't do alone)
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1200px; }

        .status-line { font-size: 0.85rem; color: #6B7280; margin-top: -0.6rem; margin-bottom: 1rem; }
        .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
        .status-dot.online { background:#1E7E34; }
        .status-dot.offline { background:#B3261E; }

        .group-header {
            font-size: 0.78rem; letter-spacing: 1.2px; text-transform: uppercase;
            color: #4B5563; font-weight: 600; border-bottom: 1px solid #E2E5EA;
            padding-bottom: 0.4rem; margin-bottom: 0.8rem;
        }

        .section-label {
            font-size: 0.78rem; letter-spacing: 1.2px; text-transform: uppercase;
            color: #6B7280; font-weight: 600; margin-top: 1.6rem; margin-bottom: 0.6rem;
        }

        .result-block {
            text-align: center; border: 1px solid #E2E5EA; border-radius: 8px;
            padding: 1rem 0.5rem; background: #FAFBFC; height: 100%;
        }
        .result-label {
            font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase;
            color: #6B7280; margin-bottom: 0.4rem;
        }
        .result-value { font-size: 1.7rem; font-weight: 700; color: #1A1A1A; line-height: 1.2; }
        .result-sub { font-size: 0.82rem; color: #6B7280; margin-top: 0.25rem; }

        .decision-block.approve { border-color:#1E7E34; background:#F1F8F3; }
        .decision-block.investigate { border-color:#B7791F; background:#FDF7EC; }
        .decision-block.disapprove { border-color:#B3261E; background:#FBEFEE; }
        .decision-value.approve { color:#1E7E34; }
        .decision-value.investigate { color:#B7791F; }
        .decision-value.disapprove { color:#B3261E; }

        .risk-scale-wrap { margin-top: 0.4rem; margin-bottom: 0.6rem; }
        .risk-scale-track { position: relative; height: 14px; border-radius: 4px; display: flex; overflow: visible; }
        .risk-scale-track .zone { height: 14px; }
        .risk-scale-track .zone.low { background:#CDEEDD; border-radius: 4px 0 0 4px; }
        .risk-scale-track .zone.moderate { background:#FAECC8; }
        .risk-scale-track .zone.high { background:#F6D3CF; border-radius: 0 4px 4px 0; }
        .risk-scale-marker {
            position: absolute; top: -7px; width: 3px; height: 28px;
            background: #1A1A1A; transform: translateX(-50%);
        }
        .risk-scale-marker-label {
            position: absolute; top: -30px; transform: translateX(-50%);
            font-size: 0.72rem; font-weight: 700; background: #1A1A1A; color: #fff;
            padding: 1px 6px; border-radius: 3px; white-space: nowrap;
        }
        .risk-scale-ticks { position: relative; height: 18px; font-size: 0.72rem; color: #6B7280; margin-top: 4px; }
        .risk-scale-ticks span { position: absolute; transform: translateX(-50%); }
        .risk-scale-zone-labels { display:flex; justify-content: space-between; font-size: 0.72rem; color:#8A93A3; margin-top: 2px; }
        .risk-scale-caveat { font-size: 0.72rem; color: #9AA2AF; margin-top: 0.6rem; font-style: italic; }

        .factor-item {
            border-left: 3px solid; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem;
            background: #FAFBFC; border-radius: 0 4px 4px 0; font-size: 0.92rem; line-height: 1.45;
        }
        .factor-item.risk { border-color: #B3261E; }
        .factor-item.protective { border-color: #1E7E34; }

        .explanation-box {
            border: 1px solid #E2E5EA; border-radius: 8px; padding: 1.25rem 1.5rem;
            background: #FAFBFC; line-height: 1.65; font-size: 0.98rem;
        }

        .key-reasons-list { margin: 0; padding-left: 1.3rem; }
        .key-reasons-list li { margin-bottom: 0.4rem; line-height: 1.45; }

        .empty-state {
            border: 1px dashed #D6DAE1; border-radius: 8px; padding: 1.5rem;
            color: #6B7280; text-align: center; font-size: 0.92rem;
        }
        .step-row { display:flex; justify-content:center; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap; }
        .step-chip {
            border: 1px solid #E2E5EA; border-radius: 4px; padding: 0.3rem 0.7rem;
            font-size: 0.78rem; color: #4B5563; background: #FFFFFF;
        }
        .step-arrow { color: #C2C7D0; align-self: center; }

        .hero-eyebrow {
            font-size: 0.72rem; letter-spacing: 1.8px; text-transform: uppercase;
            color: #8A93A3; font-weight: 700; margin-bottom: 0.6rem;
        }
        .hero-accent { height: 4px; border-radius: 2px; margin-bottom: 1.1rem; }

        .result-block.hero-decision { padding: 1.3rem 0.5rem; }
        .result-value.hero { font-size: clamp(1.5rem, 4vw, 2.1rem); white-space: nowrap; }
        .result-value.hero-support { font-size: 1.9rem; }

        .policy-legend {
            display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 0.8rem 0 0.2rem 0;
            padding: 0.7rem 1rem; border: 1px solid #E2E5EA; border-radius: 6px; background: #FFFFFF;
        }
        .policy-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #374151; }
        .policy-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
        .policy-arrow { color: #ADB3BE; }

        .why-decision-lead {
            font-size: 0.95rem; color: #374151; margin-bottom: 0.7rem; line-height: 1.5;
        }

        .stat-chip {
            border: 1px solid #E2E5EA; border-left: 4px solid; border-radius: 6px;
            padding: 0.7rem 0.9rem; background: #FAFBFC;
        }
        .stat-chip .stat-label {
            font-size: 0.68rem; letter-spacing: 0.3px; text-transform: uppercase;
            color: #6B7280; white-space: nowrap;
        }
        .stat-chip .stat-value { font-size: 1.5rem; font-weight: 700; color: #1A1A1A; margin-top: 0.15rem; }
        .stat-chip .stat-sub { font-size: 0.76rem; color: #8A93A3; margin-top: 0.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_decision_policy():
    rows = "".join(
        f"""
        <div class="policy-row">
            <span>{level} risk</span>
            <span style="color:#9AA2AF;">({score_range})</span>
            <span class="policy-arrow">&rarr;</span>
            <span class="policy-swatch" style="background:{DECISION_COLORS[decision]}"></span>
            <span style="font-weight:600; color:{DECISION_COLORS[decision]};">{decision}</span>
        </div>
        """
        for level, score_range, decision in DECISION_POLICY
    )
    st.markdown(
        f"""
        <div class="hero-eyebrow" style="margin-top:1rem;">Decision Policy</div>
        <div class="policy-legend">{rows}</div>
        """,
        unsafe_allow_html=True,
    )


def nemotron_status() -> bool:
    try:
        nemotron.get_client()
        return True
    except ValueError:
        return False


def render_header():
    st.title("Student Risk Assessment")
    st.caption("AI-assisted credit risk assessment powered by NVIDIA Nemotron")

    nem_online = nemotron_status()
    mongo_online = mongo.test_connection()

    nem_dot = "online" if nem_online else "offline"
    mongo_dot = "online" if mongo_online else "offline"
    nem_text = "Nemotron Online" if nem_online else "Nemotron Not Configured"
    mongo_text = "MongoDB Connected" if mongo_online else "MongoDB Offline"

    st.markdown(
        f"""
        <div class="status-line">
            <span class="status-dot {nem_dot}"></span>{nem_text}
            &nbsp;&nbsp;&nbsp;
            <span class="status-dot {mongo_dot}"></span>{mongo_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


# ============================================================
# RESULT RENDERING (shared by New Assessment + History detail view)
# ============================================================

def render_risk_scale(risk_score: int, risk_level: str):
    marker_pos = max(0, min(100, risk_score))
    st.markdown(
        f"""
        <div class="risk-scale-wrap">
            <div class="risk-scale-track">
                <div class="zone low" style="width:30%"></div>
                <div class="zone moderate" style="width:40%"></div>
                <div class="zone high" style="width:30%"></div>
                <div class="risk-scale-marker" style="left:{marker_pos}%"></div>
                <div class="risk-scale-marker-label" style="left:{marker_pos}%">{risk_score}</div>
            </div>
            <div class="risk-scale-ticks">
                <span style="left:0%">0</span>
                <span style="left:30%">30</span>
                <span style="left:70%">70</span>
                <span style="left:100%">100</span>
            </div>
            <div class="risk-scale-zone-labels">
                <span>LOW (0–30)</span><span style="margin-left:-1.5rem;">MODERATE (31–70)</span><span>HIGH (71–100)</span>
            </div>
            <div class="risk-scale-caveat">
                Risk score reflects this project's defined framework, not a statistically calibrated probability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(student_data: dict, assessment: dict):
    decision = assessment["loan_decision"]
    risk_level = assessment["risk_level"]
    decision_class = decision.lower()
    decision_color = DECISION_COLORS.get(decision, "#1A1A1A")

    with st.container(border=True):
        st.markdown('<div class="hero-eyebrow">Assessment Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-accent" style="background:{decision_color};"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="result-block">
                    <div class="result-label">Risk Level</div>
                    <div class="result-value hero-support" style="color:{RISK_COLORS.get(risk_level, '#1A1A1A')}">{esc(risk_level.upper())} RISK</div>
                    <div class="result-sub">{esc(assessment['default_prediction'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="result-block">
                    <div class="result-label">Risk Score</div>
                    <div class="result-value hero-support">{assessment['risk_score']} / 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="result-block hero-decision decision-block {decision_class}">
                    <div class="result-label">Loan Decision</div>
                    <div class="result-value hero decision-value {decision_class}">{esc(decision.upper())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label">Risk Score Scale</div>', unsafe_allow_html=True)
        render_risk_scale(assessment["risk_score"], risk_level)
        render_decision_policy()

    st.markdown('<div class="section-label">Why This Decision?</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="why-decision-lead">This student was assessed as '
        f'<b style="color:{RISK_COLORS.get(risk_level, "#1A1A1A")}">{esc(risk_level)} risk</b>, '
        f'resulting in a recommended action of '
        f'<b style="color:{decision_color}">{esc(decision)}</b>, based on the following:</div>',
        unsafe_allow_html=True,
    )
    if assessment["key_reasons"]:
        items = "".join(f"<li>{esc(reason)}</li>" for reason in assessment["key_reasons"])
        st.markdown(f'<ul class="key-reasons-list">{items}</ul>', unsafe_allow_html=True)
    else:
        st.markdown("_None identified._")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Risk Factors**")
        if assessment["risk_factors"]:
            for factor in assessment["risk_factors"]:
                st.markdown(f'<div class="factor-item risk">{esc(factor)}</div>', unsafe_allow_html=True)
        else:
            st.markdown("_None identified._")

    with col2:
        st.markdown("**Protective Factors**")
        if assessment["protective_factors"]:
            for factor in assessment["protective_factors"]:
                st.markdown(f'<div class="factor-item protective">{esc(factor)}</div>', unsafe_allow_html=True)
        else:
            st.markdown("_None identified._")

    st.markdown('<div class="section-label">AI Risk Assessment</div>', unsafe_allow_html=True)
    explanation_html = esc(assessment["explanation"]).replace("\n", "<br>")
    st.markdown(f'<div class="explanation-box">{explanation_html}</div>', unsafe_allow_html=True)

    with st.expander("Full assessment data (raw)"):
        st.markdown("**Student input**")
        st.json(student_data)
        st.markdown("**Nemotron assessment**")
        st.json(assessment)


# ============================================================
# NEW ASSESSMENT TAB
# ============================================================

def render_input_form():
    st.markdown('<div class="group-header">Academic Profile</div>', unsafe_allow_html=True)
    semesters_enrolled = st.number_input(
        "Semesters Enrolled", min_value=1, max_value=16, value=5,
        help=FIELD_HELP["semesters_enrolled"],
    )
    major_category = st.selectbox(
        "Major Category", options=MAJOR_CATEGORIES, help=FIELD_HELP["major_category"]
    )
    ecs = st.number_input(
        "Enrollment Continuity (ECS)", min_value=0.0, max_value=1.5, value=0.75,
        step=0.01, format="%.2f", help=FIELD_HELP["ecs"],
    )
    gdi = st.selectbox(
        "GPA Direction (GDI)", options=[1, 0, -1],
        format_func=lambda v: {1: "Improving (+1)", 0: "Stable (0)", -1: "Declining (-1)"}[v],
        help=FIELD_HELP["gdi"],
    )
    tgb = st.number_input(
        "Time-to-Graduation Buffer (TGB)", min_value=0.0, max_value=2.0, value=0.50,
        step=0.01, format="%.2f", help=FIELD_HELP["tgb"],
    )

    st.markdown('<div class="group-header" style="margin-top:1.5rem;">Financial Profile</div>', unsafe_allow_html=True)
    acr = st.number_input(
        "Aid Coverage Ratio (ACR)", min_value=0.0, max_value=5.0, value=0.80,
        step=0.01, format="%.2f", help=FIELD_HELP["acr"],
    )
    sei = st.number_input(
        "Student Economic Index (SEI)", min_value=0, max_value=100, value=60,
        help=FIELD_HELP["sei"],
    )
    synthetic_debt_load = st.number_input(
        "Synthetic Debt Load ($)", min_value=0, max_value=300000, value=50000,
        step=1000, help=FIELD_HELP["synthetic_debt_load"],
    )

    st.markdown('<div class="group-header" style="margin-top:1.5rem;">Financial Behaviour</div>', unsafe_allow_html=True)
    cash_flow_cv = st.number_input(
        "Cash-Flow Volatility (cash_flow_cv)", min_value=0.0, max_value=5.0, value=0.50,
        step=0.01, format="%.2f", help=FIELD_HELP["cash_flow_cv"],
    )
    bill_payment_rate = st.number_input(
        "Bill Payment Rate", min_value=0.0, max_value=1.0, value=0.70,
        step=0.01, format="%.2f", help=FIELD_HELP["bill_payment_rate"],
    )

    return {
        "acr": acr,
        "cash_flow_cv": cash_flow_cv,
        "bill_payment_rate": bill_payment_rate,
        "ecs": ecs,
        "gdi": gdi,
        "tgb": tgb,
        "sei": sei,
        "synthetic_debt_load": synthetic_debt_load,
        "semesters_enrolled": semesters_enrolled,
        "major_category": major_category,
    }


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            Fill in the student profile above and click <b>Generate Assessment</b>
            to run the AI risk assessment.
            <div class="step-row">
                <span class="step-chip">Student Inputs</span>
                <span class="step-arrow">&rarr;</span>
                <span class="step-chip">Nemotron</span>
                <span class="step-arrow">&rarr;</span>
                <span class="step-chip">Validation</span>
                <span class="step-arrow">&rarr;</span>
                <span class="step-chip">Decision</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_decision_policy()


def new_assessment_tab():
    st.subheader("Student Profile")

    with st.form("student_form"):
        student_data = render_input_form()
        st.write("")
        submitted = st.form_submit_button(
            "Generate Assessment", width="stretch", type="primary"
        )

    if submitted:
        # --- Diagnostic timing instrumentation (print-only, no behavior change) ---
        e2e_start = time.perf_counter()
        print(f"[STREAMLIT-TIMING] === form submitted at {time.strftime('%H:%M:%S')} ===")

        with st.spinner("Running Nemotron risk assessment..."):
            nemotron_call_start = time.perf_counter()
            outcome = nemotron.assess_student(student_data)
            print(
                f"[STREAMLIT-TIMING] nemotron.assess_student() as observed from Streamlit: "
                f"{time.perf_counter() - nemotron_call_start:.3f}s"
            )

        if not outcome["success"]:
            st.error(
                "Nemotron did not return a valid assessment, so no result is being "
                "shown. This can happen occasionally with generative models — try again."
            )
            with st.expander("Error details"):
                st.write(outcome.get("error"))
                st.code(outcome.get("raw") or "(no response)")
            st.session_state["last_assessment"] = None
            st.session_state["last_student_data"] = None
            print(f"[STREAMLIT-TIMING] END-TO-END (submit -> failure shown): {time.perf_counter() - e2e_start:.3f}s")
        else:
            assessment = outcome["assessment"]

            if mongo.test_connection():
                mongo_start = time.perf_counter()
                mongo.save_assessment(student_data, assessment, nemotron.MODEL_NAME)
                print(f"[STREAMLIT-TIMING] mongo.save_assessment(): {time.perf_counter() - mongo_start:.3f}s")
                st.toast("Assessment saved to MongoDB.")
            else:
                st.warning("MongoDB is not reachable — this assessment was NOT saved to history.")

            st.session_state["last_assessment"] = assessment
            st.session_state["last_student_data"] = student_data
            print(f"[STREAMLIT-TIMING] END-TO-END (submit -> result ready to render): {time.perf_counter() - e2e_start:.3f}s")

    st.markdown('<div class="section-label">Assessment Result</div>', unsafe_allow_html=True)
    if st.session_state.get("last_assessment"):
        render_result(st.session_state["last_student_data"], st.session_state["last_assessment"])
    else:
        render_empty_state()


# ============================================================
# HISTORY TAB
# ============================================================

def history_tab():
    st.subheader("Assessment History")

    if not mongo.test_connection():
        st.warning("MongoDB is not reachable. Start MongoDB locally to view history.")
        return

    records = mongo.get_assessments(limit=100)
    if not records:
        st.info("No assessments stored yet. Run one from the 'New Assessment' tab.")
        return

    table_rows = []
    for rec in records:
        table_rows.append({
            "id": str(rec["_id"]),
            "Timestamp": rec["created_at"].strftime("%Y-%m-%d %H:%M UTC"),
            "Decision": rec["assessment"]["loan_decision"],
            "Risk Level": rec["assessment"]["risk_level"],
            "Risk Score": rec["assessment"]["risk_score"],
            "Major": rec["student_data"].get("major_category"),
        })

    df = pd.DataFrame(table_rows)
    display_df = df.drop(columns=["id"])

    def style_decision(val):
        return f"color:{DECISION_COLORS.get(val, '#1A1A1A')}; font-weight:600;"

    def style_risk_level(val):
        return f"color:{RISK_COLORS.get(val, '#1A1A1A')}; font-weight:600;"

    styled = display_df.style.map(style_decision, subset=["Decision"]).map(
        style_risk_level, subset=["Risk Level"]
    )
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%d"
            ),
        },
    )

    st.markdown('<div class="section-label">Assessment Detail</div>', unsafe_allow_html=True)
    options = df["id"].tolist()
    labels = {
        row["id"]: f"{row['Timestamp']} — {row['Risk Level']} risk, {row['Decision']}"
        for _, row in df.iterrows()
    }
    selected_id = st.selectbox(
        "Select an assessment to view in detail",
        options=options, format_func=lambda i: labels.get(i, i),
    )
    if selected_id:
        with st.expander("View full assessment detail", expanded=False):
            record = mongo.get_assessment_by_id(selected_id)
            if record:
                render_result(record["student_data"], record["assessment"])


# ============================================================
# ANALYTICS TAB
# ============================================================

def analytics_tab():
    st.subheader("Analytics")

    if not mongo.test_connection():
        st.warning("MongoDB is not reachable. Start MongoDB locally to view analytics.")
        return

    stats = mongo.get_analytics()
    if stats["total"] == 0:
        st.info("No assessments stored yet.")
        return

    total = stats["total"]
    approve = stats["loan_decision_counts"].get("Approve", 0)
    investigate = stats["loan_decision_counts"].get("Investigate", 0)
    disapprove = stats["loan_decision_counts"].get("Disapprove", 0)
    avg_score = stats["average_risk_score"]

    with st.container(border=True):
        st.markdown('<div class="hero-eyebrow">Portfolio Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Assessments", total)
        c2.metric("Avg Risk Score", f"{avg_score:.1f}" if avg_score is not None else "N/A")

        for col, label, count, color in (
            (c3, "Approve", approve, DECISION_COLORS["Approve"]),
            (c4, "Investigate", investigate, DECISION_COLORS["Investigate"]),
            (c5, "Disapprove", disapprove, DECISION_COLORS["Disapprove"]),
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="stat-chip" style="border-left-color:{color};">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value" style="color:{color};">{count}</div>
                        <div class="stat-sub">{(count / total * 100):.0f}% of total</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-label">Decision &amp; Risk Overview</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Loan Decision Distribution**")
            decision_df = pd.DataFrame(
                [{"Decision": k, "Count": stats["loan_decision_counts"].get(k, 0)} for k in DECISION_ORDER]
            )
            chart = (
                alt.Chart(decision_df)
                .mark_bar()
                .encode(
                    x=alt.X("Decision:N", sort=DECISION_ORDER, title=None),
                    y=alt.Y("Count:Q", title="Count"),
                    color=alt.Color(
                        "Decision:N",
                        scale=alt.Scale(domain=DECISION_ORDER, range=[DECISION_COLORS[k] for k in DECISION_ORDER]),
                        legend=None,
                    ),
                    tooltip=["Decision", "Count"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, width="stretch")

        with col2:
            st.markdown("**Risk Level Distribution**")
            level_df = pd.DataFrame(
                [{"Risk Level": k, "Count": stats["risk_level_counts"].get(k, 0)} for k in RISK_LEVEL_ORDER]
            )
            chart = (
                alt.Chart(level_df)
                .mark_bar()
                .encode(
                    x=alt.X("Risk Level:N", sort=RISK_LEVEL_ORDER, title=None),
                    y=alt.Y("Count:Q", title="Count"),
                    color=alt.Color(
                        "Risk Level:N",
                        scale=alt.Scale(domain=RISK_LEVEL_ORDER, range=[RISK_COLORS[k] for k in RISK_LEVEL_ORDER]),
                        legend=None,
                    ),
                    tooltip=["Risk Level", "Count"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, width="stretch")


# ============================================================
# MAIN
# ============================================================

def main():
    inject_css()
    render_header()

    st.session_state.setdefault("last_assessment", None)
    st.session_state.setdefault("last_student_data", None)

    tab1, tab2, tab3 = st.tabs(["New Assessment", "History", "Analytics"])
    with tab1:
        new_assessment_tab()
    with tab2:
        history_tab()
    with tab3:
        analytics_tab()


if __name__ == "__main__":
    main()
