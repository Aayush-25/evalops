"""EvalOps Streamlit dashboard — run with: streamlit run dashboard/app.py"""
import os

import pandas as pd
import psycopg2
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide", page_title="EvalOps", page_icon="📊")

DATABASE_URL = os.getenv("DATABASE_URL", "")
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection():
    if not DATABASE_URL:
        st.error("DATABASE_URL is not set — add it to your .env file.")
        st.stop()
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    conn.autocommit = True
    return conn


@st.cache_data(ttl=30)
def fetch_overview_metrics() -> dict:
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                                      AS total,
                    AVG(faithfulness) FILTER (WHERE faithfulness IS NOT NULL)     AS avg_faith,
                    AVG(relevancy)    FILTER (WHERE relevancy    IS NOT NULL)     AS avg_rel,
                    COALESCE(SUM(cost_usd), 0)                                   AS total_cost
                FROM traces
                """
            )
            row = cur.fetchone()
        return {
            "total":      int(row[0]),
            "avg_faith":  float(row[1]) if row[1] is not None else None,
            "avg_rel":    float(row[2]) if row[2] is not None else None,
            "total_cost": float(row[3]),
        }
    except Exception:
        return {"total": 0, "avg_faith": None, "avg_rel": None, "total_cost": 0.0}


@st.cache_data(ttl=30)
def fetch_daily_faithfulness() -> pd.DataFrame:
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DATE(created_at) AS day, AVG(faithfulness) AS avg_faithfulness
                FROM traces
                WHERE faithfulness IS NOT NULL
                  AND created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY day
                """
            )
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["day", "avg_faithfulness"])
        return pd.DataFrame(rows, columns=["day", "avg_faithfulness"]).set_index("day")
    except Exception:
        return pd.DataFrame(columns=["day", "avg_faithfulness"])


@st.cache_data(ttl=30)
def fetch_recent_traces(limit: int = 10) -> pd.DataFrame:
    _COLS = ["id", "project_name", "model", "latency_ms", "cost_usd",
             "faithfulness", "relevancy", "eval_status", "created_at"]
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_name, model, latency_ms, cost_usd,
                       faithfulness, relevancy, eval_status, created_at
                FROM traces
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    except Exception:
        return pd.DataFrame(columns=_COLS)


@st.cache_data(ttl=30)
def fetch_traces(project_name: str, days_back: int, eval_status: str) -> pd.DataFrame:
    _COLS = ["id", "project_name", "model", "latency_ms", "cost_usd",
             "faithfulness", "relevancy", "eval_status", "created_at"]
    try:
        conn = get_connection()
        filters = ["created_at >= NOW() - (INTERVAL '1 day' * %s)"]
        params: list = [days_back]
        if project_name:
            filters.append("project_name = %s")
            params.append(project_name)
        if eval_status != "All":
            filters.append("eval_status = %s")
            params.append(eval_status)
        where = " AND ".join(filters)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, project_name, model, latency_ms, cost_usd,
                       faithfulness, relevancy, eval_status, created_at
                FROM traces
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT 200
                """,
                params,
            )
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    except Exception:
        return pd.DataFrame(columns=_COLS)


@st.cache_data(ttl=30)
def fetch_single_trace(trace_id: str) -> dict | None:
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, project_name, run_name, prompt, response, model,
                       latency_ms, input_tokens, output_tokens, cost_usd,
                       contexts, faithfulness, relevancy, eval_status, created_at
                FROM traces
                WHERE id = %s
                """,
                (trace_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        keys = ["id", "project_name", "run_name", "prompt", "response", "model",
                "latency_ms", "input_tokens", "output_tokens", "cost_usd",
                "contexts", "faithfulness", "relevancy", "eval_status", "created_at"]
        return dict(zip(keys, row))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_score(v) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _trigger_evaluation(project_name: str) -> None:
    try:
        resp = requests.post(
            f"{API_URL}/evaluate",
            json={"project_name": project_name},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        st.success(f"{data['message']} — {data['trace_count']} trace(s)")
        st.cache_data.clear()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("📊 EvalOps")
st.sidebar.caption("LLM Evaluation Platform")
st.sidebar.divider()

st.session_state.setdefault("page", "Overview")
st.session_state.setdefault("selected_trace_id", None)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Trace Explorer", "Single Trace"],
    key="page",
)

st.sidebar.divider()
st.sidebar.caption(f"API: `{API_URL}`")


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("Overview")

    metrics = fetch_overview_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Traces",      metrics["total"])
    c2.metric("Avg Faithfulness",  _fmt_score(metrics["avg_faith"]))
    c3.metric("Avg Relevancy",     _fmt_score(metrics["avg_rel"]))
    c4.metric("Total Cost (USD)",  f"${metrics['total_cost']:.4f}")

    st.subheader("Avg Faithfulness — last 30 days")
    daily = fetch_daily_faithfulness()
    if daily.empty:
        st.info("No scored traces yet — run an evaluation to see this chart.")
    else:
        st.line_chart(daily)

    st.subheader("Last 10 Traces")
    recent = fetch_recent_traces()
    if recent.empty:
        st.info("No traces in the database yet.")
    else:
        st.dataframe(recent, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Trace Explorer
# ---------------------------------------------------------------------------

def page_trace_explorer() -> None:
    st.title("Trace Explorer")

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        project_filter = st.text_input("Project name", value="")
    with col_f2:
        days_back = st.number_input("Days back", min_value=1, max_value=365, value=7)
    with col_f3:
        status_filter = st.selectbox(
            "Eval status", ["All", "pending", "complete", "failed", "running"]
        )

    df = fetch_traces(project_filter.strip(), int(days_back), status_filter)
    st.caption(f"{len(df)} trace(s) found")

    if df.empty:
        st.info("No traces match the current filters.")
    else:
        display = df.copy()
        display["id"] = display["id"].astype(str).str[:8]
        st.dataframe(display, use_container_width=True)

        full_ids = df["id"].astype(str).tolist()
        selected_id = st.selectbox(
            "Select trace to inspect:",
            full_ids,
            format_func=lambda x: x[:8],
        )
        if st.button("Open in Single Trace →"):
            st.session_state["selected_trace_id"] = selected_id
            st.session_state["page"] = "Single Trace"
            st.rerun()

    st.divider()
    eval_project = project_filter.strip() or "default"
    if st.button(f"▶ Trigger Evaluation for '{eval_project}'"):
        _trigger_evaluation(eval_project)


# ---------------------------------------------------------------------------
# Page 3 — Single Trace
# ---------------------------------------------------------------------------

def page_single_trace() -> None:
    st.title("Single Trace")

    trace_id = st.session_state.get("selected_trace_id")
    if not trace_id:
        st.info("No trace selected. Use Trace Explorer to pick one.")
        return

    trace = fetch_single_trace(str(trace_id))
    if trace is None:
        st.warning(f"Trace `{str(trace_id)[:8]}...` not found.")
        return

    st.caption(f"ID: `{trace['id']}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness", _fmt_score(trace["faithfulness"]))
    c2.metric("Relevancy",    _fmt_score(trace["relevancy"]))
    c3.metric("Latency (ms)", trace["latency_ms"])
    c4.metric("Eval Status",  trace["eval_status"])

    st.subheader("Prompt")
    st.text_area(
        "prompt", value=trace["prompt"], height=150,
        disabled=True, label_visibility="collapsed",
    )

    st.subheader("Response")
    st.text_area(
        "response", value=trace["response"], height=150,
        disabled=True, label_visibility="collapsed",
    )

    if trace["contexts"]:
        st.subheader("Contexts")
        for i, ctx in enumerate(trace["contexts"], 1):
            st.markdown(f"**[{i}]** {ctx}")

    st.subheader("Metadata")
    meta_rows = [
        ("project_name",  trace["project_name"]),
        ("run_name",      trace["run_name"] or "—"),
        ("model",         trace["model"]),
        ("input_tokens",  trace["input_tokens"]),
        ("output_tokens", trace["output_tokens"]),
        ("cost_usd",      trace["cost_usd"]),
        ("created_at",    trace["created_at"]),
    ]
    st.table(pd.DataFrame(meta_rows, columns=["field", "value"]))

    st.divider()
    if st.button("↺ Re-evaluate project"):
        _trigger_evaluation(trace["project_name"])


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Overview":
    page_overview()
elif page == "Trace Explorer":
    page_trace_explorer()
elif page == "Single Trace":
    page_single_trace()
