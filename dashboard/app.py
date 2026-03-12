"""
dashboard/app.py
Streamlit + Plotly interactive analytics dashboard for the Pricing
Intelligence Agent.

Run standalone:
    streamlit run dashboard/app.py

Or launch automatically via main.py which starts this as a subprocess.

Tabs:
  0 — 💬 Chat with Agent   (browser-based chat UI via HTTP bridge)
  1 — 📊 Rule Utilisation
  2 — 🌍 Country & Region
  3 — 💵 Revenue Opportunity
  4 — ⚠️  Pricing Leakage
  5 — 🤖 ML Model Results   (Vertex AI AutoML)
  6 — 📋 Rule Recommendations

Sidebar:
  • Dev Mode toggle  — shows raw SQL, row counts, log tail, BQ diagnostics
  • Refresh button   — reloads shared agent state from disk
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Ensure project root is on PYTHONPATH when running as subprocess
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.tools import load_state  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pricing Intelligence Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# HTTP bridge — calls the agent server started by main.py
# ---------------------------------------------------------------------------

AGENT_SERVER_URL = "http://localhost:8502"


def call_agent(message: str) -> tuple[str, dict]:
    """POST a user message to the agent HTTP server and return (response, metadata)."""
    try:
        resp = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={"message": message},
            timeout=180,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "No response from agent."), data.get("metadata", {})
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Agent server is not reachable. Make sure `main.py` is running.",
            {"error": "Connection refused to http://localhost:8502"},
        )
    except requests.exceptions.Timeout:
        return (
            "⏱️ Agent timed out (>3 min). The query may be too large — try a narrower date range.",
            {"error": "Request timeout"},
        )
    except requests.exceptions.HTTPError as exc:
        return (
            f"⚠️ Agent returned error {exc.response.status_code}: {exc.response.text}",
            {"error": str(exc)},
        )
    except Exception as exc:
        return f"Unexpected error: {exc}", {"error": str(exc)}


def check_agent_health() -> bool:
    """Returns True if the agent server is reachable and healthy."""
    try:
        resp = requests.get(f"{AGENT_SERVER_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Training status helpers  (file-based so threads can share state with UI)
# ---------------------------------------------------------------------------

TRAINING_STATUS_FILE = _ROOT / "training_status.json"


def read_training_status() -> dict:
    """Read training status from disk; returns {} if the file is absent."""
    try:
        if TRAINING_STATUS_FILE.exists():
            return json.loads(TRAINING_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_training_status(status: str, **kwargs) -> None:
    """Write training status dict to disk so the UI thread can poll it."""
    payload = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
        **kwargs,
    }
    try:
        TRAINING_STATUS_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:
        logging.warning("[DASHBOARD] Could not write training status: %s", exc)


def _run_training(date_from_str: str, date_to_str: str) -> None:
    """
    Background-thread wrapper: runs train_models() and writes status file at
    each stage so the Streamlit UI can poll it without session_state.
    """
    from ml.trainer import _release_training_lock  # noqa: PLC0415
    write_training_status("running", date_from=date_from_str, date_to=date_to_str)
    try:
        from ml.trainer import train_models  # noqa: PLC0415

        results = train_models(date_from_str, date_to_str)
        write_training_status(
            "complete",
            date_from=date_from_str,
            date_to=date_to_str,
            results=results,
        )
        logging.info("[DASHBOARD] Training complete: %s", results)
    except Exception as exc:
        write_training_status(
            "failed",
            date_from=date_from_str,
            date_to=date_to_str,
            error=str(exc),
        )
        logging.error("[DASHBOARD] Training failed: %s", exc, exc_info=True)
    finally:
        _release_training_lock()  # always release lock so next run is not blocked


# ---------------------------------------------------------------------------
# Sidebar — dev mode + refresh + examples
# ---------------------------------------------------------------------------

st.sidebar.title("💰 Pricing Intelligence")
st.sidebar.markdown("---")

# Dev mode toggle — must be evaluated before any tab content
dev_mode: bool = st.sidebar.toggle(
    "🛠 Dev Mode",
    value=False,
    help="Show raw SQL, row counts, logs, and BQ diagnostics.",
)

if dev_mode:
    st.sidebar.info("Dev mode ON — queries and logs visible in each panel.")

    # Live log tail
    st.sidebar.subheader("📋 Live Logs")
    log_path = _ROOT / "agent_session.log"
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()[-30:]
        st.sidebar.text_area(
            "agent_session.log (last 30 lines)",
            "\n".join(lines),
            height=300,
        )
    except FileNotFoundError:
        st.sidebar.caption("No log file yet — start a chat to create it.")

    # BQ diagnostics
    st.sidebar.subheader("🔍 BQ Diagnostics")
    if st.sidebar.button("Run BQ Diagnostics", use_container_width=True):
        try:
            from bq.queries import run_diagnostic_query  # noqa: PLC0415
            diag_result = run_diagnostic_query()
            st.sidebar.json(diag_result)
            st.sidebar.code("1=1  (no SCD filter — all rows active)", language="sql")
        except Exception as diag_exc:
            st.sidebar.error(f"Could not run diagnostics: {diag_exc}")

st.sidebar.markdown("---")

if st.sidebar.button("🔄  Refresh from Agent State", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Charts update automatically when the agent runs a tool.  "
    "Use the Chat tab or ask via the CLI, then refresh here."
)
st.sidebar.markdown("**Example prompts:**")
st.sidebar.markdown(
    "- *What % of quotes used the default rule?*\n"
    "- *Show me the top revenue uplift opportunities*\n"
    "- *Flag all leakage records with margin below 5 %*\n"
    "- *Run model training*\n"
    "- *What columns are available?*"
)

# ---------------------------------------------------------------------------
# Load shared state
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_state():
    return load_state()


state = get_state()

# ---------------------------------------------------------------------------
# Helper: empty-state placeholder
# ---------------------------------------------------------------------------

def _empty_state(tool_hint: str) -> None:
    st.info(
        f"No data loaded yet.  Ask the agent in the **Chat** tab: "
        f"**\"{tool_hint}\"** and then click **Refresh from Agent State** "
        "in the sidebar.",
        icon="🤖",
    )


# ---------------------------------------------------------------------------
# Dev helper: raw data expander
# ---------------------------------------------------------------------------

def _dev_expander(df: pd.DataFrame | None, label: str = "Raw Data") -> None:
    """Render a collapsible raw-data panel — only visible in dev mode."""
    if dev_mode and df is not None and not df.empty:
        with st.expander(f"🔧 {label}", expanded=False):
            st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "💬 Chat",
    "📊 Rule Utilisation",
    "🌍 Country & Region",
    "💵 Revenue Opportunity",
    "⚠️ Pricing Leakage",
    "🤖 ML Model Results",
    "📋 Rule Recommendations",
    "🧠 Train Model",
])

# ===========================================================================
# TAB 0 — Chat with Agent
# ===========================================================================

with tabs[0]:
    st.header("💬 Chat with Pricing Intelligence Agent")
    st.caption(
        "Type a question below — the agent will query BigQuery, run analysis, "
        "and update the other dashboard tabs automatically."
    )

    # Agent health banner
    _agent_ok = check_agent_health()
    if _agent_ok:
        st.success("🟢 Agent server is online (localhost:8502)", icon="✅")
    else:
        st.error(
            "🔴 Agent server is **offline** — start `main.py` to enable chat.",
            icon="🚨",
        )

    # Initialise session-state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if dev_mode and msg.get("metadata"):
                with st.expander("🔧 Dev Info", expanded=False):
                    meta = msg["metadata"]
                    if meta.get("tool_called"):
                        st.info(f"Tool called: `{meta['tool_called']}`")
                    if meta.get("rows_returned") is not None:
                        st.metric("Rows returned", meta["rows_returned"])
                    if meta.get("sql"):
                        st.code(meta["sql"], language="sql")
                    if meta.get("error"):
                        st.error(meta["error"])
                    if meta.get("logs"):
                        st.text_area("Logs", meta["logs"], height=150)

    # Chat input
    user_input = st.chat_input(
        "Ask about pricing rules, leakage, revenue opportunities…"
    )
    if user_input:
        # Show user message immediately
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "metadata": {}}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call agent and stream response
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking…"):
                response, metadata = call_agent(user_input)
            st.markdown(response)

            if dev_mode and metadata:
                with st.expander("🔧 Dev Info — Query & Logs", expanded=False):
                    if metadata.get("tool_called"):
                        st.info(f"Tool called: `{metadata['tool_called']}`")
                    if metadata.get("rows_returned") is not None:
                        st.metric("Rows returned", metadata["rows_returned"])
                    if metadata.get("sql"):
                        st.code(metadata["sql"], language="sql")
                    if metadata.get("error"):
                        st.error(metadata["error"])
                    if metadata.get("logs"):
                        st.text_area("Logs", metadata["logs"], height=150)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response, "metadata": metadata}
        )

        # Auto-refresh state so charts update in other tabs
        st.cache_data.clear()


# ===========================================================================
# TAB 1 — Rule Utilisation
# ===========================================================================

with tabs[1]:
    st.header("Pricing Rule Utilisation")
    df: pd.DataFrame | None = state.get("rule_utilization")

    if df is None or df.empty:
        _empty_state("Analyse rule utilisation")
    else:
        # KPI row
        total_records = int(df["record_count"].sum())
        fallback_records = int(
            df.loc[df["price_rule_source"].str.upper() == "DEFAULT", "record_count"].sum()
        )
        fallback_pct = fallback_records / total_records * 100 if total_records else 0
        avg_margin = df["avg_final_price_margin"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Active Records", f"{total_records:,}")
        c2.metric("Fallback (DEFAULT) Rate", f"{fallback_pct:.1f} %")
        c3.metric("Overall Avg Margin", f"{avg_margin:.4f}")

        st.markdown("---")

        source_agg = (
            df.groupby("price_rule_source", dropna=False)["record_count"]
            .sum()
            .reset_index()
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rule Source Distribution")
            fig_pie = px.pie(
                source_agg,
                names="price_rule_source",
                values="record_count",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        margin_agg = (
            df.groupby("price_rule_source", dropna=False)["avg_final_price_margin"]
            .mean()
            .reset_index()
            .sort_values("avg_final_price_margin", ascending=False)
        )
        with col2:
            st.subheader("Avg Margin by Rule Source")
            fig_bar = px.bar(
                margin_agg,
                x="price_rule_source",
                y="avg_final_price_margin",
                color="price_rule_source",
                labels={
                    "avg_final_price_margin": "Avg Margin",
                    "price_rule_source": "Rule Source",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if "country_code" in df.columns and "VENDOR_LOB_LVL2_DES" in df.columns:
            st.subheader("Fallback Rate Heatmap (Country × Vendor LOB)")
            default_df = df[df["price_rule_source"].str.upper() == "DEFAULT"]
            total_by_pair = (
                df.groupby(["country_code", "VENDOR_LOB_LVL2_DES"])["record_count"]
                .sum().reset_index().rename(columns={"record_count": "total"})
            )
            default_by_pair = (
                default_df.groupby(["country_code", "VENDOR_LOB_LVL2_DES"])["record_count"]
                .sum().reset_index().rename(columns={"record_count": "default_count"})
            )
            heatmap_df = total_by_pair.merge(
                default_by_pair, on=["country_code", "VENDOR_LOB_LVL2_DES"], how="left"
            ).fillna(0)
            heatmap_df["fallback_pct"] = heatmap_df["default_count"] / heatmap_df["total"] * 100

            pivot = heatmap_df.pivot_table(
                index="VENDOR_LOB_LVL2_DES", columns="country_code",
                values="fallback_pct", fill_value=0,
            )
            top_lobs = (
                df.groupby("VENDOR_LOB_LVL2_DES")["record_count"].sum()
                .nlargest(20).index.tolist()
            )
            top_countries = (
                df.groupby("country_code")["record_count"].sum()
                .nlargest(20).index.tolist()
            )
            pivot = pivot.loc[
                [l for l in top_lobs if l in pivot.index],
                [c for c in top_countries if c in pivot.columns],
            ]
            fig_heat = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn_r",
                labels={"color": "Fallback %"},
                aspect="auto",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        _dev_expander(df, "Raw Rule Utilisation Data")


# ===========================================================================
# TAB 2 — Country & Region Analysis
# ===========================================================================

with tabs[2]:
    st.header("Country & Region Analysis")
    df_country: pd.DataFrame | None = state.get("country_breakdown")

    if df_country is None or df_country.empty:
        _empty_state("Show me country breakdown of rule usage")
    else:
        country_agg = (
            df_country.groupby("country_code", dropna=False)
            .agg(
                record_count=("record_count", "sum"),
                avg_margin=("avg_final_price_margin", "mean"),
                fallback_rate=("fallback_rate_pct", "mean"),
            )
            .reset_index()
            .sort_values("record_count", ascending=False)
            .head(15)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 15 Countries by Volume")
            fig_country_bar = px.bar(
                country_agg,
                x="country_code",
                y="record_count",
                color="avg_margin",
                color_continuous_scale="RdYlGn",
                labels={
                    "record_count": "Record Count",
                    "country_code": "Country",
                    "avg_margin": "Avg Margin",
                },
            )
            st.plotly_chart(fig_country_bar, use_container_width=True)

        with col2:
            st.subheader("Top 15 Countries by Fallback Rate")
            fallback_sorted = country_agg.sort_values("fallback_rate", ascending=False)
            fig_fallback = px.bar(
                fallback_sorted,
                x="country_code",
                y="fallback_rate",
                color="fallback_rate",
                color_continuous_scale="RdYlGn_r",
                labels={"fallback_rate": "Fallback Rate %", "country_code": "Country"},
            )
            st.plotly_chart(fig_fallback, use_container_width=True)

        st.subheader("Rule Source Mix by Company Code")
        company_source = (
            df_country.groupby(["company_code", "price_rule_source"], dropna=False)["record_count"]
            .sum().reset_index()
        )
        fig_company_stack = px.bar(
            company_source,
            x="company_code",
            y="record_count",
            color="price_rule_source",
            barmode="stack",
            labels={
                "record_count": "Record Count",
                "company_code": "Company Code",
                "price_rule_source": "Rule Source",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_company_stack, use_container_width=True)

        _dev_expander(df_country, "Raw Country Breakdown Data")


# ===========================================================================
# TAB 3 — Revenue Opportunity
# ===========================================================================

with tabs[3]:
    st.header("Revenue Uplift Opportunity")
    df_rev: pd.DataFrame | None = state.get("revenue_opportunity")

    if df_rev is None or df_rev.empty:
        _empty_state("Show me the top revenue uplift opportunities")
    else:
        total_uplift = df_rev["estimated_uplift"].sum()
        n_skus = df_rev["sku_number"].nunique()
        avg_gap = (df_rev["non_default_avg_margin"] - df_rev["default_avg_margin"]).mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Estimated Uplift", f"{total_uplift:,.2f}")
        c2.metric("SKUs on DEFAULT Rules", f"{n_skus:,}")
        c3.metric("Avg Margin Gap (targeted − default)", f"{avg_gap:.4f}")

        st.markdown("---")

        st.subheader("Top 10 SKU/Country Uplift Opportunities")
        top10 = df_rev.head(10).copy()
        top10["label"] = (
            top10["sku_number"].astype(str) + " / " + top10["country_code"].astype(str)
        )
        fig_waterfall = go.Figure(go.Waterfall(
            name="Uplift",
            orientation="v",
            x=top10["label"].tolist(),
            y=top10["estimated_uplift"].tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            totals={"marker": {"color": "#3498db"}},
        ))
        fig_waterfall.update_layout(
            xaxis_title="SKU / Country",
            yaxis_title="Estimated Uplift",
            showlegend=False,
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.subheader("Full Opportunity Table")
        st.dataframe(
            df_rev[["sku_number", "country_code", "default_record_count",
                    "default_avg_margin", "non_default_avg_margin", "estimated_uplift"]],
            use_container_width=True,
        )

        _dev_expander(df_rev, "Raw Revenue Opportunity Data")


# ===========================================================================
# TAB 4 — Pricing Leakage
# ===========================================================================

with tabs[4]:
    st.header("Pricing Leakage Alerts")
    df_leak: pd.DataFrame | None = state.get("leakage")

    if df_leak is None or df_leak.empty:
        _empty_state("Flag all pricing leakage records with margin below 10 %")
    else:
        total_leak = len(df_leak)
        low_margin_cnt = (df_leak["leakage_type"] == "low_margin").sum()
        floor_hit_cnt = (df_leak["leakage_type"] == "floor_hit").sum()
        override_cnt = (df_leak["leakage_type"] == "override").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Leakage Records", f"{total_leak:,}")
        c2.metric("Low Margin", f"{low_margin_cnt:,}")
        c3.metric("Floor Hits", f"{floor_hit_cnt:,}")
        c4.metric("Manual Overrides", f"{override_cnt:,}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Leakage Type Breakdown")
            type_data = pd.DataFrame({
                "Type": ["Low Margin", "Floor Hit", "Manual Override"],
                "Count": [low_margin_cnt, floor_hit_cnt, override_cnt],
            })
            fig_donut = px.pie(
                type_data,
                names="Type",
                values="Count",
                hole=0.5,
                color_discrete_map={
                    "Low Margin": "#e74c3c",
                    "Floor Hit": "#e67e22",
                    "Manual Override": "#9b59b6",
                },
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            st.subheader("Price vs Margin (coloured by leakage type)")
            scatter_df = df_leak.copy()
            fig_scatter = px.scatter(
                scatter_df.head(2000),
                x="calculated_price",
                y="margin_pct",
                color="leakage_type",
                opacity=0.6,
                labels={
                    "calculated_price": "Calculated Price",
                    "margin_pct": "Margin %",
                    "leakage_type": "Leakage Type",
                },
                color_discrete_map={
                    "low_margin": "#e74c3c",
                    "floor_hit": "#e67e22",
                    "override": "#9b59b6",
                    "other": "#95a5a6",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Top 20 Leakage Records")
        display_cols = [
            "sku_number", "customer_number", "country_code",
            "pricing_rule", "calculated_price", "final_cost",
            "final_price_margin", "margin_pct", "leakage_type",
        ]
        display_cols = [c for c in display_cols if c in df_leak.columns]
        st.dataframe(df_leak[display_cols].head(20), use_container_width=True)

        _dev_expander(df_leak, "Raw Leakage Data")


# ===========================================================================
# TAB 5 — ML Model Results  (Vertex AI AutoML)
# ===========================================================================

with tabs[5]:
    st.header("ML Model Training Results")
    st.caption("Powered by Vertex AI AutoML Tabular — model-level evaluation metrics only.")
    model_results: dict | None = state.get("model_results")

    if model_results is None:
        _empty_state("Run model training")
    else:
        # ---- Conversion Classifier metrics ---------------------------------
        st.subheader("Conversion Classifier  (target: quote_accepted)")

        conv_auc     = model_results.get("conversion_auc")
        conv_logloss = model_results.get("conversion_log_loss")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "AUC-ROC",
            f"{conv_auc:.4f}" if conv_auc is not None else "N/A",
            help="Area under the ROC curve — higher is better (max 1.0).",
        )
        c2.metric(
            "Log Loss",
            f"{conv_logloss:.4f}" if conv_logloss is not None else "N/A",
            help="Cross-entropy loss — lower is better (min 0.0).",
        )
        conv_endpoint = model_results.get("conversion_endpoint", "")
        c3.metric(
            "Deployed Endpoint",
            "✅ Live" if conv_endpoint else "—",
            help=conv_endpoint or "Not yet deployed.",
        )
        if conv_endpoint:
            st.code(conv_endpoint, language=None)

        st.markdown("---")

        # ---- Margin Regressor metrics --------------------------------------
        st.subheader("Margin Regressor  (target: margin_pct)")

        margin_rmse = model_results.get("margin_rmse")
        margin_r2   = model_results.get("margin_r2")
        margin_mae  = model_results.get("margin_mae")

        c4, c5, c6, c7 = st.columns(4)
        c4.metric(
            "RMSE",
            f"{margin_rmse:.4f}" if margin_rmse is not None else "N/A",
            help="Root Mean Squared Error — lower is better.",
        )
        c5.metric(
            "R²",
            f"{margin_r2:.4f}" if margin_r2 is not None else "N/A",
            help="Coefficient of determination — higher is better (max 1.0).",
        )
        c6.metric(
            "MAE",
            f"{margin_mae:.4f}" if margin_mae is not None else "N/A",
            help="Mean Absolute Error — lower is better.",
        )
        margin_endpoint = model_results.get("margin_endpoint", "")
        c7.metric(
            "Deployed Endpoint",
            "✅ Live" if margin_endpoint else "—",
            help=margin_endpoint or "Not yet deployed.",
        )
        if margin_endpoint:
            st.code(margin_endpoint, language=None)

        st.markdown("---")

        st.info(
            "ℹ️  Vertex AI AutoML Tabular provides aggregate model-level evaluation "
            "metrics only.  Per-feature importance scores are not exposed via the API.  "
            "To inspect feature attribution, open the model in the "
            "[Google Cloud Console → Vertex AI → Model Registry]"
            "(https://console.cloud.google.com/vertex-ai/models).",
            icon="ℹ️",
        )


# ===========================================================================
# TAB 6 — Rule Recommendations
# ===========================================================================

with tabs[6]:
    st.header("Rule Recommendations")
    recommendations: dict | None = state.get("rule_recommendations")

    if recommendations is None:
        _empty_state("Give me rule recommendations")
    else:
        missing  = recommendations.get("missing_rules", [])
        consol   = recommendations.get("consolidation_candidates", [])
        redundant = recommendations.get("redundant_rules", [])

        with st.expander(
            "🔴  Missing Rules  — SKU/Country pairs with > 50 % fallback usage",
            expanded=True,
        ):
            if missing:
                st.code("\n".join(missing), language=None)
            else:
                st.success("No missing rule candidates identified.")

        with st.expander(
            "🟡  Consolidation Candidates  — Rules spanning ≥ 5 countries",
            expanded=True,
        ):
            if consol:
                st.code("\n".join(consol), language=None)
            else:
                st.success("No consolidation candidates identified.")

        with st.expander(
            "🟢  Redundant Rules  — Rules with < 5 total records",
            expanded=True,
        ):
            if redundant:
                st.code("\n".join(redundant), language=None)
            else:
                st.success("No redundant rules identified.")


# ===========================================================================
# TAB 7 — Train Model  (Vertex AI AutoML)
# ===========================================================================

with tabs[7]:
    st.header("🧠 Train Model")
    st.caption(
        "Train Vertex AI AutoML Tabular models on a specific date range of pricing data. "
        "Training runs entirely on GCP and typically takes **1–3 hours**. "
        "Estimated cost: **~$20–30 per run**."
    )

    # ---- Read current training status from disk ------------------------------
    _ts = read_training_status()
    _status = _ts.get("status")  # "running" | "complete" | "failed" | None

    # ---- Date pickers (disabled while training is in progress) ---------------
    st.subheader("1. Select Training Date Range")
    col_from, col_to = st.columns(2)
    default_from = date(2025, 12, 25)
    default_to   = date(2025, 12, 31)

    _picker_disabled = (_status == "running")

    with col_from:
        date_from = st.date_input(
            "Training data from",
            value=default_from,
            help="Lower bound on the `created` column (inclusive).",
            disabled=_picker_disabled,
        )
    with col_to:
        date_to = st.date_input(
            "Training data to",
            value=default_to,
            min_value=date_from,
            help="Upper bound on the `created` column (inclusive).",
            disabled=_picker_disabled,
        )

    date_from_str = str(date_from)
    date_to_str   = str(date_to)

    # ---- Row count preview ---------------------------------------------------
    st.subheader("2. Preview Record Count")
    if st.button(
        "🔍 Count Records in Date Range",
        use_container_width=False,
        disabled=_picker_disabled,
    ):
        with st.spinner("Querying BigQuery for row count…"):
            try:
                from google.cloud import bigquery as _bq  # noqa: PLC0415
                from config import GCP_PROJECT_ID, BQ_FULL_TABLE  # noqa: PLC0415

                _client = _bq.Client(project=GCP_PROJECT_ID)
                _sql = f"""
                    SELECT COUNT(*) AS cnt
                    FROM `{BQ_FULL_TABLE}`
                    WHERE created BETWEEN '{date_from_str}' AND '{date_to_str}'
                """
                _row_count = list(_client.query(_sql).result())[0]["cnt"]
                st.session_state["training_row_count"] = _row_count
            except Exception as _exc:
                st.session_state["training_row_count"] = None
                st.error(f"Row count query failed: {_exc}")

    if "training_row_count" in st.session_state:
        _rc = st.session_state["training_row_count"]
        if _rc is not None:
            if _rc == 0:
                st.warning(
                    f"⚠️ **0 records** found between {date_from_str} and {date_to_str}. "
                    "Widen the date range or check the `created` column in your table."
                )
            else:
                st.success(f"✅ **{_rc:,} records** available for training.")

    # ---- Existing endpoints display ------------------------------------------
    endpoints_path = _ROOT / "endpoints.json"
    if endpoints_path.exists():
        try:
            _ep = json.loads(endpoints_path.read_text())
            st.info(
                f"**Existing deployed endpoints detected** — training will redeploy:\n\n"
                f"- Conversion: `{_ep.get('conversion_endpoint', 'N/A')}`\n"
                f"- Margin: `{_ep.get('margin_endpoint', 'N/A')}`"
            )
        except Exception:
            pass

    # ---- Dev mode SQL preview ------------------------------------------------
    if dev_mode:
        with st.expander("🔧 View ML View SQL (Dev Mode)", expanded=False):
            try:
                from ml.trainer import build_ml_view_sql  # noqa: PLC0415
                st.code(build_ml_view_sql(date_from_str, date_to_str), language="sql")
            except Exception as _sql_exc:
                st.error(f"Could not generate SQL preview: {_sql_exc}")

    st.markdown("---")

    # ---- Confirm + start training -------------------------------------------
    st.subheader("3. Start Training")
    st.warning(
        "⚠️ **Cost warning:** Each training run incurs approximately **$20–30** in "
        "Vertex AI compute charges on your GCP project. "
        "Training cannot be stopped once started from this UI — cancel via "
        "[Vertex AI → Training Jobs](https://console.cloud.google.com/vertex-ai/training/training-pipelines) "
        "in the GCP Console if needed."
    )

    confirmed = st.checkbox(
        "I understand the cost and want to start training",
        value=False,
        key="training_confirmed",
        disabled=_picker_disabled,
    )

    # Button disabled while training is in progress OR checkbox not ticked
    _btn_disabled = _picker_disabled or not confirmed

    if st.button(
        f"🚀 Start Training  ({date_from_str} to {date_to_str})",
        type="primary",
        disabled=_btn_disabled,
        use_container_width=True,
    ):
        # Re-read status immediately before spawning to prevent double-spawn
        # (user double-click or rapid reruns)
        _current = read_training_status()
        if _current.get("status") == "running":
            st.warning(
                f"Training is already in progress "
                f"({_current.get('date_from','?')} to {_current.get('date_to','?')}). "
                "Please wait for it to finish before starting a new run."
            )
            st.rerun()
        else:
            # Atomically mark running before spawning thread
            write_training_status("running", date_from=date_from_str, date_to=date_to_str)
            _t = threading.Thread(
                target=_run_training,
                args=(date_from_str, date_to_str),
                daemon=True,
            )
            _t.start()
            st.rerun()

    # ---- Training status panel (polled from disk) ----------------------------
    if _status == "running":
        _d_from = _ts.get("date_from", "?")
        _d_to   = _ts.get("date_to", "?")
        _updated = _ts.get("updated_at", "")
        _lock_exists = (_ROOT / "training.lock").exists()
        if _lock_exists:
            st.info(
                f"⏳ **Training in progress** for records from **{_d_from}** to **{_d_to}**.\n\n"
                f"Last updated: {_updated[:19].replace('T', ' ')} UTC\n\n"
                "Training runs on Vertex AI and typically takes 1–3 hours. "
                "Refresh this page manually to check for updates. "
                "You can safely navigate to other dashboard tabs while waiting.",
                icon="⏳",
            )
        else:
            st.warning(
                f"⚠️ **Status is stuck on 'running'** (from {_updated[:19].replace('T', ' ')} UTC) "
                "but no training process is active. The previous run may have crashed. "
                "Click **Reset Status** to unlock the form.",
            )
            if st.button("Reset Status", type="secondary"):
                try:
                    TRAINING_STATUS_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                st.rerun()

    elif _status == "complete":
        st.success("✅ **Training complete!** Models have been deployed to Vertex AI.")
        _results = _ts.get("results", {})
        if _results:
            st.subheader("Model Evaluation Metrics")
            c1, c2, c3 = st.columns(3)
            _auc = _results.get("conversion_auc")
            _ll  = _results.get("conversion_log_loss")
            c1.metric("Conversion AUC-ROC",  f"{_auc:.4f}" if _auc else "N/A")
            c2.metric("Conversion Log Loss", f"{_ll:.4f}"  if _ll  else "N/A")
            _rmse = _results.get("margin_rmse")
            _r2   = _results.get("margin_r2")
            _mae  = _results.get("margin_mae")
            c3.metric("Margin RMSE", f"{_rmse:.4f}" if _rmse else "N/A")
            c4, c5 = st.columns(2)
            c4.metric("Margin R²",  f"{_r2:.4f}"  if _r2  else "N/A")
            c5.metric("Margin MAE", f"{_mae:.4f}" if _mae else "N/A")
            st.caption(
                "Endpoint resource names saved to `endpoints.json`. "
                "View full metrics in the **🤖 ML Model Results** tab."
            )
        # Allow re-training by clearing the status file
        if st.button("🔁 Train Again", use_container_width=False):
            try:
                TRAINING_STATUS_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()

    elif _status == "failed":
        _err = _ts.get("error", "Unknown error")
        st.error(f"❌ **Training failed:** {_err}")
        st.caption(
            "Check `agent_session.log` for full traceback details, or enable "
            "**Dev Mode** in the sidebar to view live logs."
        )
        # Allow retry
        if st.button("🔁 Retry Training", use_container_width=False):
            try:
                TRAINING_STATUS_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()
