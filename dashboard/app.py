"""
dashboard/app.py
Streamlit + Plotly interactive analytics dashboard for the Pricing
Intelligence Agent.

Run standalone:
    streamlit run dashboard/app.py

Or launch automatically via main.py which starts this as a subprocess.

State sharing:
    The dashboard reads agent/tools.dashboard_state via the pickled state
    file written by the agent process (state/dashboard_state.pkl).
    If the file does not exist yet, each tab shows an empty-state prompt
    instructing the user to ask the agent first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on PYTHONPATH when running as a subprocess
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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💰 Pricing Intelligence")
    st.markdown("---")

    if st.button("🔄  Refresh from Agent State", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(
        "Charts update automatically when the agent runs a tool.  "
        "Ask the agent to analyse data first, then refresh here."
    )
    st.markdown("**Example prompts:**")
    st.markdown(
        "- *What % of quotes used the default rule last month?*\n"
        "- *Show me the top revenue uplift opportunities*\n"
        "- *Flag all leakage records with margin below 5 %*\n"
        "- *Run model training*"
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
        f"No data loaded yet.  Ask the agent: **\"{tool_hint}\"** and then "
        "click **Refresh from Agent State** in the sidebar.",
        icon="🤖",
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "📊 Rule Utilisation",
    "🌍 Country & Region",
    "💵 Revenue Opportunity",
    "⚠️ Pricing Leakage",
    "🤖 ML Model Results",
    "📋 Rule Recommendations",
])

# ===========================================================================
# TAB 1 — Rule Utilisation
# ===========================================================================

with tabs[0]:
    st.header("Pricing Rule Utilisation")
    df: pd.DataFrame | None = state.get("rule_utilization")

    if df is None or df.empty:
        _empty_state("Analyse rule utilisation")
    else:
        # --- KPI row
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

        # --- Pie chart: rule source distribution
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

        # --- Bar chart: avg margin by rule source
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
                labels={"avg_final_price_margin": "Avg Margin", "price_rule_source": "Rule Source"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Heatmap: fallback rate by country × vendor LOB
        if "country_code" in df.columns and "VENDOR_LOB_LVL2_DES" in df.columns:
            st.subheader("Fallback Rate Heatmap (Country × Vendor LOB)")
            default_df = df[df["price_rule_source"].str.upper() == "DEFAULT"]
            total_by_pair = (
                df.groupby(["country_code", "VENDOR_LOB_LVL2_DES"])["record_count"]
                .sum()
                .reset_index()
                .rename(columns={"record_count": "total"})
            )
            default_by_pair = (
                default_df.groupby(["country_code", "VENDOR_LOB_LVL2_DES"])["record_count"]
                .sum()
                .reset_index()
                .rename(columns={"record_count": "default_count"})
            )
            heatmap_df = total_by_pair.merge(default_by_pair, on=["country_code", "VENDOR_LOB_LVL2_DES"], how="left").fillna(0)
            heatmap_df["fallback_pct"] = heatmap_df["default_count"] / heatmap_df["total"] * 100

            pivot = heatmap_df.pivot_table(
                index="VENDOR_LOB_LVL2_DES", columns="country_code", values="fallback_pct", fill_value=0
            )
            # Limit to top-20 LOBs and top-20 countries by volume for readability
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


# ===========================================================================
# TAB 2 — Country & Region Analysis
# ===========================================================================

with tabs[1]:
    st.header("Country & Region Analysis")
    df_country: pd.DataFrame | None = state.get("country_breakdown")

    if df_country is None or df_country.empty:
        _empty_state("Show me country breakdown of rule usage")
    else:
        # --- Bar: top 15 countries by record volume coloured by avg margin
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

        # --- Bar: fallback rate by country (top 15)
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

        # --- Stacked bar: rule type mix per region
        st.subheader("Rule Source Mix by Region")
        region_source = (
            df_country.groupby(["REGION", "price_rule_source"], dropna=False)["record_count"]
            .sum()
            .reset_index()
        )
        fig_region_stack = px.bar(
            region_source,
            x="REGION",
            y="record_count",
            color="price_rule_source",
            barmode="stack",
            labels={
                "record_count": "Record Count",
                "REGION": "Region",
                "price_rule_source": "Rule Source",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_region_stack, use_container_width=True)


# ===========================================================================
# TAB 3 — Revenue Opportunity
# ===========================================================================

with tabs[2]:
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

        # --- Waterfall chart: top 10 SKU/country by uplift
        st.subheader("Top 10 SKU/Country Uplift Opportunities")
        top10 = df_rev.head(10).copy()
        top10["label"] = top10["sku_number"].astype(str) + " / " + top10["country_code"].astype(str)

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

        # --- Detail table
        st.subheader("Full Opportunity Table")
        st.dataframe(
            df_rev[["sku_number", "country_code", "default_record_count",
                    "default_avg_margin", "non_default_avg_margin", "estimated_uplift"]],
            use_container_width=True,
        )


# ===========================================================================
# TAB 4 — Pricing Leakage
# ===========================================================================

with tabs[3]:
    st.header("Pricing Leakage Alerts")
    df_leak: pd.DataFrame | None = state.get("leakage")

    if df_leak is None or df_leak.empty:
        _empty_state("Flag all pricing leakage records with margin below 10 %")
    else:
        total_leak = len(df_leak)
        low_margin_cnt = (df_leak["low_margin_flag"] == "low_margin").sum()
        floor_hit_cnt = (df_leak["floor_hit_flag"] == "floor_hit").sum()
        override_cnt = (df_leak["override_flag"] == "manual_override").sum()
        margin_impact = df_leak["final_price_margin"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Leakage Records", f"{total_leak:,}")
        c2.metric("Low Margin", f"{low_margin_cnt:,}")
        c3.metric("Floor Hits", f"{floor_hit_cnt:,}")
        c4.metric("Manual Overrides", f"{override_cnt:,}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        # --- Donut: leakage type breakdown
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

        # --- Scatter: calculated_price vs final_price_margin coloured by type
        with col2:
            st.subheader("Price vs Margin (coloured by leakage type)")
            scatter_df = df_leak.copy()
            scatter_df["leakage_type"] = (
                scatter_df["low_margin_flag"]
                .fillna(scatter_df["floor_hit_flag"])
                .fillna(scatter_df["override_flag"])
                .fillna("other")
            )
            fig_scatter = px.scatter(
                scatter_df.head(2000),   # cap for performance
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
                    "manual_override": "#9b59b6",
                    "other": "#95a5a6",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Top 20 records table
        st.subheader("Top 20 Leakage Records")
        display_cols = [
            "sku_number", "customer_number", "country_code",
            "pricing_rule", "calculated_price", "final_cost",
            "final_price_margin", "margin_pct",
            "low_margin_flag", "floor_hit_flag", "override_flag",
        ]
        display_cols = [c for c in display_cols if c in df_leak.columns]
        st.dataframe(df_leak[display_cols].head(20), use_container_width=True)


# ===========================================================================
# TAB 5 — ML Model Results  (Vertex AI AutoML)
# ===========================================================================

with tabs[4]:
    st.header("ML Model Training Results")
    st.caption("Powered by Vertex AI AutoML Tabular — model-level evaluation metrics only.")
    model_results: dict | None = state.get("model_results")

    if model_results is None:
        _empty_state("Run model training")
    else:
        # ---- Conversion Classifier metrics ---------------------------------
        st.subheader("Conversion Classifier  (target: quote_accepted)")

        conv_auc      = model_results.get("conversion_auc")
        conv_logloss  = model_results.get("conversion_log_loss")

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

        # ---- AutoML note ---------------------------------------------------
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

with tabs[5]:
    st.header("Rule Recommendations")
    recommendations: dict | None = state.get("rule_recommendations")

    if recommendations is None:
        _empty_state("Give me rule recommendations")
    else:
        missing = recommendations.get("missing_rules", [])
        consol = recommendations.get("consolidation_candidates", [])
        redundant = recommendations.get("redundant_rules", [])

        with st.expander("🔴  Missing Rules  — SKU/Country pairs with > 50 % fallback usage", expanded=True):
            if missing:
                st.code("\n".join(missing), language=None)
            else:
                st.success("No missing rule candidates identified.")

        with st.expander("🟡  Consolidation Candidates  — Rules spanning ≥ 5 countries", expanded=True):
            if consol:
                st.code("\n".join(consol), language=None)
            else:
                st.success("No consolidation candidates identified.")

        with st.expander("🟢  Redundant Rules  — Rules with < 5 total records", expanded=True):
            if redundant:
                st.code("\n".join(redundant), language=None)
            else:
                st.success("No redundant rules identified.")
