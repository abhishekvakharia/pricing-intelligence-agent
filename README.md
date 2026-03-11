# Pricing Intelligence Agent

An AI-powered pricing analysis platform that connects to a BigQuery pricing
fact table, runs analytical queries and XGBoost ML models on historical pricing
data, and exposes a conversational agent interface alongside an interactive
Streamlit analytics dashboard.

---

## Architecture

```
BigQuery pricing_table
        ↓
bq/queries.py  (read-only SQL layer)
        ↓
agent/tools.py (8 agent-callable tools)
        ↓
agent/agent.py (static Gemini-backed ADK agent)
        ↓
main.py  ──────────────────────────────┐
        ↓                              ↓
CLI Chat (stdin/stdout)     dashboard/app.py (Streamlit)
                                       ↓
                            http://localhost:8501
```

---

## 1. Setup — fill in `config.py`

Open `config.py` and replace the three placeholder values:

```python
GCP_PROJECT_ID = "your-actual-gcp-project-id"
BQ_DATASET_NAME = "your_dataset"
BQ_TABLE_NAME   = "pricing_table"
```

`BQ_FULL_TABLE` is derived automatically — do not edit it.

---

## 2. Authentication

### Option A — User credentials (development)

```bash
gcloud auth application-default login
```

### Option B — Service account (production)

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

The service account needs at minimum:
- `roles/bigquery.dataViewer` on the dataset
- `roles/bigquery.jobUser` on the project

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Requires **Python 3.11+**.

---

## 4. Run

```bash
python main.py
```

This will:
1. Start the Streamlit dashboard in the background on port 8501
2. Launch the conversational agent in your terminal

---

## 5. Dashboard

The dashboard opens automatically at:

```
http://localhost:8501
```

**Tabs available:**

| Tab | Contents |
|-----|----------|
| Rule Utilisation | Pie / bar / heatmap of rule source distribution and margins |
| Country & Region | Country volume, fallback rate, regional rule mix |
| Revenue Opportunity | Waterfall chart of top uplift SKU/country pairs |
| Pricing Leakage | Donut + scatter + table of low-margin / floor-hit / override records |
| ML Model Results | AUC, MAE, R², feature importances, actual vs predicted scatter |
| Rule Recommendations | Missing / consolidation / redundant rule tables |

The dashboard reads from a shared state file written by the agent when it runs
a tool.  If a tab shows "No data loaded yet", ask the agent the corresponding
question and then click **Refresh from Agent State** in the sidebar.

---

## 6. Agent — example prompts

```
What percentage of quotes used the default pricing rule last month?

Which countries have the highest fallback rule usage?

Show me the top revenue uplift opportunities

Run model training

Flag all pricing leakage records with margin below 5 %

What columns are available in the pricing table?

SELECT sku_number, COUNT(*) as cnt FROM `project.dataset.pricing_table`
WHERE db_rec_del_flag != 'Y' GROUP BY sku_number ORDER BY cnt DESC LIMIT 10
```

---

## 7. Project structure

```
pricing-intelligence-agent/
├── config.py                  # GCP placeholders — edit before running
├── main.py                    # Entry point
├── requirements.txt
├── README.md
├── agent/
│   ├── __init__.py
│   ├── agent.py               # Static ADK agent (Gemini-backed)
│   └── tools.py               # 8 agent tools + shared dashboard state
├── bq/
│   ├── __init__.py
│   └── queries.py             # All BigQuery query functions
├── ml/
│   ├── __init__.py
│   └── trainer.py             # XGBoost training pipeline
├── dashboard/
│   ├── __init__.py
│   └── app.py                 # Streamlit + Plotly dashboard
├── models/                    # Auto-created — stores trained model files
│   ├── conversion_model.json
│   └── margin_model.json
└── state/                     # Auto-created — stores agent ↔ dashboard state
    └── dashboard_state.pkl
```

---

## 8. Key constraints

- **Read-only** — the agent never modifies BigQuery data
- **Static instructions** — agent behaviour is defined at build time in `agent/agent.py` and cannot be changed at runtime
- **All BQ references via `config.py`** — `BQ_FULL_TABLE` is the single source of truth; no table names are hardcoded elsewhere
- **SCD filter** — every query applies `db_rec_del_flag != 'Y' AND db_rec_close_date IS NULL`
- **Currency normalisation** — cross-country price comparisons use `calculated_price × currency_exchange_rate`
- **Confidence threshold** — ML recommendations are only surfaced at ≥ 70 % confidence

---

## 9. Troubleshooting

| Problem | Resolution |
|---------|-----------|
| `ModuleNotFoundError: google.adk` | Run `pip install -r requirements.txt` |
| `google.auth.exceptions.DefaultCredentialsError` | Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS` |
| `NOT_FOUND: Dataset not found` | Check `BQ_DATASET_NAME` and `GCP_PROJECT_ID` in `config.py` |
| Dashboard shows "No data loaded yet" | Ask the agent a question first, then click Refresh in the sidebar |
| XGBoost training fails with `ValueError` | Ensure the pricing table has sufficient rows (> 1 000 recommended) |
