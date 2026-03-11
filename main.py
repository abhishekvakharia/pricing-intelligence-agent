"""
main.py
Entry point for the Pricing Intelligence Agent.

Startup sequence:
  1. Initialise structured logging (JSON file + console)
  2. Start the Streamlit dashboard in a background subprocess (port 8501)
  3. Run BigQuery diagnostics — logs table health & chooses active-record filter
  4. Start the agent HTTP server (port 8502) — blocking, keeps process alive

Dev mode:
    DEV_MODE=true python main.py
    → Sets log level to DEBUG on the console; dashboard shows raw SQL/logs.

Model training:
    Trigger training manually via the '🧠 Train Model' tab in the dashboard,
    or by asking the agent: "Train the model on data from 2025-12-25 to 2025-12-31".
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Logging must be configured before any other import that uses logging
# ---------------------------------------------------------------------------

from logging_config import setup_logging  # noqa: E402 (intentionally first)

_dev_mode: bool = os.environ.get("DEV_MODE", "false").lower() == "true"
setup_logging(dev_mode=_dev_mode)

# ---------------------------------------------------------------------------
# 1. Dashboard subprocess
# ---------------------------------------------------------------------------

def start_dashboard() -> None:
    """Launch Streamlit dashboard on port 8501 (non-blocking)."""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    logging.info("[MAIN] Starting Streamlit dashboard → http://localhost:8501")
    subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# 2. BigQuery startup diagnostics
# ---------------------------------------------------------------------------

def run_startup_diagnostics() -> None:
    """
    Query the pricing table and log basic health metrics.
    price_fact_us has no SCD columns — all rows are valid.
    """
    logging.info("[MAIN] Running BigQuery startup diagnostics…")
    try:
        from bq.queries import run_diagnostic_query  # noqa: PLC0415

        diag = run_diagnostic_query()

        total      = diag.get("total_rows", 0)
        rule_srcs  = diag.get("distinct_rule_sources", 0)
        countries  = diag.get("distinct_countries", 0)
        orders     = diag.get("orders", 0)
        earliest   = diag.get("earliest_record")
        latest     = diag.get("latest_record")

        if total == 0:
            logging.error(
                "[MAIN] ⚠️  Table appears to be EMPTY (0 rows). "
                "Check GCP_PROJECT_ID, BQ_DATASET_NAME, BQ_TABLE_NAME in config.py.",
            )
        else:
            logging.info(
                "[MAIN] ✅ BQ connection healthy. "
                "%d total rows | %d orders | %d rule sources | %d countries | "
                "date range: %s → %s",
                total, orders, rule_srcs, countries, earliest, latest,
            )

        logging.info("[MAIN] No SCD filter applied — all rows in price_fact_us are active.")

    except Exception as exc:
        logging.error(
            "[MAIN] ❌ BQ diagnostic failed: %s — "
            "Check that config.py placeholders are filled and GCP auth is configured.",
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Pricing Intelligence Agent")
    print("=" * 60)
    if _dev_mode:
        print("  DEV MODE ON — debug logging enabled")
    print()

    # 1. Start dashboard (background)
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    # Give Streamlit a moment to bind before printing the ready message
    time.sleep(2)

    # 2. BQ diagnostics (blocking — typically < 5 seconds)
    run_startup_diagnostics()

    # 3. Start agent HTTP server — this blocks and keeps the process alive
    logging.info("[MAIN] Open http://localhost:8501 to chat in the browser.")
    logging.info("[MAIN] Starting agent HTTP server on port 8502…")

    from agent.server import start_agent_server  # noqa: PLC0415

    start_agent_server(port=8502)
