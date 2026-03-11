"""
main.py
Entry point for the Pricing Intelligence Agent.

Starts the Streamlit dashboard in a background subprocess, then launches
the ADK conversational agent in an interactive CLI loop.

Usage:
    python main.py
"""

import subprocess
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------

def start_dashboard() -> None:
    """Launch the Streamlit dashboard as a non-blocking subprocess."""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    try:
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
             "--server.headless", "true",
             "--server.port", "8501"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Dashboard started → http://localhost:8501")
    except Exception as exc:
        print(f"[WARNING] Could not start dashboard: {exc}")


# ---------------------------------------------------------------------------
# Agent launcher
# ---------------------------------------------------------------------------

def start_agent() -> None:
    """Initialise and run the ADK agent in interactive (CLI) mode."""
    try:
        from google.adk.runners import Runner
        from agent.agent import root_agent

        runner = Runner(agent=root_agent)
        runner.run_interactive()   # blocking — reads from stdin
    except ImportError as exc:
        print(
            f"\n[ERROR] Could not import Google ADK: {exc}\n"
            "Make sure you have installed all dependencies:\n"
            "  pip install -r requirements.txt\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Pricing Intelligence Agent")
    print("=" * 60)
    print()

    # 1. Start dashboard in background thread
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    # Give Streamlit a moment to bind to its port before printing the prompt
    time.sleep(2)

    print()
    print("Agent ready.  Type your question and press Enter.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    # 2. Run the agent (blocking)
    start_agent()
