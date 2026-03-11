"""
logging_config.py
Structured JSON + console logging for the Pricing Intelligence Agent.

Call setup_logging() once at the very top of main.py before any other
imports so that all modules share the same handlers and format.

Dev mode (DEV_MODE=true env var, or dev_mode=True arg):
  • Sets log level to DEBUG for both handlers
  • Console output includes DEBUG-level BQ queries, row counts, tool calls

Production mode (default):
  • File handler always captures DEBUG+
  • Console only shows INFO+
"""

import json
import logging
from datetime import datetime, timezone

LOG_FILE = "agent_session.log"


class JsonFormatter(logging.Formatter):
    """
    Formats each log record as a single-line JSON object so that structured
    log ingest tools (Cloud Logging, Datadog, etc.) can parse fields directly.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "module":    record.module,
            "funcName":  record.funcName,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(dev_mode: bool = False) -> None:
    """
    Configure the root logger with:
      • A rotating-style file handler (agent_session.log) — always DEBUG level
      • A console handler — DEBUG in dev mode, INFO in production

    Calling this function more than once is safe; duplicate handlers are
    removed before new ones are added.

    Parameters
    ----------
    dev_mode : bool
        If True, console output includes DEBUG-level messages (SQL queries,
        row counts, tool invocations, etc.).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root captures everything; handlers filter

    # Remove any handlers added by a previous call (e.g. hot reload in dev)
    root.handlers.clear()

    # ---- File handler — always DEBUG, JSON format --------------------------
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # ---- Console handler — human-readable, level depends on mode -----------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(module)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    console_handler.setLevel(logging.DEBUG if dev_mode else logging.INFO)
    root.addHandler(console_handler)

    logging.info(
        "Logging initialised — file: %s | dev_mode: %s", LOG_FILE, dev_mode
    )
