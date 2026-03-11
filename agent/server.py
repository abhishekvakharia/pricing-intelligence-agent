"""
agent/server.py
Lightweight HTTP server that exposes the ADK agent as a REST endpoint so
the Streamlit dashboard (running in a separate process) can call it via
requests.post().

Endpoint:
    POST http://localhost:8502/chat
    Body:  { "message": "<user prompt>" }
    Returns: { "response": "<agent reply>", "metadata": { ... } }

Metadata keys returned (best-effort — the ADK runner may not expose all):
    tool_called   : str  — name of the last tool the agent invoked
    rows_returned : int  — number of BQ rows returned by that tool
    error         : str  — exception message if the call failed
    logs          : str  — last 20 lines of agent_session.log

Called from main.py via start_agent_server().
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Bootstrap ADK runner  (imported lazily to avoid circular deps at module load)
# ---------------------------------------------------------------------------

_runner = None


def _get_runner():
    """Lazily initialise the ADK Runner the first time a request arrives."""
    global _runner
    if _runner is None:
        from google.adk.runners import Runner  # noqa: PLC0415
        from agent.agent import root_agent     # noqa: PLC0415
        _runner = Runner(agent=root_agent)
        logging.info("[SERVER] ADK Runner initialised.")
    return _runner


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class AgentHandler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send(404, {"error": f"Unknown path: {self.path}"})
            return

        # Read body
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            user_message: str = body.get("message", "").strip()
        except Exception as parse_err:
            logging.warning("[SERVER] Bad request: %s", parse_err)
            self._send(400, {"error": f"Bad request: {parse_err}"})
            return

        if not user_message:
            self._send(400, {"error": "Empty message."})
            return

        logging.info("[SERVER] Received message: %.200s", user_message)

        # Call agent
        response_text = ""
        metadata: dict[str, Any] = {}
        try:
            runner = _get_runner()

            # The ADK Runner API varies by version. We try run_turn() first
            # (returns (text, metadata)) then fall back to run() which may
            # only return text.
            if hasattr(runner, "run_turn"):
                result = runner.run_turn(user_message)
                if isinstance(result, tuple) and len(result) == 2:
                    response_text, raw_meta = result
                    if isinstance(raw_meta, dict):
                        metadata.update(raw_meta)
                else:
                    response_text = str(result)
            else:
                # Older ADK — run() blocks until the agent produces output
                response_text = runner.run(user_message)

            logging.info("[SERVER] Agent replied (first 200 chars): %.200s", response_text)

        except Exception as agent_err:
            logging.error("[SERVER] Agent error: %s", agent_err, exc_info=True)
            response_text = f"Agent error: {agent_err}"
            metadata["error"] = str(agent_err)

        # Append last 20 log lines for the dashboard dev panel
        metadata["logs"] = _tail_log(20)

        self._send(200, {"response": response_text, "metadata": metadata})

    # -------------------------------------------------------------------------

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Route BaseHTTPServer access logs through Python logging
        logging.debug("[HTTP] " + fmt, *args)


# ---------------------------------------------------------------------------
# Log tail helper
# ---------------------------------------------------------------------------

def _tail_log(n: int = 20) -> str:
    log_path = Path("agent_session.log")
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:])
    except FileNotFoundError:
        return "(no log file yet)"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_agent_server(port: int = 8502) -> None:
    """
    Start the HTTP server and serve forever (blocking call).
    Intended to be called from main.py as the last step in the startup sequence.
    """
    server = HTTPServer(("localhost", port), AgentHandler)
    logging.info("[SERVER] Agent HTTP server listening on http://localhost:%d/chat", port)
    server.serve_forever()
