"""
agent/server.py
Lightweight HTTP server that exposes the ADK agent as a REST endpoint so
the Streamlit dashboard (running in a separate process) can call it via
requests.post().

Endpoint:
    POST http://localhost:8502/chat
    Body:  { "message": "<user prompt>" }
    Returns: { "response": "<agent reply>", "metadata": { ... } }

Compatible with google-adk 0.4.x which requires Runner to receive an explicit
session_service and run() to receive a genai_types.Content message.

Called from main.py via start_agent_server().
"""

from __future__ import annotations

import asyncio
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agent.agent import root_agent, session_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME   = "pricing_intelligence_agent"
USER_ID    = "default_user"
SESSION_ID = "default_session"

# ---------------------------------------------------------------------------
# Runner + session bootstrap
# ---------------------------------------------------------------------------

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# Pre-create the session so the first message doesn't race against session init
asyncio.get_event_loop().run_until_complete(
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
)
logging.info("[SERVER] ADK Runner initialised (session: %s).", SESSION_ID)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class AgentHandler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send(404, {"error": f"Unknown path: {self.path}"})
            return

        # Parse body
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

        response_text = ""
        metadata: dict[str, Any] = {}

        try:
            # Build a Content object as required by ADK 0.4.x
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)],
            )

            # runner.run() is a synchronous generator of events
            for event in runner.run(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=content,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        response_text = event.content.parts[0].text
                    break

            logging.info(
                "[SERVER] Agent replied (first 200 chars): %.200s", response_text
            )

        except Exception as agent_err:
            logging.error("[SERVER] Agent error: %s", agent_err, exc_info=True)
            response_text = f"Agent error: {agent_err}"
            metadata["error"] = str(agent_err)

        # Attach last 20 log lines for the dashboard dev panel
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
    logging.info(
        "[SERVER] Agent HTTP server listening on http://localhost:%d/chat", port
    )
    server.serve_forever()
