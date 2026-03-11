"""
agent/server.py
Lightweight HTTP server that exposes the ADK agent as a REST endpoint.

Endpoint:
    POST http://localhost:8502/chat
    Body:    { "message": "<user prompt>" }
    Returns: { "response": "<agent reply>", "metadata": { ... } }

Health check:
    GET http://localhost:8502/health  →  { "status": "ok" }

Uses runner.run_async() (ADK 0.4.x) inside a fresh asyncio event loop
per request so the synchronous HTTP handler can await it safely.

Called from main.py via start_agent_server().
"""

import asyncio
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agent.agent import root_agent
from config import GCP_PROJECT_ID  # noqa: F401 — imported for env validation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME   = "pricing_intelligence_agent"
USER_ID    = "analyst"
SESSION_ID = "session_001"

# ---------------------------------------------------------------------------
# Session + Runner setup (module-level, shared across requests)
# ---------------------------------------------------------------------------

session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


async def _init_session() -> None:
    """Pre-create the ADK session so the first message doesn't race init."""
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )


asyncio.get_event_loop().run_until_complete(_init_session())
logging.info(
    "[SERVER] Session initialised — app=%s  user=%s  session=%s",
    APP_NAME, USER_ID, SESSION_ID,
)


# ---------------------------------------------------------------------------
# Async agent call
# ---------------------------------------------------------------------------

async def call_agent_async(user_message: str) -> tuple[str, dict]:
    """
    Send *user_message* to the ADK runner and collect the final text response.

    Returns
    -------
    (response_text, metadata_dict)
    """
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )
    metadata: dict = {"tool_called": None, "error": None}
    response_text = ""

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content,
        ):
            logging.debug("[SERVER] ADK event: %s", event)

            # Capture tool calls for dev-mode metadata
            if hasattr(event, "tool_call") and event.tool_call:
                metadata["tool_called"] = getattr(event.tool_call, "name", None)
                logging.info("[SERVER] Tool called: %s", metadata["tool_called"])

            # Collect final text response
            if event.is_final_response():
                if event.content and event.content.parts:
                    response_text = "".join(
                        p.text
                        for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )
                break

        if not response_text:
            response_text = (
                "I processed your request but have no text response to show."
            )

    except Exception as exc:
        logging.error("[SERVER] Agent run failed: %s", exc, exc_info=True)
        response_text = f"Agent error: {exc}"
        metadata["error"] = str(exc)

    return response_text, metadata


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class AgentHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        """Health-check — returns 200 for GET / or GET /health."""
        if self.path in ("/", "/health"):
            self._respond(200, {"status": "ok", "agent": APP_NAME})
        elif self.path == "/chat":
            self._respond(405, {
                "error": "Method Not Allowed",
                "message": "/chat requires POST, not GET.",
                "usage": 'POST /chat  Body: {"message": "<question>"}',
            })
        else:
            self._respond(404, {"error": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path == "/chat":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                user_message: str = body.get("message", "").strip()

                if not user_message:
                    self._respond(400, {"error": "Empty message"})
                    return

                logging.info("[SERVER] /chat received: %.200s", user_message)

                # Run the async agent call in a fresh event loop.
                # BaseHTTPRequestHandler.do_POST() is synchronous, so we
                # cannot use an existing loop — create a new one per request.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response_text, metadata = loop.run_until_complete(
                        call_agent_async(user_message)
                    )
                finally:
                    loop.close()

                self._respond(200, {"response": response_text, "metadata": metadata})

            except json.JSONDecodeError as exc:
                logging.error("[SERVER] JSON decode error: %s", exc)
                self._respond(400, {"error": f"Invalid JSON: {exc}"})
            except Exception as exc:
                logging.error("[SERVER] Handler error: %s", exc, exc_info=True)
                self._respond(500, {"error": str(exc)})

        elif self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "Not found"})

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # suppress default access log
        logging.debug("[HTTP] " + fmt, *args)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_agent_server(port: int = 8502) -> None:
    """Start the HTTP server and serve forever (blocking)."""
    server = HTTPServer(("localhost", port), AgentHandler)
    logging.info(
        "[SERVER] Agent HTTP server listening on http://localhost:%d", port
    )
    logging.info("[SERVER] Health check: http://localhost:%d/health", port)
    server.serve_forever()
