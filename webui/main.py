"""Entry point — assembles the six pieces and serves.

Run with HERMES' interpreter, not ours. Importing `run_agent` is the whole
design, and it only resolves inside that venv. Everything this server adds is
stdlib, so nothing of ours competes with hermes' dependency tree:

    /opt/hermes/.venv/bin/python main.py

`BUDA_ENDPOINTS` is a JSON list, first entry the default:

    [{"key":"dsv4","label":"DSV4","model":"dsv4-flash",
      "base_url":"http://freetoken-l3:1919/v1","maxConcurrent":4}, ...]

Unset, the single endpoint the LLM_* variables already describe is used, which
is the previous behaviour exactly.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app_routes import build_app  # noqa: E402
from hermes_agent import AgentPool, load_endpoints  # noqa: E402
from http_shell import serve  # noqa: E402
from turns import TurnManager  # noqa: E402

log = logging.getLogger("buda")


def _sessions_module():
    """hermes' own session store, in process.

    The subprocess bridge this replaces existed because the two venvs disagreed
    on some packages — a reason that disappears once this server runs inside
    hermes' interpreter. Import failure is not fatal: the chat works without a
    sidebar, and refusing to start over it would be the wrong trade.
    """
    try:
        import hermes_session_api

        return hermes_session_api
    except Exception:  # noqa: BLE001
        log.warning("session store unavailable — the sidebar will be empty", exc_info=True)
        return None


def _reaper(manager: TurnManager, stop: threading.Event) -> None:
    """Drop finished streams so a long-lived server does not accumulate them.

    A finished stream is kept for a while so a reader who was away can still
    collect its tail; it is not kept forever.
    """
    while not stop.wait(60):
        try:
            manager.forget_finished(keep=200)
        except Exception:  # noqa: BLE001
            log.exception("stream reaper failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = ap.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    endpoints = load_endpoints(
        os.environ.get("BUDA_ENDPOINTS"),
        default_home_model=os.environ.get("LLM_MODEL", ""),
    )
    log.info(
        "endpoints: %s",
        ", ".join(f"{e.key}={e.model}@{e.base_url} (<={e.max_concurrent})" for e in endpoints),
    )

    # Point hermes at the retrieval server before any agent is built: the MCP
    # list is read when an agent is constructed, so writing it afterwards would
    # leave the first conversation of every restart without retrieval. Loud but
    # not fatal — chat without retrieval beats no chat, and `/api/status`
    # reports what was configured either way.
    mcp_url = os.environ.get("MEMORY_MCP_URL", "")
    if mcp_url:
        try:
            from hermes_config import ensure_mcp_server

            cfg = Path(
                os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
            ) / "config.yaml"
            ensure_mcp_server(cfg, os.environ.get("MEMORY_MCP_NAME", "memory"), mcp_url)
        except Exception as e:  # noqa: BLE001
            log.error("could not point hermes at %s: %s", mcp_url, e)

    manager = TurnManager(AgentPool())
    app = build_app(
        manager=manager,
        endpoints=endpoints,
        static_dir=HERE / "static",
        index_html=HERE / "static" / "index.html",
        auth_user=os.environ.get("AUTH_USER", ""),
        auth_pass=os.environ.get("AUTH_PASS", ""),
        sessions=_sessions_module(),
        mcp=(
            {"name": os.environ.get("MEMORY_MCP_NAME", "memory"),
             "url": os.environ["MEMORY_MCP_URL"]}
            if os.environ.get("MEMORY_MCP_URL")
            else None
        ),
    )

    srv = serve(app, args.host, args.port)
    log.info("buda webui on http://%s:%d", args.host, args.port)

    stop = threading.Event()
    threading.Thread(target=_reaper, args=(manager, stop), name="reaper", daemon=True).start()

    # A container gets SIGTERM on rollout. Stop accepting, then exit — a turn in
    # flight is lost either way, and hanging on to the port makes the next pod
    # fail to bind.
    def _bye(*_a):
        log.info("shutting down")
        stop.set()
        srv.shutdown()

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)
    stop.wait()


if __name__ == "__main__":
    main()
