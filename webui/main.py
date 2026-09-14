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
    hermes_cfg = Path(
        os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    ) / "config.yaml"
    mcp_url = os.environ.get("MEMORY_MCP_URL", "")
    if mcp_url:
        try:
            from hermes_config import ensure_mcp_server

            ensure_mcp_server(
                hermes_cfg, os.environ.get("MEMORY_MCP_NAME", "memory"), mcp_url
            )
        except Exception as e:  # noqa: BLE001
            log.error("could not point hermes at %s: %s", mcp_url, e)

    # The toolsets live in the SAME config file, under `platform_toolsets.cli`
    # — the key hermes' own CLI reads. Seed it only if the file does not have
    # the key yet: HERMES_ACP_TOOLSETS (what the manifest used to drive
    # directly) becomes a FIRST-BOOT seed instead of a per-build lookup, and
    # from then on the file is authoritative — edit it, or run `hermes tools`.
    # The built-in default includes terminal, so an empty env and an empty file
    # still yield an agent that can run things.
    try:
        from hermes_config import ensure_platform_toolsets

        seed = [
            t.strip()
            for t in os.environ.get("HERMES_ACP_TOOLSETS", "").split(",")
            if t.strip()
        ]
        ensure_platform_toolsets(hermes_cfg, seed or None)
    except Exception as e:  # noqa: BLE001
        log.error("could not seed platform_toolsets: %s", e)

    # Context compression summarises an overflowing conversation through an
    # auxiliary model, which defaults to the ACTIVE endpoint's model — and
    # hermes refuses the session outright when that model's window is under
    # its 32K floor (the MiniMax endpoint's real, VRAM-derived ceiling).
    # Point the slot at the first endpoint that declares enough window;
    # seed-only, because a compression: mapping in the file is the operator's
    # choice, exactly like platform_toolsets.
    try:
        from hermes_agent import load_endpoints
        from hermes_config import ensure_compression_model

        ensure_compression_model(
            hermes_cfg, load_endpoints(os.environ.get("BUDA_ENDPOINTS", ""))
        )
    except Exception as e:  # noqa: BLE001
        log.error("could not seed auxiliary.compression: %s", e)

    # Say what actually resolved — the file won this boot or the seed did, and
    # the log is where the difference is visible without opening a shell.
    try:
        from hermes_cli.config import load_config
        from hermes_config import resolve_toolsets

        log.info("toolsets: %s", ", ".join(resolve_toolsets(load_config() or {})))
    except Exception:  # noqa: BLE001
        log.debug("could not log the resolved toolsets", exc_info=True)

    # Writing the server into config.yaml is not connecting to it. The ACP
    # adapter called `register_mcp_servers` itself; nothing in the library path
    # did, so `mcp_servers` sat in the config, `mcp-memory` sat in
    # enabled_toolsets, and the agent was handed eleven tools none of which
    # could reach the corpus. Asked about a sutra it ran `search_files` over a
    # filesystem that has no corpus on it and answered that the corpus "is not
    # available in this environment" — which was true, and entirely our doing.
    #
    # Once per process, before any agent is built: the registry is global and
    # `refresh_agent_mcp_tools` picks the tools up per turn from there.
    try:
        from hermes_cli.config import load_config
        from tools.mcp_tool import register_mcp_servers

        servers = (load_config() or {}).get("mcp_servers") or {}
        if servers:
            added = register_mcp_servers(servers)
            log.info("registered %d MCP tool(s) from %s: %s",
                     len(added), list(servers), ", ".join(added[:4]) + ("…" if len(added) > 4 else ""))
        else:
            log.warning("no mcp_servers in hermes config; corpus search will be unavailable")
    except Exception:  # noqa: BLE001 -- a chat box without retrieval still starts
        log.exception("could not register MCP servers; corpus search will be unavailable")

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
