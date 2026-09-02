"""Point hermes at the MCP server, through its config file.

Why the config file and not ACP `session/new {mcpServers: [...]}`: the config
shape `mcp_servers.<name>.url` is the one hermes' own tests exercise for an
HTTP-transport server, and it applies to EVERY session including ones loaded
from history. The ACP parameter's HTTP form was not verified against this
hermes build, and guessing a wire shape that silently parses to "no servers"
would look exactly like a working deploy with an agent that has no tools —
the failure mode this whole decoupling exists to avoid.

Written without pyyaml on purpose: it is not guaranteed present in the runtime,
and a missing import here would degrade into "chat works, retrieval silently
does not". hermes writes a plain block-style mapping, so an indentation-aware
edit is enough and dependency-free. The same reasoning the console applied to
reading the `model:` block.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("webui.config")


def render_block(name: str, url: str) -> list[str]:
    return [
        "mcp_servers:",
        f"  {name}:",
        f"    url: {url}",
        "    enabled: true",
    ]


def ensure_mcp_server(config_path: Path, name: str, url: str) -> bool:
    """Make `mcp_servers.<name>.url` say `url`. Returns True if the file changed.

    Idempotent: an unchanged desired state rewrites nothing, so a restart loop
    does not churn the file hermes may be reading.
    """
    lines = config_path.read_text().splitlines() if config_path.exists() else []

    start = next((i for i, ln in enumerate(lines) if ln.rstrip() == "mcp_servers:"), None)
    if start is None:
        new = [*lines, *([""] if lines and lines[-1].strip() else []), *render_block(name, url)]
        _write(config_path, new)
        log.info("hermes config: added mcp_servers.%s -> %s", name, url)
        return True

    # The block runs to the next line at column 0 that is not blank — the same
    # indentation rule hermes' own writer produces.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace():
            end = i
            break
    block = lines[start:end]

    want = f"    url: {url}"
    entry = next((i for i, ln in enumerate(block) if ln.strip() == f"{name}:"), None)
    if entry is None:
        block = [*block, f"  {name}:", want, "    enabled: true"]
    else:
        # Replace this server's url line; leave every sibling key alone so an
        # operator's headers/timeout settings survive.
        stop = len(block)
        for i in range(entry + 1, len(block)):
            if block[i].strip() and not block[i].startswith("    "):
                stop = i
                break
        sub = block[entry + 1 : stop]
        url_at = next((i for i, ln in enumerate(sub) if ln.strip().startswith("url:")), None)
        if url_at is None:
            sub = [want, *sub]
        elif sub[url_at] == want:
            return False  # already correct — do not touch the file
        else:
            sub[url_at] = want
        block = [*block[: entry + 1], *sub, *block[stop:]]

    _write(config_path, [*lines[:start], *block, *lines[end:]])
    log.info("hermes config: set mcp_servers.%s -> %s", name, url)
    return True


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: hermes may read this file at any moment, and a
    # half-written config parses as a config with no MCP servers.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n")
    tmp.replace(path)
