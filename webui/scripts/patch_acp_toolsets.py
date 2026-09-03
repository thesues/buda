#!/usr/bin/env python3
"""Let $HERMES_ACP_TOOLSETS choose an ACP session's toolsets, instead of a hardcoded list.

Every ACP session is built with the full `hermes-acp` toolset, which explicitly
includes ten `browser_*` tools and `delegate_task`. A chat box with a retrieval
server has no use for any of them: the browser tools cannot run (no Chromium in
this image) and `delegate_task` spawns subagents this UI has nowhere to show.
Worse, their mere presence costs tokens in every request's tool list and invites
the model to reach for a browser when it should be searching the corpus -- which
it did, before the MCP server was reachable.

There is no supported way to change this from the client. `acp_adapter/session.py`
calls:

    "enabled_toolsets": _expand_acp_enabled_toolsets(
        ["hermes-acp"],                      # <- hardcoded
        mcp_server_names=configured_mcp_servers,
    ),

and neither `agent.disabled_toolsets` nor `agent.enabled_toolsets` in config.yaml
reaches it (both were tried against 0.17; the browser tools stayed). The
`disabled_toolsets` path exists but the ACP adapter overrides `enabled_toolsets`
only, and an allowlist wins.

So: patch the hardcoded literal into an env lookup. `HERMES_ACP_TOOLSETS` is a
comma-separated list of toolset names; unset keeps `hermes-acp`, so an unpatched
and an unconfigured image behave identically.

MCP servers are unaffected either way -- `_expand_acp_enabled_toolsets` appends
`mcp-<name>` for each configured server after the base list, so narrowing the
base cannot drop retrieval.

Idempotent. A no-op (exit 0, loudly) if hermes changed the call, because a
failed patch must not fail the build -- the image still works, with the tools it
always had.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = "acp_adapter/session.py"

OLD = '''            "enabled_toolsets": _expand_acp_enabled_toolsets(
                ["hermes-acp"],
                mcp_server_names=configured_mcp_servers,
            ),'''

NEW = '''            "enabled_toolsets": _expand_acp_enabled_toolsets(
                # PATCHED (patch_acp_toolsets.py): was the literal
                # ["hermes-acp"]. See that script for why this cannot be done
                # from config. Unset env => unchanged behaviour.
                [
                    t.strip()
                    for t in __import__("os").environ.get(
                        "HERMES_ACP_TOOLSETS", "hermes-acp"
                    ).split(",")
                    if t.strip()
                ]
                or ["hermes-acp"],
                mcp_server_names=configured_mcp_servers,
            ),'''

MARKER = "PATCHED (patch_acp_toolsets.py)"


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        try:
            import acp_adapter
        except ImportError:
            print("patch_acp_toolsets: acp_adapter not importable — skipping", file=sys.stderr)
            return 0
        root = Path(acp_adapter.__file__).resolve().parent.parent

    path = root / TARGET
    if not path.is_file():
        print(f"patch_acp_toolsets: {path} not found — skipping", file=sys.stderr)
        return 0

    src = path.read_text()
    if MARKER in src:
        print("patch_acp_toolsets: already applied")
        return 0
    if OLD not in src:
        # The call moved or was reformatted. Say so — silence here would look
        # like a working patch and leave the toolset unchanged.
        print(
            "patch_acp_toolsets: the hardcoded ['hermes-acp'] call was not found "
            "verbatim; hermes changed it. NOT patching, tools unchanged.",
            file=sys.stderr,
        )
        return 0

    tmp = path.with_suffix(".py.tmp")
    tmp.write_text(src.replace(OLD, NEW, 1))
    tmp.replace(path)
    print(f"patch_acp_toolsets: patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
