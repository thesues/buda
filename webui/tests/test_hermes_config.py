"""`hermes_config`'s toolset half: the config file owns what a session can do.

`ensure_mcp_server` has been covered indirectly by the deploy for a while;
these tests pin the toolset functions it grew when the env-var lookup was
retired: the seed must not clobber, the resolution must degrade in order,
and terminal must be in the default — an agent that cannot run anything is
not a degraded agent, it is a broken one.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_config as hc  # noqa: E402


# ── ensure_platform_toolsets ────────────────────────────────────────────────


def test_a_config_without_the_key_gets_seeded(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("model:\n  provider: custom\n  base_url: http://a/v1\n")

    assert hc.ensure_platform_toolsets(p, ["file", "terminal"]) is True
    text = p.read_text()
    assert "platform_toolsets:" in text and "  cli:" in text
    assert "    - terminal" in text
    # The block this touched is the only one that moved.
    assert "provider: custom" in text and "base_url: http://a/v1" in text


def test_an_existing_list_is_never_clobbered(tmp_path):
    """Seed, not set. An operator (or `hermes tools`) wrote a choice; a pod
    restart must not silently overwrite it back to the default."""
    p = tmp_path / "config.yaml"
    p.write_text("platform_toolsets:\n  cli:\n    - file\n")

    assert hc.ensure_platform_toolsets(p, ["everything"]) is False
    assert "- everything" not in p.read_text()
    assert "- file" in p.read_text()


def test_seeding_is_idempotent(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("mcp_servers:\n  memory:\n    url: http://mcp\n")

    assert hc.ensure_platform_toolsets(p, ["file", "terminal"]) is True
    first = p.read_text()
    assert hc.ensure_platform_toolsets(p, ["file", "terminal"]) is False
    assert p.read_text() == first


def test_an_inline_cli_value_counts_as_configured(tmp_path):
    """`cli: [file, terminal]` — flow style, which hermes also writes — is a
    list the file owns, not an absence to seed over."""
    p = tmp_path / "config.yaml"
    p.write_text("platform_toolsets:\n  cli: [file, terminal]\n")

    assert hc.ensure_platform_toolsets(p, None) is False
    assert p.read_text().count("cli:") == 1


def test_an_empty_toolsets_argument_seeds_the_default(tmp_path):
    p = tmp_path / "config.yaml"
    assert hc.ensure_platform_toolsets(p, None) is True
    text = p.read_text()
    assert "    - terminal" in text, "the default must keep terminal"


def test_a_dangling_platform_toolsets_key_still_gets_cli(tmp_path):
    """`platform_toolsets:` present but empty — a hand edit, not a hermes
    write. The seed belongs UNDER the existing key, not as a second block."""
    p = tmp_path / "config.yaml"
    p.write_text("platform_toolsets:\n")

    assert hc.ensure_platform_toolsets(p, ["file"]) is True
    text = p.read_text()
    assert text.count("platform_toolsets:") == 1
    assert "  cli:" in text and "    - file" in text


# ── resolve_toolsets ────────────────────────────────────────────────────────


def test_resolution_prefers_hermes_own_resolver(monkeypatch):
    """The same function the CLI uses on itself — including its `mcp-<name>`
    appending. If buda re-derived that by hand, the two ends would drift."""
    fake = types.ModuleType("hermes_cli.tools_config")
    fake._get_platform_tools = lambda cfg, platform, **kw: {"file", "terminal", "mcp-memory"}
    monkeypatch.setitem(sys.modules, "hermes_cli", types.ModuleType("hermes_cli"))
    monkeypatch.setitem(sys.modules, "hermes_cli.tools_config", fake)

    assert hc.resolve_toolsets({"mcp_servers": {"memory": {}}}) == [
        "file", "mcp-memory", "terminal",
    ]


def test_resolution_falls_back_to_the_raw_config_list(monkeypatch):
    """hermes moved its resolver — an explicit list in the file is still worth
    more than a built-in guess."""
    monkeypatch.setitem(sys.modules, "hermes_cli", None)  # import fails
    cfg = {"platform_toolsets": {"cli": ["file", " terminal ", ""]}}
    assert hc.resolve_toolsets(cfg) == ["file", "terminal"]


def test_resolution_ends_at_the_default_which_keeps_terminal(monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli", None)
    got = hc.resolve_toolsets({})
    assert "terminal" in got and "file" in got
    assert got == hc.DEFAULT_TOOLSETS


def test_resolution_never_returns_an_empty_list(monkeypatch):
    """An empty `cli: []` in the file is 'misconfigured', not 'no tools': the
    honest degradation is the default, because an agent with no toolsets
    cannot even read the corpus it exists to search."""
    monkeypatch.setitem(sys.modules, "hermes_cli", None)
    cfg = {"platform_toolsets": {"cli": []}}
    assert hc.resolve_toolsets(cfg) == hc.DEFAULT_TOOLSETS
