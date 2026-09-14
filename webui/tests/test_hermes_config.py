"""`hermes_config`: the config file owns what a session can do and with what.

`ensure_mcp_server` has been covered indirectly by the deploy for a while;
these tests pin the seeders it grew afterwards: the toolset list, and the
auxiliary compression model (hermes refuses a session whose compression
model cannot hold its 32K floor, and the slot defaults to the ACTIVE
endpoint — so a small-window endpoint must be served by a declared bigger
one). In both cases the seed must not clobber, the resolution must degrade
in order, and terminal must be in the default — an agent that cannot run
anything is not a degraded agent, it is a broken one.
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


# ── ensure_compression_model ────────────────────────────────────────────────


class _EP:
    """Endpoint stand-in: only the fields the compression seeding reads."""

    def __init__(self, context, model="m", base_url="http://b/v1"):
        self.context = context
        self.model = model
        self.base_url = base_url


def test_compression_seeds_from_the_first_qualifying_endpoint(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("mcp_servers:\n  memory:\n    url: http://mcp\n")
    eps = [_EP(4096, model="small", base_url="http://small/v1"),
           _EP(62080, model="big", base_url="http://big/v1")]

    assert hc.ensure_compression_model(p, eps, min_context=32000) is True
    text = p.read_text()
    assert "  compression:" in text
    assert "    model: big" in text and "    base_url: http://big/v1" in text
    assert "    context_length: 62080" in text
    assert "url: http://mcp" in text, "the block this touched is the only one that moved"


def test_compression_skips_when_no_endpoint_qualifies(tmp_path):
    p = tmp_path / "config.yaml"

    assert hc.ensure_compression_model(p, [_EP(4096)], min_context=32000) is False
    assert not p.exists(), "nothing to seed means nothing to write"


def test_compression_never_clobbers_an_operators_choice(tmp_path):
    """Seed, not set — same contract as platform_toolsets. A model-only
    mapping is a deliberate override even if it omits base_url."""
    p = tmp_path / "config.yaml"
    p.write_text("auxiliary:\n  compression:\n    model: mine\n")

    assert hc.ensure_compression_model(p, [_EP(62080)], min_context=32000) is False
    assert "model: mine" in p.read_text()


def test_compression_fills_under_an_existing_auxiliary_key(tmp_path):
    """Other auxiliary tasks (vision, web_extract) must survive."""
    p = tmp_path / "config.yaml"
    p.write_text("auxiliary:\n  vision:\n    provider: auto\n")

    assert hc.ensure_compression_model(
        p, [_EP(62080, model="big", base_url="http://big/v1")], min_context=32000
    ) is True
    text = p.read_text()
    assert text.count("auxiliary:") == 1
    assert "  vision:" in text and "    provider: auto" in text
    assert "  compression:" in text and "    model: big" in text


def test_compression_fills_an_empty_compression_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("auxiliary:\n  compression:\n")

    assert hc.ensure_compression_model(
        p, [_EP(62080, model="big", base_url="http://big/v1")], min_context=32000
    ) is True
    text = p.read_text()
    assert text.count("compression:") == 1
    assert "    model: big" in text


def test_compression_fills_the_wizards_empty_template_in_place(tmp_path):
    """hermes' setup wizard writes provider: auto and empty-string values —
    an absence wearing a mapping. The seed must fill THAT (replacing the
    empty model/base_url lines, no duplicate keys) instead of honouring it,
    or the slot keeps falling back to the active endpoint's model."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "auxiliary:\n"
        "  compression:\n"
        "    provider: auto\n"
        "    model: ''\n"
        "    base_url: ''\n"
        "    api_key: ''\n"
        "    timeout: 120\n"
        "    extra_body: {}\n"
    )

    assert hc.ensure_compression_model(
        p, [_EP(62080, model="big", base_url="http://big/v1")], min_context=32000
    ) is True
    text = p.read_text()
    assert text.count("model:") == 1 and "    model: big" in text
    assert text.count("base_url:") == 1 and "    base_url: http://big/v1" in text
    assert "    context_length: 62080" in text
    # provider must be REWRITTEN, not kept: with `auto` the resolver ignores
    # the seeded base_url entirely and falls back to the main runtime.
    assert text.count("provider:") == 1 and "    provider: custom" in text
    assert "    api_key: ''" in text, "sibling keys stay"
    assert "    timeout: 120" in text and "    extra_body: {}" in text, "template plumbing stays"
    # and the template's non-empty-but-generic timeout must NOT be mistaken
    # for an operator choice that blocks the seed
    assert "    context_length: 62080" in text


def test_compression_seeding_is_idempotent(tmp_path):
    p = tmp_path / "config.yaml"
    eps = [_EP(62080, model="big", base_url="http://big/v1")]

    assert hc.ensure_compression_model(p, eps, min_context=32000) is True
    first = p.read_text()
    assert hc.ensure_compression_model(p, eps, min_context=32000) is False
    assert p.read_text() == first


def test_compression_floor_defaults_to_hermes_when_importable(tmp_path, monkeypatch):
    """Without an explicit floor the number must come from hermes itself, not
    a second copy that can drift."""
    fake = types.ModuleType("agent.model_metadata")
    fake.MINIMUM_CONTEXT_LENGTH = 40000
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.model_metadata", fake)

    p = tmp_path / "config.yaml"
    assert hc.ensure_compression_model(p, [_EP(32000)]) is False  # below 40K
    assert hc.ensure_compression_model(p, [_EP(40000)]) is True


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
