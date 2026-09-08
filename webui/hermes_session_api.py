"""JSON bridge to hermes' own session store — runs in HERMES' interpreter, not ours.

The webui and hermes live in separate venvs deliberately: they share most of
their packages and disagree on some, so importing `hermes_state` in-process
would shadow one venv's deps with the other's. Hence a subprocess:
`<hermes venv>/bin/python hermes_session_api.py`.

Why not parse `hermes sessions list`? It renders a fixed-width table — titles
truncated, times as "3d ago", CJK breaking the column alignment. `SessionDB` is
what that table is rendered FROM, so call it directly and emit JSON.

Read-only. Writes (delete/rename) stay on the `hermes sessions ...` CLI, which
is schema-aware (it also cleans the FTS index and related tables).
"""

import argparse
import ast
import json
import re
import sys

from hermes_state import SessionDB  # hermes venv only


def _provisional_title(preview: str) -> str:
    """A short name from the opening line. Empty stays empty.

    The preview has already had its newlines flattened to spaces, so the break
    between what the reader actually asked and the prompt template that follows
    survives only as a RUN of spaces. Splitting on that recovers the question —
    "六道", not "六道  你是佛教典籍的检索助手，工作是…".
    """
    head = re.split(r"\s{2,}", preview.strip(), maxsplit=1)[0].strip()
    head = head.split("\n", 1)[0].strip()
    if len(head) > 24:
        head = head[:24].rstrip() + "\u2026"
    return head


def list_sessions(limit: int, include_empty: bool) -> list[dict]:
    # exclude_sources=["tool"] mirrors the CLI's default: hide third-party tool
    # sessions, which are not conversations anyone opened.
    rows = SessionDB().list_sessions_rich(source=None, exclude_sources=["tool"], limit=limit)
    out = []
    for r in rows:
        # A session with no messages is a ghost. They carry no title or preview
        # and cannot be usefully loaded, so drop them unless asked for.
        if not include_empty and not r.get("message_count"):
            continue
        # hermes titles a session asynchronously, so a conversation that is
        # minutes old and fifteen messages long can still have none — and the
        # sidebar was calling those "(未命名)" while holding the opening line in
        # `preview`. Stand in with the first thing the reader said, which is
        # what they would call it themselves, until the real title lands.
        title = (r.get("title") or "").strip()
        if not title:
            title = _provisional_title(r.get("preview") or "")
        out.append({
            "id": r.get("id"),
            "title": title,
            "titleProvisional": not (r.get("title") or "").strip(),
            "preview": r.get("preview") or "",
            "lastActive": r.get("last_active") or r.get("started_at") or 0,
            "messageCount": r.get("message_count") or 0,
        })
    return out


def _tool_calls(raw) -> list[dict]:
    """`tool_calls` comes back as a list, or as its repr. Accept both."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    for parse in (json.loads, ast.literal_eval):
        try:
            v = parse(raw)
            return v if isinstance(v, list) else []
        except Exception:  # noqa: BLE001
            continue
    return []


def history(sid: str, limit: int) -> list[dict]:
    """One session's transcript, in the UI's own event shape.

    The point of this command is that it does NOT go through ACP. `session/load`
    is what MOVES the single agent process to a session, and it was the only way
    to read a transcript — so looking at another conversation while one was
    streaming meant yanking the agent out from under the turn in flight, which
    is why the UI had to refuse. Reading from the store instead decouples the
    two: viewing is free, and the agent is moved only when something is sent.

    The mapping mirrors `_to_event` in the webui server, which is what the
    client already renders. Kept lossy in the same way and for the same reason:
    this is a chat box, so assistant text and a one-line trace of tool activity
    is all of it.
    """
    msgs = SessionDB().get_messages(sid) or []
    if limit and len(msgs) > limit:
        msgs = msgs[-limit:]
    out: list[dict] = []
    for m in msgs:
        role = m.get("role")
        text = m.get("content") or ""
        if role == "user":
            if text:
                out.append({"kind": "history_user", "text": text})
        elif role == "assistant":
            if text:
                out.append({"kind": "delta", "text": text, "thought": False})
            for tc in _tool_calls(m.get("tool_calls")):
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                out.append({
                    "kind": "tool",
                    "id": tc.get("id") or tc.get("call_id"),
                    "title": fn.get("name") or tc.get("name") or "tool",
                    "status": "pending", "detail": "", "detailFull": "",
                })
        elif role == "tool":
            # The result arrives as its own row and carries the id the call was
            # announced under, so the client merges it into that same line.
            out.append({
                "kind": "tool",
                "id": m.get("tool_call_id"),
                "title": m.get("tool_name") or "",
                "status": "completed",
                "detail": text[:200], "detailFull": text[:4000],
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="list sessions as JSON")
    p_list.add_argument("--limit", type=int, default=200)
    p_list.add_argument("--include-empty", action="store_true")
    p_hist = sub.add_parser("history", help="one session's transcript as JSON")
    p_hist.add_argument("--id", required=True)
    p_hist.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    if args.cmd == "list":
        json.dump(list_sessions(args.limit, args.include_empty), sys.stdout, ensure_ascii=False)
        return 0
    if args.cmd == "history":
        json.dump(history(args.id, args.limit), sys.stdout, ensure_ascii=False)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
