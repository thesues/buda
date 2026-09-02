# hermes webui — architecture and known issues

Session management and a chat box. Nothing else, deliberately.

## The shape

```
POST /api/chat/start   -> make a TurnStream, spawn a task, return {stream_id}
GET  /api/chat/stream  -> SSE; replays from ?after_seq, then follows
GET  /api/chat/status  -> is that stream running, and at what seq
```

The turn runs in its own asyncio task writing into a sequence-numbered buffer.
**It is not bound to the connection that started it.** A browser may close,
reload, or reconnect from another tab; the turn does not notice, and the
reconnect replays only the delta.

```
 browser ──POST /chat/start──▶ TurnStream(seq=0)  ──▶ task: acp.prompt()
    │                              │  emit(seq++)        │
    │◀──SSE  after_seq=N───────────┘                     │ writes regardless of
    │  (reconnectable, replays >N)                       │ who is listening
    │                                                     ▼
    └──POST /approval/answer──▶ future ◀──parked──── on_permission
```

Three parts, and each exists because of a specific failure:

- **Sequence numbers + a bounded backlog.** Lifted from the lerobot console's
  *terminal* output buffer and applied to chat. A reconnect asks for what it
  missed instead of the whole transcript, and an overflowed window is REPORTED
  (`gap`) rather than delivered with a silent hole.
- **Approvals answered out of band.** The ACP permission callback parks on a
  future; the answer arrives on a different request. `GET /api/approval/pending`
  exists because a push can be lost and a reloaded page never saw it.
- **A shallow `/healthz`.** It answers while a turn runs and while `hermes acp`
  restarts. A deep probe would restart the pod in the moments this design is for.

## What was dropped from `lerobot-agent-console`, and why that is safe

The PTY terminal, the port proxy, service discovery, and the lerobot/volcano
endpoints. Most of that console's hardest-won fixes are terminal fixes —
process-group reaping, idle reclamation, output backlogs, the zombie-per-session
leak. **None can regress here, because there is no terminal.**

## Dependencies

| Depends on | How | Consequence |
|---|---|---|
| `hermes` | child process, ACP over stdio | one warm process; a cold `hermes chat` per turn costs ~10 s |
| `memory-mcp` | **HTTP MCP** (`MEMORY_MCP_URL`) | no spawned process, no autumn credential, **not under autumn's WIRE lockstep** |
| `freetoken` | OpenAI-compatible HTTP | set at startup via `hermes config set`, best-effort |

The MCP transport is the load-bearing choice. A stdio MCP server must be spawned
by its client, so this image would have needed `memory-mcp`'s binary, an autumn
credential, and the wire-lockstep rebuild on every cluster bump. Over HTTP the
dependency is a URL.

## Known issues

Numbered so they can be referred to. `Open` means known and unfixed, not
forgotten. Severity is impact-if-hit, not likelihood.

| ID | Sev | Description | Status |
|---|---|---|---|
| W1 | med | One turn at a time, process-wide. A second `POST /chat/start` attaches to the running turn rather than queueing. Two people using one deployment will interleave. | Open — by design for a single-user UI; revisit if it becomes multi-user |
| W2 | med | Turn state is in-process. A pod restart mid-turn loses the turn; the session DB keeps the history up to the last committed message, but the in-flight answer is gone. | Open — surviving this needs the turn journalled, not just buffered |
| W3 | low | `TurnStream` backlog caps at `BACKLOG_EVENTS`; a longer turn drops its oldest events. Reported as `gap`, never silently. | Accepted |
| W4 | med | `hermes config set model.*` runs in the pod's start script and is best-effort. If hermes renames those keys, the UI starts and chat fails at the first turn with the model unset. | Open — the failure is loud at first use, not at deploy |
| W5 | low | The MCP block is merged into `config.yaml` by a hand-rolled indentation-aware editor (no pyyaml at runtime). A hand-edited config with unusual indentation could be mis-parsed. | Open — write-then-rename means a bad write cannot leave a half file |
| W6 | med | ACP `session/new` is still called with `mcpServers: []`; MCP servers come from `config.yaml` instead. The ACP parameter's HTTP form was never verified against this hermes build, and a wrong shape would parse as "no servers" — a working deploy with a toolless agent. | Open — deliberate; verify then simplify |
| W7 | low | Approvals expire after `APPROVAL_TIMEOUT_SEC` and the turn continues as denied. A slow human loses the operation. | Accepted — the alternative is a turn pinned forever |
| W8 | low | `/api/session/load` replays history into the response, so a very long session's load is one large body rather than a stream. | Accepted — bounded and already complete |

## Inherited lessons (from the lerobot console, kept because they cost real time)

These are not bugs here; they are the reasons some code looks the way it does.

| ID | Lesson |
|---|---|
| L1 | `waitpid(WNOHANG)` right after `SIGHUP` does not reap the shell — one zombie per session, 48 in 26 h. Hence `tini` as PID 1. |
| L2 | An ACP restart that only `terminate()`s races: the old read-loop's exit cleanup failed the NEW process's `initialize`. Wait for the old process, escalate to SIGKILL. |
| L3 | `stderr=DEVNULL` on the ACP child made every "acp exited" undebuggable. It goes to a file. |
| L4 | Creating a session eagerly littered the store with titleless zero-message ghosts, one per page open. The first prompt creates it. |
| L5 | ACP `session/list` caches in memory, so CLI-deleted sessions kept reappearing. Read hermes' `SessionDB` directly. |
| L6 | The steering directive is APPENDED, not prepended, or the session auto-titles itself from the directive instead of the user's words. |
| L7 | `X-Accel-Buffering: no` — nginx buffers `text/event-stream` by default and delivers the whole "stream" at the end. |

## Testing

`tests/test_turn_stream.py` runs the real app with a fake ACP, so the transport,
routing and SSE framing under test are the production ones. It pins the claim
the design rests on: a disconnect does not stop the turn, and a reconnect
replays only the delta. Both were ablation-checked — breaking the delta logic
turns exactly those tests red.

```
python -m pytest tests/ -q
```
