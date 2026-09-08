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

Turns are keyed by SESSION, and several can run at once. One `hermes acp`
process does not mean one conversation: ACP addresses every session-scoped
frame by `sessionId` — `session/prompt`, `session/update`, `session/cancel`,
and `session/request_permission`, where the field is required rather than
optional — the Python SDK dispatches each incoming request onto its own task
instead of serialising them, and hermes' adapter keeps a `Dict[str,
SessionState]` on a `ThreadPoolExecutor(max_workers=4)`. The limit was on this
side: a single `session_id` field and a single pair of callbacks on the ACP
client, which is also why `session/load` used to read as "move the agent".

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
- **Sessions are warmed on view, debounced.** A session's first contact with the
  ACP process costs ~1.3 s — and it is not the history read: `session/new`
  measures 1310 ms against `session/load`'s 1315 ms, because both rebuild the
  agent's tool surface. Unwarmed it lands after the reader pressed enter, so
  opening a conversation schedules the introduction instead and the send finds
  it done. The debounce (`PREFETCH_DEBOUNCE_SEC`) is what keeps that from
  spending 1.3 s on every sidebar row someone scrolled past.
- **A browser id, minted by the server.** Not authentication — everyone shares
  one credential and is trusted. It exists so two people on one deployment can
  be told apart where that matters, and only there: whose double-click is whose,
  whose position in the sidebar, whose warm-up. Everything else stays shared,
  because it is one deployment and one session list.
- **Stop escalates conditionally.** `session/cancel` cannot interrupt a running
  tool, so Stop escalates to restarting the process — which kills every session,
  not just the wedged one. With another turn in flight the endpoint reports the
  cancel as unhonoured (`409 unyielding`) instead of taking a bystander down.

## What was dropped from `lerobot-agent-console`, and why that is safe

The PTY terminal, the port proxy, service discovery, and the lerobot/volcano
endpoints. Most of that console's hardest-won fixes are terminal fixes —
process-group reaping, idle reclamation, output backlogs, the zombie-per-session
leak. **None can regress here, because there is no terminal.**

## Dependencies

| Depends on | How | Consequence |
|---|---|---|
| `hermes` | child process, ACP over stdio | one warm process multiplexing up to `MAX_CONCURRENT_TURNS` sessions; a cold `hermes chat` per turn costs ~10 s |
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
| W1 | med | ~~One turn at a time, process-wide.~~ | **FIXED** — and the reason it was ever true is worth keeping: not the pipe, not the protocol, not hermes. This client held ONE `session_id` and ONE pair of callbacks, so a second turn had nowhere to deliver and `session/load` had to move a pointer. Both are keyed by session now; the read loop routes on the `sessionId` every frame already carried. `MAX_CONCURRENT_TURNS` defaults to 4, which is hermes' own `max_workers`. **This fixes the turn-level contention only** — the original row also said "two people using one deployment will interleave", and the rest of that is W13. See also W11 and W12. |
| W2 | med | Turn state is in-process. A pod restart mid-turn loses the turn; the session DB keeps the history up to the last committed message, but the in-flight answer is gone. The UI now RECOVERS from it — an EventSource error asks `/api/chat/status` before claiming a reconnect, and an unknown id is reported as lost rather than reconnected-to forever. | Open — surviving the restart itself needs the turn journalled, not just buffered |
| W3 | low | `TurnStream` backlog caps at `BACKLOG_EVENTS`; a longer turn drops its oldest events. Reported as `gap`, never silently. | Accepted |
| W4 | med | `hermes config set model.*` runs in the pod's start script and is best-effort. If hermes renames those keys, the UI starts and chat fails at the first turn with the model unset. | Open — the failure is loud at first use, not at deploy |
| W5 | low | The MCP block is merged into `config.yaml` by a hand-rolled indentation-aware editor (no pyyaml at runtime). A hand-edited config with unusual indentation could be mis-parsed. | Open — write-then-rename means a bad write cannot leave a half file |
| W6 | med | ~~ACP `session/new` called with `mcpServers: []`~~ | **FIXED** — and the risk was real: config.yaml alone gave a toolless agent. The shape is `{type: "http", name, url, headers: []}`; `type` is the union discriminator (`HttpMcpServer` subclasses `McpServerHttp` only to add it) and `headers` has no default, so omitting either is rejected outright. See W10. |
| W7 | low | Approvals expire after `APPROVAL_TIMEOUT_SEC` and the turn continues as denied. A slow human loses the operation. | Accepted — the alternative is a turn pinned forever |
| W8 | low | `/api/session/load` replays history into the response, so a very long session's load is one large body rather than a stream. | Accepted — bounded and already complete |
| W9 | low | The agent carries tools this UI cannot use (all `browser_*`, `delegate_task`). `acp_adapter/session.py` builds every session with a hardcoded `_expand_acp_enabled_toolsets(["hermes-acp"], ...)`; neither `agent.disabled_toolsets` nor `agent.enabled_toolsets` in config.yaml reaches it, and no env overrides it. Both were tried against 0.17. | Open — needs a patch to hermes; `HERMES_NO_DEP_INSTALL=1` at least stops the browser availability check from downloading Chromium |
| W10 | med | An MCP server must be declared in BOTH config.yaml and the ACP `session/new` parameter. The file decides the `mcp-<name>` toolset name; the parameter makes the connection. Either alone yields an agent with no such tool — and `hermes mcp list` shows "enabled" in both cases. | Accepted — both are written, and `/api/status` reports the URL |
| W11 | med | A Stop that the agent will not honour cannot escalate while another session is streaming: the escalation is a process restart, and the process carries every session. The endpoint answers `409 {how: "unyielding"}` and the turn keeps running until its tool returns. | Accepted — the alternative kills a bystander's reply to serve the person pressing Stop |
| W13 | med | ~~Server state that belonged to one person was shared by everyone.~~ | **FIXED** — three separate leaks, one missing concept. A second person sending into a conversation already replying hit the branch that makes a double-click harmless, and their message was discarded WITHOUT `text` ever being read; `/api/approval/pending` was global, so a bystander saw and could answer a permission prompt from a conversation they had never opened; `last_session` and `viewing` were single-valued, so a fresh page adopted someone else's position and two readers cancelled each other's warm-up. A server-minted browser cookie settles the first and third; approvals are scoped to the SESSION instead, because whoever is reading a shared conversation should be able to answer it. |
| W14 | low | The browser id is not identity. It is forgeable, everyone shares one HTTP credential, and by design anyone can read any stream, cancel any turn, and delete any session — a shared conversation must stay openable by whoever is looking at it. What is isolated is a person's POSITION, not their permissions. | Accepted — real per-user isolation needs real auth, which this deployment does not have |
| W12 | low | `MAX_CONCURRENT_TURNS` defaults to 4 because that is hermes' `ThreadPoolExecutor(max_workers=4)`. The limit BELOW it is the model: `freetoken-l3` runs `--max-running-requests 1`, so a second concurrent turn queues at the model rather than replying in parallel. Raising one without the other only moves the queue. | Accepted — queuing at the model beats being refused at the door |

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
| L8 | A permission request was awaited INSIDE the ACP read loop. It parks on a human for up to `APPROVAL_TIMEOUT_SEC`, so with sessions multiplexed it froze every other conversation's tokens behind one person's decision. It is a detached task now — with a strong reference held, because asyncio only weakly references a task and a collected one leaves the agent waiting on an answer nobody will write. |
| L9 | "One process" is not "one session", and reading it that way cost this UI its concurrency for months. The evidence was in the schema the whole time: `sessionId` is a REQUIRED field on `RequestPermissionRequest`, which only makes sense if two sessions can have a prompt outstanding at once. |

## Testing

`tests/test_turn_stream.py` runs the real app with a fake ACP, so the transport,
routing and SSE framing under test are the production ones. It pins the claim
the design rests on: a disconnect does not stop the turn, and a reconnect
replays only the delta. Both were ablation-checked — breaking the delta logic
turns exactly those tests red.

The multiplexing has its own set, and the routing ones drive the PRODUCTION
`_read_loop` over a fake pipe rather than a stubbed client. Ablation-checked
together: broadcasting updates instead of routing them, escalating a Stop
unconditionally, awaiting the permission reply inline, and dropping the
"unnamed send continues the last conversation" rule each turn exactly the
tests that name them red.

```
python -m pytest tests/ -q
```
