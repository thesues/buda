"""Images and videos, stored in autumn rather than in the transcript.

The thing this replaces was a data URL pasted into the chat text. It works for
exactly one message and then poisons everything downstream: the base64 is the
transcript, so it is re-sent to the model on every subsequent turn of that
conversation, it fills the reader's screen, and hermes' own `vision_analyze`
treats the string as a path and dies on `[Errno 36] File name too long`. A
1280x720 PNG is ~5 KB of pixels and ~7 KB of base64 — small enough to feel
harmless and large enough to ruin a conversation.

So media lives in autumn under `media/<session>/<id>.<ext>` and the transcript
carries a URL. The browser fetches it from us; we fetch it from autumn.

Why the PyO3 client and not a mount. `autumn.Fs` is a direct client — one round
trip per call to the partition server. A FUSE mount would put the kernel in the
middle of every byte for no benefit here, and its per-read cost is the same
round trip plus the kernel's. This module does whole-file reads and writes of a
few MB; it wants the shortest path, not a filesystem.

The cost, and it is real: this binds the webui image to autumn's WIRE lockstep.
`Dockerfile.webui` called that out as "the part that actually matters" when it
deliberately kept the client out. It is in now, so a cluster wire bump requires
rebuilding this image with it. There is no way to have the client and not the
lockstep — the wire format is the client.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from typing import Optional, Tuple

log = logging.getLogger("buda.media")

# `media/` under the fs namespace, beside `docs/` and `models/`.
ROOT = os.environ.get("BUDA_MEDIA_ROOT", "media")

# What a browser is willing to render inline, and nothing else. An upload whose
# type is not here is refused at the door rather than stored and then served as
# something the browser has to guess about.
CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
}

# An image a person pastes, or a clip this cluster makes. Both are bounded; the
# point of the bound is that a stray multi-GB POST cannot be held in memory
# while it is written.
MAX_BYTES = int(os.environ.get("BUDA_MEDIA_MAX_BYTES", str(256 * 1024 * 1024)))

_lock = threading.Lock()
_fs = None          # type: ignore[var-annotated]
_fs_failed: Optional[str] = None


def _read_credential(path: str) -> Tuple[str, bytes]:
    """`(principal, raw secret)` from a credential file, or `("", b"")`.

    Accepts the three shapes the Rust reader accepts: a `credential: <hex>`
    line with an optional `principal: <name>`, two bare lines, or a single bare
    hex line (anonymous). Anything else is left to fail at connect time with
    the cluster's own message rather than being guessed at here.
    """
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        log.info("no credential at %s (%s); connecting anonymously", path, e)
        return "", b""

    principal, hexed = "", ""
    bare: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("principal:"):
            principal = line.split(":", 1)[1].strip()
        elif line.startswith("credential:"):
            hexed = line.split(":", 1)[1].strip()
        else:
            bare.append(line)
    if not hexed and bare:
        hexed = bare[-1]
        if len(bare) >= 2 and not principal:
            principal = bare[0]
    if not hexed:
        return "", b""
    return principal, bytes.fromhex(hexed)


def _connect():
    """One `Fs` for the process, built on first use.

    Lazily, because the app must start and serve `/healthz` whether or not
    autumn is reachable — the mount-at-boot version of this made a cluster blip
    into a crash-loop. A failure is remembered so every later request fails the
    same way instead of re-dialing on each one, and it is cleared on success.
    """
    global _fs, _fs_failed
    with _lock:
        if _fs is not None:
            return _fs
        import autumn  # noqa: PLC0415 -- import cost stays off the startup path

        transport = os.environ.get("AUTUMN_TRANSPORT", "tcp")
        try:
            autumn.set_transport(transport)
        except Exception:  # noqa: BLE001 -- already set is not an error
            log.debug("set_transport(%s) refused; assuming it is already set", transport)

        mgr = os.environ.get("AUTUMN_MANAGER", "")
        if not mgr:
            raise RuntimeError("AUTUMN_MANAGER is unset; media storage has no cluster")

        # `Fs.connect` takes RAW credential bytes and a principal, both-or-
        # neither — not a path. The file's format has two readers already
        # (`autumn_client::parse_credential_text` and autumn_kvcache's
        # `read_credential_pair`); this is a third, kept deliberately small.
        #
        # Raw bytes, not the ASCII hex: the manager stores the SHA-256 of the
        # decoded secret, so passing the hex through authenticates as something
        # else and every protected-prefix op fails with PermissionDenied once
        # enforcement is on. That is documented where the other reader lives,
        # and it is the reason this does not just hand the file's text over.
        principal, secret = _read_credential(
            os.environ.get("AUTUMN_FS_CREDENTIAL", "/etc/autumn/cred/fs.cred")
        )
        if principal:
            _fs = autumn.Fs.connect(mgr, principal=principal, credential=secret)
        else:
            _fs = autumn.Fs.connect(mgr)
        _fs_failed = None
        log.info("media store connected: manager=%s root=%s", mgr, ROOT)
        return _fs


def available() -> Tuple[bool, str]:
    """Whether media storage can be used, and why not when it cannot.

    Used by `/api/status` so the UI can hide the attach button rather than
    offer one that fails on click.
    """
    try:
        _connect()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _split(path: str) -> list[str]:
    return [p for p in path.split("/") if p and p not in (".", "..")]


def _mkdirs(fs, path: str) -> int:
    """`mkdir -p`, returning the leaf inode.

    Each level is create-or-lookup: two callers uploading at once both try to
    make `media/<session>` and one of them loses, which is not an error.
    """
    ino = fs.resolve("/")["ino"] if isinstance(fs.resolve("/"), dict) else fs.resolve("/")
    for name in _split(path):
        try:
            ino = fs.mkdir(ino, name, 0o755)
        except Exception:  # noqa: BLE001 -- already exists, or lost the race
            got = fs.lookup(ino, name)
            ino = got["ino"] if isinstance(got, dict) else got
    return ino


def put(data: bytes, ext: str, session: str = "shared") -> str:
    """Store bytes, return the media id (`<session>/<id>.<ext>`).

    The id carries the extension because that is what decides the Content-Type
    on the way back out; storing the type separately would be a second thing to
    keep consistent with the first.
    """
    ext = ext.lower().lstrip(".")
    if ext not in CONTENT_TYPES:
        raise ValueError(f"unsupported media type: {ext!r}")
    if not data:
        raise ValueError("refusing to store an empty file")
    if len(data) > MAX_BYTES:
        raise ValueError(f"{len(data)} bytes exceeds the {MAX_BYTES} limit")

    fs = _connect()
    sess = "".join(c for c in (session or "shared") if c.isalnum() or c in "-_") or "shared"
    name = f"{secrets.token_hex(8)}.{ext}"
    parent = _mkdirs(fs, f"{ROOT}/{sess}")
    ino = fs.create(parent, name, 0o644)

    # One write. `Fs.write` takes the whole buffer; chunking here would only
    # add partial-file states that a reader could observe.
    fs.write(ino, 0, data)
    fs.flush(ino)
    log.info("stored media %s/%s (%d bytes)", sess, name, len(data))
    return f"{sess}/{name}"


def get(media_id: str) -> Tuple[bytes, str]:
    """Read a media id back, with the Content-Type its extension implies."""
    parts = _split(media_id)
    if len(parts) != 2:
        raise ValueError(f"bad media id: {media_id!r}")
    sess, name = parts
    ext = name.rsplit(".", 1)[-1].lower()
    ctype = CONTENT_TYPES.get(ext)
    if ctype is None:
        raise ValueError(f"unsupported media type: {ext!r}")

    fs = _connect()
    got = fs.resolve(f"/{ROOT}/{sess}/{name}")
    ino = got["ino"] if isinstance(got, dict) else got
    meta = fs.getattr(ino)
    size = int(meta["size"] if isinstance(meta, dict) else meta)
    if size > MAX_BYTES:
        raise ValueError(f"{media_id} is {size} bytes, over the serving limit")

    # `read` answers one call; a short read means the file is shorter than its
    # attr claimed, which is worth surfacing rather than padding over.
    data = bytes(fs.read(ino, 0, size))
    if len(data) != size:
        log.warning("%s: attr says %d bytes, read %d", media_id, size, len(data))
    return data, ctype
