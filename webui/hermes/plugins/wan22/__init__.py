"""Wan2.2 on vLLM-Omni as a hermes video_gen provider.

Why a provider and not a skill calling an API. hermes already has the whole
surface: a `video_generate` tool, a `VideoGenProvider` interface, and a
registry selected by `video_gen.provider` in config.yaml. The tool's own
description is the contract this implements — "text-to-video (prompt only) and
image-to-video (prompt + image_url), the active backend auto-routes" — and
vLLM-Omni's `/v1/videos` splits on exactly the same condition: an
`image_reference` in the request means I2V, its absence means T2V. Two
interfaces that already agree need an adapter, not a new mechanism.

The sync endpoint, not the async one. `/v1/videos/sync` blocks and returns raw
`video/mp4` bytes with the inference time in a header, so there is no job id to
poll and no place for a poll loop to lose track of one. It is documented as
being for benchmarks and testing; the cost is that a request occupies the
connection for its whole duration, which for 720p is minutes. That is the right
trade for a chat turn, which is already waiting.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    save_bytes_video,
    success_response,
)

log = logging.getLogger(__name__)

# In-cluster Service. The endpoint is overridable because this plugin outlives
# any one deployment of the model behind it.
DEFAULT_ENDPOINT = os.environ.get("WAN22_ENDPOINT", "http://wan22-i2v:8000")
DEFAULT_MODEL = os.environ.get("WAN22_MODEL", "wan22-i2v")

# 1280x720 at 16 fps for 5 s. 81 frames, not 80: Wan's temporal compression
# wants 4n+1, and vLLM-Omni's own benchmark rounds the same way.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 16
DEFAULT_STEPS = int(os.environ.get("WAN22_STEPS", "6"))
DEFAULT_ASPECT = "16:9"

# The read has to outlast a 720p generation, which is minutes, not seconds.
# Whatever this is set to is the real ceiling on `duration` x `resolution`.
REQUEST_TIMEOUT_S = int(os.environ.get("WAN22_TIMEOUT_S", "1800"))

_ASPECT = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (960, 960),
}

# Wan's VAE downsamples by 8 and the transformer patches by 2, so both edges
# must be multiples of 16. Rounding rather than truncating keeps the aspect
# closer on odd sizes.
SIZE_MULTIPLE = 16
# Roughly 720p worth of pixels. Holding AREA rather than a fixed edge is what
# lets a portrait photo stay portrait without costing more than a landscape one
# — the cost of a diffusion step is pixels, not width.
TARGET_PIXELS = 1280 * 720


def _snap(n: int) -> int:
    return max(SIZE_MULTIPLE, int(round(n / SIZE_MULTIPLE)) * SIZE_MULTIPLE)


def _size_for(image_bytes: Optional[bytes], aspect_ratio: str) -> Tuple[int, int, str]:
    """Output size, taken from the IMAGE when there is one.

    A picture that is 4:3 or 3:2 — most photographs — rendered into a fixed
    1280x720 is either stretched or cropped, and neither is what "make this
    move" asked for. So read the source's own ratio and hold the pixel budget
    instead of the shape.

    Falls back to the named ratio when there is no image or it cannot be read;
    an explicit `aspect_ratio` from the caller still wins, because someone who
    asked for 9:16 means it.
    """
    if aspect_ratio in _ASPECT and aspect_ratio != DEFAULT_ASPECT:
        w, h = _ASPECT[aspect_ratio]
        return w, h, f"caller asked for {aspect_ratio}"
    if image_bytes:
        try:
            from PIL import Image  # noqa: PLC0415
            import io  # noqa: PLC0415

            with Image.open(io.BytesIO(image_bytes)) as im:
                sw, sh = im.size
            if sw > 0 and sh > 0:
                scale = (TARGET_PIXELS / (sw * sh)) ** 0.5
                w, h = _snap(sw * scale), _snap(sh * scale)
                return w, h, f"from the image ({sw}x{sh}, {sw / sh:.2f}:1)"
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read the image's size (%s); using %s", exc, aspect_ratio)
    w, h = _ASPECT.get(aspect_ratio, _ASPECT[DEFAULT_ASPECT])
    return w, h, f"default {aspect_ratio}"


def _resolve_image(url: Optional[str]) -> Tuple[Optional[str], Optional[bytes]]:
    """Turn a webui media reference into something the video server can read.

    An image a person attached in the chat is already in autumn, and the URL it
    is referred to by — `/api/media?id=<session>/<name>` — is RELATIVE to the
    webui. Handing that to vLLM-Omni would have it resolve the path against its
    own host and find nothing. So read the bytes out of autumn here and pass a
    data URL.

    This is also what lets the skill say "pass the URL straight through". Given
    anything else — an http(s) link, a data URL already — this returns it
    untouched and the server fetches it itself.
    """
    if not url:
        return None, None
    if url.startswith("data:") and ";base64," in url:
        # Already inline: decode once here so the size can be read from it.
        try:
            import base64  # noqa: PLC0415
            return url, base64.b64decode(url.split(",", 1)[1])
        except Exception:  # noqa: BLE001
            return url, None
    if not url.startswith("/api/media"):
        # An http(s) link the server fetches itself. We never see the bytes, so
        # the size cannot be read and the named ratio is what is left.
        return url, None
    try:
        import base64  # noqa: PLC0415
        import media  # noqa: PLC0415 -- the webui's store, same process

        mid = url.split("id=", 1)[1].split("&", 1)[0]
        from urllib.parse import unquote  # noqa: PLC0415

        data, ctype = media.get(unquote(mid))
        log.info("resolved %s from autumn (%d bytes, %s)", url, len(data), ctype)
        return f"data:{ctype};base64," + base64.b64encode(data).decode(), data
    except Exception as exc:  # noqa: BLE001
        # Surface it as the caller's error rather than sending a relative URL
        # the server will fail on with a less obvious message.
        raise ValueError(f"could not read {url} from autumn: {exc}") from exc


def _frames_for(duration_s: int, fps: int) -> int:
    """Wan's VAE compresses time by 4, so frame counts are 4n+1.

    Rounded UP, not down. Down is the obvious spelling and it silently returns
    a shorter clip than asked for: 5 s at 16 fps is 80 frames, and
    `((80-1)//4)*4+1` is 77 — 4.81 s, delivered as if it were the 5 the caller
    requested. Up gives 81, which is 5.06 s: over by one frame rather than
    under by three, and the caller gets at least the duration it asked for.
    """
    n = max(1, int(round(duration_s * fps)))
    return ((n - 1 + 3) // 4) * 4 + 1


class Wan22VideoGenProvider(VideoGenProvider):
    # `name` and `display_name` are @property on the base class, not methods.
    # Defining them as plain methods makes `provider.name` a bound method, and
    # registration refuses it with "Video gen provider .name must be a
    # non-empty string" -- which is the registry doing its job, at load time.
    @property
    def name(self) -> str:
        return "wan22"

    @property
    def display_name(self) -> str:
        return "Wan2.2 (vLLM-Omni, in-cluster)"

    def is_available(self) -> bool:
        """Reachability, not readiness.

        A cold start reads ~127 GB of weights off the autumn mount and takes
        minutes, during which /health refuses the connection. Reporting
        unavailable then would hide the backend from `hermes tools` for the
        whole window; a generate() during it fails with a clear message
        instead, which is the more useful failure.
        """
        return bool(DEFAULT_ENDPOINT)

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def list_models(self) -> List[Dict[str, Any]]:
        return [{
            "id": DEFAULT_MODEL,
            "name": "Wan2.2-I2V-A14B",
            "description": "Image-to-video, 720p. This checkpoint's pipeline is "
                           "WanImageToVideoPipeline and REQUIRES an image; "
                           "text-only prompts are not served by it.",
        }]

    def capabilities(self) -> Dict[str, Any]:
        return {
            "aspect_ratios": list(_ASPECT),
            "resolutions": ["720p"],
            "max_duration": 10,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": 1,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,      # forward-compat: unknown keys are ignored, never a TypeError
    ) -> Dict[str, Any]:
        import requests   # noqa: PLC0415 -- keep the import off the plugin-load path

        mdl = model or DEFAULT_MODEL
        secs = int(duration or 5)

        # `image_url` is the whole T2V/I2V switch, on both sides of this
        # adapter: hermes documents it as the routing condition and vLLM-Omni
        # reads `image_reference` as one. `reference_image_urls` is the
        # multi-image surface, which this model has no use for; take the first
        # if that is all the caller gave us rather than silently doing T2V.
        img = image_url or (reference_image_urls[0] if reference_image_urls else None)
        try:
            img, img_bytes = _resolve_image(img)
        except ValueError as exc:
            return error_response(
                error=str(exc), error_type="bad_reference", provider=self.name,
                model=mdl, prompt=prompt, aspect_ratio=aspect_ratio,
            )

        # After resolving, because the source's own dimensions are the best
        # answer and they are in those bytes.
        w, h, why = _size_for(img_bytes, aspect_ratio)
        log.info("output %dx%d (%s)", w, h, why)
        frames = _frames_for(secs, DEFAULT_FPS)

        form: Dict[str, Any] = {
            "prompt": prompt,
            "model": mdl,
            "width": str(w),
            "height": str(h),
            "fps": str(DEFAULT_FPS),
            "num_frames": str(frames),
            "num_inference_steps": str(DEFAULT_STEPS),
        }
        if negative_prompt:
            form["negative_prompt"] = negative_prompt
        if seed is not None:
            form["seed"] = str(seed)
        if img:
            # A JSON-encoded sub-object in a multipart field: that is how the
            # server declares it (`image_reference: str | None = Form(...)`,
            # then `_parse_form_json`).
            form["image_reference"] = json.dumps({"image_url": img})

        if not img:
            # The deployed checkpoint is I2V: its pipeline class is
            # WanImageToVideoPipeline, which takes an image as a required
            # argument. Sending a text-only request would fail inside the
            # server with a stack trace about a missing tensor; saying so here
            # is the same answer, minutes earlier and in words.
            return error_response(
                error="This backend serves image-to-video only: give it an "
                      "image (image_url), or deploy the T2V checkpoint "
                      "alongside it.",
                error_type="unsupported_modality", provider=self.name,
                model=mdl, prompt=prompt, aspect_ratio=aspect_ratio,
            )

        modality = "image"
        url = f"{DEFAULT_ENDPOINT.rstrip('/')}/v1/videos/sync"

        try:
            r = requests.post(url, files={k: (None, v) for k, v in form.items()},
                              timeout=REQUEST_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"{url} unreachable: {exc}. The backend reads its weights "
                      f"off the autumn mount at start and is not servable for "
                      f"several minutes after a restart.",
                error_type="connection_error", provider=self.name,
                model=mdl, prompt=prompt, aspect_ratio=aspect_ratio,
            )

        if r.status_code != 200:
            detail = r.text[:400]
            return error_response(
                error=f"HTTP {r.status_code}: {detail}",
                error_type="provider_error", provider=self.name,
                model=mdl, prompt=prompt, aspect_ratio=aspect_ratio,
            )

        # The body IS the mp4. Metadata rides in headers.
        #
        # Into autumn, beside the image it was made from, so the webui can serve
        # it back by URL and it survives this pod. `save_bytes_video` writes to
        # `$HERMES_HOME/cache/videos/`, which is a cache in the pod: a URL
        # handed to a browser from there is a path the browser cannot fetch and
        # a file the next pod will not have. Fall back to it only if autumn is
        # unreachable, so a generation that cost minutes is not simply lost.
        try:
            import media  # noqa: PLC0415 -- the webui's store, same process

            mid = media.put(r.content, "mp4", session=kwargs.get("session") or "video")
            where = f"/api/media?id={mid}"
        except Exception as exc:  # noqa: BLE001
            log.warning("could not store the clip in autumn (%s); keeping it local", exc)
            where = str(save_bytes_video(r.content, prefix="wan22", extension="mp4"))

        return success_response(
            video=where, model=mdl, prompt=prompt, modality=modality,
            aspect_ratio=aspect_ratio, duration=secs, provider=self.name,
            extra={
                "inference_time_s": r.headers.get("X-Inference-Time-S"),
                "request_id": r.headers.get("X-Request-Id"),
                "frames": frames,
                "steps": DEFAULT_STEPS,
                "size": f"{w}x{h}",
                "size_reason": why,
            },
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(Wan22VideoGenProvider())
