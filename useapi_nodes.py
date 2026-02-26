"""ComfyUI-UseapiNet: Custom nodes for Useapi.net API integration.

Provides image and video generation via:
  - Google Flow: Imagen 4, Gemini (Nano Banana), Veo 3.1
  - Runway: Gen-4, Gen-4 Turbo, Gen-3 Turbo, Frames
"""
import os
import io
import json
import time
import uuid
import base64
import hashlib
import tempfile
import urllib.request
import urllib.error
import urllib.parse
import numpy as np
import torch
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
LOG = "[Useapi.net]"
BASE_URL = "https://api.useapi.net/v1"
VIDEO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "comfyui_useapi_videos")

# ── Shared Utilities ─────────────────────────────────────────────────────────

def _get_token(api_token: str) -> str:
    """Return API token: use direct input, else USEAPI_TOKEN env var."""
    token = (api_token or "").strip()
    if not token:
        token = os.environ.get("USEAPI_TOKEN", "").strip()
    if not token:
        raise ValueError(
            f"{LOG} API token not provided. "
            "Set the USEAPI_TOKEN environment variable or wire the api_token input."
        )
    return token


def _auth_headers(token: str) -> dict:
    """Return JSON auth headers for Useapi.net requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _make_request(url: str, method: str = "GET", headers: dict = None,
                  data: bytes = None, timeout: int = 600):
    """Make an HTTP request. Returns (status_code, response_body_bytes)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"{LOG} Network error reaching {url}: {e.reason}")
    except TimeoutError:
        raise RuntimeError(f"{LOG} Request timed out after {timeout}s: {url}")


def _check_status(status: int, body: bytes, url: str, context: str = "") -> dict:
    """Parse JSON response body; raise descriptive error if status != 200."""
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {}
    if status == 200:
        return data
    detail = data.get("error", body[:300].decode(errors="replace"))
    label = f"[{context}] " if context else ""
    if status == 429:
        raise RuntimeError(
            f"{LOG} {label}Rate limited (429). Wait 5-10s or add more Useapi.net accounts. "
            f"URL: {url}\nDetail: {detail}"
        )
    if status == 503:
        raise RuntimeError(
            f"{LOG} {label}Service unavailable (503). Retry in a moment. URL: {url}\nDetail: {detail}"
        )
    if status == 408:
        raise RuntimeError(
            f"{LOG} {label}Request timeout (408). Generation took too long. URL: {url}\nDetail: {detail}"
        )
    if status == 401:
        raise RuntimeError(
            f"{LOG} {label}Unauthorized (401). Check your Useapi.net token. URL: {url}"
        )
    raise RuntimeError(f"{LOG} {label}HTTP {status} from {url}.\nDetail: {detail}")


def _build_multipart(fields: dict, files: dict):
    """Build multipart/form-data without the requests library.

    Args:
        fields: {"name": "string_value"}
        files:  {"name": ("filename.ext", bytes_data, "mime/type")}
    Returns:
        (body_bytes, content_type_string_with_boundary)
    """
    boundary = "----ComfyUIBoundary" + uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode()
        body += b"\r\n"
    for name, (filename, data, ctype) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        body += data
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert ComfyUI IMAGE tensor (1, H, W, 3) float32 → PNG bytes."""
    arr = tensor[0].cpu().detach().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_tensor(img_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes → ComfyUI IMAGE tensor (1, H, W, 3) float32."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _download_file(url: str, ext: str = ".mp4") -> str:
    """Download URL to cache dir with MD5-hash filename. Returns local path."""
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    fname = hashlib.md5(url.encode()).hexdigest() + ext
    dest = os.path.join(VIDEO_CACHE_DIR, fname)
    if os.path.exists(dest):
        print(f"{LOG} Cache hit: {dest}")
        return dest
    print(f"{LOG} Downloading to {dest} ...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"{LOG} Downloaded {len(data):,} bytes → {dest}")
    return dest


def _runway_poll(task_id: str, token: str,
                 poll_interval: int = 10, max_wait: int = 600) -> list:
    """Poll Runway task until SUCCEEDED. Returns list of artifacts dicts."""
    poll_url = (
        f"{BASE_URL}/runwayml/tasks/?taskId={urllib.parse.quote(task_id, safe='')}"
    )
    headers = _auth_headers(token)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(poll_interval)
        status, raw = _make_request(poll_url, "GET", headers, None, 30)
        data = _check_status(status, raw, poll_url, f"Runway poll {task_id[:30]}")
        task_status = data.get("status", "")
        print(f"{LOG} Runway task {task_id[:30]}... → {task_status}")
        if task_status == "SUCCEEDED":
            artifacts = data.get("artifacts", [])
            if not artifacts:
                raise RuntimeError(
                    f"{LOG} Runway task SUCCEEDED but no artifacts in response: {data}"
                )
            return artifacts
        if task_status in ("FAILED", "CANCELLED", "THROTTLED"):
            raise RuntimeError(
                f"{LOG} Runway task ended with status '{task_status}'. task_id={task_id}"
            )
    raise RuntimeError(
        f"{LOG} Runway task timed out after {max_wait}s. task_id={task_id}"
    )


def _runway_upload_image(token: str, image_tensor: torch.Tensor,
                         email: str = "") -> str:
    """Upload a ComfyUI IMAGE tensor to Runway as an image asset. Returns assetId."""
    url = f"{BASE_URL}/runwayml/assets"
    png_bytes = _tensor_to_png_bytes(image_tensor)
    fields = {"email": email.strip()} if email.strip() else {}
    files = {"file": ("comfyui_upload.png", png_bytes, "image/png")}
    body, ct = _build_multipart(fields, files)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": ct}
    print(f"{LOG} Runway: uploading image asset...")
    status, raw = _make_request(url, "POST", headers, body, timeout=60)
    data = _check_status(status, raw, url, "Runway upload asset")
    asset_id = data.get("assetId", "")
    if not asset_id:
        raise RuntimeError(f"{LOG} Runway upload: no assetId in response: {data}")
    print(f"{LOG} Runway asset uploaded: {asset_id[:50]}...")
    return asset_id

# ── Node classes added in Tasks 3-16 ─────────────────────────────────────────

# ── Node 1: Useapi Token From Env ─────────────────────────────────────────────
class UseapiTokenFromEnv:
    """Load Useapi.net API token from an environment variable."""

    CATEGORY = "Useapi.net/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_token",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_var_name": ("STRING", {"default": "USEAPI_TOKEN"}),
            }
        }

    def execute(self, env_var_name: str):
        token = os.environ.get(env_var_name, "").strip()
        if not token:
            raise ValueError(
                f"{LOG} Environment variable '{env_var_name}' is not set or empty. "
                "Export it before launching ComfyUI."
            )
        print(f"{LOG} Token loaded from env var '{env_var_name}'")
        return (token,)

# ── ComfyUI Registration ──────────────────────────────────────────────────────
# ── Node 2: Veo Generate Video ────────────────────────────────────────────────
class UseapiVeoGenerate:
    """Generate video using Google Veo 3.1 via Google Flow.

    Server-side auto-poll: blocks until complete (~60-180s). Timeout: 600s.
    Outputs: video_url, local video_path, media_generation_id (for upscale/extend).
    """

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "media_generation_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["veo-3.1-fast", "veo-3.1-quality", "veo-3.1-fast-relaxed", "veo-3", "veo-2"],),
                "aspect_ratio": (["landscape", "portrait"],),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "count": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            },
        }

    def execute(self, prompt: str, model: str, aspect_ratio: str,
                api_token: str = "", email: str = "",
                count: int = 1, seed: int = 0):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/videos"
        body = {"prompt": prompt, "model": model, "aspectRatio": aspect_ratio, "count": count}
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()

        print(f"{LOG} Veo Generate: model={model}, prompt='{prompt[:60]}...'")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=600)
        data = _check_status(status, raw, url, "Veo generate")

        ops = data.get("operations", [])
        if not ops:
            raise RuntimeError(f"{LOG} Veo generate returned no operations: {data}")
        op = ops[0]
        op_status = op.get("status", "")
        if "FAIL" in op_status.upper():
            raise RuntimeError(f"{LOG} Veo generation failed. Status={op_status}. op={op}")
        video_meta = op.get("operation", {}).get("metadata", {}).get("video", {})
        video_url = video_meta.get("fifeUrl", "")
        media_gen_id = video_meta.get("mediaGenerationId", "")
        if not video_url:
            raise RuntimeError(f"{LOG} Veo generate: no fifeUrl in response. video_meta={video_meta}")

        print(f"{LOG} Veo Generate: complete. mediaGenerationId={media_gen_id[:50]}...")
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, media_gen_id)


# ── Node 3: Veo Upscale Video ─────────────────────────────────────────────────
class UseapiVeoUpscale:
    """Upscale a Veo-generated video to 1080p or 4K using its mediaGenerationId."""

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_generation_id": ("STRING", {"default": ""}),
                "resolution": (["1080p", "4K"],),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
            },
        }

    def execute(self, media_generation_id: str, resolution: str, api_token: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/videos/upscale"
        body = {"mediaGenerationId": media_generation_id, "resolution": resolution}
        print(f"{LOG} Veo Upscale: {resolution}, mediaGenerationId={media_generation_id[:50]}...")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=600)
        data = _check_status(status, raw, url, "Veo upscale")

        ops = data.get("operations", [])
        if not ops:
            raise RuntimeError(f"{LOG} Veo upscale returned no operations: {data}")
        video_meta = ops[0].get("operation", {}).get("metadata", {}).get("video", {})
        video_url = video_meta.get("fifeUrl", "")
        if not video_url:
            raise RuntimeError(f"{LOG} Veo upscale: no fifeUrl in response.")
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path)


# ── Node 4: Veo Extend Video ──────────────────────────────────────────────────
class UseapiVeoExtend:
    """Extend an existing Veo video with a continuation prompt."""

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "media_generation_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_generation_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
            },
        }

    def execute(self, media_generation_id: str, prompt: str, api_token: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/videos/extend"
        body = {"mediaGenerationId": media_generation_id, "prompt": prompt}
        print(f"{LOG} Veo Extend: mediaGenerationId={media_generation_id[:50]}...")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=600)
        data = _check_status(status, raw, url, "Veo extend")

        ops = data.get("operations", [])
        if not ops:
            raise RuntimeError(f"{LOG} Veo extend returned no operations: {data}")
        video_meta = ops[0].get("operation", {}).get("metadata", {}).get("video", {})
        video_url = video_meta.get("fifeUrl", "")
        media_gen_id = video_meta.get("mediaGenerationId", "")
        if not video_url:
            raise RuntimeError(f"{LOG} Veo extend: no fifeUrl in response.")
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, media_gen_id)


# ── ComfyUI Registration ──────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "UseapiTokenFromEnv": UseapiTokenFromEnv,
    "UseapiVeoGenerate":  UseapiVeoGenerate,
    "UseapiVeoUpscale":   UseapiVeoUpscale,
    "UseapiVeoExtend":    UseapiVeoExtend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UseapiTokenFromEnv": "Useapi Token From Env",
    "UseapiVeoGenerate":  "Useapi Veo 3.1 Generate Video",
    "UseapiVeoUpscale":   "Useapi Veo Upscale Video",
    "UseapiVeoExtend":    "Useapi Veo Extend Video",
}
