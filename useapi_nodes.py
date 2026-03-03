"""ComfyUI-UseapiNet: Custom nodes for Useapi.net API integration.

Provides image and video generation via:
  - Google Flow: Imagen 4, Gemini (Nano Banana), Veo 3.1
  - Runway: Gen-4, Gen-4 Turbo, Gen-3 Turbo, Frames
"""
import os
import io
import json
import logging
import time
import zlib
import base64
import ipaddress
import socket
import shutil
import tempfile
import threading
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

try:
    import folder_paths
except ImportError:
    folder_paths = None

# ── Constants ────────────────────────────────────────────────────────────────
LOG = "[Useapi.net]"
logger = logging.getLogger("useapi_net")
BASE_URL = "https://api.useapi.net/v1"
VIDEO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "comfyui_useapi_videos")
_RUNWAY_STYLES = [
    "none", "vivid", "vivid-warm", "vivid-cool", "high-contrast",
    "high-contrast-warm", "high-contrast-cool", "bw", "bw-contrast",
    "muted-pastel", "dreamscape", "nordic-minimal", "light-anime",
    "dark-anime", "painted-anime", "3d-cartoon", "sketch", "low-angle",
    "in-motion", "terracotta",
]
# Images endpoint does not accept "none" as a style value
_RUNWAY_STYLES_IMAGES = [s for s in _RUNWAY_STYLES if s != "none"]

# ── Timeout constants (seconds) ───────────────────────────────────────────────
_TIMEOUT_POLL = 30
_TIMEOUT_SHORT = 60
_TIMEOUT_MED = 120
_TIMEOUT_LONG = 300
_TIMEOUT_XLONG = 600
_ESTIMATED_SECS_VEO_GEN = 150
_ESTIMATED_SECS_VEO_UPSCALE = 60
_ESTIMATED_SECS_VEO_EXTEND = 150
_ESTIMATED_SECS_GOOGLE_FLOW_IMAGE = 120
_ESTIMATED_SECS_VEO_GIF = 30
_ESTIMATED_SECS_VEO_CONCAT = 90
_RETRY_DELAY_RECAPTCHA = 2.0

_CONFIG = {}

def _load_config():
    """Load default values from nodes_config.json if it exists."""
    global _CONFIG
    config_path = os.path.join(os.path.dirname(__file__), "nodes_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _CONFIG = json.load(f)
            logger.info(f"{LOG} Loaded config from {config_path}")
        except Exception as e:
            logger.info(f"{LOG} Error loading nodes_config.json: {e}")

_load_config()

def _get_config_value(node: str, key: str, default):
    """Get a configuration value with fallback."""
    # Check node-specific config first
    if node in _CONFIG and key in _CONFIG[node]:
        return _CONFIG[node][key]
    # Check global config
    if key in _CONFIG:
        return _CONFIG[key]
    # Check global config with "default_" prefix
    default_key = f"default_{key}"
    if default_key in _CONFIG:
        return _CONFIG[default_key]
    return default

def _get_sorted_list(original_list: list, default_val: str) -> list:
    """Move the default value to the front of the list if present."""
    if default_val in original_list:
        new_list = [default_val] + [x for x in original_list if x != default_val]
        return new_list
    return original_list

# ── Shared Utilities ─────────────────────────────────────────────────────────

def _safe_url(url: str) -> str:
    """Strip query parameters from a URL for safe logging/display."""
    if not url:
        return url
    try:
        parsed = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))
    except Exception:
        return url


def _redact_token(text, token: str) -> str:
    """Redact the API token from text if present."""
    if not text or not token:
        return text
    if not isinstance(text, str):
        text = str(text)
    return text.replace(token, "[REDACTED]")


def _is_safe_path(path: str) -> bool:
    """Validate path safety to prevent traversal and SSRF.

    Allowed paths:
    1. Inside VIDEO_CACHE_DIR
    2. Inside ComfyUI input/output/temp directories (if folder_paths available)
    """
    # Block URLs (rudimentary SSRF check)
    if "://" in path:
        return False

    try:
        real_path = os.path.realpath(path)
    except Exception:
        return False

    allowed_roots = [os.path.realpath(VIDEO_CACHE_DIR)]

    if folder_paths is not None:
        try:
            allowed_roots.append(os.path.realpath(folder_paths.get_input_directory()))
            allowed_roots.append(os.path.realpath(folder_paths.get_output_directory()))
            allowed_roots.append(os.path.realpath(folder_paths.get_temp_directory()))
        except Exception:
            pass

    for root in allowed_roots:
        if os.path.commonpath([real_path, root]) == root:
            return True

    return False


def _get_token(api_token: str) -> str:
    """Return API token: use direct input, else USEAPI_TOKEN env var."""
    token = (api_token or "").strip()
    if not token:
        token = os.environ.get("USEAPI_TOKEN", "").strip()
    if not token:
        raise ValueError(
            f"{LOG} API token is missing. Please provide it via the 'api_token' input "
            "or set the 'USEAPI_TOKEN' environment variable. Ensure it has no extra whitespace."
        )
    return token


def _auth_headers(token: str) -> dict:
    """Return JSON auth headers for Useapi.net requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _validate_url(url: str):
    """Validate outbound URL to reduce SSRF risk."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"{LOG} Invalid URL scheme '{parsed.scheme}'. Only http/https are allowed.")
    host = parsed.hostname
    if not host:
        raise ValueError(f"{LOG} Invalid URL '{url}'. Missing hostname.")

    host_l = host.lower()
    if host_l == "localhost" or host_l.endswith(".localhost"):
        raise ValueError(f"{LOG} Unsafe URL host '{host}'.")

    def _is_disallowed(ip_obj):
        return (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        )

    # Literal IP host
    try:
        ip_obj = ipaddress.ip_address(host_l)
        if _is_disallowed(ip_obj):
            raise ValueError(f"{LOG} Unsafe URL host '{host}'.")
        return
    except ValueError:
        pass

    # Resolve DNS host and block any private/local target
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"{LOG} Unable to resolve host '{host}': {e}") from e
    if not infos:
        raise ValueError(f"{LOG} Unable to resolve host '{host}'.")
    for info in infos:
        resolved = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(resolved)
        except ValueError:
            # Ignore non-IP entries returned by resolver.
            continue
        if _is_disallowed(ip_obj):
            raise ValueError(f"{LOG} Unsafe URL host '{host}' resolved to '{resolved}'.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects to non-public hosts/schemes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _make_request(url: str, method: str = "GET", headers: dict = None,
                  data: bytes = None, timeout: int = None, retries: int = 1):
    """Make an HTTP request with retry for stale connections. Returns (status_code, response_body_bytes)."""
    _validate_url(url)
    if timeout is None:
        timeout = _get_config_value("global", "default_timeout", 600)
    merged = {**_DEFAULT_HEADERS, **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=merged, method=method)
    opener = urllib.request.build_opener(_SafeRedirectHandler())

    for attempt in range(retries + 1):
        try:
            with opener.open(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except urllib.error.URLError as e:
            is_stale = isinstance(e.reason, (ConnectionResetError, BrokenPipeError)) or (
                isinstance(e.reason, OSError) and "RemoteDisconnected" in str(e.reason)
            )
            if attempt < retries and is_stale:
                continue
            raise RuntimeError(
                f"{LOG} Network error reaching {_safe_url(url)}. "
                f"Please check your internet connection. Detail: {e.reason}"
            )
        except TimeoutError:
            raise RuntimeError(
                f"{LOG} Request timed out after {timeout}s while connecting to {_safe_url(url)}. "
                "Try increasing 'default_timeout' in nodes_config.json."
            )

    raise RuntimeError(f"{LOG} Failed to reach {_safe_url(url)} after {retries + 1} attempts.")


def _check_status(status: int, body: bytes, url: str, context: str = "", token: str = "") -> dict:
    """Parse JSON response body; raise descriptive error if status != 200."""
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {}
    if status == 200:
        return data

    detail_raw = data.get("error", body[:300].decode(errors="replace"))
    detail_str = detail_raw if isinstance(detail_raw, str) else json.dumps(detail_raw)
    detail = _redact_token(detail_str, token)
    label = f"{LOG} [{context}]" if context else LOG
    safe_url = _safe_url(url)

    if status == 400:
        raise RuntimeError(
            f"{label} Bad Request (400). Please check your input parameters "
            f"(prompts, models, aspect ratios). URL: {safe_url}\nDetail: {detail}"
        )
    if status == 401:
        raise RuntimeError(
            f"{label} Unauthorized (401). Your API token is invalid or expired. "
            f"Please verify it has no extra whitespace/newlines. URL: {safe_url}"
        )
    if status == 403:
        _err_obj = data.get("error") if isinstance(data.get("error"), dict) else {}
        msg = data.get("message", "") or _err_obj.get("message", "")
        raw_text = body[:500].decode(errors="replace") if body else ""
        if "reCAPTCHA" in (msg + raw_text) or "captcha" in (msg + raw_text).lower():
            raise RuntimeError(
                f"{label} Transient reCAPTCHA error (403). This is an intermittent Google-side "
                f"limit. Retry in a few seconds. URL: {safe_url}\nDetail: {detail}"
            )
        if detail == "API error: 403":
            raise RuntimeError(
                f"{label} Google returned 403 to Useapi.net (\"API error: 403\"). "
                "This is usually a transient Google-side block \u2014 retry in a few seconds. "
                "If it persists, verify that your Google/service account has access to this "
                f"model at useapi.net. URL: {safe_url}\n"
                f"Raw response: {_redact_token(raw_text[:300], token)}"
            )
        raise RuntimeError(
            f"{label} Forbidden (403). The Google/service account linked to your "
            "Useapi.net token may not have access to this API or model. "
            f"Verify your account settings at useapi.net. URL: {safe_url}\nDetail: {detail}"
        )
    if status == 404:
        raise RuntimeError(
            f"{label} Not Found (404). The requested resource or endpoint was not found. "
            f"URL: {safe_url}\nDetail: {detail}"
        )
    if status == 408:
        raise RuntimeError(
            f"{label} Request Timeout (408). The generation took too long. "
            f"Try increasing the timeout in nodes_config.json. URL: {safe_url}\nDetail: {detail}"
        )
    if status == 429:
        raise RuntimeError(
            f"{label} Rate Limited (429). You are sending too many requests. "
            f"Wait 5-10s or add more Useapi.net accounts. URL: {safe_url}\nDetail: {detail}"
        )
    if status in (500, 502, 503, 504):
        raise RuntimeError(
            f"{label} Server Error ({status}). The service is temporarily unavailable. "
            f"Please retry in a moment. URL: {safe_url}\nDetail: {detail}"
        )

    raise RuntimeError(f"{label} HTTP {status} from {safe_url}.\nDetail: {detail}")


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
    """Download URL to cache dir with hash filename. Returns local path."""
    _validate_url(url)
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    fname = f"{zlib.adler32(url.encode()):08x}" + ext
    dest = os.path.join(VIDEO_CACHE_DIR, fname)
    if os.path.exists(dest):
        logger.info(f"{LOG} Cache hit: {dest}")
        return dest
    logger.info(f"{LOG} Downloading to {dest} ...")
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(url, timeout=_TIMEOUT_MED) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    logger.info(f"{LOG} Downloaded {len(data):,} bytes \u2192 {dest}")
    return dest


def _save_bytes_to_cache(data: bytes, ext: str) -> str:
    """Save raw bytes to VIDEO_CACHE_DIR with hash filename. Returns local path."""
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    fname = f"{zlib.adler32(data):08x}" + ext
    dest = os.path.join(VIDEO_CACHE_DIR, fname)
    with open(dest, "wb") as f:
        f.write(data)
    logger.info(f"{LOG} Saved {len(data):,} bytes \u2192 {dest}")
    return dest


def _runway_poll(task_id: str, token: str,
                 poll_interval: int = 10, max_wait: int = 600,
                 poll_path: str = "runwayml/tasks",
                 pbar=None) -> list:
    """Poll Runway task until SUCCEEDED. Returns list of artifacts dicts."""
    poll_url = (
        f"{BASE_URL}/{poll_path}/{urllib.parse.quote(task_id, safe=':@-._~')}"
    )
    headers = _auth_headers(token)
    deadline = time.time() + max_wait
    _start = time.time()
    # Adaptive polling: start with shorter interval, ramp up to poll_interval
    current_interval = min(2.0, poll_interval)
    while time.time() < deadline:
        # Sleep in chunks to allow progress updates (responsiveness)
        sleep_end = time.time() + current_interval
        while time.time() < sleep_end:
            chunk = min(0.1, sleep_end - time.time())
            if chunk > 0:
                time.sleep(chunk)
            if pbar is not None:
                _elapsed = time.time() - _start
                pbar.update_absolute(min(int(_elapsed / max_wait * 95), 95), 100)

        status, raw = _make_request(poll_url, "GET", headers, None, _TIMEOUT_POLL)
        if status in [500, 502, 503, 504, 520, 522, 524]:
            logger.info(f"{LOG} Runway poll transient error {status}. Retrying...")
            continue
        data = _check_status(status, raw, poll_url, f"Runway poll {task_id[:30]}", token=token)
        task_status = data.get("status", "")
        logger.info(f"{LOG} Runway task {task_id[:30]}... → {task_status}")

        # Increase interval for next loop, capped at poll_interval
        current_interval = min(current_interval * 1.5, poll_interval)

        if pbar is not None:
            _elapsed = time.time() - _start
            pbar.update_absolute(min(int(_elapsed / max_wait * 95), 95), 100)
        if task_status == "SUCCEEDED":
            if pbar is not None:
                pbar.update_absolute(100, 100)
            artifacts = data.get("artifacts", [])
            if not artifacts:
                raise RuntimeError(
                    f"{LOG} Runway task SUCCEEDED but no artifacts in response: "
                    f"{_redact_token(json.dumps(data), token)}"
                )
            return artifacts
        if task_status == "THROTTLED":
            logger.info(f"{LOG} Runway task {task_id[:30]}... is THROTTLED. Continuing to poll...")
            continue
        if task_status in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"{LOG} Runway task ended with status '{task_status}'. "
                f"Please check your account limits or prompt. task_id={task_id}"
            )
    raise RuntimeError(
        f"{LOG} Runway task timed out after {max_wait}s. "
        f"Try increasing the 'max_wait' input value. task_id={task_id}"
    )


def _runway_frames_poll(task_id: str, token: str,
                        poll_interval: int = 5, max_wait: int = 120,
                        pbar=None) -> list:
    """Poll Runway Frames task via the shared tasks endpoint. Returns artifacts."""
    return _runway_poll(task_id, token, poll_interval=poll_interval,
                        max_wait=max_wait, poll_path="runwayml/tasks", pbar=pbar)


def _runway_upload_image(token: str, image_tensor: torch.Tensor,
                         email: str = "", name: str = "comfyui_upload") -> str:
    """Upload a ComfyUI IMAGE tensor to Runway as a raw-binary image asset. Returns assetId."""
    png_bytes = _tensor_to_png_bytes(image_tensor)
    params = {"name": name}
    if email.strip():
        params["email"] = email.strip()
    url = f"{BASE_URL}/runwayml/assets?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "image/png"}
    logger.info(f"{LOG} Runway: uploading image asset...")
    status, raw = _make_request(url, "POST", headers, png_bytes, timeout=_TIMEOUT_SHORT)
    data = _check_status(status, raw, url, "Runway upload asset", token=token)
    asset_id = data.get("assetId", "") or data.get("id", "")
    if not asset_id:
        raise RuntimeError(
            f"{LOG} Runway upload: no assetId in response: "
            f"{_redact_token(json.dumps(data), token)}"
        )
    logger.info(f"{LOG} Runway asset uploaded: {asset_id[:50]}...")
    return asset_id


def _extract_runway_task_id(data: dict) -> str:
    """Extract taskId from either task.taskId (wrapped) or top-level taskId."""
    return data.get("task", {}).get("taskId", "") or data.get("taskId", "")


def _make_pbar(total: int = 100):
    """Create a ComfyUI ProgressBar if available, otherwise return None."""
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def _start_progress_thread(pbar, estimated_secs: float):
    """Start a daemon thread that advances pbar based on elapsed time.

    Returns (thread, done_event). Caller must call done_event.set() when the
    blocking operation finishes, then join the thread.
    Progress is capped at 95% until the caller signals completion.
    """
    done = threading.Event()
    start = time.time()

    def _run():
        while not done.wait(0.5):
            elapsed = time.time() - start
            pct = min(int(elapsed / estimated_secs * 95), 95)
            pbar.update_absolute(pct, 100)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, done


def _send_json(url: str, body: dict, token: str, timeout: int, context: str,
               captcha_retry: int = 3) -> dict:
    """Send a JSON POST request with optional reCAPTCHA retry. Returns parsed response dict."""
    headers = _auth_headers(token)
    max_retries = max(0, int(captcha_retry))
    for attempt in range(max_retries + 1):
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=timeout)
        if attempt < max_retries:
            try:
                _resp = json.loads(raw) if raw else {}
                _err = _resp.get("error") if isinstance(_resp.get("error"), dict) else {}
                _rmsg = _resp.get("message", "") or _err.get("message", "")
                if status == 403 and ("reCAPTCHA" in _rmsg or "captcha" in _rmsg.lower()):
                    wait_time = _RETRY_DELAY_RECAPTCHA * (2 ** attempt)
                    logger.info(
                        f"{LOG} {context}: reCAPTCHA 403, retrying "
                        f"({attempt + 1}/{max_retries + 1}) in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                _err_str = str(_resp.get("error", ""))
                if status == 400 and "all operations failed" in _err_str.lower():
                    wait_time = 15.0 * (2 ** attempt)
                    logger.info(
                        f"{LOG} {context}: 'All operations failed' (400), retrying "
                        f"({attempt + 1}/{max_retries + 1}) in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
            except Exception:
                pass
        break
    return _check_status(status, raw, url, context, token=token)


def _submit_with_progress(url: str, body: dict, token: str, estimated_secs: float,
                          timeout: int, context: str, captcha_retry: int = 3) -> dict:
    """Submit JSON POST with progress bar and optional reCAPTCHA retry."""
    pbar = _make_pbar()
    _pt, _done = (None, None)
    if pbar is not None:
        _pt, _done = _start_progress_thread(pbar, estimated_secs)
    try:
        data = _send_json(url, body, token, timeout, context, captcha_retry=captcha_retry)
    finally:
        if _done is not None:
            _done.set()
            _pt.join(timeout=1)
    if pbar is not None:
        pbar.update_absolute(100, 100)
    return data


def _runway_submit_and_poll(url: str, body: dict, token: str, context: str,
                            poll_interval: int, max_wait: int,
                            timeout: int = _TIMEOUT_SHORT):
    """Submit Runway task, extract taskId, poll until completion. Returns (artifacts, task_id)."""
    data = _send_json(url, body, token, timeout, context)
    task_id = _extract_runway_task_id(data)
    if not task_id:
        raise RuntimeError(
            f"{LOG} {context}: 'taskId' missing in response. "
            f"Response: {_redact_token(json.dumps(data), token)}"
        )
    logger.info(f"{LOG} {context}: task created. taskId={task_id[:50]}...")
    pbar = _make_pbar()
    artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
    return artifacts, task_id


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
        logger.info(f"{LOG} Token loaded from env var '{env_var_name}'")
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
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "media_generation_id")

    @classmethod
    def INPUT_TYPES(cls):
        # Apply defaults from config
        default_model = _get_config_value("UseapiVeoGenerate", "model", "veo-3.1-fast")
        default_ar = _get_config_value("UseapiVeoGenerate", "aspect_ratio", "landscape")

        models = ["veo-3.1-fast", "veo-3.1-quality", "veo-3.1-fast-relaxed", "veo-3", "veo-2"]
        aspect_ratios = ["landscape", "portrait"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (_get_sorted_list(models, default_model),),
                "aspect_ratio": (_get_sorted_list(aspect_ratios, default_ar),),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "count": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "captcha_retry": ("INT", {"default": 3, "min": 1, "max": 10}),
                "start_image": ("STRING", {"default": ""}),
                "end_image": ("STRING", {"default": ""}),
                "reference_image_1": ("STRING", {"default": ""}),
                "reference_image_2": ("STRING", {"default": ""}),
                "reference_image_3": ("STRING", {"default": ""}),
            },
        }

    def execute(self, prompt: str, model: str, aspect_ratio: str,
                api_token: str = "", email: str = "", count: int = 1,
                seed: int = 0, captcha_retry: int = 3,
                start_image: str = "", end_image: str = "",
                reference_image_1: str = "", reference_image_2: str = "",
                reference_image_3: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/videos"
        body = {"prompt": prompt, "model": model, "aspectRatio": aspect_ratio,
                "count": count}
        if seed != 0:
            body["seed"] = seed & 0x7FFFFFFF
        if email.strip():
            body["email"] = email.strip()
        if start_image.strip():
            body["startImage"] = start_image.strip()
        if end_image.strip():
            body["endImage"] = end_image.strip()
        for i, ref in enumerate([reference_image_1, reference_image_2, reference_image_3], 1):
            if ref.strip():
                body[f"referenceImage_{i}"] = ref.strip()

        logger.info(f"{LOG} Veo Generate: model={model}, prompt='{prompt[:60]}...'")
        data = _submit_with_progress(
            url, body, token, _ESTIMATED_SECS_VEO_GEN, None, "Veo generate",
            captcha_retry=captcha_retry
        )

        media_list = data.get("media", [])
        if not media_list:
            # Fallback: legacy operations[] format
            ops = data.get("operations", [])
            if not ops:
                raise RuntimeError(
                    f"{LOG} Veo generate returned no media or operations: "
                    f"{_redact_token(json.dumps(data), token)}"
                )
            op = ops[0]
            op_status = op.get("status", "")
            if "FAIL" in op_status.upper():
                raise RuntimeError(
                    f"{LOG} Veo generation failed. Status={op_status}. "
                    f"op={_redact_token(json.dumps(op), token)}"
                )
            video_meta = op.get("operation", {}).get("metadata", {}).get("video", {})
            video_url = video_meta.get("fifeUrl", "")
            media_gen_id = video_meta.get("mediaGenerationId", "")
        else:
            media_item = media_list[0]
            media_status = (
                media_item.get("mediaMetadata", {})
                .get("mediaStatus", {})
                .get("mediaGenerationStatus", "")
            )
            if "FAIL" in media_status.upper():
                raise RuntimeError(
                    f"{LOG} Veo generation failed. Status={media_status}. "
                    f"media={_redact_token(json.dumps(media_item), token)}"
                )
            video_url = media_item.get("videoUrl", "")
            media_gen_id = media_item.get("mediaGenerationId", "")

        if not video_url:
            raise RuntimeError(
                f"{LOG} Veo generate: video URL missing in response. "
                f"The API might have changed. Detail: {_redact_token(json.dumps(data), token)}"
            )

        logger.info(f"{LOG} Veo Generate: complete. mediaGenerationId={media_gen_id[:50]}...")
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, media_gen_id)


# ── Node 3: Veo Upscale Video ─────────────────────────────────────────────────
class UseapiVeoUpscale:
    """Upscale a Veo-generated video to 1080p or 4K using its mediaGenerationId."""

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    OUTPUT_NODE = True
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
        logger.info(f"{LOG} Veo Upscale: {resolution}, mediaGenerationId={media_generation_id[:50]}...")
        data = _submit_with_progress(url, body, token, _ESTIMATED_SECS_VEO_UPSCALE, _TIMEOUT_XLONG, "Veo upscale")

        media_list = data.get("media", [])
        if not media_list:
            ops = data.get("operations", [])
            if not ops:
                raise RuntimeError(f"{LOG} Veo upscale returned no media or operations: {data}")
            video_url = ops[0].get("operation", {}).get("metadata", {}).get("video", {}).get("fifeUrl", "")
        else:
            video_url = media_list[0].get("videoUrl", "")
        if not video_url:
            raise RuntimeError(f"{LOG} Veo upscale: no video URL in response.")
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path)


# ── Node 4: Veo Extend Video ──────────────────────────────────────────────────
class UseapiVeoExtend:
    """Extend an existing Veo video with a continuation prompt."""

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    OUTPUT_NODE = True
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
                "model": (["veo-3.1-fast", "veo-3.1-quality", "veo-3.1-fast-relaxed"],),
                "count": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "captcha_retry": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    def execute(self, media_generation_id: str, prompt: str, api_token: str = "",
                model: str = "veo-3.1-fast", count: int = 1, seed: int = 0,
                captcha_retry: int = 3):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/videos/extend"
        body = {"mediaGenerationId": media_generation_id, "prompt": prompt,
                "model": model, "count": count}
        if seed != 0:
            body["seed"] = seed & 0x7FFFFFFF
        logger.info(f"{LOG} Veo Extend: mediaGenerationId={media_generation_id[:50]}...")
        data = _submit_with_progress(
            url, body, token, _ESTIMATED_SECS_VEO_EXTEND, _TIMEOUT_XLONG, "Veo extend",
            captcha_retry=captcha_retry
        )

        media_list = data.get("media", [])
        if not media_list:
            ops = data.get("operations", [])
            if not ops:
                raise RuntimeError(f"{LOG} Veo extend returned no media or operations: {data}")
            video_meta = ops[0].get("operation", {}).get("metadata", {}).get("video", {})
            video_url = video_meta.get("fifeUrl", "")
            media_gen_id = video_meta.get("mediaGenerationId", "")
        else:
            video_url = media_list[0].get("videoUrl", "")
            media_gen_id = media_list[0].get("mediaGenerationId", "")
        if not video_url:
            raise RuntimeError(f"{LOG} Veo extend: no video URL in response.")
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, media_gen_id)


# ── Node 5: Google Flow Generate Image ───────────────────────────────────────
class UseapiGoogleFlowGenerateImage:
    """Generate images using Imagen 4, Nano Banana, or Nano Banana Pro.

    Server-side auto-poll: returns in ~10-20s. Timeout: 120s.
    Outputs first image as a ComfyUI IMAGE tensor; all URLs as JSON string.
    media_generation_id can feed into VeoGenerate or be used as a reference image.
    """

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "media_generation_id", "all_urls")

    @classmethod
    def INPUT_TYPES(cls):
        default_model = _get_config_value("UseapiGoogleFlowGenerateImage", "model", "imagen-4")
        default_ar = _get_config_value("UseapiGoogleFlowGenerateImage", "aspect_ratio", "landscape")

        models = ["imagen-4", "nano-banana", "nano-banana-pro", "nano-banana-2"]
        aspect_ratios = ["landscape", "portrait"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (_get_sorted_list(models, default_model),),
                "aspect_ratio": (_get_sorted_list(aspect_ratios, default_ar),),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "count": ("INT", {"default": 4, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "captcha_retry": ("INT", {"default": 3, "min": 1, "max": 10}),
                "reference_1": ("STRING", {"default": ""}),
                "reference_2": ("STRING", {"default": ""}),
                "reference_3": ("STRING", {"default": ""}),
                "reference_4": ("STRING", {"default": ""}),
                "reference_5": ("STRING", {"default": ""}),
                "reference_6": ("STRING", {"default": ""}),
            },
        }

    def execute(self, prompt: str, model: str, aspect_ratio: str,
                api_token: str = "", email: str = "", count: int = 4,
                seed: int = 0, captcha_retry: int = 3,
                reference_1: str = "", reference_2: str = "", reference_3: str = "",
                reference_4: str = "", reference_5: str = "", reference_6: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/images"
        body = {
            "prompt": prompt,
            "model": model,
            "aspectRatio": aspect_ratio,
            "count": count,
        }
        if seed != 0:
            body["seed"] = seed & 0x7FFFFFFF
        if email.strip():
            body["email"] = email.strip()
        refs = [reference_1, reference_2, reference_3, reference_4, reference_5, reference_6]
        for i, ref in enumerate(refs, start=1):
            if ref.strip():
                body[f"reference_{i}"] = ref.strip()

        logger.info(f"{LOG} Google Flow Image: model={model}, count={count}, prompt='{prompt[:60]}'")
        data = _submit_with_progress(
            url, body, token, _ESTIMATED_SECS_GOOGLE_FLOW_IMAGE, _TIMEOUT_MED,
            "Google Flow generate image", captcha_retry=captcha_retry
        )

        media_list = data.get("media", [])
        if not media_list:
            raise RuntimeError(f"{LOG} Google Flow image: no media in response: {data}")

        urls = []
        first_media_gen_id = ""
        for i, m in enumerate(media_list):
            gen_img = m.get("image", {}).get("generatedImage", {})
            fife_url = gen_img.get("fifeUrl", "")
            if fife_url:
                urls.append(fife_url)
            if i == 0:
                first_media_gen_id = gen_img.get("mediaGenerationId", "")

        if not urls:
            raise RuntimeError(f"{LOG} Google Flow image: no fifeUrls found in response: {data}")

        logger.info(f"{LOG} Google Flow Image: {len(urls)} image(s). mediaGenerationId={first_media_gen_id[:50]}...")

        # Download first image and convert to ComfyUI tensor
        s2, img_bytes = _make_request(urls[0], "GET", {}, None, 60)
        if s2 != 200:
            raise RuntimeError(f"{LOG} Failed to download image from {urls[0]} (HTTP {s2})")
        image_tensor = _bytes_to_tensor(img_bytes)

        return (image_tensor, urls[0], first_media_gen_id, json.dumps(urls))


# ── Node 6: Google Flow Upload Asset ─────────────────────────────────────────
class UseapiGoogleFlowUploadAsset:
    """Upload an image to Google Flow for use as a reference image.

    Returns mediaGenerationId usable as reference_1/2/3 in image generation,
    or as an image reference for Veo video generation.
    Email is required (path parameter in the API URL).
    """

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("media_generation_id",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "email": ("STRING", {"default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
            },
        }

    def execute(self, image: torch.Tensor, email: str, api_token: str = ""):
        token = _get_token(api_token)
        email_clean = email.strip()
        if not email_clean:
            raise ValueError(f"{LOG} Google Flow Upload Asset requires an email address.")

        url = f"{BASE_URL}/google-flow/assets/{urllib.parse.quote(email_clean, safe='')}"
        png_bytes = _tensor_to_png_bytes(image)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "image/png"}

        logger.info(f"{LOG} Google Flow Upload Asset: uploading for {email_clean}...")
        status, raw = _make_request(url, "POST", headers, png_bytes, timeout=_TIMEOUT_SHORT)
        data = _check_status(status, raw, url, "Google Flow upload asset", token=token)

        # Response: {"mediaGenerationId": {"mediaGenerationId": "user:..."}, ...}
        nested = data.get("mediaGenerationId", "")
        if isinstance(nested, dict):
            media_gen_id = nested.get("mediaGenerationId", "")
        else:
            media_gen_id = nested
        if not media_gen_id:
            raise RuntimeError(
                f"{LOG} Google Flow Upload Asset: no mediaGenerationId in response: {data}"
            )
        logger.info(f"{LOG} Google Flow Upload Asset: mediaGenerationId={media_gen_id[:50]}...")
        return (media_gen_id,)


# ── Node 7: Google Flow Image Upscale ────────────────────────────────────────
class UseapiGoogleFlowImageUpscale:
    """Upscale a nano-banana-pro generated image to 2K or 4K.

    IMPORTANT: Only works with images generated by the nano-banana-pro model.
    Response is base64-encoded image data (not a URL).
    """

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "media_generation_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_generation_id": ("STRING", {"default": ""}),
                "resolution": (["2k", "4k"],),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
            },
        }

    def execute(self, media_generation_id: str, resolution: str, api_token: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/images/upscale"
        body = {"mediaGenerationId": media_generation_id, "resolution": resolution}
        logger.info(f"{LOG} Google Flow Image Upscale: {resolution}, mediaGenerationId={media_generation_id[:50]}...")

        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=_TIMEOUT_LONG)
        data = _check_status(status, raw, url, "Google Flow image upscale", token=token)

        encoded = data.get("encodedImage", "")
        if not encoded:
            raise RuntimeError(
                f"{LOG} Google Flow Image Upscale: 'encodedImage' missing in response. "
                "Note: upscaling is only supported for 'nano-banana-pro' generated images."
            )
        img_bytes = base64.b64decode(encoded)
        tensor = _bytes_to_tensor(img_bytes)
        logger.info(f"{LOG} Google Flow Image Upscale: complete. Shape={tensor.shape}")
        return (tensor, media_generation_id)


# ── Node 8: Runway Upload Asset ───────────────────────────────────────────────
class UseapiRunwayUploadAsset:
    """Upload an image to Runway for use in video generation.

    Returns assetId needed by RunwayGenerate (firstImage_assetId)
    or RunwayFramesGenerate (imageAssetId1/2/3).
    """

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asset_id",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
            },
        }

    def execute(self, image: torch.Tensor, api_token: str = "", email: str = ""):
        token = _get_token(api_token)
        asset_id = _runway_upload_image(token, image, email)
        return (asset_id,)


# ── Node 9: Runway Generate Video ────────────────────────────────────────────
class UseapiRunwayGenerate:
    """Generate video using Runway Gen-4, Gen-4 Turbo, or Gen-3 Turbo.

    Async: creates a task, polls until SUCCEEDED.
    If image input provided without asset_id, auto-uploads the image first.
    Gen-4/Gen-4 Turbo require firstImage_assetId. Gen-3 Turbo supports text-to-video.
    """

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        default_model = _get_config_value("UseapiRunwayGenerate", "model", "gen4_5")
        default_ar = _get_config_value("UseapiRunwayGenerate", "aspect_ratio", "16:9")
        default_secs = str(_get_config_value("UseapiRunwayGenerate", "seconds", "10"))

        models = ["gen4_5", "gen4", "gen4turbo", "gen3turbo"]
        aspect_ratios = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
        seconds = ["5", "8", "10"]

        return {
            "required": {
                "model": (_get_sorted_list(models, default_model),),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "asset_id": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "aspect_ratio": (_get_sorted_list(aspect_ratios, default_ar),),
                "seconds": (_get_sorted_list(seconds, default_secs),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 1800}),
            },
        }

    def execute(self, model: str, text_prompt: str,
                api_token: str = "", image=None, asset_id: str = "",
                email: str = "", aspect_ratio: str = "16:9",
                seconds: str = "10", seed: int = 0,
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 600):
        token = _get_token(api_token)

        # Auto-upload image if provided without asset_id
        final_asset_id = asset_id.strip()
        if image is not None and not final_asset_id:
            logger.info(f"{LOG} Runway Generate: auto-uploading image...")
            final_asset_id = _runway_upload_image(token, image, email)

        url = f"{BASE_URL}/runwayml/{model}/create"
        body = {
            "text_prompt": text_prompt,
            "aspect_ratio": aspect_ratio,
            "seconds": int(seconds),
            "exploreMode": explore_mode,
            "maxJobs": max_jobs,
        }
        if final_asset_id:
            body["firstImage_assetId"] = final_asset_id
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()

        logger.info(f"{LOG} Runway Generate: model={model}, {seconds}s, prompt='{text_prompt[:60]}'")
        artifacts, task_id = _runway_submit_and_poll(url, body, token, f"Runway {model} create", poll_interval, max_wait)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 10: Runway Video-to-Video ────────────────────────────────────────────
class UseapiRunwayVideoToVideo:
    """Transform or extend an existing video using Runway Gen-4 or Gen-3 Turbo.

    Requires a video assetId (upload a video file to Runway first via the API).
    """

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_asset_id": ("STRING", {"default": ""}),
                "model": (["gen4", "gen3turbo"],),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seconds": (["5", "10"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 1800}),
            },
        }

    def execute(self, video_asset_id: str, model: str,
                api_token: str = "", text_prompt: str = "",
                seconds: str = "10", seed: int = 0,
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 600):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/{model}/video"
        # gen4 uses video_assetId; gen3turbo uses assetId
        asset_key = "video_assetId" if model == "gen4" else "assetId"
        body = {
            asset_key: video_asset_id,
            "text_prompt": text_prompt,
            "exploreMode": explore_mode,
            "maxJobs": max_jobs,
        }
        if model != "gen4":
            body["seconds"] = int(seconds)
        if seed != 0:
            body["seed"] = seed

        logger.info(f"{LOG} Runway Video-to-Video: model={model}, assetId={video_asset_id[:50]}...")
        artifacts, task_id = _runway_submit_and_poll(url, body, token, f"Runway {model} video-to-video create", poll_interval, max_wait)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 11: Runway Frames Generate Image ────────────────────────────────────
class UseapiRunwayFramesGenerate:
    """Generate high-quality images using Runway Frames (1080p, ~20-30s).

    Supports up to 3 reference images (image_ref_1/2/3 auto-uploaded).
    Reference images in prompt as @IMG_1, @IMG_2, @IMG_3.
    Async: creates task, polls until SUCCEEDED.
    """

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "all_urls", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],),
                "style": ("STRING", {"default": ""}),
                "diversity": ("INT", {"default": 2, "min": 0, "max": 5}),
                "num_images": (["1", "4"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "image_ref_1": ("IMAGE",),
                "image_ref_2": ("IMAGE",),
                "image_ref_3": ("IMAGE",),
                "poll_interval": ("INT", {"default": 5, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 120, "min": 60, "max": 600}),
            },
        }

    def execute(self, text_prompt: str, api_token: str = "", email: str = "",
                aspect_ratio: str = "16:9", style: str = "", diversity: int = 2,
                num_images: str = "4", seed: int = 0, explore_mode: bool = True,
                image_ref_1=None, image_ref_2=None, image_ref_3=None,
                poll_interval: int = 5, max_wait: int = 120):
        token = _get_token(api_token)

        # Auto-upload reference images
        asset_ids = []
        for i, ref_img in enumerate([image_ref_1, image_ref_2, image_ref_3], start=1):
            if ref_img is not None:
                logger.info(f"{LOG} Runway Frames: uploading image_ref_{i}...")
                aid = _runway_upload_image(token, ref_img, email)
                asset_ids.append((i, aid))

        url = f"{BASE_URL}/runwayml/frames/create"
        body = {
            "text_prompt": text_prompt,
            "aspect_ratio": aspect_ratio,
            "diversity": diversity,
            "num_images": int(num_images),
            "exploreMode": explore_mode,
        }
        if style.strip():
            body["style"] = style.strip()
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()
        for i, aid in asset_ids:
            body[f"imageAssetId{i}"] = aid

        logger.info(f"{LOG} Runway Frames: num_images={num_images}, prompt='{text_prompt[:60]}'")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway Frames create", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Frames: 'taskId' missing in response. "
                f"Response: {_redact_token(str(data), token)}"
            )
        logger.info(f"{LOG} Runway Frames: task created. taskId={task_id[:50]}...")

        pbar = _make_pbar()
        artifacts = _runway_frames_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        all_urls = [a["url"] for a in artifacts if a.get("mediaType") == "image" or "url" in a]
        if not all_urls:
            raise RuntimeError(f"{LOG} Runway Frames: no image URLs in artifacts: {artifacts}")

        first_url = all_urls[0]
        s2, img_bytes = _make_request(first_url, "GET", {}, None, 60)
        if s2 != 200:
            raise RuntimeError(f"{LOG} Runway Frames: failed to download image (HTTP {s2})")
        image_tensor = _bytes_to_tensor(img_bytes)
        return (image_tensor, first_url, json.dumps(all_urls), task_id)


# ── Node 12: Runway Image Upscaler ───────────────────────────────────────────
class UseapiRunwayImageUpscaler:
    """Upscale or resize an image using Runway's free image upscaling service.

    Works with free Runway accounts. Takes a URL, returns binary image data.
    NOTE: Response is raw binary (not JSON) — special handling required.
    """

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 2048, "min": 64, "max": 8192}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
            },
        }

    def execute(self, image_url: str, width: int, height: int,
                api_token: str = "", email: str = ""):
        token = _get_token(api_token)
        params = {"image_url": image_url, "width": width, "height": height}
        if email.strip():
            params["email"] = email.strip()
        qs = urllib.parse.urlencode(params)
        url = f"{BASE_URL}/runwayml/image_upscaler/?{qs}"

        logger.info(f"{LOG} Runway Image Upscaler: {width}x{height} for {image_url[:60]}...")
        headers = {"Authorization": f"Bearer {token}"}
        # NOTE: response is raw binary image, NOT JSON — bypass _check_status
        status, raw = _make_request(url, "GET", headers, None, timeout=120)
        if status != 200:
            raise RuntimeError(
                f"{LOG} Runway Image Upscaler: HTTP {status}. "
                f"Response: {raw[:200].decode(errors='replace')}"
            )
        tensor = _bytes_to_tensor(raw)
        logger.info(f"{LOG} Runway Image Upscaler: complete. Shape={tensor.shape}")
        return (tensor,)


# ── Node 13: Load Video Frame ─────────────────────────────────────────────────
class UseapiLoadVideoFrame:
    """Extract a specific frame from a video file as a ComfyUI IMAGE tensor.

    Requires opencv-python: pip install opencv-python
    Enables chaining: generate video → extract frame → use in next generation.
    """

    CATEGORY = "Useapi.net/Utils"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "frame_number": ("INT", {"default": 0, "min": 0, "max": 99999}),
            }
        }

    def execute(self, video_path: str, frame_number: int):
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                f"{LOG} UseapiLoadVideoFrame requires opencv-python. "
                "Install it: pip install opencv-python"
            )
        if not _is_safe_path(video_path):
            raise ValueError(f"{LOG} Security error: unsafe path or URL rejected: {video_path}")
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"{LOG} Cannot open video file: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raise RuntimeError(
                    f"{LOG} Could not read frame {frame_number} from {video_path}. "
                    f"Video has {total} frames."
                )
        finally:
            cap.release()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
        logger.info(f"{LOG} Load Video Frame: frame {frame_number} extracted. Shape={tensor.shape}")
        return (tensor,)


# ── Node 14: Preview Video ────────────────────────────────────────────────────
class UseapiPreviewVideo:
    """Display video URL and local path info as text.

    ComfyUI doesn't natively preview video. Wire the STRING output to a
    ShowText node or similar to display metadata during workflow development.
    """

    CATEGORY = "Useapi.net/Utils"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": ""}),
                "video_path": ("STRING", {"default": ""}),
            }
        }

    def execute(self, video_url: str, video_path: str):
        lines = [f"URL: {video_url}"]
        ui_videos = []

        if video_path and not _is_safe_path(video_path):
            raise ValueError(f"{LOG} Security error: unsafe path or URL rejected: {video_path}")

        if video_path and os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            lines.append(f"Path: {video_path}")
            lines.append(f"Size: {size_mb:.2f} MB")

            # Copy to ComfyUI output folder so the UI can display it
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
                filename = f"useapi_{os.path.basename(video_path)}"
                dest = os.path.join(output_dir, filename)
                shutil.copy2(video_path, dest)
                ui_videos.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "output",
                    "format": "video/mp4",
                })
                lines.append(f"Output: {dest}")
            except Exception as e:
                lines.append(f"Preview copy failed: {e}")
        elif video_path:
            lines.append(f"Path: {video_path} (not found)")

        info = "\n".join(lines)
        logger.info(f"{LOG} Preview Video:\n{info}")
        return {"ui": {"video": ui_videos}, "result": (info,)}


# ── Node 15: Veo Video to GIF ────────────────────────────────────────────────
class UseapiVeoVideoToGif:
    """Convert a Veo-generated video to an animated GIF via Google Flow."""

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gif_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_generation_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
            },
        }

    def execute(self, media_generation_id: str, api_token: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/videos/gif"
        body = {"mediaGenerationId": media_generation_id}
        logger.info(f"{LOG} Veo Video to GIF: mediaGenerationId={media_generation_id[:50]}...")
        data = _submit_with_progress(url, body, token, _ESTIMATED_SECS_VEO_GIF, _TIMEOUT_LONG, "Veo video to GIF")
        encoded = data.get("encodedGif", "")
        if not encoded:
            raise RuntimeError(f"{LOG} Veo Video to GIF: no encodedGif in response: {data}")
        gif_bytes = base64.b64decode(encoded)
        gif_path = _save_bytes_to_cache(gif_bytes, ".gif")
        logger.info(f"{LOG} Veo Video to GIF: complete. Path={gif_path}")
        return (gif_path,)


# ── Node 16: Veo Concatenate Videos ──────────────────────────────────────────
class UseapiVeoConcatenate:
    """Concatenate 2-10 Veo videos with optional per-clip trim controls."""

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(3, 11):
            optional[f"media_{i}"] = ("STRING", {"default": ""})
        for i in range(1, 11):
            optional[f"trim_start_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0})
            optional[f"trim_end_{i}"]   = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0})
        optional["api_token"] = ("STRING", {"default": ""})
        return {
            "required": {
                "media_1": ("STRING", {"default": ""}),
                "media_2": ("STRING", {"default": ""}),
            },
            "optional": optional,
        }

    def execute(self, media_1: str, media_2: str, **kwargs):
        token = _get_token(kwargs.get("api_token", ""))
        ids   = [media_1, media_2] + [kwargs.get(f"media_{i}", "") for i in range(3, 11)]
        trims = [(kwargs.get(f"trim_start_{i}", 0.0), kwargs.get(f"trim_end_{i}", 0.0))
                 for i in range(1, 11)]
        media_list = []
        for mgid, (ts, te) in zip(ids, trims):
            if mgid.strip():
                media_list.append({
                    "mediaGenerationId": mgid.strip(),
                    "trimStart": ts,
                    "trimEnd": te,
                })
        if len(media_list) < 2:
            raise ValueError(f"{LOG} Veo Concatenate: at least 2 mediaGenerationIds required.")
        url = f"{BASE_URL}/google-flow/videos/concatenate"
        body = {"media": media_list}
        logger.info(f"{LOG} Veo Concatenate: {len(media_list)} videos...")
        data = _submit_with_progress(url, body, token, _ESTIMATED_SECS_VEO_CONCAT, _TIMEOUT_XLONG, "Veo concatenate")
        encoded = data.get("encodedVideo", "")
        if not encoded:
            raise RuntimeError(f"{LOG} Veo Concatenate: no encodedVideo in response: {data}")
        video_bytes = base64.b64decode(encoded)
        video_path = _save_bytes_to_cache(video_bytes, ".mp4")
        logger.info(f"{LOG} Veo Concatenate: complete. Path={video_path}")
        return (video_path,)


# ── Node 17: Runway Generate Images ──────────────────────────────────────────
class UseapiRunwayImages:
    """Generate images using Runway nano-banana, nano-banana-pro, gen4, or gen4-turbo."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "all_urls", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["nano-banana", "nano-banana-pro", "nano-banana-2", "gen4", "gen4-turbo"],),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],),
                "resolution": (["720p", "1080p", "1K", "2K", "4K"],),
                "num_images": (["1", "4"],),
                "style": (_RUNWAY_STYLES_IMAGES,),
                "diversity": ("INT", {"default": 2, "min": 1, "max": 5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "image_asset_id_1": ("STRING", {"default": ""}),
                "image_asset_id_2": ("STRING", {"default": ""}),
                "image_asset_id_3": ("STRING", {"default": ""}),
                "image_asset_id_4": ("STRING", {"default": ""}),
                "image_asset_id_5": ("STRING", {"default": ""}),
                "image_asset_id_6": ("STRING", {"default": ""}),
                "poll_interval": ("INT", {"default": 5, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 120, "min": 30, "max": 600}),
            },
        }

    def execute(self, model: str, text_prompt: str,
                api_token: str = "", email: str = "",
                aspect_ratio: str = "16:9", resolution: str = "1080p",
                num_images: str = "1", style: str = "vivid",
                diversity: int = 2, seed: int = 0,
                image_asset_id_1: str = "", image_asset_id_2: str = "",
                image_asset_id_3: str = "", image_asset_id_4: str = "",
                image_asset_id_5: str = "", image_asset_id_6: str = "",
                poll_interval: int = 5, max_wait: int = 120):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/images/create"
        body = {
            "model": model,
            "text_prompt": text_prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "num_images": int(num_images),
            "diversity": diversity,
        }
        if style and style != "none":
            body["style"] = style
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()
        for i, aid in enumerate([image_asset_id_1, image_asset_id_2, image_asset_id_3, image_asset_id_4, image_asset_id_5, image_asset_id_6], 1):
            if aid.strip():
                body[f"imageAssetId{i}"] = aid.strip()

        logger.info(f"{LOG} Runway Images: model={model}, prompt='{text_prompt[:60]}'")
        artifacts, task_id = _runway_submit_and_poll(url, body, token, "Runway images create", poll_interval, max_wait)
        all_urls = [a["url"] for a in artifacts if "url" in a]
        if not all_urls:
            raise RuntimeError(f"{LOG} Runway Images: no URLs in artifacts: {artifacts}")

        first_url = all_urls[0]
        s2, img_bytes = _make_request(first_url, "GET", {}, None, 60)
        if s2 != 200:
            raise RuntimeError(f"{LOG} Runway Images: failed to download image (HTTP {s2})")
        image_tensor = _bytes_to_tensor(img_bytes)
        return (image_tensor, first_url, json.dumps(all_urls), task_id)


# ── Node 18: Runway Gen4 Upscale ──────────────────────────────────────────────
class UseapiRunwayGen4Upscale:
    """Upscale a Runway Gen4 or Gen4 Turbo video asset to higher resolution."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 300, "min": 60, "max": 1800}),
            },
        }

    def execute(self, asset_id: str, api_token: str = "", email: str = "",
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 300):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/gen4/upscale"
        body = {"assetId": asset_id, "exploreMode": explore_mode, "maxJobs": max_jobs}
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Runway Gen4 Upscale: assetId={asset_id[:50]}...")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway gen4 upscale", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Gen4 Upscale: 'taskId' missing in response. "
                f"Response: {data}"
            )

        pbar = _make_pbar()
        artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 19: Runway Act Two ────────────────────────────────────────────────────
class UseapiRunwayActTwo:
    """Transfer motion from a driving video to a character using Runway Gen4 Act Two."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "driving_asset_id": ("STRING", {"default": ""}),
                "character_asset_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],),
                "body_control": ("BOOLEAN", {"default": True}),
                "expression_intensity": ("INT", {"default": 3, "min": 1, "max": 5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 300, "min": 60, "max": 1800}),
            },
        }

    def execute(self, driving_asset_id: str, character_asset_id: str,
                api_token: str = "", email: str = "",
                aspect_ratio: str = "16:9", body_control: bool = True,
                expression_intensity: int = 3, seed: int = 0,
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 300):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/gen4/act-two"
        body = {
            "driving_assetId": driving_asset_id,
            "character_assetId": character_asset_id,
            "aspect_ratio": aspect_ratio,
            "body_control": body_control,
            "expression_intensity": expression_intensity,
            "exploreMode": explore_mode,
            "maxJobs": max_jobs,
        }
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()
        logger.info(
            f"{LOG} Runway Act Two: driving={driving_asset_id[:40]}..., "
            f"character={character_asset_id[:40]}..."
        )
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway act-two", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Act Two: 'taskId' missing in response. "
                f"Response: {data}"
            )

        pbar = _make_pbar()
        artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 20: Runway Act Two Voice ─────────────────────────────────────────────
class UseapiRunwayActTwoVoice:
    """Add a voice to a Runway Gen4 Act Two video using a voice ID."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_asset_id": ("STRING", {"default": ""}),
                "voice_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 300, "min": 60, "max": 1800}),
            },
        }

    def execute(self, video_asset_id: str, voice_id: str,
                api_token: str = "", email: str = "",
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 300):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/gen4/act-two-voice"
        body = {
            "video_assetId": video_asset_id,
            "voiceId": voice_id,
            "exploreMode": explore_mode,
            "maxJobs": max_jobs,
        }
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Runway Act Two Voice: video={video_asset_id[:40]}..., voiceId={voice_id}")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway act-two-voice", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Act Two Voice: 'taskId' missing in response. "
                f"Response: {data}"
            )

        pbar = _make_pbar()
        artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 21: Runway Lipsync ────────────────────────────────────────────────────
class UseapiRunwayLipsync:
    """Create a lipsync video using Runway with optional image, video, audio, or voice."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image_asset_id": ("STRING", {"default": ""}),
                "video_asset_id": ("STRING", {"default": ""}),
                "audio_asset_id": ("STRING", {"default": ""}),
                "voice_id": ("STRING", {"default": ""}),
                "voice_text": ("STRING", {"default": ""}),
                "model_id": (["eleven_multilingual_v1", "eleven_multilingual_v2"],),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 300, "min": 60, "max": 1800}),
            },
        }

    def execute(self, image_asset_id: str = "", video_asset_id: str = "",
                audio_asset_id: str = "", voice_id: str = "", voice_text: str = "",
                model_id: str = "eleven_multilingual_v2",
                api_token: str = "", email: str = "",
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 300):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/lipsync/create"
        body = {"exploreMode": explore_mode, "maxJobs": max_jobs, "model_id": model_id}
        if image_asset_id.strip():
            body["image_assetId"] = image_asset_id.strip()
        if video_asset_id.strip():
            body["video_assetId"] = video_asset_id.strip()
        if audio_asset_id.strip():
            body["audio_assetId"] = audio_asset_id.strip()
        if voice_id.strip():
            body["voiceId"] = voice_id.strip()
        if voice_text.strip():
            body["voice_text"] = voice_text.strip()
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Runway Lipsync: model_id={model_id}...")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway lipsync", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Lipsync: 'taskId' missing in response. "
                f"Response: {data}"
            )

        pbar = _make_pbar()
        artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 22: Runway Super Slow Motion ─────────────────────────────────────────
class UseapiRunwaySuperSlowMotion:
    """Create super slow-motion video from a Runway video asset."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_id": ("STRING", {"default": ""}),
                "speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 300, "min": 60, "max": 1800}),
            },
        }

    def execute(self, asset_id: str, speed: float,
                api_token: str = "", email: str = "",
                max_jobs: int = 5, poll_interval: int = 10, max_wait: int = 300):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/super_slow_motion"
        body = {"assetId": asset_id, "speed": speed, "maxJobs": max_jobs}
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Runway Super Slow Motion: assetId={asset_id[:50]}..., speed={speed}")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway super slow motion", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Super Slow Motion: 'taskId' missing in response. "
                f"Response: {data}"
            )

        pbar = _make_pbar()
        artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── Node 23: Runway Transcribe ─────────────────────────────────────────────────
class UseapiRunwayTranscribe:
    """Transcribe a Runway video or audio asset to text. Synchronous — no polling."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("full_text", "words_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_id": ("STRING", {"default": ""}),
                "language": (["en", "en_us", "en_uk", "en_au", "es", "fr", "de", "it", "pt", "nl"],),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
            },
        }

    def execute(self, asset_id: str, language: str, api_token: str = ""):
        token = _get_token(api_token)
        params = {"assetId": asset_id, "language": language}
        url = f"{BASE_URL}/runwayml/transcribe?{urllib.parse.urlencode(params)}"
        logger.info(f"{LOG} Runway Transcribe: assetId={asset_id[:50]}..., language={language}")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "GET", headers, None, timeout=120)
        data = _check_status(status, raw, url, "Runway transcribe", token=token)
        words = data.get("words", [])
        full_text = " ".join(w.get("text", "") for w in words)
        words_json = json.dumps(words)
        logger.info(f"{LOG} Runway Transcribe: {len(words)} words transcribed.")
        return (full_text, words_json)


# ── Node 24: Runway Gen3 Turbo Extend ─────────────────────────────────────────
class UseapiRunwayGen3TurboExtend:
    """Extend a Runway Gen3 Turbo video with an optional continuation prompt."""

    CATEGORY = "Useapi.net/Runway"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "explore_mode": ("BOOLEAN", {"default": True}),
                "max_jobs": ("INT", {"default": 5, "min": 1, "max": 10}),
                "poll_interval": ("INT", {"default": 10, "min": 5, "max": 60}),
                "max_wait": ("INT", {"default": 300, "min": 60, "max": 1800}),
            },
        }

    def execute(self, asset_id: str, text_prompt: str = "", seed: int = 0,
                api_token: str = "", email: str = "",
                explore_mode: bool = True, max_jobs: int = 5,
                poll_interval: int = 10, max_wait: int = 300):
        token = _get_token(api_token)
        url = f"{BASE_URL}/runwayml/gen3turbo/extend"
        body = {"assetId": asset_id, "exploreMode": explore_mode, "maxJobs": max_jobs}
        if text_prompt.strip():
            body["text_prompt"] = text_prompt.strip()
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()
        logger.info(f"{LOG} Runway Gen3 Turbo Extend: assetId={asset_id[:50]}...")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway gen3turbo extend", token=token)

        task_id = _extract_runway_task_id(data)
        if not task_id:
            raise RuntimeError(
                f"{LOG} Runway Gen3 Turbo Extend: 'taskId' missing in response. "
                f"Response: {data}"
            )

        pbar = _make_pbar()
        artifacts = _runway_poll(task_id, token, poll_interval, max_wait, pbar=pbar)
        video_url = artifacts[0]["url"]
        video_path = _download_file(video_url, ".mp4")
        return (video_url, video_path, task_id)


# ── ComfyUI Registration ──────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "UseapiTokenFromEnv":             UseapiTokenFromEnv,
    "UseapiVeoGenerate":              UseapiVeoGenerate,
    "UseapiVeoUpscale":               UseapiVeoUpscale,
    "UseapiVeoExtend":                UseapiVeoExtend,
    "UseapiGoogleFlowGenerateImage":  UseapiGoogleFlowGenerateImage,
    "UseapiGoogleFlowUploadAsset":    UseapiGoogleFlowUploadAsset,
    "UseapiGoogleFlowImageUpscale":   UseapiGoogleFlowImageUpscale,
    "UseapiRunwayUploadAsset":        UseapiRunwayUploadAsset,
    "UseapiRunwayGenerate":           UseapiRunwayGenerate,
    "UseapiRunwayVideoToVideo":       UseapiRunwayVideoToVideo,
    "UseapiRunwayFramesGenerate":     UseapiRunwayFramesGenerate,
    "UseapiRunwayImageUpscaler":      UseapiRunwayImageUpscaler,
    "UseapiLoadVideoFrame":           UseapiLoadVideoFrame,
    "UseapiPreviewVideo":             UseapiPreviewVideo,
    "UseapiVeoVideoToGif":            UseapiVeoVideoToGif,
    "UseapiVeoConcatenate":           UseapiVeoConcatenate,
    "UseapiRunwayImages":             UseapiRunwayImages,
    "UseapiRunwayGen4Upscale":        UseapiRunwayGen4Upscale,
    "UseapiRunwayActTwo":             UseapiRunwayActTwo,
    "UseapiRunwayActTwoVoice":        UseapiRunwayActTwoVoice,
    "UseapiRunwayLipsync":            UseapiRunwayLipsync,
    "UseapiRunwaySuperSlowMotion":    UseapiRunwaySuperSlowMotion,
    "UseapiRunwayTranscribe":         UseapiRunwayTranscribe,
    "UseapiRunwayGen3TurboExtend":    UseapiRunwayGen3TurboExtend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UseapiTokenFromEnv":             "Useapi Token From Env",
    "UseapiVeoGenerate":              "Useapi Veo 3.1 Generate Video",
    "UseapiVeoUpscale":               "Useapi Veo Upscale Video",
    "UseapiVeoExtend":                "Useapi Veo Extend Video",
    "UseapiGoogleFlowGenerateImage":  "Useapi Google Flow Generate Image",
    "UseapiGoogleFlowUploadAsset":    "Useapi Google Flow Upload Asset",
    "UseapiGoogleFlowImageUpscale":   "Useapi Google Flow Image Upscale",
    "UseapiRunwayUploadAsset":        "Useapi Runway Upload Asset",
    "UseapiRunwayGenerate":           "Useapi Runway Generate Video",
    "UseapiRunwayVideoToVideo":       "Useapi Runway Video-to-Video",
    "UseapiRunwayFramesGenerate":     "Useapi Runway Frames Generate Image",
    "UseapiRunwayImageUpscaler":      "Useapi Runway Image Upscaler",
    "UseapiLoadVideoFrame":           "Useapi Load Video Frame",
    "UseapiPreviewVideo":             "Useapi Preview Video",
    "UseapiVeoVideoToGif":            "Useapi Veo Video to GIF",
    "UseapiVeoConcatenate":           "Useapi Veo Concatenate Videos",
    "UseapiRunwayImages":             "Useapi Runway Generate Images",
    "UseapiRunwayGen4Upscale":        "Useapi Runway Gen4 Upscale Video",
    "UseapiRunwayActTwo":             "Useapi Runway Act Two",
    "UseapiRunwayActTwoVoice":        "Useapi Runway Act Two Voice",
    "UseapiRunwayLipsync":            "Useapi Runway Lipsync",
    "UseapiRunwaySuperSlowMotion":    "Useapi Runway Super Slow Motion",
    "UseapiRunwayTranscribe":         "Useapi Runway Transcribe",
    "UseapiRunwayGen3TurboExtend":    "Useapi Runway Gen3 Turbo Extend",
}
