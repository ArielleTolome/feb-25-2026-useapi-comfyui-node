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


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _make_request(url: str, method: str = "GET", headers: dict = None,
                  data: bytes = None, timeout: int = 600):
    """Make an HTTP request. Returns (status_code, response_body_bytes)."""
    merged = {**_DEFAULT_HEADERS, **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=merged, method=method)
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
                 poll_interval: int = 10, max_wait: int = 600,
                 poll_path: str = "runwayml/tasks") -> list:
    """Poll Runway task until SUCCEEDED. Returns list of artifacts dicts."""
    poll_url = (
        f"{BASE_URL}/{poll_path}/{urllib.parse.quote(task_id, safe=':@-._~')}"
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


def _runway_frames_poll(task_id: str, token: str,
                        poll_interval: int = 5, max_wait: int = 120) -> list:
    """Poll Runway Frames task via the shared tasks endpoint. Returns artifacts."""
    return _runway_poll(task_id, token, poll_interval=poll_interval,
                        max_wait=max_wait, poll_path="runwayml/tasks")


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


# ── Node 5: Google Flow Generate Image ───────────────────────────────────────
class UseapiGoogleFlowGenerateImage:
    """Generate images using Imagen 4, Nano Banana, or Nano Banana Pro.

    Server-side auto-poll: returns in ~10-20s. Timeout: 120s.
    Outputs first image as a ComfyUI IMAGE tensor; all URLs as JSON string.
    media_generation_id can feed into VeoGenerate or be used as a reference image.
    """

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "media_generation_id", "all_urls")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["imagen-4", "nano-banana", "nano-banana-pro"],),
                "aspect_ratio": (["landscape", "portrait"],),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "count": ("INT", {"default": 4, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
                "reference_1": ("STRING", {"default": ""}),
                "reference_2": ("STRING", {"default": ""}),
                "reference_3": ("STRING", {"default": ""}),
            },
        }

    def execute(self, prompt: str, model: str, aspect_ratio: str,
                api_token: str = "", email: str = "", count: int = 4,
                seed: int = 0, reference_1: str = "",
                reference_2: str = "", reference_3: str = ""):
        token = _get_token(api_token)
        url = f"{BASE_URL}/google-flow/images"
        body = {
            "prompt": prompt,
            "model": model,
            "aspectRatio": aspect_ratio,
            "count": count,
        }
        if seed != 0:
            body["seed"] = seed
        if email.strip():
            body["email"] = email.strip()
        for i, ref in enumerate([reference_1, reference_2, reference_3], start=1):
            if ref.strip():
                body[f"reference_{i}"] = ref.strip()

        print(f"{LOG} Google Flow Image: model={model}, count={count}, prompt='{prompt[:60]}'")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=120)
        data = _check_status(status, raw, url, "Google Flow generate image")

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

        print(f"{LOG} Google Flow Image: {len(urls)} image(s). mediaGenerationId={first_media_gen_id[:50]}...")

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
        files = {"image": ("upload.png", png_bytes, "image/png")}
        body, ct = _build_multipart({}, files)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": ct}

        print(f"{LOG} Google Flow Upload Asset: uploading for {email_clean}...")
        status, raw = _make_request(url, "POST", headers, body, timeout=60)
        data = _check_status(status, raw, url, "Google Flow upload asset")

        # mediaGenerationId may be at root or nested under media[0].image.generatedImage
        media_gen_id = data.get("mediaGenerationId", "")
        if not media_gen_id:
            media_gen_id = (
                data.get("media", [{}])[0]
                .get("image", {})
                .get("generatedImage", {})
                .get("mediaGenerationId", "")
            )
        if not media_gen_id:
            raise RuntimeError(
                f"{LOG} Google Flow Upload Asset: no mediaGenerationId in response: {data}"
            )
        print(f"{LOG} Google Flow Upload Asset: mediaGenerationId={media_gen_id[:50]}...")
        return (media_gen_id,)


# ── Node 7: Google Flow Image Upscale ────────────────────────────────────────
class UseapiGoogleFlowImageUpscale:
    """Upscale a nano-banana-pro generated image to 2K or 4K.

    IMPORTANT: Only works with images generated by the nano-banana-pro model.
    Response is base64-encoded image data (not a URL).
    """

    CATEGORY = "Useapi.net/Google Flow"
    FUNCTION = "execute"
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
        print(f"{LOG} Google Flow Image Upscale: {resolution}, mediaGenerationId={media_generation_id[:50]}...")

        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=300)
        data = _check_status(status, raw, url, "Google Flow image upscale")

        encoded = data.get("encodedImage", "")
        if not encoded:
            raise RuntimeError(
                f"{LOG} Google Flow Image Upscale: no encodedImage in response. "
                "Note: upscaling only supports nano-banana-pro generated images."
            )
        img_bytes = base64.b64decode(encoded)
        tensor = _bytes_to_tensor(img_bytes)
        print(f"{LOG} Google Flow Image Upscale: complete. Shape={tensor.shape}")
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
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_path", "task_id")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["gen4", "gen4turbo", "gen3turbo"],),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "asset_id": ("STRING", {"default": ""}),
                "email": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],),
                "seconds": (["5", "10"],),
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
            print(f"{LOG} Runway Generate: auto-uploading image...")
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

        print(f"{LOG} Runway Generate: model={model}, {seconds}s, prompt='{text_prompt[:60]}'")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, f"Runway {model} create")

        task_id = data.get("task", {}).get("taskId", "")
        if not task_id:
            raise RuntimeError(f"{LOG} Runway create: no taskId in response: {data}")
        print(f"{LOG} Runway Generate: task created. taskId={task_id[:50]}...")

        artifacts = _runway_poll(task_id, token, poll_interval, max_wait)
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
        body = {
            "assetId": video_asset_id,
            "seconds": int(seconds),
            "exploreMode": explore_mode,
            "maxJobs": max_jobs,
        }
        if text_prompt.strip():
            body["text_prompt"] = text_prompt.strip()
        if seed != 0:
            body["seed"] = seed

        print(f"{LOG} Runway Video-to-Video: model={model}, assetId={video_asset_id[:50]}...")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, f"Runway {model} video-to-video create")

        task_id = data.get("task", {}).get("taskId", "")
        if not task_id:
            raise RuntimeError(f"{LOG} Runway video-to-video: no taskId in response: {data}")

        artifacts = _runway_poll(task_id, token, poll_interval, max_wait)
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
                print(f"{LOG} Runway Frames: uploading image_ref_{i}...")
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

        print(f"{LOG} Runway Frames: num_images={num_images}, prompt='{text_prompt[:60]}'")
        headers = _auth_headers(token)
        status, raw = _make_request(url, "POST", headers, json.dumps(body).encode(), timeout=60)
        data = _check_status(status, raw, url, "Runway Frames create")

        task_id = data.get("taskId", "") or data.get("task", {}).get("taskId", "")
        if not task_id:
            raise RuntimeError(f"{LOG} Runway Frames: no taskId in response: {data}")
        print(f"{LOG} Runway Frames: task created. taskId={task_id[:50]}...")

        artifacts = _runway_frames_poll(task_id, token, poll_interval, max_wait)
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

        print(f"{LOG} Runway Image Upscaler: {width}x{height} for {image_url[:60]}...")
        headers = {"Authorization": f"Bearer {token}"}
        # NOTE: response is raw binary image, NOT JSON — bypass _check_status
        status, raw = _make_request(url, "GET", headers, None, timeout=120)
        if status != 200:
            raise RuntimeError(
                f"{LOG} Runway Image Upscaler: HTTP {status}. "
                f"Response: {raw[:200].decode(errors='replace')}"
            )
        tensor = _bytes_to_tensor(raw)
        print(f"{LOG} Runway Image Upscaler: complete. Shape={tensor.shape}")
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
        print(f"{LOG} Load Video Frame: frame {frame_number} extracted. Shape={tensor.shape}")
        return (tensor,)


# ── Node 14: Preview Video ────────────────────────────────────────────────────
class UseapiPreviewVideo:
    """Display video URL and local path info as text.

    ComfyUI doesn't natively preview video. Wire the STRING output to a
    ShowText node or similar to display metadata during workflow development.
    """

    CATEGORY = "Useapi.net/Utils"
    FUNCTION = "execute"
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
        if video_path:
            exists = os.path.exists(video_path)
            lines.append(f"Path: {video_path}")
            lines.append(f"Exists: {exists}")
            if exists:
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                lines.append(f"Size: {size_mb:.2f} MB")
        info = "\n".join(lines)
        print(f"{LOG} Preview Video:\n{info}")
        return (info,)


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
}
