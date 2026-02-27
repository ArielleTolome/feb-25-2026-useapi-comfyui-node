# ComfyUI-UseapiNet

![CI](https://github.com/ArielleTolome/feb-25-2026-useapi-comfyui-node/actions/workflows/ci.yml/badge.svg)

Generate images and videos using [Useapi.net](https://useapi.net)'s AI API proxy directly within ComfyUI workflows. Supports Google Flow (Imagen 4, Veo 3.1) and Runway (Gen-4, Gen-4 Turbo, Frames) via your existing subscriptions — no separate API keys needed, just your Useapi.net token.

---

## Installation

**1. Clone into your ComfyUI custom_nodes directory:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/ComfyUI-UseapiNet
```

**2. (Optional) Install opencv-python for the Load Video Frame node:**

```bash
pip install opencv-python
```

All other dependencies (`numpy`, `Pillow`, `torch`) are already provided by ComfyUI.

**3. Set your Useapi.net token:**

```bash
export USEAPI_TOKEN=your_token_here
```

Or wire it through the `UseapiTokenFromEnv` node in your workflow.

**4. Restart ComfyUI** — nodes appear under **Useapi.net/** in the node menu.

---

## Configuration

You can configure default values for common parameters by creating a `nodes_config.json` file in the same directory as `useapi_nodes.py`. This allows you to set defaults for model, aspect ratio, timeout, etc., without editing the code.

Example `nodes_config.json`:

```json
{
  "default_timeout": 600,
  "default_aspect_ratio": "16:9",
  "UseapiVeoGenerate": {
    "model": "veo-3.1-fast"
  }
}
```

- `default_timeout`: Sets the global timeout in seconds for API requests (default: 600).
- `default_aspect_ratio`: Sets the default aspect ratio for nodes that support it.
- Node-specific overrides: You can specify a node class name (e.g., `UseapiVeoGenerate`) and provide key-value pairs to override defaults for that specific node.

---

## Authentication

Get your API token at [https://useapi.net/docs/start-here/setup-useapi](https://useapi.net/docs/start-here/setup-useapi).

### Option A: Environment variable (recommended)

```bash
export USEAPI_TOKEN=your_token_here
```

Leave the `api_token` input on any node **empty** — it automatically falls back to `$USEAPI_TOKEN`.

### Option B: TokenFromEnv node

Add a `UseapiTokenFromEnv` node to your workflow. It reads the token from a named environment variable and passes it as a STRING output to wire into other nodes.

### Option C: Direct input

Type or paste your token directly into the `api_token` field on any node. Not recommended for shared workflows.

---

## Nodes Reference

### Useapi.net/Utils

#### `UseapiTokenFromEnv`
Load a Useapi.net API token from an environment variable.

| Input | Type | Description |
|-------|------|-------------|
| `env_var_name` | STRING | Name of the env var (default: `USEAPI_TOKEN`) |

| Output | Type | Description |
|--------|------|-------------|
| `api_token` | STRING | Token value |

---

#### `UseapiLoadVideoFrame`
Extract a specific frame from a video file as a ComfyUI IMAGE tensor.

> **Requires:** `pip install opencv-python`

| Input | Type | Description |
|-------|------|-------------|
| `video_path` | STRING | Local path to the video file |
| `frame_number` | INT | Zero-based frame index to extract |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Extracted frame as ComfyUI tensor |

---

#### `UseapiPreviewVideo`
Display video URL and local path metadata as text. Wire the output to a ShowText node.

| Input | Type | Description |
|-------|------|-------------|
| `video_url` | STRING | Remote URL of the video |
| `video_path` | STRING | Local cached path (optional) |

| Output | Type | Description |
|--------|------|-------------|
| `info` | STRING | URL, path, exists flag, file size |

---

### Useapi.net/Google Flow

#### `UseapiVeoGenerate`
Generate video using Google Veo 3.1. Server-side auto-poll (~60-180s). Timeout: 600s.

| Input | Type | Description |
|-------|------|-------------|
| `prompt` | STRING | Text description of the video |
| `model` | COMBO | `veo-3.1-fast`, `veo-3.1-quality`, `veo-3.1-fast-relaxed`, `veo-3`, `veo-2` |
| `aspect_ratio` | COMBO | `landscape` or `portrait` |
| `api_token` | STRING | Optional — falls back to `$USEAPI_TOKEN` |
| `email` | STRING | Google Flow account email (optional) |
| `count` | INT | Number of videos to generate (1-4) |
| `seed` | INT | Reproducibility seed (0 = random) |

| Output | Type | Description |
|--------|------|-------------|
| `video_url` | STRING | Remote URL of generated video |
| `video_path` | STRING | Local cached `.mp4` path |
| `media_generation_id` | STRING | ID for upscale/extend operations |

---

#### `UseapiVeoUpscale`
Upscale a Veo-generated video to 1080p or 4K using its `mediaGenerationId`.

| Input | Type | Description |
|-------|------|-------------|
| `media_generation_id` | STRING | From VeoGenerate or VeoExtend output |
| `resolution` | COMBO | `1080p` or `4K` |
| `api_token` | STRING | Optional |

| Output | Type | Description |
|--------|------|-------------|
| `video_url` | STRING | Upscaled video URL |
| `video_path` | STRING | Local cached path |

---

#### `UseapiVeoExtend`
Extend an existing Veo video with a continuation prompt.

| Input | Type | Description |
|-------|------|-------------|
| `media_generation_id` | STRING | From VeoGenerate output |
| `prompt` | STRING | Continuation prompt |
| `api_token` | STRING | Optional |

| Output | Type | Description |
|--------|------|-------------|
| `video_url` | STRING | Extended video URL |
| `video_path` | STRING | Local cached path |
| `media_generation_id` | STRING | ID for further operations |

---

#### `UseapiGoogleFlowGenerateImage`
Generate images using Imagen 4, Nano Banana, or Nano Banana Pro (~10-20s).

| Input | Type | Description |
|-------|------|-------------|
| `prompt` | STRING | Image description |
| `model` | COMBO | `imagen-4`, `nano-banana`, `nano-banana-pro` |
| `aspect_ratio` | COMBO | `landscape` or `portrait` |
| `api_token` | STRING | Optional |
| `email` | STRING | Google Flow account email (optional) |
| `count` | INT | Images to generate (1-4) |
| `seed` | INT | Reproducibility seed |
| `reference_1/2/3` | STRING | `mediaGenerationId` of reference images |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | First generated image as ComfyUI tensor |
| `image_url` | STRING | URL of first image |
| `media_generation_id` | STRING | ID of first image (for upscale/reference) |
| `all_urls` | STRING | JSON array of all generated image URLs |

---

#### `UseapiGoogleFlowUploadAsset`
Upload an image to Google Flow for use as a reference. Email is required.

| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | ComfyUI IMAGE tensor to upload |
| `email` | STRING | Google Flow account email (**required**) |
| `api_token` | STRING | Optional |

| Output | Type | Description |
|--------|------|-------------|
| `media_generation_id` | STRING | Reference ID for use as `reference_1/2/3` |

---

#### `UseapiGoogleFlowImageUpscale`
Upscale a `nano-banana-pro` generated image to 2K or 4K.

> **Note:** Only works with images from the `nano-banana-pro` model.

| Input | Type | Description |
|-------|------|-------------|
| `media_generation_id` | STRING | From GoogleFlowGenerateImage (nano-banana-pro only) |
| `resolution` | COMBO | `2k` or `4k` |
| `api_token` | STRING | Optional |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Upscaled image as ComfyUI tensor |
| `media_generation_id` | STRING | Pass-through for chaining |

---

### Useapi.net/Runway

#### `UseapiRunwayUploadAsset`
Upload an image to Runway for use in video or image generation.

| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | ComfyUI IMAGE tensor to upload |
| `api_token` | STRING | Optional |
| `email` | STRING | Runway account email (optional) |

| Output | Type | Description |
|--------|------|-------------|
| `asset_id` | STRING | Runway asset ID for other nodes |

---

#### `UseapiRunwayGenerate`
Generate video using Runway Gen-4, Gen-4 Turbo, or Gen-3 Turbo. Async: polls until complete.

If an `image` tensor is provided without an `asset_id`, auto-uploads the image first.

| Input | Type | Description |
|-------|------|-------------|
| `model` | COMBO | `gen4`, `gen4turbo`, `gen3turbo` |
| `text_prompt` | STRING | Video description |
| `api_token` | STRING | Optional |
| `image` | IMAGE | Optional — auto-uploaded as firstImage |
| `asset_id` | STRING | Optional — pre-uploaded Runway asset ID |
| `email` | STRING | Runway account email (optional) |
| `aspect_ratio` | COMBO | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` |
| `seconds` | COMBO | `5` or `10` |
| `seed` | INT | Reproducibility seed |
| `explore_mode` | BOOLEAN | Enable explore mode (default: true) |
| `max_jobs` | INT | Max concurrent jobs (1-10) |
| `poll_interval` | INT | Seconds between polls (5-60) |
| `max_wait` | INT | Max seconds to wait (60-1800) |

| Output | Type | Description |
|--------|------|-------------|
| `video_url` | STRING | Generated video URL |
| `video_path` | STRING | Local cached `.mp4` path |
| `task_id` | STRING | Runway task ID |

---

#### `UseapiRunwayVideoToVideo`
Transform or extend an existing video using Runway Gen-4 or Gen-3 Turbo.

| Input | Type | Description |
|-------|------|-------------|
| `video_asset_id` | STRING | Runway asset ID of the source video |
| `model` | COMBO | `gen4` or `gen3turbo` |
| `api_token` | STRING | Optional |
| `text_prompt` | STRING | Transformation prompt (optional) |
| `seconds` | COMBO | `5` or `10` |
| `seed` | INT | Reproducibility seed |
| `explore_mode` | BOOLEAN | Enable explore mode |
| `max_jobs` | INT | Max concurrent jobs |
| `poll_interval` | INT | Seconds between polls |
| `max_wait` | INT | Max seconds to wait |

| Output | Type | Description |
|--------|------|-------------|
| `video_url` | STRING | Generated video URL |
| `video_path` | STRING | Local cached path |
| `task_id` | STRING | Runway task ID |

---

#### `UseapiRunwayFramesGenerate`
Generate high-quality 1080p images using Runway Frames (~20-30s). Supports up to 3 reference images.

Reference images in the prompt with `@IMG_1`, `@IMG_2`, `@IMG_3`.

| Input | Type | Description |
|-------|------|-------------|
| `text_prompt` | STRING | Image description |
| `api_token` | STRING | Optional |
| `email` | STRING | Runway account email (optional) |
| `aspect_ratio` | COMBO | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` |
| `style` | STRING | Style modifier (optional) |
| `diversity` | INT | Diversity level 0-5 (default: 2) |
| `num_images` | COMBO | `1` or `4` |
| `seed` | INT | Reproducibility seed |
| `explore_mode` | BOOLEAN | Enable explore mode |
| `image_ref_1/2/3` | IMAGE | Optional reference images (auto-uploaded) |
| `poll_interval` | INT | Seconds between polls |
| `max_wait` | INT | Max seconds to wait |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | First generated image as ComfyUI tensor |
| `image_url` | STRING | URL of first image |
| `all_urls` | STRING | JSON array of all image URLs |
| `task_id` | STRING | Runway task ID |

---

#### `UseapiRunwayImageUpscaler`
Upscale or resize an image using Runway's free image upscaling service. Takes a URL as input.

| Input | Type | Description |
|-------|------|-------------|
| `image_url` | STRING | URL of the image to upscale |
| `width` | INT | Target width in pixels (64-8192) |
| `height` | INT | Target height in pixels (64-8192) |
| `api_token` | STRING | Optional |
| `email` | STRING | Runway account email (optional) |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Upscaled image as ComfyUI tensor |

---

## Example Workflows

### Chain 1: Imagen 4 → Veo 3.1 (Image-to-Video)

```
UseapiTokenFromEnv
    └─► api_token ──► UseapiGoogleFlowGenerateImage (imagen-4, landscape)
                           └─► media_generation_id ──► UseapiVeoGenerate (veo-3.1-fast)
                                                            └─► video_path ──► UseapiPreviewVideo
```

1. `UseapiTokenFromEnv` reads `$USEAPI_TOKEN`
2. `UseapiGoogleFlowGenerateImage` generates an Imagen 4 image
3. `UseapiVeoGenerate` uses the `media_generation_id` as a reference image for video generation
4. `UseapiPreviewVideo` displays the result URL and file info

### Chain 2: Runway Frames → Gen-4 (Image-to-Video)

```
UseapiTokenFromEnv
    └─► api_token ──► UseapiRunwayFramesGenerate (text_prompt, 1 image)
                           └─► image ──► UseapiRunwayGenerate (gen4, 10s)
                                             └─► video_path ──► UseapiLoadVideoFrame (frame 0)
                                                                     └─► image ──► (next generation)
```

1. `UseapiRunwayFramesGenerate` generates a high-quality Runway Frames image
2. `UseapiRunwayGenerate` animates it with Gen-4 (auto-uploads the image)
3. `UseapiLoadVideoFrame` extracts the final frame for chaining to the next generation

---

## Troubleshooting

### Rate limited (429)
```
[Useapi.net] Rate limited (429). Wait 5-10s or add more Useapi.net accounts.
```
You've hit the concurrent request limit for your Useapi.net subscription. Either wait and retry, or add additional accounts at [useapi.net](https://useapi.net).

### Token not set
```
[Useapi.net] API token not provided.
```
Export your token before launching ComfyUI: `export USEAPI_TOKEN=your_token_here`

### Timeout (408 or RuntimeError: timed out)
Video generation (especially Veo) can take 60-180 seconds. The default timeout is 600s. If you're consistently timing out, check Useapi.net service status.

### Service unavailable (503)
Retry in a moment — Useapi.net or the underlying AI service is temporarily unavailable.

### LoadVideoFrame requires opencv-python
```
[Useapi.net] UseapiLoadVideoFrame requires opencv-python.
```
Install it: `pip install opencv-python`

### Google Flow Image Upscale: no encodedImage
The upscale endpoint only works with images generated by `nano-banana-pro`. Images from `imagen-4` or `nano-banana` are not supported.
