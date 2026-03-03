# ComfyUI-UseapiNet

![CI](https://github.com/ArielleTolome/feb-25-2026-useapi-comfyui-node/actions/workflows/ci.yml/badge.svg)

Generate images and videos using [Useapi.net](https://useapi.net)'s AI API proxy directly within ComfyUI workflows. Supports Google Flow (Imagen 4, Veo 3.1) and Runway (Gen-4.5, Gen-4 incl. Aleph, Gen-4 Turbo, Gen-3 Turbo, Frames) via your existing subscriptions — no separate API keys needed, just your Useapi.net token.

---

## Installation

### 1. Clone into your ComfyUI custom_nodes directory

**Windows / Linux / Mac:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/ComfyUI-UseapiNet
```

### 2. (Optional) Install opencv-python

Only required for the `UseapiLoadVideoFrame` node.

```bash
pip install opencv-python
```

All other dependencies (`numpy`, `Pillow`, `torch`) are already provided by ComfyUI.

### 3. Set your Useapi.net token

**Mac/Linux:**
```bash
export USEAPI_TOKEN=your_token_here
```

**Windows (PowerShell):**
```powershell
$env:USEAPI_TOKEN="your_token_here"
```

Or wire it through the `UseapiTokenFromEnv` node in your workflow.

### 4. Restart ComfyUI
Nodes appear under **Useapi.net/** in the node menu.

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

Set `USEAPI_TOKEN` in your OS environment. Leave the `api_token` input on any node **empty** — it automatically falls back to `$USEAPI_TOKEN`.

### Option B: TokenFromEnv node

Add a `UseapiTokenFromEnv` node to your workflow. It reads the token from a named environment variable and passes it as a STRING output to wire into other nodes.

### Option C: Direct input

Type or paste your token directly into the `api_token` field on any node. Not recommended for shared workflows.

---

## Nodes Reference

### Useapi.net/Utils

#### `UseapiTokenFromEnv`
Load a Useapi.net API token from an environment variable.
- **Input**: `env_var_name` (default: USEAPI_TOKEN)
- **Output**: `api_token` (STRING)

#### `UseapiLoadVideoFrame`
Extract a specific frame from a video file as a ComfyUI IMAGE tensor.
- **Input**: `video_path` (STRING), `frame_number` (INT)
- **Output**: `image` (IMAGE)

#### `UseapiPreviewVideo`
Display video URL and local path metadata as text. Wire the output to a ShowText node.
- **Input**: `video_url` (STRING), `video_path` (STRING)
- **Output**: `info` (STRING)

---

### Useapi.net/Google Flow

#### `UseapiVeoGenerate`
Generate video using Google Veo 3.1. Server-side auto-poll.
- **Input**: `prompt`, `model`, `aspect_ratio`, `count`, `seed`
- **Output**: `video_url`, `video_path`, `media_generation_id`

#### `UseapiVeoUpscale`
Upscale a Veo-generated video to 1080p or 4K.
- **Input**: `media_generation_id`, `resolution` (1080p, 4K)
- **Output**: `video_url`, `video_path`

#### `UseapiVeoExtend`
Extend an existing Veo video with a continuation prompt.
- **Input**: `media_generation_id`, `prompt`, `model`
- **Output**: `video_url`, `video_path`, `media_generation_id`

#### `UseapiVeoVideoToGif`
Convert a Veo-generated video to an animated GIF.
- **Input**: `media_generation_id`
- **Output**: `gif_path`

#### `UseapiVeoConcatenate`
Concatenate 2-5 Veo videos with optional trim controls.
- **Input**: `media_1`..`media_5`, `trim_start_1`..`trim_end_5`
- **Output**: `video_path`

#### `UseapiGoogleFlowGenerateImage`
Generate images using Imagen 4, Nano Banana, or Nano Banana Pro.
- **Input**: `prompt`, `model`, `aspect_ratio`, `count`, `reference_1/2/3`
- **Output**: `image`, `image_url`, `media_generation_id`, `all_urls`

#### `UseapiGoogleFlowUploadAsset`
Upload an image to Google Flow for use as a reference.
- **Input**: `image`, `email`
- **Output**: `media_generation_id`

#### `UseapiGoogleFlowImageUpscale`
Upscale a `nano-banana-pro` generated image to 2K or 4K.
- **Input**: `media_generation_id`, `resolution`
- **Output**: `image`, `media_generation_id`

---

### Useapi.net/Runway

#### `UseapiRunwayGenerate`
Generate video using Runway Gen-4, Gen-4 Turbo, or Gen-3 Turbo.
- **Input**: `model` (gen4_5, gen4, gen4turbo, gen3turbo), `text_prompt`, `image`, `asset_id`, `aspect_ratio`, `seconds`, `explore_mode`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayVideoToVideo`
Transform or extend an existing video using Runway Gen-4 or Gen-3 Turbo.
- **Input**: `video_asset_id`, `model`, `text_prompt`, `seconds`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayFramesGenerate`
Generate high-quality 1080p images using Runway Frames.
- **Input**: `text_prompt`, `aspect_ratio`, `style`, `diversity`, `num_images`, `image_ref_1/2/3`
- **Output**: `image`, `image_url`, `all_urls`, `task_id`

#### `UseapiRunwayImages`
Generate images using Runway nano-banana, gen4, or gen4-turbo.
- **Input**: `model`, `text_prompt`, `aspect_ratio`, `resolution`, `num_images`, `style`
- **Output**: `image`, `image_url`, `all_urls`, `task_id`

#### `UseapiRunwayUploadAsset`
Upload an image to Runway for use in video or image generation.
- **Input**: `image`, `email`
- **Output**: `asset_id`

#### `UseapiRunwayImageUpscaler`
Upscale an image using Runway's free image upscaling service.
- **Input**: `image_url`, `width`, `height`
- **Output**: `image`

#### `UseapiRunwayGen4Upscale`
Upscale a Runway Gen4 video asset to higher resolution.
- **Input**: `asset_id`, `explore_mode`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayActTwo`
Transfer motion from a driving video to a character using Gen4 Act Two.
- **Input**: `driving_asset_id`, `character_asset_id`, `aspect_ratio`, `body_control`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayActTwoVoice`
Add a voice to a Runway Gen4 Act Two video using a voice ID.
- **Input**: `video_asset_id`, `voice_id`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayLipsync`
Create a lipsync video using Runway.
- **Input**: `image_asset_id`, `video_asset_id`, `audio_asset_id`, `voice_id`, `voice_text`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwaySuperSlowMotion`
Create super slow-motion video from a Runway video asset.
- **Input**: `asset_id`, `speed`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayTranscribe`
Transcribe a Runway video or audio asset to text.
- **Input**: `asset_id`, `language`
- **Output**: `full_text`, `words_json`

#### `UseapiRunwayGen3TurboExtend`
Extend a Runway Gen3 Turbo video.
- **Input**: `asset_id`, `text_prompt`, `seed`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayAleph`
Video-to-video transformation using Runway Gen4 Aleph with optional image conditioning.
- **Input**: `video_asset_id`, `text_prompt`, `image_asset_id` (optional), `image` (optional, auto-uploads)
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayGen3TurboExpand`
Expand (outpaint) a Runway Gen3 Turbo video to a different aspect ratio.
- **Input**: `video_asset_id`, `image_asset_id` (optional), `text_prompt`, `seconds`, `outpaint_aspect_ratio`
- **Output**: `video_url`, `video_path`, `task_id`

#### `UseapiRunwayGen3TurboActOne`
Transfer motion from a driving video to a character using Gen3 Turbo Act One.
- **Input**: `driving_asset_id`, `character_asset_id`, `aspect_ratio`, `motion_multiplier`
- **Output**: `video_url`, `video_path`, `task_id`

---

## Example Workflows

### Chain 1: Imagen 4 → Veo 3.1
```
UseapiTokenFromEnv
    └─► api_token ──► UseapiGoogleFlowGenerateImage
                           └─► media_generation_id ──► UseapiVeoGenerate
```

### Chain 2: Runway Act Two (Motion Transfer)
```
UseapiRunwayUploadAsset (Driving Video Frame) ──► asset_id (Driving)
UseapiRunwayUploadAsset (Character Image)     ──► asset_id (Character)
      │
      └──► UseapiRunwayActTwo (driving_asset_id, character_asset_id)
                └─► video_url ──► UseapiPreviewVideo
```

---

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed help on rate limits, timeouts, and error codes.
