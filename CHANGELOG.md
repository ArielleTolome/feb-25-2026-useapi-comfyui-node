# Changelog

All notable changes to this project will be documented in this file.

## [0.5.3] - 2026-03-03

### Fixed
- **`UseapiVeoExtend` persistent 400 "All operations failed"**: Added optional `email` field to `UseapiVeoExtend`. The Useapi.net multi-tenant API routes requests to the correct Google account via `email`; without it, extend requests were routed to a different account than the one that generated the source video, causing the error. Pass the same email used in `UseapiVeoGenerate`.
- **Improved error message for "All operations failed" (400)**: `_check_status` now detects this specific response and raises a targeted, context-aware message. Veo Extend calls get guidance about passing `email` and trying a different prompt; all other endpoints get a generic account-routing hint.

## [0.5.1] - 2026-03-03

### Added
- **`UseapiVideoToFrames` node** (`Useapi.net/Utils`): Decodes any UseAPI video output (`video_path`) into a ComfyUI `IMAGE` tensor batch compatible with `VHS_VideoCombine` and native `SaveVideo` nodes. Also shows an in-node video preview. Outputs: `frames` (IMAGE), `frame_count` (INT), `fps` (FLOAT). Requires `opencv-python`.

## [0.5.0] - 2026-03-03

### Added
- **3 New Runway Nodes**:
  - `UseapiRunwayAleph`: Video-to-video transformation using Gen4 Aleph with optional image conditioning.
  - `UseapiRunwayGen3TurboExpand`: Expand (outpaint) Gen3 Turbo videos to landscape or portrait.
  - `UseapiRunwayGen3TurboActOne`: Motion transfer from a driving video to a character using Gen3 Turbo Act One.
- Updated tests to cover the 3 new nodes (structure, contract, and category validation).
- Added `pyproject.toml` — ComfyUI now displays the pack as **UseAPI.net** instead of the folder name.

## [0.2.0] - 2026-02-25

### Added
- **10 New Nodes**:
  - `UseapiVeoVideoToGif`: Convert Veo videos to GIF.
  - `UseapiVeoConcatenate`: Concatenate multiple Veo videos with trim options.
  - `UseapiRunwayImages`: Generate images with Runway (nano-banana, gen4, gen4-turbo).
  - `UseapiRunwayGen4Upscale`: Upscale Runway Gen4 videos.
  - `UseapiRunwayActTwo`: Motion transfer from driving video to character.
  - `UseapiRunwayActTwoVoice`: Add voice to Act Two videos.
  - `UseapiRunwayLipsync`: Create lipsync videos.
  - `UseapiRunwaySuperSlowMotion`: Apply super slow-motion.
  - `UseapiRunwayTranscribe`: Transcribe video/audio assets.
  - `UseapiRunwayGen3TurboExtend`: Extend Gen3 Turbo videos.
- **Documentation**:
  - Updated `README.md` with installation steps for Windows/Linux/Mac.
  - Added full input/output reference for all 24 nodes.
  - Added `examples/` directory with simulated workflow JSON files (`google_flow_workflows.json`, `runway_workflows.json`).

## [0.1.0] - 2026-02-25

### Added
- Added CI workflow via GitHub Actions (`.github/workflows/ci.yml`) to run structure tests.
- Added support for `nodes_config.json` to allow users to customize default parameter values (e.g., model, aspect ratio, timeout).
- Added CI status badge to `README.md`.
