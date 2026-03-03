# Design: UseapiVideoToFrames Node

**Date:** 2026-03-03
**Status:** Approved

## Problem

All UseAPI video nodes (`UseapiVeoGenerate`, `UseapiRunwayGenerate`, etc.) output `video_path` as a `STRING`. ComfyUI's standard video save/combine nodes (`VHS_VideoCombine`, native `SaveVideo`) expect `IMAGE` tensor batches. There is no direct wire between these — users cannot preview or save generated videos through standard ComfyUI nodes.

## Solution: New `UseapiVideoToFrames` Node

A single dual-purpose node that:
1. Extracts all frames from a video file into a ComfyUI `IMAGE` batch tensor
2. Previews the video in the ComfyUI UI (in-node video preview panel)

## Chosen Approach

**Approach A: New dedicated converter node** (non-breaking, works for all video nodes)

Rejected alternatives:
- Approach B (add IMAGE output to every video node): too invasive, ~12 nodes to modify, adds latency even when not needed
- Approach C (batch mode in existing UseapiLoadVideoFrame): confusing mixed API

## Node Specification

**Class:** `UseapiVideoToFrames`
**Category:** `Useapi.net/Utils`
**OUTPUT_NODE:** `True` (enables UI preview)

### Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `video_path` | `STRING` | `""` | Local file path from any UseAPI video node |
| `max_frames` | `INT` | `0` | Max frames to extract; `0` = all frames |
| `start_frame` | `INT` | `0` | Frame index to start extraction from |
| `frame_step` | `INT` | `1` | Sample every Nth frame (e.g. `2` = half FPS) |

### Outputs

| Name | Type | Description |
|---|---|---|
| `frames` | `IMAGE` | Batch tensor `[N, H, W, 3]` — wires into VHS_VideoCombine or SaveVideo |
| `frame_count` | `INT` | Number of frames extracted |
| `fps` | `FLOAT` | Original video FPS (pass to VHS_VideoCombine's fps input) |

### Behavior

1. Validates path with `_is_safe_path()`
2. Guards with `_CV2_AVAILABLE` — raises clear error if opencv not installed
3. Opens video with `cv2.VideoCapture`, reads FPS and total frame count
4. Extracts frames respecting `start_frame`, `frame_step`, `max_frames`
5. Converts BGR → RGB, stacks into `[N, H, W, 3]` float32 tensor (0–1 range)
6. Copies video file to ComfyUI output dir
7. Returns `{"ui": {"video": [...]}}` for in-node preview + `(frames, frame_count, fps)`

## Example Workflow

```
UseapiVeoGenerate
  └─ video_path ──→ UseapiVideoToFrames
                        ├─ frames ──→ VHS_VideoCombine ──→ SaveVideo
                        ├─ fps    ──→ VHS_VideoCombine.fps
                        └─ [UI preview appears in node]
```

## Dependencies

- `opencv-python` (optional, same as `UseapiLoadVideoFrame`) — guarded by existing `_CV2_AVAILABLE` flag
- No new dependencies

## Registration

Add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `useapi_nodes.py`.
