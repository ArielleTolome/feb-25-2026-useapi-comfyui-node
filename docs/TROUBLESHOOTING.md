# Troubleshooting Guide

This guide covers common issues you might encounter when using ComfyUI-UseapiNet.

## Table of Contents
1. [Token not found](#token-not-found)
2. [Rate limited (429)](#rate-limited-429)
3. [Timeout errors (408)](#timeout-errors-408)
4. [Wrong tensor format](#wrong-tensor-format)
5. [Missing dependencies](#missing-dependencies)
6. [Runway/Google Account limits (403)](#runwaygoogle-account-limits-403)
7. [Service Unavailable (503)](#service-unavailable-503)
8. [Google Flow Image Upscale: no encodedImage](#google-flow-image-upscale-no-encodedimage)

---

## Token not found

**Error:** `[Useapi.net] API token not provided.`

### Cause
The node could not find your Useapi.net API token. The token is required to authenticate your requests.

### Solution
You have three options to provide the token:
1.  **Environment Variable (Recommended):** Set `USEAPI_TOKEN` in your environment before launching ComfyUI.
    ```bash
    export USEAPI_TOKEN=your_token_here
    ```
2.  **TokenFromEnv Node:** Use the `UseapiTokenFromEnv` node and wire its output to the `api_token` input of other nodes.
3.  **Direct Input:** Paste your token directly into the `api_token` widget on the node.

### How to Debug
*   Check if the environment variable is set by running `echo $USEAPI_TOKEN` in your terminal before starting ComfyUI.
*   If using the `UseapiTokenFromEnv` node, ensure the `env_var_name` matches what you set in your shell.
*   Inspect the ComfyUI console logs. The node will log `[Useapi.net] Token loaded from env var...` if successful.

---

## Rate limited (429)

**Error:** `[Useapi.net] Rate limited (429). Wait 5-10s or add more Useapi.net accounts.`

### Cause
You have exceeded the maximum number of concurrent requests allowed by your Useapi.net subscription.

### Solution
1.  **Wait and Retry:** The error is transient. Wait a few seconds before trying again.
2.  **Reduce Concurrency:** If running batch jobs, reduce the number of parallel executions.
3.  **Add Accounts:** You can add more Google or Runway accounts to your Useapi.net subscription to increase your concurrency limit.

### How to Debug
*   Check the detailed error message in the ComfyUI console or the red error toast in the UI. It often contains the specific URL that was rate-limited.
*   Monitor your usage on the [Useapi.net Dashboard](https://useapi.net).

---

## Timeout errors (408)

**Error:** `[Useapi.net] Request timeout (408). Generation took too long.` or `RuntimeError: Request timed out after 600s`

### Cause
The video generation took longer than the configured timeout period. This is common for high-quality video models like Google Veo or Runway Gen-4, which can take several minutes.

### Solution
*   **Increase Timeout:** If the node allows it, check if there is a timeout setting (most nodes default to 600s).
*   **Check Service Status:** Sometimes the underlying service (Google or Runway) is experiencing high load, causing slower generations.
*   **Veo Models:** `veo-3.1-quality` is significantly slower than `veo-3.1-fast`. Switch to the fast model if speed is critical.

### How to Debug
*   The error message will specify the URL that timed out.
*   Check if the generation eventually appears in your Useapi.net dashboard history, even if ComfyUI timed out.

---

## Wrong tensor format

**Error:** `RuntimeError: Expected tensor with shape (1, H, W, 3) but got ...`

### Cause
The node received an image input that is not in the standard ComfyUI format. ComfyUI images are expected to be:
*   **Shape:** `(batch_size, height, width, channels)` - typically `(1, H, W, 3)` for single images.
*   **Type:** `float32`
*   **Range:** `0.0` to `1.0`

### Solution
*   Ensure you are passing an `IMAGE` output from a LoadImage or similar node, not `LATENT` or `MASK`.
*   If you are preprocessing images with custom Python code, ensure you normalize pixel values to [0, 1] and add a batch dimension (`unsqueeze(0)`).

### How to Debug
*   Use a "Shape" or "Debug" node (from other custom node packs) to inspect the shape and type of the tensor you are feeding into the Useapi node.

---

## Missing dependencies

**Error:** `[Useapi.net] UseapiLoadVideoFrame requires opencv-python.`

### Cause
The `UseapiLoadVideoFrame` node relies on the `opencv-python` library to extract frames from video files, but it is not installed in your ComfyUI environment.

### Solution
Install the missing package:
```bash
pip install opencv-python
```
Restart ComfyUI after installation.

### How to Debug
*   Check the ComfyUI startup logs. You might see a warning or error about missing imports when the custom nodes are loaded.
*   Verify installation with `pip show opencv-python`.

---

## Runway/Google Account limits (403)

**Error:** `[Useapi.net] Forbidden (403).` or `API error: 403`

### Cause
This indicates an issue with the Google or Runway account linked to your Useapi.net token, not Useapi.net itself.
*   **Runway:** You might be out of credits or your plan does not support the requested feature (e.g., Gen-3 Turbo).
*   **Google:** Your account might not have access to the specific model (e.g., Veo) or you might be blocked by Google's safety filters.

### Solution
1.  **Check Credits:** Log in to your Runway or Google account and check your credit balance.
2.  **Verify Access:** Ensure your account has access to the model you are trying to use.
3.  **Transient Block:** "API error: 403" from Google is often a temporary block. Wait a few minutes and try again.

### How to Debug
*   The error message usually contains details from the upstream provider. Read the `Detail:` section in the error message carefully.
*   Log in to the respective service (RunwayML or Google Cloud/Vertex AI) directly to see if there are any alerts on your account.

---

## Service Unavailable (503)

**Error:** `[Useapi.net] Service unavailable (503).`

### Cause
The Useapi.net service or the underlying provider (Google/Runway) is temporarily down or overloaded.

### Solution
*   **Retry:** These errors are almost always temporary. Wait a minute and try again.
*   **Check Status:** Check the [Useapi.net Discord](https://discord.gg/useapi) or status page for outage announcements.

---

## Google Flow Image Upscale: no encodedImage

**Error:** `[Useapi.net] Google Flow Image Upscale: 'encodedImage' missing in response.`

### Cause
The upscale endpoint only works with images generated by `nano-banana-pro`. Images from `imagen-4` or `nano-banana` are not supported by the upscale API.

### Solution
Use `nano-banana-pro` as the model when generating images that you intend to upscale.
