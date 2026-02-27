import json
import os

# Ensure examples directory exists
os.makedirs("examples", exist_ok=True)

# Helper to create a node structure
def create_node(id, type, pos, size, inputs=None, outputs=None, widgets_values=None):
    return {
        "id": id,
        "type": type,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs if inputs else [],
        "outputs": outputs if outputs else [],
        "properties": {"Node name for S&R": type},
        "widgets_values": widgets_values if widgets_values else []
    }

# Helper to create a link
def create_link(id, origin_id, origin_slot, target_id, target_slot, type):
    return [id, origin_id, origin_slot, target_id, target_slot, type]

# 1. Imagen4 -> Veo3.1 Pipeline
def create_imagen_veo_workflow():
    nodes = []
    links = []

    # Node 1: UseapiTokenFromEnv
    nodes.append(create_node(1, "UseapiTokenFromEnv", [100, 100], [300, 50],
                             outputs=[{"name": "api_token", "type": "STRING", "links": [1, 2], "slot_index": 0}],
                             widgets_values=["USEAPI_TOKEN"]))

    # Node 2: UseapiGoogleFlowGenerateImage
    # Widgets: prompt, model, aspect_ratio, api_token(input), email, count, seed, ref1, ref2, ref3
    nodes.append(create_node(2, "UseapiGoogleFlowGenerateImage", [100, 300], [400, 300],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 1}],
                             outputs=[{"name": "image", "type": "IMAGE", "links": [3], "slot_index": 0},
                                      {"name": "image_url", "type": "STRING", "links": [], "slot_index": 1},
                                      {"name": "media_generation_id", "type": "STRING", "links": [4], "slot_index": 2},
                                      {"name": "all_urls", "type": "STRING", "links": [], "slot_index": 3}],
                             widgets_values=["A futuristic city with flying cars", "imagen-4", "landscape", "", "", 4, 0, "", "", ""]))

    # Node 3: Preview Image (Standard ComfyUI node)
    nodes.append(create_node(3, "PreviewImage", [550, 300], [300, 300],
                             inputs=[{"name": "images", "type": "IMAGE", "link": 3}],
                             widgets_values=[]))

    # Node 4: UseapiVeoGenerate
    # Widgets: prompt, model, aspect_ratio, api_token(input), email, count, seed, start_image, end_image, ref1, ref2, ref3
    nodes.append(create_node(4, "UseapiVeoGenerate", [100, 700], [400, 400],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 2},
                                     {"name": "reference_image_1", "type": "STRING", "link": 4, "widget": {"name": "reference_image_1"}}],
                             outputs=[{"name": "video_url", "type": "STRING", "links": [5], "slot_index": 0},
                                      {"name": "video_path", "type": "STRING", "links": [6], "slot_index": 1},
                                      {"name": "media_generation_id", "type": "STRING", "links": [], "slot_index": 2}],
                             widgets_values=["Cinematic flyover of the futuristic city", "veo-3.1-quality", "landscape", "", "", 1, 0, "", "", "", "", ""]))

    # Node 5: UseapiPreviewVideo
    nodes.append(create_node(5, "UseapiPreviewVideo", [550, 700], [400, 200],
                             inputs=[{"name": "video_url", "type": "STRING", "link": 5},
                                     {"name": "video_path", "type": "STRING", "link": 6}],
                             outputs=[{"name": "info", "type": "STRING", "links": [], "slot_index": 0}],
                             widgets_values=[]))

    links.append(create_link(1, 1, 0, 2, 0, "STRING")) # Token -> Imagen
    links.append(create_link(2, 1, 0, 4, 0, "STRING")) # Token -> Veo
    links.append(create_link(3, 2, 0, 3, 0, "IMAGE"))  # Imagen -> Preview
    links.append(create_link(4, 2, 2, 4, 1, "STRING")) # Imagen ID -> Veo Ref 1 (Slot 2 of Imagen to input)
    links.append(create_link(5, 4, 0, 5, 0, "STRING")) # Veo URL -> Preview
    links.append(create_link(6, 4, 1, 5, 1, "STRING")) # Veo Path -> Preview

    workflow = {
        "last_node_id": 5,
        "last_link_id": 6,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    return workflow

# 2. RunwayFrames -> Gen4 Pipeline
def create_runway_frames_gen4_workflow():
    nodes = []
    links = []

    # Node 1: Token
    nodes.append(create_node(1, "UseapiTokenFromEnv", [100, 50], [300, 50],
                             outputs=[{"name": "api_token", "type": "STRING", "links": [1, 2], "slot_index": 0}],
                             widgets_values=["USEAPI_TOKEN"]))

    # Node 2: UseapiRunwayFramesGenerate
    # Widgets: text_prompt, api_token, email, aspect_ratio, style, diversity, num_images, seed, explore_mode, ref1, ref2, ref3, poll, wait
    nodes.append(create_node(2, "UseapiRunwayFramesGenerate", [100, 200], [400, 400],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 1}],
                             outputs=[{"name": "image", "type": "IMAGE", "links": [3, 4], "slot_index": 0},
                                      {"name": "image_url", "type": "STRING", "links": [], "slot_index": 1},
                                      {"name": "all_urls", "type": "STRING", "links": [], "slot_index": 2},
                                      {"name": "task_id", "type": "STRING", "links": [], "slot_index": 3}],
                             widgets_values=["A cyberpunk detective in neon rain", "", "", "16:9", "cinematic", 2, "4", 0, True, "", "", "", 5, 120]))

    # Node 3: Preview Image
    nodes.append(create_node(3, "PreviewImage", [550, 200], [300, 300],
                             inputs=[{"name": "images", "type": "IMAGE", "link": 3}],
                             widgets_values=[]))

    # Node 4: UseapiRunwayGenerate
    # Widgets: model, text_prompt, api_token, image, asset_id, email, aspect_ratio, seconds, seed, explore, max_jobs, poll, wait
    nodes.append(create_node(4, "UseapiRunwayGenerate", [100, 700], [400, 400],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 2},
                                     {"name": "image", "type": "IMAGE", "link": 4}],
                             outputs=[{"name": "video_url", "type": "STRING", "links": [5], "slot_index": 0},
                                      {"name": "video_path", "type": "STRING", "links": [6], "slot_index": 1},
                                      {"name": "task_id", "type": "STRING", "links": [], "slot_index": 2}],
                             widgets_values=["gen4", "The detective looks around slowly", "", "", "", "16:9", "10", 0, True, 5, 10, 600]))

    # Node 5: UseapiPreviewVideo
    nodes.append(create_node(5, "UseapiPreviewVideo", [550, 700], [400, 200],
                             inputs=[{"name": "video_url", "type": "STRING", "link": 5},
                                     {"name": "video_path", "type": "STRING", "link": 6}],
                             outputs=[{"name": "info", "type": "STRING", "links": [], "slot_index": 0}],
                             widgets_values=[]))

    links.append(create_link(1, 1, 0, 2, 0, "STRING"))
    links.append(create_link(2, 1, 0, 4, 0, "STRING"))
    links.append(create_link(3, 2, 0, 3, 0, "IMAGE"))
    links.append(create_link(4, 2, 0, 4, 1, "IMAGE")) # Frames Image -> Gen4 Image Input
    links.append(create_link(5, 4, 0, 5, 0, "STRING"))
    links.append(create_link(6, 4, 1, 5, 1, "STRING"))

    workflow = {
        "last_node_id": 5,
        "last_link_id": 6,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    return workflow

# 3. Veo Text-to-Video
def create_veo_text_to_video_workflow():
    nodes = []
    links = []

    nodes.append(create_node(1, "UseapiTokenFromEnv", [100, 100], [300, 50],
                             outputs=[{"name": "api_token", "type": "STRING", "links": [1], "slot_index": 0}],
                             widgets_values=["USEAPI_TOKEN"]))

    nodes.append(create_node(2, "UseapiVeoGenerate", [100, 250], [400, 400],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 1}],
                             outputs=[{"name": "video_url", "type": "STRING", "links": [2], "slot_index": 0},
                                      {"name": "video_path", "type": "STRING", "links": [3], "slot_index": 1},
                                      {"name": "media_generation_id", "type": "STRING", "links": [], "slot_index": 2}],
                             widgets_values=["A serene mountain landscape with a flowing river", "veo-3.1-quality", "landscape", "", "", 1, 0, "", "", "", "", ""]))

    nodes.append(create_node(3, "UseapiPreviewVideo", [550, 250], [400, 200],
                             inputs=[{"name": "video_url", "type": "STRING", "link": 2},
                                     {"name": "video_path", "type": "STRING", "link": 3}],
                             outputs=[{"name": "info", "type": "STRING", "links": [], "slot_index": 0}],
                             widgets_values=[]))

    links.append(create_link(1, 1, 0, 2, 0, "STRING"))
    links.append(create_link(2, 2, 0, 3, 0, "STRING"))
    links.append(create_link(3, 2, 1, 3, 1, "STRING"))

    workflow = {
        "last_node_id": 3,
        "last_link_id": 3,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    return workflow

# 4. Image Upscale Pipeline
def create_image_upscale_workflow():
    nodes = []
    links = []

    nodes.append(create_node(1, "UseapiTokenFromEnv", [100, 100], [300, 50],
                             outputs=[{"name": "api_token", "type": "STRING", "links": [1, 2], "slot_index": 0}],
                             widgets_values=["USEAPI_TOKEN"]))

    # Important: Upscale only works with nano-banana-pro
    nodes.append(create_node(2, "UseapiGoogleFlowGenerateImage", [100, 250], [400, 300],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 1}],
                             outputs=[{"name": "image", "type": "IMAGE", "links": [3], "slot_index": 0},
                                      {"name": "image_url", "type": "STRING", "links": [], "slot_index": 1},
                                      {"name": "media_generation_id", "type": "STRING", "links": [4], "slot_index": 2},
                                      {"name": "all_urls", "type": "STRING", "links": [], "slot_index": 3}],
                             widgets_values=["A highly detailed macro shot of a dew drop", "nano-banana-pro", "landscape", "", "", 1, 0, "", "", ""]))

    nodes.append(create_node(3, "PreviewImage", [550, 250], [300, 300],
                             inputs=[{"name": "images", "type": "IMAGE", "link": 3}],
                             widgets_values=[]))

    # Upscale node
    # Widgets: media_generation_id, resolution, api_token
    nodes.append(create_node(4, "UseapiGoogleFlowImageUpscale", [100, 600], [400, 200],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 2},
                                     {"name": "media_generation_id", "type": "STRING", "link": 4, "widget": {"name": "media_generation_id"}}],
                             outputs=[{"name": "image", "type": "IMAGE", "links": [5], "slot_index": 0},
                                      {"name": "media_generation_id", "type": "STRING", "links": [], "slot_index": 1}],
                             widgets_values=["", "4k", ""]))

    nodes.append(create_node(5, "PreviewImage", [550, 600], [300, 300],
                             inputs=[{"name": "images", "type": "IMAGE", "link": 5}],
                             widgets_values=[]))

    links.append(create_link(1, 1, 0, 2, 0, "STRING"))
    links.append(create_link(2, 1, 0, 4, 0, "STRING"))
    links.append(create_link(3, 2, 0, 3, 0, "IMAGE"))
    links.append(create_link(4, 2, 2, 4, 0, "STRING"))
    links.append(create_link(5, 4, 0, 5, 0, "IMAGE"))

    workflow = {
        "last_node_id": 5,
        "last_link_id": 5,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    return workflow

# 5. Cross-platform Imagen -> Runway Pipeline
def create_cross_platform_workflow():
    nodes = []
    links = []

    nodes.append(create_node(1, "UseapiTokenFromEnv", [100, 50], [300, 50],
                             outputs=[{"name": "api_token", "type": "STRING", "links": [1, 2], "slot_index": 0}],
                             widgets_values=["USEAPI_TOKEN"]))

    # Imagen 4
    nodes.append(create_node(2, "UseapiGoogleFlowGenerateImage", [100, 200], [400, 300],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 1}],
                             outputs=[{"name": "image", "type": "IMAGE", "links": [3, 4], "slot_index": 0},
                                      {"name": "image_url", "type": "STRING", "links": [], "slot_index": 1},
                                      {"name": "media_generation_id", "type": "STRING", "links": [], "slot_index": 2},
                                      {"name": "all_urls", "type": "STRING", "links": [], "slot_index": 3}],
                             widgets_values=["A surreal landscape with floating islands", "imagen-4", "landscape", "", "", 1, 0, "", "", ""]))

    nodes.append(create_node(3, "PreviewImage", [550, 200], [300, 300],
                             inputs=[{"name": "images", "type": "IMAGE", "link": 3}],
                             widgets_values=[]))

    # Runway Gen4
    # Uses the image from Imagen as input
    nodes.append(create_node(4, "UseapiRunwayGenerate", [100, 600], [400, 400],
                             inputs=[{"name": "api_token", "type": "STRING", "link": 2},
                                     {"name": "image", "type": "IMAGE", "link": 4}],
                             outputs=[{"name": "video_url", "type": "STRING", "links": [5], "slot_index": 0},
                                      {"name": "video_path", "type": "STRING", "links": [6], "slot_index": 1},
                                      {"name": "task_id", "type": "STRING", "links": [], "slot_index": 2}],
                             widgets_values=["gen4", "Clouds moving rapidly, waterfalls flowing", "", "", "", "16:9", "10", 0, True, 5, 10, 600]))

    nodes.append(create_node(5, "UseapiPreviewVideo", [550, 600], [400, 200],
                             inputs=[{"name": "video_url", "type": "STRING", "link": 5},
                                     {"name": "video_path", "type": "STRING", "link": 6}],
                             outputs=[{"name": "info", "type": "STRING", "links": [], "slot_index": 0}],
                             widgets_values=[]))

    links.append(create_link(1, 1, 0, 2, 0, "STRING"))
    links.append(create_link(2, 1, 0, 4, 0, "STRING"))
    links.append(create_link(3, 2, 0, 3, 0, "IMAGE"))
    links.append(create_link(4, 2, 0, 4, 1, "IMAGE"))
    links.append(create_link(5, 4, 0, 5, 0, "STRING"))
    links.append(create_link(6, 4, 1, 5, 1, "STRING"))

    workflow = {
        "last_node_id": 5,
        "last_link_id": 6,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    return workflow

if __name__ == "__main__":
    workflows = {
        "01_imagen_to_veo_pipeline.json": create_imagen_veo_workflow(),
        "02_runway_frames_to_gen4_pipeline.json": create_runway_frames_gen4_workflow(),
        "03_veo_text_to_video.json": create_veo_text_to_video_workflow(),
        "04_image_upscale_pipeline.json": create_image_upscale_workflow(),
        "05_imagen_to_runway_pipeline.json": create_cross_platform_workflow()
    }

    for filename, data in workflows.items():
        filepath = os.path.join("examples", filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {filepath}")
