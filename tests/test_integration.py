"""Live integration tests using real Useapi.net API calls.

Usage:
    export USEAPI_TOKEN=your_token_here
    python -m pytest tests/test_integration.py -v -s

All tests are skipped if USEAPI_TOKEN is not set.
WARNING: Veo video generation takes 60-180 seconds.
"""
import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOKEN = os.environ.get("USEAPI_TOKEN", "")
SKIP_REASON = "USEAPI_TOKEN env var not set — skipping live API tests"
SKIP = not bool(TOKEN)


@unittest.skipIf(SKIP, SKIP_REASON)
class TestTokenFromEnv(unittest.TestCase):
    def test_token_loads(self):
        from useapi_nodes import UseapiTokenFromEnv
        node = UseapiTokenFromEnv()
        os.environ["USEAPI_TOKEN"] = TOKEN
        result = node.execute("USEAPI_TOKEN")
        self.assertEqual(result[0], TOKEN)
        print(f"  ✓ Token loaded: {TOKEN[:10]}...")


@unittest.skipIf(SKIP, SKIP_REASON)
class TestGoogleFlowGenerateImage(unittest.TestCase):
    def test_imagen4_generates(self):
        from useapi_nodes import UseapiGoogleFlowGenerateImage
        node = UseapiGoogleFlowGenerateImage()
        result = node.execute(
            prompt="A simple red circle on a white background",
            model="imagen-4",
            aspect_ratio="landscape",
            api_token=TOKEN,
            count=1,
            seed=42,
        )
        image_tensor, image_url, media_gen_id, all_urls = result
        self.assertIsNotNone(image_tensor)
        self.assertTrue(image_url.startswith("http"))
        self.assertTrue(media_gen_id.startswith("user:"))
        urls = json.loads(all_urls)
        self.assertIsInstance(urls, list)
        self.assertGreater(len(urls), 0)
        print(f"  ✓ Imagen 4 generation: mediaGenerationId={media_gen_id[:40]}...")


@unittest.skipIf(SKIP, SKIP_REASON)
class TestRunwayFramesGenerate(unittest.TestCase):
    def test_frames_generates(self):
        from useapi_nodes import UseapiRunwayFramesGenerate
        node = UseapiRunwayFramesGenerate()
        result = node.execute(
            text_prompt="A simple red cube on a white background, minimal, clean",
            api_token=TOKEN,
            aspect_ratio="16:9",
            num_images="1",
            explore_mode=True,
            timeout=180,
        )
        image_tensor, image_url, all_urls, task_id = result
        self.assertIsNotNone(image_tensor)
        self.assertTrue(image_url.startswith("http"))
        self.assertTrue(bool(task_id))
        print(f"  ✓ Runway Frames: taskId={task_id[:40]}...")


@unittest.skipIf(SKIP, SKIP_REASON)
class TestVeoGenerate(unittest.TestCase):
    """WARNING: This test takes 60-180 seconds."""

    def test_veo_generates(self):
        from useapi_nodes import UseapiVeoGenerate
        node = UseapiVeoGenerate()
        result = node.execute(
            prompt="A slow pan across a calm ocean at sunset",
            model="veo-3.1-fast",
            aspect_ratio="landscape",
            api_token=TOKEN,
            count=1,
            timeout=600,
        )
        video_url, video_path, media_gen_id = result
        self.assertTrue(video_url.startswith("http"))
        self.assertTrue(os.path.exists(video_path))
        self.assertGreater(os.path.getsize(video_path), 0)
        self.assertTrue(bool(media_gen_id))
        print(f"  ✓ Veo 3.1 generation: {video_path}, mediaGenerationId={media_gen_id[:40]}...")


if __name__ == "__main__":
    print("Running integration tests...\n")
    print(f"Token: {TOKEN[:10]}..." if TOKEN else "No token set — all tests will be skipped")
    unittest.main(verbosity=2)
