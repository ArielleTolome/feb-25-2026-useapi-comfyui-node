"""Tests for UseapiVideoToFrames node."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not present
try:
    import torch
except ImportError:
    torch = MagicMock()
    sys.modules["torch"] = torch

try:
    import numpy as np
except ImportError:
    np = MagicMock()
    sys.modules["numpy"] = np

try:
    import cv2
except ImportError:
    cv2 = MagicMock()
    sys.modules["cv2"] = cv2

import useapi_nodes
from useapi_nodes import UseapiVideoToFrames


class TestUseapiVideoToFramesContract(unittest.TestCase):
    """Verify the node satisfies the ComfyUI node contract."""

    def test_has_required_class_attributes(self):
        for attr in ("CATEGORY", "FUNCTION", "RETURN_TYPES", "RETURN_NAMES", "OUTPUT_NODE"):
            self.assertTrue(hasattr(UseapiVideoToFrames, attr), f"Missing {attr}")

    def test_output_node_is_true(self):
        self.assertTrue(UseapiVideoToFrames.OUTPUT_NODE)

    def test_return_types(self):
        self.assertEqual(UseapiVideoToFrames.RETURN_TYPES, ("IMAGE", "INT", "FLOAT"))

    def test_return_names(self):
        self.assertEqual(UseapiVideoToFrames.RETURN_NAMES, ("frames", "frame_count", "fps"))

    def test_return_types_names_same_length(self):
        self.assertEqual(len(UseapiVideoToFrames.RETURN_TYPES), len(UseapiVideoToFrames.RETURN_NAMES))

    def test_category(self):
        self.assertEqual(UseapiVideoToFrames.CATEGORY, "Useapi.net/Utils")

    def test_input_types_has_video_path(self):
        inputs = UseapiVideoToFrames.INPUT_TYPES()
        self.assertIn("video_path", inputs.get("required", {}))

    def test_input_types_has_optional_frame_controls(self):
        inputs = UseapiVideoToFrames.INPUT_TYPES()
        optional = inputs.get("optional", {})
        self.assertIn("max_frames", optional)
        self.assertIn("start_frame", optional)
        self.assertIn("frame_step", optional)


class TestUseapiVideoToFramesExecute(unittest.TestCase):
    """Test execute() with mocked cv2 and filesystem."""

    @patch("useapi_nodes._CV2_AVAILABLE", False)
    def test_raises_when_cv2_not_available(self):
        node = UseapiVideoToFrames()
        with self.assertRaises(RuntimeError) as ctx:
            node.execute(video_path="/fake/video.mp4")
        self.assertIn("opencv-python", str(ctx.exception))

    @patch("useapi_nodes._CV2_AVAILABLE", True)
    @patch("useapi_nodes._is_safe_path", return_value=False)
    def test_raises_on_unsafe_path(self, _):
        node = UseapiVideoToFrames()
        with self.assertRaises(ValueError) as ctx:
            node.execute(video_path="http://evil.com/video.mp4")
        self.assertIn("Security error", str(ctx.exception))

    @patch("useapi_nodes._CV2_AVAILABLE", True)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    @patch("useapi_nodes.cv2")
    def test_raises_when_video_cannot_open(self, mock_cv2, _):
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap
        node = UseapiVideoToFrames()
        with self.assertRaises(RuntimeError) as ctx:
            node.execute(video_path="/fake/video.mp4")
        self.assertIn("Cannot open", str(ctx.exception))

    @patch("useapi_nodes._CV2_AVAILABLE", True)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    @patch("useapi_nodes.cv2")
    def test_raises_when_no_frames_extracted(self, mock_cv2, _):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 24.0
        cap.read.return_value = (False, None)  # no frames at all
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        node = UseapiVideoToFrames()
        with self.assertRaises(RuntimeError):
            node.execute(video_path="/fake/video.mp4", start_frame=99999)


@unittest.skipIf(not useapi_nodes._CV2_AVAILABLE or isinstance(cv2, MagicMock), "OpenCV required for integration tests")
class TestUseapiVideoToFramesIntegration(unittest.TestCase):
    """Integration tests for UseapiVideoToFrames with a real video file."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import numpy as np

        if not useapi_nodes._CV2_AVAILABLE or isinstance(cv2, MagicMock):
            return

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.video_path = os.path.join(cls.temp_dir.name, "test_video.mp4")

        # Create a simple 5-frame video
        cls.width = 64
        cls.height = 64
        cls.fps = 10.0
        cls.num_frames = 5

        # Use 'mp4v' or 'XVID' codec depending on platform
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.video_path, fourcc, cls.fps, (cls.width, cls.height))

        # Write 5 frames of different shades of gray
        for i in range(cls.num_frames):
            frame = np.zeros((cls.height, cls.width, 3), dtype=np.uint8)
            val = int((i + 1) * 255 / cls.num_frames)
            frame[:] = (val, val, val)
            out.write(frame)

        out.release()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()

    @patch("useapi_nodes.folder_paths", None)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    def test_extract_all_frames(self, _):
        node = UseapiVideoToFrames()
        result = node.execute(video_path=self.video_path)

        # Result is a dict with "ui" and "result" keys
        self.assertIn("result", result)
        frames, count, fps = result["result"]

        self.assertEqual(count, self.num_frames)
        self.assertAlmostEqual(fps, self.fps, delta=0.1)

        # frames should be a tensor of shape [N, H, W, 3]
        self.assertEqual(frames.shape, (self.num_frames, self.height, self.width, 3))

        # Test tensor values are float and in [0, 1] range
        self.assertEqual(frames.dtype, torch.float32)
        import numpy as np
        frame_max = np.max(frames.numpy())
        frame_min = np.min(frames.numpy())
        self.assertLessEqual(frame_max, 1.0)
        self.assertGreaterEqual(frame_min, 0.0)

    @patch("useapi_nodes.folder_paths", None)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    def test_extract_with_max_frames(self, _):
        node = UseapiVideoToFrames()
        max_f = 2
        result = node.execute(video_path=self.video_path, max_frames=max_f)

        frames, count, fps = result["result"]

        self.assertEqual(count, max_f)
        self.assertEqual(frames.shape, (max_f, self.height, self.width, 3))

    @patch("useapi_nodes.folder_paths", None)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    def test_extract_with_start_frame(self, _):
        node = UseapiVideoToFrames()
        start_f = 2
        result = node.execute(video_path=self.video_path, start_frame=start_f)

        frames, count, fps = result["result"]

        expected_count = self.num_frames - start_f
        self.assertEqual(count, expected_count)
        self.assertEqual(frames.shape, (expected_count, self.height, self.width, 3))

    @patch("useapi_nodes.folder_paths", None)
    @patch("useapi_nodes._is_safe_path", return_value=True)
    def test_extract_with_frame_step(self, _):
        node = UseapiVideoToFrames()
        step = 2
        result = node.execute(video_path=self.video_path, frame_step=step)

        frames, count, fps = result["result"]

        # If 5 frames and step=2: indexes 0, 2, 4 -> 3 frames
        expected_count = 3
        self.assertEqual(count, expected_count)
        self.assertEqual(frames.shape, (expected_count, self.height, self.width, 3))

        # Effective fps should be src_fps / step
        self.assertAlmostEqual(fps, self.fps / step, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
