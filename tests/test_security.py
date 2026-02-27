import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile

# Create temporary directories for testing
temp_dir = tempfile.mkdtemp()
input_dir = os.path.join(temp_dir, "input")
output_dir = os.path.join(temp_dir, "output")
temp_folder_dir = os.path.join(temp_dir, "temp")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_folder_dir, exist_ok=True)

# Mock dependencies
cv2_mock = MagicMock()
sys.modules["cv2"] = cv2_mock

folder_paths_mock = MagicMock()
folder_paths_mock.get_input_directory.return_value = input_dir
folder_paths_mock.get_output_directory.return_value = output_dir
folder_paths_mock.get_temp_directory.return_value = temp_folder_dir
sys.modules["folder_paths"] = folder_paths_mock

sys.modules["torch"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()

# Import the module under test
import useapi_nodes
import numpy as np

class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.load_node = useapi_nodes.UseapiLoadVideoFrame()
        self.preview_node = useapi_nodes.UseapiPreviewVideo()
        # Ensure allowed paths are loaded if cached or init-time logic ran
        # (Though _is_safe_path runs at runtime)

    def test_load_video_frame_ssrf(self):
        """Test that UseapiLoadVideoFrame rejects URLs to prevent SSRF."""
        url = "http://169.254.169.254/latest/meta-data/"
        with self.assertRaises(ValueError, msg="Should reject URLs"):
            self.load_node.execute(url, 0)

    def test_load_video_frame_traversal(self):
        """Test that UseapiLoadVideoFrame rejects path traversal."""
        # This path is definitely outside VIDEO_CACHE_DIR and ComfyUI dirs
        unsafe_path = "/etc/passwd"
        with self.assertRaises(ValueError, msg="Should reject absolute paths outside allowed dirs"):
            self.load_node.execute(unsafe_path, 0)

        unsafe_path_2 = "../../../etc/passwd"
        with self.assertRaises(ValueError, msg="Should reject relative path traversal"):
            self.load_node.execute(unsafe_path_2, 0)

    def test_load_video_frame_allowed_path(self):
        """Test that a valid path in VIDEO_CACHE_DIR is allowed."""
        # Create a dummy file in VIDEO_CACHE_DIR
        valid_path = os.path.join(useapi_nodes.VIDEO_CACHE_DIR, "test_video.mp4")

        # We need to mock cv2.VideoCapture so it doesn't try to open the non-existent file
        # and mock isOpened to return True
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.return_value = 100
        cap_mock.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

        with patch("useapi_nodes.cv2.VideoCapture", return_value=cap_mock):
             # Should not raise ValueError
             self.load_node.execute(valid_path, 0)

    def test_preview_video_traversal(self):
        """Test that UseapiPreviewVideo rejects unsafe paths."""
        unsafe_path = "/etc/passwd"

        # Mock os.path.exists so it thinks the file exists
        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError, msg="Should reject unsafe paths"):
                self.preview_node.execute("http://example.com/video.mp4", unsafe_path)

if __name__ == "__main__":
    unittest.main()
