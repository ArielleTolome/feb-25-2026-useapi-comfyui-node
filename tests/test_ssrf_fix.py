import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add parent directory to path to import useapi_nodes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock dependencies
sys.modules["torch"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["cv2"] = MagicMock()

from useapi_nodes import _download_file

class TestSSRF(unittest.TestCase):
    def test_file_scheme_rejected(self):
        """Ensure file:// URLs raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            _download_file("file:///etc/passwd")
        self.assertIn("Invalid URL scheme: file", str(cm.exception))

    def test_ftp_scheme_rejected(self):
        """Ensure ftp:// URLs raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            _download_file("ftp://example.com/file.txt")
        self.assertIn("Invalid URL scheme: ftp", str(cm.exception))

    @patch("urllib.request.urlopen")
    def test_http_scheme_allowed(self, mock_urlopen):
        """Ensure http:// URLs are allowed (mocked network call)."""
        # Mock the response context manager
        mock_response = MagicMock()
        mock_response.read.return_value = b"video data"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # We need to ensure the cache file doesn't exist so it tries to download
        # But _download_file uses md5 of url for filename.
        # We can mock os.path.exists to always return False for this test
        with patch("os.path.exists", return_value=False), \
             patch("builtins.open", new_callable=MagicMock) as mock_open:

            _download_file("http://example.com/video.mp4")

            # Verify urlopen was called
            mock_urlopen.assert_called()
            args, _ = mock_urlopen.call_args
            self.assertEqual(args[0], "http://example.com/video.mp4")

    @patch("urllib.request.urlopen")
    def test_https_scheme_allowed(self, mock_urlopen):
        """Ensure https:// URLs are allowed (mocked network call)."""
        # Mock the response context manager
        mock_response = MagicMock()
        mock_response.read.return_value = b"video data"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with patch("os.path.exists", return_value=False), \
             patch("builtins.open", new_callable=MagicMock) as mock_open:

            _download_file("https://example.com/video.mp4")

            # Verify urlopen was called
            mock_urlopen.assert_called()
            args, _ = mock_urlopen.call_args
            self.assertEqual(args[0], "https://example.com/video.mp4")

if __name__ == "__main__":
    unittest.main()
