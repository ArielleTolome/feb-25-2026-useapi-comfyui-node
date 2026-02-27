"""Validation tests for Useapi nodes to ensure error handling works as expected.
"""
import unittest
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not in ComfyUI env
try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    np = MagicMock()
    sys.modules["torch"] = torch
    sys.modules["numpy"] = np

try:
    import cv2
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["cv2"] = MagicMock()

from useapi_nodes import UseapiVeoConcatenate

class TestVeoConcatenateValidation(unittest.TestCase):
    def test_insufficient_media_args(self):
        node = UseapiVeoConcatenate()

        # Case 1: All empty
        with self.assertRaisesRegex(ValueError, "at least 2 mediaGenerationIds required"):
            node.execute(
                media_1=" ",
                media_2=" ",
                api_token="dummy"
            )

        # Case 2: Only one valid
        with self.assertRaisesRegex(ValueError, "at least 2 mediaGenerationIds required"):
            node.execute(
                media_1="id1",
                media_2=" ",
                api_token="dummy"
            )

    @mock.patch('useapi_nodes._make_request')
    def test_valid_media_args(self, mock_make_request):
        # Configure mock to simulate network call attempt
        mock_make_request.side_effect = RuntimeError("Network call attempted")

        node = UseapiVeoConcatenate()

        # Should pass validation and hit the network mock
        with self.assertRaisesRegex(RuntimeError, "Network call attempted"):
            node.execute(
                media_1="id1",
                media_2="id2",
                api_token="dummy"
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
