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

    @mock.patch('useapi_nodes._submit_with_progress')
    def test_extended_slots_media_6_to_10(self, mock_submit):
        """Slots media_6..media_10 must appear in the payload when non-empty."""
        mock_submit.return_value = {"encodedVideo": "AAAA"}  # minimal valid response

        node = UseapiVeoConcatenate()
        node.execute(
            media_1="id1", media_2="id2",
            media_6="id6", media_8="id8", media_10="id10",
            api_token="dummy"
        )

        _, call_kwargs = mock_submit.call_args
        body = mock_submit.call_args[0][1]  # positional arg: body
        media_ids = [e["mediaGenerationId"] for e in body["media"]]
        self.assertIn("id1",  media_ids)
        self.assertIn("id2",  media_ids)
        self.assertIn("id6",  media_ids)
        self.assertIn("id8",  media_ids)
        self.assertIn("id10", media_ids)
        # Skipped (empty) slots must NOT appear
        self.assertNotIn("", media_ids)

    @mock.patch('useapi_nodes._submit_with_progress')
    def test_trim_values_on_extended_slots(self, mock_submit):
        """trim_start/trim_end for slots 6-10 must reach the API payload."""
        mock_submit.return_value = {"encodedVideo": "AAAA"}

        node = UseapiVeoConcatenate()
        node.execute(
            media_1="a", media_2="b", media_7="c",
            trim_start_7=1.5, trim_end_7=2.0,
            api_token="dummy"
        )

        body = mock_submit.call_args[0][1]
        entry_c = next(e for e in body["media"] if e["mediaGenerationId"] == "c")
        self.assertAlmostEqual(entry_c["trimStart"], 1.5)
        self.assertAlmostEqual(entry_c["trimEnd"],   2.0)

    @mock.patch('useapi_nodes._submit_with_progress')
    def test_all_ten_slots_populated(self, mock_submit):
        """All 10 videos in, all 10 should appear in payload."""
        mock_submit.return_value = {"encodedVideo": "AAAA"}

        node = UseapiVeoConcatenate()
        node.execute(
            **{f"media_{i}": f"id{i}" for i in range(1, 11)},
            api_token="dummy"
        )

        body = mock_submit.call_args[0][1]
        self.assertEqual(len(body["media"]), 10)
        for i in range(1, 11):
            self.assertIn(f"id{i}", [e["mediaGenerationId"] for e in body["media"]])

if __name__ == "__main__":
    unittest.main(verbosity=2)
